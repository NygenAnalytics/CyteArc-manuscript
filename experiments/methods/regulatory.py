import gzip
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import cytearc
import numba
import numpy as np
import pandas as pd
from cytearc import regulons as rg
from cytearc.features.enrichment.chromvar import (
    background_peaks,
    build_chromvar_model,
    score_chromvar_block,
)
from cytearc.features.genomic.sequence import gc_content
from scipy import sparse
from scipy.io import mmwrite
from threadpoolctl import threadpool_limits

from experiments.runtime.common import configure_cytearc, provenance, scrub_private, write_json
from experiments.runtime.harness import R2

SOURCES = {
    "matrix": "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5",
    "motifs": "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "genome": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/GRCh38.primary_assembly.genome.fa.gz",
    "transcriptionFactors": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
    "rankings": "https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
    "annotations": "https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
    "binding": "https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/SRX26995709.05.bed",
}
CHROMVAR_SETTINGS = {
    "backgroundIterations": 50,
    "kernelWidth": 0.1,
    "bins": 50,
    "seed": 0,
    "motifPValue": 5e-5,
    "motifBackground": "subject",
    "maxAbsoluteError": 1e-8,
}


def fetch(name, work, records):
    url = SOURCES[name]
    path = work / url.rsplit("/", 1)[1]
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
        partial.rename(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    records[name] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
    return path


def prepare_store(work, workers, inputs):
    source = fetch("matrix", work, inputs)
    reader = cytearc.CrH5Reader(str(source))
    path = work / "multiome.zarr"
    try:
        cytearc.CrToZarr(reader, zarr_loc=str(path), mem_budget="24G", nthreads=workers).dump()
    finally:
        reader.h5obj.close()
    ds = cytearc.DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=workers, mem_budget="24G"
    )
    run = ds.pipeline.run(
        filtering=True,
        umap=False,
        paris=False,
        cell_cycle=False,
        doublets=False,
        markers=False,
        params={
            "ann_index": {"ann_parallel": False},
            "leiden": {"partitions": [1.0], "selected": 1.0},
        },
    )
    return ds, run


def compare_chromvar(payload):
    configure_cytearc()
    workers = int(payload.get("workers", 8))
    numba.set_num_threads(workers)
    work = Path(payload["workDirectory"])
    work.mkdir(parents=True, exist_ok=True)
    io_check = work / "reference_io"
    io_check.mkdir()
    mmwrite(
        io_check / "counts.mtx",
        sparse.coo_matrix(np.array([[0, 1, 2, 0], [1, 1, 0, 1], [2, 0, 1, 1]])),
    )
    mmwrite(io_check / "motifs.mtx", sparse.coo_matrix(np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]])))
    np.savetxt(
        io_check / "background.tsv", [[2, 2, 2], [1, 1, 3], [2, 2, 1]], fmt="%d", delimiter="\t"
    )
    script = Path(__file__).parent / "tools" / "chromvar.R"
    checked = subprocess.run(
        ["Rscript", str(script), str(io_check), "1"], capture_output=True, text=True, timeout=120
    )
    if checked.returncode:
        raise RuntimeError(
            f"chromVAR reference I/O check failed: {(checked.stdout + checked.stderr)[-4000:]}"
        )
    expected_z = np.array(
        [
            [1.1547005383792515, -1.1547005383792517, 1.1547005383792515, -1.1547005383792515],
            [-0.5773502691896257, 0.5773502691896256, 0.5773502691896256, -0.5773502691896257],
            [3.2331615074619044, -1.1547005383792512, -4.041451884327379, 9.237604307034015],
        ]
    )
    observed_z = pd.read_csv(io_check / "reference_z.csv", header=None).to_numpy()
    if observed_z.shape != expected_z.shape or not np.allclose(
        observed_z, expected_z, atol=1e-10, rtol=1e-10
    ):
        raise RuntimeError("chromVAR reference I/O changed the published four-cell fixture values")
    shutil.rmtree(io_check)
    storage = R2(payload["r2EndpointUrl"])
    prefix = payload["recordUri"].removesuffix(".json")
    inputs = {}
    ds, run = prepare_store(work, workers, inputs)
    cells = run["analysis_cell_selection"]
    cell_indices = np.flatnonzero(ds.artifacts.load(cells)["values"][:])
    peaks = ds.features.detected(cells, from_assay="ATAC", min_cells=1)
    peak_indices = np.flatnonzero(ds.artifacts.load(peaks)["values"][:])
    original_ids = ds.ATAC.feats.fetch_all("ids").astype(str)[peak_indices]
    parts = [x.replace(":", "-").split("-") for x in original_ids]
    coordinates = np.array([f"{chrom}:{start}-{end}" for chrom, start, end in parts])
    genome_gz = fetch("genome", work, inputs)
    genome_path = work / "genome.fa"
    with gzip.open(genome_gz, "rb") as source, genome_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    motifs = rg.read_jaspar_motifs(fetch("motifs", work, inputs))
    started = time.perf_counter()
    with rg.IndexedFasta(genome_path) as genome:
        gc = gc_content(genome, coordinates)
        usable = np.isfinite(gc)
        peak_indices = peak_indices[usable]
        coordinates = coordinates[usable]
        gc = gc[usable]
        matches = rg.match_motifs(
            coordinates,
            genome,
            motifs,
            p_value=CHROMVAR_SETTINGS["motifPValue"],
            background=CHROMVAR_SETTINGS["motifBackground"],
        )
    motif_seconds = time.perf_counter() - started
    motif_ids = np.array([m.motifId for m in motifs])
    rows = pd.Index(coordinates).get_indexer(matches["target"])
    columns = pd.Index(motif_ids).get_indexer(matches["source"])
    if np.any(rows < 0) or np.any(columns < 0):
        raise RuntimeError("Motif matches contain unknown features")
    annotation = sparse.csc_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(coordinates), len(motifs))
    )
    counts = np.asarray(ds.z["ATAC/counts"].oindex[cell_indices, peak_indices], dtype=np.float64)
    fragments = counts.sum(axis=0)
    background = background_peaks(
        fragments,
        gc,
        n_iterations=CHROMVAR_SETTINGS["backgroundIterations"],
        kernel_width=CHROMVAR_SETTINGS["kernelWidth"],
        bins=CHROMVAR_SETTINGS["bins"],
        seed=CHROMVAR_SETTINGS["seed"],
    )
    model = build_chromvar_model(annotation, fragments, background)
    started = time.perf_counter()
    pairs = [
        score_chromvar_block(counts[start : start + 128], model)
        for start in range(0, len(counts), 128)
    ]
    deviations = np.vstack([x[0] for x in pairs])
    scores = np.vstack([x[1] for x in pairs])
    native_seconds = time.perf_counter() - started
    np.savez_compressed(
        work / "native_scores.npz",
        deviations=deviations,
        z=scores,
        cellIds=ds.cells.fetch_all("ids")[cell_indices],
        motifIds=motif_ids,
    )
    storage.upload(work / "native_scores.npz", f"{prefix}/native_scores.npz")
    checkpoint = {
        "provenance": provenance(),
        "inputs": inputs,
        "cells": counts.shape[0],
        "peaks": counts.shape[1],
        "motifs": len(motifs),
        "motifMatches": len(matches),
        "excludedAmbiguousPeaks": int(np.sum(~usable)),
        "settings": CHROMVAR_SETTINGS,
        "motifSeconds": motif_seconds,
        "nativeSeconds": native_seconds,
    }
    storage.put_json(f"{prefix}/native.json", checkpoint)
    with threadpool_limits(limits=workers):
        mmwrite(work / "counts.mtx", sparse.csc_matrix(counts.T))
        mmwrite(work / "motifs.mtx", annotation)
    np.savetxt(work / "background.tsv", background + 1, fmt="%d", delimiter="\t")
    import tarfile

    archive = work / "chromvar_inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in ("counts.mtx", "motifs.mtx", "background.tsv", "native_scores.npz"):
            handle.add(work / name, arcname=name)
    storage.upload(archive, f"{prefix}/chromvar_inputs.tar.gz")
    with archive.open("rb") as handle:
        checkpoint["referenceInputsSha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    checkpoint["referenceInputsUri"] = f"{prefix}/chromvar_inputs.tar.gz"
    storage.put_json(f"{prefix}/native.json", checkpoint)
    script = Path(__file__).parent / "tools" / "chromvar.R"
    process = subprocess.run(
        ["Rscript", str(script), str(work), str(workers)],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    (work / "reference.log").write_text(process.stdout + process.stderr)
    storage.upload(work / "reference.log", f"{prefix}/reference.log")
    if process.returncode:
        raise RuntimeError(f"R chromVAR failed: {(process.stdout + process.stderr)[-4000:]}")
    results = {}
    for key, native in (("deviations", deviations), ("z", scores)):
        reference = pd.read_csv(work / f"reference_{key}.csv", header=None).to_numpy().T
        if reference.shape != native.shape:
            raise RuntimeError(f"chromVAR output shape differs for {key}")
        finite = np.isfinite(reference) & np.isfinite(native)
        mask_equal = bool(np.array_equal(np.isfinite(reference), np.isfinite(native)))
        error = float(np.max(np.abs(reference[finite] - native[finite])))
        results[key] = {
            "finiteMaskEqual": mask_equal,
            "finiteEntries": int(finite.sum()),
            "maxAbsoluteError": error,
            "meetsTolerance": mask_equal and error <= CHROMVAR_SETTINGS["maxAbsoluteError"],
        }
        storage.upload(work / f"reference_{key}.csv", f"{prefix}/reference_{key}.csv")
    result = {
        "status": "ok",
        "experiment": "chromvar",
        **checkpoint,
        "results": results,
        "reference": json.loads((work / "reference.json").read_text()),
        "comparisonKind": "Deviations and z-scores using identical motif matches and background peak samples",
        "performanceClaim": False,
    }
    write_json("chromvar.json", scrub_private(result), directory=Path(payload["outputDirectory"]))
    return result


def compare_grn(payload):
    from concurrent.futures import ThreadPoolExecutor
    from importlib.metadata import version

    from arboreto.core import SGBM_KWARGS, infer_partial_network
    from cytearc.regulons.grn import GrnParameters, infer_grn

    configure_cytearc()
    workers = int(payload.get("workers", 8))
    work = Path(payload["workDirectory"])
    work.mkdir(parents=True, exist_ok=True)
    storage = R2(payload["r2EndpointUrl"])
    prefix = payload["recordUri"].removesuffix(".json")
    inputs = {}
    ds, run = prepare_store(work, workers, inputs)
    selection = run["analysis_cell_selection"]
    cells = np.flatnonzero(ds.artifacts.load(selection)["values"][:])
    detected = ds.features.detected(selection, from_assay="RNA", min_cells=20)
    gene_mask = np.asarray(ds.artifacts.load(detected)["values"][:], dtype=bool)
    names = ds.RNA.feats.fetch_all("names").astype(str)
    duplicate = pd.Index(names).duplicated(keep=False)
    gene_mask &= ~duplicate
    genes = np.flatnonzero(gene_mask)
    expression = np.asarray(ds.z["RNA/counts"].oindex[cells, genes], dtype=np.float32)
    variable = np.ptp(expression, axis=0) > 0
    expression = expression[:, variable]
    names = names[genes][variable]
    tfs = fetch("transcriptionFactors", work, inputs).read_text().split()
    tf_mask = np.isin(names, tfs)
    tf_names = names[tf_mask].tolist()
    if len(tf_names) < 100:
        raise RuntimeError(
            "Too few expressed human transcription factors for the fixed GRN comparison"
        )
    frame = pd.DataFrame(expression, columns=names, index=ds.cells.fetch_all("ids")[cells])
    frame.to_pickle(work / "expression.pkl")
    storage.upload(work / "expression.pkl", f"{prefix}/expression.pkl")
    settings = {
        "minimumDetectedCells": 20,
        "uniqueGeneNames": True,
        "variableGenesOnly": True,
        "seeds": [777, 778, 779],
        "maxMatchedImportanceError": 1e-10,
        "referenceEarlyStopping": "consecutive",
        "nativeDefaultEarlyStopping": "stage",
    }
    results = {
        "status": "running",
        "experiment": "grnboost2",
        "provenance": provenance(),
        "inputs": inputs,
        "settings": settings,
        "cells": expression.shape[0],
        "genes": expression.shape[1],
        "transcriptionFactors": len(tf_names),
        "ambiguousGeneColumnsExcluded": int(duplicate.sum()),
        "nativeRuns": [],
        "referenceVersions": {
            key: version(key) for key in ("pyscenic", "arboreto", "scikit-learn")
        },
    }
    storage.put_json(f"{prefix}/completed.json", results)
    native_first = None
    for seed in settings["seeds"]:
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            links = infer_grn(
                expression,
                names,
                tf_names,
                seed=seed,
                nthreads=workers,
                parameters=GrnParameters(earlyStopping="stage"),
            )
        seconds = time.perf_counter() - started
        path = work / f"grn_seed_{seed}.csv.gz"
        links.to_csv(path, index=False, compression="gzip")
        storage.upload(path, f"{prefix}/{path.name}")
        if set(links["target"]) != set(names):
            raise RuntimeError("Native inference omitted a nonconstant eligible target")
        results["nativeRuns"].append(
            {
                "seed": seed,
                "links": len(links),
                "seconds": seconds,
                "earlyStopping": "stage",
                "uri": f"{prefix}/{path.name}",
            }
        )
        storage.put_json(f"{prefix}/completed.json", results)
        if seed == 777:
            native_first = links
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        matched = infer_grn(
            expression,
            names,
            tf_names,
            seed=777,
            nthreads=workers,
            parameters=GrnParameters(earlyStopping="consecutive"),
        )
    results["matchedNativeSeconds"] = time.perf_counter() - started
    tf_matrix = expression[:, tf_mask]

    def reference_target(position):
        target = names[position]
        result = infer_partial_network(
            "GBM", SGBM_KWARGS, tf_matrix, tf_names, target, expression[:, position], seed=777
        )
        if result.empty or set(result["target"]) != {target}:
            raise RuntimeError(f"Arboreto failed to return the variable target {target}")
        return result

    started = time.perf_counter()
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as executor:
        reference = pd.concat(executor.map(reference_target, range(len(names))), ignore_index=True)
    results["referenceSeconds"] = time.perf_counter() - started
    reference.to_csv(work / "arboreto.csv.gz", index=False, compression="gzip")
    matched.to_csv(work / "matched_grn.csv.gz", index=False, compression="gzip")
    for path in (work / "arboreto.csv.gz", work / "matched_grn.csv.gz"):
        storage.upload(path, f"{prefix}/{path.name}")

    def agreement(native):
        a = native.set_index(["TF", "target"])["importance"]
        b = reference.set_index(["TF", "target"])["importance"]
        aligned = a.to_frame("native").join(b.rename("reference"), how="outer").fillna(0)
        result = {
            "linksInUnion": len(aligned),
            "linksOnlyNative": len(a.index.difference(b.index)),
            "linksOnlyReference": len(b.index.difference(a.index)),
            "pearson": float(aligned.corr().iloc[0, 1]),
            "maxImportanceAbsoluteDifference": float(
                np.max(np.abs(aligned.native - aligned.reference))
            ),
        }
        result["topLinkJaccard"] = {}
        for size in (1000, 10000, 100000):
            left = native.sort_values(
                ["importance", "target", "TF"], ascending=[False, True, True]
            ).head(size)
            right = reference.sort_values(
                ["importance", "target", "TF"], ascending=[False, True, True]
            ).head(size)
            x = set(zip(left.TF, left.target))
            y = set(zip(right.TF, right.target))
            result["topLinkJaccard"][str(size)] = len(x & y) / len(x | y)
        return result

    results.update(
        {
            "status": "ok",
            "matchedStopping": agreement(matched),
            "defaultStopping": agreement(native_first),
            "comparisonKind": "Actual arboreto GRNBoost2 used by pySCENIC, with matched modern scikit-learn stopping and the shipped CyteArc stopping rule reported separately",
            "performanceClaim": False,
        }
    )
    results["matchedStopping"]["meetsTolerance"] = (
        results["matchedStopping"]["maxImportanceAbsoluteDifference"]
        <= settings["maxMatchedImportanceError"]
    )
    write_json("grnboost2.json", scrub_private(results), directory=Path(payload["outputDirectory"]))
    return results


REGULATOR_PANEL = [
    "SPI1",
    "CEBPA",
    "CEBPB",
    "IRF8",
    "KLF4",
    "PAX5",
    "EBF1",
    "POU2F2",
    "TBX21",
    "EOMES",
    "RUNX3",
    "GATA3",
    "TCF7",
    "LEF1",
    "CTCF",
    "IRF4",
    "MEF2C",
    "NFKB1",
]


def binding_enrichment(targets_by_seed, universe, work, inputs):
    from collections import Counter

    from cytearc.features.genomic.annotation import read_gene_annotation, transcription_start_sites
    from scipy.stats import fisher_exact

    binding = fetch("binding", work, inputs)
    metadata_url = "https://chip-atlas.org/view?id=SRX26995709"
    request = urllib.request.Request(metadata_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        metadata = response.read()
    text = metadata.decode()
    if 'data-genome="hg38"' not in text or SOURCES["binding"] not in text:
        raise ValueError("ChIP-Atlas metadata does not verify this experiment's hg38 peak file")
    (work / "binding_metadata.html").write_bytes(metadata)
    inputs["bindingMetadata"] = {
        "url": metadata_url,
        "sha256": hashlib.sha256(metadata).hexdigest(),
        "resourceAssembly": "hg38",
        "originalAuthorAssembly": "hg19",
        "assemblyEvidence": "Experiment-specific hg38 processing section and matching hg38 BED link",
        "factorIdentitySource": "GEO sample GSM8669389: MONO, PU.1 (CUT&RUN)",
        "resourceAntigenClassification": "Unclassified",
    }
    annotation_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.basic.annotation.gtf.gz"
    annotation = work / "genes.gtf.gz"
    request = urllib.request.Request(annotation_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as source, annotation.open("wb") as target:
        shutil.copyfileobj(source, target)
    with annotation.open("rb") as handle:
        inputs["geneAnnotation"] = {
            "url": annotation_url,
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }
    genes = read_gene_annotation(annotation)
    genes = genes.loc[genes.gene_name.isin(universe)].reset_index(drop=True)
    if genes.empty:
        raise ValueError("The GRN gene universe has no GENCODE annotations")
    tss = transcription_start_sites(genes)
    genes["window_start"] = np.maximum(tss - 10000, 0)
    genes["window_end"] = tss + 10001
    peaks = rg.read_peak_file(binding)
    if not set(peaks.chrom).intersection(genes.chrom):
        raise ValueError("Binding peaks and gene annotations have no shared chromosomes")
    bound = set()
    for chromosome, frame in genes.groupby("chrom"):
        intervals = peaks.loc[peaks.chrom == chromosome].sort_values("start")
        if intervals.empty:
            continue
        starts = intervals.start.to_numpy()
        ends = np.maximum.accumulate(intervals.end.to_numpy())
        positions = np.searchsorted(starts, frame.window_end.to_numpy(), side="left") - 1
        valid = positions >= 0
        hit = np.zeros(len(frame), dtype=bool)
        hit[valid] = ends[positions[valid]] > frame.window_start.to_numpy()[valid]
        bound.update(frame.loc[hit, "gene_name"])
    eligible = set(genes.gene_name)
    counts = Counter(
        gene
        for seed_targets in targets_by_seed.values()
        for gene in seed_targets.get("SPI1", set())
    )
    consensus = {gene for gene, count in counts.items() if count >= 2} & eligible
    observed = len(consensus & bound)
    outside = eligible - consensus
    table = [[observed, len(consensus) - observed], [len(outside & bound), len(outside - bound)]]
    result = {
        "factor": "SPI1",
        "experiment": "SRX26995709",
        "sample": "GSM8669389",
        "source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8669389",
        "sampleDescription": "Monocyte PU.1 CUT&RUN",
        "scope": "One prespecified SPI1 binding-support test",
        "coordinateAssembly": "ChIP-Atlas hg38 reprocessing matched to GENCODE v32",
        "targetRule": "Present in at least two of three native regulons",
        "geneWindow": "GENCODE v32 gene-level transcription start site plus or minus 10000 bases",
        "universeGenes": len(eligible),
        "boundGenes": len(bound),
        "consensusTargets": len(consensus),
        "boundTargets": observed,
        "contingency": table,
    }
    if not consensus:
        return {
            **result,
            "status": "no_consensus_targets",
            "tests": 0,
            "pValue": None,
            "oddsRatio": None,
            "foldEnrichment": None,
        }
    test = fisher_exact(table, alternative="greater")
    result.update(
        status="evaluated",
        tests=1,
        pValue=float(test.pvalue),
        oddsRatio=float(test.statistic) if np.isfinite(test.statistic) else None,
        foldEnrichment=observed / len(consensus) / (len(bound) / len(eligible)) if bound else None,
    )
    return result


def compare_pruning(payload):
    import os
    import sys
    from importlib.metadata import version
    from itertools import combinations

    from ctxcore.genesig import GeneSignature
    from ctxcore.recovery import recovery
    from ctxcore.rnkdb import FeatherRankingDatabase
    from cytearc.regulons.cistarget import recovery_aucs, recovery_curves
    from cytearc.regulons.databases import RankingDatabase, read_motif_annotations
    from cytearc.regulons.modules import modules_from_adjacencies
    from cytearc.regulons.pruning import prune_modules, regulons_from_motifs

    configure_cytearc()
    workers = int(payload.get("workers", 8))
    work = Path(payload["workDirectory"])
    work.mkdir(parents=True, exist_ok=True)
    storage = R2(payload["r2EndpointUrl"])
    prefix = payload["recordUri"].removesuffix(".json")
    grn_prefix = payload["grnPrefix"].rstrip("/")
    grn = storage.get_json(f"{grn_prefix}/completed.json")
    if sorted(x["seed"] for x in grn.get("nativeRuns", [])) != [777, 778, 779]:
        raise RuntimeError("All three fixed native GRN seeds must finish before pruning")
    inputs = {}
    rankings = fetch("rankings", work, inputs)
    annotation_file = fetch("annotations", work, inputs)
    native_db = RankingDatabase(rankings, name="hg38")
    reference_db = FeatherRankingDatabase(str(rankings), "hg38")
    annotations = read_motif_annotations(annotation_file)
    settings = {
        "regulators": REGULATOR_PANEL,
        "seeds": [777, 778, 779],
        "rankThreshold": 5000,
        "aucThreshold": 0.05,
        "nesThreshold": 3.0,
        "pruningWeightedRecovery": False,
        "coreWeightedRecovery": True,
        "aucAbsoluteTolerance": 1e-15,
        "curveAbsoluteTolerance": 1e-8,
        "referenceBackend": "custom_multiprocessing",
        "referenceImportAdapter": "Temporary np.object=object for pySCENIC metadata dtype declaration; removed immediately after import",
    }
    native_targets = {}
    reference_targets = {}
    core_cases = []
    universe = set()
    for seed in settings["seeds"]:
        folder = work / f"seed_{seed}"
        folder.mkdir()
        path = folder / "adjacencies.csv.gz"
        storage.download(f"{grn_prefix}/grn_seed_{seed}.csv.gz", path)
        links = pd.read_csv(path, float_precision="round_trip", keep_default_na=False)
        universe.update(links.target)
        modules = modules_from_adjacencies(links)
        modules = modules.loc[modules.TF.isin(REGULATOR_PANEL)].copy()
        if modules.empty:
            raise RuntimeError(f"No modules for the prespecified regulators at seed {seed}")
        modules.to_pickle(folder / "modules.pkl")
        modules.to_csv(folder / "modules.csv.gz", index=False, compression="gzip")
        storage.upload(folder / "modules.csv.gz", f"{prefix}/seed_{seed}/modules.csv.gz")
        script = Path(__file__).parent / "tools" / "prune_regulons.py"
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(folder),
                    str(rankings),
                    str(annotation_file),
                    str(workers),
                ],
                env={**os.environ, "NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                capture_output=True,
                text=True,
                timeout=7200,
            )
        except subprocess.TimeoutExpired as exc:
            output = b"".join(
                part.encode() if isinstance(part, str) else part or b""
                for part in (exc.stdout, exc.stderr)
            )
            (folder / "reference.log").write_bytes(output)
            storage.upload(folder / "reference.log", f"{prefix}/seed_{seed}/reference.log")
            raise
        (folder / "reference.log").write_text(process.stdout + process.stderr)
        storage.upload(folder / "reference.log", f"{prefix}/seed_{seed}/reference.log")
        reference_log = process.stdout + process.stderr
        if (
            process.returncode
            or "MemoryError" in reference_log
            or "ran out of memory" in reference_log
        ):
            raise RuntimeError(f"pySCENIC pruning failed: {reference_log[-4000:]}")
        reference = pd.read_pickle(folder / "reference_motifs.pkl")
        reference_targets[seed] = {
            tf: {gene for targets in group[("Enrichment", "TargetGenes")] for gene, _ in targets}
            for tf, group in reference.groupby(level=0)
        }
        native = prune_modules(modules, [native_db], annotations, nthreads=workers)
        regulons = regulons_from_motifs(native)
        native_targets[seed] = {tf: set(group.target) for tf, group in regulons.groupby("TF")}
        native.to_pickle(folder / "native_motifs.pkl")
        native.to_csv(folder / "native_motifs.csv.gz", index=False, compression="gzip")
        regulons.to_csv(folder / "regulons.csv.gz", index=False, compression="gzip")
        for name in (
            "reference_motifs.pkl",
            "reference_motifs.csv.gz",
            "native_motifs.pkl",
            "native_motifs.csv.gz",
            "regulons.csv.gz",
        ):
            storage.upload(folder / name, f"{prefix}/seed_{seed}/{name}")
        for tf in REGULATOR_PANEL:
            candidates = modules.loc[(modules.TF == tf) & (modules.context == "top50")]
            if candidates.empty:
                core_cases.append({"seed": seed, "TF": tf, "eligible": False})
                continue
            source = sorted(candidates.source.unique())[0]
            selected = candidates.loc[candidates.source == source]
            weights_by_gene = dict(zip(selected.target, selected.weight))
            found, ranks = native_db.ranks(selected.target)
            if not len(found):
                raise RuntimeError("A selected module has no ranked genes")
            reference_ranks = reference_db.load(GeneSignature(source, weights_by_gene)).loc[
                native_db.rowNames, found
            ]
            weights = np.array([weights_by_gene[gene] for gene in found])
            reference_curves, reference_aucs = recovery(
                reference_ranks, native_db.total, weights, 5000, 0.05
            )
            aucs = recovery_aucs(ranks, native_db.total, weights, 0.05)
            curves = recovery_curves(ranks, weights, 5000)
            auc_error = float(np.max(np.abs(aucs - reference_aucs)))
            curve_error = float(np.max(np.abs(curves - reference_curves)))
            same_ranks = bool(np.array_equal(ranks, reference_ranks.to_numpy()))
            core_cases.append(
                {
                    "seed": seed,
                    "TF": tf,
                    "eligible": True,
                    "genes": len(found),
                    "motifs": len(native_db.rowNames),
                    "ranksIdentical": same_ranks,
                    "aucMaxAbsoluteError": auc_error,
                    "curveMaxAbsoluteError": curve_error,
                    "meetsTolerance": same_ranks and auc_error <= 1e-15 and curve_error <= 1e-8,
                }
            )
        storage.put_json(
            f"{prefix}/completed.json",
            {
                "settings": settings,
                "coreComparisons": core_cases,
                "nativeTargets": {
                    str(s): {tf: sorted(g) for tf, g in t.items()}
                    for s, t in native_targets.items()
                },
                "referenceTargets": {
                    str(s): {tf: sorted(g) for tf, g in t.items()}
                    for s, t in reference_targets.items()
                },
            },
        )
    stability = []
    for tf in REGULATOR_PANEL:
        for first, second in combinations(settings["seeds"], 2):
            row = {"TF": tf, "firstSeed": first, "secondSeed": second}
            for method, targets in (("native", native_targets), ("reference", reference_targets)):
                left = targets[first].get(tf, set())
                right = targets[second].get(tf, set())
                union = left | right
                row[method] = {
                    "firstTargets": len(left),
                    "secondTargets": len(right),
                    "intersection": len(left & right),
                    "union": len(union),
                    "jaccard": len(left & right) / len(union) if union else None,
                }
            stability.append(row)
    summary = {}
    for method in ("native", "reference"):
        defined = [
            row[method]["jaccard"] for row in stability if row[method]["jaccard"] is not None
        ]
        summary[method] = {
            "definedRegulatorPairs": len(defined),
            "withinMethodMedianJaccard": float(np.median(defined)) if defined else None,
            "bothSeedsEmpty": sum(row[method]["union"] == 0 for row in stability),
            "oneSeedEmpty": sum(
                (row[method]["firstTargets"] == 0) != (row[method]["secondTargets"] == 0)
                for row in stability
            ),
            "totalIntersection": sum(row[method]["intersection"] for row in stability),
            "totalUnion": sum(row[method]["union"] for row in stability),
        }
    paired = [
        row
        for row in stability
        if row["native"]["jaccard"] is not None and row["reference"]["jaccard"] is not None
    ]
    summary["paired"] = {
        "regulatorSeedPairs": len(paired),
        "allPrespecifiedPairs": len(stability),
        "nativeMedianJaccard": float(np.median([row["native"]["jaccard"] for row in paired]))
        if paired
        else None,
        "referenceMedianJaccard": float(np.median([row["reference"]["jaccard"] for row in paired]))
        if paired
        else None,
        "medianPairedDifference": float(
            np.median([row["native"]["jaccard"] - row["reference"]["jaccard"] for row in paired])
        )
        if paired
        else None,
    }
    result = {
        "status": "ok",
        "experiment": "scenic_pruning",
        "provenance": provenance(),
        "referenceVersions": {name: version(name) for name in ("pyscenic", "ctxcore", "polars")},
        "settings": settings,
        "inputs": inputs,
        "coreComparisons": core_cases,
        "coreMeetsTolerance": any(case["eligible"] for case in core_cases)
        and all(case["meetsTolerance"] for case in core_cases if case["eligible"]),
        "nativeTargets": {
            str(seed): {tf: sorted(genes) for tf, genes in targets.items()}
            for seed, targets in native_targets.items()
        },
        "referenceTargets": {
            str(seed): {tf: sorted(genes) for tf, genes in targets.items()}
            for seed, targets in reference_targets.items()
        },
        "stability": stability,
        "summary": summary,
        "comparisonKind": "Identical co-expression modules with unit recovery weights for full pruning; native per-motif leading edges compared with stock pySCENIC 0.12.1, preserving all regulators and seed pairs. Weighted recovery kernels are validated separately.",
    }
    storage.put_json(f"{prefix}/pruning.json", result)
    result["bindingEnrichment"] = binding_enrichment(native_targets, universe, work, inputs)
    storage.upload(work / SOURCES["binding"].rsplit("/", 1)[1], f"{prefix}/binding_peaks.bed")
    storage.upload(work / "binding_metadata.html", f"{prefix}/binding_metadata.html")
    write_json(
        "scenic_pruning.json", scrub_private(result), directory=Path(payload["outputDirectory"])
    )
    return result
