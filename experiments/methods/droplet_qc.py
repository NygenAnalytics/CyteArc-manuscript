import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from experiments.runtime.common import configure_cytearc, provenance, scrub_private, write_json
from experiments.runtime.harness import R2

PBMC = {"pbmc_1k_v2": "3.0.0", "pbmc_10k_v3": "3.0.0"}
SETTINGS = {
    "lower": 100,
    "retain": 10000,
    "iterations": 10000,
    "seed": 0,
    "fdr": 0.001,
    "cellCallJaccardTarget": 0.98,
    "ambientMaxAbsoluteError": 1e-10,
    "rhoMaxAbsoluteError": 1e-6,
    "correctedCountMaxAbsoluteError": 1.000001,
    "soupMaximumContamination": 1.0,
    "soupFdr": 0.05,
}
GENE_SETS = {
    "hemoglobin": ["HBB", "HBA1", "HBA2"],
    "immunoglobulin": ["IGKC", "IGHM", "IGHG1", "IGHA1", "IGLC2"],
}


def download(url, destination):
    if not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with (
            urllib.request.urlopen(request, timeout=300) as source,
            destination.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
    with destination.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_counts(path):
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        genes = group["features/id"].asstr()[:]
        symbols = group["features/name"].asstr()[:]
        barcodes = group["barcodes"].asstr()[:]
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=tuple(group["shape"][:]),
        ).T.tocsr()
    return matrix, barcodes, genes, symbols


def run_reference(work, check, workers):
    script = Path(__file__).parent / "tools" / "droplet_qc.R"
    started = time.perf_counter()
    process = subprocess.run(
        ["Rscript", str(script), str(work), check, str(workers)],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    (work / "reference.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"R {check} failed: {(process.stdout + process.stderr)[-4000:]}")
    return {
        "processSeconds": time.perf_counter() - started,
        **json.loads((work / "reference.json").read_text()),
    }


def compare_empty_drops(raw, barcodes, genes, work, workers):
    from cytearc.quality_control import empty_drops

    reference = run_reference(work, "emptydrops", workers)
    started = time.perf_counter()
    native = empty_drops(
        raw,
        lower=SETTINGS["lower"],
        retain=SETTINGS["retain"],
        n_iter=SETTINGS["iterations"],
        alpha=float(reference["alpha"]),
        random_seed=SETTINGS["seed"],
        nthreads=workers,
        barcodes=barcodes,
    )
    seconds = time.perf_counter() - started
    frame = native.to_pandas()
    r = pd.read_csv(work / "reference_calls.csv", index_col="barcode")
    joined = frame.join(r, how="inner", validate="one_to_one")
    if len(joined) != len(frame) or len(joined) != len(r):
        raise RuntimeError("EmptyDrops did not compare identical tested barcodes")
    a = joined["fdr"].to_numpy() <= SETTINGS["fdr"]
    b = joined["FDR"].to_numpy() <= SETTINGS["fdr"]
    union = int(np.sum(a | b))
    if union == 0:
        raise RuntimeError("Neither EmptyDrops implementation called any cells")
    ambient = pd.read_csv(work / "reference_ambient.csv").set_index("gene")["ambient"]
    if set(ambient.index) - set(genes):
        raise RuntimeError("Reference ambient profile contains unknown genes")
    aligned = ambient.reindex(genes, fill_value=0).to_numpy()
    ambient_error = float(np.max(np.abs(native.ambient - aligned)))
    comparable = (joined["total"] < SETTINGS["retain"]) & np.isfinite(joined["log_probability"])
    joined.to_csv(work / "emptydrops_calls.csv")
    jaccard = float(np.sum(a & b) / union)
    return {
        "reference": reference,
        "nativeSeconds": seconds,
        "nativeAlpha": native.alpha,
        "nativeAmbientDroplets": native.n_ambient,
        "testedBarcodes": len(joined),
        "nativeCells": int(a.sum()),
        "referenceCells": int(b.sum()),
        "cellCallAgreement": float(np.mean(a == b)),
        "cellCallJaccard": jaccard,
        "ambientMaxAbsoluteError": ambient_error,
        "logProbabilityMaxAbsoluteDifference": float(
            np.max(
                np.abs(
                    joined.loc[comparable, "log_probability"] - joined.loc[comparable, "LogProb"]
                )
            )
        ),
        "pValueMeanAbsoluteDifference": float(
            np.mean(np.abs(joined.loc[comparable, "p_value"] - joined.loc[comparable, "PValue"]))
        ),
        "meetsTargets": jaccard >= SETTINGS["cellCallJaccardTarget"]
        and ambient_error <= SETTINGS["ambientMaxAbsoluteError"],
        "comparisonKind": "Implementation agreement with independent Monte Carlo streams and shared model precision fitted by DropletUtils to the ambient droplets",
    }


def compare_ambient(raw, cells, cell_barcodes, genes, symbols, work, workers):
    from cytearc.quality_control.ambient import (
        estimate_contamination,
        estimation_cells,
        expressing_cells,
        gene_set_evidence,
        remove_ambient_counts,
        soup_fractions,
    )

    sets = {name: np.flatnonzero(np.isin(symbols, names)) for name, names in GENE_SETS.items()}
    if any(len(indices) == 0 for indices in sets.values()):
        raise RuntimeError("Prespecified ambient marker gene set is absent")
    (work / "gene_sets.json").write_text(
        json.dumps({k: genes[v].tolist() for k, v in sets.items()})
    )
    reference = run_reference(work, "soupx", workers)
    raw_total = np.asarray(raw.sum(axis=1)).ravel()
    empty = (raw_total > 0) & (raw_total <= SETTINGS["lower"])
    ambient_counts = np.asarray(raw[empty].sum(axis=0)).ravel()
    profile = soup_fractions(ambient_counts)
    reference_profile = pd.read_csv(work / "reference_soup.csv").set_index("gene").reindex(genes)
    r_profile = reference_profile["fraction"].to_numpy()
    same_ambient_counts = bool(
        np.array_equal(ambient_counts, reference_profile["count"].to_numpy())
    )
    if not np.isfinite(r_profile).all():
        raise RuntimeError("SoupX profile does not align to every source gene")
    observed = np.column_stack(
        [np.asarray(cells[:, indices].sum(axis=1)).ravel() for indices in sets.values()]
    )
    totals = np.asarray(cells.sum(axis=1)).ravel()
    expected, p_values = gene_set_evidence(
        observed,
        totals,
        [profile[indices].sum() for indices in sets.values()],
        max_contamination=SETTINGS["soupMaximumContamination"],
    )
    used = estimation_cells(expressing_cells(p_values, fdr=SETTINGS["soupFdr"]))
    fit = estimate_contamination(observed, expected, used, set_names=list(sets))
    used_frame = pd.read_csv(
        work / "reference_used.csv", index_col=0, true_values=["TRUE"], false_values=["FALSE"]
    )
    if not np.array_equal(used_frame.index.to_numpy(), cell_barcodes):
        raise RuntimeError("SoupX changed the cell order")
    r_used = used_frame.loc[:, list(sets)].to_numpy(dtype=bool)
    if r_used.shape != used.shape:
        raise RuntimeError("SoupX estimation matrix shape changed")
    shared_fit = estimate_contamination(observed, expected, r_used, set_names=list(sets))
    r_counts = sparse.csr_matrix(mmread(work / "reference_corrected.mtx").T)
    errors = []
    removed_total_errors = []
    for start in range(0, cells.shape[0], 256):
        stop = min(start + 256, cells.shape[0])
        counts = cells[start:stop].toarray().astype(np.uint32)
        corrected = remove_ambient_counts(
            counts,
            np.full(stop - start, reference["rho"]),
            profile,
            random_seed=SETTINGS["seed"],
            row_keys=np.arange(start, stop),
        )
        expected_counts = r_counts[start:stop].toarray()
        errors.append(float(np.max(np.abs(corrected - expected_counts))))
        removed_total_errors.append(
            float(np.max(np.abs(corrected.sum(1) - expected_counts.sum(1))))
        )
    profile_error = float(np.max(np.abs(profile - r_profile)))
    rho_error = abs(shared_fit.fraction - reference["rho"])
    return {
        "reference": reference,
        "ambientDroplets": int(empty.sum()),
        "ambientCountsIdentical": same_ambient_counts,
        "ambientDropletsIdentical": int(empty.sum()) == reference["ambientDroplets"],
        "filteredCells": cells.shape[0],
        "ambientProfileMaxAbsoluteError": profile_error,
        "nonExpressingCellMatrixAgreement": float(np.mean(used == r_used)),
        "nativeRho": fit.fraction,
        "nativeRhoOnReferenceCells": shared_fit.fraction,
        "rhoMaxAbsoluteError": rho_error,
        "correctedCountMaxAbsoluteError": max(errors),
        "correctedCellTotalMaxAbsoluteError": max(removed_total_errors),
        "meetsTargets": same_ambient_counts
        and int(empty.sum()) == reference["ambientDroplets"]
        and profile_error <= SETTINGS["ambientMaxAbsoluteError"]
        and rho_error <= SETTINGS["rhoMaxAbsoluteError"]
        and max(errors) <= SETTINGS["correctedCountMaxAbsoluteError"],
        "comparisonKind": "Ambient profile and global Poisson fit; native integer correction compared with SoupX continuous subtraction on matching rho, without cluster expansion",
    }


def run(payload):
    configure_cytearc()
    work = Path(payload["workDirectory"])
    work.mkdir(parents=True, exist_ok=True)
    dataset = payload["dataset"]
    release = PBMC[dataset]
    workers = int(payload.get("workers", 8))
    inputs = {}
    for kind in ("raw", "filtered"):
        url = f"https://cf.10xgenomics.com/samples/cell-exp/{release}/{dataset}/{dataset}_{kind}_feature_bc_matrix.h5"
        inputs[kind] = {"url": url, "sha256": download(url, work / f"{kind}.h5")}
    raw, barcodes, genes, symbols = read_counts(work / "raw.h5")
    cells, cell_barcodes, cell_genes, _ = read_counts(work / "filtered.h5")
    if not np.array_equal(genes, cell_genes):
        raise RuntimeError("Raw and filtered feature order differs")
    positions = pd.Index(barcodes).get_indexer(cell_barcodes)
    if np.any(positions < 0) or (raw[positions] != cells).nnz:
        raise RuntimeError("Filtered counts differ from the corresponding raw barcodes")
    (work / "settings.json").write_text(json.dumps(SETTINGS))
    (work / "gene_ids.txt").write_text("\n".join(genes) + "\n")
    result = {
        "status": "ok",
        "experiment": "droplet_qc",
        "provenance": provenance(),
        "dataset": dataset,
        "settings": SETTINGS,
        "inputs": inputs,
        "rawBarcodes": raw.shape[0],
        "genes": raw.shape[1],
    }
    storage = R2(payload["r2EndpointUrl"])
    prefix = payload["recordUri"].removesuffix(".json")
    operations = {
        "emptydrops": lambda: compare_empty_drops(raw, barcodes, genes, work, workers),
        "soupx": lambda: compare_ambient(raw, cells, cell_barcodes, genes, symbols, work, workers),
    }
    for check in payload.get("checks", ["emptydrops", "soupx"]):
        result[check] = operations[check]()
        storage.put_json(f"{prefix}/{check}.json", result[check])
    for path in work.iterdir():
        if path.suffix in (".csv", ".log", ".json"):
            storage.upload(path, f"{prefix}/{path.name}")
    write_json("droplet_qc.json", scrub_private(result), directory=Path(payload["outputDirectory"]))
    return result


def run_hto(payload):
    import gzip

    import cytearc
    from scipy.sparse import csr_matrix
    from sklearn.metrics import cohen_kappa_score
    from threadpoolctl import threadpool_limits

    configure_cytearc()
    work = Path(payload["workDirectory"])
    work.mkdir(parents=True, exist_ok=True)
    dataset = payload["dataset"]
    inputs = {}
    if dataset == "stoeckius":
        sources = {
            "hto": (
                "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2895nnn/GSM2895283/suppl/GSM2895283_Hashtag-HTO-count.csv.gz",
                "68ab6cd853964026a79f5c3da58a901808eab18c25684b3eecd41ebb71b9bce2",
            ),
            "rna": (
                "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2895nnn/GSM2895282/suppl/GSM2895282_Hashtag-RNA.umi.txt.gz",
                "98a2be289ce79c2a2104eeaff3e39528ea25dfa0b3f3c1faf7246843f65ce3d9",
            ),
        }
        for name, (url, expected) in sources.items():
            digest = download(url, work / f"{name}.gz")
            if digest != expected:
                raise RuntimeError(f"Published {name} input changed: {digest}")
            inputs[name] = {"url": url, "sha256": digest}
        hto = (
            pd.read_csv(work / "hto.gz", index_col=0)
            .drop(index=["bad_struct", "no_match", "total_reads"])
            .T
        )
        hto.columns = hto.columns.str.split("-").str[0]
        if hto.shape[1] != 8:
            raise RuntimeError("Expected eight Stoeckius hashtags")
        with gzip.open(work / "rna.gz", "rt") as handle:
            barcodes = next(handle).rstrip().split("\t")[1:]
            totals = np.zeros(len(barcodes), dtype=np.int64)
            for line in handle:
                gene, values = line.split("\t", 1)
                counts = np.fromstring(values, sep="\t", dtype=np.int64)
                if len(counts) != len(totals):
                    raise RuntimeError(f"RNA count width changed at {gene}")
                totals += counts
        totals = pd.Series(totals, index=barcodes)
        eligible = totals.index.intersection(hto.index)
        selected = eligible[totals.loc[eligible] >= 200]
        hto = hto.loc[selected]
        selection = (
            "All shared barcodes with at least 200 RNA UMIs; sequencing-summary rows excluded"
        )
    elif dataset == "gse245108":
        url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE245nnn/GSE245108/suppl/GSE245108_TNC-1-2-1_filtered_feature_bc_matrix.h5"
        path = work / "counts.h5"
        digest = download(url, path)
        if digest != "22d64482c0fb04a7685e641d762efdd95a6498156ac22e7b0264925b0907b589":
            raise RuntimeError(f"Published GSE245108 input changed: {digest}")
        inputs["counts"] = {"url": url, "sha256": digest}
        counts, barcodes, _, names = read_counts(path)
        expected = [
            f"TNC_{rep}_{amount}x" for rep in (1, 2) for amount in ("0_25", "0_5", "1", "2", "4")
        ]
        matches = [np.flatnonzero(names == name) for name in expected]
        if any(len(match) != 1 for match in matches):
            raise RuntimeError("Prespecified ten GSE245108 hashtags are missing")
        positions = np.array([match[0] for match in matches])
        hto = pd.DataFrame(counts[:, positions].toarray(), index=barcodes, columns=expected)
        selection = (
            "Every barcode in the published filtered matrix and all ten prespecified hashtags"
        )
    else:
        raise ValueError(f"Unknown HTO dataset: {dataset}")
    hto.index.name = "barcode"
    hto.to_csv(work / "hto_counts.csv")
    store = work / "hto.zarr"
    cytearc.SparseToZarr(
        csr_matrix(hto.to_numpy()),
        str(store),
        cell_ids=hto.index.to_numpy(),
        feature_ids=hto.columns.to_numpy(),
        assay_name="HTO",
        assay_type="HTO",
        nthreads=4,
        mem_budget="1G",
    ).dump()
    ds = cytearc.DataStore(
        str(store), default_assay="HTO", min_features_per_cell=-1, nthreads=4, mem_budget="1G"
    )
    with threadpool_limits(limits=4):
        ref = ds.qc.hto_demultiplexing(
            ds.snapshot_cell_selection("I"), random_seed=0, positive_quantile=0.99
        )
    group = ds.artifacts.load(ref)
    native = pd.DataFrame(
        {key: np.asarray(group[key][:]) for key in ("values", "max_id", "second_id", "margin")},
        index=ds.cells.fetch_all("ids"),
    )
    native["classification"] = np.where(
        np.isin(native["values"], ["Doublet", "Negative"]), native["values"], "Singlet"
    )
    script = Path(__file__).parent / "tools" / "hto_demultiplexing.R"
    process = subprocess.run(
        ["Rscript", str(script), str(work)], capture_output=True, text=True, timeout=3600
    )
    (work / "reference.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"R HTODemux failed: {(process.stdout + process.stderr)[-4000:]}")
    reference = pd.read_csv(work / "reference_hto.csv", index_col="barcode")
    if set(reference.index) != set(native.index):
        raise RuntimeError("HTO comparison cell identities changed")
    reference = reference.loc[native.index]
    agreement = float(
        np.mean(native["classification"].to_numpy() == reference["classification"].to_numpy())
    )
    result = {
        "status": "ok",
        "experiment": "hto_demultiplexing",
        "provenance": provenance(),
        "dataset": dataset,
        "inputs": inputs,
        "cells": len(hto),
        "hashtags": hto.shape[1],
        "selection": selection,
        "parameters": ds.artifacts.inspect(ref).parameters,
        "referenceVersions": json.loads((work / "reference.json").read_text()),
        "settings": {
            "normalization": "CLR per hashtag",
            "kmeansStarts": 100,
            "clusters": hto.shape[1] + 1,
            "positiveQuantile": 0.99,
            "seed": 0,
            "globalAgreementTarget": 0.99,
        },
        "identityAgreement": float(
            np.mean(native["values"].to_numpy() == reference["identity"].to_numpy())
        ),
        "globalClassAgreement": agreement,
        "globalClassKappa": float(
            cohen_kappa_score(native["classification"], reference["classification"])
        ),
        "maxIdAgreement": float(
            np.mean(native["max_id"].to_numpy() == reference["max_id"].to_numpy())
        ),
        "marginMaxAbsoluteDifference": float(
            np.max(np.abs(native["margin"].to_numpy() - reference["margin"].to_numpy()))
        ),
        "meetsGlobalAgreementTarget": agreement >= 0.99,
        "comparisonKind": "Reference implementation agreement, not donor ground truth",
        "singletRule": "CyteArc assigns the sole positive tag; Seurat assigns the maximum-CLR tag",
    }
    merged = native.add_prefix("cytearc_").join(
        reference.add_prefix("seurat_"), validate="one_to_one"
    )
    merged.to_csv(work / "hto_calls.csv")
    storage = R2(payload["r2EndpointUrl"])
    prefix = payload["recordUri"].removesuffix(".json")
    storage.upload(work / "hto_calls.csv", f"{prefix}/hto_calls.csv")
    storage.upload(work / "reference.log", f"{prefix}/reference.log")
    write_json(
        "hto_demultiplexing.json", scrub_private(result), directory=Path(payload["outputDirectory"])
    )
    return result
