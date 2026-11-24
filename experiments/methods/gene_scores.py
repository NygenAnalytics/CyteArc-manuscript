import hashlib
import shutil
import urllib.request
from pathlib import Path

import cytearc
import numpy as np
from cytearc.features.genomic.annotation import read_gene_annotation
from scipy.sparse import csr_matrix

from experiments.runtime.common import configure_cytearc, provenance, write_json

SETTINGS = {
    "genes": 256,
    "seed": 4471,
    "scale": 100000.0,
    "relativeTolerance": 2e-6,
    "absoluteTolerance": 1e-5,
}
ANNOTATION = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.basic.annotation.gtf.gz"


def run(payload):
    configure_cytearc()
    work = Path(payload["workDirectory"])
    work.mkdir(parents=True, exist_ok=True)
    annotation = work / "gencode.gtf.gz"
    if not annotation.exists():
        with (
            urllib.request.urlopen(ANNOTATION, timeout=120) as source,
            annotation.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
    with annotation.open("rb") as source:
        annotation_hash = hashlib.file_digest(source, "sha256").hexdigest()
    genes = read_gene_annotation(annotation, gene_types={"protein_coding"})
    genes = genes.iloc[
        np.sort(
            np.random.default_rng(SETTINGS["seed"]).choice(
                len(genes), SETTINGS["genes"], replace=False
            )
        )
    ]
    bed = work / "genes.bed"
    genes[["chrom", "start", "end", "gene_id", "gene_name"]].to_csv(
        bed, sep="\t", header=False, index=False
    )
    store = work / "atac.zarr"
    shutil.copytree(payload["sourcePath"], store)
    ds = cytearc.DataStore(str(store), default_assay="ATAC", nthreads=2, mem_budget="2G")
    peak_ids = ds.ATAC.feats.fetch_all("ids").astype(str)
    split = [name.replace(":", "-").split("-") for name in peak_ids]
    chromosome = np.array([x[0] for x in split])
    start = np.array([int(x[1]) for x in split])
    end = np.array([int(x[2]) for x in split])
    coordinates = np.array([f"{chrom}:{low}-{high}" for chrom, low, high in split])
    ds.ATAC.feats.insert("interval", coordinates, overwrite=True)
    raw = np.asarray(ds.z["ATAC/counts"][:], dtype=np.float64)
    totals = raw.sum(axis=1)
    if np.any(totals <= 0):
        raise RuntimeError("Gene-score input contains cells without ATAC counts")
    ds.add_melded_assay(
        from_assay="ATAC",
        external_bed_fn=str(bed),
        assay_label="GeneScores",
        peaks_col="interval",
        scalar_coeff=SETTINGS["scale"],
        renormalization=True,
        cell_key="I",
    )
    output_ids = ds.GeneScores.feats.fetch_all("ids").astype(str)
    aligned = genes.set_index("gene_id").loc[output_ids]
    mapping_rows = []
    mapping_columns = []
    overlap_counts = []
    for column, gene in enumerate(aligned.itertuples()):
        hits = np.flatnonzero((chromosome == gene.chrom) & (start < gene.end) & (end > gene.start))
        mapping_rows.extend(hits.tolist())
        mapping_columns.extend([column] * len(hits))
        overlap_counts.append(len(hits))
    mapping = csr_matrix(
        (np.ones(len(mapping_rows)), (mapping_rows, mapping_columns)),
        shape=(raw.shape[1], len(aligned)),
    )
    idf = np.log2(1 + raw.shape[0] / (np.count_nonzero(raw, axis=0) + 1))
    reference = np.asarray((raw / totals[:, None] * idf) @ mapping)
    sums = reference.sum(axis=1)
    positive = sums > 0
    reference[positive] *= SETTINGS["scale"] / sums[positive, None]
    observed = np.asarray(ds.z["GeneScores/counts"][:])
    error = np.abs(observed - reference)
    result = {
        "status": "ok",
        "experiment": "gene_scores",
        "provenance": provenance(),
        "settings": SETTINGS,
        "sourceDataset": "swanson_7K_pbmc_teaseq",
        "sourcePath": payload["sourcePath"],
        "cells": raw.shape[0],
        "peaks": raw.shape[1],
        "genes": len(aligned),
        "overlaps": len(mapping_rows),
        "genesWithoutOverlaps": int(np.count_nonzero(np.array(overlap_counts) == 0)),
        "annotation": {"url": ANNOTATION, "sha256": annotation_hash},
        "maxAbsoluteError": float(error.max()),
        "rootMeanSquaredError": float(np.sqrt(np.mean(error**2))),
        "storedDtype": str(observed.dtype),
        "meetsTolerance": bool(
            np.allclose(
                observed,
                reference,
                rtol=SETTINGS["relativeTolerance"],
                atol=SETTINGS["absoluteTolerance"],
            )
        ),
        "reference": "Independent half-open interval overlap and dense TF-IDF aggregation, normalized to 100000 per cell",
    }
    write_json("gene_scores.json", result, directory=Path(payload["outputDirectory"]))
    return result
