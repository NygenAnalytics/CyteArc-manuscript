import hashlib
from pathlib import Path

import numpy as np
import zarr
from cytearc.embeddings.reduction import fit_pca
from cytearc.matrix import ChunkedArray
from scipy.linalg import subspace_angles, svd
from sklearn.decomposition import IncrementalPCA
from threadpoolctl import threadpool_limits

from experiments.runtime.common import configure_cytearc, provenance, write_json

SETTINGS = {
    "cells": 1024,
    "features": 2000,
    "dimensions": 21,
    "seed": 4469,
    "batchSizes": [128, 256, 1024],
    "maxSubspaceAngleDegrees": 0.05,
}


def run(payload):
    configure_cytearc()
    source = zarr.open_group(payload["sourcePath"], mode="r", use_consolidated=False)
    counts = source["RNA/counts"]
    rows = np.sort(
        np.random.default_rng(SETTINGS["seed"]).choice(
            counts.shape[0], SETTINGS["cells"], replace=False
        )
    )
    raw = np.asarray(counts.oindex[rows, :], dtype=np.float64)
    total = raw.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("PCA comparison input includes empty cells")
    raw = np.log1p(raw * (10000 / total[:, None]))
    variance = raw.var(axis=0)
    selected = np.sort(np.argsort(-variance, kind="stable")[: SETTINGS["features"]])
    values = raw[:, selected]
    del raw
    scale = values.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("PCA comparison selected a constant feature")
    values = (values - values.mean(axis=0)) / scale
    threads = int(payload.get("workers", 2))
    dims = SETTINGS["dimensions"]
    records = []
    with threadpool_limits(limits=threads):
        _, singular, vt = svd(values, full_matrices=False)
        reference = vt[:dims].T
        ratio = singular[:dims] ** 2 / np.sum(singular**2)
        for batch_size in SETTINGS["batchSizes"]:
            array = ChunkedArray.from_numpy(values, block_size=batch_size, nthreads=threads)
            loadings, gram = fit_pca(
                array,
                dims=dims,
                use_for_pca=np.ones(len(values), dtype=bool),
                scale=None,
                random_state=4466,
                nthreads=threads,
            )
            incremental = IncrementalPCA(n_components=dims + 1, batch_size=batch_size)
            starts = list(range(0, len(values), batch_size))
            for start in starts[1:] + starts[:1]:
                incremental.partial_fit(values[start : start + batch_size].copy())
            for method, vectors, shares in (
                ("gram", loadings, gram.explained_variance_ratio_),
                (
                    "incremental",
                    incremental.components_[:dims].T,
                    incremental.explained_variance_ratio_[:dims],
                ),
            ):
                angles = np.degrees(subspace_angles(vectors, reference))
                records.append(
                    {
                        "method": method,
                        "batchSize": batch_size,
                        "blocks": len(starts),
                        "maxSubspaceAngleDegrees": float(angles.max()),
                        "medianSubspaceAngleDegrees": float(np.median(angles)),
                        "maxRelativeVarianceShareError": float(
                            np.max(np.abs(shares - ratio) / ratio)
                        ),
                    }
                )
    cell_ids = np.asarray(source["cellData/ids"][:])[rows].astype(str)
    gene_ids = np.asarray(source["RNA/featureData/ids"][:])[selected].astype(str)
    result = {
        "status": "ok",
        "experiment": "pca_drift",
        "provenance": provenance(),
        "settings": SETTINGS,
        "dataset": "bastidas-ponce_4K_pancreas-d15_rnaseq",
        "cellIdsSha256": hashlib.sha256("\n".join(cell_ids).encode()).hexdigest(),
        "featureIdsSha256": hashlib.sha256("\n".join(gene_ids).encode()).hexdigest(),
        "inputSha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "results": records,
        "preprocessing": "Counts scaled to 10,000 per cell and log1p transformed; top 2,000 genes by variance and population standardization; identical input for both solvers",
        "incrementalFitOrder": "First block reserved until the end, as in the superseded multi-block CyteArc fit",
        "reference": "Dense float64 SVD on the identical matrix",
        "performanceClaim": False,
    }
    write_json("pca_drift.json", result, directory=Path(payload["outputDirectory"]))
    return result
