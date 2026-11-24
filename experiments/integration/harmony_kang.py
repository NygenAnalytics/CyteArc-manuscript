import cytearc
import numpy as np
from cytearc.metrics import clisi_knn, ilisi_knn

from experiments.integration import DATA, WORK
from experiments.integration import INPUTS as DATASETS
from experiments.integration.harmony_binvignat import (
    HARMONY2_SETTINGS,
    REFERENCE_SEED,
    harmonypy_arguments,
    reference_harmony,
)
from experiments.runtime.common import (
    Stopwatch,
    catalogue_store,
    configure_cytearc,
    fresh_copy,
    write_json,
)

DATASET = "kang_29K_ctrl-ifnb_pbmc_rnaseq"
DIMS = 30
K = 15


def knn(coordinates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    distances, indices = model.kneighbors(coordinates)
    return (distances[:, 1:], indices[:, 1:])


def main() -> None:
    configure_cytearc()
    source = catalogue_store(DATASET, destination=DATASETS) / "data.zarr"
    store = fresh_copy(source, WORK / "harmony_kang" / "kang.zarr")
    ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=4, mem_budget="6G")
    columns = list(ds.cells.columns)
    batch_column = next(
        (c for c in ("stim", "condition", "sample_id", "batch", "orig.ident") if c in columns)
    )
    label_column = next(
        (
            c
            for c in ("cluster_labels", "orig_cluster_labels", "cell_type", "celltype")
            if c in columns
        )
    )
    cells = ds.snapshot_cell_selection("I")
    hvg = ds.features.hvgs(cells, min_cells=20, top_n=2000, show_plot=False)
    normalized = ds.features.normalize(cells, hvg)
    pca = ds.reduction.pca(normalized, dims=DIMS, show_elbow_plot=False)
    neighbors = ds.graph.neighbors(ds.graph.ann_index(pca), k=K)
    uncorrected = {
        "iLISI": float(ds.integration.compute_ilisi(batch_column, neighbors)),
        "cLISI": float(ds.integration.compute_clisi(label_column, neighbors)),
    }
    with Stopwatch() as harmony_time:
        corrected = ds.reduction.harmony(pca, [batch_column])
    neighbors_h = ds.graph.neighbors(ds.graph.ann_index(corrected), k=K)
    cytearc_harmony = {
        "iLISI": float(ds.integration.compute_ilisi(batch_column, neighbors_h)),
        "cLISI": float(ds.integration.compute_clisi(label_column, neighbors_h)),
        "seconds": harmony_time.seconds,
    }
    batches = np.asarray(ds.cells.fetch(batch_column, key="I")).astype(str)
    labels = np.asarray(ds.cells.fetch(label_column, key="I")).astype(str)
    comparison: dict = {"available": False}
    try:
        import harmonypy
    except ImportError:
        harmonypy = None
    if harmonypy is not None:
        coordinates = np.asarray(ds.artifacts.load(pca)["data"][:], dtype=np.float64)
        cytearc_corrected = np.asarray(ds.artifacts.load(corrected)["data"][:], dtype=np.float64)
        reference, _ = reference_harmony(coordinates, batches, REFERENCE_SEED)
        d_ref, i_ref = knn(reference, K)
        d_cytearc, i_cytearc = knn(cytearc_corrected, K)
        comparison = {
            "available": True,
            "harmonypyVersion": getattr(harmonypy, "__version__", "unknown"),
            "settings": {
                **harmonypy_arguments(HARMONY2_SETTINGS, len(batches)),
                "random_state": REFERENCE_SEED,
            },
            "iLISI": float(ilisi_knn(d_ref, i_ref, batches)),
            "cLISI": float(clisi_knn(d_ref, i_ref, labels)),
            "perDimensionPearsonMin": float(
                min(
                    (
                        np.corrcoef(cytearc_corrected[:, j], reference[:, j])[0, 1]
                        for j in range(DIMS)
                    )
                )
            ),
            "knnOverlapK15": float(
                np.mean(
                    [
                        len(set(a.tolist()) & set(b.tolist())) / K
                        for a, b in zip(i_cytearc, i_ref, strict=True)
                    ]
                )
            ),
        }
    payload = {
        "description": "Harmony on Kang control and interferon-stimulated PBMCs: condition mixing (iLISI) and cell-type separation (cLISI) before and after correction.",
        "dataset": {
            "name": DATASET,
            "cells": int(ds.cells.N),
            "batchColumn": batch_column,
            "batches": sorted(set(batches)),
            "labelColumn": label_column,
            "labels": int(len(set(labels))),
        },
        "parameters": {"hvg": 2000, "dims": DIMS, "k": K},
        "uncorrected": uncorrected,
        "cytearcHarmony": cytearc_harmony,
        "harmonypy": comparison,
    }
    write_json("harmony_kang_comparison.json", payload, directory=DATA)
    print("uncorrected", uncorrected, "| harmony", cytearc_harmony, "| harmonypy", comparison)


if __name__ == "__main__":
    main()
