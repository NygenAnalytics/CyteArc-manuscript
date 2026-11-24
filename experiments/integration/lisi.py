from collections.abc import Callable
from typing import Any

import cytearc
import numpy as np
import pandas as pd
from cytearc.metrics import clisi_knn, compute_lisi, ilisi_knn

from experiments.integration import DATA, WORK
from experiments.integration import INPUTS as DATASETS
from experiments.integration.harmony_binvignat import prepare_inputs as prepare_inputs
from experiments.runtime.agreement import numeric_agreement, package_version
from experiments.runtime.common import (
    Stopwatch,
    catalogue_store,
    configure_cytearc,
    fresh_copy,
    load_json,
    write_json,
)

EXPERIMENT = "lisi"
K = 15
PERPLEXITY = 5.0
HARMONYPY_DEFAULT_PERPLEXITY = 30.0
DIMS = 30
RA_SOURCE = DATASETS / "binvignat" / "data.zarr"
RA_SELECTION = "manuscript_ra_balanced"
RA_CELLS_PER_GROUP = 1500
RA_SEED = 42
KANG_DATASET = "kang_29K_ctrl-ifnb_pbmc_rnaseq"


def open_ra_store() -> tuple[cytearc.DataStore, dict[str, Any]]:
    target = WORK / EXPERIMENT / "ra_batch.zarr"
    source = WORK / "harmony_binvignat" / "ra_batch.zarr"
    if source.exists():
        fresh_copy(source, target)
        ds = cytearc.DataStore(
            str(target), default_assay="RNA", min_features_per_cell=0, nthreads=8
        )
        origin = "copy of experiments/integration/workspace/harmony_binvignat/ra_batch.zarr"
    else:
        if target.exists():
            import shutil

            shutil.rmtree(target)
        ds = cytearc.mount_datastore(
            str(RA_SOURCE), at=str(target), default_assay="RNA", min_features_per_cell=0, nthreads=8
        )
        batch = ds.cells.fetch_all("batch").astype(str)
        disease = ds.cells.fetch_all("disease").astype(str)
        rng = np.random.default_rng(RA_SEED)
        selected = np.zeros(ds.cells.N, dtype=bool)
        for batch_value in np.unique(batch):
            for disease_value in np.unique(disease):
                candidates = np.flatnonzero((batch == batch_value) & (disease == disease_value))
                chosen = rng.choice(candidates, size=RA_CELLS_PER_GROUP, replace=False)
                selected[chosen] = True
        ds.cells.insert(column_name=RA_SELECTION, values=selected, overwrite=True)
        origin = "rebuilt with the binvignat recipe"
    return (
        ds,
        {
            "origin": origin,
            "selectionKey": RA_SELECTION,
            "batchColumn": "batch",
            "labelColumn": "cell_type",
            "reported": "harmony_binvignat_comparison.json",
        },
    )


def open_kang_store() -> tuple[cytearc.DataStore, dict[str, Any]]:
    target = WORK / EXPERIMENT / "kang.zarr"
    source = WORK / "harmony_kang" / "kang.zarr"
    if source.exists():
        fresh_copy(source, target)
        origin = "copy of experiments/integration/workspace/harmony_kang/kang.zarr"
    else:
        fresh_copy(catalogue_store(KANG_DATASET, destination=DATASETS) / "data.zarr", target)
        origin = "rebuilt from the catalogue store with the kang recipe"
    ds = cytearc.DataStore(str(target), default_assay="RNA", nthreads=4, mem_budget="6G")
    return (
        ds,
        {
            "origin": origin,
            "selectionKey": "I",
            "batchColumn": "sample_id",
            "labelColumn": "orig_cluster_labels",
            "reported": "harmony_kang_comparison.json",
        },
    )


def exact_knn(coordinates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    distances, indices = model.kneighbors(coordinates)
    rows = np.arange(coordinates.shape[0])[:, None]
    keep = indices != rows
    missing_self = keep.all(axis=1)
    keep[missing_self, -1] = False
    return (distances[keep].reshape(-1, k), indices[keep].reshape(-1, k))


def scale(median: float, n_categories: int, *, invert: bool) -> float:
    if invert:
        return float((n_categories - median) / (n_categories - 1))
    return float((median - 1) / (n_categories - 1))


def scib_per_cell(distances: np.ndarray, indices: np.ndarray, labels: np.ndarray):
    from scib_metrics import lisi_knn
    from scib_metrics.nearest_neighbors import NeighborsResults

    neighbors = NeighborsResults(
        indices=np.asarray(indices, dtype=np.int64),
        distances=np.asarray(distances, dtype=np.float64),
    )
    return np.asarray(lisi_knn(neighbors, labels, perplexity=PERPLEXITY), np.float64)


def scib_summary(
    distances: np.ndarray, indices: np.ndarray, batches: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    from scib_metrics import clisi_knn as scib_clisi
    from scib_metrics import ilisi_knn as scib_ilisi
    from scib_metrics.nearest_neighbors import NeighborsResults

    neighbors = NeighborsResults(
        indices=np.asarray(indices, dtype=np.int64),
        distances=np.asarray(distances, dtype=np.float64),
    )
    return {
        "iLISI": float(scib_ilisi(neighbors, batches, perplexity=PERPLEXITY)),
        "cLISI": float(scib_clisi(neighbors, labels, perplexity=PERPLEXITY)),
    }


def cytearc_per_cell(distances: np.ndarray, indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"labels": labels})
    return compute_lisi(distances, indices, frame, ["labels"], perplexity=PERPLEXITY)[:, 0]


def harmonypy_summary(
    coordinates: np.ndarray, batches: np.ndarray, labels: np.ndarray, perplexity: float
) -> dict[str, Any]:
    import harmonypy

    frame = pd.DataFrame({"batch": batches, "label": labels})
    with Stopwatch() as elapsed:
        values = harmonypy.compute_lisi(
            np.ascontiguousarray(coordinates, dtype=np.float64),
            frame,
            ["batch", "label"],
            perplexity=int(perplexity),
        )
    n_batches = len(np.unique(batches))
    n_labels = len(np.unique(labels))
    return {
        "perplexity": perplexity,
        "neighbours": int(perplexity * 3),
        "iLISI": scale(float(np.nanmedian(values[:, 0])), n_batches, invert=False),
        "cLISI": scale(float(np.nanmedian(values[:, 1])), n_labels, invert=True),
        "medianBatchLisi": float(np.nanmedian(values[:, 0])),
        "medianLabelLisi": float(np.nanmedian(values[:, 1])),
        "seconds": elapsed.seconds,
        "perCellBatch": values[:, 0],
        "perCellLabel": values[:, 1],
    }


def per_cell_block(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    stats = numeric_agreement(first, second)
    return {key: stats[key] for key in ("n", "pearson", "maxAbsDiff", "meanAbsDiff")}


def evaluate_embedding(
    name: str,
    coordinates: np.ndarray,
    batches: np.ndarray,
    labels: np.ndarray,
    *,
    graph: tuple[np.ndarray, np.ndarray] | None,
    cytearc_metric: Callable[[], dict[str, float]] | None,
    reported: dict[str, float] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"embedding": name, "cells": int(coordinates.shape[0])}
    n_batches = len(np.unique(batches))
    n_labels = len(np.unique(labels))
    if reported is not None:
        result["reported"] = reported
    exact_d, exact_i = exact_knn(coordinates, K)
    if graph is not None and cytearc_metric is not None:
        distances = np.asarray(graph[0], dtype=np.float64)
        indices = np.asarray(graph[1], dtype=np.int64)
        cytearc_values = cytearc_metric()
        scib_values = scib_summary(distances, indices, batches, labels)
        cytearc_batch = cytearc_per_cell(distances, indices, batches)
        cytearc_label = cytearc_per_cell(distances, indices, labels)
        scib_batch = scib_per_cell(distances, indices, batches)
        scib_label = scib_per_cell(distances, indices, labels)
        overlap = np.mean(
            [
                len(set(a.tolist()) & set(b.tolist())) / K
                for a, b in zip(indices, exact_i, strict=True)
            ]
        )
        result["hnswGraph"] = {
            "comparisonKind": "exact parity: identical neighbour arrays and perplexity",
            "cytearc": cytearc_values,
            "scibMetrics": scib_values,
            "absDiffILISI": abs(cytearc_values["iLISI"] - scib_values["iLISI"]),
            "absDiffCLISI": abs(cytearc_values["cLISI"] - scib_values["cLISI"]),
            "perCellBatchLisiCyteArcVersusScib": per_cell_block(cytearc_batch, scib_batch),
            "perCellLabelLisiCyteArcVersusScib": per_cell_block(cytearc_label, scib_label),
            "recallAgainstExactKnn": float(overlap),
        }
        if reported is not None:
            result["hnswGraph"]["cytearcEqualsReported"] = bool(
                cytearc_values["iLISI"] == reported["iLISI"]
                and cytearc_values["cLISI"] == reported["cLISI"]
            )
    cytearc_exact = {
        "iLISI": float(ilisi_knn(exact_d, exact_i, batches, perplexity=PERPLEXITY)),
        "cLISI": float(clisi_knn(exact_d, exact_i, labels, perplexity=PERPLEXITY)),
    }
    scib_exact = scib_summary(exact_d, exact_i, batches, labels)
    cytearc_exact_batch = cytearc_per_cell(exact_d, exact_i, batches)
    scib_exact_batch = scib_per_cell(exact_d, exact_i, batches)
    result["exactKnn"] = {
        "comparisonKind": "exact parity on exact Euclidean neighbours",
        "cytearc": cytearc_exact,
        "scibMetrics": scib_exact,
        "absDiffILISI": abs(cytearc_exact["iLISI"] - scib_exact["iLISI"]),
        "absDiffCLISI": abs(cytearc_exact["cLISI"] - scib_exact["cLISI"]),
        "perCellBatchLisiCyteArcVersusScib": per_cell_block(cytearc_exact_batch, scib_exact_batch),
        "cellsDifferingByMoreThan1e-6": int(
            np.sum(np.abs(cytearc_exact_batch - scib_exact_batch) > 1e-06)
        ),
    }
    matched = harmonypy_summary(coordinates, batches, labels, PERPLEXITY)
    default = harmonypy_summary(coordinates, batches, labels, HARMONYPY_DEFAULT_PERPLEXITY)
    harmonypy_batch = matched["perCellBatch"]
    difference = np.abs(harmonypy_batch - cytearc_exact_batch)
    differing = np.flatnonzero(difference > 1e-06)
    result["harmonypy"] = {
        "matchedPerplexity5": {
            "comparisonKind": "perplexity 5: harmonypy's exact kd-tree search, 14 neighbours after dropping the cell, against 15 for CyteArc",
            **{k: v for k, v in matched.items() if not k.startswith("perCell")},
            "perCellBatchLisiVersusCyteArcExactKnn": per_cell_block(
                harmonypy_batch, cytearc_exact_batch
            ),
            "cellsDifferingByMoreThan1e-6": int(differing.size),
            "cellsWithHarmonypyFailureValueZero": int(np.sum(harmonypy_batch == 0)),
            "maxAbsDiffOtherCells": float(np.max(difference[harmonypy_batch != 0], initial=0.0)),
            "differingCells": [
                {
                    "row": int(row),
                    "cytearcBatchLisi": float(cytearc_exact_batch[row]),
                    "harmonypyBatchLisi": float(harmonypy_batch[row]),
                    "scibBatchLisi": float(scib_exact_batch[row]),
                    "nearestDistance": float(exact_d[row].min()),
                    "farthestDistance": float(exact_d[row].max()),
                }
                for row in differing[:5]
            ],
        },
        "defaultPerplexity30": {
            "comparisonKind": "harmonypy default perplexity 30 (89 neighbours after dropping the cell)",
            **{k: v for k, v in default.items() if not k.startswith("perCell")},
        },
    }
    result["categories"] = {"batches": n_batches, "labels": n_labels}
    return result


def run_dataset(ds: cytearc.DataStore, info: dict[str, Any], batch_key: str) -> dict[str, Any]:
    import harmonypy

    reported_doc = load_json(DATA / info["reported"])
    artifacts_before = len(ds.artifacts.list()) + len(ds.artifacts.list(scope="datastore"))
    with Stopwatch() as rebuild_time:
        cells = ds.snapshot_cell_selection(info["selectionKey"])
        hvg = ds.features.hvgs(cells, min_cells=20, top_n=2000, show_plot=False)
        normalized = ds.features.normalize(cells, hvg)
        pca = ds.reduction.pca(normalized, dims=DIMS, show_elbow_plot=False)
        neighbors = ds.graph.neighbors(ds.graph.ann_index(pca), k=K)
        corrected = ds.reduction.harmony(pca, [batch_key])
        neighbors_h = ds.graph.neighbors(ds.graph.ann_index(corrected), k=K)
    artifacts_after = len(ds.artifacts.list()) + len(ds.artifacts.list(scope="datastore"))
    batches = np.asarray(ds.cells.fetch(info["batchColumn"], key=info["selectionKey"])).astype(str)
    labels = np.asarray(ds.cells.fetch(info["labelColumn"], key=info["selectionKey"])).astype(str)
    coordinates = np.asarray(ds.artifacts.load(pca)["data"][:], dtype=np.float64)
    corrected_coordinates = np.asarray(ds.artifacts.load(corrected)["data"][:], dtype=np.float64)
    if coordinates.shape[0] != batches.size:
        raise RuntimeError("coordinates do not match the selection")

    def graph_arrays(ref: cytearc.ArtifactRef) -> tuple[np.ndarray, np.ndarray]:
        group = ds.artifacts.load(ref)
        distances = np.asarray(group["distances"][:], dtype=np.float64)
        indices = np.asarray(group["indices"][:], dtype=np.int64)
        if distances.shape != (batches.size, K):
            raise RuntimeError("neighbour arrays do not match the selection")
        return (distances, indices)

    def cytearc_metric(ref: cytearc.ArtifactRef) -> Callable[[], dict[str, float]]:
        return lambda: {
            "iLISI": float(ds.integration.compute_ilisi(info["batchColumn"], ref)),
            "cLISI": float(ds.integration.compute_clisi(info["labelColumn"], ref)),
        }

    fitted = harmonypy.run_harmony(
        coordinates,
        pd.DataFrame({"batch": batches}),
        ["batch"],
        sigma=0.1,
        max_iter_harmony=50,
        random_state=0,
        verbose=False,
    )
    reference = np.asarray(getattr(fitted, "Z_corr"))
    if reference.shape[0] == DIMS and reference.shape[1] != DIMS:
        reference = reference.T
    reference = reference.astype(np.float64)
    embeddings = {
        "uncorrected": evaluate_embedding(
            "uncorrected PCA",
            coordinates,
            batches,
            labels,
            graph=graph_arrays(neighbors),
            cytearc_metric=cytearc_metric(neighbors),
            reported={
                "iLISI": reported_doc["uncorrected"]["iLISI"],
                "cLISI": reported_doc["uncorrected"]["cLISI"],
            },
        ),
        "cytearcHarmony": evaluate_embedding(
            "CyteArc Harmony",
            corrected_coordinates,
            batches,
            labels,
            graph=graph_arrays(neighbors_h),
            cytearc_metric=cytearc_metric(neighbors_h),
            reported={
                "iLISI": reported_doc["cytearcHarmony"]["iLISI"],
                "cLISI": reported_doc["cytearcHarmony"]["cLISI"],
            },
        ),
        "harmonypyCorrected": evaluate_embedding(
            "harmonypy Harmony (recomputed)",
            reference,
            batches,
            labels,
            graph=None,
            cytearc_metric=None,
            reported={
                "iLISI": reported_doc["harmonypy"]["iLISI"],
                "cLISI": reported_doc["harmonypy"]["cLISI"],
            },
        ),
    }
    return {
        "store": info,
        "cells": int(batches.size),
        "artifactsBeforeRecipe": artifacts_before,
        "artifactsAfterRecipe": artifacts_after,
        "recipeSeconds": rebuild_time.seconds,
        "embeddings": embeddings,
    }


def main() -> None:
    try:
        from importlib import import_module

        import jax

        import_module("harmonypy")
        import_module("scib_metrics")
    except ImportError as error:
        raise SystemExit(
            "scib-metrics and harmonypy are required: uv run --with scib-metrics --with harmonypy python -m experiments.integration.lisi"
        ) from error
    jax.config.update("jax_enable_x64", True)
    configure_cytearc()
    results: dict[str, Any] = {}
    with Stopwatch() as total:
        ra, ra_info = open_ra_store()
        results["binvignat_pbmc"] = run_dataset(ra, ra_info, "batch")
        kang, kang_info = open_kang_store()
        results["kang_pbmc"] = run_dataset(kang, kang_info, kang_info["batchColumn"])
    for dataset, block in results.items():
        for name, embedding in block["embeddings"].items():
            hnsw = embedding.get("hnswGraph", {})
            print(
                dataset,
                name,
                "reported",
                embedding.get("reported"),
                "| cytearc",
                hnsw.get("cytearc"),
                "scib",
                hnsw.get("scibMetrics"),
                "| exact cytearc",
                embedding["exactKnn"]["cytearc"],
                "scib",
                embedding["exactKnn"]["scibMetrics"],
                "| harmonypy p5",
                {k: embedding["harmonypy"]["matchedPerplexity5"][k] for k in ("iLISI", "cLISI")},
                "p30",
                {k: embedding["harmonypy"]["defaultPerplexity30"][k] for k in ("iLISI", "cLISI")},
                flush=True,
            )
    payload = {
        "description": "iLISI and cLISI of binvignat and kang recomputed with scib-metrics and harmonypy on the same embeddings: identical HNSW neighbour arrays, exact Euclidean neighbours, and harmonypy's own search at perplexity 5 and 30.",
        "reference": {
            "scibMetricsVersion": package_version("scib-metrics"),
            "jaxVersion": package_version("jax"),
            "jaxX64": True,
            "harmonypyVersion": package_version("harmonypy"),
            "scikitLearnVersion": package_version("scikit-learn"),
        },
        "parameters": {
            "k": K,
            "perplexity": PERPLEXITY,
            "perplexityRule": "CyteArc default floor(k / 3)",
            "harmonypyDefaultPerplexity": HARMONYPY_DEFAULT_PERPLEXITY,
            "summary": "median per-cell LISI, iLISI scaled (m - 1) / (B - 1), cLISI scaled (L - m) / (L - 1)",
            "harmonypyRerun": {"sigma": 0.1, "maxIterHarmony": 50, "randomState": 0},
        },
        "datasets": results,
        "notes": [
            "scib-metrics lisi_knn masks self edges and uses distances as given; CyteArc subtracts each row's minimum distance before the Gaussian kernel, which leaves the normalised probabilities and entropies unchanged.",
            "harmonypy 0.0.10 computes LISI with scikit-learn's exact kd-tree search of 3 x perplexity neighbours, the first of which is the cell itself and is dropped, so at perplexity 5 it uses 14 neighbours where CyteArc and scib-metrics use 15.",
            "harmonypy and scib-metrics evaluate exp(-d * beta) without subtracting the row minimum. For an isolated cell whose neighbours are all far away and nearly equidistant, the perplexity search drives beta up until the kernel underflows: harmonypy then returns its failure value 0 and scib-metrics loses precision. differingCells lists such cells; CyteArc's centred kernel does not underflow.",
            "The Kang uncorrected graph separates the two conditions completely for most cells, so the median batch LISI is exactly 1 and the scaled iLISI 0.",
            "The Kang store holds 18,598 cells; the kang recipe and this record use the 18,597 cells of its I selection.",
        ],
        "runtimeSeconds": total.seconds,
    }
    write_json("lisi_reference_agreement.json", payload, directory=DATA)


if __name__ == "__main__":
    main()
