import inspect
from pathlib import Path

import cytearc
import numpy as np
from cytearc.trajectory.fate import (
    _make_transition,
    _normalize_pseudotime,
    compute_fate_probabilities,
)
from scipy.sparse import csr_matrix

from experiments.methods import DATA, WORK
from experiments.methods import INPUTS as DATASETS
from experiments.runtime import kang_labels
from experiments.runtime.catalogue_counts import build_store, input_record
from experiments.runtime.common import Stopwatch, configure_cytearc, write_json
from experiments.runtime.rna_recipe import analysis_cells, analysis_record, run_rna

_PARAMETERS = inspect.signature(compute_fate_probabilities).parameters
# Each real graph is that of CyteArc's default cell filter and RNA recipe on the dataset's pinned counts.
RUN_LABEL = "fate_comparison"
ANALYSES: dict[str, dict] = {}
BETA = float(_PARAMETERS["beta"].default) if "beta" in _PARAMETERS else 10.0


def dense_reference(transition: csr_matrix, labels: np.ndarray, sinks: list) -> np.ndarray:
    matrix = np.eye(transition.shape[0]) - transition.toarray()
    absorbing = np.isin(labels, sinks)
    matrix[absorbing] = 0.0
    matrix[np.flatnonzero(absorbing), np.flatnonzero(absorbing)] = 1.0
    probabilities = np.empty((transition.shape[0], len(sinks)), dtype=np.float64)
    for group, sink in enumerate(sinks):
        probabilities[:, group] = np.linalg.solve(
            matrix, np.asarray(labels == sink, dtype=np.float64)
        )
    return probabilities


def compare(
    graph: csr_matrix, pseudotime: np.ndarray, labels: np.ndarray, sinks: list, name: str
) -> dict:
    transition = _make_transition(
        graph.copy(), _normalize_pseudotime(pseudotime), np.isin(labels, sinks), beta=BETA
    )
    with Stopwatch() as dense_time:
        expected = dense_reference(transition, labels, sinks)
    with Stopwatch() as sparse_time:
        probabilities, valid, _ = compute_fate_probabilities(graph, pseudotime, labels, sinks)
    difference = np.abs(probabilities.astype(np.float64) - expected)
    return {
        "graph": name,
        "nodes": int(graph.shape[0]),
        "edges": int(graph.nnz),
        "sinks": len(sinks),
        "validFraction": float(valid.mean()),
        "maxAbsError": float(difference[valid].max()),
        "meanAbsError": float(difference[valid].mean()),
        "maxRowSumError": float(np.abs(probabilities[valid].sum(axis=1) - 1.0).max()),
        "denseSeconds": dense_time.seconds,
        "sparseSeconds": sparse_time.seconds,
        "outputDtype": str(probabilities.dtype),
    }


def y_graph() -> tuple:
    adjacency = np.zeros((5, 5))
    for a, b in ((0, 1), (1, 2), (0, 3), (3, 4)):
        adjacency[a, b] = adjacency[b, a] = 1.0
    return (
        csr_matrix(adjacency),
        np.array([0.0, 0.5, 1.0, 0.5, 1.0]),
        np.array(["root", "a-mid", "A", "b-mid", "B"]),
        ["A", "B"],
    )


def three_branch_graph() -> tuple:
    adjacency = np.zeros((7, 7))
    for a, b in ((0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)):
        adjacency[a, b] = adjacency[b, a] = 1.0
    return (
        csr_matrix(adjacency),
        np.array([0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]),
        np.array(["root", "a-mid", "A", "b-mid", "B", "c-mid", "C"]),
        ["A", "B", "C"],
    )


def store_graph(
    name: str, sinks: list[str], source_label: str, sink_labels: list[str], *, inputs: Path
) -> tuple:
    configure_cytearc()
    ds, run = analysed_store(name, f"{name}.zarr", inputs=inputs)
    graph_ref = run["connectivity_map"]
    column = "clusters" if "clusters" in ds.cells.columns else kang_labels.LABEL_COLUMN
    annotations = np.asarray(ds.cells.fetch_all(column)).astype(str)[analysis_cells(ds, run)]
    source_mask = annotations == source_label
    sink_mask = np.isin(annotations, sink_labels)
    vector = np.zeros(annotations.size)
    vector[source_mask] = -1.0 / max(1, source_mask.sum())
    vector[sink_mask] = 1.0 / max(1, sink_mask.sum())
    pseudotime = ds.trajectory.load_pseudotime(ds.trajectory.pseudotime(graph_ref, ss_vec=vector))
    valid = np.asarray(pseudotime.valid)
    graph = ds.graph.load(graph_ref, symmetric=True, upper_only=False)
    graph = graph[valid][:, valid].tocsr()
    return (graph, np.asarray(pseudotime.values)[valid], annotations[valid], sinks)


def analysed_store(dataset: str, store_name: str, *, inputs: Path) -> tuple:
    """A fresh store of the dataset's pinned counts and CyteArc's default analysis of it, recorded in ANALYSES.
    A Kang lane is labelled with GEO's author labels and analysed on its demuxlet singlets, the cells with a
    label."""
    store = build_store(dataset, WORK / "fate" / store_name, downloads=inputs)
    ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=4)
    record = {"input": input_record(dataset)}
    cells = "I"
    if dataset in kang_labels.CONDITIONS:
        record["labels"] = kang_labels.label_store(ds, dataset, inputs)
        cells = kang_labels.SINGLETS
    run = run_rna(
        ds,
        label=RUN_LABEL,
        cells=cells,
        umap=False,
        cell_cycle=False,
        paris=False,
        doublets=False,
        markers=False,
    )
    ANALYSES[dataset] = {**record, "analysis": analysis_record(ds, run)}
    return ds, run


def pancreas() -> dict:
    configure_cytearc()
    ds, run = analysed_store(
        "bastidas-ponce_4K_pancreas-d15_rnaseq", "pancreas_fate.zarr", inputs=DATASETS
    )
    graph = run["connectivity_map"]
    annotations = np.asarray(ds.cells.fetch_all("clusters")).astype(str)[analysis_cells(ds, run)]
    source_mask = annotations == "Ductal"
    sink_mask = np.isin(annotations, ["Alpha", "Beta", "Delta"])
    vector = np.zeros(annotations.size)
    vector[source_mask] = -1.0 / source_mask.sum()
    vector[sink_mask] = 1.0 / sink_mask.sum()
    pseudotime_ref = ds.trajectory.pseudotime(graph, ss_vec=vector)
    pseudotime = ds.trajectory.load_pseudotime(pseudotime_ref)
    cluster_values = np.asarray(run.cells.fetch("clusters"))
    cluster_labels = cluster_values.astype(str)
    valid = np.asarray(pseudotime.valid)
    means = {
        label: float(np.mean(np.asarray(pseudotime.values)[valid & (cluster_labels == label)]))
        for label in np.unique(cluster_labels[valid])
    }
    terminal_text = sorted(means, key=means.get, reverse=True)[:2]
    terminal = [cluster_values[cluster_labels == label][0].item() for label in terminal_text]
    majority = {}
    for label in terminal_text:
        mask = cluster_labels == label
        values, counts = np.unique(annotations[mask], return_counts=True)
        majority[label] = {
            "annotation": str(values[counts.argmax()]),
            "fraction": float(counts.max() / counts.sum()),
            "cells": int(mask.sum()),
        }
    with Stopwatch() as fate_time:
        fate_ref = ds.trajectory.fate(pseudotime_ref, run["clusters"], sinks=terminal)
    fate = ds.trajectory.load_fate(fate_ref)
    values = np.asarray(fate.values)
    fate_valid = np.asarray(fate.valid)
    per_annotation = {}
    for label in sorted(set(annotations)):
        mask = (annotations == label) & fate_valid
        if mask.any():
            per_annotation[label] = {
                str(sink): float(values[mask, i].mean()) for i, sink in enumerate(fate.sink_labels)
            }
    argmax_agreement = {}
    for i, sink in enumerate(fate.sink_labels):
        mask = (cluster_labels == str(sink)) & fate_valid
        argmax_agreement[str(sink)] = (
            float(np.mean(values[mask].argmax(axis=1) == i)) if mask.any() else float("nan")
        )
    return {
        "dataset": "bastidas-ponce_4K_pancreas-d15_rnaseq",
        "cells": int(annotations.size),
        "sinkClusters": [str(s) for s in fate.sink_labels],
        "sinkClusterMajorityAnnotation": majority,
        "clusterMeanPseudotime": means,
        "validFraction": float(fate_valid.mean()),
        "maxRowSumError": float(np.abs(values[fate_valid].sum(axis=1) - 1.0).max()),
        "meanProbabilityByAnnotation": per_annotation,
        "argmaxAgreementInSinkClusters": argmax_agreement,
        "seconds": fate_time.seconds,
    }


def main() -> None:
    rows = [
        compare(*y_graph(), "test-suite Y graph"),
        compare(*three_branch_graph(), "test-suite three-branch graph"),
    ]
    for name, sinks, source_label, sink_labels, inputs in (
        (
            "bastidas-ponce_4K_pancreas-d15_rnaseq",
            ["Alpha", "Beta"],
            "Ductal",
            ["Alpha", "Beta", "Delta"],
            DATASETS,
        ),
        (
            "kang_15K_pbmc_rnaseq",
            ["CD14+ Monocytes", "CD4 T cells"],
            "B cells",
            ["CD14+ Monocytes", "CD4 T cells"],
            DATASETS,
        ),
    ):
        graph, pseudotime, labels, sink_list = store_graph(
            name, sinks, source_label, sink_labels, inputs=inputs
        )
        rows.append(compare(graph, pseudotime, labels, sink_list, f"{name} connectivity map"))
    for row in rows:
        print(
            f"{row['graph']:<58} n={row['nodes']:>6} maxAbsErr={row['maxAbsError']:.2e} rowSumErr={row['maxRowSumError']:.2e} dense={row['denseSeconds']:.2f}s sparse={row['sparseSeconds']:.2f}s"
        )
    pancreas_result = pancreas()
    print(
        "pancreas",
        pancreas_result["sinkClusterMajorityAnnotation"],
        pancreas_result["argmaxAgreementInSinkClusters"],
    )
    write_json(
        "fate_probability_solver_agreement.json",
        {
            "description": "Sparse iterative fate solver against dense direct solutions on toy and real graphs; pancreas example.",
            "beta": BETA,
            "controlledGraphs": rows,
            "pancreas": pancreas_result,
            "analyses": ANALYSES,
        },
        directory=DATA,
    )


if __name__ == "__main__":
    main()
