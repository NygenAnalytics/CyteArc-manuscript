from typing import Any

import cytearc
import numpy as np
import pandas as pd
from cytearc.trajectory.pseudotime import random_walk_laplacian_transpose

from experiments.methods import DATA
from experiments.methods.method_inputs import count_store
from experiments.runtime.agreement import numeric_agreement
from experiments.runtime.catalogue_counts import input_record
from experiments.runtime.common import Stopwatch, configure_cytearc, write_json
from experiments.runtime.rna_recipe import analysis_record, run_rna

EXPERIMENT = "pba"
DATASET = "bastidas-ponce_4K_pancreas-d15_rnaseq"
# The graph is that of CyteArc's default cell filter and RNA recipe on the pancreas counts.
RUN_LABEL = "pba_comparison"
LABEL_COLUMN = "clusters"
SOURCES = ("Ductal",)
SINKS = ("Alpha", "Beta", "Delta")
N_SINGULAR_VALUES = (10, 30, 100)
RANDOM_SEED = 4444


def min_max(values: np.ndarray) -> np.ndarray:
    return (values - values.min()) / np.ptp(values)


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))


def main() -> None:
    configure_cytearc()
    with Stopwatch() as total:
        store = count_store(DATASET, EXPERIMENT, "pancreas")
        ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=4)
        run = run_rna(
            ds,
            label=RUN_LABEL,
            umap=False,
            cell_cycle=False,
            paris=False,
            doublets=False,
            markers=False,
        )
        graph = run["connectivity_map"]
        selection = ds.artifacts.load(run["analysis_cell_selection"])
        mask = np.asarray(selection["values"][:], dtype=bool)
        labels = np.asarray(ds.cells.fetch_all(LABEL_COLUMN)).astype(str)[mask]
        source = np.isin(labels, SOURCES)
        sink = np.isin(labels, SINKS)
        source_sink = np.zeros(labels.size, dtype=np.float64)
        source_sink[source] = -1.0 / source.sum()
        source_sink[sink] = 1.0 / sink.sum()
        cytearc_runs: dict[int, dict[str, Any]] = {}
        for k in N_SINGULAR_VALUES:
            with Stopwatch() as scaled_time:
                scaled_ref = ds.trajectory.pseudotime(
                    graph, ss_vec=source_sink, n_singular_vals=k, random_seed=RANDOM_SEED
                )
            scaled = ds.trajectory.load_pseudotime(scaled_ref)
            with Stopwatch() as raw_time:
                raw_ref = ds.trajectory.pseudotime(
                    graph,
                    ss_vec=source_sink,
                    n_singular_vals=k,
                    random_seed=RANDOM_SEED,
                    min_max_norm_ptime=False,
                )
            raw = ds.trajectory.load_pseudotime(raw_ref)
            stored_dtype = str(ds.artifacts.load(scaled_ref)["pseudotime"].dtype)
            cytearc_runs[k] = {
                "storedDtype": stored_dtype,
                "scaled": np.asarray(scaled.values, dtype=np.float64),
                "raw": np.asarray(raw.values, dtype=np.float64),
                "valid": np.asarray(scaled.valid, dtype=bool),
                "secondsScaled": scaled_time.seconds,
                "secondsRaw": raw_time.seconds,
            }
        valid = cytearc_runs[N_SINGULAR_VALUES[0]]["valid"]
        adjacency = ds.graph.load(graph, symmetric=True, upper_only=False)
        if not valid.all():
            adjacency = adjacency[valid][:, valid]
        dense = adjacency.toarray().astype(np.float64)
        n_cells = dense.shape[0]
        degree = dense.sum(axis=1)
        laplacian = np.eye(n_cells) - dense / degree[:, None]
        vector = source_sink[valid]
        with Stopwatch() as pinv_time:
            exact = np.linalg.pinv(laplacian, rtol=n_cells * np.finfo(np.float64).eps) @ vector
        with Stopwatch() as svd_time:
            left, singular, right_t = np.linalg.svd(laplacian.T)
        tolerance = n_cells * np.finfo(np.float64).eps * singular.max()
        order = np.argsort(singular)
        if singular[order[0]] > tolerance or singular[order[1]] <= tolerance:
            raise RuntimeError("expected exactly one null mode in the dense Laplacian")

        def potential_from_modes(modes: np.ndarray) -> np.ndarray:
            return left[:, modes] @ (right_t[modes] @ vector / singular[modes])

        cytearc_operator = random_walk_laplacian_transpose(adjacency.tocsr()).toarray()
        operator_difference = float(np.max(np.abs(cytearc_operator - laplacian.T)))
        cytearc_left, cytearc_singular, cytearc_right_t = np.linalg.svd(cytearc_operator)
        cytearc_order = np.argsort(cytearc_singular)

        def cytearc_operator_potential(k: int) -> np.ndarray:
            modes = cytearc_order[1:k]
            return cytearc_left[:, modes] @ (
                cytearc_right_t[modes] @ vector / cytearc_singular[modes]
            )

        exact_from_svd = potential_from_modes(order[1:])
        exact_scaled = min_max(exact)
        comparisons: dict[str, Any] = {}
        for k in N_SINGULAR_VALUES:
            truncated = potential_from_modes(order[1:k])
            cytearc_scaled = cytearc_runs[k]["scaled"][valid]
            cytearc_raw = cytearc_runs[k]["raw"][valid]
            comparisons[f"nSingularVals{k}"] = {
                "modesUsed": k - 1,
                "cytearcStoredDtype": cytearc_runs[k]["storedDtype"],
                "largestSingularValueUsed": float(singular[order[k - 1]]),
                "scaledCyteArcVersusExact": numeric_agreement(cytearc_scaled, exact_scaled),
                "scaledCyteArcVersusDenseTruncated": numeric_agreement(
                    cytearc_scaled, min_max(truncated)
                ),
                "scaledDenseTruncatedVersusExact": numeric_agreement(
                    min_max(truncated), exact_scaled
                ),
                "rawRelativeL2ErrorCyteArcVersusExact": relative_error(cytearc_raw, exact),
                "rawRelativeL2ErrorCyteArcVersusDenseTruncated": relative_error(
                    cytearc_raw, truncated
                ),
                "rawCyteArcVersusExact": numeric_agreement(cytearc_raw, exact),
                "rawRelativeL2ErrorCyteArcVersusDenseTruncatedOfCyteArcOperator": relative_error(
                    cytearc_raw, cytearc_operator_potential(k)
                ),
                "cytearcSecondsScaled": cytearc_runs[k]["secondsScaled"],
                "cytearcSecondsRaw": cytearc_runs[k]["secondsRaw"],
            }
            print(
                f"k={k}: scaled Spearman {comparisons[f'nSingularVals{k}']['scaledCyteArcVersusExact']['spearman']:.6f}",
                f"max|diff| {comparisons[f'nSingularVals{k}']['scaledCyteArcVersusExact']['maxAbsDiff']:.3e}",
                f"vs dense truncated max|diff| {comparisons[f'nSingularVals{k}']['scaledCyteArcVersusDenseTruncated']['maxAbsDiff']:.3e}",
                flush=True,
            )
        valid_labels = labels[valid]
        cluster_medians = (
            pd.DataFrame(
                {
                    "exact": exact_scaled,
                    **{
                        f"cytearcNSingularVals{k}": cytearc_runs[k]["scaled"][valid]
                        for k in N_SINGULAR_VALUES
                    },
                }
            )
            .groupby(valid_labels)
            .median()
        )
    payload = {
        "description": "CyteArc PBA pseudotime (truncated SVD of the transposed random-walk Laplacian) against the exact dense potential pinv(L) R on the same graph and source/sink vector, with a dense truncated SVD separating ARPACK accuracy from truncation.",
        "reference": {
            "implementation": "numpy.linalg.pinv and numpy.linalg.svd on the dense Laplacian",
            "numpyVersion": np.__version__,
            "operator": "L = I - D^-1 A, A the symmetrised connectivity map, D its degree matrix",
        },
        "comparisonKind": "exact reference: same graph, same source/sink vector, same scaling",
        "dataset": {
            "name": DATASET,
            "input": input_record(DATASET),
            "analysis": analysis_record(ds, run),
            "graph": "connectivity_map",
            "cells": int(mask.sum()),
            "cellsScored": int(valid.sum()),
            "graphEdgesSymmetric": int(adjacency.nnz),
            "connectedComponents": 1 if valid.all() else "more than one",
            "labelColumn": LABEL_COLUMN,
            "sources": list(SOURCES),
            "sinks": list(SINKS),
            "sourceCells": int(source.sum()),
            "sinkCells": int(sink.sum()),
        },
        "parameters": {
            "nSingularVals": list(N_SINGULAR_VALUES),
            "randomSeed": RANDOM_SEED,
            "sourceSinkVector": "-1/n_sources on sources, +1/n_sinks on sinks, 0 elsewhere",
            "scaling": "min-max on both sides",
        },
        "denseChecks": {
            "pinvSeconds": pinv_time.seconds,
            "svdSeconds": svd_time.seconds,
            "nullSingularValue": float(singular[order[0]]),
            "smallestNonzeroSingularValue": float(singular[order[1]]),
            "largestSingularValue": float(singular.max()),
            "maxAbsDiffPinvVersusSvdExact": float(np.max(np.abs(exact - exact_from_svd))),
            "graphWeightDtype": str(adjacency.dtype),
            "maxAbsDiffCyteArcOperatorVersusFloat64Operator": operator_difference,
        },
        "comparisons": comparisons,
        "clusterMedianScaledPseudotime": cluster_medians.round(6).to_dict(orient="index"),
        "notes": [
            "CyteArc returns U S^-1 V^T R from the k smallest singular triplets of L^T after dropping the null mode, which equals pinv(L) R restricted to those modes; the omitted modes have the largest singular values.",
            "The Laplacian convention is CyteArc's; the comparison checks the numerical method, not the choice of Laplacian.",
            "CyteArc against the dense SVD truncated to the same modes measures numerical precision; CyteArc against pinv(L) R adds the truncation error of the omitted modes.",
            "The reference operator is built in float64. CyteArc sums the degrees of the float32 graph in float32, which perturbs L^T by about 2e-8 and the raw potential by about 3e-6 (relative L2). Against a dense SVD of CyteArc's own operator the ARPACK result agrees to about 1e-11, so the solver itself contributes nothing measurable.",
        ],
        "runtimeSeconds": total.seconds,
    }
    write_json("pancreas_population_balance_agreement.json", payload, directory=DATA)


if __name__ == "__main__":
    main()
