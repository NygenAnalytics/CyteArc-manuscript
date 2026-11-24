import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cytearc
import numpy as np
import pandas as pd
from cytearc.metrics import clisi_knn, ilisi_knn

from experiments.integration import DATA, HERE, WORK
from experiments.integration import INPUTS as DATASETS
from experiments.integration.harmony_binvignat import (
    HARMONY2_SETTINGS,
    HARMONYPY_VERSION,
    harmonypy_arguments,
    reference_harmony,
)
from experiments.integration.harmony_binvignat import prepare_inputs as prepare_inputs
from experiments.runtime.common import catalogue_store, configure_cytearc, fresh_copy, write_json

SEEDS = tuple(range(10))
ITERATIONS = (1, 2, 3, 5, 10, 20)
DIMS = 30
K = 15
PERPLEXITY = 5
BINVIGNAT = DATASETS / "binvignat" / "data.zarr"
BINVIGNAT_SELECTION = "harmony_diag"
KANG = "kang_29K_ctrl-ifnb_pbmc_rnaseq"
KANG_SELECTION = "I"
# Runs that fix their number of rounds also fix the clustering passes of each round: both stopping
# rules are off, because the two tools stop on different objectives.
FIXED_ROUNDS = 10
FIXED_PASSES = 20
FIXED = {
    "max_iter_harmony": FIXED_ROUNDS,
    "max_iter_cluster": FIXED_PASSES,
    "epsilon_harmony": 0.0,
    "epsilon_cluster": 0.0,
}
# The Harmony2 parity checks run on at most this many cells of each dataset, drawn with this seed.
PARITY_CELLS = 6000
PARITY_SAMPLE_SEED = 4444
# Full-fit arms of the parity checks, as changes to HARMONY2_SETTINGS: each tool with its own
# stopping rules, and a fixed number of rounds and passes with the diversity penalty on and off.
PARITY_ARMS = {
    "ownStoppingRules": {},
    "fixedRoundsPenaltyOn": FIXED,
    "fixedRoundsPenaltyOff": {**FIXED, "theta": 0.0},
}
# The reference kernels compute in float32: an assignment step agrees within this absolute error,
# and a correction step within this fraction of the largest absolute input coordinate.
ASSIGNMENT_TOLERANCE = 2e-6
CORRECTION_TOLERANCE = 1e-5
EQUIVALENCE_RULE = (
    "CyteArc's full fits agree with the reference's as closely as the reference's own fits agree "
    "across seeds: the median 15-NN overlap of CyteArc and reference fits over all seed pairs is "
    "at least the smallest overlap between two reference seeds, and the medians of the two tools'"
    " iLISI differ by at most the range of the reference's iLISI across seeds."
)
R_REFERENCE = HERE / "harmony2_reference.R"


def knn(coordinates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    distances, indices = (
        NearestNeighbors(n_neighbors=k + 1).fit(coordinates).kneighbors(coordinates)
    )
    return (distances[:, 1:], indices[:, 1:])


def binvignat_inputs():
    target = WORK / "harmony_seed_stability" / "binvignat.zarr"
    if target.exists():
        shutil.rmtree(target)
    ds = cytearc.mount_datastore(
        str(BINVIGNAT), at=str(target), default_assay="RNA", min_features_per_cell=0, nthreads=8
    )
    batch = ds.cells.fetch_all("batch").astype(str)
    disease = ds.cells.fetch_all("disease").astype(str)
    rng = np.random.default_rng(42)
    selected = np.zeros(ds.cells.N, dtype=bool)
    for batch_value in np.unique(batch):
        for disease_value in np.unique(disease):
            candidates = np.flatnonzero((batch == batch_value) & (disease == disease_value))
            selected[rng.choice(candidates, size=1500, replace=False)] = True
    ds.cells.insert(column_name=BINVIGNAT_SELECTION, values=selected, overwrite=True)
    cells = ds.snapshot_cell_selection(BINVIGNAT_SELECTION)
    hvg = ds.features.hvgs(cells, min_cells=20, top_n=2000, show_plot=False)
    pca = ds.reduction.pca(ds.features.normalize(cells, hvg), dims=DIMS, show_elbow_plot=False)
    batches = np.asarray(ds.cells.fetch("batch", key=BINVIGNAT_SELECTION)).astype(str)
    labels = np.asarray(ds.cells.fetch("cell_type", key=BINVIGNAT_SELECTION)).astype(str)
    return (ds, pca, "batch", batches, labels)


def kang_inputs():
    store = fresh_copy(
        catalogue_store(KANG, destination=DATASETS) / "data.zarr",
        WORK / "harmony_seed_stability" / "kang.zarr",
    )
    ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=8, mem_budget="6G")
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
    cells = ds.snapshot_cell_selection(KANG_SELECTION)
    hvg = ds.features.hvgs(cells, min_cells=20, top_n=2000, show_plot=False)
    pca = ds.reduction.pca(ds.features.normalize(cells, hvg), dims=DIMS, show_elbow_plot=False)
    batches = np.asarray(ds.cells.fetch(batch_column, key=KANG_SELECTION)).astype(str)
    labels = np.asarray(ds.cells.fetch(label_column, key=KANG_SELECTION)).astype(str)
    return (ds, pca, batch_column, batches, labels)


def compare(a: np.ndarray, b: np.ndarray, neighbours: dict) -> dict:
    pearson = [float(np.corrcoef(a[:, j], b[:, j])[0, 1]) for j in range(a.shape[1])]
    overlap = float(
        np.mean(
            [
                len(set(x.tolist()) & set(y.tolist())) / K
                for x, y in zip(neighbours[id(a)], neighbours[id(b)], strict=True)
            ]
        )
    )
    rmse = float(np.sqrt(np.mean((a - b) ** 2)) / np.std(b))
    return {
        "pearsonMin": min(pearson),
        "pearsonMedian": float(np.median(pearson)),
        "knnOverlapK15": overlap,
        "rmseOverStd": rmse,
        "maxAbsoluteDifference": float(np.max(np.abs(a - b))),
    }


def summary(rows: list[dict]) -> dict:
    return {
        key: {
            "min": float(np.min([r[key] for r in rows])),
            "median": float(np.median([r[key] for r in rows])),
            "max": float(np.max([r[key] for r in rows])),
        }
        for key in rows[0]
    }


def cytearc_harmony(
    coordinates: np.ndarray, batches: np.ndarray, seed: int, **changes
) -> tuple[np.ndarray, dict]:
    """CyteArc's Harmony2 on one thread with HARMONY2_SETTINGS and the given changes; returns the
    corrected coordinates, one row per cell, and the fit's record of its options and progress.
    """
    from cytearc.embeddings.harmony import fit_harmony

    fit = fit_harmony(
        np.ascontiguousarray(coordinates.T),
        pd.DataFrame({"batch": batches}),
        random_state=seed,
        nthreads=1,
        **{**HARMONY2_SETTINGS, **changes},
    )
    return np.ascontiguousarray(np.asarray(fit.corrected, dtype=np.float64).T), fit.parameters


def run_dataset(ds, pca, batch_column: str, batches: np.ndarray, labels: np.ndarray) -> dict:
    coordinates = np.asarray(ds.artifacts.load(pca)["data"][:], dtype=np.float64)
    cytearc_runs, reference_runs, iterations = ({}, {}, {})
    for seed in SEEDS:
        corrected = ds.reduction.harmony(pca, [batch_column], harmony_params={"random_state": seed})
        cytearc_runs[seed] = np.asarray(ds.artifacts.load(corrected)["data"][:], dtype=np.float64)
        reference_runs[seed], fitted = reference_harmony(coordinates, batches, seed)
        objective = getattr(fitted, "objective_harmony", None)
        iterations[seed] = None if objective is None else len(list(objective))
    neighbours, lisi = ({}, {"cytearc": {}, "harmonypy": {}})
    for name, runs in (("cytearc", cytearc_runs), ("harmonypy", reference_runs)):
        for seed, values in runs.items():
            distances, indices = knn(values, K)
            neighbours[id(values)] = indices
            lisi[name][seed] = {
                "iLISI": float(ilisi_knn(distances, indices, batches, perplexity=PERPLEXITY)),
                "cLISI": float(clisi_knn(distances, indices, labels, perplexity=PERPLEXITY)),
            }
    limited = iteration_limits(coordinates, batches, labels)
    same_seed = [compare(cytearc_runs[s], reference_runs[s], neighbours) for s in SEEDS]
    across = [compare(cytearc_runs[s], reference_runs[t], neighbours) for s in SEEDS for t in SEEDS]
    within_reference = [
        compare(reference_runs[s], reference_runs[t], neighbours)
        for s in SEEDS
        for t in SEEDS
        if s < t
    ]
    within_cytearc = [
        compare(cytearc_runs[s], cytearc_runs[t], neighbours) for s in SEEDS for t in SEEDS if s < t
    ]
    return {
        "cells": int(coordinates.shape[0]),
        "batches": sorted(set(batches.tolist())),
        "harmonypyIterations": iterations,
        "lisi": lisi,
        "cytearcVersusHarmonypySameSeed": summary(same_seed),
        "cytearcVersusHarmonypyAnySeeds": summary(across),
        "harmonypyBetweenSeeds": summary(within_reference),
        "cytearcBetweenSeeds": summary(within_cytearc),
        "seed0": {"cytearcVersusHarmonypy": same_seed[0]},
        "iterationLimits": limited,
    }


def iteration_limits(coordinates: np.ndarray, batches: np.ndarray, labels: np.ndarray) -> dict:
    out = {}
    for n in ITERATIONS:
        changes = {**FIXED, "max_iter_harmony": n}
        cytearc_corrected, cytearc_parameters = cytearc_harmony(coordinates, batches, 0, **changes)
        reference, fitted = reference_harmony(coordinates, batches, 0, **changes)
        neighbours, lisi = ({}, {})
        for name, values in (("cytearc", cytearc_corrected), ("harmonypy", reference)):
            distances, indices = knn(values, K)
            neighbours[id(values)] = indices
            lisi[name] = {
                "iLISI": float(ilisi_knn(distances, indices, batches, perplexity=PERPLEXITY)),
                "cLISI": float(clisi_knn(distances, indices, labels, perplexity=PERPLEXITY)),
            }
        objective = getattr(fitted, "objective_harmony", None)
        out[str(n)] = {
            "agreement": compare(cytearc_corrected, reference, neighbours),
            "lisi": lisi,
            "harmonypyObjectiveEntries": None if objective is None else len(list(objective)),
            "harmonypyClusterPasses": list(map(int, fitted.kmeans_rounds)),
            "cytearcParameters": cytearc_parameters,
        }
    return out


def parity_rows(cells: int) -> np.ndarray:
    """The rows of the parity checks: every cell, or PARITY_CELLS of them drawn with
    PARITY_SAMPLE_SEED, in order."""
    rows = np.arange(cells)
    if cells > PARITY_CELLS:
        rows = np.sort(
            np.random.default_rng(PARITY_SAMPLE_SEED).choice(rows, PARITY_CELLS, replace=False)
        )
    return rows


def parity_inputs() -> dict:
    """The PCA coordinates, batches, cell types and cell IDs of both datasets for the parity checks,
    from the same recipes as the seed-stability runs, computed by the installed CyteArc."""
    inputs = {}
    for name, recipe, key in (
        ("kang", kang_inputs, KANG_SELECTION),
        ("binvignat", binvignat_inputs, BINVIGNAT_SELECTION),
    ):
        ds, pca, batch_column, batches, labels = recipe()
        coordinates = np.asarray(ds.artifacts.load(pca)["data"][:], dtype=np.float64)
        cells = np.asarray(ds.cells.fetch("ids", key=key)).astype(str)
        if not coordinates.shape[0] == len(batches) == len(labels) == len(cells):
            raise ValueError(f"The {name} PCA rows differ from the cells of its selection")
        rows = parity_rows(len(cells))
        inputs[name] = {
            "coordinates": np.ascontiguousarray(coordinates[rows]),
            "batches": batches[rows],
            "labels": labels[rows],
            "cells": cells[rows],
            "batchColumn": batch_column,
            "pcaMetadata": dict(ds.artifacts.load(pca).attrs),
        }
    return inputs


def shared_assignment_state(coordinates: np.ndarray, batches: np.ndarray) -> dict:
    """The state of the assignment checks: 200 evenly spaced cells, six of them as unit centroids,
    and observed and expected batch masses (levels by clusters) from a softmax of the cells'
    similarities to the centroids."""
    from scipy.special import softmax

    rows = np.linspace(0, len(coordinates) - 1, 200, dtype=int)
    levels, codes = np.unique(batches[rows], return_inverse=True)
    unit = coordinates[rows] / np.linalg.norm(coordinates[rows], axis=1, keepdims=True)
    centers = unit[[0, 20, 40, 60, 80, 100]]
    initial = softmax(4 * unit @ centers.T, axis=1)
    return {
        "rows": rows,
        "levels": levels,
        "codes": codes,
        "unit": unit,
        "centroids": centers,
        "observed": np.vstack(
            [initial[codes == level].sum(axis=0) for level in range(len(levels))]
        ),
        "expected": np.multiply.outer(np.bincount(codes) / len(rows), initial.sum(axis=0)),
    }


def harmony2_core(coordinates: np.ndarray, batches: np.ndarray) -> tuple[dict, dict]:
    import harmonypy
    from cytearc.embeddings.harmony import fit_harmony
    from cytearc.embeddings.harmony.kernels import assign_cells
    from harmonypy._harmony_cpp import _assignment_probabilities
    from scipy.special import softmax

    state = shared_assignment_state(coordinates, batches)
    rows, levels, codes = state["rows"], state["levels"], state["codes"]
    values, batch = coordinates[rows], batches[rows]
    centers, observed, expected = state["centroids"], state["observed"], state["expected"]
    similarity = state["unit"] @ centers.T
    distances = 2 * (1 - similarity)
    sigma = np.full(6, 0.1)
    cases, archive = [], {"rows": rows, "centroids": centers, "O": observed, "E": expected}
    for diversity in (0.0, 2.0):
        theta = np.full(len(levels), diversity)
        penalty = -theta[:, None] * np.log1p(observed / (expected + 1))
        native = np.empty_like(similarity)
        assign_cells(
            similarity.copy(),
            np.arange(len(values)),
            np.ones(len(values)),
            2 / sigma,
            penalty,
            codes[None, :].astype(np.int32),
            native,
            np.empty(len(values)),
        )
        reference = _assignment_probabilities(
            np.ascontiguousarray(distances.T),
            sigma,
            np.ascontiguousarray(expected.T),
            np.ascontiguousarray(observed.T),
            theta,
            np.ascontiguousarray(codes[None, :], dtype=np.int64),
        ).T
        native_formula = softmax(-distances / sigma + penalty[codes], axis=1)
        reference_formula = softmax(
            -distances / sigma
            + theta[codes, None] * np.log((2 * expected + 1) / (observed + expected + 1))[codes],
            axis=1,
        )
        error = float(np.max(np.abs(native - reference)))
        native_equation = float(np.max(np.abs(native - native_formula)))
        reference_equation = float(np.max(np.abs(reference - reference_formula)))
        cases.append(
            {
                "theta": diversity,
                "cells": len(values),
                "clusters": 6,
                "nativeReferenceMaxAbsoluteError": error,
                "nativeEquationMaxAbsoluteError": native_equation,
                "referenceEquationMaxAbsoluteError": reference_equation,
                "tolerance": ASSIGNMENT_TOLERANCE,
                "passed": error <= ASSIGNMENT_TOLERANCE,
                # Each kernel reproduces the float64 evaluation of its own equation.
                "equationsReproduced": native_equation <= 1e-12
                and reference_equation <= ASSIGNMENT_TOLERANCE,
            }
        )
        archive[f"native_assignment_theta_{diversity:g}"] = native
        archive[f"reference_assignment_theta_{diversity:g}"] = reference
    meta = pd.DataFrame({"batch": batch})
    native_fit = fit_harmony(
        values.T.copy(),
        meta,
        n_clusters=1,
        max_iter_harmony=1,
        max_iter_cluster=1,
        nthreads=1,
    )
    reference_fit = harmonypy.run_harmony(
        values,
        meta,
        ["batch"],
        nclust=1,
        sigma=np.array([0.1]),
        max_iter_harmony=1,
        max_iter_kmeans=1,
        ncores=1,
        verbose=False,
    )
    native, reference = np.asarray(native_fit.corrected), np.asarray(reference_fit.Z_corr)
    if native.shape != values.shape:
        native = native.T
    if reference.shape != values.shape:
        reference = reference.T
    ridge_error = float(np.max(np.abs(native - reference)))
    archive.update({"native_single_cluster": native, "reference_single_cluster": reference})
    return {
        "stateDefinition": "The same 200 real PCA rows, six unit centroids and identical observed and expected batch masses; no optimization or random initialization in the assignment check",
        "assignmentCases": cases,
        "nativeAssignmentLogit": "-distance/sigma + theta*(log(E+1)-log(O+E+1))",
        "referenceAssignmentLogit": "-distance/sigma + theta*(log(2E+1)-log(O+E+1))",
        "singleClusterDynamicRidge": {
            "cells": len(values),
            "clusters": 1,
            "rounds": 1,
            "assignmentMaxAbsoluteError": float(np.max(np.abs(np.asarray(reference_fit.R) - 1))),
            "coordinateMaxAbsoluteError": ridge_error,
            "tolerance": 1e-5,
            "passed": ridge_error <= 1e-5,
        },
        "correctionStep": correction_step(coordinates, batches),
        "scope": "Compiled float32 reference assignment kernel, a one-cluster correction and a correction step on the reference's own 100-cluster assignments; full fits keep independent initialization and convergence",
    }, archive


def ridge_correction(
    coordinates: np.ndarray,
    codes: np.ndarray,
    assignments: np.ndarray,
    *,
    keep_pruned_cells: bool,
    alpha: float = HARMONY2_SETTINGS["ridge_alpha"],
    prune_threshold: float = HARMONY2_SETTINGS["prune_threshold"],
) -> np.ndarray:
    """One Harmony2 correction step for one batch column, solved densely in float64 as an
    independent check.

    In each cluster, the coordinates are regressed on an unpenalised intercept and the indicators of
    the batches that keep at least ``prune_threshold`` of their cells' assignment mass there, each
    penalised by ``alpha`` times its expected mass; every cell then subtracts the batch terms of its
    clusters, weighted by its assignments. The cells of a pruned batch either stay in the cluster's
    regression without a batch term (``keep_pruned_cells``) or leave it.
    """
    cells, clusters = assignments.shape
    levels = int(codes.max()) + 1
    design = np.eye(levels)[codes]
    level_cells = design.sum(axis=0)
    observed = assignments.T @ design
    expected = np.outer(assignments.sum(axis=0), level_cells / cells)
    coefficients = np.zeros((clusters, levels, coordinates.shape[1]))
    for cluster in range(clusters):
        active = (observed[cluster] > 0) & (observed[cluster] >= prune_threshold * level_cells)
        weights = assignments[:, cluster]
        if not keep_pruned_cells:
            weights = weights * active[codes]
        if not active.any():
            continue
        columns = np.column_stack([np.ones(cells), design[:, active]])
        penalties = np.concatenate([[0.0], alpha * expected[cluster, active]])
        system = columns.T @ (columns * weights[:, None]) + np.diag(penalties)
        solution = np.linalg.solve(system, columns.T @ (coordinates * weights[:, None]))
        coefficients[cluster, active] = solution[1:]
    corrected = coordinates.copy()
    for level in range(levels):
        members = codes == level
        corrected[members] -= assignments[members] @ coefficients[:, level, :]
    return corrected


def correction_step(coordinates: np.ndarray, batches: np.ndarray) -> dict:
    """One correction step on the reference's own assignments after one round of 100 clusters: the
    reference's corrected coordinates, CyteArc's correction of the same assignments, and the
    float64 equations with the cells of pruned batches kept in each cluster's regression
    (CyteArc) or left out of it."""
    from cytearc.embeddings.harmony.clustering import ClusteringWorkspace
    from cytearc.embeddings.harmony.correction import correct_embedding
    from cytearc.embeddings.harmony.design import encode_batch_design
    from cytearc.embeddings.harmony.parameters import (
        harmony_work_sizes,
        resolve_harmony_parameters,
    )

    reference, fitted = reference_harmony(coordinates, batches, 0, max_iter_harmony=1)
    assignments = np.asarray(fitted.R, dtype=np.float64)
    if assignments.shape[0] != coordinates.shape[0]:
        assignments = assignments.T
    clusters = assignments.shape[1]
    levels, codes = np.unique(batches, return_inverse=True)
    design = encode_batch_design(pd.DataFrame({"batch": batches}))
    parameters = resolve_harmony_parameters({**HARMONY2_SETTINGS, "n_clusters": clusters}, design)
    _, chunk = harmony_work_sizes(coordinates.shape[0], parameters.block_size)
    workspace = ClusteringWorkspace.allocate(chunk, clusters, coordinates.shape[1], design.n_levels)
    native = np.empty_like(coordinates)
    correct_embedding(
        np.ascontiguousarray(coordinates), assignments, native, design, parameters, workspace, 0
    )
    kept = ridge_correction(coordinates, codes, assignments, keep_pruned_cells=True)
    left_out = ridge_correction(coordinates, codes, assignments, keep_pruned_cells=False)
    observed = assignments.T @ np.eye(len(levels))[codes]
    level_cells = np.bincount(codes).astype(float)
    pruned = (observed > 0) & (observed < HARMONY2_SETTINGS["prune_threshold"] * level_cells)
    tolerance = CORRECTION_TOLERANCE * float(np.max(np.abs(coordinates)))
    native_equation = float(np.max(np.abs(native - kept)))
    reference_equation = float(np.max(np.abs(reference - left_out)))
    return {
        "cells": int(coordinates.shape[0]),
        "clusters": int(clusters),
        "prunedBatchClusterPairs": int(pruned.sum()),
        "nativeReferenceMaxAbsoluteError": float(np.max(np.abs(native - reference))),
        "nativeEquationMaxAbsoluteError": native_equation,
        "referenceEquationMaxAbsoluteError": reference_equation,
        "referenceKeptCellsEquationMaxAbsoluteError": float(np.max(np.abs(reference - kept))),
        "largestCorrection": float(np.max(np.abs(reference - coordinates))),
        "tolerance": tolerance,
        "passed": native_equation <= tolerance and reference_equation <= tolerance,
        "nativeEquation": "pruned batches keep their cells in each cluster's regression, without a "
        "batch term",
        "referenceEquation": "pruned batches leave each cluster's regression with their cells",
    }


def fit_scores(corrected: np.ndarray, batches: np.ndarray, labels: np.ndarray) -> tuple:
    """The 15 exact neighbours of every cell and the fit's iLISI and cLISI on them."""
    distances, indices = knn(corrected, K)
    return indices, {
        "iLISI": float(ilisi_knn(distances, indices, batches, perplexity=PERPLEXITY)),
        "cLISI": float(clisi_knn(distances, indices, labels, perplexity=PERPLEXITY)),
    }


def full_fit_equivalence(
    cross_overlaps: list[float],
    reference_overlaps: list[float],
    cytearc_ilisi: list[float],
    reference_ilisi: list[float],
) -> dict:
    """EQUIVALENCE_RULE on the 15-NN overlaps of every CyteArc and reference seed pair and of every
    pair of reference seeds, and on each fit's iLISI."""
    if not (cross_overlaps and reference_overlaps and cytearc_ilisi and reference_ilisi):
        raise ValueError(
            "The equivalence rule needs fits of both tools and at least two reference seeds"
        )
    cross = float(np.median(cross_overlaps))
    floor = float(np.min(reference_overlaps))
    shift = abs(float(np.median(cytearc_ilisi)) - float(np.median(reference_ilisi)))
    spread = float(np.max(reference_ilisi) - np.min(reference_ilisi))
    return {
        "rule": EQUIVALENCE_RULE,
        "crossOverlapMedian": cross,
        "referenceOverlapMin": floor,
        "overlapPassed": cross >= floor,
        "cytearcILISIMedian": float(np.median(cytearc_ilisi)),
        "referenceILISIMedian": float(np.median(reference_ilisi)),
        "iLISIMedianDifference": shift,
        "referenceILISIRange": spread,
        "iLISIPassed": shift <= spread,
        "passed": cross >= floor and shift <= spread,
    }


def arm_comparison(
    cytearc_fits: dict, reference_fits: dict, batches: np.ndarray, labels: np.ndarray
) -> dict:
    """Agreement of CyteArc's and the reference's full fits of one arm, seed by seed, with each
    tool's own spread across seeds and the equivalence rule."""
    seeds = sorted(cytearc_fits)
    if sorted(reference_fits) != seeds or len(seeds) < 2:
        raise ValueError("Both tools need fits for the same seeds, at least two")
    neighbours, scores = {}, {"cytearc": {}, "reference": {}}
    for tool, fits in (("cytearc", cytearc_fits), ("reference", reference_fits)):
        for seed, values in fits.items():
            neighbours[id(values)], scores[tool][seed] = fit_scores(values, batches, labels)

    def pairs(first: dict, second: dict, within: bool) -> list[dict]:
        return [
            compare(first[s], second[t], neighbours)
            for s in seeds
            for t in seeds
            if (s < t if within else True)
        ]

    cross = pairs(cytearc_fits, reference_fits, False)
    within_reference = pairs(reference_fits, reference_fits, True)
    return {
        "seeds": seeds,
        "scores": scores,
        "cytearcVersusReferenceSameSeed": summary(
            [compare(cytearc_fits[s], reference_fits[s], neighbours) for s in seeds]
        ),
        "cytearcVersusReferenceAnySeeds": summary(cross),
        "referenceBetweenSeeds": summary(within_reference),
        "cytearcBetweenSeeds": summary(pairs(cytearc_fits, cytearc_fits, True)),
        "equivalence": full_fit_equivalence(
            [row["knnOverlapK15"] for row in cross],
            [row["knnOverlapK15"] for row in within_reference],
            [scores["cytearc"][s]["iLISI"] for s in seeds],
            [scores["reference"][s]["iLISI"] for s in seeds],
        ),
    }


def cytearc_arms(coordinates: np.ndarray, batches: np.ndarray) -> tuple[dict, dict]:
    """CyteArc's fits of every parity arm and seed, and each fit's rounds and passes."""
    fits, progress = {}, {}
    for arm, changes in PARITY_ARMS.items():
        fits[arm], progress[arm] = {}, {}
        for seed in SEEDS:
            fits[arm][seed], parameters = cytearc_harmony(coordinates, batches, seed, **changes)
            progress[arm][seed] = {
                "rounds": parameters["rounds"],
                "clusterPasses": parameters["clusterPasses"],
                "converged": parameters["converged"],
            }
    return fits, progress


def evaluate_harmony2(output: Path) -> dict:
    """Harmony2 in CyteArc against harmonypy on the same PCA coordinates: single steps on shared
    states, and full fits of every arm and seed judged by EQUIVALENCE_RULE."""
    import harmonypy

    from experiments.runtime.common import provenance

    if harmonypy.__version__ != HARMONYPY_VERSION or cytearc.__version__ != "1.3.0":
        raise ValueError(
            f"This comparison requires CyteArc 1.3.0 and harmonypy {HARMONYPY_VERSION}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    archive, fits_archive, results = {}, {}, {}
    for name, data in parity_inputs().items():
        coordinates, batches, labels = data["coordinates"], data["batches"], data["labels"]
        archive.update(
            {
                f"{name}_coordinates": coordinates,
                f"{name}_batches": batches,
                f"{name}_labels": labels,
                f"{name}_cells": data["cells"],
            }
        )
        core, core_archive = harmony2_core(coordinates, batches)
        archive.update({f"{name}_core_{key}": value for key, value in core_archive.items()})
        native, progress = cytearc_arms(coordinates, batches)
        arms = {}
        for arm, changes in PARITY_ARMS.items():
            reference, reference_progress = {}, {}
            for seed in SEEDS:
                reference[seed], fitted = reference_harmony(coordinates, batches, seed, **changes)
                reference_progress[seed] = {
                    "rounds": len(fitted.kmeans_rounds),
                    "clusterPasses": list(map(int, fitted.kmeans_rounds)),
                    "objective": list(map(float, fitted.objective_harmony)),
                }
                fits_archive[f"{name}_{arm}_native_{seed}"] = native[arm][seed]
                fits_archive[f"{name}_{arm}_reference_{seed}"] = reference[seed]
            if arm == "ownStoppingRules":
                archive[f"{name}_native_0"] = native[arm][0]
                archive[f"{name}_reference_0"] = reference[0]
            arms[arm] = {
                "changes": changes,
                "cytearcProgress": progress[arm],
                "referenceProgress": reference_progress,
                **arm_comparison(native[arm], reference, batches, labels),
            }
        results[name] = {
            "cells": len(data["cells"]),
            "dimensions": coordinates.shape[1],
            "batchColumn": data["batchColumn"],
            "batches": pd.Series(batches).value_counts().to_dict(),
            "inputPcaMetadata": data["pcaMetadata"],
            "inputCoordinateSha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
            "sharedStateChecks": core,
            "fullFits": arms,
        }
    archive_path = output.with_suffix(".npz")
    np.savez_compressed(archive_path, **archive)
    # Every fit's corrected coordinates, kept with the run's raw outputs.
    np.savez_compressed(output.with_name(output.stem + "_fits.npz"), **fits_archive)
    result = {
        "provenance": provenance(harmonypyVersion=harmonypy.__version__),
        "endpoint": "Numerical agreement on identical PCA coordinates: single steps on shared states within float32 rounding, and full fits judged against the reference's own spread across seeds; no biological accuracy or speed endpoint",
        "inputDefinition": f"The seed-stability recipes' PCA coordinates computed by the installed CyteArc, subset to at most {PARITY_CELLS} cells per dataset with seed {PARITY_SAMPLE_SEED}",
        "settings": {
            "cytearc": {**HARMONY2_SETTINGS, "nThreads": 1},
            "harmonypy": harmonypy_arguments(HARMONY2_SETTINGS, PARITY_CELLS),
            "seeds": list(SEEDS),
            "arms": PARITY_ARMS,
        },
        "criteria": {
            "assignmentStep": f"passed: with the diversity penalty off, the two kernels agree within {ASSIGNMENT_TOLERANCE}; with it on they implement different numerators, so equationsReproduced asks that each reproduce its own equation",
            "correctionStep": f"Each tool reproduces the float64 equation of its pruning rule within {CORRECTION_TOLERANCE} of the largest absolute coordinate",
            "fullFits": EQUIVALENCE_RULE,
        },
        "datasets": results,
        "archive": {
            "file": archive_path.name,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        },
    }
    write_json(output.name, result, directory=output.parent)
    return result


def raw_rows(path: Path, columns: int) -> np.ndarray:
    """A float64 matrix that harmony2_reference.R wrote, one row per cell or cluster."""
    values = np.fromfile(path, dtype="<f8")
    if values.size % columns:
        raise ValueError(f"{path.name} holds {values.size} values, not rows of {columns}")
    return values.reshape(-1, columns)


def r_reference_input(inputs: dict, folder: Path) -> Path:
    """Write the coordinates, batches, assignment-step states, arms and seeds that
    harmony2_reference.R reads."""
    folder.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name, data in inputs.items():
        coordinates = data["coordinates"]
        path = folder / f"{name}_coordinates.f64"
        coordinates.astype("<f8").tofile(path)
        state = shared_assignment_state(coordinates, data["batches"])
        rows, observed, expected = state["rows"], state["observed"], state["expected"]
        datasets[name] = {
            "coordinates": str(path),
            "cells": int(coordinates.shape[0]),
            "dims": int(coordinates.shape[1]),
            "batches": data["batches"].tolist(),
            "levels": sorted(set(data["batches"].tolist())),
            "step": {
                "rows": (rows + 1).tolist(),
                "clusters": 6,
                "seed": 0,
                "thetas": [0.0, 2.0],
                "observedInjection": observed.T.tolist(),
                "expectedInjection": expected.T.tolist(),
            },
        }
    arms = {}
    for arm, changes in PARITY_ARMS.items():
        settings = {**HARMONY2_SETTINGS, **changes}
        arms[arm] = {
            "theta": settings["theta"],
            "maxIterHarmony": settings["max_iter_harmony"],
            "maxIterCluster": settings["max_iter_cluster"],
            "epsilonCluster": settings["epsilon_cluster"],
            "epsilonHarmony": settings["epsilon_harmony"],
            "disableStopping": settings["epsilon_harmony"] == 0
            and settings["epsilon_cluster"] == 0,
        }
    settings = {
        "sigma": HARMONY2_SETTINGS["sigma"],
        "alpha": HARMONY2_SETTINGS["ridge_alpha"],
        "tau": HARMONY2_SETTINGS["tau"],
        "blockSize": HARMONY2_SETTINGS["block_size"],
        "batchPropCutoff": HARMONY2_SETTINGS["prune_threshold"],
        "nclust": 100,
    }
    path = folder / "input.json"
    path.write_text(
        json.dumps({"datasets": datasets, "arms": arms, "seeds": list(SEEDS), "settings": settings})
    )
    return path


def r_assignment_step(data: dict, step: dict, outputs: Path) -> dict:
    """R harmony's assignments after one pass against the injected O and E, against each numerator's
    equation from R's own first assignments, and against CyteArc's kernel on the same centroids
    and state."""
    from cytearc.embeddings.harmony.kernels import assign_cells
    from scipy.special import softmax

    state = shared_assignment_state(data["coordinates"], data["batches"])
    unit, codes = state["unit"], state["codes"]
    observed, expected = state["observed"], state["expected"]
    clusters, dims, theta = step["clusters"], step["dims"], float(step["theta"])
    first = raw_rows(outputs / step["firstAssignments"], clusters)
    after = raw_rows(outputs / step["assignments"], clusters)
    centroids = {
        "firstCentroids": raw_rows(outputs / step["firstCentroids"], dims).T,
        "centroidsAfterPass": raw_rows(outputs / step["centroids"], dims).T,
    }
    numerators = {"paper": expected + 1, "rCode": 2 * expected + 1}
    result = {"theta": theta}
    for numerator, value in numerators.items():
        factor = (value / (observed + expected + 1)) ** theta
        implied = first * factor[codes]
        result[f"versus{numerator[0].upper()}{numerator[1:]}EquationFromFirstAssignments"] = float(
            np.max(np.abs(after - implied / implied.sum(axis=1, keepdims=True)))
        )
    for label, matrix in centroids.items():
        similarity = unit @ matrix
        native = np.empty_like(similarity)
        assign_cells(
            similarity.copy(),
            np.arange(len(unit)),
            np.ones(len(unit)),
            np.full(clusters, 2 / HARMONY2_SETTINGS["sigma"]),
            -theta * np.log1p(observed / (expected + 1)),
            codes[None, :].astype(np.int32),
            native,
            np.empty(len(unit)),
        )
        code_form = softmax(
            -2 * (1 - similarity) / HARMONY2_SETTINGS["sigma"]
            + theta * np.log((2 * expected + 1) / (observed + expected + 1))[codes],
            axis=1,
        )
        result[f"cytearcVersusR_{label}"] = float(np.max(np.abs(native - after)))
        result[f"rCodeEquationVersusR_{label}"] = float(np.max(np.abs(code_form - after)))
    return result


def evaluate_r_reference(output: Path) -> dict:
    """Harmony2 in CyteArc against R harmony on the parity inputs: one assignment step on an
    injected state, and full fits of every arm and seed judged by EQUIVALENCE_RULE."""
    from experiments.runtime.common import provenance

    if cytearc.__version__ != "1.3.0":
        raise ValueError("This comparison requires CyteArc 1.3.0")
    inputs = parity_inputs()
    folder = output.parent / "r_reference"
    outputs = folder / "outputs"
    subprocess.run(
        ["Rscript", str(R_REFERENCE), str(r_reference_input(inputs, folder)), str(outputs)],
        check=True,
    )
    r_summary = json.loads((outputs / "summary.json").read_text())
    results, archive = {}, {}
    for name, data in inputs.items():
        r_dataset = r_summary["datasets"][name]
        steps = [r_assignment_step(data, step, outputs) for step in r_dataset["steps"]]
        native, progress = cytearc_arms(data["coordinates"], data["batches"])
        arms = {}
        for arm in PARITY_ARMS:
            reference = {
                seed: raw_rows(
                    outputs / r_dataset["fits"][arm][str(seed)]["file"],
                    data["coordinates"].shape[1],
                )
                for seed in SEEDS
            }
            if arm == "ownStoppingRules":
                archive[f"{name}_native_0"] = native[arm][0]
                archive[f"{name}_reference_0"] = reference[0]
            arms[arm] = {
                "changes": PARITY_ARMS[arm],
                "cytearcProgress": progress[arm],
                "referenceProgress": {
                    seed: {k: v for k, v in fit.items() if k != "file"}
                    for seed, fit in r_dataset["fits"][arm].items()
                },
                **arm_comparison(native[arm], reference, data["batches"], data["labels"]),
            }
        results[name] = {
            "cells": len(data["cells"]),
            "inputCoordinateSha256": hashlib.sha256(data["coordinates"].tobytes()).hexdigest(),
            "assignmentSteps": steps,
            "fullFits": arms,
        }
    archive_path = output.with_suffix(".npz")
    np.savez_compressed(archive_path, **archive)
    result = {
        "provenance": provenance(
            rHarmonyVersion=r_summary["version"], r=r_summary["r"], rBlas=r_summary["blas"]
        ),
        "reference": {
            "package": "harmony (R)",
            "version": r_summary["version"],
            "lapack": r_summary["lapack"],
        },
        "inputDefinition": f"The parity inputs of harmony2.json: the seed-stability recipes' PCA coordinates, subset to at most {PARITY_CELLS} cells per dataset with seed {PARITY_SAMPLE_SEED}",
        "settings": {
            "cytearc": {**HARMONY2_SETTINGS, "nThreads": 1},
            "seeds": list(SEEDS),
            "arms": PARITY_ARMS,
        },
        "criteria": {"fullFits": EQUIVALENCE_RULE},
        "datasets": results,
        "archive": {
            "file": archive_path.name,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        },
    }
    write_json(output.name, result, directory=output.parent)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--harmony2", action="store_true")
    modes.add_argument("--r-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.harmony2:
        configure_cytearc()
        evaluate_harmony2(args.output or DATA / "harmony2.json")
        return
    if args.r_reference:
        configure_cytearc()
        evaluate_r_reference(args.output or DATA / "harmony2_r_reference.json")
        return
    import harmonypy

    configure_cytearc()
    results = {
        "description": "Harmony2 in CyteArc and in harmonypy across seeds 0 to 9 on Binvignat and Kang PBMC inputs: every pair of corrections compared, and each correction's LISI.",
        "parameters": {
            "seeds": list(SEEDS),
            "dims": DIMS,
            "k": K,
            "perplexity": PERPLEXITY,
            "harmony2Settings": HARMONY2_SETTINGS,
            "iterationLimitPasses": FIXED_PASSES,
            "harmonypyVersion": getattr(harmonypy, "__version__", "unknown"),
        },
        "binvignat": run_dataset(*binvignat_inputs()),
        "kang": run_dataset(*kang_inputs()),
    }
    from experiments.runtime.common import provenance

    write_json(
        "harmony_seed_stability.json",
        {
            "comparisons": [
                {
                    "referenceImplementation": "harmonypy",
                    "referenceVersion": results["parameters"]["harmonypyVersion"],
                    "measurement": {"provenance": provenance(), **results},
                }
            ]
        },
        directory=DATA,
    )
    for name in ("binvignat", "kang"):
        r = results[name]
        print(
            name,
            "harmonypy iLISI by seed:",
            {s: round(v["iLISI"], 3) for s, v in r["lisi"]["harmonypy"].items()},
        )
        print(
            name,
            "CyteArc iLISI by seed:",
            {s: round(v["iLISI"], 3) for s, v in r["lisi"]["cytearc"].items()},
        )
        for key in (
            "cytearcVersusHarmonypySameSeed",
            "harmonypyBetweenSeeds",
            "cytearcBetweenSeeds",
        ):
            print(name, key, {m: round(v["median"], 3) for m, v in r[key].items()})
        for n, f in r["iterationLimits"].items():
            print(
                name,
                f"Up to {n} iterations: CyteArc iLISI {f['lisi']['cytearc']['iLISI']:.3f}, harmonypy {f['lisi']['harmonypy']['iLISI']:.3f};",
                {m: round(v, 4) for m, v in f["agreement"].items()},
            )


if __name__ == "__main__":
    main()
