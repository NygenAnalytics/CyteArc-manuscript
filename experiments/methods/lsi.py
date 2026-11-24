import hashlib
import os
import shutil
import time
from pathlib import Path

import cytearc
import numpy as np
from scipy.linalg import subspace_angles, svd
from threadpoolctl import threadpool_limits

from experiments.runtime.common import configure_cytearc, provenance, scrub_private, write_json

SETTINGS = {
    "dataset": "swanson_7K_pbmc_teaseq",
    "cells": 400,
    "cellSeed": 4467,
    "features": 25000,
    "dimensions": 30,
    "skipFirst": True,
    "seed": 4466,
    "batchSizes": [64, 400],
    "maxSubspaceAngleDegrees": 0.05,
    "maxProjectionAbsoluteError": 1e-6,
}


def run(payload: dict) -> dict:
    configure_cytearc()
    work = Path(payload["workDirectory"])
    work.mkdir(parents=True, exist_ok=True)
    settings = {
        **SETTINGS,
        "cellSelection": payload.get("cellSelection", "sampled"),
        "batchSizes": payload.get("batchSizes", SETTINGS["batchSizes"]),
        "configurations": payload.get("configurations", [{"iterations": 20, "oversamples": 100}]),
    }
    if settings["cellSelection"] not in ("sampled", "active"):
        raise ValueError("cellSelection must be sampled or active")
    if "sourcePath" in payload:
        source = Path(payload["sourcePath"])
    else:
        source = (
            Path(
                cytearc.datasets.connect("cytearc_docs").download_dataset(
                    SETTINGS["dataset"],
                    destination=str(work / "downloads"),
                    zarr=True,
                )
            )
            / "data.zarr"
        )
    hashes = {}
    if settings["cellSelection"] == "active":
        archive = source.parent / "data.zarr.tar.gz"
        with archive.open("rb") as handle:
            hashes["sourceArchive"] = hashlib.file_digest(handle, "sha256").hexdigest()
    threads = int(payload.get("workers", 2))
    memory_budget = payload.get("memoryBudget", "2G")
    original = cytearc.DataStore(
        str(source), default_assay="RNA", zarr_mode="r", nthreads=threads, mem_budget=memory_budget
    )
    eligible = np.flatnonzero(original.cells.fetch_all("I"))
    if settings["cellSelection"] == "active":
        chosen = eligible
        path = work / "analysis.zarr"
        shutil.copytree(source, path)
        ds = cytearc.DataStore(
            str(path), default_assay="RNA", nthreads=threads, mem_budget=memory_budget
        )
        mask = np.zeros(ds.cells.N, dtype=bool)
        mask[chosen] = True
        ds.cells.insert("multimodal_cells", mask, overwrite=True)
        cells = ds.snapshot_cell_selection("multimodal_cells")
    else:
        chosen = np.sort(
            np.random.default_rng(settings["cellSeed"]).choice(
                eligible,
                settings["cells"],
                replace=False,
            )
        )
        path = work / "atac.zarr"
        cytearc.SubsetZarr(
            str(path), [original.ATAC], cell_idx=chosen, mem_budget=memory_budget, nthreads=threads
        ).dump()
        ds = cytearc.DataStore(
            str(path), default_assay="ATAC", nthreads=threads, mem_budget=memory_budget
        )
        cells = ds.snapshot_cell_selection("I")
    expected_ids = original.cells.fetch_all("ids")[chosen].astype(str)
    selected_mask = np.asarray(ds.artifacts.load(cells)["values"][:], dtype=bool)
    observed_ids = ds.cells.fetch_all("ids")[selected_mask].astype(str)
    if not np.array_equal(observed_ids, expected_ids):
        raise RuntimeError("LSI input changed the selected cell identities")
    settings["cells"] = len(chosen)
    hashes["selectedRows"] = hashlib.sha256(np.ascontiguousarray(chosen).tobytes()).hexdigest()
    hashes["cellsFile"] = hashlib.sha256(("\n".join(expected_ids) + "\n").encode()).hexdigest()
    features = ds.features.prevalent_peaks(
        cells, from_assay="ATAC", top_n=settings["features"], invalidate_cache=True
    )
    feature_mask = np.asarray(ds.artifacts.load(features)["values"][:], dtype=bool)
    feature_ids = ds.ATAC.feats.fetch_all("ids")[feature_mask].astype(str)
    hashes["featuresFile"] = hashlib.sha256(("\n".join(feature_ids) + "\n").encode()).hexdigest()
    if settings["cellSelection"] == "active":
        expected_hashes = payload["expectedHashes"]
        if set(expected_hashes) != set(hashes):
            raise ValueError("Active-selection validation requires all four input hashes")
        for name, expected in expected_hashes.items():
            if hashes[name] != expected:
                raise RuntimeError(f"LSI {name} does not match the completed multimodal workflow")
    normalized = ds.features.normalize(cells, features, invalidate_cache=True)
    matrix = np.asarray(ds.artifacts.load(normalized)["data"][:], dtype=np.float64)
    if matrix.shape != (settings["cells"], settings["features"]):
        raise RuntimeError(f"Unexpected LSI matrix shape: {matrix.shape}")
    result = {
        "status": "running",
        "experiment": "lsi",
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
        "settings": settings,
        "workers": threads,
        "memoryBudget": memory_budget,
        "source": str(source),
        "inputHashes": hashes,
        "cellIdsSha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(),
        "normalizedSha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "normalization": ds.artifacts.inspect(normalized).parameters,
        "shape": list(matrix.shape),
        "results": [],
        "reference": "One dense float64 uncentered SVD of the same normalized ATAC matrix, first component omitted",
        "performanceClaim": False,
    }

    def persist():
        path = write_json(
            "lsi.json", scrub_private(result), directory=Path(payload["outputDirectory"])
        )
        if payload.get("recordUri"):
            from experiments.runtime.harness import R2

            R2(payload["r2EndpointUrl"]).upload(path, payload["recordUri"])

    persist()
    try:
        started = time.perf_counter()
        with threadpool_limits(limits=threads):
            u, singular_values, vt = svd(matrix, full_matrices=False)
        reference = vt[1 : settings["dimensions"] + 1].T.copy()
        del u, vt
        result["denseSvdSeconds"] = time.perf_counter() - started
        result["singularValues"] = singular_values[:32].tolist()
        persist()
        for configuration in settings["configurations"]:
            for batch_size in settings["batchSizes"]:
                started = time.perf_counter()
                ref = ds.reduction.lsi(
                    normalized,
                    dims=settings["dimensions"],
                    skip_first=settings["skipFirst"],
                    solver="streaming",
                    n_iter=configuration["iterations"],
                    n_oversamples=configuration["oversamples"],
                    batch_size=batch_size,
                    rand_state=settings["seed"],
                    invalidate_cache=True,
                )
                elapsed = time.perf_counter() - started
                artifact = ds.artifacts.load(ref)
                loadings = np.asarray(artifact["loadings"][:])
                coordinates = np.asarray(artifact["data"][:])
                angles = np.degrees(subspace_angles(loadings, reference))
                projection_error = float(np.max(np.abs(matrix @ loadings - coordinates)))
                status = ds.artifacts.inspect(ref)
                row = {
                    **configuration,
                    "batchSize": batch_size,
                    "maxSubspaceAngleDegrees": float(angles.max()),
                    "medianSubspaceAngleDegrees": float(np.median(angles)),
                    "maxProjectionAbsoluteError": projection_error,
                    "seconds": elapsed,
                    "parameters": status.parameters,
                    "executionOptions": status.execution_options,
                }
                row["meetsTolerance"] = (
                    row["maxSubspaceAngleDegrees"] <= settings["maxSubspaceAngleDegrees"]
                    and projection_error <= settings["maxProjectionAbsoluteError"]
                )
                result["results"].append(row)
                persist()
        result["status"] = "ok"
        persist()
    except Exception as error:
        result.update(status="failed", errorType=type(error).__name__, error=str(error))
        persist()
        raise
    return result
