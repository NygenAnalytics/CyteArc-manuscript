from pathlib import Path

GIB = 1024**3
LABEL_COLUMNS = ("cell_type", "development_stage")
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
STAGE_ALIASES = {
    "filterCells": "cells",
    "markHvgs": "features",
    "runNormalization": "normalized",
    "runPca": "pca",
    "buildEmbeddingInitialization": "initialization",
    "buildAnnIndex": "ann_index",
    "queryNeighbors": "neighbors",
    "buildConnectivityMap": "graph",
    "runUmap": "umap",
    "runLeiden": "clusters",
    "findMarkers": "markers",
}


def _prefix_bytes(prefix_uri: str) -> tuple[int, int]:
    import obstore

    from experiments.runtime.object_io import open_r2_object

    store, key = open_r2_object(prefix_uri.rstrip("/") + "/x")
    prefix = key[:-1]
    total = 0
    count = 0
    for page in obstore.list(store, prefix=prefix, chunk_size=1000):
        for item in page:
            total += int(item["size"])
            count += 1
    return (total, count)


def _open_reference(payload: dict, probe):
    import cytearc

    from experiments.runtime.object_io import storage_options
    from experiments.runtime.recording_store import wrap_store_probe

    uri = payload["referenceStoreUri"]
    options = storage_options(uri)
    location = wrap_store_probe(uri, options, probe)
    return cytearc.DataStore(
        location,
        default_assay="RNA",
        nthreads=int(payload["workers"]),
        mem_budget=int(payload["budgetBytes"]),
        zarr_mode="r+",
        zarrProfile="cloud",
        storage_options=options,
    )


def _stage_records(payload: dict) -> dict:
    from experiments.runtime.object_io import get_json

    return {stage: get_json(f"{payload['resultsPrefix']}/{stage}.json") for stage in STAGE_ALIASES}


DISCOVERY = {
    "cells": ("cell_selection", "auto_filter_cells", {}),
    "features": ("feature_selection", "select_hvgs", {}),
    "normalized": ("normalized", "run_normalization", {}),
    "pca": ("reduction", "run_pca", {}),
    "initialization": ("embedding_initialization", "build_embedding_initialization", {}),
    "ann_index": ("ann_index", "build_ann_index", {}),
    "neighbors": ("neighbors", "query_neighbors", {}),
    "graph": ("connectivity_map", "build_connectivity_map", {}),
    "umap": ("embedding", "run_umap", {}),
    "clusters": ("cluster_labels", "run_leiden_clustering", {"backend": "igraph"}),
    "markers": ("marker_table", "run_marker_search", {}),
}


def _resolve_refs(ds, records: dict) -> dict:
    from cytearc.storage import ArtifactRef

    refs = {}
    for stage, alias in STAGE_ALIASES.items():
        raw = (records.get(stage) or {}).get("details", {}).get("artifact")
        if raw:
            refs[alias] = ArtifactRef.from_dict(raw)
            continue
        kind, operation, parameters = DISCOVERY[alias]
        matches = ds.artifacts.list(
            from_assay="RNA",
            kind=kind,
            operation=operation,
            parameters=parameters or None,
            complete_only=True,
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"{alias}: expected one {kind} artifact from {operation}, found {len(matches)}"
            )
        refs[alias] = matches[0]
    return refs


def write_rows_h5ad(
    sourcePath: Path, destinationPath: Path, rows, *, windowBytes: int = 256 * 1024**2
) -> int:
    import os
    import uuid

    import h5py
    import numpy as np

    from experiments.runtime.h5ad import (
        _create_dataframe_group,
        _create_numeric_dataset,
        _create_string_dataset,
        _load_string_column,
        _validate_selected_rows,
    )

    temporary = destinationPath.with_name(f".{destinationPath.name}.{uuid.uuid4().hex}.part")
    with h5py.File(sourcePath, "r") as source, h5py.File(temporary, "w") as destination:
        matrix = source["X"]
        n_rows, n_columns = (int(v) for v in matrix.attrs["shape"])
        rows = np.sort(_validate_selected_rows(rows, n_rows))
        indptr = matrix["indptr"][:]
        starts, ends = (indptr[rows], indptr[rows + 1])
        counts = ends - starts
        nnz = int(counts.sum(dtype=np.int64))
        data, indices = (matrix["data"], matrix["indices"])
        destination.attrs["encoding-type"] = "anndata"
        destination.attrs["encoding-version"] = "0.1.0"
        out = destination.create_group("X")
        out.attrs["encoding-type"] = "csr_matrix"
        out.attrs["encoding-version"] = "0.1.0"
        out.attrs["shape"] = np.asarray([len(rows), n_columns], dtype=np.int64)
        out_data = _create_numeric_dataset(out, "data", nnz, data.dtype)
        out_indices = _create_numeric_dataset(out, "indices", nnz, indices.dtype)
        out_indptr = _create_numeric_dataset(out, "indptr", len(rows) + 1, indptr.dtype)
        out_indptr[:] = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(
            indptr.dtype
        )
        budget = max(1, windowBytes // (data.dtype.itemsize + indices.dtype.itemsize))
        position, i = (0, 0)
        while i < len(rows):
            j = i + 1
            while j < len(rows) and int(ends[j] - starts[i]) <= budget:
                j += 1
            low, high = (int(starts[i]), int(ends[j - 1]))
            window_data, window_indices = (data[low:high], indices[low:high])
            keep = np.concatenate(
                [np.arange(a - low, b - low) for a, b in zip(starts[i:j], ends[i:j])]
            )
            out_data[position : position + keep.size] = window_data[keep]
            out_indices[position : position + keep.size] = window_indices[keep]
            position += keep.size
            i = j
        if position != nnz:
            raise ValueError(f"wrote {position} values, expected {nnz}")
        obs = _create_dataframe_group(destination, "obs", ())
        _create_string_dataset(obs, "_index", len(rows))[:] = source["obs/_index"][:][rows]
        var = _create_dataframe_group(destination, "var", ("feature_name",))
        _create_string_dataset(var, "_index", n_columns)[:] = source["var/_index"][:]
        _create_string_dataset(var, "feature_name", n_columns)[:] = _load_string_column(
            source, "var/feature_name", expectedLength=n_columns
        )
    os.replace(temporary, destinationPath)
    return nnz


def mapping_job(payload: dict) -> dict:
    import os
    import time

    import numpy as np
    import pandas as pd

    os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    import cytearc
    from cytearc.storage.refs import artifact_path

    from experiments.runtime.metrics import ResourceSampler
    from experiments.runtime.object_io import download_file, put_json, upload_file
    from experiments.runtime.recording_store import StoreProbe

    cytearc.configure_output(level="WARNING", progress=False)
    work = Path("/tmp/manuscript-mapping")
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    download_file(payload["inputsUri"], work / "inputs.npz")
    inputs = np.load(work / "inputs.npz", allow_pickle=False)
    positions = np.asarray(inputs["positions"], dtype=np.int64)
    download_file(payload["sampleUri"], work / "sample.h5ad")
    timings["downloadSeconds"] = time.perf_counter() - started
    started = time.perf_counter()
    write_rows_h5ad(work / "sample.h5ad", work / "query.h5ad", positions)
    reader = cytearc.H5adReader(
        str(work / "query.h5ad"),
        matrix_key="X",
        cell_attrs_key="obs",
        cell_ids_key="_index",
        feature_attrs_key="var",
        feature_ids_key="_index",
        feature_name_key="feature_name",
    )
    cytearc.H5adToZarr(
        reader,
        str(work / "query.zarr"),
        assay_name="RNA",
        nthreads=int(payload["workers"]),
        mem_budget=int(payload["budgetBytes"]),
        profile="fast_local",
    ).dump()
    reader.h5.close()
    ds_query = cytearc.DataStore(
        str(work / "query.zarr"),
        assay_types={"RNA": "RNA"},
        default_assay="RNA",
        min_features_per_cell=10,
        nthreads=int(payload["workers"]),
        mem_budget=int(payload["budgetBytes"]),
    )
    for column in LABEL_COLUMNS:
        categories = np.asarray(inputs[f"categories_{column}"]).astype(str)
        ds_query.cells.insert(
            column_name=column,
            values=categories[np.asarray(inputs[f"query_{column}"])],
            overwrite=True,
        )
    timings["queryStoreSeconds"] = time.perf_counter() - started
    probe = StoreProbe()
    ds_reference = _open_reference(payload, probe)
    for column in LABEL_COLUMNS:
        categories = np.asarray(inputs[f"categories_{column}"]).astype(str)
        ds_reference.cells.insert(
            column_name=column,
            values=categories[np.asarray(inputs[f"reference_{column}"])],
            overwrite=True,
        )
    neighbors = _resolve_refs(ds_reference, _stage_records(payload))["neighbors"]
    probe.reset()
    started = time.perf_counter()
    reference_ref = ds_reference.mapping.build_reference(neighbors)
    timings["buildReferenceSeconds"] = time.perf_counter() - started
    build_ops = probe.to_json()
    probe.reset()
    started = time.perf_counter()
    reference = ds_reference.mapping.load_reference(reference_ref)
    timings["loadReferenceSeconds"] = time.perf_counter() - started
    load_ops = probe.to_json()
    selection = ds_query.snapshot_cell_selection("I")
    sampler = ResourceSampler(sampleIntervalSeconds=0.25, resetCgroupPeak=False)
    sampler.start()
    probe.reset()
    started = time.perf_counter()
    mapping_ref = ds_query.mapping.run(
        reference, selection, query_assay="RNA", save_k=11, missing_feature_policy="reference_mean"
    )
    timings["runMappingSeconds"] = time.perf_counter() - started
    map_ops = probe.to_json()
    measurement = sampler.stop()
    mapping = ds_query.mapping.load_result(mapping_ref, reference=reference, load_arrays=False)
    probe.reset()
    started = time.perf_counter()
    threshold_rows = []
    per_cell: dict[str, np.ndarray] = {}
    for column in LABEL_COLUMNS:
        truth = np.asarray(ds_query.cells.fetch(column, key="I")).astype(str)
        per_cell[f"truth_{column}"] = truth
        for threshold in THRESHOLDS:
            transferred = ds_query.mapping.load_label_transfer(
                ds_query.mapping.transfer_labels(
                    mapping_ref,
                    reference=reference,
                    reference_labels=column,
                    threshold_fraction=threshold,
                )
            ).labels.fillna("NA")
            values = transferred.to_numpy().astype(str)
            per_cell[f"transferred_{column}_{threshold}"] = values
            accepted = (values != "NA") & pd.notna(transferred.to_numpy())
            threshold_rows.append(
                {
                    "labelColumn": column,
                    "threshold": threshold,
                    "queryCells": int(values.size),
                    "accepted": int(accepted.sum()),
                    "abstainedFraction": float((~accepted).mean()),
                    "agreementAccepted": float(np.mean(values[accepted] == truth[accepted]))
                    if accepted.any()
                    else None,
                    "uniqueLabelsInQuery": int(len(set(truth))),
                }
            )
    timings["labelTransferSeconds"] = time.perf_counter() - started
    transfer_ops = probe.to_json()
    per_cell_uri = payload["outputUri"].replace(".json", "_cells.npz")
    np.savez_compressed(work / "cells.npz", **per_cell)
    upload_file(work / "cells.npz", per_cell_uri)
    uri = payload["referenceStoreUri"]
    reference_bytes, reference_objects = _prefix_bytes(f"{uri}/{artifact_path(reference_ref)}")
    index_bytes, _ = _prefix_bytes(f"{uri}/{artifact_path(reference.ann_index)}")
    reduction_bytes, _ = _prefix_bytes(f"{uri}/{artifact_path(reference.reduction)}")
    counts_bytes, _ = _prefix_bytes(f"{uri}/RNA/counts")
    counts_t_bytes, _ = _prefix_bytes(f"{uri}/RNA/countsT")
    result = {
        "experiment": "mapping",
        "referenceTag": payload["referenceTag"],
        "referenceStoreUri": uri,
        "container": {
            "cpu": 8,
            "memoryMb": 32768,
            "workers": payload["workers"],
            "budgetBytes": payload["budgetBytes"],
        },
        "reference": {
            "cellsSelected": int(reference.selected_cell_count),
            "cellsTotal": int(ds_reference.cells.N),
            "genesTotal": int(ds_reference.RNA.feats.N),
            "panelGenes": int(reference.feature_ids.size),
            "dims": int(reference.model.loadings.shape[1]),
            "method": reference.method,
            "referenceArtifactBytes": reference_bytes,
            "referenceArtifactObjects": reference_objects,
            "annIndexBytes": index_bytes,
            "reductionArtifactBytes": reduction_bytes,
            "countsBytes": counts_bytes,
            "countsTBytes": counts_t_bytes,
            "sampleH5adBytes": payload.get("referenceH5adBytes"),
        },
        "query": {
            "cells": int(threshold_rows[0]["queryCells"]),
            "requestedCells": int(positions.size),
            "sampleUri": payload["sampleUri"],
            "diagnostics": {
                k: v if isinstance(v, (int, float, str)) else str(v)
                for k, v in dict(mapping.diagnostics).items()
            },
            "correctionMethod": mapping.correction_method,
        },
        "seconds": timings,
        "peakRssBytes": measurement.processTreeRssPeakBytes,
        "peakCgroupBytes": measurement.cgroupMemoryCurrentPeakBytes,
        "referenceStoreOperations": {
            "buildReference": build_ops,
            "loadReference": load_ops,
            "runMapping": map_ops,
            "labelTransfer": transfer_ops,
        },
        "thresholds": threshold_rows,
    }
    put_json(payload["outputUri"], result)
    return {
        "outputUri": payload["outputUri"],
        "runMappingSeconds": timings["runMappingSeconds"],
        "mappingReadBytes": map_ops.get("readTransferredBytes"),
    }


def parallel_settings(payload: dict, ann_parameters: dict, umap_parameters: dict) -> dict:
    """The parallel neighbor-index and UMAP settings the payload sets, which must be those the stored artifacts
    record (CyteArc records ann_parallel and parallel in each artifact's identity), so that the fork's UMAP runs as
    the stored one did."""
    settings = {}
    for key, recorded, parameters in (
        ("annParallel", "ann_parallel", ann_parameters),
        ("umapParallel", "parallel", umap_parameters),
    ):
        value = payload.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"the payload must set {key} to true or false, not {value!r}")
        if parameters.get(recorded) is not value:
            raise ValueError(
                f"the stored artifact records {recorded}={parameters.get(recorded)!r}, "
                f"but the payload sets {key}={value}"
            )
        settings[key] = value
    return settings


def recorded_threads(status) -> dict:
    """The parallel flag and thread counts an artifact records: UMAP's parallel, nthreads and layout_threads, a
    neighbor index's ann_parallel and nthreads, a neighbor query's nthreads."""
    options = status.execution_options or {}
    parameters = status.parameters or {}
    return {
        "parallel": parameters.get("parallel", parameters.get("ann_parallel")),
        "nthreads": options.get("nthreads"),
        "layoutThreads": options.get("layout_threads"),
    }


def thread_settings(stored, fork) -> dict:
    """The thread settings of a stored artifact and of the fork's artifact in its place, and whether they match."""
    settings = {"stored": recorded_threads(stored), "fork": recorded_threads(fork)}
    settings["same"] = settings["stored"] == settings["fork"]
    return settings


def container(payload: dict, measurement) -> dict:
    """The container the fork ran in: the CPUs and memory the payload states, checked against the cgroup's memory
    limit where the container exposes one."""
    import os

    import numba

    memory_mb = int(payload["memoryMb"])
    limit = measurement.memoryMaxBytes
    if isinstance(limit, int) and limit != memory_mb * 1024**2:
        raise ValueError(
            f"the container's memory limit is {limit} bytes, not the {memory_mb} MiB the payload states"
        )
    return {
        "cpu": int(payload["cpu"]),
        "memoryMb": memory_mb,
        "workers": int(payload["workers"]),
        "budgetBytes": int(payload["budgetBytes"]),
        "memoryMaxBytes": limit,
        "cpuQuotaCores": measurement.cpuQuotaCores,
        "visibleCpus": len(os.sched_getaffinity(0)),
        "numbaThreads": int(numba.config.NUMBA_NUM_THREADS),
    }


def rerun_job(payload: dict) -> dict:
    import os
    import time

    import numpy as np

    os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    import cytearc
    from cytearc.storage import ArtifactRef

    from experiments.runtime.metrics import ResourceSampler
    from experiments.runtime.object_io import put_json
    from experiments.runtime.recording_store import StoreProbe

    cytearc.configure_output(level="WARNING", progress=False)
    limits = ResourceSampler(sampleIntervalSeconds=0.25, resetCgroupPeak=False)
    limits.start()
    resources = container(payload, limits.stop())
    records = _stage_records(payload)
    full_run = {stage: records[stage]["seconds"] for stage in STAGE_ALIASES}
    full_peaks = {stage: records[stage].get("peakRssBytes") for stage in STAGE_ALIASES}
    probe = StoreProbe()
    ds = _open_reference(payload, probe)
    refs = _resolve_refs(ds, records)
    workers = int(payload["workers"])
    before = {ref.artifact_id for ref in ds.artifacts.list(from_assay="RNA")}

    def timed(name: str, fn):
        sampler = ResourceSampler(sampleIntervalSeconds=0.25, resetCgroupPeak=False)
        sampler.start()
        probe.reset()
        started = time.perf_counter()
        value = fn()
        seconds = time.perf_counter() - started
        measurement = sampler.stop()
        stages[name] = {
            "seconds": seconds,
            "peakRssBytes": measurement.processTreeRssPeakBytes,
            "storeOperations": probe.to_json(),
        }
        return value

    stages: dict[str, dict] = {}
    pca_status = ds.artifacts.inspect(refs["pca"])
    ann_status = ds.artifacts.inspect(refs["ann_index"])
    umap_status = ds.artifacts.inspect(refs["umap"])
    leiden_status = ds.artifacts.inspect(refs["clusters"])
    reuse = {
        "normalization": timed(
            "reuse.normalization",
            lambda: ds.features.normalize(refs["cells"], refs["features"]) == refs["normalized"],
        ),
        "pca": timed(
            "reuse.pca",
            lambda: (
                ds.reduction.pca(
                    refs["normalized"],
                    dims=int(pca_status.parameters["dims"]),
                    show_elbow_plot=False,
                )
                == refs["pca"]
            ),
        ),
    }
    parallel = parallel_settings(payload, ann_status.parameters, umap_status.parameters)
    ann_params = {
        k: v
        for k, v in ann_status.parameters.items()
        if k in ("ann_metric", "ann_efc", "ann_ef", "ann_m", "rand_state")
    }
    reuse["annIndex"] = timed(
        "reuse.annIndex",
        lambda: (
            ds.graph.ann_index(refs["pca"], ann_parallel=parallel["annParallel"], **ann_params)
            == refs["ann_index"]
        ),
    )
    k_new = int(payload.get("kNew", 15))
    neighbors_new = timed(
        "fork.queryNeighbors", lambda: ds.graph.neighbors(refs["ann_index"], k=k_new)
    )
    graph_new = timed("fork.buildConnectivityMap", lambda: ds.graph.connectivity(neighbors_new))
    umap_params = {
        k: v
        for k, v in umap_status.parameters.items()
        if k in ("n_epochs", "random_seed", "min_dist", "spread")
    }
    umap_new = timed(
        "fork.runUmap",
        lambda: ds.embeddings.umap(
            graph_new,
            refs["initialization"],
            nthreads=workers,
            parallel=parallel["umapParallel"],
            **umap_params,
        ),
    )
    leiden_params = {
        k: v
        for k, v in leiden_status.parameters.items()
        if k in ("resolution", "backend", "random_seed")
    }
    clusters_new = timed("fork.runLeiden", lambda: ds.clusters.leiden(graph_new, **leiden_params))

    def markers():
        all_features = ds.features.snapshot(
            from_assay="RNA", mask=np.ones(ds.RNA.feats.N, dtype=bool)
        )
        return ds.markers.search(
            clusters_new, from_assay="RNA", features=all_features, nthreads=workers
        )

    markers_new = timed("fork.findMarkers", markers)
    after = {ref.artifact_id for ref in ds.artifacts.list(from_assay="RNA")}
    new_ids = after - before
    lineage = ds.artifacts.lineage(markers_new)
    graph = getattr(lineage, "graph", None) or getattr(lineage, "_graph")
    nodes = []
    for node, attributes in graph.nodes(data=True):
        ref = node if isinstance(node, ArtifactRef) else attributes.get("ref")
        artifact_id = getattr(ref, "artifact_id", str(node))
        nodes.append(
            {
                "id": str(artifact_id),
                "kind": getattr(ref, "kind", attributes.get("kind")),
                "recomputedInFork": str(artifact_id) in new_ids,
                "operation": attributes.get("operation"),
            }
        )
    edges = [
        {"source": str(getattr(a, "artifact_id", a)), "target": str(getattr(b, "artifact_id", b))}
        for a, b, _ in graph.edges(data=True)
    ]
    uri = payload["referenceStoreUri"]
    counts_bytes, _ = _prefix_bytes(f"{uri}/RNA/counts")
    counts_t_bytes, _ = _prefix_bytes(f"{uri}/RNA/countsT")
    total_bytes, total_objects = _prefix_bytes(uri)
    fork_seconds = sum((v["seconds"] for k, v in stages.items() if k.startswith("fork.")))
    result = {
        "experiment": "rerun",
        "referenceTag": payload["referenceTag"],
        "referenceStoreUri": uri,
        "container": resources,
        "threadSettings": {
            "umap": thread_settings(umap_status, ds.artifacts.inspect(umap_new)),
            "neighbors": thread_settings(
                ds.artifacts.inspect(refs["neighbors"]), ds.artifacts.inspect(neighbors_new)
            ),
            # The fork reuses the stored neighbor index, so it records that index's settings only.
            "annIndex": {"stored": recorded_threads(ann_status)},
        },
        "reuse": reuse,
        "kOriginal": int(ds.artifacts.inspect(refs["neighbors"]).parameters.get("k", 0)),
        "kNew": k_new,
        "stages": stages,
        "forkSeconds": fork_seconds,
        "fullRunStageSeconds": full_run,
        "fullRunStagePeakRssBytes": full_peaks,
        "fullRunAnalysisSeconds": sum((float(full_run[s] or 0.0) for s in STAGE_ALIASES)),
        "fullRunWallSeconds": payload.get("fullRunWallSeconds"),
        "newArtifacts": len(new_ids),
        "store": {
            "totalBytes": total_bytes,
            "objects": total_objects,
            "countsBytes": counts_bytes,
            "countsTBytes": counts_t_bytes,
            "analysisBytes": total_bytes - counts_bytes - counts_t_bytes,
        },
        "lineage": {"nodes": nodes, "edges": edges},
        "refs": {
            "neighbors": neighbors_new.to_dict(),
            "graph": graph_new.to_dict(),
            "umap": umap_new.to_dict(),
            "clusters": clusters_new.to_dict(),
            "markers": markers_new.to_dict(),
        },
    }
    put_json(payload["outputUri"], result)
    return {"outputUri": payload["outputUri"], "forkSeconds": fork_seconds, "reuse": reuse}
