import asyncio
import hashlib
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from experiments.benchmarks import stages
from experiments.benchmarks.config import (
    CountMatrixConfig,
    StageResources,
    StorageIoConfig,
    WorkflowParameters,
)
from experiments.benchmarks.stages import run_stage
from experiments.runtime.common import configure_cytearc, write_json
from experiments.runtime.harness import R2
from experiments.runtime.recording_store import StoreProbe, _is_count_chunk


class ObjectReadProbe(StoreProbe):
    def __init__(self, maximumAttempts: int = 4):
        super().__init__()
        if maximumAttempts < 1:
            raise ValueError("maximumAttempts must be at least 1")
        self.maximumAttempts = int(maximumAttempts)
        self.transport: defaultdict[str, float] = defaultdict(int)
        self.failed_statuses: defaultdict[str, int] = defaultdict(int)
        self.count_failed_statuses: defaultdict[str, int] = defaultdict(int)

    def reset(self):
        super().reset()
        with self._lock:
            self.transport.clear()
            self.failed_statuses.clear()
            self.count_failed_statuses.clear()

    def received(self, key, ranged, size, seconds, status=None):
        with self._lock:
            self.transport["getRequests"] += 1
            self.transport["rangedGetRequests"] += int(ranged)
            self.transport["requestSeconds"] += seconds
            if status is None:
                self.transport["responseBytes"] += size
            else:
                self.transport["failedGetRequests"] += 1
                self.failed_statuses[status] += 1
            if _is_count_chunk(key):
                self.transport["countGetRequests"] += 1
                self.transport["countRangedGetRequests"] += int(ranged)
                self.transport["countRequestSeconds"] += seconds
                if status is None:
                    self.transport["countResponseBytes"] += size
                else:
                    self.transport["countFailedGetRequests"] += 1
                    self.count_failed_statuses[status] += 1

    def backed_off(self, key, seconds):
        with self._lock:
            self.transport["retryBackoffSeconds"] += seconds
            if _is_count_chunk(key):
                self.transport["countRetryBackoffSeconds"] += seconds

    def to_json(self):
        result = super().to_json()
        with self._lock:
            observed = dict(self.transport)
            failed_statuses = dict(self.failed_statuses)
            count_failed_statuses = dict(self.count_failed_statuses)
        requests = observed.get("countGetRequests", 0)
        result["transport"] = {
            **observed,
            "countWholeGetRequests": requests - observed.get("countRangedGetRequests", 0),
            "meanCountRequestSeconds": observed.get("countRequestSeconds", 0) / requests
            if requests
            else None,
            "failedGetRequests": observed.get("failedGetRequests", 0),
            "countFailedGetRequests": observed.get("countFailedGetRequests", 0),
            "retryBackoffSeconds": observed.get("retryBackoffSeconds", 0),
            "countRetryBackoffSeconds": observed.get("countRetryBackoffSeconds", 0),
            "failedGetStatuses": failed_statuses,
            "countFailedGetStatuses": count_failed_statuses,
            "failedResponseBytes": None if observed.get("failedGetRequests", 0) else 0,
            "countFailedResponseBytes": None if observed.get("countFailedGetRequests", 0) else 0,
            "failedTransferBytesUnknown": bool(observed.get("failedGetRequests", 0)),
            "countFailedTransferBytesUnknown": bool(observed.get("countFailedGetRequests", 0)),
            "byteBoundary": "Returned payload bytes from completed GET calls; excludes headers, error response bodies and unknown partial transfers from failed attempts",
            "getAttemptDefinition": "One SDK-retry-disabled ObjectStore.get invocation, including failures; not a provider-billed request count",
            "maximumGetAttempts": self.maximumAttempts,
            "retryDelaysSeconds": [0.5 * 2**attempt for attempt in range(self.maximumAttempts - 1)],
            "putRetryPolicy": "Original backend policy, unchanged by GET instrumentation",
            "scope": "S3 GETs after Zarr range coalescing; SDK retries disabled",
        }
        return result


@contextmanager
def record_object_reads(probe):
    from obstore.exceptions import GenericError
    from obstore.store import S3Store
    from zarr.storage import ObjectStore

    original_get = ObjectStore.get
    read_stores = {}

    async def get(store, key, prototype, byte_range=None):
        if not isinstance(store.store, S3Store):
            return await original_get(store, key, prototype, byte_range)
        with probe._lock:
            if id(store) not in read_stores:
                backend = store.store
                read_stores[id(store)] = (
                    store,
                    ObjectStore(
                        S3Store(
                            config=backend.config,
                            prefix=backend.prefix,
                            client_options=backend.client_options,
                            credential_provider=backend.credential_provider,
                            retry_config={"max_retries": 0},
                        ),
                        read_only=True,
                    ),
                )
            read_store = read_stores[id(store)][1]
        last_attempt = probe.maximumAttempts - 1
        for attempt in range(probe.maximumAttempts):
            started = time.perf_counter()
            failure = None
            value = None
            completed = False
            try:
                value = await original_get(read_store, key, prototype, byte_range)
                completed = True
                return value
            except GenericError as error:
                failure = error
                match = re.search(r"status code: (\d{3})\b", str(error))
                status = int(match[1]) if match else None
                if attempt == last_attempt or (
                    status is not None and status not in (408, 429) and not 500 <= status < 600
                ):
                    raise
            finally:
                failure = failure or (sys.exception() if not completed else None)
                match = re.search(r"status code: (\d{3})\b", str(failure)) if failure else None
                probe.received(
                    key,
                    byte_range is not None,
                    len(value) if value is not None else 0,
                    time.perf_counter() - started,
                    (match[1] if match else type(failure).__name__) if failure else None,
                )
            backoff_started = time.perf_counter()
            try:
                await asyncio.sleep(0.5 * 2**attempt)
            finally:
                probe.backed_off(key, time.perf_counter() - backoff_started)

    ObjectStore.get = get
    try:
        yield
    finally:
        ObjectStore.get = original_get


PREPARATION_STAGES = (
    "createStore",
    "writeCountsT",
    "initializeStore",
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "runLeiden",
)
IMPORT_STAGES = ("createStore", "writeCountsT", "initializeStore")
ENDPOINTS = ("markHvgs", "runNormalization", "findMarkers")


def source_identity(path, payload):
    if (
        payload["nRows"] > 30_000
        and not {"expectedNonzeroCounts", "expectedDatasetSha256"} <= payload.keys()
    ):
        raise ValueError(
            "Large controlled workloads require canonical source checksum and nnz guards"
        )
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("Controlled count requests require an explicit CSR H5AD source")
        cells, genes = map(int, matrix.attrs["shape"])
        nonzero = int(matrix["data"].size)
        if cells != payload["nRows"]:
            raise ValueError("Source cell count differs from the fixed count-request input")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if payload.get("expectedNonzeroCounts", nonzero) != nonzero:
        raise ValueError("Source nnz differs from the canonical preparation guard")
    if payload.get("expectedDatasetSha256", digest) != digest:
        raise ValueError("Source checksum differs from the canonical preparation guard")
    return {
        "datasetUri": payload["datasetUri"],
        "sha256": digest,
        "cells": cells,
        "genes": genes,
        "nonzeroCounts": nonzero,
        "bytes": path.stat().st_size,
    }


def array_digest(array):
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], array.chunks[0]):
        block = np.ascontiguousarray(array[start : start + array.chunks[0]])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def execute_stages(
    payload,
    source,
    work,
    storage,
    result,
    sequence,
    session,
    *,
    store_uri,
    workflow,
    input_refs=None,
):
    resources = StageResources.model_validate(payload["resources"])
    policy = CountMatrixConfig.model_validate(payload["countMatrix"])
    context = storage.get_json(payload["recordUri"])
    serial_import = StorageIoConfig(readWorkers=1, computeWorkers=1, writeWorkers=1)
    for stage in sequence:
        measured = run_stage(
            stage,
            submissionId=payload["submissionId"],
            nRows=payload["nRows"],
            storeUri=store_uri,
            workflow=workflow,
            resources=resources,
            localH5adPath=source,
            countMatrix=policy,
            workDir=work,
            storageIo=serial_import if stage in {"createStore", "writeCountsT"} else None,
            containerMemoryMb=resources.modalMemoryLimitMb,
            containerCpuRequest=resources.modalCpuRequest,
            containerCpuLimit=resources.modalCpuLimit,
            session=session,
            inputRefs=input_refs,
            recordStoreOperations=True,
        ).to_json()
        result["stages"].append(measured)
        write_json("count_requests.json", result, directory=work)
        storage.put_json(payload["recordUri"], {**context, "result": result})
        if measured["status"] != "ok":
            raise RuntimeError(measured["error"])
        if result["mode"] == "measurement" and stage in ENDPOINTS:
            observed = measured["details"]["storeOperations"]
            if observed["partialGets"] or not observed["transport"].get("countGetRequests"):
                raise RuntimeError(
                    "Count reads did not follow the verified post-coalescing GET path"
                )


def prepare(payload: dict) -> dict:
    configure_cytearc()
    work = Path(payload.get("workDirectory", "/tmp/count_request_preparation"))
    work.mkdir(parents=True, exist_ok=True)
    storage = R2(payload["r2EndpointUrl"])
    source = work / "input.h5ad"
    storage.download(payload["datasetUri"], source)
    result = {
        "mode": "preparation",
        "status": "running",
        "settings": payload,
        "stages": [],
        "source": source_identity(source, payload),
    }
    session = {}
    workflow = WorkflowParameters(annParallel=False, umapParallel=False, leidenBackend="cytearc")
    execute_stages(
        payload,
        source,
        work,
        storage,
        result,
        PREPARATION_STAGES,
        session,
        store_uri=str(work / "reference.zarr"),
        workflow=workflow,
    )
    store, refs = session["store"], session["artifactRefs"]
    cells = np.asarray(store.artifacts.load(refs["filterCells"])["values"][:], dtype=bool)
    features = np.asarray(store.artifacts.load(refs["markHvgs"])["values"][:], dtype=bool)
    labels = np.asarray(store.artifacts.load(refs["runLeiden"])["values"][:])
    if labels.shape != (int(cells.sum()),) or features.sum() != workflow.topN:
        raise ValueError("Reference cells, selected features or clustering labels are incomplete")
    cell_ids = np.asarray(store.cells.fetch_all("ids"))
    feature_ids = np.asarray(store.RNA.feats.fetch_all("ids"))
    archive = work / "frozen_inputs.npz"
    np.savez_compressed(archive, cells=cells, features=features, labels=labels)
    uri = payload["frozenInputsUri"]
    storage.upload(archive, uri)
    result["frozenInputs"] = {
        "uri": uri,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source": result["source"],
        "workflow": workflow.model_dump(mode="json"),
        "cellIdsSha256": stages._ordered_id_digest(cell_ids),
        "selectedCellIdsSha256": stages._ordered_id_digest(cell_ids[cells]),
        "featureIdsSha256": stages._ordered_id_digest(feature_ids),
        "selectedFeatureIdsSha256": stages._ordered_id_digest(feature_ids[features]),
        "labelsSha256": stages._ordered_id_digest(labels),
        "selectedCells": int(cells.sum()),
        "selectedFeatures": int(features.sum()),
        "groups": int(len(np.unique(labels))),
        "normalizationSha256": array_digest(store.artifacts.load(refs["runNormalization"])["data"]),
        "normalizationShape": [int(cells.sum()), int(features.sum())],
        "normalizationDtype": str(store.artifacts.load(refs["runNormalization"])["data"].dtype),
        "scope": "One real QC mask, HVG set and native PCA/KNN/Leiden partition per dataset; fixed before layout measurements",
    }
    storage.put_json(uri.removesuffix(".npz") + ".json", result["frozenInputs"])
    result["status"] = "ok"
    return result


def run(payload: dict) -> dict:
    configure_cytearc()
    work = Path(payload.get("workDirectory", "/tmp/count_requests"))
    work.mkdir(parents=True, exist_ok=True)
    storage = R2(payload["r2EndpointUrl"])
    source = work / "input.h5ad"
    storage.download(payload["datasetUri"], source)
    identity = source_identity(source, payload)
    frozen = storage.get_json(payload["frozenInputsUri"].removesuffix(".npz") + ".json")
    if frozen["source"] != identity:
        raise ValueError("Final layout measurement differs from the frozen canonical source")
    archive = work / "frozen_inputs.npz"
    storage.download(payload["frozenInputsUri"], archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != frozen["sha256"]:
        raise ValueError("Frozen workload archive checksum changed")
    with np.load(archive, allow_pickle=False) as arrays:
        cells, features, labels = (arrays[key] for key in ("cells", "features", "labels"))
    if cells.shape != (identity["cells"],) or features.shape != (identity["genes"],):
        raise ValueError("Frozen workload dimensions differ from imported raw counts")
    if cells.dtype != bool or features.dtype != bool or labels.shape != (int(cells.sum()),):
        raise ValueError("Malformed frozen mask or label arrays")
    workflow = WorkflowParameters.model_validate(frozen["workflow"])
    result = {
        "status": "running",
        "mode": "measurement",
        "settings": {
            **payload,
            "runId": (
                f"embryo_{payload['nRows']}_{payload['layout']}_replicate_{payload['replicate']}"
            ),
        },
        "stages": [],
        "endpoints": list(ENDPOINTS),
        "frozenInputs": frozen,
        "timingBoundary": "Import and frozen-artifact setup are separate; endpoint operation excludes setup and result persistence",
    }
    session = {"storeProbe": StoreProbe()}
    execute_stages(
        payload,
        source,
        work,
        storage,
        result,
        IMPORT_STAGES,
        session,
        store_uri=payload["storeUri"],
        workflow=workflow,
    )
    store = session["store"]
    cell_ids, feature_ids = (
        np.asarray(store.cells.fetch_all("ids")),
        np.asarray(store.RNA.feats.fetch_all("ids")),
    )
    if (
        stages._ordered_id_digest(cell_ids) != frozen["cellIdsSha256"]
        or stages._ordered_id_digest(feature_ids) != frozen["featureIdsSha256"]
    ):
        raise ValueError("Freshly imported cell or gene identities differ from the frozen source")
    store.cells.insert("request_cells", cells, overwrite=True)
    selected_cells = store.snapshot_cell_selection("request_cells")
    selected_features = store.features.snapshot(from_assay="RNA", mask=features)
    full_labels = np.full(len(cells), -1, dtype=labels.dtype)
    full_labels[cells] = labels
    store.cells.insert("request_groups", full_labels, overwrite=True)
    clusters = store.snapshot_cluster_labels("request_groups", cell_selection=selected_cells)
    refs = {"cells": selected_cells, "features": selected_features, "clusters": clusters}
    result["frozenWorkloadValidation"] = {
        "selectedCells": int(cells.sum()),
        "selectedFeatures": int(features.sum()),
        "groups": int(len(np.unique(labels))),
        "selectedCellIdsSha256": stages._ordered_id_digest(cell_ids[cells]),
        "selectedFeatureIdsSha256": stages._ordered_id_digest(feature_ids[features]),
        "labelsSha256": stages._ordered_id_digest(labels),
    }
    for key in ("selectedCellIdsSha256", "selectedFeatureIdsSha256", "labelsSha256"):
        if result["frozenWorkloadValidation"][key] != frozen[key]:
            raise ValueError(f"Frozen workload identity changed: {key}")
    session.pop("store")
    del store
    probe = ObjectReadProbe()
    session["storeProbe"] = probe
    with record_object_reads(probe):
        for stage in ENDPOINTS:
            execute_stages(
                payload,
                source,
                work,
                storage,
                result,
                [stage],
                session,
                store_uri=payload["storeUri"],
                workflow=workflow,
                input_refs=refs,
            )
            store = session["store"]
            if stage == "markHvgs":
                measured = np.asarray(
                    store.artifacts.load(session["artifactRefs"][stage])["values"][:], dtype=bool
                )
                result["measuredHvgSelectionMatchesFrozen"] = bool(
                    np.array_equal(measured, features)
                )
    validation_started = time.perf_counter()
    normalized = store.artifacts.load(session["artifactRefs"]["runNormalization"])["data"]
    if (
        list(normalized.shape) != frozen["normalizationShape"]
        or str(normalized.dtype) != frozen["normalizationDtype"]
    ):
        raise ValueError("Normalized data dimensions or dtype differ from the frozen workload")
    result["normalizationSha256"] = array_digest(normalized)
    result["normalizationMatchesFrozen"] = (
        result["normalizationSha256"] == frozen["normalizationSha256"]
    )
    result["markerArtifact"] = session["artifactRefs"]["findMarkers"].to_dict()
    result["numericalValidationSeconds"] = time.perf_counter() - validation_started
    result["numericalValidationBoundary"] = (
        "Read and hash every stored normalized value after all timed endpoints; marker statistics compared separately across the paired layouts"
    )
    if not result["normalizationMatchesFrozen"]:
        context = storage.get_json(payload["recordUri"])
        storage.put_json(payload["recordUri"], {**context, "result": result})
        raise ValueError("Final normalized values differ from the frozen reference")
    result["status"] = "ok"
    return result


def compare(payload: dict) -> dict:
    import cytearc
    from cytearc.storage import ArtifactRef

    storage = R2(payload["r2EndpointUrl"])
    records = []
    for uri in payload["recordUris"]:
        envelope = storage.get_json(uri)
        if envelope["status"] != "completed" or envelope["result"].get("mode") != "measurement":
            raise ValueError("Paired comparison requires completed frozen-workload measurements")
        record = envelope["result"]
        if record["status"] != "ok" or not record["normalizationMatchesFrozen"]:
            raise ValueError("An input measurement failed its frozen numerical control")
        records.append((uri, record))
    cases = defaultdict(list)
    for uri, record in records:
        cases[record["settings"]["nRows"]].append((uri, record))
    expected = payload.get("expectedReplicates", 3)
    results = []
    for cells, cases_for_size in sorted(cases.items()):
        fixed = [(u, r) for u, r in cases_for_size if r["settings"]["layout"] == "fixed"]
        scaled = [(u, r) for u, r in cases_for_size if r["settings"]["layout"] == "scaled"]
        if len(fixed) != expected or len(scaled) != expected or len(cases_for_size) != 2 * expected:
            raise ValueError("Paired layouts lack the fixed computational repetition count")
        base_uri, baseline = sorted(fixed, key=lambda item: item[1]["settings"]["replicate"])[0]

        def open_markers(record):
            ds = cytearc.DataStore(
                record["settings"]["storeUri"],
                zarr_mode="r",
                storage_options=stages.storage_options(record["settings"]["storeUri"]),
                min_features_per_cell=-1,
                mem_budget="3G",
                nthreads=2,
            )
            return ds.artifacts.load(ArtifactRef.from_dict(record["markerArtifact"]))

        base = open_markers(baseline)
        groups = sorted(base.group_keys())
        columns = list(base.attrs["stat_columns"])
        for uri, record in cases_for_size:
            if (
                record["frozenInputs"] != baseline["frozenInputs"]
                or record["frozenWorkloadValidation"] != baseline["frozenWorkloadValidation"]
            ):
                raise ValueError("Paired layouts changed cells, genes or marker labels")
            if record["settings"]["resources"] != baseline["settings"]["resources"]:
                raise ValueError("Paired layouts changed their configured resources")
            endpoint_stages = [stage for stage in record["stages"] if stage["stage"] in ENDPOINTS]
            if sorted(stage["stage"] for stage in endpoint_stages) != sorted(ENDPOINTS):
                raise ValueError("Paired measurement omits or repeats a timed endpoint")
            transport_breakdown = {}
            for stage in endpoint_stages:
                observed = stage["details"]["storeOperations"]
                transport = observed["transport"]
                total, ranged = transport["countGetRequests"], transport["countRangedGetRequests"]
                if (
                    observed["partialGets"]
                    or transport["scope"]
                    != "S3 GETs after Zarr range coalescing; SDK retries disabled"
                    or not 0 <= ranged <= total
                    or total == 0
                ):
                    raise ValueError(
                        "Paired measurements use an unverified or different I/O counting scope"
                    )
                transport_breakdown[stage["stage"]] = {
                    "totalCountGets": total,
                    "rangedCountGets": ranged,
                    "wholeCountGets": total - ranged,
                }
            other = open_markers(record)
            if sorted(other.group_keys()) != groups or list(other.attrs["stat_columns"]) != columns:
                raise ValueError("Paired layouts produced different marker groups or statistics")
            for key in ("feature_index", "feature_names"):
                if not np.array_equal(base[key][:], other[key][:]):
                    raise ValueError("Paired marker feature identities differ")
            errors = np.zeros(len(columns), dtype=float)
            compared = np.zeros(len(columns), dtype=np.int64)
            passed = np.ones(len(columns), dtype=bool)
            finite_patterns = True
            for group in groups:
                left, right = (
                    np.asarray(base[group]["stats"][:]),
                    np.asarray(other[group]["stats"][:]),
                )
                if left.shape != right.shape:
                    raise ValueError("Paired marker statistic dimensions differ")
                if (
                    not np.array_equal(np.isfinite(left), np.isfinite(right))
                    or not np.array_equal(np.isnan(left), np.isnan(right))
                    or not np.array_equal(np.isposinf(left), np.isposinf(right))
                    or not np.array_equal(np.isneginf(left), np.isneginf(right))
                ):
                    finite_patterns = False
                for index, column in enumerate(columns):
                    finite = np.isfinite(left[:, index]) & np.isfinite(right[:, index])
                    delta = np.abs(left[finite, index] - right[finite, index])
                    compared[index] += int(finite.sum())
                    errors[index] = max(errors[index], float(delta.max(initial=0)))
                    absolute = (
                        1e-12
                        if column == "auc"
                        else 1e-7
                        if column == "neg_log10_p_value"
                        else 1e-8
                    )
                    passed[index] &= np.all(delta <= absolute + 1e-8 * np.abs(left[finite, index]))
            results.append(
                {
                    "cells": cells,
                    "runId": record["settings"]["runId"],
                    "baseline": baseline["settings"]["runId"],
                    "frozenWorkloadIdentical": True,
                    "normalizationIdentical": True,
                    "markerMaxAbsoluteErrors": {
                        column: float(error) if count else None
                        for column, error, count in zip(columns, errors, compared, strict=True)
                    },
                    "markerFiniteValuesCompared": dict(
                        zip(columns, compared.tolist(), strict=True)
                    ),
                    "markerStatisticTolerancePassed": dict(
                        zip(columns, passed.tolist(), strict=True)
                    ),
                    "markerFinitePatternsIdentical": finite_patterns,
                    "ioScopeVerified": True,
                    "countGetBreakdown": transport_breakdown,
                    "passed": bool(passed.all() and finite_patterns),
                }
            )
    return {
        "mode": "pairedLayoutValidation",
        "passed": all(row["passed"] for row in results),
        "measurements": results,
        "tolerances": {
            "aucAbsolute": 1e-12,
            "otherStatisticsAbsolute": 1e-8,
            "negLog10PAbsolute": 1e-7,
            "relative": 1e-8,
        },
        "scope": "Every normalized value hashed; every stored marker statistic compared with matched groups, genes and finite-value patterns; all validation outside timed endpoints",
    }


RECORDS = Path(__file__).resolve().parent / "records"
# Each run's record in the experiment bucket, under experiments/count_requests/, by run.
COLLECTED = {
    **{
        f"embryo_{cells}_{layout}_replicate_{replicate}": (
            f"embryo/{cells}/{layout}/replicate_{replicate}/measurement.json"
        )
        for cells in (5_000_000, 10_000_000)
        for layout in ("scaled", "fixed")
        for replicate in (1, 2, 3)
    },
    "paired_layout_validation": "embryo/comparison.json",
}
# Locations, job identifiers and the container's platform leave each run's record.
SETTINGS_DROPPED = frozenset(
    {
        "recordUri",
        "r2EndpointUrl",
        "datasetUri",
        "storeUri",
        "frozenInputsUri",
        "workDirectory",
        "submissionId",
    }
)
STAGE_DROPPED = frozenset({"datasetUri", "storeUri", "submissionId", "provenance"})


def collected_record(envelope: dict) -> dict:
    """A run's status and result, without locations, job identifiers or platform details."""
    from experiments.runtime.common import scrub_private

    result = dict(envelope["result"])
    if "settings" in result:
        result["settings"] = {
            key: value for key, value in result["settings"].items() if key not in SETTINGS_DROPPED
        }
    if "stages" in result:
        result["stages"] = [
            {key: value for key, value in stage.items() if key not in STAGE_DROPPED}
            for stage in result["stages"]
        ]
    if "frozenInputs" in result:
        frozen = {key: value for key, value in result["frozenInputs"].items() if key != "uri"}
        frozen["source"] = {
            key: value for key, value in frozen["source"].items() if key != "datasetUri"
        }
        result["frozenInputs"] = frozen
    return scrub_private({"status": envelope["status"], "result": result})


def collect() -> None:
    """Write the layout runs and their paired validation to records/count_requests.json."""
    from experiments.runtime.locations import experiment_bucket
    from experiments.runtime.object_io import get_json

    bucket = experiment_bucket()
    records = {}
    for name, key in COLLECTED.items():
        envelope = get_json(bucket.uri(f"experiments/count_requests/{key}"))
        if envelope["status"] != "completed":
            raise SystemExit(f"{name} has status {envelope['status']!r}")
        records[name] = collected_record(envelope)
    path = write_json("count_requests.json", {"records": records}, directory=RECORDS)
    print(f"wrote {path} ({len(records)} runs)")


if __name__ == "__main__":
    collect()
