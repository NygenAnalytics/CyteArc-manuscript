"""Measure whether CyteArc's count-matrix steps stay within the memory budget.

CyteArc plans each count-matrix step to fit ``mem_budget``. This job separates the memory that a
step plans from the memory its process already holds. One call takes one embryo sample, one budget
and one worker count from a profiling configuration, and runs each requested arm on its own store
on the container's local disk, importing the sample again for every arm:

- ``freshProcess``: every step in a new Python process that reads the artifacts of earlier steps.
- ``sameProcess``: every step in one process sharing one DataStore, Leiden included, as one
  script runs them.
- ``sameProcessReleased``: as ``sameProcess``, but before findMarkers the DataStore is dropped and
  garbage collected, glibc returns its free memory (``malloc_trim``), and findMarkers opens a new
  DataStore. Memory is recorded after each of these.
- ``sameProcessMmapThreshold``: as ``sameProcess`` with ``MALLOC_MMAP_THRESHOLD_=131072``, so glibc
  serves every allocation above 128 KiB with its own mapping and unmaps it when it is freed.

``stages.run_stage`` measures each step: container ``memory.current`` and process-tree RSS sampled
every 50 ms from the step's start, the step's planned reservations from CyteArc's execution
reports, and its time. Each step record adds the process's anonymous and file-backed resident
memory and glibc's allocator totals before and after the step, and the step's wall-clock window.
The parent process samples the worker and the container every 50 ms as well, outside the
worker's GIL, and reports the largest sample in each step's window beside the worker's own
measurement. Records are saved under the configuration's result prefix as ``<arm>/<stage>.json``
as each step finishes, with a per-second trace of each arm as ``<arm>/memoryTrace.json`` and a
summary of every arm as ``summary.json``.

Usage (local): ``python -m experiments benchmarks.budget_compliance run PAYLOAD`` with a payload
holding ``configPath``, ``arms``, ``containerMemoryMb`` (the container size the job was launched
with, used when the cgroup does not report a memory limit, as on Modal) and optionally
``workDirectory``. Each worker's stderr is copied to the job's log, and its last 50 lines are kept
in ``<arm>/workers.json`` and the summary.

``python -m experiments.benchmarks.budget_compliance collect`` reads the jobs of ``COLLECTED``
from the experiment bucket and writes one row per job, arm and step to
``records/memory_budget_stages.csv``.
"""

import argparse
import ctypes
import gc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from array import array
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any

from experiments.runtime.common import write_json
from experiments.runtime.harness import cgroup_memory_current, process_tree_rss

WORKER_MODULE = "experiments.benchmarks.budget_compliance"
SAMPLE_INTERVAL_SECONDS = 0.05
MMAP_THRESHOLD_BYTES = 131072
STDERR_TAIL_LINES = 50
UNLIMITED_CGROUP_BYTES = 1 << 60
IMPORT_STAGES = ("createStore", "writeCountsT", "initializeStore")
FRESH_PROCESS_STAGES = (
    *IMPORT_STAGES,
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "runLeiden",
    "findMarkers",
)
SAME_PROCESS_STAGES = (
    *IMPORT_STAGES,
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "runUmap",
    "runLeiden",
    "findMarkers",
)


# With a configured cluster source, imported Leiden labels replace the graph steps, whose
# memory at 10 million cells exceeds the small budgets this comparison tests.
IMPORTED_CLUSTER_STAGES = (
    *IMPORT_STAGES,
    "filterCells",
    "markHvgs",
    "runNormalization",
    "importClusters",
    "findMarkers",
)


@dataclass(frozen=True)
class Arm:
    """The steps of one arm, whether they share a process, and what else it changes."""

    stages: tuple[str, ...]
    sameProcess: bool
    environment: dict[str, str] = field(default_factory=dict)
    releaseBefore: str | None = None

    def stagesFor(self, config: Any, nRows: int) -> tuple[str, ...]:
        """This arm's steps for one size: imported labels replace the graph steps when configured."""
        return self.stages if config.clusterSourceFor(nRows) is None else IMPORTED_CLUSTER_STAGES


ARMS = {
    "freshProcess": Arm(FRESH_PROCESS_STAGES, sameProcess=False),
    "sameProcess": Arm(SAME_PROCESS_STAGES, sameProcess=True),
    "sameProcessReleased": Arm(SAME_PROCESS_STAGES, sameProcess=True, releaseBefore="findMarkers"),
    "sameProcessMmapThreshold": Arm(
        SAME_PROCESS_STAGES,
        sameProcess=True,
        environment={"MALLOC_MMAP_THRESHOLD_": str(MMAP_THRESHOLD_BYTES)},
    ),
}


class _MallInfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


@cache
def _glibc() -> ctypes.CDLL:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallinfo2.restype = _MallInfo2
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            "The allocator measurements need glibc 2.33 or later (mallinfo2 and malloc_trim)"
        ) from error
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    return libc


def process_memory() -> dict[str, int | None]:
    """Return this process's resident memory by kind, the container's, and glibc's totals.

    Modal's gVisor containers report only ``VmRSS`` in ``/proc/self/status``, so the anonymous,
    file-backed and shared parts are None there.
    """
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "RssAnon", "RssFile", "RssShmem"}:
            status[key] = int(value.split()[0]) * 1024
    if "VmRSS" not in status:
        raise RuntimeError("/proc/self/status has no VmRSS line, so resident memory is unknown")
    info = _glibc().mallinfo2()
    return {
        "processRssBytes": status["VmRSS"],
        "anonymousRssBytes": status.get("RssAnon"),
        "fileRssBytes": status.get("RssFile"),
        "sharedRssBytes": status.get("RssShmem"),
        "containerMemoryCurrentBytes": cgroup_memory_current(),
        "allocatorHeapBytes": int(info.arena),
        "allocatorMappedBytes": int(info.hblkhd),
        "allocatorInUseBytes": int(info.uordblks),
        "allocatorFreeHeldBytes": int(info.fordblks),
        "allocatorTrimmableBytes": int(info.keepcost),
    }


def release_memory(session: dict[str, Any]) -> dict[str, Any]:
    """Drop the shared DataStore, collect garbage and trim glibc, recording memory after each."""
    held = process_memory()
    if "store" not in session:
        raise ValueError("The release step needs the DataStore that earlier steps shared")
    del session["store"]
    unreachable = gc.collect()
    collected = process_memory()
    started = time.perf_counter()
    trimmed = bool(_glibc().malloc_trim(0))
    trim_seconds = time.perf_counter() - started
    return {
        "stage": "releaseMemory",
        "held": held,
        "afterGarbageCollection": collected,
        "afterMallocTrim": process_memory(),
        "unreachableObjects": unreachable,
        "mallocTrimReturnedMemory": trimmed,
        "mallocTrimSeconds": trim_seconds,
    }


def _save_record(request: dict[str, Any], name: str, record: dict[str, Any]) -> None:
    write_json(f"{name}.json", record, directory=Path(request["recordDirectory"]))
    if request["uploadPrefix"] is not None:
        from experiments.runtime.object_io import put_json

        put_json(f"{request['uploadPrefix']}/{name}.json", record)


def _cluster_count(store_uri: str, config: Any, session: dict | None, ref: Any) -> dict[str, int]:
    import numpy as np

    from experiments.benchmarks.stages import _open_datastore

    store = (session or {}).get("store") or _open_datastore(
        store_uri, config.workflow, config.resourcesFor("runLeiden"), initialize=False
    )
    labels = np.asarray(store.artifacts.load(ref)["values"][:])
    return {"clusters": int(np.unique(labels).size), "cellsAfterFilter": int(labels.size)}


def run_worker(request: dict[str, Any]) -> None:
    """Run the requested steps of one arm in this process, saving each step's record."""
    from cytearc import ArtifactRef

    from experiments.benchmarks.config import bind_cluster_source, load_profiling_config
    from experiments.benchmarks.scaling import CLUSTER_LABEL_STAGES
    from experiments.benchmarks.stages import profile_stage_inputs, run_stage
    from experiments.runtime.common import configure_cytearc

    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass
    configure_cytearc()
    config = load_profiling_config(request["configPath"])
    arm = request["arm"]
    spec = ARMS[arm]
    nRows = request["nRows"]
    labelled = None
    if config.clusterSourceFor(nRows) is not None and CLUSTER_LABEL_STAGES & set(request["stages"]):
        labelled = bind_cluster_source(config, nRows, spec.stagesFor(config, nRows))
    refs = {stage: ArtifactRef.from_dict(value) for stage, value in request["artifactRefs"].items()}
    session = {"artifactRefs": refs} if spec.sameProcess else None
    for stage in request["stages"]:
        # As in scaling.stage, only the steps that use the labels see the cluster source.
        workflow = (
            labelled if labelled is not None and stage in CLUSTER_LABEL_STAGES else config.workflow
        )
        if stage == spec.releaseBefore:
            assert session is not None
            _save_record(request, "releaseMemory", {"arm": arm, **release_memory(session)})
        inputs = None
        if session is None:
            needed = profile_stage_inputs(workflow, stage)
            missing = sorted({source for source, _ in needed.values()} - set(refs))
            if missing:
                raise ValueError(f"{arm} {stage} needs the artifacts of {', '.join(missing)}")
            inputs = {name: refs[source] for name, (source, _) in needed.items()}
        before = process_memory()
        started = time.time()
        record = run_stage(
            stage,
            submissionId=config.runTag,
            nRows=request["nRows"],
            storeUri=request["storeUri"],
            workflow=workflow,
            resources=config.resourcesFor(stage),
            localH5adPath=Path(request["localH5adPath"]),
            countMatrix=config.countMatrix,
            storageIo=config.storageIo,
            workDir=Path(request["workDirectory"]),
            sampleIntervalSeconds=SAMPLE_INTERVAL_SECONDS,
            recordStoreOperations=False,
            inputRefs=inputs,
            session=session,
            leidenInProcess=True,
        ).to_json()
        record.update(
            arm=arm,
            processId=os.getpid(),
            environment=spec.environment,
            startedAt=started,
            finishedAt=time.time(),
            processMemory={"before": before, "after": process_memory()},
        )
        _save_record(request, stage, record)
        if record["status"] != "ok":
            raise RuntimeError(f"{arm} {stage} failed: {record['error']}")
        artifact = (record["details"] or {}).get("artifact")
        if artifact is not None:
            refs[stage] = ArtifactRef.from_dict(artifact)
    if "runLeiden" in request["stages"]:
        _save_record(
            request,
            "clusters",
            _cluster_count(request["storeUri"], config, session, refs["runLeiden"]),
        )


def _cgroup_memory_limit() -> tuple[int | None, str]:
    """Return the memory limit of this process's cgroup (v2 or v1) and where it was read."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        lines = []
    files = []
    for line in lines:
        hierarchy, controllers, group = (line.split(":", 2) + ["", ""])[:3]
        if hierarchy == "0" and not controllers:
            files.append(("cgroupV2", Path("/sys/fs/cgroup", group.lstrip("/"), "memory.max")))
        elif "memory" in controllers.split(","):
            files.append(
                (
                    "cgroupV1",
                    Path("/sys/fs/cgroup/memory", group.lstrip("/"), "memory.limit_in_bytes"),
                )
            )
    files += [
        ("cgroupV2", Path("/sys/fs/cgroup/memory.max")),
        ("cgroupV1", Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
    ]
    for source, path in files:
        try:
            text = path.read_text().strip()
        except OSError:
            continue
        if text.isdigit() and 0 < int(text) < UNLIMITED_CGROUP_BYTES:
            return int(text), source
    return None, "unavailable"


def _container_limits(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the container's memory limit, read from its cgroup or else from the payload."""
    limit, source = _cgroup_memory_limit()
    if limit is None and payload.get("containerMemoryMb") is not None:
        limit, source = int(payload["containerMemoryMb"]) * 1024**2, "payload"
    if limit is None:
        raise ValueError(
            "The container's memory limit is not readable from its cgroup; set containerMemoryMb "
            "in the payload to the container size the job was launched with"
        )
    try:
        cpu = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except OSError:
        cpu = []
    return {
        "memoryLimitBytes": limit,
        "memoryLimitSource": source,
        "cpuQuotaCores": int(cpu[0]) / int(cpu[1]) if cpu and cpu[0].isdigit() else None,
        "cpuCount": os.cpu_count(),
    }


def _relay_stderr(stream: IO[bytes], tail: deque[str]) -> None:
    """Copy a worker's stderr to this process's stderr and keep its last lines."""
    for line in iter(stream.readline, b""):
        text = line.decode(errors="replace")
        sys.stderr.write(text)
        sys.stderr.flush()
        tail.append(text.rstrip("\n"))
    stream.close()


def _uniform_resources(config: Any, stages: set[str]) -> Any:
    resources = {config.resourcesFor(stage) for stage in stages}
    if len(resources) != 1:
        raise ValueError(
            "Every step must use the same budget, workers and container; use fixedResources"
        )
    return resources.pop()


def _input(config: Any, nRows: int, work: Path) -> tuple[Path, dict[str, Any]]:
    uri = config.datasetUri(nRows)
    if not uri.startswith("s3://"):
        path = Path(uri.removeprefix("file://"))
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, {"datasetUri": uri, "fileBytes": path.stat().st_size}
    from experiments.runtime.object_io import download_file, get_json

    expected = get_json(uri.removesuffix(".h5ad") + ".json")
    path = work / "input" / f"{nRows}.h5ad"
    started = time.perf_counter()
    download = download_file(uri, path)
    if download.fileBytes != expected["fileBytes"]:
        raise ValueError(
            f"{uri} has {download.fileBytes} bytes; its preparation record lists "
            f"{expected['fileBytes']}"
        )
    return path, {
        "datasetUri": uri,
        "fileBytes": download.fileBytes,
        "preparedSha256": expected["sha256"],
        "nonzeroCounts": expected["nnz"],
        "downloadSeconds": time.perf_counter() - started,
    }


class WorkerSampler:
    """Sample one worker's process tree and the container from this idle parent process.

    The worker's own sampler is a thread that needs the GIL, which a few CyteArc kernels hold
    while they run; this one does not, so it checks that no step peak falls between samples.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.times = array("d")
        self.processRss = array("q")
        self.container = array("q")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.times.append(time.time())
            self.processRss.append(process_tree_rss(self.pid))
            self.container.append(cgroup_memory_current() or -1)
            self._stop.wait(SAMPLE_INTERVAL_SECONDS)

    def __enter__(self) -> "WorkerSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def peaks(self, started: float, finished: float) -> dict[str, int | None]:
        """Return the largest samples taken while a step ran."""
        first = bisect_left(self.times, started)
        last = bisect_right(self.times, finished)
        container = max(self.container[first:last], default=-1)
        return {
            "externalSamples": last - first,
            "externalProcessRssPeakBytes": max(self.processRss[first:last], default=None),
            "externalContainerPeakBytes": None if container < 0 else container,
        }

    def per_second(self) -> list[list[int | None]]:
        """Return the largest samples in each second, for a memory trace of the arm."""
        rss: dict[int, int] = {}
        container: dict[int, int] = {}
        for moment, process, total in zip(self.times, self.processRss, self.container):
            second = int(moment)
            rss[second] = max(rss.get(second, 0), process)
            container[second] = max(container.get(second, -1), total)
        return [
            [second, rss[second], None if container[second] < 0 else container[second]]
            for second in rss
        ]


def stage_summary(record: dict[str, Any], sampled: dict[str, Any]) -> dict[str, Any]:
    """Return a step's memory, time and largest planned reservation."""
    reports = [
        report
        for kind in ((record.get("details") or {}).get("executionReports") or {}).values()
        for report in kind
    ]
    planned: dict[str, Any] = max(
        reports, key=lambda report: report.get("reservedBytes") or 0, default={}
    )
    baseline, peak = record["cgroupCurrentBaselineBytes"], record["peakCgroupBytes"]
    return {
        "stage": record["stage"],
        "seconds": record["seconds"],
        "wholeFunctionSeconds": record["wholeFunctionSeconds"],
        "containerBaselineBytes": baseline,
        "containerPeakBytes": peak,
        "containerIncrementBytes": None if None in (baseline, peak) else peak - baseline,
        "processRssBaselineBytes": record["rssBaselineBytes"],
        "processRssPeakBytes": record["peakRssBytes"],
        "plannedUnitKind": planned.get("unitKind"),
        "plannedReservedBytes": planned.get("reservedBytes"),
        "plannedResidentBytes": planned.get("residentBytes"),
        **sampled,
        "anonymousRssBeforeBytes": record["processMemory"]["before"]["anonymousRssBytes"],
        "allocatorFreeHeldBeforeBytes": record["processMemory"]["before"]["allocatorFreeHeldBytes"],
    }


def _run_arm(
    config: Any, configPath: str, arm: str, nRows: int, source: Path, work: Path, order: int
) -> dict[str, Any]:
    spec = ARMS[arm]
    store = work / "stores" / arm / f"{nRows}.zarr"
    prefix = f"{config.resultPrefix(nRows)}/{arm}"
    remote = prefix.startswith("s3://")
    records = work / "records" / arm if remote else Path(prefix.removeprefix("file://"))
    base = {
        "configPath": configPath,
        "nRows": nRows,
        "arm": arm,
        "storeUri": str(store),
        "localH5adPath": str(source),
        "workDirectory": str(work / "scratch" / arm),
        "recordDirectory": str(records),
        "uploadPrefix": prefix if remote else None,
    }
    stages = spec.stagesFor(config, nRows)
    batches = [stages] if spec.sameProcess else [(stage,) for stage in stages]
    environment = {**os.environ, **spec.environment}
    refs: dict[str, dict] = {}
    workers: list[dict[str, Any]] = []
    sampled: dict[str, dict] = {}
    series: list[list[int | None]] = []
    failure = None
    started = time.perf_counter()
    try:
        for batch in batches:
            request = work / "requests" / f"{arm}_{batch[0]}.json"
            request.parent.mkdir(parents=True, exist_ok=True)
            request.write_text(json.dumps({**base, "stages": list(batch), "artifactRefs": refs}))
            print(f"[budget_compliance] {arm}: {', '.join(batch)}", flush=True)
            launched = time.perf_counter()
            worker = subprocess.Popen(
                [sys.executable, "-m", WORKER_MODULE, "--request", str(request)],
                env=environment,
                stderr=subprocess.PIPE,
            )
            assert worker.stderr is not None
            tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
            relay = threading.Thread(target=_relay_stderr, args=(worker.stderr, tail), daemon=True)
            relay.start()
            with WorkerSampler(worker.pid) as sampler:
                code = worker.wait()
            relay.join()
            workers.append(
                {
                    "stages": list(batch),
                    "processId": worker.pid,
                    "returnCode": code,
                    "seconds": time.perf_counter() - launched,
                    "stderrTail": list(tail),
                }
            )
            series.extend(sampler.per_second())
            for stage in batch:
                path = records / f"{stage}.json"
                if path.is_file():
                    record = json.loads(path.read_text())
                    sampled[stage] = sampler.peaks(record["startedAt"], record["finishedAt"])
                    artifact = (record["details"] or {}).get("artifact")
                    if artifact is not None:
                        refs[stage] = artifact
            if code != 0:
                failure = {"returnCode": code, "stages": list(batch), "stderrTail": list(tail)}
                break
    finally:
        if store.parent.exists():
            shutil.rmtree(store.parent)
    _save_record(
        base, "memoryTrace", {"columns": ["second", "processRss", "container"], "rows": series}
    )
    _save_record(base, "workers", {"workers": workers})
    saved = {path.stem: json.loads(path.read_text()) for path in records.glob("*.json")}
    order_of = {stage: index for index, stage in enumerate(stages)}
    return {
        "order": order,
        "environment": spec.environment,
        "status": "failed" if failure else "ok",
        "failure": failure,
        "elapsedSeconds": time.perf_counter() - started,
        "clusters": saved.get("clusters"),
        "release": saved.get("releaseMemory"),
        "workers": workers,
        "stages": [
            stage_summary(saved[stage], sampled[stage])
            for stage in sorted(set(sampled), key=order_of.__getitem__)
        ],
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the requested arms for one sample, budget and worker count."""
    from experiments.benchmarks.config import load_profiling_config

    arms = list(payload["arms"])
    unknown = sorted(set(arms) - set(ARMS))
    if unknown or not arms or len(set(arms)) != len(arms):
        raise ValueError(f"arms must be distinct names from {sorted(ARMS)}; got {arms}")
    config = load_profiling_config(payload["configPath"])
    if config.storeBackend != "modalLocalDisk" or config.countMatrix is None:
        raise ValueError("The comparison needs local-disk stores and an explicit countMatrix")
    if len(config.targetSizes) != 1:
        raise ValueError("The comparison runs one sample size per call")
    nRows = config.targetSizes[0]
    resources = _uniform_resources(
        config, {stage for arm in arms for stage in ARMS[arm].stagesFor(config, nRows)}
    )
    container = _container_limits(payload)
    expected = {
        "memoryLimitBytes": resources.modalMemoryLimitMb * 1024**2,
        "cpuQuotaCores": resources.modalCpuLimit,
    }
    for key, value in expected.items():
        if container[key] is not None and container[key] < value:
            raise ValueError(
                f"The container has {key}={container[key]} ({container['memoryLimitSource']} "
                f"memory limit); the config needs {value}"
            )
    if config.datasetUri(nRows).startswith("s3://") or config.resultsUri.startswith("s3://"):
        os.environ["R2_ENDPOINT_URL"] = config.r2EndpointUrl
    work = Path(payload.get("workDirectory") or f"/tmp/budget_compliance/{config.runTag}")
    source, dataset = _input(config, nRows, work)
    summary: dict[str, Any] = {
        "runTag": config.runTag,
        "nRows": nRows,
        "budgetBytes": resources.cytearcMemoryBudget,
        "workers": resources.workers,
        "countMatrix": config.countMatrix.model_dump(mode="json"),
        "container": container,
        "dataset": dataset,
        "versions": {"cytearc": version("cytearc"), "python": platform.python_version()},
        "sampleIntervalSeconds": SAMPLE_INTERVAL_SECONDS,
        "recordPrefix": config.resultPrefix(nRows),
        "arms": {},
    }
    for order, arm in enumerate(arms):
        summary["arms"][arm] = _run_arm(
            config, payload["configPath"], arm, nRows, source, work, order
        )
    failed = [arm for arm, result in summary["arms"].items() if result["status"] != "ok"]
    summary["status"] = "partial" if failed else "ok"
    if failed:
        last_lines = [
            f"{arm}: {(summary['arms'][arm]['failure']['stderrTail'] or ['no stderr'])[-1]}"
            for arm in failed
        ]
        summary["error"] = f"Arms failed: {'; '.join(last_lines)}"
    uri = f"{config.resultPrefix(nRows)}/summary.json"
    if uri.startswith("s3://"):
        from experiments.runtime.object_io import put_json

        put_json(uri, summary)
    else:
        write_json("summary.json", summary, directory=Path(uri.removeprefix("file://")).parent)
    return summary


CONFIGS = Path(__file__).resolve().parent / "configs" / "benchmark"
RECORDS = Path(__file__).resolve().parent / "records"
STAGES_RECORD = "memory_budget_stages.csv"
# The jobs collected into the record, by configuration, with the arms of each that it holds.
COLLECTED = {
    "embryo_rna_budget_compliance_1000000_cells_8_gib_8_workers_replicate_1": (
        "freshProcess",
        "sameProcess",
    ),
    "embryo_rna_budget_compliance_1000000_cells_8_gib_8_workers_replicate_2": (
        "freshProcess",
        "sameProcess",
    ),
    "embryo_rna_budget_compliance_1000000_cells_8_gib_8_workers_replicate_3": (
        "freshProcess",
        "sameProcess",
    ),
    "embryo_rna_budget_compliance_10000000_cells_8_gib_8_workers_replicate_1": (
        "freshProcess",
        "sameProcess",
    ),
    "embryo_rna_budget_compliance_imported_clusters_10000000_cells_8_gib_8_workers_replicate_1": (
        "freshProcess",
    ),
}
STAGE_COLUMNS = (
    "config",
    "nCells",
    "replicate",
    "budgetBytes",
    "unitBytes",
    "chunkBytes",
    "cpus",
    "containerMemoryLimitBytes",
    "arm",
    "stage",
    "status",
    "seconds",
    "containerPeakBytes",
    "externalContainerPeakBytes",
    "plannedReservedBytes",
    "cellsAfterFilter",
    "estimatedBytes",
    "graphEdges",
    "message",
)
# A graph step that would exceed the budget stops before it allocates, with this message.
REFUSAL = re.compile(
    r"MemoryError: (?P<message>.* of (?P<cells>\d+) cells needs about (?P<estimate>\d+) bytes "
    r"in memory for a graph of (?P<edges>\d+) edges, but the operation limit is \d+ bytes\. "
    r"Raise mem_budget\.)"
)
NOT_SAMPLED = {"externalContainerPeakBytes": None}


def refusal(record: dict[str, Any]) -> dict[str, Any]:
    """The cells, memory estimate and graph edges of a step that refused to start."""
    match = REFUSAL.fullmatch(record.get("error") or "")
    if match is None:
        raise SystemExit(f"{record['stage']} failed without refusing its budget: {record['error']}")
    return {
        "status": "refused",
        "cellsAfterFilter": int(match["cells"]),
        "estimatedBytes": int(match["estimate"]),
        "graphEdges": int(match["edges"]),
        "message": match["message"],
    }


def job_rows(
    name: str,
    config: Any,
    arms: tuple[str, ...],
    summary: dict[str, Any] | None,
    stage_record: Any,
) -> list[dict[str, Any]]:
    """One row per arm and step of one job, from its summary or, without one, its step records.

    ``stage_record(arm, stage)`` returns the record a step saved, or None if the step did not run.
    """
    nRows = config.targetSizes[0]
    resources = _uniform_resources(
        config, {stage for arm in arms for stage in ARMS[arm].stagesFor(config, nRows)}
    )
    job = {
        "config": name,
        "nCells": nRows,
        "replicate": int(name.rsplit("_", 1)[1]),
        "budgetBytes": resources.cytearcMemoryBudget,
        "unitBytes": config.countMatrix.unitBytes,
        "chunkBytes": config.countMatrix.chunkBytes,
        "cpus": resources.modalCpuLimit,
    }
    rows = []
    for arm in arms:
        if summary is not None:
            if summary["budgetBytes"] != job["budgetBytes"]:
                raise SystemExit(f"{name}: the summary's budget differs from the configuration")
            limit = summary["container"]["memoryLimitBytes"]
            steps = summary["arms"][arm]["stages"]
            failure = summary["arms"][arm]["failure"]
            failed = None if failure is None else failure["stages"][-1]
        else:
            records = [
                record
                for stage in ARMS[arm].stagesFor(config, nRows)
                if (record := stage_record(arm, stage)) is not None
            ]
            limit = records[0]["memoryMaxBytes"]
            steps = [stage_summary(record, NOT_SAMPLED) for record in records]
            failed = next((r["stage"] for r in records if r["status"] != "ok"), None)
        for step in steps:
            row = {
                **job,
                "containerMemoryLimitBytes": limit,
                "arm": arm,
                "stage": step["stage"],
                "status": "ok",
                "seconds": step["seconds"],
                "containerPeakBytes": step["containerPeakBytes"],
                "externalContainerPeakBytes": step["externalContainerPeakBytes"],
                "plannedReservedBytes": step["plannedReservedBytes"],
            }
            if step["stage"] == failed:
                row.update(refusal(stage_record(arm, failed)))
            rows.append(row)
    return rows


def collect() -> None:
    """Write the steps of the collected jobs to records/memory_budget_stages.csv."""
    from experiments.benchmarks.config import load_profiling_config
    from experiments.runtime.common import write_csv
    from experiments.runtime.object_io import get_json, object_exists

    rows = []
    for name, arms in COLLECTED.items():
        config = load_profiling_config(CONFIGS / f"{name}.toml")
        prefix = config.resultPrefix(config.targetSizes[0])

        def stage_record(arm: str, stage: str, prefix: str = prefix) -> dict | None:
            uri = f"{prefix}/{arm}/{stage}.json"
            return get_json(uri) if object_exists(uri) else None

        summary_uri = f"{prefix}/summary.json"
        summary = get_json(summary_uri) if object_exists(summary_uri) else None
        rows.extend(job_rows(name, config, arms, summary, stage_record))
        print(f"{name}: {', '.join(arms)}", flush=True)
    write_csv(STAGES_RECORD, rows, directory=RECORDS, fieldnames=STAGE_COLUMNS)
    print(f"wrote {RECORDS / STAGES_RECORD} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=["collect"])
    parser.add_argument("--request", type=Path)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect()
        return
    if args.request is None:
        parser.error("give --request for a worker, or collect")
    run_worker(json.loads(args.request.read_text()))


if __name__ == "__main__":
    main()
