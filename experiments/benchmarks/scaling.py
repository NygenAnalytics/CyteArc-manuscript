import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks import DATA, RAW
from experiments.runtime.common import GIB, ROOT, write_json
from experiments.runtime.records import STAGES, load_runs, save_runs

TAG = os.environ.get("CYTEARC_RUN_TAG", "embryo_rna")
COUNT_POLICIES = {8: (500000000, 50000000)}
CLUSTER_LABEL_STAGES = frozenset({"importClusters", "findMarkers", "validateExperiment"})
PREFIX = TAG
TEMPLATE = ROOT / "experiments" / "benchmarks" / "configs" / "config.toml"
CONFIGS = ROOT / "experiments" / "benchmarks" / "configs" / "benchmark"
BENCHMARK = (
    (10000, "10k", 4, 16, 12),
    (50000, "50k", 4, 16, 12),
    (100000, "100k", 4, 16, 12),
    (500000, "500k", 8, 32, 24),
    (1000000, "1m", 8, 32, 24),
    (5000000, "5m", 16, 64, 48),
    (10000000, "10m", 16, 64, 48),
)
SWEEP = ((8, 4), (8, 8), (16, 4), (16, 8), (24, 4), (24, 8))
BRAIN_CELLS = 1306127
# The records hold replicates 4 to 6 of each configuration.
REPLICATES = (4, 5, 6)


def benchmark_tag(label: str, cpu: int, gib: int, replicate: int) -> str:
    size = next(size for size, name, *_ in BENCHMARK if name == label)
    return f"{TAG}_scaling_{size}_cells_replicate_{replicate}"


def sweep_tag(budget: int, workers: int, replicate: int) -> str:
    return f"{TAG}_memory_{budget}_gib_{workers}_workers_replicate_{replicate}"


def localdisk_tag(replicate: int) -> str:
    return f"{TAG}_1000000_cells_local_disk_replicate_{replicate}"


def brain_tag(replicate: int) -> str:
    return f"mouse_brain_rna_{BRAIN_CELLS}_cells_object_storage_replicate_{replicate}"


@dataclass(frozen=True)
class Run:
    """One profiler run: its tag, bundle group and the envelope of every stage."""

    tag: str
    group: str
    command: str
    size: int
    cpu: int
    gib: int
    budgetGib: int
    workers: int
    replicate: int


def profiling_runs() -> list[Run]:
    runs: list[Run] = []
    for replicate in REPLICATES:
        for size, label, cpu, gib, budget in BENCHMARK:
            runs.append(
                Run(
                    benchmark_tag(label, cpu, gib, replicate),
                    "rna_scaling_runs",
                    "run-e2e",
                    size,
                    cpu,
                    gib,
                    budget,
                    cpu,
                    replicate,
                )
            )
        for budget, workers in SWEEP:
            runs.append(
                Run(
                    sweep_tag(budget, workers, replicate),
                    "rna_resource_runs",
                    "run-e2e",
                    1000000,
                    8,
                    32,
                    budget,
                    workers,
                    replicate,
                )
            )
        runs.append(
            Run(
                localdisk_tag(replicate),
                "rna_resource_runs",
                "run-local",
                1000000,
                8,
                32,
                24,
                8,
                replicate,
            )
        )
    for replicate in REPLICATES:
        runs.append(
            Run(
                brain_tag(replicate),
                "rna_resource_runs",
                "run-e2e",
                BRAIN_CELLS,
                8,
                32,
                24,
                8,
                replicate,
            )
        )
    return runs


def rewrite(text: str, run: Run) -> str:
    text = re.sub("^runTag = .*$", f"runTag = '{run.tag}'", text, flags=re.M)
    text = re.sub("^targetSizes = .*$", f"targetSizes = [{run.size}]", text, flags=re.M)
    if run.budgetGib in COUNT_POLICIES:
        unit, chunk = COUNT_POLICIES[run.budgetGib]
        text = re.sub("^unitBytes = .*$", f"unitBytes = {unit}", text, flags=re.M)
        text = re.sub("^chunkBytes = .*$", f"chunkBytes = {chunk}", text, flags=re.M)
    head, *sections = re.split("(?m)^(?=\\[)", text)
    rebuilt = [head]
    for section in sections:
        if section.startswith(("[stageResources.", "[fixedResources]")):
            for key, value in (
                ("modalMemoryRequestMb", run.gib * 1024),
                ("modalMemoryLimitMb", run.gib * 1024),
                ("modalCpuRequest", float(run.cpu)),
                ("modalCpuLimit", float(run.cpu)),
                ("cytearcMemoryBudget", run.budgetGib * GIB),
                ("workers", run.workers),
            ):
                section = re.sub(f"^{key} = .*$", f"{key} = {value}", section, flags=re.M)
        rebuilt.append(section)
    return "".join(rebuilt)


def attempts(run: Run) -> list[str]:
    return sorted(
        path.stem
        for path in CONFIGS.glob(f"{run.tag}*.toml")
        if path.stem == run.tag or path.stem == run.tag + "_artifact_reuse_sensitivity"
    )


def profiling_config(tag: str):
    from experiments.benchmarks.config import load_profiling_config

    return load_profiling_config(CONFIGS / f"{tag}.toml")


def summarize_stages(config, size: int, stages: dict, elapsed: float | None = None) -> dict:
    outcomes = [stages[name] for name in config.effectiveStages]
    if any(record["status"] != "ok" for record in outcomes):
        raise ValueError("Cannot summarize an incomplete RNA workflow")
    download = sum(record.get("datasetDownloadSeconds", 0.0) for record in outcomes)
    return {
        "status": "ok",
        "runTag": config.runTag,
        "nRows": size,
        "datasetBytes": outcomes[0]["datasetBytes"],
        "storeBackend": "objectStore"
        if config.storeUri(size).startswith("s3://")
        else "modalLocalDisk",
        "datasetDownloadSeconds": download,
        "wholeFunctionSeconds": elapsed
        if elapsed is not None
        else download + sum(record["wholeFunctionSeconds"] for record in outcomes),
        "timingDefinition": "Elapsed sequential workflow including result writes and input download"
        if elapsed is not None
        else "Sum of measured sequential stage calls and input download; excludes scheduling gaps",
        "peakRssBytes": max(record["peakRssBytes"] for record in outcomes),
        "peakCgroupBytes": max(record["peakCgroupBytes"] for record in outcomes),
        "modalResources": {
            "modalCpuLimit": max(record["modalCpuLimit"] for record in outcomes),
            "modalMemoryLimitMb": max(record["modalMemoryMb"] for record in outcomes),
        },
        "outcomes": outcomes,
    }


def collect_runs(group: str) -> dict[str, dict]:
    from experiments.runtime.object_io import get_json, object_exists

    runs = [run for run in profiling_runs() if run.group == group]
    known = {tag for run in runs for tag in attempts(run)}
    bundle = load_runs(group, RAW) if (RAW / f"{group}.json").exists() else {}
    foreign = sorted(set(bundle) - known)
    if foreign:
        raise SystemExit(
            f"data/raw/{group}.json holds runs outside campaign {TAG}: {foreign[:3]} ..."
        )
    for run in runs:
        for tag in attempts(run):
            entry = bundle.get(tag)
            if (
                entry
                and len(entry["stages"]) == len(STAGES)
                and (entry["funnel"] or run.command == "run-local")
            ):
                continue
            config = profiling_config(tag)
            funnel_uri = config.workflowResultUri(run.size)
            funnel = get_json(funnel_uri) if object_exists(funnel_uri) else None
            stages = {}
            for stage in STAGES:
                uri = config.resultUri(run.size, stage)
                if object_exists(uri):
                    stages[stage] = get_json(uri)
            if funnel is None and all(
                name in stages and stages[name]["status"] == "ok" for name in config.effectiveStages
            ):
                funnel = summarize_stages(config, run.size, stages)
            if stages or funnel:
                bundle[tag] = {
                    "funnel": funnel,
                    "stages": stages,
                    "measurementPurpose": "artifact_reuse_sensitivity"
                    if tag.endswith("_artifact_reuse_sensitivity")
                    else "independent_replicate",
                }
                print(f"{tag}: funnel {('yes' if funnel else 'no')}, {len(stages)} stage records")
            else:
                print(f"{tag}: no results yet")
    save_runs(group, bundle, RAW)
    return bundle


def prepare(payload: dict) -> dict:
    from experiments.benchmarks.config import load_profiling_config

    config = load_profiling_config(payload["configPath"])
    os.environ["R2_ENDPOINT_URL"] = config.r2EndpointUrl
    if config.dataset == "mouse_brain":
        from experiments.benchmarks.scanpy import brain_prepare_job

        uri = config.datasetUri(config.targetSizes[0])
        return brain_prepare_job(
            {
                "r2EndpointUrl": config.r2EndpointUrl,
                "datasetUri": uri,
                "outputUri": uri.removesuffix(".h5ad") + ".json",
            }
        )
    if config.dataset == "tahoe":
        from experiments.benchmarks.tahoe import stage as stage_tahoe

        return stage_tahoe(
            {"datasetPrefixUri": config.datasetPrefixUri, **config.tahoe.model_dump(mode="json")}
        )
    if config.inputUri is not None:
        raise ValueError("Input preparation requires datasetPrefixUri, not a supplied inputUri")
    from experiments.benchmarks.datasets import prepare as prepare_embryo

    return prepare_embryo(
        {
            "r2EndpointUrl": config.r2EndpointUrl,
            "datasetPrefixUri": config.datasetPrefixUri,
            "targetRows": list(config.targetSizes),
            "samplingSeed": config.samplingSeed,
        }
    )


def run(payload: dict) -> dict:
    from experiments.benchmarks.config import load_profiling_config
    from experiments.runtime.object_io import put_json

    config = load_profiling_config(payload["configPath"])
    size = payload.get("nRows")
    if size is None and len(config.targetSizes) == 1:
        size = config.targetSizes[0]
    if size not in config.targetSizes:
        raise ValueError("nRows must select one configured targetSizes entry")
    resources = [config.resourcesFor(name) for name in config.effectiveStages]
    payload = {
        "containerMemoryMb": max(item.modalMemoryLimitMb for item in resources),
        "containerCpuRequest": max(item.modalCpuRequest for item in resources),
        "containerCpuLimit": max(item.modalCpuLimit for item in resources),
        **payload,
    }
    # With resourceTrace, a once-a-second trace of the container's memory, CPU and network runs beside the stages
    # and is saved next to the run's summary, also when a stage fails.
    trace = None
    if payload.pop("resourceTrace", False):
        from experiments.runtime.resource_trace import ResourceTrace

        trace = ResourceTrace().start()
    started = time.perf_counter()
    records = {}
    uri = config.workflowResultUri(size)
    try:
        for name in config.effectiveStages:
            if trace is not None:
                trace.mark(name)
            records[name] = stage({**payload, "nRows": size, "stage": name})
    finally:
        if trace is not None:
            put_json(uri.removesuffix("funnel.json") + "resourceTrace.json", trace.stop())
    result = summarize_stages(config, size, records, time.perf_counter() - started)
    if uri.startswith("s3://"):
        put_json(uri, result)
    else:
        path = Path(uri.removeprefix("file://"))
        write_json(path.name, result, directory=path.parent)
    return result


def _configured_stage(payload: dict) -> dict:
    from experiments.benchmarks.config import bind_cluster_source, load_profiling_config
    from experiments.benchmarks.stages import profile_stage_inputs
    from experiments.runtime.object_io import get_json

    config = load_profiling_config(payload["configPath"])
    os.environ["R2_ENDPOINT_URL"] = config.r2EndpointUrl
    size = payload.get("nRows")
    if size is None and len(config.targetSizes) == 1:
        size = config.targetSizes[0]
    if size not in config.targetSizes:
        raise ValueError("nRows must select one configured targetSizes entry")
    name = payload["stage"]
    if name not in config.effectiveStages:
        raise ValueError(f"Stage {name!r} is not selected in the configuration")
    # Only the steps that use imported labels need their source, which another run may still
    # be computing while this run's earlier steps go ahead.
    workflow = (
        bind_cluster_source(config, size, config.effectiveStages)
        if name in CLUSTER_LABEL_STAGES
        else config.workflow
    )
    arguments = {
        "stage": name,
        "submissionId": config.runTag,
        "nRows": size,
        "storeUri": config.storeUri(size),
        "workflow": workflow.model_dump(mode="json"),
        "resources": config.resourcesFor(name).model_dump(mode="json"),
        "recordUri": config.resultUri(size, name),
        "inputRefs": {},
    }
    for key in ("countMatrix", "storageIo"):
        value = getattr(config, key)
        if value is not None:
            arguments[key] = value.model_dump(mode="json")
    if config.objectReads is not None and name in config.objectReads.stages:
        arguments["objectReadAttempts"] = config.objectReads.maximumAttempts
    for key in (
        "localH5adPath",
        "workDir",
        "recordStoreOperations",
        "containerMemoryMb",
        "containerCpuRequest",
        "containerCpuLimit",
    ):
        if key in payload:
            arguments[key] = payload[key]
    if name == "createStore":
        if config.dataset == "tahoe":
            arguments["tahoe"] = {
                "datasetPrefixUri": config.datasetPrefixUri,
                **config.tahoe.model_dump(mode="json"),
            }
        else:
            arguments["datasetUri"] = config.datasetUri(size)
    for input_name, (source_stage, _) in profile_stage_inputs(workflow, name).items():
        uri = config.resultUri(size, source_stage)
        record = (
            get_json(uri)
            if uri.startswith("s3://")
            else json.loads(Path(uri.removeprefix("file://")).read_text())
        )
        if record["status"] != "ok":
            raise ValueError(f"{name} requires successful {source_stage}")
        arguments["inputRefs"][input_name] = record["details"]["artifact"]
    return arguments


def stage(payload: dict) -> dict:
    from cytearc.storage import ArtifactRef

    from experiments.benchmarks.config import (
        CountMatrixConfig,
        StageResources,
        StorageIoConfig,
        WorkflowParameters,
    )
    from experiments.benchmarks.stages import run_stage
    from experiments.runtime.object_io import download_file, put_json

    arguments = _configured_stage(payload) if "configPath" in payload else dict(payload)
    record_uri = arguments.pop("recordUri", None)
    object_read_attempts = arguments.pop("objectReadAttempts", None)
    dataset_uri = arguments.pop("datasetUri", None)
    tahoe = arguments.pop("tahoe", None)
    arguments["workflow"] = WorkflowParameters.model_validate(arguments["workflow"])
    arguments["resources"] = StageResources.model_validate(arguments["resources"])
    for key, model in (("countMatrix", CountMatrixConfig), ("storageIo", StorageIoConfig)):
        if arguments.get(key) is not None:
            arguments[key] = model.model_validate(arguments[key])
    for key in ("localH5adPath", "workDir"):
        if arguments.get(key) is not None:
            arguments[key] = Path(arguments[key])
    if arguments.get("inputRefs") is not None:
        arguments["inputRefs"] = {
            name: ArtifactRef.from_dict(ref) for name, ref in arguments["inputRefs"].items()
        }
    download_seconds = 0.0
    if dataset_uri is not None and arguments.get("localH5adPath") is None:
        if dataset_uri.startswith("s3://"):
            work = arguments.get("workDir") or Path("/tmp/rna_scaling")
            source = work / f"{arguments['nRows']}.h5ad"
            started = time.perf_counter()
            download_file(dataset_uri, source)
            download_seconds = time.perf_counter() - started
        else:
            source = Path(dataset_uri.removeprefix("file://"))
        arguments["localH5adPath"] = source
    if tahoe is None and object_read_attempts is not None:
        from experiments.benchmarks.count_requests import ObjectReadProbe, record_object_reads

        if arguments.get("recordStoreOperations") is False:
            raise ValueError("Counting object reads needs recordStoreOperations")
        arguments["recordStoreOperations"] = True
        probe = ObjectReadProbe(maximumAttempts=object_read_attempts)
        with record_object_reads(probe):
            result = run_stage(**arguments, storeProbe=probe).to_json()
    elif tahoe is None:
        result = run_stage(**arguments).to_json()
    else:
        from experiments.benchmarks.stages import (
            StageRunResult,
            stage_settings,
            summarize_resource_measurement,
        )
        from experiments.benchmarks.tahoe import convert
        from experiments.runtime.metrics import ResourceSampler

        resources = arguments["resources"]
        sampler = ResourceSampler().start()
        started = time.perf_counter()
        try:
            converted = convert(
                {
                    **tahoe,
                    "storeUri": arguments["storeUri"],
                    "workers": resources.workers,
                    "memoryBudget": resources.cytearcMemoryBudget,
                    "countMatrix": arguments["countMatrix"].model_dump(mode="json"),
                }
            )
        finally:
            memory = sampler.stop()
        if converted["cells"] != arguments["nRows"]:
            raise ValueError("Tahoe input cell count differs from the configured size")
        result = StageRunResult(
            submissionId=arguments["submissionId"],
            stage="createStore",
            nRows=arguments["nRows"],
            status="ok",
            seconds=converted["seconds"],
            modalMemoryMb=arguments.get("containerMemoryMb", resources.modalMemoryLimitMb),
            modalCpuRequest=arguments.get("containerCpuRequest", resources.modalCpuRequest),
            modalCpuLimit=arguments.get("containerCpuLimit", resources.modalCpuLimit),
            cytearcMemoryBudget=resources.cytearcMemoryBudget,
            storeUri=arguments["storeUri"],
            wholeFunctionSeconds=time.perf_counter() - started,
            workers=resources.workers,
            settings=stage_settings(
                arguments["workflow"],
                resources,
                countMatrix=arguments.get("countMatrix"),
                storageIo=arguments.get("storageIo"),
            ),
            details={"download": converted["download"]},
            **summarize_resource_measurement(memory),
        ).to_json()
    result["datasetDownloadSeconds"] = download_seconds
    result["placement"] = {
        "region": os.environ.get("MODAL_REGION"),
        "cloudProvider": os.environ.get("MODAL_CLOUD_PROVIDER"),
    }
    if arguments["stage"] == "createStore":
        result["datasetBytes"] = (
            Path(arguments["localH5adPath"]).stat().st_size
            if tahoe is None
            else converted["download"]["dataBytes"]
        )
    if record_uri is not None:
        if record_uri.startswith("s3://"):
            put_json(record_uri, result)
        else:
            path = Path(record_uri.removeprefix("file://"))
            write_json(path.name, result, directory=path.parent)
    else:
        write_json(f"{result['submissionId']}_{result['stage']}.json", result, directory=DATA)
    if result["status"] != "ok":
        raise RuntimeError(result["error"])
    return result
