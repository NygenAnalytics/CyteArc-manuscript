import argparse
import hashlib
import importlib
import io
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path

import modal

MODULES = {
    "lung.lung_pipeline",
    "lung.lung_robustness",
    "lung.reference_revision",
    "benchmarks.workflows",
    "benchmarks.count_requests",
    "benchmarks.datasets",
    "mapping.comparators",
    "mapping.mapping_reruns",
    "reuse.mounting",
    "reuse.composite_and_remote",
    "reuse.blood_cohort_analysis",
    "reuse.blood_corpus",
    "reuse.blood_progenitors",
    "reuse.census_query",
    "mapping.blood_store_annotation",
    "benchmarks.budget_compliance",
    "benchmarks.scanpy",
    "mapping.hlca",
    "mapping.cross_study_mapping",
    "methods.cellrank",
    "benchmarks.cluster_quality",
    "runtime.method_checks",
    "benchmarks.tahoe",
    "benchmarks.scaling",
    "catalogue.cohort_discovery",
    "catalogue.catalogue_screen",
    "methods.lsi",
    "methods.rna_numerical_reproducibility",
    "methods.gene_scores",
    "methods.pca_drift",
    "methods.regulatory",
    "methods.aucell",
    "methods.droplet_qc",
    "integration.multimodal_neighbors",
    "integration.multimodal_agreement",
    "integration.t_cell_subtypes",
}
RECORDED_MODULES = {
    "benchmarks.count_requests",
    "methods.aucell",
    "methods.droplet_qc",
    "methods.lsi",
    "methods.rna_numerical_reproducibility",
    "methods.gene_scores",
    "methods.pca_drift",
    "methods.regulatory",
    "integration.multimodal_neighbors",
    "integration.multimodal_agreement",
    "integration.t_cell_subtypes",
    "mapping.cross_study_mapping",
    "reuse.composite_and_remote",
    "reuse.blood_cohort_analysis",
    "reuse.blood_corpus",
    "reuse.blood_progenitors",
    "mapping.blood_store_annotation",
    "benchmarks.budget_compliance",
    "catalogue.catalogue_screen",
}
RECORDED_FUNCTIONS = {
    ("benchmarks.datasets", "prepare"),
    ("reuse.census_query", "run"),
}


def package_source() -> Path:
    from experiments.runtime.common import ROOT
    from experiments.runtime.locations import runs_root

    buffer = io.BytesIO()
    # Containers get the code and settings; experiment records stay on this machine.
    excluded = {"records", "__pycache__"}
    suffixes = {".py", ".R", ".json", ".txt", ".in", ".toml"}
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for directory, subdirectories, names in os.walk(ROOT / "experiments"):
            subdirectories[:] = sorted(set(subdirectories) - excluded)
            for name in sorted(names):
                path = Path(directory) / name
                if path.is_file() and path.suffix in suffixes:
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
    content = buffer.getvalue()
    # Each launch's source archive is kept with the runs, named by its hash.
    target = runs_root() / "harness" / f"source_{hashlib.sha256(content).hexdigest()}.tar"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        try:
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
    return target


def validate(module: str, function: str, payload: dict) -> None:
    if module not in MODULES or function.startswith("_"):
        raise ValueError(f"Unknown experiment entry point: {module}.{function}")
    if (module in RECORDED_MODULES or (module, function) in RECORDED_FUNCTIONS) and not payload.get(
        "recordUri"
    ):
        raise ValueError("Recorded experiments require recordUri to preserve their results")
    if (module, function) == ("benchmarks.scaling", "stage") and not (
        payload.get("recordUri") or payload.get("configPath")
    ):
        raise ValueError("Stage runs require configPath or recordUri to save their measurements")
    if (module, function) == ("benchmarks.tahoe", "convert") and not payload.get("recordUri"):
        raise ValueError("Remote stage runs require recordUri to preserve the measurement record")
    if module == "runtime.method_checks":
        if function != "run":
            raise ValueError(
                "Remote method checks use runtime.method_checks.run; inputs are prepared in that call"
            )
        if not payload.get("resultsUri"):
            raise ValueError(
                "Remote method checks require resultsUri to preserve their generated files"
            )


def execute(module: str, function: str, payload: dict, bucket: dict[str, str]):
    os.environ.update(bucket)
    validate(module, function, payload)
    os.environ["CYTEARC_CALL_ID"] = modal.current_function_call_id()
    os.environ["CYTEARC_INPUT_ID"] = modal.current_input_id()
    if payload.get("r2EndpointUrl"):
        os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    if module in RECORDED_MODULES or (module, function) in RECORDED_FUNCTIONS:
        from experiments.runtime.harness import R2

        loaded = importlib.import_module(f"experiments.{module}")
        storage = R2(payload["r2EndpointUrl"])
        started = time.time()
        context = {
            "module": module,
            "function": function,
            "callId": os.environ["CYTEARC_CALL_ID"],
            "region": os.environ.get("MODAL_REGION"),
            "startedAt": started,
        }
        storage.put_json(payload["recordUri"], {**context, "status": "running"})
        try:
            result = getattr(loaded, function)(payload)
            if isinstance(result, dict) and result.get("status") in {"error", "failed", "partial"}:
                raise RuntimeError(result.get("error") or f"{module}.{function} failed: {result}")
        except Exception as error:
            saved = storage.get_json(payload["recordUri"])
            storage.put_json(
                payload["recordUri"],
                {
                    **saved,
                    **context,
                    "status": "failed",
                    "errorType": type(error).__name__,
                    "error": str(error),
                    "elapsedSeconds": time.time() - started,
                },
            )
            raise
        storage.put_json(
            payload["recordUri"],
            {
                **context,
                "status": "completed",
                "result": result,
                "elapsedSeconds": time.time() - started,
            },
        )
    else:
        operation = getattr(importlib.import_module(f"experiments.{module}"), function)
        result = operation(payload)
    if isinstance(result, dict) and result.get("status") in {"error", "failed", "partial"}:
        raise RuntimeError(result.get("error") or f"{module}.{function} failed: {result}")
    return result


def execution_resources(module: str, function: str, payload: dict) -> dict:
    if module != "benchmarks.scaling" or "configPath" not in payload:
        return {"cpu": (8, 8), "memory": (32768, 32768), "timeout": 86400, "ephemeral_disk": 524288}
    from experiments.benchmarks.config import load_profiling_config

    config = load_profiling_config(payload["configPath"])
    if function == "prepare":
        resources = [config.prepareResources]
    elif function == "stage":
        resources = [config.resourcesFor(payload["stage"])]
    elif function == "run":
        resources = [config.resourcesFor(stage) for stage in config.effectiveStages]
    else:
        raise ValueError(f"No configured resource allocation for {function}")
    return {
        "cpu": (max(r.modalCpuRequest for r in resources), max(r.modalCpuLimit for r in resources)),
        "memory": (
            max(r.modalMemoryRequestMb for r in resources),
            max(r.modalMemoryLimitMb for r in resources),
        ),
        "timeout": max(r.timeoutSeconds for r in resources),
        "ephemeral_disk": max(r.ephemeralDiskMb for r in resources),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("module", choices=sorted(MODULES))
    run.add_argument("function")
    run.add_argument("payload", type=Path)
    run.add_argument("--image", default="cytearc")
    run.add_argument("--cpu", type=int)
    run.add_argument("--memory-gib", type=int)
    run.add_argument("--timeout", type=int)
    run.add_argument("--disk-gib", type=int, help="ephemeral disk in GiB, in place of the default")
    run.add_argument(
        "--region",
        help="Modal region to pin; omitted, Modal schedules the job without a region constraint",
    )
    run.add_argument("--environment")
    run.add_argument("--secret")
    run.add_argument("--gpu")
    result = commands.add_parser("result")
    result.add_argument("call_id")
    args = parser.parse_args()
    if args.command == "result":
        print(json.dumps(modal.FunctionCall.from_id(args.call_id).get(), default=str))
        return

    from experiments.images import IMAGES
    from experiments.runtime.locations import BUCKET_VARIABLES

    if args.image not in IMAGES:
        parser.error(f"Unknown image {args.image}; choose from {', '.join(IMAGES)}")
    payload = json.loads(args.payload.read_text())
    if not isinstance(payload, dict):
        raise TypeError("The payload must be a JSON object")
    validate(args.module, args.function, payload)
    resources = execution_resources(args.module, args.function, payload)
    configured = args.module == "benchmarks.scaling" and "configPath" in payload
    for key, value in (("cpu", args.cpu), ("memory", args.memory_gib)):
        if value is not None:
            pair = (value, value) if key == "cpu" else (value * 1024, value * 1024)
            if configured and pair != resources[key]:
                parser.error(f"--{key} conflicts with the experiment configuration")
            resources[key] = pair
    if args.timeout is not None:
        resources["timeout"] = args.timeout
    if args.disk_gib is not None:
        if configured:
            parser.error("--disk-gib conflicts with the experiment configuration")
        resources["ephemeral_disk"] = args.disk_gib * 1024
    if configured:
        from experiments.benchmarks.config import load_profiling_config

        config = load_profiling_config(payload["configPath"])
        args.environment = args.environment or config.modalEnvironmentName
        args.secret = args.secret or config.modalSecretName
        payload.update(
            containerMemoryMb=resources["memory"][1],
            containerCpuRequest=resources["cpu"][0],
            containerCpuLimit=resources["cpu"][1],
        )
    source = package_source()
    # The experiment bucket's endpoint and name travel from this machine's environment to the job.
    bucket = {name: os.environ[name] for name in BUCKET_VARIABLES if os.environ.get(name)}
    execution_image = (
        IMAGES[args.image]
        .add_local_file(source, "/tmp/experiment-source.tar", copy=True)
        .run_commands(
            "mkdir -p /workspace && tar -xf /tmp/experiment-source.tar -C /workspace "
            "&& rm /tmp/experiment-source.tar"
        )
    )
    secrets = (
        [modal.Secret.from_name(args.secret, environment_name=args.environment)]
        if args.secret
        else []
    )
    app = modal.App("cytearc-experiments")
    operation = app.function(
        name="execute",
        include_source=False,
        image=execution_image,
        secrets=secrets,
        **resources,
        region=args.region,
        nonpreemptible=not bool(args.gpu),
        max_containers=1,
        scaledown_window=2,
        retries=0,
        gpu=args.gpu,
    )(execute)
    with modal.enable_output(), app.run(detach=True, environment_name=args.environment):
        call = operation.spawn(args.module, args.function, payload, bucket)
        print(
            json.dumps(
                {
                    "appId": app.app_id,
                    "callId": call.object_id,
                    "module": args.module,
                    "function": args.function,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
