import hashlib
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from experiments.mapping import HERE

TOOLS = HERE / "tools"


def prepared_inputs(payload: dict, manifest: dict) -> dict:
    """Take input locations and checksums from a prepared input manifest."""
    part = payload.get("referencePart", "reference")
    parts = manifest["parts"]
    if part not in parts or "query" not in parts:
        raise ValueError(f"The input manifest has no {part!r} or query part")
    expected = {
        "datasetUri": parts[part]["uri"],
        "queryUri": parts["query"]["uri"],
        "inputHashes": {"reference": parts[part]["sha256"], "query": parts["query"]["sha256"]},
    }
    for key, value in expected.items():
        if payload.get(key, value) != value:
            raise ValueError(f"{key} in the payload differs from the input manifest")
    payload.update(expected)
    return {
        "inputManifestUri": payload["inputManifestUri"],
        "referencePart": part,
        "referenceCells": parts[part]["cells"],
        "referenceNonzeroCounts": parts[part]["nonzeroCounts"],
        "queryCells": parts["query"]["cells"],
    }


def time_limited(argv: list[str], seconds) -> list[str]:
    if seconds is None:
        return argv
    if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 1:
        raise ValueError(f"workflowTimeoutSeconds must be a positive integer; got {seconds!r}")
    return ["timeout", "--kill-after=120", str(seconds), *argv]


def timed_out(return_code, workflow_seconds: float, seconds) -> bool:
    return seconds is not None and return_code in (124, 137) and workflow_seconds >= seconds


def _run_mapping(payload: dict, argv: list[str]) -> dict:

    from experiments.runtime.harness import R2, Run

    if payload.get("inputMode", "file") != "file":
        raise ValueError("Mapping comparators require prepared H5AD files")
    storage = R2(payload["r2EndpointUrl"])
    inputs = None
    if payload.get("inputManifestUri"):
        inputs = prepared_inputs(payload, storage.get_json(payload["inputManifestUri"]))
    limit = payload.get("workflowTimeoutSeconds")
    argv = time_limited(argv, limit)
    query = Path("/tmp/query/query.h5ad")
    storage.download(payload["queryUri"], query)
    run = Run(payload)
    run.extra["queryUri"] = payload["queryUri"]
    if inputs is not None:
        run.extra["preparedInputs"] = inputs
    environment = {"QUERY_H5AD": str(query)}
    if "labelColumns" in payload:
        columns = payload["labelColumns"]
        if (
            not isinstance(columns, list)
            or not columns
            or any(
                not isinstance(value, str) or not value.strip() or "," in value for value in columns
            )
        ):
            raise ValueError("labelColumns must contain nonempty metadata column names")
        environment["MAPPING_LABEL_COLUMNS"] = ",".join(columns)
    if payload.get("queryBatchColumn"):
        environment["MAPPING_QUERY_BATCH"] = payload["queryBatchColumn"]
    outcome = run.execute(argv, env=environment)
    if "reason" in outcome:
        return outcome
    if timed_out(run.extra.get("returnCode"), run.extra["workflowSeconds"], limit):
        run.status = "timedOut"
        run.extra["workflowTimeoutSeconds"] = limit
    reference = run.work / "data" / Path(urlsplit(payload["datasetUri"]).path.rstrip("/")).name
    input_files = {}
    for name, path, uri in (
        ("reference", reference, payload["datasetUri"]),
        ("query", query, payload["queryUri"]),
    ):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        input_files[name] = {"uri": uri, "sha256": digest, "bytes": path.stat().st_size}
        expected = payload.get("inputHashes", {}).get(name)
        if expected is not None and digest != expected:
            run.status = "error"
            run.extra["inputFiles"] = input_files
            run.extra["inputValidationError"] = (
                f"Prepared {name} checksum differs from its manifest"
            )
            run.publish()
            raise ValueError(run.extra["inputValidationError"])
    run.extra["inputFiles"] = input_files
    run.publish()
    outcome = run.record()
    predictions = run.work / "predictions.csv"
    if predictions.exists():
        run.r2.upload(predictions, payload["outputUri"].replace(".json", "_cells.csv"))
    return outcome


def mapping_python_job(payload: dict) -> dict:
    mode = payload["tool"]
    return _run_mapping(
        payload,
        [
            sys.executable,
            "-m",
            "experiments.mapping.tools.mapping_python",
            mode,
            str(payload["workers"]),
        ],
    )


def mapping_r_job(payload: dict) -> dict:
    env_threads = str(payload["workers"])
    os.environ["OPENBLAS_NUM_THREADS"] = env_threads
    return _run_mapping(
        payload, ["Rscript", str(TOOLS / "mapping_r.R"), payload["tool"], env_threads]
    )
