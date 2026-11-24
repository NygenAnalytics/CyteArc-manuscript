import json
import sys
import time
from pathlib import Path

from experiments.benchmarks import HERE
from experiments.runtime.image_records import image_record

TOOLS = HERE / "tools"

GIB = 1024**3

CENSUS_VERSION = "2025-11-08"

EMBRYO_DATASET_ID = "dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3"

CYTEARC_BUDGET_SHARE = 0.75

TOOLS_BY_NAME = {
    "scanpy": "python_tool_job",
    "scanpy_dask_materialized_pca": "python_tool_job",
    "seurat": "r_tool_job",
    "seurat_bpcells": "r_tool_job",
    "bpcells": "r_tool_job",
    "seurat_bpcells_spca": "r_tool_job",
    "bpcells_spca": "r_tool_job",
    "bpcells_cached": "r_tool_job",
    "bpcells_cached_spca": "r_tool_job",
    "bpcells_snn": "r_tool_job",
    "bpcells_igraph": "r_tool_job",
    "scanvi": "mapping_python_job",
    "celltypist": "mapping_python_job",
    "celltypist_hvg": "mapping_python_job",
    "seurat_map": "mapping_r_job",
    "symphony": "mapping_r_job",
    "census_scanpy": "census_tool_job",
    "cytearc": "cytearc_tool_job",
    "cytearc_limit4": "cytearc_tool_job",
    "cytearc_limit8": "cytearc_tool_job",
    "cytearc_limit16": "cytearc_tool_job",
    "scanpy_csc": "python_tool_job",
    "scanpy_csc_pcainit": "python_tool_job",
    "scanpy_dask_covariance_csr": "python_tool_job",
    "scanpy_dask_covariance_csc": "python_tool_job",
    "scanpy_dask_covariance_csc_local_zarr": "python_zarr_tool_job",
    "scanpy_dask_covariance_csc_remote_zarr": "python_zarr_tool_job",
    "cytearc_matched": "cytearc_tool_job",
    "markers_cytearc": "cytearc_tool_job",
    "markers_scanpy_csr": "python_tool_job",
    "markers_scanpy_csc": "python_tool_job",
    "markers_bpcells": "r_tool_job",
    "markers_presto": "r_tool_job",
}

SCANPY_MODES = {
    "scanpy": "memory",
    "scanpy_dask_materialized_pca": "dask_materialized_pca",
    "scanpy_csc": "memory_csc",
    "scanpy_csc_pcainit": "memory_csc_pcainit",
    "scanpy_dask_covariance_csr": "dask_covariance_csr",
    "scanpy_dask_covariance_csc": "dask_covariance_csc",
    "scanpy_dask_covariance_csc_local_zarr": "dask_covariance_csc",
    "scanpy_dask_covariance_csc_remote_zarr": "dask_covariance_csc",
}

ZARR_CHUNK_VALUES = 4 * 1024 * 1024

MARKER_TOOLS = (
    "markers_cytearc",
    "markers_scanpy_csr",
    "markers_scanpy_csc",
    "markers_bpcells",
    "markers_presto",
)

EMBRYO_SIZES = (10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000)

EXTERNAL = {
    "tabula_sapiens": {
        "title": "Tabula Sapiens - All Cells",
        "collectionId": "e5f58829-1a66-40b5-a624-9046778e74f5",
        "datasetId": "53d208b0-2cfd-4366-9866-c3c6114081bc",
        "datasetVersionId": "1c88f927-bcbb-4bb1-9881-281842945a2d",
        "cells": 1136218,
    },
    "hbca_neurons": {
        "title": "Human Brain Cell Atlas v1.0 - All neurons",
        "collectionId": "283d65eb-dd53-496d-adb7-7570c7caa443",
        "datasetId": "8e10f1c4-8e98-41e5-b65f-8cd89a887122",
        "datasetVersionId": "937f8304-300e-46ff-a84c-28df40b59d2b",
        "cells": 2480956,
    },
    "mssm_cohort": {
        "title": "MSSM_Cohort",
        "collectionId": "84ce6837-548d-4a1f-919f-0bc0d9a3952f",
        "datasetId": "37a17b78-4864-4a42-b67b-31c00962795a",
        "datasetVersionId": "0e853475-e298-4b09-881a-ed0b60d5a8c9",
        "cells": 4140453,
    },
}


def _execute(payload: dict, argv: list[str], env: dict | None = None) -> dict:

    from experiments.runtime.harness import Run

    run = Run(payload)
    outcome = run.execute(argv, env=env)
    if outcome.get("status") != "ok":
        raise RuntimeError(outcome.get("error") or f"{payload['tool']} failed: {outcome}")
    sample = run.work / "sample.csv"
    if sample.exists():
        run.r2.upload(sample, payload["outputUri"].replace(".json", "_sample.csv"))
    if payload.get("keepOutputs"):
        _upload_outputs(run, payload)
    return outcome


def _keep_outputs_env(payload: dict, env: dict | None = None) -> dict | None:
    if not payload.get("keepOutputs"):
        return env
    return {**(env or {}), "KEEP_OUTPUTS": "1"}


def _upload_outputs(run, payload: dict) -> None:
    from experiments.benchmarks.tools.kept_outputs import FILE_NAME

    outputs = run.work / FILE_NAME
    if not outputs.exists():
        raise FileNotFoundError(outputs)
    uri = payload["outputUri"].replace(".json", "_outputs.h5")
    run.r2.upload(outputs, uri)
    run.extra["keptOutputs"] = {"name": uri.rsplit("/", 1)[-1], "bytes": outputs.stat().st_size}
    run.publish()


def _run_python_tool(payload: dict) -> dict:
    tool, workers = (payload["tool"], str(payload["workers"]))
    env = {"VERIFY_MARKERS": "1"} if payload.get("verifyMarkers") else None
    env = _keep_outputs_env(payload, env)
    if payload.get("inputMode") == "remote":
        env = {**(env or {}), "R2_ENDPOINT_URL": payload["r2EndpointUrl"]}
    if payload.get("workflowSettings"):
        if not SCANPY_MODES.get(tool, "").startswith("dask"):
            raise ValueError(f"workflowSettings apply to the Scanpy (Dask) tools, not {tool}")
        env = {**(env or {}), "WORKFLOW_SETTINGS": json.dumps(payload["workflowSettings"])}
    if tool in SCANPY_MODES:
        return _execute(
            payload,
            [
                sys.executable,
                "-m",
                "experiments.benchmarks.tools.workflow_scanpy",
                SCANPY_MODES[tool],
                workers,
            ],
            env,
        )
    return _execute(
        payload,
        [
            sys.executable,
            "-m",
            "experiments.benchmarks.tools.markers_scanpy",
            tool.removeprefix("markers_scanpy_"),
            workers,
        ],
        env,
    )


def python_tool_job(payload: dict) -> dict:
    return _run_python_tool(payload)


def python_zarr_tool_job(payload: dict) -> dict:
    return _run_python_tool(payload)


def zarr_prepare_job(payload: dict) -> dict:

    from importlib import metadata

    import anndata as ad
    import zarr

    from experiments.runtime.harness import R2

    r2 = R2(payload["r2EndpointUrl"])
    work = Path("/tmp/zarr_prepare")
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.h5ad"
    started = time.time()
    r2.download(payload["sourceUri"], source)
    download_seconds = time.time() - started
    started = time.time()
    adata = ad.read_h5ad(source)
    target = work / "copy.zarr"
    root = zarr.open_group(target, mode="w")
    ad.io.write_elem(root, "X", adata.X, dataset_kwargs={"chunks": (ZARR_CHUNK_VALUES,)})
    ad.io.write_elem(root, "obs", adata.obs)
    ad.io.write_elem(root, "var", adata.var)
    write_seconds = time.time() - started
    started = time.time()
    stored = r2.upload_dir(target, payload["zarrUri"])
    record = {
        "dataset": payload["dataset"],
        "nCells": int(adata.n_obs),
        "nGenes": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "xFormat": type(adata.X).__name__,
        "chunkValues": ZARR_CHUNK_VALUES,
        "zarrFormat": int(root.metadata.zarr_format),
        "storedBytes": stored,
        "objects": sum((1 for item in target.rglob("*") if item.is_file())),
        "versions": {"anndata": metadata.version("anndata"), "zarr": metadata.version("zarr")},
        "downloadSeconds": download_seconds,
        "writeSeconds": write_seconds,
        "uploadSeconds": time.time() - started,
        "finishedAt": time.time(),
    }
    r2.put_json(payload["provenanceUri"], record)
    return record


def _run_r_tool(payload: dict) -> dict:
    tool, workers = (payload["tool"], str(payload["workers"]))
    env = _keep_outputs_env(
        payload,
        {"OPENBLAS_NUM_THREADS": workers, "OMP_NUM_THREADS": workers, "PYTHON": sys.executable},
    )
    if payload.get("workflowSettings"):
        if not tool.startswith("bpcells"):
            raise ValueError(
                f"workflowSettings apply to the bpcells arms among the R tools, not {tool}"
            )
        env["WORKFLOW_SETTINGS"] = json.dumps(payload["workflowSettings"])
    if tool in MARKER_TOOLS:
        return _execute(
            payload,
            ["Rscript", str(TOOLS / "markers_r.R"), tool.removeprefix("markers_"), workers],
            env,
        )
    return _execute(payload, ["Rscript", str(TOOLS / "workflow_r.R"), tool, workers], env)


def r_tool_job(payload: dict) -> dict:
    return _run_r_tool(payload)


def _run_cytearc_tool(payload: dict) -> dict:
    tool, workers = (payload["tool"], str(payload["workers"]))
    budget = int(payload["containerMemoryMb"] * 1024 * 1024 * CYTEARC_BUDGET_SHARE)
    if tool.startswith("cytearc_limit"):
        budget = int(tool.removeprefix("cytearc_limit")) * GIB
    if tool == "markers_cytearc":
        return _execute(
            payload,
            [
                sys.executable,
                "-m",
                "experiments.benchmarks.tools.markers_cytearc",
                workers,
                str(budget),
            ],
        )
    mode = "matched" if tool == "cytearc_matched" else "standard"
    env = {"COUNT_POLICY": payload["countPolicy"]} if payload.get("countPolicy") else None
    if payload.get("stopAfter") is not None:
        if payload["stopAfter"] != "runPca":
            raise ValueError("The preprocessing measurement stops after runPca")
        env = {**(env or {}), "STOP_AFTER": payload["stopAfter"]}
    return _execute(
        payload,
        [
            sys.executable,
            "-m",
            "experiments.benchmarks.tools.workflow_cytearc",
            mode,
            workers,
            str(budget),
        ],
        _keep_outputs_env(payload, env),
    )


def cytearc_tool_job(payload: dict) -> dict:
    return _run_cytearc_tool(payload)


def markers_prepare_job(payload: dict) -> dict:
    import traceback

    from experiments.benchmarks.tools.markers_common import prepare
    from experiments.runtime.harness import R2

    r2 = R2(payload["r2EndpointUrl"])
    if r2.exists(payload["commonUri"]) and r2.exists(payload["provenanceUri"]):
        return {"status": "existing"}
    record: dict = {
        "experiment": "markersCommon",
        "dataset": payload["dataset"],
        "datasetUri": payload["datasetUri"],
        "commonUri": payload["commonUri"],
        "groupsUri": payload["groupsUri"],
        "container": {
            "cpu": payload["containerCpu"],
            "memoryMb": payload["containerMemoryMb"],
            "preemptible": False,
        },
        "image": payload["image"],
        "status": "running",
        "startedAt": time.time(),
    }
    r2.put_json(payload["recordUri"], record)
    try:
        work = Path("/tmp/markers")
        started = time.time()
        r2.download(payload["datasetUri"], work / "source.h5ad")
        record["downloadSeconds"] = time.time() - started
        made = prepare(work / "source.h5ad", work, int(payload["workers"]))
        started = time.time()
        r2.upload(made["common"], payload["commonUri"])
        r2.upload(made["groupsTable"], payload["groupsUri"])
        record["uploadSeconds"] = time.time() - started
        record.update(made["facts"])
        record["status"] = "ok"
    except BaseException:
        record["status"] = "error"
        record["error"] = traceback.format_exc()[-4000:]
    record["finishedAt"] = time.time()
    r2.put_json(payload["recordUri"], record)
    if record["status"] == "ok":
        r2.put_json(payload["provenanceUri"], record)
    return {
        "status": record["status"],
        "cellsAfterFilter": record.get("cellsAfterFilter"),
        "groups": record.get("groups"),
    }


def prepare_job(payload: dict) -> dict:
    import urllib.request

    from experiments.benchmarks.tools.prepare_cellxgene import convert
    from experiments.runtime.h5ad import sha256
    from experiments.runtime.harness import R2

    r2 = R2(payload["r2EndpointUrl"])
    if r2.exists(payload["datasetUri"]) and r2.exists(payload["provenanceUri"]):
        return {"status": "existing"}
    work = Path("/tmp/prepare")
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.h5ad"
    started = time.time()
    with urllib.request.urlopen(payload["sourceUrl"]) as response, source.open("wb") as handle:
        while chunk := response.read(64 * 1024**2):
            handle.write(chunk)
    download_seconds = time.time() - started
    facts = convert(source, work / "prepared.h5ad")
    r2.upload(work / "prepared.h5ad", payload["datasetUri"])
    record = {
        **payload["source"],
        "sourceUrl": payload["sourceUrl"],
        "sourceBytes": source.stat().st_size,
        "sourceSha256": sha256(source),
        "downloadSeconds": download_seconds,
        "datasetUri": payload["datasetUri"],
        **facts,
        "preparedAt": time.time(),
    }
    r2.put_json(payload["provenanceUri"], record)
    return {"status": "ok", **facts}


def stage_job(payload: dict) -> dict:
    import hashlib
    import urllib.request

    import anndata as ad
    import pandas as pd

    from experiments.runtime.harness import R2

    r2 = R2(payload["r2EndpointUrl"])
    if r2.exists(payload["datasetUri"]) and r2.exists(payload["provenanceUri"]):
        return {"status": "existing"}
    work = Path("/tmp/stage")
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.h5ad"
    digest = hashlib.sha256()
    started = time.time()
    with urllib.request.urlopen(payload["sourceUrl"]) as response, source.open("wb") as handle:
        while chunk := response.read(64 * 1024**2):
            handle.write(chunk)
            digest.update(chunk)
    download_seconds = time.time() - started
    adata = ad.read_h5ad(source, backed="r")
    obs = adata.obs
    tables = {}
    group = [c for c in payload["groupBy"] if c in obs.columns]
    tables["samples"] = obs.groupby(group, observed=True).size().rename("cells").reset_index()
    if payload.get("labels") in obs.columns:
        keys = [c for c in ("donor_id", "Sample_Type") if c in obs.columns]
        tables["labels"] = (
            obs.groupby([*keys, payload["labels"]], observed=True)
            .size()
            .rename("cells")
            .reset_index()
        )
    names = (
        adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in adata.var.columns
        else [""] * adata.n_vars
    )
    tables["var"] = pd.DataFrame({"feature_id": adata.var_names.astype(str), "feature_name": names})
    prefix = payload["datasetUri"].removesuffix(".h5ad")
    for name, frame in tables.items():
        path = work / f"{name}.csv"
        frame.to_csv(path, index=False)
        r2.upload(path, f"{prefix}_{name}.csv")
    r2.upload(source, payload["datasetUri"])
    record = {
        **payload["source"],
        "sourceUrl": payload["sourceUrl"],
        "sourceBytes": source.stat().st_size,
        "sourceSha256": digest.hexdigest(),
        "downloadSeconds": download_seconds,
        "nObs": int(adata.n_obs),
        "nVars": int(adata.n_vars),
        "obsColumns": [str(c) for c in obs.columns],
        "groupBy": group,
        "tables": sorted(tables),
        "stagedAt": time.time(),
    }
    r2.put_json(payload["provenanceUri"], record)
    return {"status": "ok", "nObs": record["nObs"], "nVars": record["nVars"]}


def census_tool_job(payload: dict) -> dict:

    from experiments.runtime.harness import Run

    env = {
        "CENSUS_VERSION": CENSUS_VERSION,
        "CENSUS_DATASET_ID": EMBRYO_DATASET_ID,
        "CENSUS_CELLS": str(payload["censusCells"]),
    }
    return Run(payload).execute(
        [
            sys.executable,
            "-m",
            "experiments.benchmarks.tools.workflow_scanpy",
            "census",
            str(payload["workers"]),
        ],
        env=env,
    )


def diagnose_read_job(payload: dict) -> dict:
    import subprocess

    from experiments.runtime.harness import R2

    report: dict = {}
    for path in (
        "/sys/fs/cgroup/memory.stat",
        "/proc/self/smaps_rollup",
        "/sys/fs/cgroup/memory.peak",
    ):
        try:
            report[path] = Path(path).read_text()[:1500]
        except OSError as exc:
            report[path] = f"unavailable: {exc}"
    local = Path("/tmp/sample.h5ad")
    R2(payload["r2EndpointUrl"]).download(payload["datasetUri"], local)
    gzip_run = subprocess.run(
        ["Rscript", str(TOOLS / "diagnose_read.R"), str(local)], capture_output=True, text=True
    )
    report["gzipInput"] = gzip_run.stderr[-2000:]
    started = time.time()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.benchmarks.tools.uncompress_h5ad",
            str(local),
            "/tmp/plain.h5ad",
        ],
        check=True,
    )
    report["uncompressSeconds"] = time.time() - started
    plain_run = subprocess.run(
        ["Rscript", str(TOOLS / "diagnose_read.R"), "/tmp/plain.h5ad"],
        capture_output=True,
        text=True,
    )
    report["plainInput"] = plain_run.stderr[-2000:]
    R2(payload["r2EndpointUrl"]).put_json(payload["outputUri"], report)
    return report


def _image_record(tool: str) -> dict:
    job = TOOLS_BY_NAME[tool]
    if job != "cytearc_tool_job":
        return image_record(job)
    limit = (
        {"memoryLimitGib": int(tool.removeprefix("cytearc_limit"))}
        if tool.startswith("cytearc_limit")
        else {"budgetShare": CYTEARC_BUDGET_SHARE}
    )
    return {**limit, **image_record(job)}


def _dataset_uri(config: dict, dataset: str) -> str:
    prefix = config["datasetPrefixUri"].rstrip("/")
    if dataset.startswith("embryo_"):
        size = int(dataset.split("_")[1])
        if size not in EMBRYO_SIZES:
            raise SystemExit(f"no embryo sample of {size} cells")
        return f"{prefix}/{size}.h5ad"
    if dataset == "brain_1306127":
        return f"{prefix}/1306127.h5ad"
    if dataset in EXTERNAL:
        return f"{prefix}/external/{dataset}.h5ad"
    if dataset == "hlca_heldout":
        return f"{prefix}/external/hlca_reference.h5ad"
    if dataset.startswith("census_"):
        return f"{prefix}/10000.h5ad"
    raise SystemExit(f"unknown dataset {dataset}")


def _zarr_uri(config: dict, dataset: str) -> str:
    if not dataset.startswith("embryo_"):
        raise SystemExit("the Zarr copies are of embryo samples, embryo_<cells>")
    _dataset_uri(config, dataset)
    return f"{config['datasetPrefixUri'].rstrip('/')}/zarr/{int(dataset.split('_')[1])}.zarr"


def _markers_uris(config: dict, dataset: str) -> dict:
    prefix = config["datasetPrefixUri"].rstrip("/") + "/markers"
    return {
        "commonUri": f"{prefix}/{dataset}.h5ad",
        "groupsUri": f"{prefix}/{dataset}_groups.csv.gz",
        "provenanceUri": f"{prefix}/{dataset}.json",
    }


def _config() -> dict:
    from experiments.benchmarks.config import TEMPLATE, load_profiling_config

    return load_profiling_config(TEMPLATE).model_dump(mode="json")
