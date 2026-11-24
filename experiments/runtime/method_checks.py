import importlib
import os
import subprocess
import sys
from pathlib import Path

from experiments.runtime.common import ROOT, is_in_records
from experiments.runtime.locations import results_folder

MODULES = {
    "benchmarks.layout_geometry",
    "benchmarks.planner_admission",
    "mapping.symphony_fixture",
    "integration.wnn_seurat",
    "mapping.mapping_kang",
    "methods.fate_dense",
    "reuse.provenance_reuse",
    "mapping.mapping_reads",
    "integration.harmony_binvignat",
    "integration.harmony_kang",
    "methods.cell_cycle",
    "methods.pba",
    "methods.magic",
    "methods.kang_doublet_score_comparison",
    "integration.lisi",
    "methods.aucell",
    "reuse.overview_thumbnails",
    "integration.harmony_seed_stability",
}
# Checks whose module prepares its input in the same call, with its prepare_inputs function.
PREPARED_INPUTS = {
    "integration.harmony_binvignat",
    "integration.harmony_seed_stability",
    "integration.lisi",
}


def run(payload: dict) -> dict:
    module = payload["module"]
    if module not in MODULES:
        raise ValueError(f"Unknown method check: {module}")
    output = (
        Path(payload["output"]) if "output" in payload else results_folder(module.split(".")[0])
    ).resolve()
    if is_in_records(output):
        raise ValueError("New experiments must not overwrite experiment records")
    if payload.get("r2EndpointUrl"):
        os.environ.setdefault("R2_ENDPOINT_URL", payload["r2EndpointUrl"])
    required_inputs = ()
    if module == "integration.lisi":
        required_inputs = ("harmony_binvignat_comparison.json", "harmony_kang_comparison.json")
    elif module == "benchmarks.layout_geometry":
        required_inputs = ("raw/rna_scaling_runs.json",)
    if required_inputs:
        from experiments.runtime.object_io import download_file

        for name in required_inputs:
            if (output / name).exists():
                continue
            uri = payload.get("inputs", {}).get(name)
            if not uri:
                raise ValueError(
                    f"{module} requires {name}; provide its saved URI in inputs or place it in output"
                )
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            download_file(uri, output / name)
    if module in PREPARED_INPUTS:
        importlib.import_module(f"experiments.{module}").prepare_inputs(payload)
    command = [sys.executable, "-m", f"experiments.{module}", *payload.get("arguments", [])]
    subprocess.run(
        command, cwd=ROOT, env={**os.environ, "CYTEARC_RESULTS": str(output)}, check=True
    )
    if payload.get("resultsUri"):
        from experiments.runtime.object_io import upload_file

        prefix = payload["resultsUri"].rstrip("/")
        files = []
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(output).as_posix()
            upload_file(path, f"{prefix}/{name}")
            files.append(name)
        if not files:
            raise RuntimeError(f"{module} produced no files in {output}")
        return {
            "status": "ok",
            "module": module,
            "resultsUri": prefix,
            "files": files,
            "placement": placement(),
        }
    return {"status": "ok", "module": module, "output": str(output), "placement": placement()}


def placement() -> dict:
    """The region and cloud the job's container ran in, as Modal reports them."""
    return {
        "region": os.environ.get("MODAL_REGION"),
        "cloudProvider": os.environ.get("MODAL_CLOUD_PROVIDER"),
    }
