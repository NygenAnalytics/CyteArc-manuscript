import re
import statistics

from experiments.benchmarks.tool_summary import require_verified_workflows
from experiments.runtime.common import DERIVED, load_json
from manuscript.tables.table_values import write_table

BENCHMARKS_DERIVED = DERIVED / "benchmarks"

TOOLS_SUMMARY = ".build/derived/benchmarks/toolkit_comparisons.json"
TOOL_ROWS = load_json(BENCHMARKS_DERIVED / "toolkit_comparisons.json")["workflow"]
CENSUS = {"census_1000000": "embryo_1000000", "census_10000000": "embryo_10000000"}
DATASETS = (
    ("embryo_1000000", "Embryo", "", 8),
    ("brain_1306127", "Mouse", "brain", 8),
    ("tabula_sapiens", "Tabula", "Sapiens", 8),
    ("hbca_neurons", "Brain-atlas", "neurons", 16),
    ("mssm_cohort", "PFC", "cohort", 16),
    ("embryo_10000000", "Embryo", "", 16),
)
MEMORY_FAILURES = ("killed", "outOfMemory", "killed (record incomplete)")
TOOLKIT_ARMS = (
    ("cytearc", "CyteArc", "24 GiB budget"),
    ("bpcells_cached", "BPCells", ""),
    ("seurat_bpcells", "Seurat", "(BPCells)"),
    ("seurat", "Seurat", "(memory)"),
    ("scanpy", "Scanpy", "(memory)"),
    ("scanpy_dask_covariance_csc", "Scanpy", "(Dask)"),
    ("census_scanpy", "Scanpy", "(Census)"),
)


def tool_rows(census_as_embryo: bool) -> dict[tuple[str, str], list[dict]]:
    rows: dict[tuple[str, str], list[dict]] = {}
    for r in TOOL_ROWS:
        dataset = CENSUS.get(r["dataset"], r["dataset"]) if census_as_embryo else r["dataset"]
        rows.setdefault((dataset, r["tool"]), []).append(r)
    return rows


def smallest_tier(runs: list[dict]) -> tuple[float | None, list[dict]]:
    done = [r for r in runs if r.get("analysisTimeMinutes") is not None]
    if not done:
        return (None, [])
    tier = min((r["completedAtGib"] for r in done))
    return (tier, [r for r in done if r["completedAtGib"] == tier])


def ladder_start(runs: list[dict]) -> float:
    return min((t["memoryGib"] for r in runs for t in r["tiers"]))


def crashed(runs: list[dict]) -> bool:
    return any(((r.get("outcome") or "").startswith("crashed") for r in runs))


def resolve_arm(rows: dict, dataset: str, tool: str) -> tuple[list[dict] | None, str, str]:
    runs = rows.get((dataset, tool))
    spca = rows.get((dataset, tool + "_spca"))
    if spca is not None and (runs is None or (not smallest_tier(runs)[1] and crashed(runs))):
        return (spca, tool + "_spca", "\\dag")
    return (runs, tool, "")


def final_failure(runs: list[dict]) -> dict:
    outcome = runs[0].get("outcome") or ""
    if outcome.startswith("dgCMatrix limit"):
        return {"code": "matrix limit", "tier": None, "step": None, "outcome": outcome}
    tiers = [
        t
        for r in runs
        for t in r["tiers"]
        if t["status"] in MEMORY_FAILURES or t["status"] in ("timedOut", "error")
    ]
    last = max(tiers, key=lambda t: t["memoryGib"])
    if last["status"] in MEMORY_FAILURES:
        code = "OOM"
    elif last["status"] == "timedOut":
        code = "TL"
    elif outcome.startswith("crashed"):
        code = "crash"
    else:
        code = "error"
    step = last.get("failedStep")
    if code == "crash" and step is None:
        step = re.search("during (\\w+)", outcome).group(1)
    return {
        "code": code,
        "tier": last["memoryGib"],
        "step": step,
        "limit": last.get("outcomeSource"),
        "outcome": outcome,
        "peakCgroupGib": last.get("peakCgroupGib"),
    }


def prepare() -> None:
    require_verified_workflows()
    grid = tool_rows(census_as_embryo=True)
    toolkit_rows = []
    for tool, _, _ in TOOLKIT_ARMS:
        for dataset, *_ in DATASETS:
            runs, actual_tool, mark = resolve_arm(grid, dataset, tool)
            row = {
                "tool": tool,
                "dataset": dataset,
                "actual_tool": actual_tool,
                "cells": next(
                    (r["cells"] for r in TOOL_ROWS if r["dataset"] == dataset and r.get("cells"))
                ),
                "scaled_pca": bool(mark),
                "status": "not run",
            }
            if runs is not None:
                tier, done = smallest_tier(runs)
                if done:
                    row.update(
                        status="completed",
                        mean_minutes=statistics.mean((r["analysisTimeMinutes"] for r in done)),
                        runs=len(done),
                        memory_gib=tier,
                        ladder_start_gib=ladder_start(runs),
                    )
                else:
                    f = final_failure(runs)
                    row.update(status=f["code"], memory_gib=f["tier"], failed_step=f["step"])
            toolkit_rows.append(row)
    for dataset, *_ in DATASETS:
        completed = [
            r for r in toolkit_rows if r["dataset"] == dataset and r["status"] == "completed"
        ]
        fastest = min(completed, key=lambda r: (r["mean_minutes"], r["tool"]))["tool"]
        for row in toolkit_rows:
            if row["dataset"] == dataset:
                row["fastest"] = row["tool"] == fastest
    write_table("extended_table_1/data", toolkit_rows)
