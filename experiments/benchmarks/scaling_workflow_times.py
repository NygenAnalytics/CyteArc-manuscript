"""Benchmark times from each run's own funnel record: elapsed time, and the times of the stages. Beside them, the
replicates of the full Tahoe-100M run summarised: their mean analysis hours, core-minutes, UMAP peak and Leiden
minutes, the clusters one cell line dominates, the largest exponent of a stage's seconds from the pilot to the full
atlas, and the atlas's uncompressed CSR size; and the exponent of analysis core-time against cells over the embryo
runs from 100,000 cells and the Tahoe-100M replicates, and over the runs from 1,000,000 cells, with each size's
core-minutes."""

import math
import re
from collections import defaultdict
from statistics import mean

import numpy as np

from experiments.benchmarks.analysis_times import uncompressed_csr_bytes
from experiments.runtime.common import DERIVED, ROOT, load_json, write_csv_at, write_json_at
from experiments.runtime.records import (
    RAW,
    STAGES,
    conversion_elapsed_seconds,
    elapsed_seconds,
    load_runs,
    profiler_seconds,
)

DATA = ROOT / "experiments/benchmarks/records"
BENCHMARKS_DERIVED = DERIVED / "benchmarks"
SCALING_RUNS = RAW / "rna_scaling_runs.json"
TAHOE_RESULTS = DATA / "tahoe_rna_scaling_results.json"
TAHOE_STAGE_MEASUREMENTS = DATA / "raw" / "tahoe_rna_stage_measurements.json"
# The core-time exponent is fitted to the embryo runs from this many cells up, the sizes the paper reports.
CORE_TIME_FROM_CELLS = 100_000
# A second exponent is fitted to the runs from this many cells up.
CORE_TIME_FROM_MILLION_CELLS = 1_000_000
# The embryo sizes whose mean core-minutes alone give a two-point exponent, as the toolkit runs give one.
CORE_TIME_TWO_POINT_CELLS = (1_000_000, 10_000_000)
# A Tahoe-100M Leiden cluster counts as one cell line's when that line holds at least this share of its cells.
PURITY_THRESHOLD = 0.9
# The three replicates of the full Tahoe-100M run, by their names in the stage measurements.
TAHOE_FULL_RUNS = ("full", "fullReplicate2", "fullReplicate3")
# The analysis stages of a Tahoe-100M run, from reopening the prepared store to the marker search.
TAHOE_ANALYSIS_STAGES = (
    "reopenStore",
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


SIZE_LABELS = {
    100_000: "100k",
    500_000: "500k",
    1_000_000: "1M",
    5_000_000: "5M",
    10_000_000: "10M",
}


def replicates() -> list[dict]:
    rows = []
    for _, run in sorted(load_runs("rna_scaling_runs").items()):
        funnel = run["funnel"]
        if funnel is None:
            # A run still in progress: its stage records are collected, its funnel record not yet.
            continue
        if funnel.get("status") != "ok":
            print(
                f"left out {funnel['runTag']}: funnel {funnel.get('status')} in {funnel.get('failedStage')}"
            )
            continue
        outcomes = {o["stage"]: o for o in funnel["outcomes"]}
        n = int(funnel["nRows"])
        cpus = float(funnel["modalResources"]["modalCpuLimit"])
        memory = float(funnel["modalResources"]["modalMemoryLimitMb"]) * 1024**2
        budget = float(outcomes["createStore"]["cytearcMemoryBudget"])
        seconds = {s: float(outcomes[s]["seconds"]) for s in STAGES}
        rows.append(
            {
                "tag": funnel["runTag"],
                "nRows": n,
                "cpus": cpus,
                "containerBytes": memory,
                "budgetBytes": budget,
                "elapsedSeconds": elapsed_seconds(funnel),
                "conversionSeconds": conversion_elapsed_seconds(funnel),
                "analysisSeconds": elapsed_seconds(funnel) - conversion_elapsed_seconds(funnel),
                "stageSumSeconds": sum(seconds.values()),
                "inputSetupSeconds": float(outcomes["createStore"]["inputSetupSeconds"]),
                "profilerSeconds": profiler_seconds(funnel),
                "downloadSeconds": float(funnel["datasetDownloadSeconds"]),
                "peakRssBytes": float(funnel["peakRssBytes"]),
                "peakFractionOfBudget": float(funnel["peakRssBytes"]) / budget,
                **{f"stage.{s}": v for s, v in seconds.items()},
            }
        )
    return rows


def selected_replicates(rows: list[dict]) -> list[dict]:
    selected = {}
    for row in rows:
        replicate = int(re.search(r"_replicate_(\d+)", row["tag"]).group(1))
        key = row["nRows"], replicate
        if key in selected:
            raise ValueError(f"Duplicate RNA scaling repetition: {key}")
        selected[key] = row
    return sorted(selected.values(), key=lambda row: (row["nRows"], row["tag"]))


def tahoe_core_minutes(runs: dict) -> dict:
    """CyteArc's analysis core-time on Tahoe-100M in each replicate of the full run, from the stage measurements' runs.
    Each stage ran in a Modal call of its own, so a replicate's core-time adds each analysis stage's seconds times the
    CPUs of its call."""
    cells, cpus, values = set(), set(), []
    for name in TAHOE_FULL_RUNS:
        stages = runs[name]["stages"]
        missing = [stage for stage in TAHOE_ANALYSIS_STAGES if stage not in stages]
        if missing:
            raise ValueError(f"Tahoe-100M {name} has no record of {missing}")
        records = [stages[stage] for stage in TAHOE_ANALYSIS_STAGES]
        if any(record["status"] != "ok" for record in records):
            raise ValueError(f"Tahoe-100M {name} has an analysis stage that did not complete")
        values.append(sum(record["seconds"] * record["modalCpuLimit"] for record in records) / 60)
        cpus.update(record["modalCpuLimit"] for record in records)
        cells.add(runs[name]["nRows"])
    if len(cells) != 1 or len(cpus) != 1:
        raise ValueError(
            "the Tahoe-100M replicates need one cell count and one CPU count across their analysis stages"
        )
    return {"cells": cells.pop(), "cpus": cpus.pop(), "values": values}


def core_time_fit(rows: list[dict], tahoe_runs: dict) -> dict:
    """CyteArc's analysis core-time against cells, and the power law fitted to it. On the embryo samples of
    CORE_TIME_FROM_CELLS cells or more a run's core-minutes are its analysis seconds times its container's CPUs, three
    runs per size; on Tahoe-100M they are tahoe_core_minutes of its three replicates. The exponent b in
    core-time ∝ cells^b is the slope of the least-squares line of log10 core-minutes against log10 cells, fitted to
    every run with one intercept. ``rows`` holds each embryo run's nRows, cpus and analysisSeconds, as replicates
    gives them or as the derived CSV stores them, and ``tahoe_runs`` the stage measurements' runs. Returns each size
    in order of cells, with its dataset, CPUs, core-minutes and their mean, the fit with how many runs it takes, the same fit to the runs of
    CORE_TIME_FROM_MILLION_CELLS cells or more with how many runs it takes, and the exponent between the mean
    core-minutes of the two embryo sizes of CORE_TIME_TWO_POINT_CELLS."""
    core_minutes: dict[int, list[float]] = defaultdict(list)
    cpus: dict[int, set[float]] = defaultdict(set)
    for row in rows:
        cells = int(row["nRows"])
        if cells < CORE_TIME_FROM_CELLS:
            continue
        core_minutes[cells].append(float(row["cpus"]) * float(row["analysisSeconds"]) / 60)
        cpus[cells].add(float(row["cpus"]))
    sizes = []
    for cells in sorted(core_minutes):
        if len(core_minutes[cells]) != 3 or len(cpus[cells]) != 1:
            raise ValueError(f"{cells:,} cells need three runs on one container size")
        sizes.append(
            {
                "cells": cells,
                "dataset": "embryo",
                "cpus": cpus[cells].pop(),
                "coreMinutes": core_minutes[cells],
            }
        )
    tahoe = tahoe_core_minutes(tahoe_runs)
    if len(tahoe["values"]) != 3 or tahoe["cells"] in core_minutes:
        raise ValueError("Tahoe-100M needs three runs at a size of its own")
    sizes.append(
        {
            "cells": tahoe["cells"],
            "dataset": "tahoe",
            "cpus": tahoe["cpus"],
            "coreMinutes": tahoe["values"],
        }
    )
    if len(sizes) < 3:
        raise ValueError(
            f"the exponent needs at least three sizes of {CORE_TIME_FROM_CELLS:,} cells or more"
        )
    exponent, intercept = power_law_fit(sizes)
    large = [size for size in sizes if size["cells"] >= CORE_TIME_FROM_MILLION_CELLS]
    if len(large) < 2:
        raise ValueError(
            f"the exponent from {CORE_TIME_FROM_MILLION_CELLS:,} cells needs at least two sizes there"
        )
    large_exponent, large_intercept = power_law_fit(large)
    for size in sizes:
        size["meanCoreMinutes"] = mean(size["coreMinutes"])
    embryo_means = {s["cells"]: s["meanCoreMinutes"] for s in sizes if s["dataset"] == "embryo"}
    small, large_cells = CORE_TIME_TWO_POINT_CELLS
    if small not in embryo_means or large_cells not in embryo_means:
        raise ValueError(
            f"the two-point exponent needs embryo runs at {small:,} and {large_cells:,} cells"
        )
    two_point = [embryo_means[small], embryo_means[large_cells]]
    return {
        "fromCells": CORE_TIME_FROM_CELLS,
        "runs": sum(len(size["coreMinutes"]) for size in sizes),
        "exponent": exponent,
        "intercept": intercept,
        "fromMillionCells": {
            "fromCells": CORE_TIME_FROM_MILLION_CELLS,
            "runs": sum(len(size["coreMinutes"]) for size in large),
            "exponent": large_exponent,
            "intercept": large_intercept,
        },
        "twoPoint": {
            "cells": list(CORE_TIME_TWO_POINT_CELLS),
            "meanCoreMinutes": two_point,
            "exponent": math.log(two_point[1] / two_point[0]) / math.log(large_cells / small),
        },
        "sizes": sizes,
    }


def power_law_fit(sizes: list[dict]) -> tuple[float, float]:
    """The slope and intercept of the least-squares line of log10 core-minutes against log10 cells over every run of
    ``sizes``."""
    x = np.log10([size["cells"] for size in sizes for _ in size["coreMinutes"]])
    y = np.log10([value for size in sizes for value in size["coreMinutes"]])
    exponent, intercept = np.polyfit(x, y, 1)
    return float(exponent), float(intercept)


def tahoe_replicates(results: dict) -> dict:
    """Means over the replicates of the full Tahoe-100M run of their store preparation hours, analysis hours, UMAP
    container peak, Leiden minutes and marker-testing minutes; the first replicate's Leiden clusters whose commonest cell line holds at least PURITY_THRESHOLD of their
    cells; and the exponent b in seconds ∝ cells^b of each stage from the pilot to the full atlas, with the largest."""
    rows = results["fullReplicates"]
    if not rows:
        raise ValueError("the Tahoe-100M results hold no replicate of the full run")
    clusters = results["fullStore"]["clusters"]
    scale = math.log(results["runs"]["full"]["nRows"] / results["runs"]["test"]["nRows"])
    exponents = {
        stage: math.log(s["fullOverTestSeconds"]) / scale for stage, s in results["stages"].items()
    }
    return {
        "replicates": len(rows),
        "meanPreparationHours": mean(row["preparationHours"] for row in rows),
        "meanAnalysisHours": mean(row["analysisHours"] for row in rows),
        "meanUmapPeakGiB": mean(row["stages"]["runUmap"]["peakCgroupGiB"] for row in rows),
        "meanLeidenMinutes": mean(row["stages"]["runLeiden"]["seconds"] / 60 for row in rows),
        "meanMarkerMinutes": mean(row["stages"]["findMarkers"]["seconds"] / 60 for row in rows),
        "clusters": len(clusters),
        "purityThreshold": PURITY_THRESHOLD,
        "clustersAtPurityThreshold": sum(
            cluster["commonestCellLineFraction"] >= PURITY_THRESHOLD for cluster in clusters
        ),
        "stageExponents": exponents,
        "largestStageExponent": max(exponents.values()),
    }


def tahoe_summary(results: dict, runs: dict) -> dict:
    """The replicates of the full Tahoe-100M run (tahoe_replicates), their analysis core-minutes and the mean, and the
    atlas's uncompressed CSR size. ``results`` is the collected Tahoe-100M results and ``runs`` the stage measurements'
    runs."""
    core = tahoe_core_minutes(runs)
    if core["cells"] != results["dataset"]["cells"]:
        raise ValueError(
            f"the stage measurements ran on {core['cells']} cells, the results hold {results['dataset']['cells']}"
        )
    return {
        **tahoe_replicates(results),
        "analysisCoreMinutes": core["values"],
        "meanAnalysisCoreMinutes": mean(core["values"]),
        "uncompressedCsrBytes": uncompressed_csr_bytes(results["dataset"]["storedValues"]),
    }


def main() -> None:
    rows = selected_replicates(replicates())
    write_csv_at(BENCHMARKS_DERIVED / "rna_scaling_workflow_times.csv", rows)
    stage_runs = load_json(TAHOE_STAGE_MEASUREMENTS)["runs"]
    write_json_at(
        BENCHMARKS_DERIVED / "tahoe_rna_scaling_summary.json",
        {
            "source": [
                str(TAHOE_RESULTS.relative_to(ROOT)),
                str(TAHOE_STAGE_MEASUREMENTS.relative_to(ROOT)),
                str(SCALING_RUNS.relative_to(ROOT)),
            ],
            **tahoe_summary(load_json(TAHOE_RESULTS), stage_runs),
            "coreTimeScaling": core_time_fit(rows, stage_runs),
        },
    )


if __name__ == "__main__":
    main()
