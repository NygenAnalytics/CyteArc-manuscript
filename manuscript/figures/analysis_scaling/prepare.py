"""Prepare the panel data of the figure on analysis at scale from the benchmark records and their derived summaries.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py:
store_layout.json, peak_memory.json, analysis_time.json, stage_times.json and budget_sweep.json.
Run from the repository root:

  uv run --locked python -B -m manuscript.figures.analysis_scaling.prepare
"""

import csv
import json
import math
from collections import defaultdict
from typing import NamedTuple

import numpy as np
from experiments.benchmarks.analysis_times import (
    GRAPH_STEPS,
    STAGE_DATASET,
    STAGE_TOOLS,
    STAGES,
    STREAMED_STEPS,
    UNSHOWN,
)
from experiments.benchmarks.resource_trace_summary import TRACE_LEFT_OUT, TRACE_NAMES
from experiments.benchmarks.scaling_workflow_times import TAHOE_ANALYSIS_STAGES, core_time_fit
from experiments.benchmarks.tool_summary import require_verified_workflows
from experiments.runtime.common import DERIVED, ROOT, load_json

RECORDS = ROOT / "experiments/benchmarks/records"
SUMMARY = DERIVED / "benchmarks" / "toolkit_comparisons.json"
ANALYSIS_TIMES = DERIVED / "benchmarks" / "toolkit_analysis_times.json"
STEP_MEASUREMENTS = DERIVED / "benchmarks" / "toolkit_step_measurements.csv"
LAYOUT_GEOMETRY = RECORDS / "count_layout_geometry.json"
CATALOGUE_OBJECTS = ROOT / "experiments/catalogue/records/catalogue_objects.json"
TAHOE_RESULTS = RECORDS / "tahoe_rna_scaling_results.json"
TAHOE_STAGE_MEASUREMENTS = RECORDS / "raw" / "tahoe_rna_stage_measurements.json"
TAHOE_RASTER = RECORDS / "tahoe_umap_raster.npz"
TAHOE_CLUSTER_LINES = RECORDS / "tahoe_cluster_cell_lines.csv"
# The comparator runs on Tahoe-100M, and the configuration of the figure each one stands for.
TAHOE_COMPARATORS = RECORDS / "toolkit_comparison_tahoe_runs.json"
TAHOE_COMPARATOR_ARMS = {"bpcells_cached_spca": "bpcells_cached"}
# The export's cell lines in the raster's code order, each named with its organ from Tahoe-100M's cell-line table.
TAHOE_CELL_LINES = RECORDS / "tahoe_cell_lines.csv"
# The UMAP names this many cell lines, the largest.
LABELLED_CELL_LINES = 6
# A cluster's pie draws each cell line holding at least this share of its cells; the rest share one slice.
PIE_SHARE = 0.05
# The per-second resource trace of the ten-million-cell scaling run, with its stages' resources.
RESOURCE_TRACE = RECORDS / "resource_trace_10000000_cells"
STORAGE_NPZ = ROOT / "manuscript/figures/architecture/storage.npz"
OUT = ROOT / "manuscript" / "figures" / "analysis_scaling"

# The toolkit configurations drawn, by their key in the derived summary.
TOOLS = (
    "cytearc",
    "scanpy",
    "scanpy_dask_covariance_csc",
    "census_scanpy",
    "seurat",
    "seurat_bpcells",
    "bpcells_cached",
)
# The benchmark datasets the figure draws. Tabula Sapiens (1.1M cells) is benchmarked too; the author keeps it in
# the text and tables and leaves it out of this figure.
DATASETS = (
    ("embryo_1000000", "Embryo 1M", "1M"),
    ("brain_1306127", "Mouse brain 1.3M", "1.3M"),
    ("hbca_neurons", "Brain-atlas neurons 2.5M", "2.5M"),
    ("mssm_cohort", "Prefrontal cortex 4.1M", "4.1M"),
    ("embryo_10000000", "Embryo 10M", "10M"),
)
DATASET_KEYS = [key for key, _, _ in DATASETS]
# Census runs read the embryo samples through CELLxGENE Census under their own dataset keys.
CENSUS = {"census_1000000": "embryo_1000000", "census_10000000": "embryo_10000000"}
MEMORY_FAILURES = ("killed", "outOfMemory", "killed (record incomplete)")
# A configuration whose default UMAP crashed is drawn from its run with UMAP from scaled PCA.
SPCA = "_spca"
STAND_IN = "UMAP from scaled PCA"


def point_summary(values: list[float]) -> dict:
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Figure summaries require completed finite observations")
    return {
        "mean": float(np.mean(values)),
        "low": min(values),
        "high": max(values),
        "n": len(values),
        "values": values,
    }


# ---- analysis time ------------------------------------------------------------------------------


def summary_rows() -> dict[tuple[str, str], list[dict]]:
    rows: dict[tuple[str, str], list[dict]] = {}
    document = load_json(SUMMARY)
    require_verified_workflows(document)
    for row in document["workflow"]:
        rows.setdefault((CENSUS.get(row["dataset"], row["dataset"]), row["tool"]), []).append(row)
    return rows


def smallest(runs: list[dict]) -> list[dict]:
    """The completed runs on the smallest container on which the tool completed."""
    done = [r for r in runs if r.get("analysisTimeMinutes") is not None]
    if not done:
        return []
    least = min(r["completedAtGib"] for r in done)
    return [r for r in done if r["completedAtGib"] == least]


def crashed(runs: list[dict]) -> bool:
    return any((r.get("outcome") or "").startswith("crashed") for r in runs)


def failure(runs: list[dict]) -> dict:
    """How a configuration that did not complete failed: its state, last container and step."""
    failed = [r for r in runs if r.get("outcome") not in (None, "interrupted")]
    run = failed[0]
    last = run["tiers"][-1]
    outcome = run["outcome"]
    if outcome.startswith("dgCMatrix limit"):
        state = "na"
    elif outcome.startswith("crashed"):
        state = "crash"
    elif last["status"] == "timedOut":
        state = "timeout"
    elif last["status"] in MEMORY_FAILURES:
        state = "oom"
    else:
        state = "other"
    return {
        "state": state,
        "outcome": outcome,
        "lastTierGiB": last["memoryGib"],
        "lastStep": last["failedStep"],
    }


class Arm(NamedTuple):
    """One configuration on one dataset: the runs drawn, or how it failed."""

    tool: str  # the summary key drawn: the configuration, or its scaled-PCA stand-in
    stand_in: bool
    runs: list[dict]  # completed runs, empty when it did not complete
    failure: dict | None


def arm(rows: dict, dataset: str, key: str) -> Arm | None:
    runs = rows.get((dataset, key))
    spca = rows.get((dataset, key + SPCA))
    if not runs and not spca:
        return None
    if spca and (not runs or (not smallest(runs) and crashed(runs))):
        done = smallest(spca)
        if done:
            return Arm(key + SPCA, True, done, None)
        return Arm(key, False, [], failure(runs or spca))
    done = smallest(runs)
    return Arm(key, False, done, None) if done else Arm(key, False, [], failure(runs))


def tahoe_replicates() -> dict:
    """CyteArc's analysis minutes and analysis peak container memory (GiB) on Tahoe-100M over the replicates of the
    full run. Each stage ran in its own container, so a replicate's peak is the highest of its analysis stages."""
    rows = load_json(TAHOE_RESULTS)["fullReplicates"]
    return {
        "minutes": point_summary([row["analysisHours"] * 60.0 for row in rows]),
        "peakGiB": point_summary(
            [
                max(row["stages"][stage]["peakCgroupGiB"] for stage in TAHOE_ANALYSIS_STAGES)
                for row in rows
            ]
        ),
    }


def tahoe_outcomes() -> dict:
    """How each comparator run on Tahoe-100M failed, under the configuration it stands for: out of memory when it
    was killed for memory, out of time when it reached its time limit."""
    outcomes: dict[str, dict] = {}
    for record in load_json(TAHOE_COMPARATORS)["records"].values():
        arm = TAHOE_COMPARATOR_ARMS[record["tool"]]
        if arm in outcomes:
            raise ValueError(f"Tahoe-100M has more than one run of {arm}")
        if record["status"] in MEMORY_FAILURES:
            state = "oom"
        elif record["status"] == "timedOut":
            state = "timeout"
        else:
            raise ValueError(
                f"{record['tool']} on Tahoe-100M ended with status {record['status']!r}; the status grid "
                f"draws a run killed for memory ({', '.join(MEMORY_FAILURES)}) or timed out (timedOut)"
            )
        outcomes[arm] = {
            "tool": record["tool"],
            "state": state,
            "failedStep": record["failedStep"],
            "containerGiB": record["container"]["memoryMb"] / 1024,
            "cpu": record["container"]["cpu"],
        }
    return outcomes


def analysis_time() -> dict:
    """Analysis time of each configuration on the drawn datasets: the runs on the smallest container that
    completed, checked against the means of the derived analysis-time summary."""
    rows = summary_rows()
    reference = {(a["dataset"], a["tool"]): a for a in load_json(ANALYSIS_TIMES)["arms"]}
    datasets = {}
    for key, name, tick in DATASETS:
        first = smallest(rows[(key, "cytearc")])[0]
        datasets[key] = {"name": name, "tick": tick, "cells": first["cells"]}
    arms: dict[str, dict] = {}
    for tool in TOOLS:
        arms[tool] = {}
        for dataset in DATASET_KEYS:
            found = arm(rows, dataset, tool)
            if found is None:
                arms[tool][dataset] = {"state": "not run"}
            elif found.runs:
                minutes = [r["analysisTimeMinutes"] for r in found.runs]
                summary = point_summary(minutes)
                checked = reference[(dataset, found.tool)]
                if (
                    abs(checked["meanAnalysisMinutes"] - summary["mean"]) > 1e-9
                    or checked["n"] != summary["n"]
                ):
                    raise ValueError(
                        f"{tool} on {dataset}: the drawn runs differ from {ANALYSIS_TIMES.name}"
                    )
                arms[tool][dataset] = {
                    "tool": found.tool,
                    "standIn": STAND_IN if found.stand_in else None,
                    "containerGiB": found.runs[0]["completedAtGib"],
                    "cpu": found.runs[0]["cpu"],
                    "minutes": summary,
                }
            else:
                arms[tool][dataset] = {"tool": found.tool, **found.failure}
    means = {
        tool: float(np.mean([v["minutes"]["mean"] for v in arms[tool].values() if "minutes" in v]))
        for tool in TOOLS
    }
    tahoe = load_json(TAHOE_RESULTS)
    return {
        "datasets": datasets,
        "arms": arms,
        "tahoe": {
            "cells": tahoe["dataset"]["cells"],
            "tick": "95.6M",
            "minutes": tahoe_replicates()["minutes"],
            "cpu": 16,
            "containerGiB": 256.0,
        },
        "legendOrder": sorted(TOOLS, key=lambda tool: -means[tool]),
        "source": [
            str(SUMMARY.relative_to(ROOT)),
            str(ANALYSIS_TIMES.relative_to(ROOT)),
            str(TAHOE_RESULTS.relative_to(ROOT)),
        ],
    }


def drawn_runs(rows: dict, times: dict, tool: str, dataset: str) -> list[dict]:
    """The runs analysis_time draws for one configuration on one dataset, found again in the summary and checked
    against the minutes it drew."""
    drawn = times["arms"][tool][dataset]
    runs = smallest(rows[(dataset, drawn["tool"])])
    if [r["analysisTimeMinutes"] for r in runs] != drawn["minutes"]["values"]:
        raise ValueError(
            f"{tool} on {dataset}: the runs found differ from those analysis_time draws"
        )
    return runs


# ---- peak memory --------------------------------------------------------------------------------


def peak_memory(times: dict) -> dict:
    """Peak container memory of each configuration on the drawn datasets, from the runs analysis_time draws: the
    completed runs on the smallest container on which the configuration completed."""
    rows = summary_rows()
    arms: dict[str, dict] = {}
    for tool in TOOLS:
        arms[tool] = {}
        for dataset in DATASET_KEYS:
            drawn = times["arms"][tool][dataset]
            if "minutes" not in drawn:
                arms[tool][dataset] = dict(drawn)
                continue
            runs = drawn_runs(rows, times, tool, dataset)
            arms[tool][dataset] = {
                "tool": drawn["tool"],
                "containerGiB": drawn["containerGiB"],
                "peakGiB": point_summary([r["peakCgroupGib"] for r in runs]),
            }
    containers = {row["containerGiB"] for row in arms["cytearc"].values()}
    if len(containers) != 1:
        raise ValueError(
            f"CyteArc completed on containers of {sorted(containers)} GiB; one container expected"
        )
    tahoe = load_json(TAHOE_RESULTS)
    return {
        "datasets": times["datasets"],
        "arms": arms,
        "tahoe": {
            "cells": tahoe["dataset"]["cells"],
            "tick": "95.6M",
            "peakGiB": tahoe_replicates()["peakGiB"],
            "containerGiB": 256.0,
            "outcomes": tahoe_outcomes(),
        },
        "legendOrder": times["legendOrder"],
        "cytearcContainerGiB": containers.pop(),
        "source": [
            str(SUMMARY.relative_to(ROOT)),
            str(TAHOE_RESULTS.relative_to(ROOT)),
            str(TAHOE_COMPARATORS.relative_to(ROOT)),
        ],
    }


# ---- stage times --------------------------------------------------------------------------------

# STAGES (the analysis stages drawn, in workflow order, with the recorded steps of each tool they sum) and UNSHOWN (the
# steps no stage shows) come from the benchmark experiment, whose derived stage times use them too.

# The resource trace leaves out the import (TRACE_LEFT_OUT) and draws the UMAP initialisation with the neighbour search
# (TRACE_NAMES), as the benchmark experiment's summary of the trace does.


def stage_times(times: dict) -> dict:
    """Minutes of each analysis stage of CyteArc, BPCells, Scanpy (Dask) and Seurat (BPCells) on the
    ten-million-cell embryo sample, the stages in the order the workflow runs them."""
    rows = summary_rows()
    seconds: dict[str, dict[str, float]] = defaultdict(dict)
    with STEP_MEASUREMENTS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == STAGE_DATASET and row["seconds"]:
                seconds[row["record"]][row["step"]] = float(row["seconds"])
    tools, minutes = {}, {}
    for tool in STAGE_TOOLS:
        drawn = times["arms"][tool][STAGE_DATASET]
        if drawn.get("standIn"):
            raise ValueError(f"{tool} on {STAGE_DATASET} is drawn from a stand-in configuration")
        runs = drawn_runs(rows, times, tool, STAGE_DATASET)
        per_run = []
        for run in runs:
            steps = seconds[run["sourceRecord"]]
            stage_minutes = [sum(steps[s] for s in by_tool[tool]) / 60 for _, by_tool in STAGES]
            unshown = sum(steps[s] for s in UNSHOWN[tool]) / 60
            if abs(sum(stage_minutes) + unshown - run["analysisTimeMinutes"]) > 0.05:
                raise ValueError(
                    f"{run['sourceRecord']}: the stages add up to {sum(stage_minutes) + unshown:.2f} minutes, "
                    f"not the analysis time of {run['analysisTimeMinutes']:.2f}"
                )
            per_run.append(stage_minutes)
        minutes[tool] = per_run
        tools[tool] = {
            "containerGiB": drawn["containerGiB"],
            "cpu": drawn["cpu"],
            "n": len(runs),
            "unshownMinutes": point_summary(
                [sum(seconds[r["sourceRecord"]][s] for s in UNSHOWN[tool]) / 60 for r in runs]
            )
            if UNSHOWN[tool]
            else None,
        }
    stages = []
    for i, (name, by_tool) in enumerate(STAGES):
        summaries = {tool: point_summary([run[i] for run in minutes[tool]]) for tool in STAGE_TOOLS}
        stages.append(
            {
                "stage": name,
                "steps": {tool: list(by_tool[tool]) for tool in STAGE_TOOLS},
                **summaries,
                "ratio": summaries["bpcells_cached"]["mean"] / summaries["cytearc"]["mean"],
            }
        )
    # Also include Scanpy (Dask) and Seurat (BPCells) 10M step durations (and where they failed).
    tool_step_secs: dict[tuple[str, str], list[float]] = defaultdict(list)
    with STEP_MEASUREMENTS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == STAGE_DATASET and row["seconds"]:
                tool_step_secs[(row["tool"], row["step"])].append(float(row["seconds"]))
    other_10m = {
        "scanpy_dask_covariance_csc": {
            "filter–PCA": sum(
                float(np.mean(tool_step_secs[("scanpy_dask_covariance_csc", s)]))
                for s in ("filterCells", "markHvgs", "runNormalization", "runPca")
            )
            / 60.0,
            "UMAP": float(np.mean(tool_step_secs[("scanpy_dask_covariance_csc", "runUmap")]))
            / 60.0,
            "Leiden": float(np.mean(tool_step_secs[("scanpy_dask_covariance_csc", "runLeiden")]))
            / 60.0,
            "neighbors": float(
                np.mean(tool_step_secs[("scanpy_dask_covariance_csc", "queryNeighbors")])
            )
            / 60.0,
            "markers": None,
            "failedAt": "markers (out of time: Modal's 24-hour limit at 336 GiB)",
        },
        "seurat_bpcells": {
            "filter–PCA": sum(
                float(np.mean(tool_step_secs[("seurat_bpcells", s)]))
                for s in ("filterCells", "markHvgs", "runNormalization", "runPca")
            )
            / 60.0,
            "UMAP": float(np.mean(tool_step_secs[("seurat_bpcells", "runUmap")])) / 60.0,
            "Leiden": None,
            "neighbors": float(np.mean(tool_step_secs[("seurat_bpcells", "queryNeighbors")]))
            / 60.0,
            "markers": None,
            "failedAt": "Leiden (24-hour call limit)",
        },
    }
    return {
        "dataset": STAGE_DATASET,
        "cells": times["datasets"][STAGE_DATASET]["cells"],
        "tools": tools,
        "stages": stages,
        "otherTools10M": other_10m,
        "source": [str(SUMMARY.relative_to(ROOT)), str(STEP_MEASUREMENTS.relative_to(ROOT))],
    }


# ---- memory budget sweep ------------------------------------------------------------------------


def budget_sweep() -> dict:
    """Peak container memory from input conversion through PCA on one million embryo cells, three runs
    under each memory budget."""
    source = RECORDS / "memory_preprocessing.json"
    document = load_json(source)
    expected_steps = {
        "convertMatrix",
        "createObject",
        "filterCells",
        "markHvgs",
        "runNormalization",
        "runPca",
    }
    groups = defaultdict(list)
    for name, run in document["records"].items():
        if (
            run["status"] != "ok"
            or run["mode"] != "preprocessing"
            or run["completedThrough"] != "runPca"
            or set(run["steps"]) != expected_steps
        ):
            raise ValueError(f"{name}: the budget sweep needs completed runs from input to PCA")
        if (
            run["dataset"] != "embryo_1000000"
            or run["workers"] != 8
            or run["container"]["preemptible"]
        ):
            raise ValueError(f"{name}: the run differs from the fixed input or environment")
        budget = run["budgetBytes"] / 1024**3
        if budget not in {4, 8, 16, 24}:
            raise ValueError(f"{name}: unexpected budget of {budget} GiB")
        unit, chunk = map(int, run["countPolicy"]["requested"].split(","))
        steps = run["steps"]
        groups[int(budget)].append(
            {
                "replicate": run["replicate"],
                "unitBytes": unit,
                "chunkBytes": chunk,
                "minutes": (steps["runPca"]["end"] - min(step["start"] for step in steps.values()))
                / 60,
                "containerGiB": max(step["peakCgroupBytes"] for step in steps.values()) / 1024**3,
            }
        )
    rows = []
    for budget, runs in sorted(groups.items()):
        if sorted(run["replicate"] for run in runs) != [1, 2, 3]:
            raise ValueError(f"the {budget} GiB budget needs three runs")
        policies = {(run["unitBytes"], run["chunkBytes"]) for run in runs}
        if len(policies) != 1:
            raise ValueError(
                f"the runs under {budget} GiB use different read-unit and chunk targets"
            )
        unit, chunk = policies.pop()
        rows.append(
            {
                "budgetGiB": budget,
                "unitBytes": unit,
                "chunkBytes": chunk,
                "minutes": point_summary([run["minutes"] for run in runs]),
                "containerGiB": point_summary([run["containerGiB"] for run in runs]),
            }
        )
    if [row["budgetGiB"] for row in rows] != [4, 8, 16, 24]:
        raise ValueError("the budget sweep needs runs under 4, 8, 16 and 24 GiB")
    return {"dataset": "embryo_1000000", "rows": rows, "source": str(source.relative_to(ROOT))}


# ---- analysis time against cells ----------------------------------------------------------------

SCALING_TIMES = DERIVED / "benchmarks" / "rna_scaling_workflow_times.csv"


def core_time_scaling() -> dict:
    """CyteArc's analysis core-time (analysis seconds times the container's CPUs) on embryo samples of 100,000 to
    10,000,000 cells and on the 95.6-million-cell Tahoe-100M atlas, three runs per size, and the exponent of
    core-time against cells fitted to every run shown, as the benchmarks experiment's core_time_fit gives them."""
    with SCALING_TIMES.open(newline="") as handle:
        fit = core_time_fit(
            list(csv.DictReader(handle)), load_json(TAHOE_STAGE_MEASUREMENTS)["runs"]
        )
    return {
        "sizes": [
            {
                "cells": size["cells"],
                "dataset": size["dataset"],
                "cpus": size["cpus"],
                "coreMinutes": point_summary(size["coreMinutes"]),
            }
            for size in fit["sizes"]
        ],
        "slope": {
            "fromCells": fit["fromCells"],
            "value": fit["exponent"],
            "intercept": fit["intercept"],
        },
        "source": [
            str(SCALING_TIMES.relative_to(ROOT)),
            str(TAHOE_STAGE_MEASUREMENTS.relative_to(ROOT)),
        ],
    }


# ---- store layout -------------------------------------------------------------------------------


def store_layout() -> dict:
    """The numbers the schematic of the store prints: the layout policy's read-unit and chunk targets, the decode
    amplification of a gene-wise pass at the largest planned embryo size with and without the gene-major copy, and
    the median size of the gene-major copy relative to the cell-major array across the CyteBase stores."""
    geometry = load_json(LAYOUT_GEOMETRY)
    largest = max(geometry["rows"], key=lambda row: row["nCells"])
    if (largest["unitBytes"], largest["chunkBytes"]) != (
        geometry["policy"]["unitBytes"],
        geometry["policy"]["chunkBytes"],
    ):
        raise ValueError("the largest planned size uses targets other than the layout policy's")
    catalogue = load_json(CATALOGUE_OBJECTS)
    ratios = [
        store["bytes"]["countsByGene"] / store["bytes"]["countsByCell"]
        for store in catalogue["perStore"].values()
    ]
    if (
        len(ratios) != catalogue["stores"]
        or abs(float(np.median(ratios)) - catalogue["byGeneOverByCellMedian"]) > 1e-9
    ):
        raise ValueError(
            "the median gene-major to cell-major ratio does not match the stores it summarises"
        )
    return {
        "unitBytes": geometry["policy"]["unitBytes"],
        "chunkBytes": geometry["policy"]["chunkBytes"],
        "cells": largest["nCells"],
        "amplificationCellMajorOnly": largest["amplificationCellMajorOnly"],
        "amplificationPaired": largest["decodeAmplificationPairedWrite"],
        "geneMajorOverCellMajorMedian": catalogue["byGeneOverByCellMedian"],
        "stores": catalogue["stores"],
        "heatmap": np.load(STORAGE_NPZ)["heatmap"].tolist(),
        "source": [
            str(LAYOUT_GEOMETRY.relative_to(ROOT)),
            str(CATALOGUE_OBJECTS.relative_to(ROOT)),
            str(STORAGE_NPZ.relative_to(ROOT)),
        ],
    }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def tahoe_cell_lines() -> list[dict]:
    """The export's cell lines, each with its Cellosaurus ID, name and organ, in the raster's code order."""
    with TAHOE_CELL_LINES.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["code"]) for row in rows] != list(range(len(rows))):
        raise ValueError(
            f"{TAHOE_CELL_LINES.name} does not list the codes 0 to {len(rows) - 1} in order"
        )
    return rows


def cell_line_labels(count: int) -> list[dict]:
    """The largest cell lines of the Tahoe-100M UMAP, named from Tahoe-100M's cell-line table, and where each sits on
    the raster: the cell-weighted median row and column of the pixels whose commonest cell line it is."""
    lines = tahoe_cell_lines()
    codes = [row["cellLine"] for row in lines]
    names = {row["cellLine"]: row["name"] for row in lines}
    organs = {row["cellLine"]: row["organ"] for row in lines}
    totals: dict[str, int] = defaultdict(int)
    with TAHOE_CLUSTER_LINES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            totals[row["cellLine"]] += int(row["cells"])
    with np.load(TAHOE_RASTER) as raster:
        cells, dominant = raster["cells"], raster["commonestCellLine"]
    labels = []
    for code in sorted(totals, key=lambda c: -totals[c])[:count]:
        rows, columns = np.nonzero(dominant == codes.index(code))
        if rows.size == 0:
            raise ValueError(f"{code} is the commonest cell line of no pixel of the UMAP raster")
        weights = cells[rows, columns].astype(float)
        labels.append(
            {
                "cellLine": code,
                "name": names[code],
                "organ": organs[code],
                "cells": totals[code],
                "row": weighted_median(rows, weights),
                "column": weighted_median(columns, weights),
            }
        )
    return labels


def cluster_compositions(clusters: list[dict]) -> list[dict]:
    """The clusters, each with its cell lines that hold at least PIE_SHARE of its cells, largest first, as indices
    into the UMAP raster's cell lines with their shares. The commonest line and its share must match the cluster."""
    codes = [row["cellLine"] for row in tahoe_cell_lines()]
    cells: dict[int, dict[str, int]] = defaultdict(dict)
    with TAHOE_CLUSTER_LINES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cells[int(row["cluster"])][row["cellLine"]] = int(row["cells"])
    composed = []
    for cluster in clusters:
        counts = cells.get(cluster["cluster"])
        if not counts or sum(counts.values()) != cluster["cells"]:
            raise ValueError(
                f"{TAHOE_CLUSTER_LINES.name} does not hold the cells of cluster {cluster['cluster']}"
            )
        ranked = sorted(counts.items(), key=lambda item: -item[1])
        top, top_cells = ranked[0]
        if top != cluster["commonestCellLine"] or not math.isclose(
            top_cells / cluster["cells"], cluster["commonestCellLineFraction"]
        ):
            raise ValueError(
                f"cluster {cluster['cluster']}'s commonest cell line differs between the two results"
            )
        composed.append(
            {
                **cluster,
                "composition": [
                    {"cellLine": codes.index(code), "share": n / cluster["cells"]}
                    for code, n in ranked
                    if n / cluster["cells"] >= PIE_SHARE
                ],
            }
        )
    return composed


def tahoe_umap() -> dict:
    """Summary and cluster purity of the 95.6-million-cell Tahoe-100M UMAP embedding and Leiden clustering."""
    tahoe = load_json(TAHOE_RESULTS)
    full = tahoe["fullStore"]
    return {
        "cells": tahoe["dataset"]["cells"],
        "cellsKept": full["cellsKept"],
        "leidenClusters": full["leidenClusters"],
        "cellLines": full["cellLines"],
        "medianPurity": full["medianCommonestCellLineFraction"],
        "analysisHours": tahoe["totals"]["full"]["analysisHours"],
        "peakAnalysisGiB": max(
            tahoe["stages"][s]["full"]["peakCgroupGiB"]
            for s in (
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
        ),
        "raster": str(TAHOE_RASTER.relative_to(ROOT)),
        "clusters": cluster_compositions(full["clusters"]),
        "labels": cell_line_labels(LABELLED_CELL_LINES),
        "stages": {
            s: {
                "minutes": tahoe["stages"][s]["full"]["seconds"] / 60.0,
                "peakGiB": tahoe["stages"][s]["full"]["peakCgroupGiB"],
            }
            for s in (
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
        },
        "source": [
            str(TAHOE_RESULTS.relative_to(ROOT)),
            str(TAHOE_RASTER.relative_to(ROOT)),
            str(TAHOE_CELL_LINES.relative_to(ROOT)),
            str(TAHOE_CLUSTER_LINES.relative_to(ROOT)),
        ],
    }


def step_memory(times: dict) -> dict:
    """Memory decoupling across 7 orders of magnitude of raw count-matrix size: planned block memory stays flat
    at the budget across 1M and 10M cells (memory_read_admission.json), and empirical stage peaks across 1M (20 GiB CSR),
    10M (191 GiB CSR), and 95.6M (1,560 GiB CSR) show streamed count passes staying bounded while O(N) graph stages scale."""
    seconds_and_peaks: dict[tuple[str, str], list[float]] = defaultdict(list)
    with STEP_MEASUREMENTS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["tool"] == "cytearc"
                and row["dataset"] in ("embryo_1000000", "embryo_10000000")
                and row["peakCgroupBytes"]
            ):
                seconds_and_peaks[(row["dataset"], row["step"])].append(
                    float(row["peakCgroupBytes"]) / 1024**3
                )
    tahoe = load_json(TAHOE_RESULTS)
    t_stages = tahoe["stages"]

    def max_mean(dataset: str, steps: tuple[str, ...]) -> float:
        return max(float(np.mean(seconds_and_peaks[(dataset, s)])) for s in steps)

    stream_steps, graph_steps = STREAMED_STEPS, GRAPH_STEPS
    nonzeros = load_json(
        ROOT / "experiments" / "benchmarks" / "records" / "rna_workflow_runs.json"
    )["inputs"]

    def csr_gib(dataset: str) -> float:
        """Uncompressed CSR size of a sample: a float32 value and an int64 column index per stored count."""
        return nonzeros[dataset]["nnz"] * 12 / 2**30

    return {
        "scales": [
            {
                "label": "1M",
                "csrGiB": csr_gib("embryo_1000000"),
                "cells": 1000000,
                "streamedPeakGiB": max_mean("embryo_1000000", stream_steps),
                "graphPeakGiB": max_mean("embryo_1000000", graph_steps),
                "markersPeakGiB": float(
                    np.mean(seconds_and_peaks[("embryo_1000000", "findMarkers")])
                ),
            },
            {
                "label": "10M",
                "csrGiB": csr_gib("embryo_10000000"),
                "cells": 10000000,
                "streamedPeakGiB": max_mean("embryo_10000000", stream_steps),
                "graphPeakGiB": max_mean("embryo_10000000", graph_steps),
                "markersPeakGiB": float(
                    np.mean(seconds_and_peaks[("embryo_10000000", "findMarkers")])
                ),
            },
            {
                "label": "95.6M",
                "csrGiB": 1560.4,
                "cells": 95624334,
                "streamedPeakGiB": max(t_stages[s]["full"]["peakCgroupGiB"] for s in stream_steps),
                "graphPeakGiB": max(t_stages[s]["full"]["peakCgroupGiB"] for s in graph_steps),
                "markersPeakGiB": t_stages["findMarkers"]["full"]["peakCgroupGiB"],
            },
        ],
        "source": [
            str(STEP_MEASUREMENTS.relative_to(ROOT)),
            str(TAHOE_RESULTS.relative_to(ROOT)),
        ],
    }


def tahoe_stages(order: list[str]) -> dict:
    """Minutes and peak container memory (GiB) of each analysis stage of the 95.6-million-cell Tahoe-100M run over
    its replicates, in the given stage order. A replicate's stage minutes are the sum over its steps and its stage
    peak the highest of its steps; each stage then summarises the replicates. The collector keeps only replicates
    whose every step completed, so a step missing from a replicate is an error."""
    tahoe = load_json(TAHOE_RESULTS)
    replicates = tahoe["fullReplicates"]
    steps = {name: by_tool["cytearc"] for name, by_tool in STAGES}
    rows = []
    for name in order:
        minutes, peaks = [], []
        for replicate in replicates:
            missing = [step for step in steps[name] if step not in replicate["stages"]]
            if missing:
                raise ValueError(
                    f"Tahoe-100M {name}: replicate {replicate['replicate']} has no record of {missing}"
                )
            runs = [replicate["stages"][step] for step in steps[name]]
            minutes.append(sum(run["seconds"] for run in runs) / 60.0)
            peaks.append(max(run["peakCgroupGiB"] for run in runs))
        rows.append(
            {
                "stage": name,
                "steps": list(steps[name]),
                "minutes": point_summary(minutes),
                "peakGiB": point_summary(peaks),
            }
        )
    return {
        "cells": tahoe["dataset"]["cells"],
        "rows": rows,
        "source": [str(TAHOE_RESULTS.relative_to(ROOT))],
    }


def resource_trace() -> dict:
    """One scaling run's container memory, CPU cores busy and network receive rate, sampled once a second from the
    start of filtering, with the run's stages as spans of time and the memory budget and CPU limit they ran with."""
    trace = load_json(RESOURCE_TRACE / "resourceTrace.json")
    resources = load_json(RESOURCE_TRACE / "resources.json")
    columns = trace["columns"]
    if columns[:5] != ["seconds", "memoryBytes", "cpuSeconds", "networkReceivedBytes", "stage"]:
        raise ValueError(f"unexpected trace columns {columns}")
    rows = trace["rows"]
    first = next((i for i, row in enumerate(rows) if row[4] not in TRACE_LEFT_OUT), None)
    if first is None:
        raise ValueError("the trace holds no analysis step")
    late = sorted({row[4] for row in rows[first:]} & set(TRACE_LEFT_OUT))
    if late:
        raise ValueError(f"the trace runs {late} after the analysis began")
    rows = rows[first:]
    group_of = {step: name for name, by_tool in STAGES for step in by_tool["cytearc"]} | TRACE_NAMES
    unknown = sorted({row[4] for row in rows} - set(group_of))
    if unknown:
        raise ValueError(f"the trace names steps that no trace stage holds: {unknown}")
    ran = {row[4] for row in rows}
    limits = {
        (stage["memoryBudgetBytes"], stage["cpuLimit"])
        for step, stage in resources["stages"].items()
        if step in ran
    }
    if len(limits) != 1:
        raise ValueError(
            f"the stages of {resources['runTag']} ran with different memory budgets or CPU limits {sorted(limits)};"
            " the trace panel draws one budget and one limit"
        )
    ((budget_bytes, cpu_limit),) = limits
    gaps = [row[0] for row in rows if None in row[:4]]
    if gaps:
        raise ValueError(f"{len(gaps)} trace samples lack a value, the first at {gaps[0]} s")
    seconds = np.array([row[0] for row in rows], dtype=float) - rows[0][0]
    minutes = seconds / 60
    interval = np.diff(seconds)
    spans: list[dict] = []
    for minute, row in zip(minutes, rows):
        stage = group_of[row[4]]
        if spans and spans[-1]["stage"] == stage:
            continue
        if spans:
            spans[-1]["end"] = round(float(minute), 4)
        spans.append({"stage": stage, "start": round(float(minute), 4)})
    spans[-1]["end"] = round(float(minutes[-1]), 4)
    return {
        "runTag": resources["runTag"],
        "cells": resources["cells"],
        "memoryBudgetGiB": budget_bytes / 2**30,
        "cpuLimit": cpu_limit,
        "samples": {
            "minutes": [round(float(m), 4) for m in minutes],
            "memoryGiB": [round(row[1] / 2**30, 4) for row in rows],
        },
        # Rates over each interval between samples, placed at the interval's end.
        "rates": {
            "minutes": [round(float(m), 4) for m in minutes[1:]],
            "cpuCores": [round(float(c), 3) for c in np.diff([row[2] for row in rows]) / interval],
            "networkMBps": [
                round(float(b), 3) for b in np.diff([row[3] for row in rows]) / interval / 1e6
            ],
        },
        "stages": spans,
        "source": [
            str((RESOURCE_TRACE / "resourceTrace.json").relative_to(ROOT)),
            str((RESOURCE_TRACE / "resources.json").relative_to(ROOT)),
        ],
    }


def main() -> None:
    times = analysis_time()
    stages = stage_times(times)
    # The Tahoe-100M stage charts list the stages in the stage-time panel's order, the workflow's.
    order = [stage["stage"] for stage in stages["stages"]]
    for name, values in (
        ("store_layout", store_layout()),
        ("tahoe_umap", tahoe_umap()),
        ("tahoe_stages", tahoe_stages(order)),
        ("step_memory", step_memory(times)),
        ("peak_memory", peak_memory(times)),
        ("analysis_time", times),
        ("stage_times", stages),
        ("resource_trace", resource_trace()),
        ("budget_sweep", budget_sweep()),
        ("core_time_scaling", core_time_scaling()),
    ):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(values, indent=1, allow_nan=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
