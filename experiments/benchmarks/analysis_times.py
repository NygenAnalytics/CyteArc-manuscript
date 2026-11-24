"""analysis times of the tool comparison, and the comparisons that the text reports."""

import csv
import math
from collections import defaultdict

import numpy as np
from scipy import stats

from experiments.runtime.common import DERIVED, ROOT, load_json, write_csv_at, write_json_at
from experiments.runtime.records import NEIGHBOR_INDEX_STEPS

DATA = ROOT / "experiments/benchmarks/records"
BENCHMARKS_DERIVED = DERIVED / "benchmarks"
STEP_MEASUREMENTS = BENCHMARKS_DERIVED / "toolkit_step_measurements.csv"


CYTEARC_ARMS = ("cytearc", "cytearc_limit16", "cytearc_limit8")
CENSUS = {"census_1000000": "embryo_1000000", "census_10000000": "embryo_10000000"}
SPCA = "_spca"

# The analysis stages of the ten-million-cell embryo workflow compared stage by stage, in the order the workflow runs
# them, with the recorded steps of each tool that each stage sums.
STAGE_DATASET = "embryo_10000000"
STAGE_TOOLS = ("cytearc", "bpcells_cached")
STAGES = (
    (
        "filter–PCA",
        {
            "cytearc": ("filterCells", "markHvgs", "runNormalization", "runPca"),
            "bpcells_cached": ("filterCells", "runNormalization", "markHvgs", "runPca"),
        },
    ),
    (
        "neighbors",
        {
            "cytearc": ("buildAnnIndex", "queryNeighbors", "buildConnectivityMap"),
            "bpcells_cached": ("queryNeighbors",),
        },
    ),
    (
        "UMAP",
        {"cytearc": ("buildEmbeddingInitialization", "runUmap"), "bpcells_cached": ("runUmap",)},
    ),
    ("Leiden", {"cytearc": ("runLeiden",), "bpcells_cached": ("runLeiden",)}),
    ("markers", {"cytearc": ("findMarkers",), "bpcells_cached": ("findMarkers",)}),
)
# Steps inside the analysis time that no stage holds: CyteArc reopening its prepared store.
UNSHOWN = {"cytearc": ("createObject",), "bpcells_cached": ()}
# A run's stages and unshown steps must add up to its analysis time within this many minutes.
STAGE_SUM_TOLERANCE_MINUTES = 0.05

# CyteArc's streamed count passes, whose memory the budget bounds, and its whole-atlas graph and embedding steps, which
# hold per-cell state; their peaks are compared on the one- and ten-million-cell embryo samples.
STREAMED_STEPS = ("markHvgs", "runNormalization", "runPca")
GRAPH_STEPS = ("buildAnnIndex", "queryNeighbors", "runUmap", "runLeiden")
# Marker search, which reserves up to the budget to keep its workers busy.
MARKER_STEP = "findMarkers"
# The sample whose HNSW seconds the text compares with the storage-backend runs' default index settings.
NEIGHBOR_INDEX_DATASET = "embryo_1000000"
PEAK_DATASETS = ("embryo_1000000", "embryo_10000000")
# The embryo samples whose CyteArc core-time gives the exponent between them, with their cells.
CORE_TIME_CELLS = {"embryo_1000000": 1_000_000, "embryo_10000000": 10_000_000}
# The workflow inputs, whose stored counts give each sample's uncompressed CSR size.
WORKFLOW_RUNS = DATA / "rna_workflow_runs.json"
# Bytes an uncompressed CSR matrix holds per stored count: a float32 value and an int64 column index.
CSR_BYTES_PER_VALUE = 12


def summary_rows() -> dict[tuple[str, str], list[dict]]:
    """Workflow rows by (dataset, tool), with the Census datasets as the embryo samples."""
    rows: dict[tuple[str, str], list[dict]] = {}
    for row in load_json(BENCHMARKS_DERIVED / "toolkit_comparisons.json")["workflow"]:
        rows.setdefault((CENSUS.get(row["dataset"], row["dataset"]), row["tool"]), []).append(row)
    return rows


def smallest_container(runs: list[dict]) -> list[dict]:
    """The completed runs on the smallest container on which the tool completed."""
    done = [r for r in runs if r.get("analysisTimeMinutes") is not None]
    if not done:
        return []
    smallest = min(r["completedAtGib"] for r in done)
    return [r for r in done if r["completedAtGib"] == smallest]


def crashed(runs: list[dict]) -> bool:
    return any((r.get("outcome") or "").startswith("crashed") for r in runs)


def arms(rows: dict[tuple[str, str], list[dict]]) -> tuple[list[dict], list[dict]]:
    """Every arm with a completed run, marking the scaled-PCA arms that stand in for their tool in
    Fig. 3c (where the default UMAP start crashed or was not run), and the arms with none."""
    drawn, missing = [], []
    for (dataset, tool), runs in sorted(rows.items()):
        done = smallest_container(runs)
        if not done:
            missing.append({"dataset": dataset, "tool": tool, "outcome": runs[-1].get("outcome")})
            continue
        default = rows.get((dataset, tool.removesuffix(SPCA)))
        stands_in = (
            tool.removesuffix(SPCA)
            if tool.endswith(SPCA)
            and (default is None or (not smallest_container(default) and crashed(default)))
            else None
        )
        analysis = [r["analysisTimeMinutes"] for r in done]
        preparation = [r["preparationMinutes"] for r in done]
        workflow = [r["workflowMinutes"] for r in done]
        peaks = [r["peakCgroupGib"] for r in done]
        drawn.append(
            {
                "dataset": dataset,
                "tool": tool,
                "standsInFor": stands_in,
                "n": len(done),
                "replicates": [r["replicate"] for r in done],
                "containerGib": done[0]["completedAtGib"],
                "cpu": done[0]["cpu"],
                "sourceRecords": [r.get("sourceRecord") for r in done],
                "analysisMinutes": analysis,
                "preparationMinutes": preparation,
                "workflowMinutes": workflow,
                "meanAnalysisMinutes": float(np.mean(analysis)),
                "geoMeanAnalysisMinutes": float(np.exp(np.mean(np.log(analysis)))),
                "sdAnalysisMinutes": float(np.std(analysis, ddof=1)) if len(analysis) > 1 else None,
                "meanPreparationMinutes": float(np.mean(preparation)),
                "meanWorkflowMinutes": float(np.mean(workflow)),
                "peakCgroupGib": peaks,
                "meanPeakCgroupGib": float(np.mean(peaks)),
                "completedAtOtherContainers": sorted(
                    {r["completedAtGib"] for r in runs if r.get("analysisTimeMinutes") is not None}
                    - {done[0]["completedAtGib"]}
                ),
            }
        )
    return drawn, missing


def welch_log_ratio(a: list[float], b: list[float]) -> dict:
    """Ratio of geometric means a / b, its Welch t interval on log values and the test's P value."""
    la, lb = np.log(np.asarray(a, dtype=float)), np.log(np.asarray(b, dtype=float))
    out = {
        "ratio": math.exp(la.mean() - lb.mean()),
        "low": None,
        "high": None,
        "p": None,
        "df": None,
    }
    if la.size < 2 or lb.size < 2:
        return {**out, "descriptive": True, "note": "fewer than two runs on one side"}
    va, vb = la.var(ddof=1) / la.size, lb.var(ddof=1) / lb.size
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (la.size - 1) + vb**2 / (lb.size - 1))
    t = stats.t.ppf(0.975, df)
    diff = la.mean() - lb.mean()
    return {
        **out,
        "low": math.exp(diff - t * se),
        "high": math.exp(diff + t * se),
        "p": float(stats.ttest_ind(la, lb, equal_var=False).pvalue),
        "df": float(df),
        "descriptive": False,
    }


def comparisons(drawn: list[dict]) -> list[dict]:
    by_key = {(a["dataset"], a["tool"]): a for a in drawn}
    rows = []
    for (dataset, cytearc_arm), cytearc in sorted(by_key.items()):
        if cytearc_arm not in CYTEARC_ARMS:
            continue
        for (other_dataset, tool), other in sorted(by_key.items()):
            if other_dataset != dataset or tool.startswith("cytearc"):
                continue
            test = welch_log_ratio(cytearc["analysisMinutes"], other["analysisMinutes"])
            note = test.pop("note", None)
            if other["completedAtOtherContainers"]:
                note = "; ".join(
                    filter(
                        None,
                        [
                            note,
                            f"other runs completed on {other['completedAtOtherContainers']} GiB",
                        ],
                    )
                )
            rows.append(
                {
                    "dataset": dataset,
                    "cytearcArm": cytearc_arm,
                    "tool": tool,
                    "nCytearc": cytearc["n"],
                    "nOther": other["n"],
                    "cytearcMeanMinutes": cytearc["meanAnalysisMinutes"],
                    "otherMeanMinutes": other["meanAnalysisMinutes"],
                    "otherContainerGib": other["containerGib"],
                    **test,
                    "reduction": 1 - test["ratio"],
                    "reductionLow": None if test["high"] is None else 1 - test["high"],
                    "reductionHigh": None if test["low"] is None else 1 - test["low"],
                    # How many times faster CyteArc ran: the comparator's geometric mean over CyteArc's, and its interval.
                    "speedup": 1 / test["ratio"],
                    "speedupLow": None if test["high"] is None else 1 / test["high"],
                    "speedupHigh": None if test["low"] is None else 1 / test["low"],
                    # The share of total wall time, one-time preparation included, that CyteArc saves: one minus its
                    # mean workflow minutes over the comparator's, on arithmetic means.
                    "cytearcMeanWorkflowMinutes": cytearc["meanWorkflowMinutes"],
                    "otherMeanWorkflowMinutes": other["meanWorkflowMinutes"],
                    "workflowSaving": 1
                    - cytearc["meanWorkflowMinutes"] / other["meanWorkflowMinutes"],
                    "note": note,
                }
            )
    return rows


def step_seconds(dataset: str) -> dict[str, dict[str, float]]:
    """The seconds of each recorded step of each run on one dataset, by record and step."""
    seconds: dict[str, dict[str, float]] = defaultdict(dict)
    with STEP_MEASUREMENTS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["seconds"]:
                seconds[row["record"]][row["step"]] = float(row["seconds"])
    return seconds


def step_peaks() -> dict[str, dict[str, float]]:
    """The container peak (GiB) of each recorded step of each run, by record and step, for the steps that record one."""
    peaks: dict[str, dict[str, float]] = defaultdict(dict)
    with STEP_MEASUREMENTS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["peakCgroupBytes"]:
                peaks[row["record"]][row["step"]] = float(row["peakCgroupBytes"]) / 1024**3
    return peaks


def cytearc_step_peaks(drawn: list[dict], peaks: dict[str, dict[str, float]]) -> dict:
    """CyteArc's mean container peak of each streamed and each graph step and of marker search on the embryo samples of
    PEAK_DATASETS, over the runs whose analysis times the comparisons use, and the highest of those means in each group."""
    by_key = {(a["dataset"], a["tool"]): a for a in drawn}
    result = {}
    for dataset in PEAK_DATASETS:
        arm = by_key.get((dataset, "cytearc"))
        if arm is None:
            raise ValueError(f"CyteArc has no completed run on {dataset}")
        means = {}
        for step in STREAMED_STEPS + GRAPH_STEPS + (MARKER_STEP,):
            missing = [
                record for record in arm["sourceRecords"] if step not in peaks.get(record, {})
            ]
            if missing:
                raise ValueError(f"{missing} record no peak of {step}")
            means[step] = float(np.mean([peaks[record][step] for record in arm["sourceRecords"]]))
        result[dataset] = {
            "runs": len(arm["sourceRecords"]),
            "meanPeakGiB": means,
            "streamedPeakGiB": max(means[step] for step in STREAMED_STEPS),
            "graphPeakGiB": max(means[step] for step in GRAPH_STEPS),
            "markersPeakGiB": means[MARKER_STEP],
        }
    return result


def cytearc_neighbor_index_seconds(
    drawn: list[dict], seconds: dict[str, dict[str, float]], dataset: str
) -> dict:
    """CyteArc's mean seconds of building and querying its HNSW index (NEIGHBOR_INDEX_STEPS) on one dataset, over the
    runs whose analysis times the comparisons use. ``seconds`` holds each run's step seconds (step_seconds)."""
    arm = next((a for a in drawn if (a["dataset"], a["tool"]) == (dataset, "cytearc")), None)
    if arm is None:
        raise ValueError(f"CyteArc has no completed run on {dataset}")
    per_run = []
    for record in arm["sourceRecords"]:
        missing = [step for step in NEIGHBOR_INDEX_STEPS if step not in seconds.get(record, {})]
        if missing:
            raise ValueError(f"{record} has no step measurement of {missing}")
        per_run.append(sum(seconds[record][step] for step in NEIGHBOR_INDEX_STEPS))
    return {
        "dataset": dataset,
        "steps": list(NEIGHBOR_INDEX_STEPS),
        "seconds": per_run,
        "meanSeconds": float(np.mean(per_run)),
    }


def cytearc_core_time_exponent(drawn: list[dict]) -> dict:
    """The exponent b in core-time ∝ cells^b of CyteArc between the embryo samples of CORE_TIME_CELLS: each sample's
    mean analysis minutes times its container's CPUs, compared across the two sizes."""
    by_key = {(a["dataset"], a["tool"]): a for a in drawn}
    (small, small_cells), (large, large_cells) = sorted(
        CORE_TIME_CELLS.items(), key=lambda item: item[1]
    )
    core = {
        dataset: by_key[(dataset, "cytearc")]["meanAnalysisMinutes"]
        * by_key[(dataset, "cytearc")]["cpu"]
        for dataset in (small, large)
    }
    return {
        "datasets": [small, large],
        "coreMinutes": [core[small], core[large]],
        "exponent": math.log(core[large] / core[small]) / math.log(large_cells / small_cells),
    }


def uncompressed_csr_bytes(nonzeros: int) -> int:
    """The size of a count matrix in uncompressed CSR form: CSR_BYTES_PER_VALUE bytes per stored count, leaving out the
    row pointers."""
    if nonzeros < 0:
        raise ValueError(f"a matrix stores no fewer than 0 counts, not {nonzeros}")
    return nonzeros * CSR_BYTES_PER_VALUE


def peak_dataset_csr_bytes(inputs: dict) -> dict[str, int]:
    """The uncompressed CSR size of each embryo sample of PEAK_DATASETS, from the stored counts of the workflow inputs."""
    missing = [dataset for dataset in PEAK_DATASETS if "nnz" not in inputs.get(dataset, {})]
    if missing:
        raise ValueError(f"the workflow inputs record no stored counts for {missing}")
    return {dataset: uncompressed_csr_bytes(inputs[dataset]["nnz"]) for dataset in PEAK_DATASETS}


def stage_times(drawn: list[dict], seconds: dict[str, dict[str, float]]) -> dict:
    """Minutes of each analysis stage of CyteArc and BPCells (cached) on the ten-million-cell embryo sample, in the runs
    whose analysis times the comparisons use, and the comparator's mean minutes over CyteArc's for each stage. A run
    whose stages do not add up to its analysis time is an error."""
    by_key = {(a["dataset"], a["tool"]): a for a in drawn}
    minutes: dict[str, list[list[float]]] = {}
    for tool in STAGE_TOOLS:
        arm = by_key.get((STAGE_DATASET, tool))
        if arm is None:
            raise ValueError(f"{tool} has no completed run on {STAGE_DATASET}")
        per_run = []
        for record, analysis in zip(arm["sourceRecords"], arm["analysisMinutes"], strict=True):
            steps = seconds.get(record, {})
            wanted = [s for _, by_tool in STAGES for s in by_tool[tool]] + list(UNSHOWN[tool])
            missing = [s for s in wanted if s not in steps]
            if missing:
                raise ValueError(f"{record} has no step measurement of {missing}")
            stage_minutes = [sum(steps[s] for s in by_tool[tool]) / 60 for _, by_tool in STAGES]
            total = sum(stage_minutes) + sum(steps[s] for s in UNSHOWN[tool]) / 60
            if abs(total - analysis) > STAGE_SUM_TOLERANCE_MINUTES:
                raise ValueError(
                    f"{record}: the stages add up to {total:.2f} minutes, not the analysis time of {analysis:.2f}"
                )
            per_run.append(stage_minutes)
        minutes[tool] = per_run
    cytearc, other = STAGE_TOOLS
    stages = []
    for i, (name, by_tool) in enumerate(STAGES):
        cytearc_minutes = [run[i] for run in minutes[cytearc]]
        other_minutes = [run[i] for run in minutes[other]]
        stages.append(
            {
                "stage": name,
                "cytearcSteps": list(by_tool[cytearc]),
                "otherSteps": list(by_tool[other]),
                "cytearcMinutes": cytearc_minutes,
                "otherMinutes": other_minutes,
                "cytearcMeanMinutes": float(np.mean(cytearc_minutes)),
                "otherMeanMinutes": float(np.mean(other_minutes)),
                "otherOverCytearc": float(np.mean(other_minutes) / np.mean(cytearc_minutes)),
            }
        )
    # The stages after the count passes, whose mean minutes the text adds up for each tool.
    downstream = [stage for stage in stages if stage["stage"] != STAGES[0][0]]
    return {
        "dataset": STAGE_DATASET,
        "tool": other,
        "stages": stages,
        "downstreamStages": [stage["stage"] for stage in downstream],
        "cytearcDownstreamMeanMinutes": sum(stage["cytearcMeanMinutes"] for stage in downstream),
        "otherDownstreamMeanMinutes": sum(stage["otherMeanMinutes"] for stage in downstream),
    }


def main() -> None:
    drawn, missing = arms(summary_rows())
    compared = comparisons(drawn)
    write_json_at(
        BENCHMARKS_DERIVED / "toolkit_analysis_times.json",
        {
            "definition": (
                "Analysis time spans the first recorded step after reusable format preparation "
                "through marker-search completion; output export is excluded. Ratios compare "
                "CyteArc with comparator configurations using Welch t intervals on log times; "
                "speedup is the comparator's geometric mean over CyteArc's. Stage times split the "
                "ten-million-cell analysis times into stages, and otherOverCytearc is the "
                "comparator's mean stage minutes over CyteArc's; the downstream sums add the mean minutes "
                "of the stages after the count passes. Each arm's meanPeakCgroupGib is the mean container "
                "peak of its runs, and workflowSaving is one minus CyteArc's mean workflow minutes, "
                "preparation included, over the comparator's. CyteArc's step peaks are the mean "
                "container peaks of its streamed, graph and marker steps on the one- and ten-million-cell "
                "embryo samples, and its core-time exponent compares their mean analysis minutes "
                "times the container's CPUs. uncompressedCsrBytes gives the size of those samples' "
                "counts in uncompressed CSR form, 12 bytes per stored count. cytearcNeighborIndexSeconds "
                "gives CyteArc's mean seconds of HNSW index build and query on one million cells."
            ),
            "inputValidation": load_json(BENCHMARKS_DERIVED / "toolkit_comparisons.json")[
                "workflowInputValidation"
            ],
            "arms": drawn,
            "comparisons": compared,
            "notCompleted": missing,
            "stageTimes": stage_times(drawn, step_seconds(STAGE_DATASET)),
            "cytearcStepPeaks": cytearc_step_peaks(drawn, step_peaks()),
            "cytearcCoreTimeExponent": cytearc_core_time_exponent(drawn),
            "cytearcNeighborIndexSeconds": cytearc_neighbor_index_seconds(
                drawn, step_seconds(NEIGHBOR_INDEX_DATASET), NEIGHBOR_INDEX_DATASET
            ),
            "uncompressedCsrBytes": peak_dataset_csr_bytes(load_json(WORKFLOW_RUNS)["inputs"]),
        },
    )
    write_csv_at(BENCHMARKS_DERIVED / "toolkit_analysis_comparisons.csv", compared)
    for row in compared:
        if row["cytearcArm"] == "cytearc":
            interval = (
                ""
                if row["low"] is None
                else f" [{1 - row['high']:.1%}, {1 - row['low']:.1%}] P = {row['p']:.2g}"
            )
            print(
                f"{row['dataset']:16s} {row['tool']:22s} {row['cytearcMeanMinutes']:7.1f} vs "
                f"{row['otherMeanMinutes']:7.1f} min (n {row['nCytearc']}, {row['nOther']}): "
                f"reduction {row['reduction']:.1%}{interval}"
            )


if __name__ == "__main__":
    main()
