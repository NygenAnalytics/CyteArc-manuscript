"""summarise the budget sweeps (Fig. 3a, Extended Data Fig. 2, Supplementary Table 4)."""

import re
import statistics

from experiments.runtime.common import DERIVED, ROOT, write_json_at
from experiments.runtime.records import (
    NEIGHBOR_INDEX_STEPS,
    STAGES,
    conversion_elapsed_seconds,
    elapsed_seconds,
    load_runs,
)

DATA = ROOT / "experiments/benchmarks/records"
BENCHMARKS_DERIVED = DERIVED / "benchmarks"


STREAMING = ("markHvgs", "runNormalization", "findMarkers")


def run_parameters(tag: str, run: dict) -> dict:
    funnel = run["funnel"]
    outcomes = {row["stage"]: row for row in funnel["outcomes"]}
    creation = outcomes["createStore"]
    backend = {
        "local": "modalLocalDisk",
        "modalLocalDisk": "modalLocalDisk",
        "r2": "objectStore",
        "objectStore": "objectStore",
    }[funnel["storeBackend"]]
    return {
        "tag": tag,
        "dataset": "brain1.3M" if funnel["nRows"] == 1_306_127 else "embryo1M",
        "budgetGiB": int(creation["cytearcMemoryBudget"] / 1024**3),
        "workers": creation["workers"],
        "replicate": int(re.search(r"_replicate_(\d+)", tag).group(1)),
        "storeBackend": backend,
        "measurementPurpose": run["measurementPurpose"],
    }


def cloud_runs(bundle: dict[str, dict]) -> list[dict]:
    rows = []
    for tag, run in sorted(bundle.items()):
        funnel, stages = run["funnel"], run["stages"]
        if not stages:
            print(f"{tag}: no results yet")
            continue
        info = run_parameters(tag, run)
        complete = funnel is not None and all(
            stage in stages and stages[stage]["status"] == "ok" for stage in STAGES
        )
        stage_seconds = {stage: float(rec["seconds"] or 0.0) for stage, rec in stages.items()}
        rows.append(
            {
                **info,
                "machine": "modal 8 CPU / 32 GiB",
                "complete": complete,
                "elapsedSeconds": elapsed_seconds(funnel) if complete else None,
                "conversionElapsedSeconds": (
                    conversion_elapsed_seconds(funnel) if complete else None
                ),
                "stageSumSeconds": sum(stage_seconds.values()),
                "datasetDownloadSeconds": funnel.get("datasetDownloadSeconds") if funnel else None,
                "peakRssBytes": funnel.get("peakRssBytes")
                if funnel
                else max((rec.get("peakRssBytes") or 0) for rec in stages.values()),
                "peakCgroupBytes": funnel.get("peakCgroupBytes")
                if funnel
                else max((rec.get("peakCgroupBytes") or 0) for rec in stages.values()),
                "stageSeconds": stage_seconds,
                "streamingSeconds": sum(stage_seconds.get(s, 0.0) for s in STREAMING),
                "failedStage": funnel.get("failedStage")
                if funnel
                else next((s for s, r in stages.items() if r["status"] != "ok"), None),
            }
        )
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if not row["complete"]:
            continue
        groups.setdefault(
            (
                row.get("dataset", "embryo1M"),
                row["storeBackend"],
                row["budgetGiB"],
                row["workers"],
            ),
            [],
        ).append(row)
    summary = []
    for (dataset, backend, budget, workers), group in sorted(groups.items()):
        elapsed = [float(r["elapsedSeconds"]) for r in group]
        peaks = [float(r["peakRssBytes"]) for r in group if r["peakRssBytes"]]
        summary.append(
            {
                "dataset": dataset,
                "storeBackend": backend,
                "budgetGiB": budget,
                "workers": workers,
                "replicates": len(group),
                "elapsedSecondsMean": statistics.fmean(elapsed),
                "elapsedSecondsSd": statistics.stdev(elapsed) if len(elapsed) > 1 else None,
                "peakRssBytesMean": statistics.fmean(peaks) if peaks else None,
                "peakFractionOfBudget": (statistics.fmean(peaks) / (budget * 2**30))
                if peaks
                else None,
                "streamingSecondsMean": statistics.fmean(
                    float(r["streamingSeconds"]) for r in group
                ),
                "stageSecondsMean": {
                    stage: statistics.fmean(float(r["stageSeconds"].get(stage, 0.0)) for r in group)
                    for stage in STAGES
                },
                "neighborIndexSecondsMean": statistics.fmean(
                    sum(float(r["stageSeconds"].get(step, 0.0)) for step in NEIGHBOR_INDEX_STEPS)
                    for r in group
                ),
            }
        )
    return summary


def selected_replicates(rows: list[dict]) -> list[dict]:
    selected = {}
    for row in rows:
        if row["measurementPurpose"] != "independent_replicate":
            continue
        key = (
            row["dataset"],
            row["storeBackend"],
            row["budgetGiB"],
            row["workers"],
            row["replicate"],
        )
        if key in selected:
            raise ValueError(f"Duplicate RNA resource-comparison repetition: {key}")
        selected[key] = row
    return sorted(selected.values(), key=lambda row: row["tag"])


def main() -> None:
    bundle = load_runs("rna_resource_runs")
    rows = selected_replicates(cloud_runs(bundle))
    summary = summarise(rows)
    write_json_at(
        BENCHMARKS_DERIVED / "rna_resource_comparisons.json",
        {
            "description": "Budget and worker sweep of the 1M workflow: object store on Modal and Modal local disk, plus the 1.3M brain on the object store. elapsedSeconds is the funnel's elapsed time without the dataset download (stages that ran at the same time would count once, and none did; createStore's input setup is included), conversionElapsedSeconds its part in the conversion stages; stageSeconds are the stages' own times; peak is process-tree RSS.",
            "runs": rows,
            "summary": summary,
        },
    )
    for entry in summary:
        print(
            f"{entry['dataset']:<10} {entry['storeBackend']:<14} budget {entry['budgetGiB']:>2} GiB workers {entry['workers']} n={entry['replicates']}: "
            f"elapsed {entry['elapsedSecondsMean'] / 60:6.1f} min, peak {(entry['peakRssBytesMean'] or 0) / 2**30:5.1f} GiB "
            f"({(entry['peakFractionOfBudget'] or 0):.0%} of budget), streaming stages {entry['streamingSecondsMean'] / 60:5.1f} min"
        )


if __name__ == "__main__":
    main()
