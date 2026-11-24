"""Summarise the memory budget sweep into .build/derived/benchmarks/budget_sweep.json.

For each budget, the minutes from input conversion through PCA and the container peaks of its
three runs; for the two count policies screened at 4 GiB, the same measurements of each run.
"""

import json
import statistics

from experiments.benchmarks import HERE
from experiments.runtime.common import DERIVED, GIB, ROOT, write_json

RECORD = HERE / "records" / "memory_preprocessing.json"
OUT = DERIVED / "benchmarks"


def run_summary(run: dict) -> dict:
    """Minutes from the first step's start to the end of PCA, and the largest peaks of the steps."""
    steps = run["steps"]
    unit, chunk = map(int, run["countPolicy"]["requested"].split(","))
    return {
        "budgetGiB": run["budgetBytes"] / GIB,
        "replicate": run["replicate"],
        "unitBytes": unit,
        "chunkBytes": chunk,
        "minutes": (steps["runPca"]["end"] - min(step["start"] for step in steps.values())) / 60,
        "containerPeakGiB": max(step["peakCgroupBytes"] for step in steps.values()) / GIB,
        "processTreeRssPeakGiB": max(step["peakRssBytes"] for step in steps.values()) / GIB,
    }


def budget_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for budget in sorted({run["budgetGiB"] for run in runs}):
        group = [run for run in runs if run["budgetGiB"] == budget]
        policies = {(run["unitBytes"], run["chunkBytes"]) for run in group}
        if len(policies) != 1:
            raise ValueError(f"the runs under {budget:g} GiB use different count policies")
        ((unit, chunk),) = policies
        minutes = [run["minutes"] for run in group]
        peaks = [run["containerPeakGiB"] for run in group]
        rows.append(
            {
                "budgetGiB": budget,
                "runs": len(group),
                "unitBytes": unit,
                "chunkBytes": chunk,
                "meanMinutes": statistics.mean(minutes),
                "minMinutes": min(minutes),
                "maxMinutes": max(minutes),
                "meanContainerPeakGiB": statistics.mean(peaks),
                "maxContainerPeakGiB": max(peaks),
                "belowBudget": all(peak < budget for peak in peaks),
            }
        )
    return rows


def main() -> None:
    document = json.loads(RECORD.read_text())
    runs = [run_summary(run) for run in document["records"].values()]
    summary = {
        "source": str(RECORD.relative_to(ROOT)),
        "budgets": budget_rows(runs),
        "screens": {name: run_summary(run) for name, run in document["screens"].items()},
    }
    path = write_json("budget_sweep.json", summary, directory=OUT)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
