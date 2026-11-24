"""Median AUPRC and AUROC of each doublet method over the benchmark matrices, and the number of
matrices on which CyteArc's AUPRC is higher than each other method's, from records/doublets.json.

    uv run --locked python -m experiments.methods.doublet_summary
"""

import statistics

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

RECORD = ROOT / "experiments" / "methods" / "records" / "doublets.json"
SUMMARY = DERIVED / "methods" / "doublets.json"


def summarise(record: dict) -> dict:
    scores = {
        method: {row["dataset"]: row for row in run["result"]["datasets"]}
        for method, run in record["records"].items()
    }
    datasets = sorted(scores["cytearc"])
    if any(sorted(rows) != datasets for rows in scores.values()):
        raise ValueError("Every doublet method must cover the same benchmark matrices")
    return {
        "datasets": len(datasets),
        "medianAuprc": {
            method: statistics.median(rows[name]["auprc"] for name in datasets)
            for method, rows in scores.items()
        },
        "medianAuroc": {
            method: statistics.median(rows[name]["auroc"] for name in datasets)
            for method, rows in scores.items()
        },
        "cytearcHigherThan": {
            method: sum(scores["cytearc"][name]["auprc"] > rows[name]["auprc"] for name in datasets)
            for method, rows in scores.items()
            if method != "cytearc"
        },
    }


def main() -> None:
    write_json_at(SUMMARY, summarise(load_json(RECORD)))


if __name__ == "__main__":
    main()
