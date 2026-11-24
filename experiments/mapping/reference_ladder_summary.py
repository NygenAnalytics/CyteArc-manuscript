"""Accuracy, disagreement, macro F1, donor-bootstrap intervals, rare-type means and step times of every
reference ladder run, from records/reference_ladder.json, with each run's disagreement against that of
CyteArc's unintegrated run on the same reference, and each comparator's time to annotate the query from its
saved reference against CyteArc's from its stored reference.

    uv run --locked python -m experiments.mapping.reference_ladder_summary
"""

import numpy as np

from experiments.mapping.label_accuracy import classification_statistics, donor_bootstrap
from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

RECORD = ROOT / "experiments" / "mapping" / "records" / "reference_ladder.json"
SUMMARY = DERIVED / "mapping" / "reference_ladder.json"
# The run whose disagreement every run on the same reference is compared with.
BASELINE_METHOD = "unintegrated"
# CyteArc's runs, unintegrated and Harmony-integrated, whose stored-reference times the comparators' are compared with.
CYTEARC_METHODS = ("unintegrated", "integrated")


def rare_type_tiers(statistics: dict, rare_types: dict) -> dict:
    """Mean recall and F1 over the rare types of each frequency tier that the query holds."""
    tiers = {}
    for tier in ("belowOnePercent", "belowOneTenthPercent"):
        values = {
            label: statistics["perType"][label]
            for label in rare_types[tier]
            if label in statistics["perType"]
        }
        tiers[tier] = {
            "types": sorted(values),
            "meanRecall": float(np.mean([v["recall"] for v in values.values()]))
            if values
            else None,
            "meanF1": float(np.mean([v["f1"] for v in values.values()])) if values else None,
        }
    return tiers


def step_sums(steps: dict) -> dict:
    """Seconds to map the query from a stored reference (registration or loading the saved
    reference, then the query) and for a fresh build, registration and query."""
    if ("referenceRegistration" in steps) == ("preparedReferenceLoad" in steps):
        raise ValueError(
            "A ladder run must time either referenceRegistration (CyteArc) or "
            f"preparedReferenceLoad (comparators); got {sorted(steps)}"
        )
    access = steps.get("referenceRegistration", steps.get("preparedReferenceLoad"))
    return {
        "preparedReferenceAndQuerySeconds": access + steps["queryAnnotation"],
        "freshReferenceAndQuerySeconds": sum(steps.values()),
    }


def summarize(record: dict) -> dict:
    labels = record["confusionLabels"]
    rows = []
    for row in record["rows"]:
        summary = {
            key: row[key] for key in ("referencePart", "referenceCells", "method", "outcome")
        }
        if row["outcome"] == "completed":
            matrices = np.asarray([donor["confusion"] for donor in row["donorConfusion"]])
            pooled = matrices.sum(axis=0)
            if int(pooled.sum()) != record["queryCells"]:
                raise ValueError(
                    f"{row['referencePart']} {row['method']} labels {int(pooled.sum())} query "
                    f"cells; the ladder query has {record['queryCells']}"
                )
            evaluated = {labels[i] for i in np.flatnonzero(pooled.sum(axis=1))}
            statistics = classification_statistics(pooled, labels, evaluated)
            summary.update(step_sums(row["stepSeconds"]))
            summary.update(
                {
                    "accuracy": statistics["accuracy"],
                    # The share of query cells whose label differs from the author's.
                    "disagreement": 1 - statistics["accuracy"],
                    "macroF1": statistics["macroF1"],
                    "perType": statistics["perType"],
                    "donorBootstrap95CI": donor_bootstrap(matrices, labels, evaluated),
                    "rareTypeTiers": rare_type_tiers(statistics, record["rareTypes"]),
                }
            )
        rows.append(summary)
    rows = stored_reference_speedups(disagreement_differences(rows))
    return {"source": str(RECORD.relative_to(ROOT)), "rows": rows}


def disagreement_differences(rows: list[dict]) -> list[dict]:
    """Each completed run with its disagreement minus that of BASELINE_METHOD on the same reference, in percentage
    points: positive when the run disagrees more with the author labels."""
    baseline = {
        row["referencePart"]: row["disagreement"]
        for row in rows
        if row["method"] == BASELINE_METHOD and "disagreement" in row
    }
    for row in rows:
        if "disagreement" not in row:
            continue
        if row["referencePart"] not in baseline:
            raise ValueError(
                f"{row['referencePart']} has no completed {BASELINE_METHOD} run to compare with"
            )
        row["disagreementPointsOverBaseline"] = 100 * (
            row["disagreement"] - baseline[row["referencePart"]]
        )
    return rows


def stored_reference_speedups(rows: list[dict]) -> list[dict]:
    """Each completed comparator run with how many times faster each CyteArc run on the same reference annotated the
    query from its stored reference: the comparator's preparedReferenceAndQuerySeconds over the CyteArc run's."""
    cytearc = {
        (row["referencePart"], row["method"]): row["preparedReferenceAndQuerySeconds"]
        for row in rows
        if row["method"] in CYTEARC_METHODS and "preparedReferenceAndQuerySeconds" in row
    }
    for row in rows:
        if row["method"] in CYTEARC_METHODS or "preparedReferenceAndQuerySeconds" not in row:
            continue
        missing = [m for m in CYTEARC_METHODS if (row["referencePart"], m) not in cytearc]
        if missing:
            raise ValueError(
                f"{row['referencePart']} has no completed CyteArc {missing} run to compare {row['method']} with"
            )
        row["cytearcSpeedupFromStoredReference"] = {
            method: row["preparedReferenceAndQuerySeconds"]
            / cytearc[(row["referencePart"], method)]
            for method in CYTEARC_METHODS
        }
    return rows


def main() -> None:
    write_json_at(SUMMARY, summarize(load_json(RECORD)))


if __name__ == "__main__":
    main()
