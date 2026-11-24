"""Each method's disagreement with the author labels on the held-out kidney and lung query studies, with the cells and
donors of each tissue's reference and query, from records/cross_study/evaluation.json, and the bytes Seurat and R
Symphony save to reuse each held-out reference, from records/cross_study/<tissue>/<method>.json, into
.build/derived/mapping/held_out_mapping.json.

    uv run --locked python -m experiments.mapping.held_out_summary
"""

from pathlib import Path

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

CROSS_STUDY = ROOT / "experiments" / "mapping" / "records" / "cross_study"
EVALUATION = CROSS_STUDY / "evaluation.json"
SUMMARY = DERIVED / "mapping" / "held_out_mapping.json"
# The held-out query study of each tissue beyond blood, by its key in the evaluation.
HELD_OUT_STUDIES = {"kidney": "Reck_2025", "lung": "Meyer_2019"}
# The CyteArc run whose inputs record the tissue's reference and query.
CYTEARC_METHOD = "unintegrated"
# The tissues of the held-out references, and the comparators that save a reference object with a neighbor index.
REFERENCE_TISSUES = ("blood", "kidney", "lung")
SAVED_REFERENCE_METHODS = ("seurat_map", "symphony")


def held_out(mapping: list[dict]) -> dict:
    """For each tissue of HELD_OUT_STUDIES, the cells and donors of its reference and query and each method's accuracy
    and disagreement on the query study: the share of the query's cells whose label differs from the author's, and how
    many percentage points the method's disagreement lies above CyteArc's (below it when negative). Every
    method must have scored each of the query's cells. ``mapping`` is the evaluation's list of scored runs."""
    tissues = {}
    for tissue, study in HELD_OUT_STUDIES.items():
        runs = {run["method"]: run for run in mapping if run["tissue"] == tissue}
        if CYTEARC_METHOD not in runs:
            raise ValueError(f"the evaluation holds no {CYTEARC_METHOD} run on {tissue}")
        parts = runs[CYTEARC_METHOD]["inputs"]["parts"]
        methods = {}
        for method, run in sorted(runs.items()):
            scored = run["evaluation"]["studies"][study]
            cells = sum(int(donor["cells"]) for donor in scored["donorConfusionMatrices"])
            if cells != parts["query"]["cells"]:
                raise ValueError(
                    f"{method} on {study} was scored on {cells} cells, not the query's {parts['query']['cells']}"
                )
            methods[method] = {
                "accuracy": scored["accuracy"],
                "disagreement": 1 - scored["accuracy"],
            }
        baseline = methods[CYTEARC_METHOD]["disagreement"]
        for scores in methods.values():
            scores["disagreementPointsOverCytearc"] = (scores["disagreement"] - baseline) * 100
        tissues[tissue] = {
            "study": study,
            "reference": {
                "cells": int(parts["reference"]["cells"]),
                "donors": len(parts["reference"]["donors"]),
            },
            "query": {
                "cells": int(parts["query"]["cells"]),
                "donors": len(parts["query"]["donors"]),
            },
            "methods": methods,
        }
    return tissues


def saved_reference_bytes(directory: Path = CROSS_STUDY) -> dict[str, dict[str, int]]:
    """For each tissue of REFERENCE_TISSUES, the bytes each method of SAVED_REFERENCE_METHODS saves to reuse the
    tissue's reference: its uncompressed RDS with its saved neighbor index, from ``directory``/<tissue>/<method>.json."""
    sizes: dict[str, dict[str, int]] = {}
    for tissue in REFERENCE_TISSUES:
        sizes[tissue] = {}
        for method in SAVED_REFERENCE_METHODS:
            path = directory / tissue / f"{method}.json"
            record = load_json(path)
            missing = [
                key for key in ("referenceRdsBytes", "referenceIndexBytes") if key not in record
            ]
            if missing:
                raise ValueError(
                    f"{path} records no {' or '.join(missing)}; both sizes of the saved reference are needed"
                )
            sizes[tissue][method] = int(record["referenceRdsBytes"]) + int(
                record["referenceIndexBytes"]
            )
    return sizes


def main() -> None:
    write_json_at(
        SUMMARY,
        {
            "source": str(EVALUATION.relative_to(ROOT)),
            "tissues": held_out(load_json(EVALUATION)["result"]["mapping"]),
            "savedReferenceBytes": saved_reference_bytes(),
        },
    )


if __name__ == "__main__":
    main()
