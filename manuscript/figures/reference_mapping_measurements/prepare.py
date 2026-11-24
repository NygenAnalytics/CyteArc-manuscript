"""Prepare the panel data of the supplementary figure of reference mapping measurements: registering finished analyses
as references, what each method saves to reuse a reference, and on the blood reference ladder, each method's time to
annotate the query from a stored reference and the rare-type cells it missed.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py. Run from the
repository root:

  uv run --locked python -B -m manuscript.figures.reference_mapping_measurements.prepare
"""

import json

import numpy as np
from experiments.mapping.held_out_summary import saved_reference_bytes
from experiments.runtime.common import DERIVED, ROOT, load_json

MAPPING = ROOT / "experiments/mapping/records"
CROSS_STUDY = MAPPING / "cross_study"
EVALUATION = CROSS_STUDY / "evaluation.json"
LADDER = MAPPING / "reference_ladder.json"
LADDER_SUMMARY = DERIVED / "mapping" / "reference_ladder.json"
OUT = ROOT / "manuscript" / "figures" / "reference_mapping_measurements"


def relative(path) -> str:
    return str(path.relative_to(ROOT))


def close(a: float, b: float, what: str, tolerance: float = 1e-6) -> None:
    if abs(a - b) > tolerance * max(1.0, abs(a), abs(b)):
        raise ValueError(f"{what}: {a} against {b}")


# ---- the held-out studies --------------------------------------------------------------------------------------

TISSUES = ("blood", "kidney", "lung")
ARMS = ("unintegrated", "integrated")
HELD_OUT_METHODS = ("seurat_map", "symphony", "celltypist", "unintegrated", "integrated")


def _evaluation() -> dict:
    """The evaluation's entries by tissue and method, checked: one entry each, and every method scored on every cell
    of each query study."""
    entries = {}
    for m in load_json(EVALUATION)["result"]["mapping"]:
        key = (m["tissue"], m["method"])
        if key in entries:
            raise ValueError(f"evaluation.json holds {key} twice")
        entries[key] = m
    if set(entries) != {(t, m) for t in TISSUES for m in HELD_OUT_METHODS}:
        raise ValueError(
            f"evaluation.json holds {sorted(entries)}, not each tissue and method once"
        )
    for tissue in TISSUES:
        query = entries[(tissue, "unintegrated")]["inputs"]["parts"]["query"]["cells"]
        for method in HELD_OUT_METHODS:
            studies = entries[(tissue, method)]["evaluation"]["studies"]
            scored = sum(
                int(d["cells"]) for s in studies.values() for d in s["donorConfusionMatrices"]
            )
            if scored != query:
                raise ValueError(
                    f"{method} on {tissue} was scored on {scored} cells, not the query's {query}"
                )
    return entries


def _raw(tissue: str, method: str) -> dict:
    record = load_json(CROSS_STUDY / tissue / f"{method}.json")
    return record.get("result", record)


def _held_out_parts() -> dict:
    entries = _evaluation()
    parts = {}
    for tissue in TISSUES:
        p = entries[(tissue, "unintegrated")]["inputs"]["parts"]
        if (
            entries[(tissue, "integrated")]["inputs"]["parts"]["reference"]["cells"]
            != p["reference"]["cells"]
        ):
            raise ValueError(f"the two CyteArc arms on {tissue} used different references")
        parts[tissue] = {
            "referenceCells": int(p["reference"]["cells"]),
            "queryCells": int(p["query"]["cells"]),
        }
    return parts


# ---- registering a finished analysis as a reference ------------------------------------------------------------


def registration_reads() -> dict:
    """What registering each held-out CyteArc reference read and wrote, and how long it took; checked against the jobs'
    raw records, and that no registration read a count byte."""
    entries = _evaluation()
    parts = _held_out_parts()
    rows = []
    for tissue in TISSUES:
        for arm in ARMS:
            entry = entries[(tissue, arm)]
            io = entry["registrationIo"]
            if io != _raw(tissue, arm)["arms"][arm]["registrationIo"]:
                raise ValueError(
                    f"{tissue} {arm}: the evaluation's registration reads differ from the job record"
                )
            if int(io["countReadBytes"]) or int(io["countReadRequests"]):
                raise ValueError(f"{tissue} {arm}: registration read count arrays")
            rows.append(
                {
                    "reference": tissue,
                    "arm": arm,
                    "referenceCells": parts[tissue]["referenceCells"],
                    "readBytes": int(io["readTransferredBytes"]),
                    "countReadBytes": int(io["countReadBytes"]),
                    "writtenBytes": int(io["writeTransferredBytes"]),
                    "seconds": float(entry["costs"]["groups"]["referenceRegistration"]["seconds"]),
                }
            )
    return {
        "rows": rows,
        "source": [relative(EVALUATION)]
        + [relative(CROSS_STUDY / t / f"{a}.json") for t in TISSUES for a in ARMS],
    }


# ---- the blood reference ladder ----------------------------------------------------------------------------------

LADDER_METHODS = (
    "seurat_map",
    "symphony",
    "celltypist_hvg",
    "celltypist",
    "unintegrated",
    "integrated",
)
LADDER_PARTS = ("reference_500000", "reference_1500000", "reference_3000000", "reference_all")
# R's dgCMatrix indexes at most this many nonzero values.
R_NONZERO_LIMIT = 2**31 - 1


def _ladder_rows() -> dict:
    rows = {(r["referencePart"], r["method"]): r for r in load_json(LADDER)["rows"]}
    if set(rows) != {(p, m) for p in LADDER_PARTS for m in LADDER_METHODS}:
        raise ValueError("the ladder must hold one run per reference size and method")
    return rows


def _ladder_summary() -> dict:
    """The derived summary's rows, checked against the ladder record's cells and outcomes."""
    rows = _ladder_rows()
    summary = {(r["referencePart"], r["method"]): r for r in load_json(LADDER_SUMMARY)["rows"]}
    if set(summary) != set(rows):
        raise ValueError("the ladder summary and record hold different runs")
    for key, row in rows.items():
        if (
            summary[key]["referenceCells"] != row["referenceCells"]
            or summary[key]["outcome"] != row["outcome"]
        ):
            raise ValueError(f"the ladder summary and record differ on {key}")
    return summary


def _outcome(row: dict) -> str:
    outcome = row["outcome"]
    if outcome == "completed":
        return "pass"
    if outcome == "out of memory":
        return "oom"
    if outcome == "not attempted":
        return "none"
    if "nonzero limit" in outcome:
        return "limit"
    raise ValueError(f"unknown ladder outcome {outcome!r}")


def ladder_sizes() -> list[dict]:
    rows = _ladder_rows()
    sizes = []
    for part in LADDER_PARTS:
        row = rows[(part, "unintegrated")]
        if any(rows[(part, m)]["referenceCells"] != row["referenceCells"] for m in LADDER_METHODS):
            raise ValueError(f"the methods at {part} used references of different sizes")
        sizes.append(
            {
                "part": part,
                "cells": int(row["referenceCells"]),
                "nonzeros": int(row["referenceNonzeroCounts"]),
                "overRLimit": int(row["referenceNonzeroCounts"]) > R_NONZERO_LIMIT,
            }
        )
    return sizes


def _ladder_steps(row: dict, summary: dict) -> dict:
    """A completed run's time split as on the held-out studies, checked against the summary's totals."""
    steps = row["stepSeconds"]
    build = steps["referencePreparation"] + steps.get("referencePersistence", 0.0)
    stored = steps.get("referenceRegistration", 0.0) + steps.get("preparedReferenceLoad", 0.0)
    stored += steps["queryAnnotation"]
    what = f"{row['referencePart']} {row['method']}"
    close(stored, summary["preparedReferenceAndQuerySeconds"], f"{what} stored")
    close(build + stored, summary["freshReferenceAndQuerySeconds"], f"{what} fresh")
    return {
        "buildSeconds": build,
        "storedSeconds": stored,
        "registrationSeconds": steps.get("referenceRegistration"),
        "annotationSeconds": steps["queryAnnotation"],
    }


def _ladder_completed(value) -> dict:
    """value(row, summary) for each completed run, by method in LADDER_METHODS."""
    rows, summary = _ladder_rows(), _ladder_summary()
    return {
        method: [
            {"part": part, **value(rows[(part, method)], summary[(part, method)])}
            for part in LADDER_PARTS
            if _outcome(rows[(part, method)]) == "pass"
        ]
        for method in LADDER_METHODS
    }


def ladder_annotation_time() -> dict:
    """Each completed run's time to annotate the query from the stored reference: registering it and annotating for
    CyteArc, loading it and annotating for the comparators; with the time to build the reference, against which the
    split is checked."""
    return {
        "sizes": ladder_sizes(),
        "methods": _ladder_completed(_ladder_steps),
        "source": [relative(LADDER), relative(LADDER_SUMMARY)],
    }


def ladder_rare_type_missed() -> dict:
    """Mean recall of the four types below 1% of the reference pool, for every method at each reference size; checked
    as the mean of the types' recalls."""
    ladder = load_json(LADDER)
    rare = ladder["rareTypes"]["belowOnePercent"]

    def value(row, s):
        tier = s["rareTypeTiers"]["belowOnePercent"]
        if tier["types"] != rare:
            raise ValueError(
                f"{row['referencePart']} {row['method']}: the summary's rare types differ"
            )
        close(
            tier["meanRecall"],
            float(np.mean([s["perType"][t]["recall"] for t in rare])),
            f"{row['referencePart']} {row['method']} mean rare recall",
        )
        return {
            "value": float(tier["meanRecall"]),
            "recall": {t: float(s["perType"][t]["recall"]) for t in rare},
        }

    return {
        "sizes": ladder_sizes(),
        "types": rare,
        "frequencies": {t: ladder["rareTypes"]["referencePoolFrequencies"][t] for t in rare},
        "methods": _ladder_completed(value),
        "source": [relative(LADDER_SUMMARY), relative(LADDER)],
    }


# ---- what each method saves to reuse a reference ---------------------------------------------------------------


def saved_reference_size() -> dict:
    """The bytes each method saves to reuse a reference: CyteArc's descriptor, written while registering the finished
    analysis; CellTypist's model files; Symphony's uncompressed RDS; Seurat's uncompressed RDS with its saved neighbor
    index; on the held-out tissues and at each ladder size that completed."""
    entries = _evaluation()
    saved = saved_reference_bytes()
    held_out = {}
    for tissue in TISSUES:
        seurat = _raw(tissue, "seurat_map")
        celltypist = _raw(tissue, "celltypist")
        held_out[tissue] = {
            "seurat_map": saved[tissue]["seurat_map"],
            "symphony": saved[tissue]["symphony"],
            "celltypist": int(sum(celltypist["referenceModelBytes"].values())),
            **{
                arm: int(entries[(tissue, arm)]["registrationIo"]["writeTransferredBytes"])
                for arm in ARMS
            },
        }
        held_out[tissue]["seuratRdsBytes"] = int(seurat["referenceRdsBytes"])
    rows = _ladder_rows()
    ladder = {
        method: [
            {"part": part, "bytes": int(rows[(part, method)]["persistedReferenceBytes"])}
            for part in LADDER_PARTS
            if _outcome(rows[(part, method)]) == "pass"
        ]
        for method in LADDER_METHODS
    }
    return {
        "tissues": list(TISSUES),
        "heldOut": held_out,
        "parts": _held_out_parts(),
        "sizes": ladder_sizes(),
        "ladder": ladder,
        "definition": load_json(LADDER)["persistedReferenceDefinition"],
        "source": [relative(LADDER), relative(EVALUATION)]
        + [
            relative(CROSS_STUDY / t / f"{m}.json")
            for t in TISSUES
            for m in ("seurat_map", "symphony", "celltypist")
        ],
    }


PANELS = (
    registration_reads,
    saved_reference_size,
    ladder_annotation_time,
    ladder_rare_type_missed,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        data = panel()
        path = OUT / f"{panel.__name__}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {relative(path)}")


if __name__ == "__main__":
    main()
