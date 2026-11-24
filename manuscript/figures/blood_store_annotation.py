"""The in-place annotation of the blood stores as the figures draw it: what each job read from the stored reference and
how long it took to open it, each store's mapping time and peak, and each job's disagreement of transferred with
author labels.

Each annotation job's record holds each store's mapping time and its typed and agreeing cells, which
mapping.collect_records counted from the store's per-cell labels; the records are checked against the experiment's
derived summary.
"""

from experiments.runtime.common import DERIVED, ROOT, load_json

RECORDS = sorted((ROOT / "experiments/mapping/records/blood_store_annotation").glob("*.json"))
SUMMARY = DERIVED / "mapping" / "blood_store_annotation.json"
LADDER = ROOT / "experiments/mapping/records/reference_ladder.json"
GIB = 1024**3

# Each annotation job's record is named for the study whose stores it annotated; the jobs' short names.
JOB_NAMES = {
    "gong_all_cells": "Gong",
    "perez": "Perez",
    "ren": "Ren",
    "sound_life": "Sound Life, CMV−",
    "sound_life_cmv_positive": "Sound Life, CMV+",
    "van_der_wijst": "van der Wijst",
    "wells": "Wells",
    "yoshida": "Yoshida",
}


def relative(path) -> str:
    return str(path.relative_to(ROOT))


def sources() -> list[str]:
    return [relative(p) for p in RECORDS] + [relative(SUMMARY)]


def _reference_cells() -> int:
    """The cells of the ladder's largest unintegrated reference, which every job opened."""
    rows = [
        r
        for r in load_json(LADDER)["rows"]
        if r["referencePart"] == "reference_all" and r["method"] == "unintegrated"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"the ladder holds {len(rows)} runs of its largest unintegrated reference, not one"
        )
    return int(rows[0]["referenceCells"])


def jobs() -> list[tuple[str, dict]]:
    """Each annotation job's record stem and record, checked: the job opened the ladder's largest reference and read no
    reference count, and all jobs together annotated, typed and agreed on the cells the summary counts."""
    reference_cells = _reference_cells()
    found = []
    for path in RECORDS:
        record = load_json(path)
        opening, reads = record["referenceOpening"], record["referenceReadsIncludingOpening"]
        if int(opening["countReadBytes"]) or int(reads["countReadBytes"]):
            raise ValueError(f"{path.name}: an annotation job read reference counts")
        if opening["referenceCells"] != reference_cells:
            raise ValueError(
                f"{path.name}: the job opened a reference of {opening['referenceCells']} cells"
            )
        found.append((path.stem, record))
    summary = load_json(SUMMARY)
    for field in ("eligibleCells", "typedCells", "agreeingCells"):
        total = sum(s[field] for _, r in found for s in r["stores"])
        if total != summary[field]:
            raise ValueError(
                f"the job records hold {total} {field}, not the {summary[field]} the summary counts"
            )
    return found


def most_cells_first(items: list[dict]) -> list[dict]:
    """The jobs from the most cells down."""
    return sorted(items, key=lambda j: -j["cells"])


def reference_reads() -> dict:
    """What each job read from the stored reference: opening it, and mapping its stores after opening, with how long
    opening took; checked against the summary's reads after opening."""
    after = {s["study"]: s["referenceReadBytesAfterOpening"] for s in load_json(SUMMARY)["studies"]}
    found = []
    for stem, record in jobs():
        opening = int(record["referenceOpening"]["transferredBytes"])
        total = int(record["referenceReadsIncludingOpening"]["transferredBytes"])
        if total - opening != after[stem]:
            raise ValueError(f"{stem}: the reads after opening differ from the summary")
        found.append(
            {
                "name": JOB_NAMES[stem],
                "stores": len(record["stores"]),
                "cells": sum(int(s["eligibleCells"]) for s in record["stores"]),
                "openingBytes": opening,
                "afterOpeningBytes": total - opening,
                "openingSeconds": float(record["referenceOpening"]["seconds"]),
                "countReadBytes": int(record["referenceReadsIncludingOpening"]["countReadBytes"]),
            }
        )
    return {"jobs": most_cells_first(found), "source": sources()}


def mapping_time() -> dict:
    """Each store's mapping time with label transfer onto the stored reference, with its eligible cells; the time covers
    reading the store's counts, projecting and voting."""
    stores, found = [], []
    for stem, record in jobs():
        found.append(
            {
                "name": JOB_NAMES[stem],
                "cells": sum(int(s["eligibleCells"]) for s in record["stores"]),
            }
        )
        for store in record["stores"]:
            stores.append(
                {
                    "job": JOB_NAMES[stem],
                    "eligibleCells": int(store["eligibleCells"]),
                    "storeCells": int(store["storeCells"]),
                    "seconds": float(store["mappingSeconds"]),
                }
            )
    return {"jobs": most_cells_first(found), "stores": stores, "source": sources()}


def memory() -> dict:
    """Each store's container peak while it was mapped, and each job's peak over the whole job, which adds a pseudobulk
    pass over each store."""
    found = []
    for stem, record in jobs():
        found.append(
            {
                "name": JOB_NAMES[stem],
                "cells": sum(int(s["eligibleCells"]) for s in record["stores"]),
                "mappingPeaksGiB": [s["mappingPeakContainerBytes"] / GIB for s in record["stores"]],
                "jobPeakGiB": record["peakContainerBytes"] / GIB,
                "containerGiB": record["containerMemoryBytes"] / GIB,
            }
        )
    for job in found:
        if max(job["mappingPeaksGiB"]) > job["jobPeakGiB"]:
            raise ValueError(f"{job['name']}: a store's mapping peak exceeds the job's peak")
    return {"jobs": most_cells_first(found), "source": sources()}


def disagreement() -> dict:
    """Each job's disagreement of transferred with author labels, the share of cells whose transferred label differs,
    over the cells whose author label is one of the reference's types, and over all jobs; with the reference genes each
    job's stores lacked, which were filled with the reference mean."""
    found = []
    for stem, record in jobs():
        cells = sum(int(store["eligibleCells"]) for store in record["stores"])
        typed_cells = sum(int(store["typedCells"]) for store in record["stores"])
        agreeing = sum(int(store["agreeingCells"]) for store in record["stores"])
        found.append(
            {
                "name": JOB_NAMES[stem],
                "stores": len(record["stores"]),
                "cells": cells,
                "typedCells": typed_cells,
                "agreeingCells": agreeing,
                "disagreement": 1.0 - agreeing / typed_cells,
                "missingGenes": sorted(
                    {len(s["referenceFeaturesMissing"]) for s in record["stores"]}
                ),
            }
        )
    return {
        "jobs": most_cells_first(found),
        "total": {
            "stores": sum(j["stores"] for j in found),
            "cells": sum(j["cells"] for j in found),
            "typedCells": sum(j["typedCells"] for j in found),
            "disagreement": 1.0
            - sum(j["agreeingCells"] for j in found) / sum(j["typedCells"] for j in found),
        },
        "source": sources(),
    }
