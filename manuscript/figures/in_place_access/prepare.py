"""Prepare the panel data of the supplementary figure on using stores in place: every store of the CyteBase release by
its cells and nonzero counts, and the in-place screen of the human blood stores.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py. Run from the repository
root:

  uv run --locked python -B -m manuscript.figures.in_place_access.prepare
"""

import json

from experiments.runtime.common import DERIVED, ROOT, load_json

CATALOGUE = ROOT / "experiments/catalogue/records"
WORKFLOW_RUNS = ROOT / "experiments/benchmarks/records/rna_workflow_runs.json"
CATALOGUE_OBJECTS = CATALOGUE / "catalogue_objects.json"
CYTEBASE_CATALOGUE = CATALOGUE / "cytebase_catalogue.json"
BLOOD_SCREEN = CATALOGUE / "blood_store_screen.json"
SCREEN_SUMMARY = DERIVED / "catalogue" / "blood_store_screen.json"
# The per-store rows of the two blood-screen jobs, in the order of the record's jobs; blood_store_screen.json holds
# the jobs' totals, which the rows must add up to.
SCREEN_RUNS = (
    CATALOGUE / "blood_store_screen/other_blood_stores.json",
    CATALOGUE / "blood_store_screen/immune_cohorts.json",
)
OUT = ROOT / "manuscript" / "figures" / "in_place_access"


def relative(path) -> str:
    return str(path.relative_to(ROOT))


# ---- the in-place screen of the human blood stores ---------------------------------------


def blood_screen() -> dict:
    """One row per screened store: its cells, the bytes of its count arrays (both orientations), the bytes the
    screen received for it and its seconds, in two lists: the stores the panel draws, and those read while their job
    waited on Hugging Face's request quota, whose seconds measure the wait rather than the read. With them, the
    measured downloads of source H5AD files and the screen's totals over both jobs."""
    record = load_json(BLOOD_SCREEN)
    if len(record["jobs"]) != len(SCREEN_RUNS):
        raise ValueError("the screen record and the raw results list different jobs")
    rows = []
    for job, path in zip(record["jobs"], SCREEN_RUNS):
        stores = load_json(path)["result"]["stores"]
        if sorted(store["storeId"] for store in stores) != sorted(job["storeIds"]):
            raise ValueError(f"{path.name} holds other stores than the job {job['storeSet']}")
        job_rows = [
            {
                "store": store["storeId"],
                "job": job["storeSet"],
                "cells": store["cells"],
                "countBytes": store["countBytes"],
                "responseBytes": store["io"]["http"]["responseBytes"],
                "seconds": store["seconds"],
                # The job waited on Hugging Face's request quota while this store was read.
                "rateLimited": store["io"]["quotaWaitSeconds"] > 0,
            }
            for store in stores
        ]
        for field, total in (
            ("cells", "cellsInScreenedStores"),
            ("countBytes", "storeCountBytes"),
            ("responseBytes", "responseBytes"),
        ):
            if sum(row[field] for row in job_rows) != job[total]:
                raise ValueError(f"the stores' {field} do not add up to the job's {total}")
        rows += job_rows
    screen = load_json(SCREEN_SUMMARY)["screen"]
    downloads = [
        {
            "dataset": name,
            "cells": dataset["nCells"],
            "bytes": dataset["sourceBytes"],
            "seconds": dataset["downloadSeconds"],
        }
        for name, dataset in sorted(load_json(WORKFLOW_RUNS)["inputs"].items())
        if dataset.get("downloadSeconds") is not None
    ]
    if not downloads:
        raise ValueError("the workflow runs record no download of a source H5AD")
    return {
        "stores": [row for row in rows if not row["rateLimited"]],
        "rateLimitedStores": [row for row in rows if row["rateLimited"]],
        "downloads": downloads,
        "totals": {
            "stores": screen["storesScreened"],
            "cells": screen["cellsInScreenedStores"],
            "donors": screen["studyScopedDonors"],
            "wallSeconds": screen["wallSeconds"],
            "responseBytes": screen["responseBytes"],
            "countBytes": screen["storeCountBytes"],
            "share": screen["responseShareOfStoreCountBytes"],
        },
        "source": [
            relative(BLOOD_SCREEN),
            relative(SCREEN_SUMMARY),
            *map(relative, SCREEN_RUNS),
            relative(WORKFLOW_RUNS),
        ],
    }


# ---- every store of the release by its cells and nonzero counts -----------------------------------------


def release_storage() -> dict:
    """Every store of the release by its nonzero counts and the bytes it takes in cloud storage, all its parts
    included; and the release's totals of stores, observations, nonzero counts and bytes."""
    objects = load_json(CATALOGUE_OBJECTS)
    stores = [
        {
            "cells": store["cells"],
            "nonzeros": store["nonzeros"],
            "bytes": sum(store["bytes"].values()),
        }
        for store in objects["perStore"].values()
    ]
    if sum(store["bytes"] for store in stores) != objects["totalBytes"]:
        raise ValueError("the stores' bytes do not add up to the release's total")
    catalogue = load_json(CYTEBASE_CATALOGUE)["datasets"]
    if catalogue["stores"] != len(stores):
        raise ValueError("the catalogue and the object listing hold different numbers of stores")
    if sum(store["cells"] for store in stores) != catalogue["observations"]:
        raise ValueError("the stores' cells do not add up to the catalogue's observations")
    return {
        "releaseStores": catalogue["stores"],
        "releaseObservations": catalogue["observations"],
        "releaseNonzeros": sum(store["nonzeros"] for store in stores),
        "releaseBytes": objects["totalBytes"],
        "stores": stores,
        "source": [relative(CATALOGUE_OBJECTS), relative(CYTEBASE_CATALOGUE)],
    }


def main() -> None:
    for name, values in (
        ("release_storage", release_storage()),
        ("blood_screen", blood_screen()),
    ):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(values, indent=1, allow_nan=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
