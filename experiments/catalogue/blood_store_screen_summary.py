"""Totals of the blood store screen over its jobs, and each job's share of its stores' count bytes; and the totals of
the screen's interferon rerun over its jobs.

uv run --locked python -m experiments.catalogue.blood_store_screen_summary
"""

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

RECORD = ROOT / "experiments" / "catalogue" / "records" / "blood_store_screen.json"
INTERFERON_RUNS = tuple(
    ROOT / "experiments" / "catalogue" / "records" / "interferon_screen" / f"{store_set}.json"
    for store_set in ("immune_cohorts", "other_blood_stores")
)
SUMMARY = DERIVED / "catalogue" / "blood_store_screen.json"
# Fields of the interferon rerun's job totals that add up over its jobs, because no store is screened twice.
INTERFERON_SUMMED = ("storesScreened", "primaryBloodCellsRead", "donorsWithScore", "responseBytes")
# Job fields that add up over the jobs, because no store is screened twice.
SUMMED = (
    "storesScreened",
    "storesFailed",
    "cellsInScreenedStores",
    "primaryBloodCellsRead",
    "studyScopedDonors",
    "donorsWithScore",
    "httpRequests",
    "responseBytes",
    "storeCountBytes",
    "wallSeconds",
    "rateLimitWaits",
)


def summarize(jobs: list[dict]) -> dict:
    stores = [store for job in jobs for store in job["storeIds"]]
    if len(set(stores)) != len(stores):
        raise ValueError("A store was screened by more than one job")
    screen = {key: sum(job[key] for job in jobs) for key in SUMMED}
    return {
        "source": str(RECORD.relative_to(ROOT)),
        "jobs": [
            {
                "storeSet": job["storeSet"],
                "responseShareOfStoreCountBytes": job["responseBytes"] / job["storeCountBytes"],
            }
            for job in jobs
        ],
        "screen": {
            "jobs": len(jobs),
            **screen,
            # The screen's time is the jobs' wall seconds added together.
            "wallSecondsPerStore": screen["wallSeconds"] / screen["storesScreened"],
            "responseShareOfStoreCountBytes": screen["responseBytes"] / screen["storeCountBytes"],
        },
    }


def interferon_rerun(runs: list[dict]) -> dict:
    """Totals of the interferon rerun over its jobs; its time is the jobs' wall seconds added together."""
    if any(run["status"] != "completed" for run in runs):
        raise ValueError("An interferon rerun job did not complete")
    stores = [store["storeId"] for run in runs for store in run["result"]["stores"]]
    if len(set(stores)) != len(stores):
        raise ValueError("A store was screened by more than one interferon rerun job")
    return {
        "jobs": len(runs),
        **{key: sum(run["result"]["totals"][key] for run in runs) for key in INTERFERON_SUMMED},
        "wallSeconds": sum(run["result"]["wallSeconds"] for run in runs),
    }


def main() -> None:
    write_json_at(
        SUMMARY,
        {
            **summarize(load_json(RECORD)["jobs"]),
            "interferonRerun": interferon_rerun([load_json(path) for path in INTERFERON_RUNS]),
        },
    )


if __name__ == "__main__":
    main()
