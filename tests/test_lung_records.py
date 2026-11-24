import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# collect_revision resolves its test-run folder from CYTEARC_RUNS when it is imported; the
# tests never write there, so the folder need not exist. The variable is set only for the import,
# so the other test modules see the environment as it was.
with mock.patch.dict(
    os.environ,
    {
        "CYTEARC_RUNS": os.environ.get(
            "CYTEARC_RUNS", str(Path(tempfile.gettempdir()) / "cytearc-runs-unused")
        )
    },
):
    from experiments.lung import collect_revision
from experiments.runtime.common import ROOT  # noqa: E402

RECORDS = ROOT / "experiments" / "lung" / "records"


def stats_job(**changes) -> dict:
    """A stats job record in the shape reference_revision writes, with every run-history field
    the collector must drop."""
    job = {
        "step": "stats",
        "status": "ok",
        "test": False,
        "startedAt": 100.0,
        "provenance": {
            "script": "reference_revision.py",
            "packages": {"cytearc": "1.3.0"},
            "datasets": {"hlca_core": {"cells": 10}},
            "container": {"containerCpu": 4.0, "workers": 4, "region": "eu"},
            "parameters": {"hvgCount": 2000},
            "design": {"recipe": "map both versions"},
        },
        "previousAttempt": {"status": "error"},
        "design": {"recipe": "map both versions"},
        "centres": {
            "natri": {
                "costs": {
                    "recovery": {"seconds": 21.5, "referenceReadBytes": 0},
                    "notFirstAttempt": {"phase1": False, "update": False},
                    "note": "kept",
                }
            }
        },
        "incrementalVersusComplete": {
            "incremental": {"seconds": 612.3},
            "clean": {"updateCreatedV2": True, "updateNotFirstAttempt": False, "rerunAttempt": 1},
        },
        "jobs": {"update_natri": {"status": "ok", "seconds": 730.2, "previousAttempt": None}},
        "finishedAt": 130.0,
        "steps": {"loadTables": {"startedAt": 101.0, "seconds": 6.3}},
        "currentStep": "loadTables",
        "storesOpened": [],
    }
    job.update(changes)
    return job


class SummaryRecordTest(unittest.TestCase):
    def test_drops_run_history_and_keeps_results(self):
        job = stats_job()
        before = copy.deepcopy(job)
        record = collect_revision.summary_record(job)
        self.assertEqual(job, before, "the job record must not change")
        self.assertEqual(
            list(record),
            [
                "step",
                "status",
                "test",
                "experimentalConditions",
                "design",
                "centres",
                "incrementalVersusComplete",
                "jobs",
                "steps",
                "currentStep",
                "storesOpened",
            ],
        )
        self.assertEqual(
            record["experimentalConditions"],
            {
                "datasets": {"hlca_core": {"cells": 10}},
                "container": {"containerCpu": 4.0, "workers": 4},
                "parameters": {"hvgCount": 2000},
                "design": {"recipe": "map both versions"},
            },
        )
        self.assertEqual(
            record["centres"]["natri"]["costs"],
            {"recovery": {"seconds": 21.5, "referenceReadBytes": 0}, "note": "kept"},
        )
        self.assertEqual(record["incrementalVersusComplete"]["clean"], {"updateCreatedV2": True})
        self.assertEqual(record["steps"], {"loadTables": {"seconds": 6.3}})
        # prepare_lung.revision reads the jobs' previousAttempt, so it stays.
        self.assertIn("previousAttempt", record["jobs"]["update_natri"])

    def test_refuses_other_jobs(self):
        for changes in ({"status": "error"}, {"step": "update"}):
            with self.subTest(**changes), self.assertRaisesRegex(ValueError, "stats job"):
                collect_revision.summary_record(stats_job(**changes))

    def test_stored_summary_is_what_the_collector_writes(self):
        path = RECORDS / "lung_reference_update_summary.json"
        stored = json.loads(path.read_text())
        job = {}
        for key, value in stored.items():
            job["provenance" if key == "experimentalConditions" else key] = copy.deepcopy(value)
        job["startedAt"] = 1.0
        job["provenance"]["container"]["region"] = "eu"
        for centre in job["centres"].values():
            centre["costs"]["notFirstAttempt"] = {"phase1": False, "update": False}
        job["incrementalVersusComplete"]["clean"].update(
            updateNotFirstAttempt=False, rerunAttempt=1
        )
        text = json.dumps(collect_revision.summary_record(job), indent=2) + "\n"
        self.assertEqual(text, path.read_text())


if __name__ == "__main__":
    unittest.main()
