import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.benchmarks import tool_summary

OLD = "rna_workflow_brain_1306127_scanpy_dask2_csc_cpus_8_replicate_1"
NEW = "rna_workflow_brain_1306127_scanpy_dask_covariance_csc_cpus_8_replicate_1"


class ComparisonNameTest(unittest.TestCase):
    def test_the_original_runner_name_becomes_the_comparison_name_everywhere_it_identifies_the_run(
        self,
    ):
        ladder = {
            "tool": "scanpy_dask2_csc",
            "sourceRecordId": f"{OLD}_memory_tiers",
            "payload": {"tool": "scanpy_dask2_csc", "replicate": 1},
            "attempts": [
                {"memoryGib": 128, "recordId": f"{OLD}_memory_128gib"},
                {"memoryGib": 256},
            ],
        }
        name, record = tool_summary.comparison_name(f"{OLD}_memory_tiers", ladder)
        self.assertEqual(name, f"{NEW}_memory_tiers")
        self.assertEqual(record["tool"], "scanpy_dask_covariance_csc")
        self.assertEqual(record["sourceRecordId"], f"{NEW}_memory_tiers")
        self.assertEqual(record["payload"], {"tool": "scanpy_dask_covariance_csc", "replicate": 1})
        self.assertEqual(
            record["attempts"],
            [{"memoryGib": 128, "recordId": f"{NEW}_memory_128gib"}, {"memoryGib": 256}],
        )
        # The collected record itself is left as it was read.
        self.assertEqual(ladder["tool"], "scanpy_dask2_csc")

    def test_a_tool_the_comparison_already_names_is_unchanged(self):
        record = {"tool": "bpcells_cached", "sourceRecordId": "x"}
        self.assertEqual(tool_summary.comparison_name("x", record), ("x", record))


class ToolkitRecordsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name)
        for name, value in (("DATA", self.data), ("ADDED_TOOLKIT_RUNS", ("added.json",))):
            patch = mock.patch.object(tool_summary, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def write_added(self, records: dict) -> None:
        (self.data / "added.json").write_text(json.dumps({"provenance": {}, "records": records}))

    def test_added_runs_join_the_comparison_under_its_names(self):
        self.write_added({f"{OLD}_memory_128gib": {"tool": "scanpy_dask2_csc", "status": "ok"}})
        records = tool_summary.toolkit_records({"records": {"kept": {"tool": "scanpy"}}})
        self.assertEqual(sorted(records), ["kept", f"{NEW}_memory_128gib"])
        self.assertEqual(records[f"{NEW}_memory_128gib"]["tool"], "scanpy_dask_covariance_csc")

    def test_an_added_run_that_repeats_a_measurement_is_an_error(self):
        self.write_added({f"{OLD}_memory_128gib": {"tool": "scanpy_dask2_csc"}})
        with self.assertRaisesRegex(
            ValueError, f"added.json repeats the measurement {NEW}_memory_128gib"
        ):
            tool_summary.toolkit_records(
                {"records": {f"{NEW}_memory_128gib": {"tool": "scanpy_dask_covariance_csc"}}}
            )
