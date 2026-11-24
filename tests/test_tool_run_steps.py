import importlib.util
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from experiments.benchmarks import timing, tool_summary
from experiments.runtime import harness

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_R = ROOT / "experiments" / "benchmarks" / "tools" / "workflow_r.R"
HOUR = 3600.0

# The steps of a BPCells run on Tahoe-100M as its record lists them: start and end second.
TAHOE_STEPS = {
    "convertParquet": (0, 1000),
    "uncompressInput": (1000, 1010),
    "convertMatrix": (1010, 7000),
    "transposeMatrix": (7000, 15000),
    "filterCells": (15000, 17000),
    "runNormalization": (17000, 17001),
    "markHvgs": (17001, 17300),
    "runPca": (17300, 40000),
    "queryNeighbors": (40000, 46000),
    "runUmap": (46000, 55000),
    "runLeiden": (55000, 80000),
    "findMarkers": (80000, 86000),
}


def steps(until: str | None = None) -> dict:
    """The step entries of TAHOE_STEPS, or of the steps before UNTIL with UNTIL started only."""
    entries = {}
    for name, (start, end) in TAHOE_STEPS.items():
        if name == until:
            entries[name] = {"start": start}
            break
        entries[name] = {"start": start, "end": end, "seconds": end - start}
    return entries


def tahoe_record(status: str, step_entries: dict, started_at: float, **fields) -> dict:
    return {
        "tool": "bpcells_cached_spca",
        "dataset": "tahoe_95624334",
        "replicate": 1,
        "container": {"cpu": 16, "memoryMb": 336 * 1024, "preemptible": False},
        "status": status,
        "steps": step_entries,
        "stepOrder": list(step_entries),
        "startedAt": started_at,
        **fields,
    }


class PreparationStepsTest(unittest.TestCase):
    def test_parquet_conversion_is_preparation_not_analysis(self):
        entries = steps()
        self.assertEqual(timing.workflow_time_seconds(entries), 86000)
        # Analysis starts with the cell filter; everything before it prepares the input.
        self.assertEqual(timing.analysis_time_seconds(entries), 86000 - 15000)
        self.assertEqual(timing.preparation_seconds(entries), 15000)

    def test_the_summary_counts_parquet_conversion_as_loading(self):
        record = tahoe_record("ok", steps(), 0.0, workflowSeconds=86100.0)
        (row,) = tool_summary.workflow_summary({"run": record}, collected_at=2 * 86400)
        self.assertAlmostEqual(row["loadingMinutes"], 15000 / 60)
        self.assertAlmostEqual(row["preparationMinutes"], 15000 / 60)
        self.assertAlmostEqual(row["analysisTimeMinutes"], (86000 - 15000) / 60)
        self.assertAlmostEqual(row["analysisMinutes"], (86000 - 15000 - 6000) / 60)

    def test_the_r_workflow_converts_with_a_module_that_exists(self):
        source = WORKFLOW_R.read_text()
        self.assertIn('step("convertParquet"', source)
        (module,) = re.findall(r'"-m", "(experiments\.benchmarks\.tools\.tahoe_to_h5ad)"', source)
        self.assertIsNotNone(importlib.util.find_spec(module))
        self.assertIn("convertParquet", timing.PREPARATION_STEPS)
        self.assertIn("convertParquet", tool_summary.LOADING_STEPS)


class ThreadedPcaTest(unittest.TestCase):
    def test_bpcells_svds_runs_on_the_workers(self):
        # BPCells 0.3.1's svds generic lists no threads argument, only its method for BPCells matrices does, so a
        # check of the generic's arguments left PCA on one thread.
        source = WORKFLOW_R.read_text()
        calls = re.findall(r"BPCells::svds\(([^)]*)\)", source)
        self.assertEqual(len(calls), 1)
        self.assertIn("threads = workers", calls[0])
        self.assertNotIn("formals(BPCells::svds)", source)


class TimeLimitTest(unittest.TestCase):
    def test_a_record_left_open_past_the_call_limit_is_a_timeout_in_its_running_step(self):
        collected = 100 * HOUR
        record = tahoe_record("attempting", steps("runLeiden"), collected - 25 * HOUR)
        record["currentStep"] = "runLeiden"
        (row,) = tool_summary.workflow_summary({"run": record}, collected_at=collected)
        self.assertEqual(
            row["outcome"], "did not finish within Modal's 24-hour limit per call (in runLeiden)"
        )
        self.assertEqual(row["tiers"][0]["status"], "timedOut")
        self.assertEqual(row["tiers"][0]["failedStep"], "runLeiden")
        self.assertNotIn("analysisTimeMinutes", row)

    def test_a_record_open_within_the_call_limit_is_still_running(self):
        collected = 100 * HOUR
        record = tahoe_record("attempting", steps("runLeiden"), collected - 20 * HOUR)
        record["currentStep"] = "runLeiden"
        (row,) = tool_summary.workflow_summary({"run": record}, collected_at=collected)
        self.assertEqual(row["outcome"], "running")


class FakeClient:
    """The S3 calls R2.download_dir makes, over objects held in memory, in pages of two."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.downloaded: list[str] = []

    def get_paginator(self, name: str):
        client = self

        class Paginator:
            def paginate(self, Bucket: str, Prefix: str):
                items = [
                    {"Key": key, "Size": len(value)}
                    for key, value in sorted(client.objects.items())
                    if key.startswith(Prefix)
                ]
                return [{"Contents": items[start : start + 2]} for start in range(0, len(items), 2)]

        return Paginator()

    def download_file(self, bucket: str, key: str, target: str) -> None:
        self.downloaded.append(key)
        Path(target).write_bytes(self.objects[key])


OBJECTS = {
    "revision/metadata/gene_metadata.parquet": b"g" * 3,
    "revision/data/train-00000-of-03388.parquet": b"d" * 10,
    "revision/data/train-00001-of-03388.parquet": b"d" * 20,
    "revision/index/train-00000-of-03388.json": b"r" * 4,
    "revision/index/train-00000-of-03388.cells.parquet": b"c" * 50,
    "revision/manifest.json": b"m" * 5,
    "other/data/train-00000-of-03388.parquet": b"x" * 7,
}


def storage() -> tuple[harness.R2, FakeClient]:
    r2 = harness.R2.__new__(harness.R2)
    r2.client = FakeClient(OBJECTS)
    return r2, r2.client


class DirectoryDownloadTest(unittest.TestCase):
    def test_only_the_named_files_are_downloaded(self):
        r2, client = storage()
        files = [
            "metadata/gene_metadata.parquet",
            "data/train-00000-of-03388.parquet",
            "index/train-00000-of-03388.json",
        ]
        with tempfile.TemporaryDirectory() as root:
            size = r2.download_dir("s3://bucket/revision", Path(root), files)
            kept = sorted(path.relative_to(root).as_posix() for path in Path(root).rglob("*.*"))
        self.assertEqual(size, 3 + 10 + 4)
        self.assertEqual(kept, sorted(files))
        self.assertEqual(sorted(client.downloaded), sorted(f"revision/{name}" for name in files))

    def test_a_named_file_that_is_missing_stops_before_any_download(self):
        r2, client = storage()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, "train-00002-of-03388.parquet"):
                r2.download_dir(
                    "s3://bucket/revision",
                    Path(root),
                    ["data/train-00000-of-03388.parquet", "data/train-00002-of-03388.parquet"],
                )
        self.assertEqual(client.downloaded, [])

    def test_an_empty_list_of_files_is_an_error(self):
        r2, _ = storage()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "no file"):
                r2.download_dir("s3://bucket/revision", Path(root), [])

    def test_without_files_the_whole_prefix_is_downloaded(self):
        r2, client = storage()
        with tempfile.TemporaryDirectory() as root:
            size = r2.download_dir("s3://bucket/revision", Path(root))
        self.assertEqual(size, 3 + 10 + 20 + 4 + 50 + 5)
        self.assertEqual(len(client.downloaded), 6)


# The child writes the events of a finished step and of a step it is still in when the harness
# publishes, as a run does when Modal's call limit ends it. Each step outlasts the harness's
# one-second reading of the events, so it sees each start and end apart.
CHILD = """
import json, sys, time
def emit(step, event):
    with open(sys.argv[2], "a") as handle:
        handle.write(json.dumps({"step": step, "event": event, "t": time.time()}) + "\\n")
emit("convertParquet", "start")
time.sleep(2.2)
emit("convertParquet", "end")
emit("runLeiden", "start")
time.sleep(2.5)
"""


class RunRecordTest(unittest.TestCase):
    def payload(self, work: str, **fields) -> dict:
        return {
            "tool": "bpcells_cached_spca",
            "dataset": "tahoe_first_3_files",
            "datasetUri": "s3://bucket/experiments/datasets/tahoe-100m/revision",
            "outputUri": "s3://bucket/record.json",
            "r2EndpointUrl": "https://example.invalid",
            "workDir": work,
            "containerCpu": 1,
            "containerMemoryMb": 1024,
            "workers": 1,
            "memorySampleSeconds": 0.05,
            **fields,
        }

    def test_the_named_files_reach_the_directory_download(self):
        files = ["metadata/gene_metadata.parquet", "data/train-00000-of-03388.parquet"]
        with tempfile.TemporaryDirectory() as work, mock.patch.object(harness, "R2") as r2_class:
            r2 = r2_class.return_value
            r2.exists.return_value = False
            r2.download_dir.return_value = 13
            payload = self.payload(work, inputMode="directory", datasetFiles=files)
            outcome = harness.Run(payload).execute([sys.executable, "-I", "-c", "pass"])
        self.assertEqual(outcome["status"], "ok")
        uri, local, named = r2.download_dir.call_args.args
        self.assertEqual(uri, payload["datasetUri"])
        self.assertEqual(local, Path(work) / "data" / "revision")
        self.assertEqual(named, files)

    def test_a_record_published_mid_step_keeps_finished_steps_and_the_running_one(self):
        with (
            tempfile.TemporaryDirectory() as work,
            mock.patch.object(harness, "R2") as r2_class,
            mock.patch.dict(os.environ, {"MODAL_REGION": "us-east"}),
        ):
            r2 = r2_class.return_value
            r2.exists.return_value = False
            started = time.time()
            harness.Run(self.payload(work, inputMode="remote")).execute(
                [sys.executable, "-I", "-c", CHILD]
            )
        published = [call.args[1] for call in r2.put_json.call_args_list]
        mid = [record for record in published if record["currentStep"] == "runLeiden"]
        self.assertTrue(mid, "no record was published while runLeiden ran")
        record = mid[0]
        self.assertEqual(record["status"], "attempting")
        self.assertEqual(record["completedSteps"], ["convertParquet"])
        self.assertGreater(record["steps"]["convertParquet"]["seconds"], 2.0)
        self.assertGreater(record["steps"]["convertParquet"]["peakRssBytes"], 0)
        self.assertGreaterEqual(record["steps"]["runLeiden"]["start"], started)
        self.assertEqual(record["container"]["region"], "us-east")


if __name__ == "__main__":
    unittest.main()
