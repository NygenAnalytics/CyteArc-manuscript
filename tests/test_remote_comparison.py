import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.benchmarks import remote_comparison


class WelchLogRatioTest(unittest.TestCase):
    def test_a_ratio_of_geometric_means_with_its_welch_interval_and_p(self):
        # The logs of a are those of b shifted by log 4 with the same spread: the ratio is 4, and the Welch statistic
        # is the square root of 6 on 4 degrees of freedom. The reference P value and interval come from integrating
        # the t density numerically, not from scipy.
        result = remote_comparison.welch_log_ratio([4.0, 8.0, 16.0], [1.0, 2.0, 4.0])
        self.assertAlmostEqual(result["ratio"], 4.0, places=12)
        self.assertAlmostEqual(result["df"], 4.0, places=12)
        self.assertAlmostEqual(result["p"], 0.070484, places=5)
        self.assertAlmostEqual(result["low"], 0.83107, places=4)
        self.assertAlmostEqual(result["high"], 19.2523, places=3)

    def test_a_side_of_one_run_gives_a_ratio_without_an_interval(self):
        result = remote_comparison.welch_log_ratio([4.0], [1.0, 2.0, 4.0])
        # The geometric mean of 1, 2 and 4 is 2.
        self.assertAlmostEqual(result["ratio"], 2.0, places=12)
        self.assertIsNone(result["p"])
        self.assertIsNone(result["low"])


class SpeedupTest(unittest.TestCase):
    def test_the_speedup_inverts_the_ratio_and_swaps_the_ends_of_its_interval(self):
        result = remote_comparison.with_speedup(
            {"ratio": 0.25, "low": 0.125, "high": 0.5, "p": 0.01}
        )
        self.assertEqual(
            (result["speedup"], result["speedupLow"], result["speedupHigh"]), (4.0, 2.0, 8.0)
        )
        # The comparison itself is kept.
        self.assertEqual(
            (result["ratio"], result["low"], result["high"], result["p"]), (0.25, 0.125, 0.5, 0.01)
        )

    def test_a_comparison_without_an_interval_has_a_speedup_without_one(self):
        result = remote_comparison.with_speedup(
            {"ratio": 0.5, "low": None, "high": None, "p": None}
        )
        self.assertEqual(
            (result["speedup"], result["speedupLow"], result["speedupHigh"]), (2.0, None, None)
        )

    def test_the_mean_minutes_leave_out_runs_that_did_not_complete(self):
        runs = [{"analysisMinutes": 10.0}, {"analysisMinutes": None}, {"analysisMinutes": 20.0}]
        self.assertEqual(remote_comparison.mean_minutes(runs), 15.0)
        self.assertIsNone(remote_comparison.mean_minutes([{"analysisMinutes": None}]))


def sweep_run(backend, *, elapsed, budget=24, workers=8, complete=True, dataset="embryo1M") -> dict:
    return {
        "dataset": dataset,
        "storeBackend": backend,
        "budgetGiB": budget,
        "workers": workers,
        "complete": complete,
        "elapsedSeconds": elapsed,
        "conversionElapsedSeconds": 100.0,
    }


class CytearcAnalysisSecondsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.derived = Path(temporary.name)
        patch = mock.patch.object(remote_comparison, "BENCHMARKS_DERIVED", self.derived)
        patch.start()
        self.addCleanup(patch.stop)

    def test_each_backend_keeps_its_complete_embryo_runs_at_24_gib_and_8_workers(self):
        runs = [
            sweep_run("objectStore", elapsed=700.0),
            sweep_run("objectStore", elapsed=800.0),
            sweep_run("modalLocalDisk", elapsed=500.0),
            sweep_run("objectStore", elapsed=900.0, budget=8),
            sweep_run("objectStore", elapsed=950.0, workers=4),
            sweep_run("objectStore", elapsed=990.0, complete=False),
            sweep_run("objectStore", elapsed=999.0, dataset="mouseBrain"),
        ]
        (self.derived / "rna_resource_comparisons.json").write_text(json.dumps({"runs": runs}))
        # Conversion seconds are not analysis.
        self.assertEqual(remote_comparison.cytearc_analysis_seconds("objectStore"), [600.0, 700.0])
        self.assertEqual(remote_comparison.cytearc_analysis_seconds("modalLocalDisk"), [400.0])


class SummariseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, value in (
            ("ROOT", self.root),
            ("BENCHMARKS_DERIVED", self.root),
            ("RUNS", self.root / "runs.json"),
            ("BPCELLS_RUNS", self.root / "bpcells.json"),
            ("SUMMARY", self.root / "summary.json"),
            # Each record's analysis seconds stand in its steps.
            ("analysis_time_seconds", lambda steps: steps["analysis"]),
        ):
            patch = mock.patch.object(remote_comparison, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_cytearc_is_compared_with_scanpy_on_each_backend(self):
        records = {}
        for tool, seconds in (
            ("scanpy_dask_covariance_csc_remote_zarr", (1200.0, 1500.0, 1800.0)),
            ("scanpy_dask_covariance_csc_local_zarr", (1600.0, 2000.0, 2400.0)),
        ):
            for replicate, value in enumerate(seconds, start=1):
                records[f"{tool}_replicate_{replicate}"] = {
                    "dataset": "embryo_1000000",
                    "tool": tool,
                    "replicate": replicate,
                    "status": "ok",
                    "steps": {"analysis": value},
                    "container": {"cpu": 8, "memoryMb": 32768},
                    "peakCgroupBytes": 0,
                }
        (self.root / "runs.json").write_text(json.dumps({"records": records}))
        # BPCells' local-disk runs come from another file; its memory-tier summary and other datasets stay out.
        bpcells = {
            f"bpcells_cached_replicate_{replicate}": {
                "dataset": "embryo_1000000",
                "tool": "bpcells_cached",
                "replicate": replicate,
                "status": "ok",
                "steps": {"analysis": value},
                "container": {"cpu": 8, "memoryMb": 32768},
                "peakCgroupBytes": 0,
            }
            for replicate, value in enumerate((800.0, 1000.0, 1200.0), start=1)
        }
        bpcells["bpcells_cached_replicate_1_memory_tiers"] = {
            **bpcells["bpcells_cached_replicate_1"],
            "steps": {"analysis": 1.0},
        }
        bpcells["bpcells_cached_other"] = {
            **bpcells["bpcells_cached_replicate_1"],
            "dataset": "embryo_10000000",
            "steps": {"analysis": 1.0},
        }
        (self.root / "bpcells.json").write_text(json.dumps({"records": bpcells}))
        sweep = [sweep_run("objectStore", elapsed=s) for s in (700.0, 850.0, 1000.0)] + [
            sweep_run("modalLocalDisk", elapsed=s) for s in (500.0, 600.0, 700.0)
        ]
        (self.root / "rna_resource_comparisons.json").write_text(json.dumps({"runs": sweep}))
        with contextlib.redirect_stdout(io.StringIO()):
            remote_comparison.summarise()
        summary = json.loads((self.root / "summary.json").read_text())
        comparisons = summary["comparisons"]
        self.assertEqual(
            sorted(comparisons),
            [
                "cytearcOverBpcellsLocalDisk",
                "cytearcOverScanpyLocalDisk",
                "cytearcOverScanpyObjectStorage",
                "scanpyObjectStorageOverLocalDisk",
            ],
        )
        # Scanpy's runs are 2 times CyteArc's on object storage and 4 times on local disk, run for run.
        self.assertAlmostEqual(comparisons["cytearcOverScanpyObjectStorage"]["ratio"], 0.5)
        self.assertAlmostEqual(comparisons["cytearcOverScanpyLocalDisk"]["ratio"], 0.25)
        # BPCells' runs are 2 times CyteArc's on local disk, run for run, and only its three embryo runs count.
        self.assertAlmostEqual(comparisons["cytearcOverBpcellsLocalDisk"]["ratio"], 0.5)
        self.assertEqual(comparisons["cytearcOverBpcellsLocalDisk"]["nB"], 3)
        self.assertEqual(summary["arms"]["bpcells_cached"]["backend"], "localDisk")
        self.assertEqual(
            summary["cytearcLocalDiskAnalysisMinutes"], [400.0 / 60, 500.0 / 60, 600.0 / 60]
        )
        # CyteArc ran 2, 4 and 2 times faster, run for run, so each speedup is the inverse of its ratio.
        self.assertAlmostEqual(comparisons["cytearcOverScanpyObjectStorage"]["speedup"], 2.0)
        self.assertAlmostEqual(comparisons["cytearcOverScanpyLocalDisk"]["speedup"], 4.0)
        self.assertAlmostEqual(comparisons["cytearcOverBpcellsLocalDisk"]["speedup"], 2.0)
        # Mean analysis minutes: Scanpy 20, 25 and 30 on object storage and 26.7, 33.3 and 40 on local disk; BPCells
        # 13.3, 16.7 and 20; CyteArc 600, 750 and 900 seconds on object storage and 400, 500 and 600 on local disk.
        self.assertAlmostEqual(
            summary["arms"]["scanpy_dask_covariance_csc_remote_zarr"]["meanAnalysisMinutes"], 25.0
        )
        self.assertAlmostEqual(
            summary["arms"]["scanpy_dask_covariance_csc_local_zarr"]["meanAnalysisMinutes"], 100 / 3
        )
        self.assertAlmostEqual(summary["arms"]["bpcells_cached"]["meanAnalysisMinutes"], 50 / 3)
        self.assertAlmostEqual(summary["cytearcObjectStorageMeanAnalysisMinutes"], 12.5)
        self.assertAlmostEqual(summary["cytearcLocalDiskMeanAnalysisMinutes"], 500 / 60)


if __name__ == "__main__":
    unittest.main()
