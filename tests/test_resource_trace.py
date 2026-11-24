import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from experiments.benchmarks import collect_resource_trace as collect
from experiments.benchmarks import resource_trace_summary
from experiments.benchmarks.config import load_profiling_config
from experiments.runtime import resource_trace

# Loading a configuration only builds bucket URIs, so placeholder locations stand in for the bucket.
BUCKET = {"R2_ENDPOINT_URL": "https://storage.invalid", "R2_BUCKET": "experiments-bucket"}
CONFIG = (
    collect.HERE
    / "configs"
    / "benchmark"
    / "embryo_rna_scaling_10000000_cells_resource_trace_replicate_1.toml"
)
STORED = collect.RECORDS / "resource_trace_10000000_cells"


class CollectResourceTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        patch = mock.patch.dict(os.environ, BUCKET)
        patch.start()
        self.addCleanup(patch.stop)
        self.config = load_profiling_config(CONFIG)

    def test_the_trace_lies_beside_the_runs_summary(self) -> None:
        self.assertEqual(
            collect.trace_uri(self.config),
            "s3://experiments-bucket/experiments/results/"
            "embryo_rna_scaling_10000000_cells_resource_trace_replicate_1/10000000/resourceTrace.json",
        )

    def test_the_traced_run_repeats_the_scaling_replicate_under_its_own_tag(self) -> None:
        replicate = load_profiling_config(
            collect.HERE
            / "configs"
            / "benchmark"
            / "embryo_rna_scaling_10000000_cells_replicate_1.toml"
        )
        self.assertNotEqual(self.config.runTag, replicate.runTag)
        self.assertEqual(
            self.config.model_dump(exclude={"runTag"}), replicate.model_dump(exclude={"runTag"})
        )

    def test_stored_records_are_what_the_collection_step_writes(self) -> None:
        trace = json.loads((STORED / "resourceTrace.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "resource_trace_10000000_cells"
            with mock.patch.object(collect, "HERE", Path(directory)):
                collect.write_trace(trace, collect.stage_resources(self.config), folder)
            for name in ("resourceTrace.json", "resources.json"):
                with self.subTest(name=name):
                    self.assertEqual((folder / name).read_bytes(), (STORED / name).read_bytes())

    def test_a_trace_without_samples_and_a_run_of_several_sizes_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "no samples"):
            collect.write_trace({"rows": []}, {}, Path("unused"))
        several = self.config.model_copy(update={"targetSizes": (1000, 2000)})
        with self.assertRaisesRegex(ValueError, "one size"):
            collect.stage_resources(several)


class ResourceTraceTests(unittest.TestCase):
    def test_samples_carry_the_stage_running_and_the_counters(self) -> None:
        trace = resource_trace.ResourceTrace(intervalSeconds=0.01).start()
        trace.mark("filterCells")
        time.sleep(0.05)
        trace.mark("runPca")
        time.sleep(0.05)
        record = trace.stop()
        self.assertEqual(record["columns"], resource_trace.COLUMNS)
        rows = record["rows"]
        self.assertTrue(all(len(row) == len(resource_trace.COLUMNS) for row in rows))
        stages = [row[4] for row in rows]
        # Stages follow one another: no sample of runPca comes before a sample of filterCells.
        self.assertIn("runPca", stages)
        self.assertEqual(stages[-1], "runPca")
        marked = [stage for stage in stages if stage is not None]
        self.assertEqual(marked, sorted(marked, key=["filterCells", "runPca"].index))
        seconds = [row[0] for row in rows]
        self.assertEqual(seconds, sorted(seconds))

    def test_an_interval_that_is_not_positive_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resource_trace.ResourceTrace(intervalSeconds=0)

    def test_network_bytes_leave_out_loopback(self) -> None:
        text = (
            "Inter-|   Receive\n face |bytes    packets\n"
            "    lo: 1000 10 0 0 0 0 0 0 1000 10 0 0 0 0 0 0\n"
            "  eth0: 250 3 0 0 0 0 0 0 99 1 0 0 0 0 0 0\n"
            "  eth1: 50 1 0 0 0 0 0 0 9 1 0 0 0 0 0 0\n"
        )
        with mock.patch.object(resource_trace, "_read", return_value=text):
            self.assertEqual(resource_trace.network_received_bytes(), 300)


def trace(rows: list[list]) -> dict:
    return {"columns": resource_trace_summary.COLUMNS, "rows": rows}


class StageReceivePeaksTests(unittest.TestCase):
    def test_each_stage_takes_its_fastest_interval_from_the_start_of_filtering(self) -> None:
        rows = [
            # The import's 900 MB in a second stays out, and so does the step from it into filtering.
            [0.0, 1, 0.0, 0, "createStore"],
            [1.0, 1, 0.0, 900_000_000, "reopenStore"],
            [2.0, 1, 0.0, 950_000_000, "filterCells"],
            [3.0, 1, 0.0, 1_100_000_000, "markHvgs"],
            [5.0, 1, 0.0, 1_300_000_000, "runPca"],
            # An interval belongs to the stage of its closing sample: these 300 MB are the neighbor search's.
            [6.0, 1, 0.0, 1_600_000_000, "buildEmbeddingInitialization"],
            [7.0, 1, 0.0, 1_650_000_000, "queryNeighbors"],
        ]
        peaks = resource_trace_summary.stage_receive_peaks(trace(rows))
        self.assertEqual(
            [(p["stage"], p["peakReceiveBytesPerSecond"], p["atSeconds"]) for p in peaks],
            [("filter–PCA", 150_000_000.0, 1.0), ("neighbors", 300_000_000.0, 4.0)],
        )

    def test_each_stage_takes_its_busiest_interval_in_cpu_seconds_per_second(self) -> None:
        rows = [
            # The import's 30 CPU seconds in one second stay out.
            [0.0, 1, 0.0, 0, "createStore"],
            [1.0, 1, 30.0, 0, "reopenStore"],
            [2.0, 1, 31.0, 0, "filterCells"],
            # 12 CPU seconds over a 2-second interval are 6 busy cores, not 12.
            [4.0, 1, 43.0, 0, "markHvgs"],
            [5.0, 1, 47.0, 0, "runPca"],
            [6.0, 1, 62.5, 0, "runUmap"],
            [7.0, 1, 70.0, 0, "runUmap"],
        ]
        peaks = resource_trace_summary.stage_busy_core_peaks(trace(rows))
        self.assertEqual(
            [(p["stage"], p["peakBusyCores"], p["busyCoresAtSeconds"]) for p in peaks],
            [("filter–PCA", 6.0, 2.0), ("UMAP", 15.5, 4.0)],
        )
        # The summary gives each stage both peaks.
        (pca, umap) = resource_trace_summary.stage_summary(trace(rows))
        self.assertEqual((pca["peakBusyCores"], pca["peakReceiveBytesPerSecond"]), (6.0, 0.0))
        self.assertEqual(umap["stage"], "UMAP")
        with self.assertRaisesRegex(ValueError, "lacks its time or its CPU seconds"):
            resource_trace_summary.stage_busy_core_peaks(
                trace([[0.0, 1, 0.0, 0, "filterCells"], [1.0, 1, None, 0, "markHvgs"]])
            )

    def test_the_stored_trace_keeps_up_to_15_9_of_16_cores_busy(self) -> None:
        # Paragraph 286: up to 15.9 busy cores during analysis, in the UMAP stage; no interval exceeds the 16 CPUs.
        stored = json.loads((STORED / "resourceTrace.json").read_text())
        peaks = {
            p["stage"]: p["peakBusyCores"]
            for p in resource_trace_summary.stage_busy_core_peaks(stored)
        }
        self.assertEqual(round(max(peaks.values()), 1), 15.9)
        self.assertEqual(max(peaks, key=peaks.get), "UMAP")
        self.assertLessEqual(max(peaks.values()), stored["cpuCount"])

    def test_a_step_no_stage_holds_or_a_missing_value_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no analysis stage holds: \\['exportOutputs'\\]"):
            resource_trace_summary.stage_receive_peaks(
                trace([[0.0, 1, 0.0, 0, "filterCells"], [1.0, 1, 0.0, 5, "exportOutputs"]])
            )
        with self.assertRaisesRegex(ValueError, "lacks its time or its received bytes"):
            resource_trace_summary.stage_receive_peaks(
                trace([[0.0, 1, 0.0, 0, "filterCells"], [1.0, 1, 0.0, None, "markHvgs"]])
            )
        with self.assertRaisesRegex(ValueError, "no analysis step"):
            resource_trace_summary.stage_receive_peaks(trace([[0.0, 1, 0.0, 0, "createStore"]]))

    def test_the_stored_trace_reads_counts_at_up_to_152_mb_per_second_through_pca(self) -> None:
        # Paragraph 6: count reads of up to 152 MB/s between filtering and PCA in the ten-million-cell run.
        stored = json.loads((STORED / "resourceTrace.json").read_text())
        peaks = {p["stage"]: p for p in resource_trace_summary.stage_receive_peaks(stored)}
        self.assertEqual(list(peaks), ["filter–PCA", "neighbors", "UMAP", "Leiden", "markers"])
        self.assertEqual(round(peaks["filter–PCA"]["peakReceiveBytesPerSecond"] / 1e6), 152)
        # The same peak as the figure's per-second rates, read from the received bytes between two samples.
        rows = stored["rows"]
        rates = [
            (b[3] - a[3]) / (b[0] - a[0])
            for a, b in zip(rows, rows[1:])
            if b[4] in ("filterCells", "markHvgs", "runNormalization", "runPca")
            and a[4] != "reopenStore"
        ]
        self.assertAlmostEqual(peaks["filter–PCA"]["peakReceiveBytesPerSecond"], max(rates))


if __name__ == "__main__":
    unittest.main()
