import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from experiments.benchmarks.scaling_workflow_times import TAHOE_FULL_RUNS
from manuscript.figures.analysis_scaling import prepare

GIB = 1024**3
ORDER = ["Leiden", "markers", "filter–PCA", "neighbors", "UMAP"]


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TemporaryRoot(unittest.TestCase):
    """A temporary repository root for prepare's module paths; subclasses list the paths to point into it."""

    PATHS: dict[str, str] = {}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, relative in {"ROOT": "", **self.PATHS}.items():
            patch = mock.patch.object(prepare, name, self.root / relative)
            patch.start()
            self.addCleanup(patch.stop)


class TahoeStagesTest(TemporaryRoot):
    PATHS = {"TAHOE_RESULTS": "tahoe_rna_scaling_results.json"}
    STEPS = {
        "filterCells": (4.0, 7.8),
        "markHvgs": (43.0, 14.1),
        "runNormalization": (30.0, 16.4),
        "runPca": (30.0, 5.1),
        "buildEmbeddingInitialization": (30.0, 3.0),
        "buildAnnIndex": (59.0, 34.3),
        "queryNeighbors": (64.0, 41.5),
        "buildConnectivityMap": (5.0, 30.9),
        "runUmap": (152.0, 77.9),
        "runLeiden": (14.6, 53.0),
        "findMarkers": (54.0, 55.7),
    }

    def write_results(self, missing: str | None = None) -> None:
        # Replicate 2 repeats replicate 1 with UMAP 100 minutes longer and peaking 0.5 GiB higher; replicate 3 has
        # UMAP 200 minutes longer and markers peaking 30 GiB higher.
        replicates = []
        for number, umap_extra, markers_extra in ((1, 0.0, 0.0), (2, 100.0, 0.0), (3, 200.0, 30.0)):
            stages = {
                step: {"seconds": minutes * 60, "peakCgroupGiB": peak}
                for step, (minutes, peak) in self.STEPS.items()
                if not (number == 2 and step == missing)
            }
            if "runUmap" in stages:
                stages["runUmap"]["seconds"] += umap_extra * 60
                stages["runUmap"]["peakCgroupGiB"] += 0.5 if number == 2 else 0.0
            stages["findMarkers"]["peakCgroupGiB"] += markers_extra
            replicates.append({"replicate": number, "stages": stages})
        write(
            prepare.TAHOE_RESULTS,
            {"dataset": {"cells": 95624334}, "fullReplicates": replicates},
        )

    def test_rows_follow_the_given_order_and_summarise_each_stage_over_the_replicates(self):
        self.write_results()
        rows = prepare.tahoe_stages(ORDER)["rows"]
        self.assertEqual([row["stage"] for row in rows], ORDER)
        by_stage = {row["stage"]: row for row in rows}
        # Sums and maxima of the steps above, worked out by hand; stages the replicates share have no spread.
        self.assertEqual(by_stage["filter–PCA"]["minutes"]["values"], [107.0] * 3)
        self.assertEqual(by_stage["filter–PCA"]["peakGiB"]["values"], [16.4] * 3)
        self.assertEqual(by_stage["neighbors"]["minutes"]["values"], [128.0] * 3)
        self.assertEqual(by_stage["neighbors"]["peakGiB"]["high"], 41.5)
        # UMAP is its initialization plus the layout: 182, 282 and 382 minutes, mean 282.
        self.assertEqual(by_stage["UMAP"]["minutes"]["values"], [182.0, 282.0, 382.0])
        self.assertAlmostEqual(by_stage["UMAP"]["minutes"]["mean"], 282.0)
        self.assertEqual(by_stage["UMAP"]["peakGiB"]["values"], [77.9, 78.4, 77.9])
        self.assertAlmostEqual(by_stage["Leiden"]["minutes"]["mean"], 14.6)
        self.assertEqual(by_stage["markers"]["peakGiB"]["values"], [55.7, 55.7, 85.7])
        self.assertEqual(by_stage["markers"]["peakGiB"]["n"], 3)

    def test_a_replicate_missing_a_step_is_an_error(self):
        self.write_results(missing="queryNeighbors")
        with self.assertRaisesRegex(ValueError, "neighbors: replicate 2 has no record of"):
            prepare.tahoe_stages(ORDER)


class TahoeReplicatesTest(TemporaryRoot):
    PATHS = {"TAHOE_RESULTS": "tahoe_rna_scaling_results.json"}

    @staticmethod
    def replicate(number: int, hours: float, umap_peak: float, markers_peak: float) -> dict:
        # Every analysis stage peaks at 10 GiB except UMAP and markers; preparation peaks higher than any of them.
        stages = {
            stage: {"seconds": 60.0, "peakCgroupGiB": 10.0}
            for stage in prepare.TAHOE_ANALYSIS_STAGES
        }
        stages["runUmap"]["peakCgroupGiB"] = umap_peak
        stages["findMarkers"]["peakCgroupGiB"] = markers_peak
        stages["createStore"] = {"seconds": 60.0, "peakCgroupGiB": 130.0}
        return {
            "replicate": number,
            "analysisHours": hours,
            "preparationHours": 4.0,
            "stages": stages,
        }

    def test_minutes_and_peaks_summarise_the_replicates_and_leave_out_preparation(self):
        write(
            prepare.TAHOE_RESULTS,
            {
                "fullReplicates": [
                    self.replicate(1, 8.0, 78.0, 56.0),
                    self.replicate(2, 9.5, 78.5, 55.0),
                    self.replicate(3, 10.5, 70.0, 80.0),
                ]
            },
        )
        summary = prepare.tahoe_replicates()
        # 480, 570 and 630 minutes; peaks are UMAP's in replicates 1 and 2 and markers' in replicate 3.
        self.assertEqual(summary["minutes"]["values"], [480.0, 570.0, 630.0])
        self.assertAlmostEqual(summary["minutes"]["mean"], 560.0)
        self.assertEqual((summary["minutes"]["low"], summary["minutes"]["high"]), (480.0, 630.0))
        self.assertEqual(summary["peakGiB"]["values"], [78.0, 78.5, 80.0])
        self.assertEqual(summary["peakGiB"]["n"], 3)

    def test_a_replicate_missing_an_analysis_stage_is_an_error(self):
        row = self.replicate(1, 8.0, 78.0, 56.0)
        del row["stages"]["runLeiden"]
        write(prepare.TAHOE_RESULTS, {"fullReplicates": [row]})
        with self.assertRaises(KeyError):
            prepare.tahoe_replicates()


class TahoeOutcomesTest(TemporaryRoot):
    PATHS = {"TAHOE_COMPARATORS": "toolkit_comparison_tahoe_runs.json"}

    @staticmethod
    def run(
        tool: str = "bpcells_cached_spca", status: str = "killed", failed_step: str = "runLeiden"
    ) -> dict:
        # 344,064 MiB is 336 GiB.
        return {
            "tool": tool,
            "status": status,
            "failedStep": failed_step,
            "container": {"cpu": 16, "memoryMb": 344064},
        }

    def write_runs(self, *runs: dict) -> None:
        write(
            prepare.TAHOE_COMPARATORS,
            {"records": {f"run_{i}": run for i, run in enumerate(runs)}},
        )

    def test_a_run_killed_for_memory_is_out_of_memory_on_its_configuration(self):
        self.write_runs(self.run())
        self.assertEqual(
            prepare.tahoe_outcomes(),
            {
                "bpcells_cached": {
                    "tool": "bpcells_cached_spca",
                    "state": "oom",
                    "failedStep": "runLeiden",
                    "containerGiB": 336.0,
                    "cpu": 16,
                }
            },
        )

    def test_every_memory_failure_status_is_out_of_memory(self):
        for status in ("killed", "outOfMemory", "killed (record incomplete)"):
            with self.subTest(status=status):
                self.write_runs(self.run(status=status))
                self.assertEqual(prepare.tahoe_outcomes()["bpcells_cached"]["state"], "oom")

    def test_a_run_at_its_time_limit_is_out_of_time(self):
        self.write_runs(self.run(status="timedOut", failed_step="runUmap"))
        outcome = prepare.tahoe_outcomes()["bpcells_cached"]
        self.assertEqual((outcome["state"], outcome["failedStep"]), ("timeout", "runUmap"))

    def test_a_run_that_did_not_fail_is_an_error(self):
        # A completed, unfinished or preempted run has no failure for the status grid to draw.
        for status in ("ok", "attempting", "interrupted"):
            with self.subTest(status=status):
                self.write_runs(self.run(status=status, failed_step=None))
                with self.assertRaisesRegex(ValueError, f"ended with status '{status}'"):
                    prepare.tahoe_outcomes()

    def test_two_runs_of_one_configuration_are_an_error(self):
        self.write_runs(self.run(), self.run())
        with self.assertRaisesRegex(ValueError, "more than one run of bpcells_cached"):
            prepare.tahoe_outcomes()

    def test_a_tool_without_a_drawn_configuration_is_an_error(self):
        self.write_runs(self.run(tool="scanpy"))
        with self.assertRaises(KeyError):
            prepare.tahoe_outcomes()


class TahoeOutcomesRecordTest(unittest.TestCase):
    def test_the_recorded_bpcells_run_ran_out_of_memory_in_leiden_on_16_cpus_and_336_gib(self):
        self.assertEqual(
            prepare.tahoe_outcomes()["bpcells_cached"],
            {
                "tool": "bpcells_cached_spca",
                "state": "oom",
                "failedStep": "runLeiden",
                "containerGiB": 336.0,
                "cpu": 16,
            },
        )


class CoreTimeScalingTest(TemporaryRoot):
    PATHS = {
        "SCALING_TIMES": "rna_scaling_workflow_times.csv",
        "TAHOE_STAGE_MEASUREMENTS": "raw/tahoe_rna_stage_measurements.json",
    }
    TAHOE_CELLS = 100000000
    # 22,500 seconds over the twelve analysis stages on 16 CPUs: 6,000 core-minutes, 60 per million cells.
    TAHOE_STAGE_SECONDS = 22500.0 / 12

    def setUp(self):
        super().setUp()
        self.write_tahoe()

    def write_runs(self, runs: list[tuple[int, int, float]]) -> None:
        with prepare.SCALING_TIMES.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tag", "nRows", "cpus", "analysisSeconds"])
            for index, (cells, cpus, seconds) in enumerate(runs):
                writer.writerow([f"run_{index}", cells, f"{cpus:.1f}", seconds])

    def tahoe_runs(self, scale: float = 1.0) -> dict:
        """The three full runs and a test run of another size, each with conversion stages long enough to change
        any result that counted them."""

        def run(cells: int) -> dict:
            stages = {
                stage: {"status": "ok", "seconds": 1e6, "modalCpuLimit": 16.0}
                for stage in ("createStore", "writeCountsT", "initializeStore")
            }
            for stage in prepare.TAHOE_ANALYSIS_STAGES:
                stages[stage] = {
                    "status": "ok",
                    "seconds": self.TAHOE_STAGE_SECONDS * scale,
                    "modalCpuLimit": 16.0,
                }
            return {"nRows": cells, "stages": stages}

        return {
            "test": run(6000000),
            **{name: run(self.TAHOE_CELLS) for name in TAHOE_FULL_RUNS},
        }

    def write_tahoe(self, runs: dict | None = None) -> None:
        write(prepare.TAHOE_STAGE_MEASUREMENTS, {"runs": runs or self.tahoe_runs()})

    @staticmethod
    def linear_runs() -> list[tuple[int, int, float]]:
        # From one million cells up, 60 core-minutes per million cells in every run: an exact slope of 1. Below,
        # 10,000 cells take 12 core-minutes, the fixed startup work that the slope leaves out.
        runs = [(10000, 4, 180.0)] * 3
        for cells, cpus in ((1000000, 8), (5000000, 16), (10000000, 16)):
            runs += [(cells, cpus, cells / 1e6 * 3600 / cpus)] * 3
        return runs

    def test_core_minutes_are_seconds_times_cpus_and_the_runs_below_100k_cells_are_left_out(self):
        self.write_runs(self.linear_runs())
        result = prepare.core_time_scaling()
        sizes = {size["cells"]: size for size in result["sizes"]}
        self.assertEqual(list(sizes), [1000000, 5000000, 10000000, self.TAHOE_CELLS])
        self.assertAlmostEqual(sizes[1000000]["coreMinutes"]["mean"], 60.0)
        self.assertAlmostEqual(sizes[10000000]["coreMinutes"]["mean"], 600.0)
        self.assertEqual(sizes[5000000]["cpus"], 16.0)
        self.assertEqual(sizes[10000000]["dataset"], "embryo")
        # 60 core-minutes per million cells at every size shown: an exponent of exactly 1.
        self.assertAlmostEqual(result["slope"]["value"], 1.0, places=9)
        self.assertEqual(result["slope"]["fromCells"], 100000)

    def test_the_tahoe_point_adds_each_analysis_stage_times_its_cpus_and_leaves_out_conversion(
        self,
    ):
        self.write_runs(self.linear_runs())
        tahoe = prepare.core_time_scaling()["sizes"][-1]
        self.assertEqual(
            (tahoe["cells"], tahoe["dataset"], tahoe["cpus"]), (self.TAHOE_CELLS, "tahoe", 16.0)
        )
        # 22,500 seconds on 16 CPUs; the conversion stages' million seconds each would give over 800,000.
        self.assertAlmostEqual(tahoe["coreMinutes"]["mean"], 6000.0)
        self.assertEqual(tahoe["coreMinutes"]["n"], 3)

    def test_the_tahoe_replicates_give_their_own_range(self):
        runs = self.tahoe_runs()
        for stage in runs["fullReplicate3"]["stages"].values():
            stage["seconds"] *= 1.5
        self.write_tahoe(runs)
        self.write_runs(self.linear_runs())
        summary = prepare.core_time_scaling()["sizes"][-1]["coreMinutes"]
        self.assertAlmostEqual(summary["low"], 6000.0)
        self.assertAlmostEqual(summary["high"], 9000.0)
        self.assertAlmostEqual(summary["mean"], 7000.0)

    def test_a_shown_size_with_fixed_startup_time_lowers_the_exponent(self):
        # 100,000 cells at 12 core-minutes, above the 6 a linear series gives, flatten the fit below 1 (0.91).
        self.write_runs([*self.linear_runs(), *[(100000, 4, 180.0)] * 3])
        self.assertLess(prepare.core_time_scaling()["slope"]["value"], 0.95)

    def test_the_tahoe_runs_enter_the_fit(self):
        # The embryo runs alone are exactly linear; Tahoe at twice their rate makes the fit steeper than 1 (1.16),
        # so the slope reads the Tahoe runs, not only the embryo series.
        self.write_tahoe(self.tahoe_runs(scale=2.0))
        self.write_runs(self.linear_runs())
        self.assertGreater(prepare.core_time_scaling()["slope"]["value"], 1.1)

    def test_a_tahoe_replicate_missing_an_analysis_stage_is_an_error(self):
        runs = self.tahoe_runs()
        del runs["fullReplicate2"]["stages"]["runLeiden"]
        self.write_tahoe(runs)
        self.write_runs(self.linear_runs())
        with self.assertRaisesRegex(ValueError, r"fullReplicate2 has no record of \['runLeiden'\]"):
            prepare.core_time_scaling()

    def test_a_tahoe_stage_that_did_not_complete_is_an_error(self):
        runs = self.tahoe_runs()
        runs["full"]["stages"]["findMarkers"]["status"] = "failed"
        self.write_tahoe(runs)
        self.write_runs(self.linear_runs())
        with self.assertRaisesRegex(ValueError, "full has an analysis stage that did not complete"):
            prepare.core_time_scaling()

    def test_tahoe_replicates_on_other_cpus_or_cells_are_an_error(self):
        for change in ("cpus", "cells"):
            with self.subTest(change):
                runs = self.tahoe_runs()
                if change == "cpus":
                    runs["fullReplicate3"]["stages"]["runUmap"]["modalCpuLimit"] = 32.0
                else:
                    runs["fullReplicate3"]["nRows"] = 6000000
                self.write_tahoe(runs)
                self.write_runs(self.linear_runs())
                with self.assertRaisesRegex(ValueError, "one cell count and one CPU count"):
                    prepare.core_time_scaling()

    def test_a_size_without_three_runs_is_an_error(self):
        self.write_runs(self.linear_runs()[:-1])
        with self.assertRaisesRegex(ValueError, "10,000,000 cells need three runs"):
            prepare.core_time_scaling()

    def test_a_size_on_mixed_containers_is_an_error(self):
        runs = self.linear_runs()
        runs[3] = (1000000, 16, runs[3][2])
        self.write_runs(runs)
        with self.assertRaisesRegex(
            ValueError, "1,000,000 cells need three runs on one container size"
        ):
            prepare.core_time_scaling()


if __name__ == "__main__":
    unittest.main()


class WeightedMedianTest(unittest.TestCase):
    def test_the_heavy_value_wins_whatever_the_input_order(self):
        # Half of the total weight, 3.5, is first reached at the value carrying weight 5.
        self.assertEqual(prepare.weighted_median(np.array([1, 2, 3]), np.array([1, 1, 5])), 3)
        self.assertEqual(prepare.weighted_median(np.array([3, 1, 2]), np.array([1, 5, 1])), 1)

    def test_equal_weights_give_the_middle_value(self):
        self.assertEqual(prepare.weighted_median(np.array([9, 4, 7]), np.ones(3)), 7)


def write_cell_lines(path: Path, lines: list[str], codes: list[int] | None = None) -> None:
    """The cell-line record: each line's raster code, Cellosaurus ID, name and organ."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "cellLine", "name", "organ"])
        writer.writerows(
            [code, line, f"Line-{line[-1]}", "Lung"]
            for code, line in zip(range(len(lines)) if codes is None else codes, lines, strict=True)
        )


class CellLineLabelsTest(TemporaryRoot):
    PATHS = {
        "TAHOE_RASTER": "umap_raster.npz",
        "TAHOE_CELL_LINES": "cell_lines.csv",
        "TAHOE_CLUSTER_LINES": "cluster_cell_lines.csv",
    }
    CODES = ["CVCL_A", "CVCL_B", "CVCL_C"]

    def write_inputs(self, *, line_codes=None, raster_lines=(0, 1, 2)) -> None:
        dominant = np.full((6, 6), -1, dtype=np.int16)
        cells = np.zeros((6, 6), dtype=np.int64)
        # Line A: two light pixels at the top left and one heavy pixel at row 1, column 4. The unweighted median
        # would sit at row 0, column 1; the cell-weighted median is the heavy pixel.
        for (row, column), count in {(0, 0): 1, (0, 1): 1, (1, 4): 10}.items():
            if 0 in raster_lines:
                dominant[row, column], cells[row, column] = 0, count
        # Line B: three equal pixels along row 5, so its median is the middle one.
        for column in (1, 2, 3):
            if 1 in raster_lines:
                dominant[5, column], cells[5, column] = 1, 4
        if 2 in raster_lines:
            dominant[3, 3], cells[3, 3] = 2, 2
        np.savez(prepare.TAHOE_RASTER, cells=cells, commonestCellLine=dominant)
        write_cell_lines(prepare.TAHOE_CELL_LINES, self.CODES, line_codes)
        # Line A has the most cells over its two clusters, although line B has the largest single cluster.
        with prepare.TAHOE_CLUSTER_LINES.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cluster", "cellLine", "cells"])
            writer.writerows(
                [[1, "CVCL_A", 100], [2, "CVCL_A", 50], [2, "CVCL_B", 120], [3, "CVCL_C", 10]]
            )

    def test_the_largest_lines_are_named_in_order_at_their_cell_weighted_median_pixel(self):
        self.write_inputs()
        labels = prepare.cell_line_labels(2)
        self.assertEqual([label["name"] for label in labels], ["Line-A", "Line-B"])
        self.assertEqual([label["cells"] for label in labels], [150, 120])
        self.assertEqual((labels[0]["row"], labels[0]["column"]), (1, 4))
        self.assertEqual((labels[1]["row"], labels[1]["column"]), (5, 2))
        self.assertEqual(labels[0]["organ"], "Lung")

    def test_a_cell_line_table_out_of_raster_code_order_is_an_error(self):
        # Swapped codes would name each pixel's commonest line after another line.
        self.write_inputs(line_codes=[0, 2, 1])
        with self.assertRaisesRegex(ValueError, "does not list the codes 0 to 2 in order"):
            prepare.cell_line_labels(2)

    def test_a_named_line_that_is_the_commonest_line_of_no_pixel_is_an_error(self):
        self.write_inputs(raster_lines=(1, 2))
        with self.assertRaisesRegex(ValueError, "CVCL_A is the commonest cell line of no pixel"):
            prepare.cell_line_labels(2)


class ResourceTraceTest(TemporaryRoot):
    PATHS = {"RESOURCE_TRACE": "trace"}
    COLUMNS = ["seconds", "memoryBytes", "cpuSeconds", "networkReceivedBytes", "stage"]
    ROWS = [
        [0.0, 1 * GIB, 0.0, 0, "createStore"],
        [1.0, 2 * GIB, 2.0, 1_000_000, "createStore"],
        [2.0, 2 * GIB, 6.0, 3_000_000, "reopenStore"],
        [3.0, 3 * GIB, 7.0, 3_000_000, "filterCells"],
        [5.0, 3 * GIB, 9.0, 7_000_000, "buildEmbeddingInitialization"],
        [6.0, 4 * GIB, 9.5, 7_000_000, "buildAnnIndex"],
        [7.0, 4 * GIB, 10.5, 7_000_000, "runUmap"],
    ]

    def write_trace(self, rows=None, budgets=None) -> None:
        write(
            prepare.RESOURCE_TRACE / "resourceTrace.json",
            {"columns": self.COLUMNS, "intervalSeconds": 1.0, "rows": rows or self.ROWS},
        )
        budgets = budgets or {"createStore": 48 * GIB, "filterCells": 12 * GIB, "runUmap": 12 * GIB}
        stages = {step: {"memoryBudgetBytes": b, "cpuLimit": 4.0} for step, b in budgets.items()}
        write(
            prepare.RESOURCE_TRACE / "resources.json",
            {"runTag": "trace_test", "cells": 10000, "stages": stages},
        )

    def test_the_trace_starts_at_filtering_with_rates_from_differences_between_samples(self):
        # The import ran under a larger budget; only the budget of the steps drawn counts.
        self.write_trace()
        trace = prepare.resource_trace()
        self.assertEqual((trace["memoryBudgetGiB"], trace["cpuLimit"]), (12.0, 4.0))
        self.assertEqual(trace["samples"]["minutes"], [0, round(2 / 60, 4), 0.05, round(4 / 60, 4)])
        self.assertEqual(trace["samples"]["memoryGiB"], [3, 3, 4, 4])
        # From filtering on, CPU seconds 7, 9, 9.5, 10.5 and bytes 3, 7, 7, 7 million over intervals of 2, 1, 1 s.
        self.assertEqual(trace["rates"]["cpuCores"], [1, 0.5, 1])
        self.assertEqual(trace["rates"]["networkMBps"], [2, 0, 0])
        self.assertEqual(trace["rates"]["minutes"], [round(s / 60, 4) for s in (2, 3, 4)])
        # UMAP's initialisation runs before the neighbour search and is drawn with it.
        spans = [
            (s["stage"], round(s["start"] * 60, 1), round(s["end"] * 60, 1))
            for s in trace["stages"]
        ]
        self.assertEqual(
            spans,
            [
                ("filter–PCA", 0.0, 2.0),
                ("neighbors", 2.0, 4.0),
                ("UMAP", 4.0, 4.0),
            ],
        )

    def test_drawn_steps_with_different_budgets_are_an_error(self):
        self.write_trace(budgets={"filterCells": 12 * GIB, "runUmap": 48 * GIB})
        with self.assertRaisesRegex(ValueError, "different memory budgets"):
            prepare.resource_trace()

    def test_an_import_step_after_the_analysis_began_is_an_error(self):
        rows = self.ROWS + [[8.0, 4 * GIB, 11.0, 7_000_000, "writeCountsT"]]
        self.write_trace(rows)
        with self.assertRaisesRegex(
            ValueError, r"runs \['writeCountsT'\] after the analysis began"
        ):
            prepare.resource_trace()

    def test_a_trace_of_the_import_alone_is_an_error(self):
        self.write_trace(self.ROWS[:3])
        with self.assertRaisesRegex(ValueError, "no analysis step"):
            prepare.resource_trace()

    def test_a_step_no_trace_stage_holds_is_an_error(self):
        rows = [row[:4] + ["exportResults" if row[4] == "runUmap" else row[4]] for row in self.ROWS]
        self.write_trace(rows)
        with self.assertRaisesRegex(ValueError, "exportResults"):
            prepare.resource_trace()

    def test_a_sample_without_a_value_is_an_error(self):
        rows = [list(row) for row in self.ROWS]
        rows[4][1] = None
        self.write_trace(rows)
        with self.assertRaisesRegex(ValueError, "1 trace samples lack a value, the first at 5.0 s"):
            prepare.resource_trace()


class ClusterCompositionsTest(TemporaryRoot):
    PATHS = {
        "TAHOE_CELL_LINES": "cell_lines.csv",
        "TAHOE_CLUSTER_LINES": "cluster_cell_lines.csv",
    }

    def write_inputs(self, rows) -> None:
        write_cell_lines(prepare.TAHOE_CELL_LINES, ["CVCL_A", "CVCL_B", "CVCL_C", "CVCL_D"])
        with prepare.TAHOE_CLUSTER_LINES.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cluster", "cellLine", "cells"])
            writer.writerows(rows)

    def cluster(self, number=1, cells=100, line="CVCL_C", fraction=0.6) -> dict:
        return {
            "cluster": number,
            "cells": cells,
            "commonestCellLine": line,
            "commonestCellLineFraction": fraction,
        }

    def test_lines_with_five_percent_or_more_are_listed_largest_first_by_raster_index(self):
        # Cluster 1: C 60, A 36, D 4 cells; D falls under 5% and joins the rest.
        self.write_inputs([[1, "CVCL_A", 36], [1, "CVCL_C", 60], [1, "CVCL_D", 4]])
        (cluster,) = prepare.cluster_compositions([self.cluster()])
        self.assertEqual(
            cluster["composition"], [{"cellLine": 2, "share": 0.6}, {"cellLine": 0, "share": 0.36}]
        )
        self.assertEqual(cluster["cells"], 100)

    def test_a_commonest_line_that_differs_between_the_results_is_an_error(self):
        self.write_inputs([[1, "CVCL_A", 60], [1, "CVCL_C", 40]])
        with self.assertRaisesRegex(ValueError, "commonest cell line differs"):
            prepare.cluster_compositions([self.cluster()])

    def test_cells_that_do_not_add_up_are_an_error(self):
        self.write_inputs([[1, "CVCL_C", 60], [1, "CVCL_A", 30]])
        with self.assertRaisesRegex(ValueError, "does not hold the cells of cluster 1"):
            prepare.cluster_compositions([self.cluster()])

    def test_a_cluster_missing_from_the_table_is_an_error(self):
        self.write_inputs([[1, "CVCL_C", 60], [1, "CVCL_A", 40]])
        with self.assertRaisesRegex(ValueError, "does not hold the cells of cluster 2"):
            prepare.cluster_compositions([self.cluster(), self.cluster(number=2)])
