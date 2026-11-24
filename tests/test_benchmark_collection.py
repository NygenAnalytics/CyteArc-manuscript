import csv
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.benchmarks import (
    budget_compliance,
    budget_sweep,
    collect_quality,
    collect_workflow_runs,
    count_request_summary,
    count_requests,
    scaling_workflow_times,
    store_summary,
)
from experiments.benchmarks.config import load_profiling_config

# Loading a configuration only builds bucket URIs, so placeholder locations stand in for the bucket.
BUCKET = {"R2_ENDPOINT_URL": "https://storage.invalid", "R2_BUCKET": "experiments-bucket"}
GIB = 1024**3


def setUpModule() -> None:
    patcher = mock.patch.dict(os.environ, BUCKET)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def step_record(stage: str, peak: int, error: str | None = None) -> dict:
    """A worker's step record with the fields that stage_summary reads."""
    return {
        "stage": stage,
        "status": "ok" if error is None else "error",
        "error": error,
        "seconds": 10.0,
        "wholeFunctionSeconds": 11.0,
        "cgroupCurrentBaselineBytes": GIB,
        "peakCgroupBytes": peak,
        "rssBaselineBytes": GIB,
        "peakRssBytes": peak,
        "memoryMaxBytes": 48 * GIB,
        "details": {"executionReports": {"countsTCellBand": [{"reservedBytes": 3 * GIB}]}},
        "processMemory": {
            "before": {"anonymousRssBytes": None, "allocatorFreeHeldBytes": 0},
            "after": {},
        },
    }


LEIDEN_REFUSAL = (
    "MemoryError: Leiden clustering of 8894085 cells needs about 9001076192 bytes in memory for "
    "a graph of 97834935 edges, but the operation limit is 8589934592 bytes. Raise mem_budget."
)


class BudgetStagesTest(unittest.TestCase):
    def test_a_refusal_gives_its_cells_estimate_and_edges(self):
        row = budget_compliance.refusal({"stage": "runLeiden", "error": LEIDEN_REFUSAL})
        self.assertEqual(row["status"], "refused")
        self.assertEqual(row["cellsAfterFilter"], 8_894_085)
        self.assertEqual(row["estimatedBytes"], 9_001_076_192)
        self.assertEqual(row["graphEdges"], 97_834_935)
        self.assertTrue(row["message"].startswith("Leiden clustering of 8894085 cells"))

    def test_a_failure_that_is_not_a_refusal_stops_the_collection(self):
        with self.assertRaisesRegex(SystemExit, "without refusing its budget"):
            budget_compliance.refusal({"stage": "runPca", "error": "RuntimeError: disk full"})

    def test_rows_from_step_records_mark_the_refused_step(self):
        name = "embryo_rna_budget_compliance_10000000_cells_8_gib_8_workers_replicate_1"
        config = load_profiling_config(budget_compliance.CONFIGS / f"{name}.toml")
        records = {
            ("freshProcess", "createStore"): step_record("createStore", 6 * GIB),
            ("freshProcess", "runLeiden"): step_record("runLeiden", GIB, LEIDEN_REFUSAL),
        }
        rows = budget_compliance.job_rows(
            name, config, ("freshProcess",), None, lambda arm, stage: records.get((arm, stage))
        )
        self.assertEqual([row["stage"] for row in rows], ["createStore", "runLeiden"])
        self.assertEqual([row["status"] for row in rows], ["ok", "refused"])
        self.assertEqual(rows[0]["containerPeakBytes"], 6 * GIB)
        self.assertIsNone(rows[0]["externalContainerPeakBytes"])
        self.assertEqual(rows[0]["plannedReservedBytes"], 3 * GIB)
        self.assertEqual(rows[0]["containerMemoryLimitBytes"], 48 * GIB)
        self.assertEqual((rows[0]["nCells"], rows[0]["replicate"]), (10_000_000, 1))
        self.assertEqual((rows[0]["budgetBytes"], rows[0]["unitBytes"]), (8 * GIB, 500_000_000))
        self.assertEqual(rows[1]["estimatedBytes"], 9_001_076_192)
        self.assertEqual(
            set(rows[0]),
            set(budget_compliance.STAGE_COLUMNS)
            - {"message"}
            - {
                "cellsAfterFilter",
                "estimatedBytes",
                "graphEdges",
            },
        )

    def test_rows_from_a_summary_take_both_samplers(self):
        name = "embryo_rna_budget_compliance_1000000_cells_8_gib_8_workers_replicate_2"
        config = load_profiling_config(budget_compliance.CONFIGS / f"{name}.toml")
        summary = {
            "budgetBytes": 8 * GIB,
            "container": {"memoryLimitBytes": 32 * GIB},
            "arms": {
                "freshProcess": {
                    "failure": None,
                    "stages": [
                        {
                            "stage": "findMarkers",
                            "seconds": 5.0,
                            "containerPeakBytes": 7 * GIB,
                            "externalContainerPeakBytes": 7 * GIB + 1,
                            "plannedReservedBytes": 4 * GIB,
                        }
                    ],
                }
            },
        }
        (row,) = budget_compliance.job_rows(
            name, config, ("freshProcess",), summary, lambda arm, stage: None
        )
        self.assertEqual(row["externalContainerPeakBytes"], 7 * GIB + 1)
        self.assertEqual((row["replicate"], row["containerMemoryLimitBytes"]), (2, 32 * GIB))

    def test_a_summary_with_another_budget_is_refused(self):
        name = "embryo_rna_budget_compliance_1000000_cells_8_gib_8_workers_replicate_1"
        config = load_profiling_config(budget_compliance.CONFIGS / f"{name}.toml")
        summary = {"budgetBytes": 4 * GIB, "container": {}, "arms": {}}
        with self.assertRaisesRegex(SystemExit, "budget differs"):
            budget_compliance.job_rows(name, config, ("freshProcess",), summary, lambda a, s: None)

    def test_every_collected_configuration_exists(self):
        for name in budget_compliance.COLLECTED:
            self.assertTrue((budget_compliance.CONFIGS / f"{name}.toml").is_file(), name)


class CountRequestsRecordTest(unittest.TestCase):
    def test_locations_and_job_identifiers_leave_the_record(self):
        envelope = {
            "module": "benchmarks.count_requests",
            "callId": "fc-0123456789abcdefghijkl",
            "region": "us-east",
            "status": "completed",
            "result": {
                "settings": {
                    "runId": "embryo_5000000_fixed_replicate_1",
                    "datasetUri": "s3://private-bucket/a.h5ad",
                    "r2EndpointUrl": "https://x.r2.cloudflarestorage.com",
                    "workDirectory": "/tmp/count_requests",
                    "layout": "fixed",
                },
                "stages": [{"stage": "markHvgs", "storeUri": "s3://b/s", "provenance": {}}],
                "frozenInputs": {
                    "uri": "s3://b/frozen.npz",
                    "sha256": "abc",
                    "source": {"datasetUri": "s3://b/a.h5ad", "cells": 5},
                },
            },
        }
        record = count_requests.collected_record(envelope)
        self.assertEqual(set(record), {"status", "result"})
        self.assertEqual(
            record["result"]["settings"],
            {"runId": "embryo_5000000_fixed_replicate_1", "layout": "fixed"},
        )
        self.assertEqual(record["result"]["stages"], [{"stage": "markHvgs"}])
        self.assertEqual(
            record["result"]["frozenInputs"], {"sha256": "abc", "source": {"cells": 5}}
        )

    def test_the_collected_runs_are_the_layout_runs_and_their_validation(self):
        self.assertEqual(len(count_requests.COLLECTED), 13)
        self.assertEqual(
            count_requests.COLLECTED["embryo_10000000_scaled_replicate_3"],
            "embryo/10000000/scaled/replicate_3/measurement.json",
        )


class QualityRecordTest(unittest.TestCase):
    def test_placement_dates_and_run_locations_leave_the_comparison(self):
        record = {
            "status": "ok",
            "runTag": "retained_output_agreement",
            "region": "us-east",
            "startedAt": 1.0,
            "finishedAt": 2.0,
            "container": {"cpu": 8, "region": "us"},
            "runs": {
                "cytearc_r1": {
                    "tool": "cytearc",
                    "datasetUri": "s3://b/a.h5ad",
                    "outputs": "<campaign>/x.h5",
                    "image": {"budgetShare": 0.75},
                    "container": {"cpu": 8, "region": "us-central"},
                }
            },
            "agreement": {"a|b": {"clusterAri": 0.5}},
        }
        kept = collect_quality.comparison_record(record)
        self.assertEqual(set(kept), {"status", "container", "runs", "agreement"})
        self.assertEqual(kept["container"], {"cpu": 8})
        self.assertEqual(kept["runs"], {"cytearc_r1": {"tool": "cytearc", "container": {"cpu": 8}}})


class WorkflowRecordTest(unittest.TestCase):
    OUTPUT = {
        "tool": "cytearc_limit8",
        "dataset": "embryo_1000000",
        "status": "ok",
        "callId": "fc-0123456789abcdefghijkl",
        "datasetUri": "s3://b/a.h5ad",
        "versions": {"cytearc": "1.3.0"},
        "startedAt": 1.0,
        "finishedAt": 2.0,
        "log": "...",
        "memorySeries": [[0, 1]],
        "interruptions": [],
        "image": {"memoryLimitGib": 8, "family": "python", "requirementsSha256": "abc"},
        "container": {"cpu": 8, "region": "us-central"},
        "steps": {"runPca": {"start": 3.0, "end": 4.0}},
    }

    def test_run_history_leaves_and_step_times_stay(self):
        record = collect_workflow_runs.run_record(self.OUTPUT, placement=False)
        self.assertEqual(set(record), {"tool", "dataset", "status", "image", "container", "steps"})
        self.assertEqual(record["image"], {"memoryLimitGib": 8})
        self.assertEqual(record["container"], {"cpu": 8})
        self.assertEqual(record["steps"], self.OUTPUT["steps"])

    def test_the_sweep_keeps_its_placement(self):
        record = collect_workflow_runs.run_record(self.OUTPUT, placement=True)
        self.assertEqual(record["container"], {"cpu": 8, "region": "us-central"})

    def test_an_unfinished_run_stops_the_collection(self):
        with self.assertRaisesRegex(SystemExit, "attempting"):
            collect_workflow_runs.run_record(
                {**self.OUTPUT, "status": "attempting"}, placement=True
            )

    def test_the_sweep_has_three_replicates_per_budget(self):
        names = list(collect_workflow_runs.PREPROCESSING_RUNS)
        self.assertEqual(len(names), 12)
        self.assertEqual(sorted({name.split("_")[1] for name in names}), ["16g", "24g", "4g", "8g"])
        self.assertEqual(len(collect_workflow_runs.WORKFLOW_RUNS), 36)


class DerivedSummaryTest(unittest.TestCase):
    def test_store_ratios(self):
        ratios = store_summary.size_ratios(
            {"nRows": 10, "countsBytes": 400, "countsTBytes": 300, "analysisBytes": 70}
        )
        self.assertEqual(
            ratios, {"nRows": 10, "countsTOverCounts": 0.75, "analysisOverCounts": 0.1}
        )

    def test_sweep_minutes_run_from_the_first_step_to_the_end_of_pca(self):
        run = {
            "budgetBytes": 4 * GIB,
            "replicate": 2,
            "countPolicy": {"requested": "250000000,25000000"},
            "steps": {
                "convertMatrix": {
                    "start": 60.0,
                    "end": 120.0,
                    "peakCgroupBytes": GIB,
                    "peakRssBytes": 2 * GIB,
                },
                "runPca": {
                    "start": 120.0,
                    "end": 300.0,
                    "peakCgroupBytes": 3 * GIB,
                    "peakRssBytes": GIB,
                },
            },
        }
        summary = budget_sweep.run_summary(run)
        self.assertEqual(summary["minutes"], 4.0)
        self.assertEqual(
            (summary["containerPeakGiB"], summary["processTreeRssPeakGiB"]), (3.0, 2.0)
        )
        self.assertEqual((summary["unitBytes"], summary["chunkBytes"]), (250_000_000, 25_000_000))
        rows = budget_sweep.budget_rows(
            [summary, {**summary, "minutes": 5.0, "containerPeakGiB": 4.5}]
        )
        self.assertEqual(rows[0]["meanMinutes"], 4.5)
        self.assertFalse(rows[0]["belowBudget"])
        # The container peaks of 3.0 and 4.5 GiB average 3.75, below their largest.
        self.assertEqual(
            (rows[0]["meanContainerPeakGiB"], rows[0]["maxContainerPeakGiB"]), (3.75, 4.5)
        )


def tahoe_results(
    hours, umap_peaks, leiden_seconds, purities, preparation_hours=None, marker_seconds=None
) -> dict:
    """Tahoe-100M results with one replicate per value given, a pilot of 1,000 cells and a full atlas of 100,000. Each
    replicate's store preparation takes 4 hours and its marker testing an hour, unless ``preparation_hours`` and
    ``marker_seconds`` say otherwise."""
    if preparation_hours is None:
        preparation_hours = [4.0] * len(hours)
    if marker_seconds is None:
        marker_seconds = [3600.0] * len(hours)
    return {
        "runs": {"test": {"nRows": 1000}, "full": {"nRows": 100_000}},
        "stages": {
            # 100 times the cells: 100 times the seconds is b = 1, 10 times is b = 0.5.
            "runPca": {"fullOverTestSeconds": 10.0},
            "buildAnnIndex": {"fullOverTestSeconds": 100.0},
        },
        "fullReplicates": [
            {
                "analysisHours": h,
                "preparationHours": p,
                "stages": {
                    "runUmap": {"peakCgroupGiB": u},
                    "runLeiden": {"seconds": s},
                    "findMarkers": {"seconds": m},
                },
            }
            for h, p, u, s, m in zip(
                hours, preparation_hours, umap_peaks, leiden_seconds, marker_seconds, strict=True
            )
        ],
        "fullStore": {"clusters": [{"commonestCellLineFraction": f} for f in purities]},
    }


class TahoeSummaryTest(unittest.TestCase):
    def test_the_replicates_give_their_means_and_the_clusters_one_line_dominates(self):
        summary = scaling_workflow_times.tahoe_replicates(
            tahoe_results(
                [8.0, 9.0, 10.0],
                [77.0, 78.0, 82.0],
                [840.0, 900.0, 960.0],
                [0.95, 0.9, 0.89, 0.5],
                preparation_hours=[3.0, 4.0, 8.0],
                marker_seconds=[2400.0, 3000.0, 5400.0],
            )
        )
        self.assertEqual(summary["replicates"], 3)
        self.assertEqual(summary["meanAnalysisHours"], 9.0)
        # The mean, not the median (4.0), of the preparation hours.
        self.assertEqual(summary["meanPreparationHours"], 5.0)
        self.assertEqual(summary["meanUmapPeakGiB"], 79.0)
        self.assertEqual(summary["meanLeidenMinutes"], 15.0)
        # 40, 50 and 90 minutes: the mean is 60, where the median would be 50 and unconverted seconds 3,600.
        self.assertEqual(summary["meanMarkerMinutes"], 60.0)
        # A cluster at exactly the threshold counts; one just below does not.
        self.assertEqual((summary["clusters"], summary["clustersAtPurityThreshold"]), (4, 2))
        self.assertAlmostEqual(summary["stageExponents"]["runPca"], 0.5)
        self.assertAlmostEqual(summary["largestStageExponent"], 1.0)

    def test_results_without_replicates_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "no replicate"):
            scaling_workflow_times.tahoe_replicates(tahoe_results([], [], [], [0.9]))

    def test_the_record_gives_the_quoted_values(self):
        # Paragraph 5: mean 9.4 hours, mean 78.3 GiB in UMAP, mean 15.3 minutes of Leiden; the record holds 35
        # clusters at or above 90% purity, where the text says 34. Paragraph 8: every stage exponent at or below 1.11.
        summary = scaling_workflow_times.tahoe_replicates(
            scaling_workflow_times.load_json(scaling_workflow_times.TAHOE_RESULTS)
        )
        self.assertEqual(round(summary["meanAnalysisHours"], 1), 9.4)
        # Section 1: a mean of 4.0 hours of store preparation, from 3.82, 4.24 and 3.92.
        self.assertEqual(round(summary["meanPreparationHours"], 1), 4.0)
        self.assertEqual(round(summary["meanUmapPeakGiB"], 1), 78.3)
        self.assertEqual(round(summary["meanLeidenMinutes"], 1), 15.3)
        # Marker testing on all clusters: 3,242, 2,780 and 4,489 seconds, a mean of 58 minutes.
        self.assertEqual(round(summary["meanMarkerMinutes"]), 58)
        self.assertEqual((summary["clusters"], summary["clustersAtPurityThreshold"]), (43, 35))
        self.assertEqual(round(summary["largestStageExponent"], 2), 1.11)

    def stage_runs(self, cells: int, seconds: float, cpus: float = 16.0) -> dict:
        """Stage measurements of the three full replicates: every analysis stage takes ``seconds`` on ``cpus`` CPUs,
        except in the third replicate, where each takes twice as long."""
        return {
            name: {
                "nRows": cells,
                "stages": {
                    stage: {
                        "status": "ok",
                        "seconds": seconds * (2 if i == 2 else 1),
                        "modalCpuLimit": cpus,
                    }
                    for stage in scaling_workflow_times.TAHOE_ANALYSIS_STAGES
                }
                # The import is not an analysis stage, so its seconds stay out of the core-time.
                | {"createStore": {"status": "ok", "seconds": 1e6, "modalCpuLimit": cpus}},
            }
            for i, name in enumerate(scaling_workflow_times.TAHOE_FULL_RUNS)
        }

    def test_the_summary_adds_core_minutes_and_the_csr_size(self):
        results = tahoe_results([8.0, 9.0, 10.0], [77.0, 78.0, 82.0], [840.0, 900.0, 960.0], [0.9])
        results["dataset"] = {"cells": 100_000, "storedValues": 1_000}
        stages = len(scaling_workflow_times.TAHOE_ANALYSIS_STAGES)
        summary = scaling_workflow_times.tahoe_summary(results, self.stage_runs(100_000, 30.0))
        # Each stage of 30 seconds on 16 CPUs is 8 core-minutes; the third replicate doubles them.
        self.assertEqual(
            summary["analysisCoreMinutes"], [8.0 * stages, 8.0 * stages, 16.0 * stages]
        )
        self.assertAlmostEqual(summary["meanAnalysisCoreMinutes"], 32.0 * stages / 3)
        self.assertEqual(summary["uncompressedCsrBytes"], 12_000)
        self.assertEqual(summary["clustersAtPurityThreshold"], 1)

    def test_stage_measurements_of_other_cells_than_the_results_are_an_error(self):
        results = tahoe_results([8.0], [77.0], [840.0], [0.9])
        results["dataset"] = {"cells": 100_000, "storedValues": 1_000}
        with self.assertRaisesRegex(ValueError, "ran on 99999 cells"):
            scaling_workflow_times.tahoe_summary(results, self.stage_runs(99_999, 30.0))

    def test_the_records_give_9065_core_minutes_and_a_152_tib_csr(self):
        # Paragraph 9: 9,065 mean core-minutes on Tahoe-100M, whose 139,624,620,192 stored counts take 1.52 TiB as
        # uncompressed CSR at 12 bytes per count.
        summary = scaling_workflow_times.tahoe_summary(
            scaling_workflow_times.load_json(scaling_workflow_times.TAHOE_RESULTS),
            scaling_workflow_times.load_json(scaling_workflow_times.TAHOE_STAGE_MEASUREMENTS)[
                "runs"
            ],
        )
        self.assertEqual(round(summary["meanAnalysisCoreMinutes"]), 9065)
        self.assertEqual(summary["uncompressedCsrBytes"], 139_624_620_192 * 12)
        self.assertEqual(round(summary["uncompressedCsrBytes"] / 2**40, 2), 1.52)


class CoreTimeFitTest(unittest.TestCase):
    TAHOE_CELLS = 100_000_000

    @staticmethod
    def power_law(cells: int, exponent: float = 0.8) -> float:
        """Core-minutes on a power law of the given exponent through 30 core-minutes at a million cells."""
        return 30.0 * (cells / 1e6) ** exponent

    def embryo_rows(self) -> list[dict]:
        """Three runs per size from 100,000 to 10,000,000 cells on the power law, as replicates gives them, and three
        runs of 10,000 cells far above it, which would bend the fit if it counted them."""
        rows = [{"tag": "small", "nRows": 10_000, "cpus": 4.0, "analysisSeconds": 1e5}] * 3
        for cells, cpus in ((100_000, 4.0), (1_000_000, 8.0), (10_000_000, 16.0)):
            seconds = self.power_law(cells) * 60 / cpus
            rows += [
                {"tag": f"{cells}", "nRows": cells, "cpus": cpus, "analysisSeconds": seconds}
            ] * 3
        return rows

    def tahoe_runs(self, minutes: float, cells: int = TAHOE_CELLS) -> dict:
        """The three full Tahoe-100M replicates with ``minutes`` analysis core-minutes each on 16 CPUs, and an import
        of a million seconds that is not an analysis stage."""
        stages = scaling_workflow_times.TAHOE_ANALYSIS_STAGES
        seconds = minutes * 60 / 16 / len(stages)
        return {
            name: {
                "nRows": cells,
                "stages": {
                    stage: {"status": "ok", "seconds": seconds, "modalCpuLimit": 16.0}
                    for stage in stages
                }
                | {"createStore": {"status": "ok", "seconds": 1e6, "modalCpuLimit": 16.0}},
            }
            for name in scaling_workflow_times.TAHOE_FULL_RUNS
        }

    def test_runs_on_a_power_law_give_its_exponent_and_the_runs_below_100k_cells_are_left_out(self):
        fit = scaling_workflow_times.core_time_fit(
            self.embryo_rows(), self.tahoe_runs(self.power_law(self.TAHOE_CELLS))
        )
        # log10 core-minutes = log10(30) + 0.8 * (log10 cells - 6) on every run from 100,000 cells up.
        self.assertAlmostEqual(fit["exponent"], 0.8, places=9)
        self.assertAlmostEqual(fit["intercept"], math.log10(30.0) - 4.8, places=9)
        self.assertEqual(fit["fromCells"], 100_000)
        self.assertEqual(
            [(s["cells"], s["dataset"], s["cpus"]) for s in fit["sizes"]],
            [
                (100_000, "embryo", 4.0),
                (1_000_000, "embryo", 8.0),
                (10_000_000, "embryo", 16.0),
                (self.TAHOE_CELLS, "tahoe", 16.0),
            ],
        )
        means = [size["meanCoreMinutes"] for size in fit["sizes"]]
        for mean, expected in zip(means, (30 * 0.1**0.8, 30.0, 30 * 10**0.8, 30 * 100**0.8)):
            self.assertAlmostEqual(mean, expected, places=9)
        self.assertEqual([len(size["coreMinutes"]) for size in fit["sizes"]], [3, 3, 3, 3])

    def test_rows_as_the_derived_csv_stores_them_give_the_same_fit(self):
        # The figure reads the runs from the derived CSV as text, the summary from replicates as numbers.
        rows = self.embryo_rows()
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        tahoe = self.tahoe_runs(4000.0)
        self.assertEqual(
            scaling_workflow_times.core_time_fit(list(csv.DictReader(handle)), tahoe),
            scaling_workflow_times.core_time_fit(rows, tahoe),
        )

    def test_tahoe_at_an_embryo_size_or_fewer_than_three_sizes_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "Tahoe-100M needs three runs at a size of its own"):
            scaling_workflow_times.core_time_fit(
                self.embryo_rows(), self.tahoe_runs(4000.0, cells=10_000_000)
            )
        one_size = [row for row in self.embryo_rows() if row["nRows"] <= 100_000]
        with self.assertRaisesRegex(ValueError, "at least three sizes of 100,000 cells or more"):
            scaling_workflow_times.core_time_fit(one_size, self.tahoe_runs(4000.0))

    def test_the_fit_from_a_million_cells_leaves_out_the_smaller_runs_that_bend_the_whole_fit(self):
        # Linear core-time from a million cells (exponent 1 through 30 core-minutes at a million cells), with the
        # 100,000-cell runs at 20 core-minutes, far above the line's 3: the whole fit bends below 1, the fit from a
        # million cells does not.
        rows = []
        for cells, cpus, minutes in (
            (100_000, 4.0, 20.0),
            (1_000_000, 8.0, 30.0),
            (10_000_000, 16.0, 300.0),
        ):
            rows += [{"nRows": cells, "cpus": cpus, "analysisSeconds": minutes * 60 / cpus}] * 3
        fit = scaling_workflow_times.core_time_fit(rows, self.tahoe_runs(30.0 * 100))
        large = fit["fromMillionCells"]
        self.assertAlmostEqual(large["exponent"], 1.0, places=9)
        self.assertAlmostEqual(large["intercept"], math.log10(30.0) - 6, places=9)
        self.assertEqual((large["fromCells"], large["runs"]), (1_000_000, 9))
        self.assertLess(fit["exponent"], 0.9)
        # The two embryo means alone: 300 against 30 core-minutes over a tenfold size, exponent 1.
        two_point = fit["twoPoint"]
        self.assertEqual(two_point["cells"], [1_000_000, 10_000_000])
        self.assertEqual([round(m, 9) for m in two_point["meanCoreMinutes"]], [30.0, 300.0])
        self.assertAlmostEqual(two_point["exponent"], 1.0, places=9)

    def test_the_two_point_exponent_needs_both_embryo_sizes(self):
        rows = [
            {"nRows": cells, "cpus": 8.0, "analysisSeconds": 60.0 * cells / 100_000}
            for cells in (100_000, 500_000, 1_000_000)
            for _ in range(3)
        ]
        with self.assertRaisesRegex(
            ValueError, "needs embryo runs at 1,000,000 and 10,000,000 cells"
        ):
            scaling_workflow_times.core_time_fit(rows, self.tahoe_runs(4000.0))

    def test_one_size_from_a_million_cells_is_an_error(self):
        rows = [
            {"nRows": cells, "cpus": 4.0, "analysisSeconds": 600.0 * cells / 100_000}
            for cells in (100_000, 500_000)
            for _ in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "from 1,000,000 cells needs at least two sizes"):
            scaling_workflow_times.core_time_fit(rows, self.tahoe_runs(4000.0))

    def test_the_records_give_an_exponent_of_0_87_from_24_7_to_9065_core_minutes(self):
        # Paragraph 9: analysis core-time grew with an exponent of 0.87, from 24.7 mean core-minutes at 100,000 cells
        # to 9,065 on Tahoe-100M. Worked out here from the raw records alone: an embryo run's core-minutes are its
        # whole run without the input download and the three conversion stages, times its CPUs; a Tahoe-100M
        # replicate's add every stage but the conversion stages, each times its call's CPUs; the exponent is the
        # least-squares slope of log10 core-minutes against log10 cells in closed form.
        conversion = ("createStore", "writeCountsT", "initializeStore")
        points = []
        for run in json.loads(scaling_workflow_times.SCALING_RUNS.read_text()).values():
            funnel = run["funnel"]
            whole = {o["stage"]: o["wholeFunctionSeconds"] for o in funnel["outcomes"]}
            seconds = (
                funnel["wholeFunctionSeconds"]
                - funnel["datasetDownloadSeconds"]
                - sum(whole[stage] for stage in conversion)
            )
            if funnel["nRows"] >= 100_000:
                points.append(
                    (funnel["nRows"], funnel["modalResources"]["modalCpuLimit"] * seconds / 60)
                )
        stage_runs = json.loads(scaling_workflow_times.TAHOE_STAGE_MEASUREMENTS.read_text())["runs"]
        for name in ("full", "fullReplicate2", "fullReplicate3"):
            stages = stage_runs[name]["stages"]
            minutes = sum(
                s["seconds"] * s["modalCpuLimit"]
                for stage, s in stages.items()
                if stage not in conversion
            )
            points.append((stage_runs[name]["nRows"], minutes / 60))
        x = [math.log10(cells) for cells, _ in points]
        y = [math.log10(minutes) for _, minutes in points]
        mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
        slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / sum(
            (a - mean_x) ** 2 for a in x
        )
        smallest = [minutes for cells, minutes in points if cells == 100_000]

        fit = scaling_workflow_times.core_time_fit(
            scaling_workflow_times.selected_replicates(scaling_workflow_times.replicates()),
            stage_runs,
        )
        self.assertEqual((len(points), len(smallest)), (18, 3))
        self.assertEqual(fit["runs"], len(points))
        self.assertAlmostEqual(fit["exponent"], slope, places=12)
        self.assertEqual(round(fit["exponent"], 2), 0.87)
        means = {size["cells"]: size["meanCoreMinutes"] for size in fit["sizes"]}
        self.assertAlmostEqual(means[100_000], sum(smallest) / 3, places=9)
        # Paragraph 71: between the one- and ten-million-cell embryo means alone the exponent is 0.94.
        tenfold = [
            sum(m for cells, m in points if cells == size) / 3 for size in (1_000_000, 10_000_000)
        ]
        self.assertAlmostEqual(
            fit["twoPoint"]["exponent"], math.log10(tenfold[1] / tenfold[0]), places=12
        )
        self.assertEqual(round(fit["twoPoint"]["exponent"], 2), 0.94)
        self.assertEqual(round(means[100_000], 1), 24.7)
        self.assertEqual(round(means[95_624_334]), 9065)
        # Paragraph 9: an exponent of 1.00 over the 12 runs from 1,000,000 cells, from 101.8 mean core-minutes at a
        # million cells, in the same closed form.
        large = [
            (math.log10(cells), math.log10(minutes)) for cells, minutes in points if cells >= 1e6
        ]
        mean_x = sum(a for a, _ in large) / len(large)
        mean_y = sum(b for _, b in large) / len(large)
        large_slope = sum((a - mean_x) * (b - mean_y) for a, b in large) / sum(
            (a - mean_x) ** 2 for a, _ in large
        )
        self.assertEqual(fit["fromMillionCells"]["runs"], len(large))
        self.assertEqual(len(large), 12)
        self.assertAlmostEqual(fit["fromMillionCells"]["exponent"], large_slope, places=12)
        self.assertEqual(round(fit["fromMillionCells"]["exponent"], 2), 1.00)
        self.assertEqual(round(means[1_000_000], 1), 101.8)


def layout_record(cells: int, layout: str, replicate: int, range_reads: int) -> dict:
    run_id = f"embryo_{cells}_{layout}_replicate_{replicate}"
    return {
        "status": "completed",
        "result": {
            "mode": "measurement",
            "settings": {"runId": run_id, "nRows": cells, "layout": layout, "replicate": replicate},
            "stages": [
                {"stage": "markHvgs", "details": {"storeOperations": {"rangeGets": range_reads}}},
            ],
        },
    }


class CountRequestSummaryTest(unittest.TestCase):
    def write_records(self, reads: dict, *, passed: bool = True) -> None:
        """A count_requests record with three replicates per size and layout, each reading ``reads[(cells, layout)]``
        ranges, or the list of three given there, and a paired validation that passed or not."""
        records = {}
        for (cells, layout), found in reads.items():
            for replicate, n in enumerate(
                found if isinstance(found, list) else [found] * 3, start=1
            ):
                record = layout_record(cells, layout, replicate, n)
                records[record["result"]["settings"]["runId"]] = record
        records["paired_layout_validation"] = {
            "status": "completed",
            "result": {
                "mode": "pairedLayoutValidation",
                "passed": passed,
                "measurements": [{"runId": key, "passed": True} for key in records],
            },
        }
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "count_requests.json"
        path.write_text(json.dumps({"records": records}))
        for name, value in {"COUNT_REQUESTS": path, "ROOT": Path(directory.name)}.items():
            patch = mock.patch.object(count_request_summary, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    READS = {
        (5_000_000, "fixed"): 400,
        (5_000_000, "scaled"): 100,
        (10_000_000, "fixed"): 1_600,
        (10_000_000, "scaled"): 200,
    }

    def test_the_fixed_layout_reads_over_the_scaled_at_each_size(self):
        self.write_records(self.READS)
        requests = count_request_summary.read_requests()
        self.assertEqual(len(requests["rows"]), 4)
        self.assertEqual(
            count_request_summary.fixed_over_scaled(requests["rows"]),
            {"5000000": 4.0, "10000000": 8.0},
        )

    def test_replicates_that_disagree_or_a_failed_validation_are_an_error(self):
        self.write_records(self.READS | {(10_000_000, "fixed"): [1_600, 1_600, 1_601]})
        with self.assertRaisesRegex(ValueError, "three runs with the same range reads"):
            count_request_summary.read_requests()
        self.write_records(self.READS, passed=False)
        with self.assertRaisesRegex(ValueError, "paired validation that passed"):
            count_request_summary.read_requests()

    def test_a_missing_size_or_layout_is_an_error(self):
        reads = dict(self.READS)
        del reads[(5_000_000, "scaled")]
        self.write_records(reads)
        with self.assertRaisesRegex(ValueError, "both layouts at 5 and 10 million"):
            count_request_summary.read_requests()

    def write_pass_seconds(self, seconds: dict) -> None:
        """A count_requests record whose measurement run of each size, layout and replicate takes
        ``seconds[(stage, layout, cells)][replicate - 1]`` in each stage given, with a paired validation that passed."""
        runs: dict = {}
        for (stage, layout, cells), times in seconds.items():
            for replicate, value in enumerate(times, start=1):
                if (cells, layout, replicate) not in runs:
                    runs[(cells, layout, replicate)] = layout_record(cells, layout, replicate, 1)
                    runs[(cells, layout, replicate)]["result"]["stages"] = []
                runs[(cells, layout, replicate)]["result"]["stages"].append(
                    {"stage": stage, "seconds": value}
                )
        records = {record["result"]["settings"]["runId"]: record for record in runs.values()}
        records["paired_layout_validation"] = {
            "status": "completed",
            "result": {
                "mode": "pairedLayoutValidation",
                "passed": True,
                "measurements": [{"runId": key, "passed": True} for key in records],
            },
        }
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "count_requests.json"
        path.write_text(json.dumps({"records": records}))
        for name, value in {"COUNT_REQUESTS": path, "ROOT": Path(directory.name)}.items():
            patch = mock.patch.object(count_request_summary, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def pass_seconds(small: list[float], large: list[float]) -> dict:
        """The same seconds at five and ten million cells for every pass on both layouts."""
        return {
            (stage, layout, cells): times
            for stage, _ in count_request_summary.PASSES
            for layout in count_request_summary.LAYOUTS
            for cells, times in ((5_000_000, small), (10_000_000, large))
        }

    def test_the_pass_exponent_comes_from_the_mean_seconds_with_the_range_over_run_pairs(self):
        # Doubling the cells doubles the mean seconds (110 to 220), so b = 1, where the medians (100 to 220) would give
        # 1.14; the extreme run pairs are 150 to 200 and 80 to 240 seconds.
        seconds = self.pass_seconds([80.0, 100.0, 150.0], [200.0, 220.0, 240.0])
        seconds[("findMarkers", "scaled", 10_000_000)] = [440.0, 440.0, 440.0]
        self.write_pass_seconds(seconds)
        summary = count_request_summary.pass_exponents()
        self.assertEqual(summary["cells"], [5_000_000, 10_000_000])
        rows = {(row["pass"], row["layout"]): row for row in summary["rows"]}
        self.assertEqual(len(rows), 6)
        hvg = rows[("markHvgs", "fixed")]
        self.assertAlmostEqual(hvg["exponent"], 1.0, places=12)
        self.assertAlmostEqual(hvg["low"], math.log2(200 / 150), places=12)
        self.assertAlmostEqual(hvg["high"], math.log2(3), places=12)
        self.assertEqual(
            hvg["seconds"], {"5000000": [80.0, 100.0, 150.0], "10000000": [200.0, 220.0, 240.0]}
        )
        # Four times the mean seconds at twice the cells is b = 2.
        self.assertAlmostEqual(rows[("findMarkers", "scaled")]["exponent"], 2.0, places=12)
        self.assertEqual(rows[("runNormalization", "scaled")]["name"], "normalization")

    def test_a_pass_without_both_sizes_or_three_runs_is_an_error(self):
        seconds = self.pass_seconds([100.0] * 3, [200.0] * 3)
        del seconds[("runNormalization", "scaled", 5_000_000)]
        self.write_pass_seconds(seconds)
        with self.assertRaisesRegex(
            ValueError, "runNormalization on the scaled layout needs runs at 5 and 10"
        ):
            count_request_summary.pass_exponents()
        seconds = self.pass_seconds([100.0] * 3, [200.0] * 3)
        seconds[("findMarkers", "fixed", 10_000_000)] = [200.0, 200.0]
        self.write_pass_seconds(seconds)
        with self.assertRaisesRegex(
            ValueError, "findMarkers on the fixed layout needs three runs at each size"
        ):
            count_request_summary.pass_exponents()

    def test_the_records_give_the_pass_exponents_of_note_65(self):
        # Note 65: from 5 to 10 million cells the exponents of variable-gene selection and normalization fall from 1.44
        # and 1.41 on the default layout to 0.95 on the wider one, and that of marker search from 1.10 to 1.02.
        rows = {
            (row["pass"], row["layout"]): row["exponent"]
            for row in count_request_summary.pass_exponents()["rows"]
        }
        self.assertEqual(
            {key: round(value, 2) for key, value in rows.items()},
            {
                ("markHvgs", "fixed"): 1.44,
                ("markHvgs", "scaled"): 0.95,
                ("runNormalization", "fixed"): 1.41,
                ("runNormalization", "scaled"): 0.95,
                ("findMarkers", "fixed"): 1.10,
                ("findMarkers", "scaled"): 1.02,
            },
        )

    def test_the_records_cut_variable_gene_range_reads_78_fold_at_ten_million_cells(self):
        # Paragraph 3: 83,812 range reads on the fixed layout against 10,695 on the scaled one at 10 million cells.
        rows = count_request_summary.read_requests()["rows"]
        reads = {(row["cells"], row["layout"]): row["rangeReads"] for row in rows}
        self.assertEqual(
            (reads[(10_000_000, "fixed")], reads[(10_000_000, "scaled")]), (83_812, 10_695)
        )
        folds = count_request_summary.fixed_over_scaled(rows)
        self.assertEqual(round(folds["10000000"], 1), 7.8)
        self.assertAlmostEqual(folds["10000000"], 83_812 / 10_695)

    def test_decoded_bytes_cut_is_one_minus_the_paired_over_the_cell_major_amplification(self):
        geometry = {
            "rows": [
                {
                    "nCells": 100,
                    "amplificationCellMajorOnly": 1.0,
                    "decodeAmplificationPairedWrite": 1.0,
                },
                {
                    "nCells": 200,
                    "amplificationCellMajorOnly": 10.0,
                    "decodeAmplificationPairedWrite": 2.0,
                },
                {
                    "nCells": 300,
                    "amplificationCellMajorOnly": 4.0,
                    "decodeAmplificationPairedWrite": 1.0,
                },
            ]
        }
        self.assertEqual(
            count_request_summary.decoded_bytes_cut(geometry), {"100": 0.0, "200": 0.8, "300": 0.75}
        )

    def test_a_layout_without_a_positive_amplification_is_an_error(self):
        geometry = {
            "rows": [
                {
                    "nCells": 5,
                    "amplificationCellMajorOnly": 0.0,
                    "decodeAmplificationPairedWrite": 1.0,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "at 5 cells needs positive"):
            count_request_summary.decoded_bytes_cut(geometry)

    def test_the_records_cut_decoded_bytes_by_98_9_percent_at_ten_million_cells(self):
        # Paragraph 3: 1.0 against 91 bytes decoded per byte used, a 98.9% cut.
        geometry = json.loads(count_request_summary.LAYOUT_GEOMETRY.read_text())
        largest = next(row for row in geometry["rows"] if row["nCells"] == 10_000_000)
        self.assertEqual(
            (largest["amplificationCellMajorOnly"], largest["decodeAmplificationPairedWrite"]),
            (91, 1.0),
        )
        cut = count_request_summary.decoded_bytes_cut(geometry)["10000000"]
        self.assertEqual(round(100 * cut, 1), 98.9)


if __name__ == "__main__":
    unittest.main()
