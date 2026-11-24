import unittest

from experiments.benchmarks import analysis_times


def arm(
    dataset: str,
    tool: str,
    minutes: list[float],
    records: list[str] | None = None,
    workflow: list[float] | None = None,
) -> dict:
    workflow = workflow or minutes
    return {
        "dataset": dataset,
        "tool": tool,
        "n": len(minutes),
        "analysisMinutes": minutes,
        "meanAnalysisMinutes": sum(minutes) / len(minutes),
        "meanWorkflowMinutes": sum(workflow) / len(workflow),
        "containerGib": 32.0,
        "completedAtOtherContainers": [],
        "sourceRecords": records or [f"{tool}_{i}" for i in range(len(minutes))],
    }


class SpeedupTest(unittest.TestCase):
    def test_the_speedup_inverts_the_ratio_and_its_interval(self):
        # The comparator's logs are CyteArc's shifted by log 4 with the same spread: the ratio of geometric means is
        # 1/4, so CyteArc ran 4 times faster. The reference interval of the comparator over CyteArc, 0.83107 to
        # 19.2523, comes from integrating the t density numerically on 4 degrees of freedom, not from scipy.
        (row,) = analysis_times.comparisons(
            [
                arm("embryo_1000000", "cytearc", [1.0, 2.0, 4.0]),
                arm("embryo_1000000", "bpcells", [4.0, 8.0, 16.0]),
            ]
        )
        self.assertAlmostEqual(row["ratio"], 0.25, places=12)
        self.assertAlmostEqual(row["speedup"], 4.0, places=12)
        self.assertAlmostEqual(row["speedupLow"], 0.83107, places=4)
        self.assertAlmostEqual(row["speedupHigh"], 19.2523, places=3)
        # The reduction keeps its meaning beside the speedup: CyteArc saves three quarters of the time.
        self.assertAlmostEqual(row["reduction"], 0.75, places=12)

    def test_a_comparator_with_one_run_has_a_speedup_without_an_interval(self):
        (row,) = analysis_times.comparisons(
            [
                arm("brain_1306127", "cytearc", [2.0, 2.0, 2.0]),
                arm("brain_1306127", "bpcells", [5.0]),
            ]
        )
        self.assertAlmostEqual(row["speedup"], 2.5, places=12)
        self.assertIsNone(row["speedupLow"])
        self.assertIsNone(row["speedupHigh"])

    def test_the_workflow_saving_compares_arithmetic_means_with_preparation(self):
        # Workflow minutes 2 and 6 average 4 for CyteArc against 10 for the comparator: 60% saved. The geometric means
        # (3.46 against 10) would give 65%, so a saving taken from them fails here.
        (row,) = analysis_times.comparisons(
            [
                arm("embryo_1000000", "cytearc", [1.0, 1.0], workflow=[2.0, 6.0]),
                arm("embryo_1000000", "bpcells", [5.0], workflow=[10.0]),
            ]
        )
        self.assertEqual(
            (row["cytearcMeanWorkflowMinutes"], row["otherMeanWorkflowMinutes"]), (4.0, 10.0)
        )
        self.assertAlmostEqual(row["workflowSaving"], 0.6, places=12)
        # The analysis-time reduction stays the analysis times' own: 1 against 5 minutes.
        self.assertAlmostEqual(row["reduction"], 0.8, places=12)


def workflow_row(
    tool: str, replicate: int, container: float, minutes: float | None, peak: float
) -> dict:
    return {
        "dataset": "embryo_1000000",
        "tool": tool,
        "replicate": replicate,
        "completedAtGib": container,
        "cpu": 8,
        "sourceRecord": f"{tool}_{replicate}_{container:g}",
        "analysisTimeMinutes": minutes,
        "preparationMinutes": 1.0,
        "workflowMinutes": None if minutes is None else minutes + 1.0,
        "peakCgroupGib": peak,
    }


class ArmsTest(unittest.TestCase):
    def test_an_arms_peak_is_the_mean_over_its_smallest_container_runs(self):
        # Two runs completed in 32 GiB containers at 20 and 23 GiB; a third completed only in 64 GiB at 50 GiB and
        # stays out, as does a run that did not complete.
        rows = {
            ("embryo_1000000", "bpcells"): [
                workflow_row("bpcells", 1, 32.0, 9.0, 20.0),
                workflow_row("bpcells", 2, 32.0, 10.0, 23.0),
                workflow_row("bpcells", 3, 64.0, 11.0, 50.0),
                workflow_row("bpcells", 4, 32.0, None, 31.0),
            ]
        }
        (drawn,), missing = analysis_times.arms(rows)
        self.assertEqual(missing, [])
        self.assertEqual(drawn["peakCgroupGib"], [20.0, 23.0])
        self.assertEqual(drawn["meanPeakCgroupGib"], 21.5)
        self.assertEqual(drawn["completedAtOtherContainers"], [64.0])


def cytearc_steps(scale: float) -> dict[str, float]:
    # Seconds that make 4, 2, 3, 1 and 2 minutes of the five stages and 0.1 minutes of reopening the store, times scale.
    seconds = {
        "filterCells": 60,
        "markHvgs": 60,
        "runNormalization": 60,
        "runPca": 60,
        "buildAnnIndex": 30,
        "queryNeighbors": 30,
        "buildConnectivityMap": 60,
        "buildEmbeddingInitialization": 60,
        "runUmap": 120,
        "runLeiden": 60,
        "findMarkers": 120,
        "createObject": 6,
    }
    return {step: value * scale for step, value in seconds.items()}


BPCELLS_STEPS = {
    # 8, 6, 9, 30 and 4 minutes of the five stages.
    "filterCells": 120,
    "runNormalization": 120,
    "markHvgs": 120,
    "runPca": 120,
    "queryNeighbors": 360,
    "runUmap": 540,
    "runLeiden": 1800,
    "findMarkers": 240,
}


class StageTimesTest(unittest.TestCase):
    def setUp(self):
        self.drawn = [
            arm("embryo_10000000", "cytearc", [12.1, 24.2], ["c1", "c2"]),
            arm("embryo_10000000", "bpcells_cached", [57.0, 57.0], ["b1", "b2"]),
            # Another dataset's runs stay out of the stage times.
            arm("embryo_1000000", "cytearc", [1.0], ["other"]),
        ]
        self.seconds = {
            "c1": cytearc_steps(1),
            "c2": cytearc_steps(2),
            "b1": dict(BPCELLS_STEPS),
            "b2": dict(BPCELLS_STEPS),
        }

    def test_each_stage_holds_its_mean_minutes_and_the_comparators_over_cytearcs(self):
        result = analysis_times.stage_times(self.drawn, self.seconds)
        self.assertEqual(result["dataset"], "embryo_10000000")
        self.assertEqual(result["tool"], "bpcells_cached")
        stages = {s["stage"]: s for s in result["stages"]}
        self.assertEqual(list(stages), ["filter–PCA", "neighbors", "UMAP", "Leiden", "markers"])
        expected = {
            # stage: (CyteArc's minutes of the two runs, BPCells' mean minutes)
            "filter–PCA": ([4.0, 8.0], 8.0),
            "neighbors": ([2.0, 4.0], 6.0),
            "UMAP": ([3.0, 6.0], 9.0),
            "Leiden": ([1.0, 2.0], 30.0),
            "markers": ([2.0, 4.0], 4.0),
        }
        for name, (cytearc, other) in expected.items():
            stage = stages[name]
            for got, want in zip(stage["cytearcMinutes"], cytearc, strict=True):
                self.assertAlmostEqual(got, want, places=9)
            self.assertAlmostEqual(stage["cytearcMeanMinutes"], sum(cytearc) / 2, places=9)
            self.assertAlmostEqual(stage["otherMeanMinutes"], other, places=9)
            self.assertAlmostEqual(stage["otherOverCytearc"], other / (sum(cytearc) / 2), places=9)
        # Leiden: 30 minutes against a mean of 1.5.
        self.assertAlmostEqual(stages["Leiden"]["otherOverCytearc"], 20.0, places=9)
        self.assertEqual(stages["neighbors"]["otherSteps"], ["queryNeighbors"])
        # The four stages after the count passes: CyteArc 3 + 4.5 + 1.5 + 3 and BPCells 6 + 9 + 30 + 4 mean minutes.
        self.assertEqual(result["downstreamStages"], ["neighbors", "UMAP", "Leiden", "markers"])
        self.assertAlmostEqual(result["cytearcDownstreamMeanMinutes"], 12.0, places=9)
        self.assertAlmostEqual(result["otherDownstreamMeanMinutes"], 49.0, places=9)

    def test_stages_that_do_not_add_up_to_the_analysis_time_are_an_error(self):
        # The stages and the reopened store of c1 add up to 12.1 minutes, 0.1 more than this analysis time.
        self.drawn[0]["analysisMinutes"] = [12.0, 24.2]
        with self.assertRaisesRegex(ValueError, "c1: the stages add up to 12.10 minutes"):
            analysis_times.stage_times(self.drawn, self.seconds)

    def test_a_missing_step_measurement_is_an_error(self):
        del self.seconds["b2"]["runLeiden"]
        with self.assertRaisesRegex(ValueError, "b2 has no step measurement of \\['runLeiden'\\]"):
            analysis_times.stage_times(self.drawn, self.seconds)

    def test_a_tool_without_runs_on_the_dataset_is_an_error(self):
        with self.assertRaisesRegex(
            ValueError, "bpcells_cached has no completed run on embryo_10000000"
        ):
            analysis_times.stage_times(self.drawn[:1], self.seconds)


def cytearc_arm(dataset: str, records: list[str], minutes: float = 1.0, cpu: int = 8) -> dict:
    return {**arm(dataset, "cytearc", [minutes] * len(records), records), "cpu": cpu}


class StepPeaksTest(unittest.TestCase):
    def setUp(self):
        # One million cells, two runs: variable genes 4 and 6 GiB, normalization 2 and 2, PCA 3 and 5; neighbor index 1
        # and 1, neighbor query 2 and 4, UMAP 3 and 3, Leiden 1 and 3. Ten million cells: three times as much.
        one = {
            "markHvgs": (4, 6),
            "runNormalization": (2, 2),
            "runPca": (3, 5),
            "buildAnnIndex": (1, 1),
            "queryNeighbors": (2, 4),
            "runUmap": (3, 3),
            "runLeiden": (1, 3),
            "findMarkers": (6, 8),
        }
        self.peaks = {}
        for prefix, scale in (("small", 1), ("large", 3)):
            for run in (0, 1):
                self.peaks[f"{prefix}{run}"] = {
                    step: values[run] * scale for step, values in one.items()
                }
        self.drawn = [
            cytearc_arm("embryo_1000000", ["small0", "small1"]),
            cytearc_arm("embryo_10000000", ["large0", "large1"]),
        ]

    def test_each_group_takes_its_highest_mean_step_peak(self):
        result = analysis_times.cytearc_step_peaks(self.drawn, self.peaks)
        small, large = result["embryo_1000000"], result["embryo_10000000"]
        # Means: variable genes 5, normalization 2, PCA 4; graph steps 1, 3, 3 and 2.
        self.assertEqual(small["meanPeakGiB"]["markHvgs"], 5.0)
        self.assertEqual((small["streamedPeakGiB"], small["graphPeakGiB"]), (5.0, 3.0))
        self.assertEqual((large["streamedPeakGiB"], large["graphPeakGiB"]), (15.0, 9.0))
        self.assertEqual(small["runs"], 2)
        # Marker search stands apart from both groups: means 7 and 21, above every other step.
        self.assertEqual((small["markersPeakGiB"], large["markersPeakGiB"]), (7.0, 21.0))

    def test_a_run_without_the_peak_of_a_step_is_an_error(self):
        del self.peaks["large1"]["runLeiden"]
        with self.assertRaisesRegex(ValueError, "large1.*runLeiden"):
            analysis_times.cytearc_step_peaks(self.drawn, self.peaks)


class NeighborIndexSecondsTest(unittest.TestCase):
    def test_the_index_build_and_query_seconds_are_summed_per_run_then_averaged(self):
        # Runs of 10 + 6 and 12 + 6 seconds average 17; the connectivity graph and another dataset's run stay out.
        drawn = [
            cytearc_arm("embryo_1000000", ["a", "b"]),
            cytearc_arm("embryo_10000000", ["c"]),
        ]
        seconds = {
            "a": {"buildAnnIndex": 10.0, "queryNeighbors": 6.0, "buildConnectivityMap": 50.0},
            "b": {"buildAnnIndex": 12.0, "queryNeighbors": 6.0},
            "c": {"buildAnnIndex": 100.0, "queryNeighbors": 100.0},
        }
        result = analysis_times.cytearc_neighbor_index_seconds(drawn, seconds, "embryo_1000000")
        self.assertEqual(result["seconds"], [16.0, 18.0])
        self.assertEqual(result["meanSeconds"], 17.0)
        self.assertEqual(result["steps"], ["buildAnnIndex", "queryNeighbors"])

    def test_a_run_without_an_index_step_or_a_dataset_without_cytearc_is_an_error(self):
        drawn = [cytearc_arm("embryo_1000000", ["a"])]
        with self.assertRaisesRegex(
            ValueError, "a has no step measurement of \\['queryNeighbors'\\]"
        ):
            analysis_times.cytearc_neighbor_index_seconds(
                drawn, {"a": {"buildAnnIndex": 1.0}}, "embryo_1000000"
            )
        with self.assertRaisesRegex(ValueError, "CyteArc has no completed run on brain_1306127"):
            analysis_times.cytearc_neighbor_index_seconds(drawn, {}, "brain_1306127")


class CoreTimeExponentTest(unittest.TestCase):
    def test_the_exponent_compares_minutes_times_cpus_across_the_sizes(self):
        # 6 minutes on 8 CPUs is 48 core-minutes at one million cells; 48 minutes on 16 CPUs is 768 at ten million.
        # 768 / 48 = 16 over a tenfold size gives b = log10(16) = 1.20412.
        drawn = [
            cytearc_arm("embryo_10000000", ["b"], minutes=48.0, cpu=16),
            cytearc_arm("embryo_1000000", ["a"], minutes=6.0, cpu=8),
        ]
        result = analysis_times.cytearc_core_time_exponent(drawn)
        self.assertEqual(result["datasets"], ["embryo_1000000", "embryo_10000000"])
        self.assertEqual(result["coreMinutes"], [48.0, 768.0])
        self.assertAlmostEqual(result["exponent"], 1.204120, places=6)


class UncompressedCsrTest(unittest.TestCase):
    def test_each_stored_count_takes_twelve_bytes(self):
        self.assertEqual(analysis_times.uncompressed_csr_bytes(0), 0)
        self.assertEqual(analysis_times.uncompressed_csr_bytes(1_000), 12_000)
        with self.assertRaisesRegex(ValueError, "not -1"):
            analysis_times.uncompressed_csr_bytes(-1)

    def test_the_peak_samples_take_their_stored_counts_from_the_inputs(self):
        inputs = {
            "embryo_1000000": {"nnz": 10},
            "embryo_10000000": {"nnz": 100},
            "other": {"nnz": 5},
        }
        self.assertEqual(
            analysis_times.peak_dataset_csr_bytes(inputs),
            {"embryo_1000000": 120, "embryo_10000000": 1_200},
        )
        with self.assertRaisesRegex(ValueError, "embryo_10000000"):
            analysis_times.peak_dataset_csr_bytes({"embryo_1000000": {"nnz": 10}})

    def test_the_records_give_191_and_1906_gib(self):
        # Paragraph 9: 19.1 GiB at 1 million cells (1,706,570,607 stored counts) and 190.6 GiB at 10 million.
        inputs = analysis_times.load_json(analysis_times.WORKFLOW_RUNS)["inputs"]
        sizes = analysis_times.peak_dataset_csr_bytes(inputs)
        self.assertEqual(sizes["embryo_1000000"], 1_706_570_607 * 12)
        self.assertEqual(
            [round(sizes[d] / 2**30, 1) for d in ("embryo_1000000", "embryo_10000000")],
            [19.1, 190.6],
        )


if __name__ == "__main__":
    unittest.main()
