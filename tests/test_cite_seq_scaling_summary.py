import unittest

from experiments.integration import cite_seq_scaling_summary as summary


def window(stage: str, seconds: float | None, unfinished: bool = False) -> dict:
    found = {"stage": stage, "wallSeconds": seconds}
    return found | {"unfinished": True} if unfinished else found


def cytearc(cells: int, initial: float, stored: float) -> dict:
    return {
        "status": "ok",
        "cells": cells,
        "jointMethod": "wnn",
        "initialWnnSeconds": initial,
        "revisedWnnSeconds": stored,
    }


def agreement(
    method: str,
    seed: str,
    ari: float,
    target: str = "19",
    combat: float | None = None,
    stephenson: float | None = None,
) -> dict:
    return {
        "method": method,
        "partition_rule": "matched_cluster_number",
        "target_clusters": target,
        "seed": seed,
        "ari": str(ari),
        "ari_combat": str(ari if combat is None else combat),
        "ari_stephenson": str(ari if stephenson is None else stephenson),
    }


class SeuratPhaseTest(unittest.TestCase):
    def test_a_phase_sums_its_windows_and_none_while_one_is_unfinished_or_missing(self):
        windows = [
            window("input_setup", 50.0),
            window("initial:RNA.normalization", 30.0),
            window("initial:WNN.clustering", 70.0),
            window("reuse_revision:WNN.clustering", 12.0),
            window("fresh_revision:RNA.normalization", None, unfinished=True),
        ]
        self.assertEqual(summary.seurat_phase_seconds(windows, "initial"), 100.0)
        self.assertEqual(summary.seurat_phase_seconds(windows, "reuse_revision"), 12.0)
        self.assertEqual(summary.seurat_phase_seconds(windows, "input_setup"), 50.0)
        self.assertIsNone(summary.seurat_phase_seconds(windows, "fresh_revision"))
        self.assertIsNone(summary.seurat_phase_seconds(windows, "export_graphs"))


class SizeTimesTest(unittest.TestCase):
    def test_cytearc_speedups_divide_seurats_finished_phases_by_cytearcs(self):
        seurat = {
            "cells": 10,
            "outcome": "time_limit",
            "stageWindows": [
                window("initial:RNA.normalization", 300.0),
                window("initial:WNN.clustering", 900.0),
                window("reuse_revision:WNN.clustering", 600.0),
                window("fresh_revision:WNN.clustering", None, unfinished=True),
            ],
        }
        times = summary.size_times(10, cytearc(10, 40.0, 2.0), seurat)
        # Seurat's first analysis took 1,200 s against 40 s, and its stored re-clustering 600 s against 2 s.
        self.assertEqual(times["initialSpeedup"], 30.0)
        self.assertEqual(times["storedSpeedup"], 300.0)
        self.assertEqual(
            times["seurat"],
            {"runOutcome": "time_limit", "initialSeconds": 1200.0, "storedSeconds": 600.0},
        )

    def test_a_seurat_run_stopped_in_its_first_analysis_has_no_speedups(self):
        seurat = {
            "cells": 10,
            "outcome": "memory_limit",
            "stageWindows": [window("initial:RNA.normalization", None, unfinished=True)],
        }
        times = summary.size_times(10, cytearc(10, 40.0, 2.0), seurat)
        self.assertIsNone(times["initialSpeedup"])
        self.assertIsNone(times["storedSpeedup"])

    def test_records_of_other_cells_or_an_unfinished_cytearc_run_are_an_error(self):
        seurat = {"cells": 10, "outcome": "completed", "stageWindows": []}
        with self.assertRaisesRegex(ValueError, "not a completed WNN-first run"):
            summary.size_times(10, cytearc(10, 1.0, 1.0) | {"status": "failed"}, seurat)
        with self.assertRaisesRegex(ValueError, "holds a run on 11"):
            summary.size_times(10, cytearc(10, 1.0, 1.0), seurat | {"cells": 11})


class SizeAgreementTest(unittest.TestCase):
    def test_means_over_the_seeds_of_the_seeded_cluster_number(self):
        rows = [
            agreement("cytearc_wnn", "1", 0.40),
            agreement("cytearc_wnn", "2", 0.50),
            agreement("seurat_wnn", "1", 0.30),
            agreement("seurat_wnn", "2", 0.36),
            # One seed at another cluster number, and graphs that are not WNN, stay out.
            agreement("cytearc_wnn", "1", 0.99, target="25"),
            agreement("rna", "1", 0.99),
            agreement("rna", "2", 0.99),
        ]
        found = summary.size_agreement(rows, seurat_made_graph=True)
        self.assertEqual((found["clusters"], found["seeds"]), (19, 2))
        self.assertAlmostEqual(found["meanAri"]["cytearc_wnn"], 0.45)
        self.assertAlmostEqual(found["meanAri"]["seurat_wnn"], 0.33)
        self.assertAlmostEqual(found["meanAriDifference"], 0.12)

    def test_each_cohort_averages_its_own_column_over_the_seeds_of_the_same_number(self):
        rows = [
            agreement("cytearc_wnn", "1", 0.4, combat=0.70, stephenson=0.30),
            agreement("cytearc_wnn", "2", 0.5, combat=0.80, stephenson=0.40),
            agreement("adt", "1", 0.2, combat=0.60, stephenson=0.10),
            agreement("adt", "2", 0.2, combat=0.62, stephenson=0.20),
            # Another cluster number and a graph outside the comparison stay out.
            agreement("adt", "1", 0.9, target="25", combat=0.99, stephenson=0.99),
            agreement("cytearc_snn", "1", 0.9, combat=0.99, stephenson=0.99),
        ]
        found = summary.size_agreement(rows, seurat_made_graph=False)
        cohorts = found["meanAriByCohort"]
        self.assertEqual(sorted(cohorts["combat"]), ["adt", "cytearc_wnn"])
        self.assertAlmostEqual(cohorts["combat"]["cytearc_wnn"], 0.75)
        self.assertAlmostEqual(cohorts["stephenson"]["cytearc_wnn"], 0.35)
        self.assertAlmostEqual(cohorts["combat"]["adt"], 0.61)
        self.assertAlmostEqual(cohorts["stephenson"]["adt"], 0.15)

    def test_without_seurats_graph_there_is_no_difference(self):
        rows = [agreement("cytearc_wnn", "1", 0.4), agreement("cytearc_wnn", "2", 0.5)]
        found = summary.size_agreement(rows, seurat_made_graph=False)
        self.assertEqual(list(found["meanAri"]), ["cytearc_wnn"])
        self.assertIsNone(found["meanAriDifference"])

    def test_rows_that_do_not_match_the_runs_are_an_error(self):
        rows = [agreement("cytearc_wnn", "1", 0.4), agreement("cytearc_wnn", "2", 0.5)]
        # Seurat made a graph, but the table scores none of it.
        with self.assertRaisesRegex(ValueError, "seurat_wnn has 0 rows"):
            summary.size_agreement(rows, seurat_made_graph=True)
        # Seurat scored with one seed fewer than CyteArc.
        with self.assertRaisesRegex(ValueError, "seurat_wnn has 1 rows"):
            summary.size_agreement(
                rows + [agreement("seurat_wnn", "1", 0.3)], seurat_made_graph=True
            )
        # No cluster number with several seeds.
        with self.assertRaisesRegex(ValueError, "not at one cluster number"):
            summary.size_agreement([agreement("cytearc_wnn", "1", 0.4)], seurat_made_graph=False)


class StoredRecordsTest(unittest.TestCase):
    """The COMBAT and Stephenson series as paragraph 56 reports it."""

    @classmethod
    def setUpClass(cls):
        cls.sizes = {size["cells"]: size for size in summary.summarise()["sizes"]}

    def test_cytearc_runs_41_and_748_times_faster_than_seurat_at_600000_cells(self):
        size = self.sizes[600_000]
        # 5.4 minutes against 3.8 hours for the first analysis, and 14.7 seconds against 3.1 hours to re-cluster.
        self.assertEqual(round(size["cytearc"]["initialSeconds"] / 60, 1), 5.4)
        self.assertEqual(round(size["seurat"]["initialSeconds"] / 3600, 1), 3.8)
        self.assertEqual(round(size["initialSpeedup"]), 41)
        self.assertEqual(round(size["cytearc"]["storedSeconds"], 1), 14.7)
        self.assertEqual(round(size["seurat"]["storedSeconds"] / 3600, 1), 3.1)
        self.assertEqual(round(size["storedSpeedup"]), 748)

    def test_seurat_finishes_its_first_analysis_only_up_to_600000_cells(self):
        finished = [
            cells for cells, size in self.sizes.items() if size["initialSpeedup"] is not None
        ]
        self.assertEqual(finished, [100_000, 300_000, 600_000])

    def test_cytearc_wnn_agrees_more_with_the_authors_by_007_to_008_ari(self):
        compared = [self.sizes[cells] for cells in (100_000, 300_000, 600_000)]
        self.assertEqual({size["clusters"] for size in compared}, {19})
        self.assertEqual(
            [round(size["meanAriDifference"], 2) for size in compared], [0.07, 0.08, 0.08]
        )
        cytearc_means = [round(size["meanAri"]["cytearc_wnn"], 2) for size in compared]
        seurat_means = [round(size["meanAri"]["seurat_wnn"], 2) for size in compared]
        self.assertEqual((min(cytearc_means), max(cytearc_means)), (0.43, 0.44))
        self.assertEqual((min(seurat_means), max(seurat_means)), (0.35, 0.36))

    def test_within_cohort_agreement_splits_combat_from_stephenson(self):
        # Paragraph 92: within COMBAT CyteArc WNN reached 0.77 to 0.78 (0.78 at 600,000 cells against 0.61 for Seurat
        # WNN and 0.60 for RNA alone); within Stephenson the protein graph scored 0.19 to 0.21 against 0.61 to 0.66
        # within COMBAT, WNN 0.33 to 0.37, RNA alone 0.58 to 0.62 and Seurat WNN 0.23 to 0.25.
        def span(cohort: str, graph: str) -> tuple[float, float]:
            values = [
                round(size["meanAriByCohort"][cohort][graph], 2)
                for size in self.sizes.values()
                if graph in size["meanAriByCohort"][cohort]
            ]
            return min(values), max(values)

        self.assertEqual(span("combat", "cytearc_wnn"), (0.77, 0.78))
        at_600000 = self.sizes[600_000]["meanAriByCohort"]["combat"]
        self.assertEqual(
            [round(at_600000[g], 2) for g in ("cytearc_wnn", "seurat_wnn", "rna")],
            [0.78, 0.61, 0.60],
        )
        self.assertEqual(span("stephenson", "adt"), (0.19, 0.21))
        self.assertEqual(span("combat", "adt"), (0.61, 0.66))
        self.assertEqual(span("stephenson", "cytearc_wnn"), (0.33, 0.37))
        self.assertEqual(span("stephenson", "rna"), (0.58, 0.62))
        self.assertEqual(span("stephenson", "seurat_wnn"), (0.23, 0.25))


if __name__ == "__main__":
    unittest.main()
