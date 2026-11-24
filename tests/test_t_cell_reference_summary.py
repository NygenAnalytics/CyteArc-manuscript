import unittest

import numpy as np
from experiments.integration import t_cell_reference_summary as summary


def authors(*rows) -> list[dict]:
    """Rows of a clustering run's authors table, with its columns as the job writes them."""
    return [
        {"graph": g, "resolution": r, "cluster": c, "authorLabel": a, "cells": str(n)}
        for g, r, c, a, n in rows
    ]


class SubtypeTermTest(unittest.TestCase):
    def test_both_studies_labels_share_one_term(self):
        cases = {
            "CD4.TCM": "CD4 memory",
            "CD4 TEM": "CD4 memory",
            "CD4.TREG": "Treg",
            "CD8 TEM": "CD8 effector",
            "GDT.VD2neg": "γδ T",
            "gdT": "γδ T",
        }
        for label, term in cases.items():
            with self.subTest(label=label):
                self.assertEqual(summary.vocabulary_term(label), term)

    def test_labels_outside_the_vocabulary_have_no_term(self):
        # Exact labels only: no case folding and no substring of a listed label.
        for label in ("dnT", "B naive", "", "cd4.tcm", "CD4"):
            with self.subTest(label=label):
                self.assertIsNone(summary.vocabulary_term(label))


class ClusterTermsTest(unittest.TestCase):
    def test_a_cluster_takes_the_term_most_of_its_cells_carry(self):
        rows = authors(
            ("wnn", "0.5", "1", "CD4.NAIVE", 60),
            ("wnn", "0.5", "1", "CD4.TCM", 30),
            # Two labels of one term, 35 cells together, outvote one larger label of another term.
            ("wnn", "0.5", "2", "CD4.TCM", 20),
            ("wnn", "0.5", "2", "CD4.TEFF", 15),
            ("wnn", "0.5", "2", "CD8.TEM", 30),
            # Labels outside the vocabulary count together as other, 55 cells, and outvote MAIT's 40.
            ("wnn", "0.5", "3", "B.NAIVE", 30),
            ("wnn", "0.5", "3", "NK_1", 25),
            ("wnn", "0.5", "3", "MAIT", 40),
        )
        self.assertEqual(
            summary.cluster_terms(rows, "wnn", 0.5),
            {"1": "CD4 naive", "2": "CD4 memory", "3": "other"},
        )

    def test_only_the_named_graph_and_resolution_count(self):
        rows = authors(
            ("wnn", "0.5", "1", "CD4.NAIVE", 10),
            ("rna", "0.5", "1", "MAIT", 1000),
            ("wnn", "0.8", "1", "MAIT", 1000),
            ("wnn", "0.8", "2", "MAIT", 1000),
        )
        self.assertEqual(summary.cluster_terms(rows, "wnn", 0.5), {"1": "CD4 naive"})


class AssignedQueryCellsTest(unittest.TestCase):
    def test_the_cells_that_received_a_subtype_leave_out_the_unassigned(self):
        result = {
            "queries": {
                "hao": {"cells": 30, "subtypeCells": {"MAIT": 10, "Treg": 15, "unassigned": 5}}
            }
        }
        self.assertEqual(
            summary.assigned_query_cells(result, "hao"),
            {"cells": 30, "unassigned": 5, "assignedCells": 25},
        )

    def test_subtypes_that_do_not_add_up_to_the_mapped_cells_are_an_error(self):
        result = {"queries": {"hao": {"cells": 31, "subtypeCells": {"MAIT": 10, "unassigned": 5}}}}
        with self.assertRaisesRegex(ValueError, "hold 15 cells, not the 31 it mapped"):
            summary.assigned_query_cells(result, "hao")


class StoredRecordsTest(unittest.TestCase):
    """The transfer of COMBAT's T-cell subtypes onto Hao's T cells as paragraph 56 reports it."""

    @classmethod
    def setUpClass(cls):
        cls.transfer = summary.subtype_transfer()

    def test_wnn_clusters_transfer_more_subtypes_than_rna_clusters(self):
        overall = self.transfer["overall"]
        rows = {row["subtype"]: row for row in self.transfer["rows"]}
        # 68,085 Hao T cells; 83.4% agreement from WNN clusters against 68.3% from RNA clusters; memory CD4 47.4% to
        # 87.5% and effector CD8 79.5% to 92.5%.
        self.assertEqual(overall["haoCells"], 68_085)
        self.assertEqual(
            (round(100 * overall["wnn"], 1), round(100 * overall["rna"], 1)), (83.4, 68.3)
        )
        memory, effector = rows["CD4 memory"], rows["CD8 effector"]
        self.assertEqual(
            (round(100 * memory["rna"], 1), round(100 * memory["wnn"], 1)), (47.4, 87.5)
        )
        self.assertEqual(
            (round(100 * effector["rna"], 1), round(100 * effector["wnn"], 1)), (79.5, 92.5)
        )

    def test_the_scored_cells_are_the_hao_t_cells_with_a_compared_label(self):
        # Hao's own labels of the compared terms, listed here rather than read from the vocabulary.
        compared = {
            "CD4 Naive",
            "CD4 TCM",
            "CD4 TEM",
            "Treg",
            "CD8 Naive",
            "CD8 TEM",
            "MAIT",
            "gdT",
        }
        with np.load(summary.T_CELL_SUBTYPES / "t_cell_subtypes.npz", allow_pickle=False) as saved:
            labels = [str(label) for label in saved["query_labels"]]
        self.assertEqual(
            sum(label in compared for label in labels), self.transfer["overall"]["haoCells"]
        )
        # Paragraph 288: the other 4,975 of Hao's 73,060 T cells carry labels outside the compared subtypes.
        self.assertEqual(len(labels), 73_060)
        self.assertEqual(
            sum(label not in compared for label in labels),
            self.transfer["overall"]["unscoredHaoCells"],
        )
        self.assertEqual(self.transfer["overall"]["unscoredHaoCells"], 4_975)
        memory = next(row for row in self.transfer["rows"] if row["subtype"] == "CD4 memory")
        self.assertEqual(
            sum(label in {"CD4 TCM", "CD4 TEM"} for label in labels), memory["haoCells"]
        )

    def test_the_named_reference_gave_73034_hao_t_cells_a_subtype(self):
        record = summary.load_json(summary.T_CELL_REFERENCE / "result.json")["result"]
        self.assertEqual(
            summary.assigned_query_cells(record, "hao"),
            {"cells": 73_060, "unassigned": 26, "assignedCells": 73_034},
        )


if __name__ == "__main__":
    unittest.main()
