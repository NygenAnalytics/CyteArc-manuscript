import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from experiments.mapping import blood_store_annotation_summary, collect_records, held_out_summary
from experiments.mapping.label_accuracy import (
    classification_statistics,
    donor_bootstrap,
    donor_confusion_matrices,
)
from experiments.mapping.reference_ladder_summary import (
    disagreement_differences,
    rare_type_tiers,
    step_sums,
    stored_reference_speedups,
    summarize,
)

LABELS = ["A", "B", "unassigned"]


class LabelAccuracyTests(unittest.TestCase):
    def test_statistics_match_hand_counts(self) -> None:
        # Rows are author labels, columns transferred labels: 3 of 4 A cells and 2 of 2 B cells
        # are right, and one A cell is called B.
        matrix = np.array([[3, 1, 0], [0, 2, 0], [0, 0, 0]])
        statistics = classification_statistics(matrix, LABELS, {"A", "B"})
        self.assertAlmostEqual(statistics["accuracy"], 5 / 6)
        self.assertAlmostEqual(statistics["perType"]["A"]["recall"], 3 / 4)
        self.assertAlmostEqual(statistics["perType"]["B"]["precision"], 2 / 3)
        self.assertAlmostEqual(statistics["macroF1"], (6 / 7 + 4 / 5) / 2)
        self.assertNotIn("unassigned", statistics["perType"])

    def test_statistics_refuse_an_empty_or_misshaped_matrix(self) -> None:
        with self.assertRaises(ValueError):
            classification_statistics(np.zeros((3, 3)), LABELS, {"A"})
        with self.assertRaises(ValueError):
            classification_statistics(np.ones((2, 2)), LABELS, {"A"})

    def test_donor_matrices_follow_sorted_donors(self) -> None:
        cells = pd.DataFrame(
            {
                "donor_id": ["d2", "d1", "d2"],
                "truth": ["A", "B", "A"],
                "prediction": ["A", "unassigned", "B"],
            }
        )
        donors = donor_confusion_matrices(cells, LABELS)
        self.assertEqual([d["donor"] for d in donors], ["d1", "d2"])
        self.assertEqual(donors[0]["confusion"], [[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        self.assertEqual(donors[1]["confusion"], [[1, 1, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual([d["cells"] for d in donors], [1, 2])

    def test_bootstrap_needs_three_donors(self) -> None:
        matrices = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 0]]] * 2)
        interval = donor_bootstrap(matrices, LABELS, {"A", "B"})
        self.assertEqual(interval, {"estimable": False, "donors": 2, "reason": mock.ANY})

    def test_bootstrap_of_identical_donors_has_no_spread(self) -> None:
        matrices = np.array([[[3, 1, 0], [0, 2, 0], [0, 0, 0]]] * 4)
        interval = donor_bootstrap(matrices, LABELS, {"A", "B"})
        self.assertTrue(interval["estimable"])
        self.assertEqual(interval["resamples"], 1000)
        np.testing.assert_allclose(interval["accuracy"], [5 / 6, 5 / 6])

    def test_bootstrap_interval_contains_the_pooled_accuracy(self) -> None:
        rng = np.random.default_rng(1)
        matrices = rng.integers(0, 50, size=(12, 3, 3))
        pooled = classification_statistics(matrices.sum(axis=0), LABELS, {"A", "B", "unassigned"})
        interval = donor_bootstrap(matrices, LABELS, {"A", "B", "unassigned"})
        low, high = interval["accuracy"]
        self.assertLess(low, pooled["accuracy"])
        self.assertGreater(high, pooled["accuracy"])
        self.assertEqual(interval, donor_bootstrap(matrices, LABELS, {"A", "B", "unassigned"}))


class ReferenceLadderSummaryTests(unittest.TestCase):
    def test_step_sums_use_registration_or_the_saved_reference(self) -> None:
        cytearc = {
            "referencePreparation": 10.0,
            "referenceRegistration": 2.0,
            "queryAnnotation": 5.0,
        }
        self.assertEqual(
            step_sums(cytearc),
            {"preparedReferenceAndQuerySeconds": 7.0, "freshReferenceAndQuerySeconds": 17.0},
        )
        comparator = {
            "referencePreparation": 10.0,
            "referencePersistence": 1.0,
            "preparedReferenceLoad": 3.0,
            "queryAnnotation": 5.0,
        }
        self.assertEqual(
            step_sums(comparator),
            {"preparedReferenceAndQuerySeconds": 8.0, "freshReferenceAndQuerySeconds": 19.0},
        )

    def test_step_sums_refuse_ambiguous_access(self) -> None:
        with self.assertRaises(ValueError):
            step_sums(
                {"referenceRegistration": 1.0, "preparedReferenceLoad": 1.0, "queryAnnotation": 1.0}
            )
        with self.assertRaises(ValueError):
            step_sums({"referencePreparation": 1.0, "queryAnnotation": 1.0})

    def test_rare_type_tiers_average_the_types_present(self) -> None:
        statistics = {
            "perType": {
                "A": {"recall": 0.5, "f1": 0.4},
                "B": {"recall": 1.0, "f1": 0.8},
            }
        }
        tiers = rare_type_tiers(
            statistics, {"belowOnePercent": ["A", "B", "C"], "belowOneTenthPercent": ["C"]}
        )
        self.assertEqual(tiers["belowOnePercent"]["types"], ["A", "B"])
        self.assertAlmostEqual(tiers["belowOnePercent"]["meanRecall"], 0.75)
        self.assertAlmostEqual(tiers["belowOnePercent"]["meanF1"], 0.6)
        self.assertIsNone(tiers["belowOneTenthPercent"]["meanRecall"])

    def ladder(self, query_cells: int) -> dict:
        row = {"referencePart": "reference_10", "referenceCells": 10, "method": "unintegrated"}
        return {
            "queryCells": query_cells,
            "confusionLabels": LABELS,
            "rareTypes": {"belowOnePercent": ["B"], "belowOneTenthPercent": []},
            "rows": [
                {
                    **row,
                    "outcome": "completed",
                    "stepSeconds": {
                        "referencePreparation": 4.0,
                        "referenceRegistration": 1.0,
                        "queryAnnotation": 2.0,
                    },
                    "donorConfusion": [
                        {"donor": "s::d1", "confusion": [[2, 0, 0], [1, 1, 0], [0, 0, 0]]},
                        {"donor": "s::d2", "confusion": [[1, 0, 0], [0, 0, 1], [0, 0, 0]]},
                    ],
                },
                {**row, "method": "celltypist", "outcome": "not attempted"},
            ],
        }

    def test_summary_pools_donors(self) -> None:
        summary = summarize(self.ladder(query_cells=6))
        scored, skipped = summary["rows"]
        # Three of three A cells and one of three B cells are right.
        self.assertAlmostEqual(scored["accuracy"], 4 / 6)
        self.assertAlmostEqual(scored["perType"]["B"]["recall"], 1 / 3)
        self.assertEqual(scored["preparedReferenceAndQuerySeconds"], 3.0)
        self.assertFalse(scored["donorBootstrap95CI"]["estimable"])
        self.assertAlmostEqual(scored["rareTypeTiers"]["belowOnePercent"]["meanRecall"], 1 / 3)
        self.assertEqual(skipped["outcome"], "not attempted")
        self.assertNotIn("accuracy", skipped)
        # Two of six cells disagree; the unintegrated run is its own baseline.
        self.assertAlmostEqual(scored["disagreement"], 2 / 6)
        self.assertEqual(scored["disagreementPointsOverBaseline"], 0.0)
        self.assertNotIn("disagreementPointsOverBaseline", skipped)

    def test_disagreement_differences_compare_each_run_with_the_unintegrated_run_of_its_reference(
        self,
    ) -> None:
        rows = disagreement_differences(
            [
                {"referencePart": "r1", "method": "unintegrated", "disagreement": 0.05},
                {"referencePart": "r1", "method": "seurat_map", "disagreement": 0.04},
                {"referencePart": "r1", "method": "celltypist_hvg", "disagreement": 0.08},
                {"referencePart": "r2", "method": "unintegrated", "disagreement": 0.02},
                {"referencePart": "r2", "method": "symphony", "disagreement": 0.025},
            ]
        )
        # Seurat 1 point below on r1, CellTypist 3 above; Symphony half a point above on r2, not against r1's baseline.
        self.assertEqual(
            [round(row["disagreementPointsOverBaseline"], 9) for row in rows],
            [0.0, -1.0, 3.0, 0.0, 0.5],
        )

    def test_a_reference_without_an_unintegrated_run_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "r2 has no completed unintegrated run"):
            disagreement_differences(
                [{"referencePart": "r2", "method": "seurat_map", "disagreement": 0.04}]
            )

    def test_summary_refuses_a_run_that_misses_query_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "query"):
            summarize(self.ladder(query_cells=7))

    def test_stored_reference_speedups_divide_each_comparator_by_cytearc_on_its_reference(
        self,
    ) -> None:
        def run(part: str, method: str, seconds: float | None) -> dict:
            row = {"referencePart": part, "method": method}
            return row if seconds is None else row | {"preparedReferenceAndQuerySeconds": seconds}

        rows = stored_reference_speedups(
            [
                run("r1", "unintegrated", 10.0),
                run("r1", "integrated", 20.0),
                run("r1", "seurat_map", 200.0),
                run("r1", "symphony", 300.0),
                run("r2", "unintegrated", 30.0),
                run("r2", "integrated", 40.0),
                run("r2", "seurat_map", 600.0),
                # A run that did not complete has no time to compare.
                run("r2", "symphony", None),
            ]
        )
        speedups = [row.get("cytearcSpeedupFromStoredReference") for row in rows]
        # Seurat took 20 and 10 times as long as CyteArc's two runs on r1, and 20 and 15 times on r2, not r1's.
        self.assertEqual(
            speedups,
            [
                None,
                None,
                {"unintegrated": 20.0, "integrated": 10.0},
                {"unintegrated": 30.0, "integrated": 15.0},
                None,
                None,
                {"unintegrated": 20.0, "integrated": 15.0},
                None,
            ],
        )

    def test_a_comparator_without_both_cytearc_runs_on_its_reference_is_an_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"r3 has no completed CyteArc \['integrated'\] run"
        ):
            stored_reference_speedups(
                [
                    {
                        "referencePart": "r3",
                        "method": "unintegrated",
                        "preparedReferenceAndQuerySeconds": 1.0,
                    },
                    {
                        "referencePart": "r3",
                        "method": "seurat_map",
                        "preparedReferenceAndQuerySeconds": 9.0,
                    },
                ]
            )


def job_record(function: str, result: dict, status: str = "completed") -> dict:
    return {"module": "m", "function": function, "status": status, "result": result}


def annotation_result() -> dict:
    io_counts = {"countReadBytes": 0, "transferredBytes": 100, "gets": 3}
    resources = {
        "cgroupMemoryCurrentPeakBytes": 70,
        "memoryMaxBytes": 1000,
        "processTreeRssPeakBytes": 9,
    }
    return {
        "settings": {"study": "perez", "outputPrefix": "s3://bucket/x", "workDirectory": "/tmp/x"},
        "provenance": {"cytearcVersion": "1.3.0"},
        "mappingRule": {"savedNeighbors": 30, "voteThreshold": 0.0},
        "eligibility": "Primary blood cells",
        "sampleKey": ["donor_id"],
        "referenceOpening": {
            "seconds": 5.0,
            "io": io_counts,
            "registration": {
                "cells": 40,
                "featureCount": 20,
                "storeUri": "s3://bucket/ref",
                "vocabulary": ["B cells", "T cells"],
            },
        },
        "referenceReadsDuringMapping": {**io_counts, "transferredBytes": 160},
        "resources": resources,
        "stores": [
            {
                "store": "perez_store",
                "storeCells": 12,
                "eligibleCells": 4,
                "donors": 3,
                "referenceFeaturesMissing": ["G1"],
                "mapping": {
                    "elapsedSeconds": 2.0,
                    "resources": {"cgroupMemoryCurrentPeakBytes": 50},
                },
                "outputs": {"cells.npz": "s3://bucket/perez/cells.npz"},
            }
        ],
    }


def store_labels() -> dict[str, np.ndarray]:
    """Four cells of one store, worked by hand: transferred B, T, unassigned and T against author B, B, a label outside
    the reference's types and T, so cells 1, 2 and 4 are typed and cells 1 and 4 agree."""
    return {
        "mapped_codes": np.array([0, 1, 2, 1], dtype=np.int8),
        "mapped_names": np.array(["B cells", "T cells", "unassigned"]),
        "author_codes": np.array([0, 0, 1, 2]),
        "author_type_names": np.array(["B cells", "untyped", "T cells"]),
    }


class CollectRecordsTests(unittest.TestCase):
    def test_annotation_record_keeps_measurements_only(self) -> None:
        record = collect_records.blood_store_annotation_record(
            job_record("map_and_aggregate", annotation_result()), {"perez_store": store_labels()}
        )
        self.assertEqual(
            record,
            {
                "study": "perez",
                "mappingRule": {"savedNeighbors": 30, "voteThreshold": 0.0},
                "eligibility": "Primary blood cells",
                "referenceOpening": {
                    "seconds": 5.0,
                    "countReadBytes": 0,
                    "transferredBytes": 100,
                    "referenceCells": 40,
                    "referenceFeatures": 20,
                },
                "referenceReadsIncludingOpening": {"countReadBytes": 0, "transferredBytes": 160},
                "peakContainerBytes": 70,
                "containerMemoryBytes": 1000,
                "stores": [
                    {
                        "store": "perez_store",
                        "storeCells": 12,
                        "eligibleCells": 4,
                        "donors": 3,
                        "referenceFeaturesMissing": ["G1"],
                        "mappingPeakContainerBytes": 50,
                        "mappingSeconds": 2.0,
                        "typedCells": 3,
                        "agreeingCells": 2,
                    }
                ],
            },
        )
        self.assertNotIn("s3://", json.dumps(record))

    def test_label_agreement_counts_typed_and_agreeing_cells_by_hand(self) -> None:
        vocabulary = ["B cells", "T cells"]
        self.assertEqual(
            collect_records.label_agreement(store_labels(), vocabulary),
            {"labelledCells": 4, "typedCells": 3, "agreeingCells": 2},
        )
        # The same cells with the vocabulary in another order: the counts follow the names, not their positions.
        self.assertEqual(
            collect_records.label_agreement(store_labels(), ["T cells", "B cells"]),
            {"labelledCells": 4, "typedCells": 3, "agreeingCells": 2},
        )
        # An author type the reference lacks leaves its cells untyped.
        self.assertEqual(
            collect_records.label_agreement(store_labels(), ["B cells", "T cells", "Platelets"])[
                "typedCells"
            ],
            3,
        )

    def test_label_agreement_refuses_labels_outside_the_reference_or_unaligned(self) -> None:
        labels = store_labels()
        labels["mapped_names"] = np.array(["B cells", "Mast cells", "unassigned"])
        with self.assertRaisesRegex(
            ValueError, "outside the reference's types: \\['Mast cells'\\]"
        ):
            collect_records.label_agreement(labels, ["B cells", "T cells"])
        labels = store_labels()
        labels["author_codes"] = labels["author_codes"][:3]
        with self.assertRaisesRegex(ValueError, "4 transferred labels against 3 author labels"):
            collect_records.label_agreement(labels, ["B cells", "T cells"])

    def test_a_store_without_labels_or_with_other_cells_is_refused(self) -> None:
        job = job_record("map_and_aggregate", annotation_result())
        with self.assertRaisesRegex(ValueError, "No per-cell labels for the store perez_store"):
            collect_records.blood_store_annotation_record(job, {})
        labels = store_labels()
        for name in ("mapped_codes", "author_codes"):
            labels[name] = labels[name][:3]
        with self.assertRaisesRegex(ValueError, "3 labelled cells against 4 eligible"):
            collect_records.blood_store_annotation_record(job, {"perez_store": labels})

    def test_labels_are_read_beside_a_local_job_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "perez" / "record.json"
            (job.parent / "perez_store").mkdir(parents=True)
            np.savez(job.parent / "perez_store" / "cells.npz", rows=np.arange(4), **store_labels())
            labels = collect_records.read_store_labels(str(job), {"store": "perez_store"})
        self.assertEqual(sorted(labels), sorted(collect_records.LABEL_ARRAYS))
        self.assertEqual(
            collect_records.label_agreement(labels, ["B cells", "T cells"])["agreeingCells"], 2
        )

    def test_collection_refuses_other_or_unfinished_jobs(self) -> None:
        with self.assertRaisesRegex(ValueError, "completed map_and_aggregate"):
            collect_records.blood_store_annotation_record(
                job_record("map_and_aggregate", annotation_result(), status="failed"),
                {"perez_store": store_labels()},
            )
        with self.assertRaisesRegex(ValueError, "completed evaluate_ladder"):
            collect_records.reference_ladder_record(job_record("map_and_aggregate", {}))

    def test_ladder_record_must_hold_exactly_its_fields(self) -> None:
        fields = {name: name for name in collect_records.REFERENCE_LADDER_FIELDS}
        self.assertEqual(
            collect_records.reference_ladder_record(job_record("evaluate_ladder", fields)), fields
        )
        with self.assertRaisesRegex(ValueError, "extra \\['callId'\\]"):
            collect_records.reference_ladder_record(
                job_record("evaluate_ladder", {**fields, "callId": "x"})
            )
        fields.pop("rows")
        with self.assertRaisesRegex(ValueError, "missing \\['rows'\\]"):
            collect_records.reference_ladder_record(job_record("evaluate_ladder", fields))

    def test_records_keep_number_lists_on_one_line(self) -> None:
        record = {"confusion": [[1, 20, 300], [4, 5, 6]], "labels": ["a", "b"], "x": 1.5e-07}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(collect_records, "ROOT", root), redirect_stdout(io.StringIO()):
                collect_records.write_record(root / "r.json", record)
            text = (root / "r.json").read_text()
        self.assertIn("[1, 20, 300]", text)
        self.assertIn('"a",\n', text)
        self.assertEqual(json.loads(text), record)


class BloodStoreAnnotationSummaryTests(unittest.TestCase):
    def record(self, study: str, stores: list[tuple[str, int]], peak: int) -> dict:
        # Each store maps in 100 seconds per eligible cell, types all but one of its cells and agrees on all but two.
        return {
            "study": study,
            "referenceOpening": {"seconds": 10.0 + peak, "transferredBytes": 100},
            "referenceReadsIncludingOpening": {"transferredBytes": 100 + 50 * len(stores)},
            "peakContainerBytes": peak,
            "stores": [
                {
                    "store": name,
                    "eligibleCells": cells,
                    "mappingSeconds": 100.0 * cells,
                    "typedCells": cells - 1,
                    "agreeingCells": cells - 3,
                }
                for name, cells in stores
            ],
        }

    def test_totals_cover_every_store(self) -> None:
        summary = blood_store_annotation_summary.summarize(
            [self.record("a", [("s1", 5)], 7), self.record("b", [("s2", 3), ("s3", 4)], 9)]
        )
        self.assertEqual((summary["jobs"], summary["stores"], summary["eligibleCells"]), (2, 3, 12))
        self.assertEqual(summary["peakContainerBytes"], {"lowest": 7, "highest": 9})
        self.assertEqual(
            [row["referenceReadBytesAfterOpening"] for row in summary["studies"]], [50, 100]
        )

    def test_mapping_seconds_span_the_stores_and_disagreement_pools_typed_cells(self) -> None:
        summary = blood_store_annotation_summary.summarize(
            [self.record("a", [("s1", 5)], 7), self.record("b", [("s2", 3), ("s3", 4)], 9)]
        )
        self.assertEqual(summary["mappingSeconds"], {"lowest": 300.0, "highest": 500.0})
        # Typed 4 + 2 + 3 = 9 cells, agreeing 2 + 0 + 1 = 3: pooled over cells, not averaged over stores.
        self.assertEqual((summary["typedCells"], summary["agreeingCells"]), (9, 3))
        self.assertAlmostEqual(summary["agreement"], 3 / 9)
        self.assertAlmostEqual(summary["disagreement"], 6 / 9)
        self.assertEqual(
            [(row["typedCells"], row["agreeingCells"]) for row in summary["studies"]],
            [(4, 2), (5, 1)],
        )
        # Each job's own disagreement and share of the typed cells: 2 of 4 and 4 of 5 disagree; 4 and 5 of 9 typed.
        self.assertEqual([row["disagreement"] for row in summary["studies"]], [0.5, 0.8])
        self.assertEqual([row["typedShare"] for row in summary["studies"]], [4 / 9, 5 / 9])
        self.assertEqual([row["agreement"] for row in summary["studies"]], [0.5, 0.2])
        # Job a annotated a single store and job b two, so each pool holds one job.
        self.assertEqual(
            summary["singleStoreStudies"],
            {"studies": ["a"], "typedCells": 4, "typedShare": 4 / 9, "agreement": 0.5},
        )
        self.assertEqual(summary["multiStoreStudies"]["studies"], ["b"])
        self.assertAlmostEqual(summary["multiStoreStudies"]["agreement"], 0.2)
        many = blood_store_annotation_summary.summarize(
            [self.record("b", [("s2", 3), ("s3", 4)], 9)]
        )
        self.assertIsNone(many["singleStoreStudies"]["agreement"])
        self.assertEqual(many["singleStoreStudies"]["typedShare"], 0.0)

    def test_a_store_counts_once(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than one job"):
            blood_store_annotation_summary.summarize(
                [self.record("a", [("s1", 5)], 7), self.record("b", [("s1", 5)], 9)]
            )
        with self.assertRaises(ValueError):
            blood_store_annotation_summary.summarize([])


class HeldOutSummaryTests(unittest.TestCase):
    def run_record(self, tissue: str, method: str, accuracy: float, cells: list[int]) -> dict:
        study = held_out_summary.HELD_OUT_STUDIES[tissue]
        record = {
            "tissue": tissue,
            "method": method,
            "evaluation": {
                "studies": {
                    study: {
                        "accuracy": accuracy,
                        "donorConfusionMatrices": [
                            {"donor": f"d{i}", "cells": n} for i, n in enumerate(cells)
                        ],
                    }
                }
            },
        }
        if method == "unintegrated":
            record["inputs"] = {
                "parts": {
                    "reference": {"cells": 100, "donors": [{"donor_id": "r1"}, {"donor_id": "r2"}]},
                    "query": {
                        "cells": sum(cells),
                        "donors": [{"donor_id": f"d{i}"} for i in range(len(cells))],
                    },
                }
            }
        return record

    def test_each_method_disagrees_on_the_share_of_query_cells_it_labels_otherwise(self) -> None:
        tissues = held_out_summary.held_out(
            [
                self.run_record("kidney", "unintegrated", 0.96, [30, 20]),
                self.run_record("kidney", "seurat_map", 0.75, [30, 20]),
                self.run_record("lung", "unintegrated", 0.9, [5, 5, 10]),
                # A blood run is not held out.
                {"tissue": "blood", "method": "unintegrated"},
            ]
        )
        self.assertEqual(sorted(tissues), ["kidney", "lung"])
        kidney = tissues["kidney"]
        self.assertEqual(kidney["study"], "Reck_2025")
        self.assertEqual(kidney["query"], {"cells": 50, "donors": 2})
        self.assertEqual(kidney["reference"], {"cells": 100, "donors": 2})
        self.assertAlmostEqual(kidney["methods"]["unintegrated"]["disagreement"], 0.04)
        self.assertAlmostEqual(kidney["methods"]["seurat_map"]["disagreement"], 0.25)
        # Seurat disagrees on 21 points more of the query than CyteArc; CyteArc's own gap is zero.
        self.assertAlmostEqual(
            kidney["methods"]["seurat_map"]["disagreementPointsOverCytearc"], 21.0
        )
        self.assertEqual(kidney["methods"]["unintegrated"]["disagreementPointsOverCytearc"], 0.0)
        self.assertEqual(tissues["lung"]["query"]["donors"], 3)

    def test_a_method_scored_on_other_cells_than_the_query_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "seurat_map on Reck_2025 was scored on 49 cells"):
            held_out_summary.held_out(
                [
                    self.run_record("kidney", "unintegrated", 0.9, [30, 20]),
                    self.run_record("kidney", "seurat_map", 0.9, [30, 19]),
                    self.run_record("lung", "unintegrated", 0.9, [5]),
                ]
            )

    def test_a_tissue_without_its_cytearc_run_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no unintegrated run on lung"):
            held_out_summary.held_out([self.run_record("kidney", "unintegrated", 0.9, [5])])

    def write_comparator_records(self, sizes: dict) -> Path:
        """Comparator records under a temporary cross-study folder, ``sizes[(tissue, method)]`` holding the record's
        fields."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for (tissue, method), fields in sizes.items():
            (root / tissue).mkdir(exist_ok=True)
            (root / tissue / f"{method}.json").write_text(json.dumps({"method": method, **fields}))
        return root

    def test_a_saved_reference_is_its_rds_with_its_neighbor_index(self) -> None:
        sizes = {
            (tissue, method): {
                "referenceRdsBytes": 1_000 * (i + 1) + j,
                "referenceIndexBytes": 10 * (i + 1),
            }
            for i, tissue in enumerate(held_out_summary.REFERENCE_TISSUES)
            for j, method in enumerate(held_out_summary.SAVED_REFERENCE_METHODS)
        }
        saved = held_out_summary.saved_reference_bytes(self.write_comparator_records(sizes))
        self.assertEqual(sorted(saved), ["blood", "kidney", "lung"])
        # Kidney is the second tissue: 2,000 RDS bytes and 20 index bytes for Seurat, 2,001 and 20 for R Symphony.
        self.assertEqual(saved["kidney"], {"seurat_map": 2_020, "symphony": 2_021})
        self.assertEqual(saved["lung"]["seurat_map"], 3_030)

    def test_a_comparator_record_without_its_index_size_is_an_error(self) -> None:
        sizes = {
            (tissue, method): {"referenceRdsBytes": 100, "referenceIndexBytes": 0}
            for tissue in held_out_summary.REFERENCE_TISSUES
            for method in held_out_summary.SAVED_REFERENCE_METHODS
        }
        del sizes[("lung", "symphony")]["referenceIndexBytes"]
        with self.assertRaisesRegex(
            ValueError, r"lung/symphony.json records no referenceIndexBytes"
        ):
            held_out_summary.saved_reference_bytes(self.write_comparator_records(sizes))


class StoredRecordsTests(unittest.TestCase):
    def test_stored_records_rebuild_the_reported_totals(self) -> None:
        # The blood store annotation and the reference ladder as the paper reports them.
        records = [
            json.loads(path.read_text())
            for path in sorted(blood_store_annotation_summary.RECORDS.glob("*.json"))
        ]
        summary = blood_store_annotation_summary.summarize(records)
        self.assertEqual((summary["jobs"], summary["stores"]), (8, 14))
        self.assertEqual(summary["eligibleCells"], 21_262_774)
        # Paragraph 54: each store mapped in 6.9 to 20.7 minutes, with 1.8% disagreement (98.2% agreement) over 21.1
        # million typed cells; the counts were taken from the jobs' per-cell labels.
        self.assertEqual(
            (summary["typedCells"], summary["agreeingCells"]), (21_125_905, 20_741_231)
        )
        self.assertAlmostEqual(summary["mappingSeconds"]["lowest"], 414.408, places=3)
        self.assertAlmostEqual(summary["mappingSeconds"]["highest"], 1239.698, places=3)
        self.assertAlmostEqual(summary["disagreement"], 0.018209, places=6)
        self.assertAlmostEqual(summary["agreement"], 0.981791, places=6)
        # Paragraph 287: the two four-store Sound Life jobs held 64.9% of typed cells at 0.4% and 0.8% disagreement,
        # Ren had the highest at 7.5%, and the six single-store studies pooled to 95.9% agreement.
        studies = {row["study"]: row for row in summary["studies"]}
        multi = summary["multiStoreStudies"]
        self.assertEqual(sorted(multi["studies"]), ["sound_life", "sound_life_cmv_positive"])
        self.assertEqual(round(multi["typedShare"] * 100, 1), 64.9)
        self.assertEqual(
            [
                round(studies[s]["agreement"] * 100, 1)
                for s in ("sound_life", "sound_life_cmv_positive")
            ],
            [99.6, 99.2],
        )
        self.assertEqual(
            [
                round(studies[s]["disagreement"] * 100, 1)
                for s in ("sound_life", "sound_life_cmv_positive", "ren")
            ],
            [0.4, 0.8, 7.5],
        )
        self.assertEqual(max(studies.values(), key=lambda row: row["disagreement"])["study"], "ren")
        self.assertEqual(len(summary["singleStoreStudies"]["studies"]), 6)
        self.assertEqual(round(summary["singleStoreStudies"]["agreement"] * 100, 1), 95.9)
        # Paragraph 265: on held-out kidney cells CyteArc trailed Seurat by 0.3 percentage points (3.7% against 3.4%).
        kidney = held_out_summary.held_out(
            json.loads(held_out_summary.EVALUATION.read_text())["result"]["mapping"]
        )["kidney"]
        self.assertEqual(
            round(kidney["methods"]["seurat_map"]["disagreementPointsOverCytearc"], 1), -0.3
        )
        ladder = summarize(json.loads(collect_records.REFERENCE_LADDER.read_text()))
        scored = [row for row in ladder["rows"] if "accuracy" in row]
        self.assertEqual(len(ladder["rows"]), 24)
        self.assertEqual(len(scored), 17)
        self.assertAlmostEqual(scored[0]["accuracy"], 0.9497, places=4)
        # Paragraph 54: CyteArc 4.7% to 5.0% disagreement, R Symphony 4.4% to 4.5% and Seurat 3.9% to 4.0% within 0.5 to
        # 1.1 points of it, and variable-gene CellTypist 7.9% to 8.2%, 2.9 to 3.4 points above it.
        by_method: dict[str, list[dict]] = {}
        for row in scored:
            by_method.setdefault(row["method"], []).append(row)

        def span(method: str, key: str, scale: float = 100, digits: int = 1) -> tuple[float, float]:
            values = [abs(row[key]) * scale for row in by_method[method]]
            return round(min(values), digits), round(max(values), digits)

        self.assertEqual(span("unintegrated", "disagreement"), (4.7, 5.0))
        self.assertEqual(span("symphony", "disagreement"), (4.4, 4.5))
        self.assertEqual(span("seurat_map", "disagreement"), (3.9, 4.0))
        self.assertEqual(span("celltypist_hvg", "disagreement"), (7.9, 8.2))
        self.assertEqual(span("symphony", "disagreementPointsOverBaseline", scale=1), (0.5, 0.5))
        self.assertEqual(span("seurat_map", "disagreementPointsOverBaseline", scale=1), (1.0, 1.0))
        self.assertEqual(
            span("celltypist_hvg", "disagreementPointsOverBaseline", scale=1), (2.9, 3.4)
        )

    def test_cytearc_annotates_from_stored_references_13_to_22_and_29_to_79_times_faster(
        self,
    ) -> None:
        # Paragraph 54: against Seurat and R Symphony from their saved references at 0.50 and 1.50 million cells,
        # over both CyteArc runs.
        ladder = summarize(json.loads(collect_records.REFERENCE_LADDER.read_text()))
        speedups: dict[str, list[float]] = {}
        for row in ladder["rows"]:
            for value in row.get("cytearcSpeedupFromStoredReference", {}).values():
                speedups.setdefault(row["method"], []).append(value)
        self.assertEqual(len(speedups["seurat_map"]), 4)
        self.assertEqual(round(min(speedups["seurat_map"])), 13)
        self.assertEqual(round(max(speedups["seurat_map"])), 22)
        self.assertEqual(round(min(speedups["symphony"])), 29)
        self.assertEqual(round(max(speedups["symphony"])), 79)
        # Seurat at 0.50 million cells over the unintegrated run there, from the record's step seconds: loading the
        # saved reference and annotating the query, against registering the reference and annotating the query.
        (seurat,) = [
            row
            for row in ladder["rows"]
            if (row["referencePart"], row["method"]) == ("reference_500000", "seurat_map")
        ]
        self.assertAlmostEqual(
            seurat["cytearcSpeedupFromStoredReference"]["unintegrated"],
            (26.898990154266357 + 378.69981026649475) / (2.757164388999996 + 28.245446205000007),
            places=9,
        )

    def test_saved_reference_sizes_as_note_87_reports_them(self) -> None:
        # Note 87: Seurat's saved references, with their neighbor index, took 22.4 to 35.8 GB and R Symphony's 0.56 to
        # 0.78 GB, over the blood, kidney and lung references.
        saved = held_out_summary.saved_reference_bytes()
        seurat = [
            saved[tissue]["seurat_map"] / 1e9 for tissue in held_out_summary.REFERENCE_TISSUES
        ]
        symphony = [
            saved[tissue]["symphony"] / 1e9 for tissue in held_out_summary.REFERENCE_TISSUES
        ]
        self.assertEqual((round(min(seurat), 1), round(max(seurat), 1)), (22.4, 35.8))
        self.assertEqual((round(min(symphony), 2), round(max(symphony), 2)), (0.56, 0.78))
        # The kidney reference from its record's two sizes: 22,092,471,774 RDS bytes and 292,251,488 index bytes.
        self.assertEqual(saved["kidney"]["seurat_map"], 22_092_471_774 + 292_251_488)

    def test_held_out_disagreement_as_paragraph_54_reports_it(self) -> None:
        evaluation = json.loads(held_out_summary.EVALUATION.read_text())["result"]["mapping"]
        tissues = held_out_summary.held_out(evaluation)
        percent = {
            (tissue, method): round(100 * values["disagreement"], 1)
            for tissue, entry in tissues.items()
            for method, values in entry["methods"].items()
        }
        # Kidney: CyteArc 3.7%, Seurat 3.4%, R Symphony 5.6%, CellTypist on all genes 7.1%; lung: CyteArc 7.1%,
        # Seurat 7.4%, CellTypist 7.9%, R Symphony 9.9%; lung cells from three donors.
        self.assertEqual(
            [
                percent[("kidney", m)]
                for m in ("unintegrated", "seurat_map", "symphony", "celltypist")
            ],
            [3.7, 3.4, 5.6, 7.1],
        )
        self.assertEqual(
            [
                percent[("lung", m)]
                for m in ("unintegrated", "seurat_map", "celltypist", "symphony")
            ],
            [7.1, 7.4, 7.9, 9.9],
        )
        self.assertEqual(tissues["lung"]["query"]["donors"], 3)
        # The kidney query's CyteArc disagreement recounted from its donors' confusion matrices: author labels in rows,
        # transferred labels in columns, so the diagonal holds the cells labelled alike.
        (run,) = [r for r in evaluation if (r["tissue"], r["method"]) == ("kidney", "unintegrated")]
        donors = run["evaluation"]["studies"]["Reck_2025"]["donorConfusionMatrices"]
        alike = sum(np.trace(np.asarray(donor["confusion"])) for donor in donors)
        cells = sum(np.asarray(donor["confusion"]).sum() for donor in donors)
        self.assertEqual(cells, 46_957)
        self.assertAlmostEqual(
            tissues["kidney"]["methods"]["unintegrated"]["disagreement"],
            1 - alike / cells,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()


# The jobs' raw records and per-cell labels, kept outside the repository on the machine that collected them.
BLOOD_STORE_RAW = collect_records.ROOT / ".scratch" / "runs" / "results" / "blood_replication"


@unittest.skipUnless(
    BLOOD_STORE_RAW.is_dir(), "the blood store annotation jobs' raw outputs are not on this machine"
)
class BloodStoreAnnotationRawOutputsTests(unittest.TestCase):
    """The promoted records' mapping seconds and label counts, recounted from the jobs' raw outputs by comparing label
    names cell by cell rather than through the collector's codes."""

    def test_records_hold_the_raw_mapping_seconds_and_label_counts(self) -> None:
        records = sorted(blood_store_annotation_summary.RECORDS.glob("*.json"))
        self.assertEqual(len(records), 8)
        for path in records:
            record = json.loads(path.read_text())
            raw = json.loads((BLOOD_STORE_RAW / path.stem / "record.json").read_text())["result"]
            vocabulary = raw["referenceOpening"]["registration"]["vocabulary"]
            raw_stores = {store["store"]: store for store in raw["stores"]}
            self.assertEqual([s["store"] for s in record["stores"]], list(raw_stores))
            for store in record["stores"]:
                self.assertEqual(
                    store["mappingSeconds"], raw_stores[store["store"]]["mapping"]["elapsedSeconds"]
                )
                with np.load(BLOOD_STORE_RAW / path.stem / store["store"] / "cells.npz") as cells:
                    mapped = cells["mapped_names"][cells["mapped_codes"].astype(np.int64)]
                    author = cells["author_type_names"][cells["author_codes"].astype(np.int64)]
                typed = np.isin(author, vocabulary)
                self.assertEqual(mapped.size, store["eligibleCells"], store["store"])
                self.assertEqual(int(typed.sum()), store["typedCells"], store["store"])
                self.assertEqual(
                    int((typed & (mapped == author)).sum()), store["agreeingCells"], store["store"]
                )
