import json
import unittest
from unittest import mock

import numpy as np
from experiments.mapping import reference_umap_rasters
from experiments.reuse import umap_rasters
from experiments.runtime.common import ROOT


class SquareGridTests(unittest.TestCase):
    # A layout one unit wide and half a unit high: the square grid's side is 1.06 (the width plus 3% on each side),
    # centred on the layout, so with 4 bins x = 0, 0.25, 0.75, 1 fall in columns 0, 1, 2, 3 and y = 0 and 0.5 in rows
    # 1 and 2: floor((y - (0.25 - 0.53)) / 1.06 * 4).
    UMAP = np.array([[0.0, 0.0], [0.25, 0.5], [0.75, 0.0], [1.0, 0.0], [1.0, 0.5]])

    def test_cells_fall_in_the_bins_worked_out_by_hand(self) -> None:
        for index in (
            umap_rasters.square_grid_index(self.UMAP, 4, 0.03),
            reference_umap_rasters.square_grid_index(self.UMAP, 4),
        ):
            np.testing.assert_array_equal(index, [[1, 0], [2, 1], [1, 2], [1, 3], [2, 3]])

    def test_counts_are_kept_apart_by_group_with_rows_as_umap_2(self) -> None:
        codes = np.array([0, 1, 1, 0, 0])
        with mock.patch.object(umap_rasters, "COMPOSITE_BINS", 4):
            raster = umap_rasters.blood_composite_raster(
                self.UMAP, codes, ["a", "b"], {"a": 3, "b": 2}
            )
        expected = np.zeros((2, 4, 4), dtype=np.uint32)
        expected[0, 1, 0] = expected[0, 1, 3] = expected[0, 2, 3] = 1
        expected[1, 2, 1] = expected[1, 1, 2] = 1
        np.testing.assert_array_equal(raster["studyCells"], expected)
        self.assertEqual(raster["studyCells"].dtype, np.uint32)
        self.assertEqual(raster["studies"].tolist(), ["a", "b"])
        counts = reference_umap_rasters.binned_types(self.UMAP, codes, 2, 4)
        np.testing.assert_array_equal(counts, expected)


class BloodCompositeTests(unittest.TestCase):
    UMAP = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]])

    def test_cells_per_study_must_match_the_cohort_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "cells per study"):
            umap_rasters.blood_composite_raster(
                self.UMAP, np.array([0, 1, 1]), ["a", "b"], {"a": 2, "b": 1}
            )

    def test_study_codes_beyond_the_studies_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "study codes"):
            umap_rasters.blood_composite_raster(
                self.UMAP, np.array([0, 1, 2]), ["a", "b"], {"a": 1, "b": 2}
            )

    def test_cells_without_coordinates_are_refused(self) -> None:
        umap = self.UMAP.copy()
        umap[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            umap_rasters.blood_composite_raster(
                umap, np.array([0, 1, 1]), ["a", "b"], {"a": 1, "b": 2}
            )


class TabulaSapiensRasterTests(unittest.TestCase):
    # Four cells at the corners of the unit square; with 2 bins a side each falls in a bin of its own, rows by UMAP 2.
    UMAP = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

    def test_cells_b_cells_and_their_ms4a1_values_fall_in_their_bins(self) -> None:
        with mock.patch.object(umap_rasters, "TABULA_SAPIENS_BINS", 2):
            raster = umap_rasters.tabula_sapiens_raster(
                self.UMAP, np.array([True, True, False, False]), np.array([2.5, np.nan])
            )
        np.testing.assert_array_equal(raster["cells"], [[1, 1], [1, 1]])
        np.testing.assert_array_equal(raster["bCells"], [[1, 0], [0, 1]])
        # A missing MS4A1 value counts as zero.
        np.testing.assert_array_equal(raster["bMs4a1Sum"], [[2.5, 0.0], [0.0, 0.0]])
        self.assertEqual(
            [raster[k].dtype for k in ("cells", "bCells", "bMs4a1Sum")],
            [np.int32, np.int32, np.float64],
        )

    def test_the_grid_spans_the_umap_padded_by_two_percent(self) -> None:
        # With 4 bins, the edges are -0.02, 0.24, 0.5, 0.76 and 1.02 on each axis, so 0.23 falls in the first bin and
        # 0.77 in the last.
        umap = np.array([[0.0, 0.0], [1.0, 1.0], [0.23, 0.77], [0.77, 0.23]])
        with mock.patch.object(umap_rasters, "TABULA_SAPIENS_BINS", 4):
            raster = umap_rasters.tabula_sapiens_raster(umap, np.zeros(4, dtype=bool), np.zeros(0))
        expected = np.zeros((4, 4), dtype=np.int32)
        expected[0, 0] = expected[3, 3] = expected[3, 0] = expected[0, 3] = 1
        np.testing.assert_array_equal(raster["cells"], expected)

    def test_ms4a1_values_must_belong_to_the_b_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "MS4A1 values for 2 B cells"):
            umap_rasters.tabula_sapiens_raster(
                self.UMAP, np.array([True, True, False, False]), np.ones(3)
            )

    def test_a_query_of_other_cells_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "totalCells"):
            umap_rasters.check_tabula_sapiens_query(
                4, self.UMAP, np.ones(4, dtype=bool), np.ones(4)
            )


class LadderReferenceTests(unittest.TestCase):
    # Studies r1 and r2 are reference studies, q a query study; type Z is outside the ladder's vocabulary.
    LADDER = {"design": {"referenceStudies": ["r2", "r1"]}, "vocabulary": ["T", "B"]}

    def cells(self) -> dict:
        return {
            "umap": np.array(
                [[0.0, 0.0], [1.0, 1.0], [0.6, 0.6], [9.0, 9.0], [0.2, 0.8], [5.0, -5.0]]
            ),
            "dataset_codes": np.array([0, 1, 1, 2, 0, 0]),
            "dataset_names": np.array(["r1", "r2", "q"]),
            "harmonized_cell_type_codes": np.array([0, 1, 1, 0, 0, 2]),
            "harmonized_cell_type_names": np.array(["B", "T", "Z"]),
        }

    def test_keeps_the_reference_studies_cells_of_the_ladders_types(self) -> None:
        with mock.patch.object(reference_umap_rasters, "LADDER_BINS", 4):
            arrays, record = reference_umap_rasters.ladder_reference(
                self.cells(), self.LADDER, {"r1": 3, "r2": 2, "q": 1}
            )
        # The query cell and the cell of type Z are left out; the grid spans only the kept cells, [0, 1] square.
        self.assertEqual(arrays["types"].tolist(), ["T", "B"])
        self.assertEqual(record["referenceStudies"], ["r2", "r1"])
        self.assertEqual(record["cellsPerType"], {"T": 2, "B": 2})
        self.assertEqual(list(record["cellsPerStudy"].items()), [("r2", 2), ("r1", 2)])
        expected = np.zeros((2, 4, 4), dtype=np.uint32)
        expected[1, 0, 0] = expected[1, 3, 0] = 1
        expected[0, 3, 3] = expected[0, 2, 2] = 1
        np.testing.assert_array_equal(arrays["typeCells"], expected)

    def test_cells_per_study_must_match_the_cohort_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "cohort analysis record"):
            reference_umap_rasters.ladder_reference(
                self.cells(), self.LADDER, {"r1": 2, "r2": 3, "q": 1}
            )

    def test_a_missing_reference_study_is_refused(self) -> None:
        ladder = {"design": {"referenceStudies": ["r1", "x"]}, "vocabulary": ["T"]}
        with self.assertRaisesRegex(ValueError, "lacks a reference study"):
            reference_umap_rasters.ladder_reference(
                self.cells(), ladder, {"r1": 3, "r2": 2, "q": 1}
            )


class HeldOutTests(unittest.TestCase):
    RESULT = {
        "cells": 3,
        "matchesBenchmark": {"cytearcVersion": "1.3.0", "configuration": True, "cells": True},
    }

    def cells(self) -> dict:
        return {
            "umap": np.array([[0.0, 0.0], [1.0, 1.0], [0.6, 0.1]], dtype=np.float32),
            "label_codes": np.array([1, 0, 1], dtype=np.int16),
            "label_names": np.array(["Podocytes", "Proximal tubule"]),
        }

    def test_bins_the_reference_by_label_with_the_jobs_checks_as_booleans(self) -> None:
        with mock.patch.object(reference_umap_rasters, "HELD_OUT_BINS", 4):
            arrays, record = reference_umap_rasters.held_out(self.RESULT, self.cells())
        self.assertEqual(
            record,
            {
                "cells": 3,
                "matchesBenchmark": {"cytearcVersion": True, "configuration": True, "cells": True},
                "cellsPerType": {"Podocytes": 1, "Proximal tubule": 2},
            },
        )
        self.assertEqual(arrays["typeCells"].shape, (2, 4, 4))
        self.assertEqual(int(arrays["typeCells"][1, 0, 0]), 1)

    def test_a_rebuild_that_did_not_match_the_benchmark_is_refused(self) -> None:
        result = {
            **self.RESULT,
            "matchesBenchmark": {**self.RESULT["matchesBenchmark"], "cells": False},
        }
        with self.assertRaisesRegex(ValueError, "did not match"):
            reference_umap_rasters.held_out(result, self.cells())

    def test_a_umap_of_other_cells_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "the job 4"):
            reference_umap_rasters.held_out({**self.RESULT, "cells": 4}, self.cells())


class PromotedRasterTests(unittest.TestCase):
    """The rasters in records/ hold the cells of the records the figures check them against."""

    @staticmethod
    def load(path) -> dict:
        with np.load(path, allow_pickle=False) as raster:
            return {name: raster[name] for name in raster.files}

    def test_the_tabula_sapiens_raster_holds_the_remote_querys_cells(self) -> None:
        raster = self.load(umap_rasters.TABULA_SAPIENS_RASTER)
        query = json.loads(umap_rasters.REMOTE_QUERY.read_text())["result"]["remoteQueries"][
            "tabulaSapiens1_1M"
        ]
        self.assertEqual(int(raster["cells"].sum()), query["totalCells"])
        self.assertEqual(int(raster["bCells"].sum()), query["bCellsSelected"])
        self.assertAlmostEqual(
            raster["bMs4a1Sum"].sum() / raster["bCells"].sum(),
            query["ms4a1MeanInBCells"],
            places=12,
        )

    def test_the_composite_raster_holds_the_cohort_analysis_cells_per_study(self) -> None:
        raster = self.load(umap_rasters.BLOOD_COMPOSITE_RASTER)
        cohort = json.loads(umap_rasters.BLOOD_COHORT.read_text())["result"]["sourceCellCounts"]
        per_study = raster["studyCells"].sum(axis=(1, 2))
        self.assertEqual(dict(zip(raster["studies"].tolist(), per_study.tolist())), cohort)

    def test_the_ladder_reference_raster_holds_the_ladders_types_and_studies(self) -> None:
        stem = reference_umap_rasters.LADDER_REFERENCE
        raster, record = (
            self.load(stem.with_suffix(".npz")),
            json.loads(stem.with_suffix(".json").read_text()),
        )
        ladder = json.loads(reference_umap_rasters.LADDER.read_text())
        cohort = json.loads(reference_umap_rasters.BLOOD_COHORT.read_text())["result"][
            "sourceCellCounts"
        ]
        self.assertEqual(raster["types"].tolist(), ladder["vocabulary"])
        self.assertEqual(record["referenceStudies"], ladder["design"]["referenceStudies"])
        per_type = raster["typeCells"].sum(axis=(1, 2))
        self.assertEqual(dict(zip(ladder["vocabulary"], per_type.tolist())), record["cellsPerType"])
        self.assertEqual(sum(record["cellsPerStudy"].values()), int(per_type.sum()))
        self.assertTrue(
            all(cells <= cohort[study] for study, cells in record["cellsPerStudy"].items())
        )

    def test_each_held_out_raster_holds_its_reference_cells(self) -> None:
        evaluation = json.loads(
            (ROOT / "experiments/mapping/records/cross_study/evaluation.json").read_text()
        )
        references = {
            m["tissue"]: m["inputs"]["parts"]["reference"]["cells"]
            for m in evaluation["result"]["mapping"]
            if m["method"] == "unintegrated"
        }
        for tissue in reference_umap_rasters.TISSUES:
            stem = reference_umap_rasters.HELD_OUT / tissue
            raster, record = (
                self.load(stem.with_suffix(".npz")),
                json.loads(stem.with_suffix(".json").read_text()),
            )
            per_type = raster["typeCells"].sum(axis=(1, 2))
            self.assertEqual(
                dict(zip(raster["types"].tolist(), per_type.tolist())), record["cellsPerType"]
            )
            self.assertEqual(int(per_type.sum()), record["cells"])
            self.assertEqual(record["cells"], references[tissue])
            self.assertTrue(all(record["matchesBenchmark"].values()))


if __name__ == "__main__":
    unittest.main()
