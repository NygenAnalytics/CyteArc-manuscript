import unittest

import numpy as np
import pandas as pd
from experiments.catalogue import catalogue_screen as screen


class FakeCells:
    def __init__(self, columns: dict) -> None:
        self.data = columns
        self.columns = list(columns)

    def fetch_all(self, column):
        return np.asarray(self.data[column], dtype=object)


class FakeStore:
    def __init__(self, columns: dict) -> None:
        self.cells = FakeCells(columns)


class SampleRule(unittest.TestCase):
    RULES = {
        "s": {
            "exclude": {"sort": ("CD3+",)},
            "sample": "kit",
            "time": (
                {"column": "year"},
                {"column": "visit", "pattern": r"Day (\d+)", "default": 0},
            ),
        }
    }

    def test_drops_excluded_values_and_reads_samples_and_both_time_columns(self):
        ds = FakeStore(
            {
                "sort": ["none", "CD3+", "none", "CD3+", "none"],
                "kit": ["k1", "k1", "k2", "k2", "k3"],
                "year": ["2020", "2020", "2021", "2021", "nan"],
                "visit": ["Day 7", "Day 7", "Stand-Alone", "Stand-Alone", "Day 90"],
            }
        )
        rows = np.arange(5)
        keep, excluded, codes, names, times = screen.apply_sample_rule(ds, "s", rows, self.RULES)
        self.assertEqual(keep.tolist(), [True, False, True, False, True])
        self.assertEqual(excluded, {"sort": {"CD3+": 2}})
        self.assertEqual([names[c] for c in codes], ["k1", "k1", "k2", "k2", "k3"])
        self.assertEqual(times[:, 0].tolist(), [2020.0, 2020.0, 2021.0, 2021.0, np.inf])
        self.assertEqual(times[:, 1].tolist(), [7.0, 7.0, 0.0, 0.0, 90.0])

    def test_reads_only_the_rows_it_is_given(self):
        ds = FakeStore(
            {
                "kit": ["k1", "k2", "k3"],
                "sort": ["none"] * 3,
                "year": ["1", "2", "3"],
                "visit": ["Day 1"] * 3,
            }
        )
        keep, _, codes, names, times = screen.apply_sample_rule(
            ds, "s", np.array([2, 0]), self.RULES
        )
        self.assertEqual([names[c] for c in codes], ["k3", "k1"])
        self.assertEqual(times[:, 0].tolist(), [3.0, 1.0])
        self.assertTrue(keep.all())

    def test_a_store_without_a_rule_has_one_sample_and_no_time(self):
        ds = FakeStore({"kit": ["k1", "k2"]})
        keep, excluded, codes, names, times = screen.apply_sample_rule(
            ds, "other", np.arange(2), self.RULES
        )
        self.assertTrue(keep.all())
        self.assertEqual(excluded, {})
        self.assertEqual(names, [""])
        self.assertEqual(codes.tolist(), [0, 0])
        self.assertTrue(np.isinf(times).all())

    def test_a_rule_naming_a_missing_column_fails(self):
        ds = FakeStore({"kit": ["k1"], "sort": ["none"], "year": ["1"]})
        with self.assertRaisesRegex(KeyError, "visit"):
            screen.apply_sample_rule(ds, "s", np.arange(1), self.RULES)

    def test_the_first_time_column_names_samples_without_a_sample_column(self):
        rules = {"s": {"time": ({"column": "day"},)}}
        ds = FakeStore({"day": ["3", "0", "3"]})
        _, _, codes, names, times = screen.apply_sample_rule(ds, "s", np.arange(3), rules)
        self.assertEqual([names[c] for c in codes], ["3", "0", "3"])
        self.assertEqual(times[:, 0].tolist(), [3.0, 0.0, 3.0])


class CellScoreBins(unittest.TestCase):
    def test_bins_each_cells_interferon_log_cpm_by_group(self):
        # Libraries of one million make CPM equal to the counts: log2(0 + 1) = 0, log2(3 + 1) = 2 and
        # log2(2^20) = 20, which lies above the last bin's start of 17.75.
        values = np.array([[0.0, 0.0], [1.0, 2.0], [2.0**19, 2.0**19 - 1], [5.0, 5.0]])
        library = np.array([1e6, 1e6, 1e6, 0.0])
        group = np.array([0, 1, 1, 0])
        bins = screen.cell_score_bins(values, library, group, 2)
        self.assertEqual(bins.shape, (2, screen.CELL_SCORE_BINS))
        self.assertEqual(bins.sum(), 3)
        self.assertEqual(bins[0, 0], 1)
        self.assertEqual(bins[1, 8], 1)
        self.assertEqual(bins[1, screen.CELL_SCORE_BINS - 1], 1)


def units_frame(rows: list[dict]) -> pd.DataFrame:
    """Units whose interferon counts are given per gene; genes left out are not measured (NaN). Each library is one
    million, so a count of 2^k - 1 has log2(CPM + 1) = k."""
    records = []
    for row in rows:
        record = {
            "storeId": row.get("store", "s"),
            "collectionId": row.get("collection", "c"),
            "donor_id": row["donor"],
            "sample": row.get("sample", ""),
            "disease": row.get("disease", "normal"),
            "coarse_cell_type": row.get("type", "Monocytes"),
            "cells": row.get("cells", 100),
            "librarySize": 1e6,
            "earliestSample": row.get("earliest", True),
        }
        for symbol in screen.INTERFERON_GENES:
            record[f"count_{symbol}"] = row.get("genes", {}).get(symbol, np.nan)
        records.append(record)
    return pd.DataFrame(records)


def level(k: float) -> float:
    return 2.0**k - 1


class InterferonZScores(unittest.TestCase):
    def test_scores_against_the_stores_healthy_donors_and_drops_genes_that_do_not_vary(self):
        healthy = [
            {"donor": f"h{k}", "genes": {"ISG15": level(k), "MX1": level(2)}} for k in (1, 2, 3, 4)
        ]
        units = units_frame(
            [
                *healthy,
                {
                    "donor": "d1",
                    "disease": "COVID-19",
                    "genes": {"ISG15": level(6), "MX1": level(9)},
                },
                # A later healthy sample is not part of the reference, though it is scored.
                {"donor": "h1", "earliest": False, "genes": {"ISG15": level(10), "MX1": level(2)}},
            ]
        )
        scores = screen.interferon_zscores(units)
        sd = np.std([1, 2, 3, 4], ddof=1)
        expected = [(k - 2.5) / sd for k in (1, 2, 3, 4)] + [(6 - 2.5) / sd, (10 - 2.5) / sd]
        np.testing.assert_allclose(scores, expected, rtol=1e-12)

    def test_averages_the_z_scores_of_every_gene_that_varies(self):
        healthy = [
            {"donor": f"h{k}", "genes": {"ISG15": level(k), "IFI27": level(2 * k)}}
            for k in (1, 2, 3)
        ]
        units = units_frame(
            [
                *healthy,
                {"donor": "d", "disease": "lupus", "genes": {"ISG15": level(5), "IFI27": level(5)}},
            ]
        )
        scores = screen.interferon_zscores(units)
        z_isg15 = (5 - 2) / np.std([1, 2, 3], ddof=1)
        z_ifi27 = (5 - 4) / np.std([2, 4, 6], ddof=1)
        self.assertAlmostEqual(scores[-1], (z_isg15 + z_ifi27) / 2, places=12)

    def test_needs_three_healthy_donors_with_enough_cells_per_store_and_type(self):
        units = units_frame(
            [
                {"donor": "h1", "genes": {"ISG15": level(1)}},
                {"donor": "h2", "genes": {"ISG15": level(2)}},
                {"donor": "h3", "cells": 5, "genes": {"ISG15": level(3)}},
                {"donor": "d", "disease": "COVID-19", "genes": {"ISG15": level(6)}},
                # The same donors in another cell type do not lend their reference.
                *[
                    {"donor": f"h{k}", "type": "T cells", "genes": {"ISG15": level(k)}}
                    for k in (1, 2, 3)
                ],
            ]
        )
        scores = screen.interferon_zscores(units)
        self.assertTrue(np.isnan(scores[:4]).all())
        self.assertTrue(np.isfinite(scores[4:]).all())


class EarliestSamples(unittest.TestCase):
    def test_lineage_split_stores_keep_the_same_earliest_sample(self):
        units = units_frame(
            [
                {"donor": "a", "store": "myeloid", "sample": "early", "cells": 10},
                {"donor": "a", "store": "lymphoid", "sample": "early", "cells": 10},
                {"donor": "a", "store": "myeloid", "sample": "late", "cells": 900},
                {"donor": "a", "store": "lymphoid", "sample": "late", "cells": 900},
            ]
        )
        units["sampleTime0"] = [1.0, 1.0, 5.0, 5.0]
        units["sampleTime1"] = np.inf
        self.assertEqual(screen.mark_earliest_samples(units).tolist(), [True, True, False, False])

    def test_without_times_the_donors_largest_sample_over_all_its_stores_wins(self):
        units = units_frame(
            [
                {"donor": "b", "store": "myeloid", "sample": "x", "cells": 300},
                {"donor": "b", "store": "lymphoid", "sample": "x", "cells": 10},
                {"donor": "b", "store": "myeloid", "sample": "y", "cells": 200},
                {"donor": "b", "store": "lymphoid", "sample": "y", "cells": 200},
                # The same donor name in another collection is another donor.
                {"donor": "b", "collection": "other", "sample": "z", "cells": 1},
            ]
        )
        units["sampleTime0"] = np.inf
        units["sampleTime1"] = np.inf
        self.assertEqual(
            screen.mark_earliest_samples(units).tolist(), [False, False, True, True, True]
        )


class DonorUnits(unittest.TestCase):
    def test_keeps_samples_apart_and_sums_counts_and_cell_bins(self):
        base = {
            "storeId": "s",
            "collectionId": "c",
            "donor_id": "a",
            "disease": "normal",
            "sex": "female",
            "development_stage": "adult",
            "tissue": "blood",
            "coarse_cell_type": "Monocytes",
            "librarySize": 1e6,
        }
        rows = []
        for sample, cells, time, count, first_bin in (
            ("k1", 30, 2.0, 7.0, 3),
            ("k1", 20, 1.0, 1.0, 2),
            ("k2", 40, 9.0, 15.0, 4),
        ):
            row = {
                **base,
                "sample": sample,
                "cells": cells,
                "sampleTime0": time,
                "sampleTime1": np.inf,
            }
            for symbol in screen.INTERFERON_GENES:
                row[f"count_{symbol}"] = count if symbol == "ISG15" else np.nan
            for column in screen.CELL_SCORE_COLUMNS:
                row[column] = 0
            row[screen.CELL_SCORE_COLUMNS[0]] = first_bin
            rows.append(row)
        units, dropped = screen.donor_units(pd.DataFrame(rows))
        self.assertEqual(dropped, [])
        units = units.set_index("sample")
        self.assertEqual(units.loc["k1", "cells"], 50)
        self.assertEqual(units.loc["k1", "count_ISG15"], 8.0)
        self.assertTrue(np.isnan(units.loc["k1", "count_MX1"]))
        self.assertEqual(units.loc["k1", "sampleTime0"], 1.0)
        self.assertEqual(units.loc["k1", screen.CELL_SCORE_COLUMNS[0]], 5)
        self.assertEqual(units.loc["k2", "cells"], 40)
        # Over the one gene measured: k1 pools 8 counts over two libraries of a million (CPM 4), k2 has CPM 15.
        self.assertAlmostEqual(units.loc["k1", "interferonScore"], np.log2(5), places=12)
        self.assertAlmostEqual(units.loc["k2", "interferonScore"], 4.0, places=12)


class Summaries(unittest.TestCase):
    def test_count_each_donor_once_from_its_earliest_sample(self):
        units = units_frame(
            [
                {"donor": "a", "sample": "early", "genes": {"ISG15": level(3)}},
                {"donor": "a", "sample": "late", "earliest": False, "genes": {"ISG15": level(9)}},
                {"donor": "b", "genes": {"ISG15": level(5)}},
            ]
        )
        units["interferonScore"] = screen.interferon_score(units)
        (entry,) = screen.summarize_scores(units)["byDisease"]
        self.assertEqual(entry["donors"], 2)
        self.assertAlmostEqual(entry["median"], 4.0, places=12)


if __name__ == "__main__":
    unittest.main()
