import unittest

import numpy as np
import pandas as pd
from experiments.runtime import blood_samples as bs


class TimeKeys(unittest.TestCase):
    def test_reads_numbers_orders_and_patterns_and_puts_unrecorded_times_last(self):
        inf = np.inf
        numeric = bs.time_key(pd.Series(["3", "1.5", "nan", "not applicable"]), {"column": "t"})
        self.assertEqual(list(numeric), [3.0, 1.5, inf, inf])
        ordered = bs.time_key(
            pd.Series(["M3", "Base", "D28", "X"]), {"column": "t", "order": ("Base", "D28", "M3")}
        )
        self.assertEqual(list(ordered), [2.0, 0.0, 1.0, inf])
        visits = bs.time_key(
            pd.Series(["Flu Year 1 Day 7", "Flu Year 2 Stand-Alone", "Immune Variation Day 90"]),
            {"column": "t", "pattern": r"Day (\d+)", "default": 0},
        )
        self.assertEqual(list(visits), [7.0, 0.0, 90.0])
        days = bs.time_key(pd.Series(["D0", "DE", "D13"]), {"column": "t", "pattern": r"D(\d+)"})
        self.assertEqual(list(days), [0.0, inf, 13.0])


class SampleSelection(unittest.TestCase):
    RULE = {"s": {"sample": "kit", "time": ({"column": "day"},)}}

    def frame(self, rows, columns=("kit", "day")):
        return pd.DataFrame(rows, columns=["dataset", "donor_group", *columns])

    def test_keeps_each_donors_earliest_sample_then_its_largest_then_the_first_by_name(self):
        frame = self.frame(
            [
                # Donor A: a1 on day 5 has three cells, a2 on day 2 one; the earliest wins over the larger.
                ("s", "A", "a1", "5"),
                ("s", "A", "a1", "5"),
                ("s", "A", "a1", "5"),
                ("s", "A", "a2", "2"),
                # Donor B: no time recorded, so the larger sample b2.
                ("s", "B", "b1", "nan"),
                ("s", "B", "b1", "nan"),
                ("s", "B", "b2", "nan"),
                ("s", "B", "b2", "nan"),
                ("s", "B", "b2", "nan"),
                # Donor C: same day and size, so c1 by name.
                ("s", "C", "c2", "4"),
                ("s", "C", "c1", "4"),
            ]
        )
        keep, record = bs.sample_selection(frame, self.RULE, {})
        self.assertEqual(
            list(keep), [False, False, False, True, False, False, True, True, True, False, True]
        )
        self.assertEqual(record["samples"]["donorsWithSeveralSamples"], 3)
        self.assertEqual(record["samples"]["samplesDropped"], 3)
        self.assertEqual(record["samples"]["cellsDropped"], 6)
        self.assertEqual(record["cellsKept"], 5)

    def test_compares_time_columns_in_turn(self):
        rule = {
            "s": {
                "sample": "kit",
                "time": (
                    {"column": "year"},
                    {"column": "visit", "pattern": r"Day (\d+)", "default": 0},
                ),
            }
        }
        frame = self.frame(
            [
                ("s", "A", "k1", "2020", "Flu Year 2 Day 0"),
                ("s", "A", "k2", "2019", "Flu Year 1 Day 90"),
                ("s", "A", "k3", "2019", "Flu Year 1 Day 7"),
            ],
            columns=("kit", "year", "visit"),
        )
        keep, _ = bs.sample_selection(frame, rule, {})
        # 2019 comes before 2020, and day 7 before day 90 within 2019.
        self.assertEqual(list(keep), [False, False, True])

    def test_removes_excluded_stores_and_values_before_choosing_samples(self):
        rules = {"s": {"exclude": {"kind": ("sorted B",)}, "sample": "kit"}}
        frame = self.frame(
            [
                ("sorted_store", "X", "x1", "PBMC"),
                ("s", "A", "a1", "sorted B"),
                ("s", "A", "a1", "sorted B"),
                ("s", "A", "a1", "sorted B"),
                ("s", "A", "a2", "PBMC"),
            ],
            columns=("kit", "kind"),
        )
        keep, record = bs.sample_selection(frame, rules, {"sorted_store": "B cells only"})
        # Without the sorted cells, a2 is A's only sample; a1 would otherwise win as the larger.
        self.assertEqual(list(keep), [False, False, False, False, True])
        self.assertEqual(record["excludedStores"]["sorted_store"]["cells"], 1)
        self.assertEqual(record["excludedValues"]["s"], {"kind": {"sorted B": 3}})
        self.assertEqual(record["samples"]["donorsWithSeveralSamples"], 0)

    def test_stores_split_by_lineage_share_their_donors_samples(self):
        rule = {"t": {"time": ({"column": "tp", "pattern": r"T(\d+)"},)}}
        rules = {"adaptive": rule["t"], "innate": rule["t"]}
        frame = self.frame(
            [
                ("adaptive", "D", "T1"),
                ("adaptive", "D", "T0"),
                ("innate", "D", "T0"),
                ("innate", "D", "T1"),
                ("innate", "D", "T1"),
                ("innate", "H", "HC"),
            ],
            columns=("tp",),
        )
        keep, record = bs.sample_selection(frame, rules, {})
        # T0 is D's earliest sample in both stores; the healthy donor H has one sample without a time.
        self.assertEqual(list(keep), [False, True, True, False, False, True])
        self.assertEqual(record["samples"]["cellsDroppedByStore"], {"innate": 2, "adaptive": 1})

    def test_cells_without_a_donor_leave_and_stores_without_a_rule_keep_every_cell(self):
        frame = self.frame(
            [("s", "", "a1", "1"), ("free", "F", "f1", "1"), ("free", "F", "f2", "9")]
        )
        keep, _ = bs.sample_selection(frame, self.RULE, {})
        self.assertEqual(list(keep), [False, True, True])

    def test_the_corpus_rules_name_each_column_and_at_most_two_time_columns(self):
        for store, rule in bs.SAMPLE_RULES.items():
            self.assertLessEqual(set(rule), {"exclude", "sample", "time"}, store)
            self.assertLessEqual(len(rule.get("time", ())), 2, store)
            self.assertTrue(rule.get("sample") or rule.get("time") or rule.get("exclude"), store)
            for spec in rule.get("time", ()):
                self.assertLessEqual(set(spec), {"column", "order", "pattern", "default"}, store)
        self.assertFalse(set(bs.SAMPLE_RULES) & set(bs.EXCLUDED_STORES))
        self.assertIn("Sample type", bs.rule_columns())


class EarliestSamples(unittest.TestCase):
    def test_orders_by_time_columns_then_size_then_name(self):
        inf = np.inf
        samples = pd.DataFrame(
            [
                # donor a: day 3 beats day 7, whatever the size.
                ("a", "late", 7.0, inf, 900),
                ("a", "early", 3.0, inf, 10),
                # donor b: the same first time; the second time column decides.
                ("b", "visit2", 2024.0, 90.0, 50),
                ("b", "visit1", 2024.0, 0.0, 5),
                # donor c: no time recorded; the larger sample wins.
                ("c", "small", inf, inf, 20),
                ("c", "large", inf, inf, 200),
                # donor d: a tie on time and size; the first name wins.
                ("d", "zeta", 1.0, inf, 30),
                ("d", "alpha", 1.0, inf, 30),
            ],
            columns=["donor", "name", "t0", "t1", "cells"],
        )
        chosen = bs.earliest_samples(samples).set_index("donor")["name"].to_dict()
        self.assertEqual(chosen, {"a": "early", "b": "visit1", "c": "large", "d": "alpha"})

    def test_names_the_columns_it_needs(self):
        with self.assertRaisesRegex(KeyError, "t1"):
            bs.earliest_samples(pd.DataFrame({"donor": [], "name": [], "t0": [], "cells": []}))


COMBAT = "ahern_2022_combat_project_single_cell_19_sepsis_flu_patient_pbmcs_ebc2e1ff"


class ApplyCohortRecords(unittest.TestCase):
    def values(self, extra=()) -> pd.DataFrame:
        rows = [
            (COMBAT, "p1", "s1", "Source", "Sepsis"),
            (COMBAT, "p2", "s2", "Source", "COVID_LDN"),
            (COMBAT, "p3", "s3", "Source", "COVID_SEV"),
            (COMBAT, "p3", "s4", "Source", "COVID_SEV"),
            (COMBAT, "p4", "s5", "Source", "HV"),
            # Another column of the store, which the rule does not read.
            (COMBAT, "p1", "s1", "Outcome", "Death"),
            *extra,
        ]
        return pd.DataFrame(rows, columns=["storeId", "donor_id", "sample", "column", "value"])

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # Two cells of the sepsis sample, which the store labels COVID-19.
                (COMBAT, "p1", "s1", "COVID-19"),
                (COMBAT, "p1", "s1", "COVID-19"),
                (COMBAT, "p2", "s2", "COVID-19"),
                (COMBAT, "p3", "s3", "COVID-19"),
                (COMBAT, "p3", "s4", "COVID-19"),
                (COMBAT, "p4", "s5", "normal"),
                # Another store with the same donor and sample names keeps its label.
                ("other", "p1", "s1", "COVID-19"),
            ],
            columns=["storeId", "donor_id", "sample", "disease"],
        )

    def test_relabels_the_sepsis_cohort_and_leaves_out_the_neutrophil_samples(self):
        out, changes = bs.apply_cohort_records(self.frame(), self.values())
        self.assertEqual(
            list(zip(out["storeId"], out["donor_id"], out["disease"], strict=True)),
            [
                (COMBAT, "p1", "sepsis"),
                (COMBAT, "p1", "sepsis"),
                (COMBAT, "p3", "COVID-19"),
                (COMBAT, "p3", "COVID-19"),
                (COMBAT, "p4", "normal"),
                ("other", "p1", "COVID-19"),
            ],
        )
        # Samples and donors are counted once however many rows they have.
        self.assertEqual(
            changes,
            {
                COMBAT: {
                    "relabelled": {"sepsis": {"samples": 1, "donors": 1}},
                    "excluded": {"samples": 1, "donors": 1},
                }
            },
        )

    def test_a_table_without_the_store_is_left_as_it_is(self):
        frame = self.frame()
        frame = frame[frame["storeId"] == "other"]
        out, changes = bs.apply_cohort_records(frame, self.values())
        pd.testing.assert_frame_equal(out, frame)
        self.assertEqual(changes, {})

    def test_a_categorical_disease_takes_the_new_label(self):
        frame = self.frame().assign(disease=lambda f: f["disease"].astype("category"))
        out, _ = bs.apply_cohort_records(frame, self.values())
        self.assertEqual(sorted(set(out["disease"])), ["COVID-19", "normal", "sepsis"])

    def test_a_sample_without_a_value_is_an_error(self):
        values = self.values()
        values = values[values["sample"] != "s3"]
        with self.assertRaisesRegex(ValueError, "1 samples have no 'Source' value"):
            bs.apply_cohort_records(self.frame(), values)

    def test_a_sample_with_two_values_is_an_error(self):
        values = self.values([(COMBAT, "p3", "s3", "Source", "COVID_MILD")])
        with self.assertRaisesRegex(ValueError, "two values"):
            bs.apply_cohort_records(self.frame(), values)

    def test_the_rule_names_combats_store_and_its_cohort_column(self):
        rule = bs.COHORT_RECORDS[COMBAT]
        self.assertEqual(rule["column"], "Source")
        self.assertEqual(rule["disease"], {"Sepsis": "sepsis"})
        self.assertEqual(rule["exclude"], ("COVID_LDN",))


ATLAS = "marteau_2026_single_cell_colorectal_cancer_atlas_core_atlas_05a8c945"


class ApplySourceStudies(unittest.TestCase):
    def values(self, extra=()) -> pd.DataFrame:
        rows = [
            (ATLAS, "c1", "a", "study_id", "Innsbruck"),
            (ATLAS, "c2", "b", "study_id", "Innsbruck"),
            (ATLAS, "h1", "c", "study_id", "Chen"),
            (ATLAS, "h1", "d", "study_id", "Chen"),
            # Another column of the store, which the rule does not read.
            (ATLAS, "c1", "a", "dataset", "Innsbruck_2023"),
            *extra,
        ]
        return pd.DataFrame(rows, columns=["storeId", "donor_id", "sample", "column", "value"])

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # Two cells of one donor with cancer, one of another, and two samples of one healthy donor.
                (ATLAS, "atlas", "c1", "a"),
                (ATLAS, "atlas", "c1", "a"),
                (ATLAS, "atlas", "c2", "b"),
                (ATLAS, "atlas", "h1", "c"),
                (ATLAS, "atlas", "h1", "d"),
                # Another store with the same donor and sample names keeps its collection.
                ("other", "other", "c1", "a"),
            ],
            columns=["storeId", "collectionId", "donor_id", "sample"],
        )

    def test_each_sample_of_the_atlas_takes_its_source_study_as_its_collection(self):
        out, sources = bs.apply_source_studies(self.frame(), self.values())
        self.assertEqual(
            list(out["collectionId"]),
            [
                "atlas/Innsbruck",
                "atlas/Innsbruck",
                "atlas/Innsbruck",
                "atlas/Chen",
                "atlas/Chen",
                "other",
            ],
        )
        # Nothing else changes, and no row leaves.
        pd.testing.assert_frame_equal(
            out.drop(columns="collectionId"), self.frame().drop(columns="collectionId")
        )
        # Samples and donors are counted once however many rows they have.
        self.assertEqual(
            sources,
            {
                ATLAS: {
                    "Chen": {"samples": 2, "donors": 1},
                    "Innsbruck": {"samples": 2, "donors": 2},
                }
            },
        )

    def test_the_donor_table_then_keeps_each_source_study_apart(self):
        # Through study_of, the donors with cancer and the healthy donor fall in two studies, so a contrast inside
        # one study finds no healthy donors beside the donors with cancer.
        from experiments.reuse.progenitor_effects import study_of

        out, _ = bs.apply_source_studies(self.frame(), self.values())
        studies = dict(zip(out["donor_id"] + "/" + out["storeId"], study_of(out), strict=True))
        self.assertNotEqual(studies[f"c1/{ATLAS}"], studies[f"h1/{ATLAS}"])
        self.assertEqual(studies[f"c1/{ATLAS}"], studies[f"c2/{ATLAS}"])

    def test_a_table_without_the_store_is_left_as_it_is(self):
        frame = self.frame()
        frame = frame[frame["storeId"] == "other"]
        out, sources = bs.apply_source_studies(frame, self.values())
        pd.testing.assert_frame_equal(out, frame)
        self.assertEqual(sources, {})

    def test_a_categorical_collection_takes_the_new_value(self):
        frame = self.frame().assign(collectionId=lambda f: f["collectionId"].astype("category"))
        out, _ = bs.apply_source_studies(frame, self.values())
        self.assertEqual(
            sorted(set(out["collectionId"])), ["atlas/Chen", "atlas/Innsbruck", "other"]
        )

    def test_a_sample_without_a_source_study_is_an_error(self):
        values = self.values()
        values = values[values["sample"] != "d"]
        with self.assertRaisesRegex(ValueError, "1 samples have no 'study_id' value"):
            bs.apply_source_studies(self.frame(), values)

    def test_a_sample_with_two_source_studies_is_an_error(self):
        values = self.values([(ATLAS, "h1", "c", "study_id", "Innsbruck")])
        with self.assertRaisesRegex(ValueError, "two values"):
            bs.apply_source_studies(self.frame(), values)

    def test_the_rule_names_the_atlas_and_its_study_column(self):
        self.assertEqual(bs.SOURCE_STUDY_RECORDS, {ATLAS: "study_id"})
