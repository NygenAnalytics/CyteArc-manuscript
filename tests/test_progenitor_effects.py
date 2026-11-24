import math
import os
import unittest

import numpy as np
import pandas as pd
from experiments.reuse import progenitor_effects as pe
from experiments.runtime.blood_samples import EXCLUDED_STORES

# A store of sorted T cells, which the blood corpus excludes and which is neither enriched nor left out.
SORTED_STORE = next(s for s in EXCLUDED_STORES if s not in pe.ENRICHED and s not in pe.LEFT_OUT)


def candidate(
    store="s",
    collection="c",
    donor="d1",
    sample="",
    labelled=False,
    genes=(0, 0, 0, 0),
    cell_type="T cell",
    row=0,
):
    return {
        "storeId": store,
        "collectionId": collection,
        "donor_id": donor,
        "sample": sample,
        "labelled": labelled,
        "cell_type": cell_type,
        "source_row": row,
        **dict(zip(pe.GENES, genes, strict=True)),
    }


def total(
    store="s",
    collection="c",
    donor="d1",
    sample="",
    cells=5000,
    time=np.inf,
    disease="normal",
    stage="40-year-old stage",
    sex="female",
):
    return {
        "storeId": store,
        "collectionId": collection,
        "donor_id": donor,
        "sample": sample,
        "cells": cells,
        "sampleTime0": time,
        "sampleTime1": np.inf,
        "disease": disease,
        "development_stage": stage,
        "sex": sex,
    }


class Selection(unittest.TestCase):
    def test_labels_or_two_of_four_genes_and_never_malignant_cells(self):
        cells = pd.DataFrame(
            [
                candidate(labelled=True),
                candidate(genes=(1, 0, 2, 0)),
                candidate(genes=(1, 0, 0, 0)),
                candidate(genes=(np.nan, 3, np.nan, 1)),
                candidate(labelled=True, cell_type="malignant cell"),
            ]
        )
        self.assertEqual(pe.default_selection(cells).tolist(), [True, True, False, True, False])


class DonorTable(unittest.TestCase):
    def test_counts_the_progenitors_of_each_donors_earliest_sample(self):
        cells = pd.DataFrame(
            [
                candidate(donor="d1", sample="early", labelled=True),
                candidate(donor="d1", sample="late", labelled=True),
                candidate(donor="d1", sample="late", labelled=True),
                candidate(donor="d2", genes=(1, 1, 0, 0)),
                candidate(donor="d2", genes=(1, 0, 0, 0)),
                # Enriched, sorted and left-out stores are never donors.
                candidate(store=pe.ENRICHED[0], collection="f", donor="r1", labelled=True),
                candidate(store=next(iter(pe.LEFT_OUT)), collection="x", donor="l1", labelled=True),
                candidate(store=SORTED_STORE, collection="s", donor="t1", labelled=True),
            ]
        )
        totals = pd.DataFrame(
            [
                total(
                    donor="d1",
                    sample="early",
                    time=1.0,
                    cells=4000,
                    disease="COVID-19",
                    stage="61-year-old stage",
                ),
                total(donor="d1", sample="late", time=9.0, cells=9000, disease="COVID-19"),
                total(donor="d2", cells=3000),
                total(donor="small", cells=100),
                total(store=pe.ENRICHED[0], collection="f", donor="r1"),
                total(store=next(iter(pe.LEFT_OUT)), collection="x", donor="l1"),
                total(store=SORTED_STORE, collection="s", donor="t1"),
            ]
        )
        table = pe.donor_table(cells, totals).set_index("donor_id")
        self.assertEqual(sorted(table.index), ["d1", "d2"])
        self.assertEqual(table.loc["d1", "sample"], "early")
        self.assertEqual(table.loc["d1", "progenitors"], 1)
        self.assertEqual(table.loc["d1", "cells"], 4000)
        self.assertEqual(table.loc["d1", "age"], 61.0)
        self.assertEqual(table.loc["d2", "progenitors"], 1)

    def test_a_sorted_store_adds_no_cells_to_a_donor_of_its_collection(self):
        # The same donor in a whole-blood store and in a store of sorted cells of one collection: only the
        # whole-blood cells count, or the sorted cells would dilute the donor's share of progenitors.
        cells = pd.DataFrame([candidate(store="whole", collection="c", donor="d1", labelled=True)])
        totals = pd.DataFrame(
            [
                total(store="whole", collection="c", donor="d1", cells=4000),
                total(store=SORTED_STORE, collection="c", donor="d1", cells=6000),
            ]
        )
        table = pe.donor_table(cells, totals)
        self.assertEqual(table["cells"].tolist(), [4000])

    def test_stores_given_keep_only_their_cells_and_totals(self):
        cells = pd.DataFrame(
            [
                candidate(store="a", donor="d1", labelled=True),
                candidate(store="b", donor="d2", labelled=True),
            ]
        )
        totals = pd.DataFrame([total(store="a", donor="d1"), total(store="b", donor="d2")])
        table = pe.donor_table(cells, totals, stores={"a"})
        self.assertEqual(table["donor_id"].tolist(), ["d1"])


class Studies(unittest.TestCase):
    def test_a_collection_of_independent_cohorts_splits_by_store(self):
        split = pe.COHORTS_BY_STORE[0]
        frame = pd.DataFrame(
            {"collectionId": [split, split, "other", "other"], "storeId": ["a", "b", "c", "d"]}
        )
        self.assertEqual(pe.study_of(frame).tolist(), ["a", "b", "other", "other"])

    def test_donors_of_two_cohorts_in_one_collection_stay_apart(self):
        split = pe.COHORTS_BY_STORE[0]
        cells = pd.DataFrame(
            [
                candidate(store="a", collection=split, donor="d1", labelled=True),
                candidate(store="b", collection=split, donor="d1"),
            ]
        )
        totals = pd.DataFrame(
            [
                total(store="a", collection=split, donor="d1"),
                total(store="b", collection=split, donor="d1"),
            ]
        )
        table = pe.donor_table(cells, totals).set_index("study")
        self.assertEqual(table.loc["a", "progenitors"], 1)
        self.assertEqual(table.loc["b", "progenitors"], 0)


def donors_with(
    rate_cases: float, rate_healthy: float, n: int = 12, cells: int = 100_000
) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "study": "c",
                "donor_id": f"case{i}",
                "disease": "COVID-19",
                "cells": cells,
                "progenitors": int(round(rate_cases * cells)) + (i % 3) - 1,
                "age": 40.0,
                "sex": "female",
            }
        )
        rows.append(
            {
                "study": "c",
                "donor_id": f"healthy{i}",
                "disease": "normal",
                "cells": cells,
                "progenitors": int(round(rate_healthy * cells)) + (i % 3) - 1,
                "age": 40.0,
                "sex": "female",
            }
        )
    return pd.DataFrame(rows)


class ContrastEffects(unittest.TestCase):
    def test_recovers_a_doubled_progenitor_rate(self):
        (effect,) = pe.contrast_effects(donors_with(0.0010, 0.0005), "COVID-19")
        # At rates this small the log odds ratio is the log rate ratio, log 2, within the planted jitter.
        self.assertAlmostEqual(effect["effect"], np.log(2), delta=0.05)
        self.assertAlmostEqual(effect["casesPer10k"], 10.0, delta=0.1)

    def test_needs_enough_donors_and_a_progenitor_on_each_side(self):
        self.assertEqual(pe.contrast_effects(donors_with(0.001, 0.0005, n=4), "COVID-19"), [])
        table = donors_with(0.001, 0.0005)
        table.loc[table["disease"] == "normal", "progenitors"] = 0
        self.assertEqual(pe.contrast_effects(table, "COVID-19"), [])


class AgeEffects(unittest.TestCase):
    def test_recovers_a_planted_decline_per_decade(self):
        ages = np.linspace(20, 80, 30)
        table = pd.DataFrame(
            {
                "study": "c",
                "donor_id": [f"d{i}" for i in range(30)],
                "disease": "normal",
                "cells": 200_000,
                # The rate halves every 20 years: log odds fall by log(2) / 2 per decade.
                "progenitors": np.round(200_000 * 0.002 * 0.5 ** ((ages - 20) / 20)).astype(int),
                "age": ages,
                "sex": ["male", "female"] * 15,
            }
        )
        (effect,) = pe.age_effects(table)
        self.assertAlmostEqual(effect["effect"], -np.log(2) / 2, delta=0.01)
        self.assertEqual(effect["donors"], 30)

    def test_needs_enough_adults_over_a_wide_enough_span(self):
        table = pd.DataFrame(
            {
                "study": "c",
                "donor_id": [f"d{i}" for i in range(30)],
                "disease": "normal",
                "cells": 1000,
                "progenitors": 1,
                "age": np.linspace(30, 40, 30),
                "sex": "female",
            }
        )
        self.assertEqual(pe.age_effects(table), [])


class StateNames(unittest.TestCase):
    def test_each_cluster_takes_the_state_whose_genes_stand_out(self):
        genes = ["CRHBP", "HLF", "PF4", "PPBP", "CD3E", "NKG7"]
        means = np.array(
            [
                [3.0, 2.0, 0.1, 0.1, 0.0, 0.0],
                [0.2, 0.1, 4.0, 3.0, 0.1, 0.0],
                [0.1, 0.0, 0.0, 0.1, 3.0, 2.5],
            ]
        )
        panel = pd.DataFrame(means, columns=[f"{g}_mean" for g in genes]).assign(cluster=[0, 1, 2])
        self.assertEqual(
            pe.state_names(panel), {0: "HSC/MPP", 1: "megakaryocyte", 2: "mature T/NK"}
        )

    def test_clusters_split_into_mature_lineages_and_progenitor_states(self):
        names = {
            0: "HSC/MPP",
            1: "mature B",
            2: "erythroid",
            3: "mature T/NK",
            4: "mature monocyte",
            5: "HSC/MPP",
        }
        self.assertEqual(pe.cluster_counts(names), {"matureClusters": 3, "progenitorClusters": 3})
        # Every mature lineage counts as mature, and nothing else does.
        self.assertEqual(
            pe.cluster_counts({i: state for i, state in enumerate(pe.MATURE)}),
            {"matureClusters": len(pe.MATURE), "progenitorClusters": 0},
        )
        self.assertEqual(
            pe.cluster_counts({0: "dendritic"}), {"matureClusters": 0, "progenitorClusters": 1}
        )


class Selections(unittest.TestCase):
    def test_state_selections_pick_the_cells_of_their_states_and_stores(self):
        cells = pd.DataFrame(
            [
                candidate(store="a", row=0, labelled=True),
                candidate(store="a", row=1, labelled=True),
                candidate(store="a", row=2, labelled=True),
                candidate(store="a", row=3, genes=(1, 1, 0, 0)),
                # Never analysed: no state selection picks it, whatever its label.
                candidate(store="b", row=0, labelled=True),
            ]
        )
        named = pd.DataFrame(
            {
                "dataset": ["a", "a", "a", "a"],
                "source_row": [0, 1, 2, 3],
                "state": ["HSC/MPP", "megakaryocyte", "mature T/NK", "erythroid"],
            }
        )
        chosen = {
            name: select(cells).tolist()
            for name, (select, _) in pe.selections(cells, named).items()
        }
        self.assertEqual(chosen["progenitorStates"], [True, True, False, True, False])
        self.assertEqual(chosen["stemAndMultipotent"], [True, False, False, False, False])
        self.assertEqual(chosen["withoutMegakaryocytic"], [True, False, False, True, False])
        self.assertEqual(chosen["labelsOnly"], [True, True, True, False, True])
        self.assertEqual(chosen["markersOnly"], [False, False, False, True, False])
        # The state selections count cells and donors only in the analysed stores.
        self.assertEqual(pe.selections(cells, named)["stemAndMultipotent"][1], {"a"})


class StateCovidEffects(unittest.TestCase):
    def test_a_state_doubled_with_covid_pools_to_two_and_a_steady_state_to_one(self):
        # Two studies, each with six COVID-19 and six healthy donors of 3,000 cells. COVID-19 donors carry twice the
        # HSC/MPP cells of healthy donors and as many lymphoid-primed cells.
        candidates, totals, named = [], [], []
        for study in ("c1", "c2"):
            store = f"s_{study}"
            for i in range(6):
                for disease, side in (("COVID-19", "case"), ("normal", "healthy")):
                    donor = f"{study}_{side}{i}"
                    totals.append(
                        total(
                            store=store, collection=study, donor=donor, cells=3000, disease=disease
                        )
                    )
                    jitter = (i % 3) - 1
                    counts = {
                        "HSC/MPP": (20 if side == "case" else 10) + jitter,
                        "lymphoid-primed": 10 + jitter,
                    }
                    for state, count in counts.items():
                        for _ in range(count):
                            row = len(named)
                            candidates.append(
                                candidate(store=store, collection=study, donor=donor, row=row)
                            )
                            named.append({"dataset": store, "source_row": row, "state": state})
        effects = pe.state_covid_effects(
            pd.DataFrame(candidates), pd.DataFrame(totals), pd.DataFrame(named)
        )
        self.assertEqual(
            effects["HSC/MPP"]["cells"], sum(1 for n in named if n["state"] == "HSC/MPP")
        )
        self.assertEqual(len(effects["HSC/MPP"]["covid"]), 2)
        self.assertAlmostEqual(effects["HSC/MPP"]["covidPooled"]["ratio"], 2.0, delta=0.15)
        self.assertAlmostEqual(effects["lymphoid-primed"]["covidPooled"]["ratio"], 1.0, delta=0.1)
        # A state with no cells has no contrast and no pool.
        self.assertEqual(effects["erythroid"]["covid"], [])
        self.assertIsNone(effects["erythroid"]["covidPooled"])
        self.assertNotIn("mature T/NK", effects)


def progenitor_tables(counts_by_donor: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Candidate cells, donor totals and analysed cells for donors given as (study, donor, disease) -> {state: cells},
    each donor with 3,000 blood cells in a store of its study."""
    candidates, totals, named = [], [], []
    for (study, donor, disease), counts in counts_by_donor.items():
        store = f"s_{study}"
        totals.append(
            total(store=store, collection=study, donor=donor, cells=3000, disease=disease)
        )
        for state, count in counts.items():
            for _ in range(count):
                row = len(named)
                candidates.append(candidate(store=store, collection=study, donor=donor, row=row))
                named.append({"dataset": store, "source_row": row, "state": state})
    return pd.DataFrame(candidates), pd.DataFrame(totals), pd.DataFrame(named)


class CovidComposition(unittest.TestCase):
    def donors(self, covid: dict, healthy: dict, studies=("c1", "c2"), n: int = 6) -> dict:
        # Donors of a side differ in make-up, the first state up where the second is down, so the models see spread,
        # while each side's cells add up to its counts times n.
        found = {}
        for study in studies:
            for i in range(n):
                jitter = (i % 3) - 1
                for disease, side, counts in (
                    ("COVID-19", "case", covid),
                    ("normal", "healthy", healthy),
                ):
                    found[(study, f"{study}_{side}{i}", disease)] = {
                        state: count + (jitter if k == 0 else -jitter)
                        for k, (state, count) in enumerate(counts.items())
                    }
        return found

    def test_more_progenitors_of_the_same_make_up_keep_every_share(self):
        # COVID-19 donors carry twice the progenitors of healthy donors, half HSC/MPP and half erythroid on both sides.
        donors = self.donors({"HSC/MPP": 20, "erythroid": 20}, {"HSC/MPP": 10, "erythroid": 10})
        made = pe.covid_composition(*progenitor_tables(donors))
        self.assertEqual(made["studies"], ["c1", "c2"])
        self.assertAlmostEqual(made["healthy"]["HSC/MPP"], 0.5)
        self.assertAlmostEqual(made["covid"]["erythroid"], 0.5)
        self.assertEqual(made["covid"]["cycling"], 0.0)
        self.assertAlmostEqual(sum(made["covid"].values()), 1.0)
        self.assertAlmostEqual(made["tests"]["HSC/MPP"]["ratio"], 1.0, delta=0.05)
        self.assertGreater(made["tests"]["HSC/MPP"]["p"], 0.5)
        # A state no donor holds has no test.
        self.assertIsNone(made["tests"]["cycling"])

    def test_a_state_that_gains_share_with_covid_shows_in_shares_and_test(self):
        # Three quarters of COVID-19 donors' progenitors are HSC/MPP against half of healthy donors': odds of 3 against
        # 1, an odds ratio of 3.
        donors = self.donors({"HSC/MPP": 30, "erythroid": 10}, {"HSC/MPP": 10, "erythroid": 10})
        made = pe.covid_composition(*progenitor_tables(donors))
        self.assertAlmostEqual(made["covid"]["HSC/MPP"], 0.75, delta=0.01)
        self.assertAlmostEqual(made["tests"]["HSC/MPP"]["ratio"], 3.0, delta=0.1)
        self.assertLess(made["tests"]["HSC/MPP"]["p"], 0.001)

    def test_a_study_with_too_few_donors_on_a_side_is_left_out(self):
        donors = self.donors({"HSC/MPP": 20, "erythroid": 20}, {"HSC/MPP": 10, "erythroid": 10})
        donors |= self.donors(
            {"HSC/MPP": 5, "erythroid": 1}, {"HSC/MPP": 1, "erythroid": 5}, studies=("c3",), n=4
        )
        made = pe.covid_composition(*progenitor_tables(donors))
        self.assertEqual(made["studies"], ["c1", "c2"])
        self.assertAlmostEqual(made["healthy"]["HSC/MPP"], 0.5)


class StateShares(unittest.TestCase):
    def test_each_source_shares_its_progenitors_among_the_states_without_mature_cells(self):
        enriched = pe.ENRICHED[0]
        named = pd.DataFrame(
            {
                "dataset": ["a", "a", "a", "a", "b", enriched, enriched, enriched],
                "state": [
                    "HSC/MPP",
                    "HSC/MPP",
                    "erythroid",
                    "mature T/NK",
                    "cycling",
                    "HSC/MPP",
                    "erythroid",
                    "mature B",
                ],
            }
        )
        shares = pe.state_shares(named)
        # Four blood progenitors, the mature cell left out: two HSC/MPP, one erythroid, one cycling.
        self.assertEqual(
            {state: share for state, share in shares["blood"].items() if share},
            {"HSC/MPP": 0.5, "erythroid": 0.25, "cycling": 0.25},
        )
        self.assertEqual(shares["enriched"]["HSC/MPP"], 0.5)
        self.assertEqual(shares["enriched"]["dendritic"], 0.0)
        self.assertNotIn("mature B", shares["enriched"])
        self.assertAlmostEqual(sum(shares["blood"].values()), 1.0)

    def test_a_source_without_progenitors_is_an_error(self):
        named = pd.DataFrame({"dataset": ["a"], "state": ["HSC/MPP"]})
        with self.assertRaisesRegex(ValueError, "enriched"):
            pe.state_shares(named)

    def test_disease_studies_count_a_study_once_across_its_diseases(self):
        # Study s1 contrasts two diseases, s2 and s3 one each.
        diseases = {
            "COVID-19": {"studies": [{"study": "s1"}, {"study": "s2"}]},
            "sepsis": {"studies": [{"study": "s1"}]},
            "lupus": {"studies": [{"study": "s3"}]},
        }
        self.assertEqual(pe.disease_studies(diseases), 3)

    def test_disease_donors_count_a_studys_healthy_donors_once_across_its_diseases(self):
        # Study s1 compares 3 COVID-19 and 2 sepsis donors with the same 4 healthy donors; s2 compares 2 COVID-19 donors
        # with 2 healthy ones; s1's lupus donors and s3, which contrasts nothing, stay out: 3 + 2 + 4 + 2 + 2 = 13.
        def donors(study: str, disease: str, n: int) -> list[dict]:
            return [
                {"study": study, "donor_id": f"{disease}-{i}", "disease": disease} for i in range(n)
            ]

        table = pd.DataFrame(
            donors("s1", "COVID-19", 3)
            + donors("s1", "sepsis", 2)
            + donors("s1", "normal", 4)
            + donors("s1", "lupus", 6)
            + donors("s2", "COVID-19", 2)
            + donors("s2", "normal", 2)
            + donors("s3", "normal", 7)
        )
        diseases = {
            "COVID-19": {
                "studies": [
                    {"study": "s1", "cases": 3, "healthy": 4},
                    {"study": "s2", "cases": 2, "healthy": 2},
                ]
            },
            "sepsis": {"studies": [{"study": "s1", "cases": 2, "healthy": 4}]},
        }
        self.assertEqual(pe.disease_donors(table, diseases), 13)

    def test_disease_donors_other_than_the_contrast_compared_is_an_error(self):
        table = pd.DataFrame(
            [{"study": "s1", "donor_id": str(i), "disease": "normal"} for i in range(3)]
            + [{"study": "s1", "donor_id": "c", "disease": "sepsis"}]
        )
        diseases = {"sepsis": {"studies": [{"study": "s1", "cases": 1, "healthy": 4}]}}
        with self.assertRaisesRegex(ValueError, "s1 holds 4 donors for sepsis"):
            pe.disease_donors(table, diseases)

    def test_the_largest_difference_is_the_state_furthest_apart_in_either_direction(self):
        # Erythroid is 8 points lower in blood than in the enriched stores; HSC/MPP 5 points higher, cycling 3 lower.
        shares = {
            "blood": {"HSC/MPP": 0.50, "erythroid": 0.30, "cycling": 0.20},
            "enriched": {"HSC/MPP": 0.45, "erythroid": 0.38, "cycling": 0.17},
        }
        result = pe.largest_share_difference(shares)
        self.assertEqual(result["state"], "erythroid")
        self.assertAlmostEqual(result["percentagePoints"], 8.0)


class SourceCounts(unittest.TestCase):
    def test_counts_each_step_for_the_enriched_stores_and_the_rest(self):
        enriched = pe.ENRICHED[0]
        left_out = next(iter(pe.LEFT_OUT))
        discovery = [
            {"storeId": enriched, "cellsRead": 100, "candidates": 90},
            {"storeId": "a", "cellsRead": 1000, "candidates": 20},
            {"storeId": "b", "cellsRead": 500, "candidates": 5},
            {"storeId": left_out, "cellsRead": 400, "candidates": 30},
        ]
        extracted = {enriched: {"cells": 50}, "a": {"cells": 7}}
        analysed = pd.DataFrame(
            {
                "dataset": [enriched, enriched, "a", "a", "a"],
                "state": ["HSC/MPP", "mature B", "erythroid", "mature T/NK", "HSC/MPP"],
            }
        )
        # The selection keeps labelled cells and cells with two of the four genes, never in a left-out store and never
        # labelled malignant: 2 of store a's candidates, 1 of store b's and 3 of the enriched store's.
        candidates = pd.DataFrame(
            [
                candidate(store="a", labelled=True),
                candidate(store="a", genes=(1, 1, 0, 0)),
                candidate(store="a", genes=(1, 0, 0, 0)),
                candidate(store="a", labelled=True, cell_type="malignant cell"),
                candidate(store="b", genes=(0, 2, 0, 1)),
                candidate(store=left_out, labelled=True),
                *[candidate(store=enriched, labelled=True) for _ in range(3)],
            ]
        )
        counts = pe.source_counts(discovery, extracted, analysed, candidates)
        self.assertEqual(
            counts["blood"],
            {
                "stores": 3,
                "cellsRead": 1900,
                "candidates": 55,
                "selected": 3,
                "extracted": 7,
                "analysed": 3,
                "progenitors": 2,
                "extractedNotAnalysed": 4,
                "candidateShare": 55 / 1900,
                "selectedShare": 3 / 1900,
            },
        )
        self.assertEqual(
            counts["enriched"],
            {
                "stores": 1,
                "cellsRead": 100,
                "candidates": 90,
                "selected": 3,
                "extracted": 50,
                "analysed": 2,
                "progenitors": 1,
                "extractedNotAnalysed": 48,
                "candidateShare": 0.9,
                "selectedShare": 0.03,
            },
        )


COMBAT = "ahern_2022_combat_project_single_cell_19_sepsis_flu_patient_pbmcs_ebc2e1ff"
LIU = sorted(store for store in pe.SEVERITY if store.startswith("liu_2021"))


def severity_rows(extra=()) -> list[dict]:
    """A severity record's rows: one per store of SEVERITY, so the record covers them all, then ``extra``."""
    rows = [
        {
            "storeId": store,
            "donor_id": "x",
            "sample": "",
            "column": column,
            "value": next(iter(values)),
            "cells": 1,
        }
        for store, (column, values) in pe.SEVERITY.items()
    ]
    return rows + list(extra)


def severity_row(donor, sample, value, column="Source", store=COMBAT, cells=100) -> dict:
    return {
        "storeId": store,
        "donor_id": donor,
        "sample": sample,
        "column": column,
        "value": value,
        "cells": cells,
    }


class SeveritySamples(unittest.TestCase):
    def test_reads_each_stores_own_column_and_classes_its_values(self):
        samples = pe.severity_samples(
            severity_rows(
                [
                    severity_row("p1", "s1", "COVID_CRIT"),
                    # Another column of the same store is not the severity column and is ignored.
                    severity_row("p1", "s1", "Death", column="Outcome"),
                    severity_row("p2", "s2", "Sepsis"),
                    severity_row("p3", "s3", "COVID_HCW_MILD"),
                ]
            )
        ).set_index(["storeId", "donor_id", "sample"])
        self.assertEqual(samples.loc[(COMBAT, "p1", "s1"), "value"], "COVID_CRIT")
        self.assertEqual(samples.loc[(COMBAT, "p1", "s1"), "severity"], "severe")
        self.assertEqual(samples.loc[(COMBAT, "p3", "s3"), "severity"], "mild")
        # Sepsis names no COVID-19 severity.
        self.assertTrue(pd.isna(samples.loc[(COMBAT, "p2", "s2"), "severity"]))

    def test_a_sample_with_two_values_is_an_error(self):
        rows = severity_rows(
            [severity_row("p1", "s1", "COVID_CRIT"), severity_row("p1", "s1", "COVID_MILD")]
        )
        with self.assertRaisesRegex(ValueError, "two severity values"):
            pe.severity_samples(rows)

    def test_a_store_missing_from_the_record_is_an_error(self):
        rows = [row for row in severity_rows() if row["storeId"] != COMBAT]
        with self.assertRaisesRegex(ValueError, "no rows for"):
            pe.severity_samples(rows)


class StudySeverity(unittest.TestCase):
    def samples(self, second: tuple[str, str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"storeId": LIU[0], "donor_id": "p1", "sample": "T0", "value": "Critical", "severity": "severe"},
                {"storeId": LIU[1], "donor_id": "p1", "sample": "T0", "value": second[0], "severity": second[1]},
            ]
        )  # fmt: skip

    def totals(self) -> pd.DataFrame:
        return pd.DataFrame(
            [total(store=LIU[0], collection="liu"), total(store=LIU[1], collection="liu")]
        )

    def test_the_stores_of_one_study_give_one_class_per_donor_sample(self):
        classes = pe.study_severity(self.samples(("Critical", "severe")), self.totals())
        self.assertEqual(
            classes.to_dict("records"),
            [{"study": "liu", "donor_id": "p1", "sample": "T0", "severity": "severe"}],
        )

    def test_two_classes_across_a_studys_stores_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "two severity classes"):
            pe.study_severity(self.samples(("Moderate", "mild")), self.totals())


def severity_donors(
    rates: dict[str, float], study: str = "a", n: int = 6, cells: int = 100_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A donor table of ``n`` donors per severity class at the given progenitor rates, and its classes."""
    rows, classes = [], []
    for name, rate in rates.items():
        for i in range(n):
            donor = f"{name}{i}"
            rows.append(
                {
                    "study": study,
                    "donor_id": donor,
                    "sample": "",
                    "disease": "normal" if name == "healthy" else "COVID-19",
                    "cells": cells,
                    "progenitors": int(round(rate * cells)) + (i % 3) - 1,
                    "stage": "40-year-old stage",
                }
            )
            classes.append({"study": study, "donor_id": donor, "sample": "", "severity": name})
    return pd.DataFrame(rows), pd.DataFrame(classes)


DOUBLING = {"healthy": 0.0005, "mild": 0.0010, "severe": 0.0020}


class SeverityEffects(unittest.TestCase):
    def two_studies(self, rates=DOUBLING) -> tuple[pd.DataFrame, pd.DataFrame]:
        parts = [severity_donors(rates, study) for study in ("a", "b")]
        return pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts])

    def test_recovers_a_doubling_per_step_and_from_mild_to_severe(self):
        effects = pe.severity_effects(*self.two_studies())
        # At rates this small the odds ratios are the rate ratios: two per step, and two from mild to severe.
        self.assertAlmostEqual(effects["trendPooled"]["ratio"], 2.0, delta=0.05)
        self.assertAlmostEqual(effects["severeVsMildPooled"]["ratio"], 2.0, delta=0.05)
        self.assertLess(effects["trendPooled"]["p"], 1e-6)
        first = effects["studies"][0]
        self.assertEqual(first["study"], "a")
        self.assertEqual(first["mild"]["donors"], 6)
        self.assertEqual(first["healthy"]["cells"], 600_000)
        # Six donors at 200 progenitors, jittered by -1, 0 and +1 twice over: 1,200 per 600,000 cells.
        self.assertEqual(first["severe"]["progenitors"], 1200)
        self.assertAlmostEqual(first["severe"]["per10k"], 20.0)

    def test_a_flat_course_pools_to_one(self):
        flat = {"healthy": 0.001, "mild": 0.001, "severe": 0.001}
        effects = pe.severity_effects(*self.two_studies(flat))
        self.assertAlmostEqual(effects["trendPooled"]["ratio"], 1.0, delta=0.02)
        self.assertAlmostEqual(effects["severeVsMildPooled"]["ratio"], 1.0, delta=0.02)

    def test_a_study_short_of_donors_in_one_class_is_left_out(self):
        table, classes = self.two_studies()
        short = (table["study"] == "b") & table["donor_id"].isin(["mild4", "mild5"])
        effects = pe.severity_effects(table[~short], classes)
        self.assertEqual([study["study"] for study in effects["studies"]], ["a"])
        self.assertEqual(
            effects["tooFewDonors"], [{"study": "b", "healthy": 6, "mild": 4, "severe": 6}]
        )
        # One study left: no pool.
        self.assertIsNone(effects["trendPooled"])
        self.assertIsNone(effects["severeVsMildPooled"])

    def test_counts_the_newborns_of_each_class(self):
        table, classes = self.two_studies()
        table.loc[(table["study"] == "a") & (table["donor_id"] == "mild0"), "stage"] = pe.NEWBORN
        first = pe.severity_effects(table, classes)["studies"][0]
        self.assertEqual((first["healthy"]["newborns"], first["mild"]["newborns"]), (0, 1))

    def test_donors_of_other_diseases_and_other_studies_are_not_counted(self):
        table, classes = self.two_studies()
        flu = table.iloc[[0]].assign(donor_id="flu0", disease="influenza", progenitors=10_000)
        elsewhere = table.iloc[[0]].assign(study="z", donor_id="z0")
        effects = pe.severity_effects(pd.concat([table, flu, elsewhere]), classes)
        self.assertEqual([study["study"] for study in effects["studies"]], ["a", "b"])
        self.assertEqual(effects["studies"][0]["healthy"]["donors"], 6)

    def test_a_healthy_donor_not_classed_healthy_is_an_error(self):
        table, classes = self.two_studies()
        classes.loc[classes["donor_id"] == "healthy0", "severity"] = "mild"
        with self.assertRaisesRegex(ValueError, "healthy donor"):
            pe.severity_effects(table, classes)

    def test_a_donor_with_covid_and_no_class_is_an_error(self):
        table, classes = self.two_studies()
        classes.loc[classes["donor_id"] == "severe0", "severity"] = None
        with self.assertRaisesRegex(ValueError, "no severity class"):
            pe.severity_effects(table, classes)

    def test_a_donor_without_a_record_is_an_error(self):
        table, classes = self.two_studies()
        with self.assertRaisesRegex(ValueError, "without a severity record"):
            pe.severity_effects(table, classes[classes["donor_id"] != "mild3"])


class Pooled(unittest.TestCase):
    def test_one_study_is_no_pool(self):
        self.assertIsNone(pe.pooled([{"effect": 0.5, "standardError": 0.1}]))
        pooled = pe.pooled(
            [{"effect": 0.5, "standardError": 0.1}, {"effect": 0.5, "standardError": 0.2}]
        )
        self.assertAlmostEqual(pooled["effect"], 0.5)
        # Equal effects pool to the same effect: the odds ratio is e^0.5 and its interval brackets it.
        self.assertAlmostEqual(pooled["ratio"], 1.6487, places=4)
        self.assertLess(pooled["ratioLow"], pooled["ratio"])
        self.assertGreater(pooled["ratioHigh"], pooled["ratio"])
        self.assertAlmostEqual(pooled["ratioLow"], np.exp(pooled["low"]))

    def test_a_pool_at_the_two_sided_five_percent_quantile_has_p_one_in_twenty(self):
        # Two equal studies pool to their effect with their error over root two; a pooled effect 1.959964 errors from
        # zero is the normal distribution's two-sided 5% point.
        error = 0.2 / 1.959964 * np.sqrt(2)
        pooled = pe.pooled(
            [{"effect": 0.2, "standardError": error}, {"effect": 0.2, "standardError": error}]
        )
        self.assertAlmostEqual(pooled["p"], 0.05, places=6)
        # The sign of the effect does not change the two-sided P value.
        negative = pe.pooled(
            [{"effect": -0.2, "standardError": error}, {"effect": -0.2, "standardError": error}]
        )
        self.assertAlmostEqual(negative["p"], pooled["p"])


class Estimate(unittest.TestCase):
    def test_one_study_gives_its_own_ratio_interval_and_p(self):
        # An effect 1.959964 errors from zero is the normal distribution's two-sided 5% point, and its interval's
        # lower end sits at an odds ratio of exactly 1.
        estimate = pe.estimate([{"effect": 0.4, "standardError": 0.4 / 1.959964}])
        self.assertAlmostEqual(estimate["p"], 0.05, places=6)
        self.assertAlmostEqual(estimate["ratio"], np.exp(0.4))
        self.assertAlmostEqual(estimate["ratioLow"], 1.0)
        self.assertAlmostEqual(estimate["ratioHigh"], np.exp(0.8))
        self.assertEqual((estimate["studies"], estimate["tau2"]), (1, 0.0))

    def test_several_studies_give_their_pool(self):
        effects = [{"effect": 0.5, "standardError": 0.1}, {"effect": 0.3, "standardError": 0.2}]
        self.assertEqual(pe.estimate(effects), pe.pooled(effects))

    def test_no_study_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            pe.estimate([])


def disease_donors(
    study: str, disease: str, rate: float, n: int = 12, cells: int = 100_000
) -> list[dict]:
    """``n`` donors of a study with a disease at a progenitor rate, jittered by -1, 0 and +1 so the mean stays put."""
    return [
        {
            "study": study,
            "donor_id": f"{disease}{i}",
            "sample": "",
            "disease": disease,
            "cells": cells,
            "progenitors": int(round(rate * cells)) + (i % 3) - 1,
        }
        for i in range(n)
    ]


class DiseaseEffects(unittest.TestCase):
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                *disease_donors("c", "normal", 0.0005),
                *disease_donors("c", "COVID-19", 0.0010),
                *disease_donors("c", "sepsis", 0.0020),
                *disease_donors("d", "normal", 0.0010),
                *disease_donors("d", "COVID-19", 0.0020),
                # Too few donors with lupus to contrast, so lupus has no estimate at all.
                *disease_donors("d", "systemic lupus erythematosus", 0.0040, n=4),
                # Donors with a disease but no healthy donors in their study, as in a study of patients only.
                *disease_donors("e", "lupus nephritis", 0.0040),
            ]
        )

    def test_each_disease_is_set_against_the_healthy_donors_of_its_own_study(self):
        effects = pe.disease_effects(self.table())
        self.assertEqual(list(effects), ["COVID-19", "sepsis"])
        # COVID-19 doubles the rate in both studies, which pool to two; sepsis, in one study, quadruples it.
        covid, sepsis = effects["COVID-19"], effects["sepsis"]
        self.assertEqual([e["study"] for e in covid["studies"]], ["c", "d"])
        self.assertAlmostEqual(covid["estimate"]["ratio"], 2.0, delta=0.05)
        self.assertEqual(covid["estimate"]["studies"], 2)
        self.assertAlmostEqual(sepsis["estimate"]["ratio"], 4.0, delta=0.1)
        self.assertEqual(sepsis["estimate"]["studies"], 1)
        self.assertLess(sepsis["estimate"]["ratioLow"], sepsis["estimate"]["ratio"])

    def test_each_disease_counts_the_donors_its_contrasts_compare(self):
        # Study c holds 12 healthy donors and 7 with COVID-19, study d 5 and 9; study e has no healthy donors, so
        # its donors with COVID-19 enter no contrast and no total.
        table = pd.DataFrame(
            [
                *disease_donors("c", "normal", 0.0005),
                *disease_donors("c", "COVID-19", 0.0010, n=7),
                *disease_donors("d", "normal", 0.0010, n=5),
                *disease_donors("d", "COVID-19", 0.0020, n=9),
                *disease_donors("e", "COVID-19", 0.0020, n=8),
            ]
        )
        covid = pe.disease_effects(table)["COVID-19"]
        self.assertEqual((covid["cases"], covid["healthy"]), (16, 17))
        self.assertEqual(
            (covid["cases"], covid["healthy"]),
            (
                sum(e["cases"] for e in covid["studies"]),
                sum(e["healthy"] for e in covid["studies"]),
            ),
        )

    def test_donors_of_another_study_are_never_the_controls(self):
        # Healthy donors only in study d leave study c's sepsis donors without controls.
        table = self.table()
        table = table[~((table["study"] == "c") & (table["disease"] == "normal"))]
        effects = pe.disease_effects(table)
        self.assertNotIn("sepsis", effects)
        self.assertEqual([e["study"] for e in effects["COVID-19"]["studies"]], ["d"])


def combat_donors(
    cohorts: dict[str, tuple[float, int]], cells: int = 100_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """COMBAT's donor table (study "combat") and Source records, with ``n`` donors per Source value at a progenitor
    rate, each donor's disease the store's for its cohort."""
    rows, values = [], []
    for value, (rate, n) in cohorts.items():
        for i in range(n):
            donor = f"{value}{i}"
            rows.append(
                {
                    "study": "combat",
                    "donor_id": donor,
                    "sample": f"{donor}s",
                    "disease": pe.COMBAT_DISEASES[pe.COMBAT_COHORTS[value]],
                    "cells": cells,
                    "progenitors": int(round(rate * cells)) + (i % 3) - 1,
                }
            )
            values.append(severity_row(donor, f"{donor}s", value))
    return pd.DataFrame(rows), pd.DataFrame(values)


# Healthy volunteers, then each step of COVID-19 doubling the rate; sepsis at four times healthy; three donors with
# influenza.
COMBAT_RATES = {
    "HV": (0.0005, 6),
    "COVID_MILD": (0.0010, 6),
    "COVID_SEV": (0.0020, 6),
    "COVID_CRIT": (0.0040, 6),
    "Sepsis": (0.0020, 6),
    "Flu": (0.0050, 3),
}


class CombatCohorts(unittest.TestCase):
    def test_recovers_a_doubling_per_step_and_sepsis_at_four_times_healthy(self):
        effects = pe.combat_cohorts(*combat_donors(COMBAT_RATES), "combat")
        # At rates this small the odds ratios are the rate ratios.
        self.assertAlmostEqual(effects["trend"]["ratio"], 2.0, delta=0.05)
        self.assertAlmostEqual(effects["sepsis"]["ratio"], 4.0, delta=0.1)
        self.assertLess(effects["trend"]["p"], 1e-6)
        self.assertEqual(
            [(g["cohort"], len(g["donors"]), round(g["per10k"], 6)) for g in effects["cohorts"]],
            [
                ("healthy", 6, 5.0),
                ("mild", 6, 10.0),
                ("severe", 6, 20.0),
                ("critical", 6, 40.0),
                ("sepsis", 6, 20.0),
            ],
        )
        # Each donor keeps its own cells and progenitors: the healthy volunteers' 50, jittered by -1, 0 and +1.
        self.assertEqual(
            [d["progenitors"] for d in effects["cohorts"][0]["donors"]], [49, 50, 51, 49, 50, 51]
        )
        self.assertEqual(effects["tooFewDonors"], [{"cohort": "influenza", "donors": 3}])

    def test_both_mild_cohorts_join_one_group(self):
        rates = {**COMBAT_RATES, "COVID_HCW_MILD": (0.0010, 3)}
        effects = pe.combat_cohorts(*combat_donors(rates), "combat")
        (mild,) = [g for g in effects["cohorts"] if g["cohort"] == "mild"]
        self.assertEqual(len(mild["donors"]), 9)

    def test_donors_of_other_studies_are_not_counted(self):
        table, values = combat_donors(COMBAT_RATES)
        elsewhere = table.iloc[[0]].assign(study="other", donor_id="x0", progenitors=90_000)
        effects = pe.combat_cohorts(pd.concat([table, elsewhere]), values, "combat")
        self.assertEqual(len(effects["cohorts"][0]["donors"]), 6)
        self.assertAlmostEqual(effects["cohorts"][0]["per10k"], 5.0)

    def test_a_sample_without_its_cohort_is_an_error(self):
        table, values = combat_donors(COMBAT_RATES)
        with self.assertRaisesRegex(ValueError, "no 'Source' value"):
            pe.combat_cohorts(table, values[values["donor_id"] != "Sepsis0"], "combat")

    def test_a_cohort_the_map_does_not_name_is_an_error(self):
        table, values = combat_donors(COMBAT_RATES)
        values.loc[values["donor_id"] == "HV0", "value"] = "COVID_NEW"
        with self.assertRaisesRegex(ValueError, "does not name"):
            pe.combat_cohorts(table, values, "combat")

    def test_a_sepsis_donor_still_labelled_covid_is_an_error(self):
        # The store labels the sepsis cohort COVID-19; without COHORT_RECORDS applied the cohorts would mix.
        table, values = combat_donors(COMBAT_RATES)
        table.loc[table["donor_id"] == "Sepsis0", "disease"] = "COVID-19"
        with self.assertRaisesRegex(ValueError, "disease is not their cohort"):
            pe.combat_cohorts(table, values, "combat")

    def test_too_few_donors_for_a_step_of_the_trend_is_an_error(self):
        rates = {**COMBAT_RATES, "COVID_CRIT": (0.0040, 4)}
        with self.assertRaisesRegex(ValueError, "too few donors in \\['critical'\\]"):
            pe.combat_cohorts(*combat_donors(rates), "combat")

    def test_a_study_without_donors_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no donor of COMBAT"):
            pe.combat_cohorts(*combat_donors(COMBAT_RATES), "elsewhere")


class DiseaseMultiplicity(unittest.TestCase):
    def test_each_disease_carries_its_p_value_adjusted_across_the_diseases(self):
        # Two diseases, m = 2: Benjamini-Hochberg gives the smaller P value min(2p, the larger) and the larger itself;
        # Holm gives the smaller 2p and the larger max(itself, the smaller's 2p).
        table = donors_with(0.0010, 0.0005)
        steady = table[table["disease"] == "COVID-19"].assign(disease="lupus")
        steady["donor_id"] = steady["donor_id"].str.replace("case", "lupus")
        steady["progenitors"] = table.loc[table["disease"] == "normal", "progenitors"].to_numpy()
        diseases = pe.disease_effects(pd.concat([table, steady]))
        covid, lupus = diseases["COVID-19"]["estimate"], diseases["lupus"]["estimate"]
        self.assertLess(covid["p"], lupus["p"])
        self.assertAlmostEqual(covid["bhAdjustedP"], min(2 * covid["p"], lupus["p"]))
        self.assertAlmostEqual(lupus["bhAdjustedP"], lupus["p"])
        self.assertAlmostEqual(covid["holmAdjustedP"], 2 * covid["p"])
        self.assertAlmostEqual(lupus["holmAdjustedP"], max(lupus["p"], 2 * covid["p"]))


def study_contrast(effect: float, error: float) -> dict:
    return {"study": "s", "effect": effect, "standardError": error}


class CovidAgainstChronic(unittest.TestCase):
    def diseases(self, chronic_contrasts: int = 2) -> dict:
        # Sepsis and the respiratory disorder sit far above the line and would pull the pool up if it took them.
        diseases = {
            "COVID-19": {"estimate": {"effect": 0.6, "standardError": 0.2}, "studies": []},
            "sepsis": {"studies": [study_contrast(3.0, 0.1)]},
            "respiratory system disorder": {"studies": [study_contrast(3.0, 0.1)]},
        }
        for i in range(chronic_contrasts):
            diseases[f"chronic {i}"] = {"studies": [study_contrast(0.0, 0.1)]}
        return diseases

    def test_covid_is_contrasted_with_the_chronic_pool_alone(self):
        # The two chronic contrasts pool to a log odds ratio of 0 with error 0.1 / sqrt 2; COVID-19's 0.6 lies
        # 0.6 / sqrt(0.2^2 + 0.005) = 2 sqrt 2 errors away, whose two-sided P value is erfc(2).
        result = pe.covid_against_chronic(self.diseases())
        self.assertEqual(result["diseases"], ["chronic 0", "chronic 1"])
        self.assertEqual(result["contrasts"], 2)
        self.assertAlmostEqual(result["pool"]["ratio"], 1.0)
        self.assertAlmostEqual(result["ratioOfOddsRatios"], np.exp(0.6))
        self.assertAlmostEqual(result["p"], 0.004677734981047266, places=12)
        error = np.sqrt(0.045)
        self.assertAlmostEqual(result["low"], np.exp(0.6 - 1.959964 * error))
        self.assertAlmostEqual(result["high"], np.exp(0.6 + 1.959964 * error))

    def test_one_chronic_contrast_is_no_pool(self):
        with self.assertRaisesRegex(ValueError, "fewer than two study contrasts"):
            pe.covid_against_chronic(self.diseases(chronic_contrasts=1))


def unit(store, donor, sample, cell_type, cells, library):
    return {
        "storeId": store,
        "donor_id": donor,
        "sample": sample,
        "coarse_cell_type": cell_type,
        "cells": float(cells),
        "librarySize": float(library),
    }


class LymphocyteDepth(unittest.TestCase):
    def test_sums_the_lymphocytes_of_a_study_across_its_stores(self):
        # Donor d1's sample, missing in the totals and empty in the screen, has T cells in one store and B cells in
        # another of the same collection: (100,000 + 25,000) / 150 UMIs per cell; its monocytes do not count. Donor d2's
        # sample is named "nan" in both tables. A store the totals do not hold adds nothing.
        units = pd.DataFrame(
            [
                unit("s1", "d1", "", "T cells", 100, 100_000),
                unit("s2", "d1", "", "B cells", 50, 25_000),
                unit("s1", "d1", "", "Monocytes", 1000, 9_000_000),
                unit("s1", "d2", "nan", "NK and ILC", 10, 4_000),
                unit("other", "d1", "", "T cells", 10, 1e9),
            ]
        )
        totals = pd.DataFrame(
            {
                "storeId": ["s1", "s2", "s1"],
                "collectionId": ["c", "c", "c"],
                "donor_id": ["d1", "d1", "d2"],
                "sample": [None, None, "nan"],
            }
        )
        depth = pe.lymphocyte_depth(units, totals).set_index("donor_id")
        self.assertAlmostEqual(depth.loc["d1", "logDepth"], np.log(125_000 / 150))
        self.assertAlmostEqual(depth.loc["d2", "logDepth"], np.log(400.0))
        self.assertEqual(set(depth["study"]), {"c"})
        self.assertEqual(depth.loc["d2", "sample"], "nan")


def depth_confounded_donors(study="c", n=12, sexes=("female", "male"), cells=100_000):
    """Donors whose progenitor rate follows their lymphocyte depth alone, e^depth times 5 per 10,000 cells, and whose
    COVID-19 cases sit 0.5 deeper than their healthy donors: no effect of COVID-19 once depth is in the model."""
    rows, depth = [], []
    for i in range(n):
        for disease, shift in (("COVID-19", 0.5), ("normal", 0.0)):
            donor = f"{disease}{i}"
            value = shift + 0.1 * (i % 4 - 1.5)
            rows.append(
                {
                    "study": study,
                    "donor_id": donor,
                    "sample": "",
                    "disease": disease,
                    "cells": cells,
                    "progenitors": int(round(0.0005 * np.exp(value) * cells)),
                    "sex": sexes[i % len(sexes)],
                }
            )
            depth.append({"study": study, "donor_id": donor, "sample": "", "logDepth": value})
    return pd.DataFrame(rows), pd.DataFrame(depth)


class CovidCovariateModels(unittest.TestCase):
    def test_adjusting_for_depth_removes_an_effect_that_depth_alone_makes(self):
        table, depth = depth_confounded_donors()
        (plain,) = pe.contrast_effects(table, "COVID-19")
        self.assertAlmostEqual(plain["effect"], 0.5, delta=0.02)
        models = pe.covid_covariate_models(table, depth)
        (by_depth,) = models["lymphocyteDepthAdjusted"]["studyEffects"]
        (by_both,) = models["lymphocyteDepthAndSexAdjusted"]["studyEffects"]
        self.assertAlmostEqual(by_depth["effect"], 0.0, delta=0.02)
        self.assertAlmostEqual(by_both["effect"], 0.0, delta=0.02)
        self.assertEqual(models["lymphocyteDepthAdjusted"]["donors"], 24)
        self.assertEqual(models["lymphocyteDepthAdjusted"]["estimate"]["studies"], 1)

    def test_a_study_short_of_donors_of_recorded_sex_leaves_only_the_sex_model(self):
        # Study d records sex for 4 of its 12 donors on each side, 2 female and 2 male: in the depth model, not in the
        # depth and sex model.
        table_c, depth_c = depth_confounded_donors("c")
        table_d, depth_d = depth_confounded_donors(
            "d", sexes=("female", "male", "unknown", "unknown", "unknown", "unknown")
        )
        models = pe.covid_covariate_models(
            pd.concat([table_c, table_d]), pd.concat([depth_c, depth_d])
        )
        self.assertEqual(
            [e["study"] for e in models["lymphocyteDepthAdjusted"]["studyEffects"]], ["c", "d"]
        )
        self.assertEqual(
            [e["study"] for e in models["lymphocyteDepthAndSexAdjusted"]["studyEffects"]], ["c"]
        )
        self.assertEqual(models["lymphocyteDepthAndSexAdjusted"]["donors"], 24)

    def test_a_contrast_donor_without_depth_or_of_one_sex_only_is_an_error(self):
        table, depth = depth_confounded_donors()
        with self.assertRaisesRegex(
            ValueError, "1 donors of the COVID-19 contrast have no lymphocyte depth"
        ):
            pe.covid_covariate_models(table, depth.iloc[1:])
        table, depth = depth_confounded_donors(sexes=("female",))
        with self.assertRaisesRegex(ValueError, "every donor of recorded sex"):
            pe.covid_covariate_models(table, depth)


@unittest.skipUnless(
    os.environ.get("CYTEARC_SLOW_TESTS"),
    "set CYTEARC_SLOW_TESTS=1 to summarise the progenitor records (about 25 s, 2 GB)",
)
class ProgenitorRecords(unittest.TestCase):
    def test_the_records_give_the_papers_adjusted_and_contrasted_covid_estimates(self):
        # Paragraphs 107 and 108: COVID-19 q = 0.049 by Benjamini-Hochberg and by Holm across nine diseases, sepsis
        # q = 0.15; 1.95-fold (1.33 to 2.87, P = 0.00068) over the chronic pool of 0.86 (0.77 to 0.97) from nine
        # contrasts of six diseases; 1.66 (1.14 to 2.42, P = 0.0088) over 459 donors of eight studies adjusted for
        # depth, and 1.87 (1.25 to 2.79, P = 0.0022) over 395 donors of six studies adjusted for depth and sex.
        summary = pe.summarise(pe.RECORDS)
        diseases = summary["diseases"]
        self.assertEqual(len(diseases), 9)
        covid = diseases["COVID-19"]["estimate"]
        self.assertEqual(
            (round(covid["bhAdjustedP"], 3), round(covid["holmAdjustedP"], 3)), (0.049, 0.049)
        )
        self.assertEqual(round(diseases["sepsis"]["estimate"]["bhAdjustedP"], 2), 0.15)
        chronic = summary["covidAgainstChronic"]
        self.assertEqual((len(chronic["diseases"]), chronic["contrasts"]), (6, 9))
        pool = chronic["pool"]
        self.assertEqual(
            [round(pool[k], 2) for k in ("ratio", "ratioLow", "ratioHigh")], [0.86, 0.77, 0.97]
        )
        self.assertEqual(
            [round(chronic[k], 2) for k in ("ratioOfOddsRatios", "low", "high")], [1.95, 1.33, 2.87]
        )
        self.assertEqual(round(chronic["p"], 5), 0.00068)
        models = summary["covidCovariateModels"]
        for name, studies, donors, expected in (
            ("lymphocyteDepthAdjusted", 8, 459, [1.66, 1.14, 2.42]),
            ("lymphocyteDepthAndSexAdjusted", 6, 395, [1.87, 1.25, 2.79]),
        ):
            model = models[name]
            self.assertEqual((model["estimate"]["studies"], model["donors"]), (studies, donors))
            self.assertEqual(
                [round(model["estimate"][k], 2) for k in ("ratio", "ratioLow", "ratioHigh")],
                expected,
            )
        self.assertEqual(round(models["lymphocyteDepthAdjusted"]["estimate"]["p"], 4), 0.0088)
        self.assertEqual(round(models["lymphocyteDepthAndSexAdjusted"]["estimate"]["p"], 4), 0.0022)


class InterferonUnits(unittest.TestCase):
    def test_a_donor_table_other_than_the_one_its_job_recorded_is_an_error(self):
        import gzip
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "screen.json"
            table = Path(folder) / "screen_donors.csv.gz"
            table.write_bytes(
                gzip.compress(
                    b"storeId,donor_id,sample,coarse_cell_type,cells,librarySize,disease\n"
                    b"s1,d1,nan,T cells,10,4000,normal\n"
                )
            )
            digest = hashlib.sha256(table.read_bytes()).hexdigest()
            record = {"status": "completed", "result": {"tables": {"donors": {"sha256": digest}}}}
            run.write_text(json.dumps(record))
            with mock.patch.object(pe, "INTERFERON_RUNS", (run,)):
                units = pe.interferon_units()
                self.assertEqual(units.loc[0, "sample"], "nan")
                self.assertEqual(units.loc[0, "librarySize"], 4000.0)
                record["result"]["tables"]["donors"]["sha256"] = "0" * 64
                run.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "not the donor table its job recorded"):
                    pe.interferon_units()


def pooled_study(name: str, effect: float, error: float) -> dict:
    return {"study": name, "effect": effect, "standardError": error}


class WeightSharesTest(unittest.TestCase):
    def test_each_disease_takes_its_contrasts_inverse_variance_share(self):
        # Without heterogeneity the weights are 1/0.25 = 4, 1/1 = 1 and 1/0.25 = 4: the two contrasts of one disease
        # hold 8 of 9.
        effects = [
            pooled_study("a", 0.1, 0.5),
            pooled_study("b", 0.2, 1.0),
            pooled_study("c", 0.3, 0.5),
        ]
        shares = pe.weight_shares(effects, ["cmv", "lupus", "cmv"], tau2=0.0)
        self.assertAlmostEqual(shares["cmv"], 8 / 9, places=12)
        self.assertAlmostEqual(shares["lupus"], 1 / 9, places=12)
        # Heterogeneity evens the weights: with tau2 0.75 they are 1, 0.5714 and 1.
        even = pe.weight_shares(effects, ["cmv", "lupus", "cmv"], tau2=0.75)
        self.assertAlmostEqual(even["cmv"], 2 / (2 + 1 / 1.75), places=12)

    def test_a_disease_list_that_does_not_match_the_contrasts_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "2 study contrasts but 1 diseases"):
            pe.weight_shares([pooled_study("a", 0, 1), pooled_study("b", 0, 1)], ["cmv"], 0.0)


class LeaveOneOutTest(unittest.TestCase):
    def test_leaving_out_the_outlier_removes_the_heterogeneity(self):
        # Three studies agree at 0.5 and one sits at 3.0, all with standard error 0.2: together they are heterogeneous,
        # without the outlier the DerSimonian-Laird variance is 0, and without any other study it stays above 0.
        effects = [
            pooled_study("a", 0.5, 0.2),
            pooled_study("b", 0.5, 0.2),
            pooled_study("c", 0.5, 0.2),
            pooled_study("out", 3.0, 0.2),
        ]
        result = {r["leftOut"]: r for r in pe.leave_one_out(effects)}
        self.assertEqual(list(result), ["a", "b", "c", "out"])
        self.assertEqual(result["out"]["tau2"], 0.0)
        self.assertAlmostEqual(result["out"]["ratio"], math.exp(0.5), places=12)
        self.assertGreater(result["a"]["tau2"], 1.0)
        self.assertEqual(result["a"]["studies"], 3)

    def test_two_studies_give_no_leave_one_out_pool(self):
        self.assertEqual(pe.leave_one_out([pooled_study("a", 0, 1), pooled_study("b", 1, 1)]), [])


if __name__ == "__main__":
    unittest.main()
