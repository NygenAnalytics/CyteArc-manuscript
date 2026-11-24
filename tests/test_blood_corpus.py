"""The blood corpus experiment's pure steps, checked against values derived by hand."""

import unittest

import numpy as np
import pandas as pd
from experiments.reuse import blood_corpus as bc


class GeneIdFormat(unittest.TestCase):
    def test_counts_each_kind_of_id(self):
        ids = [
            "ENSG00000000003",
            "ENSG00000000005.5",
            "GRCh38___ENSG00000000419",
            "MIR1302-2HG",
            "ENSG0000000000",
        ]
        self.assertEqual(
            bc.id_format(np.array(ids)),
            {
                "features": 5,
                "plainEnsembl": 1,
                "versionedEnsembl": 1,
                "embeddedEnsembl": 1,
                "other": 2,
            },
        )


class GeneSharing(unittest.TestCase):
    def store(self, name, features, plain=None, cells=10):
        plain = len(features) if plain is None else plain
        return {
            "storeId": name,
            "selectedCells": cells,
            "geneIds": {"features": len(features), "plainEnsembl": plain},
        }

    def test_keeps_the_exact_id_intersection_and_names_what_each_store_costs(self):
        features = {
            "a": np.array(["g1", "g2", "g3", "g4"]),
            "b": np.array(["g1", "g2", "g3"]),
            "c": np.array(["g1", "g2", "g4"]),
        }
        stores = [self.store(name, ids) for name, ids in features.items()]
        result = bc.gene_sharing(features, stores, {"minimumGenes": 3})
        self.assertEqual(result["sharedGeneIds"], ["g1", "g2"])
        # Without b the others share g1, g2, g4; without c they share g1, g2, g3; without a nothing changes.
        self.assertEqual(result["genesLostByStore"], {"b": 1, "c": 1, "a": 0})
        self.assertEqual(result["excluded"], {})

    def test_leaves_out_small_panels_and_non_ensembl_ids_before_intersecting(self):
        features = {
            "a": np.array(["g1", "g2", "g3"]),
            "b": np.array(["g1", "g2", "g3"]),
            "panel": np.array(["g1"]),
            "named": np.array(["x1", "x2", "x3"]),
        }
        stores = [
            self.store("a", features["a"]),
            self.store("b", features["b"]),
            self.store("panel", features["panel"]),
            self.store("named", features["named"], plain=0),
        ]
        result = bc.gene_sharing(features, stores, {"minimumGenes": 3})
        self.assertEqual(sorted(result["excluded"]), ["named", "panel"])
        self.assertEqual(result["kept"], ["a", "b"])
        self.assertEqual(result["sharedGenes"], 3)

    def test_leaves_out_stores_without_selected_cells(self):
        features = {"a": np.array(["g1", "g2"]), "empty": np.array(["g3", "g4"])}
        stores = [self.store("a", features["a"]), self.store("empty", features["empty"], cells=0)]
        result = bc.gene_sharing(features, stores, {"minimumGenes": 2})
        self.assertEqual(result["kept"], ["a"])
        self.assertEqual(result["noSelectedCells"], ["empty"])


class RandomEffects(unittest.TestCase):
    def test_matches_a_hand_derived_dersimonian_laird_pool(self):
        # Weights 100 each, fixed effect 0.3, Q = 100 * (0.04 + 0 + 0.04) = 8, tau2 = (8 - 2) / (300 - 100) = 0.03,
        # pooled weights 1 / 0.04 = 25 each, so the pooled effect is 0.3 with standard error sqrt(1 / 75).
        pooled = bc.random_effects(np.array([0.1, 0.3, 0.5]), np.array([0.1, 0.1, 0.1]))
        self.assertAlmostEqual(pooled["q"], 8.0)
        self.assertAlmostEqual(pooled["tau2"], 0.03)
        self.assertAlmostEqual(pooled["effect"], 0.3)
        self.assertAlmostEqual(pooled["standardError"], np.sqrt(1 / 75))
        self.assertAlmostEqual(pooled["high"] - pooled["low"], 2 * 1.959964 * np.sqrt(1 / 75))

    def test_homogeneous_studies_pool_to_the_fixed_effect(self):
        pooled = bc.random_effects(np.array([0.2, 0.2]), np.array([0.1, 0.2]))
        self.assertEqual(pooled["tau2"], 0.0)
        self.assertAlmostEqual(pooled["effect"], 0.2)
        self.assertAlmostEqual(pooled["standardError"], np.sqrt(1 / (100 + 25)))

    def test_one_study_pools_to_itself(self):
        pooled = bc.random_effects(np.array([-0.4]), np.array([0.05]))
        self.assertAlmostEqual(pooled["effect"], -0.4)
        self.assertAlmostEqual(pooled["standardError"], 0.05)
        self.assertEqual(pooled["tau2"], 0.0)


class StudyEffect(unittest.TestCase):
    def test_recovers_a_known_log_odds_slope(self):
        decades = np.repeat(np.arange(2.0, 9.0), 4)
        lineage = np.full(len(decades), 1_000_000)
        state = np.round(lineage / (1 + np.exp(-(0.5 - 0.28 * decades)))).astype(int)
        donors = pd.DataFrame({"decades": decades, "lineage": lineage, "state": state})
        effect = bc.study_effect(donors, "decades")
        self.assertAlmostEqual(effect["effect"], -0.28, places=3)
        self.assertEqual(effect["donors"], len(decades))

    def test_overdispersion_widens_the_standard_error(self):
        rng = np.random.default_rng(1)
        group = np.repeat([0.0, 1.0], 30)
        lineage = np.full(60, 5000)
        noisy = 1 / (1 + np.exp(-(-1.0 + 0.8 * group + rng.normal(0, 0.5, 60))))
        donors = pd.DataFrame(
            {"case": group, "lineage": lineage, "state": rng.binomial(lineage, noisy)}
        )
        effect = bc.study_effect(donors, "case")
        # Donor-to-donor spread of 0.5 on the logit gives a Pearson dispersion of about 5000 * 0.5**2 * p(1 - p).
        self.assertGreater(effect["dispersion"], 100)
        # The rescaled error is that of a two-group comparison of 30 donors each with spread about 0.5: near 0.13.
        self.assertGreater(effect["standardError"], 0.08)
        self.assertLess(effect["standardError"], 0.2)
        self.assertLess(abs(effect["effect"] - 0.8), 3 * effect["standardError"])


class Lineages(unittest.TestCase):
    def test_assigns_clusters_by_marker_genes_and_leaves_unmarked_ones_out(self):
        genes = bc.marker_genes()
        column = {gene: i for i, gene in enumerate(genes)}
        values = np.zeros((9, len(genes)), dtype=np.float32)
        for row in range(3):
            values[row, [column[g] for g in ("CD3E", "CD8A", "CD8B")]] = 3.0
        for row in range(3, 6):
            values[row, [column[g] for g in ("CD14", "LYZ")]] = 3.0
        clusters = np.repeat([0, 1, 2], 3)
        lineage, table = bc.assign_lineages(values, genes, clusters)
        self.assertEqual(list(lineage), ["CD8 T"] * 3 + ["monocytes"] * 3 + ["other"] * 3)
        self.assertEqual(table["2"]["lineage"], "other")

    def test_a_store_without_a_gene_does_not_count_as_zero_expression(self):
        genes = bc.marker_genes()
        column = {gene: i for i, gene in enumerate(genes)}
        # Ten clusters of one cell: 0 and 1 express CD3E, CD8A and CD8B at 3, but cluster 1's store lacks CD8B.
        values = np.zeros((10, len(genes)), dtype=np.float32)
        values[:2, [column["CD3E"], column["CD8A"], column["CD8B"]]] = 3.0
        values[1, column["CD8B"]] = np.nan
        lineage, table = bc.assign_lineages(values, genes, np.arange(10))
        # CD3E and CD8A read [3, 3, 0, ...]: mean 0.6, standard deviation 1.2, so z = 2 for clusters 0 and 1. Cluster
        # 1's CD8 T score is the mean over its known genes, 2; counting the missing CD8B as zero would give
        # (2 + 2 - 1/3) / 3, since CD8B would then read [3, 0, 0, ...] with z = -1/3 at zero.
        self.assertAlmostEqual(table["1"]["scores"]["CD8 T"], 2.0, places=5)
        self.assertEqual(list(lineage[:2]), ["CD8 T", "CD8 T"])


class States(unittest.TestCase):
    def test_cuts_at_the_healthy_donors_quantile_inside_the_lineage(self):
        genes = bc.marker_genes()
        column = {gene: i for i, gene in enumerate(genes)}
        n = 8
        values = np.zeros((n, len(genes)), dtype=np.float32)
        values[:, column["KLRC2"]] = np.arange(n, dtype=np.float32)
        lineage = np.array(["NK"] * n, dtype=object)
        healthy = np.array([True] * 4 + [False] * 4)
        calls, scored, cuts = bc.call_states(values, genes, lineage, healthy)
        # Healthy scores are 0, 1, 2, 3 (KLRC2 less FCER1G at zero); their 75th percentile is 2.25.
        self.assertAlmostEqual(cuts["adaptive NK"]["cut"], 2.25)
        self.assertEqual(
            list(calls["adaptive NK"]), [False, False, False, True, True, True, True, True]
        )

    def test_cells_without_a_states_genes_are_not_scored(self):
        genes = bc.marker_genes()
        column = {gene: i for i, gene in enumerate(genes)}
        values = np.ones((4, len(genes)), dtype=np.float32)
        values[2:, column["KLRC2"]] = np.nan
        calls, scored, cuts = bc.call_states(
            values, genes, np.array(["NK"] * 4, dtype=object), np.array([True] * 4)
        )
        self.assertEqual(cuts["adaptive NK"]["scoredCells"], 2)
        # The two cells without KLRC2 leave the state's denominator as well as its count.
        self.assertEqual(list(scored["adaptive NK"]), [True, True, False, False])
        self.assertEqual(list(calls["adaptive NK"][2:]), [False, False])

    def test_a_state_without_healthy_reference_cells_is_left_out(self):
        genes = bc.marker_genes()
        values = np.ones((4, len(genes)), dtype=np.float32)
        calls, scored, cuts = bc.call_states(
            values, genes, np.array(["NK"] * 4, dtype=object), np.array([False] * 4)
        )
        self.assertNotIn("adaptive NK", calls)
        self.assertIsNone(cuts["adaptive NK"]["cut"])


def donor_rows(
    study, ages, rng, slope=0.0, disease="normal", case_effect=0.0, scored=20000, lineage=None
):
    """Synthetic donors of one study for naive CD8 T: a logit share of 0.5 + slope per decade (+ case_effect for a
    disease) with donor-level spread 0.1, drawn binomially from the scored CD8 T cells."""
    rows = []
    for i, age in enumerate(ages):
        logit = 0.5 + slope * age / 10 + case_effect + rng.normal(0, 0.1)
        rows.append(
            {
                "study": study,
                "dataset": study,
                "orig_disease": disease,
                "orig_sex": "male" if i % 2 else "female",
                "orig_development_stage": f"{age}-year-old stage",
                "cells_CD8 T": scored if lineage is None else lineage,
                "scored_naive CD8 T": scored,
                "state_naive CD8 T": int(rng.binomial(scored, 1 / (1 + np.exp(-logit))))
                if scored
                else 0,
            }
        )
    return rows


class DonorAges(unittest.TestCase):
    def test_reads_ages_in_years_only(self):
        ages = bc.donor_ages(
            pd.Series(
                ["45-year-old stage", "fifth decade stage", "80 year-old and over stage", "unknown"]
            )
        )
        self.assertEqual(ages.iloc[0], 45)
        self.assertTrue(ages.iloc[1:].isna().all())


class Decision(unittest.TestCase):
    def test_needs_the_interval_and_two_thirds_of_studies_in_the_expected_direction(self):
        positive = {"low": 0.05, "high": 0.9}
        self.assertFalse(bc.decision(np.array([0.5, -0.1, -0.2]), positive, 1)["recovered"])
        self.assertTrue(bc.decision(np.array([0.5, 0.3, -0.1]), positive, 1)["recovered"])
        self.assertAlmostEqual(
            bc.decision(np.array([0.5, 0.3, -0.1]), positive, 1)["agreement"], 2 / 3
        )
        # An interval that excludes zero in the other direction is not a recovery.
        self.assertFalse(bc.decision(np.array([0.5, 0.3]), positive, -1)["recovered"])
        self.assertTrue(
            bc.decision(np.array([-0.5, -0.3]), {"low": -0.9, "high": -0.05}, -1)["recovered"]
        )
        self.assertFalse(
            bc.decision(np.array([0.5, 0.3]), {"low": -0.05, "high": 0.9}, 1)["recovered"]
        )


class EffectEstimates(unittest.TestCase):
    def test_recovers_an_age_slope_and_says_why_studies_are_left_out(self):
        rng = np.random.default_rng(7)
        rows = [
            *donor_rows("s1", range(20, 70, 2), rng, slope=-0.3),
            *donor_rows("s2", range(20, 80, 2), rng, slope=-0.3),
            *donor_rows("small", range(20, 40, 2), rng, slope=-0.3),
            *donor_rows("nogenes", range(20, 70, 2), rng, scored=0, lineage=5000),
            # Not healthy, not adult, or without an age in years: none of these enters the age model.
            *donor_rows("s1", [50, 60], rng, disease="COVID-19", case_effect=3.0),
            *donor_rows("s1", [15, 16], rng, case_effect=3.0),
        ]
        donors = pd.DataFrame(rows)
        decade = donors.index[donors.orig_disease.eq("normal") & donors.study.eq("s2")][0]
        donors.loc[decade, "orig_development_stage"] = "fifth decade stage"
        result = bc.effect_estimates(donors, "naive CD8 T", "age", -1)
        self.assertEqual(sorted(result["studies"]), ["s1", "s2"])
        self.assertEqual(result["studies"]["s1"]["donors"], 25)
        self.assertEqual(result["studies"]["s2"]["donors"], 29)
        for study in result["studies"].values():
            self.assertLess(abs(study["effect"] + 0.3), 3 * study["standardError"])
        self.assertEqual(result["leftOut"]["small"], "10 donors where 20 are needed")
        self.assertEqual(result["leftOut"]["nogenes"], "its stores lack the state's genes")
        self.assertLess(result["pooled"]["high"], 0)
        self.assertTrue(result["recovered"])
        self.assertEqual(result["agreement"], 1.0)

    def test_compares_a_disease_with_the_healthy_donors_of_its_study(self):
        rng = np.random.default_rng(3)
        rows = [
            *donor_rows("c1", [40] * 6, rng, disease="COVID-19", case_effect=1.0),
            *donor_rows("c1", [40] * 6, rng),
            *donor_rows("c1", [40] * 3, rng, disease="influenza", case_effect=-2.0),
            *donor_rows("c2", [40] * 4, rng, disease="COVID-19", case_effect=1.0),
            *donor_rows("c2", [40] * 10, rng),
        ]
        result = bc.effect_estimates(pd.DataFrame(rows), "naive CD8 T", "COVID-19", 1)
        self.assertEqual(list(result["studies"]), ["c1"])
        self.assertEqual(result["studies"]["c1"]["donors"], 12)
        self.assertLess(
            abs(result["studies"]["c1"]["effect"] - 1.0),
            3 * result["studies"]["c1"]["standardError"],
        )
        self.assertEqual(result["leftOut"]["c2"], "4 donors where 5 are needed")

    def test_a_state_without_a_cut_is_reported_as_left_out(self):
        donors = pd.DataFrame(donor_rows("s1", range(20, 70, 2), np.random.default_rng(1)))
        result = bc.effect_estimates(donors, "adaptive NK", "cytomegalovirus infection", 1)
        self.assertIn("no healthy reference", result["leftOut"])

    def test_the_report_holds_every_prespecified_effect_the_control_and_the_grid(self):
        donors = pd.DataFrame(
            donor_rows("s1", range(20, 70, 2), np.random.default_rng(2), slope=-0.3)
        )
        report = bc.donor_effects(donors)
        self.assertEqual(
            [(e["state"], e["covariate"]) for e in report["effects"]],
            [(s, c) for s, c, _ in bc.EFFECTS],
        )
        self.assertEqual(len(report["grid"]), len(bc.STATES) * len(bc.COVARIATES) - len(bc.EFFECTS))
        self.assertEqual(report["lupusControl"]["covariate"], "systemic lupus erythematosus")


if __name__ == "__main__":
    unittest.main()
