"""The Harmony2 comparisons with the reference implementations: matched settings, the parity subsample, the
float64 correction equations, the full-fit equivalence rule and the exchange with the R reference script."""

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.special import softmax

# The experiment modules resolve their data folders from CYTEARC_DOWNLOADS and CYTEARC_RUNS when they are
# imported; the tests never write there, so the folders need not exist.
with mock.patch.dict(
    os.environ,
    {
        "CYTEARC_DOWNLOADS": str(Path(tempfile.gettempdir()) / "cytearc-downloads-unused"),
        "CYTEARC_RUNS": str(Path(tempfile.gettempdir()) / "cytearc-runs-unused"),
    },
):
    from experiments.integration import harmony_seed_stability as stability
    from experiments.integration.harmony_binvignat import HARMONY2_SETTINGS, harmonypy_arguments

from cytearc.embeddings.harmony import fit_harmony, harmony_cluster_count  # noqa: E402


def batch_offsets(cells: int, dims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two batches of cells around three cell types, with a batch shift."""
    rng = np.random.default_rng(seed)
    types = rng.integers(0, 3, cells)
    batches = np.where(rng.random(cells) < 0.4, "a", "b")
    centres = rng.normal(0, 4, (3, dims))
    shift = np.where(batches == "a", 1.5, -1.0)[:, None]
    return centres[types] + shift + rng.normal(0, 0.5, (cells, dims)), batches


class HarmonypyArguments(unittest.TestCase):
    """CyteArc's Harmony2 options under harmonypy's names, with CyteArc's cluster count."""

    def test_names_cluster_count_and_fixed_options(self):
        self.assertEqual(
            harmonypy_arguments(HARMONY2_SETTINGS, 6000),
            {
                "theta": 2.0,
                "sigma": 0.1,
                "tau": 0.0,
                "alpha": 0.2,
                "batch_prop_cutoff": 1e-5,
                "block_size": 0.05,
                "max_iter_kmeans": 200,
                "max_iter_harmony": 40,
                "epsilon_cluster": 1e-5,
                "epsilon_harmony": 1e-4,
                "nclust": 100,
                "lamb": None,
                "ncores": 1,
                "verbose": False,
            },
        )

    def test_cluster_count_is_cytearcs_for_every_size(self):
        for cells, clusters in ((90, 3), (1000, 33), (6000, 100), (18597, 100)):
            arguments = harmonypy_arguments(HARMONY2_SETTINGS, cells)
            self.assertEqual(arguments["nclust"], clusters)
            self.assertEqual(arguments["nclust"], harmony_cluster_count(cells))

    def test_zero_tolerance_never_stops_harmonypy(self):
        arguments = harmonypy_arguments(
            {**HARMONY2_SETTINGS, "epsilon_cluster": 0.0, "epsilon_harmony": 0.0}, 500
        )
        self.assertEqual((arguments["epsilon_cluster"], arguments["epsilon_harmony"]), (-1.0, -1.0))
        kept = harmonypy_arguments({**HARMONY2_SETTINGS, "epsilon_harmony": 0.5}, 500)
        self.assertEqual(kept["epsilon_harmony"], 0.5)

    def test_options_without_an_equivalent_are_refused(self):
        with self.assertRaisesRegex(ValueError, "n_clusters"):
            harmonypy_arguments({**HARMONY2_SETTINGS, "n_clusters": 10}, 500)


class Harmony2SettingsAreCyteArcDefaults(unittest.TestCase):
    """The settings passed to the reference are the ones CyteArc applies when they are left out."""

    def test_signature_defaults(self):
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(fit_harmony).parameters.items()
        }
        for name, value in HARMONY2_SETTINGS.items():
            if name == "ridge_alpha":
                # None selects the dynamic penalty with its documented scale, 0.2.
                self.assertIsNone(defaults[name])
                self.assertIn("which means 0.2", inspect.getdoc(fit_harmony))
            else:
                self.assertEqual(defaults[name], value, name)

    def test_explicit_settings_reproduce_the_default_fit(self):
        values, batches = batch_offsets(240, 4, 3)
        meta = pd.DataFrame({"batch": batches})
        default = fit_harmony(values.T.copy(), meta, random_state=5)
        explicit = fit_harmony(values.T.copy(), meta, random_state=5, **HARMONY2_SETTINGS)
        np.testing.assert_array_equal(default.corrected, explicit.corrected)


class ParityRows(unittest.TestCase):
    def test_small_selections_keep_every_cell(self):
        np.testing.assert_array_equal(stability.parity_rows(5999), np.arange(5999))
        np.testing.assert_array_equal(stability.parity_rows(6000), np.arange(6000))

    def test_large_selections_keep_the_recorded_sample(self):
        # Values recorded from the draw that harmony2.json's inputs used (seed 4444).
        for cells, first, total, last in (
            (18597, [0, 1, 6, 9, 12], 55477644, [18587, 18590, 18593]),
            (9000, [0, 1, 4, 5, 6], 27031546, [8997, 8998, 8999]),
        ):
            rows = stability.parity_rows(cells)
            self.assertEqual(len(rows), 6000)
            self.assertTrue(np.all(np.diff(rows) > 0))
            self.assertEqual(rows[:5].tolist(), first)
            self.assertEqual(int(rows.sum()), total)
            self.assertEqual(rows[-3:].tolist(), last)


class RidgeCorrection(unittest.TestCase):
    """The float64 correction equations, against values derived by hand from the normal equations."""

    def test_one_cluster_moves_each_batch_mean_by_one_over_one_plus_alpha(self):
        values, batches = batch_offsets(300, 5, 1)
        _, codes = np.unique(batches, return_inverse=True)
        corrected = stability.ridge_correction(
            values, codes, np.ones((300, 1)), keep_pruned_cells=True
        )
        mean = values.mean(axis=0)
        expected = values.copy()
        for level in (0, 1):
            members = codes == level
            expected[members] -= (values[members].mean(axis=0) - mean) / 1.2
        np.testing.assert_allclose(corrected, expected, atol=1e-10)

    def test_pruned_cells_kept_or_left_out(self):
        # Cluster 0 holds all 30 cells of batch 0 and 3 of the 20 cells of batch 1; cluster 1 holds the other
        # 17 cells of batch 1. With a cutoff of 0.2, batch 1 is pruned from cluster 0 (3 < 0.2 * 20).
        rng = np.random.default_rng(2)
        codes = np.array([0] * 30 + [1] * 20)
        values = rng.normal(0, 1, (50, 3)) + np.where(codes == 0, 2.0, -2.0)[:, None]
        assignments = np.zeros((50, 2))
        assignments[:33, 0] = 1
        assignments[33:, 1] = 1
        left_out = stability.ridge_correction(
            values, codes, assignments, keep_pruned_cells=False, prune_threshold=0.2
        )
        kept = stability.ridge_correction(
            values, codes, assignments, keep_pruned_cells=True, prune_threshold=0.2
        )
        # Left out, cluster 0 regresses batch 0's cells alone: the intercept absorbs them, so they are not moved.
        np.testing.assert_allclose(left_out[:30], values[:30], atol=1e-12)
        # Kept, the three batch-1 cells inform the intercept: w0 = (r1 + s r0) / (O1 + s O0) with
        # s = lambda / (O0 + lambda) and lambda = 0.2 * 33 * 30 / 50, and batch 0 moves by (r0 - O0 w0) / (O0 + lambda).
        penalty = 0.2 * 33 * 30 / 50
        share = penalty / (30 + penalty)
        r0, r1 = values[:30].sum(axis=0), values[30:33].sum(axis=0)
        intercept = (r1 + share * r0) / (3 + share * 30)
        np.testing.assert_allclose(
            kept[:30], values[:30] - (r0 - 30 * intercept) / (30 + penalty), atol=1e-10
        )

    def test_cytearc_keeps_pruned_cells_in_the_regression(self):
        from cytearc.embeddings.harmony.clustering import ClusteringWorkspace
        from cytearc.embeddings.harmony.correction import correct_embedding
        from cytearc.embeddings.harmony.design import encode_batch_design
        from cytearc.embeddings.harmony.parameters import resolve_harmony_parameters

        rng = np.random.default_rng(4)
        values, batches = batch_offsets(120, 4, 4)
        _, codes = np.unique(batches, return_inverse=True)
        assignments = softmax(rng.normal(0, 3, (120, 4)), axis=1)
        design = encode_batch_design(pd.DataFrame({"batch": batches}))
        parameters = resolve_harmony_parameters(
            {"n_clusters": 4, "ridge_alpha": 0.2, "prune_threshold": 0.3}, design
        )
        workspace = ClusteringWorkspace.allocate(120, 4, 4, design.n_levels)
        native = np.empty_like(values)
        correct_embedding(values, assignments, native, design, parameters, workspace, 0)
        kept = stability.ridge_correction(
            values, codes, assignments, keep_pruned_cells=True, prune_threshold=0.3
        )
        left_out = stability.ridge_correction(
            values, codes, assignments, keep_pruned_cells=False, prune_threshold=0.3
        )
        np.testing.assert_allclose(native, kept, atol=1e-9)
        self.assertGreater(np.abs(native - left_out).max(), 1e-3)


class FullFitEquivalence(unittest.TestCase):
    def test_passes_inside_the_reference_spread(self):
        result = stability.full_fit_equivalence(
            [0.95, 0.96, 0.97], [0.94, 0.98, 0.99], [0.30, 0.31, 0.32], [0.29, 0.33, 0.34]
        )
        self.assertEqual(
            (result["crossOverlapMedian"], result["referenceOverlapMin"]), (0.96, 0.94)
        )
        # Medians 0.31 and 0.33 differ by 0.02, within the reference's range of 0.05.
        self.assertAlmostEqual(result["iLISIMedianDifference"], 0.02)
        self.assertAlmostEqual(result["referenceILISIRange"], 0.05)
        self.assertTrue(result["passed"])

    def test_fails_below_the_reference_overlap_floor(self):
        result = stability.full_fit_equivalence(
            [0.90, 0.92, 0.93], [0.94, 0.98], [0.31], [0.29, 0.33]
        )
        self.assertFalse(result["overlapPassed"])
        self.assertTrue(result["iLISIPassed"])
        self.assertFalse(result["passed"])

    def test_fails_when_the_ilisi_shift_exceeds_the_reference_range(self):
        # Medians 0.205 and 0.31 differ by 0.105, more than the reference's range of 0.04.
        result = stability.full_fit_equivalence([0.97], [0.94, 0.98], [0.20, 0.21], [0.29, 0.33])
        self.assertTrue(result["overlapPassed"])
        self.assertFalse(result["iLISIPassed"])
        self.assertFalse(result["passed"])

    def test_needs_fits_of_both_tools(self):
        with self.assertRaises(ValueError):
            stability.full_fit_equivalence([], [0.9], [0.3], [0.3])


class ArmComparison(unittest.TestCase):
    def setUp(self):
        values, batches = batch_offsets(150, 30, 7)
        rng = np.random.default_rng(8)
        self.batches = batches
        self.labels = np.where(values[:, 0] > 0, "t", "u")
        self.reference = {seed: values + rng.normal(0, 0.05, values.shape) for seed in range(3)}

    def test_identical_fits_agree_exactly(self):
        native = {seed: fit.copy() for seed, fit in self.reference.items()}
        result = stability.arm_comparison(native, self.reference, self.batches, self.labels)
        same = result["cytearcVersusReferenceSameSeed"]
        self.assertEqual(same["knnOverlapK15"]["min"], 1.0)
        self.assertEqual(same["maxAbsoluteDifference"]["max"], 0.0)
        self.assertTrue(result["equivalence"]["passed"])

    def test_unrelated_fits_fail(self):
        rng = np.random.default_rng(9)
        native = {seed: fit[rng.permutation(len(fit))] for seed, fit in self.reference.items()}
        result = stability.arm_comparison(native, self.reference, self.batches, self.labels)
        self.assertLess(result["equivalence"]["crossOverlapMedian"], 0.5)
        self.assertFalse(result["equivalence"]["passed"])

    def test_needs_the_same_seeds(self):
        native = {seed: fit.copy() for seed, fit in self.reference.items() if seed < 2}
        with self.assertRaises(ValueError):
            stability.arm_comparison(native, self.reference, self.batches, self.labels)


class RReferenceExchange(unittest.TestCase):
    """The files written for harmony2_reference.R and the reading of what it writes back."""

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        values, batches = batch_offsets(300, 6, 11)
        self.data = {"coordinates": np.ascontiguousarray(values), "batches": batches}

    def test_raw_rows_round_trip_and_width_check(self):
        matrix = np.arange(12, dtype=float).reshape(4, 3)
        matrix.astype("<f8").tofile(self.folder / "m.f64")
        np.testing.assert_array_equal(stability.raw_rows(self.folder / "m.f64", 3), matrix)
        with self.assertRaises(ValueError):
            stability.raw_rows(self.folder / "m.f64", 5)

    def test_input_file(self):
        path = stability.r_reference_input({"kang": self.data}, self.folder)
        written = json.loads(path.read_text())
        dataset = written["datasets"]["kang"]
        state = stability.shared_assignment_state(self.data["coordinates"], self.data["batches"])
        self.assertEqual(dataset["levels"], ["a", "b"])
        self.assertEqual(dataset["step"]["rows"][0], 1)
        self.assertEqual(dataset["step"]["rows"][-1], 300)
        # R holds O and E with one row per cluster and one column per batch level.
        np.testing.assert_allclose(dataset["step"]["observedInjection"], state["observed"].T)
        np.testing.assert_allclose(dataset["step"]["expectedInjection"], state["expected"].T)
        np.testing.assert_array_equal(
            stability.raw_rows(Path(dataset["coordinates"]), 6), self.data["coordinates"]
        )
        self.assertFalse(written["arms"]["ownStoppingRules"]["disableStopping"])
        self.assertTrue(written["arms"]["fixedRoundsPenaltyOff"]["disableStopping"])
        self.assertEqual(written["arms"]["fixedRoundsPenaltyOff"]["theta"], 0.0)
        self.assertEqual(written["seeds"], list(range(10)))

    def write_step(self, theta: float, numerator: str) -> dict:
        """R's files for one assignment step whose update used the given numerator."""
        state = stability.shared_assignment_state(self.data["coordinates"], self.data["batches"])
        centroids = state["centroids"].T  # dims x clusters
        first = softmax(-2 * (1 - state["unit"] @ centroids) / 0.1, axis=1)
        observed, expected = state["observed"], state["expected"]
        top = 2 * expected + 1 if numerator == "rCode" else expected + 1
        after = first * ((top / (observed + expected + 1)) ** theta)[state["codes"]]
        after /= after.sum(axis=1, keepdims=True)
        names = {
            "firstAssignments": first,
            "firstCentroids": centroids.T,
            "assignments": after,
            "centroids": centroids.T,
        }
        for key, matrix in names.items():
            matrix.astype("<f8").tofile(self.folder / f"{key}.f64")
        return {"theta": theta, "clusters": 6, "dims": 6, **{key: f"{key}.f64" for key in names}}

    def test_assignment_step_identifies_the_r_code_numerator(self):
        result = stability.r_assignment_step(self.data, self.write_step(2.0, "rCode"), self.folder)
        self.assertLess(result["versusRCodeEquationFromFirstAssignments"], 1e-12)
        self.assertGreater(result["versusPaperEquationFromFirstAssignments"], 1e-4)
        self.assertLess(result["rCodeEquationVersusR_firstCentroids"], 1e-12)
        # CyteArc's kernel follows the paper's numerator, so it differs from this update.
        self.assertGreater(result["cytearcVersusR_firstCentroids"], 1e-4)

    def test_assignment_step_without_penalty_matches_cytearc(self):
        result = stability.r_assignment_step(self.data, self.write_step(0.0, "rCode"), self.folder)
        self.assertLess(result["cytearcVersusR_firstCentroids"], 1e-12)
        self.assertLess(result["versusPaperEquationFromFirstAssignments"], 1e-12)


if __name__ == "__main__":
    unittest.main()
