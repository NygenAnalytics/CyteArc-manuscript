import unittest

import numpy as np
from experiments.runtime.multiple_testing import benjamini_hochberg, holm
from statsmodels.stats.multitest import multipletests


class MultipleTestingTest(unittest.TestCase):
    def test_four_p_values_worked_by_hand(self):
        # Sorted: 0.005, 0.01, 0.03, 0.04 of m = 4.
        # Benjamini-Hochberg: p * 4 / rank = 0.02, 0.02, 0.04, 0.04, each the minimum of itself and those after it.
        # Holm: p * (4 - rank + 1) = 0.02, 0.03, 0.06, 0.04, each the maximum of itself and those before it.
        p = [0.01, 0.04, 0.03, 0.005]
        np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
        np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])

    def test_adjusted_values_are_capped_at_one_and_nan_leaves_the_family(self):
        # Without the NaN, m = 2. Holm doubles the smaller, 0.55, to 1.1, capped at 1, and the larger, 0.6 times 1, may
        # not fall below it. Benjamini-Hochberg: 0.55 * 2 / 1 = 1.1 and 0.6 * 2 / 2 = 0.6, the smaller of each and the
        # values after it. With m = 3 both would differ.
        p = [0.6, np.nan, 0.55]
        np.testing.assert_allclose(holm(p), [1.0, np.nan, 1.0])
        np.testing.assert_allclose(benjamini_hochberg(p), [0.6, np.nan, 0.6])
        self.assertTrue(np.isnan(holm([np.nan])).all())

    def test_both_agree_with_statsmodels_on_random_families(self):
        rng = np.random.default_rng(7)
        for size in (1, 2, 9, 50):
            p = rng.uniform(0, 0.2, size) ** 2
            np.testing.assert_allclose(
                benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1], rtol=1e-12
            )
            np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1], rtol=1e-12)

    def test_ties_get_the_same_adjusted_value(self):
        p = [0.02, 0.02, 0.02]
        np.testing.assert_allclose(holm(p), [0.06, 0.06, 0.06])
        np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.02, 0.02])


if __name__ == "__main__":
    unittest.main()
