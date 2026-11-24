import unittest

import matplotlib.pyplot as plt
import numpy as np
from manuscript.figures import glyphs as g


class FolderTest(unittest.TestCase):
    def folder(self, **options):
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        ax = g.mm_axes(fig, 0, 0, 40, 20)
        g.folder(ax, 2.0, 1.0, 30.0, 10.0, tab_h=3.0, **options)
        return ax

    def test_a_right_tab_mirrors_the_left_tab_about_the_bodys_middle(self):
        left = self.folder(tab=8.0).patches[-1].get_path().vertices
        right = self.folder(tab=8.0, side="right").patches[-1].get_path().vertices
        # Every point of the right folder is a point of the left one reflected about x = 16.
        np.testing.assert_allclose(right[:, 0], 32.0 - left[:, 0])
        np.testing.assert_allclose(right[:, 1], left[:, 1])
        # The tab rises over the body's left end on one and its right end on the other.
        top = left[:, 1].max()
        self.assertLess(left[np.isclose(left[:, 1], top), 0].max(), 16.0)
        self.assertGreater(right[np.isclose(right[:, 1], top), 0].min(), 16.0)

    def test_the_label_sits_on_the_tab_of_the_named_side(self):
        for side, x, align in (("left", 3.6, "left"), ("right", 28.4, "right")):
            with self.subTest(side=side):
                (text,) = self.folder(label="user store", side=side).texts
                self.assertAlmostEqual(text.get_position()[0], x)
                self.assertEqual(text.get_horizontalalignment(), align)

    def test_a_tab_on_any_other_side_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "left or right side"):
            self.folder(side="top")


if __name__ == "__main__":
    unittest.main()
