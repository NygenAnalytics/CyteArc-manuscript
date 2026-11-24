import unittest

import matplotlib.pyplot as plt
from manuscript.figures import style as fs

WIDTH_MM, HEIGHT_MM = 160.0, 60.0


def overlap_problems(fig) -> list[str]:
    return [p for p in fs.lint(fig) if " overlaps " in p]


class TiltedTextOverlapTest(unittest.TestCase):
    """Tilted text overlaps other text only where its box, turned with it, does; upright text keeps the upright-box
    check."""

    def setUp(self):
        fs.apply()
        self.fig = plt.figure(figsize=(WIDTH_MM * fs.MM, HEIGHT_MM * fs.MM))

    def tearDown(self):
        plt.close(self.fig)

    def text(self, x_mm: float, y_mm: float, s: str, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        return self.fig.text(
            x_mm / WIDTH_MM, y_mm / HEIGHT_MM, s, fontsize=fs.NOTE, color=fs.INK, **kw
        )

    def tilted(self, x_mm: float, s: str = "Yoshida, 0.3M"):
        """A label tilted 45 degrees, anchored by its top right corner, as tick labels under a plot are."""
        return self.text(x_mm, 48.0, s, rotation=45, ha="right", va="top", rotation_mode="anchor")

    def box_mm(self, t):
        self.fig.canvas.draw()
        box = t.get_window_extent(self.fig.canvas.get_renderer())
        mm = self.fig.dpi * fs.MM
        return box.x0 / mm, box.y0 / mm, box.x1 / mm, box.y1 / mm

    def test_parallel_tilted_labels_whose_upright_boxes_overlap_do_not(self):
        # Anchors 4.3 mm apart put two 45-degree labels 4.3 * sin(45) = 3.0 mm apart across their slant, more than
        # their 2.2 mm height, while their upright boxes, about 11 mm wide, overlap by more than half.
        first, second = self.tilted(80.0), self.tilted(84.3)
        a, b = self.box_mm(first), self.box_mm(second)
        self.assertGreater(a[2] - b[0], 5.0)
        self.assertEqual(overlap_problems(self.fig), [])

    def test_tilted_labels_closer_than_their_height_overlap(self):
        # Anchors 1 mm apart put the labels 0.7 mm apart across their slant, less than their height.
        self.tilted(80.0)
        self.tilted(81.0)
        self.assertEqual(len(overlap_problems(self.fig)), 1)

    def test_tilted_labels_on_one_anchor_overlap(self):
        self.tilted(80.0)
        self.tilted(80.0, "Perez, 1.3M")
        self.assertEqual(len(overlap_problems(self.fig)), 1)

    def test_an_upright_label_across_a_tilted_one_overlaps(self):
        x0, y0, x1, y1 = self.box_mm(self.tilted(80.0, "van der Wijst, 0.6M"))
        # The middle of a tilted label's upright box is the middle of its text.
        self.text((x0 + x1) / 2, (y0 + y1) / 2, "x")
        self.assertEqual(len(overlap_problems(self.fig)), 1)

    def test_an_upright_label_in_an_empty_corner_of_a_tilted_ones_box_does_not(self):
        x0, y0, x1, y1 = self.box_mm(self.tilted(80.0, "van der Wijst, 0.6M"))
        # The text runs from the box's lower left to its upper right, so the upper left corner is empty.
        upright = self.text(x0 + 1.0, y1 - 1.0, "x")
        u = self.box_mm(upright)
        self.assertTrue(u[0] < x1 and x0 < u[2] and u[1] < y1 and y0 < u[3])
        self.assertEqual(overlap_problems(self.fig), [])

    def test_upright_labels_keep_the_upright_box_check(self):
        self.text(40.0, 30.0, "overlapping")
        self.text(42.0, 30.5, "overlapping")
        self.text(120.0, 10.0, "apart")
        self.assertEqual(len(overlap_problems(self.fig)), 1)


if __name__ == "__main__":
    unittest.main()
