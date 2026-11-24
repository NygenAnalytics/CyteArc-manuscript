"""The supplementary figures read as one set: each panel letter has a header beside it, legends sit in one row under the
headers in the tick type, and axis labels follow one wording."""

import importlib
import unittest
from unittest import mock

from manuscript.figures import style as fs

FIGURES = (
    "layout_scaling",
    "in_place_access",
    "artifact_reuse_measurements",
    "reference_mapping_measurements",
    "multimodal_graph_agreement",
)
# A header starts this far right of its panel's letter, on the letter's baseline; a legend's frame starts this far right
# of the letter, its top this far under the letter's baseline.
HEADER_AFTER_LETTER_MM = 3.8
LEGEND_AFTER_LETTER_MM = 0.5
LEGEND_UNDER_BASELINE_MM = (1.0, 1.4)
TOLERANCE_MM = 0.1


def rendered(folder: str):
    """The figure as its draw module assembles it, with the renderer that laid it out."""
    draw = importlib.import_module(f"manuscript.figures.{folder}.draw")
    out = []

    def save(fig, stem, *, folder=None, allow=None):
        fig.canvas.draw()
        out.append((fig, fig.canvas.get_renderer()))

    fs.apply()
    with mock.patch.object(draw.fs, "save", save):
        draw.render_figure()
    [(fig, renderer)] = out
    return draw, fig, renderer


class SupplementaryFigureStyleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.figures = {folder: rendered(folder) for folder in FIGURES}

    def mm(self, fig, x: float, y: float) -> tuple[float, float]:
        """Display coordinates as millimetres from the figure's lower-left corner."""
        return x / fig.dpi * 25.4, y / fig.dpi * 25.4

    def letters_and_headers(self, fig):
        letters = [t for t in fig.texts if t.get_gid() == "panel-letter"]
        headers = [t for t in fig.texts if t.get_gid() == "card-title"]
        return letters, headers

    def position_mm(self, fig, text) -> tuple[float, float]:
        x, y = text.get_position()
        return x * fig.get_figwidth() / fs.MM, y * fig.get_figheight() / fs.MM

    def test_each_panel_letter_has_a_header_on_its_baseline(self):
        for folder, (draw, fig, _) in self.figures.items():
            with self.subTest(figure=folder):
                letters, headers = self.letters_and_headers(fig)
                self.assertEqual(len(letters), len(draw.PANELS))
                self.assertEqual(len(headers), len(letters))
                for letter in letters:
                    lx, ly = self.position_mm(fig, letter)
                    matches = [
                        h
                        for h in headers
                        if abs(self.position_mm(fig, h)[0] - lx - HEADER_AFTER_LETTER_MM)
                        < TOLERANCE_MM
                        and abs(self.position_mm(fig, h)[1] - ly) < TOLERANCE_MM
                    ]
                    self.assertEqual(len(matches), 1, letter.get_text())
                    self.assertEqual(letter.get_fontsize(), fs.LETTER)
                    self.assertEqual(matches[0].get_fontsize(), fs.LABEL)

    def test_legends_sit_in_one_row_under_a_header_in_the_tick_type(self):
        for folder, (_, fig, renderer) in self.figures.items():
            letters, _ = self.letters_and_headers(fig)
            anchors = [self.position_mm(fig, t) for t in letters]
            axes_legends = [ax.get_legend() for ax in fig.axes if ax.get_legend()]
            for legend in axes_legends:
                with self.subTest(figure=folder, legend=legend.get_texts()[0].get_text()):
                    texts = legend.get_texts()
                    self.assertTrue(all(t.get_fontsize() == fs.TICK for t in texts))
                    self.assertEqual(
                        len({round(t.get_window_extent(renderer).y0) for t in texts}), 1
                    )
                    box = legend.get_window_extent(renderer)
                    x0, _ = self.mm(fig, box.x0, box.y0)
                    _, top = self.mm(fig, box.x1, box.y1)
                    under = [
                        (lx, ly)
                        for lx, ly in anchors
                        if abs(x0 - lx - LEGEND_AFTER_LETTER_MM) < TOLERANCE_MM
                        and LEGEND_UNDER_BASELINE_MM[0] <= ly - top <= LEGEND_UNDER_BASELINE_MM[1]
                    ]
                    self.assertEqual(len(under), 1)
                    # The legend stays above its plot.
                    plot = legend.axes.get_window_extent(renderer)
                    self.assertGreaterEqual(box.y0, plot.y1)
            for legend in fig.legends:
                with self.subTest(figure=folder, legend="figure"):
                    texts = legend.get_texts()
                    self.assertTrue(all(t.get_fontsize() == fs.TICK for t in texts))
                    self.assertEqual(
                        len({round(t.get_window_extent(renderer).y0) for t in texts}), 1
                    )
                    box = legend.get_window_extent(renderer)
                    self.assertTrue(all(t.get_window_extent(renderer).y1 < box.y0 for t in letters))

    def test_axis_labels_are_in_the_label_type_and_log_axes_say_log10(self):
        for folder, (_, fig, _) in self.figures.items():
            for ax in fig.axes:
                for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
                    label = axis.label.get_text()
                    if not label or not ax.axison:
                        continue
                    with self.subTest(figure=folder, label=label):
                        self.assertEqual(axis.label.get_fontsize(), fs.LABEL)
                        if scale == "log":
                            self.assertIn("log10)", label)
                        else:
                            self.assertNotIn("log", label)


if __name__ == "__main__":
    unittest.main()
