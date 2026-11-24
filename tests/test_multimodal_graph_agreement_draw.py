import unittest
from unittest import mock

from manuscript.figures import style as fs
from manuscript.figures.multimodal_graph_agreement import draw


class LayoutTest(unittest.TestCase):
    def test_the_layout_places_every_panel_once_inside_the_page(self):
        draw.check_layout()

    def test_the_agreement_by_size_is_the_figures_one_panel(self):
        self.assertEqual(draw.letters(), {"cite_agreement_by_size": "a"})

    def test_the_panel_spans_the_page(self):
        self.assertEqual(draw.LAYOUT["cite_agreement_by_size"], (0.0, 0.0))
        self.assertEqual(draw.CELLS["cite_agreement_by_size"].width, 160.0)


class HeaderAndLegendTest(unittest.TestCase):
    def setUp(self):
        self.figures = []

        def save(fig, stem, *, folder=None, allow=None):
            # The figure is closed after saving, which drops its canvas; keep the renderer that laid it out.
            fig.canvas.draw()
            self.figures.append((fig, fig.canvas.get_renderer()))

        fs.apply()
        with mock.patch.object(draw.fs, "save", save):
            draw.render_figure()
        [(self.fig, self.renderer)] = self.figures

    def test_the_panel_has_a_header(self):
        headers = [t.get_text() for t in self.fig.texts if t.get_gid() == "card-title"]
        self.assertEqual(headers, [draw.HEADERS["cite_agreement_by_size"]])

    def test_the_seeds_note_sits_in_the_legend_row_at_the_right_edge_of_the_last_plot(self):
        [panel] = [ax for ax in self.fig.axes if ax.get_gid() == "cite_agreement_by_size"]
        legend = panel.get_legend().get_window_extent(self.renderer)
        [note] = [t for ax in self.fig.axes for t in ax.texts if "seeds" in t.get_text()]
        box = note.get_window_extent(self.renderer)
        last = max(self.fig.axes, key=lambda ax: ax.get_position().x1)
        self.assertGreater(box.x0, legend.x1)
        self.assertAlmostEqual(box.x1, last.get_window_extent(self.renderer).x1, delta=1.0)
        # Centred on the legend's row within a quarter of a millimetre.
        self.assertAlmostEqual(
            (box.y0 + box.y1) / 2, (legend.y0 + legend.y1) / 2, delta=0.25 / 25.4 * self.fig.dpi
        )


class RenderTest(unittest.TestCase):
    def test_the_figure_draws_from_its_data_without_breaking_the_grammar(self):
        problems: list[str] = []

        def save(fig, stem, *, folder=None, allow=None):
            problems.extend(p for p in fs.lint(fig) if not any(key in p for key in allow or {}))

        fs.apply()
        with mock.patch.object(draw.fs, "save", save):
            draw.render_figure()
        self.assertEqual(problems, [])

    def test_the_panel_draws_alone_inside_its_cell(self):
        problems: list[str] = []

        def save(fig, stem, *, folder=None, allow=None):
            problems.extend(p for p in fs.lint(fig) if not any(key in p for key in allow or {}))

        fs.apply()
        with mock.patch.object(draw.fs, "save", save):
            draw.render_panel("cite_agreement_by_size")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
