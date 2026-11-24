import unittest
from unittest import mock

from manuscript.figures import style as fs
from manuscript.figures.layout_scaling import draw
from matplotlib.colors import to_hex


class LayoutTest(unittest.TestCase):
    def test_the_layout_places_every_panel_once_inside_the_page(self):
        draw.check_layout()

    def test_the_three_panels_share_one_row_in_their_order(self):
        self.assertEqual(draw.letters(), {"read_requests": "a", "pass_exponents": "b"})
        self.assertEqual(sum(draw.CELLS[name].width for name in draw.LAYOUT[0]), 160.0)

    def test_the_plots_share_the_rows_frame(self):
        bottom, top = draw.ROW_FRAME_MM
        for name, cell in draw.CELLS.items():
            with self.subTest(panel=name):
                _, plot_bottom, _, height = cell.plot
                self.assertEqual((plot_bottom, plot_bottom + height), (bottom, top))

    def test_a_layout_without_a_header_for_each_panel_is_refused(self):
        with mock.patch.dict(draw.HEADERS, clear=True):
            with self.assertRaisesRegex(ValueError, "HEADERS"):
                draw.check_layout()


def marks(ax) -> list[tuple]:
    """Each point marker drawn in the plot: its shape, size, fill and edge."""
    return [
        (
            line.get_marker(),
            line.get_markersize(),
            to_hex(line.get_markerfacecolor()),
            to_hex(line.get_markeredgecolor()),
        )
        for line in ax.get_lines()
        if line.get_marker() not in (None, "None", "", " ", "none")
    ]


class SharedLegendTest(unittest.TestCase):
    def setUp(self):
        self.figures = []

        def save(fig, stem, *, folder=None, allow=None):
            self.figures.append(fig)

        fs.apply()
        with mock.patch.object(draw.fs, "save", save):
            draw.render_figure()
        [self.fig] = self.figures
        self.panels = {ax.get_gid(): ax for ax in self.fig.axes}

    def test_one_legend_names_the_two_layouts(self):
        legends = [ax.get_legend() for ax in self.fig.axes if ax.get_legend()]
        self.assertEqual(len(legends), 1)
        self.assertEqual(
            [t.get_text() for t in legends[0].get_texts()], ["fixed layout", "scaled layout"]
        )

    def test_both_layout_panels_draw_the_legends_marks_and_nothing_else(self):
        legend = self.panels["read_requests"].get_legend()
        shown = set(marks_of_handles(legend.legend_handles))
        self.assertEqual(len(shown), 2)
        for name in ("read_requests", "pass_exponents"):
            with self.subTest(panel=name):
                self.assertEqual(set(marks(self.panels[name])), shown)

    def test_the_fixed_layout_is_open_and_dashed_and_the_scaled_layout_filled_and_solid(self):
        legend = self.panels["read_requests"].get_legend()
        fixed, scaled = legend.legend_handles
        self.assertEqual(to_hex(fixed.get_markerfacecolor()), to_hex(fs.WHITE))
        self.assertEqual(fixed.get_linestyle(), "--")
        self.assertEqual(to_hex(scaled.get_markerfacecolor()), to_hex(fs.BLUE))
        self.assertEqual(scaled.get_linestyle(), "-")


def marks_of_handles(handles) -> list[tuple]:
    return [
        (
            h.get_marker(),
            h.get_markersize(),
            to_hex(h.get_markerfacecolor()),
            to_hex(h.get_markeredgecolor()),
        )
        for h in handles
    ]


class RenderTest(unittest.TestCase):
    def test_the_figure_draws_from_its_data_without_breaking_the_grammar(self):
        problems: list[str] = []

        def save(fig, stem, *, folder=None, allow=None):
            problems.extend(p for p in fs.lint(fig) if not any(key in p for key in allow or {}))

        fs.apply()
        with mock.patch.object(draw.fs, "save", save):
            draw.render_figure()
        self.assertEqual(problems, [])

    def test_each_panel_draws_alone_inside_its_cell(self):
        problems: dict[str, list[str]] = {}

        def save(fig, stem, *, folder=None, allow=None):
            found = [p for p in fs.lint(fig) if not any(key in p for key in allow or {})]
            if found:
                problems[stem] = found

        fs.apply()
        with mock.patch.object(draw.fs, "save", save):
            for name in draw.PANELS:
                draw.render_panel(name)
        self.assertEqual(problems, {})


if __name__ == "__main__":
    unittest.main()
