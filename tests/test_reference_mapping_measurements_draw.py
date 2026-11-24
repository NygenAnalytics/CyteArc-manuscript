import unittest
from unittest import mock

from manuscript.figures import style as fs
from manuscript.figures.reference_mapping_measurements import draw
from matplotlib.colors import to_hex


class LayoutTest(unittest.TestCase):
    def test_the_layout_places_every_panel_once_inside_the_page(self):
        draw.check_layout()

    def test_the_four_panels_the_text_cites_take_their_letters_in_its_order(self):
        self.assertEqual(
            draw.letters(),
            {
                "registration_reads": "a",
                "saved_reference_size": "b",
                "ladder_annotation_time": "c",
                "ladder_rare_type_missed": "d",
            },
        )

    def test_registration_and_saving_share_the_top_row_and_the_ladder_panels_the_bottom_one(self):
        bottoms = {name: bottom for name, (_, bottom) in draw.LAYOUT.items()}
        self.assertEqual(bottoms["ladder_annotation_time"], 0.0)
        self.assertEqual(bottoms["ladder_rare_type_missed"], 0.0)
        ladder_top = draw.CELLS["ladder_annotation_time"].height
        self.assertEqual(bottoms["registration_reads"], ladder_top)
        self.assertEqual(bottoms["saved_reference_size"], ladder_top)
        # Registration and the saved sizes sit side by side and fill the page's width.
        left, _ = draw.LAYOUT["saved_reference_size"]
        self.assertEqual(left, draw.CELLS["registration_reads"].width)
        self.assertEqual(left + draw.CELLS["saved_reference_size"].width, 160.0)

    def test_the_two_ladder_panels_fill_the_page_width_at_one_size(self):
        annotation, rare = (
            draw.CELLS["ladder_annotation_time"],
            draw.CELLS["ladder_rare_type_missed"],
        )
        self.assertEqual(annotation, rare)
        self.assertEqual(draw.LAYOUT["ladder_annotation_time"][0], 0.0)
        self.assertEqual(draw.LAYOUT["ladder_rare_type_missed"][0], annotation.width)
        self.assertEqual(2 * annotation.width, 160.0)
        # The plot spans the cell, less the y axis on its left and a margin for the last tick label on its right.
        left, _, width, _ = annotation.plot
        self.assertGreater(width, annotation.width / 2)
        self.assertLessEqual(left + width, annotation.width - 2.0)


class MethodLegendTest(unittest.TestCase):
    """One legend names every method the figure draws, in one row across the top of the figure."""

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

    def test_the_figure_has_one_legend_and_no_panel_has_its_own(self):
        self.assertEqual(len(self.fig.legends), 1)
        self.assertEqual([ax.get_gid() for ax in self.fig.axes if ax.get_legend()], [])

    def test_the_legend_names_each_method_the_panels_draw_once(self):
        [legend] = self.fig.legends
        names = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(sorted(names), sorted(m.name for m in draw.METHODS.values()))

        # Every mark drawn on the page is in the legend: a method's marker, filled with its colour or open with an edge
        # of it.
        def mark(line) -> tuple[str, str, str]:
            return (
                line.get_marker(),
                to_hex(line.get_markerfacecolor()),
                to_hex(line.get_markeredgecolor()),
            )

        drawn = {
            mark(line)
            for ax in self.fig.axes
            for line in ax.get_lines()
            if line.get_marker() not in (None, "None", "", " ", "none")
        }
        shown = {mark(h) for h in legend.legend_handles}
        self.assertTrue(drawn)
        self.assertLessEqual(drawn, shown)

    def test_the_legend_is_one_row_above_every_panel(self):
        [legend] = self.fig.legends
        renderer = self.renderer
        box = legend.get_window_extent(renderer)
        rows = {round(t.get_window_extent(renderer).y0) for t in legend.get_texts()}
        self.assertEqual(len(rows), 1)
        self.assertLessEqual(box.x1, self.fig.bbox.x1)
        letters = [t for t in self.fig.texts if t.get_gid() == "panel-letter"]
        self.assertEqual(len(letters), len(draw.PANELS))
        self.assertTrue(all(t.get_window_extent(renderer).y1 < box.y0 for t in letters))


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
