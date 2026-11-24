import json
import unittest
from unittest import mock

import matplotlib.pyplot as plt
from manuscript.figures import style as fs
from manuscript.figures.in_place_access import draw
from manuscript.figures.zero_copy_analysis import draw as figure_2
from matplotlib.colors import to_hex


class LayoutTest(unittest.TestCase):
    def test_the_layout_places_every_panel_once_inside_the_page(self):
        draw.check_layout()

    def test_the_two_panels_share_one_row_in_their_order(self):
        # The mount footprint is not drawn here: the artifact-reuse measurements show the store sizes it repeated.
        self.assertEqual(draw.letters(), {"release_storage": "a", "blood_screen": "b"})
        self.assertEqual({bottom for _, bottom in draw.LAYOUT.values()}, {0.0})

    def test_the_row_fills_the_page_width_without_a_gap(self):
        left_a, _ = draw.LAYOUT["release_storage"]
        left_b, _ = draw.LAYOUT["blood_screen"]
        self.assertEqual(left_a, 0.0)
        self.assertEqual(left_b, draw.CELLS["release_storage"].width)
        self.assertEqual(left_b + draw.CELLS["blood_screen"].width, 160.0)

    def test_the_release_plot_and_the_screens_two_lanes_share_the_rows_frame(self):
        bottom, top = draw.ROW_FRAME_MM
        _, release_bottom, _, release_height = draw.CELLS["release_storage"].plot
        self.assertEqual((release_bottom, release_bottom + release_height), (bottom, top))
        # The cell's plot is the screen's top lane; the bytes lane follows a gap below it, at the frame's bottom.
        _, lane_bottom, _, lane_height = draw.CELLS["blood_screen"].plot
        self.assertAlmostEqual(lane_bottom + lane_height, top)
        self.assertAlmostEqual(lane_bottom - draw.SCREEN_LANE_GAP_MM - lane_height, bottom)


class DecadeAxisTest(unittest.TestCase):
    def test_every_decade_takes_a_tick_and_only_the_named_ones_a_label(self):
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        draw.decade_axis(ax.xaxis, (1e2, 3e7), {1e2: "100", 1e4: "10k"})
        ticks = list(ax.xaxis.get_major_locator().tick_values(1e2, 3e7))
        self.assertEqual(ticks, [1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
        formatter = ax.xaxis.get_major_formatter()
        self.assertEqual([formatter(t) for t in ticks], ["100", "", "10k", "", "", ""])
        # Minor ticks fill each decade and stop at the axis's end: 3e7 is in, 4e7 is not.
        minor = list(ax.xaxis.get_minor_locator().tick_values(1e2, 3e7))
        self.assertIn(2e2, minor)
        self.assertIn(3e7, minor)
        self.assertNotIn(4e7, minor)
        plt.close(fig)


class LabelsTest(unittest.TestCase):
    def test_the_release_header_gives_the_releases_observations(self):
        data = json.loads((draw.HERE / "release_storage.json").read_text())
        observations = f"{data['releaseObservations'] / 1e6:.0f}M observations"
        self.assertEqual(draw.HEADERS["release_storage"], f"CyteBase ({observations})")

    def test_the_screen_names_its_marks_as_figure_2s_query_names_its_bars(self):
        # The screen draws the CyteBase query and the download; Figure 2's query adds Census between them.
        query = json.loads((figure_2.HERE / "tabula_query.json").read_text())
        labels = {row["key"]: row["label"] for row in query["rows"]}
        self.assertEqual(draw.SCREEN_LEGEND, (labels["cytebase"], labels["download"]))

    def test_the_screens_log_axes_tick_every_decade_with_minor_ticks_between(self):
        cell = draw.CELLS["blood_screen"]
        fs.apply()
        fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
        self.addCleanup(plt.close, fig)
        draw.draw_panel(fig, "blood_screen", 0.0, 0.0)
        lanes = {ax.get_gid(): ax for ax in fig.axes}
        time, data_lane = lanes["blood_screen"], lanes["blood_screen_bytes"]
        self.assertIn(20.0, time.yaxis.get_minor_locator().tick_values(1, 3000))
        self.assertIn(3e7, data_lane.yaxis.get_minor_locator().tick_values(3e5, 3e11))
        self.assertEqual(
            [
                t
                for t in data_lane.yaxis.get_major_locator().tick_values(3e5, 3e11)
                if 3e5 <= t <= 3e11
            ],
            [1e6, 1e7, 1e8, 1e9, 1e10, 1e11],
        )


class ScreenLegendTest(unittest.TestCase):
    def test_the_legend_shows_the_marks_the_lanes_draw(self):
        cell = draw.CELLS["blood_screen"]
        fs.apply()
        fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
        self.addCleanup(plt.close, fig)
        draw.draw_panel(fig, "blood_screen", 0.0, 0.0)
        time = next(ax for ax in fig.axes if ax.get_gid() == "blood_screen")
        legend = time.get_legend()
        self.assertEqual([t.get_text() for t in legend.get_texts()], list(draw.SCREEN_LEGEND))
        # A scatter's size is its marker's area in square points; a legend marker's size is its width in points.
        drawn = [c.get_sizes()[0] for c in time.collections]
        shown = [h.get_markersize() ** 2 for h in legend.legend_handles]
        for area, legend_area in zip(drawn, shown, strict=True):
            self.assertAlmostEqual(area, legend_area)
        for collection, handle in zip(time.collections, legend.legend_handles, strict=True):
            self.assertEqual(
                to_hex(collection.get_facecolors()[0]), to_hex(handle.get_markerfacecolor())
            )
            self.assertEqual(
                to_hex(collection.get_edgecolors()[0]), to_hex(handle.get_markeredgecolor())
            )


class RenderTest(unittest.TestCase):
    def test_the_figure_draws_from_its_data_without_breaking_the_grammar(self):
        problems: list[str] = []

        def save(fig, stem, *, folder=None, allow=None):
            # As fs.save does, a violation the figure accepts with a reason is not a problem.
            problems.extend(p for p in fs.lint(fig) if not any(key in p for key in allow or {}))

        fs.apply()
        with mock.patch.object(draw.fs, "save", save):
            draw.render_figure()
        self.assertEqual(problems, [])

    def test_the_figure_accepts_no_departure_from_the_grammar(self):
        self.assertEqual(draw.accepted_exceptions(), {})

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
