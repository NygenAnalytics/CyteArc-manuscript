import json
import unittest
from unittest import mock

import matplotlib.pyplot as plt
from manuscript.figures import style as fs
from manuscript.figures.artifact_reuse_measurements import draw
from matplotlib.colors import to_hex


def top(name: str) -> float:
    return draw.LAYOUT[name][1] + draw.CELLS[name].height


class LayoutTest(unittest.TestCase):
    def test_the_layout_places_every_panel_once_inside_the_page(self):
        draw.check_layout()

    def test_the_seven_panels_take_their_letters_in_the_order_the_main_text_cites_them(self):
        # Paragraph 52 cites the requests, the fork, the store sizes, Harmony's stages and resources and the reference
        # update in this order, and paragraph 56 the revised CITE-seq clustering.
        self.assertEqual(
            draw.letters(),
            {
                "mounted_store_requests": "a",
                "neighbor_count_fork_stages": "b",
                "endothelial_store_size": "c",
                "composite_integration_stages": "d",
                "composite_integration_resources": "e",
                "reference_update_steps": "f",
                "multimodal_revision": "g",
            },
        )

    def test_rows_align_at_the_top_and_read_left_to_right(self):
        rows = (
            ("mounted_store_requests", "neighbor_count_fork_stages"),
            ("endothelial_store_size", "composite_integration_stages"),
            ("composite_integration_resources", "reference_update_steps"),
        )
        for left, right in rows:
            with self.subTest(row=left):
                self.assertEqual(top(left), top(right))
                self.assertEqual(draw.LAYOUT[left][0], 0.0)
                self.assertEqual(draw.LAYOUT[right][0], 80.0)
        # Each row starts where the taller cell of the row above ends.
        for above, below in zip(rows, rows[1:]):
            with self.subTest(row=below[0]):
                self.assertEqual(top(below[0]), min(draw.LAYOUT[name][1] for name in above))
        # The composite's counts and peak take the full height of the reference update beside them.
        self.assertEqual(
            draw.CELLS["composite_integration_resources"].height,
            draw.CELLS["reference_update_steps"].height,
        )
        # The revised CITE-seq clustering spans the page under the last row.
        self.assertEqual(draw.LAYOUT["multimodal_revision"], (0.0, 0.0))
        self.assertEqual(draw.CELLS["multimodal_revision"].width, 160.0)
        self.assertEqual(top("multimodal_revision"), min(draw.LAYOUT[name][1] for name in rows[-1]))
        # 48 + 40 + 40 + 36 millimetres down the page.
        self.assertAlmostEqual(max(top(name) for name in draw.LAYOUT), 164.0)


def drawn(name: str, data: dict | None = None):
    """The panel drawn alone into a figure of its cell's size, from its data file unless data is given."""
    fs.apply()
    cell = draw.CELLS[name]
    fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
    ax = fs.add_axes(fig, *cell.plot)
    ax.set_gid(name)
    getattr(draw, name)(ax, data or json.loads((draw.HERE / f"{name}.json").read_text()))
    return fig, ax


class LegendNamesTest(unittest.TestCase):
    """A mark means one thing across the panels, and its legend says so in the same words."""

    def legend_names(self, name: str) -> dict[str, str]:
        """Each legend entry's label by its mark: square, open or filled circle, or diamond."""
        fig, ax = drawn(name)
        self.addCleanup(plt.close, fig)
        legend = ax.get_legend()
        out = {}
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=True):
            filled = to_hex(handle.get_markerfacecolor()) != to_hex(fs.WHITE)
            mark = {"s": "square", "D": "diamond"}.get(
                handle.get_marker(), "filled" if filled else "open"
            )
            out[mark] = text.get_text()
        return out

    def test_open_circles_are_stored_and_filled_ones_computed_in_every_timed_panel(self):
        for name in (
            "mounted_store_requests",
            "neighbor_count_fork_stages",
            "composite_integration_stages",
            "reference_update_steps",
        ):
            with self.subTest(panel=name):
                names = self.legend_names(name)
                self.assertEqual((names["open"], names["filled"]), ("stored", "computed"))

    def test_the_requests_and_the_fork_name_the_published_run_alike(self):
        # Both panels set their steps beside a published run of the ten-million-cell store on 16 CPUs.
        requests = self.legend_names("mounted_store_requests")["square"]
        fork = self.legend_names("neighbor_count_fork_stages")["square"]
        self.assertEqual(requests, fork)
        self.assertEqual(requests, "initial, 16 CPUs")


class CompositeResourcesTest(unittest.TestCase):
    def test_the_peak_plot_sits_under_the_counts_plot_on_the_same_rows(self):
        fig, counts = drawn("composite_integration_resources")
        self.addCleanup(plt.close, fig)
        [peak] = [ax for ax in fig.axes if ax is not counts]
        a, b = counts.get_position(), peak.get_position()
        self.assertAlmostEqual(a.x0, b.x0)
        self.assertAlmostEqual(a.width, b.width)
        gap_mm = (a.y0 - b.y1) * fig.get_figheight() / fs.MM
        self.assertAlmostEqual(gap_mm, draw.RESOURCE_GAP_MM)
        self.assertEqual(
            [t.get_text() for t in counts.get_yticklabels()],
            [t.get_text() for t in peak.get_yticklabels()],
        )
        self.assertEqual(counts.get_xlabel(), "counts read (GB)")
        self.assertEqual(peak.get_xlabel(), "peak (GiB)")


class MultimodalRevisionTest(unittest.TestCase):
    def test_the_times_at_the_largest_size_sit_at_the_ends_of_their_lines(self):
        data = json.loads((draw.HERE / "multimodal_revision.json").read_text())
        fig, ax = drawn("multimodal_revision", data)
        self.addCleanup(plt.close, fig)
        last = len(data["sizes"]) - 1
        labels = {t.get_text(): t.get_position() for t in ax.texts}
        largest = data["sizes"][-1]
        for key, text in (("fresh", "13.6 min"), ("reuse", "20 s")):
            with self.subTest(line=key):
                x, y = labels[text]
                # Just right of the line's last point, at its height, and inside the plot.
                self.assertGreater(x, last)
                self.assertLess(x, last + 0.5)
                self.assertAlmostEqual(y, largest[f"{key}Seconds"] / 60)
                self.assertLess(x, ax.get_xlim()[1])
        # Nothing else is written in the plot.
        self.assertEqual(sorted(labels), ["13.6 min", "20 s"])

    def test_both_plots_give_each_size_the_same_width(self):
        fig, ax = drawn("multimodal_revision")
        self.addCleanup(plt.close, fig)
        [peak] = [other for other in fig.axes if other is not ax]

        def per_size(plot):
            low, high = plot.get_xlim()
            return plot.get_position().width / (high - low)

        self.assertAlmostEqual(per_size(ax), per_size(peak))


def fork_data() -> dict:
    """Twelve rows: three looked up, five recomputed, three the fork did neither, and the total."""
    stages = [
        ("filterCells", "cell filter", None, 20.0, None),
        ("markHvgs", "variable genes", None, 400.0, None),
        ("runNormalization", "normalization", "lookup", 350.0, 13.0),
        ("runPca", "PCA", "lookup", 450.0, 14.0),
        ("buildAnnIndex", "neighbor index", "lookup", 110.0, 15.0),
        ("queryNeighbors", "neighbors", "computed", 120.0, 160.0),
        ("buildConnectivityMap", "graph", "computed", 60.0, 50.0),
        ("buildEmbeddingInitialization", "UMAP initialization", None, 140.0, None),
        ("runUmap", "UMAP", "computed", 1050.0, 1600.0),
        ("runLeiden", "Leiden", "computed", 370.0, 540.0),
        ("findMarkers", "markers", "computed", 410.0, 510.0),
    ]
    return {
        "stages": [
            {"stage": s, "name": n, "fork": k, "storedSeconds": stored, "forkSeconds": f}
            for s, n, k, stored, f in stages
        ],
        "totals": {"storedSeconds": 3480.0, "forkSeconds": 2860.0},
        "cells": 10_000_000,
        "cpus": {"stored": 16, "fork": 8},
    }


class ForkPanelTest(unittest.TestCase):
    def setUp(self):
        fs.apply()
        cell = draw.CELLS["neighbor_count_fork_stages"]
        self.fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
        self.addCleanup(plt.close, self.fig)
        self.ax = fs.add_axes(self.fig, *cell.plot)
        self.ax.set_gid("neighbor_count_fork_stages")
        draw.neighbor_count_fork_stages(self.ax, fork_data())

    def marks(self) -> dict[str, list[tuple[float, float]]]:
        """Each drawn value by its mark: squares for the stored analysis, open circles for lookups and filled circles
        for recomputed stages."""
        out: dict[str, list[tuple[float, float]]] = {"square": [], "open": [], "filled": []}
        for line in self.ax.lines:
            if line.get_marker() in (None, "None", "", " ", "none"):
                continue
            (x,), (y,) = line.get_data()
            if line.get_marker() == "s":
                out["square"].append((x, y))
            elif to_hex(line.get_markerfacecolor()) == to_hex(fs.WHITE):
                out["open"].append((x, y))
            else:
                out["filled"].append((x, y))
        return out

    def test_the_fork_is_one_plot_with_a_row_per_stage_and_the_total_last(self):
        self.assertEqual(len(self.fig.axes), 1)
        self.assertEqual(
            [t.get_text() for t in self.ax.get_yticklabels()],
            [
                "cell filter",
                "variable genes",
                "normalization",
                "PCA",
                "neighbor index",
                "neighbors",
                "graph",
                "UMAP initialization",
                "UMAP",
                "Leiden",
                "markers",
                "total",
            ],
        )
        self.assertEqual(self.ax.get_xlabel(), "seconds (log10)")
        self.assertEqual(self.ax.get_xscale(), "log")

    def test_each_stage_shows_the_stored_time_beside_the_forks_lookup_or_recomputation(self):
        marks = self.marks()
        # Rows count down from 11 at the top to the total at 0.
        self.assertEqual(
            sorted(marks["square"]),
            sorted(
                [
                    (20.0, 11.0),
                    (400.0, 10.0),
                    (350.0, 9.0),
                    (450.0, 8.0),
                    (110.0, 7.0),
                    (120.0, 6.0),
                    (60.0, 5.0),
                    (140.0, 4.0),
                    (1050.0, 3.0),
                    (370.0, 2.0),
                    (410.0, 1.0),
                    (3480.0, 0.0),
                ]
            ),
        )
        self.assertEqual(sorted(marks["open"]), [(13.0, 9.0), (14.0, 8.0), (15.0, 7.0)])
        self.assertEqual(
            sorted(marks["filled"]),
            [(50.0, 5.0), (160.0, 6.0), (510.0, 1.0), (540.0, 2.0), (1600.0, 3.0), (2860.0, 0.0)],
        )

    def test_every_value_lies_inside_the_axis(self):
        low, high = self.ax.get_xlim()
        values = [x for points in self.marks().values() for x, _ in points]
        self.assertTrue(all(low < x < high for x in values), (low, high, min(values), max(values)))


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
