import json
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from manuscript.figures import glyphs as g
from manuscript.figures import style as fs
from manuscript.figures.zero_copy_analysis import draw
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation


class ByteSizeTest(unittest.TestCase):
    def test_bytes_print_as_the_text_writes_them(self):
        cases = {
            (68_616_468, 1): "69 MB",
            (771_064_541, 1): "771 MB",
            (5_661_865_252, 1): "5.7 GB",
            (64_565_123_255, 1): "64.6 GB",
            (45_013_733_785, 0): "45 GB",
            (257_610_066_994, 1): "258 GB",
        }
        for (value, decimals), text in cases.items():
            with self.subTest(value=value):
                self.assertEqual(draw.byte_size(value, decimals=decimals), text)


class LayoutTest(unittest.TestCase):
    def test_the_layout_places_every_panel_once_inside_the_page(self):
        draw.check_layout()

    def test_letters_follow_the_layout_from_the_schematic(self):
        self.assertEqual(
            draw.letters(),
            {
                "mount_overlay": "a",
                "storage_backends": "b",
                "tabula_query": "c",
                "interferon_score": "d",
                "composite_umap": "e",
                "progenitor_funnel": "f",
                "progenitor_states": "g",
                "progenitor_sources": "h",
                "progenitor_state_shares": "i",
                "progenitor_diseases": "j",
                "progenitor_covid": "k",
                "progenitor_cohorts": "l",
            },
        )

    def test_the_progenitor_row_spans_the_page_under_both_columns(self):
        # From the page's left edge: the funnel, the state map over the source map, and the state shares beside the
        # CD34+ datasets', which check the screen, each column as tall as the row, which ends where the composite and
        # the ridges begin. Then a block that goes from studies to one study's donors: the diseases beside the
        # COVID-19 studies, both as tall, over COMBAT's cohorts, as wide as both, to the page's right edge.
        row_height = draw.LAYOUT["composite_umap"][1]
        columns = (
            ("progenitor_funnel",),
            ("progenitor_states", "progenitor_sources"),
            ("progenitor_state_shares",),
        )
        edge = 0.0
        for column in columns:
            with self.subTest(column=column):
                self.assertEqual({draw.LAYOUT[name][0] for name in column}, {edge})
                self.assertEqual(
                    {draw.CELLS[name].width for name in column}, {draw.CELLS[column[0]].width}
                )
                self.assertAlmostEqual(sum(draw.CELLS[name].height for name in column), row_height)
                self.assertEqual(min(draw.LAYOUT[name][1] for name in column), 0.0)
                edge += draw.CELLS[column[0]].width
        diseases, covid, cohorts = (
            draw.CELLS[name]
            for name in ("progenitor_diseases", "progenitor_covid", "progenitor_cohorts")
        )
        self.assertEqual(draw.LAYOUT["progenitor_diseases"], (edge, cohorts.height))
        self.assertEqual(draw.LAYOUT["progenitor_covid"], (edge + diseases.width, cohorts.height))
        self.assertEqual(draw.LAYOUT["progenitor_cohorts"], (edge, 0.0))
        self.assertEqual(diseases.height, covid.height)
        self.assertAlmostEqual(diseases.height + cohorts.height, row_height)
        self.assertAlmostEqual(cohorts.width, diseases.width + covid.width)
        self.assertAlmostEqual(edge + cohorts.width, 160.0)
        # The diseases' and the COVID-19 studies' plots share their foot, so the labels under them line up.
        self.assertEqual(diseases.plot[1], covid.plot[1])
        self.assertEqual(draw.LAYOUT["interferon_score"][1], row_height)
        # The state map sits on the source map.
        self.assertEqual(
            draw.LAYOUT["progenitor_states"][1], draw.CELLS["progenitor_sources"].height
        )

    def test_the_row_under_the_schematic_shares_one_frame(self):
        # The storage chart and the query's map start and end at the same heights.
        row = ("storage_backends", "tabula_query")
        frames = {
            (draw.LAYOUT[name][1] + draw.CELLS[name].plot[1], draw.CELLS[name].plot[3])
            for name in row
        }
        bottom, top = draw.ROW_FRAME_MM
        self.assertEqual(frames, {(draw.LAYOUT[row[0]][1] + bottom, top - bottom)})

    def test_the_ridges_run_down_beside_both_rows_of_the_left_column(self):
        # Both rows of the left column start at the page's left edge, set their cells side by side and end where the
        # ridges begin; the ridges span both rows and end at the page's right edge.
        ridges_left, ridges_bottom = draw.LAYOUT["interferon_score"]
        ridges = draw.CELLS["interferon_score"]
        rows = (("storage_backends", "tabula_query"), ("composite_umap",))
        for row in rows:
            with self.subTest(row=row):
                edge = 0.0
                for name in row:
                    self.assertAlmostEqual(draw.LAYOUT[name][0], edge)
                    edge += draw.CELLS[name].width
                self.assertAlmostEqual(edge, ridges_left)
        tops = {draw.LAYOUT[name][1] + draw.CELLS[name].height for name in rows[0]}
        bottoms = {draw.LAYOUT[name][1] for name in rows[1]}
        self.assertEqual(tops, {ridges_bottom + ridges.height})
        self.assertEqual(bottoms, {ridges_bottom})
        self.assertAlmostEqual(ridges_left + ridges.width, 160.0)

    def test_overlapping_cells_are_an_error(self):
        layout = {**draw.LAYOUT, "storage_backends": (30.0, 116.0)}
        with mock.patch.object(draw, "LAYOUT", layout):
            with self.assertRaisesRegex(ValueError, "overlap"):
                draw.check_layout()

    def test_a_cell_past_the_page_width_is_an_error(self):
        layout = {**draw.LAYOUT, "composite_umap": (110.0, 0.0)}
        with mock.patch.object(draw, "LAYOUT", layout):
            with self.assertRaisesRegex(ValueError, "160 mm"):
                draw.check_layout()


class CompositeSketchTest(unittest.TestCase):
    def test_each_source_holds_every_gene_once_in_an_order_of_its_own(self):
        shared = tuple(range(len(draw.GENE_NAMES)))
        for order in draw.SOURCE_GENE_ORDERS:
            with self.subTest(order=order):
                self.assertEqual(sorted(order), list(shared))
                self.assertNotEqual(tuple(order), shared)
        self.assertEqual(
            len(set(map(tuple, draw.SOURCE_GENE_ORDERS))), len(draw.SOURCE_GENE_ORDERS)
        )

    def test_the_composite_takes_every_source_block_once_shuffled_across_sources(self):
        blocks_per_source = (len(range(8)) // draw.BLOCK_ROWS,) * len(draw.SOURCE_ROWS)
        expected = {(s, b) for s, n in enumerate(blocks_per_source) for b in range(n)}
        self.assertEqual(sorted(draw.BLOCK_ORDER), sorted(expected))
        self.assertEqual(len(set(draw.BLOCK_ORDER)), len(draw.BLOCK_ORDER))
        # Shuffled: no source's blocks run together, and the order is not the sources' own.
        sources = [s for s, _ in draw.BLOCK_ORDER]
        self.assertTrue(all(a != b for a, b in zip(sources, sources[1:])))
        self.assertNotEqual(list(draw.BLOCK_ORDER), sorted(draw.BLOCK_ORDER))
        self.assertEqual(sorted(draw.WITHIN_BLOCK_ORDER), list(range(draw.BLOCK_ROWS)))
        self.assertNotEqual(list(draw.WITHIN_BLOCK_ORDER), list(range(draw.BLOCK_ROWS)))

    def test_the_composites_rows_are_its_sources_rows_block_by_block_with_genes_in_one_order(self):
        heat = np.random.default_rng(0).uniform(0, 1, (48, 12))
        rows, sources = draw.composite_rows(heat)
        self.assertEqual(
            rows.shape, (len(draw.BLOCK_ORDER) * draw.BLOCK_ROWS, len(draw.GENE_COLUMNS))
        )
        for i, (source, block) in enumerate(draw.BLOCK_ORDER):
            start = i * draw.BLOCK_ROWS
            original = heat[draw.SOURCE_ROWS[source]][
                block * draw.BLOCK_ROWS : (block + 1) * draw.BLOCK_ROWS
            ]
            # The block's rows, permuted within it, with the named genes in the order A to D.
            np.testing.assert_allclose(
                rows[start : start + draw.BLOCK_ROWS],
                original[list(draw.WITHIN_BLOCK_ORDER)][:, list(draw.GENE_COLUMNS)],
            )
            self.assertEqual(sources[start : start + draw.BLOCK_ROWS], [source] * draw.BLOCK_ROWS)

    def test_the_user_store_reads_the_chain_up_to_the_graph_and_writes_the_rest(self):
        # The mounted reanalysis reclustered on the published graph and wrote new clusters and markers.
        self.assertEqual(draw.CHAIN[: draw.REUSED], ("genes", "loadings", "ANN index", "graph"))
        self.assertEqual(draw.CHAIN[draw.REUSED :], ("clusters", "markers"))
        fig = plt.figure()
        ax = g.mm_axes(fig, 0, 0, 20, 20)
        for name in draw.CHAIN:
            draw.chain_node(ax, name, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "no thumbnail"):
            draw.chain_node(ax, "UMAP", 0.0, 0.0)
        plt.close(fig)


class StudyNameTest(unittest.TestCase):
    def test_every_row_and_study_the_figure_draws_has_exactly_one_name(self):
        rows = json.loads((draw.HERE / "interferon_score.json").read_text())["rows"]
        labels = [draw.row_label(row) for row in rows]
        self.assertEqual(len(set(labels)), len(labels))
        effects = json.loads((draw.HERE / "progenitor_effects.json").read_text())
        for study in effects["covid"]:
            draw.study_name(study["stores"][0])

    def test_a_store_without_a_name_or_with_two_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "matches 0 study names"):
            draw.study_name("smith_2030_unknown_store")
        names = {**draw.STUDY_NAMES_BY_STORE, "wilk_2020_single": "Wilk again"}
        with mock.patch.object(draw, "STUDY_NAMES_BY_STORE", names):
            with self.assertRaisesRegex(ValueError, "matches 2 study names"):
                draw.study_name("wilk_2020_single_cell_atlas")

    def test_a_row_names_its_study_and_disease_unless_the_study_is_named_for_it(self):
        self.assertEqual(
            draw.row_label({"stores": ["he_2025_x"], "disease": "rheumatoid arthritis"}), "ALTRA RA"
        )
        self.assertEqual(
            draw.row_label(
                {
                    "stores": ["he_2025_x"],
                    "disease": "cytomegalovirus infection || rheumatoid arthritis",
                }
            ),
            "ALTRA CMV+RA",
        )
        self.assertEqual(
            draw.row_label(
                {"stores": ["unpub_0000_human_pbmc_glaucoma_x"], "disease": "open-angle glaucoma"}
            ),
            "Glaucoma atlas",
        )


class InterferonPanelTest(unittest.TestCase):
    def test_the_dashed_line_runs_between_the_last_interferon_row_and_the_first_control(self):
        data = json.loads((draw.HERE / "interferon_score.json").read_text())
        rows = data["rows"]
        n = len(rows)
        controls = next(i for i, row in enumerate(rows) if not row["interferon"])
        fig, ax = render("interferon_score")
        dashed = [
            line
            for line in ax.figure.findobj(lambda artist: hasattr(artist, "get_linestyle"))
            if getattr(line, "get_ydata", None) and line.get_linestyle() == "--"
        ]
        (ridge_line,) = [line for line in ax.lines if line.get_linestyle() == "--"]
        # Row i runs from its base, n - 1 - i, to the base above; the line sits on the last interferon row's base.
        self.assertEqual(ridge_line.get_ydata()[0], n - controls)
        labels = {t.get_text(): t.xy[1] for t in ax.texts if isinstance(t, Annotation)}
        first_control = draw.row_label(rows[controls])
        last_interferon = draw.row_label(rows[controls - 1])
        self.assertLess(labels[first_control], ridge_line.get_ydata()[0])
        self.assertGreaterEqual(labels[last_interferon], ridge_line.get_ydata()[0])
        self.assertTrue(dashed)
        plt.close(fig)

    def test_rows_and_the_matrix_share_their_order_with_interferon_diseases_first(self):
        data = json.loads((draw.HERE / "interferon_score.json").read_text())
        cell = draw.CELLS["interferon_score"]
        fs.apply()
        fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
        ax = draw.draw_panel(fig, "interferon_score", 0.0, 0.0)
        matrix = next(a for a in fig.axes if a.get_gid() == "interferon_score_lineages")
        self.assertEqual(matrix.get_ylim(), ax.get_ylim())
        # The matrix comes first, left of the ridges, and the rows' names left of it.
        self.assertLess(matrix.get_position().x1, ax.get_position().x0)
        (image,) = matrix.get_images()
        values = np.ma.filled(image.get_array(), np.nan)
        rows = data["rows"]
        self.assertEqual(values.shape, (len(rows), len(data["lineages"])))
        # The image's top row is the panel's first row.
        first = [
            np.nan if v is None else v
            for v in (rows[0]["auc"][lineage] for lineage in data["lineages"])
        ]
        np.testing.assert_allclose(values[-1], first)
        annotations = [t for t in ax.texts if isinstance(t, Annotation)]
        labels = [t.get_text() for t in sorted(annotations, key=lambda t: -t.xy[1])]
        self.assertEqual(labels, [draw.row_label(row) for row in rows])
        flags = [row["interferon"] for row in rows]
        self.assertEqual(flags, sorted(flags, reverse=True))
        plt.close(fig)

    def test_combats_sepsis_cohort_draws_as_a_control_row_of_its_own(self):
        # The store labels COMBAT's sepsis cohort COVID-19; from the cohort its authors recorded, its donors leave the
        # COVID-19 row for a sepsis row among the controls, and the two low-density neutrophil samples leave.
        data = json.loads((draw.HERE / "interferon_score.json").read_text())
        combat = {
            row["disease"]: row
            for row in data["rows"]
            if draw.study_name(row["stores"][0]) == "COMBAT"
        }
        self.assertEqual(sorted(combat), ["COVID-19", "influenza", "sepsis"])
        self.assertFalse(combat["sepsis"]["interferon"])
        self.assertEqual(draw.row_label(combat["sepsis"]), "COMBAT sepsis")
        self.assertEqual(len(combat["sepsis"]["ridge"]["cases"]), 23)
        self.assertEqual(len(combat["COVID-19"]["ridge"]["cases"]), 75)


def render(name: str):
    cell = draw.CELLS[name]
    fs.apply()
    fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
    return fig, draw.draw_panel(fig, name, 0.0, 0.0)


class ProgenitorFunnelTest(unittest.TestCase):
    def test_band_widths_follow_the_square_root_of_cells_up_to_the_widest_band(self):
        widest = 4e6
        self.assertAlmostEqual(draw.funnel_width(widest, widest), draw.FUNNEL_WIDEST_MM)
        # A quarter of the cells is half as wide, a hundredth a tenth.
        self.assertAlmostEqual(draw.funnel_width(1e6, widest), draw.FUNNEL_WIDEST_MM / 2)
        self.assertAlmostEqual(draw.funnel_width(4e4, widest), draw.FUNNEL_WIDEST_MM / 10)
        for cells in (0, -5, widest * 1.01):
            with self.subTest(cells=cells), self.assertRaisesRegex(ValueError, "funnel band"):
                draw.funnel_width(cells, widest)

    def test_shares_keep_two_significant_digits(self):
        self.assertEqual(draw.share_text(33_138 / 31_884_117), "0.10%")
        self.assertEqual(draw.share_text(0.0237), "2.4%")
        self.assertEqual(draw.share_text(790_157 / 1_011_884), "78%")

    def bands(self, ax, colour: str) -> list:
        """A source's bands from the top, told from its joins by their tint, with each one's centre height."""
        from matplotlib.colors import to_hex
        from matplotlib.patches import Polygon

        band = to_hex(fs.tint(colour, draw.FUNNEL_BAND_TINT))
        found = [
            p
            for p in ax.patches
            if isinstance(p, Polygon) and p.get_fill() and to_hex(p.get_facecolor()) == band
        ]
        # A band polygon repeats its first corner, so its centre is halfway between its lowest and highest corners.
        return sorted(
            ((p, (p.get_xy()[:, 1].max() + p.get_xy()[:, 1].min()) / 2) for p in found),
            key=lambda item: -item[1],
        )

    def test_both_funnels_draw_three_levels_as_wide_as_their_cells_with_their_counts(self):
        data = json.loads((draw.HERE / "progenitor_effects.json").read_text())
        widest = max(source["cellsRead"] for source in data["sources"].values())
        fig, ax = render("progenitor_funnel")
        texts = {t.get_text(): t for t in ax.texts}
        for (source, colour, label), centre in zip(draw.SOURCES, draw.FUNNEL_CENTRES_MM):
            with self.subTest(source=source):
                bands = self.bands(ax, colour)
                self.assertEqual(len(bands), len(draw.FUNNEL_LEVELS))
                for (patch, y), key in zip(bands, draw.FUNNEL_LEVELS):
                    cells = data["sources"][source][key]
                    xs = patch.get_xy()[:, 0]
                    width = xs.max() - xs.min()
                    self.assertAlmostEqual(width, draw.FUNNEL_WIDEST_MM * (cells / widest) ** 0.5)
                    self.assertAlmostEqual((xs.max() + xs.min()) / 2, centre)
                    # Each count in the larger type: inside a band wide enough to hold it, right of a narrower one.
                    count = texts[draw.cell_count(cells)]
                    self.assertEqual((count.get_position()[1], count.get_fontsize()), (y, fs.LABEL))
                    if width >= draw.FUNNEL_INSIDE_MM:
                        self.assertEqual(
                            (count.get_position()[0], count.get_ha()), (centre, "center")
                        )
                    else:
                        self.assertGreater(count.get_position()[0], xs.max())
                        self.assertEqual(count.get_ha(), "left")
                self.assertIn(label, texts)
        plt.close(fig)

    def test_the_screen_keeps_a_thousandth_of_all_blood_and_most_of_the_cd34_cells(self):
        # 33,138 of 31,884,117 blood cells and 790,157 of 1,011,884 CD34+ cells pass the selection; the shares sit
        # under their counts in slate, and no other share or definition is written.
        fig, ax = render("progenitor_funnel")
        texts = {t.get_text(): t for t in ax.texts}
        for count, share in (("33.1k", "0.10%"), ("790k", "78%")):
            self.assertEqual(texts[share].get_color(), fs.SLATE)
            self.assertEqual(texts[share].get_position()[0], texts[count].get_position()[0])
            self.assertLess(texts[share].get_position()[1], texts[count].get_position()[1])
        self.assertFalse(
            [
                t
                for t in texts
                if "marker" in t or "label" in t or "%" in t and t not in ("0.10%", "78%")
            ]
        )
        plt.close(fig)

    def test_an_arrow_per_step_runs_between_the_levels_beside_what_the_step_did(self):
        fig, ax = render("progenitor_funnel")
        texts = {t.get_text(): t for t in ax.texts}
        levels = [y for _, y in self.bands(ax, draw.BLOOD_SOURCE)]
        arrows = sorted(
            (t for t in ax.texts if isinstance(t, Annotation) and t.arrow_patch is not None),
            key=lambda arrow: -arrow.xyann[1],
        )
        self.assertEqual(len(arrows), 2)
        # What each step did, with no time: the figure compares no times here.
        words = ("screen for\nprogenitors", "integrate and\ncluster")
        self.assertFalse([t for t in texts if "min" in t])
        for arrow, upper, lower, step in zip(arrows, levels, levels[1:], words):
            self.assertEqual(arrow.xy[0], arrow.xyann[0])
            self.assertTrue(upper > arrow.xyann[1] > arrow.xy[1] > lower)
            x, y = texts[step].get_position()
            self.assertGreater(x, arrow.xy[0])
            self.assertAlmostEqual(y, (arrow.xyann[1] + arrow.xy[1]) / 2)
        plt.close(fig)

    def test_the_sampled_cd34_join_is_outlined_and_every_other_join_filled(self):
        from matplotlib.patches import Polygon

        fig, ax = render("progenitor_funnel")
        outlined = [p for p in ax.patches if isinstance(p, Polygon) and p.get_facecolor()[3] == 0]
        # Drawn from the CD34+ datasets' 790k selected cells down to their 48.7k cells in progenitor states.
        (join,) = outlined
        centre = draw.FUNNEL_CENTRES_MM[1]
        xs = join.get_xy()[:, 0]
        self.assertAlmostEqual((xs.max() + xs.min()) / 2, centre, places=6)
        self.assertEqual(to_hex(join.get_edgecolor()), to_hex(draw.REFERENCE_SOURCE))
        plt.close(fig)

    def test_the_header_counts_both_kinds_of_dataset(self):
        data = json.loads((draw.HERE / "progenitor_effects.json").read_text())
        sources = data["sources"]
        self.assertEqual(sources["blood"]["stores"] + sources["enriched"]["stores"], 82)
        self.assertEqual(draw.header("progenitor_funnel"), "Progenitors in 82 datasets")

    def test_the_two_sources_add_up_to_the_cells_the_discovery_read(self):
        data = json.loads((draw.HERE / "progenitor_effects.json").read_text())
        sources = data["sources"]
        self.assertEqual(
            sources["blood"]["cellsRead"] + sources["enriched"]["cellsRead"],
            data["discovery"]["cellsRead"],
        )
        self.assertEqual(
            sources["blood"]["progenitors"] + sources["enriched"]["progenitors"],
            data["progenitorCells"],
        )
        # Every blood cell the selection keeps was copied out; the CD34+ datasets gave a sample of theirs.
        self.assertEqual(sources["blood"]["selected"], sources["blood"]["extracted"])
        self.assertLess(sources["enriched"]["extracted"], sources["enriched"]["selected"])


def panel_axes(name: str):
    """An empty plot of a panel's cell, for drawing it from data a test makes."""
    cell = draw.CELLS[name]
    fs.apply()
    fig = plt.figure(figsize=(cell.width * fs.MM, cell.height * fs.MM))
    return fig, fs.add_axes(fig, *cell.plot)


class PValueTest(unittest.TestCase):
    def test_p_values_keep_two_significant_digits_and_their_trailing_zero(self):
        self.assertEqual(draw.p_text(0.0102113), "P = 0.010")
        self.assertEqual(draw.p_text(0.0047), "P = 0.0047")
        self.assertEqual(draw.p_text(0.0129), "P = 0.013")
        # Under one in a thousand, in powers of ten.
        self.assertEqual(draw.p_text(7.0628e-05), "P = 7.1 × 10$^{\\mathregular{-5}}$")
        self.assertEqual(draw.p_text(0.00012), "P = 1.2 × 10$^{\\mathregular{-4}}$")
        self.assertEqual(draw.p_text(0.001), "P = 0.0010")


class ProgenitorCovidTest(unittest.TestCase):
    STUDIES = [
        {"healthyPer10k": 2.0, "covidPer10k": 6.0, "healthyDonors": 5, "covidDonors": 9},
        {"healthyPer10k": 30.0, "covidPer10k": 45.0, "healthyDonors": 12, "covidDonors": 40},
        {"healthyPer10k": 5.0, "covidPer10k": 4.0, "healthyDonors": 7, "covidDonors": 7},
    ]
    POOL = {"ratio": 1.4, "p": 0.0312}

    def test_each_study_joins_its_healthy_rate_to_its_covid_rate(self):
        fig, ax = panel_axes("progenitor_covid")
        draw.progenitor_covid(ax, {"covid": self.STUDIES, "selections": {"default": self.POOL}})
        studies = [line for line in ax.lines if tuple(line.get_xdata()) == (0, 1)]
        self.assertEqual(
            sorted(tuple(line.get_ydata()) for line in studies),
            [(2.0, 6.0), (5.0, 4.0), (30.0, 45.0)],
        )
        # Healthy donors' rates in the ridges' healthy blue at the left, COVID-19 donors' in their rust at the right.
        for x, colour, rates in (
            (0, draw.HEALTHY_RIDGE, [2.0, 30.0, 5.0]),
            (1, draw.DISEASE_RIDGE, [6.0, 45.0, 4.0]),
        ):
            (points,) = [c for c in ax.collections if set(c.get_offsets()[:, 0]) == {x}]
            self.assertEqual(sorted(points.get_offsets()[:, 1]), sorted(rates))
            self.assertEqual(to_hex(points.get_facecolor()[0]), to_hex(colour))
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["healthy", "COVID-19"])
        plt.close(fig)

    def test_a_bracket_over_both_sides_gives_the_pooled_ratios_p_value(self):
        fig, ax = panel_axes("progenitor_covid")
        draw.progenitor_covid(ax, {"covid": self.STUDIES, "selections": {"default": self.POOL}})
        (bracket,) = [line for line in ax.lines if len(line.get_xdata()) == 4]
        self.assertEqual(list(bracket.get_xdata()), [0, 0, 1, 1])
        # The bracket clears the highest rate, 45 per 10,000.
        self.assertGreater(min(bracket.get_ydata()), 45.0)
        self.assertEqual([t.get_text() for t in ax.texts], ["P = 0.031"])
        plt.close(fig)

    def test_each_points_area_grows_with_its_study_conditions_donors(self):
        studies = [
            {**study, "healthyDonors": donors, "covidDonors": 2 * donors}
            for study, donors in zip(self.STUDIES, (5, 20, 80))
        ]
        fig, ax = panel_axes("progenitor_covid")
        draw.progenitor_covid(ax, {"covid": studies, "selections": {"default": self.POOL}})
        for x, donors in ((0, [5, 20, 80]), (1, [10, 40, 160])):
            (points,) = [c for c in ax.collections if set(c.get_offsets()[:, 0]) == {x}]
            # fs.points draws a main marker at 1.4 times its size; the area is proportional to the donors.
            np.testing.assert_allclose(
                points.get_sizes(), [1.4 * draw.DONOR_AREA * d for d in donors]
            )
        self.assertEqual(
            [t.get_text() for t in ax.get_legend().get_texts()], ["10", "50", "100 donors"]
        )
        plt.close(fig)
        with self.assertRaisesRegex(ValueError, "no point"):
            draw.donor_area(0)

    def test_a_rate_off_the_axis_is_an_error(self):
        for rate in (0.0, draw.RATE_LIMITS[1] * 2):
            with self.subTest(rate=rate):
                fig, ax = panel_axes("progenitor_covid")
                with self.assertRaisesRegex(ValueError, "outside the axis"):
                    draw.progenitor_covid(
                        ax,
                        {
                            "covid": [
                                {
                                    "healthyPer10k": 1.0,
                                    "covidPer10k": rate,
                                    "healthyDonors": 5,
                                    "covidDonors": 5,
                                }
                            ],
                            "selections": {"default": self.POOL},
                        },
                    )
                plt.close(fig)

    def test_the_figure_gives_the_summarys_pool_and_names_studies_and_donors_in_its_header(self):
        # The P value of the summary's pooled odds ratio over the eight studies, 0.0054, and their 110 healthy donors
        # and 349 donors with COVID-19, COMBAT's sepsis and low-density neutrophil samples left out.
        fig, ax = render("progenitor_covid")
        self.assertEqual([t.get_text() for t in ax.texts], ["P = 0.0054"])
        self.assertEqual(draw.header("progenitor_covid"), "8 studies, 459 donors")
        plt.close(fig)

    def test_the_p_value_sits_just_under_the_top_of_the_plot(self):
        # The axis ends close over the bracket's P value: inside the plot, with no more than 2 mm above it.
        fig, ax = render("progenitor_covid")
        self.assertLessEqual(top_gap_mm(fig, ax), 2.0)
        self.assertGreaterEqual(top_gap_mm(fig, ax), 0.0)
        plt.close(fig)


def top_gap_mm(fig, ax) -> float:
    """The millimetres between the top of a plot's one text and the top of the plot."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    (text,) = ax.texts
    return (
        (ax.get_window_extent(renderer).y1 - text.get_window_extent(renderer).y1) / fig.dpi * 25.4
    )


def horizontal_lines(ax) -> list:
    """The plot's lines with two points at one height, which run across it."""
    return [
        line
        for line in ax.lines
        if len(line.get_xdata()) == 2
        and line.get_ydata()[0] == line.get_ydata()[1]
        and line.get_xdata()[0] != line.get_xdata()[1]
    ]


class ProgenitorDiseasesTest(unittest.TestCase):
    DISEASES = [
        {"disease": "sepsis", "studies": 1, "ratio": 5.0, "low": 1.2, "high": 20.0, "p": 0.03},
        {"disease": "COVID-19", "studies": 8, "ratio": 1.7, "low": 1.2, "high": 2.4, "p": 0.005},
        {
            "disease": "systemic lupus erythematosus",
            "studies": 1,
            "ratio": 0.8,
            "low": 0.6,
            "high": 1.1,
            "p": 0.16,
        },
    ]

    def test_each_disease_gets_a_row_from_the_highest_ratio_at_the_top_in_its_colour(self):
        fig, ax = panel_axes("progenitor_diseases")
        draw.progenitor_diseases(ax, {"diseases": self.DISEASES})
        # Named as in the interferon rows, top to bottom.
        names = {
            tick: label.get_text() for tick, label in zip(ax.get_yticks(), ax.get_yticklabels())
        }
        self.assertEqual([names[y] for y in (2, 1, 0)], ["sepsis", "COVID", "SLE"])
        # Each interval a line across its row and its ratio a point on it: sepsis in its wine, COVID-19 in its rust,
        # lupus in slate.
        lines = {line.get_ydata()[0]: line for line in horizontal_lines(ax)}
        for y, entry, colour in (
            (2, self.DISEASES[0], fs.WINE),
            (1, self.DISEASES[1], draw.DISEASE_RIDGE),
            (0, self.DISEASES[2], fs.SLATE),
        ):
            with self.subTest(disease=entry["disease"]):
                self.assertEqual(list(lines[y].get_xdata()), [entry["low"], entry["high"]])
                self.assertEqual(to_hex(lines[y].get_color()), to_hex(colour))
                (point,) = [c for c in ax.collections if c.get_offsets()[0][1] == y]
                self.assertEqual(list(point.get_offsets()[0]), [entry["ratio"], y])
                self.assertEqual(to_hex(point.get_facecolor()[0]), to_hex(colour))
        plt.close(fig)

    def test_a_dashed_line_marks_an_odds_ratio_of_one(self):
        fig, ax = panel_axes("progenitor_diseases")
        draw.progenitor_diseases(ax, {"diseases": self.DISEASES})
        (one,) = [line for line in ax.lines if list(line.get_xdata()) == [1.0, 1.0]]
        self.assertNotEqual(one.get_linestyle(), "-")
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_xlabel(), "odds ratio")
        plt.close(fig)

    def test_an_interval_off_the_axis_is_an_error(self):
        for low, high in ((draw.ODDS_LIMITS[0] / 2, 2.0), (0.5, draw.ODDS_LIMITS[1] * 2)):
            with self.subTest(low=low, high=high):
                fig, ax = panel_axes("progenitor_diseases")
                entry = {**self.DISEASES[0], "low": low, "high": high}
                with self.assertRaisesRegex(ValueError, "outside the axis"):
                    draw.progenitor_diseases(ax, {"diseases": [entry]})
                plt.close(fig)

    def test_a_disease_without_a_name_is_an_error(self):
        fig, ax = panel_axes("progenitor_diseases")
        with self.assertRaisesRegex(ValueError, "without a name"):
            draw.progenitor_diseases(ax, {"diseases": [{**self.DISEASES[0], "disease": "gout"}]})
        plt.close(fig)

    def test_the_figure_draws_the_summarys_nine_diseases(self):
        # The summary's diseases from the highest odds ratio down: sepsis 5.1, other respiratory disease 1.9, COVID-19
        # 1.69 over eight studies, clonal hematopoiesis 1.6, dermatomyositis 1.06, CMV 0.89, lupus 0.81, post-COVID-19
        # condition 0.75 and rheumatoid arthritis 0.72; colorectal cancer has no study with both sides.
        fig, ax = render("progenitor_diseases")
        names = {
            tick: label.get_text() for tick, label in zip(ax.get_yticks(), ax.get_yticklabels())
        }
        self.assertEqual(
            [names[y] for y in range(8, -1, -1)],
            ["sepsis", "resp.", "COVID", "CHIP", "JDM", "CMV", "SLE", "PCC", "RA"],
        )
        self.assertEqual(draw.header("progenitor_diseases"), "9 diseases")
        plt.close(fig)


class ProgenitorCohortsTest(unittest.TestCase):
    DATA = {
        "cohorts": {
            "cohorts": [
                {"cohort": "healthy", "donorsPer10k": [5.0, 10.0, 20.0], "per10k": 11.0},
                {"cohort": "mild", "donorsPer10k": [0.0, 12.0, 30.0], "per10k": 14.0},
                {"cohort": "severe", "donorsPer10k": [20.0, 40.0], "per10k": 30.0},
                {"cohort": "critical", "donorsPer10k": [50.0, 200.0], "per10k": 90.0},
                {"cohort": "sepsis", "donorsPer10k": [30.0, 600.0], "per10k": 70.0},
            ],
            "trend": {"ratio": 2.0, "low": 1.5, "high": 2.6, "p": 1.8e-13},
        }
    }

    def points_of(self, ax, x: int) -> list:
        return [
            c
            for c in ax.collections
            if np.all(np.abs(c.get_offsets()[:, 0] - x) <= draw.COHORT_SPREAD)
        ]

    def test_each_donor_is_a_point_in_its_cohorts_column_and_colour(self):
        fig, ax = panel_axes("progenitor_cohorts")
        draw.progenitor_cohorts(ax, self.DATA)
        for x, cohort in enumerate(self.DATA["cohorts"]["cohorts"]):
            with self.subTest(cohort=cohort["cohort"]):
                colour = to_hex(draw.COHORT_COLOURS[cohort["cohort"]])
                filled = [
                    p for p in self.points_of(ax, x) if to_hex(p.get_facecolor()[0]) == colour
                ]
                (points,) = filled
                rates = [rate for rate in cohort["donorsPer10k"] if rate > 0]
                self.assertEqual(sorted(points.get_offsets()[:, 1]), sorted(rates))
        # The mild donor without a progenitor sits open, in mild's colour, on the axis floor.
        (open_point,) = [
            p for p in self.points_of(ax, 1) if to_hex(p.get_facecolor()[0]) == to_hex(fs.WHITE)
        ]
        self.assertEqual(list(open_point.get_offsets()[:, 1]), [draw.COHORT_LIMITS[0]])
        self.assertEqual(to_hex(open_point.get_edgecolor()[0]), to_hex(draw.COHORT_COLOURS["mild"]))
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()],
            ["healthy", "mild", "severe", "critical", "sepsis"],
        )
        plt.close(fig)

    def test_severity_deepens_from_mild_to_critical_and_sepsis_keeps_its_wine(self):
        lightness = [
            sum(to_rgb_tuple(draw.COHORT_COLOURS[c])) for c in ("mild", "severe", "critical")
        ]
        self.assertTrue(all(a > b for a, b in zip(lightness, lightness[1:])), lightness)
        self.assertEqual(draw.COHORT_COLOURS["critical"], draw.DISEASE_RIDGE)
        self.assertEqual(draw.COHORT_COLOURS["sepsis"], draw.DISEASE_COLOURS["sepsis"])
        self.assertEqual(draw.COHORT_COLOURS["healthy"], draw.HEALTHY_RIDGE)

    def test_each_cohorts_bar_sits_at_its_pooled_rate(self):
        fig, ax = panel_axes("progenitor_cohorts")
        draw.progenitor_cohorts(ax, self.DATA)
        bars = sorted((line.get_xdata()[0], line.get_ydata()[0]) for line in horizontal_lines(ax))
        expected = [
            (x - draw.COHORT_BAR, cohort["per10k"])
            for x, cohort in enumerate(self.DATA["cohorts"]["cohorts"])
        ]
        np.testing.assert_allclose(bars, expected)
        plt.close(fig)

    def test_a_bracket_over_healthy_to_critical_gives_the_trends_p_value(self):
        fig, ax = panel_axes("progenitor_cohorts")
        draw.progenitor_cohorts(ax, self.DATA)
        (bracket,) = [line for line in ax.lines if len(line.get_xdata()) == 4]
        self.assertEqual(list(bracket.get_xdata()), [0, 0, 3, 3])
        # The bracket clears the highest donor under it, 200 per 10,000, though not sepsis's 600 beside it.
        self.assertGreater(min(bracket.get_ydata()), 200.0)
        self.assertEqual([t.get_text() for t in ax.texts], ["P = 1.8 × 10$^{\\mathregular{-13}}$"])
        plt.close(fig)

    def test_the_spread_of_points_is_the_same_on_every_render(self):
        offsets = []
        for _ in range(2):
            fig, ax = panel_axes("progenitor_cohorts")
            draw.progenitor_cohorts(ax, self.DATA)
            offsets.append(np.concatenate([c.get_offsets()[:, 0] for c in ax.collections]))
            plt.close(fig)
        np.testing.assert_array_equal(*offsets)

    def test_cohorts_out_of_order_are_an_error(self):
        cohorts = list(self.DATA["cohorts"]["cohorts"])
        cohorts[1], cohorts[2] = cohorts[2], cohorts[1]
        fig, ax = panel_axes("progenitor_cohorts")
        with self.assertRaisesRegex(ValueError, "not healthy volunteers"):
            draw.progenitor_cohorts(ax, {"cohorts": {**self.DATA["cohorts"], "cohorts": cohorts}})
        plt.close(fig)

    def test_a_rate_off_the_axis_is_an_error(self):
        for rate in (-1.0, draw.COHORT_LIMITS[1] * 2):
            with self.subTest(rate=rate):
                cohorts = [dict(c) for c in self.DATA["cohorts"]["cohorts"]]
                cohorts[4]["donorsPer10k"] = [rate]
                fig, ax = panel_axes("progenitor_cohorts")
                with self.assertRaisesRegex(ValueError, "outside the axis"):
                    draw.progenitor_cohorts(
                        ax, {"cohorts": {**self.DATA["cohorts"], "cohorts": cohorts}}
                    )
                plt.close(fig)

    def test_the_figure_draws_the_summarys_cohorts_with_the_p_value_inside_the_plot(self):
        # COMBAT's 10 healthy volunteers, 29 donors with mild, 32 with severe and 13 with critical COVID-19 and 22
        # with sepsis, and the trend's P value, 1.8e-13.
        fig, ax = render("progenitor_cohorts")
        self.assertEqual([len(c.get_offsets()) for c in ax.collections], [10, 28, 1, 32, 13, 22])
        self.assertEqual([t.get_text() for t in ax.texts], ["P = 1.8 × 10$^{\\mathregular{-13}}$"])
        self.assertGreaterEqual(top_gap_mm(fig, ax), 0.0)
        self.assertEqual(draw.header("progenitor_cohorts"), "COMBAT: COVID-19 and sepsis")
        plt.close(fig)


def to_rgb_tuple(colour: str) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    return to_rgb(colour)


class ProgenitorSourcesTest(unittest.TestCase):
    def test_the_source_map_has_a_header_with_its_legend_between_it_and_the_map(self):
        self.assertEqual(draw.header("progenitor_sources"), "Progenitor sources")
        fig, ax = render("progenitor_sources")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend = ax.get_legend().get_window_extent(renderer)
        plot = ax.get_window_extent(renderer)
        cell = draw.CELLS["progenitor_sources"]
        # The header's baseline, in the cell's millimetres from its foot, as render_figure places it.
        header_baseline = (cell.height - 1.0 - 2.1) * fs.MM * fig.dpi
        self.assertGreater(legend.y0, plot.y1)
        self.assertLess(legend.y1, header_baseline)
        self.assertEqual(
            [t.get_text() for t in ax.get_legend().get_texts()], ["all blood", "CD34+ only"]
        )
        plt.close(fig)


class LogAxisTest(unittest.TestCase):
    def test_the_log_axes_of_the_last_two_panels_tick_every_unit_of_each_decade(self):
        from matplotlib.ticker import LogLocator, NullFormatter

        for name, axis_name in (
            ("progenitor_state_shares", "xaxis"),
            ("progenitor_diseases", "xaxis"),
            ("progenitor_covid", "yaxis"),
            ("progenitor_cohorts", "yaxis"),
        ):
            with self.subTest(panel=name):
                fig, ax = render(name)
                axis = getattr(ax, axis_name)
                self.assertIsInstance(axis.get_minor_locator(), LogLocator)
                self.assertIsInstance(axis.get_minor_formatter(), NullFormatter)
                low, high = sorted(getattr(ax, f"get_{axis_name[0]}lim")())
                minor = [v for v in axis.get_minorticklocs() if low <= v <= high]
                # Between 1 and 10, the ticks at 2 to 9.
                np.testing.assert_allclose([v for v in minor if 1 < v < 10], np.arange(2, 10))
                plt.close(fig)

    def test_the_state_shares_draw_gridlines_at_their_decades(self):
        fig, ax = render("progenitor_state_shares")
        fig.canvas.draw()
        low, high = ax.get_xlim()
        gridded = [
            tick.get_loc()
            for tick in ax.xaxis.get_major_ticks()
            if tick.gridline.get_visible() and low <= tick.get_loc() <= high
        ]
        self.assertEqual(sorted(gridded), [0.1, 1, 10, 100])
        plt.close(fig)


class ProgenitorStateSharesTest(unittest.TestCase):
    SHARES = {
        "blood": {"HSC/MPP": 0.6, "erythroid": 0.3, "cycling": 0.1},
        "enriched": {"HSC/MPP": 0.5, "erythroid": 0.2, "cycling": 0.3},
    }

    def test_each_state_gets_a_row_with_both_shares_from_the_commonest_in_all_blood(self):
        fig, ax = panel_axes("progenitor_state_shares")
        draw.progenitor_state_shares(ax, {"stateShares": self.SHARES})
        labels = sorted(ax.get_yticklabels(), key=lambda t: -t.get_position()[1])
        self.assertEqual(
            [t.get_text() for t in labels],
            # HSC/MPP, the longest name, over two lines.
            ["HSC/\nMPP", draw.STATE_NAMES["erythroid"], draw.STATE_NAMES["cycling"]],
        )
        # All blood's shares filled in its blue, the CD34+ datasets' open in their grey, in percent.
        filled, open_ = sorted(ax.collections, key=lambda c: -c.get_zorder())
        self.assertEqual(to_hex(filled.get_facecolor()[0]), to_hex(draw.BLOOD_SOURCE))
        self.assertEqual(to_hex(open_.get_edgecolor()[0]), to_hex(draw.REFERENCE_SOURCE))
        self.assertEqual(to_hex(open_.get_facecolor()[0]), to_hex(fs.WHITE))
        rows = {y: x for x, y in filled.get_offsets().tolist()}
        np.testing.assert_allclose(
            [rows[y] for y in sorted(rows, reverse=True)], [60.0, 30.0, 10.0]
        )
        rows = {y: x for x, y in open_.get_offsets().tolist()}
        np.testing.assert_allclose(
            [rows[y] for y in sorted(rows, reverse=True)], [50.0, 20.0, 30.0]
        )
        # Each row's two shares are joined by a line.
        joins = sorted(
            (tuple(line.get_xdata()) for line in ax.lines if len(line.get_xdata()) == 2),
            key=lambda pair: -pair[0],
        )
        np.testing.assert_allclose(joins, [(60.0, 50.0), (30.0, 20.0), (10.0, 30.0)])
        self.assertEqual(
            [t.get_text() for t in ax.get_legend().get_texts()], ["all blood", "CD34+ only"]
        )
        plt.close(fig)

    def test_a_share_off_the_log_axis_is_an_error(self):
        shares = {
            "blood": {"HSC/MPP": 1.0, "erythroid": 0.0},
            "enriched": {"HSC/MPP": 0.5, "erythroid": 0.5},
        }
        fig, ax = panel_axes("progenitor_state_shares")
        with self.assertRaisesRegex(ValueError, "outside the axis"):
            draw.progenitor_state_shares(ax, {"stateShares": shares})
        plt.close(fig)

    def test_the_figure_draws_the_summarys_shares(self):
        # From the summary: HSC/MPP 48.3% of all blood's progenitors against 51.8% of the CD34+ cells', and the
        # megakaryocytic state, the one that differs most, 5.3% against 2.5%.
        data = json.loads((draw.HERE / "progenitor_effects.json").read_text())
        shares = data["stateShares"]
        self.assertAlmostEqual(100 * shares["blood"]["HSC/MPP"], 48.3, places=1)
        self.assertAlmostEqual(100 * shares["enriched"]["HSC/MPP"], 51.8, places=1)
        self.assertAlmostEqual(100 * shares["blood"]["megakaryocyte"], 5.3, places=1)
        self.assertAlmostEqual(100 * shares["enriched"]["megakaryocyte"], 2.5, places=1)
        fig, ax = render("progenitor_state_shares")
        filled = max(ax.collections, key=lambda c: c.get_zorder())
        self.assertAlmostEqual(
            max(x for x, _ in filled.get_offsets().tolist()), 100 * shares["blood"]["HSC/MPP"]
        )
        self.assertEqual(draw.header("progenitor_state_shares"), "States by source")
        plt.close(fig)


def bars(ax):
    """The drawn bars of a chart, left to right."""
    from matplotlib.patches import Rectangle

    return sorted(
        (p for p in ax.patches if isinstance(p, Rectangle) and p.get_height() > 0),
        key=lambda p: p.get_x(),
    )


class NotSupportedTest(unittest.TestCase):
    def test_a_tool_without_a_reader_for_a_backend_keeps_its_slot_with_a_triangle(self):
        data = json.loads((draw.HERE / "storage_backends.json").read_text())
        missing = [
            (tool["key"], backend)
            for tool in data["tools"]
            for backend, _ in draw.BACKENDS
            if tool[backend] is None
        ]
        self.assertEqual(missing, [("bpcells", "objectStorage")])
        fig, ax = render("storage_backends")
        triangles = [line for line in ax.lines if line.get_marker() == "v"]
        self.assertEqual(len(triangles), 1)
        # BPCells' slot is the middle of object storage's three: between CyteArc's bar and Scanpy's.
        cytearc, scanpy = (p.get_x() + p.get_width() / 2 for p in bars(ax)[3:])
        self.assertAlmostEqual(triangles[0].get_xdata()[0], (cytearc + scanpy) / 2)
        self.assertIn("not supported", [t.get_text() for t in ax.texts])
        plt.close(fig)


class QueryCostTest(unittest.TestCase):
    def test_the_query_charts_draw_each_row_and_its_ratio_over_the_cytebase_query(self):
        data = json.loads((draw.HERE / "tabula_query.json").read_text())
        fig, ax = render("tabula_query")
        charts = {a.get_gid(): a for a in fig.axes}
        for name, field, scale, ratio in (
            ("tabula_query_time", "seconds", 1 / 60, "timeOverQuery"),
            ("tabula_query_data", "bytes", 1e-9, "dataOverQuery"),
        ):
            with self.subTest(chart=name):
                chart = charts[name]
                self.assertAlmostEqual(chart.get_position().y0, ax.get_position().y0)
                self.assertAlmostEqual(chart.get_position().height, ax.get_position().height)
                top = chart.get_ylim()[1]
                expected = [
                    draw.shown_height(chart, row[field]["mean"] * scale, top)
                    for row in data["rows"]
                ]
                np.testing.assert_allclose([b.get_height() for b in bars(chart)], expected)
                # No value over a bar: only each later row's ratio over the CyteBase query.
                self.assertEqual(
                    sorted(t.get_text() for t in chart.texts),
                    sorted(f"×{draw.amount(row[ratio])}" for row in data["rows"][1:]),
                )
        self.assertEqual(charts["tabula_query_time"].get_ylabel(), "time (min)")
        plt.close(fig)

    def test_a_bar_too_short_to_see_is_drawn_at_the_least_height(self):
        fig, ax = render("tabula_query")
        chart = next(a for a in fig.axes if a.get_gid() == "tabula_query_data")
        smallest = min(b.get_height() for b in bars(chart))
        top = chart.get_ylim()[1]
        self.assertAlmostEqual(smallest, draw.MIN_BAR_MM * top / draw.height_mm(chart))
        plt.close(fig)


class BarGeometryTest(unittest.TestCase):
    def test_every_bar_chart_draws_bars_alike(self):
        # Bars as wide as each other in every chart, neighbours in a group as far apart, on axes in millimetres.
        names = {
            "storage_backends",
            "tabula_query_time",
            "tabula_query_data",
            "composite_umap_disk",
            "composite_umap_memory",
        }
        charts = []
        for name in ("storage_backends", "tabula_query", "composite_umap"):
            fig, _ = render(name)
            charts += [(a.get_gid(), a) for a in fig.axes if a.get_gid() in names]
        self.assertEqual(len(charts), 5)
        for name, chart in charts:
            with self.subTest(chart=chart.get_gid() or name):
                drawn = bars(chart)
                for bar in drawn:
                    self.assertAlmostEqual(bar.get_width(), draw.BAR_MM)
                gaps = [b.get_x() - (a.get_x() + a.get_width()) for a, b in zip(drawn, drawn[1:])]
                # Neighbours in a group, a slot kept empty for a tool that cannot read the backend, and the two
                # storage backends' groups.
                allowed = (
                    draw.BAR_GAP_MM,
                    2 * draw.BAR_GAP_MM + draw.BAR_MM,
                    draw.BACKEND_GROUP_GAP_MM,
                )
                for gap in gaps:
                    self.assertTrue(any(np.isclose(gap, a) for a in allowed), gaps)
                self.assertAlmostEqual(chart.get_xlim()[1], draw.width_mm(chart))
        plt.close("all")


def composite_raster_present() -> bool:
    data = json.loads((draw.HERE / "composite_footprint.json").read_text())
    return (draw.REPO / data["umap"]["raster"]).is_file()


@unittest.skipUnless(
    composite_raster_present(), "the composite's UMAP raster is not on this machine"
)
class IntegrationStripTest(unittest.TestCase):
    INTEGRATION = {
        "unintegrated": {"studyMixing": 0.2, "sameType": 0.97},
        "integrated": {"studyMixing": 0.6, "sameType": 0.95},
    }

    def strip(self, integration: dict):
        fs.apply()
        fig = plt.figure(figsize=(40 * fs.MM, 40 * fs.MM))
        ax = fs.add_axes(fig, 15.0, 10.0, draw.INTEGRATION_WIDTH_MM, 26.0)
        draw.integration_strip(ax, integration)
        return fig, ax

    def test_each_metric_shows_its_unintegrated_and_harmony_values_side_by_side_in_its_column(self):
        fig, ax = self.strip(self.INTEGRATION)
        # The metrics along x, named under their columns; the scale from 0 to 1 along y.
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()], ["batch\nentropy", "kNN\npurity"]
        )
        self.assertEqual(list(ax.get_yticks()), [0.0, 0.5, 1.0])
        # No line joins the two values; each sits beside its column's centre, the unintegrated one left, open in
        # slate, the Harmony one right, filled in CyteArc's blue, so neither hides the other even when they are equal.
        self.assertFalse(
            [line for line in ax.lines if line.get_visible() and len(line.get_xdata()) == 2]
        )
        unintegrated, harmony = (
            next(c for c in ax.collections if to_hex(c.get_facecolor()[0]) == to_hex(colour))
            for colour in (fs.WHITE, draw.CYTEARC_BAR["facecolor"])
        )
        self.assertEqual(to_hex(unintegrated.get_edgecolor()[0]), to_hex(fs.SLATE))
        dodge = draw.INTEGRATION_DODGE
        np.testing.assert_allclose(unintegrated.get_offsets(), [[-dodge, 0.2], [1 - dodge, 0.97]])
        np.testing.assert_allclose(harmony.get_offsets(), [[dodge, 0.6], [1 + dodge, 0.95]])
        self.assertGreater(dodge, 0.1)
        # The legend sits over the chart.
        legend = ax.get_legend()
        self.assertEqual([t.get_text() for t in legend.get_texts()], ["unintegrated", "Harmony"])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        self.assertGreaterEqual(
            legend.get_window_extent(renderer).y0, ax.get_window_extent(renderer).y1
        )
        plt.close(fig)

    def test_a_metric_off_the_unit_scale_is_an_error(self):
        broken = {**self.INTEGRATION, "integrated": {"studyMixing": 1.2, "sameType": 0.95}}
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.strip(broken)
        plt.close("all")

    def test_the_figure_draws_the_records_metrics_in_the_maps_row(self):
        # The cohort analysis's record: batch entropy 0.18 unintegrated and 0.53 with Harmony; kNN purity 0.970 and
        # 0.965.
        data = json.loads((draw.HERE / "composite_footprint.json").read_text())
        integration = data["integration"]
        self.assertAlmostEqual(integration["unintegrated"]["studyMixing"], 0.182, places=3)
        self.assertAlmostEqual(integration["integrated"]["studyMixing"], 0.533, places=3)
        self.assertAlmostEqual(integration["unintegrated"]["sameType"], 0.970, places=3)
        self.assertAlmostEqual(integration["integrated"]["sameType"], 0.965, places=3)
        fig, ax = render("composite_umap")
        axes = {a.get_gid(): a.get_position() for a in fig.axes}
        strip, names, disk = (
            axes["composite_umap_integration"],
            axes["composite_umap_studies"],
            axes["composite_umap_disk"],
        )
        # In the map's row: right of the study list, left of the disk chart, overlapping the map's height, standing
        # on the charts' foot and shorter than them, which leaves room for its legend over it.
        self.assertGreater(strip.x0, names.x0)
        self.assertLess(strip.x1, disk.x0)
        self.assertLess(strip.y0, ax.get_position().y1)
        self.assertGreater(strip.y1, ax.get_position().y0)
        self.assertAlmostEqual(strip.y0, disk.y0)
        self.assertLess(strip.y1, disk.y1)
        self.assertEqual(draw.header("composite_umap"), "4.71M cells from 10 blood studies")
        plt.close(fig)


class CompositeChartsTest(unittest.TestCase):
    def setUp(self):
        # The disk and memory charts sit in the composite's panel beside its map.
        self.data = json.loads((draw.HERE / "composite_footprint.json").read_text())
        fig, _ = render("composite_umap")
        self.addCleanup(plt.close, fig)
        charts = {a.get_gid(): a for a in fig.axes}
        self.charts = {group: charts[f"composite_umap_{group}"] for group in ("disk", "memory")}

    def bars(self, group: str) -> list:
        return sorted(
            (
                p
                for p in self.charts[group].patches
                if isinstance(p, Rectangle) and p.get_height() > 0
            ),
            key=lambda p: p.get_x(),
        )

    def test_each_route_keeps_one_colour_in_both_charts_and_scanpys_routes_share_its_hue(self):
        scanpy = fs.ENTITIES["scanpy"].colour
        expected = {
            "composite": draw.CYTEARC_BAR["facecolor"],
            "bpcells": fs.bar_style(fs.ENTITIES["bpcells"].colour)["facecolor"],
            "on_disk": fs.bar_style(scanpy)["facecolor"],
            "in_memory": fs.tint(scanpy, 0.4),
        }
        for group in self.charts:
            bars = self.bars(group)
            self.assertEqual(len(bars), len(draw.ROUTE_ORDER))
            for bar, key in zip(bars, draw.ROUTE_ORDER):
                with self.subTest(group=group, route=key):
                    self.assertEqual(to_hex(bar.get_facecolor()), to_hex(expected[key]))
        self.assertEqual(draw.ROUTE_NAMES["in_memory"], "AnnData (memory)")

    def test_disk_is_in_gigabytes_and_memory_in_gibibytes(self):
        routes = {route["key"]: route for route in self.data["routes"]}
        disk_gib = sum(part["GiB"] for part in routes["on_disk"]["disk"])
        index = draw.ROUTE_ORDER.index("on_disk")
        disk_bar, memory_bar = self.bars("disk")[index], self.bars("memory")[index]
        self.assertAlmostEqual(disk_bar.get_height(), disk_gib * 1024**3 / 1e9)
        self.assertAlmostEqual(memory_bar.get_height(), routes["on_disk"]["memoryGiB"]["mean"])
        self.assertIn("(GB)", self.charts["disk"].get_ylabel())
        self.assertIn("(GiB)", self.charts["memory"].get_ylabel())

    def test_each_other_route_carries_its_value_over_the_composites(self):
        routes = {route["key"]: route for route in self.data["routes"]}
        values = {
            "disk": {k: sum(part["GiB"] for part in r["disk"]) for k, r in routes.items()},
            "memory": {k: r["memoryGiB"]["mean"] for k, r in routes.items()},
        }
        for group, chart in self.charts.items():
            with self.subTest(group=group):
                ratios = [
                    values[group][k] / values[group]["composite"]
                    for k in ("bpcells", "on_disk", "in_memory")
                ]
                labels = sorted(t.get_text() for t in chart.texts if t.get_text().startswith("×"))
                self.assertEqual(
                    labels, sorted(f"×{r:.0f}" if r >= 10 else f"×{r:.1f}" for r in ratios)
                )
                # The composite needs less of both than either other route.
                self.assertTrue(all(r > 1 for r in ratios))

    def test_bars_show_their_values_with_no_value_written_over_them(self):
        routes = {route["key"]: route for route in self.data["routes"]}
        for group, chart in self.charts.items():
            with self.subTest(group=group):
                if group == "disk":
                    values = [
                        sum(part["GiB"] for part in routes[key]["disk"]) * draw.GIB_IN_GB
                        for key in draw.ROUTE_ORDER
                    ]
                else:
                    values = [routes[key]["memoryGiB"]["mean"] for key in draw.ROUTE_ORDER]
                top = chart.get_ylim()[1]
                np.testing.assert_allclose(
                    [b.get_height() for b in self.bars(group)],
                    [draw.shown_height(chart, value, top) for value in values],
                )
                self.assertTrue(all(t.get_text().startswith("×") for t in chart.texts))


def raster_present() -> bool:
    data = json.loads((draw.HERE / "tabula_query.json").read_text())
    return (draw.REPO / data["raster"]).is_file()


@unittest.skipUnless(raster_present(), "the Tabula Sapiens query raster is not on this machine")
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
