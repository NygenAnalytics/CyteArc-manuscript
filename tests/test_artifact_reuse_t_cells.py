import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from manuscript.figures import style as fs
from manuscript.figures.artifact_reuse import draw, prepare

T_CELL_PANELS = ("t_cell_map", "t_cell_reference_comparison", "t_cell_proteins")
SHORT_NAMES = [short for short, _ in draw.T_CELL_SUBTYPES.values()]


def absolute_plot(name: str) -> tuple[float, float, float, float]:
    """A panel's plot as left, bottom, right and top in millimetres from the figure's lower-left corner."""
    left, bottom = draw.LAYOUT[name]
    x, y, w, h = draw.CELLS[name].plot
    return left + x, bottom + y, left + x + w, bottom + y + h


# Cells per subtype in the reference job's record: COMBAT's subtypes, and the subtype each Hao cell received, with the
# cells that received none.
COMBAT_CELLS = {s: 100 + i for i, s in enumerate(prepare.T_CELL_ORDER)}
HAO_CELLS = {s: 10 + i for i, s in enumerate(prepare.T_CELL_ORDER)}


class ProteinPrepareTest(unittest.TestCase):
    """The protein panel reads each study's table under that study's own protein names."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "reference"
        self.folder.mkdir()
        for name, value in {"ROOT": self.root, "T_CELL_REFERENCE": self.folder}.items():
            patch = mock.patch.object(prepare, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        record = {
            "status": "completed",
            "result": {
                "adtNormalization": "cell",
                "subtypeCells": COMBAT_CELLS,
                "queries": {"hao": {"subtypeCells": {**HAO_CELLS, "unassigned": 5}}},
            },
        }
        (self.folder / "result.json").write_text(json.dumps(record))

    def table(self, name: str, column: int, cells: dict[str, int], base: float) -> None:
        """A protein table with every subtype's median of protein j set to base + subtype index + 10 j, under the
        study's names, and rows for the cells that received no subtype."""
        with (self.folder / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cells", "cluster", "protein", "medianClr", "meanClr"])
            for i, (subtype, n) in enumerate(cells.items()):
                for j, names in enumerate(prepare.T_CELL_MARKERS.values()):
                    writer.writerow([n, subtype, names[column], base + i + 10 * j, 0.0])
                writer.writerow([n, subtype, "CD3", 9.0, 0.0])

    def write_both(self, hao_cells: dict[str, int] | None = None) -> None:
        self.table("t_cell_reference_reference_proteins.csv", 0, COMBAT_CELLS, 0.0)
        self.table(
            "t_cell_reference_hao_proteins.csv",
            1,
            {**(hao_cells or HAO_CELLS), "unassigned": 5},
            1000.0,
        )

    def test_each_study_is_read_under_its_own_protein_names(self):
        self.write_both()
        data = prepare.t_cell_proteins()
        combat, hao = data["reference"], data["query"]
        self.assertEqual([combat["study"], hao["study"]], ["COMBAT", "Hao"])
        expected = [[i + 10 * j for j in range(9)] for i in range(7)]
        self.assertEqual(combat["medianClr"], expected)
        self.assertEqual(hao["medianClr"], [[1000 + v for v in row] for row in expected])
        # The Vα7.2 antibody under each panel's name.
        self.assertEqual(combat["proteins"][7], "TCR-Va7.2")
        self.assertEqual(hao["proteins"][7], "TCR-V-7.2")
        self.assertEqual(data["markers"][7], "Vα7.2")

    def test_hao_cells_that_received_no_subtype_are_left_out(self):
        self.write_both()
        data = prepare.t_cell_proteins()
        self.assertEqual(data["reference"]["cells"], 721)
        self.assertEqual(data["query"]["cells"], 91)

    def test_a_table_of_other_cells_than_the_record_is_refused(self):
        self.write_both({**HAO_CELLS, "Treg": HAO_CELLS["Treg"] + 1})
        with self.assertRaisesRegex(ValueError, "hao_proteins.csv summarizes other cells"):
            prepare.t_cell_proteins()

    def test_a_table_under_the_other_studys_names_is_refused(self):
        # Hao's table written under COMBAT's names lacks Hao's TCR-V-7.2.
        self.table("t_cell_reference_reference_proteins.csv", 0, COMBAT_CELLS, 0.0)
        self.table("t_cell_reference_hao_proteins.csv", 0, {**HAO_CELLS, "unassigned": 5}, 1000.0)
        with self.assertRaisesRegex(ValueError, "hao_proteins.csv lacks .*CD4-1"):
            prepare.t_cell_proteins()


def protein_data(combat, hao) -> dict:
    return {
        "subtypes": list(draw.T_CELL_SUBTYPES),
        "markers": list(prepare.T_CELL_MARKERS),
        "reference": {"study": "COMBAT", "medianClr": combat},
        "query": {"study": "Hao", "medianClr": hao},
    }


def standardized(values) -> np.ndarray:
    """Each column's values in standard deviations from the column's mean, worked out apart from draw.py."""
    values = np.asarray(values, dtype=float)
    return (values - values.mean(axis=0)) / values.std(axis=0)


class ProteinDrawTest(unittest.TestCase):
    def setUp(self):
        fs.apply()
        self.fig = plt.figure(figsize=(160 * fs.MM, 30 * fs.MM))
        self.addCleanup(plt.close, self.fig)
        left, _, right, _ = absolute_plot("t_cell_proteins")
        self.ax = fs.add_axes(self.fig, left, 10.0, right - left, 14.0)
        self.ax.set_gid("t_cell_proteins")
        rng = np.random.default_rng(0)
        self.combat = rng.normal(size=(7, 9)).tolist()
        self.hao = rng.normal(size=(7, 9)).tolist()

    def right_axes(self):
        (right,) = [a for a in self.fig.axes if a is not self.ax]
        return right

    def test_hao_is_drawn_left_under_its_comparison_and_combat_right_under_its_map(self):
        draw.t_cell_proteins(self.ax, protein_data(self.combat, self.hao))
        np.testing.assert_allclose(self.ax.images[0].get_array(), standardized(self.hao))
        np.testing.assert_allclose(
            self.right_axes().images[0].get_array(), standardized(self.combat)
        )
        self.assertEqual(
            [self.ax.texts[0].get_text(), self.right_axes().texts[0].get_text()],
            ["Hao ADT", "COMBAT ADT"],
        )

    def test_each_heatmap_scales_its_own_study(self):
        # Hao's values are COMBAT's scaled and shifted, so each study standardized alone gives one image; one shared
        # standardization would not.
        hao = (3.0 * np.asarray(self.combat) + 5.0).tolist()
        draw.t_cell_proteins(self.ax, protein_data(self.combat, hao))
        np.testing.assert_allclose(
            self.ax.images[0].get_array(), self.right_axes().images[0].get_array()
        )

    def test_a_protein_in_standard_deviations_by_hand(self):
        # Values 0 to 6 have mean 3 and standard deviation 2.
        column = [[float(i)] + row[1:] for i, row in enumerate(self.hao)]
        draw.t_cell_proteins(self.ax, protein_data(self.combat, column))
        np.testing.assert_allclose(
            np.asarray(self.ax.images[0].get_array())[:, 0], [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        )

    def test_combat_heatmap_sits_level_with_the_maps_beside_haos(self):
        draw.t_cell_proteins(self.ax, protein_data(self.combat, self.hao))
        # Each plot's left, bottom, width and height in millimetres on the 160 by 30 mm figure.
        left_box, right_box = (
            np.asarray(axis.get_position().bounds) * [160, 30, 160, 30]
            for axis in (self.ax, self.right_axes())
        )
        self.assertAlmostEqual(right_box[0] - left_box[0], draw.T_CELL_HEATMAP_OFFSET_MM)
        np.testing.assert_allclose(right_box[1:], left_box[1:])
        self.assertAlmostEqual(right_box[0], draw.MAP_COLUMN_LEFT + draw.GRAPH_MAP_LEFT, places=6)

    def test_the_names_sit_left_of_hao_only(self):
        draw.t_cell_proteins(self.ax, protein_data(self.combat, self.hao))
        right = self.right_axes()
        self.assertEqual([t.get_text() for t in self.ax.get_yticklabels()], SHORT_NAMES)
        self.assertFalse(any(t.get_visible() for t in right.get_yticklabels()))
        for axis in (self.ax, right):
            self.assertEqual(
                [t.get_text() for t in axis.get_xticklabels()], list(prepare.T_CELL_MARKERS)
            )

    def test_subtypes_in_another_order_are_refused(self):
        data = protein_data(self.combat, self.combat)
        data["subtypes"] = data["subtypes"][1:] + data["subtypes"][:1]
        with self.assertRaisesRegex(ValueError, "must be the subtypes"):
            draw.t_cell_proteins(self.ax, data)

    def test_a_protein_that_does_not_vary_between_subtypes_is_refused(self):
        flat = [row[:4] + [2.0] + row[5:] for row in self.combat]
        with self.assertRaisesRegex(ValueError, "varying between them"):
            draw.t_cell_proteins(self.ax, protein_data(self.combat, flat))


class ComparisonRowsTest(unittest.TestCase):
    def setUp(self):
        fs.apply()
        self.fig = plt.figure(figsize=(4, 2))
        self.addCleanup(plt.close, self.fig)
        self.ax = self.fig.add_axes([0.3, 0.2, 0.6, 0.6])

    def data(self, names: list[str]) -> dict:
        # A different share per row, so that each row's markers can be told apart.
        return {
            "rows": [
                {"subtype": n, "wnn": (i + 1) / 10, "rna": (i + 1) / 20}
                for i, n in enumerate(names)
            ],
            "overall": {"haoCells": 100},
        }

    def test_rows_in_another_order_are_refused(self):
        moved = SHORT_NAMES[1:] + SHORT_NAMES[:1]
        with self.assertRaisesRegex(ValueError, "must be the subtypes"):
            draw.t_cell_reference_comparison(self.ax, self.data(moved))

    def test_each_subtypes_markers_sit_on_its_named_row(self):
        draw.t_cell_reference_comparison(self.ax, self.data(SHORT_NAMES))
        self.fig.canvas.draw()
        named = {tick.get_text(): tick.get_position()[1] for tick in self.ax.get_yticklabels()}
        # The first subtype is the top row.
        self.assertEqual(max(named, key=named.get), SHORT_NAMES[0])
        # Each filled WNN marker, told apart by its share, sits at the height of its subtype's name.
        filled = {
            round(line.get_xdata()[0]): line.get_ydata()[0]
            for line in self.ax.lines
            if line.get_linestyle() == "None" and line.get_markerfacecolor() != fs.WHITE
        }
        self.assertEqual(filled, {10 * (i + 1): named[n] for i, n in enumerate(SHORT_NAMES)})


class TCellLayoutTest(unittest.TestCase):
    def test_the_layout_places_every_panel_once_inside_the_page(self):
        draw.check_layout()

    def test_the_maps_keep_o_and_the_t_cell_panels_follow(self):
        letters = draw.letters()
        self.assertEqual(
            [letters[name] for name in ("cluster_agreement", "graph_umaps", *T_CELL_PANELS)],
            ["n", "o", "p", "q", "r"],
        )

    def test_k_to_n_end_level_with_the_maps(self):
        maps_bottom = draw.LAYOUT["graph_umaps"][1]
        self.assertAlmostEqual(min(draw.LAYOUT[n][1] for n in draw.CITE_COLUMN), maps_bottom)

    def test_the_comparison_continues_the_cite_seq_column(self):
        cite = absolute_plot("multimodal_time")
        comparison = absolute_plot("t_cell_reference_comparison")
        self.assertAlmostEqual(comparison[0], cite[0])
        self.assertAlmostEqual(comparison[2], cite[2])

    def test_the_t_cell_map_continues_the_map_column_at_the_size_of_its_maps(self):
        graph_left, graph_bottom = draw.LAYOUT["graph_umaps"]
        x, _, side, _ = draw.CELLS["graph_umaps"].plot
        t_map = absolute_plot("t_cell_map")
        self.assertAlmostEqual(t_map[0], graph_left + x)
        self.assertAlmostEqual(t_map[2] - t_map[0], side)
        self.assertAlmostEqual(t_map[3] - t_map[1], side)
        self.assertLess(t_map[3], graph_bottom)
        # The map's top is level with the comparison's plot.
        self.assertAlmostEqual(t_map[3], absolute_plot("t_cell_reference_comparison")[3])

    def test_the_heatmaps_start_under_the_comparison_and_under_the_map(self):
        proteins = absolute_plot("t_cell_proteins")
        self.assertAlmostEqual(proteins[0], absolute_plot("t_cell_reference_comparison")[0])
        t_map = absolute_plot("t_cell_map")
        self.assertAlmostEqual(proteins[0] + draw.T_CELL_HEATMAP_OFFSET_MM, t_map[0])
        self.assertAlmostEqual(proteins[2] - proteins[0], t_map[2] - t_map[0])

    def test_the_t_cell_panels_tile_the_block_under_k_to_o(self):
        boxes = []
        for name in T_CELL_PANELS:
            left, bottom = draw.LAYOUT[name]
            cell = draw.CELLS[name]
            boxes.append((left, bottom, left + cell.width, bottom + cell.height))
        top = draw.LAYOUT["graph_umaps"][1]
        self.assertAlmostEqual(min(b[0] for b in boxes), draw.CITE_LEFT)
        self.assertAlmostEqual(max(b[2] for b in boxes), 160.0)
        self.assertAlmostEqual(min(b[1] for b in boxes), 0.0)
        self.assertAlmostEqual(max(b[3] for b in boxes), top)
        # check_layout refuses overlaps, so cells that cover the block's whole area leave no gap.
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        self.assertAlmostEqual(area, (160.0 - draw.CITE_LEFT) * top)


if __name__ == "__main__":
    unittest.main()
