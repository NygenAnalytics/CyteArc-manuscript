import unittest

import numpy as np
from manuscript.figures.analysis_scaling import draw
from matplotlib.colors import to_rgb


class CielabTest(unittest.TestCase):
    def test_reference_colours(self):
        # White, black and mid grey sit on the neutral axis; sRGB red is the published (53.24, 80.09, 67.20).
        reference = {
            (1.0, 1.0, 1.0): (100.0, 0.0, 0.0),
            (0.0, 0.0, 0.0): (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5): (53.39, 0.0, 0.0),
            (1.0, 0.0, 0.0): (53.24, 80.09, 67.20),
        }
        lab = draw.cielab(np.array(list(reference)))
        np.testing.assert_allclose(lab, np.array(list(reference.values())), atol=0.3)


class CategoricalPaletteTest(unittest.TestCase):
    def test_the_seed_colours_come_first_and_the_added_colours_stay_in_the_band_and_apart(self):
        palette = draw.categorical_palette(50)
        seeds = np.array([to_rgb(c) for c in draw.UMAP_SEED_COLOURS])
        self.assertEqual(palette.shape, (50, 3))
        np.testing.assert_allclose(palette[: len(seeds)], seeds)
        added = draw.cielab(palette[len(seeds) :])
        chroma = np.hypot(added[:, 1], added[:, 2])
        self.assertTrue(np.all((added[:, 0] >= 35) & (added[:, 0] <= 78)))
        self.assertTrue(np.all((chroma >= 28) & (chroma <= 60)))
        lab = draw.cielab(palette)
        distances = np.linalg.norm(lab[:, None] - lab[None], axis=2)
        np.fill_diagonal(distances, np.inf)
        # No added colour is closer to another than the two closest seed colours are to each other.
        seed_distances = distances[: len(seeds), : len(seeds)]
        self.assertGreaterEqual(distances[len(seeds) :].min(), seed_distances.min())

    def test_fewer_colours_than_seeds_are_the_first_seeds(self):
        np.testing.assert_allclose(
            draw.categorical_palette(3), [to_rgb(c) for c in draw.UMAP_SEED_COLOURS[:3]]
        )


class CellLineColoursTest(unittest.TestCase):
    def raster(self):
        # Line 0 fills columns 0 to 9 and line 1 columns 10 to 15 of a 50-row raster; their bodies meet along
        # columns 9 and 10, the only dense pixels. Line 2 holds two pixels far from both.
        dominant = np.full((50, 20), -1)
        counts = np.zeros((50, 20), dtype=int)
        dominant[:, :10], counts[:, :10] = 0, 5
        dominant[:, 10:16], counts[:, 10:16] = 1, 5
        counts[:, 9] = counts[:, 10] = 50
        dominant[:2, 19], counts[:2, 19] = 2, 5
        return dominant, counts

    def test_a_bordering_line_takes_the_free_colour_farthest_from_its_neighbour(self):
        dominant, counts = self.raster()
        colours = draw.cell_line_colours(dominant, counts, 3)
        palette = draw.categorical_palette(3)
        lab = draw.cielab(palette)
        # Line 0, the largest, takes the first colour. Line 1 borders it and takes whichever of the other two
        # colours lies farther from the first; line 2 borders nothing and takes the one left.
        farther = 1 if np.linalg.norm(lab[1] - lab[0]) > np.linalg.norm(lab[2] - lab[0]) else 2
        np.testing.assert_allclose(colours[0], palette[0])
        np.testing.assert_allclose(colours[1], palette[farther])
        np.testing.assert_allclose(colours[2], palette[3 - farther])

    def test_a_raster_naming_more_lines_than_the_run_is_an_error(self):
        dominant, counts = self.raster()
        with self.assertRaisesRegex(ValueError, "names cell line 2, beyond the 2 lines"):
            draw.cell_line_colours(dominant, counts, 2)


class StatusGridOutcomeTest(unittest.TestCase):
    DATA = {
        "arms": {"bpcells_cached": {"embryo_1000000": {"state": "timeout"}}},
        "tahoe": {"outcomes": {"bpcells_cached": {"state": "timeout"}}},
    }

    def kind(self, arm: str, dataset: str) -> str:
        return draw.cell_outcome(arm, dataset, draw.outcome_entry(self.DATA, arm, dataset))

    def test_a_timeout_on_tahoe_draws_as_out_of_time(self):
        self.assertEqual(self.kind("bpcells_cached", "tahoe"), "time")

    def test_a_memory_kill_on_tahoe_draws_as_out_of_ram(self):
        data = {"tahoe": {"outcomes": {"bpcells_cached": {"state": "oom"}}}}
        entry = draw.outcome_entry(data, "bpcells_cached", "tahoe")
        self.assertEqual(draw.cell_outcome("bpcells_cached", "tahoe", entry), "oom")

    def test_a_configuration_without_a_tahoe_run_draws_as_not_run(self):
        self.assertEqual(self.kind("scanpy", "tahoe"), "none")

    def test_cytearc_passes_on_tahoe_without_a_comparator_record(self):
        self.assertEqual(self.kind("cytearc", "tahoe"), "pass")

    def test_benchmark_datasets_still_read_their_own_arms(self):
        self.assertEqual(self.kind("bpcells_cached", "embryo_1000000"), "time")
        self.assertEqual(draw.cell_outcome("scanpy", "embryo_1000000", {"state": "oom"}), "oom")
        self.assertEqual(draw.cell_outcome("scanpy", "embryo_1000000", {}), "pass")
        self.assertEqual(draw.cell_outcome("seurat", "embryo_1000000", {"state": "crash"}), "err")
