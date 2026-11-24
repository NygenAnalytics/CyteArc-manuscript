import tempfile
import unittest
from pathlib import Path

import numpy as np
from experiments.integration.multimodal_agreement import (
    SMALL_COMPONENT_CELLS,
    ResolutionSearch,
    load_inputs,
    small_component_assignments,
)
from scipy.sparse import csr_matrix

CLIQUE_A = range(0, 12)
CLIQUE_B = range(12, 24)
PAIR = range(26, 28)
NINE_CELLS = range(28, 37)
CELLS = 37


def cliques(groups, cells=CELLS, self_loops=()) -> csr_matrix:
    """A graph whose cells are joined within each group and nowhere else."""
    rows, columns = [], []
    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    rows.append(i)
                    columns.append(j)
    for cell in self_loops:
        rows.append(cell)
        columns.append(cell)
    return csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(cells, cells))


def two_cliques_and_small_components(self_loops=()) -> csr_matrix:
    """Two 12-cell cliques; cells 24 and 25 alone, a pair (26, 27) and a 9-cell clique (28-36)."""
    return cliques((CLIQUE_A, CLIQUE_B, PAIR, NINE_CELLS), self_loops=self_loops)


def neighbor_rows() -> np.ndarray:
    return np.array(
        [[1, 2, 3]] * 12
        + [[13, 14, 15]] * 12
        # Cell 24's first neighbour is the lone cell 25, and cell 25's first is itself.
        + [[25, 2, 13], [25, 14, 1]]
        # Each cell of the pair lists its partner first.
        + [[27, 3, 14], [26, 15, 4]]
        + [[29, 30, 5]] * 9
    )


class SmallComponentsTest(unittest.TestCase):
    def test_small_components_take_their_nearest_neighbour_outside(self):
        detached, nearest = small_component_assignments(
            two_cliques_and_small_components(), neighbor_rows()
        )
        self.assertEqual(detached.tolist(), list(range(24, 37)))
        self.assertEqual(nearest.tolist(), [2, 14, 3, 15] + [5] * 9)

    def test_the_size_limit_keeps_components_of_ten_cells(self):
        self.assertEqual(SMALL_COMPONENT_CELLS, 10)
        graph = cliques((range(0, 12), range(12, 22), range(22, 31)), cells=31)
        rows = np.array([[1, 2, 3]] * 31)
        detached, _ = small_component_assignments(graph, rows)
        self.assertEqual(detached.tolist(), list(range(22, 31)))

    def test_a_self_loop_does_not_connect_a_cell(self):
        # Seurat's shared-neighbour graphs store each cell's similarity to itself.
        detached, _ = small_component_assignments(
            two_cliques_and_small_components(self_loops=(24, 25)), neighbor_rows()
        )
        self.assertEqual(detached.tolist(), list(range(24, 37)))

    def test_an_incoming_edge_alone_connects_a_cell(self):
        graph = two_cliques_and_small_components().tolil()
        graph[0, 24] = 1.0
        detached, nearest = small_component_assignments(graph.tocsr(), neighbor_rows())
        self.assertEqual(detached.tolist(), list(range(25, 37)))
        self.assertEqual(nearest.tolist(), [14, 3, 15] + [5] * 9)

    def test_a_cell_without_a_neighbour_in_a_larger_component_fails(self):
        rows = neighbor_rows()
        rows[24] = [25, 26, 24]
        with self.assertRaisesRegex(ValueError, "cell 24"):
            small_component_assignments(two_cliques_and_small_components(), rows)


class ResolutionSearchTest(unittest.TestCase):
    def test_small_components_otherwise_form_clusters_of_their_own(self):
        partition = ResolutionSearch(two_cliques_and_small_components(), 1)(1.0)
        self.assertEqual(len(np.unique(partition)), 6)

    def test_small_components_join_their_neighbours_clusters(self):
        search = ResolutionSearch(two_cliques_and_small_components(), 1, neighbor_rows())
        partition = search(1.0)
        self.assertEqual(sorted(np.unique(partition).tolist()), [0, 1])
        self.assertEqual(len(set(partition[list(CLIQUE_A)])), 1)
        self.assertEqual(len(set(partition[list(CLIQUE_B)])), 1)
        self.assertNotEqual(partition[0], partition[12])
        a, b = partition[0], partition[12]
        self.assertEqual(partition[24:28].tolist(), [a, b, a, b])
        self.assertEqual(set(partition[list(NINE_CELLS)]), {a})

    def test_the_matched_cluster_number_becomes_reachable(self):
        search = ResolutionSearch(two_cliques_and_small_components(), 1, neighbor_rows())
        self.assertEqual(search.count(search.match(2, steps=8)), 2)

    def test_the_search_lowers_the_resolution_below_its_starting_points(self):
        # Ten 6-cell cliques joined in a ring by weak edges stay ten clusters from
        # resolution 0.02 to 2.0 and merge only at lower resolutions.
        rows, columns, weights = [], [], []
        for clique in range(10):
            cells = range(clique * 6, clique * 6 + 6)
            for i in cells:
                for j in cells:
                    if i != j:
                        rows.append(i)
                        columns.append(j)
                        weights.append(1.0)
            first, next_clique = clique * 6, (clique + 1) % 10 * 6 + 1
            rows += [first, next_clique]
            columns += [next_clique, first]
            weights += [0.05, 0.05]
        ring = csr_matrix((weights, (rows, columns)), shape=(60, 60))
        self.assertEqual(ResolutionSearch(ring, 1).count(0.5), 10)
        for target in (2, 3):
            search = ResolutionSearch(ring, 1)
            self.assertEqual(search.count(search.match(target, steps=16)), target)

    def test_a_connected_graph_is_partitioned_unchanged(self):
        rng = np.random.default_rng(0)
        columns = rng.integers(0, 200, size=(200, 8))
        rows = np.repeat(np.arange(200), 8)
        graph = csr_matrix((rng.random(1600), (rows, columns.ravel())), shape=(200, 200))
        graph = graph + graph.T
        plain = ResolutionSearch(graph, 1)
        with_rows = ResolutionSearch(graph, 1, columns)
        self.assertEqual(len(with_rows.detached), 0)
        for resolution in (0.5, 1.0, 2.0):
            np.testing.assert_array_equal(plain(resolution), with_rows(resolution))


if __name__ == "__main__":
    unittest.main()


class LoadInputsTest(unittest.TestCase):
    """The graphs a run saved, as the agreement evaluation reads them: the SNN fusion graph when the run built one."""

    def saved_graphs(self, directory: Path, *, snn: bool) -> Path:
        # Four cells with two neighbours each; every graph links each cell to the next two, with weights 1 and 0.5.
        columns = np.array([[1, 2], [2, 3], [3, 0], [0, 1]])
        weights = np.tile(np.array([1.0, 0.5], dtype=np.float32), (4, 1))
        arrays = {
            "cells": np.arange(4),
            "labels": np.array(["T", "T", "B", "B"]),
            "groups": np.array(["one", "two", "one", "two"]),
        }
        names = ["RNA", "ADT", "wnn"] + (["snn"] if snn else [])
        for name in names:
            arrays[f"{name}_neighbors"] = columns
            arrays[f"{name}_clusters"] = np.array([0, 0, 1, 1])
            arrays[
                f"{name}_connectivity_weights" if name in ("RNA", "ADT") else f"{name}_weights"
            ] = weights
        path = directory / "saved.npz"
        np.savez(path, **arrays)
        return path

    def load(self, *, snn: bool):
        with tempfile.TemporaryDirectory() as folder:
            path = self.saved_graphs(Path(folder), snn=snn)
            work = Path(folder) / "work"
            work.mkdir()
            return load_inputs({"graphsUri": str(path)}, work)

    def test_a_run_with_an_snn_graph_gives_four_graphs(self):
        neighbors, graphs, partitions, labels, _ = self.load(snn=True)
        self.assertEqual(sorted(graphs), ["adt", "cytearc_snn", "cytearc_wnn", "rna"])
        self.assertEqual(sorted(neighbors), sorted(graphs))
        self.assertIn(("cytearc_snn", 1.0), partitions)
        self.assertEqual(labels.tolist(), ["T", "T", "B", "B"])

    def test_a_run_without_one_gives_the_other_three(self):
        neighbors, graphs, partitions, _, _ = self.load(snn=False)
        self.assertEqual(sorted(graphs), ["adt", "cytearc_wnn", "rna"])
        self.assertEqual(sorted(neighbors), sorted(graphs))
        self.assertEqual(sorted(method for method, _ in partitions), ["adt", "cytearc_wnn", "rna"])

    def test_each_graph_holds_the_saved_edges_and_weights(self):
        _, graphs, _, _, _ = self.load(snn=False)
        # Cell 0's neighbours are cells 1 and 2, with weights 1 and 0.5, and it has no edge to cell 3.
        row = graphs["cytearc_wnn"].getrow(0).toarray().ravel()
        self.assertEqual(row.tolist(), [0.0, 1.0, 0.5, 0.0])
