import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from experiments.integration.multimodal_neighbors import select_cells, write_count_inputs
from scipy.io import mmread


class ComputedSlice:
    def __init__(self, values: np.ndarray):
        self.values = values

    def __getitem__(self, key):
        return ComputedSlice(self.values[key])

    def compute(self) -> np.ndarray:
        return self.values


def datastore(counts: np.ndarray) -> SimpleNamespace:
    names = np.array([f"gene{j}" for j in range(counts.shape[1])])
    feats = SimpleNamespace(fetch_all=lambda column: names)
    return SimpleNamespace(RNA=SimpleNamespace(rawData=ComputedSlice(counts), feats=feats))


class WriteCountInputsTest(unittest.TestCase):
    def write(self, counts: np.ndarray, cells: np.ndarray, features: np.ndarray):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            entries = write_count_inputs(datastore(counts), cells, "RNA", features, destination)
            with gzip.open(destination / "RNA_counts.mtx.gz", "rb") as stream:
                header = stream.readline().decode()
                stream.seek(0)
                matrix = mmread(stream)
            names = (destination / "RNA_features.txt").read_text().split()
        return entries, header, matrix, names

    def test_uint32_counts_are_written_as_signed_integers(self):
        # Values above the uint16 range keep the uint32 dtype that SciPy would label unsigned-integer.
        counts = np.zeros((5, 4), dtype=np.uint32)
        counts[0, 1] = 70000
        counts[3, 2] = 4
        counts[4, 0] = 1
        entries, header, matrix, _ = self.write(counts, np.arange(5), np.arange(4))
        self.assertEqual(header.split()[3], "integer")
        self.assertEqual(entries, 3)
        np.testing.assert_array_equal(matrix.toarray(), counts.T.astype(np.int64))

    def test_selection_is_features_by_cells_in_given_order(self):
        counts = np.arange(1, 25, dtype=np.uint32).reshape(6, 4)
        cells = np.array([1, 4, 5])
        features = np.array([0, 3])
        entries, _, matrix, names = self.write(counts, cells, features)
        expected = np.array([[5, 17, 21], [8, 20, 24]])
        np.testing.assert_array_equal(matrix.toarray(), expected)
        self.assertEqual(entries, 6)
        self.assertEqual(names, ["gene0", "gene3"])

    def test_float_counts_stay_real(self):
        counts = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
        _, header, matrix, _ = self.write(counts, np.arange(2), np.arange(2))
        self.assertEqual(header.split()[3], "real")
        np.testing.assert_array_equal(matrix.toarray(), counts.T)


if __name__ == "__main__":
    unittest.main()


class SelectCellsTest(unittest.TestCase):
    """The rows a run analyzes, which run and graph_umaps both take from select_cells: graph_umaps lays out a run's
    graphs only if it selects the run's cells again."""

    # Sixty cells, every fourth not eligible.
    ELIGIBLE = np.arange(60) % 4 != 0

    def select(self, cells):
        store = SimpleNamespace(cells=SimpleNamespace(fetch_all=lambda column: self.ELIGIBLE))
        return select_cells(store, cells)

    def test_without_a_count_every_eligible_cell_in_store_order(self):
        np.testing.assert_array_equal(self.select(None), np.flatnonzero(self.ELIGIBLE))

    def test_a_count_draws_the_same_cells_as_the_runs(self):
        # Drawn with the experiment's seed and sorted, as the scaling series drew each size; computed once and kept.
        expected = "3 6 9 11 14 21 22 25 30 33 35 37 39 42 43 47 49 53 55 57 58 59"
        self.assertEqual(self.select(22).tolist(), [int(row) for row in expected.split()])

    def test_every_eligible_cell_counted_gives_them_all(self):
        np.testing.assert_array_equal(self.select(45), np.flatnonzero(self.ELIGIBLE))

    def test_counts_outside_the_eligible_range_are_refused(self):
        for cells in (20, 46):
            with self.subTest(cells=cells), self.assertRaisesRegex(ValueError, "eligible range"):
                self.select(cells)
