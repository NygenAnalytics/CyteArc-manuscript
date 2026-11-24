import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from experiments.integration.t_cell_subtypes import (
    cell_clr,
    cluster_summaries,
    feature_coverage,
    named_subtypes,
    prefixed_feature_column,
    protein_values,
    select,
    t_cell_rows,
)

DESIGN = {
    "labelColumn": "cell_type",
    "labels": ("CD4 Naive", "CD8 Naive"),
    "studyColumn": "study",
    "donorColumn": "donor",
    "authorColumn": "author",
}


def store(columns: dict) -> SimpleNamespace:
    return SimpleNamespace(cells=SimpleNamespace(fetch_all=lambda column: columns[column]))


class TCellRowsTest(unittest.TestCase):
    """The T cells a reference or query holds: eligible cells with a T-cell label, of one study when asked."""

    # 120 cells: labels cycle through two T-cell labels and a monocyte label, studies alternate, every fifth cell is
    # not eligible.
    COLUMNS = {
        "cell_type": np.array(["CD4 Naive", "CD8 Naive", "CD14 Mono"] * 40, dtype=object),
        "study": np.array(["combat", "stephenson"] * 60, dtype=object),
        "I": np.arange(120) % 5 != 0,
    }

    def test_eligible_t_cells_of_every_study_or_of_one(self):
        rows = t_cell_rows(store(self.COLUMNS), DESIGN, None)
        # by hand: a row is kept when it is not a multiple of 5 and not a multiple of 3 plus 2
        self.assertEqual(rows.tolist(), [r for r in range(120) if r % 5 and r % 3 != 2])
        rows = t_cell_rows(store(self.COLUMNS), DESIGN, "stephenson")
        self.assertEqual(rows.tolist(), [r for r in range(120) if r % 5 and r % 3 != 2 and r % 2])

    def test_a_label_no_cell_carries_is_refused(self):
        design = {**DESIGN, "labels": ("CD4 Naive", "MAIT")}
        with self.assertRaisesRegex(ValueError, r"no cells labelled \['MAIT'\]"):
            t_cell_rows(store(self.COLUMNS), design, None)

    def test_an_unknown_study_or_a_store_without_studies_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no cells of study 'hao'"):
            t_cell_rows(store(self.COLUMNS), DESIGN, "hao")
        with self.assertRaisesRegex(ValueError, "no study column"):
            t_cell_rows(store(self.COLUMNS), {**DESIGN, "studyColumn": None}, "combat")

    def test_too_few_t_cells_for_a_neighbor_graph_are_refused(self):
        columns = {**self.COLUMNS, "I": np.arange(120) < 30}
        # 30 eligible cells hold 20 T cells, no more than the 20 neighbors each needs
        with self.assertRaisesRegex(ValueError, "20 T cells are too few"):
            t_cell_rows(store(columns), DESIGN, None)


class SelectTest(unittest.TestCase):
    def test_the_saved_column_marks_exactly_the_given_rows(self):
        saved = {}
        ds = SimpleNamespace(
            cells=SimpleNamespace(
                N=6, insert=lambda name, values, overwrite: saved.update({name: values})
            ),
            snapshot_cell_selection=lambda name: ("selection", name),
        )
        self.assertEqual(select(ds, np.array([1, 4]), "t_cells"), ("selection", "t_cells"))
        self.assertEqual(saved["t_cells"].tolist(), [False, True, False, False, True, False])


class FeatureCoverageTest(unittest.TestCase):
    def test_the_share_of_chosen_reference_genes_the_query_measures(self):
        reference = SimpleNamespace(
            artifacts=SimpleNamespace(load=lambda ref: {"values": np.array([1, 0, 1, 1, 0, 1])}),
            RNA=SimpleNamespace(
                feats=SimpleNamespace(
                    fetch_all=lambda column: np.array(["A", "B", "C", "D", "E", "F"])
                )
            ),
        )
        query = SimpleNamespace(
            RNA=SimpleNamespace(
                feats=SimpleNamespace(fetch_all=lambda column: np.array(["F", "B", "A", "Z"]))
            )
        )
        # chosen genes A, C, D and F, of which the query measures A and F
        self.assertEqual(feature_coverage(reference, "features", query), 0.5)


class ClusterSummariesTest(unittest.TestCase):
    def test_each_clusters_protein_author_labels_and_donors(self):
        groups = np.array([2, 1, 2, 2, 1])
        clr = np.array([[1.0, 0.0], [5.0, 2.0], [3.0, 4.0], [8.0, 6.0], [7.0, 2.0]])
        authors = np.array(["CD4.TCM", "CD8.TEM", "CD4.TCM", "CD4.NAIVE", "CD8.TEM"])
        donors = np.array(["d1", "d1", "d1", "d2", "d2"])
        proteins, labels, spread = cluster_summaries(
            groups,
            clr,
            np.array(["CD45RA", "CD45RO"]),
            authors,
            donors,
            graph="wnn",
            resolution=0.5,
        )
        # cluster 1 holds rows 1 and 4, cluster 2 rows 0, 2 and 3
        self.assertEqual(
            [
                (p["cluster"], p["protein"], p["cells"], p["medianClr"], p["meanClr"])
                for p in proteins
            ],
            [
                ("1", "CD45RA", 2, 6.0, 6.0),
                ("1", "CD45RO", 2, 2.0, 2.0),
                ("2", "CD45RA", 3, 3.0, 4.0),
                ("2", "CD45RO", 3, 4.0, 10.0 / 3),
            ],
        )
        self.assertEqual(
            [(r["cluster"], r["authorLabel"], r["cells"]) for r in labels],
            [("1", "CD8.TEM", 2), ("2", "CD4.NAIVE", 1), ("2", "CD4.TCM", 2)],
        )
        self.assertEqual(
            [(r["cluster"], r["donors"], r["largestDonorShare"]) for r in spread],
            [("1", 2, 0.5), ("2", 2, 2 / 3)],
        )
        self.assertTrue(
            all(r["graph"] == "wnn" and r["resolution"] == 0.5 for r in proteins + labels + spread)
        )


class CellClrTest(unittest.TestCase):
    """The per-cell CLR the T-cell runs normalize protein with, against values derived by hand."""

    def test_each_cell_is_centered_over_its_own_proteins(self):
        values = cell_clr(None, np.array([[0.0, 1.0, 3.0], [2.0, 2.0, 2.0]]))
        # row one: the mean log1p is (log 2 + log 4) / 3 = log 2, so the counts are halved; row two: each count is
        # divided by 3, its geometric mean of 1 + count
        np.testing.assert_allclose(
            values, np.log([[1.0, 1.5, 2.5], [5 / 3, 5 / 3, 5 / 3]]), rtol=0, atol=1e-12
        )

    def test_an_unknown_normalization_is_refused_before_reading_the_store(self):
        with self.assertRaisesRegex(ValueError, "adtNormalization must be one of"):
            protein_values(None, np.arange(3), "margin2")


class NamedSubtypesTest(unittest.TestCase):
    """The subtype each reference cell carries into the mapping: its cluster's name, blank for unnamed clusters."""

    def test_named_clusters_name_their_cells_and_the_rest_stay_blank(self):
        partition = np.array([3, 1, 2, 3, 2])
        named = named_subtypes(partition, {"3": "CD4 naive", "1": "MAIT"})
        self.assertEqual(named.tolist(), ["CD4 naive", "MAIT", "", "CD4 naive", ""])

    def test_a_name_for_a_missing_cluster_or_a_blank_name_is_refused(self):
        partition = np.array([1, 2, 2])
        with self.assertRaisesRegex(ValueError, r"clusters \['7'\]"):
            named_subtypes(partition, {"1": "Treg", "7": "MAIT"})
        with self.assertRaisesRegex(ValueError, "not blank"):
            named_subtypes(partition, {"1": " "})


class SummariesWithoutProteinTest(unittest.TestCase):
    def test_a_query_without_protein_still_gets_author_labels_and_donors(self):
        proteins, labels, spread = cluster_summaries(
            np.array(["Treg", "MAIT", "Treg"]),
            None,
            None,
            np.array(["regulatory T cell", "MAIT", "naive T cell"]),
            np.array(["d1", "d2", "d2"]),
            cells="lee",
        )
        self.assertEqual(proteins, [])
        self.assertEqual(
            [(r["cluster"], r["authorLabel"], r["cells"]) for r in labels],
            [("MAIT", "MAIT", 1), ("Treg", "naive T cell", 1), ("Treg", "regulatory T cell", 1)],
        )
        self.assertEqual(
            [(r["cluster"], r["donors"], r["largestDonorShare"]) for r in spread],
            [("MAIT", 1, 1.0), ("Treg", 2, 0.5)],
        )


def text(values) -> list[str]:
    """The reader gives IDs as the file holds them, bytes from an HDF5 string array."""
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


class PrefixedFeatureColumnTest(unittest.TestCase):
    """A query H5AD keyed by RNA:<symbol>, as the CITE-seq reference stores key their genes."""

    def write(self, path):
        import h5py
        from experiments.runtime.h5ad import _dataframe, _string_dataset

        with h5py.File(path, "w") as handle:
            handle.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
            matrix = handle.create_group("X")
            matrix.attrs.update(
                {"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": (2, 2)}
            )
            matrix["data"] = np.array([1.0, 2.0], dtype=np.float32)
            matrix["indices"] = np.array([0, 1])
            matrix["indptr"] = np.array([0, 1, 2])
            obs = _dataframe(handle, "obs", ())
            _string_dataset(obs, "_index", np.array(["c1", "c2"]))
            var = _dataframe(handle, "var", ("feature_name",))
            _string_dataset(var, "_index", np.array(["ENSG00000198851", "ENSG00000010610"]))
            _string_dataset(var, "feature_name", np.array(["CD3E", "CD4"]))

    def test_the_reader_keys_genes_by_the_prefixed_column(self):
        import h5py
        from cytearc.readers import H5adReader
        from experiments.runtime.h5ad import read_column

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "query.h5ad"
            self.write(path)
            column = prefixed_feature_column(path, "feature_name", "RNA:")
            with h5py.File(path, "r") as handle:
                self.assertEqual(
                    read_column(handle["var"], column).tolist(), ["RNA:CD3E", "RNA:CD4"]
                )
                order = [
                    c.decode() if isinstance(c, bytes) else c
                    for c in handle["var"].attrs["column-order"]
                ]
                self.assertEqual(order, ["feature_name", column])
            reader = H5adReader(str(path), feature_ids_key=column, feature_name_key="feature_name")
            try:
                self.assertEqual(text(reader.feat_ids()), ["RNA:CD3E", "RNA:CD4"])
                self.assertEqual(text(reader.feat_names()), ["CD3E", "CD4"])
            finally:
                reader.close()
            with self.assertRaisesRegex(ValueError, "already holds"):
                prefixed_feature_column(path, "feature_name", "RNA:")


if __name__ == "__main__":
    unittest.main()
