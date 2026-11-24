import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.reuse import blood_progenitors as bp


class CandidateCells(unittest.TestCase):
    def test_keeps_labelled_progenitors_and_cells_with_any_marker(self):
        labels = np.array(
            ["T cell", "hematopoietic precursor cell", "T cell", "precursor B cell", "monocyte"]
        )
        values = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [np.nan, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(
            bp.candidate_cells(labels, values).tolist(), [False, True, True, False, False]
        )

    def test_a_gene_the_store_lacks_counts_as_absent(self):
        values = np.array([[np.nan, np.nan, np.nan, 1.0]], dtype=np.float32)
        self.assertTrue(bp.candidate_cells(np.array(["B cell"]), values)[0])


class DonorSampleTotals(unittest.TestCase):
    def test_counts_cells_and_labelled_progenitors_per_donor_and_sample(self):
        frame = pd.DataFrame(
            {
                "donor_id": pd.Categorical(["a", "a", "a", "b"]),
                "sample": pd.Categorical(["s1", "s1", "s2", ""]),
                "sampleTime0": [3.0, 3.0, 9.0, np.inf],
                "sampleTime1": [np.inf] * 4,
                "labelled": [True, False, True, False],
                "disease": pd.Categorical(["COVID-19", "COVID-19", "COVID-19", "normal"]),
                "development_stage": pd.Categorical(
                    ["40-year-old stage"] * 3 + ["50-year-old stage"]
                ),
                "sex": pd.Categorical(["female"] * 3 + ["male"]),
                "cell_type": pd.Categorical(["x"] * 4),
            }
        )
        totals = bp.donor_sample_totals(frame).set_index(["donor_id", "sample"])
        self.assertEqual(totals.loc[("a", "s1"), "cells"], 2)
        self.assertEqual(totals.loc[("a", "s1"), "labelledProgenitors"], 1)
        self.assertEqual(totals.loc[("a", "s2"), "cells"], 1)
        self.assertEqual(totals.loc[("a", "s2"), "sampleTime0"], 9.0)
        self.assertEqual(totals.loc[("b", ""), "disease"], "normal")
        self.assertEqual(int(totals["cells"].sum()), 4)


class PeripheralBloodRows(unittest.TestCase):
    def test_keeps_primary_cells_from_blood_and_venous_blood_only(self):
        primary = np.array([True, True, False, True, True])
        tissue = np.array(["blood", "venous blood", "blood", "bone marrow", "umbilical cord blood"])
        self.assertEqual(bp.peripheral_blood_rows(primary, tissue).tolist(), [0, 1])


class SampleValues(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "donor_id": pd.Categorical(["a", "a", "a", "b", "b", "c"]),
                "sample": pd.Categorical(["s1", "s1", "s2", "s3", "s3", "s4"]),
                "severity": pd.Categorical(
                    ["Mild", "Mild", "Severe", "Healthy", "Healthy", "Critical"]
                ),
                "worst": pd.Categorical(
                    ["Severe", "Severe", "Severe", "Healthy", "nan", "Critical"]
                ),
                # A category no cell holds must not appear as a row of zero cells.
                "unused": pd.Categorical(["x"] * 6, categories=["x", "y"]),
            }
        )

    def test_counts_cells_per_donor_sample_and_value_of_each_column(self):
        table = bp.sample_values(self.frame(), ["severity", "worst"])
        rows = {
            (r.donor_id, r.sample, r.column, r.value): r.cells
            for r in table.itertuples(index=False)
        }
        self.assertEqual(
            rows,
            {
                ("a", "s1", "severity", "Mild"): 2,
                ("a", "s2", "severity", "Severe"): 1,
                ("b", "s3", "severity", "Healthy"): 2,
                ("c", "s4", "severity", "Critical"): 1,
                ("a", "s1", "worst", "Severe"): 2,
                ("a", "s2", "worst", "Severe"): 1,
                ("b", "s3", "worst", "Healthy"): 1,
                ("b", "s3", "worst", "nan"): 1,
                ("c", "s4", "worst", "Critical"): 1,
            },
        )
        # Each column's rows account for every cell once.
        self.assertEqual(
            table.groupby("column")["cells"].sum().to_dict(), {"severity": 6, "worst": 6}
        )

    def test_categories_without_cells_give_no_rows(self):
        frame = self.frame()
        frame["severity"] = frame["severity"].cat.add_categories(["Moderate"])
        table = bp.sample_values(frame, ["severity"])
        self.assertNotIn("Moderate", set(table["value"]))
        self.assertEqual(len(table), 4)

    def test_values_are_text_for_the_record(self):
        table = bp.sample_values(self.frame(), ["severity"])
        for key in ("donor_id", "sample", "value"):
            self.assertTrue(all(isinstance(value, str) for value in table[key]))

    def test_no_columns_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "at least one column"):
            bp.sample_values(self.frame(), [])


class ClusterPanelMeans(unittest.TestCase):
    def test_means_and_shares_per_cluster(self):
        values = np.array([[0.0, 1.0], [2.0, 0.0], [4.0, 3.0]])
        table = bp.cluster_panel_means(values, np.array([0, 0, 1]), ["CD34", "MPO"]).set_index(
            "cluster"
        )
        self.assertEqual(table.loc[0, "cells"], 2)
        self.assertAlmostEqual(table.loc[0, "CD34_mean"], 1.0)
        self.assertAlmostEqual(table.loc[0, "MPO_mean"], 0.5)
        self.assertAlmostEqual(table.loc[0, "CD34_share"], 0.5)
        self.assertAlmostEqual(table.loc[1, "MPO_share"], 1.0)


def write_extracted(path: Path, ids: list[str]) -> None:
    """A small H5AD as CyteArc's export writes it, with these feature IDs."""
    import anndata as ad
    from scipy import sparse

    matrix = sparse.csr_matrix(np.arange(2 * len(ids), dtype=np.float32).reshape(2, len(ids)))
    ad.AnnData(
        matrix, obs=pd.DataFrame(index=["c0", "c1"]), var=pd.DataFrame(index=ids)
    ).write_h5ad(path)


class BareFeatureIds(unittest.TestCase):
    def test_prefixed_and_versioned_ids_become_bare_and_the_file_still_reads(self):
        import anndata as ad

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "a.h5ad"
            write_extracted(
                path, ["GRCh38___ENSG00000174059", "ENSG00000128040.12", "ENSG00000007062"]
            )
            self.assertEqual(bp.bare_feature_ids(path), 2)
            back = ad.read_h5ad(path)
            self.assertEqual(
                list(back.var_names), ["ENSG00000174059", "ENSG00000128040", "ENSG00000007062"]
            )
            self.assertEqual(back.X[1, 2], 5.0)

    def test_bare_ids_are_left_as_they_are(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "a.h5ad"
            write_extracted(path, ["ENSG00000174059", "ENSG00000128040"])
            self.assertEqual(bp.bare_feature_ids(path), 0)

    def test_a_feature_without_a_human_ensembl_id_keeps_its_own(self):
        import anndata as ad

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "a.h5ad"
            write_extracted(path, ["ENSSASG00005000007", "ENSG00000128040.12"])
            self.assertEqual(bp.bare_feature_ids(path), 1)
            self.assertEqual(
                list(ad.read_h5ad(path).var_names), ["ENSSASG00005000007", "ENSG00000128040"]
            )

    def test_two_features_with_one_ensembl_id_are_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "a.h5ad"
            write_extracted(path, ["ENSG00000174059.1", "GRCh38___ENSG00000174059"])
            with self.assertRaisesRegex(ValueError, "share an Ensembl ID"):
                bp.bare_feature_ids(path)


if __name__ == "__main__":
    unittest.main()
