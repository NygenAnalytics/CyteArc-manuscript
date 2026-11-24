import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# The experiment modules resolve their data folders from CYTEARC_DOWNLOADS and CYTEARC_RUNS when they are
# imported; these tests write only to their own temporary folders.
with mock.patch.dict(
    os.environ,
    {
        "CYTEARC_DOWNLOADS": str(Path(tempfile.gettempdir()) / "cytearc-downloads-unused"),
        "CYTEARC_RUNS": str(Path(tempfile.gettempdir()) / "cytearc-runs-unused"),
    },
):
    from experiments.methods import fate_dense, kang_doublet_score_comparison, method_inputs
    from experiments.runtime import catalogue_counts, kang_labels

ROOT = Path(__file__).resolve().parents[1]
DOUBLET_RECORD = ROOT / "experiments/methods/records/kang_doublet_score_comparison.json"


class CountStoreTests(unittest.TestCase):
    def test_the_store_is_built_from_counts_in_the_experiments_work_folder(self) -> None:
        with (
            mock.patch.object(method_inputs, "build_store", return_value=Path("built")) as build,
            mock.patch.object(method_inputs, "WORK", Path("/work")),
        ):
            store = method_inputs.count_store(
                "tenx_5K_pbmc_rnaseq", "magic", "pbmc", inputs=Path("/downloads"), workers=8
            )
        self.assertEqual(store, Path("built"))
        build.assert_called_once_with(
            "tenx_5K_pbmc_rnaseq",
            Path("/work/magic/pbmc.zarr"),
            downloads=Path("/downloads"),
            workers=8,
        )


class DoubletSettingsTests(unittest.TestCase):
    def test_the_documentation_settings_are_those_the_record_states(self) -> None:
        record = json.loads(DOUBLET_RECORD.read_text())
        settings = kang_doublet_score_comparison.DOCUMENTATION_SETTINGS
        for condition in ("ctrl", "stim"):
            stated = record["conditions"][condition]["cytearcDocsDefaultRun"]
            with self.subTest(condition=condition):
                self.assertEqual(settings["hvg"]["top_n"], stated["hvgCount"])
                self.assertEqual(settings["pca"]["dims"], stated["pcaDims"])
                self.assertEqual(settings["neighbors"]["k"], stated["neighborsK"])
                self.assertEqual(settings["leiden"]["partitions"], stated["leiden"]["partitions"])
                filtering = stated["filtering"]
                self.assertEqual(
                    {k: settings["filtering"][k] for k in ("method", "attrs", "lows", "highs")},
                    {k: filtering[k] for k in ("method", "attrs", "lows", "highs")},
                )

    def test_stimulated_geo_barcodes_with_the_11_suffix_join_their_cells(self) -> None:
        calls = pd.DataFrame(
            {"stim": ["stim", "stim", "ctrl"], "multiplets": ["doublet", "singlet", "ambs"]},
            index=["AAAC-11", "AAAG-1", "AAAC-1"],
        )
        joined, report = kang_doublet_score_comparison.join_calls(
            calls, "stim", np.array(["AAAC-1", "AAAG-1"])
        )
        self.assertEqual(joined.tolist(), ["doublet", "singlet"])
        self.assertEqual(
            (report["matchedDirectly"], report["matchedAfterRemap"], report["unmatched"]), (1, 2, 0)
        )


def pinned(folder: Path) -> dict[str, tuple[int, str]]:
    return {
        path.name: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(folder.iterdir())
    }


class FateGraphTests(unittest.TestCase):
    """The fate solver reads author labels for the cells its analysis kept, not for every cell of the store."""

    @classmethod
    def setUpClass(cls) -> None:
        import anndata as ad
        import cytearc
        import scipy.sparse as sp

        cytearc.configure_output(level="ERROR", progress=False)
        cls.root = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(7)
        # Cells along one continuum from an Alpha programme through Ductal cells to a Beta programme, so the
        # graph is connected and the Ductal cells lie between the two sinks; the last 20 cells have a tenth of the
        # counts, so CyteArc's default filter removes them.
        position = np.sort(rng.uniform(0, 1, 380))
        labels = np.where(position < 1 / 3, "Alpha", np.where(position < 2 / 3, "Ductal", "Beta"))
        alpha, beta = rng.gamma(0.8, 1.0, (2, 1100))
        alpha[:60] *= 15
        beta[60:120] *= 15
        rates = (1 - position)[:, None] * alpha + position[:, None] * beta
        depth = np.full(position.size, 1500.0)
        low = rng.choice(position.size, 20, replace=False)
        depth[low] = 150.0
        counts = rng.poisson(rates / rates.sum(1, keepdims=True) * depth[:, None])
        folder = Path(cls.root.name) / "downloads" / "toy_h5ad"
        folder.mkdir(parents=True)
        ad.AnnData(
            X=sp.csr_matrix(counts.astype(np.float32)),
            obs=pd.DataFrame({"clusters": labels}, index=[f"cell{i}" for i in range(labels.size)]),
            var=pd.DataFrame(index=[f"G{i}" for i in range(counts.shape[1])]),
        ).write_h5ad(folder / "data.h5ad")
        cls.labels = labels
        cls.low = np.zeros(position.size, dtype=bool)
        cls.low[low] = True
        with (
            mock.patch.dict(catalogue_counts.COUNT_FILES, {"toy_h5ad": pinned(folder)}),
            mock.patch.object(fate_dense, "WORK", Path(cls.root.name) / "work"),
            mock.patch.dict(fate_dense.ANALYSES, {}, clear=True),
        ):
            cls.graph, cls.pseudotime, cls.annotations, cls.sinks = fate_dense.store_graph(
                "toy_h5ad", ["Alpha", "Beta"], "Ductal", ["Alpha", "Beta"], inputs=folder.parent
            )
            cls.record = dict(fate_dense.ANALYSES)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.cleanup()

    def test_the_default_filter_removed_the_low_count_cells(self) -> None:
        analysis = self.record["toy_h5ad"]["analysis"]
        self.assertLessEqual(analysis["analysisCells"], self.labels.size - int(self.low.sum()))
        self.assertEqual(analysis["filtering"]["method"], "mad")

    def test_annotations_and_graph_follow_the_analysis_cells(self) -> None:
        self.assertEqual(self.graph.shape[0], self.annotations.size)
        self.assertEqual(self.pseudotime.size, self.annotations.size)
        # The labels of the cells the analysis kept, in store order: never a removed low-count cell's.
        kept_labels = self.labels[~self.low]
        self.assertLessEqual(self.annotations.size, kept_labels.size)
        counts = {
            name: int((self.annotations == name).sum()) for name in ("Alpha", "Ductal", "Beta")
        }
        for name, count in counts.items():
            self.assertLessEqual(count, int((kept_labels == name).sum()))
            self.assertGreater(count, 0)

    def test_the_record_holds_the_pinned_input(self) -> None:
        files = self.record["toy_h5ad"]["input"]["files"]
        self.assertEqual(list(files), ["data.h5ad"])
        self.assertRegex(files["data.h5ad"]["sha256"], "^[0-9a-f]{64}$")


class KangGraphTests(unittest.TestCase):
    """A Kang lane is labelled with GEO's author labels and analysed on its labelled singlets only."""

    @classmethod
    def setUpClass(cls) -> None:
        import gzip

        import cytearc
        import scipy.io
        import scipy.sparse as sp

        cytearc.configure_output(level="ERROR", progress=False)
        cls.root = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(3)
        cells, genes = 300, 1100
        rates = rng.gamma(0.8, 1.0, (3, genes))
        for group in range(3):
            rates[group, group * 60 : group * 60 + 60] *= 15
        groups = np.repeat(np.arange(3), cells // 3)
        counts = rng.poisson(rates[groups] / rates[groups].sum(1, keepdims=True) * 1500)
        folder = Path(cls.root.name) / "downloads" / "toy_kang"
        folder.mkdir(parents=True)
        barcodes = [f"CELL{i:04d}-1" for i in range(cells)]
        with gzip.open(folder / "matrix.mtx.gz", "wb") as handle:
            scipy.io.mmwrite(handle, sp.coo_matrix(counts.T.astype(np.int64)), field="integer")
        with gzip.open(folder / "barcodes.tsv.gz", "wt") as handle:
            handle.writelines(f"{b}\n" for b in barcodes)
        with gzip.open(folder / "features.tsv.gz", "wt") as handle:
            handle.writelines(f"ENSG{i:011d}\tG{i}\n" for i in range(genes))
        # Every fifth barcode is a doublet; the others are singlets with their programme's label.
        calls = np.where(np.arange(cells) % 5 == 0, "doublet", "singlet")
        names = np.array(["B cells", "CD14+ Monocytes", "CD4 T cells"])[groups]
        cls.table = pd.DataFrame(
            {"stim": "ctrl", "cell": names, "multiplets": calls}, index=barcodes
        )
        cls.singlets = calls == "singlet"
        with (
            mock.patch.dict(catalogue_counts.COUNT_FILES, {"toy_kang": pinned(folder)}),
            mock.patch.dict(kang_labels.CONDITIONS, {"toy_kang": "ctrl"}),
            mock.patch.object(kang_labels, "cell_table", return_value=cls.table),
            mock.patch.object(fate_dense, "WORK", Path(cls.root.name) / "work"),
            mock.patch.dict(fate_dense.ANALYSES, {}, clear=True),
        ):
            cls.ds, cls.analysis = fate_dense.analysed_store(
                "toy_kang", "toy_kang.zarr", inputs=folder.parent
            )
            cls.record = dict(fate_dense.ANALYSES)["toy_kang"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.cleanup()

    def test_the_analysis_covers_only_labelled_singlets(self) -> None:
        from experiments.runtime import rna_recipe

        kept = rna_recipe.analysis_cells(self.ds, self.analysis)
        self.assertFalse(kept[~self.singlets].any())
        self.assertGreater(int(kept.sum()), 0)
        labels = np.asarray(self.ds.cells.fetch_all(kang_labels.LABEL_COLUMN)).astype(str)
        self.assertNotIn(kang_labels.UNLABELLED, set(labels[kept].tolist()))

    def test_the_record_counts_the_singlets_and_their_labels(self) -> None:
        labels = self.record["labels"]
        self.assertEqual(labels["singlets"], int(self.singlets.sum()))
        self.assertEqual(labels["barcodes"], self.singlets.size)
        self.assertEqual(sum(labels["cellsPerLabel"].values()), int(self.singlets.sum()))


class KangLabelTests(unittest.TestCase):
    """GEO's author labels reach the cells demuxlet called singlets, with the -11 suffix read as -1."""

    TABLE = pd.DataFrame(
        {
            "stim": ["ctrl", "ctrl", "ctrl", "stim", "stim", "stim", "stim"],
            "cell": [
                "B cells",
                "NK cells",
                "CD4 T cells",
                "B cells",
                "CD14+ Monocytes",
                "NK cells",
                None,
            ],
            "multiplets": [
                "singlet",
                "doublet",
                "singlet",
                "singlet",
                "ambs",
                "singlet",
                "singlet",
            ],
        },
        index=["AAAC-1", "AAAG-1", "AAAT-1", "AAAC-11", "AAAG-1", "AAAT-1", "AACA-1"],
    )

    def test_singlets_carry_their_label_and_other_calls_none(self) -> None:
        labels = kang_labels.singlet_labels(self.TABLE, "ctrl", ["AAAT-1", "AAAG-1", "AAAC-1"])
        self.assertEqual(labels.tolist(), ["CD4 T cells", kang_labels.UNLABELLED, "B cells"])

    def test_stimulated_barcodes_with_the_11_suffix_find_their_label(self) -> None:
        labels = kang_labels.singlet_labels(self.TABLE, "stim", ["AAAC-1", "AAAG-1", "AAAT-1"])
        self.assertEqual(labels.tolist(), ["B cells", kang_labels.UNLABELLED, "NK cells"])

    def test_a_singlet_without_a_cell_type_stays_unlabelled(self) -> None:
        labels = kang_labels.singlet_labels(self.TABLE, "stim", ["AACA-1", "AAAT-1"])
        self.assertEqual(labels.tolist(), [kang_labels.UNLABELLED, "NK cells"])

    def test_a_barcode_missing_from_the_table_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 ctrl barcodes are not in the GEO table"):
            kang_labels.singlet_labels(self.TABLE, "ctrl", ["AAAC-1", "CCCC-1"])

    def test_the_table_is_read_only_with_its_pinned_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "GSE96583" / kang_labels.FILE_NAME
            path.parent.mkdir(parents=True)
            self.TABLE.rename_axis(None).to_csv(path, sep="\t", compression="gzip")
            size, digest = path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()
            with (
                mock.patch.object(kang_labels, "BYTES", size),
                mock.patch.object(kang_labels, "SHA256", digest),
            ):
                table = kang_labels.cell_table(Path(folder))
            self.assertEqual(table.loc["AAAT-1", "cell"].tolist(), ["CD4 T cells", "NK cells"])
            self.assertTrue(pd.isna(table.loc["AACA-1", "cell"]))
            with (
                mock.patch.object(kang_labels, "BYTES", size),
                mock.patch.object(kang_labels, "SHA256", "0" * 64),
                self.assertRaisesRegex(ValueError, "not the pinned"),
            ):
                kang_labels.cell_table(Path(folder))

    def test_the_pin_is_the_table_the_doublet_check_reads(self) -> None:
        self.assertEqual(kang_labels.URL, kang_doublet_score_comparison.GEO_URL)
        self.assertEqual(kang_labels.FILE_NAME, kang_doublet_score_comparison.GEO_FILE.name)


class PlacementTests(unittest.TestCase):
    def test_the_placement_modal_reports_is_returned_and_missing_values_stay_empty(self) -> None:
        from experiments.runtime import method_checks

        with mock.patch.dict(
            os.environ, {"MODAL_REGION": "us-east", "MODAL_CLOUD_PROVIDER": "CLOUD_PROVIDER_AWS"}
        ):
            self.assertEqual(
                method_checks.placement(),
                {"region": "us-east", "cloudProvider": "CLOUD_PROVIDER_AWS"},
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(method_checks.placement(), {"region": None, "cloudProvider": None})


if __name__ == "__main__":
    unittest.main()
