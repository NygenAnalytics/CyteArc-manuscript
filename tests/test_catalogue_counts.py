import gzip
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.io
import scipy.sparse as sp
from experiments.runtime import catalogue_counts, rna_recipe

RNG = np.random.default_rng(11)
# The manual filter's upper bound on total counts, below most cells' 1,200 counts so the filter removes some.
HIGH = 1220


def counts_matrix(cells: int = 300, genes: int = 120) -> sp.csr_matrix:
    """Three expression programmes, so a graph and clusters have structure to find."""
    rates = RNG.gamma(0.8, 1.0, (3, genes))
    for group in range(3):
        rates[group, group * 20 : group * 20 + 20] *= 15
    groups = np.repeat(np.arange(3), cells // 3)
    values = RNG.poisson(rates[groups] / rates[groups].sum(1, keepdims=True) * 1200)
    return sp.csr_matrix(values.astype(np.float32))


def write_mtx(folder: Path, matrix: sp.csr_matrix) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / "matrix.mtx.gz", "wb") as handle:
        scipy.io.mmwrite(handle, sp.coo_matrix(matrix.T.astype(np.int64)), field="integer")
    with gzip.open(folder / "barcodes.tsv.gz", "wt") as handle:
        handle.writelines(f"CELL{i:05d}-1\n" for i in range(matrix.shape[0]))
    with gzip.open(folder / "features.tsv.gz", "wt") as handle:
        handle.writelines(
            f"ENSG{i:011d}\t{'MT-' if i < 3 else 'G'}{i}\n" for i in range(matrix.shape[1])
        )


def pins(folder: Path) -> dict[str, tuple[int, str]]:
    return {
        path.name: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(folder.iterdir())
    }


class PinnedFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "data.h5"
        self.path.write_bytes(b"counts" * 1000)
        self.digest = hashlib.sha256(b"counts" * 1000).hexdigest()

    def tearDown(self) -> None:
        self.folder.cleanup()

    def test_a_file_with_its_pinned_size_and_hash_passes(self) -> None:
        catalogue_counts.check_file(self.path, 6000, self.digest)

    def test_a_file_of_another_size_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "has 6000 bytes, not the pinned 5999"):
            catalogue_counts.check_file(self.path, 5999, self.digest)

    def test_a_file_with_the_same_size_and_other_bytes_is_refused(self) -> None:
        self.path.write_bytes(b"Counts" * 1000)
        with self.assertRaisesRegex(ValueError, "not the pinned"):
            catalogue_counts.check_file(self.path, 6000, self.digest)

    def test_a_missing_file_is_refused(self) -> None:
        with self.assertRaises(FileNotFoundError):
            catalogue_counts.check_file(self.path.with_name("absent.h5"), 1, self.digest)

    def test_every_documentation_dataset_pins_sha256_digests(self) -> None:
        for dataset, files in catalogue_counts.COUNT_FILES.items():
            for name, (size, digest) in files.items():
                with self.subTest(dataset=dataset, name=name):
                    self.assertGreater(size, 0)
                    self.assertRegex(digest, "^[0-9a-f]{64}$")
        record = catalogue_counts.input_record("tenx_5K_pbmc_rnaseq")
        self.assertEqual(record["catalogue"], "cytearc_docs")
        self.assertEqual(record["files"]["data.h5"]["bytes"], 18098007)


class CountFilesTests(unittest.TestCase):
    """count_files downloads only missing files and checks every file against its pin."""

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()
        self.source = Path(self.root.name) / "catalogue" / "toy_rnaseq"
        write_mtx(self.source, counts_matrix(30, 40))
        self.pins = {"toy_rnaseq": pins(self.source)}
        self.downloads = Path(self.root.name) / "downloads"
        self.requested: list[str] = []
        root = Path(self.root.name) / "catalogue"

        class Repository:
            def download(inner, path: str, destination: Path) -> None:
                self.requested.append(path)
                target = Path(destination) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((root / path).read_bytes())

        self.repository = Repository()

    def tearDown(self) -> None:
        self.root.cleanup()

    def fetch(self) -> Path:
        with (
            mock.patch.dict(catalogue_counts.COUNT_FILES, self.pins),
            mock.patch("cytearc.datasets.connect", return_value=self.repository),
        ):
            return catalogue_counts.count_files("toy_rnaseq", self.downloads)

    def test_missing_files_are_downloaded_and_present_ones_are_not(self) -> None:
        folder = self.fetch()
        self.assertEqual(
            sorted(self.requested),
            [f"toy_rnaseq/{name}" for name in sorted(self.pins["toy_rnaseq"])],
        )
        self.requested.clear()
        self.assertEqual(self.fetch(), folder)
        self.assertEqual(self.requested, [])

    def test_a_downloaded_file_that_differs_from_its_pin_is_refused(self) -> None:
        name = "barcodes.tsv.gz"
        size, _ = self.pins["toy_rnaseq"][name]
        self.pins["toy_rnaseq"][name] = (size, "0" * 64)
        with self.assertRaisesRegex(ValueError, "not the pinned"):
            self.fetch()

    def test_an_unknown_dataset_is_refused_before_any_download(self) -> None:
        with self.assertRaisesRegex(KeyError, "has no pinned count files"):
            catalogue_counts.count_files("unknown_rnaseq", self.downloads)
        self.assertEqual(self.requested, [])


class BuildStoreTests(unittest.TestCase):
    """Stores built from each input format hold the input's counts and open prepared."""

    @classmethod
    def setUpClass(cls) -> None:
        import anndata as ad
        import cytearc
        import pandas as pd

        cytearc.configure_output(level="ERROR", progress=False)
        cls.root = tempfile.TemporaryDirectory()
        cls.downloads = Path(cls.root.name) / "downloads"
        cls.matrix = counts_matrix()
        write_mtx(cls.downloads / "toy_mtx", cls.matrix)
        h5ad_folder = cls.downloads / "toy_h5ad"
        h5ad_folder.mkdir(parents=True)
        cells = cls.matrix.shape[0]
        ad.AnnData(
            X=cls.matrix,
            obs=pd.DataFrame(
                {"clusters": np.repeat(["Ductal", "Alpha", "Beta"], cells // 3)},
                index=[f"cell{i}" for i in range(cells)],
            ),
            var=pd.DataFrame(index=[f"G{i}" for i in range(cls.matrix.shape[1])]),
        ).write_h5ad(h5ad_folder / "data.h5ad")
        cls.pins = {
            "toy_mtx": pins(cls.downloads / "toy_mtx"),
            "toy_h5ad": pins(h5ad_folder),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.cleanup()

    def build(self, dataset: str):
        import cytearc

        with mock.patch.dict(catalogue_counts.COUNT_FILES, self.pins):
            store = catalogue_counts.build_store(
                dataset,
                Path(self.root.name) / f"{dataset}.zarr",
                downloads=self.downloads,
                workers=1,
            )
        return cytearc.DataStore(str(store), default_assay="RNA", zarr_mode="r", nthreads=1)

    def test_an_mtx_store_holds_the_counts_of_every_barcode(self) -> None:
        ds = self.build("toy_mtx")
        self.assertEqual((ds.cells.N, ds.RNA.feats.N), self.matrix.shape)
        totals = np.asarray(ds.cells.fetch_all("RNA_nCounts"), dtype=np.float64)
        np.testing.assert_array_equal(totals, np.asarray(self.matrix.sum(axis=1)).ravel())
        self.assertEqual(ds.cells.fetch_all("ids")[0], "CELL00000-1")

    def test_an_h5ad_store_keeps_the_cell_metadata(self) -> None:
        ds = self.build("toy_h5ad")
        self.assertEqual(ds.cells.N, self.matrix.shape[0])
        labels = np.asarray(ds.cells.fetch_all("clusters")).astype(str)
        self.assertEqual(sorted(set(labels.tolist())), ["Alpha", "Beta", "Ductal"])
        self.assertEqual(int((labels == "Alpha").sum()), self.matrix.shape[0] // 3)

    def test_a_rebuilt_store_replaces_the_earlier_one(self) -> None:
        first = self.build("toy_mtx")
        second = self.build("toy_mtx")
        self.assertEqual(first.cells.N, second.cells.N)


class RnaRecipeTests(unittest.TestCase):
    """run_rna passes each stage's settings to pipeline.run, and analysis_record reports what CyteArc resolved."""

    @classmethod
    def setUpClass(cls) -> None:
        import cytearc

        cytearc.configure_output(level="ERROR", progress=False)
        cls.root = tempfile.TemporaryDirectory()
        downloads = Path(cls.root.name) / "downloads"
        # More genes than CyteArc's default 1,000 variable genes, so the defaults run as they would on real data.
        write_mtx(downloads / "toy_mtx", counts_matrix(300, 1200))
        with mock.patch.dict(
            catalogue_counts.COUNT_FILES, {"toy_mtx": pins(downloads / "toy_mtx")}
        ):
            store = catalogue_counts.build_store(
                "toy_mtx", Path(cls.root.name) / "toy.zarr", downloads=downloads, workers=1
            )
        cls.ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=1)
        switches = dict(umap=False, cell_cycle=False, paris=False, doublets=False, markers=False)
        cls.settings_run = rna_recipe.run_rna(
            cls.ds,
            label="settings",
            filtering=False,
            hvg={"top_n": 50},
            pca={"dims": 5},
            neighbors={"k": 7},
            leiden={"partitions": [0.5], "random_seed": 4444},
            **switches,
        )
        cls.filtered = rna_recipe.run_rna(
            cls.ds,
            label="filtered",
            filtering={"method": "manual", "attrs": ["RNA_nCounts"], "lows": [0], "highs": [HIGH]},
            **switches,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.cleanup()

    def test_each_stage_ran_with_the_settings_given(self) -> None:
        ds, run = self.ds, self.settings_run
        hvg = np.asarray(
            ds.artifacts.load(run["highly_variable_features"])["values"][:], dtype=bool
        )
        self.assertEqual(int(hvg.sum()), 50)
        self.assertEqual(ds.artifacts.load(run["pca"])["data"].shape[1], 5)
        self.assertEqual(ds.artifacts.load(run["neighbors"])["indices"].shape[1], 7)
        self.assertEqual(ds.artifacts.inspect(run["clusters"]).parameters["random_seed"], 4444)

    def test_the_record_holds_the_resolved_settings_and_the_index_defaults(self) -> None:
        record = rna_recipe.analysis_record(self.ds, self.settings_run)
        self.assertEqual((record["hvgCount"], record["pcaDims"], record["neighborsK"]), (50, 5, 7))
        self.assertEqual(record["leiden"]["partitions"], [0.5])
        self.assertEqual(record["filtering"], {"enabled": False})
        # CyteArc 1.3.0's HNSW defaults, which the record must state because the checks do not choose them.
        self.assertEqual(
            {k: record["annIndex"][k] for k in ("ann_m", "ann_ef", "ann_efc")},
            {"ann_m": 24, "ann_ef": 200, "ann_efc": 200},
        )
        self.assertEqual(record["analysisCells"], self.ds.cells.N)

    def test_settings_not_given_take_cytearcs_defaults(self) -> None:
        record = rna_recipe.analysis_record(self.ds, self.filtered)
        self.assertEqual(
            (record["hvgCount"], record["pcaDims"], record["neighborsK"]), (1000, 21, 11)
        )

    def test_the_analysis_cells_are_those_the_filter_kept(self) -> None:
        kept = rna_recipe.analysis_cells(self.ds, self.filtered)
        totals = np.asarray(self.ds.cells.fetch_all("RNA_nCounts"), dtype=np.float64)
        # Without keep_bounds, a manual filter keeps the cells strictly inside its bounds.
        np.testing.assert_array_equal(kept, (totals > 0) & (totals < HIGH))
        self.assertLess(int(kept.sum()), self.ds.cells.N)
        self.assertEqual(
            rna_recipe.analysis_record(self.ds, self.filtered)["analysisCells"], int(kept.sum())
        )

    def test_a_leiden_setting_that_is_neither_a_mapping_nor_a_bool_is_refused(self) -> None:
        with self.assertRaisesRegex(TypeError, "leiden must be"):
            rna_recipe.run_rna(self.ds, label="bad", leiden=1)


if __name__ == "__main__":
    os.environ.setdefault("CYTEARC_RUNS", tempfile.gettempdir())
    unittest.main()
