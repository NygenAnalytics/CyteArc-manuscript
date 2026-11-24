import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# The experiment modules resolve their data folders from CYTEARC_DOWNLOADS and CYTEARC_RUNS when they are
# imported; these tests write only to their own temporary folder.
with mock.patch.dict(
    os.environ,
    {
        "CYTEARC_DOWNLOADS": str(Path(tempfile.gettempdir()) / "cytearc-downloads-unused"),
        "CYTEARC_RUNS": str(Path(tempfile.gettempdir()) / "cytearc-runs-unused"),
    },
):
    from experiments.integration import harmony_binvignat


class PreparedStoreTests(unittest.TestCase):
    """The Binvignat checks mount the store that prepare_binvignat converts; CyteArc mounts only a prepared store."""

    @classmethod
    def setUpClass(cls) -> None:
        import anndata as ad
        import cytearc
        import pandas as pd
        import scipy.sparse as sp

        cls.folder = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(5)
        counts = rng.poisson(2.0, (120, 60)).astype(np.float32)
        adata = ad.AnnData(
            X=sp.csr_matrix(counts),
            obs=pd.DataFrame(
                {"batch": np.repeat(["1", "2", "3"], 40)},
                index=[f"cell{i}" for i in range(120)],
            ),
            var=pd.DataFrame(
                {"feature_name": [f"G{i}" for i in range(60)]},
                index=[f"ENSG{i:011d}" for i in range(60)],
            ),
        )
        h5ad = Path(cls.folder.name) / "cells.h5ad"
        adata.write_h5ad(h5ad)
        cytearc.configure_output(level="ERROR", progress=False)
        cls.counts = counts

        def converted(name: str) -> Path:
            destination = Path(cls.folder.name) / name
            reader = cytearc.H5adReader(str(h5ad), feature_name_key="feature_name")
            try:
                cytearc.H5adToZarr(reader, zarr_loc=str(destination), nthreads=1).dump()
            finally:
                reader.close()
            return destination

        cls.converted = staticmethod(converted)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def mount(self, source: Path, name: str):
        import cytearc

        return cytearc.mount_datastore(
            str(source),
            at=str(Path(self.folder.name) / name),
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=1,
        )

    def test_a_converted_store_is_refused_until_it_is_prepared(self) -> None:
        source = self.converted("unprepared.zarr")
        with self.assertRaisesRegex(ValueError, "Assay 'RNA' is not prepared yet"):
            self.mount(source, "unprepared_mount.zarr")

    def test_a_prepared_store_mounts_with_its_cells_and_columns(self) -> None:
        source = self.converted("prepared.zarr")
        harmony_binvignat.prepare_store(source, 1)
        ds = self.mount(source, "prepared_mount.zarr")
        self.assertEqual(ds.cells.N, 120)
        batches = np.asarray(ds.cells.fetch_all("batch")).astype(str)
        self.assertEqual(sorted(set(batches.tolist())), ["1", "2", "3"])
        self.assertEqual(int(np.count_nonzero(batches == "2")), 40)


if __name__ == "__main__":
    unittest.main()
