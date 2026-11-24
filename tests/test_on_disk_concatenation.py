import shutil
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.experimental import concat_on_disk
from experiments.reuse.composite_and_remote import (
    add_feature_names,
    identifier_hash,
    prepare_for_on_disk_concatenation,
    validate_on_disk_concatenation,
)

GENES = 40


def write_export(path: Path, study: str, cells: int, clinical: pd.Series, seed: int) -> dict:
    """An H5AD laid out like the blood-cohort exports: int64 CSR, no varm, varp, layers or obsp."""
    counts = np.random.default_rng(seed).poisson(0.4, size=(cells, GENES)).astype(np.float32)
    matrix = sp.csr_matrix(counts)
    matrix.indices = matrix.indices.astype(np.int64)
    matrix.indptr = matrix.indptr.astype(np.int64)
    cell_ids = [f"{study}__cell{i}" for i in range(cells)]
    gene_ids = [f"ENSG{j:05d}" for j in range(GENES)]
    obs = pd.DataFrame({"orig_clinical": clinical.to_numpy()}, index=cell_ids)
    var = pd.DataFrame({"gene_short_name": [f"GENE{j}" for j in range(GENES)]}, index=gene_ids)
    with h5py.File(path, "w") as handle:
        handle.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        ad.io.write_elem(handle, "X", matrix)
        ad.io.write_elem(handle, "obs", obs)
        ad.io.write_elem(handle, "var", var)
        ad.io.write_elem(handle, "obsm", {})
        ad.io.write_elem(handle, "uns", {})
    return {
        "name": study,
        "cells": cells,
        "matrix": counts,
        "clinical": clinical,
        "cellIdSha256": identifier_hash(cell_ids),
        "featureIdSha256": identifier_hash(gene_ids),
    }


class OnDiskConcatenationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        # One study records a clinical value as text; the other has the column but no values,
        # which the export stores as an empty categorical, as in the blood-cohort files.
        self.inputs = [
            write_export(
                self.directory / "first.h5ad",
                "first",
                30,
                pd.Series(["0.5", "not applicable", None] * 10, dtype="str"),
                seed=1,
            ),
            write_export(
                self.directory / "second.h5ad",
                "second",
                20,
                pd.Series(pd.Categorical([None] * 20)),
                seed=2,
            ),
        ]
        self.paths = {item["name"]: self.directory / f"{item['name']}.h5ad" for item in self.inputs}
        self.nonzero = sum(int(np.count_nonzero(item["matrix"])) for item in self.inputs)

    def concatenate(self, target: Path) -> None:
        concat_on_disk(
            {name: str(path) for name, path in self.paths.items()},
            str(target),
            join="inner",
            merge="first",
        )

    def test_exported_layout_is_rejected_by_concat_on_disk(self) -> None:
        with self.assertRaises((TypeError, KeyError)):
            self.concatenate(self.directory / "as_exported.h5ad")

    def test_prepared_inputs_concatenate_to_the_stacked_counts(self) -> None:
        report = prepare_for_on_disk_concatenation(self.paths)
        self.assertEqual(report["obsColumnsStoredAsStringCategories"], ["orig_clinical"])
        self.assertEqual(report["addedEmptyMappings"]["first"], ["layers", "obsp", "varm", "varp"])
        output = self.directory / "concatenated.h5ad"
        self.concatenate(output)
        checked = validate_on_disk_concatenation(output, self.inputs, self.paths, self.nonzero, 7)
        self.assertTrue(checked["exactCountValuesAndStructure"])
        result = ad.read_h5ad(output)
        expected = np.vstack([item["matrix"] for item in self.inputs])
        np.testing.assert_array_equal(result.X.toarray(), expected)
        clinical = (
            result.obs["orig_clinical"]
            .astype(object)
            .where(result.obs["orig_clinical"].notna(), None)
        )
        self.assertEqual(clinical.tolist(), ["0.5", "not applicable", None] * 10 + [None] * 20)

    def test_validation_detects_a_changed_count(self) -> None:
        prepare_for_on_disk_concatenation(self.paths)
        output = self.directory / "concatenated.h5ad"
        self.concatenate(output)
        with h5py.File(output, "r+") as handle:
            handle["X"]["data"][self.nonzero - 1] += 1
        with self.assertRaisesRegex(AssertionError, "Counts of second changed"):
            validate_on_disk_concatenation(output, self.inputs, self.paths, self.nonzero, 7)

    def test_validation_detects_cells_out_of_order(self) -> None:
        prepare_for_on_disk_concatenation(self.paths)
        output = self.directory / "concatenated.h5ad"
        self.concatenate(output)
        swapped = [
            {**self.inputs[0], "cellIdSha256": self.inputs[1]["cellIdSha256"]},
            self.inputs[1],
        ]
        with self.assertRaisesRegex(AssertionError, "Cell identities or order of first"):
            validate_on_disk_concatenation(output, swapped, self.paths, self.nonzero, 7)

    def test_feature_names_come_from_gene_symbols_once(self) -> None:
        prepare_for_on_disk_concatenation(self.paths)
        output = self.directory / "concatenated.h5ad"
        self.concatenate(output)
        self.assertEqual(
            add_feature_names(output)["featureName"], "copied from var.gene_short_name"
        )
        self.assertEqual(add_feature_names(output)["featureName"], "present in the inputs")
        var = ad.read_h5ad(output, backed="r").var
        self.assertEqual(var["feature_name"].astype(str).tolist()[:2], ["GENE0", "GENE1"])


if __name__ == "__main__":
    unittest.main()
