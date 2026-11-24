"""Convert a CELLxGENE Discover H5AD to the layout of the benchmark samples.

The benchmark samples hold raw counts as a CSR matrix (float32 values, int64 indices)
in X, the cell identifiers in obs/_index, and the gene identifiers and symbols in
var/_index and var/feature_name. CELLxGENE files keep raw counts in raw/X when X holds
normalised values, and store gene symbols as a categorical column, so both are
normalised here. The conversion streams the matrix in slices and never loads it whole.
"""

import time
from pathlib import Path

import h5py
import numpy as np

from experiments.runtime.h5ad import (
    _column,
    _dataframe,
    _index,
    _string_dataset,
)

SLICE = 100000000


def convert(source: Path, destination: Path) -> dict:
    started = time.time()
    with h5py.File(source, "r") as h5:
        uses_raw = "raw" in h5 and "X" in h5["raw"]
        matrix = h5["raw"]["X"] if uses_raw else h5["X"]
        encoding = matrix.attrs.get("encoding-type")
        encoding = encoding.decode() if isinstance(encoding, bytes) else encoding
        if encoding != "csr_matrix":
            raise ValueError(f"expected a CSR count matrix, found {encoding!r}")
        n_cells, n_genes = (int(v) for v in matrix.attrs["shape"])
        var_group = h5["raw"]["var"] if uses_raw else h5["var"]
        gene_ids = _index(var_group)
        if "feature_name" in var_group:
            symbols = _column(var_group, "feature_name")
        else:
            lookup = dict(zip(_index(h5["var"]), _column(h5["var"], "feature_name")))
            symbols = np.asarray([lookup.get(g, g) for g in gene_ids], dtype=object)
        cells = _index(h5["obs"])
        if len(cells) != n_cells or len(gene_ids) != n_genes:
            raise ValueError("obs or var length does not match the matrix shape")
        data, indices, indptr = (matrix["data"], matrix["indices"], matrix["indptr"])
        nnz = int(data.shape[0])
        integral = bool(np.all(np.mod(data[: min(nnz, 1000000)], 1) == 0))
        schema = (
            h5["uns"]["schema_version"][()]
            if "uns" in h5 and "schema_version" in h5["uns"]
            else None
        )
        partial = destination.with_suffix(".part")
        with h5py.File(partial, "w") as out:
            out.attrs["encoding-type"] = "anndata"
            out.attrs["encoding-version"] = "0.1.0"
            x = out.create_group("X")
            x.attrs["encoding-type"] = "csr_matrix"
            x.attrs["encoding-version"] = "0.1.0"
            x.attrs["shape"] = np.asarray([n_cells, n_genes], dtype=np.int64)
            out_data = x.create_dataset(
                "data", shape=(nnz,), dtype=np.float32, chunks=(min(nnz, 1048576),)
            )
            out_indices = x.create_dataset(
                "indices", shape=(nnz,), dtype=np.int64, chunks=(min(nnz, 1048576),)
            )
            for start in range(0, nnz, SLICE):
                end = min(start + SLICE, nnz)
                out_data[start:end] = data[start:end].astype(np.float32)
                out_indices[start:end] = indices[start:end].astype(np.int64)
            x.create_dataset("indptr", data=indptr[:].astype(np.int64))
            obs = _dataframe(out, "obs", ())
            _string_dataset(obs, "_index", cells)
            var = _dataframe(out, "var", ("feature_name",))
            _string_dataset(var, "_index", gene_ids)
            _string_dataset(var, "feature_name", symbols)
        partial.replace(destination)
    return {
        "nCells": n_cells,
        "nGenes": n_genes,
        "nnz": nnz,
        "countsFrom": "raw/X" if uses_raw else "X",
        "integralCountsInFirstMillion": integral,
        "schemaVersion": schema.decode() if isinstance(schema, bytes) else schema,
        "bytes": destination.stat().st_size,
        "convertSeconds": time.time() - started,
    }
