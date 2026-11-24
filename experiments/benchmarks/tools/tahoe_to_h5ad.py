"""Write staged Tahoe-100M Parquet files as uncompressed H5AD shards, the input the R workflows read.

Usage: python -m experiments.benchmarks.tools.tahoe_to_h5ad SOURCE DESTINATION SHARDS

SOURCE is a staged Tahoe-100M revision as the harness downloads it: the first n data files in
order (data/train-NNNNN-of-03388.parquet), each one's staging record
(index/train-NNNNN-of-03388.json) and the gene table (metadata/gene_metadata.parquet). Counts are
read with tahoe.read_file_counts, the reader of CyteArc's own import, so both tools analyse the same
matrix, and each file must hold the cells and values its staging record counts.

DESTINATION receives at most SHARDS shards (shard_000.h5ad, ...), each a run of consecutive files,
so the shards in name order hold the cells in file order. One process writes each shard, in the
layout of the benchmark samples (tools/prepare_cellxgene.py): raw counts as an uncompressed CSR
matrix with float32 values and int64 indices in X, cell identifiers in obs/_index, and gene
identifiers and symbols in var/_index and var/feature_name. Each data file is deleted as soon as
its cells are in a shard, so the Parquet input and the shards share the disk only file by file.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from experiments.benchmarks.tahoe import (
    CELL_ID_COLUMN,
    GENE_METADATA_PATH,
    load_gene_table,
    read_file_counts,
    selected_data_files,
)
from experiments.runtime.h5ad import _dataframe, _string_dataset

# The chunk length of the benchmark samples' values and indices.
CHUNK = 1048576
# float32 holds every integer count below 2^24 exactly.
LARGEST_EXACT_COUNT = 2**24


def data_files(source: Path) -> list[Path]:
    """The data files under SOURCE, which must be the first n files of Tahoe-100M in order."""
    found = sorted(path.relative_to(source).as_posix() for path in source.glob("data/*.parquet"))
    if not found:
        raise ValueError(f"{source / 'data'} holds no Parquet files")
    expected = list(selected_data_files(len(found)))
    if found != expected:
        missing = sorted(set(expected) - set(found))
        raise ValueError(
            f"The data files must be the first {len(found)} of Tahoe-100M in order; "
            f"{missing[0] if missing else found[-1]} breaks the run"
        )
    return [source / name for name in found]


def staging_record(source: Path, data_file: Path) -> dict:
    path = source / "index" / f"{data_file.stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"{data_file.name} has no staging record at {path}")
    record = json.loads(path.read_text())
    return {"nCells": int(record["nCells"]), "nnz": int(record["nnz"])}


def shard_row_ends(file_indptr: np.ndarray, offset: int) -> np.ndarray:
    """A file's row ends within its shard, as int64: the file's own CSR row ends (scipy keeps them in int32 when
    the file allows) shifted by the OFFSET values the shard already holds, which passes 2^31 after about 70
    Tahoe-100M files."""
    return file_indptr[1:].astype(np.int64) + offset


def write_shard(files: list[Path], records: list[dict], genePath: Path, target: Path) -> dict:
    """Write the cells of FILES, in order, to the shard TARGET, deleting each file once written."""
    import pyarrow.parquet as pq

    genes = load_gene_table(genePath)
    n_cells = sum(record["nCells"] for record in records)
    nnz = sum(record["nnz"] for record in records)
    if nnz == 0:
        raise ValueError(f"{target.name} would hold no values")
    partial = target.with_name(f".{target.name}.part")
    cell_ids: list[str] = []
    cells = values = 0
    with h5py.File(partial, "w") as out:
        out.attrs["encoding-type"] = "anndata"
        out.attrs["encoding-version"] = "0.1.0"
        x = out.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = np.asarray([n_cells, genes.nFeatures], dtype=np.int64)
        chunks = (min(nnz, CHUNK),)
        data = x.create_dataset("data", shape=(nnz,), dtype=np.float32, chunks=chunks)
        indices = x.create_dataset("indices", shape=(nnz,), dtype=np.int64, chunks=chunks)
        indptr = np.zeros(n_cells + 1, dtype=np.int64)
        for path, record in zip(files, records, strict=True):
            matrix = read_file_counts(path, genes)
            if matrix.shape[0] != record["nCells"] or matrix.nnz != record["nnz"]:
                raise ValueError(
                    f"{path.name} holds {matrix.shape[0]} cells and {matrix.nnz} values; "
                    f"its staging record counts {record['nCells']} and {record['nnz']}"
                )
            if matrix.nnz and int(matrix.data.max()) >= LARGEST_EXACT_COUNT:
                raise ValueError(f"{path.name} has a count that float32 cannot hold exactly")
            ids = pq.read_table(path, columns=[CELL_ID_COLUMN]).column(CELL_ID_COLUMN).to_pylist()
            if len(ids) != matrix.shape[0]:
                raise ValueError(f"{path.name} names {len(ids)} cells for {matrix.shape[0]} rows")
            data[values : values + matrix.nnz] = matrix.data.astype(np.float32)
            indices[values : values + matrix.nnz] = matrix.indices.astype(np.int64)
            indptr[cells + 1 : cells + matrix.shape[0] + 1] = shard_row_ends(matrix.indptr, values)
            cell_ids.extend(ids)
            cells += matrix.shape[0]
            values += matrix.nnz
            out.flush()
            path.unlink()
        x.create_dataset("indptr", data=indptr)
        obs = _dataframe(out, "obs", ())
        _string_dataset(obs, "_index", np.asarray(cell_ids, dtype=object))
        var = _dataframe(out, "var", ("feature_name",))
        _string_dataset(var, "_index", genes.featureIds)
        _string_dataset(var, "feature_name", genes.featureNames)
    partial.replace(target)
    return {"shard": target.name, "files": len(files), "cells": cells, "values": values}


def convert(source: Path, destination: Path, shards: int) -> dict:
    """Write the data files under SOURCE as at most SHARDS H5AD shards in DESTINATION."""
    if shards < 1:
        raise ValueError("SHARDS must be at least 1")
    started = time.perf_counter()
    files = data_files(source)
    records = [staging_record(source, path) for path in files]
    gene_path = source / GENE_METADATA_PATH
    if not gene_path.exists():
        raise FileNotFoundError(f"The gene table is missing: {gene_path}")
    destination.mkdir(parents=True, exist_ok=True)
    groups = np.array_split(np.arange(len(files)), min(shards, len(files)))
    with ProcessPoolExecutor(max_workers=len(groups)) as pool:
        futures = [
            pool.submit(
                write_shard,
                [files[i] for i in group],
                [records[i] for i in group],
                gene_path,
                destination / f"shard_{number:03d}.h5ad",
            )
            for number, group in enumerate(groups)
        ]
        written = [future.result() for future in futures]
    return {
        "files": len(files),
        "shards": len(written),
        "cells": sum(item["cells"] for item in written),
        "values": sum(item["values"] for item in written),
        "seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    source, destination, shards = sys.argv[1:]
    print(json.dumps(convert(Path(source), Path(destination), int(shards))), flush=True)
