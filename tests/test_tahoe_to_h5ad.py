import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from experiments.benchmarks import tahoe
from experiments.benchmarks.tools import tahoe_to_h5ad, uncompress_h5ad

# Gene tokens out of order and with gaps, as Tahoe-100M's gene table numbers them after the marker.
TOKENS = [17, 3, 9, 4, 25, 11, 6, 30]
SYMBOLS = ["MT-CO1", "ACTB", "RPS6", "GAPDH", "CD3E", "MALAT1", "RPL13", "XIST"]
# Each file's cells as (cell ID, {token: count}); a zero count is an explicit zero the reader drops.
FILES = [
    [("a1", {3: 2, 9: 1, 17: 5}), ("a2", {4: 1}), ("a3", {30: 7, 25: 0, 6: 3})],
    [("b1", {11: 1, 3: 4}), ("b2", {9: 2, 6: 1, 25: 3, 30: 1})],
    [("c1", {17: 1}), ("c2", {4: 9, 11: 2}), ("c3", {3: 1, 30: 2}), ("c4", {25: 4})],
]


def gene_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "token_id": pa.array(TOKENS, pa.int64()),
                "ensembl_id": [f"ENSG{token:011d}" for token in TOKENS],
                "gene_symbol": SYMBOLS,
            }
        ),
        path,
    )


def data_file(path: Path, cells: list[tuple[str, dict[int, int]]]) -> None:
    """A data file in Tahoe-100M's schema: each row's genes start with marker token 1."""
    genes, values = [], []
    for _, counts in cells:
        tokens = sorted(counts)
        genes.append([tahoe.MARKER_TOKEN, *tokens])
        values.append([float(len(tokens)), *(float(counts[token]) for token in tokens)])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "genes": pa.array(genes, pa.list_(pa.int64())),
                "expressions": pa.array(values, pa.list_(pa.float32())),
                "drug": ["DMSO"] * len(cells),
                "sample": ["s1"] * len(cells),
                tahoe.CELL_ID_COLUMN: [cell for cell, _ in cells],
                "cell_line_id": ["CVCL_0023"] * len(cells),
            }
        ),
        path,
    )


def staged(root: Path, files: list = FILES) -> Path:
    """A staged revision as the harness downloads it, with each file's record counted by hand."""
    gene_table(root / tahoe.GENE_METADATA_PATH)
    for number, cells in enumerate(files):
        path = root / tahoe.data_file_path(number)
        data_file(path, cells)
        nnz = sum(1 for _, counts in cells for value in counts.values() if value)
        record = {"nCells": len(cells), "nnz": nnz}
        index = root / "index" / f"{path.stem}.json"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(record))
    return root


def expected_counts(files: list = FILES) -> tuple[np.ndarray, list[str]]:
    """The dense cells-by-genes counts, genes in token order, and the cell IDs in file order."""
    column = {token: position for position, token in enumerate(sorted(TOKENS))}
    cells = [cell for cells in files for cell in cells]
    dense = np.zeros((len(cells), len(TOKENS)), dtype=np.float32)
    for row, (_, counts) in enumerate(cells):
        for token, value in counts.items():
            dense[row, column[token]] = value
    return dense, [cell for cell, _ in cells]


def read_shards(destination: Path) -> dict:
    rows, ids = [], []
    dtypes = set()
    for path in sorted(destination.glob("*.h5ad")):
        with h5py.File(path, "r") as h5:
            x = h5["X"]
            n_cells, n_genes = (int(value) for value in x.attrs["shape"])
            data, indices, indptr = x["data"][:], x["indices"][:], x["indptr"][:]
            dtypes.add((data.dtype, indices.dtype, indptr.dtype))
            for cell in range(n_cells):
                row = np.zeros(n_genes, dtype=np.float32)
                row[indices[indptr[cell] : indptr[cell + 1]]] = data[
                    indptr[cell] : indptr[cell + 1]
                ]
                rows.append(row)
            ids += [value.decode() for value in h5["obs/_index"][:]]
            genes = [value.decode() for value in h5["var/_index"][:]]
            symbols = [value.decode() for value in h5["var/feature_name"][:]]
            encoding = x.attrs["encoding-type"]
    return {
        "dense": np.vstack(rows),
        "ids": ids,
        "genes": genes,
        "symbols": symbols,
        "dtypes": dtypes,
        "encoding": encoding,
    }


class SyntheticShardTest(unittest.TestCase):
    def test_shards_hold_the_hand_built_counts_in_file_order(self):
        with tempfile.TemporaryDirectory() as root:
            source = staged(Path(root) / "source")
            summary = tahoe_to_h5ad.convert(source, Path(root) / "shards", 2)
            shards = read_shards(Path(root) / "shards")
        dense, ids = expected_counts()
        np.testing.assert_array_equal(shards["dense"], dense)
        self.assertEqual(shards["ids"], ids)
        order = np.argsort(TOKENS)
        self.assertEqual(shards["genes"], [f"ENSG{TOKENS[i]:011d}" for i in order])
        self.assertEqual(shards["symbols"], [SYMBOLS[i] for i in order])
        # The layout of the benchmark samples that BPCells read at the six atlases.
        self.assertEqual(
            shards["dtypes"], {(np.dtype(np.float32), np.dtype(np.int64), np.dtype(np.int64))}
        )
        self.assertEqual(shards["encoding"], "csr_matrix")
        self.assertEqual(
            summary | {"seconds": 0},
            {"files": 3, "shards": 2, "cells": 9, "values": 18, "seconds": 0},
        )

    def test_each_data_file_is_deleted_and_the_shards_need_no_uncompressed_copy(self):
        with tempfile.TemporaryDirectory() as root:
            source = staged(Path(root) / "source")
            tahoe_to_h5ad.convert(source, Path(root) / "shards", 2)
            self.assertEqual(list(source.glob("data/*.parquet")), [])
            self.assertEqual(len(list(source.glob("index/*.json"))), 3)
            shards = sorted((Path(root) / "shards").iterdir())
            self.assertEqual([path.name for path in shards], ["shard_000.h5ad", "shard_001.h5ad"])
            # Uncompressed shards are linked, not copied, by the R workflow's uncompressInput.
            self.assertFalse(any(uncompress_h5ad.compressed(str(path)) for path in shards))

    def test_more_shards_than_files_gives_one_shard_per_file(self):
        with tempfile.TemporaryDirectory() as root:
            source = staged(Path(root) / "source")
            summary = tahoe_to_h5ad.convert(source, Path(root) / "shards", 16)
            self.assertEqual(summary["shards"], 3)
            np.testing.assert_array_equal(
                read_shards(Path(root) / "shards")["dense"], expected_counts()[0]
            )

    def test_row_ends_past_the_int32_range_stay_exact(self):
        # A full-size shard passes 2^31 values after about 70 files, while scipy keeps each file's
        # row ends in int32; adding the shard's offset to them in int32 overflowed on the full atlas.
        file_indptr = np.array([0, 3, 7, 2_000_000_000], dtype=np.int32)
        ends = tahoe_to_h5ad.shard_row_ends(file_indptr, 2**31 + 5)
        self.assertEqual(ends.dtype, np.int64)
        self.assertEqual(ends.tolist(), [2**31 + 8, 2**31 + 12, 2**31 + 5 + 2_000_000_000])


class RefusedInputTest(unittest.TestCase):
    def test_a_file_that_disagrees_with_its_record_stops_and_is_kept(self):
        with tempfile.TemporaryDirectory() as root:
            source = staged(Path(root) / "source")
            index = source / "index" / "train-00001-of-03388.json"
            index.write_text(json.dumps({"nCells": 2, "nnz": 7}))
            with self.assertRaisesRegex(ValueError, "staging record counts 2 and 7"):
                tahoe_to_h5ad.convert(source, Path(root) / "shards", 1)
            self.assertTrue((source / tahoe.data_file_path(1)).exists())
            self.assertTrue((source / tahoe.data_file_path(2)).exists())
            self.assertEqual(list((Path(root) / "shards").glob("*.h5ad")), [])

    def test_the_files_must_be_the_first_of_the_atlas(self):
        with tempfile.TemporaryDirectory() as root:
            source = staged(Path(root) / "source")
            (source / tahoe.data_file_path(0)).unlink()
            with self.assertRaisesRegex(ValueError, "train-00000-of-03388"):
                tahoe_to_h5ad.convert(source, Path(root) / "shards", 2)

    def test_a_file_without_its_staging_record_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            source = staged(Path(root) / "source")
            (source / "index" / "train-00002-of-03388.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "train-00002-of-03388.parquet"):
                tahoe_to_h5ad.convert(source, Path(root) / "shards", 2)
            self.assertEqual(len(list(source.glob("data/*.parquet"))), 3)

    def test_a_repeated_gene_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            source = staged(Path(root) / "source")
            path = source / tahoe.data_file_path(1)
            table = pq.read_table(path)
            genes = table.column("genes").to_pylist()
            genes[0] = [tahoe.MARKER_TOKEN, 3, 3]
            pq.write_table(
                table.set_column(0, "genes", pa.array(genes, pa.list_(pa.int64()))), path
            )
            with self.assertRaisesRegex(ValueError, "repeat or are unsorted"):
                tahoe_to_h5ad.convert(source, Path(root) / "shards", 1)

    def test_a_folder_without_data_files_or_shards_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            gene_table(source / tahoe.GENE_METADATA_PATH)
            with self.assertRaisesRegex(ValueError, "holds no Parquet files"):
                tahoe_to_h5ad.convert(source, Path(root) / "shards", 2)
            with self.assertRaisesRegex(ValueError, "at least 1"):
                tahoe_to_h5ad.convert(staged(source), Path(root) / "shards", 0)


HF_DATASET = f"https://huggingface.co/datasets/{tahoe.HF_REPOSITORY}/resolve/{tahoe.HF_REVISION}"
REAL_ROWS = 2000


def real_rows(target: Path) -> None:
    """Write the gene table and the first 2,000 rows of Tahoe-100M's first data file, read over
    HTTP ranges (two 1,000-row row groups and the footer, about 7 MB) at the pinned revision."""
    import fsspec

    gene_path = target / tahoe.GENE_METADATA_PATH
    gene_path.parent.mkdir(parents=True, exist_ok=True)
    with fsspec.open(f"{HF_DATASET}/{tahoe.GENE_METADATA_PATH}", "rb") as handle:
        gene_path.write_bytes(handle.read())
    with fsspec.open(f"{HF_DATASET}/{tahoe.data_file_path(0)}", "rb") as handle:
        parquet = pq.ParquetFile(handle)
        groups, rows = [], 0
        while rows < REAL_ROWS:
            groups.append(len(groups))
            rows += parquet.metadata.row_group(groups[-1]).num_rows
        columns = ["genes", "expressions", tahoe.CELL_ID_COLUMN, "cell_line_id", "sample"]
        table = parquet.read_row_groups(groups, columns=columns).slice(0, REAL_ROWS)
    path = target / tahoe.data_file_path(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


class RealFileTest(unittest.TestCase):
    """The first 2,000 cells of Tahoe-100M (CC0 1.0), fetched from Hugging Face; skipped offline."""

    def test_shard_equals_tahoe_read_file_counts_and_the_raw_lists(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            try:
                real_rows(source)
            except Exception as error:  # noqa: BLE001 - any failure to reach the hub skips
                self.skipTest(f"Tahoe-100M on Hugging Face is unreachable: {error}")
            path = source / tahoe.data_file_path(0)
            genes = tahoe.load_gene_table(source / tahoe.GENE_METADATA_PATH)
            expected = tahoe.read_file_counts(path, genes)
            raw = pq.read_table(path)
            values = raw.column("expressions").combine_chunks()
            lengths = pc.list_value_length(values).to_numpy()
            flat = pc.list_flatten(values).to_numpy()
            starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
            nonzero = np.add.reduceat((flat != 0).astype(np.int64), starts)
            marker_nonzero = (flat[starts] != 0).astype(np.int64)
            row_nnz = nonzero - marker_nonzero
            row_sums = np.add.reduceat(flat.astype(np.float64), starts) - flat[starts]
            record = {"nCells": REAL_ROWS, "nnz": int(row_nnz.sum())}
            (source / "index").mkdir()
            (source / "index" / f"{path.stem}.json").write_text(json.dumps(record))
            tahoe_to_h5ad.convert(source, Path(root) / "shards", 4)
            with h5py.File(Path(root) / "shards" / "shard_000.h5ad", "r") as h5:
                data, indices, indptr = (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:])
                ids = [value.decode() for value in h5["obs/_index"][:]]
                shape = tuple(int(value) for value in h5["X"].attrs["shape"])
        self.assertEqual(shape, (REAL_ROWS, genes.nFeatures))
        np.testing.assert_array_equal(data, expected.data.astype(np.float32))
        np.testing.assert_array_equal(indices, expected.indices)
        np.testing.assert_array_equal(indptr, expected.indptr)
        # Independent of the reader: per-cell values and totals from the raw lists, marker dropped.
        np.testing.assert_array_equal(np.diff(indptr), row_nnz)
        rows = np.repeat(np.arange(REAL_ROWS), np.diff(indptr))
        totals = np.bincount(rows, weights=data.astype(np.float64), minlength=REAL_ROWS)
        np.testing.assert_allclose(totals, row_sums)
        self.assertEqual(ids, raw.column(tahoe.CELL_ID_COLUMN).to_pylist())


if __name__ == "__main__":
    unittest.main()
