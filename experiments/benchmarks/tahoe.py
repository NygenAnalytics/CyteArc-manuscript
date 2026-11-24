import hashlib
import json
import re
import shutil
import time
import urllib.request
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cytearc.readers.cellranger import CrReader
from cytearc.utils.arrays import max_window_nnz
from cytearc.utils.count_values import CountValueRange

from experiments.runtime.object_io import (
    download_file,
    get_json,
    object_exists,
    object_metadata,
    put_json_if_absent,
    upload_file,
)

HF_REPOSITORY = "tahoebio/Tahoe-100M"
HF_REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
DATA_FILE_COUNT = 3388
CELL_COUNT = 95624334
GENE_METADATA_PATH = "metadata/gene_metadata.parquet"
CELL_ID_COLUMN = "BARCODE_SUB_LIB_ID"
CELL_COLUMNS = ("cell_line_id", "sample")
MARKER_TOKEN = 1
_HF_API = "https://huggingface.co/api/datasets"
_HF_RESOLVE = "https://huggingface.co/datasets"
_REVISION_PATTERN = re.compile("^[0-9a-f]{40}$")
_NEXT_LINK = re.compile('<([^>]+)>;\\s*rel="next"')
_DOWNLOAD_ATTEMPTS = 5
_HASH_BLOCK_BYTES = 8 * 1024 * 1024
_LOAD_BYTES_PER_VALUE = 32


def _local_path(uri: str) -> Path | None:
    """Return the local path of a dataset or result location, or None for an R2 object."""
    return None if uri.startswith("s3://") else Path(uri.removeprefix("file://"))


def _exists(uri: str) -> bool:
    path = _local_path(uri)
    return object_exists(uri) if path is None else path.exists()


def _read_json(uri: str) -> dict[str, Any]:
    path = _local_path(uri)
    return get_json(uri) if path is None else json.loads(path.read_text())


def _write_json(uri: str, value: dict[str, Any], *, createOnly: bool = False) -> bool:
    """Write a JSON record; with ``createOnly``, keep an existing one and return False."""
    from experiments.runtime.object_io import put_json

    path = _local_path(uri)
    if path is None:
        if createOnly:
            return put_json_if_absent(uri, value)
        put_json(uri, value)
        return True
    if createOnly and path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))
    return True


def _stored_bytes(uri: str) -> int | None:
    path = _local_path(uri)
    if path is None:
        metadata = object_metadata(uri)
        return None if metadata is None else metadata.fileBytes
    return path.stat().st_size if path.exists() else None


def _put_file(source: Path, uri: str, *, createOnly: bool = False) -> None:
    path = _local_path(uri)
    if path is None:
        upload_file(source, uri, createOnly=createOnly)
        return
    if createOnly and path.exists():
        raise FileExistsError(f"Refusing to replace existing file {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)


def _fetch(uri: str, destination: Path, *, maxWorkers: int | None = None) -> None:
    path = _local_path(uri)
    if path is None:
        download_file(uri, destination, maxWorkers=maxWorkers)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)


def require_revision(revision: str) -> str:
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("A Tahoe revision must be a 40-character commit hash")
    return revision


def stage(payload: dict) -> dict:
    from experiments.benchmarks import WORK

    revision = require_revision(payload.get("revision", HF_REVISION))
    locations = tahoe_locations(payload["datasetPrefixUri"], revision)
    manifest = fetch_manifest(revision)
    _write_json(locations.manifestUri(), manifest, createOnly=True)
    files = {item["path"]: item for item in manifest["files"]}
    paths = selected_data_files(payload.get("fileCount"))
    selected = paths[int(payload.get("start", 0)) : int(payload.get("stop", len(paths)))]
    return stage_files(
        [files[path] for path in selected],
        revision=revision,
        locations=locations,
        geneEntry=files[GENE_METADATA_PATH],
        workDir=Path(payload.get("workDir", WORK / "tahoe")),
    )


def convert(payload: dict) -> dict:
    from cytearc import CrToZarr
    from cytearc.storage.count_matrix import CountMatrixPolicy

    from experiments.benchmarks import WORK
    from experiments.runtime.common import provenance
    from experiments.runtime.object_io import put_json, storage_options

    revision = require_revision(payload.get("revision", HF_REVISION))
    locations = tahoe_locations(payload["datasetPrefixUri"], revision)
    source, download = download_tahoe_input(
        locations,
        selected_data_files(payload.get("fileCount")),
        Path(payload.get("workDir", WORK / "tahoe")),
        prefetchFiles=int(payload.get("prefetchFiles", 8)),
        maxBytes=int(payload["maxDownloadBytes"]),
        workers=int(payload["workers"]),
    )
    genes = load_gene_table(source.genePath)
    index = load_cell_index(source.indexPaths, source.records)
    reader = TahoeParquetReader(source.dataPaths, genes, index, prefetchFiles=source.prefetchFiles)
    policy = CountMatrixPolicy(**payload["countMatrix"]) if "countMatrix" in payload else None
    started = time.perf_counter()
    writer = CrToZarr(
        reader,
        payload["storeUri"],
        storage_options=storage_options(payload["storeUri"]),
        mem_budget=payload["memoryBudget"],
        nthreads=int(payload["workers"]),
        policy=policy,
    )
    writer.dump()
    result = {
        "status": "ok",
        "cells": index.nCells,
        "seconds": time.perf_counter() - started,
        "download": download,
        "revision": revision,
        "provenance": provenance(),
    }

    if payload.get("recordUri"):
        put_json(payload["recordUri"], result)
    return result


def data_file_path(index: int) -> str:
    if not 0 <= index < DATA_FILE_COUNT:
        raise ValueError(f"Data file index {index} is outside 0..{DATA_FILE_COUNT - 1}")
    return f"data/train-{index:05d}-of-{DATA_FILE_COUNT:05d}.parquet"


def selected_data_files(fileCount: int | None) -> tuple[str, ...]:
    count = DATA_FILE_COUNT if fileCount is None else int(fileCount)
    if not 1 <= count <= DATA_FILE_COUNT:
        raise ValueError(f"fileCount must lie in 1..{DATA_FILE_COUNT}")
    return tuple((data_file_path(index) for index in range(count)))


@dataclass(frozen=True, slots=True)
class TahoeLocations:
    prefixUri: str

    def stagedUri(self, path: str) -> str:
        return f"{self.prefixUri}/{path}"

    def manifestUri(self) -> str:
        return f"{self.prefixUri}/manifest.json"

    def _indexStem(self, path: str) -> str:
        return f"{self.prefixUri}/index/{Path(path).stem}"

    def cellIndexUri(self, path: str) -> str:
        return f"{self._indexStem(path)}.cells.parquet"

    def fileRecordUri(self, path: str) -> str:
        return f"{self._indexStem(path)}.json"


def tahoe_locations(datasetPrefixUri: str, revision: str) -> TahoeLocations:
    prefix = f"{datasetPrefixUri.rstrip('/')}/tahoe-100m/{require_revision(revision)}"
    return TahoeLocations(prefixUri=prefix)


def _next_link(header: str | None) -> str | None:
    if not header:
        return None
    match = _NEXT_LINK.search(header)
    return None if match is None else match.group(1)


def _list_tree(revision: str, path: str) -> list[dict[str, Any]]:
    url: str | None = f"{_HF_API}/{HF_REPOSITORY}/tree/{revision}/{path}"
    entries: list[dict[str, Any]] = []
    while url is not None:
        with urllib.request.urlopen(url, timeout=120) as response:
            entries.extend(json.load(response))
            url = _next_link(response.headers.get("Link"))
    return entries


def manifest_from_entries(revision: str, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    listed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        lfs = entry.get("lfs")
        if entry.get("type") != "file" or not isinstance(lfs, dict):
            continue
        if not lfs.get("oid"):
            continue
        listed[str(entry["path"])] = {"size": int(entry["size"]), "sha256": str(lfs["oid"])}
    expected = [*selected_data_files(None), GENE_METADATA_PATH]
    missing = [path for path in expected if path not in listed]
    if missing:
        raise ValueError(
            f"Revision {revision} lacks {len(missing)} expected LFS files, first {missing[0]}"
        )
    return {
        "repository": HF_REPOSITORY,
        "revision": revision,
        "files": [{"path": path, **listed[path]} for path in expected],
    }


def fetch_manifest(revision: str) -> dict[str, Any]:
    require_revision(revision)
    entries = [*_list_tree(revision, "data"), *_list_tree(revision, "metadata")]
    return manifest_from_entries(revision, entries)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file_identity(path: Path, entry: dict[str, Any]) -> None:
    size = path.stat().st_size
    if size != int(entry["size"]):
        raise RuntimeError(f"{entry['path']} has {size} bytes, expected {entry['size']}")
    digest = sha256_path(path)
    if digest != entry["sha256"]:
        raise RuntimeError(f"{entry['path']} has SHA-256 {digest}, expected {entry['sha256']}")


def download_hub_file(entry: dict[str, Any], revision: str, destination: Path) -> None:
    url = f"{_HF_RESOLVE}/{HF_REPOSITORY}/resolve/{revision}/{entry['path']}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f".{destination.name}.part")
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=600) as response:
                with part.open("wb") as handle:
                    for block in iter(lambda: response.read(_HASH_BLOCK_BYTES), b""):
                        handle.write(block)
            require_file_identity(part, entry)
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(min(60.0, 5.0 * attempt))
            continue
        part.replace(destination)
        return


@dataclass(frozen=True, slots=True)
class GeneTable:
    featureIds: np.ndarray
    featureNames: np.ndarray
    columnOfToken: np.ndarray

    @property
    def nFeatures(self) -> int:
        return int(self.featureIds.size)


def load_gene_table(path: Path) -> GeneTable:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["token_id", "ensembl_id", "gene_symbol"])
    table = table.sort_by("token_id")
    tokens = table.column("token_id").to_numpy()
    if np.unique(tokens).size != tokens.size:
        raise ValueError("The gene table repeats a token ID")
    if tokens.size == 0 or int(tokens.min()) <= MARKER_TOKEN:
        raise ValueError("Gene tokens must follow the marker token")
    ids = np.asarray(table.column("ensembl_id").to_pylist(), dtype=str)
    names = np.asarray(table.column("gene_symbol").to_pylist(), dtype=str)
    if np.unique(ids).size != ids.size:
        raise ValueError("The gene table repeats an Ensembl ID")
    column_of_token = np.full(int(tokens.max()) + 1, -1, dtype=np.int64)
    column_of_token[tokens] = np.arange(tokens.size)
    return GeneTable(featureIds=ids, featureNames=names, columnOfToken=column_of_token)


def _file_counts(path: Path, genes: GeneTable) -> tuple[Any, int]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from scipy.sparse import csr_matrix

    table = pq.read_table(path, columns=["genes", "expressions"])
    gene_lists = table.column("genes").combine_chunks()
    value_lists = table.column("expressions").combine_chunks()
    lengths = pc.list_value_length(gene_lists).to_numpy(zero_copy_only=False)
    if not np.array_equal(
        lengths, pc.list_value_length(value_lists).to_numpy(zero_copy_only=False)
    ):
        raise ValueError(f"{path.name} has genes and expressions of different lengths")
    if lengths.size and int(lengths.min()) < 1:
        raise ValueError(f"{path.name} has a row without its marker entry")
    tokens = pc.list_flatten(gene_lists).to_numpy()
    values = pc.list_flatten(value_lists).to_numpy()
    del table, gene_lists, value_lists
    starts = np.zeros(lengths.size, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    if np.any(tokens[starts] != MARKER_TOKEN):
        raise ValueError(f"{path.name} has a row that does not start with the marker")
    keep = np.ones(tokens.size, dtype=bool)
    keep[starts] = False
    if values.size:
        counts = values[keep]
        if float(counts.min()) < 0 or not np.array_equal(counts, np.floor(counts)):
            raise ValueError(f"{path.name} has a count that is not a non-negative integer")
        del counts
    zeros = keep & (values == 0)
    stored = keep & ~zeros
    del keep
    row_zeros = np.add.reduceat(zeros.view(np.uint8), starts, dtype=np.int64)
    explicit_zeros = int(row_zeros.sum())
    row_nnz = lengths - 1 - row_zeros
    del zeros
    tokens = tokens[stored]
    values = values[stored]
    del stored
    if tokens.size and (int(tokens.min()) < 0 or int(tokens.max()) >= genes.columnOfToken.size):
        raise ValueError(f"{path.name} has a token outside the gene table")
    columns = genes.columnOfToken[tokens]
    del tokens
    if np.any(columns < 0):
        raise ValueError(f"{path.name} has a token outside the gene table")
    indptr = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=indptr[1:])
    matrix = csr_matrix(
        (values.astype(np.uint32), columns.astype(np.int32), indptr),
        shape=(lengths.size, genes.nFeatures),
    )
    if not matrix.has_canonical_format:
        raise ValueError(f"{path.name} has a row whose genes repeat or are unsorted")
    return (matrix, explicit_zeros)


def read_file_counts(path: Path, genes: GeneTable) -> Any:
    matrix, _explicit_zeros = _file_counts(path, genes)
    return matrix


def index_file(path: Path, genes: GeneTable) -> tuple[Any, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    matrix, explicit_zeros = _file_counts(path, genes)
    metadata = pq.read_table(path, columns=[CELL_ID_COLUMN, *CELL_COLUMNS])
    row_nnz = np.diff(matrix.indptr).astype(np.uint32)
    table = pa.table(
        {
            "rowNnz": row_nnz,
            CELL_ID_COLUMN: metadata.column(CELL_ID_COLUMN),
            **{name: metadata.column(name) for name in CELL_COLUMNS},
        }
    )
    summary = {
        "nCells": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "maxCount": int(matrix.data.max()) if matrix.nnz else 0,
        "explicitZeros": explicit_zeros,
    }
    return (table, summary)


@dataclass(frozen=True, slots=True)
class CellIndex:
    cumulativeNnz: np.ndarray
    fileRows: np.ndarray
    cellIds: np.ndarray
    cellColumns: dict[str, np.ndarray]
    maxCount: int

    @property
    def nCells(self) -> int:
        return int(self.cumulativeNnz.size - 1)

    def nbytes(self) -> int:
        return int(
            self.cumulativeNnz.nbytes
            + self.fileRows.nbytes
            + self.cellIds.nbytes
            + sum((values.nbytes for values in self.cellColumns.values()))
        )


def load_cell_index(indexPaths: Sequence[Path], records: Sequence[dict[str, Any]]) -> CellIndex:
    import pyarrow.parquet as pq

    if len(indexPaths) != len(records):
        raise ValueError("Every index table needs its record")
    row_nnz: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    columns: dict[str, list[np.ndarray]] = {name: [] for name in CELL_COLUMNS}
    file_rows = np.zeros(len(records), dtype=np.int64)
    for position, (path, record) in enumerate(zip(indexPaths, records, strict=True)):
        table = pq.read_table(path)
        nnz = table.column("rowNnz").to_numpy()
        if nnz.size != int(record["nCells"]) or int(nnz.sum()) != int(record["nnz"]):
            raise ValueError(f"The index table {path.name} disagrees with its record")
        file_rows[position] = nnz.size
        row_nnz.append(nnz)
        ids.append(np.asarray(table.column(CELL_ID_COLUMN).to_pylist(), dtype=str))
        for name in CELL_COLUMNS:
            columns[name].append(np.asarray(table.column(name).to_pylist(), dtype=str))
    counts = np.concatenate(row_nnz)
    cumulative = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=cumulative[1:])
    cell_ids = np.concatenate(ids)
    if np.unique(cell_ids).size != cell_ids.size:
        raise ValueError("The selected files repeat a cell ID")
    return CellIndex(
        cumulativeNnz=cumulative,
        fileRows=file_rows,
        cellIds=cell_ids,
        cellColumns={name: np.concatenate(values) for name, values in columns.items()},
        maxCount=max((int(record["maxCount"]) for record in records)),
    )


@dataclass(frozen=True, slots=True)
class TahoeInput:
    dataPaths: tuple[Path, ...]
    genePath: Path
    indexPaths: tuple[Path, ...]
    records: tuple[dict[str, Any], ...]
    prefetchFiles: int

    def __post_init__(self) -> None:
        if not self.dataPaths:
            raise ValueError("Tahoe conversion needs at least one data file")
        if not len(self.dataPaths) == len(self.indexPaths) == len(self.records):
            raise ValueError("Every data file needs its index table and record")


class TahoeParquetReader(CrReader):
    def __init__(
        self,
        dataPaths: Sequence[Path],
        genes: GeneTable,
        index: CellIndex,
        *,
        prefetchFiles: int = 8,
    ) -> None:
        if len(dataPaths) != index.fileRows.size:
            raise ValueError("Every data file needs its index")
        if prefetchFiles <= 0:
            raise ValueError("prefetchFiles must be positive")
        self._dataPaths = tuple((Path(path) for path in dataPaths))
        self._genes = genes
        self._index = index
        self._prefetchFiles = int(prefetchFiles)
        file_starts = np.zeros(index.fileRows.size + 1, dtype=np.int64)
        np.cumsum(index.fileRows, out=file_starts[1:])
        file_nnz = index.cumulativeNnz[file_starts[1:]] - index.cumulativeNnz[file_starts[:-1]]
        self._maxFileNnz = int(file_nnz.max(initial=0))
        super().__init__({"feature_types": None})

    def _handle_version(self) -> dict[str, Any]:
        return {"feature_types": None}

    def _read_dataset(self, key: str) -> list[Any]:
        if key == "feature_ids":
            return list(self._genes.featureIds)
        if key == "feature_names":
            return list(self._genes.featureNames)
        raise KeyError(key)

    def cell_names(self) -> Any:
        return self._index.cellIds

    def get_cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        for name in CELL_COLUMNS:
            yield (name, self._index.cellColumns[name])

    @property
    def matrix_dtype(self) -> np.dtype[Any]:
        return np.dtype(np.uint32)

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        groups = 1 if featureGroups is None else int(np.max(featureGroups)) + 1
        return [CountValueRange(integral=True, maximum=self._index.maxCount) for _ in range(groups)]

    def max_window_nnz(self, window_rows: int) -> int:
        return max_window_nnz(self._index.cumulativeNnz, window_rows)

    def producer_staging_bytes(self, batch_size: int, lines_in_mem: int) -> int:
        loads = (self._prefetchFiles + 1) * self._maxFileNnz * _LOAD_BYTES_PER_VALUE
        return int(loads + 2 * self.max_window_nnz(batch_size) * 16)

    def _sparse_import_resident_bytes(self) -> int:
        return self._index.nbytes()

    def _file_matrices(self) -> Iterator[Any]:
        with ThreadPoolExecutor(max_workers=self._prefetchFiles) as pool:
            pending: list[Future[Any]] = []
            paths = iter(self._dataPaths)
            for path in paths:
                pending.append(pool.submit(read_file_counts, path, self._genes))
                if len(pending) >= self._prefetchFiles:
                    break
            done = 0
            while pending:
                matrix = pending.pop(0).result()
                if matrix.shape[0] != int(self._index.fileRows[done]):
                    raise ValueError(
                        f"{self._dataPaths[done].name} holds {matrix.shape[0]} cells, but its index records {int(self._index.fileRows[done])}"
                    )
                done += 1
                following = next(paths, None)
                if following is not None:
                    pending.append(pool.submit(read_file_counts, following, self._genes))
                yield matrix

    def consume(self, batch_size: int, lines_in_mem: int) -> Iterator[Any]:
        from scipy.sparse import vstack

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cumulative = self._index.cumulativeNnz

        def checked(batch: Any, start: int) -> Any:
            stop = start + int(batch.shape[0])
            if stop > self._index.nCells or batch.nnz != int(cumulative[stop] - cumulative[start]):
                raise ValueError("A batch disagrees with the cell index")
            return batch.tocoo()

        held: list[Any] = []
        held_rows = 0
        yielded = 0
        for matrix in self._file_matrices():
            held.append(matrix)
            held_rows += int(matrix.shape[0])
            while held_rows >= batch_size:
                rows = held[0] if len(held) == 1 else vstack(held, format="csr")
                rest = rows[batch_size:]
                held = [rest] if rest.shape[0] else []
                held_rows = int(rest.shape[0])
                yield checked(rows[:batch_size], yielded)
                yielded += batch_size
        if held_rows:
            rows = held[0] if len(held) == 1 else vstack(held, format="csr")
            yield checked(rows, yielded)
            yielded += held_rows
        if yielded != self._index.nCells:
            raise ValueError(
                f"The files hold {yielded} cells, but the index records {self._index.nCells}"
            )


def _upload_once(local: Path, uri: str, entry: dict[str, Any]) -> None:
    existing = _stored_bytes(uri)
    if existing is None:
        try:
            _put_file(local, uri, createOnly=True)
            return
        except FileExistsError:
            existing = _stored_bytes(uri)
    if existing != int(entry["size"]):
        raise RuntimeError(f"{uri} holds {existing} bytes, expected {entry['size']}")


def stage_files(
    entries: Sequence[dict[str, Any]],
    *,
    revision: str,
    locations: TahoeLocations,
    geneEntry: dict[str, Any],
    workDir: Path,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    workDir.mkdir(parents=True, exist_ok=True)
    gene_path = workDir / Path(GENE_METADATA_PATH).name
    download_hub_file(geneEntry, revision, gene_path)
    _upload_once(gene_path, locations.stagedUri(GENE_METADATA_PATH), geneEntry)
    genes = load_gene_table(gene_path)
    staged = 0
    skipped = 0
    for entry in entries:
        path = str(entry["path"])
        if _exists(locations.fileRecordUri(path)):
            skipped += 1
            continue
        started = time.perf_counter()
        local = workDir / Path(path).name
        download_hub_file(entry, revision, local)
        _upload_once(local, locations.stagedUri(path), entry)
        table, summary = index_file(local, genes)
        index_path = workDir / f"{Path(path).stem}.cells.parquet"
        pq.write_table(table, index_path)
        _put_file(index_path, locations.cellIndexUri(path))
        record = {
            "path": path,
            "revision": revision,
            "size": int(entry["size"]),
            "sha256": str(entry["sha256"]),
            "indexSha256": sha256_path(index_path),
            **summary,
            "seconds": time.perf_counter() - started,
        }
        _write_json(locations.fileRecordUri(path), record, createOnly=True)
        local.unlink()
        index_path.unlink()
        staged += 1
        print(f"[tahoe-stage] {path} staged cells={summary['nCells']}", flush=True)
    return {
        "staged": staged,
        "skipped": skipped,
        "first": entries[0]["path"] if entries else None,
        "last": entries[-1]["path"] if entries else None,
    }


def load_file_records(
    locations: TahoeLocations, paths: Sequence[str], *, workers: int = 32
) -> list[dict[str, Any] | None]:

    def load(path: str) -> dict[str, Any] | None:
        try:
            return _read_json(locations.fileRecordUri(path))
        except FileNotFoundError:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(load, paths))


def require_records(locations: TahoeLocations, paths: Sequence[str]) -> list[dict[str, Any]]:
    records = load_file_records(locations, paths)
    missing = [path for path, record in zip(paths, records, strict=True) if not record]
    if missing:
        raise ValueError(
            f"{len(missing)} selected files are not staged, first {missing[0]}; run tahoe-stage"
        )
    return [record for record in records if record is not None]


def download_tahoe_input(
    locations: TahoeLocations,
    paths: Sequence[str],
    workDir: Path,
    *,
    prefetchFiles: int,
    maxBytes: int,
    workers: int = 16,
) -> tuple[TahoeInput, dict[str, Any]]:
    records = require_records(locations, paths)
    data_bytes = sum((int(record["size"]) for record in records))
    if data_bytes > maxBytes:
        raise ValueError(
            f"The selected files hold {data_bytes} bytes, more than the {maxBytes} bytes of local disk set aside for them"
        )
    manifest = _read_json(locations.manifestUri())
    (gene_entry,) = [item for item in manifest["files"] if item["path"] == GENE_METADATA_PATH]
    started = time.perf_counter()
    gene_path = workDir / Path(GENE_METADATA_PATH).name
    _fetch(locations.stagedUri(GENE_METADATA_PATH), gene_path)
    require_file_identity(gene_path, gene_entry)

    def fetch(path: str, record: dict[str, Any]) -> tuple[Path, Path]:
        data_path = workDir / "data" / Path(path).name
        _fetch(locations.stagedUri(path), data_path, maxWorkers=4)
        require_file_identity(data_path, record)
        index_path = workDir / "index" / f"{Path(path).stem}.cells.parquet"
        _fetch(locations.cellIndexUri(path), index_path, maxWorkers=1)
        if sha256_path(index_path) != record["indexSha256"]:
            raise RuntimeError(f"The index table of {path} differs from its record")
        return (data_path, index_path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = list(pool.map(fetch, paths, records))
    source = TahoeInput(
        dataPaths=tuple((data for data, _ in fetched)),
        genePath=gene_path,
        indexPaths=tuple((index for _, index in fetched)),
        records=tuple(records),
        prefetchFiles=prefetchFiles,
    )
    return (
        source,
        {
            "downloadSeconds": time.perf_counter() - started,
            "dataFiles": len(paths),
            "dataBytes": data_bytes,
            "prefixUri": locations.prefixUri,
        },
    )


EXPORT_STAGES = ("filterCells", "markHvgs", "runUmap", "runLeiden", "findMarkers")
RASTER_BINS = 2048


def cluster_cell_line_summary(labels: np.ndarray, lines: np.ndarray) -> dict[str, Any]:
    """Return each Leiden cluster's size, cell lines and commonest line, and their summary."""
    if labels.shape != lines.shape:
        raise ValueError("Every clustered cell needs its cell line")
    clusters: list[dict[str, Any]] = []
    for label in np.unique(labels):
        members, counts = np.unique(lines[labels == label], return_counts=True)
        top = int(np.argmax(counts))
        size = int(counts.sum())
        clusters.append(
            {
                "cluster": int(label),
                "cells": size,
                "cellLines": int(members.size),
                "commonestCellLine": str(members[top]),
                "commonestCellLineFraction": int(counts[top]) / size,
            }
        )
    fractions = np.array([item["commonestCellLineFraction"] for item in clusters])
    return {
        "leidenClusters": len(clusters),
        "cellLines": int(np.unique(lines).size),
        "medianCommonestCellLineFraction": float(np.median(fractions)),
        "cellsInClustersWithOneLineAbove90Percent": int(
            sum(item["cells"] for item in clusters if item["commonestCellLineFraction"] > 0.9)
        ),
        "clusters": clusters,
    }


def umap_raster(
    coordinates: np.ndarray, lines: np.ndarray, nLines: int, bins: int = RASTER_BINS
) -> dict[str, Any]:
    """Count cells per UMAP pixel and find each pixel's commonest cell line."""
    low = coordinates.min(axis=0).astype(np.float64)
    high = coordinates.max(axis=0).astype(np.float64)
    span = np.maximum(high - low, np.finfo(np.float32).eps)
    cells = np.minimum(((coordinates - low) / span * bins).astype(np.int64), bins - 1)
    pixel = cells[:, 1] * bins + cells[:, 0]
    by_line = np.bincount(pixel * nLines + lines, minlength=bins * bins * nLines).reshape(
        bins * bins, nLines
    )
    counts = by_line.sum(axis=1)
    dominant = np.where(counts > 0, by_line.argmax(axis=1), -1)
    fraction = np.divide(by_line.max(axis=1), counts, out=np.zeros(counts.size), where=counts > 0)
    return {
        "cells": counts.reshape(bins, bins).astype(np.uint32),
        "commonestCellLine": dominant.reshape(bins, bins).astype(np.int16),
        "commonestCellLineFraction": fraction.reshape(bins, bins).astype(np.float32),
        "extent": np.array([low[0], high[0], low[1], high[1]]),
    }


def _stage_records(config: Any, nRows: int, stages: Sequence[str]) -> dict[str, dict]:
    records = {}
    for stage in stages:
        record = _read_json(config.resultUri(nRows, stage))
        if record["status"] != "ok":
            raise ValueError(f"{config.runTag} {stage} did not complete")
        records[stage] = record
    return records


def _open_store(record: dict, workers: int, memoryBudget: int) -> Any:
    from cytearc import DataStore

    from experiments.runtime.object_io import storage_options

    uri = record["storeUri"]
    return DataStore(
        uri,
        zarr_mode="r",
        storage_options=storage_options(uri),
        nthreads=workers,
        mem_budget=memoryBudget,
    )


def _artifact(store: Any, record: dict) -> Any:
    from cytearc.storage import ArtifactRef

    return store.artifacts.load(ArtifactRef.from_dict(record["details"]["artifact"]))


def marker_table(markers: Any) -> Any:
    """Return every group's marker statistics as one table, a row per group and feature."""
    import pyarrow as pa

    columns = [str(name) for name in markers.attrs["stat_columns"]]
    feature_index = np.asarray(markers["feature_index"][:])
    feature_names = np.asarray(markers["feature_names"][:]).astype(str)
    parts = []
    for group in sorted(markers.group_keys()):
        stats = np.asarray(markers[group]["stats"][:])
        if stats.shape != (feature_index.size, len(columns)):
            raise ValueError(f"Marker group {group} has statistics of shape {stats.shape}")
        parts.append(
            pa.table(
                {
                    "group": pa.array([str(group)] * feature_index.size),
                    "featureIndex": feature_index,
                    "featureName": feature_names,
                    **{name: stats[:, position] for position, name in enumerate(columns)},
                }
            )
        )
    return pa.concat_tables(parts)


def compare_layouts(store: Any, records: dict, other: Any, otherRecords: dict) -> dict[str, Any]:
    """Check that two count layouts of one input gave the same cells, genes and markers."""
    comparison: dict[str, Any] = {}
    for stage in ("filterCells", "markHvgs"):
        left = np.asarray(_artifact(store, records[stage])["values"][:], dtype=bool)
        right = np.asarray(_artifact(other, otherRecords[stage])["values"][:], dtype=bool)
        comparison[f"{stage}Identical"] = bool(np.array_equal(left, right))
    if "findMarkers" in otherRecords:
        left_table = marker_table(_artifact(store, records["findMarkers"]))
        right_table = marker_table(_artifact(other, otherRecords["findMarkers"]))
        same_rows = left_table.select(["group", "featureIndex"]).equals(
            right_table.select(["group", "featureIndex"])
        )
        errors = {}
        if same_rows:
            for name in left_table.column_names[3:]:
                left_values = left_table.column(name).to_numpy()
                right_values = right_table.column(name).to_numpy()
                finite = np.isfinite(left_values) & np.isfinite(right_values)
                errors[name] = {
                    "finitePatternIdentical": bool(
                        np.array_equal(np.isfinite(left_values), np.isfinite(right_values))
                    ),
                    "maxAbsoluteDifference": float(
                        np.abs(left_values[finite] - right_values[finite]).max(initial=0)
                    ),
                }
        comparison["markerGroupsAndFeaturesIdentical"] = bool(same_rows)
        comparison["markerStatistics"] = errors
    return comparison


def export(payload: dict) -> dict:
    """Save the analysed Tahoe store's results for Figure 1 next to its stage records.

    Writes, under ``<result prefix>/exports/``: every retained cell's store position, UMAP
    coordinates, Leiden cluster, cell line and sample (``cells.parquet``, with the category
    names in ``categories.json``); the filter's selection over every input cell
    (``filtered_cells.npz``); cells per cluster and cell line (``cluster_cell_lines.csv``);
    every marker statistic (``markers.parquet``); and a UMAP raster with the cell count and
    commonest cell line of each pixel (``umap_raster.npz``). With ``layoutConfigPath``, the run
    on the other count layout is checked for the same selections and markers.
    """
    import csv
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from experiments.benchmarks.config import load_profiling_config

    config = load_profiling_config(payload["configPath"])
    if len(config.targetSizes) != 1:
        raise ValueError("The export reads one analysed size")
    nRows = config.targetSizes[0]
    resources = config.resourcesFor("findMarkers")
    workers = int(payload.get("workers", resources.workers))
    budget = int(payload.get("memoryBudget", resources.cytearcMemoryBudget))
    bins = int(payload.get("rasterBins", RASTER_BINS))
    records = _stage_records(config, nRows, EXPORT_STAGES)
    store = _open_store(records["filterCells"], workers, budget)
    kept = np.asarray(_artifact(store, records["filterCells"])["values"][:], dtype=bool)
    if kept.size != nRows:
        raise ValueError(f"The filter selection covers {kept.size} cells, expected {nRows}")
    labels = np.asarray(_artifact(store, records["runLeiden"])["values"][:])
    coordinates = np.asarray(_artifact(store, records["runUmap"])["values"][:], dtype=np.float32)
    if labels.shape != (int(kept.sum()),) or coordinates.shape != (labels.size, 2):
        raise ValueError("The Leiden labels and UMAP coordinates do not match the retained cells")
    categories = {}
    codes = {}
    for column, name in (("cell_line_id", "cellLines"), ("sample", "samples")):
        values, inverse = np.unique(
            np.asarray(store.cells.fetch_all(column)).astype(str), return_inverse=True
        )
        if values.size > np.iinfo(np.int16).max:
            raise ValueError(f"{column} has {values.size} categories, more than int16 holds")
        categories[name] = values.tolist()
        codes[name] = inverse.astype(np.int16)[kept]
    lines = codes["cellLines"]
    summary = cluster_cell_line_summary(labels, np.asarray(categories["cellLines"])[lines])
    raster = umap_raster(coordinates, lines, len(categories["cellLines"]), bins)
    markers = marker_table(_artifact(store, records["findMarkers"]))
    prefix = f"{config.resultPrefix(nRows)}/exports"
    files = {}
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        pq.write_table(
            pa.table(
                {
                    "cellIndex": np.flatnonzero(kept).astype(np.uint32),
                    "umap1": coordinates[:, 0],
                    "umap2": coordinates[:, 1],
                    "leiden": labels.astype(np.int32),
                    "cellLine": lines,
                    "sample": codes["samples"],
                }
            ),
            work / "cells.parquet",
        )
        (work / "categories.json").write_text(json.dumps(categories))
        np.savez_compressed(work / "filtered_cells.npz", kept=np.packbits(kept), cells=nRows)
        pair_counts = np.bincount(
            labels.astype(np.int64) * len(categories["cellLines"]) + lines,
            minlength=(int(labels.max()) + 1) * len(categories["cellLines"]),
        ).reshape(-1, len(categories["cellLines"]))
        with (work / "cluster_cell_lines.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cluster", "cellLine", "cells"])
            for cluster, line in zip(*np.nonzero(pair_counts), strict=True):
                writer.writerow(
                    [int(cluster), categories["cellLines"][line], int(pair_counts[cluster, line])]
                )
        pq.write_table(markers, work / "markers.parquet")
        np.savez_compressed(work / "umap_raster.npz", **raster)
        for path in sorted(work.iterdir()):
            uri = f"{prefix}/{path.name}"
            _put_file(path, uri)
            files[path.name] = {
                "uri": uri,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
    result: dict[str, Any] = {
        "runTag": config.runTag,
        "nRows": nRows,
        "cellsKept": int(kept.sum()),
        "rasterBins": bins,
        "markerRows": markers.num_rows,
        "files": files,
        **summary,
    }
    if payload.get("layoutConfigPath"):
        other_config = load_profiling_config(payload["layoutConfigPath"])
        other_stages = [
            stage
            for stage in ("filterCells", "markHvgs", "findMarkers")
            if stage in other_config.effectiveStages
        ]
        other_records = _stage_records(other_config, nRows, other_stages)
        other = _open_store(other_records["filterCells"], workers, budget)
        result["layoutComparison"] = {
            "runTag": other_config.runTag,
            **compare_layouts(store, records, other, other_records),
        }
    _write_json(f"{prefix}/export.json", result)
    return result
