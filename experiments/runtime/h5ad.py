"""Read and write the groups of AnnData H5AD files with h5py."""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FILES = "https://datasets.cellxgene.cziscience.com/"
_H5_DATA_CHUNK_BYTES = 4 * 1024 * 1024


def _text(values) -> np.ndarray:
    return np.asarray(
        [v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object
    )


def _column(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Group):
        categories = _text(node["categories"][:])
        return categories[node["codes"][:]]
    return _text(node[:])


def _index(group: h5py.Group) -> np.ndarray:
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _column(group, key)


def _string_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    dataset = group.create_dataset(
        name, data=values.astype(object), dtype=h5py.string_dtype("utf-8")
    )
    dataset.attrs["encoding-type"] = "string-array"
    dataset.attrs["encoding-version"] = "0.2.0"


def _dataframe(parent: h5py.File, name: str, columns: tuple[str, ...]) -> h5py.Group:
    group = parent.create_group(name)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = np.asarray(columns, dtype=h5py.string_dtype("utf-8"))
    return group


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(64 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def read_column(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Group):
        codes = node["codes"][()]
        categories = node["categories"][()].astype(str)
        return np.where(codes >= 0, categories[np.clip(codes, 0, None)], "")
    values = node[()]
    return values.astype(str) if values.dtype.kind in "SO" else values


def remote(version: str):
    import fsspec

    return fsspec.open(FILES + version + ".h5ad", "rb", block_size=4 * 1024**2).open()


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _require_dataset(h5: h5py.File, path: str) -> h5py.Dataset:
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise ValueError(f"Required HDF5 dataset is missing: {path}")
    return h5[path]


def _decode_string_array(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.dtype.kind in {"U", "S", "O"}:
        decoded = [
            item.decode() if isinstance(item, bytes | np.bytes_) else str(item)
            for item in flat.tolist()
        ]
        return np.asarray(decoded, dtype=object)
    raise ValueError(f"Unsupported string array dtype: {flat.dtype}")


def _load_string_column(h5: h5py.File, path: str, *, expectedLength: int) -> np.ndarray:
    if path not in h5:
        raise ValueError(f"Required HDF5 path is missing: {path}")
    value = h5[path]
    if isinstance(value, h5py.Dataset):
        if value.shape != (expectedLength,):
            raise ValueError(f"{path} shape is {value.shape}, expected {(expectedLength,)}")
        return _decode_string_array(np.asarray(value[:]))
    if isinstance(value, h5py.Group):
        encoding = value.attrs.get("encoding-type")
        if encoding is None or _decode_text(encoding) != "categorical":
            raise ValueError(f"{path} is a group but not a categorical string column")
        categories = _decode_string_array(np.asarray(_require_dataset(h5, f"{path}/categories")[:]))
        codes = np.asarray(_require_dataset(h5, f"{path}/codes")[:])
        if codes.shape != (expectedLength,):
            raise ValueError(f"{path}/codes shape is {codes.shape}, expected {(expectedLength,)}")
        if codes.min(initial=0) < -1 or codes.max(initial=-1) >= len(categories):
            raise ValueError(f"{path}/codes contains out-of-range category ids")
        output = np.empty(expectedLength, dtype=object)
        missing = codes < 0
        output[missing] = ""
        output[~missing] = categories[codes[~missing]]
        return output
    raise ValueError(f"{path} must be a string dataset or categorical group")


def _validate_selected_rows(sourceRows: np.ndarray | Sequence[int], nRows: int) -> np.ndarray:
    rows = np.asarray(sourceRows)
    if rows.ndim != 1:
        raise ValueError("sourceRows must be one-dimensional")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("sourceRows must use an integer dtype")
    if len(rows) == 0:
        raise ValueError("sourceRows cannot be empty")
    if int(rows.min()) < 0 or int(rows.max()) >= nRows:
        raise ValueError(f"sourceRows must be between 0 and {nRows - 1}")
    rows = rows.astype(np.int64, copy=False)
    if np.any(rows[1:] <= rows[:-1]):
        raise ValueError("sourceRows must be unique and in source order")
    return rows


def _chunk_elements(length: int, dtype: np.dtype[Any]) -> int:
    desired = max(1, _H5_DATA_CHUNK_BYTES // max(1, dtype.itemsize))
    return max(1, min(max(1, length), desired))


def _create_numeric_dataset(
    group: h5py.Group, name: str, length: int, dtype: np.dtype[Any]
) -> h5py.Dataset:
    return group.create_dataset(
        name,
        shape=(length,),
        maxshape=(None,),
        dtype=dtype,
        chunks=(_chunk_elements(length, dtype),),
        compression="gzip",
        compression_opts=4,
    )


def _create_string_dataset(group: h5py.Group, name: str, length: int) -> h5py.Dataset:
    dataset = group.create_dataset(
        name,
        shape=(length,),
        maxshape=(None,),
        dtype=h5py.string_dtype(encoding="utf-8"),
        chunks=(max(1, min(max(1, length), 65536)),),
        compression="gzip",
        compression_opts=4,
    )
    dataset.attrs["encoding-type"] = "string-array"
    dataset.attrs["encoding-version"] = "0.2.0"
    return dataset


def _create_dataframe_group(h5: h5py.File, name: str, columns: Sequence[str]) -> h5py.Group:
    group = h5.create_group(name)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = np.asarray(columns, dtype=h5py.string_dtype(encoding="utf-8"))
    return group
