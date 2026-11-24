"""Per-cell outputs of a workflow run that keeps them (the keepOutputs payload setting).

A run whose workflow sees KEEP_OUTPUTS=1 writes, after its last timed step, WORKDIR/outputs.h5:
  cellIds    every input cell's identifier, in input (H5AD row) order
  positions  the 0-based input position of each retained cell, ascending; the rows of the three
             arrays below follow it
  clusters   each retained cell's Leiden cluster, as text
  pca        each retained cell's principal components (cells x 21)
  umap       each retained cell's UMAP coordinates (cells x 2)
with attributes naming the workflow and its mode. The Python workflows write it with
write_outputs (h5py); workflow_r.R writes the same datasets with hdf5r, which stores a matrix with
its dimensions reversed (21 x cells as h5py reads it), so read_outputs orients each matrix by the
number of retained cells. experiments.benchmarks.workflows uploads the file beside the attempt record as
<record>_outputs.h5, and cluster_quality.py reads it with read_outputs.

Nothing here imports a tool: the module runs in the CyteArc, Scanpy and analysis images alike.
"""

from pathlib import Path

import numpy as np

FILE_NAME = "outputs.h5"


def _check(
    n_input: int, positions: np.ndarray, clusters: np.ndarray, pca: np.ndarray, umap: np.ndarray
) -> None:
    n = positions.size
    if positions.ndim != 1 or n == 0:
        raise ValueError("positions must be a non-empty vector")
    if positions[0] < 0 or positions[-1] >= n_input or np.any(np.diff(positions) <= 0):
        raise ValueError(f"positions must be ascending input rows below {n_input}")
    if clusters.shape != (n,):
        raise ValueError(f"{clusters.shape[0]} cluster labels for {n} retained cells")
    for name, matrix in (("pca", pca), ("umap", umap)):
        if matrix.ndim != 2 or matrix.shape[0] != n:
            raise ValueError(f"{name} has shape {matrix.shape} for {n} retained cells")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} holds values that are not finite")
    if umap.shape[1] != 2:
        raise ValueError(f"umap has {umap.shape[1]} columns, not 2")


def write_outputs(
    path: str | Path, *, workflow: str, mode: str, cell_ids, positions, clusters, pca, umap
) -> None:
    import h5py

    ids = np.asarray(cell_ids).astype(str)
    positions = np.asarray(positions, dtype=np.int64)
    clusters = np.asarray(clusters).astype(str)
    pca = np.asarray(pca, dtype=np.float32)
    umap = np.asarray(umap, dtype=np.float32)
    _check(ids.size, positions, clusters, pca, umap)
    options = {"compression": "gzip", "compression_opts": 4}
    encode = getattr(np, "strings", np.char).encode
    with h5py.File(path, "w") as h5:
        h5.attrs["workflow"] = workflow
        h5.attrs["mode"] = mode
        h5.create_dataset("cellIds", data=encode(ids, "utf-8"), **options)
        h5.create_dataset("positions", data=positions, **options)
        h5.create_dataset("clusters", data=encode(clusters, "utf-8"), **options)
        h5.create_dataset("pca", data=pca, **options)
        h5.create_dataset("umap", data=umap, **options)


def _text(dataset) -> np.ndarray:
    return np.asarray(dataset.asstr(encoding="utf-8")[:], dtype=object).ravel().astype(str)


def _attribute(value) -> str:
    items = np.asarray(value, dtype=object).ravel()
    return " ".join((item.decode() if isinstance(item, bytes) else str(item) for item in items))


def _cells_by_columns(matrix: np.ndarray, n: int, name: str) -> tuple[np.ndarray, str]:
    if matrix.ndim == 2 and matrix.shape[0] == n:
        return (np.ascontiguousarray(matrix), "cells by columns")
    if matrix.ndim == 2 and matrix.shape[1] == n:
        return (np.ascontiguousarray(matrix.T), "columns by cells (transposed on reading)")
    raise ValueError(f"{name} has shape {matrix.shape}, with no axis of {n} retained cells")


def read_outputs(path: str | Path) -> dict:
    import h5py

    with h5py.File(path, "r") as h5:
        ids = _text(h5["cellIds"])
        positions = np.asarray(h5["positions"][:], dtype=np.int64).ravel()
        clusters = _text(h5["clusters"])
        n = positions.size
        pca, pca_layout = _cells_by_columns(np.asarray(h5["pca"][:], dtype=np.float64), n, "pca")
        umap, umap_layout = _cells_by_columns(
            np.asarray(h5["umap"][:], dtype=np.float64), n, "umap"
        )
        attributes = {str(key): _attribute(value) for key, value in h5.attrs.items()}
    _check(ids.size, positions, clusters, pca, umap)
    return {
        "cellIds": ids,
        "positions": positions,
        "clusters": clusters,
        "pca": pca,
        "umap": umap,
        "attributes": attributes,
        "layout": {"pca": pca_layout, "umap": umap_layout},
    }
