"""Common inputs for the marker-only comparison, made once per dataset (experiments.benchmarks.workflows).

No prepared H5AD carries an author cell-type column (the benchmark samples and the converted
datasets keep only obs/_index), so the groups are one fixed clustering computed here and
used by every tool. The output is one H5AD in the layout of the benchmark samples (raw
counts, CSR, float32 values, int64 indices, uncompressed) holding:

- the cells and genes of the other tools' filter (tools/workflow_scanpy.py): cells within the
  empirical 1st to 99th percentiles, over every cell, of total counts, detected genes and the
  percentages of mitochondrial ("MT-") and ribosomal ("RP[SL]") counts, gene symbols
  upper-cased as there, with at least 10 detected genes; then genes detected in at least 20
  retained cells;
- obs/group: the fixed clustering, integer labels. Leiden (igraph, resolution 1.0, two
  iterations, seed 4444) on the 11-neighbour graph (seed 4466) of 21 principal components of
  the 1,000 variable genes (Seurat flavour) of the normalised (total counts, then log1p) and
  scaled (max_value 10) retained matrix, computed with Scanpy on Dask-backed arrays
  (covariance-eigendecomposition PCA): the Scanpy workflow's clustering, from the matrix
  every tool receives;
- var/marker_sample and uns/marker_sample_groups: a seeded sample of genes and groups whose
  statistics every tool reports, to show that the tools compute the same test.

Each tool normalises the matrix itself (total counts of the retained genes, then log1p) and
tests every group against the rest on every gene.
"""

import gzip
import hashlib
import importlib
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from experiments.runtime.h5ad import (
    _column,
    _dataframe,
    _index,
    _string_dataset,
)

BLOCK_ROWS = 50000
SAMPLE_GENES, SAMPLE_GROUPS, SAMPLE_SEED = (200, 30, 0)
MIN_PERCENTILE, MAX_PERCENTILE = (0.01, 0.99)
MIN_GENES, GENE_MIN_CELLS = (10, 20)


def _row_blocks(matrix: h5py.Group, indptr: np.ndarray, n_genes: int):
    n_cells = indptr.shape[0] - 1
    for start in range(0, n_cells, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, n_cells)
        low, high = (int(indptr[start]), int(indptr[stop]))
        data = matrix["data"][low:high]
        indices = matrix["indices"][low:high]
        yield (
            start,
            stop,
            sp.csr_matrix(
                (data, indices, indptr[start : stop + 1] - low), shape=(stop - start, n_genes)
            ),
        )


def other_tools_filter(source: Path) -> dict:
    with h5py.File(source, "r") as h5:
        matrix = h5["X"]
        n_cells, n_genes = (int(v) for v in matrix.attrs["shape"])
        indptr = matrix["indptr"][:].astype(np.int64)
        symbols = np.char.upper(_column(h5["var"], "feature_name").astype(str))
        mito = np.flatnonzero(np.char.startswith(symbols, "MT-"))
        ribo = np.flatnonzero(
            np.char.startswith(symbols, "RPS") | np.char.startswith(symbols, "RPL")
        )
        stats = np.zeros((n_cells, 4), dtype=np.float64)
        for start, stop, block in _row_blocks(matrix, indptr, n_genes):
            block = block.astype(np.float64)
            stats[start:stop, 0] = np.asarray(block.sum(axis=1)).ravel()
            stats[start:stop, 1] = np.asarray((block > 0).sum(axis=1)).ravel()
            stats[start:stop, 2] = np.asarray(block[:, mito].sum(axis=1)).ravel()
            stats[start:stop, 3] = np.asarray(block[:, ribo].sum(axis=1)).ravel()
        total, detected = (stats[:, 0], stats[:, 1])
        metrics = {
            "totalCounts": total,
            "detectedGenes": detected,
            "percentMito": 100 * stats[:, 2] / np.maximum(total, 1),
            "percentRibo": 100 * stats[:, 3] / np.maximum(total, 1),
        }
        keep = np.ones(n_cells, dtype=bool)
        bounds = {}
        for name, values in metrics.items():
            low, high = np.quantile(values, [MIN_PERCENTILE, MAX_PERCENTILE])
            bounds[name] = [float(low), float(high)]
            keep &= (values >= low) & (values <= high)
        keep &= detected >= MIN_GENES
        cells_per_gene = np.zeros(n_genes, dtype=np.int64)
        for start, stop, block in _row_blocks(matrix, indptr, n_genes):
            kept = block[keep[start:stop]]
            cells_per_gene += np.asarray((kept > 0).sum(axis=0)).ravel().astype(np.int64)
    genes = cells_per_gene >= GENE_MIN_CELLS
    return {
        "keep": keep,
        "genes": genes,
        "bounds": bounds,
        "nCellsSource": n_cells,
        "nGenesSource": n_genes,
        "nnzSource": int(indptr[-1]),
        "mitoGenes": int(mito.size),
        "riboGenes": int(ribo.size),
    }


def write_common(source: Path, destination: Path, keep: np.ndarray, genes: np.ndarray) -> int:
    gene_index = np.flatnonzero(genes)
    with h5py.File(source, "r") as h5, h5py.File(destination, "w") as out:
        matrix = h5["X"]
        n_genes = int(matrix.attrs["shape"][1])
        indptr = matrix["indptr"][:].astype(np.int64)
        out.attrs["encoding-type"] = "anndata"
        out.attrs["encoding-version"] = "0.1.0"
        x = out.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = np.asarray([int(keep.sum()), int(gene_index.size)], dtype=np.int64)
        out_data = x.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1048576,)
        )
        out_indices = x.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1048576,)
        )
        out_indptr = [np.zeros(1, dtype=np.int64)]
        written = 0
        for start, stop, block in _row_blocks(matrix, indptr, n_genes):
            kept = block[keep[start:stop]][:, gene_index]
            kept.sort_indices()
            end = written + kept.nnz
            out_data.resize((end,))
            out_indices.resize((end,))
            out_data[written:end] = kept.data.astype(np.float32)
            out_indices[written:end] = kept.indices.astype(np.int64)
            out_indptr.append(kept.indptr[1:].astype(np.int64) + written)
            written = end
        x.create_dataset("indptr", data=np.concatenate(out_indptr))
        obs = _dataframe(out, "obs", ())
        _string_dataset(obs, "_index", _index(h5["obs"])[keep])
        var = _dataframe(out, "var", ("feature_name",))
        _string_dataset(var, "_index", _index(h5["var"])[gene_index])
        _string_dataset(var, "feature_name", _column(h5["var"], "feature_name")[gene_index])
    return written


def fixed_clustering(path: Path, workers: int) -> tuple[np.ndarray, dict]:
    import anndata as ad
    import dask
    import scanpy as sc
    from anndata.io import read_elem

    reader = None
    for module, name in (
        ("anndata.experimental", "read_elem_lazy"),
        ("anndata.io", "read_elem_lazy"),
    ):
        try:
            reader = getattr(importlib.import_module(module), name)
            break
        except ImportError, AttributeError:
            continue
    if reader is None:
        raise ImportError(f"no lazy element reader in anndata {ad.__version__}")
    dask.config.set(scheduler="threads", num_workers=workers)
    sc.settings.n_jobs = workers
    seconds = {}
    with h5py.File(path, "r") as handle:
        adata = ad.AnnData(
            X=reader(handle["X"], chunks=(50000, -1)),
            obs=read_elem(handle["obs"]),
            var=read_elem(handle["var"]),
        )
        started = time.time()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat")
        seconds["variableGenes"] = time.time() - started
        started = time.time()
        hv = adata[:, np.flatnonzero(adata.var["highly_variable"].to_numpy())].copy()
        sc.pp.scale(hv, max_value=10)
        sc.pp.pca(hv, n_comps=21, svd_solver="covariance_eigh", random_state=4466)
        hv.obsm["X_pca"] = hv.obsm["X_pca"].compute()
        seconds["pca"] = time.time() - started
    started = time.time()
    sc.pp.neighbors(hv, n_neighbors=11, n_pcs=21, random_state=4466)
    seconds["neighbors"] = time.time() - started
    started = time.time()
    sc.tl.leiden(hv, resolution=1.0, flavor="igraph", n_iterations=2, random_state=4444)
    seconds["leiden"] = time.time() - started
    labels = hv.obs["leiden"].astype(int).to_numpy().astype(np.int32)
    method = {
        "tool": "scanpy",
        "versions": {"scanpy": sc.__version__, "anndata": ad.__version__, "dask": dask.__version__},
        "normalisation": "normalize_total (median target), log1p",
        "variableGenes": {"flavor": "seurat", "nTop": 1000},
        "scale": {"maxValue": 10},
        "pca": {"nComps": 21, "svdSolver": "covariance_eigh", "arrays": "dask, 50,000-cell chunks"},
        "neighbors": {"nNeighbors": 11, "nPcs": 21, "randomState": 4466},
        "leiden": {"flavor": "igraph", "resolution": 1.0, "nIterations": 2, "randomState": 4444},
        "seconds": seconds,
    }
    return (labels, method)


def _array(group: h5py.Group, name: str, values: np.ndarray) -> None:
    dataset = group.create_dataset(name, data=values)
    dataset.attrs["encoding-type"] = "array"
    dataset.attrs["encoding-version"] = "0.2.0"


def _add_column(frame: h5py.Group, name: str, values: np.ndarray) -> None:
    _array(frame, name, values)
    order = [
        v.decode() if isinstance(v, bytes) else str(v) for v in frame.attrs.get("column-order", [])
    ]
    frame.attrs["column-order"] = np.asarray([*order, name], dtype=h5py.string_dtype("utf-8"))


def add_groups_and_sample(path: Path, labels: np.ndarray) -> dict:
    rng = np.random.default_rng(SAMPLE_SEED)
    with h5py.File(path, "r+") as h5:
        n_genes = int(h5["X"].attrs["shape"][1])
        groups = np.unique(labels)
        sample_genes = np.sort(rng.choice(n_genes, size=min(SAMPLE_GENES, n_genes), replace=False))
        sample_groups = np.sort(
            rng.choice(groups, size=min(SAMPLE_GROUPS, groups.size), replace=False)
        ).astype(np.int32)
        marked = np.zeros(n_genes, dtype=np.int8)
        marked[sample_genes] = 1
        _add_column(h5["obs"], "group", labels.astype(np.int32))
        _add_column(h5["var"], "marker_sample", marked)
        uns = h5.create_group("uns")
        uns.attrs["encoding-type"] = "dict"
        uns.attrs["encoding-version"] = "0.1.0"
        _array(uns, "marker_sample_groups", sample_groups)
        gene_ids = _index(h5["var"])[sample_genes]
    return {
        "sampleSeed": SAMPLE_SEED,
        "sampleGeneIds": [str(g) for g in gene_ids],
        "sampleGroups": [int(g) for g in sample_groups],
    }


def write_groups_table(path: Path, table: Path) -> None:
    with h5py.File(path, "r") as h5:
        cells = _index(h5["obs"])
        labels = h5["obs"]["group"][:]
    with gzip.open(table, "wt") as handle:
        handle.write("cell,group\n")
        for cell, label in zip(cells, labels):
            handle.write(f"{cell},{int(label)}\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(64 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def prepare(source: Path, work: Path, workers: int) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    common = work / "common.h5ad"
    timings = {}
    started = time.time()
    selected = other_tools_filter(source)
    timings["filter"] = time.time() - started
    started = time.time()
    nnz = write_common(source, common, selected["keep"], selected["genes"])
    timings["write"] = time.time() - started
    labels, method = fixed_clustering(common, workers)
    groups, sizes = np.unique(labels, return_counts=True)
    if sizes.min() < 2:
        raise ValueError(
            f"the clustering has a group of {int(sizes.min())} cell; every tool needs at least 2"
        )
    sample = add_groups_and_sample(common, labels)
    table = work / "groups.csv.gz"
    write_groups_table(common, table)
    started = time.time()
    digest = sha256(common)
    timings["sha256"] = time.time() - started
    return {
        "common": common,
        "groupsTable": table,
        "facts": {
            "nCellsSource": selected["nCellsSource"],
            "nGenesSource": selected["nGenesSource"],
            "nnzSource": selected["nnzSource"],
            "filter": {
                "rule": "empirical 1st to 99th percentiles over every cell of total counts, detected genes, percent MT- and RP[SL] counts; at least 10 detected genes; genes detected in at least 20 retained cells",
                "bounds": selected["bounds"],
                "mitoGenes": selected["mitoGenes"],
                "riboGenes": selected["riboGenes"],
            },
            "cellsAfterFilter": int(selected["keep"].sum()),
            "genesAfterFilter": int(selected["genes"].sum()),
            "nnz": nnz,
            "groups": int(groups.size),
            "groupSizes": {int(g): int(s) for g, s in zip(groups, sizes)},
            "clustering": method,
            "sample": sample,
            "bytes": common.stat().st_size,
            "sha256": digest,
            "timings": timings,
        },
    }
