"""Scanpy workflow for the comparison, in memory or on Dask-backed arrays.

Usage: python workflow_scanpy.py MODE WORKERS H5AD EVENTS SUMMARY WORKDIR
MODE is one of:
  memory     the whole matrix in memory.
  dask       quality control, filtering, normalisation and variable genes out of core on row
             chunks; the variable-gene matrix computed into memory for scaling and PCA; graph,
             UMAP and Leiden in memory; the Wilcoxon test on the normalised matrix assembled in
             memory by concatenating its blocks, because it has no out-of-core path.
  census     the cells are queried from CELLxGENE Census instead of read from the H5AD, then
             analysed in memory as in "memory"; the Census release, dataset and number of cells
             come from CENSUS_VERSION, CENSUS_DATASET_ID and CENSUS_CELLS.
  memory_csc          "memory" with the marker tests on the matrix converted to CSC (gene
                      columns), the conversion timed as its own step (convertCsc).
  memory_csc_pcainit  "memory_csc" with UMAP started from the principal components (umap-learn's
                      init "pca": the first two components of the representation, scaled as the
                      spectral start is) instead of the spectral start.
  dask_covariance_csr      "dask_materialized_pca" with Scanpy's Dask route kept through scaling and PCA: the variable genes stay
             a Dask array through scaling (max_value 10) and the covariance-eigendecomposition PCA
             (svd_solver "covariance_eigh"), and only the cell coordinates are computed. The
             marker input is materialised as CSR without a second copy: a first pass counts the
             values of every block, a second fills preallocated arrays block by block
             (materializeMarkerInput, its own timed step).
  dask_covariance_csc  "dask_covariance_csr" with the marker input materialised directly as CSC: a first pass counts
             the values of every gene in every block, a second scatters each block into its
             slots, so each gene's rows stay in order.

The steps and parameters match the CyteArc benchmark: cells outside the 1st to 99th
percentiles of total counts, detected genes, and mitochondrial and ribosomal fractions
are removed, as are cells with fewer than 10 detected genes and genes detected in fewer
than 20 cells; 1,000 variable genes; 21 principal components; 11 neighbours; UMAP for
300 epochs; Leiden at resolution 1.0; Wilcoxon markers of every cluster on all genes.

With VERIFY_MARKERS=1 (the smoke runs), the arms with CSC markers or a materialised marker
input also test the markers on the CSR matrix of the other route and record whether names,
scores and P values are identical (markerCheck in the summary); the check is not timed as
a workflow step that the comparison reports.

The Dask modes also read a Zarr copy of the input written by anndata (experiments.benchmarks.workflows
prepare-zarr): H5AD is then a local directory ending in .zarr, read from the container's disk,
or an s3:// location, read in place from object storage with obstore, the client CyteArc uses,
and CyteArc's retry settings; the endpoint comes from R2_ENDPOINT_URL and the credentials from
R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY. The summary records which (input).

With KEEP_OUTPUTS=1 (the keepOutputs payload setting, the quality comparison; the in-memory modes
that read the H5AD only), a last step, keepOutputs, writes the per-cell outputs to
WORKDIR/outputs.h5 (tools/kept_outputs.py): every input cell's identifier, the retained cells,
their Leiden clusters, principal components and UMAP coordinates. Without it nothing changes.

With WORKFLOW_SETTINGS (the workflowSettings payload setting; the Dask modes only), a JSON object
replaces some of the parameters above: filterCells (false skips cell and gene filtering),
variableGenes, variableGeneExclusionPattern (a regular expression on feature_name; matching genes
are not candidates), principalComponents, neighbors, umap and markers (false skips the step).
The summary records the settings used. Without it nothing changes.
"""

import json
import os
import sys
import time

import numpy as np

MODE, WORKERS, H5AD, EVENTS, SUMMARY, WORKDIR = sys.argv[1:7]
WORKERS = int(WORKERS)
SUMMARY_DATA: dict = {}
VERIFY = os.environ.get("VERIFY_MARKERS") == "1"
KEEP_OUTPUTS = os.environ.get("KEEP_OUTPUTS") == "1"
KEEP_OUTPUTS_MODES = ("memory", "memory_csc", "memory_csc_pcainit")
DEFAULT_SETTINGS = {
    "filterCells": True,
    "variableGenes": 1000,
    "variableGeneExclusionPattern": None,
    "principalComponents": 21,
    "neighbors": 11,
    "umap": True,
    "markers": True,
}
SETTINGS = {**DEFAULT_SETTINGS, **json.loads(os.environ.get("WORKFLOW_SETTINGS", "{}"))}


def emit(step: str, event: str, **extra) -> None:
    with open(EVENTS, "a") as handle:
        handle.write(json.dumps({"step": step, "event": event, "t": time.time(), **extra}) + "\n")


def step(name: str, fn):
    emit(name, "start")
    value = fn()
    emit(name, "end")
    return value


def save_summary() -> None:
    with open(SUMMARY, "w") as handle:
        json.dump(SUMMARY_DATA, handle, default=str)


def open_input():
    if H5AD.startswith("s3://"):
        from datetime import timedelta

        import zarr
        from obstore.store import S3Store

        bucket, _, prefix = H5AD.removeprefix("s3://").partition("/")
        store = S3Store(
            bucket,
            prefix=prefix,
            endpoint=os.environ["R2_ENDPOINT_URL"],
            access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region="auto",
            retry_config={"max_retries": 100, "retry_timeout": timedelta(minutes=3)},
        )
        SUMMARY_DATA["input"] = {"format": "zarr", "location": "objectStorage", "client": "obstore"}
        return zarr.open_group(zarr.storage.ObjectStore(store, read_only=True), mode="r")
    if H5AD.rstrip("/").endswith(".zarr"):
        import zarr

        SUMMARY_DATA["input"] = {"format": "zarr", "location": "localDisk"}
        return zarr.open_group(H5AD, mode="r")
    import h5py

    SUMMARY_DATA["input"] = {"format": "h5ad", "location": "localDisk"}
    return h5py.File(H5AD, "r")


def matrix_facts() -> None:
    h5 = open_input()
    SUMMARY_DATA["nnz"] = int(h5["X"]["data"].shape[0])
    shape = h5["X"].attrs["shape"]
    SUMMARY_DATA["nCells"], SUMMARY_DATA["nGenes"] = (int(shape[0]), int(shape[1]))
    if hasattr(h5, "close"):
        h5.close()
    SUMMARY_DATA["sparseMatrixBytesEstimate"] = (
        SUMMARY_DATA["nnz"] * 8 + (SUMMARY_DATA["nCells"] + 1) * 8
    )
    save_summary()


def quality_masks(names):
    upper = names.str.upper()
    return (upper.str.startswith("MT-").to_numpy(), upper.str.match("^RP[SL]").to_numpy())


def marker_results(adata, key: str = "rank_genes_groups") -> dict:
    result = adata.uns[key]
    groups = result["names"].dtype.names
    return {
        field: {g: np.asarray(result[field][g]) for g in groups}
        for field in ("names", "scores", "pvals", "logfoldchanges")
    }


def compare_markers(tested: dict, reference: dict) -> dict:
    groups = sorted(reference["names"])
    check = {
        "groups": len(groups),
        "sameGroups": sorted(tested["names"]) == groups,
        "namesEqual": all(
            (np.array_equal(tested["names"][g], reference["names"][g]) for g in groups)
        ),
        "scoresEqual": all(
            (np.array_equal(tested["scores"][g], reference["scores"][g]) for g in groups)
        ),
        "pvalsEqual": all(
            (np.array_equal(tested["pvals"][g], reference["pvals"][g]) for g in groups)
        ),
    }
    for field in ("scores", "pvals", "logfoldchanges"):
        check[f"{field}MaxAbsDiff"] = max(
            (
                float(
                    np.max(
                        np.abs(
                            tested[field][g].astype(np.float64)
                            - reference[field][g].astype(np.float64)
                        ),
                        initial=0.0,
                    )
                )
                for g in groups
            )
        )
    check["identical"] = (
        check["sameGroups"] and check["namesEqual"] and check["scoresEqual"] and check["pvalsEqual"]
    )
    return check


def verify_on_csr(tested, reference_matrix, labels, route: str) -> None:
    import anndata as ad
    import scanpy as sc

    reference = ad.AnnData(X=reference_matrix, obs=tested.obs[[]].copy(), var=tested.var[[]].copy())
    reference.obs["leiden"] = labels
    started = time.time()
    sc.tl.rank_genes_groups(reference, "leiden", method="wilcoxon", n_genes=100)
    check = compare_markers(marker_results(tested), marker_results(reference))
    check.update({"reference": route, "referenceSeconds": time.time() - started})
    SUMMARY_DATA["markerCheck"] = check
    save_summary()


def keep_outputs(hv) -> None:
    import anndata as ad
    import h5py
    import pandas as pd

    from experiments.benchmarks.tools.kept_outputs import FILE_NAME, write_outputs

    with h5py.File(H5AD, "r") as h5:
        obs = h5["obs"]
        key = obs.attrs.get("_index", "_index")
        index = obs[key.decode() if isinstance(key, bytes) else str(key)]
        ids = np.asarray(ad.io.read_elem(index), dtype=object)
        if pd.isna(ids).any():
            raise ValueError("Input cell identifiers contain missing values")
        ids = ids.astype(str)
    positions = pd.Index(ids).get_indexer(hv.obs_names)
    if (positions < 0).any():
        raise ValueError(
            f"{int((positions < 0).sum())} retained cells are not among the input cells"
        )
    write_outputs(
        f"{WORKDIR}/{FILE_NAME}",
        workflow="workflow_scanpy.py",
        mode=MODE,
        cell_ids=ids,
        positions=positions,
        clusters=hv.obs["leiden"].astype(str).to_numpy(),
        pca=hv.obsm["X_pca"],
        umap=hv.obsm["X_umap"],
    )


def census_read():
    import cellxgene_census

    version, dataset_id = (os.environ["CENSUS_VERSION"], os.environ["CENSUS_DATASET_ID"])
    cells = int(os.environ["CENSUS_CELLS"])
    with cellxgene_census.open_soma(census_version=version) as census:
        organism = census["census_data"]["mus_musculus"]

        def select():
            obs = (
                organism.obs.read(
                    value_filter=f"dataset_id == '{dataset_id}'", column_names=["soma_joinid"]
                )
                .concat()
                .to_pandas()
            )
            joinids = obs["soma_joinid"].to_numpy()
            return np.sort(joinids)[: min(cells, joinids.size)]

        chosen = step("queryObs", select)
        adata = step(
            "read",
            lambda: cellxgene_census.get_anndata(
                census,
                organism="Mus musculus",
                measurement_name="RNA",
                X_name="raw",
                obs_coords=chosen,
                column_names={
                    "obs": ["soma_joinid", "dataset_id"],
                    "var": ["feature_id", "feature_name"],
                },
            ),
        )
    adata.var_names = adata.var["feature_id"].astype(str).to_numpy()
    SUMMARY_DATA.update(
        {
            "nCells": int(adata.n_obs),
            "nGenes": int(adata.n_vars),
            "nnz": int(adata.X.nnz),
            "censusVersion": version,
            "censusDatasetId": dataset_id,
        }
    )
    save_summary()
    return adata


def run_memory(census: bool = False, csc: bool = False, umap_init: str = "spectral") -> None:
    import anndata as ad
    import scanpy as sc

    sc.settings.n_jobs = WORKERS
    adata = census_read() if census else step("read", lambda: ad.read_h5ad(H5AD))
    mt, ribo = quality_masks(adata.var["feature_name"].astype(str))
    adata.var["mt"], adata.var["ribo"] = (mt, ribo)

    def filter_cells() -> None:
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt", "ribo"], inplace=True, percent_top=None, log1p=False
        )
        keep = np.ones(adata.n_obs, dtype=bool)
        for column in ("total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"):
            values = adata.obs[column].to_numpy()
            low, high = np.quantile(values, [0.01, 0.99])
            keep &= (values >= low) & (values <= high)
        keep &= adata.obs["n_genes_by_counts"].to_numpy() >= 10
        adata._inplace_subset_obs(keep)
        sc.pp.filter_genes(adata, min_cells=20)

    step("filterCells", filter_cells)

    def normalize() -> None:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    step("runNormalization", normalize)
    step("markHvgs", lambda: sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat"))
    holder: dict = {}

    def pca() -> None:
        hv = adata[:, adata.var["highly_variable"].to_numpy()].copy()
        sc.pp.scale(hv, max_value=10)
        sc.pp.pca(hv, n_comps=21, random_state=4466)
        holder["hv"] = hv

    step("runPca", pca)
    hv = holder["hv"]
    step("queryNeighbors", lambda: sc.pp.neighbors(hv, n_neighbors=11, n_pcs=21, random_state=4466))
    step("runUmap", lambda: sc.tl.umap(hv, maxiter=300, random_state=4444, init_pos=umap_init))
    step(
        "runLeiden",
        lambda: sc.tl.leiden(
            hv, resolution=1.0, flavor="igraph", n_iterations=2, random_state=4444
        ),
    )
    if umap_init != "spectral":
        SUMMARY_DATA["umapInit"] = umap_init
    SUMMARY_DATA["markerFormat"] = "csc" if csc else "csr"
    csr = adata.X if csc and VERIFY else None
    if csc:

        def convert() -> None:
            adata.X = adata.X.tocsc()

        step("convertCsc", convert)

    def markers() -> None:
        adata.obs["leiden"] = hv.obs["leiden"].to_numpy()
        sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", n_genes=100)

    step("findMarkers", markers)
    SUMMARY_DATA.update(
        {
            "cellsAfterFilter": int(adata.n_obs),
            "genesAfterFilter": int(adata.n_vars),
            "clusters": int(hv.obs["leiden"].nunique()),
        }
    )
    if csr is not None:
        verify_on_csr(
            adata, csr, adata.obs["leiden"].to_numpy(), "the CSR matrix before conversion"
        )
    if KEEP_OUTPUTS:
        step("keepOutputs", lambda: keep_outputs(hv))


def variable_genes(adata) -> None:
    import scanpy as sc

    pattern = SETTINGS["variableGeneExclusionPattern"]
    if pattern is None:
        sc.pp.highly_variable_genes(adata, n_top_genes=SETTINGS["variableGenes"], flavor="seurat")
        return
    names = adata.var["feature_name"].astype(str)
    eligible = np.flatnonzero(~names.str.match(pattern).to_numpy())
    table = sc.pp.highly_variable_genes(
        adata[:, eligible], n_top_genes=SETTINGS["variableGenes"], flavor="seurat", inplace=False
    )
    selected = np.zeros(adata.n_vars, dtype=bool)
    selected[eligible] = table["highly_variable"].to_numpy()
    adata.var["highly_variable"] = selected


def materialize(matrix, layout: str):
    from concurrent.futures import ThreadPoolExecutor

    import scipy.sparse as sp

    if len(matrix.chunks[1]) != 1:
        matrix = matrix.rechunk({1: -1})
    blocks = matrix.to_delayed().ravel()
    starts = np.concatenate([[0], np.cumsum(matrix.chunks[0])]).astype(np.int64)
    n_rows, n_cols = (int(n) for n in matrix.shape)

    def compute(index: int):
        return sp.csr_matrix(blocks[index].compute(scheduler="synchronous"))

    def each_block(fn) -> list:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            return list(pool.map(fn, range(len(blocks))))

    if layout == "csr":
        counts = np.asarray(each_block(lambda i: compute(i).nnz), dtype=np.int64)
    else:
        counts = np.stack(
            each_block(lambda i: np.bincount(compute(i).indices, minlength=n_cols).astype(np.int64))
        )
    nnz = int(counts.sum())
    index_dtype = np.int32 if max(nnz, n_rows, n_cols) < 2**31 else np.int64
    data = np.empty(nnz, dtype=np.float32)
    indices = np.empty(nnz, dtype=index_dtype)
    if layout == "csr":
        indptr = np.empty(n_rows + 1, dtype=index_dtype)
        indptr[0] = 0
        offsets = np.concatenate([[0], np.cumsum(counts)])

        def fill_rows(index: int) -> None:
            block = compute(index)
            if block.nnz != counts[index]:
                raise RuntimeError(f"block {index} changed between the two passes")
            low = offsets[index]
            data[low : low + block.nnz] = block.data
            indices[low : low + block.nnz] = block.indices
            indptr[starts[index] + 1 : starts[index + 1] + 1] = block.indptr[1:] + low

        each_block(fill_rows)
        return sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
    import numba

    @numba.njit(nogil=True)
    def scatter(block_indptr, block_indices, block_data, row_start, slots, out_indices, out_data):
        for row in range(block_indptr.shape[0] - 1):
            for k in range(block_indptr[row], block_indptr[row + 1]):
                column = block_indices[k]
                slot = slots[column]
                out_indices[slot] = row_start + row
                out_data[slot] = block_data[k]
                slots[column] = slot + 1

    indptr = np.zeros(n_cols + 1, dtype=index_dtype)
    np.cumsum(counts.sum(axis=0), out=indptr[1:])
    first_slots = indptr[:-1].astype(np.int64)[None, :] + np.cumsum(counts, axis=0) - counts

    def fill_columns(index: int) -> None:
        block = compute(index)
        slots = first_slots[index].copy()
        scatter(block.indptr, block.indices, block.data, starts[index], slots, indices, data)
        if not np.array_equal(slots, first_slots[index] + counts[index]):
            raise RuntimeError(f"block {index} changed between the two passes")

    each_block(fill_columns)
    return sp.csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))


def run_dask(materialize_pca: bool = True, csc: bool = False) -> None:
    import importlib

    import anndata as ad
    import dask
    import scanpy as sc
    import scipy.sparse as sp
    from anndata.io import read_elem

    read_elem_lazy = None
    for module, name in (
        ("anndata.experimental", "read_elem_lazy"),
        ("anndata.io", "read_elem_lazy"),
        ("anndata.experimental", "read_elem_as_dask"),
    ):
        try:
            read_elem_lazy = getattr(importlib.import_module(module), name)
            SUMMARY_DATA["lazyReader"] = f"{module}.{name}"
            break
        except ImportError, AttributeError:
            continue
    if read_elem_lazy is None:
        raise ImportError(f"no lazy element reader in anndata {ad.__version__}")
    chunk_rows = 50000
    SUMMARY_DATA["chunkRows"] = chunk_rows
    dask.config.set(scheduler="threads", num_workers=WORKERS)
    sc.settings.n_jobs = WORKERS
    handle = open_input()

    def read_lazy():
        X = read_elem_lazy(handle["X"], chunks=(chunk_rows, -1))
        return ad.AnnData(X=X, obs=read_elem(handle["obs"]), var=read_elem(handle["var"]))

    adata = step("readLazy", read_lazy)
    names = adata.var["feature_name"].astype(str)
    mt_mask, ribo_mask = quality_masks(names)
    mt, ribo = (np.flatnonzero(mt_mask), np.flatnonzero(ribo_mask))

    def cell_stats(block):
        block = sp.csr_matrix(block)
        total = np.asarray(block.sum(axis=1)).ravel()
        n_genes = np.diff(block.indptr).astype(np.float64)
        return np.stack(
            [
                total,
                n_genes,
                np.asarray(block[:, mt].sum(axis=1)).ravel(),
                np.asarray(block[:, ribo].sum(axis=1)).ravel(),
            ],
            axis=1,
        )

    def filter_cells():
        stats = adata.X.map_blocks(
            cell_stats, dtype=np.float64, chunks=(adata.X.chunks[0], (4,)), meta=np.empty((0, 4))
        ).compute()
        total, n_genes = (stats[:, 0], stats[:, 1])
        keep = np.ones(adata.n_obs, dtype=bool)
        for values in (
            total,
            n_genes,
            100 * stats[:, 2] / np.maximum(total, 1),
            100 * stats[:, 3] / np.maximum(total, 1),
        ):
            low, high = np.quantile(values, [0.01, 0.99])
            keep &= (values >= low) & (values <= high)
        keep &= n_genes >= 10
        filtered = adata[np.flatnonzero(keep)].copy()

        def gene_cells(block):
            return (
                np.asarray((sp.csr_matrix(block) > 0).sum(axis=0)).reshape(1, -1).astype(np.float64)
            )

        counts = (
            filtered.X.map_blocks(
                gene_cells,
                dtype=np.float64,
                chunks=((1,) * len(filtered.X.chunks[0]), (filtered.n_vars,)),
                meta=np.empty((0, filtered.n_vars)),
            )
            .sum(axis=0)
            .compute()
        )
        return filtered[:, np.flatnonzero(counts >= 20)].copy()

    if SETTINGS["filterCells"]:
        adata = step("filterCells", filter_cells)
    SUMMARY_DATA["cellsAfterFilter"], SUMMARY_DATA["genesAfterFilter"] = (
        int(adata.n_obs),
        int(adata.n_vars),
    )
    save_summary()

    def normalize() -> None:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    step("runNormalization", normalize)
    step("markHvgs", lambda: variable_genes(adata))
    holder: dict = {}
    components = SETTINGS["principalComponents"]

    def pca() -> None:
        hv = adata[:, np.flatnonzero(adata.var["highly_variable"].to_numpy())].copy()
        hv.X = sp.csr_matrix(hv.X.compute())
        sc.pp.scale(hv, max_value=10)
        sc.pp.pca(hv, n_comps=components, random_state=4466)
        holder["hv"] = hv

    def pca_lazy() -> None:
        hv = adata[:, np.flatnonzero(adata.var["highly_variable"].to_numpy())].copy()
        sc.pp.scale(hv, max_value=10)
        sc.pp.pca(hv, n_comps=components, svd_solver="covariance_eigh", random_state=4466)
        hv.obsm["X_pca"] = hv.obsm["X_pca"].compute()
        holder["hv"] = hv

    step("runPca", pca if materialize_pca else pca_lazy)
    hv = holder["hv"]
    step(
        "queryNeighbors",
        lambda: sc.pp.neighbors(
            hv, n_neighbors=SETTINGS["neighbors"], n_pcs=components, random_state=4466
        ),
    )
    if SETTINGS["umap"]:
        step("runUmap", lambda: sc.tl.umap(hv, maxiter=300, random_state=4444))
    step(
        "runLeiden",
        lambda: sc.tl.leiden(
            hv, resolution=1.0, flavor="igraph", n_iterations=2, random_state=4444
        ),
    )
    if SETTINGS["markers"] and materialize_pca:

        def markers() -> None:
            blocks = adata.X.to_delayed().ravel()
            data, indices, indptr = ([], [], [np.zeros(1, dtype=np.int64)])
            offset = 0
            for block in blocks:
                b = sp.csr_matrix(block.compute())
                data.append(b.data.astype(np.float32, copy=False))
                indices.append(b.indices.astype(np.int32, copy=False))
                indptr.append(b.indptr[1:].astype(np.int64) + offset)
                offset += b.nnz
            X = sp.csr_matrix(
                (np.concatenate(data), np.concatenate(indices), np.concatenate(indptr)),
                shape=(adata.n_obs, adata.n_vars),
            )
            del data, indices, indptr
            in_memory = ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())
            in_memory.obs["leiden"] = hv.obs["leiden"].to_numpy()
            sc.tl.rank_genes_groups(in_memory, "leiden", method="wilcoxon", n_genes=100)

        step("findMarkers", markers)
    elif SETTINGS["markers"]:
        layout = "csc" if csc else "csr"
        SUMMARY_DATA.update({"pcaRoute": "dask covariance_eigh", "markerFormat": layout})
        X = step("materializeMarkerInput", lambda: materialize(adata.X, layout))
        SUMMARY_DATA["markerInputNnz"] = int(X.nnz)
        labels = hv.obs["leiden"].to_numpy()
        holder["markers"] = ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())

        def markers() -> None:
            holder["markers"].obs["leiden"] = labels
            sc.tl.rank_genes_groups(holder["markers"], "leiden", method="wilcoxon", n_genes=100)

        step("findMarkers", markers)
        if VERIFY:
            reference = materialize(adata.X, "csr") if csc else sp.csr_matrix(adata.X.compute())
            SUMMARY_DATA["markerInputEqual"] = bool(
                (X != (reference.tocsc() if csc else reference)).nnz == 0
            )
            verify_on_csr(
                holder["markers"],
                reference,
                labels,
                "CSR materialisation" if csc else "the blocks computed at once",
            )
    SUMMARY_DATA["clusters"] = int(hv.obs["leiden"].nunique())
    if hasattr(handle, "close"):
        handle.close()


MODES = {
    "memory": lambda: run_memory(),
    "memory_csc": lambda: run_memory(csc=True),
    "memory_csc_pcainit": lambda: run_memory(csc=True, umap_init="pca"),
    "census": lambda: run_memory(census=True),
    "dask_materialized_pca": lambda: run_dask(),
    "dask_covariance_csr": lambda: run_dask(materialize_pca=False),
    "dask_covariance_csc": lambda: run_dask(materialize_pca=False, csc=True),
}


def main() -> None:
    import anndata
    import scanpy

    if MODE not in MODES:
        raise SystemExit(f"MODE must be one of {sorted(MODES)}")
    if SETTINGS.keys() != DEFAULT_SETTINGS.keys():
        raise SystemExit(f"WORKFLOW_SETTINGS keys must be among {sorted(DEFAULT_SETTINGS)}")
    if SETTINGS != DEFAULT_SETTINGS and not MODE.startswith("dask"):
        raise SystemExit("WORKFLOW_SETTINGS applies to the Dask modes only")
    SUMMARY_DATA["settings"] = SETTINGS
    if KEEP_OUTPUTS and MODE not in KEEP_OUTPUTS_MODES:
        raise SystemExit(f"KEEP_OUTPUTS applies to the modes {', '.join(KEEP_OUTPUTS_MODES)} only")
    SUMMARY_DATA["versions"] = {"scanpy": scanpy.__version__, "anndata": anndata.__version__}
    if MODE.startswith("dask"):
        import dask

        SUMMARY_DATA["versions"]["dask"] = dask.__version__
    if MODE == "census":
        import cellxgene_census

        SUMMARY_DATA["versions"]["cellxgene_census"] = cellxgene_census.__version__
    else:
        matrix_facts()
    SUMMARY_DATA["mode"] = MODE
    try:
        MODES[MODE]()
    finally:
        save_summary()


if __name__ == "__main__":
    main()
