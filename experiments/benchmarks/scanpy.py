from pathlib import Path

from experiments.runtime.steps import _Steps

BRAIN_CELLS = 1306127


def brain_prepare_job(payload: dict) -> dict:
    import os
    import time

    import h5py
    import numpy as np

    os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    import cytearc

    from experiments.runtime.h5ad import (
        _create_dataframe_group,
        _create_numeric_dataset,
        _create_string_dataset,
    )
    from experiments.runtime.object_io import object_exists, put_json, upload_file

    work = Path("/tmp/brain")
    work.mkdir(parents=True, exist_ok=True)
    steps = _Steps()
    repository = cytearc.datasets.connect("cytearc_docs")
    source_dir = steps.run(
        "download",
        lambda: repository.download_dataset(
            "tenx_1.3M_brain_rnaseq", destination=str(work), zarr=False
        ),
    )
    source = next(Path(source_dir).rglob("*.h5"))
    destination = work / f"{BRAIN_CELLS}.h5ad"

    def convert() -> dict:
        with h5py.File(source, "r") as h5:
            group = None
            for key in h5:
                node = h5[key]
                if isinstance(node, h5py.Group) and "indptr" in node:
                    group = node
                    break
            if group is None:
                raise ValueError(f"no matrix group in {source}")
            shape = [int(v) for v in group["shape"][:]]
            n_genes, n_cells = shape
            if n_cells != BRAIN_CELLS:
                raise ValueError(f"expected {BRAIN_CELLS} cells, found {n_cells}")
            barcodes = group["barcodes"][:].astype(str)
            if "features" in group:
                gene_ids = group["features"]["id"][:].astype(str)
                gene_names = group["features"]["name"][:].astype(str)
            else:
                gene_ids = group["genes"][:].astype(str)
                gene_names = group["gene_names"][:].astype(str)
            data = group["data"]
            indices = group["indices"]
            indptr = group["indptr"]
            nnz = int(data.shape[0])
            temporary = destination.with_suffix(".part")
            with h5py.File(temporary, "w") as out:
                out.attrs["encoding-type"] = "anndata"
                out.attrs["encoding-version"] = "0.1.0"
                matrix = out.create_group("X")
                matrix.attrs["encoding-type"] = "csr_matrix"
                matrix.attrs["encoding-version"] = "0.1.0"
                matrix.attrs["shape"] = np.asarray([n_cells, n_genes], dtype=np.int64)
                out_data = _create_numeric_dataset(matrix, "data", nnz, np.dtype(np.float32))
                out_indices = _create_numeric_dataset(matrix, "indices", nnz, np.dtype(np.int64))
                out_indptr = _create_numeric_dataset(
                    matrix, "indptr", n_cells + 1, np.dtype(np.int64)
                )
                step = 50000000
                for start in range(0, nnz, step):
                    end = min(start + step, nnz)
                    out_data[start:end] = data[start:end].astype(np.float32)
                    out_indices[start:end] = indices[start:end].astype(np.int64)
                out_indptr[:] = indptr[:].astype(np.int64)
                obs = _create_dataframe_group(out, "obs", ())
                _create_string_dataset(obs, "_index", n_cells)[:] = barcodes
                var = _create_dataframe_group(out, "var", ("feature_name",))
                _create_string_dataset(var, "_index", n_genes)[:] = gene_ids
                _create_string_dataset(var, "feature_name", n_genes)[:] = gene_names
            os.replace(temporary, destination)
            return {
                "nCells": n_cells,
                "nGenes": n_genes,
                "nnz": nnz,
                "bytes": destination.stat().st_size,
                "sourceBytes": source.stat().st_size,
            }

    info = steps.run("convert", convert)
    target = payload["datasetUri"]
    if object_exists(target):
        info["upload"] = "existing"
    else:
        steps.run("upload", lambda: upload_file(destination, target))
        info["upload"] = "uploaded"
    result = {
        "experiment": "brain-prepare",
        "datasetUri": target,
        **info,
        "steps": steps.records,
        "finishedAt": time.time(),
    }
    put_json(payload["outputUri"], result)
    return result


def _restart_guard(payload: dict) -> dict | None:
    from experiments.runtime.object_io import get_json, object_exists, put_json

    uri = payload["outputUri"]
    if not object_exists(uri):
        return None
    previous = get_json(uri)
    if previous.get("status") != "attempting":
        return None
    restarts = int(previous.get("restarts") or 0) + 1
    if restarts < 2:
        payload["restarts"] = restarts
        return None
    step = previous.get("currentStep")
    put_json(
        uri,
        {
            **previous,
            "status": "killed",
            "restarts": restarts,
            "failedStep": step,
            "error": f"container killed during {step!r}; Modal restarted the input {restarts} times",
        },
    )
    return {"status": "killed", "failedStep": step, "restarts": restarts}


def scanpy_baseline_job(payload: dict) -> dict:
    import os
    import time
    import traceback

    import h5py
    import numpy as np

    os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    from experiments.runtime.metrics import ResourceSampler
    from experiments.runtime.object_io import download_file, put_json

    ended = _restart_guard(payload)
    if ended is not None:
        return ended
    work = Path("/tmp/scanpy")
    work.mkdir(parents=True, exist_ok=True)
    n = int(payload["size"])
    h5ad = work / f"{n}.h5ad"
    sampler = ResourceSampler(sampleIntervalSeconds=0.5, resetCgroupPeak=False)
    sampler.start()
    started = time.perf_counter()
    status, failed, error = ("ok", None, None)
    summary: dict = {}

    def progress() -> None:
        put_json(
            payload["outputUri"],
            {
                "experiment": "scanpy",
                "size": n,
                "replicate": payload.get("replicate"),
                "container": {
                    "cpu": payload["containerCpu"],
                    "memoryMb": payload["containerMemoryMb"],
                    "workers": payload["workers"],
                },
                "status": "attempting",
                "currentStep": steps.current,
                "completedSteps": [
                    name for name in steps.order if "seconds" in steps.records[name]
                ],
                "steps": steps.records,
                "elapsedSeconds": time.perf_counter() - started,
                "peakRssSoFarBytes": getattr(sampler, "_processTreePeakRssBytes", None),
                "peakCgroupSoFarBytes": getattr(sampler, "_cgroupMemoryCurrentPeakBytes", None),
                "restarts": payload.get("restarts", 0),
                **summary,
                "startedAt": started_at,
            },
        )

    started_at = time.time()
    steps = _Steps(on_change=progress)
    try:
        steps.run("download", lambda: download_file(payload["datasetUri"], h5ad))
        with h5py.File(h5ad, "r") as h5:
            nnz = int(h5["X"]["data"].shape[0])
            cells = int(h5["X"].attrs["shape"][0])
        sparse_bytes = nnz * (4 + 4) + (cells + 1) * 8
        summary["sparseMatrixBytesEstimate"] = sparse_bytes
        summary["containerMemoryBytes"] = int(payload["containerMemoryMb"]) * 1024 * 1024
        progress()
        import anndata as ad
        import scanpy as sc

        sc.settings.n_jobs = int(payload["workers"])
        adata = steps.run("read", lambda: ad.read_h5ad(h5ad))
        names = adata.var["feature_name"].astype(str)
        adata.var["mt"] = names.str.upper().str.startswith("MT-").to_numpy()
        adata.var["ribo"] = names.str.upper().str.match("^RP[SL]").to_numpy()

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

        steps.run("filterCells", filter_cells)

        def normalize() -> None:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)

        steps.run("runNormalization", normalize)
        steps.run(
            "markHvgs",
            lambda: sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat"),
        )
        holder: dict = {}

        def pca() -> None:
            hv = adata[:, adata.var["highly_variable"].to_numpy()].copy()
            sc.pp.scale(hv, max_value=10)
            sc.pp.pca(hv, n_comps=21, random_state=4466)
            holder["hv"] = hv

        steps.run("runPca", pca)
        hv = holder["hv"]
        steps.run(
            "queryNeighbors",
            lambda: sc.pp.neighbors(hv, n_neighbors=11, n_pcs=21, random_state=4466),
        )
        steps.run("runUmap", lambda: sc.tl.umap(hv, maxiter=300, random_state=4444))
        steps.run(
            "runLeiden",
            lambda: sc.tl.leiden(
                hv, resolution=1.0, flavor="igraph", n_iterations=2, random_state=4444
            ),
        )

        def markers() -> None:
            adata.obs["leiden"] = hv.obs["leiden"].to_numpy()
            sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", n_genes=100)

        steps.run("findMarkers", markers)
        summary.update(
            {
                "cellsAfterFilter": int(adata.n_obs),
                "genesAfterFilter": int(adata.n_vars),
                "clusters": int(hv.obs["leiden"].nunique()),
            }
        )
    except BaseException as exc:
        status = "error"
        failed = steps.order[-1] if steps.order else None
        error = "".join(traceback.format_exception(exc))[-4000:]
    measurement = sampler.stop()
    result = {
        "experiment": "scanpy",
        "size": n,
        "replicate": payload.get("replicate"),
        "container": {
            "cpu": payload["containerCpu"],
            "memoryMb": payload["containerMemoryMb"],
            "workers": payload["workers"],
        },
        "status": status,
        "failedStep": failed,
        "error": error,
        "wallSeconds": time.perf_counter() - started,
        "peakRssBytes": measurement.processTreeRssPeakBytes,
        "peakCgroupBytes": measurement.cgroupMemoryCurrentPeakBytes,
        "steps": steps.records,
        "stepOrder": steps.order,
        **summary,
        "versions": {},
    }
    try:
        import scanpy as sc

        result["versions"] = {"scanpy": sc.__version__}
    except Exception:
        pass
    put_json(payload["outputUri"], result)
    return {"status": status, "failedStep": failed, "wallSeconds": result["wallSeconds"]}


def scanpy_dask_job(payload: dict) -> dict:
    import os
    import time
    import traceback

    import numpy as np

    os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    from experiments.runtime.metrics import ResourceSampler
    from experiments.runtime.object_io import download_file, put_json

    ended = _restart_guard(payload)
    if ended is not None:
        return ended
    work = Path("/tmp/scanpy_dask")
    work.mkdir(parents=True, exist_ok=True)
    n = int(payload["size"])
    h5ad = work / f"{n}.h5ad"
    workers = int(payload["workers"])
    chunk_rows = int(payload.get("chunkRows", 50000))
    sampler = ResourceSampler(sampleIntervalSeconds=0.5, resetCgroupPeak=False)
    sampler.start()
    started = time.perf_counter()
    started_at = time.time()
    status, failed, error = ("ok", None, None)
    summary: dict = {}

    def progress() -> None:
        put_json(
            payload["outputUri"],
            {
                "experiment": "scanpyDask",
                "size": n,
                "replicate": payload.get("replicate"),
                "container": {
                    "cpu": payload["containerCpu"],
                    "memoryMb": payload["containerMemoryMb"],
                    "workers": workers,
                    "chunkRows": chunk_rows,
                },
                "status": "attempting",
                "currentStep": steps.current,
                "completedSteps": [
                    name for name in steps.order if "seconds" in steps.records[name]
                ],
                "steps": steps.records,
                "elapsedSeconds": time.perf_counter() - started,
                "peakRssSoFarBytes": getattr(sampler, "_processTreePeakRssBytes", None),
                "restarts": payload.get("restarts", 0),
                **summary,
                "startedAt": started_at,
            },
        )

    steps = _Steps(on_change=progress)
    try:
        steps.run("download", lambda: download_file(payload["datasetUri"], h5ad))
        import importlib

        import anndata as ad
        import dask
        import h5py
        import scanpy as sc
        import scipy.sparse as sp

        summary["versions"] = {
            "scanpy": sc.__version__,
            "anndata": ad.__version__,
            "dask": dask.__version__,
        }
        try:
            from anndata.io import read_elem
        except ImportError:
            from anndata.experimental import read_elem
        read_elem_lazy = None
        for module, name in (
            ("anndata.experimental", "read_elem_lazy"),
            ("anndata.experimental", "read_elem_as_dask"),
            ("anndata.io", "read_elem_lazy"),
        ):
            try:
                read_elem_lazy = getattr(importlib.import_module(module), name)
                summary["lazyReader"] = f"{module}.{name}"
                break
            except ImportError, AttributeError:
                continue
        if read_elem_lazy is None:
            raise ImportError(f"no lazy element reader in anndata {ad.__version__}")
        dask.config.set(scheduler="threads", num_workers=workers)
        sc.settings.n_jobs = workers
        handle = h5py.File(h5ad, "r")

        def read_lazy():
            X = read_elem_lazy(handle["X"], chunks=(chunk_rows, -1))
            obs = read_elem(handle["obs"])
            var = read_elem(handle["var"])
            return ad.AnnData(X=X, obs=obs, var=var)

        adata = steps.run("readLazy", read_lazy)
        summary["chunks"] = [int(c) for c in adata.X.chunks[0]][:3]
        names = adata.var["feature_name"].astype(str)
        mt = np.flatnonzero(names.str.upper().str.startswith("MT-").to_numpy())
        ribo = np.flatnonzero(names.str.upper().str.match("^RP[SL]").to_numpy())

        def cell_stats(block):
            block = sp.csr_matrix(block)
            total = np.asarray(block.sum(axis=1)).ravel()
            n_genes = np.diff(block.indptr).astype(np.float64)
            mt_counts = np.asarray(block[:, mt].sum(axis=1)).ravel()
            ribo_counts = np.asarray(block[:, ribo].sum(axis=1)).ravel()
            return np.stack([total, n_genes, mt_counts, ribo_counts], axis=1)

        def filter_cells():
            stats = adata.X.map_blocks(
                cell_stats,
                dtype=np.float64,
                chunks=(adata.X.chunks[0], (4,)),
                meta=np.empty((0, 4)),
            ).compute()
            total, n_genes = (stats[:, 0], stats[:, 1])
            pct_mt = 100 * stats[:, 2] / np.maximum(total, 1)
            pct_ribo = 100 * stats[:, 3] / np.maximum(total, 1)
            keep = np.ones(adata.n_obs, dtype=bool)
            for values in (total, n_genes, pct_mt, pct_ribo):
                low, high = np.quantile(values, [0.01, 0.99])
                keep &= (values >= low) & (values <= high)
            keep &= n_genes >= 10
            filtered = adata[np.flatnonzero(keep)].copy()

            def gene_cells(block):
                return (
                    np.asarray((sp.csr_matrix(block) > 0).sum(axis=0))
                    .reshape(1, -1)
                    .astype(np.float64)
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

        adata = steps.run("filterCells", filter_cells)
        summary["cellsAfterFilter"] = int(adata.n_obs)
        summary["genesAfterFilter"] = int(adata.n_vars)

        def normalize() -> None:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)

        steps.run("runNormalization", normalize)
        steps.run(
            "markHvgs",
            lambda: sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat"),
        )
        holder: dict = {}

        def pca() -> None:
            hv = adata[:, np.flatnonzero(adata.var["highly_variable"].to_numpy())].copy()
            hv.X = sp.csr_matrix(hv.X.compute())
            sc.pp.scale(hv, max_value=10)
            sc.pp.pca(hv, n_comps=21, random_state=4466)
            holder["hv"] = hv

        steps.run("runPca", pca)
        hv = holder["hv"]
        steps.run(
            "queryNeighbors",
            lambda: sc.pp.neighbors(hv, n_neighbors=11, n_pcs=21, random_state=4466),
        )
        steps.run("runUmap", lambda: sc.tl.umap(hv, maxiter=300, random_state=4444))
        steps.run(
            "runLeiden",
            lambda: sc.tl.leiden(
                hv, resolution=1.0, flavor="igraph", n_iterations=2, random_state=4444
            ),
        )

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

        steps.run("findMarkers", markers)
        summary["clusters"] = int(hv.obs["leiden"].nunique())
        handle.close()
    except BaseException as exc:
        status = "error"
        failed = steps.order[-1] if steps.order else None
        error = "".join(traceback.format_exception(exc))[-4000:]
    measurement = sampler.stop()
    result = {
        "experiment": "scanpyDask",
        "size": n,
        "replicate": payload.get("replicate"),
        "container": {
            "cpu": payload["containerCpu"],
            "memoryMb": payload["containerMemoryMb"],
            "workers": workers,
            "chunkRows": chunk_rows,
        },
        "status": status,
        "failedStep": failed,
        "error": error,
        "wallSeconds": time.perf_counter() - started,
        "peakRssBytes": measurement.processTreeRssPeakBytes,
        "peakCgroupBytes": measurement.cgroupMemoryCurrentPeakBytes,
        "steps": steps.records,
        "stepOrder": steps.order,
        **summary,
    }
    put_json(payload["outputUri"], result)
    return {"status": status, "failedStep": failed, "wallSeconds": result["wallSeconds"]}
