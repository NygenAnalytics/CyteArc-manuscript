"""CyteArc workflow for the comparison: the released package on the same input and container.

Usage: uv run python -m experiments.benchmarks.tools.workflow_cytearc MODE WORKERS BUDGET_BYTES H5AD EVENTS SUMMARY WORKDIR

The steps and parameters are those of the CyteArc benchmark (the workflow section of
experiments/benchmarks/configs/config.toml at the release): the H5AD is converted into a store on the container's
local disk, holding the count matrix by cell and by gene (one step, ``convertMatrix``, as the
public ``H5adToZarr.dump`` does it); cells outside the 1st to 99th percentiles of a Gaussian
fit to total counts, detected genes and mitochondrial and ribosomal fractions are removed, as
are cells with fewer than 10 detected genes; 1,000 variable genes among genes detected in at
least 20 cells; normalisation; 21 principal components; an embedding initialisation from 1,000
centroids; an approximate nearest-neighbour index and 11 neighbours; the connectivity map; UMAP
for 300 epochs; native Leiden at resolution 1.0; and the markers of every cluster on all
genes. CyteArc holds only what fits within BUDGET_BYTES, which the job sets to three quarters of
the container's memory, as the benchmark does (24 of 32 GiB).

MODE is "standard" (the workflow above) or "matched", which replaces CyteArc's own cell filter
with the other tools' filter (tools/workflow_scanpy.py, tools/workflow_r.R), so that every tool
analyses the same cells and genes: cells within the empirical 1st to 99th percentiles (NumPy's
linear quantiles, computed over every cell) of total counts, detected genes and the
percentages of mitochondrial ("^MT-") and ribosomal ("^RP[SL]") counts, matched without regard
to case as the other tools do after upper-casing (mouse "mt-" genes are mitochondrial), with
at least 10 detected genes; and genes detected in at least 20 retained cells. Those genes are
the candidates for variable genes, as in the standard mode, and the genes tested for markers,
instead of all genes. Every other step is the standard one.

Every call is the public API of the installed release; nothing here reads the CyteArc working
tree. Steps are reported to the harness as JSON lines, like the other workflows.

With KEEP_OUTPUTS=1 (the keepOutputs payload setting, the quality comparison), a last step,
keepOutputs, writes the per-cell outputs to WORKDIR/outputs.h5 (tools/kept_outputs.py): every
input cell's identifier, the retained cells, their Leiden clusters, principal components and UMAP
coordinates, read from the stored artifacts. Without it nothing changes.
"""

import json
import os
import platform
import sys
import time

import numpy as np

MODE, WORKERS, BUDGET, H5AD, EVENTS, SUMMARY, WORKDIR = sys.argv[1:8]
WORKERS, BUDGET = (int(WORKERS), int(BUDGET))
SUMMARY_DATA: dict = {}
KEEP_OUTPUTS = os.environ.get("KEEP_OUTPUTS") == "1"
FILTER_ATTRS = ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo"]
MIN_P, MAX_P = (0.01, 0.99)
MIN_FEATURES_PER_CELL = 10
TOP_N, HVG_MIN_CELLS = (1000, 20)
DIMS, K, N_CENTROIDS = (21, 11, 1000)
GRAPH_SEED, KMEANS_SAMPLING, KMEANS_BATCH = (4466, 0.1, 10000)
UMAP_EPOCHS, UMAP_SEED = (300, 4444)
LEIDEN_RESOLUTION, LEIDEN_SEED = (1.0, 4444)
MATCHED_MITO, MATCHED_RIBO = ("^MT-", "^RP[SL]")
MATCHED_MIN_GENES, MATCHED_GENE_MIN_CELLS = (10, 20)


def emit(step: str, event: str, **extra) -> None:
    with open(EVENTS, "a") as handle:
        handle.write(json.dumps({"step": step, "event": event, "t": time.time(), **extra}) + "\n")


def step(name: str, fn):
    from cytearc.storage.execution import execution_report_scope, execution_reports_by_kind

    from experiments.runtime.harness import (
        cgroup_anon_bytes,
        cgroup_memory_current,
        process_tree_rss,
    )

    emit(
        name,
        "start",
        preStepRssBytes=process_tree_rss(os.getpid()),
        preStepCgroupBytes=cgroup_memory_current(),
        preStepAnonBytes=cgroup_anon_bytes(),
    )
    reports = []
    try:
        with execution_report_scope() as reports:
            value = fn()
    finally:
        captured = execution_reports_by_kind(reports)
        SUMMARY_DATA.setdefault("executionReports", {})[name] = captured
    emit(
        name,
        "end",
        executionReports=captured,
        postStepRssBytes=process_tree_rss(os.getpid()),
        postStepCgroupBytes=cgroup_memory_current(),
        postStepAnonBytes=cgroup_anon_bytes(),
    )
    return value


def save_summary() -> None:
    with open(SUMMARY, "w") as handle:
        json.dump(SUMMARY_DATA, handle, default=str)


def matched_filter(ds):
    lows, highs = ([], [])
    for attr in FILTER_ATTRS:
        values = np.asarray(ds.cells.fetch_all(attr), dtype=np.float64)
        if attr.startswith("RNA_percent"):
            values = np.nan_to_num(values, nan=0.0)
        low, high = np.quantile(values, [MIN_P, MAX_P])
        lows.append(float(low))
        highs.append(float(high))
    SUMMARY_DATA["filterBounds"] = {
        attr: [low, high] for attr, low, high in zip(FILTER_ATTRS, lows, highs)
    }
    cells = ds.qc.filter(attrs=FILTER_ATTRS, lows=lows, highs=highs, keep_bounds=True)
    genes = ds.features.detected(cells, from_assay="RNA", min_cells=MATCHED_GENE_MIN_CELLS)
    return (cells, genes)


def keep_outputs(ds, cells, pca, umap, labels) -> None:
    from experiments.benchmarks.tools.kept_outputs import FILE_NAME, write_outputs

    retained = np.asarray(ds.artifacts.load(cells)["values"][:], dtype=bool)
    write_outputs(
        f"{WORKDIR}/{FILE_NAME}",
        workflow="workflow_cytearc.py",
        mode=MODE,
        cell_ids=np.asarray(ds.cells.fetch_all("ids")),
        positions=np.flatnonzero(retained),
        clusters=np.asarray(labels),
        pca=np.asarray(ds.artifacts.load(pca)["data"][:]),
        umap=np.asarray(ds.artifacts.load(umap)["values"][:]),
    )


def main() -> None:
    import cytearc
    import h5py

    if MODE not in ("standard", "matched"):
        raise SystemExit("MODE must be standard or matched")
    stop_after = os.environ.get("STOP_AFTER", "")
    if stop_after not in ("", "runPca"):
        raise ValueError("STOP_AFTER must be runPca when specified")
    if stop_after and KEEP_OUTPUTS:
        raise ValueError("Kept clustering outputs require the complete workflow")
    matched = MODE == "matched"
    SUMMARY_DATA["versions"] = {"cytearc": cytearc.__version__, "python": platform.python_version()}
    SUMMARY_DATA["workers"] = WORKERS
    SUMMARY_DATA["leidenBackend"] = "cytearc"
    SUMMARY_DATA["budgetBytes"] = BUDGET
    SUMMARY_DATA["mode"] = MODE
    with h5py.File(H5AD, "r") as h5:
        SUMMARY_DATA["nnz"] = int(h5["X"]["data"].shape[0])
        shape = h5["X"].attrs["shape"]
        SUMMARY_DATA["nCells"], SUMMARY_DATA["nGenes"] = (int(shape[0]), int(shape[1]))
    save_summary()
    cytearc.configure_output(level="WARNING", progress=False)
    store = f"{WORKDIR}/store.zarr"

    def import_counts(policy) -> None:
        reader = cytearc.H5adReader(
            H5AD,
            matrix_key="X",
            cell_attrs_key="obs",
            cell_ids_key="_index",
            feature_attrs_key="var",
            feature_ids_key="_index",
            feature_name_key="feature_name",
        )
        try:
            cytearc.H5adToZarr(
                reader, store, assay_name="RNA", mem_budget=BUDGET, nthreads=WORKERS, policy=policy
            ).dump()
        finally:
            reader.h5.close()

    def convert() -> None:
        from cytearc.storage.count_matrix import CountMatrixPolicy

        requested = os.environ.get("COUNT_POLICY", "")
        policy = None
        if requested:
            unit, chunk = (int(value) for value in requested.split(","))
            policy = CountMatrixPolicy(unitBytes=unit, chunkBytes=chunk)
        SUMMARY_DATA["countPolicy"] = {"requested": requested or "default"}
        import_counts(policy)
        save_summary()

    step("convertMatrix", convert)
    patterns = {"mito_pattern": MATCHED_MITO, "ribo_pattern": MATCHED_RIBO} if matched else {}
    ds = step(
        "createObject",
        lambda: cytearc.DataStore(
            store,
            assay_types={"RNA": "RNA"},
            default_assay="RNA",
            min_features_per_cell=MATCHED_MIN_GENES - 1 if matched else MIN_FEATURES_PER_CELL,
            nthreads=WORKERS,
            mem_budget=BUDGET,
            **patterns,
        ),
    )
    if matched:
        SUMMARY_DATA["filter"] = "matched"
        cells, genes = step("filterCells", lambda: matched_filter(ds))
        SUMMARY_DATA["cellsAfterFilter"] = int(
            np.count_nonzero(ds.artifacts.load(cells)["values"][:])
        )
        SUMMARY_DATA["genesAfterFilter"] = int(
            np.count_nonzero(ds.artifacts.load(genes)["values"][:])
        )
        save_summary()
    else:
        cells = step(
            "filterCells",
            lambda: ds.qc.auto_filter(
                attrs=FILTER_ATTRS, method="gaussian", min_p=MIN_P, max_p=MAX_P
            ),
        )
    features = step(
        "markHvgs",
        lambda: ds.features.hvgs(
            cells, from_assay="RNA", min_cells=HVG_MIN_CELLS, top_n=TOP_N, show_plot=False
        ),
    )
    normalized = step("runNormalization", lambda: ds.features.normalize(cells, features))
    pca = step(
        "runPca",
        lambda: ds.reduction.pca(normalized, dims=DIMS, local_cache="auto", show_elbow_plot=False),
    )
    SUMMARY_DATA["pcaParameters"] = ds.artifacts.inspect(pca).parameters
    if stop_after == "runPca":
        SUMMARY_DATA.update(
            mode="preprocessing",
            completedThrough="runPca",
            cellsAfterFilter=int(np.count_nonzero(ds.artifacts.load(cells)["values"][:])),
        )
        save_summary()
        return
    initialization = step(
        "buildEmbeddingInitialization",
        lambda: ds.embeddings.initialization(
            pca,
            n_centroids=N_CENTROIDS,
            rand_state=GRAPH_SEED,
            kmeans_sampling=KMEANS_SAMPLING,
            kmeans_batch_size=KMEANS_BATCH,
        ),
    )
    ef = min(100, max(K * 3, 50))
    index = step(
        "buildAnnIndex",
        lambda: ds.graph.ann_index(
            pca,
            ann_efc=ef,
            ann_ef=ef,
            ann_m=min(max(48, int(DIMS * 1.5)), 64),
            ann_parallel=True,
            rand_state=GRAPH_SEED,
        ),
    )
    neighbors = step("queryNeighbors", lambda: ds.graph.neighbors(index, k=K))
    graph = step("buildConnectivityMap", lambda: ds.graph.connectivity(neighbors))
    umap = step(
        "runUmap",
        lambda: ds.embeddings.umap(
            graph,
            initialization,
            n_epochs=UMAP_EPOCHS,
            random_seed=UMAP_SEED,
            parallel=True,
            nthreads=WORKERS,
        ),
    )
    clusters = step(
        "runLeiden",
        lambda: ds.clusters.leiden(
            graph, resolution=LEIDEN_RESOLUTION, backend="cytearc", random_seed=LEIDEN_SEED
        ),
    )

    def markers() -> None:
        tested = (
            genes
            if matched
            else ds.features.snapshot(
                from_assay="RNA", mask=np.ones(ds.get_assay("RNA").feats.N, dtype=bool)
            )
        )
        ds.markers.search(clusters, from_assay="RNA", features=tested, nthreads=WORKERS)

    step("findMarkers", markers)
    labels = ds.artifacts.load(clusters)["values"][:]
    SUMMARY_DATA.update(
        {"cellsAfterFilter": int(labels.size), "clusters": int(np.unique(labels).size)}
    )
    save_summary()
    if KEEP_OUTPUTS:
        step("keepOutputs", lambda: keep_outputs(ds, cells, pca, umap, labels))


if __name__ == "__main__":
    try:
        main()
    finally:
        save_summary()
