import hashlib
import os
import shutil

import cytearc
import numpy as np
from zarr.storage import LocalStore

from experiments.reuse import DATA, INPUTS, WORK
from experiments.runtime.common import catalogue_store, configure_cytearc, fresh_copy, write_json
from experiments.runtime.recording_store import StoreProbe, wrap_recording_store

GROUPS = ("T", "NK", "B", "Mono", "DC", "Other")
FOLD = {
    "cd4naivet": "T",
    "cd4memoryt": "T",
    "cd8t": "T",
    "tactivated": "T",
    "nk": "NK",
    "b": "B",
    "bactivated": "B",
    "cd14mono": "Mono",
    "cd16mono": "Mono",
    "dc": "DC",
    "pdc": "DC",
    "mk": "Other",
    "eryth": "Other",
}
MARKERS = {
    "T": ("CD3D", "CD3E", "IL7R"),
    "NK": ("NKG7", "GNLY", "KLRD1"),
    "B": ("MS4A1", "CD79A", "CD79B"),
    "Mono": ("LYZ", "S100A8", "FCGR3A"),
    "DC": ("FCER1A", "CLEC10A", "LILRA4"),
    "Other": ("PPBP", "PF4", "HBB"),
}
CELLS_PER_GROUP = 8
SEED = 4466
THRESHOLD = 0.6


def fold(labels: np.ndarray) -> np.ndarray:
    keys = [str(v).lower().replace(" ", "") for v in labels]
    return np.asarray([GROUPS.index(FOLD[k]) if k in FOLD else -1 for k in keys], dtype=np.int8)


def tree_bytes(path) -> int:
    return sum((p.stat().st_size for p in path.rglob("*") if p.is_file()), 0)


def tree_digest(path) -> str:
    digest = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            stat = p.stat()
            digest.update(f"{p.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def counts_for(ds, cells: np.ndarray, genes: np.ndarray) -> np.ndarray:
    rows = np.asarray(ds.get_assay("RNA").rawData[np.sort(cells)], dtype=np.float32)
    order = np.argsort(np.argsort(cells))
    return np.log1p(rows[order][:, genes])


def main() -> None:
    configure_cytearc()
    work = WORK / "thumbnails"
    work.mkdir(parents=True, exist_ok=True)
    atlas_path = fresh_copy(
        catalogue_store("kang_15K_pbmc_rnaseq", destination=INPUTS) / "data.zarr",
        work / "atlas.zarr",
    )
    query_path = fresh_copy(
        catalogue_store("tenx_5K_pbmc_rnaseq", destination=INPUTS) / "data.zarr",
        work / "query.zarr",
    )
    out: dict[str, np.ndarray] = {}
    record: dict = {
        "groups": GROUPS,
        "markers": MARKERS,
        "seed": SEED,
        "threshold": THRESHOLD,
        "cytearc": cytearc.__version__,
    }
    rng = np.random.default_rng(SEED)
    ds = cytearc.DataStore(str(atlas_path), default_assay="RNA", nthreads=4, zarr_mode="r+")
    run = ds.pipeline.open(label="docs_default")
    analysis = np.asarray(
        ds.artifacts.load(run["analysis_cell_selection"])["values"][:], dtype=bool
    )
    cells = np.flatnonzero(analysis)
    groups = fold(ds.cells.fetch_all("cluster_labels"))[cells]
    out["group"] = groups
    out["umap"] = np.asarray(ds.artifacts.load(run["umap"])["values"][:], dtype=np.float32)
    out["pca"] = np.asarray(ds.artifacts.load(run["pca"])["data"][:, :2], dtype=np.float32)
    neighbors = np.asarray(ds.artifacts.load(run["neighbors"])["indices"][:], dtype=np.int64)
    sample = np.sort(rng.choice(len(cells), 600, replace=False))
    out["knnSample"], out["knnNeighbors"] = (sample, neighbors[sample, :4])
    record["cells"], record["runNeighborsK"] = (int(len(cells)), int(neighbors.shape[1]))
    names = np.asarray(ds.get_assay("RNA").feats.fetch_all("names")).astype(str)
    gene_index = {name: i for i, name in enumerate(names)}
    marker_genes = [g for group in GROUPS for g in MARKERS[group] if g in gene_index]
    record["heatmapGenes"] = marker_genes
    genes = np.asarray([gene_index[g] for g in marker_genes])
    chosen = np.concatenate(
        [
            rng.choice(np.flatnonzero(groups == k), CELLS_PER_GROUP, replace=False)
            for k in range(len(GROUPS))
        ]
    )
    out["heatmap"] = counts_for(ds, cells[chosen], genes)
    out["heatmapGroup"] = groups[chosen]
    hvg = np.asarray(ds.artifacts.load(run["highly_variable_features"])["values"][:], dtype=bool)
    out["heatmapHvg"] = hvg[genes]
    pca_group = ds.artifacts.load(run["pca"])
    hv_index = np.flatnonzero(hvg)
    rows = [int(np.flatnonzero(hv_index == g)[0]) for g in genes if g in set(hv_index)]
    out["loadings"] = np.asarray(pca_group["loadings"][:], dtype=np.float32)[rows, :8]
    dot_genes = [MARKERS[g][0] if MARKERS[g][0] in gene_index else MARKERS[g][1] for g in GROUPS]
    record["dotGenes"] = dot_genes
    sample_cells = rng.choice(len(cells), min(3000, len(cells)), replace=False)
    values = counts_for(ds, cells[sample_cells], np.asarray([gene_index[g] for g in dot_genes]))
    sample_groups = groups[sample_cells]
    out["dotShare"] = np.asarray(
        [
            [(values[sample_groups == k, j] > 0).mean() for j in range(len(dot_genes))]
            for k in range(len(GROUPS))
        ]
    )
    out["dotMean"] = np.asarray(
        [
            [values[sample_groups == k, j].mean() for j in range(len(dot_genes))]
            for k in range(len(GROUPS))
        ]
    )
    fork = {}
    for k in (11, 15):
        nbrs = ds.graph.neighbors(run["ann_index"], k=k)
        graph = ds.graph.connectivity(nbrs)
        clusters = ds.clusters.leiden(graph, resolution=1.0)
        labels = np.asarray(ds.artifacts.load(clusters)["values"][:], dtype=np.int64)
        out[f"clustersK{k}"] = labels
        out[f"forkEdgesK{k}"] = np.asarray(ds.artifacts.load(nbrs)["indices"][:], dtype=np.int64)[
            sample, :k
        ]
        status = ds.artifacts.inspect(nbrs)
        fork[f"k{k}"] = {
            "clusters": int(np.unique(labels).size),
            "neighbors": {
                "artifactId": nbrs.artifact_id,
                "operation": status.provenance.get("operation"),
                "parameters": status.provenance.get("parameters"),
                "inputs": {
                    name: {"kind": v.get("kind"), "artifactId": v.get("artifact_id")}
                    for name, v in status.provenance.get("inputs", {}).items()
                },
            },
        }
    record["fork"] = fork
    atlas_before = (tree_bytes(atlas_path), tree_digest(atlas_path))
    user_path = work / "user.zarr"
    if user_path.exists():
        shutil.rmtree(user_path)
    user = cytearc.mount_datastore(str(atlas_path), str(user_path), default_assay="RNA", nthreads=4)
    t_mask = np.zeros(user.cells.N, dtype=bool)
    t_mask[cells[groups == GROUPS.index("T")]] = True
    user.cells.insert("tCells", t_mask)
    selection = user.snapshot_cell_selection("tCells")
    features = user.features.hvgs(
        selection, from_assay="RNA", top_n=1000, min_cells=20, show_plot=False
    )
    normalized = user.features.normalize(selection, features)
    pca = user.reduction.pca(normalized, dims=21, show_elbow_plot=False)
    index = user.graph.ann_index(pca, rand_state=SEED)
    graph = user.graph.connectivity(user.graph.neighbors(index, k=11))
    subclusters = np.asarray(
        user.artifacts.load(user.clusters.leiden(graph, resolution=1.0))["values"][:],
        dtype=np.int64,
    )
    out["tSubclusters"] = subclusters
    out["tCells"] = np.flatnonzero(groups == GROUPS.index("T"))
    atlas_after = (tree_bytes(atlas_path), tree_digest(atlas_path))
    record["mount"] = {
        "tCells": int(t_mask.sum()),
        "subclusters": int(np.unique(subclusters).size),
        "userStoreBytes": tree_bytes(user_path),
        "atlasBytes": atlas_before[0],
        "atlasUnchanged": atlas_before == atlas_after,
    }
    probe = StoreProbe()
    reference_store = wrap_recording_store(LocalStore(str(atlas_path)), probe=probe)
    ds_reference = cytearc.DataStore(
        reference_store, default_assay="RNA", nthreads=4, zarr_mode="r+", zarrProfile="fast_local"
    )
    reference_run = ds_reference.pipeline.open(label="docs_default")
    reference_ref = ds_reference.mapping.build_reference(reference_run["neighbors"])
    probe.reset()
    reference = ds_reference.mapping.load_reference(reference_ref)
    ds_query = cytearc.DataStore(str(query_path), default_assay="RNA", nthreads=4, zarr_mode="r+")
    mapping = ds_query.mapping.run(
        reference,
        ds_query.snapshot_cell_selection("I"),
        query_assay="RNA",
        save_k=3,
        missing_feature_policy="reference_mean",
    )
    transferred = ds_query.mapping.load_label_transfer(
        ds_query.mapping.transfer_labels(
            mapping,
            reference=reference,
            reference_labels="cluster_labels",
            threshold_fraction=THRESHOLD,
        )
    ).labels.fillna("NA")
    bytes_read = probe.to_json()["readTransferredBytes"]
    result = ds_query.mapping.load_result(mapping, reference=reference, load_arrays=True)
    placed = out["umap"][np.asarray(result.indices, dtype=np.int64)].mean(axis=1)
    transferred_groups = fold(np.asarray(transferred.to_numpy()).astype(str))
    query_cells = np.flatnonzero(np.asarray(ds_query.cells.fetch_all("I"), dtype=bool))
    out["queryPlaced"] = placed.astype(np.float32)
    out["queryTransferred"] = transferred_groups
    assigned = transferred_groups >= 0
    query_names = np.asarray(ds_query.get_assay("RNA").feats.fetch_all("names")).astype(str)
    query_gene_index = {name: i for i, name in enumerate(query_names)}
    shared = [g for g in marker_genes if g in query_gene_index]
    qchosen = rng.choice(len(query_cells), 18, replace=False)
    out["queryHeatmap"] = counts_for(
        ds_query, query_cells[qchosen], np.asarray([query_gene_index[g] for g in shared])
    )
    record["mapping"] = {
        "queryCells": int(len(query_cells)),
        "query": "tenx_5K_pbmc_rnaseq",
        "assignedShare": float(assigned.mean()),
        "referenceBytesRead": int(bytes_read),
        "referenceStoreBytes": tree_bytes(atlas_path),
        "placedFrom": "mean UMAP position of the three nearest reference cells",
    }
    np.savez_compressed(DATA / "kang_reference_thumbnails.npz", **out)
    write_json(DATA / "kang_reference_thumbnails.json", record, directory=DATA)
    print({k: record[k] for k in ("cells", "fork", "mount", "mapping")})
    if os.environ.get("KEEP_REFERENCE_THUMBNAILS") != "1":
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
