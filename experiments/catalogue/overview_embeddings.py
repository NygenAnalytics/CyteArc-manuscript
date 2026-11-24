"""UMAP thumbnails of Fig. 1 other than the Kang PBMCs: subsamples of two analyses of the tool comparison."""

import glob
import json
from pathlib import Path

import numpy as np
import zarr

from experiments.runtime.common import ROOT

REPO = ROOT
DATA = ROOT / "experiments/catalogue/records"


STORES = REPO / "experiments" / "benchmarks" / "inputs" / "cluster_overlap" / "stores"
RUNS = {
    "embryo": (
        "embryo_rna_scaling_1000000_cells_replicate_1",
        "1000000.zarr",
        "one-million-cell mouse embryo sample",
    ),
    "brain": (
        "mouse_brain_rna_1306127_cells_object_storage_replicate_1",
        "1306127.zarr",
        "1.3 million mouse brain cells",
    ),
}
N, SEED, NODES, K = 5000, 0, 90, 4


def artifact(store: Path, kind: str) -> tuple[np.ndarray, str, dict]:
    """The single artifact of one kind in a store: its values, identifier and provenance."""
    paths = [
        q for q in glob.glob(str(store / "RNA" / "artifacts" / kind / "*")) if Path(q).is_dir()
    ]
    assert len(paths) == 1, (store, kind, paths)
    path = Path(paths[0])
    meta = json.loads((path / "zarr.json").read_text()).get("attributes", {})
    return zarr.open_array(str(path / "values"), mode="r")[:], path.name, meta.get("provenance", {})


def main() -> None:
    out, record = {}, {"seed": SEED, "cells": N, "runs": {}}
    for name, (run, zname, title) in RUNS.items():
        store = STORES / run / zname
        umap, umap_id, umap_prov = artifact(store, "embedding")
        labels, label_id, _ = artifact(store, "cluster_labels")
        assert umap.shape[1] == 2 and len(labels) == len(umap)
        assert umap_prov.get("operation") == "run_umap"
        ids, sizes = np.unique(labels, return_counts=True)
        largest = ids[np.argsort(-sizes)][:6]
        group = np.full(len(labels), -1, dtype=np.int8)
        for i, cluster in enumerate(largest):
            group[labels == cluster] = i
        keep = np.sort(np.random.default_rng(SEED).choice(len(umap), N, replace=False))
        out[f"{name}Umap"] = umap[keep].astype(np.float32)
        out[f"{name}Group"] = group[keep]
        record["runs"][name] = {
            "title": title,
            "store": str(store.relative_to(REPO)),
            "umapArtifact": umap_id,
            "clusterArtifact": label_id,
            "cellsInRun": int(len(umap)),
            "clusters": int(len(ids)),
            "largestSixShare": float(np.isin(labels, largest).mean()),
        }
    # Graph thumbnail: NODES embryo cells, each joined to its K nearest other sampled cells in the embedding.
    xy = out["embryoUmap"]
    nodes = np.random.default_rng(SEED + 1).choice(N, NODES, replace=False)
    dist = np.linalg.norm(xy[nodes][:, None, :] - xy[nodes][None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    out["embryoNodes"] = nodes
    out["embryoEdges"] = np.argsort(dist, axis=1)[:, :K]  # positions within embryoNodes
    np.savez_compressed(DATA / "atlas_embedding_thumbnails.npz", **out)
    (DATA / "atlas_embedding_thumbnails.json").write_text(json.dumps(record, indent=2) + "\n")
    for name, rec in record["runs"].items():
        print(
            f"{name}: {rec['cellsInRun']:,} cells, {rec['clusters']} clusters, six largest {rec['largestSixShare']:.0%}"
        )
    print("wrote experiments/catalogue/records/atlas_embedding_thumbnails.npz and .json")


if __name__ == "__main__":
    main()
