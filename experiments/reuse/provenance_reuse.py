import argparse

import cytearc
import numpy as np
from cytearc.storage import ArtifactRef

from experiments.reuse import DATA, WORK
from experiments.reuse import INPUTS as DATASETS
from experiments.runtime.catalogue_counts import build_store, input_record
from experiments.runtime.common import Stopwatch, configure_cytearc, dir_bytes, write_json
from experiments.runtime.rna_recipe import analysis_record, run_rna

# The stored analysis is CyteArc's default cell filter and complete RNA recipe on the pinned counts.
RUN_LABEL = "kang_reuse"

RUN_KEYS = {
    "analysis_cell_selection": "cells",
    "highly_variable_features": "features",
    "normalized": "normalized",
    "pca": "pca",
    "embedding_initialization": "initialization",
    "ann_index": "ann_index",
    "neighbors": "neighbors",
    "connectivity_map": "graph",
    "umap": "umap",
    "clusters": "clusters",
    "markers": "markers",
}


def lineage_export(ds: cytearc.DataStore, target: ArtifactRef, new_ids: set[str]) -> dict:
    lineage = ds.artifacts.lineage(target)
    graph = getattr(lineage, "graph", None) or getattr(lineage, "_graph")
    nodes = []
    for node, attributes in graph.nodes(data=True):
        ref = node if isinstance(node, ArtifactRef) else attributes.get("ref")
        artifact_id = getattr(ref, "artifact_id", str(node))
        record = {
            "id": str(artifact_id),
            "kind": getattr(ref, "kind", attributes.get("kind")),
            "recomputedInFork": str(artifact_id) in new_ids,
        }
        for key in ("operation", "parameters", "complete"):
            if key in attributes:
                record[key] = attributes[key]
        nodes.append(record)
    edges = [
        {
            "source": str(getattr(a, "artifact_id", a)),
            "target": str(getattr(b, "artifact_id", b)),
            **({"input": d.get("name")} if isinstance(d, dict) and "name" in d else {}),
        }
        for a, b, d in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kang_15K_pbmc_rnaseq")
    parser.add_argument("--k-new", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    configure_cytearc()
    store_path = build_store(
        args.dataset,
        WORK / "provenance" / f"{args.dataset}.zarr",
        downloads=DATASETS,
        workers=args.workers,
    )
    name = args.dataset
    ds = cytearc.DataStore(
        str(store_path), default_assay="RNA", nthreads=args.workers, zarr_mode="r+"
    )
    run = run_rna(ds, label=RUN_LABEL)
    refs = {alias: run[key] for key, alias in RUN_KEYS.items() if key in run}
    report = run.report(format="dict")
    full_run_seconds = {}
    for stage in report.get("stages", []):
        started, finished = (stage.get("startedAtNs"), stage.get("finishedAtNs"))
        if started and finished:
            full_run_seconds[stage.get("stage")] = (finished - started) / 1000000000.0
    before = {ref.artifact_id for ref in ds.artifacts.list(from_assay="RNA")}
    pca_status = ds.artifacts.inspect(refs["pca"])
    ann_status = ds.artifacts.inspect(refs["ann_index"])
    umap_status = ds.artifacts.inspect(refs["umap"])
    leiden_status = ds.artifacts.inspect(refs["clusters"])
    reuse = {}
    with Stopwatch() as t:
        reuse["normalization"] = {
            "reused": ds.features.normalize(refs["cells"], refs["features"]) == refs["normalized"]
        }
    reuse["normalization"]["seconds"] = t.seconds
    with Stopwatch() as t:
        reuse["pca"] = {
            "reused": ds.reduction.pca(
                refs["normalized"], dims=int(pca_status.parameters["dims"]), show_elbow_plot=False
            )
            == refs["pca"]
        }
    reuse["pca"]["seconds"] = t.seconds
    ann_params = {
        k: v
        for k, v in ann_status.parameters.items()
        if k in ("ann_metric", "ann_efc", "ann_ef", "ann_m", "rand_state", "ann_parallel")
    }
    with Stopwatch() as t:
        reuse["annIndex"] = {
            "reused": ds.graph.ann_index(refs["pca"], **ann_params) == refs["ann_index"],
            "parameters": ann_params,
        }
    reuse["annIndex"]["seconds"] = t.seconds
    fork = {}
    with Stopwatch() as t:
        neighbors_new = ds.graph.neighbors(refs["ann_index"], k=args.k_new)
    fork["queryNeighbors"] = t.seconds
    with Stopwatch() as t:
        graph_new = ds.graph.connectivity(neighbors_new)
    fork["buildConnectivityMap"] = t.seconds
    umap_params = {
        k: v
        for k, v in umap_status.parameters.items()
        if k in ("n_epochs", "random_seed", "parallel", "min_dist", "spread")
    }
    with Stopwatch() as t:
        umap_new = ds.embeddings.umap(
            graph_new, refs["initialization"], nthreads=args.workers, **umap_params
        )
    fork["runUmap"] = t.seconds
    leiden_params = {
        k: v
        for k, v in leiden_status.parameters.items()
        if k in ("resolution", "backend", "random_seed")
    }
    with Stopwatch() as t:
        clusters_new = ds.clusters.leiden(graph_new, **leiden_params)
    fork["runLeiden"] = t.seconds
    with Stopwatch() as t:
        all_features = ds.features.snapshot(
            from_assay="RNA", mask=np.ones(ds.RNA.feats.N, dtype=bool)
        )
        markers_new = ds.markers.search(
            clusters_new, from_assay="RNA", features=all_features, nthreads=args.workers
        )
    fork["findMarkers"] = t.seconds
    after = {ref.artifact_id for ref in ds.artifacts.list(from_assay="RNA")}
    new_ids = after - before
    counts_bytes = dir_bytes(store_path / "RNA" / "counts") + dir_bytes(
        store_path / "RNA" / "countsT"
    )
    total_bytes = dir_bytes(store_path)
    payload = {
        "description": "Reuse of upstream artifacts, cost of a neighbour-count fork, lineage export, and store composition.",
        "input": input_record(name),
        "analysis": analysis_record(ds, run),
        "store": {
            "cells": int(ds.cells.N),
            "genes": int(ds.RNA.feats.N),
            "totalBytes": total_bytes,
            "countBytes": counts_bytes,
            "analysisBytes": total_bytes - counts_bytes,
        },
        "reuse": reuse,
        "fork": {
            "kOriginal": int(ds.artifacts.inspect(refs["neighbors"]).parameters.get("k", 0)),
            "kNew": args.k_new,
            "stageSeconds": fork,
            "totalSeconds": sum(fork.values()),
            "newArtifacts": len(new_ids),
            "refs": {
                "neighbors": neighbors_new.to_dict(),
                "graph": graph_new.to_dict(),
                "umap": umap_new.to_dict(),
                "clusters": clusters_new.to_dict(),
                "markers": markers_new.to_dict(),
            },
        },
        "fullRunStageSeconds": full_run_seconds,
        "fullRunTotalSeconds": sum((float(v) for v in full_run_seconds.values() if v is not None)),
        "lineage": lineage_export(ds, markers_new, new_ids),
    }
    write_json(f"artifact_reuse_{name}.json", payload, directory=DATA)
    print("reuse:", {k: v["reused"] for k, v in reuse.items()})
    print(
        "fork seconds:",
        {k: round(v, 1) for k, v in fork.items()},
        "total",
        round(sum(fork.values()), 1),
        "vs full run",
        round(payload["fullRunTotalSeconds"], 1),
    )


if __name__ == "__main__":
    main()
