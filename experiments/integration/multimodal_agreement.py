"""Agreement of multimodal neighbour graphs and partitions with reference cell labels.

Clustering at one resolution mixes method quality with partition granularity,
so every graph is compared at matched cluster numbers: each graph gets a
Leiden resolution sweep, and the resolution that gives each target cluster
number is found by bisection. k-NN purity, the fraction of a cell's neighbours
that share its label, scores the neighbour sets without clustering.

``evaluate`` reads the graph export of one ``multimodal_neighbors.run`` and,
when given, the outputs of ``multimodal_neighbors.seurat_workflow`` for the
same cells, and writes the tables below.
"""

import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cytearc.clustering.leiden import leiden_membership
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.runtime.common import configure_cytearc, provenance, write_csv, write_json

SEED = 4466
SWEEP = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
EXTRA_SEEDS = (1, 2)
NEIGHBORS_CONSIDERED = 19
# The cluster-number search halves the resolution down to this value at most.
MINIMUM_RESOLUTION = 1e-4
# Connected components of fewer cells than this join a neighbouring cluster.
SMALL_COMPONENT_CELLS = 10
UNANNOTATED = ("", "nan", "None", "NA")


def scores(labels: np.ndarray, clusters: np.ndarray) -> dict:
    return {
        "clusters": int(len(np.unique(clusters))),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
    }


def same_type_fraction(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Fraction of each cell's neighbours that carry the cell's own label."""
    return (labels[indices] == labels[:, None]).mean(axis=1)


def per_type_f1(labels: np.ndarray, clusters: np.ndarray, protein_defined=()) -> list[dict]:
    """Majority-label F1 per type, with the type's single best cluster.

    Each cluster is assigned its most frequent label; precision and recall of a
    type follow from the union of the clusters assigned to it. The best cluster
    is the one holding most cells of the type.
    """
    table = pd.crosstab(clusters, labels)
    majority = table.idxmax(axis=1)
    predicted = pd.Series(clusters).map(majority).to_numpy()
    rows = []
    for cell_type in table.columns:
        truth = labels == cell_type
        hit = predicted == cell_type
        tp, fp, fn = int((hit & truth).sum()), int((hit & ~truth).sum()), int((~hit & truth).sum())
        column = table[cell_type]
        best = column.idxmax()
        rows.append(
            {
                "cell_type": cell_type,
                "protein_defined_subset": cell_type in protein_defined,
                "type_cells": int(truth.sum()),
                "clusters_with_type_majority": int((majority == cell_type).sum()),
                "precision_fraction": tp / (tp + fp) if tp + fp else 0.0,
                "recall_fraction": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                "best_cluster_size_cells": int(table.loc[best].sum()),
                "best_cluster_recall_fraction": float(column[best] / truth.sum()),
                "best_cluster_purity_fraction": float(column[best] / table.loc[best].sum()),
            }
        )
    return rows


def small_component_assignments(
    graph: csr_matrix, neighbor_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cells in small connected components, and the nearest neighbour of each outside them.

    Seurat prunes weak shared-neighbour edges, so its wsnn graph cuts off
    single cells and small groups (at 600,000 cells: 1,236 single cells and
    53 groups of two to four). Leiden gives each such component clusters of
    its own, which no resolution can merge, so the cluster number cannot reach
    small targets. Cells in components of fewer than ``SMALL_COMPONENT_CELLS``
    cells take the cluster of their nearest neighbour in a larger component,
    read from the method's own neighbour rows.
    """
    _, component = connected_components(graph, directed=False)
    small = np.bincount(component)[component] < SMALL_COMPONENT_CELLS
    detached = np.flatnonzero(small)
    rows = neighbor_rows[detached]
    usable = ~small[rows]
    if len(detached) and not usable.any(axis=1).all():
        stranded = detached[~usable.any(axis=1)]
        raise ValueError(
            f"{len(stranded)} cells in components of fewer than {SMALL_COMPONENT_CELLS} cells "
            f"have no neighbour in a larger component among their {neighbor_rows.shape[1]} "
            f"nearest neighbours, for example cell {int(stranded[0])}"
        )
    nearest = rows[np.arange(len(detached)), np.argmax(usable, axis=1)]
    return detached, nearest


class ResolutionSearch:
    """Leiden partitions of one graph, cached by backend, resolution and seed.

    With ``neighbor_rows``, cells in small connected components join the
    cluster of their nearest neighbour outside them (see
    ``small_component_assignments``).
    """

    def __init__(
        self, graph: csr_matrix, threads: int | None = None, neighbor_rows: np.ndarray | None = None
    ):
        self.graph = graph
        self.threads = threads
        self.cache: dict[tuple[str, float, int], np.ndarray] = {}
        self.seconds = 0.0
        empty = np.empty(0, dtype=np.int64)
        self.detached, self.nearest = (
            (empty, empty)
            if neighbor_rows is None
            else small_component_assignments(graph, neighbor_rows)
        )

    def __call__(self, resolution: float, seed: int = SEED, backend: str = "cytearc") -> np.ndarray:
        key = (backend, float(resolution), seed)
        if key not in self.cache:
            started = time.perf_counter()
            membership = np.asarray(
                leiden_membership(
                    self.graph,
                    float(resolution),
                    seed,
                    backend=backend,
                    nthreads=self.threads if backend == "cytearc" else None,
                )
            )
            if len(self.detached):
                membership = membership.copy()
                membership[self.detached] = membership[self.nearest]
                membership = np.unique(membership, return_inverse=True)[1]
            self.cache[key] = membership
            self.seconds += time.perf_counter() - started
        return self.cache[key]

    def count(self, resolution: float, backend: str = "cytearc") -> int:
        return int(len(np.unique(self(resolution, backend=backend))))

    def match(self, target: int, backend: str = "cytearc", steps: int = 24) -> float:
        """Resolution whose partition has ``target`` clusters, or the closest found."""
        tried = {
            r: self.count(r, backend)
            for (b, r, s) in list(self.cache)
            if b == backend and s == SEED
        }
        if not tried:
            tried = {r: self.count(r, backend) for r in (0.5, 1.0, 2.0)}
        while max(tried.values()) < target:
            top = max(tried)
            tried[top * 1.5] = self.count(top * 1.5, backend)
        while min(tried.values()) > target and min(tried) > MINIMUM_RESOLUTION:
            bottom = min(tried) / 2
            tried[bottom] = self.count(bottom, backend)
        for resolution, clusters in sorted(tried.items()):
            if clusters == target:
                return resolution
        # Cluster number is not strictly monotone in resolution, so the bracket
        # starts at the lowest resolution above the target.
        high = min(r for r, c in tried.items() if c > target)
        low = max((r for r, c in tried.items() if c < target and r < high), default=high / 2)
        for _ in range(steps):
            middle = round((low + high) / 2, 6)
            clusters = self.count(middle, backend)
            tried[middle] = clusters
            if clusters == target:
                return middle
            if clusters < target:
                low = middle
            else:
                high = middle
        return min(tried, key=lambda r: (abs(tried[r] - target), r))


def regular_graph(columns: np.ndarray, weights: np.ndarray) -> csr_matrix:
    n = columns.shape[0]
    rows = np.repeat(np.arange(n), columns.shape[1])
    return csr_matrix((weights.ravel(), (rows, columns.ravel().astype(np.int64))), shape=(n, n))


def seurat_graph(folder: Path, name: str, cells: int) -> csr_matrix:
    """One Seurat graph from its exported dgCMatrix slots."""
    p = np.fromfile(folder / f"{name}_p.bin", dtype=np.int32)
    i = np.fromfile(folder / f"{name}_i.bin", dtype=np.int32)
    x = np.fromfile(folder / f"{name}_x.bin", dtype=np.float32)
    if len(p) != cells + 1 or len(i) != len(x) or p[-1] != len(i):
        raise ValueError(f"Seurat graph {name} slots are inconsistent with {cells} cells")
    return csc_matrix((x, i, p), shape=(cells, cells)).tocsr()


def self_free_rows(rows: np.ndarray, count: int) -> np.ndarray:
    """The first ``count`` entries of each row that are not the cell itself."""
    other = rows != np.arange(rows.shape[0])[:, None]
    if (other.sum(axis=1) < count).any():
        raise ValueError(f"Some cells have fewer than {count} non-self neighbours")
    order = np.argsort(~other, axis=1, kind="stable")
    return np.take_along_axis(rows, order, axis=1)[:, :count]


def knn_rows(graph: csr_matrix) -> np.ndarray:
    """Neighbour rows of a binary k-NN graph whose rows all hold k entries."""
    lengths = np.diff(graph.indptr)
    if lengths.min() != lengths.max():
        raise ValueError("The k-NN graph rows hold different neighbour counts")
    return graph.indices.reshape(graph.shape[0], int(lengths[0]))


def fetch(uri: str, destination: Path) -> Path:
    """Download an s3:// object, or copy a local file for local checks."""
    from experiments.runtime.object_io import download_file

    if uri.startswith("s3://"):
        download_file(uri, destination)
    else:
        destination.write_bytes(Path(uri).read_bytes())
    return destination


def load_inputs(
    payload: dict, work: Path
) -> tuple[dict, dict, dict, np.ndarray, np.ndarray | None]:
    """Neighbour rows, graphs and recorded partitions of every method."""
    path = fetch(payload["graphsUri"], work / "graphs.npz")
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    labels = arrays["labels"]
    groups = arrays.get("groups")
    neighbors = {
        "rna": arrays["RNA_neighbors"],
        "adt": arrays["ADT_neighbors"],
        "cytearc_wnn": arrays["wnn_neighbors"],
    }
    graphs = {
        "rna": regular_graph(arrays["RNA_neighbors"], arrays["RNA_connectivity_weights"]),
        "adt": regular_graph(arrays["ADT_neighbors"], arrays["ADT_connectivity_weights"]),
        "cytearc_wnn": regular_graph(arrays["wnn_neighbors"], arrays["wnn_weights"]),
    }
    partitions = {
        ("rna", 1.0): arrays["RNA_clusters"],
        ("adt", 1.0): arrays["ADT_clusters"],
        ("cytearc_wnn", 1.0): arrays["wnn_clusters"],
    }
    # The graphs of a run that built no SNN fusion graph, such as the Harmony-corrected graphs, lack it.
    if "snn_neighbors" in arrays:
        neighbors["cytearc_snn"] = arrays["snn_neighbors"]
        graphs["cytearc_snn"] = regular_graph(arrays["snn_neighbors"], arrays["snn_weights"])
        partitions[("cytearc_snn", 1.0)] = arrays["snn_clusters"]
    if payload.get("seuratOutputsUri"):
        archive = fetch(payload["seuratOutputsUri"], work / "seurat_outputs.tar.gz")
        folder = work / "seurat_outputs"
        folder.mkdir()
        with tarfile.open(archive) as handle:
            handle.extractall(folder, filter="data")
        cells = len(labels)
        for name, method in (("RNA", "seurat_rna"), ("ADT", "seurat_adt"), ("WNN", "seurat_wnn")):
            partitions[(method, 1.0)] = np.loadtxt(folder / f"{name}_clusters.csv", dtype=np.int64)
        partitions[("seurat_wnn", 1.2)] = np.loadtxt(
            folder / "reused_revision_clusters.csv", dtype=np.int64
        )
        graph_folder = folder / "graphs"
        for name, method in (("RNA", "seurat_rna"), ("ADT", "seurat_adt")):
            neighbors[method] = self_free_rows(
                knn_rows(seurat_graph(graph_folder, f"{name}_nn", cells)), NEIGHBORS_CONSIDERED
            )
            graphs[method] = seurat_graph(graph_folder, f"{name}_snn", cells)
        weighted = (
            np.fromfile(graph_folder / "weighted_nn_indices.bin", dtype=np.int32).reshape(cells, -1)
            - 1
        )
        neighbors["seurat_wnn"] = self_free_rows(weighted, NEIGHBORS_CONSIDERED)
        graphs["seurat_wnn"] = seurat_graph(graph_folder, "wsnn", cells)
    return neighbors, graphs, partitions, labels, groups


def purity_rows(neighbors: dict, labels: np.ndarray, groups, protein_defined) -> list[dict]:
    annotated = ~np.isin(labels, UNANNOTATED)
    types, counts = np.unique(labels[annotated], return_counts=True)
    rows = []
    for method, indices in neighbors.items():
        rows_considered = indices[:, :NEIGHBORS_CONSIDERED].astype(np.int64)
        fraction = same_type_fraction(labels, rows_considered)
        same_group = (
            None if groups is None else (groups[rows_considered] == groups[:, None]).mean(axis=1)
        )
        for cell_type, count in zip(types, counts, strict=True):
            members = labels == cell_type
            rows.append(
                {
                    "method": method,
                    "neighbors_considered": NEIGHBORS_CONSIDERED,
                    "cell_type": cell_type,
                    "protein_defined_subset": cell_type in protein_defined,
                    "type_cells": int(count),
                    "same_type_neighbor_fraction": float(fraction[members].mean()),
                    "cells_with_majority_same_type_fraction": float(
                        (fraction[members] > 0.5).mean()
                    ),
                    "same_group_neighbor_fraction": None
                    if same_group is None
                    else float(same_group[members].mean()),
                }
            )
        rows.append(
            {
                "method": method,
                "neighbors_considered": NEIGHBORS_CONSIDERED,
                "cell_type": "all_annotated_cells",
                "type_cells": int(annotated.sum()),
                "same_type_neighbor_fraction": float(fraction[annotated].mean()),
                "same_group_neighbor_fraction": None
                if same_group is None
                else float(same_group.mean()),
            }
        )
    return rows


def evaluate(payload: dict) -> dict:
    """Score one run's neighbour sets and partitions against its reference labels.

    Payload: ``graphsUri`` (the run's graphs.npz), optional ``seuratOutputsUri``,
    ``proteinDefinedTypes``, ``resultsUri`` (an s3:// prefix for the tables),
    ``threads`` and ``recordUri``.
    """
    from experiments.integration.multimodal_neighbors import persist_measurement
    from experiments.runtime.object_io import upload_file

    configure_cytearc()
    work = Path(payload.get("workDir", "/tmp/multimodal_agreement"))
    work.mkdir(parents=True, exist_ok=True)
    protein_defined = tuple(payload.get("proteinDefinedTypes", ()))
    neighbors, graphs, partitions, labels, groups = load_inputs(payload, work)
    annotated = ~np.isin(labels, UNANNOTATED)
    tables = {"neighbor_purity.csv": purity_rows(neighbors, labels, groups, protein_defined)}
    sweep_rows, matched_rows, f1_rows = [], [], []

    def score(method, rule, target, resolution, seed, partition):
        values = scores(labels[annotated], partition[annotated])
        f1 = per_type_f1(labels[annotated], partition[annotated], protein_defined)
        row = {
            "method": method,
            "partition_rule": rule,
            "target_clusters": target,
            "resolution": resolution,
            "seed": seed,
            **values,
            "clusters_all_cells": int(len(np.unique(partition))),
            "exact_cluster_match": None
            if target is None
            else int(len(np.unique(partition))) == target,
            "mean_f1_all_types": float(np.mean([r["f1"] for r in f1])),
            "mean_f1_protein_defined_types": float(
                np.mean([r["f1"] for r in f1 if r["protein_defined_subset"]])
            )
            if protein_defined
            else None,
        }
        if groups is not None:
            for group in np.unique(groups):
                inside = annotated & (groups == group)
                row[f"ari_{group}"] = float(adjusted_rand_score(labels[inside], partition[inside]))
        matched_rows.append(row)
        if seed == SEED:
            f1_rows.extend(
                {
                    "method": method,
                    "partition_rule": rule,
                    "target_clusters": target,
                    "resolution": resolution,
                    **r,
                }
                for r in f1
            )

    for (method, resolution), partition in partitions.items():
        score(method, "recorded_resolution", None, resolution, SEED, partition)
    targets = sorted(
        {int(len(np.unique(p))) for p in partitions.values()}
        | {int(len(np.unique(labels[annotated])))}
    )
    type_count = int(len(np.unique(labels[annotated])))
    timings = {}
    for method, graph in graphs.items():
        search = ResolutionSearch(graph, payload.get("threads", 8), neighbors[method])
        for resolution in SWEEP:
            partition = search(resolution)
            sweep_rows.append(
                {
                    "method": method,
                    "resolution": resolution,
                    "seed": SEED,
                    **scores(labels[annotated], partition[annotated]),
                    "clusters_with_at_least_100_cells": int((np.bincount(partition) >= 100).sum()),
                }
            )
        for target in targets:
            resolution = search.match(target, steps=16)
            for seed in (SEED, *(EXTRA_SEEDS if target == type_count else ())):
                score(
                    method,
                    "matched_cluster_number",
                    target,
                    resolution,
                    seed,
                    search(resolution, seed),
                )
        timings[method] = {
            "leidenRuns": len(search.cache),
            "leidenSeconds": search.seconds,
            "cellsInSmallComponents": int(len(search.detached)),
        }
        del search
    tables.update(
        {
            "resolution_sweep.csv": sweep_rows,
            "matched_cluster_agreement.csv": matched_rows,
            "per_type_f1.csv": f1_rows,
        }
    )
    result = {
        "status": "ok",
        "graphsUri": payload["graphsUri"],
        "seuratOutputsUri": payload.get("seuratOutputsUri"),
        "cells": int(len(labels)),
        "annotatedCells": int(annotated.sum()),
        "cellTypes": type_count,
        "targetClusters": targets,
        "neighborsConsidered": NEIGHBORS_CONSIDERED,
        "leiden": timings,
        "graphs": {
            "rna": "CyteArc RNA connectivity map",
            "adt": "CyteArc ADT connectivity map",
            "cytearc_wnn": "CyteArc WNN graph",
            "cytearc_snn": "CyteArc SNN fusion graph",
            "seurat_rna": "Seurat RNA_snn graph",
            "seurat_adt": "Seurat ADT_snn graph",
            "seurat_wnn": "Seurat wsnn graph",
        },
        "interpretation": "Every graph, Seurat's included, is partitioned with CyteArc's Leiden for the sweeps; cells in connected components of fewer than 10 cells (leiden.<method>.cellsInSmallComponents) take the cluster of their nearest neighbour in a larger component. recorded_resolution rows keep each tool's own partitions. Neighbour purity uses the first 19 non-self neighbours of every set.",
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
    }
    for name, rows in tables.items():
        path = write_csv(name, rows, directory=work / "tables")
        if payload.get("resultsUri"):
            upload_file(path, f"{payload['resultsUri'].rstrip('/')}/{name}")
    write_json("multimodal_agreement.json", result, directory=work)
    persist_measurement(payload, result)
    result["localTables"] = str(work / "tables")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.payload.read_text())), indent=2))
