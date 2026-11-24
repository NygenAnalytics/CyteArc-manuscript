"""Prepare the panel data of the supplementary figure on how joint RNA and protein graphs agree with author labels: each
graph of the COMBAT and Stephenson CITE-seq series cut to the authors' number of cell types, at each size.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py. Each function reads
the experiment records, checks what they state against what they list, and raises on a mismatch. Run from the
repository root:

  uv run --locked python -B -m manuscript.figures.multimodal_graph_agreement.prepare
"""

import csv
import json
from collections import defaultdict

import numpy as np
from experiments.runtime.common import ROOT

NEIGHBORS = ROOT / "experiments/integration/records/multimodal_neighbors"
CITE = NEIGHBORS / "combat_stephenson_pbmc_citeseq"
CITE_SIZES = (100000, 300000, 600000, 1000000, 1457527)
REPLICATES = (1, 2, 3)
OUT = ROOT / "manuscript" / "figures" / "multimodal_graph_agreement"

# The graphs the agreement panel follows across sizes: the two joint WNN graphs, CyteArc's SNN fusion graph, and
# CyteArc's single-modality graphs.
GRAPHS = ("cytearc_wnn", "seurat_wnn", "cytearc_snn", "rna", "adt")
METRICS = (
    "ari",
    "ari_combat",
    "ari_stephenson",
    "nmi",
    "mean_f1_all_types",
    "mean_f1_protein_defined_types",
)


def relative(path) -> str:
    return str(path.relative_to(ROOT))


# ---- the COMBAT and Stephenson CITE-seq series ---------------------------------------------------


def agreement_rows(cells: int) -> list[dict]:
    path = CITE / f"cells_{cells}/matched_cluster_agreement.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        for key in METRICS:
            r[key] = float(r[key])
            low = -1.0 if key.startswith("ari") else 0.0
            if not low <= r[key] <= 1.0:
                raise ValueError(f"{relative(path)}: {key} {r[key]} out of range")
        r["clusters"] = int(r["clusters"])
        r["target_clusters"] = int(r["target_clusters"]) if r["target_clusters"] else None
    return rows


def author_types(rows: list[dict]) -> int:
    """The authors' number of cell types: the experiment ran several Leiden seeds only at that number."""
    seeds = defaultdict(set)
    for r in rows:
        if r["partition_rule"] == "matched_cluster_number":
            seeds[r["target_clusters"]].add(r["seed"])
    seeded = [target for target, found in seeds.items() if len(found) > 1]
    if len(seeded) != 1:
        raise ValueError(f"the agreement table runs several seeds at {seeded}, not at one number")
    return seeded[0]


def summary(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "low": min(values),
        "high": max(values),
    }


def cite_agreement_by_size() -> dict:
    """At each size, every metric of each graph's partition cut to the authors' number of cell types, for each
    Leiden seed, with the seeds' mean and range and the clusters each seed gave."""
    sizes, types_seen = [], set()
    for n in CITE_SIZES:
        rows = agreement_rows(n)
        types = author_types(rows)
        types_seen.add(types)
        graphs = {}
        for graph in GRAPHS:
            runs = sorted(
                (
                    r
                    for r in rows
                    if r["method"] == graph
                    and r["partition_rule"] == "matched_cluster_number"
                    and r["target_clusters"] == types
                ),
                key=lambda r: r["seed"],
            )
            if not runs:
                if graph.startswith("seurat"):
                    continue
                raise ValueError(f"{graph} has no partition at {types} clusters on {n} cells")
            if len(runs) != len(REPLICATES):
                raise ValueError(f"{graph} at {n} cells has {len(runs)} seeds, not three")
            graphs[graph] = {
                "seeds": [int(r["seed"]) for r in runs],
                "clusters": [r["clusters"] for r in runs],
                **{key: summary([r[key] for r in runs]) for key in METRICS},
            }
        sizes.append({"cells": n, "graphs": graphs})
    if len(types_seen) != 1:
        raise ValueError(f"the authors' number of types differs between sizes: {types_seen}")
    return {
        "cellTypes": types_seen.pop(),
        "sizes": sizes,
        "source": [relative(CITE / f"cells_{n}/matched_cluster_agreement.csv") for n in CITE_SIZES],
    }


PANELS = (cite_agreement_by_size,)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        data = panel()
        path = OUT / f"{panel.__name__}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {relative(path)}")


if __name__ == "__main__":
    main()
