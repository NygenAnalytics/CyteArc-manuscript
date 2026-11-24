"""At each size of the COMBAT and Stephenson CITE-seq series, CyteArc's WNN-first run against Seurat's WNN run: the
seconds of the first analysis and of re-clustering the stored WNN graph at a new resolution, how many times faster
CyteArc ran each, and the mean ARI over Leiden seeds of each WNN graph's partition cut to the authors' number of cell
types, from records/multimodal_neighbors/combat_stephenson_pbmc_citeseq/, into
.build/derived/integration/cite_seq_scaling.json.

    uv run --locked python -m experiments.integration.cite_seq_scaling_summary
"""

import csv
from collections import defaultdict

import numpy as np

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

SERIES = (
    ROOT / "experiments/integration/records/multimodal_neighbors/combat_stephenson_pbmc_citeseq"
)
SUMMARY = DERIVED / "integration" / "cite_seq_scaling.json"
SIZES = (100000, 300000, 600000, 1000000, 1457527)
# The WNN graphs whose partitions are compared with the author labels: CyteArc's and Seurat's.
GRAPHS = ("cytearc_wnn", "seurat_wnn")
# The graphs whose agreement within each cohort is compared at the same cluster number: both WNN graphs and CyteArc's
# RNA and protein graphs alone, and each cohort's ARI column in the matched-cluster rows.
COHORT_GRAPHS = ("cytearc_wnn", "seurat_wnn", "rna", "adt")
COHORT_COLUMNS = {"combat": "ari_combat", "stephenson": "ari_stephenson"}


def seurat_phase_seconds(windows: list[dict], phase: str) -> float | None:
    """The seconds of one phase of a Seurat run, the sum of its stage windows (named '<phase>' or '<phase>:<step>'), or
    None when the phase has no window or one of its windows did not finish."""
    found = [w for w in windows if w["stage"].split(":")[0] == phase]
    if not found or any(w.get("unfinished") for w in found):
        return None
    return sum(w["wallSeconds"] for w in found)


def size_times(cells: int, cytearc: dict, seurat: dict) -> dict:
    """CyteArc's and Seurat's first analysis and stored re-clustering seconds on one size, from their run records'
    results, and how many times faster CyteArc ran each where Seurat finished it."""
    if cytearc["status"] != "ok" or cytearc["cells"] != cells or cytearc["jointMethod"] != "wnn":
        raise ValueError(f"CyteArc's record on {cells} cells is not a completed WNN-first run")
    if seurat["cells"] != cells:
        raise ValueError(f"Seurat's record on {cells} cells holds a run on {seurat['cells']}")
    initial = seurat_phase_seconds(seurat["stageWindows"], "initial")
    stored = seurat_phase_seconds(seurat["stageWindows"], "reuse_revision")
    return {
        "cytearc": {
            "initialSeconds": cytearc["initialWnnSeconds"],
            "storedSeconds": cytearc["revisedWnnSeconds"],
        },
        "seurat": {
            "runOutcome": seurat["outcome"],
            "initialSeconds": initial,
            "storedSeconds": stored,
        },
        "initialSpeedup": None if initial is None else initial / cytearc["initialWnnSeconds"],
        "storedSpeedup": None if stored is None else stored / cytearc["revisedWnnSeconds"],
    }


def size_agreement(rows: list[dict], seurat_made_graph: bool) -> dict:
    """The mean ARI over Leiden seeds of each WNN graph's partition cut to the authors' number of cell types, which is
    the one cluster number with several seeds in the matched-cluster rows, and CyteArc's mean minus Seurat's where
    Seurat made a graph; beside them, the mean ARI within each cohort of each graph of COHORT_GRAPHS cut to the same
    number. ``rows`` are one size's matched cluster agreement rows."""
    matched = [
        r for r in rows if r["partition_rule"] == "matched_cluster_number" and r["method"] in GRAPHS
    ]
    seeds = defaultdict(set)
    for r in matched:
        seeds[int(r["target_clusters"])].add(r["seed"])
    seeded = [target for target, found in seeds.items() if len(found) > 1]
    if len(seeded) != 1:
        raise ValueError(
            f"the rows run several seeds at {sorted(seeded)}, not at one cluster number"
        )
    (target,) = seeded
    aris = {
        graph: [
            float(r["ari"])
            for r in matched
            if r["method"] == graph and int(r["target_clusters"]) == target
        ]
        for graph in GRAPHS
    }
    made = {"cytearc_wnn": True, "seurat_wnn": seurat_made_graph}
    for graph, values in aris.items():
        if bool(values) != made[graph] or (values and len(values) != len(seeds[target])):
            raise ValueError(
                f"{graph} has {len(values)} rows at {target} clusters, whose run "
                f"{'made' if made[graph] else 'did not make'} a graph seeded {len(seeds[target])} times"
            )
    means = {graph: float(np.mean(values)) for graph, values in aris.items() if values}
    at_target = [
        r
        for r in rows
        if r["partition_rule"] == "matched_cluster_number" and int(r["target_clusters"]) == target
    ]
    by_cohort = {
        cohort: {
            graph: float(np.mean([float(r[column]) for r in at_target if r["method"] == graph]))
            for graph in COHORT_GRAPHS
            if any(r["method"] == graph for r in at_target)
        }
        for cohort, column in COHORT_COLUMNS.items()
    }
    return {
        "clusters": target,
        "seeds": len(seeds[target]),
        "meanAri": means,
        "meanAriDifference": means["cytearc_wnn"] - means["seurat_wnn"]
        if len(means) == 2
        else None,
        "meanAriByCohort": by_cohort,
    }


def summarise() -> dict:
    sizes = []
    for cells in SIZES:
        folder = SERIES / f"cells_{cells}"
        times = size_times(
            cells,
            load_json(folder / "cytearc_wnn_1.json")["result"],
            load_json(folder / "seurat_1.json")["result"],
        )
        with (folder / "matched_cluster_agreement.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        sizes.append(
            {
                "cells": cells,
                **times,
                **size_agreement(rows, times["seurat"]["initialSeconds"] is not None),
            }
        )
    return {"source": str(SERIES.relative_to(ROOT)), "sizes": sizes}


def main() -> None:
    write_json_at(SUMMARY, summarise())


if __name__ == "__main__":
    main()
