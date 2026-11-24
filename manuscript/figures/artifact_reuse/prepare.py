"""Prepare the panel data of the figure on reusing stored artifacts: branching an analysis, mapping onto a finished
analysis used as a reference, and joining modalities.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py. Run from the
repository root:

  uv run --locked python -B -m manuscript.figures.artifact_reuse.prepare
"""

import csv
import json
from collections import defaultdict

import numpy as np
from experiments.integration.t_cell_reference_summary import subtype_transfer
from experiments.runtime.common import DERIVED, ROOT, load_json
from manuscript.figures import blood_store_annotation as in_place
from manuscript.figures.multimodal_graph_measurements import prepare as graph_measurements

REUSE = ROOT / "experiments/reuse/records"
MAPPING = ROOT / "experiments/mapping/records"
MULTIMODAL = (
    ROOT / "experiments/integration/records/multimodal_neighbors/combat_stephenson_pbmc_citeseq"
)
MOUNTED_EMBRYO = REUSE / "embryo_endothelial_reanalysis.json"
# The published analysis the mounted store holds: replicate 3 of the ten-million-cell embryo series, on 16 CPUs.
PUBLISHED_RUN = ROOT / "experiments/benchmarks/records/raw/rna_scaling_runs.json"
PUBLISHED_TAG = "embryo_rna_scaling_10000000_cells_replicate_6"
LADDER = MAPPING / "reference_ladder.json"
LADDER_SUMMARY = DERIVED / "mapping" / "reference_ladder.json"
# The ladder reference's cells in the UMAP of CyteArc's Harmony analysis of all ten blood studies, binned by cell type
# by mapping.reference_umap_rasters from the per-cell arrays of that analysis; the .json record beside it holds the
# reference studies and the cells per type and per study.
LADDER_REFERENCE_RASTER = MAPPING / "ladder_reference_umap_raster.npz"
BLOOD_COHORT = REUSE / "blood_cohort_analysis.json"
CROSS_STUDY = MAPPING / "cross_study/evaluation.json"
MULTIMODAL_SIZES = (100000, 300000, 600000, 1000000, 1457527)
# The RNA, protein and WNN graphs of the 1,457,527-cell WNN-first run, after Harmony corrected each modality's PCA for
# study, laid out by integration.multimodal_neighbors's graph_umaps and binned by author label by
# integration.collect_cite_seq_jobs, which writes the records: the job's record, summary.json with the grid, the cells
# per label and the job's check of its rebuilt graphs against the run's, and each graph's raster_<graph>.npz.
GRAPH_UMAPS = ROOT / "experiments/integration/records/graph_umaps_harmony"
GRAPH_UMAP_GRAPHS = ("RNA", "ADT", "WNN")
OUT = ROOT / "manuscript" / "figures" / "artifact_reuse"

GIB = 1024**3


def relative(path) -> str:
    return str(path.relative_to(ROOT))


def point_summary(values: list[float]) -> dict:
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("figure summaries require completed finite observations")
    return {
        "mean": float(np.mean(values)),
        "low": float(min(values)),
        "high": float(max(values)),
        "n": len(values),
    }


# ---- pipeline runs as metro lines, from the published analysis of the mounted embryo store ------------


def _descends(edges: set, ancestor: str, descendant: str) -> bool:
    """Whether the lineage leads from ancestor to descendant: a drawn edge may pass over artifacts the figure leaves
    out, such as a gene summary between a cell selection and its variable genes."""
    children = defaultdict(set)
    for source, target in edges:
        children[source].add(target)
    frontier, seen = [ancestor], set()
    while frontier:
        node = frontier.pop()
        if node == descendant:
            return True
        if node not in seen:
            seen.add(node)
            frontier.extend(children[node])
    return False


# The steps every run of the whole atlas shares, from the selected cells to the neighbor index: each artifact's alias,
# its operation in the lineage, and the published run's stages that made it.
SHARED_CHAIN = (
    ("C1", "auto_filter_cells", ("filterCells",)),
    ("V1", "select_hvgs", ("markHvgs",)),
    ("N1", "run_normalization", ("runNormalization",)),
    ("P1", "run_pca", ("runPca",)),
    ("I1", "build_ann_index", ("buildAnnIndex",)),
)
# The new neighbor count of the job that reused the published analysis, whose neighbors the panel's card opens.
FORKED_K = 21


def provenance_runs() -> dict:
    """The mounted ten-million-cell embryo store's analyses of the whole atlas as metro lines: the published analysis
    (11 neighbors, Leiden at resolution 1), and the clusterings a later job made from its stored
    artifacts, Leiden at resolutions 0.5 and 2 on the published graph and 21 neighbors from the published neighbor
    index. Each artifact carries the seconds the run that made it spent on it: the published run's stages, on 16 CPUs,
    and the later job's steps, on 8. Every drawn edge is checked against the later job's lineage graph."""
    record = load_json(MOUNTED_EMBRYO)
    reuse = record["reuse"]
    run = load_json(PUBLISHED_RUN)[PUBLISHED_TAG]
    nodes = {n["id"]: n for n in reuse["lineageGraph"]["nodes"]}
    edges = {(e["source"], e["target"]) for e in reuse["lineageGraph"]["edges"]}
    published = {kind: a["artifactId"] for kind, a in reuse["published"].items()}

    def one(operation: str, parent: str | None = None, **parameters) -> str:
        found = [
            d
            for d, n in nodes.items()
            if n["operation"] == operation
            and all(n["parameters"].get(key) == value for key, value in parameters.items())
            and (parent is None or (parent, d) in edges)
        ]
        if len(found) != 1:
            raise ValueError(
                f"the lineage holds {len(found)} {operation} artifacts with {parameters}, not one"
            )
        return found[0]

    def stages(names) -> float:
        return float(sum(run["stages"][name]["seconds"] for name in names))

    def step(name: str) -> float:
        return float(record["steps"][name]["seconds"])

    chain = [
        {"alias": alias, "digest": one(operation), "operation": operation, "seconds": stages(names)}
        for alias, operation, names in SHARED_CHAIN
    ]
    for a, b in zip(chain[:-1], chain[1:]):
        if not _descends(edges, a["digest"], b["digest"]):
            raise ValueError(f"the lineage does not lead from {a['alias']} to {b['alias']}")
    index = chain[-1]["digest"]
    same = reuse["steps"]["sameClusters"]
    if not (same["isPublished"] and same["artifactId"] == published["clusters"]):
        raise ValueError("the published clustering must come back as the published artifact")
    if float(reuse["publishedLeidenParameters"]["resolution"]) != 1.0:
        raise ValueError("the published clustering must be Leiden at resolution 1")
    k11 = one("query_neighbors", index, k=11)
    if k11 != published["neighbors"] or (k11, published["graph"]) not in edges:
        raise ValueError(
            "the published neighbors and graph must be the k = 11 branch of the published index"
        )
    k21 = one("query_neighbors", index, k=FORKED_K)
    g21 = one("build_connectivity_map", k21)
    branches = [
        {
            "k": 11,
            "published": True,
            "neighbors": {"alias": "K1", "digest": k11, "seconds": stages(("queryNeighbors",))},
            "graph": {
                "alias": "G1",
                "digest": published["graph"],
                "seconds": stages(("buildConnectivityMap",)),
            },
            "leaves": [
                {
                    "resolution": 0.5,
                    "digest": one("run_leiden_clustering", published["graph"], resolution=0.5),
                    "seconds": step("reuse.leiden.r0.5"),
                    "step": "reuse.leiden.r0.5",
                    "published": False,
                },
                {
                    "resolution": 1.0,
                    "digest": published["clusters"],
                    "seconds": stages(("runLeiden",)),
                    "step": None,
                    "published": True,
                },
                {
                    "resolution": 2.0,
                    "digest": one("run_leiden_clustering", published["graph"], resolution=2.0),
                    "seconds": step("reuse.leiden.r2"),
                    "step": "reuse.leiden.r2",
                    "published": False,
                },
            ],
        },
        {
            "k": FORKED_K,
            "published": False,
            "neighbors": {"alias": "K2", "digest": k21, "seconds": step("reuse.neighbors.k21")},
            "graph": {"alias": "G2", "digest": g21, "seconds": step("reuse.graph.k21")},
            "leaves": [
                {
                    "resolution": 1.0,
                    "digest": one("run_leiden_clustering", g21, resolution=1.0),
                    "seconds": step(f"reuse.leiden.k{FORKED_K}_r1"),
                    "step": f"reuse.leiden.k{FORKED_K}_r1",
                    "published": False,
                }
            ],
        },
    ]
    shared = sum(a["seconds"] for a in chain)
    # The runs take their letters, and their clusterings their numbers, in the order they ran: the published run
    # first, then the later job's runs in the order of their Leiden steps. The user store only grows, so its size
    # before each step orders them.
    leaves = [leaf for branch in branches for leaf in branch["leaves"]]
    later = sorted(
        (leaf for leaf in leaves if not leaf["published"]),
        key=lambda leaf: record["steps"][leaf["step"]]["userStoreBytesBefore"],
    )
    before = [record["steps"][leaf["step"]]["userStoreBytesBefore"] for leaf in later]
    if any(a >= b for a, b in zip(before[:-1], before[1:])):
        raise ValueError("the later job's Leiden steps must each start on a larger user store")
    for number, leaf in enumerate(
        [*(leaf for leaf in leaves if leaf["published"]), *later], start=1
    ):
        leaf["alias"] = f"L{number}"
        leaf["run"] = chr(64 + number)
    # Each branch lists its runs in letter order, which the panel draws from the top down.
    for branch in branches:
        branch["leaves"].sort(key=lambda leaf: leaf["run"])
    for branch in branches:
        for leaf in branch["leaves"]:
            # What the run spent: the published run made every artifact on its path; a later run made only the
            # artifacts its path does not share with the published analysis.
            own = (
                branch["neighbors"]["seconds"] + branch["graph"]["seconds"]
                if not branch["published"]
                else 0.0
            )
            leaf["runSeconds"] = (
                shared
                + branch["neighbors"]["seconds"]
                + branch["graph"]["seconds"]
                + leaf["seconds"]
                if leaf["published"]
                else own + leaf["seconds"]
            )
    inputs = [a for a in chain if (a["digest"], k21) in edges]
    clustered = {c["cells"] for c in reuse["clusterings"].values()}
    if len(clustered) != 1:
        raise ValueError("the clusterings of the whole atlas must cover the same cells")
    return {
        "atlasCells": int(record["source"]["cells"]),
        "clusteredCells": int(clustered.pop()),
        "chain": chain,
        "branches": branches,
        "record": {
            "alias": "K2",
            "digest": k21,
            "operation": nodes[k21]["operation"],
            "parameters": {"k": FORKED_K},
            "inputs": [{"alias": a["alias"], "digest": a["digest"]} for a in inputs],
        },
        "seconds": {"shared": shared},
        "runs": len(leaves),
        "cpus": {
            "published": int(run["funnel"]["modalResources"]["modalCpuLimit"]),
            "reuse": int(record["container"]["cpu"]),
        },
        "source": [relative(MOUNTED_EMBRYO), relative(PUBLISHED_RUN)],
    }


# ---- the reference ladder -----------------------------------------------------------------------

# Every method the ladder record holds, and the ones the figure shows: CyteArc with an unintegrated reference, Seurat,
# R Symphony and CellTypist trained on variable genes.
LADDER_RECORDED = (
    "unintegrated",
    "integrated",
    "seurat_map",
    "symphony",
    "celltypist_hvg",
    "celltypist",
)
LADDER_METHODS = ("unintegrated", "seurat_map", "symphony", "celltypist_hvg")
LADDER_PARTS = ("reference_500000", "reference_1500000", "reference_3000000", "reference_all")
# R's dgCMatrix indexes at most this many nonzero values.
R_NONZERO_LIMIT = 2**31 - 1


def _ladder_rows() -> dict:
    rows = {(r["referencePart"], r["method"]): r for r in load_json(LADDER)["rows"]}
    if set(rows) != {(p, m) for p in LADDER_PARTS for m in LADDER_RECORDED}:
        raise ValueError("the ladder must hold one run per reference size and method")
    return rows


def _outcome(row: dict) -> str:
    outcome = row["outcome"]
    if outcome == "completed":
        return "pass"
    if outcome == "out of memory":
        return "oom"
    if outcome == "not attempted":
        return "none"
    if "nonzero limit" in outcome:
        return "limit"
    raise ValueError(f"unknown ladder outcome {outcome!r}")


def ladder_sizes() -> list[dict]:
    rows = _ladder_rows()
    sizes = []
    for part in LADDER_PARTS:
        row = rows[(part, "unintegrated")]
        sizes.append(
            {
                "part": part,
                "cells": int(row["referenceCells"]),
                "nonzeros": int(row["referenceNonzeroCounts"]),
                "overRLimit": int(row["referenceNonzeroCounts"]) > R_NONZERO_LIMIT,
            }
        )
    return sizes


def ladder_memory() -> dict:
    """Each method's container peak at each reference size, with how each run ended."""
    rows = _ladder_rows()
    methods = {}
    for method in LADDER_METHODS:
        runs = []
        for part in LADDER_PARTS:
            row = rows[(part, method)]
            run = {"part": part, "outcome": _outcome(row), "failedStep": row.get("failedStep")}
            if run["outcome"] in ("pass", "oom"):
                run["peakGiB"] = row["peakContainerBytes"] / GIB
            runs.append(run)
        methods[method] = runs
    return {
        "sizes": ladder_sizes(),
        "methods": methods,
        "rNonzeroLimit": R_NONZERO_LIMIT,
        "source": [relative(LADDER)],
    }


def ladder_time() -> dict:
    """Each completed run's time to build the reference and map the query onto it: each run built its reference
    afresh, so its memory peak, which the ladder's memory panel draws, covers the same steps."""
    summary = {(r["referencePart"], r["method"]): r for r in load_json(LADDER_SUMMARY)["rows"]}
    rows = _ladder_rows()
    methods = {}
    for method in LADDER_METHODS:
        methods[method] = [
            {
                "part": part,
                "seconds": float(summary[(part, method)]["freshReferenceAndQuerySeconds"]),
            }
            for part in LADDER_PARTS
            if _outcome(rows[(part, method)]) == "pass"
        ]
    return {
        "sizes": ladder_sizes(),
        "methods": methods,
        # Every run maps the same query cells, which the panel names.
        "queryCells": int(load_json(LADDER)["queryCells"]),
        "source": [relative(LADDER_SUMMARY), relative(LADDER)],
    }


def _disagreement(accuracy: float, interval: list[float]) -> dict:
    """An agreement with the author labels as disagreement, the fraction of cells whose transferred label differs from
    the author label, with the agreement's donor-bootstrap interval turned around."""
    low, high = (float(v) for v in interval)
    if not low <= accuracy <= high:
        raise ValueError(f"agreement {accuracy} lies outside its interval [{low}, {high}]")
    return {"disagreement": 1.0 - float(accuracy), "low": 1.0 - high, "high": 1.0 - low}


def ladder_reference_umap() -> dict:
    """The ladder reference's cells in the UMAP of CyteArc's Harmony analysis of all ten blood studies, binned by cell
    type, with the sizes of the ladder's whole reference and its query. Checks the raster against its summary, the
    ladder's design and the cohort analysis record."""
    summary = load_json(LADDER_REFERENCE_RASTER.with_suffix(".json"))
    ladder = load_json(LADDER)
    design = ladder["design"]
    cohort = load_json(BLOOD_COHORT)["result"]["sourceCellCounts"]
    if summary["referenceStudies"] != design["referenceStudies"]:
        raise ValueError("the raster's studies are not the ladder's reference studies")
    if any(cells > cohort[study] for study, cells in summary["cellsPerStudy"].items()):
        raise ValueError("the raster holds more cells of a study than the cohort analysis")
    with np.load(LADDER_REFERENCE_RASTER, allow_pickle=False) as raster:
        types = [str(t) for t in raster["types"]]
        per_type = raster["typeCells"].sum(axis=(1, 2))
    if (
        types != ladder["vocabulary"]
        or {t: int(n) for t, n in zip(types, per_type)} != summary["cellsPerType"]
    ):
        raise ValueError(
            "the raster's cells per type differ from its summary or the ladder's vocabulary"
        )
    rows = _ladder_rows()
    return {
        "raster": relative(LADDER_REFERENCE_RASTER),
        "types": [{"type": t, "cells": int(n)} for t, n in zip(types, per_type)],
        "rasterCells": int(per_type.sum()),
        "reference": {
            "cells": int(rows[("reference_all", "unintegrated")]["referenceCells"]),
            "studies": len(design["referenceStudies"]),
        },
        "query": {"cells": int(ladder["queryCells"]), "studies": len(design["queryStudies"])},
        "source": [relative(LADDER), relative(BLOOD_COHORT), relative(LADDER_REFERENCE_RASTER)],
    }


def ladder_accuracy() -> dict:
    """Disagreement of the transferred labels with the author labels of the ladder's query at each reference size, with
    donor-bootstrap 95% intervals, and how every run ended, for the outcome matrix under the panel."""
    summary = {(r["referencePart"], r["method"]): r for r in load_json(LADDER_SUMMARY)["rows"]}
    rows = _ladder_rows()
    methods = {}
    for method in LADDER_METHODS:
        methods[method] = [
            {
                "part": part,
                **_disagreement(
                    summary[(part, method)]["accuracy"],
                    summary[(part, method)]["donorBootstrap95CI"]["accuracy"],
                ),
            }
            for part in LADDER_PARTS
            if _outcome(rows[(part, method)]) == "pass"
        ]
    outcomes = {
        method: [{"part": part, "outcome": _outcome(rows[(part, method)])} for part in LADDER_PARTS]
        for method in LADDER_METHODS
    }
    return {
        "sizes": ladder_sizes(),
        "methods": methods,
        "outcomes": outcomes,
        "source": [relative(LADDER_SUMMARY), relative(LADDER)],
    }


# The held-out query study of each tissue beyond blood: its key in the evaluation and the first author of the publication
# it comes from. The HLCA calls the lung query Meyer_2019; its data were published by Madissoon et al. (2019).
HELD_OUT_QUERIES = {"kidney": ("Reck_2025", "Reck"), "lung": ("Meyer_2019", "Madissoon")}
# Each held-out reference's UMAP, binned by harmonized label by mapping.reference_umap_rasters from the coordinates that
# mapping.cross_study_mapping.reference_umap saved after rebuilding the reference; the .json record beside it holds the
# cells, the cells per label and the job's check that the rebuild ran the benchmark's CyteArc version and configuration
# on the same cells.
HELD_OUT_UMAP = MAPPING / "held_out_reference_umap"
# The methods the panels show. On these studies CellTypist ran only trained on all genes, unlike the ladder, where the
# figure shows it trained on variable genes.
HELD_OUT_METHODS = ("unintegrated", "seurat_map", "symphony", "celltypist")


def _held_out_high(tissue: str) -> float:
    """The highest disagreement or interval end of any method on the tissue's held-out study."""
    study, _ = HELD_OUT_QUERIES[tissue]
    highs = []
    for m in load_json(CROSS_STUDY)["result"]["mapping"]:
        if m["tissue"] == tissue and m["method"] in HELD_OUT_METHODS:
            run = m["evaluation"]["studies"][study]
            highs.append(1.0 - min(run["accuracy"], run["donorBootstrap95CI"]["accuracy"][0]))
    return max(highs)


def _held_out(tissue: str) -> dict:
    """Disagreement with the author labels on one tissue's held-out query study, with donor-bootstrap 95% intervals,
    and the sizes of the tissue's reference and query. Checks that every method was scored on the query's cells."""
    study, author = HELD_OUT_QUERIES[tissue]
    entries = {
        m["method"]: m for m in load_json(CROSS_STUDY)["result"]["mapping"] if m["tissue"] == tissue
    }
    if set(HELD_OUT_METHODS) - set(entries):
        raise ValueError(
            f"evaluation.json lacks {sorted(set(HELD_OUT_METHODS) - set(entries))} on {tissue}"
        )
    parts = entries["unintegrated"]["inputs"]["parts"]
    methods = {}
    for method in HELD_OUT_METHODS:
        study_entry = entries[method]["evaluation"]["studies"][study]
        scored = sum(int(donor["cells"]) for donor in study_entry["donorConfusionMatrices"])
        if scored != parts["query"]["cells"]:
            raise ValueError(
                f"{method} on {study} was scored on {scored} cells, not the query's {parts['query']['cells']}"
            )
        ci = study_entry["donorBootstrap95CI"]
        if not ci.get("estimable"):
            raise ValueError(f"{method} on {study}: the donor bootstrap gives no interval")
        methods[method] = _disagreement(study_entry["accuracy"], ci["accuracy"])
    reference, query = parts["reference"], parts["query"]
    raster = HELD_OUT_UMAP / f"{tissue}.npz"
    summary = load_json(raster.with_suffix(".json"))
    if summary["cells"] != reference["cells"]:
        raise ValueError(
            f"the {tissue} UMAP holds {summary['cells']} cells, the reference {reference['cells']}"
        )
    if not all(summary["matchesBenchmark"].values()):
        raise ValueError(f"the {tissue} UMAP's rebuild did not match the benchmark's reference")
    return {
        "tissue": tissue,
        "umap": {
            "raster": relative(raster),
            "cells": int(summary["cells"]),
            "types": summary["cellsPerType"],
        },
        # The kidney and lung panels share one axis, which reaches the highest end of an interval in either.
        "scaleHigh": max(_held_out_high(t) for t in HELD_OUT_QUERIES),
        "reference": {
            "cells": int(reference["cells"]),
            "donors": len(reference["donors"]),
            "datasets": len({d["study"] for d in reference["donors"]}),
        },
        "query": {"study": author, "cells": int(query["cells"]), "donors": len(query["donors"])},
        "methods": methods,
        "source": [relative(CROSS_STUDY)],
    }


def held_out_kidney() -> dict:
    """Disagreement on the held-out kidney study, mapped onto the kidney reference."""
    return _held_out("kidney")


def held_out_lung() -> dict:
    """Disagreement on the held-out lung study, mapped onto the rest of the lung atlas."""
    return _held_out("lung")


# ---- annotating blood stores in place -----------------------------------------------------------


def _fewest_cells_first(jobs: list[dict]) -> list[dict]:
    """The jobs from the fewest cells up, as the ladder orders its references, for the in-place column."""
    return sorted(jobs, key=lambda job: job["cells"])


def store_annotation_time() -> dict:
    """How long each job took to open the stored reference, and to map each of its stores, from the jobs' raw records;
    the jobs from the fewest cells up."""
    opening = {job["name"]: job["openingSeconds"] for job in in_place.reference_reads()["jobs"]}
    timed = in_place.mapping_time()
    return {
        "jobs": [
            {
                "name": job["name"],
                "cells": job["cells"],
                "openingSeconds": opening[job["name"]],
                "mappingSeconds": [
                    s["seconds"] for s in timed["stores"] if s["job"] == job["name"]
                ],
            }
            for job in _fewest_cells_first(timed["jobs"])
        ],
        "source": timed["source"],
    }


def store_annotation_memory() -> dict:
    """Each annotated store's container peak while it was mapped onto the stored 4.29-million-cell reference with label
    transfer, which the job records keep store by store apart from the jobs' later pseudobulk pass, grouped by job; the
    jobs from the fewest cells up."""
    return {
        "jobs": _fewest_cells_first(
            [
                {"name": job["name"], "cells": job["cells"], "peaksGiB": job["mappingPeaksGiB"]}
                for job in in_place.memory()["jobs"]
            ]
        ),
        "source": in_place.sources(),
    }


def store_disagreement() -> dict:
    """Each job's disagreement of transferred with author labels, and over all 14 stores; the jobs from the fewest cells
    up."""
    found = in_place.disagreement()
    return {
        "jobs": _fewest_cells_first(found["jobs"]),
        "total": found["total"],
        "source": found["source"],
    }


# ---- joint RNA and protein graphs from 100,000 to 1,457,527 cells ---------------------------------


def _seurat_outcome(seurat: dict) -> str:
    """How Seurat's first analysis ended: completed, or stopped at its time or memory limit."""
    initial = [w for w in seurat["stageWindows"] if w["stage"].startswith("initial:")]
    if initial and not any(w.get("unfinished") for w in initial):
        return "pass"
    if seurat["outcome"] == "time_limit":
        return "time"
    if seurat["outcome"] == "memory_limit":
        return "oom"
    raise ValueError(f"unknown Seurat outcome {seurat['outcome']!r}")


def _multimodal_runs() -> list[dict]:
    sizes = []
    for n in MULTIMODAL_SIZES:
        folder = MULTIMODAL / f"cells_{n}"
        cytearc = load_json(folder / "cytearc_wnn_1.json")["result"]
        if cytearc["status"] != "ok":
            raise ValueError(f"CyteArc's WNN run on {n} cells did not complete")
        seurat = load_json(folder / "seurat_1.json")["result"]
        initial = [w for w in seurat["stageWindows"] if w["stage"].startswith("initial:")]
        outcome = _seurat_outcome(seurat)
        sizes.append(
            {
                "cells": n,
                "cytearc": {
                    "initialSeconds": float(cytearc["initialWnnSeconds"]),
                    "peakGiB": max(
                        s["operationPeakBytes"]
                        for key in ("stages", "reuseStages", "freshRevisionStages")
                        for s in cytearc[key]
                    )
                    / GIB,
                },
                "seurat": {
                    "outcome": outcome,
                    "lastStage": seurat.get("lastStage"),
                    "initialSeconds": sum(w["wallSeconds"] for w in initial)
                    if outcome == "pass"
                    else None,
                    "peakGiB": seurat["stages"][0]["operationPeakBytes"] / GIB,
                    "memoryLimitGiB": seurat["memoryLimitGiB"],
                    "timeLimitSeconds": seurat["timeLimitSeconds"],
                },
            }
        )
    return sizes


def multimodal_time() -> dict:
    """At each size, the time of CyteArc's WNN-first analysis and of Seurat's first analysis where it completed,
    with how each Seurat run ended."""
    return {"sizes": _multimodal_runs(), "source": [relative(MULTIMODAL)]}


def multimodal_reclustering() -> dict:
    """At each size, the time to cluster each tool's stored WNN graph again at resolution 1.2: CyteArc's from its stored
    artifacts, which the run checked against the same clustering repeated from scratch, and Seurat's from its stored
    graph where its first analysis made one. The supplement's reading of the records checks each total against its
    stages and raises if CyteArc's labels differ from the repeat's."""
    sizes = []
    for run in graph_measurements.cite_runs():
        if run["cells"] not in MULTIMODAL_SIZES:
            raise ValueError(
                f"the supplement reads a size, {run['cells']}, that this figure does not show"
            )
        seurat = run["seurat_wnn"]
        sizes.append(
            {
                "cells": run["cells"],
                "cytearc": {"storedSeconds": run["cytearc_wnn"]["storedSeconds"]},
                "seurat": {"storedSeconds": seurat["storedSeconds"], "outcome": seurat["outcome"]},
            }
        )
    if [s["cells"] for s in sizes] != list(MULTIMODAL_SIZES):
        raise ValueError("the supplement reads other sizes than this figure shows")
    return {
        "sizes": sizes,
        "source": graph_measurements.cite_source(("cytearc_wnn_1", "seurat_1")),
    }


def multimodal_memory() -> dict:
    """At each size, CyteArc's container peak over its first analysis and both revisions, and Seurat's where its
    first analysis completed, with how each Seurat run ended."""
    return {"sizes": _multimodal_runs(), "source": [relative(MULTIMODAL)]}


# The graphs the agreement panel compares.
CLUSTER_GRAPHS = ("cytearc_wnn", "seurat_wnn")


def cluster_agreement() -> dict:
    """At each size, the ARI against author labels of each WNN graph's partition cut to the authors' number of cell
    types, for every Leiden seed, with how Seurat's run ended. The experiment ran extra seeds only at the authors'
    number of types, which identifies it; a graph has rows exactly where its run made one."""
    sizes, types = [], set()
    for run in _multimodal_runs():
        path = MULTIMODAL / f"cells_{run['cells']}/matched_cluster_agreement.csv"
        with path.open() as handle:
            rows = [
                r
                for r in csv.DictReader(handle)
                if r["partition_rule"] == "matched_cluster_number" and r["method"] in CLUSTER_GRAPHS
            ]
        seeds = defaultdict(set)
        for r in rows:
            seeds[int(r["target_clusters"])].add(r["seed"])
        seeded = [target for target, found in seeds.items() if len(found) > 1]
        if len(seeded) != 1:
            raise ValueError(
                f"{relative(path)} runs several seeds at {seeded}, not at one number of types"
            )
        types.add(seeded[0])
        graphs = {
            graph: sorted(
                (
                    {
                        "seed": int(r["seed"]),
                        "clusters": int(r["clusters"]),
                        "ari": float(r["ari"]),
                    }
                    for r in rows
                    if r["method"] == graph and int(r["target_clusters"]) == seeded[0]
                ),
                key=lambda found: found["seed"],
            )
            for graph in CLUSTER_GRAPHS
        }
        made = {"cytearc_wnn": True, "seurat_wnn": run["seurat"]["outcome"] == "pass"}
        for graph, runs in graphs.items():
            if bool(runs) != made[graph] or (runs and len(runs) != len(seeds[seeded[0]])):
                raise ValueError(
                    f"{relative(path)} holds {len(runs)} seeds of {graph}, whose run "
                    f"{'made' if made[graph] else 'did not make'} a graph"
                )
        sizes.append(
            {
                "cells": run["cells"],
                "graphs": {graph: runs for graph, runs in graphs.items() if runs},
                "seurat": {"outcome": run["seurat"]["outcome"]},
            }
        )
    if len(types) != 1:
        raise ValueError(
            f"the sizes cut their graphs to different numbers of types, {sorted(types)}"
        )
    return {
        "cellTypes": types.pop(),
        "sizes": sizes,
        "source": [
            relative(MULTIMODAL / f"cells_{n}/matched_cluster_agreement.csv")
            for n in MULTIMODAL_SIZES
        ],
    }


def _graph_umap_summary() -> dict:
    """The collected layouts' summary, checked against the run they lay out: its cells and its labels' counts."""
    summary = load_json(GRAPH_UMAPS / "summary.json")
    run = load_json(MULTIMODAL / "cells_1457527/cytearc_wnn_1.json")["result"]
    if summary["cells"] != run["cells"] or not summary["graphsIdentical"]:
        raise ValueError(
            "the layouts are not of the run's cells, or of graphs that differ from the run's"
        )
    return summary


def graph_umaps() -> dict:
    """The 1,457,527 CITE-seq cells in the UMAP of the run's RNA, protein and WNN graphs, each binned by author label.
    Checks each raster against the summary's cells per label."""
    summary = _graph_umap_summary()
    rasters = {}
    for graph in GRAPH_UMAP_GRAPHS:
        path = GRAPH_UMAPS / f"raster_{graph}.npz"
        with np.load(path, allow_pickle=False) as raster:
            labels = [str(t) for t in raster["types"]]
            per_label = raster["typeCells"].sum(axis=(1, 2))
        if {t: int(n) for t, n in zip(labels, per_label)} != summary["cellsPerLabel"]:
            raise ValueError(f"{relative(path)} holds other cells per label than its summary")
        rasters[graph] = relative(path)
    return {
        "cells": summary["cells"],
        "rasters": rasters,
        "cellsPerLabel": summary["cellsPerLabel"],
        "source": [relative(GRAPH_UMAPS / "summary.json"), *rasters.values()],
    }


# The named COMBAT T-cell reference: integration.t_cell_subtypes's t_cell_reference job, its record, protein tables,
# the raster of its WNN UMAP by subtype and the raster's summary, written by integration.collect_cite_seq_jobs.
T_CELL_REFERENCE = ROOT / "experiments/integration/records/t_cell_reference/combat"
T_CELL_REFERENCE_RASTER = T_CELL_REFERENCE / "raster.npz"
T_CELL_ORDER = (
    "CD4 naive",
    "CD4 central memory",
    "Treg",
    "CD8 naive",
    "CD8 effector",
    "MAIT",
    "Vδ2 γδ T",
)
# Each subtype's defining surface proteins that both studies measured, as COMBAT and Stephenson's shared panel names them
# and as Hao's names them; Hao measured no CCR7. Hao measured CD4 with two antibodies, which rank the subtypes alike,
# and names its Vα7.2 and Vδ2 antibodies TCR-V-7.2 and TCR-V-2: these mark the clusters of Hao's own WNN graph that its
# authors label MAIT and gdT.
T_CELL_MARKERS = {
    "CD4": ("CD4", "CD4-1"),
    "CD8": ("CD8", "CD8"),
    "CD45RA": ("CD45RA", "CD45RA"),
    "CD45RO": ("CD45RO", "CD45RO"),
    "CD57": ("CD57", "CD57"),
    "CD25": ("CD25", "CD25"),
    "CD127": ("CD127", "CD127"),
    "Vα7.2": ("TCR-Va7.2", "TCR-V-7.2"),
    "Vδ2": ("TCR-Vd2", "TCR-V-2"),
}


def _t_cell_record() -> dict:
    record = load_json(T_CELL_REFERENCE / "result.json")
    if record.get("status") != "completed":
        raise ValueError(
            f"{relative(T_CELL_REFERENCE / 'result.json')} is {record.get('status')!r}"
        )
    return record["result"]


def t_cell_map() -> dict:
    """COMBAT's T cells in the UMAP of their WNN graph, binned by named subtype, with the cells of each; checked against
    the reference job's record, and the raster against its summary's cells per subtype."""
    result = _t_cell_record()
    summary = load_json(T_CELL_REFERENCE / "raster_summary.json")
    named = {k: v for k, v in summary["cellsPerSubtype"].items() if k != "unlabelled"}
    if summary["cells"] != result["cells"] or named != result["subtypeCells"]:
        raise ValueError("the T-cell raster holds other cells than the reference job named")
    if sorted(named) != sorted(T_CELL_ORDER):
        raise ValueError(f"the reference names {sorted(named)}, not {sorted(T_CELL_ORDER)}")
    path = T_CELL_REFERENCE_RASTER
    with np.load(path, allow_pickle=False) as raster:
        per_subtype = dict(
            zip(
                [str(t) for t in raster["types"]],
                [int(n) for n in raster["typeCells"].sum(axis=(1, 2))],
                strict=True,
            )
        )
    if per_subtype != summary["cellsPerSubtype"]:
        raise ValueError(f"{relative(path)} holds other cells per subtype than its summary")
    return {
        "cells": result["cells"],
        "raster": relative(path),
        "cellsPerSubtype": summary["cellsPerSubtype"],
        "source": [
            relative(T_CELL_REFERENCE / "raster_summary.json"),
            relative(path),
            relative(T_CELL_REFERENCE / "result.json"),
        ],
    }


def _subtype_medians(path, proteins: list[str], cells: dict[str, int]) -> list[list[float]]:
    """Each named subtype's median per-cell CLR of proteins, from the reference job's table for one set of cells, after
    checking that the table summarizes the cells the job's record gives each subtype."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    medians = {(r["cluster"], r["protein"]): float(r["medianClr"]) for r in rows}
    found = {r["cluster"]: int(r["cells"]) for r in rows}
    missing = [(s, p) for s in T_CELL_ORDER for p in proteins if (s, p) not in medians]
    if missing:
        raise ValueError(f"{relative(path)} lacks {missing[:4]}")
    if {s: found.get(s) for s in T_CELL_ORDER} != cells:
        raise ValueError(
            f"{relative(path)} summarizes other cells than the reference job's record gives each subtype"
        )
    return [[medians[(s, p)] for p in proteins] for s in T_CELL_ORDER]


def t_cell_proteins() -> dict:
    """Each named subtype's median per-cell CLR of the ADT of its defining surface proteins: in COMBAT's T cells, the
    reference whose subtypes these are, and in Hao's T cells, the query, grouped by the subtype each received when
    mapped by RNA alone, so that the mapping never read their ADT. Hao's few cells that received no subtype are left
    out. Each study's proteins carry its own panel's names, the first and second names of T_CELL_MARKERS."""
    result = _t_cell_record()
    received = {
        s: n for s, n in result["queries"]["hao"]["subtypeCells"].items() if s != "unassigned"
    }
    for cells in (result["subtypeCells"], received):
        if sorted(cells) != sorted(T_CELL_ORDER):
            raise ValueError(f"the reference job names {sorted(cells)}, not {sorted(T_CELL_ORDER)}")
    sets = {
        "reference": (
            "COMBAT",
            0,
            T_CELL_REFERENCE / "t_cell_reference_reference_proteins.csv",
            result["subtypeCells"],
        ),
        "query": ("Hao", 1, T_CELL_REFERENCE / "t_cell_reference_hao_proteins.csv", received),
    }
    data = {
        "subtypes": list(T_CELL_ORDER),
        "markers": list(T_CELL_MARKERS),
        "adtNormalization": result["adtNormalization"],
    }
    for role, (study, column, path, cells) in sets.items():
        proteins = [names[column] for names in T_CELL_MARKERS.values()]
        data[role] = {
            "study": study,
            "cells": sum(cells.values()),
            "proteins": proteins,
            "medianClr": _subtype_medians(path, proteins, cells),
        }
    data["source"] = [
        *(relative(path) for _, _, path, _ in sets.values()),
        relative(T_CELL_REFERENCE / "result.json"),
    ]
    return data


def t_cell_reference_comparison() -> dict:
    """For each term both kinds of reference name, the share of Hao's T cells with that author label that receive it,
    by RNA alone, from COMBAT's WNN clusters and from its RNA clusters at resolution 0.5, as the integration
    experiment's summary derives it."""
    return subtype_transfer()


PANELS = (
    provenance_runs,
    ladder_reference_umap,
    ladder_time,
    ladder_memory,
    ladder_accuracy,
    held_out_kidney,
    held_out_lung,
    store_annotation_time,
    store_annotation_memory,
    store_disagreement,
    multimodal_time,
    multimodal_reclustering,
    multimodal_memory,
    cluster_agreement,
    graph_umaps,
    t_cell_map,
    t_cell_proteins,
    t_cell_reference_comparison,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        data = panel()
        path = OUT / f"{panel.__name__}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {relative(path)}")


if __name__ == "__main__":
    main()
