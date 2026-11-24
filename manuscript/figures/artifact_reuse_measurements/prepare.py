"""Prepare the panel data of the supplementary figure on reusing stored artifacts: the measurements behind the section on
artifact reuse, one panel for each piece of evidence.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py. Every drawn value is
read from an experiment record and checked against the totals that record states, and against a second record where
one holds the same quantity; a mismatch stops the run. Run from the repository root:

  uv run --locked python -B -m manuscript.figures.artifact_reuse_measurements.prepare
"""

import csv
import json
import math

import numpy as np
from experiments.runtime.common import ROOT, load_json
from experiments.runtime.records import ANALYSIS_STAGES, elapsed_seconds

REUSE = ROOT / "experiments/reuse/records"
MAPPING = ROOT / "experiments/mapping/records"
LUNG = ROOT / "experiments/lung/records"
MOUNTED_EMBRYO = REUSE / "embryo_endothelial_reanalysis.json"
# The published analyses of the embryo scaling series.
SCALING_RUNS = ROOT / "experiments/benchmarks/records/raw/rna_scaling_runs.json"
# The ten-million-cell store that the mount and its endothelial reanalysis reused.
MOUNTED_REFERENCE_TAG = "embryo_rna_scaling_10000000_cells_replicate_6"
# The ten-million-cell store that the neighbor-count fork reused.
FORK_REFERENCE_TAG = "embryo_rna_scaling_10000000_cells_replicate_4"
FORK_RUNS = MAPPING / "artifact_reuse_runs.json"
COMPOSITE = [
    REUSE / f"composite_and_remote/composite_integration/replicate_{replicate}.json"
    for replicate in (1, 2, 3)
]
UPDATE_SUMMARY = LUNG / "lung_reference_update_summary.json"
UPDATE_COSTS = LUNG / "lung_reference_update_costs.csv"
MULTIMODAL = (
    ROOT / "experiments/integration/records/multimodal_neighbors/combat_stephenson_pbmc_citeseq"
)
MULTIMODAL_SIZES = (100000, 300000, 600000, 1000000, 1457527)
OUT = ROOT / "manuscript" / "figures" / "artifact_reuse_measurements"

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
        "values": [float(v) for v in values],
    }


def agree(found: float, expected: float, what: str, *, tolerance: float = 1e-6) -> None:
    """Stop when two records, or a record and its own total, disagree on a value."""
    if not math.isclose(found, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{what}: {found!r} where {expected!r} was expected")


def published_runs() -> dict:
    """The published analyses of the embryo series by run tag."""
    runs = load_json(SCALING_RUNS)
    for tag, run in runs.items():
        if run["funnel"]["status"] != "ok":
            raise ValueError(f"the published run {tag} did not complete")
    return runs


def stage_artifact(run: dict, stage: str) -> str:
    return run["stages"][stage]["details"]["artifact"]["artifact_id"]


# ---- the mounted ten-million-cell embryo store ---------------------------------------------------------------------

# Each request on the published analysis, in the order the job made them: its step in the record, its name, the
# published stage it repeats or parallels, and whether it returns a stored artifact or computes a new one.
REQUESTS = (
    ("reuse.list", "list stored results", None, "stored"),
    ("reuse.same.neighbors", "neighbors, k = 11", "queryNeighbors", "stored"),
    ("reuse.same.clusters", "Leiden, resolution 1", "runLeiden", "stored"),
    ("reuse.leiden.r0.5", "Leiden, resolution 0.5", "runLeiden", "new"),
    ("reuse.leiden.r2", "Leiden, resolution 2", "runLeiden", "new"),
    ("reuse.neighbors.k21", "neighbors, k = 21", "queryNeighbors", "new"),
    ("reuse.graph.k21", "graph, k = 21", "buildConnectivityMap", "new"),
    ("reuse.leiden.k21_r1", "Leiden on k = 21", "runLeiden", "new"),
    ("reuse.markers.r2", "markers, resolution 2", "findMarkers", "new"),
)


def _mounted() -> tuple[dict, dict]:
    """The mount record and the published run it mounts, after checking that the store is that run's and that the
    record's steps add up to its totals and its user store's listing."""
    record = load_json(MOUNTED_EMBRYO)
    run = published_runs()[MOUNTED_REFERENCE_TAG]
    if record["status"] != "ok" or record["referenceTag"] != MOUNTED_REFERENCE_TAG:
        raise ValueError(f"the mount record must hold a completed mount of {MOUNTED_REFERENCE_TAG}")
    published = {kind: a["artifactId"] for kind, a in record["reuse"]["published"].items()}
    for kind, stage in (
        ("neighbors", "queryNeighbors"),
        ("graph", "buildConnectivityMap"),
        ("clusters", "runLeiden"),
    ):
        if published[kind] != stage_artifact(run, stage):
            raise ValueError(
                f"the mounted store's published {kind} is not the {stage} artifact of {MOUNTED_REFERENCE_TAG}"
            )
    steps = record["steps"]
    if set(steps) != set(record["stepOrder"]):
        raise ValueError("the mount record's steps and step order differ")
    for name, step in steps.items():
        agree(
            step["userStoreBytesWritten"],
            step["userStoreBytesAfter"] - step["userStoreBytesBefore"],
            f"bytes written by {name}",
        )
    for field, total in (
        ("seconds", "seconds"),
        ("sourceReadBytes", "sourceReadBytes"),
        ("userStoreBytesWritten", "userStoreBytesWritten"),
    ):
        agree(
            sum(s[field] for s in steps.values()),
            record["totals"][total],
            f"the mount's total {total}",
        )
        agree(
            sum(steps[s][field] for s in record["reuse"]["stepOrder"]),
            record["reuse"]["totals"][total],
            f"the requests' total {total}",
        )
    agree(
        record["totals"]["userStoreBytesWritten"],
        record["userStore"]["final"]["bytes"],
        "the final user store",
    )
    agree(
        steps["mount"]["userStoreBytesAfter"],
        record["userStore"]["afterMount"]["bytes"],
        "the mounted user store",
    )
    return record, run


def mounted_store_requests() -> dict:
    """Each request on the published analysis of the mounted ten-million-cell embryo store, on 8 CPUs, against the
    published run's stage it repeats or parallels, on 16 CPUs, with the bytes the request read from the source store
    and wrote to the user store; a request for a stored artifact writes nothing."""
    record, run = _mounted()
    steps = record["steps"]
    if [step for step, *_ in REQUESTS] != record["reuse"]["stepOrder"]:
        raise ValueError("the drawn requests must be the record's requests, in its order")
    for kind in ("sameNeighbors", "sameClusters"):
        same = record["reuse"]["steps"][kind]
        if not same["isPublished"] or same["bytesWritten"] != 0:
            raise ValueError(f"{kind} must return the published artifact without writing")
    requests = []
    for step, name, stage, kind in REQUESTS:
        s = steps[step]
        written = int(s["userStoreBytesWritten"])
        if (kind == "stored") != (written == 0):
            raise ValueError(f"{step} is drawn as {kind} but wrote {written} bytes")
        requests.append(
            {
                "step": step,
                "name": name,
                "kind": kind,
                "seconds": float(s["seconds"]),
                "publishedStage": stage,
                "publishedSeconds": float(run["stages"][stage]["seconds"]) if stage else None,
                "readBytes": int(s["sourceReadBytes"]),
                "graphReadBytes": int(
                    s["sourceReadByComponent"].get("artifact:connectivity_map", 0)
                ),
                "writtenBytes": written,
            }
        )
    analysis = sum(run["stages"][stage]["seconds"] for stage in ANALYSIS_STAGES)
    clustered = {c["cells"] for c in record["reuse"]["clusterings"].values()}
    if len(clustered) != 1:
        raise ValueError("the new clusterings of the whole atlas must cover the same cells")
    return {
        "requests": requests,
        "publishedAnalysisSeconds": float(analysis),
        "publishedRunSeconds": float(elapsed_seconds(run["funnel"])),
        "cpus": {
            "published": int(run["funnel"]["modalResources"]["modalCpuLimit"]),
            "requests": int(record["container"]["cpu"]),
        },
        "atlasCells": int(record["source"]["cells"]),
        "clusteredCells": int(clustered.pop()),
        "source": [relative(MOUNTED_EMBRYO), relative(SCALING_RUNS)],
    }


def endothelial_store_size() -> dict:
    """The sizes of the published embryo store and its source H5AD against the user store mounted on it, after the
    mount, after the endothelial reanalysis and after the requests on the whole atlas, and the bytes all steps read
    from the source store. The reanalysis is every step the job made before its first request."""
    record, run = _mounted()
    steps = record["steps"]
    order = record["stepOrder"]
    first_request = record["reuse"]["stepOrder"][0]
    if order.index(first_request) == 0:
        raise ValueError("the mount record holds no reanalysis before its first request")
    last = order[order.index(first_request) - 1]
    after_reanalysis = steps[last]["userStoreBytesAfter"]
    agree(
        after_reanalysis,
        steps[first_request]["userStoreBytesBefore"],
        "the reanalysis's store",
    )
    h5ad = run["funnel"]["datasetBytes"]
    agree(
        h5ad, run["stages"]["createStore"]["datasetBytes"], "the source H5AD of the published run"
    )
    source = record["source"]
    if not source["unchanged"] or source["before"]["bytes"] != source["after"]["bytes"]:
        raise ValueError("the mounted store must be unchanged by the job")
    referenced = record["referenced"]
    counts = referenced["counts"]["sourceBytes"] + referenced["countsT"]["sourceBytes"]
    results = record["notCopied"]["publishedResultsBytes"]
    # The published store's parts: counts in both layouts and their summaries, read in place; its analysis results,
    # read in place; and the cell and gene tables, copied into the user store. Only its small metadata files are left.
    parts = (
        counts
        + referenced["countSummaries"]["sourceBytes"]
        + results
        + sum(part["sourceBytes"] for part in record["copied"].values())
    )
    agree(parts, source["before"]["bytes"], "the published store's parts", tolerance=1e-6)
    return {
        "sizes": [
            {"name": "source H5AD", "kind": "source", "bytes": int(h5ad)},
            {"name": "remote store", "kind": "source", "bytes": int(source["before"]["bytes"])},
            {"name": "its counts, 2 layouts", "kind": "source", "bytes": int(counts)},
            {"name": "its analysis results", "kind": "source", "bytes": int(results)},
            {
                "name": "user store, mounted",
                "kind": "user",
                "bytes": int(record["userStore"]["afterMount"]["bytes"]),
            },
            {"name": "after the reanalysis", "kind": "user", "bytes": int(after_reanalysis)},
            {
                "name": "after the requests",
                "kind": "user",
                "bytes": int(record["userStore"]["final"]["bytes"]),
            },
        ],
        "readBytes": int(record["totals"]["sourceReadBytes"]),
        "notCopiedBytes": int(record["notCopied"]["publishedResultsBytes"]),
        "source": [relative(MOUNTED_EMBRYO), relative(SCALING_RUNS)],
    }


# ---- the neighbor-count fork on the stored ten-million-cell embryo analysis ----------------------------------------

# The fork's steps: the three stored artifacts it looked up and the five it computed, each with the published stage it
# stands for.
FORK_LOOKUPS = {"runNormalization": "normalization", "runPca": "pca", "buildAnnIndex": "annIndex"}
FORK_COMPUTED = ("queryNeighbors", "buildConnectivityMap", "runUmap", "runLeiden", "findMarkers")
STAGE_NAMES = {
    "filterCells": "cell filter",
    "markHvgs": "variable genes",
    "runNormalization": "normalization",
    "runPca": "PCA",
    "buildEmbeddingInitialization": "UMAP initialization",
    "buildAnnIndex": "neighbor index",
    "queryNeighbors": "neighbors",
    "buildConnectivityMap": "graph",
    "runUmap": "UMAP",
    "runLeiden": "Leiden",
    "findMarkers": "markers",
}
STAGE_ORDER = (
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "buildEmbeddingInitialization",
    "runUmap",
    "runLeiden",
    "findMarkers",
)


def _fork() -> dict:
    """The fork from k = 11 to k = 15 on the stored analysis of the ten-million-cell store, checked against the published
    run it started from. Forks the record holds on other stores are not drawn."""
    published = published_runs()[FORK_REFERENCE_TAG]
    found = {
        tag: entry["result"]
        for tag, entry in load_json(FORK_RUNS)["results"].items()
        if entry["result"]["referenceTag"] == FORK_REFERENCE_TAG
    }
    if len(found) != 1:
        raise ValueError(
            f"the fork record must hold one fork on {FORK_REFERENCE_TAG}; it holds {sorted(found)}"
        )
    [(tag, result)] = found.items()
    cells = int(published["funnel"]["nRows"])
    if cells != 10000000:
        raise ValueError(
            f"the fork must start from the ten-million-cell store; {FORK_REFERENCE_TAG} holds {cells:,} cells"
        )
    if (result["kOriginal"], result["kNew"], result["newArtifacts"]) != (11, 15, 5):
        raise ValueError(f"{tag} must fork from k = 11 to k = 15 and create 5 artifacts")
    if not all(result["reuse"][key] for key in FORK_LOOKUPS.values()):
        raise ValueError(f"{tag} did not reuse every stored upstream artifact")
    if set(result["fullRunStageSeconds"]) != set(ANALYSIS_STAGES):
        raise ValueError(f"{tag} must hold the published analysis's stages")
    for stage, seconds in result["fullRunStageSeconds"].items():
        agree(seconds, published["stages"][stage]["seconds"], f"{tag}'s published {stage}")
    agree(
        sum(result["fullRunStageSeconds"].values()),
        result["fullRunAnalysisSeconds"],
        f"{tag}'s analysis",
    )
    computed = {stage: result["stages"][f"fork.{stage}"]["seconds"] for stage in FORK_COMPUTED}
    lookups = {
        stage: result["stages"][f"reuse.{key}"]["seconds"] for stage, key in FORK_LOOKUPS.items()
    }
    if len(result["stages"]) != len(computed) + len(lookups):
        raise ValueError(f"{tag} holds fork steps the figure does not draw")
    agree(sum(computed.values()), result["forkSeconds"], f"{tag}'s fork")
    return {
        "cells": cells,
        "storedSeconds": float(result["fullRunAnalysisSeconds"]),
        "forkSeconds": float(result["forkSeconds"]),
        "lookupSeconds": {stage: float(s) for stage, s in lookups.items()},
        "computedSeconds": {stage: float(s) for stage, s in computed.items()},
        "storedStageSeconds": {
            stage: float(s) for stage, s in result["fullRunStageSeconds"].items()
        },
        "cpus": {
            "stored": int(published["funnel"]["modalResources"]["modalCpuLimit"]),
            "fork": int(result["container"]["cpu"]),
        },
    }


def neighbor_count_fork_stages() -> dict:
    """Each stage of the stored analysis of the ten-million-cell store against the fork's step for it: a lookup of the
    stored normalization, PCA and neighbor index, or a recomputation downstream of the new neighbor count; and the
    totals, the stored analysis's eleven stages against the fork's five recomputed ones."""
    fork = _fork()
    stages = []
    for stage in STAGE_ORDER:
        if stage in FORK_LOOKUPS:
            kind, seconds = "lookup", fork["lookupSeconds"][stage]
        elif stage in FORK_COMPUTED:
            kind, seconds = "computed", fork["computedSeconds"][stage]
        else:
            kind, seconds = None, None
        stages.append(
            {
                "stage": stage,
                "name": STAGE_NAMES[stage],
                "fork": kind,
                "storedSeconds": fork["storedStageSeconds"][stage],
                "forkSeconds": seconds,
            }
        )
    return {
        "stages": stages,
        "totals": {"storedSeconds": fork["storedSeconds"], "forkSeconds": fork["forkSeconds"]},
        "cells": fork["cells"],
        "cpus": fork["cpus"],
        "source": [relative(FORK_RUNS), relative(SCALING_RUNS)],
    }


# ---- Harmony integration on the four-study COVID-19 composite ----------------------------------------------------

COMPOSITE_ARMS = (
    ("unintegrated", "unintegrated"),
    ("integrationFresh", "Harmony, fresh"),
    ("integrationReusingUpstream", "Harmony, reuse"),
)
COMPOSITE_STAGES = (
    ("input_snapshot", "input snapshot"),
    ("highly_variable_features", "variable genes"),
    ("normalization", "normalization"),
    ("pca", "PCA"),
    ("harmony", "Harmony"),
    ("ann_index", "neighbor index"),
    ("neighbors", "neighbors"),
    ("connectivity", "graph"),
    ("leiden_1.0", "Leiden"),
)


def _composites() -> list[dict]:
    """The three replicates' composite records, checked: integration with reuse read no counts and reused the four
    shared upstream artifacts, and gave exactly the fresh integration's clusters and corrected coordinates."""
    composites = []
    for path in COMPOSITE:
        record = load_json(path)
        if record["status"] != "completed":
            raise ValueError(f"{relative(path)} did not complete")
        c = record["result"]["composite"]
        shared = c["sharedUpstreamArtifacts"]
        if len(shared) != 4 or not all(shared.values()):
            raise ValueError(
                f"{relative(path)}: integration must reuse the four shared upstream artifacts"
            )
        if c["integrationReusingUpstream"]["io"]["readBytes"] != 0:
            raise ValueError(f"{relative(path)}: integration with reuse must read no counts")
        agree(
            c["integrationFresh"]["io"]["readBytes"],
            c["unintegrated"]["io"]["readBytes"],
            "counts read fresh",
        )
        same = c["freshVersusReused"]
        if not same["exactClusterLabels"] or same["correctedCoordinateMaxAbsError"] != 0.0:
            raise ValueError(f"{relative(path)}: fresh and reused integration must agree exactly")
        composites.append(c)
    return composites


def composite_integration_stages() -> dict:
    """Each stage of Harmony integration on a freshly created composite and after the unintegrated analysis, reusing
    its upstream artifacts, and the time of each integration: the mean and range of three replicates. With reuse the
    upstream stages return stored artifacts and create none."""
    composites = _composites()
    for c in composites:
        for arm in ("integrationFresh", "integrationReusingUpstream"):
            stages = c[arm]["summary"]["stages"]
            if set(stages) != {stage for stage, _ in COMPOSITE_STAGES}:
                raise ValueError(f"the drawn stages must be every stage of {arm}")
            if not sum(s["seconds"] for s in stages.values()) <= c[arm]["elapsedSeconds"]:
                raise ValueError(f"the stages of {arm} must fit in its elapsed time")
    stages = []
    for stage, name in COMPOSITE_STAGES:
        fresh = [c["integrationFresh"]["summary"]["stages"][stage] for c in composites]
        reused = [c["integrationReusingUpstream"]["summary"]["stages"][stage] for c in composites]
        stored = {s["created"] == 0 for s in reused}
        if len(stored) != 1:
            raise ValueError(f"the replicates disagree on whether {stage} was reused")
        stages.append(
            {
                "stage": stage,
                "name": name,
                "fresh": point_summary([s["seconds"] for s in fresh]),
                "reuse": point_summary([s["seconds"] for s in reused]),
                "stored": stored.pop(),
            }
        )
    return {
        "stages": stages,
        "totals": {
            "fresh": point_summary([c["integrationFresh"]["elapsedSeconds"] for c in composites]),
            "reuse": point_summary(
                [c["integrationReusingUpstream"]["elapsedSeconds"] for c in composites]
            ),
        },
        "cells": int(composites[0]["cellsTotalSelected"]),
        "studies": len(composites[0]["sourceCellCounts"]),
        "threads": int(composites[0]["settings"]["nthreads"]),
        "source": [relative(path) for path in COMPOSITE],
    }


def composite_integration_resources() -> dict:
    """The count bytes each analysis of the composite read from the source stores, and its container peak, over three
    replicates."""
    composites = _composites()
    return {
        "arms": [
            {
                "arm": arm,
                "name": name,
                "readBytes": point_summary([c[arm]["io"]["readBytes"] for c in composites]),
                "peakGiB": point_summary(
                    [c[arm]["resources"]["operationPeakBytes"] / GIB for c in composites]
                ),
            }
            for arm, name in COMPOSITE_ARMS
        ],
        "source": [relative(path) for path in COMPOSITE],
    }


# ---- the lung reference revision -----------------------------------------------------------------------------------

# The steps of the Natri update in place and of the complete rerun of both versions in a fresh store, one row each:
# its name and the steps of each job it covers.
UPDATE_ROWS = (
    ("mount a fresh store", (), ("mount",)),
    ("select cells", ("select",), ("select",)),
    (
        "load reference v1",
        ("openReference_v1", "loadReference_v1"),
        ("openReference_v1", "loadReference_v1"),
    ),
    ("v1 projection", ("map_v1",), ("map_v1",)),
    (
        "v1 labels, 3 levels",
        ("transfer_v1_ann_level_1", "transfer_v1_ann_level_3", "transfer_v1_ann_finest_level"),
        ("transfer_v1_ann_level_1", "transfer_v1_ann_level_3", "transfer_v1_ann_finest_level"),
    ),
    (
        "load reference v2",
        ("openReference_v2", "loadReference_v2"),
        ("openReference_v2", "loadReference_v2"),
    ),
    ("v2 projection", ("map_v2",), ("map_v2",)),
    (
        "v2 labels, 3 levels",
        ("transfer_v2_ann_level_1", "transfer_v2_ann_level_3", "transfer_v2_ann_finest_level"),
        ("transfer_v2_ann_level_1", "transfer_v2_ann_level_3", "transfer_v2_ann_finest_level"),
    ),
    ("recover v1 results", ("recover",), ()),
)
# The cost table keeps six significant digits, so its sums match the summary to this relative tolerance.
TABLE = 5e-5
# Columns of the cost table that count bytes read from a reference: the core atlas, the full atlas and the revision.
REFERENCE_READS = ("read_core", "read_full", "read_revised")


def _costs() -> dict[str, dict[str, dict]]:
    jobs: dict[str, dict[str, dict]] = {}
    with UPDATE_COSTS.open() as handle:
        for row in csv.DictReader(handle):
            if row["step"] in jobs.setdefault(row["job"], {}):
                raise ValueError(f"the cost table holds {row['job']} {row['step']} twice")
            jobs[row["job"]][row["step"]] = row
    return jobs


def _added(row: dict) -> float:
    return float(row["bytesAdded"]) if row["bytesAdded"] else 0.0


def reference_update_steps() -> dict:
    """Each step of updating the Natri query store in place to the revised lung reference, where the v1 results are
    stored, against the same step of a complete rerun of both versions in a fresh store mounted on the published
    Natri store; with the bytes each step added and the totals the summary states."""
    summary = load_json(UPDATE_SUMMARY)
    costs = _costs()
    update, rerun = costs["update_natri"], costs["rerun_natri"]
    comparison = summary["incrementalVersusComplete"]
    if not (comparison["sameResults"]["v1"] and comparison["sameResults"]["v2"]):
        raise ValueError("the update and the complete rerun must give the same results")
    incremental = [s for _, steps, _ in UPDATE_ROWS[:-1] for s in steps]
    complete = [s for _, _, steps in UPDATE_ROWS for s in steps]
    v2 = [s for name, steps, _ in UPDATE_ROWS if "v2" in name for s in steps]
    agree(
        sum(float(update[s]["seconds"]) for s in incremental),
        comparison["incremental"]["seconds"],
        "the update",
        tolerance=TABLE,
    )
    agree(
        sum(float(update[s]["seconds"]) for s in v2),
        comparison["incremental"]["v2Seconds"],
        "the update's v2",
        tolerance=TABLE,
    )
    agree(
        sum(float(rerun[s]["seconds"]) for s in complete),
        comparison["complete"]["seconds"],
        "the rerun",
        tolerance=TABLE,
    )
    agree(
        float(rerun["mount"]["seconds"]),
        comparison["complete"]["mountSeconds"],
        "the rerun's mount",
        tolerance=TABLE,
    )
    agree(
        sum(_added(r) for r in update.values()),
        comparison["incremental"]["bytesAdded"],
        "the update's bytes",
        tolerance=TABLE,
    )
    agree(
        sum(_added(r) for r in rerun.values()),
        comparison["complete"]["resultBytes"],
        "the rerun's results",
        tolerance=TABLE,
    )
    agree(
        comparison["complete"]["mountBytes"] + comparison["complete"]["resultBytes"],
        comparison["complete"]["bytesWritten"],
        "the rerun's bytes",
    )
    recovery = summary["centres"]["natri"]["costs"]["recovery"]
    agree(float(update["recover"]["seconds"]), recovery["seconds"], "the recovery", tolerance=TABLE)
    if recovery["referenceReadBytes"] != 0 or any(
        float(update["recover"][c]) != 0 for c in REFERENCE_READS
    ):
        raise ValueError("recovering the v1 results must read no reference")
    for name, steps, _ in UPDATE_ROWS:
        if name.startswith("v1") and any(update[s]["created"] != "False" for s in steps):
            raise ValueError(f"the update must find its {name} stored")
    if rerun["mount"]["bytesAdded"]:
        raise ValueError("the cost table must leave the mount's bytes to the summary")

    def seconds(job: dict, steps: tuple, together: bool) -> list[float]:
        """A point per step, or one for the steps together: opening and loading a reference."""
        values = [float(job[s]["seconds"]) for s in steps]
        return [sum(values)] if together and values else values

    def update_kind(steps: tuple) -> str:
        """How the update met a row's steps: a stored artifact found again, results recovered from stored label
        transfers, or work done, which a loading step without an artifact also is."""
        if steps == ("recover",):
            return "recovered"
        created = {update[s]["created"] for s in steps}
        return "stored" if created == {"False"} else "new"

    rows = []
    for name, mine, theirs in UPDATE_ROWS:
        together = name.startswith("load")
        rows.append(
            {
                "name": name,
                "update": seconds(update, mine, together),
                "rerun": seconds(rerun, theirs, together),
                "updateAddedBytes": float(sum(_added(update[s]) for s in mine)),
                "rerunAddedBytes": float(
                    comparison["complete"]["mountBytes"]
                    if theirs == ("mount",)
                    else sum(_added(rerun[s]) for s in theirs)
                ),
                "updateKind": update_kind(mine),
            }
        )
    return {
        "rows": rows,
        "update": {
            "seconds": float(comparison["incremental"]["seconds"]),
            "addedBytes": float(comparison["incremental"]["bytesAdded"]),
        },
        "rerun": {
            "seconds": float(comparison["complete"]["seconds"]),
            "addedBytes": float(comparison["complete"]["bytesWritten"]),
        },
        "cells": int(summary["centres"]["natri"]["cells"]),
        "source": [relative(UPDATE_SUMMARY), relative(UPDATE_COSTS)],
    }


# ---- a revised clustering of the CITE-seq WNN analysis -------------------------------------------------------------


def multimodal_revisions() -> list[dict]:
    """At each size, the WNN clustering revised to resolution 1.2 from the stored graphs and recomputed from the
    counts, both after the first analysis, checked to give identical graphs and labels."""
    sizes = []
    for n in MULTIMODAL_SIZES:
        result = load_json(MULTIMODAL / f"cells_{n}" / "cytearc_wnn_1.json")["result"]
        if result["status"] != "ok" or result["cells"] != n:
            raise ValueError(f"CyteArc's WNN run on {n} cells did not complete")
        valid = result["reuseValidation"]
        if not (valid["clusterLabelsIdentical"] and valid["graphEdgesIdentical"]):
            raise ValueError(
                f"the revision from stored graphs on {n} cells differs from the fresh one"
            )
        agree(
            sum(s["wallSeconds"] for s in result["reuseStages"]),
            result["revisedWnnSeconds"],
            f"{n} reuse",
        )
        agree(
            sum(s["wallSeconds"] for s in result["freshRevisionStages"]),
            result["freshRevisedWnnSeconds"],
            f"{n} fresh",
        )
        sizes.append(
            {
                "cells": n,
                "reuseSeconds": float(result["revisedWnnSeconds"]),
                "freshSeconds": float(result["freshRevisedWnnSeconds"]),
                "reusePeakGiB": max(s["operationPeakBytes"] for s in result["reuseStages"]) / GIB,
                "freshPeakGiB": max(s["operationPeakBytes"] for s in result["freshRevisionStages"])
                / GIB,
                "reuseStages": len(result["reuseStages"]),
                "freshStages": len(result["freshRevisionStages"]),
            }
        )
    return sizes


def multimodal_revision() -> dict:
    """Time and peak memory to revise the WNN clustering of 100,000 to 1,457,527 CITE-seq cells to resolution 1.2, from
    the stored RNA and protein graphs against recomputing both modalities from the counts; the peak is the highest
    operation peak over each revision's stages."""
    return {"sizes": multimodal_revisions(), "source": [relative(MULTIMODAL)]}


PANELS = (
    mounted_store_requests,
    endothelial_store_size,
    neighbor_count_fork_stages,
    composite_integration_stages,
    composite_integration_resources,
    reference_update_steps,
    multimodal_revision,
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
