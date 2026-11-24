"""Prepare the panel data of the figure on zero-copy analysis from the reuse and catalogue records.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py. Run from the repository root:

  uv run --locked python -B -m manuscript.figures.zero_copy_analysis.prepare
"""

import json

import numpy as np
import pandas as pd
from experiments.runtime.common import DERIVED, ROOT, load_json

REUSE = ROOT / "experiments/reuse/records"
REMOTE_QUERIES = [
    REUSE / f"composite_and_remote/remote_queries/replicate_{r}.json" for r in (1, 2, 3)
]
BLOOD_COMPOSITE = [
    REUSE / f"composite_and_remote/blood_composite/replicate_{r}.json" for r in (2, 3, 4)
]
IN_MEMORY_CONCATENATION = [
    REUSE / f"composite_and_remote/blood_concatenation/replicate_{r}.json" for r in (1, 2, 3)
]
ON_DISK_CONCATENATION = [
    REUSE / f"blood_on_disk_concatenation/replicate_{r}.json" for r in (1, 2, 3)
]
SCANPY_DASK = [REUSE / f"blood_scanpy_dask/replicate_{r}.json" for r in (1, 2, 3)]
# BPCells (cached) on the ten-study H5AD exports through Leiden, three runs. The first run predates the harness's
# disk sampling, so the disk comes from the other two.
BPCELLS_BLOOD = [REUSE / f"blood_bpcells/replicate_{r}.json" for r in (1, 2, 3)]
COVID_COMPOSITE = [
    REUSE / f"composite_and_remote/composite_integration/replicate_{r}.json" for r in (1, 2, 3)
]
WORKFLOW_RUNS = ROOT / "experiments/benchmarks/records/rna_workflow_runs.json"
# The same Tabula Sapiens query against CELLxGENE Census, three runs placed as the CyteBase query's.
CENSUS_QUERIES = [REUSE / f"census_queries/replicate_{replicate}.json" for replicate in (1, 2, 3)]
REMOTE_SUMMARY = DERIVED / "reuse" / "composite_and_remote.json"
# CyteArc's budget and worker sweep on object storage and local disk, and Scanpy on Dask over a Zarr copy read from
# each, as the benchmarks experiment summarizes them.
STORAGE_SWEEP = DERIVED / "benchmarks" / "rna_resource_comparisons.json"
STORAGE_BACKENDS = DERIVED / "benchmarks" / "rna_storage_backend_comparisons.json"
# The progenitor experiment: its records and the summary progenitor_effects derives.
PROGENITOR_RECORDS = REUSE / "blood_progenitors"
PROGENITOR_SUMMARY = DERIVED / "reuse" / "blood_progenitors.json"
INTERFERON_CONTRASTS = DERIVED / "interferon_contrasts" / "lineage_contrasts.json"
# The source study of each sample of a store that pools several, which the progenitor experiment's source-study job
# read, so that each source study of a store in SOURCE_STUDY_RECORDS is a study of its own.
SOURCE_STUDY_VALUES = PROGENITOR_RECORDS / "source_studies.json"
PROGENITOR_RASTER = ROOT / "manuscript/figures/zero_copy_analysis/progenitor_raster.npz"
# The progenitor maps bin the UMAP on a square grid this many bins across.
PROGENITOR_GRID = 160
# The UMAP of the remote Tabula Sapiens query and the MS4A1 values of its B cells, binned by reuse.umap_rasters, which
# repeats the query.
TABULA_RASTER = REUSE / "tabula_sapiens_query_raster.npz"
# The UMAP of CyteArc's Harmony analysis of the ten-study composite, cells counted per study on a square grid by
# reuse.umap_rasters from the per-cell arrays of the cohort analysis, whose record holds its cells per study.
COMPOSITE_RASTER = REUSE / "blood_composite_umap_raster.npz"
BLOOD_COHORT = REUSE / "blood_cohort_analysis.json"
# The same run's mixing of studies and same-type neighbours in its unintegrated and Harmony analyses.
BLOOD_COHORT_MIXING = REUSE / "blood_cohort_mixing.json"
# The illustrative counts of the schematics, which the architecture figure prepares from the Kang thumbnails.
COMPOSITE_NPZ = ROOT / "manuscript/figures/architecture/composite.npz"
OUT = ROOT / "manuscript" / "figures" / "zero_copy_analysis"

GIB = 1024**3
# The MS4A1 colour scale of the UMAP ends at this quantile of the B-cell pixels' mean values.
MS4A1_TOP_QUANTILE = 0.98


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


def relative(path) -> str:
    return str(path.relative_to(ROOT))


# ---- the schematic ---------------------------------------------------------------------


def mount_overlay() -> dict:
    """The illustrative counts the schematic draws in its small count matrices, with each row's cell group."""
    with np.load(COMPOSITE_NPZ) as archive:
        heatmap, groups = archive["heatmap"], archive["heatmapGroup"]
    if heatmap.shape[0] != groups.shape[0]:
        raise ValueError("the schematic heatmap and its cell groups differ in rows")
    return {
        "heatmap": np.round(heatmap, 4).tolist(),
        "heatmapGroup": groups.astype(int).tolist(),
        "source": [relative(COMPOSITE_NPZ)],
    }


# ---- the remote Tabula Sapiens query -----------------------------------------------


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def tabula_query() -> dict:
    """What the remote query returned, as a raster of the UMAP: every cell, and the B cells with their MS4A1 values;
    the colour scale's top, and where to name the B cells. With it, the query's seconds and bytes against querying
    Census and downloading the source H5AD, from query_cost."""
    record = load_json(REMOTE_QUERIES[0])["result"]["remoteQueries"]["tabulaSapiens1_1M"]
    with np.load(TABULA_RASTER) as raster:
        cells, b_cells, b_sum = raster["cells"], raster["bCells"], raster["bMs4a1Sum"]
    if int(cells.sum()) != record["totalCells"] or int(b_cells.sum()) != record["bCellsSelected"]:
        raise ValueError("the raster's cells differ from the remote query's record")
    # The B cells' MS4A1 values summed over the bins, over the B cells, are the query's mean in B cells.
    if not np.isclose(b_sum.sum() / b_cells.sum(), record["ms4a1MeanInBCells"], rtol=1e-9):
        raise ValueError("the raster's MS4A1 values differ from the remote query's record")
    occupied = b_cells > 0
    means = b_sum[occupied] / b_cells[occupied]
    rows, columns = np.nonzero(occupied)
    weights = b_cells[occupied].astype(float)
    cost = query_cost()
    return {
        "raster": relative(TABULA_RASTER),
        "cells": record["totalCells"],
        "bCells": record["bCellsSelected"],
        "ms4a1Top": float(np.quantile(means, MS4A1_TOP_QUANTILE)),
        "bCellLabel": {
            "row": weighted_median(rows.astype(float), weights),
            "column": weighted_median(columns.astype(float), weights),
        },
        "rows": cost["rows"],
        "source": [relative(TABULA_RASTER), *cost["source"]],
    }


def query_cost() -> dict:
    """Seconds and bytes of the CyteBase query on its first open, three runs, its bytes the HTTP response bodies it
    received; of the same query against CELLxGENE Census, three runs, which reads the cell types and MS4A1 but no UMAP,
    as Census keeps none, its bytes those its container's network interface received; and of downloading the source
    H5AD inside a cloud container, one run. Each later row carries its value over the CyteBase query's."""
    first_seconds, first_bytes = [], []
    for path in REMOTE_QUERIES:
        query = load_json(path)["result"]["remoteQueries"]["tabulaSapiens1_1M"]
        first_seconds.append(query["openSeconds"] + query["embeddingAndMarkerQuerySeconds"])
        first_bytes.append(
            query["openIo"]["http"]["responseBytes"]
            + query["embeddingAndMarkerQueryIo"]["http"]["responseBytes"]
        )
    derived = load_json(REMOTE_SUMMARY)["remoteQuery"]
    if not np.isclose(np.mean(first_seconds), derived["firstOpenAndQuerySeconds"], rtol=1e-9):
        raise ValueError("the remote query's seconds differ from the derived summary")
    if not np.isclose(np.mean(first_bytes), derived["firstOpenAndQueryHttpBytes"], rtol=1e-9):
        raise ValueError("the remote query's HTTP bytes differ from the derived summary")
    download = load_json(WORKFLOW_RUNS)["inputs"]["tabula_sapiens"]
    query = load_json(REMOTE_QUERIES[0])["result"]["remoteQueries"]["tabulaSapiens1_1M"]
    if download["nCells"] != query["totalCells"]:
        raise ValueError("the downloaded H5AD and the queried store hold different cells")
    census_seconds, census_bytes = [], []
    for path in CENSUS_QUERIES:
        record = load_json(path)
        if record["status"] != "completed":
            raise ValueError(f"{path.name} did not complete")
        census = record["result"]
        if (census["totalCells"], census["cellsSelected"]) != (
            query["totalCells"],
            query["bCellsSelected"],
        ):
            raise ValueError("the Census query read other cells than the CyteBase query")
        census_seconds.append(census["openSeconds"] + census["querySeconds"])
        census_bytes.append(census["openReceivedBytes"] + census["queryReceivedBytes"])
    rows = [
        {
            "key": "cytebase",
            "label": "CyteBase",
            "seconds": point_summary(first_seconds),
            "bytes": point_summary(first_bytes),
        },
        {
            "key": "census",
            "label": "Census",
            "seconds": point_summary(census_seconds),
            "bytes": point_summary(census_bytes),
        },
        {
            "key": "download",
            "label": "H5AD download",
            "seconds": point_summary([download["downloadSeconds"]]),
            "bytes": point_summary([download["sourceBytes"]]),
        },
    ]
    for row in rows[1:]:
        row["timeOverQuery"] = row["seconds"]["mean"] / rows[0]["seconds"]["mean"]
        row["dataOverQuery"] = row["bytes"]["mean"] / rows[0]["bytes"]["mean"]
    return {
        "rows": rows,
        "source": [
            *map(relative, REMOTE_QUERIES),
            *map(relative, CENSUS_QUERIES),
            relative(WORKFLOW_RUNS),
            relative(REMOTE_SUMMARY),
        ],
    }


# ---- the interferon response of each lineage, from the screen's rerun ------------------------------------


def interferon_score() -> dict:
    """The disease-versus-healthy contrasts of donor interferon scores in each lineage, with the T-cell scores of both
    sides that the ridges draw, as the interferon contrasts experiment derives them."""
    derived = load_json(INTERFERON_CONTRASTS)
    keys = (
        "rows",
        "lineages",
        "ridgeLineage",
        "minimumDonors",
        "totals",
        "cohortRecords",
        "sourceStudies",
        "source",
    )
    return {key: derived[key] for key in keys}


# ---- circulating progenitors pulled out of the blood corpus -----------------------------------------------------


def progenitor_states_of_cells() -> pd.DataFrame:
    """The analysed progenitor subset's cells with their map position, state and source (blood study or enriched
    reference), the states named as the derived summary names the clusters."""
    summary = load_json(PROGENITOR_SUMMARY)
    cells = pd.read_parquet(PROGENITOR_RECORDS / "progenitor_cell_table.parquet")
    names = {int(cluster): state for cluster, state in summary["clusterStates"].items()}
    cells["state"] = cells["leiden_1_0"].map(names)
    if cells["state"].isna().any():
        raise ValueError("a cluster of the cell table has no state in the summary")
    from experiments.reuse.progenitor_effects import ENRICHED

    cells["reference"] = cells["dataset"].isin(ENRICHED)
    return cells


def progenitor_maps() -> dict:
    """The subset's UMAP binned in square bins, PROGENITOR_GRID across its wider side, so the raster takes the map's own
    shape; cells counted per state and per source, written to PROGENITOR_RASTER; each progenitor state's cells and the
    bin at the median of its cells, where its name goes."""
    from experiments.reuse.progenitor_effects import MATURE

    cells = progenitor_states_of_cells()
    mature = cells["state"].isin(MATURE)
    order = cells.loc[~mature, "state"].value_counts().index.tolist() + (
        ["mature"] if mature.any() else []
    )
    cells["category"] = np.where(mature, "mature", cells["state"])
    lows = cells[["umap1", "umap2"]].min().to_numpy()
    spans = cells[["umap1", "umap2"]].max().to_numpy() - lows
    size = float(spans.max()) * 1.0001 / PROGENITOR_GRID
    column = ((cells["umap1"] - lows[0]) / size).astype(int).to_numpy()
    row = ((cells["umap2"] - lows[1]) / size).astype(int).to_numpy()
    shape = (int(row.max()) + 1, int(column.max()) + 1)
    states = np.zeros((len(order), *shape), dtype=np.int32)
    np.add.at(
        states,
        (cells["category"].map({s: i for i, s in enumerate(order)}).to_numpy(), row, column),
        1,
    )
    sources = np.zeros((2, *shape), dtype=np.int32)
    np.add.at(sources, (cells["reference"].astype(int).to_numpy(), row, column), 1)
    np.savez_compressed(PROGENITOR_RASTER, states=states, sources=sources, order=np.asarray(order))
    labels = cells.assign(row=row, column=column).groupby("category")[["row", "column"]].median()
    return {
        "raster": relative(PROGENITOR_RASTER),
        "states": [
            {
                "state": state,
                "cells": int((cells["category"] == state).sum()),
                "row": float(labels.loc[state, "row"]),
                "column": float(labels.loc[state, "column"]),
            }
            for state in order
        ],
        "cells": len(cells),
        "referenceCells": int(cells["reference"].sum()),
        "source": [
            relative(PROGENITOR_RECORDS / "progenitor_cell_table.parquet"),
            relative(PROGENITOR_SUMMARY),
        ],
    }


def study_stores() -> dict[str, list[str]]:
    """The stores of each study, from the discovery table, so a study can be named by its stores."""
    from experiments.reuse.progenitor_effects import study_of

    cells = pd.read_parquet(
        PROGENITOR_RECORDS / "progenitor_cells.parquet", columns=["storeId", "collectionId"]
    ).drop_duplicates()
    cells["study"] = study_of(cells)
    return {study: sorted(part["storeId"].unique()) for study, part in cells.groupby("study")}


def ratio(pooled: dict | None) -> dict | None:
    if pooled is None:
        return None
    return {
        "ratio": float(np.exp(pooled["effect"])),
        "low": float(np.exp(pooled["low"])),
        "high": float(np.exp(pooled["high"])),
        "p": pooled["p"],
        "studies": pooled["studies"],
        "tau2": pooled["tau2"],
    }


def progenitor_effects() -> dict:
    """The cells, times and bytes from the blood corpus to the analysed progenitors, with each step's cells from the
    datasets of CD34+ cells only and from all other blood datasets, and the marker genes the screen read; per study,
    the progenitors per 10,000 blood cells of its healthy donors and of its donors with COVID-19, with their numbers
    and the study's odds ratio and 95% interval, under the default selection; the odds ratio pooled over studies under
    each selection, with its P value; each disease against the healthy donors of its own studies, from the highest
    odds ratio down, with its studies, donors, odds ratio, 95% interval and P value; COMBAT's cohorts, each with its
    donors' progenitors per 10,000 cells and the cohort's own, its donors' cells pooled, with the odds ratio per step
    from healthy volunteers to critical COVID-19 and that of sepsis against healthy volunteers; the samples the cohort
    records relabelled or left out; and each progenitor state's share of the progenitors from all blood and from the
    datasets of CD34+ cells only."""
    from experiments.reuse.progenitor_effects import GENES

    summary = load_json(PROGENITOR_SUMMARY)
    stores = study_stores()
    default = summary["variants"]["default"]

    def studies(effects: list[dict]) -> list[dict]:
        return [
            {
                "stores": stores[e["study"]],
                "healthyDonors": e["healthy"],
                "covidDonors": e["cases"],
                "healthyPer10k": e["healthyPer10k"],
                "covidPer10k": e["casesPer10k"],
                "ratio": float(np.exp(e["effect"])),
                "low": float(np.exp(e["effect"] - 1.959964 * e["standardError"])),
                "high": float(np.exp(e["effect"] + 1.959964 * e["standardError"])),
            }
            for e in sorted(effects, key=lambda e: e["effect"])
        ]

    def odds(estimate: dict) -> dict:
        return {
            "ratio": estimate["ratio"],
            "low": estimate["ratioLow"],
            "high": estimate["ratioHigh"],
            "p": estimate["p"],
        }

    diseases = [
        {
            "disease": disease,
            "studies": entry["estimate"]["studies"],
            "cases": sum(e["cases"] for e in entry["studies"]),
            "healthy": sum(e["healthy"] for e in entry["studies"]),
            **odds(entry["estimate"]),
        }
        for disease, entry in summary["diseases"].items()
    ]
    combat = summary["combatCohorts"]
    return {
        "discovery": summary["discovery"],
        "extraction": summary["extraction"],
        "analysis": summary["analysis"],
        "progenitorCells": summary["progenitorCells"],
        "sources": summary["sources"],
        "markerGenes": list(GENES),
        "donors": default["donors"],
        "covid": studies(default["covid"]),
        "selections": {name: ratio(v["covidPooled"]) for name, v in summary["variants"].items()},
        "diseases": sorted(diseases, key=lambda d: -d["ratio"]),
        "diseaseStudies": len(
            {e["study"] for entry in summary["diseases"].values() for e in entry["studies"]}
        ),
        "cohorts": {
            "cohorts": [
                {
                    "cohort": group["cohort"],
                    "donorsPer10k": sorted(
                        d["progenitors"] / d["cells"] * 1e4 for d in group["donors"]
                    ),
                    "per10k": group["per10k"],
                }
                for group in combat["cohorts"]
            ],
            "trend": odds(combat["trend"]),
            "sepsis": odds(combat["sepsis"]),
            "tooFewDonors": combat["tooFewDonors"],
        },
        "cohortRecords": summary["cohortRecords"],
        "sourceStudies": summary["sourceStudies"],
        "stateShares": summary["stateShares"],
        "source": [
            relative(PROGENITOR_SUMMARY),
            relative(PROGENITOR_RECORDS / "progenitor_cells.parquet"),
            relative(PROGENITOR_RECORDS / "severity.json"),
            relative(SOURCE_STUDY_VALUES),
        ],
    }


# ---- the ten-study composite against concatenated copies --------------------------------


def composite_footprint() -> dict:
    """Local disk and peak container memory of three routes to one ten-study blood cohort: the virtual composite
    and its analysis through Leiden; AnnData's concatenation in memory, with no analysis after it; and AnnData's
    concatenation on disk followed by Scanpy on Dask. Also the composite's sources, the raster of the UMAP that
    CyteArc's Harmony analysis of the same ten studies drew, with its cells per study; the mixing of studies and the
    share of same-type neighbours of that analysis and of the unintegrated one; and the checks of the four-study
    composite against its sources and against a materialized copy."""
    composites = [load_json(p)["result"]["composite"] for p in BLOOD_COMPOSITE]
    store_bytes = {c["localStoreAfterCreateBytes"] for c in composites}
    if len(store_bytes) != 1 or any(c["copiedCountChunkBytes"] for c in composites):
        raise ValueError("the composite runs differ in store bytes or copied count chunks")
    sources = composites[0]["sourceCellCounts"]
    if sum(sources.values()) != composites[0]["cellsTotalSelected"]:
        raise ValueError("the composite's sources do not add up to its cells")
    in_memory = [
        load_json(p)["result"]["materializedConcatenation"] for p in IN_MEMORY_CONCATENATION
    ]
    on_disk = [load_json(p)["result"]["onDiskConcatenation"] for p in ON_DISK_CONCATENATION]
    scanpy = [load_json(p) for p in SCANPY_DASK]
    input_bytes = {sum(f["bytes"] for f in run["inputFiles"]) for run in in_memory}
    input_bytes |= {sum(f["bytes"] for f in run["inputs"]) for run in on_disk}
    if len(input_bytes) != 1:
        raise ValueError("the concatenation runs read different input files")
    written = {run["concatenation"]["processIoBytes"]["wchar"] for run in on_disk}
    if len(written) != 1:
        raise ValueError("the on-disk concatenation runs wrote different bytes")
    if any(run["status"] != "ok" for run in scanpy):
        raise ValueError("a Scanpy (Dask) run did not complete")
    bpcells = [load_json(p) for p in BPCELLS_BLOOD]
    if any(run["status"] != "ok" for run in bpcells):
        raise ValueError("a BPCells run did not complete")
    if {run["download"]["bytes"] for run in bpcells} != input_bytes:
        raise ValueError("the BPCells runs read other files than the concatenation runs")
    bpcells_peaks = [
        run["peakWorkDirectoryBytes"] for run in bpcells if "peakWorkDirectoryBytes" in run
    ]
    if not bpcells_peaks:
        raise ValueError("no BPCells run recorded its disk")
    (h5ad,) = input_bytes
    (concatenated,) = written
    routes = [
        {
            "key": "composite",
            "label": "CyteArc composite",
            "entity": "cytearc",
            "disk": [{"label": "composite store", "GiB": store_bytes.pop() / GIB}],
            "memoryGiB": point_summary(
                [c["unintegrated"]["resources"]["operationPeakBytes"] / GIB for c in composites]
            ),
            "memoryStep": "analysis through Leiden",
        },
        {
            "key": "in_memory",
            "label": "AnnData, in memory",
            "entity": "anndata",
            "disk": [{"label": "H5AD files", "GiB": h5ad / GIB}],
            "memoryGiB": point_summary(
                [run["resources"]["operationPeakBytes"] / GIB for run in in_memory]
            ),
            "memoryStep": "concatenation, no analysis",
        },
        {
            "key": "on_disk",
            "label": "AnnData on disk, Scanpy (Dask)",
            "entity": "scanpy",
            "disk": [
                {"label": "H5AD files", "GiB": h5ad / GIB},
                {"label": "written by concatenation", "GiB": concatenated / GIB},
            ],
            "memoryGiB": point_summary([run["peakCgroupBytes"] / GIB for run in scanpy]),
            "memoryStep": "Scanpy (Dask) analysis",
            "concatenationMemoryGiB": point_summary(
                [run["concatenation"]["resources"]["operationPeakBytes"] / GIB for run in on_disk]
            ),
        },
        {
            "key": "bpcells",
            "label": "BPCells",
            "entity": "bpcells",
            # The peak of its work directory: the H5AD files, their uncompressed copies and BPCells' own matrices,
            # before it deletes the inputs.
            "disk": [
                {"label": "H5AD files", "GiB": h5ad / GIB},
                {"label": "written by BPCells", "GiB": (np.mean(bpcells_peaks) - h5ad) / GIB},
            ],
            "diskRuns": len(bpcells_peaks),
            "memoryGiB": point_summary([run["peakCgroupBytes"] / GIB for run in bpcells]),
            "memoryStep": "BPCells analysis through Leiden",
        },
    ]
    checks = []
    for path in COVID_COMPOSITE:
        composite = load_json(path)["result"]["composite"]
        alignment, agreement = composite["sourceAlignment"], composite["materializedAgreement"]
        checks.append(
            {
                "countsCompared": alignment["countsCompared"],
                "exactCounts": alignment["exactCounts"],
                "copiedCountChunkBytes": alignment["copiedCountChunkBytes"],
                "clusterAri": agreement["clusterAri"],
                "exactClusterLabels": agreement["exactClusterLabels"],
                "pcaSubspaceDegrees": agreement["pcaMaximumSubspaceDegrees"],
                "pcaSubspaceToleranceDegrees": composite["tolerances"]["pcaSubspaceDegrees"],
                "cells": composite["cellsTotalSelected"],
                "studies": len(composite["sourceCellCounts"]),
            }
        )
    if not all(c["exactCounts"] and c["exactClusterLabels"] for c in checks):
        raise ValueError("a four-study composite run did not match its sources and its copy")
    cohort = load_json(BLOOD_COHORT)["result"]["sourceCellCounts"]
    if cohort != sources:
        raise ValueError("the cohort analysis holds other studies or cells than the composite runs")
    with np.load(COMPOSITE_RASTER) as raster:
        studies = [str(study) for study in raster["studies"]]
        binned = raster["studyCells"].reshape(len(studies), -1).sum(axis=1)
    if dict(zip(studies, binned.tolist())) != cohort:
        raise ValueError("the UMAP raster's cells per study differ from the cohort analysis record")
    mixing = load_json(BLOOD_COHORT_MIXING)["result"]
    if mixing["sourceCellCounts"] != cohort:
        raise ValueError(
            "the cohort analysis's mixing record holds other studies or cells than its run record"
        )
    integration = {
        phase: {
            "studyMixing": mixing[phase]["mixing"]["macroStudyMixing"],
            "sameType": mixing[phase]["mixing"]["sameTypeNeighborFraction"],
        }
        for phase in ("unintegrated", "integrated")
    }
    return {
        "cells": composites[0]["cellsTotalSelected"],
        "genes": composites[0]["genesShared"],
        "sources": [
            {"study": study, "cells": cells}
            for study, cells in sorted(sources.items(), key=lambda item: -item[1])
        ],
        "routes": routes,
        "checks": checks,
        "umap": {
            "raster": relative(COMPOSITE_RASTER),
            "studies": [{"study": study, "cells": cohort[study]} for study in studies],
        },
        "integration": integration,
        "source": [
            *map(relative, BLOOD_COMPOSITE),
            *map(relative, IN_MEMORY_CONCATENATION),
            *map(relative, ON_DISK_CONCATENATION),
            *map(relative, SCANPY_DASK),
            *map(relative, BPCELLS_BLOOD),
            *map(relative, COVID_COMPOSITE),
            relative(BLOOD_COHORT),
            relative(BLOOD_COHORT_MIXING),
            relative(COMPOSITE_RASTER),
        ],
    }


# ---- the same analysis from object storage and from local disk -----------------------------------------

# CyteArc's runs of the one-million-cell embryo workflow in the budget and worker sweep that match on both storage
# backends: a 24 GiB budget and 8 workers.
BACKEND_BUDGET_GIB, BACKEND_WORKERS = 24, 8


def storage_backends() -> dict:
    """Analysis minutes of the one-million-cell embryo workflow read from local disk and from object storage: CyteArc's
    runs of the budget and worker sweep and Scanpy on Dask over a Zarr copy, three runs each; and on each backend,
    Scanpy's time over CyteArc's, the inverse of the benchmark's comparison of geometric means, with its 95% confidence
    interval and Welch P value."""
    sweep = load_json(STORAGE_SWEEP)["runs"]

    def cytearc(backend: str) -> list[float]:
        return [
            (run["elapsedSeconds"] - run["conversionElapsedSeconds"]) / 60
            for run in sweep
            if run.get("complete")
            and run["dataset"] == "embryo1M"
            and run["storeBackend"] == backend
            and run["budgetGiB"] == BACKEND_BUDGET_GIB
            and run["workers"] == BACKEND_WORKERS
        ]

    comparison = load_json(STORAGE_BACKENDS)

    def arm_minutes(arm: str) -> list[float]:
        runs = comparison["arms"][arm]["runs"]
        if any(run["status"] != "ok" for run in runs):
            raise ValueError(f"a run of {arm} did not complete")
        return [run["analysisMinutes"] for run in runs]

    tools = [
        {
            "key": "cytearc",
            "label": "CyteArc",
            "localDisk": point_summary(cytearc("modalLocalDisk")),
            "objectStorage": point_summary(cytearc("objectStore")),
        },
        {
            # BPCells reads local files only.
            "key": "bpcells",
            "label": "BPCells",
            "localDisk": point_summary(arm_minutes("bpcells_cached")),
            "objectStorage": None,
        },
        {
            "key": "scanpy",
            "label": "Scanpy (Dask)",
            "localDisk": point_summary(arm_minutes("scanpy_dask_covariance_csc_local_zarr")),
            "objectStorage": point_summary(arm_minutes("scanpy_dask_covariance_csc_remote_zarr")),
        },
    ]
    if any(
        tool[backend] is not None and tool[backend]["n"] != 3
        for tool in tools
        for backend in ("localDisk", "objectStorage")
    ):
        raise ValueError("each tool needs three runs on each storage backend it reads")
    scanpy_over_cytearc = {}
    for backend, runs, key in (
        ("localDisk", "cytearcLocalDiskAnalysisMinutes", "cytearcOverScanpyLocalDisk"),
        ("objectStorage", "cytearcObjectStorageAnalysisMinutes", "cytearcOverScanpyObjectStorage"),
    ):
        if not np.allclose(
            sorted(tools[0][backend]["values"]), sorted(comparison[runs]), rtol=1e-9
        ):
            raise ValueError(f"CyteArc's {backend} runs differ from the ones the comparison used")
        ratio = comparison["comparisons"][key]
        # Scanpy over CyteArc inverts the ratio and swaps the interval's ends; the two-sided P value stays.
        scanpy_over_cytearc[backend] = {
            "ratio": 1 / ratio["ratio"],
            "low": 1 / ratio["high"],
            "high": 1 / ratio["low"],
            "p": ratio["p"],
        }
    ratio = comparison["comparisons"]["cytearcOverBpcellsLocalDisk"]
    bpcells_over_cytearc = {
        "localDisk": {
            "ratio": 1 / ratio["ratio"],
            "low": 1 / ratio["high"],
            "high": 1 / ratio["low"],
            "p": ratio["p"],
        }
    }
    return {
        "tools": tools,
        "scanpyOverCytearc": scanpy_over_cytearc,
        "bpcellsOverCytearc": bpcells_over_cytearc,
        "source": [relative(STORAGE_SWEEP), relative(STORAGE_BACKENDS)],
    }


def main() -> None:
    for name, values in (
        ("mount_overlay", mount_overlay()),
        ("storage_backends", storage_backends()),
        ("tabula_query", tabula_query()),
        ("interferon_score", interferon_score()),
        ("progenitor_maps", progenitor_maps()),
        ("progenitor_effects", progenitor_effects()),
        ("composite_footprint", composite_footprint()),
    ):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(values, indent=1, allow_nan=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
