"""Collect the Tahoe-100M runs into records/.

Writes ``raw/tahoe_rna_stage_measurements.json`` (each stage of the pilot, the full run, its two
further replicates and the fixed-layout run: status, seconds, container peak, resources,
placement, range reads and chunk geometry), ``tahoe_rna_scaling_results.json`` (the atlas, stage
times and peaks of the full run against the pilot, preparation and analysis hours, each replicate
of the full run, the export's clusters and cell lines, and the fixed layout against the scaled
one), ``tahoe_umap_raster.npz`` (the cell counts and commonest cell line of each pixel of the
full run's UMAP raster, the arrays of the export's raster that the figure draws),
``tahoe_cluster_cell_lines.csv`` (the export's cells per Leiden cluster and cell line) and
``tahoe_cell_lines.csv`` (the export's cell lines in the raster's code order, each named with its
organ from Tahoe-100M's cell-line table).
"""

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.benchmarks import HERE
from experiments.benchmarks.config import load_profiling_config
from experiments.benchmarks.tahoe import (
    DATA_FILE_COUNT,
    data_file_path,
    sha256_path,
    tahoe_locations,
)
from experiments.runtime.common import ROOT, scrub_private, write_csv, write_json

RECORDS = HERE / "records"
REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
CONFIGS = ROOT / "experiments" / "benchmarks" / "configs" / "benchmark"
# The six-million-cell pilot and the full atlas use the same count layout: 2 GB units whose
# gene-major chunks span a whole shard.
RUNS = {
    "test": CONFIGS / "tahoe_rna_6011925_cells_scaled_layout_replicate_1.toml",
    "full": CONFIGS / "tahoe_rna_95624334_cells_scaled_layout_replicate_1.toml",
}
# The same atlas imported with 200 MB chunks repeats the count passes, with the full run's
# Leiden labels imported for the marker search.
FIXED_LAYOUT = CONFIGS / "tahoe_rna_95624334_cells_fixed_layout_replicate_1.toml"
# Replicates 2 and 3 repeat the full run, each on a store of its own and without an export.
FULL_REPLICATES = {
    2: CONFIGS / "tahoe_rna_95624334_cells_scaled_layout_replicate_2.toml",
    3: CONFIGS / "tahoe_rna_95624334_cells_scaled_layout_replicate_3.toml",
}
PREPARATION = ("createStore", "writeCountsT", "initializeStore")
ANALYSIS = (
    "reopenStore",
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "runUmap",
    "runLeiden",
    "findMarkers",
)
STAGES = PREPARATION + ANALYSIS
COUNT_PASSES = ("markHvgs", "runNormalization", "findMarkers")
GIB = 1024**3
# The export's files that records/ keeps unchanged, by the names it keeps them under.
EXPORT_FILES = {"cluster_cell_lines.csv": "tahoe_cluster_cell_lines.csv"}
# The arrays of the export's UMAP raster that records/ keeps; the export also holds each pixel's
# commonest-line share and the layout's extent, which no figure draws.
RASTER_ARRAYS = ("cells", "commonestCellLine")
STAGE_FIELDS = (
    "stage",
    "status",
    "seconds",
    "peakCgroupBytes",
    "modalCpuLimit",
    "modalMemoryMb",
    "memoryMaxBytes",
    "cytearcMemoryBudget",
    "workers",
    "placement",
)
STORE_OPERATIONS = ("rangeGets", "gets", "readTransferredBytes", "maxInFlight")
GENE_MAJOR_READS = ("countsTCellBand", "countsTReadGroup")
GEOMETRY = ("featureWidth", "featureGroupCount", "cellBandCount")
EXPORT_FIELDS = (
    "leidenClusters",
    "cellLines",
    "medianCommonestCellLineFraction",
    "cellsInClustersWithOneLineAbove90Percent",
    "clusters",
)
# Tahoe-100M's table of its cell lines, which names each line the export codes by its Cellosaurus ID,
# with the organ it comes from.
CELL_LINE_METADATA = "metadata/cell_line_metadata.parquet"
CELL_LINE_COLUMNS = ("Cell_ID_Cellosaur", "cell_name", "Organ")


def stage_measurement(record: dict) -> dict:
    """One stage record reduced to its status, time, container peak, resources and placement, and
    for the count passes its range reads and the geometry of the gene-major bands it read."""
    measurement = {key: record[key] for key in STAGE_FIELDS if key in record}
    details = record.get("details") or {}
    kept: dict[str, Any] = {}
    operations = details.get("storeOperations")
    if operations is not None:
        kept["storeOperations"] = {
            key: operations[key] for key in STORE_OPERATIONS if key in operations
        }
    reports = {
        kind: [{key: report[key] for key in GEOMETRY if key in report} for report in items]
        for kind, items in (details.get("executionReports") or {}).items()
        if kind in GENE_MAJOR_READS
    }
    if reports:
        kept["executionReports"] = reports
    if "shape" in details:
        kept["shape"] = details["shape"]
    measurement["details"] = kept
    return measurement


def stage_summary(record: dict) -> dict:
    operations = (record.get("details") or {}).get("storeOperations") or {}
    return {
        "status": record["status"],
        "seconds": record["seconds"],
        "peakCgroupGiB": (record.get("peakCgroupBytes") or 0) / GIB,
        "rangeGets": operations.get("rangeGets"),
        "getRequests": (operations.get("transport") or {}).get("getRequests"),
        "placement": record.get("placement"),
    }


def stage_records(config: Any, stages: tuple[str, ...]) -> dict[str, dict]:
    from experiments.runtime.object_io import get_json

    size = config.targetSizes[0]
    records = {stage: get_json(config.resultUri(size, stage)) for stage in stages}
    failed = [stage for stage, record in records.items() if record["status"] != "ok"]
    if failed:
        raise SystemExit(f"{config.runTag}: stages not complete: {failed}")
    print(f"{config.runTag}: {len(records)} stage records")
    return records


def layout_description(config: Any) -> str:
    policy = config.countMatrix
    return (
        f"count units of {policy.unitBytes / 1e9:g} GB with {policy.chunkBytes / 1e6:g} MB chunks"
    )


def stage_measurements(runs: dict[str, dict]) -> dict:
    """The stage file: each run's tag, cells, count layout and reduced stage records."""
    return {
        "runs": {
            label: {
                "runTag": run["runTag"],
                "nRows": run["nRows"],
                "countMatrix": run["countMatrix"],
                "stages": {
                    stage: stage_measurement(record) for stage, record in run["stages"].items()
                },
            }
            for label, run in runs.items()
        }
    }


def dataset_summary(files: list[dict]) -> dict:
    return {
        "dataFiles": len(files),
        "cells": sum((int(item["nCells"]) for item in files)),
        "storedValues": sum((int(item["nnz"]) for item in files)),
        "largestCount": max((int(item["maxCount"]) for item in files)),
        "explicitZerosDropped": sum((int(item.get("explicitZeros", 0)) for item in files)),
        "dataBytes": sum((int(item["size"]) for item in files)),
    }


def scaling_summary(
    runs: dict[str, dict],
    replicates: dict[int, dict],
    fixed: dict,
    dataset: dict,
    export: dict,
    description: str,
) -> dict:
    """The summary: stage times and peaks of the full run against the pilot, preparation and
    analysis hours, each replicate of the full run with its hours and stage times and peaks, the
    export's clusters and cell lines, and the fixed against the scaled layout."""
    full, test = (runs["full"]["stages"], runs["test"]["stages"])
    per_stage = {
        stage: {
            "full": stage_summary(full[stage]),
            "test": stage_summary(test[stage]),
            "fullOverTestSeconds": full[stage]["seconds"] / test[stage]["seconds"],
        }
        for stage in STAGES
    }

    def hours(stages: dict, names: tuple[str, ...]) -> float:
        return sum((stages[name]["seconds"] for name in names)) / 3600

    fixed_stages = {stage: stage_summary(record) for stage, record in fixed["stages"].items()}
    for stage in COUNT_PASSES:
        if not (fixed_stages[stage]["rangeGets"] and per_stage[stage]["full"]["rangeGets"]):
            raise SystemExit(f"{stage} has no range-read count in both layouts")
    comparison = {
        key: value for key, value in export["layoutComparison"].items() if key != "runTag"
    }
    return {
        "description": description,
        "revision": REVISION,
        "runs": {label: {"nRows": run["nRows"]} for label, run in runs.items()},
        "dataset": dataset,
        "stages": per_stage,
        "totals": {
            label: {
                "preparationHours": hours(run["stages"], PREPARATION),
                "analysisHours": hours(run["stages"], ANALYSIS),
                "downloadSeconds": run["stages"]["createStore"]["details"]["download"][
                    "downloadSeconds"
                ],
            }
            for label, run in runs.items()
        },
        "fullReplicates": [
            {
                "replicate": number,
                "preparationHours": hours(run["stages"], PREPARATION),
                "analysisHours": hours(run["stages"], ANALYSIS),
                "stages": {
                    stage: {
                        "seconds": record["seconds"],
                        "peakCgroupGiB": record["peakCgroupBytes"] / GIB,
                    }
                    for stage, record in run["stages"].items()
                },
            }
            for number, run in {1: runs["full"], **replicates}.items()
        ],
        "fullStore": {
            "cellsKept": export["cellsKept"],
            "cellsKeptFraction": export["cellsKept"] / runs["full"]["nRows"],
            **{key: export[key] for key in EXPORT_FIELDS},
        },
        "fixedLayout": {
            "stages": fixed_stages,
            "preparationHours": hours(fixed["stages"], PREPARATION),
            "countPasses": {
                stage: {
                    "fixedOverScaledSeconds": fixed_stages[stage]["seconds"]
                    / per_stage[stage]["full"]["seconds"],
                    "fixedOverScaledRangeGets": fixed_stages[stage]["rangeGets"]
                    / per_stage[stage]["full"]["rangeGets"],
                }
                for stage in COUNT_PASSES
            },
            "comparison": comparison,
        },
    }


def download_export_file(exports: str, name: str, expected_sha256: str, destination: Path) -> None:
    """Fetch one file of the full run's export and check it against the export's hash."""
    from experiments.runtime.object_io import download_file

    download_file(f"{exports}/{name}", destination)
    if sha256_path(destination) != expected_sha256:
        raise SystemExit(f"{destination.name} differs from the {name} the export recorded")


def download_cell_line_metadata(destination: Path) -> None:
    """Fetch Tahoe-100M's cell-line table at the pinned revision, checked against the hub's size
    and sha256."""
    from experiments.benchmarks.tahoe import _list_tree, download_hub_file

    listed = [e for e in _list_tree(REVISION, "metadata") if e.get("path") == CELL_LINE_METADATA]
    if len(listed) != 1 or not isinstance(listed[0].get("lfs"), dict):
        raise SystemExit(f"Tahoe-100M at {REVISION} lists no {CELL_LINE_METADATA} with a hash")
    entry = {
        "path": CELL_LINE_METADATA,
        "size": int(listed[0]["size"]),
        "sha256": str(listed[0]["lfs"]["oid"]),
    }
    download_hub_file(entry, REVISION, destination)


def cell_line_table(cell_lines: list[str], metadata: Any) -> list[dict]:
    """The export's cell lines in the order the UMAP raster codes them, each with its name and
    organ from Tahoe-100M's cell-line table (a DataFrame with CELL_LINE_COLUMNS)."""
    if len(set(cell_lines)) != len(cell_lines):
        raise SystemExit("The export lists a cell line more than once")
    table = metadata[list(CELL_LINE_COLUMNS)].drop_duplicates()
    if table["Cell_ID_Cellosaur"].duplicated().any():
        raise SystemExit("The cell-line table gives a cell line two names or organs")
    names = dict(zip(table["Cell_ID_Cellosaur"], table["cell_name"], strict=True))
    organs = dict(zip(table["Cell_ID_Cellosaur"], table["Organ"], strict=True))
    missing = [line for line in cell_lines if line not in names]
    if missing:
        raise SystemExit(f"The cell-line table names no cell line for {missing}")
    return [
        {"code": code, "cellLine": line, "name": names[line], "organ": organs[line]}
        for code, line in enumerate(cell_lines)
    ]


def drawn_raster(path: Path) -> dict:
    """The arrays of the export's UMAP raster that the figure draws."""
    import numpy as np

    with np.load(path, allow_pickle=False) as raster:
        missing = [name for name in RASTER_ARRAYS if name not in raster.files]
        if missing:
            raise SystemExit(f"{path.name} lacks the arrays {missing}")
        return {name: raster[name] for name in RASTER_ARRAYS}


def write_umap_raster(exports: str, export: dict) -> None:
    """Write tahoe_umap_raster.npz from the export's raster, checked against its hash."""
    import numpy as np

    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "umap_raster.npz"
        download_export_file(
            exports, "umap_raster.npz", export["files"]["umap_raster.npz"]["sha256"], path
        )
        arrays = drawn_raster(path)
    RECORDS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RECORDS / "tahoe_umap_raster.npz", **arrays)


def write_cell_lines(exports: str, export: dict) -> None:
    """Write tahoe_cell_lines.csv from the export's categories and Tahoe-100M's cell-line table."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as work:
        categories = Path(work) / "categories.json"
        download_export_file(
            exports, "categories.json", export["files"]["categories.json"]["sha256"], categories
        )
        metadata = Path(work) / "cell_line_metadata.parquet"
        download_cell_line_metadata(metadata)
        rows = cell_line_table(
            json.loads(categories.read_text())["cellLines"],
            pd.read_parquet(metadata, columns=list(CELL_LINE_COLUMNS)),
        )
    if len(rows) != export["cellLines"]:
        raise SystemExit(
            f"The export counts {export['cellLines']} cell lines but codes {len(rows)}"
        )
    write_csv("tahoe_cell_lines.csv", rows, directory=RECORDS)


def main() -> None:
    from experiments.runtime.object_io import get_json

    configs = {label: load_profiling_config(path) for label, path in RUNS.items()}
    fixed_config = load_profiling_config(FIXED_LAYOUT)
    runs = {
        label: {
            "runTag": config.runTag,
            "nRows": config.targetSizes[0],
            "countMatrix": config.countMatrix.model_dump(mode="json"),
            "stages": stage_records(config, STAGES),
        }
        for label, config in configs.items()
    }
    fixed = {
        "runTag": fixed_config.runTag,
        "nRows": fixed_config.targetSizes[0],
        "countMatrix": fixed_config.countMatrix.model_dump(mode="json"),
        "stages": stage_records(fixed_config, fixed_config.effectiveStages),
    }
    if fixed["nRows"] != runs["full"]["nRows"]:
        raise SystemExit("The fixed-layout run must analyse the same cells as the full run")
    replicates = {}
    for number, path in FULL_REPLICATES.items():
        config = load_profiling_config(path)
        replicate = {
            "runTag": config.runTag,
            "nRows": config.targetSizes[0],
            "countMatrix": config.countMatrix.model_dump(mode="json"),
            "stages": stage_records(config, STAGES),
        }
        for key in ("nRows", "countMatrix"):
            if replicate[key] != runs["full"][key]:
                raise SystemExit(f"Replicate {number} differs from the full run in {key}")
        replicates[number] = replicate
    locations = tahoe_locations(configs["full"].datasetPrefixUri, REVISION)
    with ThreadPoolExecutor(max_workers=32) as pool:
        files = list(
            pool.map(
                lambda index: get_json(locations.fileRecordUri(data_file_path(index))),
                range(DATA_FILE_COUNT),
            )
        )
    print(f"{len(files)} staging records")
    exports = f"{configs['full'].resultPrefix(runs['full']['nRows'])}/exports"
    export = get_json(f"{exports}/export.json")
    if (
        export["runTag"] != runs["full"]["runTag"]
        or export["layoutComparison"]["runTag"] != fixed["runTag"]
    ):
        raise SystemExit("The export does not belong to the full run and the fixed-layout run")
    description = (
        "Tahoe-100M RNA workflow: every stage on 16 CPUs and 256 GiB (Modal, US), CyteArc memory "
        f"limit 192 GiB, 16 workers, {layout_description(configs['full'])}, the workflow and seeds "
        "of the ten-million-cell scaling runs, one stage per Modal call. Seconds are each stage's "
        "own operation; peaks are the container's memory. Replicates 2 and 3 repeat the full run on "
        "stores of their own. The fixed-layout run imports the same "
        f"atlas with {layout_description(fixed_config)} and repeats the count passes."
    )
    summary = scaling_summary(runs, replicates, fixed, dataset_summary(files), export, description)
    for name, kept in EXPORT_FILES.items():
        download_export_file(exports, name, export["files"][name]["sha256"], RECORDS / kept)
    write_umap_raster(exports, export)
    write_cell_lines(exports, export)
    write_json(
        "tahoe_rna_stage_measurements.json",
        scrub_private(
            stage_measurements(
                {
                    **runs,
                    **{f"fullReplicate{n}": run for n, run in replicates.items()},
                    "fixedLayout": fixed,
                }
            )
        ),
        subdir="raw",
        directory=RECORDS,
    )
    write_json("tahoe_rna_scaling_results.json", scrub_private(summary), directory=RECORDS)
    totals = summary["totals"]["full"]
    store = summary["fullStore"]
    print(
        f"full run: preparation {totals['preparationHours']:.2f} h, analysis {totals['analysisHours']:.2f} h; {store['cellsKept']} cells kept, {store['leidenClusters']} clusters, median commonest-line fraction {store['medianCommonestCellLineFraction']:.3f}"
    )


if __name__ == "__main__":
    main()
