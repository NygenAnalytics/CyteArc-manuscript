"""Write the records of the COMBAT and Stephenson CITE-seq scaling series.

At each size, multimodal_neighbors.run (WNN-first and SNN-first) and
multimodal_neighbors.seurat_workflow save a run record, and multimodal_agreement.evaluate saves
the agreement tables, under the experiment bucket's key prefix
experiments/multimodal/combat_stephenson_pbmc_citeseq/. With --fetch, this script first copies
those files into the experiment's results folder. It then writes
records/multimodal_neighbors/combat_stephenson_pbmc_citeseq/cells_<n>/: seurat_1.json,
cytearc_snn_1.json and cytearc_wnn_1.json for replicate 1 (seurat_2.json and so on for later
replicates), matched_cluster_agreement.csv and, at 600,000 cells, neighbor_purity.csv. A run
record keeps the outcome, limits, stage times and memory peaks, revision times, reuse check and
agreement with author labels.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from experiments.integration import HERE
from experiments.runtime.locations import results_folder

DATASET = "combat_stephenson_pbmc_citeseq"
RECORDS = HERE / "records" / "multimodal_neighbors" / DATASET
BUCKET_FOLDER = f"multimodal/{DATASET}"
# Neighbor purity is reported at this size.
PURITY_CELLS = 600_000

RESULT_FIELDS = (
    "status",
    "dataset",
    "jointMethod",
    "cells",
    "cellSelectionSha256",
    "outcome",
    "lastStage",
    "memoryLimitGiB",
    "timeLimitSeconds",
    "containerMemoryLimitBytes",
    "biologicalAgreement",
    "stages",
    "reuseStages",
    "freshRevisionStages",
    "stageWindows",
    "reuseValidation",
    "initialWnnSeconds",
    "initialSnnSeconds",
    "revisedWnnSeconds",
    "revisedSnnSeconds",
    "freshRevisedWnnSeconds",
    "freshRevisedSnnSeconds",
    "compilationWarmupWallSeconds",
    "graphsSha256",
)
# multimodal_neighbors_summary reads operationPeakSource with every operationPeakBytes.
STAGE_FIELDS = (
    "stage",
    "wallSeconds",
    "operationPeakBytes",
    "operationPeakSource",
    "memoryMaxBytes",
)
STAGE_GROUPS = ("stages", "reuseStages", "freshRevisionStages")
WINDOW_FIELDS = ("stage", "wallSeconds", "sampledPeakBytes", "unfinished")
# Each size needs replicate 1 of every run and its agreement table.
EXPECTED = (
    "seurat_1.json",
    "cytearc_snn_1.json",
    "cytearc_wnn_1.json",
    "matched_cluster_agreement.csv",
)
RUN_RECORD = re.compile(r"cells_(\d+)/replicate_(\d+)/.+\.json")
TABLE = re.compile(r"cells_(\d+)/agreement/(matched_cluster_agreement|neighbor_purity)\.csv")


def fields(values: dict, names: tuple[str, ...]) -> dict:
    return {key: value for key, value in values.items() if key in names}


def run_record(envelope: dict, path: Path) -> dict:
    """The status and the reported measurements of one job's saved output."""
    if envelope.get("status") != "completed" or not isinstance(envelope.get("result"), dict):
        raise ValueError(
            f"{path} holds no completed job result (status {envelope.get('status')!r})"
        )
    result = fields(envelope["result"], RESULT_FIELDS)
    for group in STAGE_GROUPS:
        if group in result:
            result[group] = [fields(stage, STAGE_FIELDS) for stage in result[group]]
    if "stageWindows" in result:
        result["stageWindows"] = [
            fields(window, WINDOW_FIELDS) for window in result["stageWindows"]
        ]
    return {"status": envelope["status"], "result": result}


def record_name(envelope: dict, replicate: int, path: Path) -> str:
    function = envelope.get("function")
    if function == "seurat_workflow":
        return f"seurat_{replicate}.json"
    if function == "run":
        return f"cytearc_{envelope['result']['jointMethod']}_{replicate}.json"
    raise ValueError(f"{path} was written by {function!r}, not by run or seurat_workflow")


def fetch(source: Path) -> None:
    """Copy the series' run records and agreement tables from the experiment bucket to ``source``."""
    from experiments.runtime.collect_results import list_objects
    from experiments.runtime.locations import RESULTS_PREFIX, experiment_bucket
    from experiments.runtime.object_io import download_file

    prefix = f"{RESULTS_PREFIX}/{BUCKET_FOLDER}/"
    for item in list_objects(experiment_bucket().uri(prefix)):
        name = item["path"].removeprefix(prefix)
        if RUN_RECORD.fullmatch(name) or TABLE.fullmatch(name):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            download_file(item["uri"], target)


def write_records(source: Path, destination: Path) -> None:
    sizes: dict[int, set[str]] = {}
    for path in sorted(source.rglob("*")):
        name = path.relative_to(source).as_posix()
        if match := RUN_RECORD.fullmatch(name):
            cells, replicate = int(match[1]), int(match[2])
            envelope = json.loads(path.read_text())
            record = run_record(envelope, path)
            found = (record["result"].get("dataset"), record["result"].get("cells"))
            if found != (DATASET, cells):
                raise ValueError(f"{path} holds {found}, not {(DATASET, cells)}")
            filename = record_name(envelope, replicate, path)
            target = destination / f"cells_{cells}" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2) + "\n")
        elif match := TABLE.fullmatch(name):
            cells, table = int(match[1]), match[2]
            if table == "neighbor_purity" and cells != PURITY_CELLS:
                continue
            filename = f"{table}.csv"
            target = destination / f"cells_{cells}" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        else:
            continue
        print(f"wrote {target}", flush=True)
        sizes.setdefault(cells, set()).add(filename)
    if not sizes:
        raise ValueError(f"No run records or agreement tables of the series under {source}")
    for cells, written in sorted(sizes.items()):
        expected = {*EXPECTED, "neighbor_purity.csv"} if cells == PURITY_CELLS else set(EXPECTED)
        if missing := expected - written:
            raise ValueError(f"cells_{cells} under {source} lacks {', '.join(sorted(missing))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="folder holding the bucket prefix's files (default: the results folder of this experiment)",
    )
    parser.add_argument(
        "--fetch", action="store_true", help="copy the files from the experiment bucket first"
    )
    args = parser.parse_args()
    source = (args.source or results_folder(HERE.name) / BUCKET_FOLDER).resolve()
    if args.fetch:
        fetch(source)
    write_records(source, RECORDS)


if __name__ == "__main__":
    main()
