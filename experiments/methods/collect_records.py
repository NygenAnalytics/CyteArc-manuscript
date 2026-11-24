"""Write the methods records from the runs of the checks.

Each RECORD is a run's job record, the JSON that the Modal runner saves at the payload's
recordUri, as an s3:// URI in the experiment bucket or a local copy; or, for a run on this
machine through `python -m experiments`, the result file that the function writes to the
payload's outputDirectory. The CellRank job saves its result at its outputUri, with the per-cell
predictions beside it. Each record keeps the run's status and result, without the call, placement
and duration of the job, the package versions of its environment, and the bucket keys and folders
that its settings name.

    uv run --locked python -m experiments.methods.collect_records chromvar RECORD
    uv run --locked python -m experiments.methods.collect_records grnboost2 RECORD
    uv run --locked python -m experiments.methods.collect_records scenic-pruning RECORD
    uv run --locked python -m experiments.methods.collect_records gene-scores RECORD
    uv run --locked python -m experiments.methods.collect_records pca-drift RECORD
    uv run --locked python -m experiments.methods.collect_records lsi SAMPLED ACTIVE...
    uv run --locked python -m experiments.methods.collect_records droplet-qc RECORD...
    uv run --locked python -m experiments.methods.collect_records hto-demultiplexing RECORD...
    uv run --locked python -m experiments.methods.collect_records rna-numerical-reproducibility \
        PCA DETERMINISM
    uv run --locked python -m experiments.methods.collect_records doublets RECORD...
    uv run --locked python -m experiments.methods.collect_records fate-probabilities OUTPUT
"""

import argparse
import re
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from experiments.methods import HERE
from experiments.runtime.common import scrub_private, write_json_at

RECORDS = HERE / "records"
# Payload fields that locate a job's inputs and outputs rather than set the check.
LOCATION_SETTINGS = ("recordUri", "r2EndpointUrl", "workDirectory", "outputDirectory")
# Text that names a location private to one run: a bucket key, a container or local folder, or
# one of the authors' local working folders (.scratch, .internal), which are not part of this
# repository. Collection refuses a record that names one, so no record carries such a path.
PRIVATE_LOCATION = re.compile(r"s3://|/tmp/|/workspace/|/home/|\.scratch|\.internal")
DOUBLET_METHODS = ("cytearc", "scrublet", "doubletFinder", "scDblFinder")
FATE_PREDICTION = "pancreas_larry_fate_prediction"


def objects(location: str):
    from experiments.runtime.harness import Objects
    from experiments.runtime.locations import required

    if not location.startswith("s3://"):
        return Objects(None)
    return Objects(required("R2_ENDPOINT_URL", "the experiment bucket's S3 endpoint"))


def read_json(location: str) -> dict:
    return objects(location).get_json(location)


def fetch(location: str, folder: Path) -> Path:
    """A local copy of the file at ``location``, in ``folder``."""
    path = folder / Path(location).name
    objects(location).download(location, path)
    return path


def without(values: dict, *keys: str) -> dict:
    return {key: value for key, value in values.items() if key not in keys}


def completed(run: dict, experiment: str) -> dict:
    """The status and result of a completed run of ``experiment``, from its job record or from the
    result file of a run on this machine."""
    job: dict = run if "result" in run else {"status": "completed", "result": run}
    result: dict = job["result"]
    if job.get("status") != "completed" or result.get("status") != "ok":
        raise ValueError(
            f"Expected a completed {experiment} run; got job status {job.get('status')!r} and "
            f"result status {result.get('status')!r}"
        )
    if result.get("experiment") != experiment:
        raise ValueError(f"Expected a run of {experiment}; got {result.get('experiment')!r}")
    return {"status": "completed", "result": without(result, "provenance")}


def chromvar(job: dict) -> dict:
    run = completed(job, "chromvar")
    return {**run, "result": without(run["result"], "referenceInputsUri")}


def grnboost2(job: dict) -> dict:
    run = completed(job, "grnboost2")
    runs = [without(native, "uri") for native in run["result"]["nativeRuns"]]
    return {**run, "result": {**run["result"], "nativeRuns": runs}}


def scenic_pruning(job: dict) -> dict:
    return completed(job, "scenic_pruning")


def gene_scores(job: dict) -> dict:
    return without(completed(job, "gene_scores")["result"], "sourcePath")


def pca_drift(job: dict) -> dict:
    return completed(job, "pca_drift")["result"]


def lsi_run(job: dict) -> dict:
    return without(completed(job, "lsi")["result"], "region", "source")


def lsi(sampled: dict, active: Iterable[dict]) -> dict:
    """The run on the sampled cells, with the runs on the cells the multimodal workflow keeps."""
    record = lsi_run(sampled)
    if record["settings"].get("cellSelection") == "active":
        raise ValueError("The first LSI run must be the one on the sampled cells")
    full = [lsi_run(job) for job in active]
    if not full or any(run["settings"]["cellSelection"] != "active" for run in full):
        raise ValueError("Give the LSI runs on the active cell selection after the sampled run")
    return {**record, "fullSelectionMeasurements": full}


def by_dataset(jobs: Iterable[dict], experiment: str) -> dict:
    datasets: dict[str, dict] = {}
    for job in jobs:
        run = completed(job, experiment)
        name = run["result"]["dataset"]
        if name in datasets:
            raise ValueError(f"Two {experiment} runs on {name}")
        datasets[name] = run
    return datasets


def droplet_qc(jobs: Iterable[dict]) -> dict:
    datasets = by_dataset(jobs, "droplet_qc")
    return {"eligibleDatasets": list(datasets), "datasets": datasets}


def hto_demultiplexing(jobs: Iterable[dict]) -> dict:
    return {"experiment": "hto_demultiplexing", "datasets": by_dataset(jobs, "hto_demultiplexing")}


def rna_check(job: dict, check: str) -> dict:
    """One check of rna_numerical_reproducibility.run, without the payload's locations and the
    path of the downloaded store."""
    run = completed(job, "rna_numerical_reproducibility")
    result = run["result"]
    if result["check"] != check:
        raise ValueError(f"Expected the {check} check; got {result['check']!r}")
    result = {**result, "settings": without(result["settings"], *LOCATION_SETTINGS)}
    if "source" in result:
        result["source"] = without(result["source"], "source")
    return {**run, "result": result}


def rna_numerical_reproducibility(pca: dict, determinism: dict) -> dict:
    """The PCA and determinism checks, without the clock times of the pipeline stages."""
    determinism = rna_check(determinism, "determinism")
    result = determinism["result"]
    runs = [
        {
            **run,
            "stages": [without(stage, "startedAtNs", "finishedAtNs") for stage in run["stages"]],
        }
        for run in result["pipelineRuns"]
    ]
    determinism = {**determinism, "result": {**result, "pipelineRuns": runs}}
    return {
        "pca": {"measurement": rna_check(pca, "pca")},
        "determinism": {"measurement": determinism},
    }


def doublet_scores(location: str, dataset: str) -> str:
    """Where evaluate_doublets saves a dataset's per-cell scores, beside the job record."""
    return f"{location.removesuffix('.json')}/{dataset}/scores.csv"


def doublets(jobs: dict[str, dict], read_scores: Callable[[str, str], Any]) -> tuple[dict, dict]:
    """The four doublet runs with a check of the per-cell scores against their metrics, and the
    per-cell scores of every method for each dataset.

    ``jobs`` maps the location of each run's job record to the record, and
    ``read_scores(location, dataset)`` returns the cell, label and score table that the run saved
    for a dataset beside that record."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    runs, locations = {}, {}
    for location, job in jobs.items():
        run = rna_check(job, "doublets")
        result = run["result"]
        rows = [{**row, "timing": without(row["timing"], "versions")} for row in result["datasets"]]
        runs[result["method"]] = {**run, "result": {**result, "datasets": rows}}
        locations[result["method"]] = location
    if sorted(runs) != sorted(DOUBLET_METHODS):
        raise ValueError(f"Give one doublets run per method; got {sorted(runs)}")
    validation, tables = [], {}
    for name in runs["cytearc"]["result"]["eligibleDatasets"]:
        merged = pd.DataFrame()
        largest = 0.0
        for method in DOUBLET_METHODS:
            table = read_scores(locations[method], name)
            row = {row["dataset"]: row for row in runs[method]["result"]["datasets"]}[name]
            truth = table["label"].to_numpy() == "doublet"
            scores = table["score"].to_numpy(dtype=float)
            largest = max(
                largest,
                abs(float(average_precision_score(truth, scores)) - row["auprc"]),
                abs(float(roc_auc_score(truth, scores)) - row["auroc"]),
            )
            if merged.empty:
                merged = table[["cell", "label"]].copy()
            elif not (
                np.array_equal(merged["cell"], table["cell"])
                and np.array_equal(merged["label"], table["label"])
            ):
                raise ValueError(f"{method} scored other cells or labels than cytearc on {name}")
            merged[method] = table["score"].to_numpy(dtype=float)
        tables[name] = merged
        validation.append(
            {
                "dataset": name,
                "cells": len(merged),
                "cellIdsAndLabelsIdentical": True,
                "maxRecomputedMetricDifference": largest,
            }
        )
    return {"records": {m: runs[m] for m in DOUBLET_METHODS}, "validation": validation}, tables


def fate_probabilities(result: dict) -> dict:
    """The CellRank comparison as the bundle that runtime.records reads, without the bucket key
    of its per-cell predictions."""
    if result.get("experiment") != "cellrank" or "error" in result["larry"]:
        raise ValueError("Expected a completed cellrank_job result")
    result = {**result, "larry": without(result["larry"], "cellsUri")}
    return {"results": {FATE_PREDICTION: {"result": result}}}


def fate_cells(output: str) -> str:
    """Where cellrank_job saves the per-cell LARRY predictions, beside its result."""
    return output.replace(".json", "_larry_cells.npz")


def public(record: Any, name: str) -> Any:
    """``record``, after checking that it names no private location or identifier."""

    def strings(value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    found = [text for text in strings(record) if PRIVATE_LOCATION.search(text)]
    if found or scrub_private(record) != record:
        raise ValueError(f"{name} would name a private location or identifier: {found[:3]}")
    return record


def write(name: str, record: dict) -> None:
    write_json_at(RECORDS / name, public(record, name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("chromvar", "grnboost2", "scenic-pruning", "gene-scores", "pca-drift"):
        commands.add_parser(name).add_argument("record")
    commands.add_parser("fate-probabilities").add_argument("output")
    for name in ("droplet-qc", "hto-demultiplexing", "doublets"):
        commands.add_parser(name).add_argument("records", nargs="+")
    command = commands.add_parser("lsi")
    command.add_argument("sampled")
    command.add_argument("active", nargs="+")
    command = commands.add_parser("rna-numerical-reproducibility")
    command.add_argument("pca")
    command.add_argument("determinism")
    arguments = parser.parse_args()
    single = {
        "chromvar": (chromvar, "chromvar.json"),
        "grnboost2": (grnboost2, "grnboost2.json"),
        "scenic-pruning": (scenic_pruning, "scenic_pruning.json"),
        "gene-scores": (gene_scores, "gene_scores.json"),
        "pca-drift": (pca_drift, "pca_drift.json"),
    }
    if arguments.command in single:
        collect, name = single[arguments.command]
        write(name, collect(read_json(arguments.record)))
    elif arguments.command == "lsi":
        write("lsi.json", lsi(read_json(arguments.sampled), map(read_json, arguments.active)))
    elif arguments.command == "droplet-qc":
        write("droplet_qc.json", droplet_qc(map(read_json, arguments.records)))
    elif arguments.command == "hto-demultiplexing":
        write("hto_demultiplexing.json", hto_demultiplexing(map(read_json, arguments.records)))
    elif arguments.command == "rna-numerical-reproducibility":
        record = rna_numerical_reproducibility(
            read_json(arguments.pca), read_json(arguments.determinism)
        )
        write("rna_numerical_reproducibility.json", record)
    elif arguments.command == "doublets":
        import pandas as pd

        with tempfile.TemporaryDirectory() as folder:

            def read_scores(location: str, dataset: str) -> Any:
                target = Path(folder) / Path(location).stem / dataset
                target.mkdir(parents=True, exist_ok=True)
                path = fetch(doublet_scores(location, dataset), target)
                return pd.read_csv(path, float_precision="round_trip")

            jobs = {location: read_json(location) for location in arguments.records}
            record, tables = doublets(jobs, read_scores)
        write("doublets.json", record)
        for dataset, table in tables.items():
            path = RECORDS / "doublets" / f"{dataset}.csv.gz"
            table.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            print(f"wrote {path.relative_to(HERE.parents[1])}", flush=True)
    else:
        write("fate_probability_comparisons.json", fate_probabilities(read_json(arguments.output)))
        path = RECORDS / "raw" / "fate_probabilities" / "larry_clonal_fate_cells.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        objects(arguments.output).download(fate_cells(arguments.output), path)
        print(f"wrote {path.relative_to(HERE.parents[1])}", flush=True)


if __name__ == "__main__":
    main()
