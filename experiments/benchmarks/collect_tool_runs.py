import csv
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from experiments.runtime.common import is_in_records

RECORDS = Path(__file__).resolve().parent / "records"
# The sampled marker statistics kept as records: the first replicate on these datasets.
MARKER_SAMPLE_DATASETS = ("embryo_1000000", "brain_1306127", "tabula_sapiens")
RUN_TAG = "toolkit_comparison"
CYTEARC_TAG = "cytearc_rna_workflow"
# Time series that the comparison's records leave out.
SERIES = ("memorySeries", "diskSeries")
# What a rerun must report as the run it replaces did: the input's size and the sampled rows.
MEASURED_INPUT = ("nCells", "nGenes", "nnz", "groups", "sampleRows")
CYTEARC_NAME = re.compile(
    "(^|/)(cytearc|cytearc_limit\\d+|cytearc_matched|markers_cytearc)_c\\d+|xi_li_2021_cytearc"
)


def is_cytearc(name: str) -> bool:
    return bool(CYTEARC_NAME.search(name))


def provenance() -> dict:
    return {"capturedAt": datetime.now(UTC).isoformat()}


PRIVATE_KEYS = {
    "callId",
    "coordinatorCallId",
    "r2EndpointUrl",
    "outputPrefix",
    "ladderUri",
    "datasetUri",
    "inputUri",
    "queryUri",
    "commonUri",
    "groupsUri",
    "requiresUri",
    "git",
    "hostname",
    "lock",
    "lockSha256",
    "importCalls",
    "versions",
    "runTag",
}
MODAL_ID = re.compile("\\b(?:fc|fu|ap|ta|in|sb)-[0-9A-Za-z]{20,}\\b")


def scrubber(config: dict, tag: str = RUN_TAG):
    results = config["resultsUri"].rstrip("/")
    replacements = [
        (f"{results}/results/tools/{tag}/", "<campaign>/"),
        (f"{results}/", "<results>/"),
        (config["datasetPrefixUri"].rstrip("/") + "/", "<datasets>/"),
    ]
    if config.get("r2EndpointUrl"):
        replacements.append((config["r2EndpointUrl"], "<endpoint>"))
    for key in ("resultsUri", "datasetPrefixUri"):
        replacements.append((urlsplit(config[key]).netloc, "<bucket>"))

    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items() if k not in PRIVATE_KEYS}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            for old, new in replacements:
                value = value.replace(old, new)
            return MODAL_ID.sub("<modal-id>", value)
        return value

    return scrub


def client_and_prefix(tag: str = RUN_TAG):
    import boto3

    from experiments.benchmarks.config import TEMPLATE, load_profiling_config

    config = load_profiling_config(TEMPLATE).model_dump(mode="json")
    uri = urlsplit(config["resultsUri"].rstrip("/") + f"/results/tools/{tag}/")
    client = boto3.client(
        "s3",
        endpoint_url=config["r2EndpointUrl"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    return (client, uri.netloc, uri.path.lstrip("/"), scrubber(config, tag))


def sources() -> list[tuple[str, str, Any]]:
    found = []
    for tag in (RUN_TAG, CYTEARC_TAG):
        client, bucket, prefix, scrub = client_and_prefix(tag)
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                name = item["Key"][len(prefix) :]
                if is_cytearc(name) != (tag == CYTEARC_TAG):
                    continue
                found.append((item["Key"], name, scrub))
    return sorted(found, key=lambda entry: entry[1])


def scientific_record_name(record: dict, source_name: str, operation: str = "rna_workflow") -> str:
    if "tool" not in record:
        return source_name.removesuffix(".json").replace("/", "_")
    tool, dataset = record["tool"], record["dataset"]
    if tool.startswith("markers_"):
        operation, tool = "marker_statistics", tool.removeprefix("markers_")
    elif dataset.startswith("hlca"):
        operation, dataset = "lung_heldout_query", ""
    container, payload = record.get("container", {}), record.get("payload", {})
    cpu = record.get("cpu", container.get("cpu", payload.get("containerCpu")))
    replicate = record.get("replicate", payload.get("replicate"))
    if cpu is None or replicate is None:
        raise ValueError("A toolkit measurement omits its CPU or replicate identity")
    prefix = "_".join(value for value in (operation, dataset, tool) if value)
    stem = f"{prefix}_cpus_{cpu:g}_replicate_{replicate}"
    if "attempts" in record and "tiers" in record:
        return stem + "_memory_tiers"
    return stem + f"_memory_{container['memoryMb'] / 1024:g}gib"


def link_records(records: dict) -> None:
    names = {record["sourceRecordUri"]: name for name, record in records.items()}
    for name, record in records.items():
        del record["sourceRecordUri"]
        # Prepared inputs and checks have no tool; only tool measurements are cited by name.
        if "tool" in record:
            record["sourceRecordId"] = name
        for attempt in record.get("attempts") or []:
            uri = attempt.pop("uri", None)
            if uri is None:
                raise ValueError(f"Memory-tier attempt lacks its source record: {name}")
            attempt["recordId"] = names.get(uri) or (
                name.removesuffix("_memory_tiers") + f"_memory_{attempt['memoryGib']:g}gib"
            )


def run_records(run_tag: str, operation: str = "rna_workflow") -> tuple[dict, dict]:
    """The records of one run tag, scrubbed and named as the comparison names them, and the key of
    each record's sampled marker statistics where its run saved them."""
    client, bucket, prefix, scrub = client_and_prefix(run_tag)
    keys = [
        item["Key"]
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for item in page.get("Contents", [])
    ]
    records: dict = {}
    samples = {}
    for key in keys:
        if not key.endswith(".json"):
            continue
        source_name = key[len(prefix) :]
        record = scrub(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()))
        record.setdefault("sourceRecordUri", "<campaign>/" + source_name)
        name = scientific_record_name(record, source_name, operation)
        if name in records:
            raise ValueError(f"Duplicate scientific measurement ID: {name}")
        records[name] = record
        if key.removesuffix(".json") + "_sample.csv" in keys:
            samples[name] = key.removesuffix(".json") + "_sample.csv"
    link_records(records)
    return records, samples


def family(record: dict) -> tuple:
    container = record.get("container") or {}
    return (
        record.get("dataset"),
        record.get("tool"),
        container.get("cpu"),
        container.get("memoryMb"),
    )


def rerun_attempt(attempt: dict, run: dict) -> dict:
    """A ladder's attempt at a tier that a rerun replaced, with the rerun's outcome."""
    return {
        "memoryGib": attempt["memoryGib"],
        "recordId": attempt["recordId"],
        "status": run.get("status"),
        "failedStep": run.get("failedStep"),
        "workflowSeconds": run.get("workflowSeconds"),
        "peakCgroupBytes": run.get("peakCgroupBytes"),
    }


def replaced_runs(document: dict, runs: dict, reference: dict | None = None) -> dict:
    """``document`` with each of ``runs`` in place of its record of the same name.

    A rerun replaces a record only when it completed without a restart in a non-preemptible
    container, read the bytes that every other tool's run on its dataset read (in the records of
    ``reference``, by default the document's), and reports the input and sampled rows of the run it
    replaces; every run of its tool, dataset, CPUs and memory must be rerun, except attempts
    interrupted by preemption, which measured nothing. A memory-tier ladder whose attempts the
    reruns all replace is left out; a ladder with other attempts keeps them, and its attempts at
    the rerun tiers take the reruns' outcomes. The reruns' time series are left out, and every
    other record stays as it is, in order."""
    records = document["records"]
    added = sorted(set(runs) - set(records))
    if added:
        raise ValueError(f"Reruns replace records, and the document has no {added}")
    families = {family(run) for run in runs.values()}
    unreplaced = sorted(
        name
        for name, record in records.items()
        if name not in runs
        and "tool" in record
        and "tiers" not in record
        and record.get("status") != "interrupted"
        and family(record) in families
    )
    if unreplaced:
        raise ValueError(f"No rerun of {unreplaced}")
    for name, run in runs.items():
        if run.get("status") != "ok" or run.get("failedStep") is not None:
            raise ValueError(f"{name} did not complete: {run.get('status')!r}")
        if run.get("restarts") or run.get("interruptions"):
            raise ValueError(f"{name} restarted")
        if (run.get("container") or {}).get("preemptible") is not False:
            raise ValueError(f"{name} did not run in a non-preemptible container")
        read = (run.get("download") or {}).get("bytes")
        others = {
            record["download"]["bytes"]
            for record in (reference or records).values()
            if record.get("dataset") == run["dataset"]
            and "tool" in record
            and "download" in record
            and record["tool"] != run["tool"]
            and record["tool"].startswith("markers_") == run["tool"].startswith("markers_")
        }
        if others != {read}:
            raise ValueError(f"{name} read {read} bytes; the other tools read {sorted(others)}")
        changed = [key for key in MEASURED_INPUT if records[name].get(key) != run.get(key)]
        if changed:
            raise ValueError(f"{name} reports other {changed} than the run it replaces")
    ladders = {name.rsplit("_memory_", 1)[0] + "_memory_tiers" for name in runs} & set(records)
    dropped = {
        ladder
        for ladder in ladders
        if {attempt.get("recordId") for attempt in records[ladder]["attempts"]} <= set(runs)
    }
    kept = {}
    for name, record in records.items():
        if name in runs:
            kept[name] = {key: value for key, value in runs[name].items() if key not in SERIES}
        elif name in ladders and name not in dropped:
            kept[name] = {
                **record,
                "attempts": [
                    rerun_attempt(attempt, runs[attempt["recordId"]])
                    if attempt.get("recordId") in runs
                    else attempt
                    for attempt in record["attempts"]
                ],
            }
        elif name not in dropped:
            kept[name] = record
    return {**document, "records": kept}


def sampled_pairs(path: Path) -> set[tuple[str, str]]:
    with path.open(newline="") as handle:
        return {(row["group"], row["gene"]) for row in csv.DictReader(handle)}


def replaced_documents(documents: dict[Path, dict], runs: dict) -> dict[Path, dict]:
    """Each records document with the reruns of its own records in place (see replaced_runs), for
    the documents that hold any. Each rerun must replace a record of exactly one document, and its
    input is checked against the other tools' runs in all of them."""
    holders = {
        name: [path for path, document in documents.items() if name in document["records"]]
        for name in runs
    }
    misplaced = {
        name: [str(path) for path in paths] for name, paths in holders.items() if len(paths) != 1
    }
    if misplaced:
        raise ValueError(f"Each rerun replaces a record of exactly one file: {misplaced}")
    reference = {
        f"{path}:{name}": record
        for path, document in documents.items()
        for name, record in document["records"].items()
    }
    replaced = {}
    for path, document in documents.items():
        own = {name: run for name, run in runs.items() if holders[name] == [path]}
        if own:
            replaced[path] = replaced_runs(document, own, reference)
    return replaced


def replace_in(run_tag: str, records_paths: list[Path], destination: Path | None = None) -> None:
    """Put the runs of ``run_tag`` in place of the records of the same names in ``records_paths``
    (see replaced_documents) and write each file that changed in place, or the one file to
    ``destination``, with the sampled statistics of the first replicate on MARKER_SAMPLE_DATASETS
    in raw/ beside it; those must sample the gene and group pairs of the other tools' tables in
    RECORDS/raw."""
    if destination is not None and len(records_paths) != 1:
        raise ValueError("--destination takes the result of a single --replace-in file")
    runs, samples = run_records(run_tag)
    documents = replaced_documents(
        {path: json.loads(path.read_text()) for path in records_paths}, runs
    )
    targets = {path: destination or path for path in documents}
    client, bucket, _, _ = client_and_prefix(run_tag)
    for name, key in sorted(samples.items()):
        run = runs[name]
        if run.get("replicate") != 1 or run.get("dataset") not in MARKER_SAMPLE_DATASETS:
            continue
        raw = (
            next(
                targets[path] for path, document in documents.items() if name in document["records"]
            ).parent
            / "raw"
        )
        raw.mkdir(parents=True, exist_ok=True)
        part = raw / f"{name}.csv.part"
        client.download_file(bucket, key, str(part))
        pairs = sampled_pairs(part)
        if len(pairs) != run.get("sampleRows"):
            raise ValueError(f"{key} samples {len(pairs)} pairs, not {run.get('sampleRows')}")
        for other in RECORDS.glob(f"raw/marker_statistics_{run['dataset']}_*_replicate_1_*.csv"):
            if other.name != f"{name}.csv" and sampled_pairs(other) != pairs:
                raise ValueError(f"{key} samples other gene and group pairs than {other.name}")
        part.replace(raw / f"{name}.csv")
        print(f"wrote {raw / f'{name}.csv'}")
    for path, document in documents.items():
        target = targets[path]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2) + "\n")
        own = sum(name in document["records"] for name in runs)
        print(f"{own} records of {run_tag} in place in {target}")


def collect_run(run_tag: str, destination: Path) -> None:
    operation = (
        "rna_cpu_scaling"
        if "cpu_scaling" in destination.name
        else "rna_storage_backend"
        if "storage_backend" in destination.name
        else "rna_workflow"
    )
    records, _ = run_records(run_tag, operation)
    destination = destination.resolve()
    if is_in_records(destination):
        raise ValueError(
            "Collect new runs into experiments/benchmarks/results; records contain frozen evidence"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "provenance": provenance(),
                "records": records,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(records)} records from {run_tag} into {destination}")


def main() -> None:
    global RUN_TAG, CYTEARC_TAG
    import argparse

    from experiments.benchmarks import DATA
    from experiments.runtime.common import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--cytearc-tag")
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--replace-in",
        type=Path,
        nargs="+",
        help="records files whose records of the same names the runs of --run-tag replace, each "
        "rerun in exactly one; written in place unless --destination names another file for one",
    )
    args = parser.parse_args()
    if args.replace_in is not None:
        if args.cytearc_tag is not None:
            parser.error("--replace-in takes the runs of one --run-tag; omit --cytearc-tag")
        replace_in(args.run_tag, args.replace_in, args.destination)
        return
    if args.cytearc_tag is None:
        collect_run(args.run_tag, args.destination or DATA / f"{args.run_tag}.json")
        return
    if args.destination is not None:
        parser.error("--destination is for a single run; omit --cytearc-tag")
    RUN_TAG, CYTEARC_TAG = args.run_tag, args.cytearc_tag
    client, bucket, _, _ = client_and_prefix(RUN_TAG)
    raw = RECORDS / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records, series, attempts, steps = ({}, {}, [], [])
    csv_sources = []
    for key, name, scrub in sources():
        if key.endswith(".json"):
            record = scrub(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()))
            if "memorySeries" in record:
                series[name] = record.pop("memorySeries")
            record.setdefault("sourceRecordUri", "<campaign>/" + name)
            name = scientific_record_name(record, name)
            if name in records:
                raise ValueError(f"Duplicate scientific measurement ID: {name}")
            records[name] = record
            if name.endswith("_memory_tiers") or "tool" not in record:
                continue
            container = record.get("container") or {}
            attempts.append(
                {
                    "record": name,
                    "dataset": record.get("dataset"),
                    "tool": record.get("tool"),
                    "replicate": record.get("replicate"),
                    "cpu": container.get("cpu"),
                    "memoryGib": (container.get("memoryMb") or 0) / 1024,
                    "status": record.get("status"),
                    "failedStep": record.get("failedStep"),
                    "workflowSeconds": record.get("workflowSeconds"),
                    "peakRssBytes": record.get("peakRssBytes"),
                    "peakCgroupBytes": record.get("peakCgroupBytes"),
                    "peakAnonBytes": record.get("peakAnonBytes"),
                    "nCells": record.get("nCells"),
                    "nnz": record.get("nnz"),
                    "cellsAfterFilter": record.get("cellsAfterFilter"),
                    "clusters": record.get("clusters"),
                    "restarts": record.get("restarts"),
                }
            )
            for step in record.get("stepOrder") or []:
                entry = (record.get("steps") or {}).get(step, {})
                steps.append(
                    {
                        "record": name,
                        "dataset": record.get("dataset"),
                        "tool": record.get("tool"),
                        "memoryGib": (container.get("memoryMb") or 0) / 1024,
                        "step": step,
                        "seconds": entry.get("seconds"),
                        "peakRssBytes": entry.get("peakRssBytes"),
                        "peakCgroupBytes": entry.get("peakCgroupBytes"),
                        "peakAnonBytes": entry.get("peakAnonBytes"),
                    }
                )
        elif key.endswith((".csv", ".csv.gz")):
            csv_sources.append((key, name))
    for key, source_name in csv_sources:
        suffix = "_sample.csv" if source_name.endswith("_sample.csv") else "_cells.csv"
        source_record = source_name.removesuffix(suffix) + ".json"
        matches = [
            (name, record)
            for name, record in records.items()
            if record.get("sourceRecordUri") == "<campaign>/" + source_record
        ]
        if len(matches) != 1:
            raise ValueError(f"CSV source lacks a unique toolkit measurement: {source_name}")
        name, record = matches[0]
        if record.get("replicate") != 1 or not (
            name.startswith("lung_heldout_query") or record.get("dataset") in MARKER_SAMPLE_DATASETS
        ):
            continue
        filename = name + (".csv" if suffix == "_sample.csv" else "_predictions.csv")
        target = raw / filename
        client.download_file(bucket, key, str(target))
        print(f"wrote {target}")
    link_records(records)
    document = {"provenance": provenance(), "records": records}
    path = write_json("toolkit_comparison_runs.json", document, directory=RECORDS)
    print(f"wrote {path}")
    for filename, rows in (("toolkit_step_measurements.csv", steps),):
        if rows:
            with (DATA / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    origin = f"{RUN_TAG}, CyteArc from {CYTEARC_TAG}"
    print(f"{len(records)} records, {len(attempts)} attempts, {len(steps)} step rows from {origin}")


if __name__ == "__main__":
    main()
