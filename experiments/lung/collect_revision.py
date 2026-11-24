import argparse
import copy
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import experiments.lung.collect_lung as collect
from experiments.lung import HERE, WORK
from experiments.runtime.common import ROOT
from experiments.runtime.locations import DATASETS_PREFIX, RESULTS_PREFIX, experiment_bucket

MANUSCRIPT = ROOT
RECORDS_FOLDER = HERE / "records"


TEST_DATA = WORK / "reference_revision_test"
TAG = "lung-reference-update"
TEST_TAG = f"{TAG}-test"
PRIMARY_TAG = "lung-reference-screen"
STORES_FOLDER = "stores/lung-reference-screen"
PREFIX = "lung_reference_update"
CENTRES = ("natri", "lafyatis_2019", "misharin_budinger_2018", "schiller_2020", "sheppard_2020")
JOB_RECORDS = (
    "pipeline",
    "setup_test",
    "reference",
    *(f"centre_{c}" for c in CENTRES),
    *(f"update_{c}" for c in CENTRES),
    "rerun_natri",
)
RECORDS = (*JOB_RECORDS, "summary")
TABLES = (
    "label_changes",
    "donors",
    "transitions",
    "unexplained",
    "thresholds",
    "costs",
    "artifacts",
)
CELL_TABLES = tuple((f"cells_{c}_{v}" for c in CENTRES for v in ("v1", "v2")))
RAW_LIMIT = 5 * 1024**2
PRIMARY_RECORDS = ("convert_hlca_core", "convert_hlca_full", "convert_natri_ild", "map")
PRIMARY_TABLES = ("natri_cells.csv.gz", "replication_cells.csv.gz")
STORES = ("hlca_core", "hlca_full", "natri_ild")


class Remote:
    """The results in the experiment bucket, read with the R2 credentials of the environment
    (read-only requests)."""

    def __init__(self) -> None:
        import boto3

        bucket = experiment_bucket()
        results_uri = bucket.uri(RESULTS_PREFIX)
        results = urlsplit(results_uri + "/")
        self.client = boto3.client(
            "s3",
            endpoint_url=bucket.endpoint,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        self.bucket, self.base = (results.netloc, results.path.lstrip("/"))
        self.scrub = collect.scrubber(
            [
                (results_uri, "<results>"),
                (bucket.uri(DATASETS_PREFIX), "<datasets>"),
                (bucket.endpoint, "<endpoint>"),
            ]
        )

    def names(self, folder: str) -> list[str]:
        prefix = f"{self.base}{folder.strip('/')}/"
        keys = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix
        ):
            keys += [item["Key"][len(prefix) :] for item in page.get("Contents", [])]
        return sorted(keys)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=f"{self.base}{key}")
            return True
        except self.client.exceptions.ClientError:
            return False

    def read(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=f"{self.base}{key}")["Body"].read()

    def record(self, folder: str, name: str) -> dict | None:
        key = f"{folder}/{name}.json"
        return self.scrub(json.loads(self.read(key))) if self.exists(key) else None

    def download(self, key: str, target: Path) -> None:
        self.client.download_file(self.bucket, f"{self.base}{key}", str(target))


def _folder(test: bool) -> str:
    return f"results/biocase/{(TEST_TAG if test else TAG)}"


def provenance(run_tag: str) -> dict:

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(MANUSCRIPT), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except OSError, subprocess.CalledProcessError:
            return None

    return {
        "runTag": run_tag,
        "collector": Path(__file__).name,
        "source": f"<results>/results/biocase/{run_tag}",
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
        "capturedAt": datetime.now(UTC).isoformat(),
    }


def preflight(remote: Remote) -> None:
    problems = []
    primary = f"results/biocase/{PRIMARY_TAG}"
    for name in PRIMARY_RECORDS:
        record = remote.record(primary, name)
        status = None if record is None else record.get("status")
        line = f"{primary}/{name}.json: {status or 'missing'}"
        if name == "convert_hlca_core" and record is not None:
            reference = (record.get("reference") or {}).get("mappingReference") or {}
            line += (
                f"; mapping reference {('present' if reference.get('artifact_id') else 'missing')}"
            )
            if not reference.get("artifact_id"):
                problems.append(f"{name}: no mapping reference")
        print(line, flush=True)
        if status != "ok":
            problems.append(f"{name}.json is {status or 'missing'}")
    for name in PRIMARY_TABLES:
        found = remote.exists(f"{primary}/{name}")
        print(f"{primary}/{name}: {('present' if found else 'missing')}", flush=True)
        if not found:
            problems.append(f"{name} is missing")
    for name in STORES:
        found = remote.exists(f"{STORES_FOLDER}/{name}.zarr/zarr.json")
        print(f"{STORES_FOLDER}/{name}.zarr: {('present' if found else 'missing')}", flush=True)
        if not found:
            problems.append(f"the {name} store is missing")
    for test in (False, True):
        present = [
            name.removesuffix(".json")
            for name in remote.names(_folder(test))
            if name.endswith(".json")
        ]
        print(
            f"{_folder(test)}: {(', '.join(present) if present else 'no records yet')}", flush=True
        )
    if problems:
        print("not ready: " + "; ".join(problems), flush=True)
        raise SystemExit(1)
    print("ready: every input of the revision jobs is on R2", flush=True)


def status(remote: Remote, test: bool) -> None:
    folder = _folder(test)
    present = set(remote.names(folder))
    for name in RECORDS:
        if f"{name}.json" not in present:
            print(f"{name}: no record", flush=True)
            continue
        record = remote.record(folder, name) or {}
        started, finished = (record.get("startedAt"), record.get("finishedAt"))
        seconds = (finished or time.time()) - started if started else None
        steps = record.get("steps") or {}
        last = list(steps)[-1] if steps else None
        timing = f" after {seconds / 60:.1f} min" if seconds is not None else ""
        print(
            f"{name}: {record.get('status')}{timing}; current step {record.get('currentStep')}; last step {last}",
            flush=True,
        )
        if name == "pipeline":
            for job, outcome in (record.get("outcomes") or {}).items():
                print(
                    f"  {job}: {(outcome.get('returned') or {}).get('status') or next(iter(outcome), None)}",
                    flush=True,
                )
        if record.get("error"):
            print("  " + record["error"][-1500:].replace("\n", "\n  "), flush=True)
    others = sorted(present - {f"{n}.json" for n in RECORDS})
    print("other objects:", ", ".join(others) if others else "none", flush=True)


def summary_record(job: dict) -> dict:
    """The stats job's record as records/lung_reference_update_summary.json keeps it.

    The record keeps the revised reference, the five query stores, the update against the
    complete rerun and the job outcomes, with the datasets, container and parameters of the run.
    It leaves out the stats job's start and finish times, the region and the attempt flags of
    the update and the rerun."""
    if job.get("step") != "stats" or job.get("status") != "ok":
        raise ValueError(
            f"Expected a completed stats job record, got step {job.get('step')!r} "
            f"with status {job.get('status')!r}"
        )
    job = copy.deepcopy(job)
    record = {}
    for key, value in job.items():
        if key in ("startedAt", "finishedAt", "previousAttempt", "attempt"):
            continue
        if key == "provenance":
            record["experimentalConditions"] = {
                "datasets": value["datasets"],
                "container": {k: v for k, v in value["container"].items() if k != "region"},
                "parameters": value["parameters"],
                "design": value["design"],
            }
        elif key == "steps":
            record["steps"] = {
                name: {k: v for k, v in step.items() if k != "startedAt"}
                for name, step in value.items()
            }
        else:
            record[key] = value
    for centre in record["centres"].values():
        del centre["costs"]["notFirstAttempt"]
    clean = record["incrementalVersusComplete"]["clean"]
    del clean["updateNotFirstAttempt"], clean["rerunAttempt"]
    return record


def fetch(remote: Remote, test: bool, data: Path) -> None:
    folder = _folder(test)
    tag = TEST_TAG if test else TAG
    present = set(remote.names(folder))
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    summary = remote.record(folder, "summary")
    if summary is not None:
        path = data / f"{PREFIX}_summary.json"
        path.write_text(json.dumps(summary_record(summary), indent=2) + "\n")
        print(f"wrote {path}", flush=True)
    records = {
        name: record
        for name in JOB_RECORDS
        if f"{name}.json" in present and (record := remote.record(folder, name)) is not None
    }
    path = data / f"{PREFIX}_records.json"
    path.write_text(
        json.dumps({"provenance": provenance(tag), "records": records}, indent=1, default=str)
        + "\n"
    )
    print(
        f"wrote {path}: { {name: record.get('status') for name, record in records.items()} }",
        flush=True,
    )
    for name in TABLES:
        if f"{name}.csv" in present:
            target = data / f"{PREFIX}_{name}.csv"
            remote.download(f"{folder}/{name}.csv", target)
            print(f"wrote {target}", flush=True)
        else:
            print(f"missing {name}.csv", flush=True)
    for name in CELL_TABLES:
        if f"{name}.csv.gz" not in present:
            continue
        target = raw / f"{PREFIX}_{name}.csv.gz"
        remote.download(f"{folder}/{name}.csv.gz", target)
        if target.stat().st_size > RAW_LIMIT:
            target.unlink()
            print(f"left {name}.csv.gz on R2: larger than {RAW_LIMIT // 1024**2} MB", flush=True)
            continue
        print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "status", "show", "fetch"))
    parser.add_argument(
        "record", nargs="?", default="summary", help="for show: one of " + ", ".join(RECORDS)
    )
    parser.add_argument("--test", action="store_true", help="the test run of reference_revision.py")
    parser.add_argument(
        "--data",
        type=Path,
        help="where fetch writes (default experiments/lung/records, or reference_revision_test in the lung workspace with --test)",
    )
    args = parser.parse_args()
    remote = Remote()
    if args.command == "preflight":
        preflight(remote)
    elif args.command == "status":
        status(remote, args.test)
    elif args.command == "show":
        print(
            json.dumps(remote.record(_folder(args.test), args.record), indent=1, default=str),
            flush=True,
        )
    else:
        fetch(remote, args.test, args.data or (TEST_DATA if args.test else RECORDS_FOLDER))


if __name__ == "__main__":
    main()
