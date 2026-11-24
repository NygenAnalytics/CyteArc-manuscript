import argparse
import gzip
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from experiments.lung import HERE
from experiments.runtime.common import ROOT
from experiments.runtime.locations import DATASETS_PREFIX, RESULTS_PREFIX, experiment_bucket

MANUSCRIPT = ROOT

RUN_TAG = "lung-sensitivity"
RECORDS = (
    "check",
    "references",
    "extended_mount",
    "extended_reference",
    "query_projection",
    "query_references",
    "query_extended",
    "analysis",
)
FILES = {
    "robustness.json": "lung_sensitivity_summary.json",
    "robustness_outcomes.csv.gz": "lung_sensitivity_outcomes.csv",
    "robustness_costs.csv.gz": "lung_sensitivity_costs.csv",
    "robustness_clusters.csv.gz": "lung_sensitivity_clusters.csv.gz",
}
PRIVATE_KEYS = {
    "callId",
    "coordinatorCallId",
    "r2EndpointUrl",
    "modalFunctionCallId",
    "modalInputId",
}
S3_BUCKET = re.compile("s3://[^/\\s\\\"']+")
R2_ENDPOINT = re.compile("https://[^/\\s\\\"']+\\.r2\\.cloudflarestorage\\.com")
MODAL_ID = re.compile("\\b(?:fc|fu|ap|ta|in|sb)-[0-9A-Za-z]{20,}\\b")


def scrubber(replacements: list[tuple[str, str]]):

    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items() if k not in PRIVATE_KEYS}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            for old, new in replacements:
                value = value.replace(old, new)
            value = R2_ENDPOINT.sub("<endpoint>", S3_BUCKET.sub("s3://<bucket>", value))
            return MODAL_ID.sub("<modal-id>", value)
        return value

    return scrub


class Results:
    """The run's objects in the experiment bucket, read with the R2 credentials of the
    environment."""

    def __init__(self, run_tag: str) -> None:
        import boto3

        bucket = experiment_bucket()
        results_uri = bucket.uri(RESULTS_PREFIX)
        uri = urlsplit(f"{results_uri}/results/biocase/{run_tag}/")
        self.client = boto3.client(
            "s3",
            endpoint_url=bucket.endpoint,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        self.bucket, self.prefix = (uri.netloc, uri.path.lstrip("/"))
        self.scrub = scrubber(
            [
                (results_uri, "<results>"),
                (bucket.uri(DATASETS_PREFIX), "<datasets>"),
                (bucket.endpoint, "<endpoint>"),
            ]
        )

    def names(self) -> list[str]:
        keys = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self.prefix
        ):
            keys += [item["Key"][len(self.prefix) :] for item in page.get("Contents", [])]
        return sorted(keys)

    def read(self, name: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self.prefix + name)["Body"].read()

    def record(self, name: str) -> dict | None:
        try:
            return self.scrub(json.loads(self.read(f"{name}.json")))
        except self.client.exceptions.NoSuchKey:
            return None


def status(results: Results) -> None:
    for name in RECORDS:
        record = results.record(name)
        if record is None:
            print(f"{name}: no record", flush=True)
            continue
        steps = record.get("steps") or {}
        done = {k: round(v["seconds"], 1) for k, v in steps.items() if "seconds" in v}
        print(
            f"{name}: {record.get('status')}, current step {record.get('currentStep')}, steps {done}",
            flush=True,
        )
        if record.get("error"):
            print("  error: " + record["error"][-1500:], flush=True)


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


def collect(results: Results, data: Path, run_tag: str) -> None:
    available = set(results.names())
    for source, target in FILES.items():
        if source not in available:
            print(f"missing {source}", flush=True)
            continue
        body = results.read(source)
        path = data / target
        if source.endswith(".json"):
            document = {"collected": provenance(run_tag), **results.scrub(json.loads(body))}
            path.write_text(json.dumps(document, indent=1, default=str) + "\n")
        elif target.endswith(".csv"):
            path.write_bytes(gzip.decompress(body))
        else:
            path.write_bytes(body)
        print(f"wrote {path}", flush=True)
    records = {name: record for name in RECORDS if (record := results.record(name)) is not None}
    path = data / "lung_sensitivity_records.json"
    path.write_text(
        json.dumps({"provenance": provenance(run_tag), "records": records}, indent=1, default=str)
        + "\n"
    )
    print(f"wrote {path}: { {k: r.get('status') for k, r in records.items()} }", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--data", type=Path, default=HERE / "records", help="where the files go")
    parser.add_argument("--status", action="store_true", help="print the state of each job record")
    parser.add_argument(
        "--show", help="print one record (scrubbed), such as primary_check or references"
    )
    args = parser.parse_args()
    results = Results(args.run_tag)
    if args.status:
        status(results)
    elif args.show:
        print(json.dumps(results.record(args.show), indent=1, default=str), flush=True)
    else:
        collect(results, args.data, args.run_tag)


if __name__ == "__main__":
    main()
