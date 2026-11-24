import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from experiments.lung import DATA
from experiments.runtime.common import ROOT
from experiments.runtime.locations import DATASETS_PREFIX, RESULTS_PREFIX, experiment_bucket

RUN_TAG = "lung-reference-screen"
TABLES = ("natri_cells", "replication_cells", "unexplained_clusters", "unexplained_markers")
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


def remote_objects(run_tag: str):
    import boto3

    storage = experiment_bucket()
    results_uri = storage.uri(RESULTS_PREFIX)
    uri = urlsplit(f"{results_uri}/results/biocase/{run_tag}/")
    client = boto3.client(
        "s3",
        endpoint_url=storage.endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket, prefix = (uri.netloc, uri.path.lstrip("/"))
    replacements = [
        (results_uri, "<results>"),
        (storage.uri(DATASETS_PREFIX), "<datasets>"),
        (storage.endpoint, "<endpoint>"),
    ]
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [item["Key"] for item in page.get("Contents", [])]

    def reader(key: str):

        def read(target: Path | None = None) -> bytes | None:
            if target is None:
                return client.get_object(Bucket=bucket, Key=key)["Body"].read()
            client.download_file(bucket, key, str(target))
            return None

        return read

    return ([(key[len(prefix) :], reader(key)) for key in sorted(keys)], scrubber(replacements))


def local_objects(folder: Path):

    def reader(path: Path):

        def read(target: Path | None = None) -> bytes | None:
            if target is None:
                return path.read_bytes()
            shutil.copyfile(path, target)
            return None

        return read

    items = [
        (str(p.relative_to(folder)), reader(p)) for p in sorted(folder.rglob("*")) if p.is_file()
    ]
    return (items, scrubber([(str(folder.parents[2]), "<results>")]))


def provenance(run_tag: str, source: str) -> dict:

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except OSError, subprocess.CalledProcessError:
            return None

    return {
        "runTag": run_tag,
        "collector": "collect_lung.py",
        "source": source,
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
        "capturedAt": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--local", type=Path, help="a local results folder (the local test)")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if args.local:
        items, scrub = local_objects(args.local.resolve())
        source = "local test"
    else:
        items, scrub = remote_objects(args.run_tag)
        source = "<results>/results/biocase/" + args.run_tag
    raw = args.data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, read in items:
        if name.endswith(".json") and "/" not in name:
            records[name.removesuffix(".json")] = scrub(json.loads(read()))
        elif name.removesuffix(".csv.gz") in TABLES:
            target = raw / f"lung_reference_screen_{name}"
            read(target)
            print(f"wrote {target}", flush=True)
    document = {"provenance": provenance(args.run_tag, source), "records": records}
    path = args.data / "lung_reference_screen_records.json"
    path.write_text(json.dumps(document, indent=1, default=str) + "\n")
    print(
        f"wrote {path}: {sorted(records)} ({ {k: r.get('status') for k, r in records.items()} })",
        flush=True,
    )


if __name__ == "__main__":
    main()
