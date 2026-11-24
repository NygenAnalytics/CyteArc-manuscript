import argparse
import gzip
import hashlib
import json
import os
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

from experiments.catalogue import HERE
from experiments.catalogue import INPUTS as CACHE
from experiments.runtime.common import write_json_at

RELEASE = "2610"
BUCKET = f"Nygen/cytebase-{RELEASE}"
BASE = f"https://huggingface.co/buckets/{BUCKET}/resolve/catalog"
CATALOGUE = "cytebase.duckdb"
WORKDIR = CACHE
SPATIAL = ("Visium Spatial Gene Expression V1", "Slide-seqV2")
TERMS = (
    ("assays", "assay"),
    ("tissues", "tissue"),
    ("diseases", "disease"),
    ("cellTypes", "cell_type"),
)
HUMAN, MOUSE = ("Homo sapiens", "Mus musculus")
SELECTION = WORKDIR / "collections.json"
RECORD = HERE / "records" / "cytebase_catalogue.json"


def published_sha256() -> str:
    return (WORKDIR / f"{CATALOGUE}.sha256").read_text().split()[0]


def file_sha256() -> str:
    return hashlib.sha256((WORKDIR / CATALOGUE).read_bytes()).hexdigest()


def collect() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    for name in (f"{CATALOGUE}.sha256", CATALOGUE):
        with urllib.request.urlopen(f"{BASE}/{name}") as response:
            (WORKDIR / name).write_bytes(response.read())
    if file_sha256() != published_sha256():
        raise SystemExit("The downloaded catalogue does not match its published SHA-256")
    print(f"downloaded {BUCKET} catalogue, SHA-256 {published_sha256()}", flush=True)


def list_objects() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    token = os.environ.get("HF_TOKEN") or None
    info = api.bucket_info(BUCKET, token=token)
    rows = [
        [entry.path, entry.size or 0]
        for entry in api.list_bucket_tree(BUCKET, recursive=True, token=token)
        if getattr(entry, "type", "file") == "file"
    ]
    WORKDIR.mkdir(parents=True, exist_ok=True)
    destination = WORKDIR / "objects.json.gz"
    payload = {
        "bucket": BUCKET,
        "bucketInfo": {"size": info.size, "totalFiles": info.total_files},
        "objects": rows,
    }
    with gzip.open(destination, "wt") as handle:
        json.dump(payload, handle)
    print(f"listed {len(rows):,} objects into {destination}")


def build_summary() -> dict | None:
    if not SELECTION.exists():
        return None
    selection = json.loads(SELECTION.read_text())
    summary = selection["summary"]
    return {
        "collectionsExamined": summary["collectionCount"],
        "collectionsRegistered": summary["registrationReadyCollectionCount"],
        "collectionsHeldBack": summary["needsReviewCollectionCount"],
        "datasetsAwaitingAssayReview": summary["needsReviewDatasetCount"],
        "heldBackCollectionIds": selection["needsReviewCollectionIds"],
        "selectionSha256": hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
    }


def derive() -> None:
    import duckdb

    digest = file_sha256()
    if digest != published_sha256():
        raise SystemExit("The cached catalogue does not match its published SHA-256")
    con = duckdb.connect(str(WORKDIR / CATALOGUE), read_only=True)

    def one(sql: str):
        return con.execute(sql).fetchone()

    stores, collections, observations, primary = one(
        "select count(*), count(distinct collection_id), sum(cell_count), sum(primary_cell_count) from datasets"
    )
    smallest, largest, median = one(
        "select min(cell_count), max(cell_count), median(cell_count) from datasets"
    )
    statuses = dict(con.execute("select status, count(*) from datasets group by 1").fetchall())
    if one("select max(len(organism_labels)) from datasets")[0] != 1:
        raise SystemExit("A dataset with more than one organism: count organisms per dataset")
    organisms = dict(
        con.execute(
            "select organism_labels[1], count(*) from datasets group by 1 order by 2 desc, 1"
        ).fetchall()
    )
    others = {name: n for name, n in organisms.items() if name not in (HUMAN, MOUSE)}
    spatial = one(
        f"select count(*) from datasets where list_has_any(assay_labels, {list(SPATIAL)})"
    )[0]
    terms = {
        key: one(f"select count(distinct t) from (select unnest({column}_ids) t from datasets)")[0]
        for key, column in TERMS
    }
    total, skipped = one(
        "select sum(n_datasets_total), sum(len(skipped_dataset_ids)) from collections"
    )
    reasons = Counter()
    for (mapping,) in con.execute("select skipped_dataset_reasons from collections").fetchall():
        reasons.update((mapping or {}).values())
    if total - skipped != stores:
        raise SystemExit("Catalogued datasets minus skipped ones do not give the stores")
    build = build_summary()
    if build is not None and build["collectionsRegistered"] != collections:
        raise SystemExit("The selection's registered collections differ from the catalogue's")
    write_json_at(
        RECORD,
        {
            "release": RELEASE,
            "datasets": {
                "stores": stores,
                "collections": collections,
                "statuses": statuses,
                "observations": observations,
                "primaryObservations": primary,
                "smallest": smallest,
                "largest": largest,
                "median": median,
                "spatial": spatial,
                "spatialAssays": list(SPATIAL),
            },
            "organisms": {
                "count": len(organisms),
                "human": organisms.get(HUMAN, 0),
                "mouse": organisms.get(MOUSE, 0),
                "otherDatasets": sum(others.values()),
                "otherOrganisms": len(others),
                "perOrganism": organisms,
            },
            "terms": terms,
            "selection": {
                "cellxgeneDatasets": total,
                "skipped": skipped,
                "skippedByReason": dict(reasons.most_common()),
            },
            "build": build,
            "sources": {
                "bucket": BUCKET,
                "catalogue": f"hf://buckets/{BUCKET}/catalog/{CATALOGUE}",
                "sha256": digest,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=Path(__file__).stem.replace("_", " "))
    parser.add_argument("--no-collect", action="store_true", help="derive from the saved catalogue")
    parser.add_argument("--list-objects", action="store_true")
    parser.add_argument("--selection", type=Path, help="the build's collections.json")
    args = parser.parse_args()
    if args.list_objects:
        list_objects()
        return
    if args.selection:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.selection, SELECTION)
    if not args.no_collect:
        collect()
    derive()


if __name__ == "__main__":
    main()
