"""Measure the objects of each CyteBase store from the release's object listing and catalogue.

Reads objects.json.gz and cytebase.duckdb, which ``cytebase_catalog --list-objects`` and
``cytebase_catalog`` download into $CYTEARC_DOWNLOADS/catalogue/, and writes
records/catalogue_objects.json: per store, its cells, genes, estimated nonzero counts, organism,
bytes and objects by kind and the size of each count shard.
"""

import gzip
import json
import statistics
from collections import defaultdict

from experiments.catalogue import HERE
from experiments.runtime.common import write_json_at

RECORDS = HERE / "records"
OUT = RECORDS / "catalogue_objects.json"
BUCKET = "Nygen/cytebase-2610"
SPATIAL_ASSAYS = set(
    json.loads((RECORDS / "cytebase_catalogue.json").read_text())["datasets"]["spatialAssays"]
)

KINDS = (
    ("countsByCell", "data.zarr/RNA/counts/"),
    ("countsByGene", "data.zarr/RNA/countsT/"),
    ("cellTables", "data.zarr/cellData/"),
    ("featureTables", "data.zarr/RNA/featureData/"),
    ("summaries", "data.zarr/RNA/countSummaries/"),
    ("artifacts", "data.zarr/RNA/artifacts/"),
    ("artifacts", "data.zarr/artifacts/"),
)


def kind_of(rel: str) -> str:
    for kind, prefix in KINDS:
        if rel.startswith(prefix):
            return kind
    return "metadata"


def catalogue(path) -> dict:
    import duckdb

    con = duckdb.connect(str(path), read_only=True)
    rows = con.execute(
        "select cytebase_id, cell_count, n_genes, organism_labels, assay_labels, mean_genes_per_cell from datasets"
    ).fetchall()
    return {
        r[0]: {
            "cells": r[1],
            "genes": r[2],
            "organisms": r[3],
            "assays": r[4],
            "meanGenesPerCell": r[5],
        }
        for r in rows
    }


def derive() -> None:
    from experiments.catalogue import INPUTS

    with gzip.open(INPUTS / "objects.json.gz", "rt") as fh:
        listing = json.load(fh)
    meta = catalogue(INPUTS / "cytebase.duckdb")
    stores: dict = defaultdict(
        lambda: {"bytes": defaultdict(int), "objects": defaultdict(int), "countShards": []}
    )
    outside = defaultdict(int)
    for path, size in listing["objects"]:
        parts = path.split("/", 2)
        if parts[0] != "datasets" or len(parts) < 3:
            outside[parts[0]] += size
            continue
        store, rel = parts[1], parts[2]
        kind = kind_of(rel)
        s = stores[store]
        s["bytes"][kind] += size
        s["objects"][kind] += 1
        if kind in ("countsByCell", "countsByGene") and "/c/" in rel:
            s["countShards"].append([kind, size])
    missing = sorted(set(meta) - set(stores))
    extra = sorted(set(stores) - set(meta))
    per_store = {}
    for sid, s in sorted(stores.items()):
        if sid not in meta:
            continue
        per_store[sid] = {
            "cells": meta[sid]["cells"],
            "genes": meta[sid]["genes"],
            # nonzero values: observations times the catalogue's mean genes detected per observation
            "nonzeros": round(meta[sid]["cells"] * meta[sid]["meanGenesPerCell"]),
            "organism": (meta[sid]["organisms"] or [None])[0],
            "spatial": any(a in SPATIAL_ASSAYS for a in (meta[sid]["assays"] or [])),
            "bytes": dict(s["bytes"]),
            "objects": dict(s["objects"]),
            "countShardBytes": {
                "countsByCell": [b for k, b in s["countShards"] if k == "countsByCell"],
                "countsByGene": [b for k, b in s["countShards"] if k == "countsByGene"],
            },
        }
    result = {
        "source": f"listing of hf://buckets/{BUCKET} (objects.json.gz) and its catalogue "
        "cytebase.duckdb (cell_count, n_genes, organism_labels, assay_labels, mean_genes_per_cell)",
        "bucketInfo": listing["bucketInfo"],
        "stores": len(per_store),
        "catalogueStoresWithoutObjects": missing,
        "listedStoresNotInCatalogue": extra,
        "outsideDatasetsBytes": dict(outside),
        "totalBytes": sum(sum(s["bytes"].values()) for s in per_store.values()),
        "byGeneOverByCellMedian": statistics.median(
            s["bytes"]["countsByGene"] / s["bytes"]["countsByCell"] for s in per_store.values()
        ),
        "totalObjects": sum(sum(s["objects"].values()) for s in per_store.values()),
        "perStore": per_store,
    }
    write_json_at(OUT, result)
    print(
        f"{len(per_store)} stores, {result['totalBytes'] / 1e12:.3f} TB, "
        f"{result['totalObjects']:,} objects; missing {len(missing)}, extra {len(extra)}"
    )


def main() -> None:
    derive()


if __name__ == "__main__":
    main()
