"""Totals of the CyteBase release by object kind, store size quantiles and count shard sizes.

uv run --locked python -m experiments.catalogue.catalogue_objects_summary
"""

import statistics
from collections import defaultdict

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

RECORD = ROOT / "experiments" / "catalogue" / "records" / "catalogue_objects.json"
SUMMARY = DERIVED / "catalogue" / "catalogue_objects.json"
COUNT_KINDS = ("countsByCell", "countsByGene")


def quantiles(values: list[float]) -> dict:
    q = statistics.quantiles(values, n=20)
    return {
        "min": min(values),
        "p5": q[0],
        "p25": q[4],
        "median": statistics.median(values),
        "p75": q[14],
        "p95": q[18],
        "max": max(values),
    }


def summarize(per_store: dict) -> dict:
    total_bytes: dict[str, int] = defaultdict(int)
    total_objects: dict[str, int] = defaultdict(int)
    for store in per_store.values():
        for kind, value in store["bytes"].items():
            total_bytes[kind] += value
        for kind, value in store["objects"].items():
            total_objects[kind] += value
    nonzeros = sum(store["nonzeros"] for store in per_store.values())
    shard_sizes = [
        size
        for store in per_store.values()
        for kind in COUNT_KINDS
        for size in store["countShardBytes"][kind]
    ]
    return {
        "source": str(RECORD.relative_to(ROOT)),
        "totalBytesByKind": dict(total_bytes),
        "totalObjectsByKind": dict(total_objects),
        "totalNonzeros": nonzeros,
        "bytesPerNonzeroPooled": sum(total_bytes.values()) / nonzeros,
        "countBytesPerNonzeroPooled": sum(total_bytes[kind] for kind in COUNT_KINDS) / nonzeros,
        "storeBytes": quantiles([sum(store["bytes"].values()) for store in per_store.values()]),
        "storeObjects": quantiles([sum(store["objects"].values()) for store in per_store.values()]),
        "countShards": {"n": len(shard_sizes), "bytes": quantiles(shard_sizes)},
    }


def main() -> None:
    write_json_at(SUMMARY, summarize(load_json(RECORD)["perStore"]))


if __name__ == "__main__":
    main()
