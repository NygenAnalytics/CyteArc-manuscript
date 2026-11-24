import re

import obstore

from experiments.benchmarks import HERE
from experiments.benchmarks.scaling import REPLICATES, attempts, profiling_config, profiling_runs
from experiments.runtime.common import BENCHMARK_GENES, write_json

ARTIFACT = re.compile("^RNA/artifacts/([^/]+)/([^/]+)/")


def first_replicate_tags() -> dict[int, str]:
    from experiments.runtime.object_io import object_exists

    tags = {}
    for run in profiling_runs():
        if run.group != "rna_scaling_runs" or run.replicate != REPLICATES[0]:
            continue
        for tag in attempts(run):
            if object_exists(profiling_config(tag).workflowResultUri(run.size)):
                tags[run.size] = tag
    return tags


def split_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return (f"s3://{bucket}", key)


def classify(relative: str) -> str:
    if relative.startswith("RNA/countsT/"):
        return "countsT"
    if relative.startswith("RNA/counts/"):
        return "counts"
    match = ARTIFACT.match(relative)
    if match:
        return f"artifact:{match.group(1)}"
    if relative.startswith("cellData/"):
        return "cellData"
    if relative.startswith("RNA/featureData/"):
        return "featureData"
    if relative.startswith("pipeline/"):
        return "pipeline"
    return "other"


def main() -> None:
    from experiments.runtime.object_io import storage_options

    per_size = {}
    for n_rows, tag in first_replicate_tags().items():
        config = profiling_config(tag)
        bucket, store_key = split_uri(config.storeUri(n_rows))
        store = obstore.store.from_url(bucket, **storage_options(bucket + "/"))
        prefix = store_key.rstrip("/") + "/"
        funnel_bucket, funnel_key = split_uri(config.workflowResultUri(n_rows))
        results = obstore.store.from_url(funnel_bucket, **storage_options(funnel_bucket + "/"))
        ended = obstore.head(results, funnel_key)["last_modified"]
        totals: dict[str, int] = {}
        counts: dict[str, int] = {}
        artifact_ids: dict[str, set[str]] = {}
        artifacts: dict[tuple[str, str], list] = {}
        n_objects = 0
        for page in obstore.list(store, prefix=prefix, chunk_size=1000):
            for item in page:
                relative = item["path"][len(prefix) :]
                match = ARTIFACT.match(relative)
                if match:
                    entry = artifacts.setdefault(
                        (match.group(1), match.group(2)), [0, 0, item["last_modified"]]
                    )
                    entry[0] += int(item["size"])
                    entry[1] += 1
                    entry[2] = min(entry[2], item["last_modified"])
                    continue
                component = classify(relative)
                totals[component] = totals.get(component, 0) + int(item["size"])
                counts[component] = counts.get(component, 0) + 1
                n_objects += 1
        later: dict[str, dict[str, int]] = {}
        for (kind, artifact_id), (size, objects, first) in sorted(artifacts.items()):
            if first > ended:
                entry = later.setdefault(kind, {"artifacts": 0, "bytes": 0})
                entry["artifacts"] += 1
                entry["bytes"] += size
                continue
            component = f"artifact:{kind}"
            totals[component] = totals.get(component, 0) + size
            counts[component] = counts.get(component, 0) + objects
            artifact_ids.setdefault(kind, set()).add(artifact_id)
            n_objects += objects
        h5ad_size = None
        try:
            meta = obstore.head(store, split_uri(config.datasetUri(n_rows))[1])
            h5ad_size = int(meta["size"])
        except Exception as error:
            print(f"{n_rows}: no H5AD size ({error})")
        total = sum(totals.values())
        count_bytes = totals.get("counts", 0) + totals.get("countsT", 0)
        record = {
            "nRows": n_rows,
            "tag": tag,
            "objects": n_objects,
            "totalBytes": total,
            "countsBytes": totals.get("counts", 0),
            "countsTBytes": totals.get("countsT", 0),
            "analysisBytes": total - count_bytes,
            "denseLogicalBytes": n_rows * BENCHMARK_GENES * 4,
            "h5adSampleBytes": h5ad_size,
            "componentBytes": dict(sorted(totals.items())),
            "componentObjects": dict(sorted(counts.items())),
            "artifactCounts": {kind: len(ids) for kind, ids in sorted(artifact_ids.items())},
            "laterArtifacts": later,
        }
        per_size[str(n_rows)] = record
        print(
            f"{n_rows:>10,}: store {total / 2**30:6.2f} GiB, counts {record['countsBytes'] / 2**30:6.2f}, countsT {record['countsTBytes'] / 2**30:6.2f}, analysis {record['analysisBytes'] / 2**30:6.2f}, h5ad {(h5ad_size or 0) / 2**30:6.2f} GiB, objects {n_objects}"
        )
    write_json(
        "analysis_store_sizes.json",
        {
            "description": "Bytes by component for the first replicate store of each benchmark size, as its benchmark run left it. artifact:* keys are per artifact kind; the ANN index kind is the largest component a mapping reference must read. laterArtifacts: artifacts written by later jobs (after the run's funnel record), by kind, left out of the sums.",
            "perSize": per_size,
        },
        directory=HERE / "records",
    )


if __name__ == "__main__":
    main()
