"""Collect the cluster-quality comparisons into records/cluster_quality.json.

Each comparison is the record that ``cluster_quality.quality_job`` wrote for one dataset, at
``experiments/kept_outputs/agreements/<dataset>.json`` in the experiment bucket.
"""

from pathlib import Path

from experiments.runtime.common import scrub_private, write_json

RECORDS = Path(__file__).resolve().parent / "records"
DATASETS = ("embryo_1000000", "brain_1306127")
# Run tags, placement, dates, code hashes and versions stay out of the record.
DROPPED = frozenset(
    {
        "runTag",
        "region",
        "startedAt",
        "finishedAt",
        "sourceSha256",
        "experimentArchiveSha256",
        "versions",
    }
)
# Each compared run keeps its tool, settings and results, without locations or environment.
RUN_DROPPED = frozenset({"datasetUri", "outputs", "record", "versions", "image"})


def _container(container: dict) -> dict:
    return {key: value for key, value in container.items() if key != "region"}


def comparison_record(record: dict) -> dict:
    """A dataset's comparison without run tags, placement, dates, code hashes or versions."""
    kept = {key: value for key, value in record.items() if key not in DROPPED}
    kept["container"] = _container(record["container"])
    kept["runs"] = {
        name: {
            **{key: value for key, value in run.items() if key not in RUN_DROPPED},
            "container": _container(run["container"]),
        }
        for name, run in record["runs"].items()
    }
    return scrub_private(kept)


def main() -> None:
    from experiments.runtime.locations import experiment_bucket
    from experiments.runtime.object_io import get_json

    bucket = experiment_bucket()
    records = {}
    for dataset in DATASETS:
        record = get_json(bucket.uri(f"experiments/kept_outputs/agreements/{dataset}.json"))
        if record.get("status") != "ok":
            raise SystemExit(f"the {dataset} comparison has status {record.get('status')!r}")
        records[dataset] = comparison_record(record)
    path = write_json("cluster_quality.json", {"records": records}, directory=RECORDS)
    print(f"wrote {path} ({len(records)} datasets)", flush=True)


if __name__ == "__main__":
    main()
