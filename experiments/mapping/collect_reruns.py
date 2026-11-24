"""Collect the neighbor-count forks on embryo analysis stores into records/artifact_reuse_runs.json.

Each fork's job (mapping_reruns.rerun_job) saves its result as results/embryo_rna/rerun_<referenceTag>.json in the
experiment bucket; the record keeps each result under rerun_<referenceTag>, without the store's location.

  uv run --locked python -m experiments.mapping.collect_reruns REFERENCE_TAG [REFERENCE_TAG ...]
"""

import argparse
from pathlib import Path

from experiments.mapping import HERE
from experiments.runtime.common import scrub_private, write_json_at
from experiments.runtime.locations import RESULTS_PREFIX, experiment_bucket

# The forks keep their results with the embryo benchmark runs, as the mounts do.
PREFIX = "embryo_rna"
DROPPED = ("referenceStoreUri",)


def main() -> None:
    from experiments.runtime.object_io import get_json

    parser = argparse.ArgumentParser(description=Path(__file__).stem.replace("_", " "))
    parser.add_argument("tags", nargs="+", metavar="REFERENCE_TAG")
    args = parser.parse_args()
    folder = f"{experiment_bucket().uri(RESULTS_PREFIX)}/results/{PREFIX}"
    results = {}
    for tag in args.tags:
        result = get_json(f"{folder}/rerun_{tag}.json")
        if result.get("referenceTag") != tag or result.get("experiment") != "rerun":
            raise SystemExit(f"rerun_{tag}.json does not hold the fork of {tag}")
        results[f"rerun_{tag}"] = {
            "result": scrub_private({k: v for k, v in result.items() if k not in DROPPED})
        }
        print(
            f"{tag}: fork {result['forkSeconds']:.1f} s against {result['fullRunAnalysisSeconds']:.1f} s stored"
        )
    write_json_at(HERE / "records" / "artifact_reuse_runs.json", {"results": results})


if __name__ == "__main__":
    main()
