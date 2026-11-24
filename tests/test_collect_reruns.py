import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from experiments.mapping import collect_reruns

BUCKET = {"R2_ENDPOINT_URL": "https://storage.invalid", "R2_BUCKET": "experiments-bucket"}
TAG = "embryo_rna_scaling_10000000_cells_replicate_5"


def fork_result(tag: str = TAG) -> dict:
    return {
        "experiment": "rerun",
        "referenceTag": tag,
        "referenceStoreUri": f"s3://experiments-bucket/experiments/stores/{tag}/10000000.zarr",
        "forkSeconds": 6158.8,
        "fullRunAnalysisSeconds": 3338.9,
        "note": "read from s3://experiments-bucket/experiments/results/embryo_rna",
    }


class CollectRerunsTests(unittest.TestCase):
    def collect(self, results: dict[str, dict], tags: list[str]) -> dict:
        written = {}

        def get_json(uri: str) -> dict:
            return results[uri]

        def write_json_at(path: Path, document: dict) -> None:
            written[path] = json.loads(json.dumps(document))

        with (
            mock.patch.dict(os.environ, BUCKET),
            mock.patch.object(collect_reruns, "write_json_at", write_json_at),
            mock.patch("experiments.runtime.object_io.get_json", get_json),
            mock.patch.object(sys, "argv", ["collect_reruns", *tags]),
        ):
            collect_reruns.main()
        self.assertEqual(
            list(written), [collect_reruns.HERE / "records" / "artifact_reuse_runs.json"]
        )
        return next(iter(written.values()))

    def test_each_fork_is_kept_under_its_store_without_the_store_location(self) -> None:
        uri = f"s3://experiments-bucket/experiments/results/embryo_rna/rerun_{TAG}.json"
        record = self.collect({uri: fork_result()}, [TAG])
        self.assertEqual(list(record["results"]), [f"rerun_{TAG}"])
        result = record["results"][f"rerun_{TAG}"]["result"]
        self.assertNotIn("referenceStoreUri", result)
        self.assertEqual(
            (result["forkSeconds"], result["fullRunAnalysisSeconds"]), (6158.8, 3338.9)
        )
        self.assertNotIn("experiments-bucket", json.dumps(record))

    def test_a_result_of_another_store_is_refused(self) -> None:
        uri = f"s3://experiments-bucket/experiments/results/embryo_rna/rerun_{TAG}.json"
        other = fork_result("embryo_rna_scaling_10000000_cells_replicate_4")
        with self.assertRaisesRegex(SystemExit, "does not hold the fork"):
            self.collect({uri: other}, [TAG])


if __name__ == "__main__":
    unittest.main()
