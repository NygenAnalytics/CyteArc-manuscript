import json
import tempfile
import unittest
from pathlib import Path

from experiments.integration import multimodal_neighbors_summary as summary

FIXTURE_CHECK = {
    "experimentalConditions": {"machine": {"cpuCount": 8}},
    "description": "replay",
    "twoModality": {"cells": 300, "matchedPool": {"neighborIndicesIdentical": True}},
    "threeModality": {"cells": 48, "matchedPool": {"neighborIndicesIdentical": True}},
}


def envelope(result: dict, call_id: str, status: str = "completed") -> dict:
    return {"status": status, "callId": call_id, "result": result}


def workflow(dataset: str, key: str) -> dict:
    return {
        "dataset": dataset,
        "inputsSha256": key,
        "reusedArtifacts": [],
        "cells": 100,
        "modalities": ["RNA", "ADT"],
        "settings": {"k": 20},
        "sourceArchiveSha256": "a" * 64,
        "cellSelectionSha256": "b" * 64,
    }


def replay(key: str, passed: bool = True) -> dict:
    return {"inputsSha256": key, "matchedKernel": {"passed": passed}}


def dimensions(key: str) -> dict:
    return {"inputsSha256": key, "residentCountBytes": 1, "modalityDimensions": {"RNA": 50}}


class MultimodalNeighborsSummaryTests(unittest.TestCase):
    def write_records(self, folder: Path, envelopes: dict[str, dict]) -> None:
        for name, document in envelopes.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document))

    def complete_records(self, folder: Path, cite_passed: bool = True) -> None:
        cite, tea = summary.SMALL_DATASETS
        self.write_records(
            folder,
            {
                "cite/workflow.json": envelope(workflow(cite, "c"), "fc-1"),
                "cite/source.json": envelope(dimensions("c"), "fc-2"),
                "cite/replay.json": envelope(replay("c", cite_passed), "fc-3"),
                "tea/workflow.json": envelope(workflow(tea, "t"), "fc-4"),
                "tea/source.json": envelope(dimensions("t"), "fc-5"),
                "tea/replay.json": envelope(replay("t"), "fc-6"),
                # Incomplete calls, and later copies of a call already read, are left out.
                "tea/failed.json": envelope(replay("t", False), "fc-7", status="failed"),
                "tea/replay_copy.json": envelope(replay("t", False), "fc-6"),
            },
        )

    def test_summary_takes_the_fixture_check_from_its_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.complete_records(folder)
            result = summary.summarise(summary.completed_records(folder), FIXTURE_CHECK)
        self.assertEqual(
            list(result), ["comparison", "sharedPoolCondition", "datasets", "sharedPoolGolden"]
        )
        self.assertEqual(
            result["sharedPoolGolden"],
            {
                "twoModality": FIXTURE_CHECK["twoModality"],
                "threeModality": FIXTURE_CHECK["threeModality"],
            },
        )
        self.assertEqual(sorted(result["datasets"]), sorted(summary.SMALL_DATASETS))
        cite = result["datasets"][summary.SMALL_DATASETS[0]]
        self.assertEqual(cite["modalityDimensions"], {"RNA": 50})
        # Records are kept whole, without the Modal call ID.
        self.assertEqual(cite["records"], [{"status": "completed", "result": replay("c")}])
        tea = result["datasets"][summary.SMALL_DATASETS[1]]
        self.assertEqual(tea["records"], [{"status": "completed", "result": replay("t")}])

    def test_a_failed_kernel_replay_stops_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.complete_records(folder, cite_passed=False)
            with self.assertRaisesRegex(ValueError, "Numerical validation failed"):
                summary.summarise(summary.completed_records(folder), FIXTURE_CHECK)

    def test_a_missing_dataset_stops_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.complete_records(folder)
            for name in ("replay.json", "replay_copy.json"):
                (folder / "tea" / name).unlink()
            with self.assertRaisesRegex(ValueError, "swanson_7K_pbmc_teaseq"):
                summary.summarise(summary.completed_records(folder), FIXTURE_CHECK)

    def test_another_memory_metric_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            stage = {"operationPeakBytes": 1, "operationPeakSource": "rss"}
            self.write_records(folder, {"run.json": envelope({"stages": [stage]}, "fc-1")})
            with self.assertRaisesRegex(ValueError, "different memory metric"):
                summary.completed_records(folder)

    def test_committed_records_give_both_datasets_and_the_fixture_values(self) -> None:
        fixture_check = json.loads(
            (summary.RECORDS / "weighted_neighbors_seurat_agreement.json").read_text()
        )
        result = summary.summarise(
            summary.completed_records(summary.RECORDS / "multimodal_neighbors"), fixture_check
        )
        self.assertEqual(sorted(result["datasets"]), sorted(summary.SMALL_DATASETS))
        for name in ("twoModality", "threeModality"):
            self.assertTrue(
                result["sharedPoolGolden"][name]["matchedPool"]["neighborIndicesIdentical"]
            )
        self.assertLess(
            result["sharedPoolGolden"]["twoModality"]["matchedPool"]["modalityWeightsMaxAbs"], 1e-6
        )


if __name__ == "__main__":
    unittest.main()
