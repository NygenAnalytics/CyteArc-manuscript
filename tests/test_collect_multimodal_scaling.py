import json
import tempfile
import unittest
from pathlib import Path

from experiments.integration import collect_multimodal_scaling as collect
from experiments.integration import multimodal_neighbors_summary as summary

STAGE = {
    "stage": "RNA.features",
    "wallSeconds": 9.5,
    "operationBaselineBytes": 1,
    "operationPeakBytes": 9265803264,
    "operationPeakSource": "cgroupMemoryCurrent",
    "processTreeRssPeakBytes": 7,
    "cgroupMemoryCurrentPeakBytes": 9265803264,
    "memoryMaxBytes": 68719476736,
    "artifactPlans": [{"artifactId": "a" * 64}],
}


def cytearc_output(cells: int, joint: str) -> dict:
    return {
        "module": "integration.multimodal_neighbors",
        "function": "run",
        "callId": "fc-01AAAAAAAAAAAAAAAAAAAAAAAA",
        "region": "us-central",
        "startedAt": 1793372088.3,
        "status": "completed",
        "elapsedSeconds": 862.9,
        "result": {
            "status": "ok",
            "dataset": collect.DATASET,
            "jointMethod": joint,
            "cells": cells,
            "cellSelectionSha256": "b" * 64,
            "settings": {"seed": 4466},
            "biologicalAgreement": {"wnn": {"ari": 0.44, "clusters": 24}},
            "stages": [STAGE],
            "reuseStages": [STAGE],
            "freshRevisionStages": [STAGE],
            "reuseValidation": {"clusterLabelsIdentical": True, "clusterAri": 1.0},
            f"initial{joint.title()}Seconds": 796.28,
            f"revised{joint.title()}Seconds": 20.35,
            f"freshRevised{joint.title()}Seconds": 814.77,
            "compilationWarmupCells": 5000,
            "compilationWarmupStages": [STAGE],
            "compilationWarmupWallSeconds": 69.28,
            "provenance": {"cytearcVersion": "1.3.0"},
            "region": "us-central",
            "graphsSha256": "c" * 64,
            "graphsUri": "s3://bucket/experiments/multimodal/graphs.npz",
            "localRecord": "/tmp/multimodal/multimodal_neighbors.json",
        },
    }


def seurat_output(cells: int, outcome: str = "time_limit") -> dict:
    return {
        "module": "integration.multimodal_neighbors",
        "function": "seurat_workflow",
        "callId": "fc-01BBBBBBBBBBBBBBBBBBBBBBBB",
        "region": "us-west",
        "startedAt": 1793373529.8,
        "status": "completed",
        "result": {
            "status": "comparator_stopped",
            "dataset": collect.DATASET,
            "cells": cells,
            "inputsUri": "s3://bucket/experiments/multimodal/inputs.tar.gz",
            "cellSelectionSha256": "b" * 64,
            "outcome": outcome,
            "stages": [{**STAGE, "stage": "seurat.workflow.process"}],
            "stageWindows": [
                {
                    "stage": "input_setup",
                    "startedAt": 1.0,
                    "wallSeconds": 367.67,
                    "sampledPeakBytes": 5,
                    "samples": 9,
                },
                {
                    "stage": "initial:WNN.clustering",
                    "startedAt": 2.0,
                    "wallSeconds": None,
                    "sampledPeakBytes": 6,
                    "samples": 3,
                    "unfinished": True,
                },
            ],
            "sampledPeakBytes": 6,
            "memorySource": {
                "kind": "cgroupV1MemoryUsageInBytes",
                "file": "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            },
            "containerMemoryLimitBytes": 137438953472,
            "memoryLimitGiB": 120,
            "timeLimitSeconds": 28800,
            "provenance": {"cytearcVersion": "1.3.0"},
            "region": "us-west",
            "returnCode": -9,
            "lastStage": "initial:WNN.clustering",
            "logTail": ["Killed"],
        },
    }


class CollectScalingRecordsTests(unittest.TestCase):
    def source(self, folder: Path, cells: int) -> None:
        replicate = folder / f"cells_{cells}" / "replicate_1"
        for name, document in (
            ("wnn/workflow.json", cytearc_output(cells, "wnn")),
            ("snn_fusion/workflow.json", cytearc_output(cells, "snn")),
            ("seurat_workflow.json", seurat_output(cells)),
        ):
            (replicate / name).parent.mkdir(parents=True, exist_ok=True)
            (replicate / name).write_text(json.dumps(document))
        agreement = folder / f"cells_{cells}" / "agreement"
        agreement.mkdir(parents=True)
        (agreement / "matched_cluster_agreement.csv").write_text("method,ari\ncytearc_wnn,0.43\n")
        (agreement / "neighbor_purity.csv").write_text(
            "method,same_type_neighbor_fraction\nrna,0.81\n"
        )
        (agreement / "per_type_f1.csv").write_text("method,f1\n")

    def collect(self, sizes: tuple[int, ...]) -> Path:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for cells in sizes:
            self.source(directory / "source", cells)
        collect.write_records(directory / "source", directory / "records")
        return directory / "records"

    def test_records_keep_only_the_reported_measurements(self) -> None:
        records = self.collect((600_000,))
        wnn = json.loads((records / "cells_600000" / "cytearc_wnn_1.json").read_text())
        self.assertEqual(list(wnn), ["status", "result"])
        self.assertEqual(
            list(wnn["result"]),
            [
                "status",
                "dataset",
                "jointMethod",
                "cells",
                "cellSelectionSha256",
                "biologicalAgreement",
                "stages",
                "reuseStages",
                "freshRevisionStages",
                "reuseValidation",
                "initialWnnSeconds",
                "revisedWnnSeconds",
                "freshRevisedWnnSeconds",
                "compilationWarmupWallSeconds",
                "graphsSha256",
            ],
        )
        self.assertEqual(
            wnn["result"]["stages"],
            [
                {
                    "stage": "RNA.features",
                    "wallSeconds": 9.5,
                    "operationPeakBytes": 9265803264,
                    "operationPeakSource": "cgroupMemoryCurrent",
                    "memoryMaxBytes": 68719476736,
                }
            ],
        )
        seurat = json.loads((records / "cells_600000" / "seurat_1.json").read_text())
        self.assertEqual(seurat["result"]["outcome"], "time_limit")
        self.assertEqual(seurat["result"]["lastStage"], "initial:WNN.clustering")
        self.assertEqual(
            seurat["result"]["stageWindows"][1],
            {
                "stage": "initial:WNN.clustering",
                "wallSeconds": None,
                "sampledPeakBytes": 6,
                "unfinished": True,
            },
        )
        text = "".join(path.read_text() for path in records.rglob("*.json"))
        for private in (
            "s3://",
            "/tmp/",
            "/sys/",
            "fc-",
            "us-west",
            "us-central",
            "startedAt",
            "1.3.0",
            "Killed",
        ):
            self.assertNotIn(private, text)

    def test_tables_are_copied_and_purity_only_at_its_size(self) -> None:
        records = self.collect((100_000, 600_000))
        self.assertEqual(
            sorted(p.relative_to(records).as_posix() for p in records.rglob("*.csv")),
            [
                "cells_100000/matched_cluster_agreement.csv",
                "cells_600000/matched_cluster_agreement.csv",
                "cells_600000/neighbor_purity.csv",
            ],
        )
        self.assertEqual(
            (records / "cells_600000" / "neighbor_purity.csv").read_text(),
            "method,same_type_neighbor_fraction\nrna,0.81\n",
        )

    def test_the_summary_reads_the_records(self) -> None:
        records = self.collect((100_000,))
        found = summary.completed_records(records)
        self.assertEqual(len(found), 3)

    def test_a_failed_job_stops_the_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            self.source(source, 100_000)
            path = source / "cells_100000" / "replicate_1" / "seurat_workflow.json"
            path.write_text(json.dumps({**seurat_output(100_000), "status": "failed"}))
            with self.assertRaisesRegex(ValueError, "no completed job result"):
                collect.write_records(source, Path(directory) / "records")

    def test_a_missing_run_stops_the_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            self.source(source, 100_000)
            (source / "cells_100000" / "replicate_1" / "snn_fusion" / "workflow.json").unlink()
            with self.assertRaisesRegex(ValueError, "lacks cytearc_snn_1.json"):
                collect.write_records(source, Path(directory) / "records")

    def test_a_record_of_another_size_or_dataset_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            self.source(source, 100_000)
            path = source / "cells_100000" / "replicate_1" / "wnn" / "workflow.json"
            path.write_text(json.dumps(cytearc_output(300_000, "wnn")))
            with self.assertRaisesRegex(
                ValueError, "not \\('combat_stephenson_pbmc_citeseq', 100000\\)"
            ):
                collect.write_records(source, Path(directory) / "records")

    def test_committed_records_hold_only_the_kept_fields(self) -> None:
        for path in sorted(collect.RECORDS.rglob("*.json")):
            document = json.loads(path.read_text())
            self.assertEqual(list(document), ["status", "result"], path)
            self.assertLessEqual(set(document["result"]), set(collect.RESULT_FIELDS), path)
            for group in collect.STAGE_GROUPS:
                for stage in document["result"].get(group, []):
                    self.assertLessEqual(set(stage), set(collect.STAGE_FIELDS), path)
                    self.assertEqual(stage["operationPeakSource"], "cgroupMemoryCurrent", path)


if __name__ == "__main__":
    unittest.main()
