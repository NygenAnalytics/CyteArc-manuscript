import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import experiments.benchmarks.collect_tahoe as collect
import numpy as np
import pandas as pd
from experiments.benchmarks.config import load_profiling_config

# Loading a configuration only builds bucket URIs, so placeholder locations stand in for the bucket.
BUCKET = {"R2_ENDPOINT_URL": "https://storage.invalid", "R2_BUCKET": "experiments-bucket"}


def setUpModule() -> None:
    patcher = mock.patch.dict(os.environ, BUCKET)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


# Seconds per stage chosen so every total below can be worked out by hand.
FULL_SECONDS = {stage: 3600.0 for stage in collect.STAGES}
FIXED_SECONDS = {
    "createStore": 7200.0,
    "writeCountsT": 3600.0,
    "initializeStore": 1800.0,
    "reopenStore": 10.0,
    "filterCells": 10.0,
    "markHvgs": 4 * 3600.0,
    "runNormalization": 2 * 3600.0,
    "importClusters": 600.0,
    "findMarkers": 3 * 3600.0,
}
FULL_READS = {"markHvgs": 100, "runNormalization": 10, "findMarkers": 100}
FIXED_READS = {"markHvgs": 800, "runNormalization": 80, "findMarkers": 800}


def record(seconds: float, reads: int | None = None, status: str = "ok") -> dict:
    details: dict = {"download": {"downloadSeconds": 5.0}}
    if reads is not None:
        details["storeOperations"] = {"rangeGets": reads, "transport": {"getRequests": 2 * reads}}
    return {
        "status": status,
        "seconds": seconds,
        "peakCgroupBytes": 2 * collect.GIB,
        "peakRssBytes": collect.GIB,
        "details": details,
        "placement": {"region": "us-east"},
    }


class FakeBucket:
    """Answers the collector's object reads from synthetic stage, staging and export records."""

    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        configs = {label: load_profiling_config(path) for label, path in collect.RUNS.items()}
        fixed = load_profiling_config(collect.FIXED_LAYOUT)
        for label, config in configs.items():
            size = config.targetSizes[0]
            for stage in collect.STAGES:
                seconds = FULL_SECONDS[stage] / (16 if label == "test" else 1)
                self.objects[config.resultUri(size, stage)] = record(seconds, FULL_READS.get(stage))
        for stage in fixed.effectiveStages:
            self.objects[fixed.resultUri(fixed.targetSizes[0], stage)] = record(
                FIXED_SECONDS[stage], FIXED_READS.get(stage)
            )
        # Replicate n takes n times the full run's seconds in every stage.
        for number, path in collect.FULL_REPLICATES.items():
            replicate = load_profiling_config(path)
            for stage in collect.STAGES:
                self.objects[replicate.resultUri(replicate.targetSizes[0], stage)] = record(
                    number * FULL_SECONDS[stage], FULL_READS.get(stage)
                )
            if number == 3:
                self.replicate_3_umap_uri = replicate.resultUri(replicate.targetSizes[0], "runUmap")
        full = configs["full"]
        self.export_uri = f"{full.resultPrefix(full.targetSizes[0])}/exports/export.json"
        self.objects[self.export_uri] = {
            "runTag": full.runTag,
            "cellsKept": 80,
            "leidenClusters": 2,
            "cellLines": 3,
            "medianCommonestCellLineFraction": 0.9,
            "cellsInClustersWithOneLineAbove90Percent": 40,
            "clusters": [],
            "files": {
                "umap_raster.npz": {"sha256": "raster"},
                "cluster_cell_lines.csv": {"sha256": "lines"},
                "categories.json": {"sha256": "categories"},
            },
            "layoutComparison": {"runTag": fixed.runTag, "filterCellsIdentical": True},
        }
        self.full_uri = full.resultUri(full.targetSizes[0], "markHvgs")
        self.downloads: dict[str, tuple[str, str, str]] = {}
        self.fixed_uri = fixed.resultUri(fixed.targetSizes[0], "markHvgs")

    def download_export_file(
        self, exports: str, name: str, expected_sha256: str, destination: Path
    ) -> None:
        self.downloads[name] = (exports, expected_sha256, destination.name)
        if name == "umap_raster.npz":
            # The export's raster holds two arrays the figure does not draw.
            np.savez_compressed(
                destination,
                cells=np.array([[0, 3], [5, 0]], dtype=np.uint32),
                commonestCellLine=np.array([[-1, 2], [0, -1]], dtype=np.int16),
                commonestCellLineFraction=np.array([[0, 1], [0.6, 0]], dtype=np.float32),
                extent=np.array([-1.0, 1.0, -2.0, 2.0]),
            )
        if name == "categories.json":
            # The export codes its three cell lines in this order; the raster's codes index it.
            destination.write_text(json.dumps({"cellLines": ["CVCL_C", "CVCL_A", "CVCL_B"]}))

    def download_cell_line_metadata(self, destination: Path) -> None:
        # Tahoe-100M's table repeats a line once per driver gene and lists lines the export lacks.
        pd.DataFrame(
            {
                "cell_name": ["Line A", "Line A", "Line B", "Line C", "Line D"],
                "Cell_ID_Cellosaur": ["CVCL_A", "CVCL_A", "CVCL_B", "CVCL_C", "CVCL_D"],
                "Organ": ["Lung", "Lung", "Bowel", "Skin, upper", "Breast"],
                "Driver_Gene_Symbol": ["KRAS", "STK11", "TP53", "BRAF", "PIK3CA"],
            }
        ).to_parquet(destination)

    def get_json(self, uri: str) -> dict:
        if "/index/" in uri:
            return {"nCells": 10, "nnz": 100, "maxCount": 7, "size": 1000}
        return self.objects[uri]


class CollectTahoeTest(unittest.TestCase):
    def run_collector(self, bucket: FakeBucket) -> tuple[dict, dict]:
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch("experiments.runtime.object_io.get_json", bucket.get_json),
                mock.patch.object(collect, "download_export_file", bucket.download_export_file),
                mock.patch.object(
                    collect, "download_cell_line_metadata", bucket.download_cell_line_metadata
                ),
                mock.patch.object(collect, "RECORDS", Path(directory)),
            ):
                collect.main()
            summary = json.loads((Path(directory) / "tahoe_rna_scaling_results.json").read_text())
            raw = json.loads(
                (Path(directory) / "raw" / "tahoe_rna_stage_measurements.json").read_text()
            )
            with (Path(directory) / "tahoe_cell_lines.csv").open(newline="") as handle:
                self.cell_lines = list(csv.DictReader(handle))
            with np.load(Path(directory) / "tahoe_umap_raster.npz") as raster:
                self.raster = {name: raster[name] for name in raster.files}
        return summary, raw

    def test_totals_and_layout_ratios(self):
        summary, raw = self.run_collector(FakeBucket())
        self.assertEqual(summary["totals"]["full"]["preparationHours"], 3.0)
        self.assertEqual(summary["totals"]["full"]["analysisHours"], 12.0)
        self.assertEqual(summary["totals"]["test"]["analysisHours"], 0.75)
        self.assertEqual(summary["stages"]["markHvgs"]["fullOverTestSeconds"], 16.0)
        fixed = summary["fixedLayout"]
        self.assertEqual(fixed["preparationHours"], 3.5)
        self.assertEqual(
            fixed["countPasses"],
            {
                "markHvgs": {"fixedOverScaledSeconds": 4.0, "fixedOverScaledRangeGets": 8.0},
                "runNormalization": {
                    "fixedOverScaledSeconds": 2.0,
                    "fixedOverScaledRangeGets": 8.0,
                },
                "findMarkers": {"fixedOverScaledSeconds": 3.0, "fixedOverScaledRangeGets": 8.0},
            },
        )
        self.assertEqual(fixed["stages"]["findMarkers"]["getRequests"], 1600)
        self.assertIsNone(fixed["stages"]["importClusters"]["rangeGets"])
        self.assertEqual(summary["dataset"]["cells"], 10 * 3388)
        self.assertEqual(summary["fullStore"]["leidenClusters"], 2)
        self.assertEqual(summary["fullStore"]["cellsKeptFraction"], 80 / 95624334)
        self.assertEqual(
            set(raw["runs"]),
            {"test", "full", "fullReplicate2", "fullReplicate3", "fixedLayout"},
        )
        self.assertEqual(len(raw["runs"]["fixedLayout"]["stages"]), 9)
        # The stage file keeps range reads and placement, and leaves out resident-set peaks.
        full_hvgs = raw["runs"]["full"]["stages"]["markHvgs"]
        self.assertEqual(full_hvgs["details"]["storeOperations"], {"rangeGets": 100})
        self.assertEqual(full_hvgs["placement"], {"region": "us-east"})
        self.assertNotIn("peakRssBytes", full_hvgs)
        self.assertNotIn("peakRssGiB", summary["stages"]["markHvgs"]["full"])
        self.assertNotIn("runTag", summary["runs"]["full"])
        self.assertEqual(summary["runs"]["full"]["nRows"], 95624334)

    def test_each_replicate_of_the_full_run_is_a_row_with_its_own_hours_and_stages(self):
        summary, raw = self.run_collector(FakeBucket())
        rows = summary["fullReplicates"]
        self.assertEqual([row["replicate"] for row in rows], [1, 2, 3])
        # Twelve one-hour analysis stages and three preparation stages, times the replicate number.
        self.assertEqual([row["analysisHours"] for row in rows], [12.0, 24.0, 36.0])
        self.assertEqual([row["preparationHours"] for row in rows], [3.0, 6.0, 9.0])
        self.assertEqual(
            [row["stages"]["runUmap"] for row in rows],
            [{"seconds": 3600.0 * n, "peakCgroupGiB": 2.0} for n in (1, 2, 3)],
        )
        self.assertEqual(set(rows[0]["stages"]), set(collect.STAGES))
        # Replicate 1 is the full run, which the totals and stage comparison keep reporting.
        self.assertEqual(rows[0]["analysisHours"], summary["totals"]["full"]["analysisHours"])
        self.assertEqual(raw["runs"]["fullReplicate3"]["stages"]["runUmap"]["seconds"], 10800.0)
        self.assertEqual(
            raw["runs"]["fullReplicate2"]["runTag"],
            "tahoe_rna_95624334_cells_scaled_layout_replicate_2",
        )

    def test_a_replicate_with_an_incomplete_stage_fails(self):
        bucket = FakeBucket()
        bucket.objects[bucket.replicate_3_umap_uri]["status"] = "failed"
        with self.assertRaisesRegex(SystemExit, "replicate_3.*runUmap"):
            self.run_collector(bucket)

    def test_a_replicate_of_other_cells_fails(self):
        bucket = FakeBucket()
        with mock.patch.dict(collect.FULL_REPLICATES, {2: collect.RUNS["test"]}):
            with self.assertRaisesRegex(
                SystemExit, "Replicate 2 differs from the full run in nRows"
            ):
                self.run_collector(bucket)

    def test_the_raster_and_cluster_table_come_from_the_full_run_export(self):
        bucket = FakeBucket()
        self.run_collector(bucket)
        exports = bucket.export_uri.removesuffix("/export.json")
        self.assertEqual(
            bucket.downloads,
            {
                "umap_raster.npz": (exports, "raster", "umap_raster.npz"),
                "cluster_cell_lines.csv": (exports, "lines", "tahoe_cluster_cell_lines.csv"),
                "categories.json": (exports, "categories", "categories.json"),
            },
        )

    def test_the_raster_record_keeps_only_the_drawn_arrays_unchanged(self):
        self.run_collector(FakeBucket())
        self.assertEqual(sorted(self.raster), ["cells", "commonestCellLine"])
        self.assertEqual(self.raster["cells"].dtype, np.uint32)
        self.assertEqual(self.raster["commonestCellLine"].dtype, np.int16)
        self.assertEqual(self.raster["cells"].tolist(), [[0, 3], [5, 0]])
        self.assertEqual(self.raster["commonestCellLine"].tolist(), [[-1, 2], [0, -1]])

    def test_a_raster_without_a_drawn_array_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "umap_raster.npz"
            np.savez_compressed(path, cells=np.zeros((2, 2), dtype=np.uint32))
            with self.assertRaisesRegex(SystemExit, "lacks the arrays \\['commonestCellLine'\\]"):
                collect.drawn_raster(path)

    def test_the_cell_lines_follow_the_export_codes_with_the_hub_table_names(self):
        self.run_collector(FakeBucket())
        self.assertEqual(
            self.cell_lines,
            [
                {"code": "0", "cellLine": "CVCL_C", "name": "Line C", "organ": "Skin, upper"},
                {"code": "1", "cellLine": "CVCL_A", "name": "Line A", "organ": "Lung"},
                {"code": "2", "cellLine": "CVCL_B", "name": "Line B", "organ": "Bowel"},
            ],
        )

    def test_an_export_that_codes_other_than_its_count_of_cell_lines_fails(self):
        bucket = FakeBucket()
        bucket.objects[bucket.export_uri]["cellLines"] = 4
        with self.assertRaisesRegex(SystemExit, "counts 4 cell lines but codes 3"):
            self.run_collector(bucket)

    def test_a_cell_line_the_hub_table_does_not_name_fails(self):
        table = pd.DataFrame(
            {"Cell_ID_Cellosaur": ["CVCL_A"], "cell_name": ["Line A"], "Organ": ["Lung"]}
        )
        with self.assertRaisesRegex(SystemExit, r"names no cell line for \['CVCL_B'\]"):
            collect.cell_line_table(["CVCL_A", "CVCL_B"], table)

    def test_a_cell_line_with_two_names_or_a_repeated_code_fails(self):
        table = pd.DataFrame(
            {
                "Cell_ID_Cellosaur": ["CVCL_A", "CVCL_A"],
                "cell_name": ["Line A", "Line A2"],
                "Organ": ["Lung", "Lung"],
            }
        )
        with self.assertRaisesRegex(SystemExit, "two names or organs"):
            collect.cell_line_table(["CVCL_A"], table)
        with self.assertRaisesRegex(SystemExit, "more than once"):
            collect.cell_line_table(["CVCL_A", "CVCL_A"], table.iloc[:1])

    def test_an_export_file_that_differs_from_its_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tahoe_cluster_cell_lines.csv"

            def download_file(uri: str, path: Path) -> None:
                Path(path).write_text("cluster,cellLine,cells\n0,A,1\n")

            with mock.patch("experiments.runtime.object_io.download_file", download_file):
                with self.assertRaisesRegex(SystemExit, "differs from the cluster_cell_lines.csv"):
                    collect.download_export_file(
                        "s3://b/x", "cluster_cell_lines.csv", "0" * 64, destination
                    )
                expected = collect.sha256_path(destination)
                collect.download_export_file(
                    "s3://b/x", "cluster_cell_lines.csv", expected, destination
                )

    def test_an_incomplete_stage_fails(self):
        bucket = FakeBucket()
        bucket.objects[bucket.fixed_uri]["status"] = "failed"
        with self.assertRaisesRegex(SystemExit, "fixed_layout.*markHvgs"):
            self.run_collector(bucket)

    def test_an_export_of_another_run_fails(self):
        bucket = FakeBucket()
        bucket.objects[bucket.export_uri]["layoutComparison"]["runTag"] = "another_run"
        with self.assertRaisesRegex(SystemExit, "does not belong"):
            self.run_collector(bucket)

    def test_a_count_pass_without_reads_fails(self):
        bucket = FakeBucket()
        del bucket.objects[bucket.full_uri]["details"]["storeOperations"]
        with self.assertRaisesRegex(SystemExit, "markHvgs has no range-read count"):
            self.run_collector(bucket)


if __name__ == "__main__":
    unittest.main()


class TahoeCellLinesRecordTest(unittest.TestCase):
    """The promoted cell-line record agrees with the other records of the same export."""

    def test_the_record_codes_every_line_of_the_export_and_its_clusters(self):
        with (collect.RECORDS / "tahoe_cell_lines.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        results = json.loads((collect.RECORDS / "tahoe_rna_scaling_results.json").read_text())
        self.assertEqual(len(rows), results["fullStore"]["cellLines"])
        self.assertEqual([int(row["code"]) for row in rows], list(range(len(rows))))
        self.assertEqual(len({row["cellLine"] for row in rows}), len(rows))
        self.assertTrue(all(row["name"] and row["organ"] for row in rows))
        with (collect.RECORDS / "tahoe_cluster_cell_lines.csv").open(newline="") as handle:
            clustered = {row["cellLine"] for row in csv.DictReader(handle)}
        self.assertLessEqual(clustered, {row["cellLine"] for row in rows})
        with np.load(collect.RECORDS / "tahoe_umap_raster.npz") as raster:
            self.assertLess(int(raster["commonestCellLine"].max()), len(rows))
        # A549, Tahoe-100M's first cell line, is CVCL_0023 in Cellosaurus.
        self.assertEqual(
            rows[0], {"code": "0", "cellLine": "CVCL_0023", "name": "A549", "organ": "Lung"}
        )
