import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from experiments.integration import collect_cite_seq_jobs as cite

PRIVATE = ("fc-01ABCDEFGHIJKLMNOPQRSTUV", "s3://", "startedAt", "callId", ".scratch")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def envelope(result: dict) -> dict:
    """A job's run record as the Modal runner saves it."""
    return {
        "module": "integration.multimodal_neighbors",
        "function": "job",
        "callId": "fc-01ABCDEFGHIJKLMNOPQRSTUV",
        "region": "us-central",
        "startedAt": 1.0,
        "status": "completed",
        "elapsedSeconds": 9.0,
        "result": result,
    }


class CollectorTest(unittest.TestCase):
    """A temporary repository root with the experiment's records and a work folder in it."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        here = self.root / "experiments" / "integration"
        self.records = here / "records"
        self.work = self.root / "work"
        self.work.mkdir()
        for patch in (
            mock.patch.object(cite, "HERE", here),
            mock.patch.object(cite, "RECORDS", self.records),
            mock.patch("experiments.runtime.common.ROOT", self.root),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def assertPublic(self, path: Path) -> None:
        text = path.read_text()
        for private in PRIVATE:
            self.assertNotIn(private, text)


# Six cells: protein takes all, half, none, all, a quarter and three quarters of each cell's weight.
PROTEIN = np.array([1.0, 0.5, 0.0, 1.0, 0.25, 0.75], dtype=np.float32)
LABEL_CODES = np.array([1, 1, 2, 2, 0, 1])
STUDY_CODES = np.array([0, 0, 1, 1, 0, 1])


# The RNA layout spans 0 to 3 on both axes; with a 3% margin on each side the grid spans 3.18 from -0.09, so the last
# cell, at (3, 0), falls in column int(3.09 / 3.18 * 300) = 291 and row int(0.09 / 3.18 * 300) = 8.
RNA_LAYOUT = np.array([[0, 0], [1, 1], [2, 0], [0, 2], [3, 3], [3, 0]], dtype=np.float32)


class GraphUmapsTests(CollectorTest):
    def write_job(self, *, identical: bool = True, weights=PROTEIN, rna=RNA_LAYOUT) -> dict:
        rng = np.random.default_rng(0)
        path = self.work / "graph_umaps.npz"
        modality = np.stack([1 - weights, weights], axis=1).astype(np.float32)
        np.savez(
            path,
            RNA_umap=rna,
            ADT_umap=rng.normal(size=(6, 2)).astype(np.float32),
            WNN_umap=rng.normal(size=(6, 2)).astype(np.float32),
            label_codes=LABEL_CODES,
            label_names=np.array(["", "T cells", "B cells"]),
            study_codes=STUDY_CODES,
            study_names=np.array(["combat", "stephenson"]),
            modality_weights=modality,
            modalities=np.array(["RNA", "ADT"]),
        )
        means = modality.astype(np.float32).mean(axis=0)
        return envelope(
            {
                "graphsIdentical": identical,
                "graphsCheck": {"wnn": {"identical": identical}, "rna": {"identical": True}},
                "umapUri": "s3://bucket/experiments/multimodal/graph_umaps.npz",
                "umapSha256": sha256(path),
                "cells": 6,
                "modalityWeightMean": {"RNA": float(means[0]), "ADT": float(means[1])},
                "laidOut": "harmony",
                "harmony": {
                    "batchColumn": "study",
                    "batchCells": {"combat": 3, "stephenson": 3},
                    "correctionSeconds": 3.0,
                    "biologicalAgreement": {"wnn": 0.5},
                    "graphsUri": "s3://bucket/experiments/multimodal/harmony_graphs.npz",
                },
            }
        )

    def test_summary_rasters_and_weights_follow_the_cells(self) -> None:
        cite.graph_umaps(self.write_job(), self.work)
        folder = self.records / "graph_umaps_harmony"
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(summary["cellsPerLabel"], {"": 1, "T cells": 3, "B cells": 2})
        self.assertEqual(summary["cellsPerStudy"], {"combat": 3, "stephenson": 3})
        self.assertEqual(
            summary["proteinWeightShares"],
            {"one": 2 / 6, "zero": 1 / 6, "half": 1 / 6, "nearHalfNotHalf": 0.0},
        )
        self.assertAlmostEqual(summary["modalityWeightMean"]["ADT"], 3.5 / 6)
        self.assertEqual((summary["source"], summary["job"]), ("graph_umaps.npz", "result.json"))
        self.assertEqual(summary["harmony"]["correctionSeconds"], 3.0)
        self.assertNotIn("graphsUri", summary["harmony"])
        # The rasters join the records beside the summary; the work folder keeps only the modality weights.
        self.assertEqual(
            sorted(p.name for p in folder.iterdir()),
            ["raster_ADT.npz", "raster_RNA.npz", "raster_WNN.npz", "result.json", "summary.json"],
        )
        for graph in cite.GRAPHS:
            self.assertFalse((self.work / f"raster_{graph}.npz").exists())
            with np.load(folder / f"raster_{graph}.npz") as raster:
                self.assertEqual(raster["typeCells"].dtype, np.uint32)
                self.assertEqual(raster["typeCells"].shape, (3, cite.BINS, cite.BINS))
                self.assertEqual(raster["typeCells"].sum(axis=(1, 2)).tolist(), [1, 3, 2])
                self.assertEqual([str(t) for t in raster["types"]], ["", "T cells", "B cells"])
        with np.load(folder / "raster_RNA.npz") as raster:
            # The last cell is a T cell (label 1).
            self.assertEqual(raster["typeCells"][1, 8, 291], 1)
            self.assertEqual(int(raster["typeCells"][:, 8, 291].sum()), 1)
        with (self.work / "modality_weights.csv").open() as handle:
            rows = {row["label"]: row for row in csv.DictReader(handle)}
        # T cells hold weights 1, 0.5 and 0.75; B cells 0 and 1; the unlabelled cell 0.25.
        self.assertEqual(
            [rows["T cells"][k] for k in ("proteinLedCells", "evenCells", "rnaLedCells")],
            ["2", "1", "0"],
        )
        self.assertEqual(float(rows["T cells"]["meanProteinWeight"]), 0.75)
        self.assertEqual(float(rows["B cells"]["meanProteinWeight"]), 0.5)
        self.assertEqual(rows[""]["rnaLedCells"], "1")
        for name in ("result.json", "summary.json"):
            self.assertPublic(folder / name)
        self.assertEqual(json.loads((folder / "result.json").read_text())["region"], "us-central")

    def test_rebuilt_graphs_that_differ_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, r"saved graphs in \['wnn'\]"):
            cite.graph_umaps(self.write_job(identical=False), self.work)
        self.assertFalse(self.records.exists())

    def test_layouts_other_than_the_saved_ones_are_refused(self) -> None:
        job = self.write_job()
        job["result"]["umapSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from the file the job saved"):
            cite.graph_umaps(job, self.work)

    def test_weights_other_than_the_recorded_means_are_refused(self) -> None:
        job = self.write_job()
        job["result"]["modalityWeightMean"]["ADT"] += 1e-3
        with self.assertRaisesRegex(ValueError, "differ from the means the job recorded"):
            cite.graph_umaps(job, self.work)
        self.assertFalse(self.records.exists())

    def test_a_layout_with_a_cell_missing_writes_no_raster(self) -> None:
        job = self.write_job(rna=np.array([[0, 0], [1, 1], [2, 0], [0, 2], [3, 3], [np.nan, 0]]))
        with self.assertRaisesRegex(ValueError, "wrong cells or non-finite coordinates"):
            cite.graph_umaps(job, self.work)
        self.assertFalse(self.records.exists())
        self.assertEqual(sorted(p.name for p in self.work.iterdir()), ["graph_umaps.npz"])


class SeuratWeightsTests(CollectorTest):
    def write_job(self) -> dict:
        path = self.work / "seurat_weights.npz"
        cytearc = np.array([1.0, 0.5, 0.2, 0.8])
        seurat = np.array([0.9, 0.6, 0.1, 0.5])
        np.savez(
            path,
            cytearc_protein=cytearc,
            seurat_protein=seurat,
            label_codes=np.array([0, 0, 1, 1]),
            label_names=np.array(["T", "B"]),
        )
        return envelope(
            {
                "pairedUri": "s3://bucket/experiments/multimodal/seurat_weights.npz",
                "pairedSha256": sha256(path),
                "cells": 4,
                "proteinWeightPearson": float(np.corrcoef(cytearc, seurat)[0, 1]),
                "seurat": {
                    "Seurat": "5.5.1",
                    "SeuratObject": "5.3.0",
                    "R": "4.6.1",
                    "initialWallSeconds": 1.0,
                },
            }
        )

    def test_each_label_counts_its_leads_and_means_in_both_tools(self) -> None:
        cite.seurat_weights(self.write_job(), self.work)
        folder = self.records / "seurat_weights"
        with (folder / "weights_by_label.csv").open() as handle:
            rows = {row["label"]: row for row in csv.DictReader(handle)}
        self.assertEqual(
            [
                rows["T"][k]
                for k in ("cytearcProteinLedCells", "cytearcEvenCells", "seuratProteinLedCells")
            ],
            ["1", "1", "2"],
        )
        self.assertAlmostEqual(float(rows["B"]["seuratMeanProteinWeight"]), 0.3)
        self.assertEqual(rows["B"]["seuratEvenCells"], "1")
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(
            summary["proteinWeightShares"]["cytearc"], {"one": 0.25, "zero": 0.0, "half": 0.25}
        )
        self.assertAlmostEqual(summary["proteinWeightMean"]["seurat"], 0.525)
        self.assertEqual(summary["seurat"]["Seurat"], "5.5.1")
        self.assertPublic(folder / "result.json")
        self.assertPublic(folder / "summary.json")

    def test_weights_whose_correlation_differs_from_the_record_are_refused(self) -> None:
        job = self.write_job()
        job["result"]["proteinWeightPearson"] += 0.01
        with self.assertRaisesRegex(ValueError, "correlation differs"):
            cite.seurat_weights(job, self.work)


def t_cell_job(work: Path, stem: str, tables: dict[str, str], arrays: dict, **result) -> dict:
    """A T-cell job's record naming its arrays and tables, written into work as the job names them."""
    files = {}
    np.savez(work / f"{stem}.npz", **arrays)
    for name, text in tables.items():
        (work / f"{stem}_{name}.csv").write_text(text)
    for name in ["npz", *tables]:
        path = work / (f"{stem}.npz" if name == "npz" else f"{stem}_{name}.csv")
        files[name] = {
            "uri": f"s3://bucket/experiments/t_cells/{path.name}",
            "sha256": sha256(path),
        }
    return envelope({"files": files, **result})


class TCellJobsTests(CollectorTest):
    def reference_job(self, **changes) -> dict:
        return t_cell_job(
            self.work,
            "t_cell_reference",
            {
                "reference_proteins": "cluster,protein,medianClr\nTreg,CD25,2.0\n",
                "hao_proteins": "cluster,protein,medianClr\nTreg,CD25,1.5\n",
                "reference_authors": "cluster,authorLabel,cells\nTreg,CD4.TREG,2\n",
            },
            {
                "reference_umap": np.array([[0, 0], [1, 1], [2, 0], [0, 2], [3, 3]], dtype=float),
                "reference_subtypes": np.array(["CD4 naive", "CD4 naive", "Treg", "", "Treg"]),
            },
            **{
                "cells": 5,
                "subtypeCells": {"CD4 naive": 2, "Treg": 2},
                "unlabelledCells": 1,
                **changes,
            },
        )

    def test_reference_map_and_protein_tables(self) -> None:
        cite.t_cell_reference(self.reference_job(), self.work, "combat")
        folder = self.records / "t_cell_reference" / "combat"
        summary = json.loads((folder / "raster_summary.json").read_text())
        self.assertEqual(summary["cellsPerSubtype"], {"CD4 naive": 2, "Treg": 2, "unlabelled": 1})
        self.assertEqual(summary["source"], "t_cell_reference.npz")
        # The layout spans 0 to 3 on both axes; with a 3% margin on each side the grid spans 3.18 from -0.09, so the
        # Treg cell at (3, 3) falls in bin int(3.09 / 3.18 * 300) = 291 of both axes.
        self.assertAlmostEqual(summary["grid"]["span"], 3.18)
        self.assertEqual([round(v, 6) for v in summary["grid"]["origin"]], [-0.09, -0.09])
        self.assertFalse((self.work / "raster.npz").exists())
        with np.load(folder / "raster.npz") as raster:
            self.assertEqual(raster["typeCells"].dtype, np.uint32)
            self.assertEqual(raster["typeCells"].sum(axis=(1, 2)).tolist(), [2, 2, 1])
            self.assertEqual([str(t) for t in raster["types"]], ["CD4 naive", "Treg", "unlabelled"])
            self.assertEqual(raster["typeCells"][1, 291, 291], 1)
        # The raster and the two protein tables join the records; the authors table stays in the work folder.
        self.assertEqual(
            sorted(p.name for p in folder.iterdir()),
            [
                "raster.npz",
                "raster_summary.json",
                "result.json",
                "t_cell_reference_hao_proteins.csv",
                "t_cell_reference_reference_proteins.csv",
            ],
        )
        self.assertEqual(
            (folder / "t_cell_reference_hao_proteins.csv").read_text(),
            "cluster,protein,medianClr\nTreg,CD25,1.5\n",
        )
        self.assertPublic(folder / "result.json")

    def test_a_raster_of_other_subtypes_than_the_job_named_is_refused(self) -> None:
        job = self.reference_job(subtypeCells={"CD4 naive": 2, "Treg": 1})
        with self.assertRaisesRegex(ValueError, "do not hold the cells the job named"):
            cite.t_cell_reference(job, self.work, "combat")
        self.assertFalse(self.records.exists())
        self.assertFalse((self.work / "raster.npz").exists())

    def test_a_table_other_than_the_saved_one_is_refused(self) -> None:
        job = self.reference_job()
        (self.work / "t_cell_reference_hao_proteins.csv").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "t_cell_reference_hao_proteins.csv differs"):
            cite.t_cell_reference(job, self.work, "combat")

    def test_clusterings_go_under_their_protein_normalization(self) -> None:
        for normalization, folder in (
            ("cell", "t_cell_subtypes_cell_clr"),
            ("feature", "t_cell_subtypes"),
        ):
            with self.subTest(normalization=normalization):
                job = t_cell_job(
                    self.work,
                    "t_cell_subtypes",
                    {"authors": "graph,cluster\nwnn,1\n", "donors": "graph,donor\nwnn,d1\n"},
                    {"query_labels": np.array(["CD4 Naive"])},
                    adtNormalization=normalization,
                )
                cite.t_cell_subtypes(job, self.work, "combat")
                kept = self.records / folder / "combat"
                self.assertEqual(
                    sorted(p.name for p in kept.iterdir()),
                    ["result.json", "t_cell_subtypes.npz", "t_cell_subtypes_authors.csv"],
                )
                self.assertPublic(kept / "result.json")


if __name__ == "__main__":
    unittest.main()
