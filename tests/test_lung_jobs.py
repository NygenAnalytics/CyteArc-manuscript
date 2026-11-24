"""The lung jobs on CyteArc 1.3.0: the RNA recipe call, the read meter of the reference update, the provenance and
image the job records state, and the check of the sensitivity analysis's primary configuration against the screen
run it reuses."""

import copy
import hashlib
import inspect
import os
import re
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from experiments.lung import lung_pipeline, lung_robustness, reference_revision
from experiments.runtime.common import ROOT
from experiments.runtime.image_records import image_record

# The fields of the pre-release screen's summary (lung_analysis on its records) that screen_findings reads, as
# lung_analysis wrote them; every other field is left out.
RC19_SCREEN_SUMMARY = {
    "calibrations": {
        "allControls": {
            "unexplainedCells": 4798,
            "regionTests": {
                "Endothelial": {
                    "wilcoxon": {"qValue": 0.19217919907912603},
                    "spearman": {"rho": 0.07373142719429406},
                    "Less_fibroticVersusControl": {"qValue": 0.37243033988366214},
                    "More_fibroticVersusControl": {"qValue": 0.8344955175660529},
                },
                "Epithelial": {
                    "wilcoxon": {"qValue": 0.8328664161727948},
                    "spearman": {"rho": 0.17352057138318872},
                    "Less_fibroticVersusControl": {"qValue": 0.21323705977795235},
                    "More_fibroticVersusControl": {"qValue": 0.2971001843506629},
                },
                "Immune": {
                    "wilcoxon": {"qValue": 0.4982518907477189},
                    "spearman": {"rho": 0.45146379044684126},
                    "Less_fibroticVersusControl": {"qValue": 0.8851643565871021},
                    "More_fibroticVersusControl": {"qValue": 0.9665973586655598},
                },
                "Stroma": {
                    "wilcoxon": {"qValue": 0.8328664161727948},
                    "spearman": {"rho": 0.14060202402735905},
                    "Less_fibroticVersusControl": {"qValue": 0.8851643565871021},
                    "More_fibroticVersusControl": {"qValue": 0.9665973586655598},
                },
            },
        },
        "withoutVUHD117": {
            "unexplainedCells": 6079,
            "regionTests": {
                "Endothelial": {
                    "wilcoxon": {"qValue": 0.23013142754149687},
                    "spearman": {"rho": 0.11339781948443364},
                    "Less_fibroticVersusControl": {"qValue": 0.3637675983191839},
                    "More_fibroticVersusControl": {"qValue": 0.6950509086752135},
                },
                "Epithelial": {
                    "wilcoxon": {"qValue": 0.7469254073997339},
                    "spearman": {"rho": 0.35380630189542756},
                    "Less_fibroticVersusControl": {"qValue": 0.021787186338288387},
                    "More_fibroticVersusControl": {"qValue": 0.013189277186110305},
                },
                "Immune": {
                    "wilcoxon": {"qValue": 0.4555744730190394},
                    "spearman": {"rho": 0.4453004622496148},
                    "Less_fibroticVersusControl": {"qValue": 0.7755240020726523},
                    "More_fibroticVersusControl": {"qValue": 0.7373147305754079},
                },
                "Stroma": {
                    "wilcoxon": {"qValue": 0.8838460571465129},
                    "spearman": {"rho": 0.15176091482318121},
                    "Less_fibroticVersusControl": {"qValue": 0.8908171258685105},
                    "More_fibroticVersusControl": {"qValue": 0.7373147305754079},
                },
            },
        },
    },
    "basaloid": {
        "unexplainedClusters": {
            "clusters": ["5", "6", "11"],
            "byRegion": {
                "Control": {"allCells": 56258, "unexplainedCells": 651},
                "Less_fibrotic": {"allCells": 102716, "unexplainedCells": 1517},
                "More_fibrotic": {"allCells": 161061, "unexplainedCells": 2613},
                "Unknown": {"allCells": 2384, "unexplainedCells": 17},
            },
            "paired": {
                "inLess": 20,
                "lungsWithCells": 26,
                "meanRateRatio": 1.1744919581528526,
                "meanRateRatioCi95": {"low": 0.4409444605489842, "high": 4.227593913928585},
            },
        },
        "authorLabelled": {"unexplainedShare": 0.41392136025504783},
    },
}
# The pre-release sensitivity analysis checked its primary configuration against these values, which its own run
# of lung_analysis had given, and found every one exactly.
RC19_FINDINGS = {
    "unexplainedCells": 4798,
    "basaloidClusters": ["5", "6", "11"],
    "inLess": 20,
    "lungsWithCells": 26,
    "meanRateRatio": 1.1744919581528526,
    "meanRateRatioLow": 0.4409444605489842,
    "meanRateRatioHigh": 4.227593913928585,
    "labelledShare": 0.41392136025504783,
    "allShare": 0.014881256997881639,
    "epithelialLessQ": 0.21323705977795235,
    "epithelialMoreQ": 0.2971001843506629,
    "epithelialLessQWithoutVUHD117": 0.021787186338288387,
    "epithelialMoreQWithoutVUHD117": 0.013189277186110305,
    "immuneRho": 0.45146379044684126,
    "minPairedQ": 0.19217919907912603,
}
LESS, MORE = lung_robustness.LESS, lung_robustness.MORE


def primary_result(findings: dict) -> dict:
    """An outcome of _outcomes for the primary configuration that holds ``findings``."""
    return {
        "unexplainedCells": findings["unexplainedCells"],
        "outcomes": {
            "O1": {"clusters": [{"cluster": c} for c in findings["basaloidClusters"]]},
            "O2": {"inLess": findings["inLess"], "lungsWithCells": findings["lungsWithCells"]},
            "O3": {
                "ratio": findings["meanRateRatio"],
                "low": findings["meanRateRatioLow"],
                "high": findings["meanRateRatioHigh"],
            },
            "O4": {"labelledShare": findings["labelledShare"], "allShare": findings["allShare"]},
            "O5": {
                "own": {
                    LESS: {"q": findings["epithelialLessQ"]},
                    MORE: {"q": findings["epithelialMoreQ"]},
                },
                "withoutVUHD117": {
                    LESS: {"q": findings["epithelialLessQWithoutVUHD117"]},
                    MORE: {"q": findings["epithelialMoreQWithoutVUHD117"]},
                },
            },
            "O6": {"rho": findings["immuneRho"]},
            "O7": {"minQ": findings["minPairedQ"]},
        },
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScreenFindingsTest(unittest.TestCase):
    def test_pre_release_summary_gives_the_values_its_check_used(self):
        self.assertEqual(lung_robustness.screen_findings(RC19_SCREEN_SUMMARY), RC19_FINDINGS)

    def test_the_share_counts_every_region_of_the_census(self):
        summary = copy.deepcopy(RC19_SCREEN_SUMMARY)
        unknown = summary["basaloid"]["unexplainedClusters"]["byRegion"]["Unknown"]
        unknown.update(allCells=2384 + 1000, unexplainedCells=17 + 1000)
        # 5,798 unexplained cells among 323,419 with a lineage.
        self.assertEqual(lung_robustness.screen_findings(summary)["allShare"], 5798 / 323419)

    def test_missing_tests_give_none_as_the_check_finds_them(self):
        summary = copy.deepcopy(RC19_SCREEN_SUMMARY)
        del summary["calibrations"]["allControls"]["regionTests"]["Immune"]
        for test in summary["calibrations"]["allControls"]["regionTests"].values():
            test["wilcoxon"]["qValue"] = None
        findings = lung_robustness.screen_findings(summary)
        self.assertIsNone(findings["immuneRho"])
        self.assertIsNone(findings["minPairedQ"])

    def test_a_summary_without_clusters_is_refused(self):
        summary = copy.deepcopy(RC19_SCREEN_SUMMARY)
        summary["basaloid"] = {}
        with self.assertRaisesRegex(ValueError, "after the screen's characterise job"):
            lung_robustness.screen_findings(summary)


class PrimaryCheckTest(unittest.TestCase):
    def test_the_same_findings_are_exact(self):
        check = lung_robustness._primary_check(primary_result(RC19_FINDINGS), RC19_FINDINGS)
        self.assertTrue(check["allExact"])
        self.assertTrue(check["allWithin1e-9"])
        self.assertEqual(check["published"], RC19_FINDINGS)

    def test_a_changed_value_is_reported(self):
        found = {**RC19_FINDINGS, "inLess": 19, "allShare": RC19_FINDINGS["allShare"] * (1 + 1e-12)}
        check = lung_robustness._primary_check(primary_result(found), RC19_FINDINGS)
        changed = sorted(k for k, exact in check["exact"].items() if not exact)
        self.assertEqual(changed, ["allShare", "inLess"])
        self.assertFalse(check["allExact"])
        # A relative difference of 1e-12 is within 1e-9; one cell fewer is not.
        self.assertFalse(check["allWithin1e-9"])
        found["inLess"] = 20
        self.assertTrue(
            lung_robustness._primary_check(primary_result(found), RC19_FINDINGS)["allWithin1e-9"]
        )

    def test_the_screen_findings_must_be_given_in_full(self):
        result = primary_result(RC19_FINDINGS)
        partial = {k: v for k, v in RC19_FINDINGS.items() if k != "minPairedQ"}
        for published in (None, {}, partial, {**RC19_FINDINGS, "extra": 1}):
            with (
                self.subTest(published=published),
                self.assertRaisesRegex(ValueError, "screenFindings"),
            ):
                lung_robustness._primary_check(result, published)


class SensitivityPlanTest(unittest.TestCase):
    def test_the_analysis_and_check_carry_the_screen_findings(self):
        plan = lung_robustness.build_plan(
            "s3://bucket/experiments",
            "s3://bucket/experiments/datasets",
            None,
            screen_findings=RC19_FINDINGS,
        )
        carrying = sorted(name for name, payload in plan.items() if "screenFindings" in payload)
        self.assertEqual(carrying, ["analysis", "check"])
        self.assertEqual(plan["check"]["screenFindings"], RC19_FINDINGS)
        without = lung_robustness.build_plan("s3://b/e", "s3://b/e/datasets", None)
        self.assertFalse(any("screenFindings" in payload for payload in without.values()))


class ReadMeterTest(unittest.TestCase):
    def test_reads_of_a_metered_store_are_counted_and_writes_refused(self):
        import zarr
        from cytearc.storage import stores

        with tempfile.TemporaryDirectory() as folder:
            core, other = Path(folder) / "core.zarr", Path(folder) / "revised.zarr"
            for path in (core, other):
                group = zarr.open_group(str(path), mode="w")
                group.create_array("values", data=np.arange(1000, dtype=np.int64), chunks=(1000,))
            stored = sum(p.stat().st_size for p in core.rglob("*") if p.is_file())
            original = stores.make_store
            meter = reference_revision.ReadMeter({"core": str(core), "revised": str(other)})
            with meter:
                root = stores.load_zarr(str(core), mode="r")
                self.assertEqual(root["values"][:].sum(), 499500)
                with self.assertRaisesRegex(PermissionError, "core"):
                    stores.make_store(str(core), read_only=False)
                writable = stores.make_store(str(other), read_only=False)
                unmetered = stores.make_store(str(Path(folder) / "elsewhere.zarr"), read_only=True)
            self.assertIs(stores.make_store, original)
            totals = meter.totals()
            # Opening the group and reading the array reads every object of the store once: the two metadata
            # documents and the one chunk.
            self.assertEqual(totals["core"]["readBytes"], stored)
            self.assertGreaterEqual(totals["core"]["requests"], 3)
            self.assertEqual(totals["revised"], {"readBytes": 0, "requests": 0})
            self.assertEqual(meter.opened["core"], {"readOnly": 1, "writable": 0})
            self.assertEqual(meter.opened["revised"], {"readOnly": 0, "writable": 1})
            self.assertIsInstance(writable, str)
            self.assertIsInstance(unmetered, str)


class ProvenanceTest(unittest.TestCase):
    PAYLOAD = {"runTag": "lung-reference-update", "containerCpu": 16.0, "workers": 16}

    def test_the_update_records_the_files_it_ran_with(self):
        record = reference_revision._provenance(self.PAYLOAD)
        runtime = ROOT / "experiments" / "runtime"
        self.assertEqual(record["recordingStoreSha256"], sha256(runtime / "recording_store.py"))
        self.assertEqual(
            record["referenceScreenSourceSha256"],
            sha256(ROOT / "experiments" / "lung" / "lung_analysis.py"),
        )
        self.assertEqual(record["scriptSha256"], sha256(Path(reference_revision.__file__)))

    def test_jobs_state_the_cytearc_image_they_ran_in(self):
        source = (ROOT / "experiments" / "images.py").read_text()
        python = re.search(
            r"python_cytearc_base = \(\s*modal\.Image\.debian_slim\(python_version=\"([0-9.]+)\"\)",
            source,
        ).group(1)
        image = re.search(r"python_cytearc_base = \((.*?)\n\)", source, re.S).group(1)
        self.assertIn('RUNTIME / "python-cytearc.txt"', image)
        lock = ROOT / "experiments" / "runtime" / "python-cytearc.txt"
        for image in (
            lung_pipeline._provenance(self.PAYLOAD)["image"],
            reference_revision._provenance(self.PAYLOAD)["image"],
            lung_robustness._provenance(self.PAYLOAD)["image"],
        ):
            self.assertEqual(image, image_record("cytearc_tool_job"))
            self.assertEqual(image["base"], f"debian_slim python {python}")
            self.assertEqual(image["requirements"], str(lock.relative_to(ROOT)))
            self.assertEqual(image["requirementsSha256"], sha256(lock))


class RecipeCallTest(unittest.TestCase):
    """run_pipeline calls CyteArc's DataStore.pipeline.run with arguments and stage settings that its recipe
    accepts: the arguments of its signature, and settings of the stages that run."""

    def calls(self, **leiden) -> list[dict]:
        calls = []
        ds = types.SimpleNamespace(
            pipeline=types.SimpleNamespace(run=lambda **kwargs: calls.append(kwargs))
        )
        lung_pipeline.run_pipeline(
            ds,
            label="characterise",
            cells="chosen",
            hvg={"top_n": 2000},
            pca={"dims": 30},
            neighbors={"k": 15},
            filtering=False,
            umap=False,
            cell_cycle=False,
            paris=False,
            doublets=False,
            markers=False,
            **leiden,
        )
        return calls

    def test_arguments_and_settings_fit_the_recipe(self):
        from cytearc.datastore._pipeline_recipe import _STAGE_PARAMETERS
        from cytearc.datastore.pipeline_accessor import PipelineAccessor

        signature = inspect.signature(PipelineAccessor.run)
        partition = {"partitions": [1.0], "selected": 1.0, "random_seed": 4444}
        for leiden in (False, partition):
            (call,) = self.calls(leiden=leiden)
            signature.bind(None, **call)
            self.assertEqual((call["label"], call["cell_key"]), ("characterise", "chosen"))
            for stage, settings in call["params"].items():
                self.assertLessEqual(set(settings), _STAGE_PARAMETERS[stage], stage)
            self.assertEqual(call["leiden"], bool(leiden))
            self.assertEqual(call["params"].get("leiden"), leiden or None)

    def test_the_pre_release_call_does_not_fit(self):
        from cytearc.datastore.pipeline_accessor import PipelineAccessor

        with self.assertRaises(TypeError):
            inspect.signature(PipelineAccessor.run).bind(
                None, label="reference", cells="I", hvg={"top_n": 2000}
            )


@unittest.skipUnless(
    os.environ.get("CYTEARC_SLOW_TESTS"),
    "set CYTEARC_SLOW_TESTS=1 to run the recipe on a small store (about 30 s, mostly compilation)",
)
class RecipeTest(unittest.TestCase):
    """run_pipeline with CyteArc 1.3.0, on a store of 300 cells drawn from three expression programmes."""

    SWITCHES = dict(
        filtering=False, umap=False, cell_cycle=False, paris=False, doublets=False, markers=False
    )

    @classmethod
    def setUpClass(cls):
        import anndata as ad
        import cytearc
        import pandas as pd
        import scipy.sparse as sp

        cls.folder = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(3)
        rates = rng.gamma(0.8, 1.0, (3, 400))
        for group in range(3):
            rates[group, group * 40 : group * 40 + 40] *= 20
        cls.groups = np.repeat(np.arange(3), 100)
        counts = rng.poisson(rates[cls.groups] / rates[cls.groups].sum(1, keepdims=True) * 1500)
        cls.ids = np.array([f"cell{i}" for i in range(300)])
        adata = ad.AnnData(
            X=sp.csr_matrix(counts.astype(np.float32)),
            obs=pd.DataFrame(index=cls.ids),
            var=pd.DataFrame(
                {"feature_name": [f"G{i}" for i in range(400)]},
                index=[f"ENSG{i:011d}" for i in range(400)],
            ),
        )
        path = Path(cls.folder.name) / "cells.h5ad"
        adata.write_h5ad(path)
        cytearc.configure_output(level="ERROR", progress=False)
        reader = cytearc.H5adReader(str(path), feature_name_key="feature_name")
        try:
            cytearc.H5adToZarr(reader, str(path.with_suffix(".zarr")), assay_name="RNA").dump()
        finally:
            reader.close()
        cls.ds = cytearc.DataStore(
            str(path.with_suffix(".zarr")),
            default_assay="RNA",
            nthreads=1,
            mem_budget=512 * 1024**2,
            zarr_mode="r+",
            assay_types={"RNA": "RNA"},
            min_features_per_cell=10,
        )

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_pipeline(self, label: str, **settings):
        return lung_pipeline.run_pipeline(
            self.ds,
            label=label,
            hvg={"top_n": 120},
            pca={"dims": 8},
            neighbors={"k": 12},
            **{**self.SWITCHES, **settings},
        )

    def test_the_stages_use_the_settings_given(self):
        ds, run = self.ds, self.run_pipeline("reference", leiden=False)
        self.assertEqual(run.status, "completed")
        selected = np.asarray(ds.artifacts.load(run["highly_variable_features"])["values"][:])
        self.assertEqual(int(selected.sum()), 120)
        self.assertEqual(ds.artifacts.load(run["pca"])["data"].shape, (300, 8))
        self.assertEqual(ds.artifacts.load(run["neighbors"])["indices"].shape, (300, 12))
        self.assertNotIn("clusters", run)

    def test_leiden_runs_the_given_partition_with_the_given_seed(self):
        run = self.run_pipeline(
            "clustered", leiden={"partitions": [0.5], "selected": 0.5, "random_seed": 4444}
        )
        clusters = np.asarray(run.cells.fetch("clusters")).astype(str)
        # Each expression programme is one cluster.
        self.assertEqual(len(set(zip(self.groups.tolist(), clusters.tolist(), strict=True))), 3)
        self.assertEqual(len(set(clusters)), 3)
        config = run.report()["run"]["config"]
        self.assertEqual(config["leiden"], {"partitions": [0.5], "selected": 0.5})
        self.assertEqual(config["params"]["leiden"]["random_seed"], 4444)

    def test_the_cell_column_chooses_the_cells(self):
        chosen = np.zeros(300, dtype=bool)
        chosen[::2] = True
        self.ds.cells.insert("chosen", chosen, overwrite=True)
        run = self.run_pipeline("chosen", cells="chosen", leiden=False)
        self.assertEqual(
            np.asarray(run.cells.fetch("ids")).astype(str).tolist(), self.ids[chosen].tolist()
        )


if __name__ == "__main__":
    unittest.main()
