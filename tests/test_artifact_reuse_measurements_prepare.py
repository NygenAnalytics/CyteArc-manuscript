import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.reuse import composite_and_remote_summary
from manuscript.figures.artifact_reuse import prepare as artifact_reuse
from manuscript.figures.artifact_reuse_measurements import prepare

# A published analysis's stage times, and the fork's lookups and recomputed stages, chosen so that every total below is
# worked out by hand: the stored stages add up to 3,480 seconds and the recomputed ones to 2,860.
STORED = {
    "filterCells": 20.0,
    "markHvgs": 400.0,
    "runNormalization": 350.0,
    "runPca": 450.0,
    "buildEmbeddingInitialization": 140.0,
    "buildAnnIndex": 110.0,
    "queryNeighbors": 120.0,
    "buildConnectivityMap": 60.0,
    "runUmap": 1050.0,
    "runLeiden": 370.0,
    "findMarkers": 410.0,
}
LOOKUPS = {"normalization": 13.0, "pca": 14.0, "annIndex": 15.0}
COMPUTED = {
    "queryNeighbors": 160.0,
    "buildConnectivityMap": 50.0,
    "runUmap": 1600.0,
    "runLeiden": 540.0,
    "findMarkers": 510.0,
}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def published(cells: int = 10_000_000, scale: float = 1.0) -> dict:
    return {
        "funnel": {"status": "ok", "nRows": cells, "modalResources": {"modalCpuLimit": 16}},
        "stages": {stage: {"seconds": scale * seconds} for stage, seconds in STORED.items()},
    }


def fork(reference: str, **changes) -> dict:
    result = {
        "referenceTag": reference,
        "kOriginal": 11,
        "kNew": 15,
        "newArtifacts": 5,
        "reuse": {key: True for key in LOOKUPS},
        "fullRunStageSeconds": dict(STORED),
        "fullRunAnalysisSeconds": 3480.0,
        "stages": {
            **{f"fork.{stage}": {"seconds": s} for stage, s in COMPUTED.items()},
            **{f"reuse.{key}": {"seconds": s} for key, s in LOOKUPS.items()},
        },
        "forkSeconds": 2860.0,
        "container": {"cpu": 8},
        **changes,
    }
    return {"result": result}


class TemporaryRecords(unittest.TestCase):
    """The fork record and the published runs in a temporary folder, in place of the experiment records."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, value in {
            "ROOT": self.root,
            "SCALING_RUNS": self.root / "rna_scaling_runs.json",
            "FORK_RUNS": self.root / "artifact_reuse_runs.json",
        }.items():
            patch = mock.patch.object(prepare, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.tag = prepare.FORK_REFERENCE_TAG

    def records(self, runs: dict, forks: dict) -> None:
        write(prepare.SCALING_RUNS, runs)
        write(prepare.FORK_RUNS, {"results": forks})


def doubled_fork(reference: str) -> dict:
    """A valid fork on another store of the series, every time twice the named fork's."""
    return fork(
        reference,
        fullRunStageSeconds={stage: 2 * s for stage, s in STORED.items()},
        fullRunAnalysisSeconds=6960.0,
        stages={
            **{f"fork.{stage}": {"seconds": 2 * s} for stage, s in COMPUTED.items()},
            **{f"reuse.{key}": {"seconds": 2 * s} for key, s in LOOKUPS.items()},
        },
        forkSeconds=5720.0,
    )


class NeighborCountForkTest(TemporaryRecords):
    def test_the_fork_on_the_named_store_is_drawn_and_forks_on_other_stores_are_left_out(self):
        # Before the named fork, a valid fork on another ten-million-cell store, whose times are twice as long, and a
        # fork on a one-million-cell store, which names a run outside the scaling series and must not be looked up.
        other = "embryo_rna_scaling_10000000_cells_replicate_9"
        self.records(
            {other: published(scale=2.0), self.tag: published()},
            {
                "rerun_one_million": fork("embryo_rna_memory_8_gib_8_workers_replicate_1"),
                "rerun_other_ten_million": doubled_fork(other),
                "rerun_ten_million": fork(self.tag),
            },
        )
        data = prepare.neighbor_count_fork_stages()
        self.assertEqual(
            [(s["stage"], s["fork"], s["forkSeconds"]) for s in data["stages"]],
            [
                ("filterCells", None, None),
                ("markHvgs", None, None),
                ("runNormalization", "lookup", 13.0),
                ("runPca", "lookup", 14.0),
                ("buildAnnIndex", "lookup", 15.0),
                ("queryNeighbors", "computed", 160.0),
                ("buildConnectivityMap", "computed", 50.0),
                ("buildEmbeddingInitialization", None, None),
                ("runUmap", "computed", 1600.0),
                ("runLeiden", "computed", 540.0),
                ("findMarkers", "computed", 510.0),
            ],
        )
        self.assertEqual({s["stage"]: s["storedSeconds"] for s in data["stages"]}, STORED)
        self.assertEqual(data["totals"], {"storedSeconds": 3480.0, "forkSeconds": 2860.0})
        self.assertEqual(data["cpus"], {"stored": 16, "fork": 8})
        self.assertEqual(data["cells"], 10_000_000)
        self.assertEqual(data["source"], ["artifact_reuse_runs.json", "rna_scaling_runs.json"])

    def test_a_record_without_a_fork_on_the_named_store_is_refused(self):
        self.records({self.tag: published()}, {"rerun_other": fork("embryo_other_store")})
        with self.assertRaisesRegex(ValueError, "must hold one fork on .*; it holds \\[\\]"):
            prepare.neighbor_count_fork_stages()

    def test_two_forks_on_the_named_store_are_refused(self):
        self.records(
            {self.tag: published()}, {"rerun_first": fork(self.tag), "rerun_second": fork(self.tag)}
        )
        with self.assertRaisesRegex(ValueError, "rerun_first.*rerun_second"):
            prepare.neighbor_count_fork_stages()

    def test_a_named_store_that_is_not_ten_million_cells_is_refused(self):
        self.records({self.tag: published(cells=1_000_000)}, {"rerun": fork(self.tag)})
        with self.assertRaisesRegex(ValueError, "holds 1,000,000 cells"):
            prepare.neighbor_count_fork_stages()

    def test_stored_stage_times_that_differ_from_the_published_run_are_refused(self):
        stages = {**STORED, "runUmap": 1051.0}
        self.records(
            {self.tag: published()},
            {"rerun": fork(self.tag, fullRunStageSeconds=stages, fullRunAnalysisSeconds=3481.0)},
        )
        with self.assertRaisesRegex(ValueError, "published runUmap"):
            prepare.neighbor_count_fork_stages()

    def test_a_fork_total_that_is_not_its_recomputed_stages_is_refused(self):
        self.records({self.tag: published()}, {"rerun": fork(self.tag, forkSeconds=2861.0)})
        with self.assertRaisesRegex(ValueError, "rerun's fork"):
            prepare.neighbor_count_fork_stages()

    def test_a_fork_step_the_panel_does_not_draw_is_refused(self):
        steps = fork(self.tag)["result"]["stages"] | {"fork.runPca": {"seconds": 1.0}}
        self.records({self.tag: published()}, {"rerun": fork(self.tag, stages=steps)})
        with self.assertRaisesRegex(ValueError, "steps the figure does not draw"):
            prepare.neighbor_count_fork_stages()

    def test_a_fork_that_recomputed_a_stored_artifact_is_refused(self):
        self.records(
            {self.tag: published()},
            {"rerun": fork(self.tag, reuse={**{k: True for k in LOOKUPS}, "pca": False})},
        )
        with self.assertRaisesRegex(ValueError, "did not reuse every stored upstream artifact"):
            prepare.neighbor_count_fork_stages()


def step(before: int, after: int) -> dict:
    return {"userStoreBytesBefore": before, "userStoreBytesAfter": after}


class EndothelialStoreSizeTest(unittest.TestCase):
    """The user store after the reanalysis is its size after the last step before the first request."""

    def mount(self, order: list[str], steps: dict) -> tuple[dict, dict]:
        record = {
            "steps": steps,
            "stepOrder": order,
            "reuse": {"stepOrder": ["reuse.list", "reuse.leiden.r2"]},
            "source": {"unchanged": True, "before": {"bytes": 1000}, "after": {"bytes": 1000}},
            # Counts in both layouts, their summaries, the analysis results and the copied tables: 1,000 bytes.
            "referenced": {
                "counts": {"sourceBytes": 500},
                "countsT": {"sourceBytes": 300},
                "countSummaries": {"sourceBytes": 20},
            },
            "notCopied": {"publishedResultsBytes": 100},
            "copied": {"cellData": {"sourceBytes": 70}, "featureData": {"sourceBytes": 10}},
            "userStore": {"afterMount": {"bytes": 100}, "final": {"bytes": 200}},
            "totals": {"sourceReadBytes": 5000},
        }
        run = {"funnel": {"datasetBytes": 900}, "stages": {"createStore": {"datasetBytes": 900}}}
        return record, run

    def size(self, order: list[str], steps: dict) -> dict:
        with mock.patch.object(prepare, "_mounted", return_value=self.mount(order, steps)):
            return prepare.endothelial_store_size()

    # The reanalysis ends with a Leiden clustering, after its UMAP, so neither the last step overall nor a step named
    # by the code would give the store after the reanalysis, 170 bytes.
    STEPS = {
        "mount": step(0, 100),
        "umap.k11": step(100, 120),
        "annotate": step(120, 150),
        "leiden.k21_r1": step(150, 170),
        "reuse.list": step(170, 170),
        "reuse.leiden.r2": step(170, 200),
    }

    def test_the_reanalysis_ends_at_the_step_before_the_first_request(self):
        data = self.size(list(self.STEPS), self.STEPS)
        self.assertEqual(
            [(s["name"], s["kind"], s["bytes"]) for s in data["sizes"]],
            [
                ("source H5AD", "source", 900),
                ("remote store", "source", 1000),
                ("its counts, 2 layouts", "source", 800),
                ("its analysis results", "source", 100),
                ("user store, mounted", "user", 100),
                ("after the reanalysis", "user", 170),
                ("after the requests", "user", 200),
            ],
        )
        self.assertEqual(data["readBytes"], 5000)

    def test_a_reanalysis_store_that_the_first_request_does_not_start_from_is_refused(self):
        steps = {**self.STEPS, "leiden.k21_r1": step(150, 171)}
        with self.assertRaisesRegex(ValueError, "the reanalysis's store"):
            self.size(list(steps), steps)

    def test_a_record_that_starts_with_a_request_is_refused(self):
        order = ["reuse.list", "mount", "umap.k11", "annotate", "leiden.k21_r1", "reuse.leiden.r2"]
        with self.assertRaisesRegex(ValueError, "no reanalysis before its first request"):
            self.size(order, self.STEPS)


class PublishedRunTimesTest(unittest.TestCase):
    def test_the_figures_draw_the_published_run_times_that_the_summary_gives(self):
        # Paragraph 52 quotes the run that wrote the mounted store from the reuse summary: its clustering chain through
        # Leiden, which Figure 3a draws as the published run, and its analysis through the marker search and whole
        # run, which the mounted-store panel draws. Figure 3a adds its time from the artifacts of its own chain.
        summary = composite_and_remote_summary
        mount = summary.load_json(summary.MOUNTED_STORE)
        result = summary.mounted_store(
            mount, summary.load_json(summary.SCALING_RUNS)[mount["referenceTag"]]
        )
        published = [
            leaf
            for branch in artifact_reuse.provenance_runs()["branches"]
            for leaf in branch["leaves"]
            if leaf["published"]
        ]
        self.assertEqual(len(published), 1)
        self.assertAlmostEqual(
            published[0]["runSeconds"] / 60, result["publishedClusteringMinutes"], places=9
        )
        requests = prepare.mounted_store_requests()
        self.assertAlmostEqual(
            requests["publishedAnalysisSeconds"] / 60, result["publishedAnalysisMinutes"], places=9
        )
        self.assertAlmostEqual(
            requests["publishedRunSeconds"] / 60, result["publishedRunMinutes"], places=9
        )


if __name__ == "__main__":
    unittest.main()
