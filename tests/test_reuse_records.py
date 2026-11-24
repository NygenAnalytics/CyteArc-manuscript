import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.reuse import collect_runs, composite_and_remote_summary
from experiments.reuse.collect_runs import (
    blood_cohort_analysis,
    blood_cohort_mixing,
    blood_on_disk_concatenation,
    blood_scanpy_dask,
    composite_and_remote,
    job_run,
    tool_report,
)
from experiments.runtime.common import public_run

RECORDS = Path(collect_runs.__file__).resolve().parent / "records"
# Strings a public record must not hold: a Modal call ID, a bucket URI, a job work directory,
# the runner's start time and a donor table.
PRIVATE = ("fc-01ABCDEFGHIJKLMNOPQRSTUV", "s3://", "/tmp/", "startedAt", "callId", "donorGroup")


def resources(peak: int) -> dict:
    return {
        "operationBaselineBytes": 100,
        "operationPeakBytes": peak,
        "operationIncrementalPeakBytes": peak - 100,
        "operationPeakSource": "cgroupMemoryCurrent",
        "processTreeRssPeakBytes": peak * 2,
        "memoryMaxBytes": 1000,
    }


def remote(seconds: float, read: int) -> dict:
    return {
        "elapsedSeconds": seconds,
        "io": {
            "readBytes": read,
            "readMB": read / 1e6,
            "requests": 3,
            "raw": {"gets": 3},
            "httpResponseStatusCounts": {"206": 3},
        },
        "resources": resources(500),
    }


def local(status: str, seconds: float) -> dict:
    measured = {
        "status": status,
        "elapsedSeconds": seconds,
        "processIoBytes": {"rchar": 7, "wchar": 9, "read_bytes": 0, "write_bytes": 0},
        "resources": resources(400),
    }
    if status == "failed":
        measured.update(errorType="TypeError", error="cannot write obs")
    return measured


def cohort_run() -> dict:
    return {
        "module": "reuse.blood_cohort_analysis",
        "function": "run",
        "callId": "fc-01ABCDEFGHIJKLMNOPQRSTUV",
        "region": "us-east",
        "startedAt": 1.0,
        "status": "completed",
        "elapsedSeconds": 99.0,
        "result": {
            "settings": {
                "sources": {"a": "store_a"},
                "memBudget": "24G",
                "nthreads": 16,
                "hvgCount": 2000,
                "pcaDims": 30,
                "neighborsK": 30,
                "workDirectory": "/tmp/blood_cohort",
                "recordUri": "s3://bucket/run.json",
                "outputPrefix": "s3://bucket/run",
            },
            "timingBoundary": "creation, then the analysis",
            "sourceCellCounts": {"a": 10},
            "cells": 10,
            "genesShared": 5,
            "localStoreAfterCreateBytes": 1234,
            "creation": remote(1.5, 11),
            "unintegrated": remote(2.5, 22),
            "integrated": remote(3.5, 0),
            "annotation": {"cellTypeHarmonization": {"B": "B cells"}},
            "donors": [{"donorGroup": "a__d1", "cells": 10}],
            "pseudobulk": {"groups": 4, "features": 5, "dtype": "uint32", "bytes": 77},
            "pseudobulkMeasurement": remote(4.5, 44),
            "cellsUri": "s3://bucket/run/cells.npz",
        },
    }


def on_disk_run(first_attempt: str) -> dict:
    result = {
        "anndataVersion": "0.13.4",
        "h5pyVersion": "3.16.0",
        "function": "anndata.experimental.concat_on_disk",
        "join": "inner",
        "merge": "first",
        "maxLoadedElems": 100,
        "inputs": [{"name": "a", "cells": 10, "bytes": 50}],
        "timingBoundary": "concatenation only",
        "staging": {"seconds": 1.0, "bytes": 50, "source": "r2"},
        "asExportedAttempt": local(first_attempt, 1.0),
        "outputBytes": 60,
        "outputAllocatedBytes": 64,
        "validation": {"cells": 10, "exactCountValuesAndStructure": True},
        "validationMeasurement": local("completed", 3.0),
        "analysisInput": {"featureName": "copied"},
        "upload": {"seconds": 1.0, "bytes": 60, "uri": "s3://bucket/concatenated.h5ad"},
        "jobSeconds": 9.0,
    }
    if first_attempt == "failed":
        result["inputPreparation"] = {
            "addedEmptyMappings": {"a": ["layers"]},
            "obsColumnsChecked": 2,
            "obsColumnsStoredAsStringCategories": ["age"],
            "obsKindsByInput": {"age": {"a": "string"}},
        }
        result["inputPreparationMeasurement"] = local("completed", 2.0)
        result["concatenation"] = local("completed", 4.0)
    else:
        result["concatenation"] = result["asExportedAttempt"]
    return {
        "module": "reuse.composite_and_remote",
        "function": "run",
        "callId": "fc-01ABCDEFGHIJKLMNOPQRSTUV",
        "region": "us-central",
        "startedAt": 1.0,
        "status": "completed",
        "result": {"onDiskConcatenation": result, "sourceStorageOptions": {"block_size": 1}},
    }


def scanpy_run() -> dict:
    return {
        "tool": "scanpy_dask_covariance_csc",
        "dataset": "blood",
        "datasetUri": "s3://bucket/concatenated.h5ad",
        "replicate": 2,
        "container": {
            "cpu": 16,
            "memoryMb": 1024,
            "workers": 16,
            "preemptible": False,
            "region": "uk",
        },
        "image": {"lock": "experiments/runtime/python-tools.txt"},
        "restarts": 0,
        "interruptions": [],
        "callId": "fc-01ABCDEFGHIJKLMNOPQRSTUV",
        "startedAt": 1.0,
        "finishedAt": 2.0,
        "status": "ok",
        "steps": {"runLeiden": {"start": 1.0, "end": 2.0, "seconds": 1.0}},
        "memorySeries": [[1.0, 2, 3, 0]],
        "log": "warnings",
        "versions": {"scanpy": "1.12.4"},
        "settings": {"variableGenes": 2000},
        "input": {"format": "h5ad", "location": "localDisk"},
        "mode": "dask_covariance_csc",
        "lazyReader": "anndata.experimental.read_elem_lazy",
        "chunkRows": 50000,
        "nCells": 10,
        "nGenes": 5,
        "nnz": 20,
        "clusters": 3,
        "download": {"seconds": 4.0, "bytes": 60, "inputMode": "file"},
        "workflowSeconds": 5.0,
        "peakCgroupBytes": 600,
        "peakRssBytes": 700,
        "memorySampleSeconds": 0.5,
    }


def composite_run() -> dict:
    return {
        "module": "reuse.composite_and_remote",
        "function": "run",
        "callId": "fc-01ABCDEFGHIJKLMNOPQRSTUV",
        "region": "us-east",
        "startedAt": 1.0,
        "status": "completed",
        "result": {
            "composite": {
                "settings": {
                    "mode": "composite",
                    "memBudget": "24G",
                    "replicate": 2,
                    "workDirectory": "/tmp/blood_composite",
                    "recordUri": "s3://bucket/run.json",
                    "r2EndpointUrl": "https://example.r2.cloudflarestorage.com",
                    "materializedInputsPrefix": "s3://bucket/inputs",
                    "provenance": {"cytearcVersion": "1.3.0"},
                },
                "materializedInputs": [
                    {"name": "a", "uri": "s3://bucket/inputs/a.h5ad", "cells": 10}
                ],
            },
            "sourceStorageOptions": {"token": False, "block_size": 1},
        },
        "elapsedSeconds": 9.0,
        "provenance": {"cytearcVersion": "1.3.0"},
        "sourceSha256": "abc",
    }


class CollectRunsTests(unittest.TestCase):
    def assertPublic(self, record: dict, allowed: tuple[str, ...] = ()) -> None:
        text = json.dumps(record)
        for private in PRIVATE:
            if private not in allowed:
                self.assertNotIn(private, text)

    def test_cohort_record_keeps_the_reported_measurements(self) -> None:
        record = blood_cohort_analysis(cohort_run())
        self.assertPublic(record)
        self.assertNotIn("region", record)
        result = record["result"]
        self.assertEqual(
            set(result),
            {
                "settings",
                "timingBoundary",
                "sourceCellCounts",
                "cells",
                "genesShared",
                "localStoreAfterCreateBytes",
                "creation",
                "unintegrated",
                "pseudobulk",
                "pseudobulkMeasurement",
            },
        )
        self.assertEqual(
            result["unintegrated"],
            {
                "elapsedSeconds": 2.5,
                "io": {"readBytes": 22, "requests": 3, "httpResponseStatusCounts": {"206": 3}},
                "resources": {
                    "operationBaselineBytes": 100,
                    "operationPeakBytes": 500,
                    "operationPeakSource": "cgroupMemoryCurrent",
                    "memoryMaxBytes": 1000,
                },
            },
        )
        self.assertNotIn("resources", result["creation"])
        self.assertEqual(result["pseudobulk"], {"groups": 4, "features": 5})

    def test_a_run_that_did_not_complete_is_refused(self) -> None:
        run = cohort_run()
        run["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "status 'failed'"):
            blood_cohort_analysis(run)
        scanpy = scanpy_run()
        scanpy["status"] = "error"
        with self.assertRaisesRegex(ValueError, "expected 'ok'"):
            blood_scanpy_dask(scanpy)

    def test_a_missing_measurement_fails_loudly(self) -> None:
        run = cohort_run()
        del run["result"]["unintegrated"]["resources"]["operationPeakBytes"]
        with self.assertRaises(KeyError):
            blood_cohort_analysis(run)

    def test_on_disk_record_keeps_the_failed_attempt_and_the_preparation(self) -> None:
        record = blood_on_disk_concatenation(on_disk_run("failed"))
        self.assertPublic(record)
        result = record["result"]
        self.assertEqual(list(result), ["onDiskConcatenation"])
        kept = result["onDiskConcatenation"]
        self.assertEqual(kept["asExportedAttempt"]["errorType"], "TypeError")
        self.assertEqual(kept["asExportedAttempt"]["processIoBytes"], {"rchar": 7, "wchar": 9})
        self.assertNotIn("obsKindsByInput", kept["inputPreparation"])
        self.assertEqual(kept["inputPreparation"]["obsColumnsStoredAsStringCategories"], ["age"])
        self.assertEqual(kept["inputPreparationMeasurement"]["elapsedSeconds"], 2.0)
        self.assertEqual(kept["concatenation"]["elapsedSeconds"], 4.0)
        self.assertEqual(kept["staging"], {"bytes": 50})
        for dropped in (
            "h5pyVersion",
            "upload",
            "jobSeconds",
            "validationMeasurement",
            "analysisInput",
        ):
            self.assertNotIn(dropped, kept)

    def test_on_disk_record_without_a_failure_has_no_preparation(self) -> None:
        kept = blood_on_disk_concatenation(on_disk_run("completed"))["result"][
            "onDiskConcatenation"
        ]
        self.assertNotIn("inputPreparation", kept)
        self.assertNotIn("errorType", kept["asExportedAttempt"])
        self.assertEqual(kept["concatenation"], kept["asExportedAttempt"])

    def test_scanpy_record_keeps_the_placement_and_drops_run_history(self) -> None:
        record = blood_scanpy_dask(scanpy_run())
        self.assertPublic(record)
        self.assertEqual(record["container"]["region"], "uk")
        self.assertEqual(record["download"], {"bytes": 60, "seconds": 4.0})
        for dropped in (
            "datasetUri",
            "image",
            "steps",
            "memorySeries",
            "log",
            "restarts",
            "finishedAt",
        ):
            self.assertNotIn(dropped, record)

    def test_composite_record_drops_locations_and_keeps_the_figure_fields(self) -> None:
        record = composite_and_remote(composite_run())
        self.assertPublic(record, allowed=("startedAt",))
        # The figure reader keys records on module, function and startedAt and reads region.
        self.assertEqual(
            list(record),
            ["module", "function", "region", "startedAt", "status", "result", "elapsedSeconds"],
        )
        composite = record["result"]["composite"]
        self.assertEqual(
            composite["settings"], {"mode": "composite", "memBudget": "24G", "replicate": 2}
        )
        self.assertEqual(composite["materializedInputs"], [{"name": "a", "cells": 10}])

    def test_stored_records_are_what_the_collection_step_writes(self) -> None:
        # Collecting a stored record again changes nothing: the records hold exactly the fields
        # that collect_runs keeps.
        cases = [
            (blood_cohort_analysis, RECORDS / "blood_cohort_analysis.json"),
            *(
                (blood_on_disk_concatenation, p)
                for p in sorted((RECORDS / "blood_on_disk_concatenation").glob("*.json"))
            ),
            *(
                (blood_scanpy_dask, p)
                for p in sorted((RECORDS / "blood_scanpy_dask").glob("*.json"))
            ),
            *(
                (composite_and_remote, p)
                for p in sorted((RECORDS / "composite_and_remote").glob("*/*.json"))
            ),
        ]
        self.assertEqual(len(cases), 1 + 3 + 3 + 12)
        for trim, path in cases:
            with self.subTest(path=path.relative_to(RECORDS)):
                stored = json.loads(path.read_text())
                self.assertEqual(trim(copy.deepcopy(stored)), stored)


class CompositeAndRemoteSummaryTests(unittest.TestCase):
    def test_means_and_saving_follow_the_replicates(self) -> None:
        def composite(
            reused: float, fresh: float, copy: float = 150.0, local: float = 60.0
        ) -> dict:
            return {
                "composite": {
                    "sharedUpstreamArtifacts": {"pca": True, "normalized": True},
                    "integrationReusingUpstream": {"elapsedSeconds": reused},
                    "integrationFresh": {"elapsedSeconds": fresh},
                    "materialization": {"elapsedSeconds": copy},
                    "materializedAnalysis": {"elapsedSeconds": local},
                }
            }

        summary = composite_and_remote_summary.composite_integration(
            [
                composite(100.0, 200.0, copy=160.0, local=50.0),
                composite(140.0, 200.0, copy=150.0, local=70.0),
            ]
        )
        self.assertEqual(summary["reusedIntegrationSeconds"], 120.0)
        self.assertEqual(summary["freshIntegrationSeconds"], 200.0)
        self.assertAlmostEqual(summary["timeReductionFraction"], 0.4)
        # A run's route through a local copy adds its own two times: 210 and 220 seconds, not the sum of the extremes
        # (160 + 70 = 230) or of the shortest (150 + 50 = 200).
        self.assertEqual(summary["materializedRouteSecondsLowest"], 210.0)
        self.assertEqual(summary["materializedRouteSecondsHighest"], 220.0)

    def test_a_run_without_reuse_is_refused(self) -> None:
        broken = {
            "composite": {
                "sharedUpstreamArtifacts": {"pca": False},
                "integrationReusingUpstream": {"elapsedSeconds": 1.0},
                "integrationFresh": {"elapsedSeconds": 2.0},
            }
        }
        with self.assertRaisesRegex(ValueError, "did not reuse"):
            composite_and_remote_summary.composite_integration([broken])

    def test_first_query_adds_opening_to_reading(self) -> None:
        summary = composite_and_remote_summary.remote_query(
            [
                remote_query(4.0, 6.0, 3.0, opened=100, read=900, http=(300, 1000, 950)),
                remote_query(3.0, 5.0, 2.0, opened=100, read=1100, http=(500, 1200, 1050)),
            ]
        )
        self.assertEqual(summary["firstOpenAndQuerySeconds"], 9.0)
        self.assertEqual(summary["repeatedQuerySeconds"], 2.5)
        # The store returned 1,000 and 1,200 bytes on opening and querying.
        self.assertEqual(summary["firstOpenAndQueryBytes"], 1100.0)
        # Over HTTP: 1,300 and 1,700 bytes, of which opening took 300 and 500; the repeats 950 and 1,050.
        self.assertEqual(summary["firstOpenAndQueryHttpBytes"], 1500.0)
        self.assertEqual((summary["openHttpBytes"], summary["queryHttpBytes"]), (400.0, 1100.0))
        self.assertEqual(summary["repeatedQueryHttpBytes"], 1000.0)

    def test_census_and_the_download_are_measured_against_the_first_query(self) -> None:
        # The remote query takes 10 seconds and 1,000 bytes returned to Zarr, 1,200 over HTTP; Census 60 and 70
        # seconds for 50,000 and 70,000 bytes; the download 200 seconds for 400,000 bytes.
        remote = [remote_query(4.0, 6.0, 1.0, opened=100, read=900, http=(200, 1000, 0))]
        census = [census_query(60.0, 50_000), census_query(70.0, 70_000)]
        source = {"nCells": 1000, "downloadSeconds": 200.0, "sourceBytes": 400_000}
        result = composite_and_remote_summary.query_alternatives(remote, census, source)
        self.assertEqual((result["census"]["seconds"], result["census"]["bytes"]), (65.0, 60_000))
        self.assertEqual(
            (result["census"]["timeOverQuery"], result["census"]["dataOverQuery"]), (6.5, 60.0)
        )
        self.assertEqual(
            (result["download"]["timeOverQuery"], result["download"]["dataOverQuery"]),
            (20.0, 400.0),
        )
        self.assertEqual(result["census"]["dataOverQueryHttp"], 50.0)
        self.assertAlmostEqual(result["download"]["dataOverQueryHttp"], 400_000 / 1200, places=12)

    def test_the_records_give_68_9_mb_over_http_and_67_and_653_fold_less_data(self) -> None:
        # Paragraphs 50 and 75: first open plus query received 68.9 MB of HTTP response bodies (0.20 MB opening, 68.7
        # MB querying; 68.6 MB returned to Zarr), repeated queries 66.4 MB; Census received 67-fold and the H5AD
        # download 653-fold the HTTP bytes. Worked out from the three replicate records directly.
        summary = composite_and_remote_summary
        remote = summary.replicates("remote_queries")
        queries = [r["remoteQueries"]["tabulaSapiens1_1M"] for r in remote]
        opened = [q["openIo"]["http"]["responseBytes"] for q in queries]
        read = [q["embeddingAndMarkerQueryIo"]["http"]["responseBytes"] for q in queries]
        repeated = [q["repeatedQuery"]["io"]["http"]["responseBytes"] for q in queries]
        first = (sum(opened) + sum(read)) / 3
        result = summary.remote_query(remote)
        self.assertEqual(len(queries), 3)
        self.assertEqual(result["firstOpenAndQueryHttpBytes"], first)
        self.assertEqual(round(first / 1e6, 1), 68.9)
        self.assertEqual(round(result["openHttpBytes"] / 1e6, 2), 0.20)
        self.assertEqual(round(result["queryHttpBytes"] / 1e6, 1), 68.7)
        self.assertEqual(round(result["firstOpenAndQueryBytes"] / 1e6, 1), 68.6)
        self.assertEqual(round(sum(repeated) / 3 / 1e6, 1), 66.4)
        alternatives = summary.query_alternatives(
            remote,
            summary.records_in(summary.CENSUS_QUERIES),
            summary.load_json(summary.WORKFLOW_RUNS)["inputs"]["tabula_sapiens"],
        )
        self.assertEqual(round(alternatives["census"]["dataOverQueryHttp"]), 67)
        self.assertEqual(round(alternatives["download"]["dataOverQueryHttp"]), 653)
        self.assertEqual(round(alternatives["download"]["bytes"] / 1e9, 1), 45.0)

    def test_a_census_query_of_other_cells_or_one_that_did_not_complete_is_refused(self) -> None:
        remote = [remote_query(4.0, 6.0, 1.0, opened=100, read=900)]
        source = {"nCells": 1000, "downloadSeconds": 200.0, "sourceBytes": 400_000}
        other_cells = census_query(60.0, 50_000)
        other_cells["result"]["cellsSelected"] = 99
        with self.assertRaisesRegex(ValueError, "other cells than the CyteBase query"):
            composite_and_remote_summary.query_alternatives(remote, [other_cells], source)
        stopped = {**census_query(60.0, 50_000), "status": "failed"}
        with self.assertRaisesRegex(ValueError, "did not complete"):
            composite_and_remote_summary.query_alternatives(remote, [stopped], source)
        with self.assertRaisesRegex(ValueError, "other cells than the queried store"):
            composite_and_remote_summary.query_alternatives(
                remote, [census_query(60.0, 50_000)], {**source, "nCells": 999}
            )

    def test_the_footprint_divides_each_route_by_the_composite(self) -> None:
        # Composite: 2 GiB on disk, 3 and 5 GiB of memory (4 on average). In memory: 40 GiB of H5AD files and 80 GiB
        # of memory. On disk: the same files and 60 GiB written, then 10 and 14 GiB in Scanpy. BPCells: a work
        # directory of 70 GiB (one run kept none) and 6 GiB of memory.
        result = composite_and_remote_summary.composite_footprint(*footprint_routes())
        self.assertEqual(
            result["composite"], {"replicates": 2, "diskBytes": 2 * GIB, "memoryGiB": 4.0}
        )
        in_memory = result["inMemoryConcatenation"]
        self.assertEqual(
            (in_memory["diskOverComposite"], in_memory["memoryOverComposite"]), (20.0, 20.0)
        )
        on_disk = result["onDiskConcatenationThenScanpyDask"]
        self.assertEqual((on_disk["diskBytes"], on_disk["memoryGiB"]), (100 * GIB, 12.0))
        self.assertEqual(
            (on_disk["diskOverComposite"], on_disk["memoryOverComposite"]), (50.0, 3.0)
        )
        bpcells = result["bpcells"]
        self.assertEqual(
            (bpcells["diskBytes"], bpcells["diskReplicates"], bpcells["replicates"]),
            (70 * GIB, 1, 2),
        )
        self.assertEqual(
            (bpcells["diskOverComposite"], bpcells["memoryOverComposite"]), (35.0, 1.5)
        )

    def test_routes_that_read_other_files_or_a_composite_that_copied_counts_are_refused(
        self,
    ) -> None:
        composites, in_memory, on_disk, scanpy, bpcells = footprint_routes()
        on_disk[0]["inputs"][0]["bytes"] += 1
        with self.assertRaisesRegex(ValueError, "read different H5AD files"):
            composite_and_remote_summary.composite_footprint(
                composites, in_memory, on_disk, scanpy, bpcells
            )
        composites, in_memory, on_disk, scanpy, bpcells = footprint_routes()
        composites[1]["copiedCountChunkBytes"] = 1
        with self.assertRaisesRegex(ValueError, "copied count chunks"):
            composite_and_remote_summary.composite_footprint(
                composites, in_memory, on_disk, scanpy, bpcells
            )
        composites, in_memory, on_disk, scanpy, bpcells = footprint_routes()
        del bpcells[1]["peakWorkDirectoryBytes"]
        with self.assertRaisesRegex(ValueError, "No BPCells run recorded its disk"):
            composite_and_remote_summary.composite_footprint(
                composites, in_memory, on_disk, scanpy, bpcells
            )

    def test_stored_records_give_the_quoted_means(self) -> None:
        # Paragraphs 50 and 52 quote 9.0 and 2.6 seconds, and 3.46 and 2.36 minutes.
        remote_runs = composite_and_remote_summary.replicates("remote_queries")
        remote = composite_and_remote_summary.remote_query(remote_runs)
        self.assertEqual(round(remote["firstOpenAndQuerySeconds"], 1), 9.0)
        self.assertEqual(round(remote["repeatedQuerySeconds"], 1), 2.6)
        composite = composite_and_remote_summary.composite_integration(
            composite_and_remote_summary.replicates("composite_integration")
        )
        self.assertEqual(round(composite["freshIntegrationSeconds"] / 60, 2), 3.46)
        self.assertEqual(round(composite["reusedIntegrationSeconds"] / 60, 2), 2.36)
        # Paragraph 80: copying the four-study composite and analysing the copy took 214 to 230 seconds per run.
        self.assertEqual(
            (
                round(composite["materializedRouteSecondsLowest"]),
                round(composite["materializedRouteSecondsHighest"]),
            ),
            (214, 230),
        )
        # Paragraph 50: 68.6 MB on first open; Census 60.6 seconds and 4.65 GB, 6.7 and 68 times the query's; the
        # download 25 and 656 times.
        self.assertEqual(round(remote["firstOpenAndQueryBytes"] / 1e6, 1), 68.6)
        alternatives = composite_and_remote_summary.query_alternatives(
            remote_runs,
            composite_and_remote_summary.records_in(composite_and_remote_summary.CENSUS_QUERIES),
            composite_and_remote_summary.load_json(composite_and_remote_summary.WORKFLOW_RUNS)[
                "inputs"
            ]["tabula_sapiens"],
        )
        census, download = alternatives["census"], alternatives["download"]
        self.assertEqual(
            (round(census["seconds"], 1), round(census["bytes"] / 1e9, 2)), (60.6, 4.65)
        )
        self.assertEqual(
            (round(census["timeOverQuery"], 1), round(census["dataOverQuery"])), (6.7, 68)
        )
        self.assertEqual(
            (round(download["timeOverQuery"]), round(download["dataOverQuery"])), (25, 656)
        )

    def test_the_mounted_store_compares_its_copy_branch_and_disk_with_its_sources(self) -> None:
        # The mount copies 5 of 500 bytes; the branch takes 60, 30 and 90 seconds; the lineage leaves 8 bytes in the
        # user store against a source H5AD of 200. The published run spends 8 minutes on its clustering chain, 12
        # through the marker search and 20 from the source H5AD after its download.
        record = mounted_record()
        result = composite_and_remote_summary.mounted_store(record, published_run(MOUNTED_TAG))
        self.assertEqual(result["mountShareOfSource"], 0.01)
        self.assertEqual(result["k21BranchMinutes"], 3.0)
        self.assertEqual(result["k21BranchBytes"], 101)
        self.assertEqual(result["sourceH5adOverLineageUserStore"], 25.0)
        self.assertEqual(
            (
                result["publishedClusteringMinutes"],
                result["publishedAnalysisMinutes"],
                result["publishedRunMinutes"],
            ),
            (8.0, 12.0, 20.0),
        )
        record["source"]["unchanged"] = False
        with self.assertRaisesRegex(ValueError, "changed its source store"):
            composite_and_remote_summary.mounted_store(record, published_run(MOUNTED_TAG))

    def test_a_published_run_that_is_not_the_mounted_one_or_did_not_complete_is_refused(
        self,
    ) -> None:
        other = published_run("embryo_rna_scaling_10000000_cells_replicate_5")
        with self.assertRaisesRegex(
            ValueError, "mounted the store of .*replicate_6, not of .*replicate_5"
        ):
            composite_and_remote_summary.mounted_store(mounted_record(), other)
        failed = published_run(MOUNTED_TAG)
        failed["funnel"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, r"did not complete; unfinished stages: \[\]"):
            composite_and_remote_summary.mounted_store(mounted_record(), failed)
        unfinished = published_run(MOUNTED_TAG)
        unfinished["stages"]["runUmap"]["status"] = "failed"
        del unfinished["stages"]["findMarkers"]
        with self.assertRaisesRegex(ValueError, r"\['runUmap', 'findMarkers'\]"):
            composite_and_remote_summary.mounted_store(mounted_record(), unfinished)

    def test_the_mounted_store_record_gives_the_quoted_values(self) -> None:
        # Paragraph 49: the mount copies 1.2% of the 64.8 GB store. Paragraph 52: the k = 21 branch took 11.7 minutes,
        # and 1.72 GB of local disk is 23 times smaller than the 40.2 GB source H5AD; the 16-CPU run that wrote the
        # store took 29.7 minutes through Leiden, 58.3 through marker discovery and 83.9 from the source H5AD.
        summary = composite_and_remote_summary
        mount = summary.load_json(summary.MOUNTED_STORE)
        run = summary.load_json(summary.SCALING_RUNS)[mount["referenceTag"]]
        result = summary.mounted_store(mount, run)
        h5ad = result["sourceH5adBytes"]
        self.assertEqual(
            (
                round(result["mountShareOfSource"] * 100, 1),
                round(result["sourceStoreBytes"] / 1e9, 1),
            ),
            (1.2, 64.8),
        )
        self.assertEqual(round(result["k21BranchMinutes"], 1), 11.7)
        # Paragraph 84: the branch wrote 2.79 GB, its neighbor table, graph and labels of 1.11 GB, 1.67 GB and 6.6 MB.
        written = [mount["steps"][step]["userStoreBytesWritten"] for step in summary.K21_BRANCH]
        self.assertEqual(
            [round(written[0] / 1e9, 2), round(written[1] / 1e9, 2), round(written[2] / 1e6, 1)],
            [1.11, 1.67, 6.6],
        )
        self.assertEqual(round(result["k21BranchBytes"] / 1e9, 2), 2.79)
        self.assertEqual(
            (round(result["lineageUserStoreBytes"] / 1e9, 2), round(h5ad / 1e9, 1)), (1.72, 40.2)
        )
        self.assertEqual(round(result["sourceH5adOverLineageUserStore"]), 23)
        # Worked out from the raw record another way: the analysis through the marker search is every stage but the
        # three conversion stages and the reopening of the store; the clustering chain is that analysis without the
        # UMAP, its initialization and the marker search; the run from the source H5AD is the whole run without the
        # download.
        seconds = {stage: s["seconds"] for stage, s in run["stages"].items()}
        analysis = sum(seconds.values()) - sum(
            seconds[s] for s in ("createStore", "writeCountsT", "initializeStore", "reopenStore")
        )
        clustering = analysis - sum(
            seconds[s] for s in ("buildEmbeddingInitialization", "runUmap", "findMarkers")
        )
        funnel = run["funnel"]
        self.assertEqual(len(seconds), 15)
        self.assertEqual(funnel["modalResources"]["modalCpuLimit"], 16.0)
        for field, expected, quoted in (
            ("publishedClusteringMinutes", clustering / 60, 29.7),
            ("publishedAnalysisMinutes", analysis / 60, 58.3),
            (
                "publishedRunMinutes",
                (funnel["wholeFunctionSeconds"] - funnel["datasetDownloadSeconds"]) / 60,
                83.9,
            ),
        ):
            with self.subTest(field):
                self.assertAlmostEqual(result[field], expected, places=9)
                self.assertEqual(round(result[field], 1), quoted)

    def test_stored_records_give_the_quoted_footprint(self) -> None:
        # Paragraph 51: 27 and 14 times the composite's disk and memory in memory; 109 and 5.6 times, 110.3 GiB
        # (118.4 GB) and 68.2 GiB on disk with Scanpy on Dask; 123 and 2.4 times, 124.9 GiB (134.1 GB) for BPCells.
        summary = composite_and_remote_summary
        result = summary.composite_footprint(
            [r["composite"] for r in summary.replicates("blood_composite")],
            [r["materializedConcatenation"] for r in summary.replicates("blood_concatenation")],
            [
                r["result"]["onDiskConcatenation"]
                for r in summary.records_in(summary.ON_DISK_CONCATENATION)
            ],
            summary.records_in(summary.SCANPY_DASK),
            summary.records_in(summary.BPCELLS),
        )
        in_memory = result["inMemoryConcatenation"]
        self.assertEqual(
            (round(in_memory["diskOverComposite"]), round(in_memory["memoryOverComposite"])),
            (27, 14),
        )
        on_disk = result["onDiskConcatenationThenScanpyDask"]
        self.assertEqual(
            (round(on_disk["diskOverComposite"]), round(on_disk["memoryOverComposite"], 1)),
            (109, 5.6),
        )
        self.assertEqual(
            (round(on_disk["diskBytes"] / GIB, 1), round(on_disk["diskBytes"] / 1e9, 1)),
            (110.3, 118.4),
        )
        self.assertEqual(round(on_disk["memoryGiB"], 1), 68.2)
        bpcells = result["bpcells"]
        self.assertEqual(
            (round(bpcells["diskOverComposite"]), round(bpcells["memoryOverComposite"], 1)),
            (123, 2.4),
        )
        self.assertEqual(
            (round(bpcells["diskBytes"] / GIB, 1), round(bpcells["diskBytes"] / 1e9, 1)),
            (125.0, 134.2),
        )
        self.assertEqual((bpcells["diskReplicates"], round(bpcells["memoryGiB"], 1)), (3, 29.8))


GIB = 1024**3


def remote_query(
    open_seconds: float,
    read_seconds: float,
    repeat_seconds: float,
    *,
    opened: int,
    read: int,
    http: tuple[int, int, int] = (0, 0, 0),
) -> dict:
    """A remote query's result, as its record keeps it, of 1,000 cells with 100 B cells; ``http`` holds the HTTP
    response-body bytes of opening, querying and the repeated query."""
    return {
        "remoteQueries": {
            "tabulaSapiens1_1M": {
                "totalCells": 1000,
                "bCellsSelected": 100,
                "openSeconds": open_seconds,
                "openIo": {"readBytes": opened, "http": {"responseBytes": http[0]}},
                "embeddingAndMarkerQuerySeconds": read_seconds,
                "embeddingAndMarkerQueryIo": {
                    "readBytes": read,
                    "http": {"responseBytes": http[1]},
                },
                "repeatedQuery": {
                    "elapsedSeconds": repeat_seconds,
                    "io": {"http": {"responseBytes": http[2]}},
                },
            }
        }
    }


def census_query(seconds: float, received: int) -> dict:
    """A Census query's record, split evenly between opening and querying."""
    return {
        "status": "completed",
        "result": {
            "totalCells": 1000,
            "cellsSelected": 100,
            "openSeconds": seconds / 2,
            "querySeconds": seconds / 2,
            "openReceivedBytes": received // 2,
            "queryReceivedBytes": received - received // 2,
        },
    }


def footprint_routes() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """The composite runs and the three physical routes of test_the_footprint_divides_each_route_by_the_composite."""
    files = [{"bytes": 15 * GIB}, {"bytes": 25 * GIB}]
    composites = [
        {
            "localStoreAfterCreateBytes": 2 * GIB,
            "copiedCountChunkBytes": 0,
            "unintegrated": {"resources": {"operationPeakBytes": peak * GIB}},
        }
        for peak in (3, 5)
    ]
    in_memory = [
        {"inputFiles": copy.deepcopy(files), "resources": {"operationPeakBytes": 80 * GIB}}
    ]
    on_disk = [
        {"inputs": copy.deepcopy(files), "concatenation": {"processIoBytes": {"wchar": 60 * GIB}}}
    ]
    scanpy = [{"status": "ok", "peakCgroupBytes": peak * GIB} for peak in (10, 14)]
    bpcells = [
        {"status": "ok", "download": {"bytes": 40 * GIB}, "peakCgroupBytes": 6 * GIB},
        {
            "status": "ok",
            "download": {"bytes": 40 * GIB},
            "peakCgroupBytes": 6 * GIB,
            "peakWorkDirectoryBytes": 70 * GIB,
        },
    ]
    return composites, in_memory, on_disk, scanpy, bpcells


MOUNTED_TAG = "embryo_rna_scaling_10000000_cells_replicate_6"


def mounted_record() -> dict:
    """A reanalysis of the store of MOUNTED_TAG whose mount copies 5 of 500 bytes, whose branch at k = 21 takes 60,
    30 and 90 seconds and writes 40, 60 and 1 bytes, and whose lineage leaves 8 bytes in the user store."""
    return {
        "referenceTag": MOUNTED_TAG,
        "source": {"unchanged": True, "before": {"bytes": 500}},
        "userStore": {"afterMount": {"bytes": 5}},
        "steps": {
            "umap.k11": {"userStoreBytesAfter": 8},
            "reuse.neighbors.k21": {"seconds": 60.0, "userStoreBytesWritten": 40},
            "reuse.graph.k21": {"seconds": 30.0, "userStoreBytesWritten": 60},
            "reuse.leiden.k21_r1": {"seconds": 90.0, "userStoreBytesWritten": 1},
        },
    }


def published_run(tag: str) -> dict:
    """A scaling run from a source H5AD of 200 bytes whose eight clustering stages take 60 seconds each and whose UMAP
    initialization, UMAP and marker search take 30, 120 and 90 more. Its conversion stages and the reopening of its
    store take a million seconds each, which would change any analysis time that counted them, and the whole run
    takes 1,500 seconds, 300 of them the download."""
    seconds = dict.fromkeys(("createStore", "writeCountsT", "initializeStore", "reopenStore"), 1e6)
    clustering = (
        "filterCells",
        "markHvgs",
        "runNormalization",
        "runPca",
        "buildAnnIndex",
        "queryNeighbors",
        "buildConnectivityMap",
        "runLeiden",
    )
    seconds |= dict.fromkeys(clustering, 60.0)
    seconds |= {"buildEmbeddingInitialization": 30.0, "runUmap": 120.0, "findMarkers": 90.0}
    return {
        "funnel": {
            "runTag": tag,
            "status": "ok",
            "datasetBytes": 200,
            "wholeFunctionSeconds": 1500.0,
            "datasetDownloadSeconds": 300.0,
        },
        "stages": {stage: {"status": "ok", "seconds": s} for stage, s in seconds.items()},
    }


def progenitor_run(**result) -> dict:
    """A progenitor job's run record as the Modal runner saves it, with its payload's locations."""
    return {
        "module": "reuse.blood_progenitors",
        "function": "discover",
        "callId": "fc-01ABCDEFGHIJKLMNOPQRSTUV",
        "region": "us-east",
        "startedAt": 1.0,
        "status": "completed",
        "elapsedSeconds": 5.0,
        "result": {
            "settings": {
                "memBudget": "3G",
                "recordUri": "s3://bucket/reuse/discovery.json",
                "outputPrefix": "s3://bucket/reuse/discovery",
                "workDirectory": "/tmp/blood_progenitors",
            },
            **result,
        },
    }


class PromotedRunRecordsTests(unittest.TestCase):
    def assertPublic(self, record: dict) -> None:
        text = json.dumps(record)
        for private in PRIVATE:
            self.assertNotIn(private, text)

    def test_public_run_drops_history_locations_and_identifiers(self) -> None:
        run = progenitor_run(
            tables={"cells": {"rows": 3, "bytes": 30, "uri": "s3://bucket/reuse/cells.parquet"}},
            outputs={"a.parquet": "s3://bucket/reuse/a.parquet"},
            stores=[
                {
                    "storeId": "s",
                    "error": "read fc-01ABCDEFGHIJKLMNOPQRSTUV at s3://bucket/x failed",
                }
            ],
        )
        run["finishedAt"] = 2.0
        record = public_run(run)
        text = json.dumps(record)
        for private in (
            "fc-01ABCDEFGHIJKLMNOPQRSTUV",
            "s3://bucket",
            "/tmp/",
            "startedAt",
            "callId",
        ):
            self.assertNotIn(private, text)
        self.assertEqual(
            list(record), ["module", "function", "region", "status", "elapsedSeconds", "result"]
        )
        self.assertEqual(record["result"]["settings"], {"memBudget": "3G"})
        # A location is dropped with its key, and a mapping of nothing but locations with it; a location inside
        # other text is masked.
        self.assertEqual(record["result"]["tables"], {"cells": {"rows": 3, "bytes": 30}})
        self.assertNotIn("outputs", record["result"])
        self.assertEqual(
            record["result"]["stores"][0]["error"], "read <modal-id> at s3://<bucket>/x failed"
        )
        self.assertEqual(run["result"]["tables"]["cells"]["uri"], "s3://bucket/reuse/cells.parquet")

    def test_mixing_record_keeps_the_settings_cells_and_each_analysis_mixing(self) -> None:
        run = cohort_run()
        for phase, value in (("unintegrated", 0.2), ("integrated", 0.5)):
            run["result"][phase]["mixing"] = {"macroStudyMixing": value, "definition": "entropy"}
        record = blood_cohort_mixing(run)
        self.assertPublic(record)
        self.assertEqual(
            set(record["result"]),
            {"settings", "cells", "sourceCellCounts", "unintegrated", "integrated"},
        )
        self.assertEqual(
            record["result"]["integrated"],
            {"mixing": {"macroStudyMixing": 0.5, "definition": "entropy"}},
        )
        self.assertNotIn("workDirectory", record["result"]["settings"])
        del run["result"]["integrated"]["mixing"]
        with self.assertRaises(KeyError):
            blood_cohort_mixing(run)

    def test_tool_reports_and_job_runs_are_refused_unless_they_completed(self) -> None:
        report = {
            "tool": "bpcells_cached",
            "status": "failed",
            "callId": "fc-01ABCDEFGHIJKLMNOPQRSTUV",
        }
        with self.assertRaisesRegex(ValueError, "expected 'ok'"):
            tool_report(report)
        report["status"] = "ok"
        self.assertEqual(tool_report(report), {"tool": "bpcells_cached", "status": "ok"})
        with self.assertRaisesRegex(ValueError, "status 'running'"):
            job_run(progenitor_run() | {"status": "running"})

    def test_progenitor_tables_are_the_discovery_and_analysis_uploads(self) -> None:
        discovery = progenitor_run(
            tables={
                "cells": {"rows": 3, "bytes": 30, "uri": "s3://bucket/d/progenitor_cells.parquet"},
                "donors": {
                    "rows": 2,
                    "bytes": 20,
                    "uri": "s3://bucket/d/progenitor_donors.parquet",
                },
            }
        )
        self.assertEqual(
            collect_runs.progenitor_tables(discovery),
            {
                "progenitor_cells.parquet": ("s3://bucket/d/progenitor_cells.parquet", 30),
                "progenitor_donors.parquet": ("s3://bucket/d/progenitor_donors.parquet", 20),
            },
        )
        # The analysis's per-cell arrays stay in the bucket.
        analysis = progenitor_run(
            outputs={
                "progenitor_cells.npz": "s3://bucket/a/progenitor_cells.npz",
                "progenitor_cell_table.parquet": "s3://bucket/a/progenitor_cell_table.parquet",
                "progenitor_cluster_panel.parquet": "s3://bucket/a/progenitor_cluster_panel.parquet",
            }
        )
        self.assertEqual(
            sorted(collect_runs.progenitor_tables(analysis)),
            ["progenitor_cell_table.parquet", "progenitor_cluster_panel.parquet"],
        )

    def test_a_table_is_downloaded_unless_held_at_its_recorded_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            downloads = []

            def download_file(uri: str, destination: Path) -> None:
                downloads.append(uri)
                Path(destination).write_bytes(b"x" * 5)

            with (
                mock.patch.object(collect_runs, "HERE", folder / "experiments" / "reuse"),
                mock.patch("experiments.runtime.object_io.download_file", download_file),
            ):
                table = folder / "experiments" / "reuse" / "records" / "t.parquet"
                table.parent.mkdir(parents=True)
                table.write_bytes(b"x" * 5)
                collect_runs.fetch_table("s3://bucket/t.parquet", table, 5)
                self.assertEqual(downloads, [])
                table.write_bytes(b"x" * 4)
                collect_runs.fetch_table("s3://bucket/t.parquet", table, 5)
                self.assertEqual(downloads, ["s3://bucket/t.parquet"])
                with self.assertRaisesRegex(ValueError, "holds 5 bytes, but the job recorded 6"):
                    collect_runs.fetch_table("s3://bucket/t.parquet", table, 6)

    def test_promoted_records_are_public_and_what_the_collection_step_writes(self) -> None:
        cases = [
            (blood_cohort_mixing, RECORDS / "blood_cohort_mixing.json"),
            *((tool_report, p) for p in sorted((RECORDS / "blood_bpcells").glob("*.json"))),
            *((job_run, p) for p in sorted((RECORDS / "census_queries").glob("*.json"))),
            *((job_run, p) for p in sorted((RECORDS / "blood_progenitors").glob("*.json"))),
        ]
        # 3 BPCells runs, 3 Census queries, and the discovery, 8 extractions, the analysis and two value reads.
        self.assertEqual(len(cases), 1 + 3 + 3 + 12)
        for trim, path in cases:
            with self.subTest(path=path.relative_to(RECORDS)):
                stored = json.loads(path.read_text())
                self.assertPublic(stored)
                self.assertEqual(trim(copy.deepcopy(stored)), stored)


if __name__ == "__main__":
    unittest.main()
