import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from manuscript.figures.zero_copy_analysis import prepare

GIB = 1024**3


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TemporaryRoot(unittest.TestCase):
    """A temporary repository root for prepare's module paths; subclasses list the paths to point into it, a list of
    relative paths for a path list."""

    PATHS: dict[str, str | list[str]] = {}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, relative in {"ROOT": "", **self.PATHS}.items():
            value = (
                [self.root / r for r in relative]
                if isinstance(relative, list)
                else self.root / relative
            )
            patch = mock.patch.object(prepare, name, value)
            patch.start()
            self.addCleanup(patch.stop)


class QueryCostTest(TemporaryRoot):
    PATHS = {
        "REMOTE_QUERIES": ["q1.json", "q2.json", "q3.json"],
        "CENSUS_QUERIES": ["c1.json", "c2.json", "c3.json"],
        "REMOTE_SUMMARY": "summary.json",
        "WORKFLOW_RUNS": "workflow.json",
    }
    RUNS = ((4.0, 5.0, 2.5), (3.0, 4.0, 2.0), (5.0, 6.0, 3.0))
    # Census: opening and querying take 10, 20 and 30 seconds, and receive 140,000 bytes each time.
    CENSUS = ((4.0, 6.0), (5.0, 15.0), (10.0, 20.0))

    def write_inputs(
        self, *, summary_first=9.0, summary_http=70_000, cells=1000, census_b_cells=100
    ):
        for path, (open_seconds, query_seconds) in zip(prepare.CENSUS_QUERIES, self.CENSUS):
            write(
                path,
                {
                    "status": "completed",
                    "result": {
                        "totalCells": 1000,
                        "cellsSelected": census_b_cells,
                        "openSeconds": open_seconds,
                        "querySeconds": query_seconds,
                        "openReceivedBytes": 40_000,
                        "queryReceivedBytes": 100_000,
                    },
                },
            )
        for path, (open_seconds, query_seconds, repeat_seconds) in zip(
            prepare.REMOTE_QUERIES, self.RUNS
        ):
            write(
                path,
                {
                    "result": {
                        "remoteQueries": {
                            "tabulaSapiens1_1M": {
                                "totalCells": 1000,
                                "bCellsSelected": 100,
                                "openSeconds": open_seconds,
                                "embeddingAndMarkerQuerySeconds": query_seconds,
                                # 60,000 bytes returned to Zarr, 70,000 received over HTTP: the panel
                                # draws the HTTP bytes.
                                "openIo": {"readBytes": 1_000, "http": {"responseBytes": 3_000}},
                                "embeddingAndMarkerQueryIo": {
                                    "readBytes": 59_000,
                                    "http": {"responseBytes": 67_000},
                                },
                                "repeatedQuery": {
                                    "elapsedSeconds": repeat_seconds,
                                    "io": {"readBytes": 66_000},
                                },
                            }
                        }
                    }
                },
            )
        write(
            prepare.REMOTE_SUMMARY,
            {
                "remoteQuery": {
                    "firstOpenAndQuerySeconds": summary_first,
                    "firstOpenAndQueryHttpBytes": summary_http,
                    "repeatedQuerySeconds": 2.5,
                }
            },
        )
        write(
            prepare.WORKFLOW_RUNS,
            {
                "inputs": {
                    "tabula_sapiens": {
                        "nCells": cells,
                        "downloadSeconds": 200.0,
                        "sourceBytes": 7_000_000,
                    }
                }
            },
        )

    def test_each_query_adds_opening_to_reading_and_each_later_row_is_divided_by_cytebase(self):
        self.write_inputs()
        cost = prepare.query_cost()
        # The CyteBase query on its first open, the Census query and the download: the repeated query is not drawn.
        self.assertEqual([row["key"] for row in cost["rows"]], ["cytebase", "census", "download"])
        first, census, download = cost["rows"]
        # Opening and reading take 9, 7 and 11 seconds: mean 9, range 7 to 11.
        self.assertEqual(first["seconds"]["values"], [9.0, 7.0, 11.0])
        self.assertEqual((first["seconds"]["low"], first["seconds"]["high"]), (7.0, 11.0))
        self.assertEqual(first["bytes"]["mean"], 70_000)
        self.assertEqual(census["seconds"]["values"], [10.0, 20.0, 30.0])
        self.assertEqual(census["bytes"]["mean"], 140_000)
        self.assertEqual(download["seconds"]["n"], 1)
        self.assertAlmostEqual(census["timeOverQuery"], 20.0 / 9.0)
        self.assertAlmostEqual(census["dataOverQuery"], 2.0)
        self.assertAlmostEqual(download["timeOverQuery"], 200.0 / 9.0)
        self.assertAlmostEqual(download["dataOverQuery"], 100.0)
        self.assertNotIn("timeOverQuery", first)

    def test_a_census_query_of_other_cells_is_an_error(self):
        self.write_inputs(census_b_cells=99)
        with self.assertRaisesRegex(ValueError, "other cells"):
            prepare.query_cost()

    def test_seconds_that_differ_from_the_derived_summary_are_an_error(self):
        self.write_inputs(summary_first=9.5)
        with self.assertRaisesRegex(ValueError, "derived summary"):
            prepare.query_cost()

    def test_http_bytes_that_differ_from_the_derived_summary_are_an_error(self):
        self.write_inputs(summary_http=60_000)
        with self.assertRaisesRegex(ValueError, "HTTP bytes differ from the derived summary"):
            prepare.query_cost()

    def test_a_downloaded_file_of_other_cells_is_an_error(self):
        self.write_inputs(cells=999)
        with self.assertRaisesRegex(ValueError, "different cells"):
            prepare.query_cost()


class TabulaQueryTest(TemporaryRoot):
    PATHS = {
        "REMOTE_QUERIES": ["q1.json"],
        "TABULA_RASTER": "raster/raster.npz",
    }
    COST = {"rows": ["cytebase", "census", "download"], "source": ["q.json"]}

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(prepare, "query_cost", lambda: self.COST)
        patch.start()
        self.addCleanup(patch.stop)

    def write_inputs(self, *, mean=1.0):
        cells = np.zeros((4, 5), dtype=np.int32)
        b_cells = np.zeros_like(cells)
        b_sum = np.zeros((4, 5))
        cells[0, 0], cells[3, 4], cells[1, 2] = 5, 3, 2
        b_cells[1, 2], b_sum[1, 2] = 2, 2.0
        prepare.TABULA_RASTER.parent.mkdir(parents=True, exist_ok=True)
        np.savez(prepare.TABULA_RASTER, cells=cells, bCells=b_cells, bMs4a1Sum=b_sum)
        write(
            prepare.REMOTE_QUERIES[0],
            {
                "result": {
                    "remoteQueries": {
                        "tabulaSapiens1_1M": {
                            "totalCells": 10,
                            "bCellsSelected": 2,
                            "ms4a1MeanInBCells": mean,
                        }
                    }
                }
            },
        )

    def test_the_raster_matches_the_record_and_names_the_b_cells_where_they_are(self):
        self.write_inputs()
        query = prepare.tabula_query()
        self.assertEqual((query["cells"], query["bCells"]), (10, 2))
        self.assertEqual(query["bCellLabel"], {"row": 1.0, "column": 2.0})
        self.assertEqual(query["ms4a1Top"], 1.0)

    def test_the_query_carries_its_time_and_data_against_the_download(self):
        self.write_inputs()
        query = prepare.tabula_query()
        self.assertEqual(query["rows"], ["cytebase", "census", "download"])
        self.assertEqual(query["source"], ["raster/raster.npz", "q.json"])

    def test_values_that_differ_from_the_record_are_an_error(self):
        # The raster's 2 B cells sum to 2.0, a mean of 1.0, where the record holds 1.5.
        self.write_inputs(mean=1.5)
        with self.assertRaisesRegex(ValueError, "MS4A1"):
            prepare.tabula_query()


class CompositeFootprintTest(TemporaryRoot):
    PATHS = {
        "BLOOD_COMPOSITE": ["c2.json", "c3.json", "c4.json"],
        "IN_MEMORY_CONCATENATION": ["m1.json", "m2.json", "m3.json"],
        "ON_DISK_CONCATENATION": ["d1.json", "d2.json", "d3.json"],
        "SCANPY_DASK": ["s1.json", "s2.json", "s3.json"],
        "BPCELLS_BLOOD": ["b1.json", "b2.json", "b3.json"],
        "COVID_COMPOSITE": ["v1.json", "v2.json", "v3.json"],
        "BLOOD_COHORT": "cohort.json",
        "BLOOD_COHORT_MIXING": "cohort_mixing.json",
        "COMPOSITE_RASTER": "raster.npz",
    }

    def write_inputs(
        self,
        *,
        store_bytes=(GIB, GIB, GIB),
        copied=0,
        scanpy_status="ok",
        exact=True,
        cohort=(("small", 10), ("large", 20)),
        binned=(20, 10),
        bpcells_status="ok",
        bpcells_download=27 * GIB,
        bpcells_disk=(None, 127 * GIB, 127 * GIB),
        full_cohort=None,
    ):
        write(prepare.BLOOD_COHORT, {"result": {"sourceCellCounts": dict(cohort)}})
        # The same run's mixing record, with each analysis's mixing of studies and same-type neighbours.
        write(
            prepare.BLOOD_COHORT_MIXING,
            {
                "result": {
                    "sourceCellCounts": dict(full_cohort or cohort),
                    "unintegrated": {
                        "mixing": {"macroStudyMixing": 0.2, "sameTypeNeighborFraction": 0.97}
                    },
                    "integrated": {
                        "mixing": {"macroStudyMixing": 0.5, "sameTypeNeighborFraction": 0.96}
                    },
                }
            },
        )
        # The raster lists its studies in an order of its own, neither by name nor by size, with each study's
        # cells in bins; binned gives the cells of the large study and of the small one.
        cells = np.zeros((2, 3, 3), dtype=np.uint32)
        cells[1, 0, 0], cells[1, 2, 1] = binned[0] - 5, 5
        cells[0, 1, 2] = binned[1]
        np.savez_compressed(
            prepare.COMPOSITE_RASTER, studyCells=cells, studies=np.asarray(["small", "large"])
        )
        for path, size, peak in zip(prepare.BLOOD_COMPOSITE, store_bytes, (12, 13, 14)):
            write(
                path,
                {
                    "result": {
                        "composite": {
                            "localStoreAfterCreateBytes": size,
                            "copiedCountChunkBytes": copied,
                            "cellsTotalSelected": 30,
                            "genesShared": 7,
                            "sourceCellCounts": {"small": 10, "large": 20},
                            "unintegrated": {"resources": {"operationPeakBytes": peak * GIB}},
                        }
                    }
                },
            )
        files = [{"bytes": 10 * GIB}, {"bytes": 17 * GIB}]
        for path, peak in zip(prepare.IN_MEMORY_CONCATENATION, (166, 167, 168)):
            write(
                path,
                {
                    "result": {
                        "materializedConcatenation": {
                            "inputFiles": files,
                            "resources": {"operationPeakBytes": peak * GIB},
                        }
                    }
                },
            )
        for path, peak in zip(prepare.ON_DISK_CONCATENATION, (21, 22, 23)):
            write(
                path,
                {
                    "result": {
                        "onDiskConcatenation": {
                            "inputs": files,
                            "concatenation": {
                                "processIoBytes": {"wchar": 83 * GIB},
                                "resources": {"operationPeakBytes": peak * GIB},
                            },
                        }
                    }
                },
            )
        for path, peak in zip(prepare.SCANPY_DASK, (67, 68, 69)):
            write(path, {"status": scanpy_status, "peakCgroupBytes": peak * GIB})
        # BPCells' first run predates the disk sampling; the others record the peak of their work directory.
        for path, peak, disk in zip(prepare.BPCELLS_BLOOD, (29, 30, 31), bpcells_disk):
            record = {
                "status": bpcells_status,
                "download": {"bytes": bpcells_download},
                "peakCgroupBytes": peak * GIB,
            }
            if disk is not None:
                record["peakWorkDirectoryBytes"] = disk
            write(path, record)
        for path in prepare.COVID_COMPOSITE:
            write(
                path,
                {
                    "result": {
                        "composite": {
                            "sourceAlignment": {
                                "countsCompared": 1000,
                                "exactCounts": exact,
                                "copiedCountChunkBytes": 0,
                            },
                            "materializedAgreement": {
                                "clusterAri": 1.0,
                                "exactClusterLabels": True,
                                "pcaMaximumSubspaceDegrees": 1e-5,
                            },
                            "tolerances": {"pcaSubspaceDegrees": 0.05},
                            "cellsTotalSelected": 200,
                            "sourceCellCounts": {"a": 100, "b": 100},
                        }
                    }
                },
            )

    def test_routes_hold_the_disk_and_memory_each_route_needs(self):
        self.write_inputs()
        footprint = prepare.composite_footprint()
        composite, in_memory, on_disk, bpcells = footprint["routes"]
        self.assertEqual(composite["disk"], [{"label": "composite store", "GiB": 1.0}])
        self.assertEqual(composite["memoryGiB"]["mean"], 13.0)
        self.assertEqual(in_memory["disk"], [{"label": "H5AD files", "GiB": 27.0}])
        self.assertEqual(in_memory["memoryGiB"]["values"], [166.0, 167.0, 168.0])
        self.assertEqual(
            on_disk["disk"],
            [
                {"label": "H5AD files", "GiB": 27.0},
                {"label": "written by concatenation", "GiB": 83.0},
            ],
        )
        self.assertEqual(on_disk["memoryGiB"]["mean"], 68.0)
        self.assertEqual(on_disk["concatenationMemoryGiB"]["high"], 23.0)
        # BPCells' disk is its work directory's peak: the H5AD files it downloaded, then what it wrote beside them.
        self.assertEqual(
            bpcells["disk"],
            [
                {"label": "H5AD files", "GiB": 27.0},
                {"label": "written by BPCells", "GiB": 100.0},
            ],
        )
        self.assertEqual(bpcells["diskRuns"], 2)
        self.assertEqual(bpcells["memoryGiB"]["values"], [29.0, 30.0, 31.0])
        self.assertEqual([s["study"] for s in footprint["sources"]], ["large", "small"])
        self.assertEqual(len(footprint["checks"]), 3)
        # The map's studies in the raster's order, with the cohort record's cells.
        self.assertEqual(
            footprint["umap"],
            {
                "raster": "raster.npz",
                "studies": [{"study": "small", "cells": 10}, {"study": "large", "cells": 20}],
            },
        )
        self.assertIn("raster.npz", footprint["source"])
        self.assertIn("cohort.json", footprint["source"])
        # Each analysis's mixing of studies and share of same-type neighbours, from the mixing record.
        self.assertEqual(
            footprint["integration"],
            {
                "unintegrated": {"studyMixing": 0.2, "sameType": 0.97},
                "integrated": {"studyMixing": 0.5, "sameType": 0.96},
            },
        )
        self.assertIn("cohort_mixing.json", footprint["source"])

    def test_a_mixing_record_of_other_studies_or_cells_than_the_run_record_is_an_error(self):
        self.write_inputs(full_cohort=(("small", 10), ("large", 19)))
        with self.assertRaisesRegex(ValueError, "mixing record"):
            prepare.composite_footprint()

    def test_a_raster_whose_cells_per_study_differ_from_the_cohort_record_is_an_error(self):
        self.write_inputs(binned=(20, 9))
        with self.assertRaisesRegex(ValueError, "raster's cells per study"):
            prepare.composite_footprint()

    def test_a_cohort_of_other_studies_or_cells_than_the_composite_runs_is_an_error(self):
        for cohort in ((("small", 10), ("large", 21)), (("small", 10), ("other", 20))):
            with self.subTest(cohort=cohort):
                self.write_inputs(cohort=cohort)
                with self.assertRaisesRegex(ValueError, "other studies or cells"):
                    prepare.composite_footprint()

    def test_composite_runs_that_differ_in_store_bytes_or_copy_counts_are_an_error(self):
        for changes in ({"store_bytes": (GIB, GIB, 2 * GIB)}, {"copied": 1}):
            with self.subTest(**changes):
                self.write_inputs(**changes)
                with self.assertRaisesRegex(ValueError, "copied count chunks"):
                    prepare.composite_footprint()

    def test_a_scanpy_run_that_did_not_complete_is_an_error(self):
        self.write_inputs(scanpy_status="failed")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            prepare.composite_footprint()

    def test_a_bpcells_run_that_did_not_complete_is_an_error(self):
        self.write_inputs(bpcells_status="failed")
        with self.assertRaisesRegex(ValueError, "BPCells run did not complete"):
            prepare.composite_footprint()

    def test_bpcells_runs_that_read_other_files_are_an_error(self):
        self.write_inputs(bpcells_download=26 * GIB)
        with self.assertRaisesRegex(ValueError, "other files than the concatenation"):
            prepare.composite_footprint()

    def test_bpcells_runs_without_disk_are_an_error(self):
        self.write_inputs(bpcells_disk=(None, None, None))
        with self.assertRaisesRegex(ValueError, "recorded its disk"):
            prepare.composite_footprint()

    def test_a_check_run_whose_counts_differ_from_its_sources_is_an_error(self):
        self.write_inputs(exact=False)
        with self.assertRaisesRegex(ValueError, "did not match"):
            prepare.composite_footprint()


class StorageBackendsTest(TemporaryRoot):
    PATHS = {"STORAGE_SWEEP": "sweep.json", "STORAGE_BACKENDS": "backends.json"}

    def sweep_run(
        self, backend, minutes, *, budget=24, workers=8, complete=True, dataset="embryo1M"
    ):
        return {
            "complete": complete,
            "dataset": dataset,
            "storeBackend": backend,
            "budgetGiB": budget,
            "workers": workers,
            "conversionElapsedSeconds": 100.0,
            "elapsedSeconds": 100.0 + 60.0 * minutes,
        }

    def write_inputs(
        self, *, scanpy_status="ok", object_minutes=(9.0, 10.0, 11.0), local_minutes=(6.0, 7.0, 8.0)
    ):
        runs = [self.sweep_run("modalLocalDisk", m) for m in (6.0, 7.0, 8.0)]
        runs += [self.sweep_run("objectStore", m) for m in (9.0, 10.0, 11.0)]
        # Runs on another budget, worker count or dataset, or incomplete, are not the matched ones.
        runs += [
            self.sweep_run("objectStore", 50.0, budget=8),
            self.sweep_run("objectStore", 50.0, workers=4),
            self.sweep_run("objectStore", 50.0, dataset="brain1.3M"),
            self.sweep_run("objectStore", 50.0, complete=False),
        ]
        write(prepare.STORAGE_SWEEP, {"runs": runs})

        def arm(minutes, status="ok"):
            return {"runs": [{"status": status, "analysisMinutes": m} for m in minutes]}

        write(
            prepare.STORAGE_BACKENDS,
            {
                "arms": {
                    "bpcells_cached": arm((12.0, 14.0, 16.0)),
                    "scanpy_dask_covariance_csc_local_zarr": arm((20.0, 30.0, 40.0)),
                    "scanpy_dask_covariance_csc_remote_zarr": arm(
                        (21.0, 25.0, 29.0), scanpy_status
                    ),
                },
                "cytearcObjectStorageAnalysisMinutes": list(object_minutes),
                "cytearcLocalDiskAnalysisMinutes": list(local_minutes),
                "comparisons": {
                    "cytearcOverScanpyObjectStorage": {
                        "ratio": 0.4,
                        "low": 0.25,
                        "high": 0.5,
                        "p": 0.01,
                    },
                    "cytearcOverScanpyLocalDisk": {
                        "ratio": 0.25,
                        "low": 0.125,
                        "high": 0.5,
                        "p": 0.02,
                    },
                    "cytearcOverBpcellsLocalDisk": {
                        "ratio": 0.5,
                        "low": 0.25,
                        "high": 1.0,
                        "p": 0.03,
                    },
                },
            },
        )

    def test_matched_runs_give_each_tool_its_minutes_on_both_backends(self):
        self.write_inputs()
        backends = prepare.storage_backends()
        cytearc, bpcells, scanpy = backends["tools"]
        self.assertEqual(cytearc["localDisk"]["values"], [6.0, 7.0, 8.0])
        self.assertEqual(cytearc["objectStorage"]["mean"], 10.0)
        self.assertEqual(scanpy["objectStorage"]["low"], 21.0)
        # BPCells reads local files only.
        self.assertEqual(bpcells["localDisk"]["mean"], 14.0)
        self.assertIsNone(bpcells["objectStorage"])
        self.assertEqual(
            backends["bpcellsOverCytearc"],
            {"localDisk": {"ratio": 2.0, "low": 1.0, "high": 4.0, "p": 0.03}},
        )
        # Scanpy over CyteArc: the comparison's ratio inverted, the interval's ends inverted and swapped.
        self.assertEqual(
            backends["scanpyOverCytearc"],
            {
                "localDisk": {"ratio": 4.0, "low": 2.0, "high": 8.0, "p": 0.02},
                "objectStorage": {"ratio": 2.5, "low": 2.0, "high": 4.0, "p": 0.01},
            },
        )

    def test_local_disk_runs_other_than_the_comparisons_are_an_error(self):
        self.write_inputs(local_minutes=(6.0, 7.0, 9.0))
        with self.assertRaisesRegex(ValueError, "localDisk runs differ"):
            prepare.storage_backends()

    def test_object_storage_runs_other_than_the_comparisons_are_an_error(self):
        self.write_inputs(object_minutes=(9.0, 10.0, 12.0))
        with self.assertRaisesRegex(ValueError, "comparison used"):
            prepare.storage_backends()

    def test_a_scanpy_run_that_did_not_complete_is_an_error(self):
        self.write_inputs(scanpy_status="failed")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            prepare.storage_backends()


class ProgenitorEffectsTest(TemporaryRoot):
    PATHS = {
        "PROGENITOR_SUMMARY": "summary.json",
        "PROGENITOR_RECORDS": "records",
        "SOURCE_STUDY_VALUES": "records/source_studies.json",
    }

    def test_studies_carry_their_stores_rates_donors_and_ratios_with_intervals(self):
        import pandas as pd

        (self.root / "records").mkdir()
        pd.DataFrame(
            {"storeId": ["s1", "s2", "s3"], "collectionId": ["c1", "c1", "c2"]}
        ).to_parquet(self.root / "records" / "progenitor_cells.parquet")
        effect = {"effect": float(np.log(2.0)), "standardError": 0.5}

        def pooled(ratio: float) -> dict:
            return {
                "effect": float(np.log(ratio)),
                "low": float(np.log(ratio / 1.4)),
                "high": float(np.log(ratio * 1.4)),
                "p": 0.01 * ratio,
                "studies": 2,
                "tau2": 0.0,
            }

        def estimate(ratio: float, *, p: float, studies: int) -> dict:
            return {
                "ratio": ratio,
                "ratioLow": ratio / 2,
                "ratioHigh": ratio * 2,
                "p": p,
                "studies": studies,
            }

        def variant(ratio: float) -> dict:
            return {
                "donors": 7,
                "covid": [
                    {
                        "study": "c1",
                        "cases": 4,
                        "healthy": 6,
                        "casesPer10k": 8.0,
                        "healthyPer10k": 4.0,
                        **effect,
                    },
                    {
                        "study": "c2",
                        "cases": 6,
                        "healthy": 5,
                        "casesPer10k": 3.0,
                        "healthyPer10k": 3.0,
                        "effect": 0.0,
                        "standardError": 0.2,
                    },
                ],
                "covidPooled": pooled(ratio),
                "age": [],
                "agePooled": None,
            }

        write(
            self.root / "summary.json",
            {
                "discovery": {},
                "extraction": {},
                "analysis": {},
                "progenitorCells": 3,
                "sources": {"blood": {"cellsRead": 900}, "enriched": {"cellsRead": 100}},
                "variants": {"default": variant(1.5), "labelsOnly": variant(2.5)},
                "stateCovid": {"HSC/MPP": {"cells": 40, "covid": [], "covidPooled": pooled(1.8)}},
                "stateShares": {
                    "blood": {"HSC/MPP": 0.6, "B-lymphoid": 0.4},
                    "enriched": {"HSC/MPP": 0.7, "B-lymphoid": 0.3},
                },
                "covidSeverity": {
                    "studies": [
                        {
                            "study": "c1",
                            **{
                                name: {
                                    "donors": 5 + i,
                                    "newborns": i,
                                    "cells": 1000,
                                    "progenitors": 2 * i,
                                    "per10k": 20.0 * i,
                                }
                                for i, name in enumerate(("healthy", "mild", "severe"))
                            },
                        }
                    ],
                    "trend": [],
                    "trendPooled": pooled(1.8),
                    "severeVsMild": [],
                    "severeVsMildPooled": None,
                },
                "diseases": {
                    "COVID-19": {
                        "studies": [
                            {"study": "c1", "cases": 4, "healthy": 6},
                            {"study": "c2", "cases": 6, "healthy": 5},
                        ],
                        "estimate": estimate(1.5, p=0.02, studies=2),
                    },
                    "sepsis": {
                        "studies": [{"study": "c1", "cases": 7, "healthy": 6}],
                        "estimate": estimate(5.0, p=0.03, studies=1),
                    },
                    "systemic lupus erythematosus": {
                        "studies": [{"study": "c3", "cases": 9, "healthy": 8}],
                        "estimate": estimate(0.8, p=0.2, studies=1),
                    },
                },
                "combatCohorts": {
                    "cohorts": [
                        {
                            "cohort": "healthy",
                            "donors": [
                                {"cells": 10_000, "progenitors": 30},
                                {"cells": 20_000, "progenitors": 20},
                            ],
                            "per10k": 50 / 3,
                        },
                        {
                            "cohort": "sepsis",
                            "donors": [{"cells": 5_000, "progenitors": 0}],
                            "per10k": 0.0,
                        },
                    ],
                    "tooFewDonors": [{"cohort": "influenza", "donors": 3}],
                    "trend": estimate(2.0, p=1e-9, studies=1),
                    "sepsis": estimate(5.0, p=0.03, studies=1),
                },
                "cohortRecords": {
                    "s1": {
                        "relabelled": {"sepsis": {"samples": 3, "donors": 2}},
                        "excluded": {"samples": 1, "donors": 1},
                    }
                },
                "sourceStudies": {"atlas": {"Chen": {"samples": 2, "donors": 1}}},
            },
        )
        effects = prepare.progenitor_effects()
        self.assertEqual(effects["sources"]["enriched"], {"cellsRead": 100})
        self.assertEqual(effects["markerGenes"], ["CD34", "SPINK2", "PROM1", "CRHBP"])
        # Studies from the lowest ratio up, each with its stores, its two groups' donors and rates, and its ratio.
        first, second = effects["covid"]
        self.assertEqual(first["stores"], ["s3"])
        self.assertEqual(second["stores"], ["s1", "s2"])
        self.assertEqual(
            {
                key: second[key]
                for key in ("healthyDonors", "covidDonors", "healthyPer10k", "covidPer10k")
            },
            {"healthyDonors": 6, "covidDonors": 4, "healthyPer10k": 4.0, "covidPer10k": 8.0},
        )
        self.assertAlmostEqual(second["ratio"], 2.0)
        self.assertAlmostEqual(second["low"], 2.0 * np.exp(-1.959964 * 0.5))
        self.assertAlmostEqual(second["high"], 2.0 * np.exp(1.959964 * 0.5))
        # Each selection's pooled ratio and interval; the states' pools, which no panel draws, stay out.
        self.assertEqual(set(effects["selections"]), {"default", "labelsOnly"})
        self.assertAlmostEqual(effects["selections"]["labelsOnly"]["ratio"], 2.5)
        self.assertAlmostEqual(effects["selections"]["labelsOnly"]["low"], 2.5 / 1.4)
        self.assertNotIn("states", effects)
        self.assertNotIn("covidPooled", effects)
        self.assertAlmostEqual(effects["selections"]["default"]["p"], 0.015)
        # Each source's state shares pass through as the summary gives them.
        self.assertEqual(
            effects["stateShares"],
            {
                "blood": {"HSC/MPP": 0.6, "B-lymphoid": 0.4},
                "enriched": {"HSC/MPP": 0.7, "B-lymphoid": 0.3},
            },
        )
        self.assertNotIn("covidComposition", effects)
        # The severity classes, which no panel draws, stay out.
        self.assertNotIn("severity", effects)
        # Diseases from the highest odds ratio down, each with its donors over its studies and its estimate.
        self.assertEqual(
            [(d["disease"], d["studies"], d["cases"], d["healthy"]) for d in effects["diseases"]],
            [
                ("sepsis", 1, 7, 6),
                ("COVID-19", 2, 10, 11),
                ("systemic lupus erythematosus", 1, 9, 8),
            ],
        )
        self.assertEqual(
            {key: effects["diseases"][0][key] for key in ("ratio", "low", "high", "p")},
            {"ratio": 5.0, "low": 2.5, "high": 10.0, "p": 0.03},
        )
        # Three studies, c1 counted once though two diseases use it.
        self.assertEqual(effects["diseaseStudies"], 3)
        # COMBAT's cohorts: each donor's rate from its own cells, in rising order, beside the cohort's pooled rate.
        healthy, sepsis = effects["cohorts"]["cohorts"]
        self.assertEqual(healthy["cohort"], "healthy")
        self.assertEqual(healthy["donorsPer10k"], [10.0, 30.0])
        self.assertAlmostEqual(healthy["per10k"], 50 / 3)
        self.assertEqual(sepsis["donorsPer10k"], [0.0])
        self.assertEqual(
            effects["cohorts"]["trend"], {"ratio": 2.0, "low": 1.0, "high": 4.0, "p": 1e-9}
        )
        self.assertEqual(effects["cohorts"]["tooFewDonors"], [{"cohort": "influenza", "donors": 3}])
        self.assertEqual(
            effects["cohortRecords"]["s1"]["relabelled"], {"sepsis": {"samples": 3, "donors": 2}}
        )
        self.assertEqual(effects["sourceStudies"], {"atlas": {"Chen": {"samples": 2, "donors": 1}}})
        self.assertEqual(
            effects["source"][-2:], ["records/severity.json", "records/source_studies.json"]
        )


class ProgenitorMapsTest(TemporaryRoot):
    PATHS = {
        "PROGENITOR_SUMMARY": "summary.json",
        "PROGENITOR_RECORDS": "records",
        "PROGENITOR_RASTER": "raster.npz",
    }

    def test_the_raster_counts_every_cell_once_by_state_and_by_source(self):
        import pandas as pd
        from experiments.reuse.progenitor_effects import ENRICHED

        (self.root / "records").mkdir()
        cells = pd.DataFrame(
            {
                "dataset": ["p", "p", ENRICHED[0], "p", ENRICHED[0]],
                "source_row": range(5),
                "umap1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "umap2": [0.0, 1.0, 2.0, 3.0, 4.0],
                "leiden_1_0": [0, 0, 1, 2, 1],
            }
        )
        cells.to_parquet(self.root / "records" / "progenitor_cell_table.parquet")
        write(
            self.root / "summary.json",
            {"clusterStates": {"0": "HSC/MPP", "1": "erythroid", "2": "mature T/NK"}},
        )
        maps = prepare.progenitor_maps()
        self.assertEqual([s["state"] for s in maps["states"]], ["HSC/MPP", "erythroid", "mature"])
        self.assertEqual([s["cells"] for s in maps["states"]], [2, 2, 1])
        self.assertEqual(maps["referenceCells"], 2)
        with np.load(self.root / "raster.npz") as raster:
            self.assertEqual(int(raster["states"].sum()), 5)
            self.assertEqual(raster["sources"].sum(axis=(1, 2)).tolist(), [3, 2])

    def test_the_raster_takes_the_maps_shape_in_square_bins(self):
        import pandas as pd

        (self.root / "records").mkdir()
        # A map four units wide and one tall: as many columns as the grid and a quarter as many rows, the edges inside
        # the last bins.
        pd.DataFrame(
            {
                "dataset": ["p"] * 3,
                "source_row": range(3),
                "umap1": [0.0, 2.0, 4.0],
                "umap2": [0.0, 1.0, 0.5],
                "leiden_1_0": [0, 0, 0],
            }
        ).to_parquet(self.root / "records" / "progenitor_cell_table.parquet")
        write(self.root / "summary.json", {"clusterStates": {"0": "HSC/MPP"}})
        prepare.progenitor_maps()
        with np.load(self.root / "raster.npz") as raster:
            rows, columns = raster["states"].shape[1:]
        self.assertEqual(columns, prepare.PROGENITOR_GRID)
        self.assertEqual(rows, prepare.PROGENITOR_GRID // 4)

    def test_a_cluster_without_a_state_is_an_error(self):
        import pandas as pd

        (self.root / "records").mkdir()
        pd.DataFrame(
            {"dataset": ["p"], "source_row": [0], "umap1": [0.0], "umap2": [0.0], "leiden_1_0": [3]}
        ).to_parquet(self.root / "records" / "progenitor_cell_table.parquet")
        write(self.root / "summary.json", {"clusterStates": {"0": "HSC/MPP"}})
        with self.assertRaisesRegex(ValueError, "no state"):
            prepare.progenitor_maps()


class WeightedMedianTest(unittest.TestCase):
    def test_the_heavy_value_wins_whatever_the_input_order(self):
        self.assertEqual(
            prepare.weighted_median(np.array([5.0, 1.0, 3.0]), np.array([1, 1, 10])), 3.0
        )

    def test_equal_weights_give_the_middle_value(self):
        self.assertEqual(prepare.weighted_median(np.array([3.0, 1.0, 2.0]), np.ones(3)), 2.0)


if __name__ == "__main__":
    unittest.main()
