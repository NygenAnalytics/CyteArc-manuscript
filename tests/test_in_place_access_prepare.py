import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from manuscript.figures.in_place_access import prepare

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


class BloodScreenTest(TemporaryRoot):
    PATHS = {
        "BLOOD_SCREEN": "screen.json",
        "SCREEN_SUMMARY": "summary.json",
        "WORKFLOW_RUNS": "workflow.json",
    }

    def setUp(self):
        super().setUp()
        runs = (self.root / "first.json", self.root / "second.json")
        patch = mock.patch.object(prepare, "SCREEN_RUNS", runs)
        patch.start()
        self.addCleanup(patch.stop)
        self.runs = runs

    def store(self, name, cells, count_bytes, received, waited=0.0):
        return {
            "storeId": name,
            "cells": cells,
            "countBytes": count_bytes,
            "io": {"http": {"responseBytes": received}, "quotaWaitSeconds": waited},
            "seconds": 10.0,
        }

    def write_inputs(self, *, second_received=40, downloads=True):
        jobs = [
            {
                "storeSet": "first",
                "storeIds": ["a", "b"],
                "cellsInScreenedStores": 300,
                "storeCountBytes": 3000,
                "responseBytes": 150,
            },
            {
                "storeSet": "second",
                "storeIds": ["c"],
                "cellsInScreenedStores": 50,
                "storeCountBytes": 1000,
                "responseBytes": 40,
            },
        ]
        write(prepare.BLOOD_SCREEN, {"jobs": jobs})
        first = [self.store("b", 200, 2000, 100, waited=120.0), self.store("a", 100, 1000, 50)]
        write(self.runs[0], {"result": {"stores": first}})
        write(self.runs[1], {"result": {"stores": [self.store("c", 50, 1000, second_received)]}})
        write(
            prepare.SCREEN_SUMMARY,
            {
                "screen": {
                    "storesScreened": 3,
                    "cellsInScreenedStores": 350,
                    "studyScopedDonors": 12,
                    "wallSeconds": 30.0,
                    "responseBytes": 190,
                    "storeCountBytes": 4000,
                    "responseShareOfStoreCountBytes": 0.0475,
                }
            },
        )
        inputs = {"embryo_1000000": {"nCells": 1_000_000}}
        if downloads:
            inputs["tabula_sapiens"] = {
                "nCells": 1000,
                "sourceBytes": 4_000,
                "downloadSeconds": 20.0,
            }
        write(prepare.WORKFLOW_RUNS, {"inputs": inputs})

    def test_rows_hold_each_store_of_both_jobs_apart_from_those_read_during_quota_waits(self):
        self.write_inputs()
        screen = prepare.blood_screen()
        rows = {row["store"]: row for row in screen["stores"]}
        # Store b was read while its job waited on the request quota, so its seconds measure the wait: it is kept
        # apart from the stores the panel draws, with its bytes.
        self.assertEqual(sorted(rows), ["a", "c"])
        self.assertEqual(
            [(row["store"], row["responseBytes"]) for row in screen["rateLimitedStores"]],
            [("b", 100)],
        )
        self.assertEqual(rows["c"]["job"], "second")
        self.assertEqual(screen["totals"]["share"], 0.0475)
        # Inputs that were not downloaded, such as the embryo samples, are left out.
        self.assertEqual(
            screen["downloads"],
            [{"dataset": "tabula_sapiens", "cells": 1000, "bytes": 4_000, "seconds": 20.0}],
        )
        self.assertNotIn("tabulaQuery", screen)

    def test_stores_whose_bytes_do_not_add_up_to_their_job_are_an_error(self):
        self.write_inputs(second_received=41)
        with self.assertRaisesRegex(ValueError, "responseBytes"):
            prepare.blood_screen()

    def test_raw_results_listing_other_stores_than_the_job_are_an_error(self):
        self.write_inputs()
        write(self.runs[1], {"result": {"stores": [self.store("d", 50, 1000, 40)]}})
        with self.assertRaisesRegex(ValueError, "other stores"):
            prepare.blood_screen()

    def test_a_record_without_downloads_is_an_error(self):
        self.write_inputs(downloads=False)
        with self.assertRaisesRegex(ValueError, "no download"):
            prepare.blood_screen()


class ReleaseStorageTest(TemporaryRoot):
    PATHS = {"CATALOGUE_OBJECTS": "objects.json", "CYTEBASE_CATALOGUE": "catalogue.json"}

    def write_inputs(self, *, total=1_400, stores_counted=2, observations=50):
        stores = {
            "a": {
                "cells": 20,
                "nonzeros": 100,
                "bytes": {"countsByCell": 160, "countsByGene": 140, "cellTables": 50},
            },
            "b": {
                "cells": 30,
                "nonzeros": 300,
                "bytes": {"countsByCell": 500, "countsByGene": 400, "artifacts": 150},
            },
        }
        write(prepare.CATALOGUE_OBJECTS, {"perStore": stores, "totalBytes": total})
        write(
            prepare.CYTEBASE_CATALOGUE,
            {"datasets": {"stores": stores_counted, "observations": observations}},
        )

    def test_each_store_takes_all_its_parts_and_the_totals_add_them_up(self):
        self.write_inputs()
        storage = prepare.release_storage()
        # Every part counts toward a store's size, tables and published results included: 350 and 1,050 bytes.
        self.assertEqual(
            [(s["nonzeros"], s["bytes"]) for s in storage["stores"]], [(100, 350), (300, 1_050)]
        )
        self.assertEqual(
            (
                storage["releaseStores"],
                storage["releaseObservations"],
                storage["releaseNonzeros"],
                storage["releaseBytes"],
            ),
            (2, 50, 400, 1_400),
        )

    def test_stores_whose_bytes_do_not_add_up_to_the_release_are_an_error(self):
        self.write_inputs(total=1_300)
        with self.assertRaisesRegex(ValueError, "release's total"):
            prepare.release_storage()

    def test_a_catalogue_of_other_stores_than_the_listing_is_an_error(self):
        self.write_inputs(stores_counted=3)
        with self.assertRaisesRegex(ValueError, "different numbers of stores"):
            prepare.release_storage()

    def test_stores_whose_cells_do_not_add_up_to_the_catalogue_are_an_error(self):
        self.write_inputs(observations=51)
        with self.assertRaisesRegex(ValueError, "observations"):
            prepare.release_storage()
