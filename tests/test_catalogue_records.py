import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.catalogue import blood_store_screen_summary as screen_summary
from experiments.catalogue import catalogue_objects_summary as objects_summary
from experiments.catalogue import catalogue_screen as screen
from experiments.runtime.common import ROOT, public_run

RECORDS = ROOT / "experiments" / "catalogue" / "records"
PANEL = {"interferon": {"ISG15": "ENSG00000187608"}, "lineage": {"CD3E": "ENSG00000198851"}}


def run_record(store_ids, *, waits=None, region="us-east", status="completed", **settings):
    """A run record as the Modal runner saves it, with the private fields the record must drop."""
    stores = [{"storeId": store_id} for store_id in store_ids[:-1]]
    return {
        "module": "catalogue.catalogue_screen",
        "function": "run",
        "callId": "fc-01ABCDEFGHIJKLMNOPQRSTUVWX",
        "region": region,
        "startedAt": 1793371907.2,
        "status": status,
        "elapsedSeconds": 901.1,
        "result": {
            "settings": {
                "storeConcurrency": 4,
                "nthreads": 8,
                "memBudget": "3G",
                "recordUri": "s3://bucket/experiments/catalogue/run.json",
                "workDirectory": "/tmp/catalogue_screen",
                **settings,
            },
            "authenticated": True,
            "release": {"version": "2610", "catalogueSha256": "ab" * 32, "bucketBytes": 10},
            "panel": PANEL,
            "wallSeconds": 100.5,
            "traffic": {
                "requestsByKind": {"api": 3},
                "quotaWaits": waits if waits is not None else {},
                "quotaPolicies": ["q"],
            },
            "totals": {
                **{key: index + 1 for index, key in enumerate(screen.ROW_TOTALS)},
                "responseShareOfStoreCountBytes": 0.5,
                "storeSecondsSum": 3.0,
            },
            "stores": stores,
            "failures": [{"storeId": store_ids[-1], "errors": []}],
            "scores": {"byDisease": []},
            "labelMap": [],
            "resources": {
                "cgroupMemoryCurrentPeakBytes": 7,
                "memoryMaxBytes": 9,
                "processTreeRssPeakBytes": 8,
            },
            "tables": {"groups": {"uri": "s3://bucket/x.csv.gz"}},
        },
    }


class FakeCatalog:
    def __init__(self, identifiers):
        self.identifiers = identifiers

    def query(self, sql, parameters):
        return [{"cytebase_id": identifier} for identifier in self.identifiers]


class BloodStoreScreenRowTests(unittest.TestCase):
    def test_row_keeps_the_job_and_drops_run_history(self) -> None:
        run = run_record(
            list(screen.IMMUNE_COHORT_STORES),
            waits={"resolvers": {"waits": 5, "seconds": 9.0}, "api": {"waits": 2, "seconds": 1.0}},
        )
        row = screen.screen_row(run)
        self.assertEqual(row["storeSet"], "immune_cohorts")
        self.assertEqual(row["storeIds"], sorted(screen.IMMUNE_COHORT_STORES))
        self.assertEqual(row["rateLimitWaits"], 7)
        self.assertEqual(row["region"], "us-east")
        self.assertEqual((row["cgroupMemoryCurrentPeakBytes"], row["memoryMaxBytes"]), (7, 9))
        self.assertEqual(row["storesFailed"], screen.ROW_TOTALS.index("storesFailed") + 1)
        text = json.dumps(row)
        for private in ("fc-01", "s3://", "/tmp/", "startedAt", "callId", "1793371907"):
            self.assertNotIn(private, text)

    def test_row_without_waits_counts_none(self) -> None:
        row = screen.screen_row(run_record(["store_a", "store_b"]))
        self.assertEqual((row["storeSet"], row["rateLimitWaits"]), ("other_blood_stores", 0))

    def test_row_refuses_an_unfinished_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "status 'failed'"):
            screen.screen_row(run_record(["store_a", "store_b"], status="failed"))

    def test_row_refuses_a_payload_without_its_settings(self) -> None:
        run = run_record(["store_a", "store_b"])
        del run["result"]["settings"]["memBudget"]
        with self.assertRaisesRegex(ValueError, "memBudget"):
            screen.screen_row(run)

    def test_store_sets_are_whole(self) -> None:
        cohort = list(screen.IMMUNE_COHORT_STORES)
        for stores in (cohort[:-1], [*cohort, "store_a"], [cohort[0], "store_a"]):
            with self.subTest(stores=len(stores)), self.assertRaisesRegex(ValueError, "one whole"):
                screen.store_set_of(stores)


class BloodStoreScreenRecordTests(unittest.TestCase):
    def test_collect_orders_jobs_by_store_set(self) -> None:
        cohort = run_record(list(screen.IMMUNE_COHORT_STORES))
        other = run_record(["store_a", "store_b"], region="us-central")
        record = screen.collect([cohort, other])
        self.assertEqual(
            [job["storeSet"] for job in record["jobs"]], ["other_blood_stores", "immune_cohorts"]
        )
        self.assertEqual(record["panel"], PANEL)

    def test_collect_refuses_a_store_set_twice_and_different_panels(self) -> None:
        other = run_record(["store_a", "store_b"])
        with self.assertRaisesRegex(ValueError, "same store set"):
            screen.collect([other, run_record(["store_c", "store_d"])])
        changed = copy.deepcopy(other)
        changed["result"]["panel"] = {"interferon": {}, "lineage": {}}
        with self.assertRaisesRegex(ValueError, "different gene panels"):
            screen.collect([run_record(list(screen.IMMUNE_COHORT_STORES)), changed])

    def test_store_sets_split_the_catalogue(self) -> None:
        catalog = FakeCatalog(["store_a", *screen.IMMUNE_COHORT_STORES, "store_b"])
        cohort = [row["cytebase_id"] for row in screen.eligible_stores(catalog, "immune_cohorts")]
        other = [
            row["cytebase_id"] for row in screen.eligible_stores(catalog, "other_blood_stores")
        ]
        self.assertEqual(cohort, list(screen.IMMUNE_COHORT_STORES))
        self.assertEqual(other, ["store_a", "store_b"])
        with self.assertRaisesRegex(ValueError, "Unknown storeSet"):
            screen.eligible_stores(catalog, "all")
        with self.assertRaisesRegex(ValueError, "Not human blood stores"):
            screen.eligible_stores(FakeCatalog(["store_a"]), "other_blood_stores")

    def test_run_refuses_a_payload_without_a_store_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "storeSet"):
            screen.run({"storeIds": ["store_a"]})

    def test_record_holds_both_jobs_without_run_history(self) -> None:
        record = json.loads((RECORDS / "blood_store_screen.json").read_text())
        jobs = record["jobs"]
        self.assertEqual([job["storeSet"] for job in jobs], list(screen.STORE_SETS))
        self.assertEqual(sorted(jobs[1]["storeIds"]), sorted(screen.IMMUNE_COHORT_STORES))
        self.assertFalse(set(jobs[0]["storeIds"]) & set(jobs[1]["storeIds"]))
        for job in jobs:
            self.assertEqual(job["storesScreened"] + job["storesFailed"], len(job["storeIds"]))
        text = json.dumps(record)
        for private in ("callId", "startedAt", "s3://", "/tmp/", "fc-01"):
            self.assertNotIn(private, text)


class BloodStoreScreenSummaryTests(unittest.TestCase):
    def test_summary_of_the_record_matches_hand_sums(self) -> None:
        summary = screen_summary.summarize(
            json.loads((RECORDS / "blood_store_screen.json").read_text())["jobs"]
        )
        screen_totals = summary["screen"]
        # 67 + 15 stores, 21,315,972 + 24,400,813 cells, 2,939 + 957 donors.
        self.assertEqual(screen_totals["storesScreened"], 82)
        self.assertEqual(screen_totals["cellsInScreenedStores"], 45_716_785)
        self.assertEqual(screen_totals["studyScopedDonors"], 3_896)
        self.assertAlmostEqual(screen_totals["wallSeconds"], 1178.49, places=2)
        self.assertAlmostEqual(screen_totals["wallSecondsPerStore"], 14.372, places=3)
        # (8,290,364,794 + 3,805,816,949) / (128,043,368,596 + 129,566,698,398)
        self.assertAlmostEqual(screen_totals["responseShareOfStoreCountBytes"], 0.046955, places=6)
        shares = [job["responseShareOfStoreCountBytes"] for job in summary["jobs"]]
        self.assertEqual([round(share, 5) for share in shares], [0.06475, 0.02937])

    def test_summary_refuses_a_store_in_two_jobs(self) -> None:
        jobs = json.loads((RECORDS / "blood_store_screen.json").read_text())["jobs"]
        jobs[1]["storeIds"] = [*jobs[1]["storeIds"], jobs[0]["storeIds"][0]]
        with self.assertRaisesRegex(ValueError, "more than one job"):
            screen_summary.summarize(jobs)

    def test_interferon_rerun_totals_add_the_jobs(self) -> None:
        def job(stores: list[str], cells: int, donors: int, seconds: float) -> dict:
            totals = {
                "storesScreened": len(stores),
                "primaryBloodCellsRead": cells,
                "donorsWithScore": donors,
                "responseBytes": 10 * cells,
            }
            return {
                "status": "completed",
                "result": {
                    "stores": [{"storeId": s} for s in stores],
                    "totals": totals,
                    "wallSeconds": seconds,
                },
            }

        totals = screen_summary.interferon_rerun(
            [job(["a", "b"], 100, 7, 30.0), job(["c"], 50, 3, 60.5)]
        )
        self.assertEqual(
            totals,
            {
                "jobs": 2,
                "storesScreened": 3,
                "primaryBloodCellsRead": 150,
                "donorsWithScore": 10,
                "responseBytes": 1500,
                "wallSeconds": 90.5,
            },
        )
        with self.assertRaisesRegex(ValueError, "more than one interferon rerun job"):
            screen_summary.interferon_rerun(
                [job(["a", "b"], 100, 7, 30.0), job(["b"], 50, 3, 60.5)]
            )
        with self.assertRaisesRegex(ValueError, "did not complete"):
            screen_summary.interferon_rerun([{**job(["a"], 100, 7, 30.0), "status": "failed"}])

    def test_interferon_rerun_of_the_records_gives_the_quoted_totals(self) -> None:
        # Paragraph 50: 71 stores, 31.8 million primary blood cells and 3,586 scored donors in 18.1 minutes.
        totals = screen_summary.interferon_rerun(
            [json.loads(path.read_text()) for path in screen_summary.INTERFERON_RUNS]
        )
        self.assertEqual((totals["storesScreened"], totals["donorsWithScore"]), (71, 3_586))
        self.assertEqual(round(totals["primaryBloodCellsRead"] / 1e6, 1), 31.8)
        self.assertEqual(round(totals["wallSeconds"] / 60, 1), 18.1)


class CatalogueObjectsSummaryTests(unittest.TestCase):
    def test_summary_of_two_stores(self) -> None:
        per_store = {
            "a": {
                "nonzeros": 10,
                "bytes": {"countsByCell": 20, "countsByGene": 10, "cellTables": 5},
                "objects": {"countsByCell": 2, "countsByGene": 1, "cellTables": 1},
                "countShardBytes": {"countsByCell": [12, 8], "countsByGene": [10]},
            },
            "b": {
                "nonzeros": 30,
                "bytes": {"countsByCell": 40, "countsByGene": 30, "metadata": 1},
                "objects": {"countsByCell": 1, "countsByGene": 2, "metadata": 1},
                "countShardBytes": {"countsByCell": [40], "countsByGene": [15, 15]},
            },
        }
        summary = objects_summary.summarize(per_store)
        self.assertEqual(
            summary["totalBytesByKind"],
            {"countsByCell": 60, "countsByGene": 40, "cellTables": 5, "metadata": 1},
        )
        self.assertEqual(summary["totalNonzeros"], 40)
        self.assertEqual(summary["countBytesPerNonzeroPooled"], 100 / 40)
        self.assertEqual(summary["bytesPerNonzeroPooled"], 106 / 40)
        self.assertEqual(summary["countShards"]["n"], 6)
        self.assertEqual(summary["countShards"]["bytes"]["median"], 13.5)

    def test_summary_of_the_record_adds_up_to_its_totals(self) -> None:
        record = json.loads((RECORDS / "catalogue_objects.json").read_text())
        summary = objects_summary.summarize(record["perStore"])
        self.assertEqual(sum(summary["totalBytesByKind"].values()), record["totalBytes"])
        self.assertEqual(sum(summary["totalObjectsByKind"].values()), record["totalObjects"])
        self.assertEqual(summary["countShards"]["n"], 30_531)
        self.assertEqual(summary["countShards"]["bytes"]["median"], 30_466_444)
        self.assertNotIn("totalBytesByKind", record)


class KeptJobRecordsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.here = Path(temporary.name)
        # The records go under a temporary root, which write_json_at reports paths against.
        for patch in (
            mock.patch.object(screen, "HERE", self.here),
            mock.patch("experiments.runtime.common.ROOT", self.here),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.table = b"storeId,donor_id\nstore_a,d1\n"
        self.run = run_record(list(screen.IMMUNE_COHORT_STORES), storeSet="immune_cohorts")
        self.run["result"]["tables"]["donors"] = {
            "rows": 1,
            "sha256": hashlib.sha256(self.table).hexdigest(),
            "uri": "s3://bucket/experiments/catalogue/screen/catalogue_screen_donors.csv.gz",
        }

    def test_keep_writes_the_public_record_and_the_donor_table_under_the_store_set(self) -> None:
        downloads = []

        def download_file(uri: str, destination: Path) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(self.table)

        with mock.patch("experiments.runtime.object_io.download_file", download_file):
            path = screen.keep(self.run, "interferon_screen")
            # A table already held with the recorded sha256 is not downloaded again.
            screen.keep(self.run, "interferon_screen")
        folder = self.here / "records" / "interferon_screen"
        self.assertEqual(path, folder / "immune_cohorts.json")
        self.assertEqual((folder / "immune_cohorts_donors.csv.gz").read_bytes(), self.table)
        self.assertEqual(len(downloads), 1)
        record = json.loads(path.read_text())
        text = json.dumps(record)
        for private in ("callId", "startedAt", "s3://", "/tmp/", "fc-01"):
            self.assertNotIn(private, text)
        self.assertEqual(record["region"], "us-east")
        self.assertEqual(
            record["result"]["tables"]["donors"]["sha256"], hashlib.sha256(self.table).hexdigest()
        )

    def test_keep_refuses_a_donor_table_other_than_the_jobs(self) -> None:
        def download_file(uri: str, destination: Path) -> None:
            Path(destination).write_bytes(b"other rows\n")

        with mock.patch("experiments.runtime.object_io.download_file", download_file):
            with self.assertRaisesRegex(ValueError, "not the donor table the job recorded"):
                screen.keep(self.run, "blood_store_screen")
        self.assertFalse(
            (self.here / "records" / "blood_store_screen" / "immune_cohorts.json").exists()
        )

    def test_keep_refuses_an_unfinished_job_and_an_unknown_screen(self) -> None:
        with self.assertRaisesRegex(ValueError, "screen is one of"):
            screen.keep(self.run, "pilot")
        self.run["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "status 'failed'"):
            screen.keep(self.run, "interferon_screen")

    def test_kept_records_are_public_and_what_keep_writes(self) -> None:
        for name in ("blood_store_screen", "interferon_screen"):
            for store_set in screen.STORE_SETS:
                path = RECORDS / name / f"{store_set}.json"
                with self.subTest(path=path.relative_to(RECORDS)):
                    stored = json.loads(path.read_text())
                    self.assertEqual(public_run(copy.deepcopy(stored)), stored)
                    self.assertEqual(screen.job_store_set(stored), store_set)
                    text = json.dumps(stored)
                    for private in ("callId", "startedAt", "s3://", "/tmp/", "fc-01"):
                        self.assertNotIn(private, text)
                    table = path.with_name(f"{store_set}_donors.csv.gz")
                    self.assertEqual(
                        hashlib.sha256(table.read_bytes()).hexdigest(),
                        stored["result"]["tables"]["donors"]["sha256"],
                    )


if __name__ == "__main__":
    unittest.main()
