"""Reruns of a toolkit tool taking the place of its records in the comparison."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from experiments.benchmarks import collect_tool_runs as collect

INPUT_BYTES = 19_452_407_866


def name(tool: str, replicate: int, memory: str = "32gib") -> str:
    return f"marker_statistics_embryo_1000000_{tool}_cpus_8_replicate_{replicate}_memory_{memory}"


def run(tool: str, replicate: int, **changes) -> dict:
    """A marker-only run record as collect_tool_runs keeps it."""
    record: dict[str, Any] = {
        "tool": f"markers_{tool}",
        "dataset": "embryo_1000000",
        "replicate": replicate,
        "container": {"cpu": 8.0, "memoryMb": 32768, "workers": 8, "preemptible": False},
        "status": "ok",
        "failedStep": None,
        "restarts": 0,
        "interruptions": [],
        "download": {"seconds": 60.0, "bytes": INPUT_BYTES},
        "workflowSeconds": 240.0,
        "nCells": 954823,
        "nGenes": 32574,
        "nnz": 1613092148,
        "groups": 39,
        "sampleRows": 6000,
        "sourceRecordId": name(tool, replicate),
    }
    for key, value in changes.items():
        if key == "preemptible":
            record["container"] = {**record["container"], "preemptible": value}
        else:
            record[key] = value
    return record


def ladder(tool: str, replicate: int) -> dict:
    return {
        "tool": f"markers_{tool}",
        "dataset": "embryo_1000000",
        "cpu": 8.0,
        "tiers": [32],
        "attempts": [{"memoryGib": 32, "status": "ok", "recordId": name(tool, replicate)}],
        "status": "ok",
    }


def comparison() -> dict:
    """A comparison with an input record, a workflow run, a comparator and two pre-release
    CyteArc marker runs with their ladders."""
    records = {
        "embryo_1000000_marker_input": {
            "experiment": "markersCommon",
            "dataset": "embryo_1000000",
            "bytes": INPUT_BYTES,
        },
        "rna_workflow_embryo_1000000_scanpy_cpus_8_replicate_1_memory_64gib": {
            "tool": "scanpy",
            "dataset": "embryo_1000000",
            "container": {"cpu": 8.0, "memoryMb": 65536, "preemptible": False},
            "download": {"bytes": 4024212359},
            "status": "ok",
        },
    }
    for replicate in (1, 2):
        records[name("bpcells", replicate)] = run("bpcells", replicate)
        records[name("cytearc", replicate)] = run("cytearc", replicate, preemptible=True)
        records[name("cytearc", replicate, "tiers")] = ladder("cytearc", replicate)
    records["marker_kernel_validation_embryo_1000000_common"] = {
        "experiment": "markerKernelValidation",
        "dataset": "embryo_1000000",
        "download": {"bytes": 1},
    }
    return {"provenance": {"capturedAt": "2026-10-06T17:12:43+00:00"}, "records": records}


def reruns(**changes) -> dict:
    series = {"memorySeries": [[0.0, 1, 2, 3]], "diskSeries": [[0.0, 4]]}
    return {
        name("cytearc", replicate): run(
            "cytearc", replicate, workflowSeconds=170.0 + replicate, **series, **changes
        )
        for replicate in (1, 2)
    }


class ReplacedRuns(unittest.TestCase):
    def test_reruns_take_the_place_of_the_records_without_series(self):
        result = collect.replaced_runs(comparison(), reruns())["records"]
        for replicate in (1, 2):
            record = result[name("cytearc", replicate)]
            self.assertEqual(record["workflowSeconds"], 170.0 + replicate)
            self.assertIs(record["container"]["preemptible"], False)
            self.assertNotIn("memorySeries", record)
            self.assertNotIn("diskSeries", record)

    def test_ladders_of_the_replaced_runs_are_dropped(self):
        result = collect.replaced_runs(comparison(), reruns())["records"]
        self.assertNotIn(name("cytearc", 1, "tiers"), result)
        self.assertNotIn(name("cytearc", 2, "tiers"), result)

    def test_other_records_stay_byte_identical_and_in_order(self):
        document = comparison()
        before = json.dumps(document)
        result = collect.replaced_runs(document, reruns())
        dropped = {name("cytearc", 1, "tiers"), name("cytearc", 2, "tiers")}
        self.assertEqual(
            list(result["records"]), [n for n in document["records"] if n not in dropped]
        )
        for key, record in json.loads(before)["records"].items():
            if key in dropped or key in reruns():
                continue
            self.assertEqual(json.dumps(result["records"][key]), json.dumps(record), key)
        self.assertEqual(result["provenance"], document["provenance"])
        self.assertEqual(json.dumps(document), before, "the input document must not change")

    def test_a_rerun_that_read_other_bytes_is_refused(self):
        runs = reruns()
        runs[name("cytearc", 2)]["download"] = {"seconds": 60.0, "bytes": INPUT_BYTES - 1}
        with self.assertRaisesRegex(ValueError, "read 19452407865 bytes"):
            collect.replaced_runs(comparison(), runs)

    def test_a_preemptible_rerun_is_refused(self):
        runs = reruns()
        runs[name("cytearc", 1)] = run("cytearc", 1, preemptible=True)
        with self.assertRaisesRegex(ValueError, "non-preemptible"):
            collect.replaced_runs(comparison(), runs)

    def test_a_missing_rerun_is_refused(self):
        runs = reruns()
        del runs[name("cytearc", 2)]
        with self.assertRaisesRegex(ValueError, "No rerun of"):
            collect.replaced_runs(comparison(), runs)

    def test_a_rerun_of_a_record_the_comparison_lacks_is_refused(self):
        runs = reruns()
        runs[name("cytearc", 3)] = run("cytearc", 3)
        with self.assertRaisesRegex(ValueError, "has no"):
            collect.replaced_runs(comparison(), runs)

    def test_a_failed_or_restarted_rerun_is_refused(self):
        for changes, message in (
            ({"status": "outOfMemory", "failedStep": "convertMatrix"}, "did not complete"),
            ({"restarts": 1}, "restarted"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                collect.replaced_runs(comparison(), reruns(**changes))

    def test_a_rerun_of_another_input_size_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nCells"):
            collect.replaced_runs(comparison(), reruns(nCells=954822))

    def test_a_ladder_with_attempts_no_rerun_replaces_keeps_them_and_takes_the_reruns_outcome(self):
        # Replicate 1 was killed at 16 GiB before it completed at 32 GiB; only the 32 GiB run is rerun.
        document = comparison()
        killed = {
            "memoryGib": 16,
            "status": "killed",
            "failedStep": "runLeiden",
            "recordId": name("cytearc", 1, "16gib"),
        }
        document["records"][name("cytearc", 1, "tiers")]["attempts"].insert(0, killed)
        document["records"][name("cytearc", 1, "tiers")]["attempts"][1]["workflowSeconds"] = 999.0
        result = collect.replaced_runs(document, reruns())["records"]
        attempts = result[name("cytearc", 1, "tiers")]["attempts"]
        self.assertEqual(attempts[0], killed)
        self.assertEqual(
            attempts[1],
            {
                "memoryGib": 32,
                "recordId": name("cytearc", 1),
                "status": "ok",
                "failedStep": None,
                "workflowSeconds": 171.0,
                "peakCgroupBytes": None,
            },
        )
        # Replicate 2's ladder tried only the rerun tier, so it goes.
        self.assertNotIn(name("cytearc", 2, "tiers"), result)

    def test_an_interrupted_attempt_at_the_rerun_tier_needs_no_rerun_and_stays(self):
        document = comparison()
        interrupted = run("cytearc", 3, status="interrupted", failedStep=None)
        document["records"][name("cytearc", 3)] = interrupted
        result = collect.replaced_runs(document, reruns())["records"]
        self.assertEqual(result[name("cytearc", 3)], interrupted)

    def test_a_failed_attempt_at_the_rerun_tier_still_needs_a_rerun(self):
        # A run that failed at the tier measured the tool there, unlike one that was preempted.
        for status in ("killed", "outOfMemory", "timedOut"):
            document = comparison()
            document["records"][name("cytearc", 3)] = run("cytearc", 3, status=status)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "No rerun of"):
                collect.replaced_runs(document, reruns())

    def test_the_comparison_given_is_not_changed(self):
        document = comparison()
        copied = copy.deepcopy(document)
        collect.replaced_runs(document, reruns())
        self.assertEqual(document, copied)


class ReplacedDocuments(unittest.TestCase):
    """Reruns replacing records split across two files, as the comparison and its added replicates are."""

    def documents(self) -> dict:
        # The first file holds replicate 1 and the other tools; the second holds replicate 2 alone.
        first, second = comparison(), comparison()
        del first["records"][name("cytearc", 2)], first["records"][name("cytearc", 2, "tiers")]
        second["records"] = {
            key: second["records"][key] for key in (name("cytearc", 2), name("cytearc", 2, "tiers"))
        }
        return {Path("first.json"): first, Path("second.json"): second}

    def test_each_file_takes_the_reruns_of_its_own_records(self):
        result = collect.replaced_documents(self.documents(), reruns())
        self.assertEqual(set(result), {Path("first.json"), Path("second.json")})
        self.assertEqual(
            result[Path("first.json")]["records"][name("cytearc", 1)]["workflowSeconds"], 171.0
        )
        self.assertEqual(
            result[Path("second.json")]["records"][name("cytearc", 2)]["workflowSeconds"], 172.0
        )
        self.assertNotIn(name("cytearc", 2), result[Path("first.json")]["records"])

    def test_the_input_check_reads_the_other_tools_in_every_file(self):
        # The second file holds no other tool, so its rerun is checked against the first file's.
        runs = reruns()
        runs[name("cytearc", 2)]["download"] = {"seconds": 60.0, "bytes": INPUT_BYTES - 1}
        with self.assertRaisesRegex(ValueError, "read 19452407865 bytes"):
            collect.replaced_documents(self.documents(), runs)

    def test_a_rerun_held_by_no_file_or_by_two_is_refused(self):
        documents = self.documents()
        runs = reruns()
        runs[name("cytearc", 3)] = run("cytearc", 3)
        with self.assertRaisesRegex(ValueError, "exactly one file"):
            collect.replaced_documents(documents, runs)
        documents[Path("second.json")]["records"][name("cytearc", 1)] = run("cytearc", 1)
        with self.assertRaisesRegex(ValueError, "exactly one file"):
            collect.replaced_documents(documents, reruns())

    def test_a_file_without_reruns_is_left_out(self):
        documents = self.documents()
        documents[Path("third.json")] = {
            "records": {"other": {"tool": "scanpy", "dataset": "brain"}}
        }
        self.assertNotIn(Path("third.json"), collect.replaced_documents(documents, reruns()))


class ReplaceIn(unittest.TestCase):
    """replace_in writing the reruns into the records files themselves, with the bucket stubbed out."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.paths = []
        for file, document in ReplacedDocuments().documents().items():
            path = self.root / file.name
            path.write_text(json.dumps(document, indent=2) + "\n")
            self.paths.append(path)
        self.untouched = self.root / "third.json"
        self.untouched.write_text('{"records": {"other": {"tool": "scanpy"}}}\n')
        for target, value in (
            ("run_records", lambda tag: (reruns(), {})),
            ("client_and_prefix", lambda tag: (None, "bucket", "prefix/", None)),
        ):
            patcher = mock.patch.object(collect, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_without_a_destination_each_file_is_rewritten_in_place(self):
        before = self.untouched.read_bytes()
        collect.replace_in("tag", [*self.paths, self.untouched])
        first, second = (json.loads(path.read_text())["records"] for path in self.paths)
        self.assertEqual(first[name("cytearc", 1)]["workflowSeconds"], 171.0)
        self.assertEqual(second[name("cytearc", 2)]["workflowSeconds"], 172.0)
        self.assertNotIn("memorySeries", second[name("cytearc", 2)])
        self.assertEqual(self.untouched.read_bytes(), before)
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), ["first.json", "second.json", "third.json"]
        )

    def test_a_destination_takes_a_single_file(self):
        with self.assertRaisesRegex(ValueError, "single --replace-in file"):
            collect.replace_in("tag", self.paths, self.root / "out.json")


if __name__ == "__main__":
    unittest.main()
