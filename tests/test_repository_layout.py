import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from experiments.runtime import common, records
from experiments.runtime.collect_results import collect
from experiments.runtime.method_checks import run as run_method_check

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


class LayeringTests(unittest.TestCase):
    def test_experiments_never_import_manuscript(self) -> None:
        offending = []
        for path in sorted(EXPERIMENTS.rglob("*.py")):
            if {"inputs", "workspace", "results"} & set(path.relative_to(EXPERIMENTS).parts):
                continue
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                if any(name == "manuscript" or name.startswith("manuscript.") for name in names):
                    offending.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offending, [])

    def test_no_measurements_package_remains(self) -> None:
        self.assertFalse((ROOT / "measurements").exists())


class RecordsGuardTests(unittest.TestCase):
    def test_paths_inside_experiment_records(self) -> None:
        for path in (
            EXPERIMENTS / "lung" / "records",
            EXPERIMENTS / "lung" / "records" / "lung_cohort_design.json",
            EXPERIMENTS / "mapping" / "records" / "cross_study" / "blood" / "symphony.json",
            # A path that only reaches the records after resolving "..".
            EXPERIMENTS / "lung" / "results" / ".." / "records" / "new.json",
        ):
            with self.subTest(path=path):
                self.assertTrue(common.is_in_records(path))

    def test_paths_outside_experiment_records(self) -> None:
        for path in (
            EXPERIMENTS / "lung" / "results" / "lung_labbaseline.json",
            common.DERIVED / "lung" / "lung_labbaseline.json",
            # A top-level folder named records is not an experiment's records.
            EXPERIMENTS / "records" / "x.json",
            EXPERIMENTS / "lung" / "results" / "records" / "x.json",
            ROOT / "manuscript" / "figures" / "figure_1" / "records" / "x.json",
            EXPERIMENTS / "lung" / "records" / ".." / ".." / ".." / "manuscript",
        ):
            with self.subTest(path=path):
                self.assertFalse(common.is_in_records(path))

    def test_collection_refuses_a_records_destination(self) -> None:
        with mock.patch("experiments.runtime.collect_results.list_objects") as listing:
            with self.assertRaisesRegex(ValueError, "records contain frozen evidence"):
                collect("s3://bucket/prefix", EXPERIMENTS / "methods" / "records")
        listing.assert_not_called()

    def test_method_checks_refuse_a_records_output(self) -> None:
        payload = {
            "module": "methods.aucell",
            "output": str(EXPERIMENTS / "methods" / "records"),
        }
        with self.assertRaisesRegex(ValueError, "must not overwrite experiment records"):
            run_method_check(payload)


class WriteHelperTests(unittest.TestCase):
    def test_write_json_at_reports_the_repository_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with mock.patch.object(common, "ROOT", root), redirect_stdout(output):
                path = common.write_json_at(root / "a" / "b.json", {"x": Path("p"), "n": 1})
            self.assertEqual(path.read_text(), '{\n  "x": "p",\n  "n": 1\n}\n')
            self.assertEqual(output.getvalue(), "wrote a/b.json\n")

    def test_write_csv_at_keeps_only_the_named_columns(self) -> None:
        rows = [{"a": 1, "b": 2, "extra": 3}, {"a": 4}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(common, "ROOT", root), redirect_stdout(io.StringIO()):
                path = common.write_csv_at(root / "t.csv", rows, fieldnames=["a", "b"])
            self.assertEqual(path.read_text(), "a,b\n1,2\n4,\n")

    def test_write_csv_into_a_run_directory_refuses_unnamed_columns(self) -> None:
        rows = [{"a": 1, "extra": 3}]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                common.write_csv("t.csv", rows, directory=Path(directory), fieldnames=["a"])
            path = common.write_json("r.json", {"k": [1]}, directory=Path(directory), subdir="s")
            self.assertEqual(json.loads(path.read_text()), {"k": [1]})
            self.assertEqual(path, Path(directory) / "s" / "r.json")


class RunRecordTests(unittest.TestCase):
    def test_result_bundles_are_found_in_their_experiment_records(self) -> None:
        for path in records.RESULT_BUNDLES:
            with self.subTest(path=path):
                self.assertTrue(path.is_relative_to(EXPERIMENTS))
                self.assertIn("results", json.loads(path.read_text()))
        names = [name for name, _ in records.modal_results()]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("pancreas_larry_fate_prediction", names)

    def test_arrays_are_found_for_each_family(self) -> None:
        for name, file in (("pancreas_larry_fate_prediction", "larry_clonal_fate_cells.npz"),):
            with self.subTest(name=name):
                self.assertTrue(records.modal_array(name, file).is_file())
        with self.assertRaisesRegex(ValueError, "No array family"):
            records.modal_array("unknown", "x.npz")


if __name__ == "__main__":
    unittest.main()
