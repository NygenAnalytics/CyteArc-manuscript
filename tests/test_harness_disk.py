import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.runtime import harness


def write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


class DirectoryBytes(unittest.TestCase):
    def test_sums_nested_files_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            write_bytes(Path(root) / "a.bin", 1000)
            write_bytes(Path(root) / "counts" / "val", 12345)
            write_bytes(Path(root) / "counts" / "deep" / "idx", 7)
            self.assertEqual(harness.directory_bytes(Path(root)), 1000 + 12345 + 7)

    def test_empty_directory_has_no_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(harness.directory_bytes(Path(root)), 0)

    def test_skips_a_file_deleted_during_the_walk(self):
        with tempfile.TemporaryDirectory() as root:
            write_bytes(Path(root) / "kept", 40)
            listed = [(root, [], ["kept", "deleted_meanwhile"])]
            with mock.patch.object(harness.os, "walk", return_value=listed):
                self.assertEqual(harness.directory_bytes(Path(root)), 40)


class SamplerDisk(unittest.TestCase):
    def test_peak_keeps_the_largest_sample_after_files_are_deleted(self):
        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch.object(harness, "DISK_SAMPLE_SECONDS", 0.0),
        ):
            sampler = harness.Sampler(disk_root=Path(root))
            write_bytes(Path(root) / "input.h5ad", 5000)
            sampler.sample()
            (Path(root) / "input.h5ad").unlink()
            write_bytes(Path(root) / "counts", 300)
            sampler.sample()
            self.assertEqual(sampler.peak_disk, 5000)
            self.assertEqual([size for _, size in sampler.disk_series], [5000, 300])

    def test_walks_no_more_often_than_the_disk_interval(self):
        with tempfile.TemporaryDirectory() as root:
            sampler = harness.Sampler(disk_root=Path(root))
            sampler.sample()
            sampler.sample()
            self.assertEqual(len(sampler.disk_series), 1)

    def test_without_a_directory_it_records_no_disk(self):
        sampler = harness.Sampler()
        sampler.sample()
        self.assertEqual(sampler.peak_disk, 0)
        self.assertEqual(sampler.disk_series, [])


# The child writes a large input, holds it while the harness samples, deletes it and keeps a small output, the way the
# BPCells workflow deletes its H5AD inputs after converting them.
CHILD = """
import sys, time
from pathlib import Path
work = Path(sys.argv[4])
big = work / "input.h5ad"
big.write_bytes(b"x" * 1_000_000)
time.sleep(1.0)
big.unlink()
(work / "counts").write_bytes(b"x" * 10)
"""


class RunRecordsDisk(unittest.TestCase):
    def test_record_holds_the_transient_peak_and_the_kept_bytes(self):
        with (
            tempfile.TemporaryDirectory() as work,
            mock.patch.object(harness, "R2") as r2_class,
            mock.patch.object(harness, "DISK_SAMPLE_SECONDS", 0.0),
        ):
            r2 = r2_class.return_value
            r2.exists.return_value = False
            payload = {
                "tool": "child",
                "dataset": "toy",
                "datasetUri": "s3://bucket/toy",
                "inputMode": "remote",
                "outputUri": "s3://bucket/record.json",
                "r2EndpointUrl": "https://example.invalid",
                "workDir": work,
                "containerCpu": 1,
                "containerMemoryMb": 1024,
                "workers": 1,
                "memorySampleSeconds": 0.05,
            }
            outcome = harness.Run(payload).execute([sys.executable, "-I", "-c", CHILD])
            self.assertEqual(outcome["status"], "ok")
            record = r2.put_json.call_args_list[-1].args[1]
            self.assertGreaterEqual(record["peakWorkDirectoryBytes"], 1_000_000)
            kept = sum(path.stat().st_size for path in Path(work).rglob("*") if path.is_file())
            self.assertEqual(record["workDirectoryBytesAfter"], kept)
            self.assertLess(record["workDirectoryBytesAfter"], 1_000_000)
            self.assertTrue(os.path.exists(Path(work) / "counts"))
            self.assertTrue(record["diskSeries"])


if __name__ == "__main__":
    unittest.main()
