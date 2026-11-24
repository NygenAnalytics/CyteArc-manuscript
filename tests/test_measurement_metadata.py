import re
import subprocess
import unittest
from pathlib import Path

from experiments.benchmarks.collect_tool_runs import scrubber

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_LABEL = re.compile(
    r"(?i)(?<![a-z0-9])(?:rc\d+[a-z0-9_]*|exp\d+[a-z0-9_]*|tools-[a-z0-9_-]*\d{8})"
    r"|\d+\.\d+\.\d+(?:rc|\.dev)\d+"
)
GENE_SYMBOLS = {"RC3H1", "RC3H2"}
TEXT_SUFFIXES = {".py", ".json", ".csv", ".toml", ".tex", ".bib", ".txt", ".md", ".R"}


class MeasurementMetadataTests(unittest.TestCase):
    def test_collector_discards_campaign_tags_without_changing_measurements(self) -> None:
        collect = scrubber(
            {"resultsUri": "s3://example/results", "datasetPrefixUri": "s3://example/datasets"}
        )
        measurement = {
            "runTag": "rna_workflow",
            "seconds": 2.5,
            "payload": {"runTag": "rna_workflow", "workers": 4},
            "steps": [{"runTag": "rna_workflow", "peakBytes": 4096}],
        }
        self.assertEqual(
            collect(measurement),
            {"seconds": 2.5, "payload": {"workers": 4}, "steps": [{"peakBytes": 4096}]},
        )
        self.assertEqual(measurement["payload"]["runTag"], "rna_workflow")

    def test_project_labels_describe_experiments(self) -> None:
        names = subprocess.check_output(
            [
                "git",
                "--no-optional-locks",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "experiments",
                "measurements",
                "manuscript",
                "publication",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
        failures = []
        for name in set(names):
            path = ROOT / name
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            for match in HISTORICAL_LABEL.finditer(path.read_text()):
                if match.group() not in GENE_SYMBOLS:
                    failures.append(f"{name}: {match.group()}")
        self.assertEqual(failures, [], "\n".join(sorted(failures)[:20]))


if __name__ == "__main__":
    unittest.main()
