"""The marker-only comparison: CyteArc's runs read the comparators' common input, and Table 2's marker rows
follow from the sampled statistics that every tool reports."""

import csv
import json
import math
import unittest

import numpy as np
import pandas as pd
from experiments.runtime.common import ROOT

RECORDS = ROOT / "experiments" / "benchmarks" / "records"
TABLE_2 = ROOT / "manuscript" / "tables" / "table_2" / "data.csv"
P_FLOOR = 1e-300
# The comparators of each dataset, with the container memory of their first replicate.
COMPARATORS = {
    "embryo_1000000": {"bpcells": 32, "presto": 64, "scanpy_csc": 64},
    "brain_1306127": {"bpcells": 32, "scanpy_csc": 64},
    "tabula_sapiens": {"bpcells": 32, "scanpy_csc": 128},
}


def sample(dataset: str, tool: str, memory_gib: int) -> pd.DataFrame:
    name = f"marker_statistics_{dataset}_{tool}_cpus_8_replicate_1_memory_{memory_gib}gib.csv"
    return pd.read_csv(RECORDS / "raw" / name, dtype={"group": str, "gene": str})


class CyteArcMarkerRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads((RECORDS / "toolkit_comparison_runs.json").read_text())["records"]

    def test_each_run_read_the_input_the_comparators_read(self):
        for dataset, tools in COMPARATORS.items():
            common = self.records[f"{dataset}_marker_input"]
            for tool, memory in tools.items():
                comparator = self.records[
                    f"marker_statistics_{dataset}_{tool}_cpus_8_replicate_1_memory_{memory}gib"
                ]
                self.assertEqual(comparator["download"]["bytes"], common["bytes"], (dataset, tool))
            for replicate in (1, 2, 3):
                run = self.records[
                    f"marker_statistics_{dataset}_cytearc_cpus_8_replicate_{replicate}_memory_32gib"
                ]
                with self.subTest(dataset=dataset, replicate=replicate):
                    self.assertEqual(run["status"], "ok")
                    self.assertEqual(run["download"]["bytes"], common["bytes"])
                    self.assertEqual(
                        (run["nCells"], run["nGenes"], run["nnz"], run["groups"]),
                        (
                            common["cellsAfterFilter"],
                            common["genesAfterFilter"],
                            common["nnz"],
                            common["groups"],
                        ),
                    )
                    self.assertIs(run["container"]["preemptible"], False)
                    self.assertEqual(run["sampleRows"], 6000)

    def test_runs_without_a_ladder_have_no_ladder_record(self):
        ladders = [
            name
            for name in self.records
            if name.startswith("marker_statistics_")
            and "_cytearc_" in name
            and name.endswith("_memory_tiers")
        ]
        self.assertEqual(ladders, [])


class Table2MarkerRows(unittest.TestCase):
    """Recomputes the largest differences from the sampled tables and compares them with Table 2."""

    @classmethod
    def setUpClass(cls):
        with TABLE_2.open() as handle:
            cls.table = {
                row["metric"]: float(row["value"])
                for row in csv.DictReader(handle)
                if row["metric"].startswith("markers / ")
            }

    def test_sampled_pairs_are_the_same_for_every_tool(self):
        for dataset, tools in COMPARATORS.items():
            cytearc = sample(dataset, "cytearc", 32)
            pairs = set(zip(cytearc["group"], cytearc["gene"]))
            self.assertEqual(len(pairs), 6000)
            for tool, memory in tools.items():
                other = sample(dataset, tool, memory)
                self.assertEqual(set(zip(other["group"], other["gene"])), pairs, (dataset, tool))

    def test_largest_differences_match_the_table(self):
        largest_auroc = 0.0
        largest_log10_p = 0.0
        for dataset, tools in COMPARATORS.items():
            cytearc = sample(dataset, "cytearc", 32)
            for tool, memory in tools.items():
                merged = cytearc.merge(sample(dataset, tool, memory), on=["group", "gene"])
                with self.subTest(dataset=dataset, tool=tool):
                    if tool == "bpcells":
                        p, q = merged["pval_x"].to_numpy(), merged["pval_y"].to_numpy()
                        both = (p >= P_FLOOR) & (q >= P_FLOOR)
                        difference = float(np.max(np.abs(np.log10(p[both]) - np.log10(q[both]))))
                        key = f"markers / {dataset} / pMaxAbsLog10Diff_bpcells"
                        self.assertEqual(
                            int(both.sum()),
                            self.table[f"markers / {dataset} / pPairsAboveFloor_bpcells"],
                        )
                        largest_log10_p = max(largest_log10_p, difference)
                    else:
                        difference = float((merged["auc_x"] - merged["auc_y"]).abs().max())
                        key = f"markers / {dataset} / aurocMaxAbsDiff_{tool}"
                        largest_auroc = max(largest_auroc, difference)
                    self.assertTrue(math.isclose(self.table[key], difference, rel_tol=1e-12), key)
        self.assertTrue(math.isclose(self.table["markers / auroc"], largest_auroc, rel_tol=1e-12))
        self.assertTrue(
            math.isclose(self.table["markers / log10P"], largest_log10_p, rel_tol=1e-12)
        )


if __name__ == "__main__":
    unittest.main()
