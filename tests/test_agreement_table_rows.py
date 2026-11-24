"""Table 2's rows as rendered from the stored table data, against the records each row reports."""

import unittest

from experiments.runtime.common import ROOT, load_json
from manuscript.tables.biology import agreement_table
from manuscript.tables.render import read_metrics

METHODS = ROOT / "experiments" / "methods" / "records"


def row(name: str) -> str:
    lines = agreement_table(read_metrics("table_2/data")).splitlines()
    matches = [line for line in lines if line.strip().startswith(name + " &")]
    if len(matches) != 1:
        raise AssertionError(f"Table 2 has {len(matches)} rows named {name!r}, not one")
    return matches[0]


class StoredAgreementRowsTests(unittest.TestCase):
    def test_population_balance_row_counts_the_cells_of_its_own_check(self):
        cells = load_json(METHODS / "pancreas_population_balance_agreement.json")["dataset"][
            "cells"
        ]
        self.assertEqual(cells, 3546)
        self.assertIn("3,546 pancreatic cells; same 29 modes", row("Population-balance pseudotime"))

    def test_population_balance_row_reports_the_thirty_mode_difference(self):
        self.assertIn("$3.187 \\times 10^{-7}$", row("Population-balance pseudotime"))

    def test_magic_row_reports_the_cells_and_difference_of_note_104(self):
        text = row("MAGIC imputation")
        self.assertIn("4,331 PBMCs; 200 genes", text)
        self.assertIn("$6.7 \\times 10^{-6}$", text)

    def test_aucell_row_counts_the_cells_and_gene_sets_of_its_record(self):
        record = load_json(METHODS / "aucell.json")
        self.assertEqual(
            (record["dataset"]["cells"], len(record["network"]["sources"])), (4331, 18)
        )
        self.assertIn("4,331 cells; 18 gene sets", row("AUCell; WAGGR mean"))

    def test_cell_cycle_maximum_covers_both_bin_settings(self):
        # The largest same-control difference, 3.89e-16, is pancreas G2M with 25 bins; the main 50-bin setting
        # alone gives 3.3e-16.
        text = row("LISI; cell-cycle scores")
        self.assertIn("$3.9 \\times 10^{-16}$, respectively", text)
        self.assertNotIn("$3.3 \\times 10^{-16}$", text)


if __name__ == "__main__":
    unittest.main()
