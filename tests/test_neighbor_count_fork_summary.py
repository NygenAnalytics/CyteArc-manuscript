import unittest

from experiments.mapping import neighbor_count_fork_summary as fork
from experiments.runtime.common import load_json


def result(fork_seconds: float = 600.0) -> dict:
    # A stored analysis of 1,200 seconds; the fork recomputes 400 + 200 seconds and looks up 30 + 30.
    return {
        "kOriginal": 11,
        "kNew": 15,
        "forkSeconds": fork_seconds,
        "fullRunAnalysisSeconds": 1200.0,
        "stages": {
            "reuse.pca": {"seconds": 30.0},
            "reuse.annIndex": {"seconds": 30.0},
            "fork.queryNeighbors": {"seconds": 400.0},
            "fork.runUmap": {"seconds": 200.0},
        },
    }


class ForkSummaryTest(unittest.TestCase):
    def test_the_savings_compare_the_recomputed_and_looked_up_minutes_with_the_stored_analysis(
        self,
    ):
        summary = fork.fork_summary(result())
        self.assertEqual((summary["storedAnalysisMinutes"], summary["forkMinutes"]), (20.0, 10.0))
        self.assertEqual((summary["lookupMinutes"], summary["forkWithLookupsMinutes"]), (1.0, 11.0))
        # 600 of 1,200 seconds saves half; 660 saves 45%.
        self.assertEqual(summary["forkSaving"], 0.5)
        self.assertAlmostEqual(summary["forkWithLookupsSaving"], 0.45, places=12)

    def test_stages_that_miss_the_fork_time_or_no_lookup_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "add up to 600.0 seconds, not the fork's 650.0"):
            fork.fork_summary(result(fork_seconds=650.0))
        no_lookup = result()
        no_lookup["stages"] = {
            k: v for k, v in no_lookup["stages"].items() if k.startswith("fork.")
        }
        with self.assertRaisesRegex(ValueError, "looked up no stored artifact"):
            fork.fork_summary(no_lookup)

    def test_the_stored_fork_gives_the_quoted_values(self):
        # Paragraphs 52 and 84: forking k from 11 to 15 cut the 48.4-minute stored analysis by 27% to 35.4 minutes of
        # recomputed stages, and by 24% to 36.7 minutes with the three lookups.
        (run,) = load_json(fork.RUNS)["results"].values()
        summary = fork.fork_summary(run["result"])
        self.assertEqual((summary["kOriginal"], summary["kNew"]), (11, 15))
        self.assertEqual(
            [
                round(summary[k], 1)
                for k in ("storedAnalysisMinutes", "forkMinutes", "forkWithLookupsMinutes")
            ],
            [48.4, 35.4, 36.7],
        )
        self.assertEqual(
            [round(summary[k] * 100) for k in ("forkSaving", "forkWithLookupsSaving")], [27, 24]
        )


if __name__ == "__main__":
    unittest.main()
