import unittest

from experiments.benchmarks import sweep


def run(backend: str, index: float, query: float, complete: bool = True) -> dict:
    return {
        "complete": complete,
        "dataset": "embryo1M",
        "storeBackend": backend,
        "budgetGiB": 24,
        "workers": 8,
        "elapsedSeconds": 100.0,
        "peakRssBytes": 2**30,
        "streamingSeconds": 10.0,
        "stageSeconds": {"buildAnnIndex": index, "queryNeighbors": query, "runUmap": 99.0},
    }


class NeighborIndexSecondsTest(unittest.TestCase):
    def test_each_group_averages_its_runs_index_build_plus_query(self):
        # Local disk: 20 + 30 and 24 + 20 seconds average 47; the object store's single run stays in its own group,
        # and an incomplete run on local disk is left out.
        rows = [
            run("modalLocalDisk", 20.0, 30.0),
            run("modalLocalDisk", 24.0, 20.0),
            run("modalLocalDisk", 500.0, 500.0, complete=False),
            run("objectStore", 40.0, 40.0),
        ]
        summary = {s["storeBackend"]: s for s in sweep.summarise(rows)}
        self.assertEqual(summary["modalLocalDisk"]["neighborIndexSecondsMean"], 47.0)
        self.assertEqual(summary["modalLocalDisk"]["replicates"], 2)
        self.assertEqual(summary["objectStore"]["neighborIndexSecondsMean"], 80.0)


if __name__ == "__main__":
    unittest.main()
