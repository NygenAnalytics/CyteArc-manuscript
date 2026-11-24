"""The methods collection step and doublet summary, and the cleaned methods records."""

import json
import unittest

import pandas as pd
from experiments.methods import collect_records as collect
from experiments.methods import doublet_summary
from experiments.runtime.common import ROOT

RECORDS = ROOT / "experiments" / "methods" / "records"
RUN_HISTORY = {"region", "startedAt", "elapsedSeconds", "callId", "provenance", "machine"}


def job(result: dict, status: str = "completed") -> dict:
    """A job record as the Modal runner saves it."""
    return {
        "module": "methods.example",
        "function": "run",
        "callId": "fc-00000000000000000000000000",
        "region": "us-east",
        "startedAt": 1.0,
        "status": status,
        "result": result,
        "elapsedSeconds": 2.0,
    }


def doublet_run(method: str, rows: list[dict]) -> dict:
    return job(
        {
            "status": "ok",
            "experiment": "rna_numerical_reproducibility",
            "check": "doublets",
            "provenance": {"cytearcVersion": "1.3.0"},
            "settings": {
                "check": "doublets",
                "method": method,
                "workers": 8,
                "workDirectory": "/tmp/methods/x",
                "outputDirectory": "/tmp/methods/results",
                "recordUri": "s3://bucket/experiments/methods/x.json",
            },
            "method": method,
            "datasets": rows,
            "eligibleDatasets": [row["dataset"] for row in rows],
        }
    )


class CompletedRuns(unittest.TestCase):
    def test_keeps_status_and_result_without_run_history(self):
        run = collect.completed(job({"status": "ok", "experiment": "x", "provenance": {}}), "x")
        self.assertEqual(
            run, {"status": "completed", "result": {"status": "ok", "experiment": "x"}}
        )

    def test_accepts_the_result_file_of_a_local_run(self):
        run = collect.completed({"status": "ok", "experiment": "x", "value": 3}, "x")
        self.assertEqual(run["result"], {"status": "ok", "experiment": "x", "value": 3})

    def test_refuses_failed_jobs_failed_results_and_other_experiments(self):
        with self.assertRaisesRegex(ValueError, "job status 'failed'"):
            collect.completed(job({"status": "ok", "experiment": "x"}, "failed"), "x")
        with self.assertRaisesRegex(ValueError, "result status 'partial'"):
            collect.completed(job({"status": "partial", "experiment": "x"}), "x")
        with self.assertRaisesRegex(ValueError, "Expected a run of lsi; got 'pca_drift'"):
            collect.completed({"status": "ok", "experiment": "pca_drift"}, "lsi")


class SingleRuns(unittest.TestCase):
    def test_chromvar_and_grnboost2_drop_bucket_keys(self):
        chromvar = collect.chromvar(
            job(
                {
                    "status": "ok",
                    "experiment": "chromvar",
                    "referenceInputsUri": "s3://b/k",
                    "cells": 5,
                }
            )
        )
        self.assertEqual(
            chromvar,
            {
                "status": "completed",
                "result": {"status": "ok", "experiment": "chromvar", "cells": 5},
            },
        )
        runs = [{"seed": 777, "links": 10, "uri": "s3://b/grn_seed_777.csv.gz"}]
        grn = collect.grnboost2(
            job({"status": "ok", "experiment": "grnboost2", "nativeRuns": runs})
        )
        self.assertEqual(grn["result"]["nativeRuns"], [{"seed": 777, "links": 10}])
        collect.public(grn, "grnboost2.json")

    def test_gene_scores_drops_the_source_path(self):
        record = collect.gene_scores(
            {
                "status": "ok",
                "experiment": "gene_scores",
                "sourcePath": "/home/u/atac.zarr",
                "cells": 400,
            }
        )
        self.assertEqual(record, {"status": "ok", "experiment": "gene_scores", "cells": 400})

    def test_lsi_puts_the_active_runs_after_the_sampled_run(self):
        def run(selection, cells):
            return {
                "status": "ok",
                "experiment": "lsi",
                "region": "us-central",
                "source": "/tmp/lsi/downloads/data.zarr",
                "settings": {"cellSelection": selection, "cells": cells},
            }

        record = collect.lsi(run("sampled", 400), [run("active", 6194)])
        self.assertEqual(record["settings"]["cells"], 400)
        self.assertEqual(record["fullSelectionMeasurements"][0]["settings"]["cells"], 6194)
        self.assertNotIn("region", record)
        self.assertNotIn("source", record["fullSelectionMeasurements"][0])
        with self.assertRaisesRegex(ValueError, "sampled cells"):
            collect.lsi(run("active", 6194), [run("active", 6194)])
        with self.assertRaisesRegex(ValueError, "active cell selection"):
            collect.lsi(run("sampled", 400), [run("sampled", 400)])
        with self.assertRaisesRegex(ValueError, "active cell selection"):
            collect.lsi(run("sampled", 400), [])

    def test_droplet_qc_keys_runs_by_dataset_in_the_order_given(self):
        def run(name):
            return job({"status": "ok", "experiment": "droplet_qc", "dataset": name})

        record = collect.droplet_qc([run("pbmc_10k_v3"), run("pbmc_1k_v2")])
        self.assertEqual(record["eligibleDatasets"], ["pbmc_10k_v3", "pbmc_1k_v2"])
        self.assertEqual(set(record["datasets"]["pbmc_1k_v2"]), {"status", "result"})
        with self.assertRaisesRegex(ValueError, "Two droplet_qc runs on pbmc_1k_v2"):
            collect.droplet_qc([run("pbmc_1k_v2"), run("pbmc_1k_v2")])
        with self.assertRaisesRegex(ValueError, "Expected a run of hto_demultiplexing"):
            collect.hto_demultiplexing([run("pbmc_1k_v2")])

    def test_rna_checks_drop_locations_and_stage_clock_times(self):
        def run(check, **extra):
            return job(
                {
                    "status": "ok",
                    "experiment": "rna_numerical_reproducibility",
                    "check": check,
                    "settings": {
                        "check": check,
                        "workers": 8,
                        "workDirectory": "/tmp/w",
                        "outputDirectory": "/tmp/o",
                    },
                    "source": {"cells": 10, "source": "/tmp/w/downloads/data.zarr"},
                    **extra,
                }
            )

        stages = [{"stage": "pca", "startedAtNs": 1, "finishedAtNs": 2, "status": "completed"}]
        determinism = run("determinism", pipelineRuns=[{"mode": "deterministic", "stages": stages}])
        record = collect.rna_numerical_reproducibility(run("pca"), determinism)
        pca = record["pca"]["measurement"]["result"]
        self.assertEqual(pca["settings"], {"check": "pca", "workers": 8})
        self.assertEqual(pca["source"], {"cells": 10})
        kept = record["determinism"]["measurement"]["result"]["pipelineRuns"][0]["stages"]
        self.assertEqual(kept, [{"stage": "pca", "status": "completed"}])
        self.assertIn("startedAtNs", stages[0], "the job record given must not change")
        with self.assertRaisesRegex(ValueError, "Expected the pca check"):
            collect.rna_numerical_reproducibility(determinism, determinism)

    def test_fate_probabilities_become_the_bundle_runtime_records_reads(self):
        result = {
            "experiment": "cellrank",
            "larry": {"auroc": 0.76, "cellsUri": "s3://b/x.npz"},
            "pancreas": {},
        }
        bundle = collect.fate_probabilities(result)
        self.assertEqual(
            bundle["results"]["pancreas_larry_fate_prediction"]["result"]["larry"], {"auroc": 0.76}
        )
        with self.assertRaisesRegex(ValueError, "completed cellrank_job"):
            collect.fate_probabilities(
                {"experiment": "cellrank", "larry": {"error": "x"}, "pancreas": {}}
            )
        self.assertEqual(
            collect.fate_cells("s3://b/methods/cellrank.json"),
            "s3://b/methods/cellrank_larry_cells.npz",
        )

    def test_public_refuses_private_locations_and_identifiers(self):
        for value in (
            "s3://bucket/key",
            "/tmp/x",
            "/home/u/x",
            ".scratch/runs",
            "fc-01CCCCCCCCCCCCCCCCCCCCCCCC",
        ):
            with self.assertRaises(ValueError, msg=value):
                collect.public({"result": {"notes": [value]}}, "x.json")
        with self.assertRaises(ValueError):
            collect.public({"r2EndpointUrl": "x"}, "x.json")
        collect.public({"url": "https://zenodo.org/api/records/4062232"}, "x.json")


class Doublets(unittest.TestCase):
    # Scores of four cells, two of them doublets: AUROC 3/4, and average precision
    # 1/2 * 1 + 1/2 * 2/3 = 5/6 from the ranking 0.8 (doublet), 0.4, 0.35 (doublet), 0.1.
    TABLE = pd.DataFrame(
        {
            "cell": ["a", "b", "c", "d"],
            "label": ["singlet", "singlet", "doublet", "doublet"],
            "score": [0.1, 0.4, 0.35, 0.8],
        }
    )

    def jobs(self, auprc: float = 5 / 6) -> dict:
        row = {
            "dataset": "m1",
            "auprc": auprc,
            "auroc": 0.75,
            "timing": {"seconds": 1, "versions": {"R": "4"}},
        }
        return {f"loc/{m}.json": doublet_run(m, [row]) for m in collect.DOUBLET_METHODS}

    def test_merges_scores_and_checks_them_against_each_run(self):
        record, tables = collect.doublets(self.jobs(), lambda location, dataset: self.TABLE)
        self.assertEqual(list(record["records"]), list(collect.DOUBLET_METHODS))
        [row] = record["validation"]
        self.assertEqual(
            {k: row[k] for k in ("dataset", "cells", "cellIdsAndLabelsIdentical")},
            {"dataset": "m1", "cells": 4, "cellIdsAndLabelsIdentical": True},
        )
        self.assertAlmostEqual(row["maxRecomputedMetricDifference"], 0.0, places=12)
        self.assertEqual(list(tables["m1"].columns), ["cell", "label", *collect.DOUBLET_METHODS])
        result = record["records"]["cytearc"]["result"]
        self.assertEqual(result["datasets"][0]["timing"], {"seconds": 1})
        self.assertEqual(
            result["settings"], {"check": "doublets", "method": "cytearc", "workers": 8}
        )
        collect.public(record, "doublets.json")

    def test_reports_how_far_a_run_metric_is_from_its_scores(self):
        record, _ = collect.doublets(self.jobs(auprc=0.5), lambda location, dataset: self.TABLE)
        self.assertAlmostEqual(record["validation"][0]["maxRecomputedMetricDifference"], 1 / 3)

    def test_refuses_runs_that_scored_other_cells(self):
        other = self.TABLE.assign(cell=["a", "b", "c", "e"])

        def read(location, dataset):
            return other if "scDblFinder" in location else self.TABLE

        with self.assertRaisesRegex(ValueError, "scDblFinder scored other cells"):
            collect.doublets(self.jobs(), read)

    def test_needs_one_run_per_method(self):
        jobs = self.jobs()
        jobs.pop("loc/scrublet.json")
        with self.assertRaisesRegex(ValueError, "one doublets run per method"):
            collect.doublets(jobs, lambda location, dataset: self.TABLE)

    def test_scores_sit_beside_the_job_record(self):
        self.assertEqual(
            collect.doublet_scores("s3://b/experiments/methods/doublets_scrublet.json", "hm-6k"),
            "s3://b/experiments/methods/doublets_scrublet/hm-6k/scores.csv",
        )


class DoubletSummary(unittest.TestCase):
    def record(self, values: dict[str, list[tuple[float, float]]]) -> dict:
        return {
            "records": {
                method: {
                    "result": {
                        "datasets": [
                            {"dataset": f"m{i}", "auprc": auprc, "auroc": auroc}
                            for i, (auprc, auroc) in enumerate(rows)
                        ]
                    }
                }
                for method, rows in values.items()
            }
        }

    def test_medians_and_wins_by_hand(self):
        summary = doublet_summary.summarise(
            self.record(
                {
                    "cytearc": [(0.5, 0.7), (0.2, 0.9), (0.9, 0.8), (0.4, 0.6)],
                    "scrublet": [(0.6, 0.7), (0.1, 0.5), (0.3, 0.6), (0.4, 0.9)],
                }
            )
        )
        # cytearc AUPRC 0.2, 0.4, 0.5, 0.9 -> median 0.45; scrublet 0.1, 0.3, 0.4, 0.6 -> 0.35.
        self.assertAlmostEqual(summary["medianAuprc"]["cytearc"], 0.45)
        self.assertAlmostEqual(summary["medianAuprc"]["scrublet"], 0.35)
        self.assertAlmostEqual(summary["medianAuroc"]["scrublet"], 0.65)
        # cytearc is higher on m1 and m2, lower on m0 and tied on m3.
        self.assertEqual(summary["cytearcHigherThan"], {"scrublet": 2})
        self.assertEqual(summary["datasets"], 4)

    def test_refuses_methods_on_different_matrices(self):
        record = self.record({"cytearc": [(0.5, 0.7)], "scrublet": [(0.6, 0.7), (0.1, 0.5)]})
        with self.assertRaisesRegex(ValueError, "same benchmark matrices"):
            doublet_summary.summarise(record)


class CleanedRecords(unittest.TestCase):
    def test_records_hold_no_run_history_or_private_location(self):
        for path in sorted(RECORDS.rglob("*.json")):
            record = json.loads(path.read_text())
            collect.public(record, path.name)
            keys = set()

            def walk(value):
                if isinstance(value, dict):
                    keys.update(value)
                    for item in value.values():
                        walk(item)
                elif isinstance(value, list):
                    for item in value:
                        walk(item)

            walk(record)
            self.assertFalse(keys & RUN_HISTORY, f"{path.name}: {sorted(keys & RUN_HISTORY)}")

    def test_doublet_record_reproduces_the_reported_medians(self):
        record = json.loads((RECORDS / "doublets.json").read_text())
        self.assertTrue(all(row["cellIdsAndLabelsIdentical"] for row in record["validation"]))
        summary = doublet_summary.summarise(record)
        self.assertEqual(summary["datasets"], 16)
        self.assertEqual(round(summary["medianAuprc"]["cytearc"], 2), 0.48)
        self.assertEqual(round(summary["medianAuprc"]["scDblFinder"], 2), 0.55)
        self.assertEqual(summary["cytearcHigherThan"]["scDblFinder"], 4)

    def test_per_cell_tables_match_the_doublet_record(self):
        from sklearn.metrics import average_precision_score

        record = json.loads((RECORDS / "doublets.json").read_text())
        table = pd.read_csv(RECORDS / "doublets" / "J293t-dm.csv.gz", float_precision="round_trip")
        truth = table["label"].to_numpy() == "doublet"
        for method, run in record["records"].items():
            row = next(r for r in run["result"]["datasets"] if r["dataset"] == "J293t-dm")
            self.assertEqual(row["cells"], len(table))
            self.assertEqual(average_precision_score(truth, table[method]), row["auprc"])

    def test_fate_bundle_keeps_the_layout_manuscript_code_reads(self):
        bundle = json.loads((RECORDS / "fate_probability_comparisons.json").read_text())
        self.assertEqual(set(bundle["results"]), {"pancreas_larry_fate_prediction"})
        entry = bundle["results"]["pancreas_larry_fate_prediction"]
        self.assertEqual(set(entry), {"result"})
        self.assertEqual(entry["result"]["larry"]["evaluatedProgenitors"], 1110)


if __name__ == "__main__":
    unittest.main()
