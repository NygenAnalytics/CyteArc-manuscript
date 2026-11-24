import os
import unittest
from unittest import mock

from experiments.benchmarks.budget_compliance import ARMS, IMPORTED_CLUSTER_STAGES
from experiments.benchmarks.config import (
    SELECTED_STAGE_DEPENDENCIES,
    load_profiling_config,
)

CONFIGS = "experiments/benchmarks/configs/benchmark/"
# Loading a configuration only builds bucket URIs, so placeholder locations stand in for the bucket.
BUCKET = {"R2_ENDPOINT_URL": "https://storage.invalid", "R2_BUCKET": "experiments-bucket"}


def setUpModule() -> None:
    patcher = mock.patch.dict(os.environ, BUCKET)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


GRAPH_STEPS = {"buildAnnIndex", "queryNeighbors", "buildConnectivityMap", "runUmap", "runLeiden"}


class ArmStagesTest(unittest.TestCase):
    def test_configs_without_a_cluster_source_keep_every_arm_unchanged(self):
        for name, cells in (
            (
                "embryo_rna_budget_compliance_1000000_cells_8_gib_8_workers_replicate_1.toml",
                1000000,
            ),
            (
                "embryo_rna_budget_compliance_10000000_cells_8_gib_8_workers_replicate_1.toml",
                10000000,
            ),
        ):
            config = load_profiling_config(CONFIGS + name)
            for arm, spec in ARMS.items():
                self.assertEqual(spec.stagesFor(config, cells), spec.stages, (name, arm))

    def test_a_cluster_source_replaces_the_graph_steps_with_imported_labels(self):
        for budget in (8,):
            config = load_profiling_config(
                CONFIGS
                + f"embryo_rna_budget_compliance_imported_clusters_10000000_cells_{budget}_gib_8_workers_replicate_1.toml"
            )
            self.assertIsNotNone(config.clusterSourceFor(10000000))
            for spec in ARMS.values():
                stages = spec.stagesFor(config, 10000000)
                self.assertEqual(stages, IMPORTED_CLUSTER_STAGES)
                self.assertFalse(GRAPH_STEPS & set(stages))
                self.assertLess(stages.index("importClusters"), stages.index("findMarkers"))
                # Every step's resources come from the configuration, as the harness requires.
                self.assertEqual(len({config.resourcesFor(stage) for stage in stages}), 1)

    def test_imported_cluster_steps_respect_the_stage_graph(self):
        # The harness opens the store in every process, so reopenStore is the only step it omits.
        allowed = set(IMPORTED_CLUSTER_STAGES) | {"reopenStore"}
        for stage in IMPORTED_CLUSTER_STAGES:
            for dependency in SELECTED_STAGE_DEPENDENCIES[stage]:
                self.assertIn(dependency, allowed, stage)

    def test_the_labels_run_uses_the_same_input_filter_and_workflow(self):
        labels = load_profiling_config(CONFIGS + "embryo_rna_cluster_labels_10000000_cells.toml")
        budget = load_profiling_config(
            CONFIGS
            + "embryo_rna_budget_compliance_imported_clusters_10000000_cells_8_gib_8_workers_replicate_1.toml"
        )
        self.assertEqual(labels.datasetUri(10000000), budget.datasetUri(10000000))
        self.assertEqual(labels.workflow, budget.workflow)
        self.assertEqual(labels.storeBackend, "objectStore")
        self.assertEqual(
            budget.clusterSourceFor(10000000).leidenRecordUri,
            labels.resultUri(10000000, "runLeiden"),
        )


if __name__ == "__main__":
    unittest.main()
