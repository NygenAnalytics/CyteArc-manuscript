import os
import unittest

from cytearc.storage import ArtifactRef
from cytearc.storage.artifacts import ArtifactStatus
from experiments.mapping import mapping_reruns
from experiments.runtime.metrics import ResourceMeasurement, ResourceSampler

GIB = 1024**3
STORED_ANN = {"ann_parallel": True, "rand_state": 4466, "ann_ef": 200}
STORED_UMAP = {"parallel": True, "n_epochs": 300, "random_seed": 4444}


def status(kind: str, parameters: dict, options: dict | None) -> ArtifactStatus:
    ref = ArtifactRef(artifact_id="0" * 64, kind=kind, scope="assay", assay="RNA")
    return ArtifactStatus(
        ref=ref,
        path=f"RNA/artifacts/{kind}/{'0' * 64}",
        exists=True,
        complete=True,
        provenance={"operation": "run", "parameters": parameters},
        execution_options=options,
    )


class ParallelSettingsTests(unittest.TestCase):
    def test_settings_the_stored_artifacts_record_are_passed_on(self) -> None:
        settings = mapping_reruns.parallel_settings(
            {"annParallel": True, "umapParallel": True}, STORED_ANN, STORED_UMAP
        )
        self.assertEqual(settings, {"annParallel": True, "umapParallel": True})

    def test_a_sequential_umap_is_refused_when_the_stored_one_ran_in_parallel(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "records parallel=True, but the payload sets umapParallel=False"
        ):
            mapping_reruns.parallel_settings(
                {"annParallel": True, "umapParallel": False}, STORED_ANN, STORED_UMAP
            )

    def test_a_parallel_fork_of_a_sequential_index_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "records ann_parallel=False"):
            mapping_reruns.parallel_settings(
                {"annParallel": True, "umapParallel": True},
                {**STORED_ANN, "ann_parallel": False},
                STORED_UMAP,
            )

    def test_missing_or_non_boolean_settings_are_refused(self) -> None:
        for payload in (
            {"umapParallel": True},
            {"annParallel": "true", "umapParallel": True},
            {"annParallel": 1, "umapParallel": True},
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(ValueError, "must set annParallel to true or false"),
            ):
                mapping_reruns.parallel_settings(payload, STORED_ANN, STORED_UMAP)

    def test_a_stored_umap_without_the_flag_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "records parallel=None"):
            mapping_reruns.parallel_settings(
                {"annParallel": True, "umapParallel": True}, STORED_ANN, {"n_epochs": 300}
            )


class ThreadSettingsTests(unittest.TestCase):
    def test_umaps_laid_out_on_the_same_threads_match(self) -> None:
        stored = status("embedding", STORED_UMAP, {"nthreads": 16, "layout_threads": 16})
        fork = status("embedding", STORED_UMAP, {"nthreads": 16, "layout_threads": 16})
        settings = mapping_reruns.thread_settings(stored, fork)
        self.assertEqual(
            settings["stored"], {"parallel": True, "nthreads": 16, "layoutThreads": 16}
        )
        self.assertTrue(settings["same"])

    def test_the_eight_cpu_fork_does_not_match_its_stored_umap(self) -> None:
        # What the replicate-5 store records for its stored UMAP and for the first fork's UMAP.
        stored = status("embedding", STORED_UMAP, {"nthreads": 16, "layout_threads": 16})
        fork = status("embedding", STORED_UMAP, {"nthreads": 8, "layout_threads": 8})
        settings = mapping_reruns.thread_settings(stored, fork)
        self.assertFalse(settings["same"])
        self.assertEqual(settings["fork"]["layoutThreads"], 8)

    def test_a_sequential_layout_on_the_same_thread_count_does_not_match(self) -> None:
        stored = status("embedding", STORED_UMAP, {"nthreads": 16, "layout_threads": 16})
        fork = status(
            "embedding", {**STORED_UMAP, "parallel": False}, {"nthreads": 16, "layout_threads": 1}
        )
        self.assertFalse(mapping_reruns.thread_settings(stored, fork)["same"])

    def test_index_and_query_settings(self) -> None:
        index = status("ann_index", STORED_ANN, {"nthreads": 16, "batch_size": 1000000})
        self.assertEqual(
            mapping_reruns.recorded_threads(index),
            {"parallel": True, "nthreads": 16, "layoutThreads": None},
        )
        query = status("neighbors", {"k": 15}, {"nthreads": 8, "batch_size": 1000000})
        self.assertEqual(
            mapping_reruns.recorded_threads(query),
            {"parallel": None, "nthreads": 8, "layoutThreads": None},
        )

    def test_an_artifact_without_execution_options(self) -> None:
        self.assertEqual(
            mapping_reruns.recorded_threads(status("embedding", {"parallel": False}, None)),
            {"parallel": False, "nthreads": None, "layoutThreads": None},
        )


class ContainerTests(unittest.TestCase):
    PAYLOAD = {"cpu": 16, "memoryMb": 65536, "workers": 16, "budgetBytes": 48 * GIB}

    def test_the_stated_container_is_kept_when_the_memory_limit_agrees(self) -> None:
        record = mapping_reruns.container(
            self.PAYLOAD, ResourceMeasurement(memoryMaxBytes=64 * GIB)
        )
        self.assertEqual(
            (record["cpu"], record["memoryMb"], record["workers"], record["budgetBytes"]),
            (16, 65536, 16, 48 * GIB),
        )
        self.assertEqual(record["memoryMaxBytes"], 64 * GIB)
        self.assertEqual(record["visibleCpus"], len(os.sched_getaffinity(0)))
        self.assertGreaterEqual(record["numbaThreads"], 1)

    def test_a_container_with_another_memory_limit_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "34359738368 bytes, not the 65536 MiB"):
            mapping_reruns.container(self.PAYLOAD, ResourceMeasurement(memoryMaxBytes=32 * GIB))

    def test_an_unlimited_or_unreadable_limit_is_recorded_as_it_is(self) -> None:
        for limit in ("max", None):
            with self.subTest(limit=limit):
                record = mapping_reruns.container(
                    self.PAYLOAD, ResourceMeasurement(memoryMaxBytes=limit)
                )
                self.assertEqual(record["memoryMaxBytes"], limit)

    def test_a_payload_without_its_container_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            mapping_reruns.container({"workers": 16, "budgetBytes": GIB}, ResourceMeasurement())

    def test_a_sampler_started_and_stopped_at_once_reads_the_limit(self) -> None:
        # The job reads the limit this way before its first step.
        sampler = ResourceSampler(sampleIntervalSeconds=0.25, resetCgroupPeak=False)
        sampler.start()
        measurement = sampler.stop()
        limit = measurement.memoryMaxBytes
        self.assertTrue(limit is None or limit == "max" or (isinstance(limit, int) and limit > 0))


if __name__ == "__main__":
    unittest.main()
