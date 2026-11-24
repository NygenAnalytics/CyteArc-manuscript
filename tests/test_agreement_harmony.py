"""Table 2's Harmony2 summary, read from small records whose values are set by hand."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from manuscript.tables import prepare_agreement

ARMS = ("ownStoppingRules", "fixedRoundsPenaltyOn", "fixedRoundsPenaltyOff")


def equivalence(passed: bool, cross: float) -> dict:
    return {
        "equivalence": {
            "passed": passed,
            "crossOverlapMedian": cross,
            "referenceOverlapMin": 0.9,
            "iLISIMedianDifference": 0.01,
            "referenceILISIRange": 0.02,
        }
    }


def harmonypy_dataset(off: float, on: float, correction: float, outcomes: tuple) -> dict:
    return {
        "cells": 100,
        "sharedStateChecks": {
            "assignmentCases": [
                {
                    "theta": 0.0,
                    "nativeReferenceMaxAbsoluteError": off,
                    "nativeEquationMaxAbsoluteError": 1e-16,
                    "referenceEquationMaxAbsoluteError": 2e-7,
                },
                {
                    "theta": 2.0,
                    "nativeReferenceMaxAbsoluteError": on,
                    "nativeEquationMaxAbsoluteError": 3e-16,
                    "referenceEquationMaxAbsoluteError": 4e-7,
                },
            ],
            "singleClusterDynamicRidge": {"coordinateMaxAbsoluteError": 5e-7},
            "correctionStep": {"nativeReferenceMaxAbsoluteError": correction, "passed": True},
        },
        "fullFits": {
            arm: equivalence(passed, 0.95) for arm, passed in zip(ARMS, outcomes, strict=True)
        },
    }


def r_dataset(off: float, on: float, outcomes: tuple) -> dict:
    return {
        "assignmentSteps": [
            {
                "theta": 0.0,
                "cytearcVersusR_firstCentroids": off,
                "versusRCodeEquationFromFirstAssignments": 1e-7,
            },
            {
                "theta": 2.0,
                "cytearcVersusR_firstCentroids": on,
                "versusRCodeEquationFromFirstAssignments": 6e-7,
            },
        ],
        "fullFits": {
            arm: equivalence(passed, 0.7) for arm, passed in zip(ARMS, outcomes, strict=True)
        },
    }


def stability(name: str) -> dict:
    overlaps = {"sameSeed": 0.8, "harmonypy": 0.85, "cytearc": 0.9}
    return {
        "cells": 1000 if name == "binvignat" else 2000,
        "cytearcVersusHarmonypySameSeed": {"knnOverlapK15": {"median": overlaps["sameSeed"]}},
        "harmonypyBetweenSeeds": {"knnOverlapK15": {"median": overlaps["harmonypy"]}},
        "cytearcBetweenSeeds": {"knnOverlapK15": {"median": overlaps["cytearc"]}},
    }


class HarmonySummary(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.harmonypy = {
            "provenance": {"harmonypyVersion": "9.9.9"},
            "settings": {
                "seeds": [0, 1, 2],
                "arms": {"fixedRoundsPenaltyOn": {"max_iter_harmony": 7}},
            },
            "datasets": {
                "kang": harmonypy_dataset(1e-7, 0.008, 0.02, (True, False, False)),
                "binvignat": harmonypy_dataset(3e-7, 0.006, 0.01, (True, False, True)),
            },
        }
        self.r_harmony = {
            "reference": {"version": "8.8.8"},
            "datasets": {
                "kang": r_dataset(2e-6, 0.004, (True, False, False)),
                "binvignat": r_dataset(1e-6, 0.007, (False, False, True)),
            },
        }
        self.stability = {
            "comparisons": [
                {"measurement": {name: stability(name) for name in ("binvignat", "kang")}}
            ]
        }

    def summary(self) -> dict:
        for name, record in (
            ("harmony2.json", self.harmonypy),
            ("harmony2_r_reference.json", self.r_harmony),
            ("harmony_seed_stability.json", self.stability),
        ):
            (self.folder / name).write_text(json.dumps(record))
        with mock.patch.object(prepare_agreement, "INTEGRATION", self.folder):
            return prepare_agreement.harmony_summary()

    def test_versions_cells_and_seeds(self):
        result = self.summary()
        self.assertEqual((result["harmonypy"], result["rHarmony"]), ("9.9.9", "8.8.8"))
        self.assertEqual(result["cells"], {"kang": 100, "binvignat": 100})
        self.assertEqual(result["seeds"], 3)
        self.assertEqual(result["fixedRounds"], 7)

    def test_steps_take_the_largest_error_of_each_penalty_setting(self):
        steps = self.summary()["steps"]
        self.assertEqual(steps["assignmentPenaltyOff"], {"harmonypy": 3e-7, "rHarmony": 2e-6})
        self.assertEqual(steps["assignmentPenaltyOn"], {"harmonypy": 0.008, "rHarmony": 0.007})
        self.assertEqual(steps["harmonypyOwnNumerator"], 4e-7)
        self.assertEqual(steps["rHarmonyOwnNumerator"], 6e-7)
        self.assertEqual(steps["correctionVersusHarmonypy"], 0.02)
        self.assertTrue(steps["correctionOwnEquationsReproduced"])

    def test_full_fit_outcomes_count_the_datasets_that_pass(self):
        fits = self.summary()["fullFits"]
        counts = {
            reference: [fits[reference][arm]["datasetsPassed"] for arm in ARMS]
            for reference in ("harmonypy", "rHarmony")
        }
        self.assertEqual(counts, {"harmonypy": [2, 0, 1], "rHarmony": [1, 0, 1]})
        self.assertFalse(fits["rHarmony"]["ownStoppingRules"]["binvignat"]["passed"])
        self.assertEqual(
            fits["rHarmony"]["ownStoppingRules"]["binvignat"]["crossOverlapMedian"], 0.7
        )

    def test_seed_stability_reads_each_dataset(self):
        seed = self.summary()["seedStability"]
        self.assertEqual(seed["kang"]["cells"], 2000)
        self.assertEqual(seed["binvignat"]["harmonypyBetweenSeedsOverlapMedian"], 0.85)

    def test_refuses_records_of_different_datasets(self):
        del self.r_harmony["datasets"]["kang"]
        with self.assertRaisesRegex(ValueError, "different datasets"):
            self.summary()

    def test_refuses_more_than_one_seed_comparison(self):
        self.stability["comparisons"].append(self.stability["comparisons"][0])
        with self.assertRaisesRegex(ValueError, "one ten-seed comparison"):
            self.summary()


if __name__ == "__main__":
    unittest.main()
