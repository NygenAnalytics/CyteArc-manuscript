import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.interferon_contrasts import lineage_contrasts as contrasts


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TemporaryRoot(unittest.TestCase):
    """A temporary repository root for the contrasts' module paths; subclasses list the paths to point into it, a list of
    relative paths for a path list."""

    PATHS: dict[str, str | list[str]] = {}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, relative in {"ROOT": "", **self.PATHS}.items():
            value = (
                [self.root / r for r in relative]
                if isinstance(relative, list)
                else self.root / relative
            )
            patch = mock.patch.object(contrasts, name, value)
            patch.start()
            self.addCleanup(patch.stop)


def interferon_rows(
    store,
    collection,
    lineage,
    disease,
    scores,
    *,
    earliest="True",
    cells=50,
    donor=None,
    sample=None,
):
    return [
        {
            "storeId": store,
            "collectionId": collection,
            "donor_id": donor or f"{store}-{disease}-{i}",
            "sample": sample or f"{store}-{disease}-{i}-s",
            "disease": disease,
            "coarse_cell_type": lineage,
            "cells": cells,
            "earliestSample": earliest,
            "interferonZScore": score,
        }
        for i, score in enumerate(scores)
    ]


COMBAT = "ahern_2022_combat_project_single_cell_19_sepsis_flu_patient_pbmcs_ebc2e1ff"
ATLAS = "marteau_2026_single_cell_colorectal_cancer_atlas_core_atlas_05a8c945"


class LineageContrastsTest(TemporaryRoot):
    PATHS = {
        "INTERFERON_RUNS": ["a.json", "b.json"],
        "INTERFERON_TABLES": ["a_donors.csv.gz", "b_donors.csv.gz"],
        "COHORT_VALUES": "severity.json",
        "SOURCE_STUDY_VALUES": "source_studies.json",
    }

    def write_inputs(self, tables, *, recorded_sha=None, cohorts=(), sources=()):
        import pandas as pd

        write(
            contrasts.SOURCE_STUDY_VALUES,
            {
                "result": {
                    "rows": [
                        {
                            "storeId": store,
                            "donor_id": donor,
                            "sample": sample,
                            "column": "study_id",
                            "value": value,
                            "cells": 100,
                        }
                        for store, donor, sample, value in sources
                    ]
                }
            },
        )

        write(
            contrasts.COHORT_VALUES,
            {
                "result": {
                    "rows": [
                        {
                            "storeId": store,
                            "donor_id": donor,
                            "sample": sample,
                            "column": "Source",
                            "value": value,
                            "cells": 100,
                        }
                        for store, donor, sample, value in cohorts
                    ]
                }
            },
        )

        for i, (run, table, rows) in enumerate(
            zip(contrasts.INTERFERON_RUNS, contrasts.INTERFERON_TABLES, tables)
        ):
            pd.DataFrame(rows).to_csv(table, index=False, compression="gzip")
            sha = hashlib.sha256(table.read_bytes()).hexdigest()
            write(
                run,
                {
                    "status": "completed",
                    "result": {
                        "tables": {"donors": {"sha256": recorded_sha or sha}},
                        "wallSeconds": 100.0 + i,
                        "totals": {
                            "storesScreened": 2,
                            "primaryBloodCellsRead": 1000,
                            "donorsWithScore": 10,
                            "responseBytes": 5,
                        },
                    },
                },
            )

    def test_rows_join_a_studys_lineage_stores_and_keep_independent_cohorts_apart(self):
        split = prepare_split_collection()
        first = (
            # A study split by lineage: T cells in one store, monocytes in another.
            interferon_rows("liu_a", "c1", "T cells", "normal", [0, 0.1, 0.2, 0.3, 0.4])
            + interferon_rows("liu_a", "c1", "T cells", "COVID-19", [1, 2, 3, 4, 5])
            + interferon_rows("liu_b", "c1", "Monocytes", "normal", [0, 1, 2, 3, 4])
            + interferon_rows("liu_b", "c1", "Monocytes", "COVID-19", [0, 1, 2, 3, 4])
            # Two independent cohorts in one collection.
            + interferon_rows("jin_x", split, "T cells", "normal", [0, 0, 0, 0, 0])
            + interferon_rows("jin_x", split, "T cells", "COVID-19", [1, 1, 1, 1, 1])
            + interferon_rows("jin_y", split, "T cells", "normal", [1, 1, 1, 1, 1])
            + interferon_rows("jin_y", split, "T cells", "COVID-19", [0, 0, 0, 0, 0])
        )
        second = (
            interferon_rows("gong", "c3", "T cells", "normal", [0, 1, 2, 3, 4])
            + interferon_rows(
                "gong", "c3", "T cells", "cytomegalovirus infection", [0.5, 1.5, 2.5, 3.5, 4.5]
            )
            # Four donors are too few for a contrast; later samples and small pseudobulks stay out.
            + interferon_rows("gong", "c3", "T cells", "influenza", [9, 9, 9, 9])
            + interferon_rows("gong", "c3", "T cells", "COVID-19", [9] * 5, earliest="False")
            + interferon_rows("gong", "c3", "T cells", "rheumatoid arthritis", [9] * 5, cells=19)
        )
        self.write_inputs([first, second])
        score = contrasts.lineage_contrasts()
        rows = {(tuple(r["stores"]), r["disease"]): r for r in score["rows"]}
        self.assertEqual(
            sorted(rows),
            [
                (("gong",), "cytomegalovirus infection"),
                (("jin_x",), "COVID-19"),
                (("jin_y",), "COVID-19"),
                (("liu_a", "liu_b"), "COVID-19"),
            ],
        )
        liu = rows[(("liu_a", "liu_b"), "COVID-19")]
        # Every disease donor above every healthy one in T cells; the monocytes tie half the comparisons.
        self.assertEqual(liu["auc"]["T cells"], 1.0)
        self.assertEqual(liu["auc"]["Monocytes"], 0.5)
        self.assertIsNone(liu["auc"]["NK and ILC"])
        self.assertEqual(liu["ridge"]["cases"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(rows[(("jin_x",), "COVID-19")]["auc"]["T cells"], 1.0)
        self.assertEqual(rows[(("jin_y",), "COVID-19")]["auc"]["T cells"], 0.0)
        # Interferon diseases first by falling T-cell AUC, then the controls.
        order = [(tuple(r["stores"]), r["interferon"]) for r in score["rows"]]
        self.assertEqual([interferon for _, interferon in order], [True, True, True, False])
        self.assertEqual(order[-1][0], ("gong",))
        self.assertEqual(score["rows"][-1]["auc"]["T cells"], 0.6)
        self.assertEqual(score["totals"]["storesScreened"], 4)
        self.assertEqual(score["totals"]["wallSeconds"], 201.0)

    def other(self) -> list:
        """A second job's table with one store outside COHORT_RECORDS."""
        return interferon_rows("s", "c", "T cells", "normal", [0] * 5)

    def combat(self, cohort: str, disease: str, scores, *, earliest="True") -> tuple[list, list]:
        """COMBAT's T-cell units of one recorded cohort, one sample per donor, and their cohort records."""
        rows = [
            row
            for i, score in enumerate(scores)
            for row in interferon_rows(
                COMBAT,
                "combat",
                "T cells",
                disease,
                [score],
                donor=f"{cohort}{i}",
                sample=f"{cohort}{i}-s",
                earliest=earliest,
            )
        ]
        return rows, [(COMBAT, row["donor_id"], row["sample"], cohort) for row in rows]

    def test_combats_sepsis_cohort_is_compared_as_sepsis_and_its_neutrophil_samples_leave(self):
        healthy, healthy_cohort = self.combat("HV", "normal", [0, 0, 0, 0, 0])
        covid, covid_cohort = self.combat("COVID_SEV", "COVID-19", [1, 1, 1, 1, 1])
        # The store labels its sepsis donors COVID-19; their scores sit below every healthy one.
        sepsis, sepsis_cohort = self.combat("Sepsis", "COVID-19", [-1, -1, -1, -1, -1])
        neutrophils, neutrophil_cohort = self.combat("COVID_LDN", "COVID-19", [-5, -5])
        self.write_inputs(
            [healthy + covid + sepsis + neutrophils, self.other()],
            cohorts=healthy_cohort + covid_cohort + sepsis_cohort + neutrophil_cohort,
        )
        score = contrasts.lineage_contrasts()
        rows = {r["disease"]: r for r in score["rows"]}
        self.assertEqual(sorted(rows), ["COVID-19", "sepsis"])
        # Only the five donors of the COVID-19 cohort remain, each above every healthy donor.
        self.assertEqual(rows["COVID-19"]["ridge"]["cases"], [1.0] * 5)
        self.assertEqual(rows["COVID-19"]["auc"]["T cells"], 1.0)
        self.assertEqual(rows["sepsis"]["auc"]["T cells"], 0.0)
        self.assertFalse(rows["sepsis"]["interferon"])
        self.assertEqual(
            score["cohortRecords"],
            {
                COMBAT: {
                    "relabelled": {"sepsis": {"samples": 5, "donors": 5}},
                    "excluded": {"samples": 2, "donors": 2},
                }
            },
        )
        self.assertEqual(score["source"][-2:], ["severity.json", "source_studies.json"])

    def atlas(self, source: str, disease: str, scores) -> tuple[list, list]:
        """The colorectal cancer atlas's T-cell units of one source study, and their source-study records."""
        rows = interferon_rows(ATLAS, "atlas", "T cells", disease, scores)
        rows = [{**row, "donor_id": f"{source}-{row['donor_id']}"} for row in rows]
        return rows, [(ATLAS, row["donor_id"], row["sample"], source) for row in rows]

    def test_the_atlas_contrasts_donors_only_within_one_source_study(self):
        # The donors with cancer come from one source study and the healthy donors from another, as in the atlas: in
        # one store they would give an AUC of 0, but no source study holds both sides.
        cancer, cancer_sources = self.atlas("Innsbruck", "colorectal cancer", [-1] * 5)
        healthy, healthy_sources = self.atlas("Chen", "normal", [1] * 5)
        # A third source study holds both sides, and keeps its contrast.
        both, both_sources = self.atlas("Sathe", "colorectal cancer", [2] * 5)
        controls, control_sources = self.atlas("Sathe", "normal", [0] * 5)
        self.write_inputs(
            [cancer + healthy + both + controls, self.other()],
            sources=cancer_sources + healthy_sources + both_sources + control_sources,
        )
        score = contrasts.lineage_contrasts()
        (row,) = score["rows"]
        self.assertEqual((row["stores"], row["disease"]), ([ATLAS], "colorectal cancer"))
        self.assertEqual(row["auc"]["T cells"], 1.0)
        self.assertEqual(row["ridge"]["healthy"], [0.0] * 5)
        self.assertEqual(
            score["sourceStudies"],
            {
                ATLAS: {
                    "Chen": {"samples": 5, "donors": 5},
                    "Innsbruck": {"samples": 5, "donors": 5},
                    "Sathe": {"samples": 10, "donors": 10},
                }
            },
        )

    def test_an_atlas_sample_without_its_source_study_is_an_error(self):
        healthy, sources = self.atlas("Chen", "normal", [1] * 5)
        self.write_inputs([healthy, self.other()], sources=sources[1:])
        with self.assertRaisesRegex(ValueError, "no 'study_id' value"):
            contrasts.lineage_contrasts()

    def test_a_sample_without_its_cohort_is_an_error(self):
        healthy, cohort = self.combat("HV", "normal", [0, 0, 0, 0, 0])
        self.write_inputs([healthy, self.other()], cohorts=cohort[1:])
        with self.assertRaisesRegex(ValueError, "no 'Source' value"):
            contrasts.lineage_contrasts()

    def test_an_earliest_sample_leaving_while_its_donor_has_another_is_an_error(self):
        healthy, healthy_cohort = self.combat("HV", "normal", [0, 0, 0, 0, 0])
        neutrophils, neutrophil_cohort = self.combat("COVID_LDN", "COVID-19", [1])
        # The same donor's later sample, which the screen did not mark as its earliest.
        later = interferon_rows(
            COMBAT,
            "combat",
            "T cells",
            "COVID-19",
            [1],
            donor="COVID_LDN0",
            sample="later",
            earliest="False",
        )
        self.write_inputs(
            [healthy + neutrophils + later, self.other()],
            cohorts=healthy_cohort
            + neutrophil_cohort
            + [(COMBAT, "COVID_LDN0", "later", "COVID_SEV")],
        )
        with self.assertRaisesRegex(ValueError, "earliest sample left"):
            contrasts.lineage_contrasts()

    def test_a_donor_table_other_than_its_job_recorded_is_an_error(self):
        rows = interferon_rows("s", "c", "T cells", "normal", [0] * 5)
        self.write_inputs([rows, rows], recorded_sha="0" * 64)
        with self.assertRaisesRegex(ValueError, "not the donor table its job recorded"):
            contrasts.lineage_contrasts()


def prepare_split_collection() -> str:
    from experiments.reuse.progenitor_effects import COHORTS_BY_STORE

    return COHORTS_BY_STORE[0]


def contrast(
    interferon: bool,
    t_cells: float | None,
    others: float | None = 0.5,
    dendritic: float | None = 0.5,
) -> dict:
    """A contrast row as lineage_contrasts writes it: its T-cell and dendritic-cell AUROCs, and one AUROC for the
    other three lineages."""
    return {
        "interferon": interferon,
        "auc": {lineage: others for lineage in contrasts.LINEAGES}
        | {"T cells": t_cells, "Dendritic cells": dendritic},
    }


class ContrastGroupsTest(unittest.TestCase):
    def test_each_group_counts_its_contrasts_and_spans_their_aurocs_by_lineage_and_overall(self):
        groups = contrasts.contrast_groups(
            [
                contrast(True, 0.9, others=0.55, dendritic=0.94),
                contrast(True, 0.64, others=0.7, dendritic=0.74),
                # A contrast without T cells or dendritic cells counts, but gives those lineages no AUROC.
                contrast(True, None, others=1.0, dendritic=None),
                contrast(False, 0.12, others=0.3, dendritic=0.63),
                contrast(False, 0.54, others=0.08, dendritic=0.29),
            ]
        )
        interferon, control = groups["interferon"], groups["control"]
        self.assertEqual((interferon["contrasts"], control["contrasts"]), (3, 2))
        self.assertEqual(
            interferon["lineages"]["T cells"],
            {"contrasts": 2, "aurocLowest": 0.64, "aurocHighest": 0.9},
        )
        self.assertEqual(
            interferon["lineages"]["Dendritic cells"],
            {"contrasts": 2, "aurocLowest": 0.74, "aurocHighest": 0.94},
        )
        self.assertEqual(
            interferon["lineages"]["B cells"],
            {"contrasts": 3, "aurocLowest": 0.55, "aurocHighest": 1.0},
        )
        self.assertEqual(
            control["lineages"]["Dendritic cells"],
            {"contrasts": 2, "aurocLowest": 0.29, "aurocHighest": 0.63},
        )
        # The overall range spans every lineage: its ends come from different lineages than the T-cell range's.
        self.assertEqual((interferon["aurocLowest"], interferon["aurocHighest"]), (0.55, 1.0))
        self.assertEqual((control["aurocLowest"], control["aurocHighest"]), (0.08, 0.63))

    def test_a_group_with_a_lineage_no_contrast_scores_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no control contrast scores T cells"):
            contrasts.contrast_groups([contrast(True, 0.9), contrast(False, None)])
        with self.assertRaisesRegex(ValueError, "no interferon contrast scores Dendritic cells"):
            contrasts.contrast_groups([contrast(True, 0.9, dendritic=None), contrast(False, 0.2)])


class StoredRecordsTest(unittest.TestCase):
    """The contrasts of the stored interferon screen as paragraph 50 reports them."""

    @classmethod
    def setUpClass(cls):
        cls.contrasts = contrasts.lineage_contrasts()

    def test_23_contrasts_split_into_13_interferon_and_10_controls(self):
        groups = contrasts.contrast_groups(self.contrasts["rows"])
        self.assertEqual(len(self.contrasts["rows"]), 23)
        self.assertEqual(
            (groups["interferon"]["contrasts"], groups["control"]["contrasts"]), (13, 10)
        )
        # T-cell AUROC 0.64 to 1.00 for the interferon diseases and 0.12 to 0.54 for the controls, and in dendritic
        # cells 0.74 to 0.94 against 0.29 to 0.63: in both lineages every interferon contrast lies above every control.
        for lineage, expected in (
            ("T cells", [0.64, 1.0, 0.12, 0.54]),
            ("Dendritic cells", [0.74, 0.94, 0.29, 0.63]),
        ):
            ranges = [
                round(groups[group]["lineages"][lineage][end], 2)
                for group in ("interferon", "control")
                for end in ("aurocLowest", "aurocHighest")
            ]
            self.assertEqual(ranges, expected, lineage)
        # Every lineage an interferon contrast scores lies above 0.5: 0.55 to 1.00.
        interferon = groups["interferon"]
        self.assertEqual(
            (round(interferon["aurocLowest"], 2), interferon["aurocHighest"]), (0.55, 1.0)
        )
        scored = [
            auc
            for row in self.contrasts["rows"]
            if row["interferon"]
            for auc in row["auc"].values()
            if auc is not None
        ]
        self.assertEqual(min(scored), interferon["aurocLowest"])
        self.assertTrue(all(auc > 0.5 for auc in scored))

    def test_the_lupus_t_cell_auroc_matches_a_count_of_donor_pairs(self):
        # Perez's lupus donors against its healthy donors in T cells, from the donor table alone: each donor's earliest
        # sample with at least 20 cells, and the share of lupus-healthy pairs in which the lupus donor scores higher,
        # ties counting half.
        import pandas as pd

        table = pd.read_csv(
            contrasts.INTERFERON_TABLES[0],
            dtype={"donor_id": str, "sample": str, "disease": str},
            keep_default_na=False,
            na_values=[""],
        )
        units = table[
            table["storeId"].str.startswith("perez_2022")
            & (table["coarse_cell_type"] == "T cells")
            & (table["earliestSample"].astype(str).str.lower() == "true")
            & (table["cells"] >= 20)
            & table["interferonZScore"].notna()
        ]
        cases = units.loc[
            units["disease"] == "systemic lupus erythematosus", "interferonZScore"
        ].to_numpy()
        healthy = units.loc[units["disease"] == "normal", "interferonZScore"].to_numpy()
        pairs = (cases[:, None] > healthy[None, :]).sum() + 0.5 * (
            cases[:, None] == healthy[None, :]
        ).sum()
        (row,) = [
            r for r in self.contrasts["rows"] if r["disease"] == "systemic lupus erythematosus"
        ]
        self.assertAlmostEqual(
            row["auc"]["T cells"], pairs / (len(cases) * len(healthy)), places=12
        )
        self.assertEqual(round(row["auc"]["T cells"], 4), 0.9159)
