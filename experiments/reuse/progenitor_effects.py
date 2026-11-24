"""Donor-level summaries of the circulating progenitors pulled out of the human blood stores of CyteBase.

From the progenitor experiment's records: each study-scoped donor's earliest sample, its progenitors (cells the authors
labelled as haematopoietic stem or progenitor cells, or that carry two of CD34, SPINK2, PROM1 and CRHBP) out of its
peripheral-blood cells; against COVID-19 within studies and against age in healthy adults, as quasi-binomial models per
study pooled by random effects, under the default selection and three others; the subset analysis's clusters named by
the state panel, with each state's cells, studies and share of enriched reference cells; against the COVID-19 severity
the authors recorded, from healthy through mild to severe disease; every disease against the healthy donors of its own
study; COMBAT's donors by the cohort the authors recorded, from healthy volunteers through mild, severe and critical
COVID-19, and sepsis; each disease's P value adjusted across the diseases; COVID-19 against the pool of the chronic
diseases; COVID-19 adjusted for each donor's lymphocyte sequencing depth, from the catalogue experiment's interferon
screen, and for depth and sex; and the cells, times and bytes of the discovery, the extraction and the analysis. A
store that pools several source studies is split into them (SOURCE_STUDY_RECORDS). Writes
.build/derived/reuse/blood_progenitors.json.

  uv run python -m experiments.reuse.progenitor_effects [RECORDS_DIRECTORY]

RECORDS_DIRECTORY defaults to the experiment's records, records/blood_progenitors/.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.reuse import HERE
from experiments.reuse.blood_corpus import random_effects, study_effect
from experiments.runtime.blood_samples import (
    EXCLUDED_STORES,
    apply_cohort_records,
    apply_source_studies,
    earliest_samples,
    recorded_values,
)
from experiments.runtime.common import DERIVED, ROOT
from experiments.runtime.multiple_testing import benjamini_hochberg, holm

SUMMARY = DERIVED / "reuse" / "blood_progenitors.json"
RECORDS = HERE / "records" / "blood_progenitors"
# The catalogue experiment's interferon screen of the blood stores: per store, donor, sample and coarse cell type, the
# cells and their summed library sizes. Each donor table is checked against the sha256 its job recorded.
INTERFERON_RUNS = tuple(
    ROOT / f"experiments/catalogue/records/interferon_screen/{store_set}.json"
    for store_set in ("immune_cohorts", "other_blood_stores")
)
# The coarse cell types whose library sizes over cells give a donor's lymphocyte sequencing depth.
LYMPHOCYTES = ("T cells", "B cells", "NK and ILC")
# The diseases left out of the pool that COVID-19 is contrasted with: COVID-19 itself and the acute illnesses, sepsis
# and the respiratory system disorder of van der Wijst 2021's COVID-negative patients with moderate to critical illness.
# Every other disease of the donor table is chronic, autoimmune, post-infectious or clonal.
ACUTE = ("COVID-19", "sepsis", "respiratory system disorder")
GENES = ("CD34", "SPINK2", "PROM1", "CRHBP")
# The enriched circulating progenitors of Furer 2025: references in the subset, never donors. Like the other sorted or
# stimulated stores of EXCLUDED_STORES, they cannot give a donor's share of progenitors.
ENRICHED = ("furer_2025_human_chspcs_ultima_4a5b00e0", "furer_2025_human_chspcs_illumina_d9f3a82c")
# Stores whose cells are not circulating progenitors of blood donors, or whose gene panel is too small to join.
LEFT_OUT = {
    "chen_2022_global_dataset_infant_kmt2ar_b_all_cf83c98a": "infant KMT2A-rearranged leukaemia",
    "lim_2025_normal_malignant_cells_t_all_patients_extension_cohort_edb69e37": "childhood T-cell acute lymphoblastic leukaemia",
    "triana_2021_blood_bone_marrow_healthy_young_donor_d3566d6a": "460-gene panel",
}
# Collections that republish independent cohorts, one per store: Jin 2021's Schulte-Schrepping and Arunachalam
# cohorts. Every other collection is one study, its stores split by lineage or by donor group.
COHORTS_BY_STORE = ("b9fc3d70-5a72-4479-a046-c2cc1ab19efc",)
MINIMUM_CELLS = 2000
MINIMUM_CONTRAST_DONORS = 5
MINIMUM_AGE_DONORS, MINIMUM_AGE_SPAN, ADULT_AGE = 20, 20, 18
MISSING_DONORS = ("unknown", "na", "nan", "")
# The state whose genes' cluster means, each standardised across clusters, average highest names a cluster. Clusters
# named by mature lineage markers hold cells that passed the selection without being progenitors.
STATES = {
    "HSC/MPP": ("CRHBP", "HLF", "AVP", "MECOM", "MLLT3", "PROCR"),
    "lymphoid-primed": ("FLT3", "SATB1", "DNTT", "LTB", "CD7"),
    "B-lymphoid": ("IGLL1", "EBF1", "CD79A", "PAX5"),
    "myeloid": ("MPO", "ELANE", "AZU1", "CTSG", "CEBPA"),
    "erythroid": ("GATA1", "KLF1", "TFRC", "CA1"),
    "megakaryocyte": ("PF4", "PPBP", "ITGA2B", "GP9", "VWF"),
    "basophil/mast": ("HDC", "MS4A2", "CPA3"),
    "dendritic": ("IRF8", "SPIB"),
    "cycling": ("MKI67", "TOP2A"),
    "mature T/NK": ("CD3E", "NKG7"),
    "mature B": ("MS4A1",),
    "mature monocyte": ("CD14", "LYZ", "S100A9"),
}
MATURE = ("mature T/NK", "mature B", "mature monocyte")
# Per store of the COVID-19 studies: the column in which the authors recorded each sample's COVID-19 severity, and its
# values in three classes. Healthy donors are healthy; mild holds asymptomatic, mild and moderate disease, and severe
# holds severe and critical disease. A value the map leaves out names another condition or preparation: COMBAT's
# sepsis cohort and low-density neutrophil samples, which COHORT_RECORDS relabels and leaves out; influenza; and donors
# who tested negative.
LIU_SEVERITY = (
    "severity",
    {"": "healthy", "Moderate": "mild", "Severe": "severe", "Critical": "severe"},
)
SEVERITY = {
    "ahern_2022_combat_project_single_cell_19_sepsis_flu_patient_pbmcs_ebc2e1ff": (
        "Source",
        {"HV": "healthy", "COVID_MILD": "mild", "COVID_HCW_MILD": "mild", "COVID_SEV": "severe", "COVID_CRIT": "severe"},
    ),
    "cambridge_2021_single_cell_multi_omics_immune_response_covid_19_c7775e88": (
        "Status_on_day_collection_summary",
        {"Healthy": "healthy", "Asymptomatic": "mild", "Mild": "mild", "Moderate": "mild", "Severe": "severe", "Critical": "severe"},
    ),
    "liu_2021_time_resolved_systems_fatal_covid_19_adaptive_cells_21d3e683": LIU_SEVERITY,
    "liu_2021_time_resolved_systems_fatal_covid_19_innate_cells_30cd5311": LIU_SEVERITY,
    "ren_2021_large_scale_single_cell_covid_19_patients_9dbab10c": (
        "CoVID-19 severity",
        {"control": "healthy", "mild/moderate": "mild", "severe/critical": "severe"},
    ),
    "van_der_wijst_2021_type_i_interferon_patients_covid_19_01ad3cd7": (
        "COVID_severity",
        {"Healthy": "healthy", "Moderate_Positive": "mild", "Severe_Positive": "severe", "Critical_Positive": "severe", "Critical_Positive_anti-IFN": "severe"},
    ),
    "yoshida_2022_pbmc_2a498ace": (
        "COVID_severity",
        {"Healthy": "healthy", "Asymptomatic": "mild", "Mild": "mild", "Moderate": "mild", "Severe": "severe"},
    ),
    "jin_2021_individual_single_cell_rna_seq_data_schulte_schrepping_et_al_5e717147": (
        "COVID-19 Condition",
        {"control": "healthy", "mild_early": "mild", "mild_late": "mild", "severe_early": "severe", "severe_late": "severe"},
    ),
    "jin_2021_individual_single_cell_rna_seq_pbmc_data_arunachalam_et_al_59b69042": (
        "COVID-19 Condition",
        {"Healthy": "healthy", "COVID-19 Moderate": "mild", "COVID-19 Severe": "severe"},
    ),
}  # fmt: skip
SEVERITY_STEPS = {"healthy": 0.0, "mild": 1.0, "severe": 2.0}
NEWBORN = "newborn stage (0-28 days)"
# COMBAT's cohorts by the authors' Source column, in their order from healthy volunteers through COVID-19 of rising
# severity to sepsis and influenza, and each cohort's disease in the store after COHORT_RECORDS. COHORT_RECORDS leaves
# out the low-density neutrophil samples; a cohort with fewer than MINIMUM_CONTRAST_DONORS donors has no group.
COMBAT = "ahern_2022_combat_project_single_cell_19_sepsis_flu_patient_pbmcs_ebc2e1ff"
COMBAT_COHORTS = {
    "HV": "healthy",
    "COVID_HCW_MILD": "mild",
    "COVID_MILD": "mild",
    "COVID_SEV": "severe",
    "COVID_CRIT": "critical",
    "Sepsis": "sepsis",
    "Flu": "influenza",
}
COMBAT_DISEASES = {
    "healthy": "normal",
    "mild": "COVID-19",
    "severe": "COVID-19",
    "critical": "COVID-19",
    "sepsis": "sepsis",
    "influenza": "influenza",
}
COMBAT_STEPS = {"healthy": 0.0, "mild": 1.0, "severe": 2.0, "critical": 3.0}


def study_of(frame: pd.DataFrame) -> pd.Series:
    """Each row's study: its collection, or its store in a collection of COHORTS_BY_STORE."""
    by_store = frame["collectionId"].isin(COHORTS_BY_STORE)
    return frame["storeId"].where(by_store, frame["collectionId"])


def two_of_four(cells: pd.DataFrame) -> pd.Series:
    """Cells with at least two of the four marker genes detected; a gene a store lacks counts as undetected."""
    return (cells[list(GENES)].fillna(0) > 0).sum(axis=1) >= 2


def default_selection(cells: pd.DataFrame) -> pd.Series:
    """Cells labelled as progenitors or positive for two of the four genes, and not labelled malignant."""
    return (cells["labelled"].astype(bool) | two_of_four(cells)) & (
        cells["cell_type"] != "malignant cell"
    )


def donor_table(
    cells: pd.DataFrame, totals: pd.DataFrame, select=default_selection, stores=None
) -> pd.DataFrame:
    """Each study's donor's earliest sample with at least MINIMUM_CELLS cells: its cells, the progenitors
    ``select`` picks among them, and its disease, age and sex. The sorted, enriched or stimulated stores and the
    left-out stores stay out; ``stores``, when given, keeps only those stores, in the counts and in the cell totals
    alike."""
    out = {*EXCLUDED_STORES, *LEFT_OUT}
    if stores is not None:
        out |= set(totals["storeId"]) - set(stores)
    chosen = cells[select(cells) & ~cells["storeId"].isin(out)]
    chosen = chosen.assign(study=study_of(chosen))
    progenitors = chosen.groupby(["study", "donor_id", "sample"]).size().rename("progenitors")
    keys = ["study", "donor_id", "sample"]
    kept = totals[~totals["storeId"].isin(out)]
    samples = (
        kept.assign(study=study_of(kept))
        .groupby(keys, dropna=False)
        .agg(
            cells=("cells", "sum"),
            t0=("sampleTime0", "min"),
            t1=("sampleTime1", "min"),
            disease=("disease", "first"),
            stage=("development_stage", "first"),
            sex=("sex", "first"),
        )
        .reset_index()
        .merge(progenitors.reset_index(), how="left", on=keys)
    )
    samples["progenitors"] = samples["progenitors"].fillna(0).astype(int)
    samples["donor"] = samples["study"] + "\x1f" + samples["donor_id"].astype(str)
    samples["name"] = samples["sample"].astype(str)
    table = earliest_samples(samples)
    known = ~table["donor_id"].astype(str).str.strip().str.lower().isin(MISSING_DONORS)
    table = table[known & (table["cells"] >= MINIMUM_CELLS)].copy()
    table["age"] = pd.to_numeric(
        table["stage"].astype(str).str.extract(r"^(\d+)-year-old", expand=False), errors="coerce"
    )
    return table.drop(columns=["donor", "name"]).reset_index(drop=True)


def severity_samples(rows: list[dict]) -> pd.DataFrame:
    """Each sample of a store in SEVERITY with the value its severity column records and that value's class, None
    where the value names no class. ``rows`` are the severity record's cells per donor, sample, column and value."""
    frame = pd.DataFrame(rows)
    missing = sorted(set(SEVERITY) - set(frame["storeId"]))
    if missing:
        raise ValueError(f"the severity record holds no rows for {missing}")
    chosen = frame[
        [
            SEVERITY.get(s, ("",))[0] == c
            for s, c in zip(frame["storeId"], frame["column"], strict=True)
        ]
    ]
    keys = ["storeId", "donor_id", "sample"]
    values = chosen.groupby(keys)["value"].agg(["nunique", "first"]).reset_index()
    if (values["nunique"] > 1).any():
        raise ValueError(
            f"samples with two severity values: {values.loc[values['nunique'] > 1, keys].to_dict('records')}"
        )
    values = values.drop(columns="nunique").rename(columns={"first": "value"})
    values["severity"] = [
        SEVERITY[s][1].get(v) for s, v in zip(values["storeId"], values["value"], strict=True)
    ]
    return values


def study_severity(samples: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Each study's donor-sample with its severity class; a sample whose stores record two classes is an error."""
    collections = totals.drop_duplicates("storeId").set_index("storeId")["collectionId"]
    samples = samples.assign(collectionId=samples["storeId"].map(collections))
    samples["study"] = study_of(samples)
    grouped = samples.groupby(["study", "donor_id", "sample"])["severity"]
    if (grouped.nunique(dropna=False) > 1).any():
        raise ValueError("a donor-sample has two severity classes across its study's stores")
    return grouped.first().reset_index()


def fit(part: pd.DataFrame, covariate: str, adjust: tuple[str, ...] = ()) -> dict:
    return study_effect(
        part.rename(columns={"progenitors": "state", "cells": "lineage"}), covariate, adjust
    )


def contrast_frames(table: pd.DataFrame, disease: str):
    """Per study with MINIMUM_CONTRAST_DONORS donors with ``disease`` and as many healthy ones, the study, its donors
    with the disease, its healthy donors, and both together with ``case`` 1 for the disease and 0 for healthy. A study
    without a progenitor on one side has no finite estimate and is left out."""
    for study, part in table.groupby("study"):
        cases, healthy = part[part["disease"] == disease], part[part["disease"] == "normal"]
        if min(len(cases), len(healthy)) < MINIMUM_CONTRAST_DONORS:
            continue
        if cases["progenitors"].sum() == 0 or healthy["progenitors"].sum() == 0:
            continue
        yield study, cases, healthy, pd.concat([healthy.assign(case=0.0), cases.assign(case=1.0)])


def contrast_effects(table: pd.DataFrame, disease: str) -> list[dict]:
    """Per study of contrast_frames, the log odds of a cell being a progenitor in the disease against the healthy
    donors."""
    effects = []
    for study, cases, healthy, frame in contrast_frames(table, disease):
        effects.append(
            {
                "study": study,
                "cases": len(cases),
                "healthy": len(healthy),
                "casesPer10k": float(cases["progenitors"].sum() / cases["cells"].sum() * 1e4),
                "healthyPer10k": float(healthy["progenitors"].sum() / healthy["cells"].sum() * 1e4),
                **fit(frame, "case"),
            }
        )
    return effects


def severity_effects(table: pd.DataFrame, classes: pd.DataFrame) -> dict:
    """Per study with MINIMUM_CONTRAST_DONORS donors in each severity class (healthy donors, and donors with COVID-19 in
    each class of SEVERITY): each class's donors, newborns, cells, progenitors and progenitors per 10,000 cells. The
    log odds of a cell being a progenitor per step from healthy through mild to severe disease, and with severe
    against mild disease among the donors with COVID-19, as quasi-binomial models per study pooled by random effects.
    The studies with too few donors in a class, with each class's donors. ``table`` is a donor table and ``classes``
    each study's donor-samples with their class."""
    keys = ["study", "donor_id", "sample"]
    table = table[table["study"].isin(set(classes["study"]))]
    table = table[table["disease"].isin(["normal", "COVID-19"])]
    table = table.merge(classes, how="left", on=keys, indicator=True)
    unrecorded = table.loc[table["_merge"] == "left_only", keys]
    if not unrecorded.empty:
        raise ValueError(f"donors without a severity record: {unrecorded.to_dict('records')[:5]}")
    if ((table["disease"] == "normal") & (table["severity"] != "healthy")).any():
        raise ValueError("a healthy donor's severity record does not class it healthy")
    if ((table["disease"] == "COVID-19") & ~table["severity"].isin(["mild", "severe"])).any():
        raise ValueError(
            "a donor with COVID-19 has no severity class; COHORT_RECORDS should relabel or leave out its sample"
        )
    studies, trend, severe, short = [], [], [], []
    for study, part in table.groupby("study"):
        sizes = part["severity"].value_counts().reindex(list(SEVERITY_STEPS), fill_value=0)
        if sizes.min() < MINIMUM_CONTRAST_DONORS:
            short.append({"study": str(study), **{name: int(n) for name, n in sizes.items()}})
            continue
        entry = {"study": str(study)}
        for name in SEVERITY_STEPS:
            group = part[part["severity"] == name]
            entry[name] = {
                "donors": len(group),
                "newborns": int((group["stage"] == NEWBORN).sum()),
                "cells": int(group["cells"].sum()),
                "progenitors": int(group["progenitors"].sum()),
                "per10k": float(group["progenitors"].sum() / group["cells"].sum() * 1e4),
            }
        studies.append(entry)
        steps = part.assign(step=part["severity"].map(SEVERITY_STEPS))
        trend.append({"study": str(study), **fit(steps, "step")})
        cases = part[part["severity"] != "healthy"]
        severe.append(
            {
                "study": str(study),
                **fit(cases.assign(severe=(cases["severity"] == "severe").astype(float)), "severe"),
            }
        )
    return {
        "studies": studies,
        "trend": trend,
        "trendPooled": pooled(trend),
        "severeVsMild": severe,
        "severeVsMildPooled": pooled(severe),
        "severeVsMildLeaveOneOut": leave_one_out(severe),
        "tooFewDonors": short,
    }


def age_effects(table: pd.DataFrame) -> list[dict]:
    """Per study with MINIMUM_AGE_DONORS healthy adults of known sex spanning MINIMUM_AGE_SPAN years, the log odds of a
    cell being a progenitor per decade of age, adjusted for sex where it varies."""
    healthy = table[
        (table["disease"] == "normal")
        & table["age"].ge(ADULT_AGE)
        & table["sex"].isin(["male", "female"])
    ].copy()
    healthy["decades"] = healthy["age"] / 10
    healthy["male"] = (healthy["sex"] == "male").astype(float)
    effects = []
    for study, part in healthy.groupby("study"):
        span = part["age"].max() - part["age"].min()
        if (
            len(part) < MINIMUM_AGE_DONORS
            or span < MINIMUM_AGE_SPAN
            or part["progenitors"].sum() == 0
        ):
            continue
        adjust = ("male",) if part["male"].nunique() > 1 else ()
        effects.append(
            {
                "study": study,
                "donors": len(part),
                "youngest": float(part["age"].min()),
                "oldest": float(part["age"].max()),
                **fit(part, "decades", adjust),
            }
        )
    return effects


def pooled(effects: list[dict]) -> dict | None:
    """Random-effects pool of the per-study log odds, with its odds ratio, 95% interval and two-sided P value against
    an odds ratio of 1, or None with fewer than two studies."""
    if len(effects) < 2:
        return None
    pool = random_effects(
        np.array([e["effect"] for e in effects]), np.array([e["standardError"] for e in effects])
    )
    return {
        **pool,
        "ratio": float(np.exp(pool["effect"])),
        "ratioLow": float(np.exp(pool["low"])),
        "ratioHigh": float(np.exp(pool["high"])),
        "p": math.erfc(abs(pool["effect"] / pool["standardError"]) / math.sqrt(2)),
    }


def leave_one_out(effects: list[dict]) -> list[dict]:
    """The random-effects pool of the per-study log odds with each study left out in turn, naming the study left out;
    empty with fewer than three studies, where a pool without one study is no pool."""
    if len(effects) < 3:
        return []
    return [
        {"leftOut": left["study"], **pooled([e for e in effects if e is not left])}
        for left in effects
    ]


def weight_shares(effects: list[dict], diseases: list[str], tau2: float) -> dict[str, float]:
    """Each disease's share of a random-effects pool's inverse-variance weight, 1 / (standard error squared + tau2) per
    study contrast, with ``effects`` and ``diseases`` in the same order."""
    if len(effects) != len(diseases):
        raise ValueError(f"{len(effects)} study contrasts but {len(diseases)} diseases")
    weights = [1.0 / (e["standardError"] ** 2 + tau2) for e in effects]
    total = sum(weights)
    shares: dict[str, float] = {}
    for disease, weight in zip(diseases, weights, strict=True):
        shares[disease] = shares.get(disease, 0.0) + weight / total
    return shares


def estimate(effects: list[dict]) -> dict:
    """The estimate of a set of per-study log odds: their random-effects pool or, from one study, that study's own, each
    with its odds ratio, 95% interval and two-sided P value against an odds ratio of 1."""
    if not effects:
        raise ValueError("an estimate needs at least one study's effect")
    if len(effects) > 1:
        return pooled(effects)
    (only,) = effects
    effect, error = only["effect"], only["standardError"]
    low, high = effect - 1.959964 * error, effect + 1.959964 * error
    return {
        "effect": effect,
        "standardError": error,
        "low": low,
        "high": high,
        "tau2": 0.0,
        "studies": 1,
        "ratio": math.exp(effect),
        "ratioLow": math.exp(low),
        "ratioHigh": math.exp(high),
        "p": math.erfc(abs(effect / error) / math.sqrt(2)),
    }


def disease_effects(table: pd.DataFrame) -> dict[str, dict]:
    """For each disease of the donor table, its contrasts against the healthy donors of the same study
    (contrast_effects), the donors with the disease and the healthy donors they compare over those
    studies, and their estimate, whose P value is also adjusted across the diseases by Benjamini-Hochberg
    (bhAdjustedP) and by Holm (holmAdjustedP); a disease that no study can contrast is left out."""
    diseases = {}
    for disease in sorted(set(table["disease"].dropna().astype(str)) - {"normal"}):
        effects = contrast_effects(table, disease)
        if effects:
            diseases[disease] = {
                "studies": effects,
                "cases": sum(effect["cases"] for effect in effects),
                "healthy": sum(effect["healthy"] for effect in effects),
                "estimate": estimate(effects),
            }
    p = [entry["estimate"]["p"] for entry in diseases.values()]
    for entry, fdr, family in zip(diseases.values(), benjamini_hochberg(p), holm(p), strict=True):
        entry["estimate"]["bhAdjustedP"] = float(fdr)
        entry["estimate"]["holmAdjustedP"] = float(family)
    return diseases


def covid_against_chronic(diseases: dict[str, dict]) -> dict:
    """The COVID-19 estimate against the random-effects pool of every study contrast of the diseases outside ACUTE: the
    pool with its odds ratio and interval, each disease's share of the pool's weight, and the ratio of the two odds ratios with its 95% interval and two-sided
    Wald P value, the two log odds ratios' standard errors combined as independent."""
    covid = diseases["COVID-19"]["estimate"]
    chronic = sorted(set(diseases) - set(ACUTE))
    effects = [effect for disease in chronic for effect in diseases[disease]["studies"]]
    owners = [disease for disease in chronic for _ in diseases[disease]["studies"]]
    pool = pooled(effects)
    if pool is None:
        raise ValueError(f"the chronic diseases {chronic} hold fewer than two study contrasts")
    difference = covid["effect"] - pool["effect"]
    error = math.sqrt(covid["standardError"] ** 2 + pool["standardError"] ** 2)
    return {
        "diseases": chronic,
        "contrasts": len(effects),
        "pool": pool,
        "weightShares": weight_shares(effects, owners, pool["tau2"]),
        "ratioOfOddsRatios": math.exp(difference),
        "low": math.exp(difference - 1.959964 * error),
        "high": math.exp(difference + 1.959964 * error),
        "p": math.erfc(abs(difference / error) / math.sqrt(2)),
    }


def interferon_units() -> pd.DataFrame:
    """The interferon screen's cells and summed library sizes per store, donor, sample and coarse cell type, from both
    of its jobs, each donor table checked against the sha256 its job recorded. Sample names are read as written, so a
    sample named "nan" or "" keeps that name."""
    frames = []
    for run in INTERFERON_RUNS:
        record = json.loads(run.read_text())
        if record["status"] != "completed":
            raise ValueError(f"{run.name} did not complete")
        table = run.with_name(f"{run.stem}_donors.csv.gz")
        if (
            hashlib.sha256(table.read_bytes()).hexdigest()
            != record["result"]["tables"]["donors"]["sha256"]
        ):
            raise ValueError(f"{table.name} is not the donor table its job recorded")
        frames.append(
            pd.read_csv(
                table,
                usecols=[
                    "storeId",
                    "donor_id",
                    "sample",
                    "coarse_cell_type",
                    "cells",
                    "librarySize",
                ],
                dtype={"storeId": str, "donor_id": str, "sample": str, "coarse_cell_type": str},
                keep_default_na=False,
            )
        )
    units = pd.concat(frames, ignore_index=True)
    units[["cells", "librarySize"]] = units[["cells", "librarySize"]].astype(float)
    return units


def sample_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with its donor and sample as text, a missing sample as an empty name, to join tables on them."""
    return frame.assign(
        donor_id=frame["donor_id"].astype(str), sample=frame["sample"].fillna("").astype(str)
    )


def lymphocyte_depth(units: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Each study's donor-sample with the natural log of its lymphocyte sequencing depth: the library sizes of its T,
    B and NK/ILC cells over those cells, summed over the stores of its study. ``units`` are the interferon screen's,
    and ``totals`` the progenitor experiment's per store, donor and sample after the cohort and source-study records,
    which name each store's donor-sample's study as the donor table takes it."""
    keys = ["storeId", "donor_id", "sample"]
    studies = sample_keys(totals.assign(study=study_of(totals)))[[*keys, "study"]].drop_duplicates()
    lymphocytes = sample_keys(units[units["coarse_cell_type"].isin(LYMPHOCYTES)])
    depth = (
        lymphocytes.merge(studies, on=keys, how="inner")
        .groupby(["study", "donor_id", "sample"])[["librarySize", "cells"]]
        .sum()
        .reset_index()
    )
    depth["logDepth"] = np.log(depth["librarySize"] / depth["cells"])
    return depth[["study", "donor_id", "sample", "logDepth"]]


def covid_covariate_models(table: pd.DataFrame, depth: pd.DataFrame) -> dict:
    """COVID-19 against healthy donors in each study of contrast_frames, adjusted for each donor's log lymphocyte
    depth (lymphocyte_depth) centred on the study's mean, and in the studies with MINIMUM_CONTRAST_DONORS donors of
    recorded male or female sex on each side and a progenitor on both, for depth and sex together; each set of study
    effects with its donors and estimate. A donor of a contrast without a lymphocyte depth is an error."""
    keyed = sample_keys(table).merge(depth, on=["study", "donor_id", "sample"], how="left")
    adjusted: dict[str, list[dict]] = {
        "lymphocyteDepthAdjusted": [],
        "lymphocyteDepthAndSexAdjusted": [],
    }
    for study, _, _, frame in contrast_frames(keyed, "COVID-19"):
        missing = frame["logDepth"].isna()
        if missing.any():
            raise ValueError(
                f"{study}: {int(missing.sum())} donors of the COVID-19 contrast have no lymphocyte depth, first "
                f"{frame.loc[missing, 'donor_id'].iloc[0]!r}"
            )
        centred = frame.assign(logDepth=frame["logDepth"] - frame["logDepth"].mean())
        adjusted["lymphocyteDepthAdjusted"].append(
            {"study": study, **fit(centred, "case", ("logDepth",))}
        )
        known = frame[frame["sex"].isin(["male", "female"])]
        sides = [known[known["case"] == case] for case in (0.0, 1.0)]
        if min(len(side) for side in sides) < MINIMUM_CONTRAST_DONORS:
            continue
        if min(side["progenitors"].sum() for side in sides) == 0:
            continue
        if known["sex"].nunique() < 2:
            raise ValueError(
                f"{study}: every donor of recorded sex in the COVID-19 contrast is {known['sex'].iloc[0]}"
            )
        known = known.assign(
            logDepth=known["logDepth"] - known["logDepth"].mean(),
            male=(known["sex"] == "male").astype(float),
        )
        adjusted["lymphocyteDepthAndSexAdjusted"].append(
            {"study": study, **fit(known, "case", ("logDepth", "male"))}
        )
    return {
        name: {
            "studyEffects": effects,
            "donors": sum(effect["donors"] for effect in effects),
            "estimate": estimate(effects),
        }
        for name, effects in adjusted.items()
    }


def disease_studies(diseases: dict[str, dict]) -> int:
    """The studies that contrast at least one disease with their own healthy donors."""
    return len({effect["study"] for disease in diseases.values() for effect in disease["studies"]})


def disease_donors(table: pd.DataFrame, diseases: dict[str, dict]) -> int:
    """The donors the disease contrasts compare: in each study that contrasts a disease, its donors with the disease and
    its healthy donors, each donor counted once however many of its study's diseases it is compared with."""
    donors = set()
    for disease, entry in diseases.items():
        for effect in entry["studies"]:
            part = table[
                (table["study"] == effect["study"]) & table["disease"].isin([disease, "normal"])
            ]
            if len(part) != effect["cases"] + effect["healthy"]:
                raise ValueError(
                    f"{effect['study']} holds {len(part)} donors for {disease}, not the {effect['cases']} cases and "
                    f"{effect['healthy']} healthy donors its contrast compared"
                )
            donors.update(zip(part["study"], part["donor_id"].astype(str), strict=True))
    return len(donors)


def combat_cohorts(table: pd.DataFrame, values: pd.DataFrame, study: str) -> dict:
    """COMBAT's donors of the donor table, whose study is ``study``, by the cohort the authors recorded for their sample
    in ``values`` (COMBAT_COHORTS): per cohort with at least MINIMUM_CONTRAST_DONORS donors, each donor's cells and
    progenitors and the cohort's progenitors per 10,000 cells, its donors' cells pooled; the log odds of a cell being a
    progenitor per step from healthy volunteers through mild, severe and critical COVID-19 (COMBAT_STEPS), and with
    sepsis against healthy volunteers, as quasi-binomial models, each with its estimate; and the cohorts with too few
    donors. A sample without its cohort, a cohort COMBAT_COHORTS does not name, or a donor whose disease is not its
    cohort's is an error."""
    donors = table[table["study"] == study].reset_index(drop=True)
    if donors.empty:
        raise ValueError(f"the donor table holds no donor of COMBAT's study {study!r}")
    _, found = recorded_values(donors, np.arange(len(donors)), values, COMBAT, "Source")
    unknown = sorted(set(found) - set(COMBAT_COHORTS))
    if unknown:
        raise ValueError(f"COMBAT samples of cohorts COMBAT_COHORTS does not name: {unknown}")
    donors["cohort"] = [COMBAT_COHORTS[value] for value in found]
    mismatched = donors[donors["disease"].astype(str) != donors["cohort"].map(COMBAT_DISEASES)]
    if not mismatched.empty:
        raise ValueError(
            "COMBAT donors whose disease is not their cohort's: "
            f"{mismatched[['donor_id', 'disease', 'cohort']].to_dict('records')[:3]}"
        )
    groups, short = [], []
    for cohort in dict.fromkeys(COMBAT_COHORTS.values()):
        part = donors[donors["cohort"] == cohort]
        if len(part) < MINIMUM_CONTRAST_DONORS:
            short.append({"cohort": cohort, "donors": len(part)})
            continue
        groups.append(
            {
                "cohort": cohort,
                "donors": [
                    {"cells": int(cells), "progenitors": int(progenitors)}
                    for cells, progenitors in zip(part["cells"], part["progenitors"], strict=True)
                ],
                "per10k": float(part["progenitors"].sum() / part["cells"].sum() * 1e4),
            }
        )
    needed = {*COMBAT_STEPS, "sepsis"} - {group["cohort"] for group in groups}
    if needed:
        raise ValueError(
            f"COMBAT has too few donors in {sorted(needed)} for its trend and its sepsis contrast"
        )
    steps = donors[donors["cohort"].isin(list(COMBAT_STEPS))]
    sides = donors[donors["cohort"].isin(["healthy", "sepsis"])]
    return {
        "cohorts": groups,
        "tooFewDonors": short,
        "trend": estimate([fit(steps.assign(step=steps["cohort"].map(COMBAT_STEPS)), "step")]),
        "sepsis": estimate(
            [fit(sides.assign(case=(sides["cohort"] == "sepsis").astype(float)), "case")]
        ),
    }


def cluster_counts(names: dict[int, str]) -> dict[str, int]:
    """How many clusters ``names`` (cluster to state, as state_names gives it) names by a mature lineage, and how many
    by a progenitor state."""
    mature = sum(state in MATURE for state in names.values())
    return {"matureClusters": mature, "progenitorClusters": len(names) - mature}


def state_names(panel: pd.DataFrame) -> dict[int, str]:
    """Each cluster's state: the STATES entry whose measured genes' cluster means, standardised across clusters,
    average highest. ``panel`` has a row per cluster with <gene>_mean columns."""
    means = panel.set_index("cluster")[[c for c in panel.columns if c.endswith("_mean")]]
    means.columns = [c.removesuffix("_mean") for c in means.columns]
    spread = means.std(ddof=0).replace(0, np.nan)
    z = (means - means.mean()) / spread
    scores = pd.DataFrame(
        {
            state: z[[g for g in genes if g in z]].mean(axis=1)
            for state, genes in STATES.items()
            if any(g in z for g in genes)
        }
    )
    return {int(cluster): str(state) for cluster, state in scores.idxmax(axis=1).items()}


def source_counts(
    discovery: list[dict],
    extracted: dict[str, dict],
    analysed: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """The stores the discovery read, and their cells at each step, for the enriched stores and for all other blood
    stores: read by the discovery, with a progenitor label or any of the four genes, selected by the default selection
    outside the left-out stores, extracted, analysed, and placed in a progenitor state; the extracted cells left out of
    the analysis; and the shares of the cells read that carry a label or a gene and that the selection keeps.
    ``analysed`` has a row per analysed cell with its
    ``dataset`` and ``state``; ``candidates`` is the discovery's cell table."""
    steps = (
        "stores",
        "cellsRead",
        "candidates",
        "selected",
        "extracted",
        "analysed",
        "progenitors",
    )
    counts = {source: dict.fromkeys(steps, 0) for source in ("blood", "enriched")}

    def source(store: str) -> dict[str, int]:
        return counts["enriched" if store in ENRICHED else "blood"]

    for store in discovery:
        source(store["storeId"])["stores"] += 1
        source(store["storeId"])["cellsRead"] += int(store["cellsRead"])
        source(store["storeId"])["candidates"] += int(store["candidates"])
    selected = candidates[default_selection(candidates) & ~candidates["storeId"].isin(LEFT_OUT)]
    for store, cells in selected.groupby("storeId").size().items():
        source(str(store))["selected"] += int(cells)
    for store, entry in extracted.items():
        source(store)["extracted"] += int(entry["cells"])
    for store, part in analysed.groupby("dataset"):
        source(str(store))["analysed"] += len(part)
        source(str(store))["progenitors"] += int((~part["state"].isin(MATURE)).sum())
    for count in counts.values():
        count["extractedNotAnalysed"] = count["extracted"] - count["analysed"]
        count["candidateShare"] = count["candidates"] / count["cellsRead"]
        count["selectedShare"] = count["selected"] / count["cellsRead"]
    return counts


def state_shares(named: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Each progenitor state's share of the cells in progenitor states, for the enriched stores and for all other blood
    stores. ``named`` has a row per analysed cell with its ``dataset`` and ``state``."""
    progenitors = named[~named["state"].isin(MATURE)]
    sources = np.where(progenitors["dataset"].isin(ENRICHED), "enriched", "blood")
    states = [state for state in STATES if state not in MATURE]
    shares = {}
    for source in ("blood", "enriched"):
        part = progenitors[sources == source]
        if part.empty:
            raise ValueError(f"no analysed cell of the {source} stores is in a progenitor state")
        counts = part["state"].value_counts()
        shares[source] = {state: float(counts.get(state, 0) / len(part)) for state in states}
    return shares


def largest_share_difference(shares: dict[str, dict[str, float]]) -> dict:
    """The progenitor state whose shares differ most between the blood stores and the enriched stores, with the
    difference in percentage points: how closely the progenitors of unselected blood match the purified ones."""
    blood, enriched = shares["blood"], shares["enriched"]
    state = max(blood, key=lambda name: abs(blood[name] - enriched[name]))
    return {"state": state, "percentagePoints": 100 * abs(blood[state] - enriched[state])}


def state_selection(named: pd.DataFrame, states: list[str]):
    """A selection of the cells the subset analysis placed in ``states``, found by store and row."""
    chosen = named[named["state"].isin(states)]
    keys = set(zip(chosen["dataset"], chosen["source_row"]))

    def select(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [key in keys for key in zip(frame["storeId"], frame["source_row"])], index=frame.index
        )

    return select


def selections(cells: pd.DataFrame, named: pd.DataFrame) -> dict:
    """The default selection and five others: in the analysed stores, the cells the subset analysis placed in
    progenitor states, in the stem and multipotent state alone, or in progenitor states other than the megakaryocytic
    one, which platelet-like cells could fill; and in all stores, the authors' labels alone and the four-gene rule
    alone."""
    analysed = set(named["dataset"])
    progenitor_states = [state for state in STATES if state not in MATURE]
    not_malignant = cells["cell_type"] != "malignant cell"
    return {
        "default": (default_selection, None),
        "progenitorStates": (state_selection(named, progenitor_states), analysed),
        "stemAndMultipotent": (state_selection(named, ["HSC/MPP"]), analysed),
        "withoutMegakaryocytic": (
            state_selection(
                named, [state for state in progenitor_states if state != "megakaryocyte"]
            ),
            analysed,
        ),
        "labelsOnly": (lambda frame: frame["labelled"].astype(bool) & not_malignant, None),
        "markersOnly": (lambda frame: two_of_four(frame) & not_malignant, None),
    }


def state_covid_effects(cells: pd.DataFrame, totals: pd.DataFrame, named: pd.DataFrame) -> dict:
    """For each progenitor state, in the analysed stores: its cells, and the COVID-19 contrasts and their pool when
    only that state's cells count as a donor's progenitors."""
    analysed = set(named["dataset"])
    effects = {}
    for state in STATES:
        if state in MATURE:
            continue
        table = donor_table(cells, totals, state_selection(named, [state]), analysed)
        covid = contrast_effects(table, "COVID-19")
        effects[state] = {
            "cells": int((named["state"] == state).sum()),
            "covid": covid,
            "covidPooled": pooled(covid),
        }
    return effects


def covid_composition(cells: pd.DataFrame, totals: pd.DataFrame, named: pd.DataFrame) -> dict:
    """The make-up of the progenitors of donors with COVID-19 and of healthy donors, in the analysed stores. A donor's
    progenitors are its cells in progenitor states. Per study with MINIMUM_CONTRAST_DONORS donors on each side and
    progenitors on both: each state's share of each side's progenitors, the cells of the side's donors pooled; the
    shares averaged over those studies, each study weighing the same. Per state, COVID-19 against healthy donors in the
    odds of a progenitor being in that state, quasi-binomial per study pooled by random effects, in the studies whose
    two sides both hold progenitors in the state and outside it."""
    analysed = set(named["dataset"])
    states = [state for state in STATES if state not in MATURE]
    keys = ["study", "donor_id", "sample"]
    donors = donor_table(cells, totals, state_selection(named, states), analysed)
    donors = donors[donors["progenitors"] > 0].rename(columns={"progenitors": "inStates"})
    for state in states:
        counts = donor_table(cells, totals, state_selection(named, [state]), analysed)
        donors = donors.merge(
            counts[keys + ["progenitors"]].rename(columns={"progenitors": state}),
            on=keys,
            how="left",
        )
    donors[states] = donors[states].fillna(0)
    shares: dict[str, list[dict[str, float]]] = {"normal": [], "COVID-19": []}
    studies = []
    for study, part in donors.groupby("study"):
        sides = {disease: part[part["disease"] == disease] for disease in shares}
        if min(len(side) for side in sides.values()) < MINIMUM_CONTRAST_DONORS:
            continue
        studies.append(str(study))
        for disease, side in sides.items():
            shares[disease].append({s: side[s].sum() / side["inStates"].sum() for s in states})
    tests = {}
    for state in states:
        effects = []
        for study in studies:
            part = donors[donors["study"] == study]
            cases, healthy = part[part["disease"] == "COVID-19"], part[part["disease"] == "normal"]
            # A finite estimate needs, on each side, progenitors in the state and progenitors outside it.
            inside = [side[state].sum() for side in (cases, healthy)]
            outside = [(side["inStates"] - side[state]).sum() for side in (cases, healthy)]
            if min(inside) == 0 or min(outside) == 0:
                continue
            frame = pd.concat([healthy.assign(case=0.0), cases.assign(case=1.0)])
            effects.append(
                fit(frame.assign(progenitors=frame[state], cells=frame["inStates"]), "case")
            )
        tests[state] = pooled(effects)
    return {
        "studies": studies,
        "healthy": {s: float(np.mean([share[s] for share in shares["normal"]])) for s in states},
        "covid": {s: float(np.mean([share[s] for share in shares["COVID-19"]])) for s in states},
        "tests": tests,
    }


def summarise(records: Path) -> dict:
    cells = pd.read_parquet(records / "progenitor_cells.parquet")
    totals = pd.read_parquet(records / "progenitor_donors.parquet")
    panel = pd.read_parquet(records / "progenitor_cluster_panel.parquet")
    table_cells = pd.read_parquet(records / "progenitor_cell_table.parquet")
    discovery = json.loads((records / "discovery.json").read_text())["result"]
    extractions = [
        json.loads(p.read_text())["result"] for p in sorted(records.glob("extraction_*.json"))
    ]
    analysis = json.loads((records / "analysis.json").read_text())["result"]
    rows = json.loads((records / "severity.json").read_text())["result"]["rows"]
    sources = pd.DataFrame(
        json.loads((records / "source_studies.json").read_text())["result"]["rows"]
    )
    severity = severity_samples(rows)
    # Donor-level analyses take the disease of each sample from its cohort where COHORT_RECORDS lists the store, leave
    # out the samples of another preparation, and take each source study of a store in SOURCE_STUDY_RECORDS as a
    # study of its own; the cell-level counts, states and shares keep every cell the discovery read.
    donor_cells, _ = apply_cohort_records(cells, pd.DataFrame(rows))
    donor_totals, cohorts = apply_cohort_records(totals, pd.DataFrame(rows))
    donor_cells, _ = apply_source_studies(donor_cells, sources)
    donor_totals, source_studies = apply_source_studies(donor_totals, sources)
    default_table = donor_table(donor_cells, donor_totals)
    (combat_study,) = study_of(donor_totals[donor_totals["storeId"] == COMBAT]).unique()
    names = state_names(panel)
    table_cells["state"] = table_cells["leiden_1_0"].map(names)
    if table_cells["state"].isna().any():
        raise ValueError("a cluster of the cell table has no row in the cluster panel")
    states = table_cells.assign(reference=table_cells["dataset"].isin(ENRICHED)).merge(
        cells[["storeId", "source_row", "collectionId"]],
        how="left",
        left_on=["dataset", "source_row"],
        right_on=["storeId", "source_row"],
    )
    if states["collectionId"].isna().any():
        raise ValueError("an analysed cell is not in the discovery table")
    states["study"] = study_of(states)
    by_state = states.groupby("state").agg(
        cells=("dataset", "size"),
        clusters=("leiden_1_0", "nunique"),
        studies=("study", "nunique"),
        referenceShare=("reference", "mean"),
    )
    variants = {}
    for name, (select, stores) in selections(donor_cells, table_cells).items():
        table = donor_table(donor_cells, donor_totals, select, stores)
        covid, ages = contrast_effects(table, "COVID-19"), age_effects(table)
        variants[name] = {
            "donors": len(table),
            "studies": int(table["study"].nunique()),
            "progenitors": int(table["progenitors"].sum()),
            "cells": int(table["cells"].sum()),
            "covid": covid,
            "covidPooled": pooled(covid),
            "age": ages,
            "agePooled": pooled(ages),
        }
    extraction_stores = {
        name: entry for result in extractions for name, entry in result["stores"].items()
    }
    shares = state_shares(table_cells)
    diseases = disease_effects(default_table)
    return {
        "discovery": {
            "stores": len(discovery["stores"]),
            "cellsRead": int(sum(s["cellsRead"] for s in discovery["stores"])),
            "labelledProgenitors": int(sum(s["labelledProgenitors"] for s in discovery["stores"])),
            "candidates": int(sum(s["candidates"] for s in discovery["stores"])),
            "wallSeconds": discovery["wallSeconds"],
            "responseBytes": int(
                sum(s["io"]["http"]["responseBytes"] for s in discovery["stores"])
            ),
        },
        "extraction": {
            "jobs": len(extractions),
            "stores": len(extraction_stores),
            "cells": int(sum(e["cells"] for e in extraction_stores.values())),
            "longestJobSeconds": max(r["wallSeconds"] for r in extractions),
            "responseBytes": int(
                sum(
                    e[phase]["io"]["http"]["responseBytes"]
                    for e in extraction_stores.values()
                    for phase in ("creation", "export")
                )
            ),
        },
        "analysis": {
            "stores": int(table_cells["dataset"].nunique()),
            "cells": analysis["cells"],
            "genes": analysis["genes"],
            "studies": analysis["annotation"]["studies"],
            "donors": analysis["annotation"]["studyScopedDonors"],
            "seconds": analysis["joining"]["elapsedSeconds"]
            + analysis["analysis"]["elapsedSeconds"],
            "peakBytes": max(
                analysis[phase]["resources"]["operationPeakBytes"]
                for phase in ("joining", "analysis")
            ),
            "clusters": int(table_cells["leiden_1_0"].nunique()),
            **cluster_counts(names),
        },
        "clusterStates": names,
        "states": by_state.reset_index().to_dict("records"),
        "progenitorCells": int((~table_cells["state"].isin(MATURE)).sum()),
        # The analysed cells in clusters named by mature lineage markers.
        "matureCells": int(table_cells["state"].isin(MATURE).sum()),
        "sources": source_counts(discovery["stores"], extraction_stores, table_cells, cells),
        "variants": variants,
        "stateCovid": state_covid_effects(donor_cells, donor_totals, table_cells),
        "covidComposition": covid_composition(donor_cells, donor_totals, table_cells),
        "covidSeverity": severity_effects(default_table, study_severity(severity, totals)),
        "diseases": diseases,
        "covidAgainstChronic": covid_against_chronic(diseases),
        "covidCovariateModels": covid_covariate_models(
            default_table, lymphocyte_depth(interferon_units(), donor_totals)
        ),
        # The studies that contrast a disease with their own healthy donors, across the diseases, and their donors.
        "diseaseStudies": disease_studies(diseases),
        "diseaseDonors": disease_donors(default_table, diseases),
        "combatCohorts": combat_cohorts(default_table, pd.DataFrame(rows), combat_study),
        "cohortRecords": cohorts,
        "sourceStudies": source_studies,
        "stateShares": shares,
        "stateShareLargestDifference": largest_share_difference(shares),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "records",
        type=Path,
        nargs="?",
        default=RECORDS,
        help="folder with the progenitor experiment's records",
    )
    args = parser.parse_args()
    summary = summarise(args.records)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    default = summary["variants"]["default"]
    for key in ("covidPooled", "agePooled"):
        p = default[key]
        print(
            key,
            round(float(np.exp(p["effect"])), 3),
            round(float(np.exp(p["low"])), 3),
            round(float(np.exp(p["high"])), 3),
            p["studies"],
        )
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
