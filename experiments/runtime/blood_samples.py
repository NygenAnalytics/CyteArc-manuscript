"""The samples that donor-level analyses of CyteBase blood stores keep.

Stores whose cells were sorted, enriched or stimulated by design leave whole. Within the other stores, cells of
sorted, enriched or stimulated samples leave by the stores' own labels, and each donor keeps one sample: the earliest
by the store's time columns, or the largest when it records no time. The rules were fixed from each store's own cell
columns before any donor-level share was computed.
"""

import numpy as np
import pandas as pd

# Samples, fixed from each store's own cell columns before any donor-level share. Stores whose cells were sorted,
# enriched or stimulated by design leave the donor table whole.
EXCLUDED_STORES = {
    "furer_2025_human_chspcs_ultima_4a5b00e0": "circulating haematopoietic progenitors, enriched",
    "furer_2025_human_chspcs_illumina_d9f3a82c": "circulating haematopoietic progenitors, enriched",
    "jardine_2021_cd34_fetal_bone_marrow_fetal_liver_cord_blood_cite_seq_471647b3": "CD34+ cells",
    "knight_2026_innate_lymphocytes_4057e00c": "sorted (sorted_celltype CD45posCD3negCD7pos, DCsubsets)",
    "knight_2026_dendritic_cells_a1058181": "dendritic cells only",
    "guerreromurillo_2024_cd19_carpos_carneg_t_cells_460dfe21": "T cells of CAR T cell products",
    "szabo_2019_single_cell_transcriptomics_signatures_health_disease_01209dce": "T cells only, rested or activated",
    "ng_2026_t_cells_small_cell_lung_cancer_biopsies_b9735320": "T cells only",
    "nose_2026_t_cells_gastric_cancer_patients_ca140407": "T cells only",
    "rodriguezubreva_2022_steady_state_b_cells_scrna_a5d95a42": "B cells only",
    "rodriguezubreva_2022_activated_pbmcs_twins_scrna_d7d7e89c": "activated PBMCs without a stimulation label",
}
# Per store: cell values that leave the donor table (sorted, enriched or stimulated samples); the column naming a
# donor's samples; and the time columns ordering them, compared in turn. A time column is read as a number, by its
# place in ``order``, or by the number ``pattern`` captures (``default`` when it captures none); a time not recorded
# sorts last. Each donor keeps its earliest sample, or its largest when no time is recorded. Without a sample column
# the first time column names the samples; a store without a rule has one sample per donor.
SOUND_LIFE = {
    "sample": "sample.sampleKitGuid",
    "time": (
        {"column": "sample.drawYear"},
        {"column": "sample.visitName", "pattern": r"Day (\d+)", "default": 0},
    ),
}
LIU_2021 = {"time": ({"column": "timepoint", "pattern": r"T(\d+)"},)}
XU_2026 = {"sample": "sample", "time": ({"column": "timepoint", "order": ("Base", "D28", "M3")},)}
SAMPLE_RULES = {
    **{
        store: SOUND_LIFE
        for store in (
            "gong_2025_sound_life_young_adult_females_cmv_negative_46104f0b",
            "gong_2025_sound_life_older_adult_females_cmv_positive_aa32098f",
            "gong_2025_sound_life_older_adult_males_cmv_negative_47e00f43",
            "gong_2025_sound_life_young_adult_males_cmv_negative_420d0f5d",
            "gong_2025_sound_life_older_adult_females_cmv_negative_3cb646ab",
            "gong_2025_sound_life_older_adult_males_cmv_positive_0c0c49a2",
            "gong_2025_sound_life_young_adult_females_cmv_positive_0a8e3443",
            "gong_2025_sound_life_young_adult_males_cmv_positive_5fde5c9c",
        )
    },
    "he_2025_altra_all_samples_789ad837": {
        "sample": "sample.sampleKitGuid",
        "time": ({"column": "sample.daysSinceFirstVisit"},),
    },
    "ren_2021_large_scale_single_cell_covid_19_patients_9dbab10c": {
        "exclude": {
            "Sample type": (
                "B cells sorted from frozen PBMC (MACS, STEMCELL 19054)",
                "CD3+ T cell and CD19+ B cell sorted from fresh PBMC (FACS)",
                "CD19+ B cell sorted from fresh PBMC (MACS)",
                "CD3+ T cell sorted from fresh PBMC (FACS)",
                "CD19+ B cell sorted from fresh PBMC (FACS)",
            )
        },
        "sample": "sampleID",
        "time": ({"column": "Sampling day (Days after symptom onset)"},),
    },
    "ahern_2022_combat_project_single_cell_19_sepsis_flu_patient_pbmcs_ebc2e1ff": {
        "sample": "scRNASeq_sample_ID",
        "time": ({"column": "TimeSinceOnset"},),
    },
    "cambridge_2021_single_cell_multi_omics_immune_response_covid_19_c7775e88": {
        "exclude": {"time_after_LPS": ("90m", "10h")},
        "sample": "sample_id",
        "time": ({"column": "Collection_Day", "pattern": r"D(\d+)"},),
    },
    "van_der_wijst_2021_type_i_interferon_patients_covid_19_01ad3cd7": {
        "time": ({"column": "days_from_start"},),
    },
    "liu_2021_time_resolved_systems_fatal_covid_19_adaptive_cells_21d3e683": LIU_2021,
    "liu_2021_time_resolved_systems_fatal_covid_19_innate_cells_30cd5311": LIU_2021,
    "ivanova_2023_circulating_immune_cells_cv19_infection_vaccination_hc_242c6e7f": {
        "sample": "sampleName_with_day",
        "time": ({"column": "Day"},),
    },
    "xu_2026_myeloid_cells_3d8d8c97": XU_2026,
    "xu_2026_t_cells_f2b69d9b": XU_2026,
    "xu_2026_b_cells_ee782eb7": XU_2026,
    "hao_2021_nygc_multimodal_pbmc_ed5d841d": {"time": ({"column": "time"},)},
    "rabadam_2024_citeseq_jdm_pbmcs_a199ca73": {
        "time": ({"column": "visit", "pattern": r"V(\d+)"},),
    },
    "jin_2021_individual_single_cell_rna_seq_data_schulte_schrepping_et_al_5e717147": {
        "sample": "sample",
        "time": (
            {
                "column": "COVID-19 Condition",
                "order": ("control", "mild_early", "severe_early", "mild_late", "severe_late"),
            },
        ),
    },
    "lee_2020_immunophenotyping_covid_19_development_severe_covid_19_de2c780c": {
        "sample": "Sample ID",
        "time": ({"column": "Hospital day"},),
    },
    "chan_2020_pbmcs_fa8605cf": {"time": ({"column": "time_point", "pattern": r"D(\d+)"},)},
    "perez_2022_multiplexed_scrna_1_2_million_pbmcs_adult_lupus_samples_218acb0f": {
        "sample": "sample_uuid",
    },
    "marteau_2026_single_cell_colorectal_cancer_atlas_core_atlas_05a8c945": {
        # "naive" is the one value that names no enriched population.
        "exclude": {
            "enrichment_cell_types": (
                "CD45+",
                "CD45+/CD45-",
                "CD66b+",
                "CD8+",
                "CD3+/CD4+/CD25-",
                "CD3+/CD4+/CD25high",
                "CD3+/CD4+/CD25int",
                "CD3+/CD4+",
            )
        },
        "sample": "sample_id",
    },
    "rodriguezubreva_2022_activated_pbmcs_expanded_cohort_cite_seq_3c75a463": {
        "exclude": {"stimulation": ("CD3", "CD40L")},
        "sample": "sample",
    },
}


# Cohorts a store labels with a disease its authors' own column contradicts, and samples of another preparation, found in
# that column after the screens ran: per store, the column, the disease of the donors of each listed value, and the
# values whose samples leave. COMBAT's store labels its sepsis cohort COVID-19, and its low-density neutrophil samples
# are a separate preparation from donors with COVID-19. Donor-level analyses apply these to their tables with the
# column's value for each sample, as blood_progenitors.severity reads it.
COHORT_RECORDS = {
    "ahern_2022_combat_project_single_cell_19_sepsis_flu_patient_pbmcs_ebc2e1ff": {
        "column": "Source",
        "disease": {"Sepsis": "sepsis"},
        "exclude": ("COVID_LDN",),
    },
}


def apply_cohort_records(
    frame: pd.DataFrame, values: pd.DataFrame, rules: dict = COHORT_RECORDS
) -> tuple[pd.DataFrame, dict]:
    """``frame`` with the disease of each sample of a store in ``rules`` set from its authors' column and the samples of
    the excluded values left out, and per store the samples and donors relabelled and left out. ``frame`` has a row
    per cell, sample or unit with ``storeId``, ``donor_id``, ``sample`` and ``disease``; ``values`` has a row per
    store, donor, sample and column with the column's ``value`` in that sample. A sample of a store in ``rules``
    without one value of its column is an error."""
    out = frame.copy()
    if isinstance(out["disease"].dtype, pd.CategoricalDtype):
        out["disease"] = out["disease"].astype(object)
    keep = np.ones(len(out), dtype=bool)
    changes: dict = {}
    for store, rule in rules.items():
        mine = np.flatnonzero((out["storeId"] == store).to_numpy())
        if not len(mine):
            continue
        keys, found = recorded_values(out, mine, values, store, rule["column"])
        entry: dict = {"relabelled": {}, "excluded": {}}
        for value, disease in rule["disease"].items():
            hit = found == value
            out.iloc[mine[hit], out.columns.get_loc("disease")] = disease
            entry["relabelled"][disease] = sample_counts(keys, hit)
        hit = np.isin(found, list(rule["exclude"]))
        keep[mine[hit]] = False
        entry["excluded"] = sample_counts(keys, hit)
        changes[store] = entry
    return out[keep], changes


# Stores that pool the donors of several source studies, with the column that names each sample's source study.
# Donor-level analyses take each source study as a study of its own, so that no contrast sets the donors of one
# source study against those of another: in the colorectal cancer atlas, the donors of peripheral blood with cancer
# and the healthy ones come from different source studies.
SOURCE_STUDY_RECORDS = {
    "marteau_2026_single_cell_colorectal_cancer_atlas_core_atlas_05a8c945": "study_id",
}


def apply_source_studies(
    frame: pd.DataFrame, values: pd.DataFrame, rules: dict = SOURCE_STUDY_RECORDS
) -> tuple[pd.DataFrame, dict]:
    """``frame`` with the collection of each sample of a store in ``rules`` set to '<collection>/<source study>', the
    source study read from the column the rule names, and per store the samples and donors of each source study.
    ``frame`` has a row per cell, sample or unit with ``storeId``, ``collectionId``, ``donor_id`` and ``sample``;
    ``values`` has a row per store, donor, sample and column with the column's ``value`` in that sample. A sample of a
    store in ``rules`` without one value of its column is an error."""
    out = frame.copy()
    if isinstance(out["collectionId"].dtype, pd.CategoricalDtype):
        out["collectionId"] = out["collectionId"].astype(object)
    sources: dict = {}
    for store, column in rules.items():
        mine = np.flatnonzero((out["storeId"] == store).to_numpy())
        if not len(mine):
            continue
        keys, found = recorded_values(out, mine, values, store, column)
        collections = out["collectionId"].iloc[mine].astype(str).to_numpy()
        out.iloc[mine, out.columns.get_loc("collectionId")] = [
            f"{collection}/{source}" for collection, source in zip(collections, found, strict=True)
        ]
        sources[store] = {
            str(source): sample_counts(keys, found == source) for source in sorted(set(found))
        }
    return out, sources


def recorded_values(
    frame: pd.DataFrame, rows: np.ndarray, values: pd.DataFrame, store: str, column: str
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """The (donor, sample) keys of ``frame``'s ``rows``, all of ``store``, and each key's value of ``column`` in
    ``values``. A sample with two values of the column, or with none, is an error."""
    recorded = values[(values["storeId"] == store) & (values["column"] == column)]
    counted = recorded.groupby(["donor_id", "sample"])["value"].agg(["nunique", "first"])
    if (counted["nunique"] > 1).any():
        raise ValueError(f"{store}: samples with two values of {column!r}")
    lookup = counted["first"].to_dict()
    keys = list(
        zip(
            frame["donor_id"].iloc[rows].astype(str),
            frame["sample"].iloc[rows].astype(str),
            strict=True,
        )
    )
    found = [lookup.get(key) for key in keys]
    missing = sorted({key for key, value in zip(keys, found, strict=True) if value is None})
    if missing:
        raise ValueError(
            f"{store}: {len(missing)} samples have no {column!r} value, such as {missing[:3]}"
        )
    return keys, np.asarray(found, dtype=object)


def sample_counts(keys: list[tuple[str, str]], hit: np.ndarray) -> dict[str, int]:
    """The samples and donors among the (donor, sample) ``keys`` that ``hit`` marks."""
    samples = {key for key, marked in zip(keys, hit, strict=True) if marked}
    return {"samples": len(samples), "donors": len({donor for donor, _ in samples})}


def rule_columns(rules: dict = SAMPLE_RULES) -> list[str]:
    """The store columns the sample rules read."""
    columns = set()
    for rule in rules.values():
        columns |= set(rule.get("exclude", {}))
        columns |= {rule["sample"]} if rule.get("sample") else set()
        columns |= {spec["column"] for spec in rule.get("time", ())}
    return sorted(columns)


def time_key(values: pd.Series, spec: dict) -> np.ndarray:
    """One time column as numbers to sort samples by; a time not recorded is infinite, so it sorts last."""
    text = values.astype(str)
    if "order" in spec:
        key = text.map({value: float(i) for i, value in enumerate(spec["order"])})
    elif "pattern" in spec:
        key = pd.to_numeric(text.str.extract(spec["pattern"], expand=False), errors="coerce")
        if "default" in spec:
            key = key.fillna(float(spec["default"]))
    else:
        key = pd.to_numeric(text, errors="coerce")
    return np.nan_to_num(np.asarray(key, dtype=float), nan=np.inf)


def earliest_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Each donor's sample that the donor table keeps: the earliest by ``t0`` then ``t1``, then the one with the most
    ``cells``, then the first by ``name``. ``samples`` has one row per donor and sample."""
    missing = {"donor", "t0", "t1", "cells", "name"} - set(samples.columns)
    if missing:
        raise KeyError(f"earliest_samples needs the columns {sorted(missing)}")
    ordered = samples.sort_values(
        ["donor", "t0", "t1", "cells", "name"], ascending=[True, True, True, False, True]
    )
    return ordered.drop_duplicates("donor")


def sample_selection(
    frame: pd.DataFrame, rules: dict = SAMPLE_RULES, excluded: dict = EXCLUDED_STORES
) -> tuple[np.ndarray, dict]:
    """Which cells the donor table keeps, and what each rule removed. ``frame`` has a row per cell with its store
    (``dataset``), its ``donor_group`` (blank without a donor) and the store columns the rules read."""
    store = frame["dataset"].astype(str).to_numpy()
    keep = frame["donor_group"].astype(str).to_numpy() != ""
    sample = np.full(len(frame), "", dtype=object)
    times = np.full((len(frame), 2), np.inf)
    record: dict = {"excludedStores": {}, "excludedValues": {}}
    for name, reason in excluded.items():
        mine = store == name
        record["excludedStores"][name] = {"reason": reason, "cells": int(mine.sum())}
        keep &= ~mine
    for name, rule in rules.items():
        mine = np.flatnonzero(store == name)
        if not len(mine):
            continue
        part = frame.iloc[mine]
        for column, values in rule.get("exclude", {}).items():
            text = part[column].astype(str)
            hit = text.isin(values).to_numpy()
            record["excludedValues"].setdefault(name, {})[column] = {
                str(value): int(count) for value, count in text[hit].value_counts().items()
            }
            keep[mine[hit]] = False
        specs = rule.get("time", ())
        if len(specs) > times.shape[1]:
            raise ValueError(f"{name}: at most {times.shape[1]} time columns, got {len(specs)}")
        identity = rule.get("sample") or (specs[0]["column"] if specs else None)
        if identity:
            sample[mine] = part[identity].astype(str).to_numpy()
        for i, spec in enumerate(specs):
            times[mine, i] = time_key(part[spec["column"]], spec)
    # One sample per donor: the earliest, then the largest, then the first by name.
    rows = np.flatnonzero(keep)
    donor_codes, _ = pd.factorize(frame["donor_group"].astype(str).to_numpy()[rows])
    sample_codes, sample_names = pd.factorize(sample[rows])
    samples = (
        pd.DataFrame(
            {
                "donor": donor_codes,
                "sample": sample_codes,
                "t0": times[rows, 0],
                "t1": times[rows, 1],
            }
        )
        .groupby(["donor", "sample"], sort=False)
        .agg(cells=("t0", "size"), t0=("t0", "min"), t1=("t1", "min"))
        .reset_index()
    )
    samples["name"] = np.asarray(sample_names, dtype=object)[samples["sample"].to_numpy()]
    chosen = earliest_samples(samples)
    chosen_code = np.full(donor_codes.max() + 1 if len(rows) else 0, -1, dtype=np.int64)
    chosen_code[chosen["donor"].to_numpy()] = chosen["sample"].to_numpy()
    dropped = rows[sample_codes != chosen_code[donor_codes]]
    keep[dropped] = False
    several = samples.groupby("donor").size() > 1
    record["samples"] = {
        "donors": int(len(chosen)),
        "donorsWithSeveralSamples": int(several.sum()),
        "samplesDropped": int(len(samples) - len(chosen)),
        "cellsDropped": int(len(dropped)),
        "cellsDroppedByStore": {
            str(k): int(v) for k, v in pd.Series(store[dropped]).value_counts().items()
        },
    }
    record["cellsKept"] = int(keep.sum())
    return keep, record
