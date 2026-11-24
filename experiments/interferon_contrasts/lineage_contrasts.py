"""Disease-versus-healthy contrasts of donor interferon scores in five immune lineages across the human blood stores of
CyteBase, from the catalogue's interferon screen, with COMBAT's donors given their recorded cohort and pooled stores split
by source study from the progenitor experiment's records, into .build/derived/interferon_contrasts/lineage_contrasts.json.

    uv run --locked python -m experiments.interferon_contrasts.lineage_contrasts
"""

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

# The interferon rerun of the blood screen: two jobs, each donor's earliest sample per coarse cell type, eighteen
# interferon genes z-scored against the store's healthy donors, after the blood sample rules. Each donor table is
# checked against the sha256 its job recorded.
INTERFERON_RUNS = tuple(
    ROOT / f"experiments/catalogue/records/interferon_screen/{store_set}.json"
    for store_set in ("immune_cohorts", "other_blood_stores")
)
INTERFERON_TABLES = tuple(path.with_name(f"{path.stem}_donors.csv.gz") for path in INTERFERON_RUNS)
# The screen scores a pseudobulk only from this many cells; a contrast needs this many donors on each side.
SCREEN_MINIMUM_CELLS = 20
CONTRAST_MINIMUM_DONORS = 5
# The lineages each contrast scores, in this order, and the lineage whose donor scores each contrast keeps.
LINEAGES = ("T cells", "NK and ILC", "B cells", "Monocytes", "Dendritic cells")
RIDGE_LINEAGE = "T cells"
# Diseases expected to raise the interferon response: lupus, acute viral infections and juvenile dermatomyositis.
INTERFERON_DISEASES = (
    "systemic lupus erythematosus",
    "COVID-19",
    "influenza",
    "juvenile dermatomyositis",
)
# The kept donor scores are rounded to this many decimals.
SCORE_DECIMALS = 3

# The progenitor experiment's records: the clinical columns its severity job read in the COVID-19 studies' stores, which
# give each sample of a store in COHORT_RECORDS its cohort, and the source study of each sample of a store that pools
# several, which its source-study job read, so that each source study of a store in SOURCE_STUDY_RECORDS is a study of
# its own.
PROGENITOR_RECORDS = ROOT / "experiments/reuse/records/blood_progenitors"
COHORT_VALUES = PROGENITOR_RECORDS / "severity.json"
SOURCE_STUDY_VALUES = PROGENITOR_RECORDS / "source_studies.json"
SUMMARY = DERIVED / "interferon_contrasts" / "lineage_contrasts.json"


def relative(path) -> str:
    return str(path.relative_to(ROOT))


def auc(cases: np.ndarray, controls: np.ndarray) -> float:
    """The probability that a case scores above a control, ties counting half."""
    ranks = pd.Series(np.concatenate([cases, controls])).rank().to_numpy()
    return float(
        (ranks[: len(cases)].sum() - len(cases) * (len(cases) + 1) / 2)
        / (len(cases) * len(controls))
    )


def interferon_units() -> tuple[pd.DataFrame, dict, dict, dict]:
    """The rerun's donor units scored against their store's healthy donors, from each donor's earliest sample, with the
    disease of each sample of a store in COHORT_RECORDS taken from its cohort and the samples of another preparation
    left out, and the collection of each sample of a store in SOURCE_STUDY_RECORDS split by its source study; the two
    jobs' totals; the samples and donors the cohorts relabelled or left out; and each split store's source studies."""
    from experiments.runtime.blood_samples import apply_source_studies

    frames, totals = [], defaultdict(float)
    for run_path, table in zip(INTERFERON_RUNS, INTERFERON_TABLES, strict=True):
        record = load_json(run_path)
        if record["status"] != "completed":
            raise ValueError(f"{run_path.name} did not complete")
        result = record["result"]
        if hashlib.sha256(table.read_bytes()).hexdigest() != result["tables"]["donors"]["sha256"]:
            raise ValueError(f"{table.name} is not the donor table its job recorded")
        for key in ("storesScreened", "primaryBloodCellsRead", "donorsWithScore", "responseBytes"):
            totals[key] += result["totals"][key]
        totals["wallSeconds"] += result["wallSeconds"]
        totals["jobs"] += 1
        frames.append(
            pd.read_csv(
                table,
                usecols=[
                    "storeId",
                    "collectionId",
                    "donor_id",
                    "sample",
                    "disease",
                    "coarse_cell_type",
                    "cells",
                    "earliestSample",
                    "interferonZScore",
                ],
                dtype={"donor_id": str, "sample": str, "disease": str},
                keep_default_na=False,
                na_values=[""],
            )
        )
    units, cohorts = cohort_units(pd.concat(frames, ignore_index=True))
    units, sources = apply_source_studies(
        units, pd.DataFrame(load_json(SOURCE_STUDY_VALUES)["result"]["rows"])
    )
    earliest = units["earliestSample"].astype(str).str.lower() == "true"
    units = units[
        earliest & (units["cells"] >= SCREEN_MINIMUM_CELLS) & units["interferonZScore"].notna()
    ]
    totals_out = {key: (int(v) if key != "wallSeconds" else float(v)) for key, v in totals.items()}
    return units, totals_out, cohorts, sources


def cohort_units(units: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The screen's units with COHORT_RECORDS applied from each sample's recorded cohort. The screen marked each donor's
    earliest sample before any sample left, so a donor whose earliest sample leaves must have no other sample."""
    from experiments.runtime.blood_samples import apply_cohort_records

    values = pd.DataFrame(load_json(COHORT_VALUES)["result"]["rows"])
    kept, cohorts = apply_cohort_records(units, values)
    gone = units.drop(index=kept.index)
    keys = ["storeId", "donor_id"]
    lost = gone.loc[gone["earliestSample"].astype(str).str.lower() == "true", keys]
    stranded = lost.drop_duplicates().merge(kept[keys].drop_duplicates(), on=keys)
    if not stranded.empty:
        raise ValueError(
            "a donor's earliest sample left with its cohort while the donor has another sample, which the screen "
            f"did not mark: {stranded.to_dict('records')[:3]}"
        )
    return kept, cohorts


def lineage_contrasts() -> dict:
    """Each study and disease with at least CONTRAST_MINIMUM_DONORS donors with the disease and as many healthy ones in a
    store: per lineage, the AUC of the disease donors' interferon z-scores against the healthy donors' (from the store
    with the most donors for that lineage, which joins stores a study split by lineage; a study is its collection, its
    store in a collection of independent cohorts, or a source study of a store that pools several), and the T-cell
    z-scores of both sides, under ridge. Interferon diseases first, then the others as controls, each by falling
    T-cell AUC."""
    from experiments.reuse.progenitor_effects import study_of

    units, totals, cohorts, sources = interferon_units()
    contrasts = []
    # A store's donors are contrasted within their collection, which SOURCE_STUDY_RECORDS splits by source study.
    for (store, collection, cell_type), part in units.groupby(
        ["storeId", "collectionId", "coarse_cell_type"]
    ):
        if cell_type not in LINEAGES:
            continue
        healthy = part.loc[part["disease"] == "normal", "interferonZScore"].to_numpy()
        for disease, side in part[part["disease"] != "normal"].groupby("disease"):
            cases = side["interferonZScore"].to_numpy()
            if min(len(cases), len(healthy)) < CONTRAST_MINIMUM_DONORS:
                continue
            contrasts.append(
                {
                    "store": store,
                    "collection": collection,
                    "disease": disease,
                    "lineage": cell_type,
                    "donors": len(cases) + len(healthy),
                    "auc": auc(cases, healthy),
                    "healthy": sorted(round(float(v), SCORE_DECIMALS) for v in healthy),
                    "cases": sorted(round(float(v), SCORE_DECIMALS) for v in cases),
                }
            )
    rows = []
    frame = pd.DataFrame(contrasts)
    frame["study"] = study_of(
        frame.rename(columns={"store": "storeId", "collection": "collectionId"})
    )
    for (_, disease), part in frame.groupby(["study", "disease"]):
        chosen = part.sort_values("donors", ascending=False).drop_duplicates("lineage")
        by_lineage = {row["lineage"]: row for _, row in chosen.iterrows()}
        ridge = by_lineage.get(RIDGE_LINEAGE)
        rows.append(
            {
                "stores": sorted(chosen["store"].unique()),
                "disease": disease,
                "interferon": any(label in INTERFERON_DISEASES for label in disease.split(" || ")),
                "auc": {
                    lineage: (by_lineage[lineage]["auc"] if lineage in by_lineage else None)
                    for lineage in LINEAGES
                },
                "ridge": None
                if ridge is None
                else {"healthy": ridge["healthy"], "cases": ridge["cases"]},
            }
        )
    rows.sort(
        key=lambda r: (
            not r["interferon"],
            -(r["auc"][RIDGE_LINEAGE] if r["auc"][RIDGE_LINEAGE] is not None else -1.0),
            r["stores"][0],
        )
    )
    return {
        "rows": rows,
        "lineages": list(LINEAGES),
        "ridgeLineage": RIDGE_LINEAGE,
        "minimumDonors": CONTRAST_MINIMUM_DONORS,
        "totals": totals,
        "cohortRecords": cohorts,
        "sourceStudies": sources,
        "source": [
            *map(relative, INTERFERON_RUNS),
            *map(relative, INTERFERON_TABLES),
            relative(COHORT_VALUES),
            relative(SOURCE_STUDY_VALUES),
        ],
    }


def contrast_groups(rows: list[dict]) -> dict:
    """The contrasts of the interferon diseases and of the other diseases, the controls: how many each group holds; in
    each lineage, how many of its contrasts score that lineage and their lowest and highest AUROC; and the lowest and
    highest AUROC over every lineage its contrasts score."""
    groups = {}
    for name, interferon in (("interferon", True), ("control", False)):
        part = [row for row in rows if row["interferon"] == interferon]
        lineages = {}
        for lineage in LINEAGES:
            scored = [row["auc"][lineage] for row in part if row["auc"][lineage] is not None]
            if not scored:
                raise ValueError(f"no {name} contrast scores {lineage}")
            lineages[lineage] = {
                "contrasts": len(scored),
                "aurocLowest": min(scored),
                "aurocHighest": max(scored),
            }
        groups[name] = {
            "contrasts": len(part),
            "aurocLowest": min(span["aurocLowest"] for span in lineages.values()),
            "aurocHighest": max(span["aurocHighest"] for span in lineages.values()),
            "lineages": lineages,
        }
    return groups


def main() -> None:
    contrasts = lineage_contrasts()
    write_json_at(
        SUMMARY,
        {
            **contrasts,
            "contrasts": len(contrasts["rows"]),
            "groups": contrast_groups(contrasts["rows"]),
        },
    )


if __name__ == "__main__":
    main()
