"""the statistics of the biological case (Methods, Biological case)."""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.runtime.common import DERIVED, ROOT, _jsonable
from experiments.runtime.multiple_testing import benjamini_hochberg

DATA = ROOT / "experiments/lung/records"
OUTPUT = DERIVED / "lung"

QUERY = "neither"
CONTROL, LESS, MORE = "Control", "Less_fibrotic", "More_fibrotic"
OUTLIER = "VUHD117"
QUANTILE = 0.99
# The empirical quantile as an order statistic (the smallest control value with at least 99% of
# the control cells at or below it): it commutes with the monotone map from distance to
# percentile, so thresholds on either scale select the same cells until percentiles saturate.
QUANTILE_METHOD = "inverted_cdf"
MIN_CELLS = 10
MIN_CONTROL_DONORS = 3
LEVELS = ("ann_level_1", "ann_level_3", "ann_finest_level")
CALIBRATIONS = {"allControls": (), "withoutVUHD117": (OUTLIER,)}
PRIMARY = "allControls"
NATRI_LAB_STUDY = "Banovich_Kropski_2020"
CONTROL_DISEASE = "normal"
UNCERTAINTY = ("uncertLevel1", "uncertLevel3", "uncertLevel5")
NUMERIC = (
    "nearestDistance",
    "distancePercentile",
    "uninformative",
    *(f"vote_{level}" for level in LEVELS),
    *UNCERTAINTY,
)
BARCODE = re.compile(r"([ACGT]{16})")
SAMPLE_TOKEN = re.compile(r"(F\d{5}|HD\d+|ILD\d+(?:-\d+)?)")
# The regions of the clusters: the authors' label of aberrant basaloid cells, rates per 1,000
# cells of a group's lineage, and the bootstrap of the ratio of pooled rates between regions.
BASALOID = "KRT5-/KRT17+"
PER = 1000
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 26


def cell_key(cell_id: str) -> str | None:
    """Sample and barcode of a cell, shared by the Natri and HLCA copies of the same cell."""
    barcode, sample = BARCODE.search(cell_id), SAMPLE_TOKEN.search(cell_id)
    return f"{sample.group(1)}:{barcode.group(1)}" if barcode and sample else None


def empirical_logit(k, n):
    return np.log(
        (np.asarray(k, dtype=float) + 0.5)
        / (np.asarray(n, dtype=float) - np.asarray(k, dtype=float) + 0.5)
    )


# Inputs --------------------------------------------------------------------------------


def load_cells(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    for column in NUMERIC:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="coerce")
    frame["lineage"] = frame["label_ann_level_1"].astype(str)
    frame["informative"] = (
        (frame["uninformative"] == 0)
        & np.isfinite(frame["nearestDistance"])
        & np.isfinite(frame["distancePercentile"])
    )
    return frame


def percentile_of(reference: dict):
    """The toolkit's percentile of a distance among the reference's own nearest-neighbour distances
    (as run_label_transfer computes it from the stored quantiles)."""
    values = np.asarray(reference["distanceValues"], dtype=float)
    quantiles = np.asarray(reference["distanceQuantiles"], dtype=float)
    unique = np.unique(values)
    right = np.searchsorted(values, unique, side="right") - 1
    return lambda d: np.interp(
        np.asarray(d, dtype=float), unique, quantiles[right], left=0.0, right=1.0
    )


# Calibration -----------------------------------------------------------------------------


def thresholds(controls: pd.DataFrame, percentile) -> pd.DataFrame:
    """Per lineage, the 99th percentile of the control cells' distances (and percentiles)."""
    rows = []
    usable = controls[controls["informative"] & (controls["lineage"] != "NA")]
    for lineage, group in usable.groupby("lineage"):
        distance = float(np.quantile(group["nearestDistance"], QUANTILE, method=QUANTILE_METHOD))
        rows.append(
            {
                "lineage": lineage,
                "controlCells": int(len(group)),
                "controlDonors": int(group["donor"].nunique()),
                "thresholdDistance": distance,
                "thresholdPercentile": float(percentile(distance)),
                "percentileRuleThreshold": float(
                    np.quantile(group["distancePercentile"], QUANTILE, method=QUANTILE_METHOD)
                ),
                "controlSaturatedFraction": float((group["distancePercentile"] >= 1.0).mean()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "lineage",
            "controlCells",
            "controlDonors",
            "thresholdDistance",
            "thresholdPercentile",
            "percentileRuleThreshold",
            "controlSaturatedFraction",
        ],
    )


def unexplained(cells: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    limit = (
        cells["lineage"].map(table.set_index("lineage")["thresholdDistance"])
        if len(table)
        else pd.Series(np.nan, index=cells.index)
    )
    return cells["informative"] & limit.notna() & (cells["nearestDistance"] > limit)


def percentile_rule(cells: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    limit = (
        cells["lineage"].map(table.set_index("lineage")["percentileRuleThreshold"])
        if len(table)
        else pd.Series(np.nan, index=cells.index)
    )
    return cells["informative"] & limit.notna() & (cells["distancePercentile"] > limit)


def calibrated_flags(
    cells: pd.DataFrame, calibration: pd.DataFrame, percentile
) -> tuple[pd.Series, pd.DataFrame]:
    """Unexplained flags: thresholds from the calibration cells; a donor that is part of the
    calibration is scored with thresholds from the other calibration donors (leave one out)."""
    table = thresholds(calibration, percentile)
    flags = unexplained(cells, table)
    for donor in calibration["donor"].unique():
        own = cells["donor"] == donor
        if own.any():
            flags[own] = unexplained(
                cells[own], thresholds(calibration[calibration["donor"] != donor], percentile)
            )
    return flags, table


# Fractions and tests -------------------------------------------------------------------------


def sample_fractions(
    cells: pd.DataFrame, flags: pd.Series, keys: list[str], min_cells: int
) -> pd.DataFrame:
    usable = cells[cells["informative"] & (cells["lineage"] != "NA")].assign(unexplained=flags)
    grouped = (
        usable.groupby(["sample", "lineage"])
        .agg(cells=("cellId", "size"), unexplained=("unexplained", "sum"))
        .reset_index()
    )
    meta = (
        cells.groupby("sample")[keys]
        .agg(lambda values: values.mode().iat[0] if len(values.mode()) else "")
        .reset_index()
    )
    frame = grouped.merge(meta, on="sample", how="left")
    frame["fraction"] = frame["unexplained"] / frame["cells"]
    frame["logit"] = empirical_logit(frame["unexplained"], frame["cells"])
    frame["included"] = frame["cells"] >= min_cells
    return frame


def paired(values: pd.DataFrame) -> dict:
    """Paired Wilcoxon signed-rank test of MF against LF across donors, and their coupling."""
    wide = values.dropna(subset=[LESS, MORE])
    out = {
        "donors": int(len(wide)),
        "medianLess": _median(wide[LESS]),
        "medianMore": _median(wide[MORE]),
    }
    difference = wide[MORE] - wide[LESS]
    out["medianDifference"] = _median(difference)
    try:
        test = stats.wilcoxon(wide[MORE], wide[LESS], alternative="two-sided")
        out["wilcoxon"] = {"statistic": float(test.statistic), "pValue": float(test.pvalue)}
    except ValueError as error:
        out["wilcoxon"] = {"statistic": None, "pValue": None, "note": str(error)}
    if len(wide) >= 3 and wide[LESS].nunique() > 1 and wide[MORE].nunique() > 1:
        rho = stats.spearmanr(wide[LESS], wide[MORE])
        out["spearman"] = {
            "rho": float(rho.statistic),
            "pValue": float(rho.pvalue),
            "donors": int(len(wide)),
        }
    else:
        out["spearman"] = {"rho": None, "pValue": None, "donors": int(len(wide))}
    return out


def mixed_model(rows: pd.DataFrame) -> dict:
    """logit ~ region + processing site, donor as random intercept (REML)."""
    import statsmodels.formula.api as smf

    data = rows.assign(
        region=pd.Categorical(rows["region"], categories=[LESS, MORE]),
        site=rows["processingSite"].astype(str),
    ).reset_index(drop=True)
    formula = "logit ~ C(region)" + (" + C(site)" if data["site"].nunique() > 1 else "")
    out: dict = {
        "samples": int(len(data)),
        "donors": int(data["donor"].nunique()),
        "formula": formula,
        "siteIncluded": data["site"].nunique() > 1,
    }
    if data["donor"].nunique() < 3 or data["region"].nunique() < 2:
        return {**out, "estimate": None, "pValue": None, "note": "too few donors or regions"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = smf.mixedlm(formula, data, groups=data["donor"]).fit(reml=True)
        except (ValueError, np.linalg.LinAlgError) as error:
            return {**out, "estimate": None, "pValue": None, "note": str(error)[:300]}
    term = f"C(region)[T.{MORE}]"
    low, high = fit.conf_int().loc[term]
    site_terms = {
        name: float(fit.params[name]) for name in fit.params.index if name.startswith("C(site)")
    }
    return {
        **out,
        "term": term,
        "estimate": float(fit.params[term]),
        "standardError": float(fit.bse[term]),
        "ci95": [float(low), float(high)],
        "oddsRatio": float(np.exp(fit.params[term])),
        "pValue": float(fit.pvalues[term]),
        "siteEffects": site_terms,
        "donorVariance": float(fit.cov_re.iloc[0, 0]),
        "residualVariance": float(fit.scale),
        "converged": bool(fit.converged),
        "warnings": sorted({str(w.message)[:200] for w in caught}),
    }


def region_tests(frame: pd.DataFrame, controls: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Per lineage: paired Wilcoxon, mixed model and coupling between regions; each region
    against the controls' leave-one-out fractions; Benjamini-Hochberg across lineages."""
    diseased = frame[frame["region"].isin([LESS, MORE]) & frame["included"]]
    results = {}
    for lineage in sorted(frame["lineage"].unique()):
        rows = diseased[diseased["lineage"] == lineage]
        pooled = rows.groupby(["donor", "region"])[["cells", "unexplained"]].sum()
        wide = (
            (pooled["unexplained"] / pooled["cells"])
            .unstack("region")
            .reindex(columns=[LESS, MORE])
        )
        result = paired(wide)
        result["mixedModel"] = mixed_model(rows)
        control_rows = controls[(controls["lineage"] == lineage) & controls["included"]]
        control = control_rows["fraction"]
        result["controlMedian"] = _median(control)
        result["pooled"] = {
            LESS: _pooled(rows[rows["region"] == LESS]),
            MORE: _pooled(rows[rows["region"] == MORE]),
            CONTROL: _pooled(control_rows),
        }
        for region in (LESS, MORE):
            region_values = rows[rows["region"] == region]["fraction"]
            result[f"{region}VersusControl"] = _mann_whitney(region_values, control)
        results[lineage] = result
    for field in ("wilcoxon", "mixedModel", f"{LESS}VersusControl", f"{MORE}VersusControl"):
        adjusted = benjamini_hochberg(
            [
                results[lineage][field].get("pValue")
                if results[lineage][field].get("pValue") is not None
                else np.nan
                for lineage in results
            ]
        )
        for lineage, q in zip(results, adjusted):
            results[lineage][field]["qValue"] = None if np.isnan(q) else float(q)
    table = []
    for lineage, result in results.items():
        table.append(
            {
                "lineage": lineage,
                "donorsPaired": result["donors"],
                "medianLess": result["medianLess"],
                "medianMore": result["medianMore"],
                "medianDifference": result["medianDifference"],
                "wilcoxonP": result["wilcoxon"].get("pValue"),
                "wilcoxonQ": result["wilcoxon"].get("qValue"),
                "mixedEstimate": result["mixedModel"].get("estimate"),
                "mixedOddsRatio": result["mixedModel"].get("oddsRatio"),
                "mixedP": result["mixedModel"].get("pValue"),
                "mixedQ": result["mixedModel"].get("qValue"),
                "spearmanRho": result["spearman"].get("rho"),
                "spearmanP": result["spearman"].get("pValue"),
                "controlMedian": result["controlMedian"],
                "pooledLess": result["pooled"][LESS],
                "pooledMore": result["pooled"][MORE],
                "pooledControl": result["pooled"][CONTROL],
                "lessVersusControlQ": result[f"{LESS}VersusControl"].get("qValue"),
                "moreVersusControlQ": result[f"{MORE}VersusControl"].get("qValue"),
            }
        )
    return results, table


def _pooled(rows: pd.DataFrame) -> float | None:
    """Unexplained cells over all cells of the given sample-lineage rows."""
    total = rows["cells"].sum()
    return float(rows["unexplained"].sum() / total) if total else None


def _median(values) -> float | None:
    values = pd.Series(values, dtype=float).dropna()
    return float(values.median()) if len(values) else None


def _mann_whitney(first, second) -> dict:
    first, second = pd.Series(first, dtype=float).dropna(), pd.Series(second, dtype=float).dropna()
    out = {"n": [int(len(first)), int(len(second))], "medians": [_median(first), _median(second)]}
    if len(first) < 2 or len(second) < 2:
        return {**out, "pValue": None}
    test = stats.mannwhitneyu(first, second, alternative="two-sided")
    return {**out, "statistic": float(test.statistic), "pValue": float(test.pvalue)}


# The control lungs -----------------------------------------------------------------------


def control_comparison(frame: pd.DataFrame, min_cells: int) -> tuple[dict, list[dict]]:
    """Each control lung's leave-one-out fraction against the other controls', per lineage
    (donors with at least ``min_cells`` cells in it) and over all lineages (robust z-score and
    the rank among the controls)."""
    controls = frame[frame["region"] == CONTROL]
    per_donor = controls.groupby(["donorKey", "lineage"])[["cells", "unexplained"]].sum()
    overall = (
        controls.groupby("donorKey")[["cells", "unexplained"]]
        .sum()
        .assign(lineage="all")
        .set_index("lineage", append=True)
    )
    table = pd.concat([per_donor, overall]).reset_index()
    table = table[table["cells"] >= min_cells]
    table["fraction"] = table["unexplained"] / table["cells"]
    rows = []
    for lineage, group in table.groupby("lineage"):
        for _, row in group.iterrows():
            others = group[group["donorKey"] != row["donorKey"]]["fraction"]
            median = float(others.median()) if len(others) else np.nan
            mad = (
                float(stats.median_abs_deviation(others, scale="normal"))
                if len(others) > 1
                else np.nan
            )
            rows.append(
                {
                    "donorKey": row["donorKey"],
                    "lineage": lineage,
                    "cells": int(row["cells"]),
                    "unexplained": int(row["unexplained"]),
                    "fraction": float(row["fraction"]),
                    "othersMedian": median,
                    "robustZ": float((row["fraction"] - median) / mad)
                    if mad and np.isfinite(mad) and mad > 0
                    else None,
                    "rankFromTop": int((group["fraction"] > row["fraction"]).sum() + 1),
                    "empiricalP": float(
                        (1 + (others >= row["fraction"]).sum()) / (1 + len(others))
                    ),
                }
            )
    summary = {
        "controlDonors": int(controls["donorKey"].nunique()),
        "outlier": OUTLIER,
        "outlierRows": [r for r in rows if r["donorKey"] == OUTLIER],
        "donorsAboveOthers": sorted(
            {r["donorKey"] for r in rows if r["lineage"] == "all" and r["empiricalP"] <= 0.1}
        ),
    }
    return summary, rows


# Checks with donors in the HLCA -------------------------------------------------------------


def reference_donors(cells: pd.DataFrame, flags: pd.Series, query_controls: pd.DataFrame) -> dict:
    """The Natri donors in the core: near the reference, often on their own copy."""
    core = cells[(cells["donorClass"] == "core") & cells["informative"]]
    if core.empty:
        return {"cells": 0}
    own = core["cellId"].map(cell_key)
    nearest = core["nearestReferenceCell"].map(cell_key)
    donor_medians = core.groupby("donorKey")["distancePercentile"].median()
    control_medians = (
        query_controls[query_controls["informative"]]
        .groupby("donorKey")["distancePercentile"]
        .median()
    )
    by_lineage = (
        pd.DataFrame({"lineage": core["lineage"], "unexplained": flags[core.index]})
        .groupby("lineage")["unexplained"]
        .agg(["size", "mean"])
    )
    return {
        "donors": int(core["donorKey"].nunique()),
        "cells": int(len(core)),
        "percentile": _quantiles(core["distancePercentile"]),
        "queryControlPercentile": _quantiles(
            query_controls.loc[query_controls["informative"], "distancePercentile"]
        ),
        "fractionPercentileAtMostHalf": float((core["distancePercentile"] <= 0.5).mean()),
        "nearestIsOwnCopy": float(
            (own.notna() & (own == nearest)).sum() / max(own.notna().sum(), 1)
        ),
        "cellsWithKey": int(own.notna().sum()),
        "nearestIsSameDonor": float((core["nearestReferenceDonorKey"] == core["donorKey"]).mean()),
        "unexplainedByLineage": {
            lineage: {"cells": int(row["size"]), "fraction": float(row["mean"])}
            for lineage, row in by_lineage.iterrows()
        },
        "donorMediansVersusQueryControls": _mann_whitney(donor_medians, control_medians),
    }


def _quantiles(values) -> dict:
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return {}
    return {
        name: float(values.quantile(q))
        for name, q in (("q05", 0.05), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q95", 0.95))
    }


def extension_donors(cells: pd.DataFrame, replication: pd.DataFrame) -> dict:
    """The toolkit's percentile against the HLCA's label-transfer uncertainty for the Natri donors
    in the extension."""
    natri = cells[(cells["donorClass"] == "extension") & cells["informative"]]
    donors = set(natri["donorKey"].dropna())
    hlca = replication[
        (replication["study"] == NATRI_LAB_STUDY)
        & replication["informative"]
        & replication["donorKey"].isin(donors)
    ]
    out: dict = {"donors": len(donors), "natriCells": int(len(natri)), "hlcaCells": int(len(hlca))}
    out["hlcaCopies"] = {
        level: _spearman(hlca["distancePercentile"], hlca[level]) for level in UNCERTAINTY
    }
    matched = natri.assign(key=natri["cellId"].map(cell_key)).merge(
        hlca.assign(key=hlca["cellId"].map(cell_key))[
            ["key", *UNCERTAINTY, "distancePercentile"]
        ].rename(columns={"distancePercentile": "hlcaPercentile"}),
        on="key",
    )
    out["natriCellsMatched"] = int(len(matched))
    out["natriCopies"] = {
        level: _spearman(matched["distancePercentile"], matched[level]) for level in UNCERTAINTY
    }
    out["sameCellBothCopies"] = _spearman(matched["distancePercentile"], matched["hlcaPercentile"])
    per_donor = pd.DataFrame(
        {
            "natriPercentile": natri.groupby("donorKey")["distancePercentile"].median(),
            "hlcaPercentile": hlca.groupby("donorKey")["distancePercentile"].median(),
            "uncertLevel3": hlca.groupby("donorKey")["uncertLevel3"].median(),
        }
    ).dropna()
    out["perDonor"] = {
        "donors": int(len(per_donor)),
        "natriPercentileVersusUncertainty": _spearman(
            per_donor["natriPercentile"], per_donor["uncertLevel3"]
        ),
        "hlcaPercentileVersusUncertainty": _spearman(
            per_donor["hlcaPercentile"], per_donor["uncertLevel3"]
        ),
    }
    return out


def _spearman(first, second) -> dict:
    pair = pd.DataFrame(
        {
            "a": pd.Series(first, dtype=float).to_numpy(),
            "b": pd.Series(second, dtype=float).to_numpy(),
        }
    ).dropna()
    if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return {"rho": None, "pValue": None, "n": int(len(pair))}
    result = stats.spearmanr(pair["a"], pair["b"])
    return {"rho": float(result.statistic), "pValue": float(result.pvalue), "n": int(len(pair))}


# Replication -----------------------------------------------------------------------------


def replication(frame: pd.DataFrame, percentile, min_cells: int) -> tuple[dict, list[dict]]:
    cells = frame[(frame["study"] != NATRI_LAB_STUDY)].copy()
    cells["group"] = np.where(cells["disease"] == CONTROL_DISEASE, "control", "ILD")
    controls = cells[cells["group"] == "control"]
    summary, rows = {}, []
    for study in sorted(cells["study"].unique()):
        own = controls[controls["study"] == study]
        assay = cells.loc[cells["study"] == study, "assay"].mode().iat[0]
        same_assay = controls[(controls["study"] != study) & (controls["assay"] == assay)]
        if own["donor"].nunique() >= MIN_CONTROL_DONORS:
            calibration, source = own, "own controls"
        elif same_assay["donor"].nunique() >= MIN_CONTROL_DONORS:
            calibration, source = (
                same_assay,
                f"controls of the other studies with the same assay ({assay})",
            )
        else:
            calibration, source = (
                controls[controls["study"] != study],
                "controls of the other studies",
            )
        subject = pd.concat(
            [
                cells[cells["study"] == study],
                calibration if source != "own controls" else own.iloc[0:0],
            ]
        )
        flags, table = calibrated_flags(subject, calibration, percentile)
        usable = subject[subject["informative"] & (subject["lineage"] != "NA")].assign(
            unexplained=flags
        )
        donors = (
            usable.groupby(["donor", "group", "lineage"])
            .agg(cells=("cellId", "size"), unexplained=("unexplained", "sum"))
            .reset_index()
        )
        donors = donors[donors["cells"] >= min_cells]
        donors["fraction"] = donors["unexplained"] / donors["cells"]
        results = {}
        for lineage, group in donors.groupby("lineage"):
            ild_rows, control_rows = (
                group[group["group"] == "ILD"],
                group[group["group"] == "control"],
            )
            pooled = {"ild": _pooled(ild_rows), "control": _pooled(control_rows)}
            results[lineage] = {
                **_mann_whitney(ild_rows["fraction"], control_rows["fraction"]),
                "pooled": pooled,
                "pooledRatio": _ratio(pooled["ild"], pooled["control"]),
            }
        adjusted = benjamini_hochberg(
            [r["pValue"] if r["pValue"] is not None else np.nan for r in results.values()]
        )
        for (lineage, result), q in zip(results.items(), adjusted):
            result["qValue"] = None if np.isnan(q) else float(q)
            rows.append(
                {
                    "study": study,
                    "calibration": source,
                    "lineage": lineage,
                    "ildDonors": result["n"][0],
                    "controlDonors": result["n"][1],
                    "medianIld": result["medians"][0],
                    "medianControl": result["medians"][1],
                    "pooledIld": result["pooled"]["ild"],
                    "pooledControl": result["pooled"]["control"],
                    "pooledRatio": result["pooledRatio"],
                    "pValue": result["pValue"],
                    "qValue": result["qValue"],
                }
            )
        summary[study] = {
            "calibration": source,
            "assay": assay,
            "calibrationStudies": sorted(calibration["study"].unique()),
            "calibrationDonors": int(calibration["donor"].nunique()),
            "ildDonors": int(
                cells[(cells["study"] == study) & (cells["group"] == "ILD")]["donor"].nunique()
            ),
            "diseases": sorted(cells.loc[cells["study"] == study, "disease"].unique()),
            "thresholds": table.to_dict(orient="records"),
            "byLineage": results,
        }
    return summary, rows


def _ratio(a, b) -> float | None:
    return float(a / b) if a is not None and b not in (None, 0) else None


# Clusters --------------------------------------------------------------------------------


def clusters(query: pd.DataFrame, raw: Path, records: dict) -> tuple[dict, list[dict]]:
    path = raw / "lung_reference_screen_unexplained_clusters.csv.gz"
    if not path.exists():
        return {}, []
    assignments = pd.read_csv(path, dtype={"cluster": str}, keep_default_na=False)
    markers_path = raw / "lung_reference_screen_unexplained_markers.csv.gz"
    markers = (
        pd.read_csv(markers_path, dtype={"cluster": str}, keep_default_na=False)
        if markers_path.exists()
        else pd.DataFrame()
    )
    counterparts = {
        c["cluster"]: c for c in (records.get("characterise") or {}).get("clusters", [])
    }
    cells = assignments.merge(query, on="cellId", how="left")
    diseased = query[query["informative"] & query["region"].isin([LESS, MORE])]
    totals = diseased.groupby(["donorKey", "region"]).size()
    summary, rows, p_values = {}, [], []
    for cluster, group in cells.groupby("cluster"):
        # Author labels are read only here, after the clusters were fixed.
        authors = group["authorLabel"].value_counts(normalize=True)
        members = group[group["region"].isin([LESS, MORE])].groupby(["donorKey", "region"]).size()
        share = (
            (members.reindex(totals.index, fill_value=0) / totals)
            .unstack("region")
            .reindex(columns=[LESS, MORE])
        )
        enrichment = paired(share)
        p_values.append(enrichment["wilcoxon"].get("pValue"))
        controls = group[group["region"] == CONTROL]
        top = markers[markers["cluster"] == cluster] if len(markers) else pd.DataFrame()
        entry = {
            "cells": int(len(group)),
            "donors": int(group["donorKey"].nunique()),
            "cellsByRegion": {k: int(v) for k, v in group["region"].value_counts().items()},
            "fractionFromControls": float((group["region"] == CONTROL).mean()),
            "controlDonors": int(controls["donorKey"].nunique()),
            "controlDonorsWithoutOutlier": int(
                controls.loc[controls["donorKey"] != OUTLIER, "donorKey"].nunique()
            ),
            "cellsFromOutlier": int((group["donorKey"] == OUTLIER).sum()),
            "predictedLineage": {
                k: float(v)
                for k, v in group["lineage"].value_counts(normalize=True).head(3).items()
            },
            "predictedFinest": {
                k: float(v)
                for k, v in group["label_ann_finest_level"]
                .value_counts(normalize=True)
                .head(3)
                .items()
            },
            "authorLabel": authors.index[0] if len(authors) else None,
            "authorLabels": {k: float(v) for k, v in authors.head(3).items()},
            "regionEnrichment": enrichment,
            "nearestReferenceLabels": counterparts.get(cluster, {}).get("nearestReferenceLabels"),
            "genesUp": top[top["direction"] == "up"]
            .head(10)[["featureName", "auroc", "log2FoldChange"]]
            .to_dict(orient="records")
            if len(top)
            else [],
            "genesDown": top[top["direction"] == "down"]
            .head(10)[["featureName", "auroc", "log2FoldChange"]]
            .to_dict(orient="records")
            if len(top)
            else [],
        }
        summary[cluster] = entry
    for (cluster, entry), q in zip(
        summary.items(), benjamini_hochberg([np.nan if p is None else p for p in p_values])
    ):
        entry["regionEnrichment"]["wilcoxon"]["qValue"] = None if np.isnan(q) else float(q)
        rows.append(
            {
                "cluster": cluster,
                "cells": entry["cells"],
                "donors": entry["donors"],
                "authorLabel": entry["authorLabel"],
                "authorLabelShare": next(iter(entry["authorLabels"].values()), None),
                "fractionFromControls": entry["fractionFromControls"],
                "controlDonors": entry["controlDonors"],
                "controlDonorsWithoutOutlier": entry["controlDonorsWithoutOutlier"],
                "medianShareLess": entry["regionEnrichment"]["medianLess"],
                "medianShareMore": entry["regionEnrichment"]["medianMore"],
                "enrichmentP": entry["regionEnrichment"]["wilcoxon"].get("pValue"),
                "enrichmentQ": entry["regionEnrichment"]["wilcoxon"]["qValue"],
                "topGenesUp": " ".join(g["featureName"] for g in entry["genesUp"][:5]),
            }
        )
    return summary, rows


# Regions of the clusters -----------------------------------------------------------------------


def cluster_regions(
    query: pd.DataFrame, flags: pd.Series, raw: Path, min_cells: int
) -> tuple[dict, dict, list[dict]]:
    """Cells and donors of each cluster of unexplained cells by region, pooled and per donor; the
    same for the clusters whose most frequent author label is the basaloid one, together, and for
    every query cell with that author label. Returns the clusters' summary, the basaloid summary
    and one row per group, donor and region (zeros included)."""
    path = raw / "lung_reference_screen_unexplained_clusters.csv.gz"
    if not path.exists():
        return {}, {}, []
    assignments = pd.read_csv(path, dtype={"cluster": str}, keep_default_na=False)
    base = query[query["informative"] & (query["lineage"] != "NA")]
    cluster = base["cellId"].map(assignments.set_index("cellId")["cluster"])
    flagged = flags.reindex(base.index, fill_value=False).astype(bool)
    template, lineages = _donor_regions(base)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary: dict = {
        "rule": {
            "rate": f"cells of the group per {PER:,} cells of its lineage (the level-1 lineage of most of its cells) in the same donor and region",
            "paired": f"diseased lungs sampled in both regions with at least {min_cells} cells of the lineage in each",
            "basaloidClusters": f"clusters whose most frequent author label is {BASALOID}",
            "bootstrap": f"{BOOTSTRAP:,} resamples of the paired lungs with replacement (seed {BOOTSTRAP_SEED}); 2.5th and 97.5th percentiles of the ratio of pooled rates, less over more fibrotic",
            "meanRateRatio": "ratio of the mean per-lung rates, less over more fibrotic, over the same paired lungs, each lung weighted equally; its interval from the same resamples as the pooled ratio's",
        },
        "unexplainedCellsByRegion": {
            k: int(v) for k, v in flagged.groupby(base["region"]).sum().items()
        },
        "clusters": {},
    }
    rows: list[dict] = []
    majority: dict[str, str] = {}
    for name in sorted(cluster.dropna().unique(), key=int):
        members = cluster == name
        lineage = base.loc[members, "lineage"].value_counts().index[0]
        majority[name] = base.loc[members, "authorLabel"].value_counts().index[0]
        table = _region_table(base, template, lineages, members, lineage)
        summary["clusters"][name] = {
            "lineage": lineage,
            "authorLabel": majority[name],
            **_census(table, min_cells),
        }
        rows += _region_rows(name, "cluster", lineage, table)

    chosen = sorted((name for name, label in majority.items() if label == BASALOID), key=int)
    in_clusters = cluster.isin(chosen)
    lineage = (
        base.loc[in_clusters, "lineage"].value_counts().index[0]
        if in_clusters.any()
        else "Epithelial"
    )
    table = _region_table(base, template, lineages, in_clusters, lineage)
    unexplained = {"clusters": chosen, "lineage": lineage, **_census(table, min_cells, rng)}
    for region, entry in unexplained["byRegion"].items():
        total = int(flagged[base["region"] == region].sum())
        entry["unexplainedCells"] = total
        entry["shareOfUnexplained"] = entry["cells"] / total if total else None
    rows += _region_rows("basaloid clusters", "clusters by author label", lineage, table)

    labelled = base["authorLabel"] == BASALOID
    lineage = (
        base.loc[labelled, "lineage"].value_counts().index[0] if labelled.any() else "Epithelial"
    )
    table = _region_table(base, template, lineages, labelled, lineage)
    authors = {"lineage": lineage, **_census(table, min_cells, rng)}
    for region, entry in authors["byRegion"].items():
        own = labelled & (base["region"] == region)
        entry["unexplained"] = int((own & flagged).sum())
        entry["inBasaloidClusters"] = int((own & in_clusters).sum())
        entry["unexplainedShare"] = (
            entry["unexplained"] / entry["cells"] if entry["cells"] else None
        )
    authors["unexplained"] = int((labelled & flagged).sum())
    authors["unexplainedShare"] = (
        authors["unexplained"] / authors["cells"] if authors["cells"] else None
    )
    rows += _region_rows("basaloid author label", "author label", lineage, table)
    basaloid = {
        "authorLabel": BASALOID,
        "unexplainedClusters": unexplained,
        "authorLabelled": authors,
    }
    return summary, basaloid, rows


def _donor_regions(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per donor and region of the query: its diagnosis, samples and cells, and its cells of each lineage."""
    keys = ["donorKey", "region"]
    template = base.groupby(keys).agg(
        diagnosis=("diagnosis", "first"), samples=("sample", "nunique"), allCells=("cellId", "size")
    )
    lineages = base.groupby([*keys, "lineage"]).size().unstack("lineage", fill_value=0)
    return template, lineages


def _region_table(
    base: pd.DataFrame,
    template: pd.DataFrame,
    lineages: pd.DataFrame,
    members: pd.Series,
    lineage: str,
) -> pd.DataFrame:
    """One row per donor and region: the group's cells and the cells of its lineage."""
    table = template.copy()
    table["lineageCells"] = (
        lineages[lineage].reindex(table.index, fill_value=0) if lineage in lineages.columns else 0
    )
    table["cells"] = (
        base[members].groupby(["donorKey", "region"]).size().reindex(table.index, fill_value=0)
    )
    return table.reset_index()


def _region_rows(group: str, kind: str, lineage: str, table: pd.DataFrame) -> list[dict]:
    return [
        {
            "group": group,
            "kind": kind,
            "lineage": lineage,
            "donorKey": row["donorKey"],
            "region": row["region"],
            "diagnosis": row["diagnosis"],
            "samples": int(row["samples"]),
            "allCells": int(row["allCells"]),
            "lineageCells": int(row["lineageCells"]),
            "cells": int(row["cells"]),
            "perThousand": _per_thousand(row["cells"], row["lineageCells"]),
        }
        for row in table.to_dict(orient="records")
    ]


def _per_thousand(cells, total) -> float | None:
    return float(PER * cells / total) if total else None


def _census(table: pd.DataFrame, min_cells: int, rng: np.random.Generator | None = None) -> dict:
    """Cells and donors of one group by region, pooled; the diseased lungs that hold it; and the
    paired comparison of the two regions of each lung (with tests when rng is given)."""
    out: dict = {
        "cells": int(table["cells"].sum()),
        "donors": int(table.loc[table["cells"] > 0, "donorKey"].nunique()),
        "byRegion": {},
    }
    for region, part in table.groupby("region"):
        out["byRegion"][region] = {
            "cells": int(part["cells"].sum()),
            "donors": int((part["cells"] > 0).sum()),
            "donorsSampled": int(len(part)),
            "lineageCells": int(part["lineageCells"].sum()),
            "allCells": int(part["allCells"].sum()),
            "perThousand": _per_thousand(part["cells"].sum(), part["lineageCells"].sum()),
        }
    controls = table[table["region"] == CONTROL]
    others = controls[controls["donorKey"] != OUTLIER]
    out["controlsWithoutOutlier"] = {
        "cells": int(others["cells"].sum()),
        "donors": int((others["cells"] > 0).sum()),
        "donorsSampled": int(len(others)),
    }
    out["cellsFromOutlier"] = int(controls.loc[controls["donorKey"] == OUTLIER, "cells"].sum())
    diseased = table[table["region"].isin([LESS, MORE])]
    out["diseasedLungs"] = {
        "sampled": int(diseased["donorKey"].nunique()),
        "withCells": int(diseased.loc[diseased["cells"] > 0, "donorKey"].nunique()),
    }
    out["paired"] = _paired_regions(diseased, min_cells, rng)
    return out


def _paired_regions(
    diseased: pd.DataFrame, min_cells: int, rng: np.random.Generator | None
) -> dict:
    """The two regions of each lung sampled in both, with at least min_cells cells of the lineage in each."""
    columns = pd.MultiIndex.from_product([["cells", "lineageCells"], [LESS, MORE]])
    wide = (
        diseased.pivot(index="donorKey", columns="region", values=["cells", "lineageCells"])
        .reindex(columns=columns)
        .astype(float)
    )
    both = wide.dropna()
    enough = both[
        (both[("lineageCells", LESS)] >= min_cells) & (both[("lineageCells", MORE)] >= min_cells)
    ]
    less, more = enough[("cells", LESS)], enough[("cells", MORE)]
    less_n, more_n = enough[("lineageCells", LESS)], enough[("lineageCells", MORE)]
    rate_less, rate_more = PER * less / less_n, PER * more / more_n
    in_less, in_more = less > 0, more > 0
    holding = in_less | in_more
    out: dict = {
        "lungsSampledInBoth": int(len(both)),
        "lungs": int(len(enough)),
        "lungsWithCells": int(holding.sum()),
        "inLess": int(in_less.sum()),
        "inMore": int(in_more.sum()),
        "inBoth": int((in_less & in_more).sum()),
        "inLessOnly": int((in_less & ~in_more).sum()),
        "inMoreOnly": int((~in_less & in_more).sum()),
        "shareOfLungsWithCellsInLess": float(in_less.sum() / holding.sum())
        if holding.any()
        else None,
        "shareOfLungsWithCellsInMore": float(in_more.sum() / holding.sum())
        if holding.any()
        else None,
        "higherInLess": int((rate_less > rate_more).sum()),
        "higherInMore": int((rate_more > rate_less).sum()),
        "cellsLess": int(less.sum()),
        "cellsMore": int(more.sum()),
        "lineageCellsLess": int(less_n.sum()),
        "lineageCellsMore": int(more_n.sum()),
        "perThousandLess": _per_thousand(less.sum(), less_n.sum()),
        "perThousandMore": _per_thousand(more.sum(), more_n.sum()),
        "medianPerThousandLessAmongLungsWithCells": _median(rate_less[holding]),
        "medianPerThousandMoreAmongLungsWithCells": _median(rate_more[holding]),
    }
    out["pooledRatio"] = _ratio(out["perThousandLess"], out["perThousandMore"])
    if rng is None or not holding.any():
        return out
    unequal = out["higherInLess"] + out["higherInMore"]
    out["signTest"] = {
        "lungsWithUnequalRates": unequal,
        "pValue": float(stats.binomtest(out["higherInLess"], unequal, 0.5).pvalue)
        if unequal
        else None,
    }
    try:
        test = stats.wilcoxon(rate_more, rate_less, alternative="two-sided")
        out["wilcoxon"] = {"statistic": float(test.statistic), "pValue": float(test.pvalue)}
    except ValueError as error:
        out["wilcoxon"] = {"statistic": None, "pValue": None, "note": str(error)}
    pooled, mean = _bootstrap_ratios(
        less.to_numpy(), less_n.to_numpy(), more.to_numpy(), more_n.to_numpy(), rng
    )
    out["pooledRatioCi95"] = pooled
    # Each lung weighted equally: the mean of the lungs' rates in each region, over the same lungs.
    out["meanPerThousandLess"] = float(rate_less.mean())
    out["meanPerThousandMore"] = float(rate_more.mean())
    out["meanRateRatio"] = _ratio(out["meanPerThousandLess"], out["meanPerThousandMore"])
    out["meanRateRatioCi95"] = mean
    return out


def _bootstrap_ratios(less, less_n, more, more_n, rng: np.random.Generator) -> tuple[dict, dict]:
    """Percentile intervals, less over more fibrotic, of the ratio of pooled rates and of the ratio
    of mean per-lung rates, both from the same resamples of the lungs."""
    draws = rng.integers(0, len(less), size=(BOOTSTRAP, len(less)))
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = (less[draws].sum(axis=1) / less_n[draws].sum(axis=1)) / (
            more[draws].sum(axis=1) / more_n[draws].sum(axis=1)
        )
        mean = (less / less_n)[draws].mean(axis=1) / (more / more_n)[draws].mean(axis=1)
    return _interval(pooled), _interval(mean)


def _interval(ratio: np.ndarray) -> dict:
    finite = ratio[np.isfinite(ratio)]
    if not finite.size:
        return {"resamples": BOOTSTRAP, "finite": 0, "low": None, "high": None}
    return {
        "resamples": BOOTSTRAP,
        "finite": int(finite.size),
        "low": float(np.percentile(finite, 2.5)),
        "high": float(np.percentile(finite, 97.5)),
    }


def author_labels(query: pd.DataFrame, flags: pd.Series) -> list[dict]:
    """Each author label (read only after clustering): its cells, unexplained cells and their
    share, overall and by region; highest share first."""
    base = query[query["informative"] & (query["lineage"] != "NA")]
    frame = base.assign(unexplained=flags.reindex(base.index, fill_value=False).astype(bool))
    out = []
    for label, group in frame.groupby("authorLabel"):
        out.append(
            {
                "authorLabel": label,
                "cells": int(len(group)),
                "unexplained": int(group["unexplained"].sum()),
                "share": float(group["unexplained"].mean()),
                "byRegion": {
                    region: {
                        "cells": int(len(part)),
                        "unexplained": int(part["unexplained"].sum()),
                        "share": float(part["unexplained"].mean()),
                    }
                    for region, part in group.groupby("region")
                },
            }
        )
    return sorted(out, key=lambda entry: -entry["share"])


# Design and provenance ---------------------------------------------------------------------


def design_check(cells: pd.DataFrame, data: Path) -> dict:
    """Donor classes and query size against the cohort design from public metadata."""
    donors = cells.drop_duplicates("donor").groupby("donorClass").size()
    query = cells[cells["donorClass"] == QUERY]
    found = {
        "donorsByClass": {k: int(v) for k, v in donors.items()},
        "queryCellsMapped": int(len(query)),
        "queryCellsInformative": int(query["informative"].sum()),
        "querySamples": int(query["sample"].nunique()),
        "querySamplesByRegion": {
            k: int(v) for k, v in query.drop_duplicates("sample").groupby("region").size().items()
        },
        "queryCellsBySite": {k: int(v) for k, v in query.groupby("processingSite").size().items()},
        "abstainedFraction": {
            level: float((query[f"label_{level}"] == "NA").mean()) for level in LEVELS
        },
    }
    design_path = data / "lung_cohort_design.json"
    if design_path.exists():
        design = json.loads(design_path.read_text())
        found["expected"] = {
            "donorsByClass": {
                "core": design["queryDonors"]["inHlcaCore"],
                "extension": design["queryDonors"]["inHlcaExtensionOnly"],
                QUERY: design["queryDonors"]["inNeither"],
            },
            "queryCells": design["query"]["cells"],
            "querySamples": design["query"]["samples"],
        }
    return found


def upstream(records: dict) -> dict:
    """Resources, datasets and settings of the measured jobs."""
    out = {}
    for name, record in records.items():
        source = record.get("experimentalConditions") or record.get("provenance") or {}
        out[name] = {
            "status": record.get("status"),
            "container": source.get("container"),
        }
    first = next(
        (
            r.get("experimentalConditions") or r.get("provenance")
            for r in records.values()
            if r.get("experimentalConditions") or r.get("provenance")
        ),
        {},
    )
    return {
        "jobs": out,
        "datasets": first.get("datasets"),
        "parameters": first.get("parameters"),
    }


def write_json(path: Path, payload: dict, records: dict) -> None:
    document = {
        "experimentalConditions": upstream(records),
        **_jsonable(payload),
    }
    path.write_text(json.dumps(document, indent=1, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


# Main --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=DATA, help="folder with the collected records (default: data/)"
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=MIN_CELLS,
        help="cells a sample or donor needs in a lineage (default 10)",
    )
    args = parser.parse_args()
    data, raw = args.data, args.data / "raw"
    records = json.loads((data / "lung_reference_screen_records.json").read_text())["records"]
    percentile = percentile_of(records["map"]["reference"])
    cells = load_cells(raw / "lung_reference_screen_natri_cells.csv.gz")
    query = cells[cells["donorClass"] == QUERY]
    keys = [
        "donor",
        "donorKey",
        "region",
        "processingSite",
        "site",
        "diagnosis",
        "disease",
        "donorClass",
    ]
    summary: dict = {
        "design": design_check(cells, DATA),
        "calibrations": {},
        "rule": _rule(args.min_cells),
    }
    threshold_rows, fraction_rows, test_rows = [], [], []
    primary_flags = None
    for name, excluded in CALIBRATIONS.items():
        calibration = query[(query["region"] == CONTROL) & ~query["donorKey"].isin(excluded)]
        flags, table = calibrated_flags(query, calibration, percentile)
        frame = sample_fractions(query, flags, keys, args.min_cells)
        controls = frame[(frame["region"] == CONTROL) & ~frame["donorKey"].isin(excluded)]
        tests, rows = region_tests(frame, controls)
        disagree = int((unexplained(query, table) != percentile_rule(query, table)).sum())
        summary["calibrations"][name] = {
            "excludedControls": list(excluded),
            "controlDonors": int(calibration["donorKey"].nunique()),
            "thresholds": table.to_dict(orient="records"),
            "cellsClassifiedOtherwiseByPercentileRule": disagree,
            "unexplainedCells": int(flags.sum()),
            "unexplainedFractionByRegion": {
                k: float(v)
                for k, v in pd.Series(flags[query["informative"]])
                .groupby(query["region"])
                .mean()
                .items()
            },
            "regionTests": tests,
        }
        threshold_rows += [{"calibration": name, **row} for row in table.to_dict(orient="records")]
        fraction_rows += [{"calibration": name, **row} for row in frame.to_dict(orient="records")]
        test_rows += [{"calibration": name, **row} for row in rows]
        if name == PRIMARY:
            primary_flags, primary_frame = flags, frame
    summary["controlComparison"], control_rows = control_comparison(primary_frame, args.min_cells)
    summary["referenceDonors"] = reference_donors(
        cells,
        unexplained(cells, thresholds(query[query["region"] == CONTROL], percentile)),
        query[query["region"] == CONTROL],
    )
    replication_cells = load_cells(raw / "lung_reference_screen_replication_cells.csv.gz")
    summary["extensionDonors"] = extension_donors(cells, replication_cells)
    summary["replication"], replication_rows = replication(
        replication_cells, percentile, args.min_cells
    )
    summary["replicationSelection"] = (records["map"].get("replication") or {}).get("selection")
    summary["clusters"], cluster_rows = clusters(query, raw, records)
    summary["clusterRegions"], summary["basaloid"], region_rows = cluster_regions(
        query, primary_flags, raw, args.min_cells
    )
    summary["authorLabels"] = author_labels(query, primary_flags)
    chosen = query[primary_flags][["cellId", "donor", "sample", "region", "lineage"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(
        OUTPUT / "lung_reference_screen_unexplained_cells.csv.gz", index=False, compression="gzip"
    )
    print(
        f"wrote {OUTPUT / 'lung_reference_screen_unexplained_cells.csv.gz'} ({len(chosen)} cells)",
        flush=True,
    )
    write_json(OUTPUT / "lung_reference_screen_summary.json", summary, records)
    for name, rows in (
        ("thresholds", threshold_rows),
        ("fractions", fraction_rows),
        ("region_tests", test_rows),
        ("controls", control_rows),
        ("replication", replication_rows),
        ("clusters", cluster_rows),
        ("cluster_regions", region_rows),
    ):
        write_csv(OUTPUT / f"lung_reference_screen_{name}.csv", rows)


def _rule(min_cells: int) -> dict:
    return {
        "lineage": "label transferred at ann_level_1 with a vote of at least 0.6",
        "threshold": f"{QUANTILE:.0%} quantile of the control query cells of the lineage ({QUANTILE_METHOD}: the smallest control value with at least {QUANTILE:.0%} of the control cells at or below it)",
        "ordering": "distance to the nearest reference cell; its percentile among the reference's own nearest-neighbour distances is reported, and saturates at 1 beyond the largest",
        "minCellsPerSampleOrDonorAndLineage": min_cells,
        "logit": "empirical, log((k + 0.5) / (n - k + 0.5))",
        "controlsScoredLeaveOneOut": True,
        "minControlDonorsForOwnCalibration": MIN_CONTROL_DONORS,
    }


if __name__ == "__main__":
    main()
