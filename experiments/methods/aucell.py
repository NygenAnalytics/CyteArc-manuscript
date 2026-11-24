from pathlib import Path
from typing import Any

import cytearc
import numpy as np
import pandas as pd
from cytearc.quality_control.cell_cycle_genes import g2m_phase_genes, s_phase_genes
from scipy import sparse

from experiments.methods.method_inputs import summarize
from experiments.runtime.agreement import numeric_agreement, package_version
from experiments.runtime.catalogue_counts import build_store, input_record
from experiments.runtime.common import Stopwatch, configure_cytearc, write_json
from experiments.runtime.rna_recipe import analysis_record, run_rna

EXPERIMENT = "aucell"
DATASET = "tenx_5K_pbmc_rnaseq"
# The scored cells and the marker sets come from CyteArc's default cell filter, RNA recipe and clustering on the
# pinned PBMC counts.
RUN_LABEL = "scoring_input"
TMIN = 5
TIE_SEED = 0
MARKERS_PER_CLUSTER = 20
WAGGR_DEFAULT_CELLS = 500
SUBSET_SEED = 0
TUTORIAL_SETS = {
    "T_cell": ["CD3D", "CD3E", "TRAC", "LTB", "IL7R"],
    "B_cell": ["MS4A1", "CD79A", "CD37", "CD74", "HLA-DRA"],
    "Myeloid": ["LST1", "S100A8", "S100A9", "CTSS", "FCER1G"],
}


def build_network(ds: cytearc.DataStore, run: Any) -> tuple[pd.DataFrame, dict]:
    names = ds.RNA.feats.fetch_all("names").astype(str)
    upper = pd.Series(np.char.upper(names))
    counts = upper.value_counts()
    ambiguous = set(counts[counts > 1].index)
    rows: list[tuple[str, str, float]] = []
    for gene in s_phase_genes:
        rows.append(("S_phase", gene, 1.0))
    for gene in g2m_phase_genes:
        rows.append(("G2M_phase", gene, 1.0))
    for source, genes in TUTORIAL_SETS.items():
        for gene in genes:
            weight = 1.5 if gene in {"S100A8", "S100A9"} else 1.0
            rows.append((source, gene, weight))
    for gene in TUTORIAL_SETS["T_cell"]:
        rows.append(("T_vs_Myeloid", gene, 1.0))
    for gene in TUTORIAL_SETS["Myeloid"]:
        rows.append(("T_vs_Myeloid", gene, -1.0))
    markers = ds.markers.load(run["markers"])
    markers = markers[~markers["feature_name"].str.upper().isin(ambiguous)]
    top = (
        markers.sort_values(["group_id", "score"], ascending=[True, False])
        .groupby("group_id", sort=True)
        .head(MARKERS_PER_CLUSTER)
    )
    for group_id, frame in top.groupby("group_id", sort=True):
        for gene, score in zip(frame["feature_name"], frame["score"], strict=True):
            rows.append((f"markers_{group_id}", str(gene), float(score)))
    net = pd.DataFrame(rows, columns=["source", "target", "weight"])
    in_assay = net["target"].str.upper().isin(set(upper))
    unambiguous = ~net["target"].str.upper().isin(ambiguous)
    info = {
        "edgesListed": int(len(net)),
        "edgesAbsentFromAssay": int((~in_assay).sum()),
        "edgesDroppedAsAmbiguous": int((in_assay & ~unambiguous).sum()),
        "markerClusters": int(top["group_id"].nunique()),
    }
    return (net[in_assay & unambiguous].reset_index(drop=True), info)


def to_feature_ids(net: pd.DataFrame, names: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    lookup = {str(name).upper(): str(feature_id) for name, feature_id in zip(names, ids)}
    translated = net.copy()
    translated["target"] = [lookup[target.upper()] for target in net["target"]]
    return translated


def per_source(cytearc_scores: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    sources = sorted(set(cytearc_scores.columns) & set(reference.columns))
    table = {
        source: numeric_agreement(cytearc_scores[source], reference[source]) for source in sources
    }
    rounded = {
        source: float(
            np.max(
                np.abs(
                    cytearc_scores[source].to_numpy()
                    - reference[source].to_numpy().astype(np.float32).astype(np.float64)
                )
            )
        )
        for source in sources
    }
    return {
        "sources": sources,
        "sourcesOnlyCyteArc": sorted(set(cytearc_scores.columns) - set(reference.columns)),
        "sourcesOnlyReference": sorted(set(reference.columns) - set(cytearc_scores.columns)),
        "maxAbsDiffOverall": float(max((v["maxAbsDiff"] for v in table.values()))),
        "maxAbsDiffReferenceRoundedToFloat32": float(max(rounded.values())),
        "pearson": summarize([v["pearson"] for v in table.values()]),
        "spearman": summarize([v["spearman"] for v in table.values()]),
        "perSource": {
            source: {
                key: value[key]
                for key in ("pearson", "spearman", "maxAbsDiff", "rangeFirst", "rangeSecond")
            }
            for source, value in table.items()
        },
    }


def cytearc_scores(ds: cytearc.DataStore, ref: cytearc.ArtifactRef) -> pd.DataFrame:
    result = ds.scores.load(ref)
    return pd.DataFrame(
        np.asarray(result.data.compute(), dtype=np.float64), columns=list(result.source_names)
    )


def decoupler_scores(adata: Any, name: str) -> pd.DataFrame:
    frame = adata.obsm[f"score_{name}"]
    return frame.reset_index(drop=True).astype(np.float64)


def run(request: dict) -> dict:
    try:
        import anndata
        import decoupler as dc
    except ImportError as error:
        raise RuntimeError("decoupler is required; use the decoupler experiment image") from error
    configure_cytearc()
    with Stopwatch() as total:
        work = Path(request["workDirectory"])
        work.mkdir(parents=True, exist_ok=True)
        store = build_store(DATASET, work / "analysis.zarr", downloads=work / "inputs")
        ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=4, mem_budget="4G")
        run = run_rna(
            ds,
            label=RUN_LABEL,
            umap=False,
            paris=False,
            cell_cycle=False,
            doublets=False,
            markers=True,
        )
        cells = run["analysis_cell_selection"]
        universe = run["feature_universe"]
        mask = np.asarray(ds.artifacts.load(cells)["values"][:], dtype=bool)
        cell_idx = np.flatnonzero(mask)
        universe_mask = np.asarray(ds.artifacts.load(universe)["values"][:], bool)
        names = ds.RNA.feats.fetch_all("names").astype(str)
        ids = ds.RNA.feats.fetch_all("ids").astype(str)
        net, net_info = build_network(ds, run)
        net_ids = to_feature_ids(net, names, ids)
        exported = ds.to_anndata(run=run)
        if not np.array_equal(
            exported.obs_names.to_numpy().astype(str),
            ds.cells.fetch_all("ids").astype(str)[cell_idx],
        ) or not np.array_equal(exported.var_names.to_numpy().astype(str), ids):
            raise RuntimeError("export does not match the run selection and features")
        raw = sparse.csr_matrix(exported.X, dtype=np.float64)
        if not universe_mask.all():
            raise RuntimeError("expected the run's feature universe to be every feature")
        totals = np.asarray(ds.cells.fetch_all("RNA_nCounts"), np.float64)[cell_idx]
        totals[totals == 0] = 1.0
        size_factor = float(ds.RNA.sf)
        normalised = sparse.csr_matrix(sparse.diags(size_factor / totals) @ raw, dtype=np.float64)
        with Stopwatch() as cytearc_aucell_time:
            aucell_ref = ds.scores.aucell(
                net, cells, features=universe, tmin=TMIN, tie_seed=TIE_SEED
            )
        cytearc_aucell = cytearc_scores(ds, aucell_ref)
        aucell_group = ds.artifacts.load(aucell_ref)
        rank_feature_index = np.asarray(aucell_group["rank_feature_index"][:], np.int64)
        aucell_status = ds.artifacts.inspect(aucell_ref)
        cytearc_n_up = int(aucell_status.parameters["n_up"])
        n_features = rank_feature_index.size
        probe = anndata.AnnData(X=sparse.csr_matrix(np.ones((1, n_features), dtype=np.float64)))
        probe.var_names = [str(i) for i in range(n_features)]
        _, _, shuffled = dc.pp.extract(probe, empty=False)
        decoupler_shuffle = np.asarray(shuffled).astype(np.int64)
        input_order = np.empty(n_features, dtype=np.int64)
        input_order[decoupler_shuffle] = rank_feature_index
        matched_adata = anndata.AnnData(X=raw[:, input_order].tocsr())
        matched_adata.obs_names = exported.obs_names
        matched_adata.var_names = ids[input_order]
        _, _, check_order = dc.pp.extract(matched_adata, empty=False)
        tie_order_reproduced = bool(
            np.array_equal(np.asarray(check_order).astype(str), ids[rank_feature_index])
        )
        with Stopwatch() as decoupler_aucell_time:
            dc.mt.aucell(matched_adata, net_ids, tmin=TMIN, empty=False)
        decoupler_aucell = decoupler_scores(matched_adata, "aucell")
        print("AUCell matched done", flush=True)
        aucell_matched = {
            "comparisonKind": "exact parity: same counts, universe, n_up and tie order",
            "nUp": cytearc_n_up,
            "nFeatures": n_features,
            "tieOrderReproduced": tie_order_reproduced,
            "cytearcSeconds": cytearc_aucell_time.seconds,
            "decouplerSeconds": decoupler_aucell_time.seconds,
            **per_source(cytearc_aucell, decoupler_aucell),
        }
        log_normalised = sparse.csr_matrix(sparse.diags(10000.0 / totals) @ raw, dtype=np.float64)
        log_normalised.data = np.log1p(log_normalised.data)
        default_adata = anndata.AnnData(X=log_normalised)
        default_adata.obs_names = exported.obs_names
        default_adata.var_names = ids
        with Stopwatch() as decoupler_default_aucell_time:
            dc.mt.aucell(default_adata, net_ids, tmin=TMIN)
        decoupler_default_aucell = decoupler_scores(default_adata, "aucell")
        expressed = int((raw.getnnz(axis=0) > 0).sum())
        aucell_defaults = {
            "comparisonKind": "default versus default",
            "cytearc": {"nUp": cytearc_n_up, "nFeatures": n_features, "tieSeed": TIE_SEED},
            "decoupler": {
                "input": "log1p(counts * 1e4 / library size)",
                "featuresAfterEmptyRemoval": expressed,
                "nUp": int(np.clip(np.ceil(0.05 * expressed), 2, expressed)),
                "seconds": decoupler_default_aucell_time.seconds,
            },
            **per_source(cytearc_aucell, decoupler_default_aucell),
        }
        waggr_matched: dict[str, Any] = {}
        for log_transform in (False, True):
            with Stopwatch() as cytearc_waggr_time:
                waggr_ref = ds.scores.waggr(
                    net,
                    cells,
                    features=universe,
                    mode="wmean",
                    tmin=TMIN,
                    log_transform=log_transform,
                )
            cytearc_waggr = cytearc_scores(ds, waggr_ref)
            values = normalised.copy()
            if log_transform:
                values.data = np.log1p(values.data)
            waggr_adata = anndata.AnnData(X=values)
            waggr_adata.obs_names = exported.obs_names
            waggr_adata.var_names = ids
            with Stopwatch() as decoupler_waggr_time:
                dc.mt.waggr(waggr_adata, net_ids, tmin=TMIN, fun="wmean", times=0, empty=False)
            waggr_matched["log1p" if log_transform else "noLog"] = {
                "comparisonKind": "exact parity: same values, weights and denominator",
                "cytearcSeconds": cytearc_waggr_time.seconds,
                "decouplerSeconds": decoupler_waggr_time.seconds,
                **per_source(cytearc_waggr, decoupler_scores(waggr_adata, "waggr")),
            }
            if not log_transform:
                cytearc_waggr_default = cytearc_waggr
        print("WAGGR matched done", flush=True)
        expressed_mask = raw.getnnz(axis=0) > 0
        subset = np.sort(
            np.random.default_rng(SUBSET_SEED).choice(
                cell_idx.size, size=WAGGR_DEFAULT_CELLS, replace=False
            )
        )
        default_waggr_adata = anndata.AnnData(X=log_normalised[subset][:, expressed_mask].tocsr())
        default_waggr_adata.obs_names = exported.obs_names[subset]
        default_waggr_adata.var_names = ids[expressed_mask]
        with Stopwatch() as decoupler_default_waggr_time:
            dc.mt.waggr(default_waggr_adata, net_ids, tmin=TMIN, empty=False)
        waggr_defaults = {
            "comparisonKind": "default versus default",
            "cytearc": {
                "mode": "wmean",
                "logTransform": False,
                "values": f"counts * {size_factor:g} / library size",
            },
            "decoupler": {
                "fun": "wmean",
                "times": 1000,
                "seed": 42,
                "input": "log1p(counts * 1e4 / library size)",
                "output": "normalised enrichment score against 1,000 feature permutations",
                "features": int(expressed_mask.sum()),
                "cells": WAGGR_DEFAULT_CELLS,
                "cellSubsetSeed": SUBSET_SEED,
                "seconds": decoupler_default_waggr_time.seconds,
            },
            **per_source(
                cytearc_waggr_default.iloc[subset].reset_index(drop=True),
                decoupler_scores(default_waggr_adata, "waggr"),
            ),
        }
    print(
        "AUCell matched max|diff|",
        aucell_matched["maxAbsDiffOverall"],
        "tie order reproduced",
        aucell_matched["tieOrderReproduced"],
        "| AUCell defaults median r",
        round(aucell_defaults["pearson"]["median"], 4),
        "| WAGGR matched max|diff|",
        waggr_matched["noLog"]["maxAbsDiffOverall"],
        waggr_matched["log1p"]["maxAbsDiffOverall"],
        "| WAGGR defaults median r",
        round(waggr_defaults["pearson"]["median"], 4),
        flush=True,
    )
    payload = {
        "description": "CyteArc run_aucell and run_waggr against decoupler 2 aucell and waggr on PBMCs: exact parity with matched inputs and tie order, then default versus default.",
        "reference": {
            "package": "decoupler",
            "version": package_version("decoupler"),
            "functions": ["decoupler.mt.aucell", "decoupler.mt.waggr", "decoupler.pp.extract"],
        },
        "dataset": {
            "name": DATASET,
            "input": input_record(DATASET),
            "analysis": analysis_record(ds, run),
            "cells": int(cell_idx.size),
            "featureUniverse": "feature_universe of the run (all assay features)",
            "features": int(n_features),
            "featuresExpressedInCells": expressed,
        },
        "network": {
            **net_info,
            "edgesUsed": int(len(net)),
            "sources": sorted(net["source"].unique().tolist()),
            "tmin": TMIN,
            "weights": "1.0; S100A8 and S100A9 1.5; T_vs_Myeloid signed +1 and -1; marker sets use the marker score",
        },
        "aucellMatched": aucell_matched,
        "aucellDefaults": aucell_defaults,
        "waggrMatched": waggr_matched,
        "waggrDefaults": waggr_defaults,
        "notes": [
            "AUCell ranks within each cell, so library-size normalisation and log1p do not change its input ranks; the default comparison differs through the feature universe, n_up and tie-breaking only.",
            "decoupler removes all-zero features by default (empty=True) and breaks ties with its own fixed shuffle; CyteArc keeps the full universe and breaks ties with a permutation seeded by tie_seed.",
            "decoupler waggr returns a permutation-based normalised enrichment score by default (times=1000); times=0 returns the weighted mean itself.",
            "The permutation-based WAGGR comparison uses a seeded subset of 500 cells with the same feature universe.",
            "CyteArc stores enrichment scores as float32, so matched differences of order 1e-8 (AUCell) and 1e-7 relative (WAGGR) are storage rounding; maxAbsDiffReferenceRoundedToFloat32 repeats the comparison after rounding decoupler's float64 scores the same way.",
            "CyteArc matches targets without regard to case; decoupler received the matched feature ids, so both used the same targets.",
        ],
        "runtimeSeconds": total.seconds,
    }
    write_json("aucell.json", payload, directory=Path(request["outputDirectory"]))
    return payload


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path, nargs="?")
    args = parser.parse_args()
    if args.payload is not None:
        request = json.loads(args.payload.read_text())
    else:
        from experiments.methods import DATA, WORK

        request = {"workDirectory": str(WORK / "aucell"), "outputDirectory": str(DATA)}
    run(request)


if __name__ == "__main__":
    main()
