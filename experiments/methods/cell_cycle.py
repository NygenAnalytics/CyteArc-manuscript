from itertools import combinations, product
from typing import Any

import cytearc
import numpy as np
from cytearc.quality_control.cell_cycle_genes import g2m_phase_genes, s_phase_genes
from scipy import sparse

from experiments.methods import DATA
from experiments.methods.method_inputs import count_store, label_agreement, summarize
from experiments.runtime.agreement import numeric_agreement, package_version
from experiments.runtime.catalogue_counts import input_record
from experiments.runtime.common import Stopwatch, configure_cytearc, write_json
from experiments.runtime.rna_recipe import analysis_record, run_rna

EXPERIMENT = "cell_cycle"
DATASETS = {"pancreas": "bastidas-ponce_4K_pancreas-d15_rnaseq", "pbmc": "tenx_5K_pbmc_rnaseq"}
# The analysis each dataset is scored on: CyteArc's default cell filter and RNA recipe.
RUN_LABEL = "cell_cycle_comparison"
CELL_KEY = "cell_cycle_comparison_cells"
CYTEARC_SEEDS = (4466, 1, 2, 3, 4)
SCANPY_SEEDS = (0, 1, 2, 3, 4)
N_BINS = (50, 25)
DETERMINISTIC_CONTROL_SIZE = 1000000
PHASES = ("G1", "S", "G2M")


def map_gene_list(genes: list[str], names: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    by_name: dict[str, list[str]] = {}
    for name, feature_id in zip(names, ids, strict=True):
        by_name.setdefault(str(name).upper(), []).append(str(feature_id))
    mapped: list[str] = []
    missing: list[str] = []
    multiple: list[str] = []
    for gene in genes:
        hits = by_name.get(gene.upper(), [])
        if not hits:
            missing.append(gene)
            mapped.append(gene)
        else:
            if len(hits) > 1:
                multiple.append(gene)
            mapped.extend(hits)
    return {"mapped": mapped, "missing": missing, "multiple": multiple}


def cytearc_scores(
    ds: cytearc.DataStore, cells: cytearc.ArtifactRef, *, n_bins: int, seed: int
) -> dict[str, np.ndarray]:
    ref = ds.scores.cell_cycle(cells, n_bins=n_bins, rand_seed=seed)
    group = ds.artifacts.load(ref)
    return {
        "s": np.asarray(group["s_score"][:], dtype=np.float64),
        "g2m": np.asarray(group["g2m_score"][:], dtype=np.float64),
        "phase": np.asarray(group["phase"][:]).astype(str),
    }


def scanpy_scores(
    adata: Any, s_genes: list[str], g2m_genes: list[str], **kwargs: Any
) -> dict[str, np.ndarray]:
    import scanpy as sc

    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes, **kwargs)
    return {
        "s": adata.obs["S_score"].to_numpy(dtype=np.float64),
        "g2m": adata.obs["G2M_score"].to_numpy(dtype=np.float64),
        "phase": adata.obs["phase"].to_numpy().astype(str),
    }


def compare(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict:
    return {
        "S": numeric_agreement(first["s"], second["s"]),
        "G2M": numeric_agreement(first["g2m"], second["g2m"]),
        "phase": label_agreement(first["phase"], second["phase"], order=PHASES),
    }


def pair_summary(pairs: list[dict]) -> dict[str, Any]:
    return {
        "pairs": len(pairs),
        "pearsonS": summarize([p["S"]["pearson"] for p in pairs]),
        "pearsonG2M": summarize([p["G2M"]["pearson"] for p in pairs]),
        "maxAbsDiffS": summarize([p["S"]["maxAbsDiff"] for p in pairs]),
        "maxAbsDiffG2M": summarize([p["G2M"]["maxAbsDiff"] for p in pairs]),
        "phaseFractionIdentical": summarize([p["phase"]["fractionIdentical"] for p in pairs]),
        "phaseKappa": summarize([p["phase"]["cohenKappa"] for p in pairs]),
    }


def phase_fractions(phase: np.ndarray) -> dict[str, float]:
    return {label: float(np.mean(phase == label)) for label in PHASES}


def run_dataset(short_name: str, dataset: str) -> dict[str, Any]:
    import anndata
    import scanpy as sc

    store = count_store(dataset, EXPERIMENT, short_name)
    ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=4)
    run = run_rna(
        ds,
        label=RUN_LABEL,
        umap=False,
        cell_cycle=False,
        paris=False,
        doublets=False,
        markers=False,
    )
    cells = run["analysis_cell_selection"]
    mask = np.asarray(ds.artifacts.load(cells)["values"][:], dtype=bool)
    cell_idx = np.flatnonzero(mask)
    ds.cells.insert(column_name=CELL_KEY, values=mask, key="I", overwrite=True)
    exported = ds.to_anndata(run=run)
    cell_ids = ds.cells.fetch_all("ids").astype(str)[cell_idx]
    if not np.array_equal(exported.obs_names.to_numpy().astype(str), cell_ids):
        raise RuntimeError("exported cells do not follow the analysis selection")
    names = ds.RNA.feats.fetch_all("names").astype(str)
    ids = ds.RNA.feats.fetch_all("ids").astype(str)
    if exported.n_vars != ds.RNA.feats.N or not np.array_equal(
        exported.var_names.to_numpy().astype(str), ids
    ):
        raise RuntimeError("exported features do not cover the assay feature axis")
    raw = sparse.csr_matrix(exported.X, dtype=np.float64)
    totals = np.asarray(ds.cells.fetch_all("RNA_nCounts"), dtype=np.float64)[cell_idx]
    count_check = float(np.max(np.abs(np.asarray(raw.sum(axis=1)).ravel() - totals)))
    totals[totals == 0] = 1.0
    size_factor = float(ds.RNA.sf)
    scaled = sparse.csr_matrix(sparse.diags(size_factor / totals) @ raw, dtype=np.float64)
    matched_adata = anndata.AnnData(X=scaled, obs={"cell": cell_ids}, var={"names": names})
    matched_adata.obs_names = cell_ids
    matched_adata.var_names = ids
    matched_log_adata = anndata.AnnData(
        X=scaled.log1p(), obs={"cell": cell_ids}, var={"names": names}
    )
    matched_log_adata.obs_names = cell_ids
    matched_log_adata.var_names = ids
    s_map = map_gene_list(list(s_phase_genes), names, ids)
    g2m_map = map_gene_list(list(g2m_phase_genes), names, ids)
    s_list = s_map["mapped"]
    g2m_list = g2m_map["mapped"]
    deterministic: dict[str, Any] = {}
    for n_bins in N_BINS:
        cytearc_det = {}
        scanpy_det = {}
        for label, genes, mapped in (
            ("S", list(s_phase_genes), s_list),
            ("G2M", list(g2m_phase_genes), g2m_list),
        ):
            cytearc_det[label] = ds.RNA.score_features(
                genes,
                cell_key=CELL_KEY,
                ctrl_size=DETERMINISTIC_CONTROL_SIZE,
                n_bins=n_bins,
                rand_seed=0,
            )
            sc.tl.score_genes(
                matched_adata,
                mapped,
                ctrl_size=DETERMINISTIC_CONTROL_SIZE,
                n_bins=n_bins,
                score_name=f"det_{label}",
                random_state=0,
            )
            scanpy_det[label] = matched_adata.obs[f"det_{label}"].to_numpy(dtype=np.float64)
        deterministic[f"nBins{n_bins}"] = {
            "S": numeric_agreement(cytearc_det["S"], scanpy_det["S"]),
            "G2M": numeric_agreement(cytearc_det["G2M"], scanpy_det["G2M"]),
        }
    deterministic_public: dict[str, Any] = {}
    for n_bins in N_BINS:
        ref = ds.scores.cell_cycle(
            cells,
            ctrl_size=DETERMINISTIC_CONTROL_SIZE,
            n_bins=n_bins,
            rand_seed=0,
            log_transform=True,
        )
        group = ds.artifacts.load(ref)
        public = {
            "S": np.asarray(group["s_score"][:], dtype=np.float64),
            "G2M": np.asarray(group["g2m_score"][:], dtype=np.float64),
        }
        reference = {}
        for label, mapped in (("S", s_list), ("G2M", g2m_list)):
            sc.tl.score_genes(
                matched_log_adata,
                mapped,
                ctrl_size=DETERMINISTIC_CONTROL_SIZE,
                n_bins=n_bins,
                score_name=f"detlog_{label}",
                random_state=0,
            )
            reference[label] = matched_log_adata.obs[f"detlog_{label}"].to_numpy(dtype=np.float64)
        deterministic_public[f"nBins{n_bins}"] = {
            label: numeric_agreement(public[label], reference[label]) for label in ("S", "G2M")
        }
    matched: dict[str, Any] = {}
    seed_variability: dict[str, Any] = {}
    cytearc_default_seed: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for n_bins in N_BINS:
        cytearc_runs = {}
        with Stopwatch() as cytearc_time:
            for seed in CYTEARC_SEEDS:
                cytearc_runs[seed] = cytearc_scores(ds, cells, n_bins=n_bins, seed=seed)
        timings[f"cytearcFiveSeedsNBins{n_bins}Seconds"] = cytearc_time.seconds
        scanpy_runs = {}
        with Stopwatch() as scanpy_time:
            for seed in SCANPY_SEEDS:
                scanpy_runs[seed] = scanpy_scores(
                    matched_log_adata, s_list, g2m_list, n_bins=n_bins, random_state=seed
                )
        timings[f"scanpyFiveSeedsNBins{n_bins}Seconds"] = scanpy_time.seconds
        if n_bins == 50:
            cytearc_default_seed = cytearc_runs[CYTEARC_SEEDS[0]]
        matched[f"nBins{n_bins}"] = {
            "cytearcSeed": CYTEARC_SEEDS[0],
            "scanpySeed": SCANPY_SEEDS[0],
            **compare(cytearc_runs[CYTEARC_SEEDS[0]], scanpy_runs[SCANPY_SEEDS[0]]),
            "phaseFractionsCyteArc": phase_fractions(cytearc_runs[CYTEARC_SEEDS[0]]["phase"]),
            "phaseFractionsScanpy": phase_fractions(scanpy_runs[SCANPY_SEEDS[0]]["phase"]),
        }
        seed_variability[f"nBins{n_bins}"] = {
            "withinCyteArc": pair_summary(
                [
                    compare(cytearc_runs[a], cytearc_runs[b])
                    for a, b in combinations(CYTEARC_SEEDS, 2)
                ]
            ),
            "withinScanpy": pair_summary(
                [compare(scanpy_runs[a], scanpy_runs[b]) for a, b in combinations(SCANPY_SEEDS, 2)]
            ),
            "cytearcVersusScanpy": pair_summary(
                [
                    compare(cytearc_runs[a], scanpy_runs[b])
                    for a, b in product(CYTEARC_SEEDS, SCANPY_SEEDS)
                ]
            ),
            "sameSeed": pair_summary(
                [
                    compare(cytearc_runs[seed], scanpy_runs[seed])
                    for seed in sorted(set(CYTEARC_SEEDS) & set(SCANPY_SEEDS))
                ]
            ),
        }
    with Stopwatch() as cytearc_default_time:
        cytearc_default = cytearc_scores(ds, cells, n_bins=50, seed=4466)
    default_adata = anndata.AnnData(X=raw.copy())
    default_adata.obs_names = cell_ids
    default_adata.var_names = ids
    with Stopwatch() as scanpy_default_time:
        sc.pp.normalize_total(default_adata, target_sum=10000.0)
        sc.pp.log1p(default_adata)
        scanpy_default = scanpy_scores(default_adata, s_list, g2m_list)
    defaults = {
        "cytearc": {
            "normalization": f"counts * {size_factor:g} / library size, then log1p",
            "nBins": 50,
            "randSeed": 4466,
            "seconds": cytearc_default_time.seconds,
            "phaseFractions": phase_fractions(cytearc_default["phase"]),
        },
        "scanpy": {
            "normalization": "normalize_total(target_sum=1e4) then log1p",
            "nBins": 25,
            "randomState": 0,
            "seconds": scanpy_default_time.seconds,
            "phaseFractions": phase_fractions(scanpy_default["phase"]),
        },
        **compare(cytearc_default, scanpy_default),
    }
    if not np.array_equal(cytearc_default["s"], cytearc_default_seed["s"]):
        raise RuntimeError("repeated CyteArc default scoring was not reproducible")
    supplementary: dict[str, Any] = {}
    if {"S_score", "G2M_score"} <= set(ds.cells.columns):
        stored = {
            "s": np.asarray(ds.cells.fetch_all("S_score"), dtype=np.float64)[cell_idx],
            "g2m": np.asarray(ds.cells.fetch_all("G2M_score"), dtype=np.float64)[cell_idx],
        }
        supplementary["storedScores"] = {
            "note": "S_score and G2M_score shipped with the source AnnData of this store; their gene lists and normalisation are not recorded in the store",
            "cytearcDefaultVersusStored": {
                "S": numeric_agreement(cytearc_default["s"], stored["s"]),
                "G2M": numeric_agreement(cytearc_default["g2m"], stored["g2m"]),
            },
            "scanpyDefaultVersusStored": {
                "S": numeric_agreement(scanpy_default["s"], stored["s"]),
                "G2M": numeric_agreement(scanpy_default["g2m"], stored["g2m"]),
            },
        }
    return {
        "dataset": {
            "name": dataset,
            "input": input_record(dataset),
            "analysis": analysis_record(ds, run),
            "cellSelection": "analysis_cell_selection",
            "cells": int(cell_idx.size),
            "cellsInStore": int(ds.cells.N),
            "features": int(ds.RNA.feats.N),
            "sizeFactor": size_factor,
            "maxAbsDiffRawRowSumsVersusStoredTotals": count_check,
        },
        "genes": {
            "sListLength": len(s_phase_genes),
            "g2mListLength": len(g2m_phase_genes),
            "controlSize": min(len(s_phase_genes), len(g2m_phase_genes)),
            "sMissing": s_map["missing"],
            "g2mMissing": g2m_map["missing"],
            "sMatchedMoreThanOnce": s_map["multiple"],
            "g2mMatchedMoreThanOnce": g2m_map["multiple"],
            "sMatched": len(s_phase_genes) - len(s_map["missing"]),
            "g2mMatched": len(g2m_phase_genes) - len(g2m_map["missing"]),
        },
        "deterministicControls": deterministic,
        "deterministicControlsPublic": deterministic_public,
        "matched": matched,
        "seedVariability": seed_variability,
        "defaults": defaults,
        "supplementary": supplementary,
        "timings": timings,
    }


def main() -> None:
    try:
        import scanpy as sc
    except ImportError as error:
        raise SystemExit(
            "scanpy is required: uv run --with scanpy python -m experiments.methods.cell_cycle"
        ) from error
    configure_cytearc()
    sc.settings.verbosity = 0
    results: dict[str, Any] = {}
    with Stopwatch() as total:
        for short_name, dataset in DATASETS.items():
            with Stopwatch() as dataset_time:
                results[short_name] = run_dataset(short_name, dataset)
            results[short_name]["seconds"] = dataset_time.seconds
            matched = results[short_name]["matched"]["nBins50"]
            print(
                short_name,
                "deterministic max|diff| S",
                results[short_name]["deterministicControls"]["nBins50"]["S"]["maxAbsDiff"],
                "| matched r S",
                round(matched["S"]["pearson"], 4),
                "G2M",
                round(matched["G2M"]["pearson"], 4),
                "phase kappa",
                round(matched["phase"]["cohenKappa"], 3),
                "| defaults r S",
                round(results[short_name]["defaults"]["S"]["pearson"], 4),
                "kappa",
                round(results[short_name]["defaults"]["phase"]["cohenKappa"], 3),
                flush=True,
            )
    payload = {
        "description": "CyteArc run_cell_cycle_scoring against scanpy.tl.score_genes_cell_cycle with CyteArc's bundled S and G2M gene lists: a deterministic all-genes-per-bin control check, a matched comparison on identical normalised values and parameters, seed-to-seed variability within each implementation, and a default-versus-default comparison.",
        "reference": {
            "package": "scanpy",
            "version": package_version("scanpy"),
            "anndataVersion": package_version("anndata"),
            "functions": ["scanpy.tl.score_genes_cell_cycle", "scanpy.tl.score_genes"],
        },
        "cytearcFunctions": ["DataStore.run_cell_cycle_scoring", "RNAassay.score_features"],
        "parameters": {
            "geneLists": "cytearc.quality_control.cell_cycle_genes (human symbols, matched case-insensitively)",
            "cytearcSeeds": list(CYTEARC_SEEDS),
            "scanpySeeds": list(SCANPY_SEEDS),
            "nBins": list(N_BINS),
            "deterministicControlSize": DETERMINISTIC_CONTROL_SIZE,
        },
        "comparisonKinds": {
            "deterministicControls": "exact parity: identical inputs, parameters and control sets",
            "deterministicControlsPublic": "exact parity through DataStore.run_cell_cycle_scoring(ctrl_size, log_transform=True) against Scanpy on the same log1p values",
            "matched": "matched inputs and parameters; control genes are random draws in each implementation",
            "seedVariability": "within-implementation seed-to-seed agreement against between-implementation agreement; sameSeed compares the two implementations at equal seeds",
            "defaults": "default versus default: each implementation with its own documented normalisation and parameters",
        },
        "datasets": results,
        "notes": [
            "CyteArc scores library-size-normalised counts scaled to 1,000 and log-transformed with log1p (CyteArc's default); the matched comparison gives Scanpy exactly these values. The assay-level deterministic check uses the values without the log transform.",
            "Both implementations set the control size to the length of the shorter input list, counting symbols that are absent from the assay.",
            "CyteArc draws the control genes of every expression bin from one generator seeded with rand_seed, in the order Scanpy draws them after seeding the global NumPy generator once per score_genes call. Different seeds give different draws, so identical control sets require the deterministic setting or equal seeds (sameSeed).",
            "CyteArc matches symbols case-insensitively; Scanpy matches exact var names, so the lists given to Scanpy were translated to the matched feature ids.",
        ],
        "runtimeSeconds": total.seconds,
    }
    write_json("cell_cycle_scanpy_agreement.json", payload, directory=DATA)


if __name__ == "__main__":
    main()
