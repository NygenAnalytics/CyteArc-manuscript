import urllib.request
from typing import Any

import cytearc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.methods import DATA
from experiments.methods import INPUTS as CACHE
from experiments.methods.method_inputs import count_store
from experiments.runtime.agreement import package_version
from experiments.runtime.catalogue_counts import input_record
from experiments.runtime.common import Stopwatch, configure_cytearc, write_json
from experiments.runtime.rna_recipe import analysis_cells, run_rna

EXPERIMENT = "doublets"
CONDITIONS = {"ctrl": "kang_15K_pbmc_rnaseq", "stim": "kang_14K_ifnb-pbmc_rnaseq"}
GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.total.tsne.df.tsv.gz"
GEO_FILE = CACHE / "GSE96583" / "GSE96583_batch2.total.tsne.df.tsv.gz"
PIPELINE_LABEL = "kang_doublet_comparison"
DOCUMENTATION_LABEL = "kang_documentation_settings"
# The documentation settings, as the record of the first comparison states them for each condition's run.
DOCUMENTATION_SETTINGS = {
    "filtering": {
        "method": "manual",
        "attrs": ["RNA_nCounts", "RNA_nFeatures"],
        "lows": [500, 200],
        "highs": [15000, 4000],
    },
    "hvg": {"top_n": 2000},
    "pca": {"dims": 25},
    "neighbors": {"k": 21},
    "leiden": {"partitions": [1.0]},
}
SCRUBLET_SEED = 0


def demuxlet_calls() -> pd.DataFrame:
    if not GEO_FILE.exists():
        GEO_FILE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GEO_URL, GEO_FILE)
    return pd.read_csv(GEO_FILE, sep="\t", index_col=0)


def join_calls(
    calls: pd.DataFrame, condition: str, barcodes: np.ndarray
) -> tuple[pd.Series, dict[str, int]]:
    subset = calls[calls["stim"] == condition].copy()
    direct = pd.Index(barcodes).isin(subset.index)
    remapped_index = subset.index.str.replace("-11$", "-1", regex=True)
    remapped_count = int((remapped_index != subset.index).sum())
    subset.index = remapped_index
    if subset.index.has_duplicates:
        raise RuntimeError(f"remapping produced duplicate {condition} barcodes")
    joined = subset["multiplets"].reindex(barcodes)
    report = {
        "geoCells": int(len(subset)),
        "geoBarcodesWithSuffix11": remapped_count,
        "storeCells": int(len(barcodes)),
        "matchedDirectly": int(direct.sum()),
        "matchedAfterRemap": int(joined.notna().sum()),
        "unmatched": int(joined.isna().sum()),
    }
    return (joined, report)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def run_settings(ds: cytearc.DataStore, run) -> dict:
    """The analysis settings a run resolved: variable genes, PCs, neighbors, Leiden, filter and neighbor index."""
    config = run.report()["run"]["config"]
    index = ds.artifacts.inspect(run["ann_index"]).parameters or {}
    return {
        "hvgCount": config["hvgCount"],
        "pcaDims": config["pcaDims"],
        "neighborsK": config["neighborsK"],
        "leiden": config["leiden"],
        "filtering": config["filtering"],
        "annIndex": {k: index[k] for k in ("ann_m", "ann_ef", "ann_efc") if k in index},
    }


def cytearc_default_run(ds: cytearc.DataStore) -> tuple[np.ndarray, np.ndarray, dict]:
    with Stopwatch() as elapsed:
        run = run_rna(
            ds,
            label=PIPELINE_LABEL,
            filtering=False,
            umap=False,
            cell_cycle=False,
            paris=False,
            markers=False,
            doublets=True,
        )
    mask = analysis_cells(ds, run)
    scores = np.asarray(ds.artifacts.load(run["doublets"])["values"][:], np.float64)
    status = ds.artifacts.inspect(run["doublets"])
    info = {
        "seconds": elapsed.seconds,
        **run_settings(ds, run),
        "doubletParameters": status.parameters,
    }
    return (mask, scores, info)


def cytearc_documentation_run(ds: cytearc.DataStore) -> tuple[np.ndarray, np.ndarray, dict]:
    """Doublet scores on the clusters and graph of an analysis with the documentation settings."""
    run = run_rna(
        ds,
        label=DOCUMENTATION_LABEL,
        **DOCUMENTATION_SETTINGS,
        umap=False,
        cell_cycle=False,
        paris=False,
        doublets=False,
        markers=False,
    )
    with Stopwatch() as elapsed:
        ref = ds.qc.doublets(run["clusters"], run["connectivity_map"])
    mask = analysis_cells(ds, run)
    scores = np.asarray(ds.artifacts.load(ref)["values"][:], np.float64)
    info = {
        "seconds": elapsed.seconds,
        **run_settings(ds, run),
        "doubletParameters": ds.artifacts.inspect(ref).parameters,
    }
    return (mask, scores, info)


def scrublet_scores(ds: cytearc.DataStore, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    import anndata
    import scanpy as sc

    key = "doublet_comparison_cells"
    ds.cells.insert(column_name=key, values=mask, key="I", overwrite=True)
    exported = ds.to_anndata(cell_key=key)
    adata = anndata.AnnData(X=sparse.csr_matrix(exported.X, dtype=np.float32))
    adata.obs_names = exported.obs_names
    adata.var_names = exported.var_names
    with Stopwatch() as elapsed:
        sc.pp.scrublet(adata, random_state=SCRUBLET_SEED, verbose=False)
    return (
        adata.obs["doublet_score"].to_numpy(dtype=np.float64),
        {
            "seconds": elapsed.seconds,
            "cells": int(adata.n_obs),
            "threshold": float(adata.uns["scrublet"].get("threshold", np.nan)),
            "predictedDoubletFraction": float(adata.obs["predicted_doublet"].mean()),
        },
    )


def main() -> None:
    try:
        from importlib import import_module

        import scanpy as sc

        import_module("skimage")
    except ImportError as error:
        raise SystemExit(
            "scanpy and scikit-image are required: uv run --with scanpy --with scikit-image python -m experiments.methods.kang_doublet_score_comparison"
        ) from error
    configure_cytearc()
    sc.settings.verbosity = 0
    calls = demuxlet_calls()
    per_condition: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    with Stopwatch() as total:
        for condition, dataset in CONDITIONS.items():
            store = count_store(dataset, EXPERIMENT, condition)
            ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=4)
            barcodes = ds.cells.fetch_all("ids").astype(str)
            joined, join_report = join_calls(calls, condition, barcodes)
            if join_report["unmatched"]:
                raise RuntimeError(f"{condition}: barcodes without a demuxlet call")
            default_mask, default_scores, default_info = cytearc_default_run(ds)
            docs_mask, docs_scores, docs_info = cytearc_documentation_run(ds)
            scrublet, scrublet_info = scrublet_scores(ds, default_mask)
            frame = pd.DataFrame(
                {
                    "condition": condition,
                    "barcode": barcodes,
                    "call": joined.to_numpy(),
                    "nCounts": np.asarray(ds.cells.fetch_all("RNA_nCounts"), float),
                    "cytearcDefault": np.nan,
                    "cytearcDocsRun": np.nan,
                    "scrublet": np.nan,
                }
            )
            frame.loc[default_mask, "cytearcDefault"] = default_scores
            frame.loc[docs_mask, "cytearcDocsRun"] = docs_scores
            frame.loc[default_mask, "scrublet"] = scrublet
            frames.append(frame)
            per_condition[condition] = {
                "dataset": dataset,
                "input": input_record(dataset),
                "join": join_report,
                "callsInStore": joined.value_counts().to_dict(),
                "callsInGeoTable": calls.loc[calls["stim"] == condition, "multiplets"]
                .value_counts()
                .to_dict(),
                "storeTotalCountsRange": [
                    float(frame["nCounts"].min()),
                    float(frame["nCounts"].max()),
                ],
                "cytearcDefaultRun": {**default_info, "cellsScored": int(default_mask.sum())},
                "cytearcDocsDefaultRun": {**docs_info, "cellsScored": int(docs_mask.sum())},
                "scrublet": scrublet_info,
            }
        table = pd.concat(frames, ignore_index=True)
        methods = ("cytearcDefault", "cytearcDocsRun", "scrublet", "nCounts")
        evaluable = table["call"].isin(["singlet", "doublet"]) & table[list(methods)].notna().all(
            axis=1
        )
        results: dict[str, Any] = {}
        for scope in ("ctrl", "stim", "pooled"):
            rows = evaluable & (table["condition"].eq(scope) if scope != "pooled" else True)
            labels = (table.loc[rows, "call"] == "doublet").to_numpy(dtype=int)
            results[scope] = {
                "cells": int(rows.sum()),
                "doublets": int(labels.sum()),
                "doubletPrevalence": float(labels.mean()),
                **{
                    method: metrics(labels, table.loc[rows, method].to_numpy())
                    for method in methods
                },
            }
            print(
                scope,
                {
                    method: {k: round(v, 4) for k, v in results[scope][method].items()}
                    for method in methods
                },
                flush=True,
            )
        dropped = {
            "ambiguousCalls": int((table["call"] == "ambs").sum()),
            "unscoredByAnyMethod": int(
                (table["call"].isin(["singlet", "doublet"]) & ~evaluable).sum()
            ),
        }
    payload = {
        "description": "CyteArc doublet scores against demuxlet genotype-based multiplet calls for the Kang 2018 PBMCs, with Scrublet and total counts as comparators.",
        "groundTruth": {
            "source": "GEO GSE96583, GSE96583_batch2.total.tsne.df.tsv.gz (demuxlet calls)",
            "url": GEO_URL,
            "positiveClass": "doublet",
            "negativeClass": "singlet",
            "dropped": "ambs (ambiguous)",
            "limitation": "demuxlet detects only doublets between donors",
        },
        "reference": {
            "comparator": "scanpy.pp.scrublet with defaults (expected_doublet_rate 0.05, sim_doublet_ratio 2, 30 PCs)",
            "scanpyVersion": package_version("scanpy"),
            "scikitImageVersion": package_version("scikit-image"),
            "randomState": SCRUBLET_SEED,
            "baseline": "RNA_nCounts (total counts per cell)",
        },
        "comparisonKind": "accuracy against genotype-based ground truth",
        "conditions": per_condition,
        "dropped": dropped,
        "results": results,
        "notes": [
            "Each condition's store holds every barcode of the catalogue's count files, which are every barcode of the GEO table (callsInStore against callsInGeoTable); the documentation-settings run scores the cells its manual filter keeps, and only cells every method scored are evaluated.",
            "Pooled metrics concatenate the per-condition scores; each condition is a separate 10x lane, so doublets only form within a condition.",
            "AUROC and AUPRC use the continuous scores of every method, so they do not depend on Scrublet's automatic threshold (reported per condition with the fraction of cells it calls doublets).",
            "CyteArc pipeline filtering was disabled so that all store cells are scored; the default MAD filter would remove high-count cells, which are enriched for doublets.",
        ],
        "runtimeSeconds": total.seconds,
    }
    write_json("kang_doublet_score_comparison.json", payload, directory=DATA)


if __name__ == "__main__":
    main()
