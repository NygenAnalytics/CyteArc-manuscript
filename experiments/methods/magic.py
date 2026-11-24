import warnings
from typing import Any

import cytearc
import numpy as np
from cytearc.neighbors.diffusion import bounded_diffusion_operator
from scipy import sparse
from scipy.stats import spearmanr

from experiments.methods import DATA
from experiments.methods.method_inputs import count_store, summarize
from experiments.runtime.agreement import numeric_agreement, package_version
from experiments.runtime.catalogue_counts import input_record
from experiments.runtime.common import Stopwatch, configure_cytearc, write_json
from experiments.runtime.rna_recipe import analysis_record, run_rna

EXPERIMENT = "magic"
DATASET = "tenx_5K_pbmc_rnaseq"
RUN_LABEL = "magic_comparison"
# The graph the record documents: 11 neighbours on 15 PCs of 500 variable genes, on CyteArc's default cell filter.
GRAPH_SETTINGS = {"hvg": {"top_n": 500}, "pca": {"dims": 15}, "neighbors": {"k": 11}}
N_GENES = 200
T_MATCHED = 2
SEED = 0


def per_gene(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    pearson = np.array(
        [np.corrcoef(first[:, j], second[:, j])[0, 1] for j in range(first.shape[1])]
    )
    spearman = np.array([spearmanr(first[:, j], second[:, j])[0] for j in range(first.shape[1])])
    max_abs = np.max(np.abs(first - second), axis=0)
    scale = np.max(np.abs(second), axis=0)
    return {
        "genes": int(first.shape[1]),
        "pearson": summarize(pearson),
        "pearsonPercentile5": float(np.percentile(pearson, 5)),
        "fractionPearsonAtLeast0p9": float(np.mean(pearson >= 0.9)),
        "spearman": summarize(spearman),
        "spearmanPercentile5": float(np.percentile(spearman, 5)),
        "fractionSpearmanAtLeast0p9": float(np.mean(spearman >= 0.9)),
        "maxAbsDiff": float(max_abs.max()),
        "maxAbsDiffOverGeneMaxAbs": float(np.max(max_abs / scale)),
    }


def gene_gene(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    upper = np.triu_indices(first.shape[1], k=1)
    corr_first = np.corrcoef(first.T)[upper]
    corr_second = np.corrcoef(second.T)[upper]
    return {
        "pairs": int(corr_first.size),
        **{
            key: value
            for key, value in numeric_agreement(corr_first, corr_second).items()
            if key in {"pearson", "spearman", "maxAbsDiff", "meanAbsDiff"}
        },
        "medianAbsCorrelationFirst": float(np.median(np.abs(corr_first))),
        "medianAbsCorrelationSecond": float(np.median(np.abs(corr_second))),
    }


def main() -> None:
    try:
        import graphtools
        import magic
        import scprep
    except ImportError as error:
        raise SystemExit(
            "magic-impute is required: uv run --with magic-impute python -m experiments.methods.magic"
        ) from error
    configure_cytearc()
    np.random.seed(SEED)
    with Stopwatch() as total:
        store = count_store(DATASET, EXPERIMENT, "pbmc")
        ds = cytearc.DataStore(str(store), default_assay="RNA", nthreads=4)
        run = run_rna(
            ds,
            label=RUN_LABEL,
            **GRAPH_SETTINGS,
            umap=False,
            cell_cycle=False,
            paris=False,
            doublets=False,
            markers=False,
        )
        graph = run["connectivity_map"]
        mask = np.asarray(
            ds.artifacts.load(run["analysis_cell_selection"])["values"][:], dtype=bool
        )
        cell_idx = np.flatnonzero(mask)
        exported = ds.to_anndata(run=run)
        cell_ids = ds.cells.fetch_all("ids").astype(str)[cell_idx]
        if not np.array_equal(exported.obs_names.to_numpy().astype(str), cell_ids):
            raise RuntimeError("exported cells do not follow the analysis selection")
        raw = sparse.csr_matrix(exported.X, dtype=np.float64)
        totals = np.asarray(ds.cells.fetch_all("RNA_nCounts"), dtype=np.float64)[cell_idx]
        totals[totals == 0] = 1.0
        size_factor = float(ds.RNA.sf)
        normalised = sparse.csr_matrix(sparse.diags(size_factor / totals) @ raw, dtype=np.float64)
        names = ds.RNA.feats.fetch_all("names").astype(str)
        upper_names = np.char.upper(names)
        _, inverse, counts = np.unique(upper_names, return_inverse=True, return_counts=True)
        unique_name = counts[inverse] == 1
        hvg = np.asarray(
            ds.artifacts.load(run["highly_variable_features"])["values"][:], dtype=bool
        )
        candidates = np.flatnonzero(hvg & unique_name)
        log_values = normalised[:, candidates].toarray()
        np.log1p(log_values, out=log_values)
        variance = log_values.var(axis=0)
        genes = np.sort(candidates[np.argsort(-variance, kind="stable")[:N_GENES]])
        gene_names = names[genes]
        observed = normalised[:, genes].toarray()
        with Stopwatch() as operator_time:
            diffusion = ds.imputation.diffusion(graph, t=T_MATCHED)
        with Stopwatch() as impute_time:
            cytearc_imputed = np.asarray(
                ds.imputation.compute_imputed([str(name) for name in gene_names], diffusion),
                dtype=np.float64,
            )
        cytearc_operator = ds.imputation.load_diffusion(diffusion).tocsr()
        adjacency = ds.graph.load(graph, symmetric=True, upper_only=False)
        graph_dtype = str(adjacency.dtype)
        with warnings.catch_warnings(record=True) as graph_warnings:
            warnings.simplefilter("always")
            precomputed = graphtools.Graph(
                adjacency.astype(np.float64),
                precomputed="affinity",
                n_pca=None,
                random_state=SEED,
                verbose=0,
            )
            with Stopwatch() as matched_time:
                matched_operator = magic.MAGIC(
                    t=T_MATCHED, n_pca=None, solver="exact", random_state=SEED, verbose=0
                )
                matched_operator.fit(observed, graph=precomputed)
                magic_matched = np.asarray(
                    matched_operator.transform(observed, genes="all_genes"), dtype=np.float64
                )
        magic_step = sparse.csr_matrix(precomputed.diff_op, dtype=np.float64)
        operator_difference = float(abs(magic_step @ magic_step - cytearc_operator).max())
        float64_operator = bounded_diffusion_operator(
            adjacency.astype(np.float64).tocsr(), T_MATCHED, memory_bytes=8 * 1024**3
        ).tocsr()
        float64_difference = float(abs(magic_step @ magic_step - float64_operator).max())
        matched = {
            "comparisonKind": "exact parity: same graph, same t, same input values",
            "t": T_MATCHED,
            "graphWeightDtype": graph_dtype,
            "maxAbsDiffDiffusionOperatorPowerT": operator_difference,
            "maxAbsDiffDiffusionOperatorPowerTFromFloat64Weights": float64_difference,
            "imputed": numeric_agreement(cytearc_imputed, magic_matched),
            "perGene": per_gene(cytearc_imputed, magic_matched),
            "geneGeneCorrelation": gene_gene(cytearc_imputed, magic_matched),
            "magicSeconds": matched_time.seconds,
            "graphtoolsWarnings": sorted({str(item.message) for item in graph_warnings}),
        }
        expressed = np.flatnonzero(np.asarray(raw.sum(axis=0)).ravel() > 0)
        positions = np.searchsorted(expressed, genes)
        if not np.array_equal(expressed[positions], genes):
            raise RuntimeError("selected genes must be expressed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Stopwatch() as default_same_time:
                default_same = magic.MAGIC(random_state=SEED, verbose=0)
                default_same.fit(normalised[:, expressed])
                magic_default_same = np.asarray(
                    default_same.transform(genes=positions), dtype=np.float64
                )
            with Stopwatch() as default_sqrt_time:
                preprocessed = scprep.transform.sqrt(
                    scprep.normalize.library_size_normalize(raw[:, expressed])
                )
                default_sqrt = magic.MAGIC(random_state=SEED, verbose=0)
                default_sqrt.fit(preprocessed)
                magic_default_sqrt = np.asarray(
                    default_sqrt.transform(genes=positions), dtype=np.float64
                )
        magic_defaults = {
            "knn": default_same.knn,
            "knnMax": default_same.knn_max,
            "decay": default_same.decay,
            "t": default_same.t,
            "nPca": default_same.n_pca,
            "solver": default_same.solver,
        }
        defaults = {
            "comparisonKind": "default versus default",
            "cytearc": {
                "graph": "connectivity_map (11 neighbours, 15 PCs of 500 HVGs)",
                "t": T_MATCHED,
                "values": f"counts * {size_factor:g} / library size, no log",
            },
            "magic": magic_defaults,
            "sameInputValues": {
                "magicInput": "the same normalised values, all expressed genes",
                "perGene": per_gene(cytearc_imputed, magic_default_same),
                "geneGeneCorrelation": gene_gene(cytearc_imputed, magic_default_same),
                "magicSeconds": default_same_time.seconds,
            },
            "magicDocumentedPreprocessing": {
                "magicInput": "scprep library_size_normalize (median library size) then sqrt, all expressed genes",
                "perGene": per_gene(cytearc_imputed, magic_default_sqrt),
                "geneGeneCorrelation": gene_gene(cytearc_imputed, magic_default_sqrt),
                "magicSeconds": default_sqrt_time.seconds,
            },
        }
        observed_context = {
            "observedVersusCyteArcGeneGene": gene_gene(observed, cytearc_imputed),
            "observedVersusMagicDefaultSameInputGeneGene": gene_gene(observed, magic_default_same),
        }
    print(
        "matched max|diff|",
        matched["imputed"]["maxAbsDiff"],
        "min per-gene r",
        matched["perGene"]["pearson"]["min"],
        "| defaults same input median r",
        round(defaults["sameInputValues"]["perGene"]["pearson"]["median"], 4),
        "gene-gene r",
        round(defaults["sameInputValues"]["geneGeneCorrelation"]["pearson"], 4),
        "| documented preprocessing median rho",
        round(defaults["magicDocumentedPreprocessing"]["perGene"]["spearman"]["median"], 4),
        flush=True,
    )
    payload = {
        "description": "CyteArc diffusion-operator imputation (run_diffusion_operator and get_imputed) against magic-impute: exact parity on the same graph and t, then default versus default on 200 variable genes.",
        "reference": {
            "package": "magic-impute",
            "version": package_version("magic-impute"),
            "graphtoolsVersion": package_version("graphtools"),
            "scprepVersion": package_version("scprep"),
        },
        "dataset": {
            "name": DATASET,
            "input": input_record(DATASET),
            "analysis": analysis_record(ds, run),
            "cells": int(cell_idx.size),
            "genesImputed": int(genes.size),
            "geneSelection": "top 200 of the run's HVGs by variance of log1p(normalised values), case-insensitively unique names",
            "genesExpressed": int(expressed.size),
            "genes": gene_names.tolist(),
        },
        "timings": {
            "cytearcDiffusionOperatorSeconds": operator_time.seconds,
            "cytearcGetImputed200GenesSeconds": impute_time.seconds,
        },
        "matched": matched,
        "defaults": defaults,
        "observedContext": observed_context,
        "notes": [
            "get_imputed diffuses counts scaled to 1,000 per cell without a log transform; MAGIC diffuses whatever matrix it is given.",
            "MAGIC.transform without data diffuses the matrix the graph was built from, which for a precomputed affinity is the affinity itself; the matched run therefore passes the gene matrix to transform explicitly (MAGIC warns that this is extrapolation; with the same cells it is the same computation).",
            "CyteArc graphs have no self loops; graphtools warns about a zero kernel diagonal but uses the affinity unchanged.",
            "CyteArc stores graph weights as float32 and builds the diffusion operator in that precision; graphtools received the same weights cast to float64. The matched residual is this rounding, as the operator rebuilt by CyteArc's own function from float64 weights shows.",
            "MAGIC defaults build their own graph (alpha-decay kernel, knn 5, 100 PCs) and use t=3, so the default comparison mixes graph, kernel and t differences.",
        ],
        "runtimeSeconds": total.seconds,
    }
    write_json("pbmc_magic_imputation_agreement.json", payload, directory=DATA)


if __name__ == "__main__":
    main()
