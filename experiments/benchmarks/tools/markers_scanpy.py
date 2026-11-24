"""Scanpy's Wilcoxon marker test alone, on the common inputs of the marker-only comparison.

Usage: python markers_scanpy.py MODE WORKERS H5AD EVENTS SUMMARY WORKDIR
MODE is "csr" (the matrix as read, cell rows) or "csc" (converted to gene columns first).

The input is the common H5AD of tools/markers_common.py: the cells and genes of the other
tools' filter, raw counts, and the fixed groups in obs/group. The steps are the input
preparation (read, normalise to total counts and log1p, and for CSC the conversion, each timed
separately), then the test that the comparison times: rank_genes_groups with the Wilcoxon
method, every group against the rest on every gene, as in the Scanpy workflow. Afterwards the
statistics of a seeded sample of genes and groups (var/marker_sample,
uns/marker_sample_groups) are written to sample.csv in WORKDIR, from the same test restricted
to the sampled genes: the rank sum of a group, and so its U statistic and AUROC, follow from
Scanpy's z-score without tie correction.
"""

import json
import sys
import time

import numpy as np

MODE, WORKERS, H5AD, EVENTS, SUMMARY, WORKDIR = sys.argv[1:7]
WORKERS = int(WORKERS)
SUMMARY_DATA: dict = {}


def emit(step: str, event: str, **extra) -> None:
    with open(EVENTS, "a") as handle:
        handle.write(json.dumps({"step": step, "event": event, "t": time.time(), **extra}) + "\n")


def step(name: str, fn):
    emit(name, "start")
    value = fn()
    emit(name, "end")
    return value


def save_summary() -> None:
    with open(SUMMARY, "w") as handle:
        json.dump(SUMMARY_DATA, handle, default=str)


def sample_table(adata, sample_genes: np.ndarray, sample_groups: list[str]):
    import pandas as pd
    import scanpy as sc

    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[sample_genes] = True
    sc.tl.rank_genes_groups(
        adata, "group", method="wilcoxon", mask_var=mask, n_genes=None, key_added="sample"
    )
    result = adata.uns["sample"]
    sizes = adata.obs["group"].value_counts()
    total = adata.n_obs
    frames = []
    for group in sample_groups:
        n1 = float(sizes[group])
        n2 = total - n1
        z = np.asarray(result["scores"][group], dtype=np.float64)
        rank_sum = z * np.sqrt(n1 * n2 * (total + 1) / 12.0) + n1 * (total + 1) / 2.0
        u = rank_sum - n1 * (n1 + 1) / 2.0
        frames.append(
            pd.DataFrame(
                {
                    "group": group,
                    "gene": np.asarray(result["names"][group]).astype(str),
                    "score": z,
                    "u": u,
                    "auc": u / (n1 * n2),
                    "pval": np.asarray(result["pvals"][group], dtype=np.float64),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    import anndata as ad
    import h5py
    import pandas as pd
    import scanpy as sc

    if MODE not in ("csr", "csc"):
        raise SystemExit("MODE must be csr or csc")
    SUMMARY_DATA["versions"] = {"scanpy": sc.__version__, "anndata": ad.__version__}
    SUMMARY_DATA["markerFormat"] = MODE
    with h5py.File(H5AD, "r") as h5:
        SUMMARY_DATA["nnz"] = int(h5["X"]["data"].shape[0])
        SUMMARY_DATA["nCells"], SUMMARY_DATA["nGenes"] = (int(v) for v in h5["X"].attrs["shape"])
        sample_genes = np.flatnonzero(h5["var"]["marker_sample"][:])
        sample_groups = [str(int(g)) for g in h5["uns"]["marker_sample_groups"][:]]
    save_summary()
    sc.settings.n_jobs = WORKERS

    def read():
        adata = ad.read_h5ad(H5AD)
        adata.obs["group"] = pd.Categorical(adata.obs["group"].astype(int).astype(str))
        return adata

    adata = step("read", read)

    def normalize() -> None:
        sc.pp.normalize_total(adata, target_sum=10000.0)
        sc.pp.log1p(adata)

    step("runNormalization", normalize)
    if MODE == "csc":

        def convert() -> None:
            adata.X = adata.X.tocsc()

        step("convertCsc", convert)
    step(
        "findMarkers",
        lambda: sc.tl.rank_genes_groups(adata, "group", method="wilcoxon", n_genes=100),
    )
    SUMMARY_DATA.update(
        {
            "cellsAfterFilter": int(adata.n_obs),
            "genesAfterFilter": int(adata.n_vars),
            "groups": int(adata.obs["group"].nunique()),
        }
    )
    save_summary()
    table = step("sampleStatistics", lambda: sample_table(adata, sample_genes, sample_groups))
    table.to_csv(f"{WORKDIR}/sample.csv", index=False)
    SUMMARY_DATA["sampleRows"] = int(len(table))


if __name__ == "__main__":
    try:
        main()
    finally:
        save_summary()
