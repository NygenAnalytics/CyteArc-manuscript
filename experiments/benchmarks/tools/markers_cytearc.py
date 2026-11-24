"""CyteArc's marker search alone, on the common inputs of the marker-only comparison.

Usage: python markers_cytearc.py WORKERS BUDGET_BYTES H5AD EVENTS SUMMARY WORKDIR

The input is the common H5AD of tools/markers_common.py: the cells and genes of the other
tools' filter, raw counts, and the fixed groups in obs/group. The input preparation is the
conversion into a store on the container's local disk, which imports obs/group as a cluster
label artifact (convertMatrix), and opening the store (createObject); the step that the
comparison times is the marker search of every group against the rest on every gene, with
CyteArc's library-size normalisation (total counts of the stored genes). Afterwards the AUROC
and P value of a seeded sample of genes and groups (var/marker_sample,
uns/marker_sample_groups) are written to sample.csv in WORKDIR. CyteArc holds only what fits
within BUDGET_BYTES, three quarters of the container, as in the workflow comparison.
"""

import json
import sys
import time

import numpy as np

WORKERS, BUDGET, H5AD, EVENTS, SUMMARY, WORKDIR = sys.argv[1:7]
WORKERS, BUDGET = (int(WORKERS), int(BUDGET))
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


def main() -> None:
    import cytearc
    import h5py
    import pandas as pd

    SUMMARY_DATA["versions"] = {"cytearc": cytearc.__version__}
    SUMMARY_DATA["budgetBytes"] = BUDGET
    with h5py.File(H5AD, "r") as h5:
        SUMMARY_DATA["nnz"] = int(h5["X"]["data"].shape[0])
        SUMMARY_DATA["nCells"], SUMMARY_DATA["nGenes"] = (int(v) for v in h5["X"].attrs["shape"])
        sample_genes = np.flatnonzero(h5["var"]["marker_sample"][:])
        sample_groups = [str(int(g)) for g in h5["uns"]["marker_sample_groups"][:]]
    save_summary()
    cytearc.configure_output(level="WARNING", progress=False)
    store = f"{WORKDIR}/store.zarr"

    def convert():
        reader = cytearc.H5adReader(
            H5AD,
            matrix_key="X",
            cell_attrs_key="obs",
            cell_ids_key="_index",
            feature_attrs_key="var",
            feature_ids_key="_index",
            feature_name_key="feature_name",
            cluster_keys=["group"],
        )
        try:
            result = cytearc.H5adToZarr(
                reader, store, assay_name="RNA", mem_budget=BUDGET, nthreads=WORKERS
            ).dump()
        finally:
            reader.h5.close()
        return result.clusterArtifacts["group"]

    groups = step("convertMatrix", convert)

    def open_store():
        ds = cytearc.DataStore(
            store,
            assay_types={"RNA": "RNA"},
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=WORKERS,
            mem_budget=BUDGET,
        )
        everything = ds.features.snapshot(
            from_assay="RNA", mask=np.ones(ds.get_assay("RNA").feats.N, dtype=bool)
        )
        return (ds, everything)

    ds, everything = step("createObject", open_store)
    marker = step(
        "findMarkers",
        lambda: ds.markers.search(groups, from_assay="RNA", features=everything, nthreads=WORKERS),
    )
    labels = ds.artifacts.load(groups)["values"][:]
    SUMMARY_DATA.update(
        {
            "cellsAfterFilter": int(labels.size),
            "genesAfterFilter": int(ds.get_assay("RNA").feats.N),
            "groups": int(np.unique(labels).size),
        }
    )
    save_summary()

    def sample():
        ids = np.asarray(ds.get_assay("RNA").feats.fetch_all("ids")).astype(str)
        frames = []
        for group in sample_groups:
            frame = ds.markers.load(
                marker,
                group_id=group,
                min_score=None,
                min_frac_exp=None,
                min_auc=None,
                max_p_adj=None,
            )
            frame = frame[np.isin(frame["feature_index"].to_numpy(), sample_genes)]
            frames.append(
                pd.DataFrame(
                    {
                        "group": group,
                        "gene": ids[frame["feature_index"].to_numpy()],
                        "score": frame["score"].to_numpy(),
                        "auc": frame["auc"].to_numpy(),
                        "pval": frame["p_value"].to_numpy(),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    table = step("sampleStatistics", sample)
    table.to_csv(f"{WORKDIR}/sample.csv", index=False)
    SUMMARY_DATA["sampleRows"] = int(len(table))


if __name__ == "__main__":
    try:
        main()
    finally:
        save_summary()
