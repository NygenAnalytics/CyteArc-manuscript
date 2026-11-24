"""Reference mapping with scANVI (scArches surgery) or CellTypist on the held-out lung study.

Usage: python mapping_python.py MODE WORKERS REFERENCE EVENTS SUMMARY WORKDIR
The query H5AD is given in the QUERY_H5AD environment variable. Both files come from
prepare_hlca.py. Predictions for every query cell and annotation level, with a
confidence for abstention, are written to WORKDIR/predictions.csv.

scanvi      scVI and scANVI trained on the reference with the settings that scvi-tools
            recommends for reference mapping (2,000 variable genes by the seurat_v3 flavour
            with the study as batch, two layers, 30 latent dimensions, layer norm, no batch
            norm, encoded covariates), labels at the finest level; the query is mapped by
            scArches surgery (100 epochs, no weight decay). Coarser levels are derived from
            the finest prediction through the reference's own label hierarchy. Runs on a GPU.
celltypist  One CellTypist model per annotation level, trained on all genes of the
            log-normalised reference with mini-batch stochastic gradient descent and feature
            selection, applied without majority voting; the confidence is the probability of
            the predicted label.
celltypist_hvg  The same, on the 2,000 variable genes that scANVI uses.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

MODE, WORKERS, REFERENCE, EVENTS, SUMMARY, WORKDIR = sys.argv[1:7]
WORKERS = int(WORKERS)
QUERY = os.environ["QUERY_H5AD"]
LEVELS = tuple(
    os.environ.get("MAPPING_LABEL_COLUMNS", "ann_level_3,cell_type,ann_finest_level").split(",")
)
SUMMARY_DATA: dict = {"mode": MODE, "labelColumns": LEVELS}


def emit(step: str, event: str) -> None:
    with open(EVENTS, "a") as handle:
        handle.write(json.dumps({"step": step, "event": event, "t": time.time()}) + "\n")


def step(name: str, fn):
    emit(name, "start")
    value = fn()
    emit(name, "end")
    return value


def save() -> None:
    Path(SUMMARY).write_text(json.dumps(SUMMARY_DATA, default=str))


def study_column(frame: pd.DataFrame) -> str:
    return next((c for c in ("dataset", "study", "donor_id") if c in frame.columns))


def hierarchy(reference_obs: pd.DataFrame, level: str) -> dict:
    table = reference_obs.groupby("ann_finest_level", observed=True)[level].agg(
        lambda s: s.value_counts().index[0]
    )
    return table.to_dict()


def run_scanvi() -> pd.DataFrame:
    import anndata as ad
    import scanpy as sc
    import scvi
    import torch

    scvi.settings.seed = 4444
    SUMMARY_DATA["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    reference = step("readReference", lambda: ad.read_h5ad(REFERENCE))
    query = step("readQuery", lambda: ad.read_h5ad(QUERY))
    batch = study_column(reference.obs)

    def variable_genes():
        sc.pp.highly_variable_genes(
            reference, n_top_genes=2000, flavor="seurat_v3", batch_key=batch
        )
        return reference.var_names[reference.var["highly_variable"].to_numpy()]

    genes = step("markHvgs", variable_genes)
    reference = reference[:, genes].copy()
    holder: dict = {}

    def train_scvi():
        scvi.model.SCVI.setup_anndata(reference, batch_key=batch)
        model = scvi.model.SCVI(
            reference,
            n_layers=2,
            n_latent=30,
            gene_likelihood="nb",
            use_layer_norm="both",
            use_batch_norm="none",
            encode_covariates=True,
            dropout_rate=0.2,
        )
        model.train(accelerator="auto")
        holder["scvi"] = model
        SUMMARY_DATA["scviEpochs"] = int(model.history["elbo_train"].shape[0])

    step("trainScvi", train_scvi)

    def train_scanvi():
        model = scvi.model.SCANVI.from_scvi_model(
            holder["scvi"], labels_key="ann_finest_level", unlabeled_category="Unknown"
        )
        model.train(max_epochs=20, n_samples_per_label=100, accelerator="auto")
        holder["scanvi"] = model

    step("trainScanvi", train_scanvi)

    def map_query():
        subset = query[:, genes].copy()
        subset.obs["ann_finest_level"] = "Unknown"
        scvi.model.SCANVI.prepare_query_anndata(subset, holder["scanvi"])
        mapped = scvi.model.SCANVI.load_query_data(subset, holder["scanvi"])
        mapped.train(max_epochs=100, plan_kwargs={"weight_decay": 0.0}, accelerator="auto")
        return mapped.predict(soft=True)

    probabilities = step("mapQuery", map_query)
    finest = probabilities.idxmax(axis=1).to_numpy()
    confidence = probabilities.max(axis=1).to_numpy()
    frame = pd.DataFrame({"cell": query.obs_names})
    for level in LEVELS:
        frame[f"true_{level}"] = query.obs[level].astype(str).to_numpy()
        if level == "ann_finest_level":
            frame[f"pred_{level}"] = finest
        else:
            lookup = hierarchy(reference.obs, level)
            frame[f"pred_{level}"] = [lookup.get(label, label) for label in finest]
        frame[f"conf_{level}"] = confidence
    SUMMARY_DATA["versions"] = {
        "scvi-tools": scvi.__version__,
        "torch": torch.__version__,
        "scanpy": sc.__version__,
    }
    return frame


def run_celltypist(variable_only: bool = False) -> pd.DataFrame:
    import gc
    from importlib.metadata import version

    import anndata as ad
    import celltypist
    import numpy as np
    import scanpy as sc

    np.random.seed(4444)
    SUMMARY_DATA["seed"] = 4444
    reference = step("readReference", lambda: ad.read_h5ad(REFERENCE))
    if variable_only:

        def variable_genes():
            sc.pp.highly_variable_genes(
                reference,
                n_top_genes=2000,
                flavor="seurat_v3",
                batch_key=study_column(reference.obs),
            )
            return reference.var_names[reference.var["highly_variable"].to_numpy()]

        genes = step("markHvgs", variable_genes)

    def normalise(adata):
        sc.pp.normalize_total(adata, target_sum=10000.0)
        sc.pp.log1p(adata)

    step("normalizeReference", lambda: normalise(reference))
    if variable_only:
        reference = step("subsetReferenceFeatures", lambda: reference[:, genes].copy())
    model_paths = {}
    SUMMARY_DATA["referenceModelBytes"] = {}
    mini_batch = reference.n_obs >= 10000
    SUMMARY_DATA["miniBatch"] = mini_batch
    SUMMARY_DATA["miniBatchSize"] = 1000 if mini_batch else None
    SUMMARY_DATA["normalizationUniverse"] = (
        "All shared input genes before optional variable-gene subsetting"
    )
    for level in LEVELS:
        model = step(
            f"train_{level}",
            lambda: celltypist.train(
                reference,
                labels=level,
                use_SGD=True,
                mini_batch=mini_batch,
                feature_selection=True,
                n_jobs=WORKERS,
                check_expression=False,
                random_state=4444,
            ),
        )
        path = Path(WORKDIR) / f"reference_{level}.pkl"
        step(f"persistReference_{level}", lambda: model.write(str(path)))
        model_paths[level] = path
        SUMMARY_DATA["referenceModelBytes"][level] = path.stat().st_size
        del model
    reference = None
    gc.collect()
    query = step("readQuery", lambda: ad.read_h5ad(QUERY))
    step("normalizeQuery", lambda: normalise(query))
    if variable_only:
        query = step("subsetQueryFeatures", lambda: query[:, genes].copy())
    frame = pd.DataFrame({"cell": query.obs_names})
    for level in LEVELS:
        model = step(
            f"loadPreparedReference_{level}",
            lambda: celltypist.models.Model.load(str(model_paths[level])),
        )
        result = step(
            f"annotate_{level}",
            lambda: celltypist.annotate(query, model=model, majority_voting=False),
        )
        frame[f"true_{level}"] = query.obs[level].astype(str).to_numpy()
        frame[f"pred_{level}"] = result.predicted_labels["predicted_labels"].astype(str).to_numpy()
        frame[f"conf_{level}"] = result.probability_matrix.max(axis=1).to_numpy()
    SUMMARY_DATA["versions"] = {"celltypist": version("celltypist"), "scanpy": version("scanpy")}
    return frame


def main() -> None:
    try:
        if MODE == "scanvi":
            frame = run_scanvi()
        else:
            frame = run_celltypist(variable_only=MODE == "celltypist_hvg")
        frame.to_csv(Path(WORKDIR) / "predictions.csv", index=False)
        SUMMARY_DATA["queryCells"] = int(frame.shape[0])
        for level in LEVELS:
            SUMMARY_DATA[f"agreementNoAbstention_{level}"] = float(
                (frame[f"true_{level}"] == frame[f"pred_{level}"]).mean()
            )
    finally:
        save()


if __name__ == "__main__":
    main()
