"""CyteArc's RNA recipe, DataStore.pipeline.run, with stage settings given as each stage's keyword arguments, and
the record of the settings an analysis resolved."""

from collections.abc import Mapping
from typing import Any

INDEX_PARAMETERS = ("ann_metric", "ann_m", "ann_ef", "ann_efc", "rand_state", "ann_parallel")


def run_rna(
    ds: Any,
    *,
    label: str,
    cells: str = "I",
    filtering: bool | Mapping[str, object] = True,
    hvg: Mapping[str, object] | None = None,
    pca: Mapping[str, object] | None = None,
    neighbors: Mapping[str, object] | None = None,
    leiden: Mapping[str, object] | bool = True,
    **switches: bool,
) -> Any:
    """CyteArc's RNA recipe, DataStore.pipeline.run, with each stage's settings given as that stage's keyword
    arguments, such as hvg={"top_n": 500}, pca={"dims": 15}, neighbors={"k": 11} and leiden={"partitions": [0.5]},
    and CyteArc's defaults for every setting not given. switches turns stages on or off (umap, cell_cycle, paris,
    doublets, markers)."""
    params: dict[str, dict[str, object]] = {}
    for name, value in (("hvg", hvg), ("pca", pca), ("neighbors", neighbors)):
        if value is not None:
            params[name] = dict(value)
    if isinstance(leiden, Mapping):
        params["leiden"] = dict(leiden)
    elif not isinstance(leiden, bool):
        raise TypeError("leiden must be a mapping of Leiden settings or a bool")
    return ds.pipeline.run(
        label=label,
        cell_key=cells,
        filtering=filtering,
        leiden=bool(leiden),
        params=params or None,
        **switches,
    )


def analysis_cells(ds: Any, run: Any) -> Any:
    """The cells an analysis kept, as a boolean mask over every cell of the store; its graph, clusters and other
    per-cell outputs follow these cells in store order."""
    import numpy as np

    return np.asarray(ds.artifacts.load(run["analysis_cell_selection"])["values"][:], dtype=bool)


def analysis_record(ds: Any, run: Any) -> dict[str, Any]:
    """The settings an analysis ran with, as CyteArc resolved them: cell filter, variable genes, PCs, neighbors,
    Leiden partitions, the neighbor index's HNSW parameters, and the cells the analysis kept."""
    import cytearc

    config = run.report()["run"]["config"]
    index = ds.artifacts.inspect(run["ann_index"]).parameters or {}
    selected = analysis_cells(ds, run)
    return {
        "cytearcVersion": cytearc.__version__,
        "filtering": config["filtering"],
        "hvgCount": config["hvgCount"],
        "pcaDims": config["pcaDims"],
        "neighborsK": config["neighborsK"],
        "leiden": config["leiden"],
        "annIndex": {key: index[key] for key in INDEX_PARAMETERS if key in index},
        "analysisCells": int(selected.sum()),
    }
