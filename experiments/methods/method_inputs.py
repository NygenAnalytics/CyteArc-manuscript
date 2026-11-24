from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.methods import INPUTS as DATASETS
from experiments.methods import WORK
from experiments.runtime.catalogue_counts import build_store


def count_store(
    dataset: str, experiment: str, short_name: str, *, inputs: Path = DATASETS, workers: int = 4
) -> Path:
    """A fresh CyteArc store of the dataset's pinned catalogue count files, in the experiment's work folder."""
    return build_store(
        dataset, WORK / experiment / f"{short_name}.zarr", downloads=inputs, workers=workers
    )


def label_agreement(
    first: Sequence[Any], second: Sequence[Any], *, order: Sequence[str] | None = None
) -> dict[str, Any]:
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    a = np.asarray(first).astype(str)
    b = np.asarray(second).astype(str)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} and {b.shape}")
    labels = list(order) if order is not None else sorted(set(a) | set(b))
    table = pd.crosstab(pd.Categorical(a, categories=labels), pd.Categorical(b, categories=labels))
    return {
        "n": int(a.size),
        "fractionIdentical": float(np.mean(a == b)),
        "cohenKappa": float(cohen_kappa_score(a, b, labels=labels)),
        "labels": labels,
        "contingencyRowsFirstColumnsSecond": table.to_numpy().tolist(),
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "median": float(np.median(array)), "max": float(array.max())}
