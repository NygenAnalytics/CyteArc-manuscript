"""Accuracy and macro F1 of transferred labels from confusion matrices, with donor-bootstrap
intervals. Rows of each matrix are author labels and columns transferred labels."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 4444


def classification_statistics(matrix, labels, evaluated_labels):
    if (
        matrix.shape != (len(labels), len(labels))
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or matrix.sum() <= 0
    ):
        raise ValueError(
            "Confusion matrices must contain finite nonnegative counts and at least one cell"
        )
    diagonal = np.diag(matrix)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    denominator = support + predicted
    f1 = np.divide(
        2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator > 0
    )
    selected = [i for i, label in enumerate(labels) if label in evaluated_labels]
    return {
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "macroF1": float(f1[selected].mean()),
        "perType": {
            labels[i]: {
                "support": int(support[i]),
                "precision": float(diagonal[i] / predicted[i]) if predicted[i] else 0.0,
                "recall": float(diagonal[i] / support[i]) if support[i] else 0.0,
                "f1": float(f1[i]),
            }
            for i in selected
        },
    }


def donor_confusion_matrices(cells: pd.DataFrame, label_order: list[str]) -> list[dict]:
    """One confusion matrix per donor, in sorted donor order, from columns donor_id, truth and
    prediction."""
    return [
        {
            "donor": donor,
            "cells": len(values),
            "confusion": confusion_matrix(
                values.truth, values.prediction, labels=label_order
            ).tolist(),
        }
        for donor, values in cells.groupby("donor_id", sort=True)
    ]


def donor_bootstrap(matrices: np.ndarray, labels: list[str], evaluated_labels) -> dict:
    """95% intervals of accuracy and macro F1 over donors resampled with replacement, keeping
    each donor's cells together."""
    if len(matrices) < 3:
        return {
            "estimable": False,
            "donors": len(matrices),
            "reason": "At least three independent donors are required",
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = matrices[rng.integers(0, len(matrices), len(matrices))].sum(axis=0)
        statistics = classification_statistics(sample, labels, evaluated_labels)
        values.append([statistics["accuracy"], statistics["macroF1"]])
    interval = np.quantile(np.asarray(values), [0.025, 0.975], axis=0)
    return {
        "estimable": True,
        "accuracy": interval[:, 0].tolist(),
        "macroF1": interval[:, 1].tolist(),
        "donors": len(matrices),
        "resamples": BOOTSTRAP_RESAMPLES,
    }
