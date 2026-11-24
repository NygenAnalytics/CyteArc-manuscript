"""Adjust a family of P values for multiple testing: the Benjamini-Hochberg false discovery rate and the Holm-Bonferroni
family-wise error rate. A P value that is not finite stays NaN and is not counted in the family."""

import numpy as np


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if finite.any():
        values = p[finite]
        order = np.argsort(values)
        ranked = values[order] * values.size / np.arange(1, values.size + 1)
        adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
        result = np.empty(values.size)
        result[order] = np.minimum(adjusted, 1.0)
        out[finite] = result
    return out


def holm(p) -> np.ndarray:
    """Holm's step-down adjustment: the i-th smallest of m P values times m - i + 1, kept from falling below the one
    before it and capped at 1."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if finite.any():
        values = p[finite]
        order = np.argsort(values)
        ranked = values[order] * (values.size - np.arange(values.size))
        result = np.empty(values.size)
        result[order] = np.minimum(np.maximum.accumulate(ranked), 1.0)
        out[finite] = result
    return out
