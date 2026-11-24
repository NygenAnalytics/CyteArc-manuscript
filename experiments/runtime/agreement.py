"""Compare CyteArc's numbers with a reference implementation's, and name the reference's
version."""

from importlib import metadata
from typing import Any

import numpy as np


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def numeric_agreement(first: Any, second: Any) -> dict[str, float | int]:
    from scipy.stats import spearmanr

    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} and {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    difference = np.abs(a - b)
    constant = np.ptp(a) == 0 or np.ptp(b) == 0
    return {
        "n": int(a.size),
        "pearson": float("nan") if constant else float(np.corrcoef(a, b)[0, 1]),
        "spearman": float("nan") if constant else float(spearmanr(a, b)[0]),
        "maxAbsDiff": float(difference.max()) if a.size else float("nan"),
        "meanAbsDiff": float(difference.mean()) if a.size else float("nan"),
        "rangeFirst": float(np.ptp(a)) if a.size else float("nan"),
        "rangeSecond": float(np.ptp(b)) if a.size else float("nan"),
    }
