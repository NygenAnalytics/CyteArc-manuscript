"""The mouse embryo atlas that the benchmark samples come from, and the nested sampling of its
rows that defines each sample."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_SAMPLING_SEED = 0
DEFAULT_TARGET_SIZES = (
    10000,
    25000,
    50000,
    100000,
    250000,
    500000,
    1000000,
    2500000,
    5000000,
    10000000,
)
_SPLITMIX_INCREMENT = 11400714819323198485
_SPLITMIX_MULTIPLIER_1 = 13787848793156543929
_SPLITMIX_MULTIPLIER_2 = 10723151780598845931
_EMBRYO_SAMPLING_KEY = 11789696437827111060


@dataclass(frozen=True, slots=True)
class SourceSpec:
    datasetId: str
    versionId: str
    url: str
    nRows: int
    nColumns: int
    nnz: int
    sourceBytes: int
    matrixKey: str = "X"
    matrixEncoding: str = "csr_matrix"
    cellIdsKey: str = "obs/_index"
    featureIdsKey: str = "var/_index"
    featureNameKey: str = "var/feature_name"
    dataDtype: str = "float32"
    indicesDtype: str = "int64"
    indptrDtype: str = "int64"


SOURCE_SPEC = SourceSpec(
    datasetId="dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3",
    versionId="1bc30289-9565-4099-abf9-3326328c11ac",
    url="https://datasets.cellxgene.cziscience.com/1bc30289-9565-4099-abf9-3326328c11ac.h5ad",
    nRows=11441407,
    nColumns=45525,
    nnz=19516755155,
    sourceBytes=46292192475,
)


def sampling_salt(seed: int, sourceVersion: str = SOURCE_SPEC.versionId) -> int:
    # The canonical embryo sample uses this fixed SplitMix64 key; seeds offset it.
    source_offset = (
        0
        if sourceVersion == SOURCE_SPEC.versionId
        else int.from_bytes(hashlib.sha256(sourceVersion.encode()).digest()[:8], "little")
    )
    return (_EMBRYO_SAMPLING_KEY + seed + source_offset) % (1 << 64)


def row_priorities(
    nRows: int, *, seed: int = DEFAULT_SAMPLING_SEED, sourceVersion: str = SOURCE_SPEC.versionId
) -> np.ndarray:
    if nRows < 0:
        raise ValueError("nRows must be nonnegative")
    values = np.arange(nRows, dtype=np.uint64)
    values += np.uint64(sampling_salt(seed, sourceVersion))
    values += np.uint64(_SPLITMIX_INCREMENT)
    values = (values ^ values >> np.uint64(30)) * np.uint64(_SPLITMIX_MULTIPLIER_1)
    values = (values ^ values >> np.uint64(27)) * np.uint64(_SPLITMIX_MULTIPLIER_2)
    return values ^ values >> np.uint64(31)


def _normalize_targets(targetRows: Sequence[int], nRows: int) -> tuple[int, ...]:
    targets = tuple((int(value) for value in targetRows))
    if not targets:
        raise ValueError("At least one target row count is required")
    if any((value <= 0 for value in targets)):
        raise ValueError("Target row counts must be positive")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("Target row counts must be unique and increasing")
    if targets[-1] > nRows:
        raise ValueError(f"Largest target has {targets[-1]} rows, source has {nRows}")
    return targets


def select_rows_from_priorities(
    priorities: np.ndarray, targetRows: Sequence[int]
) -> dict[int, np.ndarray]:
    priorities = np.asarray(priorities)
    if priorities.ndim != 1:
        raise ValueError("priorities must be one-dimensional")
    if not np.issubdtype(priorities.dtype, np.integer):
        raise TypeError("priorities must use an integer dtype")
    targets = _normalize_targets(targetRows, len(priorities))
    source_rows = np.arange(len(priorities), dtype=np.int64)
    ranked = np.lexsort((source_rows, priorities.astype(np.uint64, copy=False)))
    return {
        target: np.sort(ranked[:target].astype(np.int64, copy=False), kind="stable")
        for target in targets
    }


def select_nested_rows(
    nRows: int,
    targetRows: Sequence[int] = DEFAULT_TARGET_SIZES,
    *,
    seed: int = DEFAULT_SAMPLING_SEED,
    sourceVersion: str = SOURCE_SPEC.versionId,
) -> dict[int, np.ndarray]:
    priorities = row_priorities(nRows, seed=seed, sourceVersion=sourceVersion)
    return select_rows_from_priorities(priorities, targetRows)
