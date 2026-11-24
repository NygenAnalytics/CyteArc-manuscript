"""Bin the UMAPs that the zero-copy analysis figure draws into grids of cell counts in records/.

- tabula-sapiens-query repeats the remote Tabula Sapiens query of composite_and_remote.py: it opens the published
  store over HTTP and reads its UMAP, its cell types, the active selection and MS4A1 in the active cells, checks the
  cells, B cells and MS4A1 mean against replicate 1 of the remote-query records, and bins every cell, the B cells and
  their MS4A1 values (records/tabula_sapiens_query_raster.npz).
- blood-composite bins the UMAP of CyteArc's Harmony analysis of the ten-study blood composite by study, from the
  cells.npz that blood_cohort_analysis saved, and checks the cells per study against
  records/blood_cohort_analysis.json (records/blood_composite_umap_raster.npz).

Rows of each grid are UMAP 2 and columns UMAP 1, lowest first.

    uv run --locked python -m experiments.reuse.umap_rasters tabula-sapiens-query
    uv run --locked python -m experiments.reuse.umap_rasters blood-composite CELLS_NPZ
"""

import argparse
from pathlib import Path

import numpy as np

from experiments.reuse import HERE
from experiments.runtime.common import ROOT, load_json

RECORDS = HERE / "records"
REMOTE_QUERY = RECORDS / "composite_and_remote" / "remote_queries" / "replicate_1.json"
BLOOD_COHORT = RECORDS / "blood_cohort_analysis.json"
TABULA_SAPIENS_RASTER = RECORDS / "tabula_sapiens_query_raster.npz"
BLOOD_COMPOSITE_RASTER = RECORDS / "blood_composite_umap_raster.npz"
TABULA_SAPIENS_STORE = "hf://buckets/Nygen/cytebase/cytearc_docs/tabula_sapiens/data.zarr"
# The Tabula Sapiens grid spans the UMAP with this share of its range added on each side.
TABULA_SAPIENS_BINS, TABULA_SAPIENS_PAD = 520, 0.02
# The composite grid is square: its side is the UMAP's longer range with this share added on each side.
COMPOSITE_BINS, COMPOSITE_MARGIN = 420, 0.03


def tabula_sapiens_raster(umap: np.ndarray, b_cells: np.ndarray, ms4a1: np.ndarray) -> dict:
    """Cells, B cells and the sum of the B cells' MS4A1 values in each bin of a grid of TABULA_SAPIENS_BINS a side
    over the UMAP's range on each axis, padded by TABULA_SAPIENS_PAD. b_cells marks the B cells of all cells; ms4a1
    holds the B cells' values in their order, a missing value counting as zero."""
    umap = np.asarray(umap, dtype=float)
    b_cells = np.asarray(b_cells, dtype=bool)
    ms4a1 = np.asarray(ms4a1, dtype=float)
    if umap.ndim != 2 or umap.shape[1] != 2 or b_cells.shape != (len(umap),):
        raise ValueError(
            "the UMAP must hold two coordinates per cell and the B-cell mask one value per cell"
        )
    if ms4a1.shape != (int(b_cells.sum()),):
        raise ValueError(f"{ms4a1.size} MS4A1 values for {int(b_cells.sum())} B cells")
    if not np.isfinite(umap).all():
        raise ValueError(
            f"{int((~np.isfinite(umap).all(axis=1)).sum())} cells have no UMAP coordinates"
        )
    low, high = umap.min(axis=0), umap.max(axis=0)
    pad = TABULA_SAPIENS_PAD * (high - low)
    edges = [
        np.linspace(low[i] - pad[i], high[i] + pad[i], TABULA_SAPIENS_BINS + 1) for i in range(2)
    ]
    bins = [edges[1], edges[0]]
    cells, _, _ = np.histogram2d(umap[:, 1], umap[:, 0], bins=bins)
    b_umap = umap[b_cells]
    b_counts, _, _ = np.histogram2d(b_umap[:, 1], b_umap[:, 0], bins=bins)
    b_sum, _, _ = np.histogram2d(
        b_umap[:, 1], b_umap[:, 0], bins=bins, weights=np.nan_to_num(ms4a1)
    )
    return {
        "cells": cells.astype(np.int32),
        "bCells": b_counts.astype(np.int32),
        "bMs4a1Sum": b_sum.astype(np.float64),
    }


def query_tabula_sapiens() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The remote query: the UMAP of every cell, the B-cell mask and MS4A1 in the B cells, checked against the
    remote-query record."""
    import cytearc

    store = cytearc.DataStore(
        TABULA_SAPIENS_STORE,
        zarr_mode="r",
        storage_options={"token": False, "block_size": 1},
        min_features_per_cell=-1,
    )
    reference = store.artifacts.find("embedding", imported="X_umap", from_assay="RNA")
    umap = np.asarray(store.artifacts.load(reference)["values"][:], dtype=float)
    cell_types = np.asarray(store.cells.fetch_all("cell_type"))
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    b_cells = cell_types == "B cell"
    if np.any(b_cells & ~active):
        raise ValueError("the active selection leaves out cells annotated B cell")
    values = np.asarray(store.get_cell_vals(from_assay="RNA", cell_key="I", k="MS4A1"), dtype=float)
    ms4a1 = values[b_cells[active]]
    check_tabula_sapiens_query(int(store.cells.N), umap, b_cells, ms4a1)
    return umap, b_cells, ms4a1


def check_tabula_sapiens_query(
    cells: int, umap: np.ndarray, b_cells: np.ndarray, ms4a1: np.ndarray
) -> None:
    record = load_json(REMOTE_QUERY)["result"]["remoteQueries"]["tabulaSapiens1_1M"]
    found = {
        "totalCells": (cells, record["totalCells"]),
        "bCellsSelected": (int(b_cells.sum()), record["bCellsSelected"]),
        "umapShape": (list(umap.shape), record["umapShape"]),
    }
    for name, (value, recorded) in found.items():
        if value != recorded:
            raise ValueError(f"{name}: the query found {value}, the record holds {recorded}")
    mean = float(np.nanmean(ms4a1))
    if not np.isclose(mean, record["ms4a1MeanInBCells"], rtol=1e-9):
        raise ValueError(
            f"MS4A1 mean in B cells {mean} differs from the record's {record['ms4a1MeanInBCells']}"
        )


def square_grid_index(umap: np.ndarray, bins: int, margin: float) -> np.ndarray:
    """Each cell's row (UMAP 2) and column (UMAP 1) on a square grid of bins a side around the UMAP: its side is the
    longer range plus margin on each side, centred on the UMAP's range."""
    low, high = umap.min(axis=0), umap.max(axis=0)
    centre, span = (low + high) / 2, (high - low).max() * (1 + 2 * margin)
    origin = centre - span / 2
    index = np.clip(((umap - origin) / span * bins).astype(np.int64), 0, bins - 1)
    return np.stack([index[:, 1], index[:, 0]], axis=1)


def blood_composite_raster(
    umap: np.ndarray, codes: np.ndarray, studies: list[str], cells_per_study: dict
) -> dict:
    """Cells of each study in each bin of a square grid of COMPOSITE_BINS a side, as studies by rows by columns,
    checked against the cohort analysis's cells per study."""
    umap = np.asarray(umap, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    if umap.ndim != 2 or umap.shape[1] != 2 or codes.shape != (len(umap),):
        raise ValueError(
            "the UMAP must hold two coordinates per cell and the study codes one value per cell"
        )
    if not np.isfinite(umap).all():
        raise ValueError("the UMAP holds non-finite coordinates")
    if codes.min() < 0 or codes.max() >= len(studies):
        raise ValueError(
            f"study codes run from {codes.min()} to {codes.max()} for {len(studies)} studies"
        )
    rows, columns = square_grid_index(umap, COMPOSITE_BINS, COMPOSITE_MARGIN).T
    flat = (codes * COMPOSITE_BINS + rows) * COMPOSITE_BINS + columns
    counts = np.bincount(flat, minlength=len(studies) * COMPOSITE_BINS**2).reshape(
        len(studies), COMPOSITE_BINS, COMPOSITE_BINS
    )
    per_study = {study: int(counts[i].sum()) for i, study in enumerate(studies)}
    if per_study != cells_per_study:
        raise ValueError("the raster's cells per study differ from the cohort analysis record")
    return {"studyCells": counts.astype(np.uint32), "studies": np.asarray(studies)}


def read_blood_composite(path: Path) -> dict:
    """The blood composite raster from the cells.npz of the cohort analysis."""
    record = load_json(BLOOD_COHORT)["result"]
    with np.load(path, allow_pickle=False) as cells:
        umap = cells["umap"]
        codes = cells["dataset_codes"]
        studies = [str(name) for name in cells["dataset_names"]]
    if len(umap) != record["cells"]:
        raise ValueError(
            f"{path.name} holds {len(umap)} cells, the cohort analysis {record['cells']}"
        )
    return blood_composite_raster(umap, codes, studies, record["sourceCellCounts"])


def write_raster(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tabula-sapiens-query")
    composite = commands.add_parser("blood-composite")
    composite.add_argument("cells", type=Path, help="the cells.npz that the cohort analysis saved")
    arguments = parser.parse_args()
    if arguments.command == "tabula-sapiens-query":
        write_raster(TABULA_SAPIENS_RASTER, tabula_sapiens_raster(*query_tabula_sapiens()))
    else:
        write_raster(BLOOD_COMPOSITE_RASTER, read_blood_composite(arguments.cells))


if __name__ == "__main__":
    main()
