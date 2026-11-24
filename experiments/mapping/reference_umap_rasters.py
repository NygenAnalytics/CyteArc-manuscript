"""Bin the reference UMAPs that the artifact-reuse figure draws into square grids of cell counts per cell type, and
write each into records/ with the counts its panel is checked against.

- ladder-reference keeps the cells of the reference ladder's reference studies whose harmonized type is one of the
  ladder's types, from the cells.npz that reuse.blood_cohort_analysis saved of CyteArc's Harmony analysis of the
  ten-study blood composite, and bins their UMAP by type (records/ladder_reference_umap_raster.npz), with the
  reference studies and the cells per type and per study (records/ladder_reference_umap_raster.json).
- held-out collects the UMAP that cross_study_mapping's reference_umap job saved of one held-out tissue's reference,
  checked against the sha256 in the job's result, and bins it by harmonized label
  (records/held_out_reference_umap/kidney.npz or lung.npz), with the cells, the job's checks against the benchmark's
  reference and the cells per label (kidney.json or lung.json beside it).

Rows of each grid are UMAP 2 and columns UMAP 1, lowest first.

    uv run --locked python -m experiments.mapping.reference_umap_rasters ladder-reference CELLS_NPZ
    uv run --locked python -m experiments.mapping.reference_umap_rasters held-out RESULT [--work FOLDER]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.mapping import HERE
from experiments.runtime.common import ROOT, load_json, write_json_at
from experiments.runtime.locations import results_folder

RECORDS = HERE / "records"
LADDER = RECORDS / "reference_ladder.json"
BLOOD_COHORT = ROOT / "experiments" / "reuse" / "records" / "blood_cohort_analysis.json"
LADDER_REFERENCE = RECORDS / "ladder_reference_umap_raster"
HELD_OUT = RECORDS / "held_out_reference_umap"
TISSUES = ("kidney", "lung")
# The ladder grid is as wide as the composite figure's grid of the same embedding; a held-out grid as the graph maps.
LADDER_BINS, HELD_OUT_BINS = 420, 300
# Each square grid's side is the UMAP's longer range with this share added on each side.
MARGIN = 0.03


def square_grid_index(umap: np.ndarray, bins: int) -> np.ndarray:
    """Each cell's row (UMAP 2) and column (UMAP 1) on a square grid of bins a side around the UMAP: its side is the
    longer range plus MARGIN on each side, centred on the UMAP's range."""
    low, high = umap.min(axis=0), umap.max(axis=0)
    centre, span = (low + high) / 2, (high - low).max() * (1 + 2 * MARGIN)
    origin = centre - span / 2
    index = np.clip(((umap - origin) / span * bins).astype(np.int64), 0, bins - 1)
    return np.stack([index[:, 1], index[:, 0]], axis=1)


def binned_types(umap: np.ndarray, codes: np.ndarray, types: int, bins: int) -> np.ndarray:
    """Cells of each type in each bin, as types by rows by columns."""
    umap = np.asarray(umap, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    if umap.ndim != 2 or umap.shape[1] != 2 or codes.shape != (len(umap),):
        raise ValueError(
            "the UMAP must hold two coordinates per cell and the type codes one value per cell"
        )
    if not np.isfinite(umap).all():
        raise ValueError("the UMAP holds non-finite coordinates")
    if codes.size and (codes.min() < 0 or codes.max() >= types):
        raise ValueError(f"type codes run from {codes.min()} to {codes.max()} for {types} types")
    rows, columns = square_grid_index(umap, bins).T
    flat = (codes * bins + rows) * bins + columns
    return np.bincount(flat, minlength=types * bins * bins).reshape(types, bins, bins)


def ladder_reference(cells: dict, ladder: dict, cells_per_study: dict) -> tuple[dict, dict]:
    """The raster and record of the ladder reference: the reference studies' cells of the ladder's types, binned by
    type. cells holds the cohort analysis's per-cell arrays, checked against its record's cells per study."""
    studies, vocabulary = ladder["design"]["referenceStudies"], ladder["vocabulary"]
    names = [str(name) for name in cells["dataset_names"]]
    type_names = [str(name) for name in cells["harmonized_cell_type_names"]]
    codes = np.asarray(cells["dataset_codes"], dtype=np.int64)
    types = np.asarray(cells["harmonized_cell_type_codes"], dtype=np.int64)
    if {name: int((codes == i).sum()) for i, name in enumerate(names)} != cells_per_study:
        raise ValueError("the cells per study differ from the cohort analysis record")
    if set(studies) - set(names) or set(vocabulary) - set(type_names):
        raise ValueError("the composite lacks a reference study or a cell type of the ladder")
    keep = np.isin(codes, [names.index(s) for s in studies]) & np.isin(
        types, [type_names.index(t) for t in vocabulary]
    )
    lookup = np.full(len(type_names), -1, dtype=np.int64)
    for i, name in enumerate(vocabulary):
        lookup[type_names.index(name)] = i
    counts = binned_types(
        np.asarray(cells["umap"])[keep], lookup[types[keep]], len(vocabulary), LADDER_BINS
    )
    kept_codes = codes[keep]
    record = {
        "referenceStudies": studies,
        "cellsPerType": {name: int(counts[i].sum()) for i, name in enumerate(vocabulary)},
        "cellsPerStudy": {s: int((kept_codes == names.index(s)).sum()) for s in studies},
    }
    return {"typeCells": counts.astype(np.uint32), "types": np.asarray(vocabulary)}, record


def held_out(result: dict, cells: dict) -> tuple[dict, dict]:
    """The raster and record of one held-out reference: its UMAP binned by harmonized label, with the cells and the
    job's checks that the rebuild matched the benchmark's reference."""
    if not all(result["matchesBenchmark"].values()):
        raise ValueError("the job's rebuild did not match the benchmark's reference")
    names = [str(name) for name in cells["label_names"]]
    umap = np.asarray(cells["umap"], dtype=np.float64)
    if len(umap) != result["cells"]:
        raise ValueError(f"the UMAP holds {len(umap)} cells, the job {result['cells']}")
    counts = binned_types(umap, cells["label_codes"], len(names), HELD_OUT_BINS)
    record = {
        "cells": int(result["cells"]),
        # The job stops unless the rebuild ran the benchmark's CyteArc version and configuration on the same cells.
        "matchesBenchmark": {
            check: bool(value) for check, value in result["matchesBenchmark"].items()
        },
        "cellsPerType": {name: int(counts[i].sum()) for i, name in enumerate(names)},
    }
    return {"typeCells": counts.astype(np.uint32), "types": np.asarray(names)}, record


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_umap(result: dict, work: Path) -> Path:
    """The UMAP the job saved, downloaded into the work folder unless it is there, checked against its sha256."""
    path = work / "reference_umap.npz"
    if not path.exists():
        from experiments.runtime.object_io import download_file

        download_file(result["umapUri"], path)
    if sha256(path) != result["umapSha256"]:
        raise ValueError(f"{path} differs from the UMAP the job saved")
    return path


def read_result(source: str) -> dict:
    if source.startswith("s3://"):
        from experiments.runtime.object_io import get_json

        return get_json(source)
    return json.loads(Path(source).read_text())


def write(stem: Path, arrays: dict, record: dict) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    print(f"wrote {stem.with_suffix('.npz').relative_to(ROOT)}", flush=True)
    write_json_at(stem.with_suffix(".json"), record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    ladder = commands.add_parser("ladder-reference")
    ladder.add_argument(
        "cells", type=Path, help="the cells.npz that reuse.blood_cohort_analysis saved"
    )
    held = commands.add_parser("held-out")
    held.add_argument("result", help="s3:// URI or local path of the reference_umap job's result")
    held.add_argument(
        "--work", type=Path, help="folder for the downloaded UMAP; by default in the results"
    )
    arguments = parser.parse_args()
    if arguments.command == "ladder-reference":
        with np.load(arguments.cells, allow_pickle=False) as cells:
            arrays, record = ladder_reference(
                dict(cells),
                load_json(LADDER),
                load_json(BLOOD_COHORT)["result"]["sourceCellCounts"],
            )
        write(LADDER_REFERENCE, arrays, record)
        return
    result = read_result(arguments.result)
    tissue = result["settings"]["tissue"]
    if tissue not in TISSUES:
        raise ValueError(f"the job built the {tissue} reference, not one of {TISSUES}")
    work = arguments.work or results_folder("mapping") / "held_out_reference_umap" / tissue
    with np.load(fetch_umap(result, work), allow_pickle=False) as cells:
        arrays, record = held_out(result, dict(cells))
    write(HELD_OUT / tissue, arrays, record)


if __name__ == "__main__":
    main()
