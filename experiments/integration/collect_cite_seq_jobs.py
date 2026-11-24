"""Write the records of the CITE-seq map, weight and T-cell jobs from the run records they save.

Each job's run record, which the Modal runner saves at the payload's recordUri, names the arrays and
tables the job uploaded beside it, with their sha256. This step reads one run record, from the
experiment bucket or a local file, downloads those files into a work folder (by default the
experiment's results folder) unless they are there already, checks them against the record, writes
the record into records/ without its call ID, start time and bucket, and for each kind of job:

- graph_umaps: bins each graph's UMAP into a square grid of cell counts per author label and writes
  the grids (raster_RNA.npz, raster_ADT.npz and raster_WNN.npz) with the layouts' summary.json into
  records/graph_umaps_harmony/, or records/graph_umaps/ for the run's own uncorrected graphs, and
  counts each label's protein-led, even and RNA-led cells (modality_weights.csv) in the work folder;
- seurat_weights: summarizes each author label's protein weight in Seurat and CyteArc
  (weights_by_label.csv) and the tools' agreement (summary.json) in records/seurat_weights/;
- t_cell_reference: bins the named reference's WNN UMAP by subtype (raster.npz) and keeps the grid,
  its raster_summary.json and the subtypes' protein tables of the reference and of the Hao query in
  records/t_cell_reference/<study>/;
- t_cell_subtypes: keeps the clusterings' per-cell arrays (t_cell_subtypes.npz) and author labels
  per cluster (t_cell_subtypes_authors.csv) in records/t_cell_subtypes_cell_clr/<study>/, or
  records/t_cell_subtypes/<study>/ for CyteArc's default protein normalization.

  uv run --locked python -m experiments.integration.collect_cite_seq_jobs KIND RECORD [--study STUDY] [--work FOLDER]
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.integration import HERE
from experiments.runtime.common import public_run, write_json_at
from experiments.runtime.locations import results_folder

RECORDS = HERE / "records"
GRAPHS = ("RNA", "ADT", "WNN")
# The maps are square grids of this many bins, around the layout's extent with this margin.
BINS = 300
MARGIN = 0.03
ROWS_ARE = "UMAP 2, lowest first; columns are UMAP 1, lowest first"
TOOLS = ("cytearc", "seurat")
# The files of a T-cell job that records/ keeps; the others stay in the work folder.
T_CELL_RECORDED = {
    "t_cell_reference": ("reference_proteins", "hao_proteins"),
    "t_cell_subtypes": ("npz", "authors"),
}
KINDS = ("graph_umaps", "seurat_weights", "t_cell_reference", "t_cell_subtypes")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def completed_result(run: dict) -> dict:
    if run.get("status") != "completed":
        raise ValueError(f"the job's record says {run.get('status')!r}, not completed")
    return run["result"]


def fetch(uri: str, destination: Path, expected_sha256: str) -> Path:
    """A file the job uploaded, downloaded into the work folder unless it is there already, and
    checked against the sha256 the job recorded."""
    if not destination.exists():
        from experiments.runtime.object_io import download_file

        download_file(uri, destination)
    if sha256(destination) != expected_sha256:
        raise ValueError(f"{destination.name} differs from the file the job saved")
    return destination


def square_grid(layout: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Each cell's bin on a square grid of BINS by BINS around the layout, with MARGIN on each
    side, and the grid's origin and span."""
    low, high = layout.min(axis=0), layout.max(axis=0)
    centre, span = (low + high) / 2, (high - low).max() * (1 + 2 * MARGIN)
    origin = centre - span / 2
    index = np.clip(((layout - origin) / span * BINS).astype(np.int64), 0, BINS - 1)
    return index, origin, float(span)


def binned_counts(index: np.ndarray, codes: np.ndarray, groups: int) -> np.ndarray:
    """Cells per group in each bin, as groups by rows (UMAP 2) by columns (UMAP 1)."""
    flat = codes * BINS * BINS + index[:, 1] * BINS + index[:, 0]
    return np.bincount(flat, minlength=groups * BINS * BINS).reshape(groups, BINS, BINS)


def weight_leads(values: np.ndarray) -> dict[str, int]:
    """Cells whose neighbourhood protein leads (weight above 0.5), that split their weight evenly
    (exactly 0.5) or that RNA leads (below 0.5)."""
    return {
        "ProteinLedCells": int((values > 0.5).sum()),
        "EvenCells": int((values == 0.5).sum()),
        "RnaLedCells": int((values < 0.5).sum()),
    }


def graph_umaps(run: dict, work: Path) -> Path:
    """The summary and rasters of a graph_umaps job's layouts in records/, written once every check
    passes, with the modality weights in the work folder."""
    result = completed_result(run)
    if not result["graphsIdentical"]:
        changed = [key for key, entry in result["graphsCheck"].items() if not entry["identical"]]
        raise ValueError(f"the rebuilt graphs differ from the run's saved graphs in {changed}")
    folder = RECORDS / ("graph_umaps_harmony" if result.get("harmony") else "graph_umaps")
    path = fetch(result["umapUri"], work / "graph_umaps.npz", result["umapSha256"])
    with np.load(path, allow_pickle=False) as cells:
        layouts = {graph: cells[f"{graph}_umap"].astype(np.float64) for graph in GRAPHS}
        codes = cells["label_codes"].astype(np.int64)
        names = [str(name) for name in cells["label_names"]]
        studies = [str(name) for name in cells["study_names"]]
        study_codes = cells["study_codes"].astype(np.int64)
        weights = cells["modality_weights"].astype(np.float64)
        modalities = [str(m) for m in cells["modalities"]]
    n = result["cells"]
    if any(layout.shape != (n, 2) or not np.isfinite(layout).all() for layout in layouts.values()):
        raise ValueError("a layout has the wrong cells or non-finite coordinates")
    if (
        len(codes) != n
        or weights.shape != (n, 2)
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError(
            "the labels or modality weights do not match the cells, or the weights do not sum to one"
        )
    if modalities != ["RNA", "ADT"]:
        raise ValueError(f"the weights are of {modalities}, not RNA and ADT")
    # The job averaged the float32 weights in float32; the same average reproduces its means, and
    # the summary keeps the exact float64 means, which differ from them by about 2e-5.
    recorded = dict(zip(modalities, weights.astype(np.float32).mean(axis=0).tolist(), strict=True))
    if any(abs(recorded[m] - result["modalityWeightMean"][m]) > 1e-7 for m in modalities):
        raise ValueError("the modality weights differ from the means the job recorded")
    grids, rasters = {}, {}
    for graph, layout in layouts.items():
        index, origin, span = square_grid(layout)
        counts = binned_counts(index, codes, len(names))
        if int(counts.sum()) != n:
            raise ValueError(f"the {graph} raster holds {int(counts.sum())} cells, not {n}")
        rasters[graph] = counts.astype(np.uint32)
        grids[graph] = {"origin": [float(v) for v in origin], "span": span}
    protein = weights[:, 1]
    rows = []
    for code, name in enumerate(names):
        values = protein[codes == code]
        leads = weight_leads(values)
        rows.append(
            {
                "label": name,
                "cells": int(values.size),
                "proteinLedCells": leads["ProteinLedCells"],
                "evenCells": leads["EvenCells"],
                "rnaLedCells": leads["RnaLedCells"],
                "meanProteinWeight": float(values.mean()),
            }
        )
    with (work / "modality_weights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    harmony = result.get("harmony")
    summary = {
        "source": path.name,
        "sourceSha256": result["umapSha256"],
        "job": "result.json",
        "graphsIdentical": True,
        "laidOut": result.get("laidOut", "uncorrected"),
        "harmony": None
        if harmony is None
        else {
            key: harmony[key]
            for key in ("batchColumn", "batchCells", "correctionSeconds", "biologicalAgreement")
        },
        "cells": n,
        "bins": BINS,
        "grids": grids,
        "rowsAre": ROWS_ARE,
        "labelsNote": "the empty label marks cells that no rule of the atlas configuration labels",
        "cellsPerLabel": {name: int((codes == code).sum()) for code, name in enumerate(names)},
        "cellsPerStudy": {
            name: int((study_codes == code).sum()) for code, name in enumerate(studies)
        },
        "modalityWeightMean": dict(zip(modalities, weights.mean(axis=0).tolist(), strict=True)),
        "proteinWeightShares": {
            "one": float((protein == 1.0).mean()),
            "zero": float((protein == 0.0).mean()),
            "half": float((protein == 0.5).mean()),
            # Weights within 0.001 of 0.5 but not 0.5 itself: how much the even split depends on
            # exact equality.
            "nearHalfNotHalf": float(((np.abs(protein - 0.5) < 1e-3) & (protein != 0.5)).mean()),
        },
    }
    folder.mkdir(parents=True, exist_ok=True)
    for graph, counts in rasters.items():
        np.savez_compressed(
            folder / f"raster_{graph}.npz", typeCells=counts, types=np.asarray(names)
        )
    write_json_at(folder / "result.json", public_run(run))
    return write_json_at(folder / "summary.json", summary)


def seurat_weights(run: dict, work: Path) -> Path:
    """Each author label's protein weight in Seurat and CyteArc on the same cells, and the tools'
    agreement, from a seurat_weights job's paired per-cell weights."""
    result = completed_result(run)
    folder = RECORDS / "seurat_weights"
    path = fetch(result["pairedUri"], work / "seurat_weights.npz", result["pairedSha256"])
    with np.load(path, allow_pickle=False) as paired:
        protein = {tool: paired[f"{tool}_protein"].astype(np.float64) for tool in TOOLS}
        codes = paired["label_codes"].astype(np.int64)
        names = [str(name) for name in paired["label_names"]]
    n = result["cells"]
    if any(len(values) != n or not np.isfinite(values).all() for values in protein.values()):
        raise ValueError("the weights do not cover the job's cells")
    pearson = float(np.corrcoef(protein["cytearc"], protein["seurat"])[0, 1])
    if abs(pearson - result["proteinWeightPearson"]) > 1e-6:
        raise ValueError("the weights' correlation differs from the one the job recorded")
    rows = []
    for code, name in enumerate(names):
        chosen = codes == code
        row: dict = {"label": name, "cells": int(chosen.sum())}
        for tool in TOOLS:
            values = protein[tool][chosen]
            row |= {f"{tool}{key}": value for key, value in weight_leads(values).items()}
            row[f"{tool}MeanProteinWeight"] = float(values.mean())
        rows.append(row)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "weights_by_label.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = np.array(
        [[r["cytearcMeanProteinWeight"], r["seuratMeanProteinWeight"]] for r in rows if r["label"]]
    )
    summary = {
        "source": path.name,
        "sourceSha256": result["pairedSha256"],
        "job": "result.json",
        "cells": n,
        "proteinWeightPearson": pearson,
        "labelMeanPearson": float(np.corrcoef(means[:, 0], means[:, 1])[0, 1]),
        "proteinWeightMean": {tool: float(protein[tool].mean()) for tool in TOOLS},
        "proteinWeightShares": {
            tool: {
                "one": float((protein[tool] == 1.0).mean()),
                "zero": float((protein[tool] == 0.0).mean()),
                "half": float((protein[tool] == 0.5).mean()),
            }
            for tool in TOOLS
        },
        "seurat": {
            key: result["seurat"][key]
            for key in ("Seurat", "SeuratObject", "R", "initialWallSeconds")
        },
    }
    write_json_at(folder / "result.json", public_run(run))
    return write_json_at(folder / "summary.json", summary)


def t_cell_files(run: dict, work: Path, stem: str) -> dict[str, Path]:
    """Every file a T-cell job uploaded, in the work folder under the names the job gives them:
    <stem>.npz for its arrays and <stem>_<table>.csv for each table."""
    files = {}
    for name, entry in completed_result(run)["files"].items():
        suffix = ".npz" if name == "npz" else f"_{name}.csv"
        files[name] = fetch(entry["uri"], work / f"{stem}{suffix}", entry["sha256"])
    return files


def t_cell_reference(run: dict, work: Path, study: str) -> Path:
    """The named T-cell reference's WNN UMAP binned by subtype, with its protein tables, in records/
    once every check passes."""
    result = completed_result(run)
    folder = RECORDS / "t_cell_reference" / study
    files = t_cell_files(run, work, "t_cell_reference")
    with np.load(files["npz"], allow_pickle=False) as saved:
        layout = saved["reference_umap"].astype(np.float64)
        subtypes = np.where(
            saved["reference_subtypes"] == "", "unlabelled", saved["reference_subtypes"]
        )
    if layout.shape != (result["cells"], 2) or not np.isfinite(layout).all():
        raise ValueError("the layout has the wrong cells or non-finite coordinates")
    names, codes = np.unique(subtypes, return_inverse=True)
    index, origin, span = square_grid(layout)
    counts = binned_counts(index, codes, len(names))
    if int(counts.sum()) != result["cells"]:
        raise ValueError("the raster does not hold each cell once")
    cells = {str(n): int((codes == i).sum()) for i, n in enumerate(names)}
    named = {k: v for k, v in cells.items() if k != "unlabelled"}
    if named != result["subtypeCells"] or cells.get("unlabelled", 0) != result["unlabelledCells"]:
        raise ValueError("the raster's subtypes do not hold the cells the job named")
    summary = {
        "source": files["npz"].name,
        "sourceSha256": sha256(files["npz"]),
        "job": "result.json",
        "cells": result["cells"],
        "bins": BINS,
        "grid": {"origin": [float(v) for v in origin], "span": span},
        "rowsAre": ROWS_ARE,
        "cellsPerSubtype": cells,
    }
    keep_files(files, T_CELL_RECORDED["t_cell_reference"], folder)
    np.savez_compressed(folder / "raster.npz", typeCells=counts.astype(np.uint32), types=names)
    write_json_at(folder / "result.json", public_run(run))
    return write_json_at(folder / "raster_summary.json", summary)


def t_cell_subtypes(run: dict, work: Path, study: str) -> Path:
    """A T-cell clustering job's per-cell arrays and author labels per cluster."""
    result = completed_result(run)
    normalization = result.get("adtNormalization", "feature")
    folder = RECORDS / (
        "t_cell_subtypes_cell_clr" if normalization == "cell" else "t_cell_subtypes"
    )
    files = t_cell_files(run, work, "t_cell_subtypes")
    keep_files(files, T_CELL_RECORDED["t_cell_subtypes"], folder / study)
    return write_json_at(folder / study / "result.json", public_run(run))


def keep_files(files: dict[str, Path], names: tuple[str, ...], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(files[name], folder / files[name].name)
        print(f"wrote {(folder / files[name].name).relative_to(HERE.parents[1])}", flush=True)


def read_run(source: str) -> dict:
    if source.startswith("s3://"):
        from experiments.runtime.object_io import get_json

        return get_json(source)
    return json.loads(Path(source).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("source", help="s3:// URI or local path of the job's run record")
    parser.add_argument("--study", help="the study whose T cells a T-cell job analysed")
    parser.add_argument(
        "--work",
        type=Path,
        help="folder for the job's downloaded files and the modality weights; by default its folder "
        "in the experiment's results",
    )
    args = parser.parse_args()
    if args.kind.startswith("t_cell") and not args.study:
        parser.error(f"{args.kind} records are kept per study; give --study")
    work = args.work or results_folder("integration") / args.kind / (args.study or "")
    work.mkdir(parents=True, exist_ok=True)
    run = read_run(args.source)
    if args.kind == "graph_umaps":
        graph_umaps(run, work)
    elif args.kind == "seurat_weights":
        seurat_weights(run, work)
    elif args.kind == "t_cell_reference":
        t_cell_reference(run, work, args.study)
    else:
        t_cell_subtypes(run, work, args.study)


if __name__ == "__main__":
    main()
