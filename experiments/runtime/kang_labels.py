"""The authors' cell type labels of the Kang PBMCs for the checks that need them, from the cell table of GEO
GSE96583: for every barcode of the control and interferon-stimulated lanes, the authors' cell type and demuxlet's
singlet, doublet or ambiguous call. Only singlets carry a label here. The table is pinned by URL, byte count and
SHA-256."""

import urllib.request
from pathlib import Path
from typing import Any

from experiments.runtime.catalogue_counts import check_file

URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.total.tsne.df.tsv.gz"
FILE_NAME = "GSE96583_batch2.total.tsne.df.tsv.gz"
BYTES = 756342
SHA256 = "1d57e72e92ca8695250e88cc0f1c3fa8c0be1175d974f8b427c58f1274dc6c09"
# The lane of each documentation dataset in the table's stim column.
CONDITIONS = {"kang_15K_pbmc_rnaseq": "ctrl", "kang_14K_ifnb-pbmc_rnaseq": "stim"}
LABEL_COLUMN = "geo_cell_type"
SINGLETS = "geo_singlet"
UNLABELLED = "NA"


def cell_table(downloads: Path) -> Any:
    """The pinned GEO cell table, downloaded into downloads/GSE96583 where missing."""
    import pandas as pd

    path = Path(downloads) / "GSE96583" / FILE_NAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.rename(path)
    check_file(path, BYTES, SHA256)
    return pd.read_csv(path, sep="\t", index_col=0)


def singlet_labels(table: Any, condition: str, barcodes: Any) -> Any:
    """The authors' cell type of each barcode that demuxlet called a singlet in the lane, and UNLABELLED for every
    other barcode. The table writes some stimulated barcodes with a -11 suffix where the count files write -1."""
    import numpy as np

    lane = table[table["stim"] == condition]
    if lane.empty:
        raise ValueError(f"the GEO table has no {condition!r} lane")
    index = lane.index.astype(str).str.replace("-11$", "-1", regex=True)
    if index.has_duplicates:
        raise ValueError(f"the {condition} lane repeats barcodes once -11 is read as -1")
    lane = lane.set_axis(index)
    barcodes = np.asarray(barcodes).astype(str)
    missing = ~np.isin(barcodes, lane.index.to_numpy())
    if missing.any():
        raise ValueError(f"{int(missing.sum())} {condition} barcodes are not in the GEO table")
    calls = lane.loc[barcodes]
    # A few singlets have no cell type in the table; without one they stay unlabelled.
    labelled = (calls["multiplets"] == "singlet").to_numpy() & calls["cell"].notna().to_numpy()
    cell_types = calls["cell"].fillna(UNLABELLED).to_numpy(dtype=object).astype(str)
    return np.where(labelled, cell_types, UNLABELLED)


def label_store(ds: Any, dataset: str, downloads: Path) -> dict[str, Any]:
    """Insert the singlet labels (LABEL_COLUMN) and the mask of labelled singlets (SINGLETS) into the store's cell
    table, and return the record of the labels used."""
    import numpy as np

    if dataset not in CONDITIONS:
        raise KeyError(f"{dataset!r} is not a Kang lane; known lanes: {sorted(CONDITIONS)}")
    condition = CONDITIONS[dataset]
    barcodes = ds.cells.fetch_all("ids")
    table = cell_table(downloads)
    labels = singlet_labels(table, condition, barcodes)
    singlet = labels != UNLABELLED
    ds.cells.insert(column_name=LABEL_COLUMN, values=labels, key="I", overwrite=True)
    ds.cells.insert(column_name=SINGLETS, values=singlet, key="I", overwrite=True)
    names, counts = np.unique(labels[singlet], return_counts=True)
    lane = table[table["stim"] == condition]
    untyped = int(((lane["multiplets"] == "singlet") & lane["cell"].isna()).sum())
    return {
        "source": {"url": URL, "bytes": BYTES, "sha256": SHA256},
        "condition": condition,
        "labels": "the authors' cell type of each demuxlet singlet; other barcodes unlabelled",
        "barcodes": int(labels.size),
        "singlets": int(singlet.sum()),
        "singletsWithoutCellType": untyped,
        "cellsPerLabel": {str(name): int(count) for name, count in zip(names, counts, strict=True)},
    }
