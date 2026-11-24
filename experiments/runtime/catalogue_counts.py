"""Count inputs of CyteArc's documentation datasets for the method checks.

Each check downloads a dataset's source count files from the catalogue, checks each against the byte count and
SHA-256 pinned below, and converts them into a fresh CyteArc store, on which it runs its own analysis
(experiments.runtime.rna_recipe). No prebuilt documentation store, and no analysis stored in one, is read.
"""

import hashlib
import shutil
from pathlib import Path
from typing import Any

CATALOGUE = "cytearc_docs"
# Each dataset's source count files in the catalogue, with the byte count and SHA-256 each had on 2026-11-11.
COUNT_FILES: dict[str, dict[str, tuple[int, str]]] = {
    "bastidas-ponce_4K_pancreas-d15_rnaseq": {
        "data.h5ad": (52477898, "9e3e459eca00ba06b496ec80def32941b5b2889918720e3e7aa6ffb811fbe7c6"),
    },
    "tenx_5K_pbmc_rnaseq": {
        "data.h5": (18098007, "17a47c2d3b323858ae9183e332fa31a01bf130eaa7ca6199ca9d4efe5217374f"),
    },
    "kang_15K_pbmc_rnaseq": {
        "barcodes.tsv.gz": (
            52875,
            "a4711c121443d9cb37e33e6ac57449f539ff46e0a50e4a34695861cbba666c54",
        ),
        "features.tsv.gz": (
            276960,
            "86f9914714bc554b3b8149681986521b3881c22a934a99e1b45401c3dd08aef4",
        ),
        "matrix.mtx.gz": (
            28537003,
            "8c5d623aa57bbf7e1187a643524792003cc103375ad3449e603cf6e9ccfff61d",
        ),
    },
    "kang_14K_ifnb-pbmc_rnaseq": {
        "barcodes.tsv.gz": (
            52332,
            "3864ea5bf7fa6ff8c88a93bf316bd5bea43bafc4b827b63fe64f1a5c691abe5e",
        ),
        "features.tsv.gz": (
            276960,
            "86f9914714bc554b3b8149681986521b3881c22a934a99e1b45401c3dd08aef4",
        ),
        "matrix.mtx.gz": (
            29043500,
            "a695c2cdab408ce14cb8922d42f7f2d2e28fbaa7a7cd2b29354f94af09334bd4",
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def check_file(path: Path, size: int, sha256: str) -> None:
    """Raise unless the file has the pinned byte count and SHA-256."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    found = path.stat().st_size
    if found != size:
        raise ValueError(f"{path} has {found} bytes, not the pinned {size}")
    digest = sha256_file(path)
    if digest != sha256:
        raise ValueError(f"{path} has SHA-256 {digest}, not the pinned {sha256}")


def count_files(dataset: str, downloads: Path) -> Path:
    """The folder downloads/<dataset> holding the dataset's pinned count files, downloaded from the catalogue
    where missing; raises if a file differs from its pin."""
    if dataset not in COUNT_FILES:
        raise KeyError(
            f"{dataset!r} has no pinned count files; known datasets: {sorted(COUNT_FILES)}"
        )
    folder = Path(downloads) / dataset
    repository = None
    for name, (size, sha256) in COUNT_FILES[dataset].items():
        path = folder / name
        if not path.exists():
            if repository is None:
                from cytearc.datasets import connect

                repository = connect(CATALOGUE)
            repository.download(f"{dataset}/{name}", destination=Path(downloads))
        check_file(path, size, sha256)
    return folder


def input_record(dataset: str) -> dict[str, Any]:
    """The record of a dataset's count input: the catalogue, the dataset and each file's bytes and SHA-256."""
    return {
        "catalogue": CATALOGUE,
        "dataset": dataset,
        "files": {
            name: {"bytes": size, "sha256": sha256}
            for name, (size, sha256) in COUNT_FILES[dataset].items()
        },
    }


def build_store(dataset: str, store: Path, *, downloads: Path, workers: int = 4) -> Path:
    """Convert the dataset's pinned count files into a fresh CyteArc store at store and prepare its RNA assay."""
    import cytearc

    folder = count_files(dataset, downloads)
    store = Path(store)
    if store.exists():
        shutil.rmtree(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    files = COUNT_FILES[dataset]
    if "data.h5ad" in files:
        reader = cytearc.H5adReader.from_inspect(cytearc.inspect_h5ad(str(folder / "data.h5ad")))
        try:
            cytearc.H5adToZarr(reader, zarr_loc=str(store), nthreads=workers).dump()
        finally:
            reader.close()
    elif "data.h5" in files:
        reader = cytearc.CrH5Reader(str(folder / "data.h5"))
        cytearc.CrToZarr(reader, zarr_loc=str(store), nthreads=workers).dump()
    else:
        reader = cytearc.CrDirReader(str(folder))
        cytearc.CrToZarr(reader, zarr_loc=str(store), nthreads=workers).dump()
    # CyteArc prepares a converted store's assay the first time it opens the store writable.
    cytearc.DataStore(str(store), default_assay="RNA", zarr_mode="r+", nthreads=workers)
    return store
