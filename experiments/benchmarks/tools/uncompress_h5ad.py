"""Write an uncompressed copy of an H5AD file, streaming large arrays in slices.

Usage: python uncompress_h5ad.py SOURCE DESTINATION

The benchmark samples are gzip-compressed in chunks of 4 MiB, a layout on which BPCells'
AnnData reader is slow: one pass over a 10,000-cell sample took 335 s compressed and 0.6 s
uncompressed (experiments.benchmarks.workflows.diagnose_read_job). The R arms therefore read an
uncompressed copy, made inside the workflow and timed with it. Arrays are copied in
slices, so the copy needs little memory whatever the size of the matrix.
"""

import sys
from pathlib import Path

import h5py

SLICE = 64 * 1024 * 1024


def compressed(source: str) -> bool:
    with h5py.File(source, "r") as h5:
        found = []
        h5.visititems(
            lambda name, obj: (
                found.append(name) if isinstance(obj, h5py.Dataset) and obj.compression else None
            )
        )
        return bool(found)


def copy(source: str, destination: str) -> None:
    if not compressed(source):
        Path(destination).unlink(missing_ok=True)
        Path(destination).symlink_to(Path(source).resolve())
        return
    with h5py.File(source, "r") as a, h5py.File(destination, "w") as b:
        for key, value in a.attrs.items():
            b.attrs[key] = value

        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                if obj.ndim != 1 or obj.shape[0] <= SLICE or obj.dtype.kind in "OSU":
                    out = b.create_dataset(name, data=obj[()])
                else:
                    out = b.create_dataset(name, shape=obj.shape, dtype=obj.dtype)
                    for start in range(0, obj.shape[0], SLICE):
                        end = min(start + SLICE, obj.shape[0])
                        out[start:end] = obj[start:end]
                for key, value in obj.attrs.items():
                    out.attrs[key] = value
            else:
                group = b.require_group(name)
                for key, value in obj.attrs.items():
                    group.attrs[key] = value

        a.visititems(visit)


if __name__ == "__main__":
    copy(sys.argv[1], sys.argv[2])
