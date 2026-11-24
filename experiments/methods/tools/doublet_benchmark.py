"""Read the labelled matrices of the doublet benchmark of Xi and Li (2021).

Each <name>.rds holds a list of a genes-by-cells dgCMatrix and the ground-truth labels. The reader
uses rdata, without R, and names cells and genes as doublets.R does (the matrix's names, or cell1,
cell2, ... and gene1, gene2, ..., made unique), so that every method's scores join by cell.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp


def make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    taken = set(names)
    out = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
            continue
        while True:
            seen[name] += 1
            candidate = f"{name}.{seen[name]}"
            if candidate not in taken:
                taken.add(candidate)
                out.append(candidate)
                break
    return out


def read_dataset(path: Path) -> tuple[sp.csr_matrix, list[str], list[str], np.ndarray]:
    import rdata

    converted = rdata.conversion.convert(rdata.parser.parse_file(str(path)))
    matrix, labels = (converted[0], converted[1])
    n_genes, n_cells = (int(v) for v in matrix.Dim)
    counts = sp.csc_matrix(
        (np.asarray(matrix.x, dtype=np.float32), np.asarray(matrix.i), np.asarray(matrix.p)),
        shape=(n_genes, n_cells),
    )
    dimnames = list(matrix.Dimnames) if matrix.Dimnames is not None else [None, None]
    genes = (
        [str(v) for v in dimnames[0]]
        if dimnames[0] is not None
        else [f"gene{i}" for i in range(1, n_genes + 1)]
    )
    cells = (
        [str(v) for v in dimnames[1]]
        if dimnames[1] is not None
        else [f"cell{i}" for i in range(1, n_cells + 1)]
    )
    labels = np.asarray([str(v) for v in np.asarray(labels).ravel()])
    if labels.size != n_cells:
        raise ValueError(f"{path.name}: {labels.size} labels for {n_cells} cells")
    return (counts.T.tocsr(), make_unique(cells), make_unique(genes), labels)
