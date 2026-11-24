"""Transfer of COMBAT's T-cell subtypes onto Hao's T cells by RNA alone, from COMBAT's WNN clusters and from its RNA
clusters at resolution 0.5, each cluster named by its majority author label: the share of Hao's T cells with each
compared author label that receive it, and over all of them; and the Hao T cells to which the named COMBAT reference gave
a subtype. From records/t_cell_subtypes_cell_clr/combat/ and records/t_cell_reference/combat/, into
.build/derived/integration/t_cell_reference.json.

    uv run --locked python -m experiments.integration.t_cell_reference_summary
"""

import csv
from collections import defaultdict

import numpy as np

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

# The t_cell_subtypes run on COMBAT's T cells with per-cell CLR, its WNN and RNA clusters each mapped onto Hao's T cells.
T_CELL_SUBTYPES = ROOT / "experiments/integration/records/t_cell_subtypes_cell_clr/combat"
# The named COMBAT T-cell reference job, whose record holds the subtype each Hao T cell received.
T_CELL_REFERENCE = ROOT / "experiments/integration/records/t_cell_reference/combat"
SUMMARY = DERIVED / "integration" / "t_cell_reference.json"
# One vocabulary for COMBAT's and Hao's author labels, so that a reference cluster is named by its majority author
# label and Hao's cells are scored against the same terms; labels not listed (Hao's dnT) are left out.
T_CELL_VOCABULARY = {
    "CD4 naive": ("CD4.NAIVE", "CD4 Naive"),
    "CD4 memory": ("CD4.TCM", "CD4.TEFF", "CD4 TCM", "CD4 TEM"),
    "CD4 CTL": ("CD4.TEM/TEMRA", "CD4 CTL"),
    "Treg": ("CD4.TREG", "Treg"),
    "CD8 naive": ("CD8.NAIVE", "CD8 Naive"),
    "CD8 memory": ("CD8.TCM", "CD8.TCM.CCL5", "CD8 TCM"),
    "CD8 effector": ("CD8.TEM", "CD8.TEMRA", "CD8.TEFF", "CD8 TEM"),
    "MAIT": ("MAIT",),
    "γδ T": ("GDT.VD2", "GDT.VD2neg", "gdT"),
}
# The terms both kinds of reference can name at resolution 0.5; neither cuts out CD4 CTL or CD8 memory cells there.
T_CELL_COMPARED = ("CD4 naive", "CD4 memory", "Treg", "CD8 naive", "CD8 effector", "MAIT", "γδ T")


def relative(path) -> str:
    return str(path.relative_to(ROOT))


def vocabulary_term(label: str) -> str | None:
    """The vocabulary term that lists an author label, or None for a label it does not list."""
    return next((term for term, labels in T_CELL_VOCABULARY.items() if label in labels), None)


def cluster_terms(authors: list[dict], graph: str, resolution: float) -> dict[str, str]:
    """Each cluster's term, the vocabulary term holding most of its cells' author labels; other labels count as other."""
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in authors:
        if row["graph"] == graph and float(row["resolution"]) == resolution:
            totals[(row["cluster"], vocabulary_term(row["authorLabel"]) or "other")] += int(
                row["cells"]
            )
    best: dict[str, tuple[int, str]] = {}
    for (cluster, term), count in totals.items():
        if cluster not in best or count > best[cluster][0]:
            best[cluster] = (count, term)
    return {cluster: term for cluster, (_, term) in best.items()}


def subtype_transfer() -> dict:
    """For each term both kinds of reference name, the share of Hao's T cells with that author label that receive it,
    by RNA alone, from COMBAT's WNN clusters and from its RNA clusters at resolution 0.5, each cluster named by its
    majority author label; the share over all those cells, and how many of Hao's T cells carry no compared label."""
    record = load_json(T_CELL_SUBTYPES / "result.json")["result"]
    if record.get("adtNormalization") != "cell":
        raise ValueError("the compared run did not normalize protein per cell")
    with (T_CELL_SUBTYPES / "t_cell_subtypes_authors.csv").open() as handle:
        authors = list(csv.DictReader(handle))
    with np.load(T_CELL_SUBTYPES / "t_cell_subtypes.npz", allow_pickle=False) as saved:
        hao = np.array([vocabulary_term(label) or "" for label in saved["query_labels"]])
        received = {
            graph: np.asarray(saved[f"transferred_{graph}_0.5"]).astype(str)
            for graph in ("wnn", "rna")
        }
    names = {graph: cluster_terms(authors, graph, 0.5) for graph in received}
    for graph, terms in names.items():
        if not set(T_CELL_COMPARED) <= set(terms.values()):
            raise ValueError(
                f"the {graph} clusters name {sorted(set(terms.values()))}, not each compared term"
            )
    got = {
        graph: np.array([names[graph].get(c, "unassigned") for c in received[graph]])
        for graph in received
    }
    rows = []
    for term in T_CELL_COMPARED:
        mine = hao == term
        rows.append(
            {
                "subtype": term,
                "haoCells": int(mine.sum()),
                **{graph: float(np.mean(got[graph][mine] == term)) for graph in got},
            }
        )
    scored = np.isin(hao, T_CELL_COMPARED)
    return {
        "resolution": 0.5,
        "rows": rows,
        "overall": {
            "haoCells": int(scored.sum()),
            "unscoredHaoCells": int((~scored).sum()),
            **{g: float(np.mean(got[g][scored] == hao[scored])) for g in got},
        },
        "source": [
            relative(T_CELL_SUBTYPES / "t_cell_subtypes_authors.csv"),
            relative(T_CELL_SUBTYPES / "t_cell_subtypes.npz"),
            relative(T_CELL_SUBTYPES / "result.json"),
        ],
    }


def assigned_query_cells(result: dict, query: str) -> dict:
    """The query's T cells that the named reference job mapped, those that received no subtype, and those that received
    one. ``result`` is the job record's result."""
    mapped = result["queries"][query]
    unassigned = mapped["subtypeCells"].get("unassigned", 0)
    assigned = sum(n for subtype, n in mapped["subtypeCells"].items() if subtype != "unassigned")
    if assigned + unassigned != mapped["cells"]:
        raise ValueError(
            f"the {query} query's subtypes hold {assigned + unassigned} cells, not the {mapped['cells']} it mapped"
        )
    return {"cells": mapped["cells"], "unassigned": unassigned, "assignedCells": assigned}


def main() -> None:
    record = load_json(T_CELL_REFERENCE / "result.json")
    if record.get("status") != "completed":
        raise ValueError(
            f"{relative(T_CELL_REFERENCE / 'result.json')} is {record.get('status')!r}"
        )
    write_json_at(
        SUMMARY,
        {
            "subtypeTransfer": subtype_transfer(),
            "namedReferenceQuery": {
                "source": relative(T_CELL_REFERENCE / "result.json"),
                **assigned_query_cells(record["result"], "hao"),
            },
        },
    )


if __name__ == "__main__":
    main()
