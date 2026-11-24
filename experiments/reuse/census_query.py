"""Query one dataset of CELLxGENE Census in place, as the remote CyteBase query of composite_and_remote reads Tabula
Sapiens: open the Census, then read the cell types of the dataset's cells and one gene's counts in the cells of one
type, recording the seconds and the bytes received for each part. Census is the public TileDB-SOMA store of the
CELLxGENE datasets on Amazon S3; it keeps no UMAP of a dataset, so the query reads no embedding.

  datasets  the Census datasets whose title or collection holds a text, with their cells, to find a dataset's ID
  run       the timed query, recorded
"""

import time
from pathlib import Path

import numpy as np

NET_DEV = Path("/proc/net/dev")


def received_bytes(text: str | None = None) -> int:
    """Bytes the container has received over its network interfaces, loopback left out, from /proc/net/dev or its
    text."""
    text = NET_DEV.read_text() if text is None else text
    total, interfaces = 0, 0
    for line in text.splitlines()[2:]:
        name, _, fields = line.partition(":")
        if not fields.strip() or name.strip() == "lo":
            continue
        total += int(fields.split()[0])
        interfaces += 1
    if not interfaces:
        raise RuntimeError("/proc/net/dev lists no network interface besides loopback")
    return total


def gene_values(
    joinids: np.ndarray, totals: np.ndarray, entries: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Each chosen cell's log1p of the gene's counts per 10,000 of its total counts. ``joinids`` are the chosen cells,
    sorted, with their ``totals``; ``entries`` and ``counts`` are the sparse matrix's stored cells and values for the
    gene, and a chosen cell without an entry counts zero."""
    if np.any(np.diff(joinids) <= 0):
        raise ValueError("the chosen cells must be sorted and unique")
    place = np.searchsorted(joinids, entries)
    if np.any(place >= len(joinids)) or np.any(
        joinids[np.minimum(place, len(joinids) - 1)] != entries
    ):
        raise ValueError("the query returned counts for cells it was not asked for")
    gene = np.zeros(len(joinids))
    gene[place] = counts
    return np.log1p(gene / totals * 1e4)


def sparse_entries(table) -> tuple[np.ndarray, np.ndarray]:
    """The cells and values of a SOMA sparse matrix read as an Arrow table, as NumPy arrays."""
    entries = np.asarray(table["soma_dim_0"].to_numpy(), dtype=np.int64)
    values = np.asarray(table["soma_data"].to_numpy(), dtype=float)
    return entries, values


def datasets(payload: dict) -> list[dict]:
    import cellxgene_census

    text = payload["title"]
    with cellxgene_census.open_soma(census_version=payload["censusVersion"]) as census:
        table = census["census_info"]["datasets"].read().concat().to_pandas()
    found = table["dataset_title"].str.contains(text, case=False) | table[
        "collection_name"
    ].str.contains(text, case=False)
    columns = ["dataset_id", "dataset_title", "collection_name", "dataset_total_cell_count"]
    return table.loc[found, columns].to_dict("records")


def run(payload: dict) -> dict:
    """Open the Census, then read the dataset's cell types and the gene's counts in the cells of the cell type."""
    import cellxgene_census
    import tiledbsoma

    version, dataset_id = payload["censusVersion"], payload["datasetId"]
    cell_type, gene, expected = payload["cellType"], payload["gene"], int(payload["cells"])
    before, started = received_bytes(), time.perf_counter()
    census = cellxgene_census.open_soma(census_version=version)
    try:
        human = census["census_data"]["homo_sapiens"]
        open_seconds, open_bytes = time.perf_counter() - started, received_bytes() - before
        before, started = received_bytes(), time.perf_counter()
        obs = (
            human.obs.read(
                value_filter=f"dataset_id == '{dataset_id}'",
                column_names=["soma_joinid", "cell_type", "raw_sum"],
            )
            .concat()
            .to_pandas()
        )
        chosen = obs[obs["cell_type"] == cell_type].sort_values("soma_joinid")
        joinids = chosen["soma_joinid"].to_numpy()
        with human.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(coords=(joinids,)),
            var_query=tiledbsoma.AxisQuery(value_filter=f"feature_name == '{gene}'"),
        ) as query:
            table = query.X("raw").tables().concat()
        query_seconds, query_bytes = time.perf_counter() - started, received_bytes() - before
    finally:
        census.close()
    if len(obs) != expected:
        raise ValueError(
            f"Census holds {len(obs)} cells of {dataset_id}, not the {expected} expected"
        )
    if chosen.empty:
        raise ValueError(f"{dataset_id} has no cells of type {cell_type!r}")
    entries, counts = sparse_entries(table)
    values = gene_values(joinids, chosen["raw_sum"].to_numpy(dtype=float), entries, counts)
    return {
        "censusVersion": version,
        "datasetId": dataset_id,
        "totalCells": int(len(obs)),
        "cellType": cell_type,
        "cellsSelected": int(len(joinids)),
        "gene": gene,
        "geneEntries": int(len(table)),
        "geneMeanInSelected": float(values.mean()),
        "openSeconds": open_seconds,
        "openReceivedBytes": int(open_bytes),
        "querySeconds": query_seconds,
        "queryReceivedBytes": int(query_bytes),
        "packages": {
            "cellxgene_census": cellxgene_census.__version__,
            "tiledbsoma": tiledbsoma.__version__,
        },
    }
