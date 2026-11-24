"""Pull the circulating haematopoietic progenitors out of the human blood stores of CyteBase and analyse them as one
small dataset.

``discover`` reads four progenitor marker genes for every primary peripheral-blood cell of each store, with the cell's
library size, label, donor and sample, and keeps the cells labelled as progenitors or carrying any of the four genes,
beside each donor and sample's cell total. ``extract`` builds, one store at a time, a composite of the chosen cells and
exports it to H5AD while the store's count chunks are in the composite's source cache, so each chunk is read from
Hugging Face once; a job takes a share of the stores and several jobs run at once. ``analyse`` joins the extracted
cells into one store, runs Harmony by study, UMAP and Leiden on it, and averages a fixed panel of progenitor and
lineage genes per cluster. ``severity`` reads the clinical columns that record each donor's COVID-19 severity, for the
same cells and samples as ``discover``.
"""

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.catalogue.catalogue_screen import (
    HubTraffic,
    apply_sample_rule,
    ensembl_gene_ids,
    hub_traffic,
    read_codes,
)
from experiments.reuse.blood_corpus import (
    annotate_corpus,
    corpus_settings,
    hub_token,
    jsonable,
    open_source,
    save_cells,
    storage,
)
from experiments.reuse.composite_and_remote import measure
from experiments.runtime.common import configure_cytearc
from experiments.runtime.recording_store import NetworkProbe, probe_remote_reads

WORK = Path("/tmp/blood_progenitors")
# Peripheral blood only: cord blood and bone marrow progenitors are another biology.
PERIPHERAL_BLOOD = ("blood", "venous blood")
# The four marker genes of the discovery pass; IDs as the stores' own feature tables name the symbols.
PROGENITOR_GENES = {
    "CD34": "ENSG00000174059",
    "SPINK2": "ENSG00000128040",
    "PROM1": "ENSG00000007062",
    "CRHBP": "ENSG00000145708",
}
# The CELLxGENE labels of haematopoietic stem and progenitor cells; precursor B cells are not among them.
PROGENITOR_LABELS = frozenset(
    {
        "hematopoietic precursor cell",
        "hematopoietic multipotent progenitor cell",
        "hematopoietic stem cell",
        "cord blood hematopoietic stem cell",
        "CD34-positive, CD38-negative hematopoietic stem cell",
        "common myeloid progenitor",
        "common lymphoid progenitor",
        "granulocyte monocyte progenitor cell",
        "megakaryocyte-erythroid progenitor cell",
        "common dendritic progenitor",
        "basophil mast progenitor cell",
        "erythroid progenitor cell",
        "erythroid progenitor cell, mammalian",
        "myeloid lineage restricted progenitor cell",
        "lymphoid lineage restricted progenitor cell",
        "megakaryocyte progenitor cell",
    }
)
CELL_COLUMNS = ("donor_id", "disease", "development_stage", "sex", "cell_type")
# The genes averaged per cluster to name the progenitor states, fixed before any cluster was seen: stem and
# multipotent, lymphoid, B-lymphoid, myeloid, erythroid, megakaryocyte, basophil and mast, dendritic, cycling, then
# mature lineage markers. IDs as the stores' own feature tables name the symbols. The B-lymphoid genes are IGLL1, EBF1,
# CD79A and PAX5 because four large stores do not measure IGHM or VPREB1.
STATE_PANEL = {
    "CD34": "ENSG00000174059", "SPINK2": "ENSG00000128040", "PROM1": "ENSG00000007062",
    "CRHBP": "ENSG00000145708", "HLF": "ENSG00000108924", "AVP": "ENSG00000101200",
    "MECOM": "ENSG00000085276", "MLLT3": "ENSG00000171843", "PROCR": "ENSG00000101000",
    "FLT3": "ENSG00000122025", "SATB1": "ENSG00000182568", "DNTT": "ENSG00000107447",
    "LTB": "ENSG00000227507", "CD7": "ENSG00000173762", "IGLL1": "ENSG00000128322", "EBF1": "ENSG00000164330",
    "CD79A": "ENSG00000105369", "PAX5": "ENSG00000196092", "MPO": "ENSG00000005381", "ELANE": "ENSG00000197561",
    "AZU1": "ENSG00000172232", "CTSG": "ENSG00000100448", "CEBPA": "ENSG00000245848",
    "LYZ": "ENSG00000090382", "S100A9": "ENSG00000163220", "GATA1": "ENSG00000102145",
    "KLF1": "ENSG00000105610", "TFRC": "ENSG00000072274", "CA1": "ENSG00000133742",
    "HBB": "ENSG00000244734", "PF4": "ENSG00000163737", "PPBP": "ENSG00000163736",
    "ITGA2B": "ENSG00000005961", "GP9": "ENSG00000169704", "VWF": "ENSG00000110799",
    "HDC": "ENSG00000140287", "MS4A2": "ENSG00000149534", "CPA3": "ENSG00000163751",
    "IRF8": "ENSG00000140968", "SPIB": "ENSG00000269404", "MKI67": "ENSG00000148773",
    "TOP2A": "ENSG00000131747", "CD3E": "ENSG00000198851", "MS4A1": "ENSG00000156738",
    "NKG7": "ENSG00000105374", "CD14": "ENSG00000170458",
}  # fmt: skip


def candidate_cells(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Cells labelled as progenitors or carrying any of the marker genes (NaN where the store lacks a gene)."""
    labelled = np.isin(np.asarray(labels, dtype=object), list(PROGENITOR_LABELS))
    return labelled | (np.nan_to_num(values, nan=0.0) > 0).any(axis=1)


def donor_sample_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Cells and labelled progenitors per donor and sample, with the sample's earliest time keys. ``frame`` has a row
    per cell with the columns of CELL_COLUMNS, ``sample``, ``sampleTime0``, ``sampleTime1`` and ``labelled``."""
    keys = ["donor_id", "sample"]
    grouped = frame.groupby(keys, sort=False, observed=True, dropna=False)
    totals = grouped.agg(
        cells=("labelled", "size"),
        labelledProgenitors=("labelled", "sum"),
        sampleTime0=("sampleTime0", "min"),
        sampleTime1=("sampleTime1", "min"),
        disease=("disease", "first"),
        development_stage=("development_stage", "first"),
        sex=("sex", "first"),
    )
    return totals.reset_index()


def marker_values(ds, rows: np.ndarray, genes: dict, nthreads: int) -> tuple[np.ndarray, list[str]]:
    """Raw counts of ``genes`` (symbol to Ensembl ID) for ``rows``, one column per gene in the dict's order, NaN where
    the store lacks the gene; and the genes it lacks."""
    from cytearc.assay.normalization import feature_value_blocks
    from cytearc.metadata.selection import NormalizationSpec

    feature_ids = ensembl_gene_ids(ds.RNA.feats.fetch_all("ids"))
    positions = {}
    for symbol, gene_id in genes.items():
        found = np.flatnonzero(feature_ids == gene_id)
        if len(found) > 1:
            raise ValueError(f"The store holds {gene_id} ({symbol}) more than once")
        if len(found):
            positions[symbol] = int(found[0])
    symbols = list(genes)
    values = np.full((len(rows), len(symbols)), np.nan, dtype=np.float32)
    present = sorted(positions, key=positions.get)
    if len(rows) and present:
        order = np.array([positions[symbol] for symbol in present], dtype=np.int64)
        columns = [symbols.index(symbol) for symbol in present]
        _, blocks = feature_value_blocks(
            ds.RNA,
            order,
            rows,
            NormalizationSpec(source="raw"),
            nthreads=nthreads,
            resident_bytes=int(rows.nbytes + values.nbytes),
            scratch_bytes_per_row=16 * len(order),
        )
        start = 0
        for block in blocks:
            block = np.asarray(block, dtype=np.float32)
            values[start : start + len(block), columns] = block
            start += len(block)
        if start != len(rows):
            raise AssertionError(f"Streamed {start} of {len(rows)} cells")
    return values, sorted(set(symbols) - set(positions))


def peripheral_blood_rows(primary, tissue) -> np.ndarray:
    """The rows of a store's primary cells from peripheral blood."""
    return np.flatnonzero(
        np.asarray(primary, dtype=bool) & np.isin(np.asarray(tissue, dtype=str), PERIPHERAL_BLOOD)
    )


def discover_store(name: str, location: str, collection: str, token, payload: dict) -> dict:
    started = time.perf_counter()
    threads = payload.get("nthreads", 8)
    ds = open_source(location, token, payload.get("memBudget", "3G"), threads)
    primary, tissue, totals = ds.cells.fetch_all_columns(
        ("is_primary_data", "tissue", "RNA_nCounts")
    )
    rows = peripheral_blood_rows(primary, tissue)
    totals = np.asarray(totals, dtype=np.float64)[rows]
    keep, excluded, sample_codes, sample_names, times = apply_sample_rule(ds, name, rows)
    rows, totals, sample_codes, times = rows[keep], totals[keep], sample_codes[keep], times[keep]
    frame = pd.DataFrame(
        {
            "source_row": rows,
            "sample": pd.Categorical.from_codes(sample_codes, categories=sample_names),
            "sampleTime0": times[:, 0],
            "sampleTime1": times[:, 1],
        }
    )
    for column in CELL_COLUMNS:
        codes, names, _ = read_codes(ds, column, rows)
        frame[column] = pd.Categorical.from_codes(codes, categories=names)
    values, missing = marker_values(ds, rows, PROGENITOR_GENES, threads)
    frame["labelled"] = frame["cell_type"].astype(str).isin(list(PROGENITOR_LABELS)).to_numpy()
    frame["librarySize"] = totals
    for index, symbol in enumerate(PROGENITOR_GENES):
        frame[symbol] = values[:, index]
    candidates = candidate_cells(frame["cell_type"].astype(str).to_numpy(), values)
    cells = frame[candidates].assign(storeId=name, collectionId=collection)
    donors = donor_sample_totals(frame).assign(storeId=name, collectionId=collection)
    return {
        "store": {
            "storeId": name,
            "cells": int(ds.cells.N),
            "peripheralBloodCells": int(len(keep)),
            "excludedCells": int((~keep).sum()),
            "excludedValues": excluded,
            "cellsRead": len(rows),
            "labelledProgenitors": int(frame["labelled"].sum()),
            "candidates": int(candidates.sum()),
            "missingGenes": missing,
            "seconds": time.perf_counter() - started,
        },
        "cells": cells,
        "donors": donors,
    }


def write_table(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_parquet(path, index=False)
    return {"rows": len(frame), "bytes": path.stat().st_size}


def discover(payload: dict) -> dict:
    """The discovery pass over the payload's sources: candidate cells and donor-sample totals, uploaded beside the
    run record."""
    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    token = hub_token(payload)
    traffic = HubTraffic(payload.get("quotaReserve", 200))
    started = time.perf_counter()
    stores, cells, donors, failures = [], [], [], []
    with hub_traffic(traffic), ThreadPoolExecutor(payload.get("storeConcurrency", 4)) as pool:
        futures = {
            pool.submit(
                discover_store, name, location, payload["collections"][name], token, payload
            ): name
            for name, location in payload["sources"].items()
        }
        for future in as_completed(futures):
            try:
                found = future.result()
            except Exception as error:
                failures.append({"storeId": futures[future], "error": repr(error)[:2000]})
                continue
            found["store"]["io"] = traffic.store_summary(found["store"]["storeId"])
            stores.append(found["store"])
            cells.append(found["cells"])
            donors.append(found["donors"])
    result = {
        "settings": {k: v for k, v in payload.items() if k not in ("r2EndpointUrl", "sources")},
        "genes": PROGENITOR_GENES,
        "labels": sorted(PROGENITOR_LABELS),
        "tissues": list(PERIPHERAL_BLOOD),
        "wallSeconds": time.perf_counter() - started,
        "traffic": traffic.totals(),
        "stores": sorted(stores, key=lambda s: s["storeId"]),
        "failures": failures,
        "tables": {},
    }
    r2, prefix = storage(payload), payload["outputPrefix"].rstrip("/")
    for key, frames in (("cells", cells), ("donors", donors)):
        path = work / f"progenitor_{key}.parquet"
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for column in table.columns:
            if isinstance(table[column].dtype, pd.CategoricalDtype):
                table[column] = table[column].astype(str)
        result["tables"][key] = {**write_table(table, path), "uri": f"{prefix}/{path.name}"}
        r2.upload(path, f"{prefix}/{path.name}")
    if failures:
        raise RuntimeError(f"{len(failures)} stores failed: {failures}")
    return jsonable(result)


def sample_values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cells per donor, sample and value of each of ``columns``, one row each, for a ``frame`` with a row per cell and
    the columns ``donor_id``, ``sample`` and ``columns``."""
    if not columns:
        raise ValueError("name at least one column to count the values of")
    parts = []
    for column in columns:
        counted = frame.groupby(["donor_id", "sample", column], observed=True, sort=True).size()
        parts.append(
            counted.rename("cells")
            .reset_index()
            .rename(columns={column: "value"})
            .assign(column=column)
        )
    table = pd.concat(parts, ignore_index=True)
    for key in ("donor_id", "sample", "value"):
        table[key] = table[key].astype(str)
    return table[["donor_id", "sample", "column", "value", "cells"]]


def severity_store(
    name: str, location: str, columns: list[str], token, payload: dict
) -> pd.DataFrame:
    started = time.perf_counter()
    ds = open_source(location, token, payload.get("memBudget", "3G"), payload.get("nthreads", 8))
    primary, tissue = ds.cells.fetch_all_columns(("is_primary_data", "tissue"))
    rows = peripheral_blood_rows(primary, tissue)
    keep, _, sample_codes, sample_names, _ = apply_sample_rule(ds, name, rows)
    rows = rows[keep]
    frame = pd.DataFrame(
        {"sample": pd.Categorical.from_codes(sample_codes[keep], categories=sample_names)}
    )
    for column in ("donor_id", *columns):
        codes, names, found = read_codes(ds, column, rows)
        if not found:
            raise KeyError(f"{name} has no cell column {column!r}")
        frame[column] = pd.Categorical.from_codes(codes, categories=names)
    table = sample_values(frame, list(columns)).assign(storeId=name)
    return table.assign(seconds=time.perf_counter() - started)


def severity(payload: dict) -> dict:
    """Cells per donor, sample and value of the clinical columns that ``payload["columns"]`` names for each store, over
    the cells and samples that ``discover`` counts, so each donor's recorded COVID-19 severity joins its totals."""
    configure_cytearc()
    token = hub_token(payload)
    traffic = HubTraffic(payload.get("quotaReserve", 200))
    started = time.perf_counter()
    tables, failures = [], []
    with hub_traffic(traffic), ThreadPoolExecutor(payload.get("storeConcurrency", 4)) as pool:
        futures = {
            pool.submit(
                severity_store, name, location, payload["columns"][name], token, payload
            ): name
            for name, location in payload["sources"].items()
        }
        for future in as_completed(futures):
            try:
                tables.append(future.result())
            except Exception as error:
                failures.append({"storeId": futures[future], "error": repr(error)[:2000]})
    if failures:
        raise RuntimeError(f"{len(failures)} stores failed: {failures}")
    table = pd.concat(tables, ignore_index=True)
    seconds = table.groupby("storeId")["seconds"].first().to_dict()
    table = table.drop(columns="seconds").sort_values(
        ["storeId", "donor_id", "sample", "column", "value"]
    )
    return jsonable(
        {
            "settings": {
                k: v
                for k, v in payload.items()
                if k not in ("r2EndpointUrl", "sources", "recordUri")
            },
            "tissues": list(PERIPHERAL_BLOOD),
            "wallSeconds": time.perf_counter() - started,
            "storeSeconds": seconds,
            "traffic": traffic.totals(),
            "rows": table.to_dict(orient="records"),
        }
    )


def extract(payload: dict) -> dict:
    """One composite per store of the payload's chosen rows, exported to H5AD and uploaded; a store's chunks stay in
    the composite's source cache between creation and export, so the export reads none of them again."""
    from cytearc.composite import create_composite
    from cytearc.writers import to_h5ad

    os.environ["CYTEARC_SOURCE_CACHE_BYTES"] = payload.get("sourceCacheBytes", "64G")
    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "16G"), payload.get("nthreads", 16)
    token = hub_token(payload)
    traffic, probe = HubTraffic(payload.get("quotaReserve", 200)), NetworkProbe()
    r2, prefix = storage(payload), payload["outputPrefix"].rstrip("/")
    started = time.perf_counter()
    result = {"stores": {}}
    with ExitStack() as stack:
        stack.enter_context(hub_traffic(traffic))
        stack.enter_context(probe_remote_reads(probe))
        # One store at a time, so its touched chunks fit the source cache together.
        for name, location in payload["sources"].items():
            rows = np.unique(np.asarray(payload["rows"][name], dtype=np.int64))
            entry: dict = {"cells": len(rows)}
            ds, entry["opening"] = measure(
                lambda location=location: open_source(location, token, memory, threads), probe
            )
            target = work / f"{name}.zarr"
            joint, entry["creation"] = measure(
                lambda ds=ds, rows=rows, target=target, name=name: create_composite(
                    {name: ds},
                    at=target,
                    assay="RNA",
                    features="union",
                    rows={name: rows},
                    mem_budget=memory,
                    nthreads=threads,
                ),
                probe,
            )
            if int(joint.cells.N) != len(rows):
                raise AssertionError(
                    f"{name}: the composite has {joint.cells.N} of {len(rows)} cells"
                )
            joint.cells.insert("source_row", rows, overwrite=True)
            path = work / f"{name}.h5ad"
            _, entry["export"] = measure(
                lambda joint=joint, path=path: to_h5ad(
                    joint.RNA, str(path), embeddings_cols=[], nthreads=threads
                ),
                probe,
            )
            entry["h5adBytes"] = path.stat().st_size
            entry["uri"] = f"{prefix}/{name}.h5ad"
            r2.upload(path, entry["uri"])
            entry["io"] = traffic.store_summary(name)
            result["stores"][name] = entry
            del joint, ds
            # The job's own scratch: the store's composite and export are uploaded and no longer needed.
            shutil.rmtree(target)
            path.unlink()
    result["wallSeconds"] = time.perf_counter() - started
    result["traffic"] = traffic.totals()
    result["settings"] = {k: v for k, v in payload.items() if k not in ("r2EndpointUrl", "rows")}
    return jsonable(result)


def bare_feature_ids(path: Path) -> int:
    """Rewrite an H5AD's feature index in place as bare Ensembl IDs, without version suffixes or genome prefixes, so
    stores join on the same IDs; returns how many features changed. Two features with one Ensembl ID are an error."""
    import h5py

    with h5py.File(path, "r+") as handle:
        var = handle["var"]
        key = var.attrs["_index"]
        # A plain string array, as CyteArc's export writes it, or a nullable one, as newer AnnData writes it.
        parent, name = (var[key], "values") if isinstance(var[key], h5py.Group) else (var, key)
        ids = np.asarray(parent[name].asstr()[()], dtype=object).astype(str)
        bare = ensembl_gene_ids(ids)
        # A feature without a human Ensembl gene ID, such as a SARS-CoV-2 gene of a COVID-19 reference, keeps its own
        # ID; stores without it do not share it, so the join on shared genes leaves it out.
        renamed = np.where(bare == "", ids, bare)
        if len(set(renamed)) != len(renamed):
            raise ValueError(f"{path.name}: two of its features share an Ensembl ID")
        changed = int((renamed != ids).sum())
        if changed:
            attrs = dict(parent[name].attrs)
            del parent[name]
            dataset = parent.create_dataset(
                name, data=renamed.astype(object), dtype=h5py.string_dtype()
            )
            dataset.attrs.update(attrs)
    return changed


def subset_store(paths: dict[str, Path], work: Path, memory: str, threads: int):
    """The extracted stores as one composite of local stores on the genes every one measures. Each H5AD's features
    become bare Ensembl IDs and it is converted to a store; the source's cell columns, which the extraction exported
    with an orig_ prefix, are copied without it, so the composite carries them as the orig_ columns the corpus
    annotation reads."""
    import cytearc
    from cytearc.composite import create_composite
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    stores = {}
    for name, path in paths.items():
        bare_feature_ids(path)
        target = work / "stores" / f"{name}.zarr"
        target.parent.mkdir(parents=True, exist_ok=True)
        reader = H5adReader(str(path))
        try:
            H5adToZarr(
                reader, str(target), assay_name="RNA", mem_budget=memory, nthreads=threads
            ).dump()
        finally:
            reader.close()
        store = cytearc.DataStore(
            str(target), min_features_per_cell=-1, mem_budget=memory, nthreads=threads
        )
        for column in CELL_COLUMNS:
            store.cells.insert(column, store.cells.fetch_all(f"orig_{column}"), overwrite=True)
        stores[name] = store
    return create_composite(
        stores,
        at=work / "progenitors.zarr",
        assay="RNA",
        features="intersection",
        mem_budget=memory,
        nthreads=threads,
    )


def cluster_panel_means(
    log_values: np.ndarray, clusters: np.ndarray, genes: list[str]
) -> pd.DataFrame:
    """Mean log-normalised expression and the share of cells expressing each panel gene, per cluster."""
    frame = pd.DataFrame(log_values, columns=genes)
    frame["cluster"] = clusters
    means = frame.groupby("cluster").mean().add_suffix("_mean")
    shares = (
        frame.drop(columns="cluster").gt(0).groupby(frame["cluster"]).mean().add_suffix("_share")
    )
    sizes = frame.groupby("cluster").size().rename("cells")
    return pd.concat([sizes, means, shares], axis=1).reset_index()


def analyse(payload: dict) -> dict:
    """Join the extracted cells into one composite of local stores and analyse it: Harmony by study, UMAP, Leiden,
    and the state panel's expression per cluster."""
    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "32G"), payload.get("nthreads", 16)
    r2, prefix = storage(payload), payload["outputPrefix"].rstrip("/")
    probe = NetworkProbe()
    result: dict = {}
    paths = {}
    for name, uri in payload["extracted"].items():
        paths[name] = work / "extracted" / f"{name}.h5ad"
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        r2.download(uri, paths[name])
    ds, result["joining"] = measure(lambda: subset_store(paths, work, memory, threads), probe)
    result["cells"], result["genes"] = int(ds.cells.N), int(ds.RNA.feats.N)
    result["annotation"] = annotate_corpus(ds, payload["collections"])
    run, result["analysis"] = measure(
        lambda: ds.pipeline.run(**corpus_settings(payload, integrated=True)), probe
    )
    cells_path = work / "progenitor_cells.npz"
    result["savedCells"] = save_cells(ds, run, cells_path)
    clusters = np.load(cells_path)
    # Each cell's store and row beside its map position and clusters, so donor-level counts can keep each donor's
    # earliest sample from the discovery tables.
    cell_table = pd.DataFrame(
        {
            "dataset": np.asarray(ds.cells.fetch_all("dataset"), dtype=str),
            "source_row": np.asarray(ds.cells.fetch_all("orig_source_row"), dtype=np.int64),
            "umap1": clusters["umap"][:, 0],
            "umap2": clusters["umap"][:, 1],
            **{name: clusters[name] for name in clusters.files if name.startswith("leiden")},
        }
    )
    table_path = work / "progenitor_cell_table.parquet"
    cell_table.to_parquet(table_path, index=False)
    selected = f"leiden_{payload.get('leidenResolutions', [1.0])[0]}".replace(".", "_")
    names = [name for name in clusters.files if name.startswith("leiden")]
    labels = clusters[selected] if selected in clusters.files else clusters[names[0]]
    raw, missing = marker_values(ds, np.arange(ds.cells.N), STATE_PANEL, threads)
    panel = [gene for gene in STATE_PANEL if gene not in missing]
    measured = [list(STATE_PANEL).index(gene) for gene in panel]
    library = np.asarray(ds.cells.fetch_all("RNA_nCounts"), dtype=np.float64)
    log_values = np.log1p(raw[:, measured] / library[:, None] * 1e4)
    table = cluster_panel_means(log_values, labels, panel)
    panel_path = work / "progenitor_cluster_panel.parquet"
    table.to_parquet(panel_path, index=False)
    result["panelGenes"] = panel
    result["panelGenesMissing"] = missing
    outputs = (cells_path, panel_path, table_path)
    for path in outputs:
        r2.upload(path, f"{prefix}/{path.name}")
    result["outputs"] = {path.name: f"{prefix}/{path.name}" for path in outputs}
    return jsonable(result)
