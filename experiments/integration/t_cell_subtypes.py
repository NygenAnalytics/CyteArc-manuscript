"""T-cell subtypes from one CITE-seq study's WNN graph, and another study's T cells mapped onto them by RNA alone.

For one study's T cells, t_cell_subtypes builds the RNA, protein and WNN graphs as multimodal_neighbors builds them for
whole runs, cuts the WNN and RNA graphs into clusters at several Leiden resolutions, and summarizes each cluster's
surface protein, author labels and donors. Given a query store, it then registers the T cells' RNA neighbor chain as a
mapping reference, maps the query's T cells onto it by their RNA alone, and transfers every partition's clusters to
them. Each study is analyzed on its own, so no correction between studies is needed.
"""

import hashlib
import os
import tarfile
import time
from pathlib import Path

import cytearc
import numpy as np
from cytearc.assay.normalization import clr_values

from experiments.integration import multimodal_neighbors as mn
from experiments.runtime.common import configure_cytearc, provenance, write_csv

# The Leiden resolutions each T-cell graph is cut at, from a few broad groups to many small ones.
RESOLUTIONS = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2)
# Each mapped query cell keeps as many nearest reference cells as the reference chain holds neighbors per cell.
SAVE_K = mn.SETTINGS["neighborCount"]
# The share of the reference's variable genes the query must measure for its mapping to be meaningful.
MIN_FEATURE_COVERAGE = 0.8
# Each store's T cells: the label column and the values that name them, dividing cells left out, and the columns that
# give each cell's study, donor and finest author label.
T_CELLS = {
    "combat_stephenson_pbmc_citeseq": {
        "labelColumn": "cell_type",
        "labels": ("CD4 Naive", "CD4 Memory", "Treg", "CD8 Naive", "CD8 Memory", "MAIT", "gdT"),
        "studyColumn": "study",
        "donorColumn": "donor",
        "authorColumn": "author_cell_type",
    },
    "hao_pbmc_citeseq": {
        "labelColumn": "celltype.l2",
        "labels": (
            "CD4 Naive",
            "CD4 TCM",
            "CD4 TEM",
            "CD4 CTL",
            "Treg",
            "CD8 Naive",
            "CD8 TCM",
            "CD8 TEM",
            "MAIT",
            "gdT",
            "dnT",
        ),
        "studyColumn": None,
        "donorColumn": "donor",
        "authorColumn": "celltype.l3",
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def open_store(source: dict, work: Path, name: str, payload: dict):
    """Download and unpack the store archive at source's storeArchiveUri, check it against sourceArchiveSha256 when
    given, and open it for writing; returns the store and the archive's digest."""
    from experiments.runtime.object_io import download_file

    archive = work / f"{name}.tar.gz"
    download_file(source["storeArchiveUri"], archive)
    digest = sha256(archive)
    if source.get("sourceArchiveSha256") not in (None, digest):
        raise ValueError(f"The {name} store archive differs from the one recorded")
    store = work / f"{name}.zarr"
    store.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(store, filter="data")
    ds = cytearc.DataStore(
        str(store),
        default_assay="RNA",
        nthreads=payload.get("threads", 16),
        mem_budget=payload.get("memoryBudgetGiB", 32) * 1024**3,
    )
    return ds, digest


def t_cell_rows(ds, design: dict, study: str | None) -> np.ndarray:
    """The store rows of the eligible T cells, of one study when given, in store order."""
    labels = np.asarray(ds.cells.fetch_all(design["labelColumn"])).astype(str)
    missing = sorted(set(design["labels"]) - set(labels))
    if missing:
        raise ValueError(f"{design['labelColumn']} has no cells labelled {missing}")
    keep = np.asarray(ds.cells.fetch_all("I"), dtype=bool) & np.isin(labels, design["labels"])
    if study is not None:
        if design["studyColumn"] is None:
            raise ValueError("this store has no study column to choose a study by")
        studies = np.asarray(ds.cells.fetch_all(design["studyColumn"])).astype(str)
        if study not in set(studies):
            raise ValueError(f"no cells of study {study!r}; the store holds {sorted(set(studies))}")
        keep &= studies == study
    rows = np.flatnonzero(keep)
    if len(rows) <= mn.SETTINGS["neighborCount"]:
        raise ValueError(
            f"{len(rows)} T cells are too few for {mn.SETTINGS['neighborCount']} neighbors"
        )
    return rows


def select(ds, rows: np.ndarray, column: str):
    """A cell selection of the given store rows, saved under column."""
    mask = np.zeros(ds.cells.N, dtype=bool)
    mask[rows] = True
    ds.cells.insert(column, mask, overwrite=True)
    return ds.snapshot_cell_selection(column)


def feature_coverage(ds, features, query) -> float:
    """The share of the reference's selected RNA features that the query store measures, matched by feature id."""
    chosen = np.asarray(ds.artifacts.load(features)["values"][:]) != 0
    reference = np.asarray(ds.RNA.feats.fetch_all("ids")).astype(str)[chosen]
    measured = set(np.asarray(query.RNA.feats.fetch_all("ids")).astype(str))
    return float(np.isin(reference, list(measured)).mean())


def cell_clr(_assay, counts):
    """The centered log-ratio of each cell over its proteins, as Seurat's CITE-seq vignettes normalize (CLR, margin 2):
    each row is divided by the exponential of its mean log1p value and then log1p transformed, so a cell's total
    antibody capture leaves its values. CyteArc's own ADT normalization centers each protein over the cells instead."""
    means = np.log1p(counts, dtype=np.float64).mean(axis=1)
    means = np.asarray(means.compute() if hasattr(means, "compute") else means, dtype=np.float64)
    return np.log1p(counts / np.exp(means).reshape(-1, 1))


cell_clr.artifact_identity = "experiments.integration.t_cell_subtypes:cell-axis-clr"  # type: ignore[attr-defined]
# How a payload's adtNormalization names each ADT normalization: CyteArc's default over the cells, or over each cell.
ADT_NORMALIZATIONS = ("feature", "cell")


def protein_values(
    ds, rows: np.ndarray, normalization: str = "feature", block: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    """Each protein's name and each selected cell's CLR-normalized protein, centered over the cells as multimodal_biology
    summarizes clusters, or over each cell; a cell's scale over each cell comes from its proteins other than the isotype
    controls, the ones the WNN graph uses."""
    if normalization not in ADT_NORMALIZATIONS:
        raise ValueError(
            f"adtNormalization must be one of {ADT_NORMALIZATIONS}, not {normalization!r}"
        )
    names = np.asarray(ds.ADT.feats.fetch_all("names")).astype(str)
    counts = np.vstack(
        [
            np.asarray(ds.ADT.rawData[rows[start : start + block], :].compute(), dtype=np.float64)
            for start in range(0, len(rows), block)
        ]
    )
    if normalization == "feature":
        return names, clr_values(counts)
    used = np.array(["control" not in name.lower() for name in names])
    scale = np.exp(np.log1p(counts[:, used]).mean(axis=1, keepdims=True))
    return names, np.log1p(counts / scale)


def cluster_summaries(
    groups: np.ndarray,
    clr: np.ndarray | None,
    names: np.ndarray | None,
    authors: np.ndarray,
    donors: np.ndarray,
    **key,
) -> tuple[list[dict], list[dict], list[dict]]:
    """For each group of cells: each protein's median and mean CLR where the cells have protein, the cells of each
    author label, and its donors with the largest one's share; key names the partition in every row."""
    proteins, labels, spread = [], [], []
    for group in np.unique(groups):
        inside = groups == group
        cells = int(inside.sum())
        if clr is not None:
            values = clr[inside]
            for name, median, mean in zip(
                names, np.median(values, axis=0), values.mean(axis=0), strict=True
            ):
                proteins.append(
                    {
                        **key,
                        "cluster": str(group),
                        "cells": cells,
                        "protein": str(name),
                        "medianClr": float(median),
                        "meanClr": float(mean),
                    }
                )
        found, counts = np.unique(authors[inside], return_counts=True)
        labels.extend(
            {**key, "cluster": str(group), "authorLabel": str(label), "cells": int(count)}
            for label, count in zip(found, counts, strict=True)
        )
        _, per_donor = np.unique(donors[inside], return_counts=True)
        spread.append(
            {
                **key,
                "cluster": str(group),
                "cells": cells,
                "donors": int(len(per_donor)),
                "largestDonorShare": float(per_donor.max() / cells),
            }
        )
    return proteins, labels, spread


def build_t_cell_graphs(ds, rows: np.ndarray, stages: list[dict]) -> tuple[dict, object]:
    """The RNA and protein chains of the selected T cells and their WNN graph, as multimodal_neighbors builds a run's."""
    selection = select(ds, rows, "t_cells")
    modalities = {
        assay: mn.prepare_modality(ds, selection, assay, stages, invalidate=False)
        for assay in ("RNA", "ADT")
    }
    wnn, _ = mn.integrate(
        ds, modalities, "wnn", stages, invalidate=False, resolution=RESOLUTIONS[0]
    )
    return modalities, wnn


def register_reference(ds, modalities: dict, stages: list[dict]):
    """The T cells' RNA neighbor chain registered as a mapping reference."""
    return mn.measure(
        stages,
        "mapping.registration",
        lambda: ds.mapping.load_reference(
            ds.mapping.build_reference(modalities["RNA"]["neighbors"])
        ),
    )


def map_query(
    ds,
    modalities: dict,
    model,
    query,
    rows: np.ndarray,
    stages: list[dict],
    stage: str,
    min_coverage: float = MIN_FEATURE_COVERAGE,
):
    """Map the query's selected cells onto the reference by RNA alone, after checking the query measures at least
    min_coverage of the reference's variable genes; returns the projection and that share."""
    coverage = feature_coverage(ds, modalities["RNA"]["features"], query)
    if coverage < min_coverage:
        raise ValueError(
            f"{stage}: the query measures {coverage:.1%} of the reference's variable genes, below {min_coverage:.0%}"
        )
    selection = select(query, rows, "t_cells")
    projection = mn.measure(
        stages, stage, lambda: query.mapping.run(model, selection, save_k=SAVE_K)
    )
    return projection, coverage


def t_cell_subtypes(payload: dict) -> dict:
    """Cluster one study's T cells on their WNN and RNA graphs at each resolution, and map a query's T cells onto them.

    Payload: dataset (a key of T_CELLS), storeArchiveUri, optional sourceArchiveSha256, study and adtNormalization
    (feature, CyteArc's default, or cell, see cell_clr), threads, memoryBudgetGiB and recordUri; optional query with dataset, storeArchiveUri and sourceArchiveSha256. Saves each
    cell's clusters, author label, donor and WNN modality weights, and each query cell's transferred clusters and vote
    shares, to the .npz beside the record, and the cluster summaries to CSV files beside it.
    """
    from experiments.runtime.object_io import upload_file

    configure_cytearc()
    started = time.perf_counter()
    stages: list[dict] = []
    work = Path(payload.get("workDir", "/tmp/t_cell_subtypes"))
    work.mkdir(parents=True, exist_ok=True)
    design = T_CELLS[payload["dataset"]]
    normalization = payload.get("adtNormalization", "feature")
    if normalization not in ADT_NORMALIZATIONS:
        raise ValueError(
            f"adtNormalization must be one of {ADT_NORMALIZATIONS}, not {normalization!r}"
        )
    ds, digest = open_store(payload, work, "reference", payload)
    if normalization == "cell":
        ds.ADT.normMethod = cell_clr
    rows = t_cell_rows(ds, design, payload.get("study"))
    modalities, wnn = build_t_cell_graphs(ds, rows, stages)
    graphs = {"wnn": wnn, "rna": modalities["RNA"]["graph"]}
    partitions = {}
    for graph_name, graph in graphs.items():
        for resolution in RESOLUTIONS:
            partitions[(graph_name, resolution)] = mn.measure(
                stages,
                f"{graph_name}.leiden_{resolution:g}",
                lambda graph=graph, resolution=resolution: ds.clusters.leiden(
                    graph,
                    resolution=resolution,
                    random_seed=mn.SETTINGS["seed"],
                    backend="cytearc",
                ),
            )
    clusters = {
        key: np.asarray(ds.artifacts.load(ref)["values"][:]) for key, ref in partitions.items()
    }
    if any(len(values) != len(rows) for values in clusters.values()):
        raise ValueError("a partition does not label each selected T cell once")

    def column(store, name: str, selected: np.ndarray) -> np.ndarray:
        return np.asarray(store.cells.fetch_all(name)).astype(str)[selected]

    authors = column(ds, design["authorColumn"], rows)
    donors = column(ds, design["donorColumn"], rows)
    names, clr = protein_values(ds, rows, normalization)
    summaries: dict[str, list[dict]] = {"proteins": [], "authors": [], "donors": []}
    for (graph_name, resolution), values in clusters.items():
        for name, found in zip(
            summaries,
            cluster_summaries(
                values, clr, names, authors, donors, graph=graph_name, resolution=resolution
            ),
            strict=True,
        ):
            summaries[name].extend(found)
    arrays = {
        "reference_rows": rows,
        "reference_ids": column(ds, "ids", rows),
        "reference_labels": column(ds, design["labelColumn"], rows),
        "reference_authors": authors,
        "reference_donors": donors,
        "modality_weights": np.asarray(
            ds.artifacts.load(wnn)["modality_weights"][:], dtype=np.float32
        ),
        **{
            f"{graph_name}_{resolution:g}": values
            for (graph_name, resolution), values in clusters.items()
        },
    }
    result = {
        "status": "ok",
        "dataset": payload["dataset"],
        "adtNormalization": normalization,
        "study": payload.get("study"),
        "sourceArchiveSha256": digest,
        "cells": int(len(rows)),
        "cellSelectionSha256": mn.array_hash(rows),
        "tCellLabels": list(design["labels"]),
        "proteins": int(len(names)),
        "clusters": {f"{g}_{r:g}": int(len(np.unique(v))) for (g, r), v in clusters.items()},
        "modalityWeightMean": dict(
            zip(("RNA", "ADT"), arrays["modality_weights"].mean(axis=0).tolist(), strict=True)
        ),
    }
    if "query" in payload:
        source = payload["query"]
        query_design = T_CELLS[source["dataset"]]
        model = register_reference(ds, modalities, stages)
        query, query_digest = open_store(source, work, "query", payload)
        query_rows = t_cell_rows(query, query_design, source.get("study"))
        projection, coverage = map_query(
            ds, modalities, model, query, query_rows, stages, "mapping.projection"
        )
        query_authors = column(query, query_design["authorColumn"], query_rows)
        query_donors = column(query, query_design["donorColumn"], query_rows)
        query_names, query_clr = protein_values(query, query_rows, normalization)
        query_summaries: dict[str, list[dict]] = {"proteins": [], "authors": [], "donors": []}
        transferred = {}
        for (graph_name, resolution), ref in partitions.items():
            key = f"{graph_name}_{resolution:g}"
            outcome = mn.measure(
                stages,
                f"mapping.transfer_{key}",
                lambda ref=ref: query.mapping.load_label_transfer(
                    query.mapping.transfer_labels(
                        projection, reference=model, reference_labels=ref, threshold_fraction=0.0
                    )
                ),
            )
            labels = outcome.labels.fillna("unassigned").to_numpy(dtype=str)
            if len(labels) != len(query_rows):
                raise ValueError(
                    f"the {key} transfer labels {len(labels)} of {len(query_rows)} query cells"
                )
            transferred[key] = labels
            arrays[f"transferred_{key}"] = labels
            arrays[f"vote_{key}"] = outcome.evidence.voteFraction.to_numpy(dtype=np.float32)
            for name, found in zip(
                query_summaries,
                cluster_summaries(
                    labels,
                    query_clr,
                    query_names,
                    query_authors,
                    query_donors,
                    graph=graph_name,
                    resolution=resolution,
                ),
                strict=True,
            ):
                query_summaries[name].extend(found)
        arrays.update(
            query_rows=query_rows,
            query_ids=column(query, "ids", query_rows),
            query_labels=column(query, query_design["labelColumn"], query_rows),
            query_authors=query_authors,
            query_donors=query_donors,
        )
        for name, found in query_summaries.items():
            summaries[f"query_{name}"] = found
        result["query"] = {
            "dataset": source["dataset"],
            "sourceArchiveSha256": query_digest,
            "cells": int(len(query_rows)),
            "cellSelectionSha256": mn.array_hash(query_rows),
            "proteins": int(len(query_names)),
            "featureCoverage": coverage,
            "unassigned": {
                key: int((labels == "unassigned").sum()) for key, labels in transferred.items()
            },
        }
    base = payload["recordUri"].removesuffix(".json")
    path = work / "t_cell_subtypes.npz"
    np.savez_compressed(path, **arrays)
    upload_file(path, f"{base}.npz")
    files = {"npz": {"uri": f"{base}.npz", "sha256": sha256(path)}}
    for name, found in summaries.items():
        table = write_csv(f"{name}.csv", found, directory=work)
        upload_file(table, f"{base}_{name}.csv")
        files[name] = {"uri": f"{base}_{name}.csv", "sha256": sha256(Path(table))}
    return {
        **result,
        "files": files,
        "resolutions": list(RESOLUTIONS),
        "saveK": SAVE_K,
        "stages": stages,
        "settings": mn.SETTINGS,
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
        "seconds": time.perf_counter() - started,
    }


def named_subtypes(partition: np.ndarray, names: dict[str, str]) -> np.ndarray:
    """Each cell's subtype from its cluster, blank for the clusters the names leave out; the names are keyed by cluster
    label as text."""
    found = {str(cluster) for cluster in np.unique(partition)}
    unknown = sorted(set(names) - found)
    if unknown:
        raise ValueError(f"the names give clusters {unknown}, which the partition does not hold")
    if any(not str(name).strip() for name in names.values()):
        raise ValueError("each named cluster needs a name that is not blank")
    return np.array([names.get(str(cluster), "") for cluster in partition], dtype=object).astype(
        str
    )


def prefixed_feature_column(path: Path, key: str, prefix: str) -> str:
    """Add to the H5AD's var a column holding prefix and each feature's key column value, so that its features can be
    keyed as a reference store keys its own (multimodal_neighbors' TenxCountsWriter keys genes as RNA:<name>); returns
    the new column's name."""
    import h5py

    from experiments.runtime.h5ad import read_column

    column = "prefixed_feature_id"
    with h5py.File(path, "r+") as handle:
        var = handle["var"]
        if column in var:
            raise ValueError(f"{path.name}'s var already holds a {column} column")
        values = np.char.add(prefix, read_column(var, key).astype(str))
        dataset = var.create_dataset(
            column, data=values.astype(object), dtype=h5py.string_dtype("utf-8")
        )
        dataset.attrs["encoding-type"] = "string-array"
        dataset.attrs["encoding-version"] = "0.2.0"
        order = [
            str(c.decode() if isinstance(c, bytes) else c)
            for c in var.attrs.get("column-order", [])
        ]
        var.attrs["column-order"] = np.asarray([*order, column], dtype=h5py.string_dtype("utf-8"))
    return column


def open_h5ad_query(source: dict, work: Path, payload: dict):
    """Download a prepared query H5AD, check it against source's sha256, and convert it to a store whose RNA features
    are keyed by source's featureIdsKey column, with source's featureIdPrefix in front when given, repeated keys made
    unique, keeping every cell; returns the store and the file's digest."""
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    from experiments.runtime.object_io import download_file

    path = work / f"{source['name']}.h5ad"
    download_file(source["uri"], path)
    digest = sha256(path)
    if source.get("sha256") not in (None, digest):
        raise ValueError(f"The {source['name']} H5AD differs from the one recorded")
    store = work / f"{source['name']}.zarr"
    key = source["featureIdsKey"]
    if source.get("featureIdPrefix"):
        key = prefixed_feature_column(path, key, source["featureIdPrefix"])
    reader = H5adReader(
        str(path),
        feature_ids_key=key,
        feature_name_key=source["featureIdsKey"],
        duplicate_feature_ids="make_unique",
    )
    try:
        H5adToZarr(
            reader,
            str(store),
            assay_name="RNA",
            mem_budget=f"{payload.get('memoryBudgetGiB', 32)}G",
            nthreads=payload.get("threads", 16),
        ).dump()
    finally:
        reader.close()
    ds = cytearc.DataStore(
        str(store),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=payload.get("threads", 16),
        mem_budget=payload.get("memoryBudgetGiB", 32) * 1024**3,
    )
    return ds, digest


def t_cell_reference(payload: dict) -> dict:
    """Build one study's T-cell reference with named subtypes, lay it out, and map queries' T cells onto it by RNA alone.

    Rebuilds the study's T-cell WNN graph as t_cell_subtypes does, cuts it at payload's resolution and checks that the
    partition is the one the named t_cell_subtypes run made (partitionSha256), so that subtypeNames, keyed by cluster,
    name the right cells; clusters left out stay unlabelled, and a query cell takes the vote of its named neighbors
    only. The WNN graph is laid out with UMAP from k-means of the RNA PCA, as graph_umaps lays out WNN graphs. Each
    query is a store archive of a T_CELLS dataset (kind store) or a prepared H5AD with its own T-cell design (kind
    h5ad), mapped onto the T cells' RNA neighbor chain.

    Payload: dataset, storeArchiveUri, sourceArchiveSha256, study, adtNormalization, resolution, partitionSha256,
    subtypeNames, queries, threads, memoryBudgetGiB and recordUri. Saves each reference cell's subtype, layout and
    author label, and each query cell's subtype, vote share and author label, to the .npz beside the record; and per
    subtype the protein, author labels and donors of the reference and of each query, to CSV files beside it.
    """
    from experiments.runtime.object_io import upload_file

    configure_cytearc()
    started = time.perf_counter()
    stages: list[dict] = []
    work = Path(payload.get("workDir", "/tmp/t_cell_reference"))
    work.mkdir(parents=True, exist_ok=True)
    design = T_CELLS[payload["dataset"]]
    normalization = payload["adtNormalization"]
    if normalization not in ADT_NORMALIZATIONS:
        raise ValueError(
            f"adtNormalization must be one of {ADT_NORMALIZATIONS}, not {normalization!r}"
        )
    ds, digest = open_store(payload, work, "reference", payload)
    if normalization == "cell":
        ds.ADT.normMethod = cell_clr
    rows = t_cell_rows(ds, design, payload.get("study"))
    modalities, wnn = build_t_cell_graphs(ds, rows, stages)
    resolution = float(payload["resolution"])
    clusters = mn.measure(
        stages,
        f"wnn.leiden_{resolution:g}",
        lambda: ds.clusters.leiden(
            wnn, resolution=resolution, random_seed=mn.SETTINGS["seed"], backend="cytearc"
        ),
    )
    partition = np.asarray(ds.artifacts.load(clusters)["values"][:])
    if mn.array_hash(partition) != payload["partitionSha256"]:
        raise ValueError("the rebuilt partition differs from the one whose clusters were named")
    subtypes = named_subtypes(partition, payload["subtypeNames"])
    labels = np.full(ds.cells.N, "", dtype=object)
    labels[rows] = subtypes
    ds.cells.insert("t_cell_subtype", labels.astype(str), overwrite=True)
    start = mn.measure(
        stages,
        "RNA.initialization",
        lambda: ds.embeddings.initialization(
            modalities["RNA"]["reduction"], rand_state=mn.SETTINGS["seed"]
        ),
    )
    layout_ref = mn.measure(
        stages,
        "wnn.umap",
        lambda: ds.embeddings.umap(
            wnn, start, nthreads=payload.get("threads", 16), **mn.UMAP_SETTINGS
        ),
    )
    layout_group = ds.artifacts.load(layout_ref)
    layout = np.asarray(layout_group["values"][:], dtype=np.float32)
    if layout.shape != (len(rows), 2) or not np.isfinite(layout).all():
        raise ValueError("the layout does not give each T cell finite coordinates")

    def column(store, name: str, selected: np.ndarray) -> np.ndarray:
        return np.asarray(store.cells.fetch_all(name)).astype(str)[selected]

    groups = np.where(subtypes == "", "unlabelled", subtypes)
    names, clr = protein_values(ds, rows, normalization)
    summaries = dict(
        zip(
            ("reference_proteins", "reference_authors", "reference_donors"),
            cluster_summaries(
                groups,
                clr,
                names,
                column(ds, design["authorColumn"], rows),
                column(ds, design["donorColumn"], rows),
                cellSet="reference",
            ),
            strict=True,
        )
    )
    del clr
    arrays = {
        "reference_rows": rows,
        "reference_ids": column(ds, "ids", rows),
        "reference_labels": column(ds, design["labelColumn"], rows),
        "reference_authors": column(ds, design["authorColumn"], rows),
        "reference_donors": column(ds, design["donorColumn"], rows),
        "reference_clusters": partition,
        "reference_subtypes": subtypes,
        "reference_umap": layout,
        "modality_weights": np.asarray(
            ds.artifacts.load(wnn)["modality_weights"][:], dtype=np.float32
        ),
    }
    result = {
        "status": "ok",
        "dataset": payload["dataset"],
        "study": payload.get("study"),
        "adtNormalization": normalization,
        "sourceArchiveSha256": digest,
        "cells": int(len(rows)),
        "cellSelectionSha256": mn.array_hash(rows),
        "resolution": resolution,
        "partitionSha256": mn.array_hash(partition),
        "subtypeNames": payload["subtypeNames"],
        "subtypeCells": {
            name: int((subtypes == name).sum()) for name in sorted(set(subtypes)) if name
        },
        "unlabelledCells": int((subtypes == "").sum()),
        "layoutParameters": dict(layout_group.attrs["provenance"]["parameters"]),
        "queries": {},
    }
    model = register_reference(ds, modalities, stages)
    for source in payload["queries"]:
        name = source["name"]
        if source["kind"] == "store":
            query, query_digest = open_store(source, work, name, payload)
            query_design = T_CELLS[source["dataset"]]
        elif source["kind"] == "h5ad":
            query, query_digest = mn.measure(
                stages,
                f"{name}.conversion",
                lambda source=source: open_h5ad_query(source, work, payload),
            )
            query_design = source["tCells"]
        else:
            raise ValueError(f"query {name} has kind {source['kind']!r}, not store or h5ad")
        query_rows = t_cell_rows(query, query_design, source.get("study"))
        projection, coverage = map_query(
            ds,
            modalities,
            model,
            query,
            query_rows,
            stages,
            f"{name}.projection",
            source.get("minFeatureCoverage", MIN_FEATURE_COVERAGE),
        )
        outcome = mn.measure(
            stages,
            f"{name}.transfer",
            lambda query=query, projection=projection: query.mapping.load_label_transfer(
                query.mapping.transfer_labels(
                    projection,
                    reference=model,
                    reference_labels="t_cell_subtype",
                    threshold_fraction=0.0,
                )
            ),
        )
        moved = outcome.labels.fillna("unassigned").to_numpy(dtype=str)
        if len(moved) != len(query_rows) or (moved == "").any():
            raise ValueError(
                f"{name}'s transfer does not give each query cell a named subtype or none"
            )
        authors = column(query, query_design["authorColumn"], query_rows)
        donors = column(query, query_design["donorColumn"], query_rows)
        has_protein = "ADT" in query.assay_names
        query_names, query_clr = (
            protein_values(query, query_rows, normalization) if has_protein else (None, None)
        )
        for kind, found in zip(
            ("proteins", "authors", "donors"),
            cluster_summaries(moved, query_clr, query_names, authors, donors, cellSet=name),
            strict=True,
        ):
            if found:
                summaries[f"{name}_{kind}"] = found
        arrays.update(
            {
                f"{name}_rows": query_rows,
                f"{name}_ids": column(query, "ids", query_rows),
                f"{name}_labels": column(query, query_design["labelColumn"], query_rows),
                f"{name}_authors": authors,
                f"{name}_donors": donors,
                f"{name}_subtypes": moved,
                f"{name}_votes": outcome.evidence.voteFraction.to_numpy(dtype=np.float32),
            }
        )
        if query_design.get("studyColumn"):
            arrays[f"{name}_studies"] = column(query, query_design["studyColumn"], query_rows)
        result["queries"][name] = {
            "kind": source["kind"],
            "sourceSha256": query_digest,
            "cells": int(len(query_rows)),
            "cellSelectionSha256": mn.array_hash(query_rows),
            "featureCoverage": coverage,
            "proteins": None if query_names is None else int(len(query_names)),
            "subtypeCells": {s: int((moved == s).sum()) for s in sorted(set(moved))},
            "meanVote": float(arrays[f"{name}_votes"].mean()),
        }
        del query, query_clr
    base = payload["recordUri"].removesuffix(".json")
    path = work / "t_cell_reference.npz"
    np.savez_compressed(path, **arrays)
    upload_file(path, f"{base}.npz")
    files = {"npz": {"uri": f"{base}.npz", "sha256": sha256(path)}}
    for name, found in summaries.items():
        table = write_csv(f"{name}.csv", found, directory=work)
        upload_file(table, f"{base}_{name}.csv")
        files[name] = {"uri": f"{base}_{name}.csv", "sha256": sha256(Path(table))}
    return {
        **result,
        "files": files,
        "saveK": SAVE_K,
        "umapSettings": mn.UMAP_SETTINGS,
        "stages": stages,
        "settings": mn.SETTINGS,
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
        "seconds": time.perf_counter() - started,
    }
