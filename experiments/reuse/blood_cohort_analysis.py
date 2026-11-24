"""Jointly analyse a ten-study blood cohort assembled as one virtual composite.

The composite indexes the primary cells of ten CyteBase stores without copying their
counts. An unintegrated analysis is followed by a Harmony2 analysis that reuses its
upstream artifacts. The job saves the integrated embedding and clusters, study mixing
before and after integration, and donor-by-cell-type raw-count pseudobulk profiles for
donor-level comparisons within each study.
"""

import json
from dataclasses import asdict
from pathlib import Path

import cytearc
import numpy as np
from cytearc.composite import create_composite

from experiments.reuse.composite_and_remote import (
    HVG_BLACKLIST,
    SOURCE_STORAGE_OPTIONS,
    measure,
    neighbor_label_metrics,
    stage_summary,
)
from experiments.runtime.annotations import MISSING_DONORS, blood_cell_type
from experiments.runtime.common import configure_cytearc, dir_bytes, write_json
from experiments.runtime.locations import CYTEBASE_RELEASE
from experiments.runtime.metrics import ResourceSampler
from experiments.runtime.recording_store import NetworkProbe, probe_remote_reads

WORK = Path("/tmp/blood_cohort")
SNAPSHOT_COLUMNS = ["dataset", "orig_disease", "orig_donor_id", "orig_cell_type"]


def source_location(identifier: str) -> str:
    if "://" in identifier or identifier.startswith("/"):
        return identifier
    return f"{CYTEBASE_RELEASE}/datasets/{identifier}/data.zarr"


def open_sources(identifiers: dict, memory: str, threads: int) -> dict:
    locations = [source_location(value).rstrip("/") for value in identifiers.values()]
    if len(locations) != len(set(locations)):
        raise ValueError("The same source store was included more than once")
    return {
        name: cytearc.DataStore(
            source_location(identifier),
            zarr_mode="r",
            storage_options=None if identifier.startswith("/") else SOURCE_STORAGE_OPTIONS,
            min_features_per_cell=-1,
            mem_budget=memory,
            nthreads=threads,
        )
        for name, identifier in identifiers.items()
    }


def primary_rows(sources: dict, cells_per_source: int | None) -> dict:
    rows = {}
    for name, source in sources.items():
        selected = np.flatnonzero(np.asarray(source.cells.fetch_all("is_primary_data"), dtype=bool))
        if cells_per_source:
            rng = np.random.default_rng(4444)
            selected = np.sort(
                rng.choice(selected, min(len(selected), cells_per_source), replace=False)
            )
        if not len(selected):
            raise ValueError(f"No primary cells in {name}")
        rows[name] = selected
    return rows


def encode(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    dtype = np.int16 if len(names) < 2**15 else np.int32
    return codes.astype(dtype), names.tolist()


def annotate_cells(joint) -> dict:
    """Add study batch, study-scoped donor and harmonized cell-type columns."""
    datasets = np.asarray(joint.cells.fetch_all("dataset"), dtype=str)
    donors = np.asarray(joint.cells.fetch_all("orig_donor_id"), dtype=str)
    labels = np.asarray(joint.cells.fetch_all("orig_cell_type"), dtype=str)
    known = ~np.isin(np.char.lower(np.char.strip(donors)), list(MISSING_DONORS))
    donor_groups = np.where(known, np.char.add(np.char.add(datasets, "__"), donors), "")
    label_names, label_codes = np.unique(labels, return_inverse=True)
    label_map = {name: blood_cell_type(name) for name in label_names}
    harmonized = np.array([label_map[name] for name in label_names], dtype=object)[label_codes]
    joint.cells.insert("integration_batch", datasets, overwrite=True)
    joint.cells.insert("donor_group", donor_groups.astype(object), overwrite=True)
    joint.cells.insert("harmonized_cell_type", harmonized, overwrite=True)
    return {
        "cellTypeHarmonization": label_map,
        "cellsWithoutDonor": int((~known).sum()),
        "cellsWithoutHarmonizedType": int(np.sum(harmonized == "")),
        "studyScopedDonors": int(np.unique(donor_groups[known]).size),
    }


def analysis_settings(payload: dict, *, integrated: bool) -> dict:
    settings = {
        "assay": "RNA",
        "filtering": False,
        "hvg_count": payload.get("hvgCount", 2000),
        "pca_dims": payload.get("pcaDims", 30),
        "neighbors_k": payload.get("neighborsK", 30),
        "umap": integrated,
        "cell_cycle": False,
        "paris": False,
        "doublets": False,
        "markers": False,
        "snapshot_columns": SNAPSHOT_COLUMNS,
        "params": {
            "hvg": {"blacklist": HVG_BLACKLIST},
            "ann_index": {"ann_parallel": False, "rand_state": 4444},
            "leiden": {"backend": "cytearc", "partitions": [1.0], "selected": 1.0},
            "species": "homo_sapiens",
        },
    }
    if integrated:
        settings["harmony_batch_columns"] = ["integration_batch"]
        settings["params"]["umap"] = {"parallel": True, "n_epochs": 300, "random_seed": 4444}
    return settings


def array_values(group) -> np.ndarray:
    key = "data" if "data" in group else "values"
    return np.asarray(group[key][:])


def mixing(joint, run) -> dict:
    neighbors = np.asarray(joint.artifacts.load(run["neighbors"])["indices"][:])
    return neighbor_label_metrics(
        np.asarray(joint.cells.fetch_all("harmonized_cell_type"), dtype=str),
        np.asarray(joint.cells.fetch_all("dataset"), dtype=str),
        neighbors,
    )


def cross_table(left: np.ndarray, right: np.ndarray) -> dict:
    left_names, left_codes = np.unique(left, return_inverse=True)
    right_names, right_codes = np.unique(right, return_inverse=True)
    table = np.zeros((len(left_names), len(right_names)), dtype=np.int64)
    np.add.at(table, (left_codes, right_codes), 1)
    return {
        "rows": left_names.tolist(),
        "columns": right_names.tolist(),
        "counts": table.tolist(),
    }


def save_cells(joint, run, path: Path) -> dict:
    """Save the integrated embedding, clusters and per-cell labels as compact codes."""
    umap = array_values(joint.artifacts.load(run["umap"])).astype(np.float32)
    clusters = array_values(joint.artifacts.load(run["clusters"])).astype(np.int32)
    if umap.shape[0] != joint.cells.N or clusters.shape[0] != joint.cells.N:
        raise AssertionError("Embedding or cluster rows differ from the composite's cells")
    columns = {}
    names = {}
    for column in (
        "dataset",
        "donor_group",
        "orig_disease",
        "orig_cell_type",
        "harmonized_cell_type",
    ):
        columns[column], names[column] = encode(joint.cells.fetch_all(column))
    np.savez_compressed(
        path,
        umap=umap,
        clusters=clusters,
        **{f"{column}_codes": codes for column, codes in columns.items()},
        **{f"{column}_names": np.asarray(values, dtype=str) for column, values in names.items()},
    )
    datasets = np.asarray(names["dataset"])[columns["dataset"]]
    types = np.asarray(names["harmonized_cell_type"])[columns["harmonized_cell_type"]]
    return {
        "cells": int(joint.cells.N),
        "leidenClusters": int(np.unique(clusters).size),
        "clusterByStudy": cross_table(clusters.astype(str), datasets),
        "clusterByHarmonizedType": cross_table(clusters.astype(str), types),
        "bytes": path.stat().st_size,
    }


def donor_pseudobulk(joint, path: Path) -> dict:
    """Sum raw counts per study-scoped donor and harmonized cell type."""
    bulk, cells = joint.compare.compute_bulk(
        "donor_group",
        from_assay="RNA",
        secondary_groups="harmonized_cell_type",
        aggr_type="sum",
        feature_label="id",
        remove_empty_features=False,
        return_n_cells=True,
    )
    matrix = bulk.to_numpy().T
    if matrix.min() < 0:
        raise AssertionError("Raw-count pseudobulk sums must be non-negative")
    dtype = np.uint32 if matrix.max() < 2**32 else np.uint64
    feature_ids = np.asarray(joint.RNA.feats.fetch_all("ids"), dtype=str)
    feature_names = np.asarray(joint.RNA.feats.fetch_all("names"), dtype=str)
    order = {value: index for index, value in enumerate(feature_ids)}
    names = feature_names[[order[value] for value in bulk.index.astype(str)]]
    groups = np.asarray(bulk.columns, dtype=str)
    np.savez_compressed(
        path,
        counts=matrix.astype(dtype),
        groups=groups,
        cells=cells.reindex(bulk.columns).to_numpy(dtype=np.int64),
        feature_ids=np.asarray(bulk.index, dtype=str),
        feature_names=np.asarray(names, dtype=str),
    )
    return {
        "groups": len(groups),
        "features": int(matrix.shape[1]),
        "columnLabel": "f'{donor_group}_{harmonized_cell_type}'",
        "dtype": str(np.dtype(dtype)),
        "bytes": path.stat().st_size,
    }


def donor_table(joint) -> list[dict]:
    """Cells per study-scoped donor and recorded condition."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "donorGroup": np.asarray(joint.cells.fetch_all("donor_group"), dtype=str),
            "dataset": np.asarray(joint.cells.fetch_all("dataset"), dtype=str),
            "condition": np.asarray(joint.cells.fetch_all("orig_disease"), dtype=str),
        }
    )
    frame = frame[frame.donorGroup != ""]
    counts = frame.groupby(["donorGroup", "dataset", "condition"]).size().rename("cells")
    return counts.reset_index().to_dict(orient="records")


def run(payload: dict) -> dict:
    from experiments.runtime.harness import R2

    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "24G"), payload.get("nthreads", 16)
    probe = NetworkProbe()
    result = {
        "settings": {key: value for key, value in payload.items() if key != "r2EndpointUrl"},
        "timingBoundary": "Composite creation, an unintegrated analysis through native Leiden, then a Harmony2 analysis reusing its upstream artifacts with UMAP and native Leiden; pseudobulk sums are timed separately",
    }
    with probe_remote_reads(probe):
        sampler = ResourceSampler().start()
        sources = open_sources(payload["sources"], memory, threads)
        rows = primary_rows(sources, payload.get("cellsPerSource"))
        result["sourceOpening"] = {"io": dict(probe.http), "resources": asdict(sampler.stop())}
        result["sourceCellCounts"] = {name: len(value) for name, value in rows.items()}
        joint, result["creation"] = measure(
            lambda: create_composite(
                sources,
                at=work / "composite.zarr",
                assay="RNA",
                features="intersection",
                rows=rows,
                mem_budget=memory,
                nthreads=threads,
            ),
            probe,
        )
        result["cells"] = int(joint.cells.N)
        result["genesShared"] = int(joint.RNA.feats.N)
        result["localStoreAfterCreateBytes"] = dir_bytes(work / "composite.zarr")
        result["annotation"] = annotate_cells(joint)
        unintegrated, result["unintegrated"] = measure(
            lambda: joint.pipeline.run(**analysis_settings(payload, integrated=False)), probe
        )
        result["unintegrated"]["summary"] = stage_summary(unintegrated)
        result["unintegrated"]["mixing"] = mixing(joint, unintegrated)
        integrated, result["integrated"] = measure(
            lambda: joint.pipeline.run(**analysis_settings(payload, integrated=True)), probe
        )
        result["integrated"]["summary"] = stage_summary(integrated)
        result["integrated"]["mixing"] = mixing(joint, integrated)
        result["sharedUpstreamArtifacts"] = {
            key: unintegrated[key].artifact_id == integrated[key].artifact_id
            for key in ("analysis_cell_selection", "highly_variable_features", "normalized", "pca")
        }
        if not all(result["sharedUpstreamArtifacts"].values()):
            raise AssertionError("The Harmony2 analysis did not reuse the upstream artifacts")
        result["localStoreAfterAnalysisBytes"] = dir_bytes(work / "composite.zarr")
        cells_path = work / "cells.npz"
        result["cellOutputs"], result["cellOutputMeasurement"] = measure(
            lambda: save_cells(joint, integrated, cells_path), probe
        )
        bulk_path = work / "donor_pseudobulk.npz"
        result["pseudobulk"], result["pseudobulkMeasurement"] = measure(
            lambda: donor_pseudobulk(joint, bulk_path), probe
        )
        result["donors"] = donor_table(joint)
    prefix = payload.get("outputPrefix")
    if prefix:
        storage = R2(payload["r2EndpointUrl"])
        for name, path in (("cellsUri", cells_path), ("pseudobulkUri", bulk_path)):
            uri = f"{prefix.rstrip('/')}/{path.name}"
            storage.upload(path, uri)
            result[name] = uri
    result = json.loads(json.dumps(result, default=str))
    write_json("blood_cohort_analysis.json", result, directory=work)
    return result
