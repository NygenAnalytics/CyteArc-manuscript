"""Annotate blood stores in place by mapping them onto the reference ladder's whole reference.

Every eligible cell of a CyteBase blood store is labelled by mapping it onto the ladder's
4,292,870-cell reference (the unintegrated CyteArc arm), which stays in object storage.

``build_reference`` rebuilds that reference once from the ladder's prepared H5AD with the
ladder arm's settings. It checks that the reference maps the ladder's held-out query to the
ladder's labels, then registers the mapping reference and saves the store to object storage.

``map_and_aggregate`` handles one study. Each CyteBase store is mounted into a local
writable target, so counts are read from the source store and nothing is copied. Primary
blood cells with a recorded donor are mapped onto the saved reference with the ladder's
neighbor count and vote rule. Raw counts are then summed per sample and per pair of mapped
type and author type in one pass over the store's feature-major counts.
"""

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runtime.common import configure_cytearc, dir_bytes, provenance, write_json
from experiments.runtime.locations import CYTEBASE_RELEASE

WORK = Path("/tmp/blood_store_annotation")
# The ladder arm's reference settings (cross_study_mapping.run); build_reference checks them
# against the ladder record before building.
REFERENCE_SETTINGS = {
    "filtering": False,
    "hvg_count": 2000,
    "pca_dims": 30,
    "neighbors_k": 30,
    "umap": False,
    "leiden": False,
    "cell_cycle": False,
    "paris": False,
    "doublets": False,
    "markers": False,
    "params": {"ann_index": {"ann_parallel": False}},
}
LADDER_ARTIFACTS = ("highly_variable_features", "normalized", "pca", "ann_index", "neighbors")
SAVED_NEIGHBORS = 30
VOTE_THRESHOLD = 0.0
UNASSIGNED = "unassigned"
UNTYPED = "untyped"
BLOOD = "blood"
SOUND_LIFE_CMV_NEGATIVE = [
    "gong_2025_sound_life_young_adult_females_cmv_negative_46104f0b",
    "gong_2025_sound_life_young_adult_males_cmv_negative_420d0f5d",
    "gong_2025_sound_life_older_adult_females_cmv_negative_3cb646ab",
    "gong_2025_sound_life_older_adult_males_cmv_negative_47e00f43",
]
SOUND_LIFE_CMV_POSITIVE = [
    "gong_2025_sound_life_young_adult_females_cmv_positive_0a8e3443",
    "gong_2025_sound_life_young_adult_males_cmv_positive_5fde5c9c",
    "gong_2025_sound_life_older_adult_females_cmv_positive_aa32098f",
    "gong_2025_sound_life_older_adult_males_cmv_positive_0c0c49a2",
]
# One study is one or more CyteBase stores, annotated in one job. A sample is a donor plus the
# listed fields.
STUDIES = {
    "ren": {
        "stores": ["ren_2021_large_scale_single_cell_covid_19_patients_9dbab10c"],
        "sampleColumns": ["sampleID"],
    },
    "van_der_wijst": {
        "stores": ["van_der_wijst_2021_type_i_interferon_patients_covid_19_01ad3cd7"],
        "sampleColumns": ["timepoint"],
    },
    "yoshida": {"stores": ["yoshida_2022_pbmc_2a498ace"], "sampleColumns": ["sample_id"]},
    "perez": {
        "stores": ["perez_2022_multiplexed_scrna_1_2_million_pbmcs_adult_lupus_samples_218acb0f"],
        "sampleColumns": ["sample_uuid"],
    },
    "wells": {
        "stores": ["wells_2025_immune_aging_project_all_cells_1b350d0a"],
        "sampleColumns": [],
    },
    "sound_life": {"stores": SOUND_LIFE_CMV_NEGATIVE, "sampleColumns": ["specimen.specimenGuid"]},
    "sound_life_cmv_positive": {
        "stores": SOUND_LIFE_CMV_POSITIVE,
        "sampleColumns": ["specimen.specimenGuid"],
    },
    "gong_all_cells": {
        "stores": ["gong_2025_all_cells_ff4235bb"],
        "sampleColumns": ["specimen.specimenGuid"],
    },
}


def source_location(identifier: str) -> str:
    if "://" in identifier or identifier.startswith("/"):
        return identifier
    return f"{CYTEBASE_RELEASE}/datasets/{identifier}/data.zarr"


def source_storage_options(identifier: str) -> dict | None:
    if not source_location(identifier).startswith("hf://"):
        return None
    return {"token": os.environ.get("HF_TOKEN") or False, "block_size": 1}


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_directory(objects, path: Path, uri: str) -> int:
    """Copy a local store directory to R2 or to another local directory."""
    if uri.startswith("s3://"):
        return objects._remote().upload_dir(path, uri)
    shutil.copytree(path, uri, dirs_exist_ok=True)
    return dir_bytes(Path(uri))


def ladder_reference_record(objects, uri: str) -> dict:
    record = objects.get_json(uri)
    if record.get("status") != "completed":
        raise ValueError(f"The ladder record {uri} is not completed")
    arm = record["result"]["arms"]
    if set(arm) != {"unintegrated"}:
        raise ValueError(f"The ladder record {uri} is not the unintegrated arm; arms {sorted(arm)}")
    run = arm["unintegrated"]["pipeline"]["run"]
    config = run["config"]
    expected = {
        "hvgCount": REFERENCE_SETTINGS["hvg_count"],
        "pcaDims": REFERENCE_SETTINGS["pca_dims"],
        "neighborsK": REFERENCE_SETTINGS["neighbors_k"],
        "params": REFERENCE_SETTINGS["params"],
        "harmonyBatchColumns": [],
        "umap": False,
    }
    differing = {
        key: config.get(key) for key, value in expected.items() if config.get(key) != value
    }
    if differing:
        raise ValueError(f"The ladder reference used other settings: {differing}")
    return record


def ladder_predictions(objects, record: dict, work: Path) -> pd.DataFrame:
    path = work / "ladder_predictions.csv"
    objects.download(record["result"]["arms"]["unintegrated"]["predictionsUri"], path)
    return pd.read_csv(path, keep_default_na=False, dtype={"cell": str, "prediction": str})


def map_onto(query, model, cells, *, missing_feature_policy: str):
    """The ladder's mapping rule: 30 saved neighbors and a plain majority vote."""
    projection = query.mapping.run(
        model, cells, save_k=SAVED_NEIGHBORS, missing_feature_policy=missing_feature_policy
    )
    transfer = query.mapping.load_label_transfer(
        query.mapping.transfer_labels(
            projection,
            reference=model,
            reference_labels="cell_type",
            threshold_fraction=VOTE_THRESHOLD,
        )
    )
    return projection, transfer


def build_reference(payload: dict) -> dict:
    """Rebuild, check, register and save the ladder's unintegrated whole reference."""
    import cytearc
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    from experiments.runtime.harness import Objects, Sampler
    from experiments.runtime.steps import _Steps

    configure_cytearc()
    objects = Objects(payload.get("r2EndpointUrl"))
    work = Path(payload.get("workDirectory", str(WORK / "reference")))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "24G"), int(payload.get("nthreads", 16))
    part = payload.get("referencePart", "reference_all")
    destination = payload["referenceStoreUri"].rstrip("/")
    if objects.exists(f"{destination}/zarr.json"):
        raise FileExistsError(f"A store already exists at {destination}; choose a new location")
    ladder = ladder_reference_record(objects, payload["ladderRecordUri"])
    if ladder["result"]["settings"].get("referencePart") != part:
        raise ValueError(f"The ladder record maps another part than {part}")
    manifest = objects.get_json(payload["inputManifestUri"])
    sampler = Sampler()
    sampler.start()
    steps = _Steps(on_change=lambda: sampler.set_step(steps.current))
    result = {
        "settings": {key: value for key, value in payload.items() if key != "r2EndpointUrl"},
        "provenance": provenance(),
        "referenceSettings": REFERENCE_SETTINGS,
        "mappingRule": {"savedNeighbors": SAVED_NEIGHBORS, "voteThreshold": VOTE_THRESHOLD},
    }

    def fetch(name: str) -> Path:
        item = manifest["parts"][name]
        path = work / f"{name}.h5ad"
        objects.download(item["uri"], path)
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"The prepared {name} input checksum changed")
        return path

    # The ladder used the default count layout; a smaller one lets a small machine convert
    # without changing any count.
    layout = payload.get("countMatrixPolicy")
    if layout is not None:
        from cytearc.storage.count_matrix import CountMatrixPolicy

        layout = CountMatrixPolicy(**layout)

    def convert(path: Path, store: Path):
        reader = H5adReader(str(path), feature_name_key="feature_name")
        try:
            H5adToZarr(
                reader,
                str(store),
                assay_name="RNA",
                mem_budget=memory,
                nthreads=threads,
                policy=layout,
            ).dump()
        finally:
            reader.close()
        path.unlink()
        return cytearc.DataStore(
            str(store), min_features_per_cell=-1, mem_budget=memory, nthreads=threads
        )

    try:
        source = steps.run("download_reference", lambda: fetch(part))
        store_path = work / "data.zarr"
        reference = steps.run("convert_reference", lambda: convert(source, store_path))
        if reference.cells.N != manifest["parts"][part]["cells"]:
            raise AssertionError("The converted reference lost or gained cells")
        run = steps.run("reference_analysis", lambda: reference.pipeline.run(**REFERENCE_SETTINGS))
        # Artifact identifiers are assigned at creation, so the rebuilt reference is checked
        # by its settings, its input checksum and the labels it gives the ladder's query.
        result["artifacts"] = {key: run[key].to_dict() for key in LADDER_ARTIFACTS}
        mapping_reference = steps.run(
            "registration", lambda: reference.mapping.build_reference(run["neighbors"])
        )
        model = reference.mapping.load_reference(mapping_reference)
        query_path = steps.run("download_query", lambda: fetch("query"))
        query = steps.run("convert_query", lambda: convert(query_path, work / "query.zarr"))
        cells = query.snapshot_cell_selection("I")
        _, transfer = steps.run(
            "map_ladder_query",
            lambda: map_onto(query, model, cells, missing_feature_policy="error"),
        )
        expected = ladder_predictions(objects, ladder, work)
        ids = np.asarray(query.cells.fetch_all("ids"), dtype=str)[transfer.cell_idx]
        rebuilt = pd.DataFrame(
            {"cell": ids, "rebuilt": transfer.labels.fillna(UNASSIGNED).to_numpy(dtype=str)}
        ).merge(expected[["cell", "prediction"]], on="cell", how="outer", validate="one_to_one")
        if rebuilt.isna().any().any():
            raise AssertionError("The rebuilt and ladder query predictions cover other cells")
        result["ladderQueryAgreement"] = {
            "cells": len(rebuilt),
            "identicalLabels": int((rebuilt.rebuilt == rebuilt.prediction).sum()),
            "fraction": float((rebuilt.rebuilt == rebuilt.prediction).mean()),
        }
        minimum = float(payload.get("minimumLadderAgreement", 1.0))
        if result["ladderQueryAgreement"]["fraction"] < minimum:
            raise AssertionError(
                f"The rebuilt reference reproduces {result['ladderQueryAgreement']['fraction']:.6f} "
                f"of the ladder's query labels; at least {minimum} is required"
            )
        feature_count = int(len(model.feature_ids))
        query = model = None
        shutil.rmtree(work / "query.zarr")
        result["referenceStoreBytes"] = dir_bytes(store_path)
        result["savedBytes"] = steps.run(
            "save_reference", lambda: save_directory(objects, store_path, destination)
        )
        vocabulary = sorted(set(np.asarray(reference.cells.fetch_all("cell_type"), dtype=str)))
        result["registration"] = {
            "storeUri": destination,
            "mappingReference": mapping_reference.to_dict(),
            "neighbors": run["neighbors"].to_dict(),
            "referencePart": part,
            "cells": int(reference.cells.N),
            "vocabulary": vocabulary,
            "featureCount": feature_count,
            "ladderRecordUri": payload["ladderRecordUri"],
        }
        result["pipeline"] = run.report()
    finally:
        sampler.stop()
        result["steps"] = {
            name: {**record, **sampler.step_peaks.get(name, {})}
            for name, record in steps.records.items()
        }
        result["peakContainerBytes"] = sampler.peak_cgroup
        result["peakProcessTreeRssBytes"] = sampler.peak_rss
        write_json("reference.json", result, directory=work)
    return json.loads(json.dumps(result, default=str))


def open_reference(objects, record_uri: str, probe, memory: str, threads: int):
    """Open the registered reference read-only and load its mapping reference."""
    import cytearc
    from cytearc.storage.refs import ArtifactRef
    from cytearc.storage.stores import make_store

    from experiments.runtime.object_io import storage_options
    from experiments.runtime.recording_store import RecordingStoreWrapper

    record = objects.get_json(record_uri)
    if record.get("status", "completed") != "completed":
        raise ValueError(f"The reference record {record_uri} is not completed")
    registration = (record.get("result") or record)["registration"]
    location = registration["storeUri"]
    store = make_store(location, storage_options=storage_options(location), read_only=True)
    if isinstance(store, str):
        from zarr.storage import LocalStore

        store = LocalStore(store, read_only=True)
    reference = cytearc.DataStore(
        RecordingStoreWrapper(store, probe=probe),
        zarr_mode="r",
        min_features_per_cell=-1,
        mem_budget=memory,
        nthreads=threads,
    )
    model = reference.mapping.load_reference(
        ArtifactRef.from_dict(registration["mappingReference"])
    )
    return reference, model, registration


def eligible_cells(
    store, study: str, cells_per_store: int | None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Rows of primary blood cells with a recorded donor, and their sample fields."""
    from experiments.runtime.annotations import MISSING_DONORS

    columns = ["donor_id", *STUDIES[study]["sampleColumns"]]
    primary = np.asarray(store.cells.fetch_all("is_primary_data"))
    if primary.dtype.kind != "b":
        raise TypeError("is_primary_data must be stored as a boolean column")
    tissue = np.asarray(store.cells.fetch_all("tissue"), dtype=str)
    values = {column: np.asarray(store.cells.fetch_all(column)).astype(str) for column in columns}
    known = ~np.isin(np.char.lower(np.char.strip(values["donor_id"])), list(MISSING_DONORS))
    rows = np.flatnonzero(primary & (tissue == BLOOD) & known)
    if cells_per_store and len(rows) > cells_per_store:
        rng = np.random.default_rng(4444)
        rows = np.sort(rng.choice(rows, cells_per_store, replace=False))
    if not len(rows):
        raise ValueError(f"No primary blood cells with a donor in {study}")
    return rows, pd.DataFrame({column: values[column][rows] for column in columns})


def encode(values) -> tuple[np.ndarray, np.ndarray]:
    names, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return codes.astype(np.int32), names


def labelled_column(n_cells: int, rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
    column = np.full(n_cells, "", dtype=object)
    column[rows] = labels
    return column


def count_array_geometry(assay) -> dict:
    """Stored size and chunk grid of the feature-major counts that pseudobulk sums read."""
    array = assay.rawDataT
    stored = None
    try:
        stored = int(array.nbytes_stored())
    except (NotImplementedError, TypeError, ValueError) as error:
        stored = f"unavailable: {type(error).__name__}"
    return {
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "shards": list(array.shards) if getattr(array, "shards", None) else None,
        "storedBytes": stored,
    }


def donor_sample_pseudobulk(query, cells, rows, sample_codes, type_codes, path: Path) -> dict:
    """Sum raw counts per sample and cell-type pair in one pass over the counts."""
    n_cells = int(query.cells.N)
    query.cells.insert(
        "pseudobulk_sample",
        labelled_column(n_cells, rows, np.char.add("s", sample_codes.astype(str))),
        overwrite=True,
    )
    query.cells.insert(
        "pseudobulk_cell_types",
        labelled_column(n_cells, rows, np.char.add("c", type_codes.astype(str))),
        overwrite=True,
    )
    bulk, counts = query.compare.compute_bulk(
        "pseudobulk_sample",
        from_assay="RNA",
        cell_selection=cells,
        secondary_groups="pseudobulk_cell_types",
        aggr_type="sum",
        feature_label="id",
        remove_empty_features=False,
        return_n_cells=True,
    )
    matrix = bulk.to_numpy().T
    if matrix.min() < 0:
        raise AssertionError("Raw-count pseudobulk sums must be non-negative")
    if matrix.max() >= 2**32:
        raise AssertionError("A pseudobulk sum exceeds the uint32 range")
    labels = np.asarray(bulk.columns, dtype=str)
    split = np.char.partition(labels, "_")
    group_samples = np.char.lstrip(split[:, 0], "s").astype(np.int32)
    group_types = np.char.lstrip(split[:, 2], "c").astype(np.int32)
    group_cells = counts.reindex(bulk.columns).to_numpy(dtype=np.int64)
    expected = pd.DataFrame({"sample": sample_codes, "pair": type_codes}).value_counts()
    observed = pd.Series(
        group_cells,
        index=pd.MultiIndex.from_arrays([group_samples, group_types], names=["sample", "pair"]),
    )
    if not observed.sort_index().equals(expected.sort_index().astype(np.int64).rename(None)):
        raise AssertionError("Pseudobulk cell counts differ from the per-cell labels")
    feature_ids = np.asarray(query.RNA.feats.fetch_all("ids"), dtype=str)
    feature_names = np.asarray(query.RNA.feats.fetch_all("names"), dtype=str)
    order = {value: index for index, value in enumerate(feature_ids)}
    np.savez_compressed(
        path,
        counts=matrix.astype(np.uint32),
        sample_codes=group_samples,
        cell_type_codes=group_types,
        cells=group_cells,
        feature_ids=np.asarray(bulk.index, dtype=str),
        feature_names=feature_names[[order[value] for value in bulk.index.astype(str)]],
    )
    return {
        "groups": len(labels),
        "features": int(matrix.shape[1]),
        "cells": int(group_cells.sum()),
        "bytes": path.stat().st_size,
    }


def measure(operation, probe) -> tuple[object, dict]:
    """Run one operation with its seconds, the store reads it made and its sampled memory."""
    from experiments.runtime.metrics import ResourceSampler
    from experiments.runtime.recording_store import probe_summary

    probe.reset()
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    try:
        value = operation()
    finally:
        resources = asdict(sampler.stop())
    return value, {
        "elapsedSeconds": time.perf_counter() - started,
        "io": probe_summary(probe),
        "resources": resources,
    }


def map_store(identifier: str, study: str, model, payload: dict, work: Path, probe) -> dict:
    """Map one store's eligible cells and sum its raw counts per sample and cell-type pair."""
    from cytearc.datastore.datastore import mount_datastore

    from experiments.runtime.annotations import ladder_cell_type

    memory, threads = payload.get("memBudget", "24G"), int(payload.get("nthreads", 16))
    folder = work / identifier
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "mount.zarr"
    if target.exists():
        shutil.rmtree(target)
    summary = {"store": identifier}
    query, summary["mount"] = measure(
        lambda: mount_datastore(
            source_location(identifier),
            at=str(target),
            storage_options=source_storage_options(identifier),
            mem_budget=memory,
            nthreads=threads,
            min_features_per_cell=-1,
        ),
        probe,
    )
    summary["storeCells"] = int(query.cells.N)
    summary["storeGenes"] = int(query.RNA.feats.N)
    rows, samples = eligible_cells(query, study, payload.get("cellsPerStore"))
    summary["eligibleCells"] = len(rows)
    feature_ids = set(np.asarray(query.RNA.feats.fetch_all("ids"), dtype=str))
    summary["referenceFeaturesMissing"] = sorted(set(model.feature_ids) - feature_ids)
    selected = np.zeros(query.cells.N, dtype=bool)
    selected[rows] = True
    query.cells.insert("annotated_cells", selected, overwrite=True)
    cells = query.snapshot_cell_selection("annotated_cells")
    policy = payload.get("missingFeaturePolicy", "reference_mean")
    (_, transfer), summary["mapping"] = measure(
        lambda: map_onto(query, model, cells, missing_feature_policy=policy), probe
    )
    if not np.array_equal(np.asarray(transfer.cell_idx), rows):
        raise AssertionError("Transferred labels do not follow the eligible cell rows")
    mapped = transfer.labels.fillna(UNASSIGNED).to_numpy(dtype=str)
    evidence = transfer.evidence
    summary["abstentions"] = (
        evidence["abstentionReason"].fillna("assigned").value_counts().to_dict()
    )
    author = np.asarray(query.cells.fetch_all("cell_type"), dtype=str)[rows]
    author_codes, author_names = encode(author)
    coarse_names = np.array([ladder_cell_type(name) or UNTYPED for name in author_names])
    pair = np.char.add(np.char.add(mapped, "|"), coarse_names[author_codes])
    type_codes, type_names = encode(pair)
    sample_keys = samples["donor_id"].to_numpy(dtype=str)
    for column in samples.columns[1:]:
        sample_keys = np.char.add(
            np.char.add(sample_keys, "|"), samples[column].to_numpy(dtype=str)
        )
    sample_codes, sample_names = encode(sample_keys)
    table = (
        samples.assign(sample_code=sample_codes)
        .drop_duplicates("sample_code")
        .sort_values("sample_code")
        .reset_index(drop=True)
    )
    table["sample_key"] = sample_names[table.sample_code]
    table["cells"] = np.bincount(sample_codes)[table.sample_code]
    table.insert(0, "store", identifier)
    table.to_csv(folder / "samples.csv", index=False)
    pairs = pd.Series(type_names).str.split("|", expand=True)
    pd.DataFrame(
        {
            "cell_type_code": np.arange(len(type_names)),
            "mapped_type": pairs[0],
            "author_type": pairs[1],
        }
    ).to_csv(folder / "cell_type_pairs.csv", index=False)
    labels = pd.DataFrame({"sample_code": sample_codes, "author_label": author})
    labels.value_counts().rename("cells").reset_index().to_csv(
        folder / "author_label_counts.csv", index=False
    )
    mapped_codes, mapped_names = encode(mapped)
    np.savez_compressed(
        folder / "cells.npz",
        rows=rows.astype(np.int64),
        sample_codes=sample_codes,
        mapped_codes=mapped_codes.astype(np.int8),
        mapped_names=mapped_names,
        vote_fraction=evidence["voteFraction"].to_numpy(dtype=np.float32),
        author_codes=author_codes,
        author_names=author_names,
        author_type_names=coarse_names,
    )
    summary["countGeometry"] = count_array_geometry(query.RNA)
    summary["pseudobulk"], summary["pseudobulkMeasurement"] = measure(
        lambda: donor_sample_pseudobulk(
            query, cells, rows, sample_codes, type_codes, folder / "pseudobulk.npz"
        ),
        probe,
    )
    summary["samples"] = len(table)
    summary["donors"] = int(samples.donor_id.nunique())
    summary["cellTypePairs"] = len(type_names)
    query = None
    shutil.rmtree(target)
    return summary


def map_and_aggregate(payload: dict) -> dict:
    """Map one study's stores onto the registered reference and sum their counts."""
    from experiments.runtime.harness import Objects
    from experiments.runtime.metrics import ResourceSampler
    from experiments.runtime.recording_store import NetworkProbe, StoreProbe, probe_remote_reads

    configure_cytearc()
    study = payload["study"]
    if study not in STUDIES:
        raise ValueError(f"Unknown study {study!r}; choose from {sorted(STUDIES)}")
    stores = payload.get("stores", STUDIES[study]["stores"])
    if not stores or len(stores) != len(set(stores)):
        raise ValueError("stores must name each source store once")
    objects = Objects(payload.get("r2EndpointUrl"))
    work = Path(payload.get("workDirectory", str(WORK / study)))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "24G"), int(payload.get("nthreads", 16))
    prefix = payload["outputPrefix"].rstrip("/")
    result = {
        "settings": {key: value for key, value in payload.items() if key != "r2EndpointUrl"},
        "provenance": provenance(),
        "mappingRule": {
            "savedNeighbors": SAVED_NEIGHBORS,
            "voteThreshold": VOTE_THRESHOLD,
            "missingFeaturePolicy": payload.get("missingFeaturePolicy", "reference_mean"),
        },
        "eligibility": "Primary cells (is_primary_data) with tissue 'blood' and a recorded donor",
        "sampleKey": ["donor_id", *STUDIES[study]["sampleColumns"]],
        "timingBoundary": "Each store: mount, mapping with label transfer, and one pseudobulk pass; "
        "reference opening and uploads are timed separately",
    }
    sampler = ResourceSampler().start()
    reference_probe = StoreProbe()
    probe = NetworkProbe()
    started = time.perf_counter()
    reference, model, registration = open_reference(
        objects, payload["referenceRecordUri"], reference_probe, memory, threads
    )
    result["referenceOpening"] = {
        "seconds": time.perf_counter() - started,
        "io": reference_probe.to_json(),
        "registration": registration,
    }
    result["stores"] = []
    with probe_remote_reads(probe):
        for identifier in stores:
            summary = map_store(identifier, study, model, payload, work, probe)
            for name in (
                "samples.csv",
                "cell_type_pairs.csv",
                "author_label_counts.csv",
                "cells.npz",
                "pseudobulk.npz",
            ):
                uri = f"{prefix}/{identifier}/{name}"
                objects.upload(work / identifier / name, uri)
                summary.setdefault("outputs", {})[name] = uri
            result["stores"].append(summary)
    result["referenceReadsDuringMapping"] = reference_probe.to_json()
    result["resources"] = asdict(sampler.stop())
    result = json.loads(json.dumps(result, default=str))
    write_json(f"{study}.json", result, directory=work)
    return result
