"""Analyse every human blood store in CyteBase as one virtual composite.

``inventory`` reads each human blood store's cell and gene metadata, saves every cell column of the selected cells
(primary cells with a blood tissue label) as one Parquet file per store, and finds the genes the stores share by
exact ID. ``analyse`` builds the composite of the selected cells over the shared genes, runs the unintegrated
analysis with UMAP and saves the store. ``integrate`` reopens the saved store and adds a Harmony2 analysis on study
that reuses the unintegrated run's upstream artifacts. ``export_inputs`` reopens it and writes its cells as one H5AD
file per store, the comparators' inputs. Each job records its time, memory, local disk and Hugging Face traffic.
"""

import json
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.catalogue.catalogue_screen import (
    BLOOD_TISSUES,
    CATALOGUE_VERSION,
    STORE_SETS,
    HubTraffic,
    eligible_stores,
    hub_traffic,
    open_store,
)
from experiments.reuse.blood_cohort_analysis import (
    analysis_settings,
    array_values,
    cross_table,
    encode,
)
from experiments.reuse.composite_and_remote import measure, stage_summary
from experiments.runtime.annotations import MISSING_DONORS, blood_cell_type
from experiments.runtime.blood_samples import SAMPLE_RULES, rule_columns, sample_selection
from experiments.runtime.common import configure_cytearc, dir_bytes, write_json
from experiments.runtime.metrics import ResourceSampler
from experiments.runtime.recording_store import NetworkProbe, probe_remote_reads

WORK = Path("/tmp/blood_corpus")
STORE = "composite.zarr"
# Stages of the unintegrated run that the Harmony2 run must reuse rather than recompute.
UPSTREAM = ("analysis_cell_selection", "highly_variable_features", "normalized", "pca")


def hub_token(payload: dict) -> str | bool:
    token = os.environ.get("HF_TOKEN") or False
    if payload.get("requireToken") and not token:
        raise RuntimeError("requireToken is set but HF_TOKEN is not in the environment")
    return token


def jsonable(result: dict) -> dict:
    return json.loads(json.dumps(result, default=str))


def disk_used(path: Path) -> int:
    return shutil.disk_usage(path).used


def storage(payload: dict):
    from experiments.runtime.harness import R2

    return R2(payload["r2EndpointUrl"])


def checkpoint(payload: dict, result: dict, phase: str) -> None:
    """Save the measurements so far beside the run record, so a late failure keeps the phases before it."""
    if payload.get("recordUri"):
        storage(payload).put_json(
            payload["recordUri"].removesuffix(".json") + "/partial.json",
            jsonable({**result, "lastPhase": phase}),
        )


# ---- inventory ----------------------------------------------------------------------------------


def blood_stores(catalog, store_ids: list[str] | None) -> list[dict]:
    """Every human blood store of the release, largest first, or only ``store_ids`` of them."""
    rows = [row for store_set in STORE_SETS for row in eligible_stores(catalog, store_set)]
    rows.sort(key=lambda row: -row["cell_count"])
    if store_ids is None:
        return rows
    unknown = sorted(set(store_ids) - {row["cytebase_id"] for row in rows})
    if unknown:
        raise ValueError(f"Not human blood stores of release {CATALOGUE_VERSION}: {unknown}")
    return [row for row in rows if row["cytebase_id"] in store_ids]


def id_format(feature_ids: np.ndarray) -> dict:
    """How a store writes its gene IDs: bare Ensembl IDs, with a version suffix, inside a longer string, or not."""
    ids = pd.Series(np.asarray(feature_ids, dtype=str))
    plain = ids.str.fullmatch(r"ENSG\d{11}")
    versioned = ids.str.fullmatch(r"ENSG\d{11}\.\d+")
    embedded = ids.str.contains(r"ENSG\d{11}") & ~plain & ~versioned
    return {
        "features": len(ids),
        "plainEnsembl": int(plain.sum()),
        "versionedEnsembl": int(versioned.sum()),
        "embeddedEnsembl": int(embedded.sum()),
        "other": int((~(plain | versioned | embedded)).sum()),
    }


def blood_selection(ds) -> np.ndarray:
    primary, tissue = ds.cells.fetch_all_columns(("is_primary_data", "tissue"))
    return np.flatnonzero(
        np.asarray(primary, dtype=bool) & np.isin(np.asarray(tissue, dtype=str), BLOOD_TISSUES)
    )


def cell_table(ds, rows: np.ndarray) -> pd.DataFrame:
    """Every cell column of ``rows``, one column at a time; text columns become categories."""
    columns = {"source_row": rows}
    for column in ds.cells.columns:
        values = np.asarray(ds.cells.fetch_all(column))[rows]
        columns[column] = (
            values if values.dtype.kind in "biuf" else pd.Categorical(values.astype(str))
        )
    return pd.DataFrame(columns)


def inventory_store(row: dict, traffic: HubTraffic, token, payload: dict, work: Path) -> dict:
    started = time.perf_counter()
    ds, opened = open_store(row, traffic, token, payload)
    rows = blood_selection(ds)
    table = cell_table(ds, rows)
    path = work / "cells" / f"{row['cytebase_id']}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    feature_ids = np.asarray(ds.RNA.feats.fetch_all("ids"), dtype=str)
    if len(np.unique(feature_ids)) != len(feature_ids):
        raise ValueError(f"{row['cytebase_id']} repeats gene IDs")
    donors = table["donor_id"].astype(str) if "donor_id" in table else pd.Series([], dtype=str)
    return {
        "store": {
            "storeId": row["cytebase_id"],
            "collectionId": row["collection_id"],
            "author": row["first_author"],
            "year": row["year"],
            "zarrUri": row["zarr_uri"],
            "assays": list(row["assay_labels"]),
            "cells": int(ds.cells.N),
            "selectedCells": len(rows),
            "donors": int(donors[donors != ""].nunique()),
            "cellColumns": [str(column) for column in ds.cells.columns],
            "geneIds": id_format(feature_ids),
            "cellTableBytes": path.stat().st_size,
            "seconds": time.perf_counter() - started,
            **opened,
        },
        "featureIds": feature_ids,
        "cellTable": path,
    }


def with_retries(row: dict, traffic: HubTraffic, token, payload: dict, work: Path) -> dict:
    errors = []
    for attempt in range(payload.get("attempts", 3)):
        try:
            return inventory_store(row, traffic, token, payload, work)
        except Exception as error:  # noqa: BLE001 - every attempt's failure is recorded and the last one raised
            errors.append(f"{type(error).__name__}: {error}")
            time.sleep(10 * (attempt + 1))
    return {"failure": {"storeId": row["cytebase_id"], "errors": errors}}


def gene_sharing(features: dict[str, np.ndarray], stores: list[dict], payload: dict) -> dict:
    """The genes all kept stores share by exact ID. A store is left out when fewer than ``minimumPlainShare`` of its
    IDs are bare Ensembl IDs or when it has fewer than ``minimumGenes`` genes, since exact-ID alignment would then
    collapse the shared set."""
    minimum_genes = payload.get("minimumGenes", 10000)
    minimum_plain = payload.get("minimumPlainShare", 0.9)
    excluded = {}
    for store in stores:
        ids = store["geneIds"]
        if ids["features"] < minimum_genes:
            excluded[store["storeId"]] = f"{ids['features']} genes, fewer than {minimum_genes}"
        elif ids["plainEnsembl"] < minimum_plain * ids["features"]:
            excluded[store["storeId"]] = (
                f"{ids['plainEnsembl']} of {ids['features']} IDs are bare Ensembl IDs"
            )
    kept = [
        store["storeId"]
        for store in stores
        if store["storeId"] not in excluded and store["selectedCells"]
    ]
    sets = {store: set(features[store]) for store in kept}
    shared = set.intersection(*sets.values())
    lost = {}
    for store in kept:
        others = [sets[other] for other in kept if other != store]
        lost[store] = len(set.intersection(*others)) - len(shared) if others else 0
    return {
        "rule": {"minimumGenes": minimum_genes, "minimumPlainShare": minimum_plain},
        "excluded": excluded,
        "noSelectedCells": [s["storeId"] for s in stores if not s["selectedCells"]],
        "kept": kept,
        "sharedGenes": len(shared),
        "sharedGeneIds": sorted(shared),
        "genesLostByStore": dict(sorted(lost.items(), key=lambda item: -item[1])),
    }


def inventory(payload: dict) -> dict:
    import cytebase

    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    token = hub_token(payload)
    traffic = HubTraffic(payload.get("quotaReserve", 200))
    sampler = ResourceSampler().start()
    started = time.perf_counter()
    with hub_traffic(traffic):
        catalog = cytebase.Catalog(version=CATALOGUE_VERSION, token=token)
        rows = blood_stores(catalog, payload.get("stores"))
        stores, failures, features, tables = [], [], {}, {}
        with ThreadPoolExecutor(max_workers=payload.get("storeConcurrency", 4)) as pool:
            futures = [
                pool.submit(with_retries, row, traffic, token, payload, work) for row in rows
            ]
            for future in as_completed(futures):
                result = future.result()
                if "failure" in result:
                    failures.append(result["failure"])
                    continue
                store = result["store"]
                stores.append(store)
                features[store["storeId"]] = result["featureIds"]
                tables[store["storeId"]] = result["cellTable"]
        for store in [*stores, *failures]:
            store["io"] = traffic.store_summary(store["storeId"])
    if failures:
        raise RuntimeError(f"{len(failures)} stores failed: {failures}")
    stores.sort(key=lambda store: -store["selectedCells"])
    result = {
        "settings": {key: value for key, value in payload.items() if key != "r2EndpointUrl"},
        "authenticated": bool(token),
        "release": CATALOGUE_VERSION,
        "selection": {
            "organism": "Homo sapiens",
            "tissues": list(BLOOD_TISSUES),
            "cells": "is_primary_data and a blood tissue label",
        },
        "stores": stores,
        "genes": gene_sharing(features, stores, payload),
        "totals": {
            "stores": len(stores),
            "collections": len({store["collectionId"] for store in stores}),
            "cells": sum(store["cells"] for store in stores),
            "selectedCells": sum(store["selectedCells"] for store in stores),
        },
        "wallSeconds": time.perf_counter() - started,
        "traffic": traffic.totals(),
        "resources": asdict(sampler.stop()),
    }
    kept = set(result["genes"]["kept"])
    result["totals"]["keptStores"] = len(kept)
    result["totals"]["keptCells"] = sum(s["selectedCells"] for s in stores if s["storeId"] in kept)
    prefix = (payload.get("outputPrefix") or "").rstrip("/")
    if prefix:
        bucket = storage(payload)
        for store_id, path in tables.items():
            bucket.upload(path, f"{prefix}/cells/{store_id}.parquet")
        result["cellTablesUri"] = f"{prefix}/cells/"
    result = jsonable(result)
    write_json("blood_corpus_inventory.json", result, directory=work)
    return result


# ---- the composite and its analyses ----------------------------------------------------------------


def open_source(location: str, token, memory: str, threads: int):
    """A Hugging Face store after one recursive listing, which fills the listing cache that opening the store would
    otherwise fill with a request per directory (the Hub allows 1,000 listing requests in five minutes)."""
    import cytearc
    from cytearc.storage.huggingface import HfReadStore
    from zarr.core.sync import sync

    store = HfReadStore.from_url(
        location, storage_options={"token": token, "block_size": 1}, read_only=True
    )
    sync(store.fs._find(store.path, detail=True))
    return cytearc.DataStore(
        store, zarr_mode="r", min_features_per_cell=-1, mem_budget=memory, nthreads=threads
    )


def open_sources(sources: dict, token, memory: str, threads: int, workers: int = 8) -> dict:
    """Every source opened, eight at a time, in the payload's order."""
    locations = [location.rstrip("/") for location in sources.values()]
    if len(locations) != len(set(locations)):
        raise ValueError("The same source store was included more than once")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        opened = list(
            pool.map(
                lambda location: open_source(location, token, memory, threads), sources.values()
            )
        )
    return dict(zip(sources, opened, strict=True))


def blood_rows(sources: dict, payload: dict) -> dict:
    """Each source's selected cells; ``cellsPerSource`` keeps the first that many for checks, contiguous rows that
    touch few count chunks, and ``expectedCells`` must match the full selection."""
    expected = payload.get("expectedCells", {})
    with ThreadPoolExecutor(max_workers=8) as pool:
        selections = dict(zip(sources, pool.map(blood_selection, sources.values()), strict=True))
    rows = {}
    for name in sources:
        selected = selections[name]
        if name in expected and len(selected) != expected[name]:
            raise ValueError(
                f"{name} has {len(selected)} selected cells; the inventory found {expected[name]}"
            )
        if payload.get("cellsPerSource"):
            selected = selected[: payload["cellsPerSource"]]
        if not len(selected):
            raise ValueError(f"No selected cells in {name}")
        rows[name] = selected
    return rows


def annotate_corpus(joint, collections: dict) -> dict:
    """Add the study (collection), study-scoped donor and harmonized cell-type columns; Harmony2 corrects by study,
    since one study's stores can split its donors by age, sex or CMV status."""
    # Work on factorized codes: tens of millions of cells make per-cell string arithmetic cost gigabytes.
    dataset_codes, dataset_names = pd.factorize(np.asarray(joint.cells.fetch_all("dataset")))
    study_names, study_of_dataset = np.unique(
        [collections[str(name)] for name in dataset_names], return_inverse=True
    )
    study_codes = study_of_dataset[dataset_codes]
    donor_codes, donor_names = pd.factorize(np.asarray(joint.cells.fetch_all("orig_donor_id")))
    known_donor = ~pd.Index(donor_names.astype(str)).str.strip().str.lower().isin(
        list(MISSING_DONORS)
    )
    known = known_donor[donor_codes] & (donor_codes >= 0)
    pair_values, pair_codes = np.unique(
        study_codes.astype(np.int64) * len(donor_names) + donor_codes, return_inverse=True
    )
    pair_names = np.array(
        [
            f"{study_names[value // len(donor_names)]}__{donor_names[value % len(donor_names)]}"
            for value in pair_values
        ],
        dtype=object,
    )
    donor_groups = np.where(known, pair_names[pair_codes], "")
    label_codes, label_names = pd.factorize(np.asarray(joint.cells.fetch_all("orig_cell_type")))
    label_map = {str(name): blood_cell_type(str(name)) for name in label_names}
    harmonized = np.array([label_map[str(name)] for name in label_names], dtype=object)[label_codes]
    study = study_names.astype(object)[study_codes]
    joint.cells.insert("study", study, overwrite=True)
    joint.cells.insert("integration_batch", study, overwrite=True)
    joint.cells.insert("donor_group", donor_groups.astype(object), overwrite=True)
    joint.cells.insert("harmonized_cell_type", harmonized, overwrite=True)
    return {
        "studies": len(study_names),
        "cellsWithoutDonor": int((~known).sum()),
        "cellsWithoutHarmonizedType": int(np.sum(harmonized == "")),
        "studyScopedDonors": int(np.unique(pair_codes[known]).size),
    }


def corpus_settings(payload: dict, *, integrated: bool) -> dict:
    """The ten-study cohort's settings with UMAP in both runs and Leiden at each of ``leidenResolutions``."""
    settings = analysis_settings(payload, integrated=integrated)
    resolutions = [float(value) for value in payload.get("leidenResolutions", [1.0])]
    settings["umap"] = True
    settings["params"]["umap"] = {"parallel": True, "n_epochs": 300, "random_seed": 4444}
    settings["params"]["leiden"] = {
        "backend": "cytearc",
        "partitions": resolutions,
        "selected": resolutions[0],
    }
    return settings


def partitions(joint, run) -> dict[str, np.ndarray]:
    """Each Leiden resolution's clusters, by the run's stage name."""
    found = {}
    for stage in run.report().get("stages", []):
        name = stage["stage"]
        if name.startswith("leiden_") and stage.get("status") != "skipped":
            found[name] = array_values(joint.artifacts.load(run[name])).astype(np.int32)
    if not found:
        found["leiden"] = array_values(joint.artifacts.load(run["clusters"])).astype(np.int32)
    return found


def save_cells(joint, run, path: Path) -> dict:
    """The run's UMAP and clusters at each resolution, with the cells' study, store, donor and label codes."""
    umap = array_values(joint.artifacts.load(run["umap"])).astype(np.float32)
    clusters = partitions(joint, run)
    if umap.shape[0] != joint.cells.N or any(len(v) != joint.cells.N for v in clusters.values()):
        raise AssertionError("Embedding or cluster rows differ from the composite's cells")
    codes, names = {}, {}
    for column in (
        "dataset",
        "study",
        "donor_group",
        "orig_disease",
        "orig_cell_type",
        "harmonized_cell_type",
    ):
        codes[column], names[column] = encode(joint.cells.fetch_all(column))
    np.savez_compressed(
        path,
        umap=umap,
        **{name.replace(".", "_"): values for name, values in clusters.items()},
        **{f"{column}_codes": value for column, value in codes.items()},
        **{f"{column}_names": np.asarray(value, dtype=str) for column, value in names.items()},
    )
    studies = np.asarray(names["study"])[codes["study"]]
    return {
        "cells": int(joint.cells.N),
        "clusters": {name: int(np.unique(values).size) for name, values in clusters.items()},
        "clusterByStudy": {
            name: cross_table(values.astype(str), studies) for name, values in clusters.items()
        },
        "bytes": path.stat().st_size,
    }


def study_mixing(joint, run, block: int = 1_000_000) -> dict:
    """Of each cell's neighbours, the share from its own study, overall and by harmonized cell type, read a block
    of cells at a time so the whole neighbour table is never in memory."""
    indices = joint.artifacts.load(run["neighbors"])["indices"]
    codes, names = pd.factorize(np.asarray(joint.cells.fetch_all("study")))
    type_codes, type_names = pd.factorize(np.asarray(joint.cells.fetch_all("harmonized_cell_type")))
    same = np.empty(len(codes), dtype=np.float32)
    for start in range(0, len(codes), block):
        neighbours = np.asarray(indices[start : start + block])
        same[start : start + block] = (
            codes[neighbours] == codes[start : start + block, None]
        ).mean(axis=1)
    return {
        "sameStudyNeighbourShare": float(same.mean()),
        "byHarmonizedType": {
            str(name): float(same[type_codes == i].mean())
            for i, name in enumerate(type_names)
            if str(name)
        },
        "studies": len(names),
    }


def storage_options_by_source(sources: dict, token) -> dict:
    return {name: {"token": token, "block_size": 1} for name in sources}


def analyse(payload: dict) -> dict:
    from cytearc.composite import create_composite

    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "32G"), payload.get("nthreads", 16)
    token = hub_token(payload)
    traffic, probe = HubTraffic(payload.get("quotaReserve", 200)), NetworkProbe()
    baseline_disk = disk_used(work)
    result = {
        "settings": {
            key: value
            for key, value in payload.items()
            if key not in ("r2EndpointUrl", "sources", "expectedCells")
        },
        "sources": payload["sources"],
        "authenticated": bool(token),
        "timingBoundary": "Opening the sources, composite creation, then an unintegrated analysis through UMAP and native Leiden; saving the store is timed separately",
    }
    with ExitStack() as stack:
        stack.enter_context(hub_traffic(traffic))
        stack.enter_context(probe_remote_reads(probe))
        sources, result["sourceOpening"] = measure(
            lambda: open_sources(payload["sources"], token, memory, threads), probe
        )
        rows, result["selection"] = measure(lambda: blood_rows(sources, payload), probe)
        result["sourceCellCounts"] = {name: len(value) for name, value in rows.items()}
        joint, result["creation"] = measure(
            lambda: create_composite(
                sources,
                at=work / STORE,
                assay="RNA",
                features="intersection",
                rows=rows,
                mem_budget=memory,
                nthreads=threads,
            ),
            probe,
        )
        result["cells"], result["genesShared"] = int(joint.cells.N), int(joint.RNA.feats.N)
        if payload.get("expectedGenes") and result["genesShared"] != payload["expectedGenes"]:
            raise ValueError(
                f"The composite shares {result['genesShared']} genes; the inventory found {payload['expectedGenes']}"
            )
        result["localStoreAfterCreateBytes"] = dir_bytes(work / STORE)
        result["diskUsedAfterCreateBytes"] = disk_used(work) - baseline_disk
        result["nonzeroCounts"] = int(
            np.asarray(joint.cells.fetch_all("RNA_nFeatures"), dtype=np.int64).sum()
        )
        result["cellColumns"] = [str(column) for column in joint.cells.columns]
        result["annotation"] = annotate_corpus(joint, payload["collections"])
        checkpoint(payload, result, "creation")
        run, result["unintegrated"] = measure(
            lambda: joint.pipeline.run(**corpus_settings(payload, integrated=False)), probe
        )
        result["unintegrated"]["summary"] = stage_summary(run)
        result["unintegrated"]["artifacts"] = {key: run[key].artifact_id for key in UPSTREAM}
        result["localStoreAfterAnalysisBytes"] = dir_bytes(work / STORE)
        result["diskUsedAfterAnalysisBytes"] = disk_used(work) - baseline_disk
        result["unintegrated"]["mixing"] = study_mixing(joint, run)
        checkpoint(payload, result, "unintegrated analysis")
        cells_path = work / "cells_unintegrated.npz"
        result["cellOutputs"], result["cellOutputMeasurement"] = measure(
            lambda: save_cells(joint, run, cells_path), probe
        )
    result["traffic"] = traffic.totals()
    checkpoint(payload, result, "cell outputs")
    prefix = (payload.get("outputPrefix") or "").rstrip("/")
    if prefix:
        bucket = storage(payload)
        started = time.perf_counter()
        try:
            bucket.upload(cells_path, f"{prefix}/{cells_path.name}")
            bucket.upload_dir(work / STORE, f"{prefix}/{STORE}")
        except Exception as error:
            result["upload"] = {"error": f"{type(error).__name__}: {error}"}
            checkpoint(payload, result, "upload failed")
            raise
        result["upload"] = {
            "seconds": time.perf_counter() - started,
            "storeUri": f"{prefix}/{STORE}",
            "cellsUri": f"{prefix}/{cells_path.name}",
        }
    result = jsonable(result)
    write_json("blood_corpus_analysis.json", result, directory=work)
    return result


def reopen(payload: dict, work: Path, token, memory: str, threads: int):
    """The saved composite, downloaded from ``storeUri`` unless it is already in ``work``."""
    import cytearc
    from cytearc.storage.huggingface import HfReadStore
    from zarr.core.sync import sync

    path = work / STORE
    if not path.exists():
        storage(payload).download_dir(payload["storeUri"], path)
    # One recursive listing per source, with the options the composite reopens it with, so the file system instance
    # it shares holds the listings instead of the Hub serving one request per directory.
    for location in payload["sources"].values():
        store = HfReadStore.from_url(
            location, storage_options={"token": token, "block_size": 1}, read_only=True
        )
        sync(store.fs._find(store.path, detail=True))
    return cytearc.DataStore(
        str(path),
        source_storage_options=storage_options_by_source(payload["sources"], token),
        min_features_per_cell=-1,
        mem_budget=memory,
        nthreads=threads,
    )


def integrate(payload: dict) -> dict:
    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "32G"), payload.get("nthreads", 16)
    token = hub_token(payload)
    traffic, probe = HubTraffic(payload.get("quotaReserve", 200)), NetworkProbe()
    result = {
        "settings": {
            key: value for key, value in payload.items() if key not in ("r2EndpointUrl", "sources")
        },
        "timingBoundary": "Reopening the saved composite, confirming the unintegrated run is stored, then a Harmony2 analysis on study reusing its upstream artifacts, through UMAP and native Leiden",
    }
    with ExitStack() as stack:
        stack.enter_context(hub_traffic(traffic))
        stack.enter_context(probe_remote_reads(probe))
        started = time.perf_counter()
        joint = reopen(payload, work, token, memory, threads)
        result["download"] = {
            "seconds": time.perf_counter() - started,
            "storeBytes": dir_bytes(work / STORE),
        }
        baseline_disk = disk_used(work)
        unintegrated, result["unintegratedRecall"] = measure(
            lambda: joint.pipeline.run(**corpus_settings(payload, integrated=False)), probe
        )
        result["unintegratedRecall"]["summary"] = stage_summary(unintegrated)
        if result["unintegratedRecall"]["summary"]["artifactsCreated"]:
            raise AssertionError(
                "Recalling the unintegrated run created artifacts; it must reuse every one"
            )
        run, result["integrated"] = measure(
            lambda: joint.pipeline.run(**corpus_settings(payload, integrated=True)), probe
        )
        result["integrated"]["summary"] = stage_summary(run)
        result["sharedUpstreamArtifacts"] = {
            key: unintegrated[key].artifact_id == run[key].artifact_id for key in UPSTREAM
        }
        if not all(result["sharedUpstreamArtifacts"].values()):
            raise AssertionError("The Harmony2 analysis did not reuse the upstream artifacts")
        checkpoint(payload, result, "integrated analysis")
        result["integrated"]["mixing"] = study_mixing(joint, run)
        result["unintegratedMixing"] = study_mixing(joint, unintegrated)
        result["localStoreAfterIntegrationBytes"] = dir_bytes(work / STORE)
        result["diskAddedByIntegrationBytes"] = disk_used(work) - baseline_disk
        cells_path = work / "cells_integrated.npz"
        result["cellOutputs"], result["cellOutputMeasurement"] = measure(
            lambda: save_cells(joint, run, cells_path), probe
        )
    result["traffic"] = traffic.totals()
    prefix = (payload.get("outputPrefix") or "").rstrip("/")
    if prefix:
        bucket = storage(payload)
        started = time.perf_counter()
        bucket.upload(cells_path, f"{prefix}/{cells_path.name}")
        bucket.upload_dir(
            work / STORE,
            f"{prefix}/{payload.get('integratedStoreName', 'composite_integrated.zarr')}",
        )
        result["upload"] = {"seconds": time.perf_counter() - started}
    result = jsonable(result)
    write_json("blood_corpus_integration.json", result, directory=work)
    return result


def export_inputs(payload: dict) -> dict:
    """Write the composite's cells as one H5AD file per store, with the shared genes, for the comparators."""
    import h5py
    from cytearc.writers.export import to_h5ad

    from experiments.reuse.composite_and_remote import identifier_hash
    from experiments.runtime.h5ad import read_column

    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "32G"), payload.get("nthreads", 16)
    token = hub_token(payload)
    traffic, probe = HubTraffic(payload.get("quotaReserve", 200)), NetworkProbe()
    bucket = storage(payload)
    prefix = payload["inputsPrefix"].rstrip("/")
    inputs = []
    result = {
        "settings": {
            key: value for key, value in payload.items() if key not in ("r2EndpointUrl", "sources")
        },
        "inputs": inputs,
    }
    with ExitStack() as stack:
        stack.enter_context(hub_traffic(traffic))
        stack.enter_context(probe_remote_reads(probe))
        joint = reopen(payload, work, token, memory, threads)
        datasets = np.asarray(joint.cells.fetch_all("dataset"), dtype=str)
        cell_ids = np.asarray(joint.cells.fetch_all("ids"), dtype=str)
        feature_hash = identifier_hash(joint.RNA.feats.fetch_all("ids"))
        for name in payload["sources"]:
            selected = datasets == name
            joint.cells.insert("materialization_cells", selected, overwrite=True)
            selection = joint.snapshot_cell_selection("materialization_cells")
            output = work / f"{name}.h5ad"
            _, measured = measure(
                lambda: to_h5ad(
                    joint.RNA,
                    str(output),
                    cell_selection=selection,
                    embeddings_cols=[],
                    nthreads=threads,
                ),
                probe,
            )
            cell_hash = identifier_hash(cell_ids[selected])
            with h5py.File(output, "r") as handle:
                if (
                    identifier_hash(read_column(handle["obs"], handle["obs"].attrs["_index"]))
                    != cell_hash
                ):
                    raise AssertionError(f"Exporting {name} changed its cells' identity or order")
                if (
                    identifier_hash(read_column(handle["var"], handle["var"].attrs["_index"]))
                    != feature_hash
                ):
                    raise AssertionError(
                        f"Exporting {name} changed the shared genes' identity or order"
                    )
            uri = f"{prefix}/{name}.h5ad"
            bucket.upload(output, uri)
            inputs.append(
                {
                    "name": name,
                    "uri": uri,
                    "cells": int(selected.sum()),
                    "bytes": output.stat().st_size,
                    "cellIdSha256": cell_hash,
                    "featureIdSha256": feature_hash,
                    "export": measured,
                }
            )
            output.unlink()
    result["totals"] = {
        "files": len(inputs),
        "cells": sum(i["cells"] for i in inputs),
        "bytes": sum(i["bytes"] for i in inputs),
    }
    result["traffic"] = traffic.totals()
    result = jsonable(result)
    write_json("blood_corpus_inputs.json", result, directory=work)
    return result


# ---- donor-level cell states (prespecified_effects protocol) ------------------------------------

# Lineages by their marker genes: each lineage scores a cluster by its best alternative gene set, less its
# exclusion genes, on gene means standardized across clusters.
LINEAGES = {
    "CD8 T": ([("CD3E", "CD8A", "CD8B")], ()),
    "CD4 T": ([("CD3E", "CD4"), ("CD3E", "IL7R")], ("CD8A", "CD8B")),
    "NK": ([("NKG7", "GNLY", "KLRD1")], ("CD3E",)),
    "B lineage": ([("MS4A1", "CD79A"), ("JCHAIN", "MZB1")], ()),
    "monocytes": ([("CD14", "LYZ"), ("FCGR3A", "LST1")], ()),
}
# A cluster takes its best lineage when that score reaches LINEAGE_SCORE and leads the next by LINEAGE_MARGIN.
LINEAGE_SCORE, LINEAGE_MARGIN = 1.0, 0.25
# States: their lineage, their genes and their contrast genes; a cell's score is the mean of its genes less the
# mean of the contrast genes, and the cut is the score's STATE_QUANTILE among the lineage's healthy donors' cells.
STATES = {
    "naive CD8 T": ("CD8 T", ("CCR7", "SELL", "LEF1", "TCF7"), ("GZMK", "GZMB", "GZMH", "NKG7")),
    "naive CD4 T": ("CD4 T", ("CCR7", "SELL", "LEF1", "TCF7"), ("KLRB1", "GZMK", "CCL5", "S100A4")),
    "GZMK+ CD8 T": ("CD8 T", ("GZMK",), ("GZMB", "GZMH", "CCR7")),
    "CD8 TEMRA": ("CD8 T", ("GZMB", "GZMH", "FGFBP2", "CX3CR1", "GNLY"), ("CCR7", "GZMK", "SELL")),
    "adaptive NK": ("NK", ("KLRC2",), ("FCER1G",)),
    "plasmablasts": ("B lineage", ("JCHAIN", "MZB1", "XBP1", "PRDM1"), ("MS4A1",)),
    "interferon-stimulated monocytes": (
        "monocytes",
        ("IFI27", "IFI44L", "ISG15", "MX1", "IFIT1", "IFIT3"),
        (),
    ),
}
STATE_QUANTILE = 0.75
DONOR_COLUMNS = (
    "study",
    "dataset",
    "orig_disease",
    "orig_sex",
    "orig_development_stage",
    "orig_self_reported_ethnicity",
)


def marker_genes() -> list[str]:
    genes = {
        g
        for alternatives, excluded in LINEAGES.values()
        for group in alternatives
        for g in (*group, *excluded)
    }
    genes |= {g for _, positive, contrast in STATES.values() for g in (*positive, *contrast)}
    return sorted(genes)


def log_expression(
    joint, sources: dict, genes: list[str], token, memory: str, threads: int
) -> tuple[np.ndarray, list[str], dict]:
    """Each composite cell's log(1 + 10,000 counts / library) for ``genes``, read gene-major from the cell's own store
    and matched by symbol, since the stores share fewer genes by exact ID than they hold; NaN where a store lacks the
    gene."""
    from cytearc.assay.normalization import feature_value_blocks
    from cytearc.metadata.selection import NormalizationSpec

    datasets = np.asarray(joint.cells.fetch_all("dataset"), dtype=object)
    cell_ids = np.asarray(joint.cells.fetch_all("ids"), dtype=object)
    values = np.full((len(cell_ids), len(genes)), np.nan, dtype=np.float32)
    missing = {}
    for name, location in sources.items():
        mine = np.flatnonzero(datasets == name)
        source = open_source(location, token, memory, threads)
        prefix = f"{name}__"
        wanted = pd.Index(
            np.asarray([value[len(prefix) :] for value in cell_ids[mine]], dtype=object)
        )
        positions = pd.Index(np.asarray(source.cells.fetch_all("ids"), dtype=object)).get_indexer(
            wanted
        )
        if (positions < 0).any():
            raise AssertionError(
                f"{int((positions < 0).sum())} composite cells of {name} are not in the store"
            )
        order = np.argsort(positions, kind="stable")
        rows = positions[order].astype(np.int64)
        names = np.asarray(source.RNA.feats.fetch_all("names"), dtype=str)
        first = {gene: i for i, gene in reversed(list(enumerate(names)))}
        present = [j for j, gene in enumerate(genes) if gene in first]
        missing[name] = [gene for gene in genes if gene not in first]
        if not present:
            continue
        features = np.array([first[genes[j]] for j in present], dtype=np.int64)
        totals = np.asarray(source.cells.fetch_all("RNA_nCounts"), dtype=np.float64)[rows]
        block_values = np.empty((len(rows), len(present)), dtype=np.float32)
        _, blocks = feature_value_blocks(
            source.RNA,
            features,
            rows,
            NormalizationSpec(source="raw"),
            nthreads=threads,
            resident_bytes=int(block_values.nbytes + totals.nbytes + rows.nbytes),
            scratch_bytes_per_row=16 * len(present),
        )
        start = 0
        for block in blocks:
            block = np.asarray(block, dtype=np.float64)
            stop = start + len(block)
            block_values[start:stop] = np.log1p(
                block * (1e4 / np.maximum(totals[start:stop], 1.0))[:, None]
            )
            start = stop
        if start != len(rows):
            raise AssertionError(f"{name}: streamed {start} of {len(rows)} cells")
        values[np.ix_(mine[order], np.asarray(present))] = block_values
    return values, genes, missing


def assign_lineages(
    values: np.ndarray, genes: list[str], clusters: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Each cluster's lineage from its gene means standardized across clusters, blind to donor covariates."""
    column = {gene: i for i, gene in enumerate(genes)}
    labels, codes = np.unique(clusters, return_inverse=True)
    sizes = np.bincount(codes)
    known = ~np.isnan(values)
    # A cluster's mean over the cells whose store carries the gene; undefined (NaN) when none does.
    sums = np.stack(
        [
            np.bincount(codes, np.where(known[:, i], values[:, i], 0.0), len(labels))
            for i in range(len(genes))
        ],
        axis=1,
    )
    carriers = np.stack(
        [np.bincount(codes, known[:, i], len(labels)) for i in range(len(genes))], axis=1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(carriers > 0, sums / carriers, np.nan)
        z = (means - np.nanmean(means, axis=0)) / np.maximum(np.nanstd(means, axis=0), 1e-9)

    def score(alternatives, excluded):
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            groups = [
                np.nanmean(z[:, [column[g] for g in group]], axis=1) for group in alternatives
            ]
            best = np.nanmax(np.stack(groups), axis=0) if len(groups) > 1 else groups[0]
            penalty = (
                np.nanmean(z[:, [column[g] for g in excluded]], axis=1)
                if excluded
                else np.zeros(len(z))
            )
        return np.nan_to_num(best, nan=-np.inf) - np.nan_to_num(penalty, nan=0.0)

    names = list(LINEAGES)
    scores = np.stack([score(*LINEAGES[name]) for name in names], axis=1)
    ranked = np.sort(scores, axis=1)
    best = scores.argmax(axis=1)
    lineage = np.where(
        (ranked[:, -1] >= LINEAGE_SCORE) & (ranked[:, -1] - ranked[:, -2] >= LINEAGE_MARGIN),
        np.array(names, dtype=object)[best],
        "other",
    )
    table = {
        str(label): {
            "cells": int(n),
            "lineage": str(lin),
            "scores": dict(zip(names, map(float, s), strict=True)),
        }
        for label, n, lin, s in zip(labels, sizes, lineage, scores, strict=True)
    }
    return lineage[codes], table


def call_states(
    values: np.ndarray, genes: list[str], lineage: np.ndarray, healthy: np.ndarray
) -> tuple[dict, dict, dict]:
    """Each state's calls and the cells it scores (its lineage's cells whose store carries all of its genes), with
    its cut."""
    column = {gene: i for i, gene in enumerate(genes)}
    calls, scored, cuts = {}, {}, {}
    for state, (parent, positive, contrast) in STATES.items():
        score = values[:, [column[g] for g in positive]].mean(axis=1)
        if contrast:
            score = score - values[:, [column[g] for g in contrast]].mean(axis=1)
        # A cell whose store lacks any of the state's genes has a NaN score and is not scored.
        inside = (lineage == parent) & ~np.isnan(score)
        reference = inside & healthy
        if not reference.any():
            # Left out of the donor table rather than reported as zero cells.
            cuts[state] = {
                "lineage": parent,
                "cut": None,
                "reason": "no healthy donor's cells carry all of the state's genes",
            }
            continue
        cut = float(np.quantile(score[reference], STATE_QUANTILE))
        calls[state] = np.where(inside, score > cut, False)
        scored[state] = inside
        cuts[state] = {
            "lineage": parent,
            "cut": cut,
            "scoredCells": int(inside.sum()),
            "lineageCells": int((lineage == parent).sum()),
        }
    return calls, scored, cuts


def donor_states(payload: dict) -> dict:
    """Per study-scoped donor: cells of each lineage and state, with the donor's recorded metadata, from the
    Harmony2 run's clusters at ``lineageResolution``."""
    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    memory, threads = payload.get("memBudget", "32G"), payload.get("nthreads", 16)
    token = hub_token(payload)
    traffic, probe = HubTraffic(payload.get("quotaReserve", 200)), NetworkProbe()
    result = {
        "settings": {
            key: value for key, value in payload.items() if key not in ("r2EndpointUrl", "sources")
        },
        "lineageRule": {"score": LINEAGE_SCORE, "margin": LINEAGE_MARGIN},
        "stateQuantile": STATE_QUANTILE,
    }
    with ExitStack() as stack:
        stack.enter_context(hub_traffic(traffic))
        stack.enter_context(probe_remote_reads(probe))
        joint = reopen(payload, work, token, memory, threads)
        run = joint.pipeline.run(**corpus_settings(payload, integrated=True))
        if stage_summary(run)["artifactsCreated"]:
            raise AssertionError(
                "Recalling the Harmony2 run created artifacts; it must reuse every one"
            )
        clusters = partitions(joint, run)[f"leiden_{float(payload.get('lineageResolution', 2.0))}"]
        (values, genes, result["missingGenesByStore"]), result["markerReads"] = measure(
            lambda: log_expression(
                joint, payload["sources"], marker_genes(), token, memory, threads
            ),
            probe,
        )
    cells = pd.DataFrame(
        {
            "dataset": pd.Categorical(joint.cells.fetch_all("dataset")),
            "donor_group": pd.Categorical(joint.cells.fetch_all("donor_group")),
        }
    )
    # The rules of the stores this composite holds; each of their columns must be there.
    rules = {store: rule for store, rule in SAMPLE_RULES.items() if store in set(cells["dataset"])}
    for column in rule_columns(rules):
        if f"orig_{column}" not in joint.cells.columns:
            raise KeyError(
                f"The composite has no orig_{column} column, which the sample rules read"
            )
        cells[column] = pd.Categorical(joint.cells.fetch_all(f"orig_{column}"))
    kept, result["sampleSelection"] = sample_selection(cells, rules)
    del cells
    disease = np.asarray(joint.cells.fetch_all("orig_disease"), dtype=str)
    lineage, result["clusters"] = assign_lineages(values, genes, clusters)
    # The cut is set among the healthy donors' cells the donor table keeps.
    calls, scored, result["states"] = call_states(
        values, genes, lineage, (disease == "normal") & kept
    )
    frame = pd.DataFrame(
        {
            "donor_group": np.asarray(joint.cells.fetch_all("donor_group"), dtype=str),
            "lineage": lineage,
        }
    )
    for column in DONOR_COLUMNS:
        if column in joint.cells.columns:
            frame[column] = pd.Categorical(np.asarray(joint.cells.fetch_all(column), dtype=str))
    for state, called in calls.items():
        frame[state] = called
        frame[f"scored_{state}"] = scored[state]
    frame = frame[kept]
    lineages = frame.groupby(["donor_group", "lineage"], observed=True).size().unstack(fill_value=0)
    states = frame.groupby("donor_group", observed=True)[list(calls)].sum()
    # A state's share is of the lineage cells it scores, which leaves out stores without its genes.
    scored_cells = frame.groupby("donor_group", observed=True)[
        [f"scored_{state}" for state in calls]
    ].sum()
    meta = frame.groupby("donor_group", observed=True)[
        [c for c in DONOR_COLUMNS if c in frame]
    ].agg(lambda s: "|".join(sorted(set(map(str, s)))))
    donors = (
        meta.join(lineages.add_prefix("cells_"))
        .join(states.add_prefix("state_"))
        .join(scored_cells)
        .reset_index()
    )
    path = work / "donor_states.parquet"
    donors.to_parquet(path, index=False)
    result["donors"] = len(donors)
    result["genes"] = genes
    result["traffic"] = traffic.totals()
    prefix = (payload.get("outputPrefix") or "").rstrip("/")
    if prefix:
        storage(payload).upload(path, f"{prefix}/{path.name}")
        result["donorStatesUri"] = f"{prefix}/{path.name}"
    result = jsonable(result)
    write_json("blood_corpus_donor_states.json", result, directory=work)
    return result


# ---- donor-level estimates ----------------------------------------------------------------------


def study_effect(donors: pd.DataFrame, covariate: str, adjust: tuple[str, ...] = ()) -> dict:
    """One study's log-odds effect of ``covariate`` on a state's share of its lineage: a binomial model of each
    donor's state cells out of lineage cells, its standard error scaled by the Pearson dispersion (quasi-binomial)."""
    import statsmodels.api as sm

    design = sm.add_constant(donors[[covariate, *adjust]].astype(float), has_constant="add")
    response = np.column_stack([donors["state"], donors["lineage"] - donors["state"]]).astype(float)
    # Fitted at scale 1 and rescaled here: statsmodels' scale="X2" divides by the trials of two-column counts.
    fit = sm.GLM(response, design, family=sm.families.Binomial()).fit()
    dispersion = float(fit.pearson_chi2 / fit.df_resid)
    return {
        "effect": float(fit.params[covariate]),
        "standardError": float(fit.bse[covariate] * np.sqrt(dispersion)),
        "dispersion": dispersion,
        "donors": len(donors),
    }


def random_effects(effects: np.ndarray, errors: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooling of per-study effects, with a 95% interval."""
    effects, errors = np.asarray(effects, dtype=float), np.asarray(errors, dtype=float)
    weights = 1.0 / errors**2
    fixed = float(np.sum(weights * effects) / np.sum(weights))
    q = float(np.sum(weights * (effects - fixed) ** 2))
    k = len(effects)
    tau2 = (
        max(0.0, (q - (k - 1)) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
        if k > 1
        else 0.0
    )
    pooled_weights = 1.0 / (errors**2 + tau2)
    pooled = float(np.sum(pooled_weights * effects) / np.sum(pooled_weights))
    error = float(np.sqrt(1.0 / np.sum(pooled_weights)))
    return {
        "effect": pooled,
        "standardError": error,
        "low": pooled - 1.959964 * error,
        "high": pooled + 1.959964 * error,
        "tau2": tau2,
        "q": q,
        "studies": k,
    }


# The protocol's effects: a state, the covariate it is tested against, and the expected sign of the effect.
EFFECTS = (
    ("naive CD8 T", "age", -1),
    ("naive CD4 T", "age", -1),
    ("GZMK+ CD8 T", "age", 1),
    ("adaptive NK", "cytomegalovirus infection", 1),
    ("CD8 TEMRA", "cytomegalovirus infection", 1),
    ("plasmablasts", "COVID-19", 1),
    ("interferon-stimulated monocytes", "COVID-19", 1),
)
COVARIATES = ("age", "cytomegalovirus infection", "COVID-19")
# The single-study positive control: lupus against healthy donors in Perez 2022.
LUPUS_CONTROL = (
    "interferon-stimulated monocytes",
    "systemic lupus erythematosus",
    1,
    "perez_2022_multiplexed_scrna_1_2_million_pbmcs_adult_lupus_samples_218acb0f",
)
MINIMUM_LINEAGE_CELLS = 100
MINIMUM_AGE_DONORS, MINIMUM_GROUP_DONORS, ADULT_AGE = 20, 5, 18
AGREEMENT = 2 / 3


def donor_ages(stages: pd.Series) -> pd.Series:
    """Age in years from a development stage such as "45-year-old stage"; a decade or an open range has no age."""
    years = stages.astype(str).str.extract(r"^(\d+)-year-old stage$", expand=False)
    return pd.to_numeric(years, errors="coerce")


def effect_table(donors: pd.DataFrame, state: str, covariate: str) -> pd.DataFrame:
    """The donors one effect compares, with their study, state cells, scored lineage cells and all lineage cells,
    and the covariate: age in decades with sex for healthy adults, or the disease (1) against healthy donors (0)."""
    lineage = STATES[state][0]
    disease = donors["orig_disease"].astype(str)
    table = pd.DataFrame(
        {
            "study": donors["study"].astype(str),
            "state": donors[f"state_{state}"].astype(float),
            "lineage": donors[f"scored_{state}"].astype(float),
            "lineageCells": donors.get(f"cells_{lineage}", pd.Series(0, index=donors.index)),
        }
    )
    if covariate == "age":
        age = donor_ages(donors["orig_development_stage"])
        sex = donors["orig_sex"].astype(str)
        table["decades"] = age / 10
        table["male"] = (sex == "male").astype(float)
        include = (disease == "normal") & (age >= ADULT_AGE) & sex.isin(["male", "female"])
    else:
        table["case"] = (disease == covariate).astype(float)
        include = disease.isin([covariate, "normal"])
    return table[include.to_numpy()]


def decision(effects: np.ndarray, pooled: dict, expected: int) -> dict:
    """The protocol's rule: recovered when the pooled interval excludes zero in the expected direction and at least
    two thirds of the studies agree with that direction in sign."""
    agreement = float(np.mean(np.sign(effects) == expected))
    excludes = pooled["low"] > 0 if expected > 0 else pooled["high"] < 0
    return {"agreement": agreement, "recovered": bool(excludes and agreement >= AGREEMENT)}


def effect_estimates(
    donors: pd.DataFrame, state: str, covariate: str, expected: int | None = None
) -> dict:
    """One effect within each eligible study and pooled across them, with each study left out and why."""
    result: dict = {"state": state, "covariate": covariate, "expected": expected}
    if f"state_{state}" not in donors:
        result["leftOut"] = "the state had no healthy reference cells for its cut"
        return result
    table = effect_table(donors, state, covariate)
    column, adjust = ("decades", ("male",)) if covariate == "age" else ("case", ())
    studies, left_out = {}, {}
    for study, rows in table.groupby("study", sort=True):
        enough = rows[rows["lineage"] >= MINIMUM_LINEAGE_CELLS]
        if covariate == "age":
            count, needed = len(enough), MINIMUM_AGE_DONORS
            would = int((rows["lineageCells"] >= MINIMUM_LINEAGE_CELLS).sum())
        else:
            count = int(min(enough["case"].sum(), (1 - enough["case"]).sum()))
            needed = MINIMUM_GROUP_DONORS
            lineage_rows = rows[rows["lineageCells"] >= MINIMUM_LINEAGE_CELLS]
            would = int(min(lineage_rows["case"].sum(), (1 - lineage_rows["case"]).sum()))
        if count < needed:
            left_out[study] = (
                "its stores lack the state's genes"
                if would >= needed
                else f"{count} donors where {needed} are needed"
            )
            continue
        if enough["state"].sum() == 0 or (enough["state"] == enough["lineage"]).all():
            left_out[study] = "no donor varies in the state"
            continue
        terms = tuple(term for term in adjust if enough[term].nunique() > 1)
        studies[study] = study_effect(enough, column, terms)
    result["studies"], result["leftOut"] = studies, left_out
    if studies:
        effects = np.array([value["effect"] for value in studies.values()])
        errors = np.array([value["standardError"] for value in studies.values()])
        result["pooled"] = random_effects(effects, errors)
        if expected is not None:
            result |= decision(effects, result["pooled"], expected)
    elif expected is not None:
        result["recovered"] = False
    return result


def donor_effects(donors: pd.DataFrame) -> dict:
    """The protocol's effects and decisions, the lupus control, and the exploration grid of every other state and
    covariate pair as estimates only."""
    state, disease, expected, store = LUPUS_CONTROL
    prespecified = {(s, c) for s, c, _ in EFFECTS}
    return {
        "effects": [effect_estimates(donors, s, c, e) for s, c, e in EFFECTS],
        "lupusControl": effect_estimates(
            donors[donors["dataset"] == store], state, disease, expected
        ),
        "grid": [
            effect_estimates(donors, s, c)
            for s in STATES
            for c in COVARIATES
            if (s, c) not in prespecified
        ],
    }
