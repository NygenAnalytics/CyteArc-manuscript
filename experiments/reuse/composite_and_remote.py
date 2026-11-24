"""Measure remote CyteBase queries and four-study virtual composite integration."""

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import cytearc
import numpy as np
from cytearc.composite import create_composite
from cytearc.plotting import CellField, StudyDesign

from experiments.runtime.common import configure_cytearc, dir_bytes, write_json
from experiments.runtime.locations import CYTEBASE_RELEASE
from experiments.runtime.metrics import ResourceSampler
from experiments.runtime.recording_store import (
    NetworkProbe,
    StoreProbe,
    probe_remote_reads,
    probe_summary,
)

WORK = Path("/tmp/composite_and_remote")
SOURCE_STORAGE_OPTIONS = {"token": False, "block_size": 1}

COVID_STUDIES = {
    "blood_wilk": "wilk_2020_single_cell_atlas_peripheral_sars_cov_2_infection_456e8b9b",
    "blood_lee": "lee_2020_immunophenotyping_covid_19_development_severe_covid_19_de2c780c",
    "blood_arunachalam": "jin_2021_individual_single_cell_rna_seq_pbmc_data_arunachalam_et_al_59b69042",
    "blood_schulte_schrepping": "jin_2021_individual_single_cell_rna_seq_data_schulte_schrepping_et_al_5e717147",
}

HVG_BLACKLIST = "|".join(
    [
        r"^MT-",
        r"^RP[SL]\d+[AXY]?\d*$|^RPLP[0-2]$|^RPSA$|^MRP[SL]\d+[A-C]?$",
        r"^HLA-",
        r"^HIST[1-4]H|^H1-[1-6]$|^H1F[1-6]$|^H2AC\d+$|^H2BC\d+$"
        r"|^H3C\d+$|^H4C\d+$|^H2AW$|^H3-4$|^H4-16$",
        r"^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$"
        r"|^UTY$|^TMSB4Y$|^NLGN4Y$",
    ]
)


def evaluate_remote_queries(payload: dict) -> dict:
    probe = NetworkProbe()
    results = {}
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)

    with probe_remote_reads(probe):
        probe.reset()
        t0 = time.perf_counter()
        ts_uri = "hf://buckets/Nygen/cytebase/cytearc_docs/tabula_sapiens/data.zarr"
        ts = cytearc.DataStore(
            ts_uri,
            zarr_mode="r",
            storage_options=SOURCE_STORAGE_OPTIONS,
            min_features_per_cell=-1,
        )
        open_sec = time.perf_counter() - t0
        open_io = probe_summary(probe)

        probe.reset()
        t1 = time.perf_counter()
        umap_ref = ts.artifacts.find("embedding", imported="X_umap", from_assay="RNA")
        coords = np.asarray(ts.artifacts.load(umap_ref)["values"][:])
        cell_types = np.asarray(ts.cells.fetch_all("cell_type"))
        b_mask = cell_types == "B cell"
        if not b_mask.any():
            raise ValueError("Tabula Sapiens has no cells annotated B cell")
        active = np.asarray(ts.cells.fetch_all("I"), dtype=bool)
        if np.any(b_mask & ~active):
            raise ValueError("Published active selection omits annotated B cells")
        active_b = b_mask[active]
        ms4a1_vals = ts.get_cell_vals(from_assay="RNA", cell_key="I", k="MS4A1")[active_b]
        query_sec = time.perf_counter() - t1
        query_io = probe_summary(probe)

        results["tabulaSapiens1_1M"] = {
            "totalCells": int(ts.cells.N),
            "bCellsSelected": int(b_mask.sum()),
            "umapShape": list(coords.shape),
            "ms4a1MeanInBCells": float(np.nanmean(ms4a1_vals)),
            "openSeconds": open_sec,
            "openIo": open_io,
            "embeddingAndMarkerQuerySeconds": query_sec,
            "embeddingAndMarkerQueryIo": query_io,
            "localStoreBytes": 0,
            "sourceCountBytes": payload.get("tabulaSapiensCountBytes"),
            "countMatrixAttributes": dict(ts.RNA.matrixGroup["counts"].attrs),
        }
        probe.reset()
        repeated_started = time.perf_counter()
        repeated_values = ts.get_cell_vals(from_assay="RNA", cell_key="I", k="MS4A1")[active_b]
        repeated_coords = np.asarray(ts.artifacts.load(umap_ref)["values"][:])
        if not np.array_equal(ms4a1_vals, repeated_values) or not np.array_equal(
            coords, repeated_coords
        ):
            raise AssertionError("Repeated remote query changed its result")
        results["tabulaSapiens1_1M"]["repeatedQuery"] = {
            "elapsedSeconds": time.perf_counter() - repeated_started,
            "io": probe_summary(probe),
            "sourceCountBytes": payload.get("tabulaSapiensCountBytes"),
        }

        probe.reset()
        t2 = time.perf_counter()
        wilk_id = COVID_STUDIES["blood_wilk"]
        wilk_uri = f"{CYTEBASE_RELEASE}/datasets/{wilk_id}/data.zarr"
        wilk_mount_dir = work / "wilk_mount.zarr"
        if wilk_mount_dir.exists():
            shutil.rmtree(wilk_mount_dir)
        wilk_ds = cytearc.mount_datastore(
            wilk_uri,
            at=str(wilk_mount_dir),
            storage_options=SOURCE_STORAGE_OPTIONS,
            min_features_per_cell=-1,
        )
        monocyte_mask = (
            np.asarray(wilk_ds.cells.fetch_all("cell_type")) == "CD14-positive monocyte"
        ) & np.asarray(wilk_ds.cells.fetch_all("is_primary_data"), dtype=bool)
        if not monocyte_mask.any():
            raise ValueError("Wilk has no primary CD14-positive monocytes")
        donors = np.asarray(wilk_ds.cells.fetch_all("donor_id"), dtype=str)[monocyte_mask]
        conditions = np.asarray(wilk_ds.cells.fetch_all("disease"), dtype=str)[monocyte_mask]
        group_donors = {
            group: sorted(set(donors[conditions == group])) for group in ("normal", "COVID-19")
        }
        if any(len(values) < 3 for values in group_donors.values()):
            raise ValueError(f"Insufficient donors for independent comparison: {group_donors}")
        if set(group_donors["normal"]) & set(group_donors["COVID-19"]):
            raise ValueError("Repeated donors require a paired design")
        if any(value.lower() in {"", "unknown", "nan", "none"} for value in donors):
            raise ValueError("Unknown donors cannot be biological replicates")
        wilk_ds.cells.insert("target_cells", monocyte_mask, overwrite=True)
        sel = wilk_ds.snapshot_cell_selection("target_cells")
        panel = ["ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1"]
        design = StudyDesign(sample_by="donor_id", condition_by="disease")
        test_res = wilk_ds.compare.test(
            panel,
            CellField("disease"),
            cell_selection=sel,
            groups=["normal", "COVID-19"],
            study_design=design,
            test="mann_whitney",
        )
        wilk_sec = time.perf_counter() - t2
        wilk_io = probe_summary(probe)
        pvals = {gene: float(test_res.tables[gene]["p_value_adjusted"].iloc[0]) for gene in panel}
        results["wilkCovid19DonorTest"] = {
            "totalCells": int(wilk_ds.cells.N),
            "testedCells": int(monocyte_mask.sum()),
            "seconds": wilk_sec,
            "io": wilk_io,
            "localMountMB": round(dir_bytes(wilk_mount_dir) / 1e6, 3),
            "adjustedPValues": pvals,
            "donorsByCondition": group_donors,
            "test": "Two-sided Mann-Whitney on donor means with pooled BH correction across six genes",
            "targetCellType": "CD14-positive monocyte",
        }

    return results


def stage_summary(run) -> dict:
    rep = run.report()
    stages = {}
    reused_count = 0
    created_count = 0
    for s in rep.get("stages", []):
        if s.get("status") == "skipped":
            continue
        plans = s.get("plans", [])
        reused = sum(1 for p in plans if p.get("disposition") == "reused")
        created = sum(1 for p in plans if p.get("disposition") == "created")
        reused_count += reused
        created_count += created
        stages[s["stage"]] = {
            "seconds": (s.get("metrics") or {}).get("wallSeconds"),
            "peakRssMb": ((s.get("metrics") or {}).get("rssPeakBytes") or 0) / (1024**2),
            "reused": reused,
            "created": created,
        }
    meta = rep["run"]
    wall = ((meta.get("finishedAtNs") or 0) - (meta.get("startedAtNs") or 0)) / 1e9
    return {
        "wallSeconds": wall,
        "peakRssMb": max((v["peakRssMb"] for v in stages.values()), default=0.0),
        "artifactsReused": reused_count,
        "artifactsCreated": created_count,
        "stages": stages,
    }


def neighbor_label_metrics(cell_types, studies, neighbors):
    labels = np.asarray(cell_types, dtype=str)
    studies = np.asarray(studies, dtype=str)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if neighbors.shape[0] != len(labels) or len(studies) != len(labels):
        raise ValueError("Neighbor and annotation cell axes differ")
    _, label_codes = np.unique(labels, return_inverse=True)
    study_names, study_codes = np.unique(studies, return_inverse=True)
    known = labels != ""
    neighbor_known = known[neighbors]
    matches = (label_codes[neighbors] == label_codes[:, None]) & neighbor_known & known[:, None]
    eligible = known & (neighbor_known.sum(axis=1) > 0)
    retention = np.divide(
        matches.sum(axis=1),
        neighbor_known.sum(axis=1),
        out=np.zeros(len(labels), dtype=float),
        where=neighbor_known.sum(axis=1) > 0,
    )
    if not eligible.any():
        raise ValueError("No cells have comparable harmonized neighbor annotations")
    same_count = matches.sum(axis=1)
    per_type = {}
    for label in np.unique(labels[known]):
        chosen = labels == label
        counts = np.array([np.sum(chosen & (studies == study)) for study in study_names])
        if np.any(counts < 20):
            continue
        rows = np.flatnonzero(chosen & (same_count >= 5))
        if not len(rows):
            continue
        probabilities = np.column_stack(
            [
                np.sum(matches[rows] & (study_codes[neighbors[rows]] == index), axis=1)
                / same_count[rows]
                for index in range(len(study_names))
            ]
        )
        logarithms = np.zeros_like(probabilities)
        np.log(probabilities, out=logarithms, where=probabilities > 0)
        entropy = -(probabilities * logarithms).sum(axis=1)
        proportions = counts / counts.sum()
        global_entropy = float(-np.sum(proportions * np.log(proportions)))
        if global_entropy <= 0:
            raise ValueError("Study mixing requires at least two represented studies")
        per_type[label] = {
            "cells": int(chosen.sum()),
            "evaluatedCells": len(rows),
            "cellsByStudy": dict(zip(study_names, counts.tolist(), strict=True)),
            "sameTypeNeighborFraction": float(retention[chosen & eligible].mean()),
            "normalizedStudyEntropy": float(np.minimum(entropy / global_entropy, 1).mean()),
        }
    return {
        "annotatedCellFraction": float(known.mean()),
        "sameTypeNeighborFraction": float(retention[eligible].mean()),
        "sharedTypes": per_type,
        "macroStudyMixing": float(
            np.mean([record["normalizedStudyEntropy"] for record in per_type.values()])
        )
        if per_type
        else None,
        "definition": "Within-type neighbor study entropy divided by that type's global study entropy, clipped to [0, 1]; shared types require 20 cells per study and evaluated cells require 5 same-type neighbors",
    }


def measure(operation, probe: StoreProbe) -> tuple[object, dict]:
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


def pipeline_settings(payload: dict) -> dict:
    return {
        "assay": "RNA",
        "filtering": False,
        "hvg_count": payload.get("hvgCount", 2000),
        "pca_dims": payload.get("pcaDims", 30),
        "neighbors_k": 30,
        "umap": False,
        "cell_cycle": False,
        "paris": False,
        "doublets": False,
        "markers": False,
        "snapshot_columns": ["dataset", "orig_disease", "orig_donor_id", "orig_cell_type"],
        "params": {
            "hvg": {"blacklist": HVG_BLACKLIST},
            "ann_index": {"ann_parallel": False, "rand_state": 4444},
            "leiden": {"backend": "cytearc", "partitions": [1.0], "selected": 1.0},
            "species": "homo_sapiens",
        },
    }


def validate_source_alignment(joint, sources: dict, rows: dict, *, block_rows: int = 2048) -> dict:
    from cytearc.storage.virtual_arrays import read_selection

    features = np.asarray(joint.RNA.feats.fetch_all("ids"), dtype=str)
    common = set.intersection(
        *(set(source.RNA.feats.fetch_all("ids")) for source in sources.values())
    )
    if set(features) != common or len(features) != len(common):
        raise AssertionError("Composite feature IDs do not equal the source intersection")
    source_index = np.asarray(joint.z["_composite/rowSources"][:])
    source_rows = np.asarray(joint.z["_composite/sourceRows"][:])
    copied_chunks = [
        path
        for orientation in ("counts", "countsT")
        for path in Path(joint.z.store._store.root, "RNA", orientation).rglob("*")
        if path.is_file() and path.name != "zarr.json"
    ]
    if copied_chunks:
        raise AssertionError("Composite creation wrote count chunks")
    checked = 0
    for index, (name, source) in enumerate(sources.items()):
        positions = np.flatnonzero(source_index == index)
        positions = positions[np.argsort(source_rows[positions])]
        actual_rows = source_rows[positions]
        if not np.array_equal(np.sort(actual_rows), np.sort(rows[name])):
            raise AssertionError(f"Source row selection changed for {name}")
        expected_ids = np.array(
            [f"{name}__{value}" for value in np.asarray(source.cells.fetch_all("ids"))[actual_rows]]
        )
        if not np.array_equal(np.asarray(joint.cells.fetch_all("ids"))[positions], expected_ids):
            raise AssertionError(f"Composite cell identity/order mismatch for {name}")
        source_features = {
            str(value): i for i, value in enumerate(source.RNA.feats.fetch_all("ids"))
        }
        feature_positions = np.array([source_features[value] for value in features])
        for start in range(0, len(positions), block_rows):
            selected = positions[start : start + block_rows]
            expected = read_selection(
                source.RNA.matrixGroup["counts"],
                actual_rows[start : start + block_rows],
                feature_positions,
            )
            actual = read_selection(joint.RNA.matrixGroup["counts"], selected, slice(None))
            if not np.array_equal(actual, expected):
                raise AssertionError(f"Composite count mismatch for {name}, block {start}")
            checked += actual.size
        for column in ("donor_id", "cell_type", "disease"):
            expected = np.asarray(source.cells.fetch_all(column))[actual_rows].astype(str)
            actual = np.asarray(joint.cells.fetch_all(f"orig_{column}"))[positions].astype(str)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"Composite metadata mismatch for {name}/{column}")
    return {
        "countsCompared": checked,
        "exactCounts": True,
        "exactSourceRows": True,
        "exactMetadata": True,
        "exactFeatureIntersection": True,
        "copiedCountChunkBytes": 0,
    }


def compare_runs(left_store, left_run, right_store, right_run) -> dict:
    from scipy.linalg import subspace_angles
    from sklearn.metrics import adjusted_rand_score

    left = left_store.artifacts.load(left_run["pca"])
    right = right_store.artifacts.load(right_run["pca"])
    left_scores, right_scores = np.asarray(left["data"][:]), np.asarray(right["data"][:])
    angle = float(np.degrees(subspace_angles(left_scores, right_scores)).max())
    variance_error = float(
        np.max(
            np.abs(np.asarray(left["variance_ratio"][:]) - np.asarray(right["variance_ratio"][:]))
        )
    )
    left_labels = np.asarray(left_store.artifacts.load(left_run["clusters"])["values"][:])
    right_labels = np.asarray(right_store.artifacts.load(right_run["clusters"])["values"][:])
    ari = float(adjusted_rand_score(left_labels, right_labels))
    result = {
        "pcaMaximumSubspaceDegrees": angle,
        "explainedVarianceMaxAbsError": variance_error,
        "clusterAri": ari,
        "exactClusterLabels": bool(np.array_equal(left_labels, right_labels)),
    }
    if "harmony" in left_run and "harmony" in right_run:
        corrected_left = np.asarray(left_store.artifacts.load(left_run["harmony"])["data"][:])
        corrected_right = np.asarray(right_store.artifacts.load(right_run["harmony"])["data"][:])
        result["correctedCoordinateMaxAbsError"] = float(
            np.max(np.abs(corrected_left - corrected_right))
        )
        if result["correctedCoordinateMaxAbsError"] > 1e-5:
            raise AssertionError(f"Fresh and reused Harmony coordinates disagree: {result}")
    if angle > 0.05 or variance_error > 1e-6 or ari < 0.999:
        raise AssertionError(f"Materialized/composite downstream disagreement: {result}")
    return result


def identifier_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate_composite(payload: dict) -> dict:
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr, to_h5ad

    os.environ["CYTEARC_SOURCE_CACHE_BYTES"] = "0"
    probe = NetworkProbe()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    joint_path = work / "composite.zarr"
    memory, threads = payload.get("memBudget", "16G"), payload.get("nthreads", 8)
    source_ids = payload.get("sources", COVID_STUDIES)
    result = {
        "settings": payload,
        "sourceCacheBytes": 0,
        "timingBoundary": "Pipeline from fixed primary cells through native Leiden; UMAP, doublets, markers, cell cycle and Paris disabled",
        "countCreationPolicy": "Counts streamed for selected-gene QC and fingerprint; count chunks not copied",
        "tolerances": {
            "pcaSubspaceDegrees": 0.05,
            "explainedVarianceMaxAbsError": 1e-6,
            "clusterAriMin": 0.999,
        },
    }
    with probe_remote_reads(probe):
        source_sampler = ResourceSampler().start()
        source_started = time.perf_counter()
        sources = {
            name: cytearc.DataStore(
                did
                if "://" in did or did.startswith("/")
                else f"{CYTEBASE_RELEASE}/datasets/{did}/data.zarr",
                zarr_mode="r",
                storage_options=SOURCE_STORAGE_OPTIONS if not did.startswith("/") else None,
                min_features_per_cell=-1,
                mem_budget=memory,
                nthreads=threads,
            )
            for name, did in source_ids.items()
        }
        rows = {}
        for name, source in sources.items():
            selected = np.asarray(source.cells.fetch_all("is_primary_data"), dtype=bool)
            if payload.get("conditions", ["COVID-19", "normal"]):
                selected &= np.isin(
                    source.cells.fetch_all("disease"),
                    payload.get("conditions", ["COVID-19", "normal"]),
                )
            rows[name] = np.flatnonzero(selected)
            if payload.get("cellsPerSource"):
                rng = np.random.default_rng(4444)
                rows[name] = np.sort(
                    rng.choice(
                        rows[name], min(len(rows[name]), payload["cellsPerSource"]), replace=False
                    )
                )
            if not len(rows[name]):
                raise ValueError(f"Empty primary-cell selection in {name}")
        total_cells = sum(len(indices) for indices in rows.values())
        if payload.get("scaleOnly", False):
            locations = [
                (
                    did
                    if "://" in did or did.startswith("/")
                    else f"{CYTEBASE_RELEASE}/datasets/{did}/data.zarr"
                ).rstrip("/")
                for did in source_ids.values()
            ]
            if len(locations) != len(set(locations)):
                raise ValueError("The same source store was included more than once")
            identities = set()
            collisions = {}
            for name, source in sources.items():
                current = set(
                    np.asarray(source.cells.fetch_all("observation_joinid"), dtype=str)[rows[name]]
                )
                if len(current) != len(rows[name]):
                    raise ValueError(f"Selected source-cell IDs repeat within {name}")
                collisions[name] = len(identities.intersection(current))
                identities.update(current)
            result["uniqueSourceCellIds"] = total_cells
            result["distinctObservationJoinids"] = len(identities)
            result["crossSourceObservationJoinidCollisions"] = collisions
            result["sourceCellIdentityScope"] = (
                "Source store URI plus observation_joinid; CELLxGENE hashes only obs.index, "
                "so cross-store identifier matches can reflect reused barcode strings. "
                "Primary flags and source-study provenance define the biological eligibility."
            )
            del identities
        result["sourceOpeningAndSelection"] = {
            "elapsedSeconds": time.perf_counter() - source_started,
            "io": probe_summary(probe),
            "resources": asdict(source_sampler.stop()),
        }
        if payload.get("minimumCells", 0) > total_cells or payload.get("minimumSources", 0) > len(
            sources
        ):
            raise ValueError("Selected composite does not satisfy its declared scale")
        joint, result["creation"] = measure(
            lambda: create_composite(
                sources,
                at=joint_path,
                assay="RNA",
                features="intersection",
                rows=rows,
                mem_budget=memory,
                nthreads=threads,
            ),
            probe,
        )
        result.update(
            {
                "cellsTotalSelected": int(joint.cells.N),
                "genesShared": int(joint.RNA.feats.N),
                "sourceCellCounts": {name: len(value) for name, value in rows.items()},
                "localStoreAfterCreateBytes": dir_bytes(joint_path),
            }
        )
        result["nonzeroCounts"] = int(
            np.asarray(joint.cells.fetch_all("RNA_nFeatures"), dtype=np.int64).sum()
        )
        result["logicalDenseCountBytes"] = (
            int(joint.cells.N)
            * int(joint.RNA.feats.N)
            * int(joint.RNA.matrixGroup["counts"].dtype.itemsize)
        )
        result["copiedCountChunkBytes"] = sum(
            path.stat().st_size
            for orientation in ("counts", "countsT")
            for path in (joint_path / "RNA" / orientation).rglob("*")
            if path.is_file() and path.name != "zarr.json"
        )
        if result["copiedCountChunkBytes"]:
            raise AssertionError("Composite creation copied count chunks")
        settings = pipeline_settings(payload)
        baseline, result["unintegrated"] = measure(lambda: joint.pipeline.run(**settings), probe)
        result["unintegrated"]["summary"] = stage_summary(baseline)
        if payload.get("scaleOnly", False):
            result["localStoreAfterAnalysisBytes"] = dir_bytes(joint_path)
            if payload.get("materializedInputsPrefix"):
                import h5py

                from experiments.runtime.h5ad import read_column
                from experiments.runtime.harness import R2

                storage = R2(payload["r2EndpointUrl"])
                dataset_labels = np.asarray(joint.cells.fetch_all("dataset"), dtype=str)
                cell_ids = np.asarray(joint.cells.fetch_all("ids"), dtype=str)
                feature_hash = identifier_hash(joint.RNA.feats.fetch_all("ids"))
                inputs = []
                result["materializedInputs"] = inputs
                for name in sources:
                    selected = dataset_labels == name
                    joint.cells.insert("materialization_cells", selected, overwrite=True)
                    selection = joint.snapshot_cell_selection("materialization_cells")
                    output = work / f"{name}.h5ad"
                    _, measured_export = measure(
                        lambda: to_h5ad(
                            joint.RNA,
                            str(output),
                            cell_selection=selection,
                            embeddings_cols=[],
                            nthreads=threads,
                        ),
                        probe,
                    )
                    uri = payload["materializedInputsPrefix"].rstrip("/") + f"/{name}.h5ad"
                    cell_hash = identifier_hash(cell_ids[selected])
                    with h5py.File(output, "r") as handle:
                        if (
                            identifier_hash(
                                read_column(handle["obs"], handle["obs"].attrs["_index"])
                            )
                            != cell_hash
                        ):
                            raise AssertionError(
                                "Materialization changed source cell identity/order"
                            )
                        if (
                            identifier_hash(
                                read_column(handle["var"], handle["var"].attrs["_index"])
                            )
                            != feature_hash
                        ):
                            raise AssertionError(
                                "Materialization changed shared gene identity/order"
                            )
                    storage.upload(output, uri)
                    inputs.append(
                        {
                            "name": name,
                            "uri": uri,
                            "cells": int(selected.sum()),
                            "bytes": output.stat().st_size,
                            "cellIdSha256": cell_hash,
                            "featureIdSha256": feature_hash,
                            "export": measured_export,
                        }
                    )
                    storage.put_json(
                        payload["recordUri"].removesuffix(".json") + "/completed.json", result
                    )
                    output.unlink()
                result["materializedInputs"] = inputs
            return result
        joint.cells.insert(
            "donor_group", np.asarray(joint.cells.fetch_all("orig_donor_id")), overwrite=True
        )
        joint.cells.insert(
            "clinical_condition", np.asarray(joint.cells.fetch_all("orig_disease")), overwrite=True
        )
        joint.cells.insert(
            "integration_batch", np.asarray(joint.cells.fetch_all("dataset")), overwrite=True
        )
        integrated_settings = {**settings, "harmony_batch_columns": ["integration_batch"]}
        integrated, result["integrationReusingUpstream"] = measure(
            lambda: joint.pipeline.run(**integrated_settings), probe
        )
        result["integrationReusingUpstream"]["summary"] = stage_summary(integrated)
        result["sharedUpstreamArtifacts"] = {
            key: baseline[key].artifact_id == integrated[key].artifact_id
            for key in ("analysis_cell_selection", "highly_variable_features", "normalized", "pca")
        }
        if not all(result["sharedUpstreamArtifacts"].values()):
            raise AssertionError(
                "Adding unrelated metadata and integration invalidated upstream artifacts"
            )
        for label, run_result in (
            ("unintegrated", baseline),
            ("integrationReusingUpstream", integrated),
        ):
            result[label]["iLisiStudyScaled"] = float(
                joint.integration.compute_ilisi("dataset", run_result["neighbors"], scale=True)
            )
            result[label]["cLisiPublishedLabelsScaled"] = float(
                joint.integration.compute_clisi(
                    "orig_cell_type", run_result["neighbors"], scale=True
                )
            )
            from experiments.runtime.annotations import harmonize_cell_type

            original_labels = np.asarray(joint.cells.fetch_all("orig_cell_type"), dtype=str)
            label_map = {
                value: harmonize_cell_type("blood", value) for value in np.unique(original_labels)
            }
            harmonized = np.array([label_map[value] or "" for value in original_labels])
            neighbors = np.asarray(joint.artifacts.load(run_result["neighbors"])["indices"][:])
            result[label]["harmonizedLabelMetrics"] = neighbor_label_metrics(
                harmonized, joint.cells.fetch_all("dataset"), neighbors
            )
            result["cellTypeHarmonization"] = label_map
        fresh_path = work / "fresh.zarr"
        fresh = create_composite(
            sources,
            at=fresh_path,
            assay="RNA",
            features="intersection",
            rows=rows,
            mem_budget=memory,
            nthreads=threads,
        )
        fresh.cells.insert(
            "integration_batch", np.asarray(fresh.cells.fetch_all("dataset")), overwrite=True
        )
        fresh_run, result["integrationFresh"] = measure(
            lambda: fresh.pipeline.run(**integrated_settings), probe
        )
        result["integrationFresh"]["summary"] = stage_summary(fresh_run)
        result["freshVersusReused"] = compare_runs(joint, integrated, fresh, fresh_run)
        result["sourceAlignment"], result["sourceAlignmentMeasurement"] = measure(
            lambda: validate_source_alignment(joint, sources, rows), probe
        )
        materialized_path = work / "materialized.zarr"
        h5ad_path = work / "selected_cells.h5ad"

        def materialize():
            to_h5ad(joint.RNA, str(h5ad_path), embeddings_cols=[], nthreads=threads)
            reader = H5adReader(str(h5ad_path))
            try:
                H5adToZarr(
                    reader,
                    str(materialized_path),
                    assay_name="RNA",
                    mem_budget=memory,
                    nthreads=threads,
                ).dump()
            finally:
                reader.close()
            return cytearc.DataStore(
                str(materialized_path),
                min_features_per_cell=-1,
                mem_budget=memory,
                nthreads=threads,
            )

        materialized, result["materialization"] = measure(materialize, probe)
        materialized_run, result["materializedAnalysis"] = measure(
            lambda: materialized.pipeline.run(**settings), probe
        )
        result["materializedAgreement"] = compare_runs(
            joint, baseline, materialized, materialized_run
        )
        result["materializedH5adBytes"] = h5ad_path.stat().st_size
        result["materializedStoreBytes"] = dir_bytes(materialized_path)
        result["localStoreAfterAnalysisBytes"] = dir_bytes(joint_path)
    return result


def concatenate(payload: dict) -> dict:
    from importlib.metadata import version

    import anndata

    from experiments.runtime.harness import R2

    work = Path(payload.get("workDirectory", str(WORK / "concatenation")))
    work.mkdir(parents=True, exist_ok=True)
    storage = R2(payload["r2EndpointUrl"])
    for item in payload["inputs"]:
        storage.download(item["uri"], work / f"{item['name']}.h5ad")
    sampler = ResourceSampler().start()
    started = time.perf_counter()
    inputs = {
        item["name"]: anndata.read_h5ad(work / f"{item['name']}.h5ad") for item in payload["inputs"]
    }
    loaded = time.perf_counter()
    expected_features = next(iter(inputs.values())).var_names
    if any(not values.var_names.equals(expected_features) for values in inputs.values()):
        raise AssertionError("Materialized input gene identity/order differs across sources")
    for item in payload["inputs"]:
        values = inputs[item["name"]]
        if identifier_hash(values.obs_names) != item["cellIdSha256"]:
            raise AssertionError("A materialized input has different cell identities/order")
        if identifier_hash(values.var_names) != item["featureIdSha256"]:
            raise AssertionError("A materialized input has different gene identities/order")
    validated = time.perf_counter()
    result = anndata.concat(inputs, axis=0, join="inner", merge="first")
    finished = time.perf_counter()
    resources = asdict(sampler.stop())
    if result.n_obs != sum(item["cells"] for item in payload["inputs"]):
        raise AssertionError("Concatenation changed the selected cell count")
    if not result.obs_names.is_unique:
        raise AssertionError("Concatenation produced repeated cell identities")
    nonzero = int(result.X.nnz)
    if nonzero != int(payload["expectedNonzeroCounts"]):
        raise AssertionError("Concatenation changed the selected nonzero count total")
    if not result.var_names.equals(expected_features):
        raise AssertionError("Concatenation changed gene identities or order")
    row_offset = data_offset = 0
    for values in inputs.values():
        row_end, data_end = row_offset + values.n_obs, data_offset + values.X.nnz
        if (
            not result.obs_names[row_offset:row_end].equals(values.obs_names)
            or not np.array_equal(result.X.data[data_offset:data_end], values.X.data)
            or not np.array_equal(result.X.indices[data_offset:data_end], values.X.indices)
            or not np.array_equal(
                result.X.indptr[row_offset : row_end + 1],
                values.X.indptr.astype(np.int64, copy=False) + data_offset,
            )
        ):
            raise AssertionError("Concatenation changed a source's cells or exact count values")
        row_offset, data_offset = row_end, data_end
    return {
        "anndataVersion": version("anndata"),
        "inputLoadSeconds": loaded - started,
        "inputValidationSeconds": validated - loaded,
        "concatenateSeconds": finished - validated,
        "loadAndConcatenateSeconds": finished - started,
        "cells": int(result.n_obs),
        "genes": int(result.n_vars),
        "nonzeroCounts": nonzero,
        "csrBytes": int(result.X.data.nbytes + result.X.indices.nbytes + result.X.indptr.nbytes),
        "dtype": str(result.X.dtype),
        "indexDtype": str(result.X.indices.dtype),
        "resources": resources,
        "inputFiles": payload["inputs"],
        "exactCountValuesAndStructure": True,
        "validatedSourceSegments": len(inputs),
        "resultCellIdSha256": identifier_hash(result.obs_names),
        "resultFeatureIdSha256": identifier_hash(result.var_names),
        "resultCsrSha256": {
            name: hashlib.sha256(memoryview(getattr(result.X, name)).cast("B")).hexdigest()
            for name in ("data", "indices", "indptr")
        },
    }


def process_io_bytes() -> dict | None:
    """Bytes this process has read and written (/proc/self/io), or None where it is unavailable."""
    try:
        text = Path("/proc/self/io").read_text()
    except OSError:
        return None
    values = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
    return {
        key: int(values[key])
        for key in ("rchar", "wchar", "read_bytes", "write_bytes")
        if key in values
    }


def measure_local(operation, *, record_failure: bool = False) -> tuple[object, dict]:
    """Time a local-disk operation with sampled container memory and this process's I/O counters.

    With record_failure, an exception from the operation is returned as the measurement's status
    and error instead of propagating; the failure is the measured outcome.
    """
    before = process_io_bytes()
    sampler = ResourceSampler().start()
    started = time.perf_counter()
    value, failure = None, None
    try:
        value = operation()
    except Exception as error:
        if not record_failure:
            sampler.stop()
            raise
        failure = error
    resources = asdict(sampler.stop())
    elapsed = time.perf_counter() - started
    after = process_io_bytes()
    io = (
        None
        if before is None or after is None
        else {key: after[key] - before[key] for key in after}
    )
    measured = {
        "status": "completed" if failure is None else "failed",
        "elapsedSeconds": elapsed,
        "processIoBytes": io,
        "resources": resources,
    }
    if failure is not None:
        measured["errorType"] = type(failure).__name__
        measured["error"] = " ".join([str(failure), *getattr(failure, "__notes__", [])])[-2000:]
    return value, measured


def obs_column_kind(dtype) -> str:
    import pandas as pd

    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    return "string"


ANNDATA_MAPPINGS = ("layers", "obsm", "obsp", "varm", "varp", "uns")


def prepare_for_on_disk_concatenation(paths: dict) -> dict:
    """Make the input copies readable and writable by concat_on_disk; the counts are untouched.

    concat_on_disk reads every AnnData mapping group, so mappings the export left out are added
    empty. pandas concatenates obs columns whose kind differs between inputs (for example strings
    in one study and an empty categorical in another) into object columns holding strings and NaN,
    which AnnData cannot write, so those columns are stored as string categories in every input,
    with missing values kept missing.
    """
    import anndata
    import h5py
    import pandas as pd

    added: dict[str, list[str]] = {}
    kinds: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        with h5py.File(path, "r+") as handle:
            added[name] = [key for key in ANNDATA_MAPPINGS if key not in handle]
            for key in added[name]:
                anndata.io.write_elem(handle, key, {})
            frame = anndata.io.read_elem(handle["obs"])
        for column, dtype in frame.dtypes.items():
            kinds.setdefault(column, {})[name] = obs_column_kind(dtype)
    mixed = sorted(column for column, by_input in kinds.items() if len(set(by_input.values())) > 1)
    for path in paths.values() if mixed else ():
        with h5py.File(path, "r+") as handle:
            frame = anndata.io.read_elem(handle["obs"])
            for column in mixed:
                if column in frame:
                    values = frame[column]
                    strings = values.astype(object).map(str).where(values.notna(), None)
                    frame[column] = pd.Categorical(strings)
            del handle["obs"]
            anndata.io.write_elem(handle, "obs", frame)
    return {
        "addedEmptyMappings": added,
        "obsColumnsChecked": len(kinds),
        "obsColumnsStoredAsStringCategories": mixed,
        "obsKindsByInput": {column: kinds[column] for column in mixed},
    }


def validate_on_disk_concatenation(
    output: Path, inputs: list[dict], paths: dict, expected_nonzero: int, block_values: int
) -> dict:
    """Check that the concatenated CSR holds every input's cells, genes and exact counts in order."""
    import anndata
    import h5py

    def index(frame) -> np.ndarray:
        return np.asarray(anndata.io.read_elem(frame[frame.attrs["_index"]]), dtype=str)

    with h5py.File(output, "r") as target:
        matrix = target["X"]
        if matrix.attrs["encoding-type"] != "csr_matrix":
            raise AssertionError(f"Concatenated X is {matrix.attrs['encoding-type']}, not CSR")
        cells, genes = (int(value) for value in matrix.attrs["shape"])
        data, indices, indptr = matrix["data"], matrix["indices"], matrix["indptr"]
        if data.shape[0] != expected_nonzero or indices.shape[0] != expected_nonzero:
            raise AssertionError("Concatenation changed the selected nonzero count total")
        cell_ids, feature_ids = index(target["obs"]), index(target["var"])
        if identifier_hash(feature_ids) != inputs[0]["featureIdSha256"]:
            raise AssertionError("Concatenation changed gene identities or order")
        row = offset = 0
        for item in inputs:
            with h5py.File(paths[item["name"]], "r") as source:
                part = source["X"]
                shape = tuple(int(value) for value in part.attrs["shape"])
                if shape != (item["cells"], genes):
                    raise AssertionError(
                        f"{item['name']} has shape {shape}, expected {(item['cells'], genes)}"
                    )
                source_indptr = part["indptr"][:].astype(np.int64)
                target_indptr = indptr[row : row + shape[0] + 1].astype(np.int64)
                if source_indptr[0] != 0 or not np.array_equal(
                    target_indptr - offset, source_indptr
                ):
                    raise AssertionError(f"Row pointers of {item['name']} changed")
                nonzero = int(source_indptr[-1])
                for start in range(0, nonzero, block_values):
                    end = min(start + block_values, nonzero)
                    window = slice(offset + start, offset + end)
                    if not np.array_equal(
                        part["data"][start:end], data[window]
                    ) or not np.array_equal(part["indices"][start:end], indices[window]):
                        raise AssertionError(
                            f"Counts of {item['name']} changed in values {start}-{end}"
                        )
            if identifier_hash(cell_ids[row : row + shape[0]]) != item["cellIdSha256"]:
                raise AssertionError(f"Cell identities or order of {item['name']} changed")
            row, offset = row + shape[0], offset + nonzero
        if (row, offset) != (cells, expected_nonzero):
            raise AssertionError("The inputs do not account for every concatenated cell and value")
        return {
            "cells": cells,
            "genes": genes,
            "nonzeroCounts": offset,
            "dtypes": {name: str(matrix[name].dtype) for name in ("data", "indices", "indptr")},
            "compression": {
                name: matrix[name].compression for name in ("data", "indices", "indptr")
            },
            "chunks": {name: matrix[name].chunks for name in ("data", "indices", "indptr")},
            "obsColumns": len(target["obs"].attrs["column-order"]),
            "varColumns": len(target["var"].attrs["column-order"]),
            "exactCountValuesAndStructure": True,
            "validatedSourceSegments": len(inputs),
            "resultCellIdSha256": identifier_hash(cell_ids),
            "resultFeatureIdSha256": identifier_hash(feature_ids),
        }


def add_feature_names(path: Path, source_column: str = "gene_short_name") -> dict:
    """Give the concatenated var the CELLxGENE feature_name column the Scanpy workflow reads."""
    import anndata
    import h5py

    with h5py.File(path, "r+") as handle:
        var = anndata.io.read_elem(handle["var"])
        if "feature_name" in var.columns:
            return {"featureName": "present in the inputs", "genes": len(var)}
        if source_column not in var.columns:
            raise KeyError(f"var has neither feature_name nor {source_column}")
        var["feature_name"] = var[source_column].copy()
        del handle["var"]
        anndata.io.write_elem(handle, "var", var)
    return {"featureName": f"copied from var.{source_column}", "genes": len(var)}


def concatenate_on_disk(payload: dict) -> dict:
    """Concatenate the matched H5AD inputs with anndata.experimental.concat_on_disk.

    The inputs and their order, the join and the merge rule are those of the in-memory baseline
    (concatenate). The inputs are first concatenated as exported; if that fails, the failure is
    recorded, the input copies are prepared (prepare_for_on_disk_concatenation) and the
    concatenation is repeated. Staging, preparation, validation, the feature-name column the
    Scanpy workflow reads and the upload are measured separately from the concatenation.
    """
    from importlib.metadata import version

    from anndata.experimental import concat_on_disk

    if version("anndata") != payload["anndataVersion"]:
        raise RuntimeError(
            f"anndata {version('anndata')} differs from the requested {payload['anndataVersion']}"
        )
    inputs = payload["inputs"]
    names = [item["name"] for item in inputs]
    if not inputs or len(set(names)) != len(names):
        raise ValueError("Concatenation requires named, distinct inputs")
    on_r2 = [item["uri"].startswith("s3://") for item in inputs]
    remote = all(on_r2)
    if any(on_r2) != remote or not (remote or all(Path(i["uri"]).is_absolute() for i in inputs)):
        raise ValueError("Inputs must all be s3:// objects or all be absolute local paths")
    work = Path(payload.get("workDirectory", str(WORK / "concatenation_on_disk")))
    work.mkdir(parents=True, exist_ok=True)
    output = work / "concatenated.h5ad"
    if output.exists():
        raise FileExistsError(f"{output} exists; the measurement requires a fresh output file")
    storage = None
    if remote or payload.get("outputUri"):
        from experiments.runtime.harness import R2

        storage = R2(payload["r2EndpointUrl"])
    paths = {item["name"]: work / "inputs" / f"{item['name']}.h5ad" for item in inputs}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"{work / 'inputs'} holds inputs from an earlier run")
    max_loaded = int(payload.get("maxLoadedElems", 100_000_000))
    started = time.perf_counter()
    result = {
        "anndataVersion": version("anndata"),
        "h5pyVersion": version("h5py"),
        "function": "anndata.experimental.concat_on_disk",
        "join": "inner",
        "merge": "first",
        "maxLoadedElems": max_loaded,
        "inputs": [{key: item[key] for key in ("name", "cells", "bytes")} for item in inputs],
        "timingBoundary": "concatenation: concat_on_disk from the local input copies to one local "
        "H5AD; staging, input preparation, validation, the feature-name column and upload "
        "are measured separately",
    }
    staging_started = time.perf_counter()
    for item in inputs:
        paths[item["name"]].parent.mkdir(parents=True, exist_ok=True)
        if remote:
            storage.download(item["uri"], paths[item["name"]])
        else:
            shutil.copyfile(item["uri"], paths[item["name"]])
    result["staging"] = {
        "seconds": time.perf_counter() - staging_started,
        "bytes": sum(path.stat().st_size for path in paths.values()),
        "source": "r2" if remote else "local copy",
    }
    for item in inputs:
        if paths[item["name"]].stat().st_size != item["bytes"]:
            raise AssertionError(f"{item['name']} differs in size from the exported input")

    def concatenate_to(target: Path) -> None:
        concat_on_disk(
            {name: str(path) for name, path in paths.items()},
            str(target),
            max_loaded_elems=max_loaded,
            axis=0,
            join="inner",
            merge="first",
        )

    as_exported = work / "concatenated_as_exported.h5ad"
    _, result["asExportedAttempt"] = measure_local(
        lambda: concatenate_to(as_exported), record_failure=True
    )
    if result["asExportedAttempt"]["status"] == "completed":
        os.replace(as_exported, output)
        result["concatenation"] = result["asExportedAttempt"]
    else:
        as_exported.unlink(missing_ok=True)
        result["inputPreparation"], result["inputPreparationMeasurement"] = measure_local(
            lambda: prepare_for_on_disk_concatenation(paths)
        )
        _, result["concatenation"] = measure_local(lambda: concatenate_to(output))
    stat = output.stat()
    result["outputBytes"], result["outputAllocatedBytes"] = stat.st_size, stat.st_blocks * 512
    result["validation"], result["validationMeasurement"] = measure_local(
        lambda: validate_on_disk_concatenation(
            output,
            inputs,
            paths,
            int(payload["expectedNonzeroCounts"]),
            int(payload.get("validationBlockValues", 50_000_000)),
        )
    )
    result["analysisInput"], result["analysisInputMeasurement"] = measure_local(
        lambda: add_feature_names(output)
    )
    if payload.get("outputUri"):
        upload_started = time.perf_counter()
        storage.upload(output, payload["outputUri"])
        result["upload"] = {
            "seconds": time.perf_counter() - upload_started,
            "bytes": output.stat().st_size,
            "uri": payload["outputUri"],
        }
    result["jobSeconds"] = time.perf_counter() - started
    return result


def run(payload: dict) -> dict:
    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    mode = payload.get("mode", "composite")
    if mode == "remote":
        sampler = ResourceSampler().start()
        try:
            result = {"remoteQueries": evaluate_remote_queries(payload)}
        finally:
            resources = asdict(sampler.stop())
        result["resources"] = resources
    elif mode == "composite":
        result = {"composite": evaluate_composite(payload)}
    elif mode == "concatenate":
        result = {"materializedConcatenation": concatenate(payload)}
    elif mode == "concatenateOnDisk":
        result = {"onDiskConcatenation": concatenate_on_disk(payload)}
    else:
        raise ValueError(f"Unknown experiment mode {mode!r}")
    result["sourceStorageOptions"] = SOURCE_STORAGE_OPTIONS
    write_json("composite_and_remote.json", result, directory=work)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    run(json.loads(args.payload.read_text()))


if __name__ == "__main__":
    main()
