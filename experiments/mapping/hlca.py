import argparse
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

from experiments.mapping import DATA

CORE_URL = "https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"


def selected_counts(matrix, rows, feature_indices=None):
    import numpy as np
    from scipy.sparse import csr_matrix, vstack

    if matrix.attrs["encoding-type"] != "csr_matrix":
        raise ValueError("The pinned source must use CSR counts")
    pointers = np.asarray(matrix["indptr"][:], dtype=np.int64)
    genes = int(matrix.attrs["shape"][1])
    pieces = []
    start = 0
    while start < len(rows):
        stop = start + 1
        first = int(rows[start])
        while stop < len(rows) and pointers[rows[stop] + 1] - pointers[first] <= 2_000_000:
            stop += 1
        last = int(rows[stop - 1])
        left, right = int(pointers[first]), int(pointers[last + 1])
        block = csr_matrix(
            (
                np.asarray(matrix["data"][left:right]),
                np.asarray(matrix["indices"][left:right]),
                pointers[first : last + 2] - left,
            ),
            shape=(last - first + 1, genes),
        )
        chosen = block[rows[start:stop] - first]
        pieces.append(chosen if feature_indices is None else chosen[:, feature_indices])
        start = stop
    return vstack(pieces, format="csr")


def prepare_stores(payload: dict, work: Path, probes: dict) -> tuple[object, object, dict]:
    import cytearc
    import h5py
    import numpy as np
    from cytearc.writers import SparseToZarr
    from zarr.storage import LocalStore

    from experiments.runtime.h5ad import read_column
    from experiments.runtime.object_io import _download
    from experiments.runtime.recording_store import RecordingStoreWrapper

    source_path = Path(payload.get("sourcePath", str(work / "hlca_core.h5ad")))
    if not source_path.exists():
        _download(payload.get("sourceUrl", CORE_URL), source_path)
    memory, threads = payload.get("memBudget", "8G"), payload.get("nthreads", 4)
    with h5py.File(source_path, "r") as source:
        obs = source["obs"]
        studies = read_column(obs, payload.get("studyColumn", "dataset"))
        donors = read_column(obs, payload.get("donorColumn", "donor_id"))
        labels = read_column(obs, payload.get("labelColumn", "ann_level_3"))
        ids = read_column(obs, obs.attrs["_index"])
        query_study = payload.get("queryStudy", "Krasnow_2020")
        if query_study not in studies:
            raise ValueError(f"Held-out study {query_study} is absent")
        names, sizes = np.unique(studies, return_counts=True)
        reference_studies = payload.get("referenceStudies")
        if reference_studies is None:
            reference_studies = [
                str(names[i]) for i in np.argsort(-sizes) if names[i] != query_study
            ][:3]
        if query_study in reference_studies:
            raise ValueError("A query study cannot enter reference fitting")
        valid_label = ~np.isin(np.char.lower(labels.astype(str)), ["", "nan", "unknown", "none"])
        query_rows = np.flatnonzero((studies == query_study) & valid_label)
        query_donors = set(donors[query_rows])
        reference_mask = (
            np.isin(studies, reference_studies) & valid_label & ~np.isin(donors, list(query_donors))
        )
        rng = np.random.default_rng(4444)
        reference_rows = []
        for study in reference_studies:
            rows = np.flatnonzero(reference_mask & (studies == study))
            if len(rows) < 50:
                raise ValueError(f"Reference study {study} has fewer than 50 eligible cells")
            reference_rows.extend(
                rng.choice(
                    rows, min(len(rows), payload.get("referenceCellsPerStudy", 2000)), replace=False
                ).tolist()
            )
        reference_rows = np.sort(np.asarray(reference_rows, dtype=np.int64))
        query_rows = np.sort(
            rng.choice(
                query_rows, min(len(query_rows), payload.get("queryCells", 1000)), replace=False
            )
        )
        if set(donors[reference_rows]) & set(donors[query_rows]):
            raise AssertionError("Reference and query donors overlap")
        matrix_key = "raw/X" if "raw/X" in source else "X"
        variable = source["raw/var"] if matrix_key == "raw/X" else source["var"]
        feature_ids = read_column(variable, variable.attrs["_index"])
        name_key = next(
            (key for key in ("feature_name", "gene_short_name", "names") if key in variable), None
        )
        if name_key is None:
            raise ValueError("Source has no gene-name metadata")
        feature_names = read_column(variable, name_key)
        stores = {}
        for name, rows in (("reference", reference_rows), ("query", query_rows)):
            counts = selected_counts(source[matrix_key], rows)
            path = work / f"{name}.zarr"
            SparseToZarr(
                counts,
                str(path),
                ids[rows],
                feature_ids,
                feature_names=feature_names,
                mem_budget=memory,
                nthreads=threads,
            ).dump()
            del counts
            ds = cytearc.DataStore(
                RecordingStoreWrapper(LocalStore(str(path)), probe=probes[name]),
                min_features_per_cell=-1,
                mem_budget=memory,
                nthreads=threads,
            )
            for key, values in (("study", studies), ("donor_id", donors), ("cell_type", labels)):
                ds.cells.insert(key, values[rows], overwrite=True)
            stores[name] = ds
    return (
        stores["reference"],
        stores["query"],
        {
            "sourceUrl": payload.get("sourceUrl", payload.get("sourcePath", CORE_URL)),
            "matrixKey": matrix_key,
            "referenceStudies": reference_studies,
            "queryStudy": query_study,
            "referenceCells": len(reference_rows),
            "queryCells": len(query_rows),
            "referenceDonors": sorted(set(donors[reference_rows])),
            "queryDonors": sorted(set(donors[query_rows])),
            "referenceQueryDonorOverlap": [],
            "labelColumn": payload.get("labelColumn", "ann_level_3"),
            "selectionSeed": 4444,
        },
    )


def hlca_job(payload: dict) -> dict:
    if "workers" in payload or "budgetBytes" in payload:
        raise ValueError("Specify nthreads and memBudget for the fixed released-package protocol")
    from dataclasses import asdict

    import numpy as np
    import pandas as pd
    from cytearc.mapping.features import AlignedFeatureStream
    from cytearc.mapping.symphony import (
        accumulate_sufficient_statistics,
        apply_query_correction,
        initialize_sufficient_statistics,
        project_pca,
        soft_cluster_assignments,
        solve_query_correction,
    )
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    from experiments.runtime.common import configure_cytearc, write_json
    from experiments.runtime.harness import R2
    from experiments.runtime.metrics import ResourceSampler
    from experiments.runtime.recording_store import StoreProbe
    from experiments.runtime.steps import _Steps

    configure_cytearc()
    work = Path(payload.get("workDirectory", "/tmp/lung_heldout_query_mapping"))
    work.mkdir(parents=True, exist_ok=True)
    result = {
        "evaluationPhase": payload.get("evaluationPhase", "development"),
        "settings": payload,
        "tolerances": {"projectedMaxAbs": 1e-8, "assignmentsMaxAbs": 1e-6, "correctedMaxAbs": 1e-5},
    }
    steps = _Steps()
    probes = {name: StoreProbe() for name in ("reference", "query")}
    reference_store, query_store, result["dataset"] = steps.run(
        "prepareSelectedStores", lambda: prepare_stores(payload, work, probes)
    )
    sampler = ResourceSampler().start()
    settings = {
        "filtering": False,
        "hvg_count": payload.get("hvgCount", 2000),
        "pca_dims": payload.get("pcaDims", 30),
        "neighbors_k": 30,
        "umap": False,
        "leiden": False,
        "cell_cycle": False,
        "paris": False,
        "doublets": False,
        "markers": False,
        "params": {
            "ann_index": {"ann_parallel": False},
            "harmony": {
                "batch_columns": ["study"],
                "harmony_params": {"sigma": 0.1, "random_state": 4444},
            },
        },
    }
    reference_run = steps.run(
        "referenceHarmony2Pipeline", lambda: reference_store.pipeline.run(**settings)
    )
    probes["reference"].reset()
    ref = steps.run(
        "registerReference",
        lambda: reference_store.mapping.load_reference(
            reference_store.mapping.build_reference(reference_run["neighbors"])
        ),
    )
    if ref.symphony_state is None:
        raise AssertionError("Harmony reference registration omitted Symphony correction state")
    result["registrationIo"] = probes["reference"].to_json()
    if result["registrationIo"]["countReadBytes"]:
        raise AssertionError("Registration read reference count chunks")
    probes["reference"].reset()
    probes["query"].reset()
    cells = query_store.snapshot_cell_selection("I")
    query_batch = pd.DataFrame({"study": np.asarray(query_store.cells.fetch_all("study"))})
    projected_ref = steps.run(
        "projectCorrectSearch",
        lambda: query_store.mapping.run(
            ref, cells, save_k=30, query_batches=query_batch, missing_feature_policy="error"
        ),
    )
    transfer = steps.run(
        "transferLabels",
        lambda: query_store.mapping.load_label_transfer(
            query_store.mapping.transfer_labels(
                projected_ref, reference=ref, reference_labels="cell_type", threshold_fraction=0.0
            )
        ),
    )
    result["resources"] = asdict(sampler.stop())
    result["queryAnnotationIo"] = {name: probe.to_json() for name, probe in probes.items()}
    truth = np.asarray(query_store.cells.fetch_all("cell_type"), dtype=str)
    predicted = transfer.labels.fillna("unassigned").to_numpy(dtype=str)
    result["annotation"] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macroF1": float(f1_score(truth, predicted, average="macro")),
        "perType": classification_report(truth, predicted, output_dict=True, zero_division=0),
        "assignmentCoverage": float(np.mean(predicted != "unassigned")),
    }
    stream = AlignedFeatureStream(
        query_assay=query_store.RNA,
        query_cell_indices=np.arange(query_store.cells.N),
        reference_feature_ids=ref.feature_ids,
        reference_normalized_means=ref.model.feature_means,
        reference_normalization_parameters=ref.normalization_parameters,
        missing_feature_policy="error",
        resources=query_store.resources,
    )
    blocks = list(stream.iter_blocks())
    informative = np.concatenate([block.observed for block in blocks])
    values = np.concatenate([block.values for block in blocks])[informative]
    result["symphonyExcludedUninformativeCells"] = int((~informative).sum())
    pca = project_pca(values, ref.model)
    model = ref.symphony_state
    assignments = soft_cluster_assignments(pca, model)
    codes = np.zeros(len(pca), dtype=np.int64)
    counts, sums = initialize_sufficient_statistics(1, model)
    accumulate_sufficient_statistics(counts, sums, pca, assignments, codes)
    corrected = apply_query_correction(
        pca, assignments, codes, model, solve_query_correction(counts, sums, model)
    )
    if not np.allclose(model.sigma, 0.1, atol=0, rtol=0):
        raise AssertionError("R mapQuery comparison requires the fixed common sigma")
    numeric = {
        "query": values.tolist(),
        "featureIds": [str(x) for x in ref.feature_ids],
        "featureMeans": (
            ref.model.feature_means + ref.model.center * ref.model.feature_scales
        ).tolist(),
        "featureScales": ref.model.feature_scales.tolist(),
        "loadings": ref.model.loadings.tolist(),
        "centroids": model.centroids.tolist(),
        "clusterMass": model.cluster_mass.tolist(),
        "correctedCentroids": model.corrected_centroids.tolist(),
        "sigma": 0.1,
        "projected": pca.tolist(),
        "assignments": assignments.tolist(),
        "corrected": corrected.tolist(),
        "queryBatch": query_batch["study"].to_numpy()[informative].tolist(),
        "tolerances": result["tolerances"],
    }
    numerical_path = work / "symphony.json.gz"
    with gzip.open(numerical_path, "wt") as handle:
        json.dump(numeric, handle)
    result["symphonyInputBytes"] = numerical_path.stat().st_size
    result["correctionMagnitudeMaxAbs"] = float(np.max(np.abs(corrected - pca)))
    result["correctionMethod"] = query_store.mapping.load_result(
        projected_ref, reference=ref
    ).correction_method
    result["steps"] = steps.records
    result["referencePipeline"] = reference_run.report()
    if payload.get("recordUri"):
        input_uri = payload["recordUri"].removesuffix(".json") + "/symphony.json.gz"
        R2(payload["r2EndpointUrl"]).upload(numerical_path, input_uri)
        result["symphonyInputUri"] = input_uri
    write_json("hlca.json", result, directory=DATA)
    return result


def r_symphony(payload: dict) -> dict:
    from experiments.runtime.harness import R2

    work = Path(payload.get("workDirectory", "/tmp/symphony"))
    work.mkdir(parents=True, exist_ok=True)
    source = work / "symphony.json.gz"
    destination = work / "symphony_result.json"
    R2(payload["r2EndpointUrl"]).download(payload["inputUri"], source)
    subprocess.run(
        [
            "Rscript",
            str(Path(__file__).parent / "tools" / "symphony.R"),
            str(source),
            str(destination),
        ],
        check=True,
    )
    result = json.loads(destination.read_text())
    with source.open("rb") as handle:
        result["inputSha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    result["inputUri"] = payload["inputUri"]
    return result


def run(payload: dict) -> dict:
    from experiments.runtime.common import provenance
    from experiments.runtime.harness import R2

    os.environ["R2_ENDPOINT_URL"] = payload.get("r2EndpointUrl", "")
    mode = payload.get("mode", "mapping")
    storage = R2(payload["r2EndpointUrl"]) if payload.get("recordUri") else None
    try:
        if mode == "mapping":
            result = hlca_job(payload)
        elif mode == "rSymphony":
            result = r_symphony(payload)
        else:
            raise ValueError(f"Unknown mapping mode {mode!r}")
    except Exception as error:
        if storage:
            storage.put_json(
                payload["recordUri"],
                {
                    "status": "failed",
                    "provenance": provenance(),
                    "errorType": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
    if storage:
        storage.put_json(
            payload["recordUri"],
            {"status": "completed", "provenance": provenance(), "result": result},
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    run(json.loads(parser.parse_args().payload.read_text()))
