import gc
import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

import cytearc
import numpy as np
import zarr
from scipy import sparse
from scipy.linalg import subspace_angles, svd
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.methods import DATA as RESULTS
from experiments.methods import INPUTS, WORK
from experiments.runtime.common import (
    catalogue_store,
    configure_cytearc,
    provenance,
    scrub_private,
    write_json,
)
from experiments.runtime.metrics import ResourceSampler

PANCREAS = "bastidas-ponce_4K_pancreas-d15_rnaseq"
KANG = "kang_29K_ctrl-ifnb_pbmc_rnaseq"
PCA_TOLERANCES = {"maxSubspaceAngleDegrees": 0.05, "maxRelativeVarianceShareError": 0.0005}
DOUBLET_PARAMETERS = {
    "cluster_sample_fraction": 1.0,
    "max_cells_per_cluster": None,
    "simulation_ratio": 2.0,
    "heterotypic_fraction": 0.0,
    "save_k": 30,
    "expected_doublet_rate": 0.05,
    "smoothing_t": 0,
    "normalize_scores": False,
    "random_seed": 4444,
}
XI_LI_DATASETS = (
    "HEK-HMEC-MULTI",
    "HMEC-orig-MULTI",
    "HMEC-rep-MULTI",
    "J293t-dm",
    "cline-ch",
    "hm-12k",
    "hm-6k",
    "mkidney-ch",
    "nuc-MULTI",
    "pbmc-1A-dm",
    "pbmc-1B-dm",
    "pbmc-1C-dm",
    "pbmc-2ctrl-dm",
    "pbmc-2stim-dm",
    "pbmc-ch",
    "pdx-MULTI",
)


def measured(operation):
    sampler = ResourceSampler().start()
    started = time.perf_counter()
    try:
        value = operation()
    finally:
        elapsed = time.perf_counter() - started
        memory = asdict(sampler.stop())
    return value, {"seconds": elapsed, **memory}


def checkpoint(payload, result):
    if payload.get("recordUri"):
        from experiments.runtime.harness import R2

        R2(payload["r2EndpointUrl"]).put_json(
            f"{payload['recordUri'].removesuffix('.json')}/completed.json",
            result,
        )


def prepare_store(
    dataset: str,
    work: Path,
    workers: int,
    cell_limit: int | None = None,
    memory_budget: str = "4G",
):
    from cytearc.storage.types import complete_chunks

    source = catalogue_store(dataset, destination=INPUTS) / "data.zarr"
    root = zarr.open_group(source, mode="r", use_consolidated=False)
    matrix = complete_chunks(root["RNA/counts"])
    n = matrix.shape[0] if cell_limit is None else min(cell_limit, matrix.shape[0])
    rows = np.sort(np.random.default_rng(4444).choice(matrix.shape[0], n, replace=False))
    cells = np.asarray(root["cellData/ids"][:])[rows].astype(str)
    genes = np.asarray(root["RNA/featureData/ids"][:]).astype(str)
    names = np.asarray(root["RNA/featureData/names"][:]).astype(str)
    blocks = [sparse.csr_matrix(matrix.oindex[rows[i : i + 256], :]) for i in range(0, n, 256)]
    counts = sparse.vstack(blocks, format="csr")
    del blocks
    path = work / f"{dataset}.zarr"
    if path.exists():
        raise FileExistsError(path)
    cytearc.SparseToZarr(
        counts,
        str(path),
        cells,
        genes,
        feature_names=names,
        assay_name="RNA",
        assay_type="RNA",
        nthreads=workers,
        mem_budget=memory_budget,
    ).dump()
    del counts
    ds = cytearc.DataStore(
        str(path), default_assay="RNA", nthreads=workers, mem_budget=memory_budget
    )
    if not np.array_equal(ds.cells.fetch_all("ids").astype(str), cells):
        raise RuntimeError("Cell identity changed while preparing count-only input")
    return path, {
        "dataset": dataset,
        "cells": n,
        "features": len(genes),
        "cellIdsSha256": hashlib.sha256("\n".join(cells).encode()).hexdigest(),
        "featureIdsSha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "source": str(source),
        "countsOnly": True,
        "cellLimit": cell_limit,
        "memoryBudget": memory_budget,
    }


def pipeline(ds, *, parallel=False, hvg_count=2000, pca_dims=21, umap=True):
    return ds.pipeline.run(
        filtering=True,
        hvg_count=hvg_count,
        pca_dims=pca_dims,
        neighbors_k=15,
        umap=umap,
        paris=False,
        cell_cycle=False,
        doublets=False,
        markers=False,
        params={
            "ann_index": {"ann_parallel": parallel, "rand_state": 4466},
            "leiden": {
                "partitions": [1.0],
                "selected": 1.0,
                "backend": "cytearc",
                "random_seed": 4444,
            },
            **(
                {"umap": {"parallel": parallel, "random_seed": 4444, "n_epochs": 300}}
                if umap
                else {}
            ),
        },
    )


def evaluate_leiden_determinism(ds, graph_ref, *, repetitions=3):
    import igraph
    from cytearc.clustering.leiden import leiden_membership

    group = ds.artifacts.load(graph_ref)
    edges = np.asarray(group["edges"][:], dtype=np.int64)
    weights = np.asarray(group["weights"][:], dtype=np.float64)
    n = len(
        ds.artifacts.load(ds.pipeline.list_runs(limit=1)[0]["analysis_cell_selection"])["values"][
            :
        ].nonzero()[0]
    )
    graph = sparse.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    objective_graph = igraph.Graph(n=n, edges=list(map(tuple, edges)), directed=False)
    reference = None
    runs = []
    for backend, threads in [("cytearc", t) for t in (1, 4, 8, 16)] + [("igraph", 1)]:
        leiden_membership(graph, 1.0, 4444, backend=backend, nthreads=threads)
        for repetition in range(repetitions):
            labels, timing = measured(
                lambda: leiden_membership(graph, 1.0, 4444, backend=backend, nthreads=threads)
            )
            if reference is None:
                reference = labels.copy()
            runs.append(
                {
                    "backend": backend,
                    "threads": threads,
                    "repetition": repetition + 1,
                    "clusters": int(np.unique(labels).size),
                    "modularity": objective_graph.modularity(
                        labels - 1, weights=weights, resolution=1.0
                    ),
                    "exactMatchToNativeOneThread": bool(np.array_equal(reference, labels)),
                    "ariToNativeOneThread": float(adjusted_rand_score(reference, labels)),
                    **timing,
                }
            )
    native = [r for r in runs if r["backend"] == "cytearc"]
    comparator = [r for r in runs if r["backend"] == "igraph"]
    return {
        "cells": n,
        "storedEdges": len(edges),
        "runs": runs,
        "timingBoundary": "In-memory Leiden fit, excluding graph load and one warm-up per setting",
        "allNativePartitionsIdentical": all(r["exactMatchToNativeOneThread"] for r in native),
        "nativeMinusIgraphModularity": native[0]["modularity"] - comparator[0]["modularity"],
        "modularityParityTolerance": 0.01,
    }


def evaluate_pipeline_determinism(payload, work):
    workers = int(payload.get("workers", 16))
    memory_budget = payload.get("memBudget", "4G")
    path, source = prepare_store(
        payload.get("dataset", KANG), work, workers, payload.get("cellLimit"), memory_budget
    )
    arrays_by_mode = {}
    reports = []
    leiden = None
    for parallel in (False, True):
        mode = "parallel" if parallel else "deterministic"
        arrays_by_mode[mode] = []
        for repetition in range(int(payload.get("repetitions", 3))):
            destination = work / f"pipeline_{mode}_{repetition + 1}.zarr"
            shutil.copytree(path, destination)
            ds = cytearc.DataStore(
                str(destination), default_assay="RNA", nthreads=workers, mem_budget=memory_budget
            )
            run, timing = measured(
                lambda store=ds: pipeline(
                    store,
                    parallel=parallel,
                    hvg_count=int(payload.get("hvgCount", 2000)),
                    pca_dims=int(payload.get("pcaDims", 21)),
                )
            )
            report = run.report()
            if report["summary"]["reusedArtifacts"]:
                raise RuntimeError("Fresh pipeline unexpectedly reused a stored artifact")
            keys = {
                "analysis_cell_selection": "values",
                "highly_variable_features": "values",
                "normalized": "data",
                "pca": "data",
                "neighbors": "indices",
                "umap": "values",
                "clusters": "values",
            }
            arrays = {
                key: np.asarray(ds.artifacts.load(run[key])[value][:])
                for key, value in keys.items()
            }
            arrays_by_mode[mode].append(arrays)
            reports.append(
                {
                    "mode": mode,
                    "repetition": repetition + 1,
                    "config": report["run"]["config"],
                    "stages": report["stages"],
                    "arrayDigests": {
                        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                        for key, value in arrays.items()
                    },
                    **timing,
                }
            )
            checkpoint(payload, {"source": source, "completedRuns": reports})
            if not parallel and repetition == 0:
                leiden = evaluate_leiden_determinism(
                    ds, run["connectivity_map"], repetitions=int(payload.get("repetitions", 3))
                )
            del ds
            gc.collect()
            shutil.rmtree(destination)
    comparisons = []
    for mode, arrays in arrays_by_mode.items():
        for (i, first), (j, second) in combinations(enumerate(arrays), 2):
            comparisons.append(
                {
                    "mode": mode,
                    "first": i + 1,
                    "second": j + 1,
                    "exactArrayEquality": {
                        key: bool(np.array_equal(first[key], second[key])) for key in first
                    },
                    "clusterARI": float(adjusted_rand_score(first["clusters"], second["clusters"])),
                    "pcaMaxAbsoluteDifference": float(np.max(np.abs(first["pca"] - second["pca"]))),
                    "umapMaxAbsoluteDifference": float(
                        np.max(np.abs(first["umap"] - second["umap"]))
                    ),
                }
            )
    return {
        "source": source,
        "pipelineRuns": reports,
        "comparisons": comparisons,
        "nativeLeiden": leiden,
        "timingBoundary": "Complete fresh filtering, HVGs, normalization, PCA, ANN, neighbors, graph, UMAP, Leiden",
        "deterministicTarget": "Exact equality of all saved arrays and ARI 1.0 with reuse disabled",
    }


def evaluate_operation_revision(values, work):
    from cytearc.storage.artifact_writer import artifact_transaction, plan_artifact
    from cytearc.storage.artifacts import inspect_artifact, make_provenance
    from sklearn.decomposition import IncrementalPCA

    root = zarr.open_group(work / "pca_revisions.zarr", mode="w")
    results = []
    for blocks in (1, 4):
        model = IncrementalPCA(n_components=10, batch_size=max(11, len(values) // blocks)).fit(
            values
        )
        parameters = {
            "dims": 10,
            **({"incremental_block_rows": len(values) // blocks} if blocks > 1 else {}),
        }
        planned = plan_artifact(
            root,
            scope="assay",
            assay="RNA",
            kind="reduction",
            operation="run_pca",
            parameters=parameters,
            inputs={},
            execution_options={},
            required_arrays=("loadings",),
        )
        stored_fit = replace(
            planned,
            provenance=make_provenance(operation="run_pca", parameters=parameters, inputs={}),
        )
        with artifact_transaction(root, stored_fit) as group:
            group.create_array("loadings", data=model.components_.T)
        status = inspect_artifact(root, stored_fit.ref)
        current = plan_artifact(
            root,
            scope="assay",
            assay="RNA",
            kind="reduction",
            operation="run_pca",
            parameters=parameters,
            inputs={},
            execution_options={},
            required_arrays=("loadings",),
        )
        results.append(
            {
                "blocks": blocks,
                "storedRevision": status.revision,
                "currentRevision": status.current_revision,
                "supersededBy": [r.revision for r in status.superseded_by],
                "reused": current.reused,
            }
        )
    if results[0]["reused"] is not True or results[1]["reused"] is not False:
        raise RuntimeError("Predicate-scoped PCA reuse does not match the released revision rule")
    return {
        "comparisonKind": "Stored IncrementalPCA fits replayed through current artifact planner",
        "cases": results,
    }


def evaluate_pca_subspaces(payload, work):
    workers = int(payload.get("workers", 8))
    path, source = prepare_store(
        payload.get("dataset", PANCREAS), work, workers, payload.get("cellLimit")
    )
    ds = cytearc.DataStore(str(path), default_assay="RNA", nthreads=workers, mem_budget="4G")
    cells = ds.snapshot_cell_selection("I")
    results = []
    revision = None
    for width in payload.get("featureCounts", (2000, 4095, 4096, 4097, 5000)):
        features = ds.features.hvgs(
            cells,
            top_n=width,
            min_cells=3,
            max_cells=source["cells"],
            blacklist="",
            show_plot=False,
        )
        norm = ds.features.normalize(cells, features)
        x = np.asarray(ds.artifacts.load(norm)["data"][:], dtype=np.float64)
        if x.shape[1] != width:
            raise RuntimeError(f"PCA boundary check requested {width} features, got {x.shape[1]}")
        dense = None
        for batch_size in payload.get("batchSizes", (256, 1024)):
            ref, timing = measured(
                lambda: ds.reduction.pca(
                    norm,
                    dims=int(payload.get("pcaDims", 21)),
                    feat_scaling=True,
                    batch_size=batch_size,
                    invalidate_cache=True,
                )
            )
            group = ds.artifacts.load(ref)
            status = ds.artifacts.inspect(ref)
            scaling = ds.artifacts.load(
                cytearc.ArtifactRef.from_dict(status.inputs["feature_scaling"])
            )
            values = (x - np.asarray(scaling["mean"][:])) / np.asarray(scaling["scale"][:])
            values -= values.mean(axis=0)
            dims = group["loadings"].shape[1]
            if dense is None:
                with threadpool_limits(limits=workers):
                    u, singular, vt = svd(values, full_matrices=False, check_finite=True)
                dense = (
                    u[:, :dims] * singular[:dims],
                    vt[:dims].T,
                    singular[:dims] ** 2 / np.sum(singular**2),
                )
                del u, vt, singular
            angles = np.degrees(subspace_angles(np.asarray(group["loadings"][:]), dense[1]))
            ratio = np.asarray(group["variance_ratio"][:])
            result = {
                "features": width,
                "cells": x.shape[0],
                "batchSize": batch_size,
                "solver": status.parameters.get("solver", "gram"),
                "maxSubspaceAngleDegrees": float(np.max(angles)),
                "coordinateSubspaceAngleDegrees": float(
                    np.max(np.degrees(subspace_angles(np.asarray(group["data"][:]), dense[0])))
                ),
                "maxRelativeVarianceShareError": float(np.max(np.abs(ratio - dense[2]) / dense[2])),
                **timing,
            }
            result["meetsTolerance"] = all(
                result[key] <= value for key, value in PCA_TOLERANCES.items()
            )
            results.append(result)
            checkpoint(
                payload, {"source": source, "tolerances": PCA_TOLERANCES, "completedCases": results}
            )
            if revision is None:
                revision = evaluate_operation_revision(
                    values[: min(512, len(values)), : min(100, width)], work
                )
        del x, values, dense
        gc.collect()
    return {
        "source": source,
        "tolerances": PCA_TOLERANCES,
        "results": results,
        "operationRevision": revision,
        "reference": "Dense float64 SVD of the same normalized, scaled and centered matrix",
        "timingBoundary": "CyteArc artifact PCA creation; dense reference is excluded",
    }


def score_doublets(counts, cells, genes, work, workers, memory_budget="8G"):
    path = work / "store.zarr"
    _, conversion = measured(
        lambda: cytearc.SparseToZarr(
            counts,
            str(path),
            cells,
            genes,
            assay_name="RNA",
            assay_type="RNA",
            mem_budget=memory_budget,
            nthreads=workers,
        ).dump()
    )
    ds = cytearc.DataStore(
        str(path), default_assay="RNA", nthreads=workers, mem_budget=memory_budget
    )
    fitted, preparation = measured(
        lambda: ds.pipeline.run(
            filtering=False,
            hvg_count=2000,
            pca_dims=21,
            neighbors_k=15,
            umap=False,
            paris=False,
            cell_cycle=False,
            doublets=False,
            markers=False,
            params={
                "ann_index": {"ann_parallel": False, "rand_state": 4466},
                "leiden": {
                    "partitions": [1.0],
                    "selected": 1.0,
                    "backend": "cytearc",
                    "random_seed": 4444,
                },
            },
        )
    )
    selection = np.asarray(ds.artifacts.load(fitted["analysis_cell_selection"])["values"][:])
    if not selection.all():
        raise RuntimeError("Doublet evaluation did not retain every benchmark cell")
    ref, detection = measured(
        lambda: ds.qc.doublets(
            fitted["clusters"],
            fitted["connectivity_map"],
            **DOUBLET_PARAMETERS,
            invalidate_cache=True,
        )
    )
    group = ds.artifacts.load(ref)
    scores = np.asarray(group["values"][:], dtype=float)
    if scores.shape != (len(cells),) or not np.isfinite(scores).all():
        raise RuntimeError("Doublet detector did not return finite scores for every cell")
    return scores, {
        "conversion": conversion,
        "preparation": preparation,
        "detection": detection,
        "totalAnalysisSeconds": sum(x["seconds"] for x in (conversion, preparation, detection)),
        "parameters": ds.artifacts.inspect(ref).parameters,
        "sampleSummaries": group.attrs["samples"],
        "pipelineConfig": fitted.report()["run"]["config"],
        "timingBoundary": "Separate sparse-count conversion, PCA/HNSW/graph/clustering preparation, and detection reusing those artifacts",
    }


def evaluate_doublets(payload, work):
    import subprocess
    import traceback
    import urllib.request
    import zipfile
    from importlib.metadata import version

    import pandas as pd

    from experiments.methods.tools.doublet_benchmark import read_dataset
    from experiments.runtime.harness import R2

    workers = int(payload.get("workers", 8))
    method = payload["method"]
    if method not in ("cytearc", "scrublet", "scDblFinder", "doubletFinder"):
        raise ValueError(f"Unknown doublet method: {method}")
    archive = work / "real_datasets.zip"
    source_url = "https://zenodo.org/api/records/4062232/files/real_datasets.zip/content"
    if not archive.exists():
        urllib.request.urlretrieve(source_url, archive)
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "md5").hexdigest()
    if digest != "cf49f0fe3750e1dd057a10686124fd5a":
        raise RuntimeError(f"The Xi and Li archive changed: {digest}")
    data = work / "data"
    data.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if not (data / name).resolve().is_relative_to(data.resolve()):
                raise ValueError("Archive contains a path outside the data directory")
        handle.extractall(data)
    files = {p.stem: p for p in data.rglob("*.rds")}
    if set(files) != set(XI_LI_DATASETS):
        raise RuntimeError(f"Expected all 16 fixed benchmark datasets, got {sorted(files)}")
    storage = R2(payload["r2EndpointUrl"])
    prefix = payload["recordUri"].removesuffix(".json")
    records = []
    for name in XI_LI_DATASETS:
        target = work / name
        target.mkdir(exist_ok=True)
        entry = {"dataset": name, "method": method, "status": "running"}
        try:
            counts, cells, genes, labels = read_dataset(files[name])
            if set(labels) != {"singlet", "doublet"}:
                raise RuntimeError(f"Unexpected benchmark labels: {set(labels)}")
            if method == "cytearc":
                scores, timing = score_doublets(
                    counts, cells, genes, target, workers, payload.get("memoryBudget", "8G")
                )
            elif method == "scrublet":
                import anndata
                import scanpy as sc

                annotated = anndata.AnnData(X=counts.copy())
                annotated.obs_names = cells
                with threadpool_limits(limits=workers):
                    _, timing = measured(
                        lambda: sc.pp.scrublet(
                            annotated,
                            random_state=0,
                            threshold=0.25,
                            verbose=False,
                        )
                    )
                scores = annotated.obs["doublet_score"].to_numpy(dtype=float)
                timing["timingBoundary"] = (
                    "Scrublet preprocessing, simulation and scoring on in-memory sparse counts"
                )
                timing["versions"] = {
                    key: version(key) for key in ("scanpy", "anndata", "scikit-image")
                }
                del annotated
            else:
                counts = None
                gc.collect()
                script = Path(__file__).parent / "tools" / "doublets.R"
                process, timing = measured(
                    lambda: subprocess.run(
                        ["Rscript", str(script), str(files[name]), str(work), str(workers), method],
                        capture_output=True,
                        text=True,
                        timeout=3600,
                    )
                )
                (target / "r.log").write_text(process.stdout + process.stderr)
                if process.returncode:
                    raise RuntimeError(
                        f"R {method} failed: {(process.stdout + process.stderr)[-3000:]}"
                    )
                frame = pd.DataFrame({"cell": cells}).merge(
                    pd.read_csv(target / "r_scores.csv"),
                    on="cell",
                    how="left",
                    validate="one_to_one",
                )
                scores = frame[method].to_numpy(dtype=float)
                timing.update(json.loads((target / "r_timing.json").read_text()))
                timing["versions"] = json.loads((work / "r_versions.json").read_text())
                timing["timingBoundary"] = (
                    "R method preprocessing and scoring; process elapsed and container memory also include startup and input decoding"
                )
            if len(scores) != len(cells) or not np.isfinite(scores).all():
                raise RuntimeError(
                    "Every method must return finite scores for exactly the same cells"
                )
            truth = labels == "doublet"
            entry.update(
                {
                    "status": "ok",
                    "cells": len(cells),
                    "genes": len(genes),
                    "doublets": int(truth.sum()),
                    "prevalence": float(truth.mean()),
                    "auprc": float(average_precision_score(truth, scores)),
                    "auroc": float(roc_auc_score(truth, scores)),
                    "timing": timing,
                }
            )
            table = target / "scores.csv"
            pd.DataFrame({"cell": cells, "label": labels, "score": scores}).to_csv(
                table, index=False
            )
            storage.upload(table, f"{prefix}/{name}/scores.csv")
            del counts, scores
        except Exception:
            entry.update({"status": "failed", "error": traceback.format_exc()})
        records.append(entry)
        storage.put_json(f"{prefix}/{name}/record.json", entry)
        gc.collect()
        shutil.rmtree(target)
    return {
        "status": "ok" if all(x["status"] == "ok" for x in records) else "partial",
        "sourceUrl": source_url,
        "archiveMd5": digest,
        "method": method,
        "datasets": records,
        "eligibleDatasets": XI_LI_DATASETS,
        "parametersFixedBeforeEvaluation": True,
        "configuration": DOUBLET_PARAMETERS,
        "developmentSeparation": "Released defaults fixed before outcomes; no tuning on Kang because five Xi and Li matrices derive from GSE96583",
        "sampleDefinition": "One scoring model per supplied matrix; separate capture labels are not supplied in the archive",
        "sourceAttribution": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7897250/",
    }


def run(payload: dict) -> dict:
    configure_cytearc()
    work = Path(payload.get("workDirectory", WORK / payload["check"]))
    work.mkdir(parents=True, exist_ok=True)
    operations = {
        "pca": evaluate_pca_subspaces,
        "determinism": evaluate_pipeline_determinism,
        "doublets": evaluate_doublets,
    }
    if payload["check"] not in operations:
        raise ValueError(f"Unknown methods check: {payload['check']}")
    result = operations[payload["check"]](payload, work)
    result = {
        "status": result.get("status", "ok"),
        "experiment": "rna_numerical_reproducibility",
        "check": payload["check"],
        "provenance": provenance(),
        "settings": scrub_private(payload),
        **result,
    }
    write_json(
        "rna_numerical_reproducibility.json",
        result,
        directory=Path(payload.get("outputDirectory", RESULTS)),
    )
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.payload.read_text())), default=str))


if __name__ == "__main__":
    main()
