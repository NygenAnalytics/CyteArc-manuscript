import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

import cytearc
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.runtime.common import configure_cytearc, provenance, write_json
from experiments.runtime.metrics import ResourceSampler

SETTINGS = {
    "neighborCount": 20,
    "seuratCandidateBreadths": [50, 200],
    "hvgCount": 2000,
    "rnaDimensions": 30,
    "adtDimensions": 15,
    "atacDimensions": 30,
    "peakCount": 25000,
    "resolution": 1.0,
    "seed": 4466,
    "warmupCells": 5000,
    "l2Normalize": True,
    "annParallel": False,
    "tolerances": {
        "matchedKernelMaxAbs": 1e-6,
        "candidateWeightMaxAbs": 1e-12,
        "candidateMeanNeighborOverlapMin": 0.90,
        "candidateBiologicalAriDecreaseMax": 0.05,
    },
}
DATASETS = {
    "tenx_8K_pbmc_citeseq": {"modalities": ["RNA", "ADT"], "labelColumn": None},
    "swanson_7K_pbmc_teaseq": {
        "modalities": ["RNA", "ATAC", "ADT"],
        "labelColumn": "tea_cell_type",
    },
    "hao_pbmc_citeseq": {"modalities": ["RNA", "ADT"], "labelColumn": "celltype.l2"},
    "combat_stephenson_pbmc_citeseq": {
        "modalities": ["RNA", "ADT"],
        "labelColumn": "cell_type",
        "groupColumn": "study",
    },
}


def measure(stages: list[dict], name: str, operation):
    from cytearc.storage.artifact_writer import artifact_plan_scope

    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    with artifact_plan_scope() as receipts:
        try:
            result = operation()
        finally:
            elapsed = time.perf_counter() - started
            resources = asdict(sampler.stop())
            stage = {"stage": name, "wallSeconds": elapsed, **resources}
            if receipts:
                stage["artifactPlans"] = [
                    {
                        "operation": receipt.operation,
                        "kind": receipt.ref.kind,
                        "assay": receipt.ref.assay,
                        "artifactId": receipt.ref.artifact_id,
                        "disposition": receipt.disposition,
                    }
                    for receipt in receipts
                ]
            stages.append(stage)
    return result


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def persist_measurement(payload: dict, result: dict) -> None:
    if payload.get("recordUri"):
        from experiments.runtime.harness import R2

        R2(payload["r2EndpointUrl"]).put_json(
            payload["recordUri"], {"status": "measurements_complete", "result": result}
        )


def persist_progress(payload: dict, progress: dict) -> None:
    """Save the stages finished so far; the runner keeps them if the job then fails."""
    if payload.get("recordUri"):
        from experiments.runtime.harness import R2

        R2(payload["r2EndpointUrl"]).put_json(
            payload["recordUri"], {"status": "running", "progress": progress}
        )


def count_matrix_policy(settings: dict):
    """The count-matrix layout named by a payload, with exactly its two sizes."""
    from cytearc.storage.count_matrix import CountMatrixPolicy

    if set(settings) != {"unitBytes", "chunkBytes"}:
        raise ValueError(
            f"countMatrixPolicy needs exactly unitBytes and chunkBytes, got {sorted(settings)}"
        )
    return CountMatrixPolicy(
        unitBytes=int(settings["unitBytes"]), chunkBytes=int(settings["chunkBytes"])
    )


def artifact_inventory(ds) -> set[str]:
    refs = ds.artifacts.list(scope="datastore")
    for assay in ds.assay_names:
        refs.extend(ds.artifacts.list(from_assay=assay))
    return {ref.artifact_id for ref in refs}


def verify_fresh_computation(stages: list[dict], existing: set[str]) -> dict:
    kinds = {
        "feature_summary",
        "normalized",
        "feature_scaling",
        "reduction",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "integrated_graph",
        "cluster_labels",
    }
    receipts = []
    for stage in stages:
        for receipt in stage.get("artifactPlans", []):
            receipt["existedBeforePhase"] = receipt["artifactId"] in existing
            if receipt["kind"] in kinds or receipt["operation"] in {
                "select_hvgs",
                "select_prevalent_peaks",
            }:
                receipts.append(receipt)
    if not receipts:
        raise AssertionError("No computational artifact receipts were captured")
    bad = [
        receipt
        for receipt in receipts
        if receipt["disposition"] == "reused" and receipt["existedBeforePhase"]
    ]
    if bad:
        raise AssertionError(f"Fresh workflow reused pre-existing computations: {bad}")
    return {
        "createdComputations": sum(r["disposition"] == "created" for r in receipts),
        "reusedComputationsWithinPhase": sum(r["disposition"] == "reused" for r in receipts),
        "reusedPreexistingComputations": len(bad),
    }


def neighbor_overlap(first: np.ndarray, second: np.ndarray) -> dict:
    if first.shape != second.shape:
        raise ValueError(f"Neighbor shapes differ: {first.shape} and {second.shape}")
    overlap = np.array([len(set(a) & set(b)) / len(a) for a, b in zip(first, second, strict=True)])
    return {"mean": float(overlap.mean()), "median": float(np.median(overlap))}


def label_metrics(labels: np.ndarray, reference: np.ndarray) -> dict:
    if labels.shape != reference.shape:
        raise ValueError("Labels do not align")
    selected = ~np.isin(reference.astype(str), ["", "nan", "None", "NA"])
    if len(np.unique(reference[selected])) < 2:
        raise ValueError("Fewer than two annotated cell types remain")
    return {
        "ari": float(adjusted_rand_score(reference[selected], labels[selected])),
        "nmi": float(normalized_mutual_info_score(reference[selected], labels[selected])),
        "clusters": int(len(np.unique(labels))),
        "annotatedCells": int(selected.sum()),
        "unannotatedCells": int((~selected).sum()),
    }


def prepare_modality(ds, cells, assay: str, stages: list[dict], *, invalidate: bool) -> dict:
    if assay == "RNA":
        features = measure(
            stages,
            f"{assay}.features",
            lambda: ds.features.hvgs(
                cells,
                from_assay=assay,
                top_n=SETTINGS["hvgCount"],
                show_plot=False,
                invalidate_cache=invalidate,
            ),
        )
    elif assay == "ATAC":
        features = measure(
            stages,
            f"{assay}.features",
            lambda: ds.features.prevalent_peaks(
                cells,
                from_assay=assay,
                top_n=SETTINGS["peakCount"],
                invalidate_cache=invalidate,
            ),
        )
    elif assay == "ADT":
        names = ds.ADT.feats.fetch_all("names").astype(str)
        mask = np.array(["control" not in name.lower() for name in names])
        features = measure(
            stages,
            f"{assay}.features",
            lambda: ds.features.snapshot(
                from_assay=assay,
                mask=mask,
                invalidate_cache=invalidate,
            ),
        )
    else:
        raise ValueError(f"Unsupported assay {assay}")
    normalized = measure(
        stages,
        f"{assay}.normalization",
        lambda: ds.features.normalize(
            cells,
            features,
            invalidate_cache=invalidate,
        ),
    )
    if assay == "ATAC":
        reduction = measure(
            stages,
            f"{assay}.reduction",
            lambda: ds.reduction.lsi(
                normalized,
                dims=SETTINGS["atacDimensions"],
                skip_first=True,
                solver="streaming",
                rand_state=SETTINGS["seed"],
                invalidate_cache=invalidate,
            ),
        )
    else:
        width = int(np.count_nonzero(ds.artifacts.load(features)["values"][:]))
        dims = min(SETTINGS[f"{assay.lower()}Dimensions"], width - 1)
        reduction = measure(
            stages,
            f"{assay}.reduction",
            lambda: ds.reduction.pca(
                normalized,
                dims=dims,
                invalidate_cache=invalidate,
            ),
        )
    index = measure(
        stages,
        f"{assay}.index",
        lambda: ds.graph.ann_index(
            reduction,
            ann_parallel=False,
            rand_state=SETTINGS["seed"],
            invalidate_cache=invalidate,
        ),
    )
    neighbors = measure(
        stages,
        f"{assay}.neighbors",
        lambda: ds.graph.neighbors(
            index,
            k=SETTINGS["neighborCount"],
            invalidate_cache=invalidate,
        ),
    )
    graph = measure(
        stages,
        f"{assay}.connectivity",
        lambda: ds.graph.connectivity(
            neighbors,
            invalidate_cache=invalidate,
        ),
    )
    clusters = measure(
        stages,
        f"{assay}.clustering",
        lambda: ds.clusters.leiden(
            graph,
            resolution=SETTINGS["resolution"],
            random_seed=SETTINGS["seed"],
            backend="cytearc",
            invalidate_cache=invalidate,
        ),
    )
    return {
        "features": features,
        "normalized": normalized,
        "reduction": reduction,
        "index": index,
        "neighbors": neighbors,
        "graph": graph,
        "clusters": clusters,
    }


def integrate(
    ds,
    modalities: dict,
    method: str,
    stages: list[dict],
    *,
    invalidate: bool,
    resolution: float = 1.0,
):
    sources = [refs["neighbors" if method == "wnn" else "graph"] for refs in modalities.values()]
    graph = measure(
        stages,
        f"{method}.integration",
        lambda: ds.integration.modalities(
            sources,
            method=method,
            l2_normalize=True,
            invalidate_cache=invalidate,
        ),
    )
    clusters = measure(
        stages,
        f"{method}.clustering",
        lambda: ds.clusters.leiden(
            graph,
            resolution=resolution,
            random_seed=SETTINGS["seed"],
            backend="cytearc",
            invalidate_cache=invalidate,
        ),
    )
    return graph, clusters


def export_inputs(
    ds,
    cells: np.ndarray,
    modalities: dict,
    labels,
    wnn,
    clusters,
    snn,
    snn_clusters,
    destination: Path,
    settings: dict,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "modalities": list(modalities),
        "settings": settings,
        "cells": len(cells),
        "modalityDimensions": {},
        "normalization": {},
    }
    cell_names = ds.cells.fetch_all("ids")[cells].astype(str)
    (destination / "cells.txt").write_text("\n".join(cell_names) + "\n")
    for assay, refs in modalities.items():
        embedding = np.asarray(ds.artifacts.load(refs["reduction"])["data"][:])
        manifest["modalityDimensions"][assay] = int(embedding.shape[1])
        manifest["normalization"][assay] = dict(
            ds.artifacts.load(refs["normalized"]).attrs["provenance"]["parameters"]
        )
        indices = np.asarray(ds.artifacts.load(refs["neighbors"])["indices"][:])
        np.savetxt(destination / f"{assay}_embedding.csv.gz", embedding, delimiter=",")
        np.savetxt(destination / f"{assay}_indices.csv.gz", indices, fmt="%d", delimiter=",")
        selected = np.flatnonzero(ds.artifacts.load(refs["features"])["values"][:])
        write_count_inputs(ds, cells, assay, selected, destination)
    group = ds.artifacts.load(wnn)
    edges = np.asarray(group["edges"][:])
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Unexpected integrated edge shape {edges.shape}")
    indices = edges[:, 1].reshape(len(cells), SETTINGS["neighborCount"])
    np.savez_compressed(
        destination / "cytearc.npz",
        neighbors=indices,
        affinities=np.asarray(group["weights"][:]).reshape(indices.shape),
        weights=np.asarray(group["modality_weights"][:]),
        clusters=np.asarray(ds.artifacts.load(clusters)["values"][:]),
        snnNeighbors=np.asarray(ds.artifacts.load(snn)["edges"][:])[:, 1].reshape(indices.shape),
        snnClusters=np.asarray(ds.artifacts.load(snn_clusters)["values"][:]),
    )
    if labels is not None:
        np.savetxt(destination / "labels.txt", labels.astype(str), fmt="%s")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = destination.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        for file in sorted(destination.iterdir()):
            handle.add(file, arcname=file.name)
    return archive


def export_graphs(
    ds,
    cells: np.ndarray,
    modalities: dict,
    labels,
    groups,
    joint: dict,
    revised: dict,
    destination: Path,
) -> Path:
    """Save every per-cell graph and partition of one run for later evaluation.

    Each modality keeps its 20 self-free neighbour rows, their distances and
    connectivity weights and its resolution 1.0 clusters; each joint graph
    keeps its 20 neighbours per cell and edge weights, row by row in the order
    the store holds them.
    """

    def rows(ref, name: str) -> tuple[np.ndarray, np.ndarray]:
        group = ds.artifacts.load(ref)
        edges = np.asarray(group["edges"][:])
        if not np.array_equal(
            edges[:, 0], np.repeat(np.arange(len(cells)), SETTINGS["neighborCount"])
        ):
            raise ValueError(
                f"{name} edges are not stored row by row with {SETTINGS['neighborCount']} per cell"
            )
        shape = (len(cells), SETTINGS["neighborCount"])
        return edges[:, 1].reshape(shape), np.asarray(
            group["weights"][:], dtype=np.float32
        ).reshape(shape)

    arrays = {"cells": cells}
    if labels is not None:
        arrays["labels"] = np.asarray(labels, dtype=str)
    if groups is not None:
        arrays["groups"] = np.asarray(groups, dtype=str)
    for assay, refs in modalities.items():
        neighbors = ds.artifacts.load(refs["neighbors"])
        arrays[f"{assay}_neighbors"] = np.asarray(neighbors["indices"][:])
        arrays[f"{assay}_distances"] = np.asarray(neighbors["distances"][:])
        columns, weights = rows(refs["graph"], f"{assay} connectivity")
        if not np.array_equal(columns, arrays[f"{assay}_neighbors"]):
            raise ValueError(f"{assay} connectivity columns differ from its neighbour rows")
        arrays[f"{assay}_connectivity_weights"] = weights
        arrays[f"{assay}_clusters"] = np.asarray(ds.artifacts.load(refs["clusters"])["values"][:])
    for method, (graph, clusters) in joint.items():
        arrays[f"{method}_neighbors"], arrays[f"{method}_weights"] = rows(graph, method)
        arrays[f"{method}_clusters"] = np.asarray(ds.artifacts.load(clusters)["values"][:])
        if method == "wnn":
            arrays["wnn_modality_weights"] = np.asarray(
                ds.artifacts.load(graph)["modality_weights"][:]
            )
    for name, clusters in revised.items():
        arrays[f"{name}_clusters"] = np.asarray(ds.artifacts.load(clusters)["values"][:])
    np.savez(destination, **arrays)
    return destination


def write_count_inputs(
    ds, cells: np.ndarray, assay: str, features: np.ndarray, destination: Path
) -> int:
    """Write the selected counts as a features-by-cells MatrixMarket file; return its entries."""
    from scipy.sparse import vstack

    blocks = []
    for start in range(0, len(cells), 2000):
        values = getattr(ds, assay).rawData[cells[start : start + 2000], :][:, features].compute()
        blocks.append(csr_matrix(values))
    matrix = vstack(blocks).T.tocoo()
    del blocks
    if matrix.dtype.kind == "u":
        # SciPy writes uint32 as MatrixMarket "unsigned-integer", which R's readMM rejects.
        matrix.data = matrix.data.astype(np.int64)
    with gzip.open(destination / f"{assay}_counts.mtx.gz", "wb") as stream:
        mmwrite(stream, matrix)
    names = getattr(ds, assay).feats.fetch_all("ids")[features].astype(str)
    (destination / f"{assay}_features.txt").write_text("\n".join(names) + "\n")
    return int(matrix.nnz)


def prepare_workflow_inputs(payload: dict) -> dict:
    from experiments.runtime.object_io import download_file, upload_file

    configure_cytearc()
    dataset = payload["dataset"]
    if DATASETS[dataset]["modalities"] != ["RNA", "ADT"]:
        raise ValueError("The complete native Seurat workflow requires RNA and ADT")
    work = Path(payload.get("workDir", "/tmp/multimodal_workflow_inputs"))
    work.mkdir(parents=True, exist_ok=True)
    if payload.get("storePath"):
        source = Path(payload["storePath"])
    else:
        archive = work / "source.tar.gz"
        download_file(payload["storeArchiveUri"], archive)
        source = work / "source.zarr"
        source.mkdir()
        with tarfile.open(archive) as handle:
            handle.extractall(source, filter="data")
    ds = cytearc.DataStore(
        str(source),
        default_assay="RNA",
        nthreads=payload.get("threads", 8),
        mem_budget=payload.get("memoryBudgetGiB", 24) * 1024**3,
    )
    eligible = np.flatnonzero(ds.cells.fetch_all("I"))
    ordered = np.random.default_rng(SETTINGS["seed"]).permutation(eligible)
    names = ds.ADT.feats.fetch_all("names").astype(str)
    features = {
        "RNA": np.arange(ds.RNA.feats.N),
        "ADT": np.flatnonzero(["control" not in name.lower() for name in names]),
    }
    result = {
        "status": "ok",
        "dataset": dataset,
        "requestedCellCounts": payload["cellCounts"],
        "inputs": [],
        "settings": SETTINGS,
        "provenance": provenance(),
    }
    for count in payload["cellCounts"]:
        if count > len(eligible) or count <= SETTINGS["neighborCount"]:
            raise ValueError(f"Ineligible cell count {count}")
        selected = np.sort(ordered[:count])
        destination = work / f"cells_{count}"
        destination.mkdir()
        cells = ds.cells.fetch_all("ids")[selected].astype(str)
        (destination / "cells.txt").write_text("\n".join(cells) + "\n")
        entries = {
            assay: write_count_inputs(ds, selected, assay, features[assay], destination)
            for assay in ["RNA", "ADT"]
        }
        label_column = DATASETS[dataset]["labelColumn"]
        if label_column is not None:
            labels = ds.cells.fetch_all(label_column)[selected].astype(str)
            np.savetxt(destination / "labels.txt", labels, fmt="%s")
        manifest = {
            "dataset": dataset,
            "cells": count,
            "modalities": ["RNA", "ADT"],
            "settings": SETTINGS,
            "labelColumn": label_column,
            "cellSelectionSha256": array_hash(selected),
            "features": {name: len(indices) for name, indices in features.items()},
            "nonzeroEntries": entries,
            "boundary": "Full raw RNA counts and the same selected ADT panel; no reductions or variable-gene selection have been applied.",
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        archive = Path(shutil.make_archive(str(destination), "gztar", destination))
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entry = {
            "cells": count,
            "cellSelectionSha256": manifest["cellSelectionSha256"],
            "inputsSha256": digest,
            "archiveBytes": archive.stat().st_size,
            "nonzeroEntries": entries,
            "localArchive": str(archive),
        }
        if payload.get("inputsPrefix"):
            uri = f"{payload['inputsPrefix'].rstrip('/')}/cells_{count}/inputs.tar.gz"
            upload_file(archive, uri)
            entry["inputsUri"] = uri
        result["inputs"].append(entry)
        write_json("multimodal_neighbors.json", result, directory=work)
        persist_measurement(payload, result)
    return result


class MemoryTrace(ResourceSampler):
    """Container memory over time, with a stop for a child process at a limit.

    ResourceSampler, the sampler behind every stage record, finds the memory
    file: cgroup v2 ``memory.current`` or cgroup v1 ``memory.usage_in_bytes``.
    Each sample is also kept with its time, so stages get their own peaks, and
    the watched process group is killed when a sample reaches ``limitBytes``.
    Without a readable cgroup file the samples are process-tree RSS, which
    counts pages shared by forked R workers once per process; a memory limit
    then raises before the child starts, because RSS would stop it early.
    """

    def __init__(self, limitBytes: int | None, *, sampleIntervalSeconds: float = 0.5, **options):
        # The trace never resets memory.peak, which the enclosing stage sampler reads.
        super().__init__(
            sampleIntervalSeconds=sampleIntervalSeconds, resetCgroupPeak=False, **options
        )
        self.limitBytes = limitBytes
        self.process: subprocess.Popen | None = None
        self.samples: list[tuple[float, int]] = []
        self.limitReached = False

    @property
    def memorySource(self) -> dict:
        if self._cgroupPath is None:
            return {"kind": "processTreeRss", "file": None}
        name = self._cgroup_file_name("memory.current")
        kind = "cgroupV2MemoryCurrent" if self._cgroupVersion == 2 else "cgroupV1MemoryUsageInBytes"
        return {"kind": kind, "file": str(self._cgroupPath / name)}

    def start(self) -> "MemoryTrace":
        super().start()
        if self.limitBytes is not None and self._cgroupPath is None:
            super().stop()
            raise RuntimeError(
                "A memory limit needs a readable cgroup memory file (cgroup v2 memory.current "
                "or cgroup v1 memory.usage_in_bytes); none was found, and process-tree RSS "
                "would count memory shared by forked R workers several times"
            )
        return self

    def watch(self, process: subprocess.Popen) -> None:
        self.process = process

    def _collect_sample(self):
        import signal

        sample = super()._collect_sample()
        value = (
            sample.processTreeRssBytes
            if self._cgroupPath is None
            else sample.cgroupMemoryCurrentBytes
        )
        if value is None:
            return sample
        self.samples.append((time.time(), value))
        process = self.process
        if (
            self.limitBytes is not None
            and value >= self.limitBytes
            and process is not None
            and process.poll() is None
        ):
            self.limitReached = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                # The process ended between the poll and the kill.
                pass
        return sample


def run_with_limits(
    command: list[str],
    log: Path,
    *,
    limitBytes: int | None,
    timeLimitSeconds: float | None,
    **samplerOptions,
) -> dict:
    """Run a command in its own process group, stopping it at a memory or time limit."""
    import signal

    trace = MemoryTrace(limitBytes, **samplerOptions).start()
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            trace.watch(process)
            try:
                code = process.wait(timeout=timeLimitSeconds)
                if code == 0:
                    outcome = "completed"
                elif trace.limitReached:
                    outcome = "memory_limit"
                else:
                    # A signal, such as the kernel's out-of-memory kill, or an error exit.
                    outcome = "killed_by_signal" if code < 0 else "error"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                outcome = "time_limit"
    finally:
        measurement = trace.stop()
    return {
        "outcome": outcome,
        "returnCode": code,
        "samples": trace.samples,
        "memorySource": trace.memorySource,
        "containerMemoryLimitBytes": measurement.memoryMaxBytes,
    }


def stage_windows(progress: Path, samples: list[tuple[float, int]]) -> list[dict]:
    """Pair the R stage marks and give each stage the peak of the memory samples inside it."""
    opened: dict[str, float] = {}
    windows = []
    lines = progress.read_text().splitlines() if progress.exists() else []
    times = np.array([t for t, _ in samples]) if samples else np.zeros(0)
    values = np.array([v for _, v in samples]) if samples else np.zeros(0, dtype=np.int64)
    for line in lines:
        stage, event, stamp = line.split("\t")
        if event == "start":
            opened[stage] = float(stamp)
            continue
        began = opened.pop(stage)
        inside = (times >= began) & (times <= float(stamp))
        windows.append(
            {
                "stage": stage,
                "startedAt": began,
                "wallSeconds": float(stamp) - began,
                "sampledPeakBytes": int(values[inside].max()) if inside.any() else None,
                "samples": int(inside.sum()),
            }
        )
    for stage, began in opened.items():
        inside = times >= began
        windows.append(
            {
                "stage": stage,
                "startedAt": began,
                "wallSeconds": None,
                "sampledPeakBytes": int(values[inside].max()) if inside.any() else None,
                "samples": int(inside.sum()),
                "unfinished": True,
            }
        )
    return windows


def seurat_workflow(payload: dict) -> dict:
    """Run the native Seurat v5 RNA, ADT and WNN workflow on exported raw counts.

    ``seuratMemoryLimitGiB`` stops R when sampled container memory reaches it and
    ``seuratTimeLimitSeconds`` stops it after that wall time; either stop, a
    signal or an R error is recorded with the stage it reached as status
    ``comparator_stopped`` rather than raised, because where Seurat stops is
    the measurement.
    """
    from experiments.runtime.object_io import download_file, upload_file

    configure_cytearc()
    work = Path(payload.get("workDir", "/tmp/seurat_multimodal_workflow"))
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "inputs.tar.gz"
    download_file(payload["inputsUri"], archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    inputs = work / "inputs"
    inputs.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(inputs, filter="data")
    manifest = json.loads((inputs / "manifest.json").read_text())
    output = work / "outputs"
    output.mkdir()
    limit = payload.get("seuratMemoryLimitGiB")
    deadline = payload.get("seuratTimeLimitSeconds")
    stages = []
    run = measure(
        stages,
        "seurat.workflow.process",
        lambda: run_with_limits(
            [
                "Rscript",
                str(Path(__file__).with_name("multimodal_seurat.R")),
                str(inputs),
                str(output),
                "workflow",
                str(payload.get("threads", 8)),
            ],
            work / "seurat.log",
            limitBytes=None if limit is None else int(limit * 1024**3),
            timeLimitSeconds=deadline,
        ),
    )
    windows = stage_windows(output / "progress.tsv", run["samples"])
    peak = max((value for _, value in run["samples"]), default=None)
    common = {
        "dataset": manifest["dataset"],
        "cells": manifest["cells"],
        "inputsUri": payload.get("inputsUri"),
        "inputsSha256": digest,
        "cellSelectionSha256": manifest["cellSelectionSha256"],
        "nonzeroEntries": manifest.get("nonzeroEntries"),
        "outcome": run["outcome"],
        "stages": stages,
        "stageWindows": windows,
        "sampledPeakBytes": peak,
        "memorySource": run["memorySource"],
        "containerMemoryLimitBytes": run["containerMemoryLimitBytes"],
        "memoryLimitGiB": limit,
        "timeLimitSeconds": deadline,
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
    }
    shutil.copyfile(work / "seurat.log", output / "seurat.log")
    if run["outcome"] != "completed":
        log = (work / "seurat.log").read_text().splitlines()
        result = {
            "status": "comparator_stopped",
            **common,
            "returnCode": run["returnCode"],
            "lastStage": windows[-1]["stage"] if windows else None,
            "logTail": log[-40:],
        }
    else:
        measurements = json.loads((output / "workflow.json").read_text())
        labels_path = inputs / "labels.txt"
        labels = np.asarray(labels_path.read_text().splitlines()) if labels_path.exists() else None
        biological = {}
        for method in ["RNA", "ADT", "WNN"]:
            values = np.loadtxt(output / f"{method}_clusters.csv", dtype=int)
            biological[method] = None if labels is None else label_metrics(values, labels)
        fresh = np.loadtxt(output / "fresh_revision_clusters.csv", dtype=int)
        reused = np.loadtxt(output / "reused_revision_clusters.csv", dtype=int)
        reuse_validation = {
            "clusterLabelsIdentical": bool(np.array_equal(fresh, reused)),
            "clusterAri": float(adjusted_rand_score(fresh, reused)),
        }
        if reuse_validation["clusterAri"] != 1.0:
            raise AssertionError(f"Fresh and reused Seurat partitions differ: {reuse_validation}")
        result = {
            "status": "ok",
            **common,
            "nativeWorkflow": measurements,
            "biologicalAgreement": biological,
            "reuseValidation": reuse_validation,
        }
    write_json("multimodal_neighbors.json", result, directory=work)
    persist_measurement(payload, result)
    if payload.get("outputsUri"):
        saved = Path(shutil.make_archive(str(work / "outputs"), "gztar", output))
        upload_file(saved, payload["outputsUri"])
        result["outputsUri"] = payload["outputsUri"]
    return result


def seurat_weights(payload: dict) -> dict:
    """Pair each cell's WNN modality weights in Seurat and in CyteArc on one size of a scaling series.

    Runs the series' native Seurat workflow through its first analysis on the size's exported raw counts and keeps
    each cell's RNA and ADT weights; checks that the inputs select the cells, in the same order and with the same
    author labels, of the CyteArc run whose saved graphs graphsUri names; and saves both tools' protein weights with
    each cell's author label and study to the .npz beside the record.
    """
    import csv

    from experiments.runtime.object_io import download_file, upload_file

    configure_cytearc()
    work = Path(payload.get("workDir", "/tmp/seurat_weights"))
    work.mkdir(parents=True, exist_ok=True)

    def sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    archive = work / "inputs.tar.gz"
    download_file(payload["inputsUri"], archive)
    if sha256(archive) != payload["inputsSha256"]:
        raise ValueError("The input archive differs from the one the series' Seurat run read")
    inputs = work / "inputs"
    inputs.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(inputs, filter="data")
    manifest = json.loads((inputs / "manifest.json").read_text())
    output = work / "outputs"
    output.mkdir()
    stages = []
    run = measure(
        stages,
        "seurat.weights.process",
        lambda: run_with_limits(
            [
                "Rscript",
                str(Path(__file__).with_name("multimodal_seurat.R")),
                str(inputs),
                str(output),
                "weights",
                str(payload.get("threads", 8)),
            ],
            work / "seurat.log",
            limitBytes=None,
            timeLimitSeconds=payload.get("seuratTimeLimitSeconds"),
        ),
    )
    if run["outcome"] != "completed":
        tail = "\n".join((work / "seurat.log").read_text().splitlines()[-40:])
        raise RuntimeError(f"Seurat stopped with outcome {run['outcome']}:\n{tail}")
    cells = (inputs / "cells.txt").read_text().splitlines()
    with (output / "modality_weights.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if [row["cell"] for row in rows] != cells:
        raise ValueError("Seurat's weights are not in the order of the input cells")
    seurat = np.array([[float(row["rna_weight"]), float(row["adt_weight"])] for row in rows])
    saved = work / "graphs.npz"
    download_file(payload["graphsUri"], saved)
    if sha256(saved) != payload["graphsSha256"]:
        raise ValueError("The saved graphs differ from the ones the CyteArc run recorded")
    with np.load(saved, allow_pickle=False) as graphs:
        selected = graphs["cells"]
        labels = graphs["labels"].astype(str)
        studies = graphs["groups"].astype(str)
        cytearc_weights = np.asarray(graphs["wnn_modality_weights"], dtype=np.float64)
    if len(selected) != len(cells) or array_hash(selected) != manifest["cellSelectionSha256"]:
        raise ValueError("The inputs select other cells than the CyteArc run")
    if (inputs / "labels.txt").read_text().splitlines() != labels.tolist():
        raise ValueError("The inputs' author labels differ from the CyteArc run's")
    if not np.allclose(seurat.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Seurat's modality weights do not sum to one")
    label_names, label_codes = np.unique(labels, return_inverse=True)
    study_names, study_codes = np.unique(studies, return_inverse=True)
    path = work / "seurat_weights.npz"
    np.savez_compressed(
        path,
        cytearc_protein=cytearc_weights[:, 1].astype(np.float32),
        seurat_protein=seurat[:, 1].astype(np.float32),
        label_codes=label_codes.astype(np.int16),
        label_names=label_names,
        study_codes=study_codes.astype(np.int16),
        study_names=study_names,
    )
    uri = payload["recordUri"].removesuffix(".json") + ".npz"
    upload_file(path, uri)
    return {
        "status": "ok",
        "dataset": manifest["dataset"],
        "cells": len(cells),
        "inputsSha256": payload["inputsSha256"],
        "cellSelectionSha256": manifest["cellSelectionSha256"],
        "graphsSha256": payload["graphsSha256"],
        "seurat": json.loads((output / "weights.json").read_text()),
        "proteinWeightPearson": float(np.corrcoef(cytearc_weights[:, 1], seurat[:, 1])[0, 1]),
        "proteinWeightMean": {
            "cytearc": float(cytearc_weights[:, 1].mean()),
            "seurat": float(seurat[:, 1].mean()),
        },
        "stages": stages,
        "sampledPeakBytes": max((value for _, value in run["samples"]), default=None),
        "containerMemoryLimitBytes": run["containerMemoryLimitBytes"],
        "pairedUri": uri,
        "pairedSha256": sha256(path),
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
    }


def select_cells(ds, cells: int | None) -> np.ndarray:
    """The rows a run analyzes, in store order: every eligible cell, or ``cells`` of them drawn with the experiment's
    seed."""
    selected = np.flatnonzero(ds.cells.fetch_all("I"))
    if cells is not None:
        wanted = int(cells)
        if wanted > len(selected) or wanted <= SETTINGS["neighborCount"]:
            raise ValueError("Requested cell count is outside the eligible range")
        selected = np.sort(np.random.default_rng(SETTINGS["seed"]).permutation(selected)[:wanted])
    return selected


def run(payload: dict) -> dict:
    configure_cytearc()
    dataset = payload["dataset"]
    design = DATASETS[dataset]
    joint_method = payload.get("jointMethod", "wnn")
    if joint_method not in {"wnn", "snn"}:
        raise ValueError("jointMethod must be wnn or snn")
    work = Path(payload.get("workDir", "/tmp/multimodal"))
    work.mkdir(parents=True, exist_ok=True)
    if "storeArchiveUri" in payload:
        from experiments.runtime.object_io import download_file

        archive = work / "source.tar.gz"
        download_file(payload["storeArchiveUri"], archive)
        source = work / "source.zarr"
        source.mkdir()
        with tarfile.open(archive) as handle:
            handle.extractall(source, filter="data")
    elif "storePath" in payload:
        source = Path(payload["storePath"])
    else:
        source = (
            Path(
                cytearc.datasets.connect("cytearc_docs").download_dataset(
                    dataset,
                    destination=str(work / "downloads"),
                    zarr=True,
                )
            )
            / "data.zarr"
        )
    store = work / "analysis.zarr"
    if store.exists():
        raise FileExistsError(store)
    shutil.copytree(source, store)
    ds = cytearc.DataStore(
        str(store),
        default_assay="RNA",
        nthreads=payload.get("threads", 8),
        mem_budget=payload.get("memoryBudgetGiB", 24) * 1024**3,
    )
    if set(design["modalities"]) - set(ds.assay_names):
        raise ValueError(f"Required modalities absent: {ds.assay_names}")
    published_artifacts = artifact_inventory(ds)
    selected = select_cells(ds, payload.get("cells"))
    mask = np.zeros(ds.cells.N, dtype=bool)
    mask[selected] = True
    ds.cells.insert("multimodal_cells", mask, overwrite=True)
    selection = ds.snapshot_cell_selection("multimodal_cells")
    label_column = design["labelColumn"]
    labels = (
        None if label_column is None else ds.cells.fetch_all(label_column)[selected].astype(str)
    )
    if labels is not None and len(np.unique(labels[labels != ""])) < 2:
        raise ValueError("Biological labels are degenerate")
    # Compiled kernels are built on first use. The whole workflow first runs on
    # a small selection, so the timed phases below measure warm execution.
    warmup_cells = np.sort(
        np.random.default_rng(SETTINGS["seed"] + 1).permutation(selected)[: SETTINGS["warmupCells"]]
    )
    warmup_mask = np.zeros(ds.cells.N, dtype=bool)
    warmup_mask[warmup_cells] = True
    ds.cells.insert("multimodal_warmup_cells", warmup_mask, overwrite=True)
    warmup_selection = ds.snapshot_cell_selection("multimodal_warmup_cells")
    warmup_stages = []
    warmup_started = time.perf_counter()
    warmup_modalities = {
        assay: prepare_modality(ds, warmup_selection, assay, warmup_stages, invalidate=True)
        for assay in design["modalities"]
    }
    for method in ("wnn", "snn"):
        integrate(ds, warmup_modalities, method, warmup_stages, invalidate=True)
    warmup_wall = time.perf_counter() - warmup_started
    before_initial = artifact_inventory(ds)
    stages = []
    initial_started = time.perf_counter()
    modalities = {
        assay: prepare_modality(ds, selection, assay, stages, invalidate=True)
        for assay in design["modalities"]
    }
    primary = integrate(ds, modalities, joint_method, stages, invalidate=True)
    initial_branch_wall = time.perf_counter() - initial_started
    other_method = "snn" if joint_method == "wnn" else "wnn"
    secondary = integrate(ds, modalities, other_method, stages, invalidate=True)
    (wnn, wnn_clusters), (snn, snn_clusters) = (
        (primary, secondary) if joint_method == "wnn" else (secondary, primary)
    )
    original_graph = primary[0]
    initial_freshness = verify_fresh_computation(stages, before_initial)
    metrics = {}
    if labels is not None:
        refs = {assay: state["clusters"] for assay, state in modalities.items()}
        refs.update(wnn=wnn_clusters, snn=snn_clusters)
        metrics = {
            name: label_metrics(np.asarray(ds.artifacts.load(ref)["values"][:]), labels)
            for name, ref in refs.items()
        }
    reuse_stages = []
    reuse_started = time.perf_counter()
    reused = {
        assay: prepare_modality(ds, selection, assay, reuse_stages, invalidate=False)
        for assay in design["modalities"]
    }
    identities = {
        assay: {name: state[name] == reused[assay][name] for name in state}
        for assay, state in modalities.items()
    }
    if not all(all(state.values()) for state in identities.values()):
        raise AssertionError(f"Upstream artifacts were not reused: {identities}")
    # A changed final resolution requires only clustering after the integrated graph is reused.
    reused_graph = measure(
        reuse_stages,
        f"{joint_method}.integration",
        lambda: ds.integration.modalities(
            [refs["neighbors" if joint_method == "wnn" else "graph"] for refs in reused.values()],
            method=joint_method,
            l2_normalize=True,
        ),
    )
    if reused_graph != original_graph:
        raise AssertionError(f"{joint_method} artifact was not reused")
    revised_clusters = measure(
        reuse_stages,
        f"{joint_method}.revised_clustering",
        lambda: ds.clusters.leiden(
            reused_graph,
            resolution=1.2,
            backend="cytearc",
            random_seed=SETTINGS["seed"],
        ),
    )
    reused_branch_wall = time.perf_counter() - reuse_started
    before_fresh = artifact_inventory(ds)
    fresh_stages = []
    fresh_started = time.perf_counter()
    fresh_modalities = {
        assay: prepare_modality(ds, selection, assay, fresh_stages, invalidate=True)
        for assay in design["modalities"]
    }
    fresh_graph, fresh_clusters = integrate(
        ds, fresh_modalities, joint_method, fresh_stages, invalidate=True, resolution=1.2
    )
    fresh_branch_wall = time.perf_counter() - fresh_started
    revision_freshness = verify_fresh_computation(fresh_stages, before_fresh)
    revised_values = np.asarray(ds.artifacts.load(revised_clusters)["values"][:])
    fresh_values = np.asarray(ds.artifacts.load(fresh_clusters)["values"][:])
    reuse_validation = {
        "clusterLabelsIdentical": bool(np.array_equal(revised_values, fresh_values)),
        "clusterAri": float(adjusted_rand_score(revised_values, fresh_values)),
        "graphEdgesIdentical": bool(
            np.array_equal(
                ds.artifacts.load(original_graph)["edges"][:],
                ds.artifacts.load(fresh_graph)["edges"][:],
            )
        ),
        "graphWeightsMaxAbs": float(
            np.max(
                np.abs(
                    ds.artifacts.load(original_graph)["weights"][:]
                    - ds.artifacts.load(fresh_graph)["weights"][:]
                )
            )
        ),
    }
    if joint_method == "wnn":
        reuse_validation["modalityWeightsMaxAbs"] = float(
            np.max(
                np.abs(
                    ds.artifacts.load(wnn)["modality_weights"][:]
                    - ds.artifacts.load(fresh_graph)["modality_weights"][:]
                )
            )
        )
    if (
        not reuse_validation["clusterLabelsIdentical"]
        or not reuse_validation["graphEdgesIdentical"]
        or reuse_validation["graphWeightsMaxAbs"] > 1e-6
        or (joint_method == "wnn" and reuse_validation["modalityWeightsMaxAbs"] > 1e-6)
    ):
        raise AssertionError(f"Reused and fresh revised analyses differ: {reuse_validation}")
    group = ds.artifacts.load(wnn)
    weights = np.asarray(group["modality_weights"][:])
    result = {
        "status": "ok",
        "dataset": dataset,
        "jointMethod": joint_method,
        "cells": len(selected),
        "cellSelectionSha256": array_hash(selected),
        "settings": SETTINGS,
        "labelColumn": label_column,
        "modalities": design["modalities"],
        "modalityDimensions": {
            assay: int(ds.artifacts.load(refs["reduction"])["data"].shape[1])
            for assay, refs in modalities.items()
        },
        "normalization": {
            assay: dict(ds.artifacts.load(refs["normalized"]).attrs["provenance"]["parameters"])
            for assay, refs in modalities.items()
        },
        "modalityWeightMean": dict(
            zip(design["modalities"], weights.mean(axis=0).tolist(), strict=True)
        ),
        "biologicalAgreement": metrics,
        "stages": stages,
        "reuseStages": reuse_stages,
        "freshRevisionStages": fresh_stages,
        "reuseValidation": reuse_validation,
        "publishedArtifactCount": len(published_artifacts),
        "artifactFreshness": {"initial": initial_freshness, "freshRevision": revision_freshness},
        "initialWnnSeconds": sum(
            s["wallSeconds"] for s in stages if not s["stage"].startswith("snn.")
        ),
        "initialSnnSeconds": sum(
            s["wallSeconds"] for s in stages if not s["stage"].startswith("wnn.")
        ),
        f"revised{joint_method.title()}Seconds": sum(s["wallSeconds"] for s in reuse_stages),
        f"freshRevised{joint_method.title()}Seconds": sum(s["wallSeconds"] for s in fresh_stages),
        "initialBranchWallSeconds": initial_branch_wall,
        "reusedBranchWallSeconds": reused_branch_wall,
        "freshBranchWallSeconds": fresh_branch_wall,
        "compilationWarmupCells": len(warmup_cells),
        "compilationWarmupStages": warmup_stages,
        "compilationWarmupWallSeconds": warmup_wall,
        "timingInterpretation": {
            "compilationWarmup": "The full workflow, both joint branches included, first runs on compilationWarmupCells cells so that compiled kernels are built before the timed phases; its stages are kept separately.",
            "initial": "Warm execution after the compilation warm-up. Direct wall time covers the selected jointMethod from preprocessing through clustering; method-specific stage totals sum the common upstream stages and each observed joint branch.",
            "revision": "Reused artifacts versus fresh recomputation at resolution 1.2, after initial analysis has warmed compiled kernels and source files",
        },
        "labelInterpretation": "Agreement with author annotations, which are not independent experimental ground truth"
        if labels is not None
        else "No independent reference labels are supplied",
        "reusedArtifacts": identities,
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
        "candidateBreadthInterpretation": "CyteArc's modality width, bandwidth and final degree are coupled at 20. Pure candidate breadth is varied only in Seurat with final k.nn fixed at 20.",
    }
    source_archive = (
        work / "source.tar.gz"
        if "storeArchiveUri" in payload
        else source.parent / "data.zarr.tar.gz"
    )
    if source_archive.exists():
        with source_archive.open("rb") as stream:
            result["sourceArchiveSha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json("multimodal_neighbors.json", result, directory=work)
    persist_measurement(payload, result)
    graphs = export_graphs(
        ds,
        selected,
        modalities,
        labels,
        None
        if design.get("groupColumn") is None
        else ds.cells.fetch_all(design["groupColumn"])[selected].astype(str),
        {"wnn": (wnn, wnn_clusters), "snn": (snn, snn_clusters)},
        {f"{joint_method}_revised": revised_clusters},
        work / "graphs.npz",
    )
    result["graphsSha256"] = hashlib.file_digest(graphs.open("rb"), "sha256").hexdigest()
    if payload.get("graphsUri"):
        from experiments.runtime.object_io import upload_file

        upload_file(graphs, payload["graphsUri"])
        result["graphsUri"] = payload["graphsUri"]
    if payload.get("inputsUri"):
        from experiments.runtime.object_io import upload_file

        archive = export_inputs(
            ds,
            selected,
            modalities,
            labels,
            wnn,
            wnn_clusters,
            snn,
            snn_clusters,
            work / "seurat_inputs",
            SETTINGS,
        )
        upload_file(archive, payload["inputsUri"])
        result["inputsUri"] = payload["inputsUri"]
        result["inputsSha256"] = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    path = write_json("multimodal_neighbors.json", result, directory=work)
    result["localRecord"] = str(path)
    return result


# The benchmark's UMAP settings; the k-means initialization keeps CyteArc's defaults with the experiment's seed.
UMAP_SETTINGS = {"n_epochs": 300, "random_seed": 4444, "parallel": True}


def graph_umaps(payload: dict) -> dict:
    """Lay out the RNA, protein and WNN graphs of a WNN-first run with UMAP.

    Rebuilds the run's first analysis of the same cells from the same store archive, and checks each rebuilt graph,
    partition and modality weight against the arrays the run saved at graphsUri. With harmonyBatchColumn, each
    modality's PCA is then corrected with Harmony for that column's batches and its graph, clusters and the WNN graph
    are built again on the corrected coordinates, reusing every artifact up to the PCA; the corrected graphs are saved
    at harmonyGraphsUri in the run's format, for the agreement evaluation, and they are the ones laid out. Each
    modality's connectivity map is laid out from k-means of its own coordinates, and the WNN graph from the RNA
    coordinates, as CyteArc's multimodal recipe starts a joint embedding from its first run's coordinates. Saves the
    three layouts with each cell's author label, study and WNN modality weights to the .npz beside the record.
    """
    from experiments.runtime.object_io import download_file, upload_file

    configure_cytearc()
    started = time.perf_counter()
    dataset = payload["dataset"]
    design = DATASETS[dataset]
    if design["modalities"] != ["RNA", "ADT"] or design.get("groupColumn") is None:
        raise ValueError(
            f"graph_umaps needs RNA and ADT graphs and a study column, {dataset} has {design}"
        )
    work = Path(payload.get("workDir", "/tmp/multimodal_umaps"))
    work.mkdir(parents=True, exist_ok=True)

    def sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    archive = work / "source.tar.gz"
    download_file(payload["storeArchiveUri"], archive)
    if sha256(archive) != payload["sourceArchiveSha256"]:
        raise ValueError("The store archive differs from the one the run analyzed")
    store = work / "analysis.zarr"
    store.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(store, filter="data")
    threads = payload.get("threads", 16)
    ds = cytearc.DataStore(
        str(store),
        default_assay="RNA",
        nthreads=threads,
        mem_budget=payload.get("memoryBudgetGiB", 32) * 1024**3,
    )
    selected = select_cells(ds, payload.get("cells"))
    if array_hash(selected) != payload["cellSelectionSha256"]:
        raise ValueError("The selected cells differ from the run's")
    mask = np.zeros(ds.cells.N, dtype=bool)
    mask[selected] = True
    ds.cells.insert("multimodal_cells", mask, overwrite=True)
    selection = ds.snapshot_cell_selection("multimodal_cells")
    labels = ds.cells.fetch_all(design["labelColumn"])[selected].astype(str)
    studies = ds.cells.fetch_all(design["groupColumn"])[selected].astype(str)
    stages = []
    modalities = {
        assay: prepare_modality(ds, selection, assay, stages, invalidate=True)
        for assay in design["modalities"]
    }
    wnn, wnn_clusters = integrate(ds, modalities, "wnn", stages, invalidate=True)
    rebuilt = export_graphs(
        ds,
        selected,
        modalities,
        labels,
        studies,
        {"wnn": (wnn, wnn_clusters)},
        {},
        work / "rebuilt_graphs.npz",
    )
    saved = work / "graphs.npz"
    download_file(payload["graphsUri"], saved)
    if sha256(saved) != payload["graphsSha256"]:
        raise ValueError("The saved graphs differ from the ones the run recorded")
    check = {}
    with (
        np.load(rebuilt, allow_pickle=False) as new,
        np.load(saved, allow_pickle=False) as old,
    ):
        for key in new.files:
            ours, theirs = new[key], old[key]
            same_shape = ours.shape == theirs.shape
            check[key] = {"identical": bool(same_shape and np.array_equal(ours, theirs))}
            if ours.dtype.kind == "f" and same_shape:
                difference = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
                check[key]["maxAbs"] = float(difference.max()) if difference.size else 0.0
    batch = payload.get("harmonyBatchColumn")
    harmony = None
    laid_out, laid_out_wnn = modalities, wnn
    if batch is not None:
        batches = ds.cells.fetch_all(batch)[selected].astype(str)
        correction_started = time.perf_counter()
        corrected = {}
        for assay in design["modalities"]:
            coordinates = measure(
                stages,
                f"{assay}.harmony",
                lambda assay=assay: ds.reduction.harmony(
                    modalities[assay]["reduction"], batch_columns=[batch]
                ),
            )
            index = measure(
                stages,
                f"{assay}.harmony_index",
                lambda coordinates=coordinates: ds.graph.ann_index(
                    coordinates, ann_parallel=False, rand_state=SETTINGS["seed"]
                ),
            )
            neighbors = measure(
                stages,
                f"{assay}.harmony_neighbors",
                lambda index=index: ds.graph.neighbors(index, k=SETTINGS["neighborCount"]),
            )
            graph = measure(
                stages,
                f"{assay}.harmony_connectivity",
                lambda neighbors=neighbors: ds.graph.connectivity(neighbors),
            )
            clusters = measure(
                stages,
                f"{assay}.harmony_clustering",
                lambda graph=graph: ds.clusters.leiden(
                    graph,
                    resolution=SETTINGS["resolution"],
                    random_seed=SETTINGS["seed"],
                    backend="cytearc",
                ),
            )
            corrected[assay] = {
                "reduction": coordinates,
                "neighbors": neighbors,
                "graph": graph,
                "clusters": clusters,
            }
        corrected_wnn = measure(
            stages,
            "wnn.harmony_integration",
            lambda: ds.integration.modalities(
                [corrected[assay]["neighbors"] for assay in design["modalities"]],
                method="wnn",
                l2_normalize=True,
            ),
        )
        corrected_clusters = measure(
            stages,
            "wnn.harmony_clustering",
            lambda: ds.clusters.leiden(
                corrected_wnn,
                resolution=SETTINGS["resolution"],
                random_seed=SETTINGS["seed"],
                backend="cytearc",
            ),
        )
        correction_seconds = time.perf_counter() - correction_started
        corrected_graphs = export_graphs(
            ds,
            selected,
            corrected,
            labels,
            studies,
            {"wnn": (corrected_wnn, corrected_clusters)},
            {},
            work / "harmony_graphs.npz",
        )
        upload_file(corrected_graphs, payload["harmonyGraphsUri"])
        partitions = {assay: corrected[assay]["clusters"] for assay in design["modalities"]}
        partitions["wnn"] = corrected_clusters
        harmony = {
            "batchColumn": batch,
            "batchCells": {
                name: int(count) for name, count in zip(*np.unique(batches, return_counts=True))
            },
            "parameters": {
                assay: dict(
                    ds.artifacts.load(corrected[assay]["reduction"]).attrs["provenance"][
                        "parameters"
                    ]
                )
                for assay in design["modalities"]
            },
            "correctionSeconds": correction_seconds,
            "biologicalAgreement": {
                name: label_metrics(np.asarray(ds.artifacts.load(ref)["values"][:]), labels)
                for name, ref in partitions.items()
            },
            "graphsUri": payload["harmonyGraphsUri"],
            "graphsSha256": sha256(corrected_graphs),
        }
        laid_out, laid_out_wnn = corrected, corrected_wnn
    starts = {
        assay: measure(
            stages,
            f"{assay}.initialization",
            lambda assay=assay: ds.embeddings.initialization(
                laid_out[assay]["reduction"], rand_state=SETTINGS["seed"]
            ),
        )
        for assay in design["modalities"]
    }
    graphs = {
        "RNA": (laid_out["RNA"]["graph"], starts["RNA"]),
        "ADT": (laid_out["ADT"]["graph"], starts["ADT"]),
        "WNN": (laid_out_wnn, starts["RNA"]),
    }
    layouts, parameters = {}, {}
    for name, (graph, start) in graphs.items():
        ref = measure(
            stages,
            f"{name}.umap",
            lambda graph=graph, start=start: ds.embeddings.umap(
                graph, start, nthreads=threads, **UMAP_SETTINGS
            ),
        )
        group = ds.artifacts.load(ref)
        layouts[name] = np.asarray(group["values"][:], dtype=np.float32)
        parameters[name] = dict(group.attrs["provenance"]["parameters"])
        if layouts[name].shape != (len(selected), 2) or not np.isfinite(layouts[name]).all():
            raise ValueError(f"The {name} layout does not give each cell finite coordinates")
    weights = np.asarray(ds.artifacts.load(laid_out_wnn)["modality_weights"][:], dtype=np.float32)
    if weights.shape != (len(selected), 2):
        raise ValueError(f"The WNN modality weights have shape {weights.shape}")
    label_names, label_codes = np.unique(labels, return_inverse=True)
    study_names, study_codes = np.unique(studies, return_inverse=True)
    path = work / "graph_umaps.npz"
    np.savez_compressed(
        path,
        **{f"{name}_umap": coordinates for name, coordinates in layouts.items()},
        label_codes=label_codes.astype(np.int16),
        label_names=label_names,
        study_codes=study_codes.astype(np.int16),
        study_names=study_names,
        modality_weights=weights,
        modalities=np.asarray(design["modalities"]),
    )
    uri = payload["recordUri"].removesuffix(".json") + ".npz"
    upload_file(path, uri)
    return {
        "status": "ok",
        "dataset": dataset,
        "cells": len(selected),
        "cellSelectionSha256": array_hash(selected),
        "labelColumn": design["labelColumn"],
        "groupColumn": design["groupColumn"],
        "unlabelledCells": int((labels == "").sum()),
        "graphsIdentical": all(entry["identical"] for entry in check.values()),
        "graphsCheck": check,
        "laidOut": "uncorrected" if harmony is None else "harmony",
        "harmony": harmony,
        "layoutStarts": {"RNA": "RNA", "ADT": "ADT", "WNN": "RNA"},
        "initializationParameters": {
            assay: dict(ds.artifacts.load(ref).attrs["provenance"]["parameters"])
            for assay, ref in starts.items()
        },
        "layoutParameters": parameters,
        "modalityWeightMean": dict(
            zip(design["modalities"], weights.mean(axis=0).tolist(), strict=True)
        ),
        "stages": stages,
        "umapUri": uri,
        "umapSha256": sha256(path),
        "settings": SETTINGS,
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
        "seconds": time.perf_counter() - started,
    }


def compare_seurat(payload: dict) -> dict:
    from experiments.runtime.object_io import download_file, upload_file

    configure_cytearc()
    work = Path(payload.get("workDir", "/tmp/multimodal_seurat"))
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "inputs.tar.gz"
    download_file(payload["inputsUri"], archive)
    with archive.open("rb") as stream:
        input_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    inputs = work / "inputs"
    inputs.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(inputs, filter="data")
    output = work / "seurat"
    output.mkdir()
    script = Path(__file__).with_name("multimodal_seurat.R")
    stages = []
    modes = ["matched", *map(str, SETTINGS["seuratCandidateBreadths"])]
    for mode in modes:
        measure(
            stages,
            f"seurat.{mode}.process",
            lambda: subprocess.run(
                [
                    "Rscript",
                    str(script),
                    str(inputs),
                    str(output),
                    mode,
                    str(payload.get("threads", 8)),
                ],
                check=True,
            ),
        )
    comparisons = {
        str(breadth): json.loads((output / f"measurements_{breadth}.json").read_text())
        for breadth in SETTINGS["seuratCandidateBreadths"]
    }
    cytearc_results = np.load(inputs / "cytearc.npz")
    labels_path = inputs / "labels.txt"
    labels = np.asarray(labels_path.read_text().splitlines()) if labels_path.exists() else None
    matched_weights = np.loadtxt(output / "matched_weights.csv", delimiter=",")
    matched_affinities = np.loadtxt(output / "matched_affinities.csv", delimiter=",")
    matched_neighbors = np.loadtxt(output / "matched_neighbors.csv", delimiter=",", dtype=int) - 1
    matched = {
        "weightsMaxAbs": float(np.max(np.abs(matched_weights - cytearc_results["weights"]))),
        "affinitiesMaxAbs": float(
            np.max(np.abs(matched_affinities - cytearc_results["affinities"]))
        ),
        "neighborsIdentical": bool(np.array_equal(matched_neighbors, cytearc_results["neighbors"])),
    }
    matched["passed"] = (
        matched["neighborsIdentical"]
        and max(matched["weightsMaxAbs"], matched["affinitiesMaxAbs"])
        <= SETTINGS["tolerances"]["matchedKernelMaxAbs"]
    )
    results = {}
    for breadth in SETTINGS["seuratCandidateBreadths"]:
        neighbors = np.loadtxt(output / f"neighbors_{breadth}.csv", delimiter=",", dtype=int) - 1
        weights = np.loadtxt(output / f"weights_{breadth}.csv", delimiter=",")
        clusters = np.loadtxt(output / f"clusters_{breadth}.csv", delimiter=",", dtype=int)
        results[str(breadth)] = {
            "cytearcNeighborOverlap": neighbor_overlap(cytearc_results["neighbors"], neighbors),
            "cytearcClusterAri": float(adjusted_rand_score(cytearc_results["clusters"], clusters)),
            "weightPearson": [
                float(np.corrcoef(weights[:, i], cytearc_results["weights"][:, i])[0, 1])
                for i in range(weights.shape[1])
            ],
            "biologicalAgreement": None if labels is None else label_metrics(clusters, labels),
        }
    neighbors = [
        np.loadtxt(output / f"neighbors_{b}.csv", delimiter=",", dtype=int)
        for b in SETTINGS["seuratCandidateBreadths"]
    ]
    weights = [
        np.loadtxt(output / f"weights_{b}.csv", delimiter=",")
        for b in SETTINGS["seuratCandidateBreadths"]
    ]
    candidate = {
        "neighborOverlap": neighbor_overlap(*neighbors),
        "weightsMaxAbs": float(np.max(np.abs(weights[0] - weights[1]))),
    }
    result = {
        "status": "ok",
        "matchedKernel": matched,
        "inputsUri": payload["inputsUri"],
        "inputsSha256": input_digest,
        "seuratComparisons": results,
        "candidateComparison": candidate,
        "seurat": comparisons,
        "stages": stages,
        "settings": SETTINGS,
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
    }
    write_json("multimodal_seurat.json", result, directory=work)
    persist_measurement(payload, result)
    if payload.get("outputsUri"):
        saved = shutil.make_archive(str(work / "seurat_outputs"), "gztar", output)
        upload_file(saved, payload["outputsUri"])
        result["outputsUri"] = payload["outputsUri"]
    return result


def integrate_prepared(payload: dict) -> dict:
    from cytearc.clustering.leiden import leiden_membership
    from cytearc.neighbors.integration import _wnn_integration_many
    from scipy.io import mmread

    from experiments.runtime.object_io import download_file

    configure_cytearc()
    work = Path(payload.get("workDir", "/tmp/prepared_multimodal"))
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "inputs.tar.gz"
    download_file(payload["inputsUri"], archive)
    with archive.open("rb") as stream:
        input_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    inputs = work / "inputs"
    inputs.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(inputs, filter="data")
    manifest = json.loads((inputs / "manifest.json").read_text())
    threads = payload.get("threads", 8)
    stages = []
    counts = measure(
        stages,
        "load_counts",
        lambda: [
            mmread(inputs / f"{name}_counts.mtx.gz", spmatrix=True).tocsc()
            for name in manifest["modalities"]
        ],
    )
    modalities = measure(
        stages,
        "load_reductions_and_neighbors",
        lambda: [
            (
                name,
                np.loadtxt(inputs / f"{name}_indices.csv.gz", delimiter=",", dtype=np.uint32),
                np.loadtxt(inputs / f"{name}_embedding.csv.gz", delimiter=",", dtype=np.float32),
            )
            for name in manifest["modalities"]
        ],
    )
    validations = {}
    for phase in ["first_call", "warm"]:
        graph, weights = measure(
            stages,
            f"{phase}.wnn",
            lambda: _wnn_integration_many(
                modalities,
                nthreads=threads,
                l2_normalize=SETTINGS["l2Normalize"],
            ),
        )
        clusters = measure(
            stages,
            f"{phase}.clustering",
            lambda graph=graph: leiden_membership(
                graph,
                resolution=SETTINGS["resolution"],
                random_seed=SETTINGS["seed"],
                backend="cytearc",
                nthreads=threads,
            ),
        )
        with np.load(inputs / "cytearc.npz") as saved:
            validations[phase] = {
                "neighborsIdentical": bool(
                    np.array_equal(
                        graph.col.reshape(manifest["cells"], SETTINGS["neighborCount"]),
                        saved["neighbors"],
                    )
                ),
                "weightsIdentical": bool(np.array_equal(weights, saved["weights"])),
                "clustersIdentical": bool(np.array_equal(clusters, saved["clusters"])),
            }
        if not all(validations[phase].values()):
            raise AssertionError(f"Prepared-input replay differs: {validations[phase]}")
        del graph, weights, clusters
    result = {
        "status": "ok",
        "cells": manifest["cells"],
        "modalities": manifest["modalities"],
        "modalityDimensions": {name: int(values.shape[1]) for name, _, values in modalities},
        "inputsUri": payload["inputsUri"],
        "threads": threads,
        "inputsSha256": input_digest,
        "settings": SETTINGS,
        "stages": stages,
        "validation": validations,
        "residentCountBytes": sum(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes for matrix in counts
        ),
        "residentInputs": "The same selected count matrices and reductions supplied to Seurat are loaded; CyteArc additionally consumes its saved modality neighbor rows, while Seurat computes neighbors from reductions.",
        "timingBoundary": "Joint WNN integration and clustering from prepared modality artifacts; input loading, output validation and persistence are separate. First-call and warm execution are both retained.",
        "provenance": provenance(),
        "region": os.environ.get("MODAL_REGION"),
    }
    write_json("multimodal_neighbors.json", result, directory=work)
    persist_measurement(payload, result)
    return result


def fetch_source(url: str, target: Path, sources: list[dict], *, md5: str | None = None) -> Path:
    """Download one source file, hash it and append its provenance to ``sources``."""
    import urllib.request

    started = time.perf_counter()
    sha256 = hashlib.sha256()
    checked = hashlib.md5() if md5 is not None else None
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as output:
        while chunk := response.read(8 * 1024**2):
            output.write(chunk)
            sha256.update(chunk)
            if checked is not None:
                checked.update(chunk)
    entry = {
        "url": url,
        "bytes": target.stat().st_size,
        "sha256": sha256.hexdigest(),
        "downloadSeconds": time.perf_counter() - started,
    }
    if checked is not None:
        if checked.hexdigest() != md5:
            raise ValueError(f"{url} has MD5 {checked.hexdigest()}, the repository lists {md5}")
        entry["md5"] = md5
    sources.append(entry)
    return target


class TenxCountsWriter:
    """Append cell rows to a Cell Ranger HDF5 matrix, the input format of CrToZarr.

    Features keep the order given; assays are the contiguous runs of one
    feature type, so RNA genes must precede the antibody panel.
    """

    def __init__(self, path: Path, names: list[str], types: list[str]):
        import h5py

        if len(names) != len(types):
            raise ValueError("Every feature needs one feature type")
        self.handle = h5py.File(path, "w")
        group = self.handle.create_group("matrix")
        self.data = group.create_dataset(
            "data", (0,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,)
        )
        self.indices = group.create_dataset(
            "indices", (0,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,)
        )
        self.indptr = [np.zeros(1, dtype=np.int64)]
        self.barcodes: list[np.ndarray] = []
        self.width = len(names)
        features = group.create_group("features")
        prefixes = {"Gene Expression": "RNA", "Antibody Capture": "ADT"}
        features["id"] = np.asarray(
            [f"{prefixes[t]}:{n}" for n, t in zip(names, types, strict=True)], dtype="S"
        )
        features["name"] = np.asarray(names, dtype="S")
        features["feature_type"] = np.asarray(types, dtype="S")
        features["genome"] = np.asarray(["GRCh38"] * len(names), dtype="S")

    def append(self, block: csr_matrix, barcodes: np.ndarray) -> None:
        if block.shape != (len(barcodes), self.width):
            raise ValueError(f"Block shape {block.shape} does not match {len(barcodes)} cells")
        block.sort_indices()
        values = np.asarray(block.data)
        counts = np.rint(values).astype(np.int64)
        if not np.array_equal(counts, values) or (
            counts.size and (counts.min() < 0 or counts.max() > np.iinfo(np.int32).max)
        ):
            raise ValueError("Count blocks must hold non-negative integers")
        total = self.data.shape[0]
        self.data.resize((total + block.nnz,))
        self.indices.resize((total + block.nnz,))
        self.data[total:] = counts.astype(np.int32)
        self.indices[total:] = block.indices.astype(np.int32)
        self.indptr.append(block.indptr[1:].astype(np.int64) + total)
        self.barcodes.append(np.asarray(barcodes, dtype=str))

    def close(self) -> dict:
        group = self.handle["matrix"]
        indptr = np.concatenate(self.indptr)
        barcodes = np.concatenate(self.barcodes)
        group.create_dataset("indptr", data=indptr)
        group.create_dataset("shape", data=np.array([self.width, len(barcodes)]))
        group["barcodes"] = np.asarray(barcodes, dtype="S")
        self.handle.close()
        return {"cells": len(barcodes), "features": self.width, "nonzeroEntries": int(indptr[-1])}


def store_atlas(
    work: Path,
    counts: Path,
    metadata: dict,
    payload: dict,
    stages: list[dict],
    cells: int,
    policy,
):
    """Convert a Cell Ranger HDF5 matrix to a CyteArc store with cell metadata and archive it.

    ``policy`` is the count-matrix layout, used exactly; None is CyteArc's
    default layout. A layout that does not fit the memory budget raises before
    the store is created.
    """
    from cytearc.readers import CrH5Reader
    from cytearc.writers import CrToZarr

    store = work / "data.zarr"
    budget = int(payload.get("memoryBudgetGiB", 48) * 1024**3)
    reader = CrH5Reader(str(counts))
    measure(
        stages,
        "conversion",
        lambda: CrToZarr(
            reader,
            str(store),
            nthreads=payload.get("threads", 8),
            mem_budget=budget,
            policy=policy,
        ).dump(),
    )
    ds = cytearc.DataStore(
        str(store), default_assay="RNA", nthreads=payload.get("threads", 8), mem_budget=budget
    )
    if ds.assay_names != ["ADT", "RNA"] or ds.cells.N != cells:
        raise ValueError("Converted atlas modalities or cells differ from the source")
    for column, values in metadata.items():
        ds.cells.insert(column, np.asarray(values), overwrite=True)
    if not np.all(ds.cells.fetch_all("I")):
        raise ValueError("Opening the atlas unexpectedly filtered author-selected cells")
    archive = Path(shutil.make_archive(str(work / "atlas_store"), "gztar", store))
    return ds, archive


def prepare_atlas(payload: dict) -> dict:
    import pandas as pd
    from scipy.io import mmread
    from scipy.sparse import hstack

    from experiments.runtime.object_io import upload_file

    configure_cytearc()
    work = Path(payload.get("workDir", "/tmp/hao_pbmc"))
    work.mkdir(parents=True, exist_ok=True)
    stages = []
    sources = []

    metadata = pd.read_csv(
        fetch_source(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_sc.meta.data_3P.csv.gz",
            work / "metadata.csv.gz",
            sources,
        ),
        index_col=0,
    )
    if len(metadata) != 161764 or not metadata.index.is_unique:
        raise ValueError("The atlas cell metadata differs from the fixed eligible release")
    matrices = []
    feature_names = []
    feature_types = []
    barcodes = None
    for accession, assay in [("GSM5008737", "RNA"), ("GSM5008738", "ADT")]:
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/{accession}/suppl/{accession}_{assay}_3P-"
        files = {
            kind: fetch_source(base + kind, work / f"{assay}_{kind}", sources)
            for kind in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]
        }
        cells = (
            pd.read_csv(files["barcodes.tsv.gz"], header=None, sep="\t")[0].astype(str).to_numpy()
        )
        if barcodes is None:
            barcodes = cells
        elif not np.array_equal(cells, barcodes):
            raise ValueError("RNA and ADT cell IDs or order differ")
        if not np.array_equal(cells, metadata.index.astype(str).to_numpy()):
            raise ValueError("Count cells do not exactly match the author metadata order")
        features = (
            pd.read_csv(files["features.tsv.gz"], header=None, sep="\t")[0].astype(str).to_numpy()
        )
        matrix = measure(
            stages,
            f"{assay}.read_sparse",
            lambda: mmread(
                files["matrix.mtx.gz"],
                spmatrix=True,
            ).T.tocsr(),
        )
        if matrix.shape != (161764, len(features)):
            raise ValueError(f"Unexpected {assay} shape {matrix.shape}")
        matrices.append(matrix)
        feature_names.extend(features.tolist())
        feature_types.extend(
            ["Gene Expression" if assay == "RNA" else "Antibody Capture"] * len(features)
        )
    combined = measure(
        stages, "combine_sparse_counts", lambda matrices=matrices: hstack(matrices, format="csr")
    )
    del matrices
    h5_path = work / "counts.h5"
    writer = TenxCountsWriter(h5_path, feature_names, feature_types)
    writer.append(combined, barcodes)
    writer.close()
    del combined
    ds, archive = store_atlas(
        work,
        h5_path,
        {column: metadata[column].to_numpy() for column in metadata},
        payload,
        stages,
        161764,
        None,
    )
    result = {
        "status": "ok",
        "dataset": "hao_pbmc_citeseq",
        "cells": ds.cells.N,
        "rnaFeatures": ds.RNA.feats.N,
        "adtFeatures": ds.ADT.feats.N,
        "donors": int(metadata["donor"].nunique()),
        "sourceRecords": sources,
        "stages": stages,
        "archiveBytes": archive.stat().st_size,
        "provenance": provenance(),
    }
    write_json("multimodal_neighbors.json", result, directory=work)
    persist_measurement(payload, result)
    upload_file(archive, payload["storeArchiveUri"])
    result["storeArchiveUri"] = payload["storeArchiveUri"]
    return result


def harmonized_labels(columns: dict[str, np.ndarray], rules: list[dict], cells: int) -> np.ndarray:
    """Apply ordered (column, values, label) rules; the first matching rule wins.

    Cells that no rule matches keep an empty label, which label_metrics excludes.
    """
    labels = np.full(cells, "", dtype=object)
    for rule in rules:
        values = np.asarray(columns[rule["column"]], dtype=object).astype(str)
        hit = (labels == "") & np.isin(values, rule["values"])
        labels[hit] = rule["label"]
    return labels.astype(str)


def prepare_h5ad_atlas(payload: dict) -> dict:
    """Stage paired RNA and protein counts from author h5ad files on one antibody panel.

    The atlas configuration names each source file, its raw count layer, the
    protein feature of every panel antibody, the cells to exclude and the
    rules that map author labels to one shared vocabulary. RNA keeps the genes
    that every source measures. Sources are written one after another into
    one Cell Ranger HDF5 matrix and converted to a CyteArc store.
    """
    from cytearc.readers import H5adReader

    from experiments.runtime.object_io import upload_file

    configure_cytearc()
    if "countMatrixPolicy" not in payload:
        raise ValueError(
            "prepare_h5ad_atlas needs countMatrixPolicy ({unitBytes, chunkBytes}); "
            "the layout decides the conversion memory and later read speed, so it is chosen explicitly"
        )
    policy = count_matrix_policy(payload["countMatrixPolicy"])
    config = json.loads((Path(__file__).parent / payload["atlasConfig"]).read_text())
    work = Path(payload.get("workDir", "/tmp/h5ad_atlas"))
    work.mkdir(parents=True, exist_ok=True)
    stages, sources = [], []
    readers = []
    for source in config["sources"]:
        url = payload.get("sourceUrls", {}).get(source["study"], source["url"])
        path = measure(
            stages,
            f"{source['study']}.download",
            # A payload URL replaces the published file, for a mirror or a local
            # fixture, and is hashed but not checked against the published MD5.
            lambda url=url, source=source: fetch_source(
                url,
                work / f"{source['study']}.h5ad",
                sources,
                md5=source["md5"] if url == source["url"] else None,
            ),
        )
        sources[-1]["study"] = source["study"]
        readers.append(H5adReader(str(path), matrix_key=source["countsMatrix"]))
        persist_progress(payload, {"stages": stages, "sourceRecords": sources})
    genes = None
    for source, reader in zip(config["sources"], readers, strict=True):
        ids = reader.feat_ids().astype(str)
        measured = [name for name in ids if not name.startswith(source["proteinPrefix"])]
        shared = set(measured)
        genes = measured if genes is None else [name for name in genes if name in shared]
    panel = config["proteinPanel"]
    names = genes + [entry["feature"] for entry in panel]
    types = ["Gene Expression"] * len(genes) + ["Antibody Capture"] * len(panel)
    writer = TenxCountsWriter(work / "counts.h5", names, types)
    metadata: dict[str, list[np.ndarray]] = {}
    summaries = []
    for source, reader in zip(config["sources"], readers, strict=True):
        study = source["study"]
        ids = reader.feat_ids().astype(str)
        target = np.full(len(ids), -1, dtype=np.int64)
        position = {name: index for index, name in enumerate(ids)}
        for index, gene in enumerate(genes):
            target[position[gene]] = index
        for index, entry in enumerate(panel):
            if entry[study] not in position:
                raise ValueError(f"{study} lacks the panel feature {entry[study]}")
            target[position[entry[study]]] = len(genes) + index
        columns = dict(reader.get_cell_columns())
        keep = np.ones(reader.nCells, dtype=bool)
        for column, values in source["excludeCells"].items():
            keep &= ~np.isin(np.asarray(columns[column], dtype=object).astype(str), values)
        labels = harmonized_labels(columns, source["labelRules"], reader.nCells)
        cells = np.char.add(f"{study}:", reader.cell_ids().astype(str))

        def append_blocks(reader=reader, keep=keep, cells=cells, target=target) -> int:
            written = 0
            offset = 0
            for block in reader.consume(20000):
                rows = keep[offset : offset + block.shape[0]]
                selected = block.tocsr()[rows].tocoo()
                columns_out = target[selected.col]
                kept = columns_out >= 0
                matrix = csr_matrix(
                    (selected.data[kept], (selected.row[kept], columns_out[kept])),
                    shape=(int(rows.sum()), len(names)),
                )
                writer.append(matrix, cells[offset : offset + block.shape[0]][rows])
                written += matrix.nnz
                offset += block.shape[0]
            if offset != reader.nCells:
                raise ValueError(f"{study} yielded {offset} of {reader.nCells} rows")
            return written

        nonzero = measure(stages, f"{study}.write_counts", append_blocks)
        for name, values in {
            "study": np.full(int(keep.sum()), study),
            "donor": np.asarray(columns[source["donorColumn"]], dtype=object).astype(str)[keep],
            "sample": np.asarray(columns[source["sampleColumn"]], dtype=object).astype(str)[keep],
            "condition": np.asarray(columns[source["conditionColumn"]], dtype=object).astype(str)[
                keep
            ],
            "author_cell_type": np.asarray(
                columns[source["authorLabelColumn"]], dtype=object
            ).astype(str)[keep],
            config["labelColumn"]: labels[keep],
        }.items():
            metadata.setdefault(name, []).append(values)
        persist_progress(payload, {"stages": stages, "sourceRecords": sources})
        kept_labels, label_counts = np.unique(labels[keep], return_counts=True)
        summaries.append(
            {
                "study": study,
                "sourceCells": int(reader.nCells),
                "keptCells": int(keep.sum()),
                "nonzeroEntries": int(nonzero),
                "donors": int(len(np.unique(metadata["donor"][-1]))),
                "labelCounts": dict(zip(kept_labels.tolist(), label_counts.tolist(), strict=True)),
            }
        )
        reader.close()
    counts = writer.close()
    persist_progress(payload, {"stages": stages, "sourceRecords": sources, "counts": counts})
    joined = {name: np.concatenate(values) for name, values in metadata.items()}
    ds, archive = store_atlas(
        work, work / "counts.h5", joined, payload, stages, counts["cells"], policy
    )
    result = {
        "status": "ok",
        "dataset": config["dataset"],
        "atlasConfig": payload["atlasConfig"],
        "cells": ds.cells.N,
        "rnaFeatures": ds.RNA.feats.N,
        "adtFeatures": ds.ADT.feats.N,
        "nonzeroEntries": counts["nonzeroEntries"],
        "countMatrixPolicy": {"unitBytes": policy.unitBytes, "chunkBytes": policy.chunkBytes},
        "memoryBudgetGiB": payload.get("memoryBudgetGiB", 48),
        "sources": summaries,
        "sourceRecords": sources,
        "stages": stages,
        "archiveBytes": archive.stat().st_size,
        "provenance": provenance(),
    }
    write_json("multimodal_neighbors.json", result, directory=work)
    persist_measurement(payload, result)
    if payload.get("storeArchiveUri"):
        upload_file(archive, payload["storeArchiveUri"])
        result["storeArchiveUri"] = payload["storeArchiveUri"]
    result["localArchive"] = str(archive)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.payload.read_text())), indent=2))
