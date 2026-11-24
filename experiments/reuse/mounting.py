import hashlib
import io
import json
import os
import posixpath
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from experiments.runtime.recording_store import RecordingStoreWrapper, StoreProbe

GIB = 1024**3
SIZE = 10000000
LINEAGE_COLUMN = "author_major_cell_cluster"
LINEAGE = "Endothelium"
BATCH_COLUMN = "author_experimental_id"
K_VALUES = (11, 15, 21)
RESOLUTIONS = (0.5, 1.0)
REUSE_RESOLUTIONS = (0.5, 2.0)
REUSE_K = 21
REUSE_MARKER_RESOLUTION = 2.0
PUBLISHED_STAGES = {
    "annIndex": "buildAnnIndex",
    "neighbors": "queryNeighbors",
    "graph": "buildConnectivityMap",
    "clusters": "runLeiden",
    "markers": "findMarkers",
}


def redact(text: str) -> str:
    import re

    return re.sub("s3://[^/\\s\\\"']+", "s3://<bucket>", text)


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {redact(message)}", flush=True)


class _Calls:
    """Listing and existence checks, which the probe does not count, shared by every clone."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = {"lists": 0, "existsChecks": 0}

    def add(self, kind: str) -> None:
        with self._lock:
            self._counts[kind] += 1

    def reset(self) -> None:
        with self._lock:
            self._counts = dict.fromkeys(self._counts, 0)

    def to_json(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


class _SourceStore(RecordingStoreWrapper):
    """The recording wrapper, also counting listing and existence checks."""

    def __init__(self, inner, *, probe: StoreProbe, calls: _Calls):
        super().__init__(inner, probe=probe)
        self.calls = calls

    def with_read_only(self, read_only: bool = False) -> "_SourceStore":
        return _SourceStore(
            self._inner.with_read_only(read_only), probe=self.probe, calls=self.calls
        )

    async def exists(self, key: str) -> bool:
        self.calls.add("existsChecks")
        return await self._inner.exists(key)

    def list(self):
        self.calls.add("lists")
        return self._inner.list()

    def list_prefix(self, prefix: str):
        self.calls.add("lists")
        return self._inner.list_prefix(prefix)

    def list_dir(self, prefix: str):
        self.calls.add("lists")
        return self._inner.list_dir(prefix)


class _ComponentProbe(StoreProbe):
    """The probe's totals, plus the bytes read from each part of the source (component())."""

    def __init__(self) -> None:
        super().__init__()
        self._component_bytes: dict[str, int] = {}

    def _reset_totals(self) -> None:
        super()._reset_totals()
        self._component_bytes = {}

    def record_transfer(self, kind: str, key: str, nbytes: int) -> None:
        super().record_transfer(kind, key, nbytes)
        if kind in ("get", "get_partial_values", "get_ranges"):
            part = component(key)
            with self._lock:
                self._component_bytes[part] = self._component_bytes.get(part, 0) + int(nbytes)

    def to_json(self) -> dict:
        totals = super().to_json()
        with self._lock:
            totals["readBytesByComponent"] = dict(sorted(self._component_bytes.items()))
        return totals


def _identity(location: str) -> str:
    if "://" in location and (not location.startswith("file://")):
        parts = urlsplit(location)
        return f"{parts.scheme.lower()}://{parts.netloc}{posixpath.normpath(parts.path or '/')}"
    return str(Path(location.removeprefix("file://")).expanduser().resolve())


@contextmanager
def recorded_source(source: str, probe: StoreProbe, calls: _Calls) -> Iterator[None]:
    from cytearc.storage import stores
    from zarr.storage import LocalStore

    original = stores.make_store
    wanted = _identity(source)

    def make_store(location, *, storage_options=None, read_only=False):
        store = original(location, storage_options=storage_options, read_only=read_only)
        if not isinstance(location, str) or _identity(location) != wanted:
            return store
        if not read_only:
            raise PermissionError("the source may only be opened read-only")
        if isinstance(store, str):
            store = LocalStore(store.removeprefix("file://"), read_only=True)
        return _SourceStore(store, probe=probe, calls=calls)

    stores.make_store = make_store
    try:
        yield
    finally:
        stores.make_store = original


def artifact_of(relative: str) -> tuple[str, str] | None:
    parts = relative.split("/")
    if parts[0] == "artifacts" and len(parts) > 3:
        return (parts[1], parts[2])
    if len(parts) > 4 and parts[1] == "artifacts":
        return (parts[2], parts[3])
    return None


def component(relative: str) -> str:
    parts = relative.split("/")
    found = artifact_of(relative)
    if found is not None:
        return f"artifact:{found[0]}"
    if parts[0] in ("cellData", "pipeline"):
        return parts[0]
    if len(parts) > 2 and parts[1] in ("counts", "countsT", "countSummaries", "featureData"):
        return parts[1]
    return "other"


def summarise(entries: list[tuple[str, int, str]]) -> dict:
    entries = sorted(entries)
    components: dict[str, int] = {}
    top: dict[str, int] = {}
    artifacts: dict[str, int] = {}
    for key, size, _ in entries:
        components[component(key)] = components.get(component(key), 0) + size
        top[key.split("/")[0]] = top.get(key.split("/")[0], 0) + size
        found = artifact_of(key)
        if found is not None:
            artifacts[found[1]] = artifacts.get(found[1], 0) + size
    return {
        "objects": len(entries),
        "bytes": sum((size for _, size, _ in entries)),
        "digest": hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
        "byTopLevel": dict(sorted(top.items())),
        "byComponent": dict(sorted(components.items())),
        "byArtifact": dict(sorted(artifacts.items())),
    }


def tree_entries(path: Path) -> list[tuple[str, int, str]]:
    entries = []
    for folder, _, files in os.walk(path):
        for name in files:
            full = os.path.join(folder, name)
            stat = os.stat(full)
            entries.append(
                (
                    os.path.relpath(full, path).replace(os.sep, "/"),
                    stat.st_size,
                    str(stat.st_mtime_ns),
                )
            )
    return entries


def tree_bytes(path: Path) -> int:
    return sum((size for _, size, _ in tree_entries(path))) if path.exists() else 0


def listing(location: str, storageOptions: dict | None) -> dict:
    if "://" not in location or location.startswith("file://"):
        return summarise(tree_entries(Path(location.removeprefix("file://"))))
    import obstore
    from obstore.store import from_url

    parts = urlsplit(location)
    store = from_url(f"{parts.scheme}://{parts.netloc}", **storageOptions or {})
    prefix = parts.path.strip("/") + "/"
    entries = []
    for page in obstore.list(store, prefix=prefix, chunk_size=1000):
        for item in page:
            entries.append(
                (str(item["path"])[len(prefix) :], int(item["size"]), str(item.get("e_tag")))
            )
    return summarise(entries)


@dataclass(frozen=True)
class Annotations:
    """Cell columns to add to the user's store, in the order of the source's cells.

    checkPositions and checkIds are cells whose identifiers the origin gives; the job refuses to
    add the columns unless the user's copy of the cell table holds the same identifiers there.
    """

    columns: dict[str, np.ndarray]
    origin: str
    checkPositions: np.ndarray
    checkIds: np.ndarray
    record: dict = field(default_factory=dict)


class _RangeFile(io.RawIOBase):
    """One object read by range requests in blocks of ``block`` bytes, for h5py."""

    def __init__(self, store, key: str, size: int, *, block: int = 1 << 20, cached: int = 256):
        self.store, self.key, self.size, self.block, self.cached = (store, key, size, block, cached)
        self.position = 0
        self.blocks: dict[int, bytes] = {}
        self.requests = 0
        self.bytesRead = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def _block(self, index: int) -> bytes:
        if index not in self.blocks:
            start = index * self.block
            data = bytes(
                self.store.get_range(self.key, start=start, end=min(self.size, start + self.block))
            )
            self.requests += 1
            self.bytesRead += len(data)
            if len(self.blocks) >= self.cached:
                self.blocks.pop(next(iter(self.blocks)))
            self.blocks[index] = data
        return self.blocks[index]

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.position)
        view = memoryview(buffer)
        done = 0
        while done < n:
            index, offset = divmod(self.position + done, self.block)
            chunk = self._block(index)[offset : offset + n - done]
            view[done : done + len(chunk)] = chunk
            done += len(chunk)
        self.position += max(n, 0)
        return max(n, 0)


def atlas_annotations(atlasUri: str, rows: np.ndarray, columns: tuple[str, ...]) -> Annotations:
    import h5py

    from experiments.runtime.object_io import open_r2_object

    started = time.perf_counter()
    store, key = open_r2_object(atlasUri)
    raw = _RangeFile(store, key, int(store.head(key)["size"]))
    values: dict[str, np.ndarray] = {}
    positions, ids = ([], [])
    with h5py.File(io.BufferedReader(raw, buffer_size=1 << 20), "r") as h5:
        obs = h5["obs"]
        for column in columns:
            node = obs[column]
            categories = np.asarray(node["categories"].asstr()[:], dtype=object)
            codes = np.asarray(node["codes"][:])[rows]
            if (codes < 0).any():
                raise ValueError(f"{column}: {(codes < 0).sum()} cells have no value")
            values[column] = categories[codes]
        index = obs["_index"]
        n, chunk = (int(index.shape[0]), int(index.chunks[0]))
        middle, last = (n // 2 // chunk * chunk, (n - 1) // chunk * chunk)
        for start, stop in ((0, min(chunk, n)), (middle, min(middle + chunk, n)), (last, n)):
            inside = np.flatnonzero((rows >= start) & (rows < stop))
            window = np.asarray(index.asstr()[start:stop], dtype=object)
            positions.append(inside)
            ids.append(window[rows[inside] - start])
    return Annotations(
        columns=values,
        origin="atlas H5AD obs columns, by range requests",
        checkPositions=np.concatenate(positions),
        checkIds=np.concatenate(ids).astype(str),
        record={
            "seconds": time.perf_counter() - started,
            "requests": raw.requests,
            "bytesRead": raw.bytesRead,
            "atlasRows": n,
        },
    )


@dataclass(frozen=True)
class MountOptions:
    """What run_mount does; workflow holds the benchmark's parameters (profiling WorkflowParameters)."""

    lineageColumn: str
    lineage: str
    batchColumn: str
    workflow: dict
    workers: int = 8
    budgetBytes: int = 24 * GIB
    kValues: tuple[int, ...] = K_VALUES
    resolutions: tuple[float, ...] = RESOLUTIONS
    markerK: int = 11
    markerResolution: float = 1.0
    umap: bool = True
    assay: str = "RNA"
    published: dict | None = None
    reuseResolutions: tuple[float, ...] = REUSE_RESOLUTIONS
    reuseK: int = REUSE_K
    reuseMarkerResolution: float = REUSE_MARKER_RESOLUTION


def clustering_name(k: int, resolution: float) -> str:
    return f"k{k}_r{resolution:g}"


class _Steps:
    """Runs each step and records its seconds, source reads, bytes written and peak memory."""

    def __init__(
        self, probe: StoreProbe, calls: _Calls, target: Path, onStep: Callable[[dict], None] | None
    ):
        self.probe, self.calls, self.target, self.onStep = (probe, calls, target, onStep)
        self.records: dict[str, dict] = {}

    def __call__(self, name: str, fn: Callable[[], Any]) -> Any:
        from experiments.runtime.metrics import ResourceSampler

        before = tree_bytes(self.target)
        sampler = ResourceSampler(sampleIntervalSeconds=0.25, resetCgroupPeak=False)
        sampler.start()
        self.probe.reset()
        self.calls.reset()
        started = time.perf_counter()
        value = fn()
        seconds = time.perf_counter() - started
        measurement = sampler.stop()
        operations = {**self.probe.to_json(), **self.calls.to_json()}
        after = tree_bytes(self.target)
        requests = sum(
            (
                operations[k]
                for k in (
                    "gets",
                    "rangeGets",
                    "partialGets",
                    "sizeQueries",
                    "lists",
                    "existsChecks",
                )
            )
        )
        self.records[name] = {
            "seconds": seconds,
            "sourceReadBytes": operations["readTransferredBytes"],
            "sourceReadByComponent": operations.get("readBytesByComponent", {}),
            "sourceReadRequests": requests,
            "sourceOperations": operations,
            "userStoreBytesBefore": before,
            "userStoreBytesAfter": after,
            "userStoreBytesWritten": after - before,
            "peakRssBytes": measurement.processTreeRssPeakBytes,
            "peakCgroupBytes": measurement.cgroupMemoryCurrentPeakBytes,
        }
        log(
            f"{name}: {seconds:.1f} s, read {operations['readTransferredBytes'] / 2**20:,.1f} MiB in {requests:,} requests from the source, wrote {(after - before) / 2**20:,.1f} MiB"
        )
        if self.onStep is not None:
            self.onStep(self.records)
        return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def lineage_export(
    ds, outputs: dict, clusterings: list[str], manifest: dict, sizes: dict[str, int]
) -> dict:
    import networkx as nx

    graph = ds.artifacts.lineage(outputs).graph
    order = list(
        nx.lexicographical_topological_sort(graph, key=lambda ref: (ref.kind, ref.artifact_id))
    )
    fingerprints = {
        value
        for entry in manifest["assays"].values()
        for key, value in entry.items()
        if key.endswith("Fingerprint")
    }
    nodes, counts_edges = ([], [])
    for ref in order:
        status = graph.nodes[ref]["status"]
        inputs = status.inputs or {}
        linked = {name for _, _, data in graph.in_edges(ref, data=True) for name in data["inputs"]}
        other = {
            name: value
            for name, value in inputs.items()
            if not any((n == name or n.startswith((f"{name}.", f"{name}[")) for n in linked))
        }
        parameters = _jsonable(status.parameters or {})
        nodes.append(
            {
                "id": ref.artifact_id,
                "kind": ref.kind,
                "scope": ref.scope,
                "operation": status.operation,
                "parameters": parameters,
                "otherInputs": _jsonable(other),
                "outputs": list(graph.nodes[ref]["outputs"]),
                "bytes": sizes.get(ref.artifact_id, 0),
            }
        )
        if fingerprints & set(_strings([parameters, other])):
            counts_edges.append(
                {"source": "counts", "target": ref.artifact_id, "inputs": ["dataset"]}
            )
    edges = [
        {
            "source": a.artifact_id,
            "target": b.artifact_id,
            "inputs": list(graph.edges[a, b]["inputs"]),
        }
        for a, b in graph.edges
    ]
    upstream = {}
    for name in clusterings:
        ref = outputs[name]
        ancestors = nx.ancestors(graph, ref)
        upstream[name] = [r.artifact_id for r in order if r in ancestors] + [ref.artifact_id]
    shared = set.intersection(*(set(ids) for ids in upstream.values()))
    return {
        "nodes": nodes,
        "edges": counts_edges + edges,
        "upstream": upstream,
        "sharedByAll": [node["id"] for node in nodes if node["id"] in shared],
        "counts": {
            "location": "source",
            "fingerprints": sorted(fingerprints),
            "joinedTo": [e["target"] for e in counts_edges],
        },
    }


def _everything(user, options: MountOptions):
    return user.features.snapshot(
        from_assay=options.assay, mask=np.ones(user.get_assay(options.assay).feats.N, dtype=bool)
    )


def reuse_published(
    user, steps: "_Steps", options: MountOptions, target: Path, *, everything_ref
) -> dict:
    import inspect

    from cytearc import ArtifactRef

    published = {role: ArtifactRef.from_dict(ref) for role, ref in options.published.items()}
    own_before = {
        found[1] for relative, _, _ in tree_entries(target) if (found := artifact_of(relative))
    }

    def listed():
        kinds = sorted({ref.kind for ref in published.values()})
        return {
            kind: [
                ref.artifact_id
                for ref in user.artifacts.list(
                    kind=kind, from_assay=options.assay, complete_only=True
                )
            ]
            for kind in kinds
        }

    visible = steps("reuse.list", listed)
    record: dict = {
        "published": {
            role: {
                "kind": ref.kind,
                "artifactId": ref.artifact_id,
                "listedOnMount": ref.artifact_id in visible.get(ref.kind, []),
                "inUserStore": ref.artifact_id in own_before,
            }
            for role, ref in published.items()
        },
        "steps": {},
    }

    def same(role: str, call) -> dict:
        ref = steps(f"reuse.same.{role}", call)
        written = steps.records[f"reuse.same.{role}"]["userStoreBytesWritten"]
        return {
            "artifactId": ref.artifact_id,
            "isPublished": ref.artifact_id == published[role].artifact_id,
            "bytesWritten": written,
        }

    record["steps"]["sameNeighbors"] = same(
        "neighbors", lambda: user.graph.neighbors(published["annIndex"], k=options.workflow["k"])
    )
    stored = user.artifacts.inspect(published["clusters"]).parameters or {}
    accepted = set(inspect.signature(user.clusters.leiden).parameters) - {
        "self",
        "graph",
        "invalidate_cache",
    }
    leiden_parameters = {key: value for key, value in stored.items() if key in accepted}
    record["publishedLeidenParameters"] = _jsonable(leiden_parameters)
    record["steps"]["sameClusters"] = same(
        "clusters", lambda: user.clusters.leiden(published["graph"], **leiden_parameters)
    )
    outputs = {}
    for resolution in options.reuseResolutions:
        name = f"reuse_leiden_published_r{resolution:g}"
        outputs[name] = steps(
            f"reuse.leiden.r{resolution:g}",
            lambda r=resolution: user.clusters.leiden(
                published["graph"],
                resolution=r,
                backend=options.workflow["leidenBackend"],
                random_seed=options.workflow["leidenSeed"],
            ),
        )
    neighbors = steps(
        f"reuse.neighbors.k{options.reuseK}",
        lambda: user.graph.neighbors(published["annIndex"], k=options.reuseK),
    )
    graph = steps(f"reuse.graph.k{options.reuseK}", lambda: user.graph.connectivity(neighbors))
    outputs[f"reuse_leiden_k{options.reuseK}_r1"] = steps(
        f"reuse.leiden.k{options.reuseK}_r1",
        lambda: user.clusters.leiden(
            graph,
            resolution=1.0,
            backend=options.workflow["leidenBackend"],
            random_seed=options.workflow["leidenSeed"],
        ),
    )
    marker_source = outputs[f"reuse_leiden_published_r{options.reuseMarkerResolution:g}"]
    outputs[f"reuse_markers_r{options.reuseMarkerResolution:g}"] = steps(
        f"reuse.markers.r{options.reuseMarkerResolution:g}",
        lambda: user.markers.search(
            marker_source,
            from_assay=options.assay,
            features=everything_ref,
            nthreads=options.workers,
        ),
    )
    clusters = {}
    for name, ref in outputs.items():
        if name.startswith("reuse_leiden"):
            values = np.asarray(user.artifacts.load(ref)["values"][:])
            clusters[name] = {
                "clusters": int(np.unique(values).size),
                "cells": int(values.size),
                "artifactId": ref.artifact_id,
            }
    names = [name for name in steps.records if name.startswith("reuse.")]
    record.update(
        clusterings=clusters,
        stepOrder=names,
        totals={
            "seconds": sum((steps.records[n]["seconds"] for n in names)),
            "sourceReadBytes": sum((steps.records[n]["sourceReadBytes"] for n in names)),
            "sourceReadByComponent": {
                part: sum((steps.records[n]["sourceReadByComponent"].get(part, 0) for n in names))
                for part in sorted(
                    {part for n in names for part in steps.records[n]["sourceReadByComponent"]}
                )
            },
            "userStoreBytesWritten": sum(
                (steps.records[n]["userStoreBytesWritten"] for n in names)
            ),
        },
        outputs=outputs,
    )
    return record


def run_mount(
    source: str,
    target: Path,
    options: MountOptions,
    *,
    storageOptions: dict | None = None,
    annotations: Annotations | None = None,
    onStep: Callable[[dict], None] | None = None,
) -> dict:
    import cytearc
    import pandas as pd
    import zarr
    from cytearc.storage.identity import count_fingerprint

    if target.exists():
        raise FileExistsError(f"{target} exists; the user's store must be new")
    target.parent.mkdir(parents=True, exist_ok=True)
    workflow = options.workflow
    probe, calls = (_ComponentProbe(), _Calls())
    steps = _Steps(probe, calls, target, onStep)
    source_before = listing(source, storageOptions)
    log(f"source: {source_before['objects']:,} objects, {source_before['bytes'] / GIB:.2f} GiB")
    datastore_options = {
        "default_assay": options.assay,
        "nthreads": options.workers,
        "mem_budget": options.budgetBytes,
        "storage_options": storageOptions,
    }
    with recorded_source(source, probe, calls):
        user = steps(
            "mount", lambda: cytearc.mount_datastore(source, at=str(target), **datastore_options)
        )
        copied = summarise(tree_entries(target))
        n_cells = int(user.cells.N)
        added: list[str] = []
        if annotations is not None:
            ids = np.asarray(user.cells.fetch_all("ids"))[annotations.checkPositions]
            if not np.array_equal(ids.astype(str), annotations.checkIds):
                raise ValueError(
                    "the annotation's cell identifiers differ from the user's cell table"
                )

            def annotate() -> None:
                for column, values in annotations.columns.items():
                    if len(values) != n_cells:
                        raise ValueError(f"{column}: {len(values)} values for {n_cells} cells")
                    user.cells.insert(column, np.asarray(values))
                    added.append(column)

            steps("annotate", annotate)
        active = np.asarray(user.cells.fetch_all("I"), dtype=bool)
        labels = np.asarray(user.cells.fetch_all(options.lineageColumn))
        in_lineage = active & (labels == options.lineage)
        names, counts = np.unique(labels[active], return_counts=True)
        label_counts = pd.Series(counts, index=names.astype(str)).sort_values(ascending=False)
        del labels, names, counts
        batches = np.asarray(user.cells.fetch_all(options.batchColumn))[in_lineage]
        batch_counts = pd.Series(batches).value_counts()
        del batches
        if not in_lineage.any():
            raise ValueError(f"no active cell has {options.lineageColumn} = {options.lineage!r}")
        lineage_key = "lineage_" + "".join((c if c.isalnum() else "_" for c in options.lineage))

        def select_lineage():
            user.cells.insert(lineage_key, in_lineage)
            return user.snapshot_cell_selection(lineage_key)

        cells = steps("selectLineage", select_lineage)
        genes = steps(
            "selectGenes",
            lambda: user.features.hvgs(
                cells,
                from_assay=options.assay,
                min_cells=workflow["hvgMinCells"],
                top_n=workflow["topN"],
                show_plot=False,
            ),
        )
        normalized = steps("normalize", lambda: user.features.normalize(cells, genes))
        pca = steps(
            "pca",
            lambda: user.reduction.pca(
                normalized,
                dims=workflow["dims"],
                local_cache=workflow["graphLocalCache"],
                show_elbow_plot=False,
            ),
        )
        harmony = steps("harmony", lambda: user.reduction.harmony(pca, [options.batchColumn]))
        ef = min(100, max(workflow["k"] * 3, 50))
        ann = {
            "ann_efc": ef,
            "ann_ef": ef,
            "ann_m": min(max(48, int(workflow["dims"] * 1.5)), 64),
            "ann_parallel": workflow["annParallel"],
            "rand_state": workflow["graphSeed"],
        }
        index = steps("annIndex", lambda: user.graph.ann_index(harmony, **ann))
        graphs, clusterings = ({}, {})
        for k in options.kValues:
            neighbors = steps(f"neighbors.k{k}", lambda k=k: user.graph.neighbors(index, k=k))
            graphs[k] = steps(f"graph.k{k}", lambda n=neighbors: user.graph.connectivity(n))
            for resolution in options.resolutions:
                name = clustering_name(k, resolution)
                clusterings[name] = steps(
                    f"leiden.{name}",
                    lambda k=k, r=resolution: user.clusters.leiden(
                        graphs[k],
                        resolution=r,
                        backend=workflow["leidenBackend"],
                        random_seed=workflow["leidenSeed"],
                    ),
                )
        marker_name = clustering_name(options.markerK, options.markerResolution)

        def markers():
            everything = user.features.snapshot(
                from_assay=options.assay,
                mask=np.ones(user.get_assay(options.assay).feats.N, dtype=bool),
            )
            return user.markers.search(
                clusterings[marker_name],
                from_assay=options.assay,
                features=everything,
                nthreads=options.workers,
            )

        marker_table = steps(f"markers.{marker_name}", markers)
        outputs = {**clusterings, f"markers_{marker_name}": marker_table}
        if options.umap:
            initialization = steps(
                "umapInitialization",
                lambda: user.embeddings.initialization(
                    harmony,
                    n_centroids=workflow["nCentroids"],
                    rand_state=workflow["graphSeed"],
                    kmeans_sampling=workflow["kmeansSampling"],
                    kmeans_batch_size=workflow["kmeansBatchSize"],
                ),
            )
            umap = steps(
                f"umap.k{options.markerK}",
                lambda: user.embeddings.umap(
                    graphs[options.markerK],
                    initialization,
                    n_epochs=workflow["umapEpochs"],
                    random_seed=workflow["umapSeed"],
                    parallel=workflow["umapParallel"],
                    nthreads=options.workers,
                ),
            )
            outputs[f"umap_k{options.markerK}"] = umap
        reuse = None
        if options.published:
            reuse = reuse_published(
                user, steps, options, target, everything_ref=_everything(user, options)
            )
            outputs.update(reuse.pop("outputs"))
        reopened = steps(
            "reopen", lambda: cytearc.DataStore(str(target), zarr_mode="r", **datastore_options)
        )
        matrix = reopened.get_assay(options.assay).matrixGroup
        manifest = dict(zarr.open_group(str(target), mode="r").attrs["matrixSource"])
        expected = manifest["assays"][options.assay]
        reopen = {
            "countsRemountedFromSource": isinstance(matrix.store, _SourceStore)
            and matrix.store.probe is probe,
            "manifestNamesSource": _identity(manifest["location"]) == _identity(source),
            "countsFingerprintMatches": count_fingerprint(matrix["counts"])
            == expected["countsFingerprint"],
            "datasetFingerprintMatches": reopened.get_assay(options.assay).z.attrs.get(
                "dataset_fingerprint"
            )
            == expected["datasetFingerprint"],
            "requiresTranspose": expected["requiresTranspose"],
        }
        reopen["fingerprintCheckPassed"] = all(
            (
                reopen[k]
                for k in (
                    "countsRemountedFromSource",
                    "manifestNamesSource",
                    "countsFingerprintMatches",
                    "datasetFingerprintMatches",
                )
            )
        )
        final = summarise(tree_entries(target))
        lineage = lineage_export(
            reopened, outputs, list(clusterings), manifest, final["byArtifact"]
        )
        if reuse is not None:
            reuse["lineageGraph"] = lineage_export(
                reopened,
                {name: ref for name, ref in outputs.items() if name.startswith("reuse_")},
                [name for name in outputs if name.startswith("reuse_leiden")],
                manifest,
                final["byArtifact"],
            )
        clusters = {}
        for name, ref in clusterings.items():
            values = np.asarray(reopened.artifacts.load(ref)["values"][:])
            k, resolution = name.removeprefix("k").split("_r")
            clusters[name] = {
                "k": int(k),
                "resolution": float(resolution),
                "clusters": int(np.unique(values).size),
                "cells": int(values.size),
                "artifactId": ref.artifact_id,
            }
        n_genes = int(reopened.get_assay(options.assay).feats.N)
    source_after = listing(source, storageOptions)
    records = steps.records
    return {
        "experiment": "mount",
        "status": "ok",
        "source": {
            "cells": n_cells,
            "genes": n_genes,
            "before": source_before,
            "after": source_after,
            "unchanged": source_before == source_after,
        },
        "copied": {
            name: {
                "sourceBytes": source_before["byComponent"].get(name, 0),
                "userStoreBytes": copied["byComponent"].get(name, 0),
            }
            for name in ("cellData", "featureData")
        },
        "referenced": {
            name: {"sourceBytes": source_before["byComponent"].get(name, 0)}
            for name in ("counts", "countsT", "countSummaries")
        },
        "notCopied": {
            "publishedResultsBytes": sum(
                (
                    v
                    for k, v in source_before["byComponent"].items()
                    if k.startswith("artifact:") or k == "pipeline"
                )
            ),
            "publishedResultKinds": sorted(
                (
                    k.removeprefix("artifact:")
                    for k in source_before["byComponent"]
                    if k.startswith("artifact:")
                )
            ),
        },
        "userStore": {"afterMount": copied, "final": final},
        "lineage": {
            "column": options.lineageColumn,
            "value": options.lineage,
            "cells": int(in_lineage.sum()),
            "activeCells": int(active.sum()),
            "labelCounts": {str(k): int(v) for k, v in label_counts.items()},
        },
        "batch": {
            "column": options.batchColumn,
            "batches": int(batch_counts.size),
            "cellsPerBatch": {str(k): int(v) for k, v in batch_counts.items()},
        },
        "annotations": None
        if annotations is None
        else {
            "origin": annotations.origin,
            "columns": added,
            "idsChecked": int(annotations.checkPositions.size),
            **annotations.record,
        },
        "parameters": {
            "workflow": workflow,
            "ann": ann,
            "kValues": list(options.kValues),
            "resolutions": list(options.resolutions),
            "markers": marker_name,
            "umap": options.umap,
            "workers": options.workers,
            "budgetBytes": options.budgetBytes,
        },
        "stepOrder": list(records),
        "steps": records,
        "totals": {
            "seconds": sum((r["seconds"] for r in records.values())),
            "sourceReadBytes": sum((r["sourceReadBytes"] for r in records.values())),
            "sourceReadRequests": sum((r["sourceReadRequests"] for r in records.values())),
            "userStoreBytesWritten": final["bytes"],
            "peakRssBytes": max((r["peakRssBytes"] or 0 for r in records.values())),
        },
        "clusterings": clusters,
        "markers": {"clustering": marker_name, "artifactId": marker_table.artifact_id},
        "reopen": reopen,
        "lineageGraph": lineage,
        "reuse": reuse,
    }


def mount_job(payload: dict) -> dict:
    os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    import cytearc

    from experiments.runtime.embryo_atlas import select_nested_rows
    from experiments.runtime.object_io import put_json, storage_options

    cytearc.configure_output(level="WARNING", progress=False)
    started = time.perf_counter()
    head = {
        "experiment": "mount",
        "referenceTag": payload["referenceTag"],
        "sourceStore": payload["sourceStoreLabel"],
    }
    last: dict = {}

    def progress(records: dict) -> None:
        last["steps"] = records
        put_json(payload["outputUri"], {**head, "status": "running", "steps": records})

    log(
        f"start: mount {payload['sourceStoreLabel']}, lineage {payload['lineage']!r}, batch {payload['batchColumn']!r}"
    )
    try:
        rows = select_nested_rows(
            int(payload["atlasRows"]), (int(payload["size"]),), seed=int(payload["samplingSeed"])
        )[int(payload["size"])]
        annotations = atlas_annotations(
            payload["atlasUri"], rows, (payload["lineageColumn"], payload["batchColumn"])
        )
        log(
            f"annotation: {annotations.record['requests']} requests, {annotations.record['bytesRead'] / 2**20:.0f} MiB from the atlas file"
        )
        options = MountOptions(
            lineageColumn=payload["lineageColumn"],
            lineage=payload["lineage"],
            batchColumn=payload["batchColumn"],
            workflow=payload["workflow"],
            workers=int(payload["workers"]),
            budgetBytes=int(payload["budgetBytes"]),
            published=payload.get("published"),
        )
        source = payload["sourceStoreUri"]
        result = run_mount(
            source,
            Path("/tmp/manuscript-mount/user.zarr"),
            options,
            storageOptions=storage_options(source),
            annotations=annotations,
            onStep=progress,
        )
    except Exception as error:
        put_json(
            payload["outputUri"],
            {
                **head,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "steps": last.get("steps", {}),
            },
        )
        raise
    result.update(
        {
            **head,
            "sourceSampling": {
                "atlasRows": int(payload["atlasRows"]),
                "size": int(payload["size"]),
                "seed": int(payload["samplingSeed"]),
            },
            "userStoreLocation": "container local disk",
            "container": {
                "cpu": 8,
                "memoryMb": 32768,
                "workers": options.workers,
                "budgetBytes": options.budgetBytes,
            },
            "jobSeconds": time.perf_counter() - started,
        }
    )
    put_json(payload["outputUri"], result)
    log(
        f"done: read {result['totals']['sourceReadBytes'] / GIB:.2f} GiB from the source, the user's store holds {result['totals']['userStoreBytesWritten'] / 2**20:.0f} MiB"
    )
    return {
        "status": "ok",
        "sourceReadBytes": result["totals"]["sourceReadBytes"],
        "userStoreBytes": result["totals"]["userStoreBytesWritten"],
    }


def publish_analysis(ds, workflow: dict, workers: int, assay: str = "RNA") -> dict:
    cells = ds.snapshot_cell_selection("I")
    genes = ds.features.hvgs(
        cells,
        from_assay=assay,
        min_cells=workflow["hvgMinCells"],
        top_n=workflow["topN"],
        show_plot=False,
    )
    normalized = ds.features.normalize(cells, genes)
    pca = ds.reduction.pca(
        normalized,
        dims=workflow["dims"],
        local_cache=workflow["graphLocalCache"],
        show_elbow_plot=False,
    )
    ef = min(100, max(workflow["k"] * 3, 50))
    index = ds.graph.ann_index(
        pca,
        ann_efc=ef,
        ann_ef=ef,
        ann_m=min(max(48, int(workflow["dims"] * 1.5)), 64),
        ann_parallel=workflow["annParallel"],
        rand_state=workflow["graphSeed"],
    )
    neighbors = ds.graph.neighbors(index, k=workflow["k"])
    graph = ds.graph.connectivity(neighbors)
    clusters = ds.clusters.leiden(
        graph,
        resolution=workflow["leidenResolution"],
        backend=workflow["leidenBackend"],
        random_seed=workflow["leidenSeed"],
    )
    everything = ds.features.snapshot(
        from_assay=assay, mask=np.ones(ds.get_assay(assay).feats.N, dtype=bool)
    )
    markers = ds.markers.search(clusters, from_assay=assay, features=everything, nthreads=workers)
    refs = {
        "annIndex": index,
        "neighbors": neighbors,
        "graph": graph,
        "clusters": clusters,
        "markers": markers,
    }
    return {role: ref.to_dict() for role, ref in refs.items()}


def mount_test_job(payload: dict) -> dict:
    import shutil

    import cytearc
    import zarr

    from experiments.runtime.object_io import put_json

    os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    cytearc.configure_output(level="WARNING", progress=False)
    work = Path("/tmp/manuscript-mount-test")
    cytearc.datasets.connect("cytearc_docs").download_dataset(
        payload["dataset"], destination=str(work / "catalogue"), zarr=True
    )
    source = work / "source.zarr"
    shutil.copytree(work / "catalogue" / payload["dataset"] / "data.zarr", source)
    ds = cytearc.DataStore(str(source), default_assay="RNA", nthreads=8, mem_budget=4 * GIB)
    workflow = payload["workflow"]
    published = publish_analysis(ds, workflow, 8)
    del ds
    ids = np.asarray(zarr.open_group(str(source), mode="r")["cellData/ids"][:]).astype(str)
    batch = np.where(np.random.default_rng(0).random(ids.size) < 0.5, "batch 1", "batch 2")
    annotations = Annotations(
        columns={"pseudo_batch": batch},
        origin="random two-way split, seed 0: the Kang store has no donor column",
        checkPositions=np.arange(0, ids.size, 97),
        checkIds=ids[::97],
    )
    options = MountOptions(
        lineageColumn=payload["lineageColumn"],
        lineage=payload["lineage"],
        batchColumn="pseudo_batch",
        workflow=workflow,
        workers=8,
        budgetBytes=4 * GIB,
        published=published,
    )
    result = run_mount(str(source), work / "user.zarr", options, annotations=annotations)
    result.update(
        {
            "draft": "test of the job on a catalogue store with a published analysis, not the atlas-scale result",
            "sourceStore": payload["dataset"],
            "cytearcVersion": cytearc.__version__,
        }
    )
    put_json(payload["outputUri"], json.loads(json.dumps(result, default=str)))
    reuse = result["reuse"]
    return {
        "status": result["status"],
        "sourceUnchanged": result["source"]["unchanged"],
        "reopen": result["reopen"]["fingerprintCheckPassed"],
        "reuse": {k: reuse["steps"][k] for k in reuse["steps"]},
        "reuseTotals": reuse["totals"],
        "publishedListed": {
            role: item["listedOnMount"] for role, item in reuse["published"].items()
        },
    }
