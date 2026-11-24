import hashlib
import io
import os
import re
import sys
import time
import traceback
from pathlib import Path

import experiments.benchmarks.workflows as tools

CPU, MEMORY_MB, WORKERS = (8.0, 32768, 8)
MAJOR_COLUMN = "author_major_cell_cluster"
FINE_CANDIDATES = (
    "author_cell_type",
    "author_celltype_update",
    "author_celltype",
    "author_cell_type_update",
)
NEIGHBOURS = 15
RARE_SHARE = 0.01
ID_WINDOWS = 16
RUN_FACTS = (
    "tool",
    "dataset",
    "replicate",
    "status",
    "container",
    "image",
    "versions",
    "mode",
    "restarts",
    "workflowSeconds",
    "peakCgroupBytes",
    "nCells",
    "nGenes",
    "nnz",
    "datasetUri",
    "cellsAfterFilter",
    "genesAfterFilter",
    "clusters",
    "keptOutputs",
    "workers",
    "budgetBytes",
    "countPolicy",
    "pcaParameters",
    "leidenBackend",
)
METHOD = {
    "cells": "the input cells that every tool retains (the intersection of their retained input rows)",
    "labels": "the atlas's author annotation for the atlas rows that the profiler sampled, read from the atlas H5AD",
    "clusterAgreement": "scikit-learn adjusted_rand_score and normalized_mutual_info_score (arithmetic mean)",
    "neighbours": "the 15 nearest other common cells by exact Euclidean distance (scikit-learn NearestNeighbors: brute force on the 21 principal components, a k-d tree on the 2-D UMAP), in each tool's own coordinates",
    "neighbourShare": "the share of a cell's neighbours that carry its label: mean over labelled cells (cells) and mean of the class means (classes); chance: the sum of squared class shares",
    "jaccard": "|A & B| / |A | B| of a cell's neighbour sets from two tools",
    "rare": "classes below rareShare of the labelled common cells; per class and tool the cluster with the largest F1 = 2|class & cluster| / (|class| + |cluster|), its recall |class & cluster| / |class| and precision |class & cluster| / |cluster|",
}


def _relative(payload: dict):
    campaign = payload["campaignUri"].rstrip("/") + "/"
    datasets = payload["datasetsUri"].rstrip("/") + "/"

    def relative(uri: str) -> str:
        if uri.startswith(campaign):
            return "<campaign>/" + uri[len(campaign) :]
        if uri.startswith(datasets):
            return "<datasets>/" + uri[len(datasets) :]
        return re.sub("s3://[^/\\s\\\"']+", "s3://<bucket>", uri)

    return relative


def _redact(text: str, payload: dict) -> str:
    from urllib.parse import urlsplit

    for key, placeholder in (
        ("campaignUri", "<campaign>"),
        ("datasetsUri", "<datasets>"),
        ("resultsUri", "<results>"),
        ("r2EndpointUrl", "<endpoint>"),
    ):
        root = str(payload.get(key) or "").rstrip("/")
        if root:
            text = text.replace(root, placeholder)
    text = re.sub("s3://[^/\\s\\\"']+", "s3://<bucket>", text)
    for key, placeholder in (
        ("datasetsUri", "<bucket>"),
        ("resultsUri", "<bucket>"),
        ("r2EndpointUrl", "<endpoint>"),
    ):
        host = urlsplit(str(payload.get(key) or "")).netloc
        if host:
            text = text.replace(host, placeholder)
    return text


def finished_run(r2, entry: dict, relative) -> dict:
    if "recordUri" in entry:
        attempt = {"uri": entry["recordUri"]}
    else:
        ladder = r2.get_json(entry["ladderUri"]) if r2.exists(entry["ladderUri"]) else {}
        if ladder.get("status") != "ok":
            raise RuntimeError(
                f"{relative(entry['ladderUri'])} has status {ladder.get('status', 'missing')!r}; every ladder must be ok"
            )
        attempt = [a for a in ladder.get("attempts", []) if a.get("status") == "ok"][-1]
    run = r2.get_json(attempt["uri"])
    kept = run.get("keptOutputs") or {}
    if run.get("status") != "ok" or not kept.get("name"):
        raise RuntimeError(
            f"{relative(attempt['uri'])} kept no outputs ({kept.get('error') or 'none named in the record'}); run its ladder again under another replicate and pass --replicates {entry['tool']}=N"
        )
    outputs = attempt["uri"].replace(".json", "_outputs.h5")
    facts = {key: run.get(key) for key in RUN_FACTS if key in run}
    facts.update(
        {
            "ladder": relative(entry["ladderUri"]) if "ladderUri" in entry else None,
            "record": relative(attempt["uri"]),
            "outputs": relative(outputs),
            "memoryGib": attempt.get("memoryGib", run["container"]["memoryMb"] / 1024),
            "keepOutputsSeconds": ((run.get("steps") or {}).get("keepOutputs") or {}).get(
                "seconds"
            ),
        }
    )
    return {"outputsUri": outputs, "facts": facts}


class RangeFile(io.RawIOBase):
    """One object read by range requests in blocks of ``block`` bytes, for h5py (modal_mount._RangeFile on boto3)."""

    def __init__(
        self,
        client,
        bucket: str,
        key: str,
        *,
        block: int = 1 << 20,
        cached: int = 256,
        url: str | None = None,
    ):
        self.client, self.bucket, self.key = (client, bucket, key)
        self.url = url
        if url:
            from urllib.request import Request, urlopen

            with urlopen(Request(url, method="HEAD"), timeout=120) as response:
                self.size = int(response.headers["Content-Length"])
        else:
            self.size = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
        self.block, self.cached = (block, cached)
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
            end = min(self.size, start + self.block)
            if self.url:
                from urllib.request import Request, urlopen

                request = Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urlopen(request, timeout=120) as response:
                    expected = f"bytes {start}-{end - 1}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise OSError("Atlas server did not return the requested byte range")
                    data = response.read()
            else:
                data = self.client.get_object(
                    Bucket=self.bucket, Key=self.key, Range=f"bytes={start}-{end - 1}"
                )["Body"].read()
            if len(data) != end - start:
                raise OSError("Atlas read returned an incomplete byte range")
            self.requests += 1
            self.bytesRead += len(data)
            if len(self.blocks) >= self.cached:
                self.blocks.pop(next(iter(self.blocks)))
            self.blocks[index] = data
        return self.blocks[index]

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.position)
        view = memoryview(buffer).cast("B")
        done = 0
        while done < n:
            index, offset = divmod(self.position + done, self.block)
            chunk = self._block(index)[offset : offset + n - done]
            view[done : done + len(chunk)] = chunk
            done += len(chunk)
        self.position += max(n, 0)
        return max(n, 0)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _categorical(obs, name: str) -> bool:
    import h5py

    node = obs.get(name)
    return isinstance(node, h5py.Group) and "codes" in node and ("categories" in node)


def label_columns(
    columns: list[str], categorical: set[str], requested: str
) -> dict[str, tuple[str, str]]:
    wanted = {MAJOR_COLUMN: ("major", "the atlas's major cell clusters")}
    if requested:
        wanted.setdefault(requested, ("fine", "requested"))
    else:
        named = sorted(
            (
                name
                for name in columns
                if name in categorical
                and name.startswith("author_")
                and (name != MAJOR_COLUMN)
                and ("cell_type" in name or "celltype" in name)
                and ("ontology" not in name)
            )
        )
        present = [name for name in FINE_CANDIDATES if name in categorical]
        if present:
            wanted[present[0]] = ("fine", "the first of the candidate author columns present")
        elif named:
            wanted[named[0]] = ("fine", "an author column named like a cell type")
    if "cell_type" in categorical and "cell_type" not in wanted:
        wanted["cell_type"] = (
            "ontology",
            "CELLxGENE's cell_type: Cell Ontology terms submitted with the dataset",
        )
    return wanted


def check_identifiers(index, rows, sample_ids) -> dict:
    import numpy as np

    n = int(index.shape[0])
    chunk = int(index.chunks[0]) if index.chunks else 65536
    starts = sorted(
        {int(fraction * (n - 1)) // chunk * chunk for fraction in np.linspace(0.0, 1.0, ID_WINDOWS)}
    )
    checked = mismatched = 0
    for start in starts:
        stop = min(start + chunk, n)
        low, high = (int(v) for v in np.searchsorted(rows, [start, stop]))
        if high == low:
            continue
        window = np.asarray(index.asstr()[start:stop], dtype=object).astype(str)
        mismatched += int(np.count_nonzero(window[rows[low:high] - start] != sample_ids[low:high]))
        checked += high - low
    if checked == 0 or mismatched:
        raise ValueError(
            f"cell identifiers: {mismatched} of {checked} sampled rows differ from the atlas"
        )
    return {
        "windows": len(starts),
        "chunkRows": chunk,
        "checked": checked,
        "mismatched": mismatched,
    }


def atlas_labels(r2, atlas_uri: str, n_rows: int, rows, sample_ids, requested_fine: str) -> dict:
    import h5py
    import numpy as np

    started = time.perf_counter()
    if atlas_uri.startswith("https://"):
        raw = RangeFile(None, "", "", url=atlas_uri)
    else:
        bucket, key = r2.split(atlas_uri)
        raw = RangeFile(r2.client, bucket, key)
    with h5py.File(io.BufferedReader(raw, buffer_size=1 << 20), "r") as h5:
        obs = h5["obs"]
        index_key = _text(obs.attrs.get("_index", "_index"))
        order = obs.attrs.get("column-order")
        columns = (
            [_text(c) for c in np.asarray(order, dtype=object).ravel()]
            if order is not None
            else sorted((k for k in obs if k != index_key))
        )
        categorical = {name for name in columns if _categorical(obs, name)}
        labels = {}
        for name, (role, how) in label_columns(columns, categorical, requested_fine).items():
            if name not in categorical:
                raise ValueError(f"{name} is not a categorical column of the atlas's cells")
            node = obs[name]
            categories = np.asarray(node["categories"].asstr()[:], dtype=object).astype(str)
            codes = np.asarray(node["codes"][:])
            if codes.shape != (n_rows,):
                raise ValueError(f"{name} has {codes.shape[0]} values for {n_rows} atlas rows")
            labels[name] = {
                "role": role,
                "choice": how,
                "categories": categories,
                "codes": codes[rows].astype(np.int64),
            }
        index = obs[index_key]
        if int(index.shape[0]) != n_rows:
            raise ValueError(f"the atlas has {index.shape[0]} rows, the sampling assumes {n_rows}")
        identifiers = check_identifiers(index, rows, sample_ids)
    return {
        "labels": labels,
        "obsColumns": columns,
        "categoricalColumns": sorted(categorical),
        "identifierCheck": identifiers,
        "read": {
            "seconds": time.perf_counter() - started,
            "requests": raw.requests,
            "bytesRead": raw.bytesRead,
        },
    }


def nearest(points, k: int, algorithm: str, workers: int):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=k, algorithm=algorithm, metric="euclidean", n_jobs=workers)
    model.fit(np.ascontiguousarray(points, dtype=np.float64))
    return model.kneighbors(return_distance=False).astype(np.int32)


def jaccard(first, second, k: int, chunk: int = 65536):
    import numpy as np

    out = np.empty(first.shape[0], dtype=np.float64)
    for start in range(0, first.shape[0], chunk):
        a, b = (first[start : start + chunk], second[start : start + chunk])
        shared = (a[:, :, None] == b[:, None, :]).sum(axis=(1, 2))
        out[start : start + chunk] = shared / (2 * k - shared)
    return out


def class_means(values, truth, classes: int):
    import numpy as np

    valid = truth >= 0
    sums = np.bincount(truth[valid], weights=values[valid], minlength=classes)
    counts = np.bincount(truth[valid], minlength=classes)
    return sums / np.maximum(counts, 1)


def summary(values) -> dict:
    import numpy as np

    q25, median, q75 = (float(v) for v in np.quantile(values, [0.25, 0.5, 0.75]))
    return {"mean": float(np.mean(values)), "median": median, "q25": q25, "q75": q75}


def best_clusters(truth, clusters, classes: int, n_clusters: int) -> dict:
    import numpy as np

    valid = truth >= 0
    table = np.bincount(
        truth[valid] * n_clusters + clusters[valid], minlength=classes * n_clusters
    ).reshape(classes, n_clusters)
    class_sizes, cluster_sizes = (table.sum(axis=1), table.sum(axis=0))
    f1 = 2 * table / np.maximum(class_sizes[:, None] + cluster_sizes[None, :], 1)
    best = f1.argmax(axis=1)
    overlap = table[np.arange(classes), best]
    return {
        "cluster": best,
        "clusterCells": cluster_sizes[best],
        "overlap": overlap,
        "recall": overlap / np.maximum(class_sizes, 1),
        "precision": overlap / np.maximum(cluster_sizes[best], 1),
        "f1": f1[np.arange(classes), best],
    }


def analyse(payload: dict, r2, record: dict, save) -> dict:
    from functools import reduce
    from itertools import combinations

    import h5py
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import threadpoolctl
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix

    from experiments.benchmarks.tools.kept_outputs import read_outputs
    from experiments.runtime.embryo_atlas import SOURCE_SPEC, select_nested_rows

    record["versions"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "h5py": h5py.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    steps = record["steps"]

    def timed(name: str, fn):
        started = time.perf_counter()
        value = fn()
        steps[name] = {"seconds": time.perf_counter() - started}
        save()
        return value

    relative = _relative(payload)
    workers = int(payload["workers"])
    work = Path("/tmp/quality")
    work.mkdir(parents=True, exist_ok=True)
    if len({entry["tool"] for entry in payload["tools"]}) != len(payload["tools"]):
        raise ValueError("Every retained run requires a unique comparison name")
    runs = timed(
        "readRecords",
        lambda: {entry["tool"]: finished_run(r2, entry, relative) for entry in payload["tools"]},
    )
    names = list(runs)
    record["runs"] = {tool: run["facts"] for tool, run in runs.items()}
    for name, run in runs.items():
        for key, expected in payload.get("expectedInput", {}).items():
            if run["facts"].get(key) != expected:
                raise ValueError(f"{name} input {key} differs from the fixed comparison input")
    timed(
        "download",
        lambda: [r2.download(run["outputsUri"], work / f"{tool}.h5") for tool, run in runs.items()],
    )
    outputs = timed(
        "readOutputs", lambda: {tool: read_outputs(work / f"{tool}.h5") for tool in names}
    )
    sample_ids = outputs[names[0]]["cellIds"]
    for tool in names[1:]:
        if not np.array_equal(outputs[tool]["cellIds"], sample_ids):
            raise ValueError(f"{tool}'s input cells differ from {names[0]}'s")
    n_input = int(sample_ids.size)
    record["inputCellIdsSha256"] = hashlib.sha256("\0".join(sample_ids).encode()).hexdigest()
    if payload.get("expectedCellIdsSha256") not in (None, record["inputCellIdsSha256"]):
        raise ValueError("Input cell identities differ from the fixed comparison sample")
    record["outputChecks"] = {}
    for tool in names:
        variance = outputs[tool]["pca"].var(axis=0)
        record["outputChecks"][tool] = {
            "attributes": outputs[tool]["attributes"],
            "layout": outputs[tool]["layout"],
            "pcaVarianceShare": [float(v) for v in variance / variance.sum()],
            "pcaVarianceNonIncreasing": bool(np.all(np.diff(variance) <= 1e-06 * variance[0])),
        }
    save()
    common = reduce(np.intersect1d, [outputs[tool]["positions"] for tool in names])
    if common.size < max(outputs[tool]["pca"].shape[1] for tool in names):
        raise ValueError("Too few common cells for the retained PCA dimensions")
    aligned = {}
    for tool in names:
        positions = outputs[tool]["positions"]
        index = np.searchsorted(positions, common)
        if not np.array_equal(positions[index], common):
            raise ValueError(f"{tool}'s retained cells are not in input order")
        cluster_names, cluster_codes = np.unique(
            outputs[tool]["clusters"][index], return_inverse=True
        )
        aligned[tool] = {
            "pca": outputs[tool]["pca"][index],
            "umap": outputs[tool]["umap"][index],
            "clusters": cluster_codes.ravel(),
            "clusterNames": cluster_names,
        }
    record["cells"] = {
        "input": n_input,
        "retained": {tool: int(outputs[tool]["positions"].size) for tool in names},
        "common": int(common.size),
        "identifiersAgree": True,
        "dimensions": {
            tool: {
                "pca": int(outputs[tool]["pca"].shape[1]),
                "umap": int(outputs[tool]["umap"].shape[1]),
            }
            for tool in names
        },
    }
    if payload.get("metrics") == "embedding_and_clustering":
        started = time.perf_counter()
        bases = {}
        with threadpoolctl.threadpool_limits(limits=workers):
            for tool in names:
                scores = aligned[tool]["pca"]
                centered = scores - scores.mean(axis=0, keepdims=True)
                basis, triangular = np.linalg.qr(centered, mode="reduced")
                if np.linalg.matrix_rank(triangular) != scores.shape[1]:
                    raise ValueError(f"{tool} PCA scores are rank deficient on the common cells")
                bases[tool] = basis
            agreement = {}
            for first, second in combinations(names, 2):
                table = contingency_matrix(aligned[first]["clusters"], aligned[second]["clusters"])
                forward, backward = table.argmax(axis=1), table.argmax(axis=0)
                first_labels, second_labels = (
                    aligned[first]["clusters"],
                    aligned[second]["clusters"],
                )
                correlations = np.linalg.svd(bases[first].T @ bases[second], compute_uv=False)
                if np.any(correlations > 1 + 1e-6):
                    raise ArithmeticError("Canonical correlations exceed their numerical bound")
                correlations = np.clip(correlations, 0, 1)
                agreement[f"{first}|{second}"] = {
                    "clusterAri": float(
                        adjusted_rand_score(aligned[first]["clusters"], aligned[second]["clusters"])
                    ),
                    "clusterNmi": float(
                        normalized_mutual_info_score(
                            aligned[first]["clusters"], aligned[second]["clusters"]
                        )
                    ),
                    "consistentCellsFraction": float(
                        (
                            (forward[first_labels] == second_labels)
                            & (backward[second_labels] == first_labels)
                        ).mean()
                    ),
                    "contingency": {
                        "firstClusters": aligned[first]["clusterNames"].tolist(),
                        "secondClusters": aligned[second]["clusterNames"].tolist(),
                        "cells": table.tolist(),
                    },
                    "pcaCanonicalCorrelations": correlations.tolist(),
                    "pcaMinimumCanonicalCorrelation": float(correlations.min()),
                    "pcaMedianCanonicalCorrelation": float(np.median(correlations)),
                    "pcaMaximumSubspaceDegrees": float(np.degrees(np.arccos(correlations.min()))),
                }
        steps["agreement"] = {"seconds": time.perf_counter() - started}
        result = {"agreement": agreement}
        if payload.get("atlasUri"):
            rows = timed(
                "sampleRows",
                lambda: select_nested_rows(
                    SOURCE_SPEC.nRows, (n_input,), seed=int(payload["samplingSeed"])
                )[n_input],
            )
            atlas = timed(
                "atlasLabels",
                lambda: atlas_labels(
                    r2,
                    payload["atlasUri"],
                    SOURCE_SPEC.nRows,
                    rows,
                    sample_ids,
                    payload.get("fineColumn") or "",
                ),
            )
            record["atlas"] = {
                "source": relative(payload["atlasUri"]),
                "rows": int(SOURCE_SPEC.nRows),
                "samplingSeed": int(payload["samplingSeed"]),
                "identifierCheck": atlas["identifierCheck"],
                "read": atlas["read"],
            }
            annotations = {}
            for column, label in atlas["labels"].items():
                sampled = label["codes"]
                truth = sampled[common]
                valid = truth >= 0
                categories = label["categories"]
                sample_sizes = np.bincount(sampled[sampled >= 0], minlength=len(categories))
                common_sizes = np.bincount(truth[valid], minlength=len(categories))
                tools_ = {}
                for tool in names:
                    retained_truth = sampled[outputs[tool]["positions"]]
                    retained_sizes = np.bincount(
                        retained_truth[retained_truth >= 0], minlength=len(categories)
                    )
                    tools_[tool] = {
                        "ari": float(
                            adjusted_rand_score(truth[valid], aligned[tool]["clusters"][valid])
                        ),
                        "nmi": float(
                            normalized_mutual_info_score(
                                truth[valid], aligned[tool]["clusters"][valid]
                            )
                        ),
                        "retainedClassCells": retained_sizes.tolist(),
                    }
                annotations[column] = {
                    "role": label["role"],
                    "classes": categories.tolist(),
                    "sampleClassCells": sample_sizes.tolist(),
                    "commonClassCells": common_sizes.tolist(),
                    "cellsLabelled": int(valid.sum()),
                    "tools": tools_,
                }
            result["authorLabels"] = annotations
        return result
    k, rare_share = int(payload["neighbours"]), float(payload["rareShare"])
    rows = timed(
        "sampleRows",
        lambda: select_nested_rows(
            SOURCE_SPEC.nRows, (n_input,), seed=int(payload["samplingSeed"])
        )[n_input],
    )
    atlas = timed(
        "atlasLabels",
        lambda: atlas_labels(
            r2,
            payload["atlasUri"],
            SOURCE_SPEC.nRows,
            rows,
            sample_ids,
            payload.get("fineColumn") or "",
        ),
    )
    record["atlas"] = {
        "source": relative(payload["atlasUri"]),
        "rows": int(SOURCE_SPEC.nRows),
        "sampling": {
            "function": "experiments.runtime.embryo_atlas.select_nested_rows",
            "seed": int(payload["samplingSeed"]),
            "size": n_input,
        },
        "obsColumns": atlas["obsColumns"],
        "categoricalColumns": atlas["categoricalColumns"],
        "labelColumns": {
            name: {"role": label["role"], "choice": label["choice"]}
            for name, label in atlas["labels"].items()
        },
        "fineColumn": next(
            (name for name, label in atlas["labels"].items() if label["role"] == "fine"), None
        ),
        "identifierCheck": atlas["identifierCheck"],
        "read": atlas["read"],
    }
    save()
    neighbours = {}
    with threadpoolctl.threadpool_limits(limits=workers):
        for tool in names:
            for space, algorithm in (("pca", "brute"), ("umap", "kd_tree")):
                neighbours[tool, space] = timed(
                    f"neighbours_{tool}_{space}",
                    lambda: nearest(aligned[tool][space], k, algorithm, workers),
                )
    metrics, class_rows, pair_rows = ({}, [], [])
    columns = {}
    started = time.perf_counter()
    for column, label in atlas["labels"].items():
        codes = label["codes"][common]
        present, inverse = np.unique(codes[codes >= 0], return_inverse=True)
        truth = np.full(codes.shape, -1, dtype=np.int64)
        truth[codes >= 0] = inverse.ravel()
        classes = int(present.size)
        class_names = label["categories"][present]
        valid = truth >= 0
        labelled = int(valid.sum())
        sizes = np.bincount(truth[valid], minlength=classes)
        shares = sizes / labelled
        rare = shares < rare_share
        columns[column] = {
            "role": label["role"],
            "truth": truth,
            "names": class_names,
            "sizes": sizes,
            "shares": shares,
            "rare": rare,
        }
        sampled = label["codes"]
        column_metrics: dict = {
            "role": label["role"],
            "choice": label["choice"],
            "classesInAtlas": int(label["categories"].size),
            "classesInSample": int(np.unique(sampled[sampled >= 0]).size),
            "classes": classes,
            "cellsLabelled": labelled,
            "cellsUnlabelled": int(common.size - labelled),
            "rareClasses": [str(name) for name in class_names[rare]],
            "chanceNeighbourShare": float(np.sum(shares**2)),
            "tools": {},
        }
        for tool in names:
            clusters = aligned[tool]["clusters"]
            own_truth = sampled[outputs[tool]["positions"]]
            own_clusters = np.unique(outputs[tool]["clusters"], return_inverse=True)[1].ravel()
            own_valid = own_truth >= 0
            entry: dict = {
                "clusters": int(aligned[tool]["clusterNames"].size),
                "ari": float(adjusted_rand_score(truth[valid], clusters[valid])),
                "nmi": float(normalized_mutual_info_score(truth[valid], clusters[valid])),
                "ariOwnRetained": float(
                    adjusted_rand_score(own_truth[own_valid], own_clusters[own_valid])
                ),
                "nmiOwnRetained": float(
                    normalized_mutual_info_score(own_truth[own_valid], own_clusters[own_valid])
                ),
                "cellsOwnRetained": int(own_valid.sum()),
                "neighbourShare": {},
            }
            per_class = {}
            for space in ("pca", "umap"):
                share = (truth[neighbours[tool, space]] == truth[:, None]).mean(axis=1)
                per_class[space] = class_means(share, truth, classes)
                entry["neighbourShare"][space] = {
                    "cells": float(share[valid].mean()),
                    "classes": float(per_class[space].mean()),
                }
            best = best_clusters(truth, clusters, classes, int(aligned[tool]["clusterNames"].size))
            entry["rare"] = {
                "classes": int(rare.sum()),
                "meanRecall": float(best["recall"][rare].mean()) if rare.any() else None,
                "medianRecall": float(np.median(best["recall"][rare])) if rare.any() else None,
                "meanF1": float(best["f1"][rare].mean()) if rare.any() else None,
                "medianF1": float(np.median(best["f1"][rare])) if rare.any() else None,
                "minF1": float(best["f1"][rare].min()) if rare.any() else None,
            }
            column_metrics["tools"][tool] = entry
            for c in range(classes):
                class_rows.append(
                    {
                        "labelColumn": column,
                        "role": label["role"],
                        "class": str(class_names[c]),
                        "cells": int(sizes[c]),
                        "share": float(shares[c]),
                        "rare": bool(rare[c]),
                        "tool": tool,
                        "pcaNeighbourShare": float(per_class["pca"][c]),
                        "umapNeighbourShare": float(per_class["umap"][c]),
                        "bestCluster": str(aligned[tool]["clusterNames"][best["cluster"][c]]),
                        "bestClusterCells": int(best["clusterCells"][c]),
                        "overlapCells": int(best["overlap"][c]),
                        "recall": float(best["recall"][c]),
                        "precision": float(best["precision"][c]),
                        "f1": float(best["f1"][c]),
                    }
                )
        metrics[column] = column_metrics
    steps["labelMetrics"] = {"seconds": time.perf_counter() - started}
    save()
    started = time.perf_counter()
    agreement = {}
    for first, second in combinations(names, 2):
        pair = {
            "clusterAri": float(
                adjusted_rand_score(aligned[first]["clusters"], aligned[second]["clusters"])
            ),
            "clusterNmi": float(
                normalized_mutual_info_score(
                    aligned[first]["clusters"], aligned[second]["clusters"]
                )
            ),
        }
        overlaps = {}
        for space in ("pca", "umap"):
            overlaps[space] = jaccard(neighbours[first, space], neighbours[second, space], k)
            pair[f"{space}Jaccard"] = summary(overlaps[space])
        agreement[f"{first}|{second}"] = pair
        for column, info in columns.items():
            classes = int(info["names"].size)
            means = {
                space: class_means(overlaps[space], info["truth"], classes)
                for space in ("pca", "umap")
            }
            for c in range(classes):
                pair_rows.append(
                    {
                        "labelColumn": column,
                        "role": info["role"],
                        "class": str(info["names"][c]),
                        "cells": int(info["sizes"][c]),
                        "share": float(info["shares"][c]),
                        "rare": bool(info["rare"][c]),
                        "toolA": first,
                        "toolB": second,
                        "pcaJaccard": float(means["pca"][c]),
                        "umapJaccard": float(means["umap"][c]),
                    }
                )
    steps["agreement"] = {"seconds": time.perf_counter() - started}
    for rows_, uri, name in (
        (class_rows, payload["classesUri"], "classes.csv"),
        (pair_rows, payload["pairsUri"], "pairs.csv"),
    ):
        pd.DataFrame(rows_).to_csv(work / name, index=False)
        r2.upload(work / name, uri)
    return {
        "metrics": metrics,
        "agreement": agreement,
        "tables": {
            "classes": relative(payload["classesUri"]),
            "pairs": relative(payload["pairsUri"]),
        },
    }


def quality_job(payload: dict) -> dict:
    workers = str(payload["workers"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = workers

    from experiments.runtime.harness import R2

    r2 = R2(payload["r2EndpointUrl"])
    uri = payload["recordUri"]
    if not payload.get("force") and r2.exists(uri) and (r2.get_json(uri).get("status") == "ok"):
        return {"status": "existing"}
    record: dict = {
        "experiment": "quality",
        "runTag": payload["runTag"],
        "dataset": payload["dataset"],
        "status": "running",
        "metrics": payload.get("metrics", "annotations_and_neighbors"),
        "method": METHOD
        if payload.get("metrics") != "embedding_and_clustering"
        else {
            "cells": METHOD["cells"],
            "clusterAgreement": METHOD["clusterAgreement"],
            "pcaSubspace": "Canonical correlations are the singular values of Q1.T@Q2 after float64 centering and reduced QR of each retained PCA-score matrix on all common cells",
            "contingency": "Counts of common cells in each pair of clusters; consistent cells have mutually largest-overlap cluster assignments",
            "authorLabels": "Author-label ARI and NMI use common cells; per-class retention counts use every input cell and each workflow's own filter when atlasUri is provided",
            "interpretation": "Native workflow agreement on common cells; each tool retains its own preprocessing and variable features",
        },
        "container": payload["container"],
        "image": payload["image"],
        "steps": {},
        "startedAt": time.time(),
    }

    if record["metrics"] not in {"annotations_and_neighbors", "embedding_and_clustering"}:
        raise ValueError(f"Unknown quality endpoint: {record['metrics']}")
    if record["metrics"] == "annotations_and_neighbors":
        record.update(neighbours=payload["neighbours"], rareShare=payload["rareShare"])

    def save() -> None:
        r2.put_json(uri, record)

    save()
    try:
        record.update(analyse(payload, r2, record, save))
        record["status"] = "ok"
    except BaseException:
        record["status"] = "error"
        record["error"] = _redact(traceback.format_exc()[-4000:], payload)
    record["finishedAt"] = time.time()
    save()
    return {"status": record["status"], "commonCells": (record.get("cells") or {}).get("common")}


def _replicates(text: str, names) -> dict[str, int]:
    chosen = dict.fromkeys(names, 1)
    for item in (part for part in text.split(",") if part):
        tool, _, value = item.partition("=")
        if tool not in chosen or not value.isdigit():
            raise SystemExit(f"--replicates takes TOOL=N for the tools {', '.join(chosen)}")
        chosen[tool] = int(value)
    return chosen


def plan(
    dataset: str = "",
    run_tag: str = "",
    fine_column: str = "",
    replicates: str = "",
    force: bool = False,
) -> dict:
    from experiments.benchmarks.tool_runs import (
        QUALITY_CPU,
        QUALITY_DATASET,
        QUALITY_TAG,
        QUALITY_TOOLS,
    )

    config = tools._config()
    dataset = dataset or QUALITY_DATASET
    run_tag = run_tag or QUALITY_TAG
    if not dataset.startswith("embryo_"):
        raise SystemExit(
            "the author labels come from the embryo atlas: compare an embryo_<cells> sample"
        )
    tools._dataset_uri(config, dataset)
    chosen = _replicates(replicates, QUALITY_TOOLS)
    results = config["resultsUri"].rstrip("/")
    campaign = f"{results}/results/tools/{run_tag}"
    datasets = config["datasetPrefixUri"].rstrip("/")
    folder = f"{campaign}/quality/{dataset}"
    return {
        "r2EndpointUrl": config["r2EndpointUrl"],
        "resultsUri": results,
        "campaignUri": campaign,
        "datasetsUri": datasets,
        "runTag": run_tag,
        "dataset": dataset,
        "tools": [
            {
                "tool": tool,
                "replicate": chosen[tool],
                "ladderUri": f"{campaign}/{dataset}/{tool}_c{QUALITY_CPU}_r{chosen[tool]}_ladder.json",
            }
            for tool in QUALITY_TOOLS
        ],
        "atlasUri": f"{datasets}/source.h5ad",
        "samplingSeed": int(config.get("samplingSeed", 0)),
        "fineColumn": fine_column,
        "neighbours": NEIGHBOURS,
        "rareShare": RARE_SHARE,
        "workers": WORKERS,
        "container": {
            "cpu": CPU,
            "memoryMb": MEMORY_MB,
            "workers": WORKERS,
            "preemptible": True,
        },
        "image": tools._image_record("scanpy"),
        "recordUri": f"{folder}.json",
        "classesUri": f"{folder}_classes.csv",
        "pairsUri": f"{folder}_pairs.csv",
        "force": force,
    }


def _local_r2(config: dict):
    from experiments.runtime.harness import R2

    return R2(config["r2EndpointUrl"])


def dry_run(payload: dict) -> None:
    relative = _relative(payload)
    names = ", ".join(
        (f"{entry['tool']} (replicate {entry['replicate']})" for entry in payload["tools"])
    )
    lines = [
        f"compare {names} on {payload['dataset']} under {payload['runTag']}",
        f"labels: {MAJOR_COLUMN}, {payload['fineColumn'] or 'a finer author cell-type column if the atlas has one'} and cell_type, from {relative(payload['atlasUri'])} for the rows sampled with seed {payload['samplingSeed']}",
        f"metrics: ARI and NMI, {payload['neighbours']}-NN label share in PCA and UMAP, Jaccard of 15-NN sets between tools, best-matching cluster of classes below {payload['rareShare']:.0%}",
        "container: 8 CPU, 32 GiB, preemptible; image experiments/runtime/python-tools.txt",
    ]
    try:
        r2 = _local_r2(tools._config())
    except (OSError, KeyError, ImportError) as error:
        lines.append(f"R2 not checked ({type(error).__name__}): no read credentials or boto3 here")
        r2 = None
    if r2 is not None:
        for entry in payload["tools"]:
            try:
                facts = finished_run(r2, entry, relative)["facts"]
                kept = facts.get("keptOutputs") or {}
                lines.append(
                    f"{facts['ladder']}: ok at {facts.get('memoryGib')} GiB; outputs {facts['outputs']} ({(kept.get('bytes') or 0) / 2**20:.0f} MiB)"
                )
            except Exception as error:
                lines.append(
                    _redact(f"{relative(entry['ladderUri'])}: not ready ({error})", payload)
                )
        state = (
            r2.get_json(payload["recordUri"]).get("status")
            if r2.exists(payload["recordUri"])
            else "absent"
        )
        lines.append(f"record {relative(payload['recordUri'])}: {state}")
    lines.append("dry run: nothing spawned")
    print("\n".join(lines), flush=True)
