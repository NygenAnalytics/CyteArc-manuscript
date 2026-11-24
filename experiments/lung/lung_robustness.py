import hashlib
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path

from experiments.lung import HERE
from experiments.lung.lung_pipeline import run_pipeline
from experiments.runtime.common import ROOT
from experiments.runtime.image_records import image_record

TOOLS = HERE / "tools"

RUN_TAG = "lung-sensitivity"
PRIMARY_TAG = "lung-reference-screen"
PUBLISHED_FOLDER = "stores/lung-reference-screen"
GRID_FOLDER = "stores/lung-sensitivity"
MIB = 1024**2
CPU = 16.0
MEMORY_MB = 32 * 1024
SMALL_CPU = 4.0
SMALL_MEMORY_MB = 16 * 1024
BUDGET_SHARE = 0.75
TIMEOUT = 4 * 3600
LEVEL = "ann_level_1"
VOTE = 0.6
MISSING_FEATURES = "reference_mean"
LEIDEN_SEED = 4444
CHARACTERISATION = {"hvg": 2000, "dims": 30, "neighbours": 15}
NATRI_LAB_STUDY = "Banovich_Kropski_2020"
CONTROL_DISEASE = "normal"
OUTLIER = "VUHD117"
BASALOID = "KRT5-/KRT17+"
CONTROL, LESS, MORE = ("Control", "Less_fibrotic", "More_fibrotic")
DONOR_KEY = re.compile("(VU(?:HD|ILD)\\d+|T(?:HD|ILD)\\d+)", re.IGNORECASE)
QUERY_KEY = "grid_query"
EXTENDED_KEY = "grid_extended_reference"
KEYS = [
    "donor",
    "donorKey",
    "region",
    "processingSite",
    "site",
    "diagnosis",
    "disease",
    "donorClass",
]
META = [
    "cellId",
    "donor",
    "donorKey",
    "donorClass",
    "site",
    "sample",
    "region",
    "diagnosis",
    "disease",
    "processingSite",
    "study",
    "authorLabel",
]
REFERENCES = {
    "primary": {"hvg": 2000, "dims": 30, "neighbours": 15},
    "neighbours30": {"hvg": 2000, "dims": 30, "neighbours": 30},
    "pcs50": {"hvg": 2000, "dims": 50, "neighbours": 15},
    "hvg3000": {"hvg": 3000, "dims": 30, "neighbours": 15},
}
EXTENDED = {"hvg": 2000, "dims": 30, "neighbours": 15}
_BASE = {"quantile": 0.99, "exclude": [], "resolution": 1.0}
CONFIGS = {
    "primary": {
        **_BASE,
        "choice": "primary analysis",
        "value": "as in the manuscript",
        "table": "primary",
        "cells": "primary",
    },
    "threshold975": {
        **_BASE,
        "choice": "unexplained threshold",
        "value": "97.5th percentile",
        "quantile": 0.975,
        "table": "primary",
        "cells": "threshold975",
    },
    "threshold995": {
        **_BASE,
        "choice": "unexplained threshold",
        "value": "99.5th percentile",
        "quantile": 0.995,
        "table": "primary",
        "cells": "threshold995",
    },
    "withoutVUHD117": {
        **_BASE,
        "choice": "calibration controls",
        "value": "14, without VUHD117",
        "exclude": [OUTLIER],
        "table": "primary",
        "cells": "withoutVUHD117",
    },
    "vote08": {
        **_BASE,
        "choice": "lineage vote",
        "value": "0.8",
        "table": "vote08",
        "cells": "vote08",
    },
    "resolution05": {
        **_BASE,
        "choice": "characterisation resolution",
        "value": "0.5",
        "resolution": 0.5,
        "table": "primary",
        "cells": "primary",
    },
    "resolution20": {
        **_BASE,
        "choice": "characterisation resolution",
        "value": "2.0",
        "resolution": 2.0,
        "table": "primary",
        "cells": "primary",
    },
    "hvg3000": {
        **_BASE,
        "choice": "variable genes",
        "value": "3,000",
        "table": "hvg3000",
        "cells": "hvg3000",
    },
    "pcs50": {
        **_BASE,
        "choice": "principal components",
        "value": "50",
        "table": "pcs50",
        "cells": "pcs50",
    },
    "neighbours30": {
        **_BASE,
        "choice": "neighbours",
        "value": "30",
        "table": "neighbours30",
        "cells": "neighbours30",
    },
    "extendedReference": {
        **_BASE,
        "choice": "reference",
        "value": "core HLCA plus the extension's healthy cells, without the Natri study",
        "table": "extendedReference",
        "cells": "extendedReference",
    },
}
PHASES = {
    "projection": [
        "primary",
        "resolution05",
        "resolution20",
        "threshold975",
        "threshold995",
        "withoutVUHD117",
        "vote08",
    ],
    "references": ["hvg3000", "pcs50", "neighbours30"],
    "extended": ["extendedReference"],
}
MAPPINGS = {
    "projection": {},
    "references": {
        "primaryRemap": "primary",
        "hvg3000": "hvg3000",
        "pcs50": "pcs50",
        "neighbours30": "neighbours30",
    },
    "extended": {"extendedReference": "extendedReference"},
}


def _is_remote(uri: str) -> bool:
    return uri.startswith("s3://")


def _redact(text) -> str:
    return re.sub("s3://[^/\\s\\\"']+", "s3://<bucket>", str(text))


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {_redact(message)}", flush=True)


class Objects:
    """JSON records, files and listings on R2 (credentials from the Modal secret)."""

    def __init__(self, endpoint: str | None) -> None:
        self.endpoint = endpoint
        self._r2 = None

    def _remote(self):
        if self._r2 is None:
            from experiments.runtime.harness import R2

            self._r2 = R2(self.endpoint)
        return self._r2

    def exists(self, uri: str) -> bool:
        return self._remote().exists(uri) if _is_remote(uri) else Path(uri).exists()

    def get_json(self, uri: str) -> dict:
        return (
            self._remote().get_json(uri) if _is_remote(uri) else json.loads(Path(uri).read_text())
        )

    def put_json(self, uri: str, value: dict) -> None:
        if _is_remote(uri):
            self._remote().put_json(uri, value)
            return
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        Path(uri).write_text(json.dumps(value, indent=1, default=str) + "\n")

    def download(self, uri: str, path: Path) -> int:
        if _is_remote(uri):
            return self._remote().download(uri, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(uri, path)
        return path.stat().st_size

    def upload(self, path: Path, uri: str) -> None:
        if _is_remote(uri):
            self._remote().upload(path, uri)
            return
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, uri)

    def entries(self, uri: str) -> dict[str, int]:
        if not _is_remote(uri):
            base = Path(uri)
            return (
                {str(p.relative_to(base)): p.stat().st_size for p in base.rglob("*") if p.is_file()}
                if base.exists()
                else {}
            )
        r2 = self._remote()
        bucket, prefix = r2.split(uri.rstrip("/") + "/")
        out: dict[str, int] = {}
        for page in r2.client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for item in page.get("Contents", []):
                out[item["Key"][len(prefix) :]] = int(item["Size"])
        return out


def _artifact_of(key: str) -> tuple[str, str] | None:
    parts = key.split("/")
    if parts[0] == "artifacts" and len(parts) > 3:
        return (parts[1], parts[2])
    if len(parts) > 4 and parts[1] == "artifacts":
        return (parts[2], parts[3])
    return None


class StoreLedger:
    """What each step added to a store, from listings taken before and after it."""

    def __init__(self, objects: Objects, uri: str) -> None:
        self.objects, self.uri = (objects, uri)
        self.last = objects.entries(uri)

    def size(self) -> dict:
        return {"bytes": int(sum(self.last.values())), "objects": len(self.last)}

    def take(self) -> dict:
        now = self.objects.entries(self.uri)
        before, self.last = (self.last, now)
        added = [k for k in now if k not in before]
        old = {a for a in map(_artifact_of, before) if a}
        created: dict[tuple[str, str], int] = {}
        top: dict[str, int] = {}
        for key in added:
            found = _artifact_of(key)
            if found is not None and found not in old:
                created[found] = created.get(found, 0) + now[key]
            part = f"artifact:{found[0]}" if found is not None else key.split("/")[0]
            top[part] = top.get(part, 0) + now[key]
        return {
            "bytesAdded": int(sum(now.values()) - sum(before.values())),
            "objectsAdded": len(now) - len(before),
            "newObjects": len(added),
            "changedObjects": sum((1 for k in now if k in before and before[k] != now[k])),
            "removedObjects": sum((1 for k in before if k not in now)),
            "newBytesByPart": dict(sorted(top.items())),
            "createdArtifacts": [
                {"kind": kind, "artifactId": aid, "bytes": size}
                for (kind, aid), size in sorted(created.items())
            ],
            "storeBytes": int(sum(now.values())),
            "storeObjects": len(now),
        }


def _storage_options(uri: str) -> dict | None:
    if not _is_remote(uri):
        return None
    return {
        "access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "endpoint": os.environ["R2_ENDPOINT_URL"].rstrip("/"),
    }


def _options(uri: str, payload: dict) -> dict:
    return {
        "default_assay": "RNA",
        "nthreads": int(payload["workers"]),
        "mem_budget": int(payload["budgetBytes"]),
        "storage_options": _storage_options(uri),
    }


def _open(uri: str, payload: dict, *, mode: str = "r+"):
    import cytearc

    return cytearc.DataStore(uri, zarr_mode=mode, **_options(uri, payload))


def _store_exists(objects: Objects, uri: str) -> bool:
    return objects.exists(f"{uri.rstrip('/')}/zarr.json")


def _mount(objects: Objects, payload: dict, source: str, target: str):
    import cytearc

    if _store_exists(objects, target):
        return (cytearc.DataStore(target, zarr_mode="r+", **_options(target, payload)), True)
    return (cytearc.mount_datastore(source, at=target, **_options(target, payload)), False)


def _work(payload: dict) -> Path:
    path = Path(payload.get("workDir") or "/tmp/biocase_grid")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative(value, payload: dict):
    if isinstance(value, dict):
        return {k: _relative(v, payload) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_relative(v, payload) for v in value]
    if isinstance(value, str):
        for key, placeholder in (("resultsRoot", "<results>"), ("datasetsRoot", "<datasets>")):
            root = str(payload.get(key) or "").rstrip("/")
            if root:
                value = value.replace(root, placeholder)
        return _redact(value)
    return value


def _jsonable(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(64 * MIB):
            digest.update(block)
    return digest.hexdigest()


def _donor_key(donor: str) -> str | None:
    match = DONOR_KEY.search(str(donor))
    return match.group(1).upper() if match else None


def _prepare(payload: dict) -> None:
    if payload.get("r2EndpointUrl"):
        os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    import cytearc

    cytearc.configure_output(level="WARNING", progress=False)


def _network_received() -> int | None:
    try:
        lines = Path("/proc/net/dev").read_text().splitlines()[2:]
    except OSError:
        return None
    total = 0
    for line in lines:
        name, _, fields = line.partition(":")
        if name.strip() != "lo" and fields.split():
            total += int(fields.split()[0])
    return total


def _peak_rss() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class Steps:
    """Wall time, network bytes received and peak memory of each step; ``save`` runs before and
    after every step, so a container that dies leaves the step it was in."""

    def __init__(self, save) -> None:
        self.records: dict[str, dict] = {}
        self.current: str | None = None
        self.save = save

    def run(self, name: str, fn):
        received = _network_received()
        started = time.perf_counter()
        self.current = name
        self.records[name] = {"startedAt": time.time()}
        self.save()
        log(f"start {name}")
        value = fn()
        after = _network_received()
        seconds = time.perf_counter() - started
        self.records[name].update(
            {
                "seconds": seconds,
                "networkReceivedBytes": None
                if received is None or after is None
                else after - received,
                "peakRssBytes": _peak_rss(),
            }
        )
        self.current = None
        self.save()
        log(f"done {name}: {seconds:.1f} s")
        return value

    def seconds(self, name: str) -> float | None:
        return self.records.get(name, {}).get("seconds")


def _provenance(payload: dict) -> dict:
    import platform
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in (
        "cytearc",
        "numpy",
        "scipy",
        "pandas",
        "statsmodels",
        "zarr",
        "obstore",
        "hnswlib",
        "igraph",
        "leidenalg",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    files = {
        name: _sha256(ROOT / name)
        for name in ("experiments/lung/lung_analysis.py", "experiments/runtime/common.py")
    }
    return {
        "script": Path(__file__).name,
        "scriptSha256": _sha256(Path(__file__)),
        "imports": files,
        "runTag": payload["runTag"],
        "primaryRunTag": PRIMARY_TAG,
        "python": platform.python_version(),
        "packages": packages,
        "image": image_record("cytearc_tool_job"),
        "container": {
            key: payload.get(key)
            for key in ("containerCpu", "containerMemoryMb", "workers", "budgetBytes")
        },
    }


def _job(payload: dict, step: str, body) -> dict:
    _prepare(payload)
    objects = Objects(payload.get("r2EndpointUrl"))
    uri = payload["recordUri"]
    if (
        not payload.get("force")
        and objects.exists(uri)
        and (objects.get_json(uri).get("status") == "ok")
    ):
        log(f"{step}: an ok record exists; nothing to do")
        return {"status": "existing", "step": step}
    record: dict = {
        "step": step,
        "status": "running",
        "startedAt": time.time(),
        "provenance": _provenance(payload),
    }
    steps = Steps(
        lambda: objects.put_json(
            uri,
            _relative(
                _jsonable({**record, "steps": steps.records, "currentStep": steps.current}), payload
            ),
        )
    )
    try:
        body(record, steps, objects)
        record["status"] = "ok"
    except BaseException:
        record["status"] = "error"
        record["error"] = traceback.format_exc()[-6000:]
        log(record["error"])
    record["finishedAt"] = time.time()
    record["steps"] = steps.records
    record["currentStep"] = steps.current
    objects.put_json(uri, _relative(_jsonable(record), payload))
    log(f"{step}: {record['status']}")
    if record["status"] == "error":
        raise RuntimeError(record["error"])
    return {
        "status": record["status"],
        "step": step,
        "error": _redact(record.get("error", ""))[-800:],
    }


def _pipeline(ds, label: str, **settings):
    try:
        run = ds.pipeline.open(label=label)
        if run.status == "completed":
            return (run, "existing")
    except Exception:
        pass
    return (run_pipeline(ds, label=label, **settings), "ran")


def _reference_settings(settings: dict, cell_key: str = "I") -> dict:
    return {
        "cells": cell_key,
        "filtering": False,
        "hvg": {"top_n": int(settings["hvg"])},
        "pca": {"dims": int(settings["dims"])},
        "neighbors": {"k": int(settings["neighbours"])},
        "umap": False,
        "leiden": False,
        "cell_cycle": False,
        "paris": False,
        "doublets": False,
        "markers": False,
    }


def _report(run) -> dict:
    report = run.report()
    stages, created, reused = ([], [], [])
    for stage in report["stages"]:
        if stage["status"] == "skipped":
            continue
        plans = [
            {
                "operation": p["operation"],
                "kind": p["ref"]["kind"],
                "artifactId": p["ref"]["artifact_id"],
                "disposition": p["disposition"],
            }
            for p in stage["plans"]
        ]
        for plan in plans:
            (created if plan["disposition"] == "created" else reused).append(
                {**plan, "stage": stage["stage"]}
            )
        stages.append(
            {
                "stage": stage["stage"],
                "status": stage["status"],
                "seconds": (stage.get("metrics") or {}).get("wallSeconds"),
                "peakRssBytes": (stage.get("metrics") or {}).get("rssPeakBytes"),
                "outputs": [
                    {
                        "key": o["outputKey"],
                        "kind": o["artifact"]["kind"],
                        "artifactId": o["artifact"]["artifact_id"],
                        "reused": o["reused"],
                    }
                    for o in stage["outputs"]
                ],
                "plans": plans,
            }
        )
    meta = report["run"]
    finished = meta.get("finishedAtNs")
    return {
        "runId": meta["runId"],
        "label": meta.get("label"),
        "status": meta["status"],
        "cytearcVersion": meta.get("cytearcVersion"),
        "config": meta.get("config"),
        "seconds": None if finished is None else (finished - meta["startedAtNs"]) / 1000000000.0,
        "stages": stages,
        "created": created,
        "reused": reused,
        "outputs": {o["key"]: o["artifact"] for o in meta.get("outputs", [])},
    }


def _reference_info(ds, ref) -> dict:
    import numpy as np

    reference = ds.mapping.load_reference(ref)
    return {
        "mappingReference": ref.to_dict(),
        "cellsSelected": int(reference.selected_cell_count),
        "panelGenes": int(np.asarray(reference.feature_ids).size),
        "dims": int(reference.model.loadings.shape[1]),
        "method": reference.method,
        "payloadFingerprint": reference.payload_fingerprint,
        "modelDigest": reference.model_digest,
        "inputs": {
            name: getattr(reference, name).to_dict()
            for name in (
                "reduction",
                "ann_index",
                "neighbors",
                "cell_selection",
                "feature_selection",
            )
        },
        "distanceQuantiles": np.asarray(
            reference.reference_distance_quantiles, dtype=float
        ).tolist(),
        "distanceValues": np.asarray(reference.reference_distance_values, dtype=float).tolist(),
    }


def _build_reference(
    ds,
    name: str,
    label: str,
    settings: dict,
    steps: Steps,
    ledger: StoreLedger,
    cell_key: str = "I",
) -> dict:
    ledger.take()
    run, how = steps.run(
        f"pipeline_{name}", lambda: _pipeline(ds, label, **_reference_settings(settings, cell_key))
    )
    ref = steps.run(f"build_{name}", lambda: ds.mapping.build_reference(run["neighbors"]))
    storage = ledger.take()
    report = _report(run)
    pipeline_seconds = report["seconds"] if how == "existing" else steps.seconds(f"pipeline_{name}")
    seconds = {"pipeline": pipeline_seconds, "build": steps.seconds(f"build_{name}")}
    return {
        "settings": settings,
        "cellKey": cell_key,
        "label": label,
        "runStatus": how,
        "seconds": seconds,
        "secondsTotal": sum((v or 0.0 for v in seconds.values())),
        "storage": storage,
        "report": report,
        **_reference_info(ds, ref),
    }


def _reference_screen():
    from experiments.lung import lung_analysis

    return lung_analysis


def _primary_records(objects: Objects, payload: dict) -> dict:
    prefix = payload["primaryPrefix"]
    return {
        name: objects.get_json(f"{prefix}/{name}.json")
        for name in ("map", "convert_hlca_core", "characterise")
    }


def _fetch(objects: Objects, uri: str, payload: dict, name: str) -> Path:
    path = _work(payload) / "inputs" / name
    if not path.exists():
        objects.download(uri, path)
    return path


def _primary_table(objects: Objects, payload: dict) -> Path:
    return _fetch(
        objects,
        f"{payload['primaryPrefix']}/natri_cells.csv.gz",
        payload,
        "primary_natri_cells.csv.gz",
    )


def _strings(path: Path):
    import pandas as pd

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _upload_frame(objects: Objects, frame, payload: dict, name: str, **options) -> str:
    path = _work(payload) / "tables" / f"{name}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip", **options)
    uri = f"{payload['resultsPrefix']}/{name}.csv.gz"
    objects.upload(path, uri)
    return uri


def _query(reference_screen, path: Path):
    cells = reference_screen.load_cells(path)
    return cells[cells["donorClass"] == reference_screen.QUERY]


def _percentile(reference_screen, reference: dict):
    return reference_screen.percentile_of(
        {
            "distanceValues": reference["distanceValues"],
            "distanceQuantiles": reference["distanceQuantiles"],
        }
    )


def _flags(reference_screen, query, quantile: float, exclude, percentile):
    previous = reference_screen.QUANTILE
    reference_screen.QUANTILE = float(quantile)
    try:
        calibration = query[
            (query["region"] == reference_screen.CONTROL) & ~query["donorKey"].isin(list(exclude))
        ]
        flags, table = reference_screen.calibrated_flags(query, calibration, percentile)
    finally:
        reference_screen.QUANTILE = previous
    return (flags.astype(bool), table)


def _partition_agreement(first, second) -> dict:
    import numpy as np
    import pandas as pd

    table = (
        pd.crosstab(pd.Series(np.asarray(first)), pd.Series(np.asarray(second)))
        .to_numpy()
        .astype(float)
    )

    def pairs(x):
        return float((x * (x - 1) / 2).sum())

    n = table.sum()
    index, rows, cols = (pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0)))
    expected = rows * cols / (n * (n - 1) / 2) if n > 1 else 0.0
    maximum = (rows + cols) / 2
    identical = bool(((table > 0).sum(axis=1) == 1).all() and ((table > 0).sum(axis=0) == 1).all())
    return {
        "identical": identical,
        "adjustedRand": float((index - expected) / (maximum - expected))
        if maximum != expected
        else 1.0,
        "clusters": [int(table.shape[0]), int(table.shape[1])],
    }


def _references(payload: dict, record: dict, steps: Steps, objects: Objects) -> None:
    from cytearc import ArtifactRef

    target, source = (payload["stores"]["core"], payload["published"]["hlca_core"])
    existed = _store_exists(objects, target)
    ds, _ = steps.run("mount", lambda: _mount(objects, payload, source, target))
    ledger = StoreLedger(objects, target)
    record["mount"] = {
        "store": target,
        "source": source,
        "existed": existed,
        "seconds": steps.seconds("mount"),
        "cells": int(ds.cells.N),
        "genes": int(ds.RNA.feats.N),
        **ledger.size(),
    }
    record["references"] = {}
    for name, settings in REFERENCES.items():
        record["references"][name] = _build_reference(
            ds, name, f"grid_reference_{name}", settings, steps, ledger
        )
    primary = _primary_records(objects, payload)
    published = _open(payload["published"]["hlca_core"], payload, mode="r")
    theirs = _reference_info(
        published,
        ArtifactRef.from_dict(primary["convert_hlca_core"]["reference"]["mappingReference"]),
    )
    mine = record["references"]["primary"]
    record["primaryReproduction"] = {
        "sameMappingReferenceId": mine["mappingReference"]["artifact_id"]
        == theirs["mappingReference"]["artifact_id"],
        "samePayloadFingerprint": mine["payloadFingerprint"] == theirs["payloadFingerprint"],
        "sameModelDigest": mine["modelDigest"] == theirs["modelDigest"],
        "sameInputIds": {
            k: mine["inputs"][k]["artifact_id"] == theirs["inputs"][k]["artifact_id"]
            for k in mine["inputs"]
        },
        "sameDistanceValues": mine["distanceValues"] == theirs["distanceValues"],
        "published": {
            k: theirs[k]
            for k in ("mappingReference", "payloadFingerprint", "modelDigest", "inputs")
        },
    }


def _extended_cells(ds, natri_keys: set[str]) -> tuple:
    import numpy as np
    import pandas as pd

    column = {
        name: np.asarray(ds.cells.fetch_all(name)).astype(str)
        for name in ("core_or_extension", "study", "dataset", "disease", "donor_id", LEVEL)
    }
    active = np.asarray(ds.cells.fetch_all("I"), dtype=bool)
    unique, inverse = np.unique(column["donor_id"], return_inverse=True)
    keys = np.array([_donor_key(d) or "" for d in unique], dtype=object)[inverse]
    core = column["core_or_extension"] == "core"
    extension = column["core_or_extension"] == "extension"
    healthy = column["disease"] == CONTROL_DISEASE
    natri_study = (column["study"] == NATRI_LAB_STUDY) | (column["dataset"] == NATRI_LAB_STUDY)
    natri_donor = np.isin(keys, sorted(natri_keys))
    natri = natri_study | natri_donor
    added = active & extension & healthy & ~natri
    mask = active & ~natri & (core | extension & healthy)

    def count(rows) -> dict:
        return {"cells": int(rows.sum()), "donors": int(np.unique(column["donor_id"][rows]).size)}

    by_study = (
        pd.DataFrame({"study": column["study"][added], "donor": column["donor_id"][added]})
        .groupby("study")
        .agg(cells=("donor", "size"), donors=("donor", "nunique"))
        .reset_index()
        .to_dict(orient="records")
    )
    labels = pd.Series(column[LEVEL][mask]).value_counts()
    selection = {
        "rule": f"active core cells and active extension cells with disease 'normal', without any cell whose study or dataset is {NATRI_LAB_STUDY} or whose donor key is one of the query study's donors",
        "natriDonorKeys": len(natri_keys),
        "core": count(active & core),
        "coreRemoved": {
            "natriStudyOrDataset": count(active & core & natri_study),
            "natriDonorKey": count(active & core & natri_donor),
            "either": count(active & core & natri),
        },
        "extensionHealthy": count(active & extension & healthy),
        "extensionHealthyRemoved": {
            "natriStudyOrDataset": count(active & extension & healthy & natri_study),
            "natriDonorKey": count(active & extension & healthy & natri_donor),
            "either": count(active & extension & healthy & natri),
        },
        "coreKept": count(active & core & ~natri),
        "extensionAdded": count(added),
        "extensionAddedByStudy": by_study,
        "selected": count(mask),
        "labels": {str(k): int(v) for k, v in labels.items()},
    }
    return (mask, selection)


def _extended_mount(payload: dict, record: dict, steps: Steps, objects: Objects) -> None:
    reference_screen = _reference_screen()
    target, source = (payload["stores"]["full"], payload["published"]["hlca_full"])
    existed = _store_exists(objects, target)
    ds, _ = steps.run("mount", lambda: _mount(objects, payload, source, target))
    ledger = StoreLedger(objects, target)
    record["mount"] = {
        "store": target,
        "source": source,
        "existed": existed,
        "seconds": steps.seconds("mount"),
        "cells": int(ds.cells.N),
        "genes": int(ds.RNA.feats.N),
        **ledger.size(),
    }
    natri = reference_screen.load_cells(_primary_table(objects, payload))
    keys = {k for k in natri["donorKey"].astype(str) if k}
    mask, selection = steps.run("select", lambda: _extended_cells(ds, keys))
    steps.run("insert", lambda: ds.cells.insert(EXTENDED_KEY, mask, overwrite=True))
    record["selection"] = {"cellKey": EXTENDED_KEY, **selection}
    record["storage"] = ledger.take()


def _extended_reference(payload: dict, record: dict, steps: Steps, objects: Objects) -> None:
    mounted = objects.get_json(payload["mountRecordUri"])
    if mounted.get("status") != "ok":
        raise RuntimeError("the extended mount has not completed")
    target = payload["stores"]["full"]
    ds = steps.run("open", lambda: _open(target, payload))
    ledger = StoreLedger(objects, target)
    record["references"] = {
        "extendedReference": _build_reference(
            ds,
            "extendedReference",
            "grid_reference_extended",
            EXTENDED,
            steps,
            ledger,
            cell_key=EXTENDED_KEY,
        )
    }


def _query_store(payload: dict, steps: Steps, objects: Objects, record: dict, query_ids):
    import numpy as np

    target, source = (payload["stores"]["natri"], payload["published"]["natri_ild"])
    existed = _store_exists(objects, target)
    ds, _ = steps.run("mount", lambda: _mount(objects, payload, source, target))
    ledger = StoreLedger(objects, target)
    record["mount"] = {
        "store": target,
        "source": source,
        "existed": existed,
        "seconds": steps.seconds("mount"),
        "cells": int(ds.cells.N),
        **ledger.size(),
    }
    ids = np.asarray(ds.cells.fetch_all("ids")).astype(str)
    query_ids = np.asarray(query_ids, dtype=object).astype(str)
    mask = np.isin(ids, query_ids) & np.asarray(ds.cells.fetch_all("I"), dtype=bool)
    if int(mask.sum()) != len(set(query_ids)):
        raise RuntimeError(
            f"{int(mask.sum())} query cells found in the store for {len(set(query_ids))} in the primary table"
        )
    if QUERY_KEY not in ds.cells.columns or not np.array_equal(
        np.asarray(ds.cells.fetch_all(QUERY_KEY), dtype=bool), mask
    ):
        ds.cells.insert(QUERY_KEY, mask, overwrite=True)
    selection = steps.run("querySelection", lambda: ds.snapshot_cell_selection(QUERY_KEY))
    record["query"] = {
        "cellKey": QUERY_KEY,
        "cells": int(mask.sum()),
        "selection": selection.to_dict(),
        "storage": ledger.take(),
    }
    return (ds, selection, ledger)


def _characterise(
    ds_q,
    name: str,
    flags,
    query,
    payload: dict,
    objects: Objects,
    steps: Steps,
    ledger: StoreLedger,
    rule_seconds: float,
) -> dict:
    import numpy as np
    import pandas as pd

    config = CONFIGS[name]
    chosen = query.loc[flags, ["cellId", "donor", "sample", "region", "lineage"]]
    cells = np.asarray(chosen["cellId"], dtype=object).astype(str)
    key = f"grid_unexplained_{config['cells']}"
    ids = np.asarray(ds_q.cells.fetch_all("ids")).astype(str)
    mask = np.isin(ids, cells)
    if int(mask.sum()) != len(set(cells)):
        raise RuntimeError(
            f"{name}: {int(mask.sum())} of {len(set(cells))} unexplained cells found in the query store"
        )
    ledger.take()
    steps.run(f"insert_{name}", lambda: ds_q.cells.insert(key, mask, overwrite=True))
    resolution = float(config["resolution"])
    settings = {
        "cells": key,
        "filtering": False,
        "hvg": {"top_n": CHARACTERISATION["hvg"]},
        "pca": {"dims": CHARACTERISATION["dims"]},
        "neighbors": {"k": CHARACTERISATION["neighbours"]},
        "umap": False,
        "cell_cycle": False,
        "paris": False,
        "doublets": False,
        "markers": False,
        "leiden": {"partitions": [resolution], "selected": resolution, "random_seed": LEIDEN_SEED},
    }
    run, how = steps.run(
        f"characterise_{name}", lambda: _pipeline(ds_q, f"grid_characterise_{name}", **settings)
    )
    rows = np.flatnonzero(mask)
    if not np.array_equal(np.asarray(run.cells.fetch("ids")).astype(str), ids[rows]):
        raise RuntimeError(f"{name}: the clustered cells are not the unexplained cells")
    clusters = np.asarray(run.cells.fetch("clusters")).astype(str)
    storage = ledger.take()
    frame = pd.DataFrame({"cellId": ids[rows], "cluster": clusters})
    _upload_frame(objects, frame, payload, f"clusters_{name}")
    _upload_frame(objects, chosen, payload, f"unexplained_{name}")
    report = _report(run)
    pipeline_seconds = (
        report["seconds"] if how == "existing" else steps.seconds(f"characterise_{name}")
    )
    seconds = {
        "rule": rule_seconds,
        "insert": steps.seconds(f"insert_{name}"),
        "pipeline": pipeline_seconds,
    }
    return {
        "cellKey": key,
        "resolution": resolution,
        "cells": int(rows.size),
        "clusters": int(np.unique(clusters).size),
        "label": f"grid_characterise_{name}",
        "runStatus": how,
        "settings": {k: v for k, v in settings.items() if k != "cells"},
        "seconds": seconds,
        "secondsTotal": sum((v or 0.0 for v in seconds.values())),
        "storage": storage,
        "report": report,
        "files": {
            "clusters": f"clusters_{name}.csv.gz",
            "unexplained": f"unexplained_{name}.csv.gz",
        },
    }


def _published_labels(payload: dict, objects: Objects, steps: Steps, primary: dict, ds_q) -> dict:
    import numpy as np
    import pandas as pd
    from cytearc import ArtifactRef

    published = payload["published"]
    ds_core = steps.run(
        "openPublishedReference", lambda: _open(published["hlca_core"], payload, mode="r")
    )
    reference = steps.run(
        "loadPublishedReference",
        lambda: ds_core.mapping.load_reference(
            ArtifactRef.from_dict(primary["convert_hlca_core"]["reference"]["mappingReference"])
        ),
    )
    projection = ArtifactRef.from_dict(primary["map"]["natri"]["mapping"])
    result = ds_q.mapping.load_result(projection, reference=reference, load_arrays=False)
    rows = np.flatnonzero(
        np.asarray(ds_q.artifacts.load(result.cell_selection)["values"][:], dtype=bool)
    )
    ids = np.asarray(ds_q.cells.fetch_all("ids")).astype(str)[rows]
    labels, transfers = ({}, {})
    for threshold in (VOTE, 0.8):
        transfer = steps.run(
            f"publishedLabels_{threshold:g}",
            lambda t=threshold: ds_q.mapping.transfer_labels(
                projection, reference=reference, reference_labels=LEVEL, threshold_fraction=t
            ),
        )
        evidence = ds_q.mapping.load_label_transfer(transfer).evidence
        labels[threshold] = pd.Series(
            evidence["label"].fillna("NA").astype(str).to_numpy(), index=ids
        )
        transfers[f"{threshold:g}"] = transfer.to_dict()
    return {
        "labels": labels,
        "projection": projection.to_dict(),
        "transfers": transfers,
        "mappingReference": primary["convert_hlca_core"]["reference"]["mappingReference"],
    }


def _map(
    ds_q,
    selection,
    name: str,
    spec: dict,
    meta,
    primary_query,
    payload: dict,
    objects: Objects,
    steps: Steps,
    ledger: StoreLedger,
) -> dict:
    import numpy as np
    import pandas as pd
    from cytearc import ArtifactRef

    ds_ref = steps.run(f"openReference_{name}", lambda: _open(spec["store"], payload, mode="r"))
    reference = steps.run(
        f"loadReference_{name}",
        lambda: ds_ref.mapping.load_reference(ArtifactRef.from_dict(spec["mappingReference"])),
    )
    ledger.take()
    projection = steps.run(
        f"map_{name}",
        lambda: ds_q.mapping.run(
            reference,
            selection,
            query_assay="RNA",
            save_k=int(spec["saveK"]),
            missing_feature_policy=MISSING_FEATURES,
        ),
    )
    transfer = steps.run(
        f"labels_{name}",
        lambda: ds_q.mapping.transfer_labels(
            projection, reference=reference, reference_labels=LEVEL, threshold_fraction=VOTE
        ),
    )
    evidence = ds_q.mapping.load_label_transfer(transfer).evidence
    storage = ledger.take()
    result = ds_q.mapping.load_result(projection, reference=reference, load_arrays=True)
    rows = np.flatnonzero(
        np.asarray(ds_q.artifacts.load(result.cell_selection)["values"][:], dtype=bool)
    )
    ids = np.asarray(ds_q.cells.fetch_all("ids")).astype(str)[rows]
    mapped = pd.DataFrame(
        {
            "cellId": ids,
            f"label_{LEVEL}": evidence["label"].fillna("NA").astype(str).to_numpy(),
            f"vote_{LEVEL}": evidence["voteFraction"].to_numpy(dtype=float),
            "distancePercentile": evidence["referenceDistancePercentile"].to_numpy(dtype=float),
            "nearestDistance": np.asarray(result.distances[:, 0], dtype=np.float64),
            "uninformative": np.asarray(result.uninformative, dtype=bool).astype(int),
        }
    )
    table = meta.merge(mapped, on="cellId", how="inner", validate="one_to_one")
    if len(table) != len(meta):
        raise RuntimeError(f"{name}: {len(table)} mapped cells for {len(meta)} query cells")
    _upload_frame(objects, table, payload, f"cells_{name}", float_format="%.6g")
    primary = primary_query.set_index("cellId").loc[table["cellId"]]
    same_distance = np.isclose(
        table["nearestDistance"].to_numpy(),
        primary["nearestDistance"].to_numpy(dtype=float),
        rtol=1e-05,
        atol=1e-06,
    )
    diagnostics = {
        k: v if isinstance(v, int | float | str) else str(v)
        for k, v in dict(result.diagnostics).items()
    }
    seconds = {
        k: steps.seconds(f"{k}_{name}") for k in ("openReference", "loadReference", "map", "labels")
    }
    return {
        "referenceName": spec["reference"],
        "reference": spec["mappingReference"],
        "referenceStore": spec["store"],
        "saveK": int(spec["saveK"]),
        "projection": projection.to_dict(),
        "cellsMapped": int(len(table)),
        "uninformativeCells": int(table["uninformative"].sum()),
        "abstainedFraction": float((table[f"label_{LEVEL}"] == "NA").mean()),
        "diagnostics": diagnostics,
        "seconds": seconds,
        "secondsTotal": sum((v or 0.0 for v in seconds.values())),
        "storage": storage,
        "reusedInputs": [
            {
                "kind": "mapping_reference",
                "artifactId": spec["mappingReference"]["artifact_id"],
                "store": spec["store"],
            },
            {
                "kind": "cell_selection",
                "artifactId": selection.artifact_id,
                "store": payload["stores"]["natri"],
            },
        ],
        "comparisonWithPrimary": {
            "sameLabel": float(
                (
                    table[f"label_{LEVEL}"].to_numpy()
                    == primary[f"label_{LEVEL}"].astype(str).to_numpy()
                ).mean()
            ),
            "sameNearestDistance": float(same_distance.mean()),
            "maxAbsDistanceDifference": float(
                np.max(
                    np.abs(
                        table["nearestDistance"].to_numpy()
                        - primary["nearestDistance"].to_numpy(dtype=float)
                    )
                )
            ),
        },
        "distanceQuantiles": np.asarray(
            reference.reference_distance_quantiles, dtype=float
        ).tolist(),
        "distanceValues": np.asarray(reference.reference_distance_values, dtype=float).tolist(),
        "table": f"cells_{name}.csv.gz",
    }


def _reference_specs(objects: Objects, payload: dict, phase: str) -> dict:
    if not MAPPINGS[phase]:
        return {}
    record, store = (
        ("references.json", "core")
        if phase == "references"
        else ("extended_reference.json", "full")
    )
    built = objects.get_json(f"{payload['resultsPrefix']}/{record}")
    if built.get("status") != "ok":
        raise RuntimeError(f"{record} has not completed")
    specs = {}
    for mapping, reference in MAPPINGS[phase].items():
        settings = REFERENCES.get(reference) or EXTENDED
        specs[mapping] = {
            "reference": reference,
            "store": payload["stores"][store],
            "mappingReference": built["references"][reference]["mappingReference"],
            "saveK": settings["neighbours"],
        }
    return specs


def _query_phase(payload: dict, record: dict, steps: Steps, objects: Objects) -> None:
    import numpy as np

    reference_screen = _reference_screen()
    phase = payload["phase"]
    if phase != "projection":
        previous = objects.get_json(f"{payload['resultsPrefix']}/query_projection.json")
        if previous.get("status") != "ok":
            raise RuntimeError(
                "query-projection has not completed; the query store is written by one job at a time"
            )
    if phase == "extended":
        previous = objects.get_json(f"{payload['resultsPrefix']}/query_references.json")
        if previous.get("status") != "ok":
            raise RuntimeError(
                "query-references has not completed; the query store is written by one job at a time"
            )
    primary = _primary_records(objects, payload)
    primary_path = _primary_table(objects, payload)
    primary_query = _query(reference_screen, primary_path)
    strings = _strings(primary_path)
    meta = strings.loc[strings["donorClass"] == reference_screen.QUERY, META].reset_index(drop=True)
    ds_q, selection, ledger = _query_store(
        payload, steps, objects, record, meta["cellId"].to_numpy()
    )
    tables = {
        "primary": (primary_query, _percentile(reference_screen, primary["map"]["reference"]))
    }
    record["mappings"], record["characterisations"] = ({}, {})
    if phase == "projection":
        published = _published_labels(payload, objects, steps, primary, ds_q)
        labels06, labels08 = (published["labels"][VOTE], published["labels"][0.8])
        query_strings = strings.loc[strings["donorClass"] == reference_screen.QUERY].copy()
        replaced = labels08.reindex(query_strings["cellId"].to_numpy())
        if replaced.isna().any():
            raise RuntimeError("some query cells have no label in the published projection")
        query_strings[f"label_{LEVEL}"] = replaced.to_numpy()
        _upload_frame(objects, query_strings, payload, "cells_vote08")
        path = _work(payload) / "tables" / "cells_vote08.csv.gz"
        tables["vote08"] = (_query(reference_screen, path), tables["primary"][1])
        same06 = (
            labels06.reindex(strings["cellId"].to_numpy()).to_numpy()
            == strings[f"label_{LEVEL}"].to_numpy()
        )
        record["vote"] = {
            "projection": published["projection"],
            "mappingReference": published["mappingReference"],
            "store": "our query store, mounted on the published one (the transfer at 0.6 reused from it)",
            "transfers": published["transfers"],
            "seconds": {
                k: steps.seconds(k)
                for k in ("openPublishedReference", "loadPublishedReference", "publishedLabels_0.8")
            },
            "checkSeconds": steps.seconds(f"publishedLabels_{VOTE:g}"),
            "labelsAt06MatchPrimaryTable": float(np.mean(same06)),
            "queryAbstainedAt06": float((query_strings["cellId"].map(labels06) == "NA").mean()),
            "queryAbstainedAt08": float((replaced == "NA").mean()),
            "table": "cells_vote08.csv.gz",
        }
    for name, spec in _reference_specs(objects, payload, phase).items():
        record["mappings"][name] = _map(
            ds_q, selection, name, spec, meta, primary_query, payload, objects, steps, ledger
        )
        path = _work(payload) / "tables" / f"cells_{name}.csv.gz"
        tables[name] = (
            _query(reference_screen, path),
            _percentile(reference_screen, record["mappings"][name]),
        )
    for name in PHASES[phase]:
        config = CONFIGS[name]
        query, percentile = tables[config["table"]]
        started = time.perf_counter()
        flags, _ = _flags(
            reference_screen, query, config["quantile"], config["exclude"], percentile
        )
        rule_seconds = time.perf_counter() - started
        record["characterisations"][name] = _characterise(
            ds_q, name, flags, query, payload, objects, steps, ledger, rule_seconds
        )
        if name == "primary":
            published = _strings(
                _fetch(
                    objects,
                    f"{payload['primaryPrefix']}/unexplained_clusters.csv.gz",
                    payload,
                    "primary_clusters.csv.gz",
                )
            )
            mine = _strings(_work(payload) / "tables" / "clusters_primary.csv.gz")
            merged = published.merge(
                mine, on="cellId", how="outer", suffixes=("Published", "Grid"), indicator=True
            )
            both = merged[merged["_merge"] == "both"]
            record["primaryCharacterisationReproduction"] = {
                "publishedCells": int(len(published)),
                "gridCells": int(len(mine)),
                "sameCells": bool((merged["_merge"] == "both").all()),
                **_partition_agreement(both["clusterPublished"], both["clusterGrid"]),
            }
    record["storeSize"] = ledger.size()


def _o5(tests: dict) -> dict:
    epithelial = tests.get("Epithelial") or {}
    out = {}
    for region in (LESS, MORE):
        test = epithelial.get(f"{region}VersusControl") or {}
        n = test.get("n") or [0, 0]
        statistic = test.get("statistic")
        auc = statistic / (n[0] * n[1]) if statistic is not None and n[0] and n[1] else None
        medians = test.get("medians") or [None, None]
        out[region] = {
            "q": test.get("qValue"),
            "p": test.get("pValue"),
            "auc": auc,
            "direction": None
            if auc is None
            else "excess"
            if auc > 0.5
            else "deficit"
            if auc < 0.5
            else "none",
            "medianRegion": medians[0],
            "medianControl": medians[1],
            "lungs": n[0],
            "controls": n[1],
        }
    return out


def _outcomes(
    reference_screen, name: str, config: dict, query, percentile, clusters_path: Path, work: Path
) -> dict:
    import pandas as pd

    started = time.perf_counter()
    flags, thresholds = _flags(
        reference_screen, query, config["quantile"], config["exclude"], percentile
    )
    rule_seconds = time.perf_counter() - started
    frame = reference_screen.sample_fractions(query, flags, KEYS, reference_screen.MIN_CELLS)
    controls = frame[
        (frame["region"] == reference_screen.CONTROL) & ~frame["donorKey"].isin(config["exclude"])
    ]
    tests, _ = reference_screen.region_tests(frame, controls)
    alternative = sorted(set(config["exclude"]) | {OUTLIER})
    if alternative != sorted(config["exclude"]):
        alt_flags, _ = _flags(reference_screen, query, config["quantile"], alternative, percentile)
        alt_frame = reference_screen.sample_fractions(
            query, alt_flags, KEYS, reference_screen.MIN_CELLS
        )
        alt_tests, _ = reference_screen.region_tests(
            alt_frame,
            alt_frame[
                (alt_frame["region"] == reference_screen.CONTROL)
                & ~alt_frame["donorKey"].isin(alternative)
            ],
        )
    else:
        alt_tests = tests
    raw = work / f"raw_{name}"
    raw.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(clusters_path, raw / "lung_reference_screen_unexplained_clusters.csv.gz")
    assignments = pd.read_csv(clusters_path, dtype={"cluster": str}, keep_default_na=False)
    flagged = set(query.loc[flags, "cellId"].astype(str))
    clustered = set(assignments["cellId"].astype(str))
    summary, basaloid, _rows = reference_screen.cluster_regions(
        query, flags, raw, reference_screen.MIN_CELLS
    )
    base = query[query["informative"] & (query["lineage"] != "NA")]
    author = base.set_index("cellId")["authorLabel"]
    qualifying = []
    for cluster, entry in summary["clusters"].items():
        if entry["authorLabel"] != BASALOID:
            continue
        members = assignments.loc[assignments["cluster"] == cluster, "cellId"]
        control = entry["byRegion"].get(CONTROL, {})
        qualifying.append(
            {
                "cluster": cluster,
                "lineage": entry["lineage"],
                "cells": entry["cells"],
                "donors": entry["donors"],
                "authorLabelShare": float((author.reindex(members) == BASALOID).mean()),
                "controlCells": int(control.get("cells", 0)),
                "controlDonors": int(control.get("donors", 0)),
                "controlShare": control.get("cells", 0) / entry["cells"]
                if entry["cells"]
                else None,
                "cellsFromVUHD117": entry["cellsFromOutlier"],
                "controlCellsWithoutVUHD117": entry["controlsWithoutOutlier"]["cells"],
                "diseasedLungsWithCells": entry["diseasedLungs"]["withCells"],
            }
        )
    together = basaloid["unexplainedClusters"]
    together_control = together["byRegion"].get(CONTROL, {})
    o1 = {
        "exists": bool(qualifying),
        "clusters": qualifying,
        "together": {
            "clusters": together["clusters"],
            "cells": together["cells"],
            "donors": together["donors"],
            "controlCells": int(together_control.get("cells", 0)),
            "controlShare": together_control.get("cells", 0) / together["cells"]
            if together["cells"]
            else None,
            "cellsFromVUHD117": together["cellsFromOutlier"],
            "controlCellsWithoutVUHD117": together["controlsWithoutOutlier"]["cells"],
        },
    }
    paired = together["paired"]
    o2 = {
        "inLess": paired["inLess"],
        "inMore": paired["inMore"],
        "inBoth": paired["inBoth"],
        "lungsWithCells": paired["lungsWithCells"],
        "pairedLungs": paired["lungs"],
        "share": paired.get("shareOfLungsWithCellsInLess"),
        "majority": bool(
            paired["lungsWithCells"] and 2 * paired["inLess"] > paired["lungsWithCells"]
        ),
    }
    o3 = {
        "ratio": paired.get("meanRateRatio"),
        "low": (paired.get("meanRateRatioCi95") or {}).get("low"),
        "high": (paired.get("meanRateRatioCi95") or {}).get("high"),
        "pooledRatio": paired.get("pooledRatio"),
        "pooledLow": (paired.get("pooledRatioCi95") or {}).get("low"),
        "pooledHigh": (paired.get("pooledRatioCi95") or {}).get("high"),
        "higherInLess": paired.get("higherInLess"),
        "higherInMore": paired.get("higherInMore"),
        "signTestP": (paired.get("signTest") or {}).get("pValue"),
    }
    authors = basaloid["authorLabelled"]
    in_base = flags.reindex(base.index, fill_value=False).astype(bool)
    o4 = {
        "labelledCells": authors["cells"],
        "labelledUnexplained": authors["unexplained"],
        "labelledShare": authors["unexplainedShare"],
        "allCells": int(len(base)),
        "allUnexplained": int(in_base.sum()),
        "allShare": float(in_base.mean()),
    }
    o5 = {"own": _o5(tests), "withoutVUHD117": _o5(alt_tests)}
    immune = (tests.get("Immune") or {}).get("spearman") or {}
    o6 = {"rho": immune.get("rho"), "p": immune.get("pValue"), "lungs": immune.get("donors")}
    paired_q = {lineage: result["wilcoxon"].get("qValue") for lineage, result in tests.items()}
    finite = {k: v for k, v in paired_q.items() if v is not None}
    o7 = {
        "minQ": min(finite.values()) if finite else None,
        "lineageOfMin": min(finite, key=finite.get) if finite else None,
        "byLineage": paired_q,
    }
    region_table = {
        lineage: {
            "pairedLungs": result["donors"],
            "wilcoxonQ": result["wilcoxon"].get("qValue"),
            "mixedModelQ": result["mixedModel"].get("qValue"),
            "spearmanRho": result["spearman"].get("rho"),
            "lessVersusControlQ": result[f"{LESS}VersusControl"].get("qValue"),
            "moreVersusControlQ": result[f"{MORE}VersusControl"].get("qValue"),
        }
        for lineage, result in tests.items()
    }
    return {
        "choice": config["choice"],
        "value": config["value"],
        "settings": {k: config[k] for k in ("quantile", "exclude", "resolution", "table")},
        "ruleSeconds": rule_seconds,
        "thresholds": thresholds.to_dict(orient="records"),
        "unexplainedCells": int(flags.sum()),
        "unexplainedByRegion": {
            str(k): int(v)
            for k, v in flags[query["informative"]]
            .groupby(query.loc[query["informative"], "region"])
            .sum()
            .items()
        },
        "lineageAbstained": float((query["lineage"] == "NA").mean()),
        "clusteredCellsMatchUnexplained": flagged == clustered,
        "clusters": len(summary["clusters"]),
        "clusterSummary": {
            k: {
                "lineage": v["lineage"],
                "authorLabel": v["authorLabel"],
                "cells": v["cells"],
                "donors": v["donors"],
            }
            for k, v in summary["clusters"].items()
        },
        "outcomes": {"O1": o1, "O2": o2, "O3": o3, "O4": o4, "O5": o5, "O6": o6, "O7": o7},
        "regionTests": region_table,
        "basaloid": basaloid,
        "assignments": assignments.assign(config=name)[["config", "cellId", "cluster"]],
    }


def screen_findings(summary: dict) -> dict:
    """The primary findings of a reference screen run, read from the summary that
    experiments.lung.lung_analysis writes from its records (lung_reference_screen_summary.json),
    under the names that _primary_check compares: the values the primary configuration must
    reproduce."""
    basaloid = summary.get("basaloid") or {}
    clusters = basaloid.get("unexplainedClusters")
    if not clusters:
        raise ValueError(
            "The screen summary has no clusters of unexplained cells: run lung_analysis on the "
            "screen records collected after the screen's characterise job"
        )
    tests = {
        name: summary["calibrations"][name]["regionTests"]
        for name in ("allControls", "withoutVUHD117")
    }

    def epithelial_q(calibration: str, region: str):
        test = (tests[calibration].get("Epithelial") or {}).get(f"{region}VersusControl") or {}
        return test.get("qValue")

    paired = clusters["paired"]
    # Every region's census counts all cells with a lineage and the unexplained ones among them.
    regions = list(clusters["byRegion"].values())
    cells = sum(region["allCells"] for region in regions)
    unexplained = sum(region["unexplainedCells"] for region in regions)
    paired_q = [
        q
        for result in tests["allControls"].values()
        if (q := result["wilcoxon"].get("qValue")) is not None
    ]
    return {
        "unexplainedCells": summary["calibrations"]["allControls"]["unexplainedCells"],
        "basaloidClusters": list(clusters["clusters"]),
        "inLess": paired["inLess"],
        "lungsWithCells": paired["lungsWithCells"],
        "meanRateRatio": paired.get("meanRateRatio"),
        "meanRateRatioLow": (paired.get("meanRateRatioCi95") or {}).get("low"),
        "meanRateRatioHigh": (paired.get("meanRateRatioCi95") or {}).get("high"),
        "labelledShare": basaloid["authorLabelled"]["unexplainedShare"],
        "allShare": unexplained / cells,
        "epithelialLessQ": epithelial_q("allControls", LESS),
        "epithelialMoreQ": epithelial_q("allControls", MORE),
        "epithelialLessQWithoutVUHD117": epithelial_q("withoutVUHD117", LESS),
        "epithelialMoreQWithoutVUHD117": epithelial_q("withoutVUHD117", MORE),
        "immuneRho": ((tests["allControls"].get("Immune") or {}).get("spearman") or {}).get("rho"),
        "minPairedQ": min(paired_q) if paired_q else None,
    }


def _primary_check(result: dict, published: dict | None) -> dict:
    """The primary configuration's findings against those of the screen run it reuses
    (screen_findings of that run's summary, which the payload gives as screenFindings)."""
    o = result["outcomes"]
    found = {
        "unexplainedCells": result["unexplainedCells"],
        "basaloidClusters": [c["cluster"] for c in o["O1"]["clusters"]],
        "inLess": o["O2"]["inLess"],
        "lungsWithCells": o["O2"]["lungsWithCells"],
        "meanRateRatio": o["O3"]["ratio"],
        "meanRateRatioLow": o["O3"]["low"],
        "meanRateRatioHigh": o["O3"]["high"],
        "labelledShare": o["O4"]["labelledShare"],
        "allShare": o["O4"]["allShare"],
        "epithelialLessQ": o["O5"]["own"][LESS]["q"],
        "epithelialMoreQ": o["O5"]["own"][MORE]["q"],
        "epithelialLessQWithoutVUHD117": o["O5"]["withoutVUHD117"][LESS]["q"],
        "epithelialMoreQWithoutVUHD117": o["O5"]["withoutVUHD117"][MORE]["q"],
        "immuneRho": o["O6"]["rho"],
        "minPairedQ": o["O7"]["minQ"],
    }
    if not published or set(published) != set(found):
        raise ValueError(
            "The primary configuration is checked against the findings of the screen run it "
            f"reuses: give screenFindings with the fields {sorted(found)} "
            "(lung_robustness.screen_findings of that run's summary), not "
            f"{sorted(published or {})}"
        )
    exact = {k: found[k] == v for k, v in published.items()}
    close = {
        k: abs(found[k] - v) <= 1e-09 * max(1.0, abs(v))
        if isinstance(v, float) and found[k] is not None
        else found[k] == v
        for k, v in published.items()
    }
    return {
        "found": found,
        "published": published,
        "exact": exact,
        "allExact": all(exact.values()),
        "allWithin1e-9": all(close.values()),
    }


def _decisions(results: dict) -> dict:
    primary = results.get("primary")
    directions = (
        None
        if primary is None
        else tuple((primary["outcomes"]["O5"]["own"][r]["direction"] for r in (LESS, MORE)))
    )
    rules = {
        "O1": (
            "a cluster of unexplained cells whose majority author label is KRT5-/KRT17+ exists",
            lambda o: o["O1"]["exists"],
        ),
        "O2": (
            "such cells are in the less fibrotic region of a majority of the paired lungs that have them",
            lambda o: o["O2"]["majority"],
        ),
        "O5": (
            f"the epithelial difference from controls has the primary's direction in both regions ({directions})",
            lambda o: tuple((o["O5"]["own"][r]["direction"] for r in (LESS, MORE))) == directions,
        ),
        "O7": (
            "no lineage has a paired regional q below 0.05",
            lambda o: o["O7"]["minQ"] is not None and o["O7"]["minQ"] >= 0.05,
        ),
    }
    out = {}
    for key, (text, rule) in rules.items():
        failing = [name for name, result in results.items() if not rule(result["outcomes"])]
        out[key] = {
            "rule": text,
            "stable": not failing,
            "configurations": len(results),
            "changedBy": [
                {
                    "configuration": name,
                    "choice": CONFIGS[name]["choice"],
                    "value": CONFIGS[name]["value"],
                }
                for name in failing
            ],
        }
    return out


def _kinds(entries) -> list[str]:
    return sorted({e["kind"] for e in entries})


def _costs(records: dict, results: dict) -> tuple[list[dict], dict]:
    refs = {
        **((records.get("references") or {}).get("references") or {}),
        **((records.get("extended_reference") or {}).get("references") or {}),
    }
    chars, maps = ({}, {})
    for phase in PHASES:
        record = records.get(f"query_{phase}") or {}
        chars.update(record.get("characterisations") or {})
        maps.update(record.get("mappings") or {})
    vote = (records.get("query_projection") or {}).get("vote") or {}
    mount = records.get("extended_mount") or {}
    primary_ids = {
        e["artifactId"]
        for part in (refs.get("primary"), chars.get("primary"))
        if part
        for e in part["report"]["created"]
    }
    rows, details = ([], {})
    for name, config in CONFIGS.items():
        parts = []
        if name == "extendedReference" and mount:
            parts.append(
                {
                    "step": "mount and selection",
                    "store": "hlca_full_grid",
                    "seconds": sum(
                        (
                            (mount.get("steps") or {}).get(s, {}).get("seconds") or 0.0
                            for s in ("mount", "select", "insert")
                        )
                    ),
                    "bytesAdded": (mount.get("mount") or {}).get("bytes", 0)
                    + (mount.get("storage") or {}).get("bytesAdded", 0),
                    "created": [],
                    "reused": [],
                }
            )
        if name in refs:
            entry = refs[name]
            store = "hlca_full_grid" if name == "extendedReference" else "hlca_core_grid"
            parts.append(
                {
                    "step": "reference pipeline and mapping reference",
                    "store": store,
                    "seconds": entry["secondsTotal"],
                    "bytesAdded": entry["storage"]["bytesAdded"],
                    "created": entry["report"]["created"]
                    + [
                        {"kind": a["kind"], "artifactId": a["artifactId"]}
                        for a in entry["storage"]["createdArtifacts"]
                        if a["kind"] == "mapping_reference"
                    ],
                    "reused": entry["report"]["reused"],
                }
            )
        if name == "vote08" and vote:
            parts.append(
                {
                    "step": "labels at vote 0.8 from the primary projection",
                    "store": "published query (read-only)",
                    "seconds": sum((v or 0.0 for v in vote.get("seconds", {}).values())),
                    "bytesAdded": 0,
                    "created": [],
                    "reused": [
                        {"kind": "projection", "artifactId": vote["projection"]["artifact_id"]},
                        {
                            "kind": "mapping_reference",
                            "artifactId": vote["mappingReference"]["artifact_id"],
                        },
                    ],
                }
            )
        if name in maps:
            entry = maps[name]
            parts.append(
                {
                    "step": "mapping and labels",
                    "store": "natri_ild_grid",
                    "seconds": entry["secondsTotal"],
                    "bytesAdded": entry["storage"]["bytesAdded"],
                    "created": entry["storage"]["createdArtifacts"],
                    "reused": entry["reusedInputs"],
                }
            )
        if config["table"] == "primary":
            parts.append(
                {
                    "step": "primary projection",
                    "store": "published query (read-only)",
                    "seconds": 0.0,
                    "bytesAdded": 0,
                    "created": [],
                    "reused": [
                        {
                            "kind": "projection",
                            "artifactId": (vote.get("projection") or {}).get(
                                "artifact_id", "primary projection"
                            ),
                        },
                        {
                            "kind": "mapping_reference",
                            "artifactId": (vote.get("mappingReference") or {}).get(
                                "artifact_id", "primary reference"
                            ),
                        },
                    ],
                }
            )
        if name in chars:
            entry = chars[name]
            parts.append(
                {
                    "step": "rule and characterisation",
                    "store": "natri_ild_grid",
                    "seconds": entry["secondsTotal"],
                    "bytesAdded": entry["storage"]["bytesAdded"],
                    "created": entry["report"]["created"],
                    "reused": entry["report"]["reused"],
                }
            )
        created = [e for p in parts for e in p["created"]]
        reused = [e for p in parts for e in p["reused"]]
        from_primary = (
            [e for e in reused if e.get("artifactId") in primary_ids] if name != "primary" else []
        )
        rows.append(
            {
                "configuration": name,
                "choice": config["choice"],
                "value": config["value"],
                "seconds": round(sum((p["seconds"] or 0.0 for p in parts)), 1),
                "bytesAdded": int(sum((p["bytesAdded"] or 0 for p in parts))),
                "computedArtifacts": len(created),
                "computedKinds": " ".join(_kinds(created)),
                "reusedArtifacts": len(reused),
                "reusedKinds": " ".join(_kinds(reused)),
                "reusedFromPrimaryKinds": " ".join(_kinds(from_primary)),
                "steps": "; ".join((f"{p['step']} {p['seconds'] or 0.0:.0f} s" for p in parts)),
            }
        )
        details[name] = {"parts": parts, "reusedFromPrimary": from_primary}
    shared = {
        "coreGridMount": (records.get("references") or {}).get("mount"),
        "queryGridMount": (records.get("query_projection") or {}).get("mount"),
        "querySelection": (records.get("query_projection") or {}).get("query"),
    }
    return (rows, {"configurations": details, "shared": shared})


def _outcome_row(name: str, result: dict) -> dict:
    o = result["outcomes"]
    main = max(o["O1"]["clusters"], key=lambda c: (c["donors"], c["cells"]), default=None)
    return {
        "configuration": name,
        "choice": result["choice"],
        "value": result["value"],
        "unexplainedCells": result["unexplainedCells"],
        "clusters": result["clusters"],
        "O1_exists": o["O1"]["exists"],
        "O1_basaloidClusters": len(o["O1"]["clusters"]),
        "O1_largestClusterCells": None if main is None else main["cells"],
        "O1_largestClusterDonors": None if main is None else main["donors"],
        "O1_largestClusterControlShare": None if main is None else main["controlShare"],
        "O1_togetherCells": o["O1"]["together"]["cells"],
        "O1_togetherDonors": o["O1"]["together"]["donors"],
        "O1_togetherControlShare": o["O1"]["together"]["controlShare"],
        "O1_togetherCellsFromVUHD117": o["O1"]["together"]["cellsFromVUHD117"],
        "O2_inLess": o["O2"]["inLess"],
        "O2_lungsWithCells": o["O2"]["lungsWithCells"],
        "O2_majority": o["O2"]["majority"],
        "O3_meanRateRatio": o["O3"]["ratio"],
        "O3_low": o["O3"]["low"],
        "O3_high": o["O3"]["high"],
        "O4_labelledShare": o["O4"]["labelledShare"],
        "O4_allShare": o["O4"]["allShare"],
        "O5_lessQ": o["O5"]["own"][LESS]["q"],
        "O5_moreQ": o["O5"]["own"][MORE]["q"],
        "O5_lessDirection": o["O5"]["own"][LESS]["direction"],
        "O5_moreDirection": o["O5"]["own"][MORE]["direction"],
        "O5_lessQWithoutVUHD117": o["O5"]["withoutVUHD117"][LESS]["q"],
        "O5_moreQWithoutVUHD117": o["O5"]["withoutVUHD117"][MORE]["q"],
        "O6_immuneRho": o["O6"]["rho"],
        "O6_p": o["O6"]["p"],
        "O7_minQ": o["O7"]["minQ"],
        "O7_lineage": o["O7"]["lineageOfMin"],
    }


def _analysis(payload: dict, record: dict, steps: Steps, objects: Objects) -> None:
    import pandas as pd

    reference_screen = _reference_screen()
    prefix, work = (payload["resultsPrefix"], _work(payload))
    names = payload.get("configs") or list(CONFIGS)
    primary = _primary_records(objects, payload)
    primary_path = _primary_table(objects, payload)
    records = {}
    for name in (
        "references",
        "extended_mount",
        "extended_reference",
        "query_projection",
        "query_references",
        "query_extended",
    ):
        uri = f"{prefix}/{name}.json"
        if objects.exists(uri):
            records[name] = objects.get_json(uri)
    mappings = {}
    for phase in PHASES:
        mappings.update((records.get(f"query_{phase}") or {}).get("mappings") or {})
    loaded: dict = {}

    def table(name: str):
        if name not in loaded:
            if name == "primary":
                loaded[name] = (
                    _query(reference_screen, primary_path),
                    _percentile(reference_screen, primary["map"]["reference"]),
                )
            elif name == "vote08":
                loaded[name] = (
                    _query(
                        reference_screen,
                        _fetch(
                            objects, f"{prefix}/cells_vote08.csv.gz", payload, "cells_vote08.csv.gz"
                        ),
                    ),
                    _percentile(reference_screen, primary["map"]["reference"]),
                )
            else:
                loaded[name] = (
                    _query(
                        reference_screen,
                        _fetch(
                            objects,
                            f"{prefix}/cells_{name}.csv.gz",
                            payload,
                            f"cells_{name}.csv.gz",
                        ),
                    ),
                    _percentile(reference_screen, mappings[name]),
                )
        return loaded[name]

    results, skipped, assignments = ({}, {}, [])
    for name in names:
        config = CONFIGS[name]
        clusters_uri = (
            f"{payload['primaryPrefix']}/unexplained_clusters.csv.gz"
            if name == "primary"
            else f"{prefix}/clusters_{name}.csv.gz"
        )
        if not objects.exists(clusters_uri) or (
            config["table"] not in ("primary", "vote08") and config["table"] not in mappings
        ):
            skipped[name] = "inputs missing"
            continue
        clusters_path = _fetch(objects, clusters_uri, payload, f"clusters_{name}.csv.gz")
        query, percentile = table(config["table"])
        result = steps.run(
            f"outcomes_{name}",
            lambda q=query, p=percentile, c=config, n=name, path=clusters_path: _outcomes(
                reference_screen, n, c, q, p, path, work
            ),
        )
        assignments.append(result.pop("assignments"))
        results[name] = result
    record["skipped"] = skipped
    if "primary" in results:
        record["primaryCheck"] = _primary_check(results["primary"], payload.get("screenFindings"))
    record["decisions"] = _decisions(results)
    cost_rows, cost_details = _costs(records, results)
    record["costs"] = cost_details
    outcome_rows = [_outcome_row(name, result) for name, result in results.items()]
    robustness = {
        "programme": "experiments.lung.lung_robustness.CONFIGS",
        "runTag": payload["runTag"],
        "primaryRunTag": PRIMARY_TAG,
        "configurations": results,
        "decisions": record["decisions"],
        "primaryCheck": record.get("primaryCheck"),
        "costs": {"rows": cost_rows, **cost_details},
        "checks": {
            "primaryReferenceReproduction": (records.get("references") or {}).get(
                "primaryReproduction"
            ),
            "primaryCharacterisationReproduction": (records.get("query_projection") or {}).get(
                "primaryCharacterisationReproduction"
            ),
            "primaryRemap": (mappings.get("primaryRemap") or {}).get("comparisonWithPrimary"),
            "vote": {
                k: v
                for k, v in ((records.get("query_projection") or {}).get("vote") or {}).items()
                if k.startswith(("labels", "query"))
            },
            "extendedSelection": (records.get("extended_mount") or {}).get("selection"),
        },
        "skipped": skipped,
    }
    name = payload.get("outputName") or "robustness"
    objects.put_json(f"{prefix}/{name}.json", _relative(_jsonable(robustness), payload))
    _upload_frame(objects, pd.DataFrame(outcome_rows), payload, f"{name}_outcomes")
    if name == "robustness":
        _upload_frame(objects, pd.DataFrame(cost_rows), payload, f"{name}_costs")
        if assignments:
            _upload_frame(
                objects, pd.concat(assignments, ignore_index=True), payload, f"{name}_clusters"
            )
    record["outcomes"] = outcome_rows


def references_job(payload: dict) -> dict:
    return _job(
        payload,
        "references",
        lambda record, steps, objects: _references(payload, record, steps, objects),
    )


def extended_mount_job(payload: dict) -> dict:
    outcome = _job(
        payload,
        "extended_mount",
        lambda record, steps, objects: _extended_mount(payload, record, steps, objects),
    )
    return outcome


def extended_reference_job(payload: dict) -> dict:
    return _job(
        payload,
        "extended_reference",
        lambda record, steps, objects: _extended_reference(payload, record, steps, objects),
    )


def query_job(payload: dict) -> dict:
    return _job(
        payload,
        f"query_{payload['phase']}",
        lambda record, steps, objects: _query_phase(payload, record, steps, objects),
    )


def analysis_job(payload: dict) -> dict:
    return _job(
        payload,
        payload.get("outputName") or "analysis",
        lambda record, steps, objects: _analysis(payload, record, steps, objects),
    )


def build_plan(
    results_root: str,
    datasets_root: str,
    endpoint: str | None,
    *,
    force: bool = False,
    screen_findings: dict | None = None,
) -> dict:
    """The payloads of the sensitivity jobs. The analysis and the check compare the primary
    configuration with the screen run it reuses, so their payloads need that run's
    screen_findings, which exist once the screen has been characterised and analysed."""
    results_root = results_root.rstrip("/")
    prefix = f"{results_root}/results/biocase/{RUN_TAG}"

    def common(cpu: float, memory_mb: int) -> dict:
        return {
            "r2EndpointUrl": endpoint,
            "resultsRoot": results_root,
            "datasetsRoot": datasets_root.rstrip("/"),
            "runTag": RUN_TAG,
            "workers": int(cpu),
            "budgetBytes": int(memory_mb * MIB * BUDGET_SHARE),
            "containerCpu": cpu,
            "containerMemoryMb": memory_mb,
            "force": force,
            "resultsPrefix": prefix,
            "primaryPrefix": f"{results_root}/results/biocase/{PRIMARY_TAG}",
            "published": {
                name: f"{results_root}/{PUBLISHED_FOLDER}/{name}.zarr"
                for name in ("hlca_core", "hlca_full", "natri_ild")
            },
            "stores": {
                "core": f"{results_root}/{GRID_FOLDER}/hlca_core_grid.zarr",
                "full": f"{results_root}/{GRID_FOLDER}/hlca_full_grid.zarr",
                "natri": f"{results_root}/{GRID_FOLDER}/natri_ild_grid.zarr",
            },
        }

    big, small = (common(CPU, MEMORY_MB), common(SMALL_CPU, SMALL_MEMORY_MB))
    findings = {"screenFindings": screen_findings} if screen_findings else {}
    extended_reference = {
        **big,
        "recordUri": f"{prefix}/extended_reference.json",
        "mountRecordUri": f"{prefix}/extended_mount.json",
    }
    return {
        "references": {**big, "recordUri": f"{prefix}/references.json"},
        "extended": {
            **small,
            "recordUri": f"{prefix}/extended_mount.json",
            "next": extended_reference,
        },
        "extended_reference": extended_reference,
        **{
            f"query_{phase}": {**big, "phase": phase, "recordUri": f"{prefix}/query_{phase}.json"}
            for phase in PHASES
        },
        "analysis": {**small, **findings, "recordUri": f"{prefix}/analysis.json"},
        "check": {
            **small,
            **findings,
            "recordUri": f"{prefix}/check.json",
            "configs": ["primary"],
            "outputName": "primary_check",
        },
    }
