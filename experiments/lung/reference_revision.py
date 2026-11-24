import math
import posixpath
import re
import time
import traceback
from pathlib import Path

import experiments.lung.lung_pipeline as mb
from experiments.lung import HERE
from experiments.runtime.common import ROOT

RUN_TAG = "lung-reference-update"
TEST_TAG = f"{RUN_TAG}-test"
MIB = mb.MIB
REVISION_STUDY = "Kaminski_2020"
AUTHOR_COLUMN = "original_ann_nonharmonized"
AUTHOR_BASALOID = "Aberrant_Basaloid"
NEW_LABEL = "Aberrant basaloid"
REVISED = {"ann_level_1": "Epithelial", "ann_level_3": NEW_LABEL, "ann_finest_level": NEW_LABEL}
NO_VOTE = ("Unknown", "")
COLUMN_PREFIX = "revision_"
V2_KEY = "revision_v2_reference"
V2_RUN_LABEL = "revision_reference_v2"
CENTRE_KEY = "revision_centre"
VERSIONS = ("v1", "v2")
CONTROL_REGION = "Control"
NATRI_BASALOID = "KRT5-/KRT17+"
CENTRES = {
    "natri": {
        "name": "Natri",
        "source": "natri_ild",
        "table": "natri_cells",
        "column": "donorClass",
        "value": "neither",
    },
    "lafyatis_2019": {
        "name": "Lafyatis_2019",
        "source": "hlca_full",
        "table": "replication_cells",
        "column": "study",
        "value": "Lafyatis_2019",
    },
    "misharin_budinger_2018": {
        "name": "Misharin_Budinger_2018",
        "source": "hlca_full",
        "table": "replication_cells",
        "column": "study",
        "value": "Misharin_Budinger_2018",
    },
    "schiller_2020": {
        "name": "Schiller_2020",
        "source": "hlca_full",
        "table": "replication_cells",
        "column": "study",
        "value": "Schiller_2020",
    },
    "sheppard_2020": {
        "name": "Sheppard_2020",
        "source": "hlca_full",
        "table": "replication_cells",
        "column": "study",
        "value": "Sheppard_2020",
    },
}
FULL_CENTRES = tuple((slug for slug, spec in CENTRES.items() if spec["source"] == "hlca_full"))
META = {
    "natri_ild": (
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
    ),
    "hlca_full": (
        "cellId",
        "donor",
        "donorKey",
        "sample",
        "study",
        "dataset",
        "disease",
        "lungCondition",
        "assay",
    ),
}
LABEL_COLUMNS = tuple((f"label_{level}" for level in mb.LEVELS))
VOTE_COLUMNS = tuple((f"vote_{level}" for level in mb.LEVELS))
FLOAT_COLUMNS = ("nearestDistance", "distancePercentile", *VOTE_COLUMNS)
RESULT_COLUMNS = (
    "nearestDistance",
    "distancePercentile",
    "uninformative",
    *LABEL_COLUMNS,
    *VOTE_COLUMNS,
)
METERED = ("core", "full", "natri", "revised")
PROTECTED = frozenset({"core", "full", "natri"})
TEST_SEED = 30
TEST_CORE = 2000
TEST_REVISION = 1500
TEST_CENTRE = 1000
TEST_PER_DONOR = 60
SIZES = {
    "worker": (int(mb.CPU), mb.MEMORY_MB),
    "test_worker": (8, 16 * 1024),
    "setup_worker": (8, 32 * 1024),
    "stats_worker": (4, 16 * 1024),
}
TIMEOUTS = {
    "worker": 3 * 3600,
    "test_worker": 2 * 3600,
    "setup_worker": 2 * 3600,
    "stats_worker": 3600,
}


def _redact(text) -> str:
    text = re.sub("s3://[^/\\s\\\"']+", "s3://<bucket>", str(text))
    text = re.sub("https://[^/\\s\\\"']+\\.r2\\.cloudflarestorage\\.com", "<endpoint>", text)
    return re.sub("\\b(?:fc|fu|ap|ta|in|sb)-[0-9A-Za-z]{20,}\\b", "<modal-id>", text)


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {_redact(message)}", flush=True)


def _plain(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


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


def _work(payload: dict) -> Path:
    path = Path(payload.get("workDir") or "/tmp/biocase_revision")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch(objects, uri: str, path: Path) -> Path:
    objects.download(uri, path)
    return path


def _reference_screen():
    from experiments.lung import lung_analysis

    return lung_analysis


def _label_column(version: str, level: str) -> str:
    return level if version == "v1" else f"{COLUMN_PREFIX}{level}"


def _design() -> dict:
    return {
        "source": "experiments.lung.reference_revision._design; original specification fixed on 1 October 2026 before any result",
        "versions": {
            "v1": "the published core HLCA reference of lung_pipeline.py (convert_hlca_core.json); labels ann_level_1, ann_level_3 and ann_finest_level of the core store",
            "v2": f"the core HLCA plus the {REVISION_STUDY} extension cells with an ILD diagnosis or disease '{mb.CONTROL_DISEASE}' (its COPD donors left out, as in part A), in a store mounted on the published full HLCA, with v1's parameters; labels in the columns {COLUMN_PREFIX}<level>",
        },
        "labels": {
            "coreCells": "their v1 labels, read from the core store by cell identifier",
            "revisionCells": f"the HLCA's annotation of the extension, with {list(NO_VOTE)} made blank so that they never vote",
            "revisedCells": f"{AUTHOR_COLUMN} == '{AUTHOR_BASALOID}' takes {REVISED}",
            "fallback": "a pure annotation revision is not needed: the full HLCA holds the authors' labels; the job stops if they are missing",
        },
        "centres": {
            slug: {
                "name": spec["name"],
                "source": spec["source"],
                "cells": f"{spec['column']} == '{spec['value']}' in {spec['table']}.csv.gz",
            }
            for slug, spec in CENTRES.items()
        },
        "recipe": f"snapshot of the centre's cells; per version run_mapping (save_k {mb.NEIGHBOURS}, missing features {mb.MISSING_FEATURES}) and run_label_transfer at {list(mb.LEVELS)} with threshold_fraction {mb.VOTE_THRESHOLD}; phase 1 with v1, phase 2 with v1 and v2 in the same store",
        "unexplained": "experiments.lung.lung_analysis: within each lineage (the version's ann_level_1 label), distance to the nearest reference cell above the 99th percentile (inverted_cdf) of the calibration controls' distances to the same version; controls scored leave one out",
        "calibration": f"Natri: its 15 control lungs; the other centres: their own control donors when at least 3, else the controls of the other centres of this experiment with the same assay ({REVISION_STUDY}'s controls are reference cells of v2 and calibrate nobody); the same under both versions",
        "reads": "Zarr stores opened read-only for the published stores and the revised reference, wrapped with experiments/runtime/recording_store.py (cytearc.storage.stores.make_store patched for those locations only, as mounting.py)",
        "traceability": "v1 labels, votes and distances rebuilt from the frozen v1 label transfers alone, in a new read-only handle with no reference opened, compared exactly with the v1 table of phase 1; artifacts of the v1 and v2 lineages classed reused, new or superseded",
        "completeRerun": "the Natri recipe with both versions in a fresh store mounted on the published Natri store, alone on the bucket after the Natri update",
        "parameters": {
            "hvgCount": mb.HVG,
            "pcaDims": mb.DIMS,
            "neighbours": mb.NEIGHBOURS,
            "levels": list(mb.LEVELS),
            "voteThreshold": mb.VOTE_THRESHOLD,
            "missingFeaturePolicy": mb.MISSING_FEATURES,
            "newLabel": NEW_LABEL,
            "revisedLabels": REVISED,
            "noVote": list(NO_VOTE),
            "testSample": {
                "seed": TEST_SEED,
                "core": TEST_CORE,
                "revision": TEST_REVISION,
                "perCentre": TEST_CENTRE,
                "perNatriDonor": TEST_PER_DONOR,
            },
        },
    }


def _provenance(payload: dict) -> dict:
    record = mb._provenance(payload)
    record["script"] = Path(__file__).name
    record["scriptSha256"] = mb._sha256(Path(__file__))
    record["lungReferenceScriptSha256"] = mb._sha256(HERE / "lung_pipeline.py")
    for key, path in (
        ("referenceScreenSourceSha256", HERE / "lung_analysis.py"),
        ("recordingStoreSha256", ROOT / "experiments" / "runtime" / "recording_store.py"),
    ):
        record[key] = mb._sha256(path)
    record["referenceScreenRunTag"] = mb.RUN_TAG
    record["design"] = _design()
    return record


def _identity(location: str) -> str:
    from urllib.parse import urlsplit

    if "://" in location and (not location.startswith("file://")):
        parts = urlsplit(location)
        return f"{parts.scheme.lower()}://{parts.netloc}{posixpath.normpath(parts.path or '/')}"
    return str(Path(location.removeprefix("file://")).expanduser().resolve())


class ReadMeter:
    """Bytes and requests read from named store locations while a job runs.

    CyteArc turns every location string into a Zarr store in cytearc.storage.stores.make_store (through
    open_store and load_zarr), for a store opened directly and for the source of a mounted store. The
    stores it returns for the metered locations opened read-only are wrapped with
    experiments/runtime/recording_store.py's RecordingStoreWrapper, one probe per name; other locations pass
    through. A protected (published) location opened writable raises PermissionError.
    """

    def __init__(self, locations: dict[str, str]) -> None:
        self.names = {_identity(uri): name for name, uri in locations.items() if uri}
        self.probes: dict = {}
        self.opened = {
            name: {"readOnly": 0, "writable": 0} for name in locations if locations[name]
        }
        self._original = None

    def __enter__(self) -> "ReadMeter":
        from cytearc.storage import stores
        from zarr.storage import LocalStore

        from experiments.runtime import recording_store

        self.probes = {name: recording_store.StoreProbe() for name in self.opened}
        original = stores.make_store

        def make_store(location, *, storage_options=None, read_only=False):
            store = original(location, storage_options=storage_options, read_only=read_only)
            name = self.names.get(_identity(location)) if isinstance(location, str) else None
            if name is None:
                return store
            if not read_only:
                if name in PROTECTED:
                    raise PermissionError(f"the {name} store may only be opened read-only")
                self.opened[name]["writable"] += 1
                return store
            self.opened[name]["readOnly"] += 1
            if isinstance(store, str):
                store = LocalStore(store.removeprefix("file://"), read_only=True)
            return recording_store.RecordingStoreWrapper(store, probe=self.probes[name])

        self._original = original
        stores.make_store = make_store
        return self

    def __exit__(self, *exc) -> bool:
        from cytearc.storage import stores

        if self._original is not None:
            stores.make_store = self._original
        return False

    def totals(self) -> dict:
        out = {}
        for name, probe in self.probes.items():
            values = probe.to_json()
            requests = sum(
                (int(values[key]) for key in ("gets", "rangeGets", "partialGets", "sizeQueries"))
            )
            out[name] = {"readBytes": int(values["readTransferredBytes"]), "requests": requests}
        return out


class Steps(mb.Steps):
    """mb.Steps (wall time, network bytes received, peak memory; the record saved before and after
    each step) with the bytes and requests that each step read from every metered store."""

    def __init__(self, save, meter: ReadMeter) -> None:
        super().__init__(save)
        self.meter = meter

    def run(self, name: str, fn):
        if name in self.records:
            raise ValueError(f"step {name} ran already in this job")
        before = self.meter.totals()
        log(f"start {name}")
        value = super().run(name, fn)
        after = self.meter.totals()
        reads = {}
        for store, total in after.items():
            delta = {key: total[key] - before.get(store, {}).get(key, 0) for key in total}
            if delta["readBytes"] or delta["requests"]:
                reads[store] = delta
        self.records[name]["reads"] = reads
        self.save()
        log(f"done {name}: {self.records[name]['seconds']:.1f} s")
        return value

    def seconds(self, name: str) -> float | None:
        return self.records.get(name, {}).get("seconds")


def _run(payload: dict, step: str, body, meter: dict[str, str] | None = None) -> dict:
    mb._prepare(payload)
    objects = mb.Objects(payload.get("r2EndpointUrl"))
    uri = payload["recordUri"]
    previous = objects.get_json(uri) if objects.exists(uri) else None
    if previous is not None and previous.get("status") == "ok" and (not payload.get("force")):
        log(f"{step}: an ok record exists; nothing to do")
        return {"status": "existing", "step": step}
    record: dict = {
        "step": step,
        "status": "running",
        "test": bool(payload.get("test")),
        "startedAt": time.time(),
        "provenance": _provenance(payload),
    }
    if previous is not None:
        record["previousAttempt"] = {
            key: previous.get(key)
            for key in ("status", "startedAt", "finishedAt", "currentStep", "attempt")
        }
    reads = ReadMeter(meter or {})
    steps = Steps(
        lambda: objects.put_json(
            uri,
            _relative(
                _plain({**record, "steps": steps.records, "currentStep": steps.current}), payload
            ),
        ),
        reads,
    )
    try:
        with reads:
            body(payload, record, steps, objects)
        record["status"] = "ok"
    except BaseException:
        record["status"] = "error"
        record["error"] = traceback.format_exc()[-6000:]
        log(record["error"])
    record.update(
        {
            "finishedAt": time.time(),
            "steps": steps.records,
            "currentStep": steps.current,
            "storesOpened": reads.opened,
        }
    )
    objects.put_json(uri, _relative(_plain(record), payload))
    log(f"{step}: {record['status']}")
    if record["status"] == "error":
        raise RuntimeError(record["error"])
    return {
        "status": record["status"],
        "step": step,
        "error": _redact(record.get("error", ""))[-800:],
    }


def _meter(payload: dict, *names: str) -> dict[str, str]:
    locations = {
        "core": payload["published"]["hlca_core"],
        "full": payload["sources"]["hlca_full"],
        "natri": payload["sources"]["natri_ild"],
        "revised": payload["stores"]["reference"],
    }
    return {name: locations[name] for name in names}


def _entries(objects, uri: str) -> dict[str, int]:
    if not mb._is_remote(uri):
        base = Path(uri)
        return (
            {str(p.relative_to(base)): p.stat().st_size for p in base.rglob("*") if p.is_file()}
            if base.exists()
            else {}
        )
    r2 = objects._remote()
    bucket, prefix = r2.split(uri.rstrip("/") + "/")
    out: dict[str, int] = {}
    for page in r2.client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
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


def _part(key: str) -> str:
    found = _artifact_of(key)
    if found is not None:
        return f"artifact:{found[0]}"
    parts = key.split("/")
    if len(parts) > 2 and parts[1] == "featureData":
        return "featureData"
    return parts[0] if parts[0] in ("cellData", "pipeline") else "other"


class Ledger:
    """What each request added to a store, from listings taken before and after it."""

    def __init__(self, objects, uri: str) -> None:
        self.objects, self.uri = (objects, uri)
        self.last = _entries(objects, uri)

    def size(self) -> dict:
        parts: dict[str, int] = {}
        for key, size in self.last.items():
            parts[_part(key)] = parts.get(_part(key), 0) + size
        return {
            "bytes": int(sum(self.last.values())),
            "objects": len(self.last),
            "bytesByPart": dict(sorted(parts.items())),
        }

    def own(self) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        for key, size in self.last.items():
            found = _artifact_of(key)
            if found is not None:
                out[found] = out.get(found, 0) + size
        return out

    def take(self) -> dict:
        now = _entries(self.objects, self.uri)
        before, self.last = (self.last, now)
        added = [k for k in now if k not in before]
        old = {a for a in map(_artifact_of, before) if a}
        created: dict[tuple[str, str], int] = {}
        parts: dict[str, int] = {}
        for key in added:
            found = _artifact_of(key)
            if found is not None and found not in old:
                created[found] = created.get(found, 0) + now[key]
            parts[_part(key)] = parts.get(_part(key), 0) + now[key]
        return {
            "bytesAdded": int(sum(now.values()) - sum(before.values())),
            "objectsAdded": len(now) - len(before),
            "newObjects": len(added),
            "changedObjects": sum((1 for k in now if k in before and before[k] != now[k])),
            "removedObjects": sum((1 for k in before if k not in now)),
            "newBytesByPart": dict(sorted(parts.items())),
            "createdArtifacts": [
                {"kind": kind, "artifactId": aid, "bytes": size}
                for (kind, aid), size in sorted(created.items())
            ],
            "storeBytes": int(sum(now.values())),
            "storeObjects": len(now),
        }


def _options(target: str, payload: dict) -> dict:
    return {
        "default_assay": "RNA",
        "nthreads": int(payload["workers"]),
        "mem_budget": int(payload["budgetBytes"]),
        "storage_options": mb._storage_options(target),
    }


def _mount(objects, payload: dict, source: str, target: str):
    import cytearc

    if mb._store_exists(objects, target):
        document = objects.get_json(f"{target.rstrip('/')}/zarr.json")
        if "matrixSource" not in (document.get("attributes") or {}):
            raise RuntimeError(
                f"{target} exists but is not a complete mount (an interrupted mount?); delete it to mount again"
            )
        return cytearc.DataStore(target, zarr_mode="r+", **_options(target, payload))
    return cytearc.mount_datastore(source, at=target, **_options(target, payload))


def _pipeline(ds, label: str, **settings):
    try:
        run = ds.pipeline.open(label=label)
        if run.status == "completed":
            return (run, "existing")
    except Exception:
        pass
    return (mb.run_pipeline(ds, label=label, **settings), "ran")


def _reference_settings(cell_key: str) -> dict:
    return {
        "cells": cell_key,
        "filtering": False,
        "hvg": {"top_n": mb.HVG},
        "pca": {"dims": mb.DIMS},
        "neighbors": {"k": mb.NEIGHBOURS},
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


def _reference_info(reference) -> dict:
    import numpy as np

    return {
        "mappingReference": reference.ref.to_dict(),
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


def _reference_ref(objects, payload: dict, version: str):
    from cytearc import ArtifactRef

    if version == "v1":
        uri = f"{payload['primaryPrefix']}/convert_hlca_core.json"
        converted = objects.get_json(uri) if objects.exists(uri) else {}
        if converted.get("status") != "ok" or not (converted.get("reference") or {}).get(
            "mappingReference"
        ):
            raise RuntimeError(
                "lung_pipeline.py's core conversion (convert_hlca_core.json) is not complete"
            )
        return (
            payload["published"]["hlca_core"],
            ArtifactRef.from_dict(converted["reference"]["mappingReference"]),
        )
    uri = f"{payload['resultsPrefix']}/reference.json"
    built = objects.get_json(uri) if objects.exists(uri) else {}
    if built.get("status") != "ok":
        raise RuntimeError("the revised reference (reference.json) is not complete")
    return (
        payload["stores"]["reference"],
        ArtifactRef.from_dict(built["reference"]["mappingReference"]),
    )


def _open_reference(objects, payload: dict, version: str, steps: Steps):
    store, ref = _reference_ref(objects, payload, version)
    ds_ref = steps.run(f"openReference_{version}", lambda: mb._open(store, payload, mode="r"))
    reference = steps.run(f"loadReference_{version}", lambda: ds_ref.mapping.load_reference(ref))
    info = _reference_info(reference)
    return (
        reference,
        {
            "store": store,
            **{
                k: info[k]
                for k in (
                    "mappingReference",
                    "cellsSelected",
                    "panelGenes",
                    "distanceQuantiles",
                    "distanceValues",
                )
            },
        },
    )


def _columns(ds, names) -> dict:
    import numpy as np

    return {
        name: np.asarray(values).astype(str)
        for name, values in zip(names, ds.cells.fetch_all_columns(names), strict=True)
    }


def _fingerprint(ds, ref) -> str | None:
    try:
        return ds.artifacts.load(ref).attrs.get("payload_fingerprint")
    except Exception:
        return None


def _published_table(objects, payload: dict, name: str):
    import pandas as pd

    path = _fetch(
        objects,
        f"{payload['primaryPrefix']}/{name}.csv.gz",
        _work(payload) / "inputs" / f"{name}.csv.gz",
    )
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _centre_cells(objects, payload: dict, slug: str):
    import numpy as np

    spec = CENTRES[slug]
    table = _published_table(objects, payload, spec["table"])
    chosen = table[table[spec["column"]] == spec["value"]].reset_index(drop=True)
    if chosen.empty or chosen["cellId"].duplicated().any():
        raise RuntimeError(
            f"{slug}: no cells, or repeated cell identifiers, in {spec['table']}.csv.gz"
        )
    meta = chosen[list(META[spec["source"]])].copy()
    control = (
        chosen["region"] == CONTROL_REGION
        if spec["source"] == "natri_ild"
        else chosen["disease"] == mb.CONTROL_DISEASE
    )
    meta["group"] = np.where(control.to_numpy(), "control", "ILD")
    return (meta, chosen)


def _add_author_labels(ds, meta):
    import numpy as np
    import pandas as pd

    ids = pd.Index(np.asarray(ds.cells.fetch_all("ids")).astype(str))
    labels = np.asarray(ds.cells.fetch_all(AUTHOR_COLUMN)).astype(str)
    rows = ids.get_indexer(meta["cellId"].to_numpy())
    meta = meta.copy()
    meta["authorLabel"] = np.where(rows >= 0, labels[np.clip(rows, 0, None)], "")
    return meta


def _select(ds, wanted, payload: dict):
    import numpy as np

    wanted = np.unique(np.asarray(wanted, dtype=str))
    ids = np.asarray(ds.cells.fetch_all("ids")).astype(str)
    mask = np.isin(ids, wanted) & np.asarray(ds.cells.fetch_all("I"), dtype=bool)
    if not payload.get("test") and int(mask.sum()) != wanted.size:
        raise RuntimeError(
            f"{int(mask.sum())} of the centre's {wanted.size} cells are active in its store"
        )
    if not mask.any():
        raise RuntimeError("none of the centre's cells is in its store")
    if CENTRE_KEY not in ds.cells.columns or not np.array_equal(
        np.asarray(ds.cells.fetch_all(CENTRE_KEY), dtype=bool), mask
    ):
        ds.cells.insert(CENTRE_KEY, mask, overwrite=True)
    return ds.snapshot_cell_selection(CENTRE_KEY)


def _request(name: str, fn, steps: Steps, ledger: Ledger, ds) -> tuple:
    ref = steps.run(name, fn)
    storage = ledger.take()
    created = {(a["kind"], a["artifactId"]) for a in storage["createdArtifacts"]}
    key = (ref.kind, ref.artifact_id)
    return (
        ref,
        {
            "request": name,
            "ref": ref.to_dict(),
            "created": key in created,
            "heldBy": "centre" if key in ledger.own() else "source",
            "payloadFingerprint": _fingerprint(ds, ref),
            "seconds": steps.seconds(name),
            "reads": steps.records[name].get("reads", {}),
            "storage": storage,
        },
    )


def _recipe(ds, payload: dict, objects, meta, versions, steps: Steps, ledger: Ledger) -> dict:
    out: dict = {"requests": {}, "refs": {}, "references": {}, "handles": {}}

    def request(name: str, fn):
        ref, entry = _request(name, fn, steps, ledger, ds)
        out["requests"][name] = entry
        out["refs"][name] = ref
        return ref

    selection = request("select", lambda: _select(ds, meta["cellId"].to_numpy(), payload))
    for version in versions:
        reference, info = _open_reference(objects, payload, version, steps)
        out["references"][version] = info
        out["handles"][version] = reference
        projection = request(
            f"map_{version}",
            lambda r=reference: ds.mapping.run(
                r,
                selection,
                query_assay="RNA",
                save_k=mb.NEIGHBOURS,
                missing_feature_policy=mb.MISSING_FEATURES,
            ),
        )
        for level in mb.LEVELS:
            name = f"transfer_{version}_{level}"
            column = _label_column(version, level)
            transfer = request(
                name,
                lambda r=reference, p=projection, c=column: ds.mapping.transfer_labels(
                    p, reference=r, reference_labels=c, threshold_fraction=mb.VOTE_THRESHOLD
                ),
            )
            loaded = ds.mapping.load_label_transfer(transfer)
            labels = loaded.reference_labels
            out["requests"][name].update(
                {
                    "labelSource": str(loaded.reference_label_source),
                    "frozenLabels": labels.to_dict(),
                    "frozenLabelsHeldBy": "centre"
                    if (labels.kind, labels.artifact_id) in ledger.own()
                    else "source",
                    "classes": len(loaded.categories),
                }
            )
    return out


def _transfer_table(ds, transfers: dict, meta):
    import numpy as np
    import pandas as pd

    ids = np.asarray(ds.cells.fetch_all("ids")).astype(str)
    frame, first = (None, None)
    for level in mb.LEVELS:
        result = ds.mapping.load_label_transfer(transfers[level])
        evidence = result.evidence
        if first is None:
            first = result
            frame = pd.DataFrame({"cellId": ids[np.asarray(result.cell_idx, dtype=np.int64)]})
            frame["nearestDistance"] = evidence["nearestDistance"].to_numpy(dtype=np.float64)
            frame["distancePercentile"] = evidence["referenceDistancePercentile"].to_numpy(
                dtype=np.float64
            )
            frame["uninformative"] = (
                (evidence["abstentionReason"] == "uninformative_cell").to_numpy().astype(int)
            )
        elif result.projection != first.projection or not np.array_equal(
            result.cell_idx, first.cell_idx
        ):
            raise RuntimeError("the label transfers of one version do not share their projection")
        frame[f"label_{level}"] = evidence["label"].fillna("NA").astype(str).to_numpy()
        frame[f"vote_{level}"] = evidence["voteFraction"].to_numpy(dtype=np.float64)
    table = frame.merge(meta, on="cellId", how="left", validate="one_to_one", indicator=True)
    if (table["_merge"] != "both").any():
        raise RuntimeError(
            f"{int((table['_merge'] != 'both').sum())} mapped cells are not among the centre's cells"
        )
    return table[[*meta.columns, *RESULT_COLUMNS]].reset_index(drop=True)


def _upload_table(objects, table, payload: dict, name: str) -> dict:
    path = _work(payload) / "tables" / f"{name}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, compression="gzip")
    objects.upload(path, f"{payload['resultsPrefix']}/{name}.csv.gz")
    return {
        "file": f"{name}.csv.gz",
        "rows": len(table),
        "bytes": path.stat().st_size,
        "sha256": mb._sha256(path),
    }


def _read_table(path: Path):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in FLOAT_COLUMNS:
        frame[column] = np.asarray(
            frame[column].replace("", "nan").to_numpy(dtype=object), dtype=np.float64
        )
    frame["uninformative"] = frame["uninformative"].astype(int)
    return frame


def _compare_tables(first, second) -> dict:
    import numpy as np

    same_cells = len(first) == len(second) and np.array_equal(
        first["cellId"].to_numpy(dtype=str), second["cellId"].to_numpy(dtype=str)
    )
    columns = {}
    if same_cells:
        for column in RESULT_COLUMNS:
            if column in FLOAT_COLUMNS:
                x, y = (
                    first[column].to_numpy(dtype=np.float64),
                    second[column].to_numpy(dtype=np.float64),
                )
                differ = ~((x == y) | np.isnan(x) & np.isnan(y))
                gap = np.abs(x - y)
                columns[column] = {
                    "identical": not differ.any(),
                    "differing": int(differ.sum()),
                    "maxAbsDifference": float(np.nanmax(gap)) if np.isfinite(gap).any() else 0.0,
                }
            else:
                differ = (
                    first[column].astype(str).to_numpy() != second[column].astype(str).to_numpy()
                )
                columns[column] = {"identical": not differ.any(), "differing": int(differ.sum())}
    return {
        "cells": [len(first), len(second)],
        "sameCells": bool(same_cells),
        "identical": bool(same_cells and all((c["identical"] for c in columns.values()))),
        "columns": columns,
    }


def _published_check(table, published) -> dict:
    import numpy as np

    theirs = published.set_index("cellId").reindex(table["cellId"])
    present = table["cellId"].isin(published["cellId"]).to_numpy()

    def numbers(column: str):
        return np.asarray(
            theirs[column].fillna("").replace("", "nan").to_numpy(dtype=object), dtype=np.float64
        )

    def share(values) -> float | None:
        return float(np.asarray(values)[present].mean()) if present.any() else None

    out = {
        "cells": len(table),
        "inPrimaryTable": int(present.sum()),
        "nearestDistanceAgrees": share(
            np.isclose(
                table["nearestDistance"].to_numpy(dtype=float),
                numbers("nearestDistance"),
                rtol=1e-05,
                atol=1e-06,
                equal_nan=True,
            )
        ),
    }
    for level in mb.LEVELS:
        out[f"labelAgrees_{level}"] = share(
            table[f"label_{level}"].to_numpy(dtype=str)
            == theirs[f"label_{level}"].fillna("").to_numpy(dtype=str)
        )
        out[f"voteAgrees_{level}"] = share(
            np.isclose(
                table[f"vote_{level}"].to_numpy(dtype=float),
                numbers(f"vote_{level}"),
                rtol=1e-05,
                atol=1e-06,
                equal_nan=True,
            )
        )
    return out


def _projection_check(ds, projection, reference, table) -> dict:
    import numpy as np

    result = ds.mapping.load_result(projection, reference=reference, load_arrays=True)
    informative = table["uninformative"].to_numpy() == 0
    gap = np.abs(
        np.asarray(result.distances[:, 0], dtype=np.float64)[informative]
        - table["nearestDistance"].to_numpy(dtype=np.float64)[informative]
    )
    return {
        "cells": int(result.n_cells),
        "uninformativeAgree": bool(
            np.array_equal(
                np.asarray(result.uninformative, dtype=bool).astype(int),
                table["uninformative"].to_numpy(),
            )
        ),
        "nearestDistanceIdenticalOnInformativeCells": bool(
            gap.size == 0 or float(np.nanmax(gap)) == 0.0
        ),
        "nearestDistanceMaxAbsDifference": float(np.nanmax(gap)) if gap.size else 0.0,
        "diagnostics": {
            k: v if isinstance(v, int | float | str) else str(v)
            for k, v in dict(result.diagnostics).items()
        },
    }


def _table_summary(table) -> dict:
    return {
        "cells": len(table),
        "cellsByGroup": {str(k): int(v) for k, v in table["group"].value_counts().items()},
        "donorsByGroup": {
            str(k): int(v) for k, v in table.groupby("group")["donor"].nunique().items()
        },
        "uninformativeCells": int(table["uninformative"].sum()),
        "abstainedFraction": {
            level: float((table[f"label_{level}"] == "NA").mean()) for level in mb.LEVELS
        },
        "newLabelCells": int((table["label_ann_finest_level"] == NEW_LABEL).sum()),
    }


def _inventory(ds, ledger: Ledger) -> dict:
    own = ledger.own()
    visible = set()
    for scope in ("assay", "datastore"):
        for ref in ds.artifacts.list(scope=scope):
            visible.add((ref.kind, ref.artifact_id))
    source = sorted(visible - set(own))
    return {
        "ownArtifacts": len(own),
        "sourceArtifacts": len(source),
        "own": [
            {"kind": kind, "artifactId": aid, "bytes": size}
            for (kind, aid), size in sorted(own.items())
        ],
        "fromSource": [{"kind": kind, "artifactId": aid} for kind, aid in source],
        "ownNotListed": sorted((f"{kind}:{aid}" for kind, aid in set(own) - visible)),
    }


def _lineage(ds, outputs: dict, reference, own: dict) -> dict:
    from cytearc.storage.refs import ExternalArtifactRef

    graph = ds.artifacts.lineage(outputs, references=reference).graph

    def node_id(locator) -> str:
        if isinstance(locator, ExternalArtifactRef):
            return f"reference:{locator.ref.kind}:{locator.ref.artifact_id}"
        return f"{locator.kind}:{locator.artifact_id}"

    nodes = []
    for locator in graph.nodes:
        status = graph.nodes[locator]["status"]
        external = isinstance(locator, ExternalArtifactRef)
        ref = locator.ref if external else locator
        nodes.append(
            {
                "id": node_id(locator),
                "kind": ref.kind,
                "artifactId": ref.artifact_id,
                "scope": ref.scope,
                "store": "reference"
                if external
                else "centre"
                if (ref.kind, ref.artifact_id) in own
                else "source",
                "operation": None if status is None else status.operation,
                "complete": bool(status is not None and status.complete),
                "outputs": list(graph.nodes[locator]["outputs"]),
            }
        )
    edges = [
        {"from": node_id(a), "to": node_id(b), "inputs": list(graph.edges[a, b]["inputs"])}
        for a, b in graph.edges
    ]
    return {
        "nodes": sorted(nodes, key=lambda n: n["id"]),
        "edges": sorted(edges, key=lambda e: (e["to"], e["from"])),
    }


def _classify(lineage: dict, before: dict, after: dict) -> dict:

    def local(version: str) -> dict:
        return {
            (n["kind"], n["artifactId"]): n
            for n in lineage[version]["nodes"]
            if n["store"] != "reference"
        }

    v1, v2 = (local("v1"), local("v2"))
    rows = []
    for key, node in sorted(v2.items()):
        status = "new" if key in after and key not in before else "reused"
        rows.append(
            {
                "versions": "v1 and v2" if key in v1 else "v2",
                "status": status,
                **{k: node[k] for k in ("kind", "artifactId", "store", "operation")},
            }
        )
    for key, node in sorted(v1.items()):
        if key not in v2:
            rows.append(
                {
                    "versions": "v1",
                    "status": "superseded",
                    **{k: node[k] for k in ("kind", "artifactId", "store", "operation")},
                }
            )
    for version in VERSIONS:
        for node in lineage[version]["nodes"]:
            if node["store"] == "reference":
                rows.append(
                    {
                        "versions": version,
                        "status": f"reference {version}",
                        **{k: node[k] for k in ("kind", "artifactId", "store", "operation")},
                    }
                )
    counts: dict = {}
    kinds: dict = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        kinds.setdefault(row["status"], set()).add(row["kind"])
    changed = sorted(
        (f"{kind}:{aid}" for (kind, aid), size in before.items() if after.get((kind, aid)) != size)
    )
    return {
        "artifacts": rows,
        "counts": counts,
        "kinds": {status: sorted(values) for status, values in kinds.items()},
        "ownBefore": len(before),
        "ownAfter": len(after),
        "ownUnchanged": sum((1 for key, size in before.items() if after.get(key) == size)),
        "ownChangedOrRemoved": changed,
        "ownNew": sorted((f"{kind}:{aid}" for kind, aid in set(after) - set(before))),
    }


def setup_test(payload: dict) -> dict:
    return _run(
        payload,
        "setup_test",
        _setup_test,
        meter={
            "full": payload["published"]["hlca_full"],
            "natri": payload["published"]["natri_ild"],
        },
    )


def _sample(rng, rows, n: int):
    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    return np.sort(rng.choice(rows, min(n, rows.size), replace=False)) if rows.size else rows


def _subset(ds, rows, target: str, payload: dict) -> None:
    import cytearc
    import numpy as np

    writer = cytearc.SubsetZarr(
        zarr_loc=target,
        assays=[ds.RNA],
        cell_idx=np.sort(np.asarray(rows, dtype=np.int64)),
        overwrite_existing_file=True,
        storage_options=mb._storage_options(target),
        mem_budget=int(payload["budgetBytes"]),
        nthreads=int(payload["workers"]),
    )
    writer.dump()


def _setup_test(payload: dict, record: dict, steps: Steps, objects) -> None:
    import numpy as np
    import pandas as pd

    published, sources = (payload["published"], payload["sources"])
    rng = np.random.default_rng(TEST_SEED)
    replication = steps.run(
        "loadReplicationTable", lambda: _published_table(objects, payload, "replication_cells")
    )
    natri_table = steps.run(
        "loadNatriTable", lambda: _published_table(objects, payload, "natri_cells")
    )
    ds_full = steps.run("openFull", lambda: mb._open(published["hlca_full"], payload, mode="r"))
    column = _columns(ds_full, ("ids", "core_or_extension", "study", "disease", AUTHOR_COLUMN))
    active = np.asarray(ds_full.cells.fetch_all("I"), dtype=bool)
    revision = (
        active
        & (column["core_or_extension"] == "extension")
        & (column["study"] == REVISION_STUDY)
        & np.isin(column["disease"], (*mb.ILD_DISEASES, mb.CONTROL_DISEASE))
    )
    basaloid = revision & (column[AUTHOR_COLUMN] == AUTHOR_BASALOID)
    parts = {
        "core": _sample(
            rng, np.flatnonzero(active & (column["core_or_extension"] == "core")), TEST_CORE
        ),
        "revisionBasaloid": np.flatnonzero(basaloid),
        "revisionOthers": _sample(rng, np.flatnonzero(revision & ~basaloid), TEST_REVISION),
    }
    index = pd.Index(column["ids"])
    for slug in FULL_CENTRES:
        rows = index.get_indexer(
            replication.loc[replication["study"] == CENTRES[slug]["value"], "cellId"].to_numpy()
        )
        parts[slug] = _sample(rng, rows[rows >= 0], TEST_CENTRE)
    rows = np.unique(np.concatenate(list(parts.values())))
    steps.run("subsetFull", lambda: _subset(ds_full, rows, sources["hlca_full"], payload))
    ds_test = steps.run(
        "initialiseFull", lambda: mb._open(sources["hlca_full"], payload, initialise=True)
    )
    record["full"] = {
        "store": sources["hlca_full"],
        "cells": int(rows.size),
        "parts": {name: int(part.size) for name, part in parts.items()},
        "activeCells": int(np.asarray(ds_test.cells.fetch_all("I"), dtype=bool).sum()),
        "storeSize": objects.prefix_bytes(sources["hlca_full"]),
    }
    ds_natri = steps.run("openNatri", lambda: mb._open(published["natri_ild"], payload, mode="r"))
    natri_ids = pd.Index(np.asarray(ds_natri.cells.fetch_all("ids")).astype(str))
    query = natri_table[natri_table["donorClass"] == CENTRES["natri"]["value"]]
    chosen = [
        rng.choice(group["cellId"].to_numpy(), min(len(group), TEST_PER_DONOR), replace=False)
        for _, group in query.groupby("donor", sort=True)
    ]
    rows = natri_ids.get_indexer(np.concatenate(chosen))
    rows = np.unique(rows[rows >= 0])
    steps.run("subsetNatri", lambda: _subset(ds_natri, rows, sources["natri_ild"], payload))
    ds_test = steps.run(
        "initialiseNatri", lambda: mb._open(sources["natri_ild"], payload, initialise=True)
    )
    record["natri"] = {
        "store": sources["natri_ild"],
        "cells": int(rows.size),
        "donors": int(query["donor"].nunique()),
        "activeCells": int(np.asarray(ds_test.cells.fetch_all("I"), dtype=bool).sum()),
        "storeSize": objects.prefix_bytes(sources["natri_ild"]),
    }


def build_reference(payload: dict) -> dict:
    return _run(payload, "reference", _reference, meter=_meter(payload, "core", "full"))


def _reference(payload: dict, record: dict, steps: Steps, objects) -> None:
    import numpy as np
    import pandas as pd

    target, source = (payload["stores"]["reference"], payload["sources"]["hlca_full"])
    existed = mb._store_exists(objects, target)
    ds = steps.run("mount", lambda: _mount(objects, payload, source, target))
    ledger = Ledger(objects, target)
    record["mount"] = {
        "store": target,
        "source": source,
        "existed": existed,
        "cells": int(ds.cells.N),
        "genes": int(ds.RNA.feats.N),
        **ledger.size(),
    }
    names = ("ids", "core_or_extension", "study", "disease", "donor_id", AUTHOR_COLUMN, *mb.LEVELS)
    column = steps.run("readCells", lambda: _columns(ds, names))
    active = np.asarray(ds.cells.fetch_all("I"), dtype=bool)
    core = active & (column["core_or_extension"] == "core")
    revision = (
        active & (column["core_or_extension"] == "extension") & (column["study"] == REVISION_STUDY)
    )
    eligible = np.isin(column["disease"], (*mb.ILD_DISEASES, mb.CONTROL_DISEASE))
    added = revision & eligible
    mask = core | added
    basaloid = added & (column[AUTHOR_COLUMN] == AUTHOR_BASALOID)
    if not basaloid.any():
        raise RuntimeError(
            f"no {REVISION_STUDY} cell of the selection has {AUTHOR_COLUMN} '{AUTHOR_BASALOID}': the revision cannot be built as planned"
        )
    ds_core = steps.run(
        "openCore", lambda: mb._open(payload["published"]["hlca_core"], payload, mode="r")
    )
    core_values = steps.run("readCoreLabels", lambda: _columns(ds_core, ("ids", *mb.LEVELS)))
    core_rows, added_rows = (np.flatnonzero(core), np.flatnonzero(added))
    position = pd.Index(core_values["ids"]).get_indexer(column["ids"][core_rows])
    found = position >= 0
    if not found.all():
        raise RuntimeError(
            f"{int((~found).sum())} core cells of the full HLCA are not in the core store"
        )
    labels, checks = ({}, {})
    for level in mb.LEVELS:
        hlca = column[level]
        values = np.full(hlca.size, "", dtype=object)
        from_core = core_values[level][position]
        values[core_rows] = from_core
        kept = hlca[added_rows].astype(object)
        kept[np.isin(kept, NO_VOTE)] = ""
        values[added_rows] = kept
        values[basaloid] = REVISED[level]
        labels[level] = values
        chosen = values[mask]
        checks[level] = {
            "coreCells": int(core_rows.size),
            "coreLabelsSameInFullHlca": int((from_core == hlca[core_rows]).sum()),
            "coreClasses": int(np.unique(from_core).size),
            "revisionCellsUnlabelled": int((values[added_rows] == "").sum()),
            "revisionClassesNotInCore": sorted(
                set(kept[kept != ""].tolist()) - set(from_core.tolist())
            ),
            "classes": int(np.unique(chosen[chosen != ""]).size),
            "counts": {
                str(k): int(v) for k, v in pd.Series(chosen[chosen != ""]).value_counts().items()
            },
        }

    def count(rows) -> dict:
        return {"cells": int(rows.sum()), "donors": int(np.unique(column["donor_id"][rows]).size)}

    by_disease = (
        pd.DataFrame(
            {
                "disease": column["disease"][added],
                "donor": column["donor_id"][added],
                "basaloid": basaloid[added],
            }
        )
        .groupby("disease")
        .agg(
            cells=("donor", "size"), donors=("donor", "nunique"), basaloidCells=("basaloid", "sum")
        )
        .reset_index()
        .to_dict(orient="records")
    )
    record["selection"] = {
        "rule": f"active core cells, and active {REVISION_STUDY} extension cells with disease in the ILD diagnoses or '{mb.CONTROL_DISEASE}'",
        "core": count(core),
        "revisionStudy": count(revision),
        "leftOut": {
            **count(revision & ~eligible),
            "diseases": sorted(set(column["disease"][revision & ~eligible].tolist())),
        },
        "added": count(added),
        "addedByDisease": by_disease,
        "selected": count(mask),
        "basaloid": {
            **count(basaloid),
            "authorLabel": AUTHOR_BASALOID,
            "byDisease": {
                str(k): int(v)
                for k, v in pd.Series(column["disease"][basaloid]).value_counts().items()
            },
            "hlcaLabelsBefore": {
                level: {
                    str(k): int(v)
                    for k, v in pd.Series(column[level][basaloid]).value_counts().head(12).items()
                }
                for level in mb.LEVELS
            },
        },
        "labels": checks,
    }

    def insert() -> list[str]:
        written = []
        for level in mb.LEVELS:
            name = _label_column("v2", level)
            wanted = labels[level].astype(str)
            if name not in ds.cells.columns or not np.array_equal(
                np.asarray(ds.cells.fetch_all(name)).astype(str), wanted
            ):
                ds.cells.insert(name, wanted, overwrite=True)
                written.append(name)
        if V2_KEY not in ds.cells.columns or not np.array_equal(
            np.asarray(ds.cells.fetch_all(V2_KEY), dtype=bool), mask
        ):
            ds.cells.insert(V2_KEY, mask, overwrite=True)
            written.append(V2_KEY)
        return written

    record["columns"] = {"written": steps.run("insertColumns", insert), "storage": ledger.take()}
    run, how = steps.run(
        "pipeline", lambda: _pipeline(ds, V2_RUN_LABEL, **_reference_settings(V2_KEY))
    )
    record["pipelineStorage"] = ledger.take()
    ref = steps.run("buildReference", lambda: ds.mapping.build_reference(run["neighbors"]))
    record["buildStorage"] = ledger.take()
    v2 = ds.mapping.load_reference(ref)
    if int(v2.selected_cell_count) != int(mask.sum()):
        raise RuntimeError("the revised reference does not hold the selected cells")
    report = _report(run)
    record["reference"] = {
        "label": V2_RUN_LABEL,
        "cellKey": V2_KEY,
        "labelColumns": {level: _label_column("v2", level) for level in mb.LEVELS},
        "runStatus": how,
        "pipelineRunId": run.run_id,
        "neighbors": run["neighbors"].to_dict(),
        "settings": {
            "hvgCount": mb.HVG,
            "pcaDims": mb.DIMS,
            "neighbours": mb.NEIGHBOURS,
            "filtering": False,
            "batchIntegration": None,
        },
        "pipelineSeconds": report["seconds"] if how == "existing" else steps.seconds("pipeline"),
        "buildSeconds": steps.seconds("buildReference"),
        "report": report,
        **_reference_info(v2),
    }
    _, v1_ref = _reference_ref(objects, payload, "v1")
    v1 = steps.run("loadReferenceV1", lambda: ds_core.mapping.load_reference(v1_ref))
    genes = {
        name: set(np.asarray(handle.feature_ids).astype(str).tolist())
        for name, handle in (("v1", v1), ("v2", v2))
    }
    record["comparison"] = {
        "v1": {
            "mappingReference": v1_ref.to_dict(),
            "cellsSelected": int(v1.selected_cell_count),
            "panelGenes": len(genes["v1"]),
        },
        "v2": {"cellsSelected": int(v2.selected_cell_count), "panelGenes": len(genes["v2"])},
        "panelGenesShared": len(genes["v1"] & genes["v2"]),
    }
    record["storeSize"] = ledger.size()


def map_centre(payload: dict) -> dict:
    return _run(
        payload,
        f"centre_{payload['centre']}",
        _centre,
        meter=_meter(payload, "core", "full", "natri"),
    )


def _centre(payload: dict, record: dict, steps: Steps, objects) -> None:
    slug = payload["centre"]
    spec = CENTRES[slug]
    meta, published = steps.run("loadCells", lambda: _centre_cells(objects, payload, slug))
    target, source = (payload["stores"][f"centre_{slug}"], payload["sources"][spec["source"]])
    existed = mb._store_exists(objects, target)
    ds = steps.run("mount", lambda: _mount(objects, payload, source, target))
    ledger = Ledger(objects, target)
    record["centre"] = {
        "slug": slug,
        "name": spec["name"],
        "store": target,
        "source": source,
        "cellsInPrimaryTable": len(meta),
    }
    record["mount"] = {
        "existed": existed,
        "seconds": steps.seconds("mount"),
        "cells": int(ds.cells.N),
        **ledger.size(),
    }
    if spec["source"] == "hlca_full":
        meta = _add_author_labels(ds, meta)
    done = _recipe(ds, payload, objects, meta, ("v1",), steps, ledger)
    record["requests"] = done["requests"]
    record["refs"] = {name: ref.to_dict() for name, ref in done["refs"].items()}
    record["references"] = done["references"]
    transfers = {level: done["refs"][f"transfer_v1_{level}"] for level in mb.LEVELS}
    table = steps.run("tableV1", lambda: _transfer_table(ds, transfers, meta))
    record["table"] = steps.run(
        "uploadV1", lambda: _upload_table(objects, table, payload, f"cells_{slug}_v1")
    )
    record["summary"] = _table_summary(table)
    record["checks"] = {
        "projection": _projection_check(ds, done["refs"]["map_v1"], done["handles"]["v1"], table),
        "primaryRun": _published_check(table, published),
    }
    record["inventory"] = _inventory(ds, ledger)
    record["storeSize"] = ledger.size()


def update_centre(payload: dict) -> dict:
    return _run(
        payload,
        f"update_{payload['centre']}",
        _update,
        meter=_meter(payload, "core", "full", "natri", "revised"),
    )


def _recover(objects, payload: dict, slug: str, phase1: dict, meta) -> dict:
    from cytearc import ArtifactRef

    path = _fetch(
        objects,
        f"{payload['resultsPrefix']}/{phase1['table']['file']}",
        _work(payload) / "recovery" / phase1["table"]["file"],
    )
    digest = mb._sha256(path)
    written = _read_table(path)
    ds = mb._open(payload["stores"][f"centre_{slug}"], payload, mode="r")
    transfers = {
        level: ArtifactRef.from_dict(phase1["refs"][f"transfer_v1_{level}"]) for level in mb.LEVELS
    }
    recovered = _transfer_table(ds, transfers, meta)
    return {
        "table": phase1["table"]["file"],
        "tableAsWritten": digest == phase1["table"]["sha256"],
        **_compare_tables(written, recovered),
    }


def _update(payload: dict, record: dict, steps: Steps, objects) -> None:
    import numpy as np

    slug = payload["centre"]
    spec = CENTRES[slug]
    uri = f"{payload['resultsPrefix']}/centre_{slug}.json"
    phase1 = objects.get_json(uri) if objects.exists(uri) else {}
    if phase1.get("status") != "ok":
        raise RuntimeError(f"centre_{slug}.json (phase 1) is not complete")
    target = payload["stores"][f"centre_{slug}"]
    if not mb._store_exists(objects, target):
        raise RuntimeError("the centre's store does not exist")
    meta, _ = steps.run("loadCells", lambda: _centre_cells(objects, payload, slug))
    ds = steps.run("open", lambda: mb._open(target, payload))
    if spec["source"] == "hlca_full":
        meta = _add_author_labels(ds, meta)
    ledger = Ledger(objects, target)
    before = ledger.own()
    record["centre"] = {"slug": slug, "name": spec["name"], "store": target}
    record["before"] = {"ownArtifacts": len(before), **ledger.size()}
    done = _recipe(ds, payload, objects, meta, VERSIONS, steps, ledger)
    record["requests"] = done["requests"]
    record["refs"] = {name: ref.to_dict() for name, ref in done["refs"].items()}
    record["references"] = done["references"]
    record["v1Unchanged"] = {
        name: ref["artifact_id"] == done["refs"][name].artifact_id
        for name, ref in phase1["refs"].items()
    }
    v2_requests = ["map_v2", *(f"transfer_v2_{level}" for level in mb.LEVELS)]
    record["measurement"] = {
        "v2CreatedHere": all((done["requests"][name]["created"] for name in v2_requests)),
        "notFirstAttempt": record.get("previousAttempt") is not None,
    }
    transfers = {level: done["refs"][f"transfer_v2_{level}"] for level in mb.LEVELS}
    table = steps.run("tableV2", lambda: _transfer_table(ds, transfers, meta))
    record["table"] = steps.run(
        "uploadV2", lambda: _upload_table(objects, table, payload, f"cells_{slug}_v2")
    )
    record["summary"] = _table_summary(table)
    after = ledger.own()
    record["after"] = {"ownArtifacts": len(after), **ledger.size()}
    lineage = {
        version: _lineage(
            ds,
            {level: done["refs"][f"transfer_{version}_{level}"] for level in mb.LEVELS},
            done["handles"][version],
            after,
        )
        for version in VERSIONS
    }
    record["lineage"] = lineage
    record["traceability"] = _classify(lineage, before, after)
    genes = {
        version: set(np.asarray(done["handles"][version].feature_ids).astype(str).tolist())
        for version in VERSIONS
    }
    record["panelGenes"] = {
        "v1": len(genes["v1"]),
        "v2": len(genes["v2"]),
        "shared": len(genes["v1"] & genes["v2"]),
    }
    recovery = steps.run("recover", lambda: _recover(objects, payload, slug, phase1, meta))
    reads = steps.records["recover"].get("reads", {})
    recovery["referenceReadBytes"] = int(
        sum(((reads.get(name) or {}).get("readBytes", 0) for name in ("core", "revised")))
    )
    recovery["reads"] = reads
    record["recovery"] = recovery


def rerun_centre(payload: dict) -> dict:
    return _run(
        payload, "rerun_natri", _rerun, meter=_meter(payload, "core", "full", "natri", "revised")
    )


def _rerun(payload: dict, record: dict, steps: Steps, objects) -> None:
    slug = "natri"
    spec = CENTRES[slug]
    prefix = payload["resultsPrefix"]
    jobs = {}
    for name in (f"centre_{slug}", f"update_{slug}"):
        uri = f"{prefix}/{name}.json"
        jobs[name] = objects.get_json(uri) if objects.exists(uri) else {}
        if jobs[name].get("status") != "ok":
            raise RuntimeError(f"{name}.json is not complete")
    record["attempt"] = int((record.get("previousAttempt") or {}).get("attempt") or 0) + 1
    target = f"{payload['stores']['rerun']}_{record['attempt']}.zarr"
    if mb._store_exists(objects, target):
        raise RuntimeError(f"the rerun store of attempt {record['attempt']} exists already")
    meta, _ = steps.run("loadCells", lambda: _centre_cells(objects, payload, slug))
    ds = steps.run(
        "mount", lambda: _mount(objects, payload, payload["sources"][spec["source"]], target)
    )
    ledger = Ledger(objects, target)
    record["mount"] = {"store": target, "seconds": steps.seconds("mount"), **ledger.size()}
    done = _recipe(ds, payload, objects, meta, VERSIONS, steps, ledger)
    record["requests"] = done["requests"]
    record["refs"] = {name: ref.to_dict() for name, ref in done["refs"].items()}
    comparison = {}
    for version in VERSIONS:
        transfers = {level: done["refs"][f"transfer_{version}_{level}"] for level in mb.LEVELS}
        table = steps.run(f"table_{version}", lambda t=transfers: _transfer_table(ds, t, meta))
        written = jobs[f"centre_{slug}" if version == "v1" else f"update_{slug}"]["table"]["file"]
        path = _fetch(objects, f"{prefix}/{written}", _work(payload) / "rerun" / written)
        comparison[version] = {"against": written, **_compare_tables(_read_table(path), table)}
    record["comparison"] = comparison
    update = jobs[f"update_{slug}"]["requests"]
    record["payloadFingerprints"] = {
        name: {
            "rerun": entry["payloadFingerprint"],
            "update": (update.get(name) or {}).get("payloadFingerprint"),
            "same": entry["payloadFingerprint"]
            == (update.get(name) or {}).get("payloadFingerprint"),
        }
        for name, entry in done["requests"].items()
        if name != "select"
    }
    record["storeSize"] = ledger.size()


def stats(payload: dict) -> dict:
    return _run(payload, "stats", _stats)


def _calibrate(reference_screen, frames: dict, percentile) -> dict:
    import pandas as pd

    out = {}
    natri = frames["natri"]
    calibration = natri[natri["region"] == reference_screen.CONTROL]
    flags, table = reference_screen.calibrated_flags(natri, calibration, percentile)
    out["natri"] = {
        "flags": flags.astype(bool).to_numpy(),
        "thresholds": table,
        "calibration": "own controls",
        "calibrationCentres": ["natri"],
        "controlDonors": int(calibration["donor"].nunique()),
    }
    combined = pd.concat(
        [frames[slug].assign(centre=slug) for slug in FULL_CENTRES], ignore_index=True
    )
    controls = combined[combined["group"] == "control"]
    for slug in FULL_CENTRES:
        cells = combined[combined["centre"] == slug]
        own = controls[controls["centre"] == slug]
        assay = cells["assay"].mode().iat[0]
        same = controls[(controls["centre"] != slug) & (controls["assay"] == assay)]
        if own["donor"].nunique() >= reference_screen.MIN_CONTROL_DONORS:
            chosen, source = (own, "own controls")
        elif same["donor"].nunique() >= reference_screen.MIN_CONTROL_DONORS:
            chosen, source = (same, f"controls of the other centres with the same assay ({assay})")
        else:
            chosen, source = (controls[controls["centre"] != slug], "controls of the other centres")
        subject = pd.concat([cells, chosen if source != "own controls" else own.iloc[0:0]])
        subject_flags, table = reference_screen.calibrated_flags(subject, chosen, percentile)
        out[slug] = {
            "flags": subject_flags.loc[cells.index].astype(bool).to_numpy(),
            "thresholds": table,
            "calibration": source,
            "calibrationCentres": sorted(chosen["centre"].unique().tolist()),
            "controlDonors": int(chosen["donor"].nunique()),
        }
    return out


def _label_changes(slug: str, v1, v2) -> tuple[dict, list[dict]]:
    import pandas as pd

    summary, rows = ({}, [])
    for level in mb.LEVELS:
        a, b = (v1[f"label_{level}"].astype(str), v2[f"label_{level}"].astype(str))
        pairs = (
            pd.DataFrame({"labelV1": a, "labelV2": b})
            .groupby(["labelV1", "labelV2"])
            .size()
            .reset_index(name="cells")
        )
        rows += [{"centre": slug, "level": level, **row} for row in pairs.to_dict(orient="records")]
        changed = pairs[pairs["labelV1"] != pairs["labelV2"]].sort_values("cells", ascending=False)
        moved = a != b
        both = (a != "NA") & (b != "NA")
        new = b == NEW_LABEL
        summary[level] = {
            "cells": len(a),
            "changed": int(moved.sum()),
            "changedShare": float(moved.mean()),
            "changedAmongLabelledUnderBoth": float(moved[both].mean()) if both.any() else None,
            "abstainedV1": float((a == "NA").mean()),
            "abstainedV2": float((b == "NA").mean()),
            "newLabelCells": int(new.sum()),
            "newLabelFromV1": {str(k): int(v) for k, v in a[new].value_counts().head(10).items()},
            "topChanges": changed.head(12).to_dict(orient="records"),
        }
    return (summary, rows)


def _donor_proportions(slug: str, v1, v2) -> tuple[dict, list[dict]]:
    import pandas as pd

    groups = v1.groupby("donor")["group"].first()
    totals = v1.groupby("donor").size()
    summary, rows = ({}, [])
    for level in mb.LEVELS:
        column = f"label_{level}"
        joined = pd.concat(
            {
                "cellsV1": v1.groupby(["donor", column]).size(),
                "cellsV2": v2.groupby(["donor", column]).size(),
            },
            axis=1,
        ).fillna(0)
        joined.index.names = ["donor", "label"]
        joined = joined.reset_index()
        joined["shareV1"] = joined["cellsV1"] / joined["donor"].map(totals)
        joined["shareV2"] = joined["cellsV2"] / joined["donor"].map(totals)
        joined["difference"] = joined["shareV2"] - joined["shareV1"]
        joined["group"] = joined["donor"].map(groups)
        rows += [
            {"centre": slug, "level": level, **row} for row in joined.to_dict(orient="records")
        ]
        distance = joined.groupby("donor")["difference"].apply(lambda d: 0.5 * float(d.abs().sum()))
        largest = joined.reindex(
            joined["difference"].abs().sort_values(ascending=False).index
        ).head(10)
        summary[level] = {
            "donors": int(distance.size),
            "meanTotalVariation": float(distance.mean()),
            "meanTotalVariationByGroup": {
                str(g): float(distance[groups.reindex(distance.index) == g].mean())
                for g in sorted(groups.unique())
            },
            "maxTotalVariation": {"donor": str(distance.idxmax()), "value": float(distance.max())}
            if distance.size
            else None,
            "largestShareChanges": largest[
                ["donor", "group", "label", "shareV1", "shareV2", "difference"]
            ].to_dict(orient="records"),
        }
    new = (v2["label_ann_finest_level"] == NEW_LABEL).groupby(v2["donor"]).mean()
    summary["newLabelShareByDonor"] = {
        str(g): {
            "donors": int((groups == g).sum()),
            "donorsWithCells": int((new[groups.reindex(new.index) == g] > 0).sum()),
            "meanShare": float(new[groups.reindex(new.index) == g].mean()),
        }
        for g in sorted(groups.unique())
    }
    return (summary, rows)


def _unexplained(slug: str, v1, v2, f1, f2) -> tuple[dict, list[dict], list[dict]]:
    import numpy as np
    import pandas as pd

    usable1 = (v1["informative"] & (v1["lineage"] != "NA")).to_numpy()
    usable2 = (v2["informative"] & (v2["lineage"] != "NA")).to_numpy()
    frame = pd.DataFrame(
        {
            "donor": v1["donor"].astype(str).to_numpy(),
            "group": v1["group"].astype(str).to_numpy(),
            "lineageV1": v1["lineage"].to_numpy(),
            "lineageV2": v2["lineage"].to_numpy(),
            "stateV1": np.where(~usable1, "noLineage", np.where(f1, "unexplained", "explained")),
            "stateV2": np.where(~usable2, "noLineage", np.where(f2, "unexplained", "explained")),
            "basaloidV2": (v2["label_ann_finest_level"] == NEW_LABEL).to_numpy(),
            "labelV1": v1["label_ann_finest_level"].astype(str).to_numpy(),
            "authorLabel": v1["authorLabel"].astype(str).to_numpy() if "authorLabel" in v1 else "",
        }
    )

    def tally(part) -> dict:
        was = part["stateV1"] == "unexplained"
        usable_v1, usable_v2 = (
            int((part["stateV1"] != "noLineage").sum()),
            int((part["stateV2"] != "noLineage").sum()),
        )
        out = {
            "cells": len(part),
            "usableV1": usable_v1,
            "usableV2": usable_v2,
            "unexplainedV1": int(was.sum()),
            "unexplainedV2": int((part["stateV2"] == "unexplained").sum()),
            "v1UnexplainedExplainedUnderV2": int((was & (part["stateV2"] == "explained")).sum()),
            "v1UnexplainedStillUnexplained": int((was & (part["stateV2"] == "unexplained")).sum()),
            "v1UnexplainedWithoutLineageUnderV2": int(
                (was & (part["stateV2"] == "noLineage")).sum()
            ),
            "v1UnexplainedLabelledNewUnderV2": int((was & part["basaloidV2"]).sum()),
            "v1UnexplainedExplainedOrLabelledNew": int(
                (was & ((part["stateV2"] == "explained") | part["basaloidV2"])).sum()
            ),
            "v1ExplainedUnexplainedUnderV2": int(
                ((part["stateV1"] == "explained") & (part["stateV2"] == "unexplained")).sum()
            ),
            "labelledNewUnderV2": int(part["basaloidV2"].sum()),
        }
        out["shareV1"] = out["unexplainedV1"] / usable_v1 if usable_v1 else None
        out["shareV2"] = out["unexplainedV2"] / usable_v2 if usable_v2 else None
        return out

    transitions = frame.groupby(["stateV1", "stateV2"]).size()
    new = frame[frame["basaloidV2"]]
    summary = {
        "all": tally(frame),
        "byGroup": {str(g): tally(p) for g, p in frame.groupby("group")},
        "v1UnexplainedByLineageV1": {
            str(lineage): tally(p)
            for lineage, p in frame[frame["stateV1"] == "unexplained"].groupby("lineageV1")
        },
        "transitions": {f"{a} -> {b}": int(n) for (a, b), n in transitions.items()},
        "labelledNew": {
            "cells": len(new),
            "byGroup": {
                str(g): {"cells": len(p), "donors": int(p["donor"].nunique())}
                for g, p in new.groupby("group")
            },
            "stateV1": {str(k): int(v) for k, v in new["stateV1"].value_counts().items()},
            "labelV1": {str(k): int(v) for k, v in new["labelV1"].value_counts().head(10).items()},
            "authorLabels": {
                str(k): int(v) for k, v in new["authorLabel"].value_counts().head(10).items()
            },
            "lineageV2": {str(k): int(v) for k, v in new["lineageV2"].value_counts().items()},
        },
    }
    if slug == "natri":
        labelled = frame["authorLabel"] == NATRI_BASALOID
        part = frame[labelled]
        summary["authorLabelled"] = {
            "authorLabel": NATRI_BASALOID,
            **tally(part),
            "labelledNewShare": float(part["basaloidV2"].mean()) if len(part) else None,
            "labelledNewAtLevel3Share": float(
                (v2.loc[labelled.to_numpy(), "label_ann_level_3"] == NEW_LABEL).mean()
            )
            if len(part)
            else None,
            "byGroup": {
                str(g): {**tally(p), "labelledNewShare": float(p["basaloidV2"].mean())}
                for g, p in part.groupby("group")
            },
        }
    donor_rows = [
        {"centre": slug, **row}
        for row in frame.groupby(["donor", "group", "stateV1", "stateV2", "basaloidV2"])
        .size()
        .reset_index(name="cells")
        .to_dict(orient="records")
    ]
    lineage_rows = []
    for version, part, flags, usable in (("v1", v1, f1, usable1), ("v2", v2, f2, usable2)):
        sub = pd.DataFrame(
            {
                "donor": part["donor"].astype(str),
                "group": part["group"].astype(str),
                "lineage": part["lineage"],
                "unexplained": flags,
            }
        )[usable]
        grouped = (
            sub.groupby(["donor", "group", "lineage"])
            .agg(cells=("unexplained", "size"), unexplained=("unexplained", "sum"))
            .reset_index()
        )
        grouped["fraction"] = grouped["unexplained"] / grouped["cells"]
        lineage_rows += [
            {"centre": slug, "version": version, **row} for row in grouped.to_dict(orient="records")
        ]
    return (summary, donor_rows, lineage_rows)


def _steps_of(version: str) -> list[str]:
    return [
        f"openReference_{version}",
        f"loadReference_{version}",
        f"map_{version}",
        *(f"transfer_{version}_{level}" for level in mb.LEVELS),
    ]


def _sum(job: dict, names, field: str = "seconds") -> float:
    return float(sum(((job.get("steps", {}).get(name) or {}).get(field) or 0.0 for name in names)))


def _read(job: dict, names, store: str) -> int:
    return int(
        sum(
            (
                ((job.get("steps", {}).get(name) or {}).get("reads") or {})
                .get(store, {})
                .get("readBytes", 0)
                for name in names
            )
        )
    )


def _added(job: dict, names) -> int:
    return int(
        sum(
            (
                (job.get("requests", {}).get(name) or {}).get("storage", {}).get("bytesAdded", 0)
                for name in names
            )
        )
    )


def _centre_costs(slug: str, jobs: dict) -> dict:
    centre, update = (jobs[f"centre_{slug}"], jobs[f"update_{slug}"])
    source = "natri" if CENTRES[slug]["source"] == "natri_ild" else "full"
    requests_v1 = ["map_v1", *(f"transfer_v1_{level}" for level in mb.LEVELS)]
    requests_v2 = ["map_v2", *(f"transfer_v2_{level}" for level in mb.LEVELS)]
    v2_reference = _read(update, _steps_of("v2"), "revised") + (
        _read(update, _steps_of("v2"), "full") if source != "full" else 0
    )
    return {
        "mount": {
            "seconds": _sum(centre, ["mount"]),
            "bytes": (centre.get("mount") or {}).get("bytes"),
            "bytesByPart": (centre.get("mount") or {}).get("bytesByPart"),
            "existed": (centre.get("mount") or {}).get("existed"),
        },
        "phase1": {
            "seconds": _sum(centre, ["select", *_steps_of("v1")]),
            "referenceReadBytes": _read(centre, _steps_of("v1"), "core"),
            "sourceReadBytes": _read(centre, ["select", *_steps_of("v1")], source),
            "bytesAdded": _added(centre, ["select", *requests_v1]),
            "created": sorted(
                (
                    name
                    for name, entry in (centre.get("requests") or {}).items()
                    if entry.get("created")
                )
            ),
            "fromSource": sorted(
                (
                    name
                    for name, entry in (centre.get("requests") or {}).items()
                    if entry.get("heldBy") == "source"
                )
            ),
        },
        "updateV1Requests": {
            "seconds": _sum(update, ["select", *_steps_of("v1")]),
            "referenceReadBytes": _read(update, _steps_of("v1"), "core"),
            "bytesAdded": _added(update, ["select", *requests_v1]),
            "created": sorted(
                (
                    name
                    for name in ["select", *requests_v1]
                    if (update.get("requests", {}).get(name) or {}).get("created")
                )
            ),
        },
        "updateV2": {
            "seconds": _sum(update, _steps_of("v2")),
            "referenceReadBytes": v2_reference,
            "revisedStoreReadBytes": _read(update, _steps_of("v2"), "revised"),
            "fullHlcaReadBytes": _read(update, _steps_of("v2"), "full"),
            "sourceReadBytes": _read(update, _steps_of("v2"), source),
            "bytesAdded": _added(update, requests_v2),
            "createdHere": bool((update.get("measurement") or {}).get("v2CreatedHere")),
        },
        "recovery": {
            "seconds": _sum(update, ["recover"]),
            "referenceReadBytes": (update.get("recovery") or {}).get("referenceReadBytes"),
        },
        "storeBytes": {
            "afterPhase1": (centre.get("storeSize") or {}).get("bytes"),
            "afterUpdate": (update.get("after") or {}).get("bytes"),
        },
        "notFirstAttempt": {
            "phase1": centre.get("previousAttempt") is not None,
            "update": update.get("previousAttempt") is not None,
        },
        "note": "for the full-HLCA centres the full HLCA is both the centre's source and v2's source, so its reads are reported apart"
        if source == "full"
        else "the full HLCA is only v2's source here, so its reads count towards v2",
    }


def _incremental(jobs: dict) -> dict:
    update, rerun = (jobs["update_natri"], jobs["rerun_natri"])
    recipe = ["select", *_steps_of("v1"), *_steps_of("v2")]
    requests = [
        "select",
        "map_v1",
        *(f"transfer_v1_{level}" for level in mb.LEVELS),
        "map_v2",
        *(f"transfer_v2_{level}" for level in mb.LEVELS),
    ]
    v2_requests = ["map_v2", *(f"transfer_v2_{level}" for level in mb.LEVELS)]
    mount_bytes = (rerun.get("mount") or {}).get("bytes") or 0
    incremental = {
        "seconds": _sum(update, recipe),
        "v2Seconds": _sum(update, _steps_of("v2")),
        "bytesAdded": _added(update, requests),
        "v2BytesAdded": _added(update, v2_requests),
    }
    complete = {
        "seconds": _sum(rerun, ["mount", *recipe]),
        "mountSeconds": _sum(rerun, ["mount"]),
        "bytesWritten": (rerun.get("storeSize") or {}).get("bytes"),
        "mountBytes": mount_bytes,
        "resultBytes": _added(rerun, requests),
        "v2OnlySeconds": _sum(rerun, ["mount", "select", *_steps_of("v2")]),
        "v2OnlyBytes": mount_bytes + _added(rerun, ["select", *v2_requests]),
    }

    def ratio(a, b):
        return a / b if a and b else None

    return {
        "incremental": incremental,
        "complete": complete,
        "secondsRatioCompleteToIncremental": ratio(complete["seconds"], incremental["seconds"]),
        "secondsRatioV2OnlyToIncremental": ratio(complete["v2OnlySeconds"], incremental["seconds"]),
        "bytesRatioCompleteToIncremental": ratio(
            complete["bytesWritten"], incremental["bytesAdded"]
        ),
        "bytesRatioV2OnlyToIncremental": ratio(complete["v2OnlyBytes"], incremental["bytesAdded"]),
        "sameResults": {
            version: (rerun.get("comparison") or {}).get(version, {}).get("identical")
            for version in VERSIONS
        },
        "samePayloadFingerprints": all(
            (entry.get("same") for entry in (rerun.get("payloadFingerprints") or {}).values())
        ),
        "clean": {
            "updateCreatedV2": bool((update.get("measurement") or {}).get("v2CreatedHere")),
            "updateNotFirstAttempt": update.get("previousAttempt") is not None,
            "rerunAttempt": rerun.get("attempt"),
        },
    }


def _upload_csv(objects, rows: list[dict], payload: dict, name: str) -> None:
    import pandas as pd

    path = _work(payload) / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_plain(r) for r in rows]).to_csv(path, index=False, float_format="%.6g")
    objects.upload(path, f"{payload['resultsPrefix']}/{name}.csv")


def _stats(payload: dict, record: dict, steps: Steps, objects) -> None:
    reference_screen = _reference_screen()
    prefix = payload["resultsPrefix"]
    work = _work(payload) / "stats"
    names = [
        "reference",
        *(f"centre_{s}" for s in CENTRES),
        *(f"update_{s}" for s in CENTRES),
        "rerun_natri",
    ]
    jobs = {}
    for name in names:
        uri = f"{prefix}/{name}.json"
        jobs[name] = objects.get_json(uri) if objects.exists(uri) else {}
    missing = [name for name, job in jobs.items() if job.get("status") != "ok"]
    if missing:
        raise RuntimeError(f"jobs not complete: {missing}")
    references = jobs["update_natri"]["references"]
    percentile = {
        version: reference_screen.percentile_of(references[version]) for version in VERSIONS
    }

    def load() -> dict:
        frames = {}
        for slug in CENTRES:
            for version in VERSIONS:
                frames[slug, version] = reference_screen.load_cells(
                    _fetch(
                        objects,
                        f"{prefix}/cells_{slug}_{version}.csv.gz",
                        work / f"cells_{slug}_{version}.csv.gz",
                    )
                )
            if not frames[slug, "v1"]["cellId"].equals(frames[slug, "v2"]["cellId"]):
                raise RuntimeError(f"{slug}: the v1 and v2 tables hold other cells")
        return frames

    frames = steps.run("loadTables", load)
    flags, calibrations, threshold_rows = ({}, {}, [])
    for version in VERSIONS:
        result = steps.run(
            f"calibrate_{version}",
            lambda v=version: _calibrate(
                reference_screen, {slug: frames[slug, v] for slug in CENTRES}, percentile[v]
            ),
        )
        for slug, entry in result.items():
            flags[slug, version] = entry["flags"]
            calibrations.setdefault(slug, {})[version] = {
                k: entry[k] for k in ("calibration", "calibrationCentres", "controlDonors")
            }
            threshold_rows += [
                {"centre": slug, "version": version, **row}
                for row in entry["thresholds"].to_dict(orient="records")
            ]
    centres, label_rows, donor_rows, transition_rows, unexplained_rows = ({}, [], [], [], [])
    for slug in CENTRES:
        v1, v2 = (frames[slug, "v1"], frames[slug, "v2"])
        changes, rows = _label_changes(slug, v1, v2)
        label_rows += rows
        proportions, rows = _donor_proportions(slug, v1, v2)
        donor_rows += rows
        moved, transitions, lineages = _unexplained(
            slug, v1, v2, flags[slug, "v1"], flags[slug, "v2"]
        )
        transition_rows += transitions
        unexplained_rows += lineages
        update = jobs[f"update_{slug}"]
        centres[slug] = {
            "name": CENTRES[slug]["name"],
            "cells": len(v1),
            "calibration": calibrations[slug],
            "labelChanges": changes,
            "donorProportions": proportions,
            "unexplained": moved,
            "costs": _centre_costs(slug, jobs),
            "traceability": {
                "recovery": {
                    k: (update.get("recovery") or {}).get(k)
                    for k in (
                        "identical",
                        "sameCells",
                        "tableAsWritten",
                        "referenceReadBytes",
                        "cells",
                    )
                },
                "v1IdentifiersUnchanged": all((update.get("v1Unchanged") or {}).values()),
                "counts": (update.get("traceability") or {}).get("counts"),
                "kinds": (update.get("traceability") or {}).get("kinds"),
                "ownUnchanged": (update.get("traceability") or {}).get("ownUnchanged"),
                "ownBefore": (update.get("traceability") or {}).get("ownBefore"),
                "ownChangedOrRemoved": (update.get("traceability") or {}).get(
                    "ownChangedOrRemoved"
                ),
            },
            "checks": jobs[f"centre_{slug}"].get("checks"),
            "panelGenes": update.get("panelGenes"),
        }
    cost_rows, artifact_rows = ([], [])
    for name, job in jobs.items():
        requests = job.get("requests") or {}
        for step, entry in (job.get("steps") or {}).items():
            reads = entry.get("reads") or {}
            request = requests.get(step) or {}
            cost_rows.append(
                {
                    "job": name,
                    "step": step,
                    "seconds": entry.get("seconds"),
                    **{
                        f"read_{store}": (reads.get(store) or {}).get("readBytes", 0)
                        for store in METERED
                    },
                    "networkReceivedBytes": entry.get("networkReceivedBytes"),
                    "peakRssBytes": entry.get("peakRssBytes"),
                    "bytesAdded": (request.get("storage") or {}).get("bytesAdded"),
                    "created": request.get("created"),
                    "heldBy": request.get("heldBy"),
                    "kind": (request.get("ref") or {}).get("kind"),
                    "artifactId": (request.get("ref") or {}).get("artifact_id"),
                }
            )
    for slug in CENTRES:
        artifact_rows += [
            {"centre": slug, **row}
            for row in (jobs[f"update_{slug}"].get("traceability") or {}).get("artifacts", [])
        ]
    reference = jobs["reference"]
    record.update(
        {
            "design": _design(),
            "reference": {
                "selection": reference.get("selection"),
                "mappingReference": (reference.get("reference") or {}).get("mappingReference"),
                "cellsSelected": (reference.get("reference") or {}).get("cellsSelected"),
                "panelGenes": (reference.get("reference") or {}).get("panelGenes"),
                "runStatus": (reference.get("reference") or {}).get("runStatus"),
                "seconds": {
                    "mount": _sum(reference, ["mount"]),
                    "pipeline": (reference.get("reference") or {}).get("pipelineSeconds"),
                    "build": (reference.get("reference") or {}).get("buildSeconds"),
                },
                "readBytes": {
                    store: _read(reference, list(reference.get("steps") or {}), store)
                    for store in ("core", "full")
                },
                "storage": {
                    "mount": reference.get("mount"),
                    "columns": (reference.get("columns") or {}).get("storage"),
                    "pipeline": reference.get("pipelineStorage"),
                    "build": reference.get("buildStorage"),
                    "store": reference.get("storeSize"),
                },
                "comparison": reference.get("comparison"),
            },
            "references": {
                version: {
                    k: references[version][k]
                    for k in ("mappingReference", "cellsSelected", "panelGenes")
                }
                for version in VERSIONS
            },
            "centres": centres,
            "incrementalVersusComplete": _incremental(jobs),
            "jobs": {
                name: {
                    "status": job.get("status"),
                    "seconds": (job.get("finishedAt") or 0) - (job.get("startedAt") or 0),
                    "previousAttempt": job.get("previousAttempt"),
                }
                for name, job in jobs.items()
            },
        }
    )
    for name, rows in (
        ("label_changes", label_rows),
        ("donors", donor_rows),
        ("transitions", transition_rows),
        ("unexplained", unexplained_rows),
        ("thresholds", threshold_rows),
        ("costs", cost_rows),
        ("artifacts", artifact_rows),
    ):
        steps.run(f"upload_{name}", lambda n=name, r=rows: _upload_csv(objects, r, payload, n))


def _dispatch(payload: dict) -> dict:
    return {
        "setup_test": setup_test,
        "reference": build_reference,
        "centre": map_centre,
        "update": update_centre,
        "rerun": rerun_centre,
        "stats": stats,
    }[payload["job"]](payload)


def worker(payload: dict) -> dict:
    return _dispatch(payload)


def test_worker(payload: dict) -> dict:
    return _dispatch(payload)


def setup_worker(payload: dict) -> dict:
    return _dispatch(payload)


def stats_worker(payload: dict) -> dict:
    return _dispatch(payload)


def build_plan(
    results_root: str,
    datasets_root: str,
    endpoint: str | None,
    *,
    test: bool = False,
    force: bool = False,
) -> dict:
    results_root, datasets_root = (results_root.rstrip("/"), datasets_root.rstrip("/"))
    tag = TEST_TAG if test else RUN_TAG
    prefix = f"{results_root}/results/biocase/{tag}"
    folder = f"{results_root}/{mb.STORES_FOLDER}"
    stem = "revision_test" if test else "revision"
    published = {name: f"{folder}/{name}.zarr" for name in mb.DATASETS}
    sources = dict(published)
    if test:
        sources.update(
            {
                "hlca_full": f"{folder}/{stem}_source_hlca_full.zarr",
                "natri_ild": f"{folder}/{stem}_source_natri_ild.zarr",
            }
        )
    stores = {
        "reference": f"{folder}/{stem}_reference_v2.zarr",
        **{f"centre_{slug}": f"{folder}/{stem}_centre_{slug}.zarr" for slug in CENTRES},
        "rerun": f"{folder}/{stem}_rerun_natri",
    }

    def job(kind: str, function: str, record: str, **extra) -> dict:
        cpu, memory = SIZES[function]
        payload = {
            "job": kind,
            "r2EndpointUrl": endpoint,
            "resultsRoot": results_root,
            "datasetsRoot": datasets_root,
            "runTag": tag,
            "test": test,
            "force": force,
            "workers": cpu,
            "budgetBytes": int(memory * MIB * mb.BUDGET_SHARE),
            "containerCpu": float(cpu),
            "containerMemoryMb": memory,
            "workDir": None,
            "resultsPrefix": prefix,
            "primaryPrefix": f"{results_root}/results/biocase/{mb.RUN_TAG}",
            "published": published,
            "sources": sources,
            "stores": stores,
            "recordUri": f"{prefix}/{record}.json",
        }
        return {"function": function, "payload": {**payload, **extra}}

    big = "test_worker" if test else "worker"
    jobs = {
        "reference": job("reference", big, "reference"),
        **{f"centre_{slug}": job("centre", big, f"centre_{slug}", centre=slug) for slug in CENTRES},
        **{f"update_{slug}": job("update", big, f"update_{slug}", centre=slug) for slug in CENTRES},
        "rerun_natri": job("rerun", big, "rerun_natri"),
        "stats": job("stats", "stats_worker", "summary"),
    }
    phases = [
        ["reference", *(f"centre_{slug}" for slug in CENTRES)],
        ["update_natri"],
        ["rerun_natri"],
        [f"update_{slug}" for slug in FULL_CENTRES],
        ["stats"],
    ]
    if test:
        jobs["setup_test"] = job("setup_test", "setup_worker", "setup_test", force=False)
        phases.insert(0, ["setup_test"])
    return {
        "runTag": tag,
        "test": test,
        "r2EndpointUrl": endpoint,
        "resultsPrefix": prefix,
        "jobs": jobs,
        "phases": phases,
    }


def _jobs_for(action: str, centre: str) -> list[str]:
    if centre and centre not in CENTRES:
        raise SystemExit(f"--centre must be one of {sorted(CENTRES)}")
    chosen = [centre] if centre else list(CENTRES)
    return {
        "setup": ["setup_test"],
        "reference": ["reference"],
        "centres": [f"centre_{slug}" for slug in chosen],
        "updates": [f"update_{slug}" for slug in chosen],
        "rerun": ["rerun_natri"],
        "stats": ["stats"],
    }[action]


ACTIONS = ("test", "all", "setup", "reference", "centres", "updates", "rerun", "stats")
