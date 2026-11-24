import csv
import gzip
import hashlib
import io
import os
import re
import time
import traceback
from pathlib import Path

from experiments.lung import HERE
from experiments.runtime.harness import Objects, _is_remote
from experiments.runtime.image_records import image_record
from experiments.runtime.lung_atlas import CONTROL_DISEASE, ILD_DISEASES

TOOLS = HERE / "tools"

RUN_TAG = "lung-reference-screen"
STORES_FOLDER = "stores/lung-reference-screen"
MIB = 1024**2
CPU = 16.0
MEMORY_MB = 32 * 1024
BUDGET_SHARE = 0.75
DISK_MB = 524288
DATASETS = {
    "hlca_core": {
        "datasetId": "066943a2-fdac-4b29-b348-40cede398e4e",
        "datasetVersionId": "688185ad-11c2-4172-a53a-f4f1f4076860",
        "cells": 584944,
    },
    "hlca_full": {
        "datasetId": "9f222629-9e39-47d0-b83f-e08d610c7479",
        "datasetVersionId": "5f863718-02a6-46cb-83e7-52be988b915b",
        "cells": 2282447,
    },
    "natri_ild": {
        "datasetId": "f14bc322-1322-4184-8d16-409557525ea5",
        "datasetVersionId": "c3d9262e-0dc5-4eca-bf20-56e6d96d0306",
        "cells": 471905,
    },
}
HVG = 2000
DIMS = 30
NEIGHBOURS = 15
MIN_FEATURES = 10
LEVELS = ("ann_level_1", "ann_level_3", "ann_finest_level")
VOTE_THRESHOLD = 0.6
MISSING_FEATURES = "reference_mean"
NATRI_COLUMNS = {
    "donor": "donor_id",
    "sample": "sample_name",
    "region": "Sample_Type",
    "diagnosis": "Diagnosis",
    "disease": "disease",
    "processingSite": "Processing_site",
    "study": "study",
    "authorLabel": "manual_annotation_1",
}
FULL_COLUMNS = {
    "donor": "donor_id",
    "sample": "sample",
    "study": "study",
    "dataset": "dataset",
    "disease": "disease",
    "lungCondition": "lung_condition",
    "assay": "assay",
    "hlcaLevel1": "ann_level_1",
    "uncertLevel1": "transf_ann_level_1_uncert",
    "uncertLevel3": "transf_ann_level_3_uncert",
    "uncertLevel5": "transf_ann_level_5_uncert",
}
NATRI_LAB_STUDY = "Banovich_Kropski_2020"
REPLICATION_KEY = "biocase_replication"
DONOR_KEY = re.compile("(VU(?:HD|ILD)\\d+|T(?:HD|ILD)\\d+)", re.IGNORECASE)
LEIDEN_RESOLUTION = 1.0
LEIDEN_SEED = 4444
MAX_CELLS_PER_GROUP = 5000
MIN_DETECTED = 0.01
TOP_GENES = 200
SAMPLING_SEED = 26


def _storage_options(uri: str) -> dict | None:
    if not _is_remote(uri):
        return None
    return {
        "access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "endpoint": os.environ["R2_ENDPOINT_URL"].rstrip("/"),
    }


def _open(uri: str, payload: dict, *, mode: str = "r+", initialise: bool = False):
    import cytearc

    options = {
        "default_assay": "RNA",
        "nthreads": int(payload["workers"]),
        "mem_budget": int(payload["budgetBytes"]),
        "zarr_mode": mode,
        "storage_options": _storage_options(uri),
    }
    if initialise:
        options.update({"assay_types": {"RNA": "RNA"}, "min_features_per_cell": MIN_FEATURES})
    return cytearc.DataStore(uri, **options)


def run_pipeline(
    ds,
    *,
    label: str,
    cells: str = "I",
    hvg: dict,
    pca: dict,
    neighbors: dict,
    leiden: dict | bool,
    **switches: bool,
):
    """CyteArc's RNA recipe, DataStore.pipeline.run, with the settings as the lung jobs record
    them: the cell column, the settings of the variable-gene, PCA and neighbour stages, Leiden's
    partitions or False, and the switches of the other stages."""
    params = {"hvg": hvg, "pca": pca, "neighbors": neighbors}
    if isinstance(leiden, dict):
        params["leiden"] = leiden
    return ds.pipeline.run(
        label=label, cell_key=cells, leiden=bool(leiden), params=params, **switches
    )


def _work(payload: dict) -> Path:
    return Path(payload.get("workDir") or "/tmp/biocase")


def _store_exists(objects: Objects, uri: str) -> bool:
    return objects.exists(f"{uri.rstrip('/')}/zarr.json")


def _relative(value, payload: dict):
    if isinstance(value, dict):
        return {k: _relative(v, payload) for k, v in value.items()}
    if isinstance(value, list):
        return [_relative(v, payload) for v in value]
    if isinstance(value, str):
        for key, placeholder in (("resultsRoot", "<results>"), ("datasetsRoot", "<datasets>")):
            root = str(payload.get(key) or "").rstrip("/")
            if root:
                value = value.replace(root, placeholder)
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


def _cgroup_peak() -> int | None:
    for path in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return int(Path(path).read_text().split()[0])
        except OSError, ValueError, IndexError:
            continue
    return None


class Steps:
    """Wall time, network bytes received and peak memory (so far, of the process and of the
    container) of each step. ``save`` runs before and after every step, so a container killed
    for memory leaves the step it was in."""

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
        value = fn()
        after = _network_received()
        self.records[name].update(
            {
                "seconds": time.perf_counter() - started,
                "networkReceivedBytes": None
                if received is None or after is None
                else after - received,
                "peakRssBytes": _peak_rss(),
                "peakCgroupBytes": _cgroup_peak(),
            }
        )
        self.current = None
        self.save()
        return value


def _parameters() -> dict:
    return {
        "hvgCount": HVG,
        "pcaDims": DIMS,
        "neighbours": NEIGHBOURS,
        "minFeaturesPerCell": MIN_FEATURES,
        "filtering": False,
        "batchIntegration": None,
        "levels": list(LEVELS),
        "voteThreshold": VOTE_THRESHOLD,
        "saveK": NEIGHBOURS,
        "missingFeaturePolicy": MISSING_FEATURES,
        "ildDiseases": list(ILD_DISEASES),
        "natriLabStudy": NATRI_LAB_STUDY,
        "leidenResolution": LEIDEN_RESOLUTION,
        "leidenSeed": LEIDEN_SEED,
        "maxCellsPerGroup": MAX_CELLS_PER_GROUP,
        "minDetectedFraction": MIN_DETECTED,
        "topGenes": TOP_GENES,
        "samplingSeed": SAMPLING_SEED,
    }


def _provenance(payload: dict) -> dict:
    import platform
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in (
        "cytearc",
        "numpy",
        "scipy",
        "pandas",
        "zarr",
        "obstore",
        "hnswlib",
        "igraph",
        "leidenalg",
        "h5py",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "script": Path(__file__).name,
        "scriptSha256": _sha256(Path(__file__)),
        "runTag": payload["runTag"],
        "python": platform.python_version(),
        "packages": packages,
        "image": image_record("cytearc_tool_job"),
        "datasets": DATASETS,
        "container": {
            key: payload.get(key)
            for key in ("containerCpu", "containerMemoryMb", "workers", "budgetBytes")
        },
        "parameters": _parameters(),
    }


def _job(payload: dict, step: str, body) -> dict:
    _prepare(payload)
    objects = Objects(payload.get("r2EndpointUrl"))
    uri = payload["recordUri"]
    if not payload.get("force") and objects.exists(uri):
        previous = objects.get_json(uri)
        if previous.get("status") == "ok":
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
            _relative({**record, "steps": steps.records, "currentStep": steps.current}, payload),
        )
    )
    try:
        body(record, steps, objects)
        record["status"] = "ok"
    except BaseException:
        record["status"] = "error"
        record["error"] = traceback.format_exc()[-4000:]
    record["finishedAt"] = time.time()
    record["steps"] = steps.records
    objects.put_json(uri, _relative(record, payload))
    if record["status"] == "error":
        raise RuntimeError(record["error"])
    return {"status": record["status"], "step": step, "error": record.get("error", "")[-600:]}


def convert(payload: dict) -> dict:
    return _job(
        payload,
        f"convert_{payload['name']}",
        lambda record, steps, objects: _convert(payload, record, steps, objects),
    )


def _convert(payload: dict, record: dict, steps: Steps, objects: Objects) -> None:
    import cytearc
    import numpy as np

    name = payload["name"]
    store = payload["stores"][name]
    record.update(
        {
            "dataset": name,
            "source": DATASETS[name],
            "store": store,
            "stagedUri": payload["stagedUri"],
        }
    )
    staged = objects.get_json(payload["stagedRecordUri"])
    if staged.get("datasetVersionId") != DATASETS[name]["datasetVersionId"]:
        raise ValueError(
            f"staged {name} is version {staged.get('datasetVersionId')}, not the pinned one"
        )
    local = _work(payload) / name / f"{name}.h5ad"
    record["sourceBytes"] = steps.run(
        "download", lambda: objects.download(payload["stagedUri"], local)
    )
    record["sourceSha256"] = steps.run("sha256", lambda: _sha256(local))
    if record["sourceSha256"] != staged.get("sourceSha256"):
        raise ValueError(f"the staged {name} file differs from its staging record")

    def write() -> dict:
        reader = cytearc.H5adReader(
            str(local),
            matrix_key="raw/X",
            cell_attrs_key="obs",
            cell_ids_key="_index",
            feature_attrs_key="raw/var",
            feature_ids_key="_index",
            feature_name_key="feature_name",
        )
        try:
            writer = cytearc.H5adToZarr(
                reader,
                store,
                assay_name="RNA",
                storage_options=_storage_options(store),
                mem_budget=int(payload["budgetBytes"]),
                nthreads=int(payload["workers"]),
            )
            imported = writer.dump()
        finally:
            reader.close()
        return {
            "matrixKey": "raw/X",
            "featureTable": "raw/var",
            "cells": int(reader.nCells),
            "features": int(reader.nFeatures),
            "sourceDtype": str(reader.sourceMatrixDtype),
            "storageDtype": str(writer.storageDtypes["RNA"]),
            "cellSelection": imported.cellSelection.to_dict(),
        }

    record["import"] = steps.run("convert", write)
    local.unlink(missing_ok=True)
    ds = steps.run("initialise", lambda: _open(store, payload, initialise=True))
    active = np.asarray(ds.cells.fetch_all("I"), dtype=bool)
    record["contents"] = {
        "cells": int(ds.cells.N),
        "activeCells": int(active.sum()),
        "features": int(ds.RNA.feats.N),
        "cellColumns": sorted((str(c) for c in ds.cells.columns)),
    }
    if payload.get("reference"):
        record["reference"] = _build_reference(ds, steps, objects, store)
    record["storeSize"] = steps.run("listStore", lambda: objects.prefix_bytes(store))


def _build_reference(ds, steps: Steps, objects: Objects, store: str) -> dict:
    import numpy as np
    from cytearc.storage.refs import artifact_path

    run = steps.run(
        "referencePipeline",
        lambda: run_pipeline(
            ds,
            label="reference",
            filtering=False,
            hvg={"top_n": HVG},
            pca={"dims": DIMS},
            neighbors={"k": NEIGHBOURS},
            umap=False,
            leiden=False,
            cell_cycle=False,
            paris=False,
            doublets=False,
            markers=False,
        ),
    )
    ref = steps.run("buildReference", lambda: ds.mapping.build_reference(run["neighbors"]))
    reference = ds.mapping.load_reference(ref)
    rows = np.flatnonzero(
        np.asarray(ds.artifacts.load(reference.cell_selection)["values"][:], dtype=bool)
    )
    labels = {
        level: int(np.unique(np.asarray(ds.cells.fetch_all(level)).astype(str)[rows]).size)
        for level in LEVELS
    }
    return {
        "pipelineRunId": run.run_id,
        "neighbors": run["neighbors"].to_dict(),
        "mappingReference": ref.to_dict(),
        "cellsSelected": int(reference.selected_cell_count),
        "panelGenes": int(reference.feature_ids.size),
        "dims": int(reference.model.loadings.shape[1]),
        "method": reference.method,
        "labelsPerLevel": labels,
        "artifactSizes": {
            "mappingReference": objects.prefix_bytes(f"{store}/{artifact_path(ref)}"),
            "annIndex": objects.prefix_bytes(f"{store}/{artifact_path(reference.ann_index)}"),
            "reduction": objects.prefix_bytes(f"{store}/{artifact_path(reference.reduction)}"),
            "neighbors": objects.prefix_bytes(f"{store}/{artifact_path(reference.neighbors)}"),
        },
    }


def map_queries(payload: dict) -> dict:
    return _job(
        payload, "map", lambda record, steps, objects: _map(payload, record, steps, objects)
    )


def _converted(payload: dict, objects: Objects) -> dict:
    records = {
        name: objects.get_json(f"{payload['resultsPrefix']}/convert_{name}.json")
        for name in DATASETS
    }
    missing = [name for name, r in records.items() if r.get("status") != "ok"]
    if missing:
        raise RuntimeError(f"conversions not complete: {missing}")
    return records


def _map(payload: dict, record: dict, steps: Steps, objects: Objects) -> None:
    import numpy as np
    from cytearc import ArtifactRef

    converted = _converted(payload, objects)
    stores = payload["stores"]
    ds_ref = steps.run("openReference", lambda: _open(stores["hlca_core"], payload, mode="r"))
    ref = ArtifactRef.from_dict(converted["hlca_core"]["reference"]["mappingReference"])
    reference = steps.run("loadReference", lambda: ds_ref.mapping.load_reference(ref))
    nearest = _reference_cells(ds_ref, reference)
    core_keys = _keys(ds_ref.cells.fetch_all("donor_id"))
    ds_full = steps.run("openFull", lambda: _open(stores["hlca_full"], payload, mode="r"))
    full_keys = _keys(ds_full.cells.fetch_all("donor_id"))
    record["reference"] = {
        "store": stores["hlca_core"],
        "mappingReference": ref.to_dict(),
        "cellsSelected": int(reference.selected_cell_count),
        "panelGenes": int(reference.feature_ids.size),
        "distanceQuantiles": np.asarray(
            reference.reference_distance_quantiles, dtype=float
        ).tolist(),
        "distanceValues": np.asarray(reference.reference_distance_values, dtype=float).tolist(),
        "coreDonorKeys": len(core_keys),
        "fullDonorKeys": len(full_keys),
    }
    ds_q = steps.run("openQuery", lambda: _open(stores["natri_ild"], payload))
    natri = _map_cells(ds_q, "I", reference, steps, "natri")
    table = steps.run(
        "natriTable", lambda: _cell_table(ds_q, natri, reference, NATRI_COLUMNS, nearest)
    )
    keys = table["donor"].map(_donor_key)
    table.insert(2, "donorKey", keys)
    table.insert(
        3,
        "donorClass",
        np.where(
            keys.isin(core_keys), "core", np.where(keys.isin(full_keys), "extension", "neither")
        ),
    )
    table.insert(4, "site", np.where(keys.fillna("").str.startswith("VU"), "Vanderbilt", "TGen"))
    record["natri"] = _summary(ds_q, natri, reference, table, ["donorClass", "region"])
    steps.run("natriUpload", lambda: _upload_table(objects, table, payload, "natri_cells"))
    ds_m = steps.run("mountFull", lambda: _mount(objects, payload))
    mask, selection = _replication_cells(ds_m)
    ds_m.cells.insert(REPLICATION_KEY, mask, overwrite=True)
    replication = _map_cells(ds_m, REPLICATION_KEY, reference, steps, "replication")
    table = steps.run(
        "replicationTable",
        lambda: _cell_table(
            ds_m, replication, reference, FULL_COLUMNS, nearest, with_nearest_cell=False
        ),
    )
    table.insert(2, "donorKey", table["donor"].map(_donor_key))
    record["replication"] = {
        "mountStore": payload["mountStore"],
        "rule": f"extension cells of studies with an ILD diagnosis among extension cells, other than {NATRI_LAB_STUDY}, with disease in the ILD diagnoses or '{CONTROL_DISEASE}'; and the {NATRI_LAB_STUDY} extension cells; all within the store's active cells",
        "selection": selection,
        **_summary(ds_m, replication, reference, table, ["study", "disease"]),
    }
    steps.run(
        "replicationUpload", lambda: _upload_table(objects, table, payload, "replication_cells")
    )


def _keys(donors) -> set[str]:
    import numpy as np

    return {k for k in (_donor_key(d) for d in np.unique(np.asarray(donors).astype(str))) if k}


def _reference_cells(ds_ref, reference) -> dict:
    import numpy as np

    rows = np.flatnonzero(
        np.asarray(ds_ref.artifacts.load(reference.cell_selection)["values"][:], dtype=bool)
    )
    column = {
        name: np.asarray(ds_ref.cells.fetch_all(name)).astype(str)[rows]
        for name in ("ids", "donor_id", "dataset", "ann_finest_level")
    }
    return {
        "nearestReferenceCell": column["ids"],
        "nearestReferenceDonorKey": np.array(
            [_donor_key(d) or "" for d in column["donor_id"]], dtype=object
        ),
        "nearestReferenceDataset": column["dataset"],
        "nearestReferenceLabel": column["ann_finest_level"],
    }


def _map_cells(ds, cell_key: str, reference, steps: Steps, name: str) -> dict:
    selection = ds.snapshot_cell_selection(cell_key)
    mapping = steps.run(
        f"{name}Mapping",
        lambda: ds.mapping.run(
            reference,
            selection,
            query_assay="RNA",
            save_k=NEIGHBOURS,
            missing_feature_policy=MISSING_FEATURES,
        ),
    )
    return {"selection": selection, "mapping": mapping}


def _cell_table(
    ds, mapped: dict, reference, columns: dict, nearest: dict, *, with_nearest_cell: bool = True
):
    import numpy as np
    import pandas as pd

    mapping = mapped["mapping"]
    rows = np.flatnonzero(
        np.asarray(ds.artifacts.load(mapped["selection"])["values"][:], dtype=bool)
    )
    table = pd.DataFrame({"cellId": np.asarray(ds.cells.fetch_all("ids")).astype(str)[rows]})
    for name, column in columns.items():
        table[name] = np.asarray(ds.cells.fetch_all(column))[rows]
    positions = np.arange(rows.size)
    agreement = {}
    for level in LEVELS:
        transfer = ds.mapping.load_label_transfer(
            ds.mapping.transfer_labels(
                mapping,
                reference=reference,
                reference_labels=level,
                threshold_fraction=VOTE_THRESHOLD,
            )
        )
        labels = transfer.labels
        evidence = transfer.evidence
        table[f"label_{level}"] = labels.reindex(positions).fillna("NA").astype(str).to_numpy()
        table[f"vote_{level}"] = evidence["voteFraction"].to_numpy()
        agreement[level] = bool(
            np.array_equal(
                table[f"label_{level}"].to_numpy(),
                evidence["label"].fillna("NA").astype(str).to_numpy(),
            )
        )
        if level == LEVELS[0]:
            table["distancePercentile"] = evidence["referenceDistancePercentile"].to_numpy()
    result = ds.mapping.load_result(mapping, reference=reference, load_arrays=True)
    first = np.asarray(result.indices[:, 0], dtype=np.int64)
    table["nearestDistance"] = np.asarray(result.distances[:, 0], dtype=np.float64)
    table["uninformative"] = np.asarray(result.uninformative, dtype=bool).astype(int)
    for name, values in nearest.items():
        if with_nearest_cell or name != "nearestReferenceCell":
            table[name] = values[first]
    table.attrs["labelsAgreeWithEvidence"] = agreement
    table.attrs["diagnostics"] = {
        k: v if isinstance(v, int | float | str) else str(v)
        for k, v in dict(result.diagnostics).items()
    }
    return table


def _summary(ds, mapped: dict, reference, table, groups: list[str]) -> dict:
    counts = table.groupby(groups, dropna=False).size()
    return {
        "cellSelection": mapped["selection"].to_dict(),
        "mapping": mapped["mapping"].to_dict(),
        "cellsMapped": int(len(table)),
        "cellsInStore": int(ds.cells.N),
        "uninformativeCells": int(table["uninformative"].sum()),
        "diagnostics": table.attrs.get("diagnostics"),
        "labelsAgreeWithEvidence": table.attrs.get("labelsAgreeWithEvidence"),
        "cellsBy": [
            {**dict(zip(groups, key if isinstance(key, tuple) else (key,))), "cells": int(n)}
            for key, n in counts.items()
        ],
        "abstainedFraction": {
            level: float((table[f"label_{level}"] == "NA").mean()) for level in LEVELS
        },
    }


def _upload_table(objects: Objects, table, payload: dict, name: str) -> None:
    path = _work(payload) / "tables" / f"{name}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, compression="gzip", float_format="%.6g")
    objects.upload(path, f"{payload['resultsPrefix']}/{name}.csv.gz")


def _mount(objects: Objects, payload: dict):
    import cytearc

    target = payload["mountStore"]
    options = {
        "default_assay": "RNA",
        "nthreads": int(payload["workers"]),
        "mem_budget": int(payload["budgetBytes"]),
        "storage_options": _storage_options(target),
    }
    if _store_exists(objects, target):
        return cytearc.DataStore(target, zarr_mode="r+", **options)
    return cytearc.mount_datastore(payload["stores"]["hlca_full"], at=target, **options)


def _replication_cells(ds) -> tuple:
    import numpy as np
    import pandas as pd

    columns = ("core_or_extension", "study", "disease", "donor_id")
    frame = pd.DataFrame(
        {c: pd.Categorical(np.asarray(ds.cells.fetch_all(c)).astype(object)) for c in columns}
    )
    extension = frame["core_or_extension"].to_numpy() == "extension"
    ild = frame["disease"].isin(ILD_DISEASES).to_numpy()
    studies = sorted(
        {str(study) for study in frame.loc[extension & ild, "study"]} - {NATRI_LAB_STUDY}
    )
    other_labs = frame["study"].isin(studies).to_numpy() & (
        ild | (frame["disease"].to_numpy() == CONTROL_DISEASE)
    )
    natri_lab = frame["study"].to_numpy() == NATRI_LAB_STUDY
    mask = extension & (other_labs | natri_lab) & np.asarray(ds.cells.fetch_all("I"), dtype=bool)
    chosen = (
        frame[mask]
        .groupby(["study", "disease"], observed=True)
        .agg(cells=("donor_id", "size"), donors=("donor_id", "nunique"))
        .reset_index()
    )
    return (
        mask,
        {"replicationStudies": studies, "byStudyAndDisease": chosen.to_dict(orient="records")},
    )


def characterise(payload: dict) -> dict:
    _prepare(payload)
    objects = Objects(payload.get("r2EndpointUrl"))
    if objects.exists(payload["recordUri"]) and objects.get_json(payload["recordUri"]).get(
        "cellsSha256"
    ) != _cells_digest(payload["cells"]):
        payload = {**payload, "force": True}
    return _job(
        payload,
        "characterise",
        lambda record, steps, objects: _characterise(payload, record, steps, objects),
    )


def _cells_digest(cells) -> str:
    return hashlib.sha256("\n".join(sorted(set(cells))).encode()).hexdigest()


def _characterise(payload: dict, record: dict, steps: Steps, objects: Objects) -> None:
    import numpy as np
    import pandas as pd
    from cytearc import ArtifactRef

    mapped = objects.get_json(f"{payload['resultsPrefix']}/map.json")
    if mapped.get("status") != "ok":
        raise RuntimeError("the mapping has not completed")
    stores = payload["stores"]
    cells = sorted(set(payload["cells"]))
    digest = _cells_digest(cells)
    key = f"biocase_unexplained_{digest[:12]}"
    record.update(
        {
            "cellsRequested": len(cells),
            "cellsSha256": digest,
            "cellsFile": payload.get("cellsFile"),
            "cellKey": key,
        }
    )
    ds_ref = steps.run("openReference", lambda: _open(stores["hlca_core"], payload, mode="r"))
    reference = ds_ref.mapping.load_reference(
        ArtifactRef.from_dict(mapped["reference"]["mappingReference"])
    )
    ds_q = steps.run("openQuery", lambda: _open(stores["natri_ild"], payload))
    ids = np.asarray(ds_q.cells.fetch_all("ids")).astype(str)
    mask = np.isin(ids, cells)
    record["cellsFound"] = int(mask.sum())
    ds_q.cells.insert(key, mask, overwrite=True)
    parameters = payload.get("clusterParameters") or {
        "hvg": HVG,
        "dims": DIMS,
        "neighbours": NEIGHBOURS,
    }
    run = steps.run(
        "cluster",
        lambda: run_pipeline(
            ds_q,
            label=key,
            cells=key,
            filtering=False,
            hvg={"top_n": int(parameters["hvg"])},
            pca={"dims": int(parameters["dims"])},
            neighbors={"k": int(parameters["neighbours"])},
            umap=False,
            cell_cycle=False,
            paris=False,
            doublets=False,
            markers=False,
            leiden={
                "partitions": [LEIDEN_RESOLUTION],
                "selected": LEIDEN_RESOLUTION,
                "random_seed": LEIDEN_SEED,
            },
        ),
    )
    rows = np.flatnonzero(mask)
    if not np.array_equal(np.asarray(run.cells.fetch("ids")).astype(str), ids[rows]):
        raise RuntimeError("the clustered cells are not the unexplained cells")
    clusters = np.asarray(run.cells.fetch("clusters")).astype(str)
    record["clustering"] = {
        "pipelineRunId": run.run_id,
        "parameters": parameters,
        "outputs": {k: run[k].to_dict() for k in run},
    }
    _upload_table(
        objects,
        pd.DataFrame({"cellId": ids[rows], "cluster": clusters}),
        payload,
        "unexplained_clusters",
    )
    result = ds_q.mapping.load_result(
        ArtifactRef.from_dict(mapped["natri"]["mapping"]), reference=reference, load_arrays=True
    )
    mapped_rows = np.flatnonzero(
        np.asarray(ds_q.artifacts.load(result.cell_selection)["values"][:], dtype=bool)
    )
    reference_rows = np.flatnonzero(
        np.asarray(ds_ref.artifacts.load(reference.cell_selection)["values"][:], dtype=bool)
    )
    positions = np.clip(np.searchsorted(mapped_rows, rows), 0, mapped_rows.size - 1)
    if not np.array_equal(mapped_rows[positions], rows):
        raise RuntimeError("some unexplained cells were not mapped")
    nearest = reference_rows[np.asarray(result.indices[:, 0], dtype=np.int64)[positions]]
    reference_labels = np.asarray(ds_ref.cells.fetch_all("ann_finest_level")).astype(str)
    genes = _shared_genes(ds_q, ds_ref)
    record["sharedGenes"] = int(genes["ids"].size)
    rng = np.random.default_rng(SAMPLING_SEED)
    groups = []
    for cluster in sorted(set(clusters), key=lambda c: (len(c), c)):
        members = rows[clusters == cluster]
        counterparts = np.unique(nearest[clusters == cluster])
        chosen_q = np.sort(
            rng.choice(members, min(members.size, MAX_CELLS_PER_GROUP), replace=False)
        )
        chosen_r = np.sort(
            rng.choice(counterparts, min(counterparts.size, MAX_CELLS_PER_GROUP), replace=False)
        )
        groups.append((cluster, members, counterparts, chosen_q, chosen_r))
    all_q = np.unique(np.concatenate([g[3] for g in groups]))
    all_r = np.unique(np.concatenate([g[4] for g in groups]))
    query_counts = steps.run("readQueryCounts", lambda: _counts(ds_q, all_q, genes["query"]))
    reference_counts = steps.run(
        "readReferenceCounts", lambda: _counts(ds_ref, all_r, genes["reference"])
    )
    record["cellsRead"] = {"query": int(all_q.size), "reference": int(all_r.size)}
    tables, summaries = ([], [])
    for cluster, members, counterparts, chosen_q, chosen_r in groups:
        stats = steps.run(
            f"wilcoxon_{cluster}",
            lambda: _rank_sum(
                query_counts[np.searchsorted(all_q, chosen_q)],
                reference_counts[np.searchsorted(all_r, chosen_r)],
                genes,
            ),
        )
        top = _top_genes(stats)
        top.insert(0, "cluster", cluster)
        tables.append(top)
        labels = pd.Series(reference_labels[nearest[clusters == cluster]]).value_counts(
            normalize=True
        )
        summaries.append(
            {
                "cluster": cluster,
                "cells": int(members.size),
                "cellsTested": int(chosen_q.size),
                "nearestReferenceCells": int(counterparts.size),
                "nearestReferenceCellsTested": int(chosen_r.size),
                "genesTested": int(len(stats)),
                "significantUp": int(((stats["qValue"] < 0.05) & (stats["auroc"] > 0.5)).sum()),
                "significantDown": int(((stats["qValue"] < 0.05) & (stats["auroc"] < 0.5)).sum()),
                "nearestReferenceLabels": {k: float(v) for k, v in labels.head(5).items()},
            }
        )
    record["clusters"] = summaries
    markers = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    _upload_table(objects, markers, payload, "unexplained_markers")


def _shared_genes(ds_q, ds_ref) -> dict:
    import numpy as np

    query_ids = np.asarray(ds_q.RNA.feats.fetch_all("ids")).astype(str)
    reference_ids = np.asarray(ds_ref.RNA.feats.fetch_all("ids")).astype(str)
    reference_names = np.asarray(ds_ref.RNA.feats.fetch_all("names")).astype(str)
    position = {g: i for i, g in enumerate(query_ids)}
    shared = np.array([i for i, g in enumerate(reference_ids) if g in position], dtype=np.int64)
    return {
        "ids": reference_ids[shared],
        "names": reference_names[shared],
        "reference": shared,
        "query": np.array([position[g] for g in reference_ids[shared]], dtype=np.int64),
    }


def _counts(ds, rows, features):
    from cytearc.assay.base import raw_csr

    return raw_csr(ds.RNA, rows, features)


def _rank_sum(query, reference, genes: dict):
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu

    def per_10k(counts):
        totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
        totals[totals == 0] = 1.0
        return counts.astype(np.float64).multiply(10000.0 / totals[:, None]).tocsc()

    q, r = (per_10k(query), per_10k(reference))
    detected_q = np.asarray((query > 0).sum(axis=0)).ravel() / max(query.shape[0], 1)
    detected_r = np.asarray((reference > 0).sum(axis=0)).ravel() / max(reference.shape[0], 1)
    tested = np.flatnonzero((detected_q >= MIN_DETECTED) | (detected_r >= MIN_DETECTED))
    n_q, n_r = (query.shape[0], reference.shape[0])
    u, p = (np.full(tested.size, np.nan), np.full(tested.size, np.nan))
    for start in range(0, tested.size, 1000):
        block = tested[start : start + 1000]
        outcome = mannwhitneyu(
            np.log1p(q[:, block].toarray()),
            np.log1p(r[:, block].toarray()),
            axis=0,
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        )
        u[start : start + block.size] = outcome.statistic
        p[start : start + block.size] = outcome.pvalue
    mean_q = np.asarray(q[:, tested].mean(axis=0)).ravel()
    mean_r = np.asarray(r[:, tested].mean(axis=0)).ravel()
    p = np.where(np.isfinite(p), p, 1.0)
    return pd.DataFrame(
        {
            "featureId": genes["ids"][tested],
            "featureName": genes["names"][tested],
            "auroc": u / (n_q * n_r),
            "log2FoldChange": np.log2((mean_q + 1.0) / (mean_r + 1.0)),
            "fractionQuery": detected_q[tested],
            "fractionReference": detected_r[tested],
            "pValue": p,
            "qValue": _benjamini_hochberg(p),
            "nQuery": n_q,
            "nReference": n_r,
        }
    )


def _benjamini_hochberg(p):
    import numpy as np

    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(q, 1.0)
    return out


def _top_genes(stats):
    import pandas as pd

    significant = stats[stats["qValue"] < 0.05]
    up = (
        significant[significant["auroc"] > 0.5]
        .sort_values("auroc", ascending=False)
        .head(TOP_GENES)
    )
    down = significant[significant["auroc"] < 0.5].sort_values("auroc").head(TOP_GENES)
    return pd.concat([up.assign(direction="up"), down.assign(direction="down")], ignore_index=True)


def convert_job(payload: dict) -> dict:
    return convert(payload)


def map_job(payload: dict) -> dict:
    return map_queries(payload)


def characterise_job(payload: dict) -> dict:
    return characterise(payload)


def build_plan(
    results_root: str,
    datasets_root: str,
    endpoint: str | None,
    *,
    workers: int,
    budget_bytes: int,
    cpu: float = CPU,
    memory_mb: int = MEMORY_MB,
    force: bool = False,
    work_dir: str | None = None,
) -> dict:
    results_root = results_root.rstrip("/")
    datasets_root = datasets_root.rstrip("/")
    prefix = f"{results_root}/results/biocase/{RUN_TAG}"
    stores = f"{results_root}/{STORES_FOLDER}"
    common = {
        "r2EndpointUrl": endpoint,
        "resultsRoot": results_root,
        "datasetsRoot": datasets_root,
        "runTag": RUN_TAG,
        "workers": workers,
        "budgetBytes": budget_bytes,
        "containerCpu": cpu,
        "containerMemoryMb": memory_mb,
        "force": force,
        "workDir": work_dir,
        "resultsPrefix": prefix,
        "stores": {name: f"{stores}/{name}.zarr" for name in DATASETS},
        "mountStore": f"{stores}/hlca_full_mapping.zarr",
    }
    convert = {
        name: {
            **common,
            "name": name,
            "reference": name == "hlca_core",
            "stagedUri": f"{datasets_root}/biocase/{name}.h5ad",
            "stagedRecordUri": f"{datasets_root}/biocase/{name}.json",
            "recordUri": f"{prefix}/convert_{name}.json",
        }
        for name in DATASETS
    }
    return {
        "convert": convert,
        "map": {**common, "recordUri": f"{prefix}/map.json"},
        "characterise": {**common, "recordUri": f"{prefix}/characterise.json"},
    }


def read_cells(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode() if path.suffix == ".gz" else raw.decode()
    return (
        [row["cellId"] for row in csv.DictReader(io.StringIO(text))],
        hashlib.sha256(raw).hexdigest(),
    )
