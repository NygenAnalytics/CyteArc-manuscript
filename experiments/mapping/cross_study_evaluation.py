"""Score cross-study mapping predictions against author labels, with the cost of each run."""

import hashlib
import json
import math
import runpy
from functools import cache
from pathlib import Path

import pandas as pd

from experiments.runtime.common import ROOT, load_json, scrub_private

MAPPING_METHODS = ("unintegrated", "integrated", "celltypist", "seurat_map", "symphony")
EXTERNAL_METHODS = ("celltypist", "celltypist_hvg", "seurat_map", "symphony")


def completed_records(directory: Path) -> list[tuple[Path, dict]]:
    records, seen = [], set()
    for path in sorted(directory.rglob("*.json")):
        envelope = load_json(path)
        if envelope.get("status") != "completed":
            continue
        identity = (envelope["module"], envelope["function"], envelope["startedAt"])
        if identity in seen:
            raise ValueError(f"Duplicate completed call in {path}")
        seen.add(identity)
        records.append((path, envelope))
    return records


def trace(path: Path, envelope: dict) -> dict:
    return {
        "source": str(path.resolve().relative_to(ROOT)),
        "experiment": envelope["module"],
        "function": envelope["function"],
        "region": envelope["region"],
    }


@cache
def prediction_evaluator():
    source = ROOT / "experiments/mapping/cross_study_mapping.py"
    module = runpy.run_path(str(source))
    return module["evaluate_predictions"]


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def record_hash(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(scrub_private(record), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sampled_peaks(container: int, process: int) -> dict:
    values = {}
    for name, value in (("peakContainerBytes", container), ("peakProcessTreeRssBytes", process)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid memory sample: {name}")
        if value > 0:
            values[name] = value
        else:
            values[name + "Unavailable"] = "No positive memory sample was retained"
    return values


def timed_steps(steps: dict, groups: dict[str, tuple[str, ...]]) -> dict:
    names = [name for group in groups.values() for name in group]
    if len(names) != len(set(names)) or set(names) != set(steps):
        raise ValueError(f"Timing groups omit or repeat steps: {sorted(set(steps) ^ set(names))}")
    result = {}
    for group, selected in groups.items():
        seconds = [steps[name]["seconds"] for name in selected]
        if any(not math.isfinite(value) or value < 0 for value in seconds):
            raise ValueError(f"Invalid elapsed time in {group}")
        result[group] = {"steps": list(selected), "seconds": sum(seconds)}
        for metric in ("peakCgroupBytes", "peakRssBytes"):
            if any(
                metric in steps[name]
                and (not math.isfinite(steps[name][metric]) or steps[name][metric] < 0)
                for name in selected
            ):
                raise ValueError(f"Invalid memory sample in {group}")
            observed = [steps[name][metric] for name in selected if steps[name].get(metric, 0) > 0]
            if observed:
                result[group][metric] = max(observed)
            else:
                result[group][metric + "Unavailable"] = (
                    "No positive sample was retained for these steps"
                )
    return result


def native_costs(record: dict, arm: str) -> dict:
    groups = timed_steps(
        record["steps"],
        {
            "referencePreparation": ("convert_reference", f"{arm}_reference"),
            "referenceRegistration": (f"{arm}_registration",),
            "queryAnnotation": (
                "convert_query",
                "snapshot_query_cells",
                f"{arm}_projection_correction_neighbors",
                f"{arm}_label_voting",
            ),
        },
    )
    query = groups["queryAnnotation"]["seconds"]
    access = groups["referenceRegistration"]["seconds"]
    return {
        "groups": groups,
        "completeQueryAnnotationSeconds": query,
        "preparedReferenceAndQuerySeconds": access + query,
        "freshReferenceAndQuerySeconds": sum(value["seconds"] for value in groups.values()),
        "referenceAccessDefinition": "Build and load the mapping-reference descriptor from existing analysis artifacts",
        **sampled_peaks(record["peakContainerBytes"], record["peakProcessTreeRssBytes"]),
        "memoryDefinition": record["memoryDefinition"],
        "timingBoundary": record["timingBoundary"],
    }


def external_costs(record: dict) -> dict:
    method = record["tool"]
    if record["labelColumns"] != ["cell_type"]:
        raise ValueError("Final mapping costs require one shared cell_type label vocabulary")
    if method in {"celltypist", "celltypist_hvg"}:
        variable = method == "celltypist_hvg"
        groups = {
            "referencePreparation": ("readReference", "normalizeReference")
            + (("markHvgs", "subsetReferenceFeatures") if variable else ())
            + ("train_cell_type",),
            "referencePersistence": ("persistReference_cell_type",),
            "preparedReferenceLoad": ("loadPreparedReference_cell_type",),
            "queryAnnotation": ("readQuery", "normalizeQuery")
            + (("subsetQueryFeatures",) if variable else ())
            + ("annotate_cell_type",),
        }
    elif method in {"seurat_map", "symphony"}:
        groups = {
            "referencePreparation": ("readReference", "readReferenceMetadata", "buildReference")
            + (("buildReferenceIndex",) if method == "seurat_map" else ()),
            "referencePersistence": ("persistReference",),
            "preparedReferenceLoad": ("loadPreparedReference",),
            "queryAnnotation": ("readQuery", "readQueryMetadata")
            + (
                ("createQuery", "normalizeQuery", "findAnchors")
                if method == "seurat_map"
                else ("mapQuery",)
            )
            + ("transfer_cell_type",),
        }
    else:
        raise ValueError(f"Unsupported final mapping comparator: {method}")
    costs = timed_steps(record["steps"], groups)
    query = costs["queryAnnotation"]["seconds"]
    load = costs["preparedReferenceLoad"]["seconds"]
    return {
        "groups": costs,
        "completeQueryAnnotationSeconds": query,
        "preparedReferenceAndQuerySeconds": load + query,
        "freshReferenceAndQuerySeconds": sum(value["seconds"] for value in costs.values()),
        "referenceAccessDefinition": "Load the persisted trained reference model and any persisted neighbor index",
        **sampled_peaks(record["peakCgroupBytes"], record["peakRssBytes"]),
        "memoryDefinition": "Sampled container memory.current and process-tree RSS from the same Sampler used for CyteArc",
        "timingBoundary": "Stage wall times from prepared matched H5ADs; downloads, process startup and untimed gaps are excluded",
    }


def read_native_predictions(path: Path, inputs: dict) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={name: str for name in ("cell", "study", "donor_id", "truth", "prediction")},
    )
    required = {"cell", "study", "donor_id", "truth", "prediction", "confidence"}
    if not required <= set(frame) or frame.empty or frame["cell"].duplicated().any():
        raise ValueError(f"Incomplete or duplicated native prediction rows: {path}")
    expected = inputs["parts"]["query"]
    digest = hashlib.sha256("\0".join(frame["cell"]).encode()).hexdigest()
    if len(frame) != expected["cells"] or digest != expected["cellIdSha256"]:
        raise ValueError(f"Native prediction cell IDs do not match the prepared query: {path}")
    donors = set(frame[["study", "donor_id"]].itertuples(index=False, name=None))
    if donors != {(row["study"], row["donor_id"]) for row in expected["donors"]}:
        raise ValueError(f"Native prediction donors do not match the prepared query: {path}")
    if (
        frame["truth"].value_counts().to_dict() != expected["labels"]
        or frame["study"].value_counts().to_dict() != expected["studies"]
    ):
        raise ValueError(
            f"Native prediction labels or studies differ from the prepared query: {path}"
        )
    return frame


def read_external_predictions(path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={name: str for name in ("cell", "true_cell_type", "pred_cell_type")},
    )
    required = {"cell", "true_cell_type", "pred_cell_type", "conf_cell_type"}
    if not required <= set(frame) or frame["cell"].duplicated().any():
        raise ValueError(f"Incomplete or duplicated comparator prediction rows: {path}")
    if set(frame["cell"]) != set(metadata["cell"]):
        raise ValueError(f"Comparator prediction cell IDs differ from the shared query: {path}")
    frame = frame.set_index("cell").loc[metadata["cell"]].reset_index()
    if not frame["true_cell_type"].equals(metadata["truth"].reset_index(drop=True)):
        raise ValueError(f"Comparator truth labels differ from the shared query: {path}")
    output = metadata[["cell", "study", "donor_id", "truth"]].copy().reset_index(drop=True)
    output["prediction"] = frame["pred_cell_type"].replace({"": "unassigned", "NA": "unassigned"})
    output["confidence"] = pd.to_numeric(frame["conf_cell_type"], errors="raise")
    return output


def evaluation_inputs(inputs: dict) -> tuple[list[str], list[str]]:
    frequencies = inputs["parts"]["reference"]["labels"]
    cells = inputs["parts"]["reference"]["cells"]
    labels = sorted(set(frequencies) | set(inputs["parts"]["query"]["labels"]))
    if "unassigned" in labels:
        raise ValueError("The reserved unassigned label occurs in the prepared biological labels")
    return labels + ["unassigned"], [
        label for label, count in frequencies.items() if count >= 20 and count / cells < 0.01
    ]


def matched_coverage(
    rows: list[dict], frames: dict[str, pd.DataFrame], inputs: dict[str, dict]
) -> list[dict]:
    output = []
    for tissue in sorted(inputs):
        available = [row for row in rows if row["tissue"] == tissue and row["source"] in frames]
        if not available:
            continue
        labels, rare = evaluation_inputs(inputs[tissue])
        for study in sorted(inputs[tissue]["parts"]["query"]["studies"]):
            selected = [
                (row, frames[row["source"]].loc[lambda frame: frame.study == study])
                for row in available
            ]
            cells = len(selected[0][1])
            accepted = min(
                int((frame["prediction"] != "unassigned").sum()) for _, frame in selected
            )
            targets = [
                (level, min(accepted, math.ceil(level * cells)), "fixed")
                for level in (0.5, 0.75, 0.9, 0.95)
            ] + [(accepted / cells, accepted, "maximumAvailable")]
            measurements = [[] for _ in targets]
            for row, frame in selected:
                evaluated = prediction_evaluator()(
                    frame,
                    labels,
                    rare,
                    coverage_levels=tuple(size / cells for _, size, _ in targets),
                )["studies"][study]["coverageCurve"]
                for i, (values, (_, size, _)) in enumerate(zip(evaluated, targets, strict=True)):
                    if (
                        not values["requestedCoverageAchieved"]
                        or values["actualCoverage"] != size / cells
                    ):
                        raise ValueError(
                            "Comparator methods did not achieve the same assignment coverage"
                        )
                    measurements[i].append(
                        {
                            "method": row["method"],
                            "replicate": row["replicate"],
                            "source": row["source"],
                            **values,
                        }
                    )
            for (requested, size, endpoint), values in zip(targets, measurements, strict=True):
                output.append(
                    {
                        "tissue": tissue,
                        "study": study,
                        "queryCells": cells,
                        "assignedCells": size,
                        "requestedCoverage": requested,
                        "actualCoverage": size / cells,
                        "commonAvailableCoverage": accepted / cells,
                        "requestedCoverageAvailable": math.ceil(requested * cells - 1e-8)
                        <= accepted,
                        "coverageEndpoint": endpoint,
                        "completeComparison": set(MAPPING_METHODS)
                        <= {row["method"] for row in available},
                        "measurements": values,
                    }
                )
    return output


def mapping_values(
    records: list[tuple[Path, dict]], directory: Path
) -> tuple[list[dict], list[dict], list[str]]:
    rows, missing, frames, tissue_inputs, metadata = [], [], {}, {}, {}
    for path, envelope in records:
        record = envelope["result"]
        if record.get("mode") != "benchmark" or "arms" not in record:
            continue
        if len(record["arms"]) != 1:
            raise ValueError("Each final native mapping job must contain one fresh reference arm")
        inputs = record["inputs"]
        if inputs["referenceQuerySourceCellIdOverlap"] != 0:
            raise ValueError("Final mapping contains shared reference and query cells")
        tissue = record["settings"]["tissue"]
        if tissue in tissue_inputs and tissue_inputs[tissue] != inputs:
            raise ValueError(f"Native mapping arms use different input manifests for {tissue}")
        tissue_inputs[tissue] = inputs
        for arm, value in record["arms"].items():
            if value["registrationIo"]["countReadBytes"] != 0:
                raise ValueError("Registration read reference count chunks")
            row = {
                **trace(path, envelope),
                "tissue": tissue,
                "method": arm,
                "replicate": record["settings"].get("replicate"),
                "settings": scrub_private(record["settings"]),
                "inputs": scrub_private(inputs),
                "registrationIo": value["registrationIo"],
                "queryAnnotationIo": value["queryAnnotationIo"],
                "costs": native_costs(record, arm),
                "steps": record["steps"],
            }
            predictions = path.with_suffix("") / f"{arm}_predictions.csv"
            if predictions.exists():
                frame = read_native_predictions(predictions, inputs)
                identity = frame[["cell", "study", "donor_id", "truth"]]
                if tissue in metadata and not metadata[tissue].equals(identity):
                    raise ValueError(f"Native mapping metadata differs across arms for {tissue}")
                metadata[tissue] = identity
                labels, rare = evaluation_inputs(inputs)
                row["evaluation"] = prediction_evaluator()(frame, labels, rare)
                for study, measured in value["evaluation"]["studies"].items():
                    if any(
                        not math.isclose(
                            row["evaluation"]["studies"][study][key],
                            measured[key],
                            rel_tol=0,
                            abs_tol=1e-12,
                        )
                        for key in ("accuracy", "macroF1", "assignmentCoverage")
                    ):
                        raise ValueError(
                            "Native prediction evaluation differs from its retained experiment record"
                        )
                row["predictionSource"] = str(predictions.resolve().relative_to(ROOT))
                row["predictionsSha256"] = file_hash(predictions)
                frames[row["source"]] = frame
            else:
                missing.append(f"Collected native predictions: {predictions.relative_to(ROOT)}")
            rows.append(row)
    external_calls = set()
    for path in sorted(directory.rglob("*.json")):
        envelope = load_json(path)
        record = envelope["result"] if envelope.get("status") == "completed" else envelope
        if record.get("status") != "ok" or record.get("tool") not in EXTERNAL_METHODS:
            continue
        if "inputFiles" not in record:
            missing.append(f"Verified comparator input hashes: {path.relative_to(ROOT)}")
            continue
        identity = (
            record["tool"],
            record["inputFiles"]["reference"]["sha256"],
            record["inputFiles"]["query"]["sha256"],
            record["replicate"],
        )
        if identity in external_calls:
            raise ValueError(f"Duplicate comparator call in {path}")
        external_calls.add(identity)
        if record["returnCode"] != 0:
            raise ValueError(f"Comparator claims success after a failed process: {path}")
        matches = [
            tissue
            for tissue, inputs in tissue_inputs.items()
            if all(
                record["inputFiles"][role]["sha256"] == inputs["parts"][role]["sha256"]
                for role in ("reference", "query")
            )
        ]
        if not matches:
            missing.append(f"Native matched-input record for comparator {path.relative_to(ROOT)}")
            continue
        if len(matches) != 1:
            raise ValueError(
                f"Comparator inputs do not uniquely match a native tissue input pair: {path}"
            )
        tissue = matches[0]
        for role in ("reference", "query"):
            expected = tissue_inputs[tissue]["parts"][role]
            actual = record["inputFiles"][role]
            if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
                raise ValueError(
                    f"Comparator {role} file does not match the prepared input hash and size: {path}"
                )
        row = {
            "source": str(path.resolve().relative_to(ROOT)),
            "experiment": "experiments.mapping.comparators",
            "method": record["tool"],
            "tissue": tissue,
            "replicate": record["replicate"],
            "container": record["container"],
            "inputFiles": scrub_private(record["inputFiles"]),
            "costs": external_costs(record),
            "steps": record["steps"],
        }
        predictions = path.with_name(path.stem + "_cells.csv")
        if predictions.exists() and tissue in metadata:
            frame = read_external_predictions(predictions, metadata[tissue])
            if len(frame) != record["queryCells"]:
                raise ValueError("Comparator prediction count differs from its measured workflow")
            labels, rare = evaluation_inputs(tissue_inputs[tissue])
            row["evaluation"] = prediction_evaluator()(frame, labels, rare)
            row["predictionSource"] = str(predictions.resolve().relative_to(ROOT))
            row["predictionsSha256"] = file_hash(predictions)
            frames[row["source"]] = frame
        else:
            missing.append(
                f"Comparator predictions and matched native query metadata: {predictions.relative_to(ROOT)}"
            )
        rows.append(row)
    for row in rows:
        if "peakContainerBytes" not in row["costs"]:
            missing.append(f"Measured container peak memory: {row['source']}")
    for tissue in ("blood", "kidney", "lung"):
        for method in MAPPING_METHODS:
            if not any(
                row["tissue"] == tissue and row["method"] == method and "evaluation" in row
                for row in rows
            ):
                missing.append(f"Complete matched mapping measurements for {tissue}/{method}")
    return rows, matched_coverage(rows, frames, tissue_inputs), missing
