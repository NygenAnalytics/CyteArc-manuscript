"""summarise the external-tool records collected by collect_tool_runs.py."""

import csv
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from experiments.benchmarks.timing import (
    analysis_time_seconds,
    preparation_seconds,
    workflow_time_seconds,
)
from experiments.runtime.common import DERIVED, ROOT

DATA = ROOT / "experiments/benchmarks/records"
BENCHMARKS_DERIVED = DERIVED / "benchmarks"
RAW = DATA / "raw"


GIB = 1024**3
LEVELS = ("ann_level_3", "cell_type", "ann_finest_level")
THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Steps that bring the input into the tool's own representation: reading it into memory, the
# lazy open, the Census query, and for the R arms the H5AD shards written from Tahoe-100M's Parquet
# files, the uncompressed copy and BPCells' two on-disk matrices. CyteArc's counterpart is the
# conversion to its store.
LOADING_STEPS = {
    "read",
    "readLazy",
    "queryObs",
    "convertParquet",
    "uncompressInput",
    "convertMatrix",
    "transposeMatrix",
}
# Steps of the mapping workflows: reading both files, work that depends on the query, and
# everything else, which prepares the reference (normalisation, variable genes, training).
READ_STEPS = {"readReference", "readQuery"}
QUERY_PREFIXES = ("annotate_", "transfer_", "mapQuery", "findAnchors")
# Arms added after review, whose rows carry the fields below, and the steps that prepare their
# marker input. The marker-only tools go to markersCommon instead of workflow.
REVIEW_ARMS = {
    "scanpy_csc",
    "scanpy_csc_pcainit",
    "scanpy_dask_covariance_csr",
    "scanpy_dask_covariance_csc",
    "cytearc_matched",
}
MARKER_INPUT_STEPS = {"convertCsc", "materializeMarkerInput"}
REVIEW_FIELDS = (
    "genesAfterFilter",
    "markerFormat",
    "umapInit",
    "pcaRoute",
    "filterBounds",
    "markerCheck",
    "markerInputEqual",
)
MARKER_TOOL_PREFIX = "markers_"
# The marker-only steps that are not the input preparation.
MARKER_TEST_STEP, MARKER_SAMPLE_STEP = "findMarkers", "sampleStatistics"
# Sampled statistics are compared against presto where it ran, else against CyteArc; P values
# only between the tools whose tests have tie and continuity corrections.
SAME_TEST_REFERENCES = ("markers_presto", "markers_cytearc")
CORRECTED_P_TOOLS = ("markers_cytearc", "markers_bpcells", "markers_presto")
LOG10_P_FLOOR = 1e-300
WORKFLOW_RUNS = ROOT / "experiments/benchmarks/records/rna_workflow_runs.json"
# Toolkit runs added after the comparison was collected: each file holds the records that collect_tool_runs writes
# for one run tag. The original runner names Scanpy on Dask with covariance PCA and CSC marker input
# scanpy_dask2_csc (mode dask2_csc), the mode the comparison's records name scanpy_dask_covariance_csc.
ADDED_TOOLKIT_RUNS = (
    "toolkit_comparison_brain_dask_runs.json",
    "toolkit_comparison_replicate_runs.json",
)
COMPARISON_TOOL_NAMES = {"scanpy_dask2_csc": "scanpy_dask_covariance_csc"}


def comparison_name(name: str, record: dict) -> tuple[str, dict]:
    """An added run's record under the tool name the comparison uses: its name, tool, own ID and ladder links."""
    old = record.get("tool")
    new = COMPARISON_TOOL_NAMES.get(old)
    if new is None:
        return name, record

    def renamed(text: str) -> str:
        return text.replace(f"_{old}_", f"_{new}_")

    record = {**record, "tool": new}
    if "sourceRecordId" in record:
        record["sourceRecordId"] = renamed(record["sourceRecordId"])
    if "attempts" in record:
        record["attempts"] = [
            {**attempt, "recordId": renamed(attempt["recordId"])}
            if "recordId" in attempt
            else attempt
            for attempt in record["attempts"]
        ]
    if (record.get("payload") or {}).get("tool") == old:
        record["payload"] = {**record["payload"], "tool": new}
    return renamed(name), record


def toolkit_records(document: dict) -> dict:
    """The comparison's records with the added runs, none of which may repeat a measurement."""
    records = dict(document["records"])
    for file_name in ADDED_TOOLKIT_RUNS:
        for name, record in json.loads((DATA / file_name).read_text())["records"].items():
            name, record = comparison_name(name, record)
            if name in records:
                raise ValueError(f"{file_name} repeats the measurement {name}")
            records[name] = record
    return records


def verified_workflow_records(historical: dict) -> tuple[dict, dict]:
    document = json.loads(WORKFLOW_RUNS.read_text())
    expected = {}
    for name, record in historical.items():
        if name.endswith("_memory_tiers") or record.get("status") != "ok":
            continue
        if record.get("tool") not in {
            "bpcells_cached",
            "scanpy",
            "seurat",
            "seurat_bpcells",
            "scanpy_dask_covariance_csc",
        }:
            continue
        values = tuple(record.get(key) for key in ("nCells", "nGenes", "nnz"))
        if all(value is not None for value in values):
            expected.setdefault(record["dataset"], set()).add(values)
    if any(len(values) != 1 for values in expected.values()):
        raise ValueError("Retained comparators disagree on raw input dimensions or nonzero counts")
    chosen, checks = {}, []
    for name, record in document["records"].items():
        dataset = record["dataset"]
        reference = next(iter(expected[dataset])) if dataset in expected else None
        observed = tuple(record.get(key) for key in ("nCells", "nGenes", "nnz"))
        prepared = document["inputs"].get(dataset)
        reasons = []
        if record.get("mode") not in {"standard", "matched"}:
            reasons.append("The record does not measure the complete workflow")
        if reference is None or observed != reference:
            reasons.append("Raw input dimensions/nonzero count differ from the retained comparison")
        if prepared is None:
            reasons.append("No input preparation record for this dataset")
        elif dataset.startswith("embryo_"):
            if not all(
                isinstance(prepared.get(key), str) and len(prepared[key]) == 64
                for key in ("sha256", "sourceRowsSha256")
            ):
                reasons.append("The sampled embryo input lacks file and ordered-row hashes")
            if (
                prepared.get("targetRows"),
                prepared.get("nColumns"),
                prepared.get("nnz"),
            ) != observed:
                reasons.append("The sampled input preparation does not match this runtime record")
        elif (prepared.get("nCells"), prepared.get("nGenes"), prepared.get("nnz")) != observed:
            reasons.append("The full-input preparation does not match this runtime record")
        check = {
            "record": name,
            "dataset": dataset,
            "tool": record["tool"],
            "replicate": record["replicate"],
            "eligible": not reasons,
            "reasons": reasons,
            "expectedDimensionsAndNnz": reference,
            "observedDimensionsAndNnz": observed,
            "inputFileSha256": (prepared or {}).get("sha256"),
            "sourceRowsSha256": (prepared or {}).get("sourceRowsSha256"),
            "sourceSha256": (prepared or {}).get("sourceSha256"),
        }
        checks.append(check)
        if not reasons:
            chosen[f"rna_workflows/{name}"] = record
    return chosen, {
        "records": checks,
        "rawRecords": str(WORKFLOW_RUNS.relative_to(ROOT)),
    }


def require_verified_workflows(document: dict | None = None) -> None:
    document = document or json.loads((BENCHMARKS_DERIVED / "toolkit_comparisons.json").read_text())
    datasets = (
        "embryo_1000000",
        "brain_1306127",
        "tabula_sapiens",
        "hbca_neurons",
        "mssm_cohort",
        "embryo_10000000",
    )
    missing = []
    for dataset in datasets:
        rows = [
            row
            for row in document["workflow"]
            if row["dataset"] == dataset
            and row["tool"] == "cytearc"
            and row["sourceRecord"].startswith("rna_workflows/")
            and row.get("analysisTimeMinutes") is not None
        ]
        if len(rows) != 3 or {row["replicate"] for row in rows} != {1, 2, 3}:
            missing.append(dataset)
    if missing:
        raise ValueError("Current comparison inputs/repeats are incomplete: " + ", ".join(missing))


# Attempts that the first version of the harness and ladder closed as killed for memory,
# although the container ended far below its limit while the same workflow completed at the
# same tier elsewhere: most likely preemptions. They count as interrupted, not as failures of
# the tier, and each tier was run again (replicate 4 of Scanpy; replicate 2 of Census).
REPORTED_PREEMPTIONS = {
    "rna_workflow_embryo_1000000_scanpy_cpus_8_replicate_2_memory_64gib": (
        "ended during findMarkers at 43.9 of 64 GiB, and Modal rescheduled it; replicates 1 and 3 "
        "completed at 64 GiB with peaks of 43.5 and 43.8 GiB"
    ),
    "rna_workflow_census_1000000_census_scanpy_cpus_8_replicate_1_memory_64gib": (
        "ended during findMarkers at 43.4 of 64 GiB; Modal's log for the run records a preemption"
    ),
    "rna_workflow_census_1000000_census_scanpy_cpus_8_replicate_1_memory_128gib": (
        "ended during findMarkers at 43.4 of 128 GiB; Modal's log for the run records a second preemption. "
        "The next tier (256 GiB) was stopped by hand, because replicates 2 and 3 ran at 64 GiB"
    ),
    "rna_workflow_hbca_neurons_bpcells_cpus_16_replicate_1_memory_64gib": (
        "ended during runPca at 2.0 of 64 GiB, with no step above 2.1 GiB; the ladder reran it at 128 GiB"
    ),
    "rna_workflow_mssm_cohort_bpcells_cpus_16_replicate_1_memory_64gib": (
        "ended during runUmap with at most 10.0 GiB in use, of 64 GiB, and Modal rescheduled it; the ladder "
        "reran it at 128 GiB"
    ),
    "rna_workflow_embryo_1000000_seurat_cpus_8_replicate_1_memory_256gib": (
        "ended during runUmap with at most 83.0 GiB in use during that step, of 256 GiB (the run's 165.5 GiB "
        "peak was in normalisation), and Modal rescheduled it; the ladder reran it at 336 GiB"
    ),
}


# Attempts that ran out of time but whose record does not say so. The first attempt of this
# variant was cancelled at the ladder's time limit in runLeiden; the ladder of that day took the
# cancellation for a preemption and started a retry, which was stopped by hand and whose open
# record replaced the first one (kept in experiments/benchmarks/records/raw/prefrontal_cortex_seurat_bpcells_scaled_pca_interruption.json).
REPORTED_TIMEOUTS = {
    "rna_workflow_mssm_cohort_seurat_bpcells_spca_cpus_16_replicate_1_memory_64gib": (
        "the ladder's time limit per attempt (20.5 h, about 16 h of it in Leiden)",
        "runLeiden",
    ),
}


def ladder_outcomes(records: dict) -> dict:
    """The closed outcome that a ladder recorded for each attempt record, by record name."""
    outcomes = {}
    for name, record in records.items():
        if not name.endswith("_memory_tiers"):
            continue
        for attempt in record.get("attempts") or []:
            record_id = attempt.get("recordId")
            if record_id and attempt.get("status") not in (
                None,
                "attempting",
                "noRecord",
            ):
                outcomes[record_id] = attempt
    return outcomes


# Modal ends a function call after 24 hours; a record still open well after that ran out of time.
MODAL_CALL_LIMIT_SECONDS = 24 * 3600


def timed_out(record: dict) -> str | None:
    """The limit an attempt ran into when a ladder cancelled it at its deadline, or None.

    The ladder closes such a record with its outcome in the error. (A run whose coordinator
    had been lost stays open and is ended by Modal after 24 hours; the tier loop below
    recognises it as the open record of the ladder's top tier.)
    """
    if record.get("status") in ("killed", "attempting") and "'deadline': True" in (
        record.get("error") or ""
    ):
        return "the ladder's time limit per attempt"
    return None


def marker_run_fields(completed: dict) -> dict:
    """Input preparation, test time and memory of a completed marker-only run."""
    steps = completed.get("steps") or {}
    seconds = {k: v.get("seconds") or 0 for k, v in steps.items()}
    test, sample = seconds.get(MARKER_TEST_STEP, 0), seconds.get(MARKER_SAMPLE_STEP, 0)
    peaks = {k: (v.get("peakCgroupBytes") or 0) / GIB for k, v in steps.items()}
    inputs = [k for k in steps if k not in (MARKER_TEST_STEP, MARKER_SAMPLE_STEP)]
    return {
        "completedAtGib": completed["container"]["memoryMb"] / 1024,
        "inputMinutes": sum(seconds[k] for k in inputs) / 60,
        "markersMinutes": test / 60,
        "sampleMinutes": sample / 60,
        "stepMinutes": {k: v / 60 for k, v in seconds.items()},
        "stepPeakCgroupGib": peaks,
        "inputPeakCgroupGib": max((peaks[k] for k in inputs), default=0.0),
        "markersPeakCgroupGib": peaks.get(MARKER_TEST_STEP, 0.0),
        "peakCgroupGib": (completed.get("peakCgroupBytes") or 0) / GIB,
        "peakRssGib": (completed.get("peakRssBytes") or 0) / GIB,
        "cells": completed.get("cellsAfterFilter"),
        "genes": completed.get("genesAfterFilter"),
        "groups": completed.get("groups"),
        "nnz": completed.get("nnz"),
        "markerFormat": completed.get("markerFormat"),
    }


def workflow_summary(records: dict, collected_at: float, marker_only: bool = False) -> list[dict]:
    """Ladder rows of the workflow tools, or with ``marker_only`` of the marker-only tools."""
    closed = ladder_outcomes(records)
    attempts = {}
    for name, record in records.items():
        if (
            name.endswith("_memory_tiers")
            or "tool" not in record
            or record.get("dataset", "").startswith("hlca")
        ):
            continue
        if record["tool"].startswith(MARKER_TOOL_PREFIX) != marker_only:
            continue
        if "setAside" in record:
            # Kept as evidence, not a result: the first CyteArc image built hnswlib for the
            # image builder's CPU and stopped with an illegal instruction.
            continue
        if record.get("status") == "attempting" and record.get("sourceRecordId") in closed:
            # Before the harness recorded its function call, a container that Modal
            # rescheduled after the ladder had closed a tier could rewrite that tier's record
            # as a fresh attempt. The ladder's entry keeps the closed outcome.
            entry = closed[record["sourceRecordId"]]
            record = {
                **record,
                "status": entry["status"],
                "failedStep": entry.get("failedStep"),
                "workflowSeconds": entry.get("workflowSeconds"),
                "peakCgroupBytes": entry.get("peakCgroupBytes"),
                "outcomeSource": "ladder record",
            }
        if name in REPORTED_PREEMPTIONS:
            record = {
                **record,
                "status": "interrupted",
                "outcomeSource": "reclassified: " + REPORTED_PREEMPTIONS[name],
            }
        if name in REPORTED_TIMEOUTS:
            reason, step = REPORTED_TIMEOUTS[name]
            record = {**record, "status": "timedOut", "failedStep": step, "outcomeSource": reason}
        limit = timed_out(record)
        if limit:
            record = {
                **record,
                "status": "timedOut",
                "failedStep": record.get("failedStep") or record.get("currentStep"),
                "outcomeSource": limit,
            }
        key = (
            record["dataset"],
            record["tool"],
            record.get("replicate"),
            int(record["container"]["cpu"]),
        )
        attempts.setdefault(key, []).append({**record, "sourceRecord": name})
    rows = []
    for (dataset, tool, replicate, cpu), items in sorted(attempts.items()):
        items.sort(key=lambda r: r["container"]["memoryMb"])
        tiers = []
        completed = None
        top = items[-1]["container"]["memoryMb"] if items else 0
        for record in items:
            gib = record["container"]["memoryMb"] / 1024
            status = record.get("status")
            if status == "attempting" and record["container"]["memoryMb"] < top:
                # The ladder moved to a larger tier, so this container ended without closing
                # its record (killed before or while it published).
                status = "killed (record incomplete)"
            elif (
                status == "attempting"
                and collected_at - float(record.get("startedAt") or collected_at)
                > MODAL_CALL_LIMIT_SECONDS + 1800
            ):
                # The open record of the top tier, older than any call can live: Modal ended it.
                status = "timedOut"
                record = {
                    **record,
                    "failedStep": record.get("currentStep"),
                    "outcomeSource": "Modal's 24-hour limit per call",
                }
            tiers.append(
                {
                    "memoryGib": gib,
                    "status": status,
                    "failedStep": record.get("failedStep") or record.get("currentStep"),
                    "workflowMinutes": (record.get("workflowSeconds") or 0) / 60 or None,
                    "peakCgroupGib": (record.get("peakCgroupBytes") or 0) / GIB or None,
                    "outcomeSource": record.get("outcomeSource", "attempt record"),
                    "interruptions": record.get("interruptions") or [],
                }
            )
            if record.get("status") == "ok" and completed is None:
                completed = record
        row = {"dataset": dataset, "tool": tool, "replicate": replicate, "cpu": cpu, "tiers": tiers}
        if completed and marker_only:
            row.update(marker_run_fields(completed))
            row["sourceRecord"] = completed["sourceRecord"]
        elif completed:
            steps = completed.get("steps") or {}
            workflow = workflow_time_seconds(steps)
            analysis = analysis_time_seconds(steps)
            preparation = preparation_seconds(steps)
            marker_seconds = (steps.get("findMarkers") or {}).get("seconds") or 0
            loading_seconds = sum(
                (v.get("seconds") or 0) for k, v in steps.items() if k in LOADING_STEPS
            )
            row.update(
                {
                    "completedAtGib": completed["container"]["memoryMb"] / 1024,
                    "sourceRecord": completed["sourceRecord"],
                    "workflowMinutes": None if workflow is None else workflow / 60,
                    "processWallMinutes": completed["workflowSeconds"] / 60,
                    "withoutMarkersMinutes": None
                    if workflow is None
                    else (workflow - marker_seconds) / 60,
                    "loadingMinutes": loading_seconds / 60,
                    "markersMinutes": marker_seconds / 60,
                    "analysisMinutes": (workflow - loading_seconds - marker_seconds) / 60
                    if workflow is not None
                    else None,
                    "analysisTimeMinutes": (None if analysis is None else analysis / 60),
                    "preparationMinutes": None if preparation is None else preparation / 60,
                    "outputExportMinutes": ((steps.get("keepOutputs") or {}).get("seconds") or 0)
                    / 60,
                    "stepMinutes": {k: (v.get("seconds") or 0) / 60 for k, v in steps.items()},
                    "stepPeakCgroupGib": {
                        k: (v.get("peakCgroupBytes") or 0) / GIB for k, v in steps.items()
                    },
                    "peakCgroupGib": (completed.get("peakCgroupBytes") or 0) / GIB,
                    "peakRssGib": (completed.get("peakRssBytes") or 0) / GIB,
                    "cells": completed.get("nCells"),
                    "nnz": completed.get("nnz"),
                    "cellsAfterFilter": completed.get("cellsAfterFilter"),
                    "genesAfterFilter": completed.get("genesAfterFilter"),
                    "clusters": completed.get("clusters"),
                    "workers": completed.get("workers"),
                    "preemptible": completed["container"].get("preemptible"),
                    "budgetBytes": completed.get("budgetBytes"),
                    "countPolicy": completed.get("countPolicy"),
                    "executionReports": completed.get("executionReports"),
                }
            )
            if tool in REVIEW_ARMS:
                input_seconds = sum(
                    (v.get("seconds") or 0) for k, v in steps.items() if k in MARKER_INPUT_STEPS
                )
                if input_seconds and row["analysisMinutes"] is not None:
                    row["markerInputMinutes"] = input_seconds / 60
                    row["analysisMinutes"] -= input_seconds / 60
                row.update({k: completed[k] for k in REVIEW_FIELDS if k in completed})
        else:
            last = tiers[-1] if tiers else {}
            if last.get("status") == "error":
                row["outcome"] = "failed"
            elif last.get("status") == "interrupted":
                row["outcome"] = "interrupted"
            elif last.get("status") == "attempting":
                row["outcome"] = "running"
            elif last.get("status") == "cancelled":
                row["outcome"] = "stopped by hand after the tiers shown"
            elif last.get("status") == "timedOut":
                row["outcome"] = (
                    f"did not finish within {last.get('outcomeSource')} (in {last.get('failedStep')})"
                )
            else:
                row["outcome"] = "not completed at the largest tier tried"
            error_record = items[-1] if items else {}
            if "2^31" in (error_record.get("log") or ""):
                row["outcome"] = "dgCMatrix limit: more than 2^31 - 1 non-zero values"
            elif "cannot fit one source row" in (error_record.get("log") or ""):
                # CyteArc checks a memory limit before reading: the harness saw a MemoryError in
                # the log, but nothing ran out of memory (the container was far below its size).
                row["outcome"] = "memory limit below the import's minimum: refused before reading"
                plan = re.search(
                    r"One task needs about (\d+) bytes in addition to (\d+) resident bytes, but the operation limit is (\d+) bytes",
                    error_record.get("log") or "",
                )
                if plan:
                    task, resident, limit = (int(value) for value in plan.groups())
                    row["importPlanBytes"], row["limitBytes"] = task + resident, limit
                for tier in row["tiers"]:
                    if tier["status"] == "outOfMemory":
                        tier["status"] = "refusedLimit"
            elif error_record.get("returnCode") == -11:
                row["outcome"] = (
                    f"crashed: segmentation fault during {error_record.get('failedStep')}"
                )
                if "RSpectra" in (error_record.get("log") or ""):
                    row["outcome"] += " (RSpectra eigensolver of uwot's spectral initialisation)"
            row["nnz"] = error_record.get("nnz")
        rows.append(row)
    return rows


def mapping_summary(records: dict) -> list[dict]:
    rows = []
    for name, record in records.items():
        if (
            not record.get("dataset", "").startswith("hlca_heldout")
            or name.endswith("_memory_tiers")
            or record.get("status") != "ok"
        ):
            continue
        cells_path = RAW / f"{name}_predictions.csv"
        seconds = {k: v.get("seconds") or 0 for k, v in (record.get("steps") or {}).items()}
        read = sum(v for k, v in seconds.items() if k in READ_STEPS)
        query = sum(v for k, v in seconds.items() if k.startswith(QUERY_PREFIXES))
        row = {
            "method": record["tool"],
            "cpu": record["container"]["cpu"],
            "memoryGib": record["container"]["memoryMb"] / 1024,
            "gpu": record.get("gpu"),
            "workflowMinutes": record["workflowSeconds"] / 60,
            "readMinutes": read / 60,
            "referenceMinutes": (sum(seconds.values()) - read - query) / 60,
            "queryMinutes": query / 60,
            "stepMinutes": {k: v / 60 for k, v in seconds.items()},
            "peakCgroupGib": (record.get("peakCgroupBytes") or 0) / GIB,
        }
        if cells_path.exists():
            frame = pd.read_csv(cells_path)
            for level in LEVELS:
                truth, predicted = (
                    frame[f"true_{level}"].astype(str),
                    frame[f"pred_{level}"].astype(str),
                )
                confidence = frame[f"conf_{level}"].astype(float).fillna(0)
                correct = truth == predicted
                row[f"agreement_{level}"] = float(correct.mean())
                curve = []
                for threshold in THRESHOLDS:
                    assigned = confidence >= threshold
                    curve.append(
                        {
                            "threshold": threshold,
                            "assigned": float(assigned.mean()),
                            "agreementAmongAssigned": float(correct[assigned].mean())
                            if assigned.any()
                            else None,
                        }
                    )
                row[f"curve_{level}"] = curve
                recall = [float(correct[truth == label].mean()) for label in truth.unique()]
                row[f"meanRecall_{level}"] = float(np.mean(recall))
                # As in matched_results: recall averaged over the types with at least 1, 10 or 100 query cells.
                sizes = truth.value_counts()
                row[f"recallByMinCells_{level}"] = {
                    f"min{m}": {
                        "classes": int((sizes >= m).sum()),
                        "meanRecall": float(
                            np.mean(
                                [
                                    correct[truth == label].mean()
                                    for label in sizes.index[sizes >= m]
                                ]
                            )
                        ),
                    }
                    for m in (1, 10, 100)
                }
        rows.append(row)
    return rows


def sample_table(record_name: str) -> pd.DataFrame | None:
    """The sampled statistics of a marker-only run, as fetched by collect_tool_runs.py."""
    path = RAW / f"{record_name}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["group"] = frame["group"].astype(str)
    frame["gene"] = frame["gene"].astype(str)
    return frame


def same_test(runs: list[dict], records: dict) -> list[dict]:
    """For every dataset, the sampled statistics of each tool against a reference tool."""
    by_dataset: dict[str, dict[str, pd.DataFrame]] = {}
    for row in sorted(runs, key=lambda r: (r["dataset"], r["tool"], r["replicate"])):
        if "completedAtGib" not in row:
            continue
        name = row["sourceRecord"]
        if name not in records:
            raise ValueError(f"Marker summary refers to an absent measurement: {name}")
        frame = sample_table(name)
        if frame is not None:
            # The first completed replicate stands for the tool.
            by_dataset.setdefault(row["dataset"], {}).setdefault(row["tool"], frame)
    comparisons = []
    for dataset, tables in sorted(by_dataset.items()):
        reference = next((tool for tool in SAME_TEST_REFERENCES if tool in tables), None)
        entry: dict = {
            "dataset": dataset,
            "tools": sorted(tables),
            "reference": reference,
            "auc": {},
            "pval": {},
        }
        if reference is None:
            comparisons.append(entry)
            continue
        base = tables[reference]
        entry["sampledGroups"] = int(base["group"].nunique())
        entry["sampledGenes"] = int(base["gene"].nunique())
        for tool, frame in sorted(tables.items()):
            if tool == reference:
                continue
            merged = frame.merge(base, on=["group", "gene"], suffixes=("", "_ref"))
            if "auc" in frame and "auc" in base:
                difference = (merged["auc"] - merged["auc_ref"]).abs()
                entry["auc"][tool] = {
                    "pairs": int(len(merged)),
                    "maxAbsDiff": float(difference.max()),
                    "meanAbsDiff": float(difference.mean()),
                }
            if tool in CORRECTED_P_TOOLS and reference in CORRECTED_P_TOOLS:
                p, p_ref = (
                    merged["pval"].to_numpy(dtype=float),
                    merged["pval_ref"].to_numpy(dtype=float),
                )
                both = (p >= LOG10_P_FLOOR) & (p_ref >= LOG10_P_FLOOR)
                entry["pval"][tool] = {
                    "pairs": int(len(merged)),
                    "maxAbsDiff": float(np.max(np.abs(p - p_ref), initial=0.0)),
                    "pairsAboveFloor": int(both.sum()),
                    "maxAbsLog10Diff": float(
                        np.max(np.abs(np.log10(p[both]) - np.log10(p_ref[both])), initial=0.0)
                    ),
                    "pairsWithOneBelowFloor": int(
                        ((p < LOG10_P_FLOOR) != (p_ref < LOG10_P_FLOOR)).sum()
                    ),
                }
        entry["largestAucDiff"] = max(
            (v["maxAbsDiff"] for v in entry["auc"].values()), default=None
        )
        entry["largestLog10PDiff"] = max(
            (v["maxAbsLog10Diff"] for v in entry["pval"].values()), default=None
        )
        if {"markers_scanpy_csr", "markers_scanpy_csc"} <= set(tables):
            csr, csc = tables["markers_scanpy_csr"], tables["markers_scanpy_csc"]
            merged = csr.merge(csc, on=["group", "gene"], suffixes=("_csr", "_csc"))
            entry["scanpyCsrCscIdentical"] = bool(
                len(merged) == len(csr) == len(csc)
                and np.array_equal(merged["score_csr"].to_numpy(), merged["score_csc"].to_numpy())
                and np.array_equal(merged["pval_csr"].to_numpy(), merged["pval_csc"].to_numpy())
            )
        comparisons.append(entry)
    return comparisons


def markers_common_summary(records: dict, collected_at: float) -> dict:
    """The common inputs, the marker-only runs and the comparison of their sampled statistics."""
    preparation = []
    for name, record in sorted(records.items()):
        if record.get("experiment") == "markersCommon":
            keep = (
                "dataset",
                "status",
                "nCellsSource",
                "nGenesSource",
                "nnzSource",
                "filter",
                "cellsAfterFilter",
                "genesAfterFilter",
                "nnz",
                "groups",
                "groupSizes",
                "clustering",
                "sample",
                "bytes",
                "sha256",
                "timings",
                "downloadSeconds",
                "error",
            )
            preparation.append({k: record[k] for k in keep if k in record})
    runs = workflow_summary(records, collected_at, marker_only=True)
    return {"preparation": preparation, "runs": runs, "sameTest": same_test(runs, records)}


def main() -> None:
    document = json.loads((DATA / "toolkit_comparison_runs.json").read_text())
    records = toolkit_records(document)
    collected_at = datetime.fromisoformat(document["provenance"]["capturedAt"]).timestamp()
    current, validation = verified_workflow_records(records)
    comparative = {
        name: record
        for name, record in records.items()
        if not record.get("tool", "").startswith(("cytearc", "cytearc"))
    }
    comparative.update(current)
    summary = {
        "provenance": {
            **document["provenance"],
            "summariser": "experiments/benchmarks/tool_summary.py",
        },
        "workflow": workflow_summary(comparative, collected_at),
        "workflowInputValidation": validation,
        "timingDefinition": "Recorded input-to-marker span; analysis starts after reusable format preparation. Output export is excluded. Process wall time is retained separately.",
        "mapping": mapping_summary(records),
        "markersCommon": markers_common_summary(records, collected_at),
    }
    BENCHMARKS_DERIVED.mkdir(parents=True, exist_ok=True)
    (BENCHMARKS_DERIVED / "toolkit_comparisons.json").write_text(
        json.dumps(summary, indent=1, default=str) + "\n"
    )
    with (BENCHMARKS_DERIVED / "toolkit_step_measurements.csv").open("w", newline="") as handle:
        fields = [
            "record",
            "dataset",
            "tool",
            "memoryGib",
            "step",
            "seconds",
            "peakRssBytes",
            "peakCgroupBytes",
            "peakAnonBytes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, record in comparative.items():
            if "tool" not in record or "container" not in record or name.endswith("_memory_tiers"):
                continue
            for step, values in (record.get("steps") or {}).items():
                writer.writerow(
                    {
                        "record": name,
                        "dataset": record["dataset"],
                        "tool": record["tool"],
                        "memoryGib": record["container"]["memoryMb"] / 1024,
                        "step": step,
                        **{
                            key: values.get(key)
                            for key in (
                                "seconds",
                                "peakRssBytes",
                                "peakCgroupBytes",
                                "peakAnonBytes",
                            )
                        },
                    }
                )
    markers = summary["markersCommon"]
    print(
        f"workflow rows {len(summary['workflow'])}, mapping rows {len(summary['mapping'])}, "
        f"marker-only rows {len(markers['runs'])} on {len(markers['preparation'])} common inputs"
    )


if __name__ == "__main__":
    main()
