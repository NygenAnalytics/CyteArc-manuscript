import cytearc
import numpy as np
import pandas as pd
from cytearc.storage.refs import artifact_path
from zarr.storage import LocalStore

from experiments.mapping import DATA, WORK
from experiments.mapping import INPUTS as DATASETS
from experiments.runtime import kang_labels
from experiments.runtime.catalogue_counts import build_store, input_record
from experiments.runtime.common import (
    Stopwatch,
    configure_cytearc,
    dir_bytes,
    write_csv,
    write_json,
)
from experiments.runtime.recording_store import StoreProbe, wrap_recording_store
from experiments.runtime.rna_recipe import analysis_record, run_rna

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
SAVE_K = 11
LABEL = kang_labels.LABEL_COLUMN
REFERENCE = "kang_15K_pbmc_rnaseq"
QUERY = "kang_14K_ifnb-pbmc_rnaseq"
# The reference analysis: CyteArc's default cell filter and RNA recipe on the control lane's demuxlet singlets.
RUN_LABEL = "kang_reference"


def prepared_stores(work, inputs) -> tuple:
    """The control (reference) and stimulated (query) stores built from the pinned counts, both labelled with
    GEO's author labels, and the reference analysis; returns the paths and the input, label and analysis
    records."""
    reference_path = build_store(REFERENCE, work / "reference.zarr", downloads=inputs)
    query_path = build_store(QUERY, work / "query.zarr", downloads=inputs)
    reference = cytearc.DataStore(str(reference_path), default_assay="RNA", nthreads=4)
    query = cytearc.DataStore(str(query_path), default_assay="RNA", nthreads=4)
    records = {
        "inputs": {REFERENCE: input_record(REFERENCE), QUERY: input_record(QUERY)},
        "labels": {
            REFERENCE: kang_labels.label_store(reference, REFERENCE, inputs),
            QUERY: kang_labels.label_store(query, QUERY, inputs),
        },
    }
    run = run_rna(
        reference,
        label=RUN_LABEL,
        cells=kang_labels.SINGLETS,
        umap=False,
        cell_cycle=False,
        paris=False,
        doublets=False,
        markers=False,
    )
    records["analysis"] = analysis_record(reference, run)
    return reference_path, query_path, records


def main() -> None:
    configure_cytearc()
    work = WORK / "kang"
    reference_path, query_path, records = prepared_stores(work, DATASETS)
    probe = StoreProbe()
    reference_store = wrap_recording_store(LocalStore(str(reference_path)), probe=probe)
    ds_reference = cytearc.DataStore(
        reference_store, default_assay="RNA", nthreads=4, zarr_mode="r+", zarrProfile="fast_local"
    )
    run = ds_reference.pipeline.open(label=RUN_LABEL)
    with Stopwatch() as build_time:
        reference_ref = ds_reference.mapping.build_reference(run["neighbors"])
    build_ops = probe.to_json()
    probe.reset()
    with Stopwatch() as load_time:
        reference = ds_reference.mapping.load_reference(reference_ref)
    load_ops = probe.to_json()
    probe.reset()
    ds_query = cytearc.DataStore(str(query_path), default_assay="RNA", nthreads=4)
    selection = ds_query.snapshot_cell_selection(kang_labels.SINGLETS)
    with Stopwatch() as map_time:
        mapping_ref = ds_query.mapping.run(
            reference,
            selection,
            query_assay="RNA",
            save_k=SAVE_K,
            missing_feature_policy="reference_mean",
        )
    map_ops = probe.to_json()
    probe.reset()
    mapping = ds_query.mapping.load_result(mapping_ref, reference=reference, load_arrays=True)
    query_labels = np.asarray(ds_query.cells.fetch(LABEL, key=kang_labels.SINGLETS)).astype(str)
    threshold_rows = []
    per_label_rows = []
    for threshold in THRESHOLDS:
        with Stopwatch() as transfer_time:
            transferred = ds_query.mapping.load_label_transfer(
                ds_query.mapping.transfer_labels(
                    mapping_ref,
                    reference=reference,
                    reference_labels=LABEL,
                    threshold_fraction=threshold,
                )
            ).labels.fillna("NA")
        values = transferred.to_numpy().astype(str)
        accepted = (values != "NA") & pd.notna(transferred.to_numpy())
        agreement = (
            float(np.mean(values[accepted] == query_labels[accepted]))
            if accepted.any()
            else float("nan")
        )
        threshold_rows.append(
            {
                "threshold": threshold,
                "queryCells": int(values.size),
                "accepted": int(accepted.sum()),
                "abstained": int((~accepted).sum()),
                "abstainedFraction": float((~accepted).mean()),
                "agreementAccepted": agreement,
                "agreementAllCells": float(np.mean(values == query_labels)),
                "transferSeconds": transfer_time.seconds,
            }
        )
        if threshold == 0.6:
            for label in sorted(set(query_labels)):
                mask = query_labels == label
                acc = accepted & mask
                per_label_rows.append(
                    {
                        "authorLabel": label,
                        "queryCells": int(mask.sum()),
                        "accepted": int(acc.sum()),
                        "abstainedFraction": float(1.0 - acc.sum() / mask.sum()),
                        "agreementAccepted": float(np.mean(values[acc] == label))
                        if acc.any()
                        else float("nan"),
                        "majorityTransferredLabel": str(pd.Series(values[acc]).mode().iloc[0])
                        if acc.any()
                        else None,
                    }
                )
    transfer_ops = probe.to_json()
    probe.reset()
    evidence = ds_query.mapping.load_label_transfer(
        ds_query.mapping.transfer_labels(
            mapping_ref, reference=reference, reference_labels=LABEL, threshold_fraction=0.6
        )
    ).evidence
    evidence_columns = [c for c in evidence.columns if evidence[c].dtype.kind in "fi"]
    frame = evidence.copy()
    frame["authorLabel"] = query_labels[: len(frame)]
    evidence_summary = {
        str(label): {column: float(group[column].median()) for column in evidence_columns}
        for label, group in frame.groupby("authorLabel")
    }
    n_reference = int(reference.selected_cell_count)
    n_panel = int(reference.feature_ids.size)
    dims = int(reference.model.loadings.shape[1])
    reference_bytes = dir_bytes(reference_path / artifact_path(reference_ref))
    index_bytes = dir_bytes(reference_path / artifact_path(reference.ann_index))
    reduction_bytes = dir_bytes(reference_path / artifact_path(reference.reduction))
    counts_bytes = dir_bytes(reference_path / "RNA" / "counts") + dir_bytes(
        reference_path / "RNA" / "countsT"
    )
    store_bytes = dir_bytes(reference_path)
    element_bytes = 4
    link_bytes_per_cell = max(0.0, index_bytes / n_reference - dims * element_bytes)
    predicted_bytes = n_panel * dims * element_bytes + n_reference * (
        dims * element_bytes + link_bytes_per_cell
    )
    payload = {
        "description": "Kang control PBMC reference mapped by the stimulated sample. Bytes are from a recording wrapper around the reference store, split by phase.",
        **records,
        "reference": {
            "dataset": REFERENCE,
            "cellsSelected": n_reference,
            "cellsTotal": int(ds_reference.cells.N),
            "genesTotal": int(ds_reference.RNA.feats.N),
            "panelGenes": n_panel,
            "dims": dims,
            "method": reference.method,
            "referenceArtifactBytes": reference_bytes,
            "annIndexBytes": index_bytes,
            "reductionArtifactBytes": reduction_bytes,
            "countStoreBytes": counts_bytes,
            "wholeStoreBytes": store_bytes,
            "denseLogicalCountBytes": int(ds_reference.cells.N)
            * int(ds_reference.RNA.feats.N)
            * element_bytes,
            "referencePlusIndexOverCounts": (reference_bytes + index_bytes) / counts_bytes,
            "eq5": {
                "panelTerm": n_panel * dims * element_bytes,
                "indexLinkBytesPerCell": link_bytes_per_cell,
                "predictedReadBytes": predicted_bytes,
            },
        },
        "query": {
            "dataset": QUERY,
            "cells": int(values.size),
            "saveK": SAVE_K,
            "diagnostics": dict(mapping.diagnostics),
            "correctionMethod": mapping.correction_method,
        },
        "seconds": {
            "buildReference": build_time.seconds,
            "loadReference": load_time.seconds,
            "runMapping": map_time.seconds,
        },
        "referenceStoreOperations": {
            "buildReference": build_ops,
            "loadReference": load_ops,
            "runMapping": map_ops,
            "labelTransfer": transfer_ops,
        },
        "thresholds": threshold_rows,
        "perLabelAt0p6": per_label_rows,
        "evidenceMediansByAuthorLabel": evidence_summary,
        "evidenceColumns": list(evidence.columns),
    }
    write_json("kang_stimulated_query_mapping.json", payload, directory=DATA)
    write_csv("kang_mapping_coverage.csv", threshold_rows, directory=DATA)
    write_csv("kang_mapping_label_accuracy.csv", per_label_rows, directory=DATA)
    for row in threshold_rows:
        print(
            f"theta={row['threshold']:.1f}: agreement {row['agreementAccepted']:.3f}, abstained {row['abstainedFraction']:.3f}"
        )
    print(
        f"reference reads during mapping: {map_ops.get('readTransferredBytes')} bytes; reference+index {reference_bytes + index_bytes} vs counts {counts_bytes}"
    )


if __name__ == "__main__":
    main()
