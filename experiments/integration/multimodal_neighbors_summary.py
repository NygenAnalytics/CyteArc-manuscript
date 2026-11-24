"""Summarise the WNN kernel replay of each small multimodal dataset with the Seurat fixture check.

The replay records are in records/multimodal_neighbors/, and the fixture check is the record
that wnn_seurat.py writes from the committed Seurat fixtures.
"""

import json
from pathlib import Path

from experiments.runtime.common import DERIVED, ROOT, scrub_private, write_json_at

RECORDS = ROOT / "experiments/integration/records"
OUTPUT = DERIVED / "integration" / "multimodal_neighbors.json"
SMALL_DATASETS = ("tenx_8K_pbmc_citeseq", "swanson_7K_pbmc_teaseq")


def completed_records(directory: Path) -> list[tuple[Path, dict]]:
    records = []
    seen = set()
    for path in sorted(directory.rglob("*.json")):
        envelope = json.loads(path.read_text())
        if envelope.get("status") != "completed" or not isinstance(envelope.get("result"), dict):
            continue
        for group in ("stages", "reuseStages", "freshRevisionStages"):
            for stage in envelope["result"].get(group, []):
                if (
                    "operationPeakBytes" in stage
                    and stage.get("operationPeakSource") != "cgroupMemoryCurrent"
                ):
                    raise ValueError(f"A different memory metric was measured in {path}")
        call_id = envelope.get("callId")
        if call_id is not None and call_id in seen:
            continue
        seen.add(call_id)
        records.append((path, envelope["result"]))
    return records


def summarise(records: list[tuple[Path, dict]], fixture_check: dict) -> dict:
    workflows = {
        r["inputsSha256"]: r for _, r in records if "reusedArtifacts" in r and "inputsSha256" in r
    }
    dimensions = {
        r["inputsSha256"]: r["modalityDimensions"]
        for _, r in records
        if "residentCountBytes" in r and "modalityDimensions" in r
    }
    datasets = {}
    for path, record in records:
        if "matchedKernel" not in record:
            continue
        key = record["inputsSha256"]
        source = workflows[key]
        name = source["dataset"]
        if name not in SMALL_DATASETS:
            continue
        if not record["matchedKernel"]["passed"]:
            raise ValueError(f"Numerical validation failed in {path}")
        entry = datasets.setdefault(
            name,
            {
                "cells": source["cells"],
                "modalities": source["modalities"],
                "modalityDimensions": dimensions[key],
                "settings": source["settings"],
                "sourceArchiveSha256": source["sourceArchiveSha256"],
                "cellSelectionSha256": source["cellSelectionSha256"],
                "records": [],
            },
        )
        entry["records"].append(scrub_private(json.loads(path.read_text())))
    missing = sorted(set(SMALL_DATASETS) - set(datasets))
    if missing:
        raise ValueError(f"No completed WNN kernel replay for {', '.join(missing)}")
    return {
        "comparison": "Full real-input WNN kernel replay in R on the same candidate rows and reductions",
        "sharedPoolCondition": "Self-free modality neighbor rows with k=20, nearest-to-kth bandwidth, L2-normalized reductions; public Seurat search and bandwidth are evaluated separately.",
        "datasets": datasets,
        "sharedPoolGolden": {
            "twoModality": fixture_check["twoModality"],
            "threeModality": fixture_check["threeModality"],
        },
    }


def main() -> None:
    records = completed_records(RECORDS / "multimodal_neighbors")
    fixture_check = json.loads((RECORDS / "weighted_neighbors_seurat_agreement.json").read_text())
    write_json_at(OUTPUT, summarise(records, fixture_check))


if __name__ == "__main__":
    main()
