"""Write the reference ladder and blood store annotation records from their job records.

Each job record is the runner's record of one Modal call: an s3:// URI in the experiment bucket
or a local copy of it. A blood store annotation record also counts, for each store, the cells
whose author label is one of the reference's types and those whose transferred label agrees,
from the per-cell labels (cells.npz) the job uploaded for the store: from the bucket, or beside a
local copy of the job record, in a folder named for the store.

    uv run --locked python -m experiments.mapping.collect_records reference-ladder RECORD
    uv run --locked python -m experiments.mapping.collect_records blood-store-annotation RECORD...
"""

import argparse
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from experiments.runtime.common import ROOT

RECORDS = ROOT / "experiments" / "mapping" / "records"
REFERENCE_LADDER = RECORDS / "reference_ladder.json"
BLOOD_STORE_ANNOTATION = RECORDS / "blood_store_annotation"
REFERENCE_LADDER_FIELDS = {
    "design",
    "queryCells",
    "vocabulary",
    "rareTypes",
    "confusionLabels",
    "rows",
    "stepSecondsDefinition",
    "persistedReferenceDefinition",
    "outcomeRule",
    "confusionDefinition",
}
NUMBER_LIST = re.compile(r"\[\s+([-\d.eE+]+(?:,\s+[-\d.eE+]+)*)\s+\]")
# The transferred label of a cell that a tied vote left without a type.
UNASSIGNED = "unassigned"
# The arrays of a store's cells.npz that the label agreement reads.
LABEL_ARRAYS = ("mapped_codes", "mapped_names", "author_codes", "author_type_names")


def read_job_record(location: str) -> dict:
    from experiments.runtime.harness import Objects

    return Objects(os.environ.get("R2_ENDPOINT_URL")).get_json(location)


def completed_result(record: dict, function: str) -> dict:
    if record.get("function") != function or record.get("status") != "completed":
        raise ValueError(
            f"Expected a completed {function} job record; got function "
            f"{record.get('function')!r} with status {record.get('status')!r}"
        )
    return record["result"]


def reference_ladder_record(job: dict) -> dict:
    """The record that evaluate_ladder returns, checked to hold exactly the record's fields."""
    record = completed_result(job, "evaluate_ladder")
    if set(record) != REFERENCE_LADDER_FIELDS:
        raise ValueError(
            "The evaluate_ladder result does not hold the reference ladder record's fields; "
            f"missing {sorted(REFERENCE_LADDER_FIELDS - set(record))}, "
            f"extra {sorted(set(record) - REFERENCE_LADDER_FIELDS)}"
        )
    return record


def label_agreement(labels: Mapping[str, np.ndarray], vocabulary: list[str]) -> dict[str, int]:
    """A store's labelled cells, the cells whose author label is one of the reference's types
    (typed), and of those the cells whose transferred label is that type (agreeing). A cell that a
    tied vote left unassigned agrees with no author type."""
    mapped_names = [str(name) for name in labels["mapped_names"]]
    outside = set(mapped_names) - set(vocabulary) - {UNASSIGNED}
    if outside:
        raise ValueError(f"Transferred labels outside the reference's types: {sorted(outside)}")
    mapped_type = np.array(
        [vocabulary.index(name) if name in vocabulary else -2 for name in mapped_names],
        dtype=np.int64,
    )
    author_type = np.array(
        [
            vocabulary.index(str(name)) if str(name) in vocabulary else -1
            for name in labels["author_type_names"]
        ],
        dtype=np.int64,
    )
    mapped = mapped_type[np.asarray(labels["mapped_codes"], dtype=np.int64)]
    author = author_type[np.asarray(labels["author_codes"], dtype=np.int64)]
    if mapped.shape != author.shape:
        raise ValueError(f"{mapped.size} transferred labels against {author.size} author labels")
    typed = author >= 0
    return {
        "labelledCells": int(mapped.size),
        "typedCells": int(typed.sum()),
        "agreeingCells": int(((mapped == author) & typed).sum()),
    }


def read_store_labels(location: str, store: dict) -> dict[str, np.ndarray]:
    """The per-cell labels the job uploaded for one store: from the bucket for a job record read
    from the bucket, otherwise from <job record folder>/<store>/cells.npz."""
    if location.startswith("s3://"):
        from experiments.runtime.harness import Objects

        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "cells.npz"
            Objects(os.environ.get("R2_ENDPOINT_URL")).download(store["outputs"]["cells.npz"], path)
            with np.load(path, allow_pickle=False) as cells:
                return {name: cells[name] for name in LABEL_ARRAYS}
    with np.load(Path(location).parent / store["store"] / "cells.npz", allow_pickle=False) as cells:
        return {name: cells[name] for name in LABEL_ARRAYS}


def blood_store_annotation_record(job: dict, labels: Mapping[str, Mapping]) -> dict:
    """One study's annotation: reference opening and reads, memory, and each store's cells, mapping
    seconds and label agreement, from the job record and each store's per-cell labels."""
    result = completed_result(job, "map_and_aggregate")
    vocabulary = list(result["referenceOpening"]["registration"]["vocabulary"])
    agreement = {}
    for store in result["stores"]:
        if store["store"] not in labels:
            raise ValueError(f"No per-cell labels for the store {store['store']}")
        agreement[store["store"]] = label_agreement(labels[store["store"]], vocabulary)
        if agreement[store["store"]]["labelledCells"] != store["eligibleCells"]:
            raise ValueError(
                f"{store['store']}: {agreement[store['store']]['labelledCells']} labelled cells "
                f"against {store['eligibleCells']} eligible"
            )
    opening = result["referenceOpening"]
    # The reference probe is not reset after opening, so its job total includes the opening.
    reads = result["referenceReadsDuringMapping"]
    return {
        "study": result["settings"]["study"],
        "mappingRule": result["mappingRule"],
        "eligibility": result["eligibility"],
        "referenceOpening": {
            "seconds": opening["seconds"],
            "countReadBytes": opening["io"]["countReadBytes"],
            "transferredBytes": opening["io"]["transferredBytes"],
            "referenceCells": opening["registration"]["cells"],
            "referenceFeatures": opening["registration"]["featureCount"],
        },
        "referenceReadsIncludingOpening": {
            "countReadBytes": reads["countReadBytes"],
            "transferredBytes": reads["transferredBytes"],
        },
        "peakContainerBytes": result["resources"]["cgroupMemoryCurrentPeakBytes"],
        "containerMemoryBytes": result["resources"]["memoryMaxBytes"],
        "stores": [
            {
                "store": store["store"],
                "storeCells": store["storeCells"],
                "eligibleCells": store["eligibleCells"],
                "donors": store["donors"],
                "referenceFeaturesMissing": store["referenceFeaturesMissing"],
                "mappingPeakContainerBytes": store["mapping"]["resources"][
                    "cgroupMemoryCurrentPeakBytes"
                ],
                # Reading the store's counts, projecting onto the reference and voting.
                "mappingSeconds": store["mapping"]["elapsedSeconds"],
                "typedCells": agreement[store["store"]]["typedCells"],
                "agreeingCells": agreement[store["store"]]["agreeingCells"],
            }
            for store in result["stores"]
        ],
    }


def write_record(path: Path, record: dict) -> None:
    """Indented JSON with each list of numbers on one line, so confusion matrices read as rows."""
    text = NUMBER_LIST.sub(
        lambda match: "[" + " ".join(match.group(1).split()) + "]", json.dumps(record, indent=1)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("reference-ladder").add_argument("record")
    commands.add_parser("blood-store-annotation").add_argument("records", nargs="+")
    arguments = parser.parse_args()
    if arguments.command == "reference-ladder":
        record = reference_ladder_record(read_job_record(arguments.record))
        write_record(REFERENCE_LADDER, record)
        return
    for location in arguments.records:
        job = read_job_record(location)
        labels = {
            store["store"]: read_store_labels(location, store)
            for store in completed_result(job, "map_and_aggregate")["stores"]
        }
        record = blood_store_annotation_record(job, labels)
        write_record(BLOOD_STORE_ANNOTATION / f"{record['study']}.json", record)


if __name__ == "__main__":
    main()
