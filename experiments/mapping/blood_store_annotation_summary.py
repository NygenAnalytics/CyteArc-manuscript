"""Jobs, stores, eligible cells, memory peaks, reference reads, mapping seconds and label
disagreement of the blood store annotation, from records/blood_store_annotation/.

    uv run --locked python -m experiments.mapping.blood_store_annotation_summary
"""

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

RECORDS = ROOT / "experiments" / "mapping" / "records" / "blood_store_annotation"
SUMMARY = DERIVED / "mapping" / "blood_store_annotation.json"


def pooled(studies: list[dict], typed: int) -> dict:
    """Some jobs' studies pooled over their typed cells: which studies, their typed cells and share of all ``typed``
    cells, and their agreement, which is None for no study."""
    cells = sum(study["typedCells"] for study in studies)
    return {
        "studies": [study["study"] for study in studies],
        "typedCells": cells,
        "typedShare": cells / typed,
        "agreement": sum(study["agreeingCells"] for study in studies) / cells if cells else None,
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError(f"No blood store annotation records in {RECORDS.relative_to(ROOT)}")
    stores = [store["store"] for record in records for store in record["stores"]]
    if len(set(stores)) != len(stores):
        raise ValueError("A blood store is annotated by more than one job")
    peaks = [record["peakContainerBytes"] for record in records]
    opening = [record["referenceOpening"]["seconds"] for record in records]
    mapping = [store["mappingSeconds"] for record in records for store in record["stores"]]
    typed = sum(store["typedCells"] for record in records for store in record["stores"])
    agreeing = sum(store["agreeingCells"] for record in records for store in record["stores"])
    studies = [
        {
            "study": record["study"],
            "stores": len(record["stores"]),
            "eligibleCells": sum(store["eligibleCells"] for store in record["stores"]),
            # Reference artifact bytes read while mapping, after the reference was opened.
            "referenceReadBytesAfterOpening": (
                record["referenceReadsIncludingOpening"]["transferredBytes"]
                - record["referenceOpening"]["transferredBytes"]
            ),
            "typedCells": sum(store["typedCells"] for store in record["stores"]),
            "agreeingCells": sum(store["agreeingCells"] for store in record["stores"]),
        }
        for record in records
    ]
    for study in studies:
        study["agreement"] = study["agreeingCells"] / study["typedCells"]
        study["disagreement"] = 1.0 - study["agreement"]
        study["typedShare"] = study["typedCells"] / typed
    return {
        "source": str(RECORDS.relative_to(ROOT)),
        "jobs": len(records),
        "stores": len(stores),
        "eligibleCells": sum(
            store["eligibleCells"] for record in records for store in record["stores"]
        ),
        "peakContainerBytes": {"lowest": min(peaks), "highest": max(peaks)},
        "referenceOpeningSeconds": {"lowest": min(opening), "highest": max(opening)},
        # Each store's mapping with label transfer: reading its counts, projecting and voting.
        "mappingSeconds": {"lowest": min(mapping), "highest": max(mapping)},
        # Cells whose author label is one of the reference's types, and the share of them whose
        # transferred label is that type (agreement) or is not (disagreement).
        "typedCells": typed,
        "agreeingCells": agreeing,
        "agreement": agreeing / typed,
        "disagreement": 1.0 - agreeing / typed,
        # Each job's study with its share of the typed cells and its agreement, and the jobs of one store and of
        # several stores each pooled.
        "studies": studies,
        "singleStoreStudies": pooled([s for s in studies if s["stores"] == 1], typed),
        "multiStoreStudies": pooled([s for s in studies if s["stores"] > 1], typed),
    }


def main() -> None:
    records = [load_json(path) for path in sorted(RECORDS.glob("*.json"))]
    write_json_at(SUMMARY, summarize(records))


if __name__ == "__main__":
    main()
