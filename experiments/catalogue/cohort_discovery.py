"""Select multi-study human tissues and multi-cohort diseases from the CyteBase catalog.

Each mode writes one record into records/: cohort_discovery.json, the catalogue queries and the
cell metadata of the stores given; tabula_count_bytes.json, the count array bytes of one store;
and lung_donor_overlap.json, the donors of the core Human Lung Cell Atlas and the Kaminski cohort.
"""

import argparse
from collections import Counter
from itertools import combinations

import cytebase
import numpy as np

from experiments.catalogue import HERE
from experiments.runtime.common import configure_cytearc, write_json

RECORDS = HERE / "records"


def multi_cohort_diseases(catalog: cytebase.Catalog) -> list[dict]:
    sql = """
    WITH expanded AS (
        SELECT
            cytebase_id,
            collection_id,
            first_author,
            year,
            title,
            doi,
            zarr_uri,
            cell_count,
            primary_cell_count,
            tissue_labels,
            assay_labels,
            disease_labels,
            len(cell_type_labels) AS n_cell_types,
            unnest(disease_labels) AS disease
        FROM datasets
        WHERE list_contains(organism_labels, 'Homo sapiens')
          AND NOT list_has_any(assay_labels, ['Visium Spatial Gene Expression V1', 'Slide-seqV2'])
          AND primary_cell_count >= 10000
          AND status = 'ready'
    ),
    ranked_diseases AS (
        SELECT
            disease,
            count(*) AS n_datasets,
            count(DISTINCT collection_id) AS n_collections,
            sum(primary_cell_count) AS total_primary_cells
        FROM expanded
        WHERE disease != 'normal'
        GROUP BY 1
        HAVING count(DISTINCT collection_id) >= 3
        ORDER BY n_collections DESC, n_datasets DESC
    )
    SELECT
        r.disease,
        r.n_datasets,
        r.n_collections,
        r.total_primary_cells,
        e.cytebase_id,
        e.collection_id,
        e.first_author,
        e.year,
        e.primary_cell_count,
        e.tissue_labels,
        e.assay_labels,
        e.disease_labels,
        e.n_cell_types,
        e.title
    FROM ranked_diseases r
    JOIN expanded e ON r.disease = e.disease
    ORDER BY r.n_collections DESC, r.disease, e.primary_cell_count DESC
    """
    grouped: dict[str, dict] = {}
    for row in catalog.query(sql):
        disease = row["disease"]
        entry = grouped.setdefault(
            disease,
            {
                "disease": disease,
                "nDatasets": int(row["n_datasets"]),
                "nCollections": int(row["n_collections"]),
                "totalPrimaryCells": int(row["total_primary_cells"]),
                "datasets": [],
            },
        )
        entry["datasets"].append(
            {
                "cytebaseId": row["cytebase_id"],
                "collectionId": row["collection_id"],
                "author": row["first_author"],
                "year": row["year"],
                "primaryCells": int(row["primary_cell_count"]),
                "tissues": list(row["tissue_labels"] or []),
                "assays": list(row["assay_labels"] or []),
                "diseases": list(row["disease_labels"] or []),
                "hasNormalControls": "normal" in (row["disease_labels"] or []),
                "nCellTypes": int(row["n_cell_types"] or 0),
                "title": row["title"],
            }
        )
    return list(grouped.values())


def multi_study_tissues(catalog: cytebase.Catalog) -> list[dict]:
    tissues = (
        "blood",
        "lung",
        "kidney",
        "liver",
        "ileum",
        "dorsolateral prefrontal cortex",
        "bone marrow",
        "breast",
    )
    out = []
    for tissue in tissues:
        sql = """
        SELECT
            cytebase_id,
            collection_id,
            first_author,
            year,
            primary_cell_count,
            cell_count,
            assay_labels,
            disease_labels,
            len(cell_type_labels) AS n_cell_types,
            title
        FROM datasets
        WHERE list_contains(organism_labels, 'Homo sapiens')
          AND list_contains(tissue_labels, ?)
          AND NOT list_has_any(assay_labels, ['Visium Spatial Gene Expression V1', 'Slide-seqV2'])
          AND primary_cell_count BETWEEN 15000 AND 400000
          AND len(cell_type_labels) >= 8
          AND status = 'ready'
        ORDER BY primary_cell_count DESC
        LIMIT 12
        """
        rows = [
            {
                "cytebaseId": r["cytebase_id"],
                "collectionId": r["collection_id"],
                "author": r["first_author"],
                "year": r["year"],
                "primaryCells": int(r["primary_cell_count"]),
                "totalCells": int(r["cell_count"]),
                "assays": list(r["assay_labels"] or []),
                "diseases": list(r["disease_labels"] or []),
                "nCellTypes": int(r["n_cell_types"] or 0),
                "title": r["title"],
            }
            for r in catalog.query(sql, [tissue])
        ]
        out.append({"tissue": tissue, "datasets": rows})
    return out


def large_multimodal_datasets(catalog: cytebase.Catalog) -> list[dict]:
    rows = catalog.query("""
        SELECT cytebase_id, title, doi, collection_id, primary_cell_count,
               assay_labels, zarr_uri, cell_type_labels
        FROM datasets
        WHERE status = 'ready' AND primary_cell_count >= 100000
          AND (list_contains(assay_labels, '10x multiome')
               OR lower(title) LIKE '%cite%' OR lower(title) LIKE '%combat%'
               OR lower(title) LIKE '%multimodal%'
               OR lower(first_author) LIKE '%stephenson%')
        ORDER BY primary_cell_count DESC
    """)
    return [
        {
            "cytebaseId": row["cytebase_id"],
            "title": row["title"],
            "doi": row["doi"],
            "collectionId": row["collection_id"],
            "primaryCells": int(row["primary_cell_count"]),
            "assaysReportedByPublication": list(row["assay_labels"]),
            "storeUri": row["zarr_uri"],
            "eligibility": "Requires verification of paired count assays in the store",
        }
        for row in rows
    ]


def inspect_metadata(store_uri: str) -> tuple[dict, set[str]]:
    import cytearc

    ds = cytearc.DataStore(
        store_uri,
        zarr_mode="r",
        storage_options={"token": False},
        min_features_per_cell=-1,
        mem_budget="2G",
        nthreads=2,
    )
    columns = set(ds.cells.columns)
    required = {"is_primary_data", "donor_id", "disease", "cell_type"}
    missing = sorted(required - columns)
    result = {
        "storeUri": store_uri,
        "cells": int(ds.cells.N),
        "storedAssays": list(ds.assay_names),
        "missingRequiredColumns": missing,
        "covariateColumns": sorted(
            column
            for column in columns
            if any(
                term in column.lower()
                for term in (
                    "age",
                    "sex",
                    "sample",
                    "visit",
                    "treat",
                    "medic",
                    "assay",
                    "chemistry",
                    "study",
                    "batch",
                    "smok",
                    "disease",
                    "condition",
                )
            )
        ),
    }
    if missing:
        result["eligibility"] = "Missing donor, primary-cell, condition, or label metadata"
        return result, set()
    primary = np.asarray(ds.cells.fetch_all("is_primary_data"), dtype=bool)
    donor = np.asarray(ds.cells.fetch_all("donor_id"), dtype=str)[primary]
    disease = np.asarray(ds.cells.fetch_all("disease"), dtype=str)[primary]
    labels = np.asarray(ds.cells.fetch_all("cell_type"), dtype=str)[primary]
    missing_tokens = {"", "unknown", "na", "nan", "none"}
    donor_counts = Counter(zip(donor.tolist(), disease.tolist(), strict=True))
    donors_by_condition = {
        condition: sorted(set(donor[disease == condition])) for condition in np.unique(disease)
    }
    repeated = sorted(
        value
        for value in set(donor)
        if len({condition for subject, condition in donor_counts if subject == value}) > 1
    )
    result.update(
        {
            "primaryCells": int(primary.sum()),
            "donorsByCondition": donors_by_condition,
            "donorConditionCells": [
                {"donor": subject, "condition": condition, "cells": count}
                for (subject, condition), count in sorted(donor_counts.items())
            ],
            "donorsAcrossConditions": repeated,
            "unknownDonorCells": int(sum(value.lower() in missing_tokens for value in donor)),
            "labelCells": dict(sorted(Counter(labels.tolist()).items())),
            "hasNormalControls": "normal" in donors_by_condition,
            "eligibility": "Metadata characterized; source-study independence needs confirmation",
        }
    )
    sample_columns = [
        column
        for column in ("sample", "sample_id", "sample_name", "Sample ID", "scRNASeq_sample_ID")
        if column in columns
    ]
    result["samplesPerDonor"] = {}
    for column in sample_columns:
        samples = np.asarray(ds.cells.fetch_all(column), dtype=str)[primary]
        result["samplesPerDonor"][column] = {
            subject: sorted(set(samples[donor == subject])) for subject in sorted(set(donor))
        }
    result["covariateCompleteness"] = {}
    for column in (
        "sex",
        "Age",
        "age",
        "age_or_mean_of_age_range",
        "development_stage",
        "assay",
        "Treatment",
        "Cohort",
    ):
        if column in columns:
            values = np.asarray(ds.cells.fetch_all(column), dtype=str)[primary]
            missing = np.isin(np.char.lower(values), list(missing_tokens))
            result["covariateCompleteness"][column] = {
                "knownCells": int((~missing).sum()),
                "missingCells": int(missing.sum()),
                "uniqueKnownValues": int(len(set(values[~missing]))),
            }
    cell_ids = (
        set(np.asarray(ds.cells.fetch_all("observation_joinid"), dtype=str)[primary])
        if "observation_joinid" in columns
        else set()
    )
    result["primarySourceCellIdsAvailable"] = bool(cell_ids)
    result["distinctPrimaryObservationJoinids"] = len(cell_ids) if cell_ids else None
    result["sourceCellIdScope"] = (
        "CELLxGENE observation_joinid hashes obs.index only; identifiers are scoped to the dataset"
    )
    return result, cell_ids


def inspect_cohorts(catalog: cytebase.Catalog, dataset_ids: list[str]) -> dict:
    records = {}
    cell_ids = {}
    for identifier in dataset_ids:
        found = list(
            catalog.query(
                "SELECT cytebase_id, collection_id, doi, title, zarr_uri "
                "FROM datasets WHERE cytebase_id = ?",
                [identifier],
            )
        )
        if len(found) != 1:
            raise ValueError(f"Expected exactly one catalogue entry for {identifier}")
        row = found[0]
        records[identifier], cell_ids[identifier] = inspect_metadata(row["zarr_uri"])
        records[identifier].update({"doi": row["doi"], "collectionId": row["collection_id"]})
    pairs = []
    for left, right in combinations(dataset_ids, 2):
        donors_left = set().union(*records[left].get("donorsByCondition", {}).values())
        donors_right = set().union(*records[right].get("donorsByCondition", {}).values())
        pairs.append(
            {
                "left": left,
                "right": right,
                "samePublication": records[left]["doi"] == records[right]["doi"],
                "sameCollection": records[left]["collectionId"] == records[right]["collectionId"],
                "donorIdCollisions": sorted(donors_left & donors_right),
                "sourceCellIdOverlap": len(cell_ids[left] & cell_ids[right]),
                "sourceCellOverlapMeasured": bool(cell_ids[left] and cell_ids[right]),
                "interpretation": (
                    "Donor IDs are study-local and observation_joinid hashes only obs.index. "
                    "Cross-dataset cell-ID matches can be barcode-name collisions. "
                    "Neither matching nor nonmatching identifiers establish biological overlap "
                    "without original-study and sample provenance."
                ),
            }
        )
    return {"datasets": records, "pairwiseOverlap": pairs}


def lung_donor_overlap() -> dict:
    import h5py

    from experiments.runtime.h5ad import read_column, remote
    from experiments.runtime.lung_atlas import CONTROL_DISEASE, CORE, FULL, ILD_DISEASES

    result = {
        "sources": {
            key: f"https://datasets.cellxgene.cziscience.com/{spec['versionId']}.h5ad"
            for key, spec in (("core", CORE), ("full", FULL))
        },
        "comparison": "Core and eligible Kaminski donors compared in the same full-HLCA namespace",
    }
    with remote(CORE["versionId"]) as handle, h5py.File(handle, "r") as source:
        result["coreDonors"] = sorted(set(read_column(source["obs"], "donor_id")))
    with remote(FULL["versionId"]) as handle, h5py.File(handle, "r") as source:
        obs = source["obs"]
        donors = read_column(obs, "donor_id")
        study = read_column(obs, "study")
        disease = read_column(obs, "disease")
        core = read_column(obs, "core_or_extension") == "core"
        kaminski = study == "Kaminski_2020"
        added = kaminski & np.isin(disease, (*ILD_DISEASES, CONTROL_DISEASE))
        result.update(
            {
                "fullCoreDonors": sorted(set(donors[core])),
                "kaminskiDonors": sorted(set(donors[added])),
                "fullCoreCells": int(core.sum()),
                "addedCells": int(added.sum()),
                "excludedKaminskiDiseases": sorted(set(disease[kaminski & ~added])),
            }
        )
    result["donorIdOverlap"] = sorted(set(result["fullCoreDonors"]) & set(result["kaminskiDonors"]))
    result["uniqueDonors"] = len(set(result["fullCoreDonors"]) | set(result["kaminskiDonors"]))
    return result


def store_count_bytes(store_uri: str) -> dict:
    from huggingface_hub import HfFileSystem

    if not store_uri.startswith("hf://"):
        raise ValueError("The object-size measurement expects a public Hugging Face store")
    filesystem = HfFileSystem(token=False)
    result = {"storeUri": store_uri, "orientations": {}}
    for orientation in ("counts", "countsT"):
        entries = filesystem.find(
            f"{store_uri.removeprefix('hf://').rstrip('/')}/RNA/{orientation}", detail=True
        )
        chunks = [
            int(metadata["size"])
            for name, metadata in entries.items()
            if metadata["type"] == "file" and not name.endswith("zarr.json")
        ]
        if not chunks:
            raise ValueError(f"No count objects found for {orientation}")
        result["orientations"][orientation] = {"bytes": sum(chunks), "objects": len(chunks)}
    result["countChunkBytes"] = sum(value["bytes"] for value in result["orientations"].values())
    return result


def run(payload: dict) -> dict:
    configure_cytearc()
    if payload.get("mode") == "lungDonorOverlap":
        result = lung_donor_overlap()
        write_json("lung_donor_overlap.json", result, directory=RECORDS)
        return result
    if payload.get("mode") == "countBytes":
        result = store_count_bytes(payload["storeUri"])
        write_json("tabula_count_bytes.json", result, directory=RECORDS)
        return result
    catalog = cytebase.Catalog(version=payload.get("release", "2610"), token=False)
    result = {
        "release": payload.get("release", "2610"),
        "multiCohortDiseases": multi_cohort_diseases(catalog),
        "multiStudyTissues": multi_study_tissues(catalog),
        "largeMultimodalCandidates": large_multimodal_datasets(catalog),
    }
    if payload.get("datasetIds"):
        result["cohortMetadata"] = inspect_cohorts(catalog, payload["datasetIds"])
    path = write_json("cohort_discovery.json", result, directory=RECORDS)
    if payload.get("recordUri"):
        from experiments.runtime.object_io import upload_file

        upload_file(path, payload["recordUri"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", action="append", default=[], help="a store whose cell metadata to describe"
    )
    parser.add_argument("--count-bytes", metavar="STORE_URI", help="measure this store's counts")
    parser.add_argument("--lung-donor-overlap", action="store_true")
    args = parser.parse_args()
    if args.count_bytes:
        run({"mode": "countBytes", "storeUri": args.count_bytes})
    elif args.lung_donor_overlap:
        run({"mode": "lungDonorOverlap"})
    else:
        run({"datasetIds": args.dataset})


if __name__ == "__main__":
    main()
