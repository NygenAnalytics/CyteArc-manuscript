import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import cytearc
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from zarr.storage import LocalStore

from experiments.mapping.hlca import selected_counts
from experiments.mapping.label_accuracy import (
    classification_statistics,
    donor_bootstrap,
    donor_confusion_matrices,
)
from experiments.runtime.annotations import (
    MISSING_DONORS,
    harmonize_cell_type,
    ladder_cell_type,
)
from experiments.runtime.common import ROOT, configure_cytearc, write_json
from experiments.runtime.h5ad import _dataframe, _string_dataset, read_column
from experiments.runtime.harness import R2, Objects, Sampler
from experiments.runtime.object_io import _download
from experiments.runtime.recording_store import RecordingStoreWrapper, StoreProbe
from experiments.runtime.steps import _Steps

COHORTS = {
    "blood": {
        "reference": [
            {"study": "Hao_2021", "version": "4078abd1-063d-4b35-aa04-324dddf8244e"},
            {"study": "Yoshida_2022", "version": "4554d5bf-9182-47af-8caf-6b2c93513829"},
        ],
        "query": [
            {"study": "Wilk_2020", "version": "8d7fad46-2f37-4aa0-ac9d-babbc36efb6f"},
            {"study": "Lee_2020", "version": "5e491a70-a1ea-4469-9ffa-9ddff0ec3910"},
        ],
    },
    "kidney": {
        "reference": [
            {"study": "Lake_2025", "version": "255ce81a-eb4e-48ff-9f18-e50455ccb24d"},
            {"study": "Stewart_2019", "version": "c171ca0b-5e46-4260-9dc2-c03486499308"},
        ],
        "query": [{"study": "Reck_2025", "version": "0fe5eee4-380d-4bd9-8735-ada5e03021d9"}],
    },
    "lung": {
        "reference": [{"study": "HLCA", "version": "688185ad-11c2-4172-a53a-f4f1f4076860"}],
        "query": [{"study": "Meyer_2019", "version": "688185ad-11c2-4172-a53a-f4f1f4076860"}],
    },
}
SOURCE_PUBLICATIONS = {
    "Hao_2021": {
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8238499/",
        "cohort": "Eight HVTN087 HIV-vaccine trial participants sampled before and after vaccination",
    },
    "Yoshida_2022": {
        "url": "https://www.nature.com/articles/s41586-021-04345-x",
        "cohort": "UK children and adults with healthy, COVID-19 and post-COVID blood samples",
    },
    "Wilk_2020": {
        "url": "https://www.nature.com/articles/s41591-020-0944-y",
        "cohort": "Seven Stanford COVID-19 patients and six asymptomatic adult controls",
    },
    "Lee_2020": {
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7402635/",
        "cohort": "Korean COVID-19, influenza and healthy PBMC cohort",
    },
    "Lake_2025": {
        "url": "https://www.biorxiv.org/content/10.1101/2025.09.26.678707v2.full",
        "cohort": "KPMP Atlas v2 adult kidney single-cell RNA sequencing",
    },
    "Stewart_2019": {
        "url": "https://pubmed.ncbi.nlm.nih.gov/31604275/",
        "cohort": "Mature human kidney study with sampling across renal compartments",
    },
    "Reck_2025": {
        "url": "https://www.nature.com/articles/s41467-025-59997-4",
        "cohort": "Edinburgh nephrectomy cohort with seven control and five obstructed kidneys",
    },
}


LADDER_METHODS = (
    "unintegrated",
    "integrated",
    "seurat_map",
    "symphony",
    "celltypist",
    "celltypist_hvg",
)
DGCMATRIX_NONZERO_LIMIT = 2**31 - 1
# The design values fixed before the ladder runs, as the reference ladder record states them.
LADDER_DESIGN = (
    "referenceStudies",
    "queryStudies",
    "referenceSizes",
    "queryCellsPerStudy",
    "minimumTypeCells",
    "seed",
)


class InputWriter:
    """Write a prepared CSR H5AD block by block into preallocated contiguous arrays."""

    def __init__(self, path, cells, genes, nonzeros):
        if cells < 1 or genes < 1 or nonzeros < 0:
            raise ValueError(
                f"A prepared input needs cells and genes; got {cells} cells, {genes} genes"
            )
        self.path, self.shape, self.nonzeros = Path(path), (int(cells), int(genes)), int(nonzeros)
        self.rows = self.values = 0
        self.handle = h5py.File(self.path, "w")
        self.handle.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        matrix = self.handle.create_group("X")
        matrix.attrs.update(
            {"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": self.shape}
        )
        self.data = matrix.create_dataset("data", shape=(self.nonzeros,), dtype="float32")
        self.indices = matrix.create_dataset("indices", shape=(self.nonzeros,), dtype="int64")
        self.indptr = matrix.create_dataset("indptr", shape=(self.shape[0] + 1,), dtype="int64")

    def append(self, counts):
        # Row slices of a CSR block can carry int32 pointers; offsets past 2^31 need int64.
        pointers = np.asarray(counts.indptr, dtype=np.int64)
        rows, low, high = counts.shape[0], int(pointers[0]), int(pointers[-1])
        values = high - low
        if (
            counts.shape[1] != self.shape[1]
            or self.rows + rows > self.shape[0]
            or self.values + values > self.nonzeros
        ):
            raise ValueError(f"A count block does not fit the preallocated input {self.path.name}")
        self.data[self.values : self.values + values] = counts.data[low:high]
        self.indices[self.values : self.values + values] = counts.indices[low:high]
        self.indptr[self.rows : self.rows + rows + 1] = pointers - low + self.values
        self.rows += rows
        self.values += values

    def close(self, obs, feature_ids, feature_names):
        try:
            if (self.rows, self.values, len(obs)) != (self.shape[0], self.nonzeros, self.shape[0]):
                raise ValueError(
                    f"{self.path.name} received {self.rows} rows and {self.values} values; "
                    f"expected {self.shape[0]} rows, {self.nonzeros} values and one obs row per cell"
                )
            frame = _dataframe(self.handle, "obs", list(obs.columns))
            _string_dataset(frame, "_index", obs.index.to_numpy(dtype=str))
            for column in obs:
                _string_dataset(frame, column, obs[column].to_numpy(dtype=str))
            features = _dataframe(self.handle, "var", ["feature_name"])
            _string_dataset(features, "_index", feature_ids)
            _string_dataset(features, "feature_name", feature_names)
        finally:
            self.handle.close()


def prepared_observations(prefix, studies, donors, labels, author_labels, ids, **extra):
    frame = pd.DataFrame(
        {
            "study": studies,
            "donor_id": donors,
            "cell_type": labels,
            "author_label": author_labels,
            **extra,
        },
        index=[f"{prefix}::{value}" for value in ids],
    )
    frame["ann_level_3"] = frame["cell_type"]
    frame["ann_finest_level"] = frame["cell_type"]
    return frame


def input_summary(obs, nonzeros, path):
    return {
        "cells": len(obs),
        "donors": obs[["study", "donor_id"]]
        .drop_duplicates()
        .sort_values(["study", "donor_id"])
        .to_dict(orient="records"),
        "studies": obs.study.value_counts().to_dict(),
        "labels": obs.cell_type.value_counts().to_dict(),
        "nonzeroCounts": int(nonzeros),
        "bytes": path.stat().st_size,
        "cellIdSha256": hashlib.sha256("\0".join(obs.index).encode()).hexdigest(),
    }


def prepare_inputs(payload, work):
    tissue = payload["tissue"]
    specifications = payload.get("cohorts", COHORTS[tissue])
    loaded = {"reference": [], "query": []}
    records = []
    for role, sources in specifications.items():
        for source in sources:
            path = work / f"{source['version']}.h5ad"
            url = source.get(
                "url", f"https://datasets.cellxgene.cziscience.com/{source['version']}.h5ad"
            )
            if not path.exists():
                _download(url, path)
            with h5py.File(path, "r") as handle:
                obs = handle["obs"]
                matrix_key = "raw/X" if "raw/X" in handle else "X"
                var = handle["raw/var"] if matrix_key == "raw/X" else handle["var"]
                feature_ids = read_column(var, var.attrs["_index"])
                feature_names = read_column(var, "feature_name")
                original = read_column(obs, "ann_level_3" if tissue == "lung" else "cell_type")
                label_map = {
                    label: harmonize_cell_type(tissue, label) for label in np.unique(original)
                }
                labels = np.array([label_map[label] or "" for label in original])
                donors = read_column(obs, "donor_id")
                ids = read_column(obs, obs.attrs["_index"])
                source_cell_ids = read_column(obs, "observation_joinid")
                if tissue == "lung":
                    studies = read_column(obs, "dataset")
                    query_study = payload.get("queryStudy", "Meyer_2019")
                    selected = studies == query_study if role == "query" else studies != query_study
                else:
                    studies = np.repeat(source["study"], len(ids))
                    primary = read_column(obs, "is_primary_data")
                    if primary.dtype.kind != "b":
                        raise TypeError("is_primary_data must be stored as a boolean array")
                    selected = np.asarray(primary, dtype=bool)
                selected &= labels != ""
                selected &= ~np.isin(
                    np.char.lower(donors.astype(str)), ["", "nan", "unknown", "none"]
                )
                data = {
                    "path": path,
                    "matrixKey": matrix_key,
                    "featureIds": feature_ids,
                    "featureNames": feature_names,
                    "labels": labels,
                    "originalLabels": original,
                    "donors": donors,
                    "studies": studies,
                    "ids": ids,
                    "sourceCellIds": source_cell_ids,
                    "selected": selected,
                    "source": source,
                }
                loaded[role].append(data)
                records.append(
                    {
                        "role": role,
                        "study": source["study"],
                        "sourceUrl": url,
                        "sourceCells": len(ids),
                        "eligibleCells": int(selected.sum()),
                        "labelMap": label_map,
                        "excludedLabelCells": {
                            label: int(np.sum(original == label))
                            for label, value in label_map.items()
                            if value is None
                        },
                        "primaryOnly": tissue != "lung",
                    }
                )
                if source["study"] in SOURCE_PUBLICATIONS:
                    records[-1]["primaryPublication"] = SOURCE_PUBLICATIONS[source["study"]]
    query_donors = set().union(*(set(item["donors"][item["selected"]]) for item in loaded["query"]))
    reference_donors = set().union(
        *(set(item["donors"][item["selected"]]) for item in loaded["reference"])
    )
    raw_donor_collisions = sorted(reference_donors & query_donors)
    if tissue == "lung":
        for item in loaded["reference"]:
            item["selected"] &= ~np.isin(item["donors"], list(query_donors))
    all_inputs = loaded["reference"] + loaded["query"]
    common = set.intersection(*(set(item["featureIds"]) for item in all_inputs))
    feature_ids = np.array([value for value in all_inputs[0]["featureIds"] if value in common])
    feature_names = all_inputs[0]["featureNames"][np.isin(all_inputs[0]["featureIds"], feature_ids)]
    if len(feature_ids) < 2000:
        raise ValueError(f"Only {len(feature_ids)} genes are shared by the fixed cohorts")
    source_identities = set()
    summaries = {}
    for role, sources in loaded.items():
        parts, metadata = [], []
        for item in sources:
            rng = np.random.default_rng(4444)
            rows = np.flatnonzero(item["selected"])
            cap = payload.get(f"{role}CellsPerStudy")
            if cap is not None:
                kept = []
                for study in np.unique(item["studies"][rows]):
                    candidates = rows[item["studies"][rows] == study]
                    kept.extend(rng.choice(candidates, min(len(candidates), cap), replace=False))
                rows = np.sort(np.asarray(kept, dtype=np.int64))
            if not len(rows):
                raise ValueError(f"No eligible cells for {role}/{item['source']['study']}")
            identities = set(item["sourceCellIds"][rows])
            if len(identities) != len(rows) or source_identities.intersection(identities):
                raise ValueError("Selected source-cell identities overlap")
            source_identities.update(identities)
            positions = {value: i for i, value in enumerate(item["featureIds"])}
            columns = np.array([positions[value] for value in feature_ids])
            with h5py.File(item["path"], "r") as source:
                parts.append(selected_counts(source[item["matrixKey"]], rows, columns))
            metadata.append(
                prepared_observations(
                    item["source"]["study"],
                    item["studies"][rows],
                    item["donors"][rows],
                    item["labels"][rows],
                    item["originalLabels"][rows],
                    item["ids"][rows],
                )
            )
        obs = pd.concat(metadata)
        nonzeros = sum(part.nnz for part in parts)
        writer = InputWriter(work / f"{role}.h5ad", len(obs), len(feature_ids), nonzeros)
        for part in parts:
            writer.append(part)
        writer.close(obs, feature_ids, feature_names)
        summaries[role] = input_summary(obs, nonzeros, work / f"{role}.h5ad")
        del parts
        gc.collect()
    return {
        "sources": records,
        "referenceQuerySourceCellIdOverlap": 0,
        "rawDonorIdCollisions": raw_donor_collisions,
        "donorIdentityScope": "Shared HLCA identifiers with overlapping donors excluded"
        if tissue == "lung"
        else "Donor identifiers are scoped to each primary study; source publications document recruitment and sample provenance",
        "sharedGenes": len(feature_ids),
        "parts": summaries,
    }


def ladder_design(payload: dict) -> dict:
    design = payload.get("referenceLadder")
    if not isinstance(design, dict):
        raise ValueError("referenceLadder must be an object describing sources, studies and sizes")
    reference, query = design.get("referenceStudies"), design.get("queryStudies")
    sources = design.get("sources")
    if (
        not isinstance(reference, list)
        or not isinstance(query, list)
        or not reference
        or not query
        or len(set(reference + query)) != len(reference) + len(query)
    ):
        raise ValueError("referenceStudies and queryStudies must be nonempty disjoint study lists")
    if not isinstance(sources, dict) or set(sources) != set(reference + query):
        raise ValueError("sources must give one H5AD location for every reference and query study")
    sizes = design.get("referenceSizes")
    finite = [size for size in sizes or [] if size is not None]
    if (
        not sizes
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in finite)
        or finite != sorted(set(finite))
        or sizes.count(None) > 1
        or (None in sizes and sizes[-1] is not None)
    ):
        raise ValueError(
            "referenceSizes must be increasing positive cell counts, optionally ending with null "
            f"for the whole eligible pool; got {sizes}"
        )
    within = design.get("sizesWithinDgCMatrixLimit", [])
    if not isinstance(within, list) or not set(within) <= set(finite):
        raise ValueError(
            "sizesWithinDgCMatrixLimit must list finite referenceSizes that have to stay within "
            f"{DGCMATRIX_NONZERO_LIMIT} nonzero counts; got {within!r}"
        )
    settings = {
        "queryCellsPerStudy": design.get("queryCellsPerStudy"),
        "minimumTypeCells": design.get("minimumTypeCells", 100),
        "seed": design.get("seed", 4444),
        "blockNonzeros": design.get("blockNonzeros", 64_000_000),
    }
    for key, value in settings.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer; got {value!r}")
    return {
        "sources": sources,
        "referenceStudies": reference,
        "queryStudies": query,
        "referenceSizes": sizes,
        "sizesWithinDgCMatrixLimit": within,
        **settings,
    }


def read_ladder_source(path: Path, study: str) -> dict:
    with h5py.File(path, "r") as handle:
        obs, var, matrix = handle["obs"], handle["var"], handle["X"]
        if matrix.attrs["encoding-type"] != "csr_matrix":
            raise ValueError(f"{path.name} must store CSR counts")
        studies = read_column(obs, "dataset")
        if not np.all(studies == study):
            raise ValueError(f"{path.name} holds cells outside study {study}")
        author = read_column(obs, "orig_cell_type")
        names, codes = np.unique(author, return_inverse=True)
        mapped = np.array([ladder_cell_type(name) for name in names], dtype=object)
        labels = mapped[codes].astype(str)
        donors = read_column(obs, "orig_donor_id").astype(str)
        primary = read_column(obs, "orig_is_primary_data")
        if primary.dtype.kind != "b":
            raise TypeError(f"orig_is_primary_data must be boolean in {path.name}")
        pointers = np.asarray(matrix["indptr"][:], dtype=np.int64)
        shape = tuple(int(value) for value in matrix.attrs["shape"])
        if len(pointers) != shape[0] + 1 or len(studies) != shape[0]:
            raise ValueError(f"{path.name} has inconsistent cell counts")
        known = ~np.isin(np.char.lower(np.char.strip(donors)), list(MISSING_DONORS))
        join_ids = read_column(obs, "orig_observation_joinid").astype(str)
        if len(np.unique(join_ids)) != len(join_ids):
            raise ValueError(f"Source cell identities repeat within {path.name}")
        return {
            "path": path,
            "shape": shape,
            "featureIds": read_column(var, var.attrs["_index"]).astype(str),
            "featureNames": read_column(var, "gene_short_name").astype(str),
            "ids": read_column(obs, obs.attrs["_index"]).astype(str),
            "authorLabels": author.astype(str),
            "labels": labels,
            "donors": donors,
            "disease": read_column(obs, "orig_disease").astype(str),
            "primary": primary,
            "knownDonor": known,
            "rowNonzeros": np.diff(pointers),
            "labelMap": dict(zip(names.tolist(), mapped.tolist(), strict=True)),
        }


def stratum_ranks(strata: np.ndarray, seed: int) -> tuple:
    """Stratum codes and sizes, and each cell's position within its stratum in a seeded order."""
    total = len(strata)
    priority = np.random.default_rng(seed).permutation(total)
    _, codes, counts = np.unique(strata, return_inverse=True, return_counts=True)
    order = np.lexsort((priority, codes))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rank = np.empty(total, dtype=np.int64)
    rank[order] = np.arange(total) - starts[codes[order]]
    return codes, counts, rank


def nested_selection(ranks: tuple, size: int | None) -> np.ndarray:
    """Cells of a reference with round(size x share) cells of every stratum; None takes all."""
    codes, counts, rank = ranks
    if size is None:
        return np.ones(len(rank), dtype=bool)
    if size > len(rank):
        raise ValueError(f"A {size}-cell reference exceeds the {len(rank)}-cell eligible pool")
    quota = np.floor(size * counts / len(rank) + 0.5).astype(np.int64)
    return rank < quota[codes]


def largest_reference_within(ranks: tuple, row_nonzeros: np.ndarray, limit: int) -> int:
    """Largest nested reference size whose nonzero count stays within the limit."""
    low, high = 0, len(row_nonzeros)
    if int(row_nonzeros.sum()) <= limit:
        return high
    while low < high:
        middle = (low + high + 1) // 2
        if int(row_nonzeros[nested_selection(ranks, middle)].sum()) <= limit:
            low = middle
        else:
            high = middle - 1
    return low


def stream_rows(path: Path, wanted: dict, writers: dict, block_nonzeros: int) -> None:
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        pointers = np.asarray(matrix["indptr"][:], dtype=np.int64)
        cells, genes = (int(value) for value in matrix.attrs["shape"])
        start = 0
        while start < cells:
            limit = pointers[start] + block_nonzeros
            stop = min(cells, max(start + 1, int(np.searchsorted(pointers, limit, "right")) - 1))
            chosen = {}
            for name, rows in wanted.items():
                left, right = np.searchsorted(rows, [start, stop])
                if right > left:
                    chosen[name] = rows[left:right] - start
            if chosen:
                low, high = int(pointers[start]), int(pointers[stop])
                block = csr_matrix(
                    (
                        matrix["data"][low:high],
                        matrix["indices"][low:high],
                        pointers[start : stop + 1] - low,
                    ),
                    shape=(stop - start, genes),
                )
                for name, positions in chosen.items():
                    writers[name].append(block[positions])
            start = stop


def ladder_parts(design: dict, sources: dict) -> dict:
    """Vocabulary, rare types and the rows and exact nonzero counts of every part."""
    reference, query = design["referenceStudies"], design["queryStudies"]
    for item in sources.values():
        item["eligible"] = item["primary"] & item["knownDonor"] & (item["labels"] != "")
    pool_labels = np.concatenate([sources[s]["labels"][sources[s]["eligible"]] for s in reference])
    pool_counts = pd.Series(pool_labels).value_counts()
    smallest = next((size for size in design["referenceSizes"] if size), len(pool_labels))
    expected = pool_counts * smallest / len(pool_labels)
    vocabulary = sorted(expected.index[expected >= design["minimumTypeCells"]].tolist())
    for item in sources.values():
        item["selected"] = item["eligible"] & np.isin(item["labels"], vocabulary)
    pool_rows = {study: np.flatnonzero(sources[study]["selected"]) for study in reference}
    kept = pool_counts[vocabulary]
    frequencies = kept / kept.sum()
    rare_types = {
        "belowOnePercent": sorted(frequencies.index[frequencies < 0.01].tolist()),
        "belowOneTenthPercent": sorted(frequencies.index[frequencies < 0.001].tolist()),
        "referencePoolFrequencies": frequencies.to_dict(),
        "rule": "Frequency in the whole eligible reference pool, fixed before any mapping run; "
        "the 0.1% tier is a subset of the 1% tier",
    }
    strata = np.concatenate(
        [
            index * len(vocabulary)
            + np.searchsorted(vocabulary, sources[study]["labels"][pool_rows[study]])
            for index, study in enumerate(reference)
        ]
    )
    ranks = stratum_ranks(strata, design["seed"])
    selections = {size: nested_selection(ranks, size) for size in design["referenceSizes"]}
    finite = [size for size in design["referenceSizes"] if size is not None]
    for smaller, larger in zip(finite, finite[1:] + [None], strict=True):
        if larger in selections and np.any(selections[smaller] & ~selections[larger]):
            raise AssertionError("Reference subsets are not nested")
    offsets = np.cumsum([0] + [len(pool_rows[study]) for study in reference])
    parts = {}
    for size in design["referenceSizes"]:
        name = "reference_all" if size is None else f"reference_{size}"
        parts[name] = {
            study: pool_rows[study][selections[size][offsets[index] : offsets[index + 1]]]
            for index, study in enumerate(reference)
        }
    parts["query"] = {}
    for study in query:
        rows = np.flatnonzero(sources[study]["selected"])
        if len(rows) > design["queryCellsPerStudy"]:
            rng = np.random.default_rng(design["seed"])
            rows = np.sort(rng.choice(rows, design["queryCellsPerStudy"], replace=False))
        parts["query"][study] = rows
    nonzeros = {
        name: {
            "cells": int(sum(len(value) for value in rows.values())),
            "nonzeroCounts": int(
                sum(int(sources[s]["rowNonzeros"][value].sum()) for s, value in rows.items())
            ),
        }
        for name, rows in parts.items()
    }
    pool_nonzeros = np.concatenate(
        [sources[study]["rowNonzeros"][pool_rows[study]] for study in reference]
    )
    largest = largest_reference_within(ranks, pool_nonzeros, DGCMATRIX_NONZERO_LIMIT)
    within = {
        "cells": int(nested_selection(ranks, largest).sum()),
        "nonzeroCounts": int(pool_nonzeros[nested_selection(ranks, largest)].sum()),
        "requestedSize": largest,
    }
    return {
        "vocabulary": vocabulary,
        "excludedTypes": {
            label: int(count) for label, count in pool_counts.items() if label not in vocabulary
        },
        "rareTypes": rare_types,
        "poolRows": pool_rows,
        "parts": parts,
        "partNonzeroCounts": nonzeros,
        "referencePoolCells": int(kept.sum()),
        "referencePoolNonzeroCounts": int(pool_nonzeros.sum()),
        "largestReferenceWithinDgCMatrixLimit": within,
    }


def require_within_dgcmatrix_limit(design: dict, plan: dict) -> None:
    """Fail before writing when a part meant for R comparators exceeds the dgCMatrix limit."""
    within = plan["largestReferenceWithinDgCMatrixLimit"]
    for size in design["sizesWithinDgCMatrixLimit"]:
        count = plan["partNonzeroCounts"][f"reference_{size}"]["nonzeroCounts"]
        if count > DGCMATRIX_NONZERO_LIMIT:
            raise ValueError(
                f"reference_{size} has {count} nonzero counts, above the dgCMatrix limit of "
                f"{DGCMATRIX_NONZERO_LIMIT}; the largest nested reference within it is "
                f"{within['requestedSize']} cells ({within['nonzeroCounts']} nonzero counts)"
            )


def prepare_reference_ladder(payload: dict, work: Path) -> dict:
    """Nested reference subsets of increasing size and one fixed query from other studies."""

    design = ladder_design(payload)
    reference, query = design["referenceStudies"], design["queryStudies"]
    objects = Objects(payload.get("r2EndpointUrl"))
    sources = {}
    for study in reference + query:
        path = work / "sources" / f"{study}.h5ad"
        objects.download(design["sources"][study], path)
        sources[study] = read_ladder_source(path, study)
    feature_ids = sources[reference[0]]["featureIds"]
    feature_names = sources[reference[0]]["featureNames"]
    if any(not np.array_equal(item["featureIds"], feature_ids) for item in sources.values()):
        raise ValueError("The source H5ADs do not share one gene order")
    plan = ladder_parts(design, sources)
    vocabulary, parts, pool_rows = plan["vocabulary"], plan["parts"], plan["poolRows"]
    record = {
        key: plan[key] for key in ("partNonzeroCounts", "largestReferenceWithinDgCMatrixLimit")
    }
    write_json("ladder_plan.json", record, directory=work)
    if payload.get("recordUri"):
        objects.put_json(payload["recordUri"].removesuffix(".json") + "/plan.json", record)
    require_within_dgcmatrix_limit(design, plan)
    writers = {}
    for name, rows in parts.items():
        cells = sum(len(value) for value in rows.values())
        nonzeros = plan["partNonzeroCounts"][name]["nonzeroCounts"]
        writers[name] = InputWriter(work / f"{name}.h5ad", cells, len(feature_ids), nonzeros)
    for study in reference + query:
        wanted = {name: rows[study] for name, rows in parts.items() if study in rows}
        stream_rows(sources[study]["path"], wanted, writers, design["blockNonzeros"])
        sources[study]["path"].unlink()
    summaries = {}
    for name, rows in parts.items():
        obs = pd.concat(
            [
                prepared_observations(
                    study,
                    np.repeat(study, len(value)),
                    sources[study]["donors"][value],
                    sources[study]["labels"][value],
                    sources[study]["authorLabels"][value],
                    sources[study]["ids"][value],
                    disease=sources[study]["disease"][value],
                )
                for study, value in rows.items()
            ]
        )
        writers[name].close(obs, feature_ids, feature_names)
        summaries[name] = input_summary(obs, writers[name].nonzeros, writers[name].path)
        if name == "query":
            obs.rename(columns={"cell_type": "truth"}).rename_axis("cell").reset_index()[
                ["cell", "study", "donor_id", "truth", "author_label", "disease"]
            ].to_csv(work / "query_cells.csv", index=False)
        del obs
        gc.collect()
    query_donors = {
        donor for study, rows in parts["query"].items() for donor in sources[study]["donors"][rows]
    }
    reference_donors = {
        donor for study, rows in pool_rows.items() for donor in sources[study]["donors"][rows]
    }
    records = []
    for study in reference + query:
        item = sources[study]
        records.append(
            {
                "role": "reference" if study in reference else "query",
                "study": study,
                "sourceUri": design["sources"][study],
                "sourceCells": item["shape"][0],
                "sourceNonzeroCounts": int(item["rowNonzeros"].sum()),
                "labelMap": item["labelMap"],
                "excludedCells": {
                    "notPrimary": int((~item["primary"]).sum()),
                    "withoutDonor": int((~item["knownDonor"]).sum()),
                    "withoutCoarseLabel": int((item["labels"] == "").sum()),
                    "outsideVocabulary": int((item["eligible"] & ~item["selected"]).sum()),
                },
                "eligibleCells": int(item["selected"].sum()),
            }
        )
    return {
        "design": "Nested reference subsets stratified by study and coarse label, with one fixed "
        "query sampled from studies outside the reference",
        "referenceLadder": {key: value for key, value in design.items() if key != "sources"},
        "sources": records,
        "vocabulary": vocabulary,
        "vocabularyRule": f"Coarse types expected to have at least {design['minimumTypeCells']} "
        "cells in the smallest reference; other types are removed from reference and query",
        "excludedTypes": plan["excludedTypes"],
        "rareTypes": plan["rareTypes"],
        "referencePoolCells": plan["referencePoolCells"],
        "referencePoolNonzeroCounts": plan["referencePoolNonzeroCounts"],
        "dgCMatrixNonzeroLimit": DGCMATRIX_NONZERO_LIMIT,
        "largestReferenceWithinDgCMatrixLimit": plan["largestReferenceWithinDgCMatrixLimit"],
        "referenceQuerySourceCellIdOverlap": 0,
        "sourceCellIdentityScope": "Study plus observation_joinid; reference and query are "
        "different studies, and each study's identifiers are checked to be unique",
        "rawDonorIdCollisions": sorted(reference_donors & query_donors),
        "donorIdentityScope": "Donor identifiers are scoped to each primary study; reference and "
        "query come from different studies",
        "sharedGenes": len(feature_ids),
        "parts": summaries,
        "queryCells": {"file": "query_cells.csv", "cells": summaries["query"]["cells"]},
    }


def evaluate_predictions(frame, label_order, rare_types, *, coverage_levels=(0.5, 0.75, 0.9, 1.0)):
    if frame.empty or frame["cell"].duplicated().any():
        raise ValueError("Predictions require a nonempty set of unique cell IDs")
    if len(label_order) != len(set(label_order)) or "unassigned" not in label_order:
        raise ValueError("The label vocabulary must be unique and include unassigned")
    if not np.isfinite(frame["confidence"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction confidences must be finite")
    unknown = (set(frame["truth"]) | set(frame["prediction"])) - set(label_order)
    if unknown:
        raise ValueError(
            f"Predictions contain labels outside the fixed vocabulary: {sorted(unknown)}"
        )
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in coverage_levels):
        raise ValueError("Coverage levels must be finite fractions between zero and one")
    results = {}
    for study, cells in frame.groupby("study", sort=True):
        truth = cells["truth"].to_numpy(dtype=str)
        predictions = cells["prediction"].to_numpy(dtype=str)
        evaluated = set(truth)
        matrix = confusion_matrix(truth, predictions, labels=label_order)
        if int(matrix.sum()) != len(cells):
            raise AssertionError("A query cell was omitted from the confusion matrix")
        summary = classification_statistics(matrix, label_order, evaluated)
        summary["assignmentCoverage"] = float(np.mean(predictions != "unassigned"))
        summary["rareTypes"] = {
            label: summary["perType"][label] for label in rare_types if label in evaluated
        }
        donors = donor_confusion_matrices(cells, label_order)
        summary["donorBootstrap95CI"] = donor_bootstrap(
            np.asarray([row["confusion"] for row in donors]), label_order, evaluated
        )
        available = np.flatnonzero(predictions != "unassigned")
        priority = np.random.default_rng(4444).permutation(len(cells))
        confidence = cells["confidence"].to_numpy(dtype=float)
        ranked = available[np.lexsort((priority[available], -confidence[available]))]
        curves = []
        for coverage in coverage_levels:
            size = coverage * len(cells)
            nearest = int(round(size))
            requested_count = (
                nearest if np.isclose(size, nearest, rtol=0, atol=1e-8) else int(np.ceil(size))
            )
            selected = ranked[: min(len(ranked), requested_count)]
            accepted = np.zeros(len(cells), dtype=bool)
            accepted[selected] = True
            classified = np.where(accepted, predictions, "unassigned")
            values = classification_statistics(
                confusion_matrix(truth, classified, labels=label_order), label_order, evaluated
            )
            curves.append(
                {
                    "requestedCoverage": coverage,
                    "actualCoverage": float(accepted.mean()),
                    "requestedCoverageAchieved": len(selected) == requested_count,
                    "accuracyAssigned": float(np.mean(predictions[accepted] == truth[accepted]))
                    if accepted.any()
                    else None,
                    "macroF1AllCells": values["macroF1"],
                }
            )
        summary["coverageCurve"] = curves
        summary["donorConfusionMatrices"] = donors
        results[study] = summary
    return {
        "studies": results,
        "labelOrder": label_order,
        "rareTypeRule": "Reference frequency below 1% with at least 20 reference cells",
        "bootstrapUnit": "Donor within each query study; all donor cells retained together",
        "coverageRule": "Descending confidence; ties use a fixed seeded cell permutation; unassigned cells remain unassigned",
    }


def evaluate(payload):
    from experiments.mapping.cross_study_evaluation import (
        MAPPING_METHODS,
        completed_records,
        mapping_values,
        record_hash,
    )

    storage = R2(payload["r2EndpointUrl"])
    prefix = payload["mappingPrefix"].rstrip("/")
    directory = ROOT / "experiments/mapping/records/cross_study"
    inputs = []
    for tissue in COHORTS:
        for method in MAPPING_METHODS:
            uri = f"{prefix}/{tissue}/{method}.json"
            record = storage.get_json(uri)
            native = method in {"unintegrated", "integrated"}
            if record.get("status") != ("completed" if native else "ok"):
                raise ValueError(f"The requested mapping attempt is incomplete: {tissue}/{method}")
            path = directory / tissue / f"{method}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n")
            if native:
                prediction_uri = record["result"]["arms"][method]["predictionsUri"]
                predictions = path.with_suffix("") / f"{method}_predictions.csv"
            else:
                prediction_uri = uri.removesuffix(".json") + "_cells.csv"
                predictions = path.with_name(f"{method}_cells.csv")
            storage.download(prediction_uri, predictions)
            inputs.append(
                {
                    "relativePath": str(path.relative_to(directory)),
                    "uri": uri,
                    "sha256": record_hash(record),
                    "predictionsUri": prediction_uri,
                }
            )
    mapping, coverage, missing = mapping_values(completed_records(directory), directory)
    if missing:
        raise ValueError("Mapping evaluation is incomplete: " + "; ".join(missing))
    return {
        "mode": "evaluation",
        "inputRecords": inputs,
        "mapping": mapping,
        "mappingAtMatchedCoverage": coverage,
        "missingEvidence": [],
        "evaluationSourceSha256": hashlib.sha256(
            (ROOT / "experiments/mapping/cross_study_evaluation.py").read_bytes()
        ).hexdigest(),
    }


def mapping_outcome(record: dict, *, native: bool) -> str:
    """Name how one mapping attempt ended, from its CyteArc envelope or comparator record."""
    status = record.get("status")
    if native:
        if status == "completed":
            return "completed"
        if status == "failed":
            return "out of memory" if record.get("errorType") == "MemoryError" else "error"
        return "incomplete"
    if status == "ok":
        return "completed"
    if "2^31" in (record.get("log") or ""):
        return "dgCMatrix nonzero limit"
    if status == "timedOut":
        return "timed out"
    if status == "outOfMemory":
        return "out of memory"
    if status == "killed":
        limit = int(record.get("container", {}).get("memoryMb") or 0) * 1024**2
        peak = int(record.get("peakCgroupBytes") or 0)
        return "out of memory" if limit and peak >= 0.85 * limit else "killed"
    if status == "interrupted":
        return "interrupted"
    if status == "attempting":
        return "incomplete"
    return "error"


def evaluate_ladder(payload: dict) -> dict:
    """The reference ladder record: outcome, memory, persisted reference size, step times and
    per-donor confusion counts of every method at every reference size."""
    from experiments.mapping.cross_study_evaluation import (
        external_costs,
        native_costs,
        read_external_predictions,
        read_native_predictions,
    )

    objects = Objects(payload.get("r2EndpointUrl"))
    manifest = objects.get_json(payload["inputManifestUri"])
    prefix = payload["ladderPrefix"].rstrip("/")
    work = Path(payload.get("workDirectory", "/tmp/mapping_ladder_evaluation"))
    methods = payload.get("methods", list(LADDER_METHODS))
    if not set(methods) <= set(LADDER_METHODS):
        raise ValueError(f"methods must be drawn from {LADDER_METHODS}; got {methods}")
    query = manifest["parts"]["query"]
    cells_path = work / manifest["queryCells"]["file"]
    objects.download(manifest["queryCells"]["uri"], cells_path)
    with cells_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["queryCells"]["sha256"]:
            raise ValueError("The prepared query cell table changed")
    metadata = pd.read_csv(cells_path, keep_default_na=False, dtype=str)
    if hashlib.sha256("\0".join(metadata["cell"]).encode()).hexdigest() != query["cellIdSha256"]:
        raise ValueError("The query cell table does not match the prepared query")
    label_order = sorted(manifest["vocabulary"]) + ["unassigned"]
    parts = payload.get(
        "referenceParts", [name for name in manifest["parts"] if name.startswith("reference")]
    )
    rows = []
    failures = {}
    for part in parts:
        reference = manifest["parts"][part]
        inputs = {"parts": {"reference": reference, "query": query}}
        for method in methods:
            uri = f"{prefix}/{part}/{method}.json"
            native = method in {"unintegrated", "integrated"}
            row = {
                "referencePart": part,
                "referenceCells": reference["cells"],
                "referenceNonzeroCounts": reference["nonzeroCounts"],
                "method": method,
            }
            rows.append(row)
            if not objects.exists(uri):
                row["outcome"] = "not attempted"
                if method in failures:
                    row["notAttemptedReason"] = "{} with the {} reference".format(*failures[method])
                continue
            record = objects.get_json(uri)
            row["outcome"] = mapping_outcome(record, native=native)
            if row["outcome"] != "completed":
                failures.setdefault(method, (row["outcome"], part))
            if native:
                detail = record.get("result", {})
                row["peakContainerBytes"] = detail.get("peakContainerBytes")
                if row["outcome"] != "completed":
                    continue
                costs = native_costs(detail, method)
                written = detail["arms"][method]["registrationIo"]["writeTransferredBytes"]
                row["persistedReferenceBytes"] = written
                predictions = work / part / f"{method}_predictions.csv"
                objects.download(detail["arms"][method]["predictionsUri"], predictions)
                frame = read_native_predictions(predictions, inputs)
            else:
                row.update(
                    {
                        "status": record.get("status"),
                        "failedStep": record.get("failedStep"),
                        "containerMemoryBytes": int(record["container"]["memoryMb"]) * 1024**2,
                        "peakContainerBytes": record.get("peakCgroupBytes"),
                    }
                )
                if row["outcome"] != "completed":
                    continue
                costs = external_costs(record)
                row["persistedReferenceBytes"] = (
                    sum(record["referenceModelBytes"].values())
                    if "referenceModelBytes" in record
                    else record["referenceRdsBytes"] + record["referenceIndexBytes"]
                )
                predictions = work / part / f"{method}_cells.csv"
                objects.download(uri.removesuffix(".json") + "_cells.csv", predictions)
                frame = read_external_predictions(predictions, metadata)
            row["stepSeconds"] = {name: group["seconds"] for name, group in costs["groups"].items()}
            # Donors are scoped to their study, so the pooled query names them study::donor.
            pooled = frame.assign(donor_id=frame["study"] + "::" + frame["donor_id"])
            row["donorConfusion"] = [
                {"donor": donor["donor"], "confusion": donor["confusion"]}
                for donor in donor_confusion_matrices(pooled, label_order)
            ]
    return {
        "design": {key: manifest["referenceLadder"][key] for key in LADDER_DESIGN},
        "queryCells": query["cells"],
        "vocabulary": manifest["vocabulary"],
        "rareTypes": manifest["rareTypes"],
        "confusionLabels": label_order,
        "rows": rows,
        "stepSecondsDefinition": "Wall seconds of each step group: referencePreparation builds "
        "the reference from the prepared H5AD; referenceRegistration (CyteArc) marks the finished "
        "analysis as a mapping reference; referencePersistence and preparedReferenceLoad "
        "(comparators) save and load the trained reference; queryAnnotation reads the query and "
        "transfers labels",
        "persistedReferenceDefinition": "CyteArc: bytes written while registering the mapping "
        "reference from existing analysis artifacts; CellTypist: saved model files; Seurat and "
        "Symphony: uncompressed RDS plus Seurat's saved neighbor index",
        "outcomeRule": "Comparator records: a log naming the 2^31 - 1 limit is the dgCMatrix "
        "nonzero limit; outOfMemory, or a kill at 85% or more of the container memory, is out of "
        "memory; timedOut is the workflow time limit. CyteArc envelopes: failed with MemoryError "
        "is out of memory; a record still running after its job ended is incomplete",
        "confusionDefinition": "Rows are author labels and columns transferred labels, in "
        "confusionLabels order; one matrix per query donor, named study::donor",
    }


# The analysis that makes each held-out reference: no filtering, 2,000 variable genes, 30 principal components and 30
# neighbors, without UMAP, clustering or the optional stages.
REFERENCE_SETTINGS = {
    "filtering": False,
    "hvg_count": 2000,
    "pca_dims": 30,
    "neighbors_k": 30,
    "umap": False,
    "leiden": False,
    "cell_cycle": False,
    "paris": False,
    "doublets": False,
    "markers": False,
    "params": {"ann_index": {"ann_parallel": False}},
}
# The UMAP layout the reference UMAP adds, as the blood cohort analysis lays out its UMAP.
REFERENCE_UMAP_PARAMS = {"parallel": True, "n_epochs": 300, "random_seed": 4444}


def convert_h5ad(work: Path, name: str, payload: dict) -> Path:
    """Write the prepared input work/<name>.h5ad as a CyteArc store at work/<name>.zarr and return its path."""
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    reader = H5adReader(str(work / f"{name}.h5ad"), feature_name_key="feature_name")
    try:
        H5adToZarr(
            reader,
            str(work / f"{name}.zarr"),
            assay_name="RNA",
            mem_budget=payload.get("memBudget", "24G"),
            nthreads=payload.get("nthreads", 16),
        ).dump()
    finally:
        reader.close()
    return work / f"{name}.zarr"


def run(payload):
    configure_cytearc()
    mode = payload["mode"]
    if mode not in {"prepare", "benchmark"}:
        raise ValueError("mode must be prepare or benchmark")
    if payload.get("r2EndpointUrl"):
        os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    work = Path(payload.get("workDirectory", f"/tmp/{payload['tissue']}_mapping"))
    work.mkdir(parents=True, exist_ok=True)
    result = {"settings": payload, "mode": mode, "labelColumn": "cell_type"}
    objects = Objects(payload.get("r2EndpointUrl"))
    if mode == "benchmark":
        result["inputs"] = objects.get_json(payload["inputManifestUri"])
        fetch_started = time.perf_counter()
        parts = {"reference": payload.get("referencePart", "reference"), "query": "query"}
        for name, part in parts.items():
            if part not in result["inputs"]["parts"]:
                raise ValueError(f"The input manifest has no part {part!r}")
            item = result["inputs"]["parts"][part]
            path = work / f"{name}.h5ad"
            objects.download(item["uri"], path)
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"The prepared {name} input checksum changed")
        result["inputDownloadAndVerificationSeconds"] = time.perf_counter() - fetch_started
        result["timingBoundary"] = (
            "Fresh prepared matched H5ADs; original source assembly and input downloads excluded from analysis steps"
        )
    storage = R2(payload["r2EndpointUrl"]) if payload.get("recordUri") else None
    sampler = Sampler()
    sampler.start()
    steps = _Steps()
    started = time.time()

    def report_progress():
        sampler.set_step(steps.current)
        if storage is None:
            return
        with sampler.lock:
            peaks = {name: dict(value) for name, value in sampler.step_peaks.items()}
            overall = (sampler.peak_cgroup, sampler.peak_rss)
        storage.put_json(
            payload["recordUri"].removesuffix(".json") + "/progress.json",
            {
                "currentStep": steps.current,
                "steps": {
                    name: {**record, **peaks.get(name, {})}
                    for name, record in steps.records.items()
                },
                "peakContainerBytes": overall[0],
                "peakProcessTreeRssBytes": overall[1],
                "elapsedSeconds": time.time() - started,
            },
        )

    steps.on_change = report_progress

    def finish_measurement():
        sampler.stop_event.set()
        sampler.join()
        sampler.sample()
        result["steps"] = {
            name: {**step, **sampler.step_peaks.get(name, {})}
            for name, step in steps.records.items()
        }
        result["peakProcessTreeRssBytes"] = sampler.peak_rss
        result["peakContainerBytes"] = sampler.peak_cgroup
        result["memoryDefinition"] = (
            "Sampled container memory.current and process-tree RSS, using the same Sampler as external mapping comparators"
        )
        write_json(
            f"mapping_{payload['tissue']}_{mode}.json",
            result,
            directory=work,
        )
        return result

    if mode == "prepare":
        assemble = prepare_reference_ladder if "referenceLadder" in payload else prepare_inputs
        result["inputs"] = steps.run("assembleMatchedInputs", lambda: assemble(payload, work))
        for name in result["inputs"]["parts"]:
            path = work / f"{name}.h5ad"
            with path.open("rb") as handle:
                result["inputs"]["parts"][name]["sha256"] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
            uri = (
                payload["recordUri"].removesuffix(".json") + f"/{name}.h5ad"
                if storage
                else str(path.resolve())
            )
            if storage:
                storage.upload(path, uri)
            result["inputs"]["parts"][name]["uri"] = uri
        if "queryCells" in result["inputs"]:
            item = result["inputs"]["queryCells"]
            path = work / item["file"]
            with path.open("rb") as handle:
                item["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
            item["uri"] = (
                payload["recordUri"].removesuffix(".json") + f"/{item['file']}"
                if storage
                else str(path.resolve())
            )
            if storage:
                storage.upload(path, item["uri"])
        manifest = (
            payload["recordUri"].removesuffix(".json") + "/inputs.json"
            if storage
            else str((work / "inputs.json").resolve())
        )
        objects.put_json(manifest, result["inputs"])
        result["inputManifestUri"] = manifest
        return finish_measurement()
    result["arms"] = {}
    probes = {name: StoreProbe() for name in ("reference", "query")}

    def convert(name):
        return cytearc.DataStore(
            RecordingStoreWrapper(
                LocalStore(str(convert_h5ad(work, name, payload))), probe=probes[name]
            ),
            min_features_per_cell=-1,
            mem_budget=payload.get("memBudget", "24G"),
            nthreads=payload.get("nthreads", 16),
        )

    reference = steps.run("convert_reference", lambda: convert("reference"))
    reference_labels = np.asarray(reference.cells.fetch_all("cell_type"), dtype=str)
    if "rareTypes" in result["inputs"]:
        rare_types = result["inputs"]["rareTypes"]["belowOnePercent"]
    else:
        frequencies = pd.Series(reference_labels).value_counts()
        rare_types = frequencies[
            (frequencies / len(reference_labels) < 0.01) & (frequencies >= 20)
        ].index.tolist()
    settings = REFERENCE_SETTINGS
    requested_arm = payload.get("referenceMethod", "integrated")
    if requested_arm not in {"unintegrated", "integrated"}:
        raise ValueError("referenceMethod must be unintegrated or integrated")
    for arm in (requested_arm,):
        extra = {"harmony_batch_columns": ["study"]} if arm == "integrated" else {}
        run_result = steps.run(
            f"{arm}_reference", lambda: reference.pipeline.run(**settings, **extra)
        )
        probes["reference"].reset()
        model = steps.run(
            f"{arm}_registration",
            lambda: reference.mapping.load_reference(
                reference.mapping.build_reference(run_result["neighbors"])
            ),
        )
        registration = probes["reference"].to_json()
        if registration["countReadBytes"]:
            raise AssertionError("Registration read reference count chunks")
        query = steps.run("convert_query", lambda: convert("query"))
        truth = np.asarray(query.cells.fetch_all("cell_type"), dtype=str)
        study = np.asarray(query.cells.fetch_all("study"), dtype=str)
        donors = np.asarray(query.cells.fetch_all("donor_id"), dtype=str)
        ids = np.asarray(query.cells.fetch_all("ids"), dtype=str)
        label_order = sorted(set(reference_labels) | set(truth)) + ["unassigned"]
        cells = steps.run("snapshot_query_cells", lambda: query.snapshot_cell_selection("I"))
        for probe in probes.values():
            probe.reset()
        batches = pd.DataFrame({"study": study}) if arm == "integrated" else None
        projection = steps.run(
            f"{arm}_projection_correction_neighbors",
            lambda: query.mapping.run(
                model, cells, save_k=30, query_batches=batches, missing_feature_policy="error"
            ),
        )
        transferred = steps.run(
            f"{arm}_label_voting",
            lambda: query.mapping.load_label_transfer(
                query.mapping.transfer_labels(
                    projection,
                    reference=model,
                    reference_labels="cell_type",
                    threshold_fraction=0.0,
                )
            ),
        )
        frame = pd.DataFrame(
            {
                "cell": ids,
                "study": study,
                "donor_id": donors,
                "truth": truth,
                "prediction": transferred.labels.fillna("unassigned").to_numpy(dtype=str),
                "confidence": transferred.evidence.voteFraction.to_numpy(dtype=float),
            }
        )
        result["arms"][arm] = {
            "registrationIo": registration,
            "queryAnnotationIo": {name: probe.to_json() for name, probe in probes.items()},
            "evaluation": evaluate_predictions(frame, label_order, rare_types),
            "pipeline": run_result.report(),
        }
        path = work / f"{arm}_predictions.csv"
        frame.to_csv(path, index=False)
        result["arms"][arm]["predictionsUri"] = str(path.resolve())
        if payload.get("recordUri"):
            uri = payload["recordUri"].removesuffix(".json") + f"/{arm}_predictions.csv"
            storage.upload(path, uri)
            result["arms"][arm]["predictionsUri"] = uri
            storage.put_json(payload["recordUri"].removesuffix(".json") + "/completed.json", result)
    return finish_measurement()


def reference_umap(payload: dict) -> dict:
    """Rebuild a held-out tissue's reference analysis from its prepared input with the benchmark's settings and UMAP
    on, check that the rebuild ran the benchmark's CyteArc version and configuration, apart from UMAP, on the same
    cells, and save the reference's UMAP with each cell's harmonized label, author label and study beside the record.
    Needs inputManifestUri, the tissue's prepared inputs, and benchmarkRun, the run block of the report of the
    benchmark's unintegrated reference. Artifact keys cannot be compared: each conversion of an H5AD makes a store of
    its own identity, which every key descends from."""
    configure_cytearc()
    if payload.get("r2EndpointUrl"):
        os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    objects = Objects(payload.get("r2EndpointUrl"))
    work = Path(payload.get("workDirectory", f"/tmp/{payload['tissue']}_reference_umap"))
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    item = objects.get_json(payload["inputManifestUri"])["parts"]["reference"]
    objects.download(item["uri"], work / "reference.h5ad")
    with (work / "reference.h5ad").open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != item["sha256"]:
            raise ValueError("The prepared reference input checksum changed")
    reference = cytearc.DataStore(
        LocalStore(str(convert_h5ad(work, "reference", payload))),
        min_features_per_cell=-1,
        mem_budget=payload.get("memBudget", "24G"),
        nthreads=payload.get("nthreads", 16),
    )
    settings = {
        **REFERENCE_SETTINGS,
        "umap": True,
        "params": {**REFERENCE_SETTINGS["params"], "umap": REFERENCE_UMAP_PARAMS},
    }
    pipeline_started = time.perf_counter()
    analysis = reference.pipeline.run(**settings)
    pipeline_seconds = time.perf_counter() - pipeline_started
    report = analysis.report()
    benchmark = payload["benchmarkRun"]
    if report["run"]["cytearcVersion"] != benchmark["cytearcVersion"]:
        raise AssertionError(
            f"The rebuild ran CyteArc {report['run']['cytearcVersion']}, the benchmark {benchmark['cytearcVersion']}"
        )

    def without_umap(config: dict) -> dict:
        kept = {key: value for key, value in config.items() if key not in {"umap", "shutdown"}}
        kept["params"] = {key: value for key, value in config["params"].items() if key != "umap"}
        return kept

    if without_umap(report["run"]["config"]) != without_umap(benchmark["config"]):
        raise AssertionError(
            "The rebuild's configuration differs from the benchmark's apart from UMAP"
        )
    if reference.cells.N != item["cells"]:
        raise AssertionError(
            f"The rebuild holds {reference.cells.N} cells, the prepared reference {item['cells']}"
        )
    group = reference.artifacts.load(analysis["umap"])
    umap = np.asarray(group["data" if "data" in group else "values"][:], dtype=np.float32)
    columns = {
        name: np.asarray(reference.cells.fetch_all(name), dtype=str)
        for name in ("cell_type", "author_label", "study")
    }
    if umap.shape != (reference.cells.N, 2) or any(
        len(values) != reference.cells.N for values in columns.values()
    ):
        raise AssertionError("The UMAP or the cell labels do not match the reference's cells")
    coded = {name: np.unique(values, return_inverse=True) for name, values in columns.items()}
    path = work / "reference_umap.npz"
    np.savez_compressed(
        path,
        umap=umap,
        label_codes=coded["cell_type"][1].astype(np.int16),
        label_names=coded["cell_type"][0],
        author_label_codes=coded["author_label"][1].astype(np.int16),
        author_label_names=coded["author_label"][0],
        study_codes=coded["study"][1].astype(np.int16),
        study_names=coded["study"][0],
    )
    uri = payload["recordUri"].removesuffix(".json") + "/reference_umap.npz"
    R2(payload["r2EndpointUrl"]).upload(path, uri)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "settings": {key: value for key, value in payload.items() if key != "benchmarkRun"},
        "analysisSettings": settings,
        "cells": int(reference.cells.N),
        "matchesBenchmark": {
            "cytearcVersion": benchmark["cytearcVersion"],
            "configuration": True,
            "cells": True,
        },
        "umapUri": uri,
        "umapSha256": digest,
        "pipelineSeconds": pipeline_seconds,
        "seconds": time.perf_counter() - started,
        "pipeline": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    run(json.loads(parser.parse_args().payload.read_text()))
