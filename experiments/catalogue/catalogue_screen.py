"""Query every human blood store in CyteBase in place for an interferon gene panel.

From each published store on Hugging Face the screen reads eight cell metadata columns, the
columns of the store's sample rule (experiments/runtime/blood_samples.py) and the counts of a
22-gene panel, and nothing else. Stores the rules exclude are not read, and cells of excluded
samples leave before any count is read. Panel counts and the stored library sizes
(``RNA_nCounts``) are summed per donor, sample and coarse cell type, so each group's
log2(CPM + 1) and interferon scores follow without reading whole cells; each cell's
interferon log2(CPM + 1) is also counted into bins per group. Each donor keeps its earliest
sample, and each group's interferon genes are z-scored against the healthy donors of its store
and cell type. A job screens one store set: ``immune_cohorts``, the 15 stores of the Human
Immune Health Atlas, Sound Life, lupus, COVID-19 and immune aging studies, or
``other_blood_stores``, the other human blood stores of the release. Its result keeps, per
store, the time, HTTP requests, response bytes and their share of the store's count bytes, the
cells and donors covered, and any failure, then summarizes the interferon score by disease
label across stores.

``collect`` writes records/blood_store_screen.json from the jobs' run records: one row per job
with its store set, settings, placement, cells and donors covered, time, traffic and memory.
``keep`` writes one job's run record and its donor table into records/<screen>/, named for the
job's store set: blood_store_screen/ for the screen that blood_store_screen.json summarizes, and
interferon_screen/ for the screen of each donor's samples with interferon z-scores.
"""

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import cytearc
import httpx2
import numpy as np
import pandas as pd

from experiments.catalogue import HERE
from experiments.runtime.annotations import MISSING_DONORS, blood_cell_type
from experiments.runtime.blood_samples import (
    EXCLUDED_STORES,
    SAMPLE_RULES,
    earliest_samples,
    time_key,
)
from experiments.runtime.common import configure_cytearc, public_run, write_json, write_json_at
from experiments.runtime.locations import CYTEBASE_RELEASE
from experiments.runtime.metrics import ResourceSampler
from experiments.runtime.recording_store import (
    NetworkProbe,
    RecordingStoreWrapper,
    probe_summary,
)

WORK = Path("/tmp/catalogue_screen")
RECORD = HERE / "records" / "blood_store_screen.json"
# The folders of records/ that keep each screen's job records and donor tables.
SCREENS = ("blood_store_screen", "interferon_screen")
CATALOGUE_VERSION = "2610"
BUCKET = CYTEBASE_RELEASE.removeprefix("hf://buckets/")
# UBERON blood and the two narrower blood terms that CyteBase stores carry.
BLOOD_TISSUES = ("blood", "venous blood", "umbilical cord blood")
# The stores of the Human Immune Health Atlas and Sound Life (Gong 2025), lupus (Perez 2022),
# COVID-19 (Ren 2021, van der Wijst 2021, Yoshida 2022) and immune aging (Wells 2025) studies.
IMMUNE_COHORT_STORES = (
    "gong_2025_all_cells_ff4235bb",
    "gong_2025_human_immune_health_atlas_e522d2cd",
    "gong_2025_sound_life_older_adult_females_cmv_negative_3cb646ab",
    "gong_2025_sound_life_older_adult_females_cmv_positive_aa32098f",
    "gong_2025_sound_life_older_adult_males_cmv_negative_47e00f43",
    "gong_2025_sound_life_older_adult_males_cmv_positive_0c0c49a2",
    "gong_2025_sound_life_young_adult_females_cmv_negative_46104f0b",
    "gong_2025_sound_life_young_adult_females_cmv_positive_0a8e3443",
    "gong_2025_sound_life_young_adult_males_cmv_negative_420d0f5d",
    "gong_2025_sound_life_young_adult_males_cmv_positive_5fde5c9c",
    "perez_2022_multiplexed_scrna_1_2_million_pbmcs_adult_lupus_samples_218acb0f",
    "ren_2021_large_scale_single_cell_covid_19_patients_9dbab10c",
    "van_der_wijst_2021_type_i_interferon_patients_covid_19_01ad3cd7",
    "wells_2025_immune_aging_project_all_cells_1b350d0a",
    "yoshida_2022_pbmc_2a498ace",
)
# In the order of the record's rows.
STORE_SETS = ("other_blood_stores", "immune_cohorts")
# The payload settings that each row of the record keeps.
ROW_SETTINGS = ("storeConcurrency", "nthreads", "memBudget")
# The job totals that each row of the record keeps.
ROW_TOTALS = (
    "storesScreened",
    "storesFailed",
    "cellsInScreenedStores",
    "primaryBloodCellsRead",
    "studyScopedDonors",
    "donorsWithScore",
    "httpRequests",
    "responseBytes",
    "storeCountBytes",
)
# The six interferon-stimulated genes of the first screen, then twelve more; IDs as the stores' own feature tables
# name the symbols.
INTERFERON_GENES = {
    "ISG15": "ENSG00000187608",
    "IFIT1": "ENSG00000185745",
    "IFIT2": "ENSG00000119922",
    "IFIT3": "ENSG00000119917",
    "MX1": "ENSG00000157601",
    "OAS1": "ENSG00000089127",
    "IFI27": "ENSG00000165949",
    "IFI44L": "ENSG00000137959",
    "IFI44": "ENSG00000137965",
    "IFI6": "ENSG00000126709",
    "RSAD2": "ENSG00000134321",
    "SIGLEC1": "ENSG00000088827",
    "MX2": "ENSG00000183486",
    "OAS2": "ENSG00000111335",
    "OAS3": "ENSG00000111331",
    "HERC5": "ENSG00000138646",
    "XAF1": "ENSG00000132530",
    "EPSTI1": "ENSG00000133106",
}
# One canonical marker for each of the four most abundant blood lineages.
LINEAGE_GENES = {
    "CD3E": "ENSG00000198851",
    "MS4A1": "ENSG00000156738",
    "CD14": "ENSG00000170458",
    "NKG7": "ENSG00000105374",
}
PANEL = {**INTERFERON_GENES, **LINEAGE_GENES}
SELECTION_COLUMNS = ("is_primary_data", "tissue", "RNA_nCounts")
GROUP_COLUMNS = ("donor_id", "disease", "sex", "development_stage", "tissue", "cell_type")
MINIMUM_CELLS = 20
# A group's z-scores need this many healthy donors of its store and cell type, each with MINIMUM_CELLS cells.
MINIMUM_HEALTHY_DONORS = 3
# Each cell's interferon log2(CPM + 1) is counted into bins this wide from zero; the last bin also holds what lies
# above it.
CELL_SCORE_BIN = 0.25
CELL_SCORE_BINS = 72
CELL_SCORE_COLUMNS = [f"cellScoreBin_{index:02d}" for index in range(CELL_SCORE_BINS)]
QUOTA_HEADER = re.compile(r'"(?P<name>[^"]+)";r=(?P<remaining>\d+);t=(?P<reset>\d+)')
STORE_IN_PATH = re.compile(r"/datasets/([^/]+)/")


def coarse_cell_type(label: str) -> str:
    """Coarse blood cell type of a CELLxGENE cell_type label, or "" when it has none.

    This is ``blood_cell_type`` of the blood cohort analysis, except that mast cells are
    Granulocytes and malignant cells have no type. Its substring rules would put both in
    T cells.
    """
    text = label.lower()
    if "malignant" in text:
        return ""
    if text == "mast cell":
        return "Granulocytes"
    return blood_cell_type(label)


class HubTraffic:
    """Attribute Hugging Face HTTP traffic to stores and pace it within the Hub quotas.

    Every Hub response reports the requests left in a fixed window, separately for listings
    ("api") and file reads ("resolvers"). When ``reserve`` or fewer remain, requests of that
    kind wait for the window to reset instead of failing with HTTP 429. Redirected file reads
    reach a CDN URL without the store in its path, so each redirect target inherits the store
    of the request that produced it.
    """

    def __init__(self, reserve: int):
        self.reserve = reserve
        self.lock = threading.Lock()
        self.probes: dict[str, NetworkProbe] = {}
        self.kinds: dict[str, Counter] = defaultdict(Counter)
        self.waits: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
        self.store_waits: dict[str, float] = defaultdict(float)
        self.redirects: dict[str, str] = {}
        self.resume_at: dict[str, float] = {}
        self.quota_policies: set[str] = set()

    def probe(self, store_id: str) -> NetworkProbe:
        with self.lock:
            return self.probes.setdefault(store_id, NetworkProbe())

    def owner(self, request) -> str:
        found = STORE_IN_PATH.search(request.url.path)
        if found:
            return found.group(1)
        with self.lock:
            return self.redirects.pop(str(request.url), "")

    def wait(self, kind: str, owner: str) -> None:
        while True:
            with self.lock:
                delay = self.resume_at.get(kind, 0.0) - time.monotonic()
            if delay <= 0:
                return
            time.sleep(delay)
            with self.lock:
                self.waits[kind][0] += 1
                self.waits[kind][1] += delay
                self.store_waits[owner] += delay

    def observe(self, request, response, owner: str, kind: str) -> None:
        with self.lock:
            self.kinds[owner][kind] += 1
            self.kinds[owner][f"status{response.status_code}"] += 1
            location = response.headers.get("location")
            if response.is_redirect and location and owner:
                self.redirects[location] = owner
            policy = response.headers.get("ratelimit-policy")
            if policy:
                self.quota_policies.add(policy)
            quota = QUOTA_HEADER.search(response.headers.get("ratelimit", ""))
            if quota is None:
                return
            reset = float(quota["reset"]) + 1
            if response.status_code == 429:
                reset = max(reset, float(response.headers.get("retry-after") or 0))
            if response.status_code == 429 or int(quota["remaining"]) <= self.reserve:
                name = quota["name"]
                self.resume_at[name] = max(self.resume_at.get(name, 0.0), time.monotonic() + reset)

    def store_summary(self, store_id: str) -> dict:
        with self.lock:
            kinds = dict(self.kinds.get(store_id, {}))
            waited = self.store_waits.get(store_id, 0.0)
        return {
            **probe_summary(self.probe(store_id)),
            "requestsByKind": kinds,
            "quotaWaitSeconds": waited,
        }

    def totals(self) -> dict:
        with self.lock:
            kinds = Counter()
            for counts in self.kinds.values():
                kinds.update(counts)
            unattributed = dict(self.kinds.get("", {}))
            waits = {kind: {"waits": n, "seconds": s} for kind, (n, s) in self.waits.items()}
            policies = sorted(self.quota_policies)
        return {
            "requestsByKind": dict(kinds),
            "unattributedRequestsByKind": unattributed,
            "quotaWaits": waits,
            "quotaPolicies": policies,
        }


def request_kind(request) -> str:
    if request.url.host != "huggingface.co":
        return "cdn"
    return "api" if request.url.path.startswith("/api/") else "resolvers"


@contextmanager
def hub_traffic(traffic: HubTraffic):
    """Count and pace every httpx2 request that the Hugging Face file system sends."""
    original = httpx2.Client._send_single_request

    class CountedStream(httpx2.SyncByteStream):
        def __init__(self, stream, probe):
            self.stream, self.probe = stream, probe

        def __iter__(self):
            for chunk in self.stream:
                self.probe.received(len(chunk))
                yield chunk

        def close(self):
            self.stream.close()

    def send(client, request):
        kind, owner = request_kind(request), traffic.owner(request)
        if kind != "cdn":
            traffic.wait(kind, owner)
        probe = traffic.probe(owner)
        probe.request(request)
        response = original(client, request)
        probe.response(response.status_code)
        traffic.observe(request, response, owner, kind)
        response.stream = CountedStream(response.stream, probe)
        return response

    httpx2.Client._send_single_request = send
    try:
        yield traffic
    finally:
        httpx2.Client._send_single_request = original


def eligible_stores(catalog, store_set: str) -> list[dict]:
    """The human blood stores of ``store_set``, largest first."""
    rows = catalog.query(
        """
        SELECT cytebase_id, collection_id, first_author, year, title, zarr_uri, cell_count,
               primary_cell_count, n_genes, assay_labels, tissue_labels, disease_labels
        FROM datasets
        WHERE list_contains(organism_labels, 'Homo sapiens')
          AND list_has_any(tissue_labels, ?)
          AND status = 'ready'
        ORDER BY cell_count DESC
        """,
        [list(BLOOD_TISSUES)],
    )
    stores = [dict(row) for row in rows]
    unknown = sorted(set(IMMUNE_COHORT_STORES) - {row["cytebase_id"] for row in stores})
    if unknown:
        raise ValueError(f"Not human blood stores of release {CATALOGUE_VERSION}: {unknown}")
    cohort = [row["cytebase_id"] in IMMUNE_COHORT_STORES for row in stores]
    if store_set == "immune_cohorts":
        return [row for row, chosen in zip(stores, cohort, strict=True) if chosen]
    if store_set == "other_blood_stores":
        return [row for row, chosen in zip(stores, cohort, strict=True) if not chosen]
    raise ValueError(f"Unknown storeSet {store_set!r}; choose from {', '.join(STORE_SETS)}")


def catalogue_sha256(catalog) -> str:
    with catalog.connect_catalog() as connection:
        path = next(row[2] for row in connection.execute("PRAGMA database_list").fetchall())
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def open_store(row: dict, traffic: HubTraffic, token: str | bool, payload: dict):
    """Open a store after one recursive listing, which replaces a listing per directory."""
    from cytearc.storage.huggingface import HfReadStore
    from zarr.core.sync import sync

    store = HfReadStore.from_url(
        row["zarr_uri"], storage_options={"token": token, "block_size": 1}, read_only=True
    )
    started = time.perf_counter()
    listing = sync(store.fs._find(store.path, detail=True))
    listed = time.perf_counter()
    sizes = defaultdict(int)
    for name, info in listing.items():
        relative = name.removeprefix(store.path).lstrip("/")
        part = "countsT" if relative.startswith("RNA/countsT/") else None
        part = part or ("counts" if relative.startswith("RNA/counts/") else "other")
        if not relative.endswith("zarr.json"):
            sizes[part] += int(info["size"])
    ds = cytearc.DataStore(
        RecordingStoreWrapper(store, probe=traffic.probe(row["cytebase_id"])),
        zarr_mode="r",
        min_features_per_cell=-1,
        mem_budget=payload.get("memBudget", "3G"),
        nthreads=payload.get("nthreads", os.cpu_count()),
    )
    return ds, {
        "listingSeconds": listed - started,
        "openSeconds": time.perf_counter() - listed,
        "listedObjects": len(listing),
        "storeBytes": sum(sizes.values()),
        "countBytes": sizes["counts"] + sizes["countsT"],
        "countsByGeneBytes": sizes["countsT"],
    }


def panel_layout(layout: dict, positions: np.ndarray) -> dict:
    """Gene-major bands and shard files that reading ``positions`` touches."""
    chunks, shards = layout["countsT"]["chunks"], layout["countsT"]["shards"]
    n_feats, n_cells, itemsize = layout["nFeats"], layout["nCells"], layout["itemsize"]
    bands = np.unique(positions // chunks[0])
    widths = np.minimum((bands + 1) * chunks[0], n_feats) - bands * chunks[0]
    cell_shards = -(-n_cells // shards[1])
    return {
        "bandGenes": int(chunks[0]),
        "bandsTouched": len(bands),
        "bandGeneTotal": int(widths.sum()),
        "genes": int(n_feats),
        "cellChunksPerBand": int(-(-n_cells // chunks[1])),
        "shardFilesTouched": int(len(np.unique(positions // shards[0])) * cell_shards),
        "uncompressedBandBytes": int(widths.sum()) * int(n_cells) * int(itemsize),
    }


def ensembl_gene_ids(feature_ids) -> np.ndarray:
    """Bare Ensembl gene IDs, without version suffixes (".10") or genome prefixes ("GRCh38___")."""
    found = pd.Series(np.asarray(feature_ids, dtype=str)).str.extract(r"(ENSG\d{11})", expand=False)
    return found.fillna("").to_numpy(dtype=str)


def read_codes(ds, column: str, rows: np.ndarray) -> tuple[np.ndarray, list[str], bool]:
    """Factorized values of ``column`` over ``rows``; an absent column reads as "unknown"."""
    if column not in ds.cells.columns:
        return np.zeros(len(rows), dtype=np.int64), ["unknown"], False
    codes, names = pd.factorize(np.asarray(ds.cells.fetch_all(column), dtype=str)[rows])
    return codes, [str(name) for name in names], True


def apply_sample_rule(
    ds, store_id: str, rows: np.ndarray, rules: dict = SAMPLE_RULES
) -> tuple[np.ndarray, dict, np.ndarray, list[str], np.ndarray]:
    """The store's sample rule over ``rows``: which cells stay, the cells of each excluded value, each cell's sample
    (codes and names) and its two time keys (infinite where no time is recorded). Without a sample column the first
    time column names the samples; a store without a rule has one sample per donor."""
    rule = rules.get(store_id, {})

    def column(name: str) -> tuple[np.ndarray, list[str]]:
        codes, names, found = read_codes(ds, name, rows)
        if not found:
            raise KeyError(f"{store_id} has no cell column {name!r}, which its sample rule reads")
        return codes, names

    keep = np.ones(len(rows), dtype=bool)
    excluded = {}
    for name, values in rule.get("exclude", {}).items():
        codes, names = column(name)
        text = np.asarray(names, dtype=object)[codes]
        hit = np.isin(text, list(values))
        found, counts = np.unique(text[hit].astype(str), return_counts=True)
        excluded[name] = {str(value): int(n) for value, n in zip(found, counts, strict=True)}
        keep &= ~hit
    specs = rule.get("time", ())
    if len(specs) > 2:
        raise ValueError(f"{store_id}: at most two time columns, got {len(specs)}")
    identity = rule.get("sample") or (specs[0]["column"] if specs else None)
    if identity:
        sample_codes, sample_names = column(identity)
    else:
        sample_codes, sample_names = np.zeros(len(rows), dtype=np.int64), [""]
    times = np.full((len(rows), 2), np.inf)
    for index, spec in enumerate(specs):
        codes, names = column(spec["column"])
        times[:, index] = time_key(pd.Series(names, dtype=object), spec)[codes]
    return keep, excluded, sample_codes, sample_names, times


def cell_score_bins(
    values: np.ndarray, library: np.ndarray, group: np.ndarray, n_groups: int
) -> np.ndarray:
    """Cells per group in each bin of their interferon log2(CPM + 1): ``values`` holds the cells' interferon gene
    counts, ``library`` their library sizes. Cells without a library are not counted."""
    scored = library > 0
    score = np.log2(values[scored].sum(axis=1) / library[scored] * 1e6 + 1)
    index = np.minimum((score / CELL_SCORE_BIN).astype(np.int64), CELL_SCORE_BINS - 1)
    flat = np.bincount(
        group[scored] * CELL_SCORE_BINS + index, minlength=n_groups * CELL_SCORE_BINS
    )
    return flat.reshape(n_groups, CELL_SCORE_BINS)


def screen_store(row: dict, traffic: HubTraffic, token: str | bool, payload: dict) -> dict:
    from cytearc.assay.normalization import feature_value_blocks
    from cytearc.metadata.selection import NormalizationSpec

    started = time.perf_counter()
    ds, store = open_store(row, traffic, token, payload)
    store["storeId"] = row["cytebase_id"]
    phase = time.perf_counter()
    missing = [column for column in SELECTION_COLUMNS if column not in ds.cells.columns]
    if missing:
        raise ValueError(f"{row['cytebase_id']} lacks required cell columns {missing}")
    primary, tissue, totals = ds.cells.fetch_all_columns(SELECTION_COLUMNS)
    selected = np.asarray(primary, dtype=bool) & np.isin(
        np.asarray(tissue, dtype=str), BLOOD_TISSUES
    )
    rows = np.flatnonzero(selected)
    totals = np.asarray(totals, dtype=np.float64)[rows]
    keep, store["excludedValues"], sample_codes, sample_names, times = apply_sample_rule(
        ds, row["cytebase_id"], rows
    )
    store["sampleRule"] = row["cytebase_id"] in SAMPLE_RULES
    store["excludedCells"] = int((~keep).sum())
    rows, totals, times = rows[keep], totals[keep], times[keep]
    codes, names, present = {"sample": sample_codes[keep]}, {"sample": sample_names}, {}
    for column in GROUP_COLUMNS:
        codes[column], names[column], present[column] = read_codes(ds, column, rows)
    labels = np.asarray(names["cell_type"], dtype=object)
    coarse_names, coarse_of_label = np.unique(
        [coarse_cell_type(label) for label in labels], return_inverse=True
    )
    codes["coarse_cell_type"] = coarse_of_label[codes["cell_type"]]
    names["coarse_cell_type"] = coarse_names.tolist()
    store["metadataSeconds"] = time.perf_counter() - phase
    store["cells"] = int(ds.cells.N)
    store["primaryCells"] = int(np.asarray(primary, dtype=bool).sum())
    store["selectedCells"] = len(rows)
    store["absentColumns"] = [column for column, found in present.items() if not found]
    store["labels"] = [
        {"label": label, "coarseCellType": coarse_names[coarse_of_label[index]], "cells": int(n)}
        for index, (label, n) in enumerate(
            zip(labels, np.bincount(codes["cell_type"], minlength=len(labels)), strict=True)
        )
    ]
    phase = time.perf_counter()
    feature_ids = ensembl_gene_ids(ds.RNA.feats.fetch_all("ids"))
    positions = {}
    for symbol, gene_id in PANEL.items():
        found = np.flatnonzero(feature_ids == gene_id)
        if len(found) > 1:
            raise ValueError(f"{row['cytebase_id']} stores {gene_id} ({symbol}) more than once")
        if len(found):
            positions[symbol] = int(found[0])
    store["missingGenes"] = sorted(set(PANEL) - set(positions))
    symbols = sorted(positions, key=positions.get)
    order = np.array([positions[symbol] for symbol in symbols], dtype=np.int64)
    layout = dict(ds.RNA.matrixGroup["counts"].attrs).get("cytearc:countMatrixLayout")
    store["panelLayout"] = panel_layout(layout, order) if layout and len(order) else None
    keys = [column for column in GROUP_COLUMNS if column != "cell_type"] + [
        "sample",
        "coarse_cell_type",
    ]
    frame = pd.DataFrame({column: codes[column] for column in keys})
    group = frame.groupby(keys, sort=False).ngroup().to_numpy()
    n_groups = int(group.max()) + 1 if len(group) else 0
    sums = np.zeros((n_groups, len(symbols)))
    detected = np.zeros((n_groups, len(symbols)))
    bins = np.zeros((n_groups, CELL_SCORE_BINS), dtype=np.int64)
    interferon = [index for index, symbol in enumerate(symbols) if symbol in INTERFERON_GENES]
    if len(rows) and len(symbols):
        _, blocks = feature_value_blocks(
            ds.RNA,
            order,
            rows,
            NormalizationSpec(source="raw"),
            nthreads=payload.get("nthreads", os.cpu_count()),
            resident_bytes=int(
                rows.nbytes + totals.nbytes + group.nbytes + frame.memory_usage().sum()
            ),
            scratch_bytes_per_row=16 * len(symbols),
        )
        start = 0
        for block in blocks:
            values = np.asarray(block, dtype=np.float64)
            stop = start + len(values)
            if np.any(values.sum(axis=1) > totals[start:stop] + 0.5):
                raise AssertionError(f"{row['cytebase_id']}: panel counts exceed RNA_nCounts")
            for column in range(len(symbols)):
                sums[:, column] += np.bincount(group[start:stop], values[:, column], n_groups)
                detected[:, column] += np.bincount(
                    group[start:stop], values[:, column] > 0, n_groups
                )
            if interferon:
                bins += cell_score_bins(
                    values[:, interferon], totals[start:stop], group[start:stop], n_groups
                )
            start = stop
        if start != len(rows):
            raise AssertionError(f"{row['cytebase_id']}: streamed {start} of {len(rows)} cells")
    store["panelSeconds"] = time.perf_counter() - phase
    _, first = np.unique(group, return_index=True)
    groups = pd.DataFrame(
        {
            "storeId": row["cytebase_id"],
            "collectionId": row["collection_id"],
            **{
                column: np.asarray(names[column], dtype=object)[codes[column][first]]
                for column in keys
            },
            "cells": np.bincount(group, minlength=n_groups),
            "librarySize": np.bincount(group, totals, n_groups),
        }
    )
    for column, symbol in enumerate(symbols):
        groups[f"count_{symbol}"] = sums[:, column]
        groups[f"detected_{symbol}"] = detected[:, column]
    sample_times = (
        pd.DataFrame({"group": group, "t0": times[:, 0], "t1": times[:, 1]})
        .groupby("group")
        .min()
        .reindex(range(n_groups))
    )
    groups["sampleTime0"] = sample_times["t0"].to_numpy()
    groups["sampleTime1"] = sample_times["t1"].to_numpy()
    groups[CELL_SCORE_COLUMNS] = bins
    donors = groups.donor_id.astype(str).str.strip().str.lower()
    known = ~donors.isin(MISSING_DONORS)
    store["cellsWithoutDonor"] = int(groups.cells[~known].sum())
    store["cellsWithoutCoarseType"] = int(groups.cells[groups.coarse_cell_type == ""].sum())
    store["donors"] = int(groups.donor_id[known].nunique())
    store["groups"] = len(groups)
    store["seconds"] = time.perf_counter() - started
    return {"store": store, "groups": groups}


def screen_with_retries(row: dict, traffic: HubTraffic, token: str | bool, payload: dict) -> dict:
    errors = []
    for attempt in range(1, payload.get("storeAttempts", 2) + 1):
        try:
            result = screen_store(row, traffic, token, payload)
            result["store"]["attempts"] = attempt
            result["store"]["errors"] = errors
            return result
        except Exception as error:
            errors.append(
                {"attempt": attempt, "errorType": type(error).__name__, "error": str(error)}
            )
            if attempt < payload.get("storeAttempts", 2):
                time.sleep(payload.get("retryDelaySeconds", 60))
    return {"failure": {"storeId": row["cytebase_id"], "errors": errors}}


def interferon_log_cpm(groups: pd.DataFrame) -> np.ndarray:
    """log2(CPM + 1) of each interferon gene in each pseudobulk group: NaN where the store does not measure the gene
    or the group has no library."""
    counts = groups[[f"count_{symbol}" for symbol in INTERFERON_GENES]].to_numpy(np.float64)
    library = groups.librarySize.to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log2(counts / library[:, None] * 1e6 + 1)
    values[library <= 0] = np.nan
    return values


def finite_row_means(values: np.ndarray) -> np.ndarray:
    """Each row's mean over its finite entries, NaN where it has none."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    sums = np.where(finite, values, 0.0).sum(axis=1)
    return np.divide(sums, counts, out=np.full(len(values), np.nan), where=counts > 0)


def interferon_score(groups: pd.DataFrame) -> np.ndarray:
    """Mean over the interferon genes the store measures of log2(CPM + 1) in each pseudobulk group."""
    return finite_row_means(interferon_log_cpm(groups))


def interferon_zscores(units: pd.DataFrame) -> np.ndarray:
    """Each unit's interferon genes z-scored against the healthy donors of its store and cell type, then averaged.

    The reference is the store's healthy units of that cell type from each donor's earliest sample with at least
    MINIMUM_CELLS cells, when there are MINIMUM_HEALTHY_DONORS of them. A gene the store does not measure, or that
    does not vary among the reference donors, leaves the mean. A unit without a reference has no z-score."""
    values = interferon_log_cpm(units)
    scores = np.full(len(units), np.nan)
    reference = (
        (units["disease"] == "normal").to_numpy()
        & units["earliestSample"].to_numpy(dtype=bool)
        & (units["cells"] >= MINIMUM_CELLS).to_numpy()
    )
    for index in units.groupby(["storeId", "coarse_cell_type"], sort=False).indices.values():
        healthy = values[index[reference[index]]]
        if len(healthy) < MINIMUM_HEALTHY_DONORS:
            continue
        measured = np.isfinite(healthy).all(axis=0)
        mean = healthy[:, measured].mean(axis=0)
        sd = healthy[:, measured].std(axis=0, ddof=1)
        varies = sd > 0
        if not varies.any():
            continue
        z = (values[np.ix_(index, np.flatnonzero(measured)[varies])] - mean[varies]) / sd[varies]
        scores[index] = finite_row_means(z)
    return scores


def mark_earliest_samples(units: pd.DataFrame) -> np.ndarray:
    """Whether each unit comes from its donor's earliest sample (``earliest_samples``): per study-scoped donor, the
    sample its store's time columns put first, or its largest when no time is recorded, with the donor's cells counted
    over every store and cell type, so stores split by lineage choose the same sample."""
    keys = ["collectionId", "donor_id", "sample"]
    samples = (
        units.groupby(keys, sort=False, dropna=False)
        .agg(cells=("cells", "sum"), t0=("sampleTime0", "min"), t1=("sampleTime1", "min"))
        .reset_index()
    )
    samples["donor"] = (
        samples["collectionId"].astype(str) + "\x1f" + samples["donor_id"].astype(str)
    )
    samples["name"] = samples["sample"].astype(str)
    chosen = earliest_samples(samples)[keys].assign(earliestSample=True)
    marked = units[keys].merge(chosen, how="left", on=keys)
    return marked["earliestSample"].notna().to_numpy()


def donor_units(groups: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Pool each donor's groups within a store, then keep one store per collection donor.

    A unit is a study-scoped donor, sample, disease label and coarse cell type. When stores of
    one collection share a unit, for example the same donor sequenced on two platforms, the
    store with the most cells represents it.
    """
    known = ~groups.donor_id.astype(str).str.strip().str.lower().isin(MISSING_DONORS)
    typed = groups.coarse_cell_type != ""
    keys = ["collectionId", "donor_id", "sample", "disease", "coarse_cell_type"]
    summed = [c for c in groups.columns if c.startswith(("count_", "detected_", "cellScoreBin_"))]
    pooled = groups[known & typed].groupby(["storeId", *keys], sort=False, dropna=False)
    units = (
        pooled[["cells", "librarySize", *summed]]
        .sum(min_count=1)
        .join(pooled[["sampleTime0", "sampleTime1"]].min())
        .reset_index()
    )
    units = units.sort_values("cells", ascending=False, kind="stable")
    repeated = units.duplicated(keys, keep="first")
    dropped = (
        units[repeated]
        .groupby("collectionId")
        .agg(units=("cells", "size"), cells=("cells", "sum"), stores=("storeId", "unique"))
        .reset_index()
    )
    dropped["stores"] = dropped.stores.map(sorted)
    kept = units[~repeated].copy()
    kept["interferonScore"] = interferon_score(kept)
    return kept, dropped.to_dict("records")


def quantiles(values: pd.Series) -> dict:
    return {
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
    }


def summarize_scores(units: pd.DataFrame) -> dict:
    """Interferon scores by disease, by store and disease, and within stores, from each donor's earliest sample."""
    scored = units[
        units.earliestSample & (units.cells >= MINIMUM_CELLS) & units.interferonScore.notna()
    ]
    by_disease = []
    for (disease, cell_type), part in scored.groupby(["disease", "coarse_cell_type"]):
        by_disease.append(
            {
                "disease": disease,
                "coarseCellType": cell_type,
                "stores": part.storeId.nunique(),
                "collections": part.collectionId.nunique(),
                "donors": len(part),
                "cells": int(part.cells.sum()),
                **quantiles(part.interferonScore),
            }
        )
    by_store = []
    for (store, disease, cell_type), part in scored.groupby(
        ["storeId", "disease", "coarse_cell_type"]
    ):
        by_store.append(
            {
                "storeId": store,
                "disease": disease,
                "coarseCellType": cell_type,
                "donors": len(part),
                "cells": int(part.cells.sum()),
                **quantiles(part.interferonScore),
            }
        )
    contrasts = []
    for (store, cell_type), part in scored.groupby(["storeId", "coarse_cell_type"]):
        normal = part.interferonScore[part.disease == "normal"]
        for disease, cases in part[part.disease != "normal"].groupby("disease"):
            if len(normal) >= 3 and len(cases) >= 3:
                contrasts.append(
                    {
                        "storeId": store,
                        "coarseCellType": cell_type,
                        "disease": disease,
                        "diseaseDonors": len(cases),
                        "normalDonors": len(normal),
                        "medianDifference": float(cases.interferonScore.median() - normal.median()),
                    }
                )
    return {
        "byDisease": by_disease,
        "byStoreAndDisease": by_store,
        "withinStoreContrasts": contrasts,
    }


def shared_donor_ids(units: pd.DataFrame, minimum: int = 3) -> list[dict]:
    """Collection pairs whose donor identifiers coincide, a hint of shared donors."""
    donors = units.groupby("collectionId").donor_id.agg(lambda values: set(map(str, values)))
    names = sorted(donors.index)
    pairs = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = donors[left] & donors[right]
            if len(shared) >= minimum:
                pairs.append(
                    {
                        "collections": [left, right],
                        "sharedDonorIds": len(shared),
                        "donors": [len(donors[left]), len(donors[right])],
                        "examples": sorted(shared)[:5],
                    }
                )
    return pairs


def label_map(stores: list[dict]) -> list[dict]:
    cells, found = Counter(), defaultdict(set)
    coarse = {}
    for store in stores:
        for item in store.pop("labels"):
            cells[item["label"]] += item["cells"]
            found[item["label"]].add(store["storeId"])
            coarse[item["label"]] = item["coarseCellType"]
    return [
        {"label": label, "coarseCellType": coarse[label], "cells": n, "stores": len(found[label])}
        for label, n in cells.most_common()
    ]


def totals(stores: list[dict], failures: list[dict], units: pd.DataFrame, release: dict) -> dict:
    frame = pd.DataFrame(stores)
    response = sum(store["io"]["http"]["responseBytes"] for store in stores)
    count_bytes = int(frame.countBytes.sum()) if len(frame) else 0
    return {
        "storesEligible": release["eligibleStores"],
        "storesScreened": len(stores),
        "storesFailed": len(failures),
        "storesWithoutSelectedCells": int((frame.selectedCells == 0).sum()) if len(frame) else 0,
        "cellsInScreenedStores": int(frame.cells.sum()) if len(frame) else 0,
        "primaryBloodCellsRead": int(frame.selectedCells.sum()) if len(frame) else 0,
        "cellsWithoutDonor": int(frame.cellsWithoutDonor.sum()) if len(frame) else 0,
        "cellsWithoutCoarseType": int(frame.cellsWithoutCoarseType.sum()) if len(frame) else 0,
        "studyScopedDonors": int(units[["collectionId", "donor_id"]].drop_duplicates().shape[0]),
        "donorsWithScore": int(
            units[(units.cells >= MINIMUM_CELLS) & units.interferonScore.notna()][
                ["collectionId", "donor_id"]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "responseBytes": response,
        "httpRequests": sum(store["io"]["http"]["requests"] for store in stores),
        "storeCountBytes": count_bytes,
        "storeBytes": int(frame.storeBytes.sum()) if len(frame) else 0,
        "responseShareOfStoreCountBytes": response / count_bytes if count_bytes else None,
        "responseShareOfRelease": response / release["bucketBytes"],
        "storeSecondsSum": float(frame.seconds.sum()) if len(frame) else 0.0,
    }


def run(payload: dict) -> dict:
    import cytebase
    from huggingface_hub import HfApi

    if payload.get("storeSet") not in STORE_SETS:
        raise ValueError(f"Set storeSet in the payload to one of {', '.join(STORE_SETS)}")
    configure_cytearc()
    work = Path(payload.get("workDirectory", str(WORK)))
    work.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or False
    if payload.get("requireToken") and not token:
        raise RuntimeError("requireToken is set but HF_TOKEN is not in the environment")
    traffic = HubTraffic(payload.get("quotaReserve", 200))
    sampler = ResourceSampler().start()
    started = time.perf_counter()
    with hub_traffic(traffic):
        catalog = cytebase.Catalog(version=CATALOGUE_VERSION, token=token)
        rows = eligible_stores(catalog, payload["storeSet"])
        oversized = [
            r["cytebase_id"] for r in rows if r["cell_count"] > payload.get("maxStoreCells", np.inf)
        ]
        if oversized:
            raise ValueError(f"Stores above maxStoreCells: {oversized}")
        release = {
            "version": CATALOGUE_VERSION,
            "catalogueSha256": catalogue_sha256(catalog),
            "bucketBytes": int(HfApi().bucket_info(BUCKET, token=token).size),
            "eligibleStores": len(rows),
            "eligibleCells": int(sum(r["cell_count"] for r in rows)),
            "eligiblePrimaryCells": int(sum(r["primary_cell_count"] for r in rows)),
            "eligibleCollections": len({r["collection_id"] for r in rows}),
        }
        excluded = {
            r["cytebase_id"]: EXCLUDED_STORES[r["cytebase_id"]]
            for r in rows
            if r["cytebase_id"] in EXCLUDED_STORES
        }
        screened = [r for r in rows if r["cytebase_id"] not in excluded]
        stores, failures, groups = [], [], []
        with ThreadPoolExecutor(max_workers=payload.get("storeConcurrency", 4)) as pool:
            futures = [
                pool.submit(screen_with_retries, row, traffic, token, payload) for row in screened
            ]
            for future in as_completed(futures):
                result = future.result()
                if "failure" in result:
                    failures.append(result["failure"])
                    continue
                stores.append(result["store"])
                groups.append(result["groups"])
        for store in [*stores, *failures]:
            store["io"] = traffic.store_summary(store["storeId"])
    wall = time.perf_counter() - started
    catalogue_rows = {r["cytebase_id"]: r for r in rows}
    for store in stores:
        row = catalogue_rows[store["storeId"]]
        store.update(
            collectionId=row["collection_id"],
            author=row["first_author"],
            year=row["year"],
            assays=list(row["assay_labels"]),
            catalogueCells=int(row["cell_count"]),
            cataloguePrimaryCells=int(row["primary_cell_count"]),
        )
    columns = [
        "storeId",
        "collectionId",
        "donor_id",
        "disease",
        "sex",
        "development_stage",
        "tissue",
        "sample",
        "sampleTime0",
        "sampleTime1",
        "coarse_cell_type",
        "cells",
        "librarySize",
    ]
    columns += [f"{kind}_{symbol}" for symbol in PANEL for kind in ("count", "detected")]
    columns += CELL_SCORE_COLUMNS
    table = (
        pd.concat(groups, ignore_index=True).reindex(columns=columns)
        if groups
        else pd.DataFrame(columns=columns)
    )
    table["interferonScore"] = interferon_score(table)
    units, dropped = donor_units(table)
    units["earliestSample"] = mark_earliest_samples(units)
    units["interferonZScore"] = interferon_zscores(units)
    labels = label_map(stores)
    result = {
        "settings": {key: value for key, value in payload.items() if key != "r2EndpointUrl"},
        "authenticated": bool(token),
        "sourceStorageOptions": {"token": "HF_TOKEN" if token else False, "block_size": 1},
        "release": release,
        "selection": {
            "organism": "Homo sapiens",
            "tissues": list(BLOOD_TISSUES),
            "cells": "is_primary_data and a blood tissue label",
        },
        "panel": {"interferon": INTERFERON_GENES, "lineage": LINEAGE_GENES},
        "scoreDefinition": f"interferonScore: mean over the {len(INTERFERON_GENES)} interferon genes a store measures of log2(CPM + 1) in a donor-by-sample-by-coarse-type pseudobulk; interferonZScore: the same genes' log2(CPM + 1) z-scored against the store's healthy donors of the coarse type (earliest samples, at least {MINIMUM_CELLS} cells, at least {MINIMUM_HEALTHY_DONORS} donors) and averaged over the genes that vary among them; the library is the sum of the stored RNA_nCounts; summaries use each donor's earliest sample and pseudobulks of at least {MINIMUM_CELLS} cells",
        "cellScoreBins": {
            "width": CELL_SCORE_BIN,
            "bins": CELL_SCORE_BINS,
            "definition": "each cell's log2(CPM + 1) of its summed interferon gene counts, counted per pseudobulk into bins of this width from zero; the last bin also holds higher values; cells without a library are not counted",
        },
        "sampleRules": {
            "source": "experiments/runtime/blood_samples.py",
            "excludedStores": excluded,
            "storesWithRules": sorted(s["storeId"] for s in stores if s["sampleRule"]),
        },
        "cellTypeMap": "blood_cell_type (experiments/runtime/annotations.py), with mast cells as Granulocytes and malignant cells unassigned",
        "wallSeconds": wall,
        "traffic": traffic.totals(),
        "totals": totals(stores, failures, units, release),
        "stores": sorted(stores, key=lambda s: -s["catalogueCells"]),
        "failures": failures,
        "duplicateUnitsDropped": dropped,
        "sharedDonorIdsAcrossCollections": shared_donor_ids(units),
        "scores": summarize_scores(units),
        "labelMap": labels,
        "resources": asdict(sampler.stop()),
    }
    prefix = (payload.get("outputPrefix") or "").rstrip("/")
    paths = [work / "catalogue_screen_groups.csv.gz", work / "catalogue_screen_donors.csv.gz"]
    result["tables"] = {}
    for key, frame, path in (("groups", table, paths[0]), ("donors", units, paths[1])):
        frame.to_csv(path, index=False, compression="gzip")
        result["tables"][key] = {
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **({"uri": f"{prefix}/{path.name}"} if prefix else {}),
        }
    record_path = write_json("catalogue_screen.json", result, directory=work)
    if prefix:
        from experiments.runtime.object_io import upload_file

        for path in [*paths, record_path]:
            upload_file(path, f"{prefix}/{path.name}")
    if len(failures) > payload.get("maxFailedStores", 0):
        raise RuntimeError(
            f"{len(failures)} stores failed: {[failure['storeId'] for failure in failures]}"
        )
    return result


def store_set_of(store_ids: list[str]) -> str:
    """The store set that a job's stores make up."""
    cohort = set(store_ids) & set(IMMUNE_COHORT_STORES)
    if cohort == set(store_ids) == set(IMMUNE_COHORT_STORES):
        return "immune_cohorts"
    if not cohort:
        return "other_blood_stores"
    raise ValueError(
        f"The job screened {len(cohort)} of the {len(IMMUNE_COHORT_STORES)} immune cohort stores "
        f"and {len(set(store_ids) - cohort)} other stores; a job screens one whole store set"
    )


def screen_row(run: dict) -> dict:
    """One job of the screen, from the run record that the Modal runner saves."""
    if run.get("status") != "completed":
        raise ValueError(f"The run record has status {run.get('status')!r}, expected 'completed'")
    result = run["result"]
    missing = [key for key in ROW_SETTINGS if key not in result["settings"]]
    if missing:
        raise ValueError(f"The job's payload does not set {missing}")
    store_ids = sorted(
        [store["storeId"] for store in result["stores"]]
        + [failure["storeId"] for failure in result["failures"]]
    )
    return {
        "storeSet": store_set_of(store_ids),
        "storeIds": store_ids,
        "region": run["region"],
        "authenticated": result["authenticated"],
        **{key: result["settings"][key] for key in ROW_SETTINGS},
        "cytebaseRelease": result["release"]["version"],
        "catalogueSha256": result["release"]["catalogueSha256"],
        **{key: result["totals"][key] for key in ROW_TOTALS},
        "wallSeconds": result["wallSeconds"],
        "rateLimitWaits": sum(kind["waits"] for kind in result["traffic"]["quotaWaits"].values()),
        "cgroupMemoryCurrentPeakBytes": result["resources"]["cgroupMemoryCurrentPeakBytes"],
        "memoryMaxBytes": result["resources"]["memoryMaxBytes"],
    }


def read_run(source: str) -> dict:
    if source.startswith("s3://"):
        from experiments.runtime.object_io import get_json

        return get_json(source)
    return json.loads(Path(source).read_text())


def collect(runs: list[dict]) -> dict:
    """The blood store screen record: the gene panel and one row per job, in store-set order."""
    panels = {json.dumps(run["result"]["panel"]) for run in runs}
    if len(panels) != 1:
        raise ValueError("The jobs screened different gene panels")
    rows = sorted(
        (screen_row(run) for run in runs), key=lambda row: STORE_SETS.index(row["storeSet"])
    )
    sets = [row["storeSet"] for row in rows]
    if len(set(sets)) != len(sets):
        raise ValueError(f"More than one job screened the same store set: {sets}")
    return {"panel": runs[0]["result"]["panel"], "jobs": rows}


def job_store_set(run: dict) -> str:
    """The store set a completed job screened, from the stores it screened and the stores that
    failed."""
    if run.get("status") != "completed":
        raise ValueError(f"The run record has status {run.get('status')!r}, expected 'completed'")
    result = run["result"]
    return store_set_of(
        [store["storeId"] for store in result["stores"]]
        + [failure["storeId"] for failure in result["failures"]]
    )


def keep(run: dict, screen: str) -> Path:
    """Write one job's run record, without its call ID, start time and location settings, and its
    donor table into records/<screen>/ under the job's store set. The table is downloaded unless
    records/ holds it already with the sha256 the job recorded."""
    if screen not in SCREENS:
        raise ValueError(f"screen is one of {', '.join(SCREENS)}, not {screen!r}")
    folder = HERE / "records" / screen
    folder.mkdir(parents=True, exist_ok=True)
    store_set = job_store_set(run)
    donors = run["result"]["tables"]["donors"]
    table = folder / f"{store_set}_donors.csv.gz"
    if not table.exists() or hashlib.sha256(table.read_bytes()).hexdigest() != donors["sha256"]:
        from experiments.runtime.object_io import download_file

        download_file(donors["uri"], table)
        if hashlib.sha256(table.read_bytes()).hexdigest() != donors["sha256"]:
            raise ValueError(f"{table.name} is not the donor table the job recorded")
    return write_json_at(folder / f"{store_set}.json", public_run(run))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("run", help="screen one store set on this machine")
    local.add_argument("payload", type=Path)
    gather = commands.add_parser("collect", help=f"write records/{RECORD.name} from run records")
    gather.add_argument("runs", nargs="+", help="s3:// URI or local path of each job's run record")
    kept = commands.add_parser(
        "keep", help="write one job's run record and donor table into records/<screen>/"
    )
    kept.add_argument("run", help="s3:// URI or local path of the job's run record")
    kept.add_argument("--screen", choices=SCREENS, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        write_json_at(RECORD, collect([read_run(source) for source in args.runs]))
        return
    if args.command == "keep":
        keep(read_run(args.run), args.screen)
        return
    print(json.dumps(run(json.loads(args.payload.read_text()))["totals"], indent=2))


if __name__ == "__main__":
    main()
