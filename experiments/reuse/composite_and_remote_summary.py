"""Means over the replicates in records/composite_and_remote/: the Tabula Sapiens remote query on first
open and repeated on the open store, and Harmony integration of the four-study composite with its
upstream artifacts reused and on a fresh composite. Beside them, the same query against CELLxGENE Census
and the download of the source H5AD, each over the remote query; the local disk and peak memory of
the ten-study virtual composite against three ways to assemble the same cohort physically; and the
mounted ten-million-cell store's copy, branch and disk against its source, with the minutes of the run that wrote
the store.

    uv run --locked python -m experiments.reuse.composite_and_remote_summary
"""

from pathlib import Path
from statistics import mean

from experiments.runtime.common import DERIVED, GIB, ROOT, load_json, write_json_at
from experiments.runtime.records import ANALYSIS_STAGES, elapsed_seconds

REUSE = ROOT / "experiments" / "reuse" / "records"
RECORDS = REUSE / "composite_and_remote"
# The same Tabula Sapiens query against CELLxGENE Census, three runs placed as the CyteBase query's.
CENSUS_QUERIES = REUSE / "census_queries"
# The ten-study cohort assembled physically: AnnData's concatenation on disk, Scanpy on Dask over its output, and
# BPCells (cached) over the same H5AD files.
ON_DISK_CONCATENATION = REUSE / "blood_on_disk_concatenation"
SCANPY_DASK = REUSE / "blood_scanpy_dask"
BPCELLS = REUSE / "blood_bpcells"
# The benchmarks experiment's download of the Tabula Sapiens source H5AD inside a cloud container.
WORKFLOW_RUNS = ROOT / "experiments" / "benchmarks" / "records" / "rna_workflow_runs.json"
# The reanalysis of the mounted ten-million-cell embryo store, and the benchmark scaling runs, one of which wrote the
# store and records its source H5AD's bytes and its stage times.
MOUNTED_STORE = REUSE / "embryo_endothelial_reanalysis.json"
SCALING_RUNS = ROOT / "experiments" / "benchmarks" / "records" / "raw" / "rna_scaling_runs.json"
# The reanalysis's branch of the published analysis at k = 21: its new neighbor query, graph and Leiden clustering.
K21_BRANCH = ("reuse.neighbors.k21", "reuse.graph.k21", "reuse.leiden.k21_r1")
# The last step of the lineage reanalysis, before the requests on the published analysis.
LINEAGE_END = "umap.k11"
# The published run's clustering chain: its analysis stages through Leiden, without the UMAP, the UMAP's
# initialization and the marker search.
CLUSTERING_STAGES = tuple(
    stage
    for stage in ANALYSIS_STAGES
    if stage not in ("buildEmbeddingInitialization", "runUmap", "findMarkers")
)
SUMMARY = DERIVED / "reuse" / "composite_and_remote.json"


def records_in(folder: Path) -> list[dict]:
    paths = sorted(folder.glob("replicate_*.json"))
    if not paths:
        raise ValueError(f"No records in {folder.relative_to(ROOT)}")
    return [load_json(path) for path in paths]


def replicates(folder: str) -> list[dict]:
    return [record["result"] for record in records_in(RECORDS / folder)]


def remote_query(results: list[dict]) -> dict:
    queries = [result["remoteQueries"]["tabulaSapiens1_1M"] for result in results]
    return {
        "replicates": len(queries),
        # Opening the store, then reading the UMAP, the labels and MS4A1 of the B cells.
        "firstOpenAndQuerySeconds": mean(
            q["openSeconds"] + q["embeddingAndMarkerQuerySeconds"] for q in queries
        ),
        # The bytes the store returned to Zarr for the same two calls.
        "firstOpenAndQueryBytes": mean(
            q["openIo"]["readBytes"] + q["embeddingAndMarkerQueryIo"]["readBytes"] for q in queries
        ),
        # The HTTP response-body bytes the same two calls received over the network, together and each alone.
        "firstOpenAndQueryHttpBytes": mean(
            q["openIo"]["http"]["responseBytes"]
            + q["embeddingAndMarkerQueryIo"]["http"]["responseBytes"]
            for q in queries
        ),
        "openHttpBytes": mean(q["openIo"]["http"]["responseBytes"] for q in queries),
        "queryHttpBytes": mean(
            q["embeddingAndMarkerQueryIo"]["http"]["responseBytes"] for q in queries
        ),
        "repeatedQuerySeconds": mean(q["repeatedQuery"]["elapsedSeconds"] for q in queries),
        "repeatedQueryHttpBytes": mean(
            q["repeatedQuery"]["io"]["http"]["responseBytes"] for q in queries
        ),
    }


def query_alternatives(results: list[dict], census_records: list[dict], source: dict) -> dict:
    """The remote query's alternatives: the same cell-type and MS4A1 query against Census, which reads no UMAP as Census
    keeps none, and downloading the source H5AD; each with its seconds and bytes, and both over the remote query's on
    first open, its bytes as returned to Zarr and as HTTP response bodies."""
    queries = [result["remoteQueries"]["tabulaSapiens1_1M"] for result in results]
    cells = {(q["totalCells"], q["bCellsSelected"]) for q in queries}
    if len(cells) != 1:
        raise ValueError("The remote queries selected different cells")
    ((total, selected),) = cells
    if any(record["status"] != "completed" for record in census_records):
        raise ValueError("A Census query did not complete")
    census = [record["result"] for record in census_records]
    if any((c["totalCells"], c["cellsSelected"]) != (total, selected) for c in census):
        raise ValueError("The Census query read other cells than the CyteBase query")
    if source["nCells"] != total:
        raise ValueError("The downloaded H5AD holds other cells than the queried store")
    query = remote_query(results)
    seconds, received = query["firstOpenAndQuerySeconds"], query["firstOpenAndQueryBytes"]
    transferred = query["firstOpenAndQueryHttpBytes"]
    census_seconds = mean(c["openSeconds"] + c["querySeconds"] for c in census)
    census_bytes = mean(c["openReceivedBytes"] + c["queryReceivedBytes"] for c in census)
    return {
        "census": {
            "replicates": len(census),
            "seconds": census_seconds,
            "bytes": census_bytes,
            "timeOverQuery": census_seconds / seconds,
            "dataOverQuery": census_bytes / received,
            "dataOverQueryHttp": census_bytes / transferred,
        },
        "download": {
            "seconds": source["downloadSeconds"],
            "bytes": source["sourceBytes"],
            "timeOverQuery": source["downloadSeconds"] / seconds,
            "dataOverQuery": source["sourceBytes"] / received,
            "dataOverQueryHttp": source["sourceBytes"] / transferred,
        },
    }


def composite_integration(results: list[dict]) -> dict:
    composites = [result["composite"] for result in results]
    if not all(all(c["sharedUpstreamArtifacts"].values()) for c in composites):
        raise ValueError("An integration run did not reuse every upstream artifact")
    reused = mean(c["integrationReusingUpstream"]["elapsedSeconds"] for c in composites)
    fresh = mean(c["integrationFresh"]["elapsedSeconds"] for c in composites)
    # Each run's route through a local copy: materializing the composite as a local store, then analysing that store.
    materialized = [
        c["materialization"]["elapsedSeconds"] + c["materializedAnalysis"]["elapsedSeconds"]
        for c in composites
    ]
    return {
        "replicates": len(composites),
        "reusedIntegrationSeconds": reused,
        "freshIntegrationSeconds": fresh,
        "timeReductionFraction": 1 - reused / fresh,
        "materializedRouteSecondsLowest": min(materialized),
        "materializedRouteSecondsHighest": max(materialized),
    }


def composite_footprint(
    composites: list[dict],
    in_memory: list[dict],
    on_disk: list[dict],
    scanpy: list[dict],
    bpcells: list[dict],
) -> dict:
    """Local disk and peak container memory of the ten-study virtual composite through Leiden, and of three ways to
    assemble the same cohort physically: AnnData's concatenation in memory, with no analysis after it, AnnData's
    concatenation on disk followed by Scanpy on Dask, and BPCells; with each route's disk and memory over the
    composite's. A route's disk is the H5AD files it reads, with what the on-disk concatenation writes, and for BPCells
    the peak of its work directory, which holds the H5AD files, their copies and its own matrices."""
    stores = {c["localStoreAfterCreateBytes"] for c in composites}
    if len(stores) != 1 or any(c["copiedCountChunkBytes"] for c in composites):
        raise ValueError("The composite runs differ in store bytes or copied count chunks")
    input_bytes = {sum(f["bytes"] for f in run["inputFiles"]) for run in in_memory}
    input_bytes |= {sum(f["bytes"] for f in run["inputs"]) for run in on_disk}
    input_bytes |= {run["download"]["bytes"] for run in bpcells}
    if len(input_bytes) != 1:
        raise ValueError("The physical routes read different H5AD files")
    written = {run["concatenation"]["processIoBytes"]["wchar"] for run in on_disk}
    if len(written) != 1:
        raise ValueError("The on-disk concatenation runs wrote different bytes")
    if any(run["status"] != "ok" for run in scanpy + bpcells):
        raise ValueError("A Scanpy (Dask) or BPCells run did not complete")
    work_directory = [
        run["peakWorkDirectoryBytes"] for run in bpcells if "peakWorkDirectoryBytes" in run
    ]
    if not work_directory:
        raise ValueError("No BPCells run recorded its disk")
    (store,), (h5ad,), (concatenated,) = stores, input_bytes, written
    composite = {
        "replicates": len(composites),
        "diskBytes": store,
        "memoryGiB": mean(c["unintegrated"]["resources"]["operationPeakBytes"] for c in composites)
        / GIB,
    }

    def route(disk: float, peaks: list[float], **counts: int) -> dict:
        memory = mean(peaks) / GIB
        return {
            **counts,
            "diskBytes": disk,
            "memoryGiB": memory,
            "diskOverComposite": disk / store,
            "memoryOverComposite": memory / composite["memoryGiB"],
        }

    return {
        "composite": composite,
        "inMemoryConcatenation": route(
            h5ad,
            [run["resources"]["operationPeakBytes"] for run in in_memory],
            replicates=len(in_memory),
        ),
        "onDiskConcatenationThenScanpyDask": route(
            h5ad + concatenated,
            [run["peakCgroupBytes"] for run in scanpy],
            replicates=len(scanpy),
        ),
        "bpcells": route(
            mean(work_directory),
            [run["peakCgroupBytes"] for run in bpcells],
            replicates=len(bpcells),
            diskReplicates=len(work_directory),
        ),
    }


def mounted_store(record: dict, published_run: dict) -> dict:
    """The mounted store's reanalysis: the bytes the mount copies against the source store's, the minutes of the
    branch at k = 21 on the published analysis and the bytes it writes to the user store, and the user store after the lineage reanalysis against the source
    H5AD that the store was written from. Beside them, the minutes of ``published_run``, the scaling run that wrote
    the store: its clustering chain through Leiden, its analysis through the marker search, and the whole run from
    the source H5AD with the input download left out."""
    if not record["source"]["unchanged"]:
        raise ValueError("The reanalysis changed its source store")
    funnel = published_run["funnel"]
    if funnel["runTag"] != record["referenceTag"]:
        raise ValueError(
            f"the reanalysis mounted the store of {record['referenceTag']}, not of {funnel['runTag']}"
        )
    stages = published_run["stages"]
    unfinished = [s for s in ANALYSIS_STAGES if stages.get(s, {}).get("status") != "ok"]
    if funnel["status"] != "ok" or unfinished:
        raise ValueError(
            f"the published run {funnel['runTag']} did not complete; unfinished stages: {unfinished}"
        )
    source = record["source"]["before"]["bytes"]
    mounted = record["userStore"]["afterMount"]["bytes"]
    lineage = record["steps"][LINEAGE_END]["userStoreBytesAfter"]
    source_h5ad_bytes = funnel["datasetBytes"]
    return {
        "sourceStoreBytes": source,
        "mountBytes": mounted,
        "mountShareOfSource": mounted / source,
        "k21BranchMinutes": sum(record["steps"][step]["seconds"] for step in K21_BRANCH) / 60,
        "k21BranchBytes": sum(
            record["steps"][step]["userStoreBytesWritten"] for step in K21_BRANCH
        ),
        "lineageUserStoreBytes": lineage,
        "sourceH5adBytes": source_h5ad_bytes,
        "sourceH5adOverLineageUserStore": source_h5ad_bytes / lineage,
        "publishedClusteringMinutes": sum(stages[s]["seconds"] for s in CLUSTERING_STAGES) / 60,
        "publishedAnalysisMinutes": sum(stages[s]["seconds"] for s in ANALYSIS_STAGES) / 60,
        "publishedRunMinutes": elapsed_seconds(funnel) / 60,
    }


def main() -> None:
    remote = replicates("remote_queries")
    mount = load_json(MOUNTED_STORE)
    write_json_at(
        SUMMARY,
        {
            "source": [
                str(path.relative_to(ROOT))
                for path in (
                    RECORDS,
                    CENSUS_QUERIES,
                    ON_DISK_CONCATENATION,
                    SCANPY_DASK,
                    BPCELLS,
                    WORKFLOW_RUNS,
                    MOUNTED_STORE,
                    SCALING_RUNS,
                )
            ],
            "remoteQuery": remote_query(remote),
            "queryAlternatives": query_alternatives(
                remote,
                records_in(CENSUS_QUERIES),
                load_json(WORKFLOW_RUNS)["inputs"]["tabula_sapiens"],
            ),
            "compositeIntegration": composite_integration(replicates("composite_integration")),
            "compositeFootprint": composite_footprint(
                [result["composite"] for result in replicates("blood_composite")],
                [
                    result["materializedConcatenation"]
                    for result in replicates("blood_concatenation")
                ],
                [
                    record["result"]["onDiskConcatenation"]
                    for record in records_in(ON_DISK_CONCATENATION)
                ],
                records_in(SCANPY_DASK),
                records_in(BPCELLS),
            ),
            "mountedStore": mounted_store(mount, load_json(SCALING_RUNS)[mount["referenceTag"]]),
        },
    )


if __name__ == "__main__":
    main()
