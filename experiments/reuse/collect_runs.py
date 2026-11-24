"""Write the reuse records from the run records that the jobs save.

The Modal runner saves each reuse job's run record at the payload's recordUri, and the Scanpy and
BPCells jobs save their reports at the payload's outputUri. This step reads one such record, from
the experiment bucket or a local file, keeps the fields the records hold and writes it into
records/. For the progenitor discovery and analysis it also downloads the tables the job uploaded
beside its record.
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from experiments.reuse import HERE
from experiments.runtime.common import (
    LOCATION_SETTINGS,
    public_run,
    scrub_private,
    write_json_at,
)

RECORDS = HERE / "records"
# The run record fields of the Modal runner that the composite_and_remote records keep.
RUN_FIELDS = ("module", "function", "region", "startedAt", "status", "result", "elapsedSeconds")
# The settings of the ten-study cohort analysis that its records keep.
COHORT_SETTINGS = ("sources", "memBudget", "nthreads", "hvgCount", "pcaDims", "neighborsK")
# The tables of the progenitor analysis that records/ keeps; its per-cell arrays stay in the bucket.
PROGENITOR_ANALYSIS_TABLES = ("progenitor_cell_table.parquet", "progenitor_cluster_panel.parquet")
IO_FIELDS = ("readBytes", "requests", "httpResponseStatusCounts")
MEMORY_FIELDS = (
    "operationBaselineBytes",
    "operationPeakBytes",
    "operationPeakSource",
    "memoryMaxBytes",
)


def pick(values: dict, keys: Iterable[str]) -> dict:
    return {key: values[key] for key in keys}


def completed(run: dict, status: str = "completed") -> dict:
    if run.get("status") != status:
        raise ValueError(f"The run record has status {run.get('status')!r}, expected {status!r}")
    return run


def remote_measurement(measured: dict, *, memory: bool = False) -> dict:
    """Seconds and object-store reads of a measured stage, with its memory when asked."""
    kept = {"elapsedSeconds": measured["elapsedSeconds"], "io": pick(measured["io"], IO_FIELDS)}
    if memory:
        kept["resources"] = pick(measured["resources"], MEMORY_FIELDS)
    return kept


def local_measurement(measured: dict) -> dict:
    """Outcome, seconds, bytes read and written and memory of a measured local-disk step."""
    kept = {
        "status": measured["status"],
        "elapsedSeconds": measured["elapsedSeconds"],
        "processIoBytes": pick(measured["processIoBytes"], ("rchar", "wchar")),
        "resources": pick(measured["resources"], MEMORY_FIELDS),
    }
    if measured["status"] == "failed":
        kept.update(pick(measured, ("errorType", "error")))
    return kept


def blood_cohort_analysis(run: dict) -> dict:
    """The ten-study composite run of reuse.blood_cohort_analysis: creation, the analysis through
    Leiden and the donor pseudobulk."""
    result = completed(run)["result"]
    return {
        **pick(run, ("module", "function", "status")),
        "result": {
            "settings": pick(result["settings"], COHORT_SETTINGS),
            **pick(
                result,
                (
                    "timingBoundary",
                    "sourceCellCounts",
                    "cells",
                    "genesShared",
                    "localStoreAfterCreateBytes",
                ),
            ),
            "creation": remote_measurement(result["creation"]),
            "unintegrated": remote_measurement(result["unintegrated"], memory=True),
            "pseudobulk": pick(result["pseudobulk"], ("groups", "features")),
            "pseudobulkMeasurement": remote_measurement(result["pseudobulkMeasurement"]),
        },
    }


def blood_on_disk_concatenation(run: dict) -> dict:
    """One concatenateOnDisk run of reuse.composite_and_remote on the ten exported H5AD files."""
    result = completed(run)["result"]["onDiskConcatenation"]
    kept = {
        **pick(
            result,
            (
                "function",
                "anndataVersion",
                "join",
                "merge",
                "maxLoadedElems",
                "inputs",
                "timingBoundary",
            ),
        ),
        "staging": pick(result["staging"], ("bytes",)),
        "asExportedAttempt": local_measurement(result["asExportedAttempt"]),
    }
    if result["asExportedAttempt"]["status"] == "failed":
        preparation = result["inputPreparation"]
        kept["inputPreparation"] = {k: v for k, v in preparation.items() if k != "obsKindsByInput"}
        kept["inputPreparationMeasurement"] = local_measurement(
            result["inputPreparationMeasurement"]
        )
    kept["concatenation"] = local_measurement(result["concatenation"])
    kept.update(pick(result, ("outputBytes", "validation")))
    return {**pick(run, ("module", "function", "status")), "result": {"onDiskConcatenation": kept}}


def blood_scanpy_dask(run: dict) -> dict:
    """One run of the Scanpy-on-Dask workflow on the concatenated H5AD file."""
    completed(run, "ok")
    return {
        **pick(run, ("tool", "dataset", "replicate", "status")),
        "container": pick(
            run["container"], ("cpu", "memoryMb", "workers", "preemptible", "region")
        ),
        **pick(run, ("versions", "settings", "input", "mode", "lazyReader", "chunkRows")),
        **pick(run, ("nCells", "nGenes", "nnz", "clusters")),
        "download": pick(run["download"], ("bytes", "seconds")),
        **pick(run, ("workflowSeconds", "peakCgroupBytes", "peakRssBytes", "memorySampleSeconds")),
    }


def composite_and_remote(run: dict) -> dict:
    """One run of reuse.composite_and_remote in remote, composite or concatenate mode, without the
    payload fields that locate the job's inputs and outputs."""
    kept = pick(completed(run), RUN_FIELDS)
    composite = kept["result"].get("composite")
    if composite is not None:
        composite["settings"] = {
            k: v for k, v in composite["settings"].items() if k not in LOCATION_SETTINGS
        }
        for item in composite.get("materializedInputs", []):
            item.pop("uri", None)
    concatenation = kept["result"].get("materializedConcatenation")
    if concatenation is not None:
        for item in concatenation["inputFiles"]:
            item.pop("uri", None)
    return scrub_private(kept)


def blood_cohort_mixing(run: dict) -> dict:
    """The ten-study composite run of reuse.blood_cohort_analysis: how much the studies mix among
    each cell's same-type neighbours in the unintegrated and the Harmony analysis."""
    result = completed(run)["result"]
    return {
        **pick(run, ("module", "function", "status")),
        "result": {
            "settings": pick(result["settings"], COHORT_SETTINGS),
            **pick(result, ("cells", "sourceCellCounts")),
            **{phase: pick(result[phase], ("mixing",)) for phase in ("unintegrated", "integrated")},
        },
    }


def job_run(run: dict) -> dict:
    """A completed job's run record, without its call ID, start time and location settings."""
    return public_run(completed(run))


def tool_report(run: dict) -> dict:
    """A completed run of a comparison tool's workflow, without its call ID and start and finish
    times."""
    return public_run(completed(run, "ok"))


def progenitor_tables(run: dict) -> dict[str, tuple[str, int | None]]:
    """The tables a progenitor job uploaded beside its run record, by the names records/ keeps them
    under, each with its location and the bytes the job recorded: the discovery's candidate cells
    and donor-sample totals, and the analysis's cell table and cluster panel."""
    result = run["result"]
    if "tables" in result:
        return {
            f"progenitor_{key}.parquet": (entry["uri"], entry["bytes"])
            for key, entry in result["tables"].items()
        }
    return {name: (result["outputs"][name], None) for name in PROGENITOR_ANALYSIS_TABLES}


def fetch_table(uri: str, destination: Path, size: int | None) -> None:
    """Download a table a job uploaded, unless records/ holds it already at the bytes the job
    recorded."""
    if destination.exists() and size in (None, destination.stat().st_size):
        print(f"kept {destination.relative_to(HERE.parents[1])}", flush=True)
        return
    from experiments.runtime.object_io import download_file

    download_file(uri, destination)
    if size is not None and destination.stat().st_size != size:
        raise ValueError(
            f"{destination.name} holds {destination.stat().st_size} bytes, but the job recorded {size}"
        )
    print(f"wrote {destination.relative_to(HERE.parents[1])}", flush=True)


KINDS = {
    "blood_cohort_analysis": (blood_cohort_analysis, "blood_cohort_analysis.json"),
    "blood_cohort_mixing": (blood_cohort_mixing, "blood_cohort_mixing.json"),
    "blood_on_disk_concatenation": (
        blood_on_disk_concatenation,
        "blood_on_disk_concatenation/replicate_{replicate}.json",
    ),
    "blood_scanpy_dask": (blood_scanpy_dask, "blood_scanpy_dask/replicate_{replicate}.json"),
    "blood_bpcells": (tool_report, "blood_bpcells/replicate_{replicate}.json"),
    "composite_and_remote": (
        composite_and_remote,
        "composite_and_remote/{set}/replicate_{replicate}.json",
    ),
    "census_query": (job_run, "census_queries/replicate_{replicate}.json"),
    "progenitor_discovery": (job_run, "blood_progenitors/discovery.json"),
    "progenitor_extraction": (job_run, "blood_progenitors/extraction_{job}.json"),
    "progenitor_analysis": (job_run, "blood_progenitors/analysis.json"),
    "progenitor_severity": (job_run, "blood_progenitors/severity.json"),
    "progenitor_source_studies": (job_run, "blood_progenitors/source_studies.json"),
}
# The kinds whose jobs uploaded tables that records/ keeps beside the run record.
TABLE_KINDS = ("progenitor_discovery", "progenitor_analysis")


def read_run(source: str) -> dict:
    if source.startswith("s3://"):
        from experiments.runtime.object_io import get_json

        return get_json(source)
    return json.loads(Path(source).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=sorted(KINDS))
    parser.add_argument("source", help="s3:// URI or local path of the run record")
    parser.add_argument("--replicate", type=int)
    parser.add_argument(
        "--set",
        choices=(
            "remote_queries",
            "composite_integration",
            "blood_composite",
            "blood_concatenation",
        ),
        help="the composite_and_remote record folder",
    )
    parser.add_argument(
        "--job", type=int, help="the progenitor extraction job's share of the stores, from 0"
    )
    args = parser.parse_args()
    trim, name = KINDS[args.kind]
    if "{replicate}" in name and args.replicate is None:
        parser.error(f"{args.kind} records are written per replicate; give --replicate")
    if "{set}" in name and args.set is None:
        parser.error("composite_and_remote records are written per folder; give --set")
    if "{job}" in name and args.job is None:
        parser.error("progenitor extraction records are written per job; give --job")
    run = read_run(args.source)
    path = RECORDS / name.format(replicate=args.replicate, set=args.set, job=args.job)
    record = trim(run)
    if args.kind in TABLE_KINDS:
        for table, (uri, size) in progenitor_tables(run).items():
            fetch_table(uri, path.parent / table, size)
    write_json_at(path, record)


if __name__ == "__main__":
    main()
