"""Scanpy on Dask reading the one-million-cell embryo from object storage and from local disk, and BPCells reading
it from local disk, beside CyteArc."""

import json
import math

from experiments.benchmarks.timing import analysis_time_seconds
from experiments.runtime.common import DERIVED, ROOT

DATA = ROOT / "experiments/benchmarks/records"
BENCHMARKS_DERIVED = DERIVED / "benchmarks"


DATASET = "embryo_1000000"
ARMS = {
    "scanpy_dask_covariance_csc_remote_zarr": "objectStore",
    "scanpy_dask_covariance_csc_local_zarr": "localDisk",
}
RUNS = DATA / "rna_storage_backend_runs.json"
# BPCells reads local files only: its local-disk arm is the toolkit comparison's BPCells (cached) runs of the same
# dataset on the same 8-CPU, 32 GiB container.
BPCELLS_RUNS = DATA / "toolkit_comparison_runs.json"
BPCELLS_ARM = "bpcells_cached"
SUMMARY = BENCHMARKS_DERIVED / "rna_storage_backend_comparisons.json"
GIB = 1024**3


def welch_log_ratio(a: list[float], b: list[float]) -> dict:
    """Ratio of geometric means a / b, its Welch t interval on log values and the test's P value."""
    from scipy import stats

    la, lb = [math.log(v) for v in a], [math.log(v) for v in b]
    ma, mb = sum(la) / len(la), sum(lb) / len(lb)
    out = {
        "nA": len(a),
        "nB": len(b),
        "ratio": math.exp(ma - mb),
        "low": None,
        "high": None,
        "p": None,
    }
    if len(a) < 2 or len(b) < 2:
        return out
    va = sum((v - ma) ** 2 for v in la) / (len(la) - 1) / len(la)
    vb = sum((v - mb) ** 2 for v in lb) / (len(lb) - 1) / len(lb)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(la) - 1) + vb**2 / (len(lb) - 1))
    t = stats.t.ppf(0.975, df)
    p = float(stats.ttest_ind(la, lb, equal_var=False).pvalue)
    return {
        **out,
        "low": math.exp(ma - mb - t * se),
        "high": math.exp(ma - mb + t * se),
        "p": p,
        "df": df,
    }


def with_speedup(comparison: dict) -> dict:
    """A comparison of a over b with how many times faster a ran: the inverse of its ratio and of its interval."""
    return {
        **comparison,
        "speedup": 1 / comparison["ratio"],
        "speedupLow": None if comparison["high"] is None else 1 / comparison["high"],
        "speedupHigh": None if comparison["low"] is None else 1 / comparison["low"],
    }


def mean_minutes(runs: list[dict]) -> float | None:
    """The mean analysis minutes of the runs that completed, or None when none did."""
    done = [r["analysisMinutes"] for r in runs if r["analysisMinutes"]]
    return sum(done) / len(done) if done else None


def cytearc_analysis_seconds(backend: str) -> list[float]:
    """Analysis times (s) of CyteArc's matched-comparison runs (24 GiB, 8 workers) on one storage backend, as the
    resource comparison names it: objectStore or modalLocalDisk."""
    runs = json.loads((BENCHMARKS_DERIVED / "rna_resource_comparisons.json").read_text())["runs"]
    return [
        float(r["elapsedSeconds"]) - float(r["conversionElapsedSeconds"])
        for r in runs
        if r.get("complete")
        and r["dataset"] == "embryo1M"
        and r["storeBackend"] == backend
        and r["budgetGiB"] == 24
        and r["workers"] == 8
    ]


def arm_runs(records: dict, tool: str) -> list[dict]:
    """The runs of ``tool`` on the one-million-cell embryo, leaving out the memory-tier summaries."""
    runs = []
    for name, record in sorted(records.items()):
        if (
            record.get("dataset") != DATASET
            or record.get("tool") != tool
            or name.endswith("_memory_tiers")
        ):
            continue
        steps = record.get("steps") or {}
        done = record.get("status") == "ok"
        analysis = analysis_time_seconds(steps) if done else None
        runs.append(
            {
                "record": name,
                "replicate": record.get("replicate"),
                "status": record.get("status"),
                "failedStep": None
                if done
                else (record.get("failedStep") or record.get("currentStep")),
                "cpu": (record.get("container") or {}).get("cpu"),
                "memoryGib": ((record.get("container") or {}).get("memoryMb") or 0) / 1024,
                "analysisMinutes": None if analysis is None else analysis / 60,
                "markersMinutes": float((steps.get("findMarkers") or {}).get("seconds") or 0) / 60,
                "peakCgroupGib": float(record.get("peakCgroupBytes") or 0) / GIB,
            }
        )
    return runs


def summarise() -> None:
    records = json.loads(RUNS.read_text())["records"]
    arms: dict[str, dict] = {}
    for tool, backend in ARMS.items():
        arms[tool] = {"backend": backend, "runs": arm_runs(records, tool)}
    bpcells_records = json.loads(BPCELLS_RUNS.read_text())["records"]
    arms[BPCELLS_ARM] = {"backend": "localDisk", "runs": arm_runs(bpcells_records, BPCELLS_ARM)}
    for arm in arms.values():
        arm["meanAnalysisMinutes"] = mean_minutes(arm["runs"])
    remote = [
        r["analysisMinutes"] * 60
        for r in arms["scanpy_dask_covariance_csc_remote_zarr"]["runs"]
        if r["analysisMinutes"]
    ]
    local = [
        r["analysisMinutes"] * 60
        for r in arms["scanpy_dask_covariance_csc_local_zarr"]["runs"]
        if r["analysisMinutes"]
    ]
    cytearc = cytearc_analysis_seconds("objectStore")
    cytearc_local = cytearc_analysis_seconds("modalLocalDisk")
    comparisons = {}
    if remote and cytearc:
        comparisons["cytearcOverScanpyObjectStorage"] = with_speedup(
            welch_log_ratio(cytearc, remote)
        )
    if local and cytearc_local:
        comparisons["cytearcOverScanpyLocalDisk"] = with_speedup(
            welch_log_ratio(cytearc_local, local)
        )
    bpcells = [r["analysisMinutes"] * 60 for r in arms[BPCELLS_ARM]["runs"] if r["analysisMinutes"]]
    if bpcells and cytearc_local:
        comparisons["cytearcOverBpcellsLocalDisk"] = with_speedup(
            welch_log_ratio(cytearc_local, bpcells)
        )
    if remote and local:
        comparisons["scanpyObjectStorageOverLocalDisk"] = with_speedup(
            welch_log_ratio(remote, local)
        )
    document = {
        "description": (
            "Scanpy on Dask (scanpy_dask2_csc) on a Zarr copy of the one-million-cell embryo, read on R2 and "
            "from local disk, 8 CPUs and 32 GiB; BPCells (cached) from local disk on the same container, from the "
            "toolkit comparison's runs; analysis time as in Methods (tool_times.py), minutes; "
            "CyteArc's analysis time on object storage and on local disk from the matched comparison (resource "
            "comparison), seconds in the ratios; each comparison of a over b gives the ratio of geometric means and "
            "its speedup, b over a, as how many times faster a ran."
        ),
        "arms": arms,
        "cytearcObjectStorageAnalysisMinutes": [v / 60 for v in cytearc],
        "cytearcLocalDiskAnalysisMinutes": [v / 60 for v in cytearc_local],
        "cytearcObjectStorageMeanAnalysisMinutes": sum(cytearc) / len(cytearc) / 60
        if cytearc
        else None,
        "cytearcLocalDiskMeanAnalysisMinutes": (
            sum(cytearc_local) / len(cytearc_local) / 60 if cytearc_local else None
        ),
        "comparisons": comparisons,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    for tool, arm in arms.items():
        times = ", ".join(
            f"{r['analysisMinutes']:.1f}"
            if r["analysisMinutes"]
            else f"{r['status']} in {r['failedStep']}"
            for r in arm["runs"]
        )
        print(f"{tool}: {times or 'no runs'}")
    for name, c in comparisons.items():
        interval = (
            "" if c["low"] is None else f" ({c['low']:.2f} to {c['high']:.2f}), P = {c['p']:.2g}"
        )
        print(f"{name}: {c['ratio']:.3f}{interval}")


def main() -> None:
    summarise()


if __name__ == "__main__":
    main()
