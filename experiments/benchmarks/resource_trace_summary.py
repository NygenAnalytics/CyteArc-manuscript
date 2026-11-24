"""Each analysis stage's highest network receive rate and highest count of busy CPU cores in the per-second resource
trace of the ten-million-cell scaling run, from records/resource_trace_10000000_cells/, into
.build/derived/benchmarks/resource_trace_10000000_cells.json.

    uv run --locked python -m experiments.benchmarks.resource_trace_summary
"""

from experiments.benchmarks import HERE
from experiments.benchmarks.analysis_times import STAGES
from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

TRACE = HERE / "records" / "resource_trace_10000000_cells" / "resourceTrace.json"
SUMMARY = DERIVED / "benchmarks" / "resource_trace_10000000_cells.json"
COLUMNS = ["seconds", "memoryBytes", "cpuSeconds", "networkReceivedBytes", "stage"]
# The resource trace starts at filtering: the import and the reopening of the written store before it are left out.
TRACE_LEFT_OUT = ("createStore", "writeCountsT", "initializeStore", "reopenStore")
# UMAP's initialisation runs between PCA and the neighbour search, so the trace draws it with the neighbour search
# rather than as a second UMAP span.
TRACE_NAMES = {"buildEmbeddingInitialization": "neighbors"}


def stage_peaks(trace: dict, column: int, what: str) -> list[dict]:
    """For each analysis stage, in workflow order, the highest rate at which the trace's cumulative ``column`` grew over
    an interval between two samples from the start of filtering, per second, and the seconds since filtering began at
    the end of that interval. An interval belongs to the stage running at its closing sample; ``what`` names the
    column in errors."""
    if trace["columns"][: len(COLUMNS)] != COLUMNS:
        raise ValueError(f"unexpected trace columns {trace['columns']}")
    rows = trace["rows"]
    first = next((i for i, row in enumerate(rows) if row[4] not in TRACE_LEFT_OUT), None)
    if first is None:
        raise ValueError("the trace holds no analysis step")
    rows = rows[first:]
    stage_of = {step: name for name, by_tool in STAGES for step in by_tool["cytearc"]} | TRACE_NAMES
    unknown = sorted({row[4] for row in rows} - set(stage_of))
    if unknown:
        raise ValueError(f"the trace names steps that no analysis stage holds: {unknown}")
    if any(row[0] is None or row[column] is None for row in rows):
        raise ValueError(f"a trace sample lacks its time or its {what}")
    peaks: dict[str, dict] = {}
    for before, after in zip(rows, rows[1:]):
        rate = (after[column] - before[column]) / (after[0] - before[0])
        stage = stage_of[after[4]]
        if stage not in peaks or rate > peaks[stage]["peak"]:
            peaks[stage] = {"stage": stage, "peak": rate, "atSeconds": after[0] - rows[0][0]}
    return [peaks[name] for name, _ in STAGES if name in peaks]


def stage_receive_peaks(trace: dict) -> list[dict]:
    """For each analysis stage, its highest network receive rate in bytes per second (stage_peaks)."""
    return [
        {"stage": p["stage"], "peakReceiveBytesPerSecond": p["peak"], "atSeconds": p["atSeconds"]}
        for p in stage_peaks(trace, COLUMNS.index("networkReceivedBytes"), "received bytes")
    ]


def stage_busy_core_peaks(trace: dict) -> list[dict]:
    """For each analysis stage, its highest count of busy CPU cores: CPU seconds per second of an interval
    (stage_peaks)."""
    return [
        {"stage": p["stage"], "peakBusyCores": p["peak"], "busyCoresAtSeconds": p["atSeconds"]}
        for p in stage_peaks(trace, COLUMNS.index("cpuSeconds"), "CPU seconds")
    ]


def stage_summary(trace: dict) -> list[dict]:
    """Each analysis stage's network receive peak and busy-core peak, in workflow order."""
    cores = {p["stage"]: p for p in stage_busy_core_peaks(trace)}
    return [{**p, **cores[p["stage"]]} for p in stage_receive_peaks(trace)]


def main() -> None:
    write_json_at(
        SUMMARY,
        {"source": str(TRACE.relative_to(ROOT)), "stages": stage_summary(load_json(TRACE))},
    )


if __name__ == "__main__":
    main()
