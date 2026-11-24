from experiments.benchmarks import DATA
from experiments.benchmarks.scaling import attempts, collect_runs, profiling_runs
from experiments.runtime.common import GIB, write_csv
from experiments.runtime.records import STAGES


def layouts(runs: dict[str, dict]) -> list[dict]:
    chosen: dict[tuple[int, int], dict] = {}
    for run in profiling_runs():
        if run.group != "rna_scaling_runs":
            continue
        for tag in attempts(run):
            if (runs.get(tag) or {}).get("funnel"):
                chosen[run.size, run.replicate] = {
                    "tag": tag,
                    "nRows": run.size,
                    "replicate": run.replicate,
                    "cpus": float(run.cpu),
                    "containerMb": run.gib * 1024,
                    "budgetBytes": run.budgetGib * GIB,
                    "workers": run.workers,
                }
    return [chosen[key] for key in sorted(chosen)]


def main() -> None:
    stage_rows = []
    report_rows = []
    collect_runs("rna_resource_runs")
    runs = collect_runs("rna_scaling_runs")
    for layout in layouts(runs):
        tag, n_rows = (layout["tag"], layout["nRows"])
        for stage in STAGES:
            record = runs[tag]["stages"].get(stage)
            if record is None:
                print(f"{tag}: {stage} missing")
                continue
            details = record.get("details") or {}
            ops = details.get("storeOperations") or {}
            stage_rows.append(
                {
                    "tag": tag,
                    "nRows": n_rows,
                    "stage": stage,
                    "status": record["status"],
                    "seconds": record["seconds"],
                    "wholeFunctionSeconds": record.get("wholeFunctionSeconds"),
                    "inputSetupSeconds": record.get("inputSetupSeconds"),
                    "peakCgroupBytes": record.get("peakCgroupBytes"),
                    "peakRssBytes": record.get("peakRssBytes"),
                    "processCpuSeconds": record.get("processCpuSeconds"),
                    "childCpuSeconds": record.get("childCpuSeconds"),
                    "readTransferredBytes": ops.get("readTransferredBytes"),
                    "writeTransferredBytes": ops.get("writeTransferredBytes"),
                    "gets": ops.get("gets"),
                    "sets": ops.get("sets"),
                    "maxInFlight": ops.get("maxInFlight"),
                }
            )
            for unit_kind, reports in (details.get("executionReports") or {}).items():
                for index, report in enumerate(reports):
                    report_rows.append(
                        {
                            "tag": tag,
                            "nRows": n_rows,
                            "stage": stage,
                            "unitKind": unit_kind,
                            "reportIndex": index,
                            "fetchSeconds": report.get("fetchSeconds"),
                            "computeSeconds": report.get("computeSeconds"),
                            "writeSeconds": report.get("writeSeconds"),
                            "readerWaitSeconds": report.get("readerWaitSeconds"),
                            "computeWaitSeconds": report.get("computeWaitSeconds"),
                            "unitsCompleted": report.get("unitsCompleted"),
                            "unitBytes": report.get("unitBytes"),
                            "effectiveReadWorkers": report.get("effectiveReadWorkers"),
                            "effectiveComputeWorkers": report.get("effectiveComputeWorkers"),
                            "effectiveWriteWorkers": report.get("effectiveWriteWorkers"),
                            "threadsPerComputeWorker": report.get("threadsPerComputeWorker"),
                            "reservedBytes": report.get("reservedBytes"),
                            "peakHeldBytes": report.get("peakHeldBytes"),
                            "requestedMemoryBytes": report.get("requestedMemoryBytes"),
                            "reductionReason": report.get("reductionReason"),
                        }
                    )
    write_csv("rna_scaling_stage_measurements.csv", stage_rows, directory=DATA)
    write_csv("rna_scaling_execution_reports.csv", report_rows, directory=DATA)


if __name__ == "__main__":
    main()
