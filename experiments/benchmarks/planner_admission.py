import math

import numpy as np
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.execution import AUTO_READ_WIDTH_MULTIPLIER, WorkShape, plan_operation
from cytearc.storage.sharding import plan_count_matrix_pair

from experiments.benchmarks import HERE
from experiments.runtime.common import BENCHMARK_GENES, GIB, write_json

BUDGETS_GIB = (4, 8, 12, 16, 24, 32, 48, 64)
CORES = (4, 8, 16)
SIZES = (1000000, 10000000)
ELEMENT_BYTES = 4


def unit_kinds(n_cells: int) -> list[dict[str, int | str]]:
    plan = plan_count_matrix_pair(n_cells, BENCHMARK_GENES, np.float32, profile="cloud")
    counts_chunks = plan.counts.chunks
    counts_shards = plan.counts.shards or plan.counts.chunks
    counts_t_chunks = plan.countsT.chunks
    return [
        {
            "unitKind": "cellShard",
            "nUnits": math.ceil(n_cells / int(counts_shards[0])),
            "unitBytes": int(counts_shards[0]) * int(counts_shards[1]) * ELEMENT_BYTES,
            "decodeBytes": int(counts_chunks[0]) * int(counts_chunks[1]) * ELEMENT_BYTES,
        },
        {
            "unitKind": "geneReadGroup",
            "nUnits": math.ceil(BENCHMARK_GENES / int(plan.readGroup.featureWidth)),
            "unitBytes": int(plan.readGroup.readGroupBytes),
            "decodeBytes": int(counts_t_chunks[0]) * int(counts_t_chunks[1]) * ELEMENT_BYTES,
        },
    ]


def main() -> None:
    rows = []
    for n_cells in SIZES:
        for kind in unit_kinds(n_cells):
            for budget_gib in BUDGETS_GIB:
                for cores in CORES:
                    budget = ResourceBudget(memoryBytes=budget_gib * GIB, workers=cores)
                    shape = WorkShape(
                        nUnits=int(kind["nUnits"]),
                        unitBytes=int(kind["unitBytes"]),
                        decodeBytes=int(kind["decodeBytes"]),
                    )
                    base = {
                        "nCells": n_cells,
                        "unitKind": kind["unitKind"],
                        "budgetGiB": budget_gib,
                        "cores": cores,
                        "nUnits": kind["nUnits"],
                        "unitBytes": kind["unitBytes"],
                        "decodeBytes": kind["decodeBytes"],
                    }
                    try:
                        plan = plan_operation(budget, shape)
                    except MemoryError as error:
                        rows.append({**base, "status": f"MemoryError: {error}"})
                        continue
                    memory_live = budget.memoryBytes // (
                        int(kind["unitBytes"]) + int(kind["decodeBytes"])
                    )
                    expected_read = min(
                        int(kind["nUnits"]), memory_live, AUTO_READ_WIDTH_MULTIPLIER * cores
                    )
                    rows.append(
                        {
                            **base,
                            "status": "ok",
                            "memoryLiveUnits": memory_live,
                            "unitsInFlight": plan.readWorkers,
                            "unitsInFlightMatchesEq2": plan.readWorkers == expected_read,
                            "computeWorkers": plan.computeWorkers,
                            "threadsPerComputeWorker": plan.threadsPerComputeWorker,
                            "ioConcurrency": plan.ioConcurrency,
                            "reservedBytes": plan.reservedBytes,
                            "reservedFractionOfBudget": round(
                                plan.reservedBytes / budget.memoryBytes, 4
                            ),
                            "reservedWithinBudget": plan.reservedBytes <= budget.memoryBytes,
                            "activeThreadsWithinCores": plan.computeWorkers
                            * plan.threadsPerComputeWorker
                            <= cores,
                            "reductionReason": plan.reductionReason,
                        }
                    )
    violations = [
        r
        for r in rows
        if r.get("status") == "ok"
        and (
            not (
                r["reservedWithinBudget"]
                and r["unitsInFlightMatchesEq2"]
                and r["activeThreadsWithinCores"]
            )
        )
    ]
    write_json(
        "memory_read_admission.json",
        {
            "description": "plan_operation resolved for one cell-major shard unit and one gene-major read group across budgets and cores. unitsInFlight is L of Eq. 2; reservedBytes is R + L(U + D) plus nested-read reservations.",
            "autoReadWidthMultiplier": AUTO_READ_WIDTH_MULTIPLIER,
            "violations": violations,
            "rows": rows,
        },
        directory=HERE / "records",
    )
    print(f"planner rows: {len(rows)}, violations: {len(violations)}")


if __name__ == "__main__":
    main()
