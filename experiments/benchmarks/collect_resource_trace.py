"""Write a scaling run's resource trace, and the resources its stages ran with, into records/.

A scaling run whose payload sets ``resourceTrace`` samples the container's memory, the CPU seconds
its processes used and the bytes it received over the network once a second, with the stage
running at each sample, and saves the trace as resourceTrace.json beside the run's summary in the
experiment bucket. This step reads the run's configuration, fetches the trace and writes
records/resource_trace_<cells>_cells/: resourceTrace.json, the trace as the run saved it, and
resources.json, the run's tag and cells and, for each stage it ran, CyteArc's memory budget and the
container's CPU limit.

  uv run --locked python -m experiments.benchmarks.collect_resource_trace experiments/benchmarks/configs/benchmark/embryo_rna_scaling_10000000_cells_resource_trace_replicate_1.toml
"""

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.benchmarks import HERE
from experiments.benchmarks.config import load_profiling_config

RECORDS = HERE / "records"


def run_size(config: Any) -> int:
    if len(config.targetSizes) != 1:
        raise ValueError(f"a traced run analyses one size, not {list(config.targetSizes)}")
    return config.targetSizes[0]


def trace_uri(config: Any) -> str:
    """Where the run saved its trace: beside its summary, funnel.json."""
    return (
        config.workflowResultUri(run_size(config)).removesuffix("funnel.json")
        + "resourceTrace.json"
    )


def stage_resources(config: Any) -> dict:
    """The run's tag and cells and, for each stage it ran, CyteArc's memory budget and the
    container's CPU limit."""
    return {
        "runTag": config.runTag,
        "cells": run_size(config),
        "stages": {
            stage: {
                "memoryBudgetBytes": config.resourcesFor(stage).cytearcMemoryBudget,
                "cpuLimit": config.resourcesFor(stage).modalCpuLimit,
            }
            for stage in config.effectiveStages
        },
    }


def write_trace(trace: dict, resources: dict, folder: Path) -> None:
    if not trace.get("rows"):
        raise ValueError("the trace holds no samples")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "resourceTrace.json").write_text(json.dumps(trace) + "\n")
    (folder / "resources.json").write_text(json.dumps(resources, indent=1) + "\n")
    print(f"wrote {folder.relative_to(HERE.parents[1])}: {len(trace['rows'])} samples", flush=True)


def main() -> None:
    from experiments.runtime.object_io import get_json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="the traced run's configuration")
    args = parser.parse_args()
    config = load_profiling_config(args.config)
    resources = stage_resources(config)
    folder = RECORDS / f"resource_trace_{resources['cells']}_cells"
    write_trace(get_json(trace_uri(config)), resources, folder)


if __name__ == "__main__":
    main()
