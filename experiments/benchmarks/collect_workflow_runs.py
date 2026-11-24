"""Collect the CyteArc workflow runs into records/.

Each run is the record that ``workflows.cytearc_tool_job`` wrote to its payload's ``outputUri`` in
the experiment bucket, and each input the record that prepared the dataset next to its H5AD file.

- ``rna_workflow_runs.json``: the complete workflow on six datasets, three replicates per arm
  (standard, matched filters, and memory budgets of 8 and 16 GiB on the one-million-cell embryo
  sample), with the preparation records of their inputs.
- ``memory_preprocessing.json``: input conversion through PCA on the one-million-cell embryo
  sample under budgets of 4, 8, 16 and 24 GiB, three replicates each, with the two count
  policies screened at 4 GiB.
"""

from typing import Any

from experiments.benchmarks import HERE
from experiments.runtime.common import scrub_private, write_json

RECORDS = HERE / "records"
STANDARD = ("brain_1306127", "hbca_neurons", "mssm_cohort", "tabula_sapiens")
REPLICATES = (1, 2, 3)


def _runs(name: str, key: str) -> dict[str, str]:
    return {
        f"rna_workflow_{name}_replicate_{replicate}": key.format(replicate=replicate)
        for replicate in REPLICATES
    }


WORKFLOW_RUNS = {
    **{
        name: key
        for dataset in STANDARD
        for name, key in _runs(
            f"{dataset}_cytearc", f"experiments/workflows/{dataset}_{{replicate}}.json"
        ).items()
    },
    **_runs(
        "embryo_1000000_cytearc", "experiments/workflows/embryo/1000000/standard_{replicate}.json"
    ),
    **_runs(
        "embryo_1000000_cytearc_matched",
        "experiments/workflows/embryo/1000000/matched_{replicate}.json",
    ),
    **_runs(
        "embryo_1000000_cytearc_limit8",
        "experiments/workflows/embryo/1000000/memory8_{replicate}.json",
    ),
    **_runs(
        "embryo_1000000_cytearc_limit16",
        "experiments/workflows/embryo/1000000/memory16_{replicate}.json",
    ),
    **_runs(
        "brain_1306127_cytearc_matched",
        "experiments/matched_filters/brain_1306127_{replicate}.json",
    ),
    **_runs(
        "tabula_sapiens_cytearc_matched",
        "experiments/matched_filters/tabula_sapiens_{replicate}.json",
    ),
    **_runs(
        "embryo_10000000_cytearc",
        "experiments/workflows/embryo/10000000/standard_{replicate}.json",
    ),
    **_runs(
        "embryo_10000000_cytearc_matched",
        "experiments/workflows/embryo/10000000/matched_{replicate}.json",
    ),
}
INPUTS = {
    "brain_1306127": "experiments/datasets/1306127.json",
    "hbca_neurons": "experiments/datasets/external/hbca_neurons.json",
    "tabula_sapiens": "experiments/datasets/external/tabula_sapiens.json",
    "mssm_cohort": "experiments/datasets/external/mssm_cohort.json",
    **{
        f"embryo_{cells}": f"experiments/datasets/embryo/{cells}.json"
        for cells in (1_000_000, 5_000_000, 10_000_000)
    },
}
BUDGETS = (8, 16, 24, 4)
PREPROCESSING_RUNS = {
    f"embryo_{budget}g_r{replicate}": (
        f"experiments/memory/preprocessing/embryo_{budget}g_{replicate}.json"
    )
    for budget in BUDGETS
    for replicate in REPLICATES
}
SCREEN_RUNS = {
    f"embryo_4g_unit{unit}m": f"experiments/memory/preprocessing/embryo_4g_unit{unit}m.json"
    for unit in (125, 250)
}
PREPROCESSING_INPUT = "embryo_1000000"
# Locations, job identifiers, versions, dates, logs and memory traces stay out of the records.
DROPPED = frozenset(
    {
        "datasetUri",
        "callId",
        "versions",
        "startedAt",
        "finishedAt",
        "log",
        "memorySeries",
        "interruptions",
    }
)
INPUT_DROPPED = frozenset({"datasetUri", "localPath", "finishedAt", "preparedAt"})
# The image keeps only how it sets CyteArc's budget: a share of the container or a fixed limit.
IMAGE_BUDGET = ("budgetShare", "memoryLimitGib")


def run_record(output: dict, *, placement: bool) -> dict:
    """A run's record without locations, job identifiers, versions, dates, logs or memory traces.

    The image is reduced to how it sets CyteArc's budget, and the container keeps its region only
    with ``placement``.
    """
    if output.get("status") != "ok":
        raise SystemExit(f"{output.get('tool')} on {output.get('dataset')}: {output.get('status')}")
    record = {key: value for key, value in output.items() if key not in DROPPED}
    record["image"] = {key: value for key, value in output["image"].items() if key in IMAGE_BUDGET}
    if not placement:
        record["container"] = {
            key: value for key, value in output["container"].items() if key != "region"
        }
    return scrub_private(record)


def input_record(prepared: dict) -> dict:
    return scrub_private(
        {key: value for key, value in prepared.items() if key not in INPUT_DROPPED}
    )


def workflow_document(outputs: dict[str, dict], inputs: dict[str, dict]) -> dict[str, Any]:
    return {
        "records": {name: run_record(outputs[name], placement=False) for name in WORKFLOW_RUNS},
        "inputs": {dataset: input_record(inputs[dataset]) for dataset in INPUTS},
    }


def preprocessing_document(
    outputs: dict[str, dict], screens: dict[str, dict], prepared: dict
) -> dict[str, Any]:
    # The budget sweep's figure checks each run's region, so these runs keep their placement.
    return {
        "input": input_record(prepared),
        "records": {name: run_record(outputs[name], placement=True) for name in PREPROCESSING_RUNS},
        "screens": {name: run_record(screens[name], placement=True) for name in SCREEN_RUNS},
    }


def main() -> None:
    from experiments.runtime.locations import experiment_bucket
    from experiments.runtime.object_io import get_json

    bucket = experiment_bucket()

    def fetch(keys: dict[str, str]) -> dict[str, dict]:
        return {name: get_json(bucket.uri(key)) for name, key in keys.items()}

    inputs = fetch(INPUTS)
    path = write_json(
        "rna_workflow_runs.json",
        workflow_document(fetch(WORKFLOW_RUNS), inputs),
        directory=RECORDS,
    )
    print(f"wrote {path} ({len(WORKFLOW_RUNS)} runs)", flush=True)
    path = write_json(
        "memory_preprocessing.json",
        preprocessing_document(
            fetch(PREPROCESSING_RUNS), fetch(SCREEN_RUNS), inputs[PREPROCESSING_INPUT]
        ),
        directory=RECORDS,
    )
    print(f"wrote {path} ({len(PREPROCESSING_RUNS)} runs)", flush=True)


if __name__ == "__main__":
    main()
