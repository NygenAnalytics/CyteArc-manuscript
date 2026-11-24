"""Read measurement bundles grouped by their scientific purpose."""

import fnmatch
import json
from pathlib import Path
from typing import Any

from experiments.runtime.common import ROOT, load_json, scrub_private

BENCHMARKS = ROOT / "experiments/benchmarks/records"
MAPPING = ROOT / "experiments/mapping/records"
METHODS = ROOT / "experiments/methods/records"
RAW = BENCHMARKS / "raw"
BENCHMARK_TAG = "embryo_rna"
STAGES = (
    "createStore",
    "writeCountsT",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "runUmap",
    "runLeiden",
    "findMarkers",
)
CONVERSION = ("createStore", "writeCountsT", "initializeStore")
# The HNSW steps whose seconds depend on the index's links and depths: building the index and querying it.
NEIGHBOR_INDEX_STEPS = ("buildAnnIndex", "queryNeighbors")
# The stages of a published analysis after its store was written and reopened, from cell filtering through the
# marker search.
ANALYSIS_STAGES = (
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "runUmap",
    "runLeiden",
    "findMarkers",
)
RUN_GROUPS = {"rna_scaling_runs": "rna_scaling", "rna_resource_runs": "rna_resources"}
RESULT_BUNDLES = (
    MAPPING / "artifact_reuse_runs.json",
    METHODS / "fate_probability_comparisons.json",
)
ARRAY_FOLDERS = {
    "fate_probabilities": METHODS / "raw" / "fate_probabilities",
}


def _dump(path: Path, payload: Any) -> None:
    # Every bundle is written without private locations or Modal identifiers.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(scrub_private(payload), indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def load_runs(group: str, raw: Path = RAW) -> dict[str, dict]:
    """Every run of one group (``benchmark_runs`` or ``sweep_runs``), by tag."""
    path = raw / f"{group}.json"
    return load_json(path)


def save_runs(group: str, runs: dict[str, dict], raw: Path = RAW) -> None:
    _dump(raw / f"{group}.json", runs)


def elapsed_seconds(funnel: dict) -> float:
    """Elapsed workflow time excluding the recorded input download."""
    return float(funnel["wholeFunctionSeconds"]) - float(funnel["datasetDownloadSeconds"])


def conversion_elapsed_seconds(funnel: dict) -> float:
    """The part of elapsed_seconds that converts the source file into a store: the whole-function
    times of the conversion stages, input setup included."""
    outcomes = {outcome["stage"]: outcome for outcome in funnel["outcomes"]}
    return sum(float(outcomes[stage]["wholeFunctionSeconds"]) for stage in CONVERSION)


def profiler_seconds(funnel: dict) -> float:
    """The profiler's own work inside elapsed_seconds: its check of the gene-major copy after
    writeCountsT (validationPersistenceSeconds) and the time outside every stage function, most
    of it writing each stage's record. Stages that ran at the same time (concurrentStages)
    occupy the time of the longest of them, since they start together."""
    whole = {o["stage"]: float(o["wholeFunctionSeconds"]) for o in funnel["outcomes"]}
    groups: list[set[str]] = []
    for outcome in funnel["outcomes"]:
        group = {outcome["stage"], *(outcome.get("concurrentStages") or [])}
        for joined in [g for g in groups if g & group]:
            group |= joined
            groups.remove(joined)
        groups.append(group)
    occupied = sum(max(whole[stage] for stage in group) for group in groups)
    checks = sum(float(o.get("validationPersistenceSeconds") or 0.0) for o in funnel["outcomes"])
    return elapsed_seconds(funnel) - occupied + checks


def run_funnel(tag: str, raw: Path = RAW) -> dict | None:
    """The funnel record of the run with this tag, from whichever group of runs holds it."""
    for group in RUN_GROUPS:
        funnel = (load_runs(group, raw).get(tag) or {}).get("funnel")
        if funnel is not None:
            return funnel
    return None


def modal_results(pattern: str = "*") -> list[tuple[str, dict]]:
    """Return matching result names and measurements from the scientific bundles."""
    found = {}
    for path in RESULT_BUNDLES:
        for name, entry in load_json(path)["results"].items():
            if fnmatch.fnmatchcase(name, pattern):
                if name in found:
                    raise ValueError(f"Duplicate measurement result: {name}")
                found[name] = entry["result"]
    return sorted(found.items())


def modal_result(name: str) -> dict | None:
    found = modal_results(name)
    return found[0][1] if found else None


def modal_array(name: str, file: str) -> Path:
    """Locate arrays for fate-probability measurements."""
    if name == "pancreas_larry_fate_prediction":
        folder = "fate_probabilities"
    else:
        raise ValueError(f"No array family for measurement: {name}")
    return ARRAY_FOLDERS[folder] / file
