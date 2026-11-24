"""The neighbor-count fork of a stored ten-million-cell embryo analysis, from records/artifact_reuse_runs.json: the
minutes of its recomputed stages and of its lookups of stored artifacts, the stored analysis's minutes, and the share
of them the fork saves with and without the lookups, into .build/derived/mapping/neighbor_count_fork.json.

    uv run --locked python -m experiments.mapping.neighbor_count_fork_summary
"""

from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

RUNS = ROOT / "experiments" / "mapping" / "records" / "artifact_reuse_runs.json"
SUMMARY = DERIVED / "mapping" / "neighbor_count_fork.json"
# A fork stage's name starts with the first prefix when the fork recomputed it and the second when it looked up a
# stored artifact.
RECOMPUTED, LOOKED_UP = "fork.", "reuse."
# The recomputed stages may add up to the recorded fork time to within this many seconds.
FORK_SUM_TOLERANCE_SECONDS = 1.0


def fork_summary(result: dict) -> dict:
    """Minutes of the fork's recomputed stages, of its lookups and of both, against the stored analysis's minutes, and
    the share of the stored analysis time each saves. A fork whose recomputed stages do not add up to its recorded fork
    time, or that looked nothing up, is an error."""
    stages = result["stages"]
    recomputed = sum(s["seconds"] for name, s in stages.items() if name.startswith(RECOMPUTED))
    lookups = [s["seconds"] for name, s in stages.items() if name.startswith(LOOKED_UP)]
    if abs(recomputed - result["forkSeconds"]) > FORK_SUM_TOLERANCE_SECONDS:
        raise ValueError(
            f"the recomputed stages add up to {recomputed:.1f} seconds, not the fork's {result['forkSeconds']:.1f}"
        )
    if not lookups:
        raise ValueError("the fork looked up no stored artifact")
    stored = result["fullRunAnalysisSeconds"]
    with_lookups = recomputed + sum(lookups)
    return {
        "kOriginal": result["kOriginal"],
        "kNew": result["kNew"],
        "storedAnalysisMinutes": stored / 60,
        "forkMinutes": recomputed / 60,
        "lookupMinutes": sum(lookups) / 60,
        "forkWithLookupsMinutes": with_lookups / 60,
        "forkSaving": 1 - recomputed / stored,
        "forkWithLookupsSaving": 1 - with_lookups / stored,
    }


def main() -> None:
    results = load_json(RUNS)["results"]
    if len(results) != 1:
        raise ValueError(f"expected one fork in {RUNS.name}, found {sorted(results)}")
    (run,) = results.values()
    write_json_at(SUMMARY, {"source": str(RUNS.relative_to(ROOT)), **fork_summary(run["result"])})


if __name__ == "__main__":
    main()
