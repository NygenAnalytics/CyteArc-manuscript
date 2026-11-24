"""Analysis time of a tool-comparison run, as Methods defines it, without dependencies, for the"""

PREPARATION_STEPS = frozenset(
    {"convertParquet", "uncompressInput", "convertMatrix", "transposeMatrix"}
)


def _through_markers(steps: dict) -> dict | None:
    marker = steps.get("findMarkers", {})
    if "start" not in marker or "end" not in marker:
        return None
    end = float(marker["end"])
    selected = {
        name: values
        for name, values in steps.items()
        if name != "keepOutputs"
        and (name == "findMarkers" or "start" not in values or float(values["start"]) < end)
    }
    if any("start" not in values or "end" not in values for values in selected.values()):
        return None
    if any(float(values["end"]) > end for values in selected.values()):
        raise ValueError("A computational step extends beyond the marker-search boundary")
    return selected


def workflow_time_seconds(steps: dict) -> float | None:
    selected = _through_markers(steps)
    if selected is None:
        return None
    return float(selected["findMarkers"]["end"]) - min(
        float(values["start"]) for values in selected.values()
    )


def analysis_time_seconds(steps: dict) -> float | None:
    """Analysis time of a run from its step records; None for a record without the start and end
    of every step (the first harness's smoke runs)."""
    selected = _through_markers(steps)
    if selected is None:
        return None
    analysis = [v for k, v in selected.items() if k not in PREPARATION_STEPS]
    return float(selected["findMarkers"]["end"]) - min(float(v["start"]) for v in analysis)


def preparation_seconds(steps: dict) -> float | None:
    workflow = workflow_time_seconds(steps)
    analysis = analysis_time_seconds(steps)
    return None if workflow is None or analysis is None else workflow - analysis
