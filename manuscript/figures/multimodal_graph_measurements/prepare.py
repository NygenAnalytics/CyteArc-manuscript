"""Prepare the panel data of the supplementary figure on the time and memory of joint RNA and protein graphs: the COMBAT
and Stephenson CITE-seq series of 100,000 to 1,457,527 cells, the Hao CITE-seq atlas in three replicates, and the
TEA-seq and 10x PBMCs.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py. Each function reads
the experiment records, checks the totals the records state against the stages they list, and raises on a mismatch.
Run from the repository root:

  uv run --locked python -B -m manuscript.figures.multimodal_graph_measurements.prepare
"""

import json

import numpy as np
from experiments.runtime.common import ROOT, load_json

NEIGHBORS = ROOT / "experiments/integration/records/multimodal_neighbors"
CITE = NEIGHBORS / "combat_stephenson_pbmc_citeseq"
CITE_SIZES = (100000, 300000, 600000, 1000000, 1457527)
HAO = NEIGHBORS / "hao_pbmc_citeseq"
HAO_SIZES = (10000, 50000, 100000, 161764)
REPLICATES = (1, 2, 3)
SMALL = {
    "teaseq": (NEIGHBORS / "swanson_7K_pbmc_teaseq", 6194, ["RNA", "ATAC", "ADT"]),
    "tenx": (NEIGHBORS / "tenx_8K_pbmc_citeseq", 7299, ["RNA", "ADT"]),
}
OUT = ROOT / "manuscript" / "figures" / "multimodal_graph_measurements"
GIB = 1024**3


def relative(path) -> str:
    return str(path.relative_to(ROOT))


def check_close(value: float, expected: float, what: str) -> None:
    if not np.isclose(value, expected, rtol=1e-9, atol=1e-6):
        raise ValueError(f"{what}: the record states {value}, its stages sum to {expected}")


def completed(path) -> dict:
    record = load_json(path)
    if record.get("status") != "completed":
        raise ValueError(f"{relative(path)}: the job did not complete")
    return record["result"]


# ---- CyteArc's runs -----------------------------------------------------------------------------

JOINT = {"wnn": "Wnn", "snn": "Snn"}
STAGE_LISTS = ("stages", "reuseStages", "freshRevisionStages")


def cytearc_run(path, joint: str, cells: int) -> dict:
    """One CyteArc run that built the joint graph named by joint first: its first analysis, its reclustering at a new
    resolution from stored artifacts and from scratch, and its peaks, each total checked against the stages."""
    result = completed(path)
    if result["status"] != "ok" or result["cells"] != cells:
        raise ValueError(f"{relative(path)}: not a completed run on {cells} cells")
    if (
        result.get("jointMethod") not in (joint, None)
        or f"revised{JOINT[joint]}Seconds" not in result
    ):
        raise ValueError(f"{relative(path)}: not a {joint.upper()}-first run")
    other = "snn" if joint == "wnn" else "wnn"
    initial = result[f"initial{JOINT[joint]}Seconds"]
    check_close(
        initial,
        sum(s["wallSeconds"] for s in result["stages"] if not s["stage"].startswith(f"{other}.")),
        f"{relative(path)} first analysis",
    )
    stored = result[f"revised{JOINT[joint]}Seconds"]
    check_close(
        stored,
        sum(s["wallSeconds"] for s in result["reuseStages"]),
        f"{relative(path)} reclustering from stored artifacts",
    )
    run = {
        "initialSeconds": initial,
        "storedSeconds": stored,
        "firstAnalysisPeakGiB": max(s["operationPeakBytes"] for s in result["stages"]) / GIB,
    }
    if "freshRevisionStages" in result:
        fresh = result[f"freshRevised{JOINT[joint]}Seconds"]
        check_close(
            fresh,
            sum(s["wallSeconds"] for s in result["freshRevisionStages"]),
            f"{relative(path)} reclustering from scratch",
        )
        validation = result["reuseValidation"]
        if not validation["clusterLabelsIdentical"] or validation["clusterAri"] != 1.0:
            raise ValueError(
                f"{relative(path)}: reclustering from stored artifacts changed the labels"
            )
        run |= {
            "scratchSeconds": fresh,
            "peakGiB": max(s["operationPeakBytes"] for key in STAGE_LISTS for s in result[key])
            / GIB,
            "containerGiB": {s["memoryMaxBytes"] for key in STAGE_LISTS for s in result[key]},
        }
        if len(run["containerGiB"]) != 1:
            raise ValueError(f"{relative(path)}: the stages report different container sizes")
        run["containerGiB"] = run["containerGiB"].pop() / GIB
    if "compilationWarmupWallSeconds" in result:
        run["warmupSeconds"] = result["compilationWarmupWallSeconds"]
    return run


# ---- the COMBAT and Stephenson CITE-seq series ---------------------------------------------------

OUTCOMES = {"time_limit": "time", "memory_limit": "oom"}


def seurat_cite(cells: int) -> dict:
    """Seurat's run on the COMBAT and Stephenson cells: each phase's time from its stage windows, how the first
    analysis and the whole run ended, the run's peak and its limits."""
    path = CITE / f"cells_{cells}/seurat_1.json"
    result = completed(path)
    if result["cells"] != cells:
        raise ValueError(f"{relative(path)}: not a run on {cells} cells")
    windows = result["stageWindows"]
    finished = [w for w in windows if not w.get("unfinished")]
    if any(w["wallSeconds"] is None for w in finished) or len(windows) - len(finished) > 1:
        raise ValueError(f"{relative(path)}: a finished stage window lacks its time")
    phases = {"input": [], "initial": [], "export": [], "stored": [], "scratch": []}
    names = {
        "input_setup": "input",
        "initial": "initial",
        "export_graphs": "export",
        "reuse_revision": "stored",
        "fresh_revision": "scratch",
    }
    for w in windows:
        phases[names[w["stage"].split(":")[0]]].append(w)
    process = result["stages"][0]
    if len(result["stages"]) != 1 or process["stage"] != "seurat.workflow.process":
        raise ValueError(f"{relative(path)}: expected one process stage")

    def total(phase: str) -> float | None:
        ws = phases[phase]
        if not ws or any(w.get("unfinished") for w in ws):
            return None
        return sum(w["wallSeconds"] for w in ws)

    if result["status"] == "ok":
        outcome = "pass"
        if result["outcome"] != "completed" or len(finished) != len(windows):
            raise ValueError(f"{relative(path)}: a completed run with an unfinished window")
    else:
        outcome = OUTCOMES[result["outcome"]]
        if outcome == "time" and process["wallSeconds"] < result["timeLimitSeconds"]:
            raise ValueError(f"{relative(path)}: stopped for time before its time limit")
        if (
            outcome == "oom"
            and process["operationPeakBytes"] < 0.99 * result["memoryLimitGiB"] * GIB
        ):
            raise ValueError(f"{relative(path)}: stopped for memory below its memory stop")
    if process["wallSeconds"] < sum(w["wallSeconds"] for w in finished):
        raise ValueError(f"{relative(path)}: the process ended before its windows did")
    initial = total("initial")
    return {
        "outcome": outcome,
        "firstAnalysisOutcome": "pass" if initial is not None else outcome,
        "lastStage": result.get("lastStage"),
        "inputSeconds": total("input"),
        "initialSeconds": initial,
        "storedSeconds": total("stored"),
        "scratchSeconds": total("scratch"),
        "phases": [
            {
                "phase": names[w["stage"].split(":")[0]],
                "stage": w["stage"],
                "seconds": w["wallSeconds"],
                "unfinished": bool(w.get("unfinished")),
                "sampledPeakGiB": (w.get("sampledPeakBytes") or 0) / GIB,
            }
            for w in windows
        ],
        "processSeconds": process["wallSeconds"],
        "peakGiB": process["operationPeakBytes"] / GIB,
        "memoryStopGiB": result["memoryLimitGiB"],
        "containerGiB": result["containerMemoryLimitBytes"] / GIB,
        "timeLimitSeconds": result["timeLimitSeconds"],
    }


def cite_runs() -> list[dict]:
    return [
        {
            "cells": n,
            "cytearc_wnn": cytearc_run(CITE / f"cells_{n}/cytearc_wnn_1.json", "wnn", n),
            "cytearc_snn": cytearc_run(CITE / f"cells_{n}/cytearc_snn_1.json", "snn", n),
            "seurat_wnn": seurat_cite(n),
        }
        for n in CITE_SIZES
    ]


def cite_source(tags=("cytearc_wnn_1", "cytearc_snn_1", "seurat_1")) -> list[str]:
    return [relative(CITE / f"cells_{n}/{tag}.json") for n in CITE_SIZES for tag in tags]


def cite_first_analysis_time() -> dict:
    """At each size, the first analysis of CyteArc's WNN-first and SNN-first workflows after their compilation
    warm-up, and Seurat's first analysis where it finished."""
    sizes = cite_runs()
    for s in sizes:
        for tool in ("cytearc_wnn", "cytearc_snn"):
            s[tool] = {k: s[tool][k] for k in ("initialSeconds", "warmupSeconds")}
        seurat = s["seurat_wnn"]
        s["seurat_wnn"] = {k: seurat[k] for k in ("firstAnalysisOutcome", "initialSeconds")}
    return {"sizes": sizes, "source": cite_source()}


def cite_peak_memory() -> dict:
    """At each size, the peak of each run over its first analysis and both reclusterings, how each run ended, and
    the container and memory stop each tool ran under."""
    sizes = cite_runs()
    containers = {s[t]["containerGiB"] for s in sizes for t in ("cytearc_wnn", "cytearc_snn")}
    stops = {s["seurat_wnn"]["memoryStopGiB"] for s in sizes}
    if len(containers) != 1 or len(stops) != 1:
        raise ValueError("the series ran under different limits at different sizes")
    for s in sizes:
        for tool in ("cytearc_wnn", "cytearc_snn"):
            s[tool] = {k: s[tool][k] for k in ("peakGiB", "firstAnalysisPeakGiB")}
        seurat = s["seurat_wnn"]
        s["seurat_wnn"] = {k: seurat[k] for k in ("peakGiB", "outcome", "lastStage")}
    return {
        "sizes": sizes,
        "cytearcContainerGiB": containers.pop(),
        "seuratMemoryStopGiB": stops.pop(),
        "source": cite_source(),
    }


def cite_reclustering() -> dict:
    """At each size, reclustering at resolution 1.2 from stored artifacts and from scratch: CyteArc's two workflows,
    and Seurat from its stored WNN graph and from scratch where it finished."""
    sizes = cite_runs()
    for s in sizes:
        for tool in ("cytearc_wnn", "cytearc_snn"):
            s[tool] = {k: s[tool][k] for k in ("storedSeconds", "scratchSeconds")}
        seurat = s["seurat_wnn"]
        s["seurat_wnn"] = {k: seurat[k] for k in ("storedSeconds", "scratchSeconds", "outcome")}
    return {"sizes": sizes, "source": cite_source()}


def cite_run_timelines() -> dict:
    """Each Seurat run as its phases in order, with where it stopped and its time limit, and each CyteArc WNN-first
    run's whole length: its warm-up, first analysis with both joint branches, and both reclusterings."""
    sizes = []
    for n in CITE_SIZES:
        seurat = seurat_cite(n)
        result = completed(CITE / f"cells_{n}/cytearc_wnn_1.json")
        cytearc = result["compilationWarmupWallSeconds"] + sum(
            s["wallSeconds"] for key in STAGE_LISTS for s in result[key]
        )
        sizes.append(
            {
                "cells": n,
                "seurat_wnn": {
                    k: seurat[k]
                    for k in (
                        "outcome",
                        "phases",
                        "processSeconds",
                        "timeLimitSeconds",
                        "lastStage",
                    )
                },
                "cytearcSeconds": cytearc,
            }
        )
    return {"sizes": sizes, "source": cite_source(("cytearc_wnn_1", "seurat_1"))}


# Each step of the first analysis as both tools record it: CyteArc's stages and Seurat's stage windows. CyteArc's
# neighbor step builds the index, queries it and computes the connectivities; its ADT normalization includes choosing
# the ADT features, which takes under a tenth of a second.
STAGE_STEPS = (
    ("RNA normalization", ("RNA.normalization",), ("initial:RNA.normalization",)),
    ("RNA variable genes", ("RNA.features",), ("initial:RNA.features",)),
    ("RNA scaling", (), ("initial:RNA.scaling",)),
    ("RNA PCA", ("RNA.reduction",), ("initial:RNA.reduction",)),
    (
        "RNA neighbors",
        ("RNA.index", "RNA.neighbors", "RNA.connectivity"),
        ("initial:RNA.neighbors",),
    ),
    ("RNA clustering", ("RNA.clustering",), ("initial:RNA.clustering",)),
    ("ADT normalization", ("ADT.features", "ADT.normalization"), ("initial:ADT.normalization",)),
    ("ADT scaling", (), ("initial:ADT.scaling",)),
    ("ADT PCA", ("ADT.reduction",), ("initial:ADT.reduction",)),
    (
        "ADT neighbors",
        ("ADT.index", "ADT.neighbors", "ADT.connectivity"),
        ("initial:ADT.neighbors",),
    ),
    ("ADT clustering", ("ADT.clustering",), ("initial:ADT.clustering",)),
    ("WNN graph", ("wnn.integration",), ("initial:WNN.integration",)),
    ("WNN clustering", ("wnn.clustering",), ("initial:WNN.clustering",)),
)
STAGE_CELLS = 600000


def cite_stage_times() -> dict:
    """At 600,000 cells, the largest size at which Seurat finished its first analysis: each step's time and peak in
    CyteArc's WNN-first run and in Seurat's run, and Seurat's input loading, which its first analysis excludes."""
    path = CITE / f"cells_{STAGE_CELLS}/cytearc_wnn_1.json"
    result = completed(path)
    stages = {s["stage"]: s for s in result["stages"] if not s["stage"].startswith("snn.")}
    seurat = seurat_cite(STAGE_CELLS)
    windows = {p["stage"]: p for p in seurat["phases"] if p["phase"] == "initial"}
    used_cytearc = [name for _, names, _ in STAGE_STEPS for name in names]
    used_seurat = [name for _, _, names in STAGE_STEPS for name in names]
    if sorted(used_cytearc) != sorted(stages) or sorted(used_seurat) != sorted(windows):
        raise ValueError("the steps do not cover each first-analysis stage exactly once")
    steps = []
    for step, cytearc_names, seurat_names in STAGE_STEPS:
        steps.append(
            {
                "step": step,
                "cytearcSeconds": sum(stages[n]["wallSeconds"] for n in cytearc_names)
                if cytearc_names
                else None,
                "cytearcPeakGiB": max(stages[n]["operationPeakBytes"] for n in cytearc_names) / GIB
                if cytearc_names
                else None,
                "seuratSeconds": sum(windows[n]["seconds"] for n in seurat_names),
                "seuratSampledPeakGiB": max(windows[n]["sampledPeakGiB"] for n in seurat_names),
            }
        )
    cytearc_total = sum(s["cytearcSeconds"] or 0 for s in steps)
    check_close(result["initialWnnSeconds"], cytearc_total, f"{relative(path)} first analysis")
    check_close(
        seurat["initialSeconds"], sum(s["seuratSeconds"] for s in steps), "Seurat first analysis"
    )
    return {
        "cells": STAGE_CELLS,
        "steps": steps,
        "cytearcSeconds": cytearc_total,
        "seuratSeconds": seurat["initialSeconds"],
        "seuratInputSeconds": seurat["inputSeconds"],
        "source": [relative(path), relative(CITE / f"cells_{STAGE_CELLS}/seurat_1.json")],
    }


# ---- the Hao CITE-seq atlas, three replicates at each size ------------------------------------------


def first_use_compiles(path) -> bool:
    """Whether the record says its reclustering ran after the first analysis had compiled the kernels."""
    revision = completed(path)["timingInterpretation"]["revision"]
    return "after initial analysis has warmed compiled kernels" in revision


def seurat_hao(path, cells: int) -> dict:
    result = completed(path)
    native = result["nativeWorkflow"]
    if result["status"] != "ok" or result["cells"] != cells or native["cells"] != cells:
        raise ValueError(f"{relative(path)}: not a completed run on {cells} cells")
    check_close(native["initialSeconds"], sum(native["initialStages"].values()), relative(path))
    check_close(
        native["freshRevisedSeconds"], sum(native["freshRevisionStages"].values()), relative(path)
    )
    if not result["reuseValidation"]["clusterLabelsIdentical"]:
        raise ValueError(f"{relative(path)}: reclustering from the stored graph changed the labels")
    process = result["stages"][0]
    if len(result["stages"]) != 1 or process["stage"] != "seurat.workflow.process":
        raise ValueError(f"{relative(path)}: expected one process stage")
    return {
        "inputSeconds": native["inputSetupSeconds"],
        "initialSeconds": native["initialSeconds"],
        "storedSeconds": native["reusedRevisionSeconds"],
        "scratchSeconds": native["freshRevisedSeconds"],
        "peakGiB": process["operationPeakBytes"] / GIB,
    }


def hao_runs() -> list[dict]:
    sizes = []
    for n in HAO_SIZES:
        folder = HAO / f"cells_{n}"
        size = {"cells": n}
        for tool, joint in (("cytearc_wnn", "wnn"), ("cytearc_snn", "snn")):
            runs = []
            for r in REPLICATES:
                path = folder / f"{tool}_{r}.json"
                runs.append(cytearc_run(path, joint, n))
                # No warm-up ran before these runs, so their first analysis compiles the kernels it uses.
                if "warmupSeconds" in runs[-1] or not first_use_compiles(path):
                    raise ValueError(
                        f"{relative(path)}: the first analysis no longer includes compilation"
                    )
            size[tool] = runs
        size["seurat_wnn"] = [seurat_hao(folder / f"seurat_{r}.json", n) for r in REPLICATES]
        sizes.append(size)
    return sizes


def hao_source() -> list[str]:
    return [
        relative(HAO / f"cells_{n}/{tool}_{r}.json")
        for n in HAO_SIZES
        for tool in ("cytearc_wnn", "cytearc_snn", "seurat")
        for r in REPLICATES
    ]


def replicate_summary(sizes: list[dict], fields: dict[str, tuple[str, ...]]) -> list[dict]:
    """Each size's replicate values of the named fields, with their mean and range."""
    out = []
    for s in sizes:
        row = {"cells": s["cells"]}
        for tool, keys in fields.items():
            row[tool] = {}
            for key in keys:
                values = [run[key] for run in s[tool]]
                if len(values) != len(REPLICATES) or not np.all(np.isfinite(values)):
                    raise ValueError(f"{tool} {key} at {s['cells']} cells lacks a replicate")
                row[tool][key] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                    "low": min(values),
                    "high": max(values),
                }
        out.append(row)
    return out


def hao_first_analysis_time() -> dict:
    """At each size of the Hao atlas, the first analysis of each tool in three replicates, including CyteArc's
    compilation on first use, and Seurat's input loading, which its first analysis excludes."""
    fields = {
        "cytearc_wnn": ("initialSeconds",),
        "cytearc_snn": ("initialSeconds",),
        "seurat_wnn": ("initialSeconds", "inputSeconds"),
    }
    return {"sizes": replicate_summary(hao_runs(), fields), "source": hao_source()}


def hao_peak_memory() -> dict:
    """At each size of the Hao atlas, each run's peak over its first analysis and both reclusterings."""
    fields = {tool: ("peakGiB",) for tool in ("cytearc_wnn", "cytearc_snn", "seurat_wnn")}
    return {"sizes": replicate_summary(hao_runs(), fields), "source": hao_source()}


def hao_reclustering() -> dict:
    """At each size of the Hao atlas, reclustering at resolution 1.2 from stored results and from scratch."""
    fields = {
        tool: ("storedSeconds", "scratchSeconds")
        for tool in ("cytearc_wnn", "cytearc_snn", "seurat_wnn")
    }
    return {"sizes": replicate_summary(hao_runs(), fields), "source": hao_source()}


# ---- the TEA-seq and 10x PBMCs ---------------------------------------------------------------------


def small_dataset_times() -> dict:
    """On the TEA-seq and 10x PBMCs, each workflow's first analysis, which compiles CyteArc's kernels, and its
    reclustering at resolution 1.2 from stored artifacts and, for the SNN-first workflow, from scratch."""
    datasets = []
    source = []
    for key, (folder, cells, modalities) in SMALL.items():
        workflows = {}
        for tool, name, joint in (
            ("cytearc_wnn", "workflow.json", "wnn"),
            ("cytearc_snn", "snn_fusion.json", "snn"),
        ):
            path = folder / name
            result = completed(path)
            if result["modalities"] != modalities:
                raise ValueError(f"{relative(path)}: expected the modalities {modalities}")
            run = cytearc_run(path, joint, cells)
            # No warm-up ran before these runs, so their first analysis compiles the kernels it uses.
            if "warmupSeconds" in run:
                raise ValueError(
                    f"{relative(path)}: the first analysis no longer includes compilation"
                )
            workflows[tool] = {
                k: run[k] for k in ("initialSeconds", "storedSeconds", "scratchSeconds") if k in run
            }
            source.append(relative(path))
        if not first_use_compiles(folder / "snn_fusion.json"):
            raise ValueError(f"{key}: reclustering no longer runs after the kernels are compiled")
        datasets.append(
            {"dataset": key, "cells": cells, "modalities": modalities, "workflows": workflows}
        )
    return {"datasets": datasets, "source": source}


def prepared_input_times() -> dict:
    """On the TEA-seq and 10x PBMCs, in three replicates, CyteArc's WNN graph and its clustering from the prepared
    reductions and stored modality neighbors, on the first call, which compiles the kernels, and warm; and Seurat's
    weighted nearest neighbors and clustering from the same reductions with 50 and 200 candidate neighbors."""
    datasets = []
    source = []
    for key, (folder, cells, modalities) in SMALL.items():
        cytearc = {
            "first_call": {"wnn": [], "clustering": []},
            "warm": {"wnn": [], "clustering": []},
        }
        seurat = {"50": {"wnn": [], "clustering": []}, "200": {"wnn": [], "clustering": []}}
        for r in REPLICATES:
            path = folder / f"prepared_{r}.json"
            result = completed(path)
            if result["cells"] != cells or result["modalities"] != modalities:
                raise ValueError(f"{relative(path)}: not a run on the {key} cells")
            for call, check in result["validation"].items():
                if not all(check.values()):
                    raise ValueError(
                        f"{relative(path)}: the {call} output differs from the workflow's"
                    )
            stages = {s["stage"]: s["wallSeconds"] for s in result["stages"]}
            for call in cytearc:
                for step in ("wnn", "clustering"):
                    cytearc[call][step].append(stages[f"{call}.{step}"])
            path_r = folder / f"seurat_{r}.json"
            runs = completed(path_r)["seurat"]
            for breadth in seurat:
                if runs[breadth]["knnRange"] != int(breadth) or runs[breadth]["cells"] != cells:
                    raise ValueError(f"{relative(path_r)}: unexpected Seurat settings at {breadth}")
                seurat[breadth]["wnn"].append(runs[breadth]["neighborSeconds"])
                seurat[breadth]["clustering"].append(runs[breadth]["clusteringSeconds"])
            source += [relative(path), relative(path_r)]
        datasets.append({"dataset": key, "cells": cells, "cytearc": cytearc, "seurat": seurat})
    return {"datasets": datasets, "source": source}


PANELS = (
    cite_first_analysis_time,
    cite_peak_memory,
    cite_reclustering,
    cite_run_timelines,
    cite_stage_times,
    hao_first_analysis_time,
    hao_peak_memory,
    hao_reclustering,
    small_dataset_times,
    prepared_input_times,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        data = panel()
        path = OUT / f"{panel.__name__}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {relative(path)}")


if __name__ == "__main__":
    main()
