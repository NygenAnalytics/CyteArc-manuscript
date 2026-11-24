import numpy as np
from experiments.runtime.common import DERIVED, ROOT, load_json
from experiments.runtime.records import modal_result
from manuscript.tables.table_values import write_metrics

BENCHMARKS = ROOT / "experiments/benchmarks/records"
BENCHMARKS_DERIVED = DERIVED / "benchmarks"
INTEGRATION = ROOT / "experiments/integration/records"
INTEGRATION_DERIVED = DERIVED / "integration"
MAPPING = ROOT / "experiments/mapping/records"
METHODS = ROOT / "experiments/methods/records"
RAW = BENCHMARKS / "raw"


def _sample(dataset: str, tool: str, tier: int):
    import pandas as pd

    path = RAW / f"marker_statistics_{dataset}_{tool}_cpus_8_replicate_1_memory_{tier}gib.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["group"] = frame["group"].astype(str)
    frame["gene"] = frame["gene"].astype(str)
    return frame


def marker_agreement() -> dict:
    """CyteArc's sampled marker statistics against each other tool, on the first replicate."""
    tiers = {
        "embryo_1000000": {"cytearc": 32, "bpcells": 32, "presto": 64, "scanpy_csc": 64},
        "brain_1306127": {"cytearc": 32, "bpcells": 32, "scanpy_csc": 64},
        "tabula_sapiens": {"cytearc": 32, "bpcells": 32, "scanpy_csc": 128},
    }
    out = {}
    for dataset, tools in tiers.items():
        cyte = _sample(dataset, "cytearc", tools["cytearc"])
        entry = {"pairs": int(len(cyte))}
        for tool in ("presto", "scanpy_csc"):
            other = _sample(dataset, tool, tools[tool]) if tool in tools else None
            if other is None:
                continue
            merged = cyte.merge(other, on=["group", "gene"], suffixes=("", "_o"))
            entry[f"aurocMaxAbsDiff_{tool}"] = float((merged["auc"] - merged["auc_o"]).abs().max())
            entry[f"aurocPairs_{tool}"] = int(len(merged))
        other = _sample(dataset, "bpcells", tools["bpcells"])
        merged = cyte.merge(other, on=["group", "gene"], suffixes=("", "_o"))
        p, q = merged["pval"].to_numpy(float), merged["pval_o"].to_numpy(float)
        both = (p >= 1e-300) & (q >= 1e-300)
        entry["pMaxAbsLog10Diff_bpcells"] = float(
            np.abs(np.log10(p[both]) - np.log10(q[both])).max()
        )
        entry["pPairsAboveFloor_bpcells"] = int(both.sum())
        entry["pPairsOneBelowFloor_bpcells"] = int(((p < 1e-300) != (q < 1e-300)).sum())
        out[dataset] = entry
    summary = load_json(BENCHMARKS_DERIVED / "toolkit_comparisons.json")["markersCommon"][
        "sameTest"
    ]
    out["recordLargestAucDiff"] = {
        s["dataset"]: s["largestAucDiff"] for s in summary if s["dataset"] in tiers
    }
    out["auroc"] = max(
        v for d in tiers for k, v in out[d].items() if k.startswith("aurocMaxAbsDiff")
    )
    out["log10P"] = max(out[d]["pMaxAbsLog10Diff_bpcells"] for d in tiers)
    return out


def harmony_summary() -> dict:
    """Harmony2 in CyteArc against harmonypy and R harmony on the same PCA coordinates: single steps on
    shared states, each full-fit arm judged by the records' equivalence rule, and the ten-seed comparison
    with harmonypy on the full selections."""
    harmonypy = load_json(INTEGRATION / "harmony2.json")
    r_harmony = load_json(INTEGRATION / "harmony2_r_reference.json")
    comparisons = load_json(INTEGRATION / "harmony_seed_stability.json")["comparisons"]
    if len(comparisons) != 1:
        raise ValueError("Harmony seed stability holds one ten-seed comparison with harmonypy")
    stability = comparisons[0]["measurement"]
    parity = {"harmonypy": harmonypy["datasets"], "rHarmony": r_harmony["datasets"]}
    if set(parity["harmonypy"]) != set(parity["rHarmony"]):
        raise ValueError("The harmonypy and R harmony parity records cover different datasets")

    def assignment(theta: float) -> dict:
        return {
            "harmonypy": max(
                case["nativeReferenceMaxAbsoluteError"]
                for data in parity["harmonypy"].values()
                for case in data["sharedStateChecks"]["assignmentCases"]
                if case["theta"] == theta
            ),
            "rHarmony": max(
                step["cytearcVersusR_firstCentroids"]
                for data in parity["rHarmony"].values()
                for step in data["assignmentSteps"]
                if step["theta"] == theta
            ),
        }

    checks = [data["sharedStateChecks"] for data in parity["harmonypy"].values()]
    corrections = [check["correctionStep"] for check in checks]
    full_fits = {}
    for reference, datasets in parity.items():
        arms = {}
        for arm in next(iter(datasets.values()))["fullFits"]:
            outcome = {
                name: {
                    key: data["fullFits"][arm]["equivalence"][key]
                    for key in (
                        "passed",
                        "crossOverlapMedian",
                        "referenceOverlapMin",
                        "iLISIMedianDifference",
                        "referenceILISIRange",
                    )
                }
                for name, data in datasets.items()
            }
            arms[arm] = {
                **outcome,
                "datasetsPassed": sum(value["passed"] for value in outcome.values()),
                "datasets": len(outcome),
            }
        full_fits[reference] = arms
    seed_stability = {
        name: {
            "cells": stability[name]["cells"],
            "sameSeedOverlapMedian": stability[name]["cytearcVersusHarmonypySameSeed"][
                "knnOverlapK15"
            ]["median"],
            "harmonypyBetweenSeedsOverlapMedian": stability[name]["harmonypyBetweenSeeds"][
                "knnOverlapK15"
            ]["median"],
            "cytearcBetweenSeedsOverlapMedian": stability[name]["cytearcBetweenSeeds"][
                "knnOverlapK15"
            ]["median"],
        }
        for name in ("binvignat", "kang")
    }
    return {
        "harmonypy": harmonypy["provenance"]["harmonypyVersion"],
        "rHarmony": r_harmony["reference"]["version"],
        "cells": {name: data["cells"] for name, data in parity["harmonypy"].items()},
        "seeds": len(harmonypy["settings"]["seeds"]),
        "fixedRounds": harmonypy["settings"]["arms"]["fixedRoundsPenaltyOn"]["max_iter_harmony"],
        "steps": {
            "assignmentPenaltyOff": assignment(0.0),
            "assignmentPenaltyOn": assignment(2.0),
            "cytearcOwnNumerator": max(
                case["nativeEquationMaxAbsoluteError"]
                for check in checks
                for case in check["assignmentCases"]
            ),
            "harmonypyOwnNumerator": max(
                case["referenceEquationMaxAbsoluteError"]
                for check in checks
                for case in check["assignmentCases"]
            ),
            "rHarmonyOwnNumerator": max(
                step["versusRCodeEquationFromFirstAssignments"]
                for data in parity["rHarmony"].values()
                for step in data["assignmentSteps"]
            ),
            "singleClusterCorrection": max(
                check["singleClusterDynamicRidge"]["coordinateMaxAbsoluteError"] for check in checks
            ),
            "correctionVersusHarmonypy": max(
                step["nativeReferenceMaxAbsoluteError"] for step in corrections
            ),
            "correctionOwnEquationsReproduced": all(step["passed"] for step in corrections),
        },
        "fullFits": full_fits,
        "seedStability": seed_stability,
        "sources": [
            "experiments/integration/records/harmony2.json",
            "experiments/integration/records/harmony2_r_reference.json",
            "experiments/integration/records/harmony_seed_stability.json",
        ],
    }


def agreement_values(markers: dict) -> dict:
    """Every value Table 2 prints, with its source."""
    wnn = load_json(INTEGRATION / "weighted_neighbors_seurat_agreement.json")
    lisi = load_json(INTEGRATION / "lisi_reference_agreement.json")
    cc = load_json(METHODS / "cell_cycle_scanpy_agreement.json")
    aucell = load_json(METHODS / "aucell.json")
    methods = load_json(METHODS / "rna_numerical_reproducibility.json")
    pca = methods["pca"]["measurement"]["result"]
    determinism = methods["determinism"]["measurement"]["result"]
    lsi = load_json(METHODS / "lsi.json")
    gene_scores = load_json(METHODS / "gene_scores.json")
    multimodal = load_json(INTEGRATION_DERIVED / "multimodal_neighbors.json")
    hto = load_json(METHODS / "hto_demultiplexing.json")
    grn_record = load_json(METHODS / "grnboost2.json")
    if grn_record["status"] != "completed" or grn_record["result"]["status"] != "ok":
        raise ValueError("GRNBoost2 measurements are incomplete")
    grn = grn_record["result"]
    droplet_qc = load_json(METHODS / "droplet_qc.json")
    if set(droplet_qc["datasets"]) != set(droplet_qc["eligibleDatasets"]):
        raise ValueError("Droplet QC evaluation is incomplete")
    qc_records = [droplet_qc["datasets"][name] for name in droplet_qc["eligibleDatasets"]]
    if any(row["status"] != "completed" or row["result"]["status"] != "ok" for row in qc_records):
        raise ValueError("Droplet QC measurements are incomplete")
    qc_cases = [row["result"] for row in qc_records]
    pruning_record = load_json(METHODS / "scenic_pruning.json")
    pruning = pruning_record["result"]
    if pruning_record["status"] != "completed" or pruning["status"] != "ok":
        raise ValueError("SCENIC pruning measurements are incomplete")
    pruning_core = [case for case in pruning["coreComparisons"] if case["eligible"]]
    if not pruning_core or not all(case["meetsTolerance"] for case in pruning_core):
        raise ValueError("SCENIC recovery comparisons did not meet the fixed numerical tolerance")
    chromvar_record = load_json(METHODS / "chromvar.json")
    chromvar = chromvar_record["result"]
    if chromvar_record["status"] != "completed" or chromvar["status"] != "ok":
        raise ValueError("chromVAR reference measurements are incomplete")
    hto_results = [record["result"] for record in hto["datasets"].values()]
    if any(
        record["status"] != "completed" or record["result"]["status"] != "ok"
        for record in hto["datasets"].values()
    ):
        raise ValueError("HTO measurements are incomplete")
    for name, record in (
        ("PCA", pca),
        ("determinism", determinism),
        ("LSI", lsi),
        ("GeneScores", gene_scores),
    ):
        if record["status"] != "ok":
            raise ValueError(f"{name} measurement is incomplete")
    multimodal_records = [
        record for dataset in multimodal["datasets"].values() for record in dataset["records"]
    ]
    if any(
        record["status"] != "completed" or record["result"]["status"] != "ok"
        for record in multimodal_records
    ):
        raise ValueError("Multimodal reference measurements are incomplete")
    seurat_versions = {record["result"]["seurat"]["200"]["Seurat"] for record in multimodal_records}
    if len(seurat_versions) != 1:
        raise ValueError("Multimodal comparisons used different Seurat versions")
    lsi_full = lsi["fullSelectionMeasurements"]
    if any(record["status"] != "ok" for record in lsi_full):
        raise ValueError("Full-selection LSI measurements are incomplete")
    lsi_cases = [case for record in lsi_full for case in record["results"]]
    lsi_matched = [
        row
        for row in lsi_cases
        if (row["parameters"]["n_iter"], row["parameters"]["n_oversamples"]) == (80, 100)
    ]
    if not lsi_matched or not all(row["meetsTolerance"] for row in lsi_matched):
        raise ValueError("The matched LSI convergence comparison is incomplete")
    deterministic_pairs = [
        row for row in determinism["comparisons"] if row["mode"] == "deterministic"
    ]
    magic = load_json(METHODS / "pbmc_magic_imputation_agreement.json")
    pba = load_json(METHODS / "pancreas_population_balance_agreement.json")
    symphony_record = load_json(MAPPING / "symphony.json")
    symphony = symphony_record["measurement"]["result"]
    symphony_preparation = symphony_record["referencePreparation"]["result"]
    if symphony_record["measurement"]["status"] != "completed" or not symphony["passed"]:
        raise ValueError("Shared-scaling Symphony correction validation is incomplete")
    dense = load_json(METHODS / "fate_probability_solver_agreement.json")
    fate = modal_result("pancreas_larry_fate_prediction")
    harmony = harmony_summary()
    lisi_max = max(
        comparison["maxAbsDiff"]
        for dataset in lisi["datasets"].values()
        for embedding in dataset["embeddings"].values()
        for neighbours in ("hnswGraph", "exactKnn")
        for name, comparison in embedding.get(neighbours, {}).items()
        if name.startswith("perCell") and name.endswith("CyteArcVersusScib")
    )
    cc_max = max(
        cc["datasets"][d]["deterministicControls"][b][x]["maxAbsDiff"]
        for d in cc["datasets"]
        for b in cc["datasets"][d]["deterministicControls"]
        for x in ("S", "G2M")
    )
    pancreas = fate["pancreas"]["matchedCyteArc"]["agreement"]
    return {
        "markers": markers,
        "pca": {
            "cells": pca["source"]["cells"],
            "dimensions": pca["settings"]["pcaDims"],
            "minimumFeatures": min(row["features"] for row in pca["results"]),
            "maximumFeatures": max(row["features"] for row in pca["results"]),
            "configurations": len(pca["results"]),
            "maxSubspaceAngleDegrees": max(
                row["maxSubspaceAngleDegrees"] for row in pca["results"]
            ),
            "maxRelativeVarianceError": max(
                row["maxRelativeVarianceShareError"] for row in pca["results"]
            ),
            "source": "experiments/methods/records/rna_numerical_reproducibility.json",
        },
        "leiden": {
            "cells": determinism["nativeLeiden"]["cells"],
            "threads": sorted(
                {
                    row["threads"]
                    for row in determinism["nativeLeiden"]["runs"]
                    if row["backend"] == "cytearc"
                }
            ),
            "nativePartitionsIdentical": determinism["nativeLeiden"][
                "allNativePartitionsIdentical"
            ],
            "nativeModularity": determinism["nativeLeiden"]["runs"][0]["modularity"],
            "igraphModularity": next(
                row["modularity"]
                for row in determinism["nativeLeiden"]["runs"]
                if row["backend"] == "igraph"
            ),
            "source": "experiments/methods/records/rna_numerical_reproducibility.json",
        },
        "pipelineRepeatability": {
            "cells": determinism["nativeLeiden"]["cells"],
            "runsPerMode": sum(
                row["mode"] == "deterministic" for row in determinism["pipelineRuns"]
            ),
            "allArraysIdentical": all(
                all(row["exactArrayEquality"].values()) for row in deterministic_pairs
            ),
            "deterministicARI": min(row["clusterARI"] for row in deterministic_pairs),
            "source": "experiments/methods/records/rna_numerical_reproducibility.json",
        },
        "lsi": {
            "cells": lsi_full[0]["shape"][0],
            "peaks": lsi_full[0]["shape"][1],
            "dimensions": lsi_full[0]["settings"]["dimensions"],
            "skipFirst": lsi_full[0]["settings"]["skipFirst"],
            "toleranceDegrees": lsi_full[0]["settings"]["maxSubspaceAngleDegrees"],
            "biologicalWorkflowIterations": 5,
            "biologicalWorkflowOversamples": 10,
            "sampledSelection": {
                "cells": lsi["shape"][0],
                "maxSubspaceAngleDegrees": max(
                    row["maxSubspaceAngleDegrees"] for row in lsi["results"]
                ),
            },
            "iterations": 80,
            "oversamples": 100,
            "maxSubspaceAngleDegrees": max(row["maxSubspaceAngleDegrees"] for row in lsi_matched),
            "maxProjectionAbsoluteError": max(
                row["maxProjectionAbsoluteError"] for row in lsi_matched
            ),
            "source": "experiments/methods/records/lsi.json",
        },
        "scenic": {
            "regulators": len(pruning["settings"]["regulators"]),
            "seeds": len(pruning["settings"]["seeds"]),
            "coreCases": len(pruning_core),
            "aucMaxAbsoluteError": max(case["aucMaxAbsoluteError"] for case in pruning_core),
            "curveMaxAbsoluteError": max(case["curveMaxAbsoluteError"] for case in pruning_core),
            "referenceVersion": pruning["referenceVersions"]["pyscenic"],
            "source": "experiments/methods/records/scenic_pruning.json",
        },
        "chromvar": {
            "cells": chromvar["cells"],
            "peaks": chromvar["peaks"],
            "motifs": chromvar["motifs"],
            "referenceVersion": chromvar["reference"]["version"],
            "deviationMaxAbsoluteError": chromvar["results"]["deviations"]["maxAbsoluteError"],
            "zScoreMaxAbsoluteError": chromvar["results"]["z"]["maxAbsoluteError"],
            "finiteMasksEqual": all(row["finiteMaskEqual"] for row in chromvar["results"].values()),
            "source": "experiments/methods/records/chromvar.json",
        },
        "dropletQc": {
            "datasets": [
                {
                    "name": row["dataset"],
                    "rawBarcodes": row["rawBarcodes"],
                    "testedBarcodes": row["emptydrops"]["testedBarcodes"],
                    "filteredCells": row["soupx"]["filteredCells"],
                    "cellCallJaccard": row["emptydrops"]["cellCallJaccard"],
                    "nativeCalls": row["emptydrops"]["nativeCells"],
                    "referenceCalls": row["emptydrops"]["referenceCells"],
                }
                for row in qc_cases
            ],
            "emptyDropsVersion": qc_cases[0]["emptydrops"]["reference"]["version"],
            "soupXVersion": qc_cases[0]["soupx"]["reference"]["version"],
            "emptyDropsAmbientMaxAbsoluteError": max(
                row["emptydrops"]["ambientMaxAbsoluteError"] for row in qc_cases
            ),
            "soupAmbientCountsIdentical": all(
                row["soupx"]["ambientCountsIdentical"] for row in qc_cases
            ),
            "soupProfileMaxAbsoluteError": max(
                row["soupx"]["ambientProfileMaxAbsoluteError"] for row in qc_cases
            ),
            "rhoMaxAbsoluteError": max(row["soupx"]["rhoMaxAbsoluteError"] for row in qc_cases),
            "correctedCountMaxAbsoluteError": max(
                row["soupx"]["correctedCountMaxAbsoluteError"] for row in qc_cases
            ),
            "source": "experiments/methods/records/droplet_qc.json",
        },
        "geneScores": {
            "cells": gene_scores["cells"],
            "peaks": gene_scores["peaks"],
            "genes": gene_scores["genes"],
            "maxAbsoluteError": gene_scores["maxAbsoluteError"],
            "genesWithoutOverlaps": gene_scores["genesWithoutOverlaps"],
            "source": "experiments/methods/records/gene_scores.json",
        },
        "hto": {
            "datasets": [
                {
                    "name": record["dataset"],
                    "cells": record["cells"],
                    "hashtags": record["hashtags"],
                    "identityAgreement": record["identityAgreement"],
                    "globalClassAgreement": record["globalClassAgreement"],
                }
                for record in hto_results
            ],
            "minimumIdentityAgreement": min(record["identityAgreement"] for record in hto_results),
            "globalClassesIdentical": all(
                record["globalClassAgreement"] == 1.0 for record in hto_results
            ),
            "identityDifferences": sum(
                round((1 - record["identityAgreement"]) * record["cells"]) for record in hto_results
            ),
            "seuratVersion": hto_results[0]["referenceVersions"]["Seurat"],
            "source": "experiments/methods/records/hto_demultiplexing.json",
        },
        "grnboost2": {
            "cells": grn["cells"],
            "genes": grn["genes"],
            "factors": grn["transcriptionFactors"],
            "matchedPearson": grn["matchedStopping"]["pearson"],
            "matchedMaxAbsoluteDifference": grn["matchedStopping"][
                "maxImportanceAbsoluteDifference"
            ],
            "arboretoVersion": grn["referenceVersions"]["arboreto"],
            "sklearnVersion": grn["referenceVersions"]["scikit-learn"],
            "source": "experiments/methods/records/grnboost2.json",
        },
        "harmony": harmony,
        "wnn": {
            "citeCells": multimodal["datasets"]["tenx_8K_pbmc_citeseq"]["cells"],
            "teaCells": multimodal["datasets"]["swanson_7K_pbmc_teaseq"]["cells"],
            "seuratVersion": next(iter(seurat_versions)),
            "maxAbs": max(
                record["result"]["matchedKernel"][key]
                for record in multimodal_records
                for key in ("weightsMaxAbs", "affinitiesMaxAbs")
            ),
            "neighborsIdentical": all(
                record["result"]["matchedKernel"]["neighborsIdentical"]
                for record in multimodal_records
            ),
            "condition": multimodal["sharedPoolCondition"],
            "source": ".build/derived/integration/multimodal_neighbors.json",
        },
        "wnnFixture": {
            "cellsTwo": wnn["twoModality"]["cells"],
            "cellsThree": wnn["threeModality"]["cells"],
            "maxAbs": max(
                wnn[k]["matchedPool"][f]
                for k in ("twoModality", "threeModality")
                for f in ("modalityWeightsMaxAbs", "neighborAffinitiesMaxAbs")
            ),
            "neighborsIdentical": all(
                wnn[k]["matchedPool"]["neighborIndicesIdentical"]
                for k in ("twoModality", "threeModality")
            ),
            "float32Resolution": wnn["twoModality"]["matchedPool"]["float32Resolution"],
            "defaultsOverlap": wnn["twoModality"]["seuratDefaults"]["neighborOverlap"],
        },
        "lisi": {"version": lisi["reference"]["scibMetricsVersion"], "maxPerCell": lisi_max},
        "cellCycle": {
            "version": cc["reference"]["version"],
            "cellsPancreas": cc["datasets"]["pancreas"]["dataset"]["cells"],
            "cellsPbmc": cc["datasets"]["pbmc"]["dataset"]["cells"],
            "maxAbsSameControls": cc_max,
            "randomControlsS": cc["datasets"]["pancreas"]["matched"]["nBins50"]["S"]["pearson"],
            "randomControlsG2M": cc["datasets"]["pancreas"]["matched"]["nBins50"]["G2M"]["pearson"],
        },
        "aucell": {
            "version": aucell["reference"]["version"],
            "cells": aucell["dataset"]["cells"],
            "aucell": aucell["aucellMatched"]["maxAbsDiffOverall"],
            "waggr": aucell["waggrMatched"]["noLog"]["maxAbsDiffOverall"],
            "sets": len(aucell["waggrMatched"]["noLog"]["sources"]),
            "source": "experiments/methods/records/aucell.json",
        },
        "magic": {
            "version": magic["reference"]["version"],
            "cells": magic["dataset"]["cells"],
            "genes": magic["dataset"]["genesImputed"],
            "maxAbs": magic["matched"]["imputed"]["maxAbsDiff"],
            "range": magic["matched"]["imputed"]["rangeFirst"],
            "perGenePearsonMin": magic["matched"]["perGene"]["pearson"]["min"],
            "defaultsMedianR": magic["defaults"]["sameInputValues"]["perGene"]["pearson"]["median"],
        },
        "pba": {
            "cells": pba["dataset"]["cells"],
            "spearman": pba["comparisons"]["nSingularVals30"]["scaledCyteArcVersusExact"][
                "spearman"
            ],
            "maxAbs": pba["comparisons"]["nSingularVals30"]["scaledCyteArcVersusExact"][
                "maxAbsDiff"
            ],
            "versusDenseTruncated": pba["comparisons"]["nSingularVals30"][
                "scaledCyteArcVersusDenseTruncated"
            ]["maxAbsDiff"],
            "denseTruncatedSpearman": pba["comparisons"]["nSingularVals30"][
                "scaledCyteArcVersusDenseTruncated"
            ]["spearman"],
            "modesUsed": pba["comparisons"]["nSingularVals30"]["modesUsed"],
        },
        "fate": {
            "pancreasCells": fate["pancreas"]["cells"],
            "pancreasPearsonMin": min(v["pearson"] for v in pancreas.values()),
            "pancreasMaxAbs": max(v["maxAbsDiff"] for v in pancreas.values()),
            "pancreasDefaultsPearson": fate["pancreas"]["cellrankDefaults"]["agreement"]["Alpha"][
                "pearson"
            ],
            "larryProgenitors": fate["larry"]["evaluatedProgenitors"],
            "larryPearson": fate["larry"]["cytearcVsCellrankPearson"],
            "solverMaxAbs": max(g["maxAbsError"] for g in dense["controlledGraphs"]),
        },
        "symphony": {
            "cells": symphony["queryCells"],
            "referenceCells": symphony_preparation["dataset"]["referenceCells"],
            "referenceStudies": len(symphony_preparation["dataset"]["referenceStudies"]),
            "genes": symphony["features"],
            "maxAbs": symphony["errors"]["correctedMaxAbs"],
            "projectedMaxAbs": symphony["errors"]["projectedMaxAbs"],
            "assignmentMaxAbs": symphony["errors"]["assignmentsMaxAbs"],
            "correctionShift": symphony_preparation["correctionMagnitudeMaxAbs"],
            "scaling": "Shared unclipped reference PCA scaling",
        },
    }


def prepare() -> dict:
    values = agreement_values(marker_agreement())
    write_metrics("table_2/data", values)
    return {
        "agreement": values,
    }
