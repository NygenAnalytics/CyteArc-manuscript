import json

import numpy as np
from cytearc.neighbors.integration import _wnn_integration_many

from experiments.integration import DATA, HERE
from experiments.runtime.common import write_json


def wnn_integration(name1, indices1, ld1, name2, indices2, ld2, nthreads, *, l2_normalize=True):
    names = (name1, name2) if name1 != name2 else (f"{name1} [first]", f"{name2} [second]")
    return _wnn_integration_many(
        [(names[0], indices1, ld1), (names[1], indices2, ld2)], nthreads, l2_normalize=l2_normalize
    )


TWO_WAY = HERE / "fixtures" / "seurat_wnn_5_5_1_golden.json"
THREE_WAY = HERE / "fixtures" / "seurat_wnn_3way_5_5_1_golden.json"


def two_way() -> dict:
    fixture = json.loads(TWO_WAY.read_text())
    inputs = fixture["inputs"]
    indices1 = np.asarray(inputs["rnaIndices"], dtype=np.uint32)
    indices2 = np.asarray(inputs["adtIndices"], dtype=np.uint32)
    graph, weights = wnn_integration(
        "RNA",
        indices1,
        np.asarray(inputs["rnaEmbedding"], dtype=np.float64),
        "ADT",
        indices2,
        np.asarray(inputs["adtEmbedding"], dtype=np.float64),
        nthreads=1,
        l2_normalize=fixture["provenance"]["l2Normalize"],
    )
    shape = indices1.shape
    selected = graph.col.reshape(shape)
    affinities = graph.data.reshape(shape).astype(np.float64)
    weights = weights.astype(np.float64)
    matched = fixture["matched"]
    default = fixture["seuratDefault"]
    k = selected.shape[1]
    seurat_indices = np.asarray(default["neighborIndices"])
    seurat_weights = np.asarray(default["modalityWeights"])
    overlap = float(
        np.mean(
            [
                len(set(row.tolist()) & set(ref.tolist())) / k
                for row, ref in zip(selected, seurat_indices, strict=True)
            ]
        )
    )
    return {
        "fixtureProvenance": fixture["provenance"],
        "cells": int(shape[0]),
        "k": int(k),
        "matchedPool": {
            "neighborIndicesIdentical": bool(
                np.array_equal(
                    selected, np.asarray(matched["neighborIndices"], dtype=selected.dtype)
                )
            ),
            "modalityWeightsMaxAbs": float(
                np.abs(weights - np.asarray(matched["modalityWeights"])).max()
            ),
            "neighborAffinitiesMaxAbs": float(
                np.abs(affinities - np.asarray(matched["neighborAffinities"])).max()
            ),
            "float32Resolution": float(np.finfo(np.float32).eps),
        },
        "seuratDefaults": {
            "neighborOverlap": overlap,
            "rnaWeightPearson": float(np.corrcoef(weights[:, 0], seurat_weights[:, 0])[0, 1]),
            "rnaWeightMeanCyteArc": float(weights[:, 0].mean()),
            "rnaWeightMeanSeurat": float(seurat_weights[:, 0].mean()),
        },
    }


def three_way() -> dict:
    fixture = json.loads(THREE_WAY.read_text())
    inputs = fixture["inputs"]
    modalities = [
        (
            name,
            np.asarray(inputs["neighborIndices"][name], dtype=np.uint32),
            np.asarray(inputs["embeddings"][name], dtype=np.float64),
        )
        for name in inputs["modalityNames"]
    ]
    graph, weights = _wnn_integration_many(
        modalities, nthreads=1, l2_normalize=fixture["provenance"]["l2Normalize"]
    )
    shape = modalities[0][1].shape
    selected = graph.col.reshape(shape)
    affinities = graph.data.reshape(shape).astype(np.float64)
    matched = fixture["matched"]
    return {
        "fixtureProvenance": fixture["provenance"],
        "cells": int(shape[0]),
        "k": int(shape[1]),
        "modalities": inputs["modalityNames"],
        "matchedPool": {
            "neighborIndicesIdentical": bool(
                np.array_equal(
                    selected, np.asarray(matched["neighborIndices"], dtype=selected.dtype)
                )
            ),
            "modalityWeightsMaxAbs": float(
                np.abs(weights.astype(np.float64) - np.asarray(matched["modalityWeights"])).max()
            ),
            "neighborAffinitiesMaxAbs": float(
                np.abs(affinities - np.asarray(matched["neighborAffinities"])).max()
            ),
        },
    }


def main() -> None:
    payload = {
        "description": "CyteArc WNN replayed on the Seurat 5.5.1 golden fixtures.",
        "twoModality": two_way(),
        "threeModality": three_way(),
    }
    write_json("weighted_neighbors_seurat_agreement.json", payload, directory=DATA)
    print(
        json.dumps(
            {
                "twoModality": payload["twoModality"]["seuratDefaults"],
                "matched": payload["twoModality"]["matchedPool"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
