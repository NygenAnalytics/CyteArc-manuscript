import json

import numpy as np
from cytearc.mapping.models import ScaledPCAProjectionModel, SymphonyCorrectionModel
from cytearc.mapping.symphony import (
    accumulate_sufficient_statistics,
    apply_query_correction,
    initialize_sufficient_statistics,
    project_pca,
    soft_cluster_assignments,
    solve_query_correction,
)

from experiments.mapping import DATA, HERE
from experiments.runtime.common import write_json

FIXTURE = HERE / "fixtures" / "symphony_r_0_1_3_golden.json"


def errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    absolute = np.abs(actual - expected)
    scale = np.abs(expected).max()
    return {
        "maxAbsolute": float(absolute.max()),
        "meanAbsolute": float(absolute.mean()),
        "maxRelativeToScale": float(absolute.max() / scale) if scale else 0.0,
        "shape": list(actual.shape),
    }


def replay(
    reference: dict, query: np.ndarray, batch_codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SymphonyCorrectionModel]:
    projection = ScaledPCAProjectionModel(
        feature_means=np.asarray(reference["featureMeans"]),
        center=np.zeros_like(np.asarray(reference["featureMeans"])),
        feature_scales=np.asarray(reference["featureScales"]),
        loadings=np.asarray(reference["loadings"]),
    )
    model = SymphonyCorrectionModel(
        centroids=np.asarray(reference["centroids"]),
        raw_centroids=np.asarray(reference["rawCentroids"]),
        corrected_centroids=np.asarray(reference["correctedCentroids"]),
        cluster_mass=np.asarray(reference["clusterMass"]),
        sigma=np.asarray(reference["sigma"]),
    )
    projected = project_pca(query, projection)
    assignments = soft_cluster_assignments(projected, model)
    counts, sums = initialize_sufficient_statistics(int(batch_codes.max()) + 1, model)
    accumulate_sufficient_statistics(counts, sums, projected, assignments, batch_codes)
    corrected = apply_query_correction(
        projected, assignments, batch_codes, model, solve_query_correction(counts, sums, model)
    )
    return (projected, assignments, corrected, model)


def main() -> None:
    fixture = json.loads(FIXTURE.read_text())
    query = np.asarray(fixture["query"])
    batch_codes = np.asarray(fixture["batchCodes"], dtype=np.int64)
    projected, assignments, corrected, model = replay(fixture["reference"], query, batch_codes)
    nonzero = fixture["nonzeroCorrection"]
    nz_query = np.asarray(nonzero["query"])
    nz_batches = np.asarray(nonzero["batchCodes"], dtype=np.int64)
    nz_projected, nz_assignments, nz_corrected, nz_model = replay(
        nonzero["reference"], nz_query, nz_batches
    )
    payload = {
        "description": "CyteArc Symphony correction replayed on the R Symphony 0.1.3 golden fixture.",
        "fixtureProvenance": fixture.get("provenance"),
        "queryCells": int(query.shape[0]),
        "panelGenes": int(query.shape[1]),
        "dims": int(projected.shape[1]),
        "clusters": int(model.centroids.shape[0]),
        "batches": int(batch_codes.max()) + 1,
        "correctionRidge": fixture["reference"].get("correctionRidge"),
        "projected": errors(projected, fixture["expectedProjected"]),
        "assignments": errors(assignments, fixture["expectedAssignments"]),
        "corrected": errors(corrected, fixture["expectedCorrected"]),
        "correctionMagnitude": {
            "maxAbsShift": float(np.abs(corrected - projected).max()),
            "meanAbsShift": float(np.abs(corrected - projected).mean()),
        },
        "nonzeroCorrection": {
            "queryCells": int(nz_query.shape[0]),
            "batches": int(nz_batches.max()) + 1,
            "clusters": int(nz_model.centroids.shape[0]),
            "assignments": errors(nz_assignments, nonzero["expectedAssignments"]),
            "corrected": errors(nz_corrected, nonzero["expectedCorrected"]),
            "correctionMagnitude": {
                "maxAbsShift": float(np.abs(nz_corrected - nz_projected).max()),
                "meanAbsShift": float(np.abs(nz_corrected - nz_projected).mean()),
            },
        },
    }
    write_json("symphony_golden_agreement.json", payload, directory=DATA)
    print(
        json.dumps(
            {k: payload[k] for k in ("projected", "assignments", "corrected", "nonzeroCorrection")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
