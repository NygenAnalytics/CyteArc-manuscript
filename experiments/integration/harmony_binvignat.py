import shutil
from pathlib import Path

import cytearc
import numpy as np
import pandas as pd
from cytearc.metrics import clisi_knn, ilisi_knn

from experiments.integration import DATA, WORK
from experiments.integration import INPUTS as DATASETS
from experiments.runtime.common import Stopwatch, configure_cytearc, write_json

SOURCE = DATASETS / "binvignat" / "data.zarr"
SELECTION = "manuscript_ra_balanced"
CELLS_PER_GROUP = 1500
SEED = 42
DIMS = 30
K = 15
# Options of every comparison with harmonypy: CyteArc 1.3.0's fit_harmony defaults, which CyteArc
# applies when the options are left out, passed to harmonypy under its own names so that neither
# tool's defaults decide the comparison. The cluster count follows CyteArc's rule for the cell
# count.
HARMONY2_SETTINGS = {
    "theta": 2.0,
    "sigma": 0.1,
    "tau": 0.0,
    "ridge_alpha": 0.2,
    "prune_threshold": 1e-5,
    "block_size": 0.05,
    "max_iter_cluster": 200,
    "max_iter_harmony": 40,
    "epsilon_cluster": 1e-5,
    "epsilon_harmony": 1e-4,
}
HARMONYPY_NAMES = {
    "ridge_alpha": "alpha",
    "prune_threshold": "batch_prop_cutoff",
    "max_iter_cluster": "max_iter_kmeans",
}
# The harmonypy release of the Harmony2 comparisons, the version the harmonypy image installs.
HARMONYPY_VERSION = "2.0.2"
REFERENCE_SEED = 0


def harmonypy_arguments(settings: dict, cells: int) -> dict:
    """harmonypy keyword arguments for CyteArc Harmony2 options: harmonypy's names, CyteArc's
    cluster count for the cells, the estimated ridge penalty, one thread, and -1 for a stopping
    tolerance of 0, which in CyteArc stops only on an unchanged objective and in harmonypy never
    stops."""
    from cytearc.embeddings.harmony import harmony_cluster_count

    unknown = set(settings) - set(HARMONY2_SETTINGS)
    if unknown:
        raise ValueError(f"No harmonypy equivalent for Harmony options {sorted(unknown)}")
    arguments = {HARMONYPY_NAMES.get(name, name): value for name, value in settings.items()}
    for name in ("epsilon_cluster", "epsilon_harmony"):
        if arguments.get(name) == 0:
            arguments[name] = -1.0
    return {
        **arguments,
        "nclust": harmony_cluster_count(cells),
        "lamb": None,
        "ncores": 1,
        "verbose": False,
    }


def reference_harmony(coordinates: np.ndarray, batches: np.ndarray, seed: int, **changes):
    """harmonypy's Harmony2 on the same coordinates and batches with HARMONY2_SETTINGS and the
    given changes; returns the corrected coordinates, one row per cell, and harmonypy's result
    object."""
    import harmonypy

    if harmonypy.__version__ != HARMONYPY_VERSION:
        raise RuntimeError(
            f"The Harmony2 comparisons use harmonypy {HARMONYPY_VERSION}, which the harmonypy "
            f"image installs; this environment has harmonypy {harmonypy.__version__}"
        )
    arguments = harmonypy_arguments({**HARMONY2_SETTINGS, **changes}, coordinates.shape[0])
    fitted = harmonypy.run_harmony(
        coordinates, pd.DataFrame({"batch": batches}), ["batch"], random_state=seed, **arguments
    )
    corrected = np.asarray(fitted.Z_corr, dtype=np.float64)
    if corrected.shape != coordinates.shape:
        corrected = corrected.T
    return corrected, fitted


def knn(coordinates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    distances, indices = model.kneighbors(coordinates)
    return (distances[:, 1:], indices[:, 1:])


def main() -> None:
    configure_cytearc()
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{SOURCE} is missing; execute the batch_correction tutorial once to create it"
        )
    target = WORK / "harmony_binvignat" / "ra_batch.zarr"
    if target.exists():
        shutil.rmtree(target)
    ds = cytearc.mount_datastore(
        str(SOURCE), at=str(target), default_assay="RNA", min_features_per_cell=0, nthreads=8
    )
    batch = ds.cells.fetch_all("batch").astype(str)
    disease = ds.cells.fetch_all("disease").astype(str)
    rng = np.random.default_rng(SEED)
    selected = np.zeros(ds.cells.N, dtype=bool)
    for batch_value in np.unique(batch):
        for disease_value in np.unique(disease):
            candidates = np.flatnonzero((batch == batch_value) & (disease == disease_value))
            selected[rng.choice(candidates, size=CELLS_PER_GROUP, replace=False)] = True
    ds.cells.insert(column_name=SELECTION, values=selected, overwrite=True)
    cells = ds.snapshot_cell_selection(SELECTION)
    hvg = ds.features.hvgs(cells, min_cells=20, top_n=2000, show_plot=False)
    normalized = ds.features.normalize(cells, hvg)
    with Stopwatch() as pca_time:
        pca = ds.reduction.pca(normalized, dims=DIMS, show_elbow_plot=False)
    ann = ds.graph.ann_index(pca)
    neighbors = ds.graph.neighbors(ann, k=K)
    uncorrected = {
        "iLISI": float(ds.integration.compute_ilisi("batch", neighbors)),
        "cLISI": float(ds.integration.compute_clisi("cell_type", neighbors)),
    }
    with Stopwatch() as harmony_time:
        corrected = ds.reduction.harmony(pca, ["batch"])
    ann_h = ds.graph.ann_index(corrected)
    neighbors_h = ds.graph.neighbors(ann_h, k=K)
    cytearc_harmony = {
        "iLISI": float(ds.integration.compute_ilisi("batch", neighbors_h)),
        "cLISI": float(ds.integration.compute_clisi("cell_type", neighbors_h)),
        "seconds": harmony_time.seconds,
    }
    coordinates = np.asarray(ds.artifacts.load(pca)["data"][:], dtype=np.float64)
    cytearc_corrected = np.asarray(ds.artifacts.load(corrected)["data"][:], dtype=np.float64)
    batch_selected = np.asarray(ds.cells.fetch("batch", key=SELECTION)).astype(str)
    types_selected = np.asarray(ds.cells.fetch("cell_type", key=SELECTION)).astype(str)
    if coordinates.shape[0] != batch_selected.size:
        raise RuntimeError("reduction rows do not match the selection")
    comparison: dict = {"available": False}
    try:
        import harmonypy
    except ImportError:
        harmonypy = None
    if harmonypy is not None:
        with Stopwatch() as reference_time:
            reference_corrected, _ = reference_harmony(coordinates, batch_selected, REFERENCE_SEED)
        d_ref, i_ref = knn(reference_corrected, K)
        d_cytearc, i_cytearc = knn(cytearc_corrected, K)
        overlap = float(
            np.mean(
                [
                    len(set(a.tolist()) & set(b.tolist())) / K
                    for a, b in zip(i_cytearc, i_ref, strict=True)
                ]
            )
        )
        per_dim = [
            float(np.corrcoef(cytearc_corrected[:, j], reference_corrected[:, j])[0, 1])
            for j in range(DIMS)
        ]
        comparison = {
            "available": True,
            "harmonypyVersion": getattr(harmonypy, "__version__", "unknown"),
            "settings": {
                **harmonypy_arguments(HARMONY2_SETTINGS, len(batch_selected)),
                "random_state": REFERENCE_SEED,
            },
            "seconds": reference_time.seconds,
            "iLISI": float(ilisi_knn(d_ref, i_ref, batch_selected)),
            "cLISI": float(clisi_knn(d_ref, i_ref, types_selected)),
            "cytearcCorrectedRecomputed": {
                "iLISI": float(ilisi_knn(d_cytearc, i_cytearc, batch_selected)),
                "cLISI": float(clisi_knn(d_cytearc, i_cytearc, types_selected)),
            },
            "perDimensionPearsonMedian": float(np.median(per_dim)),
            "perDimensionPearsonMin": float(np.min(per_dim)),
            "perDimensionPearson": per_dim,
            "knnOverlapK15": overlap,
            "rmseOverStd": float(
                np.sqrt(np.mean((cytearc_corrected - reference_corrected) ** 2))
                / np.std(reference_corrected)
            ),
        }
    payload = {
        "description": "Uncorrected versus Harmony-corrected neighbour graphs on a balanced RA PBMC subset, with harmonypy as the reference implementation.",
        "dataset": {
            "source": str(SOURCE),
            "cellsTotal": int(ds.cells.N),
            "cellsSelected": int(selected.sum()),
            "batches": sorted(set(batch)),
            "cellTypes": int(len(set(types_selected))),
        },
        "parameters": {
            "hvg": 2000,
            "dims": DIMS,
            "k": K,
            "seed": SEED,
            "cellsPerGroup": CELLS_PER_GROUP,
            "pcaSeconds": pca_time.seconds,
        },
        "uncorrected": uncorrected,
        "cytearcHarmony": cytearc_harmony,
        "harmonypy": comparison,
    }
    write_json("harmony_binvignat_comparison.json", payload, directory=DATA)
    print("uncorrected", uncorrected)
    print("cytearc harmony", cytearc_harmony)
    print("harmonypy", {k: v for k, v in comparison.items() if k != "perDimensionPearson"})


if __name__ == "__main__":
    main()


def prepare_inputs(payload: dict) -> None:
    """Download and convert the Binvignat PBMCs unless their store exists; runtime.method_checks
    calls this before the Binvignat checks."""
    if not SOURCE.exists():
        prepare_binvignat(payload)


def prepare_binvignat(payload: dict) -> str:
    import urllib.request

    import cytearc

    folder = DATASETS / "binvignat"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "source.h5ad"
    destination = folder / "data.zarr"
    if destination.exists():
        raise FileExistsError(destination)
    urllib.request.urlretrieve(
        "https://datasets.cellxgene.cziscience.com/3b751975-34bb-409a-a9b7-98380f0450ea.h5ad",
        source,
    )
    inspection = cytearc.inspect_h5ad(str(source))
    if inspection.matrixKey != "raw/X" or (inspection.nCells, inspection.nFeatures) != (
        108_717,
        21_648,
    ):
        raise RuntimeError("The Binvignat input does not match the recorded dataset")
    reader = cytearc.H5adReader.from_inspect(inspection)
    try:
        cytearc.H5adToZarr(
            reader, zarr_loc=str(destination), nthreads=int(payload.get("workers", 8))
        ).dump()
    finally:
        reader.h5.close()
    prepare_store(destination, int(payload.get("workers", 8)))
    return str(destination)


def prepare_store(destination: Path, workers: int) -> None:
    """Open a freshly converted store once writable, which prepares its RNA assay: CyteArc mounts only a prepared
    store, and the checks mount this one."""
    cytearc.DataStore(
        str(destination),
        default_assay="RNA",
        zarr_mode="r+",
        min_features_per_cell=0,
        nthreads=workers,
    )
