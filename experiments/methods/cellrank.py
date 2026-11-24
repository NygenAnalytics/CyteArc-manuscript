from pathlib import Path

from experiments.runtime.object_io import _download

LARRY_URL = "https://zenodo.org/api/records/11401394/files/Weinreb_in_vitro.h5ad/content"


def _cytearc_graph_pseudotime_fates(
    ds, run, source_label: str, sink_labels: list[str], annotation_column: str, n_sinks: int
) -> dict:
    import numpy as np

    graph_ref = run["connectivity_map"]
    annotations = np.asarray(ds.cells.fetch(annotation_column, key="I")).astype(str)
    source_mask = annotations == source_label
    sink_mask = np.isin(annotations, sink_labels)
    vector = np.zeros(annotations.size)
    vector[source_mask] = -1.0 / max(1, source_mask.sum())
    vector[sink_mask] = 1.0 / max(1, sink_mask.sum())
    pseudotime_ref = ds.trajectory.pseudotime(graph_ref, ss_vec=vector)
    pseudotime = ds.trajectory.load_pseudotime(pseudotime_ref)
    cluster_values = np.asarray(run.cells.fetch("clusters"))
    cluster_labels = cluster_values.astype(str)
    valid = np.asarray(pseudotime.valid)
    values = np.asarray(pseudotime.values)
    candidates = {}
    for label in np.unique(cluster_labels[valid]):
        mask = valid & (cluster_labels == label)
        majority_values, counts = np.unique(annotations[mask], return_counts=True)
        majority = str(majority_values[counts.argmax()])
        if majority in sink_labels:
            candidates[label] = (majority, float(values[mask].mean()), int(mask.sum()))
    chosen: dict[str, str] = {}
    for majority in sink_labels:
        options = [
            (label, mean, size) for label, (m, mean, size) in candidates.items() if m == majority
        ]
        if options:
            label = max(options, key=lambda item: item[1])[0]
            chosen[label] = majority
        if len(chosen) >= n_sinks:
            break
    terminal = [cluster_values[cluster_labels == label][0].item() for label in chosen]
    fate_ref = ds.trajectory.fate(pseudotime_ref, run["clusters"], sinks=terminal)
    fate = ds.trajectory.load_fate(fate_ref)
    return {
        "graphRef": graph_ref,
        "pseudotime": values,
        "valid": valid,
        "annotations": annotations,
        "clusterLabels": cluster_labels,
        "terminalClusters": {str(k): v for k, v in chosen.items()},
        "fateValues": np.asarray(fate.values),
        "fateValid": np.asarray(fate.valid),
        "fateSinks": [str(s) for s in fate.sink_labels],
    }


def _pancreas_larry_fate_predictions(
    ds, cytearc_result: dict, b: float, nu: float, normalize_time: bool
) -> dict:
    import anndata as ad
    import numpy as np
    import pandas as pd
    from cellrank.estimators import GPCCA
    from cellrank.kernels import PseudotimeKernel
    from scipy.sparse import csr_matrix

    valid = cytearc_result["valid"]
    graph = ds.graph.load(cytearc_result["graphRef"], symmetric=True, upper_only=False)
    graph = csr_matrix(graph[valid][:, valid])
    n = graph.shape[0]
    pseudotime = cytearc_result["pseudotime"][valid].astype(np.float64)
    if normalize_time:
        pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
    adata = ad.AnnData(X=csr_matrix((n, 1), dtype=np.float32))
    adata.obs["pseudotime"] = pseudotime
    adata.obsp["connectivities"] = graph
    adata.obsp["distances"] = graph
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"method": "umap", "n_neighbors": 11},
    }
    labels = cytearc_result["clusterLabels"][valid]
    terminal = pd.Series(np.full(n, np.nan, dtype=object), index=adata.obs_names)
    for cluster, name in cytearc_result["terminalClusters"].items():
        terminal[labels == cluster] = name
    kernel = PseudotimeKernel(adata, time_key="pseudotime").compute_transition_matrix(
        threshold_scheme="soft", b=b, nu=nu
    )
    estimator = GPCCA(kernel)
    estimator.set_terminal_states(states=terminal.astype("category"))
    estimator.compute_fate_probabilities(use_petsc=False, solver="gmres", tol=1e-06)
    probabilities = np.asarray(estimator.fate_probabilities.X)
    names = list(estimator.fate_probabilities.names)
    return {"probabilities": probabilities, "names": names}


def cellrank_job(payload: dict) -> dict:
    import os
    import time
    import traceback

    import numpy as np
    import pandas as pd

    os.environ["R2_ENDPOINT_URL"] = payload["r2EndpointUrl"]
    import cytearc

    from experiments.runtime.object_io import put_json

    cytearc.configure_output(level="WARNING", progress=False)
    work = Path("/tmp/cellrank")
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"experiment": "cellrank", "pancreas": {}, "larry": {}}
    started = time.perf_counter()

    def compare(cytearc_result: dict, cr: dict) -> dict:
        sv = cytearc_result["fateValues"][cytearc_result["valid"]]
        fv = cytearc_result["fateValid"][cytearc_result["valid"]]
        out = {}
        for i, sink in enumerate(cytearc_result["fateSinks"]):
            name = cytearc_result["terminalClusters"].get(sink, sink)
            j = cr["names"].index(name) if name in cr["names"] else None
            if j is None:
                continue
            a = sv[fv, i].astype(np.float64)
            b = cr["probabilities"][fv, j].astype(np.float64)
            out[name] = {
                "pearson": float(np.corrcoef(a, b)[0, 1]),
                "maxAbsDiff": float(np.abs(a - b).max()),
                "meanAbsDiff": float(np.abs(a - b).mean()),
                "cells": int(fv.sum()),
            }
        return out

    try:
        repository = cytearc.datasets.connect("cytearc_docs")
        path = repository.download_dataset(
            "bastidas-ponce_4K_pancreas-d15_rnaseq", destination=str(work), zarr=True
        )
        ds = cytearc.DataStore(f"{path}/data.zarr", nthreads=int(payload["workers"]))
        run = ds.pipeline.open(label="docs_default")
        t = time.perf_counter()
        cytearc_result = _cytearc_graph_pseudotime_fates(
            ds, run, "Ductal", ["Alpha", "Beta"], "clusters", 2
        )
        cytearc_seconds = time.perf_counter() - t
        pancreas = {
            "terminalClusters": cytearc_result["terminalClusters"],
            "cytearcSeconds": cytearc_seconds,
            "cells": int(cytearc_result["valid"].sum()),
        }
        for label, (b, nu, norm) in {
            "matchedCyteArc": (10.0, 1.0, True),
            "cellrankDefaults": (10.0, 0.5, False),
        }.items():
            t = time.perf_counter()
            cr = _pancreas_larry_fate_predictions(ds, cytearc_result, b, nu, norm)
            pancreas[label] = {
                "seconds": time.perf_counter() - t,
                "b": b,
                "nu": nu,
                "normalizedPseudotime": norm,
                "agreement": compare(cytearc_result, cr),
            }
        result["pancreas"] = pancreas
    except BaseException as exc:
        result["pancreas"] = {
            "status": "error",
            "error": "".join(traceback.format_exception(exc))[-3000:],
        }
    try:
        larry_path = work / "Weinreb_in_vitro.h5ad"
        _download(LARRY_URL, larry_path)
        import h5py

        with h5py.File(larry_path, "r") as h5:
            obs_keys = list(h5["obs"].keys())
            obsm_keys = list(h5["obsm"].keys()) if "obsm" in h5 else []
            layers = list(h5["layers"].keys()) if "layers" in h5 else []
            has_raw = "raw" in h5
        result["larry"] = {
            "obsColumns": obs_keys,
            "obsmKeys": obsm_keys,
            "layers": layers,
            "hasRaw": has_raw,
            "bytes": larry_path.stat().st_size,
        }
        inspection = cytearc.inspect_h5ad(str(larry_path))
        result["larry"]["inspection"] = {
            "matrixKey": inspection.matrixKey,
            "integerLike": inspection.integerLike,
            "nCells": inspection.nCells,
            "nFeatures": inspection.nFeatures,
        }
        import anndata as ad

        meta = ad.read_h5ad(larry_path, backed="r")
        obs = meta.obs.copy()
        time_column = next(
            (
                c
                for c in obs.columns
                if c.lower().replace(" ", "_")
                in ("time_info", "time", "time_point", "timepoint", "day")
            ),
            None,
        )
        state_column = next(
            (
                c
                for c in obs.columns
                if c.lower().replace(" ", "_")
                in (
                    "state_info",
                    "cell_type",
                    "cell_type_annotation",
                    "celltype",
                    "annotation",
                    "state",
                )
            ),
            None,
        )
        clone_key = next((k for k in obsm_keys if "clone" in k.lower()), None) or next(
            (c for c in obs.columns if c.lower() == "clone"), None
        )
        result["larry"].update(
            {"timeColumn": time_column, "stateColumn": state_column, "cloneKey": clone_key}
        )
        if time_column and state_column and clone_key:
            reader = cytearc.H5adReader.from_inspect(inspection)
            store = work / "larry.zarr"
            cytearc.H5adToZarr(
                reader,
                str(store),
                assay_name="RNA",
                nthreads=int(payload["workers"]),
                mem_budget=int(payload["budgetBytes"]),
                profile="fast_local",
            ).dump()
            reader.h5.close()
            ds = cytearc.DataStore(
                str(store),
                assay_types={"RNA": "RNA"},
                default_assay="RNA",
                min_features_per_cell=10,
                nthreads=int(payload["workers"]),
                mem_budget=int(payload["budgetBytes"]),
            )
            from cytearc.assay import norm_dummy

            ds.RNA.normMethod = norm_dummy
            result["larry"]["normalization"] = "identity: X is log-normalised"
            states = obs[state_column].astype(str).to_numpy()
            times = (
                pd.to_numeric(obs[time_column], errors="coerce")
                .astype(float)
                .astype(str)
                .to_numpy()
            )
            ds.cells.insert(column_name="state_info", values=states, overwrite=True)
            ds.cells.insert(column_name="time_info", values=times, overwrite=True)
            run = ds.pipeline.rna(
                label="larry",
                filtering=False,
                hvg={"top_n": 1000},
                pca={"dims": 21},
                neighbors={"k": 11},
                umap=False,
                cell_cycle=False,
                paris=False,
                doublets=False,
                markers=False,
            )
            earliest = sorted(set(times), key=lambda v: float(v))[0]
            annotations = np.asarray(ds.cells.fetch("state_info", key="I")).astype(str)
            times_i = np.asarray(ds.cells.fetch("time_info", key="I")).astype(str)
            source_states = np.unique(annotations[times_i == earliest])
            fates = ["Neutrophil", "Monocyte"]
            combined = np.where(
                np.isin(annotations, fates),
                annotations,
                np.where(times_i == earliest, "__source__", "other"),
            )
            ds.cells.insert(column_name="fate_anchor", values=combined, overwrite=True)
            cytearc_result = _cytearc_graph_pseudotime_fates(
                ds, run, "__source__", fates, "fate_anchor", 2
            )
            cr = _pancreas_larry_fate_predictions(ds, cytearc_result, 10.0, 1.0, True)
            import scipy.sparse as sp

            selection_ids = np.asarray(ds.cells.fetch("ids", key="I")).astype(str)
            all_ids = obs.index.astype(str).to_numpy()
            id_to_row = {cell: i for i, cell in enumerate(all_ids)}
            rows = np.array([id_to_row[c] for c in selection_ids])
            if clone_key in obsm_keys:
                clones = (
                    sp.csr_matrix(meta.obsm[clone_key])
                    if not sp.issparse(meta.obsm[clone_key])
                    else meta.obsm[clone_key].tocsr()
                )
                clone_of_cell = np.full(len(all_ids), -1)
                coo = clones.tocoo()
                clone_of_cell[coo.row] = coo.col
                n_clones = clones.shape[1]
            else:
                clone_strings = obs[clone_key].astype(str)
                extracted = clone_strings.str.extract("(\\d+)\\s*$")[0]
                numeric = pd.to_numeric(extracted, errors="coerce").to_numpy()
                unlabelled = (
                    clone_strings.str.strip()
                    .str.upper()
                    .isin(["NA", "NAN", "", "-1", "NONE"])
                    .to_numpy()
                )
                clone_of_cell = np.where(np.isfinite(numeric) & ~unlabelled, numeric, -1).astype(
                    np.int64
                )
                n_clones = int(clone_of_cell.max()) + 1
            result["larry"]["cellsWithClone"] = int((clone_of_cell >= 0).sum())
            result["larry"]["clones"] = int(len(np.unique(clone_of_cell[clone_of_cell >= 0])))
            late_neutrophil = (states == "Neutrophil") & (times != earliest)
            late_monocyte = (states == "Monocyte") & (times != earliest)
            clone_neu = np.bincount(
                clone_of_cell[late_neutrophil & (clone_of_cell >= 0)], minlength=n_clones
            )
            clone_mono = np.bincount(
                clone_of_cell[late_monocyte & (clone_of_cell >= 0)], minlength=n_clones
            )
            valid = cytearc_result["valid"] & cytearc_result["fateValid"]
            early = (times_i == earliest) & valid
            cl = clone_of_cell[rows]
            observed = []
            predicted_cytearc = []
            predicted_cellrank = []
            sinks = cytearc_result["fateSinks"]
            names = [cytearc_result["terminalClusters"].get(s, s) for s in sinks]
            i_neu, i_mono = (names.index("Neutrophil"), names.index("Monocyte"))
            cr_neu, cr_mono = (cr["names"].index("Neutrophil"), cr["names"].index("Monocyte"))
            valid_rows = np.flatnonzero(valid)
            position = {row: k for k, row in enumerate(valid_rows)}
            fate_values = cytearc_result["fateValues"]
            clone_ids = []
            for idx in np.flatnonzero(early):
                clone = cl[idx]
                if clone < 0 or clone_neu[clone] + clone_mono[clone] < 2:
                    continue
                clone_ids.append(int(clone))
                observed.append(clone_neu[clone] / (clone_neu[clone] + clone_mono[clone]))
                s = fate_values[idx]
                predicted_cytearc.append(s[i_neu] / max(1e-12, s[i_neu] + s[i_mono]))
                c = cr["probabilities"][position[idx]]
                predicted_cellrank.append(c[cr_neu] / max(1e-12, c[cr_neu] + c[cr_mono]))
            observed = np.asarray(observed)
            from scipy.stats import spearmanr
            from sklearn.metrics import roc_auc_score

            rng = np.random.default_rng(0)
            clone_ids = np.asarray(clone_ids)
            unique_clones = np.unique(clone_ids)
            members = {c: np.flatnonzero(clone_ids == c) for c in unique_clones}

            def score(pred):
                pred = np.asarray(pred)
                binary = observed > 0.5
                usable = 0 < binary.sum() < len(binary)
                auc = float(roc_auc_score(binary, pred)) if usable else None
                aucs, rhos = ([], [])
                for _ in range(1000):
                    picked = rng.integers(0, len(unique_clones), len(unique_clones))
                    idx = np.concatenate([members[unique_clones[k]] for k in picked])
                    b = binary[idx]
                    if 0 < b.sum() < len(b):
                        aucs.append(roc_auc_score(b, pred[idx]))
                    rhos.append(spearmanr(pred[idx], observed[idx])[0])
                return {
                    "pearson": float(np.corrcoef(pred, observed)[0, 1]),
                    "spearman": float(spearmanr(pred, observed)[0]),
                    "spearmanCi": [
                        float(np.percentile(rhos, 2.5)),
                        float(np.percentile(rhos, 97.5)),
                    ],
                    "auroc": auc,
                    "aurocCi": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
                    if aucs
                    else None,
                    "bootstrapUnit": "clone",
                    "clonesEvaluated": int(len(unique_clones)),
                    "positives": int(binary.sum()),
                    "negatives": int((~binary).sum()),
                    "cells": int(len(pred)),
                }

            import io

            from experiments.runtime.object_io import open_r2_object

            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                observed=observed,
                cytearc=np.asarray(predicted_cytearc),
                cellrank=np.asarray(predicted_cellrank),
                clone=clone_ids,
            )
            cells_uri = payload["outputUri"].replace(".json", "_larry_cells.npz")
            store_obj, key = open_r2_object(cells_uri)
            store_obj.put(key, buffer.getvalue())
            result["larry"]["cellsUri"] = cells_uri
            result["larry"].update(
                {
                    "earliestTime": earliest,
                    "sourceStates": source_states.tolist(),
                    "terminalClusters": cytearc_result["terminalClusters"],
                    "cells": int(ds.cells.N),
                    "evaluatedProgenitors": int(len(observed)),
                    "cytearc": score(predicted_cytearc),
                    "cellrank": score(predicted_cellrank),
                    "cytearcVsCellrankPearson": float(
                        np.corrcoef(predicted_cytearc, predicted_cellrank)[0, 1]
                    ),
                }
            )
    except BaseException as exc:
        result["larry"].update(
            {"status": "error", "error": "".join(traceback.format_exception(exc))[-3000:]}
        )
    result["wallSeconds"] = time.perf_counter() - started
    put_json(payload["outputUri"], result)
    return {
        "pancreas": result["pancreas"].get("matchedCyteArc", result["pancreas"]),
        "larry": {
            k: v
            for k, v in result["larry"].items()
            if k in ("cytearc", "cellrank", "status", "evaluatedProgenitors")
        },
    }
