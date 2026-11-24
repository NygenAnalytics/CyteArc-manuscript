import os
from pathlib import Path

import cytearc
import numpy as np
from zarr.storage import LocalStore

from experiments.mapping import DATA, WORK
from experiments.mapping import INPUTS as DATASETS
from experiments.mapping.mapping_kang import QUERY, REFERENCE, RUN_LABEL, prepared_stores
from experiments.runtime import kang_labels
from experiments.runtime.common import configure_cytearc, write_json
from experiments.runtime.recording_store import StoreProbe, wrap_recording_store

LABEL = kang_labels.LABEL_COLUMN
SAVE_K = 11
THRESHOLD = 0.6


def group_of(key: str) -> str:
    parts = key.split("/")
    if len(parts) >= 3 and parts[1] == "artifacts":
        return f"{parts[0]}/artifacts/{parts[2]}"
    if parts[0] == "artifacts" and len(parts) >= 2:
        return f"artifacts/{parts[1]}"
    return "/".join(parts[:2]) if len(parts) > 1 else key


class KeyLogProbe(StoreProbe):
    """StoreProbe that also keeps (kind, key, bytes) for every transfer; StoreProbe keeps totals
    only."""

    def __init__(self) -> None:
        super().__init__()
        self.transferred_bytes: list[tuple[str, str, int]] = []

    def reset(self) -> None:
        super().reset()
        self.transferred_bytes = []

    def record_transfer(self, kind: str, key: str, nbytes: int) -> None:
        super().record_transfer(kind, key, nbytes)
        self.transferred_bytes.append((kind, key, int(nbytes)))


def stored_bytes(root: Path, prefix: str) -> int:
    total = 0
    for directory, _dirs, files in os.walk(root / prefix):
        for name in files:
            total += (Path(directory) / name).stat().st_size
    return total


def summarise(probe: StoreProbe, root: Path) -> dict:
    by_group: dict[str, dict] = {}
    for kind, key, nbytes in probe.transferred_bytes:
        if kind not in {"get", "get_ranges", "get_partial_values"}:
            continue
        entry = by_group.setdefault(group_of(key), {"requests": 0, "bytesRead": 0, "keys": set()})
        entry["requests"] += 1
        entry["bytesRead"] += int(nbytes)
        entry["keys"].add(key)
    groups = {}
    for group, entry in sorted(by_group.items(), key=lambda kv: -kv[1]["bytesRead"]):
        stored = stored_bytes(root, group)
        groups[group] = {
            "requests": entry["requests"],
            "distinctKeys": len(entry["keys"]),
            "bytesRead": entry["bytesRead"],
            "storedBytes": stored,
            "timesRead": round(entry["bytesRead"] / stored, 2) if stored else None,
        }
    counts_requests = sum((1 for _k, key, _n in probe.transferred_bytes if "/counts" in key))
    return {
        "requests": sum((g["requests"] for g in groups.values())),
        "bytesRead": sum((g["bytesRead"] for g in groups.values())),
        "requestsUnderCountArrays": counts_requests,
        "groups": groups,
    }


def main() -> None:
    configure_cytearc()
    work = WORK / "reads"
    reference_path, query_path, records = prepared_stores(work, DATASETS)
    probe = KeyLogProbe()
    reference_store = wrap_recording_store(LocalStore(str(reference_path)), probe=probe)
    ds_reference = cytearc.DataStore(
        reference_store, default_assay="RNA", nthreads=4, zarr_mode="r+", zarrProfile="fast_local"
    )
    run = ds_reference.pipeline.open(label=RUN_LABEL)
    steps: dict[str, dict] = {}
    probe.reset()
    reference_ref = ds_reference.mapping.build_reference(run["neighbors"])
    steps["buildReference"] = summarise(probe, reference_path)
    probe.reset()
    reference = ds_reference.mapping.load_reference(reference_ref)
    steps["loadReference"] = summarise(probe, reference_path)
    probe.reset()
    ds_query = cytearc.DataStore(str(query_path), default_assay="RNA", nthreads=4)
    selection = ds_query.snapshot_cell_selection(kang_labels.SINGLETS)
    mapping_ref = ds_query.mapping.run(
        reference,
        selection,
        query_assay="RNA",
        save_k=SAVE_K,
        missing_feature_policy="reference_mean",
    )
    steps["runMapping"] = summarise(probe, reference_path)
    selection_cells = int(np.asarray(ds_query.cells.fetch_all(kang_labels.SINGLETS), bool).sum())
    for call in (1, 2):
        probe.reset()
        transferred = ds_query.mapping.load_label_transfer(
            ds_query.mapping.transfer_labels(
                mapping_ref,
                reference=reference,
                reference_labels=LABEL,
                threshold_fraction=THRESHOLD,
            )
        ).labels.fillna("NA")
        steps[f"labelTransferCall{call}"] = summarise(probe, reference_path)
    assigned = float((transferred.to_numpy().astype(str) != "NA").mean())
    write_json(
        "kang_reference_read_requests.json",
        {
            "description": "Reference-store keys requested by each mapping step on the local Kang stores, grouped by artifact, with bytes read and how many times each stored artifact was read.",
            **records,
            "reference": REFERENCE,
            "query": QUERY,
            "queryCells": int(selection_cells),
            "saveK": SAVE_K,
            "threshold": THRESHOLD,
            "assignedFraction": assigned,
            "countArrayGroups": sorted(
                {group_of(k) for k in os.listdir(reference_path / "RNA") if k.startswith("counts")}
            )
            if (reference_path / "RNA").exists()
            else [],
            "steps": steps,
        },
        directory=DATA,
    )
    for name, step in steps.items():
        print(
            f"{name}: {step['requests']} requests, {step['bytesRead'] / 2**20:.1f} MiB, count-array requests {step['requestsUnderCountArrays']}"
        )
        for group, g in list(step["groups"].items())[:6]:
            print(
                f"   {group:45s} {g['bytesRead'] / 2**20:7.2f} MiB  x{g['timesRead']}  ({g['requests']} requests, {g['distinctKeys']} keys)"
            )


if __name__ == "__main__":
    main()
