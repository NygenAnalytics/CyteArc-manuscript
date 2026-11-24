import json

import numpy as np
from experiments.runtime.common import ROOT, load_json

CATALOGUE = ROOT / "experiments/catalogue/records"
REUSE = ROOT / "experiments/reuse/records"

OUTPUT = ROOT / "manuscript" / "figures" / "architecture"
FIELDS = {
    "storage": (
        "dotMean",
        "dotShare",
        "embryoEdges",
        "embryoGroup",
        "embryoNodes",
        "embryoUmap",
        "group",
        "heatmap",
        "pca",
    ),
    "composite": ("brainUmap", "heatmap", "heatmapGroup"),
}


def main() -> None:
    with np.load(REUSE / "kang_reference_thumbnails.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(CATALOGUE / "atlas_embedding_thumbnails.npz", allow_pickle=False) as archive:
        arrays.update({key: archive[key] for key in archive.files})
    maximum = arrays["heatmap"].max(axis=0)
    arrays["heatmap"] = arrays["heatmap"] / np.where(maximum > 0, maximum, 1)
    metadata = load_json(REUSE / "kang_reference_thumbnails.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for panel, fields in FIELDS.items():
        np.savez_compressed(
            OUTPUT / f"{panel}.npz",
            **{name: arrays[name] for name in fields},
            metadata=np.asarray(
                json.dumps(
                    {
                        "experiment": metadata,
                        "heatmap_units": "log1p counts scaled by each gene maximum",
                    }
                )
            ),
        )


if __name__ == "__main__":
    main()
