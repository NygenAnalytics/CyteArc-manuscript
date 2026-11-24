"""Summarise the embryo analysis stores into .build/derived/benchmarks/analysis_store_sizes.json.

For each store size, the gene-major copy's bytes as a fraction of the cell-major counts, and the
analysis artifacts' bytes as a fraction of both count matrices.
"""

import json

from experiments.benchmarks import HERE
from experiments.runtime.common import DERIVED, ROOT, write_json

RECORD = HERE / "records" / "analysis_store_sizes.json"
OUT = DERIVED / "benchmarks"


def size_ratios(store: dict) -> dict:
    counts = store["countsBytes"] + store["countsTBytes"]
    return {
        "nRows": store["nRows"],
        "countsTOverCounts": store["countsTBytes"] / store["countsBytes"],
        "analysisOverCounts": store["analysisBytes"] / counts,
    }


def main() -> None:
    stores = json.loads(RECORD.read_text())["perSize"]
    summary = {
        "source": str(RECORD.relative_to(ROOT)),
        "perSize": {size: size_ratios(store) for size, store in stores.items()},
    }
    path = write_json("analysis_store_sizes.json", summary, directory=OUT)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
