"""Range reads of the variable-gene pass on the fixed and the scaled count layouts, from records/count_requests.json, the
fixed layout's reads over the scaled layout's at each size, the exponent of each count pass's time against cells on
each layout, and the share of decoded bytes a gene-wise pass saves on the paired layout at each planned size, from
records/count_layout_geometry.json, into .build/derived/benchmarks/count_requests.json.

    uv run --locked python -m experiments.benchmarks.count_request_summary
"""

import math
from collections import defaultdict

import numpy as np

from experiments.benchmarks import HERE
from experiments.runtime.common import DERIVED, ROOT, load_json, write_json_at

COUNT_REQUESTS = HERE / "records" / "count_requests.json"
LAYOUT_GEOMETRY = HERE / "records" / "count_layout_geometry.json"
SUMMARY = DERIVED / "benchmarks" / "count_requests.json"
# The pass whose range reads the layouts are compared on, and the sizes and layouts each needs.
READ_PASS = "markHvgs"
SIZES = (5_000_000, 10_000_000)
LAYOUTS = ("fixed", "scaled")
# The count passes whose time against cells is compared between the layouts, with their names in the figure.
PASSES = (
    ("markHvgs", "variable genes"),
    ("runNormalization", "normalization"),
    ("findMarkers", "markers"),
)


def layout_runs() -> list[dict]:
    """The measurement runs of the fixed and the scaled count layouts, each covered by the paired validation that
    both layouts gave the same results."""
    records = load_json(COUNT_REQUESTS)["records"]
    completed = [r["result"] for r in records.values() if r.get("status") == "completed"]
    validations = [r for r in completed if r.get("mode") == "pairedLayoutValidation"]
    if len(validations) != 1 or validations[0]["passed"] is not True:
        raise ValueError("the layouts need one paired validation that passed")
    validated = {row["runId"] for row in validations[0]["measurements"] if row["passed"] is True}
    runs = [result for result in completed if result.get("mode") == "measurement"]
    for result in runs:
        if result["settings"]["runId"] not in validated:
            raise ValueError(f"{result['settings']['runId']} has no passed paired validation")
    return runs


def read_requests() -> dict:
    """Range reads of the variable-gene pass on the fixed and the scaled count layouts at five and ten
    million embryo cells: three runs per layout and size, whose counts must agree, and a paired
    validation that both layouts gave the same results."""
    reads = defaultdict(dict)
    for result in layout_runs():
        settings = result["settings"]
        stage = next(s for s in result["stages"] if s["stage"] == READ_PASS)
        reads[(settings["nRows"], settings["layout"])][settings["replicate"]] = stage["details"][
            "storeOperations"
        ]["rangeGets"]
    rows = []
    for (cells, layout), by_replicate in sorted(reads.items()):
        counts = set(by_replicate.values())
        if sorted(by_replicate) != [1, 2, 3] or len(counts) != 1:
            raise ValueError(
                f"{layout} layout at {cells} cells: three runs with the same range reads expected; got {by_replicate}"
            )
        rows.append({"cells": cells, "layout": layout, "rangeReads": counts.pop()})
    if {(row["cells"], row["layout"]) for row in rows} != {
        (size, kind) for size in SIZES for kind in LAYOUTS
    }:
        raise ValueError("range reads are needed for both layouts at 5 and 10 million cells")
    return {"pass": READ_PASS, "rows": rows, "source": str(COUNT_REQUESTS.relative_to(ROOT))}


def pass_exponents() -> dict:
    """The exponent b in time ∝ cells^b of each count pass between five and ten million embryo cells, on the fixed
    and the scaled layouts: from the mean seconds of the three runs at each size, with the range of b over every
    pair of a smaller and a larger run."""
    seconds = defaultdict(list)
    for result in layout_runs():
        settings = result["settings"]
        for stage in result["stages"]:
            seconds[(stage["stage"], settings["layout"], settings["nRows"])].append(
                stage["seconds"]
            )
    rows = []
    for stage, name in PASSES:
        for layout in LAYOUTS:
            sizes = sorted(cells for s, kind, cells in seconds if (s, kind) == (stage, layout))
            if sizes != list(SIZES):
                raise ValueError(
                    f"{stage} on the {layout} layout needs runs at 5 and 10 million cells"
                )
            small, large = (seconds[(stage, layout, cells)] for cells in sizes)
            if len(small) != 3 or len(large) != 3:
                raise ValueError(f"{stage} on the {layout} layout needs three runs at each size")
            scale = math.log(sizes[1] / sizes[0])
            pairs = [math.log(b / a) / scale for a in small for b in large]
            rows.append(
                {
                    "pass": stage,
                    "name": name,
                    "layout": layout,
                    "exponent": math.log(np.mean(large) / np.mean(small)) / scale,
                    "low": min(pairs),
                    "high": max(pairs),
                    "seconds": {str(cells): seconds[(stage, layout, cells)] for cells in sizes},
                }
            )
    return {
        "cells": list(SIZES),
        "rows": rows,
        "source": str(COUNT_REQUESTS.relative_to(ROOT)),
    }


def fixed_over_scaled(rows: list[dict]) -> dict[str, float]:
    """At each size, the fixed layout's range reads over the scaled layout's, keyed by the size's cells."""
    reads = {(row["cells"], row["layout"]): row["rangeReads"] for row in rows}
    return {
        str(cells): reads[(cells, "fixed")] / reads[(cells, "scaled")]
        for cells in sorted({cells for cells, _ in reads})
    }


def decoded_bytes_cut(geometry: dict) -> dict[str, float]:
    """At each planned size, the share of decoded bytes a full-matrix gene-wise pass saves when it reads the paired
    gene-major copy instead of the cell-major counts: one minus the paired layout's decode amplification over the
    cell-major layout's, keyed by the size's cells."""
    cut = {}
    for row in geometry["rows"]:
        cell_major, paired = (
            row["amplificationCellMajorOnly"],
            row["decodeAmplificationPairedWrite"],
        )
        if cell_major <= 0 or paired <= 0:
            raise ValueError(
                f"the layout at {row['nCells']} cells needs positive decode amplifications"
            )
        cut[str(row["nCells"])] = 1 - paired / cell_major
    return cut


def main() -> None:
    requests = read_requests()
    write_json_at(
        SUMMARY,
        {
            **requests,
            "fixedOverScaledRangeReads": fixed_over_scaled(requests["rows"]),
            "passExponents": pass_exponents(),
            "decodedBytesCut": {
                "source": str(LAYOUT_GEOMETRY.relative_to(ROOT)),
                "cells": decoded_bytes_cut(load_json(LAYOUT_GEOMETRY)),
            },
        },
    )


if __name__ == "__main__":
    main()
