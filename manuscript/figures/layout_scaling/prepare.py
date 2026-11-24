"""Prepare the panel data of the supplementary figure on count-layout scaling from the benchmark records.

Writes one JSON file per panel into this folder, named for the panel's draw function in draw.py:
read_requests.json and pass_exponents.json. Run from the repository root:

  uv run --locked python -B -m manuscript.figures.layout_scaling.prepare
"""

import json

from experiments.benchmarks.count_request_summary import pass_exponents, read_requests
from experiments.runtime.common import ROOT

OUT = ROOT / "manuscript" / "figures" / "layout_scaling"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("read_requests", read_requests()),
        ("pass_exponents", pass_exponents()),
    ):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(values, indent=1, allow_nan=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
