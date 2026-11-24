import argparse
import importlib
import json
from pathlib import Path

from experiments.runtime.common import scrub_private


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    parser.add_argument("function")
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    if not all(part.isidentifier() for part in args.module.split(".")) or args.function.startswith(
        "_"
    ):
        parser.error("Use a module name and a public experiment function")
    operation = getattr(importlib.import_module(f"experiments.{args.module}"), args.function)
    payload = json.loads(args.payload.read_text())
    if not isinstance(payload, dict):
        raise TypeError("The payload must be a JSON object")
    result = operation(payload)
    if isinstance(result, dict) and result.get("status") in {"error", "failed", "partial"}:
        raise RuntimeError(result.get("error") or f"Experiment failed: {result}")
    print(json.dumps(scrub_private(result), default=str))


if __name__ == "__main__":
    main()
