import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from experiments.runtime.common import is_in_records, scrub_private


def list_objects(prefix_uri: str) -> list[dict]:
    from experiments.runtime.object_io import open_r2_object

    parsed = urlsplit(prefix_uri)
    prefix = parsed.path.lstrip("/")
    store, _ = open_r2_object(prefix_uri)
    listed = []
    for batch in store.list(prefix=prefix, chunk_size=100):
        for item in batch:
            path = str(item["path"])
            listed.append(
                {"uri": f"s3://{parsed.netloc}/{path}", "path": path, "size": int(item["size"])}
            )
    return listed


def collect(prefix_uri: str, destination: Path) -> int:
    from experiments.runtime.object_io import download_file, get_json

    destination = destination.resolve()
    if is_in_records(destination):
        raise ValueError(
            "Collect new runs into the experiment topic results directory; records contain frozen evidence"
        )
    prefix_uri = prefix_uri.rstrip("/") + "/"
    prefix = urlsplit(prefix_uri).path.lstrip("/")
    count = 0
    for item in list_objects(prefix_uri):
        name = Path(item["path"].removeprefix(prefix))
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("An object path escapes the collection directory")
        if name.suffix not in {".json", ".npz", ".csv", ".gz"}:
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.suffix == ".json":
            target.write_text(json.dumps(scrub_private(get_json(item["uri"])), indent=2) + "\n")
        else:
            download_file(item["uri"], target)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    count = collect(args.prefix, args.destination)
    print(f"Collected {count} result files into {args.destination}")


if __name__ == "__main__":
    main()
