import argparse
from pathlib import Path
from urllib.parse import urlsplit

from experiments.reuse import HERE
from experiments.runtime.common import scrub_private, write_json_at
from experiments.runtime.locations import RESULTS_PREFIX, experiment_bucket

# The mounts of the embryo stores keep their results with the embryo benchmark runs.
PREFIX = "embryo_rna"


def result_uri(folder: str, tag: str | None) -> str:
    from experiments.runtime.object_io import open_r2_object

    if tag:
        return f"{folder}/mount_{tag}.json"
    store, key = open_r2_object(f"{folder}/mount_")
    names = [
        str(item["path"])
        for page in store.list(prefix=key, chunk_size=100)
        for item in page
        if str(item["path"]).endswith(".json")
    ]
    if len(names) != 1:
        found = [name.rsplit("/", 1)[1] for name in names]
        raise SystemExit(
            f"{len(names)} mount results under {scrub_private(folder)} ({found}); name one with --reference-tag"
        )
    return f"s3://{urlsplit(folder).netloc}/{names[0]}"


def main() -> None:
    from experiments.runtime.object_io import get_json

    parser = argparse.ArgumentParser(description=Path(__file__).stem.replace("_", " "))
    parser.add_argument("--reference-tag")
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    folder = f"{experiment_bucket().uri(RESULTS_PREFIX)}/results/{args.prefix}"
    uri = result_uri(folder, args.reference_tag)
    result = get_json(uri)
    if result.get("status") != "ok":
        raise SystemExit(
            f"{scrub_private(uri)}: status {result.get('status')!r} ({result.get('error')}); not collected"
        )
    document = scrub_private(result)
    write_json_at(HERE / "records" / "embryo_endothelial_reanalysis.json", document)
    totals = document["totals"]
    print(
        f"{document['lineage']['value']} ({document['lineage']['cells']:,} cells), read {totals['sourceReadBytes'] / 1000000000.0:.1f} GB from the source, the user's store holds {totals['userStoreBytesWritten'] / 1000000.0:.0f} MB; source unchanged: {document['source']['unchanged']}"
    )


if __name__ == "__main__":
    main()
