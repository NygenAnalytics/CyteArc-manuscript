import csv
import json
import platform
import re
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
# Summaries that analysis scripts derive from experiment records; make source-data rebuilds them.
DERIVED = ROOT / ".build" / "derived"
CYTEARC_SOURCE = "pypi:cytearc==1.3.0"
GIB = 1024**3
BENCHMARK_GENES = 45_525
BENCHMARK_SIZES = (100_000, 500_000, 1_000_000, 5_000_000, 10_000_000)


PRIVATE_KEYS = frozenset(
    {"modalFunctionCallId", "modalInputId", "callId", "coordinatorCallId", "r2EndpointUrl"}
)
_S3_BUCKET = re.compile(r"s3://[^/\s\"']+")
_R2_ENDPOINT = re.compile(r"https://[^/\s\"']+\.r2\.cloudflarestorage\.com")
_MODAL_ID = re.compile(r"\b(?:fc|fu|ap|ta|in|sb)-[0-9A-Za-z]{20,}\b")


def scrub_private(value: Any) -> Any:
    """``value`` with private locations and Modal identifiers removed (see above)."""
    if isinstance(value, Mapping):
        return {k: scrub_private(v) for k, v in value.items() if k not in PRIVATE_KEYS}
    if isinstance(value, list | tuple):
        return [scrub_private(v) for v in value]
    if isinstance(value, str):
        value = _R2_ENDPOINT.sub("<endpoint>", _S3_BUCKET.sub("s3://<bucket>", value))
        return _MODAL_ID.sub("<modal-id>", value)
    return value


# Payload settings that locate a job's inputs and outputs or describe its launch, rather than set
# the experiment.
LOCATION_SETTINGS = frozenset(
    {
        "provenance",
        "r2EndpointUrl",
        "recordUri",
        "outputPrefix",
        "materializedInputsPrefix",
        "sourceMetadataRecordUri",
        "workDirectory",
    }
)
# When and as which call a job ran: run history, which stays with the raw run record.
RUN_HISTORY = ("callId", "startedAt", "finishedAt")


def without_bucket_locations(value: Any) -> Any:
    """``value`` without the s3:// locations it holds, and without the mappings that held nothing
    else."""
    if isinstance(value, Mapping):
        kept = {}
        for key, item in value.items():
            if isinstance(item, str) and item.startswith("s3://"):
                continue
            reduced = without_bucket_locations(item)
            if isinstance(item, Mapping) and item and not reduced:
                continue
            kept[key] = reduced
        return kept
    if isinstance(value, list | tuple):
        return [without_bucket_locations(item) for item in value]
    return value


def public_run(run: Mapping[str, Any]) -> dict:
    """A job's run record as an experiment's records keep it: without the call ID and the start and
    finish times, without the payload settings and bucket locations of the job's inputs and outputs,
    and with any other mention of a bucket, endpoint or Modal ID masked."""
    kept = {key: value for key, value in run.items() if key not in RUN_HISTORY}
    result = kept.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("settings"), Mapping):
        settings = {k: v for k, v in result["settings"].items() if k not in LOCATION_SETTINGS}
        kept["result"] = {**result, "settings": settings}
    return scrub_private(without_bucket_locations(kept))


def is_in_records(path: Path) -> bool:
    """Whether ``path`` is inside an experiment's records, experiments/<experiment>/records/,
    which hold the frozen evidence the paper reports."""
    path = Path(path).resolve()
    if not path.is_relative_to(EXPERIMENTS):
        return False
    parts = path.relative_to(EXPERIMENTS).parts
    return len(parts) >= 2 and parts[1] == "records"


def provenance(**extra: Any) -> dict:
    import cytearc
    import numpy
    import zarr

    return {
        "cytearcVersion": cytearc.__version__,
        "configuredCytearcSource": CYTEARC_SOURCE,
        "numpyVersion": numpy.__version__,
        "zarrVersion": zarr.__version__,
        "python": platform.python_version(),
        **extra,
    }


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return str(value) if isinstance(value, Path) else value


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(name: str, payload: Mapping, *, directory: Path, subdir: str | None = None) -> Path:
    folder = directory / subdir if subdir else directory
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    return path


def write_csv(
    name: str,
    rows: Sequence[Mapping],
    *,
    directory: Path,
    fieldnames: Iterable[str] | None = None,
    subdir: str | None = None,
) -> Path:
    folder = directory / subdir if subdir else directory
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if fieldnames is None:
        fieldnames = dict.fromkeys(str(key) for row in rows for key in row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})
    return path


def write_json_at(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` to ``path`` inside the repository and print the repository path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)
    return path


def write_csv_at(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> Path:
    """Write the ``fieldnames`` columns of ``rows`` to ``path`` inside the repository, leaving
    out other keys, and print the repository path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = dict.fromkeys(str(key) for row in rows for key in row)
    fieldnames = list(fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)
    return path


class Stopwatch:
    def __enter__(self):
        self.started = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.started


def dir_bytes(path: Path) -> int:
    return (
        path.stat().st_size
        if path.is_file()
        else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    )


def configure_cytearc(level: str = "WARNING") -> None:
    import cytearc

    cytearc.configure_output(level=level, progress=False)


def catalogue_store(name: str, *, destination: Path) -> Path:
    from cytearc.datasets import connect

    path = destination / name
    if not (path / "data.zarr").exists():
        connect("cytearc_docs").download_dataset(name, destination=str(destination), zarr=True)
    if not (path / "data.zarr").exists():
        raise FileNotFoundError(f"Dataset {name} did not produce {path}")
    return path


def fresh_copy(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination
