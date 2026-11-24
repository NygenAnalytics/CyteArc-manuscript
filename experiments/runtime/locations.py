"""Where experiments read and keep data.

Each location comes from an environment variable that experiments/README.md lists. A missing
variable stops the script with an error that names it.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# The CyteBase release on Hugging Face that the catalogue and reuse experiments read.
CYTEBASE_RELEASE = "hf://buckets/Nygen/cytebase-2610"
# Key prefixes in the experiment bucket: runs write records under RESULTS_PREFIX/results/ and
# stores under RESULTS_PREFIX/stores/, and the prepared datasets are under DATASETS_PREFIX.
RESULTS_PREFIX = "experiments"
DATASETS_PREFIX = "experiments/datasets"
# The Modal runner passes these from this machine's environment into each job.
BUCKET_VARIABLES = ("R2_ENDPOINT_URL", "R2_BUCKET")


def required(name: str, meaning: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to {meaning}; experiments/README.md lists the variables")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentBucket:
    """The S3-compatible bucket that holds the experiments' datasets, stores and run records."""

    endpoint: str
    name: str

    def uri(self, key: str) -> str:
        """The s3:// URI of a key in the bucket."""
        return f"s3://{self.name}/{key.strip('/')}"


def experiment_bucket() -> ExperimentBucket:
    return ExperimentBucket(
        endpoint=required("R2_ENDPOINT_URL", "the experiment bucket's S3 endpoint"),
        name=required("R2_BUCKET", "the experiment bucket's name"),
    )


def downloads_folder(experiment: str) -> Path:
    """The folder where an experiment's scripts download public datasets."""
    return Path(required("CYTEARC_DOWNLOADS", "the folder for downloaded datasets")) / experiment


def runs_root() -> Path:
    """The folder for work stores, results and job source archives of runs on this machine."""
    return Path(required("CYTEARC_RUNS", "the folder for work stores and run results"))


def results_folder(experiment: str) -> Path:
    """The folder where an experiment's scripts write results: CYTEARC_RESULTS when a caller sets
    it for one run, such as runtime.method_checks, otherwise the experiment's folder in
    CYTEARC_RUNS."""
    chosen = os.environ.get("CYTEARC_RESULTS")
    return Path(chosen) if chosen else runs_root() / experiment / "results"


def experiment_folder(experiment: str, name: str) -> Path:
    """INPUTS, WORK, DATA or RAW of an experiment package, resolved when a script first uses it."""
    if name == "INPUTS":
        return downloads_folder(experiment)
    if name == "WORK":
        return runs_root() / experiment / "workspace"
    if name == "DATA":
        return results_folder(experiment)
    if name == "RAW":
        return results_folder(experiment) / "raw"
    raise AttributeError(f"module 'experiments.{experiment}' has no attribute {name!r}")
