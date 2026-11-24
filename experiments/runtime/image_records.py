"""Describe the image a job ran in, as the job's record states it: base image, lock file and
its hash, and the pinned sources that experiments/images.py installs."""

import hashlib

from experiments.runtime.common import CYTEARC_SOURCE, ROOT

RUNTIME = ROOT / "experiments" / "runtime"
CRAN_SNAPSHOT = "https://packagemanager.posit.co/cran/__linux__/noble/2026-09-15"
R_BASE = "rocker/r-ver:4.6.1"
BPCELLS_REF = "v0.3.1"
PRESTO_SHA = "b5df6ee6097eb62522f2e557aa93ac21bab2d05f"
OBSTORE = "obstore==0.11.1"
SYMPHONY_SHA = "7c5905988734d9cfe6e1e97a658664717c4ba7b7"
CYTEARC_APT = ("git", "build-essential", "libfftw3-dev", "libmetis-dev", "libtbb-dev")
PYTHON_LOCKS = {
    "python_tool_job": "python-tools.txt",
    "mapping_python_job": "python-mapping.txt",
    "census_tool_job": "python-census.txt",
}


def image_record(job: str) -> dict:
    """The image of a job of this kind: cytearc_tool_job, python_zarr_tool_job, a job named in
    PYTHON_LOCKS, or an R job."""
    if job == "cytearc_tool_job":
        lock = (RUNTIME / "python-cytearc.txt").read_bytes()
        return {
            "family": "python",
            "base": "debian_slim python 3.14.4",
            "apt": list(CYTEARC_APT),
            "build": {"HNSWLIB_NO_NATIVE": "1", "noBinary": ["hnswlib"]},
            "cytearcSource": CYTEARC_SOURCE,
            "requirements": "experiments/runtime/python-cytearc.txt",
            "requirementsSha256": hashlib.sha256(lock).hexdigest(),
        }
    if job == "python_zarr_tool_job":
        lock = (RUNTIME / "python-tools.txt").read_bytes()
        return {
            "family": "python",
            "base": "debian_slim python 3.14.4",
            "lock": "experiments/runtime/python-tools.txt",
            "lockSha256": hashlib.sha256(lock).hexdigest(),
            "added": [OBSTORE],
        }
    if job in PYTHON_LOCKS:
        lock_path = RUNTIME / PYTHON_LOCKS[job]
        return {
            "family": "python",
            "base": "debian_slim python 3.14.4",
            "lock": str(lock_path.relative_to(ROOT)),
            "lockSha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        }
    record = {
        "family": "r",
        "base": R_BASE,
        "cranSnapshot": CRAN_SNAPSHOT,
        "BPCells": BPCELLS_REF,
        "presto": PRESTO_SHA,
    }
    if job == "mapping_r_job":
        record["symphony"] = SYMPHONY_SHA
    return record
