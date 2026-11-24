"""A once-a-second trace of a container's memory, CPU and network while a workflow runs, with the stage running at
each sample, for drawing the container's resource use over time.

Each row holds the seconds since the trace started, the container's memory (the cgroup file the measurement sampler
reads: memory.current, or memory.usage_in_bytes under cgroup v1), the CPU seconds
its processes have used (cgroup cpu.stat usage_usec, or the busy jiffies of /proc/stat where the cgroup file is
missing), the bytes received on its network interfaces other than loopback (/proc/net/dev), and the stage. CPU and
network are cumulative counters: a reader takes differences between rows for cores busy and bytes per second.
"""

import os
import threading
import time
from pathlib import Path

from experiments.runtime.metrics import _relative_cgroup_path, _relative_cgroup_v1_memory_path

CGROUP = Path("/sys/fs/cgroup")
COLUMNS = ["seconds", "memoryBytes", "cpuSeconds", "networkReceivedBytes", "stage"]


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def memory_file() -> Path | None:
    """The container's memory file, found as the measurement sampler finds it: under the cgroup that
    /proc/self/cgroup names (memory.current under cgroup v2, memory.usage_in_bytes under v1), else at the root."""
    text = _read(Path("/proc/self/cgroup")) or ""
    candidates = []
    relative = _relative_cgroup_path(text)
    if relative is not None:
        candidates.append(CGROUP.joinpath(*relative) / "memory.current")
    relative_v1 = _relative_cgroup_v1_memory_path(text)
    if relative_v1 is not None:
        candidates.append(CGROUP.joinpath(*relative_v1) / "memory.usage_in_bytes")
    candidates += [CGROUP / "memory.current", CGROUP / "memory" / "memory.usage_in_bytes"]
    return next((path for path in candidates if _read(path) is not None), None)


def memory_bytes(path: Path | None) -> int | None:
    value = None if path is None else _read(path)
    return int(value) if value is not None and value.strip().isdigit() else None


def cpu_seconds() -> tuple[float | None, str]:
    """The CPU seconds used so far by the container's processes, and where the value came from."""
    stat = _read(CGROUP / "cpu.stat")
    if stat is not None:
        for line in stat.splitlines():
            key, _, value = line.partition(" ")
            if key == "usage_usec" and value.strip().isdigit():
                return int(value) / 1e6, "cgroup cpu.stat usage_usec"
    proc = _read(Path("/proc/stat"))
    if proc is not None:
        fields = proc.splitlines()[0].split()
        if fields and fields[0] == "cpu":
            # Every field but idle and iowait counts as busy.
            busy = sum(int(value) for index, value in enumerate(fields[1:]) if index not in (3, 4))
            return busy / os.sysconf("SC_CLK_TCK"), "/proc/stat busy jiffies"
    return None, "unavailable"


def network_received_bytes() -> int | None:
    text = _read(Path("/proc/net/dev"))
    if text is None:
        return None
    total = 0
    for line in text.splitlines()[2:]:
        name, _, fields = line.partition(":")
        if name.strip() != "lo" and fields.split():
            total += int(fields.split()[0])
    return total


class ResourceTrace:
    """Samples memory, CPU and network once per interval on a background thread until stopped."""

    def __init__(self, intervalSeconds: float = 1.0) -> None:
        if not intervalSeconds > 0:
            raise ValueError(f"the trace interval must be positive, not {intervalSeconds}")
        self.intervalSeconds = intervalSeconds
        self.rows: list[list] = []
        self.stage: str | None = None
        self.cpuSource = "unavailable"
        self.memoryFile = memory_file()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-trace", daemon=True)
        self._origin = 0.0

    def start(self) -> "ResourceTrace":
        self._origin = time.monotonic()
        self._thread.start()
        return self

    def mark(self, stage: str) -> None:
        """Name the stage that the following samples belong to."""
        self.stage = stage

    def _sample(self) -> None:
        cpu, self.cpuSource = cpu_seconds()
        self.rows.append(
            [
                round(time.monotonic() - self._origin, 3),
                memory_bytes(self.memoryFile),
                cpu,
                network_received_bytes(),
                self.stage,
            ]
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.intervalSeconds)

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        self._sample()
        return {
            "columns": COLUMNS,
            "intervalSeconds": self.intervalSeconds,
            "cpuSource": self.cpuSource,
            "memorySource": str(self.memoryFile) if self.memoryFile else "unavailable",
            "cpuCount": os.cpu_count(),
            "rows": self.rows,
        }
