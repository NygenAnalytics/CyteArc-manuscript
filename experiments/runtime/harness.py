"""Measurement harness for external tools on Modal.

The harness runs one workflow as a child process and stays small itself. It samples the
resident memory of the process tree and of the container and the bytes of the files in the
run's work directory, where downloads and the workflow's own files go, follows the step events that
the workflow appends to a JSON-lines file, and keeps a progress record on R2 that is
rewritten at every step and every minute. A workflow killed for memory therefore still
leaves the step it was running, the steps it completed and the peak so far.

The child raises its own OOM score, so the kernel kills the workflow rather than the
harness when the container runs out of memory. A container that is itself restarted
(preemption, or a kill of the whole container) is handled by the restart guard, which tells
the two apart by the memory of the step that was running.

Nothing here imports CyteArc: the images of the external tools are independent of the
CyteArc working tree.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

GIB = 1024**3
# The work directory is walked this often; a walk lists every file, so it runs less often than the memory samples.
DISK_SAMPLE_SECONDS = 5.0
MEMORY_KILL_SHARE = 0.85
MEMORY_MARKERS = ("cannot allocate", "memoryerror", "std::bad_alloc", "out of memory", "killed")


def running_step_peak(record: dict) -> int:
    step = (record.get("steps") or {}).get(record.get("currentStep") or "") or {}
    return int(step.get("peakCgroupBytes") or record.get("peakCgroupBytes") or 0)


def _current_call_id():
    return os.environ.get("CYTEARC_CALL_ID")


def _status_fields(pid: int) -> tuple[int | None, int]:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return (None, 0)
    parent, rss = (None, 0)
    for line in text.splitlines():
        if line.startswith("PPid:"):
            parent = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    return (parent, rss)


def process_tree_rss(root: int) -> int:
    parents: dict[int, int | None] = {}
    rss: dict[int, int] = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            pid = int(name)
            parents[pid], rss[pid] = _status_fields(pid)
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        if parent is not None:
            children.setdefault(parent, []).append(pid)
    total, stack, seen = (0, [root], set())
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total


# What reading a number from a kernel file can raise when the file is missing, empty or not a number. Named, not
# written inline, so the module still parses on the Python 3.12 of the Census image, which the runner imports it into.
UNREADABLE_NUMBER = (OSError, ValueError, IndexError)


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().split()[0])
    except UNREADABLE_NUMBER:
        return None


def cgroup_anon_bytes() -> int | None:
    for path, key in (
        ("/sys/fs/cgroup/memory.stat", "anon"),
        ("/sys/fs/cgroup/memory/memory.stat", "total_rss"),
    ):
        try:
            for line in Path(path).read_text().splitlines():
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
        except UNREADABLE_NUMBER:
            continue
    return None


def cgroup_memory_current() -> int | None:
    value = _read_int("/sys/fs/cgroup/memory.current")
    if value is None:
        value = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    return value


def directory_bytes(root: Path) -> int:
    """Bytes of the files under root, skipping any file a running workflow deletes during the walk."""
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except FileNotFoundError:
                continue
    return total


class Sampler(threading.Thread):
    """Samples memory every ``interval`` seconds and attributes peaks to steps. Given a disk_root, it also records the
    bytes of the files under it every DISK_SAMPLE_SECONDS, and their peak."""

    def __init__(self, interval: float = 0.5, disk_root: Path | None = None) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.root = os.getpid()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.step: str | None = None
        self.peak_rss = 0
        self.peak_cgroup = 0
        self.peak_anon = 0
        self.step_peaks: dict[str, dict[str, int]] = {}
        self.series: list[tuple[float, int, int, int]] = []
        self._last_series = 0.0
        self.disk_root = disk_root
        self.peak_disk = 0
        self.disk_series: list[tuple[float, int]] = []
        self._last_disk = 0.0

    def set_step(self, name: str | None) -> None:
        with self.lock:
            self.step = name

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def sample(self) -> None:
        now = time.time()
        rss = process_tree_rss(self.root)
        cgroup = cgroup_memory_current() or 0
        anon = cgroup_anon_bytes() or 0
        disk = None
        if self.disk_root is not None and now - self._last_disk >= DISK_SAMPLE_SECONDS:
            disk = directory_bytes(self.disk_root)
            self._last_disk = now
        with self.lock:
            if disk is not None:
                self.peak_disk = max(self.peak_disk, disk)
                self.disk_series.append((round(now, 1), disk))
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_cgroup = max(self.peak_cgroup, cgroup)
            self.peak_anon = max(self.peak_anon, anon)
            if self.step is not None:
                peaks = self.step_peaks.setdefault(
                    self.step, {"peakRssBytes": 0, "peakCgroupBytes": 0, "peakAnonBytes": 0}
                )
                peaks["peakRssBytes"] = max(peaks["peakRssBytes"], rss)
                peaks["peakCgroupBytes"] = max(peaks["peakCgroupBytes"], cgroup)
                peaks["peakAnonBytes"] = max(peaks["peakAnonBytes"], anon)
            if now - self._last_series >= 5.0:
                self.series.append((round(now, 1), rss, cgroup, anon))
                self._last_series = now

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)
        self.sample()


class R2:
    """Minimal S3 client for the R2 bucket; credentials come from the Modal secret."""

    def __init__(self, endpoint: str) -> None:
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )

    @staticmethod
    def split(uri: str) -> tuple[str, str]:
        parts = urlsplit(uri)
        if parts.scheme != "s3":
            raise ValueError(f"expected an s3:// URI, got {uri}")
        return (parts.netloc, parts.path.lstrip("/"))

    def exists(self, uri: str) -> bool:
        bucket, key = self.split(uri)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except self.client.exceptions.ClientError:
            return False

    def get_json(self, uri: str) -> dict:
        bucket, key = self.split(uri)
        return json.loads(self.client.get_object(Bucket=bucket, Key=key)["Body"].read())

    def put_json(self, uri: str, value: dict) -> None:
        bucket, key = self.split(uri)
        body = json.dumps(value, indent=1, default=str).encode()
        self.client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

    def _transfer_config(self):
        from boto3.s3.transfer import TransferConfig

        return TransferConfig(multipart_chunksize=64 * 1024**2, max_concurrency=16)

    def download(self, uri: str, path: Path) -> int:
        bucket, key = self.split(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            self.client.download_file(bucket, key, str(partial), Config=self._transfer_config())
            os.replace(partial, path)
        return path.stat().st_size

    def upload(self, path: Path, uri: str) -> None:
        bucket, key = self.split(uri)
        self.client.upload_file(str(path), bucket, key, Config=self._transfer_config())

    def download_dir(self, uri: str, path: Path, files: list[str] | None = None) -> int:
        """Download the objects under ``uri`` into ``path``, or with ``files`` only those paths
        relative to ``uri``, every one of which must exist."""
        from concurrent.futures import ThreadPoolExecutor

        if files is not None and not files:
            raise ValueError("files names no file to download")
        bucket, prefix = self.split(uri.rstrip("/") + "/")
        keys = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            keys += [(item["Key"], item["Size"]) for item in page.get("Contents", [])]
        if not keys:
            raise FileNotFoundError(f"no objects under {prefix}")
        if files is not None:
            wanted = {prefix + name for name in files}
            missing = sorted(wanted - {key for key, _ in keys})
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of the requested files are not under {prefix}, first {missing[0]}"
                )
            keys = [item for item in keys if item[0] in wanted]

        def fetch(item: tuple[str, int]) -> None:
            target = path / item[0][len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket, item[0], str(target))

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(fetch, keys))
        return sum((size for _, size in keys))

    def upload_dir(self, path: Path, uri: str) -> int:
        from concurrent.futures import ThreadPoolExecutor

        files = [item for item in path.rglob("*") if item.is_file()]
        base = uri.rstrip("/")

        def send(item: Path) -> None:
            self.upload(item, f"{base}/{item.relative_to(path).as_posix()}")

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(send, files))
        return sum((item.stat().st_size for item in files))


def _raise_oom_score() -> None:
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass


def _classify(returncode: int, stderr_tail: str) -> str:
    if returncode == 0:
        return "ok"
    if returncode in (-9, 137):
        return "killed"
    text = stderr_tail.lower()
    if any((marker in text for marker in MEMORY_MARKERS)):
        return "outOfMemory"
    return "error"


class Run:
    """One measured attempt: download, run the workflow, record everything on R2."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.r2 = R2(payload["r2EndpointUrl"])
        self.uri = payload["outputUri"]
        self.work = Path(payload.get("workDir", "/tmp/run"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.events = self.work / "events.jsonl"
        self.summary_path = self.work / "summary.json"
        sample_seconds = float(payload.get("memorySampleSeconds", 0.5))
        if not 0 < sample_seconds < float("inf"):
            raise ValueError("memorySampleSeconds must be positive and finite")
        self.sampler = Sampler(interval=sample_seconds, disk_root=self.work)
        self.steps: dict[str, dict] = {}
        self.order: list[str] = []
        self.current: str | None = None
        self.started_at = time.time()
        self.status = "attempting"
        self.call_id = _current_call_id()
        self.extra: dict = {}
        self._events_offset = 0

    def base(self) -> dict:
        p = self.payload
        return {
            "tool": p["tool"],
            "dataset": p["dataset"],
            "datasetUri": p["datasetUri"],
            "replicate": p.get("replicate"),
            "container": {
                "cpu": p["containerCpu"],
                "memoryMb": p["containerMemoryMb"],
                "workers": p["workers"],
                "preemptible": p.get("preemptible"),
                "region": os.environ.get("MODAL_REGION"),
            },
            "image": p.get("image"),
            "restarts": p.get("restarts", 0),
            "interruptions": p.get("interruptions", []),
            "callId": self.call_id,
            "startedAt": self.started_at,
        }

    def record(self) -> dict:
        with self.sampler.lock:
            step_peaks = {k: dict(v) for k, v in self.sampler.step_peaks.items()}
            peak_rss, peak_cgroup, peak_anon = (
                self.sampler.peak_rss,
                self.sampler.peak_cgroup,
                self.sampler.peak_anon,
            )
            peak_disk = self.sampler.peak_disk
        steps = {}
        for name in self.order:
            entry = dict(self.steps[name])
            entry.update(step_peaks.get(name, {}))
            for kind in ("Rss", "Cgroup", "Anon"):
                observed = [
                    entry[key]
                    for key in (f"peak{kind}Bytes", f"preStep{kind}Bytes", f"postStep{kind}Bytes")
                    if entry.get(key) is not None
                ]
                if observed:
                    entry[f"peak{kind}Bytes"] = max(observed)
            steps[name] = entry
        peak_rss = max([peak_rss, *[v.get("peakRssBytes", 0) for v in steps.values()]])
        peak_cgroup = max([peak_cgroup, *[v.get("peakCgroupBytes", 0) for v in steps.values()]])
        peak_anon = max([peak_anon, *[v.get("peakAnonBytes", 0) for v in steps.values()]])
        return {
            **self.base(),
            "status": self.status,
            "currentStep": self.current,
            "completedSteps": [n for n in self.order if "seconds" in self.steps[n]],
            "steps": steps,
            "stepOrder": self.order,
            "peakRssBytes": peak_rss,
            "peakCgroupBytes": peak_cgroup,
            "peakAnonBytes": peak_anon,
            "peakWorkDirectoryBytes": peak_disk,
            "memorySampleSeconds": self.sampler.interval,
            "elapsedSeconds": time.time() - self.started_at,
            **self.extra,
        }

    def publish(self) -> None:
        self.r2.put_json(self.uri, self.record())

    def restart_guard(self) -> dict | None:
        if not self.r2.exists(self.uri):
            return None
        previous = self.r2.get_json(self.uri)
        if self.call_id is None or previous.get("callId") != self.call_id:
            return None
        if previous.get("status") != "attempting":
            return {
                "status": previous.get("status"),
                "failedStep": previous.get("failedStep"),
                "reason": "restart guard: record already closed",
            }
        restarts = int(previous.get("restarts") or 0) + 1
        started = bool(previous.get("currentStep") or previous.get("completedSteps"))
        limit = int(previous.get("container", {}).get("memoryMb") or 0) * 1024 * 1024
        peak = int(previous.get("peakCgroupBytes") or 0)
        memory = started and running_step_peak(previous) >= MEMORY_KILL_SHARE * limit
        interruption = {
            "step": previous.get("currentStep"),
            "peakCgroupBytes": peak,
            "elapsedSeconds": previous.get("elapsedSeconds"),
        }
        if not memory and restarts <= int(self.payload.get("maxRestarts", 1)):
            self.payload["restarts"] = restarts
            self.payload["interruptions"] = [*previous.get("interruptions", []), interruption]
            return None
        status = "killed" if memory else "interrupted"
        previous.update(
            {
                "status": status,
                "restarts": restarts,
                "failedStep": previous.get("currentStep"),
                "error": f"container ended during {previous.get('currentStep')!r}; last recorded peak {peak / GIB:.1f} GiB of {limit / GIB:.0f} GiB",
            }
        )
        self.r2.put_json(self.uri, previous)
        return {"status": status, "failedStep": previous["failedStep"], "reason": "restart guard"}

    def _read_events(self) -> bool:
        changed = False
        if self.events.exists():
            with self.events.open() as handle:
                handle.seek(self._events_offset)
                for line in handle:
                    if not line.endswith("\n"):
                        break
                    self._events_offset += len(line.encode())
                    event = json.loads(line)
                    name = event["step"]
                    if event["event"] == "start":
                        self.order.append(name)
                        self.steps[name] = {
                            "start": event["t"],
                            **{
                                key: value
                                for key, value in event.items()
                                if key not in ("step", "event", "t")
                            },
                        }
                        self.current = name
                        self.sampler.set_step(name)
                    else:
                        entry = self.steps.setdefault(name, {"start": event["t"]})
                        entry["end"] = event["t"]
                        entry["seconds"] = event["t"] - entry["start"]
                        for key, value in event.items():
                            if key not in ("step", "event", "t"):
                                entry[key] = value
                        self.current = None
                        self.sampler.set_step(None)
                    changed = True
        return changed

    def _follow_events(self, stop: threading.Event) -> None:
        limit = int(self.payload["containerMemoryMb"]) * 1024 * 1024
        last_publish, last_peak = (0.0, 0)
        while not stop.is_set():
            changed = self._read_events()
            with self.sampler.lock:
                peak = max(self.sampler.peak_cgroup, self.sampler.peak_rss)
            climbing = peak >= 0.5 * limit and peak - last_peak >= 0.05 * limit
            if changed or climbing or time.time() - last_publish > 60:
                self.publish()
                last_publish, last_peak = (time.time(), peak)
            stop.wait(1.0)

    def execute(self, argv: list[str], env: dict | None = None) -> dict:
        guarded = self.restart_guard()
        if guarded is not None:
            return guarded
        self.publish()
        mode = self.payload.get("inputMode", "file")
        local = (
            self.work / "data" / Path(urlsplit(self.payload["datasetUri"]).path.rstrip("/")).name
        )
        started = time.time()
        if mode == "remote":
            local, size = (self.payload["datasetUri"], 0)
        elif mode == "directory":
            size = self.r2.download_dir(
                self.payload["datasetUri"], local, self.payload.get("datasetFiles")
            )
        else:
            size = self.r2.download(self.payload["datasetUri"], local)
        self.extra["download"] = {
            "seconds": time.time() - started,
            "bytes": size,
            "inputMode": mode,
        }
        self.sampler.start()
        stop = threading.Event()
        follower = threading.Thread(target=self._follow_events, args=(stop,), daemon=True)
        follower.start()
        stderr_path = self.work / "stderr.txt"
        run_env = {**os.environ, **(env or {})}
        workflow_started = time.time()
        with stderr_path.open("w") as stderr:
            process = subprocess.Popen(
                [*argv, str(local), str(self.events), str(self.summary_path), str(self.work)],
                stdout=stderr,
                stderr=subprocess.STDOUT,
                env=run_env,
                preexec_fn=_raise_oom_score,
            )
            returncode = process.wait()
        self.extra["workflowSeconds"] = time.time() - workflow_started
        stop.set()
        follower.join(timeout=10)
        self._read_events()
        self.sampler.stop()
        tail = stderr_path.read_text(errors="replace")[-6000:] if stderr_path.exists() else ""
        self.status = _classify(returncode, tail)
        limit = int(self.payload["containerMemoryMb"]) * 1024 * 1024
        with self.sampler.lock:
            step_peak = (self.sampler.step_peaks.get(self.current or "") or {}).get(
                "peakCgroupBytes", self.sampler.peak_cgroup
            )
        if self.status == "error" and step_peak >= 0.95 * limit:
            self.status = "outOfMemory"
        if self.summary_path.exists():
            try:
                self.extra.update(json.loads(self.summary_path.read_text()))
            except json.JSONDecodeError:
                pass
        self.extra.update(
            {
                "returnCode": returncode,
                "failedStep": self.current if self.status != "ok" else None,
                "log": tail if self.status != "ok" else tail[-1500:],
                "memorySeries": self.sampler.series,
                "diskSeries": self.sampler.disk_series,
                "workDirectoryBytesAfter": directory_bytes(self.work),
                "finishedAt": time.time(),
            }
        )
        self.publish()
        return {
            "status": self.status,
            "failedStep": self.extra["failedStep"],
            "workflowSeconds": self.extra["workflowSeconds"],
        }


def _is_remote(uri: str) -> bool:
    return uri.startswith("s3://")


class Objects:
    """JSON records and files on R2 (s3:// URIs; credentials from the Modal secret), or on
    disk for the local test."""

    def __init__(self, endpoint: str | None) -> None:
        self.endpoint = endpoint
        self._r2 = None

    def _remote(self):
        if self._r2 is None:
            self._r2 = R2(self.endpoint)
        return self._r2

    def exists(self, uri: str) -> bool:
        return self._remote().exists(uri) if _is_remote(uri) else Path(uri).exists()

    def get_json(self, uri: str) -> dict:
        return (
            self._remote().get_json(uri) if _is_remote(uri) else json.loads(Path(uri).read_text())
        )

    def put_json(self, uri: str, value: dict) -> None:
        if _is_remote(uri):
            self._remote().put_json(uri, value)
            return
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        Path(uri).write_text(json.dumps(value, indent=1, default=str) + "\n")

    def download(self, uri: str, path: Path) -> int:
        if _is_remote(uri):
            return self._remote().download(uri, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(uri, path)
        return path.stat().st_size

    def upload(self, path: Path, uri: str) -> None:
        if _is_remote(uri):
            self._remote().upload(path, uri)
            return
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, uri)

    def prefix_bytes(self, uri: str) -> dict:
        if not _is_remote(uri):
            files = [p for p in Path(uri).rglob("*") if p.is_file()]
            return {"bytes": sum((p.stat().st_size for p in files)), "objects": len(files)}
        r2 = self._remote()
        bucket, key = r2.split(uri.rstrip("/") + "/")
        total = count = 0
        for page in r2.client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key):
            for item in page.get("Contents", []):
                total += int(item["Size"])
                count += 1
        return {"bytes": total, "objects": count}
