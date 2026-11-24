"""Count the Zarr store operations of a profiled stage.

This wrapper is instrumentation only. It is not a product request limiter.
"""

import threading
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import contextmanager
from typing import Any

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype

_READ_KINDS = frozenset({"get", "get_ranges", "get_partial_values"})


def _is_count_chunk(key: str) -> bool:
    parts = key.split("/")
    return bool({"counts", "countsT"}.intersection(parts)) and parts[-1] not in {
        "zarr.json",
        ".zarray",
        ".zattrs",
        ".zgroup",
    }


def _nbytes(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except TypeError:
        return 0


def _requested_bytes(byte_range: object | None) -> int | None:
    if byte_range is None:
        return None
    start = getattr(byte_range, "start", None)
    end = getattr(byte_range, "end", None)
    if start is not None and end is not None:
        return max(0, int(end) - int(start))
    suffix = getattr(byte_range, "suffix", None)
    if suffix is not None:
        return max(0, int(suffix))
    return None


class StoreProbe:
    """Thread-safe operation totals shared by a store and any clone Zarr makes of it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0
        self._reset_totals()

    def _reset_totals(self) -> None:
        self._count_by_kind: dict[str, int] = {}
        self._requested_total = 0
        self._transferred_total = 0
        self._read_requested_total = 0
        self._read_transferred_total = 0
        self._write_transferred_total = 0
        self._count_read_requests = 0
        self._count_read_bytes = 0
        self._max_in_flight = self._in_flight

    def reset(self) -> None:
        with self._lock:
            self._reset_totals()

    def enter(
        self,
        kind: str,
        key: str,
        byte_range: object | None = None,
        requestedBytes: int | None = None,
    ) -> None:
        requested = int(requestedBytes or 0)
        with self._lock:
            self._count_by_kind[kind] = self._count_by_kind.get(kind, 0) + 1
            self._requested_total += requested
            if kind in _READ_KINDS:
                self._read_requested_total += requested
                if _is_count_chunk(key):
                    self._count_read_requests += 1
            self._in_flight += 1
            self._max_in_flight = max(self._max_in_flight, self._in_flight)

    def record_transfer(self, kind: str, key: str, nbytes: int) -> None:
        transferred = int(nbytes)
        with self._lock:
            self._transferred_total += transferred
            if kind in _READ_KINDS:
                self._read_transferred_total += transferred
                if _is_count_chunk(key):
                    self._count_read_bytes += transferred
            elif kind == "set":
                self._write_transferred_total += transferred

    def leave(self, kind: str) -> None:
        with self._lock:
            self._in_flight -= 1

    def to_json(self) -> dict[str, int]:
        with self._lock:
            counts = dict(self._count_by_kind)
            return {
                "gets": counts.get("get", 0),
                "sets": counts.get("set", 0),
                "deletes": counts.get("delete", 0),
                "deleteDirs": counts.get("delete_dir", 0),
                "sizeQueries": counts.get("getsize", 0),
                "rangeGets": counts.get("get_ranges", 0),
                "partialGets": counts.get("get_partial_values", 0),
                "requestedBytes": self._requested_total,
                "transferredBytes": self._transferred_total,
                "readRequestedBytes": max(self._read_requested_total, self._read_transferred_total),
                "readTransferredBytes": self._read_transferred_total,
                "writeTransferredBytes": self._write_transferred_total,
                "countReadRequests": self._count_read_requests,
                "countReadBytes": self._count_read_bytes,
                "maxInFlight": self._max_in_flight,
            }


class RecordingStoreWrapper(Store):
    """Wrap any Zarr Store and count its operations, overlap, and bytes.

    Every Store method the wrapped store implements itself is forwarded, so profiled
    code takes the same storage path as unprofiled code.
    """

    def __init__(self, inner: Store, *, probe: StoreProbe | None = None):
        super().__init__(read_only=inner.read_only)
        self._inner = inner
        self.probe = probe or StoreProbe()
        self._is_open = inner._is_open

    def with_read_only(self, read_only: bool = False) -> "RecordingStoreWrapper":
        return RecordingStoreWrapper(self._inner.with_read_only(read_only), probe=self.probe)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RecordingStoreWrapper) and self._inner == other._inner

    @property
    def supports_writes(self) -> bool:
        return self._inner.supports_writes

    @property
    def supports_deletes(self) -> bool:
        return self._inner.supports_deletes

    @property
    def supports_listing(self) -> bool:
        return self._inner.supports_listing

    async def _open(self) -> None:
        await self._inner._open()
        await super()._open()

    def close(self) -> None:
        self._inner.close()
        super().close()

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        self.probe.enter("get", key, byte_range, requestedBytes=_requested_bytes(byte_range))
        try:
            result = await self._inner.get(key, prototype, byte_range)
            self.probe.record_transfer("get", key, _nbytes(result))
            return result
        finally:
            self.probe.leave("get")

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]
    ) -> list[Buffer | None]:
        pairs = list(key_ranges)
        label = next(
            (key for key, _ in pairs if _is_count_chunk(key)), pairs[0][0] if pairs else ""
        )
        requested = sum((_requested_bytes(item[1]) or 0 for item in pairs))
        self.probe.enter("get_partial_values", label, pairs, requestedBytes=requested)
        try:
            result = await self._inner.get_partial_values(prototype, pairs)
            for (key, _), item in zip(pairs, result, strict=True):
                self.probe.record_transfer("get_partial_values", key, _nbytes(item))
            return result
        finally:
            self.probe.leave("get_partial_values")

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int = 10,
        max_gap_bytes: int = 1 << 20,
        max_coalesced_bytes: int = 16 << 20,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        requested = sum((_requested_bytes(item) or 0 for item in byte_ranges))
        self.probe.enter("get_ranges", key, tuple(byte_ranges), requestedBytes=requested)
        transferred = 0
        try:
            async for group in self._inner.get_ranges(
                key,
                byte_ranges,
                prototype=prototype,
                max_concurrency=max_concurrency,
                max_gap_bytes=max_gap_bytes,
                max_coalesced_bytes=max_coalesced_bytes,
            ):
                transferred += sum((_nbytes(item[1]) for item in group))
                yield group
        finally:
            self.probe.record_transfer("get_ranges", key, transferred)
            self.probe.leave("get_ranges")

    async def exists(self, key: str) -> bool:
        return await self._inner.exists(key)

    async def getsize(self, key: str) -> int:
        self.probe.enter("getsize", key)
        try:
            return await self._inner.getsize(key)
        finally:
            self.probe.leave("getsize")

    async def getsize_prefix(self, prefix: str) -> int:
        self.probe.enter("getsize", prefix)
        try:
            return await self._inner.getsize_prefix(prefix)
        finally:
            self.probe.leave("getsize")

    async def set(self, key: str, value: Buffer) -> None:
        self.probe.enter("set", key, None, requestedBytes=_nbytes(value))
        try:
            await self._inner.set(key, value)
            self.probe.record_transfer("set", key, _nbytes(value))
        finally:
            self.probe.leave("set")

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self.probe.enter("set", key, None, requestedBytes=_nbytes(value))
        try:
            await self._inner.set_if_not_exists(key, value)
            self.probe.record_transfer("set", key, _nbytes(value))
        finally:
            self.probe.leave("set")

    async def delete(self, key: str) -> None:
        self.probe.enter("delete", key)
        try:
            await self._inner.delete(key)
        finally:
            self.probe.leave("delete")

    async def delete_dir(self, prefix: str) -> None:
        self.probe.enter("delete_dir", prefix)
        try:
            await self._inner.delete_dir(prefix)
        finally:
            self.probe.leave("delete_dir")

    async def clear(self) -> None:
        self.probe.enter("delete_dir", "")
        try:
            await self._inner.clear()
        finally:
            self.probe.leave("delete_dir")

    def list(self) -> AsyncIterator[str]:
        return self._inner.list()

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        return self._inner.list_prefix(prefix)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        return self._inner.list_dir(prefix)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_recording_store(store: Store, *, probe: StoreProbe | None = None) -> RecordingStoreWrapper:
    if isinstance(store, RecordingStoreWrapper):
        return store
    return RecordingStoreWrapper(store, probe=probe)


def wrap_store_probe(storeUri: str, options: dict[str, Any] | None, storeProbe: Any | None) -> Any:
    if storeProbe is None:
        return storeUri
    from cytearc.storage.stores import make_store
    from zarr.storage import LocalStore

    from experiments.runtime.recording_store import wrap_recording_store

    resolved = make_store(storeUri, storage_options=options)
    if isinstance(resolved, str):
        resolved = LocalStore(resolved.removeprefix("file://"))
    return wrap_recording_store(resolved, probe=storeProbe)


class NetworkProbe(StoreProbe):
    def __init__(self):
        super().__init__()
        self.http = {"requests": 0, "gets": 0, "rangedGets": 0, "responseBytes": 0}
        self.http_status = {}

    def reset(self):
        super().reset()
        with self._lock:
            self.http = {"requests": 0, "gets": 0, "rangedGets": 0, "responseBytes": 0}
            self.http_status = {}

    def request(self, request):
        with self._lock:
            self.http["requests"] += 1
            self.http["gets"] += int(request.method == "GET")
            self.http["rangedGets"] += int(request.method == "GET" and "range" in request.headers)

    def received(self, size):
        with self._lock:
            self.http["responseBytes"] += size

    def response(self, status):
        with self._lock:
            key = str(status)
            self.http_status[key] = self.http_status.get(key, 0) + 1


@contextmanager
def probe_remote_reads(probe: NetworkProbe):
    import httpx2
    from cytearc.storage import stores

    orig = stores.make_store
    original_send = httpx2.Client._send_single_request

    class CountedStream(httpx2.SyncByteStream):
        def __init__(self, stream):
            self.stream = stream

        def __iter__(self):
            for chunk in self.stream:
                probe.received(len(chunk))
                yield chunk

        def close(self):
            self.stream.close()

    def send(client, request):
        probe.request(request)
        response = original_send(client, request)
        probe.response(response.status_code)
        response.stream = CountedStream(response.stream)
        return response

    def wrapped(location, *, storage_options=None, read_only=False):
        st = orig(location, storage_options=storage_options, read_only=read_only)
        if isinstance(location, str) and location.startswith("hf://"):
            return RecordingStoreWrapper(st, probe=probe)
        return st

    stores.make_store = wrapped
    httpx2.Client._send_single_request = send
    try:
        yield
    finally:
        stores.make_store = orig
        httpx2.Client._send_single_request = original_send


def probe_summary(probe: StoreProbe) -> dict:
    d = probe.to_json()
    if d["countReadBytes"] and not probe.http["requests"]:
        raise RuntimeError("Remote count reads were observed without HTTP transport measurements")
    reqs = sum(int(d.get(k, 0)) for k in ("gets", "rangeGets", "partialGets", "sizeQueries"))
    return {
        "readBytes": int(d.get("readTransferredBytes", 0)),
        "readMB": round(int(d.get("readTransferredBytes", 0)) / 1e6, 3),
        "requests": reqs,
        "raw": d,
        "http": dict(probe.http),
        "httpClient": "httpx2",
        "httpResponseStatusCounts": dict(probe.http_status),
        "counterDefinition": "Zarr calls and returned bytes are separate from physical HTTP requests and encoded response-body bytes",
    }
