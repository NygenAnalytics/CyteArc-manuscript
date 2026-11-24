def _rss_bytes() -> int:
    import os

    from cytearc.utils.process import read_process_tree_rss_bytes

    return int(read_process_tree_rss_bytes(os.getpid()) or 0)


class _Steps:
    """Per-step wall time and process-tree RSS, tolerant to failure.

    ``on_change`` is called before and after every step so that a container
    killed for memory still leaves a record of the step that was running.
    """

    def __init__(self, on_change=None) -> None:
        self.records: dict[str, dict] = {}
        self.order: list[str] = []
        self.current: str | None = None
        self.on_change = on_change

    def _notify(self) -> None:
        if self.on_change is None:
            return
        self.on_change()

    def run(self, name: str, fn):
        import time

        started = time.perf_counter()
        self.order.append(name)
        self.current = name
        self.records[name] = {"rssBeforeBytes": _rss_bytes()}
        self._notify()
        value = fn()
        self.records[name].update(
            {"seconds": time.perf_counter() - started, "rssAfterBytes": _rss_bytes()}
        )
        self.current = None
        self._notify()
        return value
