from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time


@dataclass
class WriterMetric:
    key: str
    path: str
    rows: int = 0
    buffered_rows: int = 0
    bytes: int = 0
    last_write_ts: float | None = None
    recent: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=240))

    def rows_per_sec(self, window_seconds: float = 60.0) -> float:
        now = time.time()
        points = [(ts, rows) for ts, rows in self.recent if now - ts <= window_seconds]
        if len(points) < 2:
            return 0.0
        dt = points[-1][0] - points[0][0]
        if dt <= 0:
            return 0.0
        return (points[-1][1] - points[0][1]) / dt


@dataclass
class ConnectionMetric:
    key: str
    status: str = "starting"
    last_message_ts: float | None = None
    last_error: str | None = None
    reconnects: int = 0


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._writers: dict[str, WriterMetric] = {}
        self._connections: dict[str, ConnectionMetric] = {}
        self.started_ts = time.time()

    def ensure_writer(self, key: str, path: Path) -> None:
        with self._lock:
            self._writers.setdefault(key, WriterMetric(key=key, path=str(path)))

    def update_writer(self, key: str, path: Path, rows: int, buffered_rows: int) -> None:
        now = time.time()
        size = path.stat().st_size if path.exists() else 0
        with self._lock:
            metric = self._writers.setdefault(key, WriterMetric(key=key, path=str(path)))
            metric.path = str(path)
            metric.rows = rows
            metric.buffered_rows = buffered_rows
            metric.bytes = size
            metric.last_write_ts = now
            metric.recent.append((now, rows))

    def set_connection(self, key: str, status: str, error: str | None = None, message_seen: bool = False) -> None:
        now = time.time()
        with self._lock:
            metric = self._connections.setdefault(key, ConnectionMetric(key=key))
            if status == "reconnecting":
                metric.reconnects += 1
            metric.status = status
            if error:
                metric.last_error = error
            if message_seen:
                metric.last_message_ts = now

    def snapshot(self) -> dict:
        with self._lock:
            writers = []
            for metric in self._writers.values():
                writers.append(
                    {
                        "key": metric.key,
                        "path": metric.path,
                        "rows": metric.rows,
                        "buffered_rows": metric.buffered_rows,
                        "bytes": metric.bytes,
                        "last_write_ts": metric.last_write_ts,
                        "rows_per_sec_60s": metric.rows_per_sec(60.0),
                    }
                )
            connections = [metric.__dict__.copy() for metric in self._connections.values()]
            return {
                "uptime_seconds": time.time() - self.started_ts,
                "writers": sorted(writers, key=lambda item: item["key"]),
                "connections": sorted(connections, key=lambda item: item["key"]),
            }


metrics = MetricsStore()
