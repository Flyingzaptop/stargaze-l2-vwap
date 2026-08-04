from __future__ import annotations

import asyncio
from pathlib import Path
import time

import pyarrow as pa

from .metrics import metrics
from .record_log import append_parquet_block, copy_record_log, initialize_record_log
from .records import SCHEMA, normalize_record


class RecordStreamWriter:
    """One durable append-only file for one logical market-data stream."""

    def __init__(self, path: Path, flush_rows: int, flush_seconds: float) -> None:
        self.path = path
        recovered = initialize_record_log(path)
        self.flush_rows = flush_rows
        self.flush_seconds = flush_seconds
        self.buffer: list[dict] = []
        self.last_flush = time.monotonic()
        self.lock = asyncio.Lock()
        self.rows = recovered.rows
        self.frames = recovered.frames
        self.valid_bytes = recovered.valid_bytes
        self.key = path.stem
        self._stream = path.open("ab")
        metrics.ensure_writer(self.key, self.path)
        metrics.update_writer(self.key, self.path, self.rows, 0)

    async def write_many(self, records: list[dict]) -> None:
        if not records:
            return
        async with self.lock:
            self.buffer.extend(normalize_record(record) for record in records)
            if len(self.buffer) >= self.flush_rows:
                self._flush_locked()

    async def flush_if_due(self) -> None:
        async with self.lock:
            if self.buffer and time.monotonic() - self.last_flush >= self.flush_seconds:
                self._flush_locked()

    async def close(self) -> None:
        async with self.lock:
            if self.buffer:
                self._flush_locked()
            if not self._stream.closed:
                self._stream.close()

    async def copy_snapshot(self, target: Path) -> tuple[int, int, int]:
        async with self.lock:
            if self.buffer:
                self._flush_locked()
            valid_bytes = self.valid_bytes
            rows = self.rows
            frames = self.frames
        info = await asyncio.to_thread(
            copy_record_log,
            self.path,
            target,
            valid_bytes=valid_bytes,
            expected_rows=rows,
            expected_frames=frames,
        )
        return info.rows, info.frames, target.stat().st_size

    def _flush_locked(self) -> None:
        table = pa.Table.from_pylist(self.buffer, schema=SCHEMA)
        written = append_parquet_block(self._stream, table)
        self.rows += len(self.buffer)
        self.frames += 1
        self.valid_bytes += written
        self.buffer.clear()
        self.last_flush = time.monotonic()
        metrics.update_writer(self.key, self.path, self.rows, 0)


class WriterRegistry:
    def __init__(self, output_dir: Path, flush_rows: int, flush_seconds: float) -> None:
        self.output_dir = output_dir
        self.flush_rows = flush_rows
        self.flush_seconds = flush_seconds
        self._writers: dict[tuple[str, str, str, str], RecordStreamWriter] = {}

    def get(self, exchange: str, market: str, symbol: str, channel: str) -> RecordStreamWriter:
        key = (exchange, market, symbol, channel)
        if key not in self._writers:
            safe_symbol = symbol.replace("/", "").replace("-", "_").replace(":", "_")
            path = self.output_dir / f"{exchange}_{market}_{safe_symbol}_{channel}.mrec"
            self._writers[key] = RecordStreamWriter(path, self.flush_rows, self.flush_seconds)
        return self._writers[key]

    async def periodic_flush(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(max(0.5, min(self.flush_seconds, 5.0)))
            for writer in list(self._writers.values()):
                await writer.flush_if_due()

    async def close_all(self) -> None:
        results = await asyncio.gather(
            *(writer.close() for writer in list(self._writers.values())),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(f"failed to close {len(errors)} record-log writer(s)") from errors[0]

    async def snapshot_to(self, target_directory: Path) -> list[dict]:
        target_directory.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []
        active_paths: set[Path] = set()
        for writer in list(self._writers.values()):
            target = target_directory / writer.path.name
            rows, frames, size = await writer.copy_snapshot(target)
            active_paths.add(writer.path.resolve())
            entries.append(
                {"name": writer.path.name, "bytes": size, "rows": rows, "frames": frames}
            )

        for source in sorted(self.output_dir.glob("*.mrec")):
            if source.resolve() in active_paths:
                continue
            target = target_directory / source.name
            info = await asyncio.to_thread(copy_record_log, source, target)
            entries.append(
                {
                    "name": source.name,
                    "bytes": target.stat().st_size,
                    "rows": info.rows,
                    "frames": info.frames,
                }
            )
        return sorted(entries, key=lambda item: item["name"])
