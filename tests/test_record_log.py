from __future__ import annotations

import asyncio

from market_collector.record_log import (
    FRAME_HEADER,
    FRAME_MAGIC,
    copy_record_log,
    inspect_record_log,
)
from market_collector.writer import WriterRegistry


def _record(event_id: int) -> dict:
    return {
        "exchange": "test",
        "market": "spot",
        "symbol": "BTCUSD",
        "channel": "depth",
        "event_type": "level",
        "event_id": event_id,
        "row_idx": 0,
        "local_ts_ns": event_id,
        "side": "bid",
        "price": 100.0,
        "quantity": 1.0,
    }


def test_incomplete_tail_is_removed_before_append_resumes(tmp_path) -> None:
    async def first_run() -> None:
        registry = WriterRegistry(tmp_path, flush_rows=1, flush_seconds=60)
        writer = registry.get("test", "spot", "BTCUSD", "depth")
        await writer.write_many([_record(1)])
        await registry.close_all()

    asyncio.run(first_run())
    path = next(tmp_path.glob("*.mrec"))
    clean_size = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(FRAME_HEADER.pack(FRAME_MAGIC, 1000, 10, 123))
        stream.write(b"partial")
    assert path.stat().st_size > clean_size

    registry = WriterRegistry(tmp_path, flush_rows=10, flush_seconds=60)
    registry.get("test", "spot", "BTCUSD", "depth")
    assert path.stat().st_size == clean_size
    assert inspect_record_log(path, verify_payloads=True).rows == 1
    asyncio.run(registry.close_all())


def test_multiple_streams_create_exactly_one_file_each(tmp_path) -> None:
    async def scenario() -> None:
        registry = WriterRegistry(tmp_path, flush_rows=1, flush_seconds=60)
        first = registry.get("one", "spot", "BTCUSD", "depth")
        second = registry.get("two", "spot", "BTCUSD", "trades")
        await first.write_many([_record(1)])
        await second.write_many([_record(2)])
        await registry.close_all()

    asyncio.run(scenario())
    assert len(list(tmp_path.glob("*.mrec"))) == 2
    assert not list(tmp_path.glob("*.parquet*"))


def test_snapshot_prefix_stays_valid_while_source_keeps_growing(tmp_path) -> None:
    async def scenario() -> None:
        registry = WriterRegistry(tmp_path / "data", flush_rows=1, flush_seconds=60)
        writer = registry.get("test", "spot", "BTCUSD", "depth")
        await writer.write_many([_record(1)])
        prefix_bytes = writer.valid_bytes
        prefix_rows = writer.rows
        prefix_frames = writer.frames
        await writer.write_many([_record(2)])

        packed = tmp_path / "pack" / writer.path.name
        info = copy_record_log(
            writer.path,
            packed,
            valid_bytes=prefix_bytes,
            expected_rows=prefix_rows,
            expected_frames=prefix_frames,
        )
        assert info.rows == 1
        assert inspect_record_log(writer.path).rows == 2
        assert inspect_record_log(packed, verify_payloads=True).rows == 1
        await registry.close_all()

    asyncio.run(scenario())
