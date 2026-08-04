from __future__ import annotations

import asyncio
import json

from market_collector.record_log import inspect_record_log, iter_record_log_tables
from market_collector.snapshot import SnapshotManager
from market_collector.writer import WriterRegistry


def _record(event_id: int) -> dict:
    return {
        "exchange": "test",
        "market": "spot",
        "symbol": "BTCUSD",
        "channel": "trades",
        "event_type": "trade",
        "event_id": event_id,
        "row_idx": 0,
        "local_ts_ns": event_id,
        "price": 100.0 + event_id,
        "quantity": 2.0,
    }


def test_pack_copies_committed_prefix_without_rotating_active_file(tmp_path) -> None:
    async def scenario() -> None:
        data_dir = tmp_path / "data"
        packs_dir = tmp_path / "packs"
        registry = WriterRegistry(data_dir, flush_rows=100, flush_seconds=60)
        writer = registry.get("test", "spot", "BTCUSD", "trades")
        await writer.write_many([_record(1)])

        result = await SnapshotManager(data_dir, packs_dir).create_pack(registry)
        assert result.file_count == 1
        assert result.total_rows == 1
        packed = next((result.path / "data").glob("*.mrec"))
        assert inspect_record_log(packed, verify_payloads=True).rows == 1
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["format"] == "market-recorder-pack-v2"

        await writer.write_many([_record(2)])
        await registry.close_all()
        active = sorted(data_dir.glob("*.mrec"))
        assert len(active) == 1
        assert inspect_record_log(active[0], verify_payloads=True).rows == 2
        assert inspect_record_log(packed, verify_payloads=True).rows == 1

    asyncio.run(scenario())


def test_writer_reopens_and_appends_to_the_same_file(tmp_path) -> None:
    async def run_once(event_id: int) -> None:
        registry = WriterRegistry(tmp_path, flush_rows=1, flush_seconds=60)
        writer = registry.get("test", "spot", "BTCUSD", "trades")
        await writer.write_many([_record(event_id)])
        await registry.close_all()

    asyncio.run(run_once(1))
    asyncio.run(run_once(2))

    files = list(tmp_path.glob("*.mrec"))
    assert len(files) == 1
    tables = list(iter_record_log_tables(files[0]))
    assert [item for table in tables for item in table["event_id"].to_pylist()] == [1, 2]
