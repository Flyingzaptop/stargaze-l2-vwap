from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from market_collector.compaction import compact_parquet_segments
from market_collector.records import SCHEMA, normalize_record


def _write(path: Path, event_ids: list[int]) -> None:
    rows = [
        normalize_record(
            {
                "exchange": "test",
                "market": "spot",
                "symbol": "BTCUSD",
                "channel": "trades",
                "event_type": "trade",
                "event_id": event_id,
                "row_idx": 0,
                "local_ts_ns": event_id,
                "price": 100.0 + event_id,
                "quantity": 1.0,
            }
        )
        for event_id in event_ids
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def test_compaction_merges_segments_without_losing_or_reordering_rows(tmp_path: Path) -> None:
    data = tmp_path / "data"
    recovery = tmp_path / "recovery"
    data.mkdir()
    _write(data / "test_spot_BTCUSD_trades.parquet", [1, 2])
    _write(data / "test_spot_BTCUSD_trades_20260714_120000.parquet", [3])
    _write(data / "test_spot_BTCUSD_trades_20260714_120100.parquet", [4, 5])

    result = compact_parquet_segments(data, recovery)

    files = list(data.glob("*.parquet"))
    assert [path.name for path in files] == ["test_spot_BTCUSD_trades.parquet"]
    table = pq.read_table(files[0], columns=["event_id"])
    assert table.column("event_id").to_pylist() == [1, 2, 3, 4, 5]
    assert result.groups_compacted == 1
    assert result.source_files == 3
    assert result.total_rows == 5
    assert not list(recovery.iterdir())


def test_compaction_recovers_after_sources_were_moved_before_install(tmp_path: Path) -> None:
    data = tmp_path / "data"
    recovery = tmp_path / "recovery"
    transaction = recovery / "test_spot_BTCUSD_trades.compaction"
    sources_backup = transaction / "sources"
    data.mkdir()
    sources_backup.mkdir(parents=True)
    first = data / "test_spot_BTCUSD_trades.parquet"
    second = data / "test_spot_BTCUSD_trades_20260714_120000.parquet"
    _write(first, [1])
    _write(second, [2])
    _write(transaction / "merged.parquet", [1, 2])
    first.replace(sources_backup / first.name)
    manifest = {
        "format": "market-recorder-compaction-v1",
        "target": "test_spot_BTCUSD_trades.parquet",
        "sources": [first.name, second.name],
        "rows": 2,
    }
    (transaction / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = compact_parquet_segments(data, recovery)

    target = data / "test_spot_BTCUSD_trades.parquet"
    assert pq.read_table(target, columns=["event_id"]).column(0).to_pylist() == [1, 2]
    assert not second.exists()
    assert result.recovered_transactions == 1
    assert not list(recovery.iterdir())


def test_compaction_handles_many_rotated_segments_exactly_once(tmp_path: Path) -> None:
    data = tmp_path / "data"
    recovery = tmp_path / "recovery"
    data.mkdir()
    _write(data / "test_spot_BTCUSD_trades.parquet", [0])
    for index in range(1, 201):
        _write(
            data / f"test_spot_BTCUSD_trades_20260714_{index:06d}.parquet",
            [index],
        )

    first = compact_parquet_segments(data, recovery, batch_size=7)
    second = compact_parquet_segments(data, recovery, batch_size=7)

    output = data / "test_spot_BTCUSD_trades.parquet"
    assert pq.read_table(output, columns=["event_id"]).column(0).to_pylist() == list(range(201))
    assert first.source_files == 201
    assert first.total_rows == 201
    assert second.groups_compacted == 0
    assert second.output_files == 1
