from __future__ import annotations

from pathlib import Path

import polars as pl

from stargaze_ml.gold.ctrader_l2_recorder import (
    AtomicParquetPartWriter,
    DepthBook,
    load_recorded_l2_snapshots,
    recorded_l2_seconds,
)


class Quote:
    def __init__(self, quote_id: int, size: int, *, bid: int | None = None, ask: int | None = None):
        self.id = quote_id
        self.size = size
        self.bid = bid or 0
        self.ask = ask or 0
        self._bid = bid is not None
        self._ask = ask is not None

    def HasField(self, field: str) -> bool:
        return self._bid if field == "bid" else self._ask


def test_depth_book_applies_incremental_updates_and_deletes() -> None:
    book = DepthBook()
    book.apply_new(Quote(1, 200, bid=10_000_000))
    book.apply_new(Quote(2, 300, bid=9_990_000))
    book.apply_new(Quote(3, 100, ask=10_010_000))
    book.apply_new(Quote(4, 400, ask=10_020_000))

    snapshot = book.snapshot()
    assert snapshot is not None
    assert snapshot["best_bid"] == 100.0
    assert snapshot["best_ask"] == 100.1
    assert snapshot["bid_size_top1"] == 2.0
    assert snapshot["ask_size_top1"] == 1.0
    assert abs(snapshot["microprice"] - ((100.1 * 2 + 100.0) / 3)) < 1e-12

    deleted = book.apply_delete(1)
    assert deleted["delete_known"] is True
    assert deleted["bid"] == 100.0
    assert book.snapshot()["best_bid"] == 99.9

    unknown = book.apply_delete(999)
    assert unknown["delete_known"] is False


def test_depth_book_replaces_existing_quote_id() -> None:
    book = DepthBook()
    book.apply_new(Quote(1, 100, bid=10_000_000))
    book.apply_new(Quote(1, 250, bid=10_005_000))
    book.apply_new(Quote(2, 100, ask=10_010_000))
    snapshot = book.snapshot()
    assert snapshot is not None
    assert snapshot["best_bid"] == 100.05
    assert snapshot["bid_levels"] == 1
    assert snapshot["bid_size_top1"] == 2.5


def test_atomic_parquet_part_writer_resumes_without_overwrite(tmp_path: Path) -> None:
    writer = AtomicParquetPartWriter(tmp_path, prefix="depth")
    writer.append({"x": 1})
    first = writer.flush()
    assert first is not None and first.name == "depth_00000001.parquet"

    resumed = AtomicParquetPartWriter(tmp_path, prefix="depth")
    assert resumed.rows_written == 1
    assert resumed.parts_written == 1
    resumed.append({"x": 2})
    second = resumed.flush()
    assert second is not None and second.name == "depth_00000002.parquet"
    assert pl.concat([pl.read_parquet(first), pl.read_parquet(second)])["x"].to_list() == [1, 2]


def test_recorded_snapshots_convert_to_segment_safe_seconds(tmp_path: Path) -> None:
    parts = tmp_path / "snapshot_parts"
    writer = AtomicParquetPartWriter(parts, prefix="snapshot")
    for segment, sequence, ts_ns, bid in (
        (1, 1, 100_000_000, 100.0),
        (1, 2, 1_100_000_000, 101.0),
        (2, 3, 1_200_000_000, 200.0),
        (2, 4, 2_100_000_000, 201.0),
    ):
        writer.append(
            {
                "receive_ns": ts_ns,
                "connection_segment": segment,
                "event_sequence": sequence,
                "best_bid": bid,
                "best_ask": bid + 1.0,
                "bid_size_top1": 2.0,
                "ask_size_top1": 1.0,
                "mid": bid + 0.5,
                "book_wap": bid + 2.0 / 3.0,
            }
        )
    writer.flush()

    snapshots = load_recorded_l2_snapshots(tmp_path)
    assert snapshots.height == 4
    seconds = recorded_l2_seconds(tmp_path, max_quote_age_seconds=0)
    assert seconds.height == 3
    assert seconds["segment_id"].n_unique() == 2
    assert seconds["connection_segment"].to_list() == [1, 2, 2]
