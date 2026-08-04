from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from stargaze_ml.gold.l2_seconds import (
    aggregate_l2_seconds,
    build_l2_second_feature_matrix,
    reconstruct_l2_seconds,
    reconstruct_l2_snapshots,
)


def _add_snapshot(
    rows: list[dict[str, object]],
    ts_ns: int,
    *,
    bid: float,
    ask: float,
    bid_size: float,
    ask_size: float,
) -> None:
    for level in range(3):
        rows.append(
            {
                "timestamp": ts_ns,
                "quote_id": len(rows) + 1,
                "bid": bid - level,
                "ask": 0.0,
                "size": bid_size if level == 0 else 1.0,
                "type": "new",
            }
        )
        rows.append(
            {
                "timestamp": ts_ns,
                "quote_id": len(rows) + 1,
                "bid": 0.0,
                "ask": ask + level,
                "size": ask_size if level == 0 else 1.0,
                "type": "new",
            }
        )


def _write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows).with_columns(pl.col("size").cast(pl.Int64)).write_parquet(path)


def test_exact_snapshot_reconstruction_and_second_ohlc(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    _add_snapshot(rows, 100_000_000, bid=100.0, ask=102.0, bid_size=2, ask_size=1)
    _add_snapshot(rows, 900_000_000, bid=101.0, ask=103.0, bid_size=1, ask_size=3)
    _add_snapshot(rows, 1_100_000_000, bid=102.0, ask=104.0, bid_size=2, ask_size=2)
    # This crossed refresh must not survive the conservative packet filter.
    _add_snapshot(rows, 2_100_000_000, bid=105.0, ask=104.0, bid_size=2, ask_size=2)
    raw = tmp_path / "raw.parquet"
    _write_raw(raw, rows)

    snapshots = reconstruct_l2_snapshots(raw, tick_size=1.0, timestamp_unit="ns")
    seconds = aggregate_l2_seconds(snapshots, max_quote_age_seconds=0)

    assert snapshots.height == 3
    assert seconds["bar_start_ns"].to_list() == [0, 1_000_000_000]
    first = seconds.row(0, named=True)
    assert first["open"] == 101.0
    assert first["high"] == 102.0
    assert first["low"] == 101.0
    assert first["close"] == 102.0
    assert first["first_bid"] == 100.0
    assert first["first_ask"] == 102.0
    assert first["last_bid"] == 101.0
    assert first["last_ask"] == 103.0
    # book_wap is the last snapshot's top-of-book microprice, not trade VWAP.
    assert np.isclose(first["book_wap"], (103.0 * 1.0 + 101.0 * 3.0) / 4.0)
    assert first["snapshot_count"] == 2


def test_daily_and_rolling_book_vwap_are_weighted_and_strictly_causal(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    _add_snapshot(rows, 0, bid=100.0, ask=102.0, bid_size=1, ask_size=3)
    _add_snapshot(rows, 1_000_000_000, bid=102.0, ask=104.0, bid_size=2, ask_size=2)
    _add_snapshot(rows, 2_000_000_000, bid=104.0, ask=106.0, bid_size=3, ask_size=1)
    raw = tmp_path / "raw.parquet"
    _write_raw(raw, rows)

    baseline = reconstruct_l2_seconds(
        raw,
        tick_size=1.0,
        timestamp_unit="ns",
        max_quote_age_seconds=0,
    )
    wap = baseline["book_wap"].to_numpy()
    weight = baseline["liquidity_weight"].to_numpy()
    expected = np.cumsum(wap * weight) / np.cumsum(weight)
    np.testing.assert_allclose(baseline["daily_book_vwap"].to_numpy(), expected)
    np.testing.assert_allclose(baseline["book_vwap_60s"].to_numpy(), expected)
    np.testing.assert_allclose(baseline["book_vwap_300s"].to_numpy(), expected)
    expected_bid = np.cumsum(
        baseline["last_bid"].to_numpy() * baseline["bid_size_top1"].to_numpy()
    ) / np.cumsum(baseline["bid_size_top1"].to_numpy())
    expected_ask = np.cumsum(
        baseline["last_ask"].to_numpy() * baseline["ask_size_top1"].to_numpy()
    ) / np.cumsum(baseline["ask_size_top1"].to_numpy())
    np.testing.assert_allclose(baseline["bid_vwap_60s"].to_numpy(), expected_bid)
    np.testing.assert_allclose(baseline["ask_vwap_60s"].to_numpy(), expected_ask)

    changed_rows = rows[:-6]
    _add_snapshot(
        changed_rows,
        2_000_000_000,
        bid=10_000.0,
        ask=10_002.0,
        bid_size=3,
        ask_size=1,
    )
    changed_raw = tmp_path / "changed.parquet"
    _write_raw(changed_raw, changed_rows)
    changed = reconstruct_l2_seconds(
        changed_raw,
        tick_size=1.0,
        timestamp_unit="ns",
        max_quote_age_seconds=0,
    )
    for column in (
        "daily_book_vwap",
        "book_vwap_60s",
        "book_vwap_300s",
        "bid_vwap_60s",
        "ask_vwap_60s",
    ):
        np.testing.assert_array_equal(
            baseline[column].to_numpy()[:2],
            changed[column].to_numpy()[:2],
        )


def test_short_gaps_are_carried_with_zero_weight_and_larger_gap_resets_segment(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    _add_snapshot(rows, 0, bid=100.0, ask=102.0, bid_size=2, ask_size=2)
    _add_snapshot(rows, 3_000_000_000, bid=103.0, ask=105.0, bid_size=2, ask_size=2)
    _add_snapshot(rows, 7_000_000_000, bid=107.0, ask=109.0, bid_size=2, ask_size=2)
    raw = tmp_path / "raw.parquet"
    _write_raw(raw, rows)

    seconds = reconstruct_l2_seconds(
        raw,
        tick_size=1.0,
        timestamp_unit="ns",
        max_quote_age_seconds=2,
    )

    assert seconds["bar_start_ns"].to_list() == [
        0,
        1_000_000_000,
        2_000_000_000,
        3_000_000_000,
        4_000_000_000,
        5_000_000_000,
        7_000_000_000,
    ]
    assert seconds["observed"].to_list() == [True, False, False, True, False, False, True]
    assert seconds["segment_id"].to_list() == [0, 0, 0, 0, 0, 0, 1]
    assert seconds["quote_age_ms"].to_list() == [0.0, 1000.0, 2000.0, 0.0, 1000.0, 2000.0, 0.0]
    assert seconds["liquidity_weight"].to_list() == [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0]
    assert seconds["snapshot_count"].to_list() == [1, 0, 0, 1, 0, 0, 1]
    assert seconds["bid_vwap_60s"].to_list()[:3] == [100.0, 100.0, 100.0]
    assert seconds["ask_vwap_60s"].to_list()[:3] == [102.0, 102.0, 102.0]
    assert seconds["bid_vwap_60s"].to_list()[3:6] == [101.5, 101.5, 101.5]
    assert seconds["ask_vwap_60s"].to_list()[3:6] == [103.5, 103.5, 103.5]
    for index in (1, 2, 4, 5):
        row = seconds.row(index, named=True)
        assert row["open"] == row["high"] == row["low"] == row["close"]
        assert row["first_bid"] == row["last_bid"]
        assert row["first_ask"] == row["last_ask"]

    # All columns use simple Arrow-compatible scalar dtypes and round-trip.
    saved = tmp_path / "seconds.parquet"
    seconds.write_parquet(saved)
    restored = pl.read_parquet(saved)
    assert restored.schema == seconds.schema
    assert restored.equals(seconds)


def test_feature_matrix_uses_only_causal_ohlc_bbo_and_book_vwap(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for second in range(4):
        _add_snapshot(
            rows,
            second * 1_000_000_000,
            bid=100.0 + second,
            ask=102.0 + second,
            bid_size=2,
            ask_size=2,
        )
    raw = tmp_path / "raw.parquet"
    _write_raw(raw, rows)
    seconds = reconstruct_l2_seconds(
        raw,
        tick_size=1.0,
        timestamp_unit="ns",
        max_quote_age_seconds=0,
    )

    matrix = build_l2_second_feature_matrix(seconds, tick_size=1.0)

    assert matrix.x.shape == (4, 16)
    assert not matrix.valid_feature[0]
    assert matrix.valid_feature[1:].all()
    assert all("observed" not in name and "age" not in name for name in matrix.feature_names)
    np.testing.assert_allclose(matrix.x[1:, matrix.feature_names.index("close_delta_1s_ticks")], 1.0)
    np.testing.assert_allclose(
        matrix.x[:, matrix.feature_names.index("bid_vwap_60s")],
        [100.0, 100.5, 101.0, 101.5],
    )
    np.testing.assert_allclose(
        matrix.x[:, matrix.feature_names.index("ask_vwap_60s")],
        [102.0, 102.5, 103.0, 103.5],
    )
    np.testing.assert_allclose(
        matrix.x[1:, matrix.feature_names.index("bid_vwap_60s_minus_bid_ticks")],
        [-0.5, -1.0, -1.5],
    )
    np.testing.assert_allclose(
        matrix.x[1:, matrix.feature_names.index("ask_vwap_60s_minus_ask_ticks")],
        [-0.5, -1.0, -1.5],
    )
