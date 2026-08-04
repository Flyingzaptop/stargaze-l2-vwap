from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


SECOND_NS = 1_000_000_000
DAY_NS = 86_400 * SECOND_NS


@dataclass(frozen=True)
class L2SecondFeatureMatrix:
    """Causal candle/BBO/book-VWAP inputs aligned to a regular second grid.

    ``book_wap`` and its daily/rolling aggregates are quote-book proxies.  The
    cTrader archive used here contains no executions, so none of these fields
    represents traded-volume VWAP.
    """

    ts_ns: np.ndarray
    segment_id: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    book_wap: np.ndarray
    daily_book_vwap: np.ndarray
    book_vwap_60s: np.ndarray
    book_vwap_300s: np.ndarray
    bid_vwap_60s: np.ndarray
    ask_vwap_60s: np.ndarray
    x: np.ndarray
    feature_names: tuple[str, ...]
    valid_feature: np.ndarray


def _timestamp_ns_expression(dtype: pl.DataType, *, timestamp_unit: str) -> pl.Expr:
    if isinstance(dtype, pl.Datetime):
        return pl.col("timestamp").dt.epoch("ns")
    multiplier = {"ns": 1, "us": 1_000, "ms": 1_000_000}.get(timestamp_unit)
    if multiplier is None:
        raise ValueError("timestamp_unit must be one of: ns, us, ms")
    return pl.col("timestamp").cast(pl.Int64) * multiplier


def reconstruct_l2_snapshots(
    raw_path: Path,
    *,
    tick_size: float = 0.01,
    min_levels_per_side: int = 3,
    max_new_quotes_per_timestamp: int = 20,
    max_spread_ticks: float = 500.0,
    timestamp_unit: Literal["ns", "us", "ms"] = "ms",
) -> pl.DataFrame:
    """Reconstruct conservative exact-timestamp full-refresh L2 snapshots.

    Only positive ``new`` rows are used.  One-sided, crossed, over-wide and
    unusually large refresh packets are rejected using the same contract as
    :func:`stargaze_ml.gold.l2_angle.reconstruct_l2_bars`.
    """

    path = Path(raw_path).expanduser().resolve(strict=True)
    schema = pl.scan_parquet(path).collect_schema()
    required = {"timestamp", "quote_id", "bid", "ask", "size", "type"}
    missing = sorted(required - set(schema.names()))
    if missing:
        raise ValueError(f"raw L2 parquet is missing columns: {', '.join(missing)}")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    if min_levels_per_side <= 0 or max_new_quotes_per_timestamp <= 0:
        raise ValueError("packet size limits must be positive")
    if max_spread_ticks <= 0:
        raise ValueError("max_spread_ticks must be positive")

    source = (
        pl.scan_parquet(path)
        .filter((pl.col("type").str.to_lowercase() == "new") & (pl.col("size") > 0))
        .with_columns(
            _timestamp_ns_expression(schema["timestamp"], timestamp_unit=timestamp_unit)
            .cast(pl.Int64)
            .alias("snapshot_ts_ns")
        )
    )
    bid_mask = pl.col("bid") > 0
    ask_mask = pl.col("ask") > 0
    bid_price = pl.col("bid").filter(bid_mask)
    ask_price = pl.col("ask").filter(ask_mask)
    bid_size = pl.col("size").cast(pl.Float64).filter(bid_mask)
    ask_size = pl.col("size").cast(pl.Float64).filter(ask_mask)

    snapshots = (
        source.group_by("snapshot_ts_ns", maintain_order=True)
        .agg(
            bid_price.max().alias("best_bid"),
            ask_price.min().alias("best_ask"),
            bid_size.sort_by(bid_price).last().alias("bid_size_top1"),
            ask_size.sort_by(ask_price).first().alias("ask_size_top1"),
            bid_mask.sum().alias("bid_levels"),
            ask_mask.sum().alias("ask_levels"),
            pl.len().alias("new_quote_count"),
        )
        .with_columns(
            ((pl.col("best_ask") - pl.col("best_bid")) / float(tick_size)).alias(
                "spread_ticks"
            )
        )
        .filter(
            (pl.col("bid_levels") >= int(min_levels_per_side))
            & (pl.col("ask_levels") >= int(min_levels_per_side))
            & (pl.col("new_quote_count") <= int(max_new_quotes_per_timestamp))
            & (pl.col("spread_ticks") > 0)
            & (pl.col("spread_ticks") <= float(max_spread_ticks))
        )
        .with_columns(
            ((pl.col("best_ask") + pl.col("best_bid")) * 0.5).alias("mid"),
            (
                (
                    pl.col("best_ask") * pl.col("bid_size_top1")
                    + pl.col("best_bid") * pl.col("ask_size_top1")
                )
                / (pl.col("bid_size_top1") + pl.col("ask_size_top1"))
            ).alias("microprice"),
        )
        .with_columns(pl.col("microprice").alias("book_wap"))
        .sort("snapshot_ts_ns")
        .collect(engine="streaming")
    )
    if snapshots.is_empty():
        raise ValueError("no valid L2 snapshots survived reconstruction")
    return snapshots


def _weighted_causal_average(
    values: np.ndarray,
    weights: np.ndarray,
    group_id: np.ndarray,
    *,
    window: int | None = None,
) -> np.ndarray:
    """Weighted causal mean, reset by group and optionally bounded by rows."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    group_id = np.asarray(group_id)
    n = len(values)
    if weights.shape != (n,) or group_id.shape != (n,):
        raise ValueError("values, weights and group_id must be aligned")
    if window is not None and window <= 0:
        raise ValueError("window must be positive")

    reset = np.r_[True, group_id[1:] != group_id[:-1]]
    group_start = np.maximum.accumulate(np.where(reset, np.arange(n), 0))
    starts = group_start if window is None else np.maximum(group_start, np.arange(n) - window + 1)
    numerator = np.where(np.isfinite(values) & (weights > 0), values * weights, 0.0)
    denominator = np.where(np.isfinite(values) & (weights > 0), weights, 0.0)
    numerator_prefix = np.r_[0.0, np.cumsum(numerator, dtype=np.float64)]
    denominator_prefix = np.r_[0.0, np.cumsum(denominator, dtype=np.float64)]
    rows = np.arange(n)
    numerator_sum = numerator_prefix[rows + 1] - numerator_prefix[starts]
    denominator_sum = denominator_prefix[rows + 1] - denominator_prefix[starts]
    return np.divide(
        numerator_sum,
        denominator_sum,
        out=values.copy(),
        where=denominator_sum > 0,
    )


def aggregate_l2_seconds(
    snapshots: pl.DataFrame,
    *,
    max_quote_age_seconds: int = 2,
) -> pl.DataFrame:
    """Aggregate accepted snapshots and causally regularize short 1s gaps.

    Observed rows contain mid OHLC and first/last BBO for ``[t, t+1s)``.
    At most ``max_quote_age_seconds`` absent rows are carried from the last
    observed close.  Longer holes remain absent and start a new segment.
    Carried rows have zero quote-liquidity weight, so they cannot alter any
    quote-weighted book-VWAP proxy.
    """

    if not isinstance(max_quote_age_seconds, int) or isinstance(
        max_quote_age_seconds, bool
    ) or max_quote_age_seconds < 0:
        raise ValueError("max_quote_age_seconds must be a non-negative integer")
    required = {
        "snapshot_ts_ns",
        "best_bid",
        "best_ask",
        "bid_size_top1",
        "ask_size_top1",
        "mid",
        "book_wap",
    }
    missing = sorted(required - set(snapshots.columns))
    if missing:
        raise ValueError(f"L2 snapshots are missing columns: {', '.join(missing)}")
    if snapshots.is_empty():
        raise ValueError("cannot aggregate an empty snapshot frame")

    observed = (
        snapshots.sort("snapshot_ts_ns")
        .with_columns(
            ((pl.col("snapshot_ts_ns") // SECOND_NS) * SECOND_NS).alias("bar_start_ns")
        )
        .group_by("bar_start_ns", maintain_order=True)
        .agg(
            pl.col("snapshot_ts_ns").min().alias("first_snapshot_ts_ns"),
            pl.col("snapshot_ts_ns").max().alias("last_snapshot_ts_ns"),
            pl.col("mid").sort_by("snapshot_ts_ns").first().alias("open"),
            pl.col("mid").max().alias("high"),
            pl.col("mid").min().alias("low"),
            pl.col("mid").sort_by("snapshot_ts_ns").last().alias("close"),
            pl.col("best_bid").sort_by("snapshot_ts_ns").first().alias("first_bid"),
            pl.col("best_ask").sort_by("snapshot_ts_ns").first().alias("first_ask"),
            pl.col("best_bid").sort_by("snapshot_ts_ns").last().alias("last_bid"),
            pl.col("best_ask").sort_by("snapshot_ts_ns").last().alias("last_ask"),
            pl.col("bid_size_top1")
            .sort_by("snapshot_ts_ns")
            .last()
            .alias("bid_size_top1"),
            pl.col("ask_size_top1")
            .sort_by("snapshot_ts_ns")
            .last()
            .alias("ask_size_top1"),
            pl.col("book_wap").sort_by("snapshot_ts_ns").last().alias("book_wap"),
            pl.len().alias("snapshot_count"),
        )
        .sort("bar_start_ns")
    )

    observed_ts = observed["bar_start_ns"].to_numpy().astype(np.int64)
    if len(observed_ts) > 1:
        delta = np.diff(observed_ts)
        if np.any(delta <= 0) or np.any(delta % SECOND_NS != 0):
            raise ValueError("observed second timestamps must be unique and second-aligned")
        gap_steps = delta // SECOND_NS
    else:
        gap_steps = np.empty(0, dtype=np.int64)
    extra = np.minimum(
        np.maximum(gap_steps - 1, 0), int(max_quote_age_seconds)
    ).astype(np.int64)
    repeat_count = np.ones(len(observed_ts), dtype=np.int64)
    repeat_count[:-1] += extra
    source_index = np.repeat(np.arange(len(observed_ts), dtype=np.int64), repeat_count)
    group_starts = np.repeat(
        np.cumsum(repeat_count, dtype=np.int64) - repeat_count,
        repeat_count,
    )
    offset = np.arange(len(source_index), dtype=np.int64) - group_starts
    ts_ns = observed_ts[source_index] + offset * SECOND_NS
    is_observed = offset == 0

    observed_reset = np.r_[
        True,
        gap_steps > int(max_quote_age_seconds) + 1,
    ]
    observed_segment = np.cumsum(observed_reset, dtype=np.int64) - 1
    segment_id = observed_segment[source_index].astype(np.int32)

    def take(name: str, dtype: np.dtype[np.generic] = np.dtype(np.float64)) -> np.ndarray:
        return observed[name].to_numpy()[source_index].astype(dtype, copy=False)

    observed_open = take("open")
    observed_high = take("high")
    observed_low = take("low")
    close = take("close")
    first_bid_observed = take("first_bid")
    first_ask_observed = take("first_ask")
    last_bid = take("last_bid")
    last_ask = take("last_ask")
    open_price = np.where(is_observed, observed_open, close)
    high = np.where(is_observed, observed_high, close)
    low = np.where(is_observed, observed_low, close)
    first_bid = np.where(is_observed, first_bid_observed, last_bid)
    first_ask = np.where(is_observed, first_ask_observed, last_ask)
    bid_size = take("bid_size_top1")
    ask_size = take("ask_size_top1")
    book_wap = take("book_wap")
    liquidity_weight = np.where(is_observed, bid_size + ask_size, 0.0)
    bid_liquidity_weight = np.where(is_observed, bid_size, 0.0)
    ask_liquidity_weight = np.where(is_observed, ask_size, 0.0)
    last_snapshot_ts = take("last_snapshot_ts_ns", np.dtype(np.int64))
    quote_age_ms = np.maximum(ts_ns - last_snapshot_ts, 0).astype(np.float64) / 1e6
    snapshot_count = np.where(
        is_observed,
        take("snapshot_count", np.dtype(np.int64)),
        0,
    ).astype(np.int32)

    utc_day = ts_ns // DAY_NS
    daily_book_vwap = _weighted_causal_average(
        book_wap, liquidity_weight, utc_day
    )
    book_vwap_60s = _weighted_causal_average(
        book_wap, liquidity_weight, segment_id, window=60
    )
    book_vwap_300s = _weighted_causal_average(
        book_wap, liquidity_weight, segment_id, window=300
    )
    bid_vwap_60s = _weighted_causal_average(
        last_bid, bid_liquidity_weight, segment_id, window=60
    )
    ask_vwap_60s = _weighted_causal_average(
        last_ask, ask_liquidity_weight, segment_id, window=60
    )

    return pl.DataFrame(
        {
            "bar_start_ns": ts_ns,
            "first_snapshot_ts_ns": take("first_snapshot_ts_ns", np.dtype(np.int64)),
            "last_snapshot_ts_ns": last_snapshot_ts,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "first_bid": first_bid,
            "first_ask": first_ask,
            "last_bid": last_bid,
            "last_ask": last_ask,
            "bid_size_top1": bid_size,
            "ask_size_top1": ask_size,
            "book_wap": book_wap,
            "daily_book_vwap": daily_book_vwap,
            "book_vwap_60s": book_vwap_60s,
            "book_vwap_300s": book_vwap_300s,
            "bid_vwap_60s": bid_vwap_60s,
            "ask_vwap_60s": ask_vwap_60s,
            "liquidity_weight": liquidity_weight,
            "snapshot_count": snapshot_count,
            "observed": is_observed,
            "quote_age_ms": quote_age_ms,
            "segment_id": segment_id,
        }
    )


def reconstruct_l2_seconds(
    raw_path: Path,
    *,
    tick_size: float = 0.01,
    min_levels_per_side: int = 3,
    max_new_quotes_per_timestamp: int = 20,
    max_spread_ticks: float = 500.0,
    max_quote_age_seconds: int = 2,
    timestamp_unit: Literal["ns", "us", "ms"] = "ms",
) -> pl.DataFrame:
    """Build a write/read-friendly causal second frame from raw cTrader L2."""

    snapshots = reconstruct_l2_snapshots(
        raw_path,
        tick_size=tick_size,
        min_levels_per_side=min_levels_per_side,
        max_new_quotes_per_timestamp=max_new_quotes_per_timestamp,
        max_spread_ticks=max_spread_ticks,
        timestamp_unit=timestamp_unit,
    )
    return aggregate_l2_seconds(
        snapshots,
        max_quote_age_seconds=max_quote_age_seconds,
    )


def build_l2_second_feature_matrix(
    seconds: pl.DataFrame,
    *,
    tick_size: float = 0.01,
) -> L2SecondFeatureMatrix:
    """Build stationary causal features using only OHLC, BBO and VWAP proxies."""

    required = {
        "bar_start_ns",
        "segment_id",
        "open",
        "high",
        "low",
        "close",
        "last_bid",
        "last_ask",
        "book_wap",
        "daily_book_vwap",
        "book_vwap_60s",
        "book_vwap_300s",
        "bid_vwap_60s",
        "ask_vwap_60s",
    }
    missing = sorted(required - set(seconds.columns))
    if missing:
        raise ValueError(f"L2 seconds are missing columns: {', '.join(missing)}")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    if seconds.is_empty():
        raise ValueError("cannot build features from an empty second frame")

    ts_ns = seconds["bar_start_ns"].to_numpy().astype(np.int64)
    segment_id = seconds["segment_id"].to_numpy().astype(np.int32)
    open_price = seconds["open"].to_numpy().astype(np.float64)
    high = seconds["high"].to_numpy().astype(np.float64)
    low = seconds["low"].to_numpy().astype(np.float64)
    close = seconds["close"].to_numpy().astype(np.float64)
    bid = seconds["last_bid"].to_numpy().astype(np.float64)
    ask = seconds["last_ask"].to_numpy().astype(np.float64)
    book_wap = seconds["book_wap"].to_numpy().astype(np.float64)
    daily = seconds["daily_book_vwap"].to_numpy().astype(np.float64)
    rolling_60 = seconds["book_vwap_60s"].to_numpy().astype(np.float64)
    rolling_300 = seconds["book_vwap_300s"].to_numpy().astype(np.float64)
    bid_vwap_60 = seconds["bid_vwap_60s"].to_numpy().astype(np.float64)
    ask_vwap_60 = seconds["ask_vwap_60s"].to_numpy().astype(np.float64)

    close_delta = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) > 1:
        same_segment = segment_id[1:] == segment_id[:-1]
        close_delta[1:] = np.where(same_segment, close[1:] - close[:-1], np.nan)
    values = (
        (open_price - close) / tick_size,
        (high - close) / tick_size,
        (low - close) / tick_size,
        (high - low) / tick_size,
        close_delta / tick_size,
        (bid - close) / tick_size,
        (ask - close) / tick_size,
        (ask - bid) / tick_size,
        (book_wap - close) / tick_size,
        (daily - close) / tick_size,
        (rolling_60 - close) / tick_size,
        (rolling_300 - close) / tick_size,
        bid_vwap_60,
        ask_vwap_60,
        (bid_vwap_60 - bid) / tick_size,
        (ask_vwap_60 - ask) / tick_size,
    )
    feature_names = (
        "open_minus_close_ticks",
        "high_minus_close_ticks",
        "low_minus_close_ticks",
        "range_ticks",
        "close_delta_1s_ticks",
        "bid_minus_close_ticks",
        "ask_minus_close_ticks",
        "spread_ticks",
        "book_wap_minus_close_ticks",
        "daily_book_vwap_minus_close_ticks",
        "book_vwap_60s_minus_close_ticks",
        "book_vwap_300s_minus_close_ticks",
        "bid_vwap_60s",
        "ask_vwap_60s",
        "bid_vwap_60s_minus_bid_ticks",
        "ask_vwap_60s_minus_ask_ticks",
    )
    x64 = np.column_stack(values)
    valid = np.all(np.isfinite(x64), axis=1)
    x64[~np.isfinite(x64)] = 0.0
    return L2SecondFeatureMatrix(
        ts_ns=ts_ns,
        segment_id=segment_id,
        open=open_price,
        high=high,
        low=low,
        close=close,
        bid=bid,
        ask=ask,
        book_wap=book_wap,
        daily_book_vwap=daily,
        book_vwap_60s=rolling_60,
        book_vwap_300s=rolling_300,
        bid_vwap_60s=bid_vwap_60,
        ask_vwap_60s=ask_vwap_60,
        x=x64.astype(np.float32),
        feature_names=feature_names,
        valid_feature=valid,
    )
