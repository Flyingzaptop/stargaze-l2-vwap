from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


@dataclass(frozen=True)
class L2FeatureMatrix:
    ts_ns: np.ndarray
    segment_id: np.ndarray
    mid: np.ndarray
    microprice: np.ndarray
    x: np.ndarray
    feature_names: tuple[str, ...]
    valid_feature: np.ndarray


@dataclass(frozen=True)
class AngleTargets:
    horizons_steps: tuple[int, ...]
    step_seconds: int
    angle_radians: np.ndarray
    slope_ticks_per_second: np.ndarray
    line_end_ticks: np.ndarray
    actual_end_ticks: np.ndarray
    path_rmse_ticks: np.ndarray
    past_sigma_ticks_sqrt_second: np.ndarray
    valid: np.ndarray


def _timestamp_ns_expression(dtype: pl.DataType, *, timestamp_unit: str) -> pl.Expr:
    if isinstance(dtype, pl.Datetime):
        return pl.col("timestamp").dt.epoch("ns")
    multiplier = {"ns": 1, "us": 1_000, "ms": 1_000_000}.get(timestamp_unit)
    if multiplier is None:
        raise ValueError("timestamp_unit must be one of: ns, us, ms")
    return pl.col("timestamp").cast(pl.Int64) * multiplier


def reconstruct_l2_bars(
    raw_path: Path,
    *,
    bar_seconds: int = 2,
    tick_size: float = 0.01,
    min_levels_per_side: int = 3,
    max_new_quotes_per_timestamp: int = 20,
    max_spread_ticks: float = 500.0,
    timestamp_unit: Literal["ns", "us", "ms"] = "ms",
) -> pl.DataFrame:
    """Build conservative L2 snapshots without carrying ambiguous quote state.

    cTrader depth batches in the available archive are full refreshes encoded as
    delete/new quote rows. We aggregate only positive ``new`` rows at an exact
    timestamp, reject crossed/oversized/one-sided packets, then retain the last
    accepted snapshot in every bar.
    """

    path = Path(raw_path).expanduser().resolve(strict=True)
    schema = pl.scan_parquet(path).collect_schema()
    required = {"timestamp", "quote_id", "bid", "ask", "size", "type"}
    missing = sorted(required - set(schema.names()))
    if missing:
        raise ValueError(f"raw L2 parquet is missing columns: {', '.join(missing)}")
    if bar_seconds <= 0 or tick_size <= 0:
        raise ValueError("bar_seconds and tick_size must be positive")

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
            bid_size.sort_by(bid_price, descending=True).head(3).sum().alias("bid_depth3"),
            ask_size.sort_by(ask_price).head(3).sum().alias("ask_depth3"),
            bid_price.sort(descending=True).head(3).min().alias("bid_price3"),
            ask_price.sort().head(3).max().alias("ask_price3"),
            bid_mask.sum().alias("bid_levels"),
            ask_mask.sum().alias("ask_levels"),
            pl.len().alias("new_quote_count"),
        )
        .with_columns(
            ((pl.col("best_ask") - pl.col("best_bid")) / tick_size).alias("spread_ticks")
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
            (
                (pl.col("bid_size_top1") - pl.col("ask_size_top1"))
                / (pl.col("bid_size_top1") + pl.col("ask_size_top1"))
            ).alias("imbalance_top1"),
            (
                (pl.col("bid_depth3") - pl.col("ask_depth3"))
                / (pl.col("bid_depth3") + pl.col("ask_depth3"))
            ).alias("imbalance_top3"),
            ((pl.col("best_bid") - pl.col("bid_price3")) / tick_size).alias(
                "bid_width3_ticks"
            ),
            ((pl.col("ask_price3") - pl.col("best_ask")) / tick_size).alias(
                "ask_width3_ticks"
            ),
        )
        .sort("snapshot_ts_ns")
    )

    bar_ns = int(bar_seconds) * 1_000_000_000
    snapshots = snapshots.with_columns(
        ((pl.col("snapshot_ts_ns") // bar_ns) * bar_ns).alias("bar_start_ns")
    )
    latest_columns = (
        "best_bid",
        "best_ask",
        "bid_size_top1",
        "ask_size_top1",
        "bid_depth3",
        "ask_depth3",
        "bid_levels",
        "ask_levels",
        "spread_ticks",
        "mid",
        "microprice",
        "imbalance_top1",
        "imbalance_top3",
        "bid_width3_ticks",
        "ask_width3_ticks",
    )
    bars = (
        snapshots.group_by("bar_start_ns", maintain_order=True)
        .agg(
            pl.col("snapshot_ts_ns").max().alias("last_snapshot_ts_ns"),
            *(pl.col(name).sort_by("snapshot_ts_ns").last().alias(name) for name in latest_columns),
            pl.len().alias("snapshot_count"),
            pl.col("new_quote_count").sum().alias("new_quote_count"),
            ((pl.col("mid").max() - pl.col("mid").min()) / tick_size).alias(
                "mid_range_ticks"
            ),
            ((pl.col("mid").sort_by("snapshot_ts_ns").last()
              - pl.col("mid").sort_by("snapshot_ts_ns").first()) / tick_size).alias(
                "mid_change_ticks"
            ),
        )
        .sort("bar_start_ns")
        .collect(engine="streaming")
    )
    if bars.is_empty():
        raise ValueError("no valid L2 bars survived reconstruction")
    ts = bars["bar_start_ns"].to_numpy().astype(np.int64)
    reset = np.ones(len(ts), dtype=bool)
    reset[1:] = np.diff(ts) != bar_ns
    segment_id = np.cumsum(reset).astype(np.int32) - 1
    return bars.with_columns(pl.Series("segment_id", segment_id))


def _rolling_segment(
    values: np.ndarray,
    segment_id: np.ndarray,
    window: int,
    *,
    rms: bool = False,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    segment_id = np.asarray(segment_id)
    out = np.full(len(values), np.nan, dtype=np.float64)
    if window <= 0:
        raise ValueError("rolling window must be positive")
    starts = np.flatnonzero(np.r_[True, segment_id[1:] != segment_id[:-1]])
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        part = values[start:end]
        finite = np.isfinite(part)
        payload = np.where(finite, np.square(part) if rms else part, 0.0)
        cumulative = np.r_[0.0, np.cumsum(payload)]
        counts = np.r_[0, np.cumsum(finite)]
        if len(part) < window:
            continue
        totals = cumulative[window:] - cumulative[:-window]
        n_finite = counts[window:] - counts[:-window]
        result = np.full(len(totals), np.nan, dtype=np.float64)
        complete = n_finite == window
        result[complete] = totals[complete] / window
        if rms:
            result[complete] = np.sqrt(np.maximum(result[complete], 0.0))
        out[start + window - 1 : end] = result
    return out


def _segment_lag_delta(
    values: np.ndarray,
    segment_id: np.ndarray,
    lag: int,
) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if lag < len(values):
        same = segment_id[lag:] == segment_id[:-lag]
        delta = values[lag:] - values[:-lag]
        out[lag:] = np.where(same, delta, np.nan)
    return out


def build_l2_feature_matrix(
    bars: pl.DataFrame,
    *,
    tick_size: float = 0.01,
    minimum_history_steps: int = 60,
) -> L2FeatureMatrix:
    required = {
        "bar_start_ns",
        "segment_id",
        "mid",
        "microprice",
        "spread_ticks",
        "imbalance_top1",
        "imbalance_top3",
        "bid_size_top1",
        "ask_size_top1",
        "bid_depth3",
        "ask_depth3",
        "bid_width3_ticks",
        "ask_width3_ticks",
        "snapshot_count",
        "new_quote_count",
        "mid_range_ticks",
        "mid_change_ticks",
    }
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"L2 bars are missing columns: {', '.join(missing)}")

    ts_ns = bars["bar_start_ns"].to_numpy().astype(np.int64)
    segment = bars["segment_id"].to_numpy().astype(np.int32)
    mid = bars["mid"].to_numpy().astype(np.float64)
    micro = bars["microprice"].to_numpy().astype(np.float64)

    features: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        features.append(np.asarray(value, dtype=np.float64))

    scalar_columns = (
        "spread_ticks",
        "imbalance_top1",
        "imbalance_top3",
        "bid_width3_ticks",
        "ask_width3_ticks",
        "mid_range_ticks",
        "mid_change_ticks",
    )
    for name in scalar_columns:
        add(name, bars[name].to_numpy())
    add("micro_minus_mid_ticks", (micro - mid) / tick_size)
    for name in (
        "bid_size_top1",
        "ask_size_top1",
        "bid_depth3",
        "ask_depth3",
        "snapshot_count",
        "new_quote_count",
    ):
        add(f"log1p_{name}", np.log1p(np.maximum(bars[name].to_numpy(), 0.0)))

    micro_divergence = (micro - mid) / tick_size
    imbalance_top1 = bars["imbalance_top1"].to_numpy().astype(np.float64)
    imbalance_top3 = bars["imbalance_top3"].to_numpy().astype(np.float64)
    spread = bars["spread_ticks"].to_numpy().astype(np.float64)
    bid_size_top1 = bars["bid_size_top1"].to_numpy().astype(np.float64)
    ask_size_top1 = bars["ask_size_top1"].to_numpy().astype(np.float64)
    bid_depth3 = bars["bid_depth3"].to_numpy().astype(np.float64)
    ask_depth3 = bars["ask_depth3"].to_numpy().astype(np.float64)

    for lag in (1, 2, 3, 5, 10, 15, 30):
        add(f"mid_delta_{lag * 2}s_ticks", _segment_lag_delta(mid, segment, lag) / tick_size)
        add(
            f"micro_delta_{lag * 2}s_ticks",
            _segment_lag_delta(micro, segment, lag) / tick_size,
        )

    for lag in (1, 3, 5):
        seconds = lag * 2
        add(
            f"imbalance_top1_delta_{seconds}s",
            _segment_lag_delta(imbalance_top1, segment, lag),
        )
        add(
            f"imbalance_top3_delta_{seconds}s",
            _segment_lag_delta(imbalance_top3, segment, lag),
        )
        add(f"spread_delta_{seconds}s", _segment_lag_delta(spread, segment, lag))
        add(
            f"micro_divergence_delta_{seconds}s",
            _segment_lag_delta(micro_divergence, segment, lag),
        )
        delta_bid_top = _segment_lag_delta(bid_size_top1, segment, lag)
        delta_ask_top = _segment_lag_delta(ask_size_top1, segment, lag)
        add(
            f"top1_liquidity_flow_imbalance_{seconds}s",
            (delta_bid_top - delta_ask_top)
            / (np.abs(delta_bid_top) + np.abs(delta_ask_top) + 1.0),
        )
        delta_bid_depth = _segment_lag_delta(bid_depth3, segment, lag)
        delta_ask_depth = _segment_lag_delta(ask_depth3, segment, lag)
        add(
            f"depth3_liquidity_flow_imbalance_{seconds}s",
            (delta_bid_depth - delta_ask_depth)
            / (np.abs(delta_bid_depth) + np.abs(delta_ask_depth) + 1.0),
        )

    micro_return = _segment_lag_delta(micro, segment, 1) / tick_size
    for window in (5, 15, 30, 60):
        seconds = window * 2
        add(f"micro_return_mean_{seconds}s", _rolling_segment(micro_return, segment, window))
        add(f"micro_return_rms_{seconds}s", _rolling_segment(micro_return, segment, window, rms=True))
        add(
            f"imbalance_top3_mean_{seconds}s",
            _rolling_segment(bars["imbalance_top3"].to_numpy(), segment, window),
        )
        add(
            f"spread_mean_{seconds}s",
            _rolling_segment(bars["spread_ticks"].to_numpy(), segment, window),
        )
        add(
            f"imbalance_top1_mean_{seconds}s",
            _rolling_segment(imbalance_top1, segment, window),
        )
        add(
            f"micro_divergence_mean_{seconds}s",
            _rolling_segment(micro_divergence, segment, window),
        )
        add(
            f"snapshot_count_mean_{seconds}s",
            _rolling_segment(bars["snapshot_count"].to_numpy(), segment, window),
        )
        add(
            f"new_quote_count_mean_{seconds}s",
            _rolling_segment(bars["new_quote_count"].to_numpy(), segment, window),
        )

    seconds_of_day = (ts_ns // 1_000_000_000) % 86_400
    seconds_of_week = (ts_ns // 1_000_000_000) % (7 * 86_400)
    add("time_day_sin", np.sin(2 * np.pi * seconds_of_day / 86_400.0))
    add("time_day_cos", np.cos(2 * np.pi * seconds_of_day / 86_400.0))
    add("time_week_sin", np.sin(2 * np.pi * seconds_of_week / (7 * 86_400.0)))
    add("time_week_cos", np.cos(2 * np.pi * seconds_of_week / (7 * 86_400.0)))

    x64 = np.column_stack(features)
    finite = np.all(np.isfinite(x64), axis=1)
    position = np.zeros(len(segment), dtype=np.int32)
    for index in range(1, len(segment)):
        position[index] = position[index - 1] + 1 if segment[index] == segment[index - 1] else 0
    valid = finite & (position >= int(minimum_history_steps))
    x64[~np.isfinite(x64)] = 0.0
    return L2FeatureMatrix(
        ts_ns=ts_ns,
        segment_id=segment,
        mid=mid,
        microprice=micro,
        x=x64.astype(np.float32),
        feature_names=tuple(names),
        valid_feature=valid,
    )


def build_angle_targets(
    mid: np.ndarray,
    segment_id: np.ndarray,
    horizons_steps: tuple[int, ...],
    *,
    step_seconds: int = 2,
    tick_size: float = 0.01,
    vol_window_steps: int = 30,
    sigma_floor_ticks_sqrt_second: float = 0.05,
) -> AngleTargets:
    mid = np.asarray(mid, dtype=np.float64)
    segment = np.asarray(segment_id, dtype=np.int64)
    horizons = tuple(int(value) for value in horizons_steps)
    if mid.ndim != 1 or segment.shape != mid.shape:
        raise ValueError("mid and segment_id must be aligned one-dimensional arrays")
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("positive horizons_steps are required")
    if step_seconds <= 0 or tick_size <= 0 or vol_window_steps <= 1:
        raise ValueError("step_seconds, tick_size and vol_window_steps must be positive")

    returns_ticks = _segment_lag_delta(mid, segment, 1) / tick_size
    sigma = _rolling_segment(returns_ticks, segment, int(vol_window_steps), rms=True)
    sigma /= np.sqrt(float(step_seconds))
    sigma_used = np.maximum(sigma, float(sigma_floor_ticks_sqrt_second))

    n = len(mid)
    k = len(horizons)
    angle = np.zeros((n, k), dtype=np.float32)
    slope_out = np.zeros((n, k), dtype=np.float32)
    line_end = np.zeros((n, k), dtype=np.float32)
    actual_end = np.zeros((n, k), dtype=np.float32)
    rmse_out = np.zeros((n, k), dtype=np.float32)
    valid = np.zeros((n, k), dtype=bool)

    for column, horizon_steps in enumerate(horizons):
        count = n - horizon_steps
        if count <= 0:
            continue
        tau_seconds = np.arange(1, horizon_steps + 1, dtype=np.float64) * step_seconds
        denominator = float(np.sum(np.square(tau_seconds)))
        base = mid[:count]
        sum_tau_y = np.zeros(count, dtype=np.float64)
        sum_y2 = np.zeros(count, dtype=np.float64)
        for offset, tau in enumerate(tau_seconds, start=1):
            y = (mid[offset : offset + count] - base) / tick_size
            sum_tau_y += tau * y
            sum_y2 += y * y
        slope = sum_tau_y / denominator
        horizon_seconds = horizon_steps * step_seconds
        end = slope * horizon_seconds
        residual_ss = np.maximum(sum_y2 - np.square(sum_tau_y) / denominator, 0.0)
        rmse = np.sqrt(residual_ss / horizon_steps)
        same_segment = segment[:count] == segment[horizon_steps : horizon_steps + count]
        target_valid = same_segment & np.isfinite(sigma_used[:count])
        normalized_slope = slope * np.sqrt(float(horizon_seconds)) / sigma_used[:count]

        angle[:count, column] = np.arctan(normalized_slope).astype(np.float32)
        slope_out[:count, column] = slope.astype(np.float32)
        line_end[:count, column] = end.astype(np.float32)
        actual_end[:count, column] = (
            (mid[horizon_steps : horizon_steps + count] - base) / tick_size
        ).astype(np.float32)
        rmse_out[:count, column] = rmse.astype(np.float32)
        valid[:count, column] = target_valid

    for array in (angle, slope_out, line_end, actual_end, rmse_out):
        array[~valid] = 0.0
    return AngleTargets(
        horizons_steps=horizons,
        step_seconds=int(step_seconds),
        angle_radians=angle,
        slope_ticks_per_second=slope_out,
        line_end_ticks=line_end,
        actual_end_ticks=actual_end,
        path_rmse_ticks=rmse_out,
        past_sigma_ticks_sqrt_second=sigma_used.astype(np.float32),
        valid=valid,
    )


def angle_to_slope(
    angle_radians: np.ndarray | float,
    sigma_ticks_sqrt_second: np.ndarray | float,
    *,
    horizon_seconds: int | float,
) -> np.ndarray:
    return (
        np.tan(np.asarray(angle_radians, dtype=np.float64))
        * np.asarray(sigma_ticks_sqrt_second, dtype=np.float64)
        / np.sqrt(float(horizon_seconds))
    )
