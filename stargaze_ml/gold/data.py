from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


PRICE_COLUMNS = ("open", "high", "low", "close")
REGIME_NAMES = ("friction", "reversal", "continuation")
FRICTION = 0
REVERSAL = 1
CONTINUATION = 2


@dataclass(frozen=True)
class CandleDataset:
    ts_ns: np.ndarray
    x: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    segment_id: np.ndarray
    feature_names: tuple[str, ...]
    valid_feature: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ts_ns)
        if self.x.shape[0] != n:
            raise ValueError("feature rows must align with timestamps")
        if any(len(value) != n for value in (self.close, self.volume, self.segment_id, self.valid_feature)):
            raise ValueError("candle arrays must have the same length")
        if self.x.shape[1] != len(self.feature_names):
            raise ValueError("feature_names must align with feature columns")


@dataclass(frozen=True)
class LineTargets:
    horizons_minutes: tuple[int, ...]
    line_end_bps: np.ndarray
    actual_end_bps: np.ndarray
    slope_bps_per_minute: np.ndarray
    path_rmse_bps: np.ndarray
    quality: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        shape = self.line_end_bps.shape
        expected = (shape[0], len(self.horizons_minutes))
        for value in (
            self.actual_end_bps,
            self.slope_bps_per_minute,
            self.path_rmse_bps,
            self.quality,
            self.valid,
        ):
            if value.shape != expected:
                raise ValueError("line target arrays must share [time, horizon] shape")


@dataclass(frozen=True)
class RegimeTargets:
    horizons_minutes: tuple[int, ...]
    regime: np.ndarray
    valid: np.ndarray
    past_line_end_bps: np.ndarray
    past_quality: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.past_line_end_bps), len(self.horizons_minutes))
        if self.regime.shape != expected or self.valid.shape != expected:
            raise ValueError("regime targets must be [time, horizon]")
        if self.past_quality.shape != self.past_line_end_bps.shape:
            raise ValueError("past trend arrays must align")


def _read_candles(path: Path) -> pl.DataFrame:
    resolved = Path(path).expanduser().resolve(strict=True)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        frame = pl.read_parquet(resolved)
    elif suffix == ".csv":
        frame = pl.read_csv(resolved, try_parse_dates=True)
    else:
        raise ValueError("gold candles must be a .parquet or .csv file")
    required = {"timestamp", *PRICE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candle file is missing columns: {', '.join(missing)}")
    if "volume" not in frame.columns:
        frame = frame.with_columns(pl.lit(0.0).alias("volume"))
    timestamp = pl.col("timestamp")
    dtype = frame.schema["timestamp"]
    if dtype == pl.Utf8:
        timestamp = timestamp.str.to_datetime(time_zone="UTC", strict=True)
    elif dtype in (pl.Int64, pl.UInt64):
        timestamp = pl.from_epoch(timestamp.cast(pl.Int64), time_unit="ms").dt.replace_time_zone("UTC")
    elif isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        timestamp = timestamp.dt.replace_time_zone("UTC")
    frame = (
        frame.select(
            timestamp.alias("timestamp"),
            *(pl.col(name).cast(pl.Float64) for name in PRICE_COLUMNS),
            pl.col("volume").cast(pl.Float64),
        )
        .drop_nulls()
        .unique(subset=["timestamp"], keep="last")
        .sort("timestamp")
    )
    if frame.height < 2_000:
        raise ValueError("at least 2,000 one-minute candles are required")
    return frame


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    if window <= 0 or len(values) < window:
        return out
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    out[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return out


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(np.maximum(_rolling_mean(np.square(values), window), 0.0))


def _rolling_extreme(values: np.ndarray, window: int, *, maximum: bool) -> np.ndarray:
    series = pl.Series("value", np.asarray(values, dtype=np.float64))
    expr = pl.col("value").rolling_max(window_size=window) if maximum else pl.col("value").rolling_min(window_size=window)
    return pl.DataFrame({"value": series}).select(expr).to_series().to_numpy()


def build_candle_dataset(path: Path) -> CandleDataset:
    frame = _read_candles(path)
    timestamps = frame["timestamp"].dt.epoch("ns").to_numpy().astype(np.int64)
    open_price = frame["open"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    volume = frame["volume"].to_numpy()
    if not np.all(np.isfinite(np.column_stack((open_price, high, low, close, volume)))):
        raise ValueError("candle data contains non-finite values")
    if np.any(np.minimum.reduce((open_price, high, low, close)) <= 0.0):
        raise ValueError("candle prices must be positive")
    if np.any(high < np.maximum.reduce((open_price, close, low))):
        raise ValueError("candle high is below another OHLC value")
    if np.any(low > np.minimum.reduce((open_price, close, high))):
        raise ValueError("candle low is above another OHLC value")

    gap = np.diff(timestamps, prepend=timestamps[0])
    segment_id = np.cumsum((gap != 60_000_000_000) & (np.arange(len(gap)) > 0)).astype(np.int32)
    log_open = np.log(open_price)
    log_high = np.log(high)
    log_low = np.log(low)
    log_close = np.log(close)
    close_return = np.diff(log_close, prepend=log_close[0])
    previous_close = np.roll(log_close, 1)
    previous_close[0] = log_open[0]
    candle_range = np.maximum(log_high - log_low, 1e-9)
    body = log_close - log_open
    upper_wick = log_high - np.maximum(log_open, log_close)
    lower_wick = np.minimum(log_open, log_close) - log_low

    features: list[np.ndarray] = [
        close_return * 10_000.0,
        (log_open - previous_close) * 10_000.0,
        body * 10_000.0,
        candle_range * 10_000.0,
        upper_wick * 10_000.0,
        lower_wick * 10_000.0,
        body / candle_range,
        upper_wick / candle_range,
        lower_wick / candle_range,
        np.log1p(np.maximum(volume, 0.0)),
    ]
    names = [
        "return_1m_bps",
        "open_gap_bps",
        "body_bps",
        "range_bps",
        "upper_wick_bps",
        "lower_wick_bps",
        "body_fraction",
        "upper_wick_fraction",
        "lower_wick_fraction",
        "log_tick_volume",
    ]
    for window in (5, 15, 60):
        rolling_return = log_close - np.roll(log_close, window)
        rolling_return[:window] = np.nan
        features.extend((rolling_return * 10_000.0, _rolling_rms(close_return * 10_000.0, window)))
        names.extend((f"return_{window}m_bps", f"realized_vol_{window}m_bps"))
    for window in (60, 240, 1_440):
        rolling_high = _rolling_extreme(high, window, maximum=True)
        rolling_low = _rolling_extreme(low, window, maximum=False)
        features.extend(
            (
                np.log(close / rolling_high) * 10_000.0,
                np.log(close / rolling_low) * 10_000.0,
            )
        )
        names.extend((f"distance_high_{window}m_bps", f"distance_low_{window}m_bps"))

    minute_utc = (timestamps // 60_000_000_000) % 1_440
    week_minute = (timestamps // 60_000_000_000) % (7 * 1_440)
    features.extend(
        (
            np.sin(2.0 * np.pi * minute_utc / 1_440.0),
            np.cos(2.0 * np.pi * minute_utc / 1_440.0),
            np.sin(2.0 * np.pi * week_minute / (7.0 * 1_440.0)),
            np.cos(2.0 * np.pi * week_minute / (7.0 * 1_440.0)),
        )
    )
    names.extend(("minute_of_day_sin", "minute_of_day_cos", "minute_of_week_sin", "minute_of_week_cos"))

    x = np.column_stack(features).astype(np.float32)
    valid_feature = np.all(np.isfinite(x), axis=1)
    x[~np.isfinite(x)] = 0.0
    return CandleDataset(
        ts_ns=timestamps,
        x=x,
        close=close.astype(np.float64),
        volume=volume.astype(np.float64),
        segment_id=segment_id,
        feature_names=tuple(names),
        valid_feature=valid_feature,
    )


def build_line_targets(
    close: np.ndarray,
    segment_id: np.ndarray,
    horizons_minutes: tuple[int, ...],
    *,
    past_vol_window: int = 60,
    price_change: str = "log",
) -> LineTargets:
    close = np.asarray(close, dtype=np.float64)
    segment_id = np.asarray(segment_id, dtype=np.int64)
    horizons = tuple(int(value) for value in horizons_minutes)
    if close.ndim != 1 or segment_id.shape != close.shape:
        raise ValueError("close and segment_id must be aligned one-dimensional arrays")
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("positive horizons are required")
    if price_change not in {"log", "simple"}:
        raise ValueError("price_change must be 'log' or 'simple'")
    n = len(close)
    k = len(horizons)
    arrays = [np.full((n, k), np.nan, dtype=np.float32) for _ in range(5)]
    line_end, actual_end, slope_out, rmse_out, quality_out = arrays
    valid = np.zeros((n, k), dtype=bool)
    if price_change == "log":
        returns_bps = np.diff(np.log(close), prepend=np.log(close[0])) * 10_000.0
    else:
        previous = np.roll(close, 1)
        previous[0] = close[0]
        returns_bps = (close / previous - 1.0) * 10_000.0
    past_vol = _rolling_rms(returns_bps, int(past_vol_window))

    log_close = np.log(close)
    for column, horizon in enumerate(horizons):
        count = n - horizon
        if count <= 0:
            continue
        base = log_close[:count] if price_change == "log" else close[:count]
        sum_tau_y = np.zeros(count, dtype=np.float64)
        sum_y2 = np.zeros(count, dtype=np.float64)
        for tau in range(1, horizon + 1):
            if price_change == "log":
                y = (log_close[tau : tau + count] - base) * 10_000.0
            else:
                y = (close[tau : tau + count] / base - 1.0) * 10_000.0
            sum_tau_y += tau * y
            sum_y2 += y * y
        sum_tau2 = horizon * (horizon + 1) * (2 * horizon + 1) / 6.0
        slope = sum_tau_y / sum_tau2
        fitted_end = slope * horizon
        residual_ss = np.maximum(sum_y2 - np.square(sum_tau_y) / sum_tau2, 0.0)
        rmse = np.sqrt(residual_ss / horizon)
        if price_change == "log":
            actual = (log_close[horizon : horizon + count] - base) * 10_000.0
        else:
            actual = (close[horizon : horizon + count] / base - 1.0) * 10_000.0
        same_segment = segment_id[:count] == segment_id[horizon : horizon + count]
        finite = np.isfinite(past_vol[:count])
        target_valid = same_segment & finite
        signal_rms = np.abs(slope) * np.sqrt(sum_tau2 / horizon)
        linearity = signal_rms / (signal_rms + rmse + 1e-6)
        expected_noise = np.maximum(past_vol[:count] * np.sqrt(horizon), 1e-3)
        magnitude = np.abs(fitted_end) / (np.abs(fitted_end) + expected_noise)
        quality = np.sqrt(np.clip(linearity * magnitude, 0.0, 1.0))

        line_end[:count, column] = fitted_end.astype(np.float32)
        actual_end[:count, column] = actual.astype(np.float32)
        slope_out[:count, column] = slope.astype(np.float32)
        rmse_out[:count, column] = rmse.astype(np.float32)
        quality_out[:count, column] = quality.astype(np.float32)
        valid[:count, column] = target_valid

    for value in (line_end, actual_end, slope_out, rmse_out, quality_out):
        value[~valid] = 0.0
    return LineTargets(
        horizons_minutes=horizons,
        line_end_bps=line_end,
        actual_end_bps=actual_end,
        slope_bps_per_minute=slope_out,
        path_rmse_bps=rmse_out,
        quality=quality_out,
        valid=valid,
    )


def _past_trend(
    close: np.ndarray,
    segment_id: np.ndarray,
    *,
    context_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    end_bps = np.zeros(n, dtype=np.float32)
    quality = np.zeros(n, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    horizon = int(context_minutes) - 1
    if horizon < 2 or n <= horizon:
        return end_bps, quality, valid
    log_close = np.log(close)
    count = n - horizon
    base = log_close[:count]
    sum_tau_y = np.zeros(count, dtype=np.float64)
    sum_y2 = np.zeros(count, dtype=np.float64)
    for tau in range(1, horizon + 1):
        y = (log_close[tau : tau + count] - base) * 10_000.0
        sum_tau_y += tau * y
        sum_y2 += y * y
    sum_tau2 = horizon * (horizon + 1) * (2 * horizon + 1) / 6.0
    slope = sum_tau_y / sum_tau2
    fitted_end = slope * horizon
    residual_ss = np.maximum(sum_y2 - np.square(sum_tau_y) / sum_tau2, 0.0)
    rmse = np.sqrt(residual_ss / horizon)
    signal_rms = np.abs(slope) * np.sqrt(sum_tau2 / horizon)
    linearity = signal_rms / (signal_rms + rmse + 1e-6)
    path_rms = np.sqrt(np.maximum(sum_y2 / horizon, 0.0))
    magnitude = np.abs(fitted_end) / (np.abs(fitted_end) + path_rms + 1e-6)
    trend_quality = np.sqrt(np.clip(linearity * magnitude, 0.0, 1.0))
    indices = np.arange(horizon, n)
    same_segment = segment_id[indices] == segment_id[indices - horizon]
    end_bps[indices] = fitted_end.astype(np.float32)
    quality[indices] = trend_quality.astype(np.float32)
    valid[indices] = same_segment
    end_bps[~valid] = 0.0
    quality[~valid] = 0.0
    return end_bps, quality, valid


def build_regime_targets(
    close: np.ndarray,
    segment_id: np.ndarray,
    line_targets: LineTargets,
    *,
    context_minutes: int = 60,
    minimum_quality: float = 0.25,
) -> RegimeTargets:
    if not 0.0 <= minimum_quality <= 1.0:
        raise ValueError("minimum_quality must be in [0, 1]")
    past_end, past_quality, past_valid = _past_trend(
        np.asarray(close, dtype=np.float64),
        np.asarray(segment_id, dtype=np.int64),
        context_minutes=int(context_minutes),
    )
    n, k = line_targets.line_end_bps.shape
    if n != len(past_end):
        raise ValueError("line targets do not align with candle data")
    regime = np.full((n, k), FRICTION, dtype=np.int8)
    valid = line_targets.valid & past_valid[:, None]
    directional = (
        valid
        & (past_quality[:, None] >= float(minimum_quality))
        & (line_targets.quality >= float(minimum_quality))
    )
    same_direction = np.signbit(past_end[:, None]) == np.signbit(line_targets.line_end_bps)
    regime[directional & same_direction] = CONTINUATION
    regime[directional & ~same_direction] = REVERSAL
    regime[~valid] = FRICTION
    return RegimeTargets(
        horizons_minutes=line_targets.horizons_minutes,
        regime=regime,
        valid=valid,
        past_line_end_bps=past_end,
        past_quality=past_quality,
    )


def save_prepared_dataset(
    path: Path,
    candles: CandleDataset,
    targets: LineTargets,
    regimes: RegimeTargets,
    *,
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        ts_ns=candles.ts_ns,
        x=candles.x,
        close=candles.close,
        volume=candles.volume,
        segment_id=candles.segment_id,
        feature_names=np.asarray(candles.feature_names),
        valid_feature=candles.valid_feature,
        horizons_minutes=np.asarray(targets.horizons_minutes, dtype=np.int32),
        line_end_bps=targets.line_end_bps,
        actual_end_bps=targets.actual_end_bps,
        slope_bps_per_minute=targets.slope_bps_per_minute,
        path_rmse_bps=targets.path_rmse_bps,
        quality=targets.quality,
        target_valid=targets.valid,
        regime=regimes.regime,
        regime_valid=regimes.valid,
        past_line_end_bps=regimes.past_line_end_bps,
        past_quality=regimes.past_quality,
    )
    from ..artifacts import write_json

    write_json(destination.with_suffix(destination.suffix + ".manifest.json"), metadata)
