from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import CausalFrames, VENUE_INDEX
from ..scores import (
    compute_score_cube,
    gaussian_smooth_time,
    robust_weighted_aggregate,
    rolling_median_time,
)
from .peaks import build_peak_zones


CURVE_NAMES = (
    "LONG_backward_scoring",
    "LONG_forward_scoring",
    "SHORT_backward_scoring",
    "SHORT_forward_scoring",
)


@dataclass(frozen=True)
class FourCurveTargets:
    ts_ns: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    raw_scores: np.ndarray
    horizons_seconds: np.ndarray
    horizon_weights: np.ndarray
    high_thresholds: np.ndarray
    full_quality_thresholds: np.ndarray
    curve_names: tuple[str, ...] = CURVE_NAMES

    def __post_init__(self) -> None:
        expected = (len(self.ts_ns), len(self.curve_names))
        if self.values.shape != expected or self.valid.shape != expected or self.raw_scores.shape != expected:
            raise ValueError(f"curve arrays must have shape {expected}")
        if np.any(self.values < 0.0) or np.any(self.values > 1.0) or not np.isfinite(self.values).all():
            raise ValueError("target values must be finite and lie in [0, 1]")


def focused_horizon_weights(horizons_seconds: np.ndarray, focus_seconds: float) -> np.ndarray:
    horizons = np.asarray(horizons_seconds, dtype=np.float64)
    if horizons.ndim != 1 or horizons.size == 0 or np.any(horizons <= 0.0):
        raise ValueError("horizons_seconds must be a non-empty positive vector")
    if not np.isfinite(focus_seconds) or focus_seconds <= 0.0:
        raise ValueError("focus_seconds must be positive")
    sigma = 0.35 * float(focus_seconds)
    weights = 0.15 + np.exp(-0.5 * ((horizons - float(focus_seconds)) / sigma) ** 2)
    return weights / weights.sum(dtype=np.float64)


def execution_quotes(
    frames: CausalFrames,
    *,
    latency_ms: float,
    notional_usd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cadence_ns = int(np.median(np.diff(frames.ts_ns)))
    shift = max(0, int(np.ceil(float(latency_ms) * 1_000_000.0 / cadence_ns)))
    venue = VENUE_INDEX["binance_perpetual"]
    bid = np.full(len(frames.ts_ns), np.nan, dtype=np.float64)
    ask = np.full(len(frames.ts_ns), np.nan, dtype=np.float64)
    valid = np.zeros(len(frames.ts_ns), dtype=bool)
    stop = len(frames.ts_ns) - shift
    if stop <= 0:
        return bid, ask, valid
    source = slice(shift, None) if shift else slice(None)
    bid[:stop] = frames.bid[source, venue]
    ask[:stop] = frames.ask[source, venue]
    venue_rows = frames.venue_x[source, venue]
    bid_qty_idx = frames.venue_feature_names.index("best_bid_log_qty")
    ask_qty_idx = frames.venue_feature_names.index("best_ask_log_qty")
    bid_qty = np.expm1(np.maximum(venue_rows[:, bid_qty_idx], 0.0))
    ask_qty = np.expm1(np.maximum(venue_rows[:, ask_qty_idx], 0.0))
    required_qty = float(notional_usd) / np.maximum(ask[:stop], 1e-12)
    fresh = (venue_rows[:, 0] > 0.5) & (venue_rows[:, 1] <= 2_000.0)
    valid[:stop] = (
        fresh
        & np.isfinite(bid[:stop])
        & np.isfinite(ask[:stop])
        & (bid[:stop] < ask[:stop])
        & (bid_qty >= required_qty)
        & (ask_qty >= required_qty)
    )
    return bid, ask, valid


def _agreement(
    frames: CausalFrames,
    *,
    focus_seconds: float,
    latency_ms: float,
) -> np.ndarray:
    cadence_ns = int(np.median(np.diff(frames.ts_ns)))
    lag = max(1, int(round(float(focus_seconds) * 1e9 / cadence_ns)))
    latency = max(0, int(np.ceil(float(latency_ms) * 1_000_000.0 / cadence_ns)))
    mids = 0.5 * (frames.bid + frames.ask)
    quote_valid = np.isfinite(mids) & (frames.bid < frames.ask)
    output = np.full((len(frames.ts_ns), 4), 0.5, dtype=np.float64)
    if lag + latency >= len(frames.ts_ns):
        return output

    start = latency
    count = len(frames.ts_ns) - lag - latency
    past = mids[start : start + count]
    future = mids[start + lag : start + lag + count]
    valid = quote_valid[start : start + count] & quote_valid[start + lag : start + lag + count]
    up = np.sum(valid & (future > past), axis=1) / np.maximum(np.sum(valid, axis=1), 1)
    down = np.sum(valid & (future < past), axis=1) / np.maximum(np.sum(valid, axis=1), 1)
    output[:count, 1] = 0.5 + 0.5 * up
    output[:count, 3] = 0.5 + 0.5 * down
    output[lag : lag + count, 0] = 0.5 + 0.5 * up
    output[lag : lag + count, 2] = 0.5 + 0.5 * down
    return output


def _soft_peak_curve(
    ts_ns: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    segment_id: np.ndarray,
    fit_mask: np.ndarray,
    *,
    focus_seconds: float,
    event_quantile: float,
    peak_floor: float,
    frozen_high: float | None = None,
    frozen_full_quality: float | None = None,
) -> tuple[np.ndarray, float, float]:
    median = rolling_median_time(ts_ns, score, window_seconds=2.1, valid=valid, segment_id=segment_id)
    smooth = gaussian_smooth_time(ts_ns, median, sigma_seconds=1.5, valid=valid, segment_id=segment_id)
    if frozen_high is None or frozen_full_quality is None:
        fit_values = smooth[fit_mask & valid & np.isfinite(smooth) & (smooth > 0.0)]
        if fit_values.size == 0:
            return np.zeros(len(ts_ns), dtype=np.float32), float("inf"), float("inf")
        high = float(np.quantile(fit_values, event_quantile))
        full_quality = max(high, float(np.quantile(fit_values, 0.999)))
    else:
        high = float(frozen_high)
        full_quality = max(high, float(frozen_full_quality))
    cadence_s = float(np.median(np.diff(ts_ns))) / 1e9
    zones = build_peak_zones(
        smooth,
        valid & np.isfinite(smooth),
        segment_id,
        high=high,
        low=max(1e-12, 0.10 * high),
        min_ratio=0.0,
        nms_ticks=max(1, int(round(float(focus_seconds) / (3.0 * cadence_s)))),
    )
    target = np.zeros(len(ts_ns), dtype=np.float64)
    early_limit = min(45.0, 0.5 * float(focus_seconds))
    early_tau = max(cadence_s, early_limit / 3.0)
    late_limit = 3.0
    late_tau = max(cadence_s, late_limit / 3.0)
    amplitude_denominator = max(full_quality - high, 1e-12)
    for event in zones.events:
        amplitude = peak_floor + (1.0 - peak_floor) * np.clip(
            (event.peak_score - high) / amplitude_denominator, 0.0, 1.0
        )
        indices = np.arange(event.start, event.end, dtype=np.int64)
        delta = (indices - event.peak) * cadence_s
        keep = (delta >= -early_limit) & (delta <= late_limit)
        if not np.any(keep):
            continue
        chosen = indices[keep]
        chosen_delta = delta[keep]
        timing = np.where(chosen_delta <= 0.0, np.exp(chosen_delta / early_tau), np.exp(-chosen_delta / late_tau))
        ratio = np.clip(smooth[chosen] / max(event.peak_score, 1e-12), 0.0, 1.0)
        target[chosen] = np.maximum(target[chosen], amplitude * ratio**2 * timing)
    target[~valid] = 0.0
    return np.clip(target, 0.0, 1.0).astype(np.float32), high, full_quality


def _dense_edge_curve(
    ts_ns: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    segment_id: np.ndarray,
    *,
    high: float,
    full_quality: float,
    peak_floor: float,
) -> np.ndarray:
    """Map every economically actionable edge tick onto the public [0, 1] scale."""

    median = rolling_median_time(ts_ns, score, window_seconds=2.1, valid=valid, segment_id=segment_id)
    smooth = gaussian_smooth_time(ts_ns, median, sigma_seconds=1.5, valid=valid, segment_id=segment_id)
    target = np.zeros(len(ts_ns), dtype=np.float64)
    actionable = valid & np.isfinite(smooth) & (smooth >= float(high))
    target[actionable] = float(peak_floor) + (1.0 - float(peak_floor)) * np.clip(
        (smooth[actionable] - float(high)) / max(float(full_quality) - float(high), 1e-12),
        0.0,
        1.0,
    )
    return np.clip(target, 0.0, 1.0).astype(np.float32)


def build_four_curve_targets(
    frames: CausalFrames,
    *,
    horizons_seconds: tuple[float, ...],
    focus_seconds: float,
    fit_mask: np.ndarray,
    fee_round_trip_bps: float = 10.0,
    latency_ms: float = 250.0,
    notional_usd: float = 1_000.0,
    event_quantile: float = 0.50,
    peak_floor: float = 0.75,
    execution_bid: np.ndarray | None = None,
    execution_ask: np.ndarray | None = None,
    execution_valid: np.ndarray | None = None,
    frozen_high_thresholds: np.ndarray | None = None,
    frozen_full_quality_thresholds: np.ndarray | None = None,
    minimum_edge_bps: float | None = None,
    forward_minimum_edge_bps: float | None = None,
    full_quality_edge_bps: float | None = None,
    forward_curve_mode: str = "peak",
) -> FourCurveTargets:
    if not 0.5 < float(peak_floor) <= 1.0:
        raise ValueError("peak_floor must lie in (0.5, 1]")
    if (frozen_high_thresholds is None) != (frozen_full_quality_thresholds is None):
        raise ValueError("frozen target thresholds must be supplied together")
    if (minimum_edge_bps is None) != (full_quality_edge_bps is None):
        raise ValueError("fixed economic target thresholds must be supplied together")
    if minimum_edge_bps is not None:
        if frozen_high_thresholds is not None:
            raise ValueError("fixed economic and frozen target thresholds are mutually exclusive")
        if not np.isfinite(minimum_edge_bps) or float(minimum_edge_bps) <= 0.0:
            raise ValueError("minimum_edge_bps must be finite and positive")
        forward_edge = (
            float(minimum_edge_bps)
            if forward_minimum_edge_bps is None
            else float(forward_minimum_edge_bps)
        )
        if not np.isfinite(forward_edge) or forward_edge <= 0.0:
            raise ValueError("forward_minimum_edge_bps must be finite and positive")
        if not np.isfinite(full_quality_edge_bps) or float(full_quality_edge_bps) <= max(float(minimum_edge_bps), forward_edge):
            raise ValueError("full_quality_edge_bps must exceed all minimum edge thresholds")
        frozen_high_thresholds = np.asarray(
            (minimum_edge_bps, forward_edge, minimum_edge_bps, forward_edge),
            dtype=np.float64,
        )
        frozen_full_quality_thresholds = np.full(4, float(full_quality_edge_bps), dtype=np.float64)
    if forward_curve_mode not in {"peak", "dense_edge"}:
        raise ValueError("forward_curve_mode must be 'peak' or 'dense_edge'")
    if forward_curve_mode == "dense_edge" and frozen_high_thresholds is None:
        raise ValueError("dense_edge forward curves require fixed economic thresholds")
    frozen_high = None if frozen_high_thresholds is None else np.asarray(frozen_high_thresholds, dtype=np.float64)
    frozen_full = None if frozen_full_quality_thresholds is None else np.asarray(frozen_full_quality_thresholds, dtype=np.float64)
    if frozen_high is not None and (frozen_high.shape != (4,) or frozen_full.shape != (4,)):
        raise ValueError("frozen target thresholds must have shape (4,)")
    horizons = np.asarray(horizons_seconds, dtype=np.float64)
    weights = focused_horizon_weights(horizons, focus_seconds)
    if execution_bid is None or execution_ask is None or execution_valid is None:
        bid, ask, quote_valid = execution_quotes(
            frames, latency_ms=latency_ms, notional_usd=notional_usd
        )
    else:
        bid = np.asarray(execution_bid, dtype=np.float64)
        ask = np.asarray(execution_ask, dtype=np.float64)
        quote_valid = np.asarray(execution_valid, dtype=bool)
        if bid.shape != frames.ts_ns.shape or ask.shape != bid.shape or quote_valid.shape != bid.shape:
            raise ValueError("execution quote arrays must align with frames")
    cube = compute_score_cube(
        frames.ts_ns,
        bid,
        ask,
        horizons_seconds=horizons,
        cost_bps=float(fee_round_trip_bps),
        valid=quote_valid,
        segment_id=frames.segment_id,
        venue_names=("binance_perpetual",),
        market_kinds=("derivative",),
        min_coverage=0.95,
    )
    forward_valid = np.all(cube.forward_valid[:, :, 0], axis=1)
    backward_valid = np.all(cube.backward_valid[:, :, 0], axis=1)
    aggregate = np.column_stack(
        (
            robust_weighted_aggregate(cube.backward_long[:, :, 0], weights, mask=cube.backward_valid[:, :, 0]),
            robust_weighted_aggregate(cube.forward_long[:, :, 0], weights, mask=cube.forward_valid[:, :, 0]),
            robust_weighted_aggregate(cube.backward_short[:, :, 0], weights, mask=cube.backward_valid[:, :, 0]),
            robust_weighted_aggregate(cube.forward_short[:, :, 0], weights, mask=cube.forward_valid[:, :, 0]),
        )
    )
    valid = np.column_stack((backward_valid, forward_valid, backward_valid, forward_valid))
    aggregate *= _agreement(frames, focus_seconds=focus_seconds, latency_ms=latency_ms)
    targets = np.zeros_like(aggregate, dtype=np.float32)
    highs = np.zeros(4, dtype=np.float64)
    full = np.zeros(4, dtype=np.float64)
    fit = np.asarray(fit_mask, dtype=bool)
    for column in range(4):
        if forward_curve_mode == "dense_edge" and column in (1, 3):
            highs[column] = float(frozen_high[column])
            full[column] = float(frozen_full[column])
            targets[:, column] = _dense_edge_curve(
                frames.ts_ns,
                aggregate[:, column],
                valid[:, column],
                frames.segment_id,
                high=highs[column],
                full_quality=full[column],
                peak_floor=float(peak_floor),
            )
        else:
            targets[:, column], highs[column], full[column] = _soft_peak_curve(
                frames.ts_ns,
                aggregate[:, column],
                valid[:, column],
                frames.segment_id,
                fit,
                focus_seconds=focus_seconds,
                event_quantile=event_quantile,
                peak_floor=float(peak_floor),
                frozen_high=None if frozen_high is None else float(frozen_high[column]),
                frozen_full_quality=None if frozen_full is None else float(frozen_full[column]),
            )
    return FourCurveTargets(
        ts_ns=frames.ts_ns,
        values=targets,
        valid=valid,
        raw_scores=aggregate.astype(np.float32),
        horizons_seconds=horizons,
        horizon_weights=weights.astype(np.float32),
        high_thresholds=highs,
        full_quality_thresholds=full,
    )
