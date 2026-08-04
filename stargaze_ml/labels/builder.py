from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Action, CausalFrames, PositionSide
from .peaks import PeakZones, build_peak_zones, local_peak_indices


@dataclass(frozen=True)
class OracleEpisode:
    side: PositionSide
    entry_idx: int
    forward_peak_idx: int
    exit_peak_idx: int
    close_zone_start: int
    close_zone_end: int
    horizon_idx: int
    forward_peak_score: float
    backward_peak_score: float


@dataclass
class LabelBuildResult:
    flat_action: np.ndarray
    open_long_zone: np.ndarray
    open_short_zone: np.ndarray
    close_long_zone: np.ndarray
    close_short_zone: np.ndarray
    dominant_long_horizon: np.ndarray
    dominant_short_horizon: np.ndarray
    normalized_forward_long: np.ndarray
    normalized_forward_short: np.ndarray
    episodes: tuple[OracleEpisode, ...]
    high_long: float
    high_short: float


def _scale_horizons(values: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim == 1:
        values = values[:, None]
    scales = np.ones(values.shape[1], dtype=np.float64)
    for h in range(values.shape[1]):
        positive = values[fit_mask, h]
        positive = positive[positive > 0.0]
        if positive.size:
            scales[h] = max(float(np.quantile(positive, 0.95)), 1e-6)
    return (values / scales[None, :]).astype(np.float32), scales.astype(np.float32)


def _threshold(score: np.ndarray, fit_mask: np.ndarray, quantile: float) -> float:
    values = score[fit_mask & np.isfinite(score) & (score > 0.0)]
    if values.size == 0:
        return float("inf")
    return float(np.quantile(values, float(quantile)))


def _selected_close_zone(
    backward: np.ndarray,
    *,
    entry: int,
    segment_end: int,
    horizon_ticks: int,
    ratio: float,
    search_multiplier: float,
) -> tuple[int, int, int, float] | None:
    search_end = min(segment_end, entry + max(3, int(round(horizon_ticks * float(search_multiplier)))) + 1)
    if search_end - entry < 3:
        return None
    peaks = local_peak_indices(backward, entry + 1, search_end)
    if peaks.size:
        peak = int(peaks[np.argmax(backward[peaks])])
    else:
        peak = entry + 1 + int(np.argmax(backward[entry + 1 : search_end]))
    peak_value = float(backward[peak])
    if not np.isfinite(peak_value) or peak_value <= 0.0:
        return None
    threshold = peak_value * float(ratio)
    left = peak
    while left > entry + 1 and backward[left - 1] >= threshold:
        left -= 1
    right = peak + 1
    while right < search_end and backward[right] >= threshold:
        right += 1
    return left, right, peak, peak_value


def build_labels(
    frames: CausalFrames,
    *,
    forward_long_h: np.ndarray,
    forward_short_h: np.ndarray,
    backward_long_h: np.ndarray,
    backward_short_h: np.ndarray,
    horizons_seconds: tuple[float, ...],
    fit_mask: np.ndarray,
    event_high_quantile: float = 0.995,
    event_low_fraction: float = 0.10,
    peak_ratio: float = 0.75,
    search_multiplier: float = 2.0,
    peak_nms_seconds: float = 5.0,
) -> LabelBuildResult:
    n = len(frames.ts_ns)
    fit_mask = np.asarray(fit_mask, dtype=bool) & frames.valid
    f_long, _ = _scale_horizons(forward_long_h, fit_mask)
    f_short, _ = _scale_horizons(forward_short_h, fit_mask)
    b_long, _ = _scale_horizons(backward_long_h, fit_mask)
    b_short, _ = _scale_horizons(backward_short_h, fit_mask)
    dom_long_h = np.argmax(f_long, axis=1).astype(np.int16)
    dom_short_h = np.argmax(f_short, axis=1).astype(np.int16)
    dom_long = np.max(f_long, axis=1)
    dom_short = np.max(f_short, axis=1)
    high_long = _threshold(dom_long, fit_mask, event_high_quantile)
    high_short = _threshold(dom_short, fit_mask, event_high_quantile)
    cadence_s = float(np.median(np.diff(frames.ts_ns))) / 1e9 if n > 1 else 0.1
    nms_ticks = max(1, int(round(float(peak_nms_seconds) / cadence_s)))
    long_zones = build_peak_zones(dom_long, frames.valid, frames.segment_id, high=high_long, low=high_long * event_low_fraction, min_ratio=peak_ratio, nms_ticks=nms_ticks)
    short_zones = build_peak_zones(dom_short, frames.valid, frames.segment_id, high=high_short, low=high_short * event_low_fraction, min_ratio=peak_ratio, nms_ticks=nms_ticks)
    open_long = long_zones.zone & (~short_zones.zone | (dom_long >= dom_short))
    open_short = short_zones.zone & (~long_zones.zone | (dom_short > dom_long))
    flat_action = np.full(n, int(Action.SKIP), dtype=np.int8)
    flat_action[open_long] = int(Action.OPEN_LONG)
    flat_action[open_short] = int(Action.OPEN_SHORT)
    close_long = np.zeros(n, dtype=bool)
    close_short = np.zeros(n, dtype=bool)
    horizons_ticks = np.maximum(1, np.rint(np.asarray(horizons_seconds) / cadence_s).astype(np.int64))
    candidates: list[tuple[int, PositionSide, object]] = []
    for event in long_zones.events:
        if np.any(open_long[event.zone_start:event.zone_end]):
            entry = event.zone_start + int(np.flatnonzero(open_long[event.zone_start:event.zone_end])[0])
            candidates.append((entry, PositionSide.LONG, event))
    for event in short_zones.events:
        if np.any(open_short[event.zone_start:event.zone_end]):
            entry = event.zone_start + int(np.flatnonzero(open_short[event.zone_start:event.zone_end])[0])
            candidates.append((entry, PositionSide.SHORT, event))
    candidates.sort(key=lambda row: row[0])
    episodes: list[OracleEpisode] = []
    for entry, side, event_obj in candidates:
        event = event_obj
        hidx = int(dom_long_h[event.peak] if side == PositionSide.LONG else dom_short_h[event.peak])
        backward = b_long[:, hidx] if side == PositionSide.LONG else b_short[:, hidx]
        segment = frames.segment_id[entry]
        segment_end = entry + 1
        while segment_end < n and frames.segment_id[segment_end] == segment and frames.valid[segment_end]:
            segment_end += 1
        selected = _selected_close_zone(
            backward,
            entry=entry,
            segment_end=segment_end,
            horizon_ticks=int(horizons_ticks[hidx]),
            ratio=peak_ratio,
            search_multiplier=search_multiplier,
        )
        if selected is None:
            flat_action[event.zone_start:event.zone_end] = int(Action.SKIP)
            continue
        close_start, close_end, close_peak, close_value = selected
        target = close_long if side == PositionSide.LONG else close_short
        target[close_start:close_end] = True
        episodes.append(
            OracleEpisode(
                side=side,
                entry_idx=entry,
                forward_peak_idx=event.peak,
                exit_peak_idx=close_peak,
                close_zone_start=close_start,
                close_zone_end=close_end,
                horizon_idx=hidx,
                forward_peak_score=float(event.peak_score),
                backward_peak_score=close_value,
            )
        )
    return LabelBuildResult(
        flat_action=flat_action,
        open_long_zone=open_long,
        open_short_zone=open_short,
        close_long_zone=close_long,
        close_short_zone=close_short,
        dominant_long_horizon=dom_long_h,
        dominant_short_horizon=dom_short_h,
        normalized_forward_long=dom_long,
        normalized_forward_short=dom_short,
        episodes=tuple(episodes),
        high_long=high_long,
        high_short=high_short,
    )
