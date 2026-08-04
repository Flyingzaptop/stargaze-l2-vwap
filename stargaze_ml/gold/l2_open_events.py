"""Causal multi-horizon quote-VWAP features and VWAP excursions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .l2_seconds import _weighted_causal_average, build_l2_second_feature_matrix
from .l2_adaptive_gate import causal_adaptive_gate


VWAP_HORIZONS_SECONDS = (5, 10, 15, 30, 45, 60, 120, 300, 900)
VWAP_RIBBON_HORIZONS_SECONDS = (5, 10, 15, 30, 45, 60)


@dataclass(frozen=True)
class ExcursionTable:
    start: np.ndarray
    end: np.ndarray
    crossing_1: np.ndarray
    crossing_2: np.ndarray
    side: np.ndarray
    duration_seconds: np.ndarray
    amplitude_ticks: np.ndarray
    gate_index: np.ndarray
    gated: np.ndarray
    good: np.ndarray


@dataclass(frozen=True)
class OpenPolicyData:
    x: np.ndarray
    feature_names: tuple[str, ...]
    valid_feature: np.ndarray
    mid: np.ndarray
    primary_vwap: np.ndarray
    side: np.ndarray
    event_id: np.ndarray
    gate_open: np.ndarray
    excursions: ExcursionTable


def _carry_nonzero_sign(delta: np.ndarray, segment_id: np.ndarray) -> np.ndarray:
    raw = np.sign(delta).astype(np.int8)
    out = raw.copy()
    previous = np.int8(0)
    previous_segment = None
    for i in range(len(out)):
        segment = int(segment_id[i])
        if segment != previous_segment:
            previous = np.int8(0)
            previous_segment = segment
        if out[i] == 0:
            out[i] = previous
        else:
            previous = out[i]
    return out


def _causal_rolling_correlation(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    segment_id: np.ndarray,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal rolling correlation and y-on-x beta inside segments."""

    weight = np.asarray(valid, dtype=np.float64)
    mean_x = _weighted_causal_average(x, weight, segment_id, window=window)
    mean_y = _weighted_causal_average(y, weight, segment_id, window=window)
    mean_xy = _weighted_causal_average(x * y, weight, segment_id, window=window)
    mean_x2 = _weighted_causal_average(x * x, weight, segment_id, window=window)
    mean_y2 = _weighted_causal_average(y * y, weight, segment_id, window=window)
    covariance = mean_xy - mean_x * mean_y
    variance_x = np.maximum(mean_x2 - mean_x * mean_x, 0.0)
    variance_y = np.maximum(mean_y2 - mean_y * mean_y, 0.0)
    denominator = np.sqrt(variance_x * variance_y)
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12
    )
    beta = np.divide(
        covariance, variance_x, out=np.zeros_like(covariance), where=variance_x > 1e-12
    )
    return correlation, beta


def build_open_policy_data(
    seconds: pl.DataFrame,
    *,
    tick_size: float = 0.01,
    amplitude_threshold_ticks: float = 250.0,
    gate_fraction: float = 0.75,
    min_duration_seconds: int = 30,
    horizons: tuple[int, ...] = VWAP_HORIZONS_SECONDS,
    primary_vwap: int | str = 60,
    feature_profile: str = "leadlag",
    adaptive_gate_target_per_active_day: int | None = None,
) -> OpenPolicyData:
    """Build causal inputs and completed excursion metadata.

    The primary event line is either one horizon or the equal-weighted causal
    5/10/15/30/45/60-second VWAP ribbon centre.
    A zero delta inherits the most recent non-zero side inside its segment, so
    touching the line alone does not create two artificial events.
    """

    if feature_profile not in {"raw", "hierarchy", "leadlag"}:
        raise ValueError("feature_profile must be raw, hierarchy, or leadlag")
    if 60 not in horizons or tick_size <= 0 or amplitude_threshold_ticks <= 0:
        raise ValueError("60s must be present and thresholds must be positive")
    if isinstance(primary_vwap, str):
        if primary_vwap != "ribbon":
            raise ValueError("primary_vwap must be a horizon or 'ribbon'")
        missing_ribbon = sorted(set(VWAP_RIBBON_HORIZONS_SECONDS) - set(horizons))
        if missing_ribbon:
            raise ValueError(f"ribbon horizons are missing: {missing_ribbon}")
    elif int(primary_vwap) not in horizons:
        raise ValueError("primary_vwap horizon must be present in horizons")
    if not 0 < gate_fraction <= 1 or min_duration_seconds < 1:
        raise ValueError("gate_fraction/min_duration_seconds are invalid")
    required = {
        "bar_start_ns", "segment_id", "last_bid", "last_ask", "bid_size_top1",
        "ask_size_top1", "observed", "first_bid", "first_ask",
    }
    missing = sorted(required - set(seconds.columns))
    if missing:
        raise ValueError(f"L2 seconds are missing columns: {', '.join(missing)}")

    base = build_l2_second_feature_matrix(seconds, tick_size=tick_size)
    n = len(base.ts_ns)
    segment_id = base.segment_id
    bid = seconds["last_bid"].to_numpy().astype(np.float64)
    ask = seconds["last_ask"].to_numpy().astype(np.float64)
    mid = (bid + ask) * 0.5
    observed = seconds["observed"].to_numpy().astype(bool)
    bid_weight = np.where(observed, seconds["bid_size_top1"].to_numpy(), 0.0)
    ask_weight = np.where(observed, seconds["ask_size_top1"].to_numpy(), 0.0)

    extra_values: list[np.ndarray] = []
    extra_names: list[str] = []
    mid_vwaps: dict[int, np.ndarray] = {}
    for horizon in horizons:
        bid_vwap = _weighted_causal_average(
            bid, bid_weight, segment_id, window=int(horizon)
        )
        ask_vwap = _weighted_causal_average(
            ask, ask_weight, segment_id, window=int(horizon)
        )
        mid_vwap = (bid_vwap + ask_vwap) * 0.5
        mid_vwaps[int(horizon)] = mid_vwap
        slope = np.zeros(n, dtype=np.float64)
        same = segment_id[1:] == segment_id[:-1]
        slope[1:] = np.where(same, (mid_vwap[1:] - mid_vwap[:-1]) / tick_size, 0.0)
        if horizon != 60:  # The base matrix already contains these four 60s fields.
            extra_values.extend((
                bid_vwap, ask_vwap,
                (bid_vwap - bid) / tick_size,
                (ask_vwap - ask) / tick_size,
            ))
            extra_names.extend((
                f"bid_vwap_{horizon}s", f"ask_vwap_{horizon}s",
                f"bid_vwap_{horizon}s_minus_bid_ticks",
                f"ask_vwap_{horizon}s_minus_ask_ticks",
            ))
        extra_values.extend(((mid_vwap - mid) / tick_size, slope))
        extra_names.extend((
            f"mid_vwap_{horizon}s_minus_mid_ticks", f"mid_vwap_{horizon}s_slope_1s_ticks",
        ))

    ribbon_stack = np.column_stack(
        [mid_vwaps[horizon] for horizon in VWAP_RIBBON_HORIZONS_SECONDS]
    )
    ribbon_center = ribbon_stack.mean(axis=1)
    ribbon_width = (ribbon_stack.max(axis=1) - ribbon_stack.min(axis=1)) / tick_size
    ribbon_slope = np.zeros(n, dtype=np.float64)
    same = segment_id[1:] == segment_id[:-1]
    ribbon_slope[1:] = np.where(
        same, (ribbon_center[1:] - ribbon_center[:-1]) / tick_size, 0.0
    )
    ribbon_width_delta = np.zeros(n, dtype=np.float64)
    ribbon_width_delta[1:] = np.where(
        same, ribbon_width[1:] - ribbon_width[:-1], 0.0
    )
    ordered_horizons = tuple(sorted(mid_vwaps))
    horizon_stack = np.column_stack([mid_vwaps[h] for h in ordered_horizons])
    gap_stack_ticks = (horizon_stack - mid[:, None]) / tick_size
    log_horizon = np.log(np.asarray(ordered_horizons, dtype=np.float64))
    linear_basis = log_horizon - log_horizon.mean()
    linear_basis /= np.dot(linear_basis, linear_basis)
    quadratic_basis = linear_basis**2
    quadratic_basis -= quadratic_basis.mean()
    quadratic_basis -= (
        np.dot(quadratic_basis, linear_basis)
        / np.dot(linear_basis, linear_basis)
    ) * linear_basis
    quadratic_basis /= np.dot(quadratic_basis, quadratic_basis)
    scale_slope = gap_stack_ticks @ linear_basis
    scale_curvature = gap_stack_ticks @ quadratic_basis
    scale_consensus = np.sign(gap_stack_ticks).mean(axis=1)
    extra_values.extend((
        ribbon_center, (ribbon_center - mid) / tick_size,
        ribbon_width, ribbon_slope,
        (mid_vwaps[5] - mid_vwaps[60]) / tick_size,
    ))
    extra_names.extend((
        "mid_vwap_ribbon_5_60s",
        "mid_vwap_ribbon_5_60s_minus_mid_ticks",
        "mid_vwap_ribbon_width_ticks",
        "mid_vwap_ribbon_slope_1s_ticks",
        "mid_vwap_5s_minus_60s_ticks",
    ))
    if feature_profile in {"hierarchy", "leadlag"}:
        extra_values.extend((
            ribbon_width_delta, scale_slope, scale_curvature, scale_consensus,
        ))
        extra_names.extend((
        "mid_vwap_ribbon_width_delta_1s_ticks",
        "mid_vwap_scale_slope_ticks",
        "mid_vwap_scale_curvature_ticks",
        "mid_vwap_scale_consensus",
        ))
        for short, long in zip(ordered_horizons[:-1], ordered_horizons[1:], strict=True):
            spread = (mid_vwaps[short] - mid_vwaps[long]) / tick_size
            spread_delta = np.zeros(n, dtype=np.float64)
            spread_delta[1:] = np.where(same, spread[1:] - spread[:-1], 0.0)
            extra_values.extend((spread, spread_delta))
            extra_names.extend((
                f"mid_vwap_{short}s_minus_{long}s_ticks",
                f"mid_vwap_{short}s_minus_{long}s_delta_1s_ticks",
            ))

    # Causal lead/lag response.  At row t only returns observed through t are
    # used; the trading decision executes on the next second's BBO.
    if feature_profile == "leadlag":
        price_return = np.zeros(n, dtype=np.float64)
        return_valid = np.zeros(n, dtype=bool)
        price_return[1:] = np.where(same, (mid[1:] - mid[:-1]) / tick_size, 0.0)
        return_valid[1:] = same & observed[1:] & observed[:-1]
        price_lag = np.r_[0.0, price_return[:-1]]
        valid_lag = np.r_[False, return_valid[:-1]]
        for horizon in ordered_horizons:
            vwap_return = np.zeros(n, dtype=np.float64)
            vwap_return[1:] = np.where(
                same, (mid_vwaps[horizon][1:] - mid_vwaps[horizon][:-1]) / tick_size, 0.0
            )
            vwap_lag = np.r_[0.0, vwap_return[:-1]]
            pair_valid = return_valid & valid_lag
            price_leads_corr, vwap_response_beta = _causal_rolling_correlation(
                price_lag, vwap_return, pair_valid, segment_id, window=60
            )
            vwap_leads_corr, price_response_beta = _causal_rolling_correlation(
                vwap_lag, price_return, pair_valid, segment_id, window=60
            )
            extra_values.extend((
                price_leads_corr, vwap_leads_corr,
                price_leads_corr - vwap_leads_corr,
                vwap_response_beta, price_response_beta,
            ))
            extra_names.extend((
                f"price_leads_vwap_{horizon}s_corr_60s",
                f"vwap_{horizon}s_leads_price_corr_60s",
                f"price_vwap_{horizon}s_lead_balance_60s",
                f"vwap_{horizon}s_response_to_price_beta_60s",
                f"price_response_to_vwap_{horizon}s_beta_60s",
            ))

    primary = (
        ribbon_center if primary_vwap == "ribbon" else mid_vwaps[int(primary_vwap)]
    )
    delta_ticks = (mid - primary) / tick_size
    side = _carry_nonzero_sign(delta_ticks, segment_id)
    crossing = np.r_[True, (segment_id[1:] != segment_id[:-1]) | (side[1:] != side[:-1])]
    event_id = np.cumsum(crossing, dtype=np.int64) - 1

    age = np.zeros(n, dtype=np.float64)
    running_max = np.zeros(n, dtype=np.float64)
    running_area = np.zeros(n, dtype=np.float64)
    gate_open = np.zeros(n, dtype=bool)
    event_start = 0
    gate_ticks = float(amplitude_threshold_ticks) * float(gate_fraction)
    for i in range(n):
        if crossing[i]:
            event_start = i
        age[i] = i - event_start
        absolute = abs(delta_ticks[i])
        if i == event_start:
            running_max[i] = absolute
            running_area[i] = absolute
        else:
            running_max[i] = max(running_max[i - 1], absolute)
            running_area[i] = running_area[i - 1] + absolute
        gate_open[i] = running_max[i] >= gate_ticks

    denominator = np.maximum((age + 1.0) * running_max, 1e-6)
    fill_ratio = running_area / denominator
    current_to_max = np.abs(delta_ticks) / np.maximum(running_max, 1e-6)
    geometry = (
        age,
        delta_ticks,
        np.abs(delta_ticks),
        running_max,
        running_area,
        fill_ratio,
        current_to_max,
        side.astype(np.float64),
    )
    geometry_names = (
        "event_age_seconds", "event_signed_delta_ticks", "event_abs_delta_ticks",
        "event_running_max_delta_ticks", "event_running_area_tick_seconds",
        "event_fill_ratio", "event_current_to_max_ratio", "event_side",
    )

    # The first/second exits are the next two crossing decisions.
    all_starts = np.flatnonzero(crossing)
    cross_1 = all_starts[1:-1]
    cross_2 = all_starts[2:]
    starts = all_starts[:-2]
    same_segment = (
        (segment_id[starts] == segment_id[cross_1])
        & (segment_id[starts] == segment_id[cross_2])
    )
    starts = starts[same_segment]
    cross_1 = cross_1[same_segment]
    cross_2 = cross_2[same_segment]
    ends = cross_1 - 1
    duration = (base.ts_ns[cross_1] - base.ts_ns[starts]) / 1e9
    amplitude = np.asarray([running_max[end] for end in ends], dtype=np.float64)
    event_side = side[starts]
    gate_indices = np.full(len(starts), -1, dtype=np.int64)
    for j, (left, right) in enumerate(zip(starts, ends, strict=True)):
        hits = np.flatnonzero(gate_open[left : right + 1])
        if hits.size:
            gate_indices[j] = left + int(hits[0])
    gated = gate_indices >= 0
    if adaptive_gate_target_per_active_day is not None:
        adaptive = causal_adaptive_gate(
            ts_ns=base.ts_ns,
            absolute_delta_ticks=np.abs(delta_ticks),
            event_start=starts,
            event_end=ends,
            completed_amplitude_ticks=amplitude,
            target_gated_events_per_active_day=int(adaptive_gate_target_per_active_day),
            gate_fraction=gate_fraction,
            fallback_amplitude_ticks=amplitude_threshold_ticks,
        )
        gate_open = adaptive.gate_open
        gate_indices = adaptive.gate_index_by_event
        gated = adaptive.gated_by_event
        good = (duration >= float(min_duration_seconds)) & (
            amplitude >= adaptive.amplitude_threshold_by_event
        )
        row_gate = np.full(n, gate_ticks, dtype=np.float64)
        row_amplitude = np.full(n, amplitude_threshold_ticks, dtype=np.float64)
        for event, (left, right) in enumerate(zip(starts, ends, strict=True)):
            row_gate[left : right + 1] = adaptive.gate_threshold_by_event[event]
            row_amplitude[left : right + 1] = adaptive.amplitude_threshold_by_event[event]
        geometry = geometry + (
            row_gate,
            row_amplitude,
            np.abs(delta_ticks) / np.maximum(row_gate, 1e-6),
        )
        geometry_names = geometry_names + (
            "event_adaptive_gate_ticks",
            "event_adaptive_amplitude_ticks",
            "event_delta_to_adaptive_gate_ratio",
        )
    else:
        good = (duration >= float(min_duration_seconds)) & (
            amplitude >= float(amplitude_threshold_ticks)
        )

    x64 = np.column_stack((base.x.astype(np.float64), *extra_values, *geometry))
    valid = base.valid_feature & np.all(np.isfinite(x64), axis=1) & (side != 0)
    x64[~np.isfinite(x64)] = 0.0
    return OpenPolicyData(
        x=x64.astype(np.float32),
        feature_names=base.feature_names + tuple(extra_names) + geometry_names,
        valid_feature=valid,
        mid=mid,
        primary_vwap=primary,
        side=side,
        event_id=event_id,
        gate_open=gate_open,
        excursions=ExcursionTable(
            start=starts.astype(np.int64), end=ends.astype(np.int64),
            crossing_1=cross_1.astype(np.int64), crossing_2=cross_2.astype(np.int64),
            side=event_side.astype(np.int8), duration_seconds=duration.astype(np.float32),
            amplitude_ticks=amplitude.astype(np.float32), gate_index=gate_indices,
            gated=gated, good=good,
        ),
    )
