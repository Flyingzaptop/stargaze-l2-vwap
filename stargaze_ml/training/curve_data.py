from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import CausalFrames, VENUE_INDEX
from ..features.state import VENUE_FEATURE_NAMES
from ..labels import FourCurveTargets
from .data import RobustNormalizer


def stationary_market_features(frames: CausalFrames) -> tuple[np.ndarray, np.ndarray]:
    """Return price-level invariant causal inputs for the curve model.

    Raw exchange prices make a short training sample identify a market regime by
    BTC's absolute price.  Relative quotes retain the cross-venue information the
    model needs while remaining meaningful when the outright price changes.
    """

    venue = np.asarray(frames.venue_x, dtype=np.float32).copy()
    venue_width = venue.shape[1] * venue.shape[2]
    base = np.asarray(frames.x[:, venue_width:], dtype=np.float32).copy()
    reference = np.asarray(base[:, 0], dtype=np.float64)
    reference_valid = np.isfinite(reference) & (reference > 0.0)

    def relative_bps(values: np.ndarray) -> np.ndarray:
        values64 = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values64) & (values64 > 0.0) & reference_valid[:, None]
        result = np.zeros_like(values64)
        np.divide(values64, reference[:, None], out=result, where=valid)
        result[valid] = 1e4 * (result[valid] - 1.0)
        result[~valid] = 0.0
        return result.astype(np.float32)

    # Venue bid/ask/mid become contemporaneous dislocations from consensus.
    venue[:, :, 2:5] = relative_bps(venue[:, :, 2:5].reshape(len(reference), -1)).reshape(
        venue.shape[0], venue.shape[1], 3
    )

    # Keep the three global slots stable: reference availability plus spot and
    # derivative dislocations.  All operations use only the current frame.
    spot_derivative = relative_bps(base[:, 1:3])
    base[:, 0] = reference_valid.astype(np.float32)
    base[:, 1:3] = spot_derivative
    return base, venue


def _causal_rolling_mean(values: np.ndarray, segment_id: np.ndarray, window: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    output = np.zeros_like(matrix)
    starts = np.r_[0, np.flatnonzero(segment_id[1:] != segment_id[:-1]) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(matrix)], strict=True):
        part = matrix[start:end]
        cumulative = np.vstack((np.zeros((1, part.shape[1])), np.cumsum(part, axis=0)))
        rows = np.arange(len(part), dtype=np.int64)
        left = np.maximum(0, rows - int(window) + 1)
        count = (rows - left + 1)[:, None]
        output[start:end] = (cumulative[rows + 1] - cumulative[left]) / count
    return output.astype(np.float32)


def causal_high_order_features(frames: CausalFrames) -> tuple[np.ndarray, tuple[str, ...]]:
    """Causal rolling market-state family that reduces temporal sample complexity."""

    base, venue = stationary_market_features(frames)
    n = len(frames.ts_ns)
    reference = np.asarray(
        frames.x[:, frames.venue_x.shape[1] * frames.venue_x.shape[2]], dtype=np.float64
    )
    one_step = np.zeros(n, dtype=np.float64)
    usable_step = (
        (reference[1:] > 0.0)
        & (reference[:-1] > 0.0)
        & (frames.segment_id[1:] == frames.segment_id[:-1])
    )
    one_step[1:][usable_step] = 1e4 * np.log(reference[1:][usable_step] / reference[:-1][usable_step])
    cadence_seconds = float(np.median(np.diff(frames.ts_ns))) / 1e9

    def ticks(seconds: int) -> int:
        return max(1, int(round(float(seconds) / cadence_seconds)))

    columns: list[np.ndarray] = []
    names: list[str] = []
    momentum_seconds = (1, 2, 5, 10, 15, 20, 30, 60, 120, 240)
    for seconds in momentum_seconds:
        lag = ticks(seconds)
        momentum = np.zeros(n, dtype=np.float32)
        if lag < n:
            usable = (
                (reference[lag:] > 0.0)
                & (reference[:-lag] > 0.0)
                & (frames.segment_id[lag:] == frames.segment_id[:-lag])
            )
            values = np.zeros(n - lag, dtype=np.float64)
            values[usable] = 1e4 * np.log(reference[lag:][usable] / reference[:-lag][usable])
            momentum[lag:] = values.astype(np.float32)
        columns.append(momentum[:, None])
        names.append(f"momentum_{seconds}s_bps")
    squared = np.square(one_step)
    absolute = np.abs(one_step)
    for seconds in (5, 15, 30, 60, 120, 240):
        window = ticks(seconds)
        variance = _causal_rolling_mean(squared, frames.segment_id, window)[:, 0]
        path = _causal_rolling_mean(absolute, frames.segment_id, window)[:, 0] * float(window)
        volatility = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        net = columns[momentum_seconds.index(seconds)][:, 0]
        efficiency = (np.abs(net) / np.maximum(path, 1e-6)).astype(np.float32)
        columns.extend((volatility[:, None], efficiency[:, None]))
        names.extend((f"realized_volatility_{seconds}s_bps", f"trend_efficiency_{seconds}s"))

    fresh = (frames.venue_x[:, :, 0] > 0.5) & (frames.venue_x[:, :, 1] <= 2_000.0)
    selected_names = (
        "spread_bps", "microprice_delta_bps", "best_bid_log_qty", "best_ask_log_qty",
        *(f"imbalance_{depth}" for depth in (1, 5, 10, 25, 50, 100, 250, 1000)),
        "bid_slope_10_bps", "ask_slope_10_bps", "trade_count", "trade_buy_log_qty",
        "trade_sell_log_qty", "trade_signed_log_qty", "trade_vwap_delta_bps",
        "mark_delta_bps", "index_delta_bps", "open_interest_log", "funding_rate_bps",
        "long_liquidation_log_qty", "short_liquidation_log_qty",
    )
    selected_indices = [VENUE_FEATURE_NAMES.index(name) for name in selected_names]
    selected = venue[:, :, selected_indices].astype(np.float64)
    mask = fresh[:, :, None]
    count = np.maximum(mask.sum(axis=1), 1)
    mean = (np.where(mask, selected, 0.0).sum(axis=1) / count).astype(np.float32)
    variance = (np.where(mask, (selected - mean[:, None, :]) ** 2, 0.0).sum(axis=1) / count)
    std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    columns.extend((mean, std))
    names.extend(tuple(f"venue_mean_{name}" for name in selected_names))
    names.extend(tuple(f"venue_std_{name}" for name in selected_names))

    directional_names = (
        "microprice_delta_bps",
        *(f"imbalance_{depth}" for depth in (1, 5, 10, 25, 50, 100, 250, 1000)),
        "trade_signed_log_qty", "trade_vwap_delta_bps", "mark_delta_bps",
        "funding_rate_bps", "long_liquidation_log_qty", "short_liquidation_log_qty",
    )
    directional_columns = [selected_names.index(name) for name in directional_names]
    directional = mean[:, directional_columns]
    for seconds in (5, 15, 60, 240):
        window = ticks(seconds)
        rolling = _causal_rolling_mean(directional, frames.segment_id, window)
        columns.append(rolling)
        names.extend(tuple(f"venue_mean_{name}_rolling_{seconds}s" for name in directional_names))

    relative_mid = venue[:, :, 4].astype(np.float64)
    relative_count = np.maximum(fresh.sum(axis=1), 1)
    relative_mean = np.where(fresh, relative_mid, 0.0).sum(axis=1) / relative_count
    relative_variance = np.where(
        fresh, (relative_mid - relative_mean[:, None]) ** 2, 0.0
    ).sum(axis=1) / relative_count
    has_venue = fresh.any(axis=1)
    relative_min = np.min(np.where(fresh, relative_mid, np.inf), axis=1)
    relative_max = np.max(np.where(fresh, relative_mid, -np.inf), axis=1)
    relative_min[~has_venue] = 0.0
    relative_max[~has_venue] = 0.0
    columns.extend(
        (
            relative_mean.astype(np.float32)[:, None],
            np.sqrt(np.maximum(relative_variance, 0.0)).astype(np.float32)[:, None],
            relative_min.clip(-1e4, 1e4).astype(np.float32)[:, None],
            relative_max.clip(-1e4, 1e4).astype(np.float32)[:, None],
        )
    )
    names.extend(("venue_mid_mean_bps", "venue_mid_std_bps", "venue_mid_min_bps", "venue_mid_max_bps"))

    l3 = base[:, 12:21]
    l3_names = (
        "active_orders", "bid_qty", "ask_qty", "imbalance", "adds", "deletes",
        "modifies", "add_qty", "delete_qty",
    )
    for seconds in (5, 15, 60):
        columns.append(_causal_rolling_mean(l3, frames.segment_id, ticks(seconds)))
        names.extend(tuple(f"l3_{name}_rolling_{seconds}s" for name in l3_names))
    output = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    if output.shape[1] != len(names):
        raise RuntimeError("high-order feature names do not align")
    return output, tuple(names)


def causal_backward_score_features(
    frames: CausalFrames,
    horizons_seconds: tuple[float, ...],
    *,
    cost_bps: float,
) -> np.ndarray:
    """Build multihorizon realized-edge features using present and past quotes only."""

    horizons = np.asarray(horizons_seconds, dtype=np.float64)
    if horizons.ndim != 1 or horizons.size == 0 or np.any(horizons <= 0.0):
        raise ValueError("horizons_seconds must be a non-empty positive vector")
    cadence_ns = int(np.median(np.diff(frames.ts_ns)))
    steps = np.maximum(1, np.rint(horizons * 1e9 / cadence_ns).astype(np.int64))
    venue = VENUE_INDEX["binance_perpetual"]
    bid = np.asarray(frames.bid[:, venue], dtype=np.float64)
    ask = np.asarray(frames.ask[:, venue], dtype=np.float64)
    quote_valid = (
        (frames.venue_x[:, venue, 0] > 0.5)
        & (frames.venue_x[:, venue, 1] <= 2_000.0)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & (bid < ask)
    )
    n = len(frames.ts_ns)
    long_sum = np.zeros(n, dtype=np.float64)
    short_sum = np.zeros(n, dtype=np.float64)
    good = np.zeros(n, dtype=np.int32)
    long_h = np.zeros((n, len(steps)), dtype=np.float32)
    short_h = np.zeros_like(long_h)
    valid_h = np.zeros_like(long_h)
    by_step: dict[int, list[int]] = {}
    for column, step in enumerate(steps):
        by_step.setdefault(int(step), []).append(column)
    for lag in range(1, min(int(steps.max()), n - 1) + 1):
        count = n - lag
        pair_valid = (
            quote_valid[:count]
            & quote_valid[lag:]
            & (frames.segment_id[:count] == frames.segment_id[lag:])
        )
        long_ratio = np.ones(count, dtype=np.float64)
        short_ratio = np.ones(count, dtype=np.float64)
        np.divide(bid[lag:], ask[:count], out=long_ratio, where=pair_valid)
        np.divide(bid[:count], ask[lag:], out=short_ratio, where=pair_valid)
        long_sum[lag:] += np.where(pair_valid, np.maximum(1e4 * (long_ratio - 1.0) - float(cost_bps), 0.0), 0.0)
        short_sum[lag:] += np.where(pair_valid, np.maximum(1e4 * (short_ratio - 1.0) - float(cost_bps), 0.0), 0.0)
        good[lag:] += pair_valid
        for column in by_step.get(lag, ()):
            required = max(1, int(np.ceil(0.95 * lag)))
            same_horizon_segment = np.zeros(n, dtype=bool)
            same_horizon_segment[lag:] = frames.segment_id[lag:] == frames.segment_id[:-lag]
            usable = (good >= required) & same_horizon_segment
            long_h[usable, column] = (long_sum[usable] / good[usable]).astype(np.float32)
            short_h[usable, column] = (short_sum[usable] / good[usable]).astype(np.float32)
            valid_h[usable, column] = 1.0
    return np.concatenate((long_h, short_h, valid_h), axis=1)


def multihorizon_forward_edge_targets(
    frames: CausalFrames,
    horizons_seconds: tuple[float, ...],
    *,
    cost_bps: float,
    scale_bps: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense training-only future long/short execution edges for every horizon."""

    horizons = np.asarray(horizons_seconds, dtype=np.float64)
    if horizons.ndim != 1 or horizons.size == 0 or np.any(horizons <= 0.0):
        raise ValueError("horizons_seconds must be a non-empty positive vector")
    if not np.isfinite(scale_bps) or scale_bps <= 0.0:
        raise ValueError("scale_bps must be positive")
    cadence_ns = int(np.median(np.diff(frames.ts_ns)))
    steps = np.maximum(1, np.rint(horizons * 1e9 / cadence_ns).astype(np.int64))
    venue = VENUE_INDEX["binance_perpetual"]
    bid = np.asarray(frames.bid[:, venue], dtype=np.float64)
    ask = np.asarray(frames.ask[:, venue], dtype=np.float64)
    quote_valid = (
        (frames.venue_x[:, venue, 0] > 0.5)
        & (frames.venue_x[:, venue, 1] <= 2_000.0)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & (bid < ask)
    )
    target = np.zeros((len(frames.ts_ns), 2 * len(steps)), dtype=np.float32)
    valid = np.zeros_like(target, dtype=bool)
    for column, step_raw in enumerate(steps):
        step = int(step_raw)
        count = len(frames.ts_ns) - step
        if count <= 0:
            continue
        usable = (
            quote_valid[:count]
            & quote_valid[step:]
            & (frames.segment_id[:count] == frames.segment_id[step:])
        )
        long_edge = np.zeros(count, dtype=np.float64)
        short_edge = np.zeros(count, dtype=np.float64)
        np.divide(bid[step:], ask[:count], out=long_edge, where=usable)
        np.divide(bid[:count], ask[step:], out=short_edge, where=usable)
        long_edge = (1e4 * (long_edge - 1.0) - float(cost_bps)) / float(scale_bps)
        short_edge = (1e4 * (short_edge - 1.0) - float(cost_bps)) / float(scale_bps)
        target[:count, column] = np.clip(long_edge, -8.0, 8.0).astype(np.float32)
        target[:count, len(steps) + column] = np.clip(short_edge, -8.0, 8.0).astype(np.float32)
        valid[:count, column] = usable
        valid[:count, len(steps) + column] = usable
    target[~valid] = 0.0
    return target, valid


def causal_score_features_for_normalizer(
    frames: CausalFrames,
    normalizer: RobustNormalizer,
    horizons_seconds: tuple[float, ...],
    *,
    cost_bps: float,
) -> np.ndarray | None:
    venue_width = frames.venue_x.shape[1] * frames.venue_x.shape[2]
    global_width = frames.x.shape[1] - venue_width
    expected_extra = len(normalizer.center) - global_width
    if expected_extra == 0:
        return None
    backward = causal_backward_score_features(frames, horizons_seconds, cost_bps=cost_bps)
    if expected_extra == backward.shape[1]:
        return backward
    high_order, _ = causal_high_order_features(frames)
    if expected_extra == backward.shape[1] + high_order.shape[1]:
        return np.concatenate((backward, high_order), axis=1)
    raise ValueError("base normalizer does not match a supported causal feature schema")


def _normalized_base_features(
    frames: CausalFrames,
    normalizer: RobustNormalizer,
    causal_score_x: np.ndarray | None,
    *,
    global_x: np.ndarray | None = None,
) -> np.ndarray:
    if global_x is None:
        venue_width = frames.venue_x.shape[1] * frames.venue_x.shape[2]
        global_x = frames.x[:, venue_width:]
    raw = global_x if causal_score_x is None else np.concatenate((global_x, causal_score_x), axis=1)
    if raw.shape[1] != len(normalizer.center):
        raise ValueError("base feature matrix does not match the fitted normalizer")
    return normalizer.transform(raw)


def venue_normalizer_is_stationary(normalizer: RobustNormalizer | None) -> bool:
    """New bundles use per-venue centers; old shared-normalizer bundles stay loadable."""

    return normalizer is not None and np.asarray(normalizer.center).ndim == 2


def curve_centers(
    frames: CausalFrames,
    targets: FourCurveTargets,
    split_mask: np.ndarray,
    *,
    context_ticks: int,
    background_stride: int,
) -> np.ndarray:
    usable = np.asarray(split_mask, dtype=bool) & np.all(targets.valid, axis=1) & frames.valid
    starts = np.r_[0, np.flatnonzero(frames.segment_id[1:] != frames.segment_id[:-1]) + 1]
    segment_start = np.zeros(len(frames.ts_ns), dtype=np.int64)
    for start, end in zip(starts, np.r_[starts[1:], len(frames.ts_ns)], strict=True):
        segment_start[start:end] = start
    rows = np.flatnonzero(usable & ((np.arange(len(usable)) - segment_start + 1) >= int(context_ticks)))
    positive = np.max(targets.values[rows], axis=1) >= 0.05
    background = (rows % max(1, int(background_stride))) == 0
    return rows[positive | background].astype(np.int64, copy=False)


def causal_centers(
    frames: CausalFrames,
    split_mask: np.ndarray,
    *,
    context_ticks: int,
) -> np.ndarray:
    """Return inference rows using only present/past frame state."""

    usable = np.asarray(split_mask, dtype=bool) & frames.valid
    starts = np.r_[0, np.flatnonzero(frames.segment_id[1:] != frames.segment_id[:-1]) + 1]
    segment_start = np.zeros(len(frames.ts_ns), dtype=np.int64)
    for start, end in zip(starts, np.r_[starts[1:], len(frames.ts_ns)], strict=True):
        segment_start[start:end] = start
    enough_context = (np.arange(len(usable)) - segment_start + 1) >= int(context_ticks)
    return np.flatnonzero(usable & enough_context).astype(np.int64, copy=False)


class CurveWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        frames: CausalFrames,
        targets: FourCurveTargets,
        centers: np.ndarray,
        *,
        context_ticks: int,
        base_normalizer: RobustNormalizer,
        venue_normalizer: RobustNormalizer,
        causal_score_x: np.ndarray | None = None,
        supervision_ticks: int = 1,
        auxiliary_target: np.ndarray | None = None,
        auxiliary_valid: np.ndarray | None = None,
    ) -> None:
        self.frames = frames
        self.targets = targets
        self.centers = np.asarray(centers, dtype=np.int64)
        self.context = int(context_ticks)
        self.supervision = min(self.context, max(1, int(supervision_ticks)))
        self.auxiliary_target = None if auxiliary_target is None else np.asarray(auxiliary_target, dtype=np.float32)
        self.auxiliary_valid = None if auxiliary_valid is None else np.asarray(auxiliary_valid, dtype=bool)
        if (self.auxiliary_target is None) != (self.auxiliary_valid is None):
            raise ValueError("auxiliary target and validity must be supplied together")
        if self.auxiliary_target is not None and (
            self.auxiliary_target.shape != self.auxiliary_valid.shape
            or self.auxiliary_target.shape[0] != len(frames.ts_ns)
        ):
            raise ValueError("auxiliary arrays must align with frames")
        stationary = venue_normalizer_is_stationary(venue_normalizer)
        if stationary:
            global_x, raw_venue = stationary_market_features(frames)
        else:
            global_x, raw_venue = None, frames.venue_x
        seconds = frames.ts_ns.astype(np.float64) / 1e9
        hour_phase = 2.0 * np.pi * ((seconds % 86_400.0) / 86_400.0)
        week_phase = 2.0 * np.pi * ((seconds % 604_800.0) / 604_800.0)
        time_x = np.column_stack((np.sin(hour_phase), np.cos(hour_phase), np.sin(week_phase), np.cos(week_phase))).astype(np.float32)
        self.base_x = np.concatenate(
            (
                _normalized_base_features(
                    frames, base_normalizer, causal_score_x, global_x=global_x
                ),
                time_x,
            ),
            axis=1,
        )
        self.venue_x = venue_normalizer.transform(raw_venue)

    @property
    def input_dim(self) -> int:
        return self.base_x.shape[1]

    @property
    def venue_feature_dim(self) -> int:
        return self.venue_x.shape[-1]

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        center = int(self.centers[item])
        start = center - self.context + 1
        raw_venue = self.frames.venue_x[start : center + 1]
        venue_mask = (raw_venue[:, :, 0] > 0.5) & (raw_venue[:, :, 1] <= 2_000.0)
        target = self.targets.values[center]
        supervision_start = center - self.supervision + 1
        target_seq = self.targets.values[supervision_start : center + 1]
        result = {
            "x": torch.from_numpy(self.base_x[start : center + 1]),
            "venue_x": torch.from_numpy(self.venue_x[start : center + 1]),
            "venue_mask": torch.from_numpy(venue_mask),
            "target": torch.from_numpy(target),
            "valid": torch.from_numpy(self.targets.valid[center]),
            "weight": torch.from_numpy((1.0 + 4.0 * target).astype(np.float32, copy=False)),
            "target_seq": torch.from_numpy(target_seq),
            "valid_seq": torch.from_numpy(self.targets.valid[supervision_start : center + 1]),
            "weight_seq": torch.from_numpy((1.0 + 4.0 * target_seq).astype(np.float32, copy=False)),
            "center_idx": torch.tensor(center, dtype=torch.long),
        }
        if self.auxiliary_target is not None and self.auxiliary_valid is not None:
            result["auxiliary_target"] = torch.from_numpy(
                self.auxiliary_target[supervision_start : center + 1]
            )
            result["auxiliary_valid"] = torch.from_numpy(
                self.auxiliary_valid[supervision_start : center + 1]
            )
        return result


class CurveInferenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal model inputs with no label or future-validity dependency."""

    def __init__(
        self,
        frames: CausalFrames,
        centers: np.ndarray,
        *,
        context_ticks: int,
        base_normalizer: RobustNormalizer,
        venue_normalizer: RobustNormalizer,
        causal_score_x: np.ndarray | None = None,
    ) -> None:
        self.frames = frames
        self.centers = np.asarray(centers, dtype=np.int64)
        self.context = int(context_ticks)
        stationary = venue_normalizer_is_stationary(venue_normalizer)
        if stationary:
            global_x, raw_venue = stationary_market_features(frames)
        else:
            global_x, raw_venue = None, frames.venue_x
        seconds = frames.ts_ns.astype(np.float64) / 1e9
        hour_phase = 2.0 * np.pi * ((seconds % 86_400.0) / 86_400.0)
        week_phase = 2.0 * np.pi * ((seconds % 604_800.0) / 604_800.0)
        time_x = np.column_stack(
            (np.sin(hour_phase), np.cos(hour_phase), np.sin(week_phase), np.cos(week_phase))
        ).astype(np.float32)
        self.base_x = np.concatenate(
            (
                _normalized_base_features(
                    frames, base_normalizer, causal_score_x, global_x=global_x
                ),
                time_x,
            ),
            axis=1,
        )
        self.venue_x = venue_normalizer.transform(raw_venue)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        center = int(self.centers[item])
        start = center - self.context + 1
        raw_venue = self.frames.venue_x[start : center + 1]
        venue_mask = (raw_venue[:, :, 0] > 0.5) & (raw_venue[:, :, 1] <= 2_000.0)
        return {
            "x": torch.from_numpy(self.base_x[start : center + 1]),
            "venue_x": torch.from_numpy(self.venue_x[start : center + 1]),
            "venue_mask": torch.from_numpy(venue_mask),
            "center_idx": torch.tensor(center, dtype=torch.long),
        }
