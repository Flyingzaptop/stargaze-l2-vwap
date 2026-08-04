from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..contracts import Action, CausalFrames, PositionSide
from ..labels import LabelBuildResult


POSITION_FEATURE_NAMES = (
    "position_age_log_seconds",
    "unrealized_bps",
    "mfe_bps",
    "mae_bps",
    "giveback_bps",
    "entry_forward_long",
    "entry_forward_short",
    "entry_horizon_fraction",
)


@dataclass
class RobustNormalizer:
    center: np.ndarray
    scale: np.ndarray
    clip: float = 20.0

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray, *, clip: float = 20.0) -> "RobustNormalizer":
        array = np.asarray(values, dtype=np.float64)
        rows = array[np.asarray(mask, dtype=bool)]
        if rows.ndim > 2:
            rows = rows.reshape(-1, rows.shape[-1])
        center = np.nanmedian(rows, axis=0)
        q25 = np.nanquantile(rows, 0.25, axis=0)
        q75 = np.nanquantile(rows, 0.75, axis=0)
        scale = (q75 - q25) / 1.349
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        center = np.nan_to_num(center, nan=0.0)
        return cls(center.astype(np.float32), scale.astype(np.float32), float(clip))

    def transform(self, values: np.ndarray) -> np.ndarray:
        out = (np.asarray(values, dtype=np.float32) - self.center) / self.scale
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=self.clip, neginf=-self.clip), -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"center": self.center.tolist(), "scale": self.scale.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RobustNormalizer":
        return cls(np.asarray(payload["center"], dtype=np.float32), np.asarray(payload["scale"], dtype=np.float32), float(payload["clip"]))


@dataclass(frozen=True)
class ExampleTable:
    center_idx: np.ndarray
    state: np.ndarray
    action: np.ndarray
    entry_idx: np.ndarray
    horizon_idx: np.ndarray
    weight: np.ndarray

    def select(self, mask: np.ndarray) -> "ExampleTable":
        keep = np.asarray(mask, dtype=bool)[self.center_idx]
        return ExampleTable(*(getattr(self, name)[keep] for name in self.__dataclass_fields__))


def _segment_starts(segment_id: np.ndarray) -> np.ndarray:
    starts = np.zeros(len(segment_id), dtype=np.int64)
    current = 0
    for i in range(len(segment_id)):
        if i == 0 or segment_id[i] != segment_id[i - 1]:
            current = i
        starts[i] = current
    return starts


def build_examples(
    frames: CausalFrames,
    labels: LabelBuildResult,
    *,
    context_ticks: int,
    skip_stride: int = 1,
    hold_stride: int = 1,
) -> ExampleTable:
    segment_start = _segment_starts(frames.segment_id)
    centers: list[int] = []
    states: list[int] = []
    actions: list[int] = []
    entries: list[int] = []
    horizons: list[int] = []
    weights: list[float] = []
    for idx in np.flatnonzero(frames.valid):
        action = int(labels.flat_action[idx])
        if action == int(Action.SKIP) and idx % max(1, int(skip_stride)) != 0:
            continue
        if idx - int(segment_start[idx]) + 1 < int(context_ticks):
            continue
        if action == int(Action.OPEN_LONG):
            horizon = int(labels.dominant_long_horizon[idx])
            weight = 2.0
        elif action == int(Action.OPEN_SHORT):
            horizon = int(labels.dominant_short_horizon[idx])
            weight = 2.0
        else:
            horizon = int(labels.dominant_long_horizon[idx] if labels.normalized_forward_long[idx] >= labels.normalized_forward_short[idx] else labels.dominant_short_horizon[idx])
            weight = 1.0
        centers.append(int(idx)); states.append(int(PositionSide.FLAT)); actions.append(action); entries.append(-1); horizons.append(horizon); weights.append(weight)
    for episode in labels.episodes:
        close_action = Action.CLOSE_LONG if episode.side == PositionSide.LONG else Action.CLOSE_SHORT
        hold_action = Action.HOLD_LONG if episode.side == PositionSide.LONG else Action.HOLD_SHORT
        for idx in range(episode.entry_idx + 1, episode.close_zone_end):
            close = episode.close_zone_start <= idx < episode.close_zone_end
            if not close and (idx - episode.entry_idx) % max(1, int(hold_stride)) != 0:
                continue
            if idx - int(segment_start[idx]) + 1 < int(context_ticks):
                continue
            centers.append(idx); states.append(int(episode.side)); actions.append(int(close_action if close else hold_action)); entries.append(episode.entry_idx); horizons.append(episode.horizon_idx); weights.append(2.0 if close else 1.0)
    order = np.argsort(np.asarray(centers), kind="stable")
    return ExampleTable(
        center_idx=np.asarray(centers, dtype=np.int64)[order],
        state=np.asarray(states, dtype=np.int8)[order],
        action=np.asarray(actions, dtype=np.int8)[order],
        entry_idx=np.asarray(entries, dtype=np.int64)[order],
        horizon_idx=np.asarray(horizons, dtype=np.int16)[order],
        weight=np.asarray(weights, dtype=np.float32)[order],
    )


def consensus_bid_ask(frames: CausalFrames) -> tuple[np.ndarray, np.ndarray]:
    bid = np.nanmedian(frames.bid, axis=1)
    ask = np.nanmedian(frames.ask, axis=1)
    return bid, ask


class PolicyWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        frames: CausalFrames,
        labels: LabelBuildResult,
        examples: ExampleTable,
        *,
        forward_long_h: np.ndarray,
        forward_short_h: np.ndarray,
        backward_long_h: np.ndarray,
        backward_short_h: np.ndarray,
        horizons_seconds: tuple[float, ...],
        context_ticks: int,
        x_normalizer: RobustNormalizer,
        venue_normalizer: RobustNormalizer,
    ) -> None:
        self.frames = frames
        self.labels = labels
        self.examples = examples
        raw_forward_long = np.asarray(forward_long_h, dtype=np.float32)
        raw_forward_short = np.asarray(forward_short_h, dtype=np.float32)
        raw_backward_long = np.asarray(backward_long_h, dtype=np.float32)
        raw_backward_short = np.asarray(backward_short_h, dtype=np.float32)
        self.forward_valid = np.isfinite(raw_forward_long) & np.isfinite(raw_forward_short)
        self.backward_valid = np.isfinite(raw_backward_long) & np.isfinite(raw_backward_short)
        self.forward_long_h = np.log1p(np.maximum(np.nan_to_num(raw_forward_long, nan=0.0), 0.0))
        self.forward_short_h = np.log1p(np.maximum(np.nan_to_num(raw_forward_short, nan=0.0), 0.0))
        self.backward_long_h = np.log1p(np.maximum(np.nan_to_num(raw_backward_long, nan=0.0), 0.0))
        self.backward_short_h = np.log1p(np.maximum(np.nan_to_num(raw_backward_short, nan=0.0), 0.0))
        self.context = int(context_ticks)
        self.x = x_normalizer.transform(frames.x)
        self.venue_x = venue_normalizer.transform(frames.venue_x)
        self.raw_venue_x = frames.venue_x
        self.bid, self.ask = consensus_bid_ask(frames)
        self.cadence_s = float(np.median(np.diff(frames.ts_ns))) / 1e9 if len(frames.ts_ns) > 1 else 0.1
        self.num_horizons = self.forward_long_h.shape[1]
        if len(horizons_seconds) != self.num_horizons:
            raise ValueError("horizons_seconds must align with score columns")
        trade_idx = frames.venue_feature_names.index("trade_signed_log_qty")
        signed_flow = np.sum(frames.venue_x[:, :, trade_idx], axis=1, dtype=np.float64)
        add_idx = frames.feature_names.index("l3_add_qty_log")
        delete_idx = frames.feature_names.index("l3_delete_qty_log")
        l3_depletion = frames.x[:, delete_idx].astype(np.float64) - frames.x[:, add_idx].astype(np.float64)
        self.future_flow, self.future_valid = _future_horizon_means(frames, signed_flow, horizons_seconds)
        self.future_liquidity, liquidity_valid = _future_horizon_means(frames, l3_depletion, horizons_seconds)
        self.future_valid &= liquidity_valid

    @property
    def input_dim(self) -> int:
        return self.x.shape[1] + 3 * self.num_horizons + len(POSITION_FEATURE_NAMES)

    @property
    def venue_feature_dim(self) -> int:
        return self.venue_x.shape[-1]

    def __len__(self) -> int:
        return len(self.examples.center_idx)

    def _position_features(self, start: int, end: int, side: int, entry: int, horizon_idx: int) -> tuple[np.ndarray, np.ndarray]:
        length = end - start
        features = np.zeros((length, len(POSITION_FEATURE_NAMES)), dtype=np.float32)
        states = np.full(length, int(PositionSide.FLAT), dtype=np.int64)
        if side == int(PositionSide.FLAT) or entry < 0:
            return features, states
        # The entry action is emitted while the pre-action state is FLAT.
        # Position state and position-derived features become observable on the
        # following decision tick, matching the streaming policy replay.
        active_start = max(start, entry + 1)
        if active_start >= end:
            return features, states
        states[active_start - start :] = side
        entry_price = float(self.ask[entry] if side == int(PositionSide.LONG) else self.bid[entry])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            return features, states
        exits = self.bid[active_start:end] if side == int(PositionSide.LONG) else self.ask[active_start:end]
        if side == int(PositionSide.LONG):
            pnl = 1e4 * (exits / entry_price - 1.0)
        else:
            pnl = 1e4 * (entry_price / exits - 1.0)
        pnl = np.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0)
        mfe = np.maximum.accumulate(pnl)
        mae = np.minimum.accumulate(pnl)
        age = np.arange(active_start - entry, end - entry, dtype=np.float32) * self.cadence_s
        offset = active_start - start
        features[offset:, 0] = np.log1p(age)
        features[offset:, 1] = pnl
        features[offset:, 2] = mfe
        features[offset:, 3] = mae
        features[offset:, 4] = mfe - pnl
        features[offset:, 5] = float(self.labels.normalized_forward_long[entry])
        features[offset:, 6] = float(self.labels.normalized_forward_short[entry])
        features[offset:, 7] = float(horizon_idx) / max(1, self.num_horizons - 1)
        return features, states

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        center = int(self.examples.center_idx[item])
        start = center - self.context + 1
        end = center + 1
        side = int(self.examples.state[item])
        position, states = self._position_features(start, end, side, int(self.examples.entry_idx[item]), int(self.examples.horizon_idx[item]))
        base = np.concatenate(
            (
                self.x[start:end],
                self.backward_long_h[start:end],
                self.backward_short_h[start:end],
                self.backward_valid[start:end].astype(np.float32),
                position,
            ),
            axis=1,
        )
        venue_mask = (self.raw_venue_x[start:end, :, 0] > 0.5) & (self.raw_venue_x[start:end, :, 1] <= 2_000.0)
        return {
            "x": torch.from_numpy(base),
            "venue_x": torch.from_numpy(self.venue_x[start:end]),
            "venue_mask": torch.from_numpy(venue_mask),
            "position_state": torch.from_numpy(states),
            "action": torch.tensor(int(self.examples.action[item]), dtype=torch.long),
            "forward_long": torch.from_numpy(self.forward_long_h[center]),
            "forward_short": torch.from_numpy(self.forward_short_h[center]),
            "forward_valid": torch.from_numpy(self.forward_valid[center]),
            "horizon": torch.tensor(int(self.examples.horizon_idx[item]), dtype=torch.long),
            "future_flow": torch.from_numpy(self.future_flow[center]),
            "future_liquidity": torch.from_numpy(self.future_liquidity[center]),
            "future_valid": torch.from_numpy(self.future_valid[center]),
            "weight": torch.tensor(float(self.examples.weight[item]), dtype=torch.float32),
            "center_idx": torch.tensor(center, dtype=torch.long),
        }


def _future_horizon_means(frames: CausalFrames, signal: np.ndarray, horizons_seconds: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    n = len(frames.ts_ns)
    cadence_s = float(np.median(np.diff(frames.ts_ns))) / 1e9 if n > 1 else 0.1
    prefix = np.r_[0.0, np.cumsum(np.nan_to_num(signal, nan=0.0), dtype=np.float64)]
    output = np.zeros((n, len(horizons_seconds)), dtype=np.float32)
    valid_output = np.zeros((n, len(horizons_seconds)), dtype=bool)
    rows = np.arange(n, dtype=np.int64)
    invalid_prefix = np.r_[0, np.cumsum(~frames.valid, dtype=np.int64)]
    for column, horizon in enumerate(horizons_seconds):
        steps = max(1, int(np.floor(float(horizon) / cadence_s + 1e-9)))
        ends = rows + steps
        complete = ends < n
        same_segment = np.zeros(n, dtype=bool)
        same_segment[complete] = frames.segment_id[rows[complete]] == frames.segment_id[ends[complete]]
        window_valid = np.zeros(n, dtype=bool)
        window_valid[complete] = (invalid_prefix[ends[complete] + 1] - invalid_prefix[rows[complete] + 1]) == 0
        usable = complete & same_segment & window_valid
        output[usable, column] = ((prefix[ends[usable] + 1] - prefix[rows[usable] + 1]) / float(steps)).astype(np.float32)
        valid_output[usable, column] = True
    return output, valid_output
