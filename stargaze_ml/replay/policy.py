from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..contracts import Action, CausalFrames, PolicyLedger, PositionSide
from ..models import HierarchicalCausalTransformerPolicy
from ..training.data import POSITION_FEATURE_NAMES, RobustNormalizer, consensus_bid_ask


@dataclass
class PolicyReplayResult:
    ledger: PolicyLedger
    trades: tuple[dict[str, object], ...]
    final_state: PositionSide


def _position_row(
    idx: int,
    *,
    side: PositionSide,
    entry_idx: int,
    entry_price: float,
    bid: np.ndarray,
    ask: np.ndarray,
    cadence_s: float,
    running_mfe: float,
    running_mae: float,
    entry_forward_long: float,
    entry_forward_short: float,
    horizon_idx: int,
    num_horizons: int,
) -> tuple[np.ndarray, float, float, float]:
    values = np.zeros(len(POSITION_FEATURE_NAMES), dtype=np.float32)
    if side == PositionSide.FLAT:
        return values, running_mfe, running_mae, 0.0
    exit_price = float(bid[idx] if side == PositionSide.LONG else ask[idx])
    if not np.isfinite(exit_price) or not np.isfinite(entry_price) or min(exit_price, entry_price) <= 0.0:
        pnl = 0.0
    elif side == PositionSide.LONG:
        pnl = 1e4 * (exit_price / entry_price - 1.0)
    else:
        pnl = 1e4 * (entry_price / exit_price - 1.0)
    running_mfe = max(running_mfe, pnl)
    running_mae = min(running_mae, pnl)
    values[:] = (
        np.log1p(max(0.0, (idx - entry_idx) * cadence_s)),
        pnl,
        running_mfe,
        running_mae,
        running_mfe - pnl,
        entry_forward_long,
        entry_forward_short,
        float(horizon_idx) / max(1, num_horizons - 1),
    )
    return values, running_mfe, running_mae, pnl


@torch.no_grad()
def replay_policy(
    model: HierarchicalCausalTransformerPolicy,
    frames: CausalFrames,
    *,
    x_normalizer: RobustNormalizer,
    venue_normalizer: RobustNormalizer,
    backward_long_h: np.ndarray,
    backward_short_h: np.ndarray,
    context_ticks: int,
    device: str | None = None,
) -> PolicyReplayResult:
    target_device = torch.device(device or next(model.parameters()).device)
    model = model.to(target_device).eval()
    x = x_normalizer.transform(frames.x)
    backward_long = np.log1p(np.maximum(np.nan_to_num(np.asarray(backward_long_h, dtype=np.float32), nan=0.0), 0.0))
    backward_short = np.log1p(np.maximum(np.nan_to_num(np.asarray(backward_short_h, dtype=np.float32), nan=0.0), 0.0))
    backward_valid = np.isfinite(np.asarray(backward_long_h)) & np.isfinite(np.asarray(backward_short_h))
    if backward_long.shape != backward_short.shape or backward_long.shape != (len(frames.ts_ns), model.config.num_horizons):
        raise ValueError("backward score features must have shape [time, model.config.num_horizons]")
    venue_x = venue_normalizer.transform(frames.venue_x)
    venue_mask = (frames.venue_x[:, :, 0] > 0.5) & (frames.venue_x[:, :, 1] <= 2_000.0)
    bid, ask = consensus_bid_ask(frames)
    cadence_s = float(np.median(np.diff(frames.ts_ns))) / 1e9 if len(frames.ts_ns) > 1 else 0.1
    n = len(frames.ts_ns)
    augmented = np.zeros((n, x.shape[1] + 3 * model.config.num_horizons + len(POSITION_FEATURE_NAMES)), dtype=np.float32)
    states = np.zeros(n, dtype=np.int8)
    actions = np.zeros(n, dtype=np.int8)
    states_after = np.zeros(n, dtype=np.int8)
    probabilities = np.zeros((n, len(Action)), dtype=np.float32)
    ages = np.zeros(n, dtype=np.int32)
    state = PositionSide.FLAT
    entry_idx = -1
    entry_price = np.nan
    entry_forward_long = 0.0
    entry_forward_short = 0.0
    entry_horizon = 0
    running_mfe = 0.0
    running_mae = 0.0
    trades: list[dict[str, object]] = []
    for idx in range(n):
        states[idx] = int(state)
        position, running_mfe, running_mae, unrealized = _position_row(
            idx,
            side=state,
            entry_idx=entry_idx,
            entry_price=entry_price,
            bid=bid,
            ask=ask,
            cadence_s=cadence_s,
            running_mfe=running_mfe,
            running_mae=running_mae,
            entry_forward_long=entry_forward_long,
            entry_forward_short=entry_forward_short,
            horizon_idx=entry_horizon,
            num_horizons=model.config.num_horizons,
        )
        augmented[idx] = np.concatenate((x[idx], backward_long[idx], backward_short[idx], backward_valid[idx].astype(np.float32), position))
        ages[idx] = max(0, idx - entry_idx) if entry_idx >= 0 else 0
        start = max(0, idx - int(context_ticks) + 1)
        base_tensor = torch.from_numpy(augmented[start : idx + 1]).unsqueeze(0).to(target_device)
        venue_tensor = torch.from_numpy(venue_x[start : idx + 1]).unsqueeze(0).to(target_device)
        mask_tensor = torch.from_numpy(venue_mask[start : idx + 1]).unsqueeze(0).to(target_device)
        state_tensor = torch.from_numpy(states[start : idx + 1].astype(np.int64)).unsqueeze(0).to(target_device)
        output = model(base_tensor, state_tensor, venue_tensor, mask_tensor)
        logits = output.action_logits[0, -1]
        probability = torch.softmax(logits, dim=-1).cpu().numpy()
        action = Action(int(output.actions[0, -1]))
        probabilities[idx] = probability
        actions[idx] = int(action)
        if state == PositionSide.FLAT and action in {Action.OPEN_LONG, Action.OPEN_SHORT}:
            state = PositionSide.LONG if action == Action.OPEN_LONG else PositionSide.SHORT
            entry_idx = idx
            entry_price = float(ask[idx] if state == PositionSide.LONG else bid[idx])
            entry_forward_long = float(torch.expm1(output.forward_long[0, -1].mean()).cpu())
            entry_forward_short = float(torch.expm1(output.forward_short[0, -1].mean()).cpu())
            entry_horizon = int(torch.argmax(output.horizon_logits[0, -1]).cpu())
            running_mfe = running_mae = 0.0
        elif state == PositionSide.LONG and action == Action.CLOSE_LONG:
            trades.append({"side": "long", "entry_idx": entry_idx, "exit_idx": idx, "entry_ts_ns": int(frames.ts_ns[entry_idx]), "exit_ts_ns": int(frames.ts_ns[idx]), "hold_seconds": (idx - entry_idx) * cadence_s, "entry_price": entry_price, "exit_price": float(bid[idx]), "pnl_bps": unrealized, "mfe_bps": running_mfe, "mae_bps": running_mae})
            state = PositionSide.FLAT; entry_idx = -1
        elif state == PositionSide.SHORT and action == Action.CLOSE_SHORT:
            trades.append({"side": "short", "entry_idx": entry_idx, "exit_idx": idx, "entry_ts_ns": int(frames.ts_ns[entry_idx]), "exit_ts_ns": int(frames.ts_ns[idx]), "hold_seconds": (idx - entry_idx) * cadence_s, "entry_price": entry_price, "exit_price": float(ask[idx]), "pnl_bps": unrealized, "mfe_bps": running_mfe, "mae_bps": running_mae})
            state = PositionSide.FLAT; entry_idx = -1
        states_after[idx] = int(state)
    ledger = PolicyLedger(frames.ts_ns.copy(), states, actions, states_after, probabilities, ages)
    return PolicyReplayResult(ledger, tuple(trades), state)
