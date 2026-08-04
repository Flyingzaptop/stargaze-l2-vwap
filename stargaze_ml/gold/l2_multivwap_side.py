"""Algorithmic side selection from local and higher-horizon causal quote-VWAPs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from stargaze_ml.training.data import RobustNormalizer
from .l2_open_policy import L2OpenPolicy
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from .l2_seconds import _weighted_causal_average


HIGHER_HORIZONS = (90, 120, 300, 600, 900)


@dataclass(frozen=True)
class SideRule:
    horizons: tuple[int, ...]
    neutral_ticks: float
    consensus: float
    live_swap: bool
    exit_crossing: int
    confirmation_seconds: int = 3


def build_mid_vwaps(
    seconds: pl.DataFrame, horizons: tuple[int, ...] = (60, *HIGHER_HORIZONS)
) -> dict[int, np.ndarray]:
    observed = seconds["observed"].to_numpy().astype(bool)
    segment = seconds["segment_id"].to_numpy().astype(np.int32)
    bid = seconds["last_bid"].to_numpy().astype(np.float64)
    ask = seconds["last_ask"].to_numpy().astype(np.float64)
    bid_weight = np.where(observed, seconds["bid_size_top1"].to_numpy(), 0.0)
    ask_weight = np.where(observed, seconds["ask_size_top1"].to_numpy(), 0.0)
    result: dict[int, np.ndarray] = {}
    for horizon in horizons:
        bid_vwap = _weighted_causal_average(bid, bid_weight, segment, window=int(horizon))
        ask_vwap = _weighted_causal_average(ask, ask_weight, segment, window=int(horizon))
        result[int(horizon)] = (bid_vwap + ask_vwap) * 0.5
    return result


def multivwap_side(
    price: float,
    local_vwap: float,
    higher_vwaps: np.ndarray,
    *,
    tick_size: float,
    neutral_ticks: float,
    consensus: float,
) -> tuple[int, bool, float]:
    """Return desired position (+1 long/-1 short), inversion and vote strength."""
    local_side = 1 if float(local_vwap) > float(price) else -1
    gaps = (np.asarray(higher_vwaps, dtype=np.float64) - float(price)) / float(tick_size)
    active = np.abs(gaps) >= float(neutral_ticks)
    if not np.any(active):
        return local_side, False, 0.0
    score = float(np.sign(gaps[active]).mean())
    strength = abs(score)
    if score == 0.0 or strength < float(consensus):
        return local_side, False, strength
    global_side = 1 if score > 0 else -1
    inverted = global_side != local_side
    return (global_side if inverted else local_side), inverted, strength


def _open_entries(
    model: L2OpenPolicy,
    data: PreparedOpenData,
    normalizer: RobustNormalizer,
    events: np.ndarray,
    threshold: float,
    device: torch.device,
) -> dict[int, int]:
    entries: dict[int, int] = {}; model.eval()
    with torch.no_grad():
        for event in events:
            start = int(data.event_start[event]); crossing = int(data.event_crossing_1[event])
            x = normalizer.transform(data.x[start:crossing])[None]
            probability = torch.sigmoid(model(torch.from_numpy(x).to(device)))[0].cpu().numpy()
            allowed = (
                data.gate_open[start:crossing]
                & data.valid_feature[start:crossing]
                & data.observed[start + 1 : crossing + 1]
            )
            hit = np.flatnonzero(allowed & (probability >= float(threshold)))
            if hit.size:
                entries[int(event)] = start + int(hit[0])
    return entries


def _execute_entry(data: PreparedOpenData, decision: int, side: int, cost: float, tick_size: float) -> float:
    execution = int(decision) + 1
    return (
        -(data.first_ask[execution] / tick_size + cost)
        if int(side) > 0
        else data.first_bid[execution] / tick_size - cost
    )


def _simulate_rule(
    data: PreparedOpenData,
    event: int,
    entry: int,
    mid_vwaps: dict[int, np.ndarray],
    rule: SideRule | None,
    config: OpenReinforceConfig,
) -> tuple[float, int, bool]:
    price = data.mid
    local = mid_vwaps[60]
    if rule is None or not rule.horizons:
        side = 1 if local[entry] > price[entry] else -1
        inverted = False
        exit_crossing = 1 if rule is None else int(rule.exit_crossing)
    else:
        higher = np.asarray([mid_vwaps[h][entry] for h in rule.horizons])
        side, inverted, _ = multivwap_side(
            price[entry], local[entry], higher, tick_size=config.tick_size,
            neutral_ticks=rule.neutral_ticks, consensus=rule.consensus,
        )
        exit_crossing = int(rule.exit_crossing)
    cost = config.commission_per_fill_ticks + config.slippage_per_fill_ticks
    cash = _execute_entry(data, entry, side, cost, config.tick_size)
    swaps = 0; evidence_side = 0; evidence_count = 0
    crossing = int(data.event_crossing_1[event]) if exit_crossing == 1 else int(data.event_crossing_2[event])
    if rule is not None and rule.live_swap:
        for decision in range(int(entry) + 1, crossing):
            desired, _, strength = multivwap_side(
                price[decision], local[decision],
                np.asarray([mid_vwaps[h][decision] for h in rule.horizons]),
                tick_size=config.tick_size, neutral_ticks=rule.neutral_ticks,
                consensus=rule.consensus,
            )
            if desired != side and strength >= rule.consensus and data.observed[decision + 1]:
                if desired == evidence_side:
                    evidence_count += 1
                else:
                    evidence_side = desired; evidence_count = 1
                if evidence_count >= int(rule.confirmation_seconds):
                    execution = decision + 1
                    if side > 0:
                        cash += 2.0 * (data.first_bid[execution] / config.tick_size - cost)
                    else:
                        cash -= 2.0 * (data.first_ask[execution] / config.tick_size + cost)
                    side = desired; swaps += 1; evidence_side = 0; evidence_count = 0
            else:
                evidence_side = 0; evidence_count = 0
    exit_execution = crossing + 1
    if side > 0:
        cash += data.first_bid[exit_execution] / config.tick_size - cost
    else:
        cash -= data.first_ask[exit_execution] / config.tick_size + cost
    return float(cash), int(swaps), bool(inverted)


def _summarize(
    data: PreparedOpenData,
    entries: dict[int, int],
    mid_vwaps: dict[int, np.ndarray],
    rule: SideRule | None,
    config: OpenReinforceConfig,
) -> dict[str, object]:
    pnls=[]; swaps=[]; inverted=[]
    for event, entry in entries.items():
        pnl, swap_count, was_inverted = _simulate_rule(data, event, entry, mid_vwaps, rule, config)
        pnls.append(pnl); swaps.append(swap_count); inverted.append(was_inverted)
    values=np.asarray(pnls); swap_values=np.asarray(swaps)
    return {
        "rule": "local60" if rule is None else asdict(rule),
        "trades": int(len(values)),
        "mean_pnl_ticks": float(values.mean()) if len(values) else 0.0,
        "median_pnl_ticks": float(np.median(values)) if len(values) else 0.0,
        "win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "p05_pnl_ticks": float(np.quantile(values, 0.05)) if len(values) else 0.0,
        "total_pnl_ticks": float(values.sum()),
        "entry_inversion_fraction": float(np.mean(inverted)) if inverted else 0.0,
        "mean_swaps": float(swap_values.mean()) if len(swap_values) else 0.0,
        "swapped_trade_fraction": float((swap_values > 0).mean()) if len(swap_values) else 0.0,
    }


def run_multivwap_side_experiment(
    prepared_path: str | Path,
    seconds_path: str | Path,
    open_checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, object]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    data = PreparedOpenData(prepared_path)
    checkpoint = torch.load(Path(open_checkpoint_path).resolve(strict=True), map_location=device, weights_only=False)
    open_config = OpenReinforceConfig(**checkpoint["config"])
    normalizer = RobustNormalizer.from_dict(checkpoint["normalizer"])
    model = L2OpenPolicy(len(data.feature_names), open_config.hidden_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["validation"]["best"]["threshold"])
    seconds = pl.read_parquet(Path(seconds_path).resolve(strict=True))
    mid_vwaps = build_mid_vwaps(seconds)
    split_events = {
        "validation": _event_indices(data, data.train_end, data.validation_end, good_only=False, exit_crossing="both"),
        "test": _event_indices(data, data.validation_end, len(data.x), good_only=False, exit_crossing="both"),
    }
    split_entries = {
        name: _open_entries(model, data, normalizer, events, threshold, device)
        for name, events in split_events.items()
    }
    rules = [
        SideRule(horizons, neutral, consensus, live, exit_crossing)
        for horizons in (
            (90, 120), (90, 120, 300), (90, 120, 300, 600), HIGHER_HORIZONS,
        )
        for neutral in (0.0, 10.0, 25.0, 50.0)
        for consensus in (0.50, 0.67, 0.80, 1.0)
        for live in (False, True)
        for exit_crossing in (1, 2)
    ]
    results: dict[str, list[dict[str, object]]] = {}
    for split in ("validation", "test"):
        rows = []
        for exit_crossing in (1, 2):
            local_rule = SideRule((), 0.0, 1.0, False, exit_crossing)
            # Empty higher horizons preserve the pure local-60 baseline.
            baseline = _summarize(data, split_entries[split], mid_vwaps, local_rule, open_config)
            baseline["rule"] = asdict(local_rule)
            baseline["rule"]["name"] = "local60"
            rows.append(baseline)
        rows.extend(_summarize(data, split_entries[split], mid_vwaps, rule, open_config) for rule in rules)
        results[split] = rows
    best = max(results["validation"], key=lambda row: float(row["mean_pnl_ticks"]))
    key = json.dumps(best["rule"], sort_keys=True)
    fixed_test = next(row for row in results["test"] if json.dumps(row["rule"], sort_keys=True) == key)
    report = {
        "open_model_unchanged": str(Path(open_checkpoint_path).resolve()),
        "open_threshold": threshold,
        "side_logic": "local60 mean-reversion; invert only when higher-horizon vote disagrees with sufficient consensus",
        "validation_selected": best,
        "fixed_test": fixed_test,
        "validation": results["validation"],
        "test": results["test"],
    }
    out=Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out/"mid_vwaps.npz", **{f"mid_vwap_{h}s": values for h, values in mid_vwaps.items()})
    return report
