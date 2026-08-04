"""Causal multi-horizon VWAP evidence for mean-reversion/continuation side."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData


@dataclass(frozen=True)
class DominanceOverrideConfig:
    horizons: tuple[int, ...] = (5, 10, 15, 30, 45, 60, 120)
    alignment_consensus: float = 0.8
    slope_consensus: float = 0.8
    min_projected_slope_ticks: float = 1.0
    min_abs_delta_ticks: float = 0.0
    confirmation_seconds: int = 3
    max_swaps: int = 1


def price_dominance_evidence(
    feature_row: np.ndarray,
    feature_index: dict[str, int],
    config: DominanceOverrideConfig,
) -> dict[str, float | int | bool]:
    local_delta = float(feature_row[feature_index["mid_vwap_60s_minus_mid_ticks"]])
    relation = int(np.sign(local_delta))
    if relation == 0:
        return {"price_dominant": False, "relation": 0, "alignment": 0.0, "slope_consensus": 0.0}
    if abs(local_delta) < config.min_abs_delta_ticks:
        return {"price_dominant": False, "relation": relation, "alignment": 0.0, "slope_consensus": 0.0}
    deltas = np.asarray([
        feature_row[feature_index[f"mid_vwap_{h}s_minus_mid_ticks"]]
        for h in config.horizons
    ], dtype=np.float64)
    slopes = np.asarray([
        feature_row[feature_index[f"mid_vwap_{h}s_slope_1s_ticks"]]
        for h in config.horizons
    ], dtype=np.float64)
    alignment = float(np.mean(np.sign(deltas) == relation))
    projected_slopes = -relation * slopes
    slope_consensus = float(np.mean(projected_slopes >= config.min_projected_slope_ticks))
    dominant = alignment >= config.alignment_consensus and slope_consensus >= config.slope_consensus
    return {
        "price_dominant": bool(dominant),
        "relation": relation,
        "alignment": alignment,
        "slope_consensus": slope_consensus,
        "mean_projected_slope_ticks": float(projected_slopes.mean()),
    }


def apply_dominance_override(
    rows: list[dict[str, float | int]],
    features: np.ndarray,
    feature_names: tuple[str, ...],
    config: DominanceOverrideConfig,
) -> tuple[list[dict[str, float | int]], int]:
    index = {name: position for position, name in enumerate(feature_names)}
    changed = []; swaps = 0
    for row in rows:
        enriched = dict(row)
        entry = int(row["entry_index"])
        evidence = price_dominance_evidence(features[entry], index, config)
        relation = int(evidence["relation"])
        mean_reversion_side = relation
        if bool(evidence["price_dominant"]) and int(row["selected_side"]) == mean_reversion_side:
            enriched["selected_side"] = -mean_reversion_side
            swaps += 1
        enriched["dominance_swapped"] = int(enriched["selected_side"]) != int(row["selected_side"])
        enriched["dominance_alignment"] = float(evidence["alignment"])
        enriched["dominance_slope_consensus"] = float(evidence["slope_consensus"])
        changed.append(enriched)
    return changed, swaps


def simulate_live_dominance_swaps(
    rows: list[dict[str, float | int]],
    data: PreparedOpenData,
    market: OpenReinforceConfig,
    config: DominanceOverrideConfig,
) -> tuple[list[dict[str, float | int]], int]:
    index = {name: position for position, name in enumerate(data.feature_names)}
    cost = market.commission_per_fill_ticks + market.slippage_per_fill_ticks
    simulated = []; total_swaps = 0
    for row in rows:
        entry = int(row["entry_index"]); event = int(row["event_index"])
        crossing = int(data.event_crossing_1[event])
        side = int(row["selected_side"]); initial_side = side
        execution = entry + 1
        cash = (
            -(data.first_ask[execution] / market.tick_size + cost)
            if side > 0 else data.first_bid[execution] / market.tick_size - cost
        )
        swaps = 0; evidence_side = 0; evidence_count = 0; first_swap_delay = -1
        for decision in range(entry + 1, crossing):
            evidence = price_dominance_evidence(data.x[decision], index, config)
            desired = -int(evidence["relation"]) if bool(evidence["price_dominant"]) else 0
            executable = bool(data.valid_feature[decision] and data.observed[decision + 1])
            if executable and desired != 0 and desired != side:
                if desired == evidence_side:
                    evidence_count += 1
                else:
                    evidence_side = desired; evidence_count = 1
                if evidence_count >= config.confirmation_seconds and swaps < config.max_swaps:
                    execution = decision + 1
                    if side > 0:
                        cash += 2.0 * (data.first_bid[execution] / market.tick_size - cost)
                    else:
                        cash -= 2.0 * (data.first_ask[execution] / market.tick_size + cost)
                    side = desired; swaps += 1; total_swaps += 1
                    if first_swap_delay < 0:
                        first_swap_delay = decision - entry
                    evidence_side = 0; evidence_count = 0
                    if swaps >= config.max_swaps:
                        break
            else:
                evidence_side = 0; evidence_count = 0
        exit_execution = crossing + 1
        if side > 0:
            cash += data.first_bid[exit_execution] / market.tick_size - cost
        else:
            cash -= data.first_ask[exit_execution] / market.tick_size + cost
        enriched = dict(row)
        enriched.update({
            "initial_side": initial_side, "final_side": side,
            "live_dominance_swaps": swaps, "first_swap_delay_seconds": first_swap_delay,
            "realized_pnl": float(cash),
        })
        simulated.append(enriched)
    return simulated, total_swaps
