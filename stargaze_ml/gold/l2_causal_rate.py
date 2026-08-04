"""Causal score-quantile controller for a bounded daily trade rate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


DAY_NS = 86_400_000_000_000


@dataclass(frozen=True)
class CausalRateConfig:
    target_trades_per_day: int = 20
    history_size: int = 2_000
    min_history: int = 100


def direction_and_score(
    row: dict[str, float | int], *, mode: str, penalty: float, filter_field: str
) -> tuple[int, float, float]:
    if mode == "classifier":
        side = 1 if float(row["side_probability"]) >= 0.5 else -1
    elif mode == "value":
        side = 1 if float(row["predicted_long_pnl"]) >= float(row["predicted_short_pnl"]) else -1
    elif mode == "risk":
        long_score = float(row["predicted_long_pnl"]) - penalty * float(row["long_tail_probability"])
        short_score = float(row["predicted_short_pnl"]) - penalty * float(row["short_tail_probability"])
        side = 1 if long_score >= short_score else -1
    else:
        raise ValueError("mode must be classifier, value, or risk")

    tail = float(row["long_tail_probability"] if side > 0 else row["short_tail_probability"])
    predicted = float(row["predicted_long_pnl"] if side > 0 else row["predicted_short_pnl"])
    scores = {
        "opportunity_probability": float(row["opportunity_probability"]),
        "negative_tail_probability": -tail,
        "risk_edge": predicted - penalty * tail,
    }
    return side, scores[filter_field], tail


def causal_rate_select(
    rows: list[dict[str, float | int]],
    *,
    mode: str,
    penalty: float,
    filter_field: str,
    expected_candidates_per_day: float,
    fallback_cutoff: float,
    config: CausalRateConfig,
    initial_scores: list[float] | None = None,
) -> list[dict[str, float | int]]:
    """Select using only scores observed strictly before each decision."""
    if expected_candidates_per_day <= 0:
        raise ValueError("expected_candidates_per_day must be positive")
    quantile = float(np.clip(1.0 - config.target_trades_per_day / expected_candidates_per_day, 0.0, 1.0))
    history: deque[float] = deque(initial_scores or (), maxlen=config.history_size)
    selected: list[dict[str, float | int]] = []
    current_day: int | None = None
    daily_count = 0

    for row in sorted(rows, key=lambda item: int(item["entry_ts_ns"])):
        day = int(row["entry_ts_ns"]) // DAY_NS
        if day != current_day:
            current_day = day
            daily_count = 0
        side, score, tail = direction_and_score(
            row, mode=mode, penalty=penalty, filter_field=filter_field
        )
        cutoff = (
            float(np.quantile(np.asarray(history, dtype=np.float64), quantile))
            if len(history) >= config.min_history
            else fallback_cutoff
        )
        accepted = daily_count < config.target_trades_per_day and score >= cutoff
        history.append(score)
        if accepted:
            enriched = dict(row)
            enriched.update({"selected_side": side, "selection_score": score, "tail_probability": tail})
            selected.append(enriched)
            daily_count += 1
    return selected


def summarize_selected(rows: list[dict[str, float | int]]) -> dict[str, float | int | dict[str, int]]:
    if not rows:
        return {"trades": 0, "mean_pnl_ticks": 0.0, "total_pnl_ticks": 0.0, "win_rate": 0.0}
    pnl = np.asarray([
        float(row["long_pnl"] if int(row["selected_side"]) > 0 else row["short_pnl"])
        for row in rows
    ])
    days: dict[str, int] = {}
    for row in rows:
        key = str(int(row["entry_ts_ns"]) // DAY_NS)
        days[key] = days.get(key, 0) + 1
    return {
        "trades": int(len(pnl)),
        "mean_pnl_ticks": float(pnl.mean()),
        "median_pnl_ticks": float(np.median(pnl)),
        "total_pnl_ticks": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "p05_pnl_ticks": float(np.quantile(pnl, 0.05)),
        "standard_error_ticks": float(pnl.std(ddof=1) / np.sqrt(len(pnl))) if len(pnl) > 1 else 0.0,
        "trades_by_day": days,
    }
