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


class CausalRateController:
    """Stateful online form of the causal daily-rate selector."""

    def __init__(
        self,
        *,
        mode: str,
        penalty: float,
        filter_field: str,
        expected_candidates_per_day: float,
        fallback_cutoff: float,
        config: CausalRateConfig,
        initial_scores: list[float] | None = None,
    ) -> None:
        if expected_candidates_per_day <= 0:
            raise ValueError("expected_candidates_per_day must be positive")
        self.mode = str(mode)
        self.penalty = float(penalty)
        self.filter_field = str(filter_field)
        self.fallback_cutoff = float(fallback_cutoff)
        self.config = config
        self.quantile = float(
            np.clip(
                1.0 - config.target_trades_per_day / expected_candidates_per_day,
                0.0,
                1.0,
            )
        )
        self.history: deque[float] = deque(initial_scores or (), maxlen=config.history_size)
        self.current_day: int | None = None
        self.daily_count = 0

    def consider(self, row: dict[str, float | int]) -> dict[str, float | int] | None:
        day = int(row["entry_ts_ns"]) // DAY_NS
        if day != self.current_day:
            self.current_day = day
            self.daily_count = 0
        side, score, tail = direction_and_score(
            row,
            mode=self.mode,
            penalty=self.penalty,
            filter_field=self.filter_field,
        )
        cutoff = (
            float(np.quantile(np.asarray(self.history, dtype=np.float64), self.quantile))
            if len(self.history) >= self.config.min_history
            else self.fallback_cutoff
        )
        accepted = self.daily_count < self.config.target_trades_per_day and score >= cutoff
        self.history.append(score)
        if not accepted:
            return None
        enriched = dict(row)
        enriched.update(
            {
                "selected_side": side,
                "selection_score": score,
                "tail_probability": tail,
                "causal_cutoff": cutoff,
            }
        )
        self.daily_count += 1
        return enriched


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
    long_risk = float(row["predicted_long_pnl"]) - penalty * float(row["long_tail_probability"])
    short_risk = float(row["predicted_short_pnl"]) - penalty * float(row["short_tail_probability"])
    side_std = float(row.get("side_probability_std", 0.0))
    chosen_tail_std = float(
        row.get("long_tail_probability_std", 0.0)
        if side > 0 else row.get("short_tail_probability_std", 0.0)
    )
    risk_uncertainty = float(np.sqrt(
        float(row.get("predicted_long_pnl_std", 0.0)) ** 2
        + float(row.get("predicted_short_pnl_std", 0.0)) ** 2
        + penalty ** 2 * (
            float(row.get("long_tail_probability_std", 0.0)) ** 2
            + float(row.get("short_tail_probability_std", 0.0)) ** 2
        )
    ))
    scores = {
        "opportunity_probability": float(row["opportunity_probability"]),
        "negative_tail_probability": -tail,
        "risk_edge": predicted - penalty * tail,
        "side_confidence": abs(float(row["side_probability"]) - 0.5),
        "value_gap": abs(
            float(row["predicted_long_pnl"]) - float(row["predicted_short_pnl"])
        ),
        "classifier_value_agreement": float(
            (float(row["side_probability"]) >= 0.5)
            == (
                float(row["predicted_long_pnl"])
                >= float(row["predicted_short_pnl"])
            )
        ),
        "risk_direction_margin": abs(long_risk - short_risk),
        "negative_side_disagreement": -side_std,
        "negative_tail_disagreement": -chosen_tail_std,
        "negative_risk_uncertainty": -risk_uncertainty,
        "risk_evidence": abs(long_risk - short_risk) / (1.0 + risk_uncertainty),
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
    controller = CausalRateController(
        mode=mode,
        penalty=penalty,
        filter_field=filter_field,
        expected_candidates_per_day=expected_candidates_per_day,
        fallback_cutoff=fallback_cutoff,
        config=config,
        initial_scores=initial_scores,
    )
    selected: list[dict[str, float | int]] = []
    for row in sorted(rows, key=lambda item: int(item["entry_ts_ns"])):
        accepted = controller.consider(row)
        if accepted is not None:
            selected.append(accepted)
    return selected


def summarize_selected(rows: list[dict[str, float | int]]) -> dict[str, float | int | dict[str, int]]:
    if not rows:
        return {"trades": 0, "mean_pnl_ticks": 0.0, "total_pnl_ticks": 0.0, "win_rate": 0.0}
    pnl = np.asarray([
        float(
            row["realized_pnl"]
            if "realized_pnl" in row
            else row["long_pnl"] if int(row["selected_side"]) > 0
            else row["short_pnl"]
        )
        for row in rows
    ])
    p05 = float(np.quantile(pnl, 0.05))
    cvar05 = float(pnl[pnl <= p05].mean())
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
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
        "p05_pnl_ticks": p05,
        "cvar05_pnl_ticks": cvar05,
        "worst_pnl_ticks": float(pnl.min()),
        "profit_factor": gains / losses if losses > 0 else float("inf"),
        "standard_error_ticks": float(pnl.std(ddof=1) / np.sqrt(len(pnl))) if len(pnl) > 1 else 0.0,
        "trades_by_day": days,
    }


def robust_validation_score(metrics: dict[str, object]) -> float:
    """Conservative PnL objective for small, heavy-tailed validation samples."""
    return (
        float(metrics["mean_pnl_ticks"])
        + 0.10 * float(metrics.get("cvar05_pnl_ticks", 0.0))
        - 0.50 * float(metrics.get("standard_error_ticks", 0.0))
    )


def chronological_robust_validation(
    rows: list[dict[str, float | int]], *, split_ts_ns: int,
    min_trades_per_half: int = 5,
) -> dict[str, object]:
    """Score a policy by its weakest chronological validation half."""

    full = summarize_selected(rows)
    first = summarize_selected(
        [row for row in rows if int(row["entry_ts_ns"]) < int(split_ts_ns)]
    )
    second = summarize_selected(
        [row for row in rows if int(row["entry_ts_ns"]) >= int(split_ts_ns)]
    )
    if min_trades_per_half < 1:
        raise ValueError("min_trades_per_half must be positive")
    scores = {
        "full": robust_validation_score(full),
        "first_half": robust_validation_score(first),
        "second_half": robust_validation_score(second),
    }
    coverage_valid = (
        int(first["trades"]) >= min_trades_per_half
        and int(second["trades"]) >= min_trades_per_half
    )
    selection_score = float(min(scores.values())) if coverage_valid else -1_000_000.0
    return {
        "selection_score": selection_score,
        "coverage_valid": coverage_valid,
        "min_trades_per_half": int(min_trades_per_half),
        "robust_scores": scores,
        "first_half": first,
        "second_half": second,
    }
