from __future__ import annotations

from stargaze_ml.gold.l2_causal_rate import CausalRateConfig, causal_rate_select


def _row(second: int, score: float) -> dict[str, float | int]:
    return {
        "entry_ts_ns": second * 1_000_000_000,
        "long_pnl": 1.0, "short_pnl": -1.0,
        "side_probability": 0.9,
        "predicted_long_pnl": score, "predicted_short_pnl": 0.0,
        "long_tail_probability": 0.1, "short_tail_probability": 0.2,
        "opportunity_probability": score,
    }


def test_selector_is_unchanged_when_only_future_scores_change() -> None:
    prefix = [_row(i, i / 100.0) for i in range(150)]
    first = causal_rate_select(
        prefix + [_row(151, 0.0)], mode="classifier", penalty=0.0,
        filter_field="opportunity_probability", expected_candidates_per_day=100.0,
        fallback_cutoff=0.5, config=CausalRateConfig(target_trades_per_day=20),
    )
    second = causal_rate_select(
        prefix + [_row(151, 100.0)], mode="classifier", penalty=0.0,
        filter_field="opportunity_probability", expected_candidates_per_day=100.0,
        fallback_cutoff=0.5, config=CausalRateConfig(target_trades_per_day=20),
    )
    assert [row["entry_ts_ns"] for row in first if int(row["entry_ts_ns"]) < 151_000_000_000] == [
        row["entry_ts_ns"] for row in second if int(row["entry_ts_ns"]) < 151_000_000_000
    ]


def test_selector_enforces_daily_cap() -> None:
    rows = [_row(i, 1.0) for i in range(100)]
    selected = causal_rate_select(
        rows, mode="classifier", penalty=0.0,
        filter_field="opportunity_probability", expected_candidates_per_day=100.0,
        fallback_cutoff=0.0, config=CausalRateConfig(target_trades_per_day=10),
    )
    assert len(selected) == 10
