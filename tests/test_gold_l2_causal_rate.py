from __future__ import annotations

from stargaze_ml.gold.l2_causal_rate import (
    CausalRateConfig,
    CausalRateController,
    causal_rate_select,
    chronological_robust_validation,
    direction_and_score,
    robust_validation_score,
    summarize_selected,
)


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


def test_stateful_controller_matches_batch_selector() -> None:
    rows = [_row(i, i / 50.0) for i in range(120)]
    kwargs = dict(
        mode="classifier",
        penalty=0.0,
        filter_field="opportunity_probability",
        expected_candidates_per_day=100.0,
        fallback_cutoff=0.5,
        config=CausalRateConfig(target_trades_per_day=12),
    )
    expected = causal_rate_select(rows, **kwargs)
    controller = CausalRateController(**kwargs)
    actual = [accepted for row in rows if (accepted := controller.consider(row)) is not None]
    assert [row["entry_ts_ns"] for row in actual] == [row["entry_ts_ns"] for row in expected]


def test_summary_and_robust_score_penalize_tail_loss() -> None:
    rows = []
    for second, pnl in enumerate((10.0, 20.0, -100.0)):
        row = _row(second, 1.0)
        row.update({"long_pnl": pnl, "selected_side": 1})
        rows.append(row)
    metrics = summarize_selected(rows)
    assert metrics["worst_pnl_ticks"] == -100.0
    assert metrics["cvar05_pnl_ticks"] == -100.0
    assert robust_validation_score(metrics) < float(metrics["mean_pnl_ticks"])


def test_direction_confidence_scores_are_causal_row_functions() -> None:
    row = _row(0, 1.0)
    row.update({
        "side_probability": 0.8,
        "predicted_long_pnl": 20.0,
        "predicted_short_pnl": -5.0,
    })
    _, confidence, _ = direction_and_score(
        row, mode="classifier", penalty=0.0, filter_field="side_confidence"
    )
    _, gap, _ = direction_and_score(
        row, mode="classifier", penalty=0.0, filter_field="value_gap"
    )
    _, agreement, _ = direction_and_score(
        row, mode="classifier", penalty=0.0,
        filter_field="classifier_value_agreement",
    )
    assert abs(confidence - 0.3) < 1e-12
    assert gap == 25.0
    assert agreement == 1.0


def test_chronological_validation_penalizes_unstable_half() -> None:
    rows = []
    for index, pnl in enumerate((20.0, 10.0, -100.0, -80.0)):
        row = _row(index, pnl)
        row.update({"selected_side": 1, "long_pnl": pnl, "short_pnl": -pnl})
        rows.append(row)
    result = chronological_robust_validation(
        rows, split_ts_ns=2_000_000_000, min_trades_per_half=2
    )
    assert float(result["robust_scores"]["first_half"]) > 0.0
    assert float(result["robust_scores"]["second_half"]) < 0.0
    assert float(result["selection_score"]) < 0.0


def test_chronological_validation_requires_trade_coverage() -> None:
    row = _row(0, 1.0)
    row.update({"selected_side": 1, "long_pnl": 10.0})
    result = chronological_robust_validation(
        [row], split_ts_ns=1_000_000_000, min_trades_per_half=1
    )
    assert result["coverage_valid"] is False
    assert result["selection_score"] == -1_000_000.0
