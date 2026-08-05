from __future__ import annotations

import pytest

import numpy as np

from stargaze_ml.gold.l2_risk_direction import _restrict_mask_to_entries, _summarize


def _row(long_pnl: float, short_pnl: float, long_tail: float, short_tail: float) -> dict[str, float]:
    return {
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "side_probability": 0.8,
        "predicted_long_pnl": 20.0,
        "predicted_short_pnl": 15.0,
        "long_tail_probability": long_tail,
        "short_tail_probability": short_tail,
        "opportunity_probability": 0.9,
    }


def test_risk_penalty_can_change_selected_side() -> None:
    rows = [_row(long_pnl=-100.0, short_pnl=30.0, long_tail=0.9, short_tail=0.1)]
    result = _summarize(
        rows,
        mode="risk",
        penalty=100.0,
        filter_field="opportunity_probability",
        cutoff=0.0,
    )
    assert result["trades"] == 1
    assert result["mean_pnl_ticks"] == pytest.approx(30.0)
    assert result["mean_tail_probability"] == pytest.approx(0.1)


def test_risk_filter_excludes_low_scoring_trade() -> None:
    rows = [_row(long_pnl=10.0, short_pnl=-10.0, long_tail=0.2, short_tail=0.4)]
    result = _summarize(
        rows,
        mode="classifier",
        penalty=0.0,
        filter_field="opportunity_probability",
        cutoff=0.95,
    )
    assert result["trades"] == 0


def test_entry_only_mask_keeps_only_frozen_open_points() -> None:
    mask = np.asarray([[True, True, False], [True, True, True]])
    events = np.asarray([2, 4])
    starts = np.asarray([0, 0, 10, 0, 20])
    result = _restrict_mask_to_entries(mask, events, starts, {2: 11, 4: 22})
    assert result.tolist() == [[False, True, False], [False, False, True]]
