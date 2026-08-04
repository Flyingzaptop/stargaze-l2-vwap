from __future__ import annotations

import numpy as np

from stargaze_ml.gold.l2_dominance_model import apply_price_dominance_veto, dominance_target


def test_dominance_target_is_symmetric_in_price_direction() -> None:
    delta = np.asarray([100.0, -100.0, 100.0, -100.0])
    long_pnl = np.asarray([-10.0, 10.0, 10.0, -10.0])
    short_pnl = -long_pnl
    np.testing.assert_array_equal(dominance_target(delta, long_pnl, short_pnl), [1, 1, 0, 0])


def test_price_dominance_veto_keeps_base_side_below_threshold() -> None:
    rows = [{"selected_side": 1}, {"selected_side": 1}]
    changed = apply_price_dominance_veto(
        rows, np.asarray([0.8, 0.6]), np.asarray([100.0, 100.0]), 0.7
    )
    assert [row["selected_side"] for row in changed] == [-1, 1]
