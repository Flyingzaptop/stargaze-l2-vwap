from __future__ import annotations

import numpy as np
import torch

from stargaze_ml.gold.l2_dominance_swap import (
    L2DominanceSwapPolicy,
    _position_from_belief,
    dynamic_dominance_target,
    dynamic_reversion_target,
)


def test_dynamic_target_distinguishes_who_closes_gap() -> None:
    mid = np.asarray([12.0, 11.0, 10.0, 10.0])
    vwap = np.asarray([10.0, 10.0, 10.0, 10.0])
    np.testing.assert_array_equal(dynamic_dominance_target(mid, vwap, 0, 3, 1), [1, 1, 0])
    mid2 = np.asarray([12.0, 12.0, 12.0])
    vwap2 = np.asarray([10.0, 11.0, 12.0])
    np.testing.assert_array_equal(dynamic_dominance_target(mid2, vwap2, 0, 2, 1), [0, 0])


def test_direction_uses_both_dominance_regimes() -> None:
    assert _position_from_belief(1, 0.9) == -1
    assert _position_from_belief(1, 0.1) == 1
    assert _position_from_belief(-1, 0.9) == 1
    assert _position_from_belief(-1, 0.1) == -1


def test_two_head_shapes() -> None:
    model = L2DominanceSwapPolicy(5, 7)
    open_logits, dominance_logits, side_logits = model(torch.zeros(3, 11, 5))
    assert open_logits.shape == dominance_logits.shape == side_logits.shape == (3, 11)
