from __future__ import annotations

import numpy as np
import torch

from stargaze_ml.gold.l2_open_events import _carry_nonzero_sign
from stargaze_ml.gold.l2_open_policy import L2OpenPolicy, exploration_probability
from stargaze_ml.gold.l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _pnl_ticks, schedule


def test_zero_delta_inherits_side_but_segment_resets() -> None:
    delta = np.asarray([1.0, 0.0, -1.0, 0.0, 0.0, 2.0])
    segment = np.asarray([0, 0, 0, 0, 1, 1])
    actual = _carry_nonzero_sign(delta, segment)
    np.testing.assert_array_equal(actual, [1, 1, -1, -1, 0, 1])


def test_open_policy_has_exactly_one_output() -> None:
    model = L2OpenPolicy(7, hidden_size=11)
    assert model(torch.zeros(3, 5, 7)).shape == (3, 5)


def test_exploration_floor_moves_probability_toward_half() -> None:
    logits = torch.tensor([-5.0, 5.0])
    learned = exploration_probability(logits, temperature=1.0, random_action_floor=0.0)
    explored = exploration_probability(logits, temperature=1.0, random_action_floor=0.4)
    assert explored[0] > learned[0]
    assert explored[1] < learned[1]


def test_warmup_schedule_rises_then_falls() -> None:
    values = [schedule(0.01, 0.25, 0.005, e, 5, 30) for e in range(30)]
    assert values[0] < values[4]
    assert values[4] > values[-1]


def test_oracle_reward_chooses_better_side() -> None:
    data = object.__new__(PreparedOpenData)
    data.event_crossing_1 = np.asarray([2])
    data.event_crossing_2 = np.asarray([3])
    data.event_side = np.asarray([1])
    data.first_bid = np.asarray([100.0, 101.0, 104.0, 106.0])
    data.first_ask = np.asarray([101.0, 102.0, 105.0, 107.0])
    config = OpenReinforceConfig(
        tick_size=1.0, commission_per_fill_ticks=0.0,
        slippage_per_fill_ticks=0.0, reward_mode="oracle_best",
    )
    reward = _pnl_ticks(data, np.asarray([0]), np.asarray([0]), 1, config)
    np.testing.assert_allclose(reward, [4.0])
