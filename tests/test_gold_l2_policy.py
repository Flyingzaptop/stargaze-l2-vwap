from __future__ import annotations

import pytest
import torch
from torch import nn

from stargaze_ml.gold.l2_policy import (
    Action,
    BatchedBboEnvironment,
    L2EventPolicy,
    Position,
    TETRAHEDRAL_ACTION_CODES,
    deterministic_event_actions,
    hazard_distribution,
    mix_event_exploration_floor,
    rollout_policy,
    threshold_event_actions,
    valid_event_mask,
)


def test_state_and_execution_masks_are_applied_before_probability_normalization() -> None:
    positions = torch.tensor([Position.FLAT, Position.LONG, Position.SHORT, Position.FLAT])
    execution_valid = torch.tensor([True, True, True, False])
    expected = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(valid_event_mask(positions, execution_valid), expected)

    distribution = hazard_distribution(torch.zeros(4, 4), positions, execution_valid)
    assert torch.equal(distribution.valid_event_mask, expected)
    assert torch.equal(distribution.hazards == 0.0, ~expected)
    assert torch.all(distribution.probabilities[:, 1:][~expected] == 0.0)
    assert torch.allclose(distribution.probabilities.sum(dim=-1), torch.ones(4))
    assert distribution.probabilities[3, 0] == 1.0


def test_lstm_exposes_exactly_four_symmetric_event_channels() -> None:
    model = L2EventPolicy(input_size=7, hidden_size=11)
    logits, hidden = model(torch.randn(3, 5, 7))
    assert logits.shape == (3, 5, 4)
    assert model.lstm.num_layers == 1
    assert hidden[0].shape == (1, 3, 11)

    codes = torch.tensor(list(TETRAHEDRAL_ACTION_CODES.values()))
    assert torch.allclose(torch.linalg.vector_norm(codes, dim=1), torch.ones(4))
    distances = torch.pdist(codes)
    assert torch.allclose(distances, torch.full_like(distances, distances[0]))


def test_environment_executes_action_on_separately_passed_next_bbo() -> None:
    environment = BatchedBboEnvironment(
        1,
        device="cpu",
        commission_per_fill_ticks=0.0,
        slippage_per_fill_ticks=0.0,
    )
    opened = environment.step(
        torch.tensor([Action.OPEN_LONG]),
        next_bid=torch.tensor([109.0]),
        next_ask=torch.tensor([110.0]),
        execution_valid=torch.tensor([True]),
    )
    assert opened.cash.item() == -110.0
    assert opened.equity.item() == -1.0

    closed = environment.step(
        torch.tensor([Action.CLOSE_LONG]),
        next_bid=torch.tensor([120.0]),
        next_ask=torch.tensor([121.0]),
        execution_valid=torch.tensor([True]),
    )
    assert closed.cash.item() == 10.0
    assert closed.position.item() == Position.FLAT


@pytest.mark.parametrize(
    ("open_action", "close_action", "entry_bid", "entry_ask", "exit_bid", "exit_ask", "expected"),
    [
        (Action.OPEN_LONG, Action.CLOSE_LONG, 99.0, 100.0, 110.0, 111.0, 6.0),
        (Action.OPEN_SHORT, Action.CLOSE_SHORT, 110.0, 111.0, 99.0, 100.0, 6.0),
    ],
)
def test_exact_long_and_short_net_pnl(
    open_action: Action,
    close_action: Action,
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
    expected: float,
) -> None:
    environment = BatchedBboEnvironment(
        1,
        device="cpu",
        commission_per_fill_ticks=1.0,
        slippage_per_fill_ticks=1.0,
    )
    rewards = [
        environment.step(
            torch.tensor([open_action]),
            torch.tensor([entry_bid]),
            torch.tensor([entry_ask]),
            torch.tensor([True]),
        ).reward,
        environment.step(
            torch.tensor([close_action]),
            torch.tensor([exit_bid]),
            torch.tensor([exit_ask]),
            torch.tensor([True]),
        ).reward,
    ]
    terminal_reward, net_pnl = environment.finalize()
    total_reward = torch.stack(rewards).sum(dim=0) + terminal_reward
    assert net_pnl.item() == expected
    assert total_reward.item() == expected


class _AlwaysOpenLong(nn.Module):
    def forward(self, features: torch.Tensor, hidden=None):
        logits = torch.full((*features.shape[:2], 4), -80.0, device=features.device)
        logits[..., 0] = 80.0
        state = torch.zeros(1, features.shape[0], 1, device=features.device)
        return logits, (state, state)


class _ModerateHazards(nn.Module):
    def forward(self, features: torch.Tensor, hidden=None):
        logits = torch.full((*features.shape[:2], 4), -4.0, device=features.device)
        state = torch.zeros(1, features.shape[0], 1, device=features.device)
        return logits, (state, state)


def test_rollout_forces_terminal_close_and_dense_rewards_telescope() -> None:
    features = torch.tensor([[[1.0], [999.0], [-123.0]]])
    # These quotes are explicitly t+1 execution BBOs; feature values cannot
    # affect fill prices.
    bids = torch.tensor([[99.0, 104.0, 109.0]])
    asks = torch.tensor([[100.0, 105.0, 110.0]])
    rollout = rollout_policy(
        _AlwaysOpenLong(),  # type: ignore[arg-type]
        features,
        bids,
        asks,
        deterministic=True,
        commission_per_fill_ticks=1.0,
        slippage_per_fill_ticks=1.0,
    )
    assert torch.equal(
        rollout.actions,
        torch.tensor([[Action.OPEN_LONG, Action.NOOP, Action.NOOP]]),
    )
    assert rollout.terminal_forced_close.item()
    assert rollout.net_pnl_ticks.item() == 5.0
    assert torch.allclose(rollout.rewards.sum(dim=1), rollout.net_pnl_ticks)
    assert torch.allclose(rollout.probabilities.sum(dim=-1), torch.ones(1, 3))


def test_invalid_execution_forces_noop_during_sampled_rollout() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.zeros(2, 3, 1)
    valid = torch.zeros(2, 3, dtype=torch.bool)
    rollout = rollout_policy(
        _AlwaysOpenLong(),  # type: ignore[arg-type]
        features,
        torch.full((2, 3), float("nan")),
        torch.full((2, 3), float("nan")),
        valid,
        deterministic=False,
        generator=generator,
    )
    assert torch.all(rollout.actions == Action.NOOP)
    assert torch.all(rollout.probabilities[..., 0] == 1.0)
    assert torch.all(rollout.net_pnl_ticks == 0.0)


def test_stochastic_hazard_temperature_uses_matching_sample_log_probability() -> None:
    features = torch.zeros(4, 3, 1)
    bids = torch.full((4, 3), 99.0)
    asks = torch.full((4, 3), 100.0)
    warm = rollout_policy(
        _ModerateHazards(),  # type: ignore[arg-type]
        features,
        bids,
        asks,
        deterministic=False,
        hazard_temperature=1.3,
        event_exploration_floor=0.1,
        generator=torch.Generator().manual_seed(11),
        commission_per_fill_ticks=0.0,
        slippage_per_fill_ticks=0.0,
    )
    selected = warm.probabilities.gather(-1, warm.actions.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(warm.log_probabilities, selected.log())

    neutral = rollout_policy(
        _ModerateHazards(),  # type: ignore[arg-type]
        features,
        bids,
        asks,
        deterministic=False,
        hazard_temperature=1.0,
        generator=torch.Generator().manual_seed(11),
        commission_per_fill_ticks=0.0,
        slippage_per_fill_ticks=0.0,
    )
    assert warm.probabilities[0, 0, 1:].sum() > neutral.probabilities[0, 0, 1:].sum()

    with pytest.raises(ValueError, match="deterministic evaluation"):
        rollout_policy(
            _ModerateHazards(),  # type: ignore[arg-type]
            features,
            bids,
            asks,
            deterministic=True,
            hazard_temperature=1.3,
        )

    with pytest.raises(ValueError, match="event_exploration_floor=0"):
        rollout_policy(
            _ModerateHazards(),  # type: ignore[arg-type]
            features,
            bids,
            asks,
            deterministic=True,
            event_exploration_floor=0.01,
        )


def test_event_exploration_floor_is_uniform_only_over_valid_events() -> None:
    positions = torch.tensor([Position.FLAT, Position.LONG, Position.FLAT])
    execution_valid = torch.tensor([True, True, False])
    base = hazard_distribution(torch.full((3, 4), -4.0), positions, execution_valid)
    mixed = mix_event_exploration_floor(
        base.probabilities, base.valid_event_mask, floor=0.2
    )

    assert torch.allclose(mixed.sum(dim=-1), torch.ones(3))
    assert mixed[0, 0] == pytest.approx(float(base.probabilities[0, 0] * 0.8))
    assert torch.allclose(
        mixed[0, 1:3], base.probabilities[0, 1:3] * 0.8 + 0.1
    )
    assert torch.all(mixed[0, 3:] == 0.0)
    assert mixed[1, 3] == pytest.approx(float(base.probabilities[1, 3] * 0.8 + 0.2))
    assert torch.all(mixed[1, torch.tensor([1, 2, 4])] == 0.0)
    assert torch.equal(mixed[2], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_terminal_liquidation_uses_distinct_future_bbo() -> None:
    result = rollout_policy(
        _AlwaysOpenLong(),  # type: ignore[arg-type]
        torch.zeros(1, 1, 1),
        torch.tensor([[99.0]]),
        torch.tensor([[100.0]]),
        terminal_bid=torch.tensor([110.0]),
        terminal_ask=torch.tensor([111.0]),
        terminal_valid=torch.tensor([True]),
        deterministic=True,
        commission_per_fill_ticks=0.0,
        slippage_per_fill_ticks=0.0,
    )
    assert result.actions.item() == int(Action.OPEN_LONG)
    assert torch.allclose(result.net_pnl_ticks, torch.tensor([10.0]))
    assert torch.allclose(result.rewards.sum(dim=1), result.net_pnl_ticks)


def test_fast_deterministic_decoder_matches_rollout_state_machine() -> None:
    torch.manual_seed(31)
    model = L2EventPolicy(input_size=3, hidden_size=5, initial_event_bias=0.5)
    features = torch.randn(4, 19, 3)
    valid = torch.rand(4, 19) > 0.15
    bid = torch.full((4, 19), 100.0)
    ask = torch.full((4, 19), 101.0)
    logits, _ = model(features)
    decoded = deterministic_event_actions(logits, valid)
    rollout = rollout_policy(
        model,
        features,
        bid,
        ask,
        valid,
        deterministic=True,
        commission_per_fill_ticks=0.0,
        slippage_per_fill_ticks=0.0,
    )
    assert torch.equal(decoded, rollout.actions)


def test_threshold_decoder_respects_state_invalid_execution_and_open_tie() -> None:
    logits = torch.full((2, 6, 4), -10.0)
    # Exact open tie chooses long, then close it only after the close hazard
    # reaches the inclusive threshold.
    logits[0, 0, 0:2] = 0.0
    logits[0, 2, 2] = 0.0
    # Back in FLAT, the larger short hazard opens short.
    logits[0, 3, 0] = 0.0
    logits[0, 3, 1] = 1.0
    logits[0, 4, 3] = 10.0
    logits[0, 5, 3] = 0.0
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[0, 4] = False

    actions = threshold_event_actions(logits, valid, event_hazard_threshold=0.5)
    assert torch.equal(
        actions[0],
        torch.tensor(
            [
                Action.OPEN_LONG,
                Action.NOOP,
                Action.CLOSE_LONG,
                Action.OPEN_SHORT,
                Action.NOOP,
                Action.CLOSE_SHORT,
            ]
        ),
    )
    assert torch.all(actions[1] == Action.NOOP)

    with pytest.raises(ValueError, match="non-negative"):
        threshold_event_actions(logits, valid, event_hazard_threshold=-0.1)
