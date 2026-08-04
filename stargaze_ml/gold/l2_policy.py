"""Causal four-event L2 policy and one-lot BBO execution environment.

The neural network has exactly four raw outputs: open long, open short,
close long, and close short.  Waiting is represented implicitly by treating
the four positive outputs as competing event hazards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Position(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = 2


class Action(IntEnum):
    NOOP = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4


ACTION_NAMES: dict[Action, str] = {
    Action.NOOP: "no_op",
    Action.OPEN_LONG: "open_long",
    Action.OPEN_SHORT: "open_short",
    Action.CLOSE_LONG: "close_long",
    Action.CLOSE_SHORT: "close_short",
}

# Optional symmetric metadata for plots/serialization.  Training uses the
# categorical hazard policy below, never a regression loss on these codes.
_TETRA_SCALE = 1.0 / sqrt(3.0)
TETRAHEDRAL_ACTION_CODES: dict[Action, tuple[float, float, float]] = {
    Action.OPEN_LONG: (_TETRA_SCALE, _TETRA_SCALE, _TETRA_SCALE),
    Action.OPEN_SHORT: (_TETRA_SCALE, -_TETRA_SCALE, -_TETRA_SCALE),
    Action.CLOSE_LONG: (-_TETRA_SCALE, _TETRA_SCALE, -_TETRA_SCALE),
    Action.CLOSE_SHORT: (-_TETRA_SCALE, -_TETRA_SCALE, _TETRA_SCALE),
}

# Columns correspond to the four event actions (action id minus one).
_STATE_EVENT_MASK = (
    (True, True, False, False),   # flat: open long or open short
    (False, False, True, False),  # long: close long
    (False, False, False, True),  # short: close short
)


def valid_event_mask(
    position: Tensor,
    execution_valid: Tensor | None = None,
) -> Tensor:
    """Return the state/execution mask for the four externally visible events."""

    if position.is_floating_point():
        rounded = position.round()
        if not torch.equal(position, rounded):
            raise ValueError("position must contain integral state ids")
        position = rounded
    positions = position.to(dtype=torch.long)
    if torch.any((positions < int(Position.FLAT)) | (positions > int(Position.SHORT))):
        raise ValueError("position contains an unknown state id")

    templates = torch.tensor(_STATE_EVENT_MASK, dtype=torch.bool, device=positions.device)
    mask = templates[positions]
    if execution_valid is not None:
        valid = torch.as_tensor(execution_valid, dtype=torch.bool, device=positions.device)
        try:
            valid = torch.broadcast_to(valid, positions.shape)
        except RuntimeError as exc:
            raise ValueError("execution_valid is not broadcastable to position") from exc
        mask = mask & valid.unsqueeze(-1)
    return mask


def deterministic_event_actions(raw_logits: Tensor, execution_valid: Tensor) -> Tensor:
    """Decode hazard logits with the exact deterministic policy state machine.

    This is the evaluation-only counterpart of repeatedly calling
    :func:`hazard_distribution` followed by ``argmax``.  Passing CPU tensors is
    intentionally efficient for long sequences because it avoids one tiny GPU
    launch per second.
    """

    logits = torch.as_tensor(raw_logits, dtype=torch.float32)
    valid = torch.as_tensor(execution_valid, dtype=torch.bool, device=logits.device)
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("raw_logits must have shape [batch, time, 4]")
    if valid.shape != logits.shape[:2]:
        raise ValueError("execution_valid must have shape [batch, time]")

    batch_size, steps, _ = logits.shape
    hazards = F.softplus(logits)
    actions = torch.zeros((batch_size, steps), dtype=torch.long, device=logits.device)
    position = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    log_two = torch.log(torch.tensor(2.0, dtype=hazards.dtype, device=logits.device))

    for index in range(steps):
        valid_now = valid[:, index]
        flat = valid_now & (position == int(Position.FLAT))
        long = valid_now & (position == int(Position.LONG))
        short = valid_now & (position == int(Position.SHORT))

        flat_hazards = hazards[:, index, :2]
        total = flat_hazards.sum(dim=-1)
        no_event = torch.exp(-total)
        event_scale = -torch.expm1(-total) / total.clamp_min(torch.finfo(total.dtype).tiny)
        flat_probabilities = torch.cat(
            (no_event[:, None], flat_hazards * event_scale[:, None]), dim=-1
        )
        flat_choice = flat_probabilities.argmax(dim=-1)
        actions[:, index] = torch.where(
            flat & (flat_choice == 1),
            torch.full_like(position, int(Action.OPEN_LONG)),
            actions[:, index],
        )
        actions[:, index] = torch.where(
            flat & (flat_choice == 2),
            torch.full_like(position, int(Action.OPEN_SHORT)),
            actions[:, index],
        )
        actions[:, index] = torch.where(
            long & (hazards[:, index, 2] > log_two),
            torch.full_like(position, int(Action.CLOSE_LONG)),
            actions[:, index],
        )
        actions[:, index] = torch.where(
            short & (hazards[:, index, 3] > log_two),
            torch.full_like(position, int(Action.CLOSE_SHORT)),
            actions[:, index],
        )

        action_now = actions[:, index]
        position = torch.where(
            action_now == int(Action.OPEN_LONG),
            torch.full_like(position, int(Position.LONG)),
            position,
        )
        position = torch.where(
            action_now == int(Action.OPEN_SHORT),
            torch.full_like(position, int(Position.SHORT)),
            position,
        )
        position = torch.where(
            (action_now == int(Action.CLOSE_LONG)) | (action_now == int(Action.CLOSE_SHORT)),
            torch.full_like(position, int(Position.FLAT)),
            position,
        )
    return actions


def threshold_event_actions(
    raw_logits: Tensor,
    execution_valid: Tensor,
    event_hazard_threshold: float,
) -> Tensor:
    """Decode a sparse deterministic policy using an absolute event hazard.

    Hazards are always computed from untempered logits at ``T=1``.  In a flat
    state, the larger open hazard fires only when it reaches the threshold;
    an exact long/short tie deterministically selects open-long.  Positioned
    states consider only their matching close event.
    """

    if not torch.isfinite(torch.tensor(float(event_hazard_threshold))):
        raise ValueError("event_hazard_threshold must be finite")
    if event_hazard_threshold < 0.0:
        raise ValueError("event_hazard_threshold must be non-negative")
    logits = torch.as_tensor(raw_logits, dtype=torch.float32)
    valid = torch.as_tensor(execution_valid, dtype=torch.bool, device=logits.device)
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("raw_logits must have shape [batch, time, 4]")
    if valid.shape != logits.shape[:2]:
        raise ValueError("execution_valid must have shape [batch, time]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("raw_logits must be finite")

    batch_size, steps, _ = logits.shape
    hazards = F.softplus(logits)
    actions = torch.zeros((batch_size, steps), dtype=torch.long, device=logits.device)
    position = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    threshold = float(event_hazard_threshold)

    for index in range(steps):
        valid_now = valid[:, index]
        flat = valid_now & (position == int(Position.FLAT))
        long = valid_now & (position == int(Position.LONG))
        short = valid_now & (position == int(Position.SHORT))

        open_hazard, open_choice = hazards[:, index, :2].max(dim=-1)
        open_action = open_choice + int(Action.OPEN_LONG)
        actions[:, index] = torch.where(
            flat & (open_hazard >= threshold), open_action, actions[:, index]
        )
        actions[:, index] = torch.where(
            long & (hazards[:, index, 2] >= threshold),
            torch.full_like(position, int(Action.CLOSE_LONG)),
            actions[:, index],
        )
        actions[:, index] = torch.where(
            short & (hazards[:, index, 3] >= threshold),
            torch.full_like(position, int(Action.CLOSE_SHORT)),
            actions[:, index],
        )

        action_now = actions[:, index]
        position = torch.where(
            action_now == int(Action.OPEN_LONG),
            torch.full_like(position, int(Position.LONG)),
            position,
        )
        position = torch.where(
            action_now == int(Action.OPEN_SHORT),
            torch.full_like(position, int(Position.SHORT)),
            position,
        )
        position = torch.where(
            (action_now == int(Action.CLOSE_LONG)) | (action_now == int(Action.CLOSE_SHORT)),
            torch.full_like(position, int(Position.FLAT)),
            position,
        )
    return actions


@dataclass(frozen=True)
class HazardDistribution:
    """Five-action distribution produced from four event hazards."""

    probabilities: Tensor
    hazards: Tensor
    valid_event_mask: Tensor

    @property
    def no_op_probability(self) -> Tensor:
        return self.probabilities[..., 0]

    @property
    def event_probabilities(self) -> Tensor:
        return self.probabilities[..., 1:]


def mix_event_exploration_floor(
    probabilities: Tensor,
    valid_event_mask: Tensor,
    floor: float,
) -> Tensor:
    """Mix valid events with a uniform stochastic exploration floor.

    The no-op and learned event distribution retain ``1 - floor`` of the
    probability mass.  The remaining mass is uniform only across events that
    are valid in the current state.  With no valid execution, the floor is
    disabled and the original all-no-op distribution is preserved.
    """

    if not 0.0 <= floor <= 1.0:
        raise ValueError("event exploration floor must be between zero and one")
    if probabilities.shape[-1:] != (5,) or valid_event_mask.shape != probabilities.shape[:-1] + (4,):
        raise ValueError("probabilities/mask must end in five/four actions and otherwise align")
    if valid_event_mask.dtype is not torch.bool:
        raise ValueError("valid_event_mask must be boolean")
    if floor == 0.0:
        return probabilities

    valid_count = valid_event_mask.sum(dim=-1, keepdim=True)
    has_valid_event = valid_count > 0
    effective_floor = has_valid_event.to(probabilities.dtype) * float(floor)
    no_op = probabilities[..., :1] * (1.0 - effective_floor)
    uniform_valid = valid_event_mask.to(probabilities.dtype) / valid_count.clamp_min(1)
    events = probabilities[..., 1:] * (1.0 - effective_floor) + effective_floor * uniform_valid
    return torch.cat((no_op, events), dim=-1)


def hazard_distribution(
    raw_event_logits: Tensor,
    position: Tensor,
    execution_valid: Tensor | None = None,
) -> HazardDistribution:
    """Convert four logits into competing hazards plus an implicit no-op.

    For a one-second interval, ``p(no-op) = exp(-sum(lambda))`` and the
    remaining mass is divided between valid events in proportion to their
    softplus hazards.  Invalid hazards are masked before normalization.
    """

    if not raw_event_logits.is_floating_point() or raw_event_logits.shape[-1:] != (4,):
        raise ValueError("raw_event_logits must be floating point with final dimension 4")
    if not bool(torch.isfinite(raw_event_logits).all()):
        raise ValueError("raw_event_logits must be finite")
    if raw_event_logits.shape[:-1] != position.shape:
        raise ValueError("position shape must equal raw_event_logits.shape[:-1]")

    event_mask = valid_event_mask(position, execution_valid)
    hazards = F.softplus(raw_event_logits).masked_fill(~event_mask, 0.0)
    total_hazard = hazards.sum(dim=-1, keepdim=True)
    no_op = torch.exp(-total_hazard)
    event_mass = -torch.expm1(-total_hazard)
    denominator = total_hazard.clamp_min(torch.finfo(raw_event_logits.dtype).tiny)
    event_probabilities = event_mass * hazards / denominator
    probabilities = torch.cat((no_op, event_probabilities), dim=-1)
    return HazardDistribution(probabilities, hazards, event_mask)


def choose_actions(
    probabilities: Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Choose batched actions and return their differentiable log probability."""

    if probabilities.shape[-1:] != (5,) or not probabilities.is_floating_point():
        raise ValueError("probabilities must be floating point with final dimension 5")
    if torch.any(probabilities < 0.0) or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite and non-negative")
    if not torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("probabilities must sum to one")

    if deterministic:
        actions = probabilities.argmax(dim=-1)
    else:
        flat = probabilities.reshape(-1, 5)
        actions = torch.multinomial(flat, 1, generator=generator).reshape(probabilities.shape[:-1])
    selected = probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    log_probabilities = selected.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    return actions, log_probabilities


class L2EventPolicy(nn.Module):
    """A one-layer LSTM with exactly four raw event logits per second."""

    def __init__(self, input_size: int, hidden_size: int = 64, *, initial_event_bias: float = -5.0) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.event_head = nn.Linear(self.hidden_size, 4)
        nn.init.constant_(self.event_head.bias, float(initial_event_bias))

    def forward(
        self,
        features: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError("features must have shape [batch, time, input_size]")
        encoded, next_hidden = self.lstm(features, hidden)
        return self.event_head(encoded), next_hidden


@dataclass(frozen=True)
class EnvironmentStep:
    reward: Tensor
    equity: Tensor
    cash: Tensor
    position: Tensor


class BatchedBboEnvironment:
    """One-lot, long-or-short execution using separately supplied next-second BBO."""

    def __init__(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        commission_per_fill_ticks: float = 15.0,
        slippage_per_fill_ticks: float = 1.0,
        tick_size: float = 1.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if commission_per_fill_ticks < 0.0 or slippage_per_fill_ticks < 0.0:
            raise ValueError("execution costs must be non-negative")
        if tick_size <= 0.0:
            raise ValueError("tick_size must be positive")
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.commission = float(commission_per_fill_ticks)
        self.slippage = float(slippage_per_fill_ticks)
        self.tick_size = float(tick_size)
        self.reset()

    def reset(self) -> None:
        self.position = torch.full(
            (self.batch_size,), int(Position.FLAT), dtype=torch.long, device=self.device
        )
        self.cash = torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)
        self.equity = torch.zeros_like(self.cash)
        self.last_bid = torch.zeros_like(self.cash)
        self.last_ask = torch.zeros_like(self.cash)
        self.has_quote = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._finalized = False

    def step(
        self,
        action: Tensor,
        next_bid: Tensor,
        next_ask: Tensor,
        execution_valid: Tensor,
    ) -> EnvironmentStep:
        if self._finalized:
            raise RuntimeError("environment has already been finalized")
        actions = torch.as_tensor(action, dtype=torch.long, device=self.device)
        bid = torch.as_tensor(next_bid, dtype=self.dtype, device=self.device) / self.tick_size
        ask = torch.as_tensor(next_ask, dtype=self.dtype, device=self.device) / self.tick_size
        valid = torch.as_tensor(execution_valid, dtype=torch.bool, device=self.device)
        expected = (self.batch_size,)
        if actions.shape != expected or bid.shape != expected or ask.shape != expected or valid.shape != expected:
            raise ValueError("action, next_bid, next_ask, and execution_valid must have shape [batch]")
        if torch.any((actions < int(Action.NOOP)) | (actions > int(Action.CLOSE_SHORT))):
            raise ValueError("unknown action id")
        if torch.any(valid & (~torch.isfinite(bid) | ~torch.isfinite(ask) | (bid > ask))):
            raise ValueError("valid execution quotes must be finite and satisfy bid <= ask")

        event_mask = valid_event_mask(self.position, valid)
        event_indices = (actions - 1).clamp_min(0)
        selected_valid = event_mask.gather(-1, event_indices.unsqueeze(-1)).squeeze(-1)
        if torch.any((actions != int(Action.NOOP)) & ~selected_valid):
            raise ValueError("action is invalid for the current position or execution quote")

        open_long = actions == int(Action.OPEN_LONG)
        open_short = actions == int(Action.OPEN_SHORT)
        close_long = actions == int(Action.CLOSE_LONG)
        close_short = actions == int(Action.CLOSE_SHORT)

        zero = torch.zeros_like(self.cash)
        # ``where`` is deliberate: multiplying a false mask by an invalid NaN
        # quote would still poison cash through IEEE ``0 * NaN == NaN``.
        self.cash = self.cash - torch.where(
            open_long, ask + self.slippage + self.commission, zero
        )
        self.cash = self.cash + torch.where(
            open_short, bid - self.slippage - self.commission, zero
        )
        self.cash = self.cash + torch.where(
            close_long, bid - self.slippage - self.commission, zero
        )
        self.cash = self.cash - torch.where(
            close_short, ask + self.slippage + self.commission, zero
        )

        self.position = torch.where(
            open_long,
            torch.full_like(self.position, int(Position.LONG)),
            self.position,
        )
        self.position = torch.where(
            open_short,
            torch.full_like(self.position, int(Position.SHORT)),
            self.position,
        )
        self.position = torch.where(
            close_long | close_short,
            torch.full_like(self.position, int(Position.FLAT)),
            self.position,
        )

        marked_equity = torch.where(
            self.position == int(Position.LONG),
            self.cash + bid,
            torch.where(self.position == int(Position.SHORT), self.cash - ask, self.cash),
        )
        reward = torch.where(valid, marked_equity - self.equity, torch.zeros_like(self.equity))
        self.equity = torch.where(valid, marked_equity, self.equity)
        self.last_bid = torch.where(valid, bid, self.last_bid)
        self.last_ask = torch.where(valid, ask, self.last_ask)
        self.has_quote |= valid
        return EnvironmentStep(reward, self.equity.clone(), self.cash.clone(), self.position.clone())

    def finalize(
        self,
        final_bid: Tensor | None = None,
        final_ask: Tensor | None = None,
        final_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Force-close and return terminal reward/net PnL.

        Optional quotes provide a distinct causal terminal BBO after the last
        action execution. Invalid terminal rows fall back to the last valid
        execution quote.
        """

        if self._finalized:
            raise RuntimeError("environment has already been finalized")
        if (final_bid is None) != (final_ask is None):
            raise ValueError("final_bid and final_ask must be provided together")
        if final_bid is not None:
            bid = torch.as_tensor(final_bid, dtype=self.dtype, device=self.device) / self.tick_size
            ask = torch.as_tensor(final_ask, dtype=self.dtype, device=self.device) / self.tick_size
            valid = (
                torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
                if final_valid is None
                else torch.as_tensor(final_valid, dtype=torch.bool, device=self.device)
            )
            if bid.shape != (self.batch_size,) or ask.shape != bid.shape or valid.shape != bid.shape:
                raise ValueError("terminal quotes and validity must have shape [batch]")
            if torch.any(valid & (~torch.isfinite(bid) | ~torch.isfinite(ask) | (bid > ask))):
                raise ValueError("valid terminal quotes must be finite and satisfy bid <= ask")
            self.last_bid = torch.where(valid, bid, self.last_bid)
            self.last_ask = torch.where(valid, ask, self.last_ask)
            self.has_quote |= valid

        exposed = self.position != int(Position.FLAT)
        if torch.any(exposed & ~self.has_quote):
            raise RuntimeError("cannot liquidate a position without a valid quote")
        long_position = self.position == int(Position.LONG)
        short_position = self.position == int(Position.SHORT)
        self.cash = self.cash + long_position.to(self.dtype) * (
            self.last_bid - self.slippage - self.commission
        )
        self.cash = self.cash - short_position.to(self.dtype) * (
            self.last_ask + self.slippage + self.commission
        )
        terminal_reward = self.cash - self.equity
        self.equity = self.cash.clone()
        self.position.fill_(int(Position.FLAT))
        self._finalized = True
        return terminal_reward, self.cash.clone()


@dataclass(frozen=True)
class RolloutResult:
    actions: Tensor
    log_probabilities: Tensor
    probabilities: Tensor
    rewards: Tensor
    positions: Tensor
    terminal_forced_close: Tensor
    net_pnl_ticks: Tensor


def rollout_policy(
    model: L2EventPolicy,
    features: Tensor,
    next_bid: Tensor,
    next_ask: Tensor,
    execution_valid: Tensor | None = None,
    *,
    terminal_bid: Tensor | None = None,
    terminal_ask: Tensor | None = None,
    terminal_valid: Tensor | None = None,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
    hazard_temperature: float = 1.0,
    event_exploration_floor: float = 0.0,
    commission_per_fill_ticks: float = 15.0,
    slippage_per_fill_ticks: float = 1.0,
    tick_size: float = 1.0,
) -> RolloutResult:
    """Run independent batched episodes and force-liquidate at their final BBO."""

    if features.ndim != 3:
        raise ValueError("features must have shape [batch, time, feature]")
    if hazard_temperature <= 0.0:
        raise ValueError("hazard_temperature must be positive")
    if not 0.0 <= event_exploration_floor <= 1.0:
        raise ValueError("event_exploration_floor must be between zero and one")
    if deterministic and hazard_temperature != 1.0:
        raise ValueError("deterministic evaluation must use hazard_temperature=1.0")
    if deterministic and event_exploration_floor != 0.0:
        raise ValueError("deterministic evaluation must use event_exploration_floor=0")
    batch_size, steps, _ = features.shape
    if steps <= 0:
        raise ValueError("episodes must contain at least one step")
    expected = (batch_size, steps)
    if next_bid.shape != expected or next_ask.shape != expected:
        raise ValueError("next_bid and next_ask must have shape [batch, time]")
    if execution_valid is None:
        execution_valid = torch.ones(expected, dtype=torch.bool, device=features.device)
    else:
        execution_valid = torch.as_tensor(execution_valid, dtype=torch.bool, device=features.device)
        if execution_valid.shape != expected:
            raise ValueError("execution_valid must have shape [batch, time]")

    raw_logits, _ = model(features)
    raw_logits = raw_logits.float()
    environment = BatchedBboEnvironment(
        batch_size,
        device=features.device,
        dtype=next_bid.dtype,
        commission_per_fill_ticks=commission_per_fill_ticks,
        slippage_per_fill_ticks=slippage_per_fill_ticks,
        tick_size=tick_size,
    )
    action_steps: list[Tensor] = []
    log_probability_steps: list[Tensor] = []
    probability_steps: list[Tensor] = []
    reward_steps: list[Tensor] = []
    position_steps: list[Tensor] = []

    for index in range(steps):
        distribution = hazard_distribution(
            raw_logits[:, index] / float(hazard_temperature),
            environment.position,
            execution_valid[:, index],
        )
        action_probabilities = mix_event_exploration_floor(
            distribution.probabilities,
            distribution.valid_event_mask,
            event_exploration_floor,
        )
        actions, log_probabilities = choose_actions(
            action_probabilities,
            deterministic=deterministic,
            generator=generator,
        )
        step = environment.step(
            actions,
            next_bid[:, index],
            next_ask[:, index],
            execution_valid[:, index],
        )
        action_steps.append(actions)
        log_probability_steps.append(log_probabilities)
        probability_steps.append(action_probabilities)
        reward_steps.append(step.reward)
        position_steps.append(step.position)

    terminal_forced_close = environment.position != int(Position.FLAT)
    terminal_reward, net_pnl = environment.finalize(
        terminal_bid,
        terminal_ask,
        terminal_valid,
    )
    reward_steps[-1] = reward_steps[-1] + terminal_reward
    return RolloutResult(
        actions=torch.stack(action_steps, dim=1),
        log_probabilities=torch.stack(log_probability_steps, dim=1),
        probabilities=torch.stack(probability_steps, dim=1),
        rewards=torch.stack(reward_steps, dim=1),
        positions=torch.stack(position_steps, dim=1),
        terminal_forced_close=terminal_forced_close,
        net_pnl_ticks=net_pnl,
    )
