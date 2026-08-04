from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from stargaze_ml.training.data import RobustNormalizer

from .l2_policy import (
    Action,
    L2EventPolicy,
    deterministic_event_actions,
    rollout_policy,
    threshold_event_actions,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReinforceConfig:
    hidden_size: int = 64
    episode_length: int = 128
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 1.0
    entropy_start: float = 0.0005
    entropy_peak: float = 0.01
    entropy_end: float = 0.0005
    entropy_warmup_epochs: int = 5
    temperature_start: float = 1.0
    temperature_peak: float = 1.3
    temperature_end: float = 0.9
    event_floor_start: float = 0.005
    event_floor_peak: float = 0.03
    event_floor_end: float = 0.0005
    initial_event_bias: float = -5.0
    gradient_clip: float = 1.0
    normalizer_clip: float = 12.0
    tick_size: float = 0.01
    commission_per_fill_ticks: float = 15.0
    slippage_per_fill_ticks: float = 1.0
    seed: int = 20260804
    use_amp: bool = True

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.episode_length < 4:
            raise ValueError("hidden_size must be positive and episode_length at least four")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("learning_rate and gamma must be positive")
        if min(self.entropy_start, self.entropy_peak, self.entropy_end) < 0:
            raise ValueError("entropy coefficients must be non-negative")
        if self.entropy_peak < max(self.entropy_start, self.entropy_end):
            raise ValueError("entropy_peak must be at least entropy_start and entropy_end")
        if min(self.temperature_start, self.temperature_peak, self.temperature_end) <= 0:
            raise ValueError("hazard temperatures must be positive")
        if self.temperature_peak < max(self.temperature_start, self.temperature_end):
            raise ValueError("temperature_peak must be at least temperature_start and temperature_end")
        if not all(
            0.0 <= value <= 1.0
            for value in (self.event_floor_start, self.event_floor_peak, self.event_floor_end)
        ):
            raise ValueError("event exploration floors must be between zero and one")
        if self.event_floor_peak < max(self.event_floor_start, self.event_floor_end):
            raise ValueError("event_floor_peak must be at least event_floor_start and event_floor_end")
        if not 1 <= self.entropy_warmup_epochs <= self.epochs:
            raise ValueError("entropy_warmup_epochs must be between one and epochs")
        if self.epochs > 1 and self.entropy_warmup_epochs == self.epochs:
            raise ValueError("entropy warm-up must leave at least one decay epoch")
        if self.tick_size <= 0 or self.commission_per_fill_ticks < 0 or self.slippage_per_fill_ticks < 0:
            raise ValueError("tick size must be positive and costs non-negative")


@dataclass(frozen=True)
class PreparedPolicyData:
    ts_ns: np.ndarray
    segment_id: np.ndarray
    x: np.ndarray
    feature_names: tuple[str, ...]
    valid_feature: np.ndarray
    observed: np.ndarray
    first_bid: np.ndarray
    first_ask: np.ndarray
    train_end: int
    validation_end: int

    def __len__(self) -> int:
        return len(self.ts_ns)


def load_prepared_policy_data(path: str | Path) -> PreparedPolicyData:
    with np.load(Path(path).expanduser().resolve(strict=True), allow_pickle=False) as payload:
        feature_names = tuple(str(value) for value in payload["feature_names"].tolist())
        data = PreparedPolicyData(
            ts_ns=np.ascontiguousarray(payload["ts_ns"], dtype=np.int64),
            segment_id=np.ascontiguousarray(payload["segment_id"], dtype=np.int32),
            x=np.ascontiguousarray(payload["x"], dtype=np.float32),
            feature_names=feature_names,
            valid_feature=np.ascontiguousarray(payload["valid_feature"], dtype=bool),
            observed=np.ascontiguousarray(payload["observed"], dtype=bool),
            first_bid=np.ascontiguousarray(payload["first_bid"], dtype=np.float64),
            first_ask=np.ascontiguousarray(payload["first_ask"], dtype=np.float64),
            train_end=int(payload["train_end"]),
            validation_end=int(payload["validation_end"]),
        )
    n = len(data)
    if data.x.shape != (n, len(data.feature_names)):
        raise ValueError("prepared feature matrix is not aligned")
    if not 0 < data.train_end < data.validation_end < n:
        raise ValueError("prepared chronological split boundaries are invalid")
    return data


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


class EpisodeSampler:
    """One-pass non-overlapping chunks over every eligible decision row.

    A decision at row ``t`` is eligible when its feature is valid, its
    next-second execution BBO at ``t + 1`` is observed, and the distinct
    terminal row ``t + 2`` remains in the same train segment.  Chunks never
    overlap in decision rows.  Tail chunks are time-padded and carry an
    explicit mask; batches themselves are never padded or dropped.
    """

    def __init__(
        self,
        data: PreparedPolicyData,
        start: int,
        end: int,
        episode_length: int,
        *,
        seed: int,
    ) -> None:
        self.data = data
        self.length = int(episode_length)
        self.seed = int(seed)
        left_bound = max(0, int(start))
        right_bound = min(len(data), int(end))
        chunks: list[np.ndarray] = []
        chunk_decision_masks: list[np.ndarray] = []
        eligible_rows: list[np.ndarray] = []
        boundaries = np.flatnonzero(np.r_[True, data.segment_id[1:] != data.segment_id[:-1], True])
        for segment_left, segment_right in zip(boundaries[:-1], boundaries[1:], strict=True):
            left = max(left_bound, int(segment_left))
            right = min(right_bound, int(segment_right))
            # t, t+1 execution and t+2 terminal must remain in the segment.
            candidates = np.arange(left, right - 2, dtype=np.int64)
            if candidates.size == 0:
                continue
            decision_valid = data.valid_feature[candidates] & data.observed[candidates + 1]
            if not np.any(decision_valid):
                continue
            eligible_rows.append(candidates[decision_valid])
            # Chunk the continuous segment clock, not the sparse eligible-row
            # set.  Missing execution rows therefore advance the LSTM state
            # with execution disabled instead of creating artificial episodes.
            for offset in range(0, len(candidates), self.length):
                span = candidates[offset : offset + self.length]
                span_mask = decision_valid[offset : offset + self.length]
                if np.any(span_mask):
                    chunks.append(span)
                    chunk_decision_masks.append(span_mask)
        if not chunks:
            raise ValueError("split has no eligible decision rows")
        self.chunks = tuple(chunks)
        self.chunk_decision_masks = tuple(chunk_decision_masks)
        self.eligible_rows = np.concatenate(eligible_rows)
        self.eligible_count = int(len(self.eligible_rows))

    def iter_epoch_batches(
        self,
        batch_size: int,
        *,
        epoch: int,
    ):
        """Yield every chunk once in an epoch-specific shuffled order."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        order = np.random.default_rng(self.seed + int(epoch)).permutation(len(self.chunks))
        for batch_left in range(0, len(order), int(batch_size)):
            chunk_ids = order[batch_left : batch_left + int(batch_size)]
            count = len(chunk_ids)
            feature_indices = np.zeros((count, self.length), dtype=np.int64)
            execution_indices = np.zeros_like(feature_indices)
            decision_mask = np.zeros((count, self.length), dtype=bool)
            sequence_mask = np.zeros((count, self.length), dtype=bool)
            terminal_indices = np.zeros(count, dtype=np.int64)
            for row, chunk_id in enumerate(chunk_ids):
                decisions = self.chunks[int(chunk_id)]
                eligible = self.chunk_decision_masks[int(chunk_id)]
                real_length = len(decisions)
                feature_indices[row] = decisions[-1]
                execution_indices[row] = decisions[-1] + 1
                feature_indices[row, :real_length] = decisions
                execution_indices[row, :real_length] = decisions + 1
                decision_mask[row, :real_length] = eligible
                sequence_mask[row, :real_length] = True
                terminal_indices[row] = decisions[-1] + 2
            yield (
                feature_indices,
                execution_indices,
                terminal_indices,
                decision_mask,
                sequence_mask,
            )


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma == 1.0:
        return torch.flip(torch.cumsum(torch.flip(rewards, dims=(1,)), dim=1), dims=(1,))
    output = torch.empty_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    for step in range(rewards.shape[1] - 1, -1, -1):
        running = rewards[:, step] + float(gamma) * running
        output[:, step] = running
    return output


class ReinforceTrainer:
    def __init__(
        self,
        model: L2EventPolicy,
        data: PreparedPolicyData,
        config: ReinforceConfig,
        output_dir: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        config.validate()
        self.config = config
        self.data = data
        self.device = resolve_device(device)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = RobustNormalizer.fit(
            data.x,
            data.valid_feature & (np.arange(len(data)) < data.train_end),
            clip=config.normalizer_clip,
        )
        self.x = self.normalizer.transform(data.x)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sampler = EpisodeSampler(
            data, 0, data.train_end, config.episode_length, seed=config.seed
        )
        self.history: list[dict[str, object]] = []
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.benchmark = True

    def _entropy_coefficient(self, epoch: int) -> float:
        return self._scheduled_exploration_value(
            self.config.entropy_start,
            self.config.entropy_peak,
            self.config.entropy_end,
            epoch,
        )

    def _hazard_temperature(self, epoch: int) -> float:
        return self._scheduled_exploration_value(
            self.config.temperature_start,
            self.config.temperature_peak,
            self.config.temperature_end,
            epoch,
        )

    def _event_exploration_floor(self, epoch: int) -> float:
        return self._scheduled_exploration_value(
            self.config.event_floor_start,
            self.config.event_floor_peak,
            self.config.event_floor_end,
            epoch,
        )

    def _scheduled_exploration_value(
        self,
        start: float,
        peak: float,
        end: float,
        epoch: int,
    ) -> float:
        if not 0 <= epoch < self.config.epochs:
            raise ValueError("epoch is outside the configured training range")
        warmup = self.config.entropy_warmup_epochs
        if self.config.epochs == 1 or warmup == 1 and epoch == 0:
            return float(peak)
        if epoch < warmup:
            fraction = epoch / (warmup - 1)
            return float(start + fraction * (peak - start))
        decay_epochs = self.config.epochs - warmup
        fraction = (epoch - warmup + 1) / decay_epochs
        return float(peak + fraction * (end - peak))

    def _batch(
        self,
        feature_idx: np.ndarray,
        execution_idx: np.ndarray,
        terminal_idx: np.ndarray,
        decision_mask: np.ndarray,
        sequence_mask: np.ndarray,
    ):
        feature_values = self.x[feature_idx].copy()
        feature_values[~(self.data.valid_feature[feature_idx] & sequence_mask)] = 0.0
        features = torch.as_tensor(feature_values, device=self.device)
        bid = torch.as_tensor(self.data.first_bid[execution_idx], device=self.device, dtype=torch.float32)
        ask = torch.as_tensor(self.data.first_ask[execution_idx], device=self.device, dtype=torch.float32)
        valid = torch.as_tensor(
            decision_mask, device=self.device
        )
        mask = torch.as_tensor(decision_mask, device=self.device)
        terminal_bid = torch.as_tensor(self.data.first_bid[terminal_idx], device=self.device, dtype=torch.float32)
        terminal_ask = torch.as_tensor(self.data.first_ask[terminal_idx], device=self.device, dtype=torch.float32)
        terminal_valid = torch.as_tensor(self.data.observed[terminal_idx], device=self.device)
        return features, bid, ask, valid, terminal_bid, terminal_ask, terminal_valid, mask

    def train_epoch(self, epoch: int) -> dict[str, object]:
        self.model.train()
        entropy_coefficient = self._entropy_coefficient(epoch)
        hazard_temperature = self._hazard_temperature(epoch)
        event_exploration_floor = self._event_exploration_floor(epoch)
        policy_loss_total = 0.0
        entropy_total = 0.0
        event_probability_total = 0.0
        event_hazard_total = 0.0
        pnl_total = 0.0
        positive = 0
        forced_close = 0
        action_counts = np.zeros(5, dtype=np.int64)
        batches = 0
        episodes = 0
        covered_decision_rows = 0
        started = perf_counter()
        for feature_idx, execution_idx, terminal_idx, decision_mask, sequence_mask in (
            self.sampler.iter_epoch_batches(self.config.batch_size, epoch=epoch)
        ):
            tensors = self._batch(
                feature_idx, execution_idx, terminal_idx, decision_mask, sequence_mask
            )
            features, bid, ask, valid, terminal_bid, terminal_ask, terminal_valid, mask = tensors
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                rollout = rollout_policy(
                    self.model,
                    features,
                    bid,
                    ask,
                    valid,
                    terminal_bid=terminal_bid,
                    terminal_ask=terminal_ask,
                    terminal_valid=terminal_valid,
                    deterministic=False,
                    hazard_temperature=hazard_temperature,
                    event_exploration_floor=event_exploration_floor,
                    commission_per_fill_ticks=self.config.commission_per_fill_ticks,
                    slippage_per_fill_ticks=self.config.slippage_per_fill_ticks,
                    tick_size=self.config.tick_size,
                )
            rewards = rollout.rewards.float().clone()
            time_index = torch.arange(rewards.shape[1], device=self.device).unsqueeze(0)
            last_decision = torch.where(mask, time_index, -1).max(dim=1).values
            move_terminal = last_decision != rewards.shape[1] - 1
            if bool(move_terminal.any()):
                rows = torch.nonzero(move_terminal, as_tuple=False).flatten()
                rewards[rows, last_decision[rows]] += rewards[rows, -1]
                rewards[rows, -1] = 0.0
            rewards = rewards.masked_fill(~mask, 0.0)
            returns = discounted_returns(rewards, self.config.gamma)
            mask_float = mask.to(returns.dtype)
            counts_by_step = mask_float.sum(dim=0, keepdim=True).clamp_min(1.0)
            baseline = (returns * mask_float).sum(dim=0, keepdim=True) / counts_by_step
            advantage = returns - baseline
            valid_advantage = advantage[mask]
            advantage_scale = valid_advantage.std(unbiased=False).clamp_min(1e-6)
            policy_loss = -(
                advantage.detach()[mask]
                * rollout.log_probabilities.float()[mask]
            ).mean() / advantage_scale
            probabilities = rollout.probabilities.float().clamp_min(1e-12)
            per_decision_entropy = -(probabilities * probabilities.log()).sum(dim=-1)
            entropy = per_decision_entropy[mask].mean()
            event_probability = (1.0 - probabilities[..., 0])[mask].mean()
            total_event_hazard = -probabilities[..., 0].log()
            mean_event_hazard = total_event_hazard[mask].mean()
            loss = policy_loss - entropy_coefficient * entropy

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                actions = rollout.actions[mask]
                action_counts += torch.bincount(actions, minlength=5).cpu().numpy()
                pnl = rollout.net_pnl_ticks
                decisions_in_batch = int(mask.sum().cpu())
                policy_loss_total += float(policy_loss.cpu())
                entropy_total += float(per_decision_entropy[mask].sum().cpu())
                event_probability_total += float((1.0 - probabilities[..., 0])[mask].sum().cpu())
                event_hazard_total += float(total_event_hazard[mask].sum().cpu())
                pnl_total += float(pnl.sum().cpu())
                positive += int((pnl > 0).sum().cpu())
                forced_close += int(rollout.terminal_forced_close.sum().cpu())
                covered_decision_rows += decisions_in_batch
                episodes += int(features.shape[0])
                batches += 1
        if covered_decision_rows != self.sampler.eligible_count:
            raise RuntimeError(
                "epoch coverage invariant failed: "
                f"covered={covered_decision_rows} eligible={self.sampler.eligible_count}"
            )
        elapsed = perf_counter() - started
        event_count = int(action_counts[1:].sum())
        closes = int(action_counts[int(Action.CLOSE_LONG)] + action_counts[int(Action.CLOSE_SHORT)])
        trades = closes + forced_close
        return {
            "epoch": epoch + 1,
            "episodes": episodes,
            "covered_decision_rows": covered_decision_rows,
            "eligible_decision_rows": self.sampler.eligible_count,
            "coverage_ratio": covered_decision_rows / self.sampler.eligible_count,
            "sampled_seconds": covered_decision_rows,
            "policy_loss": policy_loss_total / max(batches, 1),
            "entropy": entropy_total / covered_decision_rows,
            "entropy_coefficient": entropy_coefficient,
            "hazard_temperature": hazard_temperature,
            "event_exploration_floor": event_exploration_floor,
            "mean_event_probability": event_probability_total / covered_decision_rows,
            "mean_total_event_hazard": event_hazard_total / covered_decision_rows,
            "events": event_count,
            "event_rate": event_count / covered_decision_rows,
            "trades_including_terminal": trades,
            "forced_terminal_closes": forced_close,
            "mean_episode_net_ticks": pnl_total / episodes,
            "mean_net_ticks_per_decision": pnl_total / covered_decision_rows,
            "positive_episode_rate": positive / episodes,
            "actions": {
                "no_op": int(action_counts[0]),
                "open_long": int(action_counts[1]),
                "open_short": int(action_counts[2]),
                "close_long": int(action_counts[3]),
                "close_short": int(action_counts[4]),
            },
            "elapsed_seconds": elapsed,
        }

    def save_checkpoint(self, name: str) -> Path:
        path = self.output_dir / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": asdict(self.config),
                "feature_names": list(self.data.feature_names),
                "normalizer": self.normalizer.to_dict(),
                "history": self.history,
                "train_end": self.data.train_end,
                "validation_end": self.data.validation_end,
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
            },
            path,
        )
        return path

    def train(self) -> list[dict[str, object]]:
        LOGGER.info(
            "training device=%s rows=%d eligible_decisions=%d chunks=%d params=%d",
            self.device,
            len(self.data),
            self.sampler.eligible_count,
            len(self.sampler.chunks),
            sum(p.numel() for p in self.model.parameters()),
        )
        for epoch in range(self.config.epochs):
            stats = self.train_epoch(epoch)
            self.history.append(stats)
            self.save_checkpoint("checkpoint.pt")
            (self.output_dir / "training_history.json").write_text(
                json.dumps(self.history, indent=2), encoding="utf-8"
            )
            LOGGER.info(
                "epoch=%d/%d coverage=%.3f entropy_coef=%.6f temperature=%.4f "
                "event_floor=%.5f entropy=%.4f "
                "event_p=%.4f hazard=%.4f actual_event_rate=%.4f "
                "episode_net=%.2f positive=%.3f elapsed=%.1fs",
                epoch + 1,
                self.config.epochs,
                stats["coverage_ratio"],
                stats["entropy_coefficient"],
                stats["hazard_temperature"],
                stats["event_exploration_floor"],
                stats["entropy"],
                stats["mean_event_probability"],
                stats["mean_total_event_hazard"],
                stats["event_rate"],
                stats["mean_episode_net_ticks"],
                stats["positive_episode_rate"],
                stats["elapsed_seconds"],
            )
        self.save_checkpoint("final.pt")
        return self.history


def _profit_factor(values: np.ndarray) -> float:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)


def _segment_ranges(data: PreparedPolicyData, start: int, end: int):
    boundaries = np.flatnonzero(np.r_[True, data.segment_id[1:] != data.segment_id[:-1], True])
    for raw_left, raw_right in zip(boundaries[:-1], boundaries[1:], strict=True):
        left = max(int(start), int(raw_left))
        right = min(int(end), int(raw_right))
        if right - left < 3:
            continue
        observed = np.flatnonzero(data.observed[left:right])
        if len(observed) < 2:
            continue
        right = left + int(observed[-1]) + 1
        if right - left >= 3:
            yield left, right


def _ledger_for_segment(
    actions: np.ndarray,
    data: PreparedPolicyData,
    left: int,
    right: int,
    config: ReinforceConfig,
) -> list[dict[str, object]]:
    side = 0
    entry_index = -1
    entry_price = 0.0
    records: list[dict[str, object]] = []
    commission_round_trip = 2.0 * config.commission_per_fill_ticks
    slippage_price = config.slippage_per_fill_ticks * config.tick_size
    for offset, action in enumerate(actions):
        execution_index = left + offset + 1
        if action == int(Action.OPEN_LONG):
            side = 1
            entry_index = execution_index
            entry_price = float(data.first_ask[execution_index]) + slippage_price
        elif action == int(Action.OPEN_SHORT):
            side = -1
            entry_index = execution_index
            entry_price = float(data.first_bid[execution_index]) - slippage_price
        elif action == int(Action.CLOSE_LONG):
            exit_price = float(data.first_bid[execution_index]) - slippage_price
            net = (exit_price - entry_price) / config.tick_size - commission_round_trip
            records.append(_trade_record(data, entry_index, execution_index, side, entry_price, exit_price, net, False))
            side = 0
        elif action == int(Action.CLOSE_SHORT):
            exit_price = float(data.first_ask[execution_index]) + slippage_price
            net = (entry_price - exit_price) / config.tick_size - commission_round_trip
            records.append(_trade_record(data, entry_index, execution_index, side, entry_price, exit_price, net, False))
            side = 0
    if side:
        exit_index = right - 1
        exit_price = (
            float(data.first_bid[exit_index]) - slippage_price
            if side > 0
            else float(data.first_ask[exit_index]) + slippage_price
        )
        net = (
            (exit_price - entry_price) / config.tick_size
            if side > 0
            else (entry_price - exit_price) / config.tick_size
        ) - commission_round_trip
        records.append(_trade_record(data, entry_index, exit_index, side, entry_price, exit_price, net, True))
    return records


def _trade_record(data, entry_index, exit_index, side, entry_price, exit_price, net, terminal):
    return {
        "entry_index": int(entry_index),
        "exit_index": int(exit_index),
        "entry_ts_ns": int(data.ts_ns[entry_index]),
        "exit_ts_ns": int(data.ts_ns[exit_index]),
        "side": "long" if side > 0 else "short",
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "holding_seconds": int((data.ts_ns[exit_index] - data.ts_ns[entry_index]) // 1_000_000_000),
        "net_ticks": float(net),
        "terminal": bool(terminal),
    }


def evaluate_sequential(
    model: L2EventPolicy,
    data: PreparedPolicyData,
    config: ReinforceConfig,
    normalizer: RobustNormalizer,
    *,
    start: int,
    end: int,
    device: str = "auto",
    event_hazard_threshold: float | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    resolved = resolve_device(device)
    model = model.to(resolved)
    model.eval()
    x = normalizer.transform(data.x)
    all_records: list[dict[str, object]] = []
    episode_net: list[float] = []
    action_counts = np.zeros(5, dtype=np.int64)
    seconds = 0
    ranges = sorted(_segment_ranges(data, start, end), key=lambda value: value[1] - value[0])

    # Variable-length segments are padded only inside a batch.  Invalid padded
    # executions force NOOP in ``hazard_distribution`` and leave environment
    # equity untouched.  Sorting by length keeps padding small, while the
    # decision-cell cap prevents an unusually long segment from exhausting GPU
    # memory.  This turns thousands of tiny LSTM launches into a few dozen.
    batches: list[list[tuple[int, int]]] = []
    batch: list[tuple[int, int]] = []
    max_decisions = 0
    max_batch_size = 128
    max_padded_decision_cells = 262_144
    for bounds in ranges:
        decisions = bounds[1] - bounds[0] - 2
        prospective_max = max(max_decisions, decisions)
        if batch and (
            len(batch) >= max_batch_size
            or prospective_max * (len(batch) + 1) > max_padded_decision_cells
        ):
            batches.append(batch)
            batch = []
            max_decisions = 0
        batch.append(bounds)
        max_decisions = max(max_decisions, decisions)
    if batch:
        batches.append(batch)

    with torch.no_grad():
        for range_batch in batches:
            lengths = np.asarray(
                [right - left - 2 for left, right in range_batch], dtype=np.int64
            )
            padded_steps = int(lengths.max())
            batch_size = len(range_batch)
            feature_batch = np.zeros(
                (batch_size, padded_steps, data.x.shape[1]), dtype=np.float32
            )
            valid_batch = np.zeros((batch_size, padded_steps), dtype=bool)
            for row, (left, right) in enumerate(range_batch):
                length = int(lengths[row])
                feature_batch[row, :length] = x[left : right - 2]
                execution_slice = slice(left + 1, right - 1)
                valid_batch[row, :length] = data.observed[execution_slice]

            raw_logits, _ = model(torch.as_tensor(feature_batch, device=resolved))
            if event_hazard_threshold is None:
                decoded = deterministic_event_actions(
                    raw_logits.cpu(), torch.as_tensor(valid_batch)
                )
            else:
                decoded = threshold_event_actions(
                    raw_logits.cpu(),
                    torch.as_tensor(valid_batch),
                    event_hazard_threshold,
                )
            actions_batch = decoded.numpy().astype(np.int8)
            for row, (left, right) in enumerate(range_batch):
                actions = actions_batch[row, : lengths[row]]
                action_counts += np.bincount(actions, minlength=5)
                records = _ledger_for_segment(actions, data, left, right, config)
                rollout_net = float(sum(record["net_ticks"] for record in records))
                all_records.extend(records)
                episode_net.append(rollout_net)
                seconds += len(actions)

    # Length-sorted inference must not change the chronological drawdown path.
    all_records.sort(key=lambda record: (record["exit_ts_ns"], record["entry_ts_ns"]))
    trades = np.asarray([record["net_ticks"] for record in all_records], dtype=np.float64)
    episode_values = np.asarray(episode_net, dtype=np.float64)
    cumulative = np.cumsum(trades) if len(trades) else np.zeros(0)
    drawdown = np.maximum.accumulate(np.r_[0.0, cumulative]) - np.r_[0.0, cumulative]
    metrics: dict[str, object] = {
        "range": {"start": int(start), "end": int(end), "seconds_rows": int(end - start)},
        "sequential_segments": len(episode_values),
        "decision_seconds": seconds,
        "events": int(action_counts[1:].sum()),
        "event_rate": float(action_counts[1:].sum() / max(action_counts.sum(), 1)),
        "event_hazard_threshold": (
            None if event_hazard_threshold is None else float(event_hazard_threshold)
        ),
        "deterministic_decoder": (
            "probability_argmax" if event_hazard_threshold is None else "hazard_threshold"
        ),
        "actions": {
            "no_op": int(action_counts[0]),
            "open_long": int(action_counts[1]),
            "open_short": int(action_counts[2]),
            "close_long": int(action_counts[3]),
            "close_short": int(action_counts[4]),
        },
        "trades": int(len(trades)),
        "total_net_ticks": float(trades.sum()),
        "mean_trade_net_ticks": float(trades.mean()) if len(trades) else 0.0,
        "median_trade_net_ticks": float(np.median(trades)) if len(trades) else 0.0,
        "hit_rate": float((trades > 0).mean()) if len(trades) else 0.0,
        "profit_factor": _profit_factor(trades),
        "max_drawdown_ticks": float(drawdown.max()) if len(drawdown) else 0.0,
        "positive_segment_rate": float((episode_values > 0).mean()) if len(episode_values) else 0.0,
        "mean_segment_net_ticks": float(episode_values.mean()) if len(episode_values) else 0.0,
        "never_trade_baseline_ticks": 0.0,
        "costs": {
            "tick_size": config.tick_size,
            "commission_per_fill_ticks": config.commission_per_fill_ticks,
            "slippage_per_fill_ticks": config.slippage_per_fill_ticks,
            "spread_accounting": "inside executable bid/ask prices, not subtracted twice",
        },
    }
    return metrics, all_records


def write_evaluation(output_dir: str | Path, metrics: dict[str, object], records: list[dict[str, object]]) -> None:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    schema = pa.schema(
        [
            ("entry_index", pa.int64()), ("exit_index", pa.int64()),
            ("entry_ts_ns", pa.int64()), ("exit_ts_ns", pa.int64()),
            ("side", pa.string()), ("entry_price", pa.float64()),
            ("exit_price", pa.float64()), ("holding_seconds", pa.int64()),
            ("net_ticks", pa.float64()), ("terminal", pa.bool_()),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path / "trades.parquet", compression="zstd")
