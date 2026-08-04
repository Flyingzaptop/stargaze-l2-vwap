from __future__ import annotations

import numpy as np
import pytest

from stargaze_ml.gold.l2_policy import L2EventPolicy
from stargaze_ml.gold.l2_reinforce import (
    EpisodeSampler,
    PreparedPolicyData,
    ReinforceConfig,
    ReinforceTrainer,
    evaluate_sequential,
)
from stargaze_ml.training.data import RobustNormalizer


def _data(rows: int = 1000) -> PreparedPolicyData:
    ts = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * 1_000_000_000
    bid = 100.0 + np.arange(rows) * 0.01
    ask = bid + 0.02
    x = np.column_stack((np.sin(np.arange(rows) / 10), np.cos(np.arange(rows) / 17))).astype(np.float32)
    return PreparedPolicyData(
        ts_ns=ts,
        segment_id=np.zeros(rows, dtype=np.int32),
        x=x,
        feature_names=("sin", "cos"),
        valid_feature=np.ones(rows, dtype=bool),
        observed=np.ones(rows, dtype=bool),
        first_bid=bid,
        first_ask=ask,
        train_end=600,
        validation_end=800,
    )


def test_episode_sampler_keeps_terminal_inside_split_and_segment() -> None:
    data = _data()
    sampler = EpisodeSampler(data, 0, data.train_end, 32, seed=4)
    batches = list(sampler.iter_epoch_batches(7, epoch=0))
    # 598 eligible decisions -> 19 chunks -> final short batch of five.
    assert [len(feature) for feature, _, _, _, _ in batches] == [7, 7, 5]

    covered = []
    for feature, execution, terminal, mask, sequence_mask in batches:
        assert np.all(execution[mask] == feature[mask] + 1)
        last_real = sequence_mask.sum(axis=1) - 1
        assert np.all(terminal == feature[np.arange(len(feature)), last_real] + 2)
        assert np.all(terminal < data.train_end)
        covered.append(feature[mask])
    covered_rows = np.concatenate(covered)
    assert len(covered_rows) == sampler.eligible_count == 598
    assert len(np.unique(covered_rows)) == len(covered_rows)
    assert np.array_equal(np.sort(covered_rows), sampler.eligible_rows)

    second_order = np.concatenate(
        [
            feature[mask]
            for feature, _, _, mask, _ in sampler.iter_epoch_batches(7, epoch=1)
        ]
    )
    assert not np.array_equal(covered_rows, second_order)
    assert np.array_equal(np.sort(second_order), sampler.eligible_rows)


def test_internal_missing_execution_is_masked_without_splitting_causal_chunk() -> None:
    data = _data()
    # Decision t=6 executes at t+1=7.  Removing that execution must not split
    # the continuous [0, 15] LSTM sequence or discard later decisions.
    data.observed[7] = False
    sampler = EpisodeSampler(data, 0, data.train_end, 16, seed=3)
    assert np.array_equal(sampler.chunks[0], np.arange(16))
    assert not sampler.chunk_decision_masks[0][6]
    assert sampler.chunk_decision_masks[0][5]
    assert sampler.chunk_decision_masks[0][7]
    assert len(sampler.chunks) == 38
    assert sampler.eligible_count == 597


def test_tiny_reinforce_epoch_runs(tmp_path) -> None:
    data = _data()
    config = ReinforceConfig(
        hidden_size=8,
        episode_length=16,
        batch_size=4,
        epochs=1,
        entropy_warmup_epochs=1,
        use_amp=False,
    )
    model = L2EventPolicy(2, 8, initial_event_bias=-2.0)
    trainer = ReinforceTrainer(model, data, config, tmp_path / "run", device="cpu")
    stats = trainer.train_epoch(0)
    assert stats["episodes"] == 38
    assert stats["covered_decision_rows"] == 598
    assert stats["eligible_decision_rows"] == 598
    assert stats["coverage_ratio"] == 1.0
    assert stats["sampled_seconds"] == 598
    assert sum(stats["actions"].values()) == 598
    assert stats["event_rate"] == stats["events"] / 598
    assert 0.0 <= stats["mean_event_probability"] <= 1.0
    assert stats["mean_total_event_hazard"] >= 0.0


def test_never_event_policy_matches_zero_baseline() -> None:
    data = _data()
    config = ReinforceConfig(
        hidden_size=8,
        episode_length=16,
        batch_size=4,
        epochs=1,
        entropy_warmup_epochs=1,
        initial_event_bias=-80.0,
        use_amp=False,
    )
    model = L2EventPolicy(2, 8, initial_event_bias=-80.0)
    normalizer = RobustNormalizer.fit(data.x, np.arange(len(data)) < data.train_end)
    metrics, records = evaluate_sequential(
        model,
        data,
        config,
        normalizer,
        start=data.train_end,
        end=data.validation_end,
        device="cpu",
        event_hazard_threshold=0.5,
    )
    assert records == []
    assert metrics["trades"] == 0
    assert metrics["total_net_ticks"] == 0.0
    assert metrics["event_hazard_threshold"] == 0.5
    assert metrics["deterministic_decoder"] == "hazard_threshold"


def test_entropy_coefficient_rises_during_warmup_then_decays(tmp_path) -> None:
    data = _data()
    config = ReinforceConfig(
        hidden_size=8,
        episode_length=16,
        batch_size=8,
        epochs=7,
        entropy_start=0.001,
        entropy_peak=0.01,
        entropy_end=0.0001,
        entropy_warmup_epochs=3,
        use_amp=False,
    )
    trainer = ReinforceTrainer(
        L2EventPolicy(2, 8), data, config, tmp_path / "schedule", device="cpu"
    )
    coefficients = [trainer._entropy_coefficient(epoch) for epoch in range(config.epochs)]
    temperatures = [trainer._hazard_temperature(epoch) for epoch in range(config.epochs)]
    event_floors = [trainer._event_exploration_floor(epoch) for epoch in range(config.epochs)]
    assert coefficients[0] == pytest.approx(0.001)
    assert coefficients[2] == pytest.approx(0.01)
    assert coefficients[-1] == pytest.approx(0.0001)
    assert coefficients[:3] == sorted(coefficients[:3])
    assert coefficients[2:] == sorted(coefficients[2:], reverse=True)
    assert temperatures[0] == pytest.approx(1.0)
    assert temperatures[2] == pytest.approx(1.3)
    assert temperatures[-1] == pytest.approx(0.9)
    assert temperatures[:3] == sorted(temperatures[:3])
    assert temperatures[2:] == sorted(temperatures[2:], reverse=True)
    assert event_floors[0] == pytest.approx(0.005)
    assert event_floors[2] == pytest.approx(0.03)
    assert event_floors[-1] == pytest.approx(0.0005)
    assert event_floors[:3] == sorted(event_floors[:3])
    assert event_floors[2:] == sorted(event_floors[2:], reverse=True)


def test_train_cli_exposes_full_pass_exploration_schedule() -> None:
    from tools.train_gold_l2_policy import parser

    help_text = parser().format_help()
    train_help = parser()._subparsers._group_actions[0].choices["train"].format_help()
    evaluate_help = parser()._subparsers._group_actions[0].choices["evaluate"].format_help()
    assert "episodes-per-epoch" not in help_text + train_help
    assert "--entropy-peak" in train_help
    assert "--entropy-warmup-epochs" in train_help
    assert "--temperature-start" in train_help
    assert "--temperature-peak" in train_help
    assert "--temperature-end" in train_help
    assert "--event-floor-start" in train_help
    assert "--event-floor-peak" in train_help
    assert "--event-floor-end" in train_help
    assert "--event-hazard-threshold" in evaluate_help
