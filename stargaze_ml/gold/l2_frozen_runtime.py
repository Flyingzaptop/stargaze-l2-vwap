from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import polars as pl

from stargaze_ml.training.data import RobustNormalizer
from .frozen_policy import FrozenPolicyBundle, load_frozen_policy_bundle
from .l2_causal_rate import CausalRateConfig, CausalRateController
from .l2_contracts import assert_feature_names
from .l2_open_policy import L2OpenPolicy
from .l2_open_events import build_open_policy_data
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from .l2_risk_direction import L2RiskDirectionPolicy, RiskDirectionConfig


@dataclass(frozen=True)
class LivePolicyView:
    ts_ns: np.ndarray
    x: np.ndarray
    feature_names: tuple[str, ...]
    valid_feature: np.ndarray
    observed: np.ndarray
    gate_open: np.ndarray
    event_id: np.ndarray
    first_bid: np.ndarray
    first_ask: np.ndarray


def build_live_policy_view(seconds: pl.DataFrame, bundle_path: Path) -> LivePolicyView:
    bundle = load_frozen_policy_bundle(bundle_path)
    preparation = bundle.policy["preparation"]
    primary_text = str(preparation["primary_vwap"])
    primary: int | str = "ribbon" if primary_text == "ribbon" else int(primary_text)
    policy = build_open_policy_data(
        seconds,
        tick_size=float(preparation["tick_size"]),
        amplitude_threshold_ticks=float(preparation["amplitude_threshold_ticks"]),
        gate_fraction=float(preparation["gate_fraction"]),
        min_duration_seconds=int(preparation["min_duration_seconds"]),
        primary_vwap=primary,
        feature_profile=str(preparation["feature_profile"]),
    )
    names = tuple(policy.feature_names)
    assert_feature_names(
        tuple(bundle.policy["feature_names"]), names, artifact="live feature builder"
    )
    return LivePolicyView(
        ts_ns=seconds["bar_start_ns"].to_numpy().astype(np.int64),
        x=policy.x,
        feature_names=names,
        valid_feature=policy.valid_feature,
        observed=seconds["observed"].to_numpy().astype(bool),
        gate_open=policy.gate_open,
        event_id=policy.event_id,
        first_bid=seconds["first_bid"].to_numpy().astype(np.float64),
        first_ask=seconds["first_ask"].to_numpy().astype(np.float64),
    )


class FrozenL2PolicyRuntime:
    """Stateful, one-row-at-a-time runtime for the frozen open/direction policy."""

    def __init__(self, bundle_path: Path, *, device_name: str = "auto") -> None:
        self.bundle: FrozenPolicyBundle = load_frozen_policy_bundle(bundle_path)
        self.device = torch.device(
            "cuda"
            if device_name == "auto" and torch.cuda.is_available()
            else device_name if device_name != "auto" else "cpu"
        )
        open_state = torch.load(
            self.bundle.open_checkpoint, map_location=self.device, weights_only=False
        )
        risk_states = [
            torch.load(path, map_location=self.device, weights_only=False)
            for path in self.bundle.risk_checkpoints
        ]
        expected_names = tuple(str(name) for name in self.bundle.policy["feature_names"])
        assert_feature_names(
            tuple(open_state["feature_names"]), expected_names, artifact="open checkpoint"
        )
        for risk_state in risk_states:
            if "feature_names" in risk_state:
                assert_feature_names(
                    tuple(risk_state["feature_names"]), expected_names, artifact="risk checkpoint"
                )
        self.feature_names = expected_names
        self.market = OpenReinforceConfig(**risk_states[0]["market_config"])
        self.risk_configs = [RiskDirectionConfig(**state["config"]) for state in risk_states]
        self.normalizer = RobustNormalizer.from_dict(risk_states[0]["normalizer"])
        self.open_model = L2OpenPolicy(len(expected_names), self.market.hidden_size).to(self.device)
        self.open_model.load_state_dict(open_state["model_state"])
        self.open_model.eval()
        self.risk_models = []
        for state in risk_states:
            model = L2RiskDirectionPolicy(len(expected_names), self.market.hidden_size).to(self.device)
            model.load_state_dict(state["model_state"])
            model.eval()
            self.risk_models.append(model)
        self.open_threshold = float(risk_states[0]["open_threshold"])
        if any(float(state["open_threshold"]) != self.open_threshold for state in risk_states[1:]):
            raise ValueError("risk ensemble has inconsistent open thresholds")
        policy = self.bundle.policy["frozen_policy"]
        self.controller = CausalRateController(
            mode=str(policy["mode"]),
            penalty=float(policy["penalty"]),
            filter_field=str(policy["filter_field"]),
            expected_candidates_per_day=float(policy["expected_candidates_per_day"]),
            fallback_cutoff=float(policy["fallback_cutoff"]),
            config=CausalRateConfig(
                target_trades_per_day=int(policy["target_trades_per_day"]),
                history_size=int(policy["history_size"]),
                min_history=int(policy["min_history"]),
            ),
            initial_scores=[float(value) for value in self.bundle.policy["score_history_tail"]],
        )
        self._event_id: int | None = None
        self._candidate_considered = False
        self._open_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._risk_states: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.risk_models
        ]

    def reset_event(self, event_id: int) -> None:
        self._event_id = int(event_id)
        self._candidate_considered = False
        self._open_state = None
        self._risk_states = [None for _ in self.risk_models]

    def _step_models(self, row: np.ndarray) -> tuple[float, tuple[float, ...]]:
        normalized = self.normalizer.transform(np.asarray(row, dtype=np.float32)[None])
        tensor = torch.from_numpy(normalized)[None].to(self.device)
        with torch.no_grad():
            open_encoded, self._open_state = self.open_model.lstm(tensor, self._open_state)
            open_logit = self.open_model.open_head(open_encoded).squeeze()
            predictions = []
            for model_index, (model, config) in enumerate(
                zip(self.risk_models, self.risk_configs, strict=True)
            ):
                encoded, state = model.lstm(tensor, self._risk_states[model_index])
                self._risk_states[model_index] = state
                raw = tuple(
                    head(encoded).squeeze()
                    for head in (
                        model.side_head,
                        model.long_value_head,
                        model.short_value_head,
                        model.long_tail_head,
                        model.short_tail_head,
                        model.opportunity_head,
                    )
                )
                predictions.append(
                    (
                        float(torch.sigmoid(raw[0]).cpu()),
                        float(np.sinh(float(raw[1].cpu())) * config.pnl_scale_ticks),
                        float(np.sinh(float(raw[2].cpu())) * config.pnl_scale_ticks),
                        float(torch.sigmoid(raw[3]).cpu()),
                        float(torch.sigmoid(raw[4]).cpu()),
                        float(torch.sigmoid(raw[5]).cpu()),
                    )
                )
        probability = float(torch.sigmoid(open_logit).cpu())
        return probability, tuple(np.mean(np.asarray(predictions), axis=0).tolist())

    def process_index(
        self,
        data: PreparedOpenData | LivePolicyView,
        index: int,
        *,
        require_next_observed: bool = True,
    ) -> dict[str, float | int | bool] | None:
        if data.feature_names != self.feature_names:
            assert_feature_names(self.feature_names, data.feature_names, artifact="prepared data")
        index = int(index)
        if index < 0 or index >= len(data.x):
            raise IndexError("runtime index outside prepared data")
        event_id = int(data.event_id[index])
        if event_id != self._event_id:
            self.reset_event(event_id)
        if self._candidate_considered:
            return None
        open_probability, risk = self._step_models(data.x[index])
        next_observed = index + 1 < len(data.x) and bool(data.observed[index + 1])
        allowed = bool(data.gate_open[index] and data.valid_feature[index])
        if require_next_observed:
            allowed = allowed and next_observed
        if not allowed or open_probability < self.open_threshold:
            return None
        self._candidate_considered = True
        side_probability, predicted_long, predicted_short, long_tail, short_tail, opportunity = risk
        row: dict[str, float | int | bool] = {
            "event_id": event_id,
            "entry_index": index,
            "entry_ts_ns": int(data.ts_ns[index]),
            "open_probability": open_probability,
            "side_probability": side_probability,
            "predicted_long_pnl": predicted_long,
            "predicted_short_pnl": predicted_short,
            "long_tail_probability": long_tail,
            "short_tail_probability": short_tail,
            "opportunity_probability": opportunity,
            "next_bbo_observed": next_observed,
        }
        accepted = self.controller.consider(row)
        if accepted is None:
            row["accepted"] = False
            return row
        accepted["accepted"] = True
        return accepted

    def replay_completed(
        self,
        data: PreparedOpenData,
        *,
        left: int,
        right: int,
    ) -> dict[str, list[dict[str, float | int | bool]]]:
        candidates: list[dict[str, float | int | bool]] = []
        events = _event_indices(data, int(left), int(right), good_only=False)
        for event in events:
            start = int(data.event_start[event])
            crossing = int(data.event_crossing_1[event])
            self.reset_event(int(event))
            for index in range(start, crossing):
                decision = self.process_index(data, index)
                if decision is not None:
                    candidates.append(decision)
                    break
        return {
            "candidates": candidates,
            "selected": [row for row in candidates if bool(row["accepted"])],
        }
