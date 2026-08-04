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
        risk_state = torch.load(
            self.bundle.risk_checkpoint, map_location=self.device, weights_only=False
        )
        expected_names = tuple(str(name) for name in self.bundle.policy["feature_names"])
        assert_feature_names(
            tuple(open_state["feature_names"]), expected_names, artifact="open checkpoint"
        )
        if "feature_names" in risk_state:
            assert_feature_names(
                tuple(risk_state["feature_names"]), expected_names, artifact="risk checkpoint"
            )
        self.feature_names = expected_names
        self.market = OpenReinforceConfig(**risk_state["market_config"])
        self.risk_config = RiskDirectionConfig(**risk_state["config"])
        self.normalizer = RobustNormalizer.from_dict(risk_state["normalizer"])
        self.open_model = L2OpenPolicy(len(expected_names), self.market.hidden_size).to(self.device)
        self.open_model.load_state_dict(open_state["model_state"])
        self.open_model.eval()
        self.risk_model = L2RiskDirectionPolicy(len(expected_names), self.market.hidden_size).to(
            self.device
        )
        self.risk_model.load_state_dict(risk_state["model_state"])
        self.risk_model.eval()
        self.open_threshold = float(risk_state["open_threshold"])
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
        self._risk_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_event(self, event_id: int) -> None:
        self._event_id = int(event_id)
        self._candidate_considered = False
        self._open_state = None
        self._risk_state = None

    def _step_models(self, row: np.ndarray) -> tuple[float, tuple[float, ...]]:
        normalized = self.normalizer.transform(np.asarray(row, dtype=np.float32)[None])
        tensor = torch.from_numpy(normalized)[None].to(self.device)
        with torch.no_grad():
            open_encoded, self._open_state = self.open_model.lstm(tensor, self._open_state)
            open_logit = self.open_model.open_head(open_encoded).squeeze()
            risk_encoded, self._risk_state = self.risk_model.lstm(tensor, self._risk_state)
            risk_values = tuple(
                head(risk_encoded).squeeze()
                for head in (
                    self.risk_model.side_head,
                    self.risk_model.long_value_head,
                    self.risk_model.short_value_head,
                    self.risk_model.long_tail_head,
                    self.risk_model.short_tail_head,
                    self.risk_model.opportunity_head,
                )
            )
        probability = float(torch.sigmoid(open_logit).cpu())
        return probability, tuple(float(value.cpu()) for value in risk_values)

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
        side_logit, long_value, short_value, long_tail_logit, short_tail_logit, opportunity_logit = risk
        row: dict[str, float | int | bool] = {
            "event_id": event_id,
            "entry_index": index,
            "entry_ts_ns": int(data.ts_ns[index]),
            "open_probability": open_probability,
            "side_probability": float(torch.sigmoid(torch.tensor(side_logit))),
            "predicted_long_pnl": float(np.sinh(long_value) * self.risk_config.pnl_scale_ticks),
            "predicted_short_pnl": float(np.sinh(short_value) * self.risk_config.pnl_scale_ticks),
            "long_tail_probability": float(torch.sigmoid(torch.tensor(long_tail_logit))),
            "short_tail_probability": float(torch.sigmoid(torch.tensor(short_tail_logit))),
            "opportunity_probability": float(torch.sigmoid(torch.tensor(opportunity_logit))),
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
