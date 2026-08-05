from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stargaze_ml.gold.frozen_policy import load_frozen_policy_bundle
from stargaze_ml.gold.l2_causal_rate import CausalRateConfig, causal_rate_select
from stargaze_ml.gold.l2_frozen_runtime import FrozenL2PolicyRuntime
from stargaze_ml.gold.l2_open_policy import L2OpenPolicy
from stargaze_ml.gold.l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from stargaze_ml.gold.l2_risk_direction import (
    L2RiskDirectionPolicy,
    RiskDirectionConfig,
    _trade_rows,
)
from stargaze_ml.gold.l2_risk_ensemble import ensemble_trade_rows
from stargaze_ml.training.data import RobustNormalizer


FIELDS = (
    "side_probability",
    "predicted_long_pnl",
    "predicted_short_pnl",
    "long_tail_probability",
    "short_tail_probability",
    "opportunity_probability",
    "side_probability_std",
    "predicted_long_pnl_std",
    "predicted_short_pnl_std",
    "long_tail_probability_std",
    "short_tail_probability_std",
    "opportunity_probability_std",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify batch and stateful LSTM inference parity")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/gold_l2_v1"))
    parser.add_argument("--events", type=int, default=25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    data = PreparedOpenData(args.prepared)
    bundle = load_frozen_policy_bundle(args.bundle)
    open_state = torch.load(bundle.open_checkpoint, map_location=device, weights_only=False)
    risk_states = [
        torch.load(path, map_location=device, weights_only=False)
        for path in bundle.risk_checkpoints
    ]
    risk_state = risk_states[0]
    market = OpenReinforceConfig(**risk_state["market_config"])
    risk_configs = [RiskDirectionConfig(**state["config"]) for state in risk_states]
    open_threshold = float(
        bundle.policy.get("open_threshold", risk_state["open_threshold"])
    )
    normalizer = RobustNormalizer.from_dict(risk_state["normalizer"])
    teacher = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    teacher.load_state_dict(open_state["model_state"])
    risk_models = []
    for state in risk_states:
        risk = L2RiskDirectionPolicy(len(data.feature_names), market.hidden_size).to(device)
        risk.load_state_dict(state["model_state"])
        risk_models.append(risk)
    events = _event_indices(data, data.validation_end, len(data.x), good_only=False)[: args.events]
    if len(risk_models) == 1:
        offline = _trade_rows(
            risk_models[0], teacher, data, normalizer, events,
            open_threshold, device, market, risk_configs[0],
        )
    else:
        offline = ensemble_trade_rows(
            risk_models, teacher, data, normalizer, events,
            open_threshold, device, market, risk_configs,
        )
    runtime = FrozenL2PolicyRuntime(args.bundle, device_name=str(device))
    streaming = []
    for event in events:
        runtime.reset_event(int(event))
        for index in range(int(data.event_start[event]), int(data.event_crossing_1[event])):
            decision = runtime.process_index(data, index)
            if decision is not None:
                streaming.append(decision)
                break
    offline_by_entry = {int(row["entry_index"]): row for row in offline}
    streaming_by_entry = {int(row["entry_index"]): row for row in streaming}
    entry_mismatches = len(set(offline_by_entry) ^ set(streaming_by_entry))
    policy = bundle.policy["frozen_policy"]
    offline_selected = causal_rate_select(
        offline,
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
        initial_scores=[float(value) for value in bundle.policy["score_history_tail"]],
    )
    offline_accepted = {int(row["entry_index"]) for row in offline_selected}
    streaming_accepted = {
        int(row["entry_index"]) for row in streaming if bool(row["accepted"])
    }
    acceptance_mismatches = len(offline_accepted ^ streaming_accepted)
    differences = {field: [] for field in FIELDS}
    for entry in set(offline_by_entry) & set(streaming_by_entry):
        expected = offline_by_entry[entry]
        actual = streaming_by_entry[entry]
        for field in FIELDS:
            differences[field].append(abs(float(expected[field]) - float(actual[field])))
    maxima = {
        field: float(np.max(values)) if values else 0.0 for field, values in differences.items()
    }
    report = {
        "events_checked": int(len(events)),
        "candidates": len(offline),
        "risk_models": len(risk_models),
        "entry_mismatches": entry_mismatches,
        "acceptance_mismatches": acceptance_mismatches,
        "max_absolute_difference": maxima,
    }
    if entry_mismatches or acceptance_mismatches or max(maxima.values(), default=0.0) > 1e-4:
        raise RuntimeError(json.dumps(report))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
