from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stargaze_ml.gold.l2_causal_rate import CausalRateConfig, causal_rate_select, summarize_selected
from stargaze_ml.gold.l2_contracts import assert_feature_names
from stargaze_ml.gold.l2_open_policy import L2OpenPolicy
from stargaze_ml.gold.l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from stargaze_ml.gold.l2_risk_direction import (
    L2RiskDirectionPolicy,
    RiskDirectionConfig,
    _trade_rows,
)
from stargaze_ml.training.data import RobustNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen policy on untouched forward L2")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--open-checkpoint", type=Path, required=True)
    parser.add_argument("--risk-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    data = PreparedOpenData(args.prepared)
    open_state = torch.load(args.open_checkpoint.resolve(strict=True), map_location=device, weights_only=False)
    risk_state = torch.load(args.risk_checkpoint.resolve(strict=True), map_location=device, weights_only=False)
    assert_feature_names(tuple(open_state["feature_names"]), data.feature_names, artifact="open checkpoint")
    if "feature_names" in risk_state:
        assert_feature_names(tuple(risk_state["feature_names"]), data.feature_names, artifact="risk checkpoint")
    elif int(risk_state["model_state"]["lstm.weight_ih_l0"].shape[1]) != len(data.feature_names):
        raise ValueError("legacy risk checkpoint input width does not match prepared features")
    policy_report = json.loads(args.policy_report.resolve(strict=True).read_text(encoding="utf-8"))
    if "frozen_policy" not in policy_report or not policy_report.get("score_history_tail"):
        raise ValueError("policy report lacks frozen_policy or causal score history")
    policy = policy_report["frozen_policy"]

    market = OpenReinforceConfig(**risk_state["market_config"])
    risk_config = RiskDirectionConfig(**risk_state["config"])
    normalizer = RobustNormalizer.from_dict(risk_state["normalizer"])
    teacher = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    teacher.load_state_dict(open_state["model_state"])
    teacher.eval()
    risk = L2RiskDirectionPolicy(len(data.feature_names), market.hidden_size).to(device)
    risk.load_state_dict(risk_state["model_state"])
    risk.eval()

    start = max(int(data.validation_end), 0)
    events = _event_indices(data, start, len(data.x), good_only=False)
    candidates = _trade_rows(
        risk,
        teacher,
        data,
        normalizer,
        events,
        float(risk_state["open_threshold"]),
        device,
        market,
        risk_config,
    )
    rate_config = CausalRateConfig(
        target_trades_per_day=int(policy["target_trades_per_day"]),
        history_size=int(policy["history_size"]),
        min_history=int(policy["min_history"]),
    )
    selected = causal_rate_select(
        candidates,
        mode=str(policy["mode"]),
        penalty=float(policy["penalty"]),
        filter_field=str(policy["filter_field"]),
        expected_candidates_per_day=float(policy["expected_candidates_per_day"]),
        fallback_cutoff=float(policy["fallback_cutoff"]),
        config=rate_config,
        initial_scores=[float(value) for value in policy_report["score_history_tail"]],
    )
    report = {
        "evaluation_contract": "frozen checkpoints, threshold, direction rule, rate and score history",
        "device": str(device),
        "rows": len(data.x),
        "completed_events": int(len(events)),
        "entry_candidates": len(candidates),
        "selected": summarize_selected(selected),
        "selected_trades": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "completed_events", "entry_candidates", "selected")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
