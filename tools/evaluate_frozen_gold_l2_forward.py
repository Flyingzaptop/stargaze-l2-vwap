from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stargaze_ml.gold.l2_causal_rate import CausalRateConfig, causal_rate_select, summarize_selected
from stargaze_ml.gold.l2_contracts import assert_feature_names
from stargaze_ml.gold.frozen_policy import load_frozen_policy_bundle
from stargaze_ml.gold.l2_open_policy import L2OpenPolicy
from stargaze_ml.gold.l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from stargaze_ml.gold.l2_risk_direction import (
    L2RiskDirectionPolicy,
    RiskDirectionConfig,
    _trade_rows,
)
from stargaze_ml.gold.l2_risk_ensemble import ensemble_trade_rows
from stargaze_ml.training.data import RobustNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen policy on untouched forward L2")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--open-checkpoint", type=Path)
    parser.add_argument("--risk-checkpoint", type=Path)
    parser.add_argument("--policy-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    if args.bundle is not None:
        bundle = load_frozen_policy_bundle(args.bundle)
        open_path = bundle.open_checkpoint
        risk_paths = list(bundle.risk_checkpoints)
        policy_report = bundle.policy
    else:
        if args.open_checkpoint is None or args.risk_checkpoint is None or args.policy_report is None:
            raise ValueError("provide --bundle or all three checkpoint/policy paths")
        open_path = args.open_checkpoint.resolve(strict=True)
        risk_paths = [args.risk_checkpoint.resolve(strict=True)]
        policy_report = json.loads(args.policy_report.resolve(strict=True).read_text(encoding="utf-8"))
    data = PreparedOpenData(args.prepared)
    open_state = torch.load(open_path, map_location=device, weights_only=False)
    risk_states = [torch.load(path, map_location=device, weights_only=False) for path in risk_paths]
    risk_state = risk_states[0]
    assert_feature_names(tuple(open_state["feature_names"]), data.feature_names, artifact="open checkpoint")
    for state in risk_states:
        if "feature_names" in state:
            assert_feature_names(tuple(state["feature_names"]), data.feature_names, artifact="risk checkpoint")
        elif int(state["model_state"]["lstm.weight_ih_l0"].shape[1]) != len(data.feature_names):
            raise ValueError("legacy risk checkpoint input width does not match prepared features")
    if "frozen_policy" not in policy_report or not policy_report.get("score_history_tail"):
        raise ValueError("policy report lacks frozen_policy or causal score history")
    policy = policy_report["frozen_policy"]
    open_threshold = float(
        policy_report.get("open_threshold", risk_state["open_threshold"])
    )
    if not 0.0 <= open_threshold <= 1.0:
        raise ValueError("frozen open threshold must be in [0, 1]")

    market = OpenReinforceConfig(**risk_state["market_config"])
    risk_configs = [RiskDirectionConfig(**state["config"]) for state in risk_states]
    normalizer = RobustNormalizer.from_dict(risk_state["normalizer"])
    teacher = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    teacher.load_state_dict(open_state["model_state"])
    teacher.eval()
    risk_models = []
    for state in risk_states:
        risk = L2RiskDirectionPolicy(len(data.feature_names), market.hidden_size).to(device)
        risk.load_state_dict(state["model_state"])
        risk.eval()
        risk_models.append(risk)

    start = max(int(data.validation_end), 0)
    events = _event_indices(data, start, len(data.x), good_only=False)
    if len(risk_models) == 1:
        candidates = _trade_rows(
            risk_models[0], teacher, data, normalizer, events,
            open_threshold, device, market, risk_configs[0],
        )
    else:
        candidates = ensemble_trade_rows(
            risk_models, teacher, data, normalizer, events,
            open_threshold, device, market, risk_configs,
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
        "risk_models": len(risk_models),
        "open_threshold": open_threshold,
        "candidate_rows": candidates,
        "selected": summarize_selected(selected),
        "selected_trades": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "completed_events", "entry_candidates", "selected")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
