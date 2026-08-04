from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stargaze_ml.gold.l2_causal_rate import (
    CausalRateConfig,
    causal_rate_select,
    direction_and_score,
    summarize_selected,
)
from stargaze_ml.gold.l2_open_policy import L2OpenPolicy
from stargaze_ml.gold.l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from stargaze_ml.gold.l2_risk_direction import L2RiskDirectionPolicy, RiskDirectionConfig, _trade_rows
from stargaze_ml.training.data import RobustNormalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--open-checkpoint", type=Path, required=True)
    parser.add_argument("--risk-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    data = PreparedOpenData(args.prepared)
    open_state = torch.load(args.open_checkpoint, map_location=device, weights_only=False)
    risk_state = torch.load(args.risk_checkpoint, map_location=device, weights_only=False)
    market = OpenReinforceConfig(**risk_state["market_config"])
    risk_config = RiskDirectionConfig(**risk_state["config"])
    normalizer = RobustNormalizer.from_dict(risk_state["normalizer"])

    teacher = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    teacher.load_state_dict(open_state["model_state"])
    risk = L2RiskDirectionPolicy(len(data.feature_names), market.hidden_size).to(device)
    risk.load_state_dict(risk_state["model_state"])
    selected = risk_state["evaluation"]["selected_on_validation"]
    mode = str(selected["mode"]); penalty = float(selected["penalty"])
    field = str(selected["filter_field"]); fallback = float(selected["cutoff"])
    threshold = float(risk_state["open_threshold"])

    val_rows = _trade_rows(
        risk, teacher, data, normalizer,
        _event_indices(data, data.train_end, data.validation_end, good_only=False),
        threshold, device, market, risk_config,
    )
    test_rows = _trade_rows(
        risk, teacher, data, normalizer,
        _event_indices(data, data.validation_end, len(data.x), good_only=False),
        threshold, device, market, risk_config,
    )
    val_days = max(len({int(row["entry_ts_ns"]) // 86_400_000_000_000 for row in val_rows}), 1)
    expected = len(val_rows) / val_days
    initial_scores = [
        direction_and_score(row, mode=mode, penalty=penalty, filter_field=field)[1]
        for row in val_rows
    ]
    grid = []
    for target in (10, 15, 20, 25):
        config = CausalRateConfig(target_trades_per_day=target)
        val_chosen = causal_rate_select(
            val_rows, mode=mode, penalty=penalty, filter_field=field,
            expected_candidates_per_day=expected, fallback_cutoff=fallback, config=config,
        )
        metrics = summarize_selected(val_chosen)
        metrics["selection_score"] = float(metrics["mean_pnl_ticks"]) + 0.05 * float(metrics.get("p05_pnl_ticks", 0.0))
        grid.append({"target_trades_per_day": target, **metrics})
    best = max(grid, key=lambda row: float(row["selection_score"]))
    fixed_config = CausalRateConfig(target_trades_per_day=int(best["target_trades_per_day"]))
    test_chosen = causal_rate_select(
        test_rows, mode=mode, penalty=penalty, filter_field=field,
        expected_candidates_per_day=expected, fallback_cutoff=fallback,
        config=fixed_config, initial_scores=initial_scores,
    )
    report = {
        "device": str(device), "mode": mode, "penalty": penalty,
        "filter_field": field, "fallback_cutoff": fallback,
        "expected_candidates_per_day": expected,
        "validation_grid": grid, "selected_on_validation": best,
        "fixed_test": summarize_selected(test_chosen),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
