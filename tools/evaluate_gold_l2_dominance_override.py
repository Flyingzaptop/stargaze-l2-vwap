from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stargaze_ml.gold.l2_causal_rate import (
    CausalRateConfig,
    causal_rate_select,
    direction_and_score,
    robust_validation_score,
    summarize_selected,
)
from stargaze_ml.gold.l2_dominance_override import (
    DominanceOverrideConfig,
    apply_dominance_override,
    simulate_live_dominance_swaps,
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
    parser.add_argument("--rate-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    data = PreparedOpenData(args.prepared)
    open_state = torch.load(args.open_checkpoint, map_location=device, weights_only=False)
    state = torch.load(args.risk_checkpoint, map_location=device, weights_only=False)
    rate_report = json.loads(args.rate_report.read_text(encoding="utf-8"))
    market = OpenReinforceConfig(**state["market_config"])
    config = RiskDirectionConfig(**state["config"])
    normalizer = RobustNormalizer.from_dict(state["normalizer"])
    teacher = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    teacher.load_state_dict(open_state["model_state"]); teacher.eval()
    model = L2RiskDirectionPolicy(len(data.feature_names), market.hidden_size).to(device)
    model.load_state_dict(state["model_state"]); model.eval()
    policy = state["evaluation"]["selected_on_validation"]
    mode = str(policy["mode"]); penalty = float(policy["penalty"])
    field = str(policy["filter_field"]); fallback = float(policy["cutoff"])
    target = int(rate_report["selected_on_validation"]["target_trades_per_day"])
    threshold = float(state["open_threshold"])
    val_rows = _trade_rows(
        model, teacher, data, normalizer,
        _event_indices(data, data.train_end, data.validation_end, good_only=False),
        threshold, device, market, config,
    )
    test_rows = _trade_rows(
        model, teacher, data, normalizer,
        _event_indices(data, data.validation_end, len(data.x), good_only=False),
        threshold, device, market, config,
    )
    day_ns = 86_400_000_000_000
    val_days = max(len({int(row["entry_ts_ns"]) // day_ns for row in val_rows}), 1)
    expected = len(val_rows) / val_days
    rate_config = CausalRateConfig(target_trades_per_day=target)
    val_selected = causal_rate_select(
        val_rows, mode=mode, penalty=penalty, filter_field=field,
        expected_candidates_per_day=expected, fallback_cutoff=fallback, config=rate_config,
    )
    initial_scores = [
        direction_and_score(row, mode=mode, penalty=penalty, filter_field=field)[1]
        for row in val_rows
    ]
    test_selected = causal_rate_select(
        test_rows, mode=mode, penalty=penalty, filter_field=field,
        expected_candidates_per_day=expected, fallback_cutoff=fallback,
        config=rate_config, initial_scores=initial_scores,
    )
    grid = [{
        "config": None, "swaps": 0,
        "selection_score": robust_validation_score(summarize_selected(val_selected)),
        **summarize_selected(val_selected),
    }]
    horizon_sets = ((5, 10, 15, 30, 45), (5, 10, 15, 30, 45, 60), (5, 10, 15, 30, 45, 60, 120))
    for horizons in horizon_sets:
        for alignment in (0.6, 0.8, 1.0):
            for slope_consensus in (0.6, 0.8, 1.0):
                for minimum in (0.0, 1.0, 3.0, 5.0):
                    for min_delta in (0.0, 200.0, 300.0, 400.0):
                        override = DominanceOverrideConfig(
                            horizons=horizons, alignment_consensus=alignment,
                            slope_consensus=slope_consensus,
                            min_projected_slope_ticks=minimum,
                            min_abs_delta_ticks=min_delta,
                        )
                        changed, swaps = apply_dominance_override(
                            val_selected, data.x, data.feature_names, override
                        )
                        metrics = summarize_selected(changed)
                        grid.append({
                            "config": {
                                "horizons": horizons, "alignment_consensus": alignment,
                                "slope_consensus": slope_consensus,
                                "min_projected_slope_ticks": minimum,
                                "min_abs_delta_ticks": min_delta,
                            },
                            "swaps": swaps, "selection_score": robust_validation_score(metrics),
                            **metrics,
                        })
    baseline = summarize_selected(val_selected)
    selected = max(grid, key=lambda row: float(row["selection_score"]))
    if selected["config"] is None:
        fixed_rows, fixed_swaps = test_selected, 0
    else:
        selected_config = DominanceOverrideConfig(**selected["config"])
        fixed_rows, fixed_swaps = apply_dominance_override(
            test_selected, data.x, data.feature_names, selected_config
        )
    report = {
        "base_policy": {"mode": mode, "penalty": penalty, "filter_field": field, "target_trades_per_day": target},
        "validation_baseline": baseline,
        "selected_on_validation": selected,
        "fixed_test_swaps": fixed_swaps,
        "fixed_test": summarize_selected(fixed_rows),
        "fixed_test_trades": fixed_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
