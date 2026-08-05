from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stargaze_ml.gold.l2_causal_rate import (
    CausalRateConfig,
    causal_rate_select,
    chronological_robust_validation,
    direction_and_score,
    summarize_selected,
)
from stargaze_ml.gold.l2_open_policy import L2OpenPolicy
from stargaze_ml.gold.l2_contracts import (
    assert_feature_names,
    assert_market_inference_contract,
    assert_normalizer_contract,
)
from stargaze_ml.gold.l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from stargaze_ml.gold.l2_risk_direction import L2RiskDirectionPolicy, RiskDirectionConfig
from stargaze_ml.gold.l2_risk_ensemble import ensemble_trade_rows
from stargaze_ml.training.data import RobustNormalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--open-checkpoint", type=Path, required=True)
    parser.add_argument("--risk-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--open-threshold",
        type=float,
        help="validation-selected threshold for a compatible replacement open model",
    )
    parser.add_argument(
        "--print-full-report",
        action="store_true",
        help="print the complete report instead of the concise summary",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    data = PreparedOpenData(args.prepared)
    open_state = torch.load(args.open_checkpoint, map_location=device, weights_only=False)
    states = [torch.load(path, map_location=device, weights_only=False) for path in args.risk_checkpoint]
    market = OpenReinforceConfig(**states[0]["market_config"])
    normalizer = RobustNormalizer.from_dict(states[0]["normalizer"])
    checkpoint_threshold = float(states[0]["open_threshold"])
    threshold = (
        checkpoint_threshold if args.open_threshold is None else float(args.open_threshold)
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("open threshold must be in [0, 1]")
    for state in states[1:]:
        assert_market_inference_contract(
            states[0]["market_config"], state["market_config"], artifact="risk ensemble"
        )
        assert_normalizer_contract(
            states[0]["normalizer"], state["normalizer"], artifact="risk ensemble"
        )
        if args.open_threshold is None and float(state["open_threshold"]) != checkpoint_threshold:
            raise ValueError("ensemble checkpoints have different open thresholds")
    for state in states:
        if "feature_names" in state:
            assert_feature_names(
                tuple(state["feature_names"]), data.feature_names, artifact="risk checkpoint"
            )
        elif int(state["model_state"]["lstm.weight_ih_l0"].shape[1]) != len(data.feature_names):
            raise ValueError("legacy risk checkpoint input width does not match prepared features")
    assert_feature_names(tuple(open_state["feature_names"]), data.feature_names, artifact="open checkpoint")

    teacher = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    teacher.load_state_dict(open_state["model_state"]); teacher.eval()
    models = []; configs = []
    for state in states:
        config = RiskDirectionConfig(**state["config"])
        model = L2RiskDirectionPolicy(len(data.feature_names), market.hidden_size).to(device)
        model.load_state_dict(state["model_state"]); model.eval()
        models.append(model); configs.append(config)

    val_rows = ensemble_trade_rows(
        models, teacher, data, normalizer,
        _event_indices(data, data.train_end, data.validation_end, good_only=False),
        threshold, device, market, configs,
    )
    test_rows = ensemble_trade_rows(
        models, teacher, data, normalizer,
        _event_indices(data, data.validation_end, len(data.x), good_only=False),
        threshold, device, market, configs,
    )
    day_ns = 86_400_000_000_000
    val_days = max(len({int(row["entry_ts_ns"]) // day_ns for row in val_rows}), 1)
    expected = len(val_rows) / val_days
    split_ts_ns = int(np.median([int(row["entry_ts_ns"]) for row in val_rows]))
    grid = []
    for mode in ("classifier", "value", "risk"):
        for penalty in ((0.0,) if mode != "risk" else (300.0, 600.0, 1000.0)):
            for field in (
                "opportunity_probability", "negative_tail_probability", "risk_edge",
                "negative_side_disagreement", "negative_tail_disagreement",
                "negative_risk_uncertainty", "risk_evidence",
                "side_confidence", "value_gap", "classifier_value_agreement",
                "risk_direction_margin",
            ):
                all_scores = [direction_and_score(row, mode=mode, penalty=penalty, filter_field=field)[1] for row in val_rows]
                for target in (10, 15, 20, 25):
                    quantile = float(np.clip(1.0 - target / expected, 0.0, 1.0))
                    fallback = float(np.quantile(all_scores, quantile))
                    config = CausalRateConfig(target_trades_per_day=target)
                    chosen = causal_rate_select(
                        val_rows, mode=mode, penalty=penalty, filter_field=field,
                        expected_candidates_per_day=expected, fallback_cutoff=fallback,
                        config=config,
                    )
                    metrics = summarize_selected(chosen)
                    stability = chronological_robust_validation(
                        chosen, split_ts_ns=split_ts_ns
                    )
                    grid.append({
                        "mode": mode, "penalty": penalty, "filter_field": field,
                        "fallback_cutoff": fallback, "target_trades_per_day": target,
                        **stability, **metrics,
                    })
    selected = max(grid, key=lambda row: float(row["selection_score"]))
    best_by_filter = {
        field: max(
            (row for row in grid if row["filter_field"] == field),
            key=lambda row: float(row["selection_score"]),
        )
        for field in sorted({str(row["filter_field"]) for row in grid})
    }
    initial_scores = [
        direction_and_score(
            row, mode=str(selected["mode"]), penalty=float(selected["penalty"]),
            filter_field=str(selected["filter_field"]),
        )[1]
        for row in val_rows
    ]
    test_chosen = causal_rate_select(
        test_rows, mode=str(selected["mode"]), penalty=float(selected["penalty"]),
        filter_field=str(selected["filter_field"]), expected_candidates_per_day=expected,
        fallback_cutoff=float(selected["fallback_cutoff"]),
        config=CausalRateConfig(target_trades_per_day=int(selected["target_trades_per_day"])),
        initial_scores=initial_scores,
    )
    all_scores = initial_scores + [
        direction_and_score(
            row,
            mode=str(selected["mode"]),
            penalty=float(selected["penalty"]),
            filter_field=str(selected["filter_field"]),
        )[1]
        for row in test_rows
    ]
    fixed_config = CausalRateConfig(
        target_trades_per_day=int(selected["target_trades_per_day"])
    )
    report = {
        "device": str(device), "checkpoint_count": len(states),
        "open_threshold": threshold,
        "validation_split_ts_ns": split_ts_ns,
        "validation_protocol": "minimum robust score across chronological halves with at least five trades per half",
        "validation_approved": bool(float(selected["selection_score"]) > 0.0),
        "expected_candidates_per_day": expected,
        "selected_on_validation": selected,
        "validation_best_by_filter": best_by_filter,
        "fixed_test": summarize_selected(test_chosen),
        "fixed_test_trades": test_chosen,
        "score_history_tail": all_scores[-fixed_config.history_size :],
        "frozen_policy": {
            "mode": str(selected["mode"]),
            "penalty": float(selected["penalty"]),
            "filter_field": str(selected["filter_field"]),
            "fallback_cutoff": float(selected["fallback_cutoff"]),
            "expected_candidates_per_day": expected,
            "target_trades_per_day": int(selected["target_trades_per_day"]),
            "history_size": fixed_config.history_size,
            "min_history": fixed_config.min_history,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.print_full_report:
        printed = report
    else:
        chosen = report["selected_on_validation"]
        printed = {
            "out": str(args.out.resolve()),
            "validation_approved": report["validation_approved"],
            "selected_on_validation": {
                key: chosen[key]
                for key in (
                    "mode", "penalty", "filter_field", "target_trades_per_day",
                    "selection_score", "trades", "mean_pnl_ticks",
                )
            },
            "chronological_half_scores": chosen["robust_scores"],
            "fixed_test": report["fixed_test"],
        }
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
