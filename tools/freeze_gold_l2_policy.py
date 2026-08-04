from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from stargaze_ml.gold.l2_contracts import assert_feature_names
from stargaze_ml.gold.frozen_policy import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a complete forward-evaluation policy bundle")
    parser.add_argument("--open-checkpoint", type=Path, required=True)
    parser.add_argument("--risk-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    open_source = args.open_checkpoint.resolve(strict=True)
    risk_sources = [path.resolve(strict=True) for path in args.risk_checkpoint]
    report_source = args.policy_report.resolve(strict=True)
    preparation_source = args.preparation_manifest.resolve(strict=True)
    open_state = torch.load(open_source, map_location="cpu", weights_only=False)
    risk_states = [torch.load(path, map_location="cpu", weights_only=False) for path in risk_sources]
    expected_features = tuple(open_state["feature_names"])
    for risk_state in risk_states:
        if "feature_names" in risk_state:
            assert_feature_names(
                expected_features,
                tuple(risk_state["feature_names"]),
                artifact="risk checkpoint",
            )
        elif int(risk_state["model_state"]["lstm.weight_ih_l0"].shape[1]) != len(expected_features):
            raise ValueError("legacy risk checkpoint input width does not match open features")
    market_config = risk_states[0]["market_config"]
    open_threshold = float(risk_states[0]["open_threshold"])
    for state in risk_states[1:]:
        if state["market_config"] != market_config:
            raise ValueError("risk ensemble has inconsistent market configs")
        if float(state["open_threshold"]) != open_threshold:
            raise ValueError("risk ensemble has inconsistent open thresholds")
    report = json.loads(report_source.read_text(encoding="utf-8"))
    preparation_report = json.loads(preparation_source.read_text(encoding="utf-8"))
    if "frozen_policy" not in report or not report.get("score_history_tail"):
        raise ValueError("rerun causal-rate evaluation before freezing the policy")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    open_destination = output / "open_policy.pt"
    shutil.copy2(open_source, open_destination)
    risk_destinations = []
    for index, source in enumerate(risk_sources, start=1):
        destination = output / f"risk_direction_{index:02d}.pt"
        shutil.copy2(source, destination)
        risk_destinations.append(destination)
    bundle = {
        "status": "research_only_not_trading_ready",
        "open_checkpoint": open_destination.name,
        "risk_checkpoints": [path.name for path in risk_destinations],
        "open_sha256": file_sha256(open_destination),
        "risk_sha256s": [file_sha256(path) for path in risk_destinations],
        "feature_names": list(expected_features),
        "preparation": {
            "primary_vwap": preparation_report["primary_vwap"],
            "feature_profile": preparation_report.get("feature_profile", "raw"),
            "amplitude_threshold_ticks": preparation_report["amplitude_threshold_ticks"],
            "gate_fraction": preparation_report["gate_fraction"],
            "min_duration_seconds": preparation_report["min_duration_seconds"],
            "tick_size": 0.01,
        },
        "frozen_policy": report["frozen_policy"],
        "score_history_tail": report["score_history_tail"],
        "historical_validation": report.get("selected_on_validation"),
        "historical_fixed_test": report.get("fixed_test"),
    }
    policy_path = output / "policy.json"
    policy_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "open_sha256": bundle["open_sha256"],
                "risk_sha256s": bundle["risk_sha256s"],
                "risk_models": len(risk_destinations),
                "features": len(bundle["feature_names"]),
                "history_scores": len(bundle["score_history_tail"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
