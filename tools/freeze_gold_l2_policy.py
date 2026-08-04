from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from stargaze_ml.gold.l2_contracts import assert_feature_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a complete forward-evaluation policy bundle")
    parser.add_argument("--open-checkpoint", type=Path, required=True)
    parser.add_argument("--risk-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    open_source = args.open_checkpoint.resolve(strict=True)
    risk_source = args.risk_checkpoint.resolve(strict=True)
    report_source = args.policy_report.resolve(strict=True)
    open_state = torch.load(open_source, map_location="cpu", weights_only=False)
    risk_state = torch.load(risk_source, map_location="cpu", weights_only=False)
    expected_features = tuple(open_state["feature_names"])
    if "feature_names" in risk_state:
        assert_feature_names(
            expected_features,
            tuple(risk_state["feature_names"]),
            artifact="risk checkpoint",
        )
    elif int(risk_state["model_state"]["lstm.weight_ih_l0"].shape[1]) != len(expected_features):
        raise ValueError("legacy risk checkpoint input width does not match open features")
    report = json.loads(report_source.read_text(encoding="utf-8"))
    if "frozen_policy" not in report or not report.get("score_history_tail"):
        raise ValueError("rerun causal-rate evaluation before freezing the policy")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    open_destination = output / "open_policy.pt"
    risk_destination = output / "risk_direction.pt"
    shutil.copy2(open_source, open_destination)
    shutil.copy2(risk_source, risk_destination)
    bundle = {
        "status": "research_only_not_trading_ready",
        "open_checkpoint": open_destination.name,
        "risk_checkpoint": risk_destination.name,
        "open_sha256": sha256(open_destination),
        "risk_sha256": sha256(risk_destination),
        "feature_names": list(expected_features),
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
                "risk_sha256": bundle["risk_sha256"],
                "features": len(bundle["feature_names"]),
                "history_scores": len(bundle["score_history_tail"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
