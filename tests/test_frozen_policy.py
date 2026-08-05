from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stargaze_ml.gold.frozen_policy import load_frozen_policy_bundle, policy_vwap_horizons


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bundle(tmp_path: Path) -> Path:
    open_bytes = b"open"
    risk_bytes = b"risk"
    (tmp_path / "open.pt").write_bytes(open_bytes)
    (tmp_path / "risk.pt").write_bytes(risk_bytes)
    policy = {
        "open_checkpoint": "open.pt",
        "risk_checkpoints": ["risk.pt"],
        "open_sha256": _hash(open_bytes),
        "risk_sha256s": [_hash(risk_bytes)],
        "feature_names": ["bid_vwap_60s"],
        "preparation": {
            "primary_vwap": "60",
            "feature_profile": "raw",
            "amplitude_threshold_ticks": 300.0,
            "gate_fraction": 0.75,
            "min_duration_seconds": 30,
            "tick_size": 0.01,
        },
        "score_history_tail": [0.1, 0.2],
        "frozen_policy": {
            "mode": "risk",
            "penalty": 1000.0,
            "filter_field": "opportunity_probability",
            "fallback_cutoff": 0.5,
            "expected_candidates_per_day": 100.0,
            "target_trades_per_day": 20,
            "history_size": 2000,
            "min_history": 100,
        },
    }
    (tmp_path / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return tmp_path


def test_frozen_policy_verifies_hashes(tmp_path: Path) -> None:
    bundle = load_frozen_policy_bundle(_bundle(tmp_path))
    assert bundle.policy["feature_names"] == ["bid_vwap_60s"]
    assert policy_vwap_horizons(bundle.policy) == (60,)


def test_frozen_policy_prefers_explicit_vwap_horizons(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    policy_path = directory / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["preparation"]["vwap_horizons_seconds"] = [5, 60, 90, 600]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    bundle = load_frozen_policy_bundle(directory)
    assert policy_vwap_horizons(bundle.policy) == (5, 60, 90, 600)


def test_frozen_policy_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    (directory / "risk.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="risk checkpoint hash mismatch"):
        load_frozen_policy_bundle(directory)


def test_adaptive_frozen_policy_requires_causal_history(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    policy_path = directory / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["preparation"]["adaptive_gate_target_per_active_day"] = 400
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a frozen amplitude history"):
        load_frozen_policy_bundle(directory)


@pytest.mark.parametrize(
    "name",
    ["gold_l2_v1", "gold_l2_v2_ensemble", "gold_l2_v3_warm15_hybrid"],
)
def test_committed_frozen_bundle_hashes_are_valid(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = load_frozen_policy_bundle(root / "artifacts" / name)
    assert bundle.open_checkpoint.is_file()
    assert bundle.risk_checkpoints
