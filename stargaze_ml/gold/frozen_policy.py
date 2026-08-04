from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenPolicyBundle:
    directory: Path
    open_checkpoint: Path
    risk_checkpoint: Path
    policy_path: Path
    policy: dict[str, Any]


def load_frozen_policy_bundle(directory: Path) -> FrozenPolicyBundle:
    """Validate filenames, hashes and causal-controller state before loading PyTorch."""

    root = Path(directory).expanduser().resolve(strict=True)
    policy_path = (root / "policy.json").resolve(strict=True)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    required = {
        "open_checkpoint",
        "risk_checkpoint",
        "open_sha256",
        "risk_sha256",
        "feature_names",
        "preparation",
        "frozen_policy",
        "score_history_tail",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"frozen policy is missing fields: {', '.join(missing)}")

    def inside(filename: str) -> Path:
        candidate = (root / filename).resolve(strict=True)
        if candidate.parent != root:
            raise ValueError("frozen checkpoint must be directly inside the bundle")
        return candidate

    open_checkpoint = inside(str(policy["open_checkpoint"]))
    risk_checkpoint = inside(str(policy["risk_checkpoint"]))
    if file_sha256(open_checkpoint) != str(policy["open_sha256"]):
        raise ValueError("open checkpoint hash mismatch")
    if file_sha256(risk_checkpoint) != str(policy["risk_sha256"]):
        raise ValueError("risk checkpoint hash mismatch")
    names = tuple(str(value) for value in policy["feature_names"])
    if not names or len(set(names)) != len(names):
        raise ValueError("frozen feature names must be non-empty and unique")
    history = np.asarray(policy["score_history_tail"], dtype=np.float64)
    if history.size == 0 or not np.all(np.isfinite(history)):
        raise ValueError("frozen causal score history is empty or non-finite")
    controller = policy["frozen_policy"]
    controller_required = {
        "mode",
        "penalty",
        "filter_field",
        "fallback_cutoff",
        "expected_candidates_per_day",
        "target_trades_per_day",
        "history_size",
        "min_history",
    }
    controller_missing = sorted(controller_required - set(controller))
    if controller_missing:
        raise ValueError(f"frozen controller is missing fields: {', '.join(controller_missing)}")
    if float(controller["expected_candidates_per_day"]) <= 0:
        raise ValueError("expected candidate rate must be positive")
    if int(controller["target_trades_per_day"]) <= 0:
        raise ValueError("daily trade cap must be positive")
    preparation = policy["preparation"]
    preparation_required = {
        "primary_vwap",
        "feature_profile",
        "amplitude_threshold_ticks",
        "gate_fraction",
        "min_duration_seconds",
        "tick_size",
    }
    preparation_missing = sorted(preparation_required - set(preparation))
    if preparation_missing:
        raise ValueError(f"frozen preparation is missing fields: {', '.join(preparation_missing)}")
    return FrozenPolicyBundle(root, open_checkpoint, risk_checkpoint, policy_path, policy)
