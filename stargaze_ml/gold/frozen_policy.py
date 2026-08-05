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
    risk_checkpoints: tuple[Path, ...]
    policy_path: Path
    policy: dict[str, Any]


def load_frozen_policy_bundle(directory: Path) -> FrozenPolicyBundle:
    """Validate filenames, hashes and causal-controller state before loading PyTorch."""

    root = Path(directory).expanduser().resolve(strict=True)
    policy_path = (root / "policy.json").resolve(strict=True)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    required = {
        "open_checkpoint",
        "open_sha256",
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
    risk_names = policy.get("risk_checkpoints")
    risk_hashes = policy.get("risk_sha256s")
    if risk_names is None:
        if "risk_checkpoint" not in policy or "risk_sha256" not in policy:
            raise ValueError("frozen policy has no risk checkpoint contract")
        risk_names = [policy["risk_checkpoint"]]
        risk_hashes = [policy["risk_sha256"]]
    if not isinstance(risk_names, list) or not isinstance(risk_hashes, list):
        raise ValueError("risk checkpoint names and hashes must be lists")
    if not risk_names or len(risk_names) != len(risk_hashes):
        raise ValueError("risk checkpoint names/hashes are empty or misaligned")
    risk_checkpoints = tuple(inside(str(name)) for name in risk_names)
    if file_sha256(open_checkpoint) != str(policy["open_sha256"]):
        raise ValueError("open checkpoint hash mismatch")
    for path, expected_hash in zip(risk_checkpoints, risk_hashes, strict=True):
        if file_sha256(path) != str(expected_hash):
            raise ValueError(f"risk checkpoint hash mismatch: {path.name}")
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
    if "open_threshold" in policy and not 0.0 <= float(policy["open_threshold"]) <= 1.0:
        raise ValueError("frozen open threshold must be in [0, 1]")
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
    adaptive_target = preparation.get("adaptive_gate_target_per_active_day")
    adaptive_history = preparation.get("adaptive_gate_history_tail")
    if adaptive_target is not None:
        if int(adaptive_target) <= 0:
            raise ValueError("adaptive gate target must be positive")
        if not isinstance(adaptive_history, dict):
            raise ValueError("adaptive policy requires a frozen amplitude history")
        amplitudes = np.asarray(adaptive_history.get("amplitude_ticks", []), dtype=np.float64)
        timestamps = np.asarray(adaptive_history.get("end_ts_ns", []), dtype=np.int64)
        if amplitudes.ndim != 1 or timestamps.shape != amplitudes.shape or not len(amplitudes):
            raise ValueError("adaptive frozen amplitude history is empty or misaligned")
        if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0):
            raise ValueError("adaptive frozen amplitudes must be finite and non-negative")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError("adaptive frozen timestamps must be monotonic")
    elif adaptive_history is not None:
        raise ValueError("fixed-gate policy cannot include adaptive amplitude history")
    return FrozenPolicyBundle(root, open_checkpoint, risk_checkpoints, policy_path, policy)
