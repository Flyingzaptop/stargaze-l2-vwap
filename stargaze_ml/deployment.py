from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import torch

from .contracts import CausalFrames, PositionSide
from .labels import CURVE_NAMES
from .models import CurveModelConfig, FourCurveCausalTransformer
from .training import (
    CurveInferenceDataset,
    RobustNormalizer,
    causal_centers,
    causal_score_features_for_normalizer,
    predict_curve_model,
)


@dataclass(frozen=True)
class FourCurveDecision:
    action: str
    scores: dict[str, float]


class FourCurveRuntime:
    """Target-free inference runtime for an exported four-curve checkpoint."""

    def __init__(
        self,
        model: FourCurveCausalTransformer,
        base_normalizer: RobustNormalizer,
        venue_normalizer: RobustNormalizer,
        *,
        context_ticks: int,
        horizons_seconds: tuple[float, ...] = (),
        fee_round_trip_bps: float = 0.0,
        calibration_reference: np.ndarray | None = None,
        curve_thresholds: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
        device: str | None = None,
    ) -> None:
        self.model = model
        self.base_normalizer = base_normalizer
        self.venue_normalizer = venue_normalizer
        self.context_ticks = int(context_ticks)
        self.horizons_seconds = tuple(float(value) for value in horizons_seconds)
        self.fee_round_trip_bps = float(fee_round_trip_bps)
        self.calibration_reference = (
            None if calibration_reference is None else np.asarray(calibration_reference, dtype=np.float32)
        )
        if self.calibration_reference is not None and (
            self.calibration_reference.ndim != 2 or self.calibration_reference.shape[1] != 4
        ):
            raise ValueError("calibration reference must have shape [N, 4]")
        self.curve_thresholds = tuple(float(value) for value in curve_thresholds)
        if len(self.curve_thresholds) != 4 or any(not 0.0 <= value <= 1.0 for value in self.curve_thresholds):
            raise ValueError("curve thresholds must contain four values in [0, 1]")
        self.device = device

    @classmethod
    def load(cls, run_dir: str | Path, *, device: str | None = None) -> FourCurveRuntime:
        root = Path(run_dir)
        policy_path = root / "economic_policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
        checkpoint_name = str(policy.get("checkpoint", "best_four_curve.pt"))
        checkpoint = torch.load(root / checkpoint_name, map_location="cpu", weights_only=True)
        model = FourCurveCausalTransformer(CurveModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        normalizers = json.loads((root / "normalizers.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "four_curve_run.json").read_text(encoding="utf-8"))
        args = manifest["args"]
        horizons = tuple(float(value) for value in str(args["horizons"]).split(",") if value.strip())
        calibration_path = root / "score_calibration.npz"
        calibration_reference = None
        if calibration_path.exists():
            with np.load(calibration_path, allow_pickle=False) as data:
                calibration_reference = np.asarray(data["sorted_reference"], dtype=np.float32)
        thresholds = policy.get("curve_thresholds", (0.5, 0.5, 0.5, 0.5))
        return cls(
            model,
            RobustNormalizer.from_dict(normalizers["base"]),
            RobustNormalizer.from_dict(normalizers["venue"]),
            context_ticks=int(args["context_ticks"]),
            horizons_seconds=horizons,
            fee_round_trip_bps=float(args["fee_round_trip_bps"]),
            calibration_reference=calibration_reference,
            curve_thresholds=tuple(float(value) for value in thresholds),
            device=device,
        )

    def score_frames(
        self,
        frames: CausalFrames,
        *,
        mask: np.ndarray | None = None,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        inference_mask = frames.valid if mask is None else np.asarray(mask, dtype=bool) & frames.valid
        centers = causal_centers(frames, inference_mask, context_ticks=self.context_ticks)
        causal_score_x = causal_score_features_for_normalizer(
            frames,
            self.base_normalizer,
            self.horizons_seconds,
            cost_bps=self.fee_round_trip_bps,
        )
        dataset = CurveInferenceDataset(
            frames,
            centers,
            context_ticks=self.context_ticks,
            base_normalizer=self.base_normalizer,
            venue_normalizer=self.venue_normalizer,
            causal_score_x=causal_score_x,
        )
        prediction_centers, scores = predict_curve_model(
            self.model,
            dataset,
            batch_size=int(batch_size),
            device=self.device,
        )
        if self.calibration_reference is not None:
            calibrated = np.empty_like(scores, dtype=np.float32)
            denominator = float(len(self.calibration_reference))
            for column in range(4):
                calibrated[:, column] = np.searchsorted(
                    self.calibration_reference[:, column], scores[:, column], side="right"
                ) / denominator
            scores = calibrated
        return prediction_centers, scores

    def score_latest(self, frames: CausalFrames) -> np.ndarray:
        mask = np.zeros(len(frames.ts_ns), dtype=bool)
        mask[-1] = True
        centers, scores = self.score_frames(frames, mask=mask, batch_size=1)
        if len(centers) != 1 or int(centers[0]) != len(frames.ts_ns) - 1:
            raise ValueError("latest frame is invalid, segmented, or lacks causal context")
        return scores[0]

    @staticmethod
    def decide(
        position: PositionSide | int,
        scores: np.ndarray,
        curve_thresholds: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
    ) -> FourCurveDecision:
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("scores must be four finite values in [0, 1]")
        long_backward, long_forward, short_backward, short_forward = (float(x) for x in values)
        thresholds = tuple(float(value) for value in curve_thresholds)
        if len(thresholds) != 4:
            raise ValueError("curve_thresholds must contain four values")
        state = PositionSide(int(position))
        if state is PositionSide.FLAT:
            long_excess = (
                (long_forward - thresholds[1]) / max(1.0 - thresholds[1], 1e-9)
                if long_backward <= thresholds[0]
                else float("-inf")
            )
            short_excess = (
                (short_forward - thresholds[3]) / max(1.0 - thresholds[3], 1e-9)
                if short_backward <= thresholds[2]
                else float("-inf")
            )
            if max(long_excess, short_excess) <= 0.0:
                action = "skip"
            elif long_excess >= short_excess:
                action = "open_long"
            else:
                action = "open_short"
        elif state is PositionSide.LONG:
            action = "close_long" if long_backward > thresholds[0] else "hold"
        else:
            action = "close_short" if short_backward > thresholds[2] else "hold"
        return FourCurveDecision(action, dict(zip(CURVE_NAMES, (float(x) for x in values), strict=True)))

    def decision(self, position: PositionSide | int, scores: np.ndarray) -> FourCurveDecision:
        return self.decide(position, scores, self.curve_thresholds)


__all__ = ["FourCurveDecision", "FourCurveRuntime"]


def export_four_curve_bundle(run_dir: str | Path, out_dir: str | Path) -> dict[str, object]:
    source = Path(run_dir)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    policy_path = source / "economic_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
    checkpoint_name = str(policy.get("checkpoint", "best_four_curve.pt"))
    files = ["best_four_curve.pt", "normalizers.json", "four_curve_run.json", "four_curve_summary.json"]
    for optional in (checkpoint_name, "economic_policy.json", "score_calibration.npz"):
        if (source / optional).exists() and optional not in files:
            files.append(optional)
    hashes: dict[str, str] = {}
    for name in files:
        target = destination / name
        shutil.copy2(source / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    summary = json.loads((source / "four_curve_summary.json").read_text(encoding="utf-8"))
    manifest = {
        "format": "stargaze_four_curve_bundle_v1",
        "curve_names": list(CURVE_NAMES),
        "source_run": str(source.resolve()),
        "status": "execution_ready" if bool(summary.get("execution_ready")) else "insufficient_data",
        "target_free_inference": True,
        "action_policy": {
            "checkpoint": checkpoint_name,
            "score_space": str(policy.get("score_space", "raw")),
            "curve_thresholds": policy.get("curve_thresholds", [0.5, 0.5, 0.5, 0.5]),
        },
        "sha256": hashes,
    }
    (destination / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


__all__.append("export_four_curve_bundle")
