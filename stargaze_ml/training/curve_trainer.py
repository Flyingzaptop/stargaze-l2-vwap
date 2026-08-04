from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models import CurveModelConfig, FourCurveCausalTransformer
from .curve_data import CurveWindowDataset


@dataclass(frozen=True)
class CurveTrainResult:
    model: FourCurveCausalTransformer
    best_epoch: int
    best_valid_loss: float
    history: tuple[dict[str, float], ...]
    peak_positive_weights: tuple[float, ...]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(
    output: object,
    batch: dict[str, torch.Tensor],
    peak_positive_weights: torch.Tensor,
) -> torch.Tensor:
    target = batch.get("target_seq", batch["target"])
    valid = batch.get("valid_seq", batch["valid"]).float()
    curve_weight = batch.get("weight_seq", batch.get("weight", torch.ones_like(target))).to(target.dtype)
    if target.ndim == 3:
        width = target.shape[1]
        logits = output.logits[:, -width:]
        scores = output.scores[:, -width:]
    else:
        logits = output.logits[:, -1]
        scores = output.scores[:, -1]
    if curve_weight.ndim == 1:
        curve_weight = curve_weight[:, None]
    peak_target = (target >= 0.5).to(target.dtype)
    curve_bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    peak_bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        peak_target,
        reduction="none",
        pos_weight=peak_positive_weights,
    )
    regression = nn.functional.smooth_l1_loss(scores, target, reduction="none")
    task_priority = torch.as_tensor([1.0, 2.0, 1.0, 2.0], dtype=target.dtype, device=target.device)
    weighted_valid = valid * curve_weight * task_priority
    weighted_denominator = weighted_valid.sum().clamp_min(1.0)
    valid_denominator = valid.sum().clamp_min(1.0)
    soft_loss = (curve_bce * weighted_valid).sum() / weighted_denominator
    peak_valid = valid * task_priority
    peak_loss = (peak_bce * peak_valid).sum() / peak_valid.sum().clamp_min(1.0)
    regression_loss = (regression * weighted_valid).sum() / weighted_denominator

    ranking_terms: list[torch.Tensor] = []
    direction_terms: list[torch.Tensor] = []
    for column in range(4):
        usable = valid[..., column] > 0.5
        positive = usable & (target[..., column] >= 0.5)
        quiet = usable & (target[..., column] <= 0.02)
        if positive.any() and quiet.any():
            positive_logits = logits[..., column][positive]
            negative_logits = logits[..., column][quiet]
            hard_count = min(int(negative_logits.numel()), max(1, 4 * int(positive_logits.numel())))
            hard_negative = torch.topk(negative_logits, hard_count).values
            ranking_terms.append(
                nn.functional.softplus(0.5 + hard_negative.mean() - positive_logits.mean())
            )
        opposite = column + 2 if column < 2 else column - 2
        if positive.any():
            direction_terms.append(
                nn.functional.softplus(
                    0.25 - logits[..., column][positive] + logits[..., opposite][positive]
                ).mean()
            )
    zero = logits.sum() * 0.0
    ranking_loss = torch.stack(ranking_terms).mean() if ranking_terms else zero
    direction_loss = torch.stack(direction_terms).mean() if direction_terms else zero

    derivative_loss = zero
    if target.ndim == 3 and target.shape[1] > 1:
        pair_valid = valid[:, 1:] * valid[:, :-1]
        predicted_diff = scores[:, 1:] - scores[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        edge_weight = 1.0 + 5.0 * torch.abs(target_diff)
        derivative_loss = (
            nn.functional.smooth_l1_loss(predicted_diff, target_diff, reduction="none")
            * pair_valid
            * edge_weight
        ).sum() / (pair_valid * edge_weight).sum().clamp_min(1.0)

    quiet = (valid > 0.5) & (target <= 0.02)
    quiet_loss = (
        nn.functional.relu(scores[quiet] - 0.05).square().mean() if quiet.any() else zero
    )
    total = (
        0.15 * soft_loss
        + 0.45 * peak_loss
        + 0.10 * regression_loss
        + 0.15 * ranking_loss
        + 0.10 * direction_loss
        + 0.03 * derivative_loss
        + 0.02 * quiet_loss
    )
    future_edges = getattr(output, "future_edges", None)
    if future_edges is not None and "auxiliary_target" in batch:
        auxiliary_target = batch["auxiliary_target"]
        auxiliary_valid = batch["auxiliary_valid"].to(auxiliary_target.dtype)
        width = auxiliary_target.shape[1]
        auxiliary_prediction = future_edges[:, -width:]
        edge_weight = 1.0 + 2.0 * (auxiliary_target > 0.0).to(auxiliary_target.dtype)
        auxiliary_regression = (
            nn.functional.smooth_l1_loss(
                auxiliary_prediction, auxiliary_target, beta=0.25, reduction="none"
            )
            * auxiliary_valid
            * edge_weight
        ).sum() / (auxiliary_valid * edge_weight).sum().clamp_min(1.0)
        directional = (auxiliary_target.abs() >= 0.2) & (auxiliary_valid > 0.5)
        if directional.any():
            sign = torch.where(auxiliary_target[directional] > 0.0, 1.0, -1.0)
            auxiliary_direction = nn.functional.softplus(
                -sign * auxiliary_prediction[directional]
            ).mean()
        else:
            auxiliary_direction = zero
        total = total + 0.15 * auxiliary_regression + 0.05 * auxiliary_direction
    return total


def _peak_weights(
    dataset: CurveWindowDataset,
    *,
    forward_cap: float,
    backward_cap: float,
) -> torch.Tensor:
    target = dataset.targets.values[dataset.centers]
    valid = dataset.targets.valid[dataset.centers]
    positive = np.sum(valid & (target >= 0.5), axis=0, dtype=np.float64)
    negative = np.sum(valid & (target < 0.5), axis=0, dtype=np.float64)
    caps = np.asarray([backward_cap, forward_cap, backward_cap, forward_cap], dtype=np.float64)
    weights = np.clip(negative / np.maximum(positive, 1.0), 1.0, caps)
    return torch.as_tensor(weights, dtype=torch.float32)


def _load_initial_model(
    model: FourCurveCausalTransformer,
    model_config: CurveModelConfig,
    checkpoint: dict[str, object],
) -> None:
    initial_config = CurveModelConfig(**checkpoint["model_config"])
    comparable_initial = initial_config.to_dict()
    comparable_model = model_config.to_dict()
    for key in ("num_aux_horizons", "auxiliary_output_dim"):
        comparable_initial.pop(key, None)
        comparable_model.pop(key, None)
    shared_to_separate = (
        not initial_config.separate_task_towers and model_config.separate_task_towers
    )
    if shared_to_separate:
        comparable_initial["separate_task_towers"] = True
    if comparable_initial != comparable_model:
        raise ValueError("initial checkpoint model configuration does not match")

    source = checkpoint["model_state"]
    if not isinstance(source, dict):
        raise ValueError("initial checkpoint model_state must be a mapping")
    if not shared_to_separate:
        incompatible = model.load_state_dict(source, strict=False)
        allowed_missing = {"future_edge_head.weight", "future_edge_head.bias"}
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise ValueError("initial checkpoint has incompatible model parameters")
        return

    mapped: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        if name.startswith("temporal."):
            suffix = name.removeprefix("temporal.")
            for task in ("backward_task", "forward_task"):
                mapped[f"task_towers.{task}.{suffix}"] = value
        elif name.startswith("score_head."):
            suffix = name.removeprefix("score_head.")
            for task, rows in (("backward_task", (0, 2)), ("forward_task", (1, 3))):
                target = value
                if suffix in {"3.weight", "3.bias"}:
                    target = value[list(rows)].clone()
                mapped[f"task_heads.{task}.{suffix}"] = target
        else:
            mapped[name] = value
    incompatible = model.load_state_dict(mapped, strict=False)
    allowed_missing = {"future_edge_head.weight", "future_edge_head.bias"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise ValueError("shared checkpoint could not initialize separate task towers")


@torch.no_grad()
def _evaluate(
    model: FourCurveCausalTransformer,
    loader: DataLoader,
    device: torch.device,
    peak_positive_weights: torch.Tensor,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for batch in loader:
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model(moved["x"], moved["venue_x"], moved["venue_mask"])
        losses.append(float(_loss(output, moved, peak_positive_weights).cpu()))
        predictions.append(output.scores[:, -1].float().cpu().numpy())
        targets.append(batch["target"].numpy())
        centers.append(batch["center_idx"].numpy())
    return float(np.mean(losses)), np.concatenate(predictions), np.concatenate(targets), np.concatenate(centers)


def train_curve_model(
    train_dataset: CurveWindowDataset,
    valid_dataset: CurveWindowDataset,
    *,
    model_config: CurveModelConfig,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-2,
    seed: int = 4105,
    device: str | None = None,
    forward_peak_weight_cap: float = 8.0,
    backward_peak_weight_cap: float = 16.0,
    initial_checkpoint: Path | None = None,
) -> CurveTrainResult:
    _seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FourCurveCausalTransformer(model_config).to(target_device)
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        _load_initial_model(model, model_config, checkpoint)
    peak_positive_weights = _peak_weights(
        train_dataset,
        forward_cap=float(forward_peak_weight_cap),
        backward_cap=float(backward_peak_weight_cap),
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=target_device.type == "cuda")
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=target_device.type == "cuda")
    use_amp = target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    checkpoint_dir = out_dir / "epoch_checkpoints"
    validation_dir = out_dir / "epoch_validation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            moved = {key: value.to(target_device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(moved["x"], moved["venue_x"], moved["venue_mask"])
                loss = _loss(output, moved, peak_positive_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        valid_loss, predictions, expected, validation_centers = _evaluate(
            model, valid_loader, target_device, peak_positive_weights
        )
        mae = float(np.mean(np.abs(predictions - expected)))
        row = {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "valid_loss": valid_loss, "valid_mae": mae}
        history.append(row)
        print({"stage": "curve_train", **row}, flush=True)
        epoch_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        torch.save(
            {
                "model_config": model_config.to_dict(),
                "model_state": epoch_state,
                "epoch": epoch,
                "metrics": row,
                "output_semantics": "shape_ranked_score_v3_with_dense_multihorizon_auxiliary",
            },
            checkpoint_dir / f"epoch_{epoch:02d}.pt",
        )
        np.savez_compressed(
            validation_dir / f"epoch_{epoch:02d}.npz",
            centers=validation_centers,
            predictions=predictions.astype(np.float32),
            targets=expected.astype(np.float32),
        )
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = epoch_state
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is None:
        raise RuntimeError("curve training did not produce a checkpoint")
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    serialized_peak_weights = tuple(float(value) for value in peak_positive_weights.cpu().tolist())
    torch.save(
        {
            "model_config": model_config.to_dict(),
            "model_state": best_state,
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "output_semantics": "shape_ranked_score_v3_with_dense_multihorizon_auxiliary",
            "peak_positive_weights": serialized_peak_weights,
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint is not None else None,
        },
        out_dir / "best_four_curve.pt",
    )
    return CurveTrainResult(model, best_epoch, best_loss, tuple(history), serialized_peak_weights)


def predict_curve_model(model: FourCurveCausalTransformer, dataset: CurveWindowDataset, *, batch_size: int = 8, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=target_device.type == "cuda")
    predictions: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["x"].to(target_device, non_blocking=True),
                batch["venue_x"].to(target_device, non_blocking=True),
                batch["venue_mask"].to(target_device, non_blocking=True),
            )
            predictions.append(output.scores[:, -1].float().cpu().numpy())
            centers.append(batch["center_idx"].numpy())
    return np.concatenate(centers), np.concatenate(predictions)
