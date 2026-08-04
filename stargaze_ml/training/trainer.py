from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models import HierarchicalCausalTransformerPolicy, PolicyConfig
from .data import PolicyWindowDataset, RobustNormalizer


@dataclass
class TrainResult:
    model: HierarchicalCausalTransformerPolicy
    best_epoch: int
    best_valid_loss: float
    best_selection_score: float
    history: list[dict[str, float]]
    checkpoint_path: Path | None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(output: Any, batch: dict[str, torch.Tensor], action_class_weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = output.action_logits[:, -1]
    weight = batch["weight"]
    action = F.cross_entropy(logits, batch["action"], weight=action_class_weights, reduction="none")
    action_loss = (action * weight).sum() / weight.sum().clamp_min(1e-6)
    def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        error = F.smooth_l1_loss(prediction, target, reduction="none")
        weights = mask.to(error.dtype)
        return (error * weights).sum() / weights.sum().clamp_min(1.0)

    forward_long = masked_smooth_l1(output.forward_long[:, -1], batch["forward_long"], batch["forward_valid"])
    forward_short = masked_smooth_l1(output.forward_short[:, -1], batch["forward_short"], batch["forward_valid"])
    horizon = F.cross_entropy(output.horizon_logits[:, -1], batch["horizon"])
    future_flow = masked_smooth_l1(output.future_flow[:, -1], batch["future_flow"], batch["future_valid"])
    future_liquidity = masked_smooth_l1(output.future_liquidity[:, -1], batch["future_liquidity"], batch["future_valid"])
    total = action_loss + 0.20 * (forward_long + forward_short) + 0.10 * horizon + 0.05 * (future_flow + future_liquidity)
    return total, {
        "action": action_loss,
        "forward_long": forward_long,
        "forward_short": forward_short,
        "horizon": horizon,
        "future_flow": future_flow,
        "future_liquidity": future_liquidity,
    }


def _assert_finite_batch(batch: dict[str, torch.Tensor]) -> None:
    for name in ("x", "venue_x", "forward_long", "forward_short", "future_flow", "future_liquidity", "weight"):
        value = batch[name]
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            bad = int((~torch.isfinite(value)).sum())
            raise ValueError(f"non-finite training tensor {name}: {bad} cells")


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, action_class_weights: torch.Tensor) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "action": 0.0, "forward_long": 0.0, "forward_short": 0.0, "horizon": 0.0, "future_flow": 0.0, "future_liquidity": 0.0, "correct": 0.0, "rows": 0.0}
    class_total = torch.zeros(7, dtype=torch.long)
    class_correct = torch.zeros(7, dtype=torch.long)
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        _assert_finite_batch(batch)
        output = model(batch["x"], batch["position_state"], batch["venue_x"], batch["venue_mask"])
        loss, parts = _loss(output, batch, action_class_weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite validation loss: { {name: float(value) for name, value in parts.items()} }")
        size = int(batch["action"].shape[0])
        totals["loss"] += float(loss) * size
        for name, value in parts.items():
            totals[name] += float(value) * size
        totals["correct"] += float((output.actions[:, -1] == batch["action"]).sum())
        prediction = output.actions[:, -1]
        for action_id in range(7):
            target_mask = batch["action"] == action_id
            class_total[action_id] += int(target_mask.sum())
            class_correct[action_id] += int(((prediction == action_id) & target_mask).sum())
        totals["rows"] += size
    rows = max(totals["rows"], 1.0)
    recalls = [float(class_correct[idx]) / int(class_total[idx]) for idx in range(7) if int(class_total[idx]) > 0]
    result = {name: value / rows for name, value in totals.items() if name not in {"rows", "correct"}}
    result.update({"action_accuracy": totals["correct"] / rows, "macro_action_recall": float(np.mean(recalls)) if recalls else 0.0, "rows": totals["rows"]})
    for idx in range(7):
        if int(class_total[idx]) > 0:
            result[f"recall_action_{idx}"] = float(class_correct[idx]) / int(class_total[idx])
    return result


def _class_weights(dataset: PolicyWindowDataset) -> torch.Tensor:
    if hasattr(dataset, "examples"):
        actions = np.asarray(dataset.examples.action, dtype=np.int64)
    else:
        actions = np.asarray([int(dataset[index]["action"]) for index in range(len(dataset))], dtype=np.int64)
    counts = np.bincount(actions, minlength=7).astype(np.float64)
    weights = np.ones(7, dtype=np.float32)
    for group in ((0, 1, 2), (3, 4), (5, 6)):
        present = [idx for idx in group if counts[idx] > 0]
        if not present:
            continue
        maximum = max(counts[idx] for idx in present)
        for idx in present:
            weights[idx] = float(min(8.0, np.sqrt(maximum / counts[idx])))
    return torch.from_numpy(weights)


def train_policy(
    train_dataset: PolicyWindowDataset,
    valid_dataset: PolicyWindowDataset,
    *,
    model_config: PolicyConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
    out_dir: Path | None = None,
    device: str | None = None,
) -> TrainResult:
    _seed_everything(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HierarchicalCausalTransformerPolicy(model_config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    action_class_weights = _class_weights(train_dataset).to(target_device)
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, num_workers=0, pin_memory=target_device.type == "cuda")
    valid_loader = DataLoader(valid_dataset, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=target_device.type == "cuda")
    use_amp = target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_selection = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(int(epochs)):
        model.train()
        train_loss = 0.0
        train_rows = 0
        for batch in train_loader:
            batch = {name: value.to(target_device, non_blocking=True) for name, value in batch.items()}
            _assert_finite_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=target_device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch["x"], batch["position_state"], batch["venue_x"], batch["venue_mask"])
                loss, _ = _loss(output, batch, action_class_weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            size = int(batch["action"].shape[0])
            train_loss += float(loss.detach()) * size
            train_rows += size
        metrics = _evaluate(model, valid_loader, target_device, action_class_weights)
        row = {"epoch": float(epoch + 1), "train_loss": train_loss / max(train_rows, 1), **{f"valid_{k}": float(v) for k, v in metrics.items()}}
        history.append(row)
        selection_score = float(metrics["action"] + (1.0 - metrics["macro_action_recall"]))
        row["selection_score"] = selection_score
        if selection_score < best_selection:
            best_selection = selection_score
            best_loss = float(metrics["loss"])
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path: Path | None = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / "best_policy.pt"
        torch.save({"model_state": best_state, "model_config": asdict(model_config), "best_epoch": best_epoch, "best_valid_loss": best_loss, "best_selection_score": best_selection, "action_class_weights": action_class_weights.cpu(), "history": history}, checkpoint_path)
        (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return TrainResult(model, best_epoch, best_loss, best_selection, history, checkpoint_path)
