from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import json
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import GoldExperimentConfig
from .data import CandleDataset, LineTargets, RegimeTargets


@dataclass(frozen=True)
class FeatureNormalizer:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, mask: np.ndarray) -> "FeatureNormalizer":
        rows = np.asarray(x, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        if len(rows) == 0:
            raise ValueError("normalizer fit mask is empty")
        center = np.nanmedian(rows, axis=0)
        q25, q75 = np.nanquantile(rows, [0.25, 0.75], axis=0)
        scale = np.maximum(q75 - q25, 1e-6)
        return cls(center.astype(np.float32), scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        result = (np.asarray(x, dtype=np.float32) - self.center) / self.scale
        return np.clip(result, -12.0, 12.0).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"center": self.center.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class GoldSplits:
    train: np.ndarray
    valid: np.ndarray
    holdout: np.ndarray
    train_end_ns: int
    valid_end_ns: int
    purge_ns: int


def chronological_gold_splits(
    ts_ns: np.ndarray,
    eligible: np.ndarray,
    config: GoldExperimentConfig,
) -> GoldSplits:
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    indices = np.flatnonzero(eligible)
    if len(indices) < 1_000:
        raise ValueError("at least 1,000 eligible windows are required")
    train_cut = indices[min(len(indices) - 2, max(1, int(len(indices) * config.train_fraction)))]
    valid_cut = indices[
        min(
            len(indices) - 1,
            max(2, int(len(indices) * (config.train_fraction + config.valid_fraction))),
        )
    ]
    train_end = int(ts_ns[train_cut])
    valid_end = int(ts_ns[valid_cut])
    purge_ns = int(config.purge_minutes * 60 * 1e9)
    train = eligible & (ts_ns < train_end - purge_ns)
    valid = eligible & (ts_ns >= train_end + purge_ns) & (ts_ns < valid_end - purge_ns)
    holdout = eligible & (ts_ns >= valid_end + purge_ns)
    if not np.any(train) or not np.any(valid) or not np.any(holdout):
        raise ValueError("purge leaves an empty gold split")
    return GoldSplits(train, valid, holdout, train_end, valid_end, purge_ns)


def eligible_centers(
    candles: CandleDataset,
    targets: LineTargets,
    *,
    context_minutes: int,
) -> np.ndarray:
    n = len(candles.ts_ns)
    context = int(context_minutes)
    invalid_feature = (~candles.valid_feature).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(invalid_feature)))
    feature_window_valid = np.zeros(n, dtype=bool)
    ends = np.arange(context - 1, n)
    starts = ends - context + 1
    feature_window_valid[ends] = (prefix[ends + 1] - prefix[starts]) == 0
    same_segment = np.zeros(n, dtype=bool)
    same_segment[ends] = candles.segment_id[ends] == candles.segment_id[starts]
    return feature_window_valid & same_segment & np.all(targets.valid, axis=1)


class GoldWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        x: np.ndarray,
        centers: np.ndarray,
        context_minutes: int,
        line_targets: LineTargets,
        regime_targets: RegimeTargets,
        line_scale: np.ndarray,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.centers = np.asarray(centers, dtype=np.int64)
        self.context = int(context_minutes)
        self.line_end = np.asarray(line_targets.line_end_bps, dtype=np.float32)
        self.quality = np.asarray(line_targets.quality, dtype=np.float32)
        self.regime = np.asarray(regime_targets.regime, dtype=np.int8)
        self.valid = np.asarray(line_targets.valid & regime_targets.valid, dtype=bool)
        self.line_scale = np.asarray(line_scale, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        center = int(self.centers[index])
        start = center - self.context + 1
        return {
            "x": torch.from_numpy(self.x[start : center + 1]),
            "line": torch.from_numpy(self.line_end[center] / self.line_scale),
            "quality": torch.from_numpy(self.quality[center]),
            "regime": torch.from_numpy(self.regime[center].astype(np.int64)),
            "valid": torch.from_numpy(self.valid[center]),
            "center": torch.tensor(center, dtype=torch.long),
        }


@dataclass
class TrainResult:
    best_epoch: int
    best_valid_loss: float
    history: list[dict[str, float]]
    checkpoint_path: Path


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def line_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = batch["valid"].to(output["mean"].dtype)
    target = batch["line"]
    sigma = output["sigma"].clamp(0.05, 20.0)
    error = output["mean"] - target
    nll = 0.5 * torch.square(error / sigma) + torch.log(sigma)
    nll = (nll * valid).sum() / valid.sum().clamp_min(1.0)
    quality = F.binary_cross_entropy_with_logits(
        output["quality_logit"],
        batch["quality"],
        reduction="none",
    )
    quality = (quality * valid).sum() / valid.sum().clamp_min(1.0)
    total = nll + 0.25 * quality
    return total, {"forecast_nll": nll, "quality_bce": quality}


def regime_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = output["regime_logits"]
    targets = batch["regime"]
    valid = batch["valid"].to(logits.dtype)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        weight=class_weights,
        reduction="none",
    ).reshape_as(valid)
    loss = (losses * valid).sum() / valid.sum().clamp_min(1.0)
    return loss, {"cross_entropy": loss}


def metric_loss(
    embedding: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    task: str,
) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    predicted = ((embedding @ embedding.T) + 1.0) * 0.5
    if task == "line":
        outcome = torch.cat((batch["line"] * batch["quality"], batch["quality"]), dim=-1)
        distance = torch.cdist(outcome, outcome) / np.sqrt(outcome.shape[-1])
        desired = torch.exp(-distance)
    else:
        regime = batch["regime"]
        valid = batch["valid"]
        matches = (regime[:, None, :] == regime[None, :, :]) & valid[:, None, :] & valid[None, :, :]
        counts = (valid[:, None, :] & valid[None, :, :]).sum(dim=-1).clamp_min(1)
        desired = matches.sum(dim=-1).to(predicted.dtype) / counts
    mask = ~torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    return F.mse_loss(predicted[mask], desired[mask])


def class_weights_from_dataset(dataset: GoldWindowDataset) -> torch.Tensor:
    regimes = dataset.regime[dataset.centers]
    valid = dataset.valid[dataset.centers]
    counts = np.bincount(regimes[valid], minlength=3).astype(np.float64)
    weights = np.ones(3, dtype=np.float32)
    positive = counts > 0
    if np.any(positive):
        reference = counts[positive].max()
        weights[positive] = np.minimum(6.0, np.sqrt(reference / counts[positive])).astype(np.float32)
    return torch.from_numpy(weights)


def train_forecaster(
    model: nn.Module,
    train_dataset: GoldWindowDataset,
    valid_dataset: GoldWindowDataset,
    *,
    task: str,
    retrieval: bool,
    config: GoldExperimentConfig,
    out_dir: Path,
    device: str = "",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    if task not in {"line", "regime"}:
        raise ValueError("task must be line or regime")
    progress = progress or (lambda _: None)
    seed_everything(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=target_device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=target_device.type == "cuda",
    )
    class_weights = class_weights_from_dataset(train_dataset).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    use_amp = target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_total = 0.0
        train_rows = 0
        for batch in train_loader:
            batch = {name: value.to(target_device, non_blocking=True) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=target_device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch["x"])
                if task == "line":
                    loss, _ = line_loss(output, batch)
                else:
                    loss, _ = regime_loss(output, batch, class_weights)
                if retrieval:
                    loss = loss + 0.30 * metric_loss(output["embedding"], batch, task=task)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite {task} training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            size = len(batch["x"])
            train_total += float(loss.detach()) * size
            train_rows += size

        valid_metrics = evaluate_loss(
            model,
            valid_loader,
            task=task,
            retrieval=retrieval,
            class_weights=class_weights,
            device=target_device,
        )
        valid_loss = valid_metrics["loss"]
        scheduler.step(valid_loss)
        row = {
            "epoch": float(epoch),
            "train_loss": train_total / max(train_rows, 1),
            "valid_loss": valid_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"valid_{name}": value for name, value in valid_metrics.items() if name != "loss"},
        }
        history.append(row)
        progress({"stage": "gold_train", "task": task, "retrieval": retrieval, **row})
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "task": task,
            "retrieval": retrieval,
            "model_shape": asdict(model.shape),
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "history": history,
        },
        checkpoint_path,
    )
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return TrainResult(best_epoch, best_loss, history, checkpoint_path)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    task: str,
    retrieval: bool,
    class_weights: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    proxy_total = 0.0
    metric_total = 0.0
    rows = 0
    for batch in loader:
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        output = model(batch["x"])
        if task == "line":
            proxy, _ = line_loss(output, batch)
        else:
            proxy, _ = regime_loss(output, batch, class_weights)
        metric = metric_loss(output["embedding"], batch, task=task) if retrieval else proxy * 0.0
        loss = proxy + 0.30 * metric
        size = len(batch["x"])
        total += float(loss) * size
        proxy_total += float(proxy) * size
        metric_total += float(metric) * size
        rows += size
    return {
        "loss": total / max(rows, 1),
        "proxy_loss": proxy_total / max(rows, 1),
        "metric_loss": metric_total / max(rows, 1),
    }


@torch.no_grad()
def predict_model(
    model: nn.Module,
    dataset: GoldWindowDataset,
    *,
    batch_size: int,
    device: str = "",
) -> dict[str, np.ndarray]:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    model.eval()
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    collected: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        output = model(batch["x"].to(target_device))
        for name, value in output.items():
            collected.setdefault(name, []).append(value.detach().cpu().numpy())
        collected.setdefault("center", []).append(batch["center"].numpy())
    return {name: np.concatenate(values) for name, values in collected.items()}
