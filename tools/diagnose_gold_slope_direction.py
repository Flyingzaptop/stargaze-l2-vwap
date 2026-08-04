from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stargaze_ml.gold.models import TCNEncoder


class DirectionWindows(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        x: np.ndarray,
        line_end_bps: np.ndarray,
        centers: np.ndarray,
        horizon_columns: np.ndarray,
        thresholds_bps: np.ndarray,
        *,
        context: int,
    ) -> None:
        self.x = x
        self.line = line_end_bps
        self.centers = centers
        self.columns = horizon_columns
        self.thresholds = thresholds_bps
        self.context = int(context)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center = int(self.centers[index])
        target = self.line[center, self.columns]
        return (
            torch.from_numpy(self.x[center - self.context + 1 : center + 1]),
            torch.from_numpy((target > 0.0).astype(np.float32)),
            torch.from_numpy((np.abs(target) >= self.thresholds).astype(np.float32)),
        )


class DirectionTCN(nn.Module):
    def __init__(self, input_size: int, horizons: int) -> None:
        super().__init__()
        self.encoder = TCNEncoder(input_size, hidden_size=96, layers=6, kernel_size=3, dropout=0.10)
        self.head = nn.Linear(96, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    material: list[np.ndarray] = []
    for x, target, mask in loader:
        logits = model(x.to(device, non_blocking=True))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target.to(device))
        losses.append(float(loss))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(target.numpy())
        material.append(mask.numpy())
    probability = np.concatenate(probabilities)
    target = np.concatenate(targets).astype(bool)
    is_material = np.concatenate(material).astype(bool)
    predicted = probability >= 0.5
    accuracy = np.mean(predicted == target, axis=0)
    material_accuracy = np.asarray(
        [
            np.mean(predicted[is_material[:, column], column] == target[is_material[:, column], column])
            for column in range(target.shape[1])
        ]
    )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy.tolist(),
        "material_accuracy": material_accuracy.tolist(),
        "probability_std": np.std(probability, axis=0).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(46947)
    np.random.seed(46947)
    torch.manual_seed(46947)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(46947)
    prepared = np.load(args.prepared)
    splits = np.load(args.splits)
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    feature_center = np.asarray(normalization["features"]["center"], dtype=np.float32)
    feature_scale = np.asarray(normalization["features"]["scale"], dtype=np.float32)
    line_scale = np.asarray(normalization["line_scale_bps"], dtype=np.float32)
    horizons = prepared["horizons_minutes"]
    selected_horizons = np.asarray((15, 30, 60))
    columns = np.asarray([int(np.flatnonzero(horizons == horizon)[0]) for horizon in selected_horizons])
    x = np.clip((prepared["x"] - feature_center) / feature_scale, -12.0, 12.0).astype(np.float32)
    line = prepared["line_end_bps"]
    train_centers = np.flatnonzero(splits["train"])[::10]
    valid_centers = np.flatnonzero(splits["valid"])[::10]
    thresholds = 0.25 * line_scale[columns]
    train = DirectionWindows(x, line, train_centers, columns, thresholds, context=60)
    valid = DirectionWindows(x, line, valid_centers, columns, thresholds, context=60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionTCN(x.shape[1], len(columns)).to(device)
    train_loader = DataLoader(
        train,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        rows = 0
        for batch_x, target, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = nn.functional.binary_cross_entropy_with_logits(model(batch_x), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(batch_x)
            rows += len(batch_x)
        metrics = evaluate(model, valid_loader, device)
        valid_loss = float(metrics["loss"])
        scheduler.step(valid_loss)
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": total / rows,
            "valid_loss": valid_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{name: value for name, value in metrics.items() if name != "loss"},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("direction diagnostic produced no checkpoint")
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, valid_loader, device)
    result = {
        "horizons_minutes": selected_horizons.tolist(),
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "validation": final_metrics,
        "history": history,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, **result}, args.out_dir / "best_model.pt")
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "complete", **{k: v for k, v in result.items() if k != "history"}}), flush=True)


if __name__ == "__main__":
    main()
