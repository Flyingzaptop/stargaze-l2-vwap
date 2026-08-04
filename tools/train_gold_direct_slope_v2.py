from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from stargaze_ml.gold.data import LineTargets, build_line_targets
from stargaze_ml.gold.models import DirectSlopeForecaster, ModelShape

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HORIZONS = (15, 30, 60)
CONTEXT = 60
SEED = 46947


@dataclass(frozen=True)
class Candidate:
    name: str
    loss: str
    transform: str = "identity"
    direction_weight: float = 0.0
    quality_weighted: bool = False
    hidden_size: int = 96
    layers: int = 6
    dropout: float = 0.10
    weight_decay: float = 1e-4


CANDIDATES = (
    Candidate("fixed_mse", "mse"),
    Candidate("fixed_huber", "huber"),
    Candidate(
        "asinh_huber_direction",
        "huber",
        transform="asinh",
        direction_weight=0.15,
    ),
    Candidate(
        "regularized_asinh_direction",
        "huber",
        transform="asinh",
        direction_weight=0.15,
        hidden_size=64,
        layers=5,
        dropout=0.25,
        weight_decay=1e-3,
    ),
    Candidate(
        "quality_regularized_asinh_direction",
        "huber",
        transform="asinh",
        direction_weight=0.15,
        quality_weighted=True,
        hidden_size=64,
        layers=5,
        dropout=0.25,
        weight_decay=1e-3,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and improve the direct anchored-slope XAUUSD TCN.",
    )
    parser.add_argument("--prepared", type=Path, default=Path("runs/gold_minute_v01/prepared_gold_m1.npz"))
    parser.add_argument("--splits", type=Path, default=Path("runs/gold_minute_v01/splits.npz"))
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("runs/gold_minute_v01/normalization.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gold_direct_line_v2"))
    parser.add_argument("--screen-train-stride", type=int, default=40)
    parser.add_argument("--screen-valid-stride", type=int, default=20)
    parser.add_argument("--screen-epochs", type=int, default=8)
    parser.add_argument("--screen-patience", type=int, default=3)
    parser.add_argument("--final-train-stride", type=int, default=10)
    parser.add_argument("--evaluation-stride", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=40)
    parser.add_argument("--final-patience", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _target_to_model(target: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return target
    if transform == "asinh":
        return torch.asinh(target)
    raise ValueError(f"unknown target transform: {transform}")


def _model_to_target(output: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return output
    if transform == "asinh":
        return torch.sinh(output.clamp(-5.0, 5.0))
    raise ValueError(f"unknown target transform: {transform}")


def slope_loss(
    raw_output: torch.Tensor,
    target: torch.Tensor,
    quality: torch.Tensor,
    candidate: Candidate,
) -> tuple[torch.Tensor, dict[str, float]]:
    transformed_target = _target_to_model(target, candidate.transform)
    if candidate.loss == "mse":
        pointwise = torch.square(raw_output - transformed_target)
    elif candidate.loss == "huber":
        pointwise = F.smooth_l1_loss(
            raw_output,
            transformed_target,
            reduction="none",
            beta=0.35,
        )
    else:
        raise ValueError(f"unknown loss: {candidate.loss}")
    if candidate.quality_weighted:
        weights = (0.25 + 0.75 * quality).to(pointwise.dtype)
        regression = (pointwise * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        regression = pointwise.mean()
    direction = raw_output.sum() * 0.0
    if candidate.direction_weight > 0.0:
        material = torch.abs(target) >= 0.50
        if bool(material.any()):
            direction = F.binary_cross_entropy_with_logits(
                raw_output[material] / 0.50,
                (target[material] > 0.0).to(raw_output.dtype),
            )
    total = regression + candidate.direction_weight * direction
    return total, {
        "regression": float(regression.detach()),
        "direction": float(direction.detach()),
    }


def _windows(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    offsets = np.arange(-CONTEXT + 1, 1, dtype=np.int64)
    return x[centers[:, None] + offsets[None, :]]


def _iter_batches(
    centers: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> list[np.ndarray]:
    order = np.arange(len(centers))
    if shuffle:
        np.random.shuffle(order)
    return [centers[order[start : start + batch_size]] for start in range(0, len(order), batch_size)]


@torch.no_grad()
def predict(
    model: nn.Module,
    *,
    x: np.ndarray,
    centers: np.ndarray,
    slope: np.ndarray,
    slope_scale: np.ndarray,
    candidate: Candidate,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for batch_centers in _iter_batches(centers, batch_size=batch_size, shuffle=False):
        batch_x = torch.from_numpy(_windows(x, batch_centers)).to(device, non_blocking=True)
        raw = model(batch_x)["slope"]
        normalized = _model_to_target(raw, candidate.transform)
        predictions.append((normalized * torch.from_numpy(slope_scale).to(device)).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty((0, slope.shape[1]), np.float32)


def metrics(
    prediction_slope: np.ndarray,
    centers: np.ndarray,
    targets: LineTargets,
    slope_scale: np.ndarray,
) -> dict[str, Any]:
    actual_slope = targets.slope_bps_per_minute[centers]
    quality = targets.quality[centers]
    per_horizon: dict[str, Any] = {}
    improvements: list[float] = []
    correlations: list[float] = []
    material_accuracies: list[float] = []
    for column, horizon in enumerate(HORIZONS):
        prediction_end = prediction_slope[:, column] * horizon
        actual_end = actual_slope[:, column] * horizon
        error = np.abs(prediction_end - actual_end)
        flat = np.abs(actual_end)
        improvement = 1.0 - float(error.mean()) / max(float(flat.mean()), 1e-12)
        correlation = (
            float(np.corrcoef(prediction_end, actual_end)[0, 1])
            if np.std(prediction_end) > 1e-10
            else 0.0
        )
        material = np.abs(actual_slope[:, column]) >= slope_scale[column]
        clear = quality[:, column] >= 0.60
        material_accuracy = float(
            np.mean(np.signbit(prediction_end[material]) == np.signbit(actual_end[material]))
        )
        clear_improvement = 1.0 - float(error[clear].mean()) / max(float(flat[clear].mean()), 1e-12)
        per_horizon[str(horizon)] = {
            "endpoint_mae_bps": float(error.mean()),
            "flat_endpoint_mae_bps": float(flat.mean()),
            "endpoint_improvement_vs_flat": improvement,
            "line_correlation": correlation,
            "prediction_std_bps_per_minute": float(np.std(prediction_slope[:, column])),
            "target_std_bps_per_minute": float(np.std(actual_slope[:, column])),
            "direction_accuracy_material": material_accuracy,
            "material_fraction": float(material.mean()),
            "clear_quality_fraction": float(clear.mean()),
            "clear_quality_improvement_vs_flat": clear_improvement,
            "clear_quality_direction_accuracy": float(
                np.mean(np.signbit(prediction_end[clear]) == np.signbit(actual_end[clear]))
            ),
        }
        improvements.append(improvement)
        correlations.append(correlation)
        material_accuracies.append(material_accuracy)
    return {
        "rows": int(len(centers)),
        "per_horizon": per_horizon,
        "mean_endpoint_improvement_vs_flat": float(np.mean(improvements)),
        "mean_line_correlation": float(np.mean(correlations)),
        "mean_direction_accuracy_material": float(np.mean(material_accuracies)),
    }


def train_candidate(
    candidate: Candidate,
    *,
    x: np.ndarray,
    targets: LineTargets,
    slope_scale: np.ndarray,
    train_centers: np.ndarray,
    valid_centers: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    out_dir: Path,
) -> tuple[DirectSlopeForecaster, dict[str, Any]]:
    seed_everything()
    shape = ModelShape(
        input_size=x.shape[1],
        horizons=len(HORIZONS),
        hidden_size=candidate.hidden_size,
        layers=candidate.layers,
        kernel_size=3,
        dropout=candidate.dropout,
    )
    model = DirectSlopeForecaster(shape).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=candidate.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    target_slope = targets.slope_bps_per_minute
    target_quality = targets.quality
    scale_device = torch.from_numpy(slope_scale).to(device)
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        rows = 0
        for batch_centers in _iter_batches(train_centers, batch_size=batch_size, shuffle=True):
            batch_x = torch.from_numpy(_windows(x, batch_centers)).to(device, non_blocking=True)
            target = torch.from_numpy(target_slope[batch_centers]).to(device) / scale_device
            quality = torch.from_numpy(target_quality[batch_centers]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(batch_x)["slope"]
                loss, _ = slope_loss(output, target, quality, candidate)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss for {candidate.name}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(batch_centers)
            rows += len(batch_centers)
        valid_prediction = predict(
            model,
            x=x,
            centers=valid_centers,
            slope=target_slope,
            slope_scale=slope_scale,
            candidate=candidate,
            batch_size=batch_size,
            device=device,
        )
        valid_metrics = metrics(valid_prediction, valid_centers, targets, slope_scale)
        normalized_maes = [
            valid_metrics["per_horizon"][str(horizon)]["endpoint_mae_bps"]
            / (slope_scale[column] * horizon)
            for column, horizon in enumerate(HORIZONS)
        ]
        selection_mae = float(np.mean(normalized_maes))
        scheduler.step(selection_mae)
        row = {
            "epoch": epoch,
            "train_loss": running / max(rows, 1),
            "selection_mae": selection_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mean_endpoint_improvement_vs_flat": valid_metrics["mean_endpoint_improvement_vs_flat"],
            "mean_line_correlation": valid_metrics["mean_line_correlation"],
            "mean_direction_accuracy_material": valid_metrics["mean_direction_accuracy_material"],
        }
        history.append(row)
        print(json.dumps({"stage": "epoch", "candidate": candidate.name, **row}), flush=True)
        if selection_mae < best_mae - 1e-5:
            best_mae = selection_mae
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{candidate.name} produced no checkpoint")
    model.load_state_dict(best_state)
    best_prediction = predict(
        model,
        x=x,
        centers=valid_centers,
        slope=target_slope,
        slope_scale=slope_scale,
        candidate=candidate,
        batch_size=batch_size,
        device=device,
    )
    best_metrics = metrics(best_prediction, valid_centers, targets, slope_scale)
    result = {
        "candidate": asdict(candidate),
        "best_epoch": best_epoch,
        "best_selection_mae": best_mae,
        "validation": best_metrics,
        "history": history,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model_shape": asdict(shape),
            "candidate": asdict(candidate),
            "slope_scale_bps_per_minute": slope_scale,
            "horizons_minutes": HORIZONS,
            "best_epoch": best_epoch,
        },
        out_dir / "best_model.pt",
    )
    (out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    return model, result


def write_prediction_plot(
    path: Path,
    *,
    centers: np.ndarray,
    prediction_slope: np.ndarray,
    close: np.ndarray,
    ts_ns: np.ndarray,
    targets: LineTargets,
) -> None:
    truth_60 = targets.slope_bps_per_minute[centers, 2]
    quantiles = (0.02, 0.15, 0.40, 0.60, 0.85, 0.98)
    ordered = np.argsort(truth_60)
    positions = [ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))] for q in quantiles]
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), squeeze=False)
    for axis, position in zip(axes.flat, positions, strict=True):
        center = int(centers[position])
        current = float(close[center])
        past_tau = np.arange(-59, 1)
        past_bps = (close[center + past_tau] / current - 1.0) * 10_000.0
        future_tau = np.arange(0, 61)
        future_bps = (close[center + future_tau] / current - 1.0) * 10_000.0
        axis.plot(past_tau, past_bps, color="#64748b", linewidth=1.1, label="past 60m")
        axis.plot(future_tau, future_bps, color="#111827", linewidth=1.5, label="actual future")
        colors = ("#f59e0b", "#dc2626", "#2563eb")
        for column, (horizon, color) in enumerate(zip(HORIZONS, colors, strict=True)):
            tau = np.arange(0, horizon + 1)
            target_line = targets.slope_bps_per_minute[center, column] * tau
            model_line = prediction_slope[position, column] * tau
            axis.plot(
                tau,
                target_line,
                color=color,
                linestyle="--",
                linewidth=1.7,
                label=f"target {horizon}m" if position == positions[0] else None,
            )
            axis.plot(
                tau,
                model_line,
                color=color,
                linewidth=2.0,
                alpha=0.55,
                label=f"model {horizon}m" if position == positions[0] else None,
            )
        timestamp = np.datetime_as_string(np.datetime64(int(ts_ns[center]), "ns"), unit="m")
        axis.set_title(
            f"{timestamp} UTC | target a60={truth_60[position]:+.3f} bps/min | "
            f"model={prediction_slope[position, 2]:+.3f}"
        )
        axis.axvline(0, color="#94a3b8", linewidth=0.8)
        axis.axhline(0, color="#cbd5e1", linewidth=0.8)
        axis.set_xlabel("minutes from forecast")
        axis.set_ylabel("price change from P(t), bps")
        axis.grid(alpha=0.20)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle("Direct slope TCN v2: exact anchored ax targets and model lines", fontsize=15)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything()
    destination = args.out_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prepared = np.load(args.prepared)
    splits = np.load(args.splits)
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    feature_center = np.asarray(normalization["features"]["center"], dtype=np.float32)
    feature_scale = np.asarray(normalization["features"]["scale"], dtype=np.float32)
    x = np.clip((prepared["x"] - feature_center) / feature_scale, -12.0, 12.0).astype(np.float32)
    targets = build_line_targets(
        prepared["close"],
        prepared["segment_id"],
        HORIZONS,
        price_change="simple",
    )
    train_all = np.flatnonzero(splits["train"])
    valid_all = np.flatnonzero(splits["valid"])
    holdout_all = np.flatnonzero(splits["holdout"])
    slope_scale = np.maximum(
        np.quantile(np.abs(targets.slope_bps_per_minute[train_all]), 0.75, axis=0),
        1e-4,
    ).astype(np.float32)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = {
        "horizons_minutes": HORIZONS,
        "context_minutes": CONTEXT,
        "target": "P(t+tau) ~= P(t) + a_price*tau; network predicts normalized a only",
        "target_price_change": "simple",
        "slope_scale_bps_per_minute": slope_scale.tolist(),
        "device": str(device),
        "args": vars(args) | {
            "prepared": str(args.prepared),
            "splits": str(args.splits),
            "normalization": str(args.normalization),
            "out_dir": str(args.out_dir),
        },
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "loaded",
                "device": str(device),
                "rows": len(x),
                "slope_scale_bps_per_minute": slope_scale.tolist(),
            }
        ),
        flush=True,
    )

    screen_train = train_all[:: max(1, args.screen_train_stride)]
    screen_valid = valid_all[:: max(1, args.screen_valid_stride)]
    screen_results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        _, result = train_candidate(
            candidate,
            x=x,
            targets=targets,
            slope_scale=slope_scale,
            train_centers=screen_train,
            valid_centers=screen_valid,
            epochs=args.screen_epochs,
            patience=args.screen_patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            out_dir=destination / "screen" / candidate.name,
        )
        screen_results.append(result)
        print(
            json.dumps(
                {
                    "stage": "screen_complete",
                    "candidate": candidate.name,
                    **{
                        key: result["validation"][key]
                        for key in (
                            "mean_endpoint_improvement_vs_flat",
                            "mean_line_correlation",
                            "mean_direction_accuracy_material",
                        )
                    },
                }
            ),
            flush=True,
        )
        del _
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ranked_screen = sorted(screen_results, key=lambda row: row["best_selection_mae"])
    finalist_names = [
        row["candidate"]["name"]
        for row in ranked_screen[: max(1, min(int(args.finalists), len(ranked_screen)))]
    ]
    (destination / "screen_summary.json").write_text(
        json.dumps(
            {
                "selection_rule": "lowest normalized validation endpoint MAE",
                "finalists": finalist_names,
                "candidates": screen_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"stage": "finalists_selected", "candidates": finalist_names}), flush=True)

    final_train = train_all[:: max(1, args.final_train_stride)]
    final_valid = valid_all[:: max(1, args.evaluation_stride)]
    finalist_results: list[dict[str, Any]] = []
    for finalist_name in finalist_names:
        finalist = next(candidate for candidate in CANDIDATES if candidate.name == finalist_name)
        finalist_model, finalist_result = train_candidate(
            finalist,
            x=x,
            targets=targets,
            slope_scale=slope_scale,
            train_centers=final_train,
            valid_centers=final_valid,
            epochs=args.final_epochs,
            patience=args.final_patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            out_dir=destination / "finalists" / finalist.name,
        )
        finalist_results.append(finalist_result)
        del finalist_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    final_result = min(finalist_results, key=lambda row: row["best_selection_mae"])
    selected_name = final_result["candidate"]["name"]
    selected = next(candidate for candidate in CANDIDATES if candidate.name == selected_name)
    selected_dir = destination / "finalists" / selected_name
    final_dir = destination / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_dir / "best_model.pt", final_dir / "best_model.pt")
    shutil.copy2(selected_dir / "training_history.json", final_dir / "training_history.json")
    selected_checkpoint = torch.load(
        selected_dir / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    final_model = DirectSlopeForecaster(ModelShape(**selected_checkpoint["model_shape"])).to(device)
    final_model.load_state_dict(selected_checkpoint["model_state"])
    print(
        json.dumps(
            {
                "stage": "final_selected",
                "candidate": selected_name,
                "validation_selection_mae": final_result["best_selection_mae"],
            }
        ),
        flush=True,
    )
    holdout_centers = holdout_all[:: max(1, args.evaluation_stride)]
    holdout_prediction = predict(
        final_model,
        x=x,
        centers=holdout_centers,
        slope=targets.slope_bps_per_minute,
        slope_scale=slope_scale,
        candidate=selected,
        batch_size=args.batch_size,
        device=device,
    )
    holdout_metrics = metrics(holdout_prediction, holdout_centers, targets, slope_scale)
    np.savez_compressed(
        final_dir / "holdout_predictions.npz",
        center=holdout_centers,
        slope_bps_per_minute=holdout_prediction,
        target_slope_bps_per_minute=targets.slope_bps_per_minute[holdout_centers],
        target_quality=targets.quality[holdout_centers],
    )
    write_prediction_plot(
        final_dir / "holdout_examples.png",
        centers=holdout_centers,
        prediction_slope=holdout_prediction,
        close=prepared["close"],
        ts_ns=prepared["ts_ns"],
        targets=targets,
    )
    summary = {
        "target_definition": {
            "equation": "P(t+tau) = P(t) + a_price * tau",
            "intercept": 0,
            "fit": "a_price = sum(tau * (P(t+tau)-P(t))) / sum(tau^2)",
            "network_output": "a in bps per minute, converted to a_price using current P(t)",
            "horizons_minutes": HORIZONS,
        },
        "selected_candidate": asdict(selected),
        "screen": screen_results,
        "finalists": finalist_results,
        "final_validation": final_result["validation"],
        "holdout": holdout_metrics,
        "best_epoch": final_result["best_epoch"],
        "slope_scale_bps_per_minute": slope_scale.tolist(),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "complete",
                "selected": selected_name,
                "best_epoch": final_result["best_epoch"],
                "validation": final_result["validation"],
                "holdout": holdout_metrics,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
