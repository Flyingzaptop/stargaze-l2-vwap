from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

from catboost import CatBoostRegressor
import matplotlib
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
import torch
from torch import nn
import torch.nn.functional as F

from stargaze_ml.gold.l2_angle import angle_to_slope
from stargaze_ml.gold.models import DirectAngleForecaster, ModelShape

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 46947848
CONTEXT_STEPS = 30
STEP_SECONDS = 2
MATERIAL_ANGLE_DEGREES = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train full-L2 CatBoost and temporal TCN on XAUUSD angle targets.",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("runs/gold_l2_angle_v1/prepared_l2_angle.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gold_l2_angle_v1/models"))
    parser.add_argument("--cat-iterations", type=int, default=1600)
    parser.add_argument("--cat-depth", type=int, default=8)
    parser.add_argument("--tcn-epochs", type=int, default=30)
    parser.add_argument("--tcn-patience", type=int, default=6)
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


def robust_normalize(
    x: np.ndarray,
    train_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(x[train_centers], axis=0).astype(np.float32)
    q25, q75 = np.quantile(x[train_centers], (0.25, 0.75), axis=0)
    scale = np.maximum(q75 - q25, 1e-3).astype(np.float32)
    normalized = np.clip((x - center) / scale, -12.0, 12.0).astype(np.float32)
    return normalized, center, scale


def _windows(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    offsets = np.arange(-CONTEXT_STEPS + 1, 1, dtype=np.int64)
    return x[centers[:, None] + offsets[None, :]]


def _batches(
    centers: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> list[np.ndarray]:
    order = np.arange(len(centers))
    if shuffle:
        np.random.shuffle(order)
    return [
        centers[order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


@torch.no_grad()
def predict_tcn(
    model: DirectAngleForecaster,
    x: np.ndarray,
    centers: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for batch in _batches(centers, batch_size=batch_size, shuffle=False):
        batch_x = torch.from_numpy(_windows(x, batch)).to(device, non_blocking=True)
        output.append(model(batch_x)["angle"].float().cpu().numpy())
    return np.concatenate(output) if output else np.empty((0, 5), dtype=np.float32)


def _selection_mae_degrees(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.degrees(np.mean(np.abs(prediction - target))))


def train_tcn(
    x: np.ndarray,
    angle: np.ndarray,
    train_centers: np.ndarray,
    validation_centers: np.ndarray,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    out_dir: Path,
) -> tuple[DirectAngleForecaster, dict[str, Any]]:
    seed_everything()
    shape = ModelShape(
        input_size=x.shape[1],
        horizons=angle.shape[1],
        hidden_size=96,
        layers=6,
        kernel_size=3,
        dropout=0.15,
    )
    model = DirectAngleForecaster(shape).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in _batches(train_centers, batch_size=batch_size, shuffle=True):
            batch_x = torch.from_numpy(_windows(x, batch)).to(device, non_blocking=True)
            target = torch.from_numpy(angle[batch]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(batch_x)["angle"]
                regression = F.smooth_l1_loss(
                    prediction,
                    target,
                    beta=np.radians(5.0),
                )
                angular = (1.0 - torch.cos(prediction - target)).mean()
                material = torch.abs(target) >= np.radians(MATERIAL_ANGLE_DEGREES)
                direction = prediction.sum() * 0.0
                if bool(material.any()):
                    direction = F.softplus(
                        -torch.sign(target[material]) * prediction[material] / 0.12
                    ).mean()
                loss = regression + 0.10 * angular + 0.04 * direction
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(batch)
            seen += len(batch)

        validation_prediction = predict_tcn(
            model,
            x,
            validation_centers,
            batch_size=batch_size,
            device=device,
        )
        validation_mae = _selection_mae_degrees(
            validation_prediction,
            angle[validation_centers],
        )
        scheduler.step(validation_mae)
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "validation_angle_mae_degrees": validation_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps({"stage": "tcn_epoch", **row}), flush=True)
        if validation_mae < best_mae - 0.01:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break

    if best_state is None:
        raise RuntimeError("TCN produced no valid checkpoint")
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model_shape": asdict(shape),
            "best_epoch": best_epoch,
            "best_validation_angle_mae_degrees": best_mae,
            "context_steps": CONTEXT_STEPS,
            "step_seconds": STEP_SECONDS,
        },
        out_dir / "tcn_best.pt",
    )
    (out_dir / "tcn_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    return model, {
        "best_epoch": best_epoch,
        "best_validation_angle_mae_degrees": best_mae,
        "history": history,
    }


def angle_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    line_end_ticks: np.ndarray,
    actual_end_ticks: np.ndarray,
    horizons_seconds: np.ndarray,
) -> dict[str, Any]:
    per_horizon: dict[str, Any] = {}
    for column, horizon in enumerate(horizons_seconds):
        pred = prediction[:, column]
        truth = target[:, column]
        abs_error = np.abs(pred - truth)
        zero_error = np.abs(truth)
        material = np.abs(truth) >= np.radians(MATERIAL_ANGLE_DEGREES)
        pred_slope = angle_to_slope(
            pred,
            sigma,
            horizon_seconds=int(horizon),
        )
        pred_end = pred_slope * int(horizon)
        oracle_end = line_end_ticks[:, column]
        actual_end = actual_end_ticks[:, column]
        line_error = np.abs(pred_end - oracle_end)
        actual_error = np.abs(pred_end - actual_end)
        per_horizon[str(int(horizon))] = {
            "angle_mae_degrees": float(np.degrees(abs_error.mean())),
            "zero_angle_mae_degrees": float(np.degrees(zero_error.mean())),
            "angle_mae_improvement_vs_zero": float(
                1.0 - abs_error.mean() / max(zero_error.mean(), 1e-12)
            ),
            "angle_correlation": (
                float(np.corrcoef(pred, truth)[0, 1])
                if np.std(pred) > 1e-10
                else 0.0
            ),
            "direction_accuracy_material": (
                float(np.mean(np.signbit(pred[material]) == np.signbit(truth[material])))
                if np.any(material)
                else 0.0
            ),
            "material_fraction": float(material.mean()),
            "prediction_std_degrees": float(np.degrees(np.std(pred))),
            "target_std_degrees": float(np.degrees(np.std(truth))),
            "line_endpoint_mae_ticks": float(line_error.mean()),
            "flat_line_endpoint_mae_ticks": float(np.abs(oracle_end).mean()),
            "line_endpoint_improvement_vs_flat": float(
                1.0 - line_error.mean() / max(np.abs(oracle_end).mean(), 1e-12)
            ),
            "actual_endpoint_mae_ticks": float(actual_error.mean()),
            "flat_actual_endpoint_mae_ticks": float(np.abs(actual_end).mean()),
            "actual_endpoint_improvement_vs_flat": float(
                1.0 - actual_error.mean() / max(np.abs(actual_end).mean(), 1e-12)
            ),
        }
    return {
        "rows": int(len(prediction)),
        "mean_angle_mae_degrees": float(
            np.mean([row["angle_mae_degrees"] for row in per_horizon.values()])
        ),
        "mean_angle_improvement_vs_zero": float(
            np.mean([row["angle_mae_improvement_vs_zero"] for row in per_horizon.values()])
        ),
        "mean_direction_accuracy_material": float(
            np.mean([row["direction_accuracy_material"] for row in per_horizon.values()])
        ),
        "per_horizon": per_horizon,
    }


def day_block_bootstrap(
    prediction: np.ndarray,
    target: np.ndarray,
    ts_ns: np.ndarray,
    *,
    baseline_prediction: np.ndarray | None = None,
    samples: int = 20_000,
) -> dict[str, Any]:
    baseline = np.zeros_like(prediction) if baseline_prediction is None else baseline_prediction
    improvement = np.abs(baseline - target) - np.abs(prediction - target)
    days = ts_ns // 86_400_000_000_000
    unique_days = np.unique(days)
    day_effect = np.asarray([improvement[days == day].mean() for day in unique_days])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(day_effect, size=(int(samples), len(day_effect)), replace=True).mean(axis=1)
    return {
        "unit": "mean absolute angle error in radians; positive favors candidate",
        "days": int(len(unique_days)),
        "mean_effect_radians": float(day_effect.mean()),
        "mean_effect_degrees": float(np.degrees(day_effect.mean())),
        "ci95_degrees": [
            float(np.degrees(np.quantile(draws, 0.025))),
            float(np.degrees(np.quantile(draws, 0.975))),
        ],
        "probability_effect_positive": float(np.mean(draws > 0)),
    }


def prediction_plot(
    path: Path,
    centers: np.ndarray,
    prediction: np.ndarray,
    target_angle: np.ndarray,
    mid: np.ndarray,
    ts_ns: np.ndarray,
    sigma: np.ndarray,
    target_slope: np.ndarray,
    horizons_seconds: np.ndarray,
) -> None:
    column = len(horizons_seconds) - 1
    ordered = np.argsort(target_angle[:, column])
    positions = ordered[(np.linspace(0.03, 0.97, 6) * (len(ordered) - 1)).astype(np.int64)]
    horizon_seconds = int(horizons_seconds[column])
    horizon_steps = horizon_seconds // STEP_SECONDS
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), squeeze=False)
    for axis, position in zip(axes.flat, positions, strict=True):
        center = int(centers[position])
        past_tau = np.arange(-CONTEXT_STEPS + 1, 1) * STEP_SECONDS
        past = (mid[center - CONTEXT_STEPS + 1 : center + 1] - mid[center]) / 0.01
        future_tau = np.arange(horizon_steps + 1) * STEP_SECONDS
        future = (mid[center : center + horizon_steps + 1] - mid[center]) / 0.01
        oracle = target_slope[position, column] * future_tau
        model_slope = angle_to_slope(
            prediction[position, column],
            sigma[position],
            horizon_seconds=horizon_seconds,
        )
        model_line = model_slope * future_tau
        axis.plot(past_tau, past, color="#64748b", linewidth=1.2, label="past mid")
        axis.plot(future_tau, future, color="#111827", linewidth=1.5, label="actual future")
        axis.plot(future_tau, oracle, "--", color="#dc2626", linewidth=1.8, label="target ax")
        axis.plot(future_tau, model_line, color="#2563eb", linewidth=2.1, label="model ax")
        axis.axvline(0, color="#94a3b8", linewidth=0.8)
        axis.axhline(0, color="#cbd5e1", linewidth=0.8)
        stamp = np.datetime_as_string(np.datetime64(int(ts_ns[position]), "ns"), unit="s")
        axis.set_title(
            f"{stamp} UTC | target={np.degrees(target_angle[position, column]):+.1f} deg | "
            f"model={np.degrees(prediction[position, column]):+.1f} deg"
        )
        axis.set_xlabel("seconds from forecast")
        axis.set_ylabel("ticks relative to mid(t)")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle("XAUUSD L2 holdout: predicted and target anchored lines", fontsize=14)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.967), ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _fit_catboost(
    name: str,
    x: np.ndarray,
    angle: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    iterations: int,
    depth: int,
    out_dir: Path,
) -> tuple[CatBoostRegressor, dict[str, Any]]:
    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=0.04,
        l2_leaf_reg=10.0,
        random_seed=SEED,
        random_strength=0.5,
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        devices="0",
        od_type="Iter",
        early_stopping_rounds=150,
        use_best_model=True,
        verbose=100,
        allow_writing_files=False,
    )
    started = time.perf_counter()
    model.fit(x[train], angle[train], eval_set=(x[validation], angle[validation]))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / f"{name}.cbm")
    return model, {
        "best_iteration": int(model.get_best_iteration()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _fit_independent_catboost(
    name: str,
    x: np.ndarray,
    angle: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    horizons_seconds: np.ndarray,
    iterations: int,
    depth: int,
    out_dir: Path,
) -> tuple[list[CatBoostRegressor], list[dict[str, Any]]]:
    models: list[CatBoostRegressor] = []
    details: list[dict[str, Any]] = []
    for column, horizon in enumerate(horizons_seconds):
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=int(iterations),
            depth=int(depth),
            learning_rate=0.04,
            l2_leaf_reg=10.0,
            random_seed=SEED + int(horizon),
            random_strength=0.5,
            task_type="GPU" if torch.cuda.is_available() else "CPU",
            devices="0",
            od_type="Iter",
            early_stopping_rounds=150,
            use_best_model=True,
            verbose=False,
            allow_writing_files=False,
        )
        started = time.perf_counter()
        model.fit(
            x[train],
            angle[train, column],
            eval_set=(x[validation], angle[validation, column]),
        )
        model.save_model(out_dir / f"{name}_{int(horizon)}s.cbm")
        models.append(model)
        row = {
            "horizon_seconds": int(horizon),
            "best_iteration": int(model.get_best_iteration()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        details.append(row)
        print(json.dumps({"stage": f"{name}_horizon_complete", **row}), flush=True)
    return models, details


def _predict_independent(
    models: list[CatBoostRegressor],
    x: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [model.predict(x).astype(np.float32) for model in models]
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    seed_everything()
    destination = args.out_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data = np.load(args.prepared, allow_pickle=False)
    x_raw = data["x"].astype(np.float32)
    feature_names = [str(value) for value in data["feature_names"]]
    angle = data["angle_radians"].astype(np.float32)
    train = np.flatnonzero(data["train"])
    validation = np.flatnonzero(data["validation"])
    holdout = np.flatnonzero(data["holdout"])
    horizons_seconds = data["horizons_seconds"].astype(np.int32)
    x, feature_center, feature_scale = robust_normalize(x_raw, train)
    np.savez_compressed(
        destination / "normalization.npz",
        center=feature_center,
        scale=feature_scale,
        feature_names=np.asarray(feature_names),
    )
    print(
        json.dumps(
            {
                "stage": "loaded",
                "features": x.shape[1],
                "train": len(train),
                "validation": len(validation),
                "holdout": len(holdout),
            }
        ),
        flush=True,
    )

    ridge = Ridge(alpha=20.0)
    ridge.fit(x[train], angle[train])
    ridge_validation = ridge.predict(x[validation]).astype(np.float32)
    ridge_holdout = ridge.predict(x[holdout]).astype(np.float32)

    price_only_names = {
        name
        for name in feature_names
        if (
            name.startswith("mid_")
            or name.startswith("time_")
        )
        and "micro" not in name
    }
    price_columns = np.asarray(
        [index for index, name in enumerate(feature_names) if name in price_only_names],
        dtype=np.int64,
    )
    print(
        json.dumps(
            {"stage": "catboost_price_start", "feature_count": len(price_columns)}
        ),
        flush=True,
    )
    cat_price, cat_price_info = _fit_catboost(
        "catboost_price_only",
        x[:, price_columns],
        angle,
        train,
        validation,
        iterations=args.cat_iterations,
        depth=args.cat_depth,
        out_dir=destination,
    )
    print(json.dumps({"stage": "catboost_full_start"}), flush=True)
    cat_full, cat_full_info = _fit_catboost(
        "catboost_full_l2",
        x,
        angle,
        train,
        validation,
        iterations=args.cat_iterations,
        depth=args.cat_depth,
        out_dir=destination,
    )
    print(json.dumps({"stage": "catboost_independent_price_start"}), flush=True)
    cat_independent_price, cat_independent_price_info = _fit_independent_catboost(
        "catboost_independent_price",
        x[:, price_columns],
        angle,
        train,
        validation,
        horizons_seconds=horizons_seconds,
        iterations=args.cat_iterations,
        depth=args.cat_depth,
        out_dir=destination,
    )
    print(json.dumps({"stage": "catboost_independent_full_l2_start"}), flush=True)
    cat_independent_full, cat_independent_full_info = _fit_independent_catboost(
        "catboost_independent_full_l2",
        x,
        angle,
        train,
        validation,
        horizons_seconds=horizons_seconds,
        iterations=args.cat_iterations,
        depth=args.cat_depth,
        out_dir=destination,
    )

    cat_price_validation = cat_price.predict(x[validation][:, price_columns]).astype(np.float32)
    cat_price_holdout = cat_price.predict(x[holdout][:, price_columns]).astype(np.float32)
    cat_full_validation = cat_full.predict(x[validation]).astype(np.float32)
    cat_full_holdout = cat_full.predict(x[holdout]).astype(np.float32)
    cat_independent_price_validation = _predict_independent(
        cat_independent_price,
        x[validation][:, price_columns],
    )
    cat_independent_price_holdout = _predict_independent(
        cat_independent_price,
        x[holdout][:, price_columns],
    )
    cat_independent_full_validation = _predict_independent(
        cat_independent_full,
        x[validation],
    )
    cat_independent_full_holdout = _predict_independent(
        cat_independent_full,
        x[holdout],
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(json.dumps({"stage": "tcn_start", "device": str(device)}), flush=True)
    tcn, tcn_info = train_tcn(
        x,
        angle,
        train,
        validation,
        epochs=args.tcn_epochs,
        patience=args.tcn_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        out_dir=destination,
    )
    tcn_validation = predict_tcn(
        tcn,
        x,
        validation,
        batch_size=args.batch_size,
        device=device,
    )
    tcn_holdout = predict_tcn(
        tcn,
        x,
        holdout,
        batch_size=args.batch_size,
        device=device,
    )

    blends = np.linspace(0.0, 1.0, 21)
    horizon_blend_weights: list[float] = []
    horizon_blend_errors: dict[str, dict[str, float]] = {}
    ensemble_validation = np.zeros_like(tcn_validation)
    ensemble_holdout = np.zeros_like(tcn_holdout)
    for column, horizon in enumerate(horizons_seconds):
        errors = [
            float(
                np.degrees(
                    np.mean(
                        np.abs(
                            weight * cat_independent_full_validation[:, column]
                            + (1.0 - weight) * tcn_validation[:, column]
                            - angle[validation, column]
                        )
                    )
                )
            )
            for weight in blends
        ]
        best_index = int(np.argmin(errors))
        weight = float(blends[best_index])
        horizon_blend_weights.append(weight)
        horizon_blend_errors[str(int(horizon))] = {
            f"{grid_weight:.2f}": error
            for grid_weight, error in zip(blends, errors, strict=True)
        }
        ensemble_validation[:, column] = (
            weight * cat_independent_full_validation[:, column]
            + (1.0 - weight) * tcn_validation[:, column]
        )
        ensemble_holdout[:, column] = (
            weight * cat_independent_full_holdout[:, column]
            + (1.0 - weight) * tcn_holdout[:, column]
        )

    sigma_holdout = data["past_sigma_ticks_sqrt_second"][holdout]
    line_end_holdout = data["line_end_ticks"][holdout]
    actual_end_holdout = data["actual_end_ticks"][holdout]
    target_holdout = angle[holdout]
    predictions = {
        "ridge": ridge_holdout,
        "catboost_price_only": cat_price_holdout,
        "catboost_full_l2": cat_full_holdout,
        "catboost_independent_price": cat_independent_price_holdout,
        "catboost_independent_full_l2": cat_independent_full_holdout,
        "tcn_full_l2": tcn_holdout,
        "ensemble_independent_full_l2": ensemble_holdout,
    }
    holdout_metrics = {
        name: angle_metrics(
            prediction,
            target_holdout,
            sigma_holdout,
            line_end_holdout,
            actual_end_holdout,
            horizons_seconds,
        )
        for name, prediction in predictions.items()
    }
    validation_mae = {
        "ridge": _selection_mae_degrees(ridge_validation, angle[validation]),
        "catboost_price_only": _selection_mae_degrees(
            cat_price_validation,
            angle[validation],
        ),
        "catboost_full_l2": _selection_mae_degrees(cat_full_validation, angle[validation]),
        "catboost_independent_price": _selection_mae_degrees(
            cat_independent_price_validation,
            angle[validation],
        ),
        "catboost_independent_full_l2": _selection_mae_degrees(
            cat_independent_full_validation,
            angle[validation],
        ),
        "tcn_full_l2": _selection_mae_degrees(tcn_validation, angle[validation]),
        "ensemble_independent_full_l2": _selection_mae_degrees(
            ensemble_validation,
            angle[validation],
        ),
    }
    bootstrap = {
        name: day_block_bootstrap(
            prediction,
            target_holdout,
            data["ts_ns"][holdout],
        )
        for name, prediction in predictions.items()
    }
    bootstrap["independent_full_l2_vs_independent_price"] = day_block_bootstrap(
        cat_independent_full_holdout,
        target_holdout,
        data["ts_ns"][holdout],
        baseline_prediction=cat_independent_price_holdout,
    )

    best_name = min(validation_mae, key=validation_mae.get)
    best_prediction = predictions[best_name]
    prediction_plot(
        destination / "holdout_angle_examples.png",
        holdout,
        best_prediction,
        target_holdout,
        data["mid"],
        data["ts_ns"][holdout],
        sigma_holdout,
        data["slope_ticks_per_second"][holdout],
        horizons_seconds,
    )
    pl.DataFrame(
        {
            "ts_ns": data["ts_ns"][holdout],
            **{
                f"target_angle_deg_{int(horizon)}s": np.degrees(target_holdout[:, column])
                for column, horizon in enumerate(horizons_seconds)
            },
            **{
                f"{name}_angle_deg_{int(horizon)}s": np.degrees(prediction[:, column])
                for name, prediction in predictions.items()
                for column, horizon in enumerate(horizons_seconds)
            },
        }
    ).write_parquet(destination / "holdout_predictions.parquet", compression="zstd")

    importances = sorted(
        zip(feature_names, cat_full.get_feature_importance(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    summary = {
        "selection_rule": "lowest validation mean angle MAE across all horizons",
        "selected_model": best_name,
        "horizons_seconds": horizons_seconds.tolist(),
        "material_angle_degrees": MATERIAL_ANGLE_DEGREES,
        "validation_mean_angle_mae_degrees": validation_mae,
        "holdout": holdout_metrics,
        "bootstrap": bootstrap,
        "catboost": {
            "price_only": cat_price_info,
            "full_l2": cat_full_info,
            "independent_price": cat_independent_price_info,
            "independent_full_l2": cat_independent_full_info,
            "price_only_features": [feature_names[index] for index in price_columns],
            "full_l2_feature_importance": [
                {"feature": name, "importance": float(value)}
                for name, value in importances
            ],
        },
        "tcn": tcn_info,
        "ensemble": {
            "catboost_weights_by_horizon": {
                str(int(horizon)): weight
                for horizon, weight in zip(
                    horizons_seconds,
                    horizon_blend_weights,
                    strict=True,
                )
            },
            "tcn_weights_by_horizon": {
                str(int(horizon)): 1.0 - weight
                for horizon, weight in zip(
                    horizons_seconds,
                    horizon_blend_weights,
                    strict=True,
                )
            },
            "validation_grid_mae_degrees": horizon_blend_errors,
        },
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "selected_model": best_name,
                "validation_mae_degrees": validation_mae[best_name],
                "holdout": holdout_metrics[best_name],
                "independent_full_l2_vs_independent_price": bootstrap[
                    "independent_full_l2_vs_independent_price"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
