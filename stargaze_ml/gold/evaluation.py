from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from .data import CandleDataset, LineTargets, REGIME_NAMES, RegimeTargets

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def line_metrics(
    prediction_bps: np.ndarray,
    centers: np.ndarray,
    candles: CandleDataset,
    targets: LineTargets,
    *,
    sigma_bps: np.ndarray | None = None,
    predicted_quality: np.ndarray | None = None,
    material_threshold_bps: np.ndarray | None = None,
) -> dict[str, Any]:
    prediction = np.asarray(prediction_bps, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.int64)
    truth = targets.line_end_bps[centers].astype(np.float64)
    actual_end = targets.actual_end_bps[centers].astype(np.float64)
    if prediction.shape != truth.shape:
        raise ValueError("line predictions do not align with targets")
    thresholds = (
        np.asarray(material_threshold_bps, dtype=np.float64)
        if material_threshold_bps is not None
        else np.maximum(np.nanmedian(np.abs(truth), axis=0) * 0.25, 1e-3)
    )
    per_horizon: dict[str, Any] = {}
    log_close = np.log(candles.close)
    for column, horizon in enumerate(targets.horizons_minutes):
        pred = prediction[:, column]
        target = truth[:, column]
        endpoint_error = np.abs(pred - target)
        actual_endpoint_error = np.abs(pred - actual_end[:, column])
        flat_endpoint = np.abs(target)
        path_error = np.zeros(len(centers), dtype=np.float64)
        flat_path_error = np.zeros(len(centers), dtype=np.float64)
        for tau in range(1, int(horizon) + 1):
            actual_path = (log_close[centers + tau] - log_close[centers]) * 10_000.0
            predicted_path = pred * tau / float(horizon)
            path_error += np.abs(predicted_path - actual_path)
            flat_path_error += np.abs(actual_path)
        path_error /= float(horizon)
        flat_path_error /= float(horizon)
        material = np.abs(target) >= thresholds[column]
        row: dict[str, Any] = {
            "line_endpoint_mae_bps": float(np.mean(endpoint_error)),
            "actual_endpoint_mae_bps": float(np.mean(actual_endpoint_error)),
            "flat_endpoint_mae_bps": float(np.mean(flat_endpoint)),
            "endpoint_improvement_vs_flat": float(1.0 - np.mean(endpoint_error) / max(np.mean(flat_endpoint), 1e-9)),
            "path_mae_bps": float(np.mean(path_error)),
            "flat_path_mae_bps": float(np.mean(flat_path_error)),
            "path_improvement_vs_flat": float(1.0 - np.mean(path_error) / max(np.mean(flat_path_error), 1e-9)),
            "line_correlation": _safe_correlation(pred, target),
            "material_threshold_bps": float(thresholds[column]),
            "material_rows": int(material.sum()),
            "direction_accuracy_material": float(np.mean(np.signbit(pred[material]) == np.signbit(target[material]))) if np.any(material) else 0.0,
        }
        if sigma_bps is not None:
            sigma = np.maximum(np.asarray(sigma_bps)[:, column], 1e-6)
            row["mean_sigma_bps"] = float(np.mean(sigma))
            row["coverage_one_sigma"] = float(np.mean(endpoint_error <= sigma))
        if predicted_quality is not None:
            quality = np.asarray(predicted_quality)[:, column]
            row["quality_error_correlation"] = _safe_correlation(quality, -endpoint_error)
        per_horizon[str(horizon)] = row
    return {
        "rows": len(centers),
        "per_horizon": per_horizon,
        "mean_path_improvement_vs_flat": float(
            np.mean([value["path_improvement_vs_flat"] for value in per_horizon.values()])
        ),
        "mean_line_correlation": float(np.mean([value["line_correlation"] for value in per_horizon.values()])),
    }


def regime_metrics(
    prediction: np.ndarray,
    centers: np.ndarray,
    targets: RegimeTargets,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.int64)
    truth = targets.regime[centers]
    if prediction.shape != truth.shape:
        raise ValueError("regime predictions do not align with targets")
    per_horizon: dict[str, Any] = {}
    overall_confusion = np.zeros((3, 3), dtype=np.int64)
    for column, horizon in enumerate(targets.horizons_minutes):
        actual = truth[:, column]
        predicted = prediction[:, column]
        confusion = np.zeros((3, 3), dtype=np.int64)
        for true_class in range(3):
            for predicted_class in range(3):
                confusion[true_class, predicted_class] = int(
                    np.sum((actual == true_class) & (predicted == predicted_class))
                )
        overall_confusion += confusion
        recalls = [
            confusion[index, index] / max(confusion[index].sum(), 1)
            for index in range(3)
        ]
        counts = np.bincount(actual, minlength=3)
        majority_accuracy = float(counts.max() / max(counts.sum(), 1))
        per_horizon[str(horizon)] = {
            "accuracy": float(np.mean(predicted == actual)),
            "macro_recall": float(np.mean(recalls)),
            "majority_baseline_accuracy": majority_accuracy,
            "confusion": confusion.tolist(),
            "class_counts": {REGIME_NAMES[index]: int(counts[index]) for index in range(3)},
        }
    return {
        "rows": len(centers),
        "per_horizon": per_horizon,
        "mean_accuracy": float(np.mean([value["accuracy"] for value in per_horizon.values()])),
        "mean_macro_recall": float(np.mean([value["macro_recall"] for value in per_horizon.values()])),
        "overall_confusion": overall_confusion.tolist(),
    }


def persistence_line_prediction(
    centers: np.ndarray,
    regimes: RegimeTargets,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.int64)
    past_horizon = 59.0
    slope = regimes.past_line_end_bps[centers] / past_horizon
    horizons = np.asarray(regimes.horizons_minutes, dtype=np.float64)
    return slope[:, None] * horizons[None, :]


def cosine_knn(
    reference_embedding: np.ndarray,
    query_embedding: np.ndarray,
    *,
    k: int,
    device: str = "",
    query_chunk: int = 1_024,
    reference_chunk: int = 32_768,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_embedding, dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("embeddings must be aligned two-dimensional arrays")
    if len(reference) == 0 or len(query) == 0:
        raise ValueError("non-empty embeddings are required")
    k = min(max(1, int(k)), len(reference))
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ref = torch.from_numpy(reference).to(target_device)
    all_scores: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for query_start in range(0, len(query), int(query_chunk)):
        q = torch.from_numpy(query[query_start : query_start + int(query_chunk)]).to(target_device)
        best_scores = torch.full((len(q), k), -torch.inf, device=target_device)
        best_indices = torch.full((len(q), k), -1, dtype=torch.long, device=target_device)
        for reference_start in range(0, len(ref), int(reference_chunk)):
            scores = q @ ref[reference_start : reference_start + int(reference_chunk)].T
            indices = (
                torch.arange(reference_start, reference_start + scores.shape[1], device=target_device)
                .unsqueeze(0)
                .expand(len(q), -1)
            )
            merged_scores = torch.cat((best_scores, scores), dim=1)
            merged_indices = torch.cat((best_indices, indices), dim=1)
            best_scores, positions = torch.topk(merged_scores, k=k, dim=1)
            best_indices = torch.gather(merged_indices, 1, positions)
        all_scores.append(best_scores.cpu().numpy())
        all_indices.append(best_indices.cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_scores)


def retrieval_line_prediction(
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
    reference_line_bps: np.ndarray,
    reference_quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(neighbor_indices, dtype=np.int64)
    similarity = np.asarray(similarities, dtype=np.float64)
    weights = np.exp((similarity - similarity.max(axis=1, keepdims=True)) / 0.08)
    weights /= weights.sum(axis=1, keepdims=True)
    values = np.asarray(reference_line_bps)[indices]
    quality_values = np.asarray(reference_quality)[indices]
    mean = np.sum(values * weights[:, :, None], axis=1)
    variance = np.sum(np.square(values - mean[:, None, :]) * weights[:, :, None], axis=1)
    quality = np.sum(quality_values * weights[:, :, None], axis=1)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 0.0)).astype(np.float32), quality.astype(np.float32)


def retrieval_regime_prediction(
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
    reference_regime: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(neighbor_indices, dtype=np.int64)
    similarity = np.asarray(similarities, dtype=np.float64)
    weights = np.exp((similarity - similarity.max(axis=1, keepdims=True)) / 0.08)
    weights /= weights.sum(axis=1, keepdims=True)
    values = np.asarray(reference_regime, dtype=np.int64)[indices]
    horizons = values.shape[-1]
    probabilities = np.zeros((len(indices), horizons, 3), dtype=np.float64)
    for class_index in range(3):
        probabilities[:, :, class_index] = np.sum(
            (values == class_index) * weights[:, :, None],
            axis=1,
        )
    return probabilities.argmax(axis=-1), probabilities.astype(np.float32)


def write_line_example_plot(
    path: Path,
    *,
    centers: np.ndarray,
    prediction_bps: np.ndarray,
    predicted_quality: np.ndarray,
    candles: CandleDataset,
    targets: LineTargets,
    context_minutes: int,
    title: str,
    examples: int = 6,
) -> None:
    centers = np.asarray(centers, dtype=np.int64)
    prediction = np.asarray(prediction_bps, dtype=np.float64)
    quality = np.asarray(predicted_quality, dtype=np.float64)
    if len(centers) == 0:
        return
    positions = np.linspace(0, len(centers) - 1, min(int(examples), len(centers)), dtype=np.int64)
    columns = 2
    rows = int(np.ceil(len(positions) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4.2 * rows), squeeze=False)
    log_close = np.log(candles.close)
    for axis, position in zip(axes.flat, positions, strict=False):
        center = int(centers[position])
        horizon_index = int(np.argmax(quality[position]))
        horizon = int(targets.horizons_minutes[horizon_index])
        past_tau = np.arange(-int(context_minutes) + 1, 1)
        past_indices = center + past_tau
        past = (log_close[past_indices] - log_close[center]) * 10_000.0
        future_tau = np.arange(0, horizon + 1)
        future = (log_close[center + future_tau] - log_close[center]) * 10_000.0
        predicted = prediction[position, horizon_index] * future_tau / float(horizon)
        oracle = targets.line_end_bps[center, horizon_index] * future_tau / float(horizon)
        axis.plot(past_tau, past, color="#64748b", linewidth=1.2, label="observed context")
        axis.plot(future_tau, future, color="#0f172a", linewidth=1.5, label="actual future")
        axis.plot(future_tau, predicted, color="#dc2626", linewidth=2.0, label="model line")
        axis.plot(future_tau, oracle, color="#2563eb", linewidth=1.4, linestyle="--", label="fitted target")
        axis.axvline(0, color="#94a3b8", linewidth=1.0)
        axis.axhline(0, color="#cbd5e1", linewidth=0.8)
        timestamp = np.datetime_as_string(
            np.datetime64(int(candles.ts_ns[center]), "ns"),
            unit="m",
        )
        axis.set_title(f"{timestamp} UTC | H={horizon}m | Q={quality[position, horizon_index]:.2f}")
        axis.set_xlabel("minutes from forecast")
        axis.set_ylabel("relative close, bps")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(positions) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(title, y=0.995, fontsize=14)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
