from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import json

import numpy as np
import torch

from ..artifacts import write_json
from .config import GoldExperimentConfig
from .data import (
    build_candle_dataset,
    build_line_targets,
    build_regime_targets,
    save_prepared_dataset,
)
from .evaluation import (
    cosine_knn,
    line_metrics,
    persistence_line_prediction,
    regime_metrics,
    retrieval_line_prediction,
    retrieval_regime_prediction,
    write_line_example_plot,
)
from .models import DirectLineForecaster, DirectRegimeForecaster, ModelShape, RetrievalForecaster
from .training import (
    FeatureNormalizer,
    GoldWindowDataset,
    chronological_gold_splits,
    eligible_centers,
    predict_model,
    train_forecaster,
)


def _subsample(mask: np.ndarray, stride: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    return indices[:: max(1, int(stride))]


def _save_predictions(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_gold_experiments(
    *,
    candles_path: Path,
    out_dir: Path,
    config: GoldExperimentConfig,
    device: str = "",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    progress = progress or (lambda payload: print(json.dumps(payload), flush=True))
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", config.to_dict())
    progress({"stage": "gold_prepare", "status": "candles"})
    candles = build_candle_dataset(Path(candles_path))
    progress({"stage": "gold_prepare", "status": "line_targets", "rows": len(candles.ts_ns)})
    lines = build_line_targets(candles.close, candles.segment_id, config.horizons_minutes)
    regimes = build_regime_targets(
        candles.close,
        candles.segment_id,
        lines,
        context_minutes=config.context_minutes,
    )
    save_prepared_dataset(
        destination / "prepared_gold_m1.npz",
        candles,
        lines,
        regimes,
        metadata={
            "source_path": str(Path(candles_path).expanduser().resolve()),
            "config": config.to_dict(),
            "rows": len(candles.ts_ns),
        },
    )
    eligible = eligible_centers(candles, lines, context_minutes=config.context_minutes)
    splits = chronological_gold_splits(candles.ts_ns, eligible, config)
    np.savez_compressed(
        destination / "splits.npz",
        train=splits.train,
        valid=splits.valid,
        holdout=splits.holdout,
        train_end_ns=np.asarray(splits.train_end_ns, dtype=np.int64),
        valid_end_ns=np.asarray(splits.valid_end_ns, dtype=np.int64),
        purge_ns=np.asarray(splits.purge_ns, dtype=np.int64),
    )
    normalizer_mask = candles.valid_feature & (candles.ts_ns < splits.train_end_ns - splits.purge_ns)
    normalizer = FeatureNormalizer.fit(candles.x, normalizer_mask)
    x = normalizer.transform(candles.x)
    train_centers = _subsample(splits.train, config.sample_stride)
    valid_centers = _subsample(splits.valid, max(1, config.sample_stride))
    holdout_centers = _subsample(splits.holdout, config.evaluation_stride)
    train_target_rows = lines.line_end_bps[train_centers]
    line_scale = np.maximum(np.quantile(np.abs(train_target_rows), 0.75, axis=0), 1.0).astype(np.float32)
    write_json(
        destination / "normalization.json",
        {
            "features": normalizer.to_dict(),
            "line_scale_bps": line_scale.tolist(),
            "feature_names": list(candles.feature_names),
        },
    )
    common = {
        "x": x,
        "context_minutes": config.context_minutes,
        "line_targets": lines,
        "regime_targets": regimes,
        "line_scale": line_scale,
    }
    train_dataset = GoldWindowDataset(centers=train_centers, **common)
    valid_dataset = GoldWindowDataset(centers=valid_centers, **common)
    holdout_dataset = GoldWindowDataset(centers=holdout_centers, **common)
    shape = ModelShape(
        input_size=x.shape[1],
        horizons=len(config.horizons_minutes),
        hidden_size=config.hidden_size,
        layers=config.tcn_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        embedding_size=config.embedding_size,
    )
    summary: dict[str, Any] = {
        "rows": len(candles.ts_ns),
        "eligible_rows": int(eligible.sum()),
        "train_examples": len(train_dataset),
        "valid_examples": len(valid_dataset),
        "holdout_examples": len(holdout_dataset),
        "horizons_minutes": list(config.horizons_minutes),
        "models": {},
    }

    persistence = persistence_line_prediction(holdout_centers, regimes)
    summary["baselines"] = {
        "flat": line_metrics(
            np.zeros_like(persistence),
            holdout_centers,
            candles,
            lines,
            material_threshold_bps=0.25 * line_scale,
        ),
        "past_slope_persistence": line_metrics(
            persistence,
            holdout_centers,
            candles,
            lines,
            material_threshold_bps=0.25 * line_scale,
        ),
    }

    direct_line = DirectLineForecaster(shape)
    direct_line_result = train_forecaster(
        direct_line,
        train_dataset,
        valid_dataset,
        task="line",
        retrieval=False,
        config=config,
        out_dir=destination / "direct_line",
        device=device,
        progress=progress,
    )
    direct_line_predictions = predict_model(
        direct_line,
        holdout_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    direct_mean = direct_line_predictions["mean"] * line_scale
    direct_sigma = direct_line_predictions["sigma"] * line_scale
    direct_quality = 1.0 / (1.0 + np.exp(-direct_line_predictions["quality_logit"]))
    direct_line_metrics = line_metrics(
        direct_mean,
        direct_line_predictions["center"],
        candles,
        lines,
        sigma_bps=direct_sigma,
        predicted_quality=direct_quality,
        material_threshold_bps=0.25 * line_scale,
    )
    _save_predictions(
        destination / "direct_line" / "holdout_predictions.npz",
        center=direct_line_predictions["center"],
        mean_bps=direct_mean,
        sigma_bps=direct_sigma,
        quality=direct_quality,
    )
    write_line_example_plot(
        destination / "direct_line" / "holdout_examples.png",
        centers=direct_line_predictions["center"],
        prediction_bps=direct_mean,
        predicted_quality=direct_quality,
        candles=candles,
        targets=lines,
        context_minutes=config.context_minutes,
        title="Direct TCN: predicted line versus realised XAUUSD path",
    )
    summary["models"]["direct_line"] = {
        "best_epoch": direct_line_result.best_epoch,
        "best_valid_loss": direct_line_result.best_valid_loss,
        **direct_line_metrics,
    }
    del direct_line
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    direct_regime = DirectRegimeForecaster(shape)
    direct_regime_result = train_forecaster(
        direct_regime,
        train_dataset,
        valid_dataset,
        task="regime",
        retrieval=False,
        config=config,
        out_dir=destination / "direct_regime",
        device=device,
        progress=progress,
    )
    direct_regime_predictions = predict_model(
        direct_regime,
        holdout_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    direct_regime_class = direct_regime_predictions["regime_logits"].argmax(axis=-1)
    direct_regime_metrics = regime_metrics(
        direct_regime_class,
        direct_regime_predictions["center"],
        regimes,
    )
    _save_predictions(
        destination / "direct_regime" / "holdout_predictions.npz",
        center=direct_regime_predictions["center"],
        logits=direct_regime_predictions["regime_logits"],
        regime=direct_regime_class,
    )
    summary["models"]["direct_regime"] = {
        "best_epoch": direct_regime_result.best_epoch,
        "best_valid_loss": direct_regime_result.best_valid_loss,
        **direct_regime_metrics,
    }
    del direct_regime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    retrieval_line = RetrievalForecaster(shape, task="line")
    retrieval_line_result = train_forecaster(
        retrieval_line,
        train_dataset,
        valid_dataset,
        task="line",
        retrieval=True,
        config=config,
        out_dir=destination / "retrieval_line",
        device=device,
        progress=progress,
    )
    line_reference = predict_model(
        retrieval_line,
        train_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    line_query = predict_model(
        retrieval_line,
        holdout_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    line_neighbors, line_similarities = cosine_knn(
        line_reference["embedding"],
        line_query["embedding"],
        k=config.retrieval_k,
        device=device,
    )
    reference_centers = line_reference["center"]
    retrieval_mean, retrieval_sigma, retrieval_quality = retrieval_line_prediction(
        line_neighbors,
        line_similarities,
        lines.line_end_bps[reference_centers],
        lines.quality[reference_centers],
    )
    retrieval_line_metrics = line_metrics(
        retrieval_mean,
        line_query["center"],
        candles,
        lines,
        sigma_bps=retrieval_sigma,
        predicted_quality=retrieval_quality,
        material_threshold_bps=0.25 * line_scale,
    )
    _save_predictions(
        destination / "retrieval_line" / "holdout_predictions.npz",
        center=line_query["center"],
        mean_bps=retrieval_mean,
        sigma_bps=retrieval_sigma,
        quality=retrieval_quality,
        neighbor_train_row=line_neighbors,
        neighbor_center=reference_centers[line_neighbors],
        neighbor_similarity=line_similarities,
    )
    write_line_example_plot(
        destination / "retrieval_line" / "holdout_examples.png",
        centers=line_query["center"],
        prediction_bps=retrieval_mean,
        predicted_quality=retrieval_quality,
        candles=candles,
        targets=lines,
        context_minutes=config.context_minutes,
        title="Historical retrieval: neighbour-weighted line versus realised XAUUSD path",
    )
    summary["models"]["retrieval_line"] = {
        "best_epoch": retrieval_line_result.best_epoch,
        "best_valid_loss": retrieval_line_result.best_valid_loss,
        "mean_nearest_similarity": float(np.mean(line_similarities[:, 0])),
        **retrieval_line_metrics,
    }
    del retrieval_line, line_reference, line_query
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    retrieval_regime = RetrievalForecaster(shape, task="regime")
    retrieval_regime_result = train_forecaster(
        retrieval_regime,
        train_dataset,
        valid_dataset,
        task="regime",
        retrieval=True,
        config=config,
        out_dir=destination / "retrieval_regime",
        device=device,
        progress=progress,
    )
    regime_reference = predict_model(
        retrieval_regime,
        train_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    regime_query = predict_model(
        retrieval_regime,
        holdout_dataset,
        batch_size=config.batch_size,
        device=device,
    )
    regime_neighbors, regime_similarities = cosine_knn(
        regime_reference["embedding"],
        regime_query["embedding"],
        k=config.retrieval_k,
        device=device,
    )
    regime_reference_centers = regime_reference["center"]
    retrieval_regime_class, retrieval_regime_probabilities = retrieval_regime_prediction(
        regime_neighbors,
        regime_similarities,
        regimes.regime[regime_reference_centers],
    )
    retrieval_regime_metrics = regime_metrics(
        retrieval_regime_class,
        regime_query["center"],
        regimes,
    )
    _save_predictions(
        destination / "retrieval_regime" / "holdout_predictions.npz",
        center=regime_query["center"],
        regime=retrieval_regime_class,
        probability=retrieval_regime_probabilities,
        neighbor_train_row=regime_neighbors,
        neighbor_center=regime_reference_centers[regime_neighbors],
        neighbor_similarity=regime_similarities,
    )
    summary["models"]["retrieval_regime"] = {
        "best_epoch": retrieval_regime_result.best_epoch,
        "best_valid_loss": retrieval_regime_result.best_valid_loss,
        "mean_nearest_similarity": float(np.mean(regime_similarities[:, 0])),
        **retrieval_regime_metrics,
    }
    summary["elapsed_seconds"] = perf_counter() - started
    write_json(destination / "summary.json", summary)
    progress({"stage": "gold_complete", "out_dir": str(destination), "elapsed_seconds": summary["elapsed_seconds"]})
    return summary
