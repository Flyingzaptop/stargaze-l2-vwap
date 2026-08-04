"""Entry-snapshot gradient boosting baseline for L2/VWAP direction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from stargaze_ml.training.data import RobustNormalizer
from .l2_causal_rate import (
    CausalRateConfig,
    causal_rate_select,
    direction_and_score,
    robust_validation_score,
    summarize_selected,
)
from .l2_multivwap_side import _open_entries
from .l2_open_policy import L2OpenPolicy
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from .l2_profit_direction import executable_side_pnls


@dataclass(frozen=True)
class TabularDirectionConfig:
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    pnl_scale_ticks: float = 100.0
    tail_threshold_ticks: float = 500.0
    seed: int = 20260814


def _entry_samples(
    teacher: L2OpenPolicy,
    data: PreparedOpenData,
    normalizer: RobustNormalizer,
    events: np.ndarray,
    threshold: float,
    device: torch.device,
    market: OpenReinforceConfig,
) -> dict[str, np.ndarray]:
    entries = _open_entries(teacher, data, normalizer, events, threshold, device)
    event_ids = np.asarray(list(entries.keys()), dtype=np.int64)
    entry_ids = np.asarray(list(entries.values()), dtype=np.int64)
    if not len(entry_ids):
        raise ValueError("no open entries in split")
    long_pnl = np.empty(len(entry_ids), dtype=np.float64)
    short_pnl = np.empty(len(entry_ids), dtype=np.float64)
    for index, (event, entry) in enumerate(zip(event_ids, entry_ids, strict=True)):
        crossing = int(data.event_crossing_1[event])
        lp, sp = executable_side_pnls(data, int(entry), crossing, market)
        long_pnl[index] = float(lp[0]); short_pnl[index] = float(sp[0])
    return {
        "event_index": event_ids,
        "entry_index": entry_ids,
        "entry_ts_ns": data.ts_ns[entry_ids].astype(np.int64),
        "x": normalizer.transform(data.x[entry_ids]),
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
    }


def _classifier(config: TabularDirectionConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=config.learning_rate, max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes, min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization, random_state=config.seed,
    )


def _regressor(config: TabularDirectionConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=config.learning_rate,
        max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization, random_state=config.seed,
    )


def _rows(models: dict[str, object], samples: dict[str, np.ndarray], config: TabularDirectionConfig) -> list[dict[str, float | int]]:
    x = samples["x"]
    side_probability = models["side"].predict_proba(x)[:, 1]
    long_value = np.sinh(models["long_value"].predict(x)) * config.pnl_scale_ticks
    short_value = np.sinh(models["short_value"].predict(x)) * config.pnl_scale_ticks
    long_tail = models["long_tail"].predict_proba(x)[:, 1]
    short_tail = models["short_tail"].predict_proba(x)[:, 1]
    opportunity = models["opportunity"].predict_proba(x)[:, 1]
    rows = []
    for index in range(len(x)):
        rows.append({
            "event_index": int(samples["event_index"][index]),
            "entry_index": int(samples["entry_index"][index]),
            "entry_ts_ns": int(samples["entry_ts_ns"][index]),
            "long_pnl": float(samples["long_pnl"][index]),
            "short_pnl": float(samples["short_pnl"][index]),
            "side_probability": float(side_probability[index]),
            "predicted_long_pnl": float(long_value[index]),
            "predicted_short_pnl": float(short_value[index]),
            "long_tail_probability": float(long_tail[index]),
            "short_tail_probability": float(short_tail[index]),
            "opportunity_probability": float(opportunity[index]),
        })
    return rows


def train_tabular_direction(
    prepared_path: str | Path,
    open_checkpoint_path: str | Path,
    output_dir: str | Path,
    config: TabularDirectionConfig,
    *,
    device_name: str = "auto",
) -> dict[str, object]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else device_name if device_name != "auto" else "cpu"
    )
    data = PreparedOpenData(prepared_path)
    checkpoint = torch.load(Path(open_checkpoint_path).resolve(strict=True), map_location=device, weights_only=False)
    market = OpenReinforceConfig(**checkpoint["config"])
    normalizer = RobustNormalizer.from_dict(checkpoint["normalizer"])
    teacher = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    teacher.load_state_dict(checkpoint["model_state"]); teacher.eval()
    threshold = float(checkpoint["validation"]["best"]["threshold"])

    split_events = {
        "train": _event_indices(data, 0, data.train_end, good_only=False),
        "validation": _event_indices(data, data.train_end, data.validation_end, good_only=False),
        "test": _event_indices(data, data.validation_end, len(data.x), good_only=False),
    }
    samples = {
        name: _entry_samples(teacher, data, normalizer, events, threshold, device, market)
        for name, events in split_events.items()
    }
    train = samples["train"]
    long_pnl = train["long_pnl"]; short_pnl = train["short_pnl"]
    side = (long_pnl >= short_pnl).astype(np.int64)
    advantage = np.abs(long_pnl - short_pnl)
    scale = max(float(np.median(advantage[advantage > 0])), 1.0)
    side_weight = np.clip(advantage / scale, 0.25, 10.0)
    long_tail = (long_pnl <= -config.tail_threshold_ticks).astype(np.int64)
    short_tail = (short_pnl <= -config.tail_threshold_ticks).astype(np.int64)
    opportunity = (np.maximum(long_pnl, short_pnl) > 0).astype(np.int64)
    tail_weight_long = 1.0 + long_tail * np.clip((-long_pnl - config.tail_threshold_ticks) / config.tail_threshold_ticks, 0.0, 5.0)
    tail_weight_short = 1.0 + short_tail * np.clip((-short_pnl - config.tail_threshold_ticks) / config.tail_threshold_ticks, 0.0, 5.0)

    models: dict[str, object] = {
        "side": _classifier(config).fit(train["x"], side, sample_weight=side_weight),
        "long_value": _regressor(config).fit(train["x"], np.arcsinh(long_pnl / config.pnl_scale_ticks)),
        "short_value": _regressor(config).fit(train["x"], np.arcsinh(short_pnl / config.pnl_scale_ticks)),
        "long_tail": _classifier(config).fit(train["x"], long_tail, sample_weight=tail_weight_long),
        "short_tail": _classifier(config).fit(train["x"], short_tail, sample_weight=tail_weight_short),
        "opportunity": _classifier(config).fit(train["x"], opportunity),
    }
    val_rows = _rows(models, samples["validation"], config)
    test_rows = _rows(models, samples["test"], config)
    day_ns = 86_400_000_000_000
    val_days = max(len({int(row["entry_ts_ns"]) // day_ns for row in val_rows}), 1)
    expected = len(val_rows) / val_days
    grid = []
    for mode in ("classifier", "value", "risk"):
        for penalty in ((0.0,) if mode != "risk" else (300.0, 600.0, 1000.0)):
            for field in ("opportunity_probability", "negative_tail_probability", "risk_edge"):
                scores = [direction_and_score(row, mode=mode, penalty=penalty, filter_field=field)[1] for row in val_rows]
                for target in (10, 15, 20, 25):
                    q = float(np.clip(1.0 - target / expected, 0.0, 1.0))
                    fallback = float(np.quantile(scores, q))
                    chosen = causal_rate_select(
                        val_rows, mode=mode, penalty=penalty, filter_field=field,
                        expected_candidates_per_day=expected, fallback_cutoff=fallback,
                        config=CausalRateConfig(target_trades_per_day=target),
                    )
                    metrics = summarize_selected(chosen)
                    robust = robust_validation_score(metrics)
                    grid.append({
                        "mode": mode, "penalty": penalty, "filter_field": field,
                        "fallback_cutoff": fallback, "target_trades_per_day": target,
                        "selection_score": robust, **metrics,
                    })
    selected = max(grid, key=lambda row: float(row["selection_score"]))
    best_by_filter = {
        field: max(
            (row for row in grid if row["filter_field"] == field),
            key=lambda row: float(row["selection_score"]),
        )
        for field in sorted({str(row["filter_field"]) for row in grid})
    }
    initial_scores = [
        direction_and_score(row, mode=str(selected["mode"]), penalty=float(selected["penalty"]), filter_field=str(selected["filter_field"]))[1]
        for row in val_rows
    ]
    fixed = causal_rate_select(
        test_rows, mode=str(selected["mode"]), penalty=float(selected["penalty"]),
        filter_field=str(selected["filter_field"]), expected_candidates_per_day=expected,
        fallback_cutoff=float(selected["fallback_cutoff"]),
        config=CausalRateConfig(target_trades_per_day=int(selected["target_trades_per_day"])),
        initial_scores=initial_scores,
    )
    report = {
        "device": str(device), "config": asdict(config),
        "feature_count": len(data.feature_names),
        "entry_counts": {name: int(len(value["x"])) for name, value in samples.items()},
        "expected_candidates_per_day": expected,
        "selected_on_validation": selected,
        "validation_best_by_filter": best_by_filter,
        "fixed_test": summarize_selected(fixed),
    }
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "config": asdict(config), "normalizer": normalizer.to_dict(), "open_threshold": threshold}, output / "final.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
