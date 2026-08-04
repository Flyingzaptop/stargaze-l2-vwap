"""Symmetric PRICE-vs-VWAP dominance classifier at an accepted entry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from stargaze_ml.training.data import RobustNormalizer
from .l2_causal_rate import (
    CausalRateConfig, causal_rate_select, direction_and_score,
    robust_validation_score, summarize_selected,
)
from .l2_open_policy import L2OpenPolicy
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from .l2_risk_direction import L2RiskDirectionPolicy, RiskDirectionConfig, _trade_rows
from .l2_tabular_direction import _entry_samples


@dataclass(frozen=True)
class DominanceModelConfig:
    max_iter: int = 300
    learning_rate: float = 0.04
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    l2_regularization: float = 2.0
    seed: int = 20260815


def dominance_target(
    local_vwap_minus_price: np.ndarray,
    long_pnl: np.ndarray,
    short_pnl: np.ndarray,
) -> np.ndarray:
    mean_reversion_side = np.sign(local_vwap_minus_price).astype(np.int8)
    oracle_side = np.where(long_pnl >= short_pnl, 1, -1).astype(np.int8)
    return (oracle_side == -mean_reversion_side).astype(np.int64)


def apply_dominance_probabilities(
    rows: list[dict[str, float | int]],
    probabilities: np.ndarray,
    local_delta: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int]]:
    changed = []
    for row, probability, delta in zip(rows, probabilities, local_delta, strict=True):
        relation = int(np.sign(delta))
        side = -relation if relation != 0 and probability >= threshold else relation
        enriched = dict(row)
        if relation != 0:
            enriched["selected_side"] = side
        enriched["price_dominance_probability"] = float(probability)
        changed.append(enriched)
    return changed


def apply_price_dominance_veto(
    rows: list[dict[str, float | int]],
    probabilities: np.ndarray,
    local_delta: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int]]:
    """Keep the base side unless PRICE-dominance evidence is sufficiently high."""
    changed = []
    for row, probability, delta in zip(rows, probabilities, local_delta, strict=True):
        enriched = dict(row)
        relation = int(np.sign(delta))
        if relation != 0 and probability >= threshold:
            enriched["selected_side"] = -relation
        enriched["price_dominance_probability"] = float(probability)
        changed.append(enriched)
    return changed


def train_dominance_model(
    prepared_path: str | Path,
    open_checkpoint_path: str | Path,
    risk_checkpoint_path: str | Path,
    rate_report_path: str | Path,
    output_dir: str | Path,
    config: DominanceModelConfig,
    *,
    device_name: str = "auto",
) -> dict[str, object]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else device_name if device_name != "auto" else "cpu"
    )
    data = PreparedOpenData(prepared_path)
    open_state = torch.load(open_checkpoint_path, map_location=device, weights_only=False)
    risk_state = torch.load(risk_checkpoint_path, map_location=device, weights_only=False)
    rate_report = json.loads(Path(rate_report_path).read_text(encoding="utf-8"))
    market = OpenReinforceConfig(**risk_state["market_config"])
    risk_config = RiskDirectionConfig(**risk_state["config"])
    normalizer = RobustNormalizer.from_dict(risk_state["normalizer"])
    teacher = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    teacher.load_state_dict(open_state["model_state"]); teacher.eval()
    risk = L2RiskDirectionPolicy(len(data.feature_names), market.hidden_size).to(device)
    risk.load_state_dict(risk_state["model_state"]); risk.eval()
    open_threshold = float(risk_state["open_threshold"])
    local_index = data.feature_names.index("mid_vwap_60s_minus_mid_ticks")

    train_samples = _entry_samples(
        teacher, data, normalizer, _event_indices(data, 0, data.train_end, good_only=False),
        open_threshold, device, market,
    )
    train_delta = data.x[train_samples["entry_index"], local_index]
    target = dominance_target(train_delta, train_samples["long_pnl"], train_samples["short_pnl"])
    advantage = np.abs(train_samples["long_pnl"] - train_samples["short_pnl"])
    scale = max(float(np.median(advantage[advantage > 0])), 1.0)
    weights = np.clip(advantage / scale, 0.25, 10.0)
    counts = np.bincount(target, minlength=2).astype(np.float64)
    class_balance = len(target) / (2.0 * np.maximum(counts, 1.0))
    weights *= class_balance[target]
    model = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate, max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes, min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization, random_state=config.seed,
    ).fit(train_samples["x"], target, sample_weight=weights)

    policy = risk_state["evaluation"]["selected_on_validation"]
    mode = str(policy["mode"]); penalty = float(policy["penalty"])
    field = str(policy["filter_field"]); fallback = float(policy["cutoff"])
    target_rate = int(rate_report["selected_on_validation"]["target_trades_per_day"])
    val_rows = _trade_rows(risk, teacher, data, normalizer, _event_indices(data, data.train_end, data.validation_end, good_only=False), open_threshold, device, market, risk_config)
    test_rows = _trade_rows(risk, teacher, data, normalizer, _event_indices(data, data.validation_end, len(data.x), good_only=False), open_threshold, device, market, risk_config)
    day_ns = 86_400_000_000_000
    val_days = max(len({int(row["entry_ts_ns"]) // day_ns for row in val_rows}), 1)
    expected = len(val_rows) / val_days
    rate_config = CausalRateConfig(target_trades_per_day=target_rate)
    val_selected = causal_rate_select(val_rows, mode=mode, penalty=penalty, filter_field=field, expected_candidates_per_day=expected, fallback_cutoff=fallback, config=rate_config)
    initial_scores = [direction_and_score(row, mode=mode, penalty=penalty, filter_field=field)[1] for row in val_rows]
    test_selected = causal_rate_select(test_rows, mode=mode, penalty=penalty, filter_field=field, expected_candidates_per_day=expected, fallback_cutoff=fallback, config=rate_config, initial_scores=initial_scores)

    def predictions(rows: list[dict[str, float | int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        entries = np.asarray([int(row["entry_index"]) for row in rows], dtype=np.int64)
        probability = model.predict_proba(normalizer.transform(data.x[entries]))[:, 1]
        delta = data.x[entries, local_index]
        labels = dominance_target(
            delta,
            np.asarray([float(row["long_pnl"]) for row in rows]),
            np.asarray([float(row["short_pnl"]) for row in rows]),
        )
        return probability, delta, labels

    val_probability, val_delta, val_labels = predictions(val_selected)
    test_probability, test_delta, test_labels = predictions(test_selected)
    grid = []
    for threshold in np.linspace(0.1, 0.9, 33):
        changed = apply_dominance_probabilities(val_selected, val_probability, val_delta, float(threshold))
        metrics = summarize_selected(changed)
        grid.append({"threshold": float(threshold), "selection_score": robust_validation_score(metrics), **metrics})
    selected = max(grid, key=lambda row: float(row["selection_score"]))
    fixed_rows = apply_dominance_probabilities(
        test_selected, test_probability, test_delta, float(selected["threshold"])
    )
    report = {
        "device": str(device), "config": asdict(config),
        "train_entries": int(len(target)),
        "train_price_dominance_rate": float(target.mean()),
        "validation_auc": float(roc_auc_score(val_labels, val_probability)),
        "test_auc_diagnostic": float(roc_auc_score(test_labels, test_probability)),
        "selected_on_validation": selected,
        "fixed_test": summarize_selected(fixed_rows),
        "fixed_test_trades": fixed_rows,
    }
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "config": asdict(config), "normalizer": normalizer.to_dict(), "threshold": selected["threshold"]}, output / "final.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
