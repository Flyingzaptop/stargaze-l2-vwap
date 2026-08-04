"""Prediction averaging for independently seeded risk-direction policies."""

from __future__ import annotations

import numpy as np
import torch

from stargaze_ml.training.data import RobustNormalizer
from .l2_multivwap_side import _open_entries
from .l2_open_policy import L2OpenPolicy
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData
from .l2_profit_direction import executable_side_pnls
from .l2_risk_direction import L2RiskDirectionPolicy, RiskDirectionConfig


def average_risk_predictions(predictions: list[dict[str, float]]) -> dict[str, float]:
    if not predictions:
        raise ValueError("at least one prediction is required")
    keys = predictions[0].keys()
    return {key: float(np.mean([row[key] for row in predictions])) for key in keys}


def risk_prediction_disagreement(predictions: list[dict[str, float]]) -> dict[str, float]:
    if not predictions:
        raise ValueError("at least one prediction is required")
    keys = predictions[0].keys()
    return {
        f"{key}_std": float(np.std([row[key] for row in predictions]))
        for key in keys
    }


def ensemble_trade_rows(
    models: list[L2RiskDirectionPolicy],
    teacher: L2OpenPolicy,
    data: PreparedOpenData,
    normalizer: RobustNormalizer,
    events: np.ndarray,
    open_threshold: float,
    device: torch.device,
    market_config: OpenReinforceConfig,
    model_configs: list[RiskDirectionConfig],
) -> list[dict[str, float | int]]:
    if len(models) != len(model_configs) or not models:
        raise ValueError("models and configs must be non-empty and have equal length")
    entries = _open_entries(teacher, data, normalizer, events, open_threshold, device)
    for model in models:
        model.eval()
    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for event, entry in entries.items():
            start = int(data.event_start[event]); crossing = int(data.event_crossing_1[event])
            x = torch.from_numpy(normalizer.transform(data.x[start:crossing]))[None].to(device)
            offset = int(entry) - start
            predictions = []
            for model, config in zip(models, model_configs, strict=True):
                outputs = model(x)
                predictions.append({
                    "side_probability": float(torch.sigmoid(outputs[1][0, offset]).cpu()),
                    "predicted_long_pnl": float(np.sinh(float(outputs[2][0, offset].cpu())) * config.pnl_scale_ticks),
                    "predicted_short_pnl": float(np.sinh(float(outputs[3][0, offset].cpu())) * config.pnl_scale_ticks),
                    "long_tail_probability": float(torch.sigmoid(outputs[4][0, offset]).cpu()),
                    "short_tail_probability": float(torch.sigmoid(outputs[5][0, offset]).cpu()),
                    "opportunity_probability": float(torch.sigmoid(outputs[6][0, offset]).cpu()),
                })
            mean = average_risk_predictions(predictions)
            disagreement = risk_prediction_disagreement(predictions)
            long_pnl, short_pnl = executable_side_pnls(data, int(entry), crossing, market_config)
            rows.append({
                "event_index": int(event), "entry_index": int(entry),
                "entry_ts_ns": int(data.ts_ns[int(entry)]),
                "long_pnl": float(long_pnl[0]), "short_pnl": float(short_pnl[0]),
                **mean, **disagreement,
            })
    return rows
