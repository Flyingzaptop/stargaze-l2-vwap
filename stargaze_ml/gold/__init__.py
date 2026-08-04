"""Minute-candle XAUUSD forecasting and historical-analogue experiments."""

from .config import CTraderCredentials, GoldExperimentConfig
from .data import (
    REGIME_NAMES,
    CandleDataset,
    LineTargets,
    RegimeTargets,
    build_candle_dataset,
    build_line_targets,
    build_regime_targets,
)

__all__ = [
    "CTraderCredentials",
    "GoldExperimentConfig",
    "CandleDataset",
    "LineTargets",
    "RegimeTargets",
    "REGIME_NAMES",
    "build_candle_dataset",
    "build_line_targets",
    "build_regime_targets",
]
