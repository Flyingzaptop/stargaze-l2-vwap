from __future__ import annotations

import numpy as np

from stargaze_ml.gold.l2_dominance_override import (
    DominanceOverrideConfig,
    price_dominance_evidence,
)


def test_aligned_falling_vwaps_below_market_mean_price_dominance() -> None:
    horizons = (5, 10, 15, 30, 45, 60, 120)
    names = [f"mid_vwap_{h}s_minus_mid_ticks" for h in horizons]
    names += [f"mid_vwap_{h}s_slope_1s_ticks" for h in horizons]
    index = {name: i for i, name in enumerate(names)}
    row = np.asarray([200.0] * len(horizons) + [-5.0] * len(horizons))
    evidence = price_dominance_evidence(row, index, DominanceOverrideConfig())
    assert evidence["relation"] == 1
    assert evidence["price_dominant"] is True


def test_rising_vwap_rejects_price_dominance_when_price_is_below() -> None:
    horizons = (5, 10, 15, 30, 45, 60, 120)
    names = [f"mid_vwap_{h}s_minus_mid_ticks" for h in horizons]
    names += [f"mid_vwap_{h}s_slope_1s_ticks" for h in horizons]
    index = {name: i for i, name in enumerate(names)}
    row = np.asarray([200.0] * len(horizons) + [5.0] * len(horizons))
    evidence = price_dominance_evidence(row, index, DominanceOverrideConfig())
    assert evidence["price_dominant"] is False
