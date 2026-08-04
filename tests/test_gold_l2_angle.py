from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from stargaze_ml.gold.l2_angle import (
    angle_to_slope,
    build_angle_targets,
    build_l2_feature_matrix,
    reconstruct_l2_bars,
)
from stargaze_ml.gold.models import DirectAngleForecaster, ModelShape


def test_reconstruct_l2_bars_rejects_crossed_snapshot(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []

    def add_snapshot(ts_ns: int, bid: float, ask: float) -> None:
        for level in range(3):
            rows.append(
                {
                    "timestamp": ts_ns,
                    "quote_id": ts_ns + level,
                    "bid": bid - level * 0.01,
                    "ask": 0.0,
                    "size": 100 * (level + 1),
                    "type": "new",
                }
            )
            rows.append(
                {
                    "timestamp": ts_ns,
                    "quote_id": ts_ns + 100 + level,
                    "bid": 0.0,
                    "ask": ask + level * 0.01,
                    "size": 100 * (level + 1),
                    "type": "new",
                }
            )

    add_snapshot(2_000_000_000, 100.00, 100.02)
    add_snapshot(2_500_000_000, 100.03, 100.02)  # crossed: must be rejected
    add_snapshot(4_100_000_000, 100.01, 100.03)
    path = tmp_path / "raw.parquet"
    pl.DataFrame(rows).write_parquet(path)

    bars = reconstruct_l2_bars(path, timestamp_unit="ns")

    assert bars.height == 2
    assert bars["bar_start_ns"].to_list() == [2_000_000_000, 4_000_000_000]
    assert np.all(bars["best_ask"].to_numpy() > bars["best_bid"].to_numpy())
    assert np.allclose(bars["microprice"].to_numpy(), [100.01, 100.02])
    assert bars["segment_id"].to_list() == [0, 0]


def test_angle_target_recovers_anchored_linear_future() -> None:
    step_seconds = 2
    mid = 100.0 + np.arange(100, dtype=np.float64) * 0.01
    segment = np.zeros(len(mid), dtype=np.int32)
    targets = build_angle_targets(
        mid,
        segment,
        horizons_steps=(5,),
        step_seconds=step_seconds,
        tick_size=0.01,
        vol_window_steps=10,
    )
    index = 40
    # One tick per two seconds => 0.5 ticks/s. Past sigma is
    # sqrt(1 tick^2 / 2 seconds).
    expected_sigma = np.sqrt(0.5)
    expected_angle = np.arctan(0.5 * np.sqrt(10.0) / expected_sigma)
    assert targets.valid[index, 0]
    assert np.isclose(targets.slope_ticks_per_second[index, 0], 0.5)
    assert np.isclose(targets.angle_radians[index, 0], expected_angle)
    recovered = angle_to_slope(
        targets.angle_radians[index, 0],
        targets.past_sigma_ticks_sqrt_second[index],
        horizon_seconds=10,
    )
    assert np.isclose(recovered, 0.5)
    assert np.isclose(targets.path_rmse_ticks[index, 0], 0.0, atol=1e-6)


def test_feature_matrix_is_causal_at_cutoff() -> None:
    n = 100
    frame = pl.DataFrame(
        {
            "bar_start_ns": np.arange(n, dtype=np.int64) * 2_000_000_000,
            "segment_id": np.zeros(n, dtype=np.int32),
            "mid": 100.0 + np.arange(n) * 0.01,
            "microprice": 100.005 + np.arange(n) * 0.01,
            "spread_ticks": np.full(n, 2.0),
            "imbalance_top1": np.linspace(-0.5, 0.5, n),
            "imbalance_top3": np.linspace(-0.25, 0.25, n),
            "bid_size_top1": np.full(n, 100.0),
            "ask_size_top1": np.full(n, 100.0),
            "bid_depth3": np.full(n, 600.0),
            "ask_depth3": np.full(n, 600.0),
            "bid_width3_ticks": np.full(n, 2.0),
            "ask_width3_ticks": np.full(n, 2.0),
            "snapshot_count": np.full(n, 2),
            "new_quote_count": np.full(n, 12),
            "mid_range_ticks": np.full(n, 1.0),
            "mid_change_ticks": np.full(n, 1.0),
        }
    )
    baseline = build_l2_feature_matrix(frame)
    changed = frame.with_columns(
        pl.when(pl.int_range(pl.len()) > 70)
        .then(pl.col("microprice") + 10.0)
        .otherwise(pl.col("microprice"))
        .alias("microprice")
    )
    perturbed = build_l2_feature_matrix(changed)
    assert np.array_equal(baseline.x[:71], perturbed.x[:71])
    assert baseline.valid_feature[70]


def test_angle_forecaster_output_is_bounded() -> None:
    model = DirectAngleForecaster(
        ModelShape(input_size=8, horizons=5, hidden_size=16, layers=2),
        max_angle_radians=1.4,
    )
    output = model(torch.randn(4, 30, 8))["angle"]
    assert output.shape == (4, 5)
    assert torch.all(torch.abs(output) <= 1.4)
