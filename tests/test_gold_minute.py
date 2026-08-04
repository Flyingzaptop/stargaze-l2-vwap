from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

from stargaze_ml.gold.config import CTraderCredentials, GoldExperimentConfig
from stargaze_ml.gold.ctrader import choose_symbol, decode_trendbar, trendbars_to_frame
from stargaze_ml.gold.data import (
    CONTINUATION,
    FRICTION,
    REVERSAL,
    build_candle_dataset,
    build_line_targets,
    build_regime_targets,
)
from stargaze_ml.gold.models import (
    DirectLineForecaster,
    DirectRegimeForecaster,
    DirectSlopeForecaster,
    ModelShape,
    RetrievalForecaster,
)
from stargaze_ml.gold.training import chronological_gold_splits, eligible_centers


def _write_candles(path: Path, close: np.ndarray, *, gap_at: int | None = None) -> None:
    n = len(close)
    timestamps = np.arange(n, dtype=np.int64) * 60_000
    if gap_at is not None:
        timestamps[gap_at:] += 60_000
    body = np.sin(np.arange(n) / 11.0) * 0.02
    open_price = close - body
    high = np.maximum(open_price, close) + 0.05
    low = np.minimum(open_price, close) - 0.05
    frame = pl.DataFrame(
        {
            "timestamp": pl.from_epoch(pl.Series(timestamps), time_unit="ms").dt.replace_time_zone("UTC"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 + np.arange(n) % 17,
        }
    )
    frame.write_parquet(path)


def test_ctrader_trendbar_decoding_and_frame() -> None:
    bar = SimpleNamespace(
        low=450_000_000,
        deltaOpen=10_000,
        deltaHigh=40_000,
        deltaClose=30_000,
        utcTimestampInMinutes=123,
        volume=321,
    )
    decoded = decode_trendbar(bar, digits=2)
    assert decoded.low == 4500.0
    assert decoded.open == 4500.1
    assert decoded.high == 4500.4
    assert decoded.close == 4500.3
    assert decoded.timestamp_ms == 123 * 60_000
    frame = trendbars_to_frame([bar, bar], digits=2)
    assert frame.height == 1
    assert frame["volume"][0] == 321.0


def test_choose_symbol_normalises_broker_names() -> None:
    symbols = [
        SimpleNamespace(symbolName="EURUSD", symbolId=1),
        SimpleNamespace(symbolName="XAU/USD", symbolId=2),
    ]
    assert choose_symbol(symbols, "XAUUSD").symbolId == 2


def test_credentials_do_not_expose_secrets_in_repr() -> None:
    credentials = CTraderCredentials(
        client_id="client",
        client_secret="private-secret",
        access_token="private-access",
        refresh_token="private-refresh",
        account_id=123,
    )
    rendered = repr(credentials)
    assert "private-secret" not in rendered
    assert "private-access" not in rendered
    assert "private-refresh" not in rendered


def test_features_targets_and_segments(tmp_path: Path) -> None:
    n = 3_200
    close = 2_000.0 * np.exp(np.arange(n) * 0.00001)
    path = tmp_path / "gold.parquet"
    _write_candles(path, close, gap_at=2_500)
    candles = build_candle_dataset(path)
    assert candles.x.shape[0] == n
    assert candles.x.shape[1] == len(candles.feature_names)
    assert candles.valid_feature[1_500]
    assert candles.segment_id[2_499] + 1 == candles.segment_id[2_500]
    lines = build_line_targets(candles.close, candles.segment_id, (5, 30, 60))
    assert lines.valid[2_000].all()
    assert np.all(lines.line_end_bps[2_000] > 0.0)
    assert not lines.valid[2_480, -1]


def test_simple_price_line_target_is_exactly_anchored_ax() -> None:
    close = np.asarray([100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0])
    segments = np.zeros(len(close), dtype=np.int32)
    targets = build_line_targets(
        close,
        segments,
        (3,),
        past_vol_window=1,
        price_change="simple",
    )
    future_bps = (close[1:4] / close[0] - 1.0) * 10_000.0
    tau = np.arange(1.0, 4.0)
    expected_slope = float(np.dot(tau, future_bps) / np.dot(tau, tau))
    assert np.isclose(targets.slope_bps_per_minute[0, 0], expected_slope)
    assert np.isclose(targets.line_end_bps[0, 0], expected_slope * 3.0)
    current_price = close[0]
    price_slope = current_price * expected_slope / 10_000.0
    fitted_price = current_price + price_slope * tau
    assert np.allclose(
        fitted_price,
        current_price * (1.0 + expected_slope * tau / 10_000.0),
    )


def test_regime_targets_distinguish_friction_continuation_and_reversal() -> None:
    n = 2_500
    center = 1_800
    log_price = np.zeros(n, dtype=np.float64)
    log_price[: center + 1] = np.arange(center + 1) * 0.00002
    log_price[center + 1 :] = log_price[center] - np.arange(1, n - center) * 0.00003
    close = 2_000.0 * np.exp(log_price)
    segments = np.zeros(n, dtype=np.int32)
    lines = build_line_targets(close, segments, (10, 30, 60))
    regimes = build_regime_targets(close, segments, lines, context_minutes=60)
    assert np.all(regimes.regime[center] == REVERSAL)

    monotonic = 2_000.0 * np.exp(np.arange(n) * 0.00002)
    monotonic_lines = build_line_targets(monotonic, segments, (10, 30, 60))
    monotonic_regimes = build_regime_targets(monotonic, segments, monotonic_lines, context_minutes=60)
    assert np.all(monotonic_regimes.regime[center] == CONTINUATION)

    flat = np.full(n, 2_000.0)
    flat_lines = build_line_targets(flat, segments, (10, 30, 60))
    flat_regimes = build_regime_targets(flat, segments, flat_lines, context_minutes=60)
    assert np.all(flat_regimes.regime[center] == FRICTION)


def test_gold_splits_are_purged_and_model_heads_align(tmp_path: Path) -> None:
    n = 3_500
    close = 2_000.0 * np.exp(np.arange(n) * 0.00001 + np.sin(np.arange(n) / 31.0) * 0.0002)
    path = tmp_path / "gold.parquet"
    _write_candles(path, close)
    candles = build_candle_dataset(path)
    lines = build_line_targets(candles.close, candles.segment_id, (5, 15, 60))
    eligible = eligible_centers(candles, lines, context_minutes=60)
    config = GoldExperimentConfig(
        horizons_minutes=(5, 15, 60),
        purge_minutes=120,
        hidden_size=16,
        tcn_layers=2,
        embedding_size=8,
        batch_size=8,
        max_epochs=2,
        early_stopping_patience=1,
    )
    splits = chronological_gold_splits(candles.ts_ns, eligible, config)
    assert candles.ts_ns[splits.train].max() < splits.train_end_ns - splits.purge_ns
    assert candles.ts_ns[splits.valid].min() >= splits.train_end_ns + splits.purge_ns
    shape = ModelShape(input_size=candles.x.shape[1], horizons=3, hidden_size=16, layers=2, embedding_size=8)
    batch = torch.randn(4, 60, candles.x.shape[1])
    line = DirectLineForecaster(shape)(batch)
    assert line["mean"].shape == (4, 3)
    assert line["sigma"].shape == (4, 3)
    slope = DirectSlopeForecaster(shape)(batch)
    assert slope["slope"].shape == (4, 3)
    assert set(slope) == {"slope"}
    regime = DirectRegimeForecaster(shape)(batch)
    assert regime["regime_logits"].shape == (4, 3, 3)
    retrieval = RetrievalForecaster(shape, task="line")(batch)
    assert retrieval["embedding"].shape == (4, 8)
    assert torch.allclose(torch.linalg.norm(retrieval["embedding"], dim=1), torch.ones(4), atol=1e-5)
