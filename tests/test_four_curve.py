from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np
import pyarrow as pa
import pytest
import torch

from market_collector.record_log import append_parquet_block, initialize_record_log
from market_collector.records import SCHEMA, normalize_record
from stargaze_ml.contracts import CausalFrames, Packet, StreamSpec, VENUES
from stargaze_ml.data import DatasetCatalog
from stargaze_ml.data.stream import iter_packets
from stargaze_ml.data.incremental import extract_record_log_extension
from market_collector.record_log import iter_record_log_tables
from stargaze_ml.features.state import BookState, MarketState, VENUE_FEATURE_NAMES
from stargaze_ml.labels import CURVE_NAMES, build_four_curve_targets
from stargaze_ml.models import CurveModelConfig, FourCurveCausalTransformer
from stargaze_ml.deployment import FourCurveRuntime, export_four_curve_bundle
from stargaze_ml.contracts import PositionSide
from stargaze_ml.training import causal_centers
from stargaze_ml.training import (
    causal_backward_score_features,
    causal_high_order_features,
    multihorizon_forward_edge_targets,
    stationary_market_features,
)
from stargaze_ml.training.curve_trainer import _load_initial_model, _loss
from stargaze_ml.scores import compute_score_cube
from stargaze_ml.curve_pipeline import _dense_score_auxiliary_targets, _percentile_scores
from stargaze_ml.cli import build_parser


def _synthetic_frames(n: int = 600) -> CausalFrames:
    ts = np.arange(n, dtype=np.int64) * 1_000_000_000
    fair = 64_000.0 + 80.0 * np.sin(np.linspace(0.0, 12.0 * np.pi, n))
    bid = np.repeat((fair - 0.05)[:, None], len(VENUES), axis=1)
    ask = np.repeat((fair + 0.05)[:, None], len(VENUES), axis=1)
    venue_x = np.zeros((n, len(VENUES), len(VENUE_FEATURE_NAMES)), dtype=np.float32)
    venue_x[:, :, 0] = 1.0
    venue_x[:, :, 1] = 10.0
    venue_x[:, :, 2] = bid
    venue_x[:, :, 3] = ask
    venue_x[:, :, 4] = 0.5 * (bid + ask)
    venue_x[:, :, VENUE_FEATURE_NAMES.index("best_bid_log_qty")] = np.log1p(10.0)
    venue_x[:, :, VENUE_FEATURE_NAMES.index("best_ask_log_qty")] = np.log1p(10.0)
    global_width = 21
    x = np.zeros((n, venue_x.shape[1] * venue_x.shape[2] + global_width), dtype=np.float32)
    x[:, : venue_x.shape[1] * venue_x.shape[2]] = venue_x.reshape(n, -1)
    return CausalFrames(
        ts, x, venue_x, bid, ask, np.ones(n, dtype=bool), np.zeros(n, dtype=np.int32),
        tuple(f"f{i}" for i in range(x.shape[1])), VENUE_FEATURE_NAMES,
    )


def test_four_curve_targets_are_named_bounded_and_censored() -> None:
    frames = _synthetic_frames()
    targets = build_four_curve_targets(
        frames,
        horizons_seconds=(15.0, 30.0, 60.0),
        focus_seconds=30.0,
        fit_mask=np.arange(len(frames.ts_ns)) < 300,
        fee_round_trip_bps=1.0,
    )
    assert targets.curve_names == CURVE_NAMES
    assert targets.values.shape == (len(frames.ts_ns), 4)
    assert np.isfinite(targets.values).all()
    assert np.all((targets.values >= 0.0) & (targets.values <= 1.0))
    assert not targets.valid[:60, 0].any()
    assert not targets.valid[-60:, 1].any()


def test_four_curve_targets_can_reuse_frozen_train_thresholds() -> None:
    frames = _synthetic_frames()
    fitted = build_four_curve_targets(
        frames,
        horizons_seconds=(15.0, 30.0, 60.0),
        focus_seconds=30.0,
        fit_mask=np.arange(len(frames.ts_ns)) < 300,
        fee_round_trip_bps=1.0,
    )
    frozen = build_four_curve_targets(
        frames,
        horizons_seconds=(15.0, 30.0, 60.0),
        focus_seconds=30.0,
        fit_mask=np.zeros(len(frames.ts_ns), dtype=bool),
        fee_round_trip_bps=1.0,
        frozen_high_thresholds=fitted.high_thresholds,
        frozen_full_quality_thresholds=fitted.full_quality_thresholds,
    )
    assert np.array_equal(frozen.high_thresholds, fitted.high_thresholds)
    assert np.array_equal(frozen.full_quality_thresholds, fitted.full_quality_thresholds)
    assert np.array_equal(frozen.values, fitted.values)


def test_four_curve_targets_support_fixed_economic_edge_scale() -> None:
    frames = _synthetic_frames()
    first = build_four_curve_targets(
        frames,
        horizons_seconds=(15.0, 30.0, 60.0),
        focus_seconds=30.0,
        fit_mask=np.arange(len(frames.ts_ns)) < 200,
        fee_round_trip_bps=1.0,
        minimum_edge_bps=0.5,
        full_quality_edge_bps=20.0,
    )
    second = build_four_curve_targets(
        frames,
        horizons_seconds=(15.0, 30.0, 60.0),
        focus_seconds=30.0,
        fit_mask=np.arange(len(frames.ts_ns)) >= 400,
        fee_round_trip_bps=1.0,
        minimum_edge_bps=0.5,
        full_quality_edge_bps=20.0,
    )
    np.testing.assert_array_equal(first.high_thresholds, np.full(4, 0.5))
    np.testing.assert_array_equal(first.full_quality_thresholds, np.full(4, 20.0))
    np.testing.assert_array_equal(first.values, second.values)


def test_dense_forward_curves_use_separate_economic_threshold() -> None:
    frames = _synthetic_frames()
    targets = build_four_curve_targets(
        frames,
        horizons_seconds=(15.0, 30.0, 60.0),
        focus_seconds=30.0,
        fit_mask=np.ones(len(frames.ts_ns), dtype=bool),
        fee_round_trip_bps=1.0,
        minimum_edge_bps=0.5,
        forward_minimum_edge_bps=6.0,
        full_quality_edge_bps=20.0,
        forward_curve_mode="dense_edge",
    )
    np.testing.assert_array_equal(targets.high_thresholds, [0.5, 6.0, 0.5, 6.0])
    assert np.all((targets.values >= 0.0) & (targets.values <= 1.0))


def test_four_curve_fixed_edge_scale_rejects_invalid_bounds() -> None:
    frames = _synthetic_frames()
    with pytest.raises(ValueError, match="must exceed"):
        build_four_curve_targets(
            frames,
            horizons_seconds=(15.0, 30.0, 60.0),
            focus_seconds=30.0,
            fit_mask=np.ones(len(frames.ts_ns), dtype=bool),
            fee_round_trip_bps=1.0,
            minimum_edge_bps=2.0,
            full_quality_edge_bps=1.0,
        )


def test_four_curve_transformer_is_causal_and_returns_only_four_scores() -> None:
    torch.manual_seed(3)
    model = FourCurveCausalTransformer(CurveModelConfig(7, 5, d_model=16, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.0, num_venues=3, use_venue_embeddings=True)).eval()
    base = torch.randn(2, 12, 7)
    venue = torch.randn(2, 12, 3, 5)
    mask = torch.ones(2, 12, 3, dtype=torch.bool)
    changed_base = base.clone(); changed_base[:, 7:] *= 100.0
    changed_venue = venue.clone(); changed_venue[:, 7:] *= -100.0
    with torch.no_grad():
        original = model(base, venue, mask)
        changed = model(changed_base, changed_venue, mask)
    assert original.scores.shape == (2, 12, 4)
    assert torch.equal(original.scores[:, :7], changed.scores[:, :7])
    assert torch.all((original.scores >= 0.0) & (original.scores <= 1.0))


def test_causal_centers_do_not_depend_on_future_target_validity() -> None:
    frames = _synthetic_frames(20)
    mask = np.ones(20, dtype=bool)
    centers = causal_centers(frames, mask, context_ticks=5)
    assert np.array_equal(centers, np.arange(4, 20))


def test_backward_score_features_are_strictly_causal() -> None:
    frames = _synthetic_frames(240)
    horizons = (15.0, 30.0, 60.0)
    original = causal_backward_score_features(frames, horizons, cost_bps=1.0)
    changed = _synthetic_frames(240)
    cutoff = 140
    changed.bid[cutoff + 1 :] *= 1.2
    changed.ask[cutoff + 1 :] *= 1.2
    changed.venue_x[cutoff + 1 :, :, 2:5] *= 1.2
    mutated = causal_backward_score_features(changed, horizons, cost_bps=1.0)
    assert np.array_equal(original[: cutoff + 1], mutated[: cutoff + 1])
    assert original.shape == (240, 3 * len(horizons))
    assert np.all((original[:, 2 * len(horizons) :] == 0.0) | (original[:, 2 * len(horizons) :] == 1.0))
    venue = 1
    cube = compute_score_cube(
        frames.ts_ns,
        frames.bid[:, [venue]],
        frames.ask[:, [venue]],
        horizons_seconds=horizons,
        cost_bps=1.0,
        valid=np.ones((len(frames.ts_ns), 1), dtype=bool),
        segment_id=frames.segment_id,
        venue_names=("binance_perpetual",),
        market_kinds=("derivative",),
        min_coverage=0.95,
    )
    assert np.allclose(original[:, : len(horizons)], np.nan_to_num(cube.backward_long[:, :, 0]), atol=1e-5)
    assert np.allclose(original[:, len(horizons) : 2 * len(horizons)], np.nan_to_num(cube.backward_short[:, :, 0]), atol=1e-5)
    assert np.array_equal(original[:, 2 * len(horizons) :] > 0.5, cube.backward_valid[:, :, 0])


def test_stationary_market_features_remove_absolute_price_and_are_causal() -> None:
    frames = _synthetic_frames(40)
    venue_width = frames.venue_x.shape[1] * frames.venue_x.shape[2]
    consensus = 0.5 * (frames.bid[:, 0] + frames.ask[:, 0])
    frames.x[:, venue_width] = consensus
    frames.x[:, venue_width + 1] = consensus - 2.0
    frames.x[:, venue_width + 2] = consensus + 3.0
    base, venue = stationary_market_features(frames)
    assert np.all(base[:, 0] == 1.0)
    assert np.max(np.abs(venue[:, :, 4])) < 1e-4
    assert np.all(base[:, 1] < 0.0)
    assert np.all(base[:, 2] > 0.0)

    shifted = _synthetic_frames(40)
    shifted.bid *= 2.0
    shifted.ask *= 2.0
    shifted.venue_x[:, :, 2:5] *= 2.0
    shifted_consensus = 2.0 * consensus
    shifted.x[:, venue_width] = shifted_consensus
    shifted.x[:, venue_width + 1] = 2.0 * (consensus - 2.0)
    shifted.x[:, venue_width + 2] = 2.0 * (consensus + 3.0)
    shifted_base, shifted_venue = stationary_market_features(shifted)
    assert np.allclose(base[:, :3], shifted_base[:, :3], atol=1e-5)
    assert np.allclose(venue[:, :, 2:5], shifted_venue[:, :, 2:5], atol=1e-5)


def test_high_order_feature_family_is_finite_aligned_and_strictly_causal() -> None:
    frames = _synthetic_frames(400)
    venue_width = frames.venue_x.shape[1] * frames.venue_x.shape[2]
    reference = 0.5 * (frames.bid[:, 0] + frames.ask[:, 0])
    frames.x[:, venue_width : venue_width + 3] = reference[:, None]
    original, names = causal_high_order_features(frames)
    cutoff = 260
    changed = _synthetic_frames(400)
    changed.x[:, venue_width : venue_width + 3] = reference[:, None]
    changed.x[cutoff + 1 :, venue_width : venue_width + 3] *= 1.1
    changed.venue_x[cutoff + 1 :, :, 2:5] *= 1.1
    mutated, mutated_names = causal_high_order_features(changed)
    assert original.shape == (400, len(names))
    assert len(names) > 100
    assert np.isfinite(original).all()
    assert names == mutated_names
    assert np.array_equal(original[: cutoff + 1], mutated[: cutoff + 1])


def test_dense_multihorizon_auxiliary_edges_match_executable_quotes() -> None:
    frames = _synthetic_frames(20)
    target, valid = multihorizon_forward_edge_targets(
        frames, (1.0, 3.0), cost_bps=1.0, scale_bps=10.0
    )
    assert target.shape == valid.shape == (20, 4)
    expected_long = (1e4 * (frames.bid[1, 1] / frames.ask[0, 1] - 1.0) - 1.0) / 10.0
    expected_short = (1e4 * (frames.bid[0, 1] / frames.ask[1, 1] - 1.0) - 1.0) / 10.0
    assert np.isclose(target[0, 0], expected_long)
    assert np.isclose(target[0, 2], expected_short)
    assert valid[0].all()
    assert not valid[-1].any()


def test_dense_score_auxiliary_replaces_invalid_nan_with_zero() -> None:
    values = np.zeros((3, 4), dtype=np.float32)
    valid = np.ones((3, 4), dtype=bool)
    raw = np.ones((3, 4), dtype=np.float32)
    raw[0, 1] = np.nan
    valid[0, 1] = False
    targets = __import__("stargaze_ml.labels", fromlist=["FourCurveTargets"]).FourCurveTargets(
        ts_ns=np.arange(3, dtype=np.int64),
        values=values,
        valid=valid,
        raw_scores=raw,
        horizons_seconds=np.array([1.0]),
        horizon_weights=np.array([1.0]),
        high_thresholds=np.zeros(4),
        full_quality_thresholds=np.ones(4),
    )
    dense, dense_valid = _dense_score_auxiliary_targets(targets)
    assert np.isfinite(dense).all()
    assert dense[0, 1] == 0.0
    assert np.array_equal(dense_valid, valid)


def test_curve_transformer_auxiliary_head_does_not_change_public_score_width() -> None:
    model = FourCurveCausalTransformer(
        CurveModelConfig(
            7, 5, d_model=16, nhead=4, num_layers=1, dim_feedforward=32,
            dropout=0.0, num_venues=3, num_aux_horizons=3,
        )
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 8, 7), torch.randn(2, 8, 3, 5))
    assert output.scores.shape == (2, 8, 4)
    assert output.future_edges is not None
    assert output.future_edges.shape == (2, 8, 6)


def test_separate_task_towers_remain_causal_and_keep_curve_order() -> None:
    torch.manual_seed(17)
    model = FourCurveCausalTransformer(
        CurveModelConfig(
            7, 5, d_model=16, nhead=4, num_layers=1, dim_feedforward=32,
            dropout=0.0, num_venues=3, separate_task_towers=True,
        )
    ).eval()
    base = torch.randn(2, 10, 7)
    venue = torch.randn(2, 10, 3, 5)
    changed = base.clone()
    changed[:, 6:] += 100.0
    with torch.no_grad():
        original = model(base, venue)
        mutated = model(changed, venue)
    assert original.scores.shape == (2, 10, 4)
    assert torch.equal(original.scores[:, :6], mutated.scores[:, :6])
    assert model.task_towers is not None
    assert model.task_heads is not None


def test_peak_balanced_loss_penalizes_missed_peak() -> None:
    class Output:
        pass

    output = Output()
    output.logits = torch.full((1, 1, 4), -1.0)
    output.scores = torch.sigmoid(output.logits)
    batch = {
        "target": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "valid": torch.ones(1, 4, dtype=torch.bool),
    }
    neutral = _loss(output, batch, torch.ones(4))
    balanced = _loss(output, batch, torch.tensor([32.0, 1.0, 1.0, 1.0]))
    assert balanced > neutral

    batch["weight"] = torch.tensor([[5.0, 1.0, 1.0, 1.0]])
    event_weighted = _loss(output, batch, torch.tensor([32.0, 1.0, 1.0, 1.0]))
    assert event_weighted > balanced


def test_four_curve_runtime_decisions_use_only_model_scores_and_position() -> None:
    assert FourCurveRuntime.decide(PositionSide.FLAT, np.array([0.1, 0.8, 0.2, 0.3])).action == "open_long"
    assert FourCurveRuntime.decide(PositionSide.FLAT, np.array([0.1, 0.2, 0.2, 0.3])).action == "skip"
    assert FourCurveRuntime.decide(PositionSide.LONG, np.array([0.8, 0.1, 0.2, 0.3])).action == "close_long"
    assert FourCurveRuntime.decide(PositionSide.SHORT, np.array([0.1, 0.2, 0.4, 0.3])).action == "hold"
    assert FourCurveRuntime.decide(PositionSide.FLAT, np.array([0.8, 0.9, 0.2, 0.1])).action == "skip"
    thresholds = (0.9, 0.8, 0.7, 0.6)
    assert FourCurveRuntime.decide(PositionSide.FLAT, np.array([0.1, 0.95, 0.2, 0.7]), thresholds).action == "open_long"
    assert FourCurveRuntime.decide(PositionSide.LONG, np.array([0.89, 0.9, 0.2, 0.1]), thresholds).action == "hold"
    assert FourCurveRuntime.decide(PositionSide.SHORT, np.array([0.1, 0.2, 0.71, 0.3]), thresholds).action == "close_short"


def _write_mrec(path: Path, channel: str, event_type: str) -> None:
    initialize_record_log(path)
    rows = [
        normalize_record({
            "exchange": "binance", "market": "um_futures", "symbol": "BTCUSDT",
            "channel": channel, "event_type": event_type, "event_id": 1, "row_idx": row,
            "local_ts_ns": 1_000_000_000, "exchange_ts_ns": 1_000_000_000,
            "is_snapshot": event_type == "snapshot", "side": side, "price": price,
            "quantity": 1.0, "action": "set", "trade_id": str(row), "taker_side": "buy",
        })
        for row, (side, price) in enumerate((("bid", 100.0), ("ask", 101.0)))
    ]
    with path.open("ab") as stream:
        append_parquet_block(stream, pa.Table.from_pylist(rows, schema=SCHEMA))


def test_catalog_and_packet_reader_support_mrec(tmp_path: Path) -> None:
    _write_mrec(tmp_path / "binance_um_futures_BTCUSDT_depth.mrec", "depth", "snapshot")
    _write_mrec(tmp_path / "binance_um_futures_BTCUSDT_trades.mrec", "trades", "trade")
    catalog = DatasetCatalog.discover(tmp_path)
    assert {stream.kind for stream in catalog.streams} == {"book", "trade"}
    packets = list(iter_packets(next(stream for stream in catalog.streams if stream.kind == "book")))
    assert len(packets) == 1
    assert packets[0].size == 2


def test_record_log_extension_keeps_overlap_and_appended_frames(tmp_path: Path) -> None:
    old = tmp_path / "old.mrec"
    live = tmp_path / "live.mrec"
    extension = tmp_path / "extension.mrec"
    initialize_record_log(old)

    def table(ts_ns: int, event_id: int) -> pa.Table:
        row = normalize_record({
            "exchange": "binance", "market": "um_futures", "symbol": "BTCUSDT",
            "channel": "depth", "event_type": "update", "event_id": event_id,
            "row_idx": 0, "local_ts_ns": ts_ns, "exchange_ts_ns": ts_ns,
            "side": "bid", "price": 100.0 + event_id, "quantity": 1.0, "action": "set",
        })
        return pa.Table.from_pylist([row], schema=SCHEMA)

    with old.open("ab") as stream:
        append_parquet_block(stream, table(1_000_000_000, 1))
        append_parquet_block(stream, table(2_000_000_000, 2))
    shutil.copy2(old, live)
    with live.open("ab") as stream:
        append_parquet_block(stream, table(3_000_000_000, 3))
    result = extract_record_log_extension(old, live, extension, after_ts_ns=1_500_000_000)
    timestamps = np.concatenate([
        block["local_ts_ns"].to_numpy() for block in iter_record_log_tables(extension)
    ])
    assert result["rows"] == 2
    assert np.array_equal(timestamps, np.array([2_000_000_000, 3_000_000_000]))


def test_book_tick_collapse_matches_sequential_updates_with_side_aliases(tmp_path: Path) -> None:
    stream = StreamSpec("binance", "um_futures", "BTCUSDT", "depth", tmp_path / "depth.mrec", "book")

    def packet(ts_ns: int, *, side: list[str], price: list[float], quantity: list[float], action: list[str], snapshot: bool) -> Packet:
        size = len(side)
        return Packet(stream, ts_ns, {
            "is_snapshot": np.asarray([snapshot] * size),
            "event_type": np.asarray(["snapshot" if snapshot else "update"] * size, dtype=object),
            "side": np.asarray(side, dtype=object),
            "price": np.asarray(price),
            "quantity": np.asarray(quantity),
            "action": np.asarray(action, dtype=object),
            "order_count": np.arange(size, dtype=float),
            "trade_id": np.asarray(["irrelevant"] * size, dtype=object),
        })

    packets = [
        packet(1, side=["bid", "ask"], price=[100.0, 101.0], quantity=[1.0, 1.0], action=["set", "set"], snapshot=True),
        packet(2, side=["Buy", "Sell", "Ask"], price=[100.0, 101.0, 102.0], quantity=[2.0, 0.0, 3.0], action=["set", "delete", "set"], snapshot=False),
    ]
    sequential = MarketState(cadence_ms=1_000)
    collapsed = MarketState(cadence_ms=1_000)
    for update in packets:
        sequential.apply(update)
    collapsed.apply_tick(packets)
    assert collapsed.books["binance_perpetual"].bids == sequential.books["binance_perpetual"].bids
    assert collapsed.books["binance_perpetual"].asks == sequential.books["binance_perpetual"].asks
    assert collapsed.books["binance_perpetual"].bid_orders == sequential.books["binance_perpetual"].bid_orders
    assert collapsed.books["binance_perpetual"].ask_orders == sequential.books["binance_perpetual"].ask_orders


def test_book_arrays_keep_exact_best_thousand_levels() -> None:
    book = BookState(
        bids={float(price): float(price) / 10.0 for price in range(1, 1_501)},
        asks={float(price): float(price) / 20.0 for price in range(2_000, 3_501)},
        warm=True,
    )
    bid_price, bid_qty, ask_price, ask_qty = book.arrays()
    assert np.array_equal(bid_price, np.arange(1_500.0, 500.0, -1.0))
    assert np.array_equal(bid_qty, bid_price / 10.0)
    assert np.array_equal(ask_price, np.arange(2_000.0, 3_000.0))
    assert np.array_equal(ask_qty, ask_price / 20.0)


def test_percentile_scores_use_an_independent_reference_per_curve() -> None:
    reference = np.asarray(
        [[0.1, 0.4, 0.3, 0.8], [0.2, 0.2, 0.5, 0.6], [0.3, 0.6, 0.1, 0.7]],
        dtype=np.float32,
    )
    sorted_reference = np.sort(reference, axis=0)
    scores = _percentile_scores(
        np.asarray([[0.2, 0.5, 0.4, 0.9]], dtype=np.float32), sorted_reference
    )
    assert np.allclose(scores, [[2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0]])


def test_four_curve_cli_reads_raw_in_place_unless_copy_is_explicit() -> None:
    parser = build_parser()
    default = parser.parse_args(["run-four-curve"])
    copied = parser.parse_args(["run-four-curve", "--copy-raw-snapshot"])

    assert default.copy_raw_snapshot is False
    assert copied.copy_raw_snapshot is True


def test_shared_checkpoint_can_initialize_separate_task_towers() -> None:
    common = dict(
        input_dim=4,
        venue_feature_dim=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        num_venues=2,
        use_venue_embeddings=True,
        num_aux_horizons=2,
        auxiliary_output_dim=4,
    )
    shared_config = CurveModelConfig(**common, separate_task_towers=False)
    separate_config = CurveModelConfig(**common, separate_task_towers=True)
    shared = FourCurveCausalTransformer(shared_config).eval()
    separate = FourCurveCausalTransformer(separate_config).eval()
    checkpoint = {
        "model_config": shared_config.to_dict(),
        "model_state": shared.state_dict(),
    }
    _load_initial_model(separate, separate_config, checkpoint)
    base = torch.randn(2, 5, 4)
    venue = torch.randn(2, 5, 2, 3)
    mask = torch.ones(2, 5, 2, dtype=torch.bool)
    with torch.no_grad():
        expected = shared(base, venue, mask)
        actual = separate(base, venue, mask)
    assert torch.allclose(actual.logits, expected.logits, atol=1e-6)
    assert torch.allclose(actual.future_edges, expected.future_edges, atol=1e-6)


def test_export_bundle_keeps_selected_policy_and_calibration(tmp_path: Path) -> None:
    source = tmp_path / "run"
    destination = tmp_path / "bundle"
    source.mkdir()
    for name in (
        "best_four_curve.pt",
        "economically_selected_four_curve.pt",
        "normalizers.json",
        "score_calibration.npz",
    ):
        (source / name).write_bytes(name.encode("ascii"))
    (source / "four_curve_run.json").write_text("{}", encoding="utf-8")
    (source / "four_curve_summary.json").write_text(
        json.dumps({"execution_ready": False}), encoding="utf-8"
    )
    policy = {
        "checkpoint": "economically_selected_four_curve.pt",
        "score_space": "percentile",
        "curve_thresholds": [0.995, 0.997, 0.995, 0.997],
    }
    (source / "economic_policy.json").write_text(json.dumps(policy), encoding="utf-8")

    manifest = export_four_curve_bundle(source, destination)

    assert (destination / "economically_selected_four_curve.pt").exists()
    assert (destination / "economic_policy.json").exists()
    assert (destination / "score_calibration.npz").exists()
    assert manifest["action_policy"] == policy
