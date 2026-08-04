from __future__ import annotations

import numpy as np

from stargaze_ml.config import DataConfig
from stargaze_ml.contracts import Action, CausalFrames, Packet, PositionSide, StreamSpec, VENUES
from stargaze_ml.features.state import FLAT_FEATURE_NAMES, MarketState, VENUE_FEATURE_NAMES
from stargaze_ml.labels import build_labels
from stargaze_ml.labels.peaks import build_peak_zones
from stargaze_ml.scores import build_score_bundle
from stargaze_ml.training import PolicyWindowDataset, RobustNormalizer, build_examples, purged_blocked_splits, purged_chronological_splits


def _frames(n: int = 500) -> CausalFrames:
    ts = np.arange(n, dtype=np.int64) * 100_000_000
    phase = np.linspace(0.0, 8.0 * np.pi, n)
    fair = 63_000.0 + 30.0 * np.sin(phase) + 0.02 * np.arange(n)
    offsets = np.linspace(-8.0, 8.0, len(VENUES))
    mid = fair[:, None] + offsets[None, :]
    bid = mid - 0.05
    ask = mid + 0.05
    venue_x = np.zeros((n, len(VENUES), len(VENUE_FEATURE_NAMES)), dtype=np.float32)
    venue_x[:, :, 0] = 1.0
    venue_x[:, :, 1] = 10.0
    venue_x[:, :, 2] = bid
    venue_x[:, :, 3] = ask
    venue_x[:, :, 4] = mid
    x = np.zeros((n, len(FLAT_FEATURE_NAMES)), dtype=np.float32)
    x[:, : venue_x.shape[1] * venue_x.shape[2]] = venue_x.reshape(n, -1)
    return CausalFrames(
        ts_ns=ts,
        x=x,
        venue_x=venue_x,
        bid=bid,
        ask=ask,
        valid=np.ones(n, dtype=bool),
        segment_id=np.zeros(n, dtype=np.int32),
        feature_names=FLAT_FEATURE_NAMES,
        venue_feature_names=VENUE_FEATURE_NAMES,
    )


def test_book_packet_is_atomic_and_snapshot_replaces_state(tmp_path) -> None:
    spec = StreamSpec("bybit", "linear", "BTCUSDT", "orderbook", tmp_path / "unused", "book")
    state = MarketState()
    packet = Packet(
        spec,
        1,
        {
            "is_snapshot": np.asarray([True, True]),
            "event_type": np.asarray(["snapshot", "snapshot"], dtype=object),
            "side": np.asarray(["bid", "ask"], dtype=object),
            "price": np.asarray([100.0, 101.0]),
            "quantity": np.asarray([2.0, 3.0]),
            "action": np.asarray(["set", "set"], dtype=object),
            "order_count": np.asarray([1.0, 1.0]),
        },
    )
    state.apply(packet)
    assert state.books["bybit_perpetual"].bbo() == (100.0, 101.0)
    # Initial snapshots warm books; only a Binance perpetual re-snapshot marks
    # a discontinuity in the executable target stream.
    assert state.segment_id == 0


def test_kraken_l3_state_truncates_to_subscribed_price_depth(tmp_path) -> None:
    spec = StreamSpec("kraken", "spot", "BTC/USD", "level3", tmp_path / "unused", "l3")
    prices = np.arange(1001, dtype=np.float64) + 10_000.0
    packet = Packet(
        spec,
        1,
        {
            "is_snapshot": np.ones(1001, dtype=bool),
            "event_type": np.full(1001, "snapshot", dtype=object),
            "side": np.full(1001, "bid", dtype=object),
            "price": prices,
            "quantity": np.ones(1001),
            "action": np.full(1001, "set", dtype=object),
            "order_id": np.asarray([f"o{i}" for i in range(1001)], dtype=object),
        },
    )
    state = MarketState()
    state.apply(packet)
    assert len(state.l3.bid_prices) == 1000
    assert prices[0] not in state.l3.bid_levels
    assert prices[-1] in state.l3.bid_levels


def test_labels_examples_and_dataset_are_position_conditioned() -> None:
    frames = _frames()
    horizons = (0.5, 1.0, 2.0)
    bundle = build_score_bundle(
        frames.ts_ns,
        frames.bid,
        frames.ask,
        horizons_seconds=horizons,
        cost_bps=0.0,
        valid=np.ones_like(frames.bid, dtype=bool),
        segment_id=frames.segment_id,
        venue_names=VENUES,
        require_all_horizons=False,
    )
    score_valid = np.any(bundle.consensus.forward_valid, axis=1) & np.any(bundle.consensus.backward_valid, axis=1)
    frames.valid &= score_valid
    splits = purged_chronological_splits(frames.ts_ns, frames.valid, purge_seconds=0.5)
    labels = build_labels(
        frames,
        forward_long_h=bundle.consensus.forward_long,
        forward_short_h=bundle.consensus.forward_short,
        backward_long_h=bundle.consensus.backward_long,
        backward_short_h=bundle.consensus.backward_short,
        horizons_seconds=horizons,
        fit_mask=splits.train,
        event_high_quantile=0.80,
    )
    assert np.any(labels.open_long_zone | labels.open_short_zone)
    assert labels.episodes
    examples = build_examples(frames, labels, context_ticks=8, skip_stride=5, hold_stride=1)
    assert np.any(examples.state != int(PositionSide.FLAT))
    assert np.any(np.isin(examples.action, [int(Action.CLOSE_LONG), int(Action.CLOSE_SHORT)]))
    x_norm = RobustNormalizer.fit(frames.x, splits.train)
    venue_norm = RobustNormalizer.fit(frames.venue_x, splits.train)
    dataset = PolicyWindowDataset(
        frames,
        labels,
        examples,
        forward_long_h=bundle.consensus.forward_long,
        forward_short_h=bundle.consensus.forward_short,
        backward_long_h=bundle.consensus.backward_long,
        backward_short_h=bundle.consensus.backward_short,
        horizons_seconds=horizons,
        context_ticks=8,
        x_normalizer=x_norm,
        venue_normalizer=venue_norm,
    )
    sample = dataset[0]
    assert sample["x"].shape == (8, dataset.input_dim)
    assert sample["venue_x"].shape == (8, len(VENUES), len(VENUE_FEATURE_NAMES))
    assert np.isfinite(sample["x"].numpy()).all()
    assert np.isfinite(dataset.forward_long_h).all()
    assert np.isfinite(dataset.forward_short_h).all()
    assert np.isfinite(dataset.backward_long_h).all()
    assert np.isfinite(dataset.backward_short_h).all()
    assert np.any(~dataset.forward_valid)
    position_rows = np.flatnonzero(examples.state != int(PositionSide.FLAT))
    chosen = int(position_rows[0])
    position_sample = dataset[chosen]
    entry = int(examples.entry_idx[chosen])
    center = int(examples.center_idx[chosen])
    entry_offset = entry - (center - 8 + 1)
    assert position_sample["position_state"][entry_offset] == int(PositionSide.FLAT)
    if entry_offset + 1 < 8:
        assert position_sample["position_state"][entry_offset + 1] == int(examples.state[chosen])


def test_purged_splits_are_disjoint() -> None:
    frames = _frames(100)
    splits = purged_chronological_splits(frames.ts_ns, frames.valid, purge_seconds=0.2)
    assert not np.any(splits.train & splits.valid)
    assert not np.any(splits.train & splits.holdout)
    assert not np.any(splits.valid & splits.holdout)
    assert np.flatnonzero(splits.train)[-1] < np.flatnonzero(splits.valid)[0] < np.flatnonzero(splits.holdout)[0]


def test_blocked_splits_keep_final_holdout_and_disjoint_validation_windows() -> None:
    frames = _frames(1000)
    splits = purged_blocked_splits(frames.ts_ns, frames.valid, purge_seconds=0.5)
    assert not np.any(splits.train & splits.valid)
    assert not np.any(splits.train & splits.holdout)
    assert not np.any(splits.valid & splits.holdout)
    assert np.flatnonzero(splits.holdout)[0] > np.flatnonzero(splits.valid)[-1]
    validation_runs = 1 + int(np.sum(np.diff(np.flatnonzero(splits.valid)) > 1))
    assert validation_runs == 2


def test_nms_peak_zones_split_multiple_peaks_above_positive_baseline() -> None:
    score = np.full(120, 1.0, dtype=np.float64)
    score[20] = 5.0
    score[70] = 6.0
    zones = build_peak_zones(
        score,
        np.ones(120, dtype=bool),
        np.zeros(120, dtype=np.int32),
        high=4.0,
        low=0.4,
        min_ratio=0.75,
        nms_ticks=10,
    )
    assert [event.peak for event in zones.events] == [20, 70]
    assert zones.zone[20] and zones.zone[70]
    assert zones.event_id[20] != zones.event_id[70]
