from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stargaze_ml.scores import (
    ConsensusScores,
    ForwardCleanupConfig,
    MarketKind,
    ScoreCube,
    aggregate_horizons,
    clean_forward_labels,
    compute_consensus,
    compute_legacy_score_cube,
    compute_score_cube,
    legacy_300ms_horizon_steps,
    legacy_300ms_token_horizons,
    legacy_horizon_weights,
    legacy_horizons_seconds,
    robust_weighted_aggregate,
    weighted_row_mean,
    weighted_row_median,
)


SECOND_NS = 1_000_000_000


def _single_venue_cube(
    prices: np.ndarray,
    *,
    horizons: tuple[float, ...] = (2.0,),
    cost_bps: float = 0.0,
) -> ScoreCube:
    ts_ns = np.arange(len(prices), dtype=np.int64) * SECOND_NS
    return compute_score_cube(
        ts_ns,
        prices,
        prices,
        horizons_seconds=horizons,
        cost_bps=cost_bps,
        venue_names=("test",),
        market_kinds=(MarketKind.SPOT,),
    )


def test_forward_backward_hand_values_clip_and_use_full_denominator() -> None:
    cube = _single_venue_cube(np.array([100.0, 110.0, 90.0, 120.0]), cost_bps=10.0)

    # Forward at t=0 sees +1000 bps then a losing point. The losing point is
    # clipped to zero but still consumes half of the horizon denominator.
    assert cube.forward_long[0, 0, 0] == pytest.approx((990.0 + 0.0) / 2.0)
    assert cube.forward_short[0, 0, 0] == pytest.approx((0.0 + (10_000.0 / 9.0 - 10.0)) / 2.0)

    # Backward at t=2 compares the current executable quotes with both past
    # entries and uses only information at or before t=2.
    assert cube.backward_long[2, 0, 0] == 0.0
    expected_short = ((10_000.0 / 9.0 - 10.0) + (20_000.0 / 9.0 - 10.0)) / 2.0
    assert cube.backward_short[2, 0, 0] == pytest.approx(expected_short)
    assert not cube.backward_valid[:2, 0, 0].any()
    assert not cube.forward_valid[-2:, 0, 0].any()
    assert np.isnan(cube.backward_long[0, 0, 0])


def test_non_integral_wall_clock_horizon_uses_only_ticks_inside_window() -> None:
    cadence_ns = 100_000_000
    ts_ns = np.arange(4, dtype=np.int64) * cadence_ns
    prices = np.array([100.0, 110.0, 90.0, 100.0])
    cube = compute_score_cube(
        ts_ns,
        prices,
        prices,
        horizons_seconds=(0.25,),
        venue_names=("test",),
        market_kinds=("spot",),
    )

    # The points at +0.1s and +0.2s are in (i, i+0.25s]; +0.3s only proves
    # that the full wall-clock horizon is observed.
    assert cube.forward_long[0, 0, 0] == pytest.approx(500.0)
    assert cube.forward_valid[:, 0, 0].tolist() == [True, False, False, False]
    assert cube.backward_valid[:, 0, 0].tolist() == [False, False, False, True]


def test_backward_scores_are_prefix_invariant() -> None:
    prices = np.arange(100.0, 108.0)
    full = _single_venue_cube(prices, horizons=(1.0, 2.0))
    changed = prices.copy()
    changed[5:] = [200.0, 70.0, 300.0]
    with_different_future = _single_venue_cube(changed, horizons=(1.0, 2.0))

    np.testing.assert_allclose(
        full.backward_long[:5],
        with_different_future.backward_long[:5],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        full.backward_short[:5],
        with_different_future.backward_short[:5],
        equal_nan=True,
    )
    np.testing.assert_array_equal(full.backward_valid[:5], with_different_future.backward_valid[:5])


def test_forward_scores_depend_on_future_quotes() -> None:
    prices = np.arange(100.0, 108.0)
    baseline = _single_venue_cube(prices, horizons=(2.0,))
    changed = prices.copy()
    changed[5] = 200.0
    future_changed = _single_venue_cube(changed, horizons=(2.0,))

    assert future_changed.forward_long[3, 0, 0] > baseline.forward_long[3, 0, 0]
    assert future_changed.backward_long[3, 0, 0] == baseline.backward_long[3, 0, 0]


def test_weighted_row_statistics_and_mask_renormalization() -> None:
    values = np.array([[1.0, 10.0, 100.0], [1.0, np.nan, 100.0]])
    weights = np.array([0.2, 0.5, 0.3])

    np.testing.assert_allclose(weighted_row_median(values[:1], weights), [10.0])
    np.testing.assert_allclose(weighted_row_mean(values[:1], weights), [35.2])
    np.testing.assert_allclose(robust_weighted_aggregate(values[:1], weights), [17.56])

    mask = np.array([[True, False, True], [False, False, False]])
    masked = robust_weighted_aggregate(np.vstack((values[0], values[0])), weights, mask=mask)
    assert masked[0] == pytest.approx(0.7 * 100.0 + 0.3 * 60.4)
    assert np.isnan(masked[1])


def _manual_cube() -> ScoreCube:
    shape = (1, 1, 6)
    forward_long = np.array([[[1.0, 2.0, 100.0, 10.0, 11.0, 1000.0]]])
    forward_short = forward_long * 2.0
    backward_long = forward_long + 1.0
    backward_short = forward_long + 2.0
    valid = np.ones(shape, dtype=np.bool_)
    return ScoreCube(
        ts_ns=np.array([0], dtype=np.int64),
        horizons_seconds=np.array([1.0]),
        venue_names=("s1", "s2", "s3", "d1", "d2", "d3"),
        market_kinds=(
            MarketKind.SPOT,
            MarketKind.SPOT,
            MarketKind.SPOT,
            MarketKind.DERIVATIVE,
            MarketKind.DERIVATIVE,
            MarketKind.DERIVATIVE,
        ),
        forward_long=forward_long,
        forward_short=forward_short,
        backward_long=backward_long,
        backward_short=backward_short,
        forward_valid=valid,
        backward_valid=valid.copy(),
        cost_bps=0.0,
        cadence_ns=SECOND_NS,
    )


def test_consensus_medians_each_market_group_then_blends_equally() -> None:
    consensus = compute_consensus(_manual_cube())

    # Spot median=2, derivative median=11. Large venue outliers do not move it.
    assert consensus.forward_long[0, 0] == pytest.approx(6.5)
    assert consensus.forward_short[0, 0] == pytest.approx(13.0)
    assert consensus.backward_long[0, 0] == pytest.approx(7.5)
    assert consensus.forward_valid[0, 0]


def test_consensus_masks_instead_of_falling_back_to_one_venue() -> None:
    cube = _manual_cube()
    valid = cube.forward_valid.copy()
    valid[0, 0, 1] = False
    masked_cube = replace(cube, forward_valid=valid)
    consensus = compute_consensus(masked_cube)

    assert not consensus.spot_forward_valid[0, 0]
    assert consensus.derivative_forward_valid[0, 0]
    assert not consensus.forward_valid[0, 0]
    assert np.isnan(consensus.forward_long[0, 0])


def test_quote_and_segment_masks_invalidate_complete_score_windows() -> None:
    ts_ns = np.arange(5, dtype=np.int64) * SECOND_NS
    prices = np.arange(100.0, 105.0)[:, None]
    valid = np.ones((5, 1), dtype=np.bool_)
    valid[2, 0] = False
    cube = compute_score_cube(
        ts_ns,
        prices,
        prices,
        horizons_seconds=(1.0,),
        valid=valid,
        segment_id=np.array([0, 0, 0, 1, 1]),
        venue_names=("test",),
        market_kinds=("spot",),
    )

    assert cube.forward_valid[:, 0, 0].tolist() == [True, False, False, True, False]
    assert cube.backward_valid[:, 0, 0].tolist() == [False, True, False, False, True]


def test_horizon_aggregation_preserves_censoring_by_default() -> None:
    ts_ns = np.array([0, SECOND_NS], dtype=np.int64)
    values = np.array([[1.0, 3.0], [2.0, np.nan]])
    mask = np.array([[True, True], [True, False]])
    consensus = ConsensusScores(
        ts_ns=ts_ns,
        horizons_seconds=np.array([1.0, 2.0]),
        forward_long=values,
        forward_short=values,
        backward_long=values,
        backward_short=values,
        forward_valid=mask,
        backward_valid=mask,
        spot_forward_valid=mask,
        derivative_forward_valid=mask,
        spot_backward_valid=mask,
        derivative_backward_valid=mask,
    )

    strict = aggregate_horizons(consensus, weights=np.array([0.5, 0.5]))
    partial = aggregate_horizons(
        consensus,
        weights=np.array([0.5, 0.5]),
        require_all_horizons=False,
    )
    assert strict.forward_valid.tolist() == [True, False]
    assert np.isnan(strict.forward_long[1])
    assert partial.forward_valid.tolist() == [True, True]
    assert partial.forward_long[1] == pytest.approx(2.0)


def test_legacy_horizons_steps_and_weights_match_old_builder() -> None:
    horizons = legacy_horizons_seconds()
    steps = legacy_300ms_horizon_steps()
    weights = legacy_horizon_weights()
    expected = 0.15 + np.exp(-0.5 * ((horizons - 22.5) / 8.0) ** 2)
    expected /= expected.sum()

    np.testing.assert_array_equal(horizons, np.arange(1.0, 61.0))
    np.testing.assert_array_equal(steps[:3], [3, 7, 10])
    assert steps[-1] == 200
    np.testing.assert_array_equal(legacy_300ms_token_horizons(), np.arange(1, 61))
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-15)
    assert weights.sum() == pytest.approx(1.0)


def test_explicit_legacy_mode_uses_rounded_300ms_token_offsets() -> None:
    cadence_ns = 300_000_000
    ts_ns = np.arange(10, dtype=np.int64) * cadence_ns
    prices = np.arange(100.0, 110.0)
    legacy = compute_legacy_score_cube(
        ts_ns,
        prices,
        prices,
        horizons_seconds=(1.0, 2.0),
        venue_names=("test",),
        market_kinds=("spot",),
    )
    wall_clock = compute_score_cube(
        ts_ns,
        prices,
        prices,
        horizons_seconds=(1.0, 2.0),
        venue_names=("test",),
        market_kinds=("spot",),
    )

    # Legacy 2 seconds rounds to seven 300 ms tokens (2.1 seconds); exact
    # wall-clock mode includes six ticks through 1.8 seconds.
    assert legacy.forward_long[0, 1, 0] == pytest.approx(400.0)
    assert wall_clock.forward_long[0, 1, 0] == pytest.approx(350.0)
    np.testing.assert_array_equal(legacy.horizons_seconds, [1.0, 2.0])


def test_forward_cleanup_removes_an_impulse_but_keeps_a_wide_lobe_and_masks_gaps() -> None:
    cadence_ns = 100_000_000
    ts_ns = np.arange(60, dtype=np.int64) * cadence_ns
    forward_long = np.zeros(60)
    forward_short = np.zeros(60)
    forward_long[20:40] = 2.0
    forward_short[10] = 2.0
    valid = np.ones(60, dtype=np.bool_)
    valid[50] = False
    config = ForwardCleanupConfig(
        median_window_seconds=0.3,
        gaussian_sigma_seconds=0.1,
        gaussian_truncate=2.0,
        min_hump_peak_bps=0.5,
        min_hump_width_seconds=0.5,
        min_hump_area_bps_seconds=0.2,
    )

    cleaned = clean_forward_labels(
        ts_ns,
        forward_long,
        forward_short,
        valid=valid,
        config=config,
    )

    assert np.nanmax(cleaned.long) == pytest.approx(2.0, abs=1e-6)
    assert np.nanmax(cleaned.short) == 0.0
    assert not cleaned.valid[50]
    assert np.isnan(cleaned.long[50])


def test_irregular_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="regular wall-clock grid"):
        compute_score_cube(
            np.array([0, SECOND_NS, 3 * SECOND_NS]),
            np.ones(3),
            np.ones(3),
            horizons_seconds=(1.0,),
            venue_names=("test",),
            market_kinds=("spot",),
        )
