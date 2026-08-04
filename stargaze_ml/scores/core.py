from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from .types import (
    AggregatedScores,
    ConsensusScores,
    MarketKind,
    ScoreBundle,
    ScoreCube,
)


BPS = 10_000.0
LEGACY_CADENCE_SECONDS = 0.3
LEGACY_HORIZON_COUNT = 60
LEGACY_HORIZONS_SECONDS = tuple(float(value) for value in range(1, LEGACY_HORIZON_COUNT + 1))
LEGACY_300MS_TOKEN_HORIZONS = tuple(range(1, LEGACY_HORIZON_COUNT + 1))
LEGACY_300MS_HORIZON_STEPS = tuple(
    int(round(seconds / LEGACY_CADENCE_SECONDS)) for seconds in LEGACY_HORIZONS_SECONDS
)

_KNOWN_MARKETS = {
    "bybit": MarketKind.DERIVATIVE,
    "okx": MarketKind.DERIVATIVE,
    "coinbase": MarketKind.SPOT,
    "kraken": MarketKind.SPOT,
    "binance_spot": MarketKind.SPOT,
    "binance_perpetual": MarketKind.DERIVATIVE,
    "bybit_perpetual": MarketKind.DERIVATIVE,
    "okx_perpetual": MarketKind.DERIVATIVE,
    "coinbase_spot": MarketKind.SPOT,
    "kraken_spot": MarketKind.SPOT,
    "deribit_perpetual": MarketKind.DERIVATIVE,
    "bitfinex_spot": MarketKind.SPOT,
    "hyperliquid_perpetual": MarketKind.DERIVATIVE,
}


def _as_timestamps_ns(ts_ns: npt.ArrayLike) -> npt.NDArray[np.int64]:
    values = np.asarray(ts_ns)
    if values.ndim != 1:
        raise ValueError("ts_ns must be one-dimensional")
    if np.issubdtype(values.dtype, np.datetime64):
        values = values.astype("datetime64[ns]").astype(np.int64)
    elif not np.issubdtype(values.dtype, np.integer):
        raise TypeError("ts_ns must contain integer nanoseconds or datetime64 values")
    return values.astype(np.int64, copy=False)


def _regular_cadence_ns(ts_ns: npt.NDArray[np.int64], cadence_ns: int | None) -> int:
    if cadence_ns is not None:
        cadence = int(cadence_ns)
        if cadence <= 0:
            raise ValueError("cadence_ns must be positive")
    elif len(ts_ns) < 2:
        raise ValueError("cadence_ns is required when fewer than two timestamps are supplied")
    else:
        cadence = int(ts_ns[1] - ts_ns[0])
    if len(ts_ns) >= 2:
        deltas = np.diff(ts_ns)
        if cadence <= 0 or bool(np.any(deltas != cadence)):
            raise ValueError("timestamps must form a strictly increasing regular wall-clock grid")
    return cadence


def _as_price_matrix(values: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [time, venue]")
    return array


def _quote_validity(
    bid: npt.NDArray[np.float64],
    ask: npt.NDArray[np.float64],
    valid: npt.ArrayLike | None,
) -> npt.NDArray[np.bool_]:
    quote_valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask > 0.0) & (bid <= ask)
    if valid is None:
        return quote_valid
    supplied = np.asarray(valid, dtype=np.bool_)
    if supplied.ndim == 1:
        if supplied.shape != (bid.shape[0],):
            raise ValueError("one-dimensional valid must align with time")
        supplied = np.broadcast_to(supplied[:, None], bid.shape)
    elif supplied.shape != bid.shape:
        raise ValueError("valid must have shape [time] or [time, venue]")
    return quote_valid & supplied


def _segments(segment_id: npt.ArrayLike | None, length: int) -> npt.NDArray:
    if segment_id is None:
        return np.zeros(length, dtype=np.int64)
    segments = np.asarray(segment_id)
    if segments.ndim != 1 or segments.shape[0] != length:
        raise ValueError("segment_id must align with timestamps")
    return segments


def _horizon_steps(
    horizons_seconds: npt.ArrayLike,
    cadence_ns: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    horizons = np.asarray(horizons_seconds, dtype=np.float64)
    if horizons.ndim != 1 or horizons.size == 0:
        raise ValueError("horizons_seconds must be a non-empty one-dimensional sequence")
    if bool(np.any(~np.isfinite(horizons))) or bool(np.any(horizons <= 0.0)):
        raise ValueError("horizons_seconds must contain finite positive values")
    if len(np.unique(horizons)) != len(horizons):
        raise ValueError("horizons_seconds cannot contain duplicates")

    exact_steps = horizons * 1_000_000_000.0 / float(cadence_ns)
    tolerance = np.maximum(1.0, np.abs(exact_steps)) * 1e-12
    sample_steps = np.floor(exact_steps + tolerance).astype(np.int64)
    coverage_steps = np.ceil(exact_steps - tolerance).astype(np.int64)
    if bool(np.any(sample_steps < 1)):
        cadence_seconds = cadence_ns / 1_000_000_000.0
        raise ValueError(f"every horizon must include at least one grid tick (cadence={cadence_seconds:g}s)")
    return horizons, sample_steps, coverage_steps


def _market_kinds(
    venue_names: tuple[str, ...],
    market_kinds: Sequence[MarketKind | str] | None,
) -> tuple[MarketKind, ...]:
    if market_kinds is not None:
        if len(market_kinds) != len(venue_names):
            raise ValueError("market_kinds must align with venue_names")
        return tuple(MarketKind.parse(value) for value in market_kinds)
    inferred: list[MarketKind] = []
    for venue in venue_names:
        key = venue.strip().lower()
        if key not in _KNOWN_MARKETS:
            raise ValueError(f"market kind is required for unknown venue {venue!r}")
        inferred.append(_KNOWN_MARKETS[key])
    return tuple(inferred)


def compute_score_cube(
    ts_ns: npt.ArrayLike,
    bid: npt.ArrayLike,
    ask: npt.ArrayLike,
    *,
    horizons_seconds: Sequence[float],
    cost_bps: float = 0.0,
    valid: npt.ArrayLike | None = None,
    segment_id: npt.ArrayLike | None = None,
    venue_names: Sequence[str] | None = None,
    market_kinds: Sequence[MarketKind | str] | None = None,
    cadence_ns: int | None = None,
    min_coverage: float = 1.0,
) -> ScoreCube:
    """Compute the NumPy reference score cube on a regular wall-clock grid.

    For a horizon ``h``, every grid point in ``(i, i+h]`` (forward) or
    ``[i-h, i)`` (backward) contributes to the denominator. Negative edges are
    clipped to zero, not removed. Incomplete, cross-segment, or invalid quote
    windows are represented by ``NaN`` and an explicit false mask.
    """

    timestamps = _as_timestamps_ns(ts_ns)
    cadence = _regular_cadence_ns(timestamps, cadence_ns)
    bids = _as_price_matrix(bid, name="bid")
    asks = _as_price_matrix(ask, name="ask")
    if bids.shape != asks.shape:
        raise ValueError("bid and ask must have identical shapes")
    if bids.shape[0] != len(timestamps):
        raise ValueError("bid/ask must align with timestamps")
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    if not np.isfinite(min_coverage) or not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must lie in (0, 1]")

    raw_names = tuple(f"venue_{i}" for i in range(bids.shape[1])) if venue_names is None else venue_names
    names = tuple(str(name) for name in raw_names)
    if len(names) != bids.shape[1] or len(set(names)) != len(names):
        raise ValueError("venue_names must be unique and align with the price matrices")
    if not names:
        raise ValueError("at least one venue is required")
    kinds = _market_kinds(names, market_kinds)
    horizons, sample_steps, coverage_steps = _horizon_steps(horizons_seconds, cadence)
    quote_valid = _quote_validity(bids, asks, valid)
    segments = _segments(segment_id, len(timestamps))

    n, venue_count = bids.shape
    shape = (n, len(horizons), venue_count)
    forward_long = np.full(shape, np.nan, dtype=np.float64)
    forward_short = np.full(shape, np.nan, dtype=np.float64)
    backward_long = np.full(shape, np.nan, dtype=np.float64)
    backward_short = np.full(shape, np.nan, dtype=np.float64)
    forward_valid = np.zeros(shape, dtype=np.bool_)
    backward_valid = np.zeros(shape, dtype=np.bool_)

    forward_long_sum = np.zeros((n, venue_count), dtype=np.float64)
    forward_short_sum = np.zeros((n, venue_count), dtype=np.float64)
    backward_long_sum = np.zeros((n, venue_count), dtype=np.float64)
    backward_short_sum = np.zeros((n, venue_count), dtype=np.float64)
    forward_good = np.zeros((n, venue_count), dtype=np.int32)
    backward_good = np.zeros((n, venue_count), dtype=np.int32)

    horizons_at_step: dict[int, list[int]] = {}
    for horizon_index, step in enumerate(sample_steps):
        horizons_at_step.setdefault(int(step), []).append(horizon_index)

    max_step = min(int(sample_steps.max()), max(0, n - 1))
    for lag in range(1, max_step + 1):
        pair_count = n - lag
        past_valid = quote_valid[:pair_count]
        future_valid = quote_valid[lag:]
        same_segment = (segments[:pair_count] == segments[lag:])[:, None]
        pair_valid = past_valid & future_valid & same_segment

        long_ratio = np.ones((pair_count, venue_count), dtype=np.float64)
        short_ratio = np.ones((pair_count, venue_count), dtype=np.float64)
        np.divide(bids[lag:], asks[:pair_count], out=long_ratio, where=pair_valid)
        np.divide(bids[:pair_count], asks[lag:], out=short_ratio, where=pair_valid)
        long_edge = np.maximum(BPS * (long_ratio - 1.0) - float(cost_bps), 0.0)
        short_edge = np.maximum(BPS * (short_ratio - 1.0) - float(cost_bps), 0.0)
        long_edge[~pair_valid] = 0.0
        short_edge[~pair_valid] = 0.0

        forward_long_sum[:pair_count] += long_edge
        forward_short_sum[:pair_count] += short_edge
        backward_long_sum[lag:] += long_edge
        backward_short_sum[lag:] += short_edge
        forward_good[:pair_count] += pair_valid
        backward_good[lag:] += pair_valid

        for horizon_index in horizons_at_step.get(lag, ()):
            coverage = int(coverage_steps[horizon_index])
            if coverage >= n:
                continue
            full_count = n - coverage
            required_pairs = max(1, int(np.ceil(float(min_coverage) * lag)))

            forward_slice = slice(0, full_count)
            forward_segment_ok = (segments[:full_count] == segments[coverage : coverage + full_count])[:, None]
            f_count = forward_good[forward_slice]
            f_valid = (f_count >= required_pairs) & forward_segment_ok
            forward_valid[forward_slice, horizon_index, :] = f_valid
            forward_long[forward_slice, horizon_index, :] = np.where(
                f_valid,
                forward_long_sum[forward_slice] / np.maximum(f_count, 1),
                np.nan,
            )
            forward_short[forward_slice, horizon_index, :] = np.where(
                f_valid,
                forward_short_sum[forward_slice] / np.maximum(f_count, 1),
                np.nan,
            )

            backward_slice = slice(coverage, n)
            backward_segment_ok = (segments[coverage:] == segments[:full_count])[:, None]
            b_count = backward_good[backward_slice]
            b_valid = (b_count >= required_pairs) & backward_segment_ok
            backward_valid[backward_slice, horizon_index, :] = b_valid
            backward_long[backward_slice, horizon_index, :] = np.where(
                b_valid,
                backward_long_sum[backward_slice] / np.maximum(b_count, 1),
                np.nan,
            )
            backward_short[backward_slice, horizon_index, :] = np.where(
                b_valid,
                backward_short_sum[backward_slice] / np.maximum(b_count, 1),
                np.nan,
            )

    return ScoreCube(
        ts_ns=timestamps,
        horizons_seconds=horizons,
        venue_names=names,
        market_kinds=kinds,
        forward_long=forward_long,
        forward_short=forward_short,
        backward_long=backward_long,
        backward_short=backward_short,
        forward_valid=forward_valid,
        backward_valid=backward_valid,
        cost_bps=float(cost_bps),
        cadence_ns=cadence,
    )


def legacy_horizons_seconds() -> npt.NDArray[np.float64]:
    """Return the legacy one-through-sixty second horizon pack."""

    return np.asarray(LEGACY_HORIZONS_SECONDS, dtype=np.float64)


def legacy_300ms_horizon_steps(
    horizons_seconds: npt.ArrayLike | None = None,
) -> npt.NDArray[np.int64]:
    """Map wall-clock horizons to the old 300 ms token offsets."""

    horizons = legacy_horizons_seconds() if horizons_seconds is None else np.asarray(horizons_seconds, dtype=np.float64)
    if horizons.ndim != 1 or bool(np.any(~np.isfinite(horizons))) or bool(np.any(horizons <= 0.0)):
        raise ValueError("horizons_seconds must contain finite positive values")
    return np.maximum(1, np.rint(horizons / LEGACY_CADENCE_SECONDS).astype(np.int64))


def legacy_300ms_token_horizons() -> npt.NDArray[np.int64]:
    """Return literal legacy token horizons 1..60 for parity fixtures."""

    return np.asarray(LEGACY_300MS_TOKEN_HORIZONS, dtype=np.int64)


def compute_legacy_score_cube(
    ts_ns: npt.ArrayLike,
    bid: npt.ArrayLike,
    ask: npt.ArrayLike,
    *,
    horizons_seconds: Sequence[float] = LEGACY_HORIZONS_SECONDS,
    cadence_ns: int = 300_000_000,
    **kwargs: object,
) -> ScoreCube:
    """Compute scores at the old rounded 300 ms token offsets.

    The returned horizon metadata remains in requested wall-clock seconds; only
    the score windows are quantized for numerical parity with the legacy
    builder. Incomplete windows remain censored instead of using a shortened
    denominator.
    """

    timestamps = _as_timestamps_ns(ts_ns)
    cadence = _regular_cadence_ns(timestamps, cadence_ns)
    if cadence != 300_000_000:
        raise ValueError("legacy score mode requires a 300 ms grid")
    requested = np.asarray(horizons_seconds, dtype=np.float64)
    steps = legacy_300ms_horizon_steps(requested)
    effective_horizons = steps.astype(np.float64) * LEGACY_CADENCE_SECONDS
    cube = compute_score_cube(
        timestamps,
        bid,
        ask,
        horizons_seconds=effective_horizons,
        cadence_ns=cadence,
        **kwargs,
    )
    return replace(cube, horizons_seconds=requested)


def legacy_horizon_weights(
    horizons_seconds: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Legacy normalized Gaussian-plus-floor horizon weights."""

    horizons = legacy_horizons_seconds() if horizons_seconds is None else np.asarray(horizons_seconds, dtype=np.float64)
    if horizons.ndim != 1 or horizons.size == 0:
        raise ValueError("horizons_seconds must be a non-empty one-dimensional sequence")
    if bool(np.any(~np.isfinite(horizons))) or bool(np.any(horizons <= 0.0)):
        raise ValueError("horizons_seconds must contain finite positive values")
    weights = 0.15 + np.exp(-0.5 * ((horizons - 22.5) / 8.0) ** 2)
    return weights / weights.sum(dtype=np.float64)


def _validated_weights(weights: npt.ArrayLike, width: int) -> npt.NDArray[np.float64]:
    output = np.asarray(weights, dtype=np.float64)
    if output.ndim != 1 or output.shape[0] != width:
        raise ValueError("weights must be one-dimensional and match the row width")
    if bool(np.any(~np.isfinite(output))) or bool(np.any(output < 0.0)) or float(output.sum()) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and have positive total weight")
    return output


def _value_mask(values: npt.NDArray[np.float64], mask: npt.ArrayLike | None) -> npt.NDArray[np.bool_]:
    finite = np.isfinite(values)
    if mask is None:
        return finite
    supplied = np.asarray(mask, dtype=np.bool_)
    try:
        supplied = np.broadcast_to(supplied, values.shape)
    except ValueError as exc:
        raise ValueError("mask must be broadcastable to values") from exc
    return finite & supplied


def weighted_row_median(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Weighted median along the final axis, renormalizing valid row weights."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("values must have at least one dimension")
    row_weights = _validated_weights(weights, array.shape[-1])
    valid = _value_mask(array, mask)
    flat_values = array.reshape(-1, array.shape[-1])
    flat_valid = valid.reshape(flat_values.shape)
    effective_weights = flat_valid * row_weights[None, :]
    total = effective_weights.sum(axis=1)

    sortable = np.where(flat_valid, flat_values, np.inf)
    order = np.argsort(sortable, axis=1, kind="stable")
    sorted_values = np.take_along_axis(sortable, order, axis=1)
    sorted_weights = np.take_along_axis(effective_weights, order, axis=1)
    cdf = np.cumsum(sorted_weights, axis=1)
    indices = np.argmax(cdf >= (0.5 * total)[:, None], axis=1)
    result = sorted_values[np.arange(len(flat_values)), indices]
    result[total <= 0.0] = np.nan
    return result.reshape(array.shape[:-1])


def weighted_row_mean(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Weighted mean along the final axis, renormalizing valid row weights."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("values must have at least one dimension")
    row_weights = _validated_weights(weights, array.shape[-1])
    valid = _value_mask(array, mask)
    effective_weights = valid * row_weights
    denominator = effective_weights.sum(axis=-1)
    numerator = np.where(valid, array, 0.0) * row_weights
    result = np.full(array.shape[:-1], np.nan, dtype=np.float64)
    np.divide(numerator.sum(axis=-1), denominator, out=result, where=denominator > 0.0)
    return result


def robust_weighted_aggregate(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Return ``0.7 * weighted median + 0.3 * weighted mean`` per row."""

    median = weighted_row_median(values, weights, mask=mask)
    mean = weighted_row_mean(values, weights, mask=mask)
    return 0.7 * median + 0.3 * mean


def _group_median(
    values: npt.NDArray[np.float64],
    validity: npt.NDArray[np.bool_],
    indices: npt.NDArray[np.int64],
    *,
    require_all_venues: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    group_values = values[..., indices]
    group_members_valid = validity[..., indices]
    if require_all_venues:
        group_valid = np.all(group_members_valid, axis=-1)
    else:
        group_valid = np.any(group_members_valid, axis=-1)
    masked = np.where(group_members_valid, group_values, np.nan)
    if require_all_venues:
        center = np.median(masked, axis=-1)
    else:
        safe = masked.copy()
        safe[..., 0] = np.where(group_valid, safe[..., 0], 0.0)
        center = np.nanmedian(safe, axis=-1)
    center = np.where(group_valid, center, np.nan)
    return center, group_valid


def compute_consensus(
    cube: ScoreCube,
    *,
    require_all_venues: bool = True,
) -> ConsensusScores:
    """Build the robust all-market consensus.

    A median is taken within spot and derivative venues, followed by an equal
    blend of the two groups. By default, one stale venue invalidates its group;
    a single remaining exchange is never used as a silent substitute.
    """

    spot = np.asarray([i for i, kind in enumerate(cube.market_kinds) if kind is MarketKind.SPOT], dtype=np.int64)
    derivative = np.asarray(
        [i for i, kind in enumerate(cube.market_kinds) if kind is MarketKind.DERIVATIVE],
        dtype=np.int64,
    )
    if spot.size == 0 or derivative.size == 0:
        raise ValueError("consensus requires at least one configured spot and derivative venue")

    sf_long, sf_valid = _group_median(
        cube.forward_long, cube.forward_valid, spot, require_all_venues=require_all_venues
    )
    sf_short, _ = _group_median(cube.forward_short, cube.forward_valid, spot, require_all_venues=require_all_venues)
    df_long, df_valid = _group_median(
        cube.forward_long, cube.forward_valid, derivative, require_all_venues=require_all_venues
    )
    df_short, _ = _group_median(
        cube.forward_short, cube.forward_valid, derivative, require_all_venues=require_all_venues
    )
    sb_long, sb_valid = _group_median(
        cube.backward_long, cube.backward_valid, spot, require_all_venues=require_all_venues
    )
    sb_short, _ = _group_median(cube.backward_short, cube.backward_valid, spot, require_all_venues=require_all_venues)
    db_long, db_valid = _group_median(
        cube.backward_long, cube.backward_valid, derivative, require_all_venues=require_all_venues
    )
    db_short, _ = _group_median(
        cube.backward_short, cube.backward_valid, derivative, require_all_venues=require_all_venues
    )

    forward_valid = sf_valid & df_valid
    backward_valid = sb_valid & db_valid
    forward_long = np.where(forward_valid, 0.5 * (sf_long + df_long), np.nan)
    forward_short = np.where(forward_valid, 0.5 * (sf_short + df_short), np.nan)
    backward_long = np.where(backward_valid, 0.5 * (sb_long + db_long), np.nan)
    backward_short = np.where(backward_valid, 0.5 * (sb_short + db_short), np.nan)
    return ConsensusScores(
        ts_ns=cube.ts_ns,
        horizons_seconds=cube.horizons_seconds,
        forward_long=forward_long,
        forward_short=forward_short,
        backward_long=backward_long,
        backward_short=backward_short,
        forward_valid=forward_valid,
        backward_valid=backward_valid,
        spot_forward_valid=sf_valid,
        derivative_forward_valid=df_valid,
        spot_backward_valid=sb_valid,
        derivative_backward_valid=db_valid,
    )


def aggregate_horizons(
    scores: ConsensusScores,
    *,
    weights: npt.ArrayLike | None = None,
    require_all_horizons: bool = True,
) -> AggregatedScores:
    """Collapse the horizon axis with the legacy robust weighted reducer."""

    horizon_weights = (
        legacy_horizon_weights(scores.horizons_seconds)
        if weights is None
        else _validated_weights(weights, len(scores.horizons_seconds))
    )
    positive_weight = horizon_weights > 0.0
    if require_all_horizons:
        forward_valid = np.all(scores.forward_valid[..., positive_weight], axis=-1)
        backward_valid = np.all(scores.backward_valid[..., positive_weight], axis=-1)
    else:
        forward_valid = np.any(scores.forward_valid[..., positive_weight], axis=-1)
        backward_valid = np.any(scores.backward_valid[..., positive_weight], axis=-1)

    def reduce(values: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_], row_valid: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
        result = robust_weighted_aggregate(values, horizon_weights, mask=mask)
        return np.where(row_valid, result, np.nan)

    return AggregatedScores(
        ts_ns=scores.ts_ns,
        forward_long=reduce(scores.forward_long, scores.forward_valid, forward_valid),
        forward_short=reduce(scores.forward_short, scores.forward_valid, forward_valid),
        backward_long=reduce(scores.backward_long, scores.backward_valid, backward_valid),
        backward_short=reduce(scores.backward_short, scores.backward_valid, backward_valid),
        forward_valid=forward_valid,
        backward_valid=backward_valid,
        horizon_weights=np.asarray(horizon_weights, dtype=np.float64),
    )


def build_score_bundle(
    ts_ns: npt.ArrayLike,
    bid: npt.ArrayLike,
    ask: npt.ArrayLike,
    *,
    horizons_seconds: Sequence[float],
    cost_bps: float = 0.0,
    valid: npt.ArrayLike | None = None,
    segment_id: npt.ArrayLike | None = None,
    venue_names: Sequence[str] | None = None,
    market_kinds: Sequence[MarketKind | str] | None = None,
    cadence_ns: int | None = None,
    require_all_venues: bool = True,
    require_all_horizons: bool = True,
) -> ScoreBundle:
    cube = compute_score_cube(
        ts_ns,
        bid,
        ask,
        horizons_seconds=horizons_seconds,
        cost_bps=cost_bps,
        valid=valid,
        segment_id=segment_id,
        venue_names=venue_names,
        market_kinds=market_kinds,
        cadence_ns=cadence_ns,
    )
    consensus = compute_consensus(cube, require_all_venues=require_all_venues)
    aggregate = aggregate_horizons(consensus, require_all_horizons=require_all_horizons)
    return ScoreBundle(cube=cube, consensus=consensus, aggregate=aggregate)


def build_cost_scenarios(
    ts_ns: npt.ArrayLike,
    bid: npt.ArrayLike,
    ask: npt.ArrayLike,
    *,
    cost_bps: Iterable[float],
    **kwargs: object,
) -> dict[float, ScoreBundle]:
    """Compute independent bundles for each transaction-cost scenario."""

    output: dict[float, ScoreBundle] = {}
    for cost in cost_bps:
        key = float(cost)
        if key in output:
            raise ValueError(f"duplicate cost scenario: {key}")
        output[key] = build_score_bundle(ts_ns, bid, ask, cost_bps=key, **kwargs)
    if not output:
        raise ValueError("at least one cost scenario is required")
    return output


# Descriptive aliases retained at the package boundary for callers ported from
# exploratory score builders.
compute_per_venue_score_cube = compute_score_cube
compute_legacy_scores = compute_legacy_score_cube
compute_robust_consensus = compute_consensus
aggregate_score_horizons = aggregate_horizons


__all__ = [
    "BPS",
    "LEGACY_CADENCE_SECONDS",
    "LEGACY_300MS_HORIZON_STEPS",
    "LEGACY_300MS_TOKEN_HORIZONS",
    "LEGACY_HORIZON_COUNT",
    "LEGACY_HORIZONS_SECONDS",
    "aggregate_horizons",
    "aggregate_score_horizons",
    "build_cost_scenarios",
    "build_score_bundle",
    "compute_consensus",
    "compute_legacy_score_cube",
    "compute_legacy_scores",
    "compute_per_venue_score_cube",
    "compute_robust_consensus",
    "compute_score_cube",
    "legacy_300ms_horizon_steps",
    "legacy_300ms_token_horizons",
    "legacy_horizon_weights",
    "legacy_horizons_seconds",
    "robust_weighted_aggregate",
    "weighted_row_mean",
    "weighted_row_median",
]
