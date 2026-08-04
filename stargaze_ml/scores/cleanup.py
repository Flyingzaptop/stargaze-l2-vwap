from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from .core import _as_timestamps_ns, _regular_cadence_ns, _segments
from .types import CleanedForwardScores, ForwardCleanupConfig


def _aligned_values(values: npt.ArrayLike, length: int, *, name: str) -> npt.NDArray[np.float64]:
    output = np.asarray(values, dtype=np.float64)
    if output.ndim != 1 or output.shape != (length,):
        raise ValueError(f"{name} must be one-dimensional and align with timestamps")
    return output


def _aligned_valid(valid: npt.ArrayLike | None, length: int) -> npt.NDArray[np.bool_]:
    if valid is None:
        return np.ones(length, dtype=np.bool_)
    output = np.asarray(valid, dtype=np.bool_)
    if output.ndim != 1 or output.shape != (length,):
        raise ValueError("valid must be one-dimensional and align with timestamps")
    return output


def _contiguous_runs(segments: npt.NDArray, valid: npt.NDArray[np.bool_]) -> Iterator[tuple[int, int]]:
    start = 0
    n = len(valid)
    while start < n:
        if not valid[start]:
            start += 1
            continue
        end = start + 1
        while end < n and valid[end] and segments[end] == segments[start]:
            end += 1
        yield start, end
        start = end


def _window_points(window_seconds: float, cadence_ns: int) -> int:
    if not np.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("window_seconds must be finite and positive")
    cadence_seconds = cadence_ns / 1_000_000_000.0
    return max(1, int(round(float(window_seconds) / cadence_seconds)))


def _rolling_median_run(values: npt.NDArray[np.float64], window_points: int) -> npt.NDArray[np.float64]:
    left = window_points // 2
    right = window_points - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_points)
    return np.median(windows, axis=-1)


def rolling_median_time(
    ts_ns: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    window_seconds: float = 2.1,
    valid: npt.ArrayLike | None = None,
    segment_id: npt.ArrayLike | None = None,
    cadence_ns: int | None = None,
) -> npt.NDArray[np.float64]:
    """Centered rolling median expressed in seconds, with edge replication."""

    timestamps = _as_timestamps_ns(ts_ns)
    cadence = _regular_cadence_ns(timestamps, cadence_ns)
    source = _aligned_values(values, len(timestamps), name="values")
    row_valid = _aligned_valid(valid, len(timestamps)) & np.isfinite(source)
    segments = _segments(segment_id, len(timestamps))
    output = np.full(len(timestamps), np.nan, dtype=np.float64)
    points = _window_points(window_seconds, cadence)
    for start, end in _contiguous_runs(segments, row_valid):
        output[start:end] = _rolling_median_run(source[start:end], points)
    return output


def _gaussian_kernel(cadence_ns: int, sigma_seconds: float, truncate: float) -> npt.NDArray[np.float64]:
    if not np.isfinite(sigma_seconds) or sigma_seconds <= 0.0:
        raise ValueError("sigma_seconds must be finite and positive")
    if not np.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("truncate must be finite and positive")
    cadence_seconds = cadence_ns / 1_000_000_000.0
    radius = max(1, int(round(truncate * sigma_seconds / cadence_seconds)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64) * cadence_seconds
    kernel = np.exp(-0.5 * (offsets / float(sigma_seconds)) ** 2)
    return kernel / kernel.sum(dtype=np.float64)


def gaussian_smooth_time(
    ts_ns: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    sigma_seconds: float = 1.5,
    truncate: float = 4.0,
    valid: npt.ArrayLike | None = None,
    segment_id: npt.ArrayLike | None = None,
    cadence_ns: int | None = None,
) -> npt.NDArray[np.float64]:
    """Centered Gaussian smoothing in wall-clock units, without crossing gaps."""

    timestamps = _as_timestamps_ns(ts_ns)
    cadence = _regular_cadence_ns(timestamps, cadence_ns)
    source = _aligned_values(values, len(timestamps), name="values")
    row_valid = _aligned_valid(valid, len(timestamps)) & np.isfinite(source)
    segments = _segments(segment_id, len(timestamps))
    output = np.full(len(timestamps), np.nan, dtype=np.float64)
    kernel = _gaussian_kernel(cadence, sigma_seconds, truncate)
    radius = len(kernel) // 2
    for start, end in _contiguous_runs(segments, row_valid):
        padded = np.pad(source[start:end], (radius, radius), mode="edge")
        output[start:end] = np.convolve(padded, kernel, mode="valid")
    return output


def remove_isolated_humps_time(
    ts_ns: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    peak_min_bps: float,
    width_min_seconds: float = 1.5,
    area_min_bps_seconds: float = 0.35,
    epsilon_bps: float = 0.025,
    valid: npt.ArrayLike | None = None,
    segment_id: npt.ArrayLike | None = None,
    cadence_ns: int | None = None,
) -> npt.NDArray[np.float64]:
    """Zero positive lobes that fail peak, duration, or area requirements."""

    timestamps = _as_timestamps_ns(ts_ns)
    cadence = _regular_cadence_ns(timestamps, cadence_ns)
    source = _aligned_values(values, len(timestamps), name="values")
    row_valid = _aligned_valid(valid, len(timestamps)) & np.isfinite(source)
    segments = _segments(segment_id, len(timestamps))
    parameters = (peak_min_bps, width_min_seconds, area_min_bps_seconds, epsilon_bps)
    if any(not np.isfinite(value) or value < 0.0 for value in parameters):
        raise ValueError("hump cleanup thresholds must be finite and non-negative")

    cadence_seconds = cadence / 1_000_000_000.0
    output = np.where(row_valid, source, np.nan).astype(np.float64, copy=False)
    for run_start, run_end in _contiguous_runs(segments, row_valid):
        cursor = run_start
        while cursor < run_end:
            if output[cursor] <= epsilon_bps:
                cursor += 1
                continue
            left = cursor
            while cursor < run_end and output[cursor] > epsilon_bps:
                cursor += 1
            right = cursor
            lobe = output[left:right]
            peak = float(np.max(lobe))
            width = float(right - left) * cadence_seconds
            area = float(np.sum(lobe, dtype=np.float64)) * cadence_seconds
            if peak < peak_min_bps or width < width_min_seconds or area < area_min_bps_seconds:
                output[left:right] = 0.0
    return output


def _clean_one_side(
    timestamps: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
    row_valid: npt.NDArray[np.bool_],
    segments: npt.NDArray,
    cadence_ns: int,
    config: ForwardCleanupConfig,
    peak_min_bps: float | None,
) -> tuple[npt.NDArray[np.float64], float]:
    median = rolling_median_time(
        timestamps,
        values,
        window_seconds=config.median_window_seconds,
        valid=row_valid,
        segment_id=segments,
        cadence_ns=cadence_ns,
    )
    smooth = gaussian_smooth_time(
        timestamps,
        median,
        sigma_seconds=config.gaussian_sigma_seconds,
        truncate=config.gaussian_truncate,
        valid=row_valid,
        segment_id=segments,
        cadence_ns=cadence_ns,
    )
    configured_peak = config.min_hump_peak_bps if peak_min_bps is None else peak_min_bps
    if configured_peak is None:
        finite = smooth[row_valid & np.isfinite(smooth)]
        configured_peak = (
            config.adaptive_peak_fraction * float(np.quantile(finite, config.adaptive_peak_quantile))
            if finite.size
            else 0.0
        )
    if not np.isfinite(configured_peak) or configured_peak < 0.0:
        raise ValueError("peak minimum must be finite and non-negative")
    clean = remove_isolated_humps_time(
        timestamps,
        smooth,
        peak_min_bps=float(configured_peak),
        width_min_seconds=config.min_hump_width_seconds,
        area_min_bps_seconds=config.min_hump_area_bps_seconds,
        epsilon_bps=config.hump_epsilon_bps,
        valid=row_valid,
        segment_id=segments,
        cadence_ns=cadence_ns,
    )
    return clean, float(configured_peak)


def clean_forward_labels(
    ts_ns: npt.ArrayLike,
    forward_long: npt.ArrayLike,
    forward_short: npt.ArrayLike,
    *,
    valid: npt.ArrayLike | None = None,
    segment_id: npt.ArrayLike | None = None,
    cadence_ns: int | None = None,
    config: ForwardCleanupConfig | None = None,
    peak_min_long_bps: float | None = None,
    peak_min_short_bps: float | None = None,
) -> CleanedForwardScores:
    """Denoise forward oracle labels only.

    The deliberately forward-specific API prevents this non-causal operation
    from being applied to backward scores used as model inputs.
    """

    timestamps = _as_timestamps_ns(ts_ns)
    cadence = _regular_cadence_ns(timestamps, cadence_ns)
    long = _aligned_values(forward_long, len(timestamps), name="forward_long")
    short = _aligned_values(forward_short, len(timestamps), name="forward_short")
    row_valid = _aligned_valid(valid, len(timestamps)) & np.isfinite(long) & np.isfinite(short)
    segments = _segments(segment_id, len(timestamps))
    cleanup_config = config or ForwardCleanupConfig()

    clean_long, used_long = _clean_one_side(
        timestamps,
        long,
        row_valid,
        segments,
        cadence,
        cleanup_config,
        peak_min_long_bps,
    )
    clean_short, used_short = _clean_one_side(
        timestamps,
        short,
        row_valid,
        segments,
        cadence,
        cleanup_config,
        peak_min_short_bps,
    )
    clean_long[~row_valid] = np.nan
    clean_short[~row_valid] = np.nan
    return CleanedForwardScores(
        long=clean_long,
        short=clean_short,
        valid=row_valid,
        peak_min_long_bps=used_long,
        peak_min_short_bps=used_short,
    )


__all__ = [
    "clean_forward_labels",
    "gaussian_smooth_time",
    "remove_isolated_humps_time",
    "rolling_median_time",
]
