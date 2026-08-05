from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


DAY_NS = 86_400_000_000_000


@dataclass(frozen=True)
class AdaptiveGateResult:
    amplitude_threshold_by_event: np.ndarray
    gate_threshold_by_event: np.ndarray
    gate_index_by_event: np.ndarray
    gated_by_event: np.ndarray
    gate_open: np.ndarray


class _RollingAmplitudeHistogram:
    def __init__(
        self,
        *,
        history_size: int,
        bin_ticks: float,
        max_ticks: float,
        max_active_gap_ns: int = 300_000_000_000,
    ) -> None:
        self.history_size = int(history_size)
        self.bin_ticks = float(bin_ticks)
        self.bin_count = int(np.ceil(max_ticks / bin_ticks)) + 1
        self.tree = np.zeros(self.bin_count + 1, dtype=np.int32)
        self.history: deque[tuple[int, int]] = deque()
        self.active_span_ns = 0
        self.max_active_gap_ns = int(max_active_gap_ns)

    def _update(self, index: int, delta: int) -> None:
        cursor = int(index) + 1
        while cursor < len(self.tree):
            self.tree[cursor] += int(delta)
            cursor += cursor & -cursor

    def _active_gap(self, left: int, right: int) -> int:
        return max(0, min(int(right) - int(left), self.max_active_gap_ns))

    def append(self, value: float, timestamp_ns: int) -> None:
        index = int(np.clip(np.rint(value / self.bin_ticks), 0, self.bin_count - 1))
        if len(self.history) >= self.history_size:
            old_index, old_timestamp = self.history.popleft()
            self._update(old_index, -1)
            if self.history:
                self.active_span_ns -= self._active_gap(
                    old_timestamp, self.history[0][1]
                )
        if self.history:
            self.active_span_ns += self._active_gap(self.history[-1][1], timestamp_ns)
        self.history.append((index, int(timestamp_ns)))
        self._update(index, 1)

    def quantile(self, quantile: float) -> float:
        if not self.history:
            raise ValueError("cannot query an empty amplitude histogram")
        rank = int(np.floor(float(np.clip(quantile, 0.0, 1.0)) * (len(self.history) - 1)))
        index = 0
        bit = 1 << (self.bin_count.bit_length() - 1)
        cumulative = 0
        while bit:
            candidate = index + bit
            if candidate < len(self.tree) and cumulative + int(self.tree[candidate]) <= rank:
                index = candidate
                cumulative += int(self.tree[candidate])
            bit >>= 1
        return min(index, self.bin_count - 1) * self.bin_ticks

    @property
    def events_per_active_day(self) -> float:
        if len(self.history) < 2 or self.active_span_ns <= 0:
            return float(len(self.history))
        return (len(self.history) - 1) / (self.active_span_ns / DAY_NS)

    def __len__(self) -> int:
        return len(self.history)


def causal_adaptive_gate(
    *,
    ts_ns: np.ndarray,
    absolute_delta_ticks: np.ndarray,
    event_start: np.ndarray,
    event_end: np.ndarray,
    completed_amplitude_ticks: np.ndarray,
    target_gated_events_per_active_day: int,
    gate_fraction: float = 0.75,
    history_size: int = 2_000,
    min_history: int = 200,
    fallback_amplitude_ticks: float = 323.3623046875,
    quantile_bin_ticks: float = 1.0,
    max_amplitude_ticks: float = 100_000.0,
    initial_amplitude_ticks: np.ndarray | None = None,
    initial_end_ts_ns: np.ndarray | None = None,
) -> AdaptiveGateResult:
    """Choose each event's gate from prior completed event amplitudes only.

    ``initial_*`` seeds a deployed policy with the frozen tail of events that
    completed before the forward sample began. Those observations are still
    strictly causal and avoid reverting to the fallback gate after a restart.
    """

    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    absolute_delta_ticks = np.asarray(absolute_delta_ticks, dtype=np.float64)
    starts = np.asarray(event_start, dtype=np.int64)
    ends = np.asarray(event_end, dtype=np.int64)
    amplitudes = np.asarray(completed_amplitude_ticks, dtype=np.float64)
    count = len(starts)
    if ends.shape != (count,) or amplitudes.shape != (count,):
        raise ValueError("event arrays must be aligned")
    if np.any(starts < 0) or np.any(ends < starts) or np.any(ends >= len(ts_ns)):
        raise ValueError("event boundaries are invalid")
    if target_gated_events_per_active_day <= 0:
        raise ValueError("target gated events must be positive")
    if not 0 < gate_fraction <= 1:
        raise ValueError("gate_fraction must be in (0, 1]")
    if history_size <= 0 or min_history <= 0 or min_history > history_size:
        raise ValueError("adaptive history limits are invalid")
    if fallback_amplitude_ticks <= 0:
        raise ValueError("fallback amplitude must be positive")
    if quantile_bin_ticks <= 0 or max_amplitude_ticks <= quantile_bin_ticks:
        raise ValueError("adaptive amplitude histogram limits are invalid")

    history = _RollingAmplitudeHistogram(
        history_size=history_size,
        bin_ticks=quantile_bin_ticks,
        max_ticks=max_amplitude_ticks,
    )
    if (initial_amplitude_ticks is None) != (initial_end_ts_ns is None):
        raise ValueError("adaptive initial amplitudes and timestamps must be provided together")
    if initial_amplitude_ticks is not None:
        initial_amplitudes = np.asarray(initial_amplitude_ticks, dtype=np.float64)
        initial_timestamps = np.asarray(initial_end_ts_ns, dtype=np.int64)
        if initial_amplitudes.ndim != 1 or initial_timestamps.shape != initial_amplitudes.shape:
            raise ValueError("adaptive initial history arrays must be aligned vectors")
        if not np.all(np.isfinite(initial_amplitudes)) or np.any(initial_amplitudes < 0):
            raise ValueError("adaptive initial amplitudes must be finite and non-negative")
        if np.any(np.diff(initial_timestamps) < 0):
            raise ValueError("adaptive initial timestamps must be monotonic")
        if count and len(initial_timestamps) and initial_timestamps[-1] >= ts_ns[starts[0]]:
            raise ValueError("adaptive initial history must predate the forward sample")
        for value, timestamp_ns in zip(initial_amplitudes, initial_timestamps, strict=True):
            history.append(float(value), int(timestamp_ns))
    amplitude_threshold = np.empty(count, dtype=np.float64)
    gate_threshold = np.empty(count, dtype=np.float64)
    gate_index = np.full(count, -1, dtype=np.int64)
    gated = np.zeros(count, dtype=bool)
    gate_open = np.zeros(len(ts_ns), dtype=bool)

    for event in range(count):
        if len(history) >= min_history:
            events_per_day = history.events_per_active_day
            quantile = float(
                np.clip(
                    1.0 - target_gated_events_per_active_day / events_per_day,
                    0.0,
                    1.0,
                )
            )
            gate = history.quantile(quantile)
        else:
            gate = float(fallback_amplitude_ticks) * gate_fraction
        threshold = gate / gate_fraction
        amplitude_threshold[event] = threshold
        gate_threshold[event] = gate
        left = int(starts[event])
        right = int(ends[event]) + 1
        hits = np.flatnonzero(absolute_delta_ticks[left:right] >= gate)
        if hits.size:
            index = left + int(hits[0])
            gate_index[event] = index
            gated[event] = True
            gate_open[index:right] = True
        history.append(float(amplitudes[event]), int(ts_ns[ends[event]]))
    return AdaptiveGateResult(
        amplitude_threshold_by_event=amplitude_threshold,
        gate_threshold_by_event=gate_threshold,
        gate_index_by_event=gate_index,
        gated_by_event=gated,
        gate_open=gate_open,
    )
