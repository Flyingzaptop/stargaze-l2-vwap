from __future__ import annotations

import numpy as np

from stargaze_ml.gold.l2_adaptive_gate import DAY_NS, causal_adaptive_gate


def test_adaptive_gate_uses_only_prior_completed_events() -> None:
    events = 6
    starts = np.arange(events) * 3
    ends = starts + 2
    ts = np.arange(events * 3, dtype=np.int64) * DAY_NS
    delta = np.tile([0.0, 5.0, 10.0], events)
    amplitudes = np.full(events, 10.0)
    baseline = causal_adaptive_gate(
        ts_ns=ts,
        absolute_delta_ticks=delta,
        event_start=starts,
        event_end=ends,
        completed_amplitude_ticks=amplitudes,
        target_gated_events_per_active_day=1,
        gate_fraction=1.0,
        history_size=10,
        min_history=2,
        fallback_amplitude_ticks=20.0,
    )
    changed = amplitudes.copy()
    changed[-1] = 10_000.0
    perturbed = causal_adaptive_gate(
        ts_ns=ts,
        absolute_delta_ticks=delta,
        event_start=starts,
        event_end=ends,
        completed_amplitude_ticks=changed,
        target_gated_events_per_active_day=1,
        gate_fraction=1.0,
        history_size=10,
        min_history=2,
        fallback_amplitude_ticks=20.0,
    )
    np.testing.assert_allclose(
        baseline.amplitude_threshold_by_event,
        perturbed.amplitude_threshold_by_event,
    )


def test_adaptive_gate_marks_first_threshold_hit() -> None:
    result = causal_adaptive_gate(
        ts_ns=np.arange(6, dtype=np.int64) * DAY_NS,
        absolute_delta_ticks=np.asarray([0.0, 3.0, 6.0, 0.0, 5.0, 8.0]),
        event_start=np.asarray([0, 3]),
        event_end=np.asarray([2, 5]),
        completed_amplitude_ticks=np.asarray([6.0, 8.0]),
        target_gated_events_per_active_day=1,
        gate_fraction=0.5,
        min_history=2,
        fallback_amplitude_ticks=10.0,
    )
    np.testing.assert_array_equal(result.gate_index_by_event, [2, 4])
    np.testing.assert_array_equal(result.gate_open, [False, False, True, False, True, True])
