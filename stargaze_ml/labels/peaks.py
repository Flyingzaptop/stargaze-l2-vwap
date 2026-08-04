from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeakEvent:
    event_id: int
    start: int
    end: int
    peak: int
    zone_start: int
    zone_end: int
    peak_score: float


@dataclass
class PeakZones:
    zone: np.ndarray
    event_id: np.ndarray
    peak_idx: np.ndarray
    peak_score: np.ndarray
    ratio: np.ndarray
    events: tuple[PeakEvent, ...]


def build_peak_zones(
    score: np.ndarray,
    valid: np.ndarray,
    segment_id: np.ndarray,
    *,
    high: float,
    low: float,
    min_ratio: float = 0.75,
    nms_ticks: int = 0,
) -> PeakZones:
    score = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.asarray(valid, dtype=bool)
    segment_id = np.asarray(segment_id)
    n = len(score)
    zone = np.zeros(n, dtype=bool)
    event_id = np.full(n, -1, dtype=np.int64)
    peak_idx = np.full(n, -1, dtype=np.int64)
    peak_score = np.zeros(n, dtype=np.float32)
    ratio = np.zeros(n, dtype=np.float32)
    events: list[PeakEvent] = []
    if int(nms_ticks) > 0:
        return _build_nms_peak_zones(
            score,
            valid,
            segment_id,
            high=float(high),
            low=float(low),
            min_ratio=float(min_ratio),
            nms_ticks=int(nms_ticks),
        )
    i = 0
    next_event = 0
    while i < n:
        if not valid[i] or score[i] <= float(low):
            i += 1
            continue
        segment = segment_id[i]
        start = i
        while i < n and valid[i] and segment_id[i] == segment and score[i] > float(low):
            i += 1
        end = i
        local = score[start:end]
        if local.size == 0 or float(np.max(local)) < float(high):
            continue
        peak = start + int(np.argmax(local))
        peak_value = float(score[peak])
        local_ratio = local / max(peak_value, 1e-12)
        core = np.flatnonzero(local_ratio >= float(min_ratio))
        if core.size == 0:
            continue
        zone_start = start + int(core[0])
        zone_end = start + int(core[-1]) + 1
        event_id[start:end] = next_event
        peak_idx[start:end] = peak
        peak_score[start:end] = peak_value
        ratio[start:end] = local_ratio.astype(np.float32)
        zone[zone_start:zone_end] = True
        events.append(PeakEvent(next_event, start, end, peak, zone_start, zone_end, peak_value))
        next_event += 1
    return PeakZones(zone, event_id, peak_idx, peak_score, ratio, tuple(events))


def _build_nms_peak_zones(
    score: np.ndarray,
    valid: np.ndarray,
    segment_id: np.ndarray,
    *,
    high: float,
    low: float,
    min_ratio: float,
    nms_ticks: int,
) -> PeakZones:
    n = len(score)
    zone = np.zeros(n, dtype=bool)
    event_id = np.full(n, -1, dtype=np.int64)
    peak_idx = np.full(n, -1, dtype=np.int64)
    peak_score = np.zeros(n, dtype=np.float32)
    ratio = np.zeros(n, dtype=np.float32)
    events: list[PeakEvent] = []
    segment_starts = np.flatnonzero(np.r_[True, (segment_id[1:] != segment_id[:-1]) | (~valid[1:]) | (~valid[:-1])])
    segment_ends = np.r_[segment_starts[1:], n]
    next_event = 0
    for seg_start, seg_end in zip(segment_starts, segment_ends, strict=True):
        if not valid[seg_start]:
            continue
        local = score[seg_start:seg_end]
        if len(local) < 3:
            continue
        candidates = np.flatnonzero((local[1:-1] >= high) & (local[1:-1] >= local[:-2]) & (local[1:-1] > local[2:])) + seg_start + 1
        selected: list[int] = []
        for candidate in sorted((int(x) for x in candidates), key=lambda idx: float(score[idx]), reverse=True):
            if any(abs(candidate - kept) <= nms_ticks for kept in selected):
                continue
            selected.append(candidate)
        selected.sort()
        for position, peak in enumerate(selected):
            left_limit = seg_start if position == 0 else (selected[position - 1] + peak) // 2 + 1
            right_limit = seg_end if position + 1 == len(selected) else (peak + selected[position + 1]) // 2 + 1
            start = peak
            while start > left_limit and score[start - 1] > low:
                start -= 1
            end = peak + 1
            while end < right_limit and score[end] > low:
                end += 1
            peak_value = float(score[peak])
            core_threshold = peak_value * min_ratio
            zone_start = peak
            while zone_start > start and score[zone_start - 1] >= core_threshold:
                zone_start -= 1
            zone_end = peak + 1
            while zone_end < end and score[zone_end] >= core_threshold:
                zone_end += 1
            event_id[start:end] = next_event
            peak_idx[start:end] = peak
            peak_score[start:end] = peak_value
            ratio[start:end] = (score[start:end] / max(peak_value, 1e-12)).astype(np.float32)
            zone[zone_start:zone_end] = True
            events.append(PeakEvent(next_event, start, end, peak, zone_start, zone_end, peak_value))
            next_event += 1
    return PeakZones(zone, event_id, peak_idx, peak_score, ratio, tuple(events))


def local_peak_indices(score: np.ndarray, start: int, end: int) -> np.ndarray:
    if end - start < 3:
        return np.empty(0, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    local = values[start:end]
    peaks = np.flatnonzero((local[1:-1] >= local[:-2]) & (local[1:-1] > local[2:])) + start + 1
    return peaks.astype(np.int64, copy=False)
