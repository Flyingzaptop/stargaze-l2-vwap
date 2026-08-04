from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedSplits:
    train: np.ndarray
    valid: np.ndarray
    holdout: np.ndarray
    train_end_ns: int
    valid_end_ns: int
    purge_ns: int


def purged_chronological_splits(
    ts_ns: np.ndarray,
    valid: np.ndarray,
    *,
    purge_seconds: float,
    train_fraction: float = 0.60,
    valid_fraction: float = 0.20,
) -> PurgedSplits:
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    n = len(ts_ns)
    if n < 10:
        raise ValueError("at least ten frames are required for chronological splits")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < valid_fraction < 1.0 or train_fraction + valid_fraction >= 1.0:
        raise ValueError("invalid split fractions")
    train_boundary = int(ts_ns[max(1, min(n - 2, int(round(n * train_fraction))))])
    valid_boundary = int(ts_ns[max(2, min(n - 1, int(round(n * (train_fraction + valid_fraction)))))])
    purge_ns = max(0, int(float(purge_seconds) * 1e9))
    train = valid & (ts_ns < train_boundary - purge_ns)
    validation = valid & (ts_ns >= train_boundary + purge_ns) & (ts_ns < valid_boundary - purge_ns)
    holdout = valid & (ts_ns >= valid_boundary + purge_ns)
    if not np.any(train) or not np.any(validation) or not np.any(holdout):
        raise ValueError("purge/split configuration leaves an empty partition")
    return PurgedSplits(train, validation, holdout, train_boundary, valid_boundary, purge_ns)


def purged_blocked_splits(
    ts_ns: np.ndarray,
    valid: np.ndarray,
    *,
    purge_seconds: float,
    holdout_fraction: float = 0.20,
    block_count: int = 8,
    validation_blocks: tuple[int, ...] = (2, 5),
) -> PurgedSplits:
    """Blocked validation across the pre-holdout history.

    The final holdout remains strictly chronological and untouched. Validation
    blocks are removed from training together with a wall-clock purge on both
    sides, so model selection sees multiple regimes without overlapping label
    horizons or contexts.
    """

    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    n = len(ts_ns)
    if n < 100 or not 0.0 < holdout_fraction < 0.5:
        raise ValueError("blocked splits require at least 100 rows and a holdout fraction in (0, 0.5)")
    if block_count < 4 or any(index < 0 or index >= block_count for index in validation_blocks):
        raise ValueError("invalid blocked-validation configuration")
    holdout_start_idx = int(round(n * (1.0 - holdout_fraction)))
    holdout_start_ns = int(ts_ns[holdout_start_idx])
    purge_ns = max(0, int(float(purge_seconds) * 1e9))
    validation = np.zeros(n, dtype=bool)
    train = valid & (ts_ns < holdout_start_ns - purge_ns)
    edges = np.linspace(0, holdout_start_idx, block_count + 1, dtype=np.int64)
    for block in validation_blocks:
        raw_start = int(ts_ns[edges[block]])
        raw_end_idx = min(n - 1, max(edges[block] + 1, edges[block + 1] - 1))
        raw_end = int(ts_ns[raw_end_idx])
        validation |= valid & (ts_ns >= raw_start + purge_ns) & (ts_ns <= raw_end - purge_ns)
        train &= ~((ts_ns >= raw_start - purge_ns) & (ts_ns <= raw_end + purge_ns))
    holdout = valid & (ts_ns >= holdout_start_ns + purge_ns)
    if not np.any(train) or not np.any(validation) or not np.any(holdout):
        raise ValueError("blocked split configuration leaves an empty partition")
    return PurgedSplits(train, validation, holdout, holdout_start_ns, holdout_start_ns, purge_ns)
