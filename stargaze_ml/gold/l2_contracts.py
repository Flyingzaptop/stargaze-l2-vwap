"""Fail-fast contracts for prepared L2 data and checkpoint compatibility."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PreparedLike(Protocol):
    x: np.ndarray
    ts_ns: np.ndarray
    segment_id: np.ndarray
    valid_feature: np.ndarray
    observed: np.ndarray
    first_bid: np.ndarray
    first_ask: np.ndarray
    feature_names: tuple[str, ...]
    event_start: np.ndarray
    event_crossing_1: np.ndarray
    event_crossing_2: np.ndarray
    train_end: int
    validation_end: int


def assert_feature_names(
    expected: tuple[str, ...] | list[str],
    actual: tuple[str, ...] | list[str],
    *,
    artifact: str,
) -> None:
    expected_tuple = tuple(expected); actual_tuple = tuple(actual)
    if expected_tuple == actual_tuple:
        return
    mismatch = next(
        (index for index, pair in enumerate(zip(expected_tuple, actual_tuple)) if pair[0] != pair[1]),
        min(len(expected_tuple), len(actual_tuple)),
    )
    expected_name = expected_tuple[mismatch] if mismatch < len(expected_tuple) else "<missing>"
    actual_name = actual_tuple[mismatch] if mismatch < len(actual_tuple) else "<missing>"
    raise ValueError(
        f"{artifact} feature contract mismatch at {mismatch}: "
        f"checkpoint={expected_name!r}, data={actual_name!r}; "
        f"counts={len(expected_tuple)}/{len(actual_tuple)}"
    )


def assert_market_inference_contract(
    expected: dict[str, object], actual: dict[str, object], *, artifact: str
) -> None:
    """Compare only fields that change model shape or executable PnL."""

    for field in (
        "hidden_size",
        "tick_size",
        "commission_per_fill_ticks",
        "slippage_per_fill_ticks",
    ):
        if expected.get(field) != actual.get(field):
            raise ValueError(
                f"{artifact} market inference contract mismatch for {field}: "
                f"{expected.get(field)!r}/{actual.get(field)!r}"
            )


def assert_normalizer_contract(
    expected: dict[str, object], actual: dict[str, object], *, artifact: str
) -> None:
    if expected != actual:
        raise ValueError(f"{artifact} normalizer contract mismatch")


def validate_prepared_open_data(data: PreparedLike) -> None:
    rows = len(data.x)
    if data.x.ndim != 2 or data.x.shape[1] != len(data.feature_names):
        raise ValueError("prepared feature matrix/name dimensions are inconsistent")
    if len(set(data.feature_names)) != len(data.feature_names):
        raise ValueError("prepared feature names are not unique")
    for name in ("ts_ns", "segment_id", "valid_feature", "observed", "first_bid", "first_ask"):
        if len(getattr(data, name)) != rows:
            raise ValueError(f"prepared row array {name} has wrong length")
    forward_only = data.train_end == 0 and data.validation_end == 0
    historical_split = 0 < data.train_end < data.validation_end < rows
    if not (forward_only or historical_split):
        raise ValueError("prepared chronological split indices are invalid")
    if np.any(np.diff(data.ts_ns.astype(np.int64)) < 0):
        raise ValueError("prepared timestamps are not monotonic")
    if np.any(~np.isfinite(data.x[data.valid_feature])):
        raise ValueError("valid prepared feature rows contain non-finite values")
    executable = data.observed & np.isfinite(data.first_bid) & np.isfinite(data.first_ask)
    if np.any(data.observed & ~executable):
        raise ValueError("observed rows contain non-finite BBO")
    if not (
        len(data.event_start) == len(data.event_crossing_1) == len(data.event_crossing_2)
    ):
        raise ValueError("prepared event arrays have different lengths")
    if len(data.event_start):
        invalid = (
            (data.event_start < 0)
            | (data.event_start >= data.event_crossing_1)
            | (data.event_crossing_1 >= data.event_crossing_2)
            | (data.event_crossing_2 >= rows)
        )
        if np.any(invalid):
            raise ValueError("prepared event boundaries are invalid")
