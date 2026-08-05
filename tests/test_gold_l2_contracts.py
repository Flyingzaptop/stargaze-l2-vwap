from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from stargaze_ml.gold.l2_contracts import (
    assert_feature_names,
    assert_market_inference_contract,
    assert_normalizer_contract,
    validate_prepared_open_data,
)


def test_feature_contract_reports_first_order_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatch at 1"):
        assert_feature_names(("bid", "vwap", "ask"), ("bid", "ask", "vwap"), artifact="model")


def test_feature_contract_accepts_exact_order() -> None:
    assert_feature_names(("bid", "vwap"), ("bid", "vwap"), artifact="model")


def _prepared(*, train_end: int, validation_end: int) -> SimpleNamespace:
    rows = 4
    return SimpleNamespace(
        x=np.ones((rows, 1), dtype=np.float32),
        feature_names=("x",),
        ts_ns=np.arange(rows, dtype=np.int64),
        segment_id=np.zeros(rows, dtype=np.int64),
        valid_feature=np.ones(rows, dtype=bool),
        observed=np.ones(rows, dtype=bool),
        first_bid=np.ones(rows),
        first_ask=np.ones(rows),
        event_start=np.asarray([], dtype=np.int64),
        event_crossing_1=np.asarray([], dtype=np.int64),
        event_crossing_2=np.asarray([], dtype=np.int64),
        train_end=train_end,
        validation_end=validation_end,
    )


def test_forward_only_prepared_contract_accepts_zero_splits() -> None:
    validate_prepared_open_data(_prepared(train_end=0, validation_end=0))


def test_prepared_contract_rejects_partial_zero_split() -> None:
    with pytest.raises(ValueError, match="split indices"):
        validate_prepared_open_data(_prepared(train_end=0, validation_end=2))


def test_market_inference_contract_ignores_training_schedule() -> None:
    expected = {
        "hidden_size": 96,
        "tick_size": 0.01,
        "commission_per_fill_ticks": 15.0,
        "slippage_per_fill_ticks": 1.0,
        "warmup_epochs": 5,
    }
    actual = {**expected, "warmup_epochs": 15}
    assert_market_inference_contract(expected, actual, artifact="ensemble")
    actual["tick_size"] = 0.1
    with pytest.raises(ValueError, match="tick_size"):
        assert_market_inference_contract(expected, actual, artifact="ensemble")


def test_normalizer_contract_is_exact() -> None:
    assert_normalizer_contract({"median": [1.0]}, {"median": [1.0]}, artifact="model")
    with pytest.raises(ValueError, match="normalizer"):
        assert_normalizer_contract({"median": [1.0]}, {"median": [2.0]}, artifact="model")
