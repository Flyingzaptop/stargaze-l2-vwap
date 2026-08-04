from __future__ import annotations

import pytest

from stargaze_ml.gold.l2_contracts import assert_feature_names


def test_feature_contract_reports_first_order_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatch at 1"):
        assert_feature_names(("bid", "vwap", "ask"), ("bid", "ask", "vwap"), artifact="model")


def test_feature_contract_accepts_exact_order() -> None:
    assert_feature_names(("bid", "vwap"), ("bid", "vwap"), artifact="model")
