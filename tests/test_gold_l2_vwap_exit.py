from __future__ import annotations

import numpy as np

from stargaze_ml.gold.l2_reinforce import PreparedPolicyData, ReinforceConfig
from stargaze_ml.gold.l2_vwap_exit import VwapCrossMarket, ledger_for_vwap_cross_segment


def _data() -> PreparedPolicyData:
    rows = 12
    ts = 1_700_000_000_000_000_000 + np.arange(rows, dtype=np.int64) * 1_000_000_000
    bid = np.asarray([99, 99, 100, 101, 100, 99, 98, 98, 98, 98, 98, 98], dtype=np.float64)
    return PreparedPolicyData(
        ts_ns=ts,
        segment_id=np.zeros(rows, dtype=np.int32),
        x=np.zeros((rows, 1), dtype=np.float32),
        feature_names=("x",),
        valid_feature=np.ones(rows, dtype=bool),
        observed=np.ones(rows, dtype=bool),
        first_bid=bid,
        first_ask=bid + 0.2,
        train_end=6,
        validation_end=9,
    )


def test_first_and_second_vwap_crossings_execute_on_next_second_bbo() -> None:
    data = _data()
    market = VwapCrossMarket(
        last_bid=data.first_bid.copy(),
        last_ask=data.first_ask.copy(),
        bid_vwap_60s=np.full(len(data), 100.0),
        ask_vwap_60s=np.full(len(data), 100.2),
    )
    hazards = np.full((len(data) - 2, 4), 0.001, dtype=np.float64)
    hazards[0, 0] = 0.03
    config = ReinforceConfig(tick_size=0.1, commission_per_fill_ticks=0, slippage_per_fill_ticks=0)

    first = ledger_for_vwap_cross_segment(
        hazards,
        data,
        market,
        0,
        len(data),
        config,
        crossing_number=1,
        event_hazard_threshold=0.02,
    )
    second = ledger_for_vwap_cross_segment(
        hazards,
        data,
        market,
        0,
        len(data),
        config,
        crossing_number=2,
        event_hazard_threshold=0.02,
    )

    assert len(first) == len(second) == 1
    assert first[0]["entry_index"] == second[0]["entry_index"] == 1
    # Bid touches VWAP at t=2, crosses above at t=3, then executes at t=4.
    assert first[0]["exit_index"] == 4
    assert first[0]["exit_reason"] == "vwap_cross_1"
    # It touches again at t=4, crosses below at t=5, then executes at t=6.
    assert second[0]["exit_index"] == 6
    assert second[0]["exit_reason"] == "vwap_cross_2"
    assert not first[0]["terminal"] and not second[0]["terminal"]


def test_cross_exit_waits_for_next_available_execution_bbo() -> None:
    data = _data()
    data.observed[4] = False
    market = VwapCrossMarket(
        last_bid=data.first_bid.copy(),
        last_ask=data.first_ask.copy(),
        bid_vwap_60s=np.full(len(data), 100.0),
        ask_vwap_60s=np.full(len(data), 100.2),
    )
    hazards = np.full((len(data) - 2, 4), 0.001, dtype=np.float64)
    hazards[0, 0] = 0.03
    records = ledger_for_vwap_cross_segment(
        hazards,
        data,
        market,
        0,
        len(data),
        ReinforceConfig(tick_size=0.1, commission_per_fill_ticks=0, slippage_per_fill_ticks=0),
        crossing_number=1,
        event_hazard_threshold=0.02,
    )
    assert records[0]["exit_index"] == 5
    assert records[0]["exit_reason"] == "vwap_cross_1"
