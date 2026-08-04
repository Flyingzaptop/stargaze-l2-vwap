from __future__ import annotations

import numpy as np

from stargaze_ml.gold.l2_open_reinforce import OpenReinforceConfig, PreparedOpenData
from stargaze_ml.gold.l2_profit_direction import executable_side_pnls


def test_executable_side_pnls_include_bbo_and_costs() -> None:
    data = object.__new__(PreparedOpenData)
    data.first_bid = np.asarray([100.0, 101.0, 106.0, 108.0])
    data.first_ask = np.asarray([102.0, 103.0, 108.0, 110.0])
    config = OpenReinforceConfig(
        tick_size=1.0, commission_per_fill_ticks=1.0,
        slippage_per_fill_ticks=0.0,
    )
    long_pnl, short_pnl = executable_side_pnls(data, 0, 2, config)
    np.testing.assert_allclose(long_pnl, [3.0, -2.0])
    np.testing.assert_allclose(short_pnl, [-11.0, -6.0])
