from __future__ import annotations

from datetime import date
from pathlib import Path
import zipfile

import numpy as np
import pytest

from stargaze_ml.contracts import VENUE_INDEX
from stargaze_ml.features.state import VENUE_FEATURE_NAMES
from stargaze_ml.historical import binance_vision


def _archive(path: Path, name: str, text: str) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr(name, text)
    return path


def test_historical_day_uses_only_asof_quotes_and_delayed_trades(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(binance_vision, "DAY_SECONDS", 5)
    book = _archive(
        tmp_path / "book.zip",
        "book.csv",
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
        "1,100.0,20,100.1,20,1682899200100,1682899200100\n"
        "2,100.2,20,100.3,20,1682899200900,1682899200900\n"
        "3,100.4,20,100.5,20,1682899201100,1682899201100\n"
        "4,100.6,20,100.7,20,1682899201900,1682899201900\n",
    )
    trades = _archive(
        tmp_path / "trades.zip",
        "trades.csv",
        "id,price,qty,quote_qty,time,is_buyer_maker\n"
        "1,100.2,2,200.4,1682899200500,false\n",
    )
    depth = _archive(
        tmp_path / "depth.zip",
        "depth.csv",
        "timestamp,percentage,depth,notional\n"
        "2023-05-01 00:00:00,-5.00,50,5000\n"
        "2023-05-01 00:00:00,-1.00,10,1000\n"
        "2023-05-01 00:00:00,1.00,12,1200\n"
        "2023-05-01 00:00:00,5.00,60,6000\n",
    )
    metrics = _archive(
        tmp_path / "metrics.zip",
        "metrics.csv",
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2023-05-01 00:00:00,BTCUSDT,1000,100000,1.1,1.2,1.3,1.4\n",
    )

    result = binance_vision.build_historical_day(
        date(2023, 5, 1),
        book_ticker_path=book,
        trades_path=trades,
        book_depth_path=depth,
        metrics_path=metrics,
        phases_ms=(250,),
    )

    venue = VENUE_INDEX["binance_perpetual"]
    assert not result.frames.valid[0]
    assert result.frames.bid[1, venue] == 100.2
    assert result.frames.ask[1, venue] == 100.3
    assert result.execution[250.0].bid[1] == 100.4
    assert result.execution[250.0].ask[1] == 100.5
    assert result.execution[250.0].valid[1]
    trade_count = VENUE_FEATURE_NAMES.index("trade_count")
    assert result.frames.venue_x[1, venue, trade_count] == pytest.approx(np.log1p(1.0))
    assert result.frames.venue_x[0, venue, trade_count] == 0.0


def test_last_indices_keeps_last_row_of_each_sorted_group() -> None:
    groups = np.asarray([1, 1, 2, 4, 4, 4])
    np.testing.assert_array_equal(binance_vision._last_indices(groups), [1, 2, 5])
