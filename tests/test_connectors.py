from __future__ import annotations

import asyncio

import pytest

from market_collector.connectors.binance import (
    BinanceConnector,
    BinanceDepthIdleTimeout,
    BinanceSequenceGap,
    next_depth_item,
)
from market_collector.connectors.bitfinex import BitfinexConnector
from market_collector.connectors.bybit import BybitConnector
from market_collector.connectors.deribit import DeribitConnector
from market_collector.connectors.hyperliquid import HyperliquidConnector
from market_collector.connectors import build_stream_tasks
from market_collector.records import SCHEMA, normalize_record

import pyarrow as pa


def connector(cls, *, market: str, symbol: str, include_raw: bool = True, **config):
    return cls(
        {"market": market, "symbol": symbol, "channels": [], **config},
        writers=object(),
        stop_event=asyncio.Event(),
        include_raw=include_raw,
    )


@pytest.mark.parametrize("exchange", ["binance", "deribit", "bitfinex", "hyperliquid"])
def test_connector_is_registered(exchange: str) -> None:
    tasks = build_stream_tasks(
        {
            "include_raw_message": False,
            "streams": [
                {
                    "enabled": True,
                    "exchange": exchange,
                    "market": "spot",
                    "symbol": "BTCUSD",
                    "channels": [],
                }
            ],
        },
        writers=object(),
        stop_event=asyncio.Event(),
    )
    assert tasks == []


def assert_schema_compatible(rows: list[dict]) -> None:
    table = pa.Table.from_pylist([normalize_record(row) for row in rows], schema=SCHEMA)
    assert table.schema == SCHEMA


def test_derivative_context_fields_are_part_of_common_schema() -> None:
    assert {
        "mark_price",
        "index_price",
        "oracle_price",
        "open_interest",
        "funding_rate",
        "next_funding_ts_ns",
        "liquidation_side",
    }.issubset(SCHEMA.names)


def test_binance_spot_snapshot_and_diff_metadata() -> None:
    item = connector(BinanceConnector, market="spot", symbol="BTCUSDT")
    snapshot_id, snapshot = item.parse_depth_snapshot(
        {"lastUpdateId": 100, "bids": [["60000", "1.2"]], "asks": [["60001", "0.7"]]},
        10,
    )
    parsed = item.parse_depth_event(
        {
            "stream": "btcusdt@depth@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1710000000000,
                "s": "BTCUSDT",
                "U": 101,
                "u": 103,
                "b": [["60000", "0"]],
                "a": [["60002", "2.5"]],
            },
        },
        20,
    )

    assert snapshot_id == 100
    assert {row["is_snapshot"] for row in snapshot} == {True}
    assert {row["sequence"] for row in snapshot} == {"100"}
    assert parsed is not None
    metadata, rows = parsed
    assert metadata == {"first": 101, "last": 103, "previous": None}
    assert item.accepts_first_diff(snapshot_id, metadata)
    assert rows[0]["action"] == "delete" and rows[0]["quantity"] == 0.0
    assert rows[1]["side"] == "ask" and rows[1]["action"] == "set"
    assert {row["sequence_start"] for row in rows} == {"101"}
    assert {row["sequence"] for row in rows} == {"103"}
    assert {row["prev_sequence"] for row in rows} == {None}
    assert rows[0]["raw_message"] is not None and rows[1]["raw_message"] is None
    assert item.rest_depth_url.endswith("/api/v3/depth")
    assert "stream.binance.com" in item.ws_base_url
    assert_schema_compatible(snapshot + rows)


def test_binance_spot_sequence_gap_detection() -> None:
    item = connector(BinanceConnector, market="spot", symbol="BTCUSDT")
    item.validate_next_diff(103, {"first": 103, "last": 105, "previous": None})
    with pytest.raises(BinanceSequenceGap, match="spot sequence gap"):
        item.validate_next_diff(105, {"first": 108, "last": 109, "previous": None})


def test_binance_depth_queue_times_out_when_socket_goes_silent() -> None:
    queue = asyncio.Queue()
    with pytest.raises(BinanceDepthIdleTimeout, match="no Binance depth message"):
        asyncio.run(next_depth_item(queue, 0.001))


def test_binance_depth_queue_returns_buffered_message() -> None:
    queue = asyncio.Queue()
    payload = ({"data": {"e": "depthUpdate"}}, 123)
    queue.put_nowait(payload)
    assert asyncio.run(next_depth_item(queue, 0.1)) == payload


def test_binance_usdt_m_uses_pu_chain_and_futures_endpoints() -> None:
    item = connector(BinanceConnector, market="usdt_m", symbol="BTCUSDT")
    parsed = item.parse_depth_event(
        {
            "e": "depthUpdate",
            "E": 1710000000000,
            "T": 1709999999999,
            "s": "BTCUSDT",
            "U": 200,
            "u": 204,
            "pu": 199,
            "b": [["60000", "1"]],
            "a": [],
        },
        20,
    )
    assert parsed is not None
    metadata, rows = parsed
    assert item.accepts_first_diff(202, metadata)
    assert rows[0]["prev_sequence"] == "199"
    assert rows[0]["engine_ts_ns"] == 1_709_999_999_999_000_000
    item.validate_next_diff(199, metadata)
    with pytest.raises(BinanceSequenceGap, match="USDT-M sequence gap"):
        item.validate_next_diff(198, metadata)
    assert item.rest_depth_url.endswith("/fapi/v1/depth")
    assert "fstream.binance.com" in item.ws_base_url


def test_binance_trade_maker_flag_is_converted_to_taker_side() -> None:
    item = connector(BinanceConnector, market="spot", symbol="BTCUSDT", include_raw=False)
    rows = item.parse_trade_event(
        {
            "e": "trade",
            "E": 1710000000001,
            "T": 1710000000000,
            "s": "BTCUSDT",
            "t": 42,
            "p": "61000.5",
            "q": "0.01",
            "m": True,
        },
        30,
    )
    assert rows[0]["trade_id"] == "42"
    assert rows[0]["taker_side"] == "sell"
    assert rows[0]["raw_message"] is None


def test_binance_mark_price_context_is_normalized() -> None:
    item = connector(BinanceConnector, market="usdt_m", symbol="BTCUSDT")
    rows = item.parse_mark_price_event(
        {
            "stream": "btcusdt@markPrice@1s",
            "data": {
                "e": "markPriceUpdate",
                "E": 1710000000000,
                "s": "BTCUSDT",
                "p": "60010.25",
                "i": "60000.50",
                "P": "60001.00",
                "r": "0.0001",
                "T": 1710028800000,
            },
        },
        31,
    )
    assert rows[0]["event_type"] == "derivative_context"
    assert rows[0]["mark_price"] == 60010.25
    assert rows[0]["index_price"] == 60000.5
    assert rows[0]["funding_rate"] == 0.0001
    assert rows[0]["next_funding_ts_ns"] == 1_710_028_800_000_000_000
    assert_schema_compatible(rows)


@pytest.mark.parametrize(
    ("order_side", "liquidation_side"),
    [("SELL", "long"), ("BUY", "short")],
)
def test_binance_force_order_has_explicit_liquidated_position_side(
    order_side: str, liquidation_side: str
) -> None:
    item = connector(BinanceConnector, market="usdt_m", symbol="BTCUSDT")
    rows = item.parse_force_order_event(
        {
            "e": "forceOrder",
            "E": 1710000000100,
            "o": {
                "s": "BTCUSDT",
                "S": order_side,
                "q": "0.5",
                "p": "59990",
                "ap": "60000",
                "z": "0.4",
                "T": 1710000000000,
            },
        },
        32,
    )
    assert rows[0]["event_type"] == "liquidation"
    assert rows[0]["liquidation_side"] == liquidation_side
    assert rows[0]["taker_side"] == order_side.lower()
    assert rows[0]["price"] == 60000.0
    assert rows[0]["quantity"] == 0.4
    assert_schema_compatible(rows)


def test_deribit_book_snapshot_and_change_sequence() -> None:
    item = connector(DeribitConnector, market="perpetual", symbol="BTC-PERPETUAL")
    snapshot = item.parse_book_message(
        {
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {
                "channel": "book.BTC-PERPETUAL.100ms",
                "data": {
                    "type": "snapshot",
                    "timestamp": 1710000000000,
                    "instrument_name": "BTC-PERPETUAL",
                    "change_id": 500,
                    "bids": [["new", 60000.0, 10.0]],
                    "asks": [["new", 60001.0, 12.0]],
                },
            },
        },
        40,
    )
    change = item.parse_book_message(
        {
            "method": "subscription",
            "params": {
                "channel": "book.BTC-PERPETUAL.100ms",
                "data": {
                    "type": "change",
                    "timestamp": 1710000000100,
                    "instrument_name": "BTC-PERPETUAL",
                    "change_id": 501,
                    "prev_change_id": 500,
                    "bids": [["delete", 60000.0, 0.0]],
                    "asks": [["change", 60001.0, 15.0]],
                },
            },
        },
        50,
    )
    assert len(snapshot) == 2 and {row["is_snapshot"] for row in snapshot} == {True}
    assert {row["sequence"] for row in snapshot} == {"500"}
    assert change[0]["prev_sequence"] == "500"
    assert change[0]["action"] == "delete" and change[0]["quantity"] == 0.0
    assert change[1]["action"] == "set" and change[1]["quantity"] == 15.0
    assert_schema_compatible(snapshot + change)


def test_deribit_trade_packet_is_atomic() -> None:
    item = connector(DeribitConnector, market="perpetual", symbol="BTC-PERPETUAL")
    rows = item.parse_trades_message(
        {
            "method": "subscription",
            "params": {
                "channel": "trades.BTC-PERPETUAL.raw",
                "data": [
                    {
                        "timestamp": 1710000000000,
                        "instrument_name": "BTC-PERPETUAL",
                        "trade_id": "BTC-1",
                        "direction": "buy",
                        "price": 60000,
                        "amount": 10,
                    },
                    {
                        "timestamp": 1710000000001,
                        "instrument_name": "BTC-PERPETUAL",
                        "trade_id": "BTC-2",
                        "direction": "sell",
                        "price": 59999.5,
                        "amount": 5,
                    },
                ],
            },
        },
        60,
    )
    assert len(rows) == 2
    assert rows[0]["event_id"] == rows[1]["event_id"]
    assert [row["row_idx"] for row in rows] == [0, 1]
    assert [row["taker_side"] for row in rows] == ["buy", "sell"]


def test_deribit_public_trades_default_to_100ms() -> None:
    item = connector(DeribitConnector, market="perpetual", symbol="BTC-PERPETUAL")
    assert item.trades_channel == "trades.BTC-PERPETUAL.100ms"
    authorized = connector(
        DeribitConnector,
        market="perpetual",
        symbol="BTC-PERPETUAL",
        trades_interval="raw",
    )
    assert authorized.trades_channel == "trades.BTC-PERPETUAL.raw"


def test_deribit_ticker_context_is_normalized() -> None:
    item = connector(DeribitConnector, market="perpetual", symbol="BTC-PERPETUAL")
    rows = item.parse_ticker_message(
        {
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {
                "channel": "ticker.BTC-PERPETUAL.100ms",
                "data": {
                    "timestamp": 1710000000000,
                    "instrument_name": "BTC-PERPETUAL",
                    "mark_price": 60010.0,
                    "index_price": 60000.0,
                    "open_interest": 502097590,
                    "current_funding": 0.000021,
                },
            },
        },
        61,
    )
    assert rows[0]["channel"] == "ticker"
    assert rows[0]["mark_price"] == 60010.0
    assert rows[0]["index_price"] == 60000.0
    assert rows[0]["open_interest"] == 502097590.0
    assert rows[0]["funding_rate"] == 0.000021
    assert_schema_compatible(rows)


def test_bitfinex_book_snapshot_update_and_delete() -> None:
    item = connector(BitfinexConnector, market="spot", symbol="BTCUSD")
    snapshot = item.parse_book_message(
        [12, [[60000, 3, 1.5], [60001, 2, -0.8]]],
        70,
    )
    deletion = item.parse_book_message([12, [60000, 0, 1]], 80)
    update = item.parse_book_message([12, [60002, 4, -2.25]], 90)
    assert item.api_symbol == "tBTCUSD"
    assert [row["side"] for row in snapshot] == ["bid", "ask"]
    assert {row["is_snapshot"] for row in snapshot} == {True}
    assert deletion[0]["side"] == "bid"
    assert deletion[0]["action"] == "delete" and deletion[0]["quantity"] == 0.0
    assert update[0]["side"] == "ask" and update[0]["quantity"] == 2.25
    assert item.parse_book_message([12, "hb"], 100) == []
    assert_schema_compatible(snapshot + deletion + update)


def test_bitfinex_records_te_once_and_ignores_tu_duplicate() -> None:
    item = connector(BitfinexConnector, market="spot", symbol="BTCUSD")
    trade = [1234, 1710000000000, -0.25, 60000.5]
    rows = item.parse_trades_message([13, "te", trade], 100)
    assert item.parse_trades_message([13, "tu", trade], 101) == []
    assert item.parse_trades_message([13, [trade]], 102) == []
    assert rows[0]["trade_id"] == "1234"
    assert rows[0]["quantity"] == 0.25
    assert rows[0]["taker_side"] == "sell"


def test_hyperliquid_l2_is_a_full_snapshot_with_order_counts() -> None:
    item = connector(HyperliquidConnector, market="perp", symbol="BTC-PERP")
    rows = item.parse_l2_book_message(
        {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "time": 1710000000000,
                "levels": [
                    [{"px": "60000", "sz": "1.25", "n": 4}],
                    [{"px": "60001", "sz": "0.75", "n": 2}],
                ],
            },
        },
        110,
    )
    assert item.coin == "BTC"
    assert len(rows) == 2
    assert {row["is_snapshot"] for row in rows} == {True}
    assert [row["side"] for row in rows] == ["bid", "ask"]
    assert [row["order_count"] for row in rows] == [4.0, 2.0]
    assert {row["sequence"] for row in rows} == {"1710000000000"}
    assert_schema_compatible(rows)


def test_hyperliquid_trade_id_is_globally_scoped() -> None:
    item = connector(HyperliquidConnector, market="perp", symbol="BTC")
    rows = item.parse_trades_message(
        {
            "channel": "trades",
            "data": [
                {"coin": "BTC", "side": "B", "px": "60000", "sz": "0.1", "time": 1710000000000, "tid": 10},
                {"coin": "BTC", "side": "A", "px": "59999", "sz": "0.2", "time": 1710000000001, "tid": 11},
            ],
        },
        120,
    )
    assert rows[0]["trade_id"] == "1710000000000:BTC:10"
    assert rows[0]["event_id"] == rows[1]["event_id"]
    assert [row["taker_side"] for row in rows] == ["buy", "sell"]


def test_hyperliquid_active_asset_context_is_normalized() -> None:
    item = connector(HyperliquidConnector, market="perp", symbol="BTC")
    rows = item.parse_active_asset_context_message(
        {
            "channel": "activeAssetCtx",
            "data": {
                "coin": "BTC",
                "ctx": {
                    "markPx": "60010.5",
                    "oraclePx": "60000.25",
                    "openInterest": "12345.67",
                    "funding": "0.0000125",
                },
            },
        },
        121,
    )
    assert rows[0]["mark_price"] == 60010.5
    assert rows[0]["oracle_price"] == 60000.25
    assert rows[0]["open_interest"] == 12345.67
    assert rows[0]["funding_rate"] == 0.0000125
    assert_schema_compatible(rows)


def test_bybit_ticker_context_is_normalized() -> None:
    item = connector(BybitConnector, market="linear", symbol="BTCUSDT")
    rows = item.parse_ticker_message(
        {
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "ts": 1760325052630,
            "cs": 9532239429,
            "data": {
                "symbol": "BTCUSDT",
                "markPrice": "66666.60",
                "indexPrice": "66660.10",
                "openInterest": "492373.72",
                "nextFundingTime": "1760342400000",
                "fundingRate": "-0.005",
            },
        },
        130,
    )
    assert rows[0]["mark_price"] == 66666.6
    assert rows[0]["index_price"] == 66660.1
    assert rows[0]["open_interest"] == 492373.72
    assert rows[0]["funding_rate"] == -0.005
    assert rows[0]["next_funding_ts_ns"] == 1_760_342_400_000_000_000
    assert rows[0]["sequence"] == "9532239429"
    assert_schema_compatible(rows)


@pytest.mark.parametrize(
    ("position_side", "liquidation_side", "forced_order_side"),
    [("Buy", "long", "sell"), ("Sell", "short", "buy")],
)
def test_bybit_all_liquidation_normalizes_position_semantics(
    position_side: str, liquidation_side: str, forced_order_side: str
) -> None:
    item = connector(BybitConnector, market="linear", symbol="BTCUSDT")
    rows = item.parse_all_liquidation_message(
        {
            "topic": "allLiquidation.BTCUSDT",
            "type": "snapshot",
            "ts": 1739502303204,
            "data": [
                {
                    "T": 1739502302929,
                    "s": "BTCUSDT",
                    "S": position_side,
                    "v": "2.5",
                    "p": "60000",
                }
            ],
        },
        131,
    )
    assert rows[0]["liquidation_side"] == liquidation_side
    assert rows[0]["taker_side"] == forced_order_side
    assert rows[0]["price"] == 60000.0
    assert rows[0]["quantity"] == 2.5
    assert_schema_compatible(rows)
