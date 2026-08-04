from __future__ import annotations

import asyncio

from market_collector.writer import WriterRegistry

from .binance import BinanceConnector
from .bitfinex import BitfinexConnector
from .bybit import BybitConnector
from .coinbase import CoinbaseConnector
from .deribit import DeribitConnector
from .hyperliquid import HyperliquidConnector
from .kraken import KrakenConnector
from .okx import OkxConnector


def build_stream_tasks(config: dict, writers: WriterRegistry, stop_event: asyncio.Event) -> list:
    tasks = []
    include_raw = bool(config.get("include_raw_message", True))
    for stream in config.get("streams", []):
        if not stream.get("enabled", False):
            continue
        exchange = stream.get("exchange")
        if exchange == "binance":
            connector = BinanceConnector(stream, writers, stop_event, include_raw)
        elif exchange == "deribit":
            connector = DeribitConnector(stream, writers, stop_event, include_raw)
        elif exchange == "bitfinex":
            connector = BitfinexConnector(stream, writers, stop_event, include_raw)
        elif exchange == "hyperliquid":
            connector = HyperliquidConnector(stream, writers, stop_event, include_raw)
        elif exchange == "okx":
            connector = OkxConnector(stream, writers, stop_event, include_raw)
        elif exchange == "bybit":
            connector = BybitConnector(stream, writers, stop_event, include_raw)
        elif exchange == "coinbase":
            connector = CoinbaseConnector(stream, writers, stop_event, include_raw)
        elif exchange == "kraken":
            connector = KrakenConnector(stream, writers, stop_event, include_raw)
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
        tasks.extend(connector.tasks())
    return tasks
