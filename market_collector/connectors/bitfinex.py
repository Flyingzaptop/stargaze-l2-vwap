from __future__ import annotations

from typing import Any

from market_collector.timeutil import ms_to_ns

from .base import BaseConnector


class BitfinexConnector(BaseConnector):
    exchange = "bitfinex"
    url = "wss://api-pub.bitfinex.com/ws/2"

    @property
    def api_symbol(self) -> str:
        symbol = self.symbol.replace("/", "").replace("-", "")
        if symbol[:1].lower() == "t":
            symbol = symbol[1:]
        return f"t{symbol.upper()}"

    def tasks(self) -> list:
        tasks = []
        channels = set(self.config.get("channels", []))
        if "book" in channels:
            tasks.append(self.book())
        if "trades" in channels:
            tasks.append(self.trades())
        return tasks

    def parse_book_message(self, payload: Any, local_ts_ns: int) -> list[dict]:
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] == "hb":
            return []
        body = payload[1]
        if not isinstance(body, list):
            return []
        is_snapshot = bool(body and isinstance(body[0], list))
        levels = body if is_snapshot else [body]
        event_id = self.next_event_id()
        records = []
        for row_idx, level in enumerate(levels):
            if len(level) < 3:
                continue
            price, count, amount = float(level[0]), int(level[1]), float(level[2])
            side = "bid" if amount > 0 else "ask"
            deleted = count == 0
            records.append(
                {
                    "exchange": self.exchange,
                    "market": self.market,
                    "symbol": self.symbol,
                    "channel": "book",
                    "event_type": "snapshot" if is_snapshot else "update",
                    "event_id": event_id,
                    "row_idx": row_idx,
                    "local_ts_ns": local_ts_ns,
                    "exchange_ts_ns": None,
                    "engine_ts_ns": None,
                    "sequence": None,
                    "is_snapshot": is_snapshot,
                    "side": side,
                    "price": price,
                    "quantity": 0.0 if deleted else abs(amount),
                    "action": "delete" if deleted else "set",
                    "order_count": float(count),
                    "raw_message": self.raw_once(payload, row_idx),
                }
            )
        return records

    async def book(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "book")
        subscribe = {
            "event": "subscribe",
            "channel": "book",
            "symbol": self.api_symbol,
            "prec": str(self.config.get("book_precision", "P0")),
            "freq": str(self.config.get("book_frequency", "F0")),
            "len": str(self.config.get("book_length", self.config.get("book_depth", "250"))),
        }

        async def on_message(payload: Any, local_ts_ns: int) -> None:
            if isinstance(payload, dict):
                print(f"bitfinex book event: {payload}", flush=True)
                return
            await writer.write_many(self.parse_book_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, subscribe)

    def parse_trades_message(self, payload: Any, local_ts_ns: int) -> list[dict]:
        if not isinstance(payload, list) or len(payload) < 3 or payload[1] != "te":
            return []
        trade = payload[2]
        if not isinstance(trade, list) or len(trade) < 4:
            return []
        trade_id, timestamp_ms, amount, price = trade[:4]
        amount = float(amount)
        return [
            {
                "exchange": self.exchange,
                "market": self.market,
                "symbol": self.symbol,
                "channel": "trades",
                "event_type": "trade",
                "event_id": self.next_event_id(),
                "row_idx": 0,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": ms_to_ns(timestamp_ms),
                "engine_ts_ns": ms_to_ns(timestamp_ms),
                "sequence": str(trade_id),
                "trade_id": str(trade_id),
                "price": float(price),
                "quantity": abs(amount),
                "taker_side": "buy" if amount > 0 else "sell",
                "raw_message": self.raw_once(payload, 0),
            }
        ]

    async def trades(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "trades")
        subscribe = {"event": "subscribe", "channel": "trades", "symbol": self.api_symbol}

        async def on_message(payload: Any, local_ts_ns: int) -> None:
            if isinstance(payload, dict):
                print(f"bitfinex trades event: {payload}", flush=True)
                return
            await writer.write_many(self.parse_trades_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, subscribe)
