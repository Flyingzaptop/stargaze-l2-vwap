from __future__ import annotations

from typing import Any

from market_collector.timeutil import ms_to_ns

from .base import BaseConnector


class DeribitConnector(BaseConnector):
    exchange = "deribit"
    url = "wss://www.deribit.com/ws/api/v2"

    def tasks(self) -> list:
        tasks = []
        channels = set(self.config.get("channels", []))
        if "book" in channels:
            tasks.append(self.book())
        if "trades" in channels:
            tasks.append(self.trades())
        if "ticker" in channels:
            tasks.append(self.ticker())
        return tasks

    @staticmethod
    def subscribe(channel: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/subscribe",
            "params": {"channels": [channel]},
        }

    def parse_book_message(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        params = payload.get("params", {})
        if payload.get("method") != "subscription" or not str(params.get("channel", "")).startswith("book."):
            return []
        data = params.get("data", {})
        event_type = str(data.get("type", "change"))
        event_id = self.next_event_id()
        change_id = data.get("change_id")
        previous_id = data.get("prev_change_id")
        common = {
            "exchange": self.exchange,
            "market": self.market,
            "symbol": data.get("instrument_name") or self.symbol,
            "channel": "book",
            "event_type": event_type,
            "event_id": event_id,
            "local_ts_ns": local_ts_ns,
            "exchange_ts_ns": ms_to_ns(data.get("timestamp")),
            "engine_ts_ns": ms_to_ns(data.get("timestamp")),
            "sequence": str(change_id) if change_id is not None else None,
            "sequence_start": None,
            "prev_sequence": str(previous_id) if previous_id is not None else None,
            "is_snapshot": event_type == "snapshot",
        }
        records = []
        row_idx = 0
        for side, key in (("bid", "bids"), ("ask", "asks")):
            for level in data.get(key, []):
                action, price, amount = level[:3]
                normalized_action = "delete" if action == "delete" or float(amount) == 0.0 else "set"
                records.append(
                    {
                        **common,
                        "row_idx": row_idx,
                        "side": side,
                        "price": float(price),
                        "quantity": 0.0 if normalized_action == "delete" else float(amount),
                        "action": normalized_action,
                        "raw_message": self.raw_once(payload, row_idx),
                    }
                )
                row_idx += 1
        return records

    async def book(self) -> None:
        interval = str(self.config.get("book_interval", "100ms"))
        channel = f"book.{self.symbol}.{interval}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "book")
        previous_change_id: int | None = None

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            nonlocal previous_change_id
            rows = self.parse_book_message(payload, local_ts_ns)
            if not rows:
                return
            current = int(rows[0]["sequence"]) if rows[0].get("sequence") is not None else None
            previous = int(rows[0]["prev_sequence"]) if rows[0].get("prev_sequence") is not None else None
            if rows[0].get("is_snapshot"):
                previous_change_id = current
            else:
                if previous_change_id is None or previous != previous_change_id:
                    raise RuntimeError(
                        f"Deribit book sequence gap: expected prev_change_id={previous_change_id}, got {previous}"
                    )
                previous_change_id = current
            await writer.write_many(rows)

        await self.ws_json_loop(self.url, on_message, self.subscribe(channel))

    def parse_trades_message(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        params = payload.get("params", {})
        if payload.get("method") != "subscription" or not str(params.get("channel", "")).startswith("trades."):
            return []
        trades = params.get("data", [])
        event_id = self.next_event_id()
        records = []
        for row_idx, trade in enumerate(trades):
            trade_id = trade.get("trade_id")
            records.append(
                {
                    "exchange": self.exchange,
                    "market": self.market,
                    "symbol": trade.get("instrument_name") or self.symbol,
                    "channel": "trades",
                    "event_type": "trade",
                    "event_id": event_id,
                    "row_idx": row_idx,
                    "local_ts_ns": local_ts_ns,
                    "exchange_ts_ns": ms_to_ns(trade.get("timestamp")),
                    "engine_ts_ns": ms_to_ns(trade.get("timestamp")),
                    "sequence": str(trade_id) if trade_id is not None else None,
                    "trade_id": str(trade_id) if trade_id is not None else None,
                    "price": float(trade["price"]),
                    "quantity": float(trade["amount"]),
                    "taker_side": str(trade.get("direction", "")).lower() or None,
                    "raw_message": self.raw_once(payload, row_idx),
                }
            )
        return records

    async def trades(self) -> None:
        channel = self.trades_channel
        writer = self.writers.get(self.exchange, self.market, self.symbol, "trades")

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_trades_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, self.subscribe(channel))

    @property
    def trades_channel(self) -> str:
        interval = str(self.config.get("trades_interval", "100ms"))
        return f"trades.{self.symbol}.{interval}"

    def parse_ticker_message(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        params = payload.get("params", {})
        if payload.get("method") != "subscription" or not str(params.get("channel", "")).startswith("ticker."):
            return []
        data = params.get("data", {})
        timestamp = data.get("timestamp")
        return [
            {
                "exchange": self.exchange,
                "market": self.market,
                "symbol": data.get("instrument_name") or self.symbol,
                "channel": "ticker",
                "event_type": "derivative_context",
                "event_id": self.next_event_id(),
                "row_idx": 0,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": ms_to_ns(timestamp),
                "engine_ts_ns": ms_to_ns(timestamp),
                "sequence": str(timestamp) if timestamp is not None else None,
                "mark_price": float(data["mark_price"]) if data.get("mark_price") is not None else None,
                "index_price": float(data["index_price"]) if data.get("index_price") is not None else None,
                "open_interest": float(data["open_interest"]) if data.get("open_interest") is not None else None,
                "funding_rate": float(data["current_funding"]) if data.get("current_funding") is not None else None,
                "raw_message": self.raw_once(payload, 0),
            }
        ]

    async def ticker(self) -> None:
        interval = str(self.config.get("ticker_interval", "100ms"))
        channel = f"ticker.{self.symbol}.{interval}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "ticker")

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_ticker_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, self.subscribe(channel))
