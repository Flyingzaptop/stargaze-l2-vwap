from __future__ import annotations

from market_collector.timeutil import iso_to_ns

from .base import BaseConnector


class CoinbaseConnector(BaseConnector):
    exchange = "coinbase"
    url = "wss://advanced-trade-ws.coinbase.com"

    def tasks(self) -> list:
        tasks = []
        channels = set(self.config.get("channels", []))
        if "level2" in channels:
            tasks.append(self.level2())
        if "market_trades" in channels:
            tasks.append(self.market_trades())
        return tasks

    def subscribe(self, channel: str) -> dict:
        payload = {"type": "subscribe", "product_ids": [self.symbol], "channel": channel}
        jwt = self.config.get("jwt")
        if jwt:
            payload["jwt"] = jwt
        return payload

    async def level2(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "level2")

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if payload.get("channel") != "l2_data" and payload.get("channel") != "level2":
                return
            records = []
            for event in payload.get("events", []):
                event_id = self.next_event_id()
                event_type = event.get("type", "update")
                row_idx = 0
                for update in event.get("updates", []):
                    quantity = float(update.get("new_quantity", 0.0))
                    records.append(
                        {
                            "exchange": self.exchange,
                            "market": self.market,
                            "symbol": update.get("product_id") or self.symbol,
                            "channel": "level2",
                            "event_type": event_type,
                            "event_id": event_id,
                            "row_idx": row_idx,
                            "local_ts_ns": local_ts_ns,
                            "exchange_ts_ns": iso_to_ns(update.get("event_time")) or iso_to_ns(payload.get("timestamp")),
                            "engine_ts_ns": iso_to_ns(update.get("event_time")),
                            "sequence": str(payload.get("sequence_num") or ""),
                            "is_snapshot": event_type == "snapshot",
                            "side": update.get("side"),
                            "price": float(update.get("price_level")),
                            "quantity": quantity,
                            "action": "delete" if quantity == 0.0 else "set",
                            "raw_message": self.raw_once(payload, row_idx),
                        }
                    )
                    row_idx += 1
            await writer.write_many(records)

        await self.ws_json_loop(self.url, on_message, self.subscribe("level2"))

    async def market_trades(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "market_trades")

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if payload.get("channel") != "market_trades":
                return
            records = []
            for event in payload.get("events", []):
                for row_idx, trade in enumerate(event.get("trades", [])):
                    records.append(
                        {
                            "exchange": self.exchange,
                            "market": self.market,
                            "symbol": trade.get("product_id") or self.symbol,
                            "channel": "market_trades",
                            "event_type": event.get("type", "update"),
                            "event_id": self.next_event_id(),
                            "row_idx": row_idx,
                            "local_ts_ns": local_ts_ns,
                            "exchange_ts_ns": iso_to_ns(trade.get("time")) or iso_to_ns(payload.get("timestamp")),
                            "engine_ts_ns": iso_to_ns(trade.get("time")),
                            "sequence": str(payload.get("sequence_num") or ""),
                            "trade_id": str(trade.get("trade_id") or ""),
                            "price": float(trade.get("price")),
                            "quantity": float(trade.get("size")),
                            "taker_side": None,
                            "side": str(trade.get("side", "")).lower(),
                            "raw_message": self.raw_once(payload, row_idx),
                        }
                    )
            await writer.write_many(records)

        await self.ws_json_loop(self.url, on_message, self.subscribe("market_trades"))
