from __future__ import annotations

from market_collector.timeutil import ms_to_ns

from .base import BaseConnector


class OkxConnector(BaseConnector):
    exchange = "okx"
    url = "wss://ws.okx.com:8443/ws/v5/public"

    def tasks(self) -> list:
        tasks = []
        channels = set(self.config.get("channels", []))
        if "books" in channels:
            tasks.append(self.books())
        if "trades" in channels:
            tasks.append(self.trades())
        return tasks

    async def books(self) -> None:
        book_channel = self.config.get("book_channel", "books")
        writer = self.writers.get(self.exchange, self.market, self.symbol, "books")
        subscribe = {"op": "subscribe", "args": [{"channel": book_channel, "instId": self.symbol}]}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if "event" in payload:
                print(f"okx event: {payload}", flush=True)
                return
            arg = payload.get("arg", {})
            if arg.get("channel") != book_channel:
                return
            records = []
            for item in payload.get("data", []):
                event_id = self.next_event_id()
                action = payload.get("action", "update")
                common = {
                    "exchange": self.exchange,
                    "market": self.market,
                    "symbol": self.symbol,
                    "channel": "books",
                    "event_type": action,
                    "event_id": event_id,
                    "local_ts_ns": local_ts_ns,
                    "exchange_ts_ns": ms_to_ns(item.get("ts")),
                    "engine_ts_ns": ms_to_ns(item.get("ts")),
                    "sequence": str(item.get("seqId") or item.get("checksum") or ""),
                    "sequence_start": str(item.get("prevSeqId") or ""),
                    "prev_sequence": str(item.get("prevSeqId") or ""),
                    "is_snapshot": action == "snapshot",
                }
                row_idx = 0
                for side, key in (("bid", "bids"), ("ask", "asks")):
                    for level in item.get(key, []):
                        price = float(level[0])
                        quantity = float(level[1])
                        order_count = float(level[3]) if len(level) > 3 and level[3] != "" else None
                        records.append(
                            {
                                **common,
                                "row_idx": row_idx,
                                "side": side,
                                "price": price,
                                "quantity": quantity,
                                "action": "delete" if quantity == 0.0 else "set",
                                "order_count": order_count,
                                "raw_message": self.raw_once(payload, row_idx),
                            }
                        )
                        row_idx += 1
            await writer.write_many(records)

        await self.ws_json_loop(self.url, on_message, subscribe)

    async def trades(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "trades")
        subscribe = {"op": "subscribe", "args": [{"channel": "trades", "instId": self.symbol}]}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if "event" in payload:
                print(f"okx event: {payload}", flush=True)
                return
            if payload.get("arg", {}).get("channel") != "trades":
                return
            records = []
            for item in payload.get("data", []):
                records.append(
                    {
                        "exchange": self.exchange,
                        "market": self.market,
                        "symbol": self.symbol,
                        "channel": "trades",
                        "event_type": "trade",
                        "event_id": self.next_event_id(),
                        "row_idx": 0,
                        "local_ts_ns": local_ts_ns,
                        "exchange_ts_ns": ms_to_ns(item.get("ts")),
                        "engine_ts_ns": ms_to_ns(item.get("ts")),
                        "sequence": str(item.get("tradeId") or ""),
                        "trade_id": str(item.get("tradeId") or ""),
                        "price": float(item.get("px")),
                        "quantity": float(item.get("sz")),
                        "taker_side": item.get("side"),
                        "raw_message": self.raw_once(payload, 0),
                    }
                )
            await writer.write_many(records)

        await self.ws_json_loop(self.url, on_message, subscribe)
