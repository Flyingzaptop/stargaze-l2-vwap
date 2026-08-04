from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse

import aiohttp

from market_collector.timeutil import iso_to_ns

from .base import BaseConnector

_last_kraken_nonce = 0


class KrakenConnector(BaseConnector):
    exchange = "kraken"
    url = "wss://ws.kraken.com/v2"

    def tasks(self) -> list:
        tasks = []
        channels = set(self.config.get("channels", []))
        if "book" in channels:
            tasks.append(self.book())
        if "trade" in channels:
            tasks.append(self.trade())
        if "level3" in channels:
            tasks.append(self.level3())
        return tasks

    async def book(self) -> None:
        depth = int(self.config.get("book_depth", 1000))
        writer = self.writers.get(self.exchange, self.market, self.symbol, "book")
        subscribe = {
            "method": "subscribe",
            "params": {"channel": "book", "symbol": [self.symbol], "depth": depth},
        }

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if payload.get("method") or payload.get("channel") != "book":
                if payload.get("method") or payload.get("success") is not None or payload.get("error"):
                    print(f"kraken event: {payload}", flush=True)
                return
            records = []
            for item in payload.get("data", []):
                event_id = self.next_event_id()
                event_type = payload.get("type", "update")
                row_idx = 0
                for side, key in (("bid", "bids"), ("ask", "asks")):
                    for level in item.get(key, []):
                        price = float(level.get("price", 0.0))
                        quantity = float(level.get("qty", level.get("quantity", 0.0)))
                        records.append(
                            {
                                "exchange": self.exchange,
                                "market": self.market,
                                "symbol": item.get("symbol") or self.symbol,
                                "channel": "book",
                                "event_type": event_type,
                                "event_id": event_id,
                                "row_idx": row_idx,
                                "local_ts_ns": local_ts_ns,
                                "exchange_ts_ns": iso_to_ns(level.get("timestamp") or item.get("timestamp")),
                                "engine_ts_ns": iso_to_ns(level.get("timestamp") or item.get("timestamp")),
                                "sequence": str(item.get("checksum") or ""),
                                "is_snapshot": event_type == "snapshot",
                                "side": side,
                                "price": price,
                                "quantity": quantity,
                                "action": "delete" if quantity == 0.0 else "set",
                                "raw_message": self.raw_once(payload, row_idx),
                            }
                        )
                        row_idx += 1
            await writer.write_many(records)

        await self.ws_json_loop(self.url, on_message, subscribe)

    async def trade(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "trade")
        subscribe = {"method": "subscribe", "params": {"channel": "trade", "symbol": [self.symbol]}}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if payload.get("method") or payload.get("channel") != "trade":
                if payload.get("method") or payload.get("success") is not None or payload.get("error"):
                    print(f"kraken event: {payload}", flush=True)
                return
            records = []
            for item in payload.get("data", []):
                records.append(
                    {
                        "exchange": self.exchange,
                        "market": self.market,
                        "symbol": item.get("symbol") or self.symbol,
                        "channel": "trade",
                        "event_type": "trade",
                        "event_id": self.next_event_id(),
                        "row_idx": 0,
                        "local_ts_ns": local_ts_ns,
                        "exchange_ts_ns": iso_to_ns(item.get("timestamp")),
                        "engine_ts_ns": iso_to_ns(item.get("timestamp")),
                        "sequence": str(item.get("trade_id") or ""),
                        "trade_id": str(item.get("trade_id") or ""),
                        "price": float(item.get("price")),
                        "quantity": float(item.get("qty")),
                        "taker_side": item.get("side"),
                        "raw_message": self.raw_once(payload, 0),
                    }
                )
            await writer.write_many(records)

        await self.ws_json_loop(self.url, on_message, subscribe)

    async def level3(self) -> None:
        api_key = self.config.get("api_key") or os.environ.get("KRAKEN_API_KEY")
        api_secret = self.config.get("api_secret") or os.environ.get("KRAKEN_API_SECRET")
        static_token = self.config.get("ws_token") or os.environ.get("KRAKEN_WS_TOKEN")
        if not static_token and not (api_key and api_secret):
            print(
                "kraken level3 skipped: authenticated channel requires ws_token "
                "from Kraken GetWebSocketsToken",
                flush=True,
            )
            return

        writer = self.writers.get(self.exchange, self.market, self.symbol, "level3")

        async def subscribe_payload() -> dict:
            token = static_token
            if not token:
                token = await self.get_ws_token(str(api_key), str(api_secret))
            return {
                "method": "subscribe",
                "params": {
                    "channel": "level3",
                    "symbol": [self.symbol],
                    "depth": int(self.config.get("level3_depth", 1000)),
                    "snapshot": True,
                    "token": token,
                },
            }

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if payload.get("method") or payload.get("success") is not None or payload.get("error"):
                print(f"kraken level3 event: {payload}", flush=True)
                return
            if payload.get("channel") != "level3":
                return
            event_id = self.next_event_id()
            event_type = payload.get("type", "update")
            records = []
            row_idx = 0
            for item in payload.get("data", []):
                common = {
                    "exchange": self.exchange,
                    "market": self.market,
                    "symbol": item.get("symbol") or self.symbol,
                    "channel": "level3",
                    "event_type": event_type,
                    "event_id": event_id,
                    "local_ts_ns": local_ts_ns,
                    "exchange_ts_ns": iso_to_ns(item.get("timestamp")),
                    "engine_ts_ns": iso_to_ns(item.get("timestamp")),
                    "sequence": str(item.get("checksum") or ""),
                    "checksum": int(item["checksum"]) if item.get("checksum") is not None else None,
                    "is_snapshot": event_type == "snapshot",
                }
                for side, key in (("bid", "bids"), ("ask", "asks")):
                    for order in item.get(key, []):
                        action = order.get("event") or ("set" if event_type == "snapshot" else None)
                        records.append(
                            {
                                **common,
                                "row_idx": row_idx,
                                "side": side,
                                "price": float(order.get("limit_price")) if order.get("limit_price") is not None else None,
                                "quantity": float(order.get("order_qty")) if order.get("order_qty") is not None else None,
                                "action": action,
                                "order_id": order.get("order_id"),
                                "raw_message": self.raw_once(payload, row_idx),
                            }
                        )
                        row_idx += 1
            await writer.write_many(records)

        await self.ws_json_loop("wss://ws-l3.kraken.com/v2", on_message, subscribe_payload)

    async def get_ws_token(self, api_key: str, api_secret: str) -> str:
        global _last_kraken_nonce
        path = "/0/private/GetWebSocketsToken"
        candidate = time.time_ns()
        if candidate <= _last_kraken_nonce:
            candidate = _last_kraken_nonce + 1
        _last_kraken_nonce = candidate
        nonce = str(candidate)
        post_data = urllib.parse.urlencode({"nonce": nonce})
        encoded = (nonce + post_data).encode("utf-8")
        message = path.encode("utf-8") + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(api_secret), message, hashlib.sha512).digest()
        api_sign = base64.b64encode(signature).decode("ascii")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.kraken.com{path}",
                data=post_data,
                headers={
                    "API-Key": api_key,
                    "API-Sign": api_sign,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=20,
            ) as resp:
                payload = await resp.json()
                if resp.status >= 400 or payload.get("error"):
                    raise RuntimeError(f"Kraken GetWebSocketsToken failed: status={resp.status} error={payload.get('error')}")
                token = payload.get("result", {}).get("token")
                if not token:
                    raise RuntimeError(f"Kraken GetWebSocketsToken returned no token: keys={sorted(payload.keys())}")
                return str(token)
