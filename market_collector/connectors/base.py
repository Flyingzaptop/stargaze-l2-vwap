from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import aiohttp

from market_collector.metrics import metrics
from market_collector.timeutil import now_ns
from market_collector.writer import WriterRegistry


class BaseConnector:
    exchange: str

    def __init__(self, config: dict, writers: WriterRegistry, stop_event: asyncio.Event, include_raw: bool) -> None:
        self.config = config
        self.writers = writers
        self.stop_event = stop_event
        self.include_raw = include_raw
        self.market = str(config.get("market", ""))
        self.symbol = str(config.get("symbol", ""))
        self._event_id = 0

    def next_event_id(self) -> int:
        self._event_id += 1
        return self._event_id

    def raw_once(self, payload: Any, row_idx: int) -> str | None:
        if not self.include_raw or row_idx != 0:
            return None
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    async def ws_json_loop(self, url: str, on_message, subscribe: dict | None = None) -> None:
        reconnect_delay = 1.0
        connection_key = f"{self.exchange}:{self.market}:{self.symbol}:{url}"
        while not self.stop_event.is_set():
            try:
                metrics.set_connection(connection_key, "connecting")
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=20, timeout=30, max_msg_size=64 * 1024 * 1024) as ws:
                        if subscribe is not None:
                            payload = subscribe() if callable(subscribe) else subscribe
                            if inspect.isawaitable(payload):
                                payload = await payload
                            await ws.send_json(payload)
                        print(f"connected {self.exchange} {self.symbol} {url}", flush=True)
                        metrics.set_connection(connection_key, "connected")
                        reconnect_delay = 1.0
                        async for msg in ws:
                            if self.stop_event.is_set():
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                metrics.set_connection(connection_key, "connected", message_seen=True)
                                await on_message(json.loads(msg.data), now_ns())
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"{self.exchange} {self.symbol} websocket error: {exc!r}; reconnecting", flush=True)
                metrics.set_connection(connection_key, "reconnecting", error=repr(exc))
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)
