from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from market_collector.metrics import metrics
from market_collector.timeutil import ms_to_ns, now_ns

from .base import BaseConnector


class BinanceSequenceGap(RuntimeError):
    """Raised when a diff-depth stream can no longer bridge the local book."""


class BinanceDepthIdleTimeout(TimeoutError):
    """Raised when an open depth websocket stops delivering diff events."""


async def next_depth_item(
    queue: asyncio.Queue[tuple[dict, int] | BaseException],
    timeout_seconds: float,
) -> tuple[dict, int] | BaseException:
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise BinanceDepthIdleTimeout(
            f"no Binance depth message for {timeout_seconds:g}s"
        ) from exc


class BinanceConnector(BaseConnector):
    exchange = "binance"

    @property
    def is_spot(self) -> bool:
        return self.market.lower() in {"spot", "cash"}

    @property
    def ws_base_url(self) -> str:
        if self.is_spot:
            return "wss://stream.binance.com:9443/stream"
        return "wss://fstream.binance.com/stream"

    @property
    def rest_depth_url(self) -> str:
        if self.is_spot:
            return "https://api.binance.com/api/v3/depth"
        return "https://fapi.binance.com/fapi/v1/depth"

    def tasks(self) -> list:
        tasks = []
        channels = set(self.config.get("channels", []))
        if "depth" in channels:
            tasks.append(self.depth())
        if "trades" in channels:
            tasks.append(self.trades())
        if not self.is_spot and ({"markPrice", "mark_price"} & channels):
            tasks.append(self.mark_price())
        if not self.is_spot and ({"forceOrder", "liquidations"} & channels):
            tasks.append(self.force_order())
        return tasks

    def parse_depth_event(self, payload: dict[str, Any], local_ts_ns: int) -> tuple[dict[str, Any], list[dict]] | None:
        data = payload.get("data", payload)
        if data.get("e") != "depthUpdate":
            return None
        metadata = {
            "first": int(data["U"]),
            "last": int(data["u"]),
            "previous": int(data["pu"]) if data.get("pu") is not None else None,
        }
        event_id = self.next_event_id()
        common = {
            "exchange": self.exchange,
            "market": self.market,
            "symbol": data.get("s") or self.symbol,
            "channel": "depth",
            "event_type": "depth_update",
            "event_id": event_id,
            "local_ts_ns": local_ts_ns,
            "exchange_ts_ns": ms_to_ns(data.get("E")),
            "engine_ts_ns": ms_to_ns(data.get("T")),
            "sequence": str(metadata["last"]),
            "sequence_start": str(metadata["first"]),
            "prev_sequence": str(metadata["previous"]) if metadata["previous"] is not None else None,
            "is_snapshot": False,
        }
        records = []
        row_idx = 0
        for side, key in (("bid", "b"), ("ask", "a")):
            for price, qty in data.get(key, []):
                quantity = float(qty)
                records.append(
                    {
                        **common,
                        "row_idx": row_idx,
                        "side": side,
                        "price": float(price),
                        "quantity": quantity,
                        "action": "delete" if quantity == 0.0 else "set",
                        "raw_message": self.raw_once(data, row_idx),
                    }
                )
                row_idx += 1
        return metadata, records

    def parse_depth_snapshot(self, snapshot: dict[str, Any], local_ts_ns: int) -> tuple[int, list[dict]]:
        last_update_id = int(snapshot["lastUpdateId"])
        event_id = self.next_event_id()
        common = {
            "exchange": self.exchange,
            "market": self.market,
            "symbol": self.symbol,
            "channel": "depth",
            "event_type": "snapshot",
            "event_id": event_id,
            "local_ts_ns": local_ts_ns,
            "exchange_ts_ns": ms_to_ns(snapshot.get("E")),
            "engine_ts_ns": ms_to_ns(snapshot.get("T")),
            "sequence": str(last_update_id),
            "sequence_start": None,
            "prev_sequence": None,
            "is_snapshot": True,
        }
        records = []
        row_idx = 0
        for side, key in (("bid", "bids"), ("ask", "asks")):
            for price, qty in snapshot.get(key, []):
                records.append(
                    {
                        **common,
                        "row_idx": row_idx,
                        "side": side,
                        "price": float(price),
                        "quantity": float(qty),
                        "action": "set",
                        "raw_message": self.raw_once(snapshot, row_idx),
                    }
                )
                row_idx += 1
        return last_update_id, records

    def accepts_first_diff(self, snapshot_id: int, metadata: dict[str, Any]) -> bool:
        target = snapshot_id if not self.is_spot else snapshot_id + 1
        return metadata["first"] <= target <= metadata["last"]

    def validate_next_diff(self, previous_id: int, metadata: dict[str, Any]) -> None:
        if self.is_spot:
            if metadata["first"] > previous_id + 1:
                raise BinanceSequenceGap(
                    f"spot sequence gap: expected <= {previous_id + 1}, got U={metadata['first']} u={metadata['last']}"
                )
        elif metadata["previous"] != previous_id:
            raise BinanceSequenceGap(
                f"USDT-M sequence gap: expected pu={previous_id}, got pu={metadata['previous']} "
                f"U={metadata['first']} u={metadata['last']}"
            )

    async def _fetch_depth_snapshot(self, session: aiohttp.ClientSession) -> tuple[int, list[dict]]:
        configured = int(self.config.get("snapshot_limit", 5000 if self.is_spot else 1000))
        limit = min(configured, 5000 if self.is_spot else 1000)
        async with session.get(
            self.rest_depth_url,
            params={"symbol": self.symbol.upper(), "limit": limit},
            timeout=20,
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        return self.parse_depth_snapshot(payload, now_ns())

    async def depth(self) -> None:
        speed = str(self.config.get("depth_speed", "100ms"))
        idle_timeout = max(1.0, float(self.config.get("depth_idle_timeout_seconds", 15.0)))
        stream = f"{self.symbol.lower()}@depth@{speed}"
        url = f"{self.ws_base_url}?streams={stream}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "depth")
        connection_key = f"{self.exchange}:{self.market}:{self.symbol}:depth"
        reconnect_delay = 1.0

        while not self.stop_event.is_set():
            reader_task: asyncio.Task | None = None
            try:
                metrics.set_connection(connection_key, "connecting")
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        url, heartbeat=20, timeout=30, max_msg_size=64 * 1024 * 1024
                    ) as ws:
                        queue: asyncio.Queue[tuple[dict, int] | BaseException] = asyncio.Queue()

                        async def read_messages() -> None:
                            try:
                                async for message in ws:
                                    if message.type == aiohttp.WSMsgType.TEXT:
                                        await queue.put((json.loads(message.data), now_ns()))
                                    elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                        break
                            except BaseException as exc:
                                await queue.put(exc)
                            finally:
                                await queue.put(ConnectionError("Binance depth websocket closed"))

                        reader_task = asyncio.create_task(read_messages())
                        snapshot_id, snapshot_records = await self._fetch_depth_snapshot(session)
                        await writer.write_many(snapshot_records)

                        previous_id = snapshot_id
                        bridged = False
                        while not self.stop_event.is_set():
                            item = await next_depth_item(queue, idle_timeout)
                            if isinstance(item, BaseException):
                                raise item
                            payload, local_ts_ns = item
                            parsed = self.parse_depth_event(payload, local_ts_ns)
                            if parsed is None:
                                continue
                            metadata, records = parsed
                            if not bridged:
                                stale = metadata["last"] <= snapshot_id if self.is_spot else metadata["last"] < snapshot_id
                                if stale:
                                    continue
                                if not self.accepts_first_diff(snapshot_id, metadata):
                                    raise BinanceSequenceGap(
                                        f"cannot bridge snapshot {snapshot_id} with U={metadata['first']} u={metadata['last']}"
                                    )
                                bridged = True
                            else:
                                if metadata["last"] <= previous_id:
                                    continue
                                self.validate_next_diff(previous_id, metadata)
                            await writer.write_many(records)
                            previous_id = metadata["last"]
                            reconnect_delay = 1.0
                            metrics.set_connection(connection_key, "connected", message_seen=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"binance {self.market} {self.symbol} depth error: {exc!r}; rebuilding snapshot", flush=True)
                metrics.set_connection(connection_key, "reconnecting", error=repr(exc))
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)
            finally:
                if reader_task is not None:
                    reader_task.cancel()
                    await asyncio.gather(reader_task, return_exceptions=True)

    def parse_trade_event(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        data = payload.get("data", payload)
        if data.get("e") != "trade":
            return []
        buyer_is_maker = bool(data.get("m"))
        return [
            {
                "exchange": self.exchange,
                "market": self.market,
                "symbol": data.get("s") or self.symbol,
                "channel": "trades",
                "event_type": "trade",
                "event_id": self.next_event_id(),
                "row_idx": 0,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": ms_to_ns(data.get("E")),
                "engine_ts_ns": ms_to_ns(data.get("T")),
                "sequence": str(data.get("t")),
                "trade_id": str(data.get("t")),
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "taker_side": "sell" if buyer_is_maker else "buy",
                "raw_message": self.raw_once(data, 0),
            }
        ]

    async def trades(self) -> None:
        stream = f"{self.symbol.lower()}@trade"
        url = f"{self.ws_base_url}?streams={stream}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "trades")

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_trade_event(payload, local_ts_ns))

        await self.ws_json_loop(url, on_message)

    def parse_mark_price_event(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        data = payload.get("data", payload)
        if data.get("e") not in (None, "markPriceUpdate"):
            return []
        mark_price = data.get("p", data.get("markPrice"))
        index_price = data.get("i", data.get("indexPrice"))
        if mark_price is None or index_price is None:
            return []
        event_ts = data.get("E", data.get("time"))
        return [
            {
                "exchange": self.exchange,
                "market": self.market,
                "symbol": data.get("s") or self.symbol,
                "channel": "markPrice",
                "event_type": "derivative_context",
                "event_id": self.next_event_id(),
                "row_idx": 0,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": ms_to_ns(event_ts),
                "engine_ts_ns": ms_to_ns(event_ts),
                "sequence": str(event_ts) if event_ts is not None else None,
                "mark_price": float(mark_price),
                "index_price": float(index_price),
                "funding_rate": float(data.get("r", data.get("lastFundingRate")))
                if data.get("r", data.get("lastFundingRate")) not in (None, "")
                else None,
                "next_funding_ts_ns": ms_to_ns(data.get("T", data.get("nextFundingTime"))),
                "raw_message": self.raw_once(data, 0),
            }
        ]

    async def mark_price(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "markPrice")
        interval = float(self.config.get("mark_price_poll_seconds", 1.0))
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        connection_key = f"{self.exchange}:{self.market}:{self.symbol}:markPrice-rest"
        reconnect_delay = 1.0
        while not self.stop_event.is_set():
            try:
                metrics.set_connection(connection_key, "connecting")
                async with aiohttp.ClientSession() as session:
                    while not self.stop_event.is_set():
                        async with session.get(
                            url,
                            params={"symbol": self.symbol.upper()},
                            timeout=20,
                        ) as response:
                            response.raise_for_status()
                            payload = await response.json()
                        await writer.write_many(self.parse_mark_price_event(payload, now_ns()))
                        metrics.set_connection(connection_key, "connected", message_seen=True)
                        reconnect_delay = 1.0
                        try:
                            await asyncio.wait_for(self.stop_event.wait(), timeout=max(0.25, interval))
                        except asyncio.TimeoutError:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"binance {self.symbol} mark price error: {exc!r}; reconnecting", flush=True)
                metrics.set_connection(connection_key, "reconnecting", error=repr(exc))
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)

    def parse_force_order_event(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        data = payload.get("data", payload)
        if data.get("e") != "forceOrder" or not isinstance(data.get("o"), dict):
            return []
        order = data["o"]
        order_side = str(order.get("S", "")).upper()
        average_price = float(order.get("ap") or 0.0)
        order_price = float(order.get("p") or 0.0)
        filled_quantity = float(order.get("z") or 0.0)
        original_quantity = float(order.get("q") or 0.0)
        event_ts = data.get("E")
        trade_ts = order.get("T")
        return [
            {
                "exchange": self.exchange,
                "market": self.market,
                "symbol": order.get("s") or self.symbol,
                "channel": "forceOrder",
                "event_type": "liquidation",
                "event_id": self.next_event_id(),
                "row_idx": 0,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": ms_to_ns(event_ts),
                "engine_ts_ns": ms_to_ns(trade_ts),
                "sequence": str(trade_ts) if trade_ts is not None else None,
                "price": average_price if average_price > 0.0 else order_price,
                "quantity": filled_quantity if filled_quantity > 0.0 else original_quantity,
                "taker_side": order_side.lower() or None,
                "liquidation_side": "long" if order_side == "SELL" else "short" if order_side == "BUY" else None,
                "raw_message": self.raw_once(data, 0),
            }
        ]

    async def force_order(self) -> None:
        stream = f"{self.symbol.lower()}@forceOrder"
        url = f"{self.ws_base_url}?streams={stream}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "forceOrder")

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_force_order_event(payload, local_ts_ns))

        await self.ws_json_loop(url, on_message)
