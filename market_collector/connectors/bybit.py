from __future__ import annotations

from market_collector.timeutil import ms_to_ns

from .base import BaseConnector


class BybitConnector(BaseConnector):
    exchange = "bybit"

    def tasks(self) -> list:
        tasks = []
        channels = set(self.config.get("channels", []))
        if "orderbook" in channels:
            tasks.append(self.orderbook())
        if "trades" in channels:
            tasks.append(self.trades())
        if "tickers" in channels or "ticker" in channels:
            tasks.append(self.tickers())
        if "allLiquidation" in channels or "liquidations" in channels:
            tasks.append(self.all_liquidation())
        return tasks

    @property
    def url(self) -> str:
        category = self.market or "linear"
        return f"wss://stream.bybit.com/v5/public/{category}"

    async def orderbook(self) -> None:
        depth = int(self.config.get("orderbook_depth", 1000))
        topic = f"orderbook.{depth}.{self.symbol}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "orderbook")
        subscribe = {"op": "subscribe", "args": [topic]}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if payload.get("op"):
                print(f"bybit event: {payload}", flush=True)
                return
            if payload.get("topic") != topic:
                return
            data = payload.get("data", {})
            event_type = payload.get("type", "delta")
            event_id = self.next_event_id()
            common = {
                "exchange": self.exchange,
                "market": self.market,
                "symbol": self.symbol,
                "channel": "orderbook",
                "event_type": event_type,
                "event_id": event_id,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": ms_to_ns(payload.get("ts")),
                "engine_ts_ns": ms_to_ns(data.get("cts")),
                "sequence": str(data.get("u") or data.get("seq") or ""),
                "sequence_start": None,
                "prev_sequence": str(data.get("seq") or ""),
                "is_snapshot": event_type == "snapshot",
            }
            records = []
            row_idx = 0
            for side, key in (("bid", "b"), ("ask", "a")):
                for level in data.get(key, []):
                    quantity = float(level[1])
                    records.append(
                        {
                            **common,
                            "row_idx": row_idx,
                            "side": side,
                            "price": float(level[0]),
                            "quantity": quantity,
                            "action": "delete" if quantity == 0.0 else "set",
                            "raw_message": self.raw_once(payload, row_idx),
                        }
                    )
                    row_idx += 1
            await writer.write_many(records)

        await self.ws_json_loop(self.url, on_message, subscribe)

    async def trades(self) -> None:
        topic = f"publicTrade.{self.symbol}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "trades")
        subscribe = {"op": "subscribe", "args": [topic]}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            if payload.get("op"):
                print(f"bybit event: {payload}", flush=True)
                return
            if payload.get("topic") != topic:
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
                        "exchange_ts_ns": ms_to_ns(item.get("T")),
                        "engine_ts_ns": ms_to_ns(item.get("T")),
                        "sequence": str(item.get("i") or ""),
                        "trade_id": str(item.get("i") or ""),
                        "price": float(item.get("p")),
                        "quantity": float(item.get("v")),
                        "taker_side": str(item.get("S", "")).lower(),
                        "raw_message": self.raw_once(payload, 0),
                    }
                )
            await writer.write_many(records)

        await self.ws_json_loop(self.url, on_message, subscribe)

    @staticmethod
    def _optional_float(value) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    def parse_ticker_message(self, payload: dict, local_ts_ns: int) -> list[dict]:
        if not str(payload.get("topic", "")).startswith("tickers."):
            return []
        data = payload.get("data", {})
        if not isinstance(data, dict):
            return []
        timestamp = payload.get("ts")
        next_funding_time = data.get("nextFundingTime")
        return [
            {
                "exchange": self.exchange,
                "market": self.market,
                "symbol": data.get("symbol") or self.symbol,
                "channel": "tickers",
                "event_type": "derivative_context",
                "event_id": self.next_event_id(),
                "row_idx": 0,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": ms_to_ns(timestamp),
                "engine_ts_ns": ms_to_ns(timestamp),
                "sequence": str(payload.get("cs")) if payload.get("cs") is not None else None,
                "mark_price": self._optional_float(data.get("markPrice")),
                "index_price": self._optional_float(data.get("indexPrice")),
                "open_interest": self._optional_float(data.get("openInterest")),
                "funding_rate": self._optional_float(data.get("fundingRate")),
                "next_funding_ts_ns": ms_to_ns(next_funding_time),
                "raw_message": self.raw_once(payload, 0),
            }
        ]

    async def tickers(self) -> None:
        topic = f"tickers.{self.symbol}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "tickers")
        subscribe = {"op": "subscribe", "args": [topic]}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_ticker_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, subscribe)

    def parse_all_liquidation_message(self, payload: dict, local_ts_ns: int) -> list[dict]:
        if not str(payload.get("topic", "")).startswith("allLiquidation."):
            return []
        items = payload.get("data", [])
        if not isinstance(items, list):
            return []
        event_id = self.next_event_id()
        records = []
        for row_idx, item in enumerate(items):
            position_side = str(item.get("S", "")).upper()
            timestamp = item.get("T")
            records.append(
                {
                    "exchange": self.exchange,
                    "market": self.market,
                    "symbol": item.get("s") or self.symbol,
                    "channel": "allLiquidation",
                    "event_type": "liquidation",
                    "event_id": event_id,
                    "row_idx": row_idx,
                    "local_ts_ns": local_ts_ns,
                    "exchange_ts_ns": ms_to_ns(payload.get("ts")),
                    "engine_ts_ns": ms_to_ns(timestamp),
                    "sequence": str(timestamp) if timestamp is not None else None,
                    "price": float(item["p"]),
                    "quantity": float(item["v"]),
                    "taker_side": "sell" if position_side == "BUY" else "buy" if position_side == "SELL" else None,
                    "liquidation_side": "long" if position_side == "BUY" else "short" if position_side == "SELL" else None,
                    "raw_message": self.raw_once(payload, row_idx),
                }
            )
        return records

    async def all_liquidation(self) -> None:
        topic = f"allLiquidation.{self.symbol}"
        writer = self.writers.get(self.exchange, self.market, self.symbol, "allLiquidation")
        subscribe = {"op": "subscribe", "args": [topic]}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_all_liquidation_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, subscribe)
