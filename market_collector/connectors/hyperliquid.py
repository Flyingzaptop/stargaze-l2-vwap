from __future__ import annotations

from typing import Any

from market_collector.timeutil import ms_to_ns

from .base import BaseConnector


class HyperliquidConnector(BaseConnector):
    exchange = "hyperliquid"
    url = "wss://api.hyperliquid.xyz/ws"

    @property
    def coin(self) -> str:
        symbol = self.symbol.upper()
        for suffix in ("-PERPETUAL", "-PERP", "PERPETUAL", "PERP", "USDT", "USD"):
            if symbol.endswith(suffix):
                symbol = symbol[: -len(suffix)]
                break
        return symbol or "BTC"

    def tasks(self) -> list:
        tasks = []
        channels = set(self.config.get("channels", []))
        if "l2Book" in channels or "book" in channels:
            tasks.append(self.l2_book())
        if "trades" in channels:
            tasks.append(self.trades())
        if "activeAssetCtx" in channels or "asset_context" in channels:
            tasks.append(self.active_asset_context())
        return tasks

    def parse_l2_book_message(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        if payload.get("channel") != "l2Book":
            return []
        data = payload.get("data", {})
        levels = data.get("levels", [[], []])
        timestamp = data.get("time")
        event_id = self.next_event_id()
        common = {
            "exchange": self.exchange,
            "market": self.market,
            "symbol": data.get("coin") or self.coin,
            "channel": "l2Book",
            "event_type": "snapshot",
            "event_id": event_id,
            "local_ts_ns": local_ts_ns,
            "exchange_ts_ns": ms_to_ns(timestamp),
            "engine_ts_ns": ms_to_ns(timestamp),
            "sequence": str(timestamp) if timestamp is not None else None,
            "is_snapshot": True,
        }
        records = []
        row_idx = 0
        for side, side_levels in zip(("bid", "ask"), levels):
            for level in side_levels:
                records.append(
                    {
                        **common,
                        "row_idx": row_idx,
                        "side": side,
                        "price": float(level["px"]),
                        "quantity": float(level["sz"]),
                        "action": "set",
                        "order_count": float(level["n"]) if level.get("n") is not None else None,
                        "raw_message": self.raw_once(payload, row_idx),
                    }
                )
                row_idx += 1
        return records

    async def l2_book(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "l2Book")
        subscription: dict[str, Any] = {"type": "l2Book", "coin": self.coin}
        if self.config.get("n_sig_figs") is not None:
            subscription["nSigFigs"] = int(self.config["n_sig_figs"])
        if self.config.get("mantissa") is not None:
            subscription["mantissa"] = int(self.config["mantissa"])
        subscribe = {"method": "subscribe", "subscription": subscription}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_l2_book_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, subscribe)

    def parse_trades_message(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        if payload.get("channel") != "trades":
            return []
        trades = payload.get("data", [])
        event_id = self.next_event_id()
        records = []
        for row_idx, trade in enumerate(trades):
            timestamp = trade.get("time")
            trade_id = trade.get("tid")
            coin = trade.get("coin") or self.coin
            unique_id = f"{timestamp}:{coin}:{trade_id}"
            side = str(trade.get("side", "")).upper()
            records.append(
                {
                    "exchange": self.exchange,
                    "market": self.market,
                    "symbol": coin,
                    "channel": "trades",
                    "event_type": "trade",
                    "event_id": event_id,
                    "row_idx": row_idx,
                    "local_ts_ns": local_ts_ns,
                    "exchange_ts_ns": ms_to_ns(timestamp),
                    "engine_ts_ns": ms_to_ns(timestamp),
                    "sequence": str(trade_id) if trade_id is not None else None,
                    "trade_id": unique_id,
                    "price": float(trade["px"]),
                    "quantity": float(trade["sz"]),
                    "taker_side": "buy" if side == "B" else "sell" if side == "A" else None,
                    "raw_message": self.raw_once(payload, row_idx),
                }
            )
        return records

    async def trades(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "trades")
        subscribe = {"method": "subscribe", "subscription": {"type": "trades", "coin": self.coin}}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_trades_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, subscribe)

    def parse_active_asset_context_message(self, payload: dict[str, Any], local_ts_ns: int) -> list[dict]:
        if payload.get("channel") != "activeAssetCtx":
            return []
        data = payload.get("data", {})
        context = data.get("ctx", {})
        return [
            {
                "exchange": self.exchange,
                "market": self.market,
                "symbol": data.get("coin") or self.coin,
                "channel": "activeAssetCtx",
                "event_type": "derivative_context",
                "event_id": self.next_event_id(),
                "row_idx": 0,
                "local_ts_ns": local_ts_ns,
                "exchange_ts_ns": None,
                "engine_ts_ns": None,
                "sequence": None,
                "mark_price": float(context["markPx"]) if context.get("markPx") is not None else None,
                "oracle_price": float(context["oraclePx"]) if context.get("oraclePx") is not None else None,
                "open_interest": float(context["openInterest"]) if context.get("openInterest") is not None else None,
                "funding_rate": float(context["funding"]) if context.get("funding") is not None else None,
                "raw_message": self.raw_once(payload, 0),
            }
        ]

    async def active_asset_context(self) -> None:
        writer = self.writers.get(self.exchange, self.market, self.symbol, "activeAssetCtx")
        subscribe = {"method": "subscribe", "subscription": {"type": "activeAssetCtx", "coin": self.coin}}

        async def on_message(payload: dict, local_ts_ns: int) -> None:
            await writer.write_many(self.parse_active_asset_context_message(payload, local_ts_ns))

        await self.ws_json_loop(self.url, on_message, subscribe)
