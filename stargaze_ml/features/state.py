from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import log1p
from bisect import bisect_left, insort

import numpy as np

from ..contracts import Packet, VENUES, VENUE_INDEX


DEPTHS = (1, 5, 10, 25, 50, 100, 250, 1000)
VENUE_FEATURE_NAMES = (
    "book_valid",
    "stale_ms",
    "bid",
    "ask",
    "mid",
    "spread_bps",
    "microprice_delta_bps",
    "best_bid_log_qty",
    "best_ask_log_qty",
    *(name for depth in DEPTHS for name in (f"bid_log_depth_{depth}", f"ask_log_depth_{depth}", f"imbalance_{depth}")),
    "bid_slope_10_bps",
    "ask_slope_10_bps",
    "trade_count",
    "trade_buy_log_qty",
    "trade_sell_log_qty",
    "trade_signed_log_qty",
    "trade_vwap_delta_bps",
    "mark_delta_bps",
    "index_delta_bps",
    "oracle_delta_bps",
    "open_interest_log",
    "funding_rate_bps",
    "seconds_to_funding_log",
    "long_liquidation_log_qty",
    "short_liquidation_log_qty",
)


def _side(value: object) -> str:
    text = str(value or "").lower()
    if text in {"bid", "buy", "b"}:
        return "bid"
    if text in {"ask", "offer", "sell", "a"}:
        return "ask"
    return ""


@dataclass
class BookState:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    bid_orders: dict[float, float] = field(default_factory=dict)
    ask_orders: dict[float, float] = field(default_factory=dict)
    warm: bool = False
    last_update_ns: int = -1
    _dirty: bool = True
    _bid_prices: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)
    _bid_qty: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)
    _ask_prices: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)
    _ask_qty: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)

    def apply(self, packet: Packet, *, quantity_scale: float = 1.0) -> bool:
        cols = packet.columns
        snapshot = bool(np.any(cols.get("is_snapshot", np.zeros(packet.size, dtype=bool)))) or any(
            str(x).lower() == "snapshot" for x in cols.get("event_type", np.empty(0, dtype=object))
        )
        if snapshot:
            self.bids.clear()
            self.asks.clear()
            self.bid_orders.clear()
            self.ask_orders.clear()
            self.warm = True
        prices = cols.get("price", np.full(packet.size, np.nan))
        quantities = cols.get("quantity", np.full(packet.size, np.nan))
        sides = cols.get("side", np.full(packet.size, "", dtype=object))
        actions = cols.get("action", np.full(packet.size, "set", dtype=object))
        order_counts = cols.get("order_count", np.full(packet.size, np.nan))
        for price_raw, qty_raw, side_raw, action_raw, count_raw in zip(prices, quantities, sides, actions, order_counts, strict=True):
            side = _side(side_raw)
            if not side or not np.isfinite(price_raw):
                continue
            price = float(price_raw)
            qty = float(qty_raw) * float(quantity_scale) if np.isfinite(qty_raw) else 0.0
            levels = self.bids if side == "bid" else self.asks
            counts = self.bid_orders if side == "bid" else self.ask_orders
            action = str(action_raw or "set").lower()
            if action == "delete" or qty <= 0.0:
                levels.pop(price, None)
                counts.pop(price, None)
            else:
                levels[price] = qty
                if np.isfinite(count_raw):
                    counts[price] = float(count_raw)
        self.last_update_ns = packet.local_ts_ns
        self._dirty = True
        return snapshot

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._dirty:
            def ordered(levels: dict[float, float], *, descending: bool) -> tuple[np.ndarray, np.ndarray]:
                size = len(levels)
                if size == 0:
                    return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
                prices = np.fromiter(levels.keys(), dtype=np.float64, count=size)
                quantities = np.fromiter(levels.values(), dtype=np.float64, count=size)
                limit = min(1_000, size)
                if size > limit:
                    if descending:
                        chosen = np.argpartition(prices, size - limit)[size - limit :]
                    else:
                        chosen = np.argpartition(prices, limit - 1)[:limit]
                    prices, quantities = prices[chosen], quantities[chosen]
                order = np.argsort(prices)
                if descending:
                    order = order[::-1]
                return prices[order], quantities[order]

            self._bid_prices, self._bid_qty = ordered(self.bids, descending=True)
            self._ask_prices, self._ask_qty = ordered(self.asks, descending=False)
            self._dirty = False
        return self._bid_prices, self._bid_qty, self._ask_prices, self._ask_qty

    def bbo(self) -> tuple[float, float]:
        bp, _, ap, _ = self.arrays()
        if not self.warm or bp.size == 0 or ap.size == 0 or bp[0] >= ap[0]:
            return np.nan, np.nan
        return float(bp[0]), float(ap[0])


@dataclass
class TradeTick:
    count: int = 0
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    notional: float = 0.0
    price_qty: float = 0.0
    last_update_ns: int = -1

    def reset(self) -> None:
        self.count = 0
        self.buy_qty = 0.0
        self.sell_qty = 0.0
        self.notional = 0.0
        self.price_qty = 0.0


@dataclass
class DerivativeTick:
    mark_price: float = np.nan
    index_price: float = np.nan
    oracle_price: float = np.nan
    open_interest: float = np.nan
    funding_rate: float = np.nan
    next_funding_ts_ns: int = -1
    last_update_ns: int = -1
    long_liquidation_qty: float = 0.0
    short_liquidation_qty: float = 0.0

    def apply_context(self, packet: Packet) -> None:
        cols = packet.columns
        for name in ("mark_price", "index_price", "oracle_price", "open_interest", "funding_rate"):
            values = cols.get(name)
            if values is None:
                continue
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                setattr(self, name, float(finite[-1]))
        funding_ts = cols.get("next_funding_ts_ns")
        if funding_ts is not None:
            valid = np.asarray(funding_ts, dtype=np.int64)
            valid = valid[valid > 0]
            if valid.size:
                self.next_funding_ts_ns = int(valid[-1])
        self.last_update_ns = packet.local_ts_ns

    def apply_liquidations(self, packet: Packet, *, quantity_scale: float) -> None:
        sides = packet.columns.get("liquidation_side", np.full(packet.size, "", dtype=object))
        quantities = packet.columns.get("quantity", np.zeros(packet.size, dtype=np.float64))
        for side_raw, qty_raw in zip(sides, quantities, strict=True):
            if not np.isfinite(qty_raw):
                continue
            side = str(side_raw or "").lower()
            qty = max(0.0, float(qty_raw) * float(quantity_scale))
            if side in {"long", "buy", "bid"}:
                self.long_liquidation_qty += qty
            elif side in {"short", "sell", "ask"}:
                self.short_liquidation_qty += qty

    def tick_features(self, mid: float, ts_ns: int) -> tuple[float, ...]:
        def delta(price: float) -> float:
            return 1e4 * (price / mid - 1.0) if mid > 0.0 and np.isfinite(price) else 0.0

        to_funding = max(0.0, (self.next_funding_ts_ns - ts_ns) / 1e9) if self.next_funding_ts_ns > 0 else 0.0
        result = (
            delta(self.mark_price),
            delta(self.index_price),
            delta(self.oracle_price),
            log1p(max(self.open_interest, 0.0)) if np.isfinite(self.open_interest) else 0.0,
            1e4 * self.funding_rate if np.isfinite(self.funding_rate) else 0.0,
            log1p(to_funding),
            log1p(self.long_liquidation_qty),
            log1p(self.short_liquidation_qty),
        )
        self.long_liquidation_qty = 0.0
        self.short_liquidation_qty = 0.0
        return result


@dataclass
class L3State:
    orders: dict[str, tuple[str, float, float]] = field(default_factory=dict)
    bid_levels: dict[float, set[str]] = field(default_factory=dict)
    ask_levels: dict[float, set[str]] = field(default_factory=dict)
    bid_prices: list[float] = field(default_factory=list)
    ask_prices: list[float] = field(default_factory=list)
    active_bid_qty: float = 0.0
    active_ask_qty: float = 0.0
    adds: int = 0
    deletes: int = 0
    modifies: int = 0
    add_qty: float = 0.0
    delete_qty: float = 0.0

    def _remove(self, order_id: str) -> tuple[str, float, float] | None:
        old = self.orders.pop(order_id, None)
        if old is not None:
            levels = self.bid_levels if old[0] == "bid" else self.ask_levels
            prices = self.bid_prices if old[0] == "bid" else self.ask_prices
            members = levels.get(old[1])
            if members is not None:
                members.discard(order_id)
                if not members:
                    levels.pop(old[1], None)
                    pos = bisect_left(prices, old[1])
                    if pos < len(prices) and prices[pos] == old[1]:
                        prices.pop(pos)
            if old[0] == "bid":
                self.active_bid_qty -= old[2]
            else:
                self.active_ask_qty -= old[2]
        return old

    def _insert(self, order_id: str, side: str, price: float, qty: float) -> None:
        self.orders[order_id] = (side, price, qty)
        levels = self.bid_levels if side == "bid" else self.ask_levels
        prices = self.bid_prices if side == "bid" else self.ask_prices
        if price not in levels:
            levels[price] = set()
            insort(prices, price)
        levels[price].add(order_id)
        if side == "bid":
            self.active_bid_qty += qty
        else:
            self.active_ask_qty += qty

    def apply(self, packet: Packet) -> bool:
        cols = packet.columns
        snapshot = bool(np.any(cols.get("is_snapshot", np.zeros(packet.size, dtype=bool)))) or any(
            str(x).lower() == "snapshot" for x in cols.get("event_type", np.empty(0, dtype=object))
        )
        if snapshot:
            self.orders.clear()
            self.bid_levels.clear()
            self.ask_levels.clear()
            self.bid_prices.clear()
            self.ask_prices.clear()
            self.active_bid_qty = self.active_ask_qty = 0.0
        ids = cols.get("order_id", np.full(packet.size, "", dtype=object))
        sides = cols.get("side", np.full(packet.size, "", dtype=object))
        prices = cols.get("price", np.full(packet.size, np.nan))
        quantities = cols.get("quantity", np.full(packet.size, np.nan))
        actions = cols.get("action", np.full(packet.size, "set", dtype=object))
        for oid_raw, side_raw, price_raw, qty_raw, action_raw in zip(ids, sides, prices, quantities, actions, strict=True):
            oid = str(oid_raw or "")
            if not oid:
                continue
            action = str(action_raw or "set").lower()
            old = self._remove(oid)
            if action == "delete":
                self.deletes += 1
                if old is not None:
                    self.delete_qty += old[2]
                continue
            side = _side(side_raw) or (old[0] if old is not None else "")
            price = float(price_raw) if np.isfinite(price_raw) else (old[1] if old is not None else np.nan)
            qty = float(qty_raw) if np.isfinite(qty_raw) else (old[2] if old is not None else np.nan)
            if not side or not np.isfinite(price) or not np.isfinite(qty) or qty <= 0.0:
                continue
            self._insert(oid, side, price, qty)
            if old is None:
                self.adds += 1
                self.add_qty += qty
            else:
                self.modifies += 1
        self._truncate_depth(1_000)
        return snapshot

    def _truncate_depth(self, depth: int) -> None:
        while len(self.bid_prices) > int(depth):
            self._drop_level("bid", self.bid_prices[0])
        while len(self.ask_prices) > int(depth):
            self._drop_level("ask", self.ask_prices[-1])

    def _drop_level(self, side: str, price: float) -> None:
        levels = self.bid_levels if side == "bid" else self.ask_levels
        members = tuple(levels.get(price, ()))
        for order_id in members:
            self._remove(order_id)

    def tick_features(self) -> np.ndarray:
        total = self.active_bid_qty + self.active_ask_qty
        imbalance = (self.active_bid_qty - self.active_ask_qty) / max(total, 1e-12)
        values = np.asarray(
            [
                log1p(len(self.orders)), log1p(max(self.active_bid_qty, 0.0)), log1p(max(self.active_ask_qty, 0.0)), imbalance,
                log1p(self.adds), log1p(self.deletes), log1p(self.modifies), log1p(self.add_qty), log1p(self.delete_qty),
            ],
            dtype=np.float32,
        )
        self.adds = self.deletes = self.modifies = 0
        self.add_qty = self.delete_qty = 0.0
        return values


class MarketState:
    def __init__(self, *, okx_contract_btc: float = 0.01, cadence_ms: int = 100) -> None:
        self.books = {venue: BookState() for venue in VENUES}
        self.trades = {venue: TradeTick() for venue in VENUES}
        self.derivatives = {venue: DerivativeTick() for venue in VENUES}
        self.l3 = L3State()
        self.seen_trade_ids: dict[str, set[str]] = {venue: set() for venue in VENUES}
        self.seen_trade_order: dict[str, deque[str]] = {venue: deque() for venue in VENUES}
        self.okx_contract_btc = float(okx_contract_btc)
        self.cadence_ms = int(cadence_ms)
        if self.cadence_ms <= 0:
            raise ValueError("cadence_ms must be positive")
        self.segment_id = 0
        self.mid_history: deque[float] = deque(maxlen=10_000)

    @staticmethod
    def _collapse_book_packets(packets: list[Packet]) -> Packet:
        last_snapshot = -1
        for index, packet in enumerate(packets):
            if bool(np.any(packet.columns.get("is_snapshot", np.zeros(packet.size, dtype=bool)))):
                last_snapshot = index
        selected_packets = packets[last_snapshot:] if last_snapshot >= 0 else packets
        required = (
            "is_snapshot",
            "event_type",
            "side",
            "price",
            "quantity",
            "action",
            "order_count",
        )
        names = tuple(name for name in required if name in selected_packets[0].columns)
        columns = {name: np.concatenate([packet.columns[name] for packet in selected_packets]) for name in names}
        prices = np.asarray(columns.get("price", np.empty(0)), dtype=np.float64)
        raw_sides = np.asarray(columns.get("side", np.empty(0, dtype=object)), dtype=object)
        sides = np.char.lower(raw_sides.astype(str))
        bid = np.isin(sides, ("bid", "buy", "b"))
        ask = np.isin(sides, ("ask", "offer", "sell", "a"))
        usable = np.isfinite(prices) & (bid | ask)
        original = np.flatnonzero(usable)
        if original.size:
            side_code = ask[usable].astype(np.int8)
            ordered = np.lexsort((original, prices[usable], side_code))
            sorted_original = original[ordered]
            sorted_side = side_code[ordered]
            sorted_price = prices[usable][ordered]
            group_end = np.r_[
                (sorted_side[1:] != sorted_side[:-1]) | (sorted_price[1:] != sorted_price[:-1]),
                True,
            ]
            keep = np.sort(sorted_original[group_end])
            columns = {name: values[keep] for name, values in columns.items()}
        if last_snapshot >= 0 and len(next(iter(columns.values()), ())):
            snapshot = np.zeros(len(next(iter(columns.values()))), dtype=bool)
            snapshot[0] = True
            columns["is_snapshot"] = snapshot
        return Packet(selected_packets[-1].stream, selected_packets[-1].local_ts_ns, columns)

    def apply_tick(self, packets: list[Packet]) -> None:
        books: dict[str, list[Packet]] = {}
        for packet in packets:
            if packet.stream.kind == "book":
                books.setdefault(packet.stream.venue, []).append(packet)
            else:
                self.apply(packet)
        for venue_packets in books.values():
            self.apply(self._collapse_book_packets(venue_packets))

    def apply(self, packet: Packet) -> None:
        venue = packet.stream.venue
        scale = self.okx_contract_btc if venue == "okx_perpetual" else 1.0
        if packet.stream.kind == "book":
            was_warm = self.books[venue].warm
            snapshot = self.books[venue].apply(packet, quantity_scale=scale)
            if venue == "binance_perpetual" and snapshot and was_warm:
                self.segment_id += 1
            return
        if packet.stream.kind == "l3":
            self.l3.apply(packet)
            return
        if packet.stream.kind == "context":
            self.derivatives[venue].apply_context(packet)
            return
        if packet.stream.kind == "liquidation":
            self.derivatives[venue].apply_liquidations(packet, quantity_scale=scale)
            return
        cols = packet.columns
        ids = cols.get("trade_id", np.full(packet.size, "", dtype=object))
        prices = cols.get("price", np.full(packet.size, np.nan))
        quantities = cols.get("quantity", np.full(packet.size, np.nan))
        takers = cols.get("taker_side", np.full(packet.size, "", dtype=object))
        makers = cols.get("side", np.full(packet.size, "", dtype=object))
        tick = self.trades[venue]
        seen = self.seen_trade_ids[venue]
        seen_order = self.seen_trade_order[venue]
        finite = np.isfinite(prices) & np.isfinite(quantities)
        if not np.any(finite):
            return
        price = np.asarray(prices[finite], dtype=np.float64)
        qty = np.asarray(quantities[finite], dtype=np.float64) * scale
        taker_values = np.asarray([_side(value) for value in takers[finite]], dtype=object)
        if venue == "coinbase_spot":
            maker_values = np.asarray([_side(value) for value in makers[finite]], dtype=object)
            missing = taker_values == ""
            taker_values[missing & (maker_values == "bid")] = "ask"
            taker_values[missing & (maker_values == "ask")] = "bid"
        tick.buy_qty += float(np.sum(qty[taker_values == "bid"]))
        tick.sell_qty += float(np.sum(qty[taker_values == "ask"]))
        tick.count += int(len(price))
        tick.notional += float(np.sum(price * qty))
        tick.price_qty += float(np.sum(price * qty))
        tick.last_update_ns = packet.local_ts_ns

    def _venue_features(self, venue: str, ts_ns: int) -> tuple[np.ndarray, float, float]:
        book = self.books[venue]
        bp, bq, ap, aq = book.arrays()
        bid, ask = book.bbo()
        stale_ms = (ts_ns - book.last_update_ns) / 1e6 if book.last_update_ns >= 0 else np.inf
        valid = np.isfinite(bid) and np.isfinite(ask)
        mid = 0.5 * (bid + ask) if valid else np.nan
        spread_bps = 1e4 * (ask - bid) / mid if valid else 0.0
        best_bq = float(bq[0]) if bq.size else 0.0
        best_aq = float(aq[0]) if aq.size else 0.0
        micro = (ask * best_bq + bid * best_aq) / max(best_bq + best_aq, 1e-12) if valid else np.nan
        micro_delta = 1e4 * (micro / mid - 1.0) if valid and np.isfinite(micro) else 0.0
        values: list[float] = [float(valid), float(min(stale_ms, 60_000.0)), float(bid if valid else 0.0), float(ask if valid else 0.0), float(mid if valid else 0.0), float(spread_bps), float(micro_delta), log1p(best_bq), log1p(best_aq)]
        for depth in DEPTHS:
            bd = float(np.sum(bq[:depth]))
            ad = float(np.sum(aq[:depth]))
            values.extend((log1p(bd), log1p(ad), (bd - ad) / max(bd + ad, 1e-12)))
        if valid and bp.size >= 10:
            bid_slope = 1e4 * (bp[0] - bp[9]) / mid
        else:
            bid_slope = 0.0
        if valid and ap.size >= 10:
            ask_slope = 1e4 * (ap[9] - ap[0]) / mid
        else:
            ask_slope = 0.0
        tick = self.trades[venue]
        total_qty = tick.buy_qty + tick.sell_qty
        signed = tick.buy_qty - tick.sell_qty
        vwap = tick.price_qty / total_qty if total_qty > 0.0 else mid
        vwap_delta = 1e4 * (vwap / mid - 1.0) if valid and np.isfinite(vwap) else 0.0
        values.extend((bid_slope, ask_slope, log1p(tick.count), log1p(tick.buy_qty), log1p(tick.sell_qty), np.sign(signed) * log1p(abs(signed)), vwap_delta))
        values.extend(self.derivatives[venue].tick_features(float(mid if valid else 0.0), ts_ns))
        tick.reset()
        return np.asarray(values, dtype=np.float32), bid, ask

    def snapshot(self, ts_ns: int, *, max_stale_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        venue_rows: list[np.ndarray] = []
        bids = np.full(len(VENUES), np.nan, dtype=np.float64)
        asks = np.full(len(VENUES), np.nan, dtype=np.float64)
        for venue in VENUES:
            row, bid, ask = self._venue_features(venue, ts_ns)
            venue_rows.append(row)
            idx = VENUE_INDEX[venue]
            bids[idx], asks[idx] = bid, ask
        venue_x = np.stack(venue_rows)
        valid_rows = venue_x[:, 0] > 0.5
        fresh_rows = venue_x[:, 1] <= float(max_stale_ms)
        mids = venue_x[:, 4]
        current = valid_rows & fresh_rows
        spot_indices = np.asarray([idx for idx, venue in enumerate(VENUES) if venue.endswith("_spot")])
        derivative_indices = np.asarray([idx for idx, venue in enumerate(VENUES) if venue.endswith("_perpetual")])
        valid = bool(
            current[VENUE_INDEX["binance_perpetual"]]
            and np.sum(current) >= 4
            and np.any(current[spot_indices])
            and np.any(current[derivative_indices])
        )
        spot = mids[spot_indices]
        deriv = mids[derivative_indices]
        spot_mid = float(np.median(spot[spot > 0])) if np.any(spot > 0) else 0.0
        deriv_mid = float(np.median(deriv[deriv > 0])) if np.any(deriv > 0) else 0.0
        consensus = 0.5 * (spot_mid + deriv_mid) if spot_mid > 0 and deriv_mid > 0 else max(spot_mid, deriv_mid)
        available = mids[mids > 0]
        dispersion = float(np.std(1e4 * (available / np.median(available) - 1.0))) if available.size > 1 else 0.0
        basis = 1e4 * (deriv_mid / spot_mid - 1.0) if spot_mid > 0 and deriv_mid > 0 else 0.0
        self.mid_history.append(consensus)
        returns = []
        for seconds in (0.1, 0.5, 1.0, 5.0, 10.0, 50.0):
            lag = max(1, int(round(seconds * 1_000.0 / self.cadence_ms)))
            if len(self.mid_history) > lag and self.mid_history[-lag - 1] > 0 and consensus > 0:
                returns.append(1e4 * (consensus / self.mid_history[-lag - 1] - 1.0))
            else:
                returns.append(0.0)
        l3_features = self.l3.tick_features()
        global_features = np.asarray(
            [consensus, spot_mid, deriv_mid, basis, dispersion, float(np.sum(valid_rows)), *returns, *l3_features.tolist()],
            dtype=np.float32,
        )
        x = np.concatenate((venue_x.reshape(-1), global_features)).astype(np.float32, copy=False)
        return x, venue_x, bids, asks, valid


GLOBAL_FEATURE_NAMES = (
    "consensus_mid", "spot_mid", "derivative_mid", "basis_bps", "venue_dispersion_bps", "valid_venue_count",
    "return_100ms_bps", "return_500ms_bps", "return_1s_bps", "return_5s_bps", "return_10s_bps", "return_50s_bps",
    "l3_active_orders_log", "l3_bid_qty_log", "l3_ask_qty_log", "l3_imbalance", "l3_adds_log", "l3_deletes_log",
    "l3_modifies_log", "l3_add_qty_log", "l3_delete_qty_log",
)


FLAT_FEATURE_NAMES = tuple(f"{venue}.{name}" for venue in VENUES for name in VENUE_FEATURE_NAMES) + GLOBAL_FEATURE_NAMES
