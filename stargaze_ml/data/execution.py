from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from ..features.state import MarketState
from .catalog import DatasetCatalog
from .stream import iter_packets


@dataclass(frozen=True)
class ExecutionQuotes:
    latency_ms: float
    bid: np.ndarray
    ask: np.ndarray
    valid: np.ndarray


def build_execution_quote_scenarios(
    catalog: DatasetCatalog,
    signal_ts_ns: np.ndarray,
    *,
    latencies_ms: Sequence[float] = (100.0, 250.0, 500.0),
    notional_usd: float = 1_000.0,
    max_stale_ms: float = 2_000.0,
    initial_state: MarketState | None = None,
) -> dict[float, ExecutionQuotes]:
    stream = next(
        item for item in catalog.streams
        if item.venue == "binance_perpetual" and item.kind == "book"
    )
    timestamps = np.asarray(signal_ts_ns, dtype=np.int64)
    requests: list[tuple[int, float, int]] = []
    for latency in latencies_ms:
        shifted = timestamps + int(round(float(latency) * 1_000_000.0))
        requests.extend((int(ts), float(latency), row) for row, ts in enumerate(shifted))
    requests.sort(key=lambda item: item[0])
    output = {
        float(latency): ExecutionQuotes(
            float(latency),
            np.full(len(timestamps), np.nan, dtype=np.float64),
            np.full(len(timestamps), np.nan, dtype=np.float64),
            np.zeros(len(timestamps), dtype=bool),
        )
        for latency in latencies_ms
    }
    state = initial_state or MarketState(cadence_ms=1_000)
    packets = iter(iter_packets(stream, end_ts_ns=requests[-1][0]))
    try:
        packet = next(packets)
    except StopIteration:
        packet = None
    pending = []
    for target_ts, latency, row in requests:
        while packet is not None and packet.local_ts_ns <= target_ts:
            pending.append(packet)
            try:
                packet = next(packets)
            except StopIteration:
                packet = None
        if pending:
            state.apply_tick(pending)
            pending = []
        book = state.books["binance_perpetual"]
        if book.warm and book.bids and book.asks:
            bid = max(book.bids)
            ask = min(book.asks)
        else:
            bid = ask = np.nan
        result = output[latency]
        result.bid[row], result.ask[row] = bid, ask
        if not np.isfinite(bid) or not np.isfinite(ask) or bid >= ask:
            continue
        bid_qty = float(book.bids[bid])
        ask_qty = float(book.asks[ask])
        required = float(notional_usd) / ask
        stale_ms = (target_ts - book.last_update_ns) / 1e6 if book.last_update_ns >= 0 else np.inf
        result.valid[row] = (
            stale_ms <= float(max_stale_ms)
            and bid_qty >= required
            and ask_qty >= required
        )
    return output
