from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import DataConfig
from ..contracts import CausalFrames
from ..features.state import FLAT_FEATURE_NAMES, MarketState, VENUE_FEATURE_NAMES
from .catalog import DatasetCatalog
from .stream import iter_merged_packets


def _ceil_grid(ts_ns: int, cadence_ns: int) -> int:
    return ((int(ts_ns) + cadence_ns - 1) // cadence_ns) * cadence_ns


class CausalReplayBuilder:
    def __init__(self, catalog: DatasetCatalog, config: DataConfig) -> None:
        self.catalog = catalog
        self.config = config

    def build(
        self,
        *,
        start_ts_ns: int | None = None,
        end_ts_ns: int | None = None,
        progress: Callable[[int, int], None] | None = None,
        initial_state: MarketState | None = None,
    ) -> CausalFrames:
        cadence_ns = int(self.config.cadence_ms * 1_000_000)
        if initial_state is None:
            requested_start = self.catalog.common_start_ns if start_ts_ns is None else max(int(start_ts_ns), self.catalog.common_start_ns)
        elif start_ts_ns is None:
            raise ValueError("start_ts_ns is required with initial_state")
        else:
            requested_start = int(start_ts_ns)
        requested_end = self.catalog.common_end_ns if end_ts_ns is None else min(int(end_ts_ns), self.catalog.common_end_ns)
        start = _ceil_grid(requested_start, cadence_ns)
        end = (requested_end // cadence_ns) * cadence_ns
        if end < start:
            raise ValueError("empty replay interval")
        ticks = np.arange(start, end + cadence_ns, cadence_ns, dtype=np.int64)
        state = initial_state or MarketState(okx_contract_btc=self.config.okx_contract_btc, cadence_ms=self.config.cadence_ms)
        packets = iter(iter_merged_packets(self.catalog.streams, end_ts_ns=end))
        try:
            packet = next(packets)
        except StopIteration:
            packet = None
        x_rows: list[np.ndarray] = []
        venue_rows: list[np.ndarray] = []
        bid_rows: list[np.ndarray] = []
        ask_rows: list[np.ndarray] = []
        valid_rows: list[bool] = []
        segment_rows: list[int] = []
        for idx, tick in enumerate(ticks):
            tick_packets = []
            while packet is not None and packet.local_ts_ns <= int(tick):
                tick_packets.append(packet)
                try:
                    packet = next(packets)
                except StopIteration:
                    packet = None
            if tick_packets:
                state.apply_tick(tick_packets)
            x, venue_x, bids, asks, valid = state.snapshot(int(tick), max_stale_ms=self.config.max_stale_ms)
            x_rows.append(x)
            venue_rows.append(venue_x)
            bid_rows.append(bids)
            ask_rows.append(asks)
            valid_rows.append(valid)
            segment_rows.append(state.segment_id)
            if progress is not None and (idx == 0 or idx + 1 == len(ticks) or (idx + 1) % 1_000 == 0):
                progress(idx + 1, len(ticks))
        return CausalFrames(
            ts_ns=ticks,
            x=np.stack(x_rows).astype(np.float32, copy=False),
            venue_x=np.stack(venue_rows).astype(np.float32, copy=False),
            bid=np.stack(bid_rows),
            ask=np.stack(ask_rows),
            valid=np.asarray(valid_rows, dtype=bool),
            segment_id=np.asarray(segment_rows, dtype=np.int32),
            feature_names=FLAT_FEATURE_NAMES,
            venue_feature_names=tuple(VENUE_FEATURE_NAMES),
        )
