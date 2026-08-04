from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


class PositionSide(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = 2


class Action(IntEnum):
    SKIP = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    HOLD_LONG = 3
    CLOSE_LONG = 4
    HOLD_SHORT = 5
    CLOSE_SHORT = 6


ACTION_NAMES = tuple(action.name.lower() for action in Action)
VENUES = (
    "binance_spot",
    "binance_perpetual",
    "bybit_perpetual",
    "okx_perpetual",
    "coinbase_spot",
    "kraken_spot",
    "deribit_perpetual",
    "bitfinex_spot",
    "hyperliquid_perpetual",
)
VENUE_INDEX = {name: idx for idx, name in enumerate(VENUES)}
VENUE_MARKET_KINDS = tuple("spot" if name.endswith("_spot") else "derivative" for name in VENUES)


@dataclass(frozen=True)
class StreamSpec:
    exchange: str
    market: str
    symbol: str
    channel: str
    path: Path
    kind: str

    @property
    def venue(self) -> str:
        if self.exchange == "binance":
            return "binance_spot" if self.market == "spot" else "binance_perpetual"
        if self.exchange in {"coinbase", "kraken", "bitfinex"}:
            return f"{self.exchange}_spot"
        if self.exchange in {"bybit", "okx", "deribit", "hyperliquid"}:
            return f"{self.exchange}_perpetual"
        raise ValueError(f"unsupported market source: {self.exchange}/{self.market}")


@dataclass
class Packet:
    stream: StreamSpec
    local_ts_ns: int
    columns: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0


@dataclass
class CausalFrames:
    ts_ns: np.ndarray
    x: np.ndarray
    venue_x: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    valid: np.ndarray
    segment_id: np.ndarray
    feature_names: tuple[str, ...]
    venue_feature_names: tuple[str, ...]
    venues: tuple[str, ...] = VENUES

    def __post_init__(self) -> None:
        n = len(self.ts_ns)
        if self.x.shape[0] != n or self.venue_x.shape[0] != n:
            raise ValueError("frame tensors must have the same leading dimension")
        if self.bid.shape != self.ask.shape or self.bid.shape != (n, len(self.venues)):
            raise ValueError("bid/ask must be [time, venue]")
        if len(self.valid) != n or len(self.segment_id) != n:
            raise ValueError("valid and segment_id must align with timestamps")

    def save(self, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        from .artifacts import write_npz_artifact

        write_npz_artifact(path, self, metadata=metadata or {})


@dataclass
class LabelPack:
    action: np.ndarray
    position_side: np.ndarray
    forward_long: np.ndarray
    forward_short: np.ndarray
    backward_long: np.ndarray
    backward_short: np.ndarray
    open_long_zone: np.ndarray
    open_short_zone: np.ndarray
    close_long_zone: np.ndarray
    close_short_zone: np.ndarray
    horizon_class: np.ndarray
    sample_weight: np.ndarray


@dataclass
class PolicyLedger:
    ts_ns: np.ndarray
    state_before: np.ndarray
    action: np.ndarray
    state_after: np.ndarray
    probabilities: np.ndarray
    entry_age_ticks: np.ndarray
