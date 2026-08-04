from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class MarketKind(StrEnum):
    """Market group used to build the all-market consensus."""

    SPOT = "spot"
    DERIVATIVE = "derivative"

    @classmethod
    def parse(cls, value: MarketKind | str) -> MarketKind:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized == "spot":
            return cls.SPOT
        if normalized in {
            "derivative",
            "derivatives",
            "future",
            "futures",
            "linear",
            "perp",
            "perpetual",
            "swap",
        }:
            return cls.DERIVATIVE
        raise ValueError(f"unsupported market kind: {value!r}")


@dataclass(frozen=True)
class ForwardCleanupConfig:
    """Wall-clock equivalent of the legacy forward-label cleanup."""

    median_window_seconds: float = 2.1
    gaussian_sigma_seconds: float = 1.5
    gaussian_truncate: float = 4.0
    hump_epsilon_bps: float = 0.025
    min_hump_peak_bps: float | None = None
    min_hump_width_seconds: float = 1.5
    min_hump_area_bps_seconds: float = 0.35
    adaptive_peak_quantile: float = 0.95
    adaptive_peak_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.median_window_seconds <= 0.0:
            raise ValueError("median_window_seconds must be positive")
        if self.gaussian_sigma_seconds <= 0.0:
            raise ValueError("gaussian_sigma_seconds must be positive")
        if self.gaussian_truncate <= 0.0:
            raise ValueError("gaussian_truncate must be positive")
        if self.hump_epsilon_bps < 0.0:
            raise ValueError("hump_epsilon_bps cannot be negative")
        if self.min_hump_peak_bps is not None and self.min_hump_peak_bps < 0.0:
            raise ValueError("min_hump_peak_bps cannot be negative")
        if self.min_hump_width_seconds < 0.0:
            raise ValueError("min_hump_width_seconds cannot be negative")
        if self.min_hump_area_bps_seconds < 0.0:
            raise ValueError("min_hump_area_bps_seconds cannot be negative")
        if not 0.0 <= self.adaptive_peak_quantile <= 1.0:
            raise ValueError("adaptive_peak_quantile must be in [0, 1]")
        if self.adaptive_peak_fraction < 0.0:
            raise ValueError("adaptive_peak_fraction cannot be negative")


@dataclass(frozen=True)
class ScoreCube:
    """Forward/backward scores with axes ``[time, horizon, venue]``."""

    ts_ns: IntArray
    horizons_seconds: FloatArray
    venue_names: tuple[str, ...]
    market_kinds: tuple[MarketKind, ...]
    forward_long: FloatArray
    forward_short: FloatArray
    backward_long: FloatArray
    backward_short: FloatArray
    forward_valid: BoolArray
    backward_valid: BoolArray
    cost_bps: float
    cadence_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "market_kinds", tuple(MarketKind.parse(value) for value in self.market_kinds))
        expected = (len(self.ts_ns), len(self.horizons_seconds), len(self.venue_names))
        arrays = (
            self.forward_long,
            self.forward_short,
            self.backward_long,
            self.backward_short,
            self.forward_valid,
            self.backward_valid,
        )
        if any(array.shape != expected for array in arrays):
            raise ValueError(f"score arrays must all have shape {expected}")
        if len(self.market_kinds) != len(self.venue_names):
            raise ValueError("market_kinds must align with venue_names")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.forward_long.shape


@dataclass(frozen=True)
class ConsensusScores:
    """Equal spot/derivative blend after a median within each group."""

    ts_ns: IntArray
    horizons_seconds: FloatArray
    forward_long: FloatArray
    forward_short: FloatArray
    backward_long: FloatArray
    backward_short: FloatArray
    forward_valid: BoolArray
    backward_valid: BoolArray
    spot_forward_valid: BoolArray
    derivative_forward_valid: BoolArray
    spot_backward_valid: BoolArray
    derivative_backward_valid: BoolArray

    def __post_init__(self) -> None:
        expected = (len(self.ts_ns), len(self.horizons_seconds))
        arrays = (
            self.forward_long,
            self.forward_short,
            self.backward_long,
            self.backward_short,
            self.forward_valid,
            self.backward_valid,
            self.spot_forward_valid,
            self.derivative_forward_valid,
            self.spot_backward_valid,
            self.derivative_backward_valid,
        )
        if any(array.shape != expected for array in arrays):
            raise ValueError(f"consensus arrays must all have shape {expected}")


@dataclass(frozen=True)
class AggregatedScores:
    """One robust horizon aggregate per timestamp and direction."""

    ts_ns: IntArray
    forward_long: FloatArray
    forward_short: FloatArray
    backward_long: FloatArray
    backward_short: FloatArray
    forward_valid: BoolArray
    backward_valid: BoolArray
    horizon_weights: FloatArray

    def __post_init__(self) -> None:
        expected = (len(self.ts_ns),)
        arrays = (
            self.forward_long,
            self.forward_short,
            self.backward_long,
            self.backward_short,
            self.forward_valid,
            self.backward_valid,
        )
        if any(array.shape != expected for array in arrays):
            raise ValueError(f"aggregated score arrays must all have shape {expected}")


@dataclass(frozen=True)
class CleanedForwardScores:
    """Denoised oracle labels; invalid/censored rows remain masked."""

    long: FloatArray
    short: FloatArray
    valid: BoolArray
    peak_min_long_bps: float
    peak_min_short_bps: float

    def __post_init__(self) -> None:
        if self.long.ndim != 1 or self.short.shape != self.long.shape:
            raise ValueError("long and short labels must be aligned one-dimensional arrays")
        if self.valid.shape != self.long.shape:
            raise ValueError("valid must align with forward labels")


@dataclass(frozen=True)
class ScoreBundle:
    """Complete score result at one transaction-cost scenario."""

    cube: ScoreCube
    consensus: ConsensusScores
    aggregate: AggregatedScores


__all__ = [
    "AggregatedScores",
    "BoolArray",
    "CleanedForwardScores",
    "ConsensusScores",
    "FloatArray",
    "ForwardCleanupConfig",
    "IntArray",
    "MarketKind",
    "ScoreBundle",
    "ScoreCube",
]
