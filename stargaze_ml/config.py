from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


LEGACY_HORIZONS_SECONDS = tuple(float(x) for x in range(1, 61))
EXTENDED_HORIZONS_SECONDS = (
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    21.0,
    34.0,
    55.0,
    89.0,
    120.0,
    180.0,
    240.0,
    300.0,
    450.0,
    600.0,
)


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path = Path("source/raw datasets")
    cadence_ms: int = 100
    max_stale_ms: int = 2_000
    book_depth_features: tuple[int, ...] = (1, 5, 10, 25, 50, 100, 250, 1000)
    okx_contract_btc: float = 0.01


@dataclass(frozen=True)
class ScoreConfig:
    horizons_seconds: tuple[float, ...] = EXTENDED_HORIZONS_SECONDS
    legacy_horizons_seconds: tuple[float, ...] = LEGACY_HORIZONS_SECONDS
    cost_bps: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0)
    primary_cost_bps: float = 10.0
    peak_ratio: float = 0.75
    event_high_quantile: float = 0.995
    event_low_fraction: float = 0.10


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 4105
    context_ticks: int = 256
    hidden_size: int = 192
    layers: int = 6
    heads: int = 6
    dropout: float = 0.12
    batch_size: int = 8
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 3e-4
    grad_clip: float = 1.0
    purge_seconds: float = 1_200.0


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    scores: ScoreConfig = field(default_factory=ScoreConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["data"]["raw_dir"] = str(self.data.raw_dir)
        return out

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        data = DataConfig(**{**raw.get("data", {}), "raw_dir": Path(raw.get("data", {}).get("raw_dir", "source/raw datasets"))})
        scores_raw = raw.get("scores", {})
        scores = ScoreConfig(
            **{
                **scores_raw,
                **({"horizons_seconds": tuple(scores_raw["horizons_seconds"])} if "horizons_seconds" in scores_raw else {}),
                **({"legacy_horizons_seconds": tuple(scores_raw["legacy_horizons_seconds"])} if "legacy_horizons_seconds" in scores_raw else {}),
                **({"cost_bps": tuple(scores_raw["cost_bps"])} if "cost_bps" in scores_raw else {}),
            }
        )
        train = TrainConfig(**raw.get("train", {}))
        return cls(data=data, scores=scores, train=train)

