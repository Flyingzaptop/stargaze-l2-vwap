from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


DEFAULT_HORIZONS_MINUTES = (5, 10, 15, 20, 30, 45, 60)


@dataclass(frozen=True)
class CTraderCredentials:
    client_id: str = field(repr=False)
    client_secret: str = field(repr=False)
    access_token: str = field(repr=False)
    refresh_token: str = field(default="", repr=False)
    account_id: int = 0
    host: str = "demo"

    @classmethod
    def from_json(cls, path: Path) -> "CTraderCredentials":
        resolved = Path(path).expanduser().resolve(strict=True)
        raw = json.loads(resolved.read_text(encoding="utf-8-sig"))
        required = ("client_id", "client_secret", "access_token", "account_id")
        missing = [name for name in required if raw.get(name) in (None, "")]
        if missing:
            raise ValueError(f"missing cTrader credential fields: {', '.join(missing)}")
        host = str(raw.get("host", "demo")).strip().lower()
        if host not in {"demo", "live"}:
            raise ValueError("cTrader host must be 'demo' or 'live'")
        account_id = int(raw["account_id"])
        if account_id <= 0:
            raise ValueError("cTrader account_id must be positive")
        return cls(
            client_id=str(raw["client_id"]).strip(),
            client_secret=str(raw["client_secret"]).strip(),
            access_token=str(raw["access_token"]).strip(),
            refresh_token=str(raw.get("refresh_token", "")).strip(),
            account_id=account_id,
            host=host,
        )

    def public_metadata(self) -> dict[str, Any]:
        return {"account_id": self.account_id, "host": self.host}


@dataclass(frozen=True)
class GoldExperimentConfig:
    context_minutes: int = 60
    horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS_MINUTES
    purge_minutes: int = 120
    train_fraction: float = 0.60
    valid_fraction: float = 0.20
    sample_stride: int = 3
    evaluation_stride: int = 5
    hidden_size: int = 96
    tcn_layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.10
    embedding_size: int = 48
    batch_size: int = 256
    max_epochs: int = 200
    early_stopping_patience: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    retrieval_k: int = 32
    seed: int = 46947

    def __post_init__(self) -> None:
        horizons = tuple(int(value) for value in self.horizons_minutes)
        if self.context_minutes < 10:
            raise ValueError("context_minutes must be at least 10")
        if not horizons or any(value <= 0 or value > 60 for value in horizons):
            raise ValueError("horizons_minutes must contain values in [1, 60]")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons_minutes must be unique and increasing")
        if self.purge_minutes < self.context_minutes + max(horizons):
            raise ValueError("purge_minutes must cover context plus maximum horizon")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("invalid train_fraction")
        if not 0.0 < self.valid_fraction < 1.0:
            raise ValueError("invalid valid_fraction")
        if self.train_fraction + self.valid_fraction >= 1.0:
            raise ValueError("train and validation fractions leave no holdout")
        if min(
            self.sample_stride,
            self.evaluation_stride,
            self.hidden_size,
            self.tcn_layers,
            self.kernel_size,
            self.embedding_size,
            self.batch_size,
            self.max_epochs,
            self.early_stopping_patience,
            self.retrieval_k,
        ) <= 0:
            raise ValueError("integer training parameters must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["horizons_minutes"] = list(self.horizons_minutes)
        return result
