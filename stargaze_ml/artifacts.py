from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import numpy as np

from .contracts import CausalFrames


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_npz_artifact(path: Path, frames: CausalFrames, *, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ts_ns=frames.ts_ns,
        x=frames.x,
        venue_x=frames.venue_x,
        bid=frames.bid,
        ask=frames.ask,
        valid=frames.valid,
        segment_id=frames.segment_id,
        feature_names=np.asarray(frames.feature_names),
        venue_feature_names=np.asarray(frames.venue_feature_names),
        venues=np.asarray(frames.venues),
    )
    write_json(path.with_suffix(path.suffix + ".manifest.json"), metadata)


def load_frames(path: Path) -> CausalFrames:
    with np.load(path, allow_pickle=False) as data:
        return CausalFrames(
            ts_ns=data["ts_ns"],
            x=data["x"],
            venue_x=data["venue_x"],
            bid=data["bid"],
            ask=data["ask"],
            valid=data["valid"],
            segment_id=data["segment_id"],
            feature_names=tuple(str(x) for x in data["feature_names"]),
            venue_feature_names=tuple(str(x) for x in data["venue_feature_names"]),
            venues=tuple(str(x) for x in data["venues"]),
        )
