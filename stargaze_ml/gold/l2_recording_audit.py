from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .ctrader_l2_recorder import recorded_l2_seconds


def _quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_p50": 0.0, f"{prefix}_p95": 0.0, f"{prefix}_p99": 0.0}
    return {
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
    }


def audit_l2_recording(output_dir: Path, *, tick_size: float = 0.01) -> dict[str, Any]:
    """Audit immutable live parts without touching a currently open buffer."""

    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    root = Path(output_dir).expanduser().resolve(strict=True)
    raw_paths = sorted((root / "raw_parts").glob("depth_*.parquet"))
    snapshot_paths = sorted((root / "snapshot_parts").glob("snapshot_*.parquet"))
    if not raw_paths or not snapshot_paths:
        raise ValueError("recording has no completed raw/snapshot parts")
    raw = pl.concat([pl.read_parquet(path) for path in raw_paths], how="diagonal_relaxed")
    snapshots = pl.concat(
        [pl.read_parquet(path) for path in snapshot_paths], how="diagonal_relaxed"
    ).sort(["connection_segment", "event_sequence"])
    spread_ticks = (
        (snapshots["best_ask"] - snapshots["best_bid"]).to_numpy().astype(np.float64)
        / tick_size
    )
    invalid = (
        (snapshots["best_bid"] <= 0)
        | (snapshots["best_ask"] <= snapshots["best_bid"])
        | (snapshots["bid_size_top1"] <= 0)
        | (snapshots["ask_size_top1"] <= 0)
    )
    receive = snapshots["receive_ns"].to_numpy().astype(np.int64)
    segment = snapshots["connection_segment"].to_numpy().astype(np.int64)
    same = segment[1:] == segment[:-1]
    delta_ms = np.diff(receive)[same].astype(np.float64) / 1e6
    deleted = raw.filter(pl.col("type") == "deleted")
    unknown_deletes = (
        int(deleted.filter(pl.col("delete_known") == False).height)  # noqa: E712
        if "delete_known" in deleted.columns
        else 0
    )
    seconds = recorded_l2_seconds(root)
    duration_seconds = max((int(receive.max()) - int(receive.min())) / 1e9, 0.0)
    result: dict[str, Any] = {
        "output_dir": str(root),
        "raw_parts": len(raw_paths),
        "snapshot_parts": len(snapshot_paths),
        "raw_rows": raw.height,
        "snapshot_rows": snapshots.height,
        "connection_segments": int(snapshots["connection_segment"].n_unique()),
        "first_receive_ns": int(receive.min()),
        "last_receive_ns": int(receive.max()),
        "duration_seconds": duration_seconds,
        "wall_clock_nonmonotonic_steps": int(np.sum((np.diff(receive) <= 0) & same)),
        "invalid_or_crossed_snapshots": int(invalid.sum()),
        "deleted_rows": deleted.height,
        "unknown_deleted_rows": unknown_deletes,
        "unknown_delete_fraction": unknown_deletes / max(deleted.height, 1),
        "second_rows": seconds.height,
        "observed_second_rows": int(seconds["observed"].sum()),
        "observed_second_fraction": float(seconds["observed"].mean()),
        "second_segments": int(seconds["segment_id"].n_unique()),
        "max_quote_age_ms": float(seconds["quote_age_ms"].max()),
        "inprogress_files": len(list(root.rglob("*.inprogress"))),
    }
    result.update(_quantiles(spread_ticks, "spread_ticks"))
    result.update(_quantiles(delta_ms, "inter_event_ms"))
    result.update(
        _quantiles(snapshots["bid_levels"].to_numpy().astype(np.float64), "bid_levels")
    )
    result.update(
        _quantiles(snapshots["ask_levels"].to_numpy().astype(np.float64), "ask_levels")
    )
    return result
