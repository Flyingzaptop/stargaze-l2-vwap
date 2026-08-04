from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts import ACTION_NAMES, PositionSide
from ..labels import LabelBuildResult
from ..replay import PolicyReplayResult


def policy_summary(result: PolicyReplayResult, labels: LabelBuildResult | None = None) -> dict[str, Any]:
    actions = result.ledger.action
    counts = {ACTION_NAMES[idx]: int(np.sum(actions == idx)) for idx in range(len(ACTION_NAMES))}
    trades = result.trades
    pnl = np.asarray([float(row["pnl_bps"]) for row in trades], dtype=np.float64)
    hold = np.asarray([float(row["hold_seconds"]) for row in trades], dtype=np.float64)
    out: dict[str, Any] = {
        "ticks": len(actions),
        "action_counts": counts,
        "trades": len(trades),
        "final_state": PositionSide(int(result.final_state)).name.lower(),
        "all_observed_positions_closed": result.final_state == PositionSide.FLAT,
        "mean_pnl_bps": float(np.mean(pnl)) if pnl.size else 0.0,
        "median_pnl_bps": float(np.median(pnl)) if pnl.size else 0.0,
        "total_pnl_bps": float(np.sum(pnl)) if pnl.size else 0.0,
        "win_rate": float(np.mean(pnl > 0.0)) if pnl.size else 0.0,
        "median_hold_seconds": float(np.median(hold)) if hold.size else 0.0,
        "p90_hold_seconds": float(np.quantile(hold, 0.90)) if hold.size else 0.0,
        "max_hold_seconds": float(np.max(hold)) if hold.size else 0.0,
    }
    if labels is not None:
        open_long = actions == 1
        open_short = actions == 2
        close_long = actions == 4
        close_short = actions == 6
        open_total = int(np.sum(open_long | open_short))
        close_total = int(np.sum(close_long | close_short))
        out["open_zone_hit_rate"] = float(
            (np.sum(open_long & labels.open_long_zone) + np.sum(open_short & labels.open_short_zone)) / max(open_total, 1)
        )
        out["close_zone_hit_rate"] = float(
            (np.sum(close_long & labels.close_long_zone) + np.sum(close_short & labels.close_short_zone)) / max(close_total, 1)
        )
    return out


def write_policy_artifacts(out_dir: Path, result: PolicyReplayResult, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = result.ledger
    payload: dict[str, Any] = {
        "ts_ns": pa.array(ledger.ts_ns),
        "state_before": pa.array([PositionSide(int(x)).name.lower() for x in ledger.state_before]),
        "action": pa.array([ACTION_NAMES[int(x)] for x in ledger.action]),
        "state_after": pa.array([PositionSide(int(x)).name.lower() for x in ledger.state_after]),
        "entry_age_ticks": pa.array(ledger.entry_age_ticks),
    }
    for idx, name in enumerate(ACTION_NAMES):
        payload[f"p_{name}"] = pa.array(ledger.probabilities[:, idx])
    pq.write_table(pa.table(payload), out_dir / "policy_ledger.parquet", compression="zstd")
    if result.trades:
        pq.write_table(pa.Table.from_pylist(list(result.trades)), out_dir / "trades.parquet", compression="zstd")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

