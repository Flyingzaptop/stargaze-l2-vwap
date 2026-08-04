from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catastrophic-ticks", type=float, default=500.0)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    trades = report["fixed_test_trades"]
    with np.load(args.prepared, allow_pickle=False) as data:
        feature_names = [str(name) for name in data["feature_names"]]
        entry = np.asarray([int(row["entry_index"]) for row in trades])
        event = np.asarray([int(row["event_index"]) for row in trades])
        side = np.asarray([int(row["selected_side"]) for row in trades])
        pnl = np.asarray([
            float(row["long_pnl"] if selected > 0 else row["short_pnl"])
            for row, selected in zip(trades, side, strict=True)
        ])
        features = data["x"][entry].astype(np.float64)
        catastrophic = pnl <= -args.catastrophic_ticks
        diagnostics = {
            "event_duration_seconds": data["event_duration_seconds"][event].astype(float),
            "event_amplitude_ticks": data["event_amplitude_ticks"][event].astype(float),
            "entry_age_seconds": (entry - data["event_start"][event]).astype(float),
            "remaining_to_cross_seconds": (data["event_crossing_1"][event] - entry).astype(float),
        }

    feature_rows = []
    for index, name in enumerate(feature_names):
        values = features[:, index]
        finite = np.isfinite(values)
        if finite.sum() < 5 or np.nanstd(values) == 0:
            continue
        correlation = float(np.corrcoef(_ranks(values[finite]), _ranks(pnl[finite]))[0, 1])
        if catastrophic.any() and (~catastrophic).any():
            scale = max(float(np.nanstd(values)), 1e-12)
            catastrophic_shift = float((np.nanmean(values[catastrophic]) - np.nanmean(values[~catastrophic])) / scale)
        else:
            catastrophic_shift = 0.0
        feature_rows.append({
            "feature": name,
            "spearman_pnl": correlation,
            "catastrophic_standardized_shift": catastrophic_shift,
        })

    result = {
        "trades": int(len(pnl)),
        "catastrophic_threshold_ticks": -args.catastrophic_ticks,
        "catastrophic_trades": int(catastrophic.sum()),
        "catastrophic_pnl_ticks": float(pnl[catastrophic].sum()),
        "non_catastrophic_pnl_ticks": float(pnl[~catastrophic].sum()),
        "long": {
            "trades": int((side > 0).sum()),
            "mean_pnl_ticks": float(pnl[side > 0].mean()) if np.any(side > 0) else 0.0,
        },
        "short": {
            "trades": int((side < 0).sum()),
            "mean_pnl_ticks": float(pnl[side < 0].mean()) if np.any(side < 0) else 0.0,
        },
        "diagnostics": {
            name: {
                "catastrophic_median": float(np.median(values[catastrophic])) if catastrophic.any() else 0.0,
                "other_median": float(np.median(values[~catastrophic])) if (~catastrophic).any() else 0.0,
            }
            for name, values in diagnostics.items()
        },
        "top_causal_feature_shifts": sorted(
            feature_rows,
            key=lambda row: abs(float(row["catastrophic_standardized_shift"])),
            reverse=True,
        )[:20],
        "top_causal_pnl_correlations": sorted(
            feature_rows,
            key=lambda row: abs(float(row["spearman_pnl"])),
            reverse=True,
        )[:20],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
