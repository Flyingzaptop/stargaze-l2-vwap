from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stargaze_ml.gold.l2_adaptive_gate import DAY_NS, causal_adaptive_gate
from stargaze_ml.gold.l2_open_reinforce import PreparedOpenData


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze causal adaptive VWAP excursion gates")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, action="append", default=[])
    args = parser.parse_args()
    data = PreparedOpenData(args.prepared)
    absolute_delta = np.abs((data.mid - data.primary_vwap) / 0.01)
    targets = args.target or [100, 150, 250, 300]
    split_ranges = {
        "train": (0, data.train_end),
        "validation": (data.train_end, data.validation_end),
        "test": (data.validation_end, len(data.x)),
    }
    rows = []
    for target in targets:
        adaptive = causal_adaptive_gate(
            ts_ns=data.ts_ns,
            absolute_delta_ticks=absolute_delta,
            event_start=data.event_start,
            event_end=data.event_end,
            completed_amplitude_ticks=data.event_amplitude_ticks,
            target_gated_events_per_active_day=int(target),
        )
        split_metrics = {}
        for name, (left, right) in split_ranges.items():
            mask = (data.event_start >= left) & (data.event_end < right)
            days = max(
                len(set((data.ts_ns[data.event_end[mask]] // DAY_NS).tolist())),
                1,
            )
            threshold = adaptive.gate_threshold_by_event[mask]
            gated = adaptive.gated_by_event[mask]
            split_metrics[name] = {
                "events": int(mask.sum()),
                "active_days": days,
                "gated_events": int(gated.sum()),
                "gated_per_active_day": float(gated.sum() / days),
                "gate_ticks_p50": float(np.median(threshold)) if len(threshold) else 0.0,
                "gate_ticks_p10": float(np.quantile(threshold, 0.1)) if len(threshold) else 0.0,
                "gate_ticks_p90": float(np.quantile(threshold, 0.9)) if len(threshold) else 0.0,
            }
        rows.append({"target_gated_events_per_active_day": int(target), "splits": split_metrics})
    report = {
        "contract": "event threshold fixed at event start using only prior completed amplitudes",
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
