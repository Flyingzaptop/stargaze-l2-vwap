from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from stargaze_ml.gold.ctrader_l2_recorder import recorded_l2_seconds
from stargaze_ml.gold.l2_frozen_runtime import FrozenL2PolicyRuntime, build_live_policy_view


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen policy against an active L2 recording")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/gold_l2_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--max-cycles", type=int)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.settle_seconds < 1:
        raise ValueError("poll must be positive and settle must be at least one second")
    runtime = FrozenL2PolicyRuntime(args.bundle, device_name=args.device)
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    processed_ts = -1
    decisions: list[dict[str, object]] = []
    cycles = 0
    while True:
        seconds = recorded_l2_seconds(args.recording)
        view = build_live_policy_view(seconds, args.bundle)
        stable_before = time.time_ns() - int(args.settle_seconds * 1e9)
        for index, timestamp in enumerate(view.ts_ns):
            timestamp = int(timestamp)
            if timestamp <= processed_ts or timestamp > stable_before or index + 1 >= len(view.ts_ns):
                continue
            decision = runtime.process_index(view, index, require_next_observed=True)
            processed_ts = timestamp
            if decision is None:
                continue
            row = dict(decision)
            row["recorded_at_ns"] = time.time_ns()
            if bool(row["accepted"]):
                execution = index + 1
                side = int(row["selected_side"])
                row["entry_execution_ts_ns"] = int(view.ts_ns[execution])
                row["paper_entry_price"] = float(
                    view.first_ask[execution] if side > 0 else view.first_bid[execution]
                )
            decisions.append(row)
            temporary = output.with_suffix(output.suffix + ".inprogress")
            temporary.write_text(
                json.dumps(
                    {
                        "contract": "paper only; open at next-second BBO; exit evaluated at VWAP crossing",
                        "last_processed_ts_ns": processed_ts,
                        "candidate_decisions": len(decisions),
                        "accepted_decisions": sum(bool(item["accepted"]) for item in decisions),
                        "decisions": decisions,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(output)
        print(
            json.dumps(
                {
                    "stage": "paper",
                    "seconds": len(view.ts_ns),
                    "last_processed_ts_ns": processed_ts,
                    "candidates": len(decisions),
                    "accepted": sum(bool(item["accepted"]) for item in decisions),
                }
            ),
            flush=True,
        )
        cycles += 1
        if args.max_cycles is not None and cycles >= args.max_cycles:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
