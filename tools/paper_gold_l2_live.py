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
    completed_trades: list[dict[str, object]] = []
    active_trade: dict[str, object] | None = None
    tick_size = float(runtime.bundle.policy["preparation"]["tick_size"])
    fill_cost = float(
        runtime.market.commission_per_fill_ticks + runtime.market.slippage_per_fill_ticks
    )

    def persist() -> None:
        temporary = output.with_suffix(output.suffix + ".inprogress")
        temporary.write_text(
            json.dumps(
                {
                    "contract": "paper only; next-second BBO entry; first VWAP-crossing exit",
                    "last_processed_ts_ns": processed_ts,
                    "candidate_decisions": len(decisions),
                    "accepted_decisions": sum(bool(item["accepted"]) for item in decisions),
                    "completed_trades": len(completed_trades),
                    "active_trade": active_trade,
                    "decisions": decisions,
                    "trades": completed_trades,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)

    cycles = 0
    while True:
        seconds = recorded_l2_seconds(args.recording)
        view = build_live_policy_view(seconds, args.bundle)
        stable_before = time.time_ns() - int(args.settle_seconds * 1e9)
        for index, timestamp in enumerate(view.ts_ns):
            timestamp = int(timestamp)
            if timestamp <= processed_ts or timestamp > stable_before or index + 1 >= len(view.ts_ns):
                continue
            current_event = int(view.event_id[index])
            if active_trade is not None and current_event != int(active_trade["event_id"]):
                execution = index + 1
                trade = dict(active_trade)
                trade["exit_decision_ts_ns"] = timestamp
                trade["exit_execution_ts_ns"] = int(view.ts_ns[execution])
                if bool(view.observed[execution]):
                    side = int(trade["selected_side"])
                    exit_price = float(
                        view.first_bid[execution] if side > 0 else view.first_ask[execution]
                    )
                    entry_price = float(trade["paper_entry_price"])
                    realized = (
                        (exit_price - entry_price) / tick_size
                        if side > 0
                        else (entry_price - exit_price) / tick_size
                    ) - 2.0 * fill_cost
                    trade.update(
                        {
                            "status": "closed",
                            "paper_exit_price": exit_price,
                            "realized_pnl_ticks": realized,
                            "holding_seconds": (
                                int(view.ts_ns[execution])
                                - int(trade["entry_execution_ts_ns"])
                            )
                            / 1e9,
                        }
                    )
                else:
                    trade["status"] = "invalid_missing_exit_bbo"
                completed_trades.append(trade)
                active_trade = None
                persist()
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
                if active_trade is not None:
                    raise RuntimeError("paper runtime attempted overlapping positions")
                active_trade = {
                    key: row[key]
                    for key in (
                        "event_id",
                        "entry_ts_ns",
                        "entry_execution_ts_ns",
                        "paper_entry_price",
                        "selected_side",
                        "selection_score",
                    )
                }
            decisions.append(row)
            persist()
        print(
            json.dumps(
                {
                    "stage": "paper",
                    "seconds": len(view.ts_ns),
                    "last_processed_ts_ns": processed_ts,
                    "candidates": len(decisions),
                    "accepted": sum(bool(item["accepted"]) for item in decisions),
                    "completed_trades": len(completed_trades),
                    "active": active_trade is not None,
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
