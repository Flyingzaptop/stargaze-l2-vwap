from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from stargaze_ml.gold.config import CTraderCredentials
from stargaze_ml.gold.ctrader import refresh_ctrader_credentials
from stargaze_ml.gold.ctrader_l2_recorder import CTraderL2Recorder


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Record live cTrader L2 and causal BBO snapshots")
    parser.add_argument("--secrets", type=Path, default=Path("secrets.gold.runtime.json"))
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / f"gold_l2_live_{timestamp}")
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--flush-seconds", type=float, default=10.0)
    parser.add_argument("--max-buffer-rows", type=int, default=20_000)
    parser.add_argument("--no-refresh", action="store_true")
    return parser.parse_args()


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def main() -> int:
    args = parse_args()
    if not args.no_refresh:
        emit(refresh_ctrader_credentials(args.secrets))
    credentials = CTraderCredentials.from_json(args.secrets)
    recorder = CTraderL2Recorder(
        credentials,
        symbol=args.symbol,
        output_dir=args.out_dir,
        flush_seconds=args.flush_seconds,
        max_buffer_rows=args.max_buffer_rows,
        progress=emit,
    )
    emit(
        recorder.record(
            duration_seconds=args.duration_seconds,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
