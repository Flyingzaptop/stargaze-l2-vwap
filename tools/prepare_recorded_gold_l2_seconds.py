from __future__ import annotations

import argparse
import json
from pathlib import Path

from stargaze_ml.gold.ctrader_l2_recorder import recorded_l2_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert live cTrader L2 snapshots to causal seconds")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-quote-age-seconds", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seconds = recorded_l2_seconds(
        args.recording,
        max_quote_age_seconds=args.max_quote_age_seconds,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".inprogress")
    seconds.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": seconds.height,
                "observed_rows": int(seconds["observed"].sum()),
                "segments": int(seconds["segment_id"].n_unique()),
                "first_ns": int(seconds["bar_start_ns"].min()),
                "last_ns": int(seconds["bar_start_ns"].max()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
