from __future__ import annotations

import argparse
import json
from pathlib import Path

from stargaze_ml.gold.l2_frozen_runtime import FrozenL2PolicyRuntime
from stargaze_ml.gold.l2_open_reinforce import PreparedOpenData


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the frozen policy one causal row at a time")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/gold_l2_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("validation", "test", "all"), default="test")
    args = parser.parse_args()
    data = PreparedOpenData(args.prepared)
    if args.split == "validation":
        left, right = data.train_end, data.validation_end
    elif args.split == "test":
        left, right = data.validation_end, len(data.x)
    else:
        left, right = 0, len(data.x)
    runtime = FrozenL2PolicyRuntime(args.bundle, device_name=args.device)
    result = runtime.replay_completed(data, left=left, right=right)
    report = {
        "contract": "stateful one-row causal replay",
        "split": args.split,
        "candidates": len(result["candidates"]),
        "selected": len(result["selected"]),
        "candidate_rows": result["candidates"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("split", "candidates", "selected")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
