from __future__ import annotations

import argparse
import json
from pathlib import Path

from stargaze_ml.gold.l2_recording_audit import audit_l2_recording


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a live cTrader L2 recording")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit_l2_recording(args.recording, tick_size=args.tick_size)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
