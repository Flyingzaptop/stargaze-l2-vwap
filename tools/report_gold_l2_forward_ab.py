from __future__ import annotations

import argparse
import json
from pathlib import Path

from stargaze_ml.gold.l2_forward_report import (
    build_forward_ab_report,
    forward_ab_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build frozen-policy forward A/B report"
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--policy", nargs=2, action="append",
        metavar=("NAME", "REPORT"), required=True,
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    parser.add_argument("--minimum-trades", type=int, default=30)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    policies = {
        str(name): json.loads(Path(path).read_text(encoding="utf-8"))
        for name, path in args.policy
    }
    report = build_forward_ab_report(
        audit, policies, minimum_trades=args.minimum_trades
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_markdown.write_text(forward_ab_markdown(report), encoding="utf-8")
    print(
        json.dumps({
            "out_json": str(args.out_json.resolve()),
            "out_markdown": str(args.out_markdown.resolve()),
            "sample_sufficient": report["sample_sufficient"],
        })
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
