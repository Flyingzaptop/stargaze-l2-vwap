from __future__ import annotations

import argparse
from pathlib import Path

from stargaze_ml.gold.l2_profit_direction import evaluate_profit_direction_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--open-checkpoint", type=Path, required=True)
    parser.add_argument("--direction-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = evaluate_profit_direction_checkpoint(
        args.prepared, args.open_checkpoint, args.direction_checkpoint,
        args.output, device_name=args.device,
    )
    print(report["selected_on_validation"])
    print(report["fixed_test"])


if __name__ == "__main__":
    main()
