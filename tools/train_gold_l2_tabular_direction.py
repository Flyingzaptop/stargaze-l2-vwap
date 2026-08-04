from __future__ import annotations

import argparse
import json
from pathlib import Path

from stargaze_ml.gold.l2_tabular_direction import TabularDirectionConfig, train_tabular_direction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--open-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tail-threshold-ticks", type=float, default=500.0)
    args = parser.parse_args()
    report = train_tabular_direction(
        args.prepared, args.open_checkpoint, args.out_dir,
        TabularDirectionConfig(
            max_iter=args.max_iter, tail_threshold_ticks=args.tail_threshold_ticks
        ),
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
