from __future__ import annotations

import argparse
from pathlib import Path

from stargaze_ml.gold.l2_profit_direction import ProfitDirectionConfig, train_profit_direction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--open-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train_profit_direction(
        args.prepared, args.open_checkpoint, args.out_dir,
        ProfitDirectionConfig(epochs=args.epochs), device_name=args.device,
    )


if __name__ == "__main__":
    main()
