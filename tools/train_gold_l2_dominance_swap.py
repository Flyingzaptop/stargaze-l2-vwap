from __future__ import annotations

import argparse
from pathlib import Path

from stargaze_ml.gold.l2_dominance_swap import DominanceConfig, train_dominance_swap


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--prepared",type=Path,default=Path("runs/gold_l2_open_v1/prepared_l2_open_policy.npz")); p.add_argument("--open-checkpoint",type=Path,default=Path("runs/gold_l2_open_v1/reinforce/final.pt")); p.add_argument("--out-dir",type=Path,default=Path("runs/gold_l2_dominance_swap_v1")); p.add_argument("--epochs",type=int,default=15); p.add_argument("--device",default="auto"); args=p.parse_args()
    train_dominance_swap(args.prepared,args.open_checkpoint,args.out_dir,DominanceConfig(epochs=args.epochs),device_name=args.device)


if __name__ == "__main__": main()

