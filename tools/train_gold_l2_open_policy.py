from __future__ import annotations

import argparse
from pathlib import Path

from stargaze_ml.gold.l2_open_reinforce import OpenReinforceConfig, train_open_policy


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--prepared",type=Path,default=Path("runs/gold_l2_open_v1/prepared_l2_open_policy.npz")); p.add_argument("--out-dir",type=Path,default=Path("runs/gold_l2_open_v1/reinforce")); p.add_argument("--epochs",type=int,default=30); p.add_argument("--device",default="auto")
    p.add_argument("--reward-mode",choices=("event_side","oracle_best"),default="event_side")
    args=p.parse_args()
    train_open_policy(args.prepared,args.out_dir,OpenReinforceConfig(epochs=args.epochs,reward_mode=args.reward_mode),device_name=args.device)


if __name__ == "__main__": main()
