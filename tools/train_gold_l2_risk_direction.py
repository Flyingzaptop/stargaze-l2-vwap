from __future__ import annotations

import argparse
from pathlib import Path

from stargaze_ml.gold.l2_risk_direction import RiskDirectionConfig, train_risk_direction


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--open-checkpoint",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--head-only-epochs",type=int,default=3)
    parser.add_argument("--learning-rate",type=float,default=3e-4)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--tail-threshold-ticks",type=float,default=300.0)
    parser.add_argument("--tail-weight",type=float,default=0.5)
    parser.add_argument("--opportunity-weight",type=float,default=0.25)
    parser.add_argument(
        "--entry-only",
        action="store_true",
        help="train direction heads only at frozen open-policy entry points",
    )
    parser.add_argument("--init-risk-checkpoint",type=Path)
    parser.add_argument("--seed",type=int,default=20260808)
    args=parser.parse_args()
    train_risk_direction(
        args.prepared,args.open_checkpoint,args.out_dir,
        RiskDirectionConfig(
            epochs=args.epochs,head_only_epochs=args.head_only_epochs,
            learning_rate=args.learning_rate,
            tail_threshold_ticks=args.tail_threshold_ticks,
            tail_weight=args.tail_weight,opportunity_weight=args.opportunity_weight,
            entry_only_training=args.entry_only,
            seed=args.seed,
        ),device_name=args.device,
        initial_checkpoint_path=args.init_risk_checkpoint,
    )


if __name__=="__main__": main()
