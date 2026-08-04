from __future__ import annotations

import argparse
import json
from pathlib import Path

from stargaze_ml.gold.l2_dominance_lstm import DominanceLSTMConfig, train_dominance_lstm


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--open-checkpoint",type=Path,required=True)
    parser.add_argument("--risk-checkpoint",type=Path,required=True)
    parser.add_argument("--rate-report",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--device",default="auto")
    args=parser.parse_args()
    report=train_dominance_lstm(args.prepared,args.open_checkpoint,args.risk_checkpoint,args.rate_report,args.out_dir,DominanceLSTMConfig(epochs=args.epochs),device_name=args.device)
    print(json.dumps({key:value for key,value in report.items() if key!="fixed_test_trades"},indent=2))


if __name__=="__main__":main()
