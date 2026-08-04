from __future__ import annotations

import argparse
import json
from pathlib import Path

from stargaze_ml.gold.l2_hierarchy_dominance import (
    HierarchyDominanceConfig,
    run_hierarchy_dominance_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate interpretable VWAP hierarchy dominance")
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("runs/gold_l2_multihorizon_v2/primary_60/prepared_l2_open_policy.npz"),
    )
    parser.add_argument(
        "--open-checkpoint",
        type=Path,
        default=Path("runs/gold_l2_multihorizon_v2/primary_60/open_oracle/final.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gold_l2_hierarchy_dominance"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_hierarchy_dominance_experiment(
        args.prepared,
        args.open_checkpoint,
        args.output_dir,
        HierarchyDominanceConfig(),
        device_name=args.device,
    )
    print(json.dumps({key: report[key] for key in (
        "selected_on_validation",
        "fixed_test",
        "fixed_test_local60_baseline_same_entries",
        "fixed_test_oracle_same_entries",
        "test_swap_fraction",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
