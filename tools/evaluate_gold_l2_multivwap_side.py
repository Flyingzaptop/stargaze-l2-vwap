from __future__ import annotations

from stargaze_ml.gold.l2_multivwap_side import run_multivwap_side_experiment


if __name__ == "__main__":
    report = run_multivwap_side_experiment(
        "runs/gold_l2_open_v1/prepared_l2_open_policy.npz",
        "runs/gold_l2_policy_v2/l2_seconds.parquet",
        "runs/gold_l2_open_v1/reinforce/final.pt",
        "runs/gold_l2_multivwap_side_v1",
    )
    print(report["validation_selected"])
    print(report["fixed_test"])

