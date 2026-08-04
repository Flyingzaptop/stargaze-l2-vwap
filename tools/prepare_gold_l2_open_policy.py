from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from stargaze_ml.gold.l2_open_events import build_open_policy_data
from stargaze_ml.gold.frozen_policy import load_frozen_policy_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=Path, default=Path("runs/gold_l2_policy_v2/l2_seconds.parquet"))
    parser.add_argument("--base", type=Path, default=Path("runs/gold_l2_policy_v2/prepared_l2_policy.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gold_l2_open_v1"))
    parser.add_argument("--amplitude-ticks", type=float, default=230.0)
    parser.add_argument("--gate-fraction", type=float, default=0.75)
    parser.add_argument("--primary-vwap", default="60")
    parser.add_argument("--feature-profile", choices=("raw", "hierarchy", "leadlag"), default="raw")
    parser.add_argument("--match-train-good-events", type=int)
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="prepare a forward-only dataset without borrowing historical split indices",
    )
    parser.add_argument("--policy-bundle", type=Path)
    args = parser.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    seconds = pl.read_parquet(args.seconds.resolve(strict=True))
    if args.all_test:
        train_end = 0
        validation_end = 0
    else:
        with np.load(args.base.resolve(strict=True), allow_pickle=False) as old:
            train_end = int(old["train_end"]); validation_end = int(old["validation_end"])
    primary_text = args.primary_vwap
    feature_profile = args.feature_profile
    gate_fraction = float(args.gate_fraction)
    amplitude_ticks = float(args.amplitude_ticks)
    min_duration_seconds = 30
    if args.policy_bundle is not None:
        if args.match_train_good_events:
            raise ValueError("--policy-bundle cannot be combined with amplitude recalibration")
        preparation = load_frozen_policy_bundle(args.policy_bundle).policy["preparation"]
        primary_text = str(preparation["primary_vwap"])
        feature_profile = str(preparation["feature_profile"])
        gate_fraction = float(preparation["gate_fraction"])
        amplitude_ticks = float(preparation["amplitude_threshold_ticks"])
        min_duration_seconds = int(preparation["min_duration_seconds"])
    primary_vwap: int | str = "ribbon" if primary_text == "ribbon" else int(primary_text)
    if args.match_train_good_events:
        pilot = build_open_policy_data(
            seconds, amplitude_threshold_ticks=1.0,
            gate_fraction=gate_fraction, primary_vwap=primary_vwap,
            feature_profile=feature_profile,
            min_duration_seconds=min_duration_seconds,
        )
        pilot_events = (
            (pilot.excursions.crossing_2 + 1 < train_end)
            & (pilot.excursions.duration_seconds >= 30.0)
        )
        candidates = np.sort(pilot.excursions.amplitude_ticks[pilot_events])
        target = min(int(args.match_train_good_events), len(candidates))
        if target < 1:
            raise ValueError("no train excursions available for amplitude calibration")
        amplitude_ticks = float(candidates[-target])
    data = build_open_policy_data(
        seconds, amplitude_threshold_ticks=amplitude_ticks,
        gate_fraction=gate_fraction, primary_vwap=primary_vwap,
        feature_profile=feature_profile,
        min_duration_seconds=min_duration_seconds,
    )
    e = data.excursions
    np.savez_compressed(
        out / "prepared_l2_open_policy.npz",
        ts_ns=seconds["bar_start_ns"].to_numpy(), segment_id=seconds["segment_id"].to_numpy(),
        x=data.x, feature_names=np.asarray(data.feature_names), valid_feature=data.valid_feature,
        observed=seconds["observed"].to_numpy(), first_bid=seconds["first_bid"].to_numpy(),
        first_ask=seconds["first_ask"].to_numpy(), last_bid=seconds["last_bid"].to_numpy(),
        last_ask=seconds["last_ask"].to_numpy(), mid=data.mid, primary_vwap=data.primary_vwap,
        side=data.side, event_id=data.event_id, gate_open=data.gate_open,
        event_start=e.start, event_end=e.end, event_crossing_1=e.crossing_1,
        event_crossing_2=e.crossing_2, event_side=e.side,
        event_duration_seconds=e.duration_seconds, event_amplitude_ticks=e.amplitude_ticks,
        event_gate_index=e.gate_index, event_gated=e.gated, event_good=e.good,
        train_end=np.asarray(train_end), validation_end=np.asarray(validation_end),
    )
    train_events = e.crossing_2 + 1 < train_end
    manifest = {
        "rows": len(data.x), "features": list(data.feature_names),
        "primary_vwap": str(primary_vwap),
        "feature_profile": feature_profile,
        "direction": "handled by a separate direction model",
        "amplitude_threshold_ticks": amplitude_ticks,
        "gate_fraction": gate_fraction,
        "gate_ticks": amplitude_ticks * gate_fraction,
        "min_duration_seconds": min_duration_seconds,
        "split_contract": "all_forward_test" if args.all_test else "borrowed_from_base",
        "train_events": int(train_events.sum()),
        "train_gated_events": int((train_events & e.gated).sum()),
        "train_good_events": int((train_events & e.good).sum()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
