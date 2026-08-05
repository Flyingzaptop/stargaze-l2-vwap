from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from stargaze_ml.gold.l2_open_events import VWAP_HORIZONS_SECONDS, build_open_policy_data
from stargaze_ml.gold.frozen_policy import load_frozen_policy_bundle, policy_vwap_horizons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=Path, default=Path("runs/gold_l2_policy_v2/l2_seconds.parquet"))
    parser.add_argument("--base", type=Path, default=Path("runs/gold_l2_policy_v2/prepared_l2_policy.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gold_l2_open_v1"))
    parser.add_argument("--amplitude-ticks", type=float, default=230.0)
    parser.add_argument("--gate-fraction", type=float, default=0.75)
    parser.add_argument("--primary-vwap", default="60")
    parser.add_argument("--feature-profile", choices=("raw", "hierarchy", "leadlag"), default="raw")
    parser.add_argument(
        "--vwap-horizons",
        default=",".join(str(value) for value in VWAP_HORIZONS_SECONDS),
        help="comma-separated causal VWAP horizons in seconds",
    )
    parser.add_argument("--match-train-good-events", type=int)
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="prepare a forward-only dataset without borrowing historical split indices",
    )
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--adaptive-gate-target", type=int)
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
    adaptive_gate_target = args.adaptive_gate_target
    adaptive_gate_initial_amplitudes = None
    adaptive_gate_initial_end_ts_ns = None
    horizons = tuple(int(value) for value in args.vwap_horizons.split(","))
    if (
        tuple(sorted(set(horizons))) != horizons
        or any(value <= 0 for value in horizons)
        or 60 not in horizons
    ):
        raise ValueError(
            "--vwap-horizons must be positive, unique, increasing and include 60"
        )
    if args.policy_bundle is not None:
        if args.match_train_good_events:
            raise ValueError("--policy-bundle cannot be combined with amplitude recalibration")
        bundle = load_frozen_policy_bundle(args.policy_bundle)
        preparation = bundle.policy["preparation"]
        horizons = policy_vwap_horizons(bundle.policy)
        primary_text = str(preparation["primary_vwap"])
        feature_profile = str(preparation["feature_profile"])
        gate_fraction = float(preparation["gate_fraction"])
        amplitude_ticks = float(preparation["amplitude_threshold_ticks"])
        min_duration_seconds = int(preparation["min_duration_seconds"])
        bundle_adaptive_target = preparation.get("adaptive_gate_target_per_active_day")
        if args.adaptive_gate_target is not None and args.adaptive_gate_target != bundle_adaptive_target:
            raise ValueError("--adaptive-gate-target conflicts with the frozen policy bundle")
        adaptive_gate_target = bundle_adaptive_target
        adaptive_history = preparation.get("adaptive_gate_history_tail")
        if adaptive_history is not None:
            adaptive_gate_initial_amplitudes = np.asarray(
                adaptive_history["amplitude_ticks"], dtype=np.float64
            )
            adaptive_gate_initial_end_ts_ns = np.asarray(
                adaptive_history["end_ts_ns"], dtype=np.int64
            )
    primary_vwap: int | str = "ribbon" if primary_text == "ribbon" else int(primary_text)
    if args.match_train_good_events:
        pilot = build_open_policy_data(
            seconds, amplitude_threshold_ticks=1.0,
            gate_fraction=gate_fraction, primary_vwap=primary_vwap,
            feature_profile=feature_profile,
            min_duration_seconds=min_duration_seconds,
            adaptive_gate_target_per_active_day=adaptive_gate_target,
            adaptive_gate_initial_amplitude_ticks=adaptive_gate_initial_amplitudes,
            adaptive_gate_initial_end_ts_ns=adaptive_gate_initial_end_ts_ns,
            horizons=horizons,
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
        adaptive_gate_target_per_active_day=adaptive_gate_target,
        adaptive_gate_initial_amplitude_ticks=adaptive_gate_initial_amplitudes,
        adaptive_gate_initial_end_ts_ns=adaptive_gate_initial_end_ts_ns,
        horizons=horizons,
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
        "vwap_horizons_seconds": list(horizons),
        "direction": "handled by a separate direction model",
        "amplitude_threshold_ticks": amplitude_ticks,
        "gate_fraction": gate_fraction,
        "gate_ticks": amplitude_ticks * gate_fraction,
        "min_duration_seconds": min_duration_seconds,
        "split_contract": "all_forward_test" if args.all_test else "borrowed_from_base",
        "adaptive_gate_target_per_active_day": adaptive_gate_target,
        "train_events": int(train_events.sum()),
        "train_gated_events": int((train_events & e.gated).sum()),
        "train_good_events": int((train_events & e.good).sum()),
    }
    if adaptive_gate_target is not None and args.policy_bundle is None:
        history_count = min(2_000, len(e.end))
        history_events = np.arange(len(e.end) - history_count, len(e.end), dtype=np.int64)
        manifest["adaptive_gate_history_tail"] = {
            "amplitude_ticks": e.amplitude_ticks[history_events].astype(float).tolist(),
            "end_ts_ns": seconds["bar_start_ns"].to_numpy()[e.end[history_events]].astype(np.int64).tolist(),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
