from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
import numpy as np

from stargaze_ml.gold.l2_angle import (
    AngleTargets,
    L2FeatureMatrix,
    build_angle_targets,
    build_l2_feature_matrix,
    reconstruct_l2_bars,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HORIZONS_SECONDS = (2, 6, 10, 30, 60)
STEP_SECONDS = 2
CONTEXT_STEPS = 30
FEATURE_HISTORY_STEPS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct conservative cTrader L2 bars and angle targets.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(r"C:\Users\r3d_flzp\Documents\GitHub\golden-den\raw.parquet"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gold_l2_angle_v1"))
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--max-new-quotes", type=int, default=20)
    parser.add_argument("--min-levels", type=int, default=3)
    parser.add_argument("--max-spread-ticks", type=float, default=500.0)
    parser.add_argument(
        "--target-price",
        choices=("mid", "microprice"),
        default="mid",
        help="Price series used to fit the future anchored line.",
    )
    return parser.parse_args()


def _iso_utc(ts_ns: int) -> str:
    return datetime.fromtimestamp(int(ts_ns) / 1e9, tz=timezone.utc).isoformat()


def _target_plot(
    path: Path,
    features: L2FeatureMatrix,
    targets: AngleTargets,
    centers: np.ndarray,
    *,
    target_price: np.ndarray,
    target_price_name: str,
    tick_size: float,
) -> None:
    horizon_column = len(targets.horizons_steps) - 1
    ordered = centers[np.argsort(targets.angle_radians[centers, horizon_column])]
    quantiles = np.linspace(0.03, 0.97, 6)
    selected = ordered[(quantiles * (len(ordered) - 1)).astype(np.int64)]
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), squeeze=False)
    horizon_steps = targets.horizons_steps[horizon_column]
    horizon_seconds = horizon_steps * targets.step_seconds
    for axis, center in zip(axes.flat, selected, strict=True):
        center = int(center)
        past_tau = np.arange(-CONTEXT_STEPS + 1, 1) * targets.step_seconds
        past = (
            target_price[center - CONTEXT_STEPS + 1 : center + 1] - target_price[center]
        ) / tick_size
        future_tau = np.arange(0, horizon_steps + 1) * targets.step_seconds
        actual = (
            target_price[center : center + horizon_steps + 1] - target_price[center]
        ) / tick_size
        line = targets.slope_ticks_per_second[center, horizon_column] * future_tau
        other_price_now = (
            features.microprice[center] - features.mid[center]
        ) / tick_size
        theta = np.degrees(targets.angle_radians[center, horizon_column])
        axis.plot(
            past_tau,
            past,
            color="#64748b",
            linewidth=1.2,
            label=f"past {target_price_name}",
        )
        axis.plot(future_tau, actual, color="#111827", linewidth=1.6, label="actual future")
        axis.plot(future_tau, line, color="#dc2626", linewidth=2.0, label="OLS target ax")
        if target_price_name == "mid":
            axis.scatter(
                [0],
                [other_price_now],
                color="#16a34a",
                s=35,
                zorder=5,
                label="microprice at t",
            )
        axis.axvline(0, color="#94a3b8", linewidth=0.8)
        axis.axhline(0, color="#cbd5e1", linewidth=0.8)
        stamp = np.datetime_as_string(
            np.datetime64(int(features.ts_ns[center]), "ns"),
            unit="s",
        )
        axis.set_title(
            f"{stamp} UTC | H={horizon_seconds}s | theta={theta:+.1f} deg | "
            f"sigma={targets.past_sigma_ticks_sqrt_second[center]:.2f}"
        )
        axis.set_xlabel("seconds from forecast")
        axis.set_ylabel("ticks relative to mid(t)")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(
        f"XAUUSD L2 volatility-normalized angle targets "
        f"(line anchored at current {target_price_name})",
        fontsize=14,
    )
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.967), ncol=4)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "reconstruct_start", "raw": str(args.raw.resolve())}), flush=True)
    bars = reconstruct_l2_bars(
        args.raw,
        bar_seconds=STEP_SECONDS,
        tick_size=args.tick_size,
        min_levels_per_side=args.min_levels,
        max_new_quotes_per_timestamp=args.max_new_quotes,
        max_spread_ticks=args.max_spread_ticks,
        timestamp_unit="ms",
    )
    bars_path = out_dir / "l2_bars_2s.parquet"
    bars.write_parquet(bars_path, compression="zstd", statistics=True)
    print(
        json.dumps(
            {
                "stage": "bars_ready",
                "rows": bars.height,
                "segments": int(bars["segment_id"].max()) + 1,
                "path": str(bars_path),
            }
        ),
        flush=True,
    )

    features = build_l2_feature_matrix(
        bars,
        tick_size=args.tick_size,
        minimum_history_steps=FEATURE_HISTORY_STEPS,
    )
    horizons_steps = tuple(value // STEP_SECONDS for value in HORIZONS_SECONDS)
    target_price = (
        features.mid
        if args.target_price == "mid"
        else features.microprice
    )
    targets = build_angle_targets(
        target_price,
        features.segment_id,
        horizons_steps,
        step_seconds=STEP_SECONDS,
        tick_size=args.tick_size,
        vol_window_steps=30,
    )
    valid = features.valid_feature & np.all(targets.valid, axis=1)
    indices = np.arange(len(valid))
    split_1 = int(len(valid) * 0.60)
    split_2 = int(len(valid) * 0.80)
    max_horizon = max(horizons_steps)
    train = valid & (indices < split_1 - max_horizon)
    validation = (
        valid
        & (indices >= split_1 + CONTEXT_STEPS)
        & (indices < split_2 - max_horizon)
    )
    holdout = valid & (indices >= split_2 + CONTEXT_STEPS)
    if min(train.sum(), validation.sum(), holdout.sum()) < 10_000:
        raise RuntimeError("insufficient valid rows in one or more chronological splits")

    prepared_path = out_dir / "prepared_l2_angle.npz"
    np.savez_compressed(
        prepared_path,
        ts_ns=features.ts_ns,
        segment_id=features.segment_id,
        mid=features.mid,
        microprice=features.microprice,
        x=features.x,
        feature_names=np.asarray(features.feature_names),
        valid_feature=features.valid_feature,
        horizons_seconds=np.asarray(HORIZONS_SECONDS, dtype=np.int32),
        angle_radians=targets.angle_radians,
        slope_ticks_per_second=targets.slope_ticks_per_second,
        line_end_ticks=targets.line_end_ticks,
        actual_end_ticks=targets.actual_end_ticks,
        path_rmse_ticks=targets.path_rmse_ticks,
        past_sigma_ticks_sqrt_second=targets.past_sigma_ticks_sqrt_second,
        target_valid=targets.valid,
        target_price_name=np.asarray(args.target_price),
        train=train,
        validation=validation,
        holdout=holdout,
    )
    audit_centers = np.flatnonzero(holdout)
    _target_plot(
        out_dir / "target_angle_examples.png",
        features,
        targets,
        audit_centers,
        target_price=target_price,
        target_price_name=args.target_price,
        tick_size=args.tick_size,
    )

    manifest = {
        "source": {
            "path": str(args.raw.resolve()),
            "size_bytes": args.raw.resolve().stat().st_size,
        },
        "reconstruction": {
            "contract": (
                "exact-timestamp positive new rows -> reject one-sided/crossed/ambiguous "
                "packets -> last accepted snapshot per 2-second bar"
            ),
            "bar_seconds": STEP_SECONDS,
            "tick_size": args.tick_size,
            "min_levels_per_side": args.min_levels,
            "max_new_quotes_per_timestamp": args.max_new_quotes,
            "max_spread_ticks": args.max_spread_ticks,
            "bars": bars.height,
            "segments": int(bars["segment_id"].max()) + 1,
            "first_utc": _iso_utc(features.ts_ns[0]),
            "last_utc": _iso_utc(features.ts_ns[-1]),
        },
        "target": {
            "horizons_seconds": HORIZONS_SECONDS,
            "formula": (
                f"a=sum(tau*({args.target_price}[t+tau]-{args.target_price}[t])/tick)"
                "/sum(tau^2); "
                "theta=atan(a*sqrt(H_seconds)/sigma_past)"
            ),
            "anchor": f"{args.target_price}(t), intercept fixed to zero",
            "past_sigma": (
                f"RMS {args.target_price} tick change per sqrt(second), trailing 60 seconds"
            ),
            "training_unit": "radians",
            "display_unit": "degrees",
        },
        "features": {
            "count": len(features.feature_names),
            "names": list(features.feature_names),
            "minimum_history_seconds": FEATURE_HISTORY_STEPS * STEP_SECONDS,
        },
        "splits": {
            "method": "chronological 60/20/20 with context and max-horizon purge",
            "context_seconds": CONTEXT_STEPS * STEP_SECONDS,
            "train_rows": int(train.sum()),
            "validation_rows": int(validation.sum()),
            "holdout_rows": int(holdout.sum()),
            "train_end_utc": _iso_utc(features.ts_ns[np.flatnonzero(train)[-1]]),
            "validation_start_utc": _iso_utc(features.ts_ns[np.flatnonzero(validation)[0]]),
            "validation_end_utc": _iso_utc(features.ts_ns[np.flatnonzero(validation)[-1]]),
            "holdout_start_utc": _iso_utc(features.ts_ns[np.flatnonzero(holdout)[0]]),
        },
        "angle_degrees": {
            str(horizon): {
                "q05": float(np.degrees(np.quantile(targets.angle_radians[valid, column], 0.05))),
                "median": float(np.degrees(np.median(targets.angle_radians[valid, column]))),
                "q95": float(np.degrees(np.quantile(targets.angle_radians[valid, column], 0.95))),
            }
            for column, horizon in enumerate(HORIZONS_SECONDS)
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out_dir / "prepared_l2_angle.manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"stage": "complete", **manifest["splits"]}), flush=True)


if __name__ == "__main__":
    main()
