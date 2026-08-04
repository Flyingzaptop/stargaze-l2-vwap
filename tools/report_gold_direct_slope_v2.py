from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HORIZONS = (15, 30, 60)
SLOPE_SCALE = np.asarray((0.57309889793396, 0.40823590755462646, 0.2908809781074524))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/gold_direct_line_v2"))
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("runs/gold_minute_v01/prepared_gold_m1.npz"),
    )
    parser.add_argument(
        "--old-predictions",
        type=Path,
        default=Path("runs/gold_minute_v01/direct_line/holdout_predictions.npz"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def weekly_bootstrap(
    prediction: np.ndarray,
    target: np.ndarray,
    week: np.ndarray,
    *,
    horizon: int,
    slope_scale: float,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    prediction_error = np.abs((prediction - target) * horizon)
    flat_error = np.abs(target * horizon)
    benefit = flat_error - prediction_error
    weeks = np.unique(week)
    benefit_sum = np.asarray([benefit[week == value].sum() for value in weeks])
    flat_sum = np.asarray([flat_error[week == value].sum() for value in weeks])
    sampled = rng.integers(0, len(weeks), size=(draws, len(weeks)))
    improvement_draw = benefit_sum[sampled].sum(axis=1) / flat_sum[sampled].sum(axis=1)

    material = np.abs(target) >= slope_scale
    correct = np.signbit(prediction[material]) == np.signbit(target[material])
    material_week = week[material]
    material_weeks = np.unique(material_week)
    centered_correct_sum = np.asarray(
        [
            correct[material_week == value].sum() - 0.5 * np.sum(material_week == value)
            for value in material_weeks
        ]
    )
    material_count = np.asarray([np.sum(material_week == value) for value in material_weeks])
    sampled_material = rng.integers(0, len(material_weeks), size=(draws, len(material_weeks)))
    excess_draw = (
        centered_correct_sum[sampled_material].sum(axis=1)
        / material_count[sampled_material].sum(axis=1)
    )
    return {
        "weeks": int(len(weeks)),
        "endpoint_improvement_vs_flat": float(benefit.sum() / flat_error.sum()),
        "endpoint_improvement_95ci": np.quantile(improvement_draw, (0.025, 0.975)).tolist(),
        "bootstrap_probability_improvement_positive": float(np.mean(improvement_draw > 0.0)),
        "direction_accuracy_material": float(np.mean(correct)),
        "direction_accuracy_material_95ci": (
            0.5 + np.quantile(excess_draw, (0.025, 0.975))
        ).tolist(),
        "bootstrap_probability_direction_above_50": float(np.mean(excess_draw > 0.0)),
    }


def write_target_audit(
    path: Path,
    *,
    centers: np.ndarray,
    target_slope: np.ndarray,
    close: np.ndarray,
    ts_ns: np.ndarray,
) -> None:
    order = np.argsort(target_slope[:, 2])
    positions = [
        order[min(len(order) - 1, int(q * (len(order) - 1)))]
        for q in (0.02, 0.15, 0.40, 0.60, 0.85, 0.98)
    ]
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), squeeze=False)
    for axis, position in zip(axes.flat, positions, strict=True):
        center = int(centers[position])
        current = float(close[center])
        past_tau = np.arange(-59, 1)
        future_tau = np.arange(0, 61)
        axis.plot(
            past_tau,
            (close[center + past_tau] / current - 1.0) * 10_000.0,
            color="#64748b",
            linewidth=1.1,
            label="past 60m",
        )
        axis.plot(
            future_tau,
            (close[center + future_tau] / current - 1.0) * 10_000.0,
            color="#111827",
            linewidth=1.5,
            label="actual future",
        )
        for column, (horizon, color) in enumerate(
            zip(HORIZONS, ("#f59e0b", "#ef4444", "#2563eb"), strict=True)
        ):
            tau = np.arange(0, horizon + 1)
            slope = target_slope[position, column]
            axis.plot(
                tau,
                slope * tau,
                color=color,
                linestyle="--",
                linewidth=2.0,
                label=f"target H={horizon}: a={slope:+.3f} bps/min",
            )
        timestamp = np.datetime_as_string(np.datetime64(int(ts_ns[center]), "ns"), unit="m")
        axis.set_title(f"{timestamp} UTC")
        axis.axhline(0, color="#cbd5e1", linewidth=0.8)
        axis.axvline(0, color="#94a3b8", linewidth=0.8)
        axis.set_xlabel("minutes from forecast")
        axis.set_ylabel("linear price change from P(t), bps")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Exact target audit: P(t+τ) = P(t) + a_price·τ, b = 0", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    predictions = np.load(run_dir / "final" / "holdout_predictions.npz")
    prepared = np.load(args.prepared)
    old = np.load(args.old_predictions)
    centers = predictions["center"]
    if not np.array_equal(centers, old["center"]):
        raise ValueError("v1 and v2 holdout centers do not align")
    target = predictions["target_slope_bps_per_minute"]
    prediction = predictions["slope_bps_per_minute"]
    old_columns = (2, 4, 6)
    old_prediction = old["mean_bps"][:, old_columns] / np.asarray(HORIZONS)
    week = (
        prepared["ts_ns"][centers]
        // (7 * 24 * 60 * 60 * 1_000_000_000)
    ).astype(np.int64)
    rng = np.random.default_rng(46947)
    statistical: dict[str, object] = {"bootstrap_unit": "UTC week", "draws": args.bootstrap_draws}
    comparison: dict[str, object] = {}
    for column, horizon in enumerate(HORIZONS):
        statistical[str(horizon)] = weekly_bootstrap(
            prediction[:, column],
            target[:, column],
            week,
            horizon=horizon,
            slope_scale=float(SLOPE_SCALE[column]),
            draws=args.bootstrap_draws,
            rng=rng,
        )
        row: dict[str, object] = {}
        material = np.abs(target[:, column]) >= SLOPE_SCALE[column]
        for name, values in (("v1", old_prediction), ("v2", prediction)):
            error = np.abs((values[:, column] - target[:, column]) * horizon)
            flat = np.abs(target[:, column] * horizon)
            row[name] = {
                "endpoint_improvement_vs_flat": float(1.0 - error.mean() / flat.mean()),
                "direction_accuracy_material": float(
                    np.mean(
                        np.signbit(values[material, column])
                        == np.signbit(target[material, column])
                    )
                ),
                "line_correlation": float(
                    np.corrcoef(values[:, column], target[:, column])[0, 1]
                ),
            }
        comparison[str(horizon)] = row
    (run_dir / "statistical_test.json").write_text(
        json.dumps(statistical, indent=2),
        encoding="utf-8",
    )
    (run_dir / "v1_v2_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    write_target_audit(
        run_dir / "audit" / "target_ax_examples_simple_price.png",
        centers=centers,
        target_slope=target,
        close=prepared["close"],
        ts_ns=prepared["ts_ns"],
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    lines = [
        "# Direct anchored-slope TCN v2",
        "",
        "## Target",
        "",
        "For each forecast point and horizon H in {15, 30, 60}, the target is the",
        "least-squares line through the current price:",
        "",
        "`P(t+tau) = P(t) + a_price * tau`,",
        "",
        "where `a_price = sum(tau * (P(t+tau)-P(t))) / sum(tau^2)` and `b = 0`.",
        "The network learns the scale-stable equivalent in bps/minute and converts",
        "it back with `a_price = P(t) * a_bps / 10000`.",
        "",
        "## Diagnosis",
        "",
        "- Stored v1 targets reproduce the analytical OLS formula to floating-point precision.",
        "- The v1 heteroscedastic NLL let predicted uncertainty absorb noisy amplitude errors.",
        "- Sign-only TCN training peaked near 51% and then overfit, showing that signal is weak.",
        "- Five loss/regularization candidates were screened; two finalists were fully trained.",
        f"- Selected candidate: `{summary['selected_candidate']['name']}`, epoch {summary['best_epoch']}.",
        "",
        "## Holdout",
        "",
        "| Horizon | v1 MAE lift vs flat | v2 MAE lift vs flat | v2 material sign | Weekly-bootstrap 95% CI for lift |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in HORIZONS:
        stats = statistical[str(horizon)]
        old_row = comparison[str(horizon)]["v1"]
        new_row = comparison[str(horizon)]["v2"]
        low, high = stats["endpoint_improvement_95ci"]
        lines.append(
            f"| {horizon}m | {old_row['endpoint_improvement_vs_flat'] * 100:.4f}% "
            f"| {new_row['endpoint_improvement_vs_flat'] * 100:.4f}% "
            f"| {new_row['direction_accuracy_material'] * 100:.2f}% "
            f"| [{low * 100:.4f}%, {high * 100:.4f}%] |"
        )
    lines.extend(
        [
            "",
            "The 15-minute result is statistically above the flat baseline under weekly",
            "block bootstrap, but the economic effect is tiny. The 30- and 60-minute",
            "confidence intervals cross zero. Correlation remains approximately zero,",
            "so this is not evidence that OHLCV-only M1 inputs reconstruct future path",
            "amplitude. It is a weak short-horizon directional effect, not a trading edge.",
            "",
        ]
    )
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"statistical": statistical, "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()
