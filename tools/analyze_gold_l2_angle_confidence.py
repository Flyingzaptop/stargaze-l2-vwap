from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 46947848
COVERAGES = (1.0, 0.5, 0.2, 0.1, 0.05)
HORIZONS_SECONDS = (2, 6, 10, 30, 60)
MODEL = "catboost_independent_full_l2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit angle quality versus forecast coverage.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("runs/gold_l2_angle_v1/models_v2/holdout_predictions.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/gold_l2_angle_v1/models_v2"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def _day_bootstrap_accuracy(
    correct: np.ndarray,
    selected: np.ndarray,
    days: np.ndarray,
    *,
    samples: int,
) -> dict[str, object]:
    minimum_rows = 100
    per_day = [
        (
            int(day),
            int(np.sum(selected & (days == day))),
            float(correct[selected & (days == day)].mean()),
        )
        for day in np.unique(days)
        if np.any(selected & (days == day))
    ]
    retained = [row for row in per_day if row[1] >= minimum_rows]
    day_values = np.asarray([row[2] for row in retained], dtype=np.float64)
    if len(day_values) == 0:
        return {
            "days": 0,
            "minimum_selected_rows_per_day": minimum_rows,
            "excluded_partial_days": len(per_day),
        }
    rng = np.random.default_rng(SEED)
    draws = rng.choice(
        day_values,
        size=(int(samples), len(day_values)),
        replace=True,
    ).mean(axis=1)
    return {
        "days": int(len(day_values)),
        "minimum_selected_rows_per_day": minimum_rows,
        "excluded_partial_days": int(len(per_day) - len(retained)),
        "equal_day_accuracy": float(day_values.mean()),
        "equal_day_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "probability_accuracy_above_half": float(np.mean(draws > 0.5)),
        "daily": [
            {"day_id": day, "rows": rows, "accuracy": accuracy}
            for day, rows, accuracy in retained
        ],
    }


def main() -> None:
    args = parse_args()
    frame = pl.read_parquet(args.predictions)
    ts_ns = frame["ts_ns"].to_numpy().astype(np.int64)
    days = ts_ns // 86_400_000_000_000
    audit: dict[str, object] = {
        "model": MODEL,
        "selection": (
            "Within each horizon select the largest absolute model angles. "
            "Coverage fractions are fixed before reading target values."
        ),
        "holdout_rows": frame.height,
        "horizons": {},
    }

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    coverages_percent = np.asarray(COVERAGES) * 100.0
    for horizon in HORIZONS_SECONDS:
        target = frame[f"target_angle_deg_{horizon}s"].to_numpy().astype(np.float64)
        prediction = frame[f"{MODEL}_angle_deg_{horizon}s"].to_numpy().astype(np.float64)
        correct = np.signbit(prediction) == np.signbit(target)
        rows: dict[str, object] = {}
        accuracies: list[float] = []
        improvements: list[float] = []
        for coverage in COVERAGES:
            threshold = (
                0.0
                if coverage == 1.0
                else float(np.quantile(np.abs(prediction), 1.0 - coverage))
            )
            selected = np.abs(prediction) >= threshold
            baseline_error = np.abs(target[selected])
            model_error = np.abs(prediction[selected] - target[selected])
            accuracy = float(correct[selected].mean())
            improvement = float(
                1.0 - model_error.mean() / max(baseline_error.mean(), 1e-12)
            )
            accuracies.append(accuracy)
            improvements.append(improvement)
            rows[f"{coverage:.2f}"] = {
                "rows": int(selected.sum()),
                "minimum_absolute_prediction_degrees": threshold,
                "direction_accuracy": accuracy,
                "angle_mae_improvement_vs_zero": improvement,
                "correlation": (
                    float(np.corrcoef(prediction[selected], target[selected])[0, 1])
                    if np.std(prediction[selected]) > 1e-12
                    else 0.0
                ),
                "day_block_bootstrap": _day_bootstrap_accuracy(
                    correct,
                    selected,
                    days,
                    samples=args.bootstrap_samples,
                ),
            }
        audit["horizons"][str(horizon)] = rows
        axes[0].plot(
            coverages_percent,
            np.asarray(accuracies) * 100.0,
            marker="o",
            label=f"{horizon}s",
        )
        axes[1].plot(
            coverages_percent,
            np.asarray(improvements) * 100.0,
            marker="o",
            label=f"{horizon}s",
        )

    for axis in axes:
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("forecast coverage, % (right = more selective)")
        axis.grid(alpha=0.25)
        axis.legend(title="horizon")
    axes[0].axhline(50.0, color="#94a3b8", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("direction accuracy, %")
    axes[0].set_title("Direction improves when the model is confident")
    axes[1].axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("angle MAE improvement vs zero, %")
    axes[1].set_title("Magnitude error improvement versus a flat line")
    figure.suptitle("XAUUSD L2 holdout confidence/coverage audit", fontsize=14)
    figure.tight_layout()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_dir / "confidence_coverage.png", dpi=160)
    plt.close(figure)
    (out_dir / "confidence_analysis.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"stage": "complete", **audit["horizons"]}), flush=True)


if __name__ == "__main__":
    main()
