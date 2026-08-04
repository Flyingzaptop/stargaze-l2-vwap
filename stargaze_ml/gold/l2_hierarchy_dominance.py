"""Interpretable multi-horizon VWAP hierarchy for entry-side selection."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from stargaze_ml.training.data import RobustNormalizer
from .l2_causal_rate import DAY_NS, robust_validation_score, summarize_selected
from .l2_contracts import assert_feature_names
from .l2_multivwap_side import _open_entries
from .l2_open_policy import L2OpenPolicy
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from .l2_profit_direction import executable_side_pnls


HORIZONS = (5, 10, 15, 30, 45, 60, 120, 300, 900)
WINDOWS = (1, 3, 5, 10, 30)


@dataclass(frozen=True)
class HierarchyDominanceConfig:
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    swap_thresholds: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)
    target_trades_per_day: tuple[int, ...] = (10, 15, 20, 25)
    history_size: int = 2_000
    min_history: int = 100
    seed: int = 20260809


def hierarchy_summary(oriented_gaps: np.ndarray, oriented_slopes: np.ndarray) -> np.ndarray:
    """Compress one causal VWAP ribbon into scale-interaction statistics."""

    gaps = np.asarray(oriented_gaps, dtype=np.float64)
    slopes = np.asarray(oriented_slopes, dtype=np.float64)
    if gaps.shape != (len(HORIZONS),) or slopes.shape != gaps.shape:
        raise ValueError("hierarchy arrays must match HORIZONS")
    scale = np.log(np.asarray(HORIZONS, dtype=np.float64))
    centered = scale - scale.mean()
    denominator = float(np.dot(centered, centered))
    gap_scale_slope = float(np.dot(centered, gaps - gaps.mean()) / denominator)
    slope_scale_slope = float(np.dot(centered, slopes - slopes.mean()) / denominator)
    short = gaps[:5]
    long = gaps[6:]
    short_slopes = slopes[:5]
    long_slopes = slopes[6:]
    return np.asarray(
        [
            float(np.mean(np.sign(gaps))),
            float(np.mean(gaps > 0)),
            float(np.mean(gaps)),
            float(np.std(gaps)),
            gap_scale_slope,
            float(np.mean(short)),
            float(np.mean(long)),
            float(np.mean(long) - np.mean(short)),
            float(np.mean(slopes)),
            float(np.std(slopes)),
            slope_scale_slope,
            float(np.mean(short_slopes)),
            float(np.mean(long_slopes)),
            float(np.mean(long_slopes) - np.mean(short_slopes)),
        ],
        dtype=np.float64,
    )


SUMMARY_NAMES = (
    "gap_sign_consensus",
    "gap_same_side_fraction",
    "gap_mean",
    "gap_std",
    "gap_scale_slope",
    "gap_short_mean",
    "gap_long_mean",
    "gap_long_minus_short",
    "slope_mean",
    "slope_std",
    "slope_scale_slope",
    "slope_short_mean",
    "slope_long_mean",
    "slope_long_minus_short",
)


def _feature_indices(data: PreparedOpenData) -> dict[str, int]:
    index = {name: i for i, name in enumerate(data.feature_names)}
    required = ["close_delta_1s_ticks", "spread_ticks", "event_age_seconds", "event_fill_ratio"]
    required += [f"mid_vwap_{h}s_minus_mid_ticks" for h in HORIZONS]
    required += [f"mid_vwap_{h}s_slope_1s_ticks" for h in HORIZONS]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"prepared data lacks hierarchy fields: {', '.join(missing)}")
    return index


def build_hierarchy_entry_features(
    data: PreparedOpenData,
    entries: list[tuple[int, int]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build stationary, causal entry features in the local-excursion frame."""

    feature_index = _feature_indices(data)
    gap_columns = np.asarray(
        [feature_index[f"mid_vwap_{h}s_minus_mid_ticks"] for h in HORIZONS], dtype=np.int64
    )
    slope_columns = np.asarray(
        [feature_index[f"mid_vwap_{h}s_slope_1s_ticks"] for h in HORIZONS], dtype=np.int64
    )
    price_delta_column = feature_index["close_delta_1s_ticks"]
    context_names = (
        "spread_ticks",
        "event_age_seconds",
        "event_abs_delta_ticks",
        "event_running_max_delta_ticks",
        "event_running_area_tick_seconds",
        "event_fill_ratio",
        "event_current_to_max_ratio",
    )
    context_columns = [feature_index[name] for name in context_names]
    names: list[str] = ["local_side"]
    for window in WINDOWS:
        names.extend(f"w{window}_{name}" for name in SUMMARY_NAMES)
        names.extend(
            (
                f"w{window}_oriented_price_velocity",
                f"w{window}_local_gap_change",
                f"w{window}_local_relative_velocity",
            )
        )
    names.extend(context_names)

    output = np.empty((len(entries), len(names)), dtype=np.float64)
    for row, (event, entry) in enumerate(entries):
        local_gap = float(data.x[entry, gap_columns[HORIZONS.index(60)]])
        local_side = 1.0 if local_gap >= 0 else -1.0
        values: list[float] = [local_side]
        event_start = int(data.event_start[event])
        for window in WINDOWS:
            left = max(event_start, int(entry) - int(window) + 1)
            valid = data.valid_feature[left : entry + 1]
            block = data.x[left : entry + 1]
            if np.any(valid):
                block = block[valid]
            else:
                block = data.x[entry : entry + 1]
            gaps = local_side * np.mean(block[:, gap_columns], axis=0)
            slopes = local_side * np.mean(block[:, slope_columns], axis=0)
            price_velocity = local_side * float(np.mean(block[:, price_delta_column]))
            current_gap = local_side * float(data.x[entry, gap_columns[HORIZONS.index(60)]])
            first_gap = local_side * float(block[0, gap_columns[HORIZONS.index(60)]])
            local_gap_change = current_gap - first_gap
            local_relative_velocity = float(slopes[HORIZONS.index(60)] - price_velocity)
            values.extend(hierarchy_summary(gaps, slopes).tolist())
            values.extend((price_velocity, local_gap_change, local_relative_velocity))
        values.extend(float(data.x[entry, column]) for column in context_columns)
        output[row] = values
    if not np.all(np.isfinite(output)):
        raise ValueError("hierarchy entry features contain non-finite values")
    return output.astype(np.float32), tuple(names)


def _entry_examples(
    data: PreparedOpenData,
    model: L2OpenPolicy,
    normalizer: RobustNormalizer,
    events: np.ndarray,
    threshold: float,
    device: torch.device,
    market: OpenReinforceConfig,
) -> list[dict[str, float | int]]:
    entries = _open_entries(model, data, normalizer, events, threshold, device)
    rows: list[dict[str, float | int]] = []
    for event, entry in entries.items():
        crossing = int(data.event_crossing_1[event])
        long_pnl, short_pnl = executable_side_pnls(data, entry, crossing, market)
        local_side = 1 if data.primary_vwap[entry] >= data.mid[entry] else -1
        local_pnl = float(long_pnl[0] if local_side > 0 else short_pnl[0])
        continuation_pnl = float(short_pnl[0] if local_side > 0 else long_pnl[0])
        rows.append(
            {
                "event_index": int(event),
                "entry_index": int(entry),
                "entry_ts_ns": int(data.ts_ns[entry]),
                "local_side": int(local_side),
                "long_pnl": float(long_pnl[0]),
                "short_pnl": float(short_pnl[0]),
                "local_pnl": local_pnl,
                "continuation_pnl": continuation_pnl,
                "dominance_target": int(continuation_pnl > local_pnl),
                "target_weight": float(np.clip(abs(continuation_pnl - local_pnl) / 100.0, 0.25, 10.0)),
            }
        )
    return rows


def _apply_policy(
    rows: list[dict[str, float | int]],
    probability: np.ndarray,
    *,
    swap_threshold: float,
) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    for source, prediction in zip(rows, probability, strict=True):
        swap = float(prediction) >= float(swap_threshold)
        chosen = -int(source["local_side"]) if swap else int(source["local_side"])
        confidence = float(prediction if swap else 1.0 - prediction)
        row = dict(source)
        row.update(
            {
                "dominance_probability": float(prediction),
                "swap": int(swap),
                "selected_side": chosen,
                "selection_score": confidence,
                "realized_pnl": float(source["continuation_pnl"] if swap else source["local_pnl"]),
            }
        )
        output.append(row)
    return output


def _causal_rate(
    rows: list[dict[str, float | int]],
    *,
    target: int,
    expected_candidates_per_day: float,
    initial_scores: list[float],
    history_size: int,
    min_history: int,
) -> list[dict[str, float | int]]:
    quantile = float(np.clip(1.0 - target / expected_candidates_per_day, 0.0, 1.0))
    history: deque[float] = deque(initial_scores, maxlen=history_size)
    chosen: list[dict[str, float | int]] = []
    current_day: int | None = None
    daily_count = 0
    for row in sorted(rows, key=lambda item: int(item["entry_ts_ns"])):
        day = int(row["entry_ts_ns"]) // DAY_NS
        if day != current_day:
            current_day = day
            daily_count = 0
        score = float(row["selection_score"])
        cutoff = (
            float(np.quantile(np.asarray(history), quantile))
            if len(history) >= min_history
            else 1.0
        )
        history.append(score)
        if daily_count < target and score >= cutoff:
            chosen.append(row)
            daily_count += 1
    return chosen


def _days(rows: list[dict[str, float | int]]) -> float:
    if len(rows) < 2:
        return 1.0
    return max((int(rows[-1]["entry_ts_ns"]) - int(rows[0]["entry_ts_ns"])) / DAY_NS, 1.0)


def run_hierarchy_dominance_experiment(
    prepared_path: str | Path,
    open_checkpoint_path: str | Path,
    output_dir: str | Path,
    config: HierarchyDominanceConfig = HierarchyDominanceConfig(),
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    np.random.seed(config.seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name if device_name != "auto" else "cpu"
    )
    data = PreparedOpenData(prepared_path)
    checkpoint = torch.load(Path(open_checkpoint_path).resolve(strict=True), map_location=device, weights_only=False)
    assert_feature_names(tuple(checkpoint["feature_names"]), data.feature_names, artifact="open checkpoint")
    market = OpenReinforceConfig(**checkpoint["config"])
    normalizer = RobustNormalizer.from_dict(checkpoint["normalizer"])
    open_model = L2OpenPolicy(len(data.feature_names), market.hidden_size).to(device)
    open_model.load_state_dict(checkpoint["model_state"])
    open_model.eval()
    threshold = float(checkpoint["validation"]["best"]["threshold"])

    ranges = {
        "train": (0, data.train_end),
        "validation": (data.train_end, data.validation_end),
        "test": (data.validation_end, len(data.x)),
    }
    examples: dict[str, list[dict[str, float | int]]] = {}
    features: dict[str, np.ndarray] = {}
    feature_names: tuple[str, ...] = ()
    for name, (left, right) in ranges.items():
        events = _event_indices(data, left, right, good_only=False)
        rows = _entry_examples(data, open_model, normalizer, events, threshold, device, market)
        matrix, current_names = build_hierarchy_entry_features(
            data, [(int(row["event_index"]), int(row["entry_index"])) for row in rows]
        )
        examples[name] = rows
        features[name] = matrix
        feature_names = current_names

    y_train = np.asarray([row["dominance_target"] for row in examples["train"]], dtype=np.int32)
    w_train = np.asarray([row["target_weight"] for row in examples["train"]], dtype=np.float64)
    expected = len(examples["train"]) / _days(examples["train"])
    grid: list[dict[str, Any]] = []
    trained: dict[float, Pipeline] = {}
    validation_probabilities: dict[float, np.ndarray] = {}
    train_probabilities: dict[float, np.ndarray] = {}
    for c_value in config.c_values:
        model = Pipeline(
            [
                ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
                (
                    "model",
                    LogisticRegression(
                        C=float(c_value),
                        max_iter=2_000,
                        random_state=config.seed,
                    ),
                ),
            ]
        )
        model.fit(features["train"], y_train, model__sample_weight=w_train)
        trained[float(c_value)] = model
        train_probability = model.predict_proba(features["train"])[:, 1]
        val_probability = model.predict_proba(features["validation"])[:, 1]
        train_probabilities[float(c_value)] = train_probability
        validation_probabilities[float(c_value)] = val_probability
        for swap_threshold in config.swap_thresholds:
            train_policy = _apply_policy(
                examples["train"], train_probability, swap_threshold=swap_threshold
            )
            validation_policy = _apply_policy(
                examples["validation"], val_probability, swap_threshold=swap_threshold
            )
            initial = [float(row["selection_score"]) for row in train_policy[-config.history_size :]]
            for target in config.target_trades_per_day:
                selected = _causal_rate(
                    validation_policy,
                    target=target,
                    expected_candidates_per_day=expected,
                    initial_scores=initial,
                    history_size=config.history_size,
                    min_history=config.min_history,
                )
                metrics = summarize_selected(selected)
                grid.append(
                    {
                        "c": float(c_value),
                        "swap_threshold": float(swap_threshold),
                        "target_trades_per_day": int(target),
                        **metrics,
                        "robust_score": robust_validation_score(metrics),
                    }
                )
    eligible = [row for row in grid if int(row["trades"]) >= 40]
    selected = max(eligible or grid, key=lambda row: float(row["robust_score"]))
    selected_c = float(selected["c"])
    selected_threshold = float(selected["swap_threshold"])
    selected_target = int(selected["target_trades_per_day"])
    chosen_model = trained[selected_c]
    train_policy = _apply_policy(
        examples["train"], train_probabilities[selected_c], swap_threshold=selected_threshold
    )
    validation_policy = _apply_policy(
        examples["validation"], validation_probabilities[selected_c], swap_threshold=selected_threshold
    )
    test_probability = chosen_model.predict_proba(features["test"])[:, 1]
    test_policy = _apply_policy(examples["test"], test_probability, swap_threshold=selected_threshold)
    initial_test = [
        float(row["selection_score"])
        for row in (train_policy + validation_policy)[-config.history_size :]
    ]
    fixed_test_rows = _causal_rate(
        test_policy,
        target=selected_target,
        expected_candidates_per_day=expected,
        initial_scores=initial_test,
        history_size=config.history_size,
        min_history=config.min_history,
    )
    fixed_test = summarize_selected(fixed_test_rows)
    local_baseline = summarize_selected(
        [{**row, "realized_pnl": float(row["local_pnl"])} for row in fixed_test_rows]
    )
    oracle = summarize_selected(
        [
            {**row, "realized_pnl": max(float(row["local_pnl"]), float(row["continuation_pnl"]))}
            for row in fixed_test_rows
        ]
    )
    report = {
        "config": asdict(config),
        "device": str(device),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "entries": {name: len(rows) for name, rows in examples.items()},
        "expected_candidates_per_day_train": expected,
        "selected_on_validation": selected,
        "fixed_test": fixed_test,
        "fixed_test_local60_baseline_same_entries": local_baseline,
        "fixed_test_oracle_same_entries": oracle,
        "test_swap_fraction": float(np.mean([row["swap"] for row in fixed_test_rows])) if fixed_test_rows else 0.0,
        "validation_grid": grid,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": chosen_model,
            "config": asdict(config),
            "feature_names": feature_names,
            "prepared_feature_names": data.feature_names,
            "open_threshold": threshold,
            "selection": selected,
        },
        output / "final.joblib",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
