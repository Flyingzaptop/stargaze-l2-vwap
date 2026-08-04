from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import load_frames, write_json
from .curve_pipeline import (
    _backtest,
    _load_execution_quotes,
    _load_targets,
    _prediction_metrics,
    _save_targets,
)
from .deployment import FourCurveRuntime
from .labels import CURVE_NAMES, build_four_curve_targets


def evaluate_four_curve_oos(args: object) -> dict[str, object]:
    """Evaluate a frozen four-curve bundle on a strictly later data extension."""

    started = perf_counter()
    candidate_dir = Path(args.candidate_dir)
    fit_run = Path(args.fit_run)
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    frames = load_frames(Path(args.frames))
    execution_scenarios = _load_execution_quotes(Path(args.execution))
    source_targets = _load_targets(fit_run / "four_curve_targets.npz")
    with np.load(fit_run / "splits.npz", allow_pickle=False) as split_data:
        source_train = np.asarray(split_data["train"], dtype=bool)
    if len(source_train) != len(source_targets.ts_ns):
        raise ValueError("source split and source targets are not aligned")
    if len(frames.ts_ns) <= len(source_targets.ts_ns):
        raise ValueError("OOS frames do not extend beyond the frozen fit dataset")
    if not np.array_equal(frames.ts_ns[: len(source_targets.ts_ns)], source_targets.ts_ns):
        raise ValueError("OOS frame cache does not preserve the frozen fit timestamp prefix")

    cutoff_index = len(source_targets.ts_ns) - 1
    cutoff_ns = int(source_targets.ts_ns[cutoff_index])
    fit_mask = np.zeros(len(frames.ts_ns), dtype=bool)
    fit_mask[: len(source_train)] = source_train
    run_manifest = json.loads((candidate_dir / "four_curve_run.json").read_text(encoding="utf-8"))
    run_args = run_manifest["args"]
    horizons = tuple(float(value) for value in str(run_args["horizons"]).split(",") if value.strip())
    primary_execution = execution_scenarios[float(run_args["latency_ms"])]

    print(json.dumps({"stage": "frozen_oos_targets", "cutoff_ns": cutoff_ns}), flush=True)
    targets = build_four_curve_targets(
        frames,
        horizons_seconds=horizons,
        focus_seconds=float(run_args["focus_seconds"]),
        fit_mask=fit_mask,
        fee_round_trip_bps=float(run_args["fee_round_trip_bps"]),
        latency_ms=float(run_args["latency_ms"]),
        notional_usd=float(run_args["notional_usd"]),
        event_quantile=float(run_args["event_quantile"]),
        peak_floor=float(run_args.get("peak_floor", 0.75)),
        execution_bid=primary_execution.bid,
        execution_ask=primary_execution.ask,
        execution_valid=primary_execution.valid,
        frozen_high_thresholds=source_targets.high_thresholds,
        frozen_full_quality_thresholds=source_targets.full_quality_thresholds,
        forward_curve_mode=str(run_args.get("forward_curve_mode", "peak")),
    )
    _save_targets(destination / "oos_targets.npz", targets)

    evaluation_start_ns = cutoff_ns + int(float(args.start_after_cutoff_seconds) * 1e9)
    evaluation_end_ns = (
        int(frames.ts_ns[-1])
        if float(args.end_after_cutoff_seconds) <= 0.0
        else cutoff_ns + int(float(args.end_after_cutoff_seconds) * 1e9)
    )
    if evaluation_end_ns <= evaluation_start_ns:
        raise ValueError("OOS evaluation interval is empty")
    oos_mask = (frames.ts_ns > evaluation_start_ns) & (frames.ts_ns <= evaluation_end_ns)
    runtime = FourCurveRuntime.load(candidate_dir, device=str(args.device) or None)
    print(json.dumps({"stage": "target_free_oos_inference", "ticks": int(oos_mask.sum())}), flush=True)
    centers, predictions = runtime.score_frames(
        frames,
        mask=oos_mask,
        batch_size=int(args.batch_size),
    )
    if len(centers) == 0 or np.any(frames.ts_ns[centers] <= cutoff_ns):
        raise RuntimeError("OOS inference produced no rows or crossed the frozen cutoff")

    target_valid = np.all(targets.valid[centers], axis=1)
    metric_centers = centers[target_valid]
    metric_predictions = predictions[target_valid]
    prediction_metrics = (
        _prediction_metrics(metric_predictions, targets.values[metric_centers])
        if len(metric_centers)
        else {}
    )
    table = {
        "ts_ns": frames.ts_ns[centers],
        **{name: predictions[:, column] for column, name in enumerate(CURVE_NAMES)},
        **{f"target_{name}": targets.values[centers, column] for column, name in enumerate(CURVE_NAMES)},
        **{f"target_valid_{name}": targets.valid[centers, column] for column, name in enumerate(CURVE_NAMES)},
    }
    pq.write_table(pa.table(table), destination / "oos_predictions.parquet", compression="zstd")

    stress: dict[str, object] = {}
    primary_trades: list[dict[str, object]] = []
    for fee in (4.0, 6.0, 10.0, 14.0):
        for latency in (100.0, 250.0, 500.0):
            result, trades = _backtest(
                frames,
                centers,
                predictions,
                latency_ms=latency,
                notional_usd=float(run_args["notional_usd"]),
                fee_round_trip_bps=fee,
                execution=execution_scenarios[latency],
                curve_thresholds=runtime.curve_thresholds,
            )
            stress[f"fee_{fee:g}bps_latency_{latency:g}ms"] = result
            if fee == float(run_args["fee_round_trip_bps"]) and latency == float(run_args["latency_ms"]):
                primary_trades = trades
    if primary_trades:
        pq.write_table(pa.Table.from_pylist(primary_trades), destination / "oos_trades.parquet", compression="zstd")

    target_events = np.sum(targets.values[metric_centers] >= 0.5, axis=0) if len(metric_centers) else np.zeros(4, dtype=int)
    predicted_events = (
        np.sum(metric_predictions >= np.asarray(runtime.curve_thresholds)[None, :], axis=0)
        if len(metric_centers)
        else np.zeros(4, dtype=int)
    )
    primary_key = f"fee_{float(run_args['fee_round_trip_bps']):g}bps_latency_{float(run_args['latency_ms']):g}ms"
    primary = stress[primary_key]
    output = {
        "protocol": "strict_later_extension_with_frozen_train_thresholds",
        "candidate_dir": str(candidate_dir.resolve()),
        "fit_run": str(fit_run.resolve()),
        "cutoff_ns": cutoff_ns,
        "requested_start_ns": evaluation_start_ns,
        "requested_end_ns": evaluation_end_ns,
        "oos_start_ns": int(frames.ts_ns[centers[0]]),
        "oos_end_ns": int(frames.ts_ns[centers[-1]]),
        "oos_hours": float(frames.ts_ns[centers[-1]] - frames.ts_ns[centers[0]]) / 3.6e12,
        "inference_rows": len(centers),
        "metric_rows": len(metric_centers),
        "frozen_high_thresholds": source_targets.high_thresholds.tolist(),
        "frozen_full_quality_thresholds": source_targets.full_quality_thresholds.tolist(),
        "target_events_by_curve": dict(zip(CURVE_NAMES, (int(value) for value in target_events), strict=True)),
        "predicted_events_by_curve": dict(zip(CURVE_NAMES, (int(value) for value in predicted_events), strict=True)),
        "curve_thresholds": list(runtime.curve_thresholds),
        "score_calibration": "empirical_cdf" if runtime.calibration_reference is not None else "none",
        "prediction_metrics": prediction_metrics,
        "stress": stress,
        "execution_ready": bool(
            int(primary["trades"]) >= 20
            and float(primary["net_bps"]) > 0.0
            and not bool(primary["unresolved_position"])
        ),
        "elapsed_seconds": perf_counter() - started,
    }
    write_json(destination / "oos_summary.json", output)
    print(json.dumps({"stage": "complete", **output}), flush=True)
    return output


__all__ = ["evaluate_four_curve_oos"]
