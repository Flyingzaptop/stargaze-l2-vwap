from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from market_collector.record_log import copy_record_log, inspect_record_log

from .artifacts import load_frames, write_json
from .config import DataConfig
from .contracts import CausalFrames
from .data import CausalReplayBuilder, DatasetCatalog, ExecutionQuotes, build_execution_quote_scenarios
from .labels import CURVE_NAMES, FourCurveTargets, build_four_curve_targets
from .labels.curves import execution_quotes
from .models import CurveModelConfig, FourCurveCausalTransformer
from .training import (
    CurveWindowDataset,
    CurveInferenceDataset,
    RobustNormalizer,
    causal_backward_score_features,
    causal_centers,
    causal_high_order_features,
    causal_score_features_for_normalizer,
    curve_centers,
    multihorizon_forward_edge_targets,
    stationary_market_features,
    purged_blocked_splits,
    predict_curve_model,
    purged_chronological_splits,
    train_curve_model,
)


def _immutable_snapshot(raw_dir: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(raw_dir.glob("*.mrec")):
        if source.name.startswith("~syncthing~") or source.stat().st_size <= 8:
            continue
        target = destination / source.name
        if target.exists():
            inspect_record_log(target)
            continue
        copy_record_log(source, target)
    return destination


def _save_targets(path: Path, targets: FourCurveTargets) -> None:
    np.savez_compressed(
        path,
        ts_ns=targets.ts_ns,
        values=targets.values,
        valid=targets.valid,
        raw_scores=targets.raw_scores,
        horizons_seconds=targets.horizons_seconds,
        horizon_weights=targets.horizon_weights,
        high_thresholds=targets.high_thresholds,
        full_quality_thresholds=targets.full_quality_thresholds,
        curve_names=np.asarray(targets.curve_names),
    )


def _load_targets(path: Path) -> FourCurveTargets:
    with np.load(path, allow_pickle=False) as data:
        return FourCurveTargets(
            ts_ns=data["ts_ns"],
            values=data["values"],
            valid=data["valid"],
            raw_scores=data["raw_scores"],
            horizons_seconds=data["horizons_seconds"],
            horizon_weights=data["horizon_weights"],
            high_thresholds=data["high_thresholds"],
            full_quality_thresholds=data["full_quality_thresholds"],
            curve_names=tuple(str(x) for x in data["curve_names"]),
        )


def _dense_score_auxiliary_targets(
    targets: FourCurveTargets,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(targets.valid, dtype=bool)
    raw = np.asarray(targets.raw_scores, dtype=np.float32)
    dense = np.zeros_like(raw, dtype=np.float32)
    dense[valid] = np.log1p(np.maximum(raw[valid], 0.0)).astype(np.float32)
    if not np.isfinite(dense).all():
        raise ValueError("valid dense score auxiliary targets must be finite")
    return dense, valid.copy()


def _normalizers(
    frames: CausalFrames,
    train_mask: np.ndarray,
    causal_score_x: np.ndarray,
) -> tuple[RobustNormalizer, RobustNormalizer]:
    global_x, venue_x = stationary_market_features(frames)
    venue_centers: list[np.ndarray] = []
    venue_scales: list[np.ndarray] = []
    for venue_index in range(venue_x.shape[1]):
        fresh = (
            np.asarray(train_mask, dtype=bool)
            & (frames.venue_x[:, venue_index, 0] > 0.5)
            & (frames.venue_x[:, venue_index, 1] <= 2_000.0)
        )
        fitted = RobustNormalizer.fit(venue_x[:, venue_index], fresh)
        venue_centers.append(fitted.center)
        venue_scales.append(fitted.scale)
    return (
        RobustNormalizer.fit(np.concatenate((global_x, causal_score_x), axis=1), train_mask),
        RobustNormalizer(np.stack(venue_centers), np.stack(venue_scales)),
    )


def _save_execution_quotes(path: Path, scenarios: dict[float, ExecutionQuotes]) -> None:
    payload: dict[str, np.ndarray] = {}
    for latency, quotes in scenarios.items():
        key = f"{latency:g}ms"
        payload[f"bid_{key}"] = quotes.bid
        payload[f"ask_{key}"] = quotes.ask
        payload[f"valid_{key}"] = quotes.valid
    np.savez_compressed(path, **payload)


def _load_execution_quotes(path: Path) -> dict[float, ExecutionQuotes]:
    result: dict[float, ExecutionQuotes] = {}
    with np.load(path, allow_pickle=False) as data:
        for latency in (100.0, 250.0, 500.0):
            key = f"{latency:g}ms"
            if f"bid_{key}" in data:
                result[latency] = ExecutionQuotes(latency, data[f"bid_{key}"], data[f"ask_{key}"], data[f"valid_{key}"])
    return result


def _prediction_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
        truth = np.asarray(labels, dtype=bool)
        positives = int(truth.sum())
        if positives == 0:
            return 0.0
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
        ranked = truth[order]
        precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
        return float(np.sum(precision[ranked]) / positives)

    mae = np.mean(np.abs(predictions - targets), axis=0)
    correlation: list[float] = []
    for column in range(4):
        if np.std(predictions[:, column]) <= 1e-12 or np.std(targets[:, column]) <= 1e-12:
            correlation.append(0.0)
        else:
            correlation.append(float(np.corrcoef(predictions[:, column], targets[:, column])[0, 1]))
    event_target = targets >= 0.5
    event_prediction = predictions >= 0.5
    tp = np.sum(event_prediction & event_target, axis=0)
    fp = np.sum(event_prediction & ~event_target, axis=0)
    fn = np.sum(~event_prediction & event_target, axis=0)
    return {
        "mae_by_curve": {name: float(mae[i]) for i, name in enumerate(CURVE_NAMES)},
        "correlation_by_curve": {name: correlation[i] for i, name in enumerate(CURVE_NAMES)},
        "peak_precision_by_curve": {
            name: float(tp[i] / max(tp[i] + fp[i], 1)) for i, name in enumerate(CURVE_NAMES)
        },
        "peak_recall_by_curve": {
            name: float(tp[i] / max(tp[i] + fn[i], 1)) for i, name in enumerate(CURVE_NAMES)
        },
        "peak_average_precision_by_curve": {
            name: average_precision(event_target[:, i], predictions[:, i])
            for i, name in enumerate(CURVE_NAMES)
        },
        "zone_average_precision_by_curve": {
            name: average_precision(targets[:, i] >= 0.05, predictions[:, i])
            for i, name in enumerate(CURVE_NAMES)
        },
    }


def _backtest(
    frames: CausalFrames,
    centers: np.ndarray,
    predictions: np.ndarray,
    *,
    latency_ms: float,
    notional_usd: float,
    fee_round_trip_bps: float,
    execution: ExecutionQuotes | None = None,
    reset_gap_seconds: float = 3_600.0,
    open_threshold: float = 0.5,
    close_threshold: float = 0.5,
    curve_thresholds: tuple[float, float, float, float] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if execution is None:
        bid, ask, valid = execution_quotes(frames, latency_ms=latency_ms, notional_usd=notional_usd)
    else:
        bid, ask, valid = execution.bid, execution.ask, execution.valid
    state = 0
    entry_idx = -1
    entry_price = 0.0
    trades: list[dict[str, object]] = []
    unresolved_positions = 0
    previous_row = -1
    thresholds = (
        tuple(float(value) for value in curve_thresholds)
        if curve_thresholds is not None
        else (float(close_threshold), float(open_threshold), float(close_threshold), float(open_threshold))
    )
    for idx, scores in zip(centers, predictions, strict=True):
        row = int(idx)
        discontinuity = previous_row >= 0 and (
            frames.segment_id[row] != frames.segment_id[previous_row]
            or frames.ts_ns[row] - frames.ts_ns[previous_row] > float(reset_gap_seconds) * 1e9
        )
        if discontinuity:
            unresolved_positions += int(state != 0)
            state, entry_idx, entry_price = 0, -1, 0.0
        previous_row = row
        if not valid[row]:
            continue
        long_backward, long_forward, short_backward, short_forward = (float(x) for x in scores)
        if state == 0:
            long_excess = (
                (long_forward - thresholds[1]) / max(1.0 - thresholds[1], 1e-9)
                if long_backward <= thresholds[0]
                else float("-inf")
            )
            short_excess = (
                (short_forward - thresholds[3]) / max(1.0 - thresholds[3], 1e-9)
                if short_backward <= thresholds[2]
                else float("-inf")
            )
            if max(long_excess, short_excess) <= 0.0:
                continue
            if long_excess >= short_excess:
                state, entry_idx, entry_price = 1, row, float(ask[row])
            else:
                state, entry_idx, entry_price = -1, row, float(bid[row])
            continue
        close_score = long_backward if state == 1 else short_backward
        close_boundary = thresholds[0] if state == 1 else thresholds[2]
        if close_score <= close_boundary:
            continue
        exit_price = float(bid[row] if state == 1 else ask[row])
        gross_bps = 1e4 * (exit_price / entry_price - 1.0) if state == 1 else 1e4 * (entry_price / exit_price - 1.0)
        net_bps = gross_bps - float(fee_round_trip_bps)
        trades.append(
            {
                "side": "long" if state == 1 else "short",
                "entry_ts_ns": int(frames.ts_ns[entry_idx]),
                "exit_ts_ns": int(frames.ts_ns[row]),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "holding_seconds": float(frames.ts_ns[row] - frames.ts_ns[entry_idx]) / 1e9,
                "gross_bps": gross_bps,
                "net_bps": net_bps,
                "net_usd": float(notional_usd) * net_bps / 1e4,
            }
        )
        state, entry_idx, entry_price = 0, -1, 0.0
    unresolved_positions += int(state != 0)
    net = np.asarray([float(row["net_bps"]) for row in trades], dtype=np.float64)
    return (
        {
            "trades": len(trades),
            "net_bps": float(net.sum()) if net.size else 0.0,
            "net_usd": float(notional_usd) * float(net.sum()) / 1e4 if net.size else 0.0,
            "mean_trade_bps": float(net.mean()) if net.size else 0.0,
            "median_trade_bps": float(np.median(net)) if net.size else 0.0,
            "win_rate": float(np.mean(net > 0.0)) if net.size else 0.0,
            "unresolved_position": bool(unresolved_positions),
            "unresolved_positions": unresolved_positions,
        },
        trades,
    )


def _percentile_scores(predictions: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float32)
    reference = np.asarray(sorted_reference, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("predictions must have shape [N, 4]")
    if reference.ndim != 2 or reference.shape[1] != 4 or len(reference) == 0:
        raise ValueError("sorted reference must have shape [N, 4]")
    return np.column_stack(
        [
            np.searchsorted(reference[:, column], values[:, column], side="right")
            / float(len(reference))
            for column in range(4)
        ]
    ).astype(np.float32)


def select_economic_epoch(args: object) -> dict[str, object]:
    run_dir = Path(args.run_dir)
    frames = load_frames(Path(args.frames))
    execution = _load_execution_quotes(Path(args.execution))[float(args.latency_ms)]
    open_thresholds = tuple(float(value) for value in str(args.open_thresholds).split(",") if value.strip())
    close_thresholds = tuple(float(value) for value in str(args.close_thresholds).split(",") if value.strip())
    if not open_thresholds or not close_thresholds:
        raise ValueError("threshold grids cannot be empty")
    score_space = str(getattr(args, "score_space", "raw"))
    if score_space not in {"raw", "percentile"}:
        raise ValueError("score_space must be 'raw' or 'percentile'")
    candidates: list[dict[str, object]] = []
    for validation_path in sorted((run_dir / "epoch_validation").glob("epoch_*.npz")):
        with np.load(validation_path, allow_pickle=False) as data:
            centers = np.asarray(data["centers"], dtype=np.int64)
            predictions = np.asarray(data["predictions"], dtype=np.float32)
            targets = np.asarray(data["targets"], dtype=np.float32)
        if len(centers) == 0:
            continue
        if score_space == "percentile":
            predictions = _percentile_scores(predictions, np.sort(predictions, axis=0))
        cuts = np.r_[0, np.flatnonzero(np.diff(frames.ts_ns[centers]) > 3_600e9) + 1, len(centers)]
        blocks = [slice(int(start), int(end)) for start, end in zip(cuts[:-1], cuts[1:], strict=True)]
        epoch = int(validation_path.stem.split("_")[-1])
        for open_threshold in open_thresholds:
            for close_threshold in close_thresholds:
                block_results = []
                active_blocks = []
                for block in blocks:
                    result, _ = _backtest(
                        frames,
                        centers[block],
                        predictions[block],
                        latency_ms=float(args.latency_ms),
                        notional_usd=float(args.notional_usd),
                        fee_round_trip_bps=float(args.fee_round_trip_bps),
                        execution=execution,
                        open_threshold=open_threshold,
                        close_threshold=close_threshold,
                    )
                    block_results.append(result)
                    forward_events = int(np.sum(targets[block][:, [1, 3]] >= 0.5))
                    active_blocks.append(forward_events >= int(args.min_trades_per_block))
                robust = bool(
                    block_results
                    and any(active_blocks)
                    and all(
                        (
                            int(result["trades"]) >= int(args.min_trades_per_block)
                            and float(result["net_bps"]) > 0.0
                        )
                        if active
                        else int(result["trades"]) == 0
                        for result, active in zip(block_results, active_blocks, strict=True)
                    )
                    and all(
                        int(result["unresolved_positions"])
                        <= int(getattr(args, "max_unresolved_per_block", 0))
                        for result in block_results
                    )
                )
                active_net = [
                    float(result["net_bps"])
                    for result, active in zip(block_results, active_blocks, strict=True)
                    if active
                ]
                candidates.append({
                    "epoch": epoch,
                    "open_threshold": open_threshold,
                    "close_threshold": close_threshold,
                    "robust": robust,
                    "min_block_net_bps": min(active_net) if active_net else float("-inf"),
                    "total_net_bps": sum(float(result["net_bps"]) for result in block_results),
                    "total_trades": sum(int(result["trades"]) for result in block_results),
                    "blocks": block_results,
                    "active_blocks": active_blocks,
                })
    eligible = [candidate for candidate in candidates if bool(candidate["robust"])]
    selected = max(
        eligible,
        key=lambda candidate: (
            float(candidate["min_block_net_bps"]),
            float(candidate["total_net_bps"]),
            int(candidate["total_trades"]),
        ),
        default=None,
    )
    output = {
        "selection_rule": "active regimes require positive net and minimum trades; zero-forward-event regimes require zero trades; enforce the configured unresolved-position limit; maximize worst active-regime net",
        "score_space": score_space,
        "fee_round_trip_bps": float(args.fee_round_trip_bps),
        "latency_ms": float(args.latency_ms),
        "min_trades_per_block": int(args.min_trades_per_block),
        "max_unresolved_per_block": int(getattr(args, "max_unresolved_per_block", 0)),
        "selected": selected,
        "eligible_candidates": len(eligible),
        "evaluated_candidates": len(candidates),
        "candidates": candidates,
    }
    write_json(run_dir / "economic_epoch_selection.json", output)
    if selected is not None:
        epoch = int(selected["epoch"])
        shutil.copy2(
            run_dir / "epoch_checkpoints" / f"epoch_{epoch:02d}.pt",
            run_dir / "economically_selected_four_curve.pt",
        )
        open_threshold = float(selected["open_threshold"])
        close_threshold = float(selected["close_threshold"])
        curve_thresholds = (close_threshold, open_threshold, close_threshold, open_threshold)
        policy = {
            "checkpoint": "economically_selected_four_curve.pt",
            "score_space": score_space,
            "open_threshold": open_threshold,
            "close_threshold": close_threshold,
            "curve_thresholds": list(curve_thresholds),
        }
        write_json(run_dir / "economic_policy.json", policy)
        calibration_path = run_dir / "score_calibration.npz"
        if score_space == "percentile":
            with np.load(
                run_dir / "epoch_validation" / f"epoch_{epoch:02d}.npz",
                allow_pickle=False,
            ) as data:
                reference = np.sort(
                    np.asarray(data["predictions"], dtype=np.float32), axis=0
                )
            np.savez_compressed(calibration_path, sorted_reference=reference)
        else:
            calibration_path.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in output.items() if key != "candidates"}), flush=True)
    return output


def run_four_curve_pipeline(args: object) -> dict[str, object]:
    started = perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    external_frames = Path(str(getattr(args, "frames_cache", ""))) if str(getattr(args, "frames_cache", "")) else None
    external_execution = Path(str(getattr(args, "execution_cache", ""))) if str(getattr(args, "execution_cache", "")) else None
    raw_dir = Path(args.raw_dir)
    snapshot_dir = raw_dir
    copy_raw_snapshot = bool(getattr(args, "copy_raw_snapshot", False))
    if copy_raw_snapshot and bool(getattr(args, "immutable_raw", False)):
        raise ValueError("--copy-raw-snapshot and --immutable-raw are mutually exclusive")
    if copy_raw_snapshot and not (
        external_frames is not None and external_execution is not None
    ):
        snapshot_dir = out_dir / "raw_snapshot"
        print(json.dumps({"stage": "snapshot"}), flush=True)
        _immutable_snapshot(raw_dir, snapshot_dir)
    else:
        print(
            json.dumps(
                {
                    "stage": "raw_source",
                    "mode": "read_in_place",
                    "path": str(raw_dir.resolve()),
                }
            ),
            flush=True,
        )
    catalog = DatasetCatalog.discover(snapshot_dir)
    write_json(out_dir / "snapshot_manifest.json", catalog.manifest())

    frames_path = out_dir / "frames.npz"
    if external_frames is not None:
        frames = load_frames(external_frames)
    elif frames_path.exists():
        frames = load_frames(frames_path)
    else:
        end = catalog.common_end_ns
        if float(args.duration_seconds) > 0.0:
            end = min(end, catalog.common_start_ns + int(float(args.duration_seconds) * 1e9))
        print(json.dumps({"stage": "frames", "start": catalog.common_start_ns, "end": end}), flush=True)
        last_percent = [-1]

        def progress(done: int, total: int) -> None:
            percent = int(100 * done / max(total, 1))
            if percent != last_percent[0] and percent % 5 == 0:
                print(json.dumps({"stage": "frames", "done": done, "total": total, "percent": percent}), flush=True)
                last_percent[0] = percent

        frames = CausalReplayBuilder(
            catalog,
            DataConfig(raw_dir=snapshot_dir, cadence_ms=int(args.cadence_ms), max_stale_ms=int(args.max_stale_ms)),
        ).build(start_ts_ns=catalog.common_start_ns, end_ts_ns=end, progress=progress)
        frames.save(frames_path, metadata={"catalog": catalog.manifest(), "cadence_ms": int(args.cadence_ms)})

    split_builder = purged_blocked_splits if str(getattr(args, "split_strategy", "blocked")) == "blocked" else purged_chronological_splits
    split_options = (
        {"holdout_fraction": float(getattr(args, "holdout_fraction", 0.20))}
        if split_builder is purged_blocked_splits
        else {}
    )
    preliminary = split_builder(frames.ts_ns, frames.valid, purge_seconds=float(args.purge_seconds), **split_options)
    execution_path = out_dir / "execution_quotes.npz"
    if external_execution is not None:
        execution_scenarios = _load_execution_quotes(external_execution)
        if not execution_path.exists():
            os.link(external_execution, execution_path)
    elif execution_path.exists():
        execution_scenarios = _load_execution_quotes(execution_path)
    else:
        print(json.dumps({"stage": "execution_quotes", "latencies_ms": [100, 250, 500]}), flush=True)
        execution_scenarios = build_execution_quote_scenarios(
            catalog,
            frames.ts_ns,
            latencies_ms=(100.0, 250.0, 500.0),
            notional_usd=float(args.notional_usd),
            max_stale_ms=float(args.max_stale_ms),
        )
        _save_execution_quotes(execution_path, execution_scenarios)
    primary_execution = execution_scenarios[250.0]
    horizons = tuple(float(value) for value in str(args.horizons).split(",") if value.strip())
    target_threshold_mode = str(getattr(args, "target_threshold_mode", "fit_quantile"))
    if target_threshold_mode not in {"fit_quantile", "fixed_edge"}:
        raise ValueError(f"unsupported target threshold mode: {target_threshold_mode}")
    targets_path = out_dir / "four_curve_targets.npz"
    if targets_path.exists():
        targets = _load_targets(targets_path)
    else:
        print(json.dumps({"stage": "targets", "horizons": horizons, "focus": float(args.focus_seconds)}), flush=True)
        targets = build_four_curve_targets(
            frames,
            horizons_seconds=horizons,
            focus_seconds=float(args.focus_seconds),
            fit_mask=preliminary.train,
            fee_round_trip_bps=float(args.fee_round_trip_bps),
            latency_ms=float(args.latency_ms),
            notional_usd=float(args.notional_usd),
            event_quantile=float(args.event_quantile),
            peak_floor=float(args.peak_floor),
            execution_bid=primary_execution.bid,
            execution_ask=primary_execution.ask,
            execution_valid=primary_execution.valid,
            minimum_edge_bps=(
                float(getattr(args, "minimum_edge_bps", 0.5))
                if target_threshold_mode == "fixed_edge"
                else None
            ),
            forward_minimum_edge_bps=(
                float(getattr(args, "forward_minimum_edge_bps", 6.0))
                if target_threshold_mode == "fixed_edge"
                else None
            ),
            full_quality_edge_bps=(
                float(getattr(args, "full_quality_edge_bps", 20.0))
                if target_threshold_mode == "fixed_edge"
                else None
            ),
            forward_curve_mode=str(getattr(args, "forward_curve_mode", "peak")),
        )
        _save_targets(targets_path, targets)

    all_valid = frames.valid & np.all(targets.valid, axis=1)
    splits = split_builder(frames.ts_ns, all_valid, purge_seconds=float(args.purge_seconds), **split_options)
    np.savez_compressed(out_dir / "splits.npz", train=splits.train, valid=splits.valid, holdout=splits.holdout)
    backward_score_x = causal_backward_score_features(
        frames,
        horizons,
        cost_bps=float(args.fee_round_trip_bps),
    )
    high_order_x, high_order_names = causal_high_order_features(frames)
    causal_score_x = np.concatenate((backward_score_x, high_order_x), axis=1)
    edge_target, edge_valid = multihorizon_forward_edge_targets(
        frames,
        horizons,
        cost_bps=float(args.fee_round_trip_bps),
    )
    dense_score_target, dense_score_valid = _dense_score_auxiliary_targets(targets)
    auxiliary_target = np.concatenate((edge_target, dense_score_target), axis=1)
    auxiliary_valid = np.concatenate((edge_valid, dense_score_valid), axis=1)
    base_normalizer, venue_normalizer = _normalizers(frames, splits.train, causal_score_x)
    write_json(
        out_dir / "normalizers.json",
        {
            "input_semantics": "stationary_relative_bps_v1",
            "high_order_features": list(high_order_names),
            "base": base_normalizer.to_dict(),
            "venue": venue_normalizer.to_dict(),
        },
    )
    context = int(args.context_ticks)
    train_centers = curve_centers(frames, targets, splits.train, context_ticks=context, background_stride=int(args.background_stride))
    valid_centers = curve_centers(frames, targets, splits.valid, context_ticks=context, background_stride=1)
    holdout_centers = curve_centers(frames, targets, splits.holdout, context_ticks=context, background_stride=1)
    common = dict(
        frames=frames,
        targets=targets,
        context_ticks=context,
        base_normalizer=base_normalizer,
        venue_normalizer=venue_normalizer,
        causal_score_x=causal_score_x,
        auxiliary_target=auxiliary_target,
        auxiliary_valid=auxiliary_valid,
    )
    train_dataset = CurveWindowDataset(
        centers=train_centers,
        supervision_ticks=int(getattr(args, "supervision_ticks", 1)),
        **common,
    )
    valid_dataset = CurveWindowDataset(centers=valid_centers, **common)
    holdout_dataset = CurveWindowDataset(centers=holdout_centers, **common)
    model_config = CurveModelConfig(
        input_dim=train_dataset.input_dim,
        venue_feature_dim=train_dataset.venue_feature_dim,
        d_model=int(args.hidden_size),
        nhead=int(args.heads),
        num_layers=int(args.layers),
        dim_feedforward=4 * int(args.hidden_size),
        dropout=float(args.dropout),
        num_venues=len(frames.venues),
        use_venue_embeddings=True,
        num_aux_horizons=len(horizons),
        auxiliary_output_dim=auxiliary_target.shape[1],
        separate_task_towers=bool(getattr(args, "separate_task_towers", False)),
    )
    print(json.dumps({"stage": "train", "train": len(train_dataset), "valid": len(valid_dataset), "model": model_config.to_dict()}), flush=True)
    trained = train_curve_model(
        train_dataset,
        valid_dataset,
        model_config=model_config,
        out_dir=out_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        device=str(args.device) or None,
        forward_peak_weight_cap=float(args.forward_peak_weight_cap),
        backward_peak_weight_cap=float(args.backward_peak_weight_cap),
        initial_checkpoint=Path(args.initial_checkpoint) if str(args.initial_checkpoint) else None,
    )
    valid_idx, valid_predictions = predict_curve_model(trained.model, valid_dataset, batch_size=int(args.batch_size), device=str(args.device) or None)
    holdout_idx, holdout_predictions = predict_curve_model(trained.model, holdout_dataset, batch_size=int(args.batch_size), device=str(args.device) or None)
    prediction_table = {"ts_ns": frames.ts_ns[holdout_idx], **{name: holdout_predictions[:, i] for i, name in enumerate(CURVE_NAMES)}, **{f"target_{name}": targets.values[holdout_idx, i] for i, name in enumerate(CURVE_NAMES)}}
    pq.write_table(pa.table(prediction_table), out_dir / "holdout_predictions.parquet", compression="zstd")
    validation_backtest, _ = _backtest(
        frames,
        valid_idx,
        valid_predictions,
        latency_ms=float(args.latency_ms),
        notional_usd=float(args.notional_usd),
        fee_round_trip_bps=float(args.fee_round_trip_bps),
        execution=primary_execution,
    )
    backtest, trades = _backtest(
        frames,
        holdout_idx,
        holdout_predictions,
        latency_ms=float(args.latency_ms),
        notional_usd=float(args.notional_usd),
        fee_round_trip_bps=float(args.fee_round_trip_bps),
        execution=primary_execution,
    )
    if trades:
        pq.write_table(pa.Table.from_pylist(trades), out_dir / "holdout_trades.parquet", compression="zstd")
    summary = {
        "curve_names": list(CURVE_NAMES),
        "ticks": len(frames.ts_ns),
        "duration_hours": float(frames.ts_ns[-1] - frames.ts_ns[0]) / 3.6e12,
        "target_threshold_mode": target_threshold_mode,
        "target_high_thresholds_bps": targets.high_thresholds.tolist(),
        "target_full_quality_thresholds_bps": targets.full_quality_thresholds.tolist(),
        "best_epoch": trained.best_epoch,
        "best_valid_loss": trained.best_valid_loss,
        "peak_positive_weights": list(trained.peak_positive_weights),
        "validation": _prediction_metrics(valid_predictions, targets.values[valid_idx]),
        "validation_backtest": validation_backtest,
        "holdout": _prediction_metrics(holdout_predictions, targets.values[holdout_idx]),
        "backtest": backtest,
        "execution_ready": bool(backtest["trades"] >= 20 and backtest["net_bps"] > 0.0 and not backtest["unresolved_position"]),
        "elapsed_seconds": perf_counter() - started,
    }
    write_json(out_dir / "four_curve_summary.json", summary)
    write_json(out_dir / "four_curve_run.json", {"args": {key: value for key, value in vars(args).items() if key != "func"}, "model": model_config.to_dict(), "summary": summary})
    print(json.dumps({"stage": "complete", **summary}), flush=True)
    return summary


def replay_four_curve_run(args: object) -> dict[str, object]:
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "four_curve_run.json").read_text(encoding="utf-8"))
    run_args = manifest["args"]
    frames_path = Path(args.frames_cache) if str(args.frames_cache) else run_dir / "frames.npz"
    frames = load_frames(frames_path)
    targets = _load_targets(run_dir / "four_curve_targets.npz")
    with np.load(run_dir / "splits.npz", allow_pickle=False) as split_data:
        split_mask = split_data[str(args.split)]
    normalizers = json.loads((run_dir / "normalizers.json").read_text(encoding="utf-8"))
    base_normalizer = RobustNormalizer.from_dict(normalizers["base"])
    venue_normalizer = RobustNormalizer.from_dict(normalizers["venue"])
    horizons = tuple(float(value) for value in str(run_args["horizons"]).split(",") if value.strip())
    causal_score_x = causal_score_features_for_normalizer(
        frames,
        base_normalizer,
        horizons,
        cost_bps=float(run_args["fee_round_trip_bps"]),
    )
    context = int(run_args["context_ticks"])
    centers = causal_centers(frames, split_mask, context_ticks=context)
    dataset = CurveInferenceDataset(
        frames,
        centers,
        context_ticks=context,
        base_normalizer=base_normalizer,
        venue_normalizer=venue_normalizer,
        causal_score_x=causal_score_x,
    )
    checkpoint_name = str(getattr(args, "checkpoint", "")) or "best_four_curve.pt"
    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FourCurveCausalTransformer(CurveModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    center_idx, predictions = predict_curve_model(
        model,
        dataset,
        batch_size=int(args.batch_size),
        device=str(args.device) or None,
    )
    table = {
        "ts_ns": frames.ts_ns[center_idx],
        **{name: predictions[:, i] for i, name in enumerate(CURVE_NAMES)},
        **{f"target_{name}": targets.values[center_idx, i] for i, name in enumerate(CURVE_NAMES)},
    }
    output_prefix = str(getattr(args, "output_prefix", "")) or f"{args.split}_replay"
    pq.write_table(pa.table(table), run_dir / f"{output_prefix}_predictions.parquet", compression="zstd")
    stress: dict[str, object] = {}
    primary_trades: list[dict[str, object]] = []
    execution_path = run_dir / "execution_quotes.npz"
    execution_scenarios = _load_execution_quotes(execution_path) if execution_path.exists() else {}
    for fee in (4.0, 6.0, 10.0, 14.0):
        for latency in (100.0, 250.0, 500.0):
            result, trades = _backtest(
                frames,
                center_idx,
                predictions,
                latency_ms=latency,
                notional_usd=float(run_args["notional_usd"]),
                fee_round_trip_bps=fee,
                execution=execution_scenarios.get(latency),
                open_threshold=float(getattr(args, "open_threshold", 0.5)),
                close_threshold=float(getattr(args, "close_threshold", 0.5)),
            )
            key = f"fee_{fee:g}bps_latency_{latency:g}ms"
            stress[key] = result
            if fee == float(run_args["fee_round_trip_bps"]) and latency == float(run_args["latency_ms"]):
                primary_trades = trades
    if primary_trades:
        pq.write_table(pa.Table.from_pylist(primary_trades), run_dir / f"{output_prefix}_trades.parquet", compression="zstd")
    output = {
        "split": str(args.split),
        "rows": len(center_idx),
        "checkpoint": str(checkpoint_path.resolve()),
        "open_threshold": float(getattr(args, "open_threshold", 0.5)),
        "close_threshold": float(getattr(args, "close_threshold", 0.5)),
        "prediction_metrics": _prediction_metrics(predictions, targets.values[center_idx]),
        "stress": stress,
    }
    write_json(run_dir / f"{output_prefix}_summary.json", output)
    print(json.dumps(output), flush=True)
    return output


def ensemble_four_curve_runs(args: object) -> dict[str, object]:
    run_dirs = [Path(value.strip()) for value in str(args.run_dirs).split(";") if value.strip()]
    if len(run_dirs) < 2:
        raise ValueError("--run-dirs requires at least two semicolon-separated runs")
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base_run = run_dirs[0]
    base_manifest = json.loads((base_run / "four_curve_run.json").read_text(encoding="utf-8"))
    run_args = base_manifest["args"]
    frames = load_frames(Path(args.frames_cache))
    targets = _load_targets(base_run / "four_curve_targets.npz")
    with np.load(base_run / "splits.npz", allow_pickle=False) as split_data:
        split_mask = split_data[str(args.split)]
    normalizers = json.loads((base_run / "normalizers.json").read_text(encoding="utf-8"))
    base_normalizer = RobustNormalizer.from_dict(normalizers["base"])
    venue_normalizer = RobustNormalizer.from_dict(normalizers["venue"])
    horizons = tuple(float(value) for value in str(run_args["horizons"]).split(",") if value.strip())
    causal_score_x = causal_score_features_for_normalizer(
        frames,
        base_normalizer,
        horizons,
        cost_bps=float(run_args["fee_round_trip_bps"]),
    )
    context = int(run_args["context_ticks"])
    centers = curve_centers(frames, targets, split_mask, context_ticks=context, background_stride=1)
    dataset = CurveInferenceDataset(
        frames,
        centers,
        context_ticks=context,
        base_normalizer=base_normalizer,
        venue_normalizer=venue_normalizer,
        causal_score_x=causal_score_x,
    )
    members: list[np.ndarray] = []
    model_dir = destination / "models"
    model_dir.mkdir(exist_ok=True)
    for index, run_dir in enumerate(run_dirs):
        checkpoint_path = run_dir / "best_four_curve.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = FourCurveCausalTransformer(CurveModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        member_centers, prediction = predict_curve_model(
            model,
            dataset,
            batch_size=int(args.batch_size),
            device=str(args.device) or None,
        )
        if not np.array_equal(member_centers, centers):
            raise RuntimeError("ensemble member predictions are not timestamp-aligned")
        members.append(prediction)
        target = model_dir / f"member_{index:02d}.pt"
        if not target.exists():
            os.link(checkpoint_path, target)
    raw_predictions = np.mean(np.stack(members, axis=0), axis=0, dtype=np.float64).astype(np.float32)
    calibration_path = destination / "calibration.json"
    if str(args.split) == "valid":
        boundary = len(centers) // 2
        calibration_rows = slice(0, boundary)
        evaluation_rows = np.arange(boundary, len(centers), dtype=np.int64)
        scales = []
        for column in range(4):
            predicted_q = float(np.quantile(raw_predictions[calibration_rows, column], 0.995))
            target_q = float(np.quantile(targets.values[centers[calibration_rows], column], 0.995))
            scales.append(float(np.clip(target_q / max(predicted_q, 1e-6), 0.5, 3.0)))
        write_json(calibration_path, {"method": "validation_first_half_q995_scale", "scales": scales, "calibration_rows": boundary})
    else:
        if not calibration_path.exists():
            raise FileNotFoundError("run the valid ensemble replay before holdout to fit calibration")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        scales = [float(value) for value in calibration["scales"]]
        evaluation_rows = np.arange(len(centers), dtype=np.int64)
    predictions = np.clip(raw_predictions * np.asarray(scales, dtype=np.float32)[None, :], 0.0, 1.0)
    evaluation_centers = centers[evaluation_rows]
    evaluation_predictions = predictions[evaluation_rows]
    table = {
        "ts_ns": frames.ts_ns[centers],
        **{name: predictions[:, i] for i, name in enumerate(CURVE_NAMES)},
        **{f"target_{name}": targets.values[centers, i] for i, name in enumerate(CURVE_NAMES)},
    }
    pq.write_table(pa.table(table), destination / f"{args.split}_ensemble_predictions.parquet", compression="zstd")
    execution_scenarios = _load_execution_quotes(base_run / "execution_quotes.npz")
    stress: dict[str, object] = {}
    primary_trades: list[dict[str, object]] = []
    for fee in (4.0, 6.0, 10.0, 14.0):
        for latency in (100.0, 250.0, 500.0):
            result, trades = _backtest(
                frames,
                evaluation_centers,
                evaluation_predictions,
                latency_ms=latency,
                notional_usd=float(run_args["notional_usd"]),
                fee_round_trip_bps=fee,
                execution=execution_scenarios[latency],
            )
            stress[f"fee_{fee:g}bps_latency_{latency:g}ms"] = result
            if fee == 10.0 and latency == 250.0:
                primary_trades = trades
    if primary_trades:
        pq.write_table(pa.Table.from_pylist(primary_trades), destination / f"{args.split}_ensemble_trades.parquet", compression="zstd")
    output = {
        "split": str(args.split),
        "members": [str(path.resolve()) for path in run_dirs],
        "rows": len(evaluation_centers),
        "calibration_scales": scales,
        "raw_prediction_metrics": _prediction_metrics(raw_predictions[evaluation_rows], targets.values[evaluation_centers]),
        "prediction_metrics": _prediction_metrics(evaluation_predictions, targets.values[evaluation_centers]),
        "stress": stress,
    }
    write_json(destination / f"{args.split}_ensemble_summary.json", output)
    write_json(
        destination / "ensemble_manifest.json",
        {
            "curve_names": list(CURVE_NAMES),
            "aggregation": "arithmetic_mean",
            "calibration": {"method": "per_curve_multiplicative_clip", "scales": scales},
            "model_files": [f"models/member_{index:02d}.pt" for index in range(len(run_dirs))],
            "context_ticks": context,
            "cadence_ms": int(run_args["cadence_ms"]),
            "normalizers": normalizers,
            "execution": {"market": "binance_um_futures_BTCUSDT", "notional_usd": float(run_args["notional_usd"]), "latency_ms": 250.0, "fee_round_trip_bps": 10.0},
        },
    )
    print(json.dumps(output), flush=True)
    return output
