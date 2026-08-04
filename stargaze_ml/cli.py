from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
from datetime import datetime, timezone

import numpy as np
import torch

from .artifacts import load_frames, write_json
from .config import DataConfig, EXTENDED_HORIZONS_SECONDS
from .data import CausalReplayBuilder, DatasetCatalog
from .labels import build_labels
from .models import HierarchicalCausalTransformerPolicy, PolicyConfig
from .replay import replay_policy
from .reports import policy_summary, write_policy_artifacts
from .scores import build_score_bundle
from .training import PolicyWindowDataset, RobustNormalizer, build_examples, purged_blocked_splits, purged_chronological_splits, train_policy
from .training.data import POSITION_FEATURE_NAMES
from .curve_pipeline import ensemble_four_curve_runs, replay_four_curve_run, run_four_curve_pipeline, select_economic_epoch
from .deployment import export_four_curve_bundle
from .incremental_pipeline import prepare_incremental_data
from .historical import prepare_binance_history
from .oos_pipeline import evaluate_four_curve_oos
from .gold.config import CTraderCredentials, GoldExperimentConfig
from .gold.ctrader import CTraderMinuteDownloader, refresh_ctrader_credentials
from .gold.pipeline import run_gold_experiments


def _floats(spec: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in str(spec).split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _ints(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(spec).split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one integer value is required")
    return values


def _utc_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.lower() == "now":
        return datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if len(text) == 10:
        text += "T00:00:00+00:00"
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _download_gold_m1(args: argparse.Namespace) -> None:
    credentials = CTraderCredentials.from_json(Path(args.secrets))
    downloader = CTraderMinuteDownloader(
        credentials,
        symbol=str(args.symbol),
        chunk_days=int(args.chunk_days),
        progress=lambda payload: print(json.dumps(payload), flush=True),
    )
    result = downloader.download(
        start=_utc_datetime(args.start),
        end=_utc_datetime(args.end),
        output_path=Path(args.out),
    )
    print(json.dumps({"stage": "complete", **result}), flush=True)


def _refresh_gold_token(args: argparse.Namespace) -> None:
    result = refresh_ctrader_credentials(Path(args.secrets))
    print(json.dumps(result), flush=True)


def _run_gold_minute(args: argparse.Namespace) -> None:
    horizons = _ints(args.horizons)
    config = GoldExperimentConfig(
        context_minutes=int(args.context_minutes),
        horizons_minutes=horizons,
        purge_minutes=int(args.purge_minutes),
        train_fraction=float(args.train_fraction),
        valid_fraction=float(args.valid_fraction),
        sample_stride=int(args.sample_stride),
        evaluation_stride=int(args.evaluation_stride),
        hidden_size=int(args.hidden_size),
        tcn_layers=int(args.layers),
        kernel_size=int(args.kernel_size),
        dropout=float(args.dropout),
        embedding_size=int(args.embedding_size),
        batch_size=int(args.batch_size),
        max_epochs=int(args.epochs),
        early_stopping_patience=int(args.patience),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        retrieval_k=int(args.retrieval_k),
        seed=int(args.seed),
    )
    summary = run_gold_experiments(
        candles_path=Path(args.candles),
        out_dir=Path(args.out_dir),
        config=config,
        device=str(args.device),
    )
    print(json.dumps(summary), flush=True)


def _prepare_binance_history(args: argparse.Namespace) -> None:
    result = prepare_binance_history(
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        out_dir=args.out_dir,
        keep_archives=bool(args.keep_archives),
    )
    print(json.dumps({"stage": "complete", **result}), flush=True)


def _slice_frames(frames: object, start: int, end: int):
    from .contracts import CausalFrames

    return CausalFrames(
        ts_ns=frames.ts_ns[start:end],
        x=frames.x[start:end],
        venue_x=frames.venue_x[start:end],
        bid=frames.bid[start:end],
        ask=frames.ask[start:end],
        valid=frames.valid[start:end],
        segment_id=frames.segment_id[start:end],
        feature_names=frames.feature_names,
        venue_feature_names=frames.venue_feature_names,
        venues=frames.venues,
    )


def _catalog(args: argparse.Namespace) -> None:
    catalog = DatasetCatalog.discover(Path(args.raw_dir))
    print(json.dumps(catalog.manifest(), indent=2), flush=True)


def _run_offline(args: argparse.Namespace) -> None:
    if str(getattr(args, "horizon_packs", "")).strip():
        _run_horizon_sweep(args)
        return
    started = perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = _floats(args.horizons)
    data_config = DataConfig(
        raw_dir=Path(args.raw_dir),
        cadence_ms=int(args.cadence_ms),
        max_stale_ms=int(args.max_stale_ms),
    )
    catalog = DatasetCatalog.discover(data_config.raw_dir)
    start = catalog.common_start_ns + int(float(args.start_offset_seconds) * 1e9)
    end = None if float(args.duration_seconds) <= 0.0 else start + int(float(args.duration_seconds) * 1e9)
    print(json.dumps({"stage": "frames", "start_ts_ns": start, "end_ts_ns": end}), flush=True)
    last_progress = [-1]

    def progress(done: int, total: int) -> None:
        pct = int(100 * done / max(total, 1))
        if pct != last_progress[0] and (pct % 10 == 0 or done == total):
            print(json.dumps({"stage": "frames", "done": done, "total": total, "percent": pct}), flush=True)
            last_progress[0] = pct

    frames = CausalReplayBuilder(catalog, data_config).build(start_ts_ns=start, end_ts_ns=end, progress=progress)
    frames.save(out_dir / "frames.npz", metadata={"catalog": catalog.manifest(), "data_config": asdict(data_config) | {"raw_dir": str(data_config.raw_dir)}})
    quote_valid = (frames.venue_x[:, :, 0] > 0.5) & (frames.venue_x[:, :, 1] <= float(data_config.max_stale_ms))
    print(json.dumps({"stage": "scores", "ticks": len(frames.ts_ns), "horizons": horizons}), flush=True)
    bundle = build_score_bundle(
        frames.ts_ns,
        frames.bid,
        frames.ask,
        horizons_seconds=horizons,
        cost_bps=float(args.cost_bps),
        valid=quote_valid,
        segment_id=frames.segment_id,
        venue_names=frames.venues,
        require_all_venues=True,
        require_all_horizons=False,
    )
    np.savez_compressed(
        out_dir / "scores.npz",
        horizons_seconds=np.asarray(horizons),
        forward_long=bundle.consensus.forward_long,
        forward_short=bundle.consensus.forward_short,
        backward_long=bundle.consensus.backward_long,
        backward_short=bundle.consensus.backward_short,
        forward_valid=bundle.consensus.forward_valid,
        backward_valid=bundle.consensus.backward_valid,
        venue_names=np.asarray(bundle.cube.venue_names),
        venue_forward_long=bundle.cube.forward_long.astype(np.float32),
        venue_forward_short=bundle.cube.forward_short.astype(np.float32),
        venue_backward_long=bundle.cube.backward_long.astype(np.float32),
        venue_backward_short=bundle.cube.backward_short.astype(np.float32),
        venue_forward_valid=bundle.cube.forward_valid,
        venue_backward_valid=bundle.cube.backward_valid,
    )
    score_valid = np.any(bundle.consensus.forward_valid, axis=1) & np.any(bundle.consensus.backward_valid, axis=1)
    frames.valid &= score_valid
    if args.split_strategy == "blocked":
        splits = purged_blocked_splits(frames.ts_ns, frames.valid, purge_seconds=float(args.purge_seconds))
    else:
        splits = purged_chronological_splits(frames.ts_ns, frames.valid, purge_seconds=float(args.purge_seconds))
    np.savez_compressed(
        out_dir / "splits.npz",
        train=splits.train,
        valid=splits.valid,
        holdout=splits.holdout,
        train_end_ns=np.asarray(splits.train_end_ns, dtype=np.int64),
        valid_end_ns=np.asarray(splits.valid_end_ns, dtype=np.int64),
        purge_ns=np.asarray(splits.purge_ns, dtype=np.int64),
    )
    print(json.dumps({"stage": "labels", "train_ticks": int(splits.train.sum()), "valid_ticks": int(splits.valid.sum()), "holdout_ticks": int(splits.holdout.sum())}), flush=True)
    labels = build_labels(
        frames,
        forward_long_h=bundle.consensus.forward_long,
        forward_short_h=bundle.consensus.forward_short,
        backward_long_h=bundle.consensus.backward_long,
        backward_short_h=bundle.consensus.backward_short,
        horizons_seconds=horizons,
        fit_mask=splits.train,
        event_high_quantile=float(args.event_high_quantile),
    )
    np.savez_compressed(
        out_dir / "labels.npz",
        flat_action=labels.flat_action,
        open_long_zone=labels.open_long_zone,
        open_short_zone=labels.open_short_zone,
        close_long_zone=labels.close_long_zone,
        close_short_zone=labels.close_short_zone,
        dominant_long_horizon=labels.dominant_long_horizon,
        dominant_short_horizon=labels.dominant_short_horizon,
    )
    examples = build_examples(frames, labels, context_ticks=int(args.context_ticks), skip_stride=int(args.skip_stride), hold_stride=int(args.hold_stride))
    train_examples = examples.select(splits.train)
    valid_examples = examples.select(splits.valid)
    holdout_examples = examples.select(splits.holdout)
    if min(len(train_examples.center_idx), len(valid_examples.center_idx), len(holdout_examples.center_idx)) == 0:
        raise RuntimeError("label construction produced an empty example partition; use more data or lower --event-high-quantile")
    x_normalizer = RobustNormalizer.fit(frames.x, splits.train)
    venue_normalizer = RobustNormalizer.fit(frames.venue_x, splits.train)
    common_dataset = dict(
        frames=frames,
        labels=labels,
        forward_long_h=bundle.consensus.forward_long,
        forward_short_h=bundle.consensus.forward_short,
        backward_long_h=bundle.consensus.backward_long,
        backward_short_h=bundle.consensus.backward_short,
        horizons_seconds=horizons,
        context_ticks=int(args.context_ticks),
        x_normalizer=x_normalizer,
        venue_normalizer=venue_normalizer,
    )
    train_dataset = PolicyWindowDataset(examples=train_examples, **common_dataset)
    valid_dataset = PolicyWindowDataset(examples=valid_examples, **common_dataset)
    np.savez_compressed(
        out_dir / "auxiliary_targets.npz",
        horizons_seconds=np.asarray(horizons),
        future_signed_flow=train_dataset.future_flow,
        future_l3_depletion=train_dataset.future_liquidity,
    )
    model_config = PolicyConfig(
        input_dim=train_dataset.input_dim,
        venue_feature_dim=train_dataset.venue_feature_dim,
        d_model=int(args.hidden_size),
        nhead=int(args.heads),
        num_layers=int(args.layers),
        dim_feedforward=int(args.hidden_size) * 4,
        dropout=float(args.dropout),
        num_horizons=len(horizons),
    )
    print(json.dumps({"stage": "train", "train_examples": len(train_dataset), "valid_examples": len(valid_dataset), "model": asdict(model_config)}), flush=True)
    trained = train_policy(
        train_dataset,
        valid_dataset,
        model_config=model_config,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
        out_dir=out_dir,
        device=args.device or None,
    )
    write_json(out_dir / "normalizers.json", {"x": x_normalizer.to_dict(), "venue": venue_normalizer.to_dict()})
    write_json(
        out_dir / "feature_schema.json",
        {
            "base_features": list(frames.feature_names),
            "venue_features": list(frames.venue_feature_names),
            "position_features": list(POSITION_FEATURE_NAMES),
            "backward_score_features": [
                *(f"backward_long_{h:g}s" for h in horizons),
                *(f"backward_short_{h:g}s" for h in horizons),
                *(f"backward_valid_{h:g}s" for h in horizons),
            ],
            "venues": list(frames.venues),
            "horizons_seconds": list(horizons),
            "action_selection": "state-conditioned model argmax",
            "auxiliary_targets": ["future_signed_flow_all_venues", "future_kraken_l3_depletion"],
        },
    )
    holdout_idx = np.flatnonzero(splits.holdout)
    replay_start = max(0, int(holdout_idx[0]) - int(args.context_ticks) + 1)
    replay_end = int(holdout_idx[-1]) + 1
    holdout_frames = _slice_frames(frames, replay_start, replay_end)
    print(json.dumps({"stage": "causal_replay", "ticks": len(holdout_frames.ts_ns)}), flush=True)
    replay = replay_policy(
        trained.model,
        holdout_frames,
        x_normalizer=x_normalizer,
        venue_normalizer=venue_normalizer,
        backward_long_h=bundle.consensus.backward_long[replay_start:replay_end],
        backward_short_h=bundle.consensus.backward_short[replay_start:replay_end],
        context_ticks=int(args.context_ticks),
        device=args.device or None,
    )
    sliced_labels = build_labels(
        holdout_frames,
        forward_long_h=bundle.consensus.forward_long[replay_start:replay_end], forward_short_h=bundle.consensus.forward_short[replay_start:replay_end],
        backward_long_h=bundle.consensus.backward_long[replay_start:replay_end], backward_short_h=bundle.consensus.backward_short[replay_start:replay_end],
        horizons_seconds=horizons, fit_mask=np.ones(len(holdout_frames.ts_ns), dtype=bool), event_high_quantile=float(args.event_high_quantile),
    )
    summary = policy_summary(replay, sliced_labels)
    duration_seconds = float(frames.ts_ns[-1] - frames.ts_ns[0]) / 1e9 if len(frames.ts_ns) > 1 else 0.0
    status = "smoke_only" if duration_seconds < 86_400.0 or len(labels.episodes) < 100 else "research_candidate"
    best_metrics = trained.history[trained.best_epoch - 1]
    summary.update({
        "best_epoch": trained.best_epoch,
        "best_valid_loss": trained.best_valid_loss,
        "best_selection_score": trained.best_selection_score,
        "best_valid_action_loss": float(best_metrics["valid_action"]),
        "best_valid_action_accuracy": float(best_metrics["valid_action_accuracy"]),
        "best_valid_macro_action_recall": float(best_metrics["valid_macro_action_recall"]),
        "elapsed_seconds": perf_counter() - started,
        "data_duration_seconds": duration_seconds,
        "horizons_seconds": horizons,
        "cost_bps": float(args.cost_bps),
        "episodes": len(labels.episodes),
        "holdout_examples": len(holdout_examples.center_idx),
        "checkpoint_status": status,
        "execution_ready": False,
    })
    write_policy_artifacts(out_dir, replay, summary)
    serialized_args = {name: value for name, value in vars(args).items() if name != "func"}
    write_json(out_dir / "run_manifest.json", {"args": serialized_args, "catalog": catalog.manifest(), "model_config": asdict(model_config), "summary": summary})
    print(json.dumps({"stage": "complete", **summary}), flush=True)


def _run_horizon_sweep(args: argparse.Namespace) -> None:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    packs = [chunk.strip() for chunk in str(args.horizon_packs).split(";") if chunk.strip()]
    if len(packs) < 2:
        raise ValueError("--horizon-packs requires at least two semicolon-separated packs")
    rows: list[dict[str, object]] = []
    for index, pack in enumerate(packs):
        horizons = _floats(pack)
        child = copy.copy(args)
        child.horizon_packs = ""
        child.horizons = pack
        child.out_dir = str(root / f"pack_{index:02d}_hmax_{max(horizons):g}s")
        print(json.dumps({"stage": "horizon_sweep", "pack": index, "horizons": horizons}), flush=True)
        _run_offline(child)
        summary = json.loads((Path(child.out_dir) / "summary.json").read_text(encoding="utf-8"))
        rows.append({"pack": index, "horizons": list(horizons), "run_dir": child.out_dir, **summary})
    selected = min(rows, key=lambda row: float(row["best_selection_score"]))
    write_json(
        root / "horizon_sweep.json",
        {
            "selection_rule": "minimum validation action loss plus one minus macro action recall; holdout metrics are not used",
            "selected_pack": int(selected["pack"]),
            "selected_horizons": selected["horizons"],
            "runs": rows,
        },
    )
    print(json.dumps({"stage": "horizon_sweep_complete", "selected_pack": selected["pack"], "selected_horizons": selected["horizons"]}), flush=True)


def _replay_checkpoint(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    frames = load_frames(Path(args.frames) if args.frames else run_dir / "frames.npz")
    normalizers = json.loads((run_dir / "normalizers.json").read_text(encoding="utf-8"))
    x_normalizer = RobustNormalizer.from_dict(normalizers["x"])
    venue_normalizer = RobustNormalizer.from_dict(normalizers["venue"])
    checkpoint = torch.load(run_dir / "best_policy.pt", map_location="cpu", weights_only=True)
    model_config = PolicyConfig(**checkpoint["model_config"])
    model = HierarchicalCausalTransformerPolicy(model_config)
    model.load_state_dict(checkpoint["model_state"])
    with np.load(Path(args.scores) if args.scores else run_dir / "scores.npz", allow_pickle=False) as score_data:
        backward_long = score_data["backward_long"]
        backward_short = score_data["backward_short"]
    if args.split != "all":
        with np.load(run_dir / "splits.npz", allow_pickle=False) as split_data:
            indices = np.flatnonzero(split_data[args.split])
        if indices.size == 0:
            raise RuntimeError(f"saved {args.split} split is empty")
        start = max(0, int(indices[0]) - int(args.context_ticks) + 1)
        end = int(indices[-1]) + 1
        frames = _slice_frames(frames, start, end)
        backward_long = backward_long[start:end]
        backward_short = backward_short[start:end]
    replay = replay_policy(
        model,
        frames,
        x_normalizer=x_normalizer,
        venue_normalizer=venue_normalizer,
        backward_long_h=backward_long,
        backward_short_h=backward_short,
        context_ticks=int(args.context_ticks),
        device=args.device or None,
    )
    summary = policy_summary(replay)
    destination = Path(args.out_dir) if args.out_dir else run_dir / "checkpoint_replay"
    write_policy_artifacts(destination, replay, summary)
    print(json.dumps(summary), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal multi-exchange BTC modeling pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog", help="Inspect the raw dataset catalog")
    catalog.add_argument("--raw-dir", default="source/raw datasets")
    catalog.set_defaults(func=_catalog)
    run = sub.add_parser("run-offline", help="Build data, train the model, and run a causal holdout replay")
    run.add_argument("--raw-dir", default="source/raw datasets")
    run.add_argument("--out-dir", default="runs/btc_offline_v01")
    run.add_argument("--start-offset-seconds", type=float, default=0.0)
    run.add_argument("--duration-seconds", type=float, default=0.0, help="0 uses the complete common interval")
    run.add_argument("--cadence-ms", type=int, default=100)
    run.add_argument("--max-stale-ms", type=int, default=2000)
    run.add_argument("--horizons", default=",".join(str(x) for x in EXTENDED_HORIZONS_SECONDS))
    run.add_argument("--horizon-packs", default="", help="Semicolon-separated horizon packs; trains every pack and selects only by validation loss")
    run.add_argument("--cost-bps", type=float, default=10.0)
    run.add_argument("--event-high-quantile", type=float, default=0.995)
    run.add_argument("--purge-seconds", type=float, default=1200.0)
    run.add_argument("--split-strategy", choices=["blocked", "tail"], default="blocked")
    run.add_argument("--context-ticks", type=int, default=256)
    run.add_argument("--skip-stride", type=int, default=1)
    run.add_argument("--hold-stride", type=int, default=1)
    run.add_argument("--hidden-size", type=int, default=192)
    run.add_argument("--layers", type=int, default=6)
    run.add_argument("--heads", type=int, default=6)
    run.add_argument("--dropout", type=float, default=0.12)
    run.add_argument("--epochs", type=int, default=12)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--weight-decay", type=float, default=3e-4)
    run.add_argument("--grad-clip", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=4105)
    run.add_argument("--device", default="")
    run.set_defaults(func=_run_offline)
    replay = sub.add_parser("replay-checkpoint", help="Load a saved policy and emit model-only actions")
    replay.add_argument("--run-dir", required=True)
    replay.add_argument("--frames", default="")
    replay.add_argument("--scores", default="")
    replay.add_argument("--out-dir", default="")
    replay.add_argument("--context-ticks", type=int, default=256)
    replay.add_argument("--device", default="")
    replay.add_argument("--split", choices=["all", "train", "valid", "holdout"], default="all")
    replay.set_defaults(func=_replay_checkpoint)
    curves = sub.add_parser("run-four-curve", help="Train the autonomous four-score BTC Transformer")
    curves.add_argument("--raw-dir", default=r"C:\Users\r3d_flzp\Sync\Clean Stargaze live data")
    curves.add_argument(
        "--immutable-raw",
        action="store_true",
        help="Deprecated compatibility flag; raw data is read in place by default",
    )
    curves.add_argument(
        "--copy-raw-snapshot",
        action="store_true",
        help="Explicitly copy raw record logs into the run directory for an immutable snapshot",
    )
    curves.add_argument("--out-dir", default="runs/btc_four_curve_v01")
    curves.add_argument("--frames-cache", default="", help="Reuse an immutable frames.npz from another run")
    curves.add_argument("--execution-cache", default="", help="Reuse exact Binance execution_quotes.npz")
    curves.add_argument("--duration-seconds", type=float, default=0.0)
    curves.add_argument("--cadence-ms", type=int, default=1000)
    curves.add_argument("--max-stale-ms", type=int, default=2000)
    curves.add_argument("--horizons", default="60,120,180,240,360,480,600,900,1200")
    curves.add_argument("--focus-seconds", type=float, default=480.0)
    curves.add_argument("--fee-round-trip-bps", type=float, default=10.0)
    curves.add_argument("--latency-ms", type=float, default=250.0)
    curves.add_argument("--notional-usd", type=float, default=1000.0)
    curves.add_argument("--event-quantile", type=float, default=0.50)
    curves.add_argument("--peak-floor", type=float, default=0.75)
    curves.add_argument(
        "--target-threshold-mode",
        choices=["fixed_edge", "fit_quantile"],
        default="fixed_edge",
    )
    curves.add_argument("--minimum-edge-bps", type=float, default=0.5)
    curves.add_argument("--forward-minimum-edge-bps", type=float, default=6.0)
    curves.add_argument("--full-quality-edge-bps", type=float, default=20.0)
    curves.add_argument(
        "--forward-curve-mode",
        choices=["dense_edge", "peak"],
        default="dense_edge",
    )
    curves.add_argument("--purge-seconds", type=float, default=600.0)
    curves.add_argument("--split-strategy", choices=["blocked", "tail"], default="blocked")
    curves.add_argument("--holdout-fraction", type=float, default=0.20, help="Final internal holdout fraction for blocked splits")
    curves.add_argument("--context-ticks", type=int, default=600)
    curves.add_argument("--background-stride", type=int, default=1)
    curves.add_argument("--supervision-ticks", type=int, default=1)
    curves.add_argument("--separate-task-towers", action="store_true")
    curves.add_argument("--hidden-size", type=int, default=128)
    curves.add_argument("--layers", type=int, default=4)
    curves.add_argument("--heads", type=int, default=8)
    curves.add_argument("--dropout", type=float, default=0.10)
    curves.add_argument("--epochs", type=int, default=20)
    curves.add_argument("--batch-size", type=int, default=4)
    curves.add_argument("--learning-rate", type=float, default=2e-4)
    curves.add_argument("--forward-peak-weight-cap", type=float, default=8.0)
    curves.add_argument("--backward-peak-weight-cap", type=float, default=16.0)
    curves.add_argument("--initial-checkpoint", default="")
    curves.add_argument("--seed", type=int, default=4105)
    curves.add_argument("--device", default="")
    curves.set_defaults(func=run_four_curve_pipeline)
    curve_replay = sub.add_parser("replay-four-curve", help="Replay an exported four-score checkpoint without target rules")
    curve_replay.add_argument("--run-dir", required=True)
    curve_replay.add_argument("--frames-cache", default="")
    curve_replay.add_argument("--split", choices=["train", "valid", "holdout"], default="holdout")
    curve_replay.add_argument("--batch-size", type=int, default=64)
    curve_replay.add_argument("--device", default="")
    curve_replay.add_argument("--checkpoint", default="")
    curve_replay.add_argument("--open-threshold", type=float, default=0.5)
    curve_replay.add_argument("--close-threshold", type=float, default=0.5)
    curve_replay.add_argument("--output-prefix", default="")
    curve_replay.set_defaults(func=replay_four_curve_run)
    ensemble = sub.add_parser("ensemble-four-curve", help="Average aligned four-score checkpoints and replay the ensemble")
    ensemble.add_argument("--run-dirs", required=True, help="Semicolon-separated member run directories")
    ensemble.add_argument("--frames-cache", required=True)
    ensemble.add_argument("--out-dir", default="runs/btc_four_curve_ensemble_v01")
    ensemble.add_argument("--split", choices=["valid", "holdout"], default="holdout")
    ensemble.add_argument("--batch-size", type=int, default=64)
    ensemble.add_argument("--device", default="")
    ensemble.set_defaults(func=ensemble_four_curve_runs)
    export = sub.add_parser("export-four-curve", help="Export a target-free four-score inference bundle")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--out-dir", required=True)
    export.set_defaults(func=lambda args: print(json.dumps(export_four_curve_bundle(args.run_dir, args.out_dir)), flush=True))
    incremental = sub.add_parser("prepare-incremental", help="Build new frames from a verified MREC suffix and saved market state")
    incremental.add_argument("--old-raw-dir", required=True)
    incremental.add_argument("--live-raw-dir", required=True)
    incremental.add_argument("--base-frames", required=True)
    incremental.add_argument("--base-execution", required=True)
    incremental.add_argument("--out-dir", required=True)
    incremental.add_argument("--cadence-ms", type=int, default=1000)
    incremental.add_argument("--max-stale-ms", type=int, default=2000)
    incremental.add_argument("--notional-usd", type=float, default=1000.0)
    incremental.add_argument("--workers", type=int, default=9)
    incremental.set_defaults(func=prepare_incremental_data)
    oos = sub.add_parser("evaluate-four-curve-oos", help="Evaluate a frozen four-score model on a later extension")
    oos.add_argument("--candidate-dir", required=True)
    oos.add_argument("--fit-run", required=True, help="Original run containing frozen targets and train split")
    oos.add_argument("--frames", required=True)
    oos.add_argument("--execution", required=True)
    oos.add_argument("--out-dir", required=True)
    oos.add_argument("--start-after-cutoff-seconds", type=float, default=0.0)
    oos.add_argument("--end-after-cutoff-seconds", type=float, default=0.0, help="0 uses the complete extension")
    oos.add_argument("--batch-size", type=int, default=64)
    oos.add_argument("--device", default="")
    oos.set_defaults(func=evaluate_four_curve_oos)
    select_epoch = sub.add_parser("select-economic-epoch", help="Select an epoch and score thresholds across validation regimes")
    select_epoch.add_argument("--run-dir", required=True)
    select_epoch.add_argument("--frames", required=True)
    select_epoch.add_argument("--execution", required=True)
    select_epoch.add_argument("--open-thresholds", default="0.5,0.6,0.7,0.8,0.9")
    select_epoch.add_argument("--close-thresholds", default="0.5,0.6,0.7,0.8,0.9")
    select_epoch.add_argument("--score-space", choices=["raw", "percentile"], default="percentile")
    select_epoch.add_argument("--min-trades-per-block", type=int, default=3)
    select_epoch.add_argument("--max-unresolved-per-block", type=int, default=0)
    select_epoch.add_argument("--latency-ms", type=float, default=250.0)
    select_epoch.add_argument("--fee-round-trip-bps", type=float, default=10.0)
    select_epoch.add_argument("--notional-usd", type=float, default=1000.0)
    select_epoch.set_defaults(func=select_economic_epoch)
    history = sub.add_parser(
        "prepare-binance-history",
        help="Download verified Binance Vision BBO/trades/depth/metrics and build causal frames",
    )
    history.add_argument("--start-date", required=True)
    history.add_argument("--end-date", required=True)
    history.add_argument("--out-dir", default="runs/binance_history")
    history.add_argument("--keep-archives", action="store_true")
    history.set_defaults(func=_prepare_binance_history)
    gold_download = sub.add_parser(
        "download-gold-m1",
        help="Download resumable XAUUSD M1 trendbars from the read-only cTrader Open API",
    )
    gold_download.add_argument("--secrets", default="secrets.gold.runtime.json")
    gold_download.add_argument("--symbol", default="XAUUSD")
    gold_download.add_argument("--start", default="2015-01-01")
    gold_download.add_argument("--end", default="now")
    gold_download.add_argument("--chunk-days", type=int, default=7)
    gold_download.add_argument("--out", default="source/ctrader/xauusd_m1.parquet")
    gold_download.set_defaults(func=_download_gold_m1)
    gold_refresh = sub.add_parser(
        "refresh-gold-token",
        help="Refresh gitignored cTrader OAuth tokens without printing token values",
    )
    gold_refresh.add_argument("--secrets", default="secrets.gold.runtime.json")
    gold_refresh.set_defaults(func=_refresh_gold_token)
    gold_run = sub.add_parser(
        "run-gold-minute",
        help="Train direct/retrieval TCNs on line and friction/reversal/continuation targets",
    )
    gold_run.add_argument("--candles", default="source/ctrader/xauusd_m1.parquet")
    gold_run.add_argument("--out-dir", default="runs/gold_minute_v01")
    gold_run.add_argument("--context-minutes", type=int, default=60)
    gold_run.add_argument("--horizons", default="5,10,15,20,30,45,60")
    gold_run.add_argument("--purge-minutes", type=int, default=120)
    gold_run.add_argument("--train-fraction", type=float, default=0.60)
    gold_run.add_argument("--valid-fraction", type=float, default=0.20)
    gold_run.add_argument("--sample-stride", type=int, default=3)
    gold_run.add_argument("--evaluation-stride", type=int, default=5)
    gold_run.add_argument("--hidden-size", type=int, default=96)
    gold_run.add_argument("--layers", type=int, default=6)
    gold_run.add_argument("--kernel-size", type=int, default=3)
    gold_run.add_argument("--dropout", type=float, default=0.10)
    gold_run.add_argument("--embedding-size", type=int, default=48)
    gold_run.add_argument("--batch-size", type=int, default=256)
    gold_run.add_argument("--epochs", type=int, default=200)
    gold_run.add_argument("--patience", type=int, default=20)
    gold_run.add_argument("--learning-rate", type=float, default=3e-4)
    gold_run.add_argument("--weight-decay", type=float, default=1e-4)
    gold_run.add_argument("--grad-clip", type=float, default=1.0)
    gold_run.add_argument("--retrieval-k", type=int, default=32)
    gold_run.add_argument("--seed", type=int, default=46947)
    gold_run.add_argument("--device", default="")
    gold_run.set_defaults(func=_run_gold_minute)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
