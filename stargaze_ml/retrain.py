from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import shutil

import numpy as np

from .artifacts import load_frames, write_json
from .labels import build_labels
from .models import PolicyConfig
from .replay import replay_policy
from .reports import policy_summary, write_policy_artifacts
from .training import PolicyWindowDataset, RobustNormalizer, build_examples, purged_blocked_splits, purged_chronological_splits, train_policy
from .training.data import POSITION_FEATURE_NAMES


def _slice_frames(frames, start: int, end: int):
    from .contracts import CausalFrames

    return CausalFrames(
        ts_ns=frames.ts_ns[start:end], x=frames.x[start:end], venue_x=frames.venue_x[start:end],
        bid=frames.bid[start:end], ask=frames.ask[start:end], valid=frames.valid[start:end],
        segment_id=frames.segment_id[start:end], feature_names=frames.feature_names,
        venue_feature_names=frames.venue_feature_names, venues=frames.venues,
    )


def run(args: argparse.Namespace) -> None:
    started = perf_counter()
    source = Path(args.source_run)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(source / "frames.npz")
    with np.load(source / "scores.npz", allow_pickle=False) as scores:
        score = {name: scores[name] for name in scores.files}
    horizons = tuple(float(x) for x in score["horizons_seconds"])
    frames.valid &= np.any(score["forward_valid"], axis=1) & np.any(score["backward_valid"], axis=1)
    split_fn = purged_blocked_splits if args.split_strategy == "blocked" else purged_chronological_splits
    splits = split_fn(frames.ts_ns, frames.valid, purge_seconds=float(args.purge_seconds))
    labels = build_labels(
        frames,
        forward_long_h=score["forward_long"], forward_short_h=score["forward_short"],
        backward_long_h=score["backward_long"], backward_short_h=score["backward_short"],
        horizons_seconds=horizons, fit_mask=splits.train,
        event_high_quantile=float(args.event_high_quantile), peak_nms_seconds=float(args.peak_nms_seconds),
    )
    examples = build_examples(frames, labels, context_ticks=int(args.context_ticks), skip_stride=1, hold_stride=1)
    train_examples = examples.select(splits.train)
    valid_examples = examples.select(splits.valid)
    holdout_examples = examples.select(splits.holdout)
    x_normalizer = RobustNormalizer.fit(frames.x, splits.train)
    venue_normalizer = RobustNormalizer.fit(frames.venue_x, splits.train)
    common = dict(
        frames=frames, labels=labels,
        forward_long_h=score["forward_long"], forward_short_h=score["forward_short"],
        backward_long_h=score["backward_long"], backward_short_h=score["backward_short"],
        horizons_seconds=horizons, context_ticks=int(args.context_ticks),
        x_normalizer=x_normalizer, venue_normalizer=venue_normalizer,
    )
    train_dataset = PolicyWindowDataset(examples=train_examples, **common)
    valid_dataset = PolicyWindowDataset(examples=valid_examples, **common)
    model_config = PolicyConfig(
        input_dim=train_dataset.input_dim, venue_feature_dim=train_dataset.venue_feature_dim,
        d_model=int(args.hidden_size), nhead=int(args.heads), num_layers=int(args.layers),
        dim_feedforward=int(args.hidden_size) * 4, dropout=float(args.dropout), num_horizons=len(horizons),
    )
    print(json.dumps({"stage": "retrain", "train_examples": len(train_dataset), "valid_examples": len(valid_dataset), "episodes": len(labels.episodes), "model": asdict(model_config)}), flush=True)
    trained = train_policy(
        train_dataset, valid_dataset, model_config=model_config, epochs=int(args.epochs),
        batch_size=int(args.batch_size), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip), seed=int(args.seed), out_dir=out, device=args.device or None,
    )
    np.savez_compressed(out / "splits.npz", train=splits.train, valid=splits.valid, holdout=splits.holdout)
    np.savez_compressed(
        out / "labels.npz", flat_action=labels.flat_action, open_long_zone=labels.open_long_zone,
        open_short_zone=labels.open_short_zone, close_long_zone=labels.close_long_zone,
        close_short_zone=labels.close_short_zone, dominant_long_horizon=labels.dominant_long_horizon,
        dominant_short_horizon=labels.dominant_short_horizon,
    )
    write_json(out / "normalizers.json", {"x": x_normalizer.to_dict(), "venue": venue_normalizer.to_dict()})
    write_json(out / "feature_schema.json", {
        "base_features": list(frames.feature_names), "venue_features": list(frames.venue_feature_names),
        "position_features": list(POSITION_FEATURE_NAMES), "venues": list(frames.venues),
        "horizons_seconds": list(horizons), "backward_score_values_and_validity_are_model_inputs": True,
        "action_selection": "state-conditioned model argmax",
    })
    for name in ("frames.npz", "frames.npz.manifest.json", "scores.npz", "auxiliary_targets.npz"):
        source_path = source / name
        if source_path.exists():
            shutil.copy2(source_path, out / name)
    holdout_idx = np.flatnonzero(splits.holdout)
    replay_start = max(0, int(holdout_idx[0]) - int(args.context_ticks) + 1)
    replay_end = int(holdout_idx[-1]) + 1
    replay_frames = _slice_frames(frames, replay_start, replay_end)
    replay = replay_policy(
        trained.model, replay_frames, x_normalizer=x_normalizer, venue_normalizer=venue_normalizer,
        backward_long_h=score["backward_long"][replay_start:replay_end],
        backward_short_h=score["backward_short"][replay_start:replay_end],
        context_ticks=int(args.context_ticks), device=args.device or None,
    )
    replay_labels = build_labels(
        replay_frames, forward_long_h=score["forward_long"][replay_start:replay_end],
        forward_short_h=score["forward_short"][replay_start:replay_end],
        backward_long_h=score["backward_long"][replay_start:replay_end],
        backward_short_h=score["backward_short"][replay_start:replay_end], horizons_seconds=horizons,
        fit_mask=np.ones(len(replay_frames.ts_ns), dtype=bool), event_high_quantile=float(args.event_high_quantile),
        peak_nms_seconds=float(args.peak_nms_seconds),
    )
    summary = policy_summary(replay, replay_labels)
    best = trained.history[trained.best_epoch - 1]
    summary.update({
        "source_run": str(source.resolve()), "best_epoch": trained.best_epoch,
        "best_valid_loss": trained.best_valid_loss, "best_selection_score": trained.best_selection_score,
        "best_valid_macro_action_recall": float(best["valid_macro_action_recall"]),
        "episodes": len(labels.episodes), "train_examples": len(train_examples.center_idx),
        "valid_examples": len(valid_examples.center_idx), "holdout_examples": len(holdout_examples.center_idx),
        "elapsed_seconds": perf_counter() - started, "checkpoint_status": "pilot_only", "execution_ready": False,
    })
    write_policy_artifacts(out, replay, summary)
    write_json(out / "run_manifest.json", {"args": {k: v for k, v in vars(args).items() if k != "func"}, "model_config": asdict(model_config), "summary": summary})
    print(json.dumps({"stage": "complete", **summary}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retrain a policy from saved causal frames and scores")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-strategy", choices=["blocked", "tail"], default="blocked")
    parser.add_argument("--purge-seconds", type=float, default=120.0)
    parser.add_argument("--event-high-quantile", type=float, default=0.95)
    parser.add_argument("--peak-nms-seconds", type=float, default=5.0)
    parser.add_argument("--context-ticks", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4105)
    parser.add_argument("--device", default="")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
