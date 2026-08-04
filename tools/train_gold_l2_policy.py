from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

import torch

from stargaze_ml.gold.l2_policy import L2EventPolicy, TETRAHEDRAL_ACTION_CODES
from stargaze_ml.gold.l2_reinforce import (
    ReinforceConfig,
    ReinforceTrainer,
    evaluate_sequential,
    load_prepared_policy_data,
    write_evaluation,
)
from stargaze_ml.training.data import RobustNormalizer


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Target-free REINFORCE over four XAU L2 trade events.")
    root.add_argument("--log-level", default="INFO")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--episode-length", type=int, default=128)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--entropy-start", type=float, default=0.0005)
    train.add_argument("--entropy-peak", type=float, default=0.01)
    train.add_argument("--entropy-end", type=float, default=0.0005)
    train.add_argument("--entropy-warmup-epochs", type=int, default=5)
    train.add_argument("--temperature-start", type=float, default=1.0)
    train.add_argument("--temperature-peak", type=float, default=1.3)
    train.add_argument("--temperature-end", type=float, default=0.9)
    train.add_argument("--event-floor-start", type=float, default=0.005)
    train.add_argument("--event-floor-peak", type=float, default=0.03)
    train.add_argument("--event-floor-end", type=float, default=0.0005)
    train.add_argument("--initial-event-bias", type=float, default=-5.0)
    train.add_argument("--commission-per-fill-ticks", type=float, default=15.0)
    train.add_argument("--slippage-per-fill-ticks", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=20260804)
    train.add_argument("--device", default="auto")
    train.add_argument("--no-amp", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prepared", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--out-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--event-hazard-threshold", type=float)
    return root


def _config(args: argparse.Namespace) -> ReinforceConfig:
    return ReinforceConfig(
        hidden_size=args.hidden_size,
        episode_length=args.episode_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        entropy_start=args.entropy_start,
        entropy_peak=args.entropy_peak,
        entropy_end=args.entropy_end,
        entropy_warmup_epochs=args.entropy_warmup_epochs,
        temperature_start=args.temperature_start,
        temperature_peak=args.temperature_peak,
        temperature_end=args.temperature_end,
        event_floor_start=args.event_floor_start,
        event_floor_peak=args.event_floor_peak,
        event_floor_end=args.event_floor_end,
        initial_event_bias=args.initial_event_bias,
        commission_per_fill_ticks=args.commission_per_fill_ticks,
        slippage_per_fill_ticks=args.slippage_per_fill_ticks,
        seed=args.seed,
        use_amp=not args.no_amp,
    )


def _load_checkpoint(path: Path, device: str):
    resolved = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    payload = torch.load(path, map_location=resolved, weights_only=False)
    config_values = dict(payload["config"])
    # Pre-full-pass checkpoints used a with-replacement episode budget.  It is
    # irrelevant for evaluation and must not leak back into the new contract.
    config_values.pop("episodes_per_epoch", None)
    config = ReinforceConfig(**config_values)
    model = L2EventPolicy(
        len(payload["feature_names"]),
        config.hidden_size,
        initial_event_bias=config.initial_event_bias,
    )
    model.load_state_dict(payload["model_state"])
    normalizer = RobustNormalizer.from_dict(payload["normalizer"])
    return payload, config, model, normalizer


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    data = load_prepared_policy_data(args.prepared)
    if args.command == "train":
        config = _config(args)
        config.validate()
        model = L2EventPolicy(
            len(data.feature_names),
            config.hidden_size,
            initial_event_bias=config.initial_event_bias,
        )
        output_dir = args.out_dir.resolve()
        trainer = ReinforceTrainer(model, data, config, output_dir, device=args.device)
        history = trainer.train()
        metrics, records = evaluate_sequential(
            model,
            data,
            config,
            trainer.normalizer,
            start=data.train_end,
            end=data.validation_end,
            device=args.device,
        )
        metrics["scope"] = "chronological_validation_60_80"
        write_evaluation(output_dir / "validation", metrics, records)
        result = {
            "config": asdict(config),
            "rows": len(data),
            "train_end": data.train_end,
            "validation_end": data.validation_end,
            "feature_names": list(data.feature_names),
            "four_visible_action_codes": {
                action.name.lower(): list(code)
                for action, code in TETRAHEDRAL_ACTION_CODES.items()
            },
            "implicit_action": "no_op from total event hazard; not a fifth neural output",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
            "validation": metrics,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        payload, config, model, normalizer = _load_checkpoint(args.checkpoint, args.device)
        if tuple(payload["feature_names"]) != data.feature_names:
            raise ValueError("checkpoint feature contract does not match prepared data")
        if args.split == "validation":
            start, end = data.train_end, data.validation_end
        else:
            start, end = data.validation_end, len(data)
        metrics, records = evaluate_sequential(
            model,
            data,
            config,
            normalizer,
            start=start,
            end=end,
            device=args.device,
            event_hazard_threshold=args.event_hazard_threshold,
        )
        metrics["scope"] = f"chronological_{args.split}"
        metrics["checkpoint"] = str(args.checkpoint.resolve())
        write_evaluation(args.out_dir, metrics, records)
        result = metrics
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
