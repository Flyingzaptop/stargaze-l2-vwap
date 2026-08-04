from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from stargaze_ml.gold.l2_policy import L2EventPolicy
from stargaze_ml.gold.l2_reinforce import ReinforceConfig, load_prepared_policy_data
from stargaze_ml.gold.l2_vwap_exit import VwapCrossMarket, evaluate_vwap_cross_variants
from stargaze_ml.training.data import RobustNormalizer


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate model-open plus causal first/second same-side VWAP crossing exits."
    )
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--seconds", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--split", choices=("validation", "test"), default="test")
    result.add_argument("--event-hazard-threshold", type=float, default=0.02)
    result.add_argument("--device", default="auto")
    return result


def _checkpoint(path: Path):
    payload = torch.load(path.expanduser().resolve(strict=True), map_location="cpu", weights_only=False)
    config_values = dict(payload["config"])
    config_values.pop("episodes_per_epoch", None)
    config = ReinforceConfig(**config_values)
    model = L2EventPolicy(
        len(payload["feature_names"]),
        config.hidden_size,
        initial_event_bias=config.initial_event_bias,
    )
    model.load_state_dict(payload["model_state"])
    return payload, config, model, RobustNormalizer.from_dict(payload["normalizer"])


def _write(path: Path, metrics: dict[str, object], records: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    schema = pa.schema(
        [
            ("entry_index", pa.int64()),
            ("exit_index", pa.int64()),
            ("entry_ts_ns", pa.int64()),
            ("exit_ts_ns", pa.int64()),
            ("side", pa.string()),
            ("entry_price", pa.float64()),
            ("exit_price", pa.float64()),
            ("holding_seconds", pa.int64()),
            ("net_ticks", pa.float64()),
            ("exit_reason", pa.string()),
            ("crossings_seen", pa.int64()),
            ("terminal", pa.bool_()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path / "trades.parquet")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    data = load_prepared_policy_data(args.prepared)
    payload, config, model, normalizer = _checkpoint(args.checkpoint)
    if tuple(payload["feature_names"]) != data.feature_names:
        raise ValueError("checkpoint feature contract does not match prepared data")
    market = VwapCrossMarket.from_parquet(args.seconds)
    start, end = (
        (data.train_end, data.validation_end)
        if args.split == "validation"
        else (data.validation_end, len(data))
    )
    evaluations = evaluate_vwap_cross_variants(
        model,
        data,
        config,
        normalizer,
        market,
        start=start,
        end=end,
        crossing_numbers=(1, 2),
        event_hazard_threshold=args.event_hazard_threshold,
        device=args.device,
    )
    output = args.out_dir.expanduser().resolve()
    summary: dict[str, object] = {
        "scope": f"chronological_{args.split}",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "seconds": str(args.seconds.expanduser().resolve()),
        "crossing_definition": {
            "long": "sign change of last_bid - bid_vwap_60s after entry fill",
            "short": "sign change of last_ask - ask_vwap_60s after entry fill",
            "execution": "cross detected at t; fill at first available BBO from t+1",
        },
        "variants": {},
    }
    for crossing_number, (metrics, records) in evaluations.items():
        metrics["scope"] = f"chronological_{args.split}"
        _write(output / f"cross_{crossing_number}", metrics, records)
        summary["variants"][str(crossing_number)] = metrics
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
