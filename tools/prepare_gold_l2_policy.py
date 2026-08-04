from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from stargaze_ml.gold.l2_seconds import (
    build_l2_second_feature_matrix,
    reconstruct_l2_seconds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal XAUUSD 1s candles, next-BBO execution data and quote-book VWAP proxies."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(r"C:\Users\r3d_flzp\Documents\GitHub\golden-den\raw.parquet"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gold_l2_policy_v1"))
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--max-quote-age-seconds", type=int, default=2)
    parser.add_argument("--min-levels", type=int, default=3)
    parser.add_argument("--max-new-quotes", type=int, default=20)
    parser.add_argument("--max-spread-ticks", type=float, default=500.0)
    return parser.parse_args()


def _iso(ts_ns: int) -> str:
    return datetime.fromtimestamp(int(ts_ns) / 1e9, tz=timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    raw_path = args.raw.expanduser().resolve(strict=True)
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "reconstruct", "raw": str(raw_path)}), flush=True)
    seconds = reconstruct_l2_seconds(
        raw_path,
        tick_size=args.tick_size,
        min_levels_per_side=args.min_levels,
        max_new_quotes_per_timestamp=args.max_new_quotes,
        max_spread_ticks=args.max_spread_ticks,
        max_quote_age_seconds=args.max_quote_age_seconds,
    )
    seconds_path = output_dir / "l2_seconds.parquet"
    seconds.write_parquet(seconds_path, compression="zstd", statistics=True)
    features = build_l2_second_feature_matrix(seconds, tick_size=args.tick_size)

    rows = len(features.ts_ns)
    train_end = int(rows * 0.60)
    validation_end = int(rows * 0.80)
    observed = seconds["observed"].to_numpy().astype(bool)
    quote_age_ms = seconds["quote_age_ms"].to_numpy().astype(np.float32)
    first_bid = seconds["first_bid"].to_numpy().astype(np.float64)
    first_ask = seconds["first_ask"].to_numpy().astype(np.float64)
    last_bid = seconds["last_bid"].to_numpy().astype(np.float64)
    last_ask = seconds["last_ask"].to_numpy().astype(np.float64)
    np.savez_compressed(
        output_dir / "prepared_l2_policy.npz",
        ts_ns=features.ts_ns,
        segment_id=features.segment_id,
        x=features.x,
        feature_names=np.asarray(features.feature_names),
        valid_feature=features.valid_feature,
        observed=observed,
        quote_age_ms=quote_age_ms,
        first_bid=first_bid,
        first_ask=first_ask,
        last_bid=last_bid,
        last_ask=last_ask,
        open=features.open,
        high=features.high,
        low=features.low,
        close=features.close,
        book_wap=features.book_wap,
        daily_book_vwap=features.daily_book_vwap,
        book_vwap_60s=features.book_vwap_60s,
        book_vwap_300s=features.book_vwap_300s,
        bid_vwap_60s=features.bid_vwap_60s,
        ask_vwap_60s=features.ask_vwap_60s,
        train_end=np.asarray(train_end, dtype=np.int64),
        validation_end=np.asarray(validation_end, dtype=np.int64),
    )
    segment_count = int(features.segment_id.max()) + 1
    manifest = {
        "source": {
            "path": str(raw_path),
            "rows": 75_978_432,
            "first_utc": _iso(features.ts_ns[0]),
            "last_utc": _iso(features.ts_ns[-1]),
            "historical_book": "inferred_full_refresh_reconstruction",
        },
        "seconds": {
            "rows": rows,
            "observed_rows": int(observed.sum()),
            "carried_rows": int((~observed).sum()),
            "segments": segment_count,
            "max_quote_age_seconds": args.max_quote_age_seconds,
            "decision_time": "after [t,t+1) is complete",
            "execution_time": "first accepted BBO in second t+1; no event when t+1 is carried",
        },
        "vwap": {
            "kind": "quote_liquidity_proxy_not_trade_vwap",
            "trade_prints_available": False,
            "book_wap": "top-of-book microprice",
            "daily_book_vwap": "causal UTC-day cumulative quote-liquidity weighted book_wap",
            "rolling": [60, 300],
            "bid_vwap_60s": "causal last_bid weighted by observed bid_size_top1 over [t-59s,t]",
            "ask_vwap_60s": "causal last_ask weighted by observed ask_size_top1 over [t-59s,t]",
            "carried_second_weight": 0,
        },
        "features": {
            "count": len(features.feature_names),
            "names": list(features.feature_names),
            "contract": (
                "only causal 1s mid OHLC, BBO, raw quote-VWAP levels and "
                "side-matched quote-VWAP distances"
            ),
        },
        "splits": {
            "method": "chronological 60/20/20; normalizer fitted on train only",
            "train_end": train_end,
            "validation_end": validation_end,
            "train_end_utc": _iso(features.ts_ns[train_end - 1]),
            "validation_end_utc": _iso(features.ts_ns[validation_end - 1]),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "prepared_l2_policy.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", **manifest["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
