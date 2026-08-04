from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from stargaze_ml.gold.l2_open_reinforce import (
    OpenReinforceConfig,
    PreparedOpenData,
    _event_indices,
    _pnl_ticks,
)


def main() -> None:
    prepared = Path("runs/gold_l2_open_v1/prepared_l2_open_policy.npz")
    seconds_path = Path("runs/gold_l2_policy_v2/l2_seconds.parquet")
    out = Path("runs/gold_l2_open_v1/dominance").resolve(); out.mkdir(parents=True, exist_ok=True)
    data = PreparedOpenData(prepared); config = OpenReinforceConfig(); n = len(data.x)
    mid = data.mid.astype(np.float64); vwap = data.primary_vwap.astype(np.float64)
    segment = data.segment_id; same = np.r_[False, segment[1:] == segment[:-1]]
    gap_previous = np.zeros(n); price_change = np.zeros(n); vwap_change = np.zeros(n)
    gap_previous[1:] = np.where(same[1:], mid[:-1] - vwap[:-1], 0.0)
    price_change[1:] = np.where(same[1:], mid[1:] - mid[:-1], 0.0)
    vwap_change[1:] = np.where(same[1:], vwap[1:] - vwap[:-1], 0.0)
    reset = np.r_[True, segment[1:] != segment[:-1]]
    group_start = np.maximum.accumulate(np.where(reset, np.arange(n), 0)); rows = np.arange(n)

    def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
        starts = np.maximum(group_start, rows - window + 1)
        prefix = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
        return prefix[rows + 1] - prefix[starts]

    splits = {
        "validation": _event_indices(data, data.train_end, data.validation_end, good_only=False),
        "test": _event_indices(data, data.validation_end, n, good_only=False),
    }
    split_gate = {name: data.event_gate_index[events] for name, events in splits.items()}
    split_pnl = {name: _pnl_ticks(data, events, split_gate[name], 1, config) for name, events in splits.items()}
    indicators: dict[str, dict[str, float]] = {}

    def score_indicator(name: str, values: np.ndarray) -> None:
        result: dict[str, float] = {}
        for split, gate in split_gate.items():
            x = values[gate]; pnl = split_pnl[split]; valid = np.isfinite(x)
            result[f"{split}_auc_profitable"] = float(roc_auc_score(pnl[valid] > 0, x[valid]))
            q20, q80 = np.quantile(x[valid], [0.2, 0.8])
            result[f"{split}_bottom20_mean_pnl"] = float(pnl[valid & (x <= q20)].mean())
            result[f"{split}_top20_mean_pnl"] = float(pnl[valid & (x >= q80)].mean())
        indicators[name] = result

    sign_gap = np.sign(gap_previous)
    for horizon in (30, 60, 120, 300, 900):
        denominator = rolling_sum(gap_previous * gap_previous, horizon)
        alpha_price = np.divide(
            rolling_sum(gap_previous * price_change, horizon), denominator,
            out=np.zeros(n), where=denominator > 1e-12,
        )
        alpha_vwap = np.divide(
            rolling_sum(gap_previous * vwap_change, horizon), denominator,
            out=np.zeros(n), where=denominator > 1e-12,
        )
        ecm = (-alpha_price - alpha_vwap) / (np.abs(alpha_price) + np.abs(alpha_vwap) + 1e-12)
        score_indicator(f"error_correction_{horizon}s", ecm)
        price_toward = rolling_sum(np.maximum(-sign_gap * price_change, 0.0), horizon)
        vwap_toward = rolling_sum(np.maximum(sign_gap * vwap_change, 0.0), horizon)
        motion = (price_toward - vwap_toward) / (price_toward + vwap_toward + 1e-12)
        score_indicator(f"closure_share_{horizon}s", motion)

    names = {name: i for i, name in enumerate(data.feature_names)}
    vwap15 = (data.x[:, names["bid_vwap_15s"]] + data.x[:, names["ask_vwap_15s"]]) * 0.5
    vwap900 = (data.x[:, names["bid_vwap_900s"]] + data.x[:, names["ask_vwap_900s"]]) * 0.5
    score_indicator("vwap_fan_15s_900s", -data.side * (vwap15 - vwap900) / config.tick_size)
    score_indicator("vwap60_slope_reversion", -data.side * data.x[:, names["mid_vwap_60s_slope_1s_ticks"]])
    seconds = pl.read_parquet(seconds_path, columns=["bid_size_top1", "ask_size_top1", "book_wap"])
    bid_size = seconds["bid_size_top1"].to_numpy(); ask_size = seconds["ask_size_top1"].to_numpy()
    imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-12)
    score_indicator("top_book_imbalance_reversion", -data.side * imbalance)
    score_indicator("microprice_pressure_reversion", -data.side * ((seconds["book_wap"].to_numpy() - mid) / config.tick_size))

    dominance: dict[str, object] = {}
    for split, events in splits.items():
        gate = split_gate[split]; crossing = data.event_crossing_1[events]; side = data.event_side[events]
        price_closure = -side * (mid[crossing] - mid[gate]) / config.tick_size
        vwap_closure = side * (vwap[crossing] - vwap[gate]) / config.tick_size
        price_wins = price_closure > vwap_closure
        pnl = split_pnl[split]
        dominance[split] = {
            "events": int(len(events)), "price_moves_to_vwap_fraction": float(price_wins.mean()),
            "mean_price_closure_ticks": float(price_closure.mean()),
            "mean_vwap_closure_ticks": float(vwap_closure.mean()),
            "mean_pnl_when_price_moves_more": float(pnl[price_wins].mean()),
            "mean_pnl_when_vwap_moves_more": float(pnl[~price_wins].mean()),
        }

    ordered = sorted(indicators, key=lambda key: abs(indicators[key]["validation_auc_profitable"] - 0.5), reverse=True)
    report = {"definition": "positive score means VWAP dominance / mean-reversion pressure", "dominance": dominance,
              "indicator_ranking_by_validation": ordered, "indicators": indicators}
    (out / "dominance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    shown = ordered[:10]; y = np.arange(len(shown)); val = [indicators[k]["validation_auc_profitable"] for k in shown]; test = [indicators[k]["test_auc_profitable"] for k in shown]
    fig, ax = plt.subplots(figsize=(12, 6)); ax.barh(y - 0.18, val, height=0.34, label="validation"); ax.barh(y + 0.18, test, height=0.34, label="test")
    ax.axvline(0.5, color="black", ls="--", lw=1); ax.set_yticks(y, shown); ax.invert_yaxis(); ax.set_xlim(0.35, 0.65); ax.set_xlabel("AUC: predicts profitable first-cross trade"); ax.legend(); ax.grid(axis="x", alpha=0.2); fig.tight_layout(); fig.savefig(out / "indicator_auc.png", dpi=160); plt.close(fig)
    print(json.dumps({"dominance": dominance, "top": [(k, indicators[k]) for k in ordered[:5]]}), flush=True)


if __name__ == "__main__":
    main()
