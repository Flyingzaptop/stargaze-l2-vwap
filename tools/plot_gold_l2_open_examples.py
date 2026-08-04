from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from stargaze_ml.gold.l2_open_policy import L2OpenPolicy
from stargaze_ml.gold.l2_open_reinforce import (
    OpenReinforceConfig,
    PreparedOpenData,
    _event_indices,
    _pnl_ticks,
)
from stargaze_ml.training.data import RobustNormalizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, default=Path("runs/gold_l2_open_v1/prepared_l2_open_policy.npz"))
    p.add_argument("--checkpoint", type=Path, default=Path("runs/gold_l2_open_v1/reinforce/final.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/gold_l2_open_v1/reinforce/open_examples"))
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--threshold", type=float, default=0.0070710678118654745)
    args = p.parse_args()

    data = PreparedOpenData(args.prepared)
    checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    config = OpenReinforceConfig(**checkpoint["config"])
    model = L2OpenPolicy(len(data.feature_names), config.hidden_size)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalizer = RobustNormalizer.from_dict(checkpoint["normalizer"])
    events = _event_indices(data, data.validation_end, len(data.x), good_only=False)

    candidates: list[tuple[int, int, np.ndarray]] = []
    with torch.no_grad():
        for event in events:
            left = int(data.event_start[event]); right = int(data.event_crossing_1[event])
            gate = int(data.event_gate_index[event]); x = normalizer.transform(data.x[left:right])[None]
            probability = torch.sigmoid(model(torch.from_numpy(x)))[0].numpy()
            hit = np.flatnonzero(probability[gate - left :] >= args.threshold)
            if hit.size:
                candidates.append((int(event), gate + int(hit[0]), probability))
    if len(candidates) < args.count:
        raise RuntimeError(f"only {len(candidates)} model trades are available")
    selected = np.random.default_rng(args.seed).choice(len(candidates), args.count, replace=False)
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    names = {name: i for i, name in enumerate(data.feature_names)}
    raw_boundaries = np.flatnonzero(
        np.r_[True, data.event_id[1:] != data.event_id[:-1], True]
    )

    for number, candidate_index in enumerate(selected, start=1):
        event, entry, probability = candidates[int(candidate_index)]
        start = int(data.event_start[event]); cross1 = int(data.event_crossing_1[event]); cross2 = int(data.event_crossing_2[event])
        gate = int(data.event_gate_index[event]); left = max(start - 60, 0); right = min(cross2 + 61, len(data.x))
        rows = np.arange(left, right); time = (data.ts_ns[rows] - data.ts_ns[start]) / 1e9
        side = "LONG" if int(data.event_side[event]) < 0 else "SHORT"
        entries = np.asarray([entry], dtype=np.int64); ev = np.asarray([event], dtype=np.int64)
        pnl1 = float(_pnl_ticks(data, ev, entries, 1, config)[0]); pnl2 = float(_pnl_ticks(data, ev, entries, 2, config)[0])
        bid300 = data.x[rows, names["bid_vwap_300s"]]; ask300 = data.x[rows, names["ask_vwap_300s"]]
        bid900 = data.x[rows, names["bid_vwap_900s"]]; ask900 = data.x[rows, names["ask_vwap_900s"]]
        full_probability = np.full(len(rows), np.nan, dtype=np.float64)
        first_group = max(0, int(np.searchsorted(raw_boundaries, left, side="right") - 1))
        last_group = min(len(raw_boundaries) - 2, int(np.searchsorted(raw_boundaries, right - 1, side="right") - 1))
        with torch.no_grad():
            for group in range(first_group, last_group + 1):
                group_left = int(raw_boundaries[group]); group_right = int(raw_boundaries[group + 1])
                group_x = normalizer.transform(data.x[group_left:group_right])[None]
                group_probability = torch.sigmoid(model(torch.from_numpy(group_x)))[0].numpy()
                visible_left = max(left, group_left); visible_right = min(right, group_right)
                full_probability[visible_left-left:visible_right-left] = group_probability[
                    visible_left-group_left:visible_right-group_left
                ]
        event_reset_visible = np.r_[True, data.event_id[rows][1:] != data.event_id[rows][:-1]]
        upward = np.flatnonzero(
            np.isfinite(full_probability)
            & (full_probability >= args.threshold)
            & (event_reset_visible | np.r_[True, (~np.isfinite(full_probability[:-1])) | (full_probability[:-1] < args.threshold)])
        )

        fig, (ax, output_ax) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(time, data.last_bid[rows], color="#2ca02c", lw=0.9, label="Bid")
        ax.plot(time, data.last_ask[rows], color="#d62728", lw=0.9, label="Ask")
        ax.plot(time, data.primary_vwap[rows], color="#1f77b4", lw=1.8, label="VWAP mid 60s")
        ax.plot(time, (bid300 + ask300) * 0.5, color="#9467bd", lw=1.0, alpha=0.75, label="VWAP mid 300s")
        ax.plot(time, (bid900 + ask900) * 0.5, color="#8c564b", lw=1.0, alpha=0.65, label="VWAP mid 900s")
        for index, color, label in ((gate, "#ff7f0e", "gate ON"), (entry, "#111111", "OPEN signal"), (cross1, "#17becf", "cross #1"), (cross2, "#e377c2", "cross #2")):
            xmark = (data.ts_ns[index] - data.ts_ns[start]) / 1e9
            ax.axvline(xmark, color=color, lw=1.4, ls="--", label=label)
        fill_index = entry + 1
        fill_price = data.first_ask[fill_index] if side == "LONG" else data.first_bid[fill_index]
        ax.scatter(
            [(data.ts_ns[fill_index] - data.ts_ns[start]) / 1e9], [fill_price],
            marker="^" if side == "LONG" else "v", color="#111111", s=75,
            zorder=7, label="actual fill t+1",
        )
        ax.set_ylabel("XAUUSD")
        ax.set_title(f"Random test trade {number}: {side} | PnL cross#1 {pnl1:+.0f} ticks | cross#2 {pnl2:+.0f} ticks")
        ax.grid(alpha=0.2); ax.legend(ncol=4, fontsize=8, loc="best")

        output_ax.plot(time, full_probability, color="#111111", lw=1.2, label="full model P(open)")
        output_ax.axhline(args.threshold, color="#d62728", ls="--", lw=1.0, label=f"threshold {args.threshold:.4f}")
        output_ax.fill_between(
            time, 0, 1, where=data.gate_open[rows], transform=output_ax.get_xaxis_transform(),
            color="#ff7f0e", alpha=0.10, label="gate active",
        )
        output_ax.scatter(
            time[upward], full_probability[upward], marker="x", color="#d62728",
            s=38, zorder=5, label="threshold reached / event reset",
        )
        output_ax.scatter(
            [(data.ts_ns[entry] - data.ts_ns[start]) / 1e9],
            [full_probability[entry-left]], color="#111111", s=48, zorder=6,
            label="accepted OPEN signal",
        )
        output_ax.set_ylabel("P(open)"); output_ax.set_xlabel("Seconds from event start")
        output_ax.set_ylim(bottom=0); output_ax.grid(alpha=0.2); output_ax.legend(ncol=4, fontsize=8, loc="best")
        fig.tight_layout(); fig.savefig(out / f"open_{number:02d}.png", dpi=150); plt.close(fig)

    print(out)


if __name__ == "__main__":
    main()
