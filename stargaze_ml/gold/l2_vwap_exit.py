from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from stargaze_ml.training.data import RobustNormalizer

from .l2_policy import L2EventPolicy
from .l2_reinforce import (
    PreparedPolicyData,
    ReinforceConfig,
    _segment_ranges,
    resolve_device,
)


@dataclass(frozen=True)
class VwapCrossMarket:
    """Causal per-second BBO and same-side 60-second VWAP values."""

    last_bid: np.ndarray
    last_ask: np.ndarray
    bid_vwap_60s: np.ndarray
    ask_vwap_60s: np.ndarray

    @classmethod
    def from_parquet(cls, path: str | Path) -> "VwapCrossMarket":
        table = pq.read_table(
            Path(path).expanduser().resolve(strict=True),
            columns=["last_bid", "last_ask", "bid_vwap_60s", "ask_vwap_60s"],
        )
        return cls(
            **{
                name: np.ascontiguousarray(table[name].to_numpy(), dtype=np.float64)
                for name in table.column_names
            }
        )

    def validate(self, rows: int) -> None:
        for name in ("last_bid", "last_ask", "bid_vwap_60s", "ask_vwap_60s"):
            values = getattr(self, name)
            if values.shape != (rows,):
                raise ValueError(f"{name} must have one value per prepared row")


def _cross_side(value: float) -> int:
    if not np.isfinite(value) or value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def _spread_side(market: VwapCrossMarket, index: int, position: int) -> int:
    if position > 0:
        return _cross_side(market.last_bid[index] - market.bid_vwap_60s[index])
    return _cross_side(market.last_ask[index] - market.ask_vwap_60s[index])


def _trade_record(
    data: PreparedPolicyData,
    entry_index: int,
    exit_index: int,
    side: int,
    entry_price: float,
    exit_price: float,
    net_ticks: float,
    *,
    exit_reason: str,
    crossings_seen: int,
) -> dict[str, object]:
    return {
        "entry_index": int(entry_index),
        "exit_index": int(exit_index),
        "entry_ts_ns": int(data.ts_ns[entry_index]),
        "exit_ts_ns": int(data.ts_ns[exit_index]),
        "side": "long" if side > 0 else "short",
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "holding_seconds": int(
            (data.ts_ns[exit_index] - data.ts_ns[entry_index]) // 1_000_000_000
        ),
        "net_ticks": float(net_ticks),
        "exit_reason": exit_reason,
        "crossings_seen": int(crossings_seen),
        "terminal": exit_reason == "forced_terminal",
    }


def _close_record(
    data: PreparedPolicyData,
    config: ReinforceConfig,
    entry_index: int,
    exit_index: int,
    position: int,
    entry_price: float,
    *,
    exit_reason: str,
    crossings_seen: int,
) -> dict[str, object]:
    slippage = config.slippage_per_fill_ticks * config.tick_size
    commission_round_trip = 2.0 * config.commission_per_fill_ticks
    if position > 0:
        exit_price = float(data.first_bid[exit_index]) - slippage
        gross_ticks = (exit_price - entry_price) / config.tick_size
    else:
        exit_price = float(data.first_ask[exit_index]) + slippage
        gross_ticks = (entry_price - exit_price) / config.tick_size
    return _trade_record(
        data,
        entry_index,
        exit_index,
        position,
        entry_price,
        exit_price,
        gross_ticks - commission_round_trip,
        exit_reason=exit_reason,
        crossings_seen=crossings_seen,
    )


def ledger_for_vwap_cross_segment(
    hazards: np.ndarray,
    data: PreparedPolicyData,
    market: VwapCrossMarket,
    left: int,
    right: int,
    config: ReinforceConfig,
    *,
    crossing_number: int,
    event_hazard_threshold: float,
) -> list[dict[str, object]]:
    """Open from model hazards and close on the selected same-side VWAP crossing.

    A decision at second ``t`` executes against the first BBO at ``t + 1``.
    Crossing state begins at the actual entry fill. Exact touches are ignored
    until price leaves VWAP; a change between non-zero sides counts once.
    """

    if crossing_number not in (1, 2):
        raise ValueError("crossing_number must be one or two")
    if not np.isfinite(event_hazard_threshold) or event_hazard_threshold < 0.0:
        raise ValueError("event_hazard_threshold must be finite and non-negative")
    expected = right - left - 2
    values = np.asarray(hazards, dtype=np.float64)
    if values.shape != (expected, 4):
        raise ValueError("hazards must have shape [segment decisions, 4]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("hazards must be finite and non-negative")

    position = 0
    entry_index = -1
    entry_price = 0.0
    last_side = 0
    crossings_seen = 0
    exit_pending = False
    records: list[dict[str, object]] = []
    slippage = config.slippage_per_fill_ticks * config.tick_size

    for offset, event_hazards in enumerate(values):
        decision_index = left + offset
        execution_index = decision_index + 1
        execution_valid = bool(data.observed[execution_index])

        if position:
            if exit_pending:
                if execution_valid:
                    records.append(
                        _close_record(
                            data,
                            config,
                            entry_index,
                            execution_index,
                            position,
                            entry_price,
                            exit_reason=f"vwap_cross_{crossing_number}",
                            crossings_seen=crossings_seen,
                        )
                    )
                    position = 0
                    exit_pending = False
                continue

            current_side = _spread_side(market, decision_index, position)
            if current_side:
                if last_side and current_side != last_side:
                    crossings_seen += 1
                last_side = current_side
            if crossings_seen >= crossing_number:
                if execution_valid:
                    records.append(
                        _close_record(
                            data,
                            config,
                            entry_index,
                            execution_index,
                            position,
                            entry_price,
                            exit_reason=f"vwap_cross_{crossing_number}",
                            crossings_seen=crossings_seen,
                        )
                    )
                    position = 0
                else:
                    exit_pending = True
            continue

        if not execution_valid:
            continue
        open_choice = int(np.argmax(event_hazards[:2]))
        if event_hazards[open_choice] < event_hazard_threshold:
            continue
        position = 1 if open_choice == 0 else -1
        entry_index = execution_index
        entry_price = (
            float(data.first_ask[entry_index]) + slippage
            if position > 0
            else float(data.first_bid[entry_index]) - slippage
        )
        last_side = _spread_side(market, entry_index, position)
        crossings_seen = 0
        exit_pending = False

    if position:
        records.append(
            _close_record(
                data,
                config,
                entry_index,
                right - 1,
                position,
                entry_price,
                exit_reason="forced_terminal",
                crossings_seen=crossings_seen,
            )
        )
    return records


def _hazards_by_segment(
    model: L2EventPolicy,
    data: PreparedPolicyData,
    normalizer: RobustNormalizer,
    ranges: list[tuple[int, int]],
    *,
    device: str,
) -> dict[tuple[int, int], np.ndarray]:
    resolved = resolve_device(device)
    model = model.to(resolved)
    model.eval()
    x = normalizer.transform(data.x)
    ordered = sorted(ranges, key=lambda bounds: bounds[1] - bounds[0])
    batches: list[list[tuple[int, int]]] = []
    batch: list[tuple[int, int]] = []
    max_steps = 0
    for bounds in ordered:
        steps = bounds[1] - bounds[0] - 2
        prospective = max(max_steps, steps)
        if batch and (len(batch) >= 128 or prospective * (len(batch) + 1) > 262_144):
            batches.append(batch)
            batch = []
            max_steps = 0
        batch.append(bounds)
        max_steps = max(max_steps, steps)
    if batch:
        batches.append(batch)

    result: dict[tuple[int, int], np.ndarray] = {}
    with torch.no_grad():
        for range_batch in batches:
            lengths = np.asarray([right - left - 2 for left, right in range_batch])
            features = np.zeros(
                (len(range_batch), int(lengths.max()), data.x.shape[1]), dtype=np.float32
            )
            for row, (left, right) in enumerate(range_batch):
                features[row, : lengths[row]] = x[left : right - 2]
            logits, _ = model(torch.as_tensor(features, device=resolved))
            hazards = F.softplus(logits.float()).cpu().numpy()
            for row, bounds in enumerate(range_batch):
                result[bounds] = np.ascontiguousarray(hazards[row, : lengths[row]])
    return result


def _profit_factor(values: np.ndarray) -> float:
    gains = float(values[values > 0.0].sum())
    losses = float(-values[values < 0.0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def _metrics(
    records: list[dict[str, object]],
    *,
    crossing_number: int,
    event_hazard_threshold: float,
    start: int,
    end: int,
) -> dict[str, object]:
    pnl = np.asarray([record["net_ticks"] for record in records], dtype=np.float64)
    holds = np.asarray([record["holding_seconds"] for record in records], dtype=np.int64)
    ordered = sorted(records, key=lambda record: (record["exit_ts_ns"], record["entry_ts_ns"]))
    chronological = np.asarray([record["net_ticks"] for record in ordered], dtype=np.float64)
    cumulative = np.cumsum(chronological) if len(chronological) else np.zeros(0)
    drawdown = np.maximum.accumulate(np.r_[0.0, cumulative]) - np.r_[0.0, cumulative]
    return {
        "range": {"start": int(start), "end": int(end), "seconds_rows": int(end - start)},
        "exit_policy": f"same_side_vwap_60s_cross_{crossing_number}",
        "crossing_number": crossing_number,
        "event_hazard_threshold": float(event_hazard_threshold),
        "trades": int(len(records)),
        "long_trades": int(sum(record["side"] == "long" for record in records)),
        "short_trades": int(sum(record["side"] == "short" for record in records)),
        "vwap_cross_closes": int(sum(not record["terminal"] for record in records)),
        "forced_terminal_closes": int(sum(record["terminal"] for record in records)),
        "total_net_ticks": float(pnl.sum()),
        "mean_trade_net_ticks": float(pnl.mean()) if len(pnl) else 0.0,
        "median_trade_net_ticks": float(np.median(pnl)) if len(pnl) else 0.0,
        "hit_rate": float((pnl > 0.0).mean()) if len(pnl) else 0.0,
        "profit_factor": _profit_factor(pnl),
        "max_drawdown_ticks": float(drawdown.max()) if len(drawdown) else 0.0,
        "mean_holding_seconds": float(holds.mean()) if len(holds) else 0.0,
        "median_holding_seconds": float(np.median(holds)) if len(holds) else 0.0,
        "holding_buckets": {
            "le_15s": int((holds <= 15).sum()),
            "16_30s": int(((holds > 15) & (holds <= 30)).sum()),
            "31_60s": int(((holds > 30) & (holds <= 60)).sum()),
            "gt_60s": int((holds > 60).sum()),
        },
    }


def evaluate_vwap_cross_variants(
    model: L2EventPolicy,
    data: PreparedPolicyData,
    config: ReinforceConfig,
    normalizer: RobustNormalizer,
    market: VwapCrossMarket,
    *,
    start: int,
    end: int,
    crossing_numbers: Iterable[int] = (1, 2),
    event_hazard_threshold: float = 0.02,
    device: str = "auto",
) -> dict[int, tuple[dict[str, object], list[dict[str, object]]]]:
    market.validate(len(data))
    variants = tuple(dict.fromkeys(int(value) for value in crossing_numbers))
    if not variants or any(value not in (1, 2) for value in variants):
        raise ValueError("crossing_numbers must contain one and/or two")
    ranges = list(_segment_ranges(data, start, end))
    hazards = _hazards_by_segment(model, data, normalizer, ranges, device=device)
    result: dict[int, tuple[dict[str, object], list[dict[str, object]]]] = {}
    for crossing_number in variants:
        records: list[dict[str, object]] = []
        for left, right in ranges:
            records.extend(
                ledger_for_vwap_cross_segment(
                    hazards[(left, right)],
                    data,
                    market,
                    left,
                    right,
                    config,
                    crossing_number=crossing_number,
                    event_hazard_threshold=event_hazard_threshold,
                )
            )
        records.sort(key=lambda record: (record["exit_ts_ns"], record["entry_ts_ns"]))
        result[crossing_number] = (
            _metrics(
                records,
                crossing_number=crossing_number,
                event_hazard_threshold=event_hazard_threshold,
                start=start,
                end=end,
            ),
            records,
        )
    return result
