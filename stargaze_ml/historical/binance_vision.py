from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from ..contracts import CausalFrames, VENUES, VENUE_INDEX
from ..data import ExecutionQuotes
from ..features.state import FLAT_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, VENUE_FEATURE_NAMES


BASE_URL = "https://data.binance.vision/data/futures/um/daily"
DAY_SECONDS = 86_400


@dataclass(frozen=True)
class HistoricalDay:
    frames: CausalFrames
    execution: dict[float, ExecutionQuotes]


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date cannot precede start date")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _archive_name(kind: str, day: date) -> str:
    return f"BTCUSDT-{kind}-{day.isoformat()}.zip"


def _archive_url(kind: str, day: date) -> str:
    name = _archive_name(kind, day)
    return f"{BASE_URL}/{kind}/BTCUSDT/{name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(kind: str, day: date, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / _archive_name(kind, day)
    url = _archive_url(kind, day)
    checksum_text = urllib.request.urlopen(f"{url}.CHECKSUM", timeout=60).read().decode("ascii")
    expected = checksum_text.split()[0].lower()
    if target.exists() and _sha256(target) == expected:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=180) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        actual = _sha256(partial)
        if actual != expected:
            raise IOError(f"checksum mismatch for {target.name}: expected {expected}, got {actual}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _csv_batches(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in {path}, found {len(names)}")
        with archive.open(names[0]) as stream:
            reader = pacsv.open_csv(
                pa.input_stream(stream),
                read_options=pacsv.ReadOptions(block_size=32 * 1024 * 1024),
            )
            yield from reader


def _last_indices(groups: np.ndarray) -> np.ndarray:
    if len(groups) == 0:
        return np.empty(0, dtype=np.int64)
    return np.r_[np.flatnonzero(groups[1:] != groups[:-1]), len(groups) - 1]


def _forward_fill(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    positions = np.where(valid, np.arange(len(values)), -1)
    np.maximum.accumulate(positions, out=positions)
    output = np.full_like(values, np.nan)
    usable = positions >= 0
    output[usable] = values[positions[usable]]
    return output


def _book_ticker(path: Path, day_start_ms: int, phases_ms: tuple[int, ...]) -> dict[str, np.ndarray]:
    names_to_store = ("bid", "bid_qty", "ask", "ask_qty")
    end_fields = {name: np.full(DAY_SECONDS, np.nan) for name in (*names_to_store, "event_ms")}
    phase_fields = {
        phase: {name: np.full(DAY_SECONDS, np.nan) for name in names_to_store}
        for phase in phases_ms
    }
    for batch in _csv_batches(path):
        names = batch.schema.names
        event = batch.column(names.index("event_time")).to_numpy(zero_copy_only=False).astype(np.int64)
        second = event // 1000 - day_start_ms // 1000
        in_day = (second >= 0) & (second < DAY_SECONDS)
        if not np.any(in_day):
            continue
        second = second[in_day]
        event = event[in_day]
        columns = {
            "bid": batch.column(names.index("best_bid_price")).to_numpy(zero_copy_only=False)[in_day].astype(np.float64),
            "bid_qty": batch.column(names.index("best_bid_qty")).to_numpy(zero_copy_only=False)[in_day].astype(np.float64),
            "ask": batch.column(names.index("best_ask_price")).to_numpy(zero_copy_only=False)[in_day].astype(np.float64),
            "ask_qty": batch.column(names.index("best_ask_qty")).to_numpy(zero_copy_only=False)[in_day].astype(np.float64),
        }
        chosen = _last_indices(second)
        slots = second[chosen].astype(np.int64)
        for name, values in columns.items():
            end_fields[name][slots] = values[chosen]
        end_fields["event_ms"][slots] = event[chosen]
        remainder = event % 1000
        for phase in phases_ms:
            early = remainder <= int(phase)
            if not np.any(early):
                continue
            early_second = second[early]
            early_chosen = _last_indices(early_second)
            early_slots = early_second[early_chosen].astype(np.int64)
            for name, values in columns.items():
                phase_fields[phase][name][early_slots] = values[early][early_chosen]
    state: dict[str, np.ndarray] = {}
    for name in (*names_to_store, "event_ms"):
        filled = _forward_fill(end_fields[name])
        state[name] = np.r_[np.nan, filled[:-1]]
    result = {f"state_{name}": values for name, values in state.items()}
    for phase in phases_ms:
        for name in names_to_store:
            values = phase_fields[phase][name]
            result[f"exec_{phase}_{name}"] = np.where(np.isfinite(values), values, state[name])
    return result


def _trades(path: Path, day_start_ms: int) -> dict[str, np.ndarray]:
    output = {
        "count": np.zeros(DAY_SECONDS),
        "buy_qty": np.zeros(DAY_SECONDS),
        "sell_qty": np.zeros(DAY_SECONDS),
        "price_qty": np.zeros(DAY_SECONDS),
    }
    for batch in _csv_batches(path):
        names = batch.schema.names
        event = batch.column(names.index("time")).to_numpy(zero_copy_only=False).astype(np.int64)
        second = event // 1000 - day_start_ms // 1000
        keep = (second >= 0) & (second < DAY_SECONDS)
        if not np.any(keep):
            continue
        second = second[keep].astype(np.int64)
        price = batch.column(names.index("price")).to_numpy(zero_copy_only=False)[keep].astype(np.float64)
        qty = batch.column(names.index("qty")).to_numpy(zero_copy_only=False)[keep].astype(np.float64)
        maker = batch.column(names.index("is_buyer_maker")).to_numpy(zero_copy_only=False)[keep].astype(bool)
        np.add.at(output["count"], second, 1.0)
        np.add.at(output["buy_qty"], second, np.where(maker, 0.0, qty))
        np.add.at(output["sell_qty"], second, np.where(maker, qty, 0.0))
        np.add.at(output["price_qty"], second, price * qty)
    return {name: np.r_[0.0, values[:-1]] for name, values in output.items()}


def _timestamp_seconds(column: pa.Array, day_start_s: int) -> np.ndarray:
    values = column.to_numpy(zero_copy_only=False)
    return values.astype("datetime64[s]").astype(np.int64) - int(day_start_s)


def _metrics(path: Path, day_start_s: int) -> dict[str, np.ndarray]:
    fields = {name: np.full(DAY_SECONDS, np.nan) for name in (
        "open_interest", "top_count_ratio", "top_position_ratio", "global_ratio", "taker_ratio"
    )}
    mapping = {
        "open_interest": "sum_open_interest",
        "top_count_ratio": "count_toptrader_long_short_ratio",
        "top_position_ratio": "sum_toptrader_long_short_ratio",
        "global_ratio": "count_long_short_ratio",
        "taker_ratio": "sum_taker_long_short_vol_ratio",
    }
    for batch in _csv_batches(path):
        names = batch.schema.names
        seconds = _timestamp_seconds(batch.column(names.index("create_time")), day_start_s)
        keep = (seconds >= 0) & (seconds < DAY_SECONDS)
        slots = seconds[keep].astype(np.int64)
        for output_name, source_name in mapping.items():
            fields[output_name][slots] = batch.column(names.index(source_name)).to_numpy(zero_copy_only=False)[keep]
    return {name: _forward_fill(values) for name, values in fields.items()}


def _book_depth(path: Path, day_start_s: int) -> dict[str, np.ndarray]:
    fields = {name: np.full(DAY_SECONDS, np.nan) for name in (
        "bid_depth_1pct", "ask_depth_1pct", "bid_depth_5pct", "ask_depth_5pct"
    )}
    percentage_names = {-1.0: "bid_depth_1pct", 1.0: "ask_depth_1pct", -5.0: "bid_depth_5pct", 5.0: "ask_depth_5pct"}
    for batch in _csv_batches(path):
        names = batch.schema.names
        seconds = _timestamp_seconds(batch.column(names.index("timestamp")), day_start_s)
        percentage = batch.column(names.index("percentage")).to_numpy(zero_copy_only=False).astype(np.float64)
        depth = batch.column(names.index("depth")).to_numpy(zero_copy_only=False).astype(np.float64)
        for value, field in percentage_names.items():
            keep = (seconds >= 0) & (seconds < DAY_SECONDS) & np.isclose(percentage, value)
            fields[field][seconds[keep].astype(np.int64)] = depth[keep]
    return {name: _forward_fill(values) for name, values in fields.items()}


def build_historical_day(
    day: date,
    *,
    book_ticker_path: Path,
    trades_path: Path,
    book_depth_path: Path,
    metrics_path: Path,
    phases_ms: tuple[int, ...] = (100, 250, 500),
    notional_usd: float = 1_000.0,
) -> HistoricalDay:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    day_start_s = int(start.timestamp())
    day_start_ms = day_start_s * 1000
    book = _book_ticker(book_ticker_path, day_start_ms, phases_ms)
    trades = _trades(trades_path, day_start_ms)
    metrics = _metrics(metrics_path, day_start_s)
    depth = _book_depth(book_depth_path, day_start_s)
    ts_ns = (day_start_s + np.arange(DAY_SECONDS, dtype=np.int64)) * 1_000_000_000
    venue_x = np.zeros((DAY_SECONDS, len(VENUES), len(VENUE_FEATURE_NAMES)), dtype=np.float32)
    bids = np.full((DAY_SECONDS, len(VENUES)), np.nan)
    asks = np.full_like(bids, np.nan)
    venue = VENUE_INDEX["binance_perpetual"]
    bid, ask = book["state_bid"], book["state_ask"]
    bid_qty, ask_qty = book["state_bid_qty"], book["state_ask_qty"]
    tick_ms = day_start_ms + np.arange(DAY_SECONDS, dtype=np.int64) * 1000
    stale_ms = tick_ms - book["state_event_ms"]
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid < ask) & (stale_ms >= 0) & (stale_ms <= 2_000)
    mid = 0.5 * (bid + ask)
    total_best = np.maximum(bid_qty + ask_qty, 1e-12)
    micro = (ask * bid_qty + bid * ask_qty) / total_best

    def put(name: str, values: np.ndarray) -> None:
        venue_x[:, venue, VENUE_FEATURE_NAMES.index(name)] = np.nan_to_num(values).astype(np.float32)

    put("book_valid", valid.astype(np.float64))
    put("stale_ms", np.clip(stale_ms, 0.0, 60_000.0))
    put("bid", bid)
    put("ask", ask)
    put("mid", mid)
    put("spread_bps", 1e4 * (ask - bid) / mid)
    put("microprice_delta_bps", 1e4 * (micro / mid - 1.0))
    put("best_bid_log_qty", np.log1p(np.maximum(bid_qty, 0.0)))
    put("best_ask_log_qty", np.log1p(np.maximum(ask_qty, 0.0)))
    put("bid_log_depth_1", np.log1p(np.maximum(bid_qty, 0.0)))
    put("ask_log_depth_1", np.log1p(np.maximum(ask_qty, 0.0)))
    put("imbalance_1", (bid_qty - ask_qty) / total_best)
    put("bid_log_depth_1000", np.log1p(np.maximum(depth["bid_depth_1pct"], 0.0)))
    put("ask_log_depth_1000", np.log1p(np.maximum(depth["ask_depth_1pct"], 0.0)))
    deep_total = np.maximum(depth["bid_depth_1pct"] + depth["ask_depth_1pct"], 1e-12)
    put("imbalance_1000", (depth["bid_depth_1pct"] - depth["ask_depth_1pct"]) / deep_total)
    total_qty = trades["buy_qty"] + trades["sell_qty"]
    signed_qty = trades["buy_qty"] - trades["sell_qty"]
    vwap = np.divide(trades["price_qty"], total_qty, out=mid.copy(), where=total_qty > 0.0)
    put("trade_count", np.log1p(trades["count"]))
    put("trade_buy_log_qty", np.log1p(trades["buy_qty"]))
    put("trade_sell_log_qty", np.log1p(trades["sell_qty"]))
    put("trade_signed_log_qty", np.sign(signed_qty) * np.log1p(np.abs(signed_qty)))
    put("trade_vwap_delta_bps", 1e4 * (vwap / mid - 1.0))
    put("open_interest_log", np.log1p(np.maximum(metrics["open_interest"], 0.0)))
    bids[:, venue], asks[:, venue] = bid, ask
    global_x = np.zeros((DAY_SECONDS, len(GLOBAL_FEATURE_NAMES)), dtype=np.float32)

    def global_put(name: str, values: np.ndarray) -> None:
        global_x[:, GLOBAL_FEATURE_NAMES.index(name)] = np.nan_to_num(values).astype(np.float32)

    global_put("consensus_mid", mid)
    global_put("derivative_mid", mid)
    global_put("valid_venue_count", valid.astype(np.float64))
    for lag, name in ((1, "return_1s_bps"), (5, "return_5s_bps"), (10, "return_10s_bps"), (50, "return_50s_bps")):
        values = np.zeros(DAY_SECONDS)
        pair = valid[lag:] & valid[:-lag]
        indices = np.flatnonzero(pair) + lag
        values[indices] = 1e4 * (mid[indices] / mid[indices - lag] - 1.0)
        global_put(name, values)
    x = np.concatenate((venue_x.reshape(DAY_SECONDS, -1), global_x), axis=1).astype(np.float32, copy=False)
    frames = CausalFrames(
        ts_ns, x, venue_x, bids, asks, valid, np.zeros(DAY_SECONDS, dtype=np.int32),
        tuple(FLAT_FEATURE_NAMES), tuple(VENUE_FEATURE_NAMES),
    )
    execution: dict[float, ExecutionQuotes] = {}
    for phase in phases_ms:
        phase_bid, phase_ask = book[f"exec_{phase}_bid"], book[f"exec_{phase}_ask"]
        phase_bid_qty, phase_ask_qty = book[f"exec_{phase}_bid_qty"], book[f"exec_{phase}_ask_qty"]
        required = float(notional_usd) / np.maximum(phase_ask, 1e-12)
        phase_valid = valid & np.isfinite(phase_bid) & np.isfinite(phase_ask) & (phase_bid < phase_ask) & (phase_bid_qty >= required) & (phase_ask_qty >= required)
        execution[float(phase)] = ExecutionQuotes(float(phase), phase_bid, phase_ask, phase_valid)
    return HistoricalDay(frames, execution)


def _concat(days: list[HistoricalDay]) -> HistoricalDay:
    if not days:
        raise ValueError("no historical days were prepared")
    frames = CausalFrames(
        np.concatenate([item.frames.ts_ns for item in days]),
        np.concatenate([item.frames.x for item in days]),
        np.concatenate([item.frames.venue_x for item in days]),
        np.concatenate([item.frames.bid for item in days]),
        np.concatenate([item.frames.ask for item in days]),
        np.concatenate([item.frames.valid for item in days]),
        np.concatenate([np.full(len(item.frames.ts_ns), index, dtype=np.int32) for index, item in enumerate(days)]),
        days[0].frames.feature_names,
        days[0].frames.venue_feature_names,
    )
    execution = {
        phase: ExecutionQuotes(
            phase,
            np.concatenate([item.execution[phase].bid for item in days]),
            np.concatenate([item.execution[phase].ask for item in days]),
            np.concatenate([item.execution[phase].valid for item in days]),
        )
        for phase in days[0].execution
    }
    return HistoricalDay(frames, execution)


def prepare_binance_history(
    *, start_date: str, end_date: str, out_dir: str | Path, keep_archives: bool = False
) -> dict[str, object]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloads = destination / "downloads"
    prepared: list[HistoricalDay] = []
    manifest_days: list[dict[str, object]] = []
    for day in _date_range(date.fromisoformat(start_date), date.fromisoformat(end_date)):
        paths = {kind: download_verified(kind, day, downloads) for kind in ("bookTicker", "trades", "bookDepth", "metrics")}
        item = build_historical_day(
            day,
            book_ticker_path=paths["bookTicker"], trades_path=paths["trades"],
            book_depth_path=paths["bookDepth"], metrics_path=paths["metrics"],
        )
        prepared.append(item)
        manifest_days.append({
            "date": day.isoformat(), "valid_ticks": int(item.frames.valid.sum()),
            "archives": {kind: {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for kind, path in paths.items()},
        })
        print(json.dumps({"stage": "historical_day", **manifest_days[-1]}), flush=True)
        if not keep_archives:
            for path in paths.values():
                path.unlink(missing_ok=True)
    combined = _concat(prepared)
    frames_path = destination / "frames.npz"
    combined.frames.save(frames_path, metadata={"source": "Binance Vision", "days": manifest_days})
    execution_path = destination / "execution_quotes.npz"
    payload: dict[str, np.ndarray] = {}
    for phase, quotes in combined.execution.items():
        key = f"{phase:g}ms"
        payload[f"bid_{key}"], payload[f"ask_{key}"], payload[f"valid_{key}"] = quotes.bid, quotes.ask, quotes.valid
    np.savez_compressed(execution_path, **payload)
    manifest = {
        "source": "https://data.binance.vision", "symbol": "BTCUSDT", "market": "USD-M futures",
        "start_date": start_date, "end_date": end_date, "ticks": len(combined.frames.ts_ns),
        "valid_ticks": int(combined.frames.valid.sum()), "frames": str(frames_path.resolve()),
        "execution_quotes": str(execution_path.resolve()), "days": manifest_days,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = ["HistoricalDay", "build_historical_day", "download_verified", "prepare_binance_history"]
