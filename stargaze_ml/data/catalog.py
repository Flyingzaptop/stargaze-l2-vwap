from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import pyarrow.parquet as pq
import pyarrow as pa

from ..contracts import StreamSpec
from market_collector.record_log import (
    FILE_MAGIC,
    FRAME_HEADER,
    FRAME_MAGIC,
    FRAME_TRAILER,
    inspect_record_log,
)


def _kind(channel: str) -> str:
    normalized = channel.lower()
    if normalized in {"depth", "orderbook", "books", "level2", "book", "l2book"}:
        return "book"
    if normalized == "level3":
        return "l3"
    if normalized in {"trades", "market_trades", "trade"}:
        return "trade"
    if normalized in {"markprice", "ticker", "tickers", "activeassetctx"}:
        return "context"
    if normalized in {"forceorder", "allliquidation"}:
        return "liquidation"
    raise ValueError(f"unsupported channel: {channel}")


def _infer(path: Path) -> StreamSpec:
    name = path.stem.lower()
    if name.startswith("binance_spot_btcusdt_"):
        exchange, market, symbol = "binance", "spot", "BTCUSDT"
        channel = name.removeprefix("binance_spot_btcusdt_")
    elif name.startswith("binance_um_futures_btcusdt_"):
        exchange, market, symbol = "binance", "um_futures", "BTCUSDT"
        channel = name.removeprefix("binance_um_futures_btcusdt_")
    elif name.startswith("bybit_linear_btcusdt_"):
        exchange, market, symbol = "bybit", "linear", "BTCUSDT"
        channel = name.removeprefix("bybit_linear_btcusdt_")
    elif name.startswith("okx_swap_btc_usdt_swap_"):
        exchange, market, symbol = "okx", "swap", "BTC-USDT-SWAP"
        channel = name.removeprefix("okx_swap_btc_usdt_swap_")
    elif name.startswith("coinbase_spot_btc_usd_"):
        exchange, market, symbol = "coinbase", "spot", "BTC-USD"
        channel = name.removeprefix("coinbase_spot_btc_usd_")
    elif name.startswith("kraken_spot_btcusd_"):
        exchange, market, symbol = "kraken", "spot", "BTC/USD"
        channel = name.removeprefix("kraken_spot_btcusd_")
    elif name.startswith("deribit_perpetual_btc_perpetual_"):
        exchange, market, symbol = "deribit", "perpetual", "BTC-PERPETUAL"
        channel = name.removeprefix("deribit_perpetual_btc_perpetual_")
    elif name.startswith("bitfinex_spot_tbtcusd_"):
        exchange, market, symbol = "bitfinex", "spot", "tBTCUSD"
        channel = name.removeprefix("bitfinex_spot_tbtcusd_")
    elif name.startswith("hyperliquid_perpetual_btc_"):
        exchange, market, symbol = "hyperliquid", "perpetual", "BTC"
        channel = name.removeprefix("hyperliquid_perpetual_btc_")
    else:
        raise ValueError(f"cannot infer stream from {path.name}")
    return StreamSpec(exchange, market, symbol, channel, path, _kind(channel))


def parquet_time_bounds(path: Path, column: str = "local_ts_ns") -> tuple[int, int]:
    parquet = pq.ParquetFile(path)
    field_idx = parquet.schema_arrow.get_field_index(column)
    if field_idx < 0:
        raise ValueError(f"{path} has no {column}")
    lows: list[int] = []
    highs: list[int] = []
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(field_idx).statistics
        if stats is not None and stats.has_min_max:
            lows.append(int(stats.min))
            highs.append(int(stats.max))
    if lows:
        return min(lows), max(highs)
    column_data = parquet.read(columns=[column]).column(0)
    return int(column_data[0].as_py()), int(column_data[-1].as_py())


def record_log_time_bounds(path: Path, column: str = "local_ts_ns") -> tuple[int, int] | None:
    info = inspect_record_log(path)
    if info.rows == 0:
        return None
    first_payload: tuple[int, int] | None = None
    last_payload: tuple[int, int] | None = None
    with path.open("rb") as stream:
        stream.seek(len(FILE_MAGIC))
        while stream.tell() < info.valid_bytes:
            raw = stream.read(FRAME_HEADER.size)
            magic, payload_size, _, _ = FRAME_HEADER.unpack(raw)
            if magic != FRAME_MAGIC:
                raise ValueError(f"invalid record-log frame in {path}")
            payload_offset = stream.tell()
            pair = (payload_offset, int(payload_size))
            first_payload = pair if first_payload is None else first_payload
            last_payload = pair
            stream.seek(payload_size + FRAME_TRAILER.size, 1)

        def read_bound(payload: tuple[int, int], take_min: bool) -> int:
            stream.seek(payload[0])
            table = pq.read_table(pa.BufferReader(stream.read(payload[1])), columns=[column])
            values = table.column(0).to_numpy(zero_copy_only=False)
            return int(values.min() if take_min else values.max())

        assert first_payload is not None and last_payload is not None
        return read_bound(first_payload, True), read_bound(last_payload, False)


@dataclass(frozen=True)
class DatasetCatalog:
    root: Path
    streams: tuple[StreamSpec, ...]
    bounds: dict[Path, tuple[int, int]]

    @classmethod
    def discover(cls, root: Path) -> "DatasetCatalog":
        paths = sorted((*root.rglob("*.parquet"), *root.rglob("*.mrec")))
        paths = [path for path in paths if not path.name.startswith("~syncthing~")]
        if not paths:
            raise FileNotFoundError(f"no parquet or mrec files under {root}")
        inferred = tuple(_infer(path) for path in paths)
        bounds: dict[Path, tuple[int, int]] = {}
        streams: list[StreamSpec] = []
        for stream in inferred:
            bound = record_log_time_bounds(stream.path) if stream.path.suffix.lower() == ".mrec" else parquet_time_bounds(stream.path)
            if bound is None:
                continue
            streams.append(stream)
            bounds[stream.path] = bound
        keys = {(stream.venue, stream.kind) for stream in streams}
        required = {("binance_perpetual", "book"), ("binance_perpetual", "trade")}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"missing execution streams: {missing}")
        return cls(root=root, streams=tuple(streams), bounds=bounds)

    @property
    def read_start_ns(self) -> int:
        return min(low for low, _ in self.bounds.values())

    @property
    def common_start_ns(self) -> int:
        required = [self.bounds[s.path] for s in self.streams if s.kind in {"book", "l3"}]
        return max(low for low, _ in required)

    @property
    def common_end_ns(self) -> int:
        required = [self.bounds[s.path] for s in self.streams if s.kind in {"book", "l3"}]
        return min(high for _, high in required)

    def manifest(self) -> dict[str, object]:
        return {
            "root": str(self.root.resolve()),
            "read_start_ns": self.read_start_ns,
            "common_start_ns": self.common_start_ns,
            "common_end_ns": self.common_end_ns,
            "streams": [
                {
                    "exchange": stream.exchange,
                    "market": stream.market,
                    "symbol": stream.symbol,
                    "channel": stream.channel,
                    "kind": stream.kind,
                    "venue": stream.venue,
                    "path": str(stream.path.resolve()),
                    "size_bytes": stream.path.stat().st_size,
                    "first_local_ts_ns": self.bounds[stream.path][0],
                    "last_local_ts_ns": self.bounds[stream.path][1],
                }
                for stream in self.streams
            ],
        }
