from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import time

import polars as pl

from .config import CTraderCredentials
from .ctrader import choose_symbol


PRICE_SCALE = 100_000.0
SIZE_SCALE = 100.0


@dataclass(frozen=True)
class DepthLevel:
    side: str
    price: float
    size: float


class DepthBook:
    """Causal in-memory reconstruction of cTrader incremental depth quotes."""

    def __init__(self) -> None:
        self._quotes: dict[int, DepthLevel] = {}

    def clear(self) -> None:
        self._quotes.clear()

    def apply_new(self, quote: Any) -> dict[str, Any]:
        quote_id = int(quote.id)
        has_bid = bool(quote.HasField("bid"))
        has_ask = bool(quote.HasField("ask"))
        if has_bid == has_ask:
            raise ValueError("depth quote must contain exactly one side")
        side = "bid" if has_bid else "ask"
        raw_price = int(quote.bid if has_bid else quote.ask)
        level = DepthLevel(
            side=side,
            price=raw_price / PRICE_SCALE,
            size=int(quote.size) / SIZE_SCALE,
        )
        self._quotes[quote_id] = level
        return {
            "quote_id": quote_id,
            "bid": level.price if side == "bid" else 0.0,
            "ask": level.price if side == "ask" else 0.0,
            "size": level.size,
            "type": "new",
            "delete_known": None,
        }

    def apply_delete(self, quote_id: int) -> dict[str, Any]:
        quote_id = int(quote_id)
        previous = self._quotes.pop(quote_id, None)
        return {
            "quote_id": quote_id,
            "bid": previous.price if previous is not None and previous.side == "bid" else 0.0,
            "ask": previous.price if previous is not None and previous.side == "ask" else 0.0,
            "size": previous.size if previous is not None else 0.0,
            "type": "deleted",
            "delete_known": previous is not None,
        }

    def snapshot(self) -> dict[str, Any] | None:
        bids = [level for level in self._quotes.values() if level.side == "bid" and level.size > 0]
        asks = [level for level in self._quotes.values() if level.side == "ask" and level.size > 0]
        if not bids or not asks:
            return None
        best_bid = max(level.price for level in bids)
        best_ask = min(level.price for level in asks)
        top_bids = [level for level in bids if level.price == best_bid]
        top_asks = [level for level in asks if level.price == best_ask]
        bid_size = sum(level.size for level in top_bids)
        ask_size = sum(level.size for level in top_asks)
        denominator = bid_size + ask_size
        microprice = (
            (best_ask * bid_size + best_bid * ask_size) / denominator
            if denominator > 0
            else (best_bid + best_ask) * 0.5
        )
        total_bid_size = sum(level.size for level in bids)
        total_ask_size = sum(level.size for level in asks)
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size_top1": bid_size,
            "ask_size_top1": ask_size,
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "mid": (best_bid + best_ask) * 0.5,
            "microprice": microprice,
            "book_wap": microprice,
            "book_vwap_bid": sum(level.price * level.size for level in bids) / total_bid_size,
            "book_vwap_ask": sum(level.price * level.size for level in asks) / total_ask_size,
        }


class AtomicParquetPartWriter:
    """Append rows as immutable atomic Parquet parts."""

    def __init__(self, directory: Path, *, prefix: str) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = str(prefix)
        existing = sorted(self.directory.glob(f"{self.prefix}_*.parquet"))
        self._index = 1
        if existing:
            self._index = max(int(path.stem.rsplit("_", 1)[1]) for path in existing) + 1
        self._rows: list[dict[str, Any]] = []
        self.rows_written = 0
        self.parts_written = 0

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(row)

    @property
    def pending_rows(self) -> int:
        return len(self._rows)

    def flush(self) -> Path | None:
        if not self._rows:
            return None
        path = self.directory / f"{self.prefix}_{self._index:08d}.parquet"
        temporary = path.with_suffix(path.suffix + ".inprogress")
        pl.DataFrame(self._rows).write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(path)
        count = len(self._rows)
        self._rows.clear()
        self.rows_written += count
        self.parts_written += 1
        self._index += 1
        return path


class CTraderL2Recorder:
    """Live cTrader L2 recorder with causal BBO reconstruction."""

    def __init__(
        self,
        credentials: CTraderCredentials,
        *,
        symbol: str = "XAUUSD",
        output_dir: Path,
        flush_seconds: float = 10.0,
        max_buffer_rows: int = 20_000,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if flush_seconds <= 0:
            raise ValueError("flush_seconds must be positive")
        if max_buffer_rows <= 0:
            raise ValueError("max_buffer_rows must be positive")
        self.credentials = credentials
        self.symbol_name = str(symbol)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_seconds = float(flush_seconds)
        self.max_buffer_rows = int(max_buffer_rows)
        self.progress = progress or (lambda _: None)
        self.raw_writer = AtomicParquetPartWriter(self.output_dir / "raw_parts", prefix="depth")
        self.snapshot_writer = AtomicParquetPartWriter(
            self.output_dir / "snapshot_parts", prefix="snapshot"
        )
        self.book = DepthBook()
        self._client: Any | None = None
        self._reactor: Any | None = None
        self._symbol_id = 0
        self._symbol_digits: int | None = None
        self._event_sequence = 0
        self._connection_segment = 0
        self._started_ns = 0
        self._finished_ns = 0
        self._stopping = False
        self._error: BaseException | None = None

    def record(self, *, duration_seconds: float | None = None) -> dict[str, Any]:
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        try:
            from ctrader_open_api import Client, EndPoints, TcpProtocol
            from twisted.internet import reactor
        except ImportError as exc:
            raise RuntimeError("install ctrader-open-api to record cTrader L2") from exc
        host = (
            EndPoints.PROTOBUF_DEMO_HOST
            if self.credentials.host == "demo"
            else EndPoints.PROTOBUF_LIVE_HOST
        )
        self._reactor = reactor
        self._client = Client(
            host,
            EndPoints.PROTOBUF_PORT,
            TcpProtocol,
            numberOfMessagesToSendPerSecond=50,
        )
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)
        self._started_ns = time.time_ns()
        self._client.startService()
        reactor.callLater(self.flush_seconds, self._periodic_flush)
        reactor.callLater(5.0, self._heartbeat)
        if duration_seconds is not None:
            reactor.callLater(float(duration_seconds), self._finish)
        reactor.run()
        self._flush()
        self._finished_ns = time.time_ns()
        manifest = self._write_manifest()
        if self._error is not None:
            raise RuntimeError(f"cTrader L2 recording failed: {self._error}") from self._error
        if self.snapshot_writer.rows_written == 0:
            raise RuntimeError("cTrader returned no complete L2 book snapshots")
        return manifest

    def _send(self, request: Any, success: Callable[[Any], None], *, timeout: int = 30) -> None:
        assert self._client is not None
        deferred = self._client.send(request, responseTimeoutInSeconds=timeout)
        deferred.addCallback(lambda message: self._extract_and_call(message, success))
        deferred.addErrback(self._on_failure)

    @staticmethod
    def _extract_and_call(message: Any, success: Callable[[Any], None]) -> None:
        from ctrader_open_api import Protobuf

        success(Protobuf.extract(message))

    def _on_connected(self, _: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAApplicationAuthReq

        self._connection_segment += 1
        self.book.clear()
        self.progress({"stage": "connected", "segment": self._connection_segment})
        self._send(
            ProtoOAApplicationAuthReq(
                clientId=self.credentials.client_id,
                clientSecret=self.credentials.client_secret,
            ),
            self._on_application_auth,
        )

    def _on_application_auth(self, _: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAGetAccountListByAccessTokenReq,
        )

        self._send(
            ProtoOAGetAccountListByAccessTokenReq(accessToken=self.credentials.access_token),
            self._on_account_list,
        )

    def _on_account_list(self, response: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAAccountAuthReq

        accounts = list(response.ctidTraderAccount)
        matching = [
            account
            for account in accounts
            if int(account.ctidTraderAccountId) == self.credentials.account_id
        ]
        if not matching:
            raise ValueError("configured cTrader account is not authorised by this token")
        actual_host = "live" if bool(matching[0].isLive) else "demo"
        if actual_host != self.credentials.host:
            raise ValueError(
                f"configured cTrader host is {self.credentials.host!r}, account requires {actual_host!r}"
            )
        self._send(
            ProtoOAAccountAuthReq(
                ctidTraderAccountId=self.credentials.account_id,
                accessToken=self.credentials.access_token,
            ),
            self._on_account_auth,
        )

    def _on_account_auth(self, _: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolsListReq

        self._send(
            ProtoOASymbolsListReq(
                ctidTraderAccountId=self.credentials.account_id,
                includeArchivedSymbols=False,
            ),
            self._on_symbols,
        )

    def _on_symbols(self, response: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolByIdReq

        symbol = choose_symbol(list(response.symbol), self.symbol_name)
        self._symbol_id = int(symbol.symbolId)
        self.symbol_name = str(symbol.symbolName)
        self._send(
            ProtoOASymbolByIdReq(
                ctidTraderAccountId=self.credentials.account_id,
                symbolId=[self._symbol_id],
            ),
            self._on_symbol_details,
        )

    def _on_symbol_details(self, response: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASubscribeDepthQuotesReq

        details = list(response.symbol)
        if details:
            self._symbol_digits = int(details[0].digits)
        self._send(
            ProtoOASubscribeDepthQuotesReq(
                ctidTraderAccountId=self.credentials.account_id,
                symbolId=[self._symbol_id],
            ),
            self._on_subscribed,
        )

    def _on_subscribed(self, _: Any) -> None:
        self.progress(
            {
                "stage": "subscribed",
                "symbol": self.symbol_name,
                "symbol_id": self._symbol_id,
                "digits": self._symbol_digits,
            }
        )

    def _on_message(self, _: Any, message: Any) -> None:
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOADepthEvent

        if int(message.payloadType) != int(ProtoOADepthEvent().payloadType):
            return
        event = Protobuf.extract(message)
        if int(event.ctidTraderAccountId) != self.credentials.account_id:
            return
        if int(event.symbolId) != self._symbol_id:
            return
        receive_ns = time.time_ns()
        monotonic_ns = time.perf_counter_ns()
        self._event_sequence += 1
        common = {
            "timestamp": receive_ns,
            "receive_ns": receive_ns,
            "monotonic_ns": monotonic_ns,
            "event_sequence": self._event_sequence,
            "connection_segment": self._connection_segment,
            "symbol_id": self._symbol_id,
        }
        for quote in event.newQuotes:
            self.raw_writer.append({**common, **self.book.apply_new(quote)})
        for quote_id in event.deletedQuotes:
            self.raw_writer.append({**common, **self.book.apply_delete(int(quote_id))})
        snapshot = self.book.snapshot()
        if snapshot is not None:
            self.snapshot_writer.append(
                {
                    **common,
                    **snapshot,
                    "new_quote_count": len(event.newQuotes),
                    "deleted_quote_count": len(event.deletedQuotes),
                }
            )
        if (
            self.raw_writer.pending_rows >= self.max_buffer_rows
            or self.snapshot_writer.pending_rows >= self.max_buffer_rows
        ):
            self._flush()

    def _heartbeat(self) -> None:
        if self._stopping:
            return
        if self._client is not None and self._client.isConnected:
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoHeartbeatEvent

            connected = self._client.whenConnected(failAfterFailures=1)
            connected.addCallback(
                lambda protocol: protocol.send(ProtoHeartbeatEvent(), instant=True)
            )
            connected.addErrback(lambda _: None)
        if self._reactor is not None and self._reactor.running:
            self._reactor.callLater(5.0, self._heartbeat)

    def _periodic_flush(self) -> None:
        if self._stopping:
            return
        self._flush()
        self.progress(
            {
                "stage": "recording",
                "events": self._event_sequence,
                "raw_rows": self.raw_writer.rows_written,
                "snapshot_rows": self.snapshot_writer.rows_written,
            }
        )
        if self._reactor is not None and self._reactor.running:
            self._reactor.callLater(self.flush_seconds, self._periodic_flush)

    def _flush(self) -> None:
        self.raw_writer.flush()
        self.snapshot_writer.flush()

    def _finish(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._flush()
        if self._client is not None:
            self._client.stopService()
        if self._reactor is not None and self._reactor.running:
            self._reactor.stop()

    def _on_failure(self, failure: Any) -> None:
        self._error = failure.value if hasattr(failure, "value") else RuntimeError(str(failure))
        self._finish()

    def _on_disconnected(self, _: Any, reason: Any) -> None:
        self.progress({"stage": "disconnected", "reason": str(reason)})
        if not self._stopping:
            self.book.clear()

    def _write_manifest(self) -> dict[str, Any]:
        manifest = {
            "source": "cTrader Open API ProtoOASubscribeDepthQuotesReq / ProtoOADepthEvent",
            "host": self.credentials.host,
            "account_id": self.credentials.account_id,
            "symbol": self.symbol_name,
            "symbol_id": self._symbol_id,
            "digits": self._symbol_digits,
            "timestamp_source": "local_receive_wall_clock",
            "timestamp_unit": "ns",
            "started_ns": self._started_ns,
            "finished_ns": self._finished_ns,
            "events": self._event_sequence,
            "connection_segments": self._connection_segment,
            "raw_rows": self.raw_writer.rows_written,
            "snapshot_rows": self.snapshot_writer.rows_written,
            "raw_parts": self.raw_writer.parts_written,
            "snapshot_parts": self.snapshot_writer.parts_written,
            "credential_fields_persisted": False,
        }
        path = self.output_dir / "manifest.json"
        temporary = path.with_suffix(path.suffix + ".inprogress")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return {"output_dir": str(self.output_dir), **manifest}


def load_recorded_l2_snapshots(output_dir: Path) -> pl.DataFrame:
    """Load and validate the causally reconstructed snapshot part stream."""

    directory = Path(output_dir).expanduser().resolve(strict=True)
    paths = sorted((directory / "snapshot_parts").glob("snapshot_*.parquet"))
    if not paths:
        raise ValueError("recording contains no snapshot Parquet parts")
    frame = (
        pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed")
        .unique(subset=["connection_segment", "event_sequence"], keep="last")
        .sort(["connection_segment", "event_sequence"])
        .with_columns(pl.col("receive_ns").cast(pl.Int64).alias("snapshot_ts_ns"))
    )
    required = {
        "snapshot_ts_ns",
        "connection_segment",
        "event_sequence",
        "best_bid",
        "best_ask",
        "bid_size_top1",
        "ask_size_top1",
        "mid",
        "book_wap",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"recorded L2 snapshots are missing columns: {', '.join(missing)}")
    valid = frame.filter(
        (pl.col("best_bid") > 0)
        & (pl.col("best_ask") > pl.col("best_bid"))
        & (pl.col("bid_size_top1") > 0)
        & (pl.col("ask_size_top1") > 0)
    )
    if valid.is_empty():
        raise ValueError("recording contains no valid uncrossed L2 snapshots")
    return valid


def recorded_l2_seconds(
    output_dir: Path,
    *,
    max_quote_age_seconds: int = 2,
) -> pl.DataFrame:
    """Convert live snapshot parts into segment-safe causal one-second bars."""

    from .l2_seconds import aggregate_l2_seconds

    snapshots = load_recorded_l2_snapshots(output_dir)
    frames: list[pl.DataFrame] = []
    segment_offset = 0
    for source_segment in snapshots["connection_segment"].unique(maintain_order=True).to_list():
        part = snapshots.filter(pl.col("connection_segment") == source_segment)
        seconds = aggregate_l2_seconds(part, max_quote_age_seconds=max_quote_age_seconds)
        seconds = seconds.with_columns(
            (pl.col("segment_id") + segment_offset).cast(pl.Int32).alias("segment_id"),
            pl.lit(int(source_segment)).cast(pl.Int32).alias("connection_segment"),
        )
        segment_offset = int(seconds["segment_id"].max()) + 1
        frames.append(seconds)
    return (
        pl.concat(frames, how="vertical_relaxed")
        .sort(["bar_start_ns", "connection_segment"])
        .unique(subset=["bar_start_ns"], keep="last", maintain_order=True)
    )
