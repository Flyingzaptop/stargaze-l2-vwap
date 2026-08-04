from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
import json

import polars as pl

from .config import CTraderCredentials


UTC = timezone.utc
MILLISECONDS_PER_MINUTE = 60_000
DEFAULT_CHUNK_DAYS = 7
DEFAULT_MAX_BARS = 14_000


def refresh_ctrader_credentials(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    credentials = CTraderCredentials.from_json(resolved)
    if not credentials.refresh_token:
        raise ValueError("cTrader refresh_token is missing")
    try:
        from ctrader_open_api import Auth
    except ImportError as exc:
        raise RuntimeError("install ctrader-open-api to refresh cTrader credentials") from exc
    response = Auth(credentials.client_id, credentials.client_secret, "").refreshToken(
        credentials.refresh_token
    )
    if response.get("errorCode") or not response.get("accessToken"):
        code = response.get("errorCode") or "missing_access_token"
        description = response.get("description") or "cTrader returned no access token"
        raise RuntimeError(f"cTrader token refresh failed: {code}: {description}")
    raw = json.loads(resolved.read_text(encoding="utf-8-sig"))
    raw["access_token"] = str(response["accessToken"])
    if response.get("refreshToken"):
        raw["refresh_token"] = str(response["refreshToken"])
    temporary = resolved.with_suffix(resolved.suffix + ".inprogress")
    temporary.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    temporary.replace(resolved)
    return {
        "status": "refreshed",
        "host": credentials.host,
        "account_id": credentials.account_id,
        "expires_in": int(response.get("expiresIn", 0)),
        "tokens_printed": False,
    }


@dataclass(frozen=True)
class DecodedTrendbar:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def decode_trendbar(bar: Any, *, digits: int | None = None) -> DecodedTrendbar:
    low_relative = int(bar.low)
    scale = 100_000.0
    values = (
        low_relative + int(bar.deltaOpen),
        low_relative + int(bar.deltaHigh),
        low_relative,
        low_relative + int(bar.deltaClose),
    )
    prices = tuple(value / scale for value in values)
    if digits is not None:
        prices = tuple(round(value, int(digits)) for value in prices)
    return DecodedTrendbar(
        timestamp_ms=int(bar.utcTimestampInMinutes) * MILLISECONDS_PER_MINUTE,
        open=prices[0],
        high=prices[1],
        low=prices[2],
        close=prices[3],
        volume=float(bar.volume),
    )


def _normalise_symbol_name(value: str) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def choose_symbol(symbols: list[Any], requested: str) -> Any:
    target = _normalise_symbol_name(requested)
    exact = [symbol for symbol in symbols if _normalise_symbol_name(symbol.symbolName) == target]
    if exact:
        return exact[0]
    gold_aliases = {"XAUUSD", "GOLD", "GOLDUSD"}
    if target in gold_aliases:
        candidates = [
            symbol
            for symbol in symbols
            if _normalise_symbol_name(symbol.symbolName) in gold_aliases
            or "XAUUSD" in _normalise_symbol_name(symbol.symbolName)
        ]
        if len(candidates) == 1:
            return candidates[0]
    available = ", ".join(sorted(str(symbol.symbolName) for symbol in symbols if "XAU" in str(symbol.symbolName).upper())[:20])
    suffix = f"; XAU candidates: {available}" if available else ""
    raise ValueError(f"cTrader symbol {requested!r} was not found{suffix}")


def _empty_candle_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("ms", time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )


def trendbars_to_frame(bars: list[Any], *, digits: int | None = None) -> pl.DataFrame:
    decoded = [decode_trendbar(bar, digits=digits) for bar in bars]
    if not decoded:
        return _empty_candle_frame()
    frame = pl.DataFrame(
        {
            "timestamp_ms": [bar.timestamp_ms for bar in decoded],
            "open": [bar.open for bar in decoded],
            "high": [bar.high for bar in decoded],
            "low": [bar.low for bar in decoded],
            "close": [bar.close for bar in decoded],
            "volume": [bar.volume for bar in decoded],
        }
    )
    return (
        frame.with_columns(pl.from_epoch("timestamp_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("timestamp"))
        .drop("timestamp_ms")
        .unique(subset=["timestamp"], keep="last")
        .sort("timestamp")
    )


class CTraderMinuteDownloader:
    """Read-only cTrader M1 downloader with resumable calendar chunks."""

    def __init__(
        self,
        credentials: CTraderCredentials,
        *,
        symbol: str = "XAUUSD",
        chunk_days: int = DEFAULT_CHUNK_DAYS,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if chunk_days <= 0 or chunk_days > 9:
            raise ValueError("chunk_days must be in [1, 9] to stay below the historical bar cap")
        self.credentials = credentials
        self.symbol_name = str(symbol)
        self.chunk_days = int(chunk_days)
        self.progress = progress or (lambda _: None)
        self._client: Any | None = None
        self._reactor: Any | None = None
        self._error: BaseException | None = None
        self._output_path: Path | None = None
        self._parts_dir: Path | None = None
        self._ranges: list[tuple[int, int]] = []
        self._range_index = 0
        self._requested_count = DEFAULT_MAX_BARS
        self._symbol_id = 0
        self._symbol_digits: int | None = None

    def download(self, *, start: datetime, end: datetime, output_path: Path) -> dict[str, Any]:
        start = self._as_utc(start)
        end = self._as_utc(end)
        if start >= end:
            raise ValueError("download start must be before end")
        now = datetime.now(UTC)
        if end > now + timedelta(minutes=1):
            raise ValueError("download end cannot be in the future")
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = destination
        self._parts_dir = destination.with_suffix(destination.suffix + ".parts")
        self._parts_dir.mkdir(parents=True, exist_ok=True)
        self._ranges = self._make_ranges(start, end)
        self._range_index = 0

        try:
            from ctrader_open_api import Client, EndPoints, TcpProtocol
            from twisted.internet import reactor
        except ImportError as exc:
            raise RuntimeError("install ctrader-open-api to download cTrader history") from exc
        host = EndPoints.PROTOBUF_DEMO_HOST if self.credentials.host == "demo" else EndPoints.PROTOBUF_LIVE_HOST
        self._reactor = reactor
        self._client = Client(
            host,
            EndPoints.PROTOBUF_PORT,
            TcpProtocol,
            numberOfMessagesToSendPerSecond=5,
        )
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.startService()
        reactor.run()
        if self._error is not None:
            raise RuntimeError(f"cTrader history download failed: {self._error}") from self._error
        return self._consolidate(start=start, end=end)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _make_ranges(self, start: datetime, end: datetime) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        cursor = start
        delta = timedelta(days=self.chunk_days)
        while cursor < end:
            chunk_end = min(cursor + delta, end)
            ranges.append((int(cursor.timestamp() * 1000), int(chunk_end.timestamp() * 1000)))
            cursor = chunk_end
        return ranges

    def _part_path(self, start_ms: int, end_ms: int) -> Path:
        assert self._parts_dir is not None
        return self._parts_dir / f"m1_{start_ms}_{end_ms}.parquet"

    def _send(self, request: Any, success: Callable[[Any], None], *, timeout: int = 45) -> None:
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

        request = ProtoOAApplicationAuthReq(
            clientId=self.credentials.client_id,
            clientSecret=self.credentials.client_secret,
        )
        self.progress({"stage": "ctrader_auth", "status": "application"})
        self._send(request, self._on_application_auth)

    def _on_application_auth(self, _: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetAccountListByAccessTokenReq

        request = ProtoOAGetAccountListByAccessTokenReq(
            accessToken=self.credentials.access_token,
        )
        self._send(request, self._on_account_list)

    def _on_account_list(self, response: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAAccountAuthReq

        accounts = list(response.ctidTraderAccount)
        matching = [
            account
            for account in accounts
            if int(account.ctidTraderAccountId) == self.credentials.account_id
        ]
        if not matching:
            available = [
                {
                    "account_id": int(account.ctidTraderAccountId),
                    "host": "live" if bool(account.isLive) else "demo",
                }
                for account in accounts
            ]
            raise ValueError(
                f"configured cTrader account is not authorised by this access token; "
                f"authorised accounts: {available}"
            )
        actual_host = "live" if bool(matching[0].isLive) else "demo"
        if actual_host != self.credentials.host:
            raise ValueError(
                f"configured cTrader host is {self.credentials.host!r}, "
                f"but account {self.credentials.account_id} requires {actual_host!r}"
            )
        request = ProtoOAAccountAuthReq(
            ctidTraderAccountId=self.credentials.account_id,
            accessToken=self.credentials.access_token,
        )
        self.progress({"stage": "ctrader_auth", "status": "account"})
        self._send(request, self._on_account_auth)

    def _on_account_auth(self, _: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolsListReq

        request = ProtoOASymbolsListReq(
            ctidTraderAccountId=self.credentials.account_id,
            includeArchivedSymbols=False,
        )
        self._send(request, self._on_symbols)

    def _on_symbols(self, response: Any) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolByIdReq

        symbol = choose_symbol(list(response.symbol), self.symbol_name)
        self._symbol_id = int(symbol.symbolId)
        self.symbol_name = str(symbol.symbolName)
        request = ProtoOASymbolByIdReq(
            ctidTraderAccountId=self.credentials.account_id,
            symbolId=[self._symbol_id],
        )
        self._send(request, self._on_symbol_details)

    def _on_symbol_details(self, response: Any) -> None:
        details = list(response.symbol)
        if details:
            self._symbol_digits = int(details[0].digits)
        self.progress(
            {
                "stage": "ctrader_symbol",
                "symbol": self.symbol_name,
                "symbol_id": self._symbol_id,
                "digits": self._symbol_digits,
            }
        )
        self._request_next_range()

    def _request_next_range(self) -> None:
        while self._range_index < len(self._ranges):
            start_ms, end_ms = self._ranges[self._range_index]
            part_path = self._part_path(start_ms, end_ms)
            if part_path.exists():
                self.progress(
                    {
                        "stage": "ctrader_chunk",
                        "index": self._range_index + 1,
                        "total": len(self._ranges),
                        "status": "cached",
                    }
                )
                self._range_index += 1
                continue
            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq
            from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod

            request = ProtoOAGetTrendbarsReq(
                ctidTraderAccountId=self.credentials.account_id,
                fromTimestamp=start_ms,
                toTimestamp=end_ms,
                period=ProtoOATrendbarPeriod.Value("M1"),
                symbolId=self._symbol_id,
                count=min(
                    DEFAULT_MAX_BARS,
                    max(2, int((end_ms - start_ms) // MILLISECONDS_PER_MINUTE) + 2),
                ),
            )
            self._requested_count = int(request.count)
            self.progress(
                {
                    "stage": "ctrader_chunk",
                    "index": self._range_index + 1,
                    "total": len(self._ranges),
                    "status": "request",
                }
            )
            self._send(request, self._on_trendbars, timeout=90)
            return
        self._finish()

    def _on_trendbars(self, response: Any) -> None:
        start_ms, end_ms = self._ranges[self._range_index]
        frame = trendbars_to_frame(list(response.trendbar), digits=self._symbol_digits)
        start_dt = datetime.fromtimestamp(start_ms / 1000.0, UTC)
        end_dt = datetime.fromtimestamp(end_ms / 1000.0, UTC)
        raw_rows = frame.height
        raw_first = frame["timestamp"][0] if raw_rows else None
        frame = frame.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt))
        if raw_rows >= self._requested_count and raw_first is not None and raw_first > start_dt:
            raise RuntimeError(
                f"cTrader response did not reach the requested chunk start for {start_ms}..{end_ms}; "
                "reduce chunk_days to avoid silent truncation"
            )
        part_path = self._part_path(start_ms, end_ms)
        temporary = part_path.with_suffix(part_path.suffix + ".inprogress")
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(part_path)
        self.progress(
            {
                "stage": "ctrader_chunk",
                "index": self._range_index + 1,
                "total": len(self._ranges),
                "status": "saved",
                "rows": frame.height,
            }
        )
        self._range_index += 1
        assert self._reactor is not None
        self._reactor.callLater(0.22, self._request_next_range)

    def _finish(self) -> None:
        if self._client is not None:
            self._client.stopService()
        if self._reactor is not None and self._reactor.running:
            self._reactor.stop()

    def _on_failure(self, failure: Any) -> None:
        self._error = failure.value if hasattr(failure, "value") else RuntimeError(str(failure))
        self._finish()

    def _on_disconnected(self, _: Any, reason: Any) -> None:
        if self._range_index < len(self._ranges) and self._error is None:
            self._error = RuntimeError(str(reason))
            if self._reactor is not None and self._reactor.running:
                self._reactor.stop()

    def _consolidate(self, *, start: datetime, end: datetime) -> dict[str, Any]:
        assert self._parts_dir is not None and self._output_path is not None
        paths = sorted(self._parts_dir.glob("m1_*.parquet"))
        if not paths:
            raise RuntimeError("cTrader returned no historical candle parts")
        frames = [pl.read_parquet(path) for path in paths]
        frame = (
            pl.concat(frames, how="vertical")
            .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
            .unique(subset=["timestamp"], keep="last")
            .sort("timestamp")
        )
        if frame.is_empty():
            raise RuntimeError("cTrader returned no XAUUSD M1 bars for the requested interval")
        temporary = self._output_path.with_suffix(self._output_path.suffix + ".inprogress")
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(self._output_path)
        from ..artifacts import write_json

        metadata = {
            "source": "cTrader Open API ProtoOAGetTrendbarsReq",
            "host": self.credentials.host,
            "account_id": self.credentials.account_id,
            "symbol": self.symbol_name,
            "symbol_id": self._symbol_id,
            "digits": self._symbol_digits,
            "period": "M1",
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "rows": frame.height,
            "first_timestamp": str(frame["timestamp"][0]),
            "last_timestamp": str(frame["timestamp"][-1]),
            "credential_fields_persisted": False,
        }
        write_json(self._output_path.with_suffix(self._output_path.suffix + ".manifest.json"), metadata)
        return {"path": str(self._output_path), **metadata}
