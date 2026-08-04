from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import pickle

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from market_collector.record_log import (
    FILE_MAGIC,
    FRAME_HEADER,
    FRAME_MAGIC,
    FRAME_TRAILER,
    append_parquet_block,
    initialize_record_log,
    inspect_record_log,
)

from ..contracts import StreamSpec, VENUES
from ..features.state import BookState, DerivativeTick, L3State, MarketState
from .catalog import DatasetCatalog
from .stream import iter_packets


def _frame_payloads(path: Path) -> list[tuple[int, int]]:
    info = inspect_record_log(path)
    result: list[tuple[int, int]] = []
    with path.open("rb") as stream:
        stream.seek(len(FILE_MAGIC))
        while stream.tell() < info.valid_bytes:
            raw = stream.read(FRAME_HEADER.size)
            magic, payload_size, _, _ = FRAME_HEADER.unpack(raw)
            if magic != FRAME_MAGIC:
                raise ValueError(f"invalid frame header in {path}")
            result.append((stream.tell(), int(payload_size)))
            stream.seek(int(payload_size) + FRAME_TRAILER.size, 1)
    return result


def _prefix_sample_hash(path: Path, prefix_bytes: int) -> str:
    sample = min(1024 * 1024, prefix_bytes)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(sample))
        if prefix_bytes > sample:
            stream.seek(prefix_bytes - sample)
            digest.update(stream.read(sample))
    return digest.hexdigest()


def extract_record_log_extension(
    old_path: Path,
    live_path: Path,
    target_path: Path,
    *,
    after_ts_ns: int,
) -> dict[str, int]:
    """Create an MREC containing old overlap rows plus newly appended frames."""

    old_info = inspect_record_log(old_path)
    live_info = inspect_record_log(live_path)
    if live_info.valid_bytes < old_info.valid_bytes:
        raise ValueError(f"live record log is shorter than old snapshot: {live_path}")
    if _prefix_sample_hash(old_path, old_info.valid_bytes) != _prefix_sample_hash(
        live_path, old_info.valid_bytes
    ):
        raise ValueError(f"live record log does not preserve the old committed prefix: {live_path}")

    overlap: list[pa.Table] = []
    with old_path.open("rb") as stream:
        for payload_offset, payload_size in reversed(_frame_payloads(old_path)):
            stream.seek(payload_offset)
            table = pq.read_table(pa.BufferReader(stream.read(payload_size)))
            local = table["local_ts_ns"]
            maximum = int(pc.max(local).as_py())
            if maximum <= int(after_ts_ns):
                break
            filtered = table.filter(pc.greater(local, pa.scalar(int(after_ts_ns), type=pa.int64())))
            if filtered.num_rows:
                overlap.append(filtered)
    overlap.reverse()

    target_path.unlink(missing_ok=True)
    initialize_record_log(target_path)
    with target_path.open("ab") as target:
        for table in overlap:
            append_parquet_block(target, table)
        with live_path.open("rb") as live:
            live.seek(old_info.valid_bytes)
            remaining = live_info.valid_bytes - old_info.valid_bytes
            while remaining:
                chunk = live.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError(f"unexpected EOF while copying extension from {live_path}")
                target.write(chunk)
                remaining -= len(chunk)
            target.flush()
    result = inspect_record_log(target_path, verify_payloads=True)
    return {
        "frames": result.frames,
        "rows": result.rows,
        "valid_bytes": result.valid_bytes,
        "overlap_frames": len(overlap),
        "appended_source_frames": live_info.frames - old_info.frames,
    }


def build_record_log_extension(
    old_root: Path,
    live_root: Path,
    destination: Path,
    *,
    after_ts_ns: int,
) -> dict[str, dict[str, int]]:
    destination.mkdir(parents=True, exist_ok=True)
    output: dict[str, dict[str, int]] = {}
    for old_path in sorted(old_root.glob("*.mrec")):
        live_path = live_root / old_path.name
        if not live_path.exists():
            raise FileNotFoundError(live_path)
        output[old_path.name] = extract_record_log_extension(
            old_path,
            live_path,
            destination / old_path.name,
            after_ts_ns=after_ts_ns,
        )
    return output


def _rebuild_venue(args: tuple[str, tuple[StreamSpec, ...], int, float]) -> tuple[str, BookState, DerivativeTick, L3State | None]:
    venue, streams, end_ts_ns, okx_contract_btc = args
    book = BookState()
    derivative = DerivativeTick()
    l3 = L3State() if venue == "kraken_spot" else None
    scale = okx_contract_btc if venue == "okx_perpetual" else 1.0
    for stream in streams:
        if stream.kind == "book":
            for packet in iter_packets(stream, end_ts_ns=end_ts_ns):
                book.apply(packet, quantity_scale=scale)
        elif stream.kind == "context":
            for packet in iter_packets(stream, end_ts_ns=end_ts_ns):
                derivative.apply_context(packet)
        elif stream.kind == "l3" and l3 is not None:
            for packet in iter_packets(stream, end_ts_ns=end_ts_ns):
                l3.apply(packet)
    if l3 is not None:
        l3.adds = l3.deletes = l3.modifies = 0
        l3.add_qty = l3.delete_qty = 0.0
    return venue, book, derivative, l3


def rebuild_market_state(
    catalog: DatasetCatalog,
    *,
    end_ts_ns: int,
    cadence_ms: int,
    segment_id: int,
    mid_history: list[float],
    okx_contract_btc: float = 0.01,
    workers: int | None = None,
) -> MarketState:
    grouped = {
        venue: tuple(
            stream
            for stream in catalog.streams
            if stream.venue == venue and stream.kind in {"book", "context", "l3"}
        )
        for venue in VENUES
    }
    state = MarketState(okx_contract_btc=okx_contract_btc, cadence_ms=cadence_ms)
    jobs = [
        (venue, streams, int(end_ts_ns), float(okx_contract_btc))
        for venue, streams in grouped.items()
    ]
    with ProcessPoolExecutor(max_workers=workers or min(len(jobs), 9)) as executor:
        futures = {executor.submit(_rebuild_venue, job): job[0] for job in jobs}
        for future in as_completed(futures):
            venue, book, derivative, l3 = future.result()
            state.books[venue] = book
            state.derivatives[venue] = derivative
            if l3 is not None:
                state.l3 = l3
    state.segment_id = int(segment_id)
    state.mid_history = deque((float(value) for value in mid_history), maxlen=10_000)
    return state


def save_market_state(path: Path, state: MarketState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_market_state(path: Path) -> MarketState:
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if not isinstance(state, MarketState):
        raise TypeError("state checkpoint does not contain MarketState")
    return state


__all__ = [
    "build_record_log_extension",
    "extract_record_log_extension",
    "load_market_state",
    "rebuild_market_state",
    "save_market_state",
]
