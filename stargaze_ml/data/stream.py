from __future__ import annotations

from collections.abc import Iterator
from heapq import heappop, heappush
from itertools import count
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from market_collector.record_log import iter_record_log_tables

from ..contracts import Packet, StreamSpec


PACKET_COLUMNS = (
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "event_id",
    "row_idx",
    "is_snapshot",
    "side",
    "price",
    "quantity",
    "action",
    "order_count",
    "order_id",
    "trade_id",
    "taker_side",
    "mark_price",
    "index_price",
    "oracle_price",
    "open_interest",
    "funding_rate",
    "next_funding_ts_ns",
    "liquidation_side",
)

KIND_COLUMNS = {
    "book": (
        "local_ts_ns", "event_id", "event_type", "is_snapshot", "side", "price",
        "quantity", "action", "order_count",
    ),
    "trade": (
        "local_ts_ns", "event_id", "side", "price", "quantity", "trade_id", "taker_side",
    ),
    "context": (
        "local_ts_ns", "event_id", "mark_price", "index_price", "oracle_price",
        "open_interest", "funding_rate", "next_funding_ts_ns",
    ),
    "liquidation": (
        "local_ts_ns", "event_id", "quantity", "liquidation_side",
    ),
    "l3": (
        "local_ts_ns", "event_id", "event_type", "is_snapshot", "side", "price",
        "quantity", "action", "order_id",
    ),
}


def _numpy_column(array: pa.Array) -> np.ndarray:
    dtype = array.type
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return np.asarray(array.to_pylist(), dtype=object)
    if pa.types.is_boolean(dtype):
        return pc.fill_null(array, False).to_numpy(zero_copy_only=False)
    if pa.types.is_integer(dtype):
        return pc.fill_null(array, -1).to_numpy(zero_copy_only=False)
    if pa.types.is_floating(dtype):
        return pc.fill_null(array, np.nan).to_numpy(zero_copy_only=False)
    return np.asarray(array.to_pylist(), dtype=object)


def _merge_columns(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.concatenate((left[name], right[name])) for name in left}


def _table_batches(stream: StreamSpec, batch_size: int) -> Iterator[pa.RecordBatch]:
    requested = KIND_COLUMNS.get(stream.kind, PACKET_COLUMNS)
    if stream.path.suffix.lower() == ".mrec":
        for table in iter_record_log_tables(stream.path, columns=requested):
            yield from table.to_batches(max_chunksize=batch_size)
        return
    parquet = pq.ParquetFile(stream.path)
    columns = [name for name in requested if name in parquet.schema_arrow.names]
    yield from parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=True)


def iter_packets(
    stream: StreamSpec,
    *,
    end_ts_ns: int | None = None,
    batch_size: int = 131_072,
) -> Iterator[Packet]:
    carry: tuple[tuple[int, int], dict[str, np.ndarray]] | None = None
    done = False
    for batch in _table_batches(stream, batch_size):
        columns = batch.schema.names
        arrays = {name: _numpy_column(batch.column(i)) for i, name in enumerate(columns)}
        local = arrays["local_ts_ns"].astype(np.int64, copy=False)
        if end_ts_ns is not None:
            keep = local <= int(end_ts_ns)
            if not bool(np.any(keep)):
                if local.size and int(local[0]) > int(end_ts_ns):
                    done = True
                    break
                continue
            last = int(np.flatnonzero(keep)[-1]) + 1
            arrays = {name: values[:last] for name, values in arrays.items()}
            local = local[:last]
            done = last < len(keep)
        if local.size == 0:
            if done:
                break
            continue
        events = arrays.get("event_id", np.arange(len(local), dtype=np.int64)).astype(np.int64, copy=False)
        starts = np.r_[0, np.flatnonzero((local[1:] != local[:-1]) | (events[1:] != events[:-1])) + 1]
        ends = np.r_[starts[1:], len(local)]
        groups = [((int(local[start]), int(events[start])), {name: values[start:end] for name, values in arrays.items()}) for start, end in zip(starts, ends, strict=True)]
        if carry is not None:
            if groups and groups[0][0] == carry[0]:
                groups[0] = (carry[0], _merge_columns(carry[1], groups[0][1]))
            else:
                yield Packet(stream, carry[0][0], carry[1])
        carry = groups.pop() if groups else carry
        for key, values in groups:
            yield Packet(stream, key[0], values)
        if done:
            break
    if carry is not None:
        yield Packet(stream, carry[0][0], carry[1])


def iter_merged_packets(streams: tuple[StreamSpec, ...], *, end_ts_ns: int) -> Iterator[Packet]:
    serial = count()
    heap: list[tuple[int, int, Packet, Iterator[Packet]]] = []
    for stream in streams:
        iterator = iter_packets(stream, end_ts_ns=end_ts_ns)
        try:
            packet = next(iterator)
        except StopIteration:
            continue
        event_id = int(packet.columns.get("event_id", np.asarray([-1]))[0])
        heappush(heap, (packet.local_ts_ns, event_id, next(serial), packet, iterator))
    while heap:
        _, _, _, packet, iterator = heappop(heap)
        yield packet
        try:
            following = next(iterator)
        except StopIteration:
            continue
        event_id = int(following.columns.get("event_id", np.asarray([-1]))[0])
        heappush(heap, (following.local_ts_ns, event_id, next(serial), following, iterator))
