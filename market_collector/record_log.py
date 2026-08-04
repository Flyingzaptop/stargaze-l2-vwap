"""Crash-safe append-only container for independently compressed Parquet blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import struct
import zlib

import pyarrow as pa
import pyarrow.parquet as pq

from .records import SCHEMA


FILE_MAGIC = b"MRLOG001"
FRAME_MAGIC = b"MRF1"
TRAILER_MAGIC = b"MRE1"
FRAME_HEADER = struct.Struct("<4sQQI")
FRAME_TRAILER = struct.Struct("<4sQI")
MAX_FRAME_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class RecordLogInfo:
    frames: int
    rows: int
    valid_bytes: int
    truncated_bytes: int = 0


class CorruptRecordLog(ValueError):
    pass


def _truncate_or_raise(stream, path: Path, offset: int, size: int, repair: bool, reason: str) -> int:
    if not repair:
        raise CorruptRecordLog(f"{path}: {reason} at byte {offset}")
    stream.seek(offset)
    stream.truncate()
    return size - offset


def inspect_record_log(
    path: str | Path,
    *,
    repair: bool = False,
    verify_payloads: bool = False,
) -> RecordLogInfo:
    """Validate committed frames and optionally discard an incomplete tail."""

    source = Path(path)
    mode = "r+b" if repair else "rb"
    with source.open(mode) as stream:
        size = source.stat().st_size
        if size < len(FILE_MAGIC):
            truncated = _truncate_or_raise(
                stream, source, 0, size, repair, "incomplete file header"
            )
            stream.write(FILE_MAGIC)
            stream.flush()
            return RecordLogInfo(0, 0, len(FILE_MAGIC), truncated)
        if stream.read(len(FILE_MAGIC)) != FILE_MAGIC:
            raise CorruptRecordLog(f"{source}: unsupported file header")

        offset = len(FILE_MAGIC)
        frames = 0
        rows = 0
        while offset < size:
            stream.seek(offset)
            raw_header = stream.read(FRAME_HEADER.size)
            if len(raw_header) != FRAME_HEADER.size:
                truncated = _truncate_or_raise(
                    stream, source, offset, size, repair, "incomplete frame header"
                )
                return RecordLogInfo(frames, rows, offset, truncated)
            magic, payload_size, frame_rows, checksum = FRAME_HEADER.unpack(raw_header)
            if magic != FRAME_MAGIC or payload_size > MAX_FRAME_BYTES:
                truncated = _truncate_or_raise(
                    stream, source, offset, size, repair, "invalid frame header"
                )
                return RecordLogInfo(frames, rows, offset, truncated)

            payload_offset = offset + FRAME_HEADER.size
            trailer_offset = payload_offset + payload_size
            end_offset = trailer_offset + FRAME_TRAILER.size
            if end_offset > size:
                truncated = _truncate_or_raise(
                    stream, source, offset, size, repair, "incomplete frame payload"
                )
                return RecordLogInfo(frames, rows, offset, truncated)

            stream.seek(trailer_offset)
            trailer = stream.read(FRAME_TRAILER.size)
            end_magic, end_size, end_checksum = FRAME_TRAILER.unpack(trailer)
            if (
                end_magic != TRAILER_MAGIC
                or end_size != payload_size
                or end_checksum != checksum
            ):
                truncated = _truncate_or_raise(
                    stream, source, offset, size, repair, "invalid frame commit marker"
                )
                return RecordLogInfo(frames, rows, offset, truncated)

            if verify_payloads:
                stream.seek(payload_offset)
                payload = stream.read(payload_size)
                if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
                    raise CorruptRecordLog(f"{source}: checksum mismatch at byte {offset}")
                parquet = pq.ParquetFile(pa.BufferReader(payload))
                if not parquet.schema_arrow.equals(SCHEMA, check_metadata=False):
                    raise CorruptRecordLog(f"{source}: schema mismatch at byte {offset}")
                if parquet.metadata.num_rows != frame_rows:
                    raise CorruptRecordLog(f"{source}: row-count mismatch at byte {offset}")

            frames += 1
            rows += frame_rows
            offset = end_offset

        return RecordLogInfo(frames, rows, offset)


def initialize_record_log(path: str | Path) -> RecordLogInfo:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(FILE_MAGIC)
            stream.flush()
    return inspect_record_log(target, repair=True)


def encode_parquet_block(table: pa.Table) -> bytes:
    if not table.schema.equals(SCHEMA, check_metadata=False):
        raise ValueError("record-log table schema does not match recorder schema")
    output = pa.BufferOutputStream()
    pq.write_table(
        table,
        output,
        compression="zstd",
        use_dictionary=[
            "exchange",
            "market",
            "symbol",
            "channel",
            "event_type",
            "side",
            "action",
            "taker_side",
        ],
        write_statistics=True,
    )
    return output.getvalue().to_pybytes()


def append_parquet_block(stream, table: pa.Table) -> int:
    payload = encode_parquet_block(table)
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    start = stream.tell()
    try:
        stream.write(FRAME_HEADER.pack(FRAME_MAGIC, len(payload), table.num_rows, checksum))
        stream.write(payload)
        stream.write(FRAME_TRAILER.pack(TRAILER_MAGIC, len(payload), checksum))
        stream.flush()
    except Exception:
        stream.seek(start)
        stream.truncate()
        raise
    return len(payload) + FRAME_HEADER.size + FRAME_TRAILER.size


def copy_record_log(
    source: str | Path,
    target: str | Path,
    *,
    valid_bytes: int | None = None,
    expected_rows: int | None = None,
    expected_frames: int | None = None,
) -> RecordLogInfo:
    """Copy the currently committed prefix to an immutable pack file."""

    source_path = Path(source)
    target_path = Path(target)
    if valid_bytes is None:
        info = inspect_record_log(source_path)
    else:
        if expected_rows is None or expected_frames is None:
            raise ValueError("expected rows and frames are required with a fixed prefix")
        if valid_bytes < len(FILE_MAGIC):
            raise ValueError("record-log prefix is shorter than its file header")
        info = RecordLogInfo(expected_frames, expected_rows, valid_bytes)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(target_path.suffix + ".part")
    with source_path.open("rb") as source_stream, temporary.open("wb") as target_stream:
        remaining = info.valid_bytes
        while remaining:
            chunk = source_stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise OSError(f"unexpected EOF while copying {source_path}")
            target_stream.write(chunk)
            remaining -= len(chunk)
        target_stream.flush()
    shutil.copystat(source_path, temporary)
    temporary.replace(target_path)
    copied = inspect_record_log(target_path, verify_payloads=True)
    if copied.rows != info.rows or copied.frames != info.frames:
        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"record-log snapshot verification failed for {source_path}")
    return copied


def iter_record_log_tables(path: str | Path, *, columns: list[str] | tuple[str, ...] | None = None):
    """Yield verified Parquet tables from a record log in append order."""

    source = Path(path)
    inspect_record_log(source)
    with source.open("rb") as stream:
        stream.seek(len(FILE_MAGIC))
        while True:
            raw_header = stream.read(FRAME_HEADER.size)
            if not raw_header:
                return
            magic, payload_size, frame_rows, checksum = FRAME_HEADER.unpack(raw_header)
            if magic != FRAME_MAGIC:
                raise CorruptRecordLog(f"{source}: invalid frame header")
            payload = stream.read(payload_size)
            trailer = stream.read(FRAME_TRAILER.size)
            end_magic, end_size, end_checksum = FRAME_TRAILER.unpack(trailer)
            if (
                end_magic != TRAILER_MAGIC
                or end_size != payload_size
                or end_checksum != checksum
                or zlib.crc32(payload) & 0xFFFFFFFF != checksum
            ):
                raise CorruptRecordLog(f"{source}: invalid frame payload")
            selected = None if columns is None else list(columns)
            table = pq.read_table(pa.BufferReader(payload), columns=selected, schema=SCHEMA)
            if table.num_rows != frame_rows:
                raise CorruptRecordLog(f"{source}: row-count mismatch")
            yield table
