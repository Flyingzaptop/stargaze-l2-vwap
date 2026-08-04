from __future__ import annotations

import pyarrow as pa


SCHEMA = pa.schema(
    [
        ("exchange", pa.string()),
        ("market", pa.string()),
        ("symbol", pa.string()),
        ("channel", pa.string()),
        ("event_type", pa.string()),
        ("event_id", pa.int64()),
        ("row_idx", pa.int32()),
        ("local_ts_ns", pa.int64()),
        ("exchange_ts_ns", pa.int64()),
        ("engine_ts_ns", pa.int64()),
        ("sequence", pa.string()),
        ("sequence_start", pa.string()),
        ("prev_sequence", pa.string()),
        ("is_snapshot", pa.bool_()),
        ("side", pa.string()),
        ("price", pa.float64()),
        ("quantity", pa.float64()),
        ("action", pa.string()),
        ("order_count", pa.float64()),
        ("order_id", pa.string()),
        ("checksum", pa.int64()),
        ("trade_id", pa.string()),
        ("taker_side", pa.string()),
        ("mark_price", pa.float64()),
        ("index_price", pa.float64()),
        ("oracle_price", pa.float64()),
        ("open_interest", pa.float64()),
        ("funding_rate", pa.float64()),
        ("next_funding_ts_ns", pa.int64()),
        ("liquidation_side", pa.string()),
        ("raw_message", pa.string()),
    ]
)


FIELDS = [field.name for field in SCHEMA]


def normalize_record(record: dict) -> dict:
    return {field: record.get(field) for field in FIELDS}
