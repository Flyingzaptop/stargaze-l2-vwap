"""Crash-recoverable compaction of recorder-created Parquet segments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import threading

import pyarrow.parquet as pq

from .records import SCHEMA


_ROTATED_SUFFIX = re.compile(r"_\d{8}_\d{6}(?:_\d+)?$")


@dataclass(frozen=True)
class CompactionResult:
    groups_compacted: int
    source_files: int
    output_files: int
    total_rows: int
    recovered_transactions: int = 0


def logical_parquet_name(path: Path) -> str:
    stem = _ROTATED_SUFFIX.sub("", path.stem)
    return stem + ".parquet"


def _validate(path: Path, expected_rows: int | None = None) -> int:
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(SCHEMA, check_metadata=False):
        raise ValueError(f"schema mismatch in {path}")
    rows = parquet.metadata.num_rows
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"row-count mismatch in {path}: expected {expected_rows}, got {rows}")
    return rows


def _write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _commit_transaction(data_dir: Path, transaction: Path, manifest: dict) -> None:
    target = data_dir / manifest["target"]
    new_file = transaction / "merged.parquet"
    expected_rows = int(manifest["rows"])
    sources_dir = transaction / "sources"
    sources_dir.mkdir(exist_ok=True)

    if not new_file.exists():
        if target.exists() and _validate(target, expected_rows) == expected_rows:
            shutil.rmtree(transaction)
            return
        raise RuntimeError(f"compaction transaction lost both merged output and final target: {transaction}")

    _validate(new_file, expected_rows)
    for name in manifest["sources"]:
        source = data_dir / name
        backup = sources_dir / name
        if source.exists():
            if backup.exists():
                source.unlink()
            else:
                source.replace(backup)

    os.replace(new_file, target)
    _validate(target, expected_rows)
    shutil.rmtree(transaction)


def _recover_transactions(data_dir: Path, recovery_root: Path) -> int:
    recovered = 0
    if not recovery_root.exists():
        return recovered
    for transaction in sorted(path for path in recovery_root.iterdir() if path.is_dir()):
        manifest_path = transaction / "manifest.json"
        if not manifest_path.is_file():
            shutil.rmtree(transaction)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        _commit_transaction(data_dir, transaction, manifest)
        recovered += 1
    return recovered


def compact_parquet_segments(
    data_directory: str | Path,
    recovery_directory: str | Path,
    *,
    cancel_event: threading.Event | None = None,
    batch_size: int = 131_072,
) -> CompactionResult:
    """Merge rotated closed segments into one file per logical stream.

    Original files remain authoritative until a merged file is closed and its
    schema and total row count are verified. Commit then moves originals into
    a same-volume recovery transaction before atomically installing the merge.
    """

    data_dir = Path(data_directory).resolve()
    recovery_root = Path(recovery_directory).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    recovery_root.mkdir(parents=True, exist_ok=True)
    recovered = _recover_transactions(data_dir, recovery_root)

    groups: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob("*.parquet")):
        groups.setdefault(logical_parquet_name(path), []).append(path)

    groups_compacted = 0
    source_files = 0
    total_rows = 0
    for target_name, sources in sorted(groups.items()):
        if cancel_event is not None and cancel_event.is_set():
            break
        if len(sources) < 2:
            total_rows += _validate(sources[0])
            continue

        transaction = recovery_root / (Path(target_name).stem + ".compaction")
        if transaction.exists():
            raise RuntimeError(f"unexpected existing compaction transaction: {transaction}")
        transaction.mkdir(parents=True)
        merged = transaction / "merged.parquet"
        expected_rows = 0
        writer: pq.ParquetWriter | None = None
        try:
            writer = pq.ParquetWriter(
                merged,
                SCHEMA,
                compression="zstd",
                use_dictionary=[
                    "exchange", "market", "symbol", "channel", "event_type",
                    "side", "action", "taker_side",
                ],
                write_statistics=True,
            )
            for source in sources:
                parquet = pq.ParquetFile(source)
                if not parquet.schema_arrow.equals(SCHEMA, check_metadata=False):
                    raise ValueError(f"schema mismatch in {source}")
                expected_rows += parquet.metadata.num_rows
                for batch in parquet.iter_batches(batch_size=batch_size):
                    writer.write_batch(batch)
                del parquet
            writer.close()
            writer = None
            _validate(merged, expected_rows)
            manifest = {
                "format": "market-recorder-compaction-v1",
                "target": target_name,
                "sources": [path.name for path in sources],
                "rows": expected_rows,
            }
            _write_manifest(transaction / "manifest.json", manifest)
            _commit_transaction(data_dir, transaction, manifest)
        except Exception:
            if writer is not None:
                writer.close()
            # Before the manifest exists, no source has moved and the partial
            # transaction can be discarded. Manifested transactions are kept
            # for deterministic recovery on the next startup.
            if not (transaction / "manifest.json").exists():
                shutil.rmtree(transaction, ignore_errors=True)
            raise

        groups_compacted += 1
        source_files += len(sources)
        total_rows += expected_rows

    output_files = len(list(data_dir.glob("*.parquet")))
    return CompactionResult(
        groups_compacted,
        source_files,
        output_files,
        total_rows,
        recovered,
    )
