from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .writer import WriterRegistry


@dataclass(frozen=True)
class PackResult:
    path: Path
    manifest_path: Path
    file_count: int
    total_bytes: int
    total_rows: int


class SnapshotManager:
    def __init__(self, data_dir: Path, packs_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.packs_dir = packs_dir.resolve()
        self._lock = asyncio.Lock()

    async def create_pack(self, writers: WriterRegistry) -> PackResult:
        async with self._lock:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            pack_dir = self.packs_dir / f"market_data_{stamp}"
            data_target = pack_dir / "data"
            data_target.mkdir(parents=True, exist_ok=False)
            try:
                entries = await writers.snapshot_to(data_target)
                total_bytes = sum(int(item["bytes"]) for item in entries)
                total_rows = sum(int(item["rows"]) for item in entries)
                manifest = {
                    "format": "market-recorder-pack-v2",
                    "container": "mrec-framed-parquet-v1",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_dir": str(self.data_dir),
                    "file_count": len(entries),
                    "total_bytes": total_bytes,
                    "total_rows": total_rows,
                    "files": entries,
                }
                manifest_path = pack_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                return PackResult(
                    pack_dir,
                    manifest_path,
                    len(entries),
                    total_bytes,
                    total_rows,
                )
            except Exception:
                shutil.rmtree(pack_dir, ignore_errors=True)
                raise
