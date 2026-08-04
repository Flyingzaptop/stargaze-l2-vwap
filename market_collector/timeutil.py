from __future__ import annotations

from datetime import datetime, timezone
import time


def now_ns() -> int:
    return time.time_ns()


def ms_to_ns(value: int | str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value) * 1_000_000


def us_to_ns(value: int | str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value) * 1_000


def iso_to_ns(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)
