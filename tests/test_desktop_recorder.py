from __future__ import annotations

import asyncio
import queue

from desktop_recorder import RecorderService


def test_service_stop_is_safe_after_event_loop_has_closed() -> None:
    service = RecorderService(queue.Queue())
    loop = asyncio.new_event_loop()
    service.loop = loop
    service.stop_event = asyncio.Event()
    loop.close()

    service.stop()
