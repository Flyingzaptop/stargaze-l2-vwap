from __future__ import annotations

import argparse
import asyncio
import json
import signal
from pathlib import Path

from market_collector.connectors import build_stream_tasks
from market_collector.writer import WriterRegistry


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_secrets(config: dict, secrets_path: Path | None) -> None:
    if secrets_path is None:
        return
    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    for stream in config.get("streams", []):
        exchange = stream.get("exchange")
        exchange_secrets = secrets.get(exchange, {})
        for key, value in exchange_secrets.items():
            stream.setdefault(key, value)


async def main_async(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    apply_secrets(config, args.secrets)
    stop_event = asyncio.Event()

    def request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass
    if args.seconds is not None:
        loop.call_later(args.seconds, request_stop)

    output_dir = Path(config.get("output_dir", "data"))
    writers = WriterRegistry(
        output_dir=output_dir,
        flush_rows=int(config.get("flush_rows", 5000)),
        flush_seconds=float(config.get("flush_seconds", 5)),
    )

    tasks = build_stream_tasks(config, writers, stop_event)
    if not tasks:
        raise SystemExit("No enabled streams in config")

    print(f"starting {len(tasks)} stream task(s); output_dir={output_dir.resolve()}", flush=True)
    running = [asyncio.create_task(task) for task in tasks]
    flush_task = asyncio.create_task(writers.periodic_flush(stop_event))

    try:
        await stop_event.wait()
    finally:
        print("stopping streams...", flush=True)
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        flush_task.cancel()
        await asyncio.gather(flush_task, return_exceptions=True)
        await writers.close_all()
        print("stopped cleanly", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--secrets", type=Path, default=None, help="Optional gitignored API secrets JSON")
    parser.add_argument("--seconds", type=float, default=None, help="Stop automatically after N seconds")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
