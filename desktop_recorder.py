from __future__ import annotations

import asyncio
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import pystray

from market_collector.connectors import build_stream_tasks
from market_collector.metrics import metrics
from market_collector.network_setup import (
    configure_windows_firewall,
    expected_firewall_rules,
    network_setup_is_current,
)
from market_collector.sharing import (
    Aria2SeedManager,
    PackSeedManager,
    ServiceState,
    SharingExecutor,
    SyncthingShareManager,
    TorrentArtifact,
    create_v1_torrent,
)
from market_collector.snapshot import PackResult, SnapshotManager
from market_collector.writer import WriterRegistry
from run_collector import apply_secrets


APP_NAME = "Market Recorder"
LOG_FILE = None
INSTANCE_MUTEX_HANDLE = None


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", app_dir()))


def storage_root(config: dict | None = None) -> Path:
    # The desktop build intentionally has one fixed storage contract. A stale
    # config beside the EXE must never redirect high-volume writes to drive C.
    return Path("E:/MarketRecorder/dataset").resolve()


def persistent_tool(name: str, state_dir: Path) -> Path:
    source = resource_dir() / name
    if not source.is_file():
        return source
    target_dir = state_dir / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    source_digest = hashlib.sha256(source.read_bytes()).digest()
    target_digest = None
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        target_digest = hashlib.sha256(target.read_bytes()).digest()
    if target_digest != source_digest:
        temporary = target.with_suffix(target.suffix + ".part")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    return target


def load_json_resource(name: str, local_override: bool = True) -> dict:
    local = app_dir() / name
    if local_override and local.exists():
        return json.loads(local.read_text(encoding="utf-8-sig"))
    bundled = resource_dir() / name
    if bundled.exists():
        return json.loads(bundled.read_text(encoding="utf-8-sig"))
    return {}


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return ""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def age_text(ts: float | None) -> str:
    if not ts:
        return ""
    age = max(0.0, time.time() - ts)
    if age < 60:
        return f"{age:.0f}s ago"
    if age < 3600:
        return f"{age / 60:.1f}m ago"
    return f"{age / 3600:.1f}h ago"


def acquire_single_instance() -> bool:
    global INSTANCE_MUTEX_HANDLE
    if sys.platform != "win32":
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, "Local\\CleanStargazeMarketRecorder")
    if not handle:
        return False
    INSTANCE_MUTEX_HANDLE = handle
    return kernel32.GetLastError() != 183


def _is_windows_admin() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except OSError:
        return False


def _network_setup_state() -> tuple[Path, Path, Path]:
    data_dir = storage_root(load_json_resource("config.json"))
    state_dir = data_dir.parent / "sharing_state"
    return state_dir, state_dir / "bin" / "syncthing.exe", state_dir / "bin" / "aria2c.exe"


def _show_network_setup_error(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} network setup warning: {message}", flush=True)
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            0,
            message,
            f"{APP_NAME} network setup",
            0x30,
        )


def _install_packaged_network_setup(state_dir: Path) -> bool:
    if not _is_windows_admin():
        _show_network_setup_error("Windows Firewall setup helper did not receive administrator rights.")
        return False
    try:
        syncthing_path = persistent_tool("syncthing.exe", state_dir)
        aria2_path = persistent_tool("aria2c.exe", state_dir)
        rules = expected_firewall_rules(sys.executable, syncthing_path, aria2_path)
        marker = configure_windows_firewall(state_dir, rules)
        print(f"{time.strftime('%H:%M:%S')} network setup completed: {marker}", flush=True)
        return True
    except Exception as exc:
        _show_network_setup_error(f"Automatic Windows Firewall setup failed: {exc}")
        return False


def ensure_packaged_network_setup(args: argparse.Namespace) -> bool:
    """Run setup directly or via a short-lived elevated helper."""

    if (
        sys.platform != "win32"
        or not getattr(sys, "frozen", False)
        or args.skip_network_setup
    ):
        return True

    state_dir, syncthing_path, aria2_path = _network_setup_state()
    rules = expected_firewall_rules(sys.executable, syncthing_path, aria2_path)
    if args.network_setup_only:
        _install_packaged_network_setup(state_dir)
        return False
    if network_setup_is_current(state_dir, rules):
        print(f"{time.strftime('%H:%M:%S')} network setup is current", flush=True)
        return True

    if not _is_windows_admin():
        import ctypes

        arguments = [
            item
            for item in sys.argv[1:]
            if item not in ("--network-setup-only", "--skip-network-setup")
        ]
        arguments.append("--network-setup-only")
        parameters = subprocess.list2cmdline(arguments)
        shell_execute = ctypes.windll.shell32.ShellExecuteW
        shell_execute.restype = ctypes.c_void_p
        result = shell_execute(
            None,
            "runas",
            str(sys.executable),
            parameters,
            str(app_dir()),
            1,
        )
        if result is not None and int(result) > 32:
            print(f"{time.strftime('%H:%M:%S')} waiting for elevated network bootstrap", flush=True)
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                if network_setup_is_current(state_dir, rules):
                    print(f"{time.strftime('%H:%M:%S')} elevated network bootstrap completed", flush=True)
                    return True
                time.sleep(0.25)
            _show_network_setup_error(
                "Elevated network setup did not complete within two minutes. Recording will continue without confirmed firewall rules."
            )
            return True
        _show_network_setup_error(
            "Administrator approval was cancelled. Recording will start, but Windows Firewall may block Syncthing and Pack and Send."
        )
        return True

    _install_packaged_network_setup(state_dir)
    return True


class RecorderService:
    def __init__(self, log_queue: queue.Queue[str]) -> None:
        self.log_queue = log_queue
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: asyncio.Event | None = None
        self.running = False
        self.last_error: str | None = None
        self.writers: WriterRegistry | None = None
        self.snapshot_manager: SnapshotManager | None = None
        self.data_dir = storage_root(load_json_resource("config.json"))
        self.packs_dir = self.data_dir.parent / "packs"

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        loop = self.loop
        stop_event = self.stop_event
        if loop is None or stop_event is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(stop_event.set)
        except RuntimeError:
            # A connector startup failure may close the loop just before the UI
            # receives its shutdown callback. Stopping is intentionally safe to retry.
            return

    def create_pack(self) -> concurrent.futures.Future[PackResult]:
        if self.loop is None or self.snapshot_manager is None or self.writers is None:
            future: concurrent.futures.Future[PackResult] = concurrent.futures.Future()
            future.set_exception(RuntimeError("Recorder is not ready yet"))
            return future
        return asyncio.run_coroutine_threadsafe(
            self.snapshot_manager.create_pack(self.writers),
            self.loop,
        )

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self.last_error = repr(exc)
            self.log(f"fatal recorder error: {exc!r}")
        finally:
            self.running = False

    async def _run(self) -> None:
        self.running = True
        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()

        config = load_json_resource("config.json")
        secrets = load_json_resource("secrets.json", local_override=True)
        if not secrets:
            secrets = load_json_resource("secrets.runtime.json", local_override=False)
        if secrets:
            temp_path = app_dir() / ".secrets.loaded.json"
            try:
                temp_path.write_text(json.dumps(secrets), encoding="utf-8")
                apply_secrets(config, temp_path)
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass

        output_dir = storage_root(config)
        config["output_dir"] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = output_dir
        self.packs_dir = output_dir.parent / "packs"
        self.writers = WriterRegistry(
            output_dir=output_dir,
            flush_rows=int(config.get("flush_rows", 5000)),
            flush_seconds=float(config.get("flush_seconds", 5)),
        )
        self.snapshot_manager = SnapshotManager(output_dir, self.packs_dir)

        while not self.stop_event.is_set():
            tasks = build_stream_tasks(config, self.writers, self.stop_event)
            if not tasks:
                raise RuntimeError("No enabled streams in config")
            self.log(f"starting {len(tasks)} stream task(s), output={output_dir}")
            running = [asyncio.create_task(task) for task in tasks]
            flush_task = asyncio.create_task(self.writers.periodic_flush(self.stop_event))
            stop_task = asyncio.create_task(self.stop_event.wait())
            done, pending = await asyncio.wait(
                [*running, flush_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self.stop_event.is_set():
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    if task is not stop_task and not task.done():
                        task.cancel()
                await asyncio.gather(*done, return_exceptions=True)
                break
            errors = [task.exception() for task in done if not task.cancelled() and task.exception()]
            if errors:
                self.log(f"top-level task failed, restarting all streams: {errors[0]!r}")
            else:
                self.log("top-level task ended, restarting all streams")
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.sleep(5)

        await self.writers.close_all()
        self.log("recorder stopped cleanly")

    def log(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {text}"
        self.log_queue.put(line)
        if LOG_FILE is not None:
            print(line, flush=True)


class DesktopApp:
    def __init__(self, self_test_seconds: float | None = None) -> None:
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.geometry("1120x680")
        self.root.minsize(900, 520)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.service = RecorderService(self.log_queue)
        self.icon: pystray.Icon | None = None
        self.pack_future: concurrent.futures.Future[PackResult] | None = None
        self.last_pack: PackResult | None = None
        self.last_torrent: TorrentArtifact | None = None
        self.seed_hashes: set[str] = set()
        self.share_operation = None
        self.stream_operation = None
        self.stream_health_operation = None
        self.stream_started = False
        self.stream_transitioning = False
        self.stream_unhealthy_checks = 0
        self.stream_retry_seconds = 5
        self.closing = False
        state_dir = self.service.data_dir.parent / "sharing_state"
        aria_executable = persistent_tool("aria2c.exe", state_dir)
        syncthing_executable = persistent_tool("syncthing.exe", state_dir)
        self.seed_manager = PackSeedManager(
            Aria2SeedManager(state_dir / "aria2", executable=aria_executable)
        )
        self.syncthing = SyncthingShareManager(
            state_dir,
            executable=syncthing_executable,
            home_directory=state_dir / "syncthing",
            auto_accept_pending=True,
        )
        self._disk_warning_active = False
        self._critical_disk_stop = False
        self._last_disk_check = 0.0
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)
        self.root.after(500, self.refresh)
        if self_test_seconds is not None:
            self.root.after(int(self_test_seconds * 1000), lambda: self.quit_app(prompt=False))

    def _build_ui(self) -> None:
        self.status_var = tk.StringVar(value="Starting")
        self.summary_var = tk.StringVar(value="")
        self.operation_var = tk.StringVar(value="")
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text=APP_NAME, font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        files_frame = ttk.Frame(notebook, padding=8)
        conns_frame = ttk.Frame(notebook, padding=8)
        log_frame = ttk.Frame(notebook, padding=8)
        share_frame = ttk.Frame(notebook, padding=8)
        notebook.add(files_frame, text="Files")
        notebook.add(conns_frame, text="Connections")
        notebook.add(share_frame, text="Sharing")
        notebook.add(log_frame, text="Log")

        self.files_tree = ttk.Treeview(
            files_frame,
            columns=("rows", "rps", "buffer", "size", "last", "path"),
            show="tree headings",
            height=18,
        )
        self.files_tree.heading("#0", text="Stream")
        for col, title, width in (
            ("rows", "Rows", 110),
            ("rps", "Rows/sec 60s", 120),
            ("buffer", "Buffer", 80),
            ("size", "Size", 100),
            ("last", "Last write", 100),
            ("path", "Path", 480),
        ):
            self.files_tree.heading(col, text=title)
            self.files_tree.column(col, width=width, anchor=tk.W)
        self.files_tree.pack(fill=tk.BOTH, expand=True)

        self.conns_tree = ttk.Treeview(
            conns_frame,
            columns=("status", "reconnects", "last", "error"),
            show="tree headings",
            height=18,
        )
        self.conns_tree.heading("#0", text="Connection")
        for col, title, width in (
            ("status", "Status", 120),
            ("reconnects", "Reconnects", 90),
            ("last", "Last msg", 100),
            ("error", "Last error", 520),
        ):
            self.conns_tree.heading(col, text=title)
            self.conns_tree.column(col, width=width, anchor=tk.W)
        self.conns_tree.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=16, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.share_status_var = tk.StringVar(value="Live stream starting automatically...")
        self.share_link_var = tk.StringVar(value="")
        self.pack_link_var = tk.StringVar(value="")
        ttk.Label(share_frame, textvariable=self.share_status_var, font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 10)
        )
        ttk.Label(share_frame, text="Live Syncthing Link").pack(anchor=tk.W)
        link_row = ttk.Frame(share_frame)
        link_row.pack(fill=tk.X, pady=(2, 10))
        ttk.Entry(link_row, textvariable=self.share_link_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(link_row, text="Copy Link", command=self.copy_share_link).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Label(share_frame, text="Last Pack Magnet").pack(anchor=tk.W)
        pack_link_row = ttk.Frame(share_frame)
        pack_link_row.pack(fill=tk.X, pady=(2, 12))
        ttk.Entry(pack_link_row, textvariable=self.pack_link_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(pack_link_row, text="Copy Magnet", command=self.copy_pack_link).pack(
            side=tk.RIGHT, padx=(8, 0)
        )
        actions = ttk.Frame(share_frame)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Pack", command=self.pack_now).pack(side=tk.LEFT)
        self.pack_send_button = ttk.Button(actions, text="Pack and Send", command=self.pack_and_send)
        self.pack_send_button.pack(side=tk.LEFT, padx=(8, 0))
        self.stream_button = ttk.Button(
            actions, text="Restart Stream Data", command=self.restart_stream
        )
        self.stream_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Open Packs", command=self.open_packs_folder).pack(side=tk.LEFT, padx=(8, 0))

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.summary_var).pack(side=tk.LEFT)
        ttk.Label(bottom, textvariable=self.operation_var).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Button(bottom, text="Open Data Folder", command=self.open_data_folder).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(bottom, text="Hide to Tray", command=self.hide_to_tray).pack(side=tk.RIGHT, padx=(6, 0))
        self.pack_button = ttk.Button(bottom, text="Pack", command=self.pack_now)
        self.pack_button.pack(side=tk.RIGHT, padx=(6, 0))

    def start(self) -> None:
        self.service.start()
        self.start_tray()
        self.root.after(1000, self._stream_watchdog)
        self.root.mainloop()

    def refresh(self) -> None:
        snap = metrics.snapshot()
        self.status_var.set("Running" if self.service.running else "Stopped")
        free_text = ""
        try:
            free_text = f" | Free: {human_bytes(shutil.disk_usage(self.service.data_dir).free)}"
        except OSError:
            pass
        self.summary_var.set(
            f"Uptime: {snap['uptime_seconds'] / 60:.1f}m | "
            f"Files: {len(snap['writers'])} | Connections: {len(snap['connections'])}{free_text}"
        )
        self._check_disk_space()

        existing = set(self.files_tree.get_children(""))
        for item in snap["writers"]:
            key = item["key"]
            values = (
                f"{item['rows']:,}",
                f"{item['rows_per_sec_60s']:.1f}",
                f"{item['buffered_rows']:,}",
                human_bytes(item["bytes"]),
                age_text(item["last_write_ts"]),
                item["path"],
            )
            if key in existing:
                self.files_tree.item(key, values=values)
                existing.remove(key)
            else:
                self.files_tree.insert("", tk.END, iid=key, text=key, values=values)
        for key in existing:
            self.files_tree.delete(key)

        existing = set(self.conns_tree.get_children(""))
        for item in snap["connections"]:
            key = item["key"]
            values = (
                item.get("status") or "",
                item.get("reconnects") or 0,
                age_text(item.get("last_message_ts")),
                item.get("last_error") or "",
            )
            if key in existing:
                self.conns_tree.item(key, values=values)
                existing.remove(key)
            else:
                self.conns_tree.insert("", tk.END, iid=key, text=key, values=values)
        for key in existing:
            self.conns_tree.delete(key)

        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, line + "\n")
            self.log_text.see(tk.END)

        self.root.after(1000, self.refresh)

    def start_tray(self) -> None:
        import pystray

        image = Image.new("RGB", (64, 64), (17, 24, 39))
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 12, 52, 52), fill=(34, 197, 94))
        draw.rectangle((29, 20, 35, 44), fill=(255, 255, 255))
        menu = pystray.Menu(
            pystray.MenuItem("Show", lambda: self.root.after(0, self.show_window)),
            pystray.MenuItem("Quit", lambda: self.root.after(0, self.quit_app)),
        )
        self.icon = pystray.Icon(APP_NAME, image, APP_NAME, menu)
        threading.Thread(target=self.icon.run, daemon=True).start()

    def hide_to_tray(self) -> None:
        self.root.withdraw()

    def show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()

    def quit_app(self, prompt: bool = True) -> None:
        if prompt and not messagebox.askokcancel(APP_NAME, "Stop recorder and exit?"):
            return
        self.closing = True
        if self.share_operation is not None and not self.share_operation.done():
            self.share_operation.cancel()
        if self.stream_operation is not None and not self.stream_operation.done():
            self.stream_operation.cancel()
        if self.stream_health_operation is not None and not self.stream_health_operation.done():
            self.stream_health_operation.cancel()
        self.service.stop()
        try:
            self.syncthing.stop(timeout=2.0)
        except Exception as exc:
            self.service.log(f"syncthing stop failed: {exc!r}")
        for info_hash in tuple(self.seed_hashes):
            try:
                self.seed_manager.aria2.stop_seed(info_hash)
            except Exception as exc:
                self.service.log(f"aria2 stop failed for {info_hash}: {exc!r}")
        if self.icon is not None:
            self.icon.stop()
        self.status_var.set("Stopping and closing record files...")
        self.root.after(500, self._destroy_when_stopped)

    def _destroy_when_stopped(self) -> None:
        thread = self.service.thread
        if thread is not None and thread.is_alive():
            self.root.after(500, self._destroy_when_stopped)
            return
        self.root.destroy()

    def open_data_folder(self) -> None:
        path = self.service.data_dir
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(path)
        except OSError as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def open_packs_folder(self) -> None:
        path = self.service.packs_dir
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(path)
        except OSError as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def pack_now(self) -> None:
        if self.pack_future is not None and not self.pack_future.done():
            return
        self.operation_var.set("Packing snapshot...")
        self.pack_button.state(["disabled"])
        self.pack_future = self.service.create_pack()
        self.pack_future.add_done_callback(
            lambda future: self.root.after(0, self._pack_finished, future)
        )

    def _pack_finished(self, future: concurrent.futures.Future[PackResult]) -> None:
        self.pack_button.state(["!disabled"])
        try:
            result = future.result()
        except Exception as exc:
            self.operation_var.set("Pack failed")
            self.service.log(f"pack failed: {exc!r}")
            messagebox.showerror(APP_NAME, f"Pack failed:\n{exc}")
            return
        self.last_pack = result
        self.operation_var.set(
            f"Packed {result.file_count} files / {human_bytes(result.total_bytes)}"
        )
        self.service.log(f"pack ready: {result.path}")

    def pack_and_send(self) -> None:
        if self.share_operation is not None and not self.share_operation.done():
            return
        self.pack_send_button.state(["disabled"])
        self.share_status_var.set("Creating consistent pack...")
        future = self.service.create_pack()
        future.add_done_callback(lambda item: self.root.after(0, self._pack_for_send_ready, item))

    def _pack_for_send_ready(self, future: concurrent.futures.Future[PackResult]) -> None:
        try:
            pack = future.result()
        except Exception as exc:
            self.pack_send_button.state(["!disabled"])
            self.share_status_var.set(f"Pack failed: {exc}")
            self.service.log(f"pack and send failed before torrent: {exc!r}")
            return
        self.last_pack = pack
        torrent_path = pack.path.with_suffix(".torrent")

        def create_and_seed(*, cancel_event):
            artifact = create_v1_torrent(
                pack.path,
                torrent_path,
                trackers=(
                    "udp://tracker.opentrackr.org:1337/announce",
                    "udp://open.stealth.si:80/announce",
                    "udp://tracker.torrent.eu.org:451/announce",
                ),
                comment="Clean Stargaze market data pack",
                cancel_event=cancel_event,
            )
            status = self.seed_manager.start_seed(
                artifact.torrent_path,
                artifact.content_root,
                verify_timeout=90.0,
                cancel_event=cancel_event,
            )
            return artifact, status

        self.share_status_var.set("Hashing pack and starting seed...")
        self.share_operation = SharingExecutor.submit(create_and_seed)
        self.root.after(250, self._poll_pack_send)

    def _poll_pack_send(self) -> None:
        if self.share_operation is None or not self.share_operation.done():
            self.root.after(250, self._poll_pack_send)
            return
        self.pack_send_button.state(["!disabled"])
        try:
            artifact, status = self.share_operation.result()
        except Exception as exc:
            self.share_status_var.set(f"Pack and Send failed: {exc}")
            self.service.log(f"pack and send failed: {exc!r}")
            return
        self.last_torrent = artifact
        self.seed_hashes.add(artifact.info_hash)
        self.pack_link_var.set(artifact.magnet_uri)
        self.copy_pack_link()
        self.share_status_var.set(status.message)
        self.service.log(f"torrent ready: {artifact.torrent_path} {artifact.info_hash} {status.message}")

    def _stream_watchdog(self) -> None:
        if self.closing:
            return
        try:
            if not self.service.running or self.service.writers is None:
                self.share_status_var.set("Waiting for recorder before starting live stream...")
            elif self.stream_transitioning:
                pass
            elif not self.stream_started:
                self._start_stream()
            elif self.stream_health_operation is None:
                self.stream_health_operation = SharingExecutor.submit(self.syncthing.status)
                self.root.after(250, self._poll_stream_health)
        finally:
            self.root.after(15_000, self._stream_watchdog)

    def _poll_stream_health(self) -> None:
        operation = self.stream_health_operation
        if operation is None:
            return
        if not operation.done():
            self.root.after(250, self._poll_stream_health)
            return
        self.stream_health_operation = None
        if self.closing:
            return
        try:
            status = operation.result()
        except Exception as exc:
            self.service.log(f"stream health check failed: {exc!r}")
            self._schedule_stream_retry(f"Health check failed: {exc}")
            return
        if status.state is ServiceState.RUNNING and status.verified:
            self.stream_unhealthy_checks = 0
            self.stream_retry_seconds = 5
            self.share_status_var.set(status.message)
            return
        if status.state is ServiceState.STARTING and status.running:
            self.stream_unhealthy_checks += 1
            self.share_status_var.set(status.message)
            if self.stream_unhealthy_checks < 3:
                return
            self.service.log("syncthing remained unverified for three watchdog checks; restarting")
            self.restart_stream()
            return
        self.service.log(f"syncthing watchdog detected {status.state.value}: {status.message}")
        self._schedule_stream_retry(status.message)

    def _schedule_stream_retry(self, reason: str) -> None:
        if self.closing:
            return
        self.stream_started = False
        self.stream_transitioning = False
        self.stream_button.state(["!disabled"])
        delay = self.stream_retry_seconds
        self.stream_retry_seconds = min(60, delay * 2)
        self.share_status_var.set(f"{reason} Retrying in {delay}s...")
        self.root.after(delay * 1000, self._ensure_stream_active)

    def _ensure_stream_active(self) -> None:
        if (
            self.closing
            or self.stream_started
            or self.stream_transitioning
            or not self.service.running
            or self.service.writers is None
        ):
            return
        self._start_stream()

    def _start_stream(self) -> None:
        if self.closing or self.stream_transitioning or self.stream_started:
            return
        self.stream_transitioning = True
        self.stream_button.state(["disabled"])
        self.share_status_var.set("Sharing append-only record files...")
        self.stream_operation = SharingExecutor.submit(
            self.syncthing.start, self.service.data_dir
        )
        self.root.after(250, self._poll_stream_start)

    def _poll_stream_start(self) -> None:
        operation = self.stream_operation
        if operation is None:
            return
        if not operation.done():
            self.root.after(250, self._poll_stream_start)
            return
        self.stream_operation = None
        self.stream_transitioning = False
        self.stream_button.state(["!disabled"])
        if self.closing:
            return
        try:
            share, status = operation.result()
        except Exception as exc:
            self.service.log(f"stream start failed: {exc!r}")
            self._schedule_stream_retry(f"Stream start failed: {exc}")
            return
        if share is None or status.state not in (ServiceState.RUNNING, ServiceState.STARTING):
            self._schedule_stream_retry(status.message)
            return
        self.stream_started = True
        self.stream_unhealthy_checks = 0
        self.stream_retry_seconds = 5
        self.share_link_var.set(share.connection_info)
        self.copy_share_link()
        self.share_status_var.set(status.message)
        self.service.log(f"live stream started: {share.connection_info}")

    def restart_stream(self) -> None:
        if self.closing or self.stream_transitioning:
            return
        self.stream_started = False
        self.stream_transitioning = True
        self.stream_button.state(["disabled"])
        self.share_status_var.set("Restarting continuous P2P share...")
        self.stream_operation = SharingExecutor.submit(self.syncthing.stop)
        self.root.after(250, self._poll_stream_restart)

    def _poll_stream_restart(self) -> None:
        operation = self.stream_operation
        if operation is None:
            return
        if not operation.done():
            self.root.after(250, self._poll_stream_restart)
            return
        self.stream_operation = None
        self.stream_transitioning = False
        if self.closing:
            return
        try:
            operation.result()
        except Exception as exc:
            self.service.log(f"syncthing restart stop phase failed: {exc!r}")
        self._start_stream()

    def copy_share_link(self) -> None:
        value = self.share_link_var.get().strip()
        if not value:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self.root.update_idletasks()

    def copy_pack_link(self) -> None:
        value = self.pack_link_var.get().strip()
        if not value:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self.root.update_idletasks()

    def _check_disk_space(self) -> None:
        now = time.monotonic()
        if now - self._last_disk_check < 30:
            return
        self._last_disk_check = now
        try:
            free = shutil.disk_usage(self.service.data_dir).free
        except OSError:
            return
        threshold = 10 * 1024**3
        critical = 1024**3
        if free <= critical and not self._critical_disk_stop:
            self._critical_disk_stop = True
            self.operation_var.set("CRITICAL: recorder stopping, disk below 1 GB")
            self.service.log(f"CRITICAL DISK SPACE: {human_bytes(free)} remaining; stopping recorder safely")
            self.service.stop()
        if free <= threshold and not self._disk_warning_active:
            self._disk_warning_active = True
            text = "Андрей, скажи Денису, что мало места. Он поймет"
            self.service.log(f"LOW DISK SPACE: {human_bytes(free)} remaining")
            if self.icon is not None:
                try:
                    self.icon.notify(text, APP_NAME)
                except Exception as exc:
                    self.service.log(f"tray notification failed: {exc!r}")
            if self.root.state() != "withdrawn":
                messagebox.showwarning(APP_NAME, text)
        elif free > threshold + 1024**3:
            self._disk_warning_active = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-seconds", type=float, default=None)
    parser.add_argument("--network-setup-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-network-setup", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    global LOG_FILE
    args = parse_args()
    log_path = app_dir() / "recorder.log"
    try:
        log = log_path.open("a", encoding="utf-8", buffering=1)
        LOG_FILE = log
        sys.stdout = log
        sys.stderr = log
    except OSError:
        pass
    print(f"{time.strftime('%H:%M:%S')} {APP_NAME} booting app_dir={app_dir()} resource_dir={resource_dir()}", flush=True)
    if not ensure_packaged_network_setup(args):
        return
    if not acquire_single_instance():
        print(f"{time.strftime('%H:%M:%S')} another recorder instance is already running", flush=True)
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, "Market Recorder is already running.", APP_NAME, 0x30)
        return
    DesktopApp(self_test_seconds=args.self_test_seconds).start()


if __name__ == "__main__":
    main()
