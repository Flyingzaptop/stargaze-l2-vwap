"""Immutable pack sharing and continuous local P2P sharing primitives.

BitTorrent metadata is content addressed: changing any shared byte changes the
info hash and therefore the magnet URI.  Closed pack directories are shared as
v1 torrents.  A continuously changing recorder directory is exposed through a
persistent Syncthing identity and a ``sendonly`` folder instead.

The module deliberately does not download or install third-party executables.
It only discovers locally installed qBittorrent and Syncthing instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, build_opener
import uuid
import xml.etree.ElementTree as ET


ProgressCallback = Callable[[int, int, Path], None]
T = TypeVar("T")

PUBLIC_TRACKERS = (
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://open.stealth.si:80/announce",
    "udp://tracker.openbittorrent.com:6969/announce",
)


class SharingCancelled(RuntimeError):
    """Raised when a cancellable sharing operation is stopped by its caller."""


class ServiceState(str, Enum):
    MISSING = "missing"
    NEEDS_CONFIGURATION = "needs_configuration"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class ServiceStatus:
    service: str
    state: ServiceState
    message: str
    executable: Path | None = None
    endpoint: str | None = None
    running: bool = False
    verified: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TorrentArtifact:
    torrent_path: Path
    magnet_uri: str
    info_hash: str
    content_root: Path
    file_count: int
    total_bytes: int
    piece_length: int


@dataclass(frozen=True)
class SyncthingShare:
    folder_path: Path
    folder_id: str
    device_id: str
    home_path: Path

    @property
    def connection_info(self) -> str:
        """Stable share code; this is intentionally not a mutable magnet."""

        return f"syncthing://{self.device_id}/{quote(self.folder_id, safe='')}"


@dataclass(frozen=True)
class PairingResult:
    accepted_device_ids: tuple[str, ...]
    pending_count: int
    errors: tuple[str, ...] = ()


class OperationHandle(Generic[T]):
    """Small polling-friendly background handle suitable for a Tk ``after`` loop."""

    def __init__(self, target: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._cancel_event = threading.Event()
        self._done = threading.Event()
        self._result: T | None = None
        self._error: BaseException | None = None
        kwargs = dict(kwargs)
        kwargs.setdefault("cancel_event", self._cancel_event)

        def run() -> None:
            try:
                self._result = target(*args, **kwargs)
            except BaseException as exc:  # surfaced by result(), never hidden
                self._error = exc
            finally:
                self._done.set()

        self._thread = threading.Thread(target=run, name=f"sharing:{target.__name__}", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()

    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout: float | None = None) -> T:
        if not self._done.wait(timeout):
            raise TimeoutError("sharing operation is still running")
        if self._error is not None:
            raise self._error
        return self._result  # type: ignore[return-value]


class SharingExecutor:
    """Factory for independent, cancellable background operations."""

    @staticmethod
    def submit(target: Callable[..., T], *args: Any, **kwargs: Any) -> OperationHandle[T]:
        return OperationHandle(target, args, kwargs)


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise SharingCancelled("sharing operation cancelled")


def _bencode(value: Any) -> bytes:
    if isinstance(value, bytes):
        return str(len(value)).encode("ascii") + b":" + value
    if isinstance(value, str):
        return _bencode(value.encode("utf-8"))
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii") + b"e"
    if isinstance(value, (list, tuple)):
        return b"l" + b"".join(_bencode(item) for item in value) + b"e"
    if isinstance(value, Mapping):
        items = []
        for key, item in sorted(value.items(), key=lambda pair: _as_bytes(pair[0])):
            items.append(_bencode(_as_bytes(key)))
            items.append(_bencode(item))
        return b"d" + b"".join(items) + b"e"
    raise TypeError(f"cannot bencode {type(value).__name__}")


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("bencode dictionary keys must be str or bytes")


def _bdecode(data: bytes) -> Any:
    """Minimal decoder used to inspect locally generated torrent metadata."""

    def parse(offset: int) -> tuple[Any, int]:
        token = data[offset : offset + 1]
        if token == b"i":
            end = data.index(b"e", offset)
            return int(data[offset + 1 : end]), end + 1
        if token == b"l":
            result = []
            offset += 1
            while data[offset : offset + 1] != b"e":
                item, offset = parse(offset)
                result.append(item)
            return result, offset + 1
        if token == b"d":
            result = {}
            offset += 1
            while data[offset : offset + 1] != b"e":
                key, offset = parse(offset)
                item, offset = parse(offset)
                result[key] = item
            return result, offset + 1
        colon = data.index(b":", offset)
        length = int(data[offset:colon])
        start = colon + 1
        return data[start : start + length], start + length

    value, end = parse(0)
    if end != len(data):
        raise ValueError("trailing data in bencoded payload")
    return value


def _choose_piece_length(total_bytes: int) -> int:
    # Keep metadata reasonably small while retaining useful verification chunks.
    piece_length = 256 * 1024
    while total_bytes > piece_length * 2_000 and piece_length < 16 * 1024 * 1024:
        piece_length *= 2
    return piece_length


def create_v1_torrent(
    pack_directory: str | Path,
    torrent_path: str | Path,
    *,
    trackers: Iterable[str] = PUBLIC_TRACKERS,
    piece_length: int | None = None,
    comment: str | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> TorrentArtifact:
    """Hash a closed directory, atomically write a v1 torrent, and return its magnet.

    Symbolic links are rejected so a pack cannot silently include mutable content
    outside its root.  File size and mtime are checked again after hashing.
    """

    root = Path(pack_directory).resolve(strict=True)
    destination = Path(torrent_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if destination == root or root in destination.parents:
        raise ValueError("torrent_path must be outside pack_directory")

    files: list[tuple[Path, tuple[int, int]]] = []
    for path in root.rglob("*"):
        _check_cancelled(cancel_event)
        if path.is_symlink():
            raise ValueError(f"symbolic links are not supported: {path}")
        if path.is_file():
            stat = path.stat()
            files.append((path, (stat.st_size, stat.st_mtime_ns)))
    files.sort(key=lambda item: item[0].relative_to(root).as_posix().encode("utf-8"))
    if not files:
        raise ValueError("pack_directory contains no files")

    total_bytes = sum(snapshot[0] for _, snapshot in files)
    selected_piece_length = piece_length or _choose_piece_length(total_bytes)
    if selected_piece_length <= 0 or selected_piece_length & (selected_piece_length - 1):
        raise ValueError("piece_length must be a positive power of two")

    pieces: list[bytes] = []
    pending = bytearray()
    completed = 0
    file_entries = []
    for path, before in files:
        _check_cancelled(cancel_event)
        relative_parts = path.relative_to(root).parts
        file_entries.append({b"length": before[0], b"path": [part.encode("utf-8") for part in relative_parts]})
        with path.open("rb") as stream:
            while True:
                _check_cancelled(cancel_event)
                chunk = stream.read(min(1024 * 1024, selected_piece_length - len(pending)))
                if not chunk:
                    break
                pending.extend(chunk)
                completed += len(chunk)
                if len(pending) == selected_piece_length:
                    pieces.append(hashlib.sha1(pending).digest())
                    pending.clear()
                if progress is not None:
                    progress(completed, total_bytes, path)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != before:
            raise RuntimeError(f"pack changed while torrent was being created: {path}")
    if pending:
        pieces.append(hashlib.sha1(pending).digest())

    info = {
        b"name": root.name.encode("utf-8"),
        b"piece length": selected_piece_length,
        b"pieces": b"".join(pieces),
        b"files": file_entries,
    }
    tracker_list = tuple(dict.fromkeys(item.strip() for item in trackers if item.strip()))
    metainfo: dict[bytes, Any] = {
        b"created by": b"clean_stargaze",
        b"info": info,
    }
    if tracker_list:
        metainfo[b"announce"] = tracker_list[0]
        metainfo[b"announce-list"] = [[tracker] for tracker in tracker_list]
    if comment:
        metainfo[b"comment"] = comment

    _check_cancelled(cancel_event)
    payload = _bencode(metainfo)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        _check_cancelled(cancel_event)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    info_hash = hashlib.sha1(_bencode(info)).hexdigest()
    magnet_parts = [
        "magnet:?xt=urn:btih:" + info_hash,
        "dn=" + quote(root.name, safe=""),
        "xl=" + str(total_bytes),
    ]
    magnet_parts.extend("tr=" + quote(tracker, safe="") for tracker in tracker_list)
    return TorrentArtifact(
        torrent_path=destination,
        magnet_uri="&".join(magnet_parts),
        info_hash=info_hash,
        content_root=root,
        file_count=len(files),
        total_bytes=total_bytes,
        piece_length=selected_piece_length,
    )


def _torrent_info_hash(path: Path) -> str:
    metainfo = _bdecode(path.read_bytes())
    return hashlib.sha1(_bencode(metainfo[b"info"])).hexdigest()


def _find_windows_executable(explicit: str | Path | None, names: tuple[str, ...], candidates: Iterable[Path]) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


@dataclass
class _AriaProcess:
    process: subprocess.Popen[Any]
    endpoint: str
    secret: str
    info_hash: str
    torrent_path: Path


class Aria2SeedManager:
    """Run bundled aria2c as a hidden, long-lived pack seeder.

    ``executable`` is intended to receive the resource path resolved by the
    desktop/build layer.  No import from that layer and no runtime download is
    performed here.
    """

    def __init__(
        self,
        state_directory: str | Path,
        *,
        executable: str | Path | None = None,
        popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        opener: Any | None = None,
        timeout: float = 1.5,
    ) -> None:
        self.state_directory = Path(state_directory).resolve()
        self._explicit_executable = executable
        self._popen = popen
        self._opener = opener or build_opener()
        self.timeout = timeout
        self._lock = threading.RLock()
        self._seeds: dict[str, _AriaProcess] = {}

    def discover_executable(self) -> Path | None:
        candidates = [
            self.state_directory.parent / "tools" / "aria2c.exe",
            self.state_directory.parent / "aria2c.exe",
        ]
        return _find_windows_executable(self._explicit_executable, ("aria2c.exe", "aria2c"), candidates)

    @staticmethod
    def _free_local_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            stream.bind(("127.0.0.1", 0))
            return int(stream.getsockname()[1])

    def _rpc(self, seed: _AriaProcess, method: str, params: list[Any] | None = None) -> Any:
        rpc_params = [f"token:{seed.secret}", *(params or [])]
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": "clean-stargaze", "method": method, "params": rpc_params},
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(seed.endpoint + "/jsonrpc", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with self._opener.open(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if "error" in result:
            raise RuntimeError(f"aria2 RPC {method} failed: {result['error']}")
        return result.get("result")

    def _seed_status(self, seed: _AriaProcess) -> ServiceStatus:
        executable = self.discover_executable()
        if seed.process.poll() is not None:
            return ServiceStatus(
                "aria2", ServiceState.ERROR,
                f"aria2c exited with code {seed.process.returncode}; inspect {self.state_directory / 'aria2.log'}",
                executable=executable, endpoint=seed.endpoint,
                details={"info_hash": seed.info_hash},
            )
        try:
            active = self._rpc(seed, "aria2.tellActive", [["gid", "status", "totalLength", "completedLength", "uploadLength", "bittorrent"]])
        except (HTTPError, URLError, OSError, TimeoutError, RuntimeError, json.JSONDecodeError):
            return ServiceStatus(
                "aria2", ServiceState.STARTING,
                "aria2c process is alive; RPC has not confirmed torrent state yet",
                executable=executable, endpoint=seed.endpoint, running=True,
                details={"info_hash": seed.info_hash},
            )
        for item in active or []:
            if "bittorrent" not in item:
                continue
            total = int(item.get("totalLength", 0))
            completed = int(item.get("completedLength", 0))
            if item.get("status") == "active" and total > 0 and completed == total:
                return ServiceStatus(
                    "aria2", ServiceState.RUNNING,
                    f"aria2c verified pack {seed.info_hash} and is seeding",
                    executable=executable, endpoint=seed.endpoint, running=True, verified=True,
                    details={
                        "info_hash": seed.info_hash,
                        "gid": item.get("gid"),
                        "uploaded_bytes": int(item.get("uploadLength", 0)),
                    },
                )
        return ServiceStatus(
            "aria2", ServiceState.STARTING,
            "aria2c is checking the existing pack; seeding is not verified yet",
            executable=executable, endpoint=seed.endpoint, running=True,
            details={"info_hash": seed.info_hash},
        )

    def status(self, info_hash: str | None = None) -> ServiceStatus:
        with self._lock:
            executable = self.discover_executable()
            if executable is None:
                return ServiceStatus(
                    "aria2", ServiceState.MISSING,
                    "aria2c was not found. Pass the bundled resource path as executable=...",
                )
            if info_hash is not None:
                seed = self._seeds.get(info_hash.lower())
                if seed is None:
                    return ServiceStatus("aria2", ServiceState.STOPPED, "This pack is not managed by aria2c", executable=executable)
                return self._seed_status(seed)
            live = [seed for seed in self._seeds.values() if seed.process.poll() is None]
            if not live:
                return ServiceStatus("aria2", ServiceState.READY, "Bundled aria2c is available", executable=executable, verified=True)
            states = [self._seed_status(seed) for seed in live]
            verified = all(item.verified for item in states)
            return ServiceStatus(
                "aria2", ServiceState.RUNNING if verified else ServiceState.STARTING,
                f"aria2c manages {len(live)} pack seed(s); {sum(item.verified for item in states)} verified",
                executable=executable, running=True, verified=verified,
                details={"seeds": [dict(item.details) for item in states]},
            )

    def start_seed(
        self,
        torrent_path: str | Path,
        content_root: str | Path,
        *,
        verify_timeout: float = 30.0,
        cancel_event: threading.Event | None = None,
    ) -> ServiceStatus:
        torrent = Path(torrent_path).resolve(strict=True)
        root = Path(content_root).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        metainfo = _bdecode(torrent.read_bytes())
        torrent_name = metainfo[b"info"][b"name"].decode("utf-8")
        if torrent_name != root.name:
            raise ValueError(f"torrent root {torrent_name!r} does not match content_root {root.name!r}")
        info_hash = hashlib.sha1(_bencode(metainfo[b"info"])).hexdigest()
        with self._lock:
            previous = self._seeds.get(info_hash)
            if previous is not None and previous.process.poll() is None:
                return self._seed_status(previous)
            executable = self.discover_executable()
            if executable is None:
                return self.status(info_hash)
            _check_cancelled(cancel_event)
            self.state_directory.mkdir(parents=True, exist_ok=True)
            port = self._free_local_port()
            secret = uuid.uuid4().hex
            endpoint = f"http://127.0.0.1:{port}"
            command = [
                str(executable),
                f"--dir={root.parent}",
                "--check-integrity=true",
                "--bt-hash-check-seed=true",
                "--bt-seed-unverified=false",
                "--seed-time=5256000",
                "--seed-ratio=0.0",
                "--enable-dht=true",
                "--enable-peer-exchange=true",
                "--bt-enable-lpd=true",
                "--bt-stop-timeout=0",
                "--auto-file-renaming=false",
                "--allow-overwrite=false",
                "--file-allocation=none",
                "--enable-rpc=true",
                "--rpc-listen-all=false",
                f"--rpc-listen-port={port}",
                f"--rpc-secret={secret}",
                f"--dht-file-path={self.state_directory / 'dht.dat'}",
                f"--dht-file-path6={self.state_directory / 'dht6.dat'}",
                f"--log={self.state_directory / 'aria2.log'}",
                "--log-level=notice",
                "--console-log-level=warn",
                str(torrent),
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = self._popen(
                command,
                cwd=str(root.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            seed = _AriaProcess(process, endpoint, secret, info_hash, torrent)
            self._seeds[info_hash] = seed

        deadline = time.monotonic() + verify_timeout
        try:
            while True:
                _check_cancelled(cancel_event)
                status = self._seed_status(seed)
                if status.state in (ServiceState.RUNNING, ServiceState.ERROR) or time.monotonic() >= deadline:
                    return status
                time.sleep(0.2)
        except SharingCancelled:
            if seed.process.poll() is None:
                seed.process.terminate()
                try:
                    seed.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    seed.process.kill()
                    seed.process.wait(timeout=2.0)
            with self._lock:
                self._seeds.pop(info_hash, None)
            raise

    def stop_seed(
        self,
        info_hash: str,
        *,
        timeout: float = 5.0,
        cancel_event: threading.Event | None = None,
    ) -> ServiceStatus:
        with self._lock:
            seed = self._seeds.get(info_hash.lower())
            if seed is None or seed.process.poll() is not None:
                self._seeds.pop(info_hash.lower(), None)
                return ServiceStatus("aria2", ServiceState.STOPPED, "Pack seed is already stopped", executable=self.discover_executable())
            _check_cancelled(cancel_event)
            try:
                self._rpc(seed, "aria2.shutdown")
            except (HTTPError, URLError, OSError, TimeoutError, RuntimeError, json.JSONDecodeError):
                seed.process.terminate()
        deadline = time.monotonic() + timeout
        while seed.process.poll() is None and time.monotonic() < deadline:
            _check_cancelled(cancel_event)
            time.sleep(0.1)
        if seed.process.poll() is None:
            seed.process.terminate()
            try:
                seed.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                seed.process.kill()
                seed.process.wait(timeout=2.0)
        if seed.process.poll() is None:
            return ServiceStatus(
                "aria2", ServiceState.ERROR, "aria2c did not stop; terminate it from Task Manager",
                executable=self.discover_executable(), running=True,
            )
        with self._lock:
            self._seeds.pop(info_hash.lower(), None)
        return ServiceStatus("aria2", ServiceState.STOPPED, "aria2c pack seed stopped", executable=self.discover_executable(), verified=True)


class QBittorrentSeedManager:
    """Discover qBittorrent and add immutable packs through its verified Web API."""

    def __init__(
        self,
        *,
        executable: str | Path | None = None,
        web_api_urls: Iterable[str] = ("http://127.0.0.1:8080",),
        username: str | None = None,
        password: str | None = None,
        opener: Any | None = None,
        timeout: float = 1.5,
    ) -> None:
        self._explicit_executable = executable
        self.web_api_urls = tuple(url.rstrip("/") for url in web_api_urls)
        self.username = username
        self.password = password
        self.timeout = timeout
        self._opener = opener or build_opener()
        self._lock = threading.RLock()
        self._endpoint: str | None = None

    def discover_executable(self) -> Path | None:
        program_files = [Path(os.environ.get(key, "")) for key in ("ProgramFiles", "ProgramFiles(x86)")]
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates = [base / "qBittorrent" / "qbittorrent.exe" for base in program_files if str(base)]
        candidates.append(local / "qBittorrent" / "qbittorrent.exe")
        return _find_windows_executable(self._explicit_executable, ("qbittorrent.exe", "qbittorrent"), candidates)

    def _open(self, request: Request) -> tuple[int, bytes]:
        with self._opener.open(request, timeout=self.timeout) as response:
            return int(response.status), response.read()

    def _login(self, endpoint: str) -> bool:
        if self.username is None or self.password is None:
            return False
        body = urlencode({"username": self.username, "password": self.password}).encode("ascii")
        request = Request(endpoint + "/api/v2/auth/login", data=body, method="POST")
        status, payload = self._open(request)
        return status == 200 and payload.strip() == b"Ok."

    def _probe(self, endpoint: str) -> tuple[bool, str]:
        request = Request(endpoint + "/api/v2/app/version", method="GET")
        try:
            status, payload = self._open(request)
            return status == 200, payload.decode("utf-8", errors="replace").strip()
        except HTTPError as exc:
            if exc.code in (401, 403) and self._login(endpoint):
                status, payload = self._open(request)
                return status == 200, payload.decode("utf-8", errors="replace").strip()
            return False, f"HTTP {exc.code}"
        except (URLError, OSError, TimeoutError) as exc:
            return False, str(exc)

    def status(self) -> ServiceStatus:
        with self._lock:
            executable = self.discover_executable()
            failures = []
            for endpoint in self.web_api_urls:
                ok, detail = self._probe(endpoint)
                if ok:
                    self._endpoint = endpoint
                    return ServiceStatus(
                        "qbittorrent",
                        ServiceState.READY,
                        f"qBittorrent Web API is reachable ({detail})",
                        executable=executable,
                        endpoint=endpoint,
                        running=True,
                        verified=True,
                        details={"version": detail},
                    )
                failures.append(f"{endpoint}: {detail}")
            if executable is not None:
                return ServiceStatus(
                    "qbittorrent",
                    ServiceState.NEEDS_CONFIGURATION,
                    "qBittorrent is installed, but its Web UI/API is unreachable. Enable Web UI, bind it to localhost, and pass its URL/credentials.",
                    executable=executable,
                    details={"probes": failures},
                )
            return ServiceStatus(
                "qbittorrent",
                ServiceState.MISSING,
                "qBittorrent was not found. Install it or provide executable=...; enable its localhost Web UI for managed seeding.",
                details={"probes": failures},
            )

    @staticmethod
    def _multipart(fields: Mapping[str, str], filename: str, file_bytes: bytes) -> tuple[str, bytes]:
        boundary = "----clean-stargaze-" + uuid.uuid4().hex
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode("utf-8"),
                    b"\r\n",
                ]
            )
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="torrents"; filename="{filename}"\r\n'.encode(),
                b"Content-Type: application/x-bittorrent\r\n\r\n",
                file_bytes,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        return boundary, b"".join(chunks)

    def start_seed(
        self,
        torrent_path: str | Path,
        content_root: str | Path,
        *,
        verify_timeout: float = 8.0,
        cancel_event: threading.Event | None = None,
    ) -> ServiceStatus:
        """Submit a torrent and verify qBittorrent knows its info hash."""

        torrent = Path(torrent_path).resolve(strict=True)
        root = Path(content_root).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        _check_cancelled(cancel_event)
        current = self.status()
        if current.state is not ServiceState.READY or current.endpoint is None:
            return current

        fields = {
            "savepath": str(root.parent),
            "root_folder": "true",
            "paused": "false",
            "skip_checking": "false",
            "autoTMM": "false",
        }
        boundary, body = self._multipart(fields, torrent.name, torrent.read_bytes())
        request = Request(
            current.endpoint + "/api/v2/torrents/add",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            status, payload = self._open(request)
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            return ServiceStatus(
                "qbittorrent", ServiceState.ERROR, f"qBittorrent rejected the torrent request: {exc}",
                executable=current.executable, endpoint=current.endpoint, running=True,
            )
        if status != 200 or payload.strip() not in (b"", b"Ok."):
            return ServiceStatus(
                "qbittorrent", ServiceState.ERROR,
                f"qBittorrent add request was not accepted (HTTP {status}: {payload[:200]!r})",
                executable=current.executable, endpoint=current.endpoint, running=True,
            )

        info_hash = _torrent_info_hash(torrent)
        deadline = time.monotonic() + verify_timeout
        while True:
            _check_cancelled(cancel_event)
            query = urlencode({"hashes": info_hash})
            try:
                _, data = self._open(Request(current.endpoint + "/api/v2/torrents/info?" + query, method="GET"))
                torrents = json.loads(data.decode("utf-8"))
            except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                return ServiceStatus(
                    "qbittorrent", ServiceState.ERROR, f"Torrent was submitted but verification failed: {exc}",
                    executable=current.executable, endpoint=current.endpoint, running=True,
                    details={"info_hash": info_hash},
                )
            if torrents:
                torrent_state = str(torrents[0].get("state", "unknown"))
                actively_seeding = torrent_state in {"uploading", "stalledUP", "forcedUP", "queuedUP"}
                message = (
                    f"qBittorrent verified torrent {info_hash} in state {torrent_state}"
                    if actively_seeding
                    else f"qBittorrent accepted torrent {info_hash}; current state is {torrent_state}, not yet verified as seeding"
                )
                return ServiceStatus(
                    "qbittorrent",
                    ServiceState.RUNNING if actively_seeding else ServiceState.STARTING,
                    message,
                    executable=current.executable,
                    endpoint=current.endpoint,
                    running=True,
                    verified=actively_seeding,
                    details={"info_hash": info_hash, "torrent_state": torrent_state},
                )
            if time.monotonic() >= deadline:
                return ServiceStatus(
                    "qbittorrent", ServiceState.ERROR,
                    "qBittorrent returned success but the torrent did not appear in its torrent list",
                    executable=current.executable, endpoint=current.endpoint, running=True,
                    details={"info_hash": info_hash},
                )
            time.sleep(0.2)


class PackSeedManager:
    """Prefer bundled aria2c and use qBittorrent Web API only as fallback."""

    def __init__(self, aria2: Aria2SeedManager, qbittorrent: QBittorrentSeedManager | None = None) -> None:
        self.aria2 = aria2
        self.qbittorrent = qbittorrent or QBittorrentSeedManager()

    def status(self) -> ServiceStatus:
        aria_status = self.aria2.status()
        if aria_status.state is not ServiceState.MISSING:
            return aria_status
        qbit_status = self.qbittorrent.status()
        return ServiceStatus(
            qbit_status.service, qbit_status.state,
            f"Bundled aria2c unavailable; qBittorrent fallback: {qbit_status.message}",
            executable=qbit_status.executable, endpoint=qbit_status.endpoint,
            running=qbit_status.running, verified=qbit_status.verified,
            details={"backend": "qbittorrent", **dict(qbit_status.details)},
        )

    def start_seed(
        self,
        torrent_path: str | Path,
        content_root: str | Path,
        *,
        verify_timeout: float = 30.0,
        cancel_event: threading.Event | None = None,
    ) -> ServiceStatus:
        try:
            aria_status = self.aria2.start_seed(
                torrent_path, content_root, verify_timeout=verify_timeout, cancel_event=cancel_event,
            )
        except OSError as exc:
            aria_status = ServiceStatus("aria2", ServiceState.ERROR, f"aria2c could not start: {exc}")
        if aria_status.state not in (ServiceState.MISSING, ServiceState.ERROR):
            return aria_status
        fallback = self.qbittorrent.start_seed(
            torrent_path, content_root, verify_timeout=verify_timeout, cancel_event=cancel_event,
        )
        return ServiceStatus(
            fallback.service, fallback.state,
            f"aria2c unavailable or failed ({aria_status.message}); qBittorrent fallback: {fallback.message}",
            executable=fallback.executable, endpoint=fallback.endpoint,
            running=fallback.running, verified=fallback.verified,
            details={"backend": "qbittorrent", **dict(fallback.details)},
        )


class SyncthingShareManager:
    """Own a persistent Syncthing identity and one mutable send-only share."""

    def __init__(
        self,
        app_directory: str | Path,
        *,
        executable: str | Path | None = None,
        home_directory: str | Path | None = None,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        opener: Any | None = None,
        timeout: float = 1.5,
        auto_accept_pending: bool = True,
        pairing_interval: float = 5.0,
        gui_address: str = "127.0.0.1:18384",
        listen_addresses: Iterable[str] = (
            "tcp://0.0.0.0:22001",
            "quic://0.0.0.0:22001",
            "dynamic+https://relays.syncthing.net/endpoint",
        ),
    ) -> None:
        self.app_directory = Path(app_directory).resolve()
        self.home_directory = Path(home_directory).resolve() if home_directory else self.app_directory / "syncthing-share"
        self._explicit_executable = executable
        self._run = run
        self._popen = popen
        self._opener = opener or build_opener()
        self.timeout = timeout
        self.auto_accept_pending = auto_accept_pending
        self.pairing_interval = pairing_interval
        self.gui_address = gui_address.strip()
        self.listen_addresses = tuple(listen_addresses)
        if not self.gui_address:
            raise ValueError("gui_address must not be empty")
        if not self.listen_addresses:
            raise ValueError("listen_addresses must not be empty")
        self._lock = threading.RLock()
        self._process: subprocess.Popen[Any] | None = None
        self._pairing_stop = threading.Event()
        self._pairing_thread: threading.Thread | None = None
        self._active_folder_id: str | None = None
        self._last_pairing = PairingResult((), 0)

    @property
    def config_path(self) -> Path:
        return self.home_directory / "config.xml"

    @property
    def pid_path(self) -> Path:
        return self.home_directory / "sharing.pid"

    def discover_executable(self) -> Path | None:
        program_files = [Path(os.environ.get(key, "")) for key in ("ProgramFiles", "ProgramFiles(x86)")]
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates = [
            self.app_directory / "syncthing.exe",
            *(base / "Syncthing" / "syncthing.exe" for base in program_files if str(base)),
            local / "Syncthing" / "syncthing.exe",
        ]
        return _find_windows_executable(self._explicit_executable, ("syncthing.exe", "syncthing"), candidates)

    def _generate_config(self, executable: Path, cancel_event: threading.Event | None) -> None:
        _check_cancelled(cancel_event)
        self.home_directory.mkdir(parents=True, exist_ok=True)
        result = self._run(
            [str(executable), "generate", f"--home={self.home_directory}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        _check_cancelled(cancel_event)
        if result.returncode != 0 or not self.config_path.is_file():
            raise RuntimeError(f"Syncthing could not generate persistent identity/config: {result.stderr or result.stdout}")

    @staticmethod
    def _local_device_id(root: ET.Element) -> str:
        for device in root.findall("device"):
            device_id = device.get("id", "").strip()
            if device_id:
                return device_id
        raise RuntimeError("Syncthing config contains no local device identity")

    def _configure_managed_endpoints(self, root: ET.Element) -> None:
        gui = root.find("gui")
        if gui is None:
            gui = ET.SubElement(root, "gui")
        address = gui.find("address")
        if address is None:
            address = ET.SubElement(gui, "address")
        address.text = self.gui_address

        options = root.find("options")
        if options is None:
            options = ET.SubElement(root, "options")
        for item in list(options.findall("listenAddress")):
            options.remove(item)
        for value in self.listen_addresses:
            ET.SubElement(options, "listenAddress").text = value

    def _managed_endpoints_current(self, root: ET.Element) -> bool:
        gui = root.find("gui")
        address = (gui.findtext("address") if gui is not None else "") or ""
        options = root.find("options")
        listeners = (
            tuple((item.text or "").strip() for item in options.findall("listenAddress"))
            if options is not None
            else ()
        )
        return address.strip() == self.gui_address and listeners == self.listen_addresses

    def _configured_device_id(self) -> str | None:
        if not self.config_path.is_file():
            return None
        try:
            return self._local_device_id(ET.parse(self.config_path).getroot())
        except (OSError, RuntimeError, ET.ParseError):
            return None

    def _api_system_status(self) -> tuple[int, dict[str, Any]]:
        status, payload = self._api_request("/rest/system/status")
        system = json.loads(payload.decode("utf-8"))
        if not isinstance(system, dict):
            raise RuntimeError("Syncthing returned an invalid system-status response")
        return status, system

    def _api_identity_matches_config(self) -> bool:
        expected = self._configured_device_id()
        if not expected:
            return False
        status, system = self._api_system_status()
        return status == 200 and system.get("myID") == expected

    @staticmethod
    def _upsert_folder_config(
        root: ET.Element,
        folder_path: Path,
        folder_id: str,
        label: str,
        remote_device_ids: Iterable[str],
    ) -> str:
        local_id = SyncthingShareManager._local_device_id(root)
        folder = next((item for item in root.findall("folder") if item.get("id") == folder_id), None)
        if folder is None:
            folder = ET.Element("folder")
            first_device = root.find("device")
            insert_at = list(root).index(first_device) if first_device is not None else 0
            root.insert(insert_at, folder)
        folder.attrib.update(
            {
                "id": folder_id,
                "label": label,
                "path": str(folder_path),
                "type": "sendonly",
                "rescanIntervalS": "60",
                "fsWatcherEnabled": "false",
                "fsWatcherDelayS": "10",
            }
        )
        existing_children = {item.get("id"): item for item in folder.findall("device")}
        preserved_remote_ids = [
            device_id
            for device_id in existing_children
            if device_id and device_id != local_id
        ]
        desired_ids = [
            local_id,
            *dict.fromkeys(
                [
                    *preserved_remote_ids,
                    *(item.strip() for item in remote_device_ids if item.strip()),
                ]
            ),
        ]
        for child in list(folder.findall("device")):
            if child.get("id") not in desired_ids:
                folder.remove(child)
        for device_id in desired_ids:
            if device_id not in existing_children:
                ET.SubElement(folder, "device", {"id": device_id})
            if device_id != local_id and not any(item.get("id") == device_id for item in root.findall("device")):
                ET.SubElement(root, "device", {"id": device_id, "name": f"remote-{device_id[:7]}", "compression": "metadata"})
        return local_id

    def prepare(
        self,
        folder_path: str | Path,
        *,
        folder_id: str = "clean-stargaze-live",
        label: str = "Clean Stargaze live data",
        remote_device_ids: Iterable[str] = (),
        cancel_event: threading.Event | None = None,
    ) -> SyncthingShare | ServiceStatus:
        """Create persistent identity/config and configure a send-only folder."""

        with self._lock:
            folder = Path(folder_path).resolve(strict=True)
            if not folder.is_dir():
                raise NotADirectoryError(folder)
            if not folder_id.strip():
                raise ValueError("folder_id must not be empty")
            executable = self.discover_executable()
            if executable is None:
                return ServiceStatus(
                    "syncthing", ServiceState.MISSING,
                    "Syncthing was not found. Install it or place syncthing.exe next to the app; binaries are never downloaded automatically.",
                )
            if self._is_running():
                return ServiceStatus(
                    "syncthing", ServiceState.NEEDS_CONFIGURATION,
                    "Stop the managed Syncthing process before changing its shared folder or remote devices.",
                    executable=executable, running=True,
                )
            if not self.config_path.is_file():
                self._generate_config(executable, cancel_event)
            _check_cancelled(cancel_event)
            tree = ET.parse(self.config_path)
            root = tree.getroot()
            self._configure_managed_endpoints(root)
            local_id = self._upsert_folder_config(root, folder, folder_id, label, remote_device_ids)
            temporary = self.config_path.with_suffix(".xml.tmp")
            tree.write(temporary, encoding="utf-8", xml_declaration=True)
            _check_cancelled(cancel_event)
            os.replace(temporary, self.config_path)
            return SyncthingShare(folder, folder_id, local_id, self.home_directory)

    def _api_config(self) -> tuple[str, str] | None:
        if not self.config_path.is_file():
            return None
        root = ET.parse(self.config_path).getroot()
        gui = root.find("gui")
        if gui is None:
            return None
        address = (gui.findtext("address") or "127.0.0.1:8384").strip()
        api_key = (gui.findtext("apikey") or "").strip()
        if not api_key:
            return None
        if "://" not in address:
            address = "http://" + address
        parsed = urlparse(address)
        host = "127.0.0.1" if parsed.hostname in ("0.0.0.0", "::") else parsed.hostname
        endpoint = f"{parsed.scheme}://{host}:{parsed.port or 8384}"
        return endpoint, api_key

    def _api_request(self, path: str, method: str = "GET", json_body: Any | None = None) -> tuple[int, bytes]:
        config = self._api_config()
        if config is None:
            raise RuntimeError("Syncthing API configuration is unavailable")
        endpoint, api_key = config
        data = None
        headers = {"X-API-Key": api_key}
        if json_body is not None:
            data = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif method in ("POST", "PUT", "PATCH", "DELETE"):
            data = b""
        request = Request(endpoint + path, data=data, headers=headers, method=method)
        with self._opener.open(request, timeout=self.timeout) as response:
            return int(response.status), response.read()

    def accept_pending_devices(
        self,
        folder_id: str | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> PairingResult:
        """Accept every pending device and attach it to the public live folder.

        Anyone who knows the sender Device ID can attempt pairing.  This method
        intentionally accepts all such attempts because the caller designated
        this dataset public-by-code.
        """

        target_folder = folder_id or self._active_folder_id
        if not target_folder:
            raise ValueError("folder_id is required before a share has been started")
        _check_cancelled(cancel_event)
        _, payload = self._api_request("/rest/cluster/pending/devices")
        pending = json.loads(payload.decode("utf-8"))
        if not isinstance(pending, dict):
            raise RuntimeError("Syncthing returned an invalid pending-device response")
        accepted: list[str] = []
        errors: list[str] = []
        for device_id, description in pending.items():
            _check_cancelled(cancel_event)
            try:
                _, template_payload = self._api_request("/rest/config/defaults/device")
                device = json.loads(template_payload.decode("utf-8"))
                device["deviceID"] = device_id
                device["name"] = (description or {}).get("name") or f"public-{device_id[:7]}"
                device["addresses"] = ["dynamic"]
                device["autoAcceptFolders"] = False
                self._api_request("/rest/config/devices", method="POST", json_body=device)

                folder_url = "/rest/config/folders/" + quote(target_folder, safe="")
                _, folder_payload = self._api_request(folder_url)
                folder = json.loads(folder_payload.decode("utf-8"))
                devices = folder.setdefault("devices", [])
                if not any(item.get("deviceID") == device_id for item in devices):
                    devices.append({"deviceID": device_id, "introducedBy": "", "encryptionPassword": ""})
                    self._api_request(folder_url, method="PUT", json_body=folder)
                self._api_request(
                    "/rest/cluster/pending/devices?" + urlencode({"device": device_id}),
                    method="DELETE",
                )
                accepted.append(device_id)
            except (HTTPError, URLError, OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                errors.append(f"{device_id}: {exc}")
        result = PairingResult(tuple(accepted), len(pending), tuple(errors))
        with self._lock:
            self._last_pairing = result
        return result

    def reconcile_folder_devices(
        self,
        folder_id: str | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, ...]:
        """Attach every configured remote device to the public live folder.

        A pending device can disappear as soon as it is accepted even when a
        subsequent folder update is interrupted by a reconnect. Reconciliation
        makes that two-step operation eventually consistent.
        """

        target_folder = folder_id or self._active_folder_id
        if not target_folder:
            raise ValueError("folder_id is required before a share has been started")
        _check_cancelled(cancel_event)
        _, status_payload = self._api_request("/rest/system/status")
        local_id = str(json.loads(status_payload.decode("utf-8")).get("myID") or "")
        _, devices_payload = self._api_request("/rest/config/devices")
        devices = json.loads(devices_payload.decode("utf-8"))
        if not isinstance(devices, list):
            raise RuntimeError("Syncthing returned an invalid configured-device response")
        remote_ids = {
            str(item.get("deviceID") or "")
            for item in devices
            if isinstance(item, dict) and item.get("deviceID") and item.get("deviceID") != local_id
        }

        folder_url = "/rest/config/folders/" + quote(target_folder, safe="")
        _, folder_payload = self._api_request(folder_url)
        folder = json.loads(folder_payload.decode("utf-8"))
        folder_devices = folder.setdefault("devices", [])
        attached = {str(item.get("deviceID") or "") for item in folder_devices if isinstance(item, dict)}
        added = tuple(sorted(remote_ids - attached))
        for device_id in added:
            folder_devices.append(
                {"deviceID": device_id, "introducedBy": "", "encryptionPassword": ""}
            )
        if added:
            _check_cancelled(cancel_event)
            self._api_request(folder_url, method="PUT", json_body=folder)
        return added

    def _start_pairing_worker(self, folder_id: str) -> None:
        if not self.auto_accept_pending:
            return
        with self._lock:
            self._active_folder_id = folder_id
            if self._pairing_thread is not None and self._pairing_thread.is_alive():
                return
            self._pairing_stop.clear()

            def pair_forever() -> None:
                while not self._pairing_stop.is_set():
                    try:
                        self.accept_pending_devices(folder_id, cancel_event=self._pairing_stop)
                        self.reconcile_folder_devices(
                            folder_id, cancel_event=self._pairing_stop
                        )
                    except SharingCancelled:
                        break
                    except (HTTPError, URLError, OSError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                        with self._lock:
                            self._last_pairing = PairingResult((), 0, (str(exc),))
                    self._pairing_stop.wait(self.pairing_interval)

            self._pairing_thread = threading.Thread(target=pair_forever, name="sharing:syncthing-pairing", daemon=True)
            self._pairing_thread.start()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _is_running(self) -> bool:
        owned_process_alive = self._process is not None and self._process.poll() is None
        pid_hint_alive = False
        try:
            pid_hint_alive = self._pid_alive(
                int(self.pid_path.read_text(encoding="ascii").strip())
            )
        except (OSError, ValueError):
            pass

        if self.config_path.is_file():
            try:
                # The API is authoritative only when it serves this manager's
                # certificate. Another Syncthing may legitimately own 8384.
                return self._api_identity_matches_config()
            except (
                HTTPError,
                URLError,
                OSError,
                TimeoutError,
                RuntimeError,
                json.JSONDecodeError,
                ET.ParseError,
            ):
                pass
        return owned_process_alive or pid_hint_alive

    def status(
        self, *, cancel_event: threading.Event | None = None
    ) -> ServiceStatus:
        with self._lock:
            _check_cancelled(cancel_event)
            executable = self.discover_executable()
            if executable is None:
                return ServiceStatus(
                    "syncthing", ServiceState.MISSING,
                    "Syncthing was not found. Install it or place syncthing.exe next to the app.",
                )
            configured = self.config_path.is_file()
            running = self._is_running()
            if running:
                try:
                    _check_cancelled(cancel_event)
                    status, system = self._api_system_status()
                    _check_cancelled(cancel_event)
                    if status == 200:
                        expected_id = self._configured_device_id()
                        actual_id = system.get("myID")
                        if expected_id and actual_id != expected_id:
                            return ServiceStatus(
                                "syncthing",
                                ServiceState.ERROR,
                                "Syncthing API identity collision: another instance owns the configured GUI port",
                                executable=executable,
                                endpoint=self._api_config()[0],
                                running=False,
                                verified=False,
                                details={
                                    "expected_device_id": expected_id,
                                    "actual_device_id": actual_id,
                                },
                            )
                        return ServiceStatus(
                            "syncthing", ServiceState.RUNNING, "Managed Syncthing identity and local API are verified",
                            executable=executable, endpoint=self._api_config()[0], running=True, verified=True,
                            details={
                                "device_id": system.get("myID"),
                                "pairing": "auto_accept_all_pending" if self.auto_accept_pending else "manual_remote_device_required",
                                "last_pairing_pending": self._last_pairing.pending_count,
                                "last_pairing_accepted": list(self._last_pairing.accepted_device_ids),
                                "last_pairing_errors": list(self._last_pairing.errors),
                            },
                        )
                except (HTTPError, URLError, OSError, TimeoutError, RuntimeError, json.JSONDecodeError):
                    return ServiceStatus(
                        "syncthing", ServiceState.STARTING,
                        "Syncthing process exists, but the local API has not confirmed readiness",
                        executable=executable, running=True, verified=False,
                    )
            if configured:
                try:
                    status, system = self._api_system_status()
                    expected_id = self._configured_device_id()
                    actual_id = system.get("myID")
                    if status == 200 and expected_id and actual_id != expected_id:
                        return ServiceStatus(
                            "syncthing",
                            ServiceState.ERROR,
                            "Syncthing API identity collision: recorder will migrate to its private port",
                            executable=executable,
                            endpoint=self._api_config()[0],
                            running=False,
                            verified=False,
                            details={
                                "expected_device_id": expected_id,
                                "actual_device_id": actual_id,
                            },
                        )
                except (
                    HTTPError,
                    URLError,
                    OSError,
                    TimeoutError,
                    RuntimeError,
                    json.JSONDecodeError,
                    ET.ParseError,
                ):
                    pass
            state = ServiceState.STOPPED if configured else ServiceState.NEEDS_CONFIGURATION
            message = "Syncthing share is configured but stopped" if configured else "Syncthing is installed; call prepare() or start() to create its persistent share"
            return ServiceStatus("syncthing", state, message, executable=executable)

    def start(
        self,
        folder_path: str | Path,
        *,
        folder_id: str = "clean-stargaze-live",
        label: str = "Clean Stargaze live data",
        remote_device_ids: Iterable[str] = (),
        ready_timeout: float = 10.0,
        cancel_event: threading.Event | None = None,
    ) -> tuple[SyncthingShare | None, ServiceStatus]:
        """Prepare and start the managed process, returning stable connection info."""

        with self._lock:
            requested_folder = Path(folder_path).resolve()
            current = self.status()
            if current.state is ServiceState.RUNNING:
                tree = ET.parse(self.config_path).getroot()
                configured_folder = next(
                    (item for item in tree.findall("folder") if item.get("id") == folder_id),
                    None,
                )
                configured_path = (
                    Path(configured_folder.get("path", "")).resolve()
                    if configured_folder is not None
                    else None
                )
                if (
                    configured_path == requested_folder
                    and self._managed_endpoints_current(tree)
                ):
                    self._start_pairing_worker(folder_id)
                    return SyncthingShare(
                        requested_folder,
                        folder_id,
                        self._local_device_id(tree),
                        self.home_directory,
                    ), current
                stopped = self.stop(timeout=8.0, cancel_event=cancel_event)
                if stopped.state is not ServiceState.STOPPED:
                    return None, stopped
            prepared = self.prepare(
                folder_path, folder_id=folder_id, label=label,
                remote_device_ids=remote_device_ids, cancel_event=cancel_event,
            )
            if isinstance(prepared, ServiceStatus):
                return None, prepared
            executable = self.discover_executable()
            assert executable is not None
            _check_cancelled(cancel_event)
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = self._popen(
                [
                    str(executable), "serve", f"--home={self.home_directory}",
                    "--no-browser", "--no-console", "--no-restart", "--no-upgrade",
                    f"--log-file={self.home_directory / 'syncthing.log'}",
                ],
                cwd=str(self.app_directory),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self.pid_path.write_text(str(self._process.pid), encoding="ascii")
            self._active_folder_id = folder_id

        deadline = time.monotonic() + ready_timeout
        try:
            while time.monotonic() < deadline:
                _check_cancelled(cancel_event)
                current = self.status()
                if current.state in (ServiceState.RUNNING, ServiceState.ERROR):
                    if current.state is ServiceState.RUNNING:
                        self._start_pairing_worker(folder_id)
                    return prepared, current
                if self._process is not None and self._process.poll() is not None:
                    return prepared, ServiceStatus(
                        "syncthing", ServiceState.ERROR,
                        f"Syncthing exited with code {self._process.returncode}; inspect {self.home_directory / 'syncthing.log'}",
                        executable=executable,
                    )
                time.sleep(0.2)
            return prepared, self.status()
        except SharingCancelled:
            self.stop(cancel_event=None)
            raise

    def stop(self, *, timeout: float = 8.0, cancel_event: threading.Event | None = None) -> ServiceStatus:
        self._pairing_stop.set()
        with self._lock:
            if not self._is_running():
                self.pid_path.unlink(missing_ok=True)
                return ServiceStatus("syncthing", ServiceState.STOPPED, "Syncthing share is already stopped", executable=self.discover_executable())
            _check_cancelled(cancel_event)
            try:
                self._api_request("/rest/system/shutdown", method="POST")
            except (HTTPError, URLError, OSError, TimeoutError, RuntimeError):
                if self._process is not None and self._process.poll() is None:
                    self._process.terminate()

        deadline = time.monotonic() + timeout
        while self._is_running() and time.monotonic() < deadline:
            _check_cancelled(cancel_event)
            time.sleep(0.1)
        with self._lock:
            if self._is_running():
                if self._process is not None and self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=2.0)
                if self._is_running():
                    return ServiceStatus(
                        "syncthing", ServiceState.ERROR,
                        "Syncthing did not stop; close it from its local Web UI or Task Manager",
                        executable=self.discover_executable(), running=True,
                    )
            self.pid_path.unlink(missing_ok=True)
            self._process = None
            return ServiceStatus("syncthing", ServiceState.STOPPED, "Syncthing share stopped", executable=self.discover_executable(), verified=True)
