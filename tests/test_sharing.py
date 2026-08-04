from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import xml.etree.ElementTree as ET

import pytest

from market_collector.sharing import (
    Aria2SeedManager,
    PackSeedManager,
    QBittorrentSeedManager,
    ServiceState,
    SharingCancelled,
    SharingExecutor,
    SyncthingShareManager,
    _bdecode,
    _bencode,
    create_v1_torrent,
)


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return self.payload


class QbitOpener:
    def __init__(self) -> None:
        self.requests = []
        self.info_hash = ""

    def open(self, request, timeout=0):
        self.requests.append(request)
        if request.full_url.endswith("/api/v2/app/version"):
            return FakeResponse(b"5.1.0")
        if request.full_url.endswith("/api/v2/torrents/add"):
            return FakeResponse(b"Ok.")
        if "/api/v2/torrents/info?" in request.full_url:
            return FakeResponse(json.dumps([{"hash": self.info_hash, "state": "stalledUP"}]).encode())
        raise AssertionError(request.full_url)


class FakeProcess:
    def __init__(self, command) -> None:
        self.command = command
        self.pid = 12345
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0


class AriaOpener:
    def open(self, request, timeout=0):
        rpc = json.loads(request.data)
        assert rpc["params"][0].startswith("token:")
        if rpc["method"] == "aria2.tellActive":
            result = [{
                "gid": "abc",
                "status": "active",
                "totalLength": "12",
                "completedLength": "12",
                "uploadLength": "0",
                "bittorrent": {"info": {"name": "pack"}},
            }]
            return FakeResponse(json.dumps({"jsonrpc": "2.0", "id": rpc["id"], "result": result}).encode())
        if rpc["method"] == "aria2.shutdown":
            return FakeResponse(json.dumps({"jsonrpc": "2.0", "id": rpc["id"], "result": "OK"}).encode())
        raise AssertionError(rpc["method"])


def _config(path: Path, device_id: str = "LOCAL-DEVICE") -> None:
    root = ET.Element("configuration")
    ET.SubElement(root, "device", {"id": device_id, "name": "local"})
    gui = ET.SubElement(root, "gui")
    ET.SubElement(gui, "address").text = "127.0.0.1:8384"
    ET.SubElement(gui, "apikey").text = "test-key"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_create_v1_torrent_is_deterministic_and_magnet_matches_info_hash(tmp_path: Path) -> None:
    pack = tmp_path / "pack 001"
    (pack / "nested").mkdir(parents=True)
    (pack / "a.bin").write_bytes(b"abc")
    (pack / "nested" / "b.bin").write_bytes(b"defgh")

    artifact = create_v1_torrent(
        pack,
        tmp_path / "pack.torrent",
        trackers=("udp://tracker.example:80/announce",),
        piece_length=4,
    )
    meta = _bdecode(artifact.torrent_path.read_bytes())
    info_hash = hashlib.sha1(_bencode(meta[b"info"])).hexdigest()

    assert artifact.file_count == 2
    assert artifact.total_bytes == 8
    assert artifact.info_hash == info_hash
    assert f"urn:btih:{info_hash}" in artifact.magnet_uri
    assert "dn=pack%20001" in artifact.magnet_uri
    assert meta[b"info"][b"pieces"] == hashlib.sha1(b"abcd").digest() + hashlib.sha1(b"efgh").digest()
    assert [entry[b"path"] for entry in meta[b"info"][b"files"]] == [[b"a.bin"], [b"nested", b"b.bin"]]
    second = create_v1_torrent(
        pack,
        tmp_path / "pack-second.torrent",
        trackers=("udp://tracker.example:80/announce",),
        piece_length=4,
    )
    assert artifact.torrent_path.read_bytes() == second.torrent_path.read_bytes()


def test_create_v1_torrent_cancellation_does_not_publish_partial_file(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "large.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    destination = tmp_path / "cancelled.torrent"
    cancel = threading.Event()

    def progress(done: int, total: int, path: Path) -> None:
        cancel.set()

    with pytest.raises(SharingCancelled):
        create_v1_torrent(pack, destination, progress=progress, cancel_event=cancel)
    assert not destination.exists()


def test_default_torrent_contains_public_tracker_tiers(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "data.bin").write_bytes(b"data")
    artifact = create_v1_torrent(pack, tmp_path / "pack.torrent", piece_length=4)
    meta = _bdecode(artifact.torrent_path.read_bytes())

    assert meta[b"announce"].startswith(b"udp://")
    assert len(meta[b"announce-list"]) >= 3
    assert artifact.magnet_uri.count("&tr=") >= 3


def test_background_handle_surfaces_result_and_supports_cancellation() -> None:
    entered = threading.Event()

    def operation(*, cancel_event):
        entered.set()
        cancel_event.wait(2)
        if cancel_event.is_set():
            raise SharingCancelled("cancelled")
        return 42

    handle = SharingExecutor.submit(operation)
    assert entered.wait(1)
    handle.cancel()
    with pytest.raises(SharingCancelled):
        handle.result(1)
    assert handle.done()


def test_qbittorrent_reports_actionable_status_when_executable_exists_but_api_is_missing(tmp_path: Path) -> None:
    executable = tmp_path / "qbittorrent.exe"
    executable.write_bytes(b"")

    class MissingOpener:
        def open(self, request, timeout=0):
            raise OSError("connection refused")

    manager = QBittorrentSeedManager(executable=executable, opener=MissingOpener())
    status = manager.status()
    assert status.state is ServiceState.NEEDS_CONFIGURATION
    assert "Web UI/API" in status.message
    assert not status.verified


def test_qbittorrent_adds_existing_pack_and_verifies_real_seeding_state(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "data.parquet").write_bytes(b"parquet-data")
    artifact = create_v1_torrent(pack, tmp_path / "pack.torrent", piece_length=4)
    opener = QbitOpener()
    opener.info_hash = artifact.info_hash
    manager = QBittorrentSeedManager(opener=opener)

    status = manager.start_seed(artifact.torrent_path, pack, verify_timeout=0)

    assert status.state is ServiceState.RUNNING
    assert status.verified
    assert status.details["torrent_state"] == "stalledUP"
    add_request = next(item for item in opener.requests if item.full_url.endswith("/api/v2/torrents/add"))
    assert str(pack.parent).encode() in add_request.data
    assert artifact.torrent_path.read_bytes() in add_request.data


def test_aria2_is_primary_hidden_long_lived_pack_seeder(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "data.parquet").write_bytes(b"parquet-data")
    artifact = create_v1_torrent(pack, tmp_path / "pack.torrent", piece_length=4)
    executable = tmp_path / "aria2c.exe"
    executable.write_bytes(b"")
    launches = []

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return FakeProcess(command)

    aria = Aria2SeedManager(tmp_path / "seed-state", executable=executable, popen=popen, opener=AriaOpener())
    manager = PackSeedManager(aria, QBittorrentSeedManager(opener=QbitOpener()))
    status = manager.start_seed(artifact.torrent_path, pack, verify_timeout=1)

    assert status.service == "aria2"
    assert status.state is ServiceState.RUNNING
    assert status.verified
    command, kwargs = launches[0]
    assert f"--dir={pack.parent}" in command
    assert "--seed-time=5256000" in command
    assert "--seed-ratio=0.0" in command
    assert "--enable-dht=true" in command
    assert "--enable-peer-exchange=true" in command
    assert "--bt-enable-lpd=true" in command
    assert kwargs["stdout"] is not None and kwargs["stderr"] is not None


def test_pack_seed_manager_falls_back_to_qbittorrent_when_bundled_aria_is_missing(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "data.bin").write_bytes(b"data")
    artifact = create_v1_torrent(pack, tmp_path / "pack.torrent", piece_length=4)
    qbit_opener = QbitOpener()
    qbit_opener.info_hash = artifact.info_hash
    manager = PackSeedManager(
        Aria2SeedManager(tmp_path / "state", executable=tmp_path / "missing-aria2c.exe"),
        QBittorrentSeedManager(opener=qbit_opener),
    )

    status = manager.start_seed(artifact.torrent_path, pack, verify_timeout=0)

    assert status.service == "qbittorrent"
    assert status.state is ServiceState.RUNNING
    assert status.verified
    assert status.details["backend"] == "qbittorrent"


def test_syncthing_prepare_creates_send_only_folder_and_persistent_descriptor(tmp_path: Path) -> None:
    app = tmp_path / "app"
    home = app / "syncthing-share"
    folder = tmp_path / "live-data"
    app.mkdir()
    home.mkdir()
    folder.mkdir()
    executable = app / "syncthing.exe"
    executable.write_bytes(b"")
    _config(home / "config.xml")
    manager = SyncthingShareManager(app, executable=executable, home_directory=home)

    share = manager.prepare(folder, remote_device_ids=("REMOTE-ONE",))

    assert not hasattr(share, "state")
    root = ET.parse(home / "config.xml").getroot()
    configured = root.find("folder")
    assert configured is not None
    assert configured.get("path") == str(folder.resolve())
    assert configured.get("type") == "sendonly"
    assert configured.get("rescanIntervalS") == "60"
    assert configured.get("fsWatcherEnabled") == "false"
    assert {item.get("id") for item in configured.findall("device")} == {"LOCAL-DEVICE", "REMOTE-ONE"}
    assert root.findtext("gui/address") == "127.0.0.1:18384"
    assert [item.text for item in root.findall("options/listenAddress")] == [
        "tcp://0.0.0.0:22001",
        "quic://0.0.0.0:22001",
        "dynamic+https://relays.syncthing.net/endpoint",
    ]
    assert share.connection_info == "syncthing://LOCAL-DEVICE/clean-stargaze-live"


def test_syncthing_connection_info_survives_manager_restart_and_folder_move(tmp_path: Path) -> None:
    app = tmp_path / "app"
    home = app / "syncthing-share"
    first_folder = tmp_path / "first-live-data"
    second_folder = tmp_path / "second-live-data"
    app.mkdir()
    home.mkdir()
    first_folder.mkdir()
    second_folder.mkdir()
    executable = app / "syncthing.exe"
    executable.write_bytes(b"")
    _config(home / "config.xml")

    first_manager = SyncthingShareManager(app, executable=executable, home_directory=home)
    first_share = first_manager.prepare(first_folder, remote_device_ids=("REMOTE-DEVICE",))
    second_manager = SyncthingShareManager(app, executable=executable, home_directory=home)
    second_share = second_manager.prepare(second_folder)

    assert first_share.connection_info == second_share.connection_info
    assert second_share.connection_info == "syncthing://LOCAL-DEVICE/clean-stargaze-live"
    configured = ET.parse(home / "config.xml").getroot().find("folder")
    assert configured is not None
    assert configured.get("path") == str(second_folder.resolve())
    assert {item.get("id") for item in configured.findall("device")} == {
        "LOCAL-DEVICE",
        "REMOTE-DEVICE",
    }


def test_syncthing_status_uses_api_when_windows_launcher_process_exits(tmp_path: Path) -> None:
    app = tmp_path / "app"
    home = app / "syncthing-share"
    app.mkdir()
    home.mkdir()
    executable = app / "syncthing.exe"
    executable.write_bytes(b"")
    _config(home / "config.xml")

    class StatusOpener:
        def open(self, request, timeout=0):
            assert request.full_url.endswith("/rest/system/status")
            return FakeResponse(json.dumps({"myID": "LOCAL-DEVICE"}).encode())

    manager = SyncthingShareManager(
        app, executable=executable, home_directory=home, opener=StatusOpener()
    )
    launcher = FakeProcess([])
    launcher.returncode = 0
    manager._process = launcher

    status = manager.status()

    assert status.state is ServiceState.RUNNING
    assert status.verified
    assert status.details["device_id"] == "LOCAL-DEVICE"


def test_syncthing_rejects_api_from_a_different_local_identity(tmp_path: Path) -> None:
    app = tmp_path / "app"
    home = app / "syncthing-share"
    app.mkdir()
    home.mkdir()
    executable = app / "syncthing.exe"
    executable.write_bytes(b"")
    _config(home / "config.xml")

    class ForeignStatusOpener:
        def open(self, request, timeout=0):
            assert request.full_url.endswith("/rest/system/status")
            return FakeResponse(json.dumps({"myID": "FOREIGN-DEVICE"}).encode())

    manager = SyncthingShareManager(
        app, executable=executable, home_directory=home, opener=ForeignStatusOpener()
    )
    status = manager.status()

    assert status.state is ServiceState.ERROR
    assert not status.verified
    assert status.details["expected_device_id"] == "LOCAL-DEVICE"
    assert status.details["actual_device_id"] == "FOREIGN-DEVICE"


def test_syncthing_auto_accepts_pending_device_and_adds_it_to_public_folder(tmp_path: Path) -> None:
    app = tmp_path / "app"
    home = app / "syncthing-share"
    app.mkdir()
    home.mkdir()
    executable = app / "syncthing.exe"
    executable.write_bytes(b"")
    _config(home / "config.xml")

    class SyncthingOpener:
        def __init__(self) -> None:
            self.requests = []
            self.folder = {"id": "clean-stargaze-live", "devices": [{"deviceID": "LOCAL-DEVICE"}]}

        def open(self, request, timeout=0):
            self.requests.append(request)
            url = request.full_url
            if url.endswith("/rest/cluster/pending/devices") and request.method == "GET":
                return FakeResponse(json.dumps({"REMOTE-DEVICE": {"name": "receiver"}}).encode())
            if url.endswith("/rest/config/defaults/device"):
                return FakeResponse(json.dumps({"deviceID": "", "addresses": []}).encode())
            if url.endswith("/rest/config/devices"):
                return FakeResponse(b"{}")
            if url.endswith("/rest/config/folders/clean-stargaze-live") and request.method == "GET":
                return FakeResponse(json.dumps(self.folder).encode())
            if url.endswith("/rest/config/folders/clean-stargaze-live") and request.method == "PUT":
                self.folder = json.loads(request.data)
                return FakeResponse(b"{}")
            if "/rest/cluster/pending/devices?" in url and request.method == "DELETE":
                return FakeResponse(b"")
            raise AssertionError((request.method, url))

    opener = SyncthingOpener()
    manager = SyncthingShareManager(app, executable=executable, home_directory=home, opener=opener)
    result = manager.accept_pending_devices("clean-stargaze-live")

    assert result.accepted_device_ids == ("REMOTE-DEVICE",)
    assert result.errors == ()
    assert {item["deviceID"] for item in opener.folder["devices"]} == {"LOCAL-DEVICE", "REMOTE-DEVICE"}


def test_syncthing_reconciles_known_device_when_pending_assignment_was_lost(tmp_path: Path) -> None:
    app = tmp_path / "app"
    home = app / "syncthing-share"
    app.mkdir()
    home.mkdir()
    executable = app / "syncthing.exe"
    executable.write_bytes(b"")
    _config(home / "config.xml")

    class ReconcileOpener:
        def __init__(self) -> None:
            self.folder = {
                "id": "clean-stargaze-live",
                "devices": [{"deviceID": "LOCAL-DEVICE"}],
            }

        def open(self, request, timeout=0):
            url = request.full_url
            if url.endswith("/rest/system/status"):
                return FakeResponse(json.dumps({"myID": "LOCAL-DEVICE"}).encode())
            if url.endswith("/rest/config/devices"):
                return FakeResponse(
                    json.dumps(
                        [
                            {"deviceID": "LOCAL-DEVICE"},
                            {"deviceID": "ALREADY-CONFIGURED-REMOTE"},
                        ]
                    ).encode()
                )
            if url.endswith("/rest/config/folders/clean-stargaze-live") and request.method == "GET":
                return FakeResponse(json.dumps(self.folder).encode())
            if url.endswith("/rest/config/folders/clean-stargaze-live") and request.method == "PUT":
                self.folder = json.loads(request.data)
                return FakeResponse(b"{}")
            raise AssertionError((request.method, url))

    opener = ReconcileOpener()
    manager = SyncthingShareManager(app, executable=executable, home_directory=home, opener=opener)

    added = manager.reconcile_folder_devices("clean-stargaze-live")

    assert added == ("ALREADY-CONFIGURED-REMOTE",)
    assert {item["deviceID"] for item in opener.folder["devices"]} == {
        "LOCAL-DEVICE",
        "ALREADY-CONFIGURED-REMOTE",
    }


def test_syncthing_missing_executable_never_claims_streaming(tmp_path: Path) -> None:
    app = tmp_path / "app"
    folder = tmp_path / "live"
    app.mkdir()
    folder.mkdir()
    manager = SyncthingShareManager(app, executable=tmp_path / "missing.exe")

    prepared = manager.prepare(folder)
    status = manager.status()

    assert prepared.state is ServiceState.MISSING
    assert status.state is ServiceState.MISSING
    assert not status.running
    assert "never downloaded" in prepared.message
