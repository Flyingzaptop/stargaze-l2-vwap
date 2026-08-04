from __future__ import annotations

from pathlib import Path
import subprocess

from market_collector.network_setup import (
    configure_windows_firewall,
    expected_firewall_rules,
    network_setup_is_current,
)


def test_firewall_setup_is_versioned_and_bound_to_exact_binaries(tmp_path: Path) -> None:
    recorder = tmp_path / "MarketRecorder.exe"
    syncthing = tmp_path / "syncthing.exe"
    aria2 = tmp_path / "aria2c.exe"
    for executable in (recorder, syncthing, aria2):
        executable.write_bytes(b"binary")
    state = tmp_path / "state"
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "Ok.", "")

    rules = expected_firewall_rules(recorder, syncthing, aria2)
    marker = configure_windows_firewall(state, rules, runner=runner)

    assert marker.is_file()
    assert network_setup_is_current(state, rules)
    assert len(calls) == len(rules) * 2
    add_commands = [command for command in calls if "add" in command]
    assert all(any(part.startswith("program=") for part in command) for command in add_commands)
    assert all(not any(part.startswith("localport=") for part in command) for command in add_commands)
    assert sum("dir=in" in command for command in add_commands) == 2
    assert sum("dir=out" in command for command in add_commands) == 3


def test_firewall_marker_invalidates_when_a_binary_moves_or_disappears(tmp_path: Path) -> None:
    recorder = tmp_path / "MarketRecorder.exe"
    syncthing = tmp_path / "syncthing.exe"
    aria2 = tmp_path / "aria2c.exe"
    for executable in (recorder, syncthing, aria2):
        executable.write_bytes(b"binary")
    state = tmp_path / "state"
    rules = expected_firewall_rules(recorder, syncthing, aria2)

    configure_windows_firewall(
        state,
        rules,
        runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    syncthing.unlink()

    assert not network_setup_is_current(state, rules)
