"""One-time elevated Windows Firewall setup for the packaged recorder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Callable, Sequence


NETWORK_SETUP_VERSION = 2
RULE_PREFIX = "Clean Stargaze Market Recorder"


@dataclass(frozen=True)
class FirewallRule:
    name: str
    direction: str
    program: Path
    protocol: str = "any"
    local_port: str | None = None


def expected_firewall_rules(
    recorder_executable: str | Path,
    syncthing_executable: str | Path,
    aria2_executable: str | Path,
) -> tuple[FirewallRule, ...]:
    recorder = Path(recorder_executable).resolve()
    syncthing = Path(syncthing_executable).resolve()
    aria2 = Path(aria2_executable).resolve()
    return (
        FirewallRule(f"{RULE_PREFIX} - Recorder outbound", "out", recorder),
        FirewallRule(f"{RULE_PREFIX} - Syncthing outbound", "out", syncthing),
        FirewallRule(f"{RULE_PREFIX} - Syncthing inbound", "in", syncthing),
        FirewallRule(f"{RULE_PREFIX} - aria2 outbound", "out", aria2),
        FirewallRule(f"{RULE_PREFIX} - aria2 inbound", "in", aria2),
    )


def marker_path(state_directory: str | Path) -> Path:
    return Path(state_directory).resolve() / "network_setup.json"


def _marker_payload(rules: Sequence[FirewallRule]) -> dict:
    return {
        "version": NETWORK_SETUP_VERSION,
        "rules": [
            {
                **asdict(rule),
                "program": str(rule.program),
            }
            for rule in rules
        ],
    }


def network_setup_is_current(
    state_directory: str | Path,
    rules: Sequence[FirewallRule],
) -> bool:
    path = marker_path(state_directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == _marker_payload(rules) and all(rule.program.is_file() for rule in rules)


def _netsh_path() -> Path:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    return system_root / "System32" / "netsh.exe"


def _rule_command(netsh: Path, rule: FirewallRule) -> list[str]:
    command = [
        str(netsh),
        "advfirewall",
        "firewall",
        "add",
        "rule",
        f"name={rule.name}",
        f"dir={rule.direction}",
        "action=allow",
        f"program={rule.program}",
        f"protocol={rule.protocol}",
        "profile=any",
        "enable=yes",
    ]
    if rule.local_port is not None:
        command.append(f"localport={rule.local_port}")
    return command


def configure_windows_firewall(
    state_directory: str | Path,
    rules: Sequence[FirewallRule],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    """Replace owned firewall rules and persist an exact-version marker."""

    state = Path(state_directory).resolve()
    state.mkdir(parents=True, exist_ok=True)
    netsh = _netsh_path()
    failures: list[str] = []
    for rule in rules:
        runner(
            [
                str(netsh),
                "advfirewall",
                "firewall",
                "delete",
                "rule",
                f"name={rule.name}",
            ],
            capture_output=True,
            text=True,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        result = runner(
            _rule_command(netsh, rule),
            capture_output=True,
            text=True,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown netsh error").strip()
            failures.append(f"{rule.name}: {detail}")
    if failures:
        raise RuntimeError("Windows Firewall setup failed: " + "; ".join(failures))

    path = marker_path(state)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_marker_payload(rules), indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path
