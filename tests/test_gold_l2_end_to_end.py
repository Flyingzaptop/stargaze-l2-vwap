from __future__ import annotations

from pathlib import Path

from tools.run_gold_l2_end_to_end import build_commands, parser


def test_end_to_end_runner_wires_fixed_validation_contract(tmp_path: Path) -> None:
    args = parser().parse_args([
        "--seconds", str(tmp_path / "seconds.parquet"),
        "--base", str(tmp_path / "base.npz"),
        "--out-dir", str(tmp_path / "run"),
        "--dry-run",
    ])
    commands = build_commands(args)
    assert [name for name, _, _ in commands] == ["prepare", "open", "risk", "rate"]
    flat = [item for _, command, _ in commands for item in command]
    assert "oracle_best" in flat
    assert "500.0" in flat
    assert "20260810" in flat
