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
    assert "--warmup-epochs" in flat
    assert "--vwap-horizons" in flat
    assert "5,10,15,30,45,60,90,120,300,600,900" in flat


def test_end_to_end_runner_wires_adaptive_gate(tmp_path: Path) -> None:
    args = parser().parse_args([
        "--seconds", str(tmp_path / "seconds.parquet"),
        "--base", str(tmp_path / "base.npz"),
        "--out-dir", str(tmp_path / "run"),
        "--adaptive-gate-target", "400",
    ])
    prepare = build_commands(args)[0][1]
    assert prepare[-2:] == ["--adaptive-gate-target", "400"]


def test_end_to_end_runner_wires_direction_ensemble(tmp_path: Path) -> None:
    args = parser().parse_args([
        "--seconds", str(tmp_path / "seconds.parquet"),
        "--base", str(tmp_path / "base.npz"),
        "--out-dir", str(tmp_path / "run"),
        "--risk-seed", "10",
        "--risk-seed", "11",
    ])
    commands = build_commands(args)
    assert [name for name, _, _ in commands] == [
        "prepare", "open", "risk_seed_10", "risk_seed_11", "rate"
    ]
    rate = commands[-1][1]
    assert "tools/evaluate_gold_l2_risk_ensemble.py" in rate
    assert rate.count("--risk-checkpoint") == 2
