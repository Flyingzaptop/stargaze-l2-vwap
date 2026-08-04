from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Reproduce the causal L2/VWAP pipeline")
    result.add_argument("--seconds", type=Path, required=True)
    result.add_argument("--base", type=Path, required=True)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--primary-vwap", default="60")
    result.add_argument("--feature-profile", choices=("raw", "hierarchy", "leadlag"), default="raw")
    result.add_argument("--match-train-good-events", type=int, default=2850)
    result.add_argument("--open-epochs", type=int, default=30)
    result.add_argument("--risk-epochs", type=int, default=15)
    result.add_argument("--tail-threshold-ticks", type=float, default=500.0)
    result.add_argument("--risk-seed", type=int, default=20260810)
    result.add_argument("--device", default="auto")
    result.add_argument("--dry-run", action="store_true")
    return result


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    output = args.out_dir.resolve()
    prepared_dir = output / "prepared"
    prepared = prepared_dir / "prepared_l2_open_policy.npz"
    open_dir = output / "open_oracle"
    risk_dir = output / "risk_direction"
    rate_report = output / "causal_rate_report.json"
    python = sys.executable
    return [
        (
            "prepare",
            [python, "tools/prepare_gold_l2_open_policy.py", "--seconds", str(args.seconds.resolve()),
             "--base", str(args.base.resolve()), "--out-dir", str(prepared_dir),
             "--primary-vwap", str(args.primary_vwap), "--match-train-good-events",
             str(args.match_train_good_events), "--feature-profile", str(args.feature_profile)],
            prepared,
        ),
        (
            "open",
            [python, "tools/train_gold_l2_open_policy.py", "--prepared", str(prepared),
             "--out-dir", str(open_dir), "--epochs", str(args.open_epochs),
             "--reward-mode", "oracle_best", "--device", str(args.device)],
            open_dir / "final.pt",
        ),
        (
            "risk",
            [python, "tools/train_gold_l2_risk_direction.py", "--prepared", str(prepared),
             "--open-checkpoint", str(open_dir / "final.pt"), "--out-dir", str(risk_dir),
             "--epochs", str(args.risk_epochs), "--tail-threshold-ticks",
             str(args.tail_threshold_ticks), "--tail-weight", "1.0", "--seed",
             str(args.risk_seed), "--device", str(args.device)],
            risk_dir / "final.pt",
        ),
        (
            "rate",
            [python, "tools/evaluate_gold_l2_causal_rate.py", "--prepared", str(prepared),
             "--open-checkpoint", str(open_dir / "final.pt"), "--risk-checkpoint",
             str(risk_dir / "final.pt"), "--out", str(rate_report),
             "--device", str(args.device)],
            rate_report,
        ),
    ]


def _fingerprint(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {"path": str(resolved), "bytes": stat.st_size, "sha256": digest.hexdigest()}


def main() -> None:
    args = parser().parse_args()
    commands = build_commands(args)
    configuration = {
        "primary_vwap": args.primary_vwap,
        "feature_profile": args.feature_profile,
        "match_train_good_events": args.match_train_good_events,
        "open_epochs": args.open_epochs,
        "risk_epochs": args.risk_epochs,
        "tail_threshold_ticks": args.tail_threshold_ticks,
        "risk_seed": args.risk_seed,
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps({"config": configuration, "commands": [command for _, command, _ in commands]}, indent=2))
        return
    output = args.out_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "config": configuration,
        "inputs": {"seconds": _fingerprint(args.seconds), "base": _fingerprint(args.base)},
        "steps": [],
    }
    manifest_path = output / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        for name, command, artifact in commands:
            started = datetime.now(timezone.utc)
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
            if not artifact.exists():
                raise FileNotFoundError(f"step {name} did not create {artifact}")
            manifest["steps"].append({
                "name": name, "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifact": str(artifact),
            })
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
