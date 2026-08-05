from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from stargaze_ml.gold.ctrader_l2_recorder import recorded_l2_seconds
from stargaze_ml.gold.frozen_policy import file_sha256
from stargaze_ml.gold.l2_forward_report import (
    build_forward_ab_report,
    forward_ab_markdown,
)
from stargaze_ml.gold.l2_recording_audit import audit_l2_recording


def normalized_policies(pairs: list[list[str]]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for name, raw_path in pairs:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe policy name: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate policy name: {name}")
        seen.add(name)
        result.append((name, Path(raw_path).expanduser().resolve(strict=True)))
    return result


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _run_tool(root: Path, arguments: list[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(root) if not old_pythonpath else str(root) + os.pathsep + old_pythonpath
    )
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"tool failed ({' '.join(arguments)}):\n{detail}")
    return {
        "command": [sys.executable, *arguments],
        "stdout": completed.stdout[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize an untouched recording and frozen-policy A/B"
    )
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument(
        "--policy",
        nargs=2,
        action="append",
        metavar=("NAME", "BUNDLE"),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--max-quote-age-seconds", type=int, default=2)
    parser.add_argument("--minimum-trades", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    recording = args.recording.expanduser().resolve(strict=True)
    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies = normalized_policies(args.policy)

    audit = audit_l2_recording(recording, tick_size=float(args.tick_size))
    audit_path = output / "audit.json"
    _write_text_atomic(audit_path, json.dumps(audit, indent=2) + "\n")

    seconds = recorded_l2_seconds(
        recording,
        max_quote_age_seconds=int(args.max_quote_age_seconds),
    )
    seconds_path = output / "l2_seconds.parquet"
    seconds_temporary = seconds_path.with_suffix(".parquet.inprogress")
    seconds.write_parquet(seconds_temporary, compression="zstd", statistics=True)
    seconds_temporary.replace(seconds_path)

    steps: list[dict[str, Any]] = []
    policy_reports: dict[str, dict[str, Any]] = {}
    for name, bundle in policies:
        prepared_dir = output / f"prepared_{name}"
        report_path = output / f"forward_{name}.json"
        steps.append(
            _run_tool(
                root,
                [
                    "tools/prepare_gold_l2_open_policy.py",
                    "--seconds",
                    str(seconds_path),
                    "--out-dir",
                    str(prepared_dir),
                    "--all-test",
                    "--policy-bundle",
                    str(bundle),
                ],
            )
        )
        prepared_path = prepared_dir / "prepared_l2_open_policy.npz"
        steps.append(
            _run_tool(
                root,
                [
                    "tools/evaluate_frozen_gold_l2_forward.py",
                    "--prepared",
                    str(prepared_path),
                    "--bundle",
                    str(bundle),
                    "--out",
                    str(report_path),
                    "--device",
                    str(args.device),
                ],
            )
        )
        policy_reports[name] = json.loads(report_path.read_text(encoding="utf-8"))

    report = build_forward_ab_report(
        audit,
        policy_reports,
        minimum_trades=int(args.minimum_trades),
    )
    report_json = output / "forward_ab.json"
    report_markdown = output / "forward_ab.md"
    _write_text_atomic(report_json, json.dumps(report, indent=2) + "\n")
    _write_text_atomic(report_markdown, forward_ab_markdown(report))
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "recording": str(recording),
        "seconds_sha256": file_sha256(seconds_path),
        "policies": {name: str(bundle) for name, bundle in policies},
        "steps": steps,
        "outputs": {
            "audit": str(audit_path),
            "forward_ab_json": str(report_json),
            "forward_ab_markdown": str(report_markdown),
        },
    }
    manifest_path = output / "finalization_manifest.json"
    _write_text_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": int(seconds.height),
                "policies": list(policy_reports),
                "sample_sufficient": bool(report["sample_sufficient"]),
                "report": str(report_json),
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
