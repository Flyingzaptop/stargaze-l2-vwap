"""Untouched forward A/B report for frozen L2/VWAP policies."""

from __future__ import annotations

from typing import Any


def _policy_summary(report: dict[str, Any]) -> dict[str, Any]:
    selected = report.get("selected", {})
    return {
        "rows": int(report.get("rows", 0)),
        "completed_events": int(report.get("completed_events", 0)),
        "entry_candidates": int(report.get("entry_candidates", 0)),
        "trades": int(selected.get("trades", 0)),
        "mean_pnl_ticks": float(selected.get("mean_pnl_ticks", 0.0)),
        "total_pnl_ticks": float(selected.get("total_pnl_ticks", 0.0)),
        "win_rate": float(selected.get("win_rate", 0.0)),
    }


def build_forward_ab_report(
    audit: dict[str, Any],
    policies: dict[str, dict[str, Any]],
    *,
    minimum_trades: int = 30,
) -> dict[str, Any]:
    if minimum_trades < 1:
        raise ValueError("minimum_trades must be positive")
    source_hashes = {
        str(report.get("provenance", {}).get("source_seconds_sha256", ""))
        for report in policies.values()
    }
    source_hashes.discard("")
    if len(source_hashes) > 1:
        raise ValueError("policy reports were evaluated on different source seconds")
    provenance_complete = bool(policies) and all(
        bool(report.get("provenance", {}).get("source_seconds_sha256"))
        and bool(report.get("provenance", {}).get("prepared_sha256"))
        and bool(report.get("provenance", {}).get("policy_sha256"))
        for report in policies.values()
    )
    summaries = {name: _policy_summary(report) for name, report in policies.items()}
    enough = bool(summaries) and all(
        int(summary["trades"]) >= minimum_trades for summary in summaries.values()
    )
    return {
        "contract": "untouched forward; frozen models/controller; next-second BBO; first VWAP crossing",
        "provenance": {
            "complete": provenance_complete,
            "source_seconds_sha256": next(iter(source_hashes), None),
            "prepared_sha256s": {
                name: report.get("provenance", {}).get("prepared_sha256")
                for name, report in policies.items()
            },
            "policy_sha256s": {
                name: report.get("provenance", {}).get("policy_sha256")
                for name, report in policies.items()
            },
        },
        "recording": {
            "duration_seconds": float(audit.get("duration_seconds", 0.0)),
            "raw_rows": int(audit.get("raw_rows", 0)),
            "snapshot_rows": int(audit.get("snapshot_rows", 0)),
            "second_rows": int(audit.get("second_rows", 0)),
            "observed_second_fraction": float(
                audit.get("observed_second_fraction", 0.0)
            ),
            "invalid_or_crossed_snapshots": int(
                audit.get("invalid_or_crossed_snapshots", 0)
            ),
            "unknown_deleted_rows": int(audit.get("unknown_deleted_rows", 0)),
            "spread_ticks_p50": float(audit.get("spread_ticks_p50", 0.0)),
        },
        "policies": summaries,
        "minimum_trades_for_comparison": int(minimum_trades),
        "sample_sufficient": enough,
        "conclusion": (
            "forward sample reached the configured descriptive threshold"
            if enough
            else "insufficient forward trades; report data quality and outcomes without ranking policies"
        ),
    }


def forward_ab_markdown(report: dict[str, Any]) -> str:
    recording = report["recording"]
    lines = [
        "# Untouched forward L2/VWAP A/B",
        "",
        f"- Duration: {float(recording['duration_seconds']) / 3600.0:.2f} h",
        f"- Snapshots: {int(recording['snapshot_rows']):,}",
        f"- Observed seconds: {100.0 * float(recording['observed_second_fraction']):.2f}%",
        f"- Invalid/crossed snapshots: {int(recording['invalid_or_crossed_snapshots'])}",
        f"- Unknown deletes: {int(recording['unknown_deleted_rows'])}",
        f"- Provenance hashes complete: {bool(report['provenance']['complete'])}",
        "",
        "| Policy | Completed events | Candidates | Trades | Mean ticks | Win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, policy in report["policies"].items():
        lines.append(
            f"| {name} | {int(policy['completed_events'])} | "
            f"{int(policy['entry_candidates'])} | {int(policy['trades'])} | "
            f"{float(policy['mean_pnl_ticks']):+.2f} | "
            f"{100.0 * float(policy['win_rate']):.1f}% |"
        )
    lines.extend(("", f"Conclusion: {report['conclusion']}.", ""))
    return "\n".join(lines)
