from __future__ import annotations

from stargaze_ml.gold.l2_forward_report import (
    build_forward_ab_report,
    forward_ab_markdown,
)


def _policy(
    trades: int, mean: float, *, source_hash: str = "seconds-a"
) -> dict[str, object]:
    return {
        "provenance": {
            "source_seconds_sha256": source_hash,
            "prepared_sha256": f"prepared-{trades}-{mean}",
            "policy_sha256": f"policy-{trades}-{mean}",
        },
        "rows": 100,
        "completed_events": 4,
        "entry_candidates": 3,
        "selected": {
            "trades": trades,
            "mean_pnl_ticks": mean,
            "total_pnl_ticks": trades * mean,
            "win_rate": 0.5,
        },
    }


def test_forward_report_refuses_to_rank_tiny_sample() -> None:
    audit = {
        "duration_seconds": 3600.0,
        "snapshot_rows": 500,
        "observed_second_fraction": 0.9,
    }
    report = build_forward_ab_report(
        audit, {"v2": _policy(1, -10.0), "v3": _policy(0, 0.0)}
    )
    assert report["sample_sufficient"] is False
    assert report["provenance"]["complete"] is True
    assert report["provenance"]["source_seconds_sha256"] == "seconds-a"
    assert "without ranking" in str(report["conclusion"])
    markdown = forward_ab_markdown(report)
    assert "| v2 | 4 | 3 | 1 | -10.00 | 50.0% |" in markdown


def test_forward_report_marks_descriptive_threshold() -> None:
    report = build_forward_ab_report(
        {}, {"v2": _policy(30, 1.0), "v3": _policy(30, 2.0)}
    )
    assert report["sample_sufficient"] is True


def test_forward_report_rejects_mixed_source_data() -> None:
    try:
        build_forward_ab_report(
            {},
            {
                "v2": _policy(30, 1.0, source_hash="seconds-a"),
                "v3": _policy(30, 2.0, source_hash="seconds-b"),
            },
        )
    except ValueError as exc:
        assert "different source seconds" in str(exc)
    else:
        raise AssertionError("mixed forward datasets must be rejected")
