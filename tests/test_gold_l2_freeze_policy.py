from __future__ import annotations

import pytest

from tools.freeze_gold_l2_policy import assert_freezable_report


def _report(*, approved: bool) -> dict[str, object]:
    return {
        "frozen_policy": {"mode": "risk"},
        "score_history_tail": [0.1],
        "validation_approved": approved,
    }


def test_rejected_policy_requires_explicit_research_override() -> None:
    with pytest.raises(ValueError, match="strict chronological validation"):
        assert_freezable_report(
            _report(approved=False), allow_rejected_validation=False
        )
    assert_freezable_report(
        _report(approved=False), allow_rejected_validation=True
    )


def test_approved_policy_can_be_frozen() -> None:
    assert_freezable_report(
        _report(approved=True), allow_rejected_validation=False
    )
