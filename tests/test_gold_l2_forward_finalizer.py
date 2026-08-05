from __future__ import annotations

from pathlib import Path

import pytest

from tools.finalize_gold_l2_forward import normalized_policies


def test_forward_finalizer_normalizes_unique_policy_names(tmp_path: Path) -> None:
    first = tmp_path / "v2"
    second = tmp_path / "v3"
    first.mkdir()
    second.mkdir()
    assert normalized_policies([["v2", str(first)], ["v3", str(second)]]) == [
        ("v2", first.resolve()),
        ("v3", second.resolve()),
    ]


@pytest.mark.parametrize("name", ["../escape", "two words", "a/b"])
def test_forward_finalizer_rejects_unsafe_policy_name(
    tmp_path: Path, name: str
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(ValueError, match="unsafe policy name"):
        normalized_policies([[name, str(bundle)]])


def test_forward_finalizer_rejects_duplicate_policy_name(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(ValueError, match="duplicate policy name"):
        normalized_policies([["v2", str(bundle)], ["v2", str(bundle)]])
