from __future__ import annotations

import pytest

from tools.install_ctrader_secure import numeric_version


def test_numeric_version_parses_release_and_suffix() -> None:
    assert numeric_version("26.4.0") == (26, 4, 0)
    assert numeric_version("7.35.1rc2") == (7, 35, 1, 2)


def test_numeric_version_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="no numeric"):
        numeric_version("release")
