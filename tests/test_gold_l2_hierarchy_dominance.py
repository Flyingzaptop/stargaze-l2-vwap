from __future__ import annotations

import numpy as np
import pytest

from stargaze_ml.gold.l2_hierarchy_dominance import HORIZONS, hierarchy_summary


def test_hierarchy_summary_detects_scale_ordering() -> None:
    increasing = np.arange(1, len(HORIZONS) + 1, dtype=np.float64)
    summary = hierarchy_summary(increasing, -increasing)
    assert summary[0] == 1.0
    assert summary[1] == 1.0
    assert summary[4] > 0
    assert summary[7] > 0
    assert summary[10] < 0


def test_hierarchy_summary_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="HORIZONS"):
        hierarchy_summary(np.ones(2), np.ones(2))
