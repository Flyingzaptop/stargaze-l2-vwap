from __future__ import annotations

import pytest

from stargaze_ml.gold.l2_risk_ensemble import (
    average_risk_predictions,
    risk_prediction_disagreement,
)


def test_average_risk_predictions_averages_each_head() -> None:
    result = average_risk_predictions([
        {"side": 0.2, "tail": 0.8},
        {"side": 0.6, "tail": 0.4},
    ])
    assert result == {"side": pytest.approx(0.4), "tail": pytest.approx(0.6)}


def test_risk_prediction_disagreement_reports_population_std() -> None:
    result = risk_prediction_disagreement([
        {"side": 0.2, "tail": 0.8},
        {"side": 0.6, "tail": 0.4},
    ])
    assert result == {"side_std": pytest.approx(0.2), "tail_std": pytest.approx(0.2)}
