from __future__ import annotations

import numpy as np

from stargaze_ml.gold.l2_multivwap_side import multivwap_side


def test_global_agreement_keeps_local_mean_reversion_side() -> None:
    side, inverted, strength = multivwap_side(
        100.0, 101.0, np.asarray([102.0, 103.0, 104.0]),
        tick_size=1.0, neutral_ticks=0.0, consensus=0.67,
    )
    assert (side, inverted, strength) == (1, False, 1.0)


def test_global_disagreement_inverts_local_side() -> None:
    side, inverted, strength = multivwap_side(
        100.0, 101.0, np.asarray([99.0, 98.0, 102.0]),
        tick_size=1.0, neutral_ticks=0.0, consensus=0.33,
    )
    assert side == -1 and inverted and np.isclose(strength, 1 / 3)


def test_neutral_higher_vwaps_do_not_override_local_side() -> None:
    side, inverted, strength = multivwap_side(
        100.0, 99.0, np.asarray([100.1, 99.9]),
        tick_size=1.0, neutral_ticks=0.5, consensus=0.5,
    )
    assert (side, inverted, strength) == (-1, False, 0.0)
