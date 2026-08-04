# L2/VWAP experiment ledger

## Contract

- Source: reconstructed demo cTrader L2 BBO, aggregated to causal one-second rows.
- Decision at second `t`; execution at the first observed BBO of `t+1`.
- Entry model emits only an open signal.
- Direction is decided by a separate policy.
- Exit is the configured VWAP crossing.
- Spread, commission, and slippage are included.
- Split: train before 2026-02-24 06:54 UTC; validation until
  2026-03-02 18:47 UTC; test until 2026-03-06 21:57 UTC.

## Features

Raw causal bid/ask quote-VWAPs are calculated at 5, 10, 15, 30, 45, 60,
120, 300, and 900 seconds. The model also receives distances, slopes,
adjacent-horizon differences, ribbon center/width, hierarchy curvature and
causal rolling PRICE-to-VWAP response fields.

The 60-second VWAP remains the primary excursion boundary. Experiments with
15, 30, 45 seconds and a ribbon target produced noisier entries. Lead/lag
features reduced direction quality in their current form.

## Direction results

All choices below were trained on train and thresholded on validation. PnL is
in configured price ticks after modeled costs.

| Direction policy | Validation-selected daily cap | Validation mean | Fixed test trades | Fixed test mean | Test win rate | Test p05 |
|---|---:|---:|---:|---:|---:|---:|
| Tail threshold 150 | 10 | +14.47 | 40 | -251.10 | 50.0% | -774.4 |
| Tail threshold 300 | 10 | +10.34 | 42 | -57.67 | 59.5% | -596.2 |
| Tail threshold 500 | 25 | +42.59 | 100 | +28.07 | 67.0% | -468.5 |

The tail-500 run is the first near-positive fixed-test result: +2,807 ticks
total over 100 trades. Its mean standard error is about 32 ticks, larger than
the +28-tick mean, so this is not statistically established edge.

## Causal rate control

The fixed validation cutoff drifted from roughly 14 trades/day on validation
to roughly 95/day on test. The rate controller therefore maintains a rolling
distribution of scores, calculates its cutoff from scores observed strictly
before the current decision, and applies a hard daily cap selected from
10/15/20/25 on validation. Future-score perturbation and daily-cap behavior
are covered by regression tests.

## Interpretation

Entry timing contains useful information: with oracle direction, most selected
events have a profitable side. Direction remains the bottleneck. Tail-aware
training and rate control materially improve it, but rare direction mistakes
still dominate average PnL.

The test period has now been inspected across several model variants and is
therefore an exploratory holdout, not a pristine final test. Before deployment,
freeze the pipeline and evaluate once on newer untouched L2 days, then run a
forward paper-trading period with the same execution contract.
