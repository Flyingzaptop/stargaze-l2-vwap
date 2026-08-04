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

### Seed and model robustness

| Model | Validation-selected cap | Fixed test trades | Fixed test mean | Win rate |
|---|---:|---:|---:|---:|
| Tail-500 seed 10 | 25/day | 100 | +28.07 | 67.0% |
| Tail-500 seed 11 | 10/day | 40 | +24.10 | 75.0% |
| Tail-500 seed 12 | 20/day | 88 | -22.35 | 60.2% |
| Tail-500 seed 13 | 10/day | 40 | -27.28 | 67.5% |
| Four-seed ensemble | 10/day | 40 | -18.85 | 62.5% |
| Snapshot gradient boosting | 10/day | 29 | -94.41 | 65.5% |

The sign is seed-dependent despite similar validation AUC. High win rates and
positive medians coexist with negative means because a few wrong-side trades
dominate total PnL.

### Dominance diagnosis

In the positive seed-10 test, five losses below -500 ticks contributed -4,933
ticks while the other 95 trades contributed +7,740. All five were incorrect
longs. At entry, price was below most VWAP horizons while 5–120 second VWAP
slopes were already following price downward: a clear PRICE-dominance pattern.

Several causal implementations were tested without test-based authorization:

- A deterministic multi-horizon entry override was rejected by validation.
- A live, confirmation-based dominance swap improved validation but reduced
  test mean from +28.07 to +12.01 because evidence often arrived after the loss.
- A symmetric snapshot dominance model had validation/test AUC 0.43/0.50.
- A symmetric LSTM dominance model reached entry AUC 0.59/0.56, but validation
  selected no veto; the fixed strategy therefore kept the original side.

The physical hypothesis remains plausible, but the current dataset cannot
calibrate a safe dominance override. More untouched days are required.

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
