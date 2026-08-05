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
| Tail threshold 150 | 10 | -6.19 | 40 | -160.58 | 60.0% | -357.0 |
| Tail threshold 300 | 10 | -0.08 | 42 | -55.90 | 59.5% | -598.3 |
| Tail threshold 500 | 25 | +43.50 | 100 | +27.28 | 64.0% | -462.7 |

The tail-500 seed-10 run produced +2,728 ticks over 100 fixed-test trades. Its
mean standard error is about 30 ticks versus a +27-tick mean, so it is not
statistically established.

### Seed and model robustness

| Model | Validation-selected cap | Fixed test trades | Fixed test mean | Win rate |
|---|---:|---:|---:|---:|
| Tail-500 seed 10 | 25/day | 100 | +27.28 | 64.0% |
| Tail-500 seed 11 | 10/day | 40 | +80.68 | 75.0% |
| Tail-500 seed 12 | 15/day | 65 | -18.77 | 58.5% |
| Tail-500 seed 13 | 10/day | 40 | +35.43 | 70.0% |
| Four-seed ensemble | 10/day | 40 | +44.23 | 72.5% |
| Snapshot gradient boosting | 10/day | 29 | -94.41 | 65.5% |

Three of four seeds and the validation-selected ensemble are now positive, but
seed 12 remains negative. The ensemble mean (+44.23) is approximately equal to
its standard error (+45.37), so rare wrong-side tails still prevent an edge
claim.

These figures enforce an observed BBO for entry and the configured first
VWAP-crossing exit. Earlier reports sometimes used a carried quote when the
exact exit second had no update; those non-executable events are now excluded.

### Dominance diagnosis

In the corrected seed-10 test, four losses below -500 ticks contributed -3,638
ticks while the other 96 trades contributed +6,366. All four were incorrect
longs. At entry, price was below most VWAP horizons while 5-120 second VWAP
slopes were already following price downward: a clear PRICE-dominance pattern.

Several causal implementations were tested without test-based authorization:

- A deterministic multi-horizon entry override was rejected by validation.
- A live, confirmation-based dominance swap improved validation but reduced
  test mean from +28.07 to +12.01 because evidence often arrived after the loss.
- A symmetric snapshot dominance model had validation/test AUC 0.43/0.50.
- A symmetric LSTM dominance model reached entry AUC 0.59/0.56, but validation
  selected no veto; the fixed strategy therefore kept the original side.
- An interpretable 93-field hierarchy model combined scale consensus, ribbon
  ordering, gap expansion and lagged VWAP/price velocity. Validation selected
  48 trades at +90.10 ticks/trade, but fixed test lost -182.49 ticks/trade and
  swapped only 2.0% of selected entries.

The physical hypothesis remains plausible, but the current dataset cannot
calibrate a safe dominance override. More untouched days are required.

## Causal rate control

The fixed validation cutoff drifted from roughly 14 trades/day on validation
to roughly 95/day on test. The rate controller therefore maintains a rolling
distribution of scores, calculates its cutoff from scores observed strictly
before the current decision, and applies a hard daily cap selected from
10/15/20/25 on validation. Future-score perturbation and daily-cap behavior
are covered by regression tests.

### Adaptive excursion gate

A prior-event-only rolling amplitude quantile was tested to reduce the regime
drift of the fixed 242.5-tick gate. With a nominal target of 400 promising
events/day, realized candidate rates were 291.5/day on train, 297.3/day on
validation, and 313.2/day on test. The threshold is fixed at event start and
uses only completed earlier excursions. A frozen deployment bundle also carries
the last 2,000 causal amplitude observations so a restart does not silently
return to the fallback threshold.

Entry timing remained strong with oracle direction: validation/test first-cross
means were +125.86/+203.33 ticks over 1,503/1,188 entries. Direction did not
survive. The four-seed validation-selected ensemble produced -9.22 ticks over
32 validation trades and -70.86 ticks over 14 fixed test trades. Therefore the
adaptive gate is retained as infrastructure, but this trained adaptive policy is
rejected and is not the preferred frozen bundle.

### Longer exploration warmup

The fixed-gate open policy was retrained with exploration rising for 15 full
train passes before decaying. Oracle first-cross mean improved from
+166.30/+189.35 ticks on validation/test to +174.98/+196.18, while the number
of entries fell from 851/2,179 to 556/1,373. Newly trained direction seeds did
not improve test stability. Reusing the established four-seed direction models
with the new open model and its validation-selected 0.3 threshold produced a
more conservative hybrid: validation +52.66 ticks over 38 trades; fixed test
+29.25 over 40, 60.0% wins, 33.65-tick standard error and -451.55-tick p05.
It is frozen as `gold_l2_v3_warm15_hybrid` for forward A/B, not promoted over
v2 without more untouched data.

## Interpretation

Entry timing contains useful information: with oracle direction, most selected
events have a profitable side. Direction remains the bottleneck. Tail-aware
training and rate control materially improve it, but rare direction mistakes
still dominate average PnL.

The test period has now been inspected across several model variants and is
therefore an exploratory holdout, not a pristine final test. Before deployment,
freeze the pipeline and evaluate once on newer untouched L2 days, then run a
forward paper-trading period with the same execution contract.

The preferred frozen four-seed bundle is in `artifacts/gold_l2_v2_ensemble`;
`gold_l2_v1` preserves seed 10. Each bundle contains the open model, exact
rolling-score history, controller settings, hashes, and preparation contract.
The live recorder writes immutable depth and rebuilt BBO parts, so a new sample
can be evaluated without refitting.
