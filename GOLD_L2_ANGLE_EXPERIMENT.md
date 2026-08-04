# XAUUSD L2 volatility-normalized angle experiment

## Outcome

The idea has a real but narrow proof of concept at microstructure horizons.

- Across every holdout point, the gain over a zero-angle forecast is small:
  **0.82% at 2 s**, **0.28% at 6 s**, and effectively zero beyond 10 s.
- At fixed 5% forecast coverage, selected only by the model's absolute predicted
  angle, the full-L2 CatBoost reaches **68.97% direction accuracy at 2 s**,
  **61.51% at 6 s**, and **59.50% at 10 s**.
- The corresponding price-only models reach 51.94%, 49.99%, and 52.22%.
  The confidence signal therefore comes from L2 state, not ordinary recent
  price momentum.
- Fitting the target itself to reconstructed microprice is worse than fitting
  future mid while using microprice as an input. The recommended contract is:
  **future mid angle as target; reconstructed microprice and L2 flow as features**.
- There is no useful general forecast at 30–60 s in this one-month dataset.

This is not a trading result. It ignores costs and tests only whether an
anchored local trend line is statistically predictable.

## Data and reconstruction

Source: 75,978,432 raw cTrader depth rows from 2026-02-03 through 2026-03-06.

The previous generated L2 files were not used as ground truth because 67–78%
of their rows had `best_bid > best_ask`. The new reconstruction:

1. Reads positive `new` depth quotes at an exact raw timestamp.
2. Requires at least three bid and three ask levels.
3. Rejects crossed, one-sided, over-wide, and packets with more than 20 new
   quotes at one timestamp.
4. Keeps the latest accepted snapshot in every 2-second bar.
5. Starts a new causal segment whenever a 2-second bar is missing.

This produced 810,312 bars and 45,781 segments. Only continuous runs with
enough past context and complete future labels enter training.

## Target

For horizon \(H\), with future offsets \(\tau=2,4,\ldots,H\) seconds:

\[
a_H =
\frac{\sum_\tau \tau(P_{t+\tau}-P_t)/tick}
     {\sum_\tau \tau^2}
\]

\[
\theta_H =
\arctan\left(
\frac{a_H\sqrt{H}}
     {\sigma_{\mathrm{past}}}
\right)
\]

- The intercept is fixed: the line passes through the current price.
- `sigma_past` is the trailing 60-second RMS tick change per square-root second.
- Training uses radians; reports and plots use degrees.
- Horizons are 2, 6, 10, 30, and 60 seconds.

## Causal features

The final matrix contains 82 features:

- current BBO spread, top-1 and top-3 imbalance;
- reconstructed microprice displacement from mid;
- depth, width, and top-level sizes;
- liquidity-flow imbalance over 2, 6, and 10 seconds;
- price/microprice changes from 2 through 60 seconds;
- rolling volatility, imbalance, spread, microprice divergence, and quote
  activity over 10 through 120 seconds;
- time-of-day and time-of-week cycles.

No future value is used in a feature. A dedicated test mutates future
microprices and verifies that all earlier feature rows are byte-identical.

## Splits and models

- Chronological 60/20/20 train/validation/holdout split.
- Context and maximum-label horizon are purged at both boundaries.
- Train: 149,697 rows.
- Validation: 59,070 rows.
- Holdout: 78,809 rows, from 2026-03-02 through 2026-03-06.
- Models: Ridge baseline, price-only CatBoost, full-L2 CatBoost, separate
  CatBoost per horizon, 6-layer causal TCN, and a validation-selected ensemble.
- The TCN was allowed 30 epochs and stopped at epoch 9 after validation stopped
  improving. This is genuine early stopping, not an arbitrary 10-epoch cap.

## Main holdout result: future mid target

The table uses the independent full-L2 CatBoost, which is the clearest model
for comparing horizons.

| Horizon | Angle MAE | Zero MAE | MAE gain | Correlation | Material direction |
|---:|---:|---:|---:|---:|---:|
| 2 s | 29.522° | 29.766° | 0.820% | 0.102 | 51.67% |
| 6 s | 32.712° | 32.802° | 0.276% | 0.063 | 50.93% |
| 10 s | 33.792° | 33.843° | 0.152% | 0.049 | 50.86% |
| 30 s | 34.772° | 34.799° | 0.079% | 0.029 | 51.03% |
| 60 s | 34.977° | 34.993° | 0.048% | 0.026 | 50.70% |

The model correctly shrinks most predictions toward zero. It cannot draw a
useful line on every bar.

## Confidence/coverage result

Coverage is fixed in advance. A row is selected only from the magnitude of the
model prediction, without reading its target.

| Horizon | Coverage | Direction | Angle MAE gain | Correlation |
|---:|---:|---:|---:|---:|
| 2 s | top 5% | 68.97% | 14.42% | 0.421 |
| 6 s | top 5% | 61.51% | 5.43% | 0.288 |
| 10 s | top 5% | 59.50% | 3.02% | 0.225 |
| 30 s | top 5% | 54.45% | 0.69% | 0.099 |
| 60 s | top 5% | 52.75% | 0.09% | 0.026 |

For the four complete holdout days, the equal-day bootstrap intervals for
top-5% direction accuracy are:

- 2 s: 67.92–69.68%.
- 6 s: 60.35–63.19%.
- 10 s: 57.71–61.59%.

Four days are far too few for a production claim, even though the daily
consistency is encouraging.

## Microprice as the target

The second complete experiment fits the future line to reconstructed
microprice rather than mid.

| Horizon | Mid-target correlation | Microprice-target correlation | Mid-target top-5% direction | Microprice-target top-5% direction |
|---:|---:|---:|---:|---:|
| 2 s | 0.102 | 0.037 | 68.97% | 58.44% |
| 6 s | 0.063 | 0.011 | 61.51% | 52.25% |
| 10 s | 0.049 | 0.003 | 59.50% | 49.58% |
| 30 s | 0.029 | 0.019 | 54.45% | 52.46% |
| 60 s | 0.026 | 0.026 | 52.75% | 54.97% |

Microprice is useful as a leading feature but is a noisier regression target
in this reconstruction.

## Limitations

- Historical cTrader parquet does not preserve an explicit protobuf event
  boundary. Exact timestamp plus packet-size filtering is conservative but
  still an inference.
- The holdout contains only four complete trading days plus one partial day.
- Confidence thresholds were evaluated by fixed coverage, not yet frozen as
  deployable absolute thresholds on a later untouched month.
- Consecutive 2-second labels overlap heavily. Day-block statistics reduce,
  but do not eliminate, dependence.
- No spread, slippage, latency, or execution logic is included.
- The 30–60 second outputs are not useful and should not be promoted because
  the 2-second result looks good.

## Recommended next experiment

Keep only the 2, 6, and 10 second heads. Train a two-stage system:

1. angle regressor on future mid;
2. calibrated abstention/gating head that predicts whether the angle error
   will be below a threshold.

Freeze coverage thresholds on validation, then collect at least another month
of raw protobuf-bounded L2 and evaluate once on that untouched period. That is
the shortest honest route from this proof of concept to evidence that might
survive outside these four days.
