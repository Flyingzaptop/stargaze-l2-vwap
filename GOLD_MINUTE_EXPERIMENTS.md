# XAUUSD Minute-Candle Experiments

## Objective

Test whether the last 60 one-minute XAUUSD candles contain out-of-sample
information about the shape of the next 5–60 minutes. This is a forecasting
experiment, not a trading strategy or a profitability claim.

The candle path avoids the historical cTrader L2 reconstruction ambiguity:
historical trendbars retain their original minute boundary, whereas the older
flattened depth parquet did not retain every original `ProtoOADepthEvent`
boundary.

## Four Experiments

The pipeline trains the same causal TCN family under a two-by-two design:

| Architecture | Line target | Regime target |
|---|---:|---:|
| Direct TCN forecaster | `direct_line` | `direct_regime` |
| Learned embedding plus historical retrieval | `retrieval_line` | `retrieval_regime` |

Using the same encoder family keeps the comparison focused on direct
generalisation versus historical analogue retrieval.

## Line Target

For every horizon `H` in `5,10,15,20,30,45,60`, the future close path is
expressed in basis points relative to the current close:

```text
y[tau] = 10000 * log(close[t + tau] / close[t])
```

An origin-anchored least-squares line is fitted to the complete future path.
The saved targets are:

- `R_H`: fitted line endpoint in basis points;
- `A_H = R_H / H`: slope in basis points per minute;
- `Q_H`: deterministic target quality based on directional magnitude and
  residual linearity;
- path RMSE and the actual endpoint for evaluation.

The direct model predicts a conditional mean, conditional sigma and quality for
every horizon. The retrieval model returns a similarity-weighted mean and
dispersion of its historical neighbours.

## Regime Target

The previous 60-minute fitted trend is compared with every future-horizon
trend:

- `friction`: the past or future trend is too weak/choppy;
- `reversal`: both are directional and their signs differ;
- `continuation`: both are directional and their signs agree.

These classes deliberately do not encode long or short. Direction remains a
property of the observed current trend; the target describes what happens to
that trend.

## Leakage Controls

- All features are causal.
- Windows cannot cross missing-minute or market-closure gaps.
- Train, validation and final holdout are chronological.
- A 120-minute wall-clock purge covers both the 60-minute input and maximum
  60-minute target horizon.
- Feature normalization and target scales are fitted on training history only.
- Historical retrieval uses only training embeddings as its searchable index.
- Holdout evaluation uses a separate five-minute stride by default so adjacent,
  nearly identical one-minute windows do not masquerade as independent evidence.

## Download

Create a gitignored secrets file based on `secrets.gold.example.json`. Only
read-only history messages are implemented; there is no order message in the
gold package.

```powershell
python -m stargaze_ml.cli download-gold-m1 `
  --secrets secrets.gold.runtime.json `
  --symbol XAUUSD `
  --start 2015-01-01 `
  --end now `
  --out source/ctrader/xauusd_m1.parquet
```

Downloads are split into seven-day parts, rate-limited by the official SDK and
resumable. Credentials are never written to manifests or model artifacts.

## Train

```powershell
python -m stargaze_ml.cli run-gold-minute `
  --candles source/ctrader/xauusd_m1.parquet `
  --out-dir runs/gold_minute_v01 `
  --epochs 200 --patience 20 --batch-size 256
```

Training stops only after validation has failed to improve for the configured
patience. The final holdout does not select epochs or hyperparameters.
