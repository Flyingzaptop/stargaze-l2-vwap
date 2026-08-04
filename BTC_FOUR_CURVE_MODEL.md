# BTC Four-Curve Causal Model

## Contract

The model consumes only information received at or before the current grid tick and emits exactly four bounded scores:

1. `LONG_backward_scoring`
2. `LONG_forward_scoring`
3. `SHORT_backward_scoring`
4. `SHORT_forward_scoring`

The target-free runtime maps these scores and the current position to `open_long`, `open_short`, `close_long`, `close_short`, `skip`, or `hold`. There is no maximum holding timer or price-based stop in the model replay.

## Data and Features

The causal replay reconstructs nine named venue states: Binance spot/perpetual, Bybit perpetual, OKX perpetual, Coinbase spot, Kraken spot/L3, Deribit perpetual, Bitfinex spot, and Hyperliquid perpetual. Features include L2 depth and imbalance at 1/5/10/25/50/100/250/1000 levels, BBO, spread, microprice, slopes, trades, multi-timescale returns, spot/perpetual basis, dispersion, funding, open interest, liquidations, and Kraken L3 lifecycle statistics.

Venue identity is represented by learned embeddings before cross-venue attention. A causal temporal Transformer then produces the four curves. The saved model has no target or future-validity input.

## Targets

Targets use exact Binance USDT-M executable BBO at `t + 250 ms`, require enough top-level quantity for `$1,000`, enter long at ask and exit at bid (the reverse for short), and subtract `10 bps` round-trip fees. The current economically stationary target pack is:

```text
horizons: 60, 120, 180, 240, 360, 480, 600, 900, 1200 seconds
focus: 480 seconds
backward minimum edge: 0.5 bps
forward minimum edge: 6.0 bps
full-quality edge: 20.0 bps
peak floor: 0.60
```

Backward curves remain local peak zones. Forward curves are dense economic opportunity zones: every smoothed tick above the fixed 6 bps net-edge boundary is supervised, while the local maximum still receives the largest score. This avoids train-period quantiles that produced entire later regimes with no labels.

The fixed-edge sweep is stored in `runs/fixed_edge_target_sweep_v01`. At the selected 480 second focus, the target oracle produced 115 completed trades, `+283.04 bps`, a 119 second median holding time, and no unresolved position. This is a target feasibility result, not model performance.

## Validation Result

The current common execution-quality frame set covers 66.11 hours. A sparse-peak Transformer, dense-forward Transformer, and XGBoost causal teacher were tested. The dense-forward Transformer reached the following validation result at epoch 2 and a frozen common threshold of `0.7`:

```text
validation replay at 10 bps / 250 ms:
  completed trades     5
  net                 +58.44 bps
  win rate            100%
  unresolved position  false

strict internal holdout:
  target events        27 / 50 / 7 / 0
  predicted opens      0
  trades               0
  net                   0 bps
  unresolved position  false
```

The checkpoint therefore fails deployment qualification. The XGBoost teacher also failed to produce a transferable policy, which shows that increasing Transformer capacity is not the immediate bottleneck. The 66-hour sample contains too few repeated volatility/direction regimes. No current artifact is represented as a profitable execution-ready model.

## Artifacts

Current diagnostic artifacts are:

```text
runs/btc_four_curve_dev_v12_dense_forward_seed_18301
runs/btc_xgb_dense_forward_teacher_v03
runs/fixed_edge_target_sweep_v01
runs/btc_walk_forward_feasibility_v01
```

The next candidate must be trained after materially more recorder data arrives and must pass chronological walk-forward folds plus a new untouched later holdout with completed profitable trades and no unresolved position.

## Commands

```powershell
python -m stargaze_ml.cli run-four-curve `
  --raw-dir "C:\Users\r3d_flzp\Sync\Clean Stargaze live data" `
  --out-dir runs/btc_four_curve_next `
  --horizons "60,120,180,240,360,480,600,900,1200" `
  --focus-seconds 480 --target-threshold-mode fixed_edge `
  --minimum-edge-bps 0.5 --forward-minimum-edge-bps 6 `
  --full-quality-edge-bps 20 --forward-curve-mode dense_edge `
  --peak-floor 0.6 --split-strategy blocked --background-stride 2

# Export only after the walk-forward and untouched-holdout gates pass.
```

Snapshots now physically copy and verify the committed MREC prefix. They no longer rely on mutable hardlinks.
