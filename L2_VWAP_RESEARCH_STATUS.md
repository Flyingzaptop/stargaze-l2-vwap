# L2/VWAP research status

## Validated findings

- The entry model selects excursions with larger side-agnostic opportunity than
  the amplitude gate alone, but mostly fires immediately at the gate.
- Among primary lines 15s, 30s, 45s, 60s and the 5-60s ribbon, 60s was strongest
  on chronological validation.
- Shorter primary VWAPs generate much more microstructure noise.
- Raw multi-horizon inputs beat the first explicit hierarchy experiment for
  direction selection.
- The best current direction model remains unprofitable after next-BBO spread,
  commission and slippage because rare wrong-side trades dominate the mean.
- A fixed probability threshold is not rate-stable across validation and test.

## Current research direction

1. Add causal PRICE-to-VWAP and VWAP-to-PRICE lead/lag response fields.
2. Weight side mistakes by `abs(PnL_long - PnL_short)`.
3. Regress both executable long and short PnL as auxiliary values.
4. Replace fixed probability-rate assumptions with a causal adaptive selector.
5. Keep entry, direction and execution accounting independently auditable.

## Leakage contract

- Rolling windows contain rows at or before the decision timestamp only.
- A decision at `t` executes at the first valid BBO in `t+1`.
- The closing crossing is never an input at entry time.
- Normalization fits on train only.
- Hyperparameters and thresholds are chosen on validation only.
- Test metrics are not used to select a model.
