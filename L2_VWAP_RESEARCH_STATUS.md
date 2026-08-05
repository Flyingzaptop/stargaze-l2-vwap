# L2/VWAP research status

## Validated findings

- The entry model selects excursions with larger side-agnostic opportunity than
  the amplitude gate alone, but mostly fires immediately at the gate.
- Among primary lines 15s, 30s, 45s, 60s and the 5-60s ribbon, 60s was strongest
  on chronological validation.
- Shorter primary VWAPs generate much more microstructure noise.
- Raw multi-horizon inputs beat the first explicit hierarchy experiment for
  direction selection.
- Multi-horizon lead/lag, explicit VWAP hierarchy and entry-only direction
  fine-tuning did not produce a validation-stable direction edge.
- Some policies look profitable on full validation or exploratory test, but no
  policy survives the required chronological first-half/second-half validation.
- Therefore no frozen policy is approved for trading. Historical bundles are
  research artifacts only.
- A fixed probability threshold is not rate-stable across validation and test.
- The untouched forward recorder and frozen streaming evaluator are operational;
  fresh coverage is still too small to estimate expectancy.

## Completed pipeline

1. Causal BBO reconstruction, second bars and 5/10/15/30/45/60/90/120/300/
   600/900s bid/ask VWAP hierarchy.
2. Entry model, direction model, value heads, lead/lag fields, adaptive rate gate
   and warm-up exploration.
3. Frozen ensembles with live/batch parity, uncertainty, next-BBO execution and
   costs.
4. Raw L2 recorder, integrity audit, converter, replay, forward A/B report and
   strict all-test contract.
5. Chronological robust validation that rejects unstable policies.

## Next evidence required

1. Collect multiple untouched days and market regimes without model selection.
2. Evaluate already-frozen bundles only; require at least 30 completed trades per
   policy before ranking and substantially more before any trading conclusion.
3. Treat all reused historical test results as exploratory.

Detailed results and rejected variants are in `L2_VWAP_EXPERIMENTS.md`.

## Leakage contract

- Rolling windows contain rows at or before the decision timestamp only.
- A decision at `t` executes at the first valid BBO in `t+1`.
- The closing crossing is never an input at entry time.
- Normalization fits on train only.
- Hyperparameters and thresholds are chosen on validation only.
- Test metrics are not used to select a model.
