# Gold L2: causal one-second VWAP policy experiment

## Result

The complete pipeline works, but the first profit-only REINFORCE experiment did
not discover a tradable policy.  With executable BBO prices, 15 ticks of
commission per fill, one tick of slippage per fill, and spread paid through the
quotes, the policy converged to waiting:

| Split | Decision seconds | Events | Trades | Net PnL |
|---|---:|---:|---:|---:|
| Validation | 324,828 | 0 | 0 | 0 ticks |
| Test | 330,290 | 0 | 0 | 0 ticks |

Zero equals the explicit never-trade baseline.  It is not evidence of a
profitable strategy.  During stochastic training, event frequency fell from
0.931% in epoch 1 to 0.032% in epoch 30 while mean episode PnL remained
negative.  The economically rational local optimum was therefore no trade.

## Data contract

Source: `C:\Users\r3d_flzp\Documents\GitHub\golden-den\raw.parquet`.

- 75,978,432 quote events from 2026-02-03 through 2026-03-06 UTC.
- The source contains quote additions/deletions but no trade prints.  A true
  traded-volume VWAP cannot be recovered from it.
- The implemented `daily_book_vwap`, `book_vwap_60s`, and `book_vwap_300s` are
  explicitly labelled causal quote-liquidity proxies.  They never use future
  data.
- The historical book is an inferred full-refresh reconstruction because the
  exact original protobuf event boundaries were not retained.
- A second is actionable only after its `[t,t+1)` candle is complete.  An event
  is filled at the first accepted BBO in second `t+1`.  Carried quotes cannot be
  used for execution.
- Missing seconds are carried for at most two seconds; larger gaps start a new
  segment.  Every evaluation segment starts flat and is force-liquidated using
  its own next observed BBO.

The generated dataset has 1,689,355 one-second rows: 1,379,271 observed and
310,084 causally carried rows in 22,884 segments.  Splits are chronological
60/20/20, and the robust normalizer is fitted on train only.

## Model and actions

The model is a single-layer LSTM with 20,228 parameters and exactly four neural
outputs:

1. `OPEN_LONG`
2. `OPEN_SHORT`
3. `CLOSE_LONG`
4. `CLOSE_SHORT`

The outputs are competing positive event hazards.  `NOOP` is the implicit
probability that no hazard fires, not a fifth neural output.  Invalid commands
are masked before probability normalization according to position state:

- flat: open long or open short;
- long: close long;
- short: close short.

For symmetric serialization/visualization, the four visible events are encoded
as the vertices of a regular tetrahedron in R3.  Each code is the same distance
from zero and every pair is the same distance apart.  Complex roots
`{1, i, -1, -i}` would be equally distant from zero, but adjacent and opposite
commands would not be equally distant from one another.  Training is
categorical REINFORCE and does not regress these geometric codes.

The input consists of 12 stationary, causal values derived from one-second mid
OHLC, executable bid/ask, top-book microprice, and the three quote-VWAP proxies.
Absolute price levels are represented as tick distances from the current close.

## Reward and accounting

There are no supervised targets.  Reward is the change in one-lot liquidation
equity at each step; dense rewards telescope exactly to final net PnL.  Opens
and closes use executable ask/bid respectively.  Spread is therefore included
once in fills and is not subtracted a second time.  Commission and slippage are
charged on every fill.

The training run used 30 epochs, 4,096 random 128-second episodes per epoch,
batch size 512, hidden size 64, and seed 20260804 on CUDA.  Test was opened only
after the final checkpoint and validation result were fixed.

## Reproduce

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH='.'
python tools\prepare_gold_l2_policy.py `
  --raw 'C:\Users\r3d_flzp\Documents\GitHub\golden-den\raw.parquet' `
  --out-dir runs\gold_l2_policy_v1

python tools\train_gold_l2_policy.py train `
  --prepared runs\gold_l2_policy_v1\prepared_l2_policy.npz `
  --out-dir runs\gold_l2_policy_v1\reinforce_seed_20260804 `
  --hidden-size 64 --episode-length 128 `
  --episodes-per-epoch 4096 --batch-size 512 --epochs 30 `
  --learning-rate 0.0003 --entropy-start 0.01 --entropy-end 0.0005 `
  --initial-event-bias -5 --commission-per-fill-ticks 15 `
  --slippage-per-fill-ticks 1 --seed 20260804 --device auto

python tools\train_gold_l2_policy.py evaluate `
  --prepared runs\gold_l2_policy_v1\prepared_l2_policy.npz `
  --checkpoint runs\gold_l2_policy_v1\reinforce_seed_20260804\final.pt `
  --out-dir runs\gold_l2_policy_v1\reinforce_seed_20260804\test `
  --split test --device auto
```

## Sensible next experiment

Do not interpret the zero-trade policy as success.  Before increasing model
size, verify the real per-fill commission and lot-to-tick conversion.  Then use
a constrained-exploration training run that preserves a minimum event hazard
during warm-up, while keeping reward strictly equal to net PnL.  Select the
constraint and checkpoint on validation only, compare against the same
never-trade baseline, and open test once.  If it still collapses or is negative
after costs across several seeds, this input/reward formulation has no evidence
of edge and should not be developed into live trading.

## V2: side VWAP inputs and explicit exploration warm-up

The second experiment corrected the sampling contract and added four causal
inputs: raw `bid_vwap_60s`, raw `ask_vwap_60s`,
`(bid_vwap_60s - current_bid) / tick_size`, and
`(ask_vwap_60s - current_ask) / tick_size`.

Each epoch now covers all 785,335 decision-eligible train seconds exactly once
using non-overlapping chunks inside causal segments. Missing execution BBOs
mask only their own decision and no longer split the LSTM sequence. Tail
padding is excluded from policy loss, entropy, probabilities, and metrics.

Exploration has three independently logged schedules:

- entropy coefficient: `0.005 -> 0.03 -> 0.0005`;
- hazard temperature: `1.0 -> 1.3 -> 0.9`;
- admissible-event probability floor: `0.005 -> 0.03 -> 0.0005`.

The probability floor is part of the stochastic sampling distribution, and
the recorded policy log-probability comes from that same distribution. Reward
remains pure executable net PnL; no reward for trading and no supervised target
were added.

In the 128-second run, actual event rate rose from 1.55% to 4.91% during the
five warm-up epochs. After cooling it fell to 0.06%, mean train episode PnL
remained negative, and deterministic validation again selected no trades.

A second full-pass run used 512-second episodes so the policy could express the
120-300 second mean-reversion horizon observed on validation. Exploration rose
from 1.51% to 5.91% during warm-up and did not collapse early. The final epoch
still had negative mean PnL (`-17.14 ticks/episode`, `-0.283 ticks/decision`).
Deterministic probability argmax produced no validation events.

The longer model did learn the intended geometry on validation:

- correlation of `OPEN_LONG` hazard with `VWAP Bid - Bid`: `+0.440`;
- correlation of `OPEN_SHORT` hazard with `VWAP Ask - Ask`: `-0.300`.

Nevertheless, three stochastic validation runs without temperature or event
floor lost between 105,352 and 122,435 ticks. A coarse validation-only hazard
threshold scan found a narrow positive point at `0.020`: 46 trades, +2,232
ticks, 65.2% hit rate, and profit factor 1.12, but with 12,913 ticks of
drawdown. That threshold was fixed before opening test.

Held-out chronological test failed:

- 50 events, all `OPEN_LONG`;
- no model-generated close events;
- all positions liquidated only at segment termination;
- total net PnL: `-25,673 ticks`;
- mean trade: `-513.46 ticks`;
- hit rate: `50%`;
- profit factor: `0.494`;
- maximum drawdown: `39,135 ticks`.

The conclusion is negative for live trading. The network learned a plausible
VWAP-relative entry ranking, but profit-only REINFORCE did not learn a complete
open/close policy, and the validation threshold did not generalize to test.
