# Stargaze: causal L2/VWAP research pipeline

Research code for reconstructing second-level BBO/L2 state, building causal
multi-horizon quote-VWAP features, detecting VWAP excursions, and training
separate entry and risk-aware direction policies. The repository also contains the
multi-exchange market recorder and older BTC/XAUUSD experiments used to build
the current pipeline.

This is research software, not a profitable trading system. Current holdout
results show that entry opportunity detection is materially easier than side
selection. The four-seed ensemble produced +44.23 ticks/trade over 40 exploratory
test trades, but its standard error was +45.37 ticks. Heavy-tail direction errors
and the small sample prevent calling the strategy tradable.

The preferred reproducible policy is in `artifacts/gold_l2_v2_ensemble`: one
open model, four direction seeds, and the exact causal-rate state used for
untouched forward evaluation. `gold_l2_v1` preserves the earlier single-seed
bundle. Raw/prepared L2 is excluded because it is roughly 434 MB.

## Current L2/VWAP pipeline

```text
raw cTrader demo L2
  -> conservative exact-timestamp snapshot reconstruction
  -> causal 1-second OHLC + first/last BBO
  -> bid/ask quote-VWAPs: 5/10/15/30/45/60/120/300/900s
  -> VWAP ribbon geometry + causal PRICE<->VWAP lead/lag response
  -> excursions between price/primary-VWAP crossings
  -> fixed or causal adaptive amplitude gate
  -> LSTM entry policy
  -> separate magnitude- and tail-risk-aware direction policy
  -> next-BBO execution, spread, commission and slippage
  -> chronological validation/test report
```

All rolling fields are causal. A decision at second `t` executes using the
observed BBO of `t+1`; VWAP-crossing exits require the same exact BBO contract.
Dataset splits are chronological. The historical test has been inspected across
variants and is exploratory; the included policy is frozen for newer forward data.

### Installation

Python 3.12+ and a CUDA-capable PyTorch build are recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Copy the placeholder secrets file and fill it locally. Runtime secret files are
gitignored and must never be committed.

```powershell
Copy-Item secrets.gold.example.json secrets.gold.runtime.json
```

### Prepare second-level open-policy data

The input Parquet/NPZ files are intentionally not stored in Git.

To collect a fresh forward L2 sample from cTrader, run the live incremental
depth recorder. It stores immutable raw-event parts and a causally reconstructed
BBO snapshot after every depth update. cTrader depth events have no exchange
timestamp, so `timestamp` is the local receive time in nanoseconds.

```powershell
python tools\record_gold_l2_live.py `
  --secrets secrets.gold.runtime.json `
  --symbol XAUUSD `
  --out-dir runs\gold_l2_live
```

Use `--duration-seconds 60` for a short connection test. Runtime credentials and
all recorded data remain gitignored.

Inspect an active or completed recording without touching its open buffer:

```powershell
python tools\audit_gold_l2_recording.py --recording runs\gold_l2_live
```

Convert the resulting snapshot stream into the same causal one-second contract
used by the research pipeline:

```powershell
python tools\prepare_recorded_gold_l2_seconds.py `
  --recording runs\gold_l2_live `
  --output runs\gold_l2_live\l2_seconds.parquet
```

Prepare that sample strictly as untouched forward data and evaluate only the
already-frozen checkpoints/controller:

```powershell
python tools\prepare_gold_l2_open_policy.py `
  --seconds runs\gold_l2_live\l2_seconds.parquet `
  --out-dir runs\gold_l2_live\prepared --all-test `
  --policy-bundle artifacts\gold_l2_v2_ensemble

python tools\evaluate_frozen_gold_l2_forward.py `
  --prepared runs\gold_l2_live\prepared\prepared_l2_open_policy.npz `
  --bundle artifacts\gold_l2_v2_ensemble `
  --out runs\gold_l2_live\forward_report.json
```

For parity testing, replay the same policy one causal second at a time (LSTM
states and rate-controller state are retained exactly as they would be live):

```powershell
python tools\replay_frozen_gold_l2.py `
  --prepared runs\gold_l2_live\prepared\prepared_l2_open_policy.npz `
  --bundle artifacts\gold_l2_v2_ensemble `
  --split all --out runs\gold_l2_live\replay.json
```

Run paper-only decisions against the active atomic recording (no orders are
sent):

```powershell
python tools\paper_gold_l2_live.py `
  --recording runs\gold_l2_live --bundle artifacts\gold_l2_v2_ensemble `
  --out runs\gold_l2_live\paper_decisions.json
```

For a complete reproducible run, use the orchestrator. It fingerprints both
inputs and writes a step-by-step `pipeline_manifest.json`:

```powershell
python tools\run_gold_l2_end_to_end.py `
  --seconds runs\gold_l2_policy_v2\l2_seconds.parquet `
  --base runs\gold_l2_policy_v2\prepared_l2_policy.npz `
  --out-dir runs\gold_l2_e2e `
  --primary-vwap 60 --feature-profile raw --device auto
```

```powershell
$env:PYTHONPATH='.'
python tools\prepare_gold_l2_open_policy.py `
  --seconds runs\gold_l2_policy_v2\l2_seconds.parquet `
  --base runs\gold_l2_policy_v2\prepared_l2_policy.npz `
  --primary-vwap 60 `
  --match-train-good-events 2850 `
  --out-dir runs\gold_l2_multihorizon\primary_60
```

To keep the candidate rate more stable across changing volatility regimes, use
the adaptive gate. Each event threshold is fixed at its start and is computed
only from previously completed excursion amplitudes:

```powershell
python tools\prepare_gold_l2_open_policy.py `
  --seconds runs\gold_l2_policy_v2\l2_seconds.parquet `
  --base runs\gold_l2_policy_v2\prepared_l2_policy.npz `
  --primary-vwap 60 --adaptive-gate-target 400 `
  --out-dir runs\gold_l2_adaptive_gate\target400

python tools\analyze_gold_l2_adaptive_gate.py `
  --prepared runs\gold_l2_adaptive_gate\target400\prepared_l2_open_policy.npz `
  --target 300 --target 400 `
  --out runs\gold_l2_adaptive_gate\gate_report.json
```

### Train entry and direction

```powershell
python tools\train_gold_l2_open_policy.py `
  --prepared runs\gold_l2_multihorizon\primary_60\prepared_l2_open_policy.npz `
  --out-dir runs\gold_l2_multihorizon\primary_60\open_oracle `
  --epochs 30 --warmup-epochs 15 --reward-mode oracle_best --device auto

python tools\train_gold_l2_profit_direction.py `
  --prepared runs\gold_l2_multihorizon\primary_60\prepared_l2_open_policy.npz `
  --open-checkpoint runs\gold_l2_multihorizon\primary_60\open_oracle\final.pt `
  --out-dir runs\gold_l2_multihorizon\primary_60\profit_direction `
  --epochs 15 --device auto

python tools\train_gold_l2_risk_direction.py `
  --prepared runs\gold_l2_multihorizon\primary_60\prepared_l2_open_policy.npz `
  --open-checkpoint runs\gold_l2_multihorizon\primary_60\open_oracle\final.pt `
  --out-dir runs\gold_l2_multihorizon\primary_60\risk_direction `
  --epochs 15 --tail-threshold-ticks 500 --tail-weight 1.0 `
  --seed 20260810 --device auto

python tools\evaluate_gold_l2_causal_rate.py `
  --prepared runs\gold_l2_multihorizon\primary_60\prepared_l2_open_policy.npz `
  --open-checkpoint runs\gold_l2_multihorizon\primary_60\open_oracle\final.pt `
  --risk-checkpoint runs\gold_l2_multihorizon\primary_60\risk_direction\final.pt `
  --out runs\gold_l2_multihorizon\primary_60\risk_direction\causal_rate_report.json `
  --device auto
```

The preferred four-seed ensemble produced +44.23 ticks/trade over 40 exploratory
holdout trades with 72.5% wins, but its standard error was 45.37 ticks and the
5th percentile was -663.4 ticks. This is not evidence of deployable edge. The
adaptive score controller uses only earlier scores and enforces a
validation-chosen 10-25 trade daily cap. See
[`L2_VWAP_EXPERIMENTS.md`](L2_VWAP_EXPERIMENTS.md) for the full experiment
ledger and limitations. Raw datasets and non-frozen runs remain gitignored.

### Tests

```powershell
python -m pytest -q
```

Key implementation files:

- `stargaze_ml/gold/l2_seconds.py` — L2/BBO reconstruction and causal seconds.
- `stargaze_ml/gold/l2_open_events.py` — multi-horizon VWAPs, ribbon,
  lead/lag fields and excursion segmentation.
- `stargaze_ml/gold/l2_open_reinforce.py` — entry-only REINFORCE policy.
- `stargaze_ml/gold/l2_profit_direction.py` — magnitude-aware side/value model.
- `stargaze_ml/gold/l2_risk_direction.py` — value, opportunity and tail-risk heads.
- `stargaze_ml/gold/l2_causal_rate.py` — causal rolling-quantile rate controller.
- `stargaze_ml/gold/l2_adaptive_gate.py` — prior-event-only adaptive excursion gate.
- `stargaze_ml/gold/l2_dominance_override.py` — causal multi-VWAP dominance evidence and live-swap simulator.
- `stargaze_ml/gold/l2_dominance_lstm.py` — symmetric PRICE/VWAP dominance sequence model.
- `tests/test_gold_l2_*.py` — causal and execution-contract regression tests.

## Market data collector

This collector records exchange-native BTC market data into separate Parquet streams:

- Binance spot and USDT-M futures: sequence-checked depth, trades
- OKX: books, trades
- Bybit: orderbook, public trades
- Coinbase Advanced: level2, market_trades
- Kraken: book, trade, level3/raw
- Deribit: BTC perpetual book, trades, derivative context
- Bitfinex: spot book and trades
- Hyperliquid: perpetual L2, trades, derivative context

Raw streams are kept separate. Cross-exchange aggregation should be built later as a derived dataset, not written into the raw files.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy config.example.json config.json
python run_collector.py --config config.json
```

Stop with `Ctrl+C`; the collector flushes committed blocks and closes the same
append-only `*.mrec` files. A later launch continues appending to them.
Binance depth streams also have an inactivity watchdog: a silent connection is
closed after 15 seconds, followed by a fresh REST snapshot and sequence bridge.

Authenticated channels can use a separate gitignored secrets file:

```powershell
python run_collector.py --config config.json --secrets secrets.json
```

`secrets.json` shape:

```json
{
  "kraken": {
    "api_key": "...",
    "api_secret": "..."
  }
}
```

## Output

The desktop recorder writes only to `E:/MarketRecorder/dataset`. If drive E is
unavailable, startup fails instead of silently filling drive C.

```text
dataset/binance_um_futures_BTCUSDT_depth.mrec
dataset/binance_um_futures_BTCUSDT_trades.mrec
dataset/okx_swap_BTC_USDT_SWAP_books.mrec
```

Each row has a common schema with exchange, symbol, channel, timestamps, sequence IDs, side, price, quantity, action, trade fields, and optional raw message on the first row of each event.

## Important

Only one recorder instance is allowed. Each logical exchange/market/symbol/channel
stream owns exactly one append-only `*.mrec` file. Every committed frame is an
independent zstd-compressed Parquet block with a length, row count and checksum.
After a crash, only an incomplete trailing frame is discarded and recording
continues in the same file. There is no time-based rotation or compaction.

## Sharing

- `Pack` creates an immutable point-in-time copy under
  `E:/MarketRecorder/packs`. It copies only committed frames while recording
  continues, then verifies every copied Parquet block and checksum.
- `Pack and Send` creates a standard `.torrent`, starts the bundled aria2 seed,
  and copies the magnet link to the clipboard.
- Live streaming starts automatically with the recorder and shares the active
  `dataset` directory through a stable `syncthing://DEVICE-ID/FOLDER-ID` code.
  Syncthing rescans once per minute and transfers changed blocks, so no
  minute-sized mirror files are created.
  A watchdog restarts Syncthing after a process or connectivity failure; the
  `Restart Stream Data` button performs the same recovery manually. The sender automatically accepts a
  remote Syncthing device that knows this code. The receiver accepts the folder
  invitation once; later segments arrive without restarting the transfer.

A normal BitTorrent magnet cannot remain constant when files change because its
infohash addresses immutable content. The stream code uses Syncthing for that
reason. Anyone who knows the stream code can request the public dataset.

On the first packaged-EXE launch, the recorder requests administrator approval
once. A short-lived elevated helper copies the bundled Syncthing/aria2 binaries
to the persistent `sharing_state/bin` directory and installs program-bound
Windows Firewall rules for Syncthing and aria2 inbound/outbound traffic plus
recorder outbound traffic. Program-bound rules are used because Syncthing can
select a different listen port when its default is already occupied. The main
recorder continues without administrator rights. The exact rule version and
binary paths are stored in `sharing_state/network_setup.json`; moving the EXE
or removing the setup causes another one-time prompt.

Windows elevation cannot configure an ISP CGNAT. Syncthing uses its public relay
fallback when direct NAT traversal is unavailable. BitTorrent has no equivalent
relay, so Pack and Send still requires at least one reachable peer or a router/
VPN port mapping when both sides are behind restrictive NAT.

At 10 GiB free space the app sends the configured Windows warning. At 1 GiB it
stops the recorder gracefully to avoid corrupting active files.

## Offline Causal BTC Model

`stargaze_ml` builds a receive-time causal stream from all nine BTC datasets,
reconstructs L2/L3 state, ports the multihorizon forward/backward scores,
trains a state-conditioned causal Transformer, and emits model-only actions.

```powershell
python -m stargaze_ml.cli catalog --raw-dir "source/raw datasets"
python -m stargaze_ml.cli run-offline --raw-dir "source/raw datasets" --out-dir runs/btc_offline_v01
```

Fast end-to-end smoke:

```powershell
python -m stargaze_ml.cli run-offline --duration-seconds 180 --horizons "0.5,1,2,5,10" --purge-seconds 2 --context-ticks 32 --hidden-size 48 --layers 2 --heads 4 --epochs 2 --out-dir runs/btc_real_smoke_v01
```

Replay a checkpoint without label rules, timers, or score gates:

```powershell
python -m stargaze_ml.cli replay-checkpoint --run-dir runs/btc_real_smoke_v01 --context-ticks 32
```

Short runs are marked `smoke_only`: they prove the causal software path, not
trading quality. Production evaluation requires a multi-day chronological
holdout and Binance execution data.

The four-curve targets, exact execution assumptions, validation results, and
target-free runtime are documented in
[`BTC_FOUR_CURVE_MODEL.md`](BTC_FOUR_CURVE_MODEL.md).

## XAUUSD Minute-Candle Research

The separate gold pipeline downloads cTrader M1 trendbars and runs four
experiments: direct versus historical-retrieval TCNs, each trained on either
multi-horizon fitted lines or `friction/reversal/continuation` regimes.

See [`GOLD_MINUTE_EXPERIMENTS.md`](GOLD_MINUTE_EXPERIMENTS.md) for the target
definitions, leakage controls and commands.
