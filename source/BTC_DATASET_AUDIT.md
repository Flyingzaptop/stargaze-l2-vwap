# BTC multi-exchange dataset audit

Дата проверки: 2026-07-12

## Короткий вывод

Датасет пригоден для построения исследовательского pipeline и предварительного обучения. Это не просто набор невалидированных depth updates: Bybit и OKX имеют непрерывные sequence, Kraken L2 и L3 полностью воспроизводятся с биржевыми CRC, а ни одна восстановленная книга не стала crossed.

Главное ограничение сейчас не чистота, а объём истории: здесь только около 8 часов одного режима рынка. Этого достаточно для реализации reconstruction, features, labels, симулятора, проверки причинности и controlled overfit. Этого недостаточно для честной оценки торговой стратегии или production-модели.

Итоговая рекомендация: использовать Bybit BTCUSDT linear perpetual как первоначальный execution target, OKX swap как второй derivatives venue, Coinbase и Kraken как spot price-discovery источники, а Kraken L3 как источник queue/order-flow признаков. Нативный Kraken L2 лучше оставить также как независимый integrity oracle для L3-derived L2.

## Инвентарь

Все файлы имеют общую 24-колоночную схему:

```text
exchange, market, symbol, channel, event_type, event_id, row_idx,
local_ts_ns, exchange_ts_ns, engine_ts_ns,
sequence, sequence_start, prev_sequence, is_snapshot,
side, price, quantity, action, order_count, order_id, checksum,
trade_id, taker_side, raw_message
```

Общее покрытие live-потоков: примерно `2026-07-07 07:57:11 UTC` - `15:59:54 UTC`, около 8 часов 3 минут.

| Набор | Размер | Строки | События/пакеты | Назначение |
|---|---:|---:|---:|---|
| Bybit orderbook | 292.34 MiB | 25,072,044 | 144,831 | L2, depth 1000, linear perpetual |
| Bybit trades | 125.89 MiB | 1,219,428 | 367,896 пакетов | Public executions, taker side |
| Coinbase level2 | 236.05 MiB | 12,617,156 | 543,611 | Полный spot L2 |
| Coinbase market_trades | 41.33 MiB | 265,961 raw / 265,854 unique | 94,122 пакета | Spot trades, maker side |
| Kraken book | 249.77 MiB | 8,949,891 | 5,338,185 | Spot L2, depth 1000, CRC |
| Kraken level3 | 420.44 MiB | 10,454,654 | 6,362,367 | Spot order-by-order book, CRC |
| Kraken trade | 17.83 MiB | 16,091 | 9,323 пакета | Spot executions, taker side |
| OKX books | 284.41 MiB | 20,339,538 | 286,789 | L2, depth 400, linear swap |
| OKX trades | 44.10 MiB | 674,185 | 674,185 | Swap executions, taker side |

Всего: `79,608,948` строк, около `13.82 млн` packet/event boundaries. Средняя интенсивность около 478 пакетов/с и 2,750 нормализованных строк/с. В 120-секундном окне получается примерно 57 тысяч пакетов, поэтому один flat Transformer поверх каждого события не является хорошим представлением даже при неограниченном compute.

## Compact head каждого набора

Ниже показаны первые фактические записи. Полностью пустые поля и длинный `raw_message` опущены, но значения не пересчитаны.

```text
Bybit orderbook
  snapshot e=2 r=0 ts=1783411028136000000 bid 63106.7 @ 4.697 action=set u=18170895
  snapshot e=2 r=1 ts=1783411028136000000 bid 63106.6 @ 0.002 action=set u=18170895

Bybit trades
  trade e=1  ts=1783411031163000000 63110.7 @ 0.007 taker=buy  id=9f8f2507-2a8d-5079-8e0e-33c807057128
  trade e=22 ts=1783411032089000000 63110.6 @ 0.002 taker=sell id=17fc065e-e75a-57e4-96c4-4541b208918b

Coinbase level2
  snapshot e=103 r=0 ts=1783411031017968896 bid 63077.78 @ 0.72166449 action=set
  snapshot e=103 r=1 ts=1783411031017968896 bid 63077.77 @ 0.26112069 action=set

Coinbase market_trades
  snapshot e=1 ts=1783411031016343040 63077.78 @ 0.00000014 maker_side=buy id=1051622760
  snapshot e=2 ts=1783411030718328832 63077.78 @ 0.00000004 maker_side=buy id=1051622759

Kraken book
  snapshot e=138 r=0 ts=1783411031597539840 bid 63078.0 @ 0.03167996 action=set checksum=2112536528
  snapshot e=138 r=1 ts=1783411031597539840 bid 63077.9 @ 0.00005100 action=set checksum=2112536528

Kraken level3
  snapshot e=1 r=0 ts=1783411030698332928 bid 63077.8 @ 0.03167996 order=OTPZNL-OVY4S-3D6NPE
  snapshot e=1 r=1 ts=1783411030698332928 bid 63077.7 @ 0.29971773 order=OZO3V4-GZHTL-54ZCP6

Kraken trade
  trade e=4969  ts=1783411042213551104 63063.2 @ 0.01276255 taker=sell id=103405967
  trade e=11506 ts=1783411069819889152 63064.7 @ 0.00001532 taker=buy  id=103405968

OKX books
  snapshot e=2 r=0 ts=1783411031402000000 bid 63106.0 @ 982.27 contracts action=set orders=33 seq=328515903283
  snapshot e=2 r=1 ts=1783411031402000000 bid 63105.9 @ 0.03 contracts action=set orders=2  seq=328515903283

OKX trades
  trade e=1 ts=1783411031658000000 63106.0 @ 13.05 contracts taker=sell id=2768879544
  trade e=6 ts=1783411031843000000 63106.0 @ 13.36 contracts taker=sell id=2768879545
```

## Integrity verdict

### Общая чистота

- 0 null `exchange_ts_ns`.
- 0 неположительных цен.
- 0 отрицательных quantities.
- 0 случаев `local_ts_ns < exchange_ts_ns`.
- 0 нарушений `event_id/row_idx` в book-потоках.
- 0 crossed reconstructed books на Bybit, Coinbase, Kraken и OKX.
- Все snapshots нужно считать границей нового валидного сегмента; книгу при snapshot необходимо полностью очищать.

### Sequence и reconnect

- OKX books: `286,789` событий, 0 нарушений `prevSeqId == previous seqId`.
- Bybit orderbook: `144,831` событий, update id `u` всегда строго `+1`, 0 reset/backward/equal.
- Coinbase L2: один ранний разрыв `sequence_num 2 -> 4`. Коллектор не сохраняет zero-row packets, поэтому нельзя доказать, был это пустой служебный пакет или потерянный update.
- Reconnect snapshots: Bybit 1; Coinbase L2 2; Coinbase trades 2; Kraken L2 6; Kraken L3 5; OKX 3.
- Все derived features должны иметь `segment_id`, `is_warm`, `time_since_snapshot` и availability mask по каждому потоку.

### Полный replay L2

| Venue | Replayed events | Crossed | Missing delete | Финальный BBO | Spread p50 / p95 / p99 |
|---|---:|---:|---:|---|---|
| Bybit | 144,831 | 0 | 0 | 63913.0 / 63913.1 | $0.10 / $0.10 / $0.10 |
| Coinbase | 543,611 | 0 | idempotent deletes присутствуют | 63885.54 / 63885.56 | $0.01 / $0.86 / $2.44 |
| Kraken | 5,338,185 | 0 | 0 | 63871.9 / 63872.0 | $0.10 / $13.00 / $18.20 |
| OKX | 286,789 | 0 | 0 | 63913.4 / 63913.5 | $0.10 / $0.10 / $0.10 |

Kraken L2 проверен не только по BBO: биржевой CRC совпал после каждого из `5,338,185` событий.

Coinbase имеет один sequence-contiguous mass reset: event `614675` удалил всю bid-сторону и почти всю ask-сторону, следующий event `614773` примерно через 2.03 секунды exchange time восстановил полный book. Это не crossed и не sequence loss, но решения в этом one-sided окне нужно запрещать или помечать invalid.

### Полный replay Kraken L3

Проверено `6,362,367` событий и `2,284,376` состояний, в которых изменялся CRC. Итог: `0` CRC mismatch, `0` missing modify, `0` missing delete, `0` duplicate add, `0` crossed.

Точное правило очереди, подтверждённое CRC:

1. Snapshot полностью заменяет книгу.
2. `add` вставляет order в price-level queue по исходному per-order timestamp, а не всегда в хвост. Это важно, когда старый уровень снова входит в subscribed depth.
3. `modify` с `new_qty < old_qty` является fill/reduction и сохраняет queue priority.
4. `modify` с `new_qty >= old_qty` является amend и теряет priority; order переставляется по новому timestamp, обычно в хвост.
5. `delete` удаляет order.
6. После каждого сообщения книга обрезается до 1000 price levels на каждой стороне. Для выпавших уровней отдельный delete не гарантирован.
7. CRC считается по individual orders всех top-10 price levels: asks low-to-high, затем bids high-to-low, с сохранением queue order внутри уровня.

В нормализованной строке нет per-order timestamp. Он пока восстанавливается из `raw_message`; в будущей записи его нужно вынести в отдельную колонку `order_ts_ns`.

### Trades

| Venue | Unique trades | BTC volume | Approx USD notional | Side semantics |
|---|---:|---:|---:|---|
| Bybit | 1,219,428 | 41,352.61 BTC | $2.620B | taker side |
| OKX | 674,185 | 43,639.49 BTC | $2.764B | taker side; stored qty is contracts |
| Coinbase | 265,854 | 4,046.09 BTC | $256.2M | stored side is maker side; invert for aggression |
| Kraken | 16,091 | 924.43 BTC | $58.5M | taker side |

Нормализация OKX: текущая спецификация BTC-USDT-SWAP имеет `ctVal=0.01 BTC`, поэтому `base_qty = contracts * 0.01`. Метаданные инструмента нужно сохранять вместе с каждым recording session, а не полагаться на сегодняшнее значение.

- Coinbase содержит 107 повторных строк после reconnect snapshot. Dedupe: `(exchange, symbol, trade_id)`.
- Kraken trade IDs имеют 5 forward gaps общей величиной 11 IDs. Документация не обещает, что ID является безразрывной per-symbol sequence, поэтому это warning, а не доказанная потеря.
- Bybit trade packets содержат до 1024 trades, что совпадает с лимитом API.
- Coinbase trades агрегированы примерно по 250 ms; сортировка каждой сделки только по exchange time создаст историческую информацию раньше момента получения пакета.

## Исправления normalization перед feature engineering

1. Построить `packet_id` отдельно от `event_id`. Один WebSocket payload является атомарной единицей доступности.
2. Сортировать глобальный causal stream по `local_ts_ns`; `exchange_ts_ns` использовать как feature/event age. Нельзя выпускать содержимое пакета до его receive timestamp.
3. Внутри L2 packet применить все updates в wire order и только затем публиковать состояние книги.
4. Coinbase `offer -> ask`; Coinbase trade `aggressor_side = opposite(maker_side)`.
5. Bybit orderbook `cts` находится на top level payload. Сейчас `engine_ts_ns` полностью null из-за чтения `data.cts`; значение восстанавливается из raw.
6. Сохранить Bybit trade `seq`, OKX trade `seqId`, true packet sequence и channel sequence отдельными полями.
7. Сохранять exchange timestamps как integer из исходной строки без round-trip через Python `datetime.timestamp()`. Сейчас ISO timestamps теряют sub-microsecond fidelity в нормализованной колонке, хотя raw-строка её содержит.
8. Сохранять Kraken `order_ts_ns` и исходную decimal precision. Текущий `raw_message` является повторной JSON-сериализацией уже распарсенного payload, а не исходными wire bytes.
9. Persist zero-row packets, heartbeats и reconnect markers, иначе sequence gap нельзя однозначно классифицировать.
10. Timestamp сообщения нужно ставить сразу после socket receive, а Parquet compression/write вынести с event loop. В текущих данных есть общие collector stalls до 11.8 s.
11. Хранить versioned instrument metadata: tick size, lot size, contract value, fees, funding and product state.

## Derived representation

Не нужно объединять raw Parquet физически. Нужны immutable raw streams и общий derived event index:

```text
raw venue streams
  -> validation + segment boundaries
  -> packet-normalized causal stream
  -> per-venue L2/L3 state machines
  -> decision snapshots + event-flow windows
  -> labels based on executable prices
  -> purged walk-forward datasets
```

Базовая derived event schema:

```text
venue, instrument, channel, packet_id, row_idx,
recv_ts_ns, exchange_msg_ts_ns, event_ts_ns, order_ts_ns,
segment_id, sequence_valid, checksum_valid,
side, action, price_ticks, qty_base, qty_quote, qty_contracts,
order_id, trade_id, aggressor_side
```

Для BTC нельзя строить глобальный абсолютный cent-grid от 0 до максимальной цены. Coinbase snapshot содержит уровни от `$0.01` до `$129,034,888`. Представление должно быть центрировано относительно текущего BBO:

- exact top-N levels, например 64 или 128 на сторону;
- fixed distance bands в ticks/bps от mid;
- отдельные tail aggregates для 10/25/50/100/250/500 bps;
- цены как signed distance, а не absolute dollar level.

## Feature families

### 1. Book state

- spread, relative spread, best sizes, microprice;
- top-N depth и cumulative depth по ticks/bps;
- imbalance на каждом уровне и cumulative imbalance;
- book slope, convexity, curvature, gaps, first empty tick;
- concentration/entropy/Gini/HHI quantities;
- executable VWAP и slippage для нескольких target sizes;
- resiliency: время и путь восстановления depth после shock.

### 2. Event flow

- Cont-style order-flow imbalance по уровням;
- signed add/set/delete flow;
- add/cancel/update intensities на 10/50/100/250 ms, 1/2/5/10/30/60 s;
- velocity, acceleration и change-of-acceleration depth/OFI;
- queue depletion/replenishment bursts;
- event inter-arrival, burstiness, silence/staleness;
- flow conditioned on spread, volatility and current imbalance.

### 3. Trades

- aggressive buy/sell count, size, notional and CVD;
- trade-size quantiles, concentration, block/sweep indicators;
- trade-through depth, number of consumed levels, signed impact;
- realized impact and reversion after 10/50/250 ms, 1/5/30/120 s;
- Kyle-like lambda, Amihud-like impact, toxicity proxies;
- book-trade consistency and replenishment after aggressive flow.

### 4. Kraken L3

- orders per level and priority-weighted depth;
- queue age min/median/max/quantiles, oldest-order share;
- add/cancel/modify rates, survival and cancellation hazard;
- quantity reductions that preserve priority versus amendments that requeue;
- queue position churn, age-weighted imbalance, order-size entropy;
- estimated cancellation versus execution: join L3 reductions/deletes with Kraken trades;
- replenishment, iceberg-like repeated re-add patterns, spoof-like short-lived large orders;
- cohort features by order age, size and distance from BBO.

### 5. Cross-venue

- spot-perpetual basis: Bybit/OKX versus Coinbase/Kraken;
- venue BBO dispersion and consolidated executable BBO;
- lead-lag returns and flow at several causal lags;
- stale quote age and latency-adjusted divergence;
- cross-venue OFI agreement/disagreement;
- price-discovery contribution and shock propagation order;
- venue liquidity share and migration after spread/depth changes.

### 6. High-order families

- tensor derivatives over both time and depth;
- interactions such as `aggressive_flow x opposite_depth`, `OFI x thinness`, `basis x perp_flow`;
- state-dependent Hawkes intensities for add/cancel/trade excitation;
- survival/hazard models for queue depletion and time-to-fill;
- spectral/PCA/eigenbook factors and multiscale wavelet energy;
- sequence motifs: sweep -> refill, cancel wave -> move, spot shock -> perp catch-up;
- regime embeddings for volatility, liquidity, spread and cross-venue coherence;
- learned residual features: prediction error of expected next book/trade event.

Самые важные сначала: executable book state, multiscale OFI, aggressive trades, cross-venue basis/lead-lag, затем L3 queue dynamics. Spectral/Hawkes/motif families следует добавлять только после ablation, иначе они легко создадут красивый in-sample шум.

## Targets и actions

Нельзя обучать market-only 6-class classifier без состояния позиции. Одинаковый рынок требует разных действий для flat, long и short.

Первичный predictor должен выдавать распределения net executable returns:

```text
long_return(h)  = future_bid(h) / current_ask - 1 - fees - slippage
short_return(h) = current_bid / future_ask(h) - 1 - fees - slippage
```

Horizons: `0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120 s`.

Дополнительные heads:

- quantiles/CDF long and short return;
- MFE, MAE and first-passage time to profit/loss barriers;
- future spread, volatility and liquidity;
- fill probability and queue time, если будут passive orders;
- uncertainty/calibration.

Policy получает predictor outputs плюс `position_side`, `entry_price`, `holding_time`, `unrealized_pnl`, fee tier, latency и risk budget. Invalid actions маскируются:

```text
flat  -> OpenLong | OpenShort | Skip
long  -> Hold | CloseLong
short -> Hold | CloseShort
```

Сначала supervised distributional prediction и deterministic execution-aware policy. Offline RL имеет смысл только после появления валидного event-driven simulator и существенно большего числа дней.

## Architecture options

| Вариант | Плюс | Главный минус | Роль |
|---|---|---|---|
| DeepLOB-style CNN/TCN over frames | сильный, устойчивый spatial baseline | теряет packet/L3 detail | обязательный baseline/ensemble member |
| Flat event Transformer | сохраняет raw detail | около 57k packet tokens на 120 s, attention dilution | отклонить |
| Hierarchical multi-stream causal Transformer | сохраняет venue/channel structure и long context | сложнее preprocessing/training | основной кандидат |
| Transformer + state-space long-memory block | длинная память | выше research risk | ablation после baseline |

Рекомендуемая модель:

1. Отдельные venue/channel encoders для Bybit L2+trades, OKX L2+trades, Coinbase L2+trades, Kraken L3+trades.
2. Spatial encoder top-64/128 levels плюс multiscale depth bands. Это может быть compact CNN/MLP/level-attention block.
3. Packet encoder без искусственных intra-packet timestamps. Для unordered L2 deltas допустим set attention; для L3 сохраняется wire/queue order.
4. Local temporal encoder с окнами 10/50/250 ms и event-driven salient tokens.
5. Causal cross-venue fusion с continuous relative-time embedding, venue/instrument/channel embeddings и staleness masks.
6. Hierarchical memory tokens на 1/5/15/30/60/120 s вместо full attention по десяткам тысяч raw events.
7. Multi-task distributional heads и отдельный inventory-conditioned action/value head.
8. Ensemble с DeepLOB/TCN и gradient boosting на engineered features. Если интересует только качество, хороший ensemble предпочтительнее одной большой модели.

Kraken native L2 не нужно подавать как второй равноправный поток рядом с L3-derived L2: это почти дублирование и будет искусственно перевешивать Kraken. Использовать native L2 как integrity/availability channel, а в модель подавать L3-derived aggregate book плюс L3-only queue features.

Полезный pretraining:

- masked packet/event reconstruction;
- next-event type/side/size/time prediction;
- cross-venue contrastive alignment одного рыночного состояния;
- book denoising/reconstruction;
- prediction of future flow, spread and volatility before return fine-tuning.

## Evaluation

- Только chronological purged walk-forward split, purge/embargo минимум 120 s.
- Никаких random rows/windows split.
- Весь preprocessing fit только на train segment.
- Decision time определяется receive time; exchange time не должен открывать будущий пакет раньше получения.
- Backtest использует executable bid/ask, walking the book, fees, latency, slippage and funding.
- Метрики: net PnL, Sharpe/Sortino, drawdown, turnover, hit rate только вторично, calibration, expected shortfall, performance by regime.
- Нынешние 8 часов использовать для pipeline tests, leakage tests, reconstruction tests и intentional overfit. Для выбора модели нужны недели, для устойчивого результата - разные месяцы и режимы.

## References

- Bybit orderbook: https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook
- Bybit public trades: https://bybit-exchange.github.io/docs/v5/websocket/public/trade
- Coinbase WebSocket channels: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-channels
- Kraken L2: https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/book
- Kraken L3: https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/level3
- Kraken L2 checksum: https://docs.kraken.com/exchange/guides/websockets/book-checksum-v2
- Kraken L3 checksum: https://docs.kraken.com/exchange/guides/websockets/l3-checksum-v2
- OKX API: https://www.okx.com/docs-v5/en/
- DeepLOB: https://arxiv.org/abs/1808.03668
- Transformers for Limit Order Books: https://arxiv.org/abs/2003.00130
- TLOB: https://arxiv.org/abs/2502.15757
- State-dependent Hawkes LOB model: https://arxiv.org/abs/1809.08060
