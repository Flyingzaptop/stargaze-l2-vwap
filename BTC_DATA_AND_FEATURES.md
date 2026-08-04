# BTC data: что собирается и что из этого строить

## 1. Общая организация

Recorder пишет каждый сырой поток в отдельный Parquet по ключу
`exchange / market / symbol / channel`. В одну сырую книгу данные разных бирж
не смешиваются: у площадок разные matching engines, тик-сайзы, единицы объема,
задержки и правила sequence.

Все строки приведены к общей схеме. В ней есть:

- идентичность источника: `exchange`, `market`, `symbol`, `channel`, `event_type`;
- порядок события: `event_id`, `row_idx`, `sequence`, `sequence_start`,
  `prev_sequence`, `is_snapshot`;
- три времени: локальное получение, exchange timestamp и engine timestamp;
- книга: `side`, `price`, `quantity`, `action`, `order_count`, `order_id`, `checksum`;
- сделки: `trade_id`, `taker_side`;
- деривативы: mark/index/oracle, open interest, funding и ликвидации;
- исходное сообщение `raw_message` для аудита парсера.

`event_id + row_idx` сохраняют пакетную структуру. Нельзя независимо сортировать
строки одного update по искусственно разнесенному времени: update применяется
атомарно в нативном порядке конкретной биржи.

## 2. L2 order book

### Источники

- Binance Spot и Binance USDT-M Futures;
- OKX BTC-USDT-SWAP;
- Bybit BTCUSDT Linear;
- Coinbase BTC-USD Spot;
- Kraken BTC/USD Spot;
- Deribit BTC-PERPETUAL;
- Bitfinex BTC/USD Spot;
- Hyperliquid BTC Perpetual.

L2 хранит агрегированный объем на ценовом уровне. Snapshot создает начальное
состояние, incremental updates меняют или удаляют уровни. Для Binance отдельно
проверяется непрерывность sequence между REST snapshot и WebSocket diff.

### Польза для торговли

L2 показывает доступную ликвидность, приблизительную цену немедленного
исполнения и локальный дисбаланс спроса/предложения. Несколько площадок дают
информацию о price discovery: движение часто начинается не одновременно.

### Базовые признаки

- best bid, best ask, mid, spread в USD, ticks и basis points;
- microprice и отклонение microprice от mid;
- объем bid/ask в первых N уровнях и в полосах 1/2/5/10/25/50/100 bps;
- depth imbalance `(bid_depth - ask_depth) / total_depth`;
- order-flow imbalance: добавления, снятия и изменения объема по сторонам;
- расстояние до ближайшего крупного уровня, gap и liquidity vacuum;
- slope, convexity и curvature каждой стороны книги;
- концентрация объема: top-level share, entropy, HHI/Gini;
- ожидаемый slippage для фиксированного BTC- и USD-notional;
- скорость replenishment и время восстановления глубины после сделки;
- age/staleness книги и флаг sequence gap.

## 3. Kraken L3

L3 содержит отдельные resting orders и `order_id`, а не только сумму на уровне.
Это позволяет видеть жизненный цикл заявки: появление, изменение и удаление.

### Признаки L3

- число заявок, средний/медианный размер и распределение размеров на уровне;
- arrival/cancel rate и отношение cancel-to-add;
- lifetime и survival/hazard заявок;
- queue churn и оценка скорости продвижения очереди;
- доля старого и нового объема на уровне;
- replenishment одним участником или серией новых order IDs;
- краткоживущие крупные уровни как наблюдаемый паттерн, без утверждения о
  намерении участника;
- divergence L3 pressure Kraken против L2 imbalance остальных площадок.

L3 Kraken нельзя буквально переносить в очередь Binance. Он используется как
внешний сигнал состояния участников и ликвидности.

## 4. Public trades

Trades записываются с каждой доступной площадки. `taker_side` означает сторону
агрессора: buyer-initiated trade снимает ask, seller-initiated снимает bid.

### Признаки сделок

- signed volume, cumulative volume delta и buy/sell imbalance;
- trades/sec, volume/sec и inter-arrival time;
- VWAP и отклонение сделки от локального mid;
- квантили размера, large-trade indicator и кластеризация крупных сделок;
- burst intensity, acceleration и смена направления потока;
- trade-through нескольких уровней и consumed depth;
- immediate impact, impact decay и permanent/transient component;
- realized spread и adverse selection после 100ms/1s/5s/30s;
- Kyle lambda, Amihud-like illiquidity и VPIN-like toxicity;
- lead-lag агрессивного потока между биржами.

## 5. Деривативный контекст

Binance Futures, Bybit, Deribit и Hyperliquid дают часть следующего набора:
mark price, index price, oracle price, funding rate, open interest и время
следующего funding. Наличие конкретного поля зависит от площадки и канала.

### Признаки деривативов

- perp basis: `(perp - spot_consensus) / spot_consensus`;
- mark-index и oracle-index divergence;
- funding level, momentum, z-score и dispersion между площадками;
- изменение open interest в BTC и USD;
- совместные режимы return x delta-OI как эвристика открытия и закрытия риска;
- crowding score из basis, funding, OI и directional trade flow;
- time-to-funding и поведение потока около funding timestamp;
- расхождение mark/last/microprice как индикатор напряжения книги.

## 6. Liquidations

Binance Futures и Bybit передают forced-liquidation events. Это редкий поток,
поэтому отсутствие строк в коротком окне нормально.

### Признаки ликвидаций

- signed liquidation quantity и USD notional;
- long/short liquidation imbalance;
- события/сек, notional/сек, acceleration и cascade duration;
- расстояние liquidation price до локального mid;
- ликвидации относительно доступной глубины по направлению движения;
- contagion lag между площадками;
- post-liquidation continuation/reversal и время восстановления spread/depth.

## 7. Cross-venue признаки

Сначала каждая книга независимо восстанавливается и валидируется. Затем все
состояния сдвигаются только назад к последнему известному на receive time.

- consolidated best bid/ask и executable NBBO с учетом комиссий;
- robust spot consensus и robust perpetual consensus;
- dispersion mid/microprice между площадками;
- отклонение каждой биржи от consensus и скорость возврата;
- pairwise lead-lag на горизонтах 10ms-10s;
- stale-venue score по времени последнего update и отклонению цены;
- cross-venue OFI/trade-flow agreement или disagreement;
- spot-perp basis по каждой площадке и агрегированный basis;
- liquidity migration: где глубина исчезает и где появляется;
- graph features: биржи как узлы, lead-lag/liquidity transfer как ребра.

## 8. Higher-order feature families

Все базовые признаки считаются на нескольких causal scales, например 100ms,
250ms, 500ms, 1s, 2s, 5s, 10s, 30s, 60s и 120s.

### Динамика

- level, delta, velocity, acceleration, EWMA и rolling z-score;
- realized volatility, jump score и volatility-of-volatility;
- event-time и wall-clock представления одновременно;
- Hawkes-like intensity для trades/adds/cancels/liquidations;
- impulse-response kernels: событие -> impact -> resilience.

### Взаимодействия

- aggressive flow x opposite-side thinness;
- OFI x spread x short-term volatility;
- cancellation pressure x trade aggression;
- liquidation intensity x basis x delta-OI;
- funding crowding x book imbalance;
- microprice divergence x venue lead-lag confidence;
- L3 queue churn x cross-venue L2 migration.

### Структурные представления

- price-distance x time tensors для bid и ask отдельно;
- venue x time x feature tensors с маской stale/missing;
- learned exchange embeddings и market-type embeddings;
- regime state: liquidity, volatility, toxicity, trend и deleveraging.

## 9. Нормализация и causal integrity

До объединения необходимо:

1. Применить snapshot/delta по нативным sequence rules каждой биржи.
2. Сохранить атомарность пакетов и порядок `event_id`, `row_idx`.
3. Перевести количество в BTC и USD notional с учетом contract multiplier.
4. Привести расстояния цены к ticks/bps, не только к абсолютным USD.
5. Использовать receive time как основной causal clock; exchange time оставить
   признаком задержки и средством диагностики.
6. Никогда не делать forward-fill из будущего и не интерполировать неизвестное
   состояние через sequence gap.
7. Добавлять masks для отсутствующих, stale и поврежденных данных.

## 10. Что подавать модели и что предсказывать

Для качества имеет смысл оставить отдельные ветки encoder:

- Binance execution state;
- остальные L2 books;
- trades/event flow;
- Kraken L3;
- derivatives/liquidations.

Их causal representations объединяются cross-attention/venue attention. Так
модель не теряет происхождение сигнала и может научиться, какой источник ведет
в конкретном режиме.

Практичные targets:

- multihorizon forward score для направления и размера будущего движения;
- multihorizon backward/local-peak score для выхода;
- future executable return Binance после spread, fee и slippage;
- future adverse excursion и favorable excursion;
- future liquidity/spread/slippage как execution-risk auxiliary targets;
- future aggressive flow, volatility и liquidation intensity как auxiliary
  targets для более устойчивого representation learning.

Финальный policy head переводит состояние позиции и causal representation в
`OpenLong`, `OpenShort`, `CloseLong`, `CloseShort`, `Hold` или `Skip`. Для
реальной оценки главный PnL target должен использовать исполнимые Binance bid/ask,
а не abstract mid или лучшую цену другой биржи.
