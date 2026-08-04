# cTrader Order Book Reconstruction Report

## Context

We started with a cTrader depth parquet where many rows had identical timestamps. The original rows looked like flattened depth quote changes:

```text
timestamp
quote_id
bid
ask
size
type        # new / deleted
```

The working assumption at the beginning was:

- `new` opens/adds a depth quote.
- `deleted` removes/closes a depth quote.
- `quote_id` is the lifecycle key.
- exactly one of `bid` or `ask` is non-zero for each row.
- raw cTrader price scale is `price / 100000`.
- raw cTrader size scale is `size / 100`.

The main goal was to reconstruct, for every row:

- corrected timestamp
- best bid
- best ask
- spread
- side
- human price
- cent-level active volume after the row

## Timestamp Fix

Problem: rows arrived in batches, and every row in the same arrival batch got the same timestamp.

Fix: spread rows inside each equal-timestamp batch uniformly over the available time interval between neighboring batches.

Important constraint: intervals were not stable, so the interval had to be recomputed per packet.

Result:

- row count stayed unchanged: `134,598,767`
- timestamps became unique / strictly increasing
- output at that stage was the adjusted parquet with the same raw depth rows but fixed timestamps

This fixed row ordering for modeling, but it did not create true exchange event boundaries.

## Phantom Quote Check

We checked whether a quote_id could receive `new` again without a prior `deleted`.

Counts from the old check:

```text
new rows:     67,300,083
deleted rows: 67,298,684
```

Historical `new -> new` without explicit `deleted` was found for only `8` quote_ids:

```text
349925490
349925491
349925492
349925493
349925494
349925495
349925497
349925498
```

Conclusion at that point: phantom quote reuse existed, but it was tiny relative to the dataset size.

## Initial Book Reconstruction

The first reconstruction used active quote state:

- on `new`: add quote_id to active book
- on `deleted`: remove quote_id from active book
- best bid = max active bid price
- best ask = min active ask price
- spread = ask - bid

This produced useful BBO columns, but crossed states appeared in historical data.

A crossed book means:

```text
best_bid > best_ask
```

Example from the first historical cross:

```text
before:
  best_bid = 4405.33
  best_ask = 4405.46

after:
  best_bid = 4405.87
  best_ask = 4405.49
  spread   = -0.38
```

The crossed state was caused by an old ask at `4405.49` remaining active while newer bids moved above it.

## Bid / Ask Orientation Check

We explicitly tested the hypothesis that the parquet columns were physically swapped:

```text
dataset column bid = real ask
dataset column ask = real bid
```

That hypothesis failed.

First packet example:

```text
as labeled:
  best bid = 4427.32
  best ask = 4427.48
  spread   = +0.16

swapped:
  best bid = 4429.72
  best ask = 4424.37
  spread   = -5.35
```

On the first `2,000,000` rows, with the same packet-ordering logic:

```text
as_labeled:
  events:           106,360
  crossed resets:   2
  crossed pct:      0.00188%

swapped:
  events:           106,360
  crossed resets:   106,262
  crossed pct:      ~99.91%
```

Conclusion: the bid/ask columns were not swapped.

## Live cTrader Verification

We wrote a live cTrader connector from scratch and compared live reconstructed depth BBO against live cTrader spot prices.

Live BTCUSD run duration:

```text
600 seconds
```

Results:

```text
depth_events:     1,507
new_quotes:       10,138
deleted_quotes:   10,128
unknown_deletes:  0
cross_resets:     0
comparisons:      2,470
exact_matches:    1,504
mismatches:       966
```

The important split:

```text
depth-trigger comparisons:
  comparisons: 1,503
  exact:       1,503
  mismatch:   0

spot-trigger comparisons:
  comparisons: 967
  exact:       1
  mismatch:   966
```

Interpretation:

- when a depth event arrived, reconstructed depth BBO matched live spot exactly.
- spot-trigger mismatches were caused by spot updates arriving before the corresponding depth event.
- live cTrader depth reconstruction logic was correct when using real `ProtoOADepthEvent` boundaries.

This was the strongest evidence that the basic cTrader reconstruction method was valid.

## Cent-Level Volume Feature

We then added active volume per price level with `1 cent` granularity.

Desired semantics:

- for each `new`, `level_volume` = active volume on that side/price after opening the quote
- for each `deleted`, `level_volume` = remaining active volume on that side/price after deleting the quote

First attempt was wrong:

- it accumulated by side/price too naively
- stale humps stayed in the animation
- old levels looked active even after they should have disappeared

Second attempt used quote-id active state:

- active quote_id -> side, price, size
- level volume derived only from currently active quote_ids
- size scaled as `size / 100`
- price scaled as `price / 100000`

Final level file schema at that stage:

```text
timestamp
quote_id
bid
ask
size
volume
type
bid_price
ask_price
spread
side
price
price_cent
level_volume_raw
level_volume
```

Removed columns:

```text
midprice
level_price
```

Final rebuilt stats:

```text
rows:             134,598,767
events:           7,707,997
unknown_delete:   146
phantom_replace:  0
bad_new_side:     0
bad_lookup:       0
prior_splits:     168
cross resets:     119
active_quotes_end: 0
```

## Crossed Book Handling

We tried several strategies.

### Full Reset On Cross

If reconstructed book became crossed:

```text
best_bid > best_ask
```

we cleared active quote state and replayed the current logical event.

This produced a clean BBO:

```text
valid_bbo rows:        134,598,153
bid_price > ask_price: 0
negative_spread:       0
spread min/max:        0.0 / 5.0
```

But this is a repair heuristic, not ground truth.

It proves:

```text
the reconstructed state became inconsistent
```

It does not prove:

```text
every removed quote was explicitly deleted by the exchange
```

### Ignore Incoming Cross-Making Quote

We tested a third idea:

> if a new quote makes the book crossed, ignore only that quote and keep going.

This failed badly.

On the first `10,000,000` rows:

```text
accepted_new: 2,588,621
ignored_new:  2,411,422
ignored pct:  48.23%
cross events after strategy: 0
BBO got stuck around: 4405.33 / 4405.49
```

Reason:

- a stale ask remained in the book
- every later valid bid above that stale ask was rejected
- the book became artificially clean but dead

Conclusion: ignoring the incoming quote is worse than reset.

### Better Clean-Book Idea

A better clean-book heuristic would be:

- if a new bid crosses old asks, quarantine/remove only old asks below that bid
- if a new ask crosses old bids, quarantine/remove only old bids above that ask
- mark those removals as inferred, not real deletes

This was discussed but not implemented in the final parquet.

## Visualization Findings

We generated 3D and 2D visualizations for a random 60-second window.

Window used:

```text
start: 2026-05-21T15:34:22.969894180+00:00
end:   2026-05-21T15:35:22.969894180+00:00
```

Initial animation showed stale “poles/humps”.

Root cause:

- visualization was based on touched levels / stale cumulative state
- not full active book replay

Fix:

- replay full active quote-id book up to `t=0`
- apply logical events atomically
- render bid and ask separately

After the quote-state fix, the 60-second window had:

```text
events: 3,258

BBO:
  bid_price: 4506.58 .. 4508.44
  ask_price: 4506.71 .. 4508.60

positive level rows:
  ask count: 821
  ask range: 4506.71 .. 4511.91
  bid count: 809
  bid range: 4502.27 .. 4508.44

max level volume:
  ask: 50.0
  bid: 50.0
```

Full active-book animation metadata:

```text
resets_inside_window: 0
price_min:            4502.27
price_max:            4511.91
price_cent_count:     965
max_bid_volume:       50.0
max_ask_volume:       50.0
nonzero_bid_cells:    615
nonzero_ask_cells:    607
```

## Critical Discovery About Recorded Parquet

The strongest later finding was that the recorded historical parquet is not equivalent to raw `ProtoOADepthEvent` messages.

We found that one identical timestamp could contain several logical depth events in sequence:

```text
new
deleted
new
deleted
```

In one example, the second `deleted` block removed quote_ids created in the first `new` block under the same timestamp.

Therefore:

```text
same timestamp != one cTrader DepthEvent
```

This matters because order book reconstruction depends on event boundaries.

Live reconstruction worked because we consumed actual API events.

Historical reconstruction was weaker because the parquet had flattened rows with arrival timestamps, not guaranteed original message boundaries.

## Final Conclusions

1. The basic cTrader depth logic is:

```text
new quote -> add quote_id to active book
deleted quote_id -> remove quote_id
best_bid -> max active bid
best_ask -> min active ask
```

2. Bid/ask columns were not swapped.

3. Live cTrader depth reconstruction matched live spot exactly on depth-triggered comparisons.

4. The recorded parquet is not a perfect source for reconstructing the real book because original event boundaries appear to be lost or flattened.

5. Crossed historical states are not proof that bid/ask is wrong. They are evidence that reconstructed historical state lost synchronization somewhere.

6. Full reset on cross gives a clean book, but it is a heuristic.

7. Ignoring the incoming quote that causes a cross is worse; it can freeze the book behind stale blockers.

8. The most honest representation would keep two layers:

```text
raw explicit lifecycle:
  only explicit new/deleted, crossed states allowed and flagged

clean inferred book:
  repaired state with flags like reset_inferred / removed_inferred_cross / book_epoch
```

9. For modeling, the safest training unit is not an individual row. It should be based on event/message boundaries:

```text
book_before_event
event_updates
book_after_event
target_after_horizon
```

If event boundaries are unavailable, any reconstruction from this historical parquet must be treated as inferred rather than ground truth.

