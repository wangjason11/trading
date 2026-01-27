# Market Structure Spec — CTS/BOS/Range/Reversal (through Week 6)

Implementation: `MarketStructure` in `market_structure.py`.【fileciteturn1file0】

This document is meant to be the canonical behavior reference.

---

## Core principles

- **Sequential / event-driven**: process anchor `i`, decide what becomes known at or after `i`, then advance.
- **No skipping**: the engine always moves forward in time; it may “jump” the anchor index to the post-confirm candle for live-like timing, and it may **rewind + replay** only when needed to correct thresholds (primarily range evaluation).
- **Structure IDs**: structure_id partitions regimes; increments on terminal reversal. Zones and charts use structure_id to filter.

---

## State machine overview

MarketStructure maintains an internal state object (MarketStructureState) with:
- `struct_direction` (+1 / -1)
- `structure_id` (regime id)
- CTS lifecycle: `cts_cycle_id`, `cts`, `cts_threshold`, `cts_phase_debug`, `cycle_stage`
- BOS lifecycle: `bos_confirmed`, `bos_threshold`, `bos_event`
- Range: `range_active`, `range_hi`, `range_lo`, `range_start_idx`, `range_confirm_idx`
- Reversal watch: `reversal_watch_active`, `reversal_bos_th_frozen`, and pending reversal fields【fileciteturn1file3】

---

## Definitions

### CTS
A continuation level established within the current structure direction.
- It is confirmed by pullback logic (a valid pullback pattern).
- CTS emits:
  - `CTS_ESTABLISHED` when a new CTS cycle begins (level anchored at an extreme).
  - `CTS_CONFIRMED` when the pullback pattern confirms CTS timing.
  - `CTS_UPDATED` when the CTS extreme is extended within the appropriate stage (rules depend on cycle stage).

### BOS
A break level; confirmed by breakout logic.
- BOS emits `BOS_CONFIRMED` when confirmed.

### Range
A consolidation state bounded by (range_hi, range_lo).
- Can be started by explicit range candidate logic and/or by pullback rules (see below).
- Can expand (update hi/lo) based on later candles and based on pullback pattern window extremes.

### Reversal watch (frozen BOS barrier)
Starts **only** when a candle **close-breaks** the active BOS threshold.
During watch:
- BOS threshold must not update.
- Reversal candidates are detected relative to a frozen barrier.
If no valid reversal pattern appears within watch window:
- BOS threshold updates to the close-break anchor candle wick extreme,
- watch clears,
- execution rewinds to anchor_idx+1 and proceeds normally.【fileciteturn1file3】

---

## Range behavior (canonical rules)

### Range reset vs not reset
- **Breakout**: may reset/deactivate range (depending on whether range was active).
- **Pullback**: **never resets range**.

### Pullback creates range (Week 5 rule update)
If a valid pullback pattern occurs and no range is active, it *creates* a range:
- range starts from the confirmed CTS index (range_start_idx = CTS idx)
- bounds:
  - if struct_direction=+1:
    - range_hi = CTS price
    - range_lo = lowest low of the original pullback pattern window (excluding confirmation candle)
  - if struct_direction=-1:
    - range_lo = CTS price
    - range_hi = highest high of the original pullback pattern window (excluding confirmation candle)

If a range exists, pullback expands it to include the pullback pattern window extremes (does not reset).【fileciteturn2file5】

### Threshold syncing from range
When a range is active:
- `cts_threshold` mirrors the breakout bound:
  - sd=+1: cts_threshold = range_hi
  - sd=-1: cts_threshold = range_lo
- This sync is a first-class event source for zones (CTS threshold updates) and is emitted when cts_threshold changes due to range sync.【fileciteturn2file14】

---

## Pattern application timing

MarketStructure uses the notion of “apply candle”:
- If pattern is SUCCESS: apply_idx = end_idx
- If pattern is CONFIRMED: apply_idx = confirmation_idx【fileciteturn2file5】

The engine advances so that after a successful evaluation:
- it continues from **apply_idx + 1**, which corresponds to “the next candle that has opened but not closed yet” in live time.

---

## Rewind + replay

Range evaluation uses a lookahead window (min/max K) and may need to:
1) mark a candle as range start candidate
2) evaluate later candles to decide true range / breakout
3) correct break_thresholds for pattern evaluation between candidate start and decision

To do this deterministically:
- the engine stores a seed snapshot
- rewinds to range start
- replays forward with corrected thresholds
- returns to “decision point + 1”

This avoids “cheating” while still allowing batch computation.

---

## DF outputs (selected)

MarketStructure writes:
- `market_state` (labels)
- per-row event markers: `cts_event`, `bos_event`
- level info: `cts_idx`, `cts_price`, `bos_idx`, `bos_price`
- `range_active`, `range_hi`, `range_lo`, `range_start_idx`, `range_break_frac`
- `structure_id`, `struct_direction` per row【fileciteturn1file11】

---

## Invariants / guard checks

MarketStructure includes df-level invariant checks (low-noise):
- range_lo must not exceed range_hi while active
- CTS_CONFIRMED coherence with phase/stage
- BOS_CONFIRMED coherence
- reversal is terminal (cannot leave reversal once entered)【fileciteturn1file11】

