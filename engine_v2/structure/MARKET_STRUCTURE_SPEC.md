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
- reversal is terminal (cannot leave reversal once entered)【fileciteturn1file11】

---

## Compute_structure variants (orchestration layer)

`MarketStructure` is the underlying state-machine engine. Three orchestration
functions wrap it for different start-identification strategies:

| Function | Initial start source | Phase 1 BOS_0 probe | Multi-structure continuation | Use case |
|---|---|---|---|---|
| `compute_structure` | Scenario 1 (auto-identify via `identify_start_scenario_1`) | — | ✓ | H1 main pipeline (orchestrator) |
| `compute_structure_from_start` | Caller-provided | — | ✓ | UC1 M15 (start pre-validated by H1 reverse probe) |
| `compute_structure_scenario_3` | Caller-provided + Phase 1 refinement | ✓ | ✓ if `run_continuation=True` (gated) | UC1 H1 reverse probe (`run_continuation=False`); tests |

All three share the same per-reversal continuation logic (Scenario 2 →
Exception 1 → Exception 2 probes). They differ only in **how the very first
start_idx is determined**.

### Starting-point rigor hierarchy

```
Lowest:  Caller picks start, no validation
         → compute_structure_from_start
Medium:  Auto-identify via Scenario 1 (lookback search)
         → compute_structure
Highest: Caller picks candidate + iterative probe validates/refines
         → compute_structure_scenario_3 (Phase 1)
```

---

## Probes

Two distinct probe mechanics exist in the structure layer. They share
common patterns (run on `df.copy()`, iterative with max cap, scan window
starts at `CTS_EST + 1`) but answer different questions.

### Probe type 1 — Exception 2 probe

**Question:** "Is the next-structure start (chosen by Exception 1) actually
a structural start, or just a pullback candle?"

**Where used:** Per reversal in all three `compute_structure*` variants
(main pipeline, M15 pipeline, Scenario 3 Phase 2).

**Bounds:** `[exc2_candidate, reversal_confirmed_idx]` — bounded probe.

**Mechanics:** Run MarketStructure from candidate up to reversal. If a
CTS_EST fires in the probe AND any candle between `CTS_EST + 1` and
`reversal_confirmed_idx` reaches the prior CTS zone (within 10 pip
tolerance), exception triggers. New candidate = the reach-back candle
(strictly **later** than the previous candidate). Iterate up to 10 times.

**Outcomes:**
- **Exception triggered (any iter):** discard ALL probes, use settled
  candidate as start_idx, run **unbounded** MarketStructure from there in
  the outer loop.
- **No exception ever:** keep first probe's data (events, levels, df),
  continue outer loop from `reversal_confirmed_idx + 1`. Sid N+1 ends up
  split across two MarketStructure invocations — the bounded probe
  (pre-reversal portion) and the post-reversal continuation.

No status field — outcome encoded as boolean `exc2_triggered`.

### Probe type 2 — Scenario 3 BOS_0 probe (Phase 1 of `compute_structure_scenario_3`)

**Question:** "Is the arbitrary candidate start a true cycle-0 point, or
is it part of an older still-extending structure?"

**Where used:** `compute_structure_scenario_3` Phase 1. Currently invoked
in production only by `_run_h1_reverse_probe` (UC1 multi-TF) with
`run_continuation=False`.

**Bounds:** `[start_idx, end_idx]` — `end_idx` optional.

**Mechanics:** Run MarketStructure from candidate. After 2 CTS_EST fire
in the probe, check if any candle between `cts_est[0].idx + 1` and
`cts_est[1].idx` reaches the BOS_0 zone inner bound (within
`pip_tolerance_pips`). If so, restart from that reach-back candle (later
than current). `original_bos0_bounds` captured at iteration 0 only and
preserved across iterations.

**Status field — four conditions:**

| Condition | Trigger | Status |
|---|---|---|
| 1 | 2 CTS_EST + no exception | finalized |
| 2 | Exception triggered | (loop continues — no status) |
| 3 | Reversal in probe before 2nd CTS_EST | finalized |
| 4a | `end_idx is not None` AND probe reached it without 2 CTS_EST | finalized |
| 4b | `end_idx is None` AND probe ran out of df data | **pending** |
| (max iter) | 10 iterations all triggered exception | pending |

**Pending semantics:** Caller may re-invoke with same or advanced
`start_idx` once more data arrives. `_run_h1_reverse_probe` returns
`None` on pending so M15 isn't built for that trigger. Pending path is
dormant in current UC1 backtest (always passes `end_idx=proximity_trigger_idx`).

### Phase 1 vs Phase 2 (Scenario 3 only)

- **Phase 1** = the BOS_0 probe loop above. Validates `start_idx`.
  Returns `Scenario3Result` with status, validated start, probe data.
- **Phase 2** = multi-structure continuation from the validated start —
  identical mechanics to `compute_structure`'s post-reversal handling
  (Exception 2 probe per reversal). Gated by `status == "finalized" AND
  run_continuation=True`.

**Currently unused in production:** Phase 2 is exercised only by
`tests/test_scenario3.py`. All production callers of
`compute_structure_scenario_3` pass `run_continuation=False` (only
`_run_h1_reverse_probe` calls it, probe-only). Worth knowing if a future
feature needs multi-structure continuation from an arbitrary validated
start — the path exists.

### Common probe patterns

- **Always on `df.copy()`** — no mutation of outer state until result accepted
- **Max iterations cap** (10) — prevents infinite loops
- **`CTS_EST + 1` scan window start** — excludes the pullback-confirmation candle (naturally near the zone, would cause false exceptions)
- **Pip tolerance scales with timeframe** — H1=10, M15=3, M5=1 (Scenario 3); Exception 2 always 10 pips on H1 main, scaled in `compute_structure_scenario_3` Phase 2 via `pip_tolerance_pips`【fileciteturn1file11】

