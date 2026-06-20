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
- It is confirmed by EITHER a pullback pattern OR sd zone proximity
  (whichever fires first — see "Dual CTS confirmation paths" below).
- CTS emits:
  - `CTS_ESTABLISHED` when a new CTS cycle begins (level anchored at an extreme).
  - `CTS_CONFIRMED` when CTS is confirmed by the first of pullback / proximity.
    `meta["confirmation_method"]` ∈ {`"pullback"`, `"sd_zone_proximity"`}.
  - `CTS_UPDATED` when the CTS extreme is extended within the appropriate stage (rules depend on cycle stage).
  - `CTS_RECONFIRMED` (new) when a valid pullback pattern fires AFTER CTS was
    already confirmed via proximity. The original CTS_CONFIRMED stays at the
    proximity idx; the CTS zone meta is upgraded to `confirmation_method = "pullback"`
    with `pb_reconfirm_idx` recorded.

### Cycle-0 pre-CTS_0 scan-from-start mode (`enforce_cts0_new_extreme`)

> **Repurposed 2026-06-07 (true-first-breakout cycle-0 redesign)** from the
> earlier partial anchor-extreme gate to the full pre-CTS_0 scan-from-start
> mode. Full design: `memory/project_true_first_breakout_cycle0.md`; probe-side
> detail: `PART4_REFACTOR_SPEC.md` §4.4.

`MarketStructure` accepts `enforce_cts0_new_extreme: bool = False` and
`bos0_inner: Optional[float] = None`. When the flag is True (the
**pre-CTS_0 scan mode**), `bos0_inner` is **REQUIRED** (raises otherwise):
while cycle 0 is unestablished MS delegates the entire breakout search to
the shared `find_true_first_breakout` routine (mechanism B against
`bos0_inner` + strict full-pattern new extreme + cycle-0 tie-break),
establishes CTS_0 at the located winner via its NORMAL cycle-0 path, then
resumes. Because the unified probe used the SAME routine with the SAME
`bos0_inner` to decide the start, MS re-finds the identical CTS_0 by
construction (no seed-and-resume). Cycles ≥ 1 are unaffected (subsequent
CTSes break the prior CTS by construction).

Used by `unified_probe` (deterministic method + phase 2) and by
`compute_bounded_structure` for every M15 sub (Commit 1). The same
mechanism is the planned wiring for the deferred main `sid=0 cycle=0` fix
(Commit 2). The old partial-gate helper `_cts0_new_extreme_passes` was
removed.

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

## Dual CTS confirmation paths

After `CTS_ESTABLISHED`, the engine watches for both:
1. **A valid pullback pattern** (existing path, all cycles)
2. **First sd zone proximity hit** (uses BOS inner + active POI inners) — gated by **both**:
   - **Rule 1 (narrow-gap gate):** `|cts_price − bos_threshold| ≥ min_gap_threshold` (per-TF table)
   - **Cycle-0 carve-out:** `cts_cycle_id > 0` (BOS_0's "initial_prior_extreme" status makes it semantically suspect regardless of gap magnitude)

   Both gates must pass for sd-proximity to be eligible to confirm CTS. See "Rule 1: narrow-gap gate" and "Cycle-0 carve-out" sections below for the rationale of each.

### Rule 1: narrow-gap gate

The dual-CTS sd-proximity confirmation path requires `|cts_price −
bos_threshold| ≥ min_gap_threshold` at the candidate candle. The
per-TF threshold (`zones/zone_proximity.py::DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS`)
is:

| Timeframe | min_gap_pips |
|---|---|
| H1 | 50 |
| M15 | 30 |
| M5 | 15 |

**Why:** when the BOS-to-CTS gap is small (< 50 pips on H1), the sd
proximity buffer can overlap the CTS price level itself or fire on
trivially-small retracements — producing spurious confirmations and a
cascade of premature cycle transitions. The 50-pip floor ensures the
V/λ retracement pattern has room to develop before proximity can
substitute for pullback as a confirmation signal.

**Monotonicity:** the gap is monotonically non-decreasing within a
cycle (BOS extends in struct direction via probe; CTS extends via range
sync — both widen the gap). So a cycle can cross the threshold at most
once (narrow → wide), never the reverse. Once wide, the gate opens for
the rest of the cycle.

**Evaluation point:** uses `st.cts.price` (current CTS extreme) and
`st.bos_threshold` (the running BOS level). No self-rescue concern in
the in-MarketStructure check: sd-direction wicks of the candidate
candle wick AWAY from CTS, so they can't extend `cts_threshold` this
candle (extensions happen on struct-dir wicks, the opposite
direction).

**Implication for CTS_RECONFIRMED:** since proximity can't confirm CTS
in narrow cycles, the only way a narrow cycle's CTS gets confirmed is
via pullback (or, in the mid-cycle-crossing case, via proximity AFTER
the gap widens). The `CTS_RECONFIRMED` event — which fires when
proximity confirmed first AND pullback later fires — therefore only
ever appears in wide cycles or after a mid-cycle crossing where
proximity confirmed post-crossing before pullback.

**Range-under-proximity-only path narrows scope similarly:** the
"Option B" range seeding (see below) only fires when proximity confirms
CTS, which after Rule 1 is restricted to wide cycles / post-crossing.

### Cycle-0 carve-out (proximity disabled)

The proximity confirmation path is **skipped on cycle 0** of every
structure_id. Cycle 0's `BOS_0` is the swing extreme that existed
*before* the structure began (`source: "initial_prior_extreme"`), so the
`BOS_0`–`CTS_0` gap is unconstrained — it can be arbitrarily small.
When that gap is smaller than ~2× `proximity_pips`, the sd-inner
proximity buffer overlaps the `CTS_0` price level and the very first
candle after `CTS_ESTABLISHED` trivially fires the trigger, producing a
premature `CTS_CONFIRMED` and a spurious cycle-1 `BOS_CONFIRMED` at the
narrow proximity-only retracement window. Cycles k>0 don't have this
pathology because `BOS_k` is the pullback extreme from cycle k-1, which
guarantees a structurally meaningful gap.

Cycle 0 therefore relies on **pullback confirmation only**. If no
pullback pattern fires before a same-direction breakout, the breakout
calls `_emit_cts_updated` (extending cycle 0's CTS to the new extreme),
not `_emit_cts_established` — keeping the same cycle alive rather than
spawning a phantom cycle 1.

The proximity check picks the closest-to-current-price sd inner across
BOS and POIs. POI inners are refreshed at `CTS_ESTABLISHED` (new cycle)
and at each `CTS_UPDATED` (CTS extended → Fib bounds expand → IC
candidates may shift). The snapshot is per-cycle in
`MarketStructureState.poi_inners_for_cycle`. Both refreshes still run
on cycle 0 (cheap; downstream code reads `bos_inner_for_cycle` /
`poi_inners_for_cycle` for other purposes), but the per-candle check
is gated off.

**Snapshot vs per-candle — deliberate approximation:** The proximity
check uses a per-cycle POI snapshot, NOT a full per-candle activity check.
Refresh points are CTS_ESTABLISHED and CTS_UPDATED only — between those,
POI inners stay fixed. Trade-off: a POI whose IC qualification changes
between two CTS_UPDATED events (e.g., an imbalance fill mid-cycle that
disqualifies an IC candidate) is not reflected immediately by the
snapshot. In practice the deviation is small — POIs typically materialize
at CTS_ESTABLISHED time and don't shift much during the cycle. The
performance cost of full per-candle recomputation (Fib + IC scan + variant
selection) is substantial. The major refactor may revisit this.

Whichever fires first confirms the CTS at that candle's idx:
- `CTS_CONFIRMED.meta["confirmation_method"] = "pullback"` if pullback won
  (or pullback fired at the same candle as the proximity wick)
- `CTS_CONFIRMED.meta["confirmation_method"] = "sd_zone_proximity"` if proximity won

If proximity confirmed first AND a valid pullback later fires, the engine
emits `CTS_RECONFIRMED` at the pullback idx. The original `CTS_CONFIRMED`
event is not modified (append-only contract). KL zone derivation post-pass
upgrades the CTS zone's `confirmation_method` to `"pullback"` and records
`pb_reconfirm_idx`.

### Range under proximity-only confirmation (Option B)

When CTS is confirmed via proximity (no pullback pattern fired yet), the
engine creates a range with bounds seeded by the proximity candle's wick:
- sd=+1: `range_hi = cts.price`, `range_lo = candle.low at proximity_idx`
- sd=-1: `range_lo = cts.price`, `range_hi = candle.high at proximity_idx`

This preserves all downstream behavior (CTS_THRESHOLD_UPDATED via range
sync, breakout detection via range thresholds) for proximity-confirmed
cycles. A subsequent pullback pattern (CTS_RECONFIRMED case) expands the
range as usual via `_ensure_range_on_pullback`.

### BOS_n+1 derivation rule

When transitioning to cycle n+1 via breakout pattern:
- **If cycle n had a pullback** (existing behavior):
  BOS_n+1 = pullback's deepest extreme in `[last_pullback_pat_apply_idx, breakout_apply_idx]`
- **If cycle n was proximity-only** (no pullback fired):
  BOS_n+1 = max retracement in `[cts_n_confirmed_idx, breakout_apply_idx]`

The implementation is in `_select_bos_on_breakout` and switches based on
`pullback_fired_for_cycle` state.

### Trigger threshold per timeframe

`MarketStructure.__init__` accepts `timeframe`, `proximity_pips`, and
`min_gap_pips` parameters. Both pip parameters fall back to per-TF
table lookups in `zones/zone_proximity.py` when None:

| Parameter | Source | H1 | M15 | M5 |
|---|---|---|---|---|
| `proximity_pips` | `DEFAULT_PROXIMITY_PIPS` | 9 | 6 | 3 |
| `min_gap_pips` (Rule 1) | `DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS` | 50 | 30 | 15 |

Pip size derived from `df.attrs["pair"]` (0.01 for JPY pairs, 0.0001
otherwise). The orchestrator wires both lookups via
`structure/structure_engine.py::_make_market_structure`.

The `min_gap_pips` value also feeds Rules 2 & 3 in the post-facto
`zones/zone_proximity.py::check_zone_proximity` scan — see that
module's docstring for narrow-cycle scan semantics.

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
| `compute_structure` | Scenario 1 (auto-identify via `identify_start_scenario_1`) | — | ✓ (reversal start via `unified_probe` + scan-from-start) | H1 main pipeline (orchestrator) |
| `compute_structure_from_start` | Caller-provided | — | ✓ (legacy Scenario 2 + Exc1 + Exc2) | Legacy / tests — no production caller (subs use `compute_bounded_structure`) |
| `compute_structure_scenario_3` | Caller-provided + Phase 1 refinement | ✓ | ✓ if `run_continuation=True` (legacy Scenario 2 + Exc1 + Exc2; gated) | Subordinate probe across all multi-TF variants (counter and confluence; `run_continuation=False`); tests |

**Per-reversal continuation differs by function (Step 4, 2026-06-20).**
`compute_structure` (H1 main) now selects every post-reversal start via the
**`unified_probe` + scan-from-start** path — the SAME primitive the
subordinate reversals use: reference = the prior sid's most recent
`{CONF/UPD/EST}` CTS; the probe runs in the flipped direction over
`[prior-CTS-extreme, reversal apply idx]` and hands back a DECISION (start +
BOS_0 inner), NOT events; the reversed structure's cycle-0 CTS_0 is then
established by a fresh **unbounded** scan-from-start MS run gated on that BOS_0
inner (`enforce_cts0_new_extreme` + `bos0_inner`). This replaced the old
**Scenario 2 → Exception 1 → Exception 2** chain, which now survives only in
`compute_structure_from_start` (no production caller) and
`compute_structure_scenario_3` Phase 2 (test-only).
`identify_start_scenario_2_after_reversal` (incl. its Exception 1) is therefore
no longer reached from the main pipeline; deletion is deferred until those two
legacy callers are retired. The three functions otherwise still differ in
**how the very first start_idx is determined**.

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

**Where used:** Per reversal in the LEGACY continuation only —
`compute_structure_from_start` (no production caller) and
`compute_structure_scenario_3` Phase 2 (test-only). **NOT** in
`compute_structure` (H1 main), which migrated to the `unified_probe` +
scan-from-start reversal handoff in Step 4 (2026-06-20).

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

**Where used:** `compute_structure_scenario_3` Phase 1. In production
invoked by `_run_subordinate_probe` (all multi-TF use cases — first/
subsequent counter and confluence) with `run_continuation=False`.

**Bounds:** `[start_idx, end_idx]` — `end_idx` optional.

**Mechanics:** Run MarketStructure from candidate. After ≥1 `CTS_EST`
fires AND the check window can be bounded (via `end_idx` or
`cts_est[1].idx`), check if any candle in
`[cts_est[0].idx + 1, exc_upper]` reaches the BOS_0 zone inner bound
(within `pip_tolerance_pips`). If so, restart from that reach-back
candle (later than current). `original_bos0_bounds` captured at iteration
0 only and preserved across iterations.

**Exception check window:**
- Lower bound: `cts_est[0].idx + 1` (excludes pullback-confirmation candle)
- Upper bound (`exc_upper`):
  - `end_idx` when defined (supersedes `cts_est[1]` per LANDMINES "Probe
    `end_idx` Is the Supreme Bound" — `end_idx` is a caller-defined hard
    bound; inner rules like "2 CTS_EST" don't narrow it)
  - else `cts_est[1].idx` when ≥2 CTS_EST exist (live-mode fallback)
  - else no upper bound → pending

**Status field — conditions:**

| Condition | Trigger | Status |
|---|---|---|
| 1 | ≥1 CTS_EST + no exception found in `[cts_est[0]+1, exc_upper]` | finalized |
| 2 | Exception triggered (within bounded check window) | (loop continues — no status) |
| 3 | Reversal in probe before 2nd CTS_EST | finalized |
| 4a | No CTS_EST in probe (or no BOS_0 bounds) AND `end_idx is not None` | finalized |
| 4b | No CTS_EST in probe (or no BOS_0 bounds) AND `end_idx is None` | **pending** |
| 4c | 1 CTS_EST + `end_idx is None` (no way to bound check) | **pending** |
| (max iter) | 10 iterations all triggered exception | pending |

**Pending semantics:** Caller may re-invoke with same or advanced
`start_idx` once more data arrives. `_run_subordinate_probe` returns
`None` on pending so the sub isn't built for that trigger. Pending path
is dormant in current backtest (all triggers pass `end_idx` definitively
— CTS_CONFIRMED idx for first_confluence, first sd zone-proximity
trigger candle for first_counter, etc.).

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
`_run_subordinate_probe` calls it, probe-only). Worth knowing if a future
feature needs multi-structure continuation from an arbitrary validated
start — the path exists.

### Common probe patterns

- **Always on `df.copy()`** — no mutation of outer state until result accepted
- **Max iterations cap** (10) — prevents infinite loops
- **`CTS_EST + 1` scan window start** — excludes the pullback-confirmation candle (naturally near the zone, would cause false exceptions)
- **Pip tolerance scales with timeframe** — values from `zones/zone_proximity.py::DEFAULT_PROBE_RESET_PIPS` (H1=3, M15=2.5, M5=2; type is `float` because M15 is fractional). Used by `compute_structure_scenario_3` Phase 1 probe AND by the legacy Exception 2 probes (`compute_structure_from_start` + `compute_structure_scenario_3` Phase 2; `compute_structure` no longer runs Exception 2 after Step 4 — its H1-main reversal handoff uses the `unified_probe` reset tolerances from the same table). Invariant: `DEFAULT_PROBE_RESET_PIPS[tf] < DEFAULT_PROXIMITY_PIPS[tf]` per TF (asserted at module load).【fileciteturn1file11】

