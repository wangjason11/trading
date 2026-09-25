# Architecture & System Design (through Week 8)

This doc explains the “shape” of the system so a new engineer can extend it without breaking project invariants.

---

## Design goals

1. **Explainable decisions**
   - Every trade-relevant claim must be backed by (a) dataframe columns, and (b) emitted events.
2. **Deterministic replay**
   - Given the same input candles, the full pipeline should produce identical outputs.
3. **Event-driven composition**
   - Each stage produces explicit outputs that downstream stages consume.
4. **Visualization-first**
   - The canonical debugging tool is the replay chart.

---

## Execution model

### 1) Batch / Replay (today)
- We simulate “live timing” by:
  - Computing features/patterns on the full df
  - Emitting structure events at the candle index where they would have become known
  - Using rewinds only when thresholds are known after a lookahead window

The MarketStructure engine is explicitly sequential and uses internal rewind/replay logic when it must evaluate ranges with corrected thresholds.【fileciteturn1file0】

### 2) Live (future)
- Same logic should be usable incrementally:
  - Candle features per new candle
  - Pattern detection per new candle
  - Market structure update per new candle
  - Zones updated by structure events (no additional rewinds/waits)

#### Stateless function + caller-managed state pattern (live-mode prep)

When adding components that may run repeatedly as new candles arrive,
prefer **stateless functions with explicit pending/finalized status** over
stateful tracker classes that internalize iteration state. Callers manage
the small amount of state they need (e.g., the current best `start_idx`)
externally and re-invoke the function per new candle.

**Example (already in place):** `compute_structure_scenario_3` Phase 1 —
Condition 4 splits on `end_idx`:
- `end_idx` defined → `finalized` (caller's bound is a real terminal)
- `end_idx is None` → `pending` (more candles may resolve later)

A live caller re-invokes the probe with the same or advanced `start_idx`
each new candle, and uses status to decide whether to start downstream
work. No tracker class needed. New similar features should follow this
pattern unless there's a strong reason to encapsulate state in a class.

---

## Data model contracts

### The dataframe is the shared “truth”
Each stage:
- Adds well-scoped columns (avoid overwriting unrelated columns)
- Optionally writes debug columns (suffix `_debug` recommended)
- Leaves earlier columns intact

### Event contracts
Downstream components must rely on events over inference. An `event.meta` key is renamed, or the meaning of `ev.idx` /
`ev.price` / a meta key changed, only by an atomic migration — LANDMINES "Event Contract Rules" rule 3 (type names
are never changed and meta fields never removed: rules 1–2).

#### PatternEvent
Produced by structure patterns:
- `name`: continuous / double_maru / one_maru_continuous / one_maru_opposite
- `status`: SUCCESS / CONFIRMED / FAIL_NEEDS_CONFIRM
- `start_idx`, `end_idx`, `confirmation_idx` (if confirmed)
- `confirmation_threshold` (for confirmation lookahead)
- `break_threshold_used` (range or BOS/CTS thresholds)

Pattern priority rules are defined in BreakoutPatterns.【fileciteturn2file7】

#### StructureEvent
Produced by MarketStructure:
- `category`: STRUCTURE / RANGE / (etc)
- `type` examples:
  - `CTS_ESTABLISHED`, `CTS_CONFIRMED`, `CTS_UPDATED`
  - `BOS_CONFIRMED`
  - `RANGE_STARTED`, `RANGE_UPDATED`, `RANGE_RESET`
  - threshold events such as `CTS_THRESHOLD_UPDATED` (used for zones)

**`ev.idx` convention — IMPORTANT (the CANONICAL per-event field table; every other doc points here).**
`ev.idx` is **not** uniformly "when the event is known". For the structural events below it is the
**price location** (a historical, element-definition field); the candle at which the event becomes
knowable is a separate meta field. In the naming standard (GLOSSARY "Naming Standard") that location
is the element's **anchor** (a structure endpoint); the knowable candle is its **moment**. Timing / lifecycle reads (scan windows, lifecycle starts/ends,
activation gates, "knowable-at" clips) must use the **moment** column, never `ev.idx` — the
"real-time vs historical: never mix" rule. Verified against the emitters in
`structure/market_structure.py` and live on the reference window (zones-pass audit, 2026-09-22; the
`CTS_THRESHOLD_UPDATED` row added by Plan F, 2026-09-24, verified against its emitter).

| Event | `ev.idx` (price location — for CTS/BOS the element's anchor) | The moment (knowable at) | Other index meta |
|---|---|---|---|
| `CTS_ESTABLISHED` | the new cycle's CTS **anchor**: the breakout pattern's extreme candle — the FIRST argmax(`h`) (sd +1) / argmin(`l`) (sd −1) over the winning pattern's span `[start_idx, max(end_idx, confirmation_idx)]` (`_cts_from_breakout_event`) — adopted as the cycle's CTS; `ev.price` = that high/low. When the anchor precedes the apply candle it is retro-stamped: no CTS event is emitted when the anchor candle closes (raw CTS updates are off during the anchor→apply back-fill of an establishing cycle; the back-fill still runs `_bos_barrier_step` and the reversal-watch checks, which can emit their own events) | `meta["confirmed_at"]` = the pattern's apply candle (`end_idx` on SUCCESS, `confirmation_idx` on CONFIRMED). Equals the same cycle's `BOS_CONFIRMED.meta["confirmed_at"]` **by construction** (one `apply_idx`, one emission block; `multitf/parent_tables.py` asserts it) | `meta["pattern_anchor_idx"]` = the breakout pattern's FIRST candle (`ev.start_idx`, the MS scan candle) — the PATTERN-realm anchor (renamed from `anchor_idx`, Plan E E1), **not necessarily the CTS anchor** (the two coincide when the pattern's first candle holds the extreme; 0 of 34 `CTS_ESTABLISHED` rows on the reference window) and never a timing value. `meta["cts_anchor_idx"]` = the CTS anchor (Plan E E2a, 2026-09-24; additive, contract rule 4) — `== ev.idx` until Plan E E4a flips `ev.idx` to the moment; read it through `event_fields.cts_anchor_idx` |
| `BOS_CONFIRMED` | the BOS **anchor**: the extreme found by the BOS search (cycle 0: the extreme in `[start, cts_idx−1]`; cycle ≥ 1: the retracement extreme) — `<= confirmed_at` on the normal path (34/34 on the reference window) but NOT asserted in code: `_select_bos_on_breakout` swaps a reversed `[window_start, apply]` window, which would put the search past the apply candle | `meta["confirmed_at"]` (the same `apply_idx` as the cycle's `CTS_ESTABLISHED`; it is only ever emitted together with one) | `meta["bos_anchor_idx"]` = the BOS anchor (Plan E E2a; additive) — `== ev.idx` until Plan E E4b; read through `event_fields.bos_anchor_idx`. MS state `st.bos` (was `st.bos_confirmed`) is built from it, not from `ev.idx` |
| `CTS_UPDATED` | the new CTS anchor. Raw path (`meta["via"] == CTS_UPDATED_RAW_VIA`, `"replay_raw"` — the constant in `structure/event_fields.py`, one definition of "raw path"): the processed candle whose wick made a new high/low — knowable at its close. Pattern path (`via` = a pattern name; a breakout while the cycle is unconfirmed): the pattern's extreme candle, which can precede the pattern's apply candle; when it precedes the apply candle and extends the extreme, a raw-path `CTS_UPDATED` at the same idx/price was already emitted at that candle's close (raw updates stay on during a non-establishing back-fill) — 1 of 37 pattern-path rows on the reference window (confluence sub 2, 2468, a same-price duplicate) | raw path: `ev.idx`. Pattern path: the apply candle, which is **NOT recorded** (no `confirmed_at` on any `CTS_UPDATED`; `event_moment` returns `None`) | — |
| `CTS_CONFIRMED` / `CTS_RECONFIRMED` | the confirmation candle (pullback apply candle, or the proximity candle) | `ev.idx` == `meta["confirmed_at"]` | `CTS_CONFIRMED.meta["cts_anchor_idx"]` = `st.cts.idx` at confirmation = the CURRENT CTS anchor (equals `CTS_ESTABLISHED.idx` unless a `CTS_UPDATED` moved it). Measured: equal to the same cycle's `CTS_ESTABLISHED.idx` on only 7 of the 30 `CTS_CONFIRMED` rows on the reference window (6 of 25 unique cycles; sub 7 cycle 1 is on both lenses) — never assume either way |
| `CTS_THRESHOLD_UPDATED` | NOT a price location: the processing candle whose range sync moved the CTS threshold — emitted only by `_sync_thresholds_from_range(i)` (active-range upkeep `_update_active_range(i)`, a pullback's apply candle, the sd-proximity confirmation candle); `ev.price` = the new threshold (the range breakout bound) | `ev.idx` | — (`meta["prev"]` is a price) |
| `REVERSAL_CANDIDATE` | the reversal pattern's first candle — its pattern-realm anchor (`meta["pattern_anchor_idx"]`, same value; = the close-break candle) | `meta["apply_idx"]` = the SCHEDULED apply — a prediction that can expire; the confirmed reversal is `STATE_CHANGED(to=reversal)` at its `ev.idx` | `meta["expires_idx"]` — slice-local (not shifted by `slice_begin`) in the M15 lens CSVs |
| `REVERSAL_WATCH_START` | the close-break candle MS processed when it emitted it (`_bos_barrier_step`) | `ev.idx` | `meta["pattern_anchor_idx"]` = that same close-break candle; `meta["expires_idx"]` — slice-local in the M15 lens CSVs |
| `STATE_CHANGED` | the candle at which the state change takes effect: the processed candle, a pattern's apply candle, or a range's confirm candle | `ev.idx` | `meta["effective_idx"]` defaults to `ev.idx` but is the range-start anchor for `to=range` (`reason="range_confirmed"`), and is slice-local (not shifted by `slice_begin`) in the M15 lens CSVs |

Not audited in this table: `RANGE_*` and `BOS_THRESHOLD_UPDATED` (`RANGE_STARTED` from
`_finalize_range_candidate_offline` is stamped at the `is_range_confirm_idx` label — see LANDMINES "Bounded MS
Runs Must Not Read Past `end_idx`"; pullback- / proximity-created ranges are stamped at the pullback apply /
proximity candle).

**In code — `structure/event_fields.py` (Plan E E2a, 2026-09-24).** The module names each index's role;
callers import the MODULE and call it qualified (`from engine_v2.structure import event_fields as ef`;
`ef.cts_anchor_idx(ev)`) — a direct import would collide with the many locals of the same names
(`tests/test_event_fields.py` bans it). Direct indexing only, never `.get(key, ev.idx)`:
- `ef.event_moment(ev)` — this table's moment column (added in Plan F in `market_structure`, moved here and
  extended in E2a): `CTS_ESTABLISHED` / `BOS_CONFIRMED` → `meta["confirmed_at"]`; `CTS_CONFIRMED` /
  `CTS_RECONFIRMED` / `CTS_THRESHOLD_UPDATED` → `ev.idx`; `CTS_UPDATED` → `ev.idx` on the raw path / `None` on
  the pattern path (no recorded moment); any other type raises `ValueError` (a new consumer must first define
  its event's moment here). Why it matters: IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule".
- `ef.cts_anchor_idx(ev)` — `meta["cts_anchor_idx"]` on `CTS_ESTABLISHED` / `CTS_CONFIRMED` / `CTS_RECONFIRMED`,
  `ev.idx` on `CTS_UPDATED`; `ef.bos_anchor_idx(ev)` — `BOS_CONFIRMED.meta["bos_anchor_idx"]`;
  `ef.pattern_anchor_idx(ev)`; `ef.processing_order_key(ev)` — today's event processing order `(ev.idx,
  ev.type)` frozen on the anchors (PLAN_E §6.2). Other types raise.
- `ef.stamped_idx(ev)` — the index `ev.idx` holds TODAY, frozen against the Plan E E4 flip (the anchor for
  `CTS_ESTABLISHED` / `BOS_CONFIRMED` / `CTS_UPDATED`, `ev.idx` otherwise). Neither a location nor a moment:
  only the sort keys and the E2 TIME halves over mixed CTS types, each written with a `# Plan E E3x → moment`
  marker naming the stage that switches it to `ef.event_moment` (user decision 2026-09-24).
- `CTS_UPDATED_RAW_VIA` lives here too (`market_structure` imports it).
- **Declared raw `ev.idx` readers** (the only production reads of a `CTS_ESTABLISHED` / `BOS_CONFIRMED` index
  outside this module after Plan E E2c): the events CSV writer (`debug/export_events.py`); the M15 mirror and the
  sibling clip's `new_ev.idx` (they copy the raw index); KL meta `source_event_idx` (write-only; deleted in Plan E
  E4b-pre); the `[kl_zones]` debug prints (they print the raw idx next to the anchor); the Plan A bounded-run
  assert; the debug probe script `debug/probe_fc_finalize.py`'s Phase-2 leak check (`max_ev_idx` / `past_bound`
  over all events — after E4 the BOS rows count at their moment, the right value for a bound check). Proof: the E4 variant replays (`plans/plan_e_inputs/review_scripts/e4flip_plugin.py`) and
  `tests/test_e4_simulation.py` (PLAN_E §6.4, §6.7).

**Bound and frequency (CTS_ESTABLISHED):** `meta["pattern_anchor_idx"]` (pattern anchor) `<= ev.idx` (CTS anchor)
`<= meta["confirmed_at"]` (moment) `<= meta["pattern_anchor_idx"] + range_max_k (5)`. The CTS anchor (the pattern's extreme candle) precedes the moment whenever an
earlier pattern candle (SUCCESS path) or a candle the confirmation search SKIPPED (CONFIRMED path, `_price_confirmation` / `_price_confirmation_1step` (continuous) — wrong direction,
wrong candle type (e.g. a pinbar), or a close just short of the threshold) holds the extreme. CTS anchor (`ev.idx`) == apply
candle is the COMMON case, not luck: on the reference window 31 of 34 CSV rows; the 3 others are 2 unique M15 cycles (sub 3 is mirrored into
both lenses), both `one_maru_opposite` CONFIRMED with lag 1 — sub 0 cyc 2: 1223 (a bull PINBAR,
skipped by the type filter, holds the highest high) vs 1224; sub 3 cyc 1: 2828 (closes 0.2 pip
short of the threshold, prints the lowest low) vs 2829. All five H1 cycles have lag 0.

**"Anchor" has two realms (GLOSSARY "Naming Standard"); the key names which one.** *Pattern realm* — a
candle pattern's FIRST candle, event meta `pattern_anchor_idx` (Plan E E1, 2026-09-24; was the bare
`anchor_idx`): `CTS_ESTABLISHED.meta["pattern_anchor_idx"]` (the breakout pattern) and `REVERSAL_CANDIDATE` /
`REVERSAL_WATCH_START.meta["pattern_anchor_idx"]` (the close-break candle = the first candle of the reversal
pattern searched from it). *Market-structure realm* — an ENDPOINT
(start or end) of a structure element: `CTS_CONFIRMED.meta["cts_anchor_idx"]` (the CTS anchor at
confirmation), `CTS_ESTABLISHED.meta["cts_anchor_idx"]` / `BOS_CONFIRMED.meta["bos_anchor_idx"]` (Plan E E2a), KL-zone `meta["anchor_idx"]` (the BOS anchor for a BOS zone, the CTS anchor for a CTS
zone — the candle its base pattern is found around, `zones/KL_ZONES_SPEC.md`), the fib anchors.

Every **timing / lifecycle** read of a cycle start — `zones/structure_lifecycle.compute_cycle_lifecycle`,
`multitf/parent_tables.build_parent_tables` (`cts_moment`), the POI activation floor's cycle term
(`zones/poi_zones.derive_poi_zones`, direct index — Plan D), the Plan-B early stop's finalize, FibTracker's
imbalance questions at a `CTS_ESTABLISHED` (`ef.event_moment`, direct index — Plan F) — uses
`meta["confirmed_at"]`. A `CTS_ESTABLISHED` without it makes the first two raise (`AssertionError`) and the
direct-index readers raise `KeyError`;
`unified_probe`'s finalize / `cts0_est_idx` reads (`_second_cts_moment`, Phase 2) read it through
`ef.event_moment` too (Plan E E2d removed their `.get("confirmed_at", ev.idx)` fallback to the CTS anchor, and the
other cross-kind `.get` fallbacks — PLAN_E_inputs §3 #3). Known sites that still read `ev.idx` as a time are
recorded as separate causes (the POI activation sweep's `CTS_UPDATED` transitions at `ev.idx`, pre-window and
in-window — correct on the raw path, whose `ev.idx` IS the moment; on the pattern path it is the CTS anchor and no
moment is recorded — an event-contract change, parked;
FibTracker's stamped activation / terminal timing (`activated_at` holds the CTS anchor — GLOSSARY "Naming
Standard") and the fill horizon `check_to_idx` of its imbalance reads at `CTS_ESTABLISHED` — the knowability cut
itself is on the moment (`evaluated_at`, Plan F); the pool's `knowable_at_idx` / sibling clip — PART4 §17.12; `unified_probe` Phase 2's retrace-window
start `check_lo = first_cts.idx + 1`, the CTS anchor, vs Phase 1's moment-based `tfb.est_idx + 1`; the MS
in-flight POI-inner resolver, which evaluates fills as-of `st.cts.idx` — the processing candle after a raw
`CTS_UPDATED`, the CTS anchor after a `CTS_ESTABLISHED` or a pattern-path update — `_refresh_poi_inners_for_cycle` →
`compute_poi_inners_for_cycle` / `select_fib_anchor_for_cycle`, and `_update_cycle0_data`, and feeds the
sd-zone-proximity CTS confirmation: its imbalance-EXISTENCE half is settled — no c3 cut, by decision, because the
POI-inner snapshot's only reader is gated `i > st.cts.idx` and the cycle-0 mirror is read only at a later cycle-1
refresh, so no decision uses a gap before it forms (Plan F, measured 24/24 CSVs identical with a cut;
`structure/MARKET_STRUCTURE_SPEC.md` "Snapshot vs per-candle"; the one accepted MS/FibTracker activation divergence
this leaves: IMBALANCE_FILL_SEMANTICS.md "Decided at the event") — its fill-horizon half is not yet measured). Site lists + staging: `plans/PLAN_E_inputs.md`.

The engine maintains a stable downstream interface by converting structure events into StructureLevels (CTS/BOS list).【fileciteturn1file14】

#### KLZone
Produced by `derive_kl_zones_v1` from structure events (not structure levels).【fileciteturn2file4】

---

## Lifecycle state convention (active / inactive / ended)

A long-lived object (POI zone, Fib, KL zone, structure cycle) has **two
orthogonal axes of state** that must be kept distinct in its representation:

| Axis | Nature | Reversible? | Determined by |
|---|---|---|---|
| **active / inactive** | Condition-state — "are all my required conditions true right now?" | **Yes** — flips back and forth as conditions change | A defined set of activation conditions evaluated at any candle `t` |
| **ended (terminal)** | Time-based irrelevance | **No** — once ended, stays ended | A defined set of end conditions (next cycle, reversal, the owning sub's window end — `parent_end` / `same_dir_replacement` — etc.) |

Once ended, "active" is undefined / always False. Before ended, the object
flips between active and inactive based on its activation conditions.

**Two tiers — not every lifecycle object has the condition (active/inactive)
axis.** It applies ONLY to objects with a genuinely **reversible condition** —
today just **POI** and **Fib**, whose condition is the unfilled-imbalance state
(flips as imbalances form / commit-fill; an imbalance FORMS at the close of its first c3,
`ImbalanceInstance.formed_at` = `start_idx + 1` — IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3
rule"). Objects with **no reversible
condition** — **structure cycles, structures, KL zones, WVMI** — are **tier-1**:
their "active" simply means **started-and-not-ended**, fully derivable from
`start_idx`/`end_idx`. They need no stored `active` flag and no
`activation_history`; `status` derives from start/end alone. So when adding a new
lifecycle object, ask **"does it have a reversible condition?"** — if not, give it
only `start_idx` + `end_idx`/`end_reason` + derived `status` (do NOT add an
`active` axis just for symmetry; it would be redundant).

`activation_history` (the per-flip list) is itself only **load-bearing for POI**,
whose chart fill + proximity gate walk it per-candle. **Fib dropped it** for
scalar `start_idx` logging (`FIB_LIFECYCLE_SPEC.md §15`, 2026-05-27 — no consumer
walked a fib history). **KL** carries a degenerate single-entry history purely as
a chart-fill convenience. Tier-1 objects carry none.

Recommended representation for a new lifecycle object (full form — trim per tier):

| Field | Type | Meaning |
|---|---|---|
| `end_idx` | `Optional[int]` | Terminal idx if known; `None` if not yet ended |
| `end_reason` | `Optional[str]` | Why it ended. Cycle-owned zones/fibs: `"next_cycle"` \| `"reversal"` \| the sub's `cap_reason` \| `"obsolete:<reason>"` \| ... Sub-structure pool objects (`TriggerRecord`, unique sub) and everything capped by a sub's window: `"reversal"` \| `"same_dir_replacement"` \| `"parent_end"` \| `None` (Plan C, 2026-09-20 — `"lifecycle_end"` is no longer emitted by any production path; `next_cycle` stays internal to `compute_cycle_lifecycle`) |
| `activation_history` | `list[{idx, active, reason}]` | All activate/deactivate flips within `[first_active, end_idx)`, driven by the object's activation conditions |
| `status` (derived) | `"active"` \| `"inactive"` \| `"ended"` | Computed from `end_idx` + `activation_history` at any candle `t` |

Derivation at candle `t`:
```python
if end_idx is not None and t >= end_idx:
    status = "ended"
elif current_state_per_activation_history(t) is True:
    status = "active"
else:
    status = "inactive"
```

**Following this convention:** `POIZone` (tier-2, full `activation_history`),
`KLZone` (tier-1, degenerate single-entry history), and `FibState` (tier-2,
scalar — see below); `WVMI` designed (tier-1, see below). POI/KL store `end_idx` +
`end_reason` + `activation_history` + derived 3-state `status` in `meta`. `KLZone` joined the convention in the Phase 3 unified-lifecycle pass
(2026-05-25): it computes no end of its own and inherits its owning cycle's
resolved end (reversal / next-cycle CTS-established / parent-cycle-end for
subs) — see `zones/KL_ZONES_SPEC.md` "Lifecycle". KL has no reversible
condition-state, so its `activation_history` is the single interval
`[{idx: confirmed_idx, active: True, reason: "confirmed"}]` and `status` is
`"active"` until `end_idx`, then `"ended"`.

POI activation conditions
(per-candle, in `zones/poi_zones.py::_compute_poi_activation_history`):
  1. `cts_at(t) >= ic_idx` — IC lies within the fib's bounds at t
     (the fib's `cts_idx` only grows via `CTS_UPDATED`).
  2. `ic_idx <= t` — IC candle exists.
  3. `has_unfilled_imbalance(df, ic_idx + 1, t, check_to_idx=t,
     direction=sd, evaluated_at=t)` — an sd-direction imbalance FORMED by
     t (its first c3 has closed: `inst.formed_at <= t` — Plan F) overlaps
     `(ic_idx, t]` and is not yet *committed-filled* per the two-stroke
     state machine (stroke 1 = 70% retrace; stroke 2 = close past gap
     outer in instance direction). The sweep evaluates it event-driven,
     not through the wrapper: an instance enters the unfilled set at
     `max(inst.formed_at, first_active)` and leaves at its cached stroke-2
     candle (`_compute_fill_idx_cache`). See `IMBALANCE_FILL_SEMANTICS.md`
     ("Knowability — the c3 rule" + the predicate). Flips as imbalances
     form / commit-fill.
  5. Variant ≥ V30 — IC candle overlaps the 61.8-80% fib zone
     (computed from `bos_price` and the time-varying `cts_price_at(t)`)
     by at least 30%. Variants can downgrade (V90 → V60 → V30) or
     vanish entirely as `cts_price` extends and the zone slides.
Scenario conditions (#4: per-scenario idx/price constraints) are gated
at IC identification and not re-checked per candle (once the
constraining event fires, the comparison is fixed).

**Activation floor (both POI and KL).** A zone's first-active is clamped to
its owning cycle's lifecycle-start: `first_active = max(<zone-specific
confirm idx>, cycle_lifecycle_start)`, where `cycle_lifecycle_start =
max(CTS_n ESTABLISHED MOMENT (meta["confirmed_at"]), structure lifecycle-start[,
lifecycle_floor])` — `zones/structure_lifecycle.compute_cycle_lifecycle`
(Plan C, 2026-09-20: the canonical cycle-start is the established **moment**,
never `CTS_ESTABLISHED.idx`, the CTS anchor — see the `ev.idx` convention above).
`lifecycle_floor` is `None` for main; for a sub it is the unique sub's
real-time `start_idx` (slice-local, passed by
`multitf/entity_df_mutation.render_sub_projection`) — the pre-Plan-C
"parent sid / parent_cycle_id floors" (plan B1, 2026-05-27) are now folded
into that one value, because the record's `start_idx` already contains the
parent floor (`parent_floor_idx`). This prevents a zone from activating before
its structure is alive — e.g. a post-reversal cycle-0 zone whose
`CTS_ESTABLISHED` precedes the reversal confirmation. See
`PART4_REFACTOR_SPEC.md §5` / `§17.4` (starting_idx vs lifecycle-start + the
clamp). NB the per-zone `<zone-specific confirm idx>` term: for KL it is the
zone's own `confirmed_idx` (a BOS zone's `confirmed_idx` IS
`BOS_CONFIRMED.meta["confirmed_at"]` = the moment) clamped to the per-sid
structure start (`compute_struct_start_by_sid`); for POI it is `ic_idx`, and
the cycle term is `cts_established_idx = CTS_ESTABLISHED.meta["confirmed_at"]`
(the moment — Plan D, 2026-09-23; until then it was `CTS_ESTABLISHED.idx`, the
CTS anchor), plus the `struct_start_by_sid` floor. Exception: a (sid, cycle)
with no `CTS_ESTABLISHED` has no cycle-lifecycle entry, and its POI cycle term
falls back to `fib_state.cts_idx` (`zones/POI_ZONES_SPEC.md` §4).

**`FibState` — NOW following (2026-05-27).** Tier-2 (it has the reversible
imbalance condition). Sessions 1 & 2 separated the overloaded `active` into
condition-only `active` + terminal `end_idx`/`end_reason` + derived `status`; the
cross-fib versioning is handled by making the **cycle the lifecycle identity**
with versions an internal sub-axis. The **§15 simplification** then DROPPED
`activation_history` for a sticky scalar `start_idx` (no fib consumer walked the
history) and wired fib onto the shared `compute_cycle_lifecycle` (clamp start to
the structure floor; cycle-end as an earliest-wins terminal candidate). Canonical:
`zones/FIB_LIFECYCLE_SPEC.md` (§15 is authoritative); `memory/project_fib_lifecycle_design.md`.

**`WVMI` — DESIGNED, impl pending (2026-05-27).** Tier-1 (created-once/locked-once,
no reversible condition): scalar `start_idx` (= creation `CTS_n` CONFIRMED, clamped)
+ inherited `end_idx` + derived `status`; NO active/inactive, NO `activation_history`;
its existing `created/updated/locked` computation axis is renamed `lp_status` so
`status` is the lifecycle label. Canonical: `zones/WVMI_SPEC.md` "Lifecycle
convention"; `memory/project_wvmi_lifecycle_deferred.md`.

So after WVMI lands, **every** lifecycle-like object will be on the convention
(`project_lifecycle_convention_klzone_fibstate.md` tracked the original migration).

**Sub-structure pool objects — `TriggerRecord` + unique sub (`PooledStructure`)
are tier-1 (Plan C, landed 2026-09-20; canonical: `PART4_REFACTOR_SPEC.md §17`,
code: `multitf/sub_structure_pool.py`, driver: `multitf/lifecycle_sweep.py`).**
Both have no reversible condition: "active" means started-and-not-ended, fully
derivable from `start_idx` / `end_idx`; no stored `active` flag, no
`activation_history`. Their lifecycle fields are **real-time** (what was
tradeable when); their pattern/element fields (`starting_idx`, `trigger_idx`,
`probe_finalize_idx`, every anchor) are **historical** and are only ever
*inputs* to a lifecycle value, never used *as* one in an is-active-at-`t` test.

| Object | `start_idx` | end |
|---|---|---|
| `TriggerRecord` (a triggered instance of a unique sub; identity `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)`, FK `sub_id`) | `max(probe_finalize_idx, trigger_idx, parent_floor_idx)` — the record EXISTS from here and nothing earlier. `parent_floor_idx` = `LOH(max(struct_start[S], cts_moment[(S,C)]))`, the parent cycle's clamped lifecycle-start on the CTS-established **moment** (`multitf/parent_tables.py`). | `trigger_end_idx` = the first end condition to fire (own reversal / `same_dir_replacement` at the replacing record's true `start_idx`, same lens only / `parent_end` = `end_m15[(S,C)]`), or the sub's frozen end on a post-end re-trigger; `end_idx = max(trigger_end_idx, start_idx)`; `end_reason ∈ {reversal, same_dir_replacement, parent_end, None}`; `ended_by_sub_id` names the replacing sub. `is_zero_length = trigger_end_idx is not None and trigger_end_idx <= start_idx` — such a record participates in nothing (logged only). |
| unique sub (`PooledStructure`; identity `(parent_path, sub_tf, direction, starting_idx)`, `sub_id` global creation index; NOT bound to a parent — spans parent cycles and sids) | the **first live (non-zero-length) record's** `start_idx`; set once (sweep phase 2 `SUB_START`). | re-evaluated at every candle `t` where one of its live records started or ended: over the live records that **exist at `t`** (`start_idx <= t`), `max_start = max(start_idx)`; candidates = non-null record `end_idx`s **strictly `> max_start`**; `end_idx = min(candidates)`, ties broken by `_END_REASON_PRIORITY` (`reversal > parent_end > same_dir_replacement`). Set once, frozen. `end_reason` = the winning record's. A sub with no live record has no `start_idx` and is logged, not rendered. |

`status` for both derives from `start_idx` / `end_idx` alone. The sub's window
`[start_idx, end_idx or data edge]` is the ONE lifecycle every lens draws
(§16.5 rev 2 ownership, `charting/CHARTING_SPEC.md` "M15 Dedicated Chart")
and the `lifecycle_floor` / `lifecycle_cap` / `cap_reason` that
`render_sub_projection` hands the zones layer (so every KL/POI/fib on a sub
inherits the sub's `end_reason` vocabulary as its cap reason).

**Deliberately NOT lifecycle objects:** structure events
(`df.attrs["structure_events"]`) and candle / structure patterns. These are
**immutable, append-only historical facts** (`PART4_REFACTOR_SPEC.md §7`) —
"what the algorithm believed at the time." They are never mutated and carry
no `active`/`ended` state. Their only time-varying property is **currency**
("which sid/cycle owns candle *t* for display"), and that is a *query-time
derivation* (the §16.5 most-recent-sid chart filter on H1; on the M15 charts
`export_m15_chart._compute_owner_by_idx_dir` — owner per `(candle, direction)`
over each unique sub's real-time window `[start_idx, end_idx or edge]`), not
state stored on the event. This is a settled decision, not a deferral: giving
an immutable fact a mutable lifecycle would contradict the append-only event
contract.

**Why decouple:** mixing "condition-state" and "terminal-state" into one
field (today's `active=False`) loses information. A consumer can't tell
whether a fib's `active=False` means "imbalances temporarily filled, could
come back" vs "this cycle is permanently done." Different consumers care
about different distinctions — charting wants to know "currently-tradeable
vs historical"; debug tooling wants to know "why did this go inactive?".
The 3-state status carries enough to answer both without overloading.

---

## Module boundaries

### Pipeline / Orchestration
`run_pipeline(df)` owns ordering and returns a single bundle for replay and future live use:
- df (enriched)
- pattern events
- structure levels
- meta (including zones)

Ordering is intentionally locked for Week 6: base features must be computed **before** structure so zone resolution is stable.【fileciteturn2file1】

### Wave Candles (`zones/wave_candles.py`)
Identifies boundary candles between consecutive waves at each KL zone. For each zone, produces a `WaveCandleResult` with `last_wave_candle_idx` (end of prior wave) and `first_wave_candle_idx` (start of new wave). BIB zones use an event-driven multi-step search; non-BIB zones use a ±5 candle window. Results stored in `df.attrs["wave_candles"]`. See `WAVE_CANDLES_SPEC.md`.

### WVMI (`zones/wvmi.py`)
Measures BOS zone strength via volume ratios of wave candle pairs. Runs **after POI zones** because it depends on POI zone inner bounds for its activation gate. Lifecycle:
0. **Gated by first sd zone-proximity trigger** — `check_zone_proximity()` (in `zones/zone_proximity.py`) scans candles from CTS_CONFIRMED to zone deactivation (next BOS or reversal). It produces a list of alternating sd / opp_sd trigger candles per cycle. The orchestrator uses only the first sd trigger as the WVMI gate (preserves pre-refactor behavior). Threshold defaults: H1 = 9 pips, M15 = 6 pips, M5 = 3 pips (caller-overridable).
1. **Created** at CTS_n confirmation (only if activated) — breakout momentum locked from FB/LB volumes
2. **Updated** each candle — temporary LP shifts to closest qualified candle near outer bound
3. **Locked** at BOS_n+1 confirmation — LP finalizes, pullback momentum locked

Results stored in `df.attrs["wvmi"]` (list of `WVMIRecord`). See `WVMI_SPEC.md`.

### Scenario 3 (`structure/structure_engine.py`)
Arbitrary-start structure analysis with iterative BOS_0 probe. Phase 1 validates/refines `start_idx` by checking if price reaches the BOS_0 zone inner bound (within configurable pip tolerance: H1=10, M15=3, M5=1). Phase 2 continues multi-structure analysis from the finalized probe using the same logic as `compute_structure`. Returns `Scenario3Result` with status always "finalized" (probe accepts current start when bound is reached).

Parameters: `end_idx` bounds the probe window (passed to MarketStructure); `run_continuation=False` skips Phase 2 for probe-only use (e.g. H1 reverse probe that only needs the validated `start_idx`).

### Structure From Start (`structure/structure_engine.py`)
`compute_structure_from_start()` runs multi-structure analysis from a known start without Scenario 1 identification or Scenario 3 probes. Same Exception 1/2 handling on reversals as `compute_structure()`. Used for lower-TF structures where the start has been pre-validated by a higher-TF probe.

### Multi-TF Analysis (`multitf/`) — the sub-structure pool (Plan C, landed 2026-09-20)
Subordinate lower-TF (M15) structures triggered by H1 events. Canonical spec:
`PART4_REFACTOR_SPEC.md §17` (rev 2). The pre-pool description (UC1 "H1 reverse
probe" → `compute_structure_from_start` → `lower_tf_pipeline.py`, one build per
trigger, lifecycle bounded by the parent cycle) is history: `lower_tf_pipeline.py`
is deleted, and the Phase-1 two-entity cadence chain (`_ChainCursor` /
`build_two_entity_parent_cycle` / `build_parent_cycle_chain` / `build_one_sid`)
was replaced by the sweep below.

**Flow (`pipeline/orchestrator._run_multi_tf_dual`):**
1. Detect the five trigger types on H1: `first_confluence` (`first_confluence_pipeline`),
   `first_counter` (`uc1_trigger.detect_uc1_triggers`), `subsequent_confluence`,
   `subsequent_counter` (their `*_pipeline.to_multi_tf_trigger`); `reversal` is
   synthesised by the sweep from a sub's own natural reversal. Each H1 trigger is
   lens-tagged (`sub_structure_pool.resolve_lens(use_case)`: `*_confluence` →
   confluence, `*_counter` → counter; a `reversal` inherits the spawning record's
   lens) and timing-mapped: `trigger_idx = LOH(trigger_event_idx)`
   (`entity_df_mutation._map_parent_idx_to_m15_hour_end`, last M15 candle of the
   H1 hour — the mapper for EVERY timing value; the price-extreme mapper
   `data_bridge.map_candle_to_lower_tf` is used only for the `first_confluence`
   probe's structural inputs; never unify them).
2. Prepare ONE shared M15 feature frame (`prepare_lower_tf_data`, once) plus two
   **lens dfs** (copies; views for the chart/export readers).
3. `multitf/parent_tables.build_parent_tables(sorted_events, h1_df, m15)` — the
   static parent tables from the H1 events: `cts_moment[(S,C)] =
   CTS_ESTABLISHED.meta["confirmed_at"]`, `floor_h1 = max(struct_start[S],
   cts_moment)`, `end_h1 = floor_h1[(S,C+1)]` else `rev_by_sid[S]` else None,
   `floor_m15` / `end_m15` = `LOH(...)`, `degenerate = end_m15 is not None and
   floor_m15 >= end_m15`. Asserts (never degrades): every LOH map succeeds;
   `BOS_CONFIRMED(S,C).confirmed_at == CTS_ESTABLISHED(S,C).confirmed_at`.
4. `multitf/lifecycle_sweep.run_lifecycle_sweep` — a priority-queue sweep over
   moments, phases at one idx in the order `TRIGGER_FIRE`/`REVERSAL_SPAWN` (0) →
   `RECORD_START` (1) → `SUB_START` (2) → `RECORD_END` (3) → `SUB_END` (4), each
   to completion; start-before-end at the same idx is load-bearing (a same-candle
   handover keeps a sub continuous). Phase 0 resolves a trigger: degenerate
   parent cycle → `UnresolvedTrigger(reason="degenerate_parent_cycle")` (no probe,
   no MS, no `sub_id`); else probe via the injected resolver
   (`_resolve_trigger_m15_start` for the four H1 types — `first_confluence` on its
   own ad-hoc BOS_0 with Phase 2, the sibling types on the OTHER lens's most
   recent qualifying CTS read **from the pool** (`_build_sibling_cts_ref_zone_from_pool`,
   records clipped to their own live window ∩ `[lo, hi]`, `hi` = the reading
   trigger's `trigger_idx`); `_resolve_reversal_start` for a reversal), every
   probe behind the probe cache (`_probe_with_cache`, key `(parent_path, sub_tf,
   direction, initial_input_idx)`; first probe to finalize is the truth for the
   key; a hit skips `unified_probe`, logs `[probe_cache] hit|APPROX hit` and
   `REF-ZONE DIFFERS` when the hitting reference inner ≠ the cached `bos0_inner`);
   then `build_or_get_geometry` (MS first with run cap = the DATA EDGE, pool entry
   only on success); then the `TriggerRecord` (`start_idx = max(finalize,
   trigger_idx, floor)`), with a same-`(lens, S, C)` re-trigger of the same sub
   absorbed into the existing record (`extra_trigger_idxs`).
5. ONE projection per unique sub (`entity_df_mutation.render_sub_projection` →
   `pooled_structure_build.project_to_window` with `floor = sub.start_idx`,
   `cap = sub.end_idx`, `cap_reason = sub.end_reason`, slice-local) mirrored into
   every lens df in `sub.lenses()` by `mirror_lower_tf_result_to_entity_df`, in
   `start_idx` order (later-live wins overlapping structure columns).
6. Sub WVMI, one sweep per unique sub (`_assign_sub_wvmi_per_sub`, §17.10
   minimal — not settled; `zones/WVMI_SPEC.md` "Sub entities").
7. Per lens df: `attrs["sids"]` (one `SidRecord` per unique sub on that lens,
   `sub_id` set, `sub_sid = None`), `attrs["triggers"]` (that lens's
   `TriggerRecord`s incl. zero-length), `attrs["unresolved_triggers"]`
   (pool-wide); registry registration; `meta["sub_pool"]`, `meta["parent_tables"]`.

**Key design decisions:**
- Identity of a unique sub = `(parent_path, sub_tf, direction, starting_idx)`
  with ABSOLUTE `direction` (`StructureKey`); `sub_id` = global creation index.
  Confluence vs counter lives on the record only: `relative_dir` (semantic:
  `direction == parent_sd` of the parent sid) vs `lens` (which chart) — they
  legitimately differ (reversal-stickiness).
- A record cannot outlive its parent cycle (`parent_end`); a unique sub can
  (it inherits parent bounds only through the aggregation rule above).
- Geometry is cap-free (one MS run to the data edge, slice-local events/df +
  `slice_begin`; the M15 slice keeps the 50-candle lookback and re-runs
  `compute_imbalance` + the `is_range_*` re-derivation after `reset_index`).
- KL zones are BOS-only for subs (`source_kinds=["BOS"]`); Fib uses
  `fib_mode="cross_cycle"`.
- Attribution stamped on every mirrored event/zone/POI/fib/wave-candle/WVMI
  record: `structure_path_id`, `timeframe`, `parent_tf`, **`sub_id`** (the
  identity) + informational `use_case`, `parent_sid`, `parent_cycle_id`,
  `started_by` from the sub's FIRST live record. No consumer may use the
  informational three for identity. `sub_sid` survives only on main-entity
  `SidRecord`s (`sub_sid = structure_id`); `trigger_sub_sid` lives on records.
- Exports (decoupled from the chart loop, `debug/export_sub_tables.py`, written
  BEFORE the M15 charts in `run_replay.py`): `*_M15_{lens}_subs.csv`,
  `*_M15_{lens}_triggers.csv`, `*_M15_unresolved_triggers.csv`; the old
  `*_M15_{lens}_sids.csv` is gone.
- Lifecycle values are real-time; `starting_idx` / `trigger_idx` /
  `probe_finalize_idx` are historical. The probe's search bound is
  `probe_end_idx` (a compute bound, unrelated to lifecycle `end_idx`); the
  probe's output anchor is `ProbeResult.starting_idx`.

### Charting
Charting reads from:
- dataframe columns
- `df.attrs["kl_zones"]`
- `df.attrs["wave_candles"]`
- `df.attrs["poi_zones"]`
- `df.attrs["fib_states"]`
- `df.attrs["wvmi"]`
- `df.attrs["prev_bos_lines"]`
- `df.attrs["structure_events"]`
It should not mutate algorithm state.

---

## Debug & QA invariants

### Structure invariants
The MarketStructure engine runs lightweight df-level invariant checks:
- range_lo <= range_hi while active
- CTS_CONFIRMED rows coherent with stage/phase
- BOS_CONFIRMED rows coherent
- reversal is terminal (once reversal appears, it never leaves reversal)【fileciteturn1file11】

### Zone visualization invariants
- Chart shows zones for most recent structure_id
- Within that structure, active zones are most recent buy and sell
- Deterministic draw ordering (inactive under active; older under newer)【fileciteturn2file0】

---

## Branching + versioning rules (process)

- **One branch per week** (e.g., `week6-kl-zones`) branched off `main`.
- Short-lived day/topic branches allowed.
- Merge to `main` only when that week’s Definition of Done is satisfied.
- Keep replay outputs for “golden” scenarios to detect regressions.

(These are project-level agreements; treat them as hard guardrails.)

