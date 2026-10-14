# Imbalance Fill Semantics

> Canonical reference for when an `ImbalanceInstance` **exists** (the c3
> knowability rule) and what it means for it to be **filled**. Consumed by
> FibTracker, the shared cross-cycle routine (`zones/cross_cycle_fib.py`),
> MarketStructure's in-flight snapshots, POI identification, and POI's
> per-candle activation sweep — so this doc lives outside `POI_ZONES_SPEC.md` to
> make clear it isn't POI-specific.

---

## Knowability — the c3 rule (Plan F, 2026-09-24)

`compute_imbalance` flags **c2**, the middle candle of a 3-candle FVG, and merges
consecutive same-direction c2s into one `ImbalanceInstance(start_idx, end_idx)`.
The gap is defined by c1 and **c3**, so it exists only once a c3 has closed.

**Rule (R1, "first c3").** An instance **exists from its first c3**:

```
ImbalanceInstance.formed_at = start_idx + 1          (engine_v2/common/types.py)
```

At a moment K it is visible iff `formed_at <= K`, and only its **formed prefix**
`[start_idx, min(end_idx, K - 1)]` — the c2s whose c3 has closed — is tested
against a window (`ImbalanceInstance.overlaps_formed_prefix`). A merged run keeps
growing until `end_idx + 1`, but an unfilled gap already exists from `formed_at`.

Worked example (H1 instance #154, bearish, c2s 997-999, reference window):

| Candle closes | Knowable then |
|---|---|
| 997 | nothing — 997 is a c2 whose c3 (998) has not happened |
| 998 | the first gap: 996 low 0.57492 > 998 high 0.57486 — the instance EXISTS (`formed_at` = 998) |
| 999 | the second gap forms (997 low 0.57450 > 999 high 0.57429) — the run grows |
| 1000 | the third gap forms; the run's final bottom (0.57366) is fixed |

The sell POIs IC 865 / 860 re-activate on this instance at 998 (at 997 before
Plan F — one candle before the gap existed). R2 ("last c3", `end_idx + 1`) would
wait until 1000 although an unfilled gap exists from 998, so it is rejected.

**Exactness.** For every K in `(start_idx, end_idx]` the formed prefix's only
scanned candle is its own c3, which cannot arm its own gap (bullish: the c3 low
IS the prefix's `gap_top`; stroke 1 needs a low `<= gap_top − 0.70·size`; needs
`fill_threshold > 0`). So "prefix unfilled" == "instance unfilled", and R1 gives
exactly a live engine's answer while still using the final merged bounds.
**Caveat:** a prefix and its merged run can differ in degeneracy (`gap_size <= 0`,
which `is_filled` treats as filled) — 0 cases on the reference data (H1 0/218
prefixes, M15 0/842).

### The cut is keyed on the MOMENT the question is asked — never on `check_to_idx`

Every read has two as-ofs:

- **`check_to_idx` — the fill horizon.** `is_filled` scans `(end_idx, check_to_idx]`.
  On several fib sites it is still a CTS **anchor** (a retro-stamp that can precede
  the moment — Plan E E3 territory).
- **`evaluated_at` — the moment of the question.** The candle at which the decision
  is taken; the knowability cut uses it.

Keying the cut on `check_to_idx` would drop gaps that ARE knowable: with a lag-1
CTS_ESTABLISHED (anchor 1223, moment 1224) a gap whose c2 is the anchor has formed
by the moment. `has_unfilled_imbalance` therefore takes `evaluated_at` as a
**keyword-only, REQUIRED** argument; `evaluated_at=None` is an explicit "no cut"
that every such caller justifies (table below).

**Event moments** — `structure/event_fields.py::event_moment(ev)` (Plan F; moved
out of `market_structure` and extended to BOS_CONFIRMED → `confirmed_at` and
CTS_CONFIRMED / CTS_RECONFIRMED → `ev.idx` by Plan E E2a): CTS_ESTABLISHED →
`meta["confirmed_at"]` (== `ev.idx` since Plan E E4a); CTS_UPDATED →
`ev.idx` on the raw path (`meta["via"] == CTS_UPDATED_RAW_VIA`), `meta["confirmed_at"]`
(the apply candle; `== ev.idx` since Plan E E4c, the anchor in `cts_anchor_idx`) on the pattern path (Plan E E3·0,
2026-09-24 — before it no moment was recorded and the pattern path was uncut;
measured on the reference window: the cut changes 0 cells);
CTS_THRESHOLD_UPDATED → `ev.idx` (the processing candle); any other type raises.
Canonical `ev.idx` table: ARCHITECTURE.md "`ev.idx` convention".

### Cached values are judged at the moment they are USED

A value computed at one candle and read at a later one needs the cut at the READ
moment, not the write:

- **MarketStructure's in-flight POI-inner snapshot** (refreshed at each
  CTS_ESTABLISHED / CTS_UPDATED; since PLAN_I S5 the embedded FibTracker's
  representative fib → `poi_inner_for_fib`): its FIB is decided at the fed event's
  moment like the drawn fib's (FibTracker below); the IP pick on it
  (`select_ic`'s IC cond3 — the immediate gap since PLAN_P — / its tiebreak and ranking rules' imbalance reads) is built at the refresh candle and may count a
  gap whose c3 is the NEXT candle — but the snapshot's only reader
  (`_maybe_confirm_cts_via_proximity`) is gated `i > st.cts.idx`, so every gap it
  counts has formed by the candle it is used on. No cut on the pick (see
  MARKET_STRUCTURE_SPEC "Snapshot vs per-candle").
- (The cycle-0 liveness caches — MS's `cycle0_data["has_unfilled"]` and FibTracker's
  twin, both Scenario-2 cond2 stored UNCUT — are retired: FibTracker's with the
  H1-main logic in PLAN_I S4, MS's with its resolver in S5.)

### Decided at the event — and the accepted divergence

FibTracker re-evaluates only at CTS events, so a gap that is not yet formed at an
event can **drop** a fib / cross rather than delay it by one candle (reference
window: counter sub 5's pre-created cycle-1 cross fib — sub 5 never establishes a
cycle 1 — and its IC 3654 twin POI no longer exist). That is the outcome the
engine already gives when a gap forms one candle after an event. "Re-ask when the
gap's c3 closes" is a logged follow-up (PLAN_F §7).

**M1 divergence (accepted 2026-09-24) — RETIRED by PLAN_I S5 (2026-10-08):** the retired MS in-flight resolver
was uncut and built its fib as always-active while FibTracker cut at the event, so in the drop case (the ONLY sd gap of
a lag-0 CTS_ESTABLISHED or raw CTS_UPDATED has c2 == the event candle) MS kept a POI inner for a fib FibTracker never
created (0 cases on the reference window). Since S5 MS reads its embedded FibTracker's decision — the same cut, the
same fib (pinned: `test_imbalance_c3_knowability.py::test_m1_retired_ms_reads_no_inner_where_fibtracker_creates_no_fib`).
The old text: `zones/CROSS_CYCLE_FIB_SPEC.md` Appendix A, part 2.

---

## The predicate: `ImbalanceInstance.is_filled`

Defined in `engine_v2/common/types.py::ImbalanceInstance.is_filled`. One
method, one rule:

```
is_filled(df, check_to_idx, fill_threshold=0.70) -> bool
```

An instance is **filled** iff both strokes of a two-stroke state machine
have fired by `check_to_idx`:

| Stroke | Direction | Condition |
|---|---|---|
| **Stroke 1 — armed** | bullish | first candle in `(end_idx, check_to_idx]` with `low <= gap_top - gap_size * fill_threshold` |
| | bearish | first candle in `(end_idx, check_to_idx]` with `high >= gap_bottom + gap_size * fill_threshold` |
| **Stroke 2 — confirmed** | bullish | first candle in `[armed_idx, check_to_idx]` with `close >= gap_top` |
| | bearish | first candle in `[armed_idx, check_to_idx]` with `close <= gap_bottom` |

Both strokes can fire on the same candle (rare but legal —
`armed_idx == confirmed_fill_idx`). Stroke 2 must be at `idx >= armed_idx`;
a close past the gap outer **before** any stroke 1 does not count. The scan
starts at `end_idx + 1` — the last c3, which cannot arm its own gap.

The state machine is **monotonic in `check_to_idx`**: once `is_filled`
returns True at some `check_to_idx = t`, it returns True for all larger
values. Strokes latch once and never un-latch. (`has_unfilled_imbalance` asked at
a moment is NOT monotone in that moment: an instance enters when it forms and
leaves when it fills.)

### Edge cases

| Case | Behavior |
|---|---|
| `gap_size <= 0` (degenerate gap) | Returns **True** — safe default; treat invalid gap as filled so it doesn't pollute "unfilled" counts |
| `end_idx >= check_to_idx` (empty scan range) | Returns **False** (unfilled). Correct for a FORMED prefix (its c3 cannot arm it). An instance that has not formed at the moment of the question is not an imbalance at all — the caller's `evaluated_at` cut excludes it before `is_filled` is asked |
| Stroke 2 condition met before any stroke 1 | Returns **False** — stroke 2 only counts at `idx >= armed_idx` |
| Stroke 1 fires but stroke 2 never does | Returns **False** forever — instance stays "armed but not committed" until terminal end of any containing window |

### Why two strokes

A single-stroke definition (just the 70% retrace) treats every gap touch as
invalidation. Real markets often touch a gap and bounce — the touch alone
doesn't commit the move. The two-stroke definition requires both an
engagement (price retraces into the gap by 70%) AND a follow-through
commitment (price subsequently closes past the gap's outer edge in the
imbalance's direction). Until both happen, the gap is still "in play."

Concrete effect on POI activation: a per-candle sweep that flips
deactivate→reactivate→deactivate on every wick into a gap (the
pre-2026-05-23 behavior) collapses to a single deactivate at the candle
where stroke 2 confirms.

---

## The wrappers

### `has_unfilled_imbalance`

```
has_unfilled_imbalance(df, start_idx, end_idx, check_to_idx,
                       fill_threshold=0.70, *, direction=None, evaluated_at) -> bool
```

In `engine_v2/patterns/imbalance.py`. For each instance in
`df.attrs["imbalances"]`:

1. **Direction filter** — skip unless `direction is None` or
   `inst.direction == direction` (every production caller passes `sd`).
2. **Window + knowability** — `evaluated_at=None`: skip unless
   `inst.overlaps(start_idx, end_idx)`. Otherwise skip unless
   `inst.overlaps_formed_prefix(start_idx, end_idx, evaluated_at)` (formed by the
   moment, formed prefix intersects the window).
3. **Fill check** — `inst.is_filled(df, check_to_idx, fill_threshold)`.

Returns `True` on the first unfilled survivor. `evaluated_at` is keyword-only
and has no default: a caller that forgets its moment fails with a `TypeError`.

**`iter_unfilled_imbalances`** (same arguments, PLAN_H landing A 2026-10-06)
yields those survivors lazily in `df.attrs["imbalances"]` order;
`has_unfilled_imbalance` IS `any(iter_unfilled_imbalances(...))` (it stops at
the same instance). IC condition 3 (the immediate unfilled imbalance, IC + 1 … IC + 2 — Step 1's
`immediate_unfilled_imbalance_first_c2_idx` since PLAN_P; a tiebreak / ranking rule before it) and the tiebreak and
(since PLAN_L L2) ranking rules' most-recent-before-imbalance reference (the earliest unfilled sd gap after the oldest
IC / candidate in play) read from it — one "unfilled" test for both (POI_ZONES_SPEC §3.1 / §3.4 / §3.6).

### `get_unfilled_imbalances` / `has_imbalance_in_range` — DELETED 2026-09-30

They had no production caller and no knowability cut (user decision: delete;
the parked "direction filter" for `get_unfilled_imbalances` was dropped with it —
`has_unfilled_imbalance(..., direction=sd, evaluated_at=...)` asks the
direction-filtered question at a moment). `has_unfilled_imbalance` is the one
wrapper.

---

## Consumer call-site matrix (as landed, Plan F — HEAD line numbers at landing)

Each consumer tunes four knobs: the **window**, the fill horizon
**`check_to_idx`**, the moment **`evaluated_at`**, and the **direction** (always
`sd`). Sites are named by function (their line numbers drifted with every edit and were dropped 2026-09-28 — grep the function).

### POI

| Site | Window | `check_to_idx` | `evaluated_at` | Question |
|---|---|---|---|---|
| `poi_zones._compute_poi_activation_history` (the enter candle) | `(ic_idx, t]` | `t` | `t` (enters at `max(inst.formed_at, first_active)`) | "Is a FORMED sd imbalance after the IC unfilled at candle t?" — the per-candle sweep, via the fill cache (below). Equals `has_unfilled_imbalance(df, ic_idx+1, t, check_to_idx=t, direction=sd, evaluated_at=t)` |
| `poi_zones._has_unfilled_imbalance_after` (the RETIRED IC condition 3 since PLAN_P — information on every check + the unwired V30 finder's filter; `check_ic_conditions`, was `find_ic_candidates` until PLAN_L L1) — caller `derive_poi_zones` | `(candidate_idx, cts_idx]` | the final fib's `cts_idx` (anchor) | `None` — retrospective IC identification; the sweep enforces WHEN the POI can go live (Plan E §2.4 item 8) | the retired IC cond3 (information) |
| same — caller MS in-flight `poi_inner_for_fib` (PLAN_I S5; was `compute_poi_inners_for_cycle`) | `(candidate_idx, cts_idx]` | the representative fib's `cts_idx` (anchor, ≤ `st.cts.idx`) | `None` — read only at `i > st.cts.idx` (Knowability §"Cached values") | the retired IC cond3 (information) in the proximity snapshot's selection |
| `poi_zones.immediate_unfilled_imbalance_first_c2_idx` — `check_ic_conditions` (every check), both callers (the IP tiebreaker update, 2026-10-09; IC condition 3 since PLAN_P) | `(ic_idx, min(ic_idx + 2, cts_idx)]` | the fib's `cts_idx` (anchor) | `None` — as IC cond3 | "Is there an unfilled sd imbalance immediately after the IC?" — IC condition 3 since PLAN_P (until then the first tiebreak rule and ranking rule 2) + the IC CSV column |
| `poi_zones.most_recent_before_imbalance` (the shared rule helper; called by `apply_tiebreak_rules` and `apply_ranking_rules`, PLAN_L L2) — the most-recent-before-imbalance reference | `(oldest_idx_in_play, cts_idx]` | the fib's `cts_idx` (anchor) | `None` — as IC cond3 | "The earliest unfilled sd imbalance after the oldest IC / candidate in play?" — the last tiebreak rule (2) / ranking rule (3) — rules 3 / 4 until PLAN_P |

### FibTracker — `evaluated_at = self._evaluated_at` = `event_moment` of the handled event

`on_cts_established` / `on_cts_updated` / `on_cts_threshold_updated` run under
`_evaluating(event)`; `_has_unfilled(...)` passes the moment (a pattern-path
CTS_UPDATED's apply candle since Plan E E3·0). Two instances run the same rows: the
downstream's (the drawn fib) and, since PLAN_I S5, MarketStructure's embedded one
(MS's in-flight read), each fed the same events.

| Site | Window | `check_to_idx` | Question |
|---|---|---|---|
| `_on_cts_established` | `[BOS_n, CTS_n]` | the EST moment `confirmed_at` (Plan E E3a; the anchor before) | a single's activation at EST (cycle 0; a cycle past the cross-fib limit) — a cycle within the limit asks in `_cross_check` |
| `_handle_cross_cycle_cts_updated` | `[BOS_n, CTS]` | the update's moment (raw `idx` / pattern `confirmed_at`; Plan E E3a) | a single's first activation on update (cycle 0; past the limit — PLAN_I S3) |
| `_update_fib_cts` | `[bos, cts]` | the update's moment (raw `idx` / pattern `confirmed_at`; Plan E E3a) | reactivate / deactivate (`all_imbalances_filled` = no FORMED unfilled imbalance) |
| `_update_fib_cts` cross branch | — | — | DELETED 2026-09-28 (dead: every caller passes a single `(sid, cycle)` key) |
| `_cross_check` | via the routine | — | every structure's cross check (EST / UPDATED / the pre-established threshold path) |

Retired with the H1-main logic (PLAN_I S4, 2026-10-08; rows in `git show 0846441:engine_v2/IMBALANCE_FILL_SEMANTICS.md`):
`_handle_sid1plus_cts_established`'s cycle-0 cache write, `_handle_cycle0_cts_updated` (the cache re-snapshot +
`_c0_has_unfilled_now`), `_update_cycle1_main` (its three checks + the own-window assert), the cycle-1
`select_fib_anchor_for_cycle` call (`_handle_cycle1_scenarios`), the §11b peek `_maybe_activate_main_cross`.

### The shared routine — `cross_cycle_fib.resolve_cross_cycle_eligibility` (`evaluated_at` required, forwarded)

| Site | Window | `check_to_idx` | Question |
|---|---|---|---|
| own test | `[own_imb_start, own_window_end_idx]` | `fill_horizon_idx` | "Does the target cycle have its own FORMED unfilled imbalance?" — the site the cut can change (the window ends AT `own_window_end_idx`) |
| walk | `[BOS_k, CTS_k]` | `fill_horizon_idx` | prior-cycle liveness — already bounded (`CTS_k < conf_k <=` the moment) |

Plan E E2b split the old `current_candle` / `own_imb_start` into their LOCATION role
(`own_window_end_idx`, `own_imb_start`: the window) and their TIME role
(`fill_horizon_idx`: keyword-only, required). `fill_horizon_idx` is the handled
event's moment since Plan E E3a (2026-09-24; before it the CTS anchor on
CTS_ESTABLISHED / pattern-path CTS_UPDATED). The moment of the question stays ONE
parameter (`evaluated_at`); FibTracker, the only caller since PLAN_I S5, passes the
same value for both. (The "snapshot" walk of the retired Scenario-2 selector — cond3
at `snapshot_horizon_idx`, the BOS_1 moment — and `select_fib_anchor_for_cycle`
itself were deleted in PLAN_I S5.)

### MarketStructure — no imbalance read of its own (PLAN_I S5)

MS's in-flight POI read asks its imbalance questions through its embedded FibTracker (the rows above, cut at each
fed event's moment) and the IP pick (`poi_inner_for_fib` → `select_ic`: the IC cond3 rows of the POI table — the immediate gap since PLAN_P — uncut).
Its own uncut mirrors — `_update_cycle0_data` (the Scenario-2 cond2 cycle-0 snapshot) and the resolver's
`select_fib_anchor_for_cycle` cond1 — were deleted in S5 (rows in `git show 0de98ca:engine_v2/IMBALANCE_FILL_SEMANTICS.md`).

---

## Why every Fib + scenario consumer is strict to sd

The pre-2026-05-23 design left these calls permissive (no direction
filter), arguing that the BOS→CTS span is structurally directional and a
counter-direction imbalance inside it is geometrically unusual but should
still count when it appears.

Counter-direction imbalances inside a structural cycle's swing never
produce POIs (POIs are sd-direction by construction — see
`POI_ZONES_SPEC.md §4` "POIs are always sd-direction"). Since the
downstream consequence of a Fib activation / Scenario 2 decision is the
POI set, counter-direction imbalances driving those decisions creates a
disconnect between cause (counter-direction unfilled imbalance) and
effect (sd-direction POI). The strict filter aligns each consumer's input
with what it can actually influence downstream.

POI IC validation (IC condition 3 — then `find_ic_candidates`, since PLAN_L L1 `_has_unfilled_imbalance_after`, since PLAN_P the immediate gap) was already strict — it asks a
slightly different question ("did an institutional candle produce a
same-direction continuation push after the candidate?") which has always
required directional alignment. The asymmetry that used to exist between
Fib activation (permissive) and POI IC validation (strict) — discussed in
the old `POI_ZONES_SPEC.md §1` section — is now gone.

---

## The cache layer (POI-internal optimization)

`zones/poi_zones.py::_compute_fill_idx_cache` precomputes
`(armed_idx, confirmed_fill_idx)` per imbalance instance, returns a dict
keyed by `id(inst)`, AND stashes both indices on `inst.meta` for debug
exporters. This is consumed by `_compute_poi_activation_history` so its
per-candle sweep doesn't re-scan the `(end_idx, t]` window for every
candidate `t`.

The sweep's two transitions per instance: **enter** at
`max(inst.formed_at, first_active)` (the first c3 — the knowability rule; the
relevance filter keeps `inst.formed_at <= scan_end`) and **leave** at
`confirmed_fill_idx` (always `>= end_idx + 2`, so after the enter).

**Equivalence invariant:** a caller of `is_filled(check_to_idx=t)` would
get the same boolean as
`confirmed_fill_idx is not None and confirmed_fill_idx <= t`. The cache
is a performance optimization, not a divergent semantics — any drift
between the two is a bug.

**Why it's POI-only:** the cache exists to amortize the cost of
per-candle evaluation. FibTracker and MarketStructure call
`has_unfilled_imbalance` once per CTS event, not per candle, so they
don't need it.

**Live-mode note:** under live (incremental) execution, `armed_idx` is
known at the moment stroke 1 fires; `confirmed_fill_idx` is None until
stroke 2 fires. The cache built today computes both eagerly over the
full df because backtest knows the entire price path up front. A live
adaptation would maintain both indices as state per instance, advancing
them as new candles arrive — and would create the instance at `formed_at`,
growing its prefix while the run continues.

---

## Deliberate behavior shifts

### Two-stroke fill (2026-05-23)

The universal change (rolled in 2026-05-23) replaces the old single-stroke
70% retrace predicate with the two-stroke state machine. Where this matters:

| Consumer | Old (70% retrace only) | New (two-stroke) |
|---|---|---|
| **POI per-candle sweep** | Activation history can flip D↔A on every gap touch (e.g., M15.confluence sid=1 cyc=0 in the per-POI table showed 7 flips at idx 1893→1936) | Collapses to one deactivate at the stroke-2 candle (or none if stroke 2 never fires within the lifecycle) |
| **POI Layer 1 (`find_ic_candidates`)** | An IC qualifies if any post-IC sd-direction imbalance has not yet 70%-retraced | An IC qualifies if any post-IC sd-direction imbalance has not yet committed-filled — slightly **more** ICs qualify (some POIs that didn't exist now do) |
| **Fib activation** | Mostly invariant — in a healthy BOS→CTS swing, stroke 1 doesn't fire | Effectively unchanged |
| **Scenario 2 cond3** | Filled if cycle 0's imbalance 70%-retraces by BOS_1 | By the geometric invariant ("BOS_1 must close past gap_top to form, since BOS_1 > BOS_0 > gap_top"), if stroke 1 fires before BOS_1, stroke 2 also fires — **invariant in the dominant case** |
| **Dead-cycle walks (`cross_cycle` mode)** | Cycle marked dead the first candle its imbalance 70%-fills | Cycle stays alive longer (until stroke 2 confirms); cross-fib chains can extend further back |

The geometric invariant for cond3 means most production behavior outside
POI Layer 2 stays put. The exception is `cross_cycle` dead-cycle walks
(M15 subordinate pipelines), where `check_to_idx` is `current_candle` — the
handled event's moment (since Plan E E3a, 2026-09-24; before it the CTS anchor
on CTS_ESTABLISHED and pattern-path CTS_UPDATED) — rather than a fixed structural
event; those genuinely see "cycle stays alive longer" under the new semantics.

### c3 knowability (2026-09-24, Plan F)

`plans/PLAN_F_imbalance_c3_knowability.md` (as landed §9; inputs + the 39-row
consumer audit: `plans/PLAN_F_inputs.md`). Measured on the reference window vs
save `20260923_172626_0a4eadc`: 19/24 CSVs byte-identical.

| Consumer | Delta |
|---|---|
| POI sweep | H1 sid 1 cyc 2 IC 865 / 860 re-activations 953→954 and 997→998 (`confirmed_idx` 997→998); M15 sub 7 IC 4048 4118→4119 (both lenses) |
| FibTracker CTS_ESTABLISHED (lag 0) | M15 conf sub 2 cyc 0 activates on the next raw update (`activated_at` 53→54, `activated_on: update`) |
| FibTracker raw CTS_UPDATED | M15 sub 3 cyc 0 `activated_at` 61→62 (both lenses) |
| Cross-check at CTS_THRESHOLD_UPDATED | counter sub 5's pre-created cycle-1 cross fib and its IC 3654 twin POI no longer exist; sub 5 cyc 0 fib ends at 4083 (`same_dir_replacement`) instead of 3806 (`new_cycle`); counter chart 153/125 → 151/124 |
| MS in-flight snapshot | none (uncut by decision then; 3 inner lists would change, none is ever read — the read is the embedded FibTracker's since PLAN_I S5) |
| IC identification | none (uncut; retrospective) |

(The M15 fib `activated_at` values in this table are slice-local — the frame the M15 fib CSVs exported until
Post-E·2, 2026-09-26; entity-absolute = + the sub's `slice_begin`: sub 2's 54 = 2369, sub 3's 62 = 2651.)

---

## When to revisit

This predicate is one of the load-bearing primitives of the system. Reasons
that would justify a future redesign:

- **A consumer needs a different stroke 2 definition.** E.g., POI's per-
  candle sweep wants close-past-outer but Fib activation prefers
  high/low-past-outer. Current design has one universal predicate; adding
  consumer-specific variants means rebuilding the matrix above with
  per-consumer fill predicates.
- **Live-mode incremental computation** requires per-instance state
  rather than a full-df scan per call. The cache abstraction already
  supports this — `is_filled` would become a thin wrapper over a stateful
  per-instance tracker, and instances would be created at `formed_at`.
- **A third stroke is needed** (e.g., a re-entry confirmation after
  stroke 2). Today's state machine is monotonic 0→1→2; a more complex
  state machine would need new naming for the additional state.
- **Re-ask when a gap forms** (PLAN_F §7): FibTracker re-evaluating a failed
  decision at the candle a relevant gap's c3 closes would turn "dropped" into
  "one candle later" (the M1 divergence it was also to close is retired since
  PLAN_I S5 — MS reads FibTracker's decision).
