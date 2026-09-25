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
`meta["confirmed_at"]`; CTS_UPDATED →
`ev.idx` on the raw path (`meta["via"] == CTS_UPDATED_RAW_VIA`), `meta["confirmed_at"]`
(the apply candle) on the pattern path, whose `idx` is the CTS anchor (Plan E E3·0,
2026-09-24 — before it no moment was recorded and the pattern path was uncut;
measured on the reference window: the cut changes 0 cells);
CTS_THRESHOLD_UPDATED → `ev.idx` (the processing candle); any other type raises.
Canonical `ev.idx` table: ARCHITECTURE.md "`ev.idx` convention".

### Cached values are judged at the moment they are USED

A value computed at one candle and read at a later one needs the cut at the READ
moment, not the write:

- **MarketStructure's in-flight POI-inner snapshot** (`compute_poi_inners_for_cycle`,
  refreshed at each CTS_ESTABLISHED / CTS_UPDATED) is built at the refresh candle
  and may count a gap whose c3 is the NEXT candle — but its only reader
  (`_maybe_confirm_cts_via_proximity`) is gated `i > st.cts.idx`, so every gap it
  counts has formed by the candle it is used on. No cut (see
  MARKET_STRUCTURE_SPEC "Snapshot vs per-candle").
- **The cycle-0 liveness cache** (FibTracker `_cross_cycle_data[sid]["cycle0"]
  ["has_unfilled"]` and its MS mirror `cycle0_data["has_unfilled"]`) is Scenario-2
  cond2, read at CTS_1 ESTABLISHED (> CTS_0) when every gap in `[BOS_0, CTS_0]` has
  formed → stored UNCUT. A decision taken at the write event (the Scenario-1
  cycle-0 activation) re-asks it at that event's moment
  (`FibTracker._c0_has_unfilled_now`).

### Decided at the event — and the accepted divergence

FibTracker re-evaluates only at CTS events, so a gap that is not yet formed at an
event can **drop** a fib / cross rather than delay it by one candle (reference
window: counter sub 5's pre-created cycle-1 cross fib — sub 5 never establishes a
cycle 1 — and its IC 3654 twin POI no longer exist). That is the outcome the
engine already gives when a gap forms one candle after an event. "Re-ask when the
gap's c3 closes" is a logged follow-up (PLAN_F §7).

**M1 divergence (accepted, 2026-09-24):** the MS in-flight resolver is uncut and
builds its fib as always-active, FibTracker cuts at the event. In the drop case
(the ONLY sd gap of a lag-0 CTS_ESTABLISHED or raw CTS_UPDATED has c2 == the event
candle) MS keeps a POI inner for a fib FibTracker never creates — permanent on H1
cycle ≥ 1 (one-shot EST activation). 0 cases on the reference window; the same
class already exists for gaps that form after an H1 cycle-≥1 EST. Cond2 and cond3
still agree between the layers (LANDMINES "Scenario 2 anchor agreement").

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

### `get_unfilled_imbalances` / `has_imbalance_in_range`

No production caller (the import in `fib_tracker.py` is unused). Neither has a
knowability cut — do not use them for a question asked at a moment (hygiene
follow-up: delete).

---

## Consumer call-site matrix (as landed, Plan F — HEAD line numbers at landing)

Each consumer tunes four knobs: the **window**, the fill horizon
**`check_to_idx`**, the moment **`evaluated_at`**, and the **direction** (always
`sd`). Sites are named by function; line numbers are hints.

### POI

| Site | Window | `check_to_idx` | `evaluated_at` | Question |
|---|---|---|---|---|
| `poi_zones._compute_poi_activation_history` (enter `:930`) | `(ic_idx, t]` | `t` | `t` (enters at `max(inst.formed_at, first_active)`) | "Is a FORMED sd imbalance after the IC unfilled at candle t?" — the per-candle sweep, via the fill cache (below). Equals `has_unfilled_imbalance(df, ic_idx+1, t, check_to_idx=t, direction=sd, evaluated_at=t)` |
| `poi_zones.find_ic_candidates` `:233` — caller `derive_poi_zones` | `(candidate_idx, cts_idx]` | the final fib's `cts_idx` (anchor) | `None` — retrospective IC identification; the sweep enforces WHEN the POI can go live (Plan E §2.4 item 8) | IC cond3 |
| same — caller MS in-flight `compute_poi_inners_for_cycle` | `(candidate_idx, cts_idx]` | in-flight `st.cts.idx` | `None` — read only at `i > st.cts.idx` (Knowability §"Cached values") | IC cond3 for the proximity snapshot |

### FibTracker — `evaluated_at = self._evaluated_at` = `event_moment` of the handled event

`on_cts_established` / `on_cts_updated` / `on_cts_threshold_updated` run under
`_evaluating(event)`; `_has_unfilled(...)` passes the moment (a pattern-path
CTS_UPDATED's apply candle since Plan E E3·0).

| Site | Window | `check_to_idx` | Question |
|---|---|---|---|
| `_on_cts_established` `:616` | `[BOS_n, CTS_n]` | the EST moment `confirmed_at` (Plan E E3a; the anchor before) | fib activation at EST (single / sid0 / sid≥1 / Scenario 1) |
| `_handle_sid1plus_cts_established` cycle 0 `:776` | `[BOS_0, CTS_0]` | `cts_idx` | the cycle-0 liveness **cache** (cond2) — `evaluated_at=None`, uncut |
| `_handle_cross_cycle_cts_updated` `:1288` | `[BOS_0, CTS]` | the update's moment (raw `idx` / pattern `confirmed_at`; Plan E E3a) | cross_cycle cycle-0 first activation on update |
| `_handle_cycle0_cts_updated` `:1423` | `[BOS_0, CTS]` | update `idx` | the cycle-0 **cache** re-snapshot — `evaluated_at=None`, uncut |
| `_handle_cycle0_cts_updated` `:1435`, `:1456` (`_c0_has_unfilled_now`) | cached `[bos_idx, cts_idx]` | cached `cts_idx` (the anchor until Plan E E3a′ — at CTS_0 EST the same decision's horizon is already the moment) | Scenario-1 cycle-0 activation on update, asked at the moment |
| `_update_fib_cts` `:1559` | `[bos, cts]` | the update's moment (raw `idx` / pattern `confirmed_at`; Plan E E3a) | reactivate / deactivate (`all_imbalances_filled` = no FORMED unfilled imbalance) |
| `_update_fib_cts` cross branch `:1541/:1547/:1550` | — | — | dead code (never reached; hygiene follow-up) |
| `_update_cycle1_main` `:1622` (cycle 0 @CTS_0), `:1627` (cycle 1 own), `:1629` (@BOS_1) | as named | `c0_cts_idx` (E3a′) / the update's moment (E3a) / `cycle1_bos_idx` (E3a′) | H1 cycle-1 cross update — the code's cond1/cond2 labels are the REVERSE of `select_fib_anchor_for_cycle`'s docstring; read the questions, not the labels |
| `_update_cycle1_main` `:1657` | `[BOS_1, CTS_1]` | the update's moment (raw `idx` / pattern `confirmed_at`; Plan E E3a) | create-on-fail single (lock-step with `:1627`) |
| `select_fib_anchor_for_cycle` call `:972` | via the routine | — | H1 cycle-1 Scenario 2/3 at CTS_1 EST |
| `_maybe_activate_main_cross` `:2074`, `_m15_cross_check` `:2253` | via the routine | — | §11b main cross peek / subordinate cross check |

### The shared routine — `cross_cycle_fib.resolve_cross_cycle_eligibility` (`evaluated_at` required, forwarded)

| Site | Window | `check_to_idx` | Question |
|---|---|---|---|
| own test | `[own_imb_start, own_window_end_idx]` | `fill_horizon_idx` | "Does the target cycle have its own FORMED unfilled imbalance?" — the site the cut can change (the window ends AT `own_window_end_idx`) |
| snapshot walk | `[BOS_0, CTS_0]` | `snapshot_horizon_idx` (= BOS_1) | cond3 — already bounded (window ends before the moment) |
| current walk | `[BOS_k, CTS_k]` | `fill_horizon_idx` | prior-cycle liveness — already bounded (`CTS_k < conf_k <=` the moment) |

Plan E E2b split the old `current_candle` / `own_imb_start` into their LOCATION role
(`own_window_end_idx`, `own_imb_start`: the window) and their TIME role
(`fill_horizon_idx`, `snapshot_horizon_idx`: keyword-only, required). `fill_horizon_idx`
is the handled event's moment since Plan E E3a (2026-09-24; before it the CTS anchor
on CTS_ESTABLISHED / pattern-path CTS_UPDATED); `snapshot_horizon_idx` is still the
BOS_1 anchor until E3a′. The MS in-flight resolver passes the triggering event's
moment as `fill_horizon_idx` too (lock-step). The moment of the question stays ONE parameter
(`evaluated_at`). `select_fib_anchor_for_cycle` also
requires `evaluated_at`: FibTracker passes the CTS_1 moment, the MS in-flight
resolver `None`.

### MarketStructure — uncut by decision

| Site | Window | `check_to_idx` | `evaluated_at` | Question |
|---|---|---|---|---|
| `market_structure._update_cycle0_data` `:2083` | `[BOS_0, CTS_0]` | `cts_idx` (E3a′) | `None` (cond2 mirror; read later) | In-flight Scenario-2 cycle-0 snapshot, mirroring FibTracker's cache |
| `market_structure._refresh_poi_inners_for_cycle` → `compute_poi_inners_for_cycle` → `select_fib_anchor_for_cycle` cond1 | `[BOS_1, CTS_1]` | `fill_horizon_idx` = the triggering event's moment (apply / processing candle; Plan E E3a, lock-step with FibTracker) | `None` | In-flight Scenario-2 cond1 |

This mirror is load-bearing: the in-flight POI resolver reads
`cycle0_data["has_unfilled"]` to make the same Scenario 2 decision as
FibTracker's downstream layer. Both use the same direction filter AND both stay
uncut, or anchor selection diverges (see `LANDMINES.md` "Scenario 2 anchor
agreement").

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

POI IC validation (`find_ic_candidates`) was already strict — it asks a
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
| MS in-flight snapshot | none (uncut by decision; 3 inner lists would change, none is ever read) |
| IC identification | none (uncut; retrospective) |

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
  "one candle later" and close the M1 divergence.
