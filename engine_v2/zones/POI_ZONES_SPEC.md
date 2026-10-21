# POI Zones Specification

> Point of Interest (POI) Zones using Fibonacci retracement and Institutional Candle identification.

---

## Overview

POI Zones are derived from:
1. **Imbalance Candle Pattern** — Identifies large price movements (standard FVG)
2. **Fibonacci Retracement Levels** — Drawn between BOS and CTS anchor points
3. **Institutional Candle (IC)** — ONE per fib record, chosen by the *institutional pattern* among the IC candidates
   (§3 Step 1 — every IC condition, the immediate unfilled imbalance and the 61.8%-80% touch included; Step 1A: the
   80%-100% touch when Step 1 finds none — PLAN_P); its high/low define the zone (since PLAN_H landing B, 2026-10-06 — until then up to 3
   ICs per fib, the V30 / V60 / V90 variants, §3.3)

---

## 1. Imbalance Candle Pattern

### Two concepts

- **Imbalance candle** — a c2 (middle) candle whose neighbors form an FVG
  *and* whose own direction matches the gap direction. Flagged per-candle via
  the `is_imbalance` dataframe column. Charting consumes this.
- **Imbalance instance** — one imbalance candle or a run of consecutive
  same-direction imbalance candles merged into a single entity with merged
  gap bounds. Stored as `ImbalanceInstance` objects in
  `df.attrs["imbalances"]`. Fib/POI logic consumes this.

A single-candle instance has `start_idx == end_idx`.

### Detection Logic (per-candle)
- **Bullish imbalance candle:** `c1.high < c3.low` AND `c2.direction == +1`
- **Bearish imbalance candle:** `c1.low > c3.high` AND `c2.direction == -1`
- The c2-direction requirement prevents counter-direction candles from being
  flagged (e.g., a gap with a bearish c2 is not a bullish imbalance).

### Merging Rule
Consecutive imbalance candles with the **same direction** are merged into a
single `ImbalanceInstance`. A direction break or a non-imbalance candle ends
the run.

### Merged Gap Bounds
- **Bullish:** `gap_bottom = df[start_idx-1].h` (first c1.high),
  `gap_top = df[end_idx+1].l` (last c3.low)
- **Bearish:** `gap_bottom = df[end_idx+1].h` (last c3.high),
  `gap_top = df[start_idx-1].l` (first c1.low)

### Implementation
- DF column `is_imbalance` (0/1) — every imbalance candle gets the flag
- `df.attrs["imbalances"]` — list of `ImbalanceInstance` with merged bounds
- Computed once in pipeline before market structure (`compute_imbalance`)
- Lower-TF (M15) pipelines re-run `compute_imbalance` after slicing because
  `df.attrs["imbalances"]` indices don't survive `reset_index(drop=True)`

### Role in POI Zones
- An unfilled imbalance must exist **overlapping the Fib's BOS→CTS span** for
  the Fib — and so its POIs — to exist (FibTracker's `has_unfilled_imbalance` check)
- Imbalance must be **immediately after the IC candle** (its first gap candle
  at IC + 1 … IC + 2 — IC condition 3 since PLAN_P; any gap between the IC and
  the CTS anchor until then) — POI IC validation passes `direction=sd` so only
  structure-direction imbalances qualify; the POI's activation keeps its own
  any-gap-after-the-IC condition (§4 "Lifecycle")
- **"Exists" means FORMED (Plan F, 2026-09-24).** An instance exists from its
  first c3, `ImbalanceInstance.formed_at = start_idx + 1` — never from its c2.
  A question asked at a moment counts only instances formed by then, and only
  their **formed prefix** (`overlaps_formed_prefix`); the POI activation sweep
  enters an instance at `formed_at` (§4 "Lifecycle"). Canonical:
  `IMBALANCE_FILL_SEMANTICS.md` "Knowability — the c3 rule".

### All Fib + scenario + POI imbalance checks are sd-direction strict
Every consumer of `has_unfilled_imbalance` — Fib activation (the downstream
FibTracker and, since PLAN_I S5, MarketStructure's embedded one), the cross-
cycle own test + dead-cycle walk, AND POI IC validation — passes `direction=sd`
(the Scenario 2 cond1/cond2/cond3 and MS's cycle-0 snapshot were retired in
PLAN_I S4 / S5). The
rationale is uniform: **Fibs only ever produce sd-direction POIs by
construction** (POIs are sd-direction — see §4 "POIs are always sd-direction"),
so a counter-direction imbalance inside a structural span cannot influence
any downstream POI and shouldn't drive the Fib's lifecycle either.

The pre-2026-05-23 design left Fib activation permissive on the grounds that
the BOS→CTS span is structurally directional. That reasoning still holds —
counter-direction imbalances in the span are geometrically unusual — but the
extra permissiveness was disconnecting cause (any-direction unfilled imbalance)
from effect (sd-direction POIs). The strict filter aligns the question
("should this Fib be drawn?") with what it actually affects downstream
("the sd-direction POI set").

The more load-bearing distinction across call sites is the primitive's **two
as-ofs** (Plan F, 2026-09-24):

- **`check_to_idx` — the fill horizon** (`is_filled` scans `(end_idx,
  check_to_idx]`). Fib lifecycle and scenario checks pass varying horizons: the
  handled event's moment (FibTracker, since Plan E E3a / E3a′), a
  fixed reference event's moment (BOS_1 for cond3, since E3a′), or the cross-cycle routine's
  `current_candle`. POI IC validation passes `check_to_idx = cts_idx`, the
  fib's CTS (IC condition 3's window ends at min(IC + 2, `cts_idx`) since
  PLAN_P; the retired any-gap test — information — at `end_idx = cts_idx`).
- **`evaluated_at` — the moment the question is asked** (keyword-only,
  REQUIRED): only instances formed by then count. FibTracker passes the handled
  CTS event's moment (`event_fields.event_moment`); POI IC validation passes
  `evaluated_at=None`, an explicit "no cut" (§3.1 condition 3) — in the drawn
  POI and in MarketStructure's in-flight pick (whose FIB, since PLAN_I S5, is its
  embedded FibTracker's, cut at the moment) (IMBALANCE_FILL_SEMANTICS call-site
  matrix). The cut is keyed on the moment,
  never on `check_to_idx`.

### Fill Check (per instance) — two-stroke state machine

Filled iff BOTH strokes fire by `check_to_idx`:

- **Stroke 1 — armed (65 % retrace — `fill_threshold`, the one constant `IMBALANCE_FILL_THRESHOLD`; 70 % until
  PLAN_S step 1, 2026-10-21).** First candle in `(end_idx, check_to_idx]`:
  - Bullish: `low <= gap_top - gap_size * fill_threshold`
  - Bearish: `high >= gap_bottom + gap_size * fill_threshold`
- **Stroke 2 — confirmed (close at / past gap outer).** First candle at idx
  `> armed_idx` — a separate, later candle; the arming candle's own close never counts (PLAN_S step 2):
  - Bullish: `close >= gap_top`
  - Bearish: `close <= gap_bottom`

Both strokes latch monotonically (stroke 1 then stroke 2; neither un-latches).
Scan starts at `end_idx + 1` — the **last c3**, which cannot arm its own gap
(its wick IS the gap edge; stroke 1 needs 65 % of the gap past it); the
instance's c2s are excluded.

See `engine_v2/IMBALANCE_FILL_SEMANTICS.md` for the canonical reference —
state machine details, edge cases, consumer call-site matrix, and the
rationale for the two-stroke definition.

---

## 2. Fibonacci Retracement Levels

### Levels
```python
FIB_LEVELS = [30, 50, 61.8, 80]  # percentages
```

### Anchor Points
- **Anchor 1 (BOS):** the BOS anchor (`ef.bos_anchor_idx`) & its price — **LOCKED** once established
- **Anchor 2 (CTS):** idx & price of CTS — **UPDATES** as CTS moves to new extreme
- **Only draw Fib if unfilled imbalance exists between the anchor points**

### Calculation
For bullish swing (retracement from high):
```python
fib_price = anchor_high - (anchor_high - anchor_low) * (level_pct / 100)
```

For bearish swing (retracement from low):
```python
fib_price = anchor_low + (anchor_high - anchor_low) * (level_pct / 100)
```

### FibTracker Lifecycle

```
CTS_ESTABLISHED (cycle 1+)
    ↓
Check: unfilled imbalance between BOS and CTS, FORMED by the event's moment?
    ↓ (yes)
Fib ACTIVATED: BOS idx/price → CTS idx/price
    ↓
On CTS_UPDATED:
  - Anchor 2 (CTS) UPDATES to new extreme
  - Re-check imbalance condition (formed by the update's moment) → can DEACTIVATE or REACTIVATE
    ↓
CTS_CONFIRMED
    ↓
Fib LOCKED (anchor 2 stops updating)
```

**Key behaviors** (since PLAN_I S4, 2026-10-08, ONE logic for main and sub — `CROSS_CYCLE_FIB_SPEC.md`):
- **Every structure's cycle 0** gets its single Fib (main's sid 0 had none before S4 — Q6)
- **A cycle ≥ 1** gets a cross within the structure's cross-fib limit, else a single (the Scenario Logic below is
  RETIRED — kept as the pre-S4 record)
- **Deactivation/Reactivation:** Fib can toggle active state based on imbalance conditions at each CTS update
- **Each check is asked at the handled event's MOMENT** (Plan F, 2026-09-24),
  `event_fields.event_moment(ev)` — per-event values in ARCHITECTURE
  "`ev.idx` convention" (a pattern-path CTS_UPDATED: its apply candle, since Plan E E3·0).
  FibTracker re-asks only at CTS events, so a
  gap whose c3 closes after an event counts only if a later event re-asks — none
  re-asks a cycle ≥ 1 cross that failed its own test at EST until a CTS_UPDATED (the single re-asks on every update;
  until PLAN_I S4 an H1 cycle ≥ 1 fib was one-shot at EST — dropped, not delayed; IMBALANCE_FILL_SEMANTICS "Decided
  at the event")
- **Obsolescence:** When new cycle forms, previous cycle's Fib becomes obsolete

### Scenario Logic (Post-Reversal, sid 1+) — RETIRED (PLAN_I S4, 2026-10-08)

> The H1-main rule below no longer runs: main follows the one logic (`CROSS_CYCLE_FIB_SPEC.md`; this rule's own
> record is its Appendix A). Kept as the pre-S4 description.

For structures after a reversal, Fib activation follows a 3-scenario system:

#### Scenario 1: Normal Cycle 0 Fib
**Condition:** the CTS_0 event's moment >= reversal_confirmed_idx (the EST
`confirmed_at` / the update's moment — Plan E E3a; before it the CTS_0 anchor idx;
`reversal_confirmed_idx` = the realised reversal of the previous sid,
`STATE_CHANGED(to=reversal).idx` — the same value as the zone end below — since
2026-09-27; before, the last `REVERSAL_CANDIDATE`'s scheduled apply)

**Behavior:**
- Cycle 0 gets normal Fib (if unfilled imbalance)
- Cycle 1 gets normal Fib (if unfilled imbalance)
- Skip Scenario 2/3 checks

**Revert Condition (checked at CTS_1 ESTABLISHED):**
- If BOS_1 price touches/crosses into prev structure's last BOS zone → revert to FALSE
- Deactivate cycle 0 Fib, proceed with Scenario 2/3
- "Touch" means: BOS_1 >= zone outer (for buy zone) or BOS_1 <= zone outer (for sell zone)

#### Scenario 2: Cross-Cycle Fib
**Condition:** Scenario 1 is FALSE AND all 3 conditions met:
1. cond1: Cycle 1 has unfilled imbalance
2. cond2: Cycle 0 has unfilled imbalance
3. cond3: BOS_1 doesn't fill cycle 0's imbalances

As-ofs (Plan F): FibTracker asks cond1 at the CTS_1 ESTABLISHED moment (formed
gaps only); cond2 reads the cycle-0 liveness cache, stored UNCUT (every gap in
`[BOS_0, CTS_0]` has formed by CTS_1 ESTABLISHED); cond3's window ends before any
moment. MarketStructure's in-flight resolver asks all three uncut, so the layers
agree on cond2 / cond3 and can differ on cond1 — the accepted M1 divergence
(IMBALANCE_FILL_SEMANTICS "Knowability — the c3 rule"; LANDMINES "Scenario 2
anchor agreement"). [MS's side retired too, PLAN_I S5: MS reads its embedded
FibTracker's decision — CROSS_CYCLE_FIB_SPEC §8.]

**Behavior:**
- No cycle 0 Fib
- Cycle 1 gets cross-cycle Fib: BOS_0 → CTS_1
- Normal Fib stored as fallback

#### Scenario 3: Normal Cycle 1 Fib
**Condition:** Scenario 1 is FALSE AND Scenario 2 conditions not met

**Behavior:**
- No cycle 0 Fib
- Cycle 1 gets normal Fib (if unfilled imbalance)

### Cross-Cycle Mode (every structure since PLAN_I S4; was `fib_mode="cross_cycle"`, subs only)

> Formerly named `m15_reverse`. Renamed when subordinate structures grew
> beyond M15 + counter direction to include confluence variants (same
> direction as parent) and potentially deeper TFs.

Subordinate-structure pipelines use a Fib model that generalizes the
cross-cycle concept to **any** cycle (not just cycle 1) AND allows cross
fibs to form **before** a cycle's CTS is established. Used by both counter
and confluence subordinate variants regardless of direction. **Since PLAN_I S3
(2026-10-08) a cross is bounded by the sub's cross-fib LIMIT:** a cycle past it
(or any cycle >= 1 of a sub without a reference price) gets a single fib
(`via: "cross_fib_limit"`) and no pre-established cross — canonical
`CROSS_CYCLE_FIB_SPEC.md` §2 / §3.

#### Phase state per (sid, cycle_id)
- `pre_established` — cycle's CTS not yet established. Only cross fib checks
  run (single fib requires CTS).
- `established` — CTS_n ESTABLISHED fired. Cross-first-then-single dispatch.
- `confirmed` — CTS_n CONFIRMED fired. Fib locked. Phase for cycle n+1 flips
  to pre_established.

Cycle 0 has no pre_established phase (there's no CTS_-1 to trigger it).

#### Cross-fib condition (generalized)

For target cycle n+1, walk backward from cycle n to 0, skipping cycles in
the `_dead_cycles` cache (permanently-filled cycles). For each live cycle
k, check whether its own swing `[BOS_k, CTS_k]` still has unfilled
imbalance with fill-check extended to `current_candle` (Interpretation B — the
fill horizon: the handled event's moment since Plan E E3a; CROSS_CYCLE_FIB_SPEC).

The earliest contiguous cycle `x` (where all cycles `x..n` are live) becomes
the cross fib's BOS anchor. If no prior cycle qualifies, cross fails and —
in established phase — single fib is activated as fallback.

#### Anchor by phase
- **pre_established** (cycle n+1): anchor = running extreme past CTS_n;
  own imbalance range = `[prospective_BOS_n+1, current_candle]` where
  `prospective_BOS_n+1` is the deepest pullback since CTS_n CONFIRMED
- **established** (cycle n+1): anchor = CTS_n+1; own range = `[BOS_n+1, CTS_n+1]`

The own-imbalance test counts only gaps FORMED by the event's moment
(`evaluated_at`, Plan F). In pre_established the moment IS `current_candle`, so a
gap whose c2 is `current_candle` does not count yet. Reference window: the cross
fib that counter sub 5 used to PRE-CREATE at 3806 for a cycle 1 it never
establishes (sub 5 only ever establishes cycle 0), and that fib's IC 3654 twin
POI, no longer exist — the only gap in its own-imbalance window was the instance
(3806, 3806), formed at 3807, and no later event re-checks it (§4 field list).

#### Triggers
- pre_established: `CTS_THRESHOLD_UPDATED` events for cycle n drive re-checks
  (natural trigger — fires on every new running extreme past CTS_n; its `ev.idx`
  is the processing candle = the moment)
- established: `CTS_UPDATED` events for cycle n+1

#### State transitions (cross fib)
- Extension (same earliest cycle `x`, new anchor): replace in place, same version
- Shrink (`x` moves forward because a cycle went dead): deactivate old version
  with `deactivated_by="cross_shortened"`, create new with `version+1`
- Cross fails (earliest_x == target_cycle): deactivate with
  `deactivated_by="cross_failed"`, fall back to single (established only)
- Own imbalance filled: deactivate cross with `deactivated_by="own_imb_filled"`
- Lifecycle end (the sub's window ended — `reversal` / `same_dir_replacement` /
  `parent_end`): still-active unlocked fibs get a terminal `end_idx` /
  `end_reason` = the sub's `end_reason` via `compute_cycle_lifecycle`'s
  `lifecycle_cap` / `cap_reason` (FIB_LIFECYCLE_SPEC §15.6). Historical: the
  pre-Phase-3 form was `deactivated_by="lifecycle_end"`; `"lifecycle_end"` is
  no longer emitted (Plan C, 2026-09-20)
- CTS_n+1 CONFIRMED: lock active fib (`locked=True`)

#### Storage — versioned keys
Cross fibs live in `_fibs` under key `(sid, cycle_id, "cross", version)`
alongside single fibs at `(sid, cycle_id)`. Old versions are preserved as
inactive snapshots so the chart can render them with faded styling.

### Prev BOS Line (Visualization Helper)

After each reversal, a black horizontal line shows the previous structure's last BOS (the Scenario 1 revert
threshold until PLAN_I S4 retired the revert; the line is still drawn):
- **Start idx:** the previous structure's last BOS anchor (`ef.bos_anchor_idx`)
- **End idx:** the CTS anchor of the new sid's earliest-moment CTS_ESTABLISHED / CTS_UPDATED whose moment is
  at or after reversal_confirmed_idx (Plan E E3d; the END is a location, Q6)
- **Price:** Last BOS price of previous structure

### Unfilled vs Filled Imbalance

- **FVG gap** = distance between candle 1 wick and candle 3 wick
- The gap exists once its c3 closes (`formed_at`, §1 "Role in POI Zones")
- Check candles in **`(end_idx, check_to_idx]`** — from the last c3
- **Filled** requires TWO strokes within the scan range (see §1
  "Fill Check" and `IMBALANCE_FILL_SEMANTICS.md` for the canonical
  definition):
  - Stroke 1: price retraces ≥ 65 % into the gap (low past 65 % for
    bullish, high past 65 % for bearish) → instance is **armed**
  - Stroke 2: at a LATER candle (never the arming one — PLAN_S step 2), close at / past the
    gap outer in the imbalance's direction (≥ gap_top bullish,
    ≤ gap_bottom bearish) → instance is **confirmed-filled**
- **Unfilled:** stroke 1 hasn't fired yet, OR stroke 1 fired but stroke 2
  hasn't yet — both cases keep the instance "in play". Only a FORMED instance
  can be in play: one whose first c3 has not closed by the moment of the
  question is not an imbalance yet (neither filled nor unfilled)

---

## 3. Institutional Candle (IC) Identification

> **One IC per fib record since PLAN_H landing B (2026-10-06; Week 9 Part 1 item 1, register `IDEA_PARKING_LOT.md`
> §D.1).** Canonical reading + every decision: `../plans/PLAN_H_institutional_pattern_poi.md` §1 / §6; the pattern
> itself: `patterns/institutional_pattern.py`. Before B, §3.3's V30 / V60 / V90 selection picked 1–3 ICs per fib (its
> code is kept, unwired — PLAN_H X1).

IC identification, per fib record (BOS anchor `b`, CTS anchor `c`, direction `sd`), in the user's STEPS — each step,
and each rule set inside it, runs in order (the flow PLAN_H §6 RX1-F; the holistic write-up
`../plans/plan_h_inputs/ic_identification_revamp_spec.md`, decisions PLAN_H §6 ICR-S, ICR-1 … ICR-15; built by
`../plans/PLAN_L_ic_identification_steps.md`, landing 1 = this structure, 2026-10-09; landing 2 = Step 3, 2026-10-10):

1. **Step 1 — the IC candidates** (§3.1, `check_ic_conditions` / `find_ic_candidates`): every candle of the fib's
   `b … c` window (the scope, not a condition) whose agreed direction is −sd (condition 1) is checked against the
   other IC conditions — (2) the IC reference condition (§3.2), (3) an unfilled imbalance IMMEDIATELY after it (its
   first gap candle at IC + 1 … IC + 2 — since PLAN_P, 2026-10-13; any unfilled imbalance after it until then), (4)
   the touch of the 61.8–80 zone — ALL applied TOGETHER. An **IC candidate** passes all four
   (`ICCandleCheck.is_ic_candidate`). None → Step 1A.
   **Step 1A — the 80–100 fallback zone** (PLAN_P, the user's change 3): condition 4 re-evaluated on the 80–100 zone,
   conditions 1–3 unchanged (`ic_zone_band` `80-100`); it runs ONLY when Step 1 finds no IC candidate — never after
   Steps 2–3 (an all-pinbar answer stays no IC). None again → no IC (`no_ic_candidate` — never attributed to one
   condition; Step 4 would run).
2. **Step 2 — the institutional pattern** (§3.4, `select_institutional_pattern`) among the IC candidates: the best
   (rank, shape); patterns tied on it → the **tiebreak rules** (`apply_tiebreak_rules`: the IC's candle type → the
   most recent before the imbalance; the immediate unfilled imbalance ran first until PLAN_P). A pattern → the IC,
   **`ip_outcome` `pattern`**; none → `ip_outcome` **`no_pattern`** → Step 3.
3. **Step 3 — the ranking rules** (§3.6, `apply_ranking_rules`; PLAN_L landing 2, 2026-10-10) over the UNFILTERED
   Step 1 list — exclude pinbars → maru > normal → the most recent before the imbalance (after the oldest candidate
   still in play; the immediate unfilled imbalance ran second until PLAN_P); a rule set of its OWN, separate from the
   tiebreak rules (ICR-S; one helper per rule the two sets share, each given its own input list — ICR-9). Every IC
   candidate a pinbar → no IC (`all_ic_candidates_pinbar`).
4. **Step 4 — the lower-TF fallback: DEFERRED, possibly never** (register `IDEA_PARKING_LOT.md` §D.2): Steps 1–3 on
   the lower TF when no IC candidate remains (`no_ic_candidate` / `all_ic_candidates_pinbar`).

The user's flow (write-up 2026-10-13, verbatim in `../plans/qa_ic_crossfib_inputs/user_writeup_20261013.md`):
Step 1 → candidates → Steps 2 / 3 → (no IC → Step 4); no candidates → Step 1A → candidates → Steps 2 / 3 → (no IC →
Step 4); none → Step 4. "It is not something applicable further downstream."

`select_ic` runs the steps → `ICSelection` (`pick`, Step 2's `tiebreak`, `ip_outcome`, Step 3's `ranking`,
`no_ic_reason`, `ic_zone_band` — the zone condition 4 was evaluated in: `61.8-80`, or `80-100` when Step 1A ran; a
no-IC answer with `80-100` = Step 1A ran and found none either). No IC → **no POI** for that record (PLAN_H D7; the no-IC reasons `no_ic_candidate` /
`all_ic_candidates_pinbar`, §3.5 — RX1-N, ICR-3): MS's sd-proximity and the zone-proximity scan then read the BOS
inner only for that cycle. A pattern's IC and a ranked IC are drawn and read alike (one POI, one inner — PLAN_H D2,
ICR-12); they differ only in the activation floor's pattern term (§4) and in the hover / CSV fields.

### 3.1 Step 1 — the IC conditions, applied together

`check_ic_conditions` builds one `ICCandleCheck` per candle of the window whose agreed direction is −sd (condition 1 =
the row set; index order), with the outcomes of conditions 2–4; `find_ic_candidates` = the checks passing all (index
order). The numbering is the user's (the write-up's Step 1):

- **The window (the scope, inclusive):** `BOS_idx <= candle_idx <= CTS_idx` (a missing index skipped).
1. **Opposite direction of struct_direction — the AGREED direction:** `agreed_direction(direction, vol_dir) ==
   -struct_direction` (`features/candles_v2.agreed_direction`, GLOSSARY "Agreed direction"): the candle and volume
   directions agree, or one is 0 and the other decides — so a **volume-only doji** can be an IC and a **split
   candle** (opposite candle / volume directions) cannot (PLAN_H VD). The unwired §3.3 path's finder
   `find_ic_variant_candidates(ic_direction=None)` keeps the candle direction.
2. **The IC reference condition** (§3.2) — `ICCandleCheck.ic_ref_reason` (None = passes).
3. **The IMMEDIATE unfilled imbalance** (since PLAN_P, 2026-10-13 — the user's change 1: "There must be immediate
   unfilled imbalance after the candle, within IC + 2 (<=)"; until then the first tiebreak rule and the second ranking
   rule, PLAN_H §6 TBU-1 … TBU-5): an unfilled sd imbalance whose first gap candle (its c2; a merged imbalance's
   first) is at IC + 1 … min(IC + `POIConfig.immediate_imbalance_candles` (2), the CTS anchor) — raw candle offsets (a
   true doji takes a slot), the candle may be the gap's c1, the gap's c3 never counts — "unfilled" = the two-stroke
   fill through the fib's CTS anchor (a gap that fills only after the anchor, e.g. on the retrace into the POI, counts
   as unfilled) — `immediate_unfilled_imbalance_first_c2_idx`, `ICCandleCheck.immediate_imbalance_first_c2_idx` (the
   gap's first candle; None = fails), computed on EVERY check. Consecutive sd gap candles merge into one instance and
   no gap straddles a checked candle (its candle direction is −sd or 0), so at most one imbalance sits in the window.
   **No knowability cut** (`evaluated_at=None`, Plan F): ICs are identified retrospectively on the final fib (and,
   in MarketStructure's in-flight snapshot, read only at candles after the CTS); WHEN the POI can go live is the
   activation sweep's job, which counts only formed gaps (§4 "Lifecycle").
   The user's reasoning (verbatim): "Having immediate imbalance is actually a stronger indication of a good IC and
   where future price likely would retrace to … having imbalance in general in a cycle is already required via Fib
   levels, and isn't as important of an indicator of IC strength as having IMMEDIATE imbalance after the IC
   candidate."
   **The retired condition 3** — at least one unfilled sd imbalance anywhere in `(candidate_idx, CTS_idx]`, fill
   horizon `CTS_idx` (`_has_unfilled_imbalance_after`) — is kept as INFORMATION on every check
   (`ICCandleCheck.has_unfilled_imbalance_after`, the CSV column; the user, 2026-10-13: "Keep as information") and as
   the unwired §3.3 finder's filter; the pick never reads it. (An immediate gap implies it.)
4. **The touch:** the candle touches the fib's 61.8–80 zone (inclusive: `h >= zone_lo and l <= zone_hi`;
   `fib_zone_bounds` + `ic_touches_zone`) — `ICCandleCheck.touches_zone`. **Step 1A** (PLAN_P): when no check passes
   all four, `select_ic` re-evaluates `touches_zone` on the **80–100 zone** (`POIConfig.ic_fallback_fib_min` /
   `ic_fallback_fib_max` = 80 / 100 through `ic_zone_config` — the one place a band becomes zone parameters; for a
   buy fib [the BOS price, the 80 % price]), conditions 1–3 reused (none depends on the zone), and records
   `ic_zone_band` = `80-100`. The candidates it finds go through Steps 2–3 as usual. Never after Steps 2–3. On the
   pick, 80–100 and 61.8–100 are indistinguishable (a check touching 61.8–80 already failed condition 2 or 3); the
   band shows in the POI activation's touch re-check (§4 "Lifecycle") and the CSV's `touches_zone`. The 100 % level
   is the BOS anchor's extreme, so the BOS anchor candle always touches the fallback zone (inclusive edges) — an
   agreed −sd BOS anchor candle with an immediate gap can be a Step 1A IC (confluence sub 1 cycle 2's IC 1934 is its
   record's BOS anchor).

If Fib is deactivated (neither active nor locked), there are no bounds → no checks (and no Step 1A: band
`61.8-80`).

### 3.2 The IC reference condition (PLAN_K — computed + exported since landing A, applied since landing B, 2026-10-09)

> The user's rule (2026-10-09; PLAN_H §8.9 "RX1" + §6 RX1-*, the write-up verbatim
> `../plans/plan_h_inputs/ic_reference_conditions_spec.md`), built by `../plans/PLAN_K_ic_reference_condition.md`. It
> replaces the per-scenario conditions designed in Week 7 and never implemented (Appendix A). **Landing A** computes it
> for every §3.1 candidate and exports it (`ICCandleCheck.ic_ref_reason` since PLAN_L L1, the CSV column §3.5, the fib meta + hover) with
> the switch `poi_zones._IC_REFERENCE_FILTER = False`: the pick is UNCHANGED. **Landing B** (2026-10-09) applies it
> in both consumers (`ICCandleCheck.is_ic_candidate` since PLAN_L L1: every IC condition; the switch removed). Any IC filter stays outside the institutional pattern and runs before it (PLAN_H
> D4).

ONE bundled condition (RX1-B) per checked candle `i` (§3.1) of a fib record — structure `sid`, direction `sd`, target cycle
`T` (`fib_state.cycle_id` — a single's cycle, a cross's last cycle):

1. **The references** — built by ONE method, `FibTracker.ic_reference` → `ICReference(price, source,
   break_pattern_anchor_idx, break_confirmed_idx)` (`zones/cross_cycle_fib.py`; `ICReference.NONE` = no reference):
   - **`prev_cycle_cts`** (when `T ≥ 1` — a single and a cross alike, `../plans/PLAN_O_ic_reference_target_cycle.md`,
     the user 2026-10-13: "The CTS reference will always be the max cycle-1 CTS"; until then a cross used its START
     cycle's previous CTS): cycle `T−1`'s `CTS_CONFIRMED` price — the confirmed anchor, never a `cts_threshold`. Its
     bound = the first candle of the pattern that ESTABLISHED cycle `T` (that `CTS_ESTABLISHED`'s `pattern_anchor_idx`
     — the pattern that broke `CTS_{T−1}`), known from that event's moment. A cross whose target is not established
     yet (pre-established, CROSS_CYCLE_FIB_SPEC §4) has no bound — the price side only (the user, 2026-10-13); only the
     stamp can read one (a cross still pre-established at its structure's end / the data edge — the drawn POI gets no
     record without a CTS_ESTABLISHED, MS reads only its established cycle; PLAN_O §2). A cross from cycle 0 therefore
     has a `prev_cycle_cts` too.
   - **The structure reference** (when the structure has a cross-fib limit `P` and `M` is None or `T ≤ M`, cycle 0
     included — CROSS_CYCLE_FIB_SPEC §3; no limit entry or `P` None → none; for a MOVING `P` the gate is T's cross
     permission, fixed at T's establishment, else read live as of the reading moment — PLAN_M CL-1):
     - **`prev_structure_bos`** (`P` = `reversal_bos_frozen` — a main or CONFLUENCE-sub reversal start; a COUNTER
       sub's reversal start has `parent_bos_threshold`, below): the reversing structure's most recent
       `BOS_CONFIRMED` price by its reversal (`CrossFibLimit.ref_anchor_price`); its bound = the REALISED reversal
       pattern's first candle (`STATE_CHANGED(to=reversal).meta["pattern_anchor_idx"]`), known for the structure's
       whole run, like `P` (RX1-K7);
     - **`cross_limit_anchor`** (every other source with a `P`, RX1-2): the anchor behind `P` (`ref_anchor_price`,
       CROSS_CYCLE_FIB_SPEC §3's table); its bound = the cross-limit breakout watch's break of `P`
       (CROSS_CYCLE_FIB_SPEC §3), counted once its confirmation ≤ the reading moment — none → no bound. A limit whose
       `P` MOVES (PLAN_M CL-7; CROSS_CYCLE_FIB_SPEC §3 "A moving `P`") is watched at `P` as of the reading moment: a
       break of a `P` the parent later moved past stops counting.
   - **The choice:** both → the one further opposite the structure (sd +1 the lower price, sd −1 the higher — also when
     the previous CTS already went past the BOS anchor, RX1-3); EQUAL prices → the structure reference and its bound
     (RX1-K5); neither → the condition PASSES (RX1-B / RX1-4).
2. **The price side:** sd +1 the candle's HIGH strictly below the reference, sd −1 its LOW strictly above
   (`passes_price_constraint`); else `ic_past_reference`.
3. **The before-the-break side:** `i` strictly before the bound when one is known (no known bound → passes, RX1-2a);
   else `ic_not_before_break`. Both failing → `ic_past_reference` (the price side is reported first, K-Q2). An IC can
   never BE the bound candle (a breakout / reversal pattern's first candle has the structure's candle direction).
4. **Where:** `poi_zones.ic_reference_reason`, called by `check_ic_conditions` per checked candle, with the other
   IC conditions (Step 1) and before the pattern (RX1-E) — both consumers: the drawn POI with the record's FINAL reference (as of the structure's end,
   `evaluated_at=lifecycle_cap`, the watch's detector on the whole frame — the same call
   `FibTracker.stamp_ic_references` exports; PLAN_K §11.2 LA1); MS's in-flight read
   with its embedded tracker AT the fed event's moment (`poi_inner_for_fib(..., tracker=, evaluated_at=)`, built
   inside the read's re-raising scope — LANDMINES "The POI Resolver Fails Loudly"). Not in the activation sweep; `M`,
   the legs, the touch, the pattern and the tiebreakers are unchanged (since B, the tiebreakers run over the surviving
   tied ICs).

On the reference window (PLAN_K §5 / §11, S_K_A): every one of the 26 POI records' values; e.g. sub 0 cycle 3 (a buy):
`prev_cycle_cts` 0.57774 (CTS_2), bound 1720 — 209 candidates, 85 pass; its landing-A pick 1604 (high 0.57880) is
past the reference, so from landing B the pick was 1447 1A.iv (high 0.57768; over 1660 by the most recent before the
imbalance) — and since the IP tiebreaker update (2026-10-09, §3.4) it is 1660, whose immediate gap 1662 is unfilled
while 1447's (1448) filled at 1480. The only other change: sub 3 cycle 2 keeps 2969 but loses its candle-type tie
(2988 cut); no structure event moves (PLAN_K §11.3 /
§11.4). `ic_not_before_break` cuts no candidate on the window (the price side already cuts every one the bound
would).

Since PLAN_O (2026-10-13, vs S_N_hover; PLAN_O §6): the window's 4 cross records change, the singles do not — H1 sid 1
cross 0→1 0.5736 `prev_structure_bos` → 0.57902 `prev_cycle_cts` (CTS_0, bound 747; pick 732 kept); sub 3 cross 1→2
0.5818 (CTS_0) → 0.57902 (CTS_1, bound 2991; pick 2821 → 2837 1A.iv); sub 7 cross 0→1 / 0→2 0.58105
`cross_limit_anchor` → 0.57674 (CTS_0, bound 4154) / 0.57798 (CTS_1, bound 4178; picks 4051 / 4083 kept); no
structure event moves. (Since PLAN_P, 2026-10-13, the picks follow IC condition 3 + Step 1A — §3.1; sub 0 cycle 3
keeps 1660, now without a tie: 1447 fails condition 3.)

### 3.3 IC Variants — UNWIRED since PLAN_H landing B (2026-10-06)

> Kept in the code (`select_ic_variants`, `POIConfig.v30 / v60 / v90`) with its tests as a possible no-pattern
> fallback (PLAN_H X1: "There is still a chance that I use those as fallbacks … we might still delete in the future").
> The user's IC identification write-up (2026-10-09) gave the no-pattern role to Step 3, the ranking rules (§3.6,
> PLAN_L L2), and kept this code unwired ("Keep it for now", PLAN_H §6 ICR-5); the remaining fallback is register §D.2 "Step 4 of
> IC identification". Nothing calls it. The rule as it was:

From IC candidates, find the **most recent candle** meeting each overlap threshold:

| Variant | Min Overlap | Description |
|---------|-------------|-------------|
| **V30** | 30% | Most lenient |
| **V60** | 60% | Middle |
| **V90** | 90% | Most stringent |

**Overlap Calculation:** What % of candle falls within 61.8%-80% Fib zone (`calculate_candle_overlap_pct`).

**Selection:** For each variant, scan candidates from most recent (highest idx) and pick first that meets threshold;
grouped by unique IC (`versions`, e.g. `["V60", "V90"]` — the POI meta key until landing B).

### 3.4 Step 2 — the institutional pattern (`select_institutional_pattern`) and its tiebreak rules

Around an IC candidate, read with the agreed direction (PLAN_H §1 items 2–9):

- **initial leg** (agreed sd) · **opp leg** (agreed −sd; the IC is its LAST candle — after the IC and any true dojis
  the next candle must be agreed sd) · **continuation leg** (agreed sd);
- legs are the maximal consecutive runs, cut at the fib's BOS / CTS anchors (the anchor candles are members); the
  initial leg = its run's most recent `cap` candles, the continuation = its run's first `cap` (`POIConfig.
  ip_leg_cap_by_tf`: H1 3, M15 3 — an unknown TF raises); the opp run is never cut;
- **true dojis** (candle AND volume direction 0) never break a run and are full members of the leg they are assigned
  to: between opp and continuation → the continuation; between initial and opp → tried in each leg (J1); never the
  pattern's first or last candle (E1);
- a leg is **stacked** (first open, max high, min low, last close) and classified by the SAME classifier and per-TF
  params as a single candle; big flags = the single-candle ratio vs the 5 marus before the pattern's first candle
  (`pattern_anchor_idx`): big_maru = maru + ratio ≥ 0.65, big_normal = normal + ratio ≥ 0.5;
- **requirement:** the continuation is big_maru with a stacked body strictly greater than the opp leg's;
- **ranks** (best first) 1A big_maru·maru·big_maru, 1B maru·maru, 2A maru·normal (or big_maru·normal), 2B
  big_normal·maru, 2C big_normal·normal, 3A maru·pinbar (or big_maru·pinbar), 3B big_normal·pinbar, 3C maru·MULTI opp,
  3D a SINGLE big_normal (cap 1, evaluated next to the cap-`cap` initial)·MULTI opp — the IC's `candle_type` is the
  opp type of a single opp leg; then the **shape** i–iv (single / multi initial × single / multi continuation; 3D i–ii);
  unlisted combinations → no pattern. The table is ONE data constant (`SINGLE_OPP_RANK` / `MULTI_OPP_RANK`);
- **the tiebreak rules** (`apply_tiebreak_rules` — Step 2's own rule set, run ONLY over the ICs of the patterns tied
  on the best (rank, shape); ICR-S) in this order — each over the ICs the previous one left in play, the
  first to leave exactly one deciding; `ip_tiebreak` = the deciding one's name (the IP tiebreaker update, 2026-10-09,
  PLAN_H §6 TBU-1 … TBU-5, the user's text `../plans/plan_h_inputs/ip_tiebreaker_update_spec.md`; before it only the
  two below ran, as "TB1" / "TB2"):
  1. **the IC's candle type** (`ic_candle_type`) maru > normal > pinbar (`prefer_candle_type(include_pinbar=True)`) —
     it separates only multi-opp ranks (a single opp leg's type is in its rank);
  2. **the most recent before the imbalance** (`most_recent_before_imbalance`, the helper of the same name) — the most
     recent IC in play before the earliest unfilled sd imbalance after the OLDEST IC in play (an IC may be the gap's
     c1). Since PLAN_P it is ALWAYS their oldest: every IC passed condition 3, so the reference is the oldest one's own
     immediate gap, and an IC is the last candle of its opp leg, so no other tied IC sits before that gap (TBU-4; the
     user, 2026-10-13: "Yes" — the last rule picks the oldest candidate in play). The ranking rules have no such
     structure (two neighbours sharing a gap → the closer one, ICR-1).
  **Removed by PLAN_P (2026-10-13):** the first rule until then, **the immediate unfilled imbalance**
  (`immediate_unfilled_imbalance` — a filter: exactly one IC with an unfilled sd gap at IC + 1 … IC + 2 → it; two or
  more → only they go on; none → all go on). It became IC condition 3 (§3.1), so every tied IC passes it and it could
  never decide (the user's write-up: "they are no longer needed in Steps 2 & 3"). A save before S_P can hold
  `ip_tiebreak` `immediate_unfilled_imbalance`.
  On the reference window (PLAN_H §8.9 "The IP tiebreaker update"): three tied records; confluence sub 0 cycle 3
  moved 1447 → 1660 (the immediate unfilled imbalance); sub 1 cycle 1 (1875) and sub 7 cycle 2 (4083) kept their
  picks (no IC passed it). Since PLAN_P (vs S_O): sub 0 cycle 3 keeps 1660 with NO tie (1447 fails condition 3);
  sub 1 cycle 1 → 1890 (3D.ii); sub 7 cross 0→2 → 4131 (3B.iii, a Step 1A pick).

The pick's **`pattern_confirmed_idx`** (R1, a MOMENT) = when its continuation leg is known complete: the leg's last
candle when its run (before the E1 trim) reached the cap, else the next candle; it floors the POI's activation (§4).

**ONE selector, two consumers (PLAN_H D2):** the drawn POI (`derive_poi_zones`, on the FINAL fib record) and
MarketStructure's in-flight read (`poi_inner_for_fib`, on its embedded FibTracker's AS-OF representative fib at each
CTS_ESTABLISHED / CTS_UPDATED — PLAN_I S5, the drawn fib's own decision; its pick can differ from the final one while
the CTS still extends) call the same `select_ic`, each with the record's IC reference (§3.2) from the same builder
at its own moment (PLAN_H D2; PLAN_K §3.1). The
candle params and the leg cap come from the FRAME (`df.attrs["timeframe"]`, stamped by
`apply_candle_classification`; a missing stamp raises). The selection is computed once per pipeline run
(`select_ic_records`: the lifecycle gate, then the cycle's CTS_ESTABLISHED, then `select_ic(..., classify_all=True)`)
and handed to both
the drawn POIs and the debug CSV (§3.5). The read's fail-loud scope: LANDMINES "The POI Resolver Fails Loudly".

### 3.5 The debug export `*_ic_candidates.csv` (PLAN_H landing A, 2026-10-06; renamed + every checked candle since PLAN_L L1)

`poi_zones.derive_ic_candidates` turns the same records into debug rows — in every downstream run (the main, every
sub projection); the picked rows ARE the drawn POIs since PLAN_H landing B. Named `*_ip_candidates.csv` (and
`derive_ip_candidates` / `attrs["ip_candidates"]`) until PLAN_L L1 — a save before S_L1 holds those names.

- **Where:** `downstream["ic_candidates"]` → the main's `meta` / `df.attrs["ic_candidates"]`; a sub's
  `LowerTFResult.ic_candidates`, mirrored (shifted by `entity_df_mutation._IC_CANDIDATE_IDX_FIELDS`, the lens's
  attribution on `meta`) onto each lens df like the POIs.
- **Export:** `debug/export_zones.export_ic_candidates` → `*_ic_candidates.csv` (H1) and `*_M15_{lens}_ic_candidates.csv`
  (one copy per lens). One row per Step 1 CHECK of each record — every candle of the window **with the agreed
  direction** −sd (so a volume-only doji is listed and a split candle is not), whatever its other conditions (since
  PLAN_L L1, PLAN_H §6 ICR-4; before it only the candles passing condition 3 were listed), each with its pattern
  classified (ICR-11 — debug only: only an IC candidate is ever picked), + one **no-IC row** (`ic_idx` empty, the
  `IP: none` row until PLAN_L L2) per record without a pick, `reason` = `no_ic_candidate` (no candle passes every IC
  condition, never attributed to one condition; was `no_candidates` for "no §3.1 candidate" until PLAN_K landing B,
  PLAN_H §6 RX1-N) or `all_ic_candidates_pinbar` (no pattern and every IC candidate a pinbar — §3.6; since PLAN_L L2).
  Until PLAN_L L2 a third reason, `no_pattern` (IC candidates, none patterned), sat there — such a record now has a
  RANKED pick, and `no_pattern` is Step 2's `ip_outcome`. Columns: the attribution (`structure_path_id`, `sub_id`, `parent_sid`, `parent_cycle_id`),
  the record (`structure_id`, `cycle_id`, `bos_anchor_idx`, `cts_anchor_idx`, `sd`), the candidate (`ic_idx`,
  `candle_type`, `agreed_dir`, `touches_zone` (the touch of the record's `ic_zone_band` — 80–100 on EVERY row of a
  record where Step 1A ran, PLAN_P; the 61.8–80 touch is then not exported), `ic_ref_reason` — §3.2's outcome: empty
  = passes or no reference, `ic_past_reference`, `ic_not_before_break`; empty on the no-IC row; since PLAN_K landing A
  — a save before S_K_A lacks the column), `immediate_imbalance_first_c2_idx` (IC condition 3 since PLAN_P: the first
  gap candle of the candle's unfilled sd imbalance immediately after it, §3.1; empty = fails, and on the no-IC row;
  the first tiebreak rule's value from the IP tiebreaker update until PLAN_P — the CSV carried it on every row then
  too (`classify_all`), so its VALUES are unchanged by PLAN_P, only its role; a save before the IP tiebreaker update
  lacks the column), `has_unfilled_imbalance_after` (the RETIRED condition 3 since
  PLAN_P — information only, never a condition; condition 3 until then) and `is_ic_candidate` (all four conditions
  pass — its condition 3 is the immediate gap since PLAN_P) — both empty on the no-IC row; since PLAN_L L1 —,
  **`ic_zone_band`** (the zone of condition 4 — `61.8-80`, or `80-100` when Step 1A ran — on EVERY row of the record,
  the no-IC row included; since PLAN_P, a save before S_P lacks the column), **`ip_outcome`**
  (Step 2's outcome on EVERY row of the record: `pattern` / `no_pattern`, empty when Step 2 never ran — no IC
  candidate; since PLAN_L L2), the pattern (`ip_rank` e.g. "1A.iv", `reason` = the first failing rule
  when there is none, `pattern_anchor_idx`, `opp_leg_first_idx`, `pattern_end_idx`, `pattern_confirmed_idx`,
  `ip_leg_candle_counts` "a-b-c", `legs` "first..last" per leg, located from `ic_idx` + the counts), each leg stack's
  `*_type` / `*_ratio` (vs the 5 marus before the pattern's first candle) / `*_body_pips`, and `picked` + the tiebreak
  (`ip_tiebreak` = the deciding tiebreaker's name, §3.4; `ip_tb_runner_up_ic_idx` — the candle type: the most recent
  IC of the best losing type, the most recent before the imbalance: the most recent other IC in play;
  `ip_tied_count` — the size of the group the DECIDING rule broke: the (rank, shape) tie / the candle type's
  survivors; `ip_tb_imbalance_first_c2_idx` — the reference imbalance, empty for the candle type; until PLAN_P also
  the immediate unfilled imbalance's runner-up / gap) on the picked row, then Step 3's own **`ranking_rule`**, **`ranking_runner_up_ic_idx`**,
  **`ranking_in_play_count`**, **`ranking_imbalance_first_c2_idx`** (§3.6) on a RANKED pick's row (since PLAN_L L2).
  A failing candidate's `ip_leg_candle_counts` / stacks are its first failing variant's — a ranked pick's row too: it
  keeps its pattern DIAGNOSTICS (`reason` = the first failing pattern rule, the counts, `legs`, the stacks) like every
  failing row; `ip_outcome` `no_pattern` + `ranking_rule` say why it was picked (PLAN_H §6 ICR-15 (c)).

### 3.6 Step 3 — the ranking rules (`apply_ranking_rules`; PLAN_L landing 2, 2026-10-10)

> The user's write-up `../plans/plan_h_inputs/ic_identification_revamp_spec.md` (Step 3, "general rankings"); the
> decisions PLAN_H §6 ICR-S, ICR-1, ICR-2, ICR-3, ICR-8 … ICR-10, ICR-12, ICR-14, ICR-15; built by PLAN_L §4.

Run by `select_ic` ONLY when Step 2 found no pattern (`ip_outcome` `no_pattern`), over the UNFILTERED IC candidates
(every Step 1 — or Step 1A — candidate, not the patterns' ICs), their CHECKS (the candle type from the frame —
nothing re-derived). Its own rule set (ICR-S): each rule over the candidates the previous one left in play, the first
to leave exactly one deciding (`ranking_rule`):

1. **exclude pinbars** (`exclude_pinbar`; `candle_type` "pinbar") — none left → **no IC** (`all_ic_candidates_pinbar`;
   Step 4 would run — deferred; Step 1A does NOT run here); one left of two or more → it (rules 2–3 never run: no
   runner-up, ICR-14);
2. **the candle type** maru > normal (`ic_candle_type`; `prefer_candle_type(include_pinbar=False)` — rule 1 already
   removed the pinbars);
3. **the most recent before the imbalance** (`most_recent_before_imbalance`, the SAME helper as tiebreak rule 2): the
   reference = the earliest unfilled sd imbalance after the OLDEST candidate STILL IN PLAY after rules 1–2 (ICR-8 —
   not the oldest Step 1 candidate); the pick = the most recent candidate in play before the reference's first gap
   candle; a candidate may be the gap's c1 — of two neighbours sharing one gap, the one closer to it (ICR-1: "The
   IC candidate closer to the imbalance is better"). It always decides when reached. Since PLAN_P every candidate
   passed condition 3, so the reference IS the oldest in play's own immediate gap and the pick is that oldest
   candidate — or its in-play neighbour sharing the gap (the user, 2026-10-13: "Yes").

**Removed by PLAN_P (2026-10-13):** rule 2 until then, **the immediate unfilled imbalance** (`immediate_unfilled_
imbalance` — a filter, the same rule as the tiebreak rules' first, `filter_immediate_unfilled_imbalance`): it became
IC condition 3 (§3.1), so every candidate passes it. A save before S_P can hold `ranking_rule`
`immediate_unfilled_imbalance`.

Step 1 (or Step 1A) found exactly ONE IC candidate and it is not a pinbar → it is the IC with `ranking_rule` EMPTY
(nothing was ranked, ICR-10; a single PINBAR candidate → no IC). The two rule sets differ as wholes: on an all-pinbar list the
ranking rules give no IC where the tiebreak rules (pinbars ranked last, not excluded) would take the oldest. (Until
PLAN_P the H1 sid 0 cycle 0 example showed the order through the immediate rule — 177 pinbar, 180 maru, 183 pinbar,
only 183 with an immediate gap → rule 1 excluded 183 first → 180; since PLAN_P 180 fails condition 3 and the record
has no IC: `all_ic_candidates_pinbar`.)

**The fields** (`ICSelection.ranking` → the POI meta and the CSV's picked row, §3.5 / §4): `ranking_rule`;
`ranking_runner_up_ic_idx` — rule 2: the most recent normal; rule 3: the most recent other candidate in play; rule 1:
EMPTY; `ranking_in_play_count` — the candidates entering the deciding rule (rule 1: the Step 1 count);
`ranking_imbalance_first_c2_idx` — rule 3: the reference gap; else empty. All four empty with no ranking rule. Every
value an `int` / `str` (the sub mirror shifts only Python ints — `entity_df_mutation._ZONE_META_IDX_KEYS` /
`_IC_CANDIDATE_IDX_FIELDS` hold the two `*_idx`).

**The consumers** — ONE selector (PLAN_H D2, ICR-12): the drawn POI (its activation floor has no pattern term — §4,
ICR-2) and MS's in-flight read (`poi_inner_for_fib` returns a ranked pick's inner like a pattern pick's). The
ranking rules' imbalance reads have no knowability cut, like the tiebreak rules' (MARKET_STRUCTURE_SPEC "Snapshot vs
per-candle"). **On the reference window** (PLAN_L §4.3, the S_L1 → L2 compare): 7 records gain an IC — H1 180
(`exclude_pinbar`), 694 (`most_recent_before_imbalance`); confluence 670, 1804 (`most_recent_before_imbalance`), 929
(`exclude_pinbar`), 2575, 3809 (`immediate_unfilled_imbalance`); no structure event, trigger, sub or WVMI moves.
**Since PLAN_P** (vs S_O; `../plans/PLAN_P_ic_condition3_step1a.md` §6): H1 180 and 694 → no IC (all pinbar), 832 →
859 (`exclude_pinbar`); confluence 670 → 650 and 1804 → none (Step 1A ran in both: 650 `exclude_pinbar`), 929 → 923
(a Step 1A PATTERN pick, 3C.iv), 2575 / 3809 kept (`exclude_pinbar` — the rule decides now), 1117 → 1216
(`ic_candle_type`), 1772 → 1778 (the only candidate), 1934 (a Step 1A pick, `exclude_pinbar`).

---

## 4. POI Zone Construction

### POIs are always sd-direction (invariant)

A POI zone is always in the **structure direction** by construction:
- Fib levels span `BOS → CTS` within a cycle, so the 61.8%–80% retrace zone
  sits entirely within that span
- IC candidates must be opposite-direction candles (agreed direction `== -sd`)
  located inside that span, with an sd-direction unfilled imbalance after them
- The resulting POI zone's `side` mirrors the structure direction:
  `"buy"` if `sd == +1`, `"sell"` if `sd == -1`

**Consequence:** within ONE structure there are no opp_sd POIs — a structure's
own POIs are always on its sd side, and the parent's own zone-proximity ladder
(`zones/zone_proximity.py`) still reads its POIs as sd rungs only.

**But the parent's opp_sd side now reads a counter sub's POI** (PLAN_R step 2,
2026-10-20 — the proximity overlay, `multitf/proximity_overlay.py`): a
counter-direction sub's POI is a POI in the SUB's direction, i.e. opposite to the
parent's, so it joins the parent's opp_sd proximity reference ladder (with the
sub's BOS zones); a same-direction sub's zones beyond the parent POI join the sd
side. The side is assigned explicitly by the sub's direction relative to the
parent, never inferred from "POI ⟹ sd". The invariant above is per structure and
still holds; the register item "Opposite-direction POIs" is closed by this.

### Zone Boundaries
- **Top:** IC candle high
- **Bottom:** IC candle low
- **At most one zone per fib record** — the IC: the institutional-pattern pick (§3.4) or, with no pattern, the
  ranked one (§3.6); none without an IC

### Zone Data Fields
- `side`: "buy" if sd=+1, "sell" if sd=-1
- `structure_id`, `struct_direction`, `cycle_id`
- `ic_idx`: Index of the IC candle (rectangle start)
- `confirmed_idx`: **LAST** activate idx (lossy scalar — `poi_zones.py` overwrites it on every activation and never clears it on deactivation). A POI can flap active/inactive within a cycle, so this scalar does NOT bound the active interval. To ask "is the POI active as of candle X?" use `zones/poi_lifecycle.py::poi_active_as_of` (walks `activation_history`), never `confirmed_idx`. See GOTCHAS "POI `confirmed_idx` Is a Lossy Scalar". (The zone-proximity scan asks neither since PLAN_Q, 2026-10-14: it reads a POI like a KL zone, at every candle of its scan — §4 "Activation floor".)
- `top`, `bottom`: IC high/low
- `ip_outcome` (PLAN_L L2, 2026-10-10): `"pattern"` (a pattern's IC, §3.4) / `"no_pattern"` (a ranked IC, §3.6) —
  the `*_poi_zones.csv` `ip_rank` column is empty on a ranked IC
- `ic_zone_band` (PLAN_P, 2026-10-13): the zone the IC was identified in — `"61.8-80"` (Step 1) or `"80-100"` (Step
  1A, the fallback, §3.1); the activation's touch re-check uses the same band (§4 "Lifecycle"). A save before S_P
  lacks the key
- **The institutional pattern** (PLAN_H D6 / landing B, 2026-10-06 — a rule-3 migration: `versions` /
  `current_versions` were removed in the same commit, no alias; the `*_poi_zones.csv` column `versions` → `ip_rank`)
  — **ALL None on a ranked IC** (`classify_ics` leaves a no-pattern candidate its first failing variant's counts;
  `derive_poi_zones` nulls them explicitly, PLAN_L A1):
  - `ip_rank`: the pick's rank + shape, e.g. `"1A.iii"` (§3.4)
  - `ip_leg_candle_counts`: `"a-b-c"` = the initial / opp / continuation legs' candle counts
  - `pattern_anchor_idx` (the initial leg's first candle), `opp_leg_first_idx`, `pattern_end_idx` (the continuation
    leg's last candle): the legs are contiguous around `ic_idx` (the opp leg's last candle)
  - `pattern_confirmed_idx`: a MOMENT — the continuation leg known complete (R1); a term of the activation floor
  - `ip_tiebreak` (`None` / `"ic_candle_type"` / `"most_recent_before_imbalance"` — §3.4; `"TB1"` / `"TB2"` = the
    candle type / the most recent before the imbalance until the IP tiebreaker update, 2026-10-09;
    `"immediate_unfilled_imbalance"` from it until PLAN_P), `ip_tb_runner_up_ic_idx`, `ip_tied_count` (a count),
    `ip_tb_imbalance_first_c2_idx` (the most recent before the imbalance's reference): set only when a tiebreaker
    decided
  - every index key is in the mirror's `entity_df_mutation._ZONE_META_IDX_KEYS` (entity-absolute on sub POIs)
- **The ranking** (PLAN_L L2, §3.6): `ranking_rule` (`None` / `"exclude_pinbar"` / `"ic_candle_type"` /
  `"most_recent_before_imbalance"`; `"immediate_unfilled_imbalance"` until PLAN_P), `ranking_runner_up_ic_idx`, `ranking_in_play_count` (a
  count), `ranking_imbalance_first_c2_idx`: set only when a ranking rule decided — each per §3.6 "The fields" (rule 1
  `exclude_pinbar` sets no runner-up / gap; a ranked IC with ONE IC candidate carries all four None); the two `*_idx` keys in `_ZONE_META_IDX_KEYS`
- `status`: "active" | "inactive" | "ended" (+ "disappeared" — a reserved terminal-invalidation status; see Zone States). Charting note (2026-09-20): a POI whose CYCLE collapsed (its BOS KL zone is `inactive` with clamped `confirmed_idx >= end_idx`) is not drawn on any chart (`charting/_zone_render.collapsed_cycles` / `is_poi_of_collapsed_cycle`); an `"inactive"` POI whose cycle is live (never met its activation conditions) is still drawn as an outline. POI fills are side-tinted since 2026-09-21 (buy gold-lime, sell amber — CHARTING_SPEC §7).
- `end_idx` / `end_reason`: terminal axis inherited from the owning cycle (`compute_cycle_lifecycle`): `"reversal"` | `"next_cycle"` (the next cycle's clamped lifecycle-start = the CTS-established **moment** `meta["confirmed_at"]`, Plan C 2026-09-20) | for subs the sub's `cap_reason` — `"reversal"` | `"same_dir_replacement"` | `"parent_end"` — when the unique sub's `end_idx` capped the cycle (`"lifecycle_end"` is no longer emitted) | `None`
- `activation_history`: per-candle activate/deactivate flips (condition axis) — `[{"idx", "active", "reason"}]`
  (entries carried `versions` too until landing B)
- `cts_established_idx`: the owning cycle's CTS-established **moment** (`CTS_ESTABLISHED.meta["confirmed_at"]`)
  — the activation floor's cycle term (see "Activation floor" below). **Meaning changed by Plan D (2026-09-23):**
  saves before it hold the CTS anchor (then `CTS_ESTABLISHED.idx`) under this key. **A cycle with no
  `CTS_ESTABLISHED` builds NO POI (user decision 2026-09-29, zones-audit "fallback POI" option N)** — always a moment
  since then. Such a cycle's fib is the pre-established cross FibTracker pre-creates for a next cycle that never
  establishes (subordinate-only; FIB_LIFECYCLE_SPEC §6): a reversal or the sub's lifecycle cap ends it (§15.4
  candidates 1 / 3) and an ended, unlocked fib builds no POIs, so only a still-LIVE one (an open-ended sub, at the data
  edge) reached this case — and its POIs could never activate (the floor's cycle term is the establishment moment; the
  sweep's progression comes only from the cycle's own CTS events). Its POIs now appear once the cycle establishes,
  exactly as for a pre-established cross that does establish (their floor is that moment either way). A LOCKED fib
  without `CTS_ESTABLISHED` breaks the event contract (a fib locks at its cycle's `CTS_CONFIRMED`) and fails loudly.
  Before: the fallback `fib_state.cts_idx` — the fib's CTS anchor, NOT a moment (the GLOSSARY "Naming Standard"
  exception of PLAN_D §7.3) — the running extreme near the data edge, so such a POI was drawn inactive and never
  activated, then vanished when the sub was capped or reversed (a repaint). Measured before: reference window 0
  (byte-identical); suite 2 (the open-ended fixture of `tests/test_fib_never_established_cap.py` + a constructed
  locked case). Rejected: keep the inert POI with the key set to None (visible at the live edge, but it vanishes when
  the sub closes; the key's meaning widened); activate pre-established POIs early from the fib's pre-creation (a design
  change — it would move 10 window fibs that were pre-established and later established). No live case on the
  reference window since Plan F (2026-09-24): the only one was the IC 3654
  twin POI on the cross fib counter sub 5 PRE-CREATED at 3806 for a cycle 1 it never establishes (value 3806, never
  activated; sub 5 was capped, so the 2026-09-28 cap would now drop it too); Plan F no longer creates that fib (§2
  "Anchor by phase"). The sub 5 cycle-0 IC 3654 POI is a different row and unchanged.
  Rebased to entity-absolute on sub POIs by the mirror (`entity_df_mutation._ZONE_META_IDX_KEYS`). Readers: none
  that decide anything (the env-gated `POI_LIFECYCLE_DEBUG` print only).
- `bos_anchor_idx` / `cts_anchor_idx` (`bos_idx` / `cts_idx` until Post-E·4, 2026-09-29d — saves before it carry the old
  keys): the owning fib's `bos_idx` / `cts_idx` (its BOS / CTS anchors) copied when the POI is built — the same values
  as the fib CSV's own `bos_idx` / `cts_idx` columns. Rebased to entity-absolute on sub POIs by the mirror since Post-E·2
  (2026-09-26; slice-local in the M15 POI CSVs before — e.g. confluence sub 0 cycle 0 IC 678: 50 / 380, now 454 / 784).
  Readers: none that decide anything (the same debug print).
- `end_time`: When zone ends (None = extends to chart end)

### Zone States

Per the lifecycle convention (ARCHITECTURE.md "Lifecycle state convention"),
`status` is **derived** from the two orthogonal axes stored on the zone:
`end_idx`/`end_reason` (terminal, irreversible) + `activation_history`
(condition, reversible). POI currently produces `{active, inactive, ended}`
(`poi_zones.py` status derivation); `disappeared` is a **reserved** terminal-
invalidation status the chart still filters but POI does **not** currently
produce (the planned FibState lifecycle work will produce it for
`scenario1_revert` — see `zones/FIB_LIFECYCLE_SPEC.md`).

| Status | Meaning | Charting |
|--------|---------|----------|
| `active` | Condition holds at end-of-data (the activation conditions: the IC within the fib, an unfilled sd imbalance after the IC, the touch of the IC's band — §4 "Lifecycle") | Rendered fully (bright tier) |
| `inactive` | Alive but condition currently off (reversible — e.g. imbalance filled; can reactivate) | Rendered faded |
| `ended` | Terminal — cycle/structure ended (`reversal` \| `next_cycle` (next cycle's clamped start on the CTS-established moment) \| the sub's window end: `reversal` / `same_dir_replacement` / `parent_end`). Irreversible | Rendered faded |
| `disappeared` | Terminal **+ suppressed** (invalidation). Reserved; no producer since PLAN_I S4 retired `scenario1_revert` (a FibState hook — FIB_LIFECYCLE_SPEC §10), not by POI | NOT rendered (kept in list for history) |

### Zone End Time (Priority Order)
1. **Reversal:** `end_time = reversal_confirmed_idx` (all zones end immediately)
2. **New CTS:** Cycle N zones end at cycle N+1's clamped lifecycle-start = `max(CTS_{N+1} ESTABLISHED.meta["confirmed_at"], struct_start[, lifecycle_floor])` — the CTS-established **moment**, not the CTS anchor (`meta["cts_anchor_idx"]`, `CTS_ESTABLISHED.idx` until Plan E E4a; Plan C, 2026-09-20)
3. **Sub window end (subs only):** the unique sub's `end_idx` (`lifecycle_cap`), tagged with the sub's `end_reason`
4. **No event:** Zone remains active (`end_time = None`)
`end_idx` is `min(...)` of the applicable candidates (earliest wins; reversal wins an equal-idx tie) — the cycle table from `zones/structure_lifecycle.compute_cycle_lifecycle`, shared with KL and fib.

### Lifecycle
- Zone activates the first time its conditions hold; **every** activate/deactivate flip is recorded in `activation_history` (`[{"idx", "active", "reason"}]`)
- The conditions (`_compute_poi_activation_history`): cond1 the fib's as-of CTS anchor has reached the IC; cond3 the
  POI's OWN imbalance condition below; **cond5 the touch — the IC touches the zone of the AS-OF fib in the band the
  IC was identified in** (61.8–80, or 80–100 for a Step 1A IC — `ic_zone_band` → `ic_zone_config`, PLAN_P; the user,
  2026-10-13: "POI activation check should always reference the zone parameters the IC came from") (`bos_price` +
  the latest known CTS price; the selection's own `fib_zone_bounds` + `ic_touches_zone`, inclusive edges; PLAN_H
  D1a), re-checked at each CTS event's moment. A consequence of the band: while the CTS has not yet extended, a
  Step 1A IC can sit in the as-of 61.8–80 zone and NOT touch the as-of 80–100 one — its POI is then inactive
  (`ic_not_touching_fib_zone`); 0 such deactivations on the reference window (the 4 Step 1A POIs, PLAN_P §2).
  Deactivation reasons: `ic_out_of_fib_bounds` (cond1),
  `imbalance_filled` (cond3), `ic_not_touching_fib_zone` (cond5, R3 — `variant_below_v30` until landing B, when cond5
  was the V30 overlap)
- Zone can deactivate when an activation condition stops holding on subsequent candles (cond1 / cond3 / cond5 above
  — not a re-run of the IC selection), then re-activate later — a cycle can flap multiple times
- The imbalance condition (an sd imbalance in `(ic_idx, t]` that has formed and is not committed-filled — ANY gap
  after the IC, not only its immediate one) is the POI's OWN condition, DECOUPLED from IC identification since PLAN_P
  (IC condition 3 is the immediate gap, §3.1; this stays as it was — the user, 2026-10-13: "The concept of keeping a
  POI zone active centers more around that we currently still have unfilled imbalances in our current cycle.
  Therefore, keeping the old rule suffices in checking that. IC identification on the other hand is trying to find
  the best candidate to build the zone around, making the immediate imbalance part crucial." — gaps between the BOS
  and the IC do not count). It is swept by
  events (`_compute_poi_activation_history`): each instance **enters** the unfilled set at
  `max(inst.formed_at, first_active)` — its first c3 — and **leaves** at its cached stroke-2 candle
  `confirmed_fill_idx` (`_compute_fill_idx_cache`). Identity: `has_unfilled_imbalance(df, ic_idx+1, t,
  check_to_idx=t, direction=sd, evaluated_at=t)`. Plan F (2026-09-24) moved the enter from the c2 (`inst.start_idx`,
  one candle before the gap existed) to `formed_at`: on the reference window H1 sid 1 cyc 2 IC 865 and IC 860 each
  re-activate at 954 and 998 (were 953 and 997), and M15 sub 7 IC 4048 re-activates at 4119 (was 4118)
- `confirmed_idx` collapses to the **LAST** activate idx (NOT the first). Per-candle activation lives in `activation_history`; query it via `zones/poi_lifecycle.py` (`poi_active_as_of`, `poi_confirmed_idx_as_of`), never the scalar. Consumers that treat `[confirmed_idx, end_idx]` as one active span are blind to earlier active stretches (this caused the sid1-cyc2 proximity-trigger loss — see GOTCHAS; the proximity scan reads no activation since PLAN_Q)

### Activation floor — cycle lifecycle-start clamp (REVISED 2026-05-26; floor terms REVISED by Plan C 2026-09-20; cycle term moved to the MOMENT by Plan D 2026-09-23)

POI activation is gated to start no earlier than the owning cycle's
lifecycle-start. The activation scan begins at
(`poi_zones._compute_poi_activation_history`):

```
first_active = max(cts_established_idx, ic_idx, lifecycle_floor_idx[, pattern_confirmed_idx])
cts_established_idx = CTS_ESTABLISHED(sid, cycle).meta["confirmed_at"]   # the cycle's established MOMENT
pattern_confirmed_idx = the IC's continuation leg known complete            # PLAN_H R1 (landing B), a MOMENT —
                                                                            # a pattern's IC only
```

`pattern_confirmed_idx` (2026-10-06): a pattern is knowable only once its continuation leg has ended — its last
candle when the leg hit the cap, else the next candle; past the frame's last candle the POI never activates. On the
reference window it binds nowhere (PLAN_H §8.4a). **A ranked IC (§3.6) has no pattern term** (PLAN_H §6 ICR-2 — "For
Step 3, we simply apply the new Rule 1-4 to the Step 1 candidates"): `_compute_poi_activation_history(...,
pattern_confirmed_idx=None)` → `max(cts_established_idx, ic_idx, lifecycle_floor_idx)` under the unchanged live
conditions. The parameter is keyword-required with no default (an IP caller cannot drop the term silently, PLAN_L A7),
and `derive_poi_zones` asserts `pattern_confirmed_idx is None` ⇔ `ip_outcome == "no_pattern"`.

where `lifecycle_floor_idx = struct_start_by_sid[sid]` =
`compute_struct_start_by_sid(events, reversal_idx_by_sid, lifecycle_floor)` —
the structure lifecycle-start (the first CTS_ESTABLISHED moment since Plan E E3f / the reversal handoff), raised to
`lifecycle_floor`: `None` for main; for a sub the **unique sub's real-time
`start_idx`** (slice-local, passed by `render_sub_projection` →
`project_to_window`), which already contains the parent-cycle floor
(`TriggerRecord.parent_floor_idx = LOH(max(struct_start[S], cts_moment[(S,C)]))`
on the CTS-established **moment**) because the record's `start_idx =
max(probe_finalize_idx, trigger_idx, parent_floor_idx)`. (Historical: plan B1,
2026-05-27, wrote the sub floor as separate "parent sid / parent_cycle_id
lifecycle-start" terms; Plan C folds them into the one `lifecycle_floor`.)
Before B1 the floor was just `max(cts_established_idx, ic_idx)`; the added
term clamps **post-reversal cycle-0 POIs** (and sub analogues) whose
`CTS_ESTABLISHED` precedes the structure's lifecycle-start (e.g. probe placed
the anchor historically).

**One helper, two callers (PLAN_Q, 2026-10-14):** the floor is `poi_zones.poi_activation_floor_idx(cts_established_idx,
ic_idx, *, pattern_confirmed_idx, lifecycle_floor_idx)` (both optional terms keyword-required, `None` skipped). The
activation sweep passes every term; `zone_proximity.poi_unclamped_activation_floor_idx` passes
`lifecycle_floor_idx=None` — the floor in the POI's own structure's timeline. The zone-proximity scan counts a POI at
every candle of its cycle's scan, exactly like the BOS KL zone, never reading `activation_history` (the user: "I would
like POI to be treated exactly like a KL zone"), and asserts that unclamped floor is at or before the scan start
(CTS_CONFIRMED) — the IC is at or before the fib's final CTS anchor, the pattern confirms by the anchor + 1, and the anchor
precedes the confirmation — so neither the lifecycle clamp nor an off-stretch hides a POI any more: a collapsed
cycle's POI (an empty history) now feeds the sd trigger like its BOS zone (`plans/PLAN_Q_proximity_poi_like_kl.md`).
The trigger hovers show the floor as a POI marker's `zone_confirmed_idx`.

`max(cts_established_idx, lifecycle_floor_idx)` equals the cycle's clamped lifecycle-start
(`compute_cycle_lifecycle`); a cycle without a `CTS_ESTABLISHED` builds no POI (2026-09-29 — see the field list;
before it the cycle term fell back to `fib_state.cts_idx`). The lookup indexes `meta["confirmed_at"]` directly — no fallback to
`ev.idx`; `compute_cycle_lifecycle` asserts the key first for every event carrying `structure_id`/`cycle_id`.

> **Resolved (Plan D, 2026-09-23).** Until Plan D the cycle term was `CTS_ESTABLISHED.idx` — then the CTS
> **anchor** (the breakout pattern's extreme candle, retro-stamped; the idx is the moment since Plan E E4a), not the moment the cycle became knowable —
> while the canonical cycle lifecycle-start and every cycle END were already on the moment
> (`compute_cycle_lifecycle`, Plan C). On a cycle whose anchor precedes its moment a POI could go live
> `confirmed_at − idx` candles (≤ 5) before its cycle existed, overlap the previous cycle's POI on those candles,
> and activate on a cycle collapsed at its moment. On the reference window 3 CSV rows = 2 unique M15 cycles lag by 1
> (sub 0 cyc 2 1223/1224; sub 3 cyc 1 2828/2829, masked by the sub's floor). Measured at landing (PLAN_D §9; exactly the §4 prediction,
> as predicted): M15 confluence sub 0 cyc 2 IC 678 activation 1223 → 1224, plus 3 `cts_established_idx` meta cells
> re-valued to the moment; the other 22 CSVs byte-identical; chart counts unchanged. Pinned by
> `tests/test_poi_activation_moment.py`.

Any activate/deactivate flips before `first_active` are discarded; if the
POI would have been active earlier, its first active snaps to
`first_active`. If `first_active >= end_idx` the POI never activates
(`activation_history = []`, `status` never `"active"`), kept in the list as
history.

---

## 5. Charting

### Toggle
```python
cfg = {
    "zones": {"POI": True},
    "fib": {"lines": True},
    "imbalance": {"highlight": True},  # Candle color highlighting
}
```

### Visual Elements
1. **POI Zone Rectangle** — Yellow fill
   - Starts at `ic_idx` (the IC candle)
   - Ends at `end_time` (or chart end if None)
   - Horizontal lines at top/bottom bounds
2. **Confirm Line(s)** — one darker vertical line per activate event in `activation_history` (legacy fallback: a single line at `confirmed_idx` for zones predating the history)
3. **Fibonacci Lines** — Dotted lines at 0%/100% anchors, rectangle at 61.8-80%
4. **Imbalance Candle Highlighting** — Entire candle (body + wicks) colored distinctly. Marks each flagged c2
   (`is_imbalance`) — the pattern's location, not the moment its gap exists (CHARTING_SPEC §6):
   - Bullish imbalance: Lime Green `rgba(50, 205, 50, 0.8)`
   - Bearish imbalance: Amber Yellow `rgba(235, 190, 0, 0.8)`

### Zone Rendering Rules
- `active` zones: Full opacity
- `inactive` / `ended` zones: Faded opacity (3-tier by status + sid-recency)
- `disappeared` zones: NOT rendered (terminal-invalidation; filtered out — reserved, not produced by POI today)

### Hover Data
- Side, structure_id, cycle_id
- IC_idx, confirmed_idx
- The institutional pattern (PLAN_H HV, landing B — `versions=` until then): `IP <rank> · legs a-b-c (initial |
  opp | continuation)`, each leg `first–last` (one candle: its index), + `Tiebreak over <IC>: IC candle type` /
  `…: most recent before imbalance <c2>` when a tiebreaker decided (§3.4; `…: immediate unfilled imbalance <c2>`
  until PLAN_P); a RANKED IC (§3.6, PLAN_L L2): `No pattern · ranking rule <rule>` + ` over <runner-up>` + ` (gap
  <c2>)` when the deciding rule has them (e.g. confluence sub 2 cycle 1's IC 2575: `No pattern · ranking rule
  exclude_pinbar` since PLAN_P — `… immediate_unfilled_imbalance over 2574 (gap 2577)` until then), or `No pattern ·
  only IC candidate` — NO `IC <idx>` there (the hover's own `ic_idx=` line names the IC; PLAN_H §6 ICR-15); + a
  line `IC zone 80-100 (fallback)` on a Step 1A IC (PLAN_P; nothing on the 61.8–80 band) — ONE builder for the three
  POI hover sites (`charting/_zone_render.poi_ic_hover_text`)
- Top/bottom bounds
- The fib hover (the H1 chart AND the M15 lens charts since PLAN_H §8.9 F1, 2026-10-06): its status cell =
  status · `life=[start, end] reason` · the IC outcome (`IP <rank> · IC <idx>` — a pattern's IC; `No pattern · IC
  <idx> · ranking rule <rule>` / `No pattern · IC <idx> · only IC candidate` — a ranked IC, the IC index KEPT: the fib
  box's only pointer to its IC; `IC: none (<reason>)` on a record without an IC — `no_ic_candidate` /
  `all_ic_candidates_pinbar` (§3.5), no POI, PLAN_H C1 —; or `IC: not evaluated (no POI record)`; `IP: none` / `IP: not
  evaluated` until PLAN_L L2; + ` · zone 80-100 (fallback)` when Step 1A ran — on a pick, or on `IC: none (…)` = no
  IC in either zone, PLAN_P; `charting/_zone_render.fib_hover_status_text`) · since PLAN_I S2 the structure's cross-fib limit line
  (`fib_hover_limit_text`). Canonical: `charting/CHARTING_SPEC.md` §8

### Style Keys (style_registry.py)
```python
"zone.poi.buy"         # Buy-side POI zone fill (Yellow)
"zone.poi.sell"        # Sell-side POI zone fill (Yellow)
"zone.poi.hover_line"  # Invisible hover hitbox
"fib.line"             # Fibonacci level lines
"imbalance.bullish"    # Bullish imbalance candle rgba
"imbalance.bearish"    # Bearish imbalance candle rgba
```

---

## Data Flow

```
Fib active for cycle
    ↓
Step 1 — check every agreed −sd candle of the window against the IC conditions (§3.1: the IC reference §3.2,
the immediate unfilled imbalance (IC + 1 … IC + 2), the touch of the 61.8-80% zone) → the IC candidates; none →
    ↓
Step 1A — the same checks, the touch of the 80-100% zone (PLAN_P) → the IC candidates; none → no POI
    ↓
Step 2 — the best institutional pattern among them + the tiebreak rules (§3.4); none →
    ↓
Step 3 — the ranking rules over the same IC candidates (§3.6); every one a pinbar → no POI (Step 4 deferred)
    ↓
Create the POI zone (at most one per fib record, bounds = IC high/low)
    ↓
Track lifecycle (active ⇄ inactive condition flips; → ended at terminal. disappeared = reserved terminal-invalidation, not produced by POI today)
    ↓
Charting renders zones + Fib lines + imbalance highlighting
```

---

## Files

| File | Purpose |
|------|---------|
| `patterns/imbalance.py` | Imbalance (FVG) pattern detection + fill checking |
| `features/fibonacci.py` | Fib level calculation |
| `zones/fib_tracker.py` | FibTracker lifecycle (activation/update/lock) |
| `zones/poi_zones.py` | POI zone derivation (one per fib record); the IC selection in steps (`select_ic` — `check_ic_conditions` / `find_ic_candidates`, `select_institutional_pattern` / `apply_tiebreak_rules`, `apply_ranking_rules` and the shared rule helpers —, `select_ic_records`, `derive_ic_candidates`, §3–3.6); MS's in-flight read `poi_inner_for_fib` (PLAN_I S5) |
| `patterns/institutional_pattern.py` | The institutional pattern of one IC candidate (legs, stacks, ranks — PLAN_H §1) |
| `zones/cross_cycle_fib.py` | `ICReference` + `CrossFibLimit.ref_anchor_price` / the reversal break (§3.2, PLAN_K) |
| `patterns/cross_limit_watch.py` | The cross-limit breakout watch (`find_cross_limit_break`, §3.2 / CROSS_CYCLE_FIB_SPEC §3) |
| `charting/export_plotly.py` | Rendering (Fib lines + zones) |
| `charting/style_registry.py` | Visual styles |

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Imbalance pattern (columns) | Done |
| Imbalance fill checking | Done |
| Imbalance candle highlighting | Done |
| Fibonacci dataclass | Done |
| FibTracker (activation/update/lock) | Done |
| Cross-cycle Fib exception | Done |
| Fib charting (0%/100% lines + 61.8-80% rect) | Done |
| IC candidate identification | Done |
| IC variant selection (V30/V60/V90) | Done; UNWIRED since PLAN_H landing B (kept, X1) |
| The institutional-pattern IC (one POI per fib record) | Done (PLAN_H landing B, 2026-10-06) |
| The IC reference condition (§3.2) | Done — computed + exported (PLAN_K landing A), applied (landing B), 2026-10-09 |
| The IC identification in steps (§3) | Done — Steps 1–2 restructured (PLAN_L landing 1, 2026-10-09); Step 3 the ranking rules (§3.6, landing 2, 2026-10-10); IC condition 3 = the immediate unfilled imbalance + Step 1A, the 80–100 fallback zone (PLAN_P, 2026-10-13); Step 4 deferred |
| POI zone creation | Done |
| POI zone lifecycle management | Done |
| POI zone charting (yellow rectangles) | Done |

**Status:** Complete. Ready for Week 7 Part 2 (Volume Patterns & Indicators).

---

## Appendix A — the per-scenario IC conditions as designed (Week 7; never implemented)

History. Designed with the POI build (`0033c53`, Week 7) and never applied by the code (found 2026-09-30:
`passes_price_constraint` had no caller — REUSED by PLAN_K landing A as §3.2's price side —, `find_ic_candidates`'
`scenario_context` was never read — retired by PLAN_K landing A); kept unapplied by the user's decision 2026-10-02 (PLAN_H D4); re-specified by the user 2026-10-09 as §3.2's
ONE bundled condition for today's one-logic fibs (the Scenario 1 / 2 / 3 fib logic itself retired with PLAN_I S4,
CROSS_CYCLE_FIB_SPEC Appendix A). The tables as they stood:

**Scenario 1: cycle 0 normal fib, cycle 1+ normal fib**

| Cycle | Idx Constraint | Price Constraint |
|-------|----------------|------------------|
| 0 | `idx < reversal_confirmed_idx` | Entire candle below (sd=+1) or above (sd=-1) prev structure's last BOS price |
| 1+ | `idx < CTS_N_established_idx` | Entire candle below (sd=+1) or above (sd=-1) CTS_N-1 price |

**Scenario 2: no cycle 0 fib, cycle 1 cross-cycle fib, cycle 2+ normal fib**

| Cycle | Sub-condition | Idx Constraint | Price Constraint |
|-------|---------------|----------------|------------------|
| 1 (cross) | CTS_0 < reversal_idx | `idx < CTS_1_established_idx` | Entire candle below/above CTS_0 price |
| 1 (cross) | CTS_0 >= reversal_idx | `idx < reversal_confirmed_idx` | Entire candle below/above prev structure's last BOS price |
| 2+ | — | `idx < CTS_N_established_idx` | Entire candle below/above CTS_N-1 price |

**Scenario 3: no cycle 0 fib, cycle 1+ normal fib**

| Cycle | Idx Constraint | Price Constraint |
|-------|----------------|------------------|
| 1+ | `idx < CTS_N_established_idx` | Entire candle below (sd=+1) or above (sd=-1) CTS_N-1 price |

**Price Constraint Definition:**
- sd=+1 (bullish): Entire candle (HIGH) must be BELOW reference price
- sd=-1 (bearish): Entire candle (LOW) must be ABOVE reference price

