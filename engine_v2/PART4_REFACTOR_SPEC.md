# Part 4 Refactor Spec — Multi-TF Structure Hierarchy

> **Status:** Specs complete; entering build phase. All sections (1–16)
> locked across sessions 2026-04-29 / 04-30 / 05-01 / 05-04.
>
> **⚠ REVISED 2026-05-25 — subordinate lifecycle model.** §2, §5, §6, §7
> were rewritten to replace the entity-wide-`entity_sid` + cascade-overwrite
> model with **per-parent-cycle sid identity `(parent_sid, parent_cycle_id,
> sid)` + a merge-and-bound sequential build + a unified KL/POI lifecycle
> (pass-through ends)**. The §13.5.c/d migration substeps describe the OLD
> (cascade) mechanism and are superseded — see the banners there. Full
> rationale + phased implementation plan:
> `memory/project_sub_structure_lifecycle_redesign.md`.
>
> **Working document.** Will be split / renamed / merged into canonical spec
> files (`MARKET_STRUCTURE_SPEC.md`, new `MULTI_TF_SPEC.md`, etc.) once the
> refactor lands and behavior is stable.
>
> Companion file: `PRE_REFACTOR_INVARIANTS.md` — what currently-correct
> behavior must survive (or is explicitly being changed).
>
> **Vision:** A multi-TF system where every structure entity (main + every
> subordinate TF / role / parent combination) runs in its own isolated df,
> producing independent signals. Entries are derived by **weaving**
> confluence and counter signals across timeframes — capturing intricate
> price movements that single-TF analysis misses. The same set of TF
> candles can support multiple parallel structures (e.g., 5M as both a
> child of H1.main and a child of M15.counter) with no shared state. This
> is why per-entity isolation and airtight documentation matter.

---

## 1. Naming Conventions (Locked)

These names replace prior shorthand throughout code, comments, charts, and
docs. Drop the old terms entirely once the refactor lands.

| New name | Replaces |
|---|---|
| `trading_open` | Scenario 1 |
| `reversal` | Scenario 2 |
| `subordinate` | Scenario 3 |
| `first_confluence` | (Scenario 3) Variation 1 |
| `first_counter` | (Scenario 3) Variation 2 / "UC1" |
| `subsequent_confluence` | (Scenario 3) Variation 3 |
| `subsequent_counter` | (Scenario 3) Variation 4 |

**Rationale for committing to these:** prior labels (Scenario 1/2/3, UC1,
Mode C, Variation N) were opaque. New names describe trigger or function so
they're self-explanatory on a return visit. Drop old labels, don't keep
both.

---

## 2. Structure Identity Model (Locked)

Every market structure output is uniquely identified by a tuple of:

1. **`timeframe`** — the TF of its candles (e.g., H1, M15, M5)
2. **`role`** — `main` or `subordinate`
3. **`starting_alignment`** *(subordinate only)* — `confluence` or `counter`
   - Set at structure creation by trigger origin
   - **Sticky for the structure's lifetime** — does not flip on internal
     reversals
   - Does not exist for `main`
4. **`parent_structure`** *(subordinate only)* — full identity of the
   immediate parent (recursive — parents up the chain to `main`)
5. **`parent_sid`** *(subordinate only)* — sid of the parent at creation
6. **`parent_cycle_id`** *(subordinate only)* — cycle_id within `parent_sid`
   that this subordinate is bound to

Additional structure-entity-level state (subordinate only):

- **`starting_sd`** ∈ {+1, -1} — sub's sd at creation. Fixed for the
  structure entity's lifetime.
- **`current_sd`** — sub's sd as of the current sid. Flips on internal
  reversals (each new sid has its own sd).

Useful derived comparisons:
- `current_sd == starting_sd` → "subordinate hasn't reversed since
  inception"
- `current_sd == parent.current_sd` → "currently aligned with parent",
  regardless of `starting_alignment` label

`starting_alignment` ↔ `starting_sd` mapping:
- `confluence` → `starting_sd = parent.current_sd`
- `counter` → `starting_sd = -parent.current_sd`

**`sid` is per-parent-cycle (REVISED 2026-05-25).** Within a subordinate
entity, `sid` resets to 0 at the start of each `(parent_sid,
parent_cycle_id)` and increments on each new structure. A true sub
structure is uniquely identified by `(parent_sid, parent_cycle_id, sid)`;
there are never two of the same `sid` within one `(parent_sid,
parent_cycle_id)`. This **replaces** the earlier entity-wide monotonic
`entity_sid` convention (documented in §13.5.c, now superseded). Each sid
records `started_by` ∈ {`first_confluence`, `subsequent_confluence`,
`first_counter`, `subsequent_counter`, `reversal`} and `start_trigger_idx`,
so "which trigger spawned this sid" is tracked without a separate
trigger-counter field. (`main` keeps `sid` incrementing on reversal only,
as today.)

**Two distinct "start" idxs (REVISED 2026-05-25).** A structure (and
likewise a cycle) has both:
- **`starting_idx`** — the structural anchor: the BOS-equivalent candle the
  structure is computed from. May lie in the past relative to when the
  structure becomes active.
- **lifecycle start** (first *active* idx) — the candle where the
  structure/cycle becomes active (its trigger / handoff). Before this idx
  the prior structure/cycle is still active.

These mirror main structure: a cycle's `starting_idx` is the prior BOS
candle, but the cycle becomes active at its CTS-established trigger. For a
`subsequent_*` sub sid, `starting_idx` is the probe-validated anchor while
the lifecycle start is the (later) trigger idx. The active window is
`[lifecycle_start, end_idx)`; `starting_idx` is only the geometric anchor.

**Identity is a path, not a flat tuple.** A structure like
`5M.confluence` under `15M.counter` under `H1.main` carries its full
ancestry. The path bottoms out at `main` on the highest TF.

> **Open (deferred):** the canonical encoding of this path for use as a
> dict key / chart attribution / event meta. Likely a stable
> `structure_path_id` (e.g.,
> `H1.main.sid=2.cycle=3 → M15.counter.sid=0 → M5.confluence.sid=1`),
> but exact format is TBD with the data model.

---

## 3. Recursion Model (Locked)

The hierarchy is recursive. Every subordinate treats its **immediate
parent** as its sole reference. It does not look at grandparents or main
directly.

- A 5M sub can be a child of `H1.main` **or** of `M15.<sub>`. These are
  distinct structures even if they share `(TF=5M, role=subordinate,
  starting_alignment=...)` — they differ by `parent_structure`.
- All trigger conditions, input zones, and end_idx values for a sub come
  from its **immediate parent's events / zones / TF**. Never main's
  (unless main *is* the immediate parent).
- The four `subordinate` variations apply uniformly at every nesting level,
  with "parent" replaced by "immediate parent."
- Subordinate reversals are **internal bookkeeping** — they create a new
  sid within the same structure entity but do not perturb the parent's
  cadence of variation triggers.

Future configurations the model must accommodate:
- 1H / 15M / 5M
- 4H / 15M / 1M
- arbitrary 3-TF combinations

---

## 4. Start-Candle Scenarios (Locked)

There are exactly three named scenarios for identifying `starting_idx`.
Each one has well-defined trigger, inputs, end_idx, reference zone, and
output. The `subordinate` scenario has four variations (one per
combination of {first, subsequent} × {confluence, counter}).

### 4.1 `trading_open` — Main only, used once

Used at trading session open to backfill main structure history from the
most recent candle.

| Field | Value |
|---|---|
| **Use case** | Open chart and start trading from the most recent candle |
| **Idx input** | Most recent candle |
| **Trigger** | Start of trading (one-shot per session) |
| **Output** | `starting_idx` for `main` (highest TF) |
| **Applies to** | `main` only |

No probe iteration in the new sense — uses the existing
`identify_start_scenario_1` logic. No mechanical change vs today.

### 4.2 `reversal` — Main or subordinate, fires per-reversal

Used to find the start of the structure that follows a reversal. Applies
to `main` (on highest TF) and to any `subordinate` on its own TF when the
sub itself reverses.

| Field | Value |
|---|---|
| **Use case** | Build structure after a reversal |
| **Idx input** | Most recent CTS prior to reversal |
| **Probe end_idx** | `reversal_idx` |
| **Probe reference zone** | Most recent CTS zone prior to reversal |
| **Trigger** | Reversal event |
| **Output** | `starting_idx` for the structure after reversal |
| **Applies to** | `main` and any `subordinate` |

For subordinates, `reversal` operates entirely on the subordinate's own
data and zones — not the parent's. The new sid stays within the same
subordinate entity (same starting_alignment, same parent linkage).

No mechanical change vs today's Scenario 2 logic — but probe reset
tolerance becomes TF-keyed (see §4.4).

### 4.3 `subordinate` — Four variations

Used to find `starting_idx` for new subordinate structures.

> **Session 3 (2026-05-31):** the probe now runs on the **structure's OWN sub
> TF** (the `unified_probe` primitive), NOT on the parent's TF. The old
> "probe always runs on the parent's TF, produces a parent-TF starting_idx,
> then map down" model is retired for all four variations. Each variation's
> `input_idx` and `end_idx` are mapped to the sub TF FIRST, then a single
> sub-TF probe runs. See §4.4 for the unified-probe mechanics.

#### 4.3.1 Shared probe mechanics

The probe is the `unified_probe` primitive (`engine_v2/structure/unified_probe.py`)
— a two-phase design (Phase 1 deterministic `df.pat` walk for every caller;
Phase 2 MS-based, only `first_confluence`). Full mechanics in §4.4. Key
contract:

- `input_idx` is the *initial* starting_idx candidate (already on the sub TF);
  the validated `starting_idx` is `≥ input_idx` per the forward retrace-reset
  walk.
- `reference_zone` is consulted for the 2-condition retrace reset (proximity to
  inner + toward-zone wick cap).
- Terminal conditions: reversal before 2nd CTS_EST; no qualifying retrace;
  `end_idx` reached; or (live mode) pending.

**Per-variation sd/TF + input+reference source:**

| Variation | Probe sd | Probe TF | input_idx + reference source |
|---|---|---|---|
| `first_confluence` | `+parent_sd` | sub TF | parent BOS extreme (price→M15) + own ad-hoc BOS_0 |
| `first_counter` | `-parent_sd` | sub TF | sibling confluence CTS (input == ref's `source_event_idx`) |
| `subsequent_confluence` | `+parent_sd` | sub TF | sibling counter CTS (input == ref's `source_event_idx`) |
| `subsequent_counter` | `-parent_sd` | sub TF | sibling confluence CTS (input == ref's `source_event_idx`) |

**Time/price mapping rules (sub-TF translation, Session 2 generalization
2026-05-29, still current):**

- **Price-based** (`first_confluence` input only): `parent_extreme_dir =
  -lower_sd`; the parent BOS extreme maps to the M15 candle whose extreme on
  the `-lower_sd` side touches the OUTER of the sub's reference zone.
  (`map_candle_to_lower_tf` in `data_bridge.py`.)
- **Time-based** (every probe `end_idx`, every sub window endpoint):
  `_map_parent_idx_to_m15_hour_end` — the parent gate candle maps to the LAST
  M15 candle of its hour.
- The three sibling-referencing variations need NO parent→sub price map for
  their input: `input_idx` is the sibling's CTS extreme, already an
  entity-absolute M15 idx (read directly under the frame-alignment guard).

The LANDMINE "Subordinate `parent_extreme_dir` must use `-lower_sd`" tracks the
one remaining price-map case (`first_confluence`).

#### 4.3.2 `first_confluence` — first sub of a parent cycle, sd = parent_sd

Find `starting_idx` for the **first** confluence subordinate of a parent
cycle (sid=0 of the confluence entity).

| Field | Value |
|---|---|
| **Trigger** | Most recent parent BOS confirmed (== CTS established for the new cycle, by definition same candle) |
| **Idx input** | Idx of the newly confirmed BOS |
| **Probe end_idx** | The confirmed CTS's **extreme idx** (`cts_anchor_idx` on the `CTS_CONFIRMED` event) in the same parent cycle — *not* the confirmation candle (`CTS_CONFIRMED.idx == confirmed_at`, which is later). NULL until parent `CTS_CONFIRMED` fires. |
| **Probe reference zone** | Newly confirmed active parent BOS zone |
| **Output** | `starting_idx` for first confluence sub (sid=0) |

**NULL end_idx:** This is the **only** variation where end_idx may
initially be NULL. We **wait** — no first_confluence sub is built until
parent CTS_CONFIRMED resolves end_idx (consistent with today's
"pending" handling, but the result is "do not produce" rather than
"produce tentatively"). When it resolves, `end_idx` is set to the confirmed
CTS's **extreme** idx (`cts_anchor_idx`), which is *earlier* than the
confirmation candle: the confirmation candle gates only *when* the value
becomes known; the CTS extreme is the value that bounds the probe. Using the
confirmation candle would over-extend the probe window past the CTS, shifting
the confluence sub's validated start.

**Cycle ends before parent CTS_CONFIRMED:** in theory impossible — a
reversal requires a pullback pattern that breaks the parent BOS, which
itself confirms parent CTS (either via pullback or via parent sd-proximity
during the breakout). If it does happen, no first_confluence sub is built
for that cycle.

#### 4.3.3 `first_counter` — first sub of a parent cycle, sd = -parent_sd

Find `starting_idx` for the **first** counter subordinate of a parent
cycle (sid=0 of the counter entity). Replaces today's UC1 H1-reverse
probe; mechanics unchanged.

| Field | Value |
|---|---|
| **Trigger** | First parent sd-zone proximity trigger (BOS or POI) after parent CTS |
| **Idx input** | Sibling first_confluence's most recent CTS **extreme** on the sub TF (= the reference zone's `source_event_idx`; CONFIRMED → `cts_anchor_idx`, UPDATED/EST → event idx). **Same candle as the reference zone** — input and ref are co-sourced from the sibling CTS event (Session 3 uniform rule). |
| **Probe end_idx** | First parent sd-zone proximity trigger candle after CTS (time-mapped to the sub TF's last-of-hour) |
| **Probe reference zone** | Sibling first_confluence's most recent CTS (existing KL zone if CONFIRMED; ad-hoc CTS zone if UPDATED/EST), within the sub-TF window `[parent-cycle start, this trigger]` |
| **Output** | `starting_idx` for first counter sub (sid=0) |

> **Session 3 (2026-05-31):** input_idx changed from the price-mapped parent
> CTS extreme to the sibling first_confluence CTS extreme — the SAME event the
> reference zone is built from. This completes the Session 2 pivot, which moved
> the *reference zone* to the sibling CTS but had left the *input* on the parent
> extreme. first_counter now uses the sibling CTS + zone that first_confluence
> created, identical in form to `subsequent_*`.

**Sequencing:** by trigger construction, this fires *after* parent
CTS_CONFIRMED. Since parent sd-proximity is one of the two paths that can
confirm parent CTS, parent CTS is necessarily already confirmed at this
moment — meaning `first_confluence`'s end_idx has already resolved. No
explicit "must wait for first_confluence" guard needed.

#### 4.3.4 `subsequent_confluence` — new confluence sid within same parent cycle

After a confluence sub exists in this parent cycle, each time the trigger
conditions fire, end the previous confluence sid and create
`confluence_sid_n+1` with this variation.

| Field | Value |
|---|---|
| **Trigger** | Parent CTS-zone proximity trigger AND most recent prior parent proximity trigger was to sd zones |
| **Idx input** | Sibling **counter** entity's most recent CTS **extreme** on the sub TF (= the reference zone's `source_event_idx`; CONFIRMED → `cts_anchor_idx`, UPDATED/EST → event idx). **Same candle as the reference zone.** |
| **Probe end_idx** | Current parent CTS-zone proximity trigger candle (time-mapped to the sub TF's last-of-hour) |
| **Probe reference zone** | Sibling **counter** entity's most recent CTS (existing KL zone if CONFIRMED; ad-hoc CTS zone if UPDATED/EST), within the sub-TF window `[last-M15-of prior_sd_prox hour, last-M15-of this_cts_prox hour]` |
| **Output** | `starting_idx` for next confluence sid |

**Reference zone + input resolution (Session 3 uniform rule, 2026-05-31):**
1. Build the sub-TF idx window: `[_map_parent_idx_to_m15_hour_end(prior_sd_prox_idx), _map_parent_idx_to_m15_hour_end(this_cts_prox_idx)]` (entity-absolute M15).
2. Walk the **counter** entity's events (filtered to this `(parent_sid, parent_cycle_id)`) within that window, take the most recent of {CTS_CONFIRMED, CTS_UPDATED, CTS_ESTABLISHED} via `build_reference_zone_from_cts_event`.
3. `input_idx = ref_zone.source_event_idx`; `reference_zone = ref_zone`. Probe runs on the confluence entity's own M15 frame (sibling read is by entity-absolute M15 idx, valid under the frame-alignment guard).
4. **Fallback** (sibling has zero CTS events in the window — extremely rare): input_idx = window extreme toward parent BOS computed on the confluence M15; reference = own ad-hoc BOS_0 from that candle.

> **Session 3 supersedes** the old "parent-TF window extreme toward BOS" input
> rule + the "counter-sub zone attached to the lower-TF extreme" drill-in
> reference rule. The probe now runs on the SUB TF (not parent TF), and input
> and reference are co-sourced from the sibling counter CTS event — uniform
> with first_counter and subsequent_counter. The pre-Session-3 drill-in
> resolution + its 20-pip fallbacks are retired.

#### 4.3.5 `subsequent_counter` — new counter sid within same parent cycle

After a counter sub exists in this parent cycle, each time the trigger
conditions fire, end the previous counter sid and create
`counter_sid_n+1` with this variation.

| Field | Value |
|---|---|
| **Trigger** | Parent sd-zone proximity trigger AND most recent prior parent proximity trigger was to CTS zone AND the proximity trigger before *that* was to sd zones (forms Λ in bullish parent / V in bearish parent) |
| **Idx input** | Sibling **confluence** entity's most recent CTS **extreme** on the sub TF (= the reference zone's `source_event_idx`; CONFIRMED → `cts_anchor_idx`, UPDATED/EST → event idx). **Same candle as the reference zone.** |
| **Probe end_idx** | Current parent sd-zone proximity trigger candle (time-mapped to the sub TF's last-of-hour) |
| **Probe reference zone** | Sibling **confluence** entity's most recent CTS (existing KL zone if CONFIRMED; ad-hoc CTS zone if UPDATED/EST), within the sub-TF window `[last-M15-of prior_cts_prox hour, last-M15-of this_sd_prox hour]` |
| **Output** | `starting_idx` for next counter sid |

> **Session 3 (2026-05-31)** supersedes the old "parent-TF candle closest to
> parent CTS zone outer bound (Λ apex / V trough)" input rule + "active parent
> CTS zone" reference rule. Input and reference are now co-sourced from the
> sibling confluence CTS event, uniform with first_counter and
> subsequent_confluence. The Λ/V apex geometry below is retained as the
> conceptual picture of WHERE the counter structure starts, but the actual
> anchor is the sibling confluence CTS, not a parent-TF apex candle.

**Lambda / V geometry (canonical names):**
- **Bullish parent** (parent_sd = +1): BOS zone bottom, POI middle, CTS
  zone top. Three trigger sequence sd → CTS → sd traces a **Λ** (lambda)
  with apex at parent CTS zone.
- **Bearish parent** (parent_sd = -1): mirrors — BOS top, POI middle,
  CTS bottom. Three trigger sequence sd → CTS → sd traces a **V** with
  trough at parent CTS zone.

Reference image: `artifacts/bullish_parent_lambda_proximity.jpeg`.

#### 4.3.6 Cadence of triggers within a parent cycle

By construction, the four variations naturally trigger in this order
within each parent cycle:

```
parent BOS_confirmed
    → first_confluence              (input: own ad-hoc BOS_0, end: parent CTS confirmed)
parent 1st sd-proximity (post-CTS)
    → first_counter                 (input+ref: sibling confluence CTS, end: this trigger candle)
parent CTS-proximity (after sd-prox)
    → subsequent_confluence         (input+ref: sibling counter CTS, end: this trigger candle)
parent sd-proximity (after CTS-prox after sd-prox = Λ/V apex at CTS)
    → subsequent_counter            (input+ref: sibling confluence CTS, end: this trigger candle)
parent CTS-proximity
    → subsequent_confluence
parent sd-proximity
    → subsequent_counter
... alternating until parent cycle ends
```

**Alternation is strictly enforced** by the parent's proximity trigger
state machine (`zone_proximity.py` already does sd/opp_sd alternation).
Both `subsequent_*` variations require the prior trigger to be the
opposite kind; consecutive same-kind proximity events do not exist by
construction.

**Trigger collisions on the same candle** are extremely rare but allowed
— neither variation blocks the other. Each variation's own trigger
predicate self-gates; we don't need explicit ordering logic in code. (One
known impossible-in-practice case: a single candle simultaneously
triggering sd-proximity and CTS-proximity is geometrically infeasible
because a green candle moves toward CTS and a red candle moves toward
BOS/POI, so a single candle can only trigger one.)

**Bootstrap (first parent cycle after `trading_open`):** if
`trading_open` lands the main start mid-parent-cycle, no
`first_confluence` is built for the in-progress cycle. The cadence begins
at the next parent BOS_confirmed.

#### 4.3.7 Variations summary

| Variation | Trigger | Input idx | End idx | Reference zone | Probe sd |
|---|---|---|---|---|---|
| `first_confluence` | Parent BOS_confirmed | Parent BOS extreme (price-mapped to M15) | Parent CTS **extreme** (`cts_anchor_idx`; NULL until parent CTS_confirmed fires) | **Ad-hoc BOS_0 on sub TF** (derived from input_idx candle on M15) | `+parent_sd` |
| `first_counter` | 1st parent sd-proximity post-CTS | **Sibling confluence CTS extreme** (= ref's `source_event_idx`) | This trigger candle | **Sibling first_confluence's most recent CTS** (existing zone if CONFIRMED; ad-hoc CTS if UPDATED/EST) | `-parent_sd` |
| `subsequent_confluence` | Parent CTS-proximity after sd-prox | **Sibling counter CTS extreme** (= ref's `source_event_idx`) | This trigger candle | Sibling counter's most recent CTS (in M15 window; same rule) | `+parent_sd` |
| `subsequent_counter` | Parent sd-prox forming Λ / V | **Sibling confluence CTS extreme** (= ref's `source_event_idx`) | This trigger candle | Sibling confluence's most recent CTS (in M15 window; same rule) | `-parent_sd` |

**Uniform input+reference rule (2026-05-31, Session 3).** All three
sibling-referencing variations (`first_counter`, `subsequent_confluence`,
`subsequent_counter`) co-source BOTH `input_idx` AND `reference_zone` from the
**sibling entity's most recent CTS event** (CONFIRMED → existing KL zone +
`cts_anchor_idx`; UPDATED/EST → ad-hoc CTS zone + event idx), found by walking
the sibling's events within the trigger's sub-TF idx window. `first_confluence`
is the only exception — it has no sibling/prior structure yet, so it anchors on
its own ad-hoc BOS_0 from the parent-BOS-extreme input. The probe always runs
on the structure's OWN sub TF.

**Reference-zone pivot history.** Session 2 (2026-05-29, post-Gate-1) first
moved `first_*` off the parent's wide H1 zones onto sub-TF zones (ad-hoc BOS_0
for confluence; sibling CTS for counter) — but left `first_counter`'s *input*
on the parent CTS extreme, and left `subsequent_*` on the legacy parent-TF
Scenario-3 probe. Session 3 (2026-05-31) completed the migration: `subsequent_*`
moved to the unified probe, and all three sibling-referencing variations adopted
the co-sourced input+reference rule above. The earlier "parent extreme input" /
"drill-in counter-sub zone" / "active parent CTS zone" rules are retired.

**Build-order dependency.** Every sibling-CTS lookup (first_counter →
confluence; subsequent_confluence → counter; subsequent_counter →
confluence) requires the referenced sibling sid to be built before the
reading probe fires. Because the reads point in OPPOSITE directions, no
"build one entity fully then the other" order satisfies all of them.
Session 3 (2026-05-31) replaced the confluence-first serial pair
(`_run_first_confluence_multi_tf` then `_run_multi_tf`) with a single
two-entity cadence driver, `_run_multi_tf_dual` →
`build_two_entity_parent_cycle`, that interleaves both M15 chains in
trigger-cadence order (advancing whichever `_ChainCursor` has the smaller
next M15 trigger boundary). Cadence guarantees each sibling CTS read is
strictly earlier than the reading sid's own trigger, so the sibling sid is
always already built. See LANDMINES "Cross-entity sibling references
require cadence-order interleaving".

---

## 4.4 Probe reset thresholds (applies to unified `reversal` + `subordinate` probe)

The unified probe primitive (Phase 1 design 2026-05-29 — see
`memory/project_unified_identify_start_probe.md`, code in
`engine_v2/structure/unified_probe.py`) replaces today's four asymmetric
paths. Per iteration it picks the single most-extreme retrace candle in
`[first_CTS_EST.idx + 1, end_idx]` and applies a **two-condition reset**;
both must hold for the probe to restart from that candidate.

### Two-condition reset

**Condition 1 — proximity to inner.** Candle wick extreme within
`DEFAULT_PROBE_RESET_PIPS[tf]` pips of `reference_zone.inner`.
- `direction=+1`: `low ≤ inner + reset_tol`
- `direction=-1`: `high ≥ inner - reset_tol`

**Condition 2 — toward-zone wick cap.** Body-bounded
(`body_top = max(o,c)`, `body_bottom = min(o,c)`). The "toward-zone"
wick is the side opposite `direction` (= the side pointing at the zone
outer). Must be ≤ `DEFAULT_PROBE_RESET_WICK[tf]` pips.
- `direction=+1`: `body_bottom - low ≤ wick_cap`  (lower wick)
- `direction=-1`: `high - body_top ≤ wick_cap`    (upper wick)

Condition 2 is the new ingredient — rejects single-candle stab wicks
that pass condition 1 today but represent transient spikes rather than
structural retraces.

### Probe methods + cycle-0 true-first-breakout (2026-06-07)

> Canonical summary; full design + rationale in
> `memory/project_true_first_breakout_cycle0.md`. Supersedes the prior
> "two-phase / `df.pat` walk / partial-gate" text.

**The cycle-0 CTS (CTS_0_EST) is the EARLIEST true breakout** meeting all of:
1. anchor's **close** past the **BOS_0 inner** (`bos0_inner`) in the probe
   direction — a hard gate even on the confirm path;
2. a **valid breakout pattern** re-detected with `break_threshold = bos0_inner`
   (mechanism B — NOT the threshold-free `df.pat`); 30%-body failures may be
   CONFIRMED via the existing pattern-extreme confirmation;
3. a **strict** new full-pattern extreme over `[current_start, extreme_candle)`
   (`>`/`<`; ties do NOT count);
4. **earliest apply/confirm idx** wins, tie-break `continuous > double_maru >
   one_maru_continuous > one_maru_opposite` (a cycle-0-specific order — the
   global `detect_best_for_anchor` priority is unchanged).

This is implemented ONCE in `engine_v2/structure/true_first_breakout.py::find_true_first_breakout`,
called by both the probe and MS.

**Two zones (don't conflate):** the **BOS_0 threshold** that gates the breakout
MOVES with `current_start` (iter 1 = the reference zone inner; iter 2+ = a fresh
ad-hoc `bos=True` BOS_0 at the reset start, via
`reference_zone.build_ad_hoc_bos0_reference_zone`). The **retrace-reset
reference** is CONSTANT (the very-first reference zone) and drives the unchanged
2-condition reset (cond1 proximity to reference inner; cond2 toward-zone wick
cap).

**Deterministic vs iterative = data availability, not two logics** (same
conditions, same routine): resolve in one windowed pass over historical data
(deterministic); process candle-by-candle past the live edge (iterative). Any
run uses deterministic over available history, then hands to iterative past the
last available candle. **Commit-1 is all backtest → the iterative path is
dormant** (like the `pending` finalize conditions).

**Per caller:**
- **non-FC** (`first_counter` / `subsequent_*` / `reversal`): `end_idx = trigger`
  (all historical) → fully **deterministic** — find the true breakout + the
  max-retrace reset over `[CTS_0_EST+1, end_idx]`, multiple resets, NO MS, no
  `cts_anchor`.
- **first_confluence** (hybrid): deterministic window upper = the time-mapped
  parent CTS_EST; the real `end_idx` (parent CTS anchor) is treated as NULL →
  the probe runs MS (`enable_phase2=True`) only to reach CTS_0_CONFIRMED →
  `cts_anchor`, which bounds the retrace (`[CTS_0_EST+1, cts_anchor-1]`, else
  `[CTS_0_EST+1, end_idx]`).
- **main sid0|cyc0** (`trading_open`): arbitrary ad-hoc BOS_0 at the
  `identify_start_scenario_1` start; single-shot, no resets (Commit 2 — not yet
  wired).

### MS pre-CTS_0 scan-from-start (`enforce_cts0_new_extreme` + `bos0_inner`)

The probe hands MS a **decision, not events**: `{finalized current_start, BOS_0
bounds}` (+ `cts0_est_idx` as a sanity-assert). MS does **NOT** seed state at
CTS_0 (the earlier "seed-and-resume" was rejected — see the design doc's UPDATE
block). Instead, with `enforce_cts0_new_extreme=True` MS runs the **pre-CTS_0
scan-from-start** mode: while cycle 0 is unestablished it delegates the breakout
search to the SAME `find_true_first_breakout` routine (using the **handed
`bos0_inner`** — REQUIRED in scan mode, else `ValueError`), establishes CTS_0 at
the located winner via its **normal cycle-0 path** (the establishment bundle:
`BOS_CONFIRMED` anchored at start w/ `confirmed_at=CTS0_EST`, `CTS_ESTABLISHED`,
thresholds, BOS_0 zone, state), then resumes normal MS. Because probe and MS use
the same routine + same `bos0_inner` over the same data, they agree on CTS_0 **by
construction**. Cycles ≥ 1 are unaffected (they break the prior CTS). The old
partial anchor-extreme gate (`_cts0_new_extreme_passes`) was removed.

`bos0_inner` is a **price** (slice-invariant), threaded through
`compute_bounded_structure(enforce_cts0_new_extreme, bos0_inner)` and the
`build_one_sid` / `_ChainCursor` handoff (`SidBuildOutcome.next_bos0_inner` for
reversal-born subs) — no index remap.

**Scope:** all M15 subs use this (Commit 1, every trigger). **Deferred:** main
`sid0|cyc0` (Commit 2 — flip the flag + pass `bos0_inner` at the main
`compute_structure` call); main reversals H1 `sid≥1` (Step 4 — migrate to
`unified_probe` + scan-from-start + delete Exception 1). The legacy
`_resolve_via_legacy_probe` is an empty escape hatch (`_LEGACY_PROBE_USE_CASES`).

### Threshold tables

All four tables live in `engine_v2/zones/zone_proximity.py`. Module-load
asserts guarantee no inversion at startup.

| TF | Probe reset cond 1 (`reset_pips`) | Probe reset cond 2 (`wick_cap`) | Proximity trigger | Narrow-cycle min gap |
|---|---|---|---|---|
| H1 | 4 | 16 | 8 | 50 |
| M15 | 3 | 12 | 5 | 30 |
| M5 | 2 | 8 | 3 | 15 |

### Invariants (asserted at module load)

```python
# Existing — locks probe-reset semantics below proximity-trigger semantics:
assert DEFAULT_PROXIMITY_PIPS[tf] > DEFAULT_PROBE_RESET_PIPS[tf]

# Existing — keeps narrow-gap restriction strictly above proximity:
assert DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS[tf] > DEFAULT_PROXIMITY_PIPS[tf]

# NEW — a wick cap below the proximity tolerance is self-contradictory
# (would make condition 1 unreachable):
assert DEFAULT_PROBE_RESET_WICK[tf] > DEFAULT_PROBE_RESET_PIPS[tf]
```

### Why proximity > reset matters

If probe reset ≥ proximity trigger on the same TF, the probe could detect a
"return to BOS_0 zone" at a pips distance that the zone-proximity-trigger
pipeline already counted as a real proximity event. The probe would then
push `starting_idx` forward into territory the variation logic considers
post-trigger — making the variation's `end_idx` and `starting_idx` overlap
or invert. Strict `proximity > reset` is the cleanest invariant.

### Migration state

- **Tables wired** as of Phase 1 Session 1 (this update). The probe
  primitive itself exists at `engine_v2/structure/unified_probe.py` but
  no caller has migrated yet — Sessions 2–6 of Phase 1 migrate per
  trigger. Until then, the legacy probes
  (`compute_structure_scenario_3` Phase 1, `compute_structure` Exception
  2) continue to use `DEFAULT_PROBE_RESET_PIPS` only (single-condition).
- **Proximity tuning deferred**: a follow-up tightens `DEFAULT_PROXIMITY_PIPS`
  to `{H1: 8, M15: 6, M5: 4}`. Held back from Session 1 so the unified
  probe lands byte-identical to the `804d19d` baseline.

---

## 5. Computing Structures — Function Routing per Role

For `main` (highest TF), continue using `compute_structure`:

- Initial start: `trading_open` scenario (today's `identify_start_scenario_1`).
- On reversal: `reversal` scenario (today's `identify_start_scenario_2_after_reversal` + Exception 1/2).
- Loops until end of data.

For `subordinate` (any role/alignment, any TF), each **sid** is built by a
**bounded single-structure run** of `compute_structure_from_start`
(REVISED 2026-05-25):

- Start is pre-validated by the upstream variation probe (for
  `subsequent_*` sids) or by `identify_start` (for `reversal` sids) — no
  internal Scenario 1 logic.
- The run is **bounded** to `[starting_idx, end_idx]` where `end_idx =
  min(next subsequent-variation trigger, parent-cycle-end)`, and it
  **stops at its own first reversal** if one occurs inside that window. A
  sub sid is therefore a SINGLE directional structure — it never rolls
  past a reversal (the reversal is the boundary to the next sid). This is
  the change that eliminates the "phantom" structure/zones a sub
  previously produced by running past where the next trigger should have
  taken over.

The legacy `compute_structure_scenario_3` ad-hoc path is **removed**. Its
only purpose (creating WVMI when parent was in range) is now subsumed by
per-subordinate WVMI — see §8.

### sid increment rules

| Role | sid +1 triggers |
|---|---|
| `main` | reversal only |
| `subordinate` | reversal **OR** subsequent variation (var 3 / var 4) |

Sids within a `(parent_sid, parent_cycle_id)` form one **sequential,
non-overlapping chain** (merge-and-bound — see §6.1). Each sid's bounded
run ends exactly where the next sid begins; sids never overlap and never
overwrite one another. They share only the entity's df, the append-only
events list, and the zone/POI/fib/WVMI collections (append-only, keyed by
`sid` + `cycle_id`).

---

### Unified lifecycle & start/end model (Locked — REVISED 2026-05-26)

Applies to **both** main and subordinate. Starts are the primitives; ends
are **pass-throughs** of the next start's idx. Zones compute no end of
their own — they inherit their owning cycle's resolved end. A *structure*
is identified by `sid` on main and by `(parent_sid, parent_cycle_id, sid)`
on a sub.

```
Zones end when:              its cycle ends

Cycles end when:             1. next cycle of the SAME structure starts
                             2. its structure ends

Structures end when:         1. next structure starts
                                  main: sid+1
                                  sub:  (parent_sid, parent_cycle_id, sid+1)   [parent_* unchanged]
                             2. (sub only) same parent's next cycle starts
                             3. (sub only) parent structure ends

ANY new cycle starts when:   new CTS established
                             — CLAMP: a cycle's lifecycle-start may not precede
                               its structure's lifecycle-start (main & sub). For a
                               sub it ALSO may not precede the parent sid's
                               lifecycle-start NOR the parent_cycle_id's
                               lifecycle-start. If the CTS-established idx is
                               earlier than any floor, the cycle's true
                               lifecycle-start is the LATEST of the (up to 3)
                               floors: own-structure start, parent_sid start,
                               parent_cycle_id start (the tuple identifier).

Next structure starts when:  1. reversal triggers
                             2. (sub only) subsequent use_case triggers
                             — CLAMP (sub only): a new sub structure's
                               lifecycle-start may not precede the parent sid's
                               lifecycle-start NOR the parent_cycle_id's
                               lifecycle-start. If earlier than either, it snaps to
                               the LATEST of the 2 parent floors.

Zones start when:            the first time they become active (first confirmed),
                             clamped to their cycle's lifecycle-start (see below).
```

- **`end_idx` = the idx of the start event that supersedes.** When a start
  fires at idx X, X becomes the `end_idx` of whatever it ends. Ends use the
  *clamped* next-start idx (matters only for subs/collapse — see below).
- **Propagation closes the open child.** Whenever a *structure* ends (for
  ANY reason in the Structures-end list — including the sub-only
  parent-next-cycle / parent-structure-end cases), its currently-open
  *cycle* ends at the same idx, and that cycle's *zones* end with it. This is
  how the last cycle of a structure that ends via parent-cycle-end still
  closes cleanly even though no "next cycle / next structure starts" event
  fired inside the entity.
- **"Any new cycle" vs "next structure."** *Every* cycle start — whether the
  next cycle on the same sid, cycle 0 of a reversed sid, or cycle 0 of a new
  sub sid — is triggered by a new CTS established, so the cycle-start rule
  and its clamp are universal. "Next structure" is deliberately narrower
  (the `sid+1` successor with parent identity fixed); the parent-driven sub
  endings are separate structure-end causes.

##### starting_idx vs lifecycle-start, and the clamp (REVISED 2026-05-26)

Each structure and each cycle has two distinct idxs (the §2 split,
generalized):

- **`starting_idx`** — the structural anchor: the retroactive
  BOS-equivalent candle the probe / `identify_start` selected. May sit
  *historically* before the entity is even alive.
- **lifecycle-start** — the first idx at which it is *active*:
  - **main structure:** sid 0 → its `starting_idx`; sid N≥1 → the reversal
    confirmation idx of sid N−1 (`STATE_CHANGED to=='reversal'`).
  - **sub structure:** its trigger idx (`start_trigger_idx`, §6.1), clamped
    `≥ max(parent sid lifecycle-start, parent_cycle_id lifecycle-start)`, i.e.
    `lifecycle-start = max(start_trigger_idx, parent_sid_start, parent_cycle_start)`.
  - **cycle (any):** `max(CTS_established_idx, owning-structure
    lifecycle-start)`. The structure's lifecycle-start already embeds BOTH parent
    floors (sid + cycle) for subs, so the parent floors enter once (at the
    structure level) and cycles inherit them transitively.

  **Parent floor mapping (H1→M15, 2026-05-27).** A sub's parent floors are
  parent-TF idxs and must be mapped to the entity's TF:
  - `parent_cycle_id lifecycle-start` = the parent cycle's `CTS_ESTABLISHED`
    `ev.idx` (H1), mapped to M15 via **last-of-hour**
    (`_map_parent_idx_to_m15_hour_end`). Last-of-hour (not first) because the H1
    candle isn't closed until its 4th M15 sub-candle, AND because the prior
    cycle's sub *end* maps the same way → cycle k's start lands on the same candle
    the prior cycle's sub ended (shared boundary, no gap). NB in this engine BOS
    `confirmed_at` == CTS-established `ev.idx`, so the start anchor coincides with
    the prior-cycle end anchor.
  - `parent_sid lifecycle-start` = the parent cycle-0 floor, reversal-aware for
    sid≥1: `max(map(CTS_0-established), map(reversal_confirmed[sid]))`.

  **STATUS (2026-05-27).** The `parent_sid` floor is effectively in force today
  (auto-satisfied: a sub trigger is always within its parent sid's life). The
  `parent_cycle_id` floor is **IMPLEMENTED 2026-05-27 (plan B1)**: a shared
  `compute_struct_start_by_sid` pure-leaf helper (`zones/structure_lifecycle.py`)
  used by both KL + POI, and `build_one_sid`'s `lifecycle_floor` widened to
  `max(start_trigger_idx, parent_sid_start, parent_cycle_start)` (parent floors
  mapped H1->M15 last-of-hour in `build_parent_cycle_chain`). Validated /compare
  vs `310c395`: H1 + counter + WVMI + sids byte-identical, chart counts unchanged;
  only 5 `first_confluence` bootstrap subs floored their lifecycle-start (none
  collapsed). Only the `parent_cycle_id` term changed behavior. **Lifecycle-only:**
  `base_idx`,
  rectangle outline, BOS/CTS, the MS run, `start_trigger_idx` (sub-WVMI window),
  and `SidRecord.creation_event_idx` are all unchanged — only first-active /
  `confirmed_idx` / fill move. The END-side unification (+ dedup of the reversal
  dict currently duplicated across KL `_get_reversal_confirmed_by_sid_from_events`
  and POI inline `reversal_idx_by_sid`) is step "B2" — end-condition verification
  is **done** (2026-05-27) and the agreed pass-through end model + Phase A/B plan
  are locked below in "End resolution as start-passthrough". Full writeup:
  `memory/project_cycle_lifecycle_parent_cycle_floor.md`.

Why the clamp is needed: a reversal's confirmation candle is when the new
sid's lifecycle begins, but the probe can place the new sid's `starting_idx`
historically, so its cycle-0 `CTS_ESTABLISHED` can fall *before* the
reversal confirmation. That cycle cannot have started before its structure
did. (Concrete instance in the 2025-12 NZD_USD run: sid 1 cycle 0
`CTS_ESTABLISHED` at idx 703, sid 0 reversal at idx 710 → cycle 0's
lifecycle-start clamps to 710.) The same floor applies down the hierarchy so
sub cycles/structures never start before their parent.

- **Collapse is allowed.** If a clamped lifecycle-start lands at or past the
  owning cycle's/structure's end, that cycle/zone simply never has an active
  window (it stays in the list as history, `status` never `active`). For
  subs whose anchor sits well behind the parent start, several early cycles
  can collapse this way.
- **Anchor unchanged; only the active window moves.** A zone rectangle is
  still *drawn* from its `starting_idx` / base / IC candle; the clamp only
  raises its **first-active / `confirmed_idx` / `status`** to the cycle
  lifecycle-start. Structure events, df columns, patterns, and the structural
  `starting_idx` are immutable facts — the clamp never rewrites them (so
  `structure_events` stays byte-identical; any shift there is a red flag).

- **Unifies KL and POI (Phase 3, 2026-05-26).** POI already resolved
  `end_idx` + `end_reason` by the reversal / next_cycle priority and gated
  activation at `max(cts_established_idx, ic_idx)`. KL now uses the same
  end-resolution (its prior scattered `end_time` mechanisms — CTS_ESTABLISHED
  early-end, same-side replacement, reversal/lifecycle caps — are removed)
  and adopts the active/inactive/ended convention
  (`end_idx`/`end_reason`/`activation_history`/`status` in `meta`). Both
  zone kinds also floor first-active at the cycle lifecycle-start per the
  clamp. **End-side change:** both BOS and CTS zones of cycle *n* now end at
  the next cycle's `CTS_ESTABLISHED` `ev.idx` (the CTS extreme — the idx POI
  uses). The CTS-zone end is unchanged (already this idx). The BOS-zone end
  moves from the old next-BOS-`confirmed_at` (breakout) to that extreme idx —
  identical when the breakout candle *is* the extreme, else ≤1 candle earlier.
  Empirically the H1 main is byte-identical (breakout == extreme for all
  sampled cycles); the M15 subs (BOS-only zones) show one 1-candle BOS-end
  shift each. **Start-side change:** post-reversal cycle-0 zones (and sub
  analogues) shift their active-start forward to the clamped lifecycle-start.
  See `zones/KL_ZONES_SPEC.md` "Lifecycle" and `ARCHITECTURE.md`.
- **Scope of the unification.** The pass-through lifecycle model governs
  **zones and cycles** (KL + POI). It deliberately does **not** apply to
  structure events or patterns — those are immutable append-only facts whose
  only time-varying property is currency (`owner_by_idx` / §16.5), not
  lifecycle state. `FibState` *should* eventually adopt the convention (it has
  both axes) but its migration is deferred to its own session. Both decisions
  are written up in `ARCHITECTURE.md` "Lifecycle state convention".

##### End resolution as start-passthrough — the B2 unification (Locked 2026-05-27)

End-condition verification (2026-05-27, against commit `2573990`) confirmed the
model above is correct and the two zone-end resolutions (KL `kl_zones_v1`, POI
`poi_zones`) are already byte-identical. This subsection locks how the ends are
*computed*, completing on the end side the pass-through model B1 began on the
start side.

**Ends are derived from starts — never computed independently.** Every "end" is
the lifecycle-**start** idx of whatever supersedes it, propagated down
(structure-end → cycle-end → zone-end). So the same start machinery B1 unified is
the single source for all ends:

- **cycle end** = `min(` next-cycle-on-same-structure **clamped** start,
  own-structure end, [sub] parent-cycle end, [sub] parent-sid end `)`.
- **structure end** = `min(` next-structure start, [sub] parent-cycle end,
  [sub] parent-sid end `)` — itself a min of superseding starts.

These nest but are computed at once. **`min` for end mirrors `max` for start:** a
cycle starts no earlier than the latest of its floors (can't begin before any is
alive) and ends at the earliest of its caps (the first end condition wins; a
later one is moot).

**Symmetric floor / cap (the sub cross-layer seam).** Main computes its whole
table from the H1 event stream. A sub cannot — its within-structure ends (next
sub-cycle, next sub_sid) live in the bounded-run/chain while its parent ends
(parent-cycle / parent-sid paths) live in the H1 layer. So, exactly mirroring
B1's start `lifecycle_floor`:

- **`lifecycle_floor`** (single int, `max`) — start floor. [B1]
- **`lifecycle_cap`** (single int, `min`) — end cap. [B2]

Both are `None` for main and supplied (slice-local) by the multitf layer for
subs. The cap is **load-bearing**: for a sub it is normally `end_m15_abs`, which
is also the upper bound of the M15 slice the bounded structure runs on — so it
must be computed *before* the run (in `build_parent_cycle_chain` /
`build_one_sid`, as today).

**The data/window boundary is NOT a lifecycle terminator (2026-05-27).** A
lifecycle ends only on a **real event** — a reversal, or a genuine next
cycle/structure forming — never because the data ran out. The backtest's right
edge is just "the present" (in live, every moment between candles you sit at the
last available candle, yet active elements stay active). So:

- The **`lifecycle_cap` is a real structural end ONLY**: the sub's own reversal,
  or a parent-cycle/next-sub boundary that genuinely formed. When a sub instead
  runs to the **open data edge** — its parent cycle is the open last one
  (`trigger.lifecycle_end_idx is None`, the signal that `_find_m15_lifecycle_end`
  fell back to `entity_df.index[-1]`), with no subsequent trigger after it
  (`cap_open` in `build_parent_cycle_chain`) — the cap is **`None`**. KL / POI /
  Fib then stay **active to the edge**, exactly like the H1 main's open last
  cycle (whose cap is always `None`). This makes subs consistent with main.
- **Run bound vs lifecycle cap are separate.** The bounded structure still *runs*
  on `[start, end_m15_abs]` (you can only run on data you have), and
  `SidRecord` run-metadata still records `end_m15_abs`. Only the lifecycle **cap**
  (what *terminates* zones) is dropped to `None`. `cap_open` is keyed on
  `lifecycle_end_idx is None`, NOT on `end_m15_abs == last idx`, so a rare
  H1→M15 mapping failure (which also falls back to the edge) is still treated as
  a real cap.

**Implementation (`zones/structure_lifecycle.py`).** A pure-leaf
`compute_cycle_lifecycle(events, reversal_idx_by_sid, lifecycle_floor,
lifecycle_cap) -> Dict[(sid, cycle), (start_idx, end_idx, end_reason)]`:
  1. *Pass 1 — clamped cycle starts:* `start = max(CTS_ESTABLISHED.ev.idx,
     struct_start, floor)` (`struct_start` from `compute_struct_start_by_sid`,
     the B1 helper).
  2. *Pass 2 — ends from next starts:* `end = min(next-cycle clamped start,
     reversal_idx_by_sid[sid], cap)`; `end_reason` records which won
     (`next_cycle` / `reversal` / `lifecycle_end`). End uses the next cycle's
     **clamped** start (not its raw `CTS_EST.idx`); these differ only for
     collapsed sub cycles.
The reversal dict is built once by a shared `compute_reversal_idx_by_sid(events)`
(retiring the duplicated KL `_get_reversal_confirmed_by_sid_from_events` vs POI
inline `reversal_idx_by_sid`) and passed to both the start and the end helpers.

**Elements inherit the END only; they keep their own START.** When a cycle ends,
every attached element ends with it, so each KL / POI (later Fib / WVMI) looks up
its `(sid, cycle)` `end_idx` / `end_reason` from the table. **Nothing inherits
cycle start** — each element computes its own first-active by its own
active/inactive logic, clamped up to the cycle/structure floor (B1). BOS KL zones
*coincide* with cycle start but keep computing their own breakout start
("Option 2", 2026-05-27) so the unification stays byte-identical; the ≤1-candle
extreme-vs-breakout divergence on a few M15 sub cycles is accepted as the zone
confirming one candle into its just-opened cycle. active/inactive state stays
element-specific (POI's per-candle `activation_history`, KL's single interval).

**Collapsed cycles** (clamped `start >= end`, common for reversal-born subs whose
anchor sits behind the parent start) → `status = "inactive"`, empty
`activation_history` (renders outline-only). This standardizes the prior split:
the inner KL derivation said `"ended"`, the `build_one_sid` cap said `"inactive"`.

**Phase B (DONE 2026-05-27, byte-identical).** The parent-cycle / parent-sid
paths (the sub `lifecycle_cap`) ARE the *next parent `(sid, cycle)`'s clamped
start* (the canonical `CTS_EST.ev.idx`). Previously the cap's next-cycle term came
from `trigger.lifecycle_end_idx` = next-parent-cycle `BOS_CONFIRMED.confirmed_at`
(the apply candle) — a latent overlap hazard, since the start-floor uses
`CTS_EST.ev.idx` and the two diverge if a parent H1 cycle's CTS-extreme ≠ breakout
candle. **Implementation (approach A):** the four trigger detectors (`uc1`,
`first_confluence`, `subsequent_confluence`, `subsequent_counter`) now compute the
next-cycle term from the next cycle's `CTS_ESTABLISHED.ev.idx` instead of
`BOS_CONFIRMED.confirmed_at`; the reversal term (`REVERSAL_CANDIDATE.apply_idx`) is
unchanged. `lifecycle_end_idx` is **kept** (not retired — still the carrier read by
`_find_m15_lifecycle_end`); full retirement (sourcing the cap directly from
`parent_cycle_floor_h1`) was rejected because it would also swap the reversal-cap
source and is not guaranteed byte-identical. Byte-identical on this data because
`confirmed_at == CTS_EST.ev.idx` for all 6 H1 cycles (verified 0 mismatches) — a
pure robustness fix, zero current behavior change.

---

## 6. Subordinate Cadence and Lifecycle Rules

(§4.3.6 covers the within-parent-cycle cadence diagram. This section
specifies how those triggers compose into the sequential sid chain and how
lifecycles bound it. **REVISED 2026-05-25** — the prior "overwrite
semantics" framing, including the cascade, is removed.)

### 6.1 Within parent cycle — sequential sids (merge-and-bound), no overwrite (REVISED 2026-05-25)

Within one `(entity, parent_sid, parent_cycle_id)`, sids are built as a
single **sequential, non-overlapping chain** — NOT as independent
overlapping runs reconciled by an overwrite cascade (the prior design,
removed).

Build:
1. Collect this entity+parent-cycle's trigger idxs in time order
   (`first_confluence` / `first_counter`, then the `subsequent_*` triggers
   per the §4.3.6 cadence).
2. **sid=0** starts at the first-variation validated start
   (`first_confluence` for the confluence entity, `first_counter` for the
   counter entity). **sid=0 is always a first-variation; a `subsequent_*`
   can never be sid=0.** If the first variation never yields a valid sub
   for this parent cycle, the entity has no sequence here and the cycle's
   `subsequent_*` triggers do not build (no sid=0 to follow).
3. Run sid=0's bounded single-structure (§5) to the first of {its own
   reversal, the next `subsequent_*` trigger, parent-cycle-end}.
4. That boundary starts **sid+1**:
   - boundary = `subsequent_*` trigger → sid+1 starts at the trigger's
     probe-validated start, with the use_case's direction (same-direction
     consecutive sids ARE allowed for subs — the boundary is the trigger,
     not a reversal).
   - boundary = reversal → sid+1 via `identify_start` (`reversal`
     scenario), flipped direction (exactly like main).
5. Repeat to parent-cycle-end.

**No overwrite, no bounds-capping, no cascade.** Each sid's run ends
exactly where the next begins, so there is no overlap to reconcile. "Only
the most recent sub of a type is alive" falls out for free: a new sid (or
new cycle) ends the prior one at the start idx via the §5 lifecycle model.
Old sids' events / zones / POIs / fibs / WVMI persist append-only with
`(parent_sid, parent_cycle_id, sid)` + `cycle_id` attribution; their ends
are the resolved lifecycle ends, not a cascade-imposed cap.

**Degenerate / short-lived triggers still create sids** — valid while
active (live-trading semantics). No min-span skip.

**Handoff idx.** A `subsequent_*` sid's lifecycle start is its *trigger*
idx (when the new sub comes into existence); its `starting_idx` is the
earlier probe-validated anchor. The prior sid stays active until the
trigger idx (its `end_idx` = the trigger idx). See §2 (starting_idx vs
lifecycle-start).

### 6.2 Across parent cycles — one df, lifecycle-bounded (REVISED 2026-05-25)

There is **one entity df per (TF, role, parent_path)** for the entire
session; all parent cycles' subs of the same type live in it, with
`(parent_sid, parent_cycle_id, sid)` identity (`sid` resets per parent
cycle). `parent_cycle_id` is meta, not part of entity identity.

When parent cycle K ends (next parent BOS or parent reversal), §6.5
lifecycle propagation ends all of cycle-K's sub sids / cycles / zones at
the parent-cycle-end idx with `deactivated_by="lifecycle_end"`. Parent
cycle K+1's subs start fresh at sid=0. There is **no cross-parent-cycle
overwrite** — the prior design's in-place overwrite of overlapping
territory is removed; the K and K+1 sequences are bounded by the
parent-cycle boundary, not reconciled by capping. (The new K+1
`first_confluence`'s `starting_idx` may physically anchor inside cycle K's
territory — that is the structural anchor only; its lifecycle start is its
trigger idx, after cycle K has ended.)

Consumers query by `(parent_sid, parent_cycle_id)` meta to scope to a
specific parent cycle.

### 6.3 Sub reversal mid-cycle — IS the sid boundary, does not perturb parent cadence (REVISED 2026-05-25)

A sub reversal is the boundary between `sid` and `sid+1` — NOT an internal
multi-structure roll within one run:

- The bounded run for the current sid **stops at its first reversal**
  (§5). `sid+1` is then built via the `reversal` scenario on the sub's own
  TF / data / zones (`identify_start`, flipped direction).
  `starting_alignment` / `starting_sd` (entity-level) remain sticky.
- The reversal **competes with the next `subsequent_*` trigger** to be the
  boundary; whichever idx is earlier starts `sid+1` (§6.1 step 4). If the
  reversal wins, the would-be `subsequent_*` trigger that comes later still
  starts a further sid (it is detected from parent events, independent of
  the sub's internal reversal).
- Parent's variation cadence is unaffected — the parent's proximity state
  machine and event emissions continue regardless.
- WVMI sweep "starting from most recent same-type sub sid's start" uses
  whichever event birthed the *current* sid (`reversal`- or
  `subsequent_*`-induced), per each sid's `started_by`.

### 6.4 Parent reversal — bootstrap rule for new parent cycle

When parent reverses:

- By the `REVERSAL_CONFIRMED` candle, the new parent's sid+1 may already
  have multiple cycles complete (cycle 0, cycle 1, …).
- Subordinates are only built starting from the **active parent cycle**
  (the cycle in progress at `REVERSAL_CONFIRMED`).
- No retroactive building for past cycles of the new parent's sid+1.

### 6.5 Lifecycle propagation

Parent cycle ending (next BOS or reversal) ends **all** active descendants
transitively:

- Direct children of this parent ⇒ end.
- Grandchildren under those children ⇒ end (their immediate parent ended,
  so their `parent_cycle_id` is now stale).
- Each ended entity caps its open-ended zones / POIs / fibs at the
  lifecycle boundary with `deactivated_by="lifecycle_end"`.
- Each ended entity locks any in-progress WVMI with
  `locked_by="lifecycle_end"`.

---

## 7. Persistence Model — What Overwrites vs What Persists

Persistence rules apply *within* an entity's df. Each entity's df is
isolated from every other entity's df.

(REVISED 2026-05-25 — sids are sequential & non-overlapping per §6.1, so
there is no per-candle overwrite *between sids* and no bounds-capping
cascade. Each candle is written once by its owning sid.)

| Data type | Storage | Policy |
|---|---|---|
| df columns (`structure_id`, `cycle_id`, `cts_phase`, `range_lo/hi`, etc.) | per-candle | Written once per candle by the owning sid; sids don't overlap, so no cross-sid overwrite. (A sid's own bounded run still writes its own candles.) |
| Structure events (`StructureEvent` list) | `df.attrs["events"]` (append-only) | **Persist forever** with `(parent_sid, parent_cycle_id, sid)` + cycle attribution; never deleted or mutated |
| Zones (KL, POI), Fibs, Wave candles, Imbalances | `df.attrs[...]` keyed by sid + cycle_id | Persist; `end_idx` resolved via the §5 lifecycle model (next cycle / next structure / parent-cycle-end). No `overwritten_by` tagging, no bounds-capping. |
| WVMI records | `df.attrs["wvmi"]` keyed by sid + cycle_id | Persist as snapshots; locked at the owning sid/cycle's lifecycle end |
| Zone proximity triggers | `df.attrs["zone_proximity_triggers"]` | Persist — drive children via the event bus |

**Rationale:**

- df columns are "current truth" per candle — cheap to overwrite, charts
  only need one view per candle.
- Events are historical truth — what the algorithm believed at the time.
  Replay/debug/audit benefits from full history. Today's append-only
  invariant extends to multi-sid: events from all sids coexist with full
  attribution.
- Zones/POIs/fibs/WVMI are snapshots derived from events. Old snapshots
  are valid history; new snapshots are current. Both keyed for filtering.

**Append-only event extension:** today's invariant ("events never modified
after emission; augmentations replace events in the list") extends
naturally to multi-sid: new sid's events append; old sid's events stay.
The replacement permission (e.g., imbalance-instance refactor) only
applies to *the same event*, never deletes events from other sids.

---

## 8. WVMI Redesign

WVMI now exists per structure entity, keyed by
`(structure_path_id, sid, cycle_id)`.

### 8.1 Removed: ad-hoc gate

- The "first sd zone-proximity trigger gates WVMI creation at parent
  CTS_CONFIRMED" pattern is removed.
- "Cycle passed the gate" and "parent CTS_n confirmed" merge into one
  event because parent CTS_CONFIRMED can now be fired by sd-zone-proximity
  (dual CTS confirmation, Stage 1 + 2 already done).
- `WVMITracker.add_scenario3_record` / `discard_scenario3` — removed.
- `source ∈ {"main", "scenario3"}` keying — replaced by
  `structure_path_id`.

### 8.2 Main WVMI

- Trigger: main's first sd-zone proximity after main CTS (= the candle
  that confirms main CTS via sd-prox method, OR the first sd-prox after a
  pullback-confirmed main CTS).
- One per main cycle.
- Lock: at next main `BOS_CONFIRMED`. Unchanged from today.

### 8.3 Confluence sub WVMI

- Triggered initially by main's first sd-prox after CTS (same trigger as
  main WVMI, same trigger as `first_counter` / var 2).
- Re-triggered each time `subsequent_counter` (var 4) fires.
- At each trigger: initiate a per-cycle WVMI tracker on the **current**
  confluence sub sid (whichever sid is active — created by var 1, var 3,
  or sub internal reversal).
- Tracker is **continuous**: each new cycle within the sid gets its own
  WVMI as it forms; current cycle has temp LP, locks at next sub
  `BOS_CONFIRMED`.
- Past cycles of the sid (already completed by trigger time) are calc'd
  and immediately locked since their next-BOS already exists.
- "Starting from most recent same-type sub's start" = whichever event
  birthed the current confluence sub sid.

### 8.4 Counter sub WVMI

- Triggered each time `subsequent_confluence` (var 3) fires.
- First counter WVMI calc happens at the first var 3 fire (= first parent
  CTS-prox after the parent first sd-prox).
- Otherwise mirrors §8.3 with "counter" substituted.

### 8.5 Cadence summary

| Trigger event | WVMI calcs initiated |
|---|---|
| Main first sd-prox after main CTS | Main WVMI + Confluence sub WVMI sweep on current confluence sid |
| Var 3 (parent CTS-prox after sd-prox) | Counter sub WVMI sweep on current counter sid |
| Var 4 (parent sd-prox forming Λ/V) | Confluence sub WVMI sweep on current confluence sid |

Symmetry: confluence sub WVMI fires on sd-prox-class events; counter sub
WVMI fires on CTS-prox-class events. Each side's WVMI is initiated at the
*opposite* side's trigger moment.

### 8.6 Lock semantics (per-cycle)

| Lock cause | When |
|---|---|
| Normal | Next `BOS_CONFIRMED` of the same entity (mirrors main today) |
| Lifecycle end | Parent cycle ends (transitive — propagates to all descendants) |
| Sub reversal | Sub's own `REVERSAL_CONFIRMED` |
| Same-type overwrite | New sid via subsequent variation overwrites previous sid's open cycle |

Each of these populates `locked_by` meta on the WVMI record.

### 8.7 Meta schema update

- **Drop:** `proximity_trigger_idx` (was the gate).
- **Add:** `triggered_by_event_idx` (the var 1 / var 3 / var 4 / main
  first-sd-prox candle that initiated this sweep) and
  `triggered_by_event_type`.
- **Carry:** `structure_path_id`, `sid`, `cycle_id` for full attribution.

---

## 9. Storage Architecture — Per-Entity DFs + Registry

### 9.0 Terminology

| Term | Meaning |
|---|---|
| **entity** | Unit of structural isolation. Identified by `structure_path_id`. Owns one df + one set of stateful objects (MarketStructure, FibTracker, WVMITracker). Created lazily, persists for the session. |
| **sid** | One run/instance within an entity. For `main`, each reversal increments sid. For subs, each parent_cycle produces a fresh sub-instance whose sids are bounded by that parent cycle. |
| **cycle / cycle_id** | One CTS-to-CTS span within a sid (a sub or main internally manages cycles). |
| **parent_cycle_id** | Per-sid meta on a sub: which parent_cycle this sid was born under. Determines lifecycle bounds and cross-entity zone lookups. |
| **structure_path_id** | The string key identifying an entity (e.g., `H1.main >> M15.counter`). Pure structural path; no sids/cycles in the path itself. |

**Per-entity instantiation:** every entity gets its own
`MarketStructure`, `FibTracker`, `WVMITracker` instance. None of these
are shared across entities. This is the foundation of isolation — sibling
or parent entities cannot accidentally read or mutate each other's state
through a shared singleton.

### 9.1 `structure_path_id` format

Stable string, used as both registry key and chart label root. **Pure
structural path** — no sids, no cycles. Identity is the
(TF, role, parent_path) chain only. Sid numbers and parent_cycle_id are
runtime data inside the entity's df, not part of identity.

```
H1.main
H1.main >> M15.counter
H1.main >> M15.counter >> M5.confluence
```

- `>>` separator (ASCII; chart labels may render as `⇒`).
- One df per path; many sids and many parent cycles share that one df.

### 9.2 `StructureRegistry`

```python
class StructureRegistry:
    _entities: Dict[str, EntityState]  # key = structure_path_id

    def register(path_id: str, ...) -> EntityState  # explicit creation
    def get(path_id: str) -> EntityState | None     # None if not registered
    def parent_of(path_id: str) -> EntityState | None  # None for main
    def children_of(path_id: str) -> List[EntityState]  # [] if no children
    def all() -> Iterable[EntityState]
```

**Error semantics (locked):**
- `get()` returns `None` for unregistered paths — does not auto-create.
- `parent_of()` returns `None` for `main` (top of chain).
- `children_of()` returns `[]` for leaf entities.
- Entities are created **only** via explicit `register()` — typically
  called by the event-bus when a variation fires for the first time.

`EntityState` (entity-level, immutable):
- `df: pd.DataFrame` — TF candles + structure columns + attrs
- `parent_path_id: str | None`
- `starting_alignment: "confluence" | "counter" | None` — sub only
- `timeframe: str`
- `role: "main" | "subordinate"`

Per-sid attribution lives **in the df**, not on `EntityState`:
- `df.attrs["sids"]` — list of sid records, each with `sub_sid`,
  `parent_sid`, `parent_cycle_id`, `starting_sd`, `creation_event_idx`,
  `end_event_idx`, `end_reason`. The identity is the tuple `(parent_sid,
  parent_cycle_id, sub_sid)` (`parent_*` None for main → identity reduces
  to `sub_sid == structure_id`). `sub_sid` resets to 0 each parent cycle;
  each sid's `starting_sd` is derived from parent's current_sd at that
  moment.
- Events / zones / POIs / WVMI all carry the identity tuple
  (`sub_sid + parent_sid + parent_cycle_id`) plus `cycle_id` meta to
  support per-cycle filtering.

### 9.3 Per-entity df contents

| Path | Contents |
|---|---|
| df columns | TF candles (own copy) + `structure_id`, `cycle_id`, `cts_phase`, etc. |
| `df.attrs["structure_path_id"]` | This entity's id |
| `df.attrs["parent_path_id"]` | None for main |
| `df.attrs["sids"]` | Per-sid records: `sub_sid`, `parent_sid`, `parent_cycle_id`, `starting_sd`, `creation_event_idx`, `end_event_idx`, `end_reason` |
| `df.attrs["events"]` | Append-only events list, all sids of this entity, with identity-tuple (`sub_sid + parent_sid + parent_cycle_id`) + `cycle_id` meta |
| `df.attrs["kl_zones"]`, `["poi_zones"]`, `["wave_candles"]`, `["fib_states"]`, `["wvmi"]`, `["imbalances"]` | All entity-local; identity-tuple + cycle keyed (sids sequential & non-overlapping — no cascade) |
| `df.attrs["zone_proximity_triggers"]` | This entity's own triggers (children consume via registry) |

### 9.4 Cross-entity lookups

When a sub needs parent zones (var 1 reference zone, dual CTS
confirmation needs parent BOS+POI inners). The sub's currently-active sid
record carries `parent_sid` and `parent_cycle_id`:

```python
parent = registry.parent_of(self.path_id)
sid_rec = self.current_sid_record()  # from df.attrs["sids"]
parent_bos_zones = [
    z for z in parent.df.attrs["kl_zones"]
    if z.source_kind == "BOS"
       and z.meta["structure_id"] == sid_rec.parent_sid
       and z.meta["cycle_id"] == sid_rec.parent_cycle_id
]
```

No snapshot copying. Parent is the source of truth.

### 9.5 Lifecycle

- **Created lazily** on first variation fire that needs the entity.
- **Persists for the trading session.**
- **Cleared on `trading_open`** — fresh session.
- **Replay/backtest:** registry rebuilds deterministically from main's
  candle data + initial start. Every entity df is a pure function of
  parent events.

### 9.6 Deterministic same-TF feature sharing (deferred optimization)

Two entities on the same TF (e.g., `H1.main >> M15.counter` and
`H1.main >> M15.confluence`) compute identical per-candle features
(candle types, imbalance instances, volume features, etc.) since the
candle data is the same. Each currently maintains its own copy. A future
optimization can share an upstream "TF feature df" by reference. Defer
until proven necessary — isolation correctness wins until performance
forces a change.

---

## 10. Event Routing Layer

When a parent's MarketStructure emits an event that may trigger a child
variation, an event bus dispatches it.

### 10.1 Trigger subscription matrix

| Parent event | Subscriber |
|---|---|
| `BOS_CONFIRMED` | `first_confluence` (var 1) |
| `CTS_CONFIRMED` | resolves var 1's NULL `end_idx` if pending |
| First sd-zone proximity trigger after parent CTS | `first_counter` (var 2) + main WVMI + confluence sub WVMI initial sweep |
| Subsequent sd-zone proximity trigger forming Λ/V | `subsequent_counter` (var 4) + confluence sub WVMI sweep |
| CTS-zone proximity trigger after sd-prox | `subsequent_confluence` (var 3) + counter sub WVMI sweep |
| `REVERSAL_CONFIRMED` | parent cycle ends → all active children lifecycle_end (transitive) |

### 10.2 Within-candle ordering

When multiple subscribers fire on the same candle, the deterministic
order is:

1. Parent emits its event(s) for this candle.
2. Variation triggers evaluate against parent state. Each variation's own
   predicate is self-gating; alternation is enforced by parent's
   proximity state machine.
3. Probes run on parent's TF data (synchronous — they don't span candles
   in backtest; live mode handles pending state per §14).
4. Sub entities created (if first time) or have new sid appended.
5. Sub structures process the candle (their own MarketStructure updates,
   derive zones, fibs, POIs).
6. WVMI sweeps run on whichever entities the trigger initiated for.
7. Children of newly-emitted sub events recursively trigger their own
   grandchildren via the same bus.

This ordering must be honored at every recursion level.

### 10.3 Recursive triggering

Every event carries the emitting entity's `structure_path_id`. The
dispatcher routes only to **direct children** of that path_id — never to
grandchildren or main. Each level treats its immediate parent as the sole
event source.

A 15M counter sub emitting its own `BOS_CONFIRMED` triggers var 1 for
`H1.main >> M15.counter >> M5.confluence` (which lives as a single entity
in the registry; the specific sid created carries `parent_sid` and
`parent_cycle_id` referencing the M15 counter event that birthed it).

---

## 11. Per-Entity Feature Pipelines

Every entity df runs its own deterministic feature pipeline before
structure compute, in this order:

1. **Candle classification** (`features/candles_v2`) — pinbar, maru,
   normal, star, body_pct, is_big_normal_as0, is_big_maru_as0, etc.
2. **Structure patterns** — continuous, double_maru,
   one_maru_continuous, one_maru_opposite.
3. **Imbalance detection** — per-candle `is_imbalance` flag plus
   `df.attrs["imbalances"]` list of `ImbalanceInstance`. Detection requires
   c2 direction match; consecutive same-direction merges with bounds from
   first c1 to last c3.
4. **Volume features** — `vol_dir`, `vol_ema20`, `vol_spike_ratio`,
   `is_vol_spike`.

These are deterministic functions of candle data. The same TF candles
produce identical features regardless of which entity's df owns them.

The lower-TF slicing pattern from today's UC1 (50-candle lookback buffer,
`compute_imbalance` re-run after slicing because attrs indices don't
survive `reset_index`) generalizes to every entity df at every nesting
level.

---

## 12. Replay Determinism & commit-save Under Multi-DF

`commit-save` and `compare` workflows extend to per-entity dfs:

- **commit-save** writes one CSV per entity df + chart per entity df.
- Output folder: `artifacts/commits/{ts_sha}/{path_id_sanitized}/`
  (e.g., `H1.main/`, `H1.main__M15.counter/`,
  `H1.main__M15.counter__M5.confluence/`).
- **compare** walks every entity in both snapshots and diffs row-by-row.
  - New entity in latest run not in baseline → flagged "new entity."
  - Entity removed → flagged "removed."
  - Per-entity column diffs work as today.

**Determinism guarantee:** for the same input candle data and the same
`trading_open` start, the registry rebuilds identically. Every entity
df's contents are a pure function of parent events and candle data. This
is essential for the replay/optimize loop, which deliberately mimics live
execution (no foreknowledge) so the algorithm can learn / improve.

---

## 13. Migration Plan

The refactor is large. Recommended incremental path, with `/compare`
between every step:

1. **Stand up registry alongside today's monolithic df.** Main builds
   into `H1.main` entity. UC1 lower-TF data routed into
   `H1.main >> M15.counter` entity via the existing
   `_run_lower_tf_pipeline` — minimal logic change, just rerouting where
   data lands. Rename UC1's H1 reverse probe to `_run_subordinate_probe`
   (or similar) to match the generalized role. UC1 itself mechanically
   *is* `first_counter`; rename in code, no new logic. Wire the
   TF-keyed `DEFAULT_PROBE_RESET_PIPS` table now (§4.4) since it's small
   and TF-routing makes it relevant.

   **Parity criteria for Step 1:** chart visuals + sid/cycle event counts
   identical to today, even though the underlying df shape differs.
   Per-row CSV parity will not hold once data moves into the registry —
   that's expected.
2. **Switch chart consumer to registry.** Chart reads from registry, not
   `df.attrs`. Old df.attrs path deprecated but kept for parity.
3. **Add confluence subs (var 1 + var 3).** Build confluence sub
   entities. WVMI rewired for confluence sub.
4. **Add subsequent counter (var 4).** Counter sub now has multiple sids
   per parent cycle. WVMI rewired for counter sub.
5. **Remove old monolithic df.attrs and clean up transitional shims.**
   Implementation split into substeps a–e (running plan in
   `memory/project_part4_progress.md`). Substep summaries:

   - **§13.5.a (mechanical field cleanups, landed):** the
     `add_scenario3_record` / `discard_scenario3` / "first sd-prox gate"
     code paths were already removed in 3c; in §13.5.a we drop the
     back-compat `proximity_trigger_idx` write in `proximity_candles`,
     replaced by `triggered_by_event_idx` (§8.7 schema), and remove the
     `WVMIRecord.source = "main" | "scenario3"` field — replaced by
     `structure_path_id` on every record.

   - **§13.5.b (resolve `structure/` → `zones/` import inversion via
     dependency injection):** today's `structure/proximity_helpers.py`
     derives BOS / POI zone inners inline during MarketStructure's
     per-candle dual CTS check. Zones can't be derived earlier in the
     pipeline (they depend on structure events), and copying the
     derivation logic into structure/ is forbidden by LANDMINES (drift
     risk). The cleanup is therefore **resolver injection**, not
     reordering or duplication.

     Concrete cleanup:
     - Delete `structure/proximity_helpers.py`. Relocate its three
       functions:
       - `compute_bos_inner_from_event` → `zones/kl_zones_v1.py` (its
         natural home alongside `identify_base_pattern` /
         `zone_thresholds`)
       - `compute_poi_inners_for_cycle` → `zones/poi_zones.py` (natural
         home alongside Fib + IC scan primitives)
       - `check_sd_proximity_at_candle` → private helper inside
         `structure/market_structure.py` (it has no `zones/` deps — just
         inner prices and a threshold)
     - `MarketStructure.__init__` accepts two callable resolvers:
       - `bos_inner_resolver: Callable[[int, int], Optional[float]]` —
         `(bos_idx, struct_direction) → inner_price`
       - `poi_inners_resolver: Callable[[int, float, int, float, int,
         int, int], List[float]]` — `(bos_idx, bos_price, cts_idx,
         cts_price, sd, sid, cycle_id) → list of POI inner prices`
     - The orchestrator (`compute_structure` in `structure_engine.py`)
       constructs MarketStructure with closures that bind to the
       relocated `zones/` functions. This is the only place the resolver
       wiring lives.
     - After §13.5.b, `structure/market_structure.py` has zero imports
       from `zones/`. The structure→zones inversion is gone.
     - LANDMINES "Backward Dependency: structure/ → zones/
       (transitional)" entry replaced with: "MarketStructure must not
       import from `zones/` directly; consume zone derivations via the
       resolver protocol passed at construction."

   - **§13.5.c (in-place overwrite infrastructure, §6.1 / §6.2):**

     > **⚠ SUPERSEDED 2026-05-25 — REPLACED IN CODE by redesign Phase 2.**
     > This substep implemented the cascade-overwrite model (entity-wide
     > `entity_sid` + `_tag_old_sid_on_overwrite` + per-trigger
     > `apply_trigger_to_entity_df`). Redesign Phase 2 (DONE 2026-05-25)
     > replaced it with the merge-and-bound sequential sid build
     > (`build_parent_cycle_chain` + `build_one_sid`; no overwrite,
     > per-parent-cycle sid). `apply_trigger_to_entity_df` is **removed**;
     > the cascade (`_tag_old_sid_on_overwrite` + `cascade_*` fields) is
     > **dead** (never fires — `prior_sid_id=None` always) and slated for
     > deletion in redesign Phase 4. The c.i/c.ii/c.iii prose below is
     > retained for HISTORICAL context only — it no longer describes the
     > code. See `memory/project_sub_structure_lifecycle_redesign.md`.

     entity-df mutation when a new sid overwrites an older one in
     `[starting_idx, current_candle]`; old sid's zones / POIs / fibs /
     WVMI tagged `deactivated_by="overwritten_by_sid_{n+1}"` with
     bounds capped at the overwrite boundary; old sid's open WVMI
     locked with same reason.

     **Mechanism (decided 2026-05-05): mutate the entity df in place.**
     Each trigger goes through a new entry point
     `apply_trigger_to_entity_df(entity_df, trigger, parent_df,
     new_sid_id)` which:

     1. Runs the parent-TF probe (unchanged from today —
        `compute_structure_scenario_3` on `parent_df`).
     2. Maps validated parent idx to entity-df ABSOLUTE idx (no
        slicing, no `reset_index` — the entity df has full lookback
        by construction).
     3. Calls `_tag_old_sid_on_overwrite(entity_df, prior_sid_id,
        new_sid_id, boundary_idx)` which iterates
        `entity_df.attrs["kl_zones"] / ["poi_zones"] / ["fib_states"]`
        and sets `meta["deactivated_by"] = "overwritten_by_sid_{N}"`
        on every prior-sid snapshot, capping `end_time` at the
        overwrite boundary for any still-open snapshots; iterates
        `entity_df.attrs["wvmi"]` and sets `lp_locked=True` +
        `meta["lp_locked_by"] = "overwritten_by_sid_{N}"` on any
        prior-sid record where `lp_locked` was False.
     4. Calls `compute_structure_from_start(entity_df,
        start_idx=mapped_M15_idx, struct_direction=trigger.lower_sd,
        timeframe=trigger.lower_tf)` — this overwrites df columns
        (`structure_id`, `cycle_id`, `cts_phase`, `range_lo/hi`, etc.)
        from `start_idx` forward and emits new events. The events get
        `sid=new_sid_id` attribution and append to
        `entity_df.attrs["events"]`.
     5. Runs the downstream pipeline (`_run_downstream_pipeline` with
        `source_kinds=["BOS"]`, `fib_mode="cross_cycle"`,
        `skip_wvmi=True`) on the new sid's events, appending the
        resulting kl_zones / poi_zones / fib_states / wave_candles to
        `entity_df.attrs[...]` keyed by `sid + cycle_id` meta.
     6. Computes parent-driven sub WVMI for the new sid via
        `compute_parent_driven_sub_wvmi` — adapted to consume an
        entity-df + sid filter rather than a `LowerTFResult` (or
        wrapped via the §13.5.c.i facade helper).

     The alternative considered (build fresh `LowerTFResult` per
     trigger, post-merge into the entity df) was rejected: it
     requires translating every event / zone / POI / fib idx from
     slice-local to entity-absolute coordinates at merge time,
     retains today's slicing + 50-candle lookback + `reset_index` +
     per-trigger `compute_imbalance` machinery as workarounds, and is
     awkward in live mode where each new candle's pending result
     would need merging on arrival. Mutation in place lines up with
     §7's "df columns are current truth" model and with the
     live-mode shape described in §14.

     Substep split (in order):

     - **§13.5.c.i — mirror infrastructure + production pilot (one
       trigger).** Land `multitf/entity_df_mutation.py` with:

         * `mirror_lower_tf_result_to_entity_df(entity_df, result,
           new_sid_id, prior_sid_id=None)` — takes a freshly-built
           `LowerTFResult` (slice-shape, slice-local idx) and
           translates its events / kl_zones / poi_zones / fib_states /
           wave_candles to entity-absolute idx, appending them to
           `entity_df.attrs["events"] / ["kl_zones"] / ["poi_zones"] /
           ["fib_states"] / ["wave_candles"]` with `entity_sid =
           new_sid_id` attribution. Also mirrors structure columns
           (`structure_id`, `cycle_id`, `cts_phase`, `range_lo/hi`,
           `market_state`) from `result.df` back to
           `entity_df.iloc[slice_begin:slice_end+1, ...]`.

         * `_tag_old_sid_on_overwrite(entity_df, prior_sid_id,
           new_sid_id, boundary_idx)` — the cascade helper. Iterates
           `entity_df.attrs[...]` and tags prior-sid snapshots with
           `deactivated_by="overwritten_by_sid_{N}"`, caps bounds,
           locks open WVMI records. No-op when `prior_sid_id` is None.

       Wire ONE pilot trigger through the new path — recommended
       pilot: var 4 at `(parent_sid=0, parent_cycle_id=3)`
       (non-degenerate window `input_idx=703 / end_idx=705`). The
       pilot still calls `run_lower_tf_pipeline` to build the
       slice-shape `LowerTFResult` (chart compatibility preserved —
       no facade needed in c.i), THEN calls
       `mirror_lower_tf_result_to_entity_df` to populate
       `entity_df.attrs[...]`. The original `LowerTFResult` is
       appended to `lower_tf_results` exactly as before so the chart
       renders identically. All other triggers keep the old path
       (no mirror call).

       Cascade does NOT fire in c.i production (pilot is the only
       trigger writing to `entity_df.attrs`; no prior sid to
       overwrite). Cascade is exercised by a unit test on a synthetic
       entity df fixture with two sids of fabricated state.

       **Why this scope (not entity-direct compute):** running
       `compute_structure_from_start` directly on the entity df with
       `end_idx` capping is the right end state, but it requires
       (a) modifying `compute_structure_from_start` to accept
       `end_idx`, (b) translating events/zones the OTHER direction
       for chart compatibility, and (c) verifying that downstream
       derivations behave the same on the entity df as they do on a
       slice. Each of those is a real risk; bundling them with the
       mirror semantics in one substep makes parity debugging
       harder. c.i validates the persistence model end-to-end
       (entity_df.attrs holds the pilot's data with correct
       attribution + cascade-ready shape) using minimal-risk
       infrastructure; c.ii commits to entity-direct compute.

     - **§13.5.c.ii — entity-direct compute + thread through full
       var 1 + var 2 + var 3 + var 4 sets on both entities.** Add
       `end_idx` parameter to `compute_structure_from_start`. Replace
       `run_lower_tf_pipeline + mirror` with a single
       `apply_trigger_to_entity_df` that runs
       `compute_structure_from_start(entity_df, start_idx,
       end_idx=lifecycle_end_idx)` directly on the entity df —
       eliminating the slice + 50-candle lookback + `reset_index` +
       per-trigger `compute_imbalance` machinery. Sort all triggers
       per entity by `trigger_event_idx` and apply sequentially.
       Real cascade fires in production (var 4 over var 2 in same
       parent cycle, var 3 over var 1, cross-cycle overwrites from
       §6.2). Build `LowerTFResult` facades with entity→slice idx
       translation for chart compatibility. Carve-outs still in
       place. `run_lower_tf_pipeline` deleted at end of c.ii.

     - **§13.5.c.iii — chart consumer migrates to entity-df reading
       (LANDED).** `export_m15_chart_plotly` reads
       `m15_df.attrs["events" / "kl_zones" / "poi_zones" / "fib_states"
       / "wave_candles" / "wvmi" / "prev_bos_lines"]` directly,
       iterating `m15_df.attrs["sids"]` SidRecords and grouping the
       attrs lists by `meta["entity_sid"]`. Two new helpers
       (`_compute_m15_tier_context_from_sids`, `_compute_owner_by_idx`)
       drive the §16.5 most-recent-sid filter — sid-tied elements
       (CTS/BOS dots, swing lines, PB markers, wave-candle lines,
       prev_bos lines) check `owner_by_idx[rendered_candle_idx] ==
       this_entity_sid` per render site, and the swing-line extension
       at the most-recent internal sid stops at the last candle still
       owned by this entity_sid (walking back from
       `sid_rec.end_event_idx`). Persisting events (KL/POI zones)
       opacity-attenuate by `meta["deactivated_by"]`:
       `overwritten_by_sid_{N}` → `prior_inactive` tier;
       `lifecycle_end` falls back to today's parent-sid/cycle tiering.
       The H1 chart's M15-zone overlay (`export_chart_plotly`) was
       migrated in the same substep — it now reads zones from
       `registry.get(f"{path_id} >> M15.counter").df.attrs["kl_zones"]`
       instead of `dfx.attrs["lower_tf_results"]`. Orchestrator's
       three `lower_tf_results` attr writes (M15.counter,
       M15.confluence, and the deprecated H1) are removed; the local
       `lower_tf_results` list inside `_run_multi_tf` /
       `_run_first_confluence_multi_tf` stays — it still feeds
       `build_sid_records_for_subordinate` and the parent-driven
       sub-WVMI helpers. Dead facade builders
       (`_build_facade_lower_tf_result`, `_FACADE_LOOKBACK`,
       `_shift_zone` / `_shift_event` / `_shift_poi` / `_shift_fib` /
       `_shift_wave_candle`) deleted from `multitf/entity_df_mutation.py`.
       `mirror_lower_tf_result_to_entity_df` extended to translate +
       persist `prev_bos_lines` (entity-absolute idx,
       entity_sid-attributed) so the M15 chart can read them from
       `m15_df.attrs["prev_bos_lines"]`.

       > **⚠ SUPERSEDED (chart key-swap + §13.5.e, post-redesign).** The
       > c.iii body above describes the pre-redesign state. Two later
       > changes apply:
       > 1. **`entity_sid` removed.** Grouping + `owner_by_idx` + all
       >    snapshot attribution now key on the identity tuple
       >    `(parent_sid, parent_cycle_id, sub_sid)` — there is no
       >    `entity_sid` / `meta["entity_sid"]`. `sub_sid` is the per-parent
       >    -cycle counter; the tuple is the identity.
       > 2. **`deactivated_by` opacity gone.** The cascade was deleted
       >    (redesign Phase 4); zone opacity is now a per-TF tier
       >    (`_m15_opacity_tier_for_zone`), not a `overwritten_by_sid_{N}`
       >    /`prior_inactive` lookup.
       > 3. **§13.5.e chart-fallback portion DONE (both charts).**
       >    `export_m15_chart_plotly` and `export_chart_plotly` are both
       >    registry-only — the positional `m15_df`/`h1_df`/`df` fallbacks
       >    were removed. The OTHER §13.5.e item (delete the orchestrator's
       >    deprecated `s_res.df.attrs[...]` writes) is NOT done — the H1
       >    chart still reads overlays from `dfx.attrs[...]`.

     **Sid numbering convention (clarifies §6.1 below):** sids are
     **entity-wide** monotonically increasing integers, NOT
     per-parent-cycle. Each new trigger increments the entity's sid
     counter regardless of whether it opens a fresh parent cycle or
     overwrites an open sid in the current parent cycle. `parent_sid`
     and `parent_cycle_id` live in each `SidRecord`'s meta. Today's
     `build_sid_records_for_subordinate` already enumerates
     entity-wide; this is documentation of the existing convention,
     not a behavior change.

   - **§13.5.d (remove var 3 + var 4 last-per-cycle carve-outs):** drop
     `var3_last_per_cycle` filter in `_run_first_confluence_multi_tf`
     and `var4_last_per_cycle` filter in `_run_multi_tf`. Build all
     detected triggers; rely on §13.5.c overwrite-in-place semantics to
     deactivate older sids. Paired removal — neither carve-out can be
     removed without the other (per LANDMINES).

     > **⚠ SUBSUMED 2026-05-25.** The revised §6.1 merge-and-bound build
     > already builds every `subsequent_*` trigger as a sequential sid
     > (bounded by the next), so the last-per-cycle carve-outs disappear
     > as a side effect — there is nothing left to "remove" once the new
     > build lands, and the deactivation no longer relies on §13.5.c
     > overwrite semantics (there is no overwrite). Fold this into the
     > merge-and-bound implementation phase rather than treating it as a
     > separate step. See `memory/project_sub_structure_lifecycle_redesign.md`.

   - **§13.5.e (remove transitional `df.attrs` writes + chart positional
     fallback):** delete the DEPRECATED block in
     `pipeline/orchestrator.py:run_pipeline` that writes
     `s_res.df.attrs[...]` after the registry is set up; delete the
     positional-df fallback in chart entry points
     (`export_chart_plotly` / `export_m15_chart_plotly`) that exists
     for ad-hoc inspection scripts. After §13.5.e, the registry is the
     only routing mechanism for chart data; per-row CSV parity may
     genuinely shift (the registry has been a routing veneer over the
     same df object until this step). Acceptance criterion: visual
     chart parity + event-count match (Step 1 baseline standard), not
     byte-identical CSVs.
6. **Recursive depth.** 5M subs under 15M subs. Validate event routing
   handles nested recursion cleanly.
7. **Delete `PRE_REFACTOR_INVARIANTS.md`.** Refactor complete; merge spec
   sections into canonical `MARKET_STRUCTURE_SPEC.md` /
   `MULTI_TF_SPEC.md`.

---

## 14. Replay / Live Mode Pending State

**Scope:** live trading infrastructure is **not built** in Part 4. But
the architecture (probes, waits, events, pending state, append-only
events) must be designed so that a future live mode is a thin layer over
the same machinery. Backtesting in this engine is intentionally
"live-shaped" — traversing historical candles one at a time and making
decisions as if the future were unknown — so backtest and live differ
mostly in their data source, not their decision logic.

Concrete implications:

- `first_confluence` with NULL `end_idx` must be modeled as genuinely
  pending — never silently using future data.
- Pending subs are not visible (zones, charts, downstream consumers)
  until `end_idx` resolves. Per §16.6: hidden entirely.
- Once resolved, the sub becomes visible at the resolving candle's time,
  retroactively across the resolved range.
- Any logic that would "peek ahead" in backtest is a bug — it would not
  work in live mode and corrupts the learn/optimize loop.

---

## 15. Open Items

Most items raised during early sessions are now closed (covered in §§5–14
and §16).

Remaining open work:

- **Probe-iteration safety per TF** — `max_probe_iterations = 10` is
  calibrated for H1. Lower-TF probes (M15 parent of M5 sub, etc.) may need
  different caps based on candle count per cycle. Revisit when wiring up
  the TF-keyed thresholds (see §4.4).
- **Validation strategy** — how to systematically validate per-entity
  isolation, registry correctness, event-bus ordering, and recursive
  triggering. Likely needs new test infrastructure since today's tests
  assume single global df.

### Closed since early drafts

- sid termination semantics — §5, §6.1
- lifecycle-end propagation — §6.5, §8.6
- multiplicity rules — §5 (sid increment matrix)
- cross-TF event / zone storage — §9
- mapping_sd rule — §4.3.1 (`mapping_sd = -sub_sd`)
- chart layout, overlay, toggle, zone labels, pending display — §16
- live-mode pending-state UX — §16.6 (hide entirely)

---

## 16. Charting

### 16.1 One chart per entity

Each registry entity gets its own chart. Lazy: only entities that ever
fired produce a chart.

For the user's planned strategy (1H / 15M / 5M, no nesting deeper than
depth 1), this is up to four charts:

- `H1.main`
- `H1.main >> M15.confluence`
- `H1.main >> M15.counter`
- `H1.main >> M5.counter`

The architecture supports deeper nesting (e.g., `H1.main >> M15.counter >>
M5.confluence`). v1 implementation does not need to exercise this; add
when actually used.

### 16.2 Layout — same family as today's M15-with-H1-overlay chart

Each sub chart renders:

- Sub TF candles (own TF, no parent candles ever overlaid)
- Sub structure elements: KL/POI zones, BOS/CTS markers, range bounds,
  wave candle lines, fibs, WVMI hover overlays, connector lines
- Sub TF volume (own TF)
- Optional **parent overlay** (see §16.3)

Main chart (`H1.main`) has no overlay — it's the highest TF in its chain.

### 16.3 Parent overlay

Element set (matches today's H1-overlay-on-M15):

- Parent KL zones (color fill, no border)
- Parent POI zones (color fill, no border)
- Parent BOS / CTS markers (black)
- Parent connector lines (solid, black)
- Parent wave candle lines (solid)
- Parent WVMI (carried via parent wave-candle hover overlay — automatic
  once parent wave candle lines are rendered)

Explicitly **not** in the overlay:

- Parent candles
- Parent volume
- Parent's own deactivated/overwritten history

Recursion rule (relevant only at depth 2+): each chart overlays only its
**immediate parent**, never grandparents.

### 16.4 Toggle for parent overlay

A single legend group titled `Parent (<parent_path_id>)` contains every
parent overlay element. Plotly's native legend-group toggle controls all
of them at once. Default: ON.

No additional toggles in v1. Sid-tied display (see §16.5) and pending-sub
display (§16.6) are governed by hardcoded rules, not user toggles.

### 16.5 Persistence and display rules

> **REVISED 2026-05-25 — terminology only, rules unchanged.** Under the
> revised §6 model sids are sequential & non-overlapping, so there is no
> "overwrite"; "older sid hidden / overwritten" becomes "older sid
> lifecycle-ended." The display rules below still hold as written. Two
> confirmed points (this redesign session): (a) sid-tied **structure**
> elements keep the existing `owner_by_idx` / §16.5 behavior and render at
> their structural anchors (`starting_idx`); (b) **zone** rectangles draw
> from their base/IC anchor but fill only over the active window — no
> charting change needed for zones.

| Element class | Display rule |
|---|---|
| Constants — candle patterns, OHLC, candle types | Always shown; never affected by sid changes |
| Sid-tied — CTS dots, BOS markers, range bounds, market_state regions | **Most recent sid only** per candle (via `owner_by_idx`). Older sids' data persists in df with lifecycle end meta but is hidden by default |
| Persisting events — KL zones, POI zones, imbalances | All rendered. Inactive / lifecycle-ended ones use the existing opacity attenuation logic (older = more transparent); fills gated to the active window |
| WVMI | Locked records show final values via hover; in-progress records re-render whenever `update_temporary_lp` shifts the temp LP |

### 16.6 Pending subordinate display

Per §14, subs with NULL `end_idx` (currently only `first_confluence` while
parent CTS is unconfirmed) are in **pending** state.

- Hide entirely. No chart elements produced.
- When `end_idx` resolves, the sub's data appears at the resolving
  candle's time, retroactively visible across the resolved range.

### 16.7 Zone text labels

Every zone (parent and sub, KL and POI) carries a small text label:

- Format: `<TF> <ZoneType> <Direction>` →
  `1H POI Buy`, `15M KL Sell`, `1H BOS KL Buy`, `15M CTS KL Sell`
- Position: top-right corner of the zone, just above the upper bound

**Stacking rule** for overlap (parent and sub zones in similar price
range): parent zone labels positioned slightly higher than sub zone
labels — stable two-row stacking. Predictable and simple.

### 16.8 Style registry

`style_registry.py` keys by `(element, TF)`. Confluence and counter share
styles within a TF; the chart title (and the path label per §16.9)
differentiates them. This keeps the registry tight and avoids
combinatorial growth as more roles are added.

### 16.9 Chart title and labels

- **Title:** full `structure_path_id` (e.g., `H1.main >> M15.counter`).
  Wrapped if the chart is narrow.
- **Subtitle (optional, human-friendly):** one-line description, e.g.,
  *"Counter sub of H1 main"*.
- **Future enhancement (deferred):** breadcrumb header with cross-chart
  navigation. Not in v1.

### 16.10 File naming and on-disk layout

```
artifacts/charts/{commit_or_session}/
    H1.main.html
    H1.main__M15.confluence.html
    H1.main__M15.counter.html
    H1.main__M5.counter.html
```

Sanitization rule: replace `>>` with `__` (double underscore); strip any
non-filesystem-safe character. Original `structure_path_id` lives in
`df.attrs["structure_path_id"]` and the chart title.

### 16.11 commit-save / compare integration

- `commit-save` copies all entity charts (and per-entity CSVs) into the
  timestamped commit folder, mirroring the §12 layout.
- `compare` walks both snapshots' chart sets:
  - Common entities → diff their data CSVs row-by-row (today's mechanism
    per chart)
  - Entities only in latest → flagged "new entity"
  - Entities only in baseline → flagged "removed entity"

### 16.12 Bootstrap (no annotation needed)

When `trading_open` lands main mid-parent-cycle, the in-progress first
parent cycle has no first_confluence (per §4.3.6). On the confluence sub
chart, that cycle's range simply has no sub-specific elements — only the
sub TF candles and (if toggled) the parent overlay.

This is self-explanatory once the bootstrap rule is understood. No
special chart annotation.

---

**This file is transitional.** Once the refactor lands and behavior is
stable, sections will be merged into:
- `MARKET_STRUCTURE_SPEC.md` — start-scenario logic, compute routing
- New `MULTI_TF_SPEC.md` — identity model, recursion, cadence, registry,
  event routing
- New `CHARTING_SPEC.md` updates — per-entity chart family, overlay
  toggle, zone labels
- Updated `LANDMINES.md` / `GOTCHAS.md` — pending-state rules,
  `mapping_sd` rule, probe-walks-forward direction

`PRE_REFACTOR_INVARIANTS.md` will be deleted at the same time.
