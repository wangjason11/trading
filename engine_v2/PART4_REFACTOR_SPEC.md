# Part 4 Refactor Spec — Multi-TF Structure Hierarchy

> **Status:** Specs complete; entering build phase. All sections (1–16)
> locked across sessions 2026-04-29 / 04-30 / 05-01 / 05-04.
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

Used to find `starting_idx` for new subordinate structures. Probe
**always runs on the immediate parent's TF** and produces a parent-TF
`starting_idx`, which is then mapped down to the subordinate's TF.

#### 4.3.1 Shared probe mechanics

The probe is the existing Scenario 3 BOS_0 iterative probe (today's
`compute_structure_scenario_3`). Behavior:

- Walks **forward in time monotonically**. `input_idx` is the *initial*
  starting_idx candidate. The validated `starting_idx` is `≥ input_idx`
  (never earlier).
- If the probe never resets, `starting_idx == input_idx`.
- Iteration: from `current_start`, run a `MarketStructure` forward
  bounded by `end_idx`. Find first two `CTS_ESTABLISHED` events for sid=0.
  Search candles in `(CTS_EST[0]+1, CTS_EST[1])` for one that returned
  to BOS_0's zone within tolerance. If found, that becomes the new
  `current_start`; restart probe.
- Terminal conditions:
  - **Reversal before 2nd CTS_EST** → finalized
  - **No exception found** (Condition 1) → finalized
  - **Data exhausted with `end_idx` defined** (Condition 4a) → finalized
  - **Data exhausted with `end_idx = NULL`** (Condition 4b) → pending;
    caller may re-run when more data arrives

**Per-variation TF/sd reminders:**

| Variation | Probe sd | Probe TF |
|---|---|---|
| `first_confluence` | `parent_sd` | parent's TF |
| `first_counter` | `-parent_sd` | parent's TF |
| `subsequent_confluence` | `parent_sd` | parent's TF |
| `subsequent_counter` | `-parent_sd` | parent's TF |

After the probe, the parent-TF `starting_idx` is mapped down to the
subordinate's TF using:

> `mapping_sd = -sub_sd`

Where `sub_sd` is the subordinate's `starting_sd`. This rule is
direction-of-the-sub-being-built and does not require knowing parent_sd
directly. (`map_candle_to_lower_tf` in `data_bridge.py` already
implements the "highest high" / "lowest low" selection given
`mapping_sd`.)

> **Refactor note:** today's UC1 sets `mapping_sd = parent_sd` (which
> happens to equal `-sub_sd` for counter subs). Change to
> `mapping_sd = -sub_sd` for the unified rule.

#### 4.3.2 `first_confluence` — first sub of a parent cycle, sd = parent_sd

Find `starting_idx` for the **first** confluence subordinate of a parent
cycle (sid=0 of the confluence entity).

| Field | Value |
|---|---|
| **Trigger** | Most recent parent BOS confirmed (== CTS established for the new cycle, by definition same candle) |
| **Idx input** | Idx of the newly confirmed BOS |
| **Probe end_idx** | Idx where parent CTS_CONFIRMED occurs in the same parent cycle |
| **Probe reference zone** | Newly confirmed active parent BOS zone |
| **Output** | `starting_idx` for first confluence sub (sid=0) |

**NULL end_idx:** This is the **only** variation where end_idx may
initially be NULL. We **wait** — no first_confluence sub is built until
parent CTS_CONFIRMED resolves end_idx (consistent with today's
"pending" handling, but the result is "do not produce" rather than
"produce tentatively").

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
| **Idx input** | Most recent parent CTS idx (== CTS of the active CTS zone) |
| **Probe end_idx** | First parent sd-zone proximity trigger candle after CTS |
| **Probe reference zone** | Active parent CTS zone |
| **Output** | `starting_idx` for first counter sub (sid=0) |

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
| **Idx input** | Parent-TF candle closest to parent BOS, within window `[prior_parent_sd_proximity_trigger_idx (inclusive), current_parent_CTS_proximity_trigger_idx]`. |
| **Probe end_idx** | Current parent CTS-zone proximity trigger candle |
| **Probe reference zone** | Counter-sub zone attached to the lower-TF candle that hits the extreme inside the parent input_idx candle's lower-TF window. See "Reference zone resolution" below. |
| **Output** | `starting_idx` for next confluence sid |

**Reference zone resolution (subsequent_confluence-specific):**
1. Take the parent-TF `input_idx` candle from the rule above.
2. Drill into its lower-TF window (the lower-TF candles with timestamps
   inside the parent candle's hour/period).
3. Find the lower-TF candle that hits the extreme **toward parent BOS**
   (lowest low for bullish parent, highest high for bearish parent).
4. The reference zone is the counter-sub zone that this lower-TF candle
   **anchors** — i.e., the candle is the BOS or CTS extreme of a counter-sub
   zone (not merely contained in one).
5. **Fallback** (in theory should never trigger): if the lower-TF extreme
   candle is not anchoring any counter-sub zone:
   - First fallback: parent sd zone (POI or BOS) closest to parent BOS,
     whose inner bound is within 20 pips of the parent-TF input_idx
     extreme.
   - Final fallback: parent sd zone (POI or BOS) closest to the parent-TF
     input_idx candle.

**Why the input rule is not "CTS of most recent counter sub":** if the
counter sub has reversed before this trigger fires, the candle of interest
on the counter sub may now be a BOS rather than a CTS. The window-based
extreme rule handles both cases automatically.

**Why the probe runs on parent TF, not lower TF:** consistency with the
other variations and avoids an extra mapping step. The reference zone is
allowed to come from a different TF because the probe only consumes its
price bounds.

#### 4.3.5 `subsequent_counter` — new counter sid within same parent cycle

After a counter sub exists in this parent cycle, each time the trigger
conditions fire, end the previous counter sid and create
`counter_sid_n+1` with this variation.

| Field | Value |
|---|---|
| **Trigger** | Parent sd-zone proximity trigger AND most recent prior parent proximity trigger was to CTS zone AND the proximity trigger before *that* was to sd zones (forms Λ in bullish parent / V in bearish parent) |
| **Idx input** | Parent-TF candle closest to active parent CTS zone outer bound, between the two parent sd-proximity trigger candles |
| **Probe end_idx** | Current parent sd-zone proximity trigger candle |
| **Probe reference zone** | Active parent CTS zone |
| **Output** | `starting_idx` for next counter sid |

**Lambda / V geometry (canonical names):**
- **Bullish parent** (parent_sd = +1): BOS zone bottom, POI middle, CTS
  zone top. Three trigger sequence sd → CTS → sd traces a **Λ** (lambda)
  with apex at parent CTS zone.
- **Bearish parent** (parent_sd = -1): mirrors — BOS top, POI middle,
  CTS bottom. Three trigger sequence sd → CTS → sd traces a **V** with
  trough at parent CTS zone.
- `idx_input` = the candle furthest into the parent CTS zone (apex of
  the Λ in bullish, trough of the V in bearish) between the two parent
  sd-proximity trigger candles.

Reference image: `artifacts/bullish_parent_lambda_proximity.jpeg`.

#### 4.3.6 Cadence of triggers within a parent cycle

By construction, the four variations naturally trigger in this order
within each parent cycle:

```
parent BOS_confirmed
    → first_confluence              (input: BOS idx, end: parent CTS confirmed)
parent 1st sd-proximity (post-CTS)
    → first_counter                 (input: CTS idx, end: this trigger candle)
parent CTS-proximity (after sd-prox)
    → subsequent_confluence         (input: window extreme, end: this trigger candle)
parent sd-proximity (after CTS-prox after sd-prox = Λ/V apex at CTS)
    → subsequent_counter            (input: Λ/V apex candle, end: this trigger candle)
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
| `first_confluence` | Parent BOS_confirmed | Parent BOS extreme | Parent CTS_confirmed (NULL until set) | Active parent BOS zone | `+parent_sd` |
| `first_counter` | 1st parent sd-proximity post-CTS | Parent CTS idx | This trigger candle | Active parent CTS zone | `-parent_sd` |
| `subsequent_confluence` | Parent CTS-proximity after sd-prox | Parent-TF window extreme toward BOS | This trigger candle | Counter-sub zone at lower-TF extreme (with fallback) | `+parent_sd` |
| `subsequent_counter` | Parent sd-prox forming Λ / V | Parent-TF Λ apex / V trough at CTS zone | This trigger candle | Active parent CTS zone | `-parent_sd` |

---

## 4.4 Probe Reset Thresholds (applies to `reversal` and `subordinate`)

Both probe-using scenarios depend on a "did price return to the BOS_0 / last
CTS zone within X pips" check. That X is the **probe reset threshold**. It
must always be **strictly less than** the **zone proximity trigger threshold**
on the same TF, otherwise probe-reset and proximity-trigger semantics overlap
and `end_idx` / `starting_idx` can invert.

### Target values (Part 4)

`reversal` and `subordinate` probes share a single TF-keyed table.

| TF | Probe reset (target) | Proximity trigger | Margin |
|---|---|---|---|
| H1 | **10** | 20 | 2× ✓ |
| M15 | **5** | 10 | 2× ✓ |
| M5 | **3** | 5 | ~1.67× ✓ |

Updated from earlier spec values (H1=10, M15=3, M5=1) on 2026-04-30 — M15 and
M5 widened so the probe reset margin is more uniform across TFs while still
strictly less than the proximity trigger.

### Today's state (pre-refactor)

- **`reversal` Exception 2** in `compute_structure` (`structure_engine.py:122`) —
  hardcoded `10 * pip_size`, **not TF-aware**.
- **`reversal` Exception 2** when reached via Phase 2 of
  `compute_structure_scenario_3` — inherits caller's `pip_tolerance_pips`
  (Phase 2 not used in production today).
- **`subordinate`** (today's Scenario 3 / H1 reverse probe) —
  `pip_tolerance_pips: int = 10` default in `compute_structure_scenario_3`;
  H1 reverse probe in `lower_tf_pipeline.py:102` passes `10` explicitly.
- **Proximity trigger** — `DEFAULT_PROXIMITY_PIPS` table in
  `zones/zone_proximity.py:32`. TF-aware.

The new M15 / M5 values exist only in spec prose for now; code has no per-TF
lookup. Since today's only probe path is on H1 data (where 10 pips is
already correct), no current behavior changes — but Part 4 introduces probes
on lower-TF parents (e.g., M15 parent of an M5 sub), which forces the issue.

### Refactor changes

1. Add `DEFAULT_PROBE_RESET_PIPS = {"H1": 10, "M15": 5, "M5": 3}` next to
   `DEFAULT_PROXIMITY_PIPS` (same module, or a shared
   `zones/thresholds.py`).
2. Make every probe-using scenario consult that table by TF instead of
   hardcoding 10:
   - `compute_structure_scenario_3` — replace `pip_tolerance_pips: int = 10`
     default with a TF lookup (default `None`, look up by TF).
   - `compute_structure` reversal Exception 2 — replace hardcoded
     `10 * pip_size` with the TF lookup.
3. Add a startup invariant assertion that for every supported TF:

   ```python
   assert DEFAULT_PROXIMITY_PIPS[tf] > DEFAULT_PROBE_RESET_PIPS[tf], (
       f"{tf}: probe reset {DEFAULT_PROBE_RESET_PIPS[tf]} pips must be "
       f"strictly less than proximity trigger {DEFAULT_PROXIMITY_PIPS[tf]} pips"
   )
   ```

   so future tweaks to either table can't silently invert the relationship.

### Why the constraint matters

If probe reset ≥ proximity trigger on the same TF, the probe could detect a
"return to BOS_0 zone" at a pips distance that the zone-proximity-trigger
pipeline already counted as a real proximity event. The probe would then
push `starting_idx` forward into territory the variation logic considers
post-trigger — making the variation's `end_idx` and `starting_idx` overlap
or invert. Strict `trigger > reset` is the cleanest invariant.

---

## 5. Computing Structures — Function Routing per Role

For `main` (highest TF), continue using `compute_structure`:

- Initial start: `trading_open` scenario (today's `identify_start_scenario_1`).
- On reversal: `reversal` scenario (today's `identify_start_scenario_2_after_reversal` + Exception 1/2).
- Loops until end of data.

For `subordinate` (any role/alignment, any TF), use
`compute_structure_from_start`:

- Start is pre-validated by the upstream variation probe — no internal
  Scenario 1 logic.
- On internal reversal: same `reversal` scenario flow as main.

The legacy `compute_structure_scenario_3` ad-hoc path is **removed**. Its
only purpose (creating WVMI when parent was in range) is now subsumed by
per-subordinate WVMI — see §8.

### sid increment rules

| Role | sid +1 triggers |
|---|---|
| `main` | reversal only |
| `subordinate` | reversal **OR** subsequent variation (var 3 / var 4) |

Each sid is a conceptually independent compute_structure_from_start run
within an entity. Sids do not share live state — only the entity's df,
events list, and zone collections (which they overwrite or append to).

---

## 6. Subordinate Cadence and Overwrite Rules

(§4.3.6 covers the within-parent-cycle cadence diagram. This section
specifies overwrite semantics.)

### 6.1 Within parent cycle, same-type overwrite

When a subsequent variation fires (var 3 → confluence, var 4 → counter),
or when the sub itself reverses, a new sub sid begins at the new
`starting_idx`. In the same entity df:

- df columns on candles `[starting_idx, current_candle]` are **overwritten
  in place** by the new sid's compute_structure_from_start run.
- Old sid's events stay in `df.attrs["events"]` (append-only) — never
  removed; their attribution (sid + cycle) preserves history.
- Old sid's zones / POIs / fibs / WVMI persist in their respective
  `df.attrs[...]` lists with `deactivated_by="overwritten_by_sid_{n+1}"`
  meta and bounds capped at the overwrite boundary.
- Old sid's still-open WVMI (in-progress current cycle) locked with
  `locked_by="overwritten_by_sid_{n+1}"` at the boundary.

### 6.2 Across parent cycles — same df, in-place overwrite

There is **one entity df per (TF, role, parent_path)** for the entire
session. All parent cycles' subs of the same type live in this single df.
`parent_cycle_id` is meta on each sid / event / zone, not part of entity
identity.

When parent cycle K ends and parent cycle K+1 begins:

- The new parent cycle's `first_confluence` starts at the parent BOS
  extreme candle (which physically lives inside parent cycle K's
  territory, since `BOS_CONFIRMED.ev.idx` = extreme, not confirmation).
- Where it overlaps with parent cycle K's last same-type sub data, the
  §6.1 in-place overwrite rule applies:
  - df cols overwritten in place by the new sid
  - old sid's events / zones / POIs / fibs / WVMI persist with
    `deactivated_by="overwritten_by_sid_{n+1}"` and bounds capped at the
    overwrite boundary
  - old sid's still-open WVMI locked with the same `locked_by` reason
- Consumers query by `parent_cycle_id` meta when they need to scope to a
  specific parent cycle.

### 6.3 Sub reversal mid-cycle — does not perturb parent cadence

When a sub reverses internally:

- The sub gets a new sid via `reversal` scenario applied to the sub's own
  TF, data, and zones. `starting_alignment` and `starting_sd` remain
  fixed (sticky) for the entity.
- Parent's variation cadence is unaffected — parent's proximity state
  machine and event emissions continue as if nothing happened.
- WVMI sweep "starting from most recent same-type sub sid's start" uses
  whichever event birthed the *current* sid (reversal-induced or
  subsequent-trigger-induced), not strictly "most recent var 3/4 output."

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

| Data type | Storage | Overwrite policy |
|---|---|---|
| df columns (`structure_id`, `cycle_id`, `cts_phase`, `range_lo/hi`, etc.) | per-candle | Overwritten in place by new sid; only current owner's view per candle |
| Structure events (`StructureEvent` list) | `df.attrs["events"]` (append-only) | **Persist forever** with sid + cycle attribution; never deleted or mutated |
| Zones (KL, POI), Fibs, Wave candles, Imbalances | `df.attrs[...]` keyed by sid + cycle_id | Persist; old sid entries marked `deactivated_by="…"` and bounds capped |
| WVMI records | `df.attrs["wvmi"]` keyed by sid + cycle_id | Persist as snapshots; old sid in-progress locked at boundary |
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
- `df.attrs["sids"]` — list of sid records, each with `parent_sid`,
  `parent_cycle_id`, `starting_sd`, `creation_event_idx`,
  `end_event_idx`, `end_reason`. These are per-sid because one entity has
  many sub instances (one per parent_cycle), each with its own
  `starting_sd` derived from parent's current_sd at that moment.
- Events / zones / POIs / WVMI all carry `sid + cycle + parent_cycle_id`
  meta to support per-cycle filtering.

### 9.3 Per-entity df contents

| Path | Contents |
|---|---|
| df columns | TF candles (own copy) + `structure_id`, `cycle_id`, `cts_phase`, etc. |
| `df.attrs["structure_path_id"]` | This entity's id |
| `df.attrs["parent_path_id"]` | None for main |
| `df.attrs["sids"]` | Per-sid records: `parent_sid`, `parent_cycle_id`, `starting_sd`, `creation_event_idx`, `end_event_idx`, `end_reason` |
| `df.attrs["events"]` | Append-only events list, all sids of this entity, with `sid + cycle_id + parent_cycle_id` meta |
| `df.attrs["kl_zones"]`, `["poi_zones"]`, `["wave_candles"]`, `["fib_states"]`, `["wvmi"]`, `["imbalances"]` | All entity-local; sid + cycle keyed; old sid entries with `deactivated_by` meta |
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
       and z.meta["sid"] == sid_rec.parent_sid
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
   - Delete `add_scenario3_record` / `discard_scenario3` /
     "first sd-prox gate" code path.
   - Absorb `structure/proximity_helpers.py` into the registry pattern.
     Today's inline parent BOS/POI inner derivation inside MarketStructure
     becomes a standard cross-entity lookup
     (`registry.parent_of(self).df.attrs[...]`). The transitional
     backward dependency from `structure/` → `zones/` is then gone.
   - Drop `WVMIRecord.source = "main" | "scenario3"` keying — replaced by
     `structure_path_id` on every record.
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

### 16.5 Persistence and overwrite display rules

| Element class | Display rule |
|---|---|
| Constants — candle patterns, OHLC, candle types | Always shown; never affected by sid changes |
| Sid-tied — CTS dots, BOS markers, range bounds, market_state regions | **Most recent sid only** per candle. Older sids' data exists in df with `deactivated_by` meta but hidden by default |
| Persisting events — KL zones, POI zones, imbalances | All rendered. Deactivated / overwritten ones use the existing opacity attenuation logic (older = more transparent) |
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
