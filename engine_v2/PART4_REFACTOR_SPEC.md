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
> **⚠ REVISED 2026-09-20 — sub-structure pool (Plan C LANDED 2026-09-21,
> commit `afaa326`, save `20260921_125218_afaa326`).** The per-parent-cycle sid identity + merge-and-bound chain
> above are themselves superseded for subordinates by **§17** (`TriggerRecord`
> + unique sub, `sub_id`, the lifecycle sweep, cycle lifecycle-start = the
> CTS-established **moment**). §2, §4.3.6, §5, §6, §7, §9, §16.5 bodies are
> rewritten to rev 2; §13.5 is annotated; §17 is authoritative where any body
> still reads differently. Rationale: `memory/project_sub_structure_pool_architecture.md`;
> contract: `plans/PLAN_C_lifecycle_rewrite.md`.
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

> **⚠ REVISED by §17 (Sub-Structure Pool, rev 2 2026-09-19) for subordinates.**
> Under the pool: a unique sub's identity is `(parent_path, sub_TF, direction,
> starting_idx)` with `direction` **absolute**; `sub_id` is a single monotonic
> pool index (the per-parent-cycle counter survives only as `trigger_sub_sid`
> on `TriggerRecord`s, per `(lens, parent_sid, parent_cycle_id)`);
> `starting_alignment` is **not sticky and not part of identity** — it splits
> into `relative_dir` (semantic, `direction == parent_sd`) and `lens` (chart),
> both on the record (§17.3). Sticky behavior survives only as the lens rule for
> `reversal` records. See §17.2–§17.4. **Body rewritten to rev 2 in the Plan C
> commit (2026-09-20).**

**Main entity (unchanged).** A `main` market structure output is identified
by `timeframe` (the TF of its candles) + `role = main`; within it, `sid`
(`structure_id`) increments on reversal only, and `cycle_id` counts CTS-to-CTS
spans within a sid. Main's `SidRecord` keeps `sub_sid = structure_id` and
`sub_id = None` (§9.2); nothing below changes main.

**Subordinate — the unique sub (Plan C, §17.2–§17.3).** A sub structure is
identified by **`sub_id`**, the global monotonic creation index of a
**unique sub** (`PooledStructure`) whose key is

```
StructureKey = (parent_path, sub_tf, direction, starting_idx)
```

- `parent_path` — the immediate parent's `structure_path_id` (the constant
  `"H1.main"` today; the recursion-ready element — an M5 under `M15.counter`
  cannot collide with an M5 under `M15.confluence`).
- `sub_tf` — the sub's own TF (`"M15"`).
- `direction ∈ {+1, −1}` — **absolute** struct direction of the MS run. NOT a
  confluence/counter label.
- `starting_idx` — the probe-validated structural anchor (entity-absolute on
  the shared sub-TF frame; historical, §17 header).

Two triggers whose probes land on the same key resolve to the **same** sub —
the sub is independent of which trigger produced it (§17.1). A unique sub is
**not bound to a parent**: it spans parent cycles and parent sids.

**Parent attribution lives on the `TriggerRecord`, not on the structure.** Each
trigger that resolves to a sub produces (or is absorbed into) a record with
identity `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)` and FK
`sub_id` (§17.4). `trigger_sub_sid` starts at 0 per `(lens, parent_sid,
parent_cycle_id)` and increments each time a trigger in that scope resolves to
a **new** unique sub — it is the only survivor of the old per-parent-cycle
counter. Each record carries `trigger_type` ∈ {`first_confluence`,
`subsequent_confluence`, `first_counter`, `subsequent_counter`, `reversal`},
`trigger_idx`, `probe_finalize_idx` and its own real-time `start_idx`, so
"which trigger spawned this instance" is tracked on the record.

> **Retired (Plan C, 2026-09-20).** The per-parent-cycle identity tuple
> `(parent_sid, parent_cycle_id, sub_sid)` of the 2026-05-25 revision, and the
> entity-wide `entity_sid` before it (§13.5.c), are gone from every structural
> artifact: events, KL/POI/fib/WVMI meta, `SidRecord` and CSV columns carry
> **`sub_id`**; `parent_sid` / `parent_cycle_id` / `use_case` on a mirrored
> snapshot are informational copies from the sub's **first live record** and
> are never used for identity (§17.9). `sub_sid` survives only on main
> `SidRecord`s (`= structure_id`). `started_by` on a snapshot is the first live
> record's `trigger_type`; `start_trigger_idx` is split into `trigger_idx` /
> `probe_finalize_idx` (historical) and `start_idx` (real-time).

**Confluence vs counter is two record properties, not a structure property
(§17.3 — supersedes the sticky `starting_alignment`).** The rev-1 rule was
"`starting_alignment` is set at creation by trigger origin and sticky for the
structure's lifetime". Under the pool it splits:

- **`relative_dir`** (semantic): `confluence` iff the record's `direction ==
  parent_sd` of its parent **sid** (`CTS_ESTABLISHED.meta["struct_direction"]`,
  one direction per sid), else `counter`. On the unique sub it is a **step
  function** over time (`relative_dir_segments`: at each candle, the value of
  the active record with the latest `start_idx`, carried forward when none is
  active) — it can flip across a parent-sid change without the structure
  changing.
- **`lens`** (chart): which chart the record draws on — `resolve_lens`:
  `first_confluence` / `subsequent_confluence` → `confluence`, `first_counter`
  / `subsequent_counter` → `counter`; a `reversal` record inherits the lens of
  the record whose sub reversed (the only surviving stickiness). A sub's
  lenses = the union over its non-zero-length records.

The old `starting_sd` / `current_sd` entity state reduces to the sub's
absolute `direction` (a unique sub is a SINGLE directional structure — its
natural reversal spawns a **different** sub of the opposite direction, §17.5),
and "aligned with parent" is `relative_dir` at that candle.

**Two distinct "start" idxs (REVISED 2026-05-25; restated for rev 2).** A
structure (and likewise a cycle) has both:
- **`starting_idx`** — the structural anchor: the BOS-equivalent candle the
  structure is computed from (the probe's output `ProbeResult.starting_idx`).
  May lie in the past relative to when the structure becomes active.
  HISTORICAL.
- **lifecycle start** (`start_idx`, first *active* idx) — the candle where
  the structure/cycle becomes active. REAL-TIME. Before this idx the prior
  structure/cycle is still active.

These mirror main structure: a cycle's `starting_idx` is the prior BOS
candle, but the cycle becomes active at the CTS-established **moment**
(`CTS_ESTABLISHED.meta["confirmed_at"]`, §5 / §17.6 — never the CTS anchor, a
location; `.idx` IS the moment since Plan E E4a). For a sub, a **record**'s `start_idx = max(probe_finalize_idx,
trigger_idx, parent_floor_idx)` and the **unique sub**'s `start_idx` is its
first non-zero-length record's `start_idx` (§17.4–§17.5); `starting_idx` is
only the geometric anchor. The active window is `[start_idx, end_idx)` —
half-open for the record's is-active test (`TriggerRecord.is_active_at`:
`start_idx <= t < trigger_end_idx`) and for the non-overlap assert; the
chart's owner map paints the sub through `end_idx` inclusive (§16.5), the
boundary candle being shared with whatever starts there.

**Identity is a path, not a flat tuple.** A structure like
`5M.confluence` under `15M.counter` under `H1.main` carries its full
ancestry through `parent_path`. The path bottoms out at `main` on the highest
TF.

> **Open (deferred):** the canonical encoding of this path for use as a
> dict key / chart attribution / event meta. Today `parent_path` is the parent
> entity's `structure_path_id` (§9.1, `H1.main`) and the lens dfs are
> `H1.main >> M15.confluence` / `H1.main >> M15.counter`; a deeper path
> (e.g. `H1.main >> M15.counter >> M5.confluence`) is the same format, but no
> M5 nesting is built (§17.12).

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
| **Probe `probe_end_idx`** | `reversal_idx` |
| **Probe reference zone** | Most recent CTS zone prior to reversal |
| **Trigger** | Reversal event |
| **Output** | `starting_idx` (`ProbeResult.starting_idx`) for the structure after reversal |
| **Applies to** | `main` and any `subordinate` |

For subordinates, `reversal` operates entirely on the subordinate's own
data and zones — not the parent's (`_resolve_reversal_start`: reference = the
reversing sub's most recent {CONFIRMED/UPDATED/ESTABLISHED} CTS read from its
own geometry, `probe_end_idx` = its natural reversal idx). Under the pool
(§17.5) the successor is a **different unique sub** of the opposite
`direction`, spawned once per record of the reversing sub that is live at the
reversal; each successor record inherits that record's `lens` and parent
linkage (`lens`, `parent_sid`, `parent_cycle_id`).

No mechanical change vs today's Scenario 2 logic — but probe reset
tolerance becomes TF-keyed (see §4.4).

### 4.3 `subordinate` — Four variations

Used to find `starting_idx` for new subordinate structures.

> **Session 3 (2026-05-31):** the probe now runs on the **structure's OWN sub
> TF** (the `unified_probe` primitive), NOT on the parent's TF. The old
> "probe always runs on the parent's TF, produces a parent-TF starting_idx,
> then map down" model is retired for all four variations. Every probe input
> and bound is on the sub TF before the single sub-TF probe runs: only
> `first_confluence` maps its H1 input (price-mapped, §4.3.1; its
> `probe_end_idx` = the H1 CTS anchor `parent_cts_anchor_idx`, price-mapped); the three sibling types take
> their M15 input from the sibling CTS (§4.3.3–§4.3.5; the H1 input is
> informational — `parent_input_idx`) and end at their trigger's LOH `hi`.
> See §4.4 for the unified-probe mechanics.

> **Naming (Plan C, 2026-09-20).** The probe's search bound is
> **`probe_end_idx`** everywhere — `unified_probe(probe_end_idx=…)`,
> `_probe_with_cache` / `ProbeCacheEntry` (the FC trigger's H1 field / meta key had the
> same name until Plan E Post-E·3, 2026-09-27: now `parent_cts_anchor_idx`)
> — a *compute* bound (the inclusive upper edge of the search window, like the
> MS run cap) with no relation to the lifecycle `end_idx` of §17. The probe's
> **output** is `ProbeResult.starting_idx` — the structural anchor and the pool
> key (historical), never a lifecycle value. The "Probe end_idx" rows below
> are this `probe_end_idx`.

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
  `probe_end_idx` reached (`end_idx_reached`); or (live mode) pending.
- Every sub probe runs behind the §17.8 **probe cache** (`_probe_with_cache`
  for the four H1 types; the same key logic inline in `_resolve_reversal_start`):
  key `(parent_path, sub_tf, direction, input_idx)`; a hit returns the cached
  `starting_idx` / `finalize_idx` / `finalize_condition` / `bos0_inner` and
  skips `unified_probe`.

**Per-variation sd/TF + input+reference source:**

| Variation | Probe sd | Probe TF | input_idx + reference source |
|---|---|---|---|
| `first_confluence` | `+parent_sd` | sub TF | parent BOS anchor (price→M15) + own ad-hoc BOS_0 |
| `first_counter` | `-parent_sd` | sub TF | sibling confluence CTS, same-dir only (input == ref's `anchor_idx`) |
| `subsequent_confluence` | `+parent_sd` | sub TF | sibling counter CTS, same-dir only (input == ref's `anchor_idx`) |
| `subsequent_counter` | `-parent_sd` | sub TF | sibling confluence CTS, same-dir only (input == ref's `anchor_idx`) |

**Time/price mapping rules (sub-TF translation, Session 2 generalization
2026-05-29, still current):**

- **Price-based** (`first_confluence` input **and end**): `parent_extreme_dir =
  -lower_sd`; the parent BOS anchor maps to the M15 candle whose extreme on
  the `-lower_sd` side touches the OUTER of the sub's reference zone.
  (`map_candle_to_lower_tf` in `data_bridge.py`.) **CORRECTED 2026-09-19:**
  `first_confluence`'s probe end (the parent CTS anchor,
  `cts_anchor_idx`, carried as `parent_cts_anchor_idx`) is ALSO price-mapped (`+lower_sd` side) — it is a PRICE
  bound for the search, not a temporal gate (`entity_df_mutation.py`
  `_resolve_first_confluence_via_unified_probe`). This section previously said
  "every probe `end_idx`" is time-mapped; that was never true for
  `first_confluence` and the code is authoritative. Consequence: when the
  probe finalizes on the `probe_end_idx` branch (`no_retrace` else /
  `end_idx_reached`), `finalize_idx` inherits the price-mapped value. Verified
  live on the 2025-11→2026-01 window
  (`memory/reference_pool_redesign_groundtruth.md`). Unchanged by Plan C
  (§17.8 "FC probe end mapping — unchanged").
- **Time-based** (every **sibling-referencing** `probe_end_idx` —
  `first_counter` / `subsequent_*`, where it equals the trigger's
  `trigger_idx` — and every sub window endpoint / lifecycle value:
  `trigger_idx`, the parent floors and ends in `multitf/parent_tables.py`, the
  WVMI trigger candle): `_map_parent_idx_to_m15_hour_end` (LOH) — the parent
  gate candle maps to the LAST M15 candle of its hour. Never unified with the
  price mapper.
- The three sibling-referencing variations need NO parent→sub price map for
  their input: `input_idx` is the sibling's CTS anchor, already an
  entity-absolute M15 idx — read from the **pool** (§17.8
  `_build_sibling_cts_ref_zone_from_pool`: the sibling lens's records for this
  parent cycle, their subs' geometry shifted to entity-absolute, clipped to
  each record's live window ∩ `[lo, hi]`).

The LANDMINE "Subordinate `parent_extreme_dir` must use `-lower_sd`" tracks the
one remaining price-map case (`first_confluence`).

#### 4.3.2 `first_confluence` — first sub of a parent cycle, sd = parent_sd

Find `starting_idx` for the **first** confluence subordinate of a parent
cycle (sid=0 of the confluence entity).

| Field | Value |
|---|---|
| **Trigger** | Most recent parent BOS confirmed (== CTS established for the new cycle, by definition same candle: `BOS_CONFIRMED.meta["confirmed_at"] == CTS_ESTABLISHED.meta["confirmed_at"]`) |
| **Idx input** | Idx of the newly confirmed BOS (`FirstConfluenceTrigger.input_idx`, price-mapped to M15 on the `-lower_sd` side) |
| **Probe `probe_end_idx`** | The confirmed CTS's **anchor** (`cts_anchor_idx` on the `CTS_CONFIRMED` event) in the same parent cycle — *not* the confirmation candle (`CTS_CONFIRMED.idx == confirmed_at`, which is later). Price-mapped to M15 on the `+lower_sd` side. NULL (`FirstConfluenceTrigger.parent_cts_anchor_idx = None`, `status = "pending"`) until parent `CTS_CONFIRMED` fires. |
| **Probe reference zone** | Own ad-hoc BOS_0 on the sub TF from the mapped input candle (`_build_first_confluence_ref_zone`) |
| **Output** | `starting_idx` for the first confluence sub of the cycle (the record gets `trigger_sub_sid = 0` on the confluence lens if it is the first to resolve there) |

**NULL `parent_cts_anchor_idx`:** This is the **only** variation where the probe bound
may initially be NULL. We **wait** — no first_confluence sub is built until
parent CTS_CONFIRMED resolves it (consistent with today's "pending" handling,
but the result is "do not produce" rather than "produce tentatively"); under
the pool a still-pending trigger is logged as
`UnresolvedTrigger(reason="pending")` (§17.7). When it resolves,
`parent_cts_anchor_idx` is set to the confirmed CTS's **anchor**
(`cts_anchor_idx`), which is *earlier* than the confirmation candle: the
confirmation candle gates only *when* the value becomes known; the CTS anchor
is the value that bounds the probe. Using the confirmation candle would
over-extend the probe window past the CTS, shifting the confluence sub's
validated start.

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
| **Trigger** | First parent sd-zone proximity trigger (BOS or POI) after parent CTS (`WVMIRecord.meta["triggered_by_event_idx"]` → `trigger_event_idx`) |
| **Idx input** | Sibling confluence lens's most recent qualifying CTS **anchor** on the sub TF (= the reference zone's `anchor_idx`; CONFIRMED → `cts_anchor_idx`, UPDATED/EST → the CTS anchor `ef.cts_anchor_idx` — the event idx on UPDATED, meta `cts_anchor_idx` on EST, whose idx is the moment since Plan E E4a). **Same candle as the reference zone** — input and ref are co-sourced from the sibling CTS event (Session 3 uniform rule). |
| **Probe `probe_end_idx`** | First parent sd-zone proximity trigger candle after CTS, time-mapped to the sub TF's last-of-hour = the trigger's `trigger_idx` (`hi`) |
| **Probe reference zone** | Sibling confluence lens's most recent qualifying CTS, built ad hoc from the winning CTS event (§17.8: `kl_zones=[]` — subs receive BOS zones only, so no derived CTS zone exists), read from the pool within the sub-TF window `[0, this trigger]` (`_sibling_cts_idx_window`; the pool read is already scoped to this parent cycle) |
| **Output** | `starting_idx` for the first counter sub of the cycle |

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
| **Idx input** | Sibling **counter** lens's most recent qualifying CTS **anchor** on the sub TF (= the reference zone's `anchor_idx`; CONFIRMED → `cts_anchor_idx`, UPDATED/EST → the CTS anchor `ef.cts_anchor_idx` — the event idx on UPDATED, meta `cts_anchor_idx` on EST, whose idx is the moment since Plan E E4a). **Same candle as the reference zone.** |
| **Probe `probe_end_idx`** | Current parent CTS-zone proximity trigger candle, time-mapped to the sub TF's last-of-hour = the trigger's `trigger_idx` (`hi`) |
| **Probe reference zone** | Sibling **counter** lens's most recent qualifying CTS, built ad hoc from the winning event (`kl_zones=[]`, §17.8), read from the pool within the sub-TF window `[last-M15-of prior_sd_prox hour, last-M15-of this_cts_prox hour]` |
| **Output** | `starting_idx` for the next confluence record's sub (a new unique sub, or an existing one if the key repeats) |

**Reference zone + input resolution (Session 3 uniform rule, 2026-05-31;
sibling source = the pool since Plan C):**
1. Build the sub-TF idx window: `[_map_parent_idx_to_m15_hour_end(prior_sd_prox_idx), _map_parent_idx_to_m15_hour_end(this_cts_prox_idx)]` (entity-absolute M15; `_sibling_cts_idx_window`).
2. `_build_sibling_cts_ref_zone_from_pool(pool, other_lens="counter", S, C, probe_direction, (lo, hi), m15)`: take the **counter** lens's records for this `(parent_sid, parent_cycle_id)` with `direction == -probe_direction`, non-zero-length and live somewhere in the window; collect their subs' {CTS_CONFIRMED, CTS_UPDATED, CTS_ESTABLISHED} events (geometry shifted to entity-absolute, clipped to each record's live window ∩ `[lo, hi]`); the most recent wins via `build_reference_zone_from_cts_event`.
3. `input_idx = ref_zone.anchor_idx`; `reference_zone = ref_zone`. Probe runs on the ONE shared M15 feature frame (Phase 1 only, `probe_end_idx = hi`), behind the probe cache.
4. **Fallback** (sibling has zero qualifying CTS events in the window — extremely rare): input_idx = window extreme on the `-lower_sd` side computed on the shared M15 frame (`_window_extreme_idx`); reference = own ad-hoc BOS_0 from that candle.

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
| **Idx input** | Sibling **confluence** lens's most recent qualifying CTS **anchor** on the sub TF (= the reference zone's `anchor_idx`; CONFIRMED → `cts_anchor_idx`, UPDATED/EST → the CTS anchor `ef.cts_anchor_idx` — the event idx on UPDATED, meta `cts_anchor_idx` on EST, whose idx is the moment since Plan E E4a). **Same candle as the reference zone.** |
| **Probe `probe_end_idx`** | Current parent sd-zone proximity trigger candle, time-mapped to the sub TF's last-of-hour = the trigger's `trigger_idx` (`hi`) |
| **Probe reference zone** | Sibling **confluence** lens's most recent qualifying CTS, built ad hoc from the winning event (`kl_zones=[]`, §17.8), read from the pool within the sub-TF window `[last-M15-of prior_cts_prox hour, last-M15-of this_sd_prox hour]` (same pool read as §4.3.4 step 2 with `other_lens="confluence"`) |
| **Output** | `starting_idx` for the next counter record's sub |

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

#### 4.3.6 Cadence of triggers within a parent cycle — the sweep (REPLACED by §17.6, Plan C 2026-09-20)

By construction, the four variations naturally **fire** in this order within
each parent cycle (the parent's proximity state machine, `zone_proximity.py`,
enforces sd/opp_sd alternation; both `subsequent_*` variations require the
prior trigger to be the opposite kind):

```
parent BOS_confirmed
    → first_confluence              (input: own ad-hoc BOS_0, probe_end_idx: parent CTS anchor, price-mapped)
parent 1st sd-proximity (post-CTS)
    → first_counter                 (input+ref: sibling confluence CTS, probe_end_idx: this trigger candle)
parent CTS-proximity (after sd-prox)
    → subsequent_confluence         (input+ref: sibling counter CTS, probe_end_idx: this trigger candle)
parent sd-proximity (after CTS-prox after sd-prox = Λ/V apex at CTS)
    → subsequent_counter            (input+ref: sibling confluence CTS, probe_end_idx: this trigger candle)
... alternating until parent cycle ends
```

**The driver is the lifecycle sweep (`multitf/lifecycle_sweep.py::run_lifecycle_sweep`,
§17.6), not a cadence chain.** The rev-1 diagram above was also the *build*
order of a two-entity chain (`build_two_entity_parent_cycle` / `_ChainCursor`
advancing whichever entity had the smaller next M15 boundary). Under the pool
the H1 triggers of ALL parent cycles are collected first (each tagged
`lens = resolve_lens(use_case)`, `trigger_idx = LOH(trigger_event_idx)`,
`direction = lower_sd`, parent `(S, C)`; a `first_confluence` with `status !=
"finalized"` is `pending`) and pushed onto ONE priority queue of **moments**
`(idx, phase, order_key)`. At each idx the phases run in this order, each to
completion before the next:

| phase | moment | action |
|---|---|---|
| 0 | `TRIGGER_FIRE` (an H1 trigger at its `trigger_idx`) / `REVERSAL_SPAWN` (a built sub's `natural_reversal_idx`) | resolve: degenerate-cycle check (§17.7) → probe (behind the cache, §17.8) → `build_or_get_geometry` → create the `TriggerRecord` (or absorb into the existing record of the same `(lens, S, C)` → same sub) → queue its `RECORD_START` and any end already known |
| 1 | `RECORD_START` | if the sub already ended **before** this idx → freeze the record zero-length; else mark active and queue a `same_dir_replacement` `RECORD_END` for the same-lens, same-`(S, C, direction)` incumbent of a *different* sub at this idx |
| 2 | `SUB_START` | set `sub.start_idx` if unset and a live record started here |
| 3 | `RECORD_END` | for a record not yet ended: apply the highest-priority end entry at this idx (`reversal > parent_end > same_dir_replacement`); entries for an already-ended record are stale and dropped |
| 4 | `SUB_END` | re-evaluate the §17.5 sub-end rule for every sub that had a live record start or end at this idx |

**Order within a phase.** `TRIGGER_FIRE` sorts by `(lens_rank, type_rank,
trigger_event_idx)` — confluence (0) before counter (1), `first_*` (0) before
`subsequent_*` (1) — i.e. the cadence order above, which still matters because
the second trigger's sibling read at `hi = trigger_idx` **inclusive** can see the
first's sub. `REVERSAL_SPAWN` sorts **before** `TRIGGER_FIRE` at the same idx
(the successor must exist before an H1 trigger at that candle runs its sibling
read). Every other moment orders by record creation `seq`. **Start-before-end
at the same idx is load-bearing** (§17.5's same-candle handover). The heap is
idx-monotone (a moment is never queued in the past — asserted; a natural
reversal `R <= trigger_idx` spawns nothing).

**Trigger collisions on the same candle** are handled by that order key —
neither variation blocks the other, and no variation's predicate needs to know
about the other. (One known impossible-in-practice case: a single candle
simultaneously triggering sd-proximity and CTS-proximity is geometrically
infeasible because a green candle moves toward CTS and a red candle moves
toward BOS/POI, so a single candle can only trigger one.)

**Triggers are independent (§17.4).** A `subsequent_*` trigger is processed
even when its parent cycle's `first_*` trigger was unresolved or produced a
zero-length record. The rev-1 "no sid 0 → no `subsequent_*`" rule (old §6.1
step 2) is retired.

**Bootstrap (first parent cycle after `trading_open`):** if
`trading_open` lands the main start mid-parent-cycle, no
`first_confluence` is built for the in-progress cycle. The cadence begins
at the next parent BOS_confirmed.

#### 4.3.7 Variations summary

| Variation | Trigger | Input idx | `probe_end_idx` | Reference zone | Probe sd |
|---|---|---|---|---|---|
| `first_confluence` | Parent BOS_confirmed | Parent BOS anchor (price-mapped to M15) | Parent CTS **anchor** (`cts_anchor_idx`, price-mapped; NULL until parent CTS_confirmed fires) | **Ad-hoc BOS_0 on sub TF** (derived from input_idx candle on M15) | `+parent_sd` |
| `first_counter` | 1st parent sd-proximity post-CTS | **Sibling confluence CTS anchor** (= ref's `anchor_idx`) | This trigger candle (`trigger_idx`) | **Sibling confluence lens's most recent same-direction CTS** (`struct_direction == +parent_sd`; ad-hoc zone from the winning event, from the pool) | `-parent_sd` |
| `subsequent_confluence` | Parent CTS-proximity after sd-prox | **Sibling counter CTS anchor** (= ref's `anchor_idx`) | This trigger candle (`trigger_idx`) | Sibling counter lens's most recent same-direction CTS (`struct_direction == -parent_sd`; in M15 window; same rule) | `+parent_sd` |
| `subsequent_counter` | Parent sd-prox forming Λ / V | **Sibling confluence CTS anchor** (= ref's `anchor_idx`) | This trigger candle (`trigger_idx`) | Sibling confluence lens's most recent same-direction CTS (`struct_direction == +parent_sd`; in M15 window; same rule) | `-parent_sd` |

Output of every row: `ProbeResult.starting_idx` → the pool key
`(parent_path, "M15", lower_sd, starting_idx)` (§17.3); `ProbeResult.finalize_idx`
→ the record's `probe_finalize_idx` (§17.4).

**Uniform input+reference rule (2026-05-31, Session 3).** All three
sibling-referencing variations (`first_counter`, `subsequent_confluence`,
`subsequent_counter`) co-source BOTH `input_idx` AND `reference_zone` from the
**sibling entity's most recent CTS event** (CONFIRMED → existing KL zone +
`cts_anchor_idx`; UPDATED/EST → ad-hoc CTS zone + the CTS anchor `ef.cts_anchor_idx` — not
an EST's `ev.idx`, the moment since Plan E E4a), found by walking
the sibling's events within the trigger's sub-TF idx window. `first_confluence`
is the only exception — it has no sibling/prior structure yet, so it anchors on
its own ad-hoc BOS_0 from the parent-BOS-anchor input. The probe always runs
on the structure's OWN sub TF.

**Direction-qualified sibling CTS (2026-06-15).** "Most recent sibling CTS"
means the most recent CTS **from a sibling sub still in its OWN expected
(bootstrap) direction** — uniformly `struct_direction == -lower_sd` (the
referencing trigger and its sibling run in opposite directions, and each
variation reads the OTHER entity, so the qualifying sibling direction is always
the opposite of the trigger's own `lower_sd`):

| Trigger (`lower_sd`) | Sibling read | Qualifying sibling `struct_direction` |
|---|---|---|
| `first_counter` (`-parent_sd`) | confluence | `+parent_sd` (`= -lower_sd`) |
| `subsequent_counter` (`-parent_sd`) | confluence | `+parent_sd` (`= -lower_sd`) |
| `subsequent_confluence` (`+parent_sd`) | counter | `-parent_sd` (`= -lower_sd`) |

A sibling sub that has **reversed** away from its expected direction is no
longer a genuine confluence/counter relative to the parent, so its CTS does NOT
qualify to seed the trigger — even if it is the latest CTS in the window. The
original Session-3 rule overlooked that a sibling can reverse *before* this
trigger fires; an in-progress reversed CTS that keeps UPDATING up to the trigger
candle would otherwise drag the anchor to the trigger and collapse the probe
window. If the sibling later reverses **back** into the expected direction, those
re-aligned CTS qualify again (most-recent qualifying wins). When **no**
same-direction sibling CTS exists in the window, the per-variation own-entity
fallback (step below) applies — unchanged. Enforced in
`_build_sibling_cts_ref_zone_from_pool` (Plan C; formerly
`_build_sibling_cts_ref_zone` over a sibling entity df) by filtering the
sibling lens's **records** to `direction == -probe_direction` before collecting
their subs' CTS events — under the pool a reversed sibling is a *different*
unique sub with its own records, so the direction filter is on the record's
`direction`, not on a per-event `struct_direction`.

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
confluence) requires the referenced sibling sub to be built before the
reading probe fires. Because the reads point in OPPOSITE directions, no
"build one entity fully then the other" order satisfies all of them.
Session 3 (2026-05-31) replaced the confluence-first serial pair
(`_run_first_confluence_multi_tf` then `_run_multi_tf`) with a single
two-entity cadence driver (`build_two_entity_parent_cycle` / `_ChainCursor`,
interleaving both M15 chains in trigger-cadence order). **Plan C (2026-09-20)
replaced that driver with the lifecycle sweep** (§4.3.6 / §17.6): triggers
are processed in `(trigger_idx, phase, order_key)` order across ALL parent
cycles, and a sibling read at trigger `T` sees exactly the sibling lens's
records that are live somewhere in `[lo, hi = T.trigger_idx]` — every such
record's sub was built at a `TRIGGER_FIRE` / `REVERSAL_SPAWN` moment ≤ the
reading moment, so the sibling is always already built and the read is
causal. See LANDMINES "Cross-entity sibling references require cadence-order
interleaving" (the order key preserves the cadence order at an equal idx).

---

## 4.4 Probe reset thresholds (applies to unified `reversal` + `subordinate` probe)

The unified probe primitive (Phase 1 design 2026-05-29 — see
`memory/project_unified_identify_start_probe.md`, code in
`engine_v2/structure/unified_probe.py`) replaces today's four asymmetric
paths. Per iteration it picks the single most-extreme retrace candle in
`[first CTS_ESTABLISHED moment + 1, probe_end_idx]` (Phase 2: the upper bound is
the CTS_0_CONFIRMED anchor − 1 when cycle 0 confirmed) and applies a **two-condition reset**;
both must hold for the probe to restart from that candidate.

> **Both phases start the retrace window at the CTS_0 established MOMENT + 1.**
> Phase 1 (deterministic): `tfb.est_idx + 1` (`est_idx` = the true first
> breakout's apply candle). Phase 2 (MS-based, first_confluence only):
> `check_lo = ef.event_moment(cts_est[0]) + 1` since Plan E E3c (2026-09-25;
> before it `cts_est[0].idx + 1`, the CTS_0 EXTREME — the divergence verified
> 2026-09-22, where Phase 2 also scanned the candles after the extreme up to
> and including the moment). 0 cells on the reference window (its one Phase-2
> run has a lag-0 first CTS). Any change can move the reset candidate (so
> `starting_idx` = the pool key) and is its own `/compare`.

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
3. a **strict** new full-pattern extreme over `[current_start, pattern_extreme_idx)`
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
- **non-FC** (`first_counter` / `subsequent_*` / `reversal`): `probe_end_idx =
  trigger` (the trigger's `trigger_idx` for the H1 types; the natural reversal
  idx for `reversal` — all historical) → fully **deterministic** — find the
  true breakout + the max-retrace reset over `[CTS_0_EST+1, probe_end_idx]`,
  multiple resets, NO MS, no `cts_anchor`. `finalize_idx == probe_end_idx ==
  trigger_idx` by construction (asserted for the three sibling types in
  `_resolve_sibling_cts_via_unified_probe` and in the sweep, for a probe that
  actually ran; a cache-hit record inherits an earlier finalize and is exempt,
  §17.4; for `reversal` the equality holds by the same construction but is
  not separately asserted).
- **first_confluence** (hybrid) — **CORRECTED 2026-09-19 to match the code**:
  both the deterministic pass and Phase 2 receive `probe_end_idx` = the
  **price-mapped parent CTS anchor** (`cts_anchor_idx`, §4.3.1); Phase 2 runs
  MS **bounded at `probe_end_idx`** (`_make_market_structure(...,
  end_idx=probe_end_idx)` — MS's own bound parameter keeps its `end_idx` name)
  — it is NOT treated as NULL. The retrace window is `[CTS_0_EST+1,
  cts0_anchor-1]` when MS confirms cycle 0 inside the window, else
  `[CTS_0_EST+1, probe_end_idx]`. Exit classification: `second_cts_reached` if ≥2
  CTS established (finalize = the 2nd `CTS_ESTABLISHED`'s `confirmed_at`,
  native M15), else `no_retrace` (finalize = `CTS_0_CONFIRMED.idx` if cycle 0
  confirmed, **else `probe_end_idx`** — the common case). The double-CTS rule
  is an **early stop** (Plan B, landed 2026-09-20): Phase-2 MS is handed
  `stop_after_cts_established=2` and stops at the first quiescent point (no
  reversal watch / pending reversal; MS has no rewind since 2026-09-29) after the 2nd
  `CTS_ESTABLISHED` — in addition to the `probe_end_idx` bound, never instead
  of it (`n_cts ≤ 1` runs still reach `probe_end_idx`); finalize = that CTS's
  moment (`confirmed_at`), not its CTS anchor (`cts_anchor_idx`; `.idx` until Plan E
  E4a). Byte-identical on the
  reference window (FC(0,0) stops at 1021 for finalize 1020, FC(0,1) at 2609
  for 2608; `.idx == confirmed_at` for both). Two different "anchor"s here:
  the parent's `cts_anchor_idx` (H1, the probe bound) vs the probe's own M15
  `cts0_anchor` (from Phase-2 MS) — see GLOSSARY. Under the pool the FC probe
  runs only for a trigger whose parent cycle is not degenerate and whose key
  is not already in the probe cache (§17.7/§17.8 — measured 2026-09-20:
  FC(0,1)'s Phase-2 probe was skipped on a cache hit, §17.8).
- **main sid0|cyc0** (`trading_open`): arbitrary ad-hoc BOS_0 at the
  `identify_start_scenario_1` start; single-shot, no resets (Commit 2 — not yet
  wired).

### MS pre-CTS_0 scan-from-start (`enforce_cts0_new_extreme` + `bos0_inner`)

The probe hands MS a **decision, not events**: `{finalized current_start, BOS_0
bounds}` (+ `cts0_established_idx`, informational — logged, no production reader). MS does **NOT** seed state at
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
`compute_bounded_structure(enforce_cts0_new_extreme, bos0_inner)` via the
resolver's `ResolvedStart.bos0_inner` → the sweep → `build_or_get_geometry`
(Plan C; formerly the `build_one_sid` / `_ChainCursor` handoff with
`SidBuildOutcome.next_bos0_inner` for reversal-born subs — both gone). It is
stored on the `PooledStructure` (`sub.bos0_inner`) and in the probe-cache
entry; it is **not** part of the pool key (§17.8) — no index remap.

**Scope:** all M15 subs use this (Commit 1, every trigger). **Main `sid0|cyc0`**
landed in Commit 2 (flip the flag + pass `bos0_inner` at the main
`compute_structure` call). **Main reversals H1 `sid≥1`** landed in Step 4
(2026-06-20): `compute_structure`'s per-reversal handoff now runs `unified_probe`
(reference = prior sid's most recent `{CONF/UPD/EST}` CTS; flipped direction;
`probe_end_idx` = the reversal apply idx) and feeds its `bos0_inner` to a per-sid
scan-from-start MS run — the SAME path the subs use. The old Scenario 2 + Exc1 +
Exc2 chain is gone from the main loop; `/compare` byte-identical on the NZD_USD
window (the lone main reversal reproduces the old refined start 689 → CTS_0 703,
established by `find_true_first_breakout`, scan-on==scan-off verified). Exception 1
is no longer reached from main; it was left in
`identify_start_scenario_2_after_reversal` until that function and its last
callers (`compute_structure_from_start` [no prod caller] +
`compute_structure_scenario_3` Phase 2 [tests]) were **deleted 2026-09-30**
(user decision; MARKET_STRUCTURE_SPEC "Compute_structure variants"). The legacy `_resolve_via_legacy_probe` escape hatch
(`_LEGACY_PROBE_USE_CASES`, always empty) and `lower_tf_pipeline.py` were
**deleted by Plan C** (2026-09-20): it returned `finalize_idx=None`, which the
§17.4 non-FC assert (`finalize_idx == trigger_idx`) would have tripped.

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
post-trigger — making the variation's `probe_end_idx` and `starting_idx`
overlap or invert. Strict `proximity > reset` is the cleanest invariant.

### Migration state

- **Tables wired** as of Phase 1 Session 1 (this update). The probe
  primitive itself exists at `engine_v2/structure/unified_probe.py` but
  no caller has migrated yet — Sessions 2–6 of Phase 1 migrate per
  trigger. The legacy probes that remained
  (`compute_structure_scenario_3` Phase 1; the Exception 2 probe in
  `compute_structure_from_start` + `compute_structure_scenario_3` Phase 2 —
  `compute_structure` itself migrated off Exception 2 in Step 4, 2026-06-20)
  used `DEFAULT_PROBE_RESET_PIPS` only (single-condition) until they were
  deleted 2026-09-30.
- **Proximity tuning** (held back from Session 1 so the unified probe landed
  byte-identical to the `804d19d` baseline; the plan then was
  `{H1: 8, M15: 6, M5: 4}`): the code's `DEFAULT_PROXIMITY_PIPS` is
  `{H1: 8, M15: 5, M5: 3}` (checked 2026-09-30) — the one source; this
  bullet is history.

---

## 5. Computing Structures — Function Routing per Role

> **⚠ REVISED by §17 (Sub-Structure Pool, rev 2 2026-09-19) for subordinates.**
> Under the pool a unique sub's geometry runs to the **data edge** (run cap),
> not bounded at the next trigger. Lifecycle lives on two objects: each
> `TriggerRecord` starts at `max(probe_finalize_idx, trigger_idx,
> parent_floor_idx)` and ends at the earliest of `{own reversal,
> same_dir_replacement (same lens + parent + direction, at the new record's true
> start_idx), parent_end}` (§17.4); the unique sub starts at its first live
> record's `start_idx` and ends at `min(record end > max(started records'
> start_idx))` (§17.5). At most one record is active per `(lens, parent_sid,
> parent_cycle_id, direction)`. The clamp/pass-through machinery below still
> applies but is projected **once** per unique sub (§17.9). The cycle
> lifecycle-start becomes the CTS-established **moment**
> (`meta["confirmed_at"]`), not the CTS anchor (§17.6) — this reaches `main` too
> (byte-identical on the reference window).

For `main` (highest TF), continue using `compute_structure`:

- Initial start: `trading_open` scenario (today's `identify_start_scenario_1`).
- On reversal: `reversal` scenario (then `identify_start_scenario_2_after_reversal` + Exception 1/2; since Step 4 `unified_probe` + scan-from-start — the old function was deleted 2026-09-30).
- Loops until end of data.

For `subordinate` (any lens, any TF), each **unique sub** is built **once** by
a bounded single-structure run of `compute_bounded_structure` inside
`entity_df_mutation.build_or_get_geometry` (Plan C, 2026-09-20; REVISED from
the 2026-05-25 per-sid bounded run):

- Start (`starting_idx`) is pre-validated by the trigger's probe (§4.3) or by
  the reversal handoff (`_resolve_reversal_start`) — no internal Scenario 1
  logic. The pool key `(parent_path, sub_tf, direction, starting_idx)` is
  looked up first; a hit reuses the existing geometry, a miss runs MS and
  creates the pool entry **only on success** (a failed build consumes no
  `sub_id`, §17.7).
- The run is bounded to `[starting_idx, run_cap]` with **run cap = the data
  edge** (`len(m15) − 1`) for every sub — a *compute* bound only (§17.5) — and
  it **stops at its own first reversal** if one occurs. A unique sub is
  therefore a SINGLE directional structure — it never rolls past a reversal;
  the reversal (`natural_reversal_idx`) is what spawns the opposite-direction
  successor sub (§17.5). The rev-1 bound `min(next subsequent-variation
  trigger, parent-cycle-end)` is gone: those are now **lifecycle** ends on the
  `TriggerRecord` (§17.4), applied at projection time, not run bounds.
- `bounded.events` / `bounded.df` are slice-local (50-candle lookback,
  `reset_index`, `compute_imbalance` re-run, `is_range_*` re-derived);
  `slice_begin` makes them entity-absolute. The geometry is shared by every
  record of the sub and never mutated in place.

The legacy `compute_structure_scenario_3` ad-hoc path is **removed** (the function itself deleted 2026-09-30). Its
only purpose (creating WVMI when parent was in range) is now subsumed by
per-subordinate WVMI — see §8.

### sid increment rules

| Role | what increments |
|---|---|
| `main` | `sid` (`structure_id`) +1 on reversal only |
| `subordinate` | a **new `sub_id`** each time a trigger's probe lands on a new pool key (a new `starting_idx` for that direction) — whether the trigger is a `first_*`, a `subsequent_*` or a `reversal`; a **new `trigger_sub_sid`** (per `(lens, parent_sid, parent_cycle_id)`, from 0) each time a trigger in that scope resolves to a sub it has not yet recorded |

Under the pool there is no per-parent-cycle sid chain (the merge-and-bound
chain of the 2026-05-25 revision is retired, §6). What is sequential and
non-overlapping is the set of **live records per `(lens, parent_sid,
parent_cycle_id, direction)`**: at most one is active at any candle, their
windows are non-overlapping half-open intervals `[start_idx, end_idx)`
(asserted after the sweep), and a new one ends the incumbent by
`same_dir_replacement` at its own `start_idx` (§17.4). Subs of opposite
direction may overlap on one lens. All subs share the one M15 feature frame;
their events / zones / POIs / fibs / WVMI are append-only in the lens dfs,
keyed by `sub_id` (+ the sub's own `structure_id` — 0 for a single-structure
run — and `cycle_id`).

---

### Unified lifecycle & start/end model (Locked — REVISED 2026-05-26; sub rules restated to rev 2, Plan C 2026-09-20)

Applies to **both** main and subordinate. Starts are the primitives; ends
are **pass-throughs** of the next start's idx. Zones compute no end of
their own — they inherit their owning cycle's resolved end. A *structure*
is identified by `sid` on main and by `sub_id` (the unique sub) on a sub; a
sub additionally has parent-bound **instances** (`TriggerRecord`s, §17.4)
whose lifecycles aggregate into the sub's (§17.5).

```
Zones end when:              its cycle ends

Cycles end when:             1. next cycle of the SAME structure starts
                             2. its structure ends

Records end when (sub only): the FIRST of (earliest idx wins; at an equal idx
                             reversal > parent_end > same_dir_replacement):
                             1. own reversal      — the sub's natural_reversal_idx
                             2. same_dir_replacement — a new record of the same
                                (lens, parent_sid, parent_cycle_id, direction) for a
                                DIFFERENT sub starts (at ITS start_idx)
                             3. parent_end        — the record's own parent cycle ends
                                (end_m15[(S,C)], §17.6)

Structures end when:         main: next structure starts (sid+1 = reversal)
                             sub:  §17.5 — over the sub's non-zero-length records
                                   that EXIST at t (start_idx <= t): the earliest
                                   record end_idx strictly > max(start_idx), and
                                   nothing else. A sub does NOT end because a
                                   parent cycle / parent structure ends — only a
                                   RECORD does (parent_end), and the sub inherits
                                   it through the aggregation. A sub spans parent
                                   cycles and parent sids.

ANY new cycle starts when:   new CTS established — at the MOMENT it is established
                             (CTS_ESTABLISHED.meta["confirmed_at"] == BOS_CONFIRMED
                             .meta["confirmed_at"]; NEVER the CTS anchor —
                             CTS_ESTABLISHED.idx until Plan E E4a, the moment since)
                             — CLAMP: a cycle's lifecycle-start may not precede
                               its structure's lifecycle-start (main & sub). For a
                               sub the structure's lifecycle-start is the sub's
                               start_idx, which already embeds the parent floor
                               (below), so the cycle's true lifecycle-start is
                               max(cts_moment, structure lifecycle-start).

Next structure starts when:  main: reversal triggers (sid N's start = sid N-1's
                                   STATE_CHANGED→reversal idx)
                             sub:  a record's start_idx (§17.4) —
                                   max(probe_finalize_idx, trigger_idx, parent_floor_idx)
                                   where parent_floor_idx = floor_m15[(S,C)] =
                                   LOH(max(struct_start[S], cts_moment[(S,C)])) is the
                                   parent cycle's CLAMPED lifecycle-start; the SUB's
                                   start_idx = its first non-zero-length record's
                                   start_idx.

Zones start when:            the first time they become active (first confirmed),
                             clamped to their STRUCTURE's lifecycle-start (B1 —
                             "elements inherit the END only; they keep their own
                             START", below; the sub's start_idx for a sub).
```

- **`end_idx` = the idx of the start event that supersedes.** When a start
  fires at idx X, X becomes the `end_idx` of whatever it ends. Ends use the
  *clamped* next-start idx (matters only for subs/collapse — see below). For a
  record, `same_dir_replacement` is exactly this: the replacing record's true
  `start_idx`, not its `trigger_idx` or finalize.
- **Propagation closes the open child.** Whenever a *structure* ends, its
  currently-open *cycle* ends at the same idx, and that cycle's *zones* end
  with it. For a sub this is the projection's `lifecycle_cap = sub.end_idx`
  with `cap_reason = sub.end_reason` (§17.9): the last cycle of a sub that
  ended via a record's `parent_end` or `same_dir_replacement` still closes
  cleanly even though no "next cycle" event fired inside the sub's own MS run.
- **"Any new cycle" vs "next structure."** *Every* cycle start — whether the
  next cycle on the same sid, cycle 0 of a reversed main sid, or cycle 0 of a
  unique sub — is triggered by a new CTS established, so the cycle-start rule
  (the moment) and its clamp are universal. "Next structure" is deliberately
  narrower: main's `sid+1`, or a sub record's start. Under the pool a sub's
  own reversal does **not** start a "next sid" of the same structure — it
  spawns a **different** unique sub (opposite direction) with its own records
  (§17.5), and the reversing sub's records end `reversal`.

##### starting_idx vs lifecycle-start, and the clamp (REVISED 2026-05-26)

Each structure and each cycle has two distinct idxs (the §2 split,
generalized):

- **`starting_idx`** — the structural anchor: the retroactive
  BOS-equivalent candle the probe / `identify_start` selected. May sit
  *historically* before the entity is even alive.
- **lifecycle-start** — the first idx at which it is *active*:
  - **main structure:** sid 0 → its `starting_idx`; sid N≥1 → the reversal
    confirmation idx of sid N−1 (`STATE_CHANGED to=='reversal'`,
    `compute_reversal_idx_by_sid` → `compute_struct_start_by_sid`).
  - **sub — record vs sub (Plan C, 2026-09-20; supersedes the
    `start_trigger_idx` rule of 2026-05-26):**
    - **`TriggerRecord.start_idx = max(probe_finalize_idx, trigger_idx,
      parent_floor_idx)`** — REAL-TIME; **the record exists from here** and
      nothing earlier (it is not an incumbent, not an end candidate, not a
      lens member before it). All three terms are load-bearing (§17.4):
      - `probe_finalize_idx` (HISTORICAL) — when **this record's probe**
        finalized (`ProbeResult.finalize_idx`, raw; or the cached finalize on
        a probe-cache hit, inherited raw). Per `finalize_condition`: Phase-1
        `no_retrace` / `end_idx_reached` → `probe_end_idx`; Phase-2
        `second_cts_reached` → the 2nd `CTS_ESTABLISHED`'s moment
        (`meta["confirmed_at"]`, Plan B; the "double CTS", earlier than the
        parent-CTS bound); `reversal_in_probe` → reversal idx; Phase-2
        `no_retrace` → `CTS_0_CONFIRMED` idx **if cycle 0 confirmed inside the
        probe window, else `probe_end_idx`** (the else-branch is the COMMON case
        — measured after Plan A landed (2026-09-19, `debug/probe_fc_finalize.py`):
        all three `no_retrace` FCs on the 2025-11→2026-01 window take it —
        FC(1,0) 2843, FC(1,1) 3047, FC(1,2) 3621 = the price-mapped parent CTS
        anchor; FC(0,0) 1020 and FC(0,1) 2608 are `second_cts_reached`. Before
        Plan A FC(1,0) was 2844 via the if-branch, from a `CTS_CONFIRMED` the
        bounded MS had leaked one candle past its bound). For every non-FC type
        (`first_counter` / `subsequent_*` / `reversal`) `finalize_idx ==
        trigger_idx` by construction (asserted for a probe that ran). A
        probe-resolved structure is not *known* until its probe finalized, so
        this term is what starts an FC record honestly (FC(0,0): trigger 463 →
        start 1020).
      - `trigger_idx` (HISTORICAL) — the candle the trigger fired,
        `LOH(trigger_event_idx)`; native M15 for `reversal`. For a record
        whose probe actually ran this term is redundant (FC's `trigger_idx =
        LOH(BOS_CONFIRMED.confirmed_at) = LOH(cts_moment) ≤ parent_floor_idx`;
        every other type has `finalize_idx == trigger_idx`). It binds when the
        structure was already known before this trigger fired — a cache-hit
        record inheriting an earlier finalize (measured 2026-09-20: three such
        hits on the reference window, each absorbed by this term or the floor,
        §17.8).
      - `parent_floor_idx` = `floor_m15[(S,C)]` = `LOH(max(struct_start[S],
        cts_moment[(S,C)]))` — the parent cycle's **clamped lifecycle-start on
        the CTS-established moment** (`multitf/parent_tables.py`, §17.6). This
        single term is both old parent floors at once: `struct_start[S]` is the
        parent sid's reversal-handoff start, `cts_moment[(S,C)]` the parent
        cycle's established moment. Binds when the parent structure was not
        alive yet (an FC finalize under a later parent floor).
    - **unique sub `start_idx`** = its **first non-zero-length record's**
      `start_idx` (set once, in sweep phase 2; a sub with no live record has
      no `start_idx`, no lens, and is not rendered).
    - **what the sub passes downstream** (`render_sub_projection` →
      `project_to_window` → `_run_downstream_pipeline`): `lifecycle_floor =
      sub.start_idx`, `lifecycle_cap = sub.end_idx`, `cap_reason =
      sub.end_reason` (all slice-local — `− slice_begin`). KL / POI / fib see
      one int each; `compute_struct_start_by_sid(lifecycle_floor)` raises the
      sub's own `struct_start` to it; the sub's WVMI, computed in the same
      projection since Plan G, reads the same floor / cap, and its per-lens
      trigger-attribution window starts at the same `start_idx`
      (`LowerTFResult.meta["start_idx"]`, §17.10).
  - **cycle (any):** `max(cts_moment, owning-structure lifecycle-start)` where
    `cts_moment = CTS_ESTABLISHED.meta["confirmed_at"]` — **the moment the cycle
    was established, never the CTS anchor (the pattern extreme; `CTS_ESTABLISHED.idx` until
    Plan E E4a)** (Plan C,
    §17.6; `compute_cycle_lifecycle` Pass 1). The structure's lifecycle-start
    already embeds the parent floor for subs (through the record's
    `parent_floor_idx` → the sub's `start_idx` → `lifecycle_floor`), so the
    parent floor enters once (at the structure level) and cycles inherit it
    transitively.

  **Parent floor mapping (H1→M15, 2026-05-27; restated on the moment, Plan C
  2026-09-20).** A sub's parent floor is a parent-TF idx and is mapped to the
  entity's TF in `multitf/parent_tables.py::build_parent_tables`:
  - `parent_cycle_id lifecycle-start` = the parent cycle's **`CTS_ESTABLISHED
    .meta["confirmed_at"]`** (H1 — the moment; the 2026-05-27 text said
    `ev.idx`, then the anchor, a historical location), clamped
    `max(struct_start[S], cts_moment[(S,C)])` = `floor_h1[(S,C)]`, then mapped to
    M15 via **last-of-hour** (`_map_parent_idx_to_m15_hour_end`, LOH). Last-of-hour
    (not first) because the H1 candle isn't closed until its 4th M15 sub-candle,
    AND because the prior cycle's record *end* maps the same way (`end_m15[(S,C)]
    = LOH(floor_h1[(S,C+1)])`) → cycle k's floor lands on the same candle the
    prior cycle's records ended (shared boundary, no gap). In this engine
    `BOS_CONFIRMED.meta["confirmed_at"] == CTS_ESTABLISHED.meta["confirmed_at"]`
    (definitional, asserted in `build_parent_tables`), so the start anchor
    coincides with the prior-cycle end anchor.
  - `parent_sid lifecycle-start` = `struct_start[S]` from
    `compute_struct_start_by_sid` — reversal-aware for sid≥1 (sid N's start =
    sid N−1's `STATE_CHANGED→reversal` idx). It is folded into `floor_h1` above,
    not a separate term.

  **STATUS (2026-05-27 — history; superseded in mechanism by Plan C).** The
  `parent_sid` floor was effectively in force (auto-satisfied: a sub trigger is
  always within its parent sid's life). The `parent_cycle_id` floor was
  **IMPLEMENTED 2026-05-27 (plan B1)**: a shared `compute_struct_start_by_sid`
  pure-leaf helper (`zones/structure_lifecycle.py`) used by both KL + POI, and
  the then-live `build_one_sid`'s `lifecycle_floor` widened to
  `max(start_trigger_idx, parent_sid_start, parent_cycle_start)` (parent floors
  mapped H1->M15 last-of-hour in the then-live `build_parent_cycle_chain`).
  Validated /compare vs `310c395`: H1 + counter + WVMI + sids byte-identical,
  chart counts unchanged; only 5 `first_confluence` bootstrap subs floored their
  lifecycle-start (none collapsed). Only the `parent_cycle_id` term changed
  behavior. **Lifecycle-only:** `base_idx`, rectangle outline, BOS/CTS, the MS
  run, and `SidRecord.creation_event_idx` were all unchanged — only first-active
  / `confirmed_idx` / fill moved. The END-side unification (+ dedup of the
  reversal dict then duplicated across KL
  `_get_reversal_confirmed_by_sid_from_events` and POI inline
  `reversal_idx_by_sid`) is step "B2" — end-condition verification was **done**
  (2026-05-27) and the pass-through end model + Phase A/B plan are locked below
  in "End resolution as start-passthrough". Full writeup:
  `memory/project_cycle_lifecycle_parent_cycle_floor.md`.
  > **Plan C (2026-09-20):** `build_one_sid`, `build_parent_cycle_chain` and
  > `start_trigger_idx` are gone. The floor lives on the record
  > (`TriggerRecord.parent_floor_idx`, applied inside `start_idx`), the
  > parent tables are built once by `build_parent_tables` on the moment, and
  > `compute_struct_start_by_sid`'s `lifecycle_floor` for a sub is the unique
  > sub's `start_idx`. The B1 helper itself is unchanged.

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

- **Unifies KL and POI (Phase 3, 2026-05-26 — as landed then; POI's cycle term was `CTS_ESTABLISHED.idx`,
  the CTS anchor, until Plan D moved it to the moment on 2026-09-23 — see the §17.6 "As landed" blockquote).** POI already resolved
  `end_idx` + `end_reason` by the reversal / next_cycle priority and gated
  activation at `max(cts_established_idx, ic_idx)`. KL now uses the same
  end-resolution (its prior scattered `end_time` mechanisms — CTS_ESTABLISHED
  early-end, same-side replacement, reversal/lifecycle caps — are removed)
  and adopts the active/inactive/ended convention
  (`end_idx`/`end_reason`/`activation_history`/`status` in `meta`). Both
  zone kinds also floor first-active at the cycle lifecycle-start per the
  clamp. **End-side change (2026-05-26, as landed then):** both BOS and CTS
  zones of cycle *n* ended at the next cycle's `CTS_ESTABLISHED` `ev.idx` (the
  CTS extreme — the idx POI used). The CTS-zone end was unchanged (already this
  idx). The BOS-zone end moved from the old next-BOS-`confirmed_at` (breakout)
  to that extreme idx — identical when the breakout candle *is* the extreme,
  else earlier (1 candle observed; bound `confirmed_at <= meta["pattern_anchor_idx"] + 5`, the pattern anchor). Empirically the H1 main was byte-identical (breakout
  == extreme for all sampled cycles); the M15 subs (BOS-only zones) showed one
  1-candle BOS-end shift each. **Start-side change:** post-reversal cycle-0
  zones (and sub analogues) shift their active-start forward to the clamped
  lifecycle-start. See `zones/KL_ZONES_SPEC.md` "Lifecycle" and
  `ARCHITECTURE.md`.
  > **Plan C (2026-09-20) — the cycle boundary is the MOMENT, not the
  > extreme.** `compute_cycle_lifecycle` now starts every cycle at
  > `CTS_ESTABLISHED.meta["confirmed_at"]` (== `BOS_CONFIRMED.confirmed_at`), so
  > cycle *n*'s zones end at cycle *n+1*'s established **moment**. This undoes
  > the 2026-05-26 end-side move where extreme ≠ moment: the boundary is back
  > on the apply candle (which is the breakout candle for BOS), not the CTS
  > extreme. Where extreme == moment nothing changes (H1 on the reference
  > window: byte-identical). Measured on the M15 subs: one visible +1 shift
  > (sub `454/+1` cycle-1 end 1223 → 1224); the only other extreme ≠ moment
  > cycle (sub `2639/−1` cycle 1, 2828 → 2829 — two CSV rows because the sub
  > is mirrored into both lenses) is masked by the identical record floor 2829
  > (§17.11).
- **Scope of the unification.** The pass-through lifecycle model governs
  **zones and cycles** (KL + POI). It deliberately does **not** apply to
  structure events or patterns — those are immutable append-only facts whose
  only time-varying property is currency (H1 `owner_by_idx`; M15
  `owner_by_idx_dir` over the sub's lifecycle window, §16.5 rev 2), not
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
sub-cycle, own reversal) live in its own MS run while its instance ends
(`parent_end`, `same_dir_replacement`) live in the record layer fed by the H1
parent tables. So, exactly mirroring B1's start `lifecycle_floor`:

- **`lifecycle_floor`** (single int, `max`) — start floor. [B1]
- **`lifecycle_cap`** (single int, `min`) — end cap. [B2]

Both are `None` for main and supplied (slice-local) by the multitf layer for
subs — **after the sweep, at projection time** (Plan C: `render_sub_projection`
passes `floor = sub.start_idx − slice_begin`, `cap = sub.end_idx − slice_begin`
or `None`, `cap_reason = sub.end_reason` into `project_to_window`). The cap is
**load-bearing** for what terminates zones, but it is **no longer the run
bound**: the geometry runs to the data edge (§5 intro), and the cap clips the
shared geometry by knowable-at (`clip_events_to_window`) before the downstream
derivation. (Under the 2026-05-27 chain the cap was `end_m15_abs`, also the MS
slice bound, and had to be computed before the run in `build_parent_cycle_chain`
/ `build_one_sid` — both gone.)

**The data/window boundary is NOT a lifecycle terminator (2026-05-27).** A
lifecycle ends only on a **real event** — a reversal, or a genuine next
cycle/structure forming — never because the data ran out. The backtest's right
edge is just "the present" (in live, every moment between candles you sit at the
last available candle, yet active elements stay active). So:

- The **`lifecycle_cap` is a real structural end ONLY**: under the pool it is
  the unique sub's `end_idx`, which is set only by a record end that genuinely
  formed — the sub's own `reversal`, a `same_dir_replacement` by a later live
  record, or a record's `parent_end` (the next parent cycle's clamped start or
  the parent sid's reversal, `end_m15[(S,C)]`) that survives the §17.5
  aggregation. When no such end exists — the sub's live records are in the open
  last parent cycle (`end_m15[(S,C)] is None`), it has not reversed and has not
  been replaced — `sub.end_idx is None` and the cap is **`None`**. KL / POI / Fib
  then stay **active to the edge**, exactly like the H1 main's open last cycle
  (whose cap is always `None`). This makes subs consistent with main. (Rev 1
  keyed this on `trigger.lifecycle_end_idx is None` / `cap_open` in
  `build_parent_cycle_chain`; those are gone.)
- **Run bound vs lifecycle cap are separate.** The structure *runs* on
  `[starting_idx − 50, data edge]` (you can only run on data you have; the run
  cap is a compute bound); the projection's `LowerTFResult.meta["m15_end_idx"]`
  records `end_idx` or the edge for the mirror's column write. Only the
  lifecycle **cap** (what *terminates* zones) is `None` for an open sub. A
  failed H1→M15 map is an **assert** in `build_parent_tables` (§17.7), no
  longer a silent fall-back to the edge.

**Implementation (`zones/structure_lifecycle.py`).** A pure-leaf
`compute_cycle_lifecycle(events, reversal_idx_by_sid, lifecycle_floor,
lifecycle_cap, cap_reason) -> Dict[(sid, cycle), (start_idx, end_idx, end_reason)]`:
  1. *Pass 1 — clamped cycle starts:* `start = max(CTS_ESTABLISHED
     .meta["confirmed_at"], struct_start, floor)` — **the established MOMENT
     (Plan C, 2026-09-20; was `CTS_ESTABLISHED.ev.idx`, the anchor, until
     then)**; asserts that every `CTS_ESTABLISHED` carries `confirmed_at`
     (`struct_start` from `compute_struct_start_by_sid`, the B1 helper, which
     for a sub is already floored at the sub's `start_idx`).
  2. *Pass 2 — ends from next starts:* `end = min(next-cycle clamped start,
     reversal_idx_by_sid[sid], cap)`; `end_reason` records which won
     (`next_cycle` / `reversal` / `cap_reason`). `cap_reason` is the value the
     caller passes: for a sub the unique sub's `end_reason` ∈ {`reversal`,
     `same_dir_replacement`, `parent_end`}; the leaf's default string
     `"lifecycle_end"` is never reached in production (main passes no cap; a
     sub with a cap passes its `end_reason`). End uses the next cycle's
     **clamped** start (not its raw established moment); these differ only for
     collapsed sub cycles.
The reversal dict is built once by a shared `compute_reversal_idx_by_sid(events)`
(retiring the duplicated KL `_get_reversal_confirmed_by_sid_from_events` vs POI
inline `reversal_idx_by_sid`) and passed to both the start and the end helpers.
The same two helpers build the H1 parent tables for the sub layer
(`multitf/parent_tables.py`: `rev_by_sid`, `struct_start`, `cts_moment`,
`floor_h1 = max(struct_start, cts_moment)`, `end_h1 = floor_h1[(S,C+1)]` else
`rev_by_sid[S]` else None) — so the sub end-cap's next-cycle source and the
record's start floor are **the same rule on the same moment** as
`compute_cycle_lifecycle` itself. That is the B2 consistency argument, restated
on the moment: floor and cap come from one derivation and cannot diverge.

**Elements inherit the END only; they keep their own START.** When a cycle ends,
every attached element ends with it, so each KL / POI (later Fib / WVMI) looks up
its `(sid, cycle)` `end_idx` / `end_reason` from the table. **Nothing inherits
cycle start** — each element computes its own first-active by its own
active/inactive logic, clamped up to the cycle/structure floor (B1). BOS KL zones
keep computing their own breakout start ("Option 2", 2026-05-27), which since
Plan C *equals* the cycle start by construction (a BOS zone's confirm is
`BOS_CONFIRMED.meta["confirmed_at"]` = the cycle's moment) — the 2026-05-27
"≤1-candle extreme-vs-breakout divergence" disappeared when the cycle start moved
to the moment. POI's first-active takes the moment as its cycle term since Plan D
(2026-09-23). active/inactive state stays
element-specific (POI's per-candle `activation_history`, KL's single interval).

**Collapsed cycles** (clamped `start >= end`, e.g. a sub cycle whose
established moment precedes the sub's `start_idx` and whose next cycle's
established moment is also at or before that `start_idx` — both clamp to the
same floor) → `status = "inactive"`, empty `activation_history` (renders
outline-only). This standardized the prior split (2026-05-27): the inner KL
derivation said `"ended"`, the then-live `build_one_sid` cap said `"inactive"`.
Under the pool a *record* that would collapse is instead **zero-length**
(§17.4) and a *sub* with no live record is not rendered at all; cycle-level
collapse inside a rendered sub still follows this rule.

**Phase B (DONE 2026-05-27, byte-identical — history).** The parent-cycle /
parent-sid paths (the sub `lifecycle_cap`) ARE the *next parent `(sid,
cycle)`'s clamped start* (then taken as the canonical `CTS_EST.ev.idx`).
Previously the cap's next-cycle term came from `trigger.lifecycle_end_idx` =
next-parent-cycle `BOS_CONFIRMED.confirmed_at` (the apply candle) — a latent
overlap hazard, since the start-floor used `CTS_EST.ev.idx` and the two diverge
if a parent H1 cycle's CTS-extreme ≠ breakout candle. **Implementation
(approach A):** the four trigger detectors (`uc1`, `first_confluence`,
`subsequent_confluence`, `subsequent_counter`) computed the next-cycle term
from the next cycle's `CTS_ESTABLISHED.ev.idx` instead of
`BOS_CONFIRMED.confirmed_at`; the reversal term (`REVERSAL_CANDIDATE.apply_idx`)
was unchanged. `lifecycle_end_idx` was **kept** then (still the carrier read by
`_find_m15_lifecycle_end`); full retirement (sourcing the cap directly from
`parent_cycle_floor_h1`) was rejected at the time because it would also swap
the reversal-cap source and was not guaranteed byte-identical. Byte-identical
on this data because `confirmed_at == CTS_EST.ev.idx` for all 6 H1 cycles
(verified 0 mismatches) — a pure robustness fix, zero behavior change then.

> **Clarified 2026-09-19:** the definitional identity is
> **`BOS_CONFIRMED.meta["confirmed_at"] == CTS_ESTABLISHED.meta["confirmed_at"]`**
> — both are the same `apply_idx` (`market_structure.py` ~1414-1451; the BOS
> confirms at the candle that establishes the new cycle's CTS — one real-time
> moment). `CTS_ESTABLISHED.idx` was then a DIFFERENT thing (until Plan E E4a,
> 2026-09-25, made it the moment; the anchor is `meta["cts_anchor_idx"]`): the CTS **anchor** (the pattern extreme)
> within the pattern span, which can precede the apply candle (3 such pairs in
> the saved M15 event streams — 1223/1224, 2828/2829; 0 of 5 on H1 on this
> window). The hedge above is therefore *right* about `CTS_EST.ev.idx` vs
> `confirmed_at` (they can differ) and only wrong in attributing it to the
> migrating `cts_anchor_idx`. Consequence for the pool (as then planned): the
> parent-cycle floor was built on `CTS_EST.idx` (the extreme), FC's trigger on
> `confirmed_at` (the moment), so a `TriggerRecord` needs `trigger_idx` as a
> floor term (Plan C §2.1/§3) and the planned assert is on `confirmed_at`, not
> `.idx`.
>
> **Plan C (LANDED 2026-09-20) — Phase B completed on the moment.** The
> approach-A carrier is retired: `lifecycle_end_idx` on the four trigger
> dataclasses is **unread** (the field is kept for one commit for the
> detectors' tests; `MultiTFTrigger.lifecycle_end_idx` likewise — both deleted in
> Plan E E1b, 2026-09-24), and
> `_find_m15_lifecycle_end`, `parent_end_lookup`, `parent_struct_end_m15` and
> `parent_cycle_floor_h1` are deleted. Both the record's start floor
> (`parent_floor_idx = floor_m15[(S,C)]`) and the record's `parent_end`
> (`end_m15[(S,C)] = LOH(floor_h1[(S,C+1)])`, else `LOH(rev_by_sid[S])`, else
> None) come from **one** helper, `multitf/parent_tables.py::build_parent_tables`,
> on **`CTS_ESTABLISHED.meta["confirmed_at"]`** — the reversal term there is
> `STATE_CHANGED→reversal` (`compute_reversal_idx_by_sid`), NOT
> `REVERSAL_CANDIDATE.apply_idx` (a prediction that may not realise). The identity
> `BOS_CONFIRMED.confirmed_at == CTS_ESTABLISHED.confirmed_at` is asserted per
> cycle in `build_parent_tables` (never against `.idx`). The overlap hazard
> Phase B guarded against cannot arise: floor and cap are the same value from
> the same table. `trigger_idx` stays a floor term for the cache-hit case only
> (above).

---

## 6. Subordinate Cadence and Lifecycle Rules

> **⚠ REVISED by §17 (Sub-Structure Pool, rev 2 2026-09-19).** The
> per-parent-cycle merge-and-bound sid chain below is superseded: a unique sub
> spans parent cycles AND parent sids (its records are parent-bound, it is not),
> storage is one shared M15 frame with lens dfs populated by the mirror, and the
> two charts draw each sub over the sub's single window. The cadence driver
> (`build_two_entity_parent_cycle` / `_ChainCursor`) is **replaced** by the
> ordered sweep (§17.6). §6.1's "no bootstrap → no `subsequent_*`" rule is
> **retired** (§17.4: triggers are independent). §6.5's "a sub can't outlive
> its parent" holds for a *record*, not a sub (§17.2). See §17.4–§17.8.
> **Bodies rewritten to rev 2 in the Plan C commit (2026-09-20); the
> 2026-05-25 merge-and-bound text is in git history.**

(§4.3.6 gives the trigger cadence and the sweep's moment/phase order. This
section specifies the rules the sweep applies: what a trigger becomes, when an
instance ends, and how the unique sub aggregates. **REVISED 2026-05-25** — the
"overwrite semantics" framing, including the cascade, was removed then;
**REVISED 2026-09-20 (Plan C)** — the merge-and-bound sequential sid chain is
retired in favour of independent triggers → records → one sub lifecycle.)

### 6.1 Within parent cycle — sequential sids (merge-and-bound), no overwrite (REVISED 2026-05-25) → RETIRED: independent triggers, records are the instances, the sweep is the driver (Plan C 2026-09-20)

**The merge-and-bound chain is retired.** There is no per-`(entity,
parent_sid, parent_cycle_id)` sid sequence, no "sid=0 is always a
first-variation", and no "if the first variation never yields a valid sub the
cycle's `subsequent_*` triggers do not build" rule. Under the pool:

1. **Triggers are independent.** Every H1 trigger of every parent cycle
   (`first_confluence`, `first_counter`, `subsequent_confluence`,
   `subsequent_counter`) is resolved on its own at its `trigger_idx` (sweep
   phase 0, §4.3.6): degenerate-cycle check (§17.7) → probe → geometry. A
   `subsequent_*` trigger is processed even when its cycle's `first_*` trigger
   was unresolved or produced a zero-length record — `subsequent_confluence`
   reads the *counter* sibling, not its own bootstrap.
2. **Records are the instances.** A resolved trigger becomes a `TriggerRecord`
   `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)` → `sub_id` (§17.4).
   `trigger_sub_sid` starts at 0 per `(lens, parent_sid, parent_cycle_id)` and
   increments per **new unique sub** in that scope, in creation order — an FC
   record is created at its `trigger_idx` and may start hundreds of candles
   later. A later trigger in the same `(lens, parent_sid, parent_cycle_id)`
   whose probe lands on the **same** sub is absorbed into the existing record
   (`extra_trigger_idxs`); no new record, no `trigger_sub_sid` consumed.
3. **The record exists from `start_idx = max(probe_finalize_idx, trigger_idx,
   parent_floor_idx)`** (§5). Same-direction consecutive records ARE allowed —
   a `subsequent_*` trigger of the same direction as the live record starts a
   new record for a **different** sub and ends the incumbent by
   `same_dir_replacement` (§6.2 / §17.4 end condition 2).
4. **The sweep is the driver** (`run_lifecycle_sweep`, §17.6): the priority
   queue of moments processes every trigger, record start, record end and sub
   end in `(idx, phase, order_key)` order; nothing is built "to the next
   trigger" — geometry runs once per unique sub to the data edge, and lifecycle
   is evaluated on the records.

**No overwrite, no bounds-capping, no cascade** (unchanged since
2026-05-25). "Only the most recent sub of a direction is alive on a lens"
falls out of the record rule: ≤ 1 active record per `(lens, parent_sid,
parent_cycle_id, direction)` (asserted). Old subs' events / zones / POIs /
fibs / WVMI persist append-only with `sub_id` attribution; their ends are the
resolved lifecycle ends, not a cascade-imposed cap.

**Degenerate / short-lived triggers.** A trigger in a **degenerate parent
cycle** (floor ≥ end, §17.7) is logged as `UnresolvedTrigger` and builds
nothing (no probe, no MS, no `sub_id`). A trigger in a live cycle whose record
would be empty (its end condition ≤ its `start_idx`) becomes a **zero-length
record** — it has a `sub_id` but participates in nothing (§17.4). Otherwise a
short-lived record is valid while active (live-trading semantics); no min-span
skip.

**Handoff idx.** A record's lifecycle start is its `start_idx` (when the
instance comes into existence in real time); its `starting_idx` is the earlier
probe-validated anchor. The incumbent record of the same `(lens, parent_sid,
parent_cycle_id, direction)` stays active until that `start_idx` (its
`trigger_end_idx` = the new record's `start_idx`, reason
`same_dir_replacement`, `ended_by_sub_id` = the new sub). See §2
(starting_idx vs lifecycle-start).

### 6.2 Across parent cycles — one df, lifecycle-bounded (REVISED 2026-05-25) → record end conditions (§17.4)

There is **one shared M15 feature frame** for the session and one lens df per
chart (`H1.main >> M15.confluence`, `H1.main >> M15.counter`) populated by the
mirror (§9). Every sub of every parent cycle lives in the pool with `sub_id`
identity; `parent_sid` / `parent_cycle_id` are **record** attributes.

**A record ends at the first of three conditions** (earliest idx wins; at an
equal idx `reversal > parent_end > same_dir_replacement`; write-once):

1. **`reversal`** — the sub's own `natural_reversal_idx` (known at build time
   because geometry runs to the data edge).
2. **`same_dir_replacement`** — a new record of the same `(lens, parent_sid,
   parent_cycle_id, direction)` for a **different** sub starts; the incumbent
   ends **at the new record's true `start_idx`** — same-lens only (a counter
   record never ends a confluence record).
3. **`parent_end`** — the record's own parent cycle ends: `end_m15[(S,C)]` =
   `LOH(floor_h1[(S,C+1)])` (the next cycle's **clamped** start on the
   CTS-established **moment**), else `LOH(rev_by_sid[S])` (the parent sid's
   `STATE_CHANGED→reversal` idx), else None (open) — `multitf/parent_tables.py`,
   §17.6.

`trigger_end_idx` holds the condition's idx (or the sub's frozen end on a
post-end re-trigger); `end_idx = max(trigger_end_idx, start_idx)`;
`is_zero_length = trigger_end_idx is not None and trigger_end_idx <= start_idx`.

**When parent cycle K ends**, every record in K ends `parent_end` at
`end_m15[(S,K)]` — but the **sub** does not necessarily end: a record of the
same sub in cycle K+1 that started **at** that idx keeps the sub continuous
(§6.5 / §17.5: sub `2365/+1` — record (0,0) `[2470, 2611] parent_end`, record
(0,1) `[2611, 2829] reversal` → one sub `[2470, 2829]`). Parent cycle K+1's
records start at `trigger_sub_sid = 0` in their scope. There is no
cross-parent-cycle overwrite and no capping of K's geometry at the boundary —
K's records simply end, and the sub's projection is capped at the **sub's**
`end_idx`. (A K+1 `first_confluence`'s `starting_idx` may physically anchor
inside cycle K's territory — that is the structural anchor only; its record
starts at `max(probe_finalize_idx, trigger_idx, floor_m15[(S,K+1)])`.)

Consumers scope to a parent cycle through the **record table**
(`attrs["triggers"]`, `pool.records_for(lens, parent_sid, parent_cycle_id)`),
never through a snapshot's informational `parent_sid` / `parent_cycle_id`.

### 6.3 Sub reversal mid-cycle — IS the sid boundary, does not perturb parent cadence (REVISED 2026-05-25) → successor spawn (§17.5)

A sub's own reversal is the boundary between the reversing **sub** and its
opposite-direction **successor sub** — NOT an internal multi-structure roll
within one run, and not a "sid+1" of the same structure:

- The MS run of a unique sub **stops at its first reversal** (§5);
  `natural_reversal_idx = R` is known at build time. Every record of the sub
  ends `reversal` at `R` (end condition 1).
- **Successor spawn rule (§17.5):** at `R` (sweep phase 0, `REVERSAL_SPAWN`,
  sorted before any `TRIGGER_FIRE` at the same idx) one `reversal` trigger is
  synthesised **per record of the reversing sub that is live at `R`**
  (`is_live_at_reversal`: non-zero-length, `start_idx <= R`, and
  `trigger_end_idx is None or trigger_end_idx >= R` — a record whose parent
  ends *at* `R` is still live at `R`): `(lens = r.lens, r.parent_sid,
  r.parent_cycle_id, trigger_type = "reversal", trigger_idx = R, direction =
  −sub.direction)`, with a `MultiTFTrigger` synthesised from that record's own
  `source_trigger` (`_synth_reversal_trigger`). Its probe is the reversal
  handoff over the reversing sub's own geometry (`_resolve_reversal_start`,
  §4.2). Two live records on two lenses ⇒ two triggers ⇒ they dedup into ONE
  successor sub with a record per lens (`4027/+1` on the reference window).
  **If no record is live at `R` there is no successor** — the reversal is
  inert. Successor record `start_idx = R` (finalize = trigger = R; floor ≤ R
  because the spawning record is live).
- The reversal **does not compete with** the next `subsequent_*` trigger for
  a boundary any more: a later `subsequent_*` trigger is detected from parent
  events, resolves independently (§6.1), and — if it lands on a different sub
  of the same direction while the successor's record is active — ends that
  record by `same_dir_replacement`.
- Parent's variation cadence is unaffected — the parent's proximity state
  machine and event emissions continue regardless.
- The successor record's `lens` is the spawning record's lens
  (sticky-per-chart, §17.3); its `relative_dir` is recomputed from its own
  direction vs the parent sid's direction (`2639/−1` under H1 sid 0 (+1):
  `relative_dir = counter` on the **confluence** chart — intended).
- WVMI (§17.10, Plan G 2026-09-30): computed inside each unique sub's
  projection, ungated, over the same floor / cap as its zones; each lens
  copy's trigger metadata names the first WVMI-class parent trigger of THAT
  lens inside the sub's `[start_idx, end_idx]` (the data edge for an open sub; attribution only, None if
  none). There is no "current sid" any more; `started_by` on a snapshot is
  the sub's first live record's `trigger_type`.

### 6.4 Parent reversal — bootstrap rule for new parent cycle

When parent reverses (`STATE_CHANGED→reversal` of parent sid `S` at idx
`rev_by_sid[S]`):

- Records in the **last** cycle of sid `S` end `parent_end` at
  `LOH(rev_by_sid[S])` (no `(S,C+1)` exists, so `end_h1` falls through to the
  reversal idx); records in earlier cycles already ended at their next cycle's
  clamped start.
- The new parent sid `S+1` has `struct_start[S+1] = rev_by_sid[S]` (reversal
  handoff), so every cycle of `S+1` established **before** that idx has
  `floor_h1 = max(struct_start, cts_moment) = struct_start` — retroactive
  cycles. Where the next cycle's floor equals this floor the cycle is
  **degenerate** (floor ≥ end, §17.7: on the reference window (1,0) and (1,1),
  established 703 / 748 under `struct_start = 902`); its triggers are logged
  `degenerate_parent_cycle` and nothing is built. Only the cycle in progress
  at the reversal (and later cycles) is real.
- No retroactive building for past cycles of the new parent's sid — by
  construction of the floor, not by a special case.

### 6.5 Lifecycle propagation

Parent cycle ending (the next cycle's clamped start on the moment, or the
parent reversal) ends the parent's **direct instances**, and a record's end
propagates down within its sub:

- Every record in that parent cycle ⇒ ends `parent_end` at `end_m15[(S,C)]`
  (§6.2). **The unique sub does not end unless the §17.5 aggregation says so**
  (a same-sub record in the next cycle that started at that idx keeps it
  alive) — "a sub can't outlive its parent" is a **record** property.
- When the sub does end (`sub.end_idx` / `sub.end_reason` from the earliest
  record `end_idx` strictly greater than the latest started record's
  `start_idx`, priority-tied), the projection caps its open cycles / zones /
  POIs / fibs at that idx with `end_reason = sub.end_reason` ∈ {`reversal`,
  `same_dir_replacement`, `parent_end`} (the projection's `lifecycle_cap` /
  `cap_reason`, §5). The rev-1 tag `deactivated_by="lifecycle_end"` no longer
  exists.
- Grandchildren (an M5 under an M15 sub) are not built (§17.12); when they
  are, the same record rule applies one level down (their `parent_end` is the
  M15 sub's record end).
- WVMI under the pool (§17.10, Plan G): a record locks only at its cycle's
  successor BOS (knowable at the cap); a cycle the cap or a reversal ends
  stays unlocked with its temp LP bounded to the cycle's `end − 1`; a lock LP
  past the cap falls back to the temp LP. Computed over the sub's window and
  persisted into every lens df the sub is on, each copy with that lens's path.

---

## 7. Persistence Model — What Overwrites vs What Persists

Persistence rules apply *within* an entity's df. Each entity's df is
isolated from every other entity's df.

(REVISED 2026-05-25 — no bounds-capping cascade. **Plan C 2026-09-20:** for
the M15 lens dfs the "owning sid" is the unique sub (`sub_id`); the mirror
writes subs' structure columns in `start_idx` order, so a later-live sub wins
an overlapping candle — the only overlap possible is between subs of
**opposite** direction, §17.5/§17.9.)

| Data type | Storage | Policy |
|---|---|---|
| df columns (`structure_id`, `cycle_id`, `cts_phase`, `range_lo/hi`, etc.) | per-candle | Main: written once per candle by the owning sid. Lens dfs: mirrored from each sub's projection in `start_idx` order (later-live wins). |
| Structure events (`StructureEvent` list) | `df.attrs["events"]` (append-only) | **Persist forever** with `sub_id` (identity) + informational `parent_sid` / `parent_cycle_id` / `use_case` + cycle attribution; never deleted or mutated |
| Zones (KL, POI), Fibs, Wave candles, Imbalances | `df.attrs[...]` keyed by `sub_id` + cycle_id | Persist; `end_idx` resolved via the §5 lifecycle model (next cycle / reversal / the sub's `lifecycle_cap` with `end_reason = sub.end_reason`). No `overwritten_by` tagging, no bounds-capping. |
| WVMI records | `df.attrs["wvmi"]` keyed by `sub_id` + cycle_id | Persist as snapshots: computed in each unique sub's projection (Plan G), one copy per lens df the sub is on with that lens's path + trigger metadata; `cycle_collapsed` marks a record whose cycle's window is empty (§17.10) |
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

> **Plan G (landed 2026-09-30, `plans/PLAN_G_wvmi_unique_sub.md`) — sub WVMI lives on the unique sub, like
> zones (§17.10).** Every rendered sub's CTS_CONFIRMED gets a record, computed inside its projection (no gate),
> bounded by the cycle lifecycle table (temp LP to `end − 1`, `cycle_collapsed` = an empty window); the mirror
> persists one copy per lens df the sub is on, each with that lens's path; each lens copy's
> `triggered_by_event_idx` / `_type` name the lens's first WVMI-class trigger (§8.5's classes) inside the sub's
> window — ATTRIBUTION, None when none lands there. §8.3 / §8.4 / §8.5's "initiate a sweep at each trigger" and
> §8.6's lock table are superseded (dated design below); the main (§8.2) keeps its first-sd gate. Plan C
> (2026-09-20 → 30) had one trigger-gated sweep per unique sub, persisted into every lens with the SWEEPING lens's
> path on the field (the counter rows' column contradicted their meta).

WVMI now exists per structure entity, keyed by
`(structure_path_id, sid, cycle_id)`.

### 8.1 Removed: ad-hoc gate

- **Not as written (2026-09-30):** the main KEPT its first-sd gate (§8.2,
  `_first_sd_prox_gate`); what Plan G removed is any gate on SUB WVMI.
  Original text: the "first sd zone-proximity trigger gates WVMI creation at parent
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

> **Superseded by Plan G (§8 banner, §17.10):** no sweep is "initiated" at a trigger — every cycle of a rendered
> sub gets its record in the projection; the triggers below are the confluence lens's ATTRIBUTION stream.

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

> **Superseded by Plan G** likewise: var 3 is the counter lens's attribution stream.

- Triggered each time `subsequent_confluence` (var 3) fires.
- First counter WVMI calc happens at the first var 3 fire (= first parent
  CTS-prox after the parent first sd-prox).
- Otherwise mirrors §8.3 with "counter" substituted.

### 8.5 Cadence summary

> **Since Plan G** the right column reads "the lens whose rows this trigger ATTRIBUTES" (`_wvmi_trigger_streams_by_lens`:
> confluence = sd-prox class, counter = CTS-prox class — Q8, confirmed after the cold review showed the class is
> cross-lens); only the main's WVMI is still created at its trigger.

| Trigger event | WVMI calcs initiated |
|---|---|
| Main first sd-prox after main CTS | Main WVMI + Confluence sub WVMI sweep on current confluence sid |
| Var 3 (parent CTS-prox after sd-prox) | Counter sub WVMI sweep on current counter sid |
| Var 4 (parent sd-prox forming Λ/V) | Confluence sub WVMI sweep on current confluence sid |

Symmetry: confluence sub WVMI fires on sd-prox-class events; counter sub
WVMI fires on CTS-prox-class events. Each side's WVMI is initiated at the
*opposite* side's trigger moment.

### 8.6 Lock semantics (per-cycle)

> **Not implemented as tabled; replaced by Plan G (2026-09-30).** A record locks ONLY at its cycle's successor
> `BOS_CONFIRMED` (`locked_by_cycle_id`, main and subs); a cycle ended by a reversal or a sub's cap is never locked
> — it keeps its temp LP, bounded to the cycle's lifecycle `end − 1`; a lock LP outside the tracker's frame falls
> back to the temp LP. No `locked_by` meta exists. The dated table:

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
- **Plan G (2026-09-30):** on a SUB row `triggered_by_event_idx` / `_type` mean "the lens's first WVMI-class
  trigger inside the sub's window" (attribution, stamped per lens after the mirror; both None when none — the
  exporter writes the idx as nullable `Int64`), `parent_path_id` is always `"H1.main"`, and they are the meta's
  FIRST keys (a declared rule-3 meaning change; main rows keep the gate meaning). The record FIELD
  `structure_path_id` of a lens copy == its meta's. New field `cycle_collapsed` (bool): the record's cycle has an
  empty lifecycle window — exported, inert, like a collapsed cycle's zones (main and subs).

---

## 9. Storage Architecture — Per-Entity DFs + Registry

> **⚠ REVISED by §17 (Sub-Structure Pool, rev 2 2026-09-19) for subordinates.**
> Structural storage is **one** shared M15 feature frame carrying the pool of
> unique subs (keyed `(parent_path, sub_TF, direction, starting_idx)`, single
> monotonic `sub_id`; geometry slice-local + `slice_begin`). The confluence /
> counter **lens dfs** still exist but only as mirror targets — each sub's one
> projection is mirrored into every lens df it belongs to, for the chart and
> export readers (§17.9). Attribution on every mirrored event/zone: `sub_id`
> (identity) + informational `parent_sid` / `parent_cycle_id` / `use_case` from
> the sub's first record. `H1.main` is unchanged. See §17.9. **Body rewritten
> to rev 2 in the Plan C commit (2026-09-20).**

**Single store for subs (Plan C).** `_run_multi_tf_dual` prepares **one**
shared M15 feature frame (`prepare_lower_tf_data` once: candles + imbalance +
volume features) that carries the pool — every probe runs on it, every
sibling read resolves against it, and every unique sub's geometry is a slice
of it (`slice_begin` + 50-candle lookback, §5). The two **lens dfs** are
`m15.copy()` each (`H1.main >> M15.confluence`, `H1.main >> M15.counter`),
populated **only** by `mirror_lower_tf_result_to_entity_df` from each sub's
single projection (`render_sub_projection`, once per sub, into every lens in
`sub.lenses()`). They are views for the chart / export readers, not separate
structural storage: no probe, MS run or derivation reads a lens df. The pool
itself and the parent tables are exposed as `meta["sub_pool"]` /
`meta["parent_tables"]` on the pipeline result.

### 9.0 Terminology

| Term | Meaning |
|---|---|
| **entity** | Unit of structural isolation for the chart / export readers. Identified by `structure_path_id`. `H1.main` owns one df + its stateful objects (MarketStructure, FibTracker, WVMITracker). An M15 **lens** entity owns one lens df (a mirror target) — the structural state of subs lives in the pool, not per lens. Created lazily, persists for the session. |
| **sid** | One run/instance within `main`: each reversal increments `sid` (`structure_id`). For subs the analogue is the **unique sub** (`sub_id`, §2); a sub's own MS run has `structure_id = 0`. |
| **record** | A `TriggerRecord` — a parent-bound instance of a unique sub, `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)` → `sub_id` (§17.4). |
| **cycle / cycle_id** | One CTS-to-CTS span within a sid / sub (a sub or main internally manages cycles). |
| **parent_cycle_id** | On a **record**: which parent cycle the trigger fired in. Determines that record's floor and `parent_end` (§6.2) and the sibling-read scope (§17.8). On a snapshot it is an informational copy from the sub's first live record. |
| **structure_path_id** | The string key identifying an entity (e.g., `H1.main >> M15.counter`). Pure structural path; no sids/cycles in the path itself. For a sub it is the **lens** path the snapshot was mirrored into. |

**Per-entity instantiation:** `H1.main` gets its own `MarketStructure`,
`FibTracker`, `WVMITracker` instance. Each unique sub gets its own bounded MS
run (`compute_bounded_structure`) and its own downstream derivation at
projection time; nothing is shared across subs except the read-only feature
frame. This is the foundation of isolation — sibling or parent entities cannot
accidentally read or mutate each other's state through a shared singleton
(geometry objects are shared between a sub's records and its lenses and are
therefore never mutated in place — the projection and the mirror deep-copy
what they stamp).

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
- `starting_alignment: "confluence" | "counter" | None` — sub only; under
  the pool this is the **lens** of the lens df (`registry.register(...,
  starting_alignment=lens)`), not a structure property (§2 / §17.3)
- `timeframe: str`
- `role: "main" | "subordinate"`

Per-sid / per-sub attribution lives **in the df**, not on `EntityState`
(`SidRecord` in `multitf/types.py` — ONE frozen class, two roles; Plan C):
- **`df.attrs["sids"]` on `H1.main`** (`build_sid_records_for_main`): one
  `SidRecord` per `structure_id` — `sub_sid = structure_id`, `sub_id = None`,
  `starting_sd`, `creation_event_idx` (first event idx), `end_event_idx`
  (the realised reversal, `STATE_CHANGED(to=reversal)` via
  `compute_reversal_idx_by_sid` — since 2026-09-28; it was the last
  `REVERSAL_CANDIDATE.apply_idx`, a scheduled apply), `end_reason`
  (`"reversal"` | None), parent fields None. Chart identity = `sub_sid`.
- **`df.attrs["sids"]` on a lens df** (`build_sid_records_for_subordinate`):
  **one `SidRecord` per unique sub rendered on that lens**, in `start_idx`
  order — `sub_id` set, **`sub_sid = None`**, `parent_sid = None`,
  `parent_cycle_id = None` (parent attribution lives on the record table),
  `starting_sd = direction`, `creation_event_idx = starting_idx` (the
  structural anchor, historical), `start_idx` (real-time lifecycle start),
  `end_event_idx = end_idx` (real-time; None while open), `end_reason` ∈
  {`reversal`, `same_dir_replacement`, `parent_end`, None}, `lenses`,
  `relative_dir_segments`, and `meta = {natural_reversal_idx, n_records,
  first_record: {lens, parent_sid, parent_cycle_id, trigger_type, trigger_idx,
  start_idx}, slice_begin}` (`validated_parent_start` deleted in Plan E E1b —
  unread). Chart identity = `sub_id`
  (`_sid_record_identity`: `sub_id` when set, else `sub_sid`).
- **`df.attrs["triggers"]` on a lens df**: the `TriggerRecord`s whose `lens`
  is this lens, **including zero-length ones**, creation (`seq`) order —
  every §17.4 field (`source_trigger` is internal, never exported).
- **`df.attrs["unresolved_triggers"]` on a lens df**: the pool-wide
  `UnresolvedTrigger` list (§17.7) — the same list on every lens df.
- Events / zones / POIs / fibs / wave candles / WVMI records in a lens df all
  carry **`sub_id`** (identity) + `structure_path_id` / `timeframe` /
  `parent_tf`, plus — informational only, from the sub's first live record —
  `parent_sid`, `parent_cycle_id`, `use_case`, `started_by`; and `cycle_id`
  meta to support per-cycle filtering. No consumer may group or join on the
  informational three.

**Exports (`debug/export_sub_tables.py`, called by `run_replay.py` BEFORE
the M15 chart loop in its own try/except, so a chart-export exception cannot
lose them):**

| File | Rows |
|---|---|
| `{basename}_M15_{lens}_subs.csv` | one row per sub on the lens: `sub_id, direction, starting_idx, start_idx, end_idx, end_reason, natural_reversal_idx, lenses, relative_dir_segments, n_records, first_record_lens, first_record_parent_sid, first_record_parent_cycle_id, first_record_trigger_type, first_record_trigger_idx` |
| `{basename}_M15_{lens}_triggers.csv` | one row per record on the lens (incl. zero-length): every `TriggerRecord` field except `source_trigger`, plus `is_zero_length` |
| `{basename}_M15_unresolved_triggers.csv` | one row per unresolved trigger (pool-wide, written once) |

`{basename}_M15_{lens}_sids.csv` is **removed** (its role is split across the
two per-lens files above). A header row is always written, even for an empty
table. The events / zones / POI / fib / WVMI exporters carry `sub_id` (the
WVMI exporter as a first-class column, formerly `sub_sid`; the others inside
the serialised `meta`).

### 9.3 Per-entity df contents

| Path | Contents |
|---|---|
| df columns | TF candles (a copy of the shared M15 feature frame for a lens df) + `structure_id`, `cycle_id`, `cts_phase`, etc. (lens df: mirrored per sub in `start_idx` order) |
| `df.attrs["structure_path_id"]` | This entity's id |
| `df.attrs["parent_path_id"]` | None for main |
| `df.attrs["sids"]` | Main: per-sid records (`sub_sid = structure_id`). Lens df: **one `SidRecord` per unique sub on this lens** (`sub_id`, `sub_sid = None`, parent fields None, `creation_event_idx = starting_idx`, `start_idx`, `end_event_idx = end_idx`, `end_reason`, `lenses`, `relative_dir_segments`) — §9.2 |
| `df.attrs["triggers"]` (lens df only) | This lens's `TriggerRecord`s incl. zero-length (§17.4) |
| `df.attrs["unresolved_triggers"]` (lens df only) | Pool-wide `UnresolvedTrigger` list (§17.7) |
| `df.attrs["events"]` (lens df) / `df.attrs["structure_events"]` (main — a different key) | Append-only events list — main: all sids with `structure_id`; lens df: every sub mirrored into this lens, clipped by knowable-at to the sub's `end_idx`, with `sub_id` + informational parent fields + `cycle_id` meta |
| `df.attrs["kl_zones"]`, `["poi_zones"]`, `["wave_candles"]`, `["fib_states"]`, `["wvmi"]`, `["prev_bos_lines"]`, `["imbalances"]` | All entity-local; `sub_id` + cycle keyed on a lens df (lifecycle-bounded by the sub's window, no cascade). `["imbalances"]` on a lens df is the shared frame's full-window list (residual, LANDMINES) |
| `df.attrs["zone_proximity_triggers"]` | Main's own triggers (the sub triggers consume them via the detectors) |

### 9.4 Cross-entity lookups

When a sub needs parent zones (`first_confluence`'s BOS anchor input; the
main's proximity triggers that fire the counter/subsequent variations) it
reads them **at trigger-detection time on H1** (`first_confluence_trigger`,
`uc1_trigger`, `subsequent_*_trigger`) and carries what it needs on the
`MultiTFTrigger` (`parent_sid`, `parent_cycle_id`; meta `parent_input_idx`
(H1, all four types), `parent_cts_anchor_idx` (H1, FC only), `trigger_event_idx`, `prior_*`
(subsequent_*)). The record's own `parent_sid` /
`parent_cycle_id` scope every later cross-lens read:

```python
# sibling read (§17.8) — the pool, scoped by the RECORD's parent cycle
recs = pool.records_for(other_lens, rec.parent_sid, rec.parent_cycle_id)
# parent tables (§17.6) — the record's floor / parent_end
floor = tables.floor(rec.parent_sid, rec.parent_cycle_id)
end   = tables.end(rec.parent_sid, rec.parent_cycle_id)
```

A snapshot's informational `parent_sid` / `parent_cycle_id` (first live
record) is for hover / tier context only (`_sid_parent` in the M15 chart);
a lookup keyed on it would be wrong for a sub that spans parent cycles. No
snapshot copying. Parent is the source of truth.

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
candle data is the same. **Done for the structural side by Plan C
(2026-09-20):** the features are computed once on the shared M15 frame
(`prepare_lower_tf_data`) and every probe / geometry build reads that one
frame; the two lens dfs are `m15.copy()` mirror targets whose feature columns
are never recomputed. Sharing the lens dfs' feature columns by reference
(rather than by copy) remains deferred — isolation correctness wins until
performance forces a change.

---

## 10. Event Routing Layer

When a parent's MarketStructure emits an event that may trigger a child
variation, an event bus dispatches it.

### 10.1 Trigger subscription matrix

| Parent event | Subscriber |
|---|---|
| `BOS_CONFIRMED` | `first_confluence` (var 1) |
| `CTS_CONFIRMED` | resolves var 1's NULL `parent_cts_anchor_idx` if pending |
| First sd-zone proximity trigger after parent CTS | `first_counter` (var 2) + main WVMI (its gate) + the confluence lens's sub-WVMI trigger attribution (Plan G) |
| Subsequent sd-zone proximity trigger forming Λ/V | `subsequent_counter` (var 4) + the confluence lens's sub-WVMI trigger attribution |
| CTS-zone proximity trigger after sd-prox | `subsequent_confluence` (var 3) + the counter lens's sub-WVMI trigger attribution |
| `REVERSAL_CONFIRMED` (`STATE_CHANGED→reversal`) | parent cycle ends → every child **record** in it ends `parent_end` (§6.5 / §17.4; the rev-1 `lifecycle_end` reason no longer exists); the child **sub** ends only through the §17.5 aggregation |

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
   derive zones, fibs, POIs — and WVMI, ungated, since Plan G).
6. WVMI: the main's gated record at its trigger; a sub's per-lens trigger
   ATTRIBUTION is stamped onto its (already computed) records.
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
     >
     > **⚠ SUPERSEDED AGAIN — Plan C, 2026-09-20 (§17).** The merge-and-bound
     > chain that replaced this substep is itself gone: `build_parent_cycle_chain`,
     > `build_one_sid`, `build_two_entity_parent_cycle`, `_ChainCursor`,
     > `_find_m15_lifecycle_end`, `lower_tf_pipeline.py` and the `lifecycle_end_idx`
     > carrier are deleted / unread. The live model is the sub-structure pool:
     > `multitf/parent_tables.py` → `multitf/lifecycle_sweep.py` (records +
     > unique subs) → `entity_df_mutation.build_or_get_geometry` /
     > `render_sub_projection` → the mirror. Every `entity_sid` /
     > `(parent_sid, parent_cycle_id, sub_sid)` identity mentioned in this
     > migration log is now `sub_id` (§2, §9.2, §17.9). The mirror
     > (`mirror_lower_tf_result_to_entity_df`) survives from c.i as the ONLY
     > writer of a lens df — one call per (sub, lens), from the sub's single
     > projection, not per trigger. This log is history; do not implement from
     > it.

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
       >    "deprecated" `s_res.df.attrs[...]` writes) was CLOSED WITHOUT
       >    DELETION on 2026-09-30 — those writes are the `H1.main` entity's
       >    §9.3 store; see the §13.5.e bullet below.
       > 4. **Plan C (2026-09-20).** The chart key is now **`sub_id`**
       >    (`_sub_identity` / `_sid_record_identity`), grouping every attrs
       >    list by `meta["sub_id"]`; `_compute_owner_by_idx` became
       >    `_compute_owner_by_idx_dir` — `owner_by_idx_dir[(candle,
       >    direction)]` over each sub's real-time window `[start_idx,
       >    end_event_idx or edge]` (§16.5 rev 2), not its anchor. The
       >    `(parent_sid, parent_cycle_id, sub_sid)` tuple of note 1 is gone
       >    from all its sites; tier context (`_compute_m15_tier_context_from_sids`)
       >    reads each sub's `meta["first_record"]`. `SidRecord.end_event_idx`
       >    for a sub is the sub's lifecycle `end_idx` (the swing-line
       >    extension stops at the last owned candle as before).

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

     > **CLOSED 2026-09-30 — the positional fallback deleted earlier; the
     > `attrs` block KEPT (user decision).** Measured before acting: the
     > registry's `H1.main` entity is `s_res.df` itself (`entity.df is
     > res.df`), `EntityState` holds only identity + the df (§9.2, locked),
     > and every entity's artifacts live in its df's `attrs` (§9.3). So the
     > block is the registry's ONLY store for `H1.main`, not a copy beside
     > it: both charts already read it through the registry
     > (`export_chart_plotly` → `registry.get(path_id).df`;
     > `export_m15_chart_plotly` → `registry.parent_of(path_id).df`), and
     > deleting it would have blanked every H1 overlay on all three charts.
     > The task was written when the registry was expected to own a
     > separate store; §9.2 later fixed it as identity + df. Landed instead:
     > the stale "DEPRECATED" comment relabelled; two dead writes in
     > `run_replay.py` deleted (`attrs["structure_levels"]` — no reader;
     > `attrs["kl_zones"]` re-assigned to the same object); this spec's
     > §9.3 main-events key corrected (`structure_events`). No object
     > changed, so the replay is byte-identical (24 CSVs + figure JSON) —
     > the "per-row CSV parity may shift" allowance was never needed.
     > Remaining duplicate channel (accepted): `res.meta[k]` holds the SAME
     > objects as `H1.main`'s `attrs[k]` for the 11 entity keys; the H1 CSV
     > exporters in `run_replay.py` read `meta`, the charts read the registry.
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

- `first_confluence` with NULL `parent_cts_anchor_idx` must be modeled as genuinely
  pending — never silently using future data (under the pool: an
  `UnresolvedTrigger(reason="pending")`, §17.7).
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
  wave candle lines, WVMI hover overlays, connector lines (sub fibs are
  CSV-only — no M15 fib renderer; decided 2026-09-30)
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

> **⚠ REVISED by §17 (Sub-Structure Pool, rev 2 2026-09-19) for the M15
> charts.** Ownership becomes `owner_by_idx_dir[(candle, direction)]` over the
> sub's **lifecycle window** `[start_idx, end_idx]` — not the structural anchor
> — so a `+1` and a `−1` sub may both own a candle and pre-`start_idx` sid-tied
> dots are hidden (later revised by the chart reviews of 2026-09-20 → 22 —
> items 1-7 below: the FORMING layer draws the pre-`start_idx` geometry, and
> collapsed-cycle zones are hidden on both charts); chart identity is `sub_id`. Point (a) below is superseded
> for subs by §17.9; point (b) stands ("draw from the anchor, active from
> `start_idx`"). The H1 chart is unchanged. **Body rewritten to rev 2 in the
> Plan C commit (2026-09-20).**
>
> **REVISED 2026-05-25 — terminology only, rules unchanged (history).** Under
> the revised §6 model of that date sids were sequential & non-overlapping, so
> there was no "overwrite"; "older sid hidden / overwritten" became "older sid
> lifecycle-ended." Two points confirmed then: (a) sid-tied **structure**
> elements kept the `owner_by_idx` behavior and rendered at their structural
> anchors (`starting_idx`) — **superseded for subs** below; (b) **zone**
> rectangles draw from their base/IC anchor but fill only over the active
> window — still the rule.

**M15 lens charts (`export_m15_chart.py`, rev 2).**

- **Identity = `sub_id`.** The manifest is `attrs["sids"]` (one `SidRecord`
  per unique sub on this lens, §9.2); every drawn attrs list (events, KL, POI,
  wave candles, WVMI, prev-BOS lines; fibs are CSV-only) is grouped by `meta["sub_id"]`
  (`_sub_identity`); `attrs["triggers"]` supplies the record list for hover.
- **Ownership = the sub's real-time lifecycle window, keyed per direction.**
  `_compute_owner_by_idx_dir(sid_records, edge_idx)` builds
  `owner_by_idx_dir[(candle, direction)] = sub_id` over `[start_idx,
  end_event_idx or edge_idx]` for every sub row with a `start_idx` — the
  **lifecycle** window, NOT the structural anchor — walked in `(start_idx,
  sub_id)` order so a later start wins a same-direction overlap. A `+1` and a
  `−1` sub may both own one candle (opposite-direction overlap is allowed,
  §17.5 — `4027/+1` on confluence from 4083 while `3760/−1` runs to 4200).
- **Sid-tied elements draw only where owned:** CTS/BOS dots, swing lines, PB
  markers and prev-BOS lines of sub `X` render at candle `i` only if `X` owns
  `(i, X.direction)`. As WRITTEN on 2026-09-19 this hid the pre-`start_idx`
  geometry; the chart review REFINED it (see "Chart review 2026-09-20/21"
  below): the pre-`start_idx` portion is drawn from the anchor in the FORMING
  style, live ownership deciding only among live subs. A replaced sub stops at
  the replacement.
- **Persisting elements draw from the anchor, active from `start_idx`:**
  KL / POI rectangles are unchanged in mechanism (sub fibs are CSV-only) — drawn from their
  base / IC / anchor candle, filled only over their active stretch, which the
  KL/POI clamp already floors at the sub's `start_idx` and caps at its
  `end_idx` (§5). Opacity is a per-TF tier (`_m15_opacity_tier_for_zone`).
- **Every lens draws the sub over the sub's window.** A sub on both charts
  (`2639/−1`, `4027/+1`) has the same `[start_idx, end_idx]` on each — the
  counter chart draws `2639/−1` from the sub's 2829, not from its own
  record's 2843. Over any candle range only one structure per direction was
  truly tradeable; the charts show that one.

**Chart review 2026-09-20/21 (user decisions, landed in the Plan C commit) —
three refinements of the rules above; the CSVs are untouched by all three:**

1. **Forming phase (option 2).** Hiding everything before `start_idx` broke
   every BOS→CTS line mid-structure (sub `1797/−1` lost its first two cycles,
   `2639/−1` its cycles 0–1, `3304/−1` was a single point) while the KL/POI
   rectangles of those cycles were still drawn from the anchor. Rule now: a
   sub's sid-tied elements are drawn continuously **from the structural anchor**
   — the pre-`start_idx` portion in the **forming** style (navy, dotted,
   opacity 0.75; open-circle dots; hover `phase=forming`), the rest in the live
   style (`phase=live`; the bridging segment to the first live point is
   dotted). Ownership is two independent layers (`export_m15_chart.py`):
   `_compute_owner_by_idx_dir` (live: `[start_idx, end]`, later start wins
   among live subs) decides a sub's LIVE elements; `_compute_forming_by_idx_dir`
   (forming: `[starting_idx, start_idx)`, later anchor wins among forming subs)
   decides its FORMING elements — a structure forming under a live sub of the
   same direction stays visible (`3304/−1` forming 3304→3621 under live
   `2639/−1`; `3760/−1` forming under live `3304/−1`); the styles make the
   overlap legible. Styles: `structure.m15.swing_line_forming`,
   `structure.m15.{cts,bos}_forming`; all M15 structure dots +20%
   (`size 3.6`). ("The bridging segment is dotted" was superseded by the wave
   rule, item 4.)
2. **Collapsed-cycle zones are not drawn (option 1)** — on BOTH charts.
   A KL/POI zone whose cycle collapsed under its structure's lifecycle floor
   (`status="inactive"` with clamped `confirmed_idx >= end_idx`; on a sub the
   forming-phase cycles ended at/before `start_idx`, on H1 the retroactive
   cycles (1,0)/(1,1) of the post-reversal sid — the degenerate parent cycles)
   existed geometrically but was never active in real time; the forming
   dots/lines already show that geometry. Shared predicate
   `_zone_render.is_collapsed_cycle_zone`; POIs are skipped via their cycle's
   BOS KL zone (`collapsed_cycles` / `is_poi_of_collapsed_cycle`), because a
   POI's own `status` cannot distinguish "collapsed cycle" from "never
   activated inside a live cycle" — the latter is still drawn as an outline
   (sub `3621/+1` cycle-1 POI on the reference window). Applied in
   `export_plotly.py` (H1 chart), `export_m15_chart.py` (sub zones + the H1
   overlay). The rows stay in the CSVs.
3. **POI zones are side-tinted** like KL zones: buy = gold-lime
   `rgb(225, 220, 30)` (confirm line dark olive), sell = amber
   `rgb(255, 180, 30)` (confirm line dark brick) — `zone.poi.buy` /
   `zone.poi.sell` in `style_registry.py`; opacities unchanged.
4. **Wave rule (2026-09-21).** Forming vs live is decided per WAVE (one
   segment between two consecutive drawn points, over the anchor candles the
   line runs through — not the confirmation candles): a wave is solid if any
   part of its candle span lies inside the sub's `[start_idx, end_idx]`, and
   forming only if none of it does (`_wave_touches_window` /
   `_split_polyline_by_wave`; PB→BOS lines follow the same rule; the
   extension to the last owned candle is a wave). The bridge wave that crosses
   `start_idx` — the one the first live KL BOS zone hangs on — is therefore
   solid (item 1 had drawn it dotted, so live zones sat on dotted waves).
   Dots follow their waves (filled iff they end a solid wave); the hover
   `phase` stays the per-candle fact. Prev-BOS lines carry no lifecycle
   formatting (always solid, sub and main). On the reference window the five
   subs whose pre-start span sits inside one wave (`2365/+1`, `3304/−1`,
   `3621/+1`, `3760/−1`, `4027/+1`) lose their forming run entirely.
5. **H1 overlay structure lines are lifecycle-filtered per wave (2026-09-21).**
   On the sub charts (`_render_h1_overlay`) an H1 wave / PB→BOS line is drawn
   iff its span intersects its sid's lifecycle window `[struct_start_by_sid,
   reversal idx]` (the reversal-handoff start; canonical
   `zones.structure_lifecycle` helpers) — never-live waves are not drawn (sid
   1's retroactive 689→826 waves on the reference window; 826→905 spans 902 and
   is drawn whole); an H1 dot is drawn iff a drawn wave touches it (BOS@689
   stays as the end of sid 0's PB→BOS line). The H1 chart is deliberately
   unchanged (every sid in full, prior dimmed): the high-level picture wants
   the most recent structure's full geometry, the trading charts do not.

6. **Recent vs prior — what solid and dotted mean (2026-09-22).** Item 4's
   lifecycle test is REPLACED for the sub charts (the H1 overlay filter, item
   5, still uses it). Where two different structures draw segments over the
   same candles, the MOST RECENT one is solid and the PRIOR one's whole segment
   is dotted; a segment nothing overlaps is solid, whether or not the structure
   was ever live in real time. Recency = the hierarchical `(parent_sid,
   parent_cycle_id, sub_id)` tuple (`_recency_key`; orders like `sub_id` on
   every window measured). Overlap = more than one shared candle (a single join
   candle does not count) and is direction-agnostic (sub `3760/−1` is dotted
   under sub `4027/+1` although both are live). Formatting is per whole
   segment. Computed per lens (`_prior_line_segments` over that chart's drawn
   segments), so a sub can be prior on one lens and solid on the other (sub
   `2639/−1`'s 3304→3611: dotted on confluence, solid on counter) — a
   deliberate exception to "every lens draws the sub identically", which still
   holds for the window and the geometry. Ownership is unchanged and still
   decides which points exist (a same-direction structure the ownership layers
   collapse is hidden, not dotted). Dots follow their segments (filled iff the
   dot ends ≥1 solid segment); prev-BOS lines are always solid and never make
   anything prior. The hover carries both facts: `phase=live|forming` (real
   time) and `layer=recent|prior` (why this style); the real-time lifecycle is
   otherwise carried by the zones. Consequence accepted at review: an ACTIVE
   zone can hang on a dotted segment once a later structure supersedes it — the
   inverse of item 4's motivation, with a different meaning. Styles renamed
   `structure.m15.*_forming` → `*_prior`.

7. **Replaced subs run through one more structural point (2026-09-22).** A sub
   ended by `same_dir_replacement` breaks its final segment at the counter-move
   extreme over `(last drawn point, the REPLACING structure's anchor]`
   (`_replacement_break_point`, a PB dot; `−1` → highest high, `+1` → lowest
   low). Bounding at the replacing anchor yields the STRUCTURAL swing, not a
   later marginal overshoot (sub `3304/−1`: 3760 @ 0.57806, not the literal high
   3806 @ 0.57827), and the new segments mirror the sibling structures point for
   point (`3304/−1`'s 3621→3760 = `3621/+1`'s segment; `3621/+1`'s 3760→4000 =
   `3760/−1`'s), which also splits the partial overlap into one solid and one
   prior piece. Replacement only: every reversal-ended sub's final segment
   already ends at its counter-move extreme (0–2 candles on the reference
   window) and `parent_end` ends at the parent's candle.

Chart counts after all seven (the standard counts from the 2026-09-22 chart
review on): H1 85/245, `M15.counter` 152/125, `M15.confluence` 293/233; shapes
never move (items 4–6 are trace-level). Measured against the Plan C save
(`20260921_125218_afaa326`, 157/125 and 301/233): confluence −8 = 7 forming
swing traces replaced by 6 prior ones (−1: sub `4027/+1` has no prior segment)
plus 7 forming dot traces merged into their subs' live dot traces (only sub
`2639/−1`'s 3611 CTS dot is still prior); counter −5 = 3 forming swing traces
replaced by 1 prior one plus 3 dot-trace merges. Intermediate value after items
4–5 alone (2026-09-21): counter 154/125, confluence 295/233.

- **Hover on a sub's dots:** `sub_id`, `struct_direction`, `relative_dir` at
  that candle (`SidRecord.relative_dir_segments` step function), the sub
  window `[start_idx, end_idx or open] end_reason`, and the record list
  `lens(S,C) trigger_type trigger_idx→start_idx` (zero-length records marked
  `†`) from `attrs["triggers"]`; plus the first record's `parent_sid` /
  `parent_cycle_id` (informational).
- **Tier logic** (`m15_most_recent_psid` / `recent_cycle_ids`,
  `_compute_m15_tier_context_from_sids`) — never applied to any trace (audit
  2026-09-21) and DELETED 2026-09-30; M15 dots render at flat opacity.

| Element class | Display rule |
|---|---|
| Constants — candle patterns, OHLC, candle types | Always shown; never affected by sub changes |
| Sid-tied — CTS dots, BOS markers, swing / PB / prev-BOS lines, range bounds, market_state regions | H1: **most recent sid only** per candle (`owner_by_idx`). M15: **the owning sub per `(candle, direction)`** over its lifecycle window (`owner_by_idx_dir`). Older / ended subs' data persists in the lens df with lifecycle end meta but is hidden where not owned |
| Persisting events — KL zones, POI zones, imbalances | All rendered EXCEPT collapsed-cycle KL/POI zones (item 2 above — never active in real time; they stay in the CSVs). Inactive / lifecycle-ended ones use the existing opacity attenuation logic (older = more transparent); fills gated to the active window |
| WVMI | NOT rendered on any chart (H1 or M15: wave-candle hover only, no momentum — verified 2026-09-30); the `_wvmi.csv` exports are the surface. The planned design: locked records show final values via hover; in-progress records re-render whenever `update_temporary_lp` shifts the temp LP. M15 data: one record set per unique sub, present on every lens df the sub is on, each copy with its lens's path (§17.10, Plan G) |

### 16.6 Pending subordinate display

Per §14, pending triggers (currently only `first_confluence` while parent CTS
is unconfirmed — a NULL `parent_cts_anchor_idx`,
`FirstConfluenceTrigger.status == "pending"`) are in **pending** state.

- Hide entirely. No chart elements produced — under the pool the trigger is
  an `UnresolvedTrigger(reason="pending")` row in
  `*_M15_unresolved_triggers.csv` (§17.7), no record, no sub.
- When `parent_cts_anchor_idx` resolves, the sub's data appears at the resolving
  candle's time, retroactively visible across the resolved range (its record
  still starts no earlier than `max(probe_finalize_idx, trigger_idx,
  parent_floor_idx)`).

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

## 17. Sub-Structure Pool (Phase 2) — Design LOCKED 2026-09-19 (rev 2) — LANDED by Plan C

> **Status:** **LANDED by Plan C — commit `afaa326` (2026-09-21), save
> `20260921_125218_afaa326` (the next `/compare` baseline).**
> Design LOCKED 2026-09-19 (rev 2). Rev 1 (2026-07-08) was landed
> byte-identically as Stages 1–3.2a (`3cb6513`, `22969d3`, `c932610`) and then
> found wrong on chart review at Stage 3.2b (`9fd3143`, reverted as `5a658dc`,
> kept in history). This revision replaces rev 1's lifecycle
> model (§17.4–§17.11 of the old text, in git history) with the
> **`TriggerRecord` + unique-sub** model below. §17.1–§17.3 survive with
> rewording. The landed code: `multitf/sub_structure_pool.py` (data model),
> `multitf/parent_tables.py` (§17.6 static tables), `multitf/lifecycle_sweep.py`
> (the §17.6 sweep — the driver), `multitf/entity_df_mutation.py` (resolvers,
> probe cache, `build_or_get_geometry`, `render_sub_projection`, mirror),
> `pipeline/orchestrator.py::_run_multi_tf_dual` (wiring),
> `charting/export_m15_chart.py` (§16.5 rev-2 ownership), `debug/export_sub_tables.py`
> (the three CSVs). Measured on the first Plan C replay (2026-09-20): 8 subs /
> 11 records / 4 unresolved, every window and record equal to the predicted
> table; H1 8 of 9 CSVs byte-identical (`_wvmi.csv` header-only: `sub_sid` →
> `sub_id`); 633 tests + 1 strict xfail (§17.11).
>
> **Implementation:** `plans/PLAN_C_lifecycle_rewrite.md` (the implementation
> contract — file-level mechanics, tests, acceptance), landed **after**
> `plans/PLAN_A_*` (MS range look-ahead clamp) and `plans/PLAN_B_*` (double-CTS
> early stop), each with its own replay + `/compare` + chart-review pause
> (§17.11). Rationale history: `memory/project_sub_structure_pool_architecture.md`;
> hard numbers + the acceptance table: `memory/reference_pool_redesign_groundtruth.md`.
> Vocabulary: `GLOSSARY.md` "Sub-Structure Pool Terms".
>
> **Precedence:** for subordinates §17 wins over the §2 / §5 / §6 / §9 / §13.5 /
> §16.5 prose. Those sections carry pointer banners and, as of the Plan C commit,
> their bodies are **rewritten to rev 2** (§2 identity, §4.3.6 cadence → sweep,
> §5 lifecycle-start rules, §6 record/sub rules, §9 storage + `attrs["sids"]`,
> §13.5 migration log annotated, §16.5 ownership); where a body and §17 still
> read differently, §17 is authoritative. The pool modules' docstrings cite the
> rev-2 numbering below.
>
> **One principle every rule below respects.** Lifecycle fields (`start_idx`,
> `end_idx`) are **real-time** — what was tradeable when. Pattern/element
> fields (`starting_idx`, `trigger_idx`, `probe_finalize_idx`, every anchor) are
> **historical** — where things logically sit. Historical fields are legitimate
> *inputs* when constructing a lifecycle value; they are never used *as* one in
> an is-active-at-t test, and a lifecycle floor is never applied to a historical
> field. Every candle index is named `*_idx` (never `*_dt`).

### 17.1 Motivation

There are exactly five triggers that can start a new sub structure: the four
confluence/counter variations (`first_confluence`, `first_counter`,
`subsequent_confluence`, `subsequent_counter`) and `reversal`. Every one uses
the **same unified probe** to find `starting_idx`. A sub structure is therefore
fully determined by four things — its own TF, its parent path, its direction,
and its final `starting_idx` — **independent of which trigger produced it**. If
two triggers/probes land on the same four, the resulting MS run + all derived
elements are byte-identical, so re-running is pure waste (16 per-trigger
builds → 11 unique structures on the NZD_USD 2025-11-15→2026-01-20 window).

Phase 2 exploits this: **compute each unique sub once**, and let every trigger
that lands on it point at that one shared structure. Rev 1 treated the pool as
a dedup cache with a post-pass lifecycle; Stage 3.2b showed that the pool *is*
the lifecycle model. Two objects have genuinely different lifetimes — a trigger
fires **inside one parent cycle** and its instance cannot outlive that parent,
while the physical structure it points at is **not bound to any parent** and
can stay tradeable across parent cycles and parent sids. Modelling both
explicitly (§17.2) is what makes the charts, the trading window and the
attribution all agree.

### 17.2 Structure ↔ TriggerRecord split

| Object | Role | Identity | Cardinality |
|---|---|---|---|
| **`PooledStructure`** (= **unique sub**) | one MS run + its cap-free derived geometry (KL/POI/fib/wave-candle bounds & anchors), computed **once**, run cap = **data edge**; carries ONE real-time lifecycle aggregated from its records. **This is what we trade on.** | `(parent_path, sub_TF, direction, starting_idx)`; `sub_id` global monotonic (creation order) | one per unique tuple |
| **`TriggerRecord`** | a triggered **instance** of a unique sub, with its **own parent-bound lifecycle**; carries provenance (trigger, probe) and the chart lens | `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)`; FK `sub_id` (**never None**) | N ≥ 1 per built sub; two triggers in the same `(lens, parent_sid, parent_cycle_id)` resolving to the same sub share ONE record |
| **`UnresolvedTrigger`** | a trigger that produced no record: `pending` (FC probe not finalized), `degenerate_parent_cycle`, `probe_failed`, `geometry_failed` (§17.7) | — (log row; no `sub_id`, no `trigger_sub_sid`) | one per unresolved trigger |

`PooledStructure` stores its elements in slice-local coords + `slice_begin`;
the one shared M15 feature frame is the store (§17.9). Geometry is
**cap-free** (computed to the data edge, no lifecycle floor/cap baked in); the
lifecycle projection is applied **once per unique sub** (§17.9).

**A record cannot outlive its parent structure; a unique sub can.** This
retracts rev 1's "a sub can't outlive its parent structure (§6.5)": the parent
bound is a *record* property (`parent_end`, §17.4), and the sub inherits it only
through the aggregation rule (§17.5).

### 17.3 Identity — `direction` is absolute; `relative_dir` (semantic) vs `lens` (chart)

- `direction ∈ {+1, −1}` is **absolute**, not confluence/counter. Dedup keys
  off the actual MS run, which is absolute-direction.
- `starting_idx` is an entity-absolute M15 idx (unique in time) and is nearly
  sufficient alone; the full tuple is the clear key and generalizes to deeper
  nesting (`parent_path`, not just `parent_TF`, so an M5-under-counter can't
  collide with an M5-under-confluence). `parent_path` is the constant
  `"H1.main"` today.
- Confluence vs counter is **two different things, both on the record, neither
  on the structure**:
  - **`relative_dir`** (semantic): `confluence` iff `direction == parent_sd` of
    the record's parent **sid** (`CTS_ESTABLISHED.meta["struct_direction"]` —
    one direction per sid), else `counter`.
  - **`lens`** (chart): which chart the record draws on. Named triggers map by
    `use_case` (`resolve_lens`: `first_confluence` / `subsequent_confluence` →
    confluence; `first_counter` / `subsequent_counter` → counter). A
    **`reversal`** record inherits the lens of the record whose sub reversed
    (sticky-per-chart, §17.5). The `"confluence" in sub_path_id` substring test
    is retired.
  - These legitimately differ: sub `2639/−1` under H1 sid 0 (`+1`) is
    `relative_dir = counter` yet draws on the **confluence** chart through
    reversal-stickiness — intended. Rev 1's "confluence iff `direction ==
    parent.current_sd` … per-chart label" conflated the two; that conflict is
    what this split retires.
- A unique sub's `relative_dir` over time is a **step function**
  (`relative_dir_segments`): at each candle the value of the active record with
  the latest `start_idx`, carried forward when none is active. It can flip
  across a parent-sid change without the structure changing.

**This supersedes §2's sticky `starting_alignment`.** Alignment is not part of
structure identity and is not sticky on the structure; stickiness survives only
as the lens rule for `reversal` records.

### 17.4 `TriggerRecord` — a triggered instance with its own lifecycle

All idxs are entity-absolute M15 ints, except `parent_bos_anchor_idx` (H1, FC only). `LOH(h)` = `_map_parent_idx_to_m15_hour_end`
(last M15 candle of H1 candle `h`) — the mapper for every **timing** value; the
price-extreme mapper (`map_candle_to_lower_tf`) is used only for the FC probe's
structural inputs (§17.8) — and, for display, the M15 chart's H1 zone-proximity
markers (Plan E E5·2). Do not unify them.

| Field | Rule |
|---|---|
| `lens`, `parent_sid`, `parent_cycle_id`, `trigger_sub_sid` | identity. `trigger_sub_sid` starts at **0** per `(lens, parent_sid, parent_cycle_id)` and increments each time a trigger in that scope resolves to a **new** unique sub. It is **creation-ordered**, not `start_idx`-ordered (an FC record is created at its `trigger_idx` and may start hundreds of candles later). |
| `sub_id` | FK → unique sub. Never None. |
| `trigger_type` | `first_confluence` \| `subsequent_confluence` \| `first_counter` \| `subsequent_counter` \| `reversal` |
| `trigger_idx` | HISTORICAL. The candle the trigger fired: `LOH(trigger_event_idx)` for the four H1 types (`first_counter`'s event idx is `WVMIRecord.meta["triggered_by_event_idx"]`); the native M15 reversal idx for `reversal`. |
| `probe_finalize_idx` | HISTORICAL. When **this record's probe** finalized, where "this record's probe" is the run keyed by `(direction, initial input)` (§17.8). Probe **ran** → its own `ProbeResult.finalize_idx`, raw (may precede `trigger_idx`: FC(0,1) finalized 2608 < trigger 2611 — logged as is). Probe **skipped** (cache hit) → the cached finalize, inherited raw (that run already finished; the re-trigger does not wait). A different-input probe that *converges* on a known structure keeps its **own** finalize — it could not know it mapped there until it finished. Non-FC types whose probe RAN: `finalize == trigger_idx` by construction (asserted in `_resolve_sibling_cts_via_unified_probe` and in the sweep); a cache-hit record is exempt — it inherits the earlier finalize, which is normally below its own `trigger_idx` (measured: `first_counter`(0,1) 2829 < 2843). |
| `parent_floor_idx` | `LOH(max(struct_start[S], cts_moment[(S,C)]))` — the parent cycle's **clamped lifecycle-start**, on the CTS-established **moment** (§17.6). Diagnostic copy of the floor that was applied. |
| **`start_idx`** | **REAL-TIME. `max(probe_finalize_idx, trigger_idx, parent_floor_idx)`. The record EXISTS from here** and nothing earlier. All three terms are load-bearing, each for a distinct case: `probe_finalize_idx` when the probe finished after the trigger (FC(0,0): trigger 463 → start 1020); `trigger_idx` when the structure was already known before this trigger fired (a cache-hit record inheriting an earlier finalize 3487 with its own trigger at 3611 → 3611); `parent_floor_idx` when the parent structure was not alive yet (an FC finalize 2844 under a floor of 3611 → 3611). The last two are illustrative — on the reference window the cache-hit pair never meets and the floor-bound record sits in a degenerate cycle (§17.7), so the floor binds only in a tie with `trigger_idx` there — but each term is what makes the corresponding case start honestly. Historical fields are never adjusted; only `start_idx` is. |
| `trigger_end_idx` | the first end condition to fire (below), or the sub's frozen end on a post-end re-trigger. |
| `end_idx` | REAL-TIME. `max(trigger_end_idx, start_idx)`; None while open. |
| `end_reason` | `reversal` \| `same_dir_replacement` \| `parent_end` \| None. `ended_by_sub_id` names the replacing sub for `same_dir_replacement`. |
| `starting_idx`, `direction`, `sub_tf`, `relative_dir`, `parent_bos_anchor_idx`, `probe_input_idx`, `probe_finalize_condition` | structural / provenance copies (denormalised for the export). `parent_bos_anchor_idx` (was `validated_parent_idx` until Plan E E5·4, 2026-09-26 — it mixed frames, PLAN_E Q4) = `first_confluence` only: the **H1** parent BOS anchor (`FirstConfluenceTrigger.input_idx`) that seeded its M15 probe input; None for the three sibling types (their input, the M15 sibling CTS anchor `ref_zone.anchor_idx`, is `probe_input_idx`) and for reversal-born records. `probe_input_idx` (Plan E E5·4b) = the M15 candle this record's probe started from, for every resolved type (FC: the price-mapped parent BOS anchor; siblings: the sibling CTS anchor, or the own-frame ad-hoc BOS_0 anchor when the sibling has none (§4.3.4 step 4); reversal: the handoff input) — copied from `ResolvedStart.probe_input_idx`. The sub-level copy `SidRecord.meta["validated_parent_start"]` was deleted in Plan E E1b (unread). Diagnostic only — no lifecycle value reads it (Plan C §2.1's "H1 candle" wording was imprecise; corrected 2026-09-20). |
| `extra_trigger_idxs` | later triggers in the same `(lens, parent_sid, parent_cycle_id)` that resolved to the same sub — **absorbed into this record**, no new record, `trigger_sub_sid` not consumed. The match includes a ZERO-LENGTH record (a re-trigger into a scope whose record was frozen — post-end or collision — is absorbed, not revived; acausal cases only). |

**End conditions — record level ONLY** (the sub never has its own; it only
aggregates, §17.5). Earliest wins; write-once; at an equal idx the priority is
`reversal > parent_end > same_dir_replacement`:

1. **own reversal** — the sub's natural reversal idx (geometry runs to the data
   edge, so it is known at build time).
2. **same_dir_replacement** — a NEW record in the same
   `(lens, parent_sid, parent_cycle_id, sub_TF, direction)` mapping to a
   **different** unique sub ends the active one **at the new record's true
   `start_idx`** (not its trigger or finalize idx — the old record stays
   tradeable until the new one is live). **Same-lens only**: a counter record
   never ends a confluence record. Invariant: ≤ 1 active record per
   `(lens, parent_sid, parent_cycle_id, direction)` ⇒ ≤ 4 per parent cycle;
   asserted.
3. **parent_end** — the record's OWN parent cycle ends: the next cycle's
   **clamped** lifecycle-start (`floor_h1[(S,C+1)]`, §17.6), else the sid's
   `STATE_CHANGED→reversal` idx (**not** `REVERSAL_CANDIDATE.apply_idx` — a
   prediction that may not realise), else None (open). Mapped with `LOH`. One
   helper replaces today's three derivations (`parent_end_lookup`, the trigger
   detectors' `lifecycle_end_idx`, `_find_m15_lifecycle_end`).

**Zero-length records.** `is_zero_length = trigger_end_idx is not None and
trigger_end_idx <= start_idx` (defined on `trigger_end_idx`, not `end_idx`, so
it is already decidable before `end_idx` is written). A zero-length record has
a `sub_id` but **participates in nothing** — not the sub's start, not
`max_start`, not an end candidate, not an incumbent, not lens membership, not
rendering. It is logged only. Two sources: a **post-end re-trigger** (the sub
had already ended before this record's `start_idx` → the record freezes with
the sub's `end_idx` / `end_reason`, linked and logged — the probe still ran,
because its output is the pool key), and a structure whose own reversal
precedes the record's floored start inside a live cycle.

**Triggers are independent.** A `subsequent_*` trigger is processed even when
its parent cycle's `first_*` trigger was unresolved or zero-length. §6.1's
"no sid 0 → no subsequents" rule is **retired** (it was an artifact of the
sequential chain; `subsequent_confluence` reads the *counter* sibling, not its
own bootstrap).

### 17.5 Unique sub — one lifecycle aggregated from its records

A unique sub is **not bound to a parent**: it spans parent cycles AND parent
sids. It has one lifecycle, one projection, and is rendered over the **same
window on every lens** it belongs to.

| Field | Rule |
|---|---|
| `start_idx` | the **first non-zero-length record's** `start_idx`. Set once. |
| `end_idx` | at each candle `t`, over the sub's non-zero-length records that **exist at `t`** (`start_idx <= t` — a record does not exist before its `start_idx`): `max_start = max(start_idx)`; candidates = non-null record `end_idx`s **strictly `> max_start`**; `end_idx = min(candidates)`. Set once, frozen. Equal-idx ties between candidates break by the record end-reason priority. |
| `end_reason` | the winning candidate's reason. |
| `relative_dir_segments` | the step function of §17.3. |
| lenses | union over the sub's non-zero-length records' `lens`. A sub with **no** live record has no lens: it is logged on stdout (`[pool] sub_id=… has no live record — logged, not rendered`) and **not rendered**; being on no lens, it appears in no per-lens `_subs.csv` (its zero-length records do appear in the lens `_triggers.csv`). |
| `natural_reversal_idx`, `bos0_inner` | from the build (§17.8). |

**Why strict `>` and why records must exist only from `start_idx`.** Two
cases, both falling out of the evaluation order (§17.6):

- the sub's last live record ends at `E` **before** a pending record's
  `start_idx` → the sub ends at `E` in real time (at `E` only started records
  are visible); when the pending record reaches its `start_idx` it finds the
  sub frozen → it becomes zero-length with the sub's end (linked, logged).
- the end lands **on** the new record's `start_idx` → the new record starts
  first, `max_start = E`, the old record's end `E > E` is false → the sub
  persists through the new record until a later condition. This is the
  same-candle handover: sub `2365/+1`'s record in H1 (0,0) ends `parent_end`
  at 2611 and its record in (0,1) starts at 2611 → one continuous sub
  `[2470, 2829]` ending on its own reversal. Rev 1's `≥` would have ended it
  at 2611 and restarted it — and could discard a sub's own reversal.

**Consequences (intended, not leaks):**

- **One lifecycle, drawn on every lens.** A same-lens replacement of a sub's
  *confluence* record ends the SUB, so the counter chart stops there too even
  if the sub's counter record would have run longer. (Not observable on this
  window.)
- **Cross-parent continuity** is the `2365/+1` case above (records in H1 (0,0)
  and (0,1), one window). **Cross-lens, one window:** sub `2639/−1` has a
  confluence record (reversal-born from `2365/+1`) and a counter record
  (`first_counter`), both in H1 (0,1), both `parent_end` at 3611 → one sub
  `[2829, 3611]` on both charts; the counter chart draws it from **2829** (the
  sub's window), not from its own record's 2843.
- **Opposite-direction overlap is allowed.** A `+1` and a `−1` sub may both be
  live on one chart (`4027/+1` on confluence from 4083 while `3760/−1` runs to
  4200). Rev 1's "per-chart non-overlap" claim was false; ownership is keyed
  per direction (§17.9).

**Run cap = data edge, for every sub.** The geometry builder passes
`run_cap_abs = len(m15) − 1`. MS is ~23 s total, affordable; this removes rev 1's
first-writer-wins cap, the silent `natural_reversal_idx = None`, and the
`_time(end_idx)` KeyError path. The run cap is a **compute** bound only.

**Successor spawn rule.** A sub's natural reversal `R` spawns the
opposite-direction successor **iff `R` falls inside a live record's window**
(`start_idx <= R` and the record has not ended before `R`; a record whose
parent ends *at* `R` is still live at `R`). One `reversal` trigger is
synthesised **per live record** (so per lens): `(lens = r.lens, r.parent_sid,
r.parent_cycle_id, type = reversal, trigger_idx = R, direction = −sub.direction)`,
probed by the reversal handoff over the reversing sub's own geometry; the
successors from two lenses **dedup into one sub with a record per lens**
(`4027/+1`: `subsequent_counter` at 4083 + reversal-born from `3760/−1` at
4200 on confluence). If no record is live at `R` there is no successor —
the reversal is inert (closes rev 1's "ended by reversal but nothing drawn
after"). Successor `start_idx = R` (finalize = trigger = R; floor ≤ R because
the spawning record is live).

### 17.6 Evaluation — an ordered sweep, equivalent to per-candle

Every rule in §17.4–§17.5 is monotone and write-once, so a **priority-queue
sweep over moments** is provably identical to iterating candle-by-candle. This
**replaces `_ChainCursor` / `build_two_entity_parent_cycle` /
`build_parent_cycle_chain`** — a driver rewrite, not an edit. The real
per-candle driver is Phase 3 (per-candle dual-lens trading); the sweep's step
body is that loop body.

**Moments and phases.** At each idx, phases run in this order, each to
completion before the next:

| phase | moment | action |
|---|---|---|
| 0 | `TRIGGER_FIRE` (an H1 trigger at its `trigger_idx`) / `REVERSAL_SPAWN` (a sub's natural reversal, inserted when its geometry is built) | resolve (§17.7 degenerate check → probe → geometry) → create the record → queue its `RECORD_START` and any end already known |
| 1 | `RECORD_START` | if the sub already ended **before** this idx → freeze the record zero-length; else mark active and queue a `same_dir_replacement` candidate for the same-lens incumbent (if any, and of a different sub) at this idx |
| 2 | `SUB_START` | set `sub.start_idx` if unset and a live record started here |
| 3 | `RECORD_END` | for a record not yet ended: apply the highest-priority end entry at this idx; entries for an already-ended record are stale and dropped |
| 4 | `SUB_END` | re-evaluate the §17.5 end rule for every sub that had a live record start or end at this idx |

**Order within a phase.** `TRIGGER_FIRE`: `(lens_rank, type_rank,
trigger_event_idx)` with confluence before counter and `first_*` before
`subsequent_*` — today's cadence order, which matters because the second
trigger's sibling read at `hi = t` inclusive can see the first's sub (§17.8).
`REVERSAL_SPAWN` sorts **before** `TRIGGER_FIRE` at the same idx (the successor
must exist before an H1 trigger at that candle runs its sibling read). Every
other moment orders by record creation sequence. Two records of the same
`(lens, parent_sid, parent_cycle_id, direction)` starting at the same idx for
different subs cannot occur outside acausal cases; if it happens the later one
replaces the earlier and a `WARNING [sweep] same-idx start collision` is logged.
As landed (`lifecycle_sweep._phase1_record_start`): the incumbent is the record
of the same `(lens, parent_sid, parent_cycle_id, direction)` that has
**started** (its `RECORD_START` already ran) and is still active at `t` — not
the bare interval test, which would also see a same-idx record whose start
moment is still queued; a collision incumbent (`start_idx == t`) is frozen
zero-length **in phase 1** rather than queued for phase 3, so phase 2 never
reads it as live. `pool.active_record` (the interval rule, asserting ≤ 1) is
the post-sweep query (used by the tests; no production reader yet).

**Start-before-end at the same idx is load-bearing** (§17.5's handover case).
The heap is idx-monotone: a moment is never queued in the past (a reversal
`R <= t` at build time spawns nothing — no record can be live at `R`; the
triggering record becomes zero-length). Asserted.

**Static parent tables** (computed once from the H1 event stream before any
sub work; one helper replaces `parent_cycle_floor_h1`, `parent_struct_end_m15`,
`parent_end_lookup`):

```
rev_by_sid[S]     = STATE_CHANGED→reversal idx of sid S          (compute_reversal_idx_by_sid)
struct_start[S]   = reversal handoff floor                       (compute_struct_start_by_sid)
cts_moment[(S,C)] = CTS_ESTABLISHED.meta["confirmed_at"]          # the MOMENT the cycle was established
                                                                 #   (== BOS_CONFIRMED.confirmed_at, definitional;
                                                                 #   last-seen per (S,C)) — NOT the CTS anchor (a location)
parent_sd[S]      = CTS_ESTABLISHED.meta["struct_direction"]
floor_h1[(S,C)]   = max(struct_start[S], cts_moment[(S,C)])      # == the cycle's CLAMPED lifecycle-start
end_h1[(S,C)]     = floor_h1[(S,C+1)] if it exists, else rev_by_sid[S], else None
floor_m15 / end_m15 = LOH(...)                                   # every map must succeed (assert)
degenerate[(S,C)] = end_m15 is not None and floor_m15 >= end_m15
```

**Cycle lifecycle-start = the CTS-established MOMENT, not the anchor
(decided 2026-09-19).** A cycle's real-time lifecycle begins when it is
*established* (`CTS_ESTABLISHED.meta["confirmed_at"]`, the apply candle);
its CTS anchor (`meta["cts_anchor_idx"]`; `CTS_ESTABLISHED.idx` until Plan E E4a
made that idx the moment) is where its price sits — a historical location,
like the BOS anchor. Rev 1 and §5 used the anchor ("the
canonical cycle-start idx"). Plan C changes the canonical rule in
`zones/structure_lifecycle.py::compute_cycle_lifecycle` — cycle start =
`max(CTS_ESTABLISHED.meta["confirmed_at"], struct_start, floor)` — so main, sub
cycles and this parent table all agree. On the reference window H1 is
byte-identical (anchor == moment on all five cycles); two unique M15 sub
cycles have anchor ≠ moment, each by one candle — three CSV rows, because
sub `2639/−1` is mirrored into both lenses (1223/1224; 2828/2829 ×2) — and
were PREDICTED to shift +1 — measured: ONE visible shift (the 1223→1224 case,
as the END of sub `454/+1`'s cycle 1), the 2828→2829 cycle (both lens rows)
masked by an equal record floor (see the blockquote below and GOTCHAS "A
Predicted +1 Shift Can Be Masked by an Equal Floor"). Anchor == moment is the
common case (31 of the 34 `CTS_ESTABLISHED` rows), not luck — but it is not
guaranteed, and the one-candle lag is empirical, not a bound (`ARCHITECTURE.md`
"`ev.idx` convention").

> **Resolved for POI — Plan D (`0a4eadc`), 2026-09-23.** POI's activation floor is now
> `max(cts_established_idx, ic_idx, lifecycle_floor_idx)` with `cts_established_idx =
> CTS_ESTABLISHED.meta["confirmed_at"]` (a cycle with no `CTS_ESTABLISHED` builds no POI since
> 2026-09-29 — before, the fallback `fib_state.cts_idx`), and the POI meta key `cts_established_idx` holds that moment (it held
> `.idx`, the CTS anchor, before). Measured on the reference window: M15 confluence sub 0 cycle 2
> IC 678 activation 1223 → 1224 and 3 `cts_established_idx` meta cells re-valued (1223 → 1224,
> 2828 → 2829 on both lenses); the other 22 CSVs byte-identical; chart counts unchanged
> (`plans/PLAN_D_poi_activation_moment.md`). KL's start side is unchanged. The text below is the
> Plan C "as landed" record, in the past tense for POI.
>
> **As landed (2026-09-20) — what the moment rule actually reaches.**
> `compute_cycle_lifecycle` Pass 1 is on the moment, and the cycle START feeds
> only the END side (cycle *n* ends at cycle *n+1*'s clamped start — B2:
> "elements inherit the END only; they keep their own START", §5). Measured:
> one visible +1 shift (sub `454/+1` cycle-1 **end** 1223 → 1224); the two
> 2828 → 2829 cases are masked by the identical record floor 2829. The zone
> START side was unchanged by Plan C and was **not** on the moment: KL clamps a
> zone's own `confirmed_idx` to the **structure** start
> (`compute_struct_start_by_sid`, B1) — for a BOS zone that confirm IS the
> moment by construction; POI gated `first_active = max(cts_established_idx,
> ic_idx, struct_start)` with `cts_established_idx = CTS_ESTABLISHED.idx` (the
> CTS **anchor**, `poi_zones.py`) until Plan D. So the "KL/POI `confirmed_idx`
> clamps following" above was true only of their ends; a POI could activate up
> to (moment − anchor) candles before its cycle's clamped lifecycle-start.
> Flagged then as a code-vs-spec residual (Plan C §1 scoped `zones/*` out except
> via the leaf); resolved by Plan D (above).

**Assert** `BOS_CONFIRMED(S,C).meta["confirmed_at"] ==
CTS_ESTABLISHED(S,C).meta["confirmed_at"]` (definitional — both are the same
`apply_idx`); **never** assert it against the CTS anchor `meta["cts_anchor_idx"]`
(`CTS_ESTABLISHED.idx` until Plan E E4a; false on 3
of the 34 saved `CTS_ESTABLISHED` rows — 2 unique M15 cycles; true on all 5 H1
cycles here because anchor == apply candle is the common case, not a
guarantee). The per-event field table (`ev.idx` vs `meta["confirmed_at"]` vs
the anchor keys of the two anchor realms, for every structural event) is canonical
in `ARCHITECTURE.md` "`ev.idx` convention".

Reference window values: floors `(0,0)=463 (0,1)=2611 (1,x)=3611`; ends
`2611 / 3611 / 3611 / 3611 / None`; degenerate `{(1,0), (1,1)}`.

### 17.7 Degenerate parent cycles, unresolved triggers

- **Degenerate parent cycle** = lifecycle floor ≥ lifecycle end (zero or
  inverted length). Cause: the reversal handoff on a **retroactive** parent —
  H1 sid 1's `struct_start = reversal(sid 0) = 902`, so cycles (1,0)
  (established 703) and (1,1) (established 748) have floor 902 ≥ their ends;
  only (1,2) is real. Any record in a degenerate cycle is provably zero-length.
- **Rule: log, don't build.** A trigger in a degenerate cycle produces an
  `UnresolvedTrigger(reason="degenerate_parent_cycle")` — **no probe, no MS,
  no downstream, no `sub_id`**. Reason not to probe: sibling-referencing probes
  would read siblings that were never built → phantom `starting_idx`s. A
  degenerate trigger is not an instance; it records "the rebuilt structure
  would have looked like this". Reversal-born successors inside a degenerate
  cycle never fire (the reversal is inert too). One `WARNING [parent_tables]
  degenerate parent cycle` per cycle.
- **Unresolved-trigger log** (`_unresolved_triggers.csv`; separate from records):
  `reason ∈ {pending, degenerate_parent_cycle, probe_failed, geometry_failed}`
  with the trigger's identity, `trigger_idx`, direction, the parent trigger's
  H1 input (`parent_input_idx`; None for `reversal`), the M15 probe input the
  resolver had reached (`probe_input_idx` — a `ProbeFailure`'s, or the
  successful `ResolvedStart`'s on a `geometry_failed` row; empty when it never
  got one — every `pending` / `degenerate_parent_cycle` row) — one frame per
  column since Plan E Post-E·1 (2026-09-26, PLAN_E §9.2) — and free-text
  detail. `probe_failed` covers every resolver `ProbeFailure` (a missing /
  unmappable / out-of-bounds input or end, no reference zone, a degenerate
  window (input ≥ end), a probe that did not finalize) and a resolver returning
  None; a failed
  geometry build creates **no pool entry** (MS runs first; the entry is created
  only on success — rev 1's `get_or_create`-first order left a geometry-less
  entry that consumed a `sub_id` and turned every later trigger to that key
  into a false hit). Every unresolved trigger is logged
  `[sweep] UNRESOLVED (skipping) …` — the word "skipping" is what the
  `/compare` skill's log grep keys on.
- **Graceful-degradation paths become asserts**: a missing parent-cycle
  `CTS_ESTABLISHED`, a failed H1→M15 map, `finalize_idx = None` on a finalized
  probe all raise; nothing silently degrades to an unfloored or uncapped sub.

### 17.8 Build model — geometry once; probe cache; sibling reads via the pool

**Geometry.** One builder (`build_or_get_geometry`, today's
`_build_or_get_sub_geometry` with `run_cap_abs = len(m15) − 1`) is the single
owner of pool lookup + MS run + `natural_reversal_idx` + `bos0_inner`. On a
miss it runs MS **first** and creates the pool entry only on success; on a hit
it returns the existing sub. `bos0_inner` (the first probe's BOS_0 threshold)
is **not** in the key: a later trigger reaching the same `(direction,
starting_idx)` with a different inner is `WARNING`-logged, not raised (the same
structure can legitimately be reached via an ad-hoc BOS_0 and via a sibling
CTS-zone inner). No KL derivation happens in the builder (see the sibling read).

**Probe cache — accepted approximation (design, 2026-09-19).** Key
`(parent_path, sub_TF, direction, initial_input_idx)`: "same probe" = same
direction + same initial input, where the input is the FC's price-mapped BOS
anchor, a sibling type's `ref_zone.anchor_idx`, or the reversal
handoff input. **The first probe to finalize for a key is the truth for every
later probe of that key — (a) regardless of its search bound `probe_end_idx`
and (b) regardless of its reference zone.** (b) is a real assumption: the
two-condition reset tests every candidate against the reference *inner*, and
FC / sibling / reversal triggers derive different zones at the same input
candle, so a later same-input probe run on its own reference could in
principle reset differently. The model chooses first-probe-is-truth.
Tripwire: on every hit the hitting trigger's reference inner is compared to
the cached probe's OWN reference inner (`ProbeCacheEntry.ref_inner` —
iteration 1's BOS_0 threshold; NOT the cached `bos0_inner`, which is the
FINAL iteration's threshold and moves on every reset — cold review
2026-09-20) and `[probe_cache] REF-ZONE DIFFERS …` is logged when they
differ, so a `/compare` delta is traceable to the assumption. A hit
returns `starting_idx`, `finalize_idx`, `bos0_inner` and **skips
`unified_probe`**; `[probe_cache] APPROX hit` when the bound differs. A
different-input probe that converges on the same `starting_idx` is a different
run: it probes, keeps its own finalize, and the **pool** (not the cache) dedups
the MS. A re-probe of an identical `(input, probe_end_idx)` must be
byte-identical (asserted). (A window-exact "decision_idx" hit rule and a
reference-zone-in-key were both considered and rejected.) **Skip rule:** a
re-trigger of an ended sub still runs its probe (the probe output IS the pool
key), then skips MS + downstream.

> **Measured on the first Plan C replay (2026-09-20) — THREE hits on the
> reference window.** The earlier "zero hits" statement in this subsection was
> a *prediction* that only counted pairs of H1-trigger probes; it did not count
> the **reversal-handoff probes**, whose cache keys are shifted to
> entity-absolute (`_resolve_reversal_start`: `input_abs = anchor_idx +
> slice_begin`) and therefore share `(direction, initial_input_idx)` with later
> H1 triggers. The three hits:
>
> | hitting probe | key hit (written by) | inherited `finalize_idx` / condition | tripwire |
> |---|---|---|---|
> | `first_confluence`(0,1), input 2365 | the sub-1 (`1797/−1`) reversal probe at 2470 | 2470, `no_retrace`; the FC's own Phase-2 probe was **skipped** | — |
> | `first_counter`(0,1), input 2609 | the sub-2 (`2365/+1`) reversal probe at 2829 | 2829 | — (reference inners equal) |
> | the sub-6 (`3760/−1`) reversal probe at 4200, input 4000 | `subsequent_counter`(1,2) at 4083 | 4083 | — (reference inners equal) |
>
> No lifecycle value changed on this window: `start_idx = max(probe_finalize_idx,
> trigger_idx, parent_floor_idx)` absorbed each inherited finalize (2470 ≤ the
> (0,1) floor 2611; 2829 ≤ trigger 2843; 4083 ≤ trigger 4200), and every
> `starting_idx` equals the predicted table. All three hits crossed trigger
> types (reversal ↔ `first_confluence` / `first_counter` / `subsequent_counter`),
> i.e. they could have exercised assumption (b) above — but the tripwire
> fired on NONE of them: each hitting trigger derived the same reference
> inner as the cached probe (the first replay's two `REF-ZONE DIFFERS` lines
> were spurious — the tripwire then compared against the post-reset final
> `bos0_inner`; fixed the same day). **Accepted as designed (user decision,
> 2026-09-20):** the
> cross-type sharing IS the specced key (the reversal handoff's input is part
> of the key space by this subsection's own definition); the "zero hits"
> sentence was an enumeration error, not a rule. The record's
> `probe_finalize_idx` is therefore the INHERITED value on a hit (§17.4) and
> the `trigger_idx` term of `start_idx` is what keeps such a record honest.
> The alternatives (exclude the reversal handoff from the cache; write-only
> for reversal probes; a per-probe-kind key discriminator) were considered
> and NOT adopted; revisiting any of them is its own `/compare`.

**FC probe end mapping — unchanged.** The FC probe's search bound is the
parent's confirmed-CTS **anchor** (`cts_anchor_idx`), **price-mapped** — it is
a price bound for the search, and changing it moves `starting_idx` = the pool
key (§4.3.1). It is renamed **`probe_end_idx`** everywhere (the trigger meta
already used that name — the H1 trigger field / meta key became
`parent_cts_anchor_idx` in Plan E Post-E·3, 2026-09-27): a compute bound like the run cap, unrelated to
lifecycle `end_idx`. Likewise the probe's output `ProbeResult.start_idx` (the
structural anchor) is renamed `starting_idx`. `probe_finalize_idx` mixes
native-M15 and mapped values by finalize condition (§5's table) and is taken
as is; the causally-honest "known-at" alternative (`LOH(CTS_CONFIRMED.idx)`,
+739 / +124 / +86 candles on FC(0,0)/(0,1)/(1,2), breaks sub 2365's
continuity) is recorded as **not chosen**.

**Sibling-CTS reads become pool queries** (retires the per-trigger scratch
build + mirror-then-discard and the two entity dfs). The sibling-referencing
probes (`first_counter`, `subsequent_*`) read "the most recent qualifying CTS
of the opposite-direction sub on the other lens in this parent cycle":

- candidates = that lens's records for `(S, C)` with `direction ==
  −probe_direction`, non-zero-length, **live somewhere in the idx window**
  (`start_idx <= hi` and not ended before `lo`);
- each record's events are taken from its sub's geometry shifted to
  entity-absolute, **clipped to the record's own live window ∩ `[lo, hi]`** —
  this is what reproduces today's candidate set now that geometry runs to
  the data edge (a REPLACED sub's later `CTS_UPDATED`s must not compete: sub
  `3304/−1` is replaced at 3819 but its geometry continues, and its post-3819
  CTS events would otherwise move `subsequent_counter`(1,2)@4083's
  `starting_idx` 4027 — a pool key);
- `hi` = the reading trigger's `trigger_idx`: **the window is the only thing
  keeping the read causal**; never widen it. The clip keys every
  CTS type on its MOMENT since Plan E E3b (2026-09-25; before it on `ev.idx`,
  the ANCHOR for `CTS_ESTABLISHED` — the cold-review known limit of
  2026-09-20), so a CTS whose anchor is `<= hi` but whose moment is after it is
  not a candidate. Changing it moves `starting_idx` = pool keys → its own `/compare`;
- the reference zone is built ad hoc from the winning CTS event with
  `kl_zones=[]` — behaviour-preserving, not a shortcut: subs only ever receive
  BOS zones (`source_kinds=["BOS"]`), so the primitive's CONFIRMED-zone branch
  has been dead for subs since the narrowing; the `ref=cts_confirmed` label in
  replay logs is the winning event's *type*, not evidence of a derived zone.
  Whether subs *should* get the derived CTS zone is a follow-up.

**Expected behavioural delta:** within a record's live window the sibling now
sees the natural-end CTS stream (a `CTS_UPDATED` the old per-trigger bound had
cut) → small `starting_idx` shifts are possible; each must be explained by such
an event inside the clipped window. The reversal-handoff probe reads the
reversing sub's own events the same way.

### 17.9 Rendering, storage, exports

**Storage.** One shared M15 feature frame (candles + imbalance + volume
features, prepared once) carries the pool. The confluence and counter **lens
dfs** are populated by the mirror from each sub's single projection — they are
views for the chart/export readers, not separate structural storage (§9
pointer banner). `_STRUCTURE_COLS` are painted in `start_idx` order
(later-live wins overlapping candles).

**One projection per sub, mirrored per lens.** `project_to_window` runs once
per unique sub with floor `= sub.start_idx` and cap `= sub.end_idx`
(`cap_reason = sub.end_reason`), then `mirror_lower_tf_result_to_entity_df`
into every lens df in `sub.lenses()`. The projection's `LowerTFResult.trigger`
is the sub's first live record's source trigger (its `use_case` / parent
fields are informational only). Attribution stamped on every event/zone:
`structure_path_id`, `timeframe`, `parent_tf`, **`sub_id`**, plus — informational,
never identity — `parent_sid`, `parent_cycle_id`, `use_case` from the first
record. `clip_events_to_window` deep-copies (geometry objects are shared).

**Chart (`export_m15_chart.py`).**
- Identity everywhere = **`sub_id`**. `SidRecord` is **one row per unique sub**
  (`sub_id` set, `sub_sid = None`, parent fields None, `creation_event_idx =
  starting_idx`, `start_idx`, `end_event_idx = end_idx`, `end_reason`, lenses);
  main's `SidRecord` is unchanged (`sub_sid = structure_id`, `sub_id = None`).
- **Ownership = the lifecycle window, per direction:** `owner_by_idx_dir[(candle,
  direction)]` over `[start_idx, end_idx or edge]` — the lifecycle, NOT the
  anchor. Sid-tied elements (CTS dots, swing lines, PB markers, prev-BOS lines)
  draw only where owned. REFINED by the chart review 2026-09-20/21 (§16.5):
  pre-`start_idx` elements are drawn from the anchor in the FORMING style
  (dotted/dimmed) rather than hidden, and the dotted/solid split is decided
  per SEGMENT by RECENCY, not by the lifecycle: where two structures draw over
  the same candles the most recent is solid and the prior one's whole segment
  is dotted, and a segment nothing overlaps is always solid (§16.5 item 6;
  dots follow their segments; prev-BOS lines always solid; hover keeps
  `phase=live|forming` and adds `layer=recent|prior`); the H1
  overlay on the sub charts is lifecycle-FILTERED per wave (never-live H1 waves
  not drawn, §16.5 item 5; the H1 chart unchanged); collapsed-cycle KL/POI
  zones (never active in real time) are not drawn on either chart; POI zones
  are side-tinted. Persisting elements (KL/POI rectangles; sub fibs are CSV-only) are otherwise
  unchanged: **drawn from the anchor, active from `start_idx`** (already the
  KL/POI rule).
- **Every lens draws the sub over the sub's window** — the same `[start_idx,
  end_idx]` on both charts (reverses 3.2b's per-lens "earliest trigger of this
  lens" start). Rationale: over any candle range only one structure per
  direction was truly tradeable; the charts show that one.
- Hover: `sub_id`, direction, `relative_dir` at that candle, `[start_idx,
  end_idx]` + reason, and the record list `(lens, (S,C), trigger_type,
  trigger_idx → start_idx)`. (The M15 dot tier logic was deleted
  2026-09-30 — it was never applied.)

**Exports** (decoupled from the chart loop — written even if the chart export
raises): per lens `*_M15_{lens}_subs.csv` (one row per sub on this lens) and
`*_M15_{lens}_triggers.csv` (one row per record on this lens, incl.
zero-length); pool-wide `*_M15_unresolved_triggers.csv`. `*_M15_{lens}_sids.csv`
is **removed** (split across the two above). **`sub_sid` → `sub_id` is a hard
rename** on every structural artifact (events, KL/POI/fib/WVMI meta,
`SidRecord`, CSV columns); `trigger_sub_sid` lives on records only; no alias.
`end_reason` vocabulary `{reversal, same_dir_replacement, parent_end, None}`
replaces `{reversal, lifecycle_end}` (`next_cycle` unchanged, internal); the
one consumer that branches on it (`wave_candles.py` LP-lock on `next_cycle`) is
correct for the new values by construction.

### 17.10 WVMI — on the unique sub, like zones (Plan G, 2026-09-30)

**Settled (Plan G, `plans/PLAN_G_wvmi_unique_sub.md`; canonical rule: WVMI_SPEC "Sub entities").** No gate on
subs: `project_to_window` passes `_run_downstream_pipeline(wvmi="none")`, so every rendered sub's CTS_CONFIRMED gets
a record, computed in the sub's ONE projection with its floor / cap / cap reason by the helper the main uses
(`_compute_wvmi_records`: tracker frame `df.iloc[:cap + 1]`, temp LP to the cycle's `end − 1`, `cycle_collapsed` =
`start >= end`, a lock LP outside the frame → the temp LP). Exported like zones: the mirror deep-copies each record
into every lens df the sub is on, each copy's field and meta `structure_path_id` = that lens's path. Trigger metadata
per lens (`_stamp_sub_wvmi_trigger_meta`): the lens's first §8.5 WVMI-class parent trigger inside the sub's
`[start_idx, m15_end_idx]` → `triggered_by_event_idx` (parent coords) / `_type` (None if none), `parent_path_id`
`"H1.main"` — attribution, never a gate. `multitf/sub_wvmi.py` and `persist_facade_wvmi_to_entity_df` are deleted.
Reference window: 18 sub records on 8 subs (was 8 on 5); the REVISIT below is answered by "like zones" (both lenses,
each with its own path).

**Dated history (Plan C, 2026-09-20 → 30):** WVMI's design under the pool is deferred to its own pass. Plan C implements
only the stated lean so the code runs: **one sweep per unique sub** (WVMI is a
sub property) over the sub's `[start_idx, end_idx]`, fed by the union of the
confluence and counter trigger streams restricted to the sub's lenses; the
first trigger inside the window sweeps; dedup key `sub_id`; persisted into
every lens df the sub is on. Rev 1's "per-trigger WVMI, no dedup" is
withdrawn but the WVMI pass may return to per-(sub, lens) sweeps — do not treat
§17.10 as settled [that caveat was answered by Plan G above]. WVMI record meta `sub_sid` → `sub_id`.

**REVISIT (user decision 2026-09-20 — "accept for now, revisit later"):** the
persist-into-every-lens rule puts a sub's records on BOTH lens CSVs/charts
(measured: sub `2639/−1`'s three reversal-swept records now also on the
counter lens; sub `4027/+1`'s two `subsequent_counter`-swept records also on
confluence). The WVMI pass must decide whether a record belongs to the sub
(both lenses) or to the sweeping trigger's lens only.

### 17.11 Validation — sequencing, predicted table, `/compare`

**Sequencing (one cause per `/compare`).** `git revert 9fd3143` (tree = Stage
3.2a; landed `5a658dc`, save `20260919_215955_5a658dc`, byte-identical to
`c932610`) → **Plan A** (**LANDED 2026-09-19** — bounded MS runs have
truncation semantics at five sites: `D`, the `is_range_confirm_idx` label,
the detector's visible length, the reversal-watch expiry, the resolvers'
frame; post-run assert + property test; the only production change was the
first_confluence probe's Phase-2 run — FC(1,0) finalize 2844 → 2843,
`starting_idx` 2803 unchanged, H1 byte-identical; `plans/PLAN_A_ms_bounds_leak.md`)
→ **Plan B** (**LANDED 2026-09-20**, `189c127` — `second_cts_reached` is a true early
stop in `unified_probe` Phase 2: `MarketStructure(stop_after_cts_established=2)`
ends the run at the first quiescent point after the 2nd `CTS_ESTABLISHED`,
finalize = its `confirmed_at`; byte-identical on all 21 CSVs — nothing past
the 2nd `CTS_ESTABLISHED` is read; `plans/PLAN_B_double_cts_early_stop.md`)
→ **Plan C** (this section; one behavioural change,
one replay). Each with its own replay, `/compare`, chart-review pause and
`/commit-save`.

**Plan C acceptance** (the pool table is checked, not eyeballed):

1. The emitted `[pool]` log lines / `_subs.csv` / `_triggers.csv` **equal the
   hand-derived predicted table** in `memory/reference_pool_redesign_groundtruth.md`,
   matched by `(direction, starting_idx)` — on the reference window **8 subs /
   11 records / 4 unresolved** from the baseline's 16 per-trigger sids. The
   same table is a unit test (`test_lifecycle_sweep_predicted_table.py`, stubbed
   probe + geometry, < 1 s) — the internal checkpoint that validates the sweep
   without a replay.
2. H1: all CSVs byte-identical.
3. M15 intended deltas: `_sids.csv` gone, three new CSVs; subs `2803/−1`,
   `2915/−1`, `3230/+1` absent (their parent cycles are degenerate — baseline
   drew them as `inactive` outlines); sub `2639/−1` on the counter chart from
   2829 (was 2843); `4027/+1` on confluence from 4083 overlapping `3760/−1`
   over [4083, 4200] (opposite directions); `end_reason` vocabulary;
   `sub_sid` → `sub_id`; `lifecycle_end` absent; possible small
   `starting_idx` shifts from the sibling read (§17.8, each explained); exactly
   three sub-cycle lifecycle starts +1 candle (§17.6 — three lens rows, two
   unique sub cycles: `2639/−1` is on both lenses; measured: one visible, the
   other masked — below). Own reversals for the
   subs whose baseline build was cut at a parent bound may now be discovered
   with the run cap at the data edge — if one lands before the listed end,
   that end moves earlier and a successor appears (intended, document it).
4. Anything else is a regression. The `/compare` skill reports NEW / MISSING
   files explicitly (no baseline exists for the new CSVs on first run).
5. **Pause for chart review** before `/commit-save` — ownership and hover
   changed.

> **Measured — first Plan C replay + `/compare` vs the Plan B save
> `20260920_104606_189c127` (2026-09-20; landed as `afaa326`, save
> `20260921_125218_afaa326` — the chart-review refinements of §16.5 changed
> only the charts, every CSV of the save is byte-identical to that first replay):**
> 1. pool = **8 subs / 11 records / 4 unresolved** (all four
>    `degenerate_parent_cycle`); every sub window and every record
>    `start_idx` / `end_idx` / `end_reason` equals the predicted table;
>    `sub_id` 0–7 in creation order = `454/+1, 1797/−1, 2365/+1, 2639/−1,
>    3304/−1, 3621/+1, 3760/−1, 4027/+1`. No `starting_idx` shift from the
>    sibling read on this window (every key equals the table).
> 2. H1: 8 of 9 CSVs byte-identical; `_wvmi.csv` differs in its header only
>    (the shared exporter's column `sub_sid` → `sub_id`).
> 3. Moment rule (§17.6): **one** visible +1 shift (sub `454/+1` cycle-1 end
>    1223 → 1224); the two predicted 2828 → 2829 shifts are masked — the
>    record floor 2829 is identical on both lenses, so the clamped value is the
>    same before and after. Sub WVMI: one sweep per sub, records persisted into
>    every lens df the sub is on (sub `2639/−1` rows now also on counter, sub
>    `4027/+1` rows also on confluence) — accepted for now, to be revisited
>    in the deferred WVMI pass (§17.10). Probe cache: three hits (§17.8 —
>    accepted as designed, user decision 2026-09-20).
> 4. Replay 74.7 s wall (`multi_tf_dual` 10.2 s; three charts 60.3 s). Chart
>    element counts: H1 107/250 unchanged; M15 counter 174/128 (was 198/133),
>    confluence 321/238 (was 342/222). Tests: 633 passed + 1 strict xfail.

**Tests that break by design** (rewritten in the Plan C commit): the pool-wide
cross-chain end (`test_cross_chain_reversal_ends_active_counter_sub` — inverted:
a confluence-lens successor does not end a counter-lens record), the
"overextension is the known edge" test (becomes the guard for the fixed
semantics), `test_two_cycles_sub_sid_resets` (→ `trigger_sub_sid` resets per
(lens, parent)), the `finalize_lifecycles` / `select_lifecycle_end` tests, the
14 `test_sub_chain.py` chain tests, and the sibling-read tests that build
sibling dfs. `test_sub_id_is_monotonic_and_stable` must survive unchanged.

### 17.12 Out of scope (v1)

- **Main** (`H1.main`) stays on `compute_structure`; the pool is
  subordinate-only. The moment-not-anchor rule (§17.6) does reach main through
  `structure_lifecycle` — byte-identical on the reference window.
- **Deeper nesting** (M5 under an M15 sub): the identity tuple is recursion-ready
  but M5 nesting is not built now.
- **WVMI design** under the pool (§17.10) — DONE (Plan G, 2026-09-30); **live-mode pool GC** (evict when
  parent ended + no live reference), and the **Phase 3 per-candle dual-lens
  driver** (the sweep's step body is its loop body) — leave hooks.
- `knowable_at_idx` keys `BOS_CONFIRMED`, `CTS_ESTABLISHED` and pattern-path
  `CTS_UPDATED` on their moment `confirmed_at` (the last two since Plan E E3b,
  2026-09-25 — closed for them); `REVERSAL_CANDIDATE` (applies at
  `meta["apply_idx"]`) still straddles and can be half-clipped at a window edge.
  Not fixed here — note any half-clipped reversal seen during chart review. (Half
  closed 2026-09-27: the fib terminal / prev-BOS line read the realised reversal —
  LANDMINES "Sub-Structure Pool: Run Cap ≠ Lifecycle End; Knowable-At Clip on Render".)
  Measured 2026-09-30 on the reference window: **0 half-clipped** — the 4 kept
  `REVERSAL_CANDIDATE`s (confluence subs 0 / 1 / 2 / 6) each apply at their own sub's
  window end (the reversal that ends the sub: 1940 / 2470 / 2829 / 4200), and the
  subs capped for another reason (3 `parent_end`, 4 and 5 `same_dir_replacement`)
  hold none. A trigger item in the register (`IDEA_PARKING_LOT.md` §E).
- Whether subs should receive the **derived** CTS KL zone as a probe reference
  (§17.8 — the branch is dead for subs today).

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
