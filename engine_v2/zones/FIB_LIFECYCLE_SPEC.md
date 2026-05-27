# FibState Lifecycle Spec — Design & Implementation Plan

> **Status:** Design complete (2026-05-27). **Session 1 / Step 1 IMPLEMENTED
> (2026-05-27, byte-identical) — Sessions 2 & 3 NOT started.** This document is
> the cold-readable, canonical design produced from a full design session. It is
> the FIRST spec doc for `zones/fib_tracker.py` (previously the only description
> was the `cross_cycle` mode memory entry). Per-session status lives in §12.
>
> **Scope:** Bring `FibState` onto the active/inactive/ended **lifecycle
> convention** (ARCHITECTURE.md "Lifecycle state convention"), which `POIZone`
> and `KLZone` already follow. FibState is the **last** lifecycle-like object
> off the convention.
>
> **Line numbers** below are as of 2026-05-27 and are hints — anchor on the
> method/field names, which are stable.

---

## 0. Why this exists / the one-sentence summary

Today `FibState.active` is a single stored boolean that does **double duty** —
it mixes a *reversible condition* (are there unfilled imbalances?) with
*irreversible terminal* reasons (`new_cycle`, `scenario1_revert`, `cross_failed`,
`lifecycle_end`). That overload loses information (a consumer can't tell
"imbalances temporarily filled, could revive" from "this cycle is permanently
done"). The migration **separates the axes**: `active` becomes condition-only,
terminal moves to `end_idx`/`end_reason`, `locked` stays its own axis, and a
3-state `status` is **derived**. The hard part is NOT the separation — it's the
**cross-fib versioning**, which this design resolves by making the **cycle the
lifecycle identity** and treating **versions as an internal sub-axis**.

---

## 1. Background — the convention, and why FibState is hard

The convention (ARCHITECTURE.md) gives a long-lived object **two orthogonal
axes**, stored separately:

| Axis | Nature | Reversible? | Stored as |
|---|---|---|---|
| **active / inactive** | condition-state ("are my conditions true at candle t?") | **yes** | `activation_history` |
| **ended (terminal)** | time-based irrelevance | **no** | `end_idx` / `end_reason` |

`status ∈ {active, inactive, ended}` is a **derived** label (`ended` wins).
POIZone and KLZone already do this. The reason FibState was deferred is the
**cross-fib versioning**: a single logical fib for a cycle is realized as
multiple `FibState` records with *different geometry* (different start anchors),
and the convention assumes one fixed object flipping over time. The resolution
(Section 3) is to define the **lifecycle identity = the cycle**, with versions
as an internal axis below the active/inactive flip.

---

## 2. Current FibState shape (what exists today)

`FibState` (frozen dataclass, `fib_tracker.py:29-58`) fields: identity
(`structure_id`, `cycle_id`, `struct_direction`), anchors (`bos_idx`,
`bos_price`, `cts_idx`, `cts_price`), state (`active: bool`, `locked: bool`),
the `fib` retracement, `meta: dict`, and `cts_history: tuple`.

**`active` is overloaded.** It is set `False` by, today:

| Reason (stored in meta) | Set where | True nature |
|---|---|---|
| `all_imbalances_filled` (`reason`) + `deactivated_at` | `_update_fib_cts:1120-1123` | **condition** (reversible — can reactivate `:1116-1119`) |
| `own_imb_filled` (cross) | `_deactivate_active_cross` ← `_m15_cross_check:1683-1687` | **condition** (reversible) |
| `new_cycle` (`obsolete_reason`) | `_activate_fib:786`, `_obsolete_prev_cycle_all_fibs:1572` | **terminal** |
| `scenario1_revert` (`deactivated_by`) | `_deactivate_cycle0_fib:1480` | **terminal + invalidation** (hidden entirely) |
| `cross_failed` (`deactivated_by`) | `_deactivate_cross` ← `_m15_cross_check:1725` | **version-internal** (cross→single handoff) |
| `cross_shortened` (`deactivated_by`) | `_deactivate_cross` ← `_m15_cross_check:1755` | **version-internal** (re-anchor, NOT an end) |
| `lifecycle_end` / `reversal` (`deactivated_by`) | external cap, `entity_df_mutation.py:669-678` | **terminal** |

`locked` is already a **clean separate axis**: `False` at creation, `True` at
CTS_CONFIRMED (`_handle_*_cts_confirmed`, `meta["locked_at"]`). Once locked,
`_update_fib_cts:1029` early-returns — anchors and condition are **frozen**.

**There is no history stored today.** Each event overwrites the current
`active`/`locked` via `replace`; transitions are only `print`ed, never persisted.

---

## 3. Target model — cycle identity, internal versions, orthogonal axes

### 3.1 The lifecycle identity is the CYCLE `(sid, cycle_id)`

A fib is tied to a cycle exactly like the cycle's BOS zone and CTS zone. **Each
cycle has exactly one active fib (one active version).** So:

- `start_idx` / `end_idx` of a fib are tied to the **cycle's** start/end
  (with one subordinate-only exception, Section 6/7).
- This applies to cross fibs too: a cross fib is **assigned to the most-recent
  cycle** (`target_cycle`) even though its start anchor reaches back to a prior
  cycle. When a new cycle begins, the cross is recomputed for the new cycle.

### 3.2 Versions are an INTERNAL axis below the active/inactive flip

Within a cycle there can be multiple **versions**, but **only one is active at a
time**. A version is a **distinct start-anchor identity** (Section 4). When a
version changes (a handoff), the cycle's fib does **not** end and does **not**
flip active/inactive — that version simply becomes inactive/superseded while the
cycle fib continues as the new version.

### 3.3 The three axes (stored ORTHOGONALLY — this is the whole point)

| Axis | Field(s) | Meaning |
|---|---|---|
| **condition** | `active` (repurposed: condition-ONLY) + `activation_history` | unfilled sd-direction imbalance in the **live version's** `[bos, cts]` range as of t. Reversible. |
| **terminal** | `end_idx`, `end_reason` | cycle end. Irreversible. |
| **computation** | `locked` (UNCHANGED) | CTS_CONFIRMED → bounds frozen. A genuinely separate axis (analogous to WVMI's `created/updated/locked`). |
| **derived** | `status ∈ {active, inactive, ended, disappeared}` | computed from `end_idx` + condition. **`locked` is NOT folded into `status`.** Lossy convenience label (like POI's). |

`status` derivation (cycle-identity level): `ended`/`disappeared` if `end_idx`
reached (the latter for invalidation-class `end_reason`); else `active`/`inactive`
from the condition. **`status` lives at the cycle-identity level**, NOT per
record (Section 9.1 explains why the chart gate must use raw per-record flags
instead).

### 3.4 Mapping of today's deactivation reasons onto the new model

| Today's reason | New bucket | Representation |
|---|---|---|
| `all_imbalances_filled`, `own_imb_filled` | **condition flip** | `active=False`, an `activation_history` entry; reversible |
| `new_cycle`, `reversal`, `lifecycle_end` | **terminal** | `end_idx` + `end_reason`; `status="ended"` |
| `scenario1_revert` | **terminal + invalidation** | `end_idx` + `end_reason="scenario1_revert"`; `status="disappeared"` (Section 10) |
| `cross_shortened`, `cross_failed`→single, re-create-after-inactive | **version-internal** | NOT a cycle-level event; handled inside the version axis (Sections 4, 8.3) |

---

## 4. Versioning (detailed)

A cross fib spans `[BOS_{earliest_x} … CTS_{target_cycle}]`. The earliest
eligible start cycle `earliest_x` is found by the dead-cycle backward walk
(`_m15_cross_check:1693-1717`). Keys: single = `(sid, cycle)`; cross =
`(sid, cycle, "cross", version)`. `_cross_version[(sid,cycle)]` = highest
version; `_get_latest_cross` returns the current one.

**A version is defined by its START anchor.** Transitions:

| Situation | Today's action | Version effect |
|---|---|---|
| CTS (right/target) anchor extends to new extreme | `_m15_extend_cross_anchor:1750` | **same version, in-place** (no bump) |
| start anchor moves forward (prior cycle died → shrink), `earliest_x > active_x` | `cross_shortened` + `_m15_create_cross:1755-1759` | **new version (v+1)** |
| no prior cycle eligible, `earliest_x == target` | `cross_failed:1722-1732` | cross ends; established-phase → single fallback |
| no active cross yet | `_m15_create_cross:1740` | create **v0** |

**DECISION — same-anchor reactivation is IN-PLACE (a change from today).**
Today (`_m15_cross_check:1739-1743`) a revival after `own_imb_filled` creates a
*new* version even when the start anchor is unchanged. **We change this:** a
revival of an *existing* start anchor reactivates the **same** version in place;
a new version is created **only** when the start anchor actually moves. Rationale:
keep versions purely anchor-defined, don't proliferate. (See Section 12 — this is
a **behavior change**, not byte-identical under the old chart.)

**Version count:** Scenario 2 (subsystem B) = at most **2** anchor-versions
(`BOS_0` cross vs `BOS_1` normal). Subordinate cross (subsystem A) = up to **N**
(one per start cycle it shrinks through). Single/normal fibs = **1**.

---

## 5. The two subsystems (A, B) + the "project, don't unify" decision

The "one cycle, multiple representations, one active" abstraction is realized
**two different ways** in the code, and they are NOT interchangeable in storage:

| | **A — subordinate `cross_cycle` mode** | **B — H1 main Scenario 2** |
|---|---|---|
| Where reps stored | **all** versions in `_fibs` (`(sid,cyc,"cross",v)` + `(sid,cyc)` single fallback) | candidates in `_cross_cycle_data[sid]`; **only the active winner** mirrored to `_fibs[(sid,1)]`; loser stays in scratch |
| Transition dynamics | monotonic supersede (old anchor permanently dead) | toggle (cross ↔ normal, either can re-win) |
| Versioning mechanism | integer `_cross_version` | named slots, no integer version |
| Reaches charting | all versions (dead drawn faded today) | only the active one |
| On which structure | subordinate (M15) | **main (byte-identical zone)** |

`_cross_cycle_data[sid]` contents (H1-main only): `["cycle0"]` = a plain **dict**
`{bos_idx, bos_price, cts_idx, cts_price, struct_direction, has_unfilled, locked}`
(Scenario-2 condition inputs); `["normal_cycle1"]` = a **FibState**;
`["cross_cycle"]` = a **FibState**.

**DECISION — DO NOT unify the storage.** Collapsing `_cross_cycle_data` into
versioned `_fibs` entries would refactor main-structure fib selection — the
byte-identical-risk zone. Instead, maintain the lifecycle layer
(`activation_history` + derived `status`) at the **cycle-identity level** in a
**tracker-level dict** that *both* subsystems feed (Section 8.4). That makes the
lifecycle **uniform above the storage** while leaving A's versioned `_fibs` and
B's `_cross_cycle_data` untouched. **This is "project, don't unify."** The two
subsystems are conceptually the same model (versions per cycle, one active); only
their storage differs, and the projection bridges them.

> The cross→single (`cross_failed`) handoff in A and the cross↔normal fallback
> in B are treated identically at the cycle level: a **handoff / `reanchor`**, NOT
> a cycle-level active/inactive flip (the cycle fib stays active, just changes
> representation).

---

## 6. Start anchor

**`start_idx` = the cycle-fib IDENTITY's FIRST version's birth idx.** Normally
that is the cycle's lifecycle-start (the CTS-established floor, as zones use).

**The one exception — pre-established cross (SUBORDINATE-ONLY).** In `cross_cycle`
mode a cross fib for cycle `n+1` is created during cycle `n`'s **tail**
(pre-established phase: after `CTS_n` CONFIRMED, before `CTS_{n+1}` ESTABLISHED,
on `CTS_THRESHOLD_UPDATED`). At that point the cycle-`n+1` fib is **valid and
active**, so its `start_idx` legitimately **precedes its own cycle's CTS-
established**. This is the **only** exception to "cycle elements follow cycle
start." Rationale: the fib really is live and drawn then, so the start must
reflect it.

Corollaries:
- The `cross_failed` → single handoff **keeps the early start** — a single that
  descends from a cross lineage inherits the identity's early start (start
  belongs to the identity, not to a version).
- **Main (h1) has no pre-established phase** (`_m15_phase` is never set on the
  main path; the Scenario-2 cross is created at `CTS_1` ESTABLISHED). So main
  fibs always start at cycle-start — the exception is subordinate-only. This is
  why the start-anchor change **does not** touch H1-main fib timing.
- Implementation: the value already exists as `meta["activated_at"]` on the
  first version (`_m15_create_cross:1818`); take it from the **earliest** version
  and never overwrite it on later version creation.

---

## 7. End anchor — Option A (chosen)

**`end_idx` = the cycle's end.** The subtle case is the mirror of Section 6.

A next-cycle (`n+1`) pre-established cross is created *during cycle n's tail*, and
that creation is **exactly** when `_obsolete_prev_cycle_all_fibs:1827` retires
cycle `n`. So:

> **Option A (CHOSEN):** when an `n+1` pre-established cross forms, it
> simultaneously **starts** the new fib AND **ends** the previous cycle's fib
> (whether the previous was single OR cross), at the **pre-established cross
> creation idx**.

Consequences (accepted):
- Cycle `n`'s fib end is **earlier** than the zone end for cycle `n` (zones end
  at the next-cycle `CTS_{n+1}` ESTABLISHED idx, `poi_zones.py:476-482`). So
  **fib cycle-end diverges from zone cycle-end at this one boundary.** This is
  the price of the honest early *start*; start and end shift earlier *together*.
- It preserves the **"exactly one active fib per structure"** invariant (cycle
  `n` dies the instant `n+1` is born — no overlap). The rejected Option B
  (pin `n`'s end to the zone end) would have created a brief two-active-fib
  overlap in the gap. We chose A.

**Confirmed cases when NO `n+1` cross forms before `CTS_{n+1}` ESTABLISHED:**
1. **No single fib can exist yet** — verified: the only single-creation path in
   `cross_cycle` mode (`_activate_or_update_single_m15`) is reachable only from
   the `cross_failed` branch *after* the pre-established early-return
   (`_m15_cross_check:1726-1727`); and `_handle_cross_cycle_cts_established:338`
   flips phase to `established` before any single can be created. So singles
   only ever appear at/after `CTS` established.
2. **Cycle `n`'s fib (single OR cross) end = passed-through cycle-`n`-end value**
   (= next-cycle `CTS_{n+1}` ESTABLISHED when non-terminal). Zone-consistent.
3. **Cycle `n+1`'s fib start = its first-active = cycle start.** Zone-consistent.

**Scope of Option A's early-end:** same-sid next-cycle progression only — which,
because pre-established is subordinate-only, is a **subordinate phenomenon**. For
**every other terminal** — reversal (`sid → sid+1`), parent-cycle change / parent
reversal (subordinate `lifecycle_end`) — `end_idx` = the **passed-through
cycle-end value**, exactly as KL/POI. **Precedence:** passed-through terminal
(reversal/parent) wins; absent that, the same-sid next-cycle boundary sets it
(early if a pre-established cross forms, else next-cycle CTS-established).

---

## 8. `activation_history` + derived `status` (the condition axis)

### 8.1 Two independent axes inside the cross logic

- **Condition axis** (active/inactive): "unfilled sd-direction imbalance in the
  **live version's** `[bos, cts]` range as of t." Driven by the fib's own
  imbalances (`all_imbalances_filled` / `own_imb_filled` / reactivate).
- **Version/anchor axis**: driven by prior-cycle deaths (the dead-cycle walk
  shrinking the span). **Independent** of the condition. **Version handoffs are
  NOT cycle-level active/inactive flips** — the cycle stays active across them.

### 8.2 KEY: `activation_history` is NOT load-bearing for fib

Unlike POI (whose `activation_history` is *walked per-candle* by the proximity
gate and chart fill-stretches), **no fib consumer walks a fib history:**

- **Chart** is a static end-of-data snapshot; it needs only the **final**
  condition value + `locked` (no stretch fills — Section 9.3). It does NOT need
  a per-candle history.
- **POI** consumes fib **geometry** + the **final** `active`/`locked` flags as a
  gate (`poi_zones.py:434`), and computes its **own** activation history
  independently (it never reads the fib's history). POI also derives its **own**
  `end_idx` from structure events (`poi_zones.py:471-482`), NOT from the fib — so
  POI is insulated from the fib's terminal axis except for that gate.

Therefore fib's `activation_history` is an **informational / debug** artifact, not
load-bearing. The **essential** migration work is the axis separation (Section 3.3);
the history is a cheap byproduct (Section 8.5 — kept, not dropped).

### 8.3 Granularity — COARSE (event-granularity), DECIDED

Today the fib re-checks `has_unfilled` only at **CTS events** (`_update_fib_cts`
on `CTS_UPDATED`), NOT every candle — so its condition is coarse (an imbalance
filling *between* CTS events isn't reflected until the next event). **Decision:
keep this coarse, event-granularity logging** — record flips where the tracker
*already* detects them, add **no** per-candle walk. Rationale: no consumer needs
per-candle fib accuracy; it matches current behavior (clean `/compare`); and it's
consistent-in-principle with POI (both store *sparse* flip lists — only the
*computation* resolution differs, matched to each consumer's needs). The
per-candle / live-mode concern is a **separately-deferred** item (the `is_filled`
incremental state machine) and explicitly out of scope here. **User confirmed
coarse is fine for both backtest and live.**

### 8.4 Where the history lives — tracker-level `(sid, cycle)` dict

`activation_history` is a **cycle-identity** artifact, but FibStates are
**version** instances. So it must NOT live on a single version object. Maintain a
**tracker-level dict** `_activation_history[(sid, cycle)]`, appended as events
fire — **both subsystems (A and B) feed the same dict** keyed by `(sid, cycle)`.
This mirrors the existing `cts_history` accumulation pattern (FibState already
grows a `cts_history: tuple` via `replace`, just at the wrong granularity). At the
end, attach the cycle history to the representative FibState or expose it via a
tracker method. **This is what resolves the A/B unify question (Section 5): the
history projection unifies the subsystems above their split storage.**

### 8.5 Flip entries + handoff recording

Each flip entry: `{idx, active, reason}` (+ optional `version` debug breadcrumb).
`reason ∈ {"activated", "imbalance_filled", "imbalance_reformed", "reanchor", …}`.
**Terminal reasons go to `end_reason`, NOT here** (that's the axis separation).

**DECISION — LOG the handoff active with `reason="reanchor"`** (do not suppress).
When a version handoff occurs with no inactive gap (cycle stays active), record an
`active=True` entry tagged `reanchor`. Rationale:
- Makes `activation_history` **self-sufficient** for a (hypothetical) fib
  `confirmed_idx` = "the current version's activation" = the handoff idx (the
  *shape* was confirmed at the handoff, even though the cycle never deactivated).
- Documents anchor changes for debugging.
- **Cost (cosmetic only):** breaks strict A/D alternation (`A…A` is now possible),
  unlike POI/KL. **Harmless** because the same list serves two correct reads:
  - **Active stretches** (`active_stretches_from_history`, `poi_lifecycle.py`):
    the helper *ignores* a consecutive `active` (the `active_start is None`
    guard), so stretches stay correct.
  - **`confirmed_idx`** (most-recent-active scan): *honors* it → reports the
    handoff idx.

**DECISION — KEEP `activation_history`** (rather than store only the final
condition flag). It's a cheap event byproduct, gives debug parity with POI/KL,
makes the `confirmed_idx` derivation self-contained, and future-proofs a live
consumer. It is **not** load-bearing, so this is a low-cost nicety.

### 8.6 Terminal is NOT in the history

The terminal lives in `end_idx`/`end_reason`; `status` lets `ended`/`disappeared`
win. So `activation_history` ends on a *condition* event (often a trailing
`active`), and any open active stretch is **capped at `end_idx`** by the consumer
(mirrors POI's `active_stretches_from_history` capping at `render_end_idx`). No
"ended" marker is pushed into the history.

---

## 9. Charting

### 9.1 The draw gate is PER-RECORD (not status-based)

The chart draws **records** (version FibStates). The gate must distinguish the
live/locked version from dead ones *within* a cycle — and the cycle-identity
`status` cannot do that (all versions of a live cycle share `status="active"`, so
a status-keyed gate would draw every version, dead ones included). So the gate is
**primarily per-record**:

> **Draw a record iff (as IMPLEMENTED, Session 2 — `export_plotly.py`):**
> `status != "disappeared" AND ( record.locked OR (record.active AND status not in {"ended","disappeared"}) )`

The **raw per-record flags** `active`/`locked` do the *version distinction*
(picking the live-or-locked representative, dropping superseded dead versions
— which carry `active=False`). The **cycle-level `status`** is consulted ONLY
for the *terminal* (`ended` / `disappeared`) check, which is genuinely
cycle-level (all version records share it). This split is the whole point:
status for the cycle terminal, per-record flags for which version.

> **Reconciliation note (Session 2).** The original design wrote this gate as
> the bare `not scenario1_revert AND (active OR locked)`, on the implicit
> assumption that an ended fib also has `active=False`. But Session 2 makes
> `active` **condition-only** — terminals (new_cycle / reversal / lifecycle_end)
> set `end_idx`/`end_reason` and leave `active` alone (Section 3.3 / §11). So an
> ended cycle's open unlocked record keeps `active=True`; the bare
> `(active OR locked)` would wrongly **draw** it instead of vanishing it (§9.2
> cat 3). The implemented gate adds the `status not in {ended,disappeared}`
> terminal check to fix this, while still using per-record `active`/`locked`
> for version distinction. (Superseded **version-internal** records —
> `cross_failed` / `cross_shortened` — still carry per-record `active=False`,
> so they vanish via the `active` term even while their cycle is alive.)

> NB on `status` vs the gate: this is subtle — `status` is a single cycle-level
> label; the per-record flags are finer-grained. The gate uses BOTH, at the
> levels each is meaningful. Do not conflate them.

### 9.2 What this gate CHANGES (intentional, not byte-identical)

Today the chart draws **every** fib in `get_fibs_for_charting()` (all of `_fibs`
except `scenario1_revert`), styling `is_active = active and not locked` → active
bright, **everything else faded** (`export_plotly.py:2515-2522`). Under the new
gate, **unlocked-and-not-active records VANISH** (were faded). Three categories:

1. **Dead cross versions** (`cross_shortened`, unlocked) — the bulk of the
   **M15/sub** cleanup (the "dead-version trail").
2. **Inactive-unlocked fibs** — activated then imbalance-filled, never confirmed
   (never locked).
3. **Ended-unlocked fibs** — capped by `lifecycle_end`/`reversal` while
   `active and not locked` (the `entity_df_mutation.py:669-678` cap); open sub
   crosses that never established a CTS — **common on M15/sub.**

**Impact distribution:** H1 **main** barely changes (main fibs almost always
lock; only a trailing still-forming cycle whose imbalance filled could vanish).
**M15/sub** changes substantially. `scenario1_revert` is **not** a change — it
was already hidden, just via a different mechanism now (Section 10).

What still shows: **active** fibs (bright) and **locked** fibs of any cycle
(shown, faded-by-tier — the confirmed historical records, incl. locked-then-
`ended`).

### 9.3 NO stretch fills for fib

Decision: a fib is shown when **active**, **vanishes** when inactive, and is
**always** shown once **locked**. No active-stretch fills (unlike POI). Rationale:
**fibs differ from POI** — an *inactive* fib has no levels worth showing (there is
no fib to draw), whereas an inactive POI zone is still a meaningful price region
(so POI renders it faded). In live/replay this reads as the fib **appearing and
disappearing** as it flips active/inactive.

### 9.4 Tiers — 2-tier now, 3-tier parked

Because inactive fibs vanish, the only things ever drawn are **active** (bright)
and **locked** (faded) → naturally **2-tier**. The zones' third tier exists only
to distinguish *inactive* members by sid-recency, and fib has no visible inactive
members. A 3-tier split (recent-locked vs prior-locked) is a pure aesthetic and is
**parked with the uniform-opacity refactor** (Section 13).

> **Reference — how zones tier today (the template if/when fib adopts 3-tier):**
> POI uses a status-driven 3-tier opacity (`export_plotly.py:2306-2312`):
> `status=="active"` → brightest; else most-recent sid → mid; else faded; plus
> active-stretch fills + `end_time`-bounded extent. **Planned refinement** (a
> separate session, see Section 13) re-keys those tiers onto pure lifecycle:
> brightest = cycle **not ended**; mid = cycle ended but **sid not ended**;
> faded = **sid ended**.

---

## 10. `scenario1_revert` → `disappeared` (the one "hide entirely" case)

### 10.1 What `scenario1_revert` is

It **reverts the Scenario 1 determination, TRUE → FALSE** (H1-main only; the
`cross_cycle` sub path has no Scenario-1 logic).

- **Scenario 1** (h1 main, sid ≥ 1): `CTS_0 idx >= reversal_confirmed_idx` →
  cycle 0 is treated as a legit post-reversal structure → the **cycle-0 single
  fib** activates. If it *stays* TRUE, cycle 1 also gets a normal single fib →
  this is the **"2 single fibs, no cross"** outcome.
- **The revert:** at `CTS_1` ESTABLISHED, if `BOS_1` reaches into the **previous
  structure's last BOS zone** (price crosses its outer edge —
  `_should_revert_scenario1:1450-1469`), the clean-reversal premise is broken →
  Scenario 1 flips to FALSE → the cycle-0 fib is killed
  (`_deactivate_cycle0_fib`, `deactivated_by="scenario1_revert"`) → cycle 1 is
  re-routed to Scenario 2 (cross) / Scenario 3 (single).

So `scenario1_revert` **originates from** the Scenario-1-TRUE (2-single-fib)
setup, and it is the branch that **aborts** it. It's an **invalidation /
retraction** — "the premise that created this fib was retracted," distinct from a
normal `ended` (historical) and from condition-`inactive`.

### 10.2 Why it needs special handling (the gate would wrongly show it)

The reverted cycle-0 fib is **`locked`** (CTS_0 was confirmed before CTS_1 could
establish). So under the per-record gate `(active OR locked)`, `locked=True` would
make it **render** — the opposite of what we want. Today it doesn't show only
because of the explicit filter in `get_fibs_for_charting:1971`
(`deactivated_by != "scenario1_revert"`); remove that filter and the gate would
draw it (faded, since `is_active=False`).

### 10.3 The mapping — reuse the existing (vestigial) `disappeared` filter

The chart **already** has a `status != "disappeared"` filter
(`export_plotly.py:2294`, `export_m15_chart.py:1107, 1944`), and
`POI_ZONES_SPEC.md:393` documents `disappeared` = "NOT rendered." **But nothing
sets it anymore** — the Phase-3 convention migration replaced POI's old
"disappeared" (hide-entirely) with "render faded as inactive," so the POI status
derivation (`poi_zones.py:554-560`) only produces `ended`/`active`/`inactive`. The
filter is currently **dead code**.

**Mapping:** `scenario1_revert` → `end_idx = revert_idx`,
`end_reason = "scenario1_revert"`, derived **`status = "disappeared"`** (the
derived label for **invalidation-class** `end_reason`s — terminal **and**
suppressed; NOT a new storage axis). The existing `status != "disappeared"` filter
then hides it, for fibs and zones alike, **overriding** the `(active OR locked)`
gate (the filter runs first). This **preserves today's behavior exactly**
(reverted fibs stay hidden → clean `/compare`), revives a dormant-but-correct
mechanism, and handles invalidation uniformly.

### 10.4 `disappeared` ≠ normal inactive-vanish

Two distinct "vanish" behaviors that must NOT be collapsed:

| | trigger | reversible? | live/replay |
|---|---|---|---|
| **inactive vanish** | condition (imbalance fill), via the gate `(active OR locked)` | **yes** | reappears when it reactivates |
| **`disappeared`** | terminal invalidation (`scenario1_revert`) | **no** | gone for good |

Identical on a static end-of-data chart; different in live/replay. `disappeared`
is reserved for **terminal invalidation** (today: only `scenario1_revert`); the
ordinary active/inactive flicker is just the condition axis driving the gate.

---

## 11. Consumers (current reads → post-migration)

| Consumer | Reads today | Post-migration |
|---|---|---|
| **POI** (`poi_zones.py:434`) | geometry + final `active`/`locked` as gate (`if not active and not locked: skip`); derives its OWN `end_idx` + activation history | **per-record** gate `(active AND end_idx is None) OR locked` — see reconciliation note |
| **Chart H1** (`export_plotly.py:2504-2654`) | geometry + `is_active = active and not locked`; hover label "active/locked/inactive" (`:2625`) | per-record gate (Section 9.1); `is_active` (bright) = `active AND not locked AND status=="active"`; hover label from `status`+`locked` |
| **Chart M15** (`export_m15_chart.py`) | fib lines OFF by default (`"fib": {"lines": False}`); `sid_fibs` fetched but unused | unchanged — no fib rendering to gate |
| **Cap** (`entity_df_mutation.py` open-sub-fib cap) | sets `active=False` + `deactivated_by` on `active and not locked` fibs | set `end_idx` + `end_reason` + `status` instead (leave `active`); recompute cycle `status` across all version records of capped cycles |
| `get_fibs_for_charting` | returns all `_fibs.values()` except `scenario1_revert` | drop the special filter; rely on the uniform `status != "disappeared"` filter |

**Reconciliation note — the POI gate is PER-RECORD, not `status=="active" OR
locked` (Session 2).** The original design proposed `status=="active" OR locked`
and claimed byte-identical. That holds for **H1-main** (fib_mode `"h1"` has one
record per cycle — no cross versions in `_fibs`), but NOT for **subordinate**
structures: a superseded (dead) cross version of a *still-live* cycle carries
per-record `active=False` yet inherits the cycle-level `status="active"`, so the
cycle-status gate would wrongly **process** it (extra/incorrect POIs on subs).
This is the exact reason §9.1 uses per-record flags for version distinction. The
implemented byte-identical-preserving form is therefore per-record:

> `(fib.active AND fib.end_idx is None) OR fib.locked`

— equivalent to today's `(active OR locked)` because (a) the only records whose
`active` value changed are terminal-ended ones (terminals stopped resetting
`active`), and the added `end_idx is None` check restores today's skip for them;
(b) dead cross versions still carry `active=False` and skip; (c) ended-**locked**
fibs still pass via `locked` (feed IC detection). **`ended` must not leak into
the gate as an active-pass** — `locked` stays the pass-through.

---

## 12. Implementation sequencing — session-by-session plan

This is a **staged, multi-session build.** Project norms apply throughout:
**one step per session**, self-contained kickoff, `/compare` **before**
`/commit-save`, show the `=== Replay Timing ===` block after every replay,
preserve prior config windows as commented blocks. **Validate between sessions;
do NOT batch.** And the standing constraint: **H1-main structure events +
`end_time` must stay byte-identical** the whole way (this is sub/chart/lifecycle-
meta work only).

Three sessions = the three steps, each independently `/compare`-validated.

> **Why `active`-repurpose lands in Session 2, not Session 1:** today `active` is
> overloaded (condition + terminal) AND consumed by POI/chart. Repurposing it to
> condition-only *changes what consumers read* → not byte-identical. So Session 1
> only **adds** the new axes *alongside* the untouched overloaded `active`;
> Session 2 repurposes `active` and switches consumers together (they're coupled).

### Session 1 — Step 1: additive lifecycle fields (BYTE-IDENTICAL) — DONE (2026-05-27)

> **IMPLEMENTED 2026-05-27, `/compare` byte-identical** (13/13 CSVs + 3/3 chart
> PNGs identical; HTML identical after Plotly div-UUID normalization; 245/245
> tests pass). What landed:
> - `FibState` gained `end_idx`, `end_reason`, `status`, `activation_history`
>   (defaults; unconsumed). `active`/`locked`/`meta`/`fib`/anchors untouched.
> - `FibTracker` gained `_activation_history[(sid,cycle)]` + `_terminal[(sid,
>   cycle)]` dicts (fed by both subsystems) + helpers `_log_flip`,
>   `_set_terminal` (set-if-absent), `_cycle_currently_active`, and
>   `_finalize_lifecycle_fields` (projects axes onto every version record;
>   cycle-level `status`). Orchestrator calls finalize before
>   `get_fibs_for_charting` in `_run_downstream_pipeline` (covers H1 + subs).
> - Flip/terminal sites instrumented additively: `_activate_fib` (+`flip_reason`),
>   `_m15_create_cross` (activated/reanchor + Option-A early-end via obsolete),
>   `_update_fib_cts`, `_update_cycle1_fibs` (cross representative only),
>   `_deactivate_cross` (own_imb_filled only), `_activate_or_update_single_m15`
>   (reanchor-vs-activated), `_obsolete_prev_cycle_all_fibs` (+end_idx),
>   `_deactivate_cycle0_fib` (+revert_idx → scenario1_revert/disappeared).
> - `entity_df_mutation.py`: the mirror now shifts `end_idx` + translates
>   `activation_history[*]["idx"]` by `slice_begin`; the open-sub-fib cap
>   additively sets `end_idx`/`end_reason`/`status` (the §7 passed-through
>   `lifecycle_end` terminal), mirroring the sibling KL cap.
>
> **Two deferrals carried into Session 2 (both confirmed acceptable, unconsumed):**
> 1. **H1-main reversal terminal NOT wired** — reversal_confirmed_idx isn't
>    threaded into FibTracker, so a reversal-ended H1 fib keeps `end_idx=None`
>    and derives `status` from its condition (often "active"). The §7
>    passed-through reversal terminal is Session 2 work.
> 2. **Cosmetic:** a dead cross version (`cross_shortened`) of an end-capped
>    cycle keeps the pre-cap cycle `status` (the cap only re-touches the open
>    record). Session 2's per-record gate + cycle-level status recompute fixes it.

**Goal:** persist the separated axes as NEW data, changing **nothing** any
consumer sees.

**Do:**
- Add fields `end_idx`, `end_reason` (terminal), `activation_history` (condition
  flips), and derived `status`. Populate per §3, §7, §8, §10 (`disappeared` for
  `scenario1_revert`).
- Maintain `activation_history` in a tracker-level `_activation_history[(sid,cycle)]`
  dict fed by BOTH subsystems (§8.4); log handoffs as `reanchor` (§8.5).
- **Do NOT repurpose `active`; do NOT touch any consumer; do NOT change
  versioning.** The overloaded `active`/`locked` stay exactly as today and remain
  what POI/chart read; the new fields are populated *in parallel, unconsumed*.
  (`status` derives from the new `end_idx` + `activation_history`, so it is correct
  even while `active` stays overloaded.)

**Read first:** this spec (§3, §7, §8, §10); `fib_tracker.py`; `poi_lifecycle.py`
(convention reference); `poi_zones.py:554-560` (POI's status derivation — the
pattern to mirror).

**`/compare`:** **byte-identical** (all charts + CSVs). Any diff = a leaked
behavior change (you probably touched `active` or versioning) — back it out.

**Done when:** new fields present + correct in debug exports, `/compare` clean,
`/commit-save`.

### Session 2 — Step 2: repurpose `active` + switch consumers + per-record gate + in-place reactivation — DONE (2026-05-27)

> **IMPLEMENTED 2026-05-27.** `/compare` vs the Session-1 baseline: **all 13
> CSVs + 3 chart PNGs byte-identical**; the ONLY diff is the H1 chart fib hover
> label text (now status-based, e.g. "ended (locked)" — §11). 245/245 tests pass.
> What landed:
> - `active` repurposed to condition-only: the cycle TERMINALS stopped setting
>   `active=False` and now set `end_idx`/`end_reason` — `_obsolete_prev_cycle_all_fibs`
>   + `_activate_fib`'s new_cycle obsolete, `_deactivate_cycle0_fib`
>   (scenario1_revert), and the `entity_df_mutation.py` open-sub-fib cap.
> - **Two reasoned deviations from the original plan** (both validated
>   byte-identical, spec §9.1/§11 reconciliation notes + LANDMINES "FibState
>   Lifecycle Gate Is Per-Record"): (1) `_deactivate_cross`
>   (`cross_failed`/`cross_shortened`) KEEPS per-record `active=False` — they are
>   version-internal supersedes, not cycle terminals; (2) the POI gate is the
>   per-record `(active AND end_idx is None) OR locked`, NOT the cycle-level
>   `status=="active" OR locked` (which would process dead cross versions on subs).
> - Consumers switched: chart per-record gate + status-based hover/styling;
>   `get_fibs_for_charting` → `status != "disappeared"`.
> - In-place cross reactivation (`_m15_reactivate_cross_in_place`) — latent no-op
>   in the validated window (no cross shrink/revival fired).
> - Reversal terminal wired (`FibTracker.set_reversal_terminals`, called before
>   `_finalize_lifecycle_fields` in `_run_downstream_pipeline`) — Session-1
>   deferral 1 closed. Cap recomputes cycle `status` across version records —
>   Session-1 deferral 2 closed.
>
> **The §9.2 visible vanish is LATENT in the default config:** H1-main fibs all
> lock (no unlocked-inactive/ended H1 fib), M15 sub charts have fib lines OFF, and
> fibs are in no CSV — so the sub dead-version/ended-unlocked vanish exists in the
> data but isn't rendered. Enable M15 fib lines to see it.

**Goal:** flip to the new model; accept the deliberate chart change.

**Do:**
- Repurpose `active` to **condition-only** — terminal reasons stop setting
  `active=False` and set `end_idx`/`end_reason` instead (the
  `entity_df_mutation.py:669-678` cap + the `_obsolete_*`/`_deactivate_*` paths).
- Switch consumers: POI gate → `status=="active" OR locked` (§11, the byte-
  identical-preserving form); chart → the **per-record draw gate**
  `not scenario1_revert AND (active OR locked)` (§9.1); drop
  `get_fibs_for_charting`'s special filter for the uniform `status != "disappeared"`
  filter (§10.3).
- Make the **in-place-reactivation** change (§4): same-anchor revival reactivates
  the existing version; a new version only on a start-anchor move. (This is the
  change that is NOT byte-identical under the old chart but **masked** under the
  new gate — which is exactly why it belongs here, with the gate.)

**Read first:** this spec (§4, §9, §10, §11); the chart fib block
`export_plotly.py:2504-2654` + `export_m15_chart.py`; `poi_zones.py:434` (the gate
to preserve).

**`/compare`:** **intentional visual diff** — inactive/ended-unlocked fibs and
dead cross versions **vanish** (§9.2), concentrated on **M15/sub** charts; H1-main
minimally changed; **H1-main structure events + `end_time` still byte-identical**.
**Eyeball sign-off** that the vanished fibs are exactly the three §9.2 categories
(not a real regression).

**Done when:** the §9.2 vanish-categories confirmed as the only fib chart change,
H1 data byte-identical, tests green, user signs off, `/commit-save`.

### Session 3 — Step 3 (OPTIONAL): uniform 3-tier opacity refactor

**Goal:** re-key chart opacity onto pure lifecycle, chart-wide. **Independent of
fib lifecycle** — do only if/when wanted (§13).

**Do:** build `cycle_ended_by_(sid,cycle)` + `sid_ended_by_sid` lookups + a shared
`opacity_for(sid, cycle)` helper (brightest = cycle not ended; mid = sid not
ended; faded = sid ended); thread into every element renderer (fib, KL, POI, wave
candles, WVMI, dots, connectors).

**`/compare`:** opacity-only diffs across many elements; eyeball sign-off.

**Done when:** opacity diffs validated as intended, `/commit-save`.

### Tests (every session)

`test_cross_cycle_fib.py` and the `test_sub_chain.py` guard must stay green or be
updated deliberately.

### Self-contained kickoff (drop-in for any session)

> "Implementing FibState lifecycle **Session N / Step N** per
> `engine_v2/zones/FIB_LIFECYCLE_SPEC.md §12`. Read the spec (esp. the §-refs
> listed for this step) + `memory/project_fib_lifecycle_design.md` first. Do only
> this step, follow the `/compare` expectation stated for it, then `/commit-save`.
> Prior steps' status: <Session 1 done? Session 2 done?>."

---

## 13. Out of scope / deferred / cleanups

**Explicitly parked (do not re-litigate):**
- **Uniform 3-tier opacity refactor** — re-key chart opacity onto pure lifecycle
  (brightest = cycle not ended; mid = sid not ended; faded = sid ended), shared
  `opacity_for(sid, cycle)` helper threaded into every element renderer (fib,
  zones, wave candles, WVMI, dots, connectors). Its own session; only opacity
  `/compare` diffs expected. Nice synergy: if it lands first, the fib renderer
  reuses the helper for free. **NOT** bundled with the fib data changes (would
  make `/compare` illegible).
- **Collapsing `_cross_cycle_data` into `_fibs` versions** — NOT needed; the
  tracker-level history unifies above storage (Section 5). Leave it.
- **Live per-candle fib evaluation** — separately deferred (the `is_filled`
  incremental state-machine item). The coarse CTS-event granularity stands.
- **WVMI lifecycle** — still deferred / design-gated
  (`memory/project_wvmi_lifecycle_deferred.md`). **This FibState design is the
  precedent template** if WVMI is ever revisited (cycle identity, orthogonal
  axes, derived status, sparse history, gate).

**Cleanups:**
- ~~Generalize `POI_ZONES_SPEC.md`'s `disappeared` definition from "IC no longer
  qualifies" to the lifecycle-wide "terminal + suppressed (invalidation)."~~
  **DONE 2026-05-27** — POI_ZONES_SPEC "Zone States" updated to the convention
  (`{active, inactive, ended}` + `disappeared` reserved for the fib
  `scenario1_revert` case the chart already filters).

---

## 14. Decision log (quick reference)

1. **Lifecycle identity = the cycle `(sid, cycle)`**; versions are an internal
   sub-axis; one active version per cycle.
2. **Orthogonal axes:** `active` (condition-only) + `end_idx`/`end_reason`
   (terminal) + `locked` (computation, separate, NOT in `status`); `status`
   derived (`{active, inactive, ended, disappeared}`).
3. **Version = start-anchor identity.** CTS extension = same version; start-anchor
   shrink = new version; **same-anchor revival = in-place reactivation** (a
   behavior change from today's new-version-on-revival).
4. **Start = first version's birth.** Subordinate-only exception: a pre-established
   cross can start before its cycle's CTS-established. Cross→single handoff keeps
   the early start.
5. **End = cycle end; Option A:** an `n+1` pre-established cross ends the previous
   fib early (at the cross-creation idx) — diverges from the zone end at this one
   boundary, preserves "one active fib." Other terminals (reversal, parent) =
   passed-through cycle-end.
6. **`activation_history`:** sparse, coarse CTS-event granularity, kept (not
   dropped), tracker-level `(sid,cycle)` dict fed by BOTH subsystems; handoffs
   logged as `reanchor` actives; terminal not in the history.
7. **"Project, don't unify":** keep A's and B's storage; unify the lifecycle at
   the cycle-identity projection.
8. **Chart gate (per record):** `not scenario1_revert AND (active OR locked)`. No
   stretch fills; 2-tier; inactive/ended-unlocked fibs vanish (intentional diff,
   M15/sub-heavy).
9. **`scenario1_revert` → `status="disappeared"`**, reusing the existing
   (vestigial) chart filter; distinct from reversible inactive-vanish.
10. **Sequencing:** additive byte-identical build → consumer-switch + gate
    (intentional diff, sign-off) + in-place-reactivation → optional opacity
    refactor (separate session).
