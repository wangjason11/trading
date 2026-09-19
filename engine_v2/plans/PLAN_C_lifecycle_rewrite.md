# Plan C — Sub-Structure Pool Lifecycle Rewrite (TriggerRecord + unique sub)

**Status:** READY TO IMPLEMENT (written 2026-09-19; every design decision closed).
**Canonical decisions + rationale:** `memory/project_sub_structure_pool_architecture.md` (read first).
**Ground truth + acceptance table:** `memory/reference_pool_redesign_groundtruth.md`.
**Vocabulary:** `GLOSSARY.md` "Sub-Structure Pool Terms".

This document is the implementation contract. It is written so a session with NO memory of the
design discussion can execute it. Where it says "decided", do not re-open; where it says "implementer's
choice", pick and document.

---

## 0. Prerequisites, base, sequencing

| Step | Must be true before starting |
|---|---|
| P0 | `9fd3143` (Stage 3.2b) has been **`git revert`ed** on `week8-volmom-multitf` (kept in history). Tree = Stage 3.2a semantics. `render_unique_sub`, `TriggerRecord.meta["mt"]`, the `conf_render`/`ctr_render` loop are gone again. |
| P1 | **Plan A** (`PLAN_A_ms_bounds_leak.md` — bounded MS runs read nothing past their bound) landed with its own `/commit-save`. Its only production footprint is the FC probe's Phase-2 run (sub builds slice to their bound already); its residual (FC(1,0) finalize 2843, possibly a moved `starting_idx` 2803) does NOT reach Plan C's predicted table — FC(1,0) is in a degenerate cycle and is never probed under Plan C. The §9.2 test (written in this plan) runs on the Plan-A+B base and is the proof; if it disagrees with the memory table, update fixture + table together. |
| P2 | **Plan B** (double-CTS early stop) landed, byte-identical, with its own `/commit-save`. |
| P3 | 418+ tests green at the base. |

Plan C is **one behavioural change** ("the lifecycle model") ⇒ one replay, one `/compare`, one chart-review
pause. Internal checkpoints are **unit tests** (§9), especially the predicted-table test, which validates the
sweep in <1 s without a replay. Do not split Plan C across commit-saves — its deltas are not separable.

Baseline for `/compare` = the Plan B save. Per-row parity will NOT hold for M15 (storage shape changes);
acceptance is §10.

---

## 1. Scope

**Changes:** the multi-TF sub layer only — `multitf/*`, `_run_multi_tf_dual`, `export_m15_chart.py`,
`sid_records.py`, `run_replay.py` exports, the tests in §9, the docs in §11.

**Changes one shared leaf:** `zones/structure_lifecycle.py` — cycle lifecycle-start becomes the
CTS-established *moment* (`meta["confirmed_at"]`) instead of the extreme (`.idx`) (§3). H1-byte-identical
on this window; three sub cycles shift +1 candle.

**Does NOT change:** `H1.main` (`compute_structure`), `unified_probe` (except reading its result),
`compute_bounded_structure`, `zones/*` derivations (`kl_zones_v1`, `poi_zones`, `fib_tracker`,
`wave_candles`) other than via the leaf above, `zone_proximity`, the trigger detectors'
detection logic (only their `lifecycle_end_idx` output is retired), `export_plotly.py` (H1 chart — its
optional M15 overlay reads only `kl_zones` from the counter entity and never `sub_sid`; nothing to change).

**Principle every line must respect:** lifecycle fields (`start_idx`, `end_idx`) are *real-time* — what
was tradeable when. Pattern/element-definition fields (`starting_idx`, `trigger_idx`, `probe_finalize_idx`,
every anchor) are *historical* — where things logically sit. Historical fields are legitimate INPUTS when
*constructing* a lifecycle value (`start_idx = max(finalize, trigger, floor)` is exactly that); they must
never be used *as* a lifecycle value in an is-active-at-t test, and a lifecycle floor must never be
applied to a historical field.

---

## 2. Data model

All idxs are **entity-absolute M15 ints** unless stated. `parent_path` stays the constant `"H1.main"`.
Suffix convention: `_idx` everywhere; no `_dt`.

### 2.1 `TriggerRecord` (rewrite in `multitf/sub_structure_pool.py`)

```python
@dataclass
class TriggerRecord:
    # identity (lens, parent_sid, parent_cycle_id, trigger_sub_sid)
    lens: str                       # "confluence" | "counter"
    parent_sid: int
    parent_cycle_id: int
    trigger_sub_sid: int            # per (lens, parent_sid, parent_cycle_id); +1 per NEW unique sub
    # foreign key
    sub_id: int                     # NEVER None
    # provenance (historical)
    trigger_type: str               # first_confluence | subsequent_confluence | first_counter
                                    # | subsequent_counter | reversal
    trigger_idx: int                # LOH(parent trigger candle); native M15 for reversal
    probe_finalize_idx: int         # HISTORICAL: when THIS RECORD'S PROBE finalized, where "this record's
                                    # probe" is the run identified by (direction, initial input) — §5.3:
                                    #   - probe RAN (new input): its own ProbeResult.finalize_idx, as-is (may
                                    #     precede trigger_idx, e.g. FC(0,1) 2608 < 2611 — logged raw)
                                    #   - probe SKIPPED (same direction + initial input → cache hit): the
                                    #     cached finalize, inherited raw (an earlier probe of that key already
                                    #     finished; the re-trigger does not wait — decided 2026-09-19)
                                    # A different-input probe that CONVERGES on a known structure keeps its OWN
                                    # finalize: it could not know it maps to that structure until it finished.
    probe_finalize_condition: str   # ProbeResult.finalize_condition of that probe (diagnostic)
    validated_parent_idx: Optional[int]  # H1 candle that seeded the probe (restores 3.2a's
                                    # validated_h1_start; None for reversal-born)
    starting_idx: int               # == sub.starting_idx (denormalised for the CSV)
    direction: int                  # == sub.direction
    sub_tf: str                     # "M15"
    relative_dir: str               # "confluence" iff direction == parent_sd(parent cycle) else "counter"
    parent_floor_idx: int           # the floor that was applied (diagnostic)
    source_trigger: MultiTFTrigger  # INTERNAL (not exported): the originating trigger object; feeds
                                    # LowerTFResult.trigger for the mirror / WVMI / sid_records readers
    # lifecycle (real-time)
    start_idx: int                  # max(probe_finalize_idx, trigger_idx, parent_floor_idx) — REAL-TIME; the
                                    # record EXISTS from here. All three terms are load-bearing, each for a
                                    # distinct case: probe_finalize_idx when the probe finished after the
                                    # trigger (FC 463→1020); trigger_idx when the structure was already known
                                    # before this trigger fired (re-trigger: inherited finalize 3487, trigger
                                    # 3611); parent_floor_idx when the parent structure was not alive yet
                                    # (2844→3611). Historical fields are never adjusted; only start_idx is.
    trigger_end_idx: Optional[int]  # first end condition, or the sub's frozen end on a post-end re-trigger
    end_idx: Optional[int]          # max(trigger_end_idx, start_idx); None while open
    end_reason: Optional[str]       # "reversal" | "same_dir_replacement" | "parent_end" | None
    ended_by_sub_id: Optional[int]  # for same_dir_replacement: the replacing sub (explainability)
    # bookkeeping
    seq: int                        # creation order (deterministic tie-break, never exposed as identity)
    extra_trigger_idxs: List[int]   # further triggers in the same (lens, parent) that resolved to the same sub
```

`is_zero_length = trigger_end_idx is not None and trigger_end_idx <= start_idx` — defined on
`trigger_end_idx`, NOT `end_idx`, so it is already correct in phases 1–2 of the idx at which the record
is ended (phase 3 writes `end_idx` later the same idx). A zero-length record **participates in nothing**
(sub start, `max_start`, end candidates, lens membership, rendering) — it is logged only. `trigger_sub_sid`
is CREATION-ordered, not `start_idx`-ordered (an FC record is created at its `trigger_idx` and may start
hundreds of candles later).

### 2.2 `PooledStructure` (= unique sub; keep the class name, add fields)

```python
@dataclass
class PooledStructure:
    key: StructureKey               # (parent_path, sub_tf, direction, starting_idx) — unchanged
    sub_id: int                     # global monotonic — unchanged
    geometry: Any                   # (bounded, slice_begin) — run cap = DATA EDGE (§5.4). bounded.events /
                                    # bounded.df are SLICE-LOCAL; add slice_begin for entity-absolute
    natural_reversal_idx: Optional[int]
    bos0_inner: Optional[float]     # NEW: from the first probe; WARNING-logged (not asserted) on mismatch
    records: List[TriggerRecord]    # renamed from trigger_records
    # lifecycle (real-time; set once)
    start_idx: Optional[int]        # FIRST non-zero-length record's start_idx
    end_idx: Optional[int]          # §4.4
    end_reason: Optional[str]
    relative_dir_segments: List[Tuple[int, str]]   # [(from_idx, "confluence"|"counter"), ...] step function
    def live_records(self) -> List[TriggerRecord]   # non-zero-length
    def lenses(self) -> Set[str]                    # union over live_records
```

Delete `lifecycle_start` / `lifecycle_end` / `lifecycle_end_reason` / `memberships()` /
`earliest_membership()` on `PooledStructure`, and `SubStructurePool.for_lens()` (replace with a filter
over `lenses()`).

### 2.3 `UnresolvedTrigger` (new, same module)

```python
@dataclass(frozen=True)
class UnresolvedTrigger:
    lens: str; parent_sid: int; parent_cycle_id: int
    trigger_type: str; trigger_idx: int; direction: int
    probe_input_idx: Optional[int]  # whatever was known (H1 or M15 per type)
    reason: str                     # "pending" | "degenerate_parent_cycle" | "probe_failed" | "geometry_failed"
    detail: str                     # free text (e.g. "floor 3611 >= end 2995")
```

### 2.4 `SubStructurePool` API

Keep `get`, `get_or_create`, `all`, and the probe cache. Add:
`get_by_id(sub_id) -> PooledStructure`,
`records_for(lens, parent_sid, parent_cycle_id) -> List[TriggerRecord]`,
`next_trigger_sub_sid(lens, parent_sid, parent_cycle_id) -> int` (**starts at 0** per scope, matching
today's per-cycle `sub_sid=0` convention),
`unresolved: List[UnresolvedTrigger]`,
`active_record(lens, parent_sid, parent_cycle_id, direction, at_idx) -> Optional[TriggerRecord]` —
**interval-based, on `trigger_end_idx`**: a record is active at `at_idx` iff `not is_zero_length and
start_idx <= at_idx and (trigger_end_idx is None or trigger_end_idx > at_idx)`. (Not an "active" flag,
not `end_idx` — in the 3819 handover the incumbent must be found during phase 1 and must NOT be counted
after it, while its `end_idx` is only written in phase 3.) **Assert** at most one.
`records_for(...)` returns every record including zero-length ones; callers filter.

`finalize_lifecycles`, `select_lifecycle_end`: **delete** (replaced by §4). **Keep `_END_REASON_PRIORITY`**
(`reversal` > `parent_end` > `same_dir_replacement`) — it is used by phase 3 (§4.4).

### 2.5 `SidRecord` (`multitf/types.py`) — ONE class, two roles; sub rows are one per unique sub

`SidRecord` stays a single frozen dataclass shared by main and subs. Add `sub_id: Optional[int] = None`
and `start_idx: Optional[int] = None`, `lenses: Tuple[str, ...] = ()`, `relative_dir_segments`.
- **main** (`build_sid_records_for_main`): unchanged — `sub_sid = structure_id`, `sub_id = None`,
  parent fields None. Chart identity for main = `sub_sid` (as today).
- **sub** (`build_sid_records_for_subordinate`): `sub_id` set, **`sub_sid = None`**, `parent_sid` /
  `parent_cycle_id` **None** (parent attribution lives on the record table), `creation_event_idx =
  starting_idx`, `start_idx`, `end_event_idx = end_idx`, `end_reason`, `lenses`,
  `meta = {"natural_reversal_idx", "n_records", "first_record": {...}}`. Chart identity for subs = `sub_id`.
`_sid_record_identity` / `_sub_identity` in the chart return `sub_id` when set else `sub_sid`.

---

## 3. Static tables (computed once from H1 events, before any sub work)

In `_run_multi_tf_dual`, replacing today's `parent_cycle_floor_h1` (computed in `run_pipeline`),
`parent_struct_end_m15`, and `parent_end_lookup`. One helper, e.g.
`multitf/parent_tables.py::build_parent_tables(sorted_events, h1_df, m15_df) -> ParentTables`:

```
rev_by_sid[S]      = compute_reversal_idx_by_sid(events)            # STATE_CHANGED→reversal (NOT REVERSAL_CANDIDATE)
struct_start[S]    = compute_struct_start_by_sid(events, rev_by_sid, None)
cts_moment[(S,C)]  = CTS_ESTABLISHED.meta["confirmed_at"]   # the MOMENT the cycle was established
                                                            #   (== BOS_CONFIRMED.confirmed_at, definitional).
                                                            #   NOT .idx, which is the CTS EXTREME (historical).
                                                            #   LAST-seen per (S,C).
parent_sd[S]       = CTS_ESTABLISHED.meta["struct_direction"] of any event of S (one direction per sid)
floor_h1[(S,C)]    = max(struct_start[S], cts_moment[(S,C)])                 # == the CLAMPED cycle start
end_h1[(S,C)]      = floor_h1[(S,C+1)] if (S,C+1) in cts_moment              # next cycle's CLAMPED start,
                     else rev_by_sid.get(S) else None                        #   same rule as compute_cycle_lifecycle
floor_m15[(S,C)]   = LOH(floor_h1)          # _map_parent_idx_to_m15_hour_end
end_m15[(S,C)]     = LOH(end_h1) or None
degenerate[(S,C)]  = end_m15 is not None and floor_m15 >= end_m15
```
**Cycle lifecycle-start = the moment, not the extreme (decided 2026-09-19).** A cycle's real-time
lifecycle begins when it is *established* (`meta["confirmed_at"]` = the apply candle); `CTS_ESTABLISHED.idx`
is where its extreme sits — a historical anchor, exactly like `BOS_CONFIRMED.idx`. Today
`compute_cycle_lifecycle` / `compute_struct_start_by_sid` / `parent_cycle_floor_h1` all use `.idx`
(PART4 §5 called the extreme "the canonical cycle-start idx"). **Plan C changes the canonical rule in
`zones/structure_lifecycle.py`** — cycle start = `max(CTS_ESTABLISHED.meta["confirmed_at"], struct_start,
floor)` — so main, sub cycles, and this parent table all agree. Consequences: H1 byte-identical on this
window (extreme == moment on all five cycles); three sub cycles shift their lifecycle start by +1 candle
(1223→1224, 2828→2829 ×2) — the KL/POI `confirmed_idx` clamp follows; and `trigger_idx` is provably
≤ the floor for every trigger type, so it is NOT a floor term (assert it instead). Using the clamped next
start for the end keeps this helper identical to `compute_cycle_lifecycle`; on this window (1,0)'s end
becomes 3611 (still degenerate).
`ARCHITECTURE.md`'s "`ev.idx` convention" lists `BOS_CONFIRMED` as the only extreme-not-apply exception;
`CTS_ESTABLISHED` is a second one — fix in §11.

**Asserts (raise, do not degrade):** every `(S,C)` with a trigger has a `CTS_ESTABLISHED`;
**`BOS_CONFIRMED(S,C).meta["confirmed_at"] == CTS_ESTABLISHED(S,C).meta["confirmed_at"]`** for every BOS
with a matching cycle — this is the definitional identity (both are the same `apply_idx`,
`market_structure.py` ~1414-1451). **NOT `CTS_ESTABLISHED.idx`**: that is the CTS *extreme* within the
pattern span and can precede the apply candle (3 such pairs in the saved M15 streams — 1223 vs 1224,
2828 vs 2829; 0 of 5 on H1 only by luck). Every LOH map returns non-None. Log one `WARNING
[parent_tables] degenerate parent cycle (S,C): floor=… end=…` per degenerate cycle.

`trigger_idx` stays a floor term in `start_idx` (§2.1). For a record whose probe actually ran it is
redundant once the floor is on the moment (FC's `trigger_idx = LOH(BOS.confirmed_at) = LOH(cts_moment)
≤ floor`; every other type has `finalize == trigger`) — but a *cache-hit* record inherits an earlier
identical probe's finalize, which normally precedes its own `trigger_idx`, and there the term is what
stops the instance starting before its trigger fired.

Retire: `lifecycle_end_idx` on the four trigger dataclasses (keep the field for one commit if convenient,
but nothing may read it), `_find_m15_lifecycle_end`, the `parent_end_lookup` block, the
`parent_struct_end_m15` block, the `parent_cycle_floor_h1` block in `run_pipeline`.

On this window: floors 463/2611/3611/3611/3611; ends 2611/3611/3611/3611/None; degenerate = {(1,0),(1,1)}.
`first_counter`'s `trigger_event_idx` comes from `WVMIRecord.meta["triggered_by_event_idx"]` — assert
non-None (it always is for a record that reached `detect_uc1_triggers`).

---

## 4. The sweep (replaces `_ChainCursor` / `build_two_entity_parent_cycle` / `build_parent_cycle_chain`)

New module `multitf/lifecycle_sweep.py`. Pure orchestration; probe/geometry/sibling access injected so
the predicted-table test (§9.2) can stub them.

### 4.1 Inputs
- `h1_triggers`: `MultiTFTrigger`s from the four detectors — `first_confluence_pipeline`,
  `subsequent_confluence_pipeline`, `subsequent_counter_pipeline` via their `to_multi_tf_trigger`, and
  `uc1_trigger.detect_uc1_triggers` which builds `MultiTFTrigger` directly (there is no
  `first_counter_pipeline`) — each tagged `lens` (`resolve_lens(use_case)` — the pool module's function;
  the `"confluence" in sub_path_id` substring test is deleted), `trigger_idx = LOH(trigger_event_idx)`,
  parent `(S,C)`, `direction = lower_sd`. `first_confluence` with `status != "finalized"` →
  `UnresolvedTrigger(reason="pending")`.
- `ParentTables` (§3). `pool`. The single shared M15 feature frame `m15` (`prepare_lower_tf_data` once).
- `direction`-aware probe resolver = today's `_resolve_trigger_m15_start`, with its sibling read
  rewired to §5.1.

### 4.2 Moments and phases
A priority queue of `(idx, phase, order_key)`; phases at the same idx run **in this order, each to
completion before the next**:

| phase | moment | action |
|---|---|---|
| 0 | `TRIGGER_FIRE` (H1 trigger at its `trigger_idx`) / `REVERSAL_SPAWN` (a sub's natural reversal) | resolve → create record(s) → insert their `RECORD_START` / known `RECORD_END`s |
| 1 | `RECORD_START` | freeze if the sub already ended (§4.4); else mark active and queue a same-dir replacement candidate for the incumbent (§4.3′) |
| 2 | `SUB_START` | set `sub.start_idx` if unset and a live record started here |
| 3 | `RECORD_END` | set `trigger_end_idx`/`end_idx`/`end_reason` if unset; mark inactive |
| 4 | `SUB_END` | re-evaluate `sub.end_idx` for every sub touched at this idx (§4.4) |

`order_key` within a phase: for `TRIGGER_FIRE` = `(lens_rank, type_rank, trigger_event_idx)` with
`lens_rank` confluence=0, counter=1 (today's cadence order — matters because the second trigger's sibling
read at `hi = t` inclusive can see the first's sub) and `type_rank` first_* < subsequent_*; for every
other moment = `(record.seq)`. `REVERSAL_SPAWN` sorts before `TRIGGER_FIRE` at the same idx (its successor
must exist before an H1 trigger at the same candle runs its sibling read). Two records of the same
`(lens, S, C, direction)` starting at the same idx for DIFFERENT subs cannot occur outside acausal cases;
if it does, the later `seq` replaces the earlier (§4.3′ as written) and a `WARNING [sweep] same-idx start
collision` is logged. Start-before-end at the same idx is **load-bearing** (§4.4).

### 4.3 Phase 0 — resolving a trigger into a record

```
T = trigger (lens, S, C, type, trigger_idx=t, direction=d, probe inputs)
1. if degenerate[(S,C)]:  unresolved.append(reason="degenerate_parent_cycle"); return
2. probe (the probe cache lives INSIDE the resolver, wrapping the `unified_probe` call — §5.3):
     res = resolve_start(T, pool, tables, hi=t)   # = today's _resolve_trigger_m15_start for the four H1
                                                  #   types (sibling read per §5.1); for type "reversal" =
                                                  #   the reversal branch lifted out of build_one_sid
                                                  #   (edm ~1341-1420), run on the reversing sub's
                                                  #   slice-local geometry with EVERY returned idx shifted
                                                  #   by slice_begin; its failure branches (no CTS ref
                                                  #   zone / input >= reversal / pending) → probe_failed
     if res is None: unresolved.append(reason="probe_failed"); return
     starting_idx, validated_parent_idx, bos0_inner, finalize_idx, finalize_condition = res   # entity-absolute
     assert finalize_idx == t for every non-FC, non-reversal type (§5.2)
3. built = build_or_get_geometry(pool, key=StructureKey(parent_path, "M15", d, starting_idx), bos0_inner)  (§5.4)
     — the SINGLE owner of get_or_create + MS run + natural_reversal_idx + bos0_inner. On a miss it builds
     FIRST and only then creates the pool entry (a failed build → returns None, no sub_id consumed):
     if built is None: unresolved.append(reason="geometry_failed"); return
     sub, created = built
     if created and sub.natural_reversal_idx is not None:
         R = sub.natural_reversal_idx
         if R > t: queue (R, phase 0, REVERSAL_SPAWN, sub)
         # R <= t: the structure reversed before this trigger fired — nothing to spawn (no record can be
         # live at R since every start_idx >= t > R); step 6 makes this record zero-length. Never queue
         # a moment in the past (the heap must stay idx-monotone; assert it).
     if not created and not math.isclose(sub.bos0_inner, bos0_inner, abs_tol=1e-9):
         log WARNING [sweep] bos0_inner mismatch … (keep the first; do NOT raise — the same (dir, start)
         can legitimately be reached via an ad-hoc BOS_0 and via a sibling CTS-zone inner)
4. existing = [r for r in pool.records_for(lens,S,C) if r.sub_id == sub.sub_id]
     if existing: existing[0].extra_trigger_idxs.append(t); return        # same (lens,parent) → same record
5. rec = TriggerRecord(lens,S,C, trigger_sub_sid=pool.next_trigger_sub_sid(lens,S,C), sub_id=sub.sub_id,
         trigger_type, trigger_idx=t,
         probe_finalize_idx=finalize_idx,        # from step 2 — own run, or the cached one on a cache hit
         ...,
         relative_dir = "confluence" if d == parent_sd[S] else "counter",
         parent_floor_idx = floor_m15[(S,C)],
         start_idx = max(finalize_idx, t, floor_m15[(S,C)]),          # the three-term real-time floor
         source_trigger = T)
6. known-in-advance ends:
     if sub.end_idx is not None and rec.start_idx > sub.end_idx:        # post-end re-trigger (frozen sub)
         rec.trigger_end_idx = sub.end_idx; rec.end_reason = sub.end_reason
         rec.ended_by_sub_id = the ending record's ended_by_sub_id (copied)
     else:
         cands = []
         if sub.natural_reversal_idx is not None: cands.append((sub.natural_reversal_idx, "reversal"))
         if end_m15[(S,C)] is not None:            cands.append((end_m15[(S,C)], "parent_end"))
         e = min(cands) if cands else None
         if e is not None and e[0] <= rec.start_idx:   rec.trigger_end_idx, rec.end_reason = e   # zero-length now
         elif e is not None:                            queue (e[0], phase 3, RECORD_END, rec, e[1])
   rec.end_idx = max(rec.trigger_end_idx, rec.start_idx) if rec.trigger_end_idx is not None else None
7. sub.records.append(rec)
   if not rec.is_zero_length: queue (rec.start_idx, phase 1, RECORD_START, rec)
```

Step 6's "known in advance" lets phase 2 know a record is zero-length before its end moment would have
been processed. Note that creation order is NOT `start_idx` order in general (an FC record created at its
`trigger_idx` 463 starts at 1020; anything created in between with an earlier start breaks the
ordering) — nothing in the sweep depends on it, and `trigger_sub_sid` is explicitly creation-ordered.

**Triggers are independent.** A `subsequent_*` trigger is processed even when its parent cycle's `first_*`
trigger was unresolved or produced a zero-length record — §6.1's "no sid 0 → no subsequents" rule is
retired (decided 2026-09-19; it was an artifact of the sequential chain).

**Reversal-born successor (`REVERSAL_SPAWN` at R for sub X):** for each record `r` of X that is active at
R per `active_record`'s interval rule (`not zero-length`, `start_idx <= R`, `trigger_end_idx is None or
trigger_end_idx >= R` — a record whose parent ends at R is still live *at* R): spawn `T' = (lens=r.lens,
S=r.S, C=r.C, type="reversal", trigger_idx=R, direction=-X.direction)` with a `MultiTFTrigger`
synthesised from `r.source_trigger` (today's `_synth_reversal_trigger(bootstrap, sd, rev_idx)` takes the
cycle's bootstrap; it now takes the record's own source trigger — same fields), and run §4.3 on it
immediately (still phase 0 at R). Its probe is the reversal handoff over X's geometry (§4.3 step 2), reading
X's own events for its reference (§5.1 note). Two live records on two lenses ⇒ two `T'` ⇒ they dedup into
ONE successor sub with a record per lens. If X has no live record at R ⇒ no successor (decided: "the
reversal is inert too"). Successor `start_idx = max(finalize=R, trigger=R, floor) = R` (floor ≤ R because `r` is live).

### 4.3′ Phase 1 — `RECORD_START` for `rec` at t
```
mark rec active
inc = pool.active_record(rec.lens, rec.S, rec.C, rec.direction, at_idx=t)   # excluding rec
if inc is not None and inc.sub_id != rec.sub_id and inc.trigger_end_idx is None:
    queue (t, phase 3, RECORD_END, inc, reason="same_dir_replacement", ended_by=rec.sub_id)   # candidate only
assert pool.active_record(...) count <= 1 after this
```
Same-lens only: `active_record` is keyed on `lens`. Replacement idx = the new record's **true `start_idx`**
(this `t`), decided — not its `trigger_idx`, not its finalize.

### 4.4 Phases 1–4 (detail)
```
phase 1 RECORD_START (rec at t):
    if sub.end_idx is not None and sub.end_idx < rec.start_idx:      # the sub froze while rec was pending
        rec.trigger_end_idx = sub.end_idx; rec.end_reason = sub.end_reason   # → zero-length, never active
        return                                                         # (strict <: == t is the handover case)
    mark active; incumbent replacement per §4.3′
phase 2 SUB_START: for each sub with a record that started at t and sub.start_idx is None
                   and that record is not zero-length (per the trigger_end_idx definition): sub.start_idx = t
phase 3 RECORD_END: every RECORD_END entry carries (idx, reason[, ended_by]). For a record with one or
    more entries at t: if rec.trigger_end_idx is not None (ended earlier) → the entries are STALE, drop
    them, do not mark the sub "touched". Else pick the entry with the highest _END_REASON_PRIORITY
    (reversal > parent_end > same_dir_replacement), set rec.trigger_end_idx = t, rec.end_reason,
    rec.ended_by_sub_id, rec.end_idx = max(t, rec.start_idx); mark inactive. Earliest idx always wins
    across different idxs because earlier entries are processed first and later ones become stale.
phase 4 SUB_END: for each sub touched at t (a live record STARTED or ENDED here):
    if sub.end_idx is not None: continue
    live = [r for r in sub.records if not r.is_zero_length and r.start_idx <= t]   # records that EXIST at t
    if not live: continue
    max_start = max(r.start_idx for r in live)
    cands = [(r.end_idx, _END_REASON_PRIORITY[r.end_reason], r) for r in live
             if r.end_idx is not None and r.end_idx > max_start]
    if cands: sub.end_idx, sub.end_reason = the min-idx candidate (priority breaks equal-idx ties)
```

> **A record EXISTS from its `start_idx` — nothing earlier (decided; restated by the user 2026-09-19).**
> `trigger_idx` and `probe_finalize_idx` are historical loggers. The sweep may *instantiate* the record
> object at phase 0 of its `trigger_idx` (every historical field is already known), but that object is
> invisible to every lifecycle computation until its `RECORD_START` moment at `start_idx`: it is not in
> `max_start`, not an end candidate, not an incumbent, not a lens member. "All the existing
> triggerRecords" in the sub-end rule therefore means exactly the STARTED ones (`start_idx <= t`) — there
> is no other reading. Two cases, both fall out of the phase order:
> - the sub's last live record ends at `E` **before** the new record's `start_idx` → the sub ends at `E`
>   in real time (phase 4 at `E` sees only started records); when the new record reaches `start_idx` it
>   finds the sub frozen → phase 1 freezes it to zero-length with the sub's end — linked, logged, never
>   active;
> - the end lands **on** `start_idx` → phase 1 starts the new record first, `max_start = E`, the old
>   record's end `E > E` is false → the sub persists through the new record (`end_idx` stays None until a
>   later condition). Start-before-end is the whole mechanism.
> The §9.2 fixture cannot distinguish these from a wrong ordering (no sub on the window has a pending
> record across an end); §9.3's unit tests must.

Assert after the sweep: every non-zero-length record satisfies `start_idx <= sub.end_idx` or
`sub.end_idx is None`.

The strict `>` with start-before-end is what keeps a sub continuous across a same-candle handover (record A
ends at t, record B started at t ⇒ `max_start = t`, `t > t` false ⇒ no end). Verified by hand on subs
`2365` (→ 2829) and `3304` (→ 3819); both are §9.2 fixtures (an independent cold hand-run of the whole
sweep reproduced the full predicted table at every step).

### 4.5 After the sweep
```
for sub in pool.all():
    sub.relative_dir_segments = step function over [start_idx, end_idx]: at each record start/end, the
        active record with the latest start_idx → its relative_dir; carry forward when none is active
    if sub.start_idx is None: continue                # no live record: logged (_subs.csv), NOT rendered
                                                      # (decided 2026-09-19; such a sub has no lens)
    render(sub) per §6
```
Asserts after the sweep: every non-zero-length record has `start_idx < end_idx` or `end_idx is None`;
every sub with `start_idx` has `end_idx is None or end_idx > start_idx`; per `(lens,S,C,direction)` the
live records' windows do not overlap as **half-open** intervals `[start_idx, end_idx)` (a handover shares
its boundary candle — rec7 `[3621,3819]` / rec9 `[3819,…]`); the moment heap never popped a smaller idx
than the previous one.

**Records are not what is drawn (intended).** A same-lens replacement of a sub's *confluence* record ends
the SUB, and the counter chart draws the sub's window — so the counter chart also stops there even if the
sub's counter record would have run longer. That is the "one lifecycle, drawn on every lens" decision, not
a leak. (Not observable on this window.)

---

## 5. Probe, geometry, sibling reads, cache

### 5.1 Sibling-CTS read → pool query (retires the scratch dfs)
`_build_sibling_cts_ref_zone(sibling_entity_df, ...)` becomes
`_build_sibling_cts_ref_zone_from_pool(pool, other_lens, S, C, probe_direction, idx_window)`:
```
recs   = [r for r in pool.records_for(other_lens, S, C)
          if r.direction == -probe_direction              # direction-qualified (as today)
          and not r.is_zero_length
          and r.start_idx <= hi                            # the sibling record must be LIVE somewhere in
          and (r.trigger_end_idx is None or r.trigger_end_idx >= lo)]   #   the window — not merely exist
events = []   # ENTITY-ABSOLUTE copies: pool geometry is SLICE-LOCAL, shift every idx / idx-bearing meta by slice_begin
for r in recs:
    bounded, slice_begin = pool.get_by_id(r.sub_id).geometry
    r_lo, r_hi = max(lo, r.start_idx), min(hi, r.trigger_end_idx if r.trigger_end_idx is not None else hi)
    events += [shift(ev, slice_begin) for ev in bounded.events
               if r_lo <= ev.idx + slice_begin <= r_hi and ev.type in _CTS_EVENT_TYPES]   # ev.idx == knowable-at for CTS types
winner = max(events, key=(idx, type order))
return build_reference_zone_from_cts_event(events=events, kl_zones=[], df=m15 (the shared entity-absolute
                                           frame — NOT the winner's slice-local bounded.df), sid=0,
                                           probe_direction, idx_window=(lo, hi))
```
**Clip each record's events to its own live window** — this is what reproduces today's candidate set.
Today a replaced/ended sibling sid simply has no events past its bounded build; under the pool its
geometry runs to the data edge, so without the clip a REPLACED sub's later `CTS_UPDATED`s would compete
(concrete: `subsequent_counter(1,2)`@4083 reads confluence (1,2); sub 7 was replaced at 3819 but its
geometry continues — its post-3819 CTS events must NOT be candidates, else `starting_idx` 4027, the key
both sub-10 records dedup on, can move). A record that exists but has not started (FC between
`trigger_idx` and `start_idx`) is likewise excluded by `start_idx <= hi`.

**`kl_zones=[]` is behaviour-preserving, not a shortcut.** The primitive's CONFIRMED branch looks up a
zone with `source_kind == "CTS"` (`reference_zone.py:_find_existing_cts_kl_zone`), but subs only ever
receive BOS zones: `_run_downstream_pipeline` derives the full set internally and returns
`source_kinds=["BOS"]` for subs, and that BOS-only list is what both today's readers get
(`downstream["kl_zones"]` for the reversal probe; the mirrored `attrs["kl_zones"]` for the sibling read —
the sub KL CSVs contain zero CTS rows). So the branch has been dead for subs since the narrowing and
every sub reference zone is ad-hoc-derived today. **Do not be misled by `ref=cts_confirmed` in replay
logs:** the ad-hoc branch labels its result by the winning event's TYPE (`reference_zone.py:383-388`), so
that string never distinguishes derived from ad-hoc. Follow-up (out of scope): whether subs *should* get
the derived CTS zone.

`hi` = the reading trigger's `trigger_idx` — this window is what keeps the read causal now that geometry
runs to the data edge. **Expected delta:** within a record's live window the sibling now sees the
natural-end CTS stream (a `CTS_UPDATED` the old per-trigger bound had cut) — small `starting_idx` shifts
are possible; each must be explained by such an event inside `[r_lo, r_hi]`.
The reversal-handoff probe (§4.3 spawn) reads the reversing sub's OWN events the same way (today:
`bounded.events` + `downstream["kl_zones"]` of the current sid, `idx_window=None`) — also `kl_zones=[]`.
Delete `conf_m15`/`ctr_m15`, `_assert_m15_frames_aligned`, the per-trigger `build_one_sid` mirror.

### 5.2 Probe result carries `finalize_condition` and `bos0_inner`
`_resolve_trigger_m15_start` returns `(starting_idx, validated_parent_idx, bos0_inner, finalize_idx)`
today; add `finalize_condition`. Non-FC callers: `finalize_idx == trigger_idx` by construction — assert it.

### 5.3 Probe cache (wire `probe_cached_start` / `record_probe_start`)
**Placement:** inside the resolver, wrapping the `unified_probe` call — AFTER the sibling read / ad-hoc
BOS_0 derivation, because the key's `initial_input_idx` (FC: price-mapped BOS extreme; sibling types:
`ref_zone.source_event_idx`; reversal: the handoff input, shifted to entity-absolute) and the hit rule's
`end_idx` are only known there. Key `(parent_path, "M15", direction, initial_input_idx)`. Store the full
result (`starting_idx, finalize_idx, finalize_condition, bos0_inner, end_idx`).

**"Same probe" = same direction + same initial input.** Key: `(parent_path, "M15", direction,
initial_input_idx)` — exactly as specced in §17.7. **Two explicit assumptions ride on this key, both
accepted as DESIGN (user, 2026-09-19): the first probe to finalize for a key is the truth for every
later probe of that key, (a) regardless of its search bound `end_idx`, and (b) regardless of its
reference zone.** (b) is a real assumption, not a triviality: the two-condition reset tests every
candidate against the reference *inner*, and different trigger types derive different zones at the same
input candle (FC: ad-hoc BOS_0; sibling types: the sibling's CTS zone; reversal: the reversing
structure's CTS zone), so a later same-input probe run on its own reference could in principle reset
differently. The model chooses to treat the first probe's reference as applicable to all. Tripwire:
on every hit compare the hitting trigger's reference-zone inner to the cached `bos0_inner` (iteration 1's
BOS_0 threshold IS the reference inner) and log `[probe_cache] REF-ZONE DIFFERS …` when they are not
`isclose` — so a `/compare` delta can be traced to the assumption having bitten.

**Hit rule:** a hit returns the cached `starting_idx`, `finalize_idx` and `bos0_inner` and skips
`unified_probe`; the record's `probe_finalize_idx` IS the cached finalize (§2.1). A different-input
trigger that converges on the same `starting_idx` is a different run: it probes, keeps its own finalize,
and the POOL (not the cache) dedups the MS build. Log every hit; when `end_idx` differs from the cached
bound log it as `[probe_cache] APPROX hit …`. (An earlier "decision_idx" window-exact hit rule and an
earlier reference-zone-in-key proposal were both considered and NOT adopted — the model wants
first-probe-is-truth, period.)

Hit rate on this window: **zero** (the only same-input pair, sub 7's `subsequent_confluence`(1,1) and
FC(1,2), never meets — the (1,1) trigger is in a degenerate cycle and is not probed). Dormant here;
Plan B's early stop is the real probe saving. Wire it anyway (small, and it is the specced mechanism).
On a re-probe of an identical `(input, end_idx)`, the result must be byte-identical — assert.

### 5.4 Geometry: run cap = data edge; one builder
**`_build_or_get_sub_geometry` is the surviving builder** (rename to `build_or_get_geometry`;
`run_cap_abs=len(m15)-1` always). It is the real implementation (slicing + 50-lookback + `reset_index` +
`compute_imbalance` + `is_range` re-derivation). Delete `pooled_structure_build.build_structure_geometry`
(never had a live caller) and point `test_pooled_structure_build.py` at the survivor (`run_cap_abs` is a
required int there — pass `len(df)-1`; with `start=0` the slice is the whole df so `slice_begin=0` and the
equivalence assertions hold unchanged). Delete the `parent_struct_end_m15` plumbing and `run_cap_eff =
max(run_cap_abs, end_m15_abs)`. **Ordering on a miss:** run MS first; create the pool entry only on
success (today `get_or_create` runs first and a failed build leaves a geometry-less `PooledStructure`
that consumed a `sub_id` and makes every later trigger to that key a false cache hit). Return
`(sub, created)` or `None`. No KL derivation happens here (§5.1 explains why none is needed).
`build_one_sid` itself is **deleted**: its window-projection + mirror role moves to §6.1, its record
creation to §4.3, its floor to the record, its reversal-probe branch to the reversal resolver (§4.3 step 2).

---

## 6. Rendering + exports

### 6.1 One projection per sub, mirrored per lens
```
floor_local = sub.start_idx - slice_begin ; cap_local = (sub.end_idx - slice_begin) if sub.end_idx else None
down = project_to_window(bounded, floor=floor_local, cap=cap_local, cap_reason=sub.end_reason, direction=…)
first  = min(sub.live_records(), key=lambda r: (r.start_idx, r.seq))
result = LowerTFResult(trigger=first.source_trigger,        # required field; read by the mirror (use_case/
                                                            # parent_*/lower_tf/lower_sd), the WVMI persister
                                                            # and build_sid_records_for_subordinate
                       df=bounded.df.iloc[0:(cap_local if cap_local is not None else len-1)+1],
                       events=deepcopy(down["events"]), ..., meta={"sub_id", "m15_start_idx"=starting_idx,
                       "start_idx", "m15_end_idx"=end_idx or m15_edge, "end_reason", "natural_reversal_idx",
                       "slice_begin", "lenses", "validated_h1_start"=first.validated_parent_idx,
                       "first_record": {lens, S, C, trigger_type, trigger_idx}})
for lens in sub.lenses(): mirror_lower_tf_result_to_entity_df(lens_df[lens], result, structure_path_id=path[lens])
```
`clip_events_to_window` must **deepcopy** before returning (it hands out shared geometry objects today).
Attribution stamped by the mirror: `structure_path_id`, `timeframe`, `parent_tf`, **`sub_id`**, and —
informational only, from `first_record` — `parent_sid`, `parent_cycle_id`, `use_case`. No consumer may
use the informational three for identity.

`_STRUCTURE_COLS` mirror: write subs in **`start_idx` order** (later-live wins overlapping candles).

### 6.2 Chart (`export_m15_chart.py`)
- Identity everywhere = `sub_id` (`_sub_identity`, `_sid_record_identity` → int). Grouping of
  events/zones/POIs/fibs/wave/WVMI by `meta["sub_id"]`.
- **`owner_by_idx` → `owner_by_idx_dir[(candle, direction)]`**, range = `[start_idx, end_idx or edge]`
  (the lifecycle, NOT the anchor). Sid-tied elements (dots, swing lines, PB markers, prev-BOS lines) draw
  only where owned. Persisting elements (KL/POI rectangles, fibs) unchanged — drawn from the anchor,
  active from `start_idx` (already the KL/POI rule).
- Hover: `sub_id`, direction, `relative_dir` at that candle, `[start_idx, end_idx]` + reason, and the
  record list `(lens, (S,C), trigger_type, trigger_idx→start_idx)` read from `attrs["triggers"]`.
- `m15_most_recent_psid` / `recent_cycle_ids` tier logic: keep, sourced from `first_record` of each sub
  (implementer's choice to simplify; must not change H1-overlay behaviour).
- Retire the `(parent_sid, parent_cycle_id, sub_sid)` tuple at all 12 sites.

### 6.3 Exports (`run_replay.py`, `debug/`)
Per lens df: `attrs["sids"]` (sub-level `SidRecord`s), `attrs["triggers"]` (records whose `lens` is this
lens, incl. zero-length), `attrs["unresolved_triggers"]` (pool-wide; write once). CSVs, **decoupled from the
chart loop** (write even if the chart export raises):
- `*_M15_{lens}_subs.csv` — one row per sub on this lens: `sub_id, direction, starting_idx, start_idx,
  end_idx, end_reason, natural_reversal_idx, lenses, relative_dir_segments, n_records, first_record_*`
- `*_M15_{lens}_triggers.csv` — every §2.1 field, one row per record on this lens
- `*_M15_unresolved_triggers.csv` — every §2.3 field
- `*_M15_{lens}_sids.csv` is **removed** (its role is split across the two above).
`export_wvmi.py` / events / zones / poi / fib CSVs: `sub_sid` column → `sub_id`.

### 6.4 WVMI — minimal change; implements the user's stated LEAN, not a WVMI design
The user deferred WVMI to its own pass and said "my current leaning is that WVMI will be a property of the
unique sub … I think that would mean one sweep". This section implements that lean so the code runs; the
deferred WVMI pass may revert to today's per-(sub, lens) sweeps. Do not treat it as settled.
`_assign_trigger_centric_sub_wvmi` runs **once over the unique subs**: window = `[sub.start_idx,
sub.end_idx or edge]`; stream = union of the confluence stream and the counter stream **restricted to the
sub's lenses**; first trigger (by idx) inside the window sweeps the sub once; dedup key `sub_id`;
`persist_facade_wvmi_to_entity_df` into every lens df the sub is on. Delete `_merge_wvmi_counts`,
`conf_results_by_cycle` / `ctr_results_by_cycle`, and the `meta["m15_df_prepared"]` write (post-revert
there is one, at `orchestrator.py` ~863; nothing reads it but a docstring). WVMI record meta `sub_sid` →
`sub_id`. Everything else about WVMI is out of scope.

---

## 7. Removals and renames (checklist)

**Delete:** `_ChainCursor` (incl. its `_boundary_for_trigger` method), `build_two_entity_parent_cycle`,
`build_parent_cycle_chain`, `build_one_sid`, `render_unique_sub` (already gone after P0),
`_find_m15_lifecycle_end` and `_run_subordinate_probe` + `_resolve_via_legacy_probe` + the
`_LEGACY_PROBE_USE_CASES` escape hatch (empty set, dead; it returns `finalize_idx=None`, which would trip
the §5.2 assert — retire it and delete `lower_tf_pipeline.py` outright), `finalize_lifecycles`,
`select_lifecycle_end`, `PooledStructure.memberships/earliest_membership`, `SubStructurePool.for_lens`,
`_merge_wvmi_counts`, `_assert_m15_frames_aligned`, `_sub_path_id_for_use_case`,
`build_structure_geometry`, `TriggerRecord.meta`, the `"confluence" in sub_path_id` lens test,
`lifecycle_end_idx` readers. **Keep** `_END_REASON_PRIORITY`.

**Rename `sub_sid` → `sub_id`** on artifacts (hard, no alias): `entity_df_mutation.py` (mirror
attribution), `export_m15_chart.py` (12 sites), `sid_records.py` (the ONE sub-level site, `:99`; the main
path `:57` keeps `sub_sid`), `debug/export_wvmi.py` (2 — the only exporter with a first-class `sub_sid`
column; the events/zones/POI/fib exporters carry it inside the serialised `meta` dict and follow the
mirror rename automatically), `run_replay.py` (1), `orchestrator.py:733` (WVMI dedup key),
`_build_sibling_cts_ref_zone` (→ §5.1), CSV headers. Grep `"sub_sid"` must return only the main-entity
`SidRecord` path afterwards.

**Split `start_trigger_idx`** into `trigger_idx` / `probe_finalize_idx` / `start_idx` (§2.1). Every reader
of `meta["start_trigger_idx"]` — the WVMI window (`orchestrator.py` ~719), sid records (`sid_records.py`
~111), and the KL/POI/fib `lifecycle_floor` plumbing (the chart never reads it) — moves to the field that
matches its role: lifecycle readers → `start_idx`; provenance → the other two. `_floor_abs` inside
`build_one_sid` disappears — the floor is a record field.

**`_dt` → `_idx`** anywhere a candle index is named `*_dt` (`TriggerRecord.trigger_dt`, log strings).

**Probe search bound `end_idx` → `probe_end_idx`** (user-agreed 2026-09-19; naming collision with the new
record/sub lifecycle `end_idx`). The probe's bound is a *compute* bound (inclusive upper edge of the search
window, like `run_cap`) and has nothing to do with lifecycle. Sites: `unified_probe(end_idx=…)` (all three
signatures, `unified_probe.py:339/451/652`) and its `ProbeResult` docstring table; `FirstConfluenceTrigger.end_idx`
(`first_confluence_trigger.py:101/104`, and the H1-side `FirstConfluenceTrigger` doc); the §5.3 cache
hit/APPROX log strings; PART4 §4.3 "Probe end_idx" rows (already so named there). The `MultiTFTrigger.meta`
key is already `probe_end_idx` — unify the rest on it. Same collision, same fix, for the probe's **output**:
`ProbeResult.start_idx` is the structural anchor (the pool key), NOT lifecycle — rename it `starting_idx`
so the plan's vocabulary (`starting_idx` historical / `start_idx` real-time, §2.1) is what the code says.
After both renames `grep -n "end_idx\|start_idx" structure/unified_probe.py` must return only the
`probe_end_idx` / `starting_idx` spellings (plus unrelated locals), and every `end_idx` / `start_idx` on
`TriggerRecord` / `PooledStructure` is a lifecycle value.

---

## 8. Logging (stdout, greppable — the `/compare` log grep in the skill relies on these)
- `[parent_tables]` one line per parent cycle: floor/end/degenerate; WARNING per degenerate.
- `[sweep] fire …`, `[sweep] record …`, `[sweep] replace …`, `[sweep] spawn …`, `[sweep] sub_end …`.
- `[sweep] UNRESOLVED (skipping) reason=… lens=… parent=(S,C) type=… trigger_idx=…` — the word
  "skipping" is deliberate: the `/compare` skill greps `warning|skipping|unavailable|degenerate|pending|no sid`.
- `[probe_cache] hit|APPROX hit|miss`.
- `[pool] sub_id=… key=… start=… end=… reason=… lenses=… records=[…]` (the table §10 is checked against).

---

## 9. Tests

### 9.1 Break by design — rewrite in the same commit
| test | action |
|---|---|
| `test_sub_structure_pool::test_cross_chain_reversal_ends_active_counter_sub` | invert: a confluence-lens −1 successor does NOT end a counter-lens −1 record; add the same-lens case that does |
| `…::test_replaced_then_retriggered_overextension_is_the_known_edge` | becomes the guard for the FIXED semantics. Its three triggers are all `(confluence, 0, 0)`, so under §4.3 step 4 the 600 re-trigger is **absorbed** into S's existing record (`extra_trigger_idxs`) — S's record and sub still end at R's start (400). Add a second case with the re-trigger in the NEXT parent cycle `(confluence, 0, 1)`: that one creates a **zero-length record** with the frozen end (§4.3 step 6) |
| `…::test_3304_cross_parent_cycle_stays_continuous`, `…::test_earlier_membership_parent_end_does_not_cap_multicycle_sub`, `…::test_lifecycle_start_is_earliest_trigger`, `…::test_same_direction_replacement_ends_prior_sub`, `…::test_own_reversal_ends_sub`, `…::test_select_end_*` | port to the sweep API; 3304 and 2365 become §9.2 fixtures |
| `test_sub_chain.py` (14) | the chain is gone. Layer-1 `build_one_sid` tests → port to the geometry builder; Layer-2 cursor tests → delete; `test_two_cycles_sub_sid_resets` → assert `trigger_sub_sid` resets per (lens, parent) instead |
| `test_sid_records::test_subordinate_*` | sub-level `SidRecord` shape (§2.5) |
| `test_pooled_structure_build.py` | point at the surviving geometry function; keep the equivalence assertions |
| `test_first_trigger_migration.py::TestResolveSiblingCts` (~12, `:336-620`) + `TestSiblingCtsIdxWindow` | they call `_resolve_trigger_m15_start(..., sibling_entity_df=…)` / `_build_sibling_cts_ref_zone(sibling_entity_df, …)` and build sibling dfs whose events carry `sub_sid`. §5.1 changes both signatures → wholesale rewrite against a stub pool (records + slice-local geometry) |
| `test_sub_structure_pool::test_memberships_and_earliest`, `::test_sub_on_both_charts_via_lens_union` (uses `for_lens`), `::test_probe_cache_roundtrip_and_setdefault` (cache value shape + overwrite semantics change), and the module-level `_tr()` helper (`TriggerRecord(trigger_dt=…, meta=…)`) | delete / rewrite to the §2 shapes |
| `test_sub_structure_pool::test_sub_id_is_monotonic_and_stable` | expected to SURVIVE unchanged (`get_or_create` / `sub_id` assignment are untouched); if it fails, the pool API changed unintentionally |

### 9.2 New — the predicted-table test (the internal checkpoint)
`test_lifecycle_sweep_predicted_table.py`: feed the sweep the 16 H1-derived + reversal triggers from
`reference_pool_redesign_groundtruth.md` with **stubbed** probe results (known `starting_idx`,
`finalize_idx`, `finalize_condition`) and **stubbed** geometry (known `natural_reversal_idx`: 1940, 2470,
2829, None, None, None, 3589, None, None, 4200, None), plus the §3 tables built from the saved H1 events
CSV. Assert the full predicted table (every record's `start/end/reason/trigger_sub_sid/relative_dir`, every
sub's `start/end/reason/lenses`, the 4 unresolved rows). **Key subs by `(direction, starting_idx)`, not
by the table's labels**: the memory table's "sub 7/8/9/10" are the BASELINE's numbering; under Plan C
`sub_id` is assigned at creation, so the 8 built subs are `sub_id` 0–7 in creation order
(454, 1797, 2365, 2639, 3304, 3621, 3760, 4027). Runs in <1 s. Plan A's leak fix touches no
value in this table (FC(1,0)/(1,1) are unresolved under Plan C; FC(0,0)'s leaked candles 1722–1725 lie
outside its `[459..783]` retrace window) — **run this test after Plan A lands to prove that**, and if it
fails, update fixture + memory table together.

### 9.3 New — unit
Record floor (`start = max(finalize, floor)`); degenerate-cycle detection (inverted + zero cases);
zero-length exclusion from all four aggregations; strict `>` with same-candle handover (both directions);
per-lens replacement (+ the cross-lens non-replacement); post-end re-trigger → zero-length with the frozen
end; reversal-spawn only when a live record covers R; `bos0_inner` assert on cache hit; probe-cache hit
rule (equal `end_idx` / native / re-probe); `BOS≡CTS_EST` assert; `finalize_idx` per-condition table
(`test_unified_probe` has ZERO assertions on it today); `relative_dir` step function incl. a parent-sid flip;
`owner_by_idx_dir` with an opposite-direction overlap; exports written when the chart export raises.

---

## 10. Acceptance (`/compare` + chart review)

1. `[pool]` log lines / `_subs.csv` / `_triggers.csv` **equal the predicted table** in
   `reference_pool_redesign_groundtruth.md`, matched by `(direction, starting_idx)` (the emitted `sub_id`s
   are 0–7 in creation order, not the table's baseline labels). Own reversals for subs 3/7/8 may now be
   discovered with the run cap at the data edge — if one lands before the listed end, that end moves
   earlier and a successor appears; document it as intended.
2. H1: all 9 CSVs byte-identical.
3. M15 intended deltas vs the Plan-B save: `*_sids.csv` gone, three new CSVs; the baseline's **16**
   per-trigger sid rows (11 conf + 5 ctr) become **8 subs / 11 records / 4 unresolved** (the 16th, the
   counter reversal in (1,1), never fires because sub 6 is never built); subs 4/5/6 absent (were
   `inactive` outlines); sub 3 drawn on the counter chart from 2829 (was
   2843); sub 10 on confluence from 4083 overlapping sub 9 over [4083,4200] (opposite directions —
   intended); `end_reason` vocabulary; `sub_sid`→`sub_id` column renames; `lifecycle_end` absent; possible
   small `starting_idx` shifts from §5.1 (each must be explained by a sibling `CTS_UPDATED` inside the
   window that the old bounded build had cut); exactly three sub-cycle lifecycle starts +1 candle from
   the moment-not-extreme rule (§3 — the cycles whose `CTS_ESTABLISHED.idx` precedes `confirmed_at`:
   1223→1224, 2828→2829 ×2) with their KL/POI `confirmed_idx` clamps following.
4. Anything else = regression. The `/compare` skill now reports NEW/MISSING files explicitly.
5. **Pause for chart review** before `/commit-save` (ownership + hover changed).

---

## 11. Docs in the same commit
- **PART4 §17**: **ALREADY REWRITTEN (rev 2, 2026-09-19)** from the memory file + this plan — §17.1
  motivation, §17.2 split, §17.3 identity (`relative_dir` vs `lens`), §17.4 records, §17.5 sub lifecycle,
  §17.6 sweep + parent tables + moment rule, §17.7 degenerate/unresolved, §17.8 build model (geometry,
  probe cache, sibling reads), §17.9 rendering/storage/exports, §17.10 WVMI minimal, §17.11 validation,
  §17.12 out of scope; the §2/§5/§6/§9/§16.5 pointer banners state the rev-2 rules. In THIS commit:
  update the §17 header (drop the "until Plan C lands the docstrings cite rev 1" note; status → LANDED)
  and rewrite the section BODIES the banners cover: **§6** (merge-and-bound chain + §6.1 bootstrap rule
  retired; §4.3.6 cadence diagram → sweep), **§5** "sub structure lifecycle-start" bullet → record vs
  sub, **§9** single-store note → one feature frame + N lens dfs by mirror, §9.4 `attrs["sids"]` table →
  `sub_id` + the three CSVs, **§13.5** `lifecycle_end_idx` prose. Keep §17.x numbering — the pool
  modules' docstrings must cite the rev-2 numbers after this commit.
- `WVMI_SPEC.md` "start_trigger_idx IS the sub's lifecycle-start" → sub `start_idx`.
- `KL_ZONES_SPEC.md` + `POI_ZONES_SPEC.md`: `end_reason` vocabulary; "cap applied in `build_one_sid`" → the
  sweep/projection.
- `LANDMINES.md`: update "Sub Lifecycle-Start Clamp" (floor lives on the record; `build_parent_cycle_chain`
  gone), "Lower-TF Zones … Must Be Capped" (`deactivated_by` → `end_reason`), the four pool entries
  (replace the PARTIALLY SUPERSEDED banner with the new rules); add "The Sweep's Phase Order Is
  Load-Bearing".
- `GOTCHAS.md` "Cross-referencing a sub record to its zone: join on `sub_sid`" → `sub_id`.
- `ARCHITECTURE.md` lifecycle convention: `TriggerRecord` + unique sub are tier-1 objects. **And the
  "`ev.idx` convention" block: `CTS_ESTABLISHED.idx` is the CTS extreme (a second extreme-not-apply
  exception beside `BOS_CONFIRMED`); timing/lifecycle reads use `meta["confirmed_at"]`.**
- **PART4 §5** "cycle (any): `max(CTS_established_idx, …)` … the CTS extreme — canonical cycle-start idx"
  → the CTS-established **moment** (`confirmed_at`); B2 Phase B's floor/cap consistency argument restated
  on the moment. `KL_ZONES_SPEC.md` "Lifecycle-start clamp" and "End-side change: BOS ends align to the
  cycle boundary (CTS-established extreme)" → moment.
- PART4 **§16.5** "Persistence and display rules" (the numbered §16.5 lives in PART4, not
  `CHARTING_SPEC.md`): ownership = lifecycle window, per direction; and the matching prose in
  `CHARTING_SPEC.md`'s M15 overlap section.
- `GOTCHAS.md`: add the `ref=cts_confirmed` label trap (§5.1) and that the CONFIRMED-zone branch is dead
  for subs.
- `GLOSSARY.md`: already current.
- Memory: `project_sub_structure_pool_architecture.md` status → "Plan C LANDED (commit …)", the
  predicted-table memory → "VALIDATED".

---

## 12. Landmines to keep in view while implementing
- Geometry objects are SHARED; anything that stamps meta must deepcopy (mirror does; `clip_events_to_window`
  must).
- `knowable_at_idx` special-cases only `BOS_CONFIRMED`; `CTS_ESTABLISHED`/`REVERSAL_CANDIDATE` straddle
  too. Not fixed here (out of scope) — do not "improve" it mid-plan; note any half-clipped cycle you see.
- The sibling idx window `[lo, hi]` is the only thing keeping the sibling read causal once geometry runs
  to the edge. Never widen it.
- `LOH` maps on timestamps, not `4h+3`; the identity is a cross-check only.
- Phase order at equal idx (0→4) and start-before-end are not implementation details.
- The H1 chart's optional M15 overlay reads sub `attrs` too — grep `export_plotly.py` for `sub_sid`.
- Config window: keep the current window commented-out blocks intact (`feedback_config_window_preservation`).
- Display the `=== Replay Timing ===` block after the replay.

## 13. Definition of done
All §9 tests green; §10.1–10.3 satisfied and each delta explained; chart review signed off; §11 docs
updated in the same commit; `/commit-save`; memory status updated.
