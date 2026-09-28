# Plan F — imbalance c3 knowability: an FVG counts only once it has formed (zones pass, 2026-09-23/24)

**Status:** **LANDED 2026-09-24** (as-landed record §9; every §4 prediction held; chart review "charts look good").
rev 2 was cold-reviewed (§10); user GO with M1 divergence = accept + document and `evaluated_at` = required keyword.
**Inputs (read first):** [`PLAN_F_inputs.md`](PLAN_F_inputs.md) — the verified 39-row consumer audit, the shadow
measurement of every call site, and the per-scope cascade replays (§2). Measurement scripts: `plan_f_inputs/`.
**Base:** HEAD `754a642` (docs-only on Plan D `0a4eadc`). `/compare` baseline = save `20260923_172626_0a4eadc`.
Line numbers are HEAD coordinates.

## 1. The defect

`compute_imbalance` flags **c2**, the middle candle. An `ImbalanceInstance(start_idx, end_idx)` is a run of c2s, and
its gap exists only once a **c3** has closed. Two places count it one candle early:

1. **The POI activation sweep** — `poi_zones.py:920` `enter_idx = max(inst.start_idx, first_active)` puts the
   instance in the unfilled set at its first c2.
2. **The shared primitive** — `has_unfilled_imbalance` (`imbalance.py:159-166`) counts every overlapping instance
   that `inst.is_filled(check_to_idx)` reports unfilled, and `is_filled` reports UNFILLED on an empty scan range
   (`types.py:223-224`, `end_idx >= check_to_idx`). So an instance whose c3 has not closed by the moment of the
   question counts as an unfilled imbalance. On this window this reaches three FibTracker decisions (§4).

Worked example (H1 instance #154, bearish, c2s 997-999): nothing is knowable at 997's close; the first gap
(996 low 0.57492 > 998 high 0.57486) exists at 998; the run grows at 999 and is final at 1000. Today the sell POIs
IC 865 / 860 re-activate at 997; the rule makes it 998.

## 2. Decisions (user, 2026-09-23)

- **Scope: the POI sweep + every FibTracker read that has a real moment, in ONE `/compare`.**
- **Rule R1 — "first c3":** an instance exists from `formed_at = start_idx + 1`. At a moment K it is visible iff
  `formed_at <= K`, and only its **formed prefix** `[start_idx, min(end_idx, K-1)]` is tested against the window.
  (Not R2 "last c3" = `end_idx + 1`: it waits for the run to finish although an unfilled gap already exists —
  measured 997→1000, 4118→4120, and it wrongly flips fib reads.) R1 is exact: a formed prefix's only scanned candle
  is its own c3, which cannot arm its own gap (`fill_threshold > 0`), so "prefix unfilled" == "instance unfilled" for
  every K in `(start_idx, end_idx]`. Caveat: a prefix and its merged run can differ in degeneracy
  (`gap_size <= 0`); 0 cases on this data (H1 0/218 prefixes, M15 0/842).
- **The cut is keyed on the MOMENT the question is asked (`evaluated_at`), never on `check_to_idx`.** Many callers
  pass an anchor as `check_to_idx` (the fill as-of — Plan E E3 territory); keying the cut on it would drop gaps that
  are knowable at the moment (measured on 2 pattern-path calls; lag-1 cycles 1223/1224 and 2828/2829 have the same
  shape).
- **Names (GLOSSARY Naming Standard — a moment is `*_at`):** `ImbalanceInstance.formed_at` (property,
  `start_idx + 1`) and the keyword `evaluated_at`. (`knowable_at` avoided: the pool has `knowable_at_idx`.)
- **MS in-flight snapshot: NO CHANGE + a spec note.** The POI-inner snapshot is built at the CTS refresh candle and
  can include a gap whose c3 is the next candle, but its only reader (`market_structure.py:1957`) is gated
  `i > st.cts.idx` and `cts_cycle_id > 0` (`:1975-1981`), so no decision uses a gap before it exists — the spec's
  per-candle model ("Snapshot vs per-candle — deliberate approximation"). Measured: 459 calls would change, 3 inner
  lists, 0 output cells.
- **FibTracker decides at the event.** FibTracker re-asks only at CTS events, so removing a not-yet-formed gap can
  DROP a fib/cross rather than delay it (counter sub 5's pre-created "cycle 1" cross fib — sub 5 only ever
  establishes cycle 0 — and its IC 3654 twin POI disappear). Accepted: the same outcome today's engine gives when the
  gap forms one candle later. "Re-ask when the gap's c3 closes" → follow-up (§7).
- **Cached values are judged at their USE moment.** FibTracker's cycle-0 liveness cache
  (`_cross_cycle_data[sid]["cycle0"]["has_unfilled"]`, read later as Scenario-2 **cond2**) stays UNCUT — at its use
  (CTS_1 ESTABLISHED > CTS_0) every gap in `[BOS_0, CTS_0]` has formed — which keeps **cond2** in agreement with the
  unchanged MS mirror (LANDMINES "Scenario 2 anchor agreement"). cond3 (its window ends at CTS_0, before any moment)
  agrees too; only cond1 and the activation can diverge (M1, next bullet).
- **Consequence to acknowledge at go (cold review M1): MS/FibTracker activation divergence.** The MS in-flight
  resolver stays uncut (decision above) and builds its fib as always-active (`poi_zones.py:1084-1094`); FibTracker
  now cuts at the event. In the drop case — a lag-0 CTS_ESTABLISHED or a raw CTS_UPDATED whose ONLY sd gap has
  c2 == the event candle — MS keeps a POI inner (usable for sd-proximity CTS confirmation from the next candle) for a
  fib/POI that FibTracker never creates. Permanent on H1 cycle ≥ 1 (fib activation is one-shot at EST,
  `fib_tracker.py:1246-1251`, `:1284-1288`), until the next CTS_UPDATED on subs. **0 cases on this window** (ms
  variant 24/24; H1 fib unchanged). This is a new instance of a divergence class that ALREADY exists today: a gap that
  forms AFTER an H1 cycle-≥1 EST enters the MS inners at the next CTS refresh but never activates the downstream fib
  (not measured). The "re-ask when the gap forms" follow-up closes the c3 instance.

## 3. Code change

### 3.1 `common/types.py` — `ImbalanceInstance`
- Property `formed_at -> int` = `start_idx + 1`, docstring: "the close of the instance's FIRST c3 — the moment its
  gap exists (a merged run keeps growing until `end_idx + 1`); see IMBALANCE_FILL_SEMANTICS 'Knowability'". Not a
  dataclass field: `export_imbalances.py` writes explicit columns → no CSV change; `__deepcopy__` returning `self`
  is unaffected.
- Method `overlaps_formed_prefix(start_idx, end_idx, evaluated_at) -> bool` next to `overlaps` (`:139-141`):
  False if `formed_at > evaluated_at`; else `self.start_idx <= end_idx and min(self.end_idx, evaluated_at - 1) >=
  start_idx`. (The R1 geometry lives on the instance — one definition.)
- `is_filled` docstring (`:194`): the empty-scan case stays UNFILLED (correct for a formed prefix); callers asking at
  a moment must apply the `formed_at` cut. No logic change.

### 3.2 `patterns/imbalance.py` — `has_unfilled_imbalance`
Signature: `(df, start_idx, end_idx, check_to_idx, fill_threshold=0.70, *, direction=None, evaluated_at)` —
**`evaluated_at` keyword-only and REQUIRED** (cold review S6: a silent default would be the defect class this plan
fixes, and `select_fib_anchor_for_cycle` is called positionally by both layers — GOTCHAS "Positional resolver args
drift"). `evaluated_at=None` is an explicit, documented "no knowability cut" (retrospective questions, the unchanged
MS in-flight path, the uncut caches, and events with no recorded moment) and is byte-identical to today:
```python
for inst in df.attrs.get("imbalances", []):
    if direction is not None and inst.direction != direction:
        continue
    if evaluated_at is None:
        if not inst.overlaps(start_idx, end_idx):
            continue
    elif not inst.overlaps_formed_prefix(start_idx, end_idx, evaluated_at):
        continue
    if not inst.is_filled(df, check_to_idx, fill_threshold):
        return True
return False
```
Docstring rewritten: the two as-ofs (`check_to_idx` = fill horizon, `evaluated_at` = the moment of the question);
sd-direction strict everywhere (the "permissive / no direction" text at `:149-154` is stale since 2026-05-23).
`get_unfilled_imbalances` / `has_imbalance_in_range`: untouched (no production caller — §7).

### 3.3 `zones/poi_zones.py`
- **Sweep** `:920` `enter_idx = max(inst.formed_at, first_active)`. Leave (`confirmed_fill_idx >= end_idx + 2`),
  the relevance filter `:912-918` (`end_idx > ic_idx` ⇔ `start_idx > ic_idx`: the IC is a −sd candle, never a c2 of
  an sd run) and the debug dump `:640-646` unchanged. Comments: the documented identity (`:516-518`) becomes
  `has_unfilled_imbalance(df, ic_idx+1, t, check_to_idx=t, direction=sd, evaluated_at=t)`; `:809-812`, `:817-819`
  ("enter unfilled set" = `inst.formed_at`), `:924-925`.
- **`find_ic_candidates` `:229`**: `evaluated_at=None` explicit, with a one-line reason per caller — retrospective
  final-fib IC identification (`derive_poi_zones`, Plan E §2.4 item 8; measured 0 cells even if cut) and the
  unchanged MS in-flight resolver (§2). Fix the stale "permissive" note `:224-228`.
- **`compute_poi_inners_for_cycle`**: its `select_fib_anchor_for_cycle` call (`:1052-1063`) passes
  `evaluated_at=None` explicitly; docstring notes the M1 divergence (§2).

### 3.4 `zones/cross_cycle_fib.py` — `resolve_cross_cycle_eligibility`
Keyword-only REQUIRED `evaluated_at: Optional[int]`, passed to all three `has_unfilled_imbalance` calls (`:117` own
imbalance, `:149` snapshot liveness, `:156` current liveness). `:149`/`:156` are already bounded (windows end before
the moment; 0 of 170 calls in scope; `dead_cycles` cannot latch wrongly — every walked window ends at
`CTS_k < conf_k <= the moment`) — passing it keeps the routine's questions uniformly "as of the moment".
`prior_cached_liveness` (cond2) untouched (§2 cache principle). Docstring + the Plan E note (§6: after E3a
`current_candle` and `evaluated_at` carry the same value on the moment paths — E3a keeps ONE moment parameter).

### 3.5 `structure/market_structure.py` — the moment, defined next to its emitter (no behaviour change)
- Constant `CTS_UPDATED_RAW_VIA = "replay_raw"`; used at `:638`; `_maybe_update_cts_pre_confirm`'s dead `"raw"`
  default (`:1701`) removed (the sole caller passes it). (Cold review S5: one definition of "raw path".)
- Public `event_moment(ev: StructureEvent) -> Optional[int]` next to `StructureEvent` (`:156`):
  - `CTS_ESTABLISHED` → `int(ev.meta["confirmed_at"])` (direct index — the event contract; Plan D);
  - `CTS_UPDATED` → `int(ev.idx)` iff `ev.meta["via"] == CTS_UPDATED_RAW_VIA` (raw `idx` = the processing candle),
    else `None` — the pattern path's `idx` is the CTS anchor and no moment is recorded (ARCHITECTURE "`ev.idx`
    convention"; Plan E). `meta["via"]` by direct index (every emitter sets it: `:1551`, `:1716`, `:1722`);
  - `CTS_THRESHOLD_UPDATED` → `int(ev.idx)` (the processing candle, `_sync_thresholds_from_range(i)`,
    `:1767-1790`);
  - any other type → `raise ValueError` (fail loudly; a new consumer must define its moment).
  (The pool's `knowable_at_idx` — EST → `ev.idx`, §17.12 known limit — and `unified_probe._second_cts_moment` encode
  the same concept; they converge onto `event_moment` in Plan E E3b, §7.)

### 3.6 `zones/fib_tracker.py`
- **`select_fib_anchor_for_cycle`**: keyword-only REQUIRED `evaluated_at`, forwarded to
  `resolve_cross_cycle_eligibility`; the docstring's "cond1 … as of cts_idx" gains the moment.
- **Event scope**: a context manager `_evaluating(event)` that sets `self._evaluated_at = event_moment(event)` for
  the handler body and RESTORES the previous value on exit, asserting it is not entered while already inside a
  handler (none nest today — no `self.on_cts_` call in the file). `on_cts_established`, `on_cts_updated`,
  `on_cts_threshold_updated` run their bodies under it (the three public entry points that reach an imbalance read;
  `on_cts_confirmed`, `set_reversal_terminals`, `_finalize_lifecycle_fields` reach none). `self._evaluated_at` is
  `None` outside a handler; inside one, `None` means the event has no recorded moment (pattern-path CTS_UPDATED). A
  helper `_has_unfilled(df, lo, hi, check_to, sd)` passes `evaluated_at=self._evaluated_at`.
- **Call sites** (all 11 production calls):
  | Site | Change |
  |---|---|
  | `:536` `on_cts_established` | `has_unfilled` for decisions via `_has_unfilled` (moment = `confirmed_at`). ALSO compute `c0_has_unfilled_uncut` with `evaluated_at=None`, passed down `_handle_sid1plus_cts_established` → `_handle_cycle0_scenario1`, which caches THAT at `:763`; its own activation decision (`:773`) uses the cut value. |
  | `:1183` cross_cycle cycle-0 first activation on update | `_has_unfilled` |
  | `:1318` cycle-0 cache re-snapshot | the cache write stays `evaluated_at=None`; the immediate reads at `:1330` and `:1351` become a fresh `_has_unfilled` over the cached `[bos_idx, cts_idx]` window with `check_to = c0["cts_idx"]` — absent `c0` (reachable: `:1347` sets Scenario 1 without checking it) → False, today's default; else direct index |
  | `:1436/:1445/:1451` `_update_fib_cts` cross branch | `_has_unfilled` (dead — `is_cross_cycle` never True on a 2-tuple key; routed for uniformity, deletion is §7) |
  | `:1462` `_update_fib_cts` normal branch | `_has_unfilled` |
  | `:1527` cond1 (cycle 0 @CTS_0), `:1540` cond3 (@BOS_1) | `_has_unfilled` — already bounded, no effect |
  | `:1535` cond2, `:1571` create-on-fail | `_has_unfilled` (lock-step) |
  | `:880` `select_fib_anchor_for_cycle` | `evaluated_at=self._evaluated_at` |
  | `_m15_cross_check` `:2158`, `_maybe_activate_main_cross` `:1991` | `evaluated_at=self._evaluated_at` |
- **Explainability:** the "NOT activated … no unfilled imbalance" / "NO FIB" prints gain `evaluated_at=` so a log
  reader can tell "not yet formed" from "no gap" (log only; not compared).
- `:870-875` comment: MS and FibTracker agree on cond2 and cond3 (cond3's window ends at CTS_0, before the moment,
  so the cut cannot change it); cond1 may diverge (M1). `fib_tracker.py:25` unused import: leave (§7).
- **Pattern-path duplicates:** a pattern-path CTS_UPDATED at the same idx as a raw one (1 of 37 rows,
  ARCHITECTURE `:100`) is dispatched right after it (stable sort, `orchestrator.py:152`) with no moment → re-asks
  uncut at `:1183`/`:1330`/`:1351`. Correct in knowledge terms (its apply candle is ≥ idx+1); 0 cells.

### 3.7 Not changed
The MS in-flight resolver and cycle-0 snapshot (`market_structure.py:2050` passes `evaluated_at=None` explicitly,
reason: §2), `find_ic_candidates` IC identification (Plan E), `_compute_fill_idx_cache`, all transitive consumers
(proximity triggers, WVMI, trigger detectors, mirrors, charts — they move with the POI rows), exports, the
`is_imbalance` chart highlight (a rendering of the pattern's c2 location, not a time).

## 4. Numeric prediction (stated before code; anything else = STOP and name the mechanism)

Provenance: `PLAN_F_inputs.md` §2.2 — the sweep-only and fib-only replays; sweep+ms+fib was replayed and is their
exact union, ms-only is 24/24, so sweep+fib is DERIVED, not replayed. The fib-only harness cut differed from §3.6
only where 0 cells change (pattern-path CTS_UPDATED cut; the two cycle-0 cache writes cut) — verified by the cold
review (D1-D4).

**19/24 CSVs byte-identical** — `final`, `raw`, `imbalance_instances`, `structure_events` (H1 + both lenses),
`structure_levels`, `kl_zones` (all), `wvmi` (all), `fib_lifecycle` (H1), `subs`, `triggers`,
`unresolved_triggers`.

| CSV | Rows | Cells |
|---|---|---|
| H1 `poi_zones` | 2 of 5 | sid 1 cyc 2 IC **865** and **860**: `confirmed_idx` 997→998; `activation_history` re-activations 953→954 and 997→998 |
| M15 confluence `poi_zones` | 1 of 30 | sub 7 cyc 0 IC **4048**: `confirmed_idx` 4118→4119 (+ its history entry) |
| M15 counter `poi_zones` | 13→**12** | IC 4048 as above; the sub 5 pre-created "cycle 1" IC **3654** row (inactive, no end) **removed**. The sub 5 cycle-0 IC 3654 row (activated 3707, ended 4083) is **unchanged** |
| M15 confluence `fib_lifecycle` | 2 of 21 | sub 2 cyc 0: meta `activated_at` 53→54 **+ `activated_on: 'update'`**; sub 3 cyc 0: `activated_at` 61→62 |
| M15 counter `fib_lifecycle` | 9→**8** | sub 3 cyc 0: `activated_at` 61→62; sub 5 cyc 0: `end_idx` 3806→**4083**, `end_reason` `new_cycle`→**`same_dir_replacement`**, meta `obsolete_reason` removed; sub 5 pre-created "cycle 1" cross row (start 3806, active, v0) **removed** |

The three flipped calls: `fib_tracker.py:536` CTS_ESTABLISHED local 53 = abs 2368 (lag 0); `:1183` raw CTS_UPDATED
local 61 = abs 2650; `cross_cycle_fib.py:117` CTS_THRESHOLD_UPDATED local 235 = abs 3806. Fib meta
`activated_at`/`reactivated_at`/`locked_at` stay slice-local (export-hygiene item); `deactivated_at`,
`start_idx`/`end_idx`, `bos_idx`/`cts_idx` are shifted (`entity_df_mutation.py:274-289`).

**Charts:** H1 **85/245** and M15 confluence **294/233** unchanged; M15 counter **153/125 → 151/124** (−2 POI
hover traces and −1 outline rect: the removed IC 3654 "cycle 1" POI; M15 fibs are not rendered,
`export_m15_chart.py:109`). Moved, not counted: H1 POI stretch starts / confirm lines 953→954 and 997→998 (hover
`zone_confirmed_idx` 997→998) on the H1 chart AND the H1 overlay on both M15 charts; M15 IC 4048 stretch / hover
4118→4119.
**Log:** in the counter sub-5 projection `[fib_tracker] total fibs=` (`orchestrator.py:262`) and
`[poi_zones] total=` (`:318`) each −1; the `CROSS (0->1) v0 ACTIVATED` line disappears; the three decisions' `[fib]`
lines change (+ the new `evaluated_at=` text). `[sweep] UNRESOLVED` set and `[probe_cache]` lines unchanged.
**Tests:** 739 + the new ones pass, 1 strict xfail.

## 5. Tests (tests first)

**Behaviour tests — each FAILS on HEAD:**
- `test_imbalance.py` (after `:345`): `formed_at == start_idx + 1`; single instance `(s,s)` with
  `evaluated_at == s` → not counted; merged `(1,3)`, window `[3,3]`, `evaluated_at=3` → NOT counted (prefix `[1,2]`
  does not reach the window); `overlaps_formed_prefix` unit cases.
- POI sweep (new `test_imbalance_c3_knowability.py`): an sd instance with c2 == `first_active` enters at
  `first_active + 1`; `_make_multicycle_data` POI (0,2) IC 12 full history `[(15,A)]` → `[(15,A),(19,D,
  imbalance_filled)]` (reproduced by the cold review); relabel the superseded comment at
  `test_poi_activation_moment.py:157-159`.
- FibTracker: lag-0 CTS_ESTABLISHED whose only sd gap has c2 == anchor == `confirmed_at` → no activation at EST;
  raw CTS_UPDATED (`via=CTS_UPDATED_RAW_VIA`) at t with the only gap c2 == t → the cross_cycle cycle-0 first
  activation does not fire at t and fires at the next raw update (no pattern-path duplicate in the fixture);
  `_update_fib_cts`: a raw UPDATED at t whose only unfilled sd gap has c2 == t deactivates at t with
  `reason='all_imbalances_filled'` (now "no formed unfilled imbalance") and reactivates at the next raw UPDATED;
  pre-established CTS_THRESHOLD_UPDATED with own gap c2 == `current_candle` → no cross.
- `event_moment`: an MS-emitted raw CTS_UPDATED → its idx; a pattern CTS_UPDATED → None; EST → `confirmed_at`;
  THRESHOLD → idx; an unsupported type → `ValueError`.
- Required keyword: calling `has_unfilled_imbalance` / `resolve_cross_cycle_eligibility` /
  `select_fib_anchor_for_cycle` without `evaluated_at` → `TypeError`.

**Guard pins — pass on HEAD, each catches a named wrong variant:**
- `test_imbalance.py`: merged `(1,3)` at `evaluated_at=2` → counted (catches **R2**); `evaluated_at=None` →
  today's results (byte-identity).
- A merged run enters the sweep at `start_idx + 1`, not `end_idx + 1` (catches **R2** in the sweep).
- **Lag-1** CTS_ESTABLISHED (anchor 9, `confirmed_at` 10, gap c2 = 9) → STILL activates (catches a cut keyed on
  **`check_to_idx`**).
- Pattern-path CTS_UPDATED with the gap at c2 == idx → no cut (catches cutting on an **anchor** as if a moment).
- Lag-0 cycle-0 EST (sid ≥ 1, h1 mode) with the gap at c2 == CTS_0 → `c0["has_unfilled"]` True while the Scenario-1
  activation at that EST does not fire (catches a **cut cycle-0 cache**).
- Pre-established CTS_THRESHOLD_UPDATED with own gap c2 == `current_candle − 1` → cross created (positive coverage;
  none today).
- `resolve_cross_cycle_eligibility(evaluated_at=None)` / `select_fib_anchor_for_cycle(..., evaluated_at=None)` =
  today (pins the unchanged MS in-flight path).
- **M1 pin:** a lag-0 cycle-1 EST whose only sd gap has c2 == CTS_1 → `compute_poi_inners_for_cycle` returns a
  non-empty list while FibTracker creates no cycle-1 fib (documents the accepted divergence; flips when the re-ask
  follow-up lands).

**Fixtures / mechanical:** the `_ev` helpers in `test_cross_cycle_fib.py:40-46` and `test_main_versioned_cross.py:
38-42`: CTS_ESTABLISHED gets `confirmed_at = idx` (lag 0), CTS_UPDATED gets `via = CTS_UPDATED_RAW_VIA`; a
pattern-path test passes a pattern name explicitly. Measured by the cold review (a direct-index plugin): 20
EST-driven tests (11 + 9), 8 of which also drive CTS_UPDATED (`test_cross_cycle_fib.py:102/118/267/306/340`,
`test_main_versioned_cross.py:104/119/155`); fixture gaps sit at 13/15/23/35/55 vs event idx 20/25/40/45/60/65, so
lag 0 + raw flips no pinned assertion. Required-keyword updates: `test_imbalance.py:336/340/341/358`,
`test_cross_cycle_fib_routine.py:61/74/204/221`. Relabel `test_poi_lifecycle.py:17-24` ("as of save 0a4eadc,
pre-c3").

## 6. Docs, skill values and memory that move in the SAME commit (implement-against-docs)

- **`IMBALANCE_FILL_SEMANTICS.md` — the canonical home.** New "Knowability — the c3 rule" section before the
  predicate: `formed_at`; R1 + the formed prefix + the exactness proof + the degeneracy caveat; the cut is keyed on the
  MOMENT (`evaluated_at`, required, `None` = explicit no-cut), never `check_to_idx`; cached values are judged at their
  use moment (MS snapshot, FibTracker cycle-0 cache); the M1 divergence; the per-consumer table. Also: `:34-36` (the
  `has_unfilled` answer is not monotone in the moment), `:43` (split the empty-range row: unfilled for a formed prefix
  / excluded when not formed), `:65-81` the keyword + callers, `:83-87` (no production caller), `:97-102` (sweep row +
  the MS in-flight path), `:104-116` regenerate the FibTracker matrix (stale line numbers; add `fib_tracker.py:1183`,
  `cross_cycle_fib.py:117/149/156`; an as-of-kind column; the cond1/cond2 label swap), `:118-128` (the MS site is
  `:2050` `_update_cycle0_data`; `_capture_cycle0_snapshot` does not exist), `:157-182` the enter side, `:186-203` a
  dated "c3 knowability (2026-09-23)" subsection with the deltas; fix `:199-203` ("live candle" is wrong for
  EST/UPDATED).
- **`POI_ZONES_SPEC.md`:** `:56-61`, `:80-85` (the two as-ofs), `:100-101` (the scan STARTS at the last c3 — "(incl.
  the last c3) are excluded" is wrong), `:137-143`, `:222-226`, `:263` (undefined `imbalance_idx`), `:272-273`,
  `:289`, `:393-397` (the fallback-POI live case "counter sub 5 cycle 1, IC 3654" — no live case on this window since
  Plan F; the general defect stays parked), `:427-430` (the sweep enters at `formed_at`).
- **`ARCHITECTURE.md`:** `:100` (`CTS_UPDATED_RAW_VIA`, `event_moment`), `:106-109` (narrow "Not audited" to
  BOS_THRESHOLD_UPDATED), add a `CTS_THRESHOLD_UPDATED` row (idx = processing candle), `:129-134` (FibTracker joins
  the direct-index `confirmed_at` readers), `:135-142` (raw vs anchor path; the "not yet measured" text), `:166-167`
  define "formed", `:212-222` the sweep identity with `evaluated_at=t`.
- **`GLOSSARY.md`:** `:82` (raw path = `CTS_UPDATED_RAW_VIA`), `:145-153` add `formed_at` / `evaluated_at` /
  `event_moment` (moments under the Naming Standard); `:153` the stale single-stroke fill definition.
- **`MARKET_STRUCTURE_SPEC.md`:** `:203-212` the consumer-gate bound (the snapshot may include a gap formed at the next
  candle; the only reader is gated `i > cts.idx`, so no decision uses it before it forms — do NOT add a refresh-time
  cut); `:326-332` (the bounded-run residual: the existence half is unobservable).
- **`LANDMINES.md`:** `:188-223` (Scenario-2 agreement holds for cond2 — FibTracker's cycle-0 cache stays uncut; the
  M1 activation divergence is a documented exception), `:1512-1517`, `:1535-1538` (the consumer list), `:2098` (stale
  name), `:2112-2119` (residual wording).
- **`GOTCHAS.md`:** `:840-862` (the existence case is closed; merged bounds harmless except degenerate gaps),
  `:1405` (quotes the pre-c3 history), `:1563-1566` (the late-activation fix is now cut at the moment); file the
  Windows >260-char path gotcha (scratchpad paths + long CSV names break Python `os.path`; use `\\?\`).
- **`CROSS_CYCLE_FIB_SPEC.md`:** `:66` cross-link, `:90-95`, `:133-135` (`current_candle` is a moment only on
  THRESHOLD and raw UPDATED; `evaluated_at` is the moment; E3a keeps ONE moment parameter).
- **`FIB_LIFECYCLE_SPEC.md`:** `:139`, `:337-339`, `:364-372` (`all_imbalances_filled` now means "no formed unfilled
  imbalance"), `:919-925` (the worked example 61→62); stale refs `:97`, `:582-583`.
- **`PRE_REFACTOR_INVARIANTS.md:221-233`**, **`CHARTING_SPEC.md:89-94`** (the `is_imbalance` highlight marks the c2
  location — a rendering rule, not the moment the gap exists).
- **Plans:** `PLAN_E_inputs.md` — `:86` done; `:504-505` + `:656-661` item 2 (E3a keeps ONE moment parameter,
  `evaluated_at`; `current_candle` renamed to its fill-horizon role or folded in); `:633` ("current_candle is a
  moment only on THRESHOLD" overruled — raw UPDATED is too; strike `:1446`); `:816` + `:888` row 19 (the fallback-POI
  post-E delta 13→12 / 125→124 / 153→151 is now 0 here); annotate items 3-6, 8-9. `PLAN_D_poi_activation_moment.md`
  `:257` (DONE marker), `:271-272` (same fallback numbers).
- **Skill values:** `.claude/skills/compare/SKILL.md:470` (`M15.counter` traces=153, shapes=125 → **151/124**) + its
  history paragraph `:473-490`.
- **Memory:** `MEMORY.md` (line 12 counts + tests; priority 1(a) → DONE), `project_sub_structure_pool_architecture.md`
  (`:81` counts/tests; `:149-153`), `project_zones_timing_audit_20260922.md` (queue item 1 → DONE; the fallback-POI
  item's live case gone), `project_item_3_poi_lifecycle.md:39,:110` (the identity); drain `_INBOX.md` (§10 of the
  inputs asks for it).

## 7. Out of scope (follow-ups, logged)

- **Re-ask when the gap forms** (user-accepted): FibTracker re-evaluates a failed decision at the candle a relevant
  gap's c3 closes → "dropped" becomes "one candle later"; also closes the M1 c3 instance.
- **The pre-existing MS/FibTracker activation divergence** (a gap forming after an H1 cycle-≥1 EST: MS inners include
  it, the one-shot fib never activates) — measure on this window; decide with the re-ask design.
- **Pattern-path CTS_UPDATED has no recorded moment** → no cut there (0 cells). Closed when the event carries its
  moment (Plan E).
- **One moment concept:** the pool's `knowable_at_idx` (EST → `ev.idx`, §17.12) and `unified_probe._second_cts_moment`
  (a `.get` fallback) converge onto `event_moment` in Plan E E3b.
- **The fill as-of on an anchor** (`check_to_idx` = CTS anchor at the EST / fib sites; retrospective IC
  identification) — Plan E E3.
- **Never-established-cycle fallback POI** — stays parked in general.
- Hygiene: the dead `_update_fib_cts` cross branch (`:1421-1457`) and its `.get` defaults (`:1433/:1442/:1524/
  :1532`); `get_unfilled_imbalances` / `has_imbalance_in_range` and the unused import at `fib_tracker.py:25`.
- Latent items (inputs synthesis §4): MS vs FibTracker cycle-0 snapshots can diverge on a REGRESS pattern-path update
  (closed 2026-09-27: MS emits a pattern-path update only on a strict new extreme);
  `inst.meta` mutated on instances shared across projections; the chart hover picks a POI by nearest inner price;
  single-candle stretches dropped (`sx1 <= sx0`).

## 8. Procedure

1. User go on rev 2 (incl. acknowledging M1 and the required-keyword hardening).
2. Tests first (§5): behaviour tests fail on HEAD; guard pins pass on HEAD and stay green; fixtures gain
   `confirmed_at` (EST) and `via` (CTS_UPDATED).
3. Implement §3; `pytest` (739 + new, 1 strict xfail).
4. Replay (`python -m engine_v2.run_replay > run.log 2>&1`), show the timing block, `/compare` in run mode with the
   fetch gate; check EVERY §4 cell, chart count and log line; anything else → STOP.
5. Landing cold review (diff vs this plan + docs).
6. Chart-review pause (H1 997→998 on the H1 chart and the M15 overlays; M15 4118→4119; counter sub 5 without the
   phantom "cycle 1" POI — the dropped fib shows only in the fib_lifecycle CSV).
7. Docs / skill values / memory (§6) in the same commit; `/commit-save` when the user says done.

## 9. As landed (2026-09-24) — every §4 prediction held

- **Tests first:** the 10 behaviour tests failed on the pre-Plan-F engine for the intended reason (sweep 6 vs 7, 8 vs
  9; IC-12 without the 19 deactivation; fibs / the cross existing; `_update_fib_cts` staying active); the 4
  behaviour-level guard pins passed on it. API-level tests (`formed_at`, `evaluated_at`) necessarily errored there.
- **Code as §3,** with two landing-review refinements: `_evaluating` resolves `event_moment` BEFORE touching state (a
  malformed event no longer wedges the tracker's scope flag — a real latent bug the review reproduced); the sweep's
  relevance filter and the POI debug dump both filter on `inst.formed_at <= scan_end` (behaviour-equivalent — an
  instance formed past `scan_end` was already skipped by the enter check). Log only: `evaluated_at=` on the "no
  unfilled imbalance" / DEACTIVATED prints.
- **`/compare` vs save `20260923_172626_0a4eadc`** (run mode, fetch gate PASS; replay 43.9 s): **all 24 CSVs
  byte-identical to the measured prediction** (19 identical to the baseline + the 5 §4 files cell for cell); charts H1
  85/245, counter **151/124**, confluence 294/233; log: counter sub-5 `total fibs` / `poi_zones total` 2→1, the
  `CROSS (0->1) v0 ACTIVATED … anchor_idx=235` line gone, the three decisions' `[fib]` lines as predicted; UNRESOLVED
  / probe-cache / parent-table lines unchanged. Re-verified identically after the landing-review edits.
- **Tests:** 777 passed + 1 strict xfail (739 + 38 new: `test_imbalance_c3_knowability.py` 30, `test_imbalance.py` 6,
  `test_cross_cycle_fib_routine.py` 2). The landing review MUTATION-tested the pins (each wrong variant — R2, a cut on
  `check_to_idx`, a cut on the pattern-path anchor, a cut cycle-0 cache, the uncut MS call sites cut — fails at least
  one test) and added the 11 pins for the sites a mutant had survived.
- **Chart review 2026-09-24:** user "charts look good".
- Cold reviews: plan `plan_f_inputs/cold_review_rev1.md`; landing `plan_f_inputs/landing_review.md`.

## 10. Cold review (2026-09-24) — what changed from rev 1

Must-fix: **M1** (the MS/FibTracker activation divergence was mis-stated as agreement → §2 consequence bullet, §5 pin,
§6 LANDMINES, §7), **M2** (`via` by direct index + fixture spec: 20 tests measured). Should-fix: S1 §4 provenance
(sweep+fib derived); S2 chart/log predictions (M15 fibs not rendered; the 151/124 source; the overlay + hover moves;
two `total=` log lines); S3 tests split into behaviour tests vs guard pins; S4 context manager restoring the previous
value, `event_moment` raises on unknown types; S5 `CTS_UPDATED_RAW_VIA` + public `event_moment` next to the emitter;
S6 `evaluated_at` keyword-only REQUIRED; S7 Plan E E3a collision annotated; S8 fallback-POI docs; S9 skill + memory
values. Nits folded: absent-`c0` form, `overlaps_formed_prefix`, index spaces in §4, pattern-path duplicates,
`_update_fib_cts` test, explainability prints, extra doc lines. Refuted: "−1 fib trace" (M15 fibs are not drawn).
Verified OK: the §3.2 filter == the measured harness; the sweep identity; 11 FibTracker call sites; only three
handlers reach a read; every production EST has `confirmed_at` and every CTS_UPDATED has `via`; index spaces
consistent (slice-local events + slice-local instances); the harness differences change 0 cells; the §4 cells and
151/124; the IC-12 prediction.
