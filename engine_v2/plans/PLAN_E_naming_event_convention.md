# Plan E — naming standard + event-index convention: `ev.idx` = the moment (zones pass, 2026-09-24)

**Status:** rev 2 — cold-reviewed (§11: 3 lenses, 1 BLOCKER + 15 MAJOR findings folded in). **All §10 questions
DECIDED by the user 2026-09-24** (Q11 wording approved as §4.2). **Progress (2026-09-25): E1–E4 LANDED** —
as-landed records §5.1 (E1/E1b), §6.7 (E2), §7.2 (E3), §8.1 (E4a / E4b-pre / E4b / E4c, each with its landing
review); `ev.idx` is the moment on every CTS / BOS event. **E5 COMPLETE (2026-09-26)** (§9.1): E5·1 renames,
E5·2 dead code, E5·3 prose + Q12, E5·4a/b the exported Q4 change — each with its landing review. Post-E: §9.
**Inputs (canonical site lists — read first):** [`PLAN_E_inputs.md`](PLAN_E_inputs.md) (the digest; §0.1 = the
user's FINAL decisions) + the raw inventories [`plan_e_inputs/`](plan_e_inputs/README.md). This plan does NOT
re-list every site: it schedules them, states each stage's mechanism and numeric prediction, and cites the inputs by
section (`IN §2.2 L1` = PLAN_E_inputs.md §2.2 row L1). Where this plan and the inputs disagree, this plan wins.
**Base:** HEAD `2d4d2b4` (Plan F `399b8f4` + its save commit). `/compare` baseline = save
**`20260924_121032_399b8f4`** (24 CSVs). Charts H1 85/245, M15.counter 151/124, M15.confluence 294/233. Tests
**777 passed + 1 strict xfail** (run pytest from the REPO ROOT; from `engine_v2/` the OANDA smoke test fails on
`oanda.cfg`).
**Measurement script:** `plan_e_inputs/measure_baseline.py <save_dir>` (re-derives every count in §3.1).

---

## 0. Coordinates (re-verified against HEAD `2d4d2b4`, 2026-09-24)

- `git diff --stat e2e0f89 HEAD -- '*.py'`: only **7 production files** moved since the inventories were taken:
  `common/types.py`, `patterns/imbalance.py`, `structure/market_structure.py`, `zones/cross_cycle_fib.py`,
  `zones/fib_tracker.py`, `zones/poi_zones.py`, `zones/structure_lifecycle.py`. **Every other production site in the
  inputs is exact at its `e2e0f89` coordinate.** The seven convert with IN §0's two offset tables (Plan D, then
  Plan F). Spot checks all held; the review re-confirmed every test-pin line in IN §2.7 at HEAD.
- Load-bearing sites, HEAD coordinates:

| Site | HEAD |
|---|---|
| `event_moment` + `CTS_UPDATED_RAW_VIA` | `market_structure.py:167`, `:170-192`; importers: `fib_tracker.py:27`, `tests/test_imbalance_c3_knowability.py:33` (grep again at E2) |
| `MarketStructureState.bos_confirmed` | `market_structure.py:212` |
| `REVERSAL_WATCH_START` / `REVERSAL_CANDIDATE` meta `anchor_idx` emits | `:828` / `:967` (`_schedule_reversal_from_anchor` `:923`) |
| `pending_reversal_anchor_idx` | `:38`, `:279`, `:920`, `:949`, `:954`, `:991`, `:2457`, `:2511-2512` |
| establish block / `CTS_ESTABLISHED` emit (meta `anchor_idx` `:1520`, `confirmed_at` `:1522`) | `:1491` / `:1514-1525` |
| `BOS_CONFIRMED` emits (cycle 0 / ≥ 1) | `:1531`, `:1546` |
| pattern-path `CTS_UPDATED` emit / "Update current CTS point (always)" | `:1580` / `:1583-1585` |
| Plan A assert (reads `.idx`; safe) | `:528` |
| `_emit_cts_established` / `_emit_bos_confirmed` (`st.bos_confirmed` set `:1966`) | `:1852` / `:1956` |
| POI-inner resolver + cycle0 snapshot (read `st.bos_confirmed`) | `:2024-2042`, `:2064-2070` |
| latent `return self.df, self.events, self.levels` | `:479` |
| FibTracker EST / UPD handlers `cts_idx = int(event.idx)` | `fib_tracker.py:600` (fill read `:616`) / `:1236` |
| `activated_at` default `.get("activated_at", cts_idx)` / `cycle1_bos_idx` `.get(..., cts_idx)` | `:1149` / `:1544`, `:1624` |
| cross `current_candle = int(event.idx)`; `select_fib_anchor_for_cycle` | `:2185`; `:180-194` (callers FibTracker `:972`, MS resolver via `poi_zones.py:1072`) |
| `resolve_cross_cycle_eligibility`: `current_candle` = window end AND fill horizon; `own_imb_start` = range start AND snapshot horizon | `cross_cycle_fib.py:130-135`, `:160` |
| POI `cts_events_by_key` sort / pre-window split / transition time / cond1 | `poi_zones.py:422` / `:856`, `:899-907` / `:917`, `:946`, `:964` / `:980` |
| POI `cts_established_idx` lookup (Plan D) | `poi_zones.py:487` |
| `compute_struct_start_by_sid` | `structure_lifecycle.py:151` |
| event sorts `(idx, type)` | `orchestrator.py:146`, `sub_wvmi.py:68`, `zone_proximity.py:205` |
| event sorts by `idx` | `wave_candles.py:124`, `:482`; `reference_zone.py:348-356` (REVERSE, `(idx, _TYPE_ORDER)`, CONFIRMED > UPDATED > EST) |
| zone_proximity timeline pointer (BOS `.idx` as a time) | `zone_proximity.py:193-205`, `:379`, `:403` |
| prev-BOS lines (filter walks CTS_EST AND CTS_UPDATED) | `orchestrator.py:266-298` |
| E1 readers of event meta `anchor_idx` | `wave_candles.py:295`, `:492` (read BEFORE its type guard `:493`; list includes CTS_UPDATED `:478-481`), `:553`; `export_plotly.py:1551`, `:1558` |
| KL debug prints (EST and BOS idx) | `kl_zones_v1.py:752-772` (EST print `:756-759`) |
| `_EVENT_META_IDX_KEYS` / `_ZONE_META_IDX_KEYS`; `_shift_meta_indices` (shifts Python `int` only) | `entity_df_mutation.py:103-112` (`:106` `anchor_idx`, `:108-109` `cts_anchor_idx`/`bos_anchor_idx`), `:116`; `:133-140` |
| mirror / sibling clip (`new_ev.idx = abs_idx`) | `entity_df_mutation.py:220-224` / `:478-489` |
| chart figure loader | `debug/chart_census.py:10` `load_fig` |

- The **KL-zone** meta `anchor_idx` (`orchestrator.py:124`, `wave_candles.py:403`) and the **wave-candle** result
  meta `anchor_idx` (`wave_candles.py:272`, `:444`) are MS-realm keys and STAY (IN §1.10).
- **Doc** line numbers in the inputs are stale. Every doc edit re-greps.

## 1. Goal — the end state

**The user's ideal (2026-09-23):** "CTS_ESTABLISHED.idx actually logs the moment while
CTS_CONFIRMED.meta['cts_anchor_idx'] logs the extreme." Generalised and decided in IN §0.1:

| Event | `ev.idx` after Plan E | `ev.price` after Plan E (Q19) | Anchor (location) | Pattern-realm anchor |
|---|---|---|---|---|
| `CTS_ESTABLISHED` | **the moment** (`confirmed_at` kept, `== idx` asserted) | the anchor's price (unchanged) | `meta["cts_anchor_idx"]` | `meta["pattern_anchor_idx"]` |
| `BOS_CONFIRMED` | **the moment** (`confirmed_at` kept, `== idx`) | the anchor's price (unchanged) | `meta["bos_anchor_idx"]` | — |
| `CTS_CONFIRMED` / `RECONFIRMED` | the moment (unchanged) | unchanged | `meta["cts_anchor_idx"]` (unchanged) | — |
| `CTS_UPDATED` raw path | the moment (== anchor, unchanged) | unchanged | `ev.idx` (+ meta if Q1 = flip) | — |
| `CTS_UPDATED` pattern path | E3·0 records the moment in `meta["confirmed_at"]`; `idx` → moment only if Q1 = flip (E4c) | unchanged | `ev.idx` until E4c | — |
| `REVERSAL_CANDIDATE` / `REVERSAL_WATCH_START` | unchanged | unchanged | — | `meta["pattern_anchor_idx"]` |

Plus: every production read of a CTS/BOS event's index goes through an accessor that names its role (anchor or
moment); no silent `.get(key, fallback)` on a migrated key; the event contract allows atomic migrations (§4.2);
renamed internals follow GLOSSARY "Naming Standard".

**Why it matters:** a retro-stamped `ev.idx` is a look-ahead trap for every timing reader. Plans D and F were two
instances of the class; IN §2.3 lists the rest. After Plan E a reader that uses `ev.idx` as a time is correct by
default, and a reader that wants a location must say so.

## 2. Construction principle (what makes each stage's prediction exact)

1. **E2 makes the flip a no-op for every reader.** E2 routes every production read of `CTS_ESTABLISHED` /
   `BOS_CONFIRMED` `.idx` through a named accessor. The **anchor** accessor reads the new meta key (== today's
   `ev.idx`). Every **time** half of a split call is written in E2 as the anchor accessor too, with a
   `# Plan E E3x → moment` marker naming the stage that will switch it (or `# stays: <decision>`, §7.1). E2 is
   byte-identical; afterwards only the declared raw readers (§6.3) read the raw `ev.idx`.
2. **E3 switches time halves** from the anchor accessor to `event_moment`, one cause per `/compare`, against the OLD
   emitted idx, so each delta is measured alone. **Every CTS/BOS event needs a recorded moment first:** E3·0 records
   one on pattern-path `CTS_UPDATED` (the review's BLOCKER: `event_moment` returns `None` there, and the prev-BOS
   filter, the sibling clip, the reference-zone window and FibTracker's `_on_cts_updated` all walk `CTS_UPDATED`).
3. **E4 then changes only what reads the raw `ev.idx` by design:** the events CSV `idx` column, and
   the KL debug prints (KL `source_event_idx` also reads it until E4b-pre deletes it, Q21). **E2 ends with an E4 variant replay** (emitters patched to pass
   `apply_idx`, not committed) that must diff from E2's save in exactly §8's cells, **including a figure-JSON diff of
   the 3 charts** (x/y arrays via `debug/chart_census.load_fig`; `/compare`'s count parity cannot see a BOS dot
   moving 2–411 candles). The variant is E2's completeness proof and is re-run after the last E3 stage.
4. **Sort keys are pinned in E2** to reproduce today's order exactly, so the flip reorders nothing. The one
   reverse-sorted recency pick (`reference_zone.py:348-356`) is a time value and moves in E3b, not a pinned order.

## 3. Stage map

| Stage | Cause (one per `/compare`) | Behaviour? | Predicted `/compare` vs the previous stage's save |
|---|---|---|---|
| **E1** | pattern-anchor rename + contract amendment + entity-absolute guard test | no (key text) | 5 CSVs change key text only (4 if the final.csv column were excluded); 19 identical (§4.5) |
| **E1b** | dead-code deletions (IN §1.8) + `self.levels` fix (Q17) | no | 24/24 identical; run.log loses `[uc1_trigger] lifecycle_end_idx=` lines |
| **E2a** | accessor module + emitters write `cts_anchor_idx`/`bos_anchor_idx` + `st.bos` decoupling + fixture factory + ALL test fixtures migrated + conftest validator | no (additive meta) | 4 CSVs gain meta keys; key-stripped diff empty; charts identical |
| **E2b** | CTS readers + CTS dual-role splits + helper parameter splits + sort pins | no | 24/24 identical; EST-only E4 variant = §8 E4a cells exactly |
| **E2c** | BOS readers + BOS dual-role splits | no | 24/24 identical; BOS-only E4 variant = §8 E4b cells exactly |
| **E2d** | fallback removals (cross-kind `.get`) + coupled renames | no | 24/24 identical; run.log `cts1_ext=` → `cts1_anchor=` |
| **E3·0** | pattern-path `CTS_UPDATED` records `meta["confirmed_at"]`; `event_moment` returns it | yes (Plan F cut turns on there) | events meta +37 keys; knowability-cut effect **measure first** (Plan F measured 0 cells) |
| **E3a** | FibTracker + MS-mirror fill horizons/time halves → moment (merged with the old "E3e", Q18) | yes | ≥ 4 fib_lifecycle cells + at-risk list (§7) — **measure first** |
| **E3a′** | *(Q8)* `cycle1_bos_idx` cond3 / c0 cond2 fill horizons → moment | yes | measure first (H1 BOS lag 14–76 makes H1 cells possible) |
| **E3b** | pool clock: `knowable_at_idx`, sibling clip time, reference-zone window + recency → moment | yes | 0 |
| **E3c** | probe Phase-2 `check_lo` → moment + 1 | yes | 0 |
| **E3d** | prev-BOS line filter → moment | yes | 0 (`sid=1 start=591 end=902`) |
| **E3f** | *(Q5 = moment)* `struct_start` base → first moment | yes | measure first (raw H1 sid 0 96→115; expected masked) |
| **E3g** | *(Q20)* remaining time halves (§7.1: POI transitions, zone_proximity pointer, chart PB bound) | yes | 0 predicted each; figure diff for the chart row |
| **E4a** | `CTS_ESTABLISHED.idx` → moment + its docs | contract | 3 events `idx` cells + 2 run.log lines (§8) |
| **E4b-pre** | delete the write-only KL meta `source_event_idx` (Q21) | no reader | KL meta loses the key on 39 rows (H1 10 / conf 21 / counter 8); nothing else |
| **E4b** | `BOS_CONFIRMED.idx` → moment + its docs | contract | 34 events `idx` cells + run.log KL prints |
| **E4c** | *(Q1 = flip)* pattern-path `CTS_UPDATED.idx` → moment | contract | 1 idx cell (conf sub 2: 2468 → 2470) |
| **E5** | remaining renames + prose | no / exported header | internal 24/24 identical; exported renames under a rename map |

Order constraints: E1 before E2 (removes the bare-`anchor_idx` collision on `CTS_ESTABLISHED`). E2a–d before every
E3. **E3·0 before every other E3** (they read `event_moment` on `CTS_UPDATED`). All E3 before E4 (user decision).
E4a/E4b in either order once E2's sort pins exist. E5 last.

**Session sizing:** E1 + E1b = Standard. **E2a–d = Heavy** (≈25 sites in ≈15 modules, the accessor module, 17 test
files ≈ 301 fixture events): recommend ultracode for that session, landing review with a mutation lens. E3 = Standard
(one variant replay per measured stage). E4 = Standard (docs-heavy). E5 = Light/Standard.

### 3.1 Baseline counts (measured on `20260924_121032_399b8f4` with `measure_baseline.py`; re-confirmed by review)

| Fact | H1 | conf | counter |
|---|---|---|---|
| events rows carrying meta `anchor_idx`: CTS_EST / RWS / RC | 5 / 1 / 1 | 21 / 7 / 4 | 8 / 0 / 0 |
| `CTS_ESTABLISHED` rows; `idx != confirmed_at` | 5; 0 | 21; 2 (1223→1224 sub 0 cyc 2; 2828→2829 sub 3 cyc 1) | 8; 1 (2828→2829 sub 3 cyc 1) |
| `BOS_CONFIRMED` rows; lag | 5; 5 (96→115, 591→652, 689→703, 728→748, 826→902) | 21; 21 (2–411) | 8; 8 (4–285) |
| `CTS_UPDATED` rows / pattern-path | 53 / 7 | 259 / 23 | 114 / 7 |
| KL rows by `source_kind` | BOS 5, CTS 5 | BOS 21 | BOS 8 |
| `structure_levels.csv` (H1) | 63 rows (CTS 58, BOS 5); 5 carry meta `anchor_idx` (levels meta = a copy of the event meta, `market_structure.py:2686-2707`) | — | — |
| fib_lifecycle E3a rows | — | line 14 (sub 3 cyc 0: `end_idx` 2828.0, `new_cycle`), line 15 (sub 3 cyc 1: meta `activated_at` 239, slice-local) | line 2, line 3 (same) |

Totals: E1 = **47** events rows (H1 7, conf 32, counter 8) + 5 levels rows. Only `:828`, `:967`, `:1520` emit the
key.

---

## 4. E1 — pattern-anchor rename + event-contract amendment

### 4.1 Code (one atomic commit)

| What | Where (HEAD) |
|---|---|
| Emit `pattern_anchor_idx` instead of `anchor_idx` on `REVERSAL_WATCH_START`, `REVERSAL_CANDIDATE`, `CTS_ESTABLISHED` | `market_structure.py:828`, `:967`, `:1520` |
| `CTS_ESTABLISHED`: drop the never-taken `else int(apply_idx)` fallback (a moment under an anchor name) → `int(ev.start_idx)` + `assert ev.start_idx is not None`; fix the stale comment "required for Scenario 2 Exception #2" (`:1519`) | `:1519-1520` |
| Readers → direct index `meta["pattern_anchor_idx"]`. **`wave_candles.py:492`: test `ev.type == "CTS_ESTABLISHED"` FIRST, then index** (the list holds `CTS_UPDATED` too; a naive direct index raises `KeyError`) | `wave_candles.py:295`, `:492-493`, `:553`; `export_plotly.py:1551`, `:1558` |
| Shift list: `"anchor_idx"` → `"pattern_anchor_idx"` in `_EVENT_META_IDX_KEYS` only | `entity_df_mutation.py:106` |
| `pending_reversal_anchor_idx` → `pending_reversal_pattern_anchor_idx` (**decided 2026-09-23**, memory audit "Renames confirmed"; final.csv column + state field). The `[RV_SCHEDULE] anchor=` / `[RV_APPLY] anchor=` log labels stay | `market_structure.py` rows in §0; `unified_probe.py:113`; `entity_df_mutation.py:95` |
| Pattern-realm internals: `TrueFirstBreakout.anchor_idx` → `pattern_anchor_idx`; `_price_confirmation(anchor_idx)` / `_price_confirmation_1step(anchor_end_idx)` → `pattern_end_idx`; wave_candles locals `first_bo_anchor`/`anchor`/`pattern_anchor` → `pattern_anchor_idx` | IN §1.3 rows 1–3 |

### 4.2 The contract amendment — DRAFT WORDING FOR THE USER'S APPROVAL (Q11)

`LANDMINES.md` "Event Contract Rules", rules 1–3 become (rules 4–5 unchanged; the "Planned amendment" note is
deleted):

> 1. **Never change event type names** — downstream consumers filter by exact string match.
> 2. **Never remove a field from `event.meta`.** Existing code may depend on it.
> 3. **Never rename an `event.meta` key, or change the meaning of an event field (`ev.idx`, `ev.price` or a meta
>    key), except by an atomic migration.**
>    - The migration is declared in its commit message together with its `/compare` prediction: the exact CSV cells
>      and run.log lines that may change.
>    - The **migration commit** — the one in which the emitted key or meaning changes — also moves every site that
>      reads **or writes** the field (production, debug, charts, test fixtures), every registry that lists it (e.g.
>      `_EVENT_META_IDX_KEYS`), and every current doc and skill that names it. Dated records (landed plans, save
>      folders, commit messages) are history and stay as written; memory is updated at the same checkpoint.
>    - The emitter sets the key on every event of that type (`None` allowed). Every `.get(key)`,
>      `.get(key, default)` and `key in meta` read of it becomes `meta[key]`. **No alias** is kept.
>    - Readers may be made independent of a meaning change in earlier commits (e.g. by routing them through an
>      accessor). The migration commit must then carry a proof that no reader still depends on the old meaning: a
>      variant replay or test whose diff is exactly the declared cells.
> 4. **Adding fields is OK** — document them in the relevant spec file.

(Today's rule 3 "Adding fields is OK" becomes rule 4; "append-only" and "must include structure_id" follow.)
`ARCHITECTURE.md` "Event contracts" gains one sentence pointing to rule 3. `PROJECT_PRINCIPLES.md` "## 2. Interfaces
Before Behavior" (`:22`) gains: "an event contract changes only by an atomic migration (LANDMINES "Event Contract
Rules" rule 3)". MEMORY "Critical Constraints" drops its PENDING note. Moving `event_moment` between modules (E2a)
is an API change, not an event-contract change.

### 4.3 Guard test (IN R5) — lands in E1, before any key moves

`tests/test_event_meta_idx_keys.py`, on the `geometry` fixture with `_two_lens_records` + `_render` from
`test_render_sub_projection.py` (0.1 s; it mirrors CTS_ESTABLISHED, REVERSAL_CANDIDATE and REVERSAL_WATCH_START).
Assert that every event meta key ending in `_idx` / `_at` on a mirrored sub event, plus an explicit extra list
(`pb_start`, which has neither suffix), is in `_EVENT_META_IDX_KEYS` or in an explicit `KNOWN_SLICE_LOCAL` allow-list:
`confirm_idx`, `cts_idx`, `pullback_apply_idx`, `start_idx`, `expires_idx`, `effective_idx`, `pb_start`. Also
assert that every value under a listed key is a Python `int` or `None` (`_shift_meta_indices` shifts only `int`). Zone
meta is checked the same way against `_ZONE_META_IDX_KEYS`. Mutation check: remove `"pattern_anchor_idx"` from the
list, and the test fails. E2a extends it: `meta["cts_anchor_idx"] == ev.idx` (pre-E4) and both new keys are `int`.

### 4.4 Tests + docs (same commit)

- **Tests.** Event fixtures carrying the key, both as dict literals AND as keyword arguments:
  - `test_structure_lifecycle_moment.py:47`, `:64`;
  - `test_parent_tables.py:52`, `:69`;
  - `test_lifecycle_sweep_predicted_table.py:98`, `102`, `107`, `109`, `115`, `119`, `123`;
  - the event fixtures in `test_wave_candles.py` (e.g. `:390`, `:434`);
  - `test_true_first_breakout.py` (`anchor_idx`).

  Separate the realms by hand. The KL-zone / wave-candle `anchor_idx` fixtures stay: `test_wave_candles.py:58`,
  `:163`, `:275`…; `test_sub_wvmi.py:60`; `test_wvmi.py:54`; `test_render_sub_projection.py:705-710`. New: a
  direct-index pin (a `CTS_ESTABLISHED` without `pattern_anchor_idx` raises in `wave_candles`); the guard test.
- **Grep gate:** `rg -n '\banchor_idx\b' engine_v2 --glob '!legacy_2025*' --glob '!plans/**'`. Classify every hit
  by hand; zero may remain in the event-meta realm (KL-zone, wave-candle, fib and base-pattern hits stay).
- **Docs (re-grep all):**
  - the amendment (§4.2);
  - ARCHITECTURE: the "`ev.idx` convention" table (`:104-105`), "Anchor has two realms", "Bound and frequency"
    (`:120-121`);
  - GLOSSARY: `anchor_idx` (`:82`), the "Bare element idx" row (+ `RANGE_STARTED.meta["cts_idx"]`);
  - GOTCHAS `:588`, `:603-617` (snippet with `.get("anchor_idx")` at `:608`), `:1841-1842`;
  - LANDMINES `:1794`;
  - PART4_REFACTOR_SPEC `:1126`, `:3034`;
  - PRE_REFACTOR_INVARIANTS `:77`, `:92-102`;
  - KL_ZONES_SPEC `:36-39`, `:105-106` (only where the EVENT key is meant);
  - WAVE_CANDLES_SPEC `:106-107` ("if it has anchor_idx in meta" → unconditional);
  - the `structure_lifecycle.py:73-74` docstring;
  - memory: MEMORY "Key Architecture Points", `project_zones_timing_audit_20260922.md`,
    `project_cycle_lifecycle_parent_cycle_floor.md` (`:26`, `:67-68`, `:119`).
  - Optional (Q17): the frozen-bridge rows missing from the ARCHITECTURE table (IN §1.9 last bullet).

### 4.5 Prediction (vs `20260924_121032_399b8f4`)

- Events CSVs: meta **key text** `anchor_idx` → `pattern_anchor_idx` on exactly 47 rows (H1 7, conf 32, counter 8),
  values identical. `structure_levels.csv`: the same on 5 meta rows. final.csv: 1 header cell.
- 19 other CSVs byte-identical. Reverse substitution → **24/24 byte-identical**. WVMI byte-identical proves the
  `wave_candles` readers moved.
- Charts identical (85/245, 151/124, 294/233). run.log identical except timings.
- Tests: 777 + new, 1 xfail.

## 5. E1b — deletions + the latent `self.levels` fix (Q17 = before E2)

- **Deletions:** IN §1.8 rows 1–8:
  - the retired `lifecycle_end_idx` chain;
  - `validated_h1_start` / `validated_parent_start`;
  - `MultiTFTrigger.start_time` / `start_price`;
  - `proximity_confirmed_idx`;
  - `_is_new_cts_extreme` + `_cts_price_at` + their commented callers;
  - the unread `probe_end_idx` meta keys;
  - uc1 meta `cts_idx`;
  - `debug/zone_proximity_diag.py`;
  - `common/types.Zone`.

  Plus the write-only `FibRetracement.anchor_high_idx/anchor_low_idx` if Q15 = delete.
- **Latent bug:** `market_structure.py:479` `self.levels` (IN §3 #1; kept out of E1's atomic migration).
- **Why before E2:** E2's migration would otherwise have to carry dead `.idx` reads (`cts_est_idx_by_key` reads
  `CTS_ESTABLISHED.idx` in 4 trigger modules).
- **Prediction:** 24/24 byte-identical. run.log loses the `[uc1_trigger] lifecycle_end_idx=` lines and nothing else.
  Tests lose the pins in IN §1.8.

### 5.1 E1 + E1b as landed (2026-09-24)

- **E1 = `5facfe0`.** `/compare` measured == §4.5 exactly (47 + 5 + 1 key-text cells; 24/24 after reverse
  substitution; the 3 chart figures JSON-identical to the baseline). Tests 789 + 1 xfail. Deviations / additions:
  the guard needed a ZONE allow-list too (`bos_idx`, `cts_idx`, `source_event_idx` — slice-local KL meta); the
  landing review's mutation lens found no test pinned the key's VALUE, so a pattern-bound pin (guard) + an exact
  pin (`test_unified_probe`: OMO c0 9, close-break 11) were added; 2 GOTCHAS lines missing from §4.4 were fixed.
  Not covered by the guard (→ E2a): `WaveCandleResult.meta["anchor_idx"]` (exported slice-local),
  `RANGE_STARTED.meta["proximity_apply_idx"]`.
- **E1b.** `/compare` vs E1: 24/24 byte-identical, figures identical; run.log: the 3 `[uc1_trigger]` lines lose
  `start_time=… lifecycle_end_idx=…` (both fields deleted). Tests 783 + 1 xfail (−7 retired-chain pins, +1
  `self.levels` pin). Deviations: (1) `debug/zone_proximity_diag.py` KEPT — the `/compare` skill ("Per-Cycle
  Proximity Trigger Counts") and GOTCHAS use it as the per-cycle proximity tool; (2) `Subsequent*Trigger.end_idx`
  untouched in E1b — after the unread meta `probe_end_idx` went it is unread and == `trigger_event_idx`; the sibling
  probe's end is the sweep's `hi`, so the §1.8 rename to `probe_end_idx` would misname it. **User 2026-09-24: keep
  the diag file; delete `end_idx`** → deleted in its own commit after E1b (24/24 CSVs + figures + run.log identical);
  (3) the fib anchor-idx deletion reached 7 call sites (fib_tracker ×6, poi_zones ×1), not only `fibonacci.py`.

---

## 6. E2 — explicit anchor fields (four byte-identical commits; the bulk)

### 6.1 E2a — emitters, MS state, accessor module, fixtures

- **Emitters (`market_structure.py`):**
  - `_emit_cts_established(idx, price, *, cts_anchor_idx, meta)` writes `meta["cts_anchor_idx"] =
    int(cts_anchor_idx)`.
  - `_emit_bos_confirmed(idx, price, *, bos_anchor_idx, meta)` writes `meta["bos_anchor_idx"] = int(...)`, and
    builds **`st.bos = Point(bos_anchor_idx, price)`** + `bos_threshold` from the PARAM, not from `idx` (IN §2.1:
    the MS state decoupling). Rename `st.bos_confirmed` → `st.bos` (def `:212`; set `:1966`; reads `:2024-2042`,
    `:2064-2070`, `:2531-2532`, the `:1623` comment).
  - `int()`-wrap both values: `_shift_meta_indices` shifts only Python `int`, and the degenerate branch at `:1777`
    passes the caller's type through.
  - Both keys are already in `_EVENT_META_IDX_KEYS` (`:108-109`); the M15 event paths (mirror `:220-224`, sibling
    clip `:486-488`) shift through that list, and no other export copies event meta.
- **MS state stays on the anchor** (IN §2.1, R8): `st.cts`, `_initial_bos_before_first_cts(cts_anchor_idx)`,
  `range_start_idx`, the sd-prox gate `i > st.cts.idx`, the POI-inner resolver, `cycle0_data`, the Plan A assert
  `:528`. Flag the dormant `_select_bos_on_breakout` fallback that passes a moment into
  `_initial_bos_before_first_cts` (IN §3 #11) with a comment.
- **Accessor module `structure/event_fields.py`** (new). Callers import the MODULE and call it qualified —
  `from engine_v2.structure import event_fields as ef`; `ef.cts_anchor_idx(ev)`. A test bans
  `from ...event_fields import cts_anchor_idx`: a direct import would collide with the many locals and params named
  `cts_anchor_idx` / `bos_anchor_idx` / `pattern_anchor_idx` (e.g. `fib_tracker.py:1723`; IN §1.5) and raise
  `UnboundLocalError` (`cts_anchor_idx = cts_anchor_idx(ev)`). Q16 decides the names.

  ```python
  CTS_UPDATED_RAW_VIA = "replay_raw"      # MOVED here; market_structure imports it from here
  def cts_anchor_idx(ev) -> int           # CTS_EST / CTS_CONFIRMED / CTS_RECONFIRMED: meta["cts_anchor_idx"];
                                          # CTS_UPDATED: ev.idx (until E4c); other types raise
  def bos_anchor_idx(ev) -> int           # BOS_CONFIRMED: meta["bos_anchor_idx"]; other types raise
  def pattern_anchor_idx(ev) -> int       # CTS_EST / REVERSAL_CANDIDATE / REVERSAL_WATCH_START
  def event_moment(ev) -> Optional[int]   # MOVED from market_structure (Plan F), EXTENDED to BOS_CONFIRMED
                                          # (confirmed_at) and CTS_CONFIRMED/RECONFIRMED (ev.idx). This reverses
                                          # Plan F's "any other type raises" for those three types — deliberate
  def processing_order_key(ev)            # (_location_idx(ev), ev.type) — today's order, frozen (§6.2)
  def _location_idx(ev) -> int            # PRIVATE: the anchor for CTS_EST / BOS / CTS_UPDATED, ev.idx otherwise
  ```

  - Import direction: `event_fields` imports nothing from `market_structure` (types come from `common/types`), so
    no cycle.
  - `event_moment` is moved, not re-exported. Update every importer (grep: `fib_tracker.py:27`,
    `test_imbalance_c3_knowability.py:33`) and invert the pin `test_imbalance_c3_knowability.py:331-332`
    (`event_moment(CTS_CONFIRMED)` raising).
- **Fixture factory `tests/_event_factory.py`**, keyword-only, with parameter names identical to the meta keys:
  `make_cts_established(*, cts_anchor_idx, confirmed_at, pattern_anchor_idx, price, cycle_id, structure_id, …)`,
  `make_bos_confirmed(*, bos_anchor_idx, confirmed_at, …)`. `idx` defaults to today's value (the anchor); E4 flips
  the default.
- **All fixtures migrate in E2a, not "on first touch":** the census has 17 files, 23 construction sites and ≈ 301
  events. None carries the new keys, and 10 files build CTS_EST/BOS without `confirmed_at`. §6.2 routes every
  CTS_EST/BOS through `processing_order_key`, which direct-indexes the anchor keys. **`tests/conftest.py`
  validator:** every test-built `CTS_ESTABLISHED` / `BOS_CONFIRMED` carries `confirmed_at` and its anchor key (from
  E4: `idx == confirmed_at`).
- Re-point the 7 LOCATION pins to meta (IN §2.7 + `test_poi_activation_moment.py:87`) and re-author the
  contract-illegal Plan D fixtures (IN §2.7).
- Measured by the review (emitter patched on HEAD, readers untouched): an EST-only flip fails 3 tests and a BOS-only
  flip fails 15 (13 in `test_poi_activation_moment`). The suite alone cannot prove E2; hence §6.4.

### 6.2 E2b — CTS readers, splits, sort pins

- **CTS LOCATION** L1–L10 (IN §2.2) → `ef.cts_anchor_idx(ev)`:
  - L1 through the renamed `_cts_anchor_idx_for_event`;
  - L5 `StructureLevel.time` re-sourced from the accessors;
  - L6–L8 chart dots;
  - L9 prev-BOS end (Q6);
  - L10 direct index.
- **CTS dual-role calls** (IN §2.4 items 1–8, 10–13): split into an anchor local and a moment local. In E2 the
  moment local = the anchor accessor + a `# Plan E E3x → moment` marker. Name the moment local with a moment
  spelling (`cts_established_idx`), or reuse `self._evaluated_at` where the handler already holds the moment
  (FibTracker, Plan F). Never `*_time_idx`: `_time` means a timestamp in this codebase.
- **Helper parameter splits (review MAJOR).** Both helpers get explicit separate parameters, each passed today's
  value in E2, so that E3a moves only the horizon:
  - `resolve_cross_cycle_eligibility`: `current_candle` → `own_window_end_idx` (location) + `fill_horizon_idx`
    (time); `own_imb_start` → range start + `snapshot_horizon_idx` (`cross_cycle_fib.py:130-135`, `:160`);
  - `select_fib_anchor_for_cycle` (`fib_tracker.py:180-194`), shared by FibTracker `:972` and the MS resolver
    (`poi_zones.py:1072`).
- **Sort pins.** `orchestrator.py:146`, `sub_wvmi.py:68`, `zone_proximity.py:205`, `poi_zones.py:422`, `:856`,
  `wave_candles.py:124`, `:482` → `ef.processing_order_key(ev)` (or its idx component where the key is idx-only
  today).
  - The review confirmed it reproduces today's order for every type, on both `CTS_UPDATED` paths, before and after
    E4. H1–H3 stay as safe as today and H4 is removed by the splits.
  - **`reference_zone.py:348-356` is NOT pinned there:** it is a reverse recency pick with its own
    `_TYPE_ORDER` (CONFIRMED > UPDATED > EST). E2b writes it as `(ef.cts_anchor_idx(e)  # Plan E E3b → moment,
    _TYPE_ORDER[e.type])`, plus a tie test.
- **The mirror and sibling clip** keep the raw index: the mirror `:222` and `new_ev.idx` in the sibling clip
  (`:482-484`) become `ev.idx + slice_begin`, split from the clip TIME local (E3b moves only the time).
  Otherwise E3b would silently rewrite a mirrored `CTS_ESTABLISHED.idx` before E4.

### 6.3 E2c — BOS readers; declared raw readers

- **BOS LOCATION** B1–B12 (IN §2.2):
  - B1 = §6.1.
  - B2: KL → `ef.bos_anchor_idx(ev)` for the zone base; `source_event_idx` per Q21.
  - B4: orchestrator `bos_by_cycle`.
  - B5: FC `input_idx` (**a pool key**).
  - B6: prev-BOS start.
  - B7: chart dots.
  - B9 `compute_struct_start_by_sid` / B10 `creation_event_idx` → the **explicit** accessors (`ef.bos_anchor_idx`
    for BOS_0, `ev.idx` for the rest), pinned to today's values until Q5.
  - B12: KL debug prints → print idx + meta anchor for **both** EST (`:756-759`) and BOS.
- **Declared raw-`ev.idx` readers after E2:**
  - the events CSV writer (`debug/export_events.py`);
  - the mirror + sibling-clip `new_ev.idx`;
  - KL `source_event_idx` (until E4b-pre deletes it, Q21; so the E2c BOS-only variant also shows its 34 BOS-zone
    cells: H1 96→115, 591→652, 689→703, 728→748, 826→902 + 29 slice-local M15);
  - the accessor module;
  - the Plan A assert (`market_structure.py:528`).

  Every other raw read is a miss. The gate is **dynamic** (§6.4); a static grep is a reviewer aid only.

### 6.4 E2d + proofs

- **E2d:**
  - the "moment → anchor" fallbacks (IN §3 #3: 5 sites) → `ef.event_moment(ev)`;
  - the remaining 11 cross-kind fallbacks + `unified_probe.py:605` → direct index / accessor. Check fixtures that
    omit keys first (`test_zone_proximity` `_bos_confirmed(confirmed_at=None)`);
  - the coupled renames: IN §1.4 and §1.5 rows marked E2 (`_cts_anchor_idx_for_event`, `parent_bos_anchor_idx`,
    the `cts1_ext=` → `cts1_anchor=` label, …);
  - delete or invert `test_unified_probe.py:847-853`; re-point the label pin `:782`.
- **E4-simulation test (unit):** `_run_downstream_pipeline` (`orchestrator.py:54`) on
  `_make_second_cts_moment_after_extreme_data` (EST (9, 10); lagging BOS (0, 2), (7, 10)) AND `_make_multicycle_data`
  (3 lagging EST, 4 lagging BOS), in both `h1` and `cross_cycle` modes. Clone the events with EST/BOS
  `idx := confirmed_at` and assert that the outputs are identical except the declared raw readers. The comparator
  excludes the `fib_tracker` repr (object address) and `sorted_events`. It fails at HEAD in both modes (it bites)
  and must pass after E2c.
- **E4 variant replays** (not committed): after E2b, EST-only = §8 E4a exactly; after E2c, BOS-only = §8 E4b
  exactly; each includes the 3-chart figure-JSON x/y diff. After E2c the BOS dots and the struct-start readers
  (charts, triggers, pool, parent tables, the `structure_engine` handoff) are covered, where the unit test cannot
  reach.
- **Pool-key fixtures (IN R1):** L1 (EST-winner reference with lag > 0; invisible on the window) and B5 (lagging BOS
  → `first_confluence_trigger` `input_idx`) pin the anchor now and after E4.
- **`st.bos` decoupling pin:** `_emit_bos_confirmed(idx=X, bos_anchor_idx=Y)` → `st.bos.idx == Y`.
- **Sort-pin tests:** H1–H3 on `processing_order_key`; H5 on the reference-zone tie.

### 6.5 E2 predictions (each vs the previous commit's replay)

- **E2a:** events meta **gains** `cts_anchor_idx` on 34 `CTS_ESTABLISHED` rows (5/21/8) and `bos_anchor_idx` on 34
  `BOS_CONFIRMED` rows (5/21/8), each equal to the row's `idx`. `structure_levels.csv` meta gains 5 + 5. **A
  meta-key-stripped diff of all 24 CSVs is empty**; 20 CSVs byte-identical; charts identical.
- **E2b, E2c:** 24/24 byte-identical; figures identical; the variant replays as above.
- **E2d:** 24/24 identical; run.log `cts1_ext=` → `cts1_anchor=` (1 line, values identical); the `[kl_zones]` debug
  prints gain the anchor (B12).

### 6.6 E2 docs (review MAJOR: rev 1 listed none)

- the ARCHITECTURE table: the new meta keys (rule 4 "document them");
- the `event_moment` move + extension: ARCHITECTURE `:113-118`, `:141`; GLOSSARY `:68`; IMBALANCE_FILL_SEMANTICS
  `:67`, `:211`; LANDMINES `:216`, `:1591`; GOTCHAS `:697`, `:784`; PRE_REFACTOR_INVARIANTS `:79`;
  CROSS_CYCLE_FIB_SPEC `:160`; FIB_LIFECYCLE_SPEC `:375`; POI_ZONES_SPEC `:97`, `:172`; memory
  `reference_key_files.md:22`;
- the sort invariant (these docs state `(e.idx, e.type)`): LANDMINES `:578-602`, PRE_REFACTOR_INVARIANTS `:86`,
  GOTCHAS `:989`;
- `st.bos`: MARKET_STRUCTURE_SPEC `:23`, GOTCHAS `:1682`;
- `.claude/skills/compare/SKILL.md:239-250`: add the figure-JSON diff step for Plan E stages and track the anchor
  keys from E2 on.

### 6.7 E2 as landed (2026-09-24)

- **E2a.** `/compare` vs `20260924_171755_31f51ee` == §6.5 exactly: 4 CSVs change and only by key additions —
  H1 events 10 cells, `structure_levels` 10, conf events 42, counter 16 (EST `cts_anchor_idx` 5/21/8, BOS
  `bos_anchor_idx` 5/21/8, each `== idx`); a meta-key-stripped diff of all 24 is empty; the 3 figures JSON-identical;
  run.log differs only by a warning's line number and the parked `by_lens` set order. Tests 783 → 805 + 1 xfail.
  Deviations / additions:
  - the `tests/conftest.py` validator raises `EventContractViolation(BaseException)`, not `AssertionError`: with an
    AssertionError one `pytest.raises((KeyError, AssertionError))` test went vacuous (it no longer reached the
    production leaf), and engine `except Exception` skip paths could swallow a violation. It checks the full
    pre-E4 contract (`idx == anchor`, both keys Python `int`), not only presence; E4 flips `EVENT_IDX_IS`.
  - factory: besides the typed `make_cts_established` / `make_bos_confirmed`, a generic `make_event(etype, idx,
    **meta)` (idx = the anchor for CTS_EST / BOS) so the per-file `_ev` helpers route through the factory without
    rewriting ≈300 call sites; an attribution argument set to `None` omits its key.
  - the E1 guard gaps: `proximity_apply_idx` allow-listed + a static AST scan of every event-meta key MS writes;
    wave-candle meta `anchor_idx` allow-listed (checked on the mirrored fixture). Nothing shifted.
  - NOT re-authored (legal until E4, E4's job): the Plan D `mutate=` hooks (`test_poi_activation_moment.py:138`,
    `test_render_sub_projection.py:789`) rewrite `confirmed_at` on engine-built events — the factory cannot build
    them; E4 must move `idx` with `confirmed_at` there.
  - landing review (2 lenses, ≈241k): conformance found 0 BLOCKER / 0 MAJOR (MINORs folded in: 2 missed idx pins
    in `test_unified_probe`, strict-int validator, `e4flip_plugin.py` updated to the keyword-only emitters,
    validator limits documented); mutation lens killed 23/26 — the 3 survivors were the validator's own checks →
    `tests/test_event_contract_validator.py`.
  - **Plan correction for E2b:** §6.2's reference-zone recency key `ef.cts_anchor_idx(e)` is wrong for a
    CTS_CONFIRMED (it returns the anchor; today's key is its `ev.idx`, the confirmation candle). Mixed-type time
    halves need "today's `ev.idx`" (`_location_idx`) — see E2b.
- **E2b.** **User decision 2026-09-24:** the planned private `_location_idx` is public as **`ef.stamped_idx(ev)`**
  ("the index `ev.idx` holds today, frozen against E4" — neither a location nor a moment); used only by
  `processing_order_key` / the sort pins and by the E2 time halves over mixed CTS types, each with its
  `# Plan E E3x → moment` marker. `/compare` vs E2a: 24/24 CSVs byte-identical, figures identical, run.log only
  the parked `by_lens` order. **EST-only E4 variant** (`e4flip_plugin.py`, `FLIP=est`) vs E2b == §8 E4a exactly:
  3 events `idx` cells (conf 1223→1224, 2828→2829; counter 2828→2829), run.log the 2 `[kl_zones]` lines
  (819→820, 239→240), figures JSON-identical. Sites:
  - FibTracker EST: `cts_idx` (anchor) + `cts_established_idx` (time, `# Plan E E3a`) threaded through the three
    EST handlers; time uses = `activated_at`, the EST fill horizon, the uncut cycle-0 cache horizon (lock-step
    with the MS mirror), Scenario 1 (T5), the revert terminal; helper splits — `resolve_cross_cycle_eligibility`
    (`current_candle` → `own_window_end_idx` + kw `fill_horizon_idx`; kw `snapshot_horizon_idx`),
    `select_fib_anchor_for_cycle` (kw `fill_horizon_idx` / `snapshot_horizon_idx`, marked E3a / E3a′ at both
    callers incl. the MS resolver), `_m15_cross_check` (kw `own_window_end_idx`), `_maybe_activate_main_cross`,
    `_run_main_cross_check`.
  - reference_zone L1 (`_extreme_idx_for_cts_event` deleted → `ef.cts_anchor_idx`), window + recency
    (`ef.stamped_idx`, E3b); sibling clip time (E3b) split from the mirror's raw `new_ev.idx`; POI sorts, pre-window
    split + transition time (`ef.stamped_idx`, E3g-1) vs cond1 (`ef.cts_anchor_idx`); wave candles L4 (location
    walk); `structure_levels` CTS time (L5); chart dots L6–L8; prev-BOS END (anchor, Q6) vs filter (E3d);
    unified_probe `check_lo` (E3c) + the CTS_EST sort; the Q10 test-only structure_engine paths; the debug
    `probe_fc_finalize` reads; sort pins `orchestrator` / `sub_wvmi` / `zone_proximity` → `processing_order_key`.
  - **Deviation:** CTS_UPDATED — only its LOCATION read is routed (`ef.cts_anchor_idx`, for E4c); its time halves
    (the update handlers' activation stamps / horizons, ~6 helpers) are NOT split in E2 — the raw path's idx is
    its moment already, and the pattern path's moment only exists from E3·0; E3·0/E3a own that split.
  - Tests: `tests/test_event_order_pins.py` (H1 in all four flip orders, H2, H3, H5 tie, L1 EST-winner pool key,
    the stamped window filter; each in today's AND the E4 shape; 3 fail under a raw `(idx, type)` key).
    806 → 820 + 1 xfail.
  - Landing review — role/completeness lens (≈184k): 0 BLOCKER / 0 MAJOR. Folded in: **plan contradiction
    resolved** — §7 lists the c0 cond2 fill horizon under both E3a (`_update_cycle0_data`) and E3a′; per Q8 it is
    **E3a′** in BOTH layers (FibTracker's uncut cycle-0 cache `c0_fill_horizon_idx` and MS `_update_cycle0_data`,
    each `# Plan E E3a′`); `pooled_structure_build` `knowable_at_idx` input → `ef.stamped_idx` (E3b); the Phase-2
    `cts0_est=` label → accessor (E3c re-sources it); per-site `UPD time half: E3·0/E3a` markers on the update
    path; snapshot-horizon check `assert` → `ValueError`. **E2c inherits:** zone_proximity's threshold timeline is
    now sorted by `processing_order_key` (BOS anchor) but its pointer walk (`:380`, `:404`) compares raw `.idx` —
    after E4b the walk would stop at a BOS stamped at its moment (T3) → E2c must walk on `ef.stamped_idx`.
  - Mutation lens (≈118k): 5/30 killed — the splits had no unit pin (the replay + the EST variant were their only
    proof). Added: `tests/test_e4_simulation.py` (§6.4's unit E4 simulation, pulled forward: EST flip on
    `_make_second_cts_moment_after_extreme_data` in both fib modes passes; the BOS / both flips are strict xfails
    until E2c; bites — reverting FibTracker's EST read to raw `ev.idx` fails it. **Correction:**
    `_make_multicycle_data` has 4 lagging BOS and NO lagging EST, not "3 lagging EST"); `tests/test_plan_e_role_pins.py`
    (FibTracker EST location vs time in both modes; the routine's window vs horizon; the anchor selector's cond3
    at the snapshot horizon; POI cond1 = the anchor + a positive control; an AST guard: no sort/max key in the
    pinned modules reads `.idx`). Re-mutated: M01 (fib location → moment), M09 (raw orchestrator sort), M23 (POI
    pre-window cond1 → moment) now killed. Known unkillable before E3: the in-window POI cond1 read (only an
    UPDATED reaches it, where anchor == idx) and the prev-BOS END (inside `_run_downstream_pipeline`, no reversal
    in the fixtures). Suite 829 passed + 9 xfailed (1 + 8 E2c-pending); replay after the fixes == E2b's.
- **E2c.** Sites: KL BOS zone base (`anchor_idx` ← `ef.bos_anchor_idx`; `source_event_idx` stays raw, declared);
  B12 `[kl_zones]` prints = (raw idx, anchor, sid, cycle) for EST and BOS, the dead `bos_prev` read dropped;
  orchestrator `bos_by_cycle` (B4) + prev-BOS START (B6); FC `input_idx` (B5, pool key); `structure_levels` BOS
  time (B8); `compute_struct_start_by_sid` / `SidRecord.creation_event_idx` (B9/B10) on `ef.stamped_idx`
  `# Plan E E3f`; zone_proximity's timeline pointer (T3) on `ef.stamped_idx` `# Plan E E3g-2` (the E2b coupling);
  chart BOS dots / PB→BOS ends (location) vs the PB-search bound (`# Plan E E3g-3`, T4) in all three charts;
  FibTracker cond3 horizon (`cycle1_bos_idx`) `# Plan E E3a′`. `/compare` vs E2b: 24/24 CSVs + figures identical;
  run.log = the `[kl_zones]` prints only (declared B12; §6.5 had put them under E2d). **BOS-only variant** vs E2c
  == §8 E4b + §6.3: 34 events `idx` (H1 96→115, 591→652, 689→703, 728→748, 826→902; conf 21; counter 8) + 34 KL
  `source_event_idx` (key-only), figures identical, run.log the BOS prints' raw idx. **The first BOS variant run
  caught a miss** — `structure_levels` BOS `time` (B8; E2b had migrated only the CTS half of L5): 5 cells —
  fixed + pinned (`test_structure_levels_are_timed_at_the_anchors`). `test_e4_simulation`'s BOS / both flips
  now pass (the 8 strict xfails removed); FC input + struct_start/creation pins added. The Plan-B-save CSV test
  translates legacy rows (pre-E2a: `idx` IS the anchor). Tests 840 passed + 1 xfail.
  - Landing review (≈230k: role 148k + mutation 81k): 0 BLOCKER / 0 MAJOR. Folded in: the prev-BOS block is a
    pure helper `orchestrator._prev_bos_lines` (byte-identical) + pin `test_prev_bos_line_runs_anchor_to_anchor`
    (no fixture had a reversal, so START (E2c) and END (E2b) were unpinned; 3 mutants now killed); the debug
    probe script's leak check declared a raw reader; docs (WAVE_CANDLES_SPEC, GOTCHAS ×2, KL docstring, CHARTING_SPEC,
    struct_start / creation / pool docstrings, `stamped_idx` docstring); E3f / E3g notes in §7 (the all-type
    minima and the BOS_THRESHOLD timeline hold types `event_moment` does not define; E3g-2 must re-sort). Mutation
    11/27 killed; survivors: 9 chart-only (covered by the variant figure diff — accepted), the zone_proximity
    pointer (equivalent on every contract-legal stream: a BOS moment == its cycle's EST moment <= the
    CTS_CONFIRMED candle = `scan_start`), and the pre-existing FibTracker cond3 call sites (`:1582`, `:1661`) —
    E3a′ changes that value and must pin it there. Tests 841 passed + 1 xfail; replay after the fixes == E2c's.
- **E2d.** Fallbacks → direct: moment ← anchor (`ef.event_moment`): FC `trigger_event_idx`, zone_proximity
  `bos_conf_idx_by_key` + `scan_start`, unified_probe `_second_cts_moment` + both `cts0_est_idx`, KL
  `confirmed_idx`, the debug `cts1_moment`; anchor ← CTS_CONFIRMED moment (`ef.cts_anchor_idx`): the 3 chart
  CTS dots, uc1 `cts_idx`, FibTracker `on_cts_confirmed`, KL CTS base, unified_probe `cts0_anchor_idx`;
  FibTracker-internal `c0.get(...)` / `meta.get("cycle1_bos_idx", cts_idx)` → direct (the keys exist on every
  path that reaches them; the versioned §11b crosses lack `cycle1_bos_idx` but never reach those branches).
  Renames: `cts1_ext=` → `cts1_anchor=` (unified_probe label, value now the accessor — the E2b miss the E2c
  review found; the debug script), `parent_extreme_*` → `parent_bos_anchor_*`, orchestrator `bos_anchor_by_cycle`
  / `last_bos_anchor_by_sid`, reference_zone `cts_anchor_idx` (+ `_derive_cts_zone_ad_hoc` param), POI
  `cts_anchor_idx_at_t`, MS establish-block locals + `_initial_bos_before_first_cts(cts_anchor_idx)`,
  `compute_bos_inner_from_event(bos_anchor_idx)`, structure_engine Q10 locals `cts0_anchor_idx`, test
  `bos_anchor_abs`. **Not renamed (deviations):** the charts' `next_bos_idx` — since §7.1 T4 it is the PB-search
  TIME bound (E3g-3 → moment), so `next_bos_anchor_idx` would be wrong after E3g-3; `FibTracker.on_cts_established
  (bos_idx)` → E5 (the fib-family `bos_idx` vocabulary + test call sites). Tests: the fallback pin inverted
  (`test_raises_without_the_meta_key`), the label regex re-pointed. `/compare` vs E2c: 24/24 + figures identical,
  run.log exactly 1 line (`cts1_ext=1020` → `cts1_anchor=1020`).
- **E2 completeness proof (the full E4 variant, `FLIP=both`) vs E2d == §8 E4a + E4b:** 37 events `idx` cells (H1 5
  BOS; conf 21 BOS + 1223→1224, 2828→2829; counter 8 BOS + 2828→2829), 34 KL `source_event_idx` (key-only), 3
  figures JSON-identical, run.log only the `[kl_zones]` prints' raw idx. Re-run it after the last E3 stage (§7).
- E2d landing review (combined lens ≈135k): 0 BLOCKER / 0 MAJOR. The FibTracker direct-index conversions are
  safe (traced: `_update_fib_cts`'s cross branch is unreachable; `_update_cycle1_main` runs only on the `:1070`
  cross, which carries `cycle1_bos_idx`, after the cycle-0 guard). Four reintroduced `.get(key, ev.idx)` mutants
  survived (equivalent on contract-legal streams) → a static guard `test_no_production_module_falls_back_on_a_contract_index_key`
  (kills all four); it also caught `wave_candles.py:252` `.get("confirmed_at", len(df) - 1)` → direct. Also: the
  orchestrator tuple local `bos_anchor_idx`, GOTCHAS `:680` (no display-read fallback), the `cts1_anchor` docs.
  Tests 842 passed + 1 xfail; replay == E2d's.
- **E2 session totals (2026-09-24):** landing reviews ≈0.9M subagent tokens (E2a 241k, E2b 302k, E2c 230k, E2d
  135k); tests 783 → 842 + 1 xfail; four byte-identical `/compare`s (E2a: the additive keys only).

---

## 7. E3 — timing fixes (each its own `/compare`; each flips `# Plan E E3x` markers to `ef.event_moment`)

| Stage | Sites (IN) | Prediction | Verification |
|---|---|---|---|
| **E3·0** pattern-path moment | `market_structure.py:1580` (at E3·0: `:1545–1553`): pattern-path `CTS_UPDATED` gains `meta["confirmed_at"] = int(apply_idx)`; `event_moment` returns it (and the raw path keeps `ev.idx`). Plan F's knowability cut then applies on the pattern path (Plan F §7 follow-up) | events meta +37 keys (7/23/7); the cut effect was measured by Plan F as 0 cells → **re-measure with a variant**; expected 0 other cells | unit: a pattern-path update with apply > idx is cut at the apply. Docs: ARCHITECTURE table `CTS_UPDATED` row; IMBALANCE_FILL_SEMANTICS |
| **E3a** fib + MS-mirror timing | IN §2.3 E3a + §2.4 items 1–6 and 9:<br>- `activated_at` writes;<br>- the fill horizon at EST (`fib_tracker.py:616`);<br>- the split `fill_horizon_idx` of `resolve_cross_cycle_eligibility` / `select_fib_anchor_for_cycle` (§6.2);<br>- the new_cycle terminal / `_mark_first_active`;<br>- the set-if-absent merge;<br>- the `scenario1_revert` terminal;<br>- `:1149` `.get("activated_at", cts_idx)` → direct;<br>- **the MS mirror** `_refresh_poi_inners_for_cycle` / `_update_cycle0_data` horizons (the old proposed "E3e", merged per Q18: IN §2.4 item 9 requires MS ↔ FibTracker parity "in the same change").<br>Scenario 1 per Q2 | **Named: 4 fib_lifecycle cells.** conf line 14 + counter line 2 (sub 3 cyc 0) `end_idx` 2828.0→2829.0; conf line 15 + counter line 3 (sub 3 cyc 1) meta `activated_at` 239→240 (slice-local).<br>**At risk** (the horizon moves one candle at 2829 and at 1224):<br>- sub 3 cyc 1's `cross_failed` single fib + POI IC 2808 (both lenses) if a fill confirms at 2829;<br>- conf sub 0 cyc 2's re-run `_m15_cross_check` on the cross started at 1169 (reactivate / deactivate / version cells);<br>- via the MS mirror, MS events (sd-prox CTS confirmation).<br>H1 0 **only if Q8 stays out of E3a**.<br>**Measure first** with a variant replay; the variant's cell list becomes the prediction | unit: `_make_second_cts_moment_after_extreme_data` via `_run_downstream_pipeline` → cycle-1 fib `start_idx` / `activated_at` 9→10, cycle-0 fib `end_idx` 9→10. Spec: FIB_LIFECYCLE_SPEC §7 cases 2–3, §15.3 |
| **E3a′** *(Q8)* | `cycle1_bos_idx` cond3 fill horizon (`:1544`, `:1624`) + c0 cond2 fill horizon (IN §2.4 item 5) → moment | measure first; H1 cells possible (BOS lag 14–76) | per measurement |
| **E3b** pool clock | `knowable_at_idx` (`sub_structure_pool.py:84-100`) on `ef.event_moment` for `CTS_ESTABLISHED`; the sibling-clip TIME (split in E2b; comment `:481`); `reference_zone.py:335-357` window + the recency sort key; `unified_probe._second_cts_moment` converges on `event_moment`; rewrite the false comment `reference_zone.py:344-347` (hazard H5) | **0**: no lagging EST straddles a sub cap (caps 1940, 2470, 2829, 3611, 3819, 4200) | synthetic cap in [anchor, moment); `test_sub_structure_pool.py:562-564` expectation 10 → 14. Closes PART4 §17.12 for EST |
| **E3c** probe Phase 2 | `unified_probe.py:602` `check_lo` → moment + 1; label `cts0_est=` (`:671`) re-sourced to the moment local | **0**; run.log identical (the window's one Phase-2 run: `p2_iter=1`, lag 0) | unit on the (9, 10) fixture: `check_lo` 10 → 11 |
| **E3d** prev-BOS filter | `orchestrator.py:279-288` `ef.event_moment(ev) >= rv_idx` over CTS_EST + CTS_UPDATED (needs E3·0; rev 1 would have raised `TypeError` at sid 1, pattern-path CTS_UPDATED 710). END value per Q6 | **0**: `[prev_bos_line] sid=1 start_idx=591 end_idx=902` | run.log identical |
| **E3f** *(Q5 = moment; landing-review note 2026-09-24: `compute_struct_start_by_sid` / `SidRecord.creation_event_idx` take a min over ALL event types — `event_moment` raises on STATE_CHANGED / RANGE_* / REVERSAL_* and is None on pattern-path CTS_UPDATED, so E3f must define the per-type moment first; the H1-overlay wave window (`w_start`) moves with it → figure diff)* | `compute_struct_start_by_sid` → min over moments; main `creation_event_idx` likewise or split | **measure first**. Raw base: H1 sid 0 96→115; subs 454→458, 1797→1816, 2365→2368, 2639→2649, 3304→3306, 3760→3786, 4027→4031, 3621→3656 (IN B9). Expected masked (KL/POI/fib start at moments ≥ the base) | `test_structure_lifecycle_moment.py:179-184`, `test_parent_tables.py:116` |
| **E3g** *(Q20; note: E3g-2's zone_proximity timeline holds BOS_THRESHOLD_UPDATED, on which `event_moment` raises, and is SORTED by `processing_order_key` — a moment-keyed walk needs the timeline re-sorted on the moment too)* | the remaining time halves of §7.1 marked "moment" | 0 each (one `/compare` per row; chart row with the figure diff) | per row |

After the last E3 stage, re-run both E4 variants; they must still equal §8.

### 7.1 Time halves that E2 routes through the anchor accessor and no E3 row above switches — a decision each (Q20)

| # | Site | Today | Rec |
|---|---|---|---|
| T1 | POI transition time + pre-window split (IN §2.4 item 7; `poi_zones.py:899-907`, `:917`, `:946`) | anchor | **moment**, E3g-1. Predicted 0: IC 678 cyc 2 / IC 2808 activate "initial" at first_active 1224 / 2829, where an imbalance-enter transition already exists |
| T2 | `find_ic_candidates` fill horizon `check_to_idx=cts_idx` (IN §2.4 item 8) | anchor | **stays** (retrospective IC identification on the final fib; Plan F set its cut to `None` by decision); record as `# stays` |
| T3 | `zone_proximity.py:379`, `:403` timeline pointer reading BOS `.idx` (IN §2.3 wrongly lists it as "already correct") | anchor | **moment**, E3g-2. 0 on the window: scan_start ≥ the BOS moment |
| T4 | charts `next_bos_idx` PB-search bound + H1-overlay `_wave_touches_window` (IN §2.4 item 14) | anchor | **moment**, E3g-3, with the figure diff; counts identical |
| T5 | fib Scenario-1 comparison (Q2) | anchor | **moment**, in E3a |

---

### 7.2 E3 as landed

- **E3·0 (2026-09-24).** `market_structure` pattern-path emit: `meta["confirmed_at"] = int(apply_idx)`;
  `ef.event_moment` returns it on the pattern path (`meta["confirmed_at"]`, direct — a pattern-path
  CTS_UPDATED without it raises `KeyError`) and is now `-> int` (never None). Measured on the change itself
  (`cmp_save.py --strip confirmed_at` vs `20260924_202017_4ef2607`): **0 real cells**; key-only cells H1 events 7,
  conf 23, counter 7 (== §7's 37) **+ `structure_levels` 7** (not in §7's prediction: its `meta` column carries the
  H1 CTS events' meta, as in E2a); figures JSON-identical (85/245, 151/124, 294/233); run.log only the FutureWarning
  line number (2627 → 2632). The one lagging row: conf sub 2 cyc 0, idx 2468, `confirmed_at` 2470 (entity-absolute —
  already in `_EVENT_META_IDX_KEYS`; the E1 guard needed no change, its static scan sees the new key). The Plan F
  knowability cut now applies on the pattern path (0 cells, as Plan F measured). Tests 842 → 848 + 1 xfail: Plan F's
  `test_guard_pattern_path_update_is_not_cut` replaced by `test_pattern_path_update_is_cut_at_its_apply_not_its_anchor`
  (anchor 25, apply 26, gap c2 25 → counted; kills a cut too early, e.g. at the anchor) +
  `test_pattern_path_update_gap_at_its_apply_waits_one_update` (anchor = apply = 25, gap c2 25 → not counted; the next
  update at 26 activates — the only shape where cut and uncut differ); the MS emit test runs a new lagging fixture (`_make_multicycle_data` + 24 maru / 25 small
  bear → `one_maru_opposite`, anchor 24 / apply 25) and pins every pattern update `event_moment == confirmed_at >= idx`.
  The FibTracker UPDATED time halves (`# UPD time half: Plan E E3·0/E3a`) still read the anchor — E3a. Docs:
  ARCHITECTURE table row + the `event_moment` bullet + the POI-sweep note; GLOSSARY `event_moment` / `event.idx` /
  `CTS_UPDATED` / `evaluated_at`; IMBALANCE_FILL_SEMANTICS; GOTCHAS ×4; LANDMINES ×4; PART4 §17.12 note; CROSS_CYCLE /
  FIB_LIFECYCLE / POI_ZONES specs; PRE_REFACTOR_INVARIANTS; `imbalance.py` / `fib_tracker.py` comments.
  - Landing review (1 combined lens, ≈96k): 0 BLOCKER. **MAJOR** — my first second pin (gap c2 26, past the anchor)
    passed with no cut at all (outside the fib range), so the exact pre-E3·0 behaviour (M6p: `_evaluated_at = None`
    on the pattern path) survived → replaced by the anchor = apply = c2 pin above (M6p now killed). MINORs folded
    in: the FibTracker markers `UPD time half: Plan E E3·0/E3a` → `Plan E E3a` (§6.7's text is historical); the
    conftest contract validator now checks `CTS_UPDATED` (pattern path: int `confirmed_at >= idx`; raw: none) +
    `test_event_contract_validator.py` pins (5). The validator surfaced three legacy constructions: 3 test-built
    pattern updates in `test_event_fields.py` (key added) and the Plan-B-save CSV test (pre-E3·0 rows: stand in
    `confirmed_at = idx`, H1 apply == idx on all 7 rows). Mutation 10/11 killed (M10, the dropped `int()` cast, is
    equivalent: `apply_idx` is already an int). `apply_idx >= cts_anchor_idx` holds by construction
    (`_cts_from_breakout_event` spans `[start, max(end, confirmation)]`, apply = confirmation or end).

- **E3a (2026-09-24).** FibTracker: `_moment()` (= `_evaluated_at`, asserted inside a handler) is the TIME half of
  every handler. EST: `cts_established_idx = self._moment()` (one line flips everything E2b threaded: `activated_at`,
  the EST fill horizon, the routine / anchor-selector `fill_horizon_idx`, Scenario 1 (T5), the revert terminal,
  `current_candle`); the new_cycle terminal and `_mark_first_active` follow from `activated_at`. Update path (the
  `UPD time half` sites): the cross-cycle cycle-0 late activation (horizon + `activated_at`), `current_candle` of
  `_m15_cross_check` / `_run_main_cross_check`, `_handle_cycle0_cts_updated` (both `activated_at` + the Scenario-1
  comparison, Q2), `_update_fib_cts` (cond2 / own fill horizon, `reactivated_at` / `deactivated_at`),
  `_update_cycle1_main` (cond2 horizon, stamps, create-on-fail horizon + `activated_at`). `_activate_fib`:
  `meta["activated_at"]` direct. **Not moved (E3a′):** cond3 (`cycle1_bos_idx`), the c0 cond2 caches, the anchor
  selector's `snapshot_horizon_idx`; cond1's c0 horizon (`c0_cts_idx`) is the cycle-0 cache's (E3a′ too). **MS
  mirror:** `_refresh_poi_inners_for_cycle(moment_idx)` — the pattern branch passes `apply_idx`, the raw update `i`
  — → `compute_poi_inners_for_cycle(..., *, fill_horizon_idx)` (keyword-only, REQUIRED; `PoiInnersResolver` is now
  `Callable[..., List[float]]`) → `select_fib_anchor_for_cycle(fill_horizon_idx=...)`. No real choice here: the
  refresh runs ON the triggering event's moment, so "apply candle" and "processing candle" coincide. IC cond3 inside
  the resolver (`find_ic_candidates`, `check_to_idx = cts_idx`) stays (T2).
  **Measured (the change itself vs `20260924_202017_4ef2607`, `--strip confirmed_at`) == §7's named prediction
  exactly: 4 real cells** — conf fib_lifecycle line 14 + counter line 2 `end_idx` 2828.0 → 2829.0 (sub 3 cyc 0,
  `new_cycle`); conf line 15 + counter line 3 meta `activated_at` 239 → 240 (sub 3 cyc 1, slice-local). **None of the
  at-risk items moved** (IC 2808 / the `cross_failed` single, conf sub 0 cyc 2's re-run `_m15_cross_check` at 1224,
  MS events via the mirror); H1 0; the 3 figures JSON-identical; run.log only the parked `by_lens` order. (Plus
  E3·0's key-only cells, unchanged.) Tests 848 → 853 + 1 xfail: the role pin `…_the_moment_for_activated_at` (20 →
  22), `test_fib_tracker_update_fill_horizon_and_stamp_are_the_moment` (gap fills at 31; update anchor 30 / moment 32
  → deactivated, stamped 32), `test_ms_inflight_poi_refresh_fill_horizon_is_the_moment` (spy: (9, 10) on the lagging
  EST fixture, (24, 25) on the lagging pattern update, raw refreshes (r, r)), §7's unit
  `test_lagging_est_fib_lifecycle_is_timed_at_the_moment` (cycle-1 fib start / `activated_at` 10, cross_cycle cycle-0
  `new_cycle` end 10; both modes); the E3·0 pin's stamp 25 → 26; 2 direct resolver calls pass `fill_horizon_idx`.
  Docs: FIB_LIFECYCLE_SPEC §6 / §7 case 2 / §15.3 (the three "known limit"s resolved), ARCHITECTURE "known sites"
  paragraph, GLOSSARY Naming Standard (`activated_at` mismatch), IMBALANCE_FILL_SEMANTICS horizons, LANDMINES
  bounded-run residual, CROSS_CYCLE_FIB_SPEC, routine / selector / `_m15_cross_check` docstrings.
  - Landing review (2 lenses, ≈396k: conformance 198k, mutation 198k). **Conformance:** 0 BLOCKER, 2 MAJOR, 8 MINOR.
    MAJOR 1 — the IMBALANCE_FILL_SEMANTICS FibTracker table + POI_ZONES_SPEC horizon prose were stale → rewritten (+ a
    MarketStructure-table row for the resolver's cond1). MAJOR 2 — four cycle-0 horizons correctly left on the anchor
    had no marker → `# Plan E E3a′ → moment` at `_c0_has_unfilled_now`, the update-path c0 cache re-snapshot, and
    both `c0_cts_idx` cond1 reads. **E3a′ note:** after E3a the Scenario-1 cycle-0 activation asks at the moment at
    CTS_0 EST but at the cached CTS_0 anchor on CTS_0 UPDATED (`_c0_has_unfilled_now`) — E3a′ closes it. MINORs
    folded: Scenario-1 docs say "the CTS_0 event's moment >= rv" (LANDMINES, 3 specs, 5 docstrings/comments); the
    revert terminal "moment" wording; "as of `fill_horizon_idx`" docstrings; LANDMINES "Scenario 2 anchor agreement"
    now states the horizon lock-step rule; ARCHITECTURE's "fill-horizon half not yet measured" updated; `_moment()`
    raises `RuntimeError` (not a bare assert); `_activate_fib(meta)` required. **Declared deviation (§2.4 item 2):**
    FibTracker still threads `current_candle` / `cts_established_idx` through `_m15_cross_check` /
    `_run_main_cross_check` / `_maybe_activate_main_cross`; every value equals `self._moment()` (reviewer-traced) —
    folding them is E5 hygiene. Not changed: the Scenario-1 prints (`at idx={cts_idx}`) log the anchor (run.log kept
    byte-identical); `plan_f_inputs/inners_shadow.py` already declares it runs at `754a642` only.
    **Mutation:** the landed suite killed only 16/41 site reverts (every H1-main sid≥1 EST / update path, the
    Scenario-1 comparisons, the revert terminal, the §11b peek / stamps, the bearish MS raw refresh, and
    `compute_poi_inners_for_cycle` ignoring its horizon survived) → the reviewer's 20 pins adopted as
    `tests/test_e3a_mutation_pins.py` (lagging grid, moment = anchor + 2; mirrored sd −1 MS fixture): 39/41 killed;
    spot-re-killed here (Scenario-1 EST comparison → anchor; resolver horizon → `cts_idx`). Equivalent survivors:
    M24 — `_update_fib_cts`'s `is_cross_cycle` branch is DEAD (no single key ever holds `cross_cycle: True`; delete
    in the dead-code hygiene pass — DELETED 2026-09-28, hygiene 5b); M32 — the h1 create-on-fail stamp is unreachable (cond2 and the normal check ask
    the same window at the same horizon). Tests 853 → 873 + 1 xfail; replay after the fixes == the measured E3a.

- **E3a′ (2026-09-25; Q8).** Both layers, one change. FibTracker: cond3 "has BOS_1 filled cycle 0?" at the BOS_1
  MOMENT (== the CTS_1 ESTABLISHED moment, definitional): at EST the anchor selector's `snapshot_horizon_idx =
  cts_established_idx`, on the update path `_update_cycle1_main` / `_update_fib_cts` read
  `self._bos_moment_by_cycle[(sid, 1)]` (new, set at every EST next to `_bos_by_cycle`); the cycle-0 cond2 cache at
  the moment of its write (EST: `c0_fill_horizon_idx = cts_established_idx`; update re-snapshot: `self._moment()`),
  recorded as `c0["fill_horizon_idx"]` and read by the update path's cond1-c0 re-asks; `_c0_has_unfilled_now` asks
  at `self._moment()` (closes E3a's EST/UPD asymmetry). MS: `_update_cycle0_data(moment_idx)` (the refresh's
  moment); new state field `cts_established_moment_idx` (set in `_emit_cts_established` from `confirmed_at`) →
  `compute_poi_inners_for_cycle(*, snapshot_horizon_idx)` (keyword-only, REQUIRED) → the selector. No `E3a′`
  markers remain.
  **Measured (vs `20260925_073236_9bfc4bc`): 0 real cells; figures JSON-identical; run.log identical.** The plan
  expected H1 cells possible (BOS lag 14–76), so a shadow run (scratch `e3ap_shadow.py`: old vs new horizon + answer
  at every E3a′ call) explains the 0: cond3's horizon moved at all 16 of its calls (FibTracker EST 2, update 6, MS
  mirror 8) — the Q8 example itself, H1 sid 1 cyc 1 BOS 728 → 748, windows [689, 710] / [728, 748] — and **no answer
  flipped** (cycle 0's gap is still unfilled at 748); the MS cycle-0 cache moved once (an M15 frame, 153 → 155,
  same answer); the FibTracker caches / cond1-c0 re-asks have lag 0 here; `_c0_has_unfilled_now` is not reached on
  this window. (One replay's M15 fetch failed with an OANDA 504 — 3648/4228 candles, 1255+ event cells "changed" —
  the fetch gate caught it; the re-run passed.)
  Tests 873 → 880 + 1 xfail (883 after the review): 7 `test_e3ap_*` pins in `tests/test_e3a_mutation_pins.py` (EST / update cycle-0 cache,
  cond3 at the BOS_1 moment via the selector, `_c0_has_unfilled_now`, the MS cycle-0 snapshot, `compute_poi_inners`
  forwarding `snapshot_horizon_idx`, the MS spy: every cycle-1 refresh on the lagging-EST fixture carries 10); 4
  direct calls pass the new arguments. **Own mutation loop (scratch `mut_e3ap.py`, full suite per mutant): 7/9
  killed**; the 2 survivors are equivalent by lock-step — the update path's cond3 and cond1-c0 re-asks: a fill that
  makes the anchor and moment horizons disagree fails the same check at CTS_1 EST, so no cross reaches
  `_update_cycle1_main`. (Harness lesson: `subprocess.run(["python", …])` resolved to a Python without pytest and
  reported every mutant "SURVIVED" — use `sys.executable` and treat a non-zero rc without FAILED lines as an error.)
  Docs: ARCHITECTURE (2 places), IMBALANCE_FILL_SEMANTICS (FibTracker + MS tables, horizons prose), LANDMINES
  "Scenario 2 anchor agreement", CROSS_CYCLE_FIB_SPEC, POI_ZONES_SPEC, selector / resolver docstrings.
  - Landing review (1 combined lens, ≈139k): 0 BLOCKER. **MAJOR (fixed in E3a′):** E3a′ itself created an MS ↔
    FibTracker cycle-0 cache divergence — MS re-snapshots on EVERY cycle-0 refresh, incl. a pattern update RESTATING
    the current anchor (raw @i, then the pattern applies at i+k with the same extreme), but FibTracker re-snapshotted
    only on a strictly later anchor; with both keyed on the moment, a gap filling in (i, i+k] made the layers
    disagree on Scenario 2 (before E3a′ both asked at the same anchor). Fix: `_handle_cycle0_cts_updated`
    re-snapshots on every unlocked update with anchor `>=` the cached one (the anchor moves only on `>`). The shadow
    run's "MS cache moved once (M15 153 → 155)" was an MS-only cache (subs run no FibTracker cycle-0 cache), so this
    window has no instance; the replay with the fix == 0 cells again. MS still re-snapshots on a REGRESSING pattern
    anchor — the known latent "pattern-path CTS_UPDATED regressing st.cts" bug, left as is (FIXED 2026-09-27: a
    pattern-path update now needs a strict new extreme — MARKET_STRUCTURE_SPEC "CTS"; no regressing or equal anchor
    reaches either cache). Pins: the reviewer's
    `test_e3ap_c0_now_on_an_equal_anchor_pattern_update` (horizon 25 after the equal-anchor update — kills the pre-fix
    `>`), a parity pin feeding FibTracker and MS `_update_cycle0_data` the same stream, and
    `test_e3ap_c0_now_asks_at_the_moment_not_the_cache_horizon` (the regressing-anchor stream: kills the reviewer's
    surviving M6, `_c0_has_unfilled_now` reading the cache horizon). **MINORs:** IMBALANCE_FILL_SEMANTICS three stale
    rows, the `has_unfilled_imbalance` docstring, a fib_tracker comment — fixed. **Stronger equivalence** (the
    reviewer's proof): the update-path cond3 / cond1-c0 re-asks in `_update_cycle1_main` are TAUTOLOGICAL — a cycle-1
    cross exists only if cond3 (at the CTS_1 moment m) and cond2 (at the cache horizon h) were True at EST; a
    two-stroke fill never un-fills, BOS_1 anchor <= m and `c0_cts_idx` <= h, the c0 range is frozen by then and
    `_maybe_activate_main_cross` handles cycles >= 2 only, so both reads are always True where reached (only cond2
    decides) → dead-code hygiene list, with `_update_fib_cts`'s dead cross branch (2026-09-28, hygiene 5b: the
    re-asks are KEPT — deleting them would change the run.log `cross-cycle check: cond1=… cond3=…` line; line-traced
    True on every reach, 6/6 replay + 7/7 suite; the create-on-fail single they made unreachable is DELETED and its
    invariant asserted). Its mutants M1 / M2 / M5
    (`_bos_moment_by_cycle` = the anchor) survive for that reason. Mutation 7/10 killed by the reviewer + both of mine
    re-killed here. Tests 880 → 883 + 1 xfail.

- **E3b (2026-09-25).** The pool clock on the moment. `knowable_at_idx` keys `CTS_ESTABLISHED` and **pattern-path
  `CTS_UPDATED`** (declared extension: its moment exists since E3·0 and it had the same §17.12 straddle) on
  `confirmed_at`, like `BOS_CONFIRMED` (a raw `CTS_UPDATED` carries none → `ev.idx`, its moment); the sibling clip
  (`entity_df_mutation._build_sibling_cts_ref_zone_from_pool`) and the reference-zone window + recency sort key on
  `ef.event_moment`; the false H5 comment ("never at the same idx") rewritten; `unified_probe._second_cts_moment`
  already read `ef.event_moment` (E2d). §17.12 now names only `REVERSAL_CANDIDATE`.
  **Measured (vs `20260925_081206_efc04bd`) == §7's 0:** 0 real cells, figures JSON-identical, run.log identical
  (the parked `by_lens` order aside). Shadow run (scratch `e3b_shadow.py`): the pool clip saw 1417 events, the key
  moved on 3, none crossed a cap; the reference builder saw 216 candidates, 2 keys moved, the winner's anchor never
  changed (the sibling clip itself is upstream of that wrap — not separately shadowed).
  Tests 883 → 890 + 1 xfail (891 after the review): the stamped-window pin → `test_reference_window_filters_on_the_moment_since_e3b` (anchor 9 /
  moment 12: outside [5, 10], inside [5, 12]); `test_reference_recency_is_the_moment_since_e3b`;
  `test_clip_keys_est_and_pattern_update_on_the_moment`; `TestSiblingClipOnTheMoment` ×3 (incl. a record window
  narrower than the read window — the first two alone let the sibling-clip revert survive, masked by the
  downstream window); §7's `test_sub_structure_pool.py` expectation 10 → 14. **The conftest validator now requires
  `via` on every `CTS_UPDATED`** (E3b's readers all ask `event_moment`, which needs it): it surfaced two legacy
  fixtures (`test_first_trigger_migration._cts_event` → pattern path; `test_wave_candles._make_event` → raw path by
  default) + a self-pin. Own mutation loop: 5/5 killed. Docs: ARCHITECTURE known-sites, LANDMINES (sibling window,
  run cap, "Run Cap ≠ Lifecycle End" clip rule), PART4 §17.8 note + §17.12.
  - Note: the recency change also reaches the two `idx_window=None` callers — the main-H1 reversal reference
    (`structure_engine.py`) and the entity reversal path — in scope, 0 cells.
  - Landing review (1 combined lens, ≈123k): 0 BLOCKER. **MAJOR:** the mirror's raw-idx split
    (`new_ev.idx = ev.idx + slice_begin`, clip on the moment) lost its pin once the clip time differed from the
    stamped idx — a mutant writing the clip time moved a pattern-path winner's probe input / pool key from the anchor
    to the moment and survived → the reviewer's `test_mirror_keeps_the_raw_idx_of_a_pattern_path_update` (anchor 17,
    moment 19, slice_begin 3 → 20). **MINOR 1 (done):** `knowable_at_idx(ev)` now takes the event and delegates to
    `ef.event_moment` for BOS / EST / CTS_UPDATED (no hand-copied per-type rule, no `.get("confirmed_at")`
    fallback); its two test callers updated. MINORs 2–3: the `project_to_window` docstring and 4 `reference_zone`
    "reverse-idx / on event idx" phrases. Tie analysis (reviewer): on every reachable stream a same-moment tie within
    a sid is EST(N) + CONFIRMED(N) at the apply candle (same anchor; CONFIRMED wins); an EST vs raw UPDATED tie is
    impossible (the span contains the apply candle); E3b also fixes a pre-E3b misorder (CONFIRMED(N) at c with
    anchor(N+1) < c < moment(N+1) used to beat EST(N+1)); saved data has no same-sub moment tie. The tie-order
    UPDATED/EST swap survives as equivalent. Mutation 6/6 (+ M9) killed on the final code; replay after the fixes
    == 0 cells. Tests 883 → 891 + 1 xfail.

- **E3c (2026-09-25).** `unified_probe._run_phase2`: `check_lo = cts0_est_idx + 1` (the first CTS_ESTABLISHED's
  moment, the existing `ef.event_moment` local) — Phase 1 and Phase 2 now open the retrace window at the same
  place; the `[unified_probe phase2] reset triggered … cts0_est=` label prints that moment. **Measured (vs
  `20260925_081206_efc04bd`; E3b changed nothing) == §7's 0:** 0 real cells, figures JSON-identical, run.log
  identical — the window's one Phase-2 run (`p2_iter=1`, `cts1_anchor=1020 cts1_moment=1020`) has a lag-0 first CTS,
  so the value is the same there (no shadow run needed: the plan named the single call). Pin:
  `test_phase2_retrace_window_opens_after_the_first_cts_moment` (stubbed MS run, CTS_0 anchor 9 / moment 12, no
  confirmation, probe_end 20 → the candidate search is asked over [13, 20]; fails on the anchor revert — checked).
  The label is a debug print, not pinned. Docs: ARCHITECTURE known-sites, PART4 "What `first_CTS_EST.idx` is"
  (the verified divergence → closed). Tests 891 → 892 + 1 xfail.

- **E3d (2026-09-25).** `orchestrator._prev_bos_lines`: "the first CTS of the new sid known at/after the reversal"
  filters on `ef.event_moment(ev) >= rv_idx` over CTS_ESTABLISHED + CTS_UPDATED (possible since E3·0 — rev 1 would
  have raised on the pattern-path CTS_UPDATED 710); the line END stays the anchor (Q6); iteration stays in processing
  order (within one sid the stamped order and the moment order agree: a raw update needs `st.cts`, set at the EST's
  apply). **Measured == §7's 0:** 0 real cells, figures identical, run.log identical incl.
  `[prev_bos_line] sid=1: start_idx=591 end_idx=902`. Pin: `test_prev_bos_line_filter_is_the_moment` (sid 1's CTS_0
  anchor 13 / moment 16, reversal 15, a later raw update at 20 → END 13; the stamped revert ends at 20 — checked).
  Tests 892 → 893 + 1 xfail.
- **E3c + E3d landing review** (1 combined lens, ≈107k): 0 BLOCKER, 0 MAJOR. **The ordering claim above was
  incomplete:** a pattern-path CTS_UPDATED that REGRESSES the CTS (zones-audit latent bug (a); FIXED 2026-09-27 — MS no
  longer emits one, the `min(…, key=ef.event_moment)` pick stays as defence in depth) is stamped before a
  raw update it is known after (EST 13/16, raw 17, pattern anchor 15 / moment 20, reversal 17 → first-in-order picks
  END 15, the earliest-known is 17) → `_prev_bos_lines` now takes `min(qualifying, key=ef.event_moment)` (stable;
  byte-identical wherever the orders agree; pin `test_prev_bos_line_picks_the_earliest_moment_not_the_first_stamped`
  kills the first-in-order rule — checked). Surviving mutants pinned: `>=` → `>`
  (`…_inclusive_at_the_reversal`) and dropping CTS_UPDATED (`…_falls_through_to_a_raw_cts_update`). E3c: nothing
  else belongs to it (`check_hi` = the CTS_0_CONFIRMED anchor − 1 is a location bound; `_no_retrace_fin` /
  `_second_cts_fin` already moments); an empty window finalizes identically; reachable non-zero cases (an extreme
  in [anchor+1, moment] passing / failing the reset) all follow the intended rule. MINORs: PART4 window sentence,
  `ProbeResult.cts0_est_idx` docstring (Phase 2 fills it), `_run_phase2` window docstring. Replay after the fix ==
  0 cells. Tests 893 → 896 + 1 xfail.

- **E3f (2026-09-25; Q5).** **User decision 2026-09-25** (AskUserQuestion with the window's first-event table): the
  `compute_struct_start_by_sid` base = the structure's **first CTS_ESTABLISHED moment** (== its BOS_0 moment) —
  chosen over "min over every event's moment" (which would first need moments for STATE_CHANGED / RANGE_* /
  REVERSAL_*) and over keeping the anchor; `SidRecord.creation_event_idx` **stays the anchor** (a historical field,
  like the sub side's `starting_idx`; `# stays`). A sid that emitted events but never established (data edge only)
  keeps its first stamped idx (the reversal handoff overrides it) — explicit, pinned. Evidence: on every structure
  of the window the first two events are BOS_0 (at its anchor) then CTS_0 ESTABLISHED (at the moment), nothing
  earlier. **Measured (variant first, then the change; vs `20260925_081206_efc04bd`): 0 real cells, figures
  JSON-identical** — the plan note's "the H1-overlay wave window `w_start` moves → figure diff" did not materialise;
  run.log identical. Base moves: H1 sid 0 96 → 115, sid 1 689 → 703 (then the handoff: 902), subs slice-local
  50 → 52..85 (then the sub floor) — masked everywhere, since every zone already clamps at or after the moment.
  Tests: 7 expectations follow the new base (parent tables `struct_start` 96 → 115 / 5 → 11 / 5 → 10, the
  structure_lifecycle test, the role pin → `test_struct_start_is_the_moment_and_creation_idx_the_anchor`
  (22 / 20), the POI missing-`confirmed_at` test now fails loudly as a KeyError at the struct_start read) + a
  never-established assertion. Own mutation loop 4/4 killed (base → stamped, base → EST anchor, the fallback
  dropped, creation → moment). Docs: KL_ZONES_SPEC, POI_ZONES_SPEC, LANDMINES, `stamped_idx` docstring, sid_records
  / structure_lifecycle docstrings. Tests stay 896 + 1 xfail.

- **E3g-1 (2026-09-25; §7.1 T1).** `poi_zones._compute_poi_activation_history`: the CTS events' pre-window split,
  in-window transition time AND the sweep's order key on `ef.event_moment` (the two stamped-idx sorts — the by-key
  pre-group and `sorted_cts_events` — were unmarked in E2 but are the same time order); cond1 keeps the anchor. The
  "load-bearing" pre-window comment rewritten for the moment key (the establishing EST, known at
  `cts_established_idx <= first_active`, is pre-window when strictly before, else a transition at `first_active`
  applied before that candle's evaluation). **Measured (vs `20260925_081206_efc04bd`) == T1's 0:** 0 real cells,
  figures identical, run.log identical. Pins (`test_poi_activation_moment.py`): the transition time (IC 7 past the EST
  anchor; a pattern update anchored 8 / known 11 → active at 11, not 9), the pre-window split (floor 9 → the same
  update is in-window at 11), the order (a regressing pattern update — latent (a) — known at 12 after a raw update to
  9: the latest KNOWN CTS, anchor 8 < IC 9 → no activation). Own mutation loop 3/4 killed; the by-key pre-sort is
  equivalent (the sweep re-sorts by moment, stably, and an EST / pattern-UPDATED pair cannot tie on a moment). Docs:
  ARCHITECTURE known-sites. Tests 896 → 899 + 1 xfail.

- **E3g-2 (2026-09-25; §7.1 T3).** First the per-type moment the note asked for: `ef.event_moment` now defines
  `BOS_THRESHOLD_UPDATED` → `ev.idx` (every emitter — `_bos_barrier_step`, `_maybe_expire_reversal_watch` — stamps the
  processing candle `i`; a fact, like CTS_THRESHOLD_UPDATED). zone_proximity: `_build_cycle_threshold_timeline` sorts
  on `(ef.event_moment, type)` (re-sorted, per the note) and both pointer walks (the initial `<= scan_start` pass and
  the per-candle `< i`) compare moments. **Measured == T3's 0:** 0 real cells, figures identical, run.log identical.
  Pins: `event_moment(BOS_THRESHOLD_UPDATED) == idx`; `test_threshold_timeline_is_ordered_by_moment` (a BOS anchored 5
  / known 12 sorts after a threshold event at 10). Own mutation loop 2/4: the two pointer reverts are EQUIVALENT on
  contract-legal streams (the E2c landing review's argument: a cycle's BOS moment == its EST moment <= its
  CTS_CONFIRMED candle = `scan_start`, and threshold events are stamped at their moment, so both keys agree wherever
  the walk runs). Docs: GOTCHAS narrow-cycle timeline, LANDMINES "Event Sort Order", `event_moment` / `stamped_idx`
  docstrings. Tests 899 → 900 + 1 xfail.

- **E3g-3 (2026-09-25; §7.1 T4).** The charts' PB-search upper bound (`next_bos_idx`: the next sid's first BOS) is its
  MOMENT, and "first" is picked by moment, at all three sites (`export_m15_chart.py` ×2, `export_plotly.py`); the
  H1-overlay `_wave_touches_window` start already moved with E3f's struct_start. **Measured with the figure diff:
  0 real cells, 3 figures JSON-identical** (== T4's prediction), run.log identical. Why 0 (data check): no
  previous-sid `STATE_CHANGED` lies in `[first BOS anchor, its moment)` on any structure (H1 sid 1: 689 → 703; the
  subs have one sid each). The chart sites have no unit harness — as accepted in the E2c review, their proof is the
  figure diff. Tests stay 900 + 1 xfail. **All §7 / §7.1 E3 rows are now landed; no `# Plan E E3` markers remain.**

- **E4 variant re-run after the last E3 stage (2026-09-25, at `0ba0c84`) == §8 E4a + E4b exactly:** `FLIP=both`
  vs the normal replay of the same code: 37 events `idx` cells (H1 5 BOS; conf 21 BOS + 1223→1224, 2828→2829; counter
  8 BOS + 2828→2829) — every one in the `idx` column; 34 KL `source_event_idx` (key-only); 3 figures JSON-identical;
  run.log only the 22 `[kl_zones]` print lines. The E3 stages changed no reader's role; E4 is unblocked.
- **E3f + E3g landing review** (1 combined lens, ≈163k): 0 BLOCKER, 0 MAJOR. Verified: E3f matches the user
  decision exactly (the fallback only reachable for a never-established sid 0 — harmless, no zone exists); E3g-1 an
  EST at `first_active` gives the same history in-window as pre-window (atomic same-idx evaluation — the pre-window
  `<` vs `<=` mutant is EQUIVALENT, do not pin it); E3g-2 every BOS_THRESHOLD_UPDATED emitter stamps `i`, pointer
  reverts equivalent; E3g-3 conforms; no undeclared anchor-as-time read remains in production. **Folded in:** the
  E3g-2 initial pass's `<=` was unpinned and NOT equivalent (a touch on the CTS_CONFIRMED candle of a narrow cycle
  counts against the Rule-3 cap) → `test_initial_pass_applies_events_known_at_scan_start`; stale comments / docstrings
  (poi_zones struct_start, zone_proximity "idx" → "moment", ARCHITECTURE known-sites wording, parent_tables header,
  test docstrings); the PB-search LOWER bound declared a location (3 chart sites); the `event_moment` threshold
  caveat (after a reversal-watch expiry rewind, replayed candles re-emit threshold events stamped j < i — true in the
  replay frame only; pre-existing). **Open — user decision (MINOR-1):** the H1-overlay wave filter on the sub charts
  compares anchor-to-anchor wave spans against `w_start = struct_start` (now the moment): for H1 sid 0 the first leg
  (BOS_0 anchor → cycle-0 CTS_CONFIRMED anchor) would be hidden whenever that CTS anchor < CTS_0's moment (a lagging
  EST with no CTS extension before confirmation) — 0 on this window. The E3f bullet's "masked everywhere" holds for
  the zones, not for this filter. Tests 900 → 901 + 1 xfail.
  **MINOR-1 resolved (user decision 2026-09-25, AskUserQuestion with the lagging-CTS_0 example):** the overlay's window
  start is a LOCATION — `export_m15_chart._h1_overlay_window_start_by_sid` = the sid's first structural anchor (min
  `ef.stamped_idx`, BOS_0's) or, for a sid whose predecessor reversed, the handoff — used by the wave filter and the
  PB→BOS line filter instead of `compute_struct_start_by_sid`. Replay: 0 cells, 3 figures JSON-identical. Pin
  `test_h1_overlay_window_starts_at_the_first_anchor_not_the_moment` (BOS_0 96 / CTS_0 anchor 110 known 115: the
  defining leg is drawn; keyed on the moment it would not be; sid 1 handoff 902 still hides its retroactive leg).
  Docs: CHARTING_SPEC "H1 overlay structure lines", LANDMINES. Tests 901 → 902 + 1 xfail.

## 8. E4 — the flip (emit sites only + the docs that invert)

**E4a — `CTS_ESTABLISHED`.** `market_structure.py:1514` passes `int(apply_idx)` as `idx` and asserts
`idx == meta["confirmed_at"]`. Prediction (vs the last E3 save):
- **Events `idx`: 3 cells** — conf 1223→1224 (sub 0 cyc 2), conf 2828→2829 and counter 2828→2829 (sub 3 cyc 1).
  H1 0.
- **run.log: 2 `[kl_zones]` lines** unless B12 prints the anchor explicitly: `(819, 0, 2)` → `(820, 0, 2)`
  (run.log:8483) and `(239, 0, 1)` → `(240, 0, 1)` (run.log:8692). These are slice-local values.
- **Everything else byte-identical, including figures:** POI `activation_history` of IC 678 / IC 2808 (cond1 reads
  the anchor, and the transition time is already the moment after T1), `structure_levels`, POI
  `cts_established_idx`, triggers / subs / pool keys.
- **Docs:** every "`CTS_ESTABLISHED` `ev.idx` IS the anchor" sentence inverts. The list is built before E4 by
  re-grepping IN §1.9's per-file list for the EST sentences (IN §1.9 has no "EST part"). Includes the GOTCHAS heading
  "…Not the CTS Extreme", the ARCHITECTURE row (with the `ev.price` column, Q19) and the
  `test_poi_activation_moment.py` docstring.

**E4b — `BOS_CONFIRMED`.** `:1531`, `:1546` pass `apply_idx`. Prediction:
- **Events `idx`: 34 cells** — H1 96→115, 591→652, 689→703, 728→748, 826→902; conf 21 (lag 2–411); counter 8
  (lag 4–285).
- **KL: 0** — `source_event_idx` was deleted in E4b-pre (Q21); KL geometry reads `bos_anchor_idx`.
- **run.log:** the `[kl_zones]` BOS prints (B12).
- **Everything else byte-identical, including figures:** KL geometry, fib `bos_idx` / `cycle1_bos_idx`, FC probe
  inputs / pool keys / `validated_parent_idx` 96/591/826, prev-BOS line start 591, chart dots, `structure_levels`,
  final.csv `bos_idx` (MS state). The events CSVs are written in emission order, so no rows reorder.
- **Docs:** the BOS half (GOTCHAS "BOS_CONFIRMED `ev.idx` Is the BOS Extreme…" — renamed "…Was the BOS Extreme Until Plan E E4b…"; the ARCHITECTURE row;
  KL_ZONES_SPEC event indexing (its `source_event_idx` part done in E4b-pre); MEMORY "Key Architecture Points").
- **Lesson from the E4a mutation lens (§8.1), apply BEFORE the flip:** a flip turns every raw-`ev.idx` LOCATION
  read into a moment read, and the E4a lens found 3 production location readers with no lagging fixture (all
  existing fixtures had lag 0 at those sites). For E4b, enumerate every BOS LOCATION reader (`ef.bos_anchor_idx`
  call sites: KL geometry, fib `bos_idx` / `cycle1_bos_idx`, FC probe input, prev-BOS line start, the chart
  dots / PB bound, `structure_levels`) and check each has a lagging-BOS pin (anchor < moment) — the mutation lens
  then confirms. The BOS lag is large on this window (H1 14–76, M15 2–411), so an unpinned reader is likely to
  move the figures, not only the CSVs.

**E4c — pattern-path `CTS_UPDATED`** (Q1 = flip). `idx := apply_idx` + meta `cts_anchor_idx` (E2 readers then use
`ef.cts_anchor_idx` for every CTS_UPDATED, and E2b must already route CTS_UPDATED LOCATION reads through it).
Baseline: 37 pattern-path rows, **1** lagging (conf sub 2 cyc 0: 2468 vs apply 2470, masked by sub 2's start 2470).
Prediction: 1 idx cell; everything else identical (the moment already drives the timing reads since E3·0).

**E4 tests:** the expected-value edits (IN §2.7 "E4, EST part" / "E4, BOS part"); the factory default `idx =
confirmed_at` and the conftest validator's `idx == confirmed_at`; the E4-simulation test becomes a regression pin of
the real emitter.

### 8.1 E4 as landed

- **E4a (2026-09-25).** The establish block calls `_emit_cts_established(int(apply_idx), cts_price,
  cts_anchor_idx=…)`; `_emit_cts_established` asserts `idx == meta["confirmed_at"]` BEFORE building the event (the
  single EST emit site — grep-verified; `MarketStructure._rewind_to` replays from scratch, it re-stamps nothing).
  **Measured (vs `20260925_112624_c031972`) == §8 exactly:** events `idx` 3 cells (conf 1223→1224, 2828→2829;
  counter 2828→2829), H1 0; the 3 figures JSON-identical; run.log only the two `[kl_zones]` EST prints — B12
  prints the anchor next to the idx: `(819, 819, 0, 2)` → `(820, 819, 0, 2)`, `(239, 239, 0, 1)` → `(240, 239, 0,
  1)` (plus the FutureWarning line number and the parked `by_lens` order). Replay 48.4 s wall.
  **Contract flip, same commit:** `conftest.EVENT_IDX_IS` is per type (`{"CTS_ESTABLISHED": "moment",
  "BOS_CONFIRMED": "anchor"}` — E4b flips the second); the factory's EST default `idx` = `confirmed_at`
  (`make_event`'s `idx` ARGUMENT stays the anchor); `test_e4_simulation.py` → the real-emitter pin (every EST
  `idx == confirmed_at`, the lagging one (anchor 9, idx 10)), the emitter-assert pin, and a role SWAP test (EST
  back to its anchor = the pre-E4a shape, BOS to its moment = the E4b shape → downstream outputs identical); the
  Plan D `mutate=` hooks move the EST `idx` with `confirmed_at`, and their "anchor == moment" preconditions read
  `cts_anchor_idx` (the old `ev.idx == confirmed_at` had become tautological); the mirror pin →
  `test_anchor_keys_are_int_and_idx_is_the_contract_index`; the validator / `event_fields` / order-pin tests follow
  (shapes renamed "anchor" / "moment"); 3 EST-only role pins dropped `illegal_event_contract` (legal now); the
  Plan-B-save CSV loader translates a pre-E4a EST row (`idx := confirmed_at`) and sorts with
  `ef.processing_order_key`. Only 6 tests failed on the bare flip (validator ×2, `event_fields` ×2, the old
  simulation's EST half ×2) — E2's accessors had made the rest of the suite role-neutral. Tests 902 → 905 + 1 xfail.
  **Docs (the inversion list, built by grep first):** ARCHITECTURE "`ev.idx` convention" (intro; table re-cut into
  `ev.idx` / moment / anchor / `ev.price` (Q19) / other-meta columns, every `ev.price` cell checked against its
  emitter; the `ef.*` bullets; declared raw readers; the bound paragraph; the activation-floor paragraph); GLOSSARY
  (Status + "`ev.idx`/`ev.price` are not a bare-element pair", `confirmed_at` — also its stale "absent on every
  CTS_UPDATED" since E3·0 —, moment, `event_moment`, `event.idx`, `CTS_ESTABLISHED`, `cts_moment`, parent
  `cts_anchor_idx`); GOTCHAS (the "BOS_CONFIRMED `ev.idx`…" convention bullets, BIB scan-back, exception window +
  its code snippet → `ef.cts_anchor_idx(…) + 1`, proximity scan, CTS_THRESHOLD note); LANDMINES (rule 3's E4a use +
  the per-type test contract + the `mutate=` rule, Scenario 3 constraint 4, Event Sort Order, parent-tables
  assert, cycle start, knowable-at clip); PRE_REFACTOR_INVARIANTS; WORKFLOWS; KL / POI / FIB_LIFECYCLE /
  WAVE_CANDLES (the event walk is `ef.cts_anchor_idx`, not `ev.idx`) / MARKET_STRUCTURE specs; PART4 (5
  present-tense rules + the 2026-09-19 note); IMBALANCE_FILL_SEMANTICS; production comments (`event_fields`,
  `knowable_at_idx`, the orchestrator sort, the KL docstring, the mirror clip, `parent_tables`, `poi_zones`,
  `structure_lifecycle`); test-helper comments ("the `idx` argument is the anchor"). Dated history left as is.
  `e4flip_plugin.py` `FLIP=est` is a no-op from here on (README).
  - Landing review, conformance lens (≈294k): 0 BLOCKER; verified the single emit site, that nothing re-stamps an
    EST (rewind replays from scratch; mirror / sibling clip shift `idx` and `confirmed_at` by one offset; the pool
    clip keys the moment), every production `.idx` read type-filtered away from EST or a declared raw reader, no
    read silently correct only because anchor == moment, KL `source_event_idx` never written for an EST (the EST
    branch `continue`s first), and every cell of the re-cut ARCHITECTURE table against the emitters. **MAJOR
    (fixed):** `test_pattern_anchor_idx_values_obey_the_pattern_bound` (not in the diff) had become half-vacuous —
    `pa <= ev.idx <= conf` read `conf <= conf` → now `pa <= ef.cts_anchor_idx(ev) <= conf <= pa + 5` and
    `ev.idx == conf`; MARKET_STRUCTURE_SPEC's Scenario-3 "Mechanics" window (`cts_est[0].idx + 1`) contradicted
    the fixed lines ten below; a GOTCHAS next-BOS sentence ("its `.idx` is the CTS extreme"); the tracked `/compare`
    skill's events paragraph (now split by type: EST → track `cts_anchor_idx`). **MINOR (fixed):** stale
    present-tense comments (`structure_lifecycle`, `parent_tables` ×2, `_second_cts_moment`, `structure_engine`'s
    exception-window comments — incl. the retracted "that candle is the pullback confirmation" —,
    `probe_fc_finalize`); `stamped_idx`'s use list (the E3 marker bullet → the H1-overlay window start, also in
    ARCHITECTURE); test docstrings (`test_unified_probe` fixture + phase-2 comment, `test_ms_stop_after_cts`,
    `test_parent_tables` negative control "idx 10" → "cts_anchor_idx 10" — its test still kills the anchor
    mutant —, `test_structure_lifecycle_moment`, the POI sweep comment, the H5 docstring); three test `_sorted`
    helpers now use `ef.processing_order_key` (they modelled the orchestrator with a raw `(idx, type)`, which
    differs for a lagging EST since E4a; no assertion depended on it); the emitter-assert pin skips under `-O`;
    two history lines disambiguated (ARCHITECTURE activation floor, LANDMINES 2026-05-13 case). Tests 905 + 1 xfail.
  - Landing review, mutation lens (≈229k; 22 mutants on a scratch copy, full suite each): the contract core is
    pinned — the emitter revert (33 kills), its assert (1: the emitter-assert pin — without it the validator
    still raises, but as `EventContractViolation`), the factory / conftest / validator, `ef.stamped_idx` /
    `cts_anchor_idx` for EST, and even `event_moment` for EST → `ev.idx` (equivalent on legal events, killed by
    the deliberately illegal swap clones and "anchor"-shape order pins). **Survivors → 6 pins adopted
    (`apply_pins.py`, each verified to fail on its mutant; wave-candle and POI pre-window re-killed here):**
    P1/P2 the Plan D `mutate=` hooks re-validate the contract after the edit (dropping `est.idx = …` left an
    illegal event nothing noticed); P3 `test_cts_bib_event_walk_checks_a_lagging_est_at_its_anchor` — the
    wave-candle BIB walk's `ev_idx` read as `ev.idx` survived (every wave fixture has lag 0; only the AST sort
    guard caught the sort-key half); P4 the POI sweep's PRE-WINDOW cond1 branch (`lifecycle_floor_idx` past the
    moment; the existing pin reached only the in-window branch) + its positive control; P5 the H1 chart's
    "CTS (unconfirmed)" marker at the CTS anchor (no pytest coverage before); P6 the Scenario-3 exception window
    opens at the CTS_0 anchor + 1 (Q10 test-only path). Equivalent: the Plan-B-save CSV loader translation (all
    5 H1 ESTs lag 0). N/A: no EST location read in `first_confluence_trigger` / `kl_zones_v1`. Declared residual:
    the two M15-chart twins of P5 (`_build_sub_polylines`, `_render_h1_overlay`) read `ef.cts_anchor_idx` and are
    covered only by the figure diff (as accepted in the E2c review). Side note (pre-existing, hygiene list): the
    `compute_structure_scenario_3` docstring still names `_run_h1_reverse_probe`, which no longer exists (fixed
    2026-09-28, hygiene 5c).
    Tests 905 → **910 + 1 xfail**. Commit `e583f8a`; save `20260925_121648_e583f8a`.

- **E4b-pre (2026-09-25; Q21).** `kl_zones_v1.derive_kl_zones_v1` no longer writes the KL meta
  `source_event_idx` (the source event's raw `ev.idx`, write-only — the only KL-side raw reader). Grep-verified
  no reader: every other `source_event_idx` in the code is the separate `ReferenceZone.source_event_idx` (the
  probe input; E5 renames it). **Measured (vs `20260925_121648_e583f8a`, `--strip source_event_idx`) == Q21's
  prediction:** 39 key-only KL meta cells (H1 10 / conf 21 / counter 8), **0 real cells**, figures
  JSON-identical, run.log only the parked `by_lens` order. Tests: the guard's `KNOWN_SLICE_LOCAL_ZONE` drops the
  key; `test_e4_simulation`'s BOS swap now compares the KL output WHOLE (no exclusion) — so a raw-`ev.idx` KL
  key re-added for a BOS zone fails it. Docs: KL_ZONES_SPEC (field list → a dated removal note; the stale
  "Creating a zone" steps rewritten to the accessors — `confirmed_idx = ef.event_moment`, anchors via
  `ef.bos_anchor_idx` / `ef.cts_anchor_idx`, no fallback), ARCHITECTURE declared raw readers, the
  `kl_zones_v1` docstring. Tests 910 + 1 xfail. E4b therefore changes only the events CSV `idx` column + the KL
  prints (§8 E4b "KL: 0"). Commit `6bcd2e9`; save `20260925_124105_6bcd2e9`.
  - Landing review (1 combined lens, ≈121k; scratch copy built from the commit): 0 BLOCKER / 0 MAJOR. No reader
    of the KL key anywhere (skills and plan tools included; `_shift_meta_indices` never listed it); the
    rewritten zone-creation steps match the code sentence by sentence; rule 2 covers EVENT meta, a KL zone's meta
    is zone meta. Mutation: re-adding `"source_event_idx": int(ev.idx)` is killed by 9 tests (the 8 BOS / both
    swap cases + the zone-meta guard). MINORs folded in the E4b commit: KL_ZONES_SPEC's `confirmed_idx` bullet
    still claimed a "fallback `ev.idx`" (→ `ef.event_moment`, no fallback), a stale "module docstring … `.py`
    follow-up" sentence, the removal note's wording (the raw, unclamped confirm candle) and a stray `*`;
    memory reconciled.

- **E4b (2026-09-25).** Both `_emit_bos_confirmed` call sites (cycle 0 `initial_prior_extreme`, cycle ≥ 1
  `pullback_extreme`) pass `int(apply_idx)`; the emitter asserts `idx == meta["confirmed_at"]` before building the
  event; `st.bos` stays built from the `bos_anchor_idx` parameter. **Measured (vs `20260925_124105_6bcd2e9`) == §8
  exactly:** 34 events `idx` cells — H1 96→115, 591→652, 689→703, 728→748, 826→902; conf 21; counter 8 (lag 2–411)
  — every one in the `idx` column; the 3 figures JSON-identical; run.log only the `[kl_zones]` BOS prints (both
  forms; the idx moves, the anchor stays) + the FutureWarning line number. Replay 104.6 s wall (machine variance;
  46–105 s this session). Contract flip in the same commit: `EVENT_IDX_IS` both "moment"; the factory BOS default
  `idx = confirmed_at`; `test_e4_simulation`'s emitter pin covers both types (BOS lags (0,2), (7,10) on the
  second-CTS fixture, (0,2), (7,10), (12,15), (17,20) on the multicycle one) + a BOS emitter-assert pin, and the
  swap returns BOS to its anchor; the E4-shaped BOS pins lost their `illegal_event_contract` markers; the
  `mutate=` hooks move the BOS idx too (their E4a re-validation caught it); three tests that read a BOS's raw
  `.idx` as its anchor moved to `ef.bos_anchor_idx` (the render-projection straddle + KL floor tests) and the
  Plan-B-save CSV loader translates pre-E4b BOS rows. Tests 910 → 913 + 1 xfail. Commit `7216c9d`; save
  `20260925_125921_7216c9d`.
  - **Pre-review mutation loop (mine, per the E4a lesson; `plan_e_inputs/review_scripts/mut_loc_e4b.py`):** each of the 12 BOS
    LOCATION readers switched from `ef.bos_anchor_idx(X)` to the raw `int(X.idx)` (the moment since E4b), full
    suite each: 7 killed (FC input, prev-BOS start, fib BOS point, structure levels, KL anchor, `stamped_idx`, the
    H1 BOS dot — incidentally), **5 chart sites survived** (the M15 sub-chart BOS dot + PB→BOS line end, the H1
    overlay's twins, the H1 chart's PB→next-BOS line end) — only the figure diff guarded them.
  - Landing review, mutation lens (≈197k; copy from the commit): all 5 survivors now killed by one pin each + an
    explicit H1 BOS-dot pin (a hand-built sid-handoff stream with lagging BOS 2/5 and 13/16, through
    `_build_sub_polylines`, `_render_h1_overlay` and `export_chart_plotly`; each pin kills only its site). The 10
    contract mutants all killed (emitter revert 193, its assert 1 — only that pin; factory 39; conftest 229;
    `stamped_idx` 22; `bos_anchor_idx` 50; `event_moment` 8 — equivalent on legal events, caught by the
    deliberately illegal swap clones; both hooks via their re-validation; `st.bos` from `idx` 2). Own mutants:
    **D1 survived** — zone_proximity's next-BOS scan bound read as the anchor (the existing test had anchor ==
    moment; the swap test cannot see a meta-only read; pre-existing since before E4b) → pinned
    (`test_scan_window_ends_before_the_next_bos_moment_not_its_anchor`, re-killed here); D2 (KL BOS
    `confirmed_idx` → anchor) and D3 (the threshold timeline sort) killed; `uc1_trigger` / WVMI `on_bos_confirmed`
    read no BOS index. Tests 913 → **920 + 1 xfail**.
  - Landing review, conformance lens (≈304k): 0 BLOCKER; no production location read became wrong (every raw BOS
    `.idx` read is type-filtered away or a declared raw reader; all timing reads `ef.event_moment`, all location
    reads `ef.bos_anchor_idx`); the mirror / sibling clip shift `idx`, `confirmed_at` and `bos_anchor_idx` by one
    offset; nothing correct only by this window's data (one pre-existing latent: `_select_bos_on_breakout`'s
    swap branch could put a BOS anchor after its apply candle (asserted against at the emit since 2026-09-28) → the anchor-keyed sort would run EST before BOS —
    34/34 anchors <= moment here). **MAJOR ×2 (E4a leftovers, fixed):** KL_ZONES_SPEC "Why the two definitions
    can differ" and PART4 ×2 still said the EST `ev.idx` is the extreme. **MINOR (fixed):** `stamped_idx` /
    GLOSSARY `event_moment` / LANDMINES sort-order / ARCHITECTURE "verified" note still E4a-only;
    `BOS_THRESHOLD_UPDATED → ev.idx` (E3g-2) missing from the three `event_moment` lists; POI_ZONES_SPEC bare "BOS
    idx" (now the anchor, named); a `test_ms_stop_after_cts` comment ("a later BOS idx can be <= stop"); five
    test-helper comments EST-only; the render-projection oracle docstrings + the straddle test renamed
    `…_clipped_by_its_moment_not_its_anchor`; `reference_zone` docstring (EST → the anchor accessor); the review
    README (`FLIP=bos` is a no-op too). **Hardening:** both BOS emit sites now pass `"confirmed_at": int(apply_idx)`
    like the EST site (a numpy value would shift `idx` but not `confirmed_at` in the mirror). Replay after the
    fold-ins: 0 cells, figures + run.log identical; tests 920 + 1 xfail. Fold-in commit `19a6391`.

- **E4c (2026-09-25; Q1 = flip).** The pattern-path `_emit_cts_updated` call passes `int(apply_idx)` as `idx` with
  meta `confirmed_at` (E3·0) and the new `cts_anchor_idx`; `_emit_cts_updated` asserts, on the pattern path, `idx
  == confirmed_at` and `cts_anchor_idx <= idx`; the raw path is unchanged (idx = the processing candle, anchor and
  moment at once, neither key). `ef.cts_anchor_idx` on `CTS_UPDATED` branches on `via` (raw → `ev.idx`, pattern →
  `meta["cts_anchor_idx"]`), so every E2 location reader — FibTracker's update handler, the POI sweep (both
  branches), the wave-candle walk, the reference zone, the prev-BOS line END, `structure_levels`, the chart
  unconfirmed-CTS markers — and `stamped_idx` (the processing order) keep the anchor. `st.cts` stays built from
  the local anchor. **Measured (vs `20260925_125921_7216c9d`, `--strip cts_anchor_idx`) == §8:** **1 real cell**
  (conf sub 2 cyc 0: `idx` 2468 → 2470) + the new key on exactly the 37 pattern-path rows (H1 7 / conf 23 /
  counter 7) and the 7 H1 `structure_levels` rows (their meta copies the event's) — the same footprint as E3·0;
  figures JSON-identical; run.log only the FutureWarning line number and the parked `by_lens` order. Replay 82.2 s.
  Test contract: the validator's pattern path now requires `idx == confirmed_at` and an int `cts_anchor_idx <=
  idx` (raw: neither key) — 3 new reject shapes; `tests/_event_factory.make_cts_updated` (+ `make_event` routes
  `CTS_UPDATED`, its `idx` ARGUMENT the anchor); the fixture helpers of `test_imbalance_c3_knowability` (also
  used by the role pins), `test_e3a_mutation_pins`, `test_first_trigger_migration`, and three hand-built events
  (role pins, pooled build) moved to the moment shape; the MS emit test pins `event_moment == confirmed_at ==
  idx >= cts_anchor_idx` and the lagging (24, 25); the Plan-B-save CSV loader stands in `cts_anchor_idx = idx`.
  31 tests failed on the bare flip, every one a fixture shape. Tests 920 → 923 + 1 xfail. Docs: ARCHITECTURE
  (intro — `ev.idx` is now the moment on every type —, the `CTS_UPDATED` row incl. the anchor column, the
  `ef.cts_anchor_idx` / `stamped_idx` bullets), GLOSSARY (moment, `event.idx`, `confirmed_at`, `CTS_UPDATED`),
  GOTCHAS, LANDMINES knowable-at clip, PRE_REFACTOR_INVARIANTS, WORKFLOWS, IMBALANCE_FILL_SEMANTICS,
  WAVE_CANDLES_SPEC, the `/compare` skill, code comments. Commit `c182eb6`; save `20260925_133849_c182eb6`.
  - **Pre-review mutation loop (mine; `plan_e_inputs/review_scripts/mut_loc_e4c.py`):** each of 14 CTS location reads switched to
    the raw `ev.idx` for CTS_UPDATED only, full suite each: 5 killed (the reference zone, FibTracker's update
    handler, the POI pre-window branch, `stamped_idx`, and the wave-walk sort — by the AST guard only), **9
    survived** — the 4 chart unconfirmed-CTS sites, the prev-BOS line END, `structure_levels`, the POI in-window
    cond1, the wave walk's `ev_idx` / `next_ev_idx`: before E4c `ev.idx` WAS the anchor on the pattern path, and
    almost no fixture has a lagging pattern-path update → the landing review's combined lens writes the pins.
  - Landing review (1 combined lens, ≈272k; copy from the commit): 0 BLOCKER. **All 9 survivors pinned** (lagging
    pattern-path updates through `make_cts_updated`: the sub-chart unconfirmed-CTS filter + dot, the H1 overlay
    dot, the H1 chart marker, the prev-BOS END on a pattern-update winner, `structure_levels`, the POI in-window
    cond1 (IC between anchor 5 and moment 7), the wave walk `ev_idx` and `next_ev_idx` — the latter nearly
    equivalent: it differs only on an unconfirmed cycle, where the correct walk never scans past the last
    anchor; U10 re-killed here). Contract mutants: emitter revert 3 kills, **the E4c assert unpinned (V2
    survived)** → `test_emit_cts_updated_asserts_the_pattern_path_idx_is_the_moment`; accessor branch 20;
    validator 1; factory default 17. Plus the E4c swap case (lagging fixture back to its anchor → downstream
    outputs identical; also kills the FibTracker update read). **MAJOR (docs, fixed):** the new "`ev.idx` is the
    moment on every event type" (ARCHITECTURE intro, GLOSSARY moment / `event.idx`) contradicted the table's
    `REVERSAL_CANDIDATE` row (pattern-first candle; moment `meta["apply_idx"]`; `event_moment` raises on it) →
    scoped to the CTS / BOS family with the exception named. **MINOR (fixed):** the `CTS_UPDATED` row's `ev.idx`
    cell opened "the new CTS anchor"; the mirror-test docstring ("keeps the RAW idx (the anchor)"); a garbled
    mirror-clip comment; the `/compare` skill's anchor-stamped sentence (pre-E4 saves only). Tests 923 → **935
    + 1 xfail**. The fold-in changes tests, docs and one comment only.

## 9. E5 — remaining renames + prose

- **Scope:** IN §1.3 (E5 rows), §1.4 (E5 rows), §1.5 (E5 rows), §1.6, §1.7, the §1.9 prose, §3 #13–15, and
  `cts0_est_idx` → `cts0_established_idx`.
- **Internal renames are byte-identical:** 24/24, run.log identical; the `[fib] CTS idx=` labels stay (IN §1.10).
- **Exported:** `validated_parent_idx` per Q4 (triggers CSV header, 2 files).
- Chart hover / legend text stays out of byte-identical steps.

**Post-E (not Plan E):**
- the coordinate-hygiene families (IN §2.8 last row; one `/compare` each). **Census at HEAD (2026-09-26, save
  `20260926_122844_532df17`; M15 rows confluence / counter; values slice-local, e.g. sub 0 `slice_begin` 404):**
  F1 `STATE_CHANGED.effective_idx` 294 / 94 (event 458 → 54); F2 `RANGE_STARTED` `start_idx` 32 / 13, `cts_idx`
  39 / 15, `confirm_idx` 32 / 13, `pullback_apply_idx` 7 / 2 (`proximity_apply_idx` 0); F3 `REVERSAL_WATCH_START` /
  `REVERSAL_CANDIDATE.expires_idx` 7 + 4 / 0 (1537 next to event 1936); F4 `BOS_CONFIRMED.pb_start` 14 / 5; F5 KL
  meta `expanded_last_idx` 4 / 0; F6 POI meta `bos_idx` / `cts_idx` 30 + 30 / 12 + 12 (50 / 380 vs the fib CSV's
  454 / 784); F7 fib meta `activated_at` 21 / 8, `locked_at` 19 / 6, `reactivated_at` 1 / 1 (`cycle1_bos_idx` 0);
  F8 `WaveCandleResult.meta["anchor_idx"]` (attrs only, 0 CSV cells). KL `source_event_idx` is gone (E4b-pre). Not a
  family: WVMI `triggered_by_event_idx` (the parent trigger's H1 idx 710 / 926 / 1020 — a frame-name question).
  Guard `tests/test_event_meta_idx_keys.py`: `KNOWN_SLICE_LOCAL` = the F1–F4 keys; `KNOWN_SLICE_LOCAL_ZONE` =
  `bos_idx` / `cts_idx` commented "KL zone" — on this window they appear in POI meta (check); F5 / F7 not covered.
  **DONE 2026-09-26 (Post-E·2, §9.3): all eight families shifted in place in one commit; the allow-lists are gone.**
- the never-established-cycle fallback POI (0 on this window since Plan F);
- the fetch-gate N/A edge case; **DONE 2026-09-27 (fetch-integrity commit, with the loud fetch):** user picked
  gate-only N/A — the snippet reads N/A from the log when no M15 fetch ran (no `[data_bridge]` line): no
  `[multi_tf:dual]` line (no lower TF) or the ASCII prefix `[multi_tf:dual] no triggers` (its `—` is cp1252 `0x97`
  in run.log). `data_bridge.fetch_lower_tf_data` retries a failed chunk request twice, then raises; an all-empty
  fetch raises. Canonical: compare skill §2b; pins `tests/test_data_bridge_fetch.py`;
- moment-order processing (Q3);
- the unresolved-triggers CSV `probe_input_idx` frame mix (H1 "whatever was known" for the parent-triggered
  types vs M15) — split into `parent_input_idx` (H1) + `probe_input_idx` (M15, when the resolver got one — §9.2); one exported
  `/compare` (user 2026-09-26: "fix later"). **DONE 2026-09-26 (Post-E·1a + 1b, §9.2).**
- the `probe_end_idx` name / frame split (found in the Post-E·1a review): `MultiTFTrigger.meta["probe_end_idx"]` /
  `FirstConfluenceTrigger.probe_end_idx` hold the H1 CTS anchor, `unified_probe(probe_end_idx=)` / the probe cache's
  (`ProbeCacheEntry`) the M15 bound (`ProbeResult` has no such field — corrected in the Post-E·3 review) — the `parent_input_idx` / `probe_input_idx` pattern would give the H1 one a `parent_*` name
  (byte-identical rename; FC only). **User 2026-09-27: `parent_cts_anchor_idx`** (says what it IS — the parent
  CTS anchor, FC(0,0) = 430 vs the M15 bound 1721, (0,1) 652 / 2609, (1,2) 905 / 3621 — and matches the FC
  resolver's local of that name and its sibling `parent_bos_anchor_idx`); rejected `parent_probe_end_idx` (hides
  that it is the anchor). No CSV carries it. Post-E·3 (§9.4).
- the run.log `[multi_tf:dual] sub wvmi … by_lens={…}` key order — a false delta in every run.log diff ("the parked
  `by_lens` order" in the entries above): `orchestrator._assign_sub_wvmi_per_sub` counted over the sub's lens SET, so
  the dict's insertion order followed the per-process string hash seed. **DONE 2026-09-26:** counts over
  `sorted(lenses)` (as the persist loop above it already did). Measured vs `20260926_122844_532df17` == prediction:
  24/24 CSVs byte-identical, the 3 figures JSON-identical, run.log only that line, now `by_lens={'confluence': 7,
  'counter': 6}` in every process. Pin `test_sub_wvmi_per_sub.py::test_by_lens_key_order_is_sorted_not_set_order`
  (the orchestrator's `set` swapped for a reverse-iterating one — kills the bare `for l in lenses` deterministically,
  not only under an unlucky hash seed). Tests 944 + 1 xfail.

### 9.1 E5 as landed

- **E5·1 — internal renames (2026-09-25; byte-identical).** Every IN §1.3–1.7 E5 row re-grepped at `47e6a66`.
  Q13 = no optional renames (not done: MS local `cts_anchor`, the `"EXT"` point tag, `continuous()`'s own
  `extreme`); Q14 reversal-scope names separate.
  - §1.3: `TrueFirstBreakout.extreme_idx` / `extreme_price` → `pattern_extreme_idx` / `pattern_extreme_price`;
    `_full_pattern_extreme` deleted — ONE `patterns/structure_patterns.pattern_extreme(highs, lows, pat, direction)`
    (numpy arrays; an undefined / out-of-bounds span → None) behind both `find_true_first_breakout` and
    `MarketStructure._cts_from_breakout_event`, now a thin adapter (asserts in bounds; returns `(cts_anchor_idx,
    cts_price)`) — the only place a pattern extreme becomes a CTS anchor; the TFB locals and the
    `_is_strict_new_extreme` params → `pattern_extreme_idx` / `_price`; the `extreme_candle` pseudo-id + its test
    name; `close_to_ext` → `c1_close_near_c0_extreme`.
  - §1.4: fixture `_make_second_cts_moment_after_anchor_data` (5 test files); the 10 test names that contrasted the
    moment with "the extreme" → `…_not_the_anchor` / `…_anchor_equals_moment` / `…_probe_end_idx_is_cts_anchor_…` /
    `…_record_anchor`.
  - §1.5: `st.cts_confirmed_for_idx` → `confirmed_cts_anchor_idx`; identify_start scenario-2 meta
    `last_confirmed_cts_anchor_idx` / `last_confirmed_cts_price` + local `cts_anchor_idx` (no reader, not exported;
    the df column `cts_idx` it reads stays); `ReferenceZone.source_event_idx` → `anchor_idx` (incl.
    `_zone_to_reference`'s param; PART4 ×13, LANDMINES ×3); FC trigger local `probe_end_idx`; FC resolver
    `parent_cts_anchor_idx` / `parent_cts_anchor_time` / `m15_probe_end_idx` (the projection meta key
    `"m15_end_idx"` — the lifecycle end — stays); sibling fallback `fallback_anchor_idx`; uc1 local
    `cts_anchor_idx`; `debug/probe_fc_finalize` `bos_anchor_h1` + header `BOSanch`; `cts0_ref` → `bos0_ref`.
    Superseded rows: the sibling-clip `abs_idx` (E3b named it `clip_idx`); uc1's `.get` fallback and meta
    `cts_idx` (gone since E1b / E2d).
  - §1.6: zone_proximity `bos_confirmed_idx_by_key` / `next_bos_confirmed_idx`; `compute_cycle_lifecycle`
    `cts_established_idx_by_key` / `cts_established_idx`; wave_candles `raw_cts_confirmed_idx`; **plus** the three
    chart `next_bos_idx` locals (a MOMENT since E3g-3 — IN §3 #15's row was superseded) → `next_bos_confirmed_idx`,
    one name for one role (closes the #15 collision).
  - §1.7: `_replacement_break_point`'s `ext_end_idx` → `extend_to_idx`.
  - `cts0_est_idx` → `cts0_established_idx` (`unified_probe` results + tests; ARCHITECTURE, PART4). The run.log
    label `cts0_est=` is kept, so run.log stays identical.
  - Not done (outside §1.3–1.7): the §1.2 `debug/zone_proximity_diag.py` CSV columns (`cts_idx` /
    `this_cts_idx` / `window_span_post_cts`, /tmp output only) — **user 2026-09-25: "leave the diag columns as
    is"**.
  **Measured (vs `20260925_133849_c182eb6`):** 24/24 CSVs byte-identical, the 3 figures JSON-identical, fetch
  gate PASS; run.log only the FutureWarning line number (MS 16 lines shorter) and the parked `by_lens` order.
  Replay 45.4 s wall. Tests 935 + 1 xfail (unchanged). Docs: GOTCHAS "`_cts_from_breakout_event`: Include
  Confirmation Candle" names the shared function; memory names reconciled. Commit `029401f`; save
  `20260925_145355_029401f`.
  - Landing review (1 combined conformance lens, ≈204k; full suite on a `git archive` copy; 20,000-case fuzz of
    `pattern_extreme` against the old TFB helper and the old MS body: 0 differences; the new MS assert is
    unreachable — MS patterns come only from `BreakoutPatterns(df, end_idx=_effective_end)`, bounded by
    `n_visible`): 0 BLOCKER / 0 MAJOR. **MINOR / NIT (folded in):** `review_scripts/e4sim.py` still imported the
    old fixture name; the `extreme_candle` pseudo-id in GOTCHAS, PART4 and the TFB design memory; this status
    line; GLOSSARY `anchor_idx` now names the `ReferenceZone.anchor_idx` field; a test stub's `extreme_idx`
    param; the "sanity-assert" claim on `cts0_established_idx` (no production reader — informational);
    `_cts_from_breakout_event`'s span wording (`max(end_idx, confirmation_idx)`). Deferred to E5·2/E5·3: the
    `ReferenceZone` docstring defines `anchor_idx` only for the `cts_*` sources (omits `ad_hoc_bos_0`, the
    common case; the value is slice-local on the reversal path, entity-absolute on the sibling path).
- **E5·2 — IN §3 #13 dead code (2026-09-25; byte-identical).** `_is_new_cts_extreme` / `proximity_confirmed_idx`
  were already gone (E1b). Deleted: the never-constructed `ReferenceZone.source` literals `parent_cts` /
  `parent_bos` + their docstring bullet; the M15 chart's `_find_m15_by_extreme` — **user 2026-09-25: "Delegate"**:
  the H1 zone-proximity markers now call `data_bridge.map_candle_to_lower_tf(h1_time, -1 if approach_from_above
  else 1, dfx)` (lazy import, as in `entity_df_mutation`), one price-extreme mapper (ARCHITECTURE / PART4 §17.4 /
  GLOSSARY LOH: "used only for the FC probe's structural inputs — and, for display, the M15 chart's H1
  zone-proximity markers"); the 41-line commented `_cts_price_at` caller block E1b missed in
  `_apply_pattern_at_apply_idx` — **user: "E1b leftover only"**; the other four commented-out legacy blocks in
  `market_structure.py` (the `i in (387, 388)` debug print, the old `_initial_bos_before_first_cts(cts_idx)`, the
  old `_emit_bos_confirmed` setting the retired `st.bos_confirmed`, the old `_select_bos_price_on_breakout`) →
  the dead-code hygiene list (DELETED 2026-09-28, hygiene 5c, with their commented call sites; the commented old
  `_emit_cts_established` / `_emit_cts_updated` bodies were not on that list and remain). **Measured (vs `20260925_145355_029401f`):** 24/24 CSVs byte-identical, the 3
  figures JSON-identical (incl. the 12 delegated markers, 6 per M15 chart; no empty hour, so no new
  `[data_bridge] WARNING`), fetch gate PASS; run.log only the FutureWarning line number (2648 → 2607). Replay
  46.2 s. Tests 935 + 1 xfail. Commit `9ac70ba`; save `20260925_152746_9ac70ba`.
  - Landing review (1 conformance lens, ≈125k; `git archive` copy; the old vs new mapper fuzzed on 34,684 cases
    — gaps, ties, NaN, shifted / sliced indices, string times: 0 differences; flipping the sign moves all 12
    markers, so the figure diff pins the direction): 0 BLOCKER / 0 MAJOR. **Folded in:** the mapper's own
    docstring still claimed one "universal" `-lower_sd` rule (the FC `probe_end_idx` uses `+lower_sd`; the
    chart passes the wick side) → names both callers; the chart comment notes that a SLICED chart now logs a
    `[data_bridge] WARNING` per clipped trigger (production never slices); the four kept MS blocks joined the
    memory hygiene line. Deferred to E5·3 (prose, listed below).
- **E5·3 — prose + Q12 (2026-09-25; byte-identical).** **User 2026-09-25: "Full present-tense sweep"** of "CTS / BOS
  extreme" where the value is an anchor. Code (~35 comment / docstring lines): MS (range seed, CTS_CONFIRMED price
  comment, cycle-0 data, `_select_bos_on_breakout`, the stale "new-extreme check below" breakout comment, "clear
  pullback anchor" (a moment), `_initial_bos_before_first_cts` "Cycle 1 BOS" → BOS_0 anchor — MS line count kept, so
  run.log stays identical), `reference_zone` (module docstring: the retired "parent BOS / CTS zone" design → the
  current callers; `ReferenceZone.anchor_idx` defined for every source incl. `ad_hoc_bos_0` + its frame;
  `_zone_to_reference`), `structure_engine`, `entity_df_mutation`, the first-confluence trigger / pipeline /
  `types`, `uc1`, `parent_tables`, `unified_probe`, `fib_tracker`, `cross_cycle_fib`, KL `base_idx` ("first candle"),
  `wave_candles` `anchor_idx` param, both charts. Docs (present tense; ~60 lines): PART4 (§4.3 tables, §17.4 row,
  §17.8), GOTCHAS (incl. heading "…Not the CTS Anchor"), LANDMINES (incl. a bound still written `<= idx <=`
  since E4a → `cts_anchor_idx`), PRE_REFACTOR_INVARIANTS, MARKET_STRUCTURE_SPEC, KL / FIB / CROSS_CYCLE_FIB /
  WAVE_CANDLES / CHARTING specs, GLOSSARY; kept: "the pattern's extreme candle" (pattern realm, where an anchor
  comes from), raw "wick extreme", price searches, dated history / bug narratives. GLOSSARY Naming-Standard
  "Status" rewritten (the standard holds; kept-by-decision list; history's "extreme" = anchor). §3 #14:
  CROSS_CYCLE_FIB_SPEC's stale `function:line` refs (9) dropped; LANDMINES "`parent_extreme_dir` Must Use
  `-trigger.lower_sd`" named deleted / renamed call sites AND overstated its scope → probe-INPUT mappings only
  (the FC `probe_end_idx` map uses `+lower_sd`, the chart markers the wick side). **Q12:** `KLZone.source_time` =
  the source event's MOMENT (raw, unclamped `ef.event_moment`) next to `source_price` = the anchor's price —
  documented in `common/types.py`, KL_ZONES_SPEC "Zone indexing" (H1 BOS zone sid 0 cyc 0: time of 115, price of
  96) and ARCHITECTURE (with `StructureLevel.time` = the anchor's time). **Measured (vs
  `20260925_152746_9ac70ba`):** 24/24 byte-identical, figures JSON-identical, fetch gate PASS, run.log identical.
  Replay 46.1 s. Tests 935 + 1 xfail. Commit `6c37053`; save `20260925_165323_6c37053`.
  - Landing review (1 conformance lens, ≈255k — over the 150–200k estimate): 0 BLOCKER; every rewritten sentence
    true to the code (Q12 example verified in the save: `source_time` 2025-11-21 17:00 = candle 115, `source_price`
    0.55808 = candle 96's low). **MAJOR (folded in):** the sweep's regex (`CTS/BOS extreme`, `extreme idx|candle`)
    missed bare forms ("the extreme", "extreme ==", "(the extreme)", "extreme precedes", "the price extreme") —
    ~40 present-tense sites: the tracked `/compare` skill + WORKFLOWS, KL_ZONES_SPEC ×9, LANDMINES ×7, GOTCHAS ×5,
    PART4 ×15 (incl. the §17.6 heading → "…MOMENT, not the anchor" and "the moment-not-anchor rule"), FIB ×5,
    WAVE, `export_plotly`, `probe_fc_finalize`; the GLOSSARY Status claim was false until then. **MINOR (folded
    in):** two wrong FACTS stale since E4a / E4b — FIB_LIFECYCLE_SPEC "not the extreme `.idx`" and LANDMINES's
    rendered-candle table "BOS_CONFIRMED dot | `ev.idx`" (the charts draw it at `ef.bos_anchor_idx`); the
    `ReferenceZone.anchor_idx` docstring said "the probe's `input_idx` (hence the pool key)" — not an input for
    main sid 0 / the moving BOS_0, and the input feeds the probe-CACHE key; the module docstring's first lines +
    a stale `_build_sibling_cts_ref_zone` name; the Status kept-list (+ `ParentTables.cts_moment`,
    `TrueFirstBreakout.est_idx`, the pending E5·4). **NIT (folded in):** `_select_bos_on_breakout` ("Cycle k>1"
    → ">= 1"; its `bos_idx` locals → `bos_anchor_idx`, an E2 row's leftover); CROSS_CYCLE_FIB_SPEC's `file:line`
    refs → function names; ARCHITECTURE's `StructureLevel.meta` = `{"event": ev.type, **ev.meta}`; the
    `first_confluence` frame. Correction to the bullet above: a few history-adjacent parentheticals (the PART4
    blockquote near "was then a DIFFERENT thing", LANDMINES "What was not foreseen", PRE_REFACTOR "before it
    …") WERE reworded, meaning unchanged (reviewer-checked). Lesson: a prose sweep's regex must include the
    bare-noun forms; classify EVERY `extreme` hit, not a phrase list.
- **E5·4a — Q4 = (b): triggers CSV `validated_parent_idx` → FC-only `parent_bos_anchor_idx` (2026-09-26; EXPORTED).**
  Measured first (user-reviewed table): the value was the H1 BOS anchor on the 3 FC rows (96 / 591 / 826) and the
  M15 probe input on the 4 sibling rows (conf sub 6 3760; counter subs 3 / 5 / 7: 2609 / 3621 / 4000), None on
  reversal rows — and those sibling cells were the ONLY place the CSVs exported a probe's M15 input (the subs CSV has
  no input column; `probe_input_idx` was only in the UNRESOLVED CSV; run.log carries every input, incl. FC 385 /
  2365 / 3304 and reversals 1794 / 2365 / 2609 / 4000). **User 2026-09-26: "(b), then add probe_input_idx"** — E5·4a
  here, E5·4b = the new column. Change: the field on `ResolvedStart` + `TriggerRecord` (→ `_TRIGGER_COLUMNS`, auto)
  renamed; the sibling resolver returns `None` (its M15 input stays `ResolvedStart.probe_input_idx`); FC / reversal
  unchanged; `debug/probe_fc_finalize`; GLOSSARY row (+ the Status "pending" line removed), PART4 §17.4 row,
  LANDMINES probe-cache note, docstrings; tests: the two sibling pins (`test_first_trigger_migration` (1) / (2),
  its PLAN-AMBIGUITY comment resolved) now pin `None`, 41 renamed occurrences in 8 test files, the pass-through stubs
  (sweep, export) keep non-None values with a comment naming the real rule. **Measured (vs
  `20260925_165323_6c37053`, rename map `parent_bos_anchor_idx` → `validated_parent_idx`) == prediction:** both
  triggers headers renamed only; 4 cells → empty (conf sub 6; counter subs 3 / 5 / 7); FC 96.0 / 591.0 / 826.0
  unchanged; the other 22 CSVs byte-identical; figures JSON-identical; run.log identical; fetch gate PASS. Replay
  44.4 s. Tests 935 + 1 xfail. Commit `5d149b0`; save `20260926_104311_5d149b0`.
- **E5·4b — `TriggerRecord.probe_input_idx` → a triggers-CSV column (2026-09-26; EXPORTED, additive).** The M15
  candle the probe started from, for every resolved type (FC: the price-mapped parent BOS anchor; siblings: the
  sibling CTS anchor; reversal: the handoff input) — the sweep copies `ResolvedStart.probe_input_idx`, which all
  three resolvers already set (four return sites). Required field (no default), placed after `parent_bos_anchor_idx`. **User 2026-09-26:
  name it `probe_input_idx`** (= the field it copies) and fix the unresolved CSV later: `UnresolvedTrigger
  .probe_input_idx` is "whatever was known" — H1 for the four parent-triggered types when no probe ran (this
  window: 689 / 728 / 761 / 826), M15 for reversal / a failed sibling probe — the same frame mix Q4 removed from
  the triggers CSV (follow-up below). Tests: the 7 constructors pass it; pins — the sweep copies the resolver's
  value (`test_lifecycle_sweep_unit`, FC(0,0) 385) and the export writes it (`test_sub_tables_export`); the
  column-order pin. Docs: GLOSSARY `probe_input_idx` row (both CSVs' meanings), PART4 §17.4. **Measured (vs
  `20260926_104311_5d149b0`) == prediction (from run.log's probe lines):** conf 385 / 1794 / 2365 / 2365 / 2609 /
  3304 / 3760 / 4000, counter 2609 / 3621 / 4000, the column right after `parent_bos_anchor_idx`, the rest of both
  files identical; the other 22 CSVs byte-identical; figures JSON-identical; run.log identical; fetch gate PASS.
  Replay 44.1 s. Tests 935 + 1 xfail. Commit `76ad347`; save `20260926_105208_76ad347`.
  - Landing review of E5·4a + E5·4b (2 lenses). **Conformance** (≈165k): 0 BLOCKER / 0 MAJOR; every production
    resolver path (FC run + APPROX hit, sibling run + hit, reversal hit + miss) sets an int M15 entity-absolute
    `probe_input_idx`; only the FC resolver writes a non-None `parent_bos_anchor_idx`; nothing reads the triggers
    CSV by position. MINOR / NIT folded in: the GLOSSARY TriggerRecord row lacked `probe_input_idx`; "all idxs
    entity-absolute M15" (`TriggerRecord` docstring, PART4 §17.4) now excepts `parent_bos_anchor_idx` (H1); the
    sibling type's `probe_input_idx` can be the own-frame ad-hoc BOS_0 anchor (PART4 §4.3.4 step 5 — pinned by a
    test, 0 on this window); the unresolved column's rule stated precisely (M15 once the resolver had an M15
    input, else the trigger's H1 value); "this record's probe"; the PLAN_E status line + "Remaining E5"; stale
    test wording / the `_rs(validated=)` kwarg → `parent_bos_anchor=`; `probe_fc_finalize`'s unused local; the
    `ResolvedStart.probe_input_idx` comment (still defaulted None — test stubs rely on it; enforcing it is
    hygiene); "three resolvers (four return sites)". **Mutation** (≈145k; 39 mutants on a `git archive` copy):
    resolvers (FC / sibling incl. fallback / reversal hit + miss), export (omit / move / mis-source the column)
    and the dataclass contract all killed; the "required" field is enforced by dataclass ordering (a default in
    place fails at import; moving it to the tail is caught by the field-order pin). **3 survivors** — the sweep
    copying `probe_input_idx` only for FC / dropping it for reversal / for sibling types (the only sweep-level
    assertion was the FC record) → pin `test_record_copies_probe_input_idx_for_every_resolved_type` adopted
    (verified by the lens to kill all three + M4a–f / M4k). Tests 935 → **936 + 1 xfail**.
- **Remaining E5:** ~~E5·3 = the IN §1.9 prose +
  §3 #14 + the Q12 `KLZone.source_time` documentation + the GLOSSARY Naming-Standard "Status" paragraph + from
  the E5·1 / E5·2 reviews: the `ReferenceZone` docstring (`anchor_idx` defined for every constructed source,
  incl. `ad_hoc_bos_0`, the common case; slice-local on the reversal path, entity-absolute on the sibling path),
  the `reference_zone` module docstring + `_zone_to_reference` still describing the retired "parent BOS / CTS
  zone" design, LANDMINES "both call sites" of `map_candle_to_lower_tf` (names deleted / renamed functions;
  misses the chart caller)~~ (done, above); ~~E5·4 =
  the exported Q4 change (triggers CSV `validated_parent_idx` → FC-only `parent_bos_anchor_idx`; measured table
  first).~~ (done: E5·4a / E5·4b above). **E5 is complete** — what remains is the post-E list in §9.

### 9.2 Post-E·1 — the unresolved-triggers CSV input split (2026-09-26)

- **Traced (HEAD `bec284e`).** `UnresolvedTrigger.probe_input_idx` mixed frames from three sources: (1) the sweep's
  fallback to `SweepTrigger.probe_input_idx` — **H1** for the four parent-triggered types — on `pending`,
  `degenerate_parent_cycle`, a resolver returning `None`, and a `ProbeFailure` carrying `None`; (2) the FC
  resolver's own `ProbeFailure` passed the H1 `parent_bos_anchor_idx` on three branches
  (`entity_df_mutation.py:649/652/657` — at 652/657 the M15 input was already mapped); (3) every other failure
  branch passed M15. So one reason (`probe_failed`) could carry either frame.
- **Only FC price-maps its H1 input into the probe.** For `first_counter` (H1 CTS anchor, `uc1_trigger.py`) and the
  two `subsequent_*` types (H1 window extreme) the H1 input is informational: the sibling resolver never reads it —
  it co-sources its M15 input from the sibling CTS (or the M15 window-extreme fallback).
- **User 2026-09-26: shape A** — `UnresolvedTrigger` gains `parent_input_idx` (H1: the parent trigger's input
  candle, every parent-triggered type; `None` for `reversal`) and `probe_input_idx` becomes M15-only (the input the
  resolver mapped / co-sourced; empty when none was). The FC `ProbeFailure` branches pass M15 or `None` (649 →
  `None`, 652 / 657 → the mapped M15 input). **"Both, 2 commits"**: `SweepTrigger.probe_input_idx` →
  `parent_input_idx` in the exported commit (Post-E·1a); `MultiTFTrigger.meta["probe_input_idx"]` (H1) →
  `"parent_input_idx"` in a byte-identical commit right after (Post-E·1b) — afterwards `probe_input_idx` names the probe's own-frame input: M15 on every multi-TF field, meta key and CSV column (its one other use, `structure_engine`'s local, is the main H1 structure's reversal probe — its own frame). Rejected: B (mirror the triggers CSV — drops 761 / 826), C (no H1 column).
- **Prediction (vs `20260926_105208_76ad347`):** `*_M15_unresolved_triggers.csv` only — +1 header
  (`parent_input_idx`, between `direction` and `probe_input_idx`); the 4 rows (all `degenerate_parent_cycle`, no
  probe ran): `parent_input_idx` 689 / 728 / 761 / 826 (FC(1,0) BOS_0 anchor, FC(1,1) BOS_1 anchor, first_counter
  (1,1) cycle-1 CTS anchor, subsequent_confluence (1,1) window extreme = BOS_2's anchor), `probe_input_idx` 689 /
  728 / 761 / 826 → empty. The other 23 CSVs, the 3 figures and run.log unchanged (no log line prints either
  input).
- **Post-E·1a measured (2026-09-26) == prediction.** 23/24 CSVs byte-identical; the unresolved CSV: header
  `…, direction, parent_input_idx, probe_input_idx, reason, detail`, `parent_input_idx` 689 / 728 / 761 / 826,
  `probe_input_idx` 4 cells → empty, every other column equal (keyed by column name — `cmp_save.py`'s positional
  cell diff reports 16 "cells" because the inserted column shifts the row). Figures JSON-identical (H1 85/245,
  counter 151/124, confluence 294/233); run.log only the parked `by_lens` set-order line; replay 45.1 s wall.
  Code: `UnresolvedTrigger.parent_input_idx` (before `probe_input_idx`); `SweepTrigger.probe_input_idx` →
  `parent_input_idx` (orchestrator ×4, spawn); `_Sweep._unresolved` takes H1 from the trigger and M15 only from
  the resolver (no cross-frame fallback); FC `ProbeFailure` 649 → None, 652 / 657 → the mapped M15 input. Tests
  936 → 939 + 1 xfail: new pins — the unresolved reversal row (`parent_input_idx` None + the handoff's M15), the
  FC end-out-of-bounds / end-mapping-failure branches (M15 13, not H1 3; both were untested); the independence
  parametrization now expects `probe_input_idx` None on the `None` / `pending` rows (a restored fallback would
  put the H1 25 there); geometry_failed asserts both inputs. Docs: GLOSSARY `probe_input_idx` / new
  `parent_input_idx` / `SweepTrigger` / `ResolvedStart / ProbeFailure` / `UnresolvedTrigger`; PART4 §17.7;
  `uc1_trigger.py` comment. History kept: PLAN_C §2 `UnresolvedTrigger` sketch, PLAN_E §9.1 E5·4b text.
  **Landed `1756c30`, save `20260926_120641_1756c30`** (save `c2a569b`, trunk `f6e9fbe`).
- **Post-E·1a landing review (2026-09-26; 2 parallel lenses ≈334k).** Conformance (≈186k): 0 BLOCKER / MAJOR;
  every unresolved path one frame per column; every `ProbeFailure` in the three resolvers M15 or None (reversal
  shifted by `slice_begin`); "only FC price-maps its H1 input" verified. MINOR: GLOSSARY `parent_input_idx` said
  "from `MultiTFTrigger.meta`" (true only for `first_counter`; the others read the typed trigger's `input_idx`);
  `uc1_trigger.py` "the CTS anchor seeds the probe input" (pre-existing, contradicts the sibling resolver); no
  unit pin on the orchestrator's four `SweepTrigger` constructions (a wrong `subsequent_counter` source is
  invisible even to `/compare` — no such unresolved row on the window). NIT: the `geometry_failed` row carries a
  SUCCESSFUL resolver's input ("before it failed" was wrong ×3); PART4 §17.7's `probe_failed` list was
  reversal-only; the §9 Post-E bullet "(M15, when a probe ran)"; GLOSSARY Naming Standard — `*_input_idx` is a
  role name; the `cmp_save.py` positional caveat only in the README; FC end-out-of-bounds detail assert matched
  the input branch too, the FC input-out-of-bounds branch unpinned; the FC detector comment "FC pool key".
  Mutation (≈148k; 31 mutants on a `git archive` copy): 23 killed, 8 survived — O1–O7 (every orchestrator
  `SweepTrigger` source → None / a wrong H1 idx: no test ran `_run_multi_tf_dual`) and SB2 (the unchanged sibling
  "probe pending" branch → None); no defect. Pins: `test_orchestrator_sweep_triggers_carry_each_types_h1_parent_input`
  (monkeypatches the in-function imports, captures the sweep's triggers; kills O1–O7 — O4 re-verified by the main
  loop) and `TestResolveSiblingCts::test_pending_probe_returns_probe_failure_with_m15_input` (kills SB2). All
  folded in (byte-identical fold-in) + `test_input_out_of_parent_bounds_returns_probe_failure_without_input`;
  tests 939 → **942 + 1 xfail**. Kept: the predicted-table sibling stand-ins (820 / 871, disclosed).
- **Post-E·1b sites (from the conformance lens):** writers `first_confluence_pipeline.py:45` (+ docstring :5),
  `subsequent_confluence_pipeline.py:42` / `subsequent_counter_pipeline.py:42` (+ their docstrings :5-6 — also
  false: the H1 input is informational there), `uc1_trigger.py:82` (+ comments); readers `entity_df_mutation.py:623`
  (`.get` → `meta[...]`; print / detail strings :626-632 put the H1 value under the M15 name) and
  `orchestrator.py:888` (`.get` → `meta[...]`, so a missed writer fails loudly); implicit: `_synth_reversal_trigger`
  copies `{**source_trigger.meta}` (reversal triggers inherit the unread H1 key — decide keep / drop); tests: the
  three pipeline tests, `test_first_trigger_migration.py` `_make_trigger` / `_fc` helpers + callers + comment,
  the new orchestrator pin's `v2.meta`; docs: GLOSSARY `parent_input_idx` + LOH rows, PART4 §9.4 (~1830) the
  `MultiTFTrigger` field list, memory `project_part4_progress.md` (meta-key line). Out of scope: `meta["probe_end_idx"]`
  stays H1 while the probe's `probe_end_idx` (`unified_probe` / `ProbeCacheEntry`) is M15 (the same name / frame split
  for the end bound) → **done, Post-E·3 §9.4**.
- **Post-E·1b — `MultiTFTrigger.meta["probe_input_idx"]` (H1) → `"parent_input_idx"` (2026-09-26; user: go ahead,
  keep `_synth_reversal_trigger`'s full meta copy — reversal triggers inherit the unread H1 key under its new
  name).** Writers: the three pipelines + `uc1_trigger` (+ their docstrings: the sibling types' H1 input is
  informational); readers: the FC resolver (keeps its tested `.get` + "missing → ProbeFailure" path; its print /
  detail strings name `parent_input_idx`), the orchestrator's `first_counter` read → `v2.meta["parent_input_idx"]`
  (a missed writer now raises instead of a silent None). Tests: the helpers `_make_trigger` / `_fc` take
  `parent_input_idx=`; the three pipeline tests also assert the old key is absent (no alias). Docs: GLOSSARY
  `parent_input_idx` + LOH rows, PART4 §9.4. **Prediction:** byte-identical — 24/24 CSVs, figures, run.log (the
  renamed print / detail strings fire only on an FC input failure: 0 on the window). After 1b `probe_input_idx` names the probe's own-frame input: M15 on every multi-TF field, meta key and CSV column (its one other use, `structure_engine`'s local, is the main H1 structure's reversal probe — its own frame). **Measured == prediction** (vs `20260926_120641_1756c30`): 24/24 CSVs
  byte-identical, figures JSON-identical, run.log only the parked `by_lens` set-order line; tests 942 + 1 xfail
  (unchanged count — the no-alias asserts sit in the existing pipeline tests). **Landed `532df17`, save
  `20260926_122844_532df17`** (save `06b300a`, trunk `3a5481c`; reuse-mode save of the verified replay).
- **Post-E·1b landing review (2026-09-26; 1 conformance lens ≈174k): 0 BLOCKER / MAJOR.** Verified: pure rename;
  the orchestrator's direct index is safe (uc1 is the only `first_counter` producer, always writes the key); the
  inherited reversal key is never read; no-alias asserts fail on an old-key alias (in-memory mutants). MINOR,
  folded: the uc1 docstring still said "probe input"; the `subsequent_*` pipelines' "Mapping (§4.3.1 …)"
  paragraphs (pre-existing, false — the sibling resolver never maps an H1 input); the "M15 everywhere" claim
  (scoped: `structure_engine`'s local is the main H1 reversal probe's own-frame input); no test ran
  `detect_uc1_triggers` (an old-key alias there survived → pin `test_uc1_trigger_writes_the_h1_parent_input…`);
  PART4 §4.3 Session-3 note "each variation's `input_idx` … mapped to the sub TF FIRST" (pre-existing, false for
  the sibling types). NIT, folded: the renamed detail strings pinned; GLOSSARY reversal-inheritance note + role
  wording; PART4 §9.4 per-key scope; "§4.3.4 step 5" → step 4 (the fallback; 5 sites) and the pre-Session-3
  `types.py` / `subsequent_confluence_trigger.py` docstrings ("the probe derives its own BOS_0 internally") —
  rewritten to the co-sourced sibling rule; the reversal synth's inheritance of `parent_input_idx` pinned (the
  user's keep decision). **Post-E·1 DONE.**

### 9.3 Post-E·2 — the coordinate-hygiene families: exported slice-local M15 meta keys (2026-09-26)

- **Census re-verified from the data** (save `20260926_122844_532df17`; every family count == the §9 bullet).
  `slice_begin` = the sub's `starting_idx − 50` (run.log `geometry built … slice_begin=`: 404 / 1747 / 2315 / 2589 /
  3254 / 3571 / 3710 / 3977). A wider scan of EVERY int-valued meta path (nested too) in the 10 M15 lens CSVs found
  no other unshifted candle index: the rest are shifted keys or non-indices (`cross_start_cycle`, fib `version`,
  `proximity_pips`, WVMI's H1 `triggered_by_event_idx`).
- **Reader inventory (grep + measurement).** No production code reads an F1–F8 key AFTER the mirror: the keys are
  read only inside MS / FibTracker (`activated_at`, `cycle1_bos_idx`) on the slice itself, before the mirror; the
  M15 chart reads none of them (and draws no range rectangles — `export_plotly`'s `RANGE_STARTED.start_idx` reader is
  the H1 chart's own events); the CSV exporters dump meta verbatim; `_build_sibling_cts_ref_zone_from_pool` shares
  `_EVENT_META_IDX_KEYS` but copies only CTS events, which carry no family key. **Variant replay** (scratch plugin
  shifting every family at the mirror, readers untouched): 715 keys in 547 meta cells change == the census, EVERY
  change exactly `+slice_begin` of its sub (key-level diff), no non-meta column changes, the subs / triggers /
  unresolved / H1 CSVs byte-identical, the 3 figures JSON-identical, run.log content identical. **The full suite
  with the variant active: 944 + 1 xfail pass** — no test depends on the slice-local values, and none pins the
  entity-absolute ones either (F5 / F7 had no guard coverage at all) → the fix adds value-level coverage. No coupled
  reader → no plan cold review (§ item 2 method (e)).
- **Real examples** (before → after): conf sub 0 `STATE_CHANGED`@458 `effective_idx` 54 → 458; `RANGE_STARTED`@462
  `start_idx` / `confirm_idx` / `cts_idx` 56 / 58 / 56 → 460 / 462 / 460; `REVERSAL_WATCH_START`@1936 `expires_idx`
  1537 → 1941 (= 1936 + k 5); `BOS_CONFIRMED`@1020 `pb_start` 398 → 802; KL (anchor 1761) `expanded_last_idx` 1494 →
  1898; POI ic 678 `bos_idx` / `cts_idx` 50 / 380 → 454 / 784 (the fib CSV's own columns); fib cycle 0
  `activated_at` 54 → 458 (the EST moment), `locked_at` 398 → 802 (= that BOS's `pb_start`).
- **User decisions (2026-09-26):** ONE commit for all families (the key-level verifier attributes every changed key,
  so per-family `/compare`s add nothing once no reader is coupled); **shift in place** — same key names, values
  entity-absolute (what the mirror's docstring already claimed); the Naming-Standard renames (`pb_start`, the
  `cts_idx` / `bos_idx` anchors) stay on the Later hygiene list; **F8 fixed too** (byte-identical: attrs only), so
  every allow-list in the guard can go.
- **As landed (2026-09-26).** `multitf/entity_df_mutation.py`: `_EVENT_META_IDX_KEYS` += F1–F4 (`effective_idx`,
  `start_idx`, `confirm_idx`, `cts_idx`, `pullback_apply_idx`, `proximity_apply_idx`, `expires_idx`, `pb_start`);
  `_ZONE_META_IDX_KEYS` += F5 `expanded_last_idx`, F6 `bos_idx` / `cts_idx`; new `_FIB_META_IDX_KEYS`
  (`deactivated_at` + F7 `activated_at` / `reactivated_at` / `locked_at` / `cycle1_bos_idx`) replaces the fib site's
  `("deactivated_at",)` literal; new `_WAVE_CANDLE_META_IDX_KEYS` (F8 `anchor_idx`) at the wave-candle copy. **Found
  on the way (0 cells):** `pb_reconfirm_idx` sat in the EVENT list, but no event carries it — it is KL CTS-zone meta
  (`kl_zones_v1` CTS_RECONFIRMED upgrade; subs get BOS zones only, so it never reached an M15 row) → moved to
  `_ZONE_META_IDX_KEYS`; the static emitter scan requires it there. And `poi_zones`' `inst.meta["armed_idx"]` /
  `["confirmed_fill_idx"]` are ImbalanceInstance meta (the fill cache) — never mirrored, not a family.
  **Guard** `tests/test_event_meta_idx_keys.py`: `KNOWN_SLICE_LOCAL` / `_ZONE` / `_WAVE_CANDLE` deleted (every index key
  must be in its shift list); + fib-meta and wave-candle checks on the fixture; + a static `meta=` scan of
  `kl_zones_v1` / `poi_zones` / `fib_tracker` / `wave_candles` (parametrized, each with a must-see set so it is not
  vacuous — covers the keys the fixture never produces: KL `expanded_last_idx` / `pb_reconfirm_idx`, fib
  `reactivated_at` / `cycle1_bos_idx`); + a VALUE pin pairing every mirrored element with its slice-local source
  (listed int keys == source + `slice_begin`, the nested `bounds_steps` / `activation_history` likewise — the former
  had no pin — every other key unchanged). My pre-review mutation loop (scratch `git archive` copy, 13 mutants: the
  fib / wave sites reverted, each family key dropped from its list, an off-by-one in the shift): 13/13 killed.
  **Measured vs `20260926_122844_532df17` == prediction:** 715 keys in 547 meta cells (events conf 358 / counter 114,
  POI 30 / 12, KL 4 / 0, fib 21 / 8), every change exactly `+slice_begin` of its sub; the other 17 CSVs byte-identical;
  the landed CSVs byte-identical to the variant's; the 3 figures JSON-identical; run.log byte-identical to the
  `by_lens` commit's. Replay 43 s wall. Tests 944 → 950 + 1 xfail. Docs: ARCHITECTURE event table (`expires_idx` ×2,
  `effective_idx`), LANDMINES rule-3 guard note + "Mirror Translation…" rule, FIB_LIFECYCLE_SPEC sub 3 example
  (`activated_at` 2651), KL_ZONES_SPEC `expanded_last_idx`, POI_ZONES_SPEC `bos_idx` / `cts_idx`, an
  IMBALANCE_FILL_SEMANTICS note (its Plan F table's values are slice-local). Saves before this commit carry the
  slice-local values. **Landed `574de2a`, save `20260926_214932_574de2a`** (save `4d1566d`, trunk `760a382`;
  reuse-mode save of the verified replay).
- **Rule 3's "set on every event" / "`.get` → `meta[key]`" bullets do not apply here:** a pure change of FRAME —
  names, presence and emit paths are unchanged (the RANGE_STARTED keys stay path-specific: `start_idx` /
  `confirm_idx` on the offline path, `pullback_apply_idx` / `proximity_apply_idx` on their own); the six
  `export_plotly` `.get("start_idx", e.idx)` reads are H1-only with a deliberate fallback.
- **Landing review (2026-09-26/27; 2 parallel lenses ≈465k — conformance ≈281k, mutation ≈184k; `3082aca` in
  scope): 0 BLOCKER / MAJOR.** Conformance independently re-verified the data (715 / 547, each `+slice_begin`), the
  semantics on the new save (RANGE_STARTED `confirm_idx == idx` 45/45, `pullback_apply_idx == idx` 9/9, `cts_idx <=
  idx` 54/54; `effective_idx` 388/388; `expires_idx` = anchor + 5 11/11; KL `expanded_last_idx` = a THRESHOLD_UPDATED
  idx 4/4; fib `locked_at` = a CTS_CONFIRMED idx 25/25; POI `(bos_idx, cts_idx)` = its fib row 42/42), no reader, the
  docs, and the `by_lens` fix. Mutation: 43 mutants, 23 killed, 6 equivalent, 14 real survivors. **Found (folded in
  one byte-identical commit):** (1) PRE-EXISTING unpinned shift sites — fib dataclass `bos_idx` / `cts_idx` /
  `end_idx` (the M15 fib CSV columns + the fib drawing), wave-candle `first_/last_wave_candle_idx` and
  `prev_bos_lines` (M15 chart positions), the mirror's WVMI loop → a synthetic every-site mirror test; (2) the value
  pin is self-referential — `cycle_id` added to the event list survived → a list-validity test (index-like names
  only, a never-listed set of known non-indices, no duplicates); (3) the static scans' blind spots (`**{...}` splats,
  attribute subscripts, `setdefault`, unsuffixed keys) → a broad emitter AST scan with a documented `_NOT_MIRRORED`
  set + a value-type classification of every int meta value on the fixture; (4) `_shift_meta_indices`' contract (0,
  None, float, no in-place) → a unit test; (5) `FibState.cts_history` `(idx, price)` entries still slice-local after
  the mirror (attrs only, no reader) → **user: shift it** (`entity_df_mutation` fib `replace`); (6) five DEAD list
  entries (event `confirmed_idx` / `deactivated_at`, zone `start_idx` / `ic_idx` / `deactivated_at` — nothing emits
  them) → **user: delete + an inverse guard** (every listed key is emitted by its element kind — kills
  `pb_reconfirm_idx` misfiled again); (7) stale docs — LANDMINES "Mirror Translation…" (the removed
  `FibState.activation_history` row, no KL `activation_history` row, a non-existent `cap_idx_local`, a stale chart
  line ref), GOTCHAS `pb_start` ("pullback start" → the last pullback pattern's apply candle, on both BOS `source`s)
  and the Plan F record's unlabelled slice-local 53 / 54, the POI example's "cycle 0", this rule-3 note; my own
  mirror comment on `pb_start` ("pullback_extreme; None otherwise" — wrong). **Not taken (user):** a child-pytest
  under an adversarial `PYTHONHASHSEED` for the `by_lens` pin (a `frozenset` rewrite escapes the monkeypatched
  `set`; run.log order only). **Left as NIT:** `_shift_meta_indices` skips a non-Python-int value silently (numpy
  ints; all 715 values here are Python ints — casting at the emitters is the E2a pattern if one ever appears). The
  fold-in's own check: the 17 re-run survivors all KILLED (scratch copy). Fold-in `/compare` vs
  `20260926_214932_574de2a`: 24/24 byte-identical, figures JSON-identical, run.log identical; tests 950 → **961 + 1
  xfail**. Fold-in `fa07f1d`. **Post-E·2 DONE.**

### 9.4 Post-E·3 — the FC H1 `probe_end_idx` → `parent_cts_anchor_idx` (2026-09-27; byte-identical)

- **Why:** `FirstConfluenceTrigger.probe_end_idx` / `MultiTFTrigger.meta["probe_end_idx"]` held the H1 parent CTS
  ANCHOR while `unified_probe(probe_end_idx=)` / `_probe_with_cache` / `ProbeCacheEntry.probe_end_idx` hold the M15 search
  bound — one name, two frames (found in the Post-E·1a review). Reference window: FC(0,0) 430 → 1721, (0,1) 652 →
  2609, (1,2) 905 → 3621. **User: `parent_cts_anchor_idx`** (what it IS; the FC resolver's local already had that
  name; sibling of `parent_bos_anchor_idx`). No CSV column carries it — its name and H1 value appear only in the FC
  resolver's `probe_failed` detail text (missing / out of parent bounds) in the unresolved CSV: 0 such rows here; the H1
  warnings that print it never fire here.
- **Sites (the H1 carrier only):** `multitf/types.py` (field + docstring), `first_confluence_trigger.py` (local,
  constructor kw, docstrings), `first_confluence_pipeline.py` (the meta writer + module doc), the FC resolver in
  `entity_df_mutation.py` (the meta read — keeps its tested `.get` → logged-`ProbeFailure` path, as Post-E·1b did
  for `parent_input_idx` — two WARNING prints + two `ProbeFailure` detail strings), `data_bridge.py` docstring,
  `debug/probe_fc_finalize.py` (2 reads). Tests: the `_make_trigger` / `_fc` fixture kw (29 call sites, AST-located
  so the M15 probe-kwarg / `ProbeCacheEntry` / `unified_probe` uses stay) + the helper's meta key + a detail-string
  assert; `test_first_confluence_pipeline.py` (constructor + meta asserts, + a no-alias assert that
  `"probe_end_idx"` is absent from the meta); `test_first_confluence_trigger.py` (5 field reads + a test name);
  `test_lifecycle_sweep_unit.py` (the inherited-meta pin + a `SimpleNamespace` FC stand-in). **Not renamed (the M15
  / own-frame bound):** `unified_probe` / `_run_phase1` / `_run_phase2`, `_probe_with_cache`, `ProbeCacheEntry`, the
  `[probe_cache]` prints,
  `structure_engine`'s main reversal probe. Docs (every `probe_end_idx` hit classified; only the H1-carrier ones
  moved): LANDMINES "Probe `end_idx` Is the Supreme Bound" rename note + the mapper-scope paragraph, GLOSSARY (new
  `parent_cts_anchor_idx` entry; `probe_end_idx`; LOH), PART4 §4.3 (the mapping + naming notes) / §4.3.1 / §4.3.2
  (the var-1 row + the NULL paragraph) / §9.4 (the meta list) / §10.1 (the subscription matrix) / §14 / §16.6 /
  §17.8 (the Plan C rename history). **Landed `f770c47`** (no save: byte-identical).
- **Landing review (2026-09-27; 1 conformance lens ≈157k): 0 BLOCKER; 1 MAJOR.** Verified: the code rename complete
  and correct (every remaining bare `probe_end_idx` in code / tests is the M15 or own-frame bound; the writer → reader
  → `+lower_sd` map → M15 bound chain; pending FC triggers go `status` → `SweepTrigger.pending` → `UnresolvedTrigger`
  without reading the name; no CSV / HTML carries it; the example values). **MAJOR (mine, propagated from this plan's
  own §9 bullet and §9.2):** "`ProbeResult.probe_end_idx`" — `ProbeResult` has no such field (never had) → corrected in
  LANDMINES, PART4 §4.3, GLOSSARY (which also claimed the meta key was once `end_idx` — it already was
  `probe_end_idx`), here (§9 bullet, §9.2, §9.4); the commit message `f770c47` carries the same slip. **MINOR,
  folded:** the `_make_trigger` fixture put the FC-only key on 18 non-FC triggers (its value was really their
  `trigger_event_idx`) → the helper now takes `trigger_event_idx` separately and writes `parent_cts_anchor_idx` on FC
  only (default 5 there); three `unified_probe.py` docstrings described the H1 carrier's pending NULL under the M15
  name; the `.get` "(tested)" claim covered only the input half → a test for the missing anchor + a no-reader-alias
  case (a meta holding only the old key is "missing"); "Not exported" → the `probe_failed` detail text is its only
  CSV trace. NITs folded: GLOSSARY `cts_anchor_idx (parent)` cross-link, PART4 §4.3 wording + §16.6 circularity, the
  renamed test's docstring, this §9.2 cross-link, memory `reference_key_files.md`. Fold-in `/compare` vs
  `20260926_214932_574de2a`: 24/24 byte-identical, figures JSON-identical, run.log identical; tests 961 → **962 + 1
  xfail**. **Post-E·3 DONE.**

### 9.5 Post-E·4 — the Naming-Standard meta renames (2026-09-29d; plan approved + LANDED the same day)

- **Why:** the last items of the Later hygiene list (§9.3, user 2026-09-26: "the Naming-Standard renames stay Later"):
  four exported meta keys whose names do not say which kind of candle they hold (GLOSSARY "Naming Standard"). They
  must land before the strategy layer reads the keys. A save-format boundary.
- **Scope (the user's list) — values unchanged, keys renamed IN PLACE in the emitting dict literal (same position):**

  | Old key | Element | New key | Kind | Value (reference window, save `20260928_145121_fa172bc`) |
  |---|---|---|---|---|
  | `pb_start` | `BOS_CONFIRMED` meta (both emit sites) + its copy on the H1 BOS `StructureLevel` | `last_pullback_apply_idx` (proposed; §9.5 Q1) | moment (`apply_idx`) | `st.last_pullback_pat_apply_idx` = the apply candle of the LAST pullback pattern since the previous cycle was established (reset at each new cycle; also the start of `_select_bos_on_breakout`'s window). Measured: cycle 0 → None (12/12); cycle >= 1 → == the structure's last `STATE_CHANGED(to=pullback, reason=pullback_pattern)` (22/22). H1: BOS @652 (sid 0 cycle 1) 439; @748 (1, 1) 727; @902 (1, 2) 810 (pullback patterns so far 727, 810) |
  | `cts_idx` | `RANGE_STARTED` meta (3 paths: offline finalize, `pullback_created_range`, `proximity_created_range`) | `cts_anchor_idx` | MS anchor | `st.cts.idx` when the range starts; pairs with `cts_price` (kept). H1: RANGE_STARTED @313 `cts_idx` 311 |
  | `bos_idx` / `cts_idx` | POI zone meta | `bos_anchor_idx` / `cts_anchor_idx` | MS anchor | the owning fib's `bos_idx` / `cts_idx` copied at build. H1: IC 732 (sid 1 cycle 1) 689 / 761 (its `cts_established_idx` 748 — anchor != moment) |

- **Out of scope (stay GLOSSARY "Bare element idx"; §9.5 Q2):** `FibState.bos_idx` / `cts_idx` + the fib_lifecycle.csv
  columns (≈62 refs in `fib_tracker`, the fib drawing, the mirror, many tests), the final.csv columns `cts_idx` /
  `bos_idx` (MS output df — read by `identify_start`, `unified_probe`, the mirror's aux list, the df invariants), the
  cycle-0 caches (`st.cycle0_data`, FibTracker `_cross_cycle_data`), function parameters, the `zone_proximity_diag`
  CSV (user 2026-09-25: leave the diag columns). Consequence: a POI's `bos_anchor_idx` == its fib row's `bos_idx`.
- **Inventory (at `bfe8178`; grep of every `.py` / current `.md` + the census of the save):**
  - Emitters: `market_structure.py` RANGE_STARTED `:1074` / `:1191` / `:1968`; BOS_CONFIRMED `:1331` (cycle 0) /
    `:1345` (cycle >= 1); `poi_zones.py` `:616-617`. The H1 levels copy the BOS meta (`_events_to_structure_levels`).
  - Readers: NONE that decide anything — only `poi_zones`' env-gated `POI_LIFECYCLE_DEBUG` print (`:645-646`,
    `.get` → `meta[...]` per rule 3). No chart / hover / sweep / zone / strategy read (export_plotly, export_m15_chart,
    debug/, multitf/, pipeline/ grepped); run.log carries none of the keys (0 lines).
  - Registry: `entity_df_mutation._EVENT_META_IDX_KEYS` — `cts_idx` goes (RANGE_STARTED's new key is the listed
    `cts_anchor_idx`), `pb_start` → the new name; `_ZONE_META_IDX_KEYS` — `bos_idx` / `cts_idx` → the new names; comments.
  - Guard `tests/test_event_meta_idx_keys.py`: `EXTRA_EVENT_IDX_KEYS = {"pb_start"}` deleted (every index key is then
    suffixed; the int-value classification test still covers unsuffixed keys); must-see sets; `_NOT_MIRRORED
    ["market_structure"]` += `cts_idx` (the `st.cycle0_data` cache, beside `bos_idx`); the fixture's shift pin keyed by
    event type for RANGE_STARTED (its new key is shared with the CTS events, so a key-only pin would pass vacuously).
  - Fixtures: `test_lifecycle_sweep_predicted_table.py` (5 `pb_start=`), `_event_factory.py` docstring; no test builds a
    POI / RANGE_STARTED meta with the old keys (the `c0 = {"bos_idx": …}` dicts are the cycle-0 cache — out of scope).
  - Tools: `meta_census.py` / `guard.py` keep `pb_start` index-like (old saves); `hyg_variant_plugin.py` is Post-E·2
    history; new `cmp_meta_rename.py` (every CSV: BASE with the listed keys renamed in place == CUR, else exit 1).
  - Current docs: GLOSSARY "Naming Standard" (the Bare-element row loses "POI meta" + `RANGE_STARTED.meta["cts_idx"]`;
    Status line), POI_ZONES_SPEC field list, GOTCHAS `pb_start` bullet, LANDMINES proximity-gate note (`pb_start: None`),
    the mirror comments, review_scripts README. Dated records (plans, saves, commit messages) stay as written.
- **Predicted `/compare` vs `20260928_145121_fa172bc` (`cmp_meta_rename.py … pb_start=<new> cts_idx=cts_anchor_idx
  bos_idx=bos_anchor_idx`):**

  | CSV | Renamed keys | Cells |
  |---|---|---|
  | H1 `structure_events` | BOS_CONFIRMED `pb_start` 5 (3 non-null) + RANGE_STARTED `cts_idx` 10 | 15 |
  | H1 `structure_levels` | BOS `pb_start` 5 | 5 |
  | H1 `poi_zones` | `bos_idx` 5 + `cts_idx` 5 | 5 |
  | M15 confluence `structure_events` | `pb_start` 21 + RANGE_STARTED `cts_idx` 39 | 60 |
  | M15 confluence `poi_zones` | `bos_idx` 30 + `cts_idx` 30 | 30 |
  | M15 counter `structure_events` | `pb_start` 8 + RANGE_STARTED `cts_idx` 15 | 23 |
  | M15 counter `poi_zones` | `bos_idx` 12 + `cts_idx` 12 | 12 |
  | **total** | **197 keys** | **150 cells** |

  Every value and key position unchanged; the other 17 CSVs byte-identical; the 3 figures identical (85/245,
  151/124, 294/233); run.log identical except timing. Tests: 1073 + the new pins, all passing.
- **Landing:** ONE atomic migration commit (LANDMINES "Event Contract Rules" rule 3: code + registry + guard + fixtures
  + current docs; the prediction in the message; no alias) → `/compare` → chart pause → `/commit-save` = the new
  baseline (the save-format boundary recorded in memory `project_plan_e_event_convention.md`). Landing review: my own
  mutation loop first (each old key restored at one emitter; each new key dropped from its list; the debug print left
  on `.get`), then 1 conformance lens (≈150–250k tokens; asked before launch).
- **Q1 (name for `pb_start`):** (A, recommended) `last_pullback_apply_idx` — the moment kind (`apply_idx`), "last"
  is literal; (B) `pullback_apply_idx` — the key RANGE_STARTED already uses for its creating pullback, drops "last"
  (on the window every retracement holds exactly ONE pullback pattern — 22/22 — so "last" is the code's rule, not a
  window fact: the state keeps the latest pullback apply until the next cycle resets it); (C) `last_pullback_pat_apply_idx` — the state
  field's exact name. **Q2:** confirm the scope above (FibState / final.csv / caches stay).
- **USER DECISIONS (2026-09-29d):** Q1 (A) `last_pullback_apply_idx`; Q2 the four meta keys only (FibState /
  fib CSV / final.csv / caches stay "bare element idx"); landing review = 1 conformance lens after my mutation loop.
- **As landed (2026-09-29d).** Code: the 5 MS emit literals + the POI meta literal renamed in place (line-neutral —
  the FutureWarning stays `:2437`); the POI debug print reads `meta["bos_anchor_idx"]` / `["cts_anchor_idx"]`;
  `_EVENT_META_IDX_KEYS` (`cts_idx` dropped — the listed `cts_anchor_idx` covers RANGE_STARTED; `pb_start` →
  `last_pullback_apply_idx`) and `_ZONE_META_IDX_KEYS` (POI pair) with comments. Guard: `EXTRA_EVENT_IDX_KEYS`
  deleted, must-see sets moved, `_NOT_MIRRORED["market_structure"]` += `cts_idx` (the cycle-0 cache), the value pin
  keyed by type for RANGE_STARTED (the fixture has no cycle >= 1 BOS, so the BOS key's VALUE shift stays unpinned
  there, as `pb_start`'s was). New pins `tests/test_meta_key_renames.py` (5): every RANGE_STARTED / BOS emit literal
  (static, all 3 + 2 paths), no `pb_start` constant in MS, no `bos_idx` / `cts_idx` constant in `poi_zones`, and a
  real run (`_make_double_rewind_data`, sd ±1: all three RANGE_STARTED paths, BOS cycles 0 / 1 / 2 → None / 4 /
  None). **Measured == the prediction:** `cmp_meta_rename.py` exit 0 — the 13 (CSV, type, key) counts above,
  150 cells / 197 keys, every other cell identical, the meta-less CSVs byte-identical; `cmp_save.py` 150 cells;
  figures traces_xy_equal + shapes_equal (85/245, 151/124, 294/233); FETCH GATE PASS; run.log == the warm-up
  replay's line for line except timing (9045 lines); replay 43.5 s. Tests 1073 → 1078. **Own mutation loop (12
  mutants, scratch trees, full suite): 12/12 KILLED** — an old key restored at each of the 7 emit sites, each new
  key dropped from its list (3), the debug print back on `.get`, an alias (both POI keys). Without the new file two
  SURVIVED: `pb_start` restored at the cycle >= 1 BOS (no suffix → invisible to the name guard; the fixture's only
  BOS is cycle 0) and the debug print. Docs: GLOSSARY (the Bare-element row, Status), ARCHITECTURE (the BOS row's
  meta column + RANGE_STARTED's key), POI_ZONES_SPEC, GOTCHAS, LANDMINES, review_scripts README (+ `cmp_meta_rename.py`,
  `meta_census.py` keeps `pb_start` for old saves). **Save-format boundary:** saves before this commit carry
  `pb_start`, RANGE_STARTED `cts_idx`, POI `bos_idx` / `cts_idx`.
- **Landing review (2026-09-29d; 1 conformance lens, 234,242 tokens incl. the warm-up `bfe8178`): (A) CONFORMS WITH
  FIXES, (B) CONFORMS; 0 BLOCKER.** Verified independently: no missed current site (whole-repo grep, every hit
  classified), 150 / 197 by its own raw-text script, 22/22 + 12/12 `pb_start` semantics, 47/47 POI anchors == the fib
  row. Its 4 mutants all SURVIVED my pins → folded: **MAJOR 1** an extra `cts_idx` in the cycle >= 1 BOS literal
  exported slice-local with no test failing — `_ms_emitted_event_meta_keys` read only a direct `meta=` dict, not the
  one inside `_end_watch_superseded_by_new_cycle(...)`, and my `_NOT_MIRRORED` `cts_idx` blinded the module scan →
  both literal scans now read wrapper-call dicts (`_meta_literal_keys`) and the real-run pin forbids the old keys on
  EVERY event and level; **MINOR 2** values unpinned (the proximity path's key set to the candle; the POI pair
  swapped) → the RANGE_STARTED values pinned (4/2, 9/8, 13/8, 14/12) + POI anchors ∈ its (sid, cycle) fibs' pairs
  on the rendered sub; **MINOR 3** GLOSSARY overclaimed "the last four" → "the four the user scoped" + the keys still
  non-standard (fib `cycle1_bos_idx`, KL `expanded_last_idx` / `pb_reconfirm_idx`; not scoped, not decided); **MINOR
  4** an alias added in an exporter is caught by no test (no exporter test exists) → the guard for the export layer
  is `/compare` with `cmp_meta_rename.py`; **NIT 5** `cmp_meta_rename.py` compared pandas values (`652.0` → `652`,
  `1` vs `True` passed) → raw text via the csv module + `str(meta)` round-trip (both perturbations now exit 1);
  **NIT 6** GLOSSARY Moment row + `*_apply_idx`; **NIT 7** (B) the lingering case can now set `early_stop_idx` (the
  probe's print only) → stated in MARKET_STRUCTURE_SPEC. Side find (pre-existing, not this migration's): in
  `_make_double_rewind_data` the offline finalize of step 9 emits a second RANGE_STARTED (start 9, stamped at its
  confirm 13, carrying the cycle-1 anchor 8) on a range the proximity path opened at 9, listed before the breakout's
  RANGE_RESET @12 — the offline-finalize look-ahead class (zones audit (b)).

### 9.6 Post-E·5 — the last three non-standard exported meta keys (2026-09-30; decided + LANDED `896ffd4`, review fold-in the same day)

- **Why:** the three keys §9.5's landing review left non-standard (GLOSSARY "Naming Standard" Status; not scoped then).
- **Inventory (at `890e764`; grep of every `.py` / current `.md` + the baseline save `20260929_232136_79168b7`; the
  3 figures and run.log carry none of the keys):**

  | Key | Writer | Value — kind | Readers | Window rows |
  |---|---|---|---|---|
  | `cycle1_bos_idx` | fib meta of the cycle-1 CROSS fib (`fib_tracker.py:1111`, the `scenario_2_cross` branch, `"scenario": 2` — the mirror list's "Scenario 3" comment is stale) | `_handle_cycle1_scenarios`' `bos_idx` = the cycle-1 BOS **anchor** (the cross fib's own `bos_idx` is BOS_0's anchor) — MS anchor | **production:** `_update_cycle1_main` `:1684` (cond2's window `[BOS_1, CTS_1]`) + the assert message `:1721`. Tests: `test_e3a_mutation_pins.py:523` (mutates it), `test_main_versioned_cross.py:101`, the `.get` ban regex `test_event_fields.py:161` | H1 fib, 1 row: sid 1 cycle 1 cross (`bos_idx` 689 = BOS_0, `cts_idx` 761) → **728** = BOS_CONFIRMED(1, 1) `bos_anchor_idx` (its moment 748). M15: 0 |
  | `expanded_last_idx` (+ siblings `expanded_last_price` / `expanded_last_event`) | KL meta on a threshold expansion (`kl_zones_v1.py:867`) | the `*_THRESHOLD_UPDATED` event's `ev.idx` — a **moment** (the processing candle; ARCHITECTURE: CTS `range_sync` is "NOT a price location"). The trio is written from the same values, in the same `replace`, as the step appended to `bounds_steps` → == `bounds_steps[-1]` (`start_idx` / `price` / `event`) by construction; 7/7 on the window | none (the charts + the orchestrator read `bounds_steps`; no test reads the trio) | H1 3 CTS zones (`range_sync` @650 / 746 / 880), confluence 4 BOS zones (`probe_no_break` @1898 / 2820 / 4195, `rv_anchor_failed` @2460); counter 0. The price sits on the idx candle on all 7 (the BOS paths by construction — candle `i`'s wick; `range_sync` need not) |
  | `pb_reconfirm_idx` | KL CTS-zone meta, the CTS_RECONFIRMED post-pass (`kl_zones_v1.py:1019`) | CTS_RECONFIRMED `ev.idx` = the late pullback's apply candle — a **moment**; the zone's `confirmed_idx` keeps the original CTS_CONFIRMED moment | none; no test pins its value | **0 exported**: subs export BOS zones only, H1 has 0 CTS_RECONFIRMED. The one window instance is internal: confluence sub 1 cycle 1 — CTS_CONFIRMED by proximity @1902, CTS_RECONFIRMED @1904 (`one_maru_opposite`), anchor 1898 |

  Registry: `_FIB_META_IDX_KEYS` / `_ZONE_META_IDX_KEYS` (the mirror shifts all three). Guard: the must-see sets of
  `test_event_meta_idx_keys.py` (`:303-306`) + docstrings. Current docs: GLOSSARY (Status + the CTS-confirmation
  rows), KL_ZONES_SPEC `:208`, MARKET_STRUCTURE_SPEC `:56` / `:384`, PRE_REFACTOR_INVARIANTS `:53` / `:121`, the MS
  docstring `:1717`. Dated records (plans, plan inputs, `hyg_variant_plugin.py`) stay as written.
- **USER DECISIONS (2026-09-30), all three as recommended:** fib `cycle1_bos_idx` → **`cycle1_bos_anchor_idx`** (an
  MS anchor, the event key's spelling); KL **delete the `expanded_last_idx` / `_price` / `_event` trio** (a duplicate of
  `bounds_steps[-1]`, no reader — the E4b-pre `source_event_idx` precedent; the `expanded` flag stays); KL
  `pb_reconfirm_idx` → **`reconfirmed_idx`** (the past-participle moment, beside the zone's `confirmed_idx`). Light
  tier; landing review = 1 conformance lens after my mutation loop (asked before launch).
- **Predicted `/compare` vs `20260929_232136_79168b7`:** H1 `fib_lifecycle` 1 cell (the key renamed, value 728 and
  position kept); H1 `kl_zones` 3 cells + M15 confluence `kl_zones` 4 cells (the trio removed from each; `expanded`
  and `bounds_steps` unchanged) — **8 cells**; the other 21 CSVs byte-identical; the 3 figures identical (85/245,
  151/124, 294/233); run.log identical except timing (no key is printed; the MS docstring edit is line-neutral, so the
  FutureWarning stays `:2437`).
- **As landed (2026-09-30).** Code: the fib writer literal + its reader (`_update_cycle1_main`, local renamed too) +
  the assert message; KL: the trio removed from the expansion's `replace` literal (a comment says where the last
  expansion lives), `reconfirmed_idx` in the CTS_RECONFIRMED post-pass; the MS docstring; the mirror lists (the
  `expanded_last_idx` entry gone, the other two renamed; the fib entry's stale "Scenario 3" → Scenario 2). Tests: the
  guard's must-see sets; `test_main_versioned_cross` (VALUE 30 = BOS_1 vs the fib's `bos_idx` 10, no alias);
  `test_e3a_mutation_pins` (the forced-divergence pin moves the new key, and first asserts its value 30); the
  `.get`-fallback regex in `test_event_fields`; new pins in `tests/test_meta_key_renames.py` (5): no production string
  constant holds an old key (every module — an exporter alias included), the fixture's expansion (cycle 1's BOS zone
  @9: `bounds_steps[-1]` = moment / price / event, bounds, `expanded` ⇔ multi-step, no `expanded_last_*`) and the
  reconfirm upgrade (a CTS_RECONFIRMED @11 injected into `_make_double_rewind_data`'s run: `reconfirmed_idx` 11,
  `confirmed_idx` keeps 9, anchor 8, only the cycle-1 CTS zone, nothing else changes), sd ±1. **Measured == the
  prediction:** `cmp_meta_rename.py` exit 0 — 8 cells / 22 keys (fib 1 rename; KL 3 + 4 rows × 3 deletions), every
  other cell text-identical, meta-less CSVs byte-identical; `cmp_save.py` 8 cells; figures traces_xy_equal +
  shapes_equal (85/245, 151/124, 294/233); FETCH GATE PASS; run.log == the Post-E·4 replay's line for line up to
  timing (9030 lines); replay 43.7 s. Tests 1079 → 1084. Tools: `cmp_meta_rename.py` takes `old=` (a deletion);
  `meta_census.py` counts the pre-rename names as shifted (an old save showed them UNSHIFTED — a false alarm since
  Post-E·4). Docs: GLOSSARY (Status + the CTS-confirmation rows), KL_ZONES_SPEC, MARKET_STRUCTURE_SPEC (×2, + the
  sub-internal note), PRE_REFACTOR_INVARIANTS (×2), review_scripts README. **Save-format boundary:** saves before
  it carry fib `cycle1_bos_idx`, KL `pb_reconfirm_idx` (0 rows here) and the KL `expanded_last_*` trio.
- **Own mutation loop** (`mutants_post_e5.py`, 24 mutants, full suite each; value / swap / extra-key / exporter-alias,
  not only reverts — the 2026-09-29d lesson): **24/24 KILLED.** Fib 6 (revert, alias, BOS_0's anchor for BOS_1's,
  the EST moment for the anchor, the reader on `_bos_by_cycle`, a `.get` fallback), KL expansion 7 (the trio's keys
  back one at a time, the flag dropped, the step's price / start / event wrong), KL reconfirm 6 (revert, alias, the
  proximity moment / the anchor for the moment, `confirmed_idx` overwritten, the CTS-kind filter dropped), lists 4,
  exporter alias 1. **11 die ONLY on the new pins** (K2 / K3 — non-index keys the name guard cannot see; K4–K7 and
  K10–K13 — values; X1 — the exporter alias, uncatchable before by any test); F6 only on the updated `.get` regex, F5
  only on the updated forced-divergence pin.
- **Landing review (2026-09-30; 1 conformance lens, 212,338 tokens; worktree reset 8e32dd2 → `896ffd4` by the brief's
  step 1; frozen outputs): CONFORMS WITH FIXES, 0 BLOCKER / MAJOR, no code wrong.** Verified independently: no missed
  current site (every hit classified), 8 cells / 22 keys by its own raw-text script (the expected text built from BASE
  by string edits), figures identical (UUID-normalised), run.log == through line 9030, 728 = BOS_1's anchor, the trio
  == `bounds_steps[-1]` with NO later writer trimming either, `reconfirmed_idx` a moment, sub KL export BOS-only
  (`project_to_window` default); `cmp_meta_rename.py` rejects 6 wrong CUR variants. Its 10 new mutants: **6 SURVIVED**
  → folded (tests / docs / tools; byte-identical): **MINOR 1** no exporter test (X2 an alias built from string pieces,
  X3 `bounds_steps` dropped by the KL exporter) → `test_exporters_write_every_meta_dict_verbatim` (KL with an expansion
  + a reconfirm, fib, events from the fixture's run, POI from a rendered sub: each row's meta text == `str(obj.meta)`);
  **MINOR 2** `bounds_steps[-1]` is now the only record of the last expansion but its ORDER (K14 `insert`) and its
  mirror SHIFT (M1 INIT-only) were unpinned → a second, more extreme threshold @10 injected after the @9 one (steps
  [4, 9, 10], outer = its price) + a two-step `_synthetic_result` in the mirror pin; **MINOR 3** the docs replacing the
  trio blurred a moment with an anchor (on an unexpanded zone the last step is INIT, `start_idx` = `base_idx`) →
  KL_ZONES_SPEC / PRE_REFACTOR_INVARIANTS state the `expanded` / `event != "INIT"` guard; **NIT 4** this heading;
  **NIT 5** a copy under a NEW name (K15 `last_expansion_price`) → the expanded zone's key set == the same zone
  without the threshold event + `{"expanded"}`; **NIT 6** the Scenario-3 single writing the cross-only key (F7) → two
  Scenario-3 asserts; **NIT 7** `confirmed_idx` is clamped up to the structure's lifecycle start (`kl_zones_v1`
  lifecycle pass) → GLOSSARY / MARKET_STRUCTURE_SPEC / the post-pass comment say so; **NIT 8** the reconfirm moment
  read via `ef.event_moment(ev)` (same value); **NIT 9** `meta_census.py` tagged the pre-rename names SHIFTED (a
  re-introduced old key on a current save would hide) → a PRE-RENAME tag, always printed, + KL
  `bounds_steps[].start_idx` counted; **NIT 10** the compare skill's `cmp_meta_rename.py` line + `engine_v2/CLAUDE.md`
  "Post-E·1–5". Tests 1084 → 1087. `mutants_post_e5.py` now holds all 34 (mine + the review's 10): **34/34 KILLED**
  after the fold-in (M1 by the two-step mirror pin, K14 / K5–K7 by the second-expansion pin, K15 by the key-set
  pin, F7 by the Scenario-3 asserts, X2 / X3 by the exporter test).

## 10. Open questions for the user (recommendation first; concrete window data)

**User decisions 2026-09-24:** "Unless I note otherwise below, your recommendations sound good" → **ACCEPTED as
recommended:** Q17 (E1b), Q15 (delete), Q19 (keep `ev.price` = anchor price, documented), Q3 (pin; moment order
post-E), Q10 (migrate), Q18 (merge into E3a), Q2 (timing), Q8 (moment, own E3a′), Q5 (moment, E3f), Q6 (anchor),
Q20 (T1/T3/T4 → moment via E3g, T2 stays, T5 in E3a), Q7 (two), Q4 (b), Q12 (keep), Q13 (none), Q14 (separate).
**Then (same day, after clarifications):** "approve Q11, Q1 yes, Q16 ef., Q21 delete" → Q11 §4.2 wording APPROVED
as written; Q1 = (c) flip pattern-path `CTS_UPDATED.idx` in E4c; Q16 = qualified `ef.` calls; Q21 = delete KL meta
`source_event_idx` in its own commit **E4b-pre** before E4b.

**Already decided (not re-asked):**
- `pending_reversal_pattern_anchor_idx` (2026-09-23);
- everything in IN §4 "Resolved".

**Needed before E1:**
- **Q11 — contract wording.** Approve §4.2 (review-tightened: rename / re-mean by atomic migration, removal still
  banned; covers fixtures, registries, bare `.get(key)` / `key in meta`; proof-carrying for staged migrations like
  E2 → E4), or edit it.
- **Q17 — deletions: E1b (before E2) or E5?** Rec: **E1b**, so E2 does not carry the dead `cts_est_idx_by_key`
  reads in 4 trigger modules.
- **Q15 — the write-only `FibRetracement.anchor_high_idx/anchor_low_idx`.** Rec: delete in E1b.

**Needed before E2:**
- **Q1 — pattern-path `CTS_UPDATED`.** Its MOMENT is now required (E3·0, the review's blocker), so the remaining
  choice is only its `idx`:
  - (a) keep `idx` = anchor, a documented residual;
  - (c) flip it in E4c.

  Window: 37 rows, 1 lagging (conf sub 2 cyc 0: pattern-path at 2468, a same-price duplicate of the raw update at
  2468; apply 2470). Rec: **(c)** — it makes "`ev.idx` = the moment" universal for CTS events. Cost: CTS_UPDATED
  LOCATION reads join E2b + one 1-cell `/compare`.
- **Q16 — accessor naming.** Example of the clash: `cts_anchor_idx = cts_anchor_idx(ev)` raises
  `UnboundLocalError`, and `fib_tracker.py:1723` already has a local `cts_anchor_idx`. Rec: module
  `structure/event_fields.py`, always called qualified (`ef.cts_anchor_idx(ev)`), a test bans the direct import,
  and keep `event_moment`. Alternative: `*_of(ev)` names (`cts_anchor_idx_of(ev)`).
- **Q19 — `ev.price` after the flip.** It stays the anchor's price (e.g. H1 BOS_0: `idx` 96 → 115, price = the
  low of candle 96, 0.55808), which breaks the GLOSSARY rule "a bare idx paired with a price is the anchor". Rec:
  **keep `ev.price` as the anchor price and document it** (an `ev.price` column in the ARCHITECTURE table; the
  price belongs to `cts_anchor_idx` / `bos_anchor_idx`). Alternative: also emit meta `*_anchor_price`.
- **Q21 — KL meta `source_event_idx` (revised after the user's question).** Written at `kl_zones_v1.py:888`/`:951`
  as `int(ev.idx)` of the zone's source event; **no code reads it** (the `source_event_idx` read in
  `entity_df_mutation` / `structure_engine` is a DIFFERENT field, `ReferenceZone.source_event_idx`, renamed in E5);
  it is only exported in the `*_kl_zones.csv` meta (39 rows) and is slice-local on M15 (not in
  `_ZONE_META_IDX_KEYS`). Example H1 BOS zone sid 0 cyc 0: `source_event_idx` 96, `anchor_idx` 96,
  `confirmed_idx` 115. After E4b it would equal `confirmed_idx` on every zone (CTS zones already do: their source is
  CTS_CONFIRMED). Options: (a) keep raw → a redundant copy of `confirmed_idx`; (b) pin to the anchor → a redundant
  copy of `anchor_idx`; (c) **delete it** (rec) as its own byte-identical-except-that-key commit before E4b (39 KL
  meta rows lose the key; nothing else), so E4b = the 34 events idx cells only.
- **Q3 — processing order.** Pin today's order permanently, or later move to moment order + an explicit type rank?
  Example: H1 sid 0 cyc 0 — the BOS is processed at its anchor 96, 19 candles before it is knowable at 115. Rec:
  **pin in E2 (required either way); moment order = post-E**, for the live / Phase-3 driver.
- **Q10 — test-only Scenario-3 / Exception-2 paths** (`structure_engine` scenario 3,
  `compute_structure_from_start`; `test_scenario3.py`, `test_bounded_structure.py`). Rec: **migrate to the anchor
  accessor in E2b** (cheap, behaviour kept). Deleting them is a separate decision about test coverage.

**Needed before E3:**
- **Q18 — the MS in-flight fill horizon (the old E3e).** Rec: **merge into E3a**, because MS ↔ FibTracker parity must
  move together (IN §2.4 item 9; LANDMINES "Scenario 2 anchor agreement"). Measure the merged variant first.
- **Q2 — Fib Scenario 1** `CTS_0 idx >= reversal_confirmed_idx`: a location or a timing question? Example: H1 sid 1
  CTS_0 703 (anchor == moment) vs RC apply 902 → False either way. Rec: **timing**, in E3a.
- **Q8 — `cycle1_bos_idx` cond3 / c0 cond2 fill horizons.** Example: H1 sid 1 cyc 1 BOS anchor 728, moment 748 —
  "did BOS_1 fill cycle 0?" is asked at 728 today; at 748 a fill in (728, 748] would count. Rec: **moment, as its
  own E3a′ `/compare`**, measured first.
- **Q5 — `struct_start` base.** Example: H1 sid 0 base 96 (BOS_0 anchor) vs 115 (the moment CTS_0 + BOS_0 became
  known). Rec: **moment, as E3f, measured**: lifecycle fields are real-time, and nothing of sid 0 can be live
  before 115.
- **Q6 — prev-BOS line END.** Example: `sid=1 start=591 end=902`; identical either way. Rec: **anchor** (a drawn
  line ends at a price location).
- **Q20 — the §7.1 table** (T1, T3, T4 → moment via E3g, one `/compare` each; T2 stays; T5 in E3a).

**Needed before E4:**
- **Q7 — E4 as one `/compare` or two?** Rec: **two** (E4a EST: 3 cells; E4b BOS: 34 + 34), EST first.

**Needed before E5 (can wait):**
- **Q4 `validated_parent_idx`.** Rec: **(b)** an FC-only `parent_bos_anchor_idx`: today it mixes H1 96 with M15 454,
  against its docstring.
- **Q12 `KLZone.source_time` / `StructureLevel.time`.** Example: KL `source_time` holds the timestamp of candle
  115 (the moment) next to `source_price` = the price at 96 (the anchor). Rec: keep both names; document
  `source_time` as the moment's timestamp.
- **Q13 — optional renames.** Rec: none.
- **Q14 — reversal-scope names.** Rec: separate, after Plan E.

## 11. Cold review (2026-09-24) — rev 1 → rev 2

Three parallel lenses (mechanism + predictions; sequencing + tests; contract + naming + docs), ≈ 626k subagent
tokens. I re-verified every BLOCKER / MAJOR claim at HEAD before editing.

| Finding | Severity | Fold-in |
|---|---|---|
| E3d (and E3b / E3a pattern-path) call `event_moment` on pattern-path CTS_UPDATED → `None` → `TypeError` at H1 sid 1 CTS_UPDATED 710 | BLOCKER | new stage E3·0 before every E3; Q1 narrowed |
| E4 completeness proof blind to charts (count parity only) | MAJOR ×2 | figure-JSON x/y diff in every variant; per-lens E2 variants |
| Time halves with no E3 stage (POI transitions, find_ic horizon, chart PB bound, zone_proximity pointer) | MAJOR ×2 | §7.1 table + Q20 |
| `current_candle` / `own_imb_start` are window-end AND horizon in one parameter | MAJOR | §6.2 parameter splits in E2b |
| E3a breaks MS ↔ FibTracker parity (E3e later / parked) | MAJOR | E3e merged into E3a (Q18) |
| `reference_zone` reverse `_TYPE_ORDER` sort ≠ `processing_order_key` | MAJOR | not pinned; own key + tie test |
| Sibling clip `new_ev.idx = abs_idx` would carry E3b's moment into the mirror | MAJOR | split clip time vs raw idx; declared |
| Contract draft: over-reach (removal), conflict with staged E2→E4, gaps | MAJOR | §4.2 rewritten |
| `ev.price` after the flip unspecified | MAJOR | Q19 + ARCHITECTURE price column |
| E1 misses kwarg fixtures; grep gate blind; `wave_candles:492` would raise | MAJOR ×2 | §4.1 / §4.4 |
| E1 / E2 docs lists incomplete (E2 had none) | MAJOR ×2 | §4.4, §6.6 |
| Accessor names collide with locals (`UnboundLocalError`) | MAJOR | qualified-module rule; Q16 |
| E4a run.log 2 KL lines; E3a at-risk cells; `int` shift trap; guard test `pb_start`; importer count; fixture census 17 files / ≈ 301 events; E2 → 4 commits; `location_idx` private; factory kwarg names; `cts_time_idx` illegal; Q9 already decided; `source_event_idx` silent → Q21; E1 grep hits wave-candle realm; PROJECT_PRINCIPLES heading | MINOR / NIT | folded in where they sit |

Confirmed by the review:
- E1 47 + 5;
- E2 keys already in the shift list; 4 CSVs change;
- E3a's 4 named cells trace to `_set_terminal(prev_cycle, activated_at)`;
- E3b / E3d 0;
- E4a 3 cells with IC 678 / 2808 unchanged;
- E4b 34 + 34 in emission order;
- `processing_order_key` reproduces today's order for every type;
- `_run_downstream_pipeline` exists; the E4-simulation test fails at HEAD in both fib modes;
- the IN §2.7 pin lines hold at HEAD;
- baseline 777 + 1 xfail.
