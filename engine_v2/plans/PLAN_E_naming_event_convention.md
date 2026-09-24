# Plan E — naming standard + event-index convention: `ev.idx` = the moment (zones pass, 2026-09-24)

**Status:** rev 2 — cold-reviewed (§11: 3 lenses, 1 BLOCKER + 15 MAJOR findings folded in). **All §10 questions
DECIDED by the user 2026-09-24** (Q11 wording approved as §4.2). Next: E1 + E1b.
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
  untouched — after the unread meta `probe_end_idx` went it is unread and == `trigger_event_idx`; the sibling
  probe's end is the sweep's `hi`, so the §1.8 rename to `probe_end_idx` would misname it (delete vs keep: open);
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

---

## 7. E3 — timing fixes (each its own `/compare`; each flips `# Plan E E3x` markers to `ef.event_moment`)

| Stage | Sites (IN) | Prediction | Verification |
|---|---|---|---|
| **E3·0** pattern-path moment | `market_structure.py:1580`: pattern-path `CTS_UPDATED` gains `meta["confirmed_at"] = int(apply_idx)`; `event_moment` returns it (and the raw path keeps `ev.idx`). Plan F's knowability cut then applies on the pattern path (Plan F §7 follow-up) | events meta +37 keys (7/23/7); the cut effect was measured by Plan F as 0 cells → **re-measure with a variant**; expected 0 other cells | unit: a pattern-path update with apply > idx is cut at the apply. Docs: ARCHITECTURE table `CTS_UPDATED` row; IMBALANCE_FILL_SEMANTICS |
| **E3a** fib + MS-mirror timing | IN §2.3 E3a + §2.4 items 1–6 and 9:<br>- `activated_at` writes;<br>- the fill horizon at EST (`fib_tracker.py:616`);<br>- the split `fill_horizon_idx` of `resolve_cross_cycle_eligibility` / `select_fib_anchor_for_cycle` (§6.2);<br>- the new_cycle terminal / `_mark_first_active`;<br>- the set-if-absent merge;<br>- the `scenario1_revert` terminal;<br>- `:1149` `.get("activated_at", cts_idx)` → direct;<br>- **the MS mirror** `_refresh_poi_inners_for_cycle` / `_update_cycle0_data` horizons (the old proposed "E3e", merged per Q18: IN §2.4 item 9 requires MS ↔ FibTracker parity "in the same change").<br>Scenario 1 per Q2 | **Named: 4 fib_lifecycle cells.** conf line 14 + counter line 2 (sub 3 cyc 0) `end_idx` 2828.0→2829.0; conf line 15 + counter line 3 (sub 3 cyc 1) meta `activated_at` 239→240 (slice-local).<br>**At risk** (the horizon moves one candle at 2829 and at 1224):<br>- sub 3 cyc 1's `cross_failed` single fib + POI IC 2808 (both lenses) if a fill confirms at 2829;<br>- conf sub 0 cyc 2's re-run `_m15_cross_check` on the cross started at 1169 (reactivate / deactivate / version cells);<br>- via the MS mirror, MS events (sd-prox CTS confirmation).<br>H1 0 **only if Q8 stays out of E3a**.<br>**Measure first** with a variant replay; the variant's cell list becomes the prediction | unit: `_make_second_cts_moment_after_extreme_data` via `_run_downstream_pipeline` → cycle-1 fib `start_idx` / `activated_at` 9→10, cycle-0 fib `end_idx` 9→10. Spec: FIB_LIFECYCLE_SPEC §7 cases 2–3, §15.3 |
| **E3a′** *(Q8)* | `cycle1_bos_idx` cond3 fill horizon (`:1544`, `:1624`) + c0 cond2 fill horizon (IN §2.4 item 5) → moment | measure first; H1 cells possible (BOS lag 14–76) | per measurement |
| **E3b** pool clock | `knowable_at_idx` (`sub_structure_pool.py:84-100`) on `ef.event_moment` for `CTS_ESTABLISHED`; the sibling-clip TIME (split in E2b; comment `:481`); `reference_zone.py:335-357` window + the recency sort key; `unified_probe._second_cts_moment` converges on `event_moment`; rewrite the false comment `reference_zone.py:344-347` (hazard H5) | **0**: no lagging EST straddles a sub cap (caps 1940, 2470, 2829, 3611, 3819, 4200) | synthetic cap in [anchor, moment); `test_sub_structure_pool.py:562-564` expectation 10 → 14. Closes PART4 §17.12 for EST |
| **E3c** probe Phase 2 | `unified_probe.py:602` `check_lo` → moment + 1; label `cts0_est=` (`:671`) re-sourced to the moment local | **0**; run.log identical (the window's one Phase-2 run: `p2_iter=1`, lag 0) | unit on the (9, 10) fixture: `check_lo` 10 → 11 |
| **E3d** prev-BOS filter | `orchestrator.py:279-288` `ef.event_moment(ev) >= rv_idx` over CTS_EST + CTS_UPDATED (needs E3·0; rev 1 would have raised `TypeError` at sid 1, pattern-path CTS_UPDATED 710). END value per Q6 | **0**: `[prev_bos_line] sid=1 start_idx=591 end_idx=902` | run.log identical |
| **E3f** *(Q5 = moment)* | `compute_struct_start_by_sid` → min over moments; main `creation_event_idx` likewise or split | **measure first**. Raw base: H1 sid 0 96→115; subs 454→458, 1797→1816, 2365→2368, 2639→2649, 3304→3306, 3760→3786, 4027→4031, 3621→3656 (IN B9). Expected masked (KL/POI/fib start at moments ≥ the base) | `test_structure_lifecycle_moment.py:179-184`, `test_parent_tables.py:116` |
| **E3g** *(Q20)* | the remaining time halves of §7.1 marked "moment" | 0 each (one `/compare` per row; chart row with the figure diff) | per row |

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
- **Docs:** the BOS half (GOTCHAS "BOS_CONFIRMED `ev.idx` Is the BOS Extreme…"; the ARCHITECTURE row;
  KL_ZONES_SPEC `source_event_idx`; MEMORY "Key Architecture Points").

**E4c — pattern-path `CTS_UPDATED`** (Q1 = flip). `idx := apply_idx` + meta `cts_anchor_idx` (E2 readers then use
`ef.cts_anchor_idx` for every CTS_UPDATED, and E2b must already route CTS_UPDATED LOCATION reads through it).
Baseline: 37 pattern-path rows, **1** lagging (conf sub 2 cyc 0: 2468 vs apply 2470, masked by sub 2's start 2470).
Prediction: 1 idx cell; everything else identical (the moment already drives the timing reads since E3·0).

**E4 tests:** the expected-value edits (IN §2.7 "E4, EST part" / "E4, BOS part"); the factory default `idx =
confirmed_at` and the conftest validator's `idx == confirmed_at`; the E4-simulation test becomes a regression pin of
the real emitter.

## 9. E5 — remaining renames + prose

- **Scope:** IN §1.3 (E5 rows), §1.4 (E5 rows), §1.5 (E5 rows), §1.6, §1.7, the §1.9 prose, §3 #13–15, and
  `cts0_est_idx` → `cts0_established_idx`.
- **Internal renames are byte-identical:** 24/24, run.log identical; the `[fib] CTS idx=` labels stay (IN §1.10).
- **Exported:** `validated_parent_idx` per Q4 (triggers CSV header, 2 files).
- Chart hover / legend text stays out of byte-identical steps.

**Post-E (not Plan E):**
- the coordinate-hygiene families (IN §2.8 last row; one `/compare` each);
- the never-established-cycle fallback POI (0 on this window since Plan F);
- the fetch-gate N/A edge case;
- moment-order processing (Q3).

---

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
