# Plan E inputs — naming standard + event-index convention inventory (zones pass, 2026-09-23)

## 0. What this is

**This is the INPUT for writing Plan E. It is not the plan.** Nothing here is approved for implementation: Plan E is
written from this file, cold-reviewed, and its open questions (§4) and the contract-amendment wording go to the user
before any code. It digests the inventories the zones-pass session produced (2026-09-22/23), which until then lived only
in the ephemeral session scratchpad.

- **State:** written at HEAD `fba64f1` (= Plan D source commit `0a4eadc` + its save commit). The `/compare` baseline is
  save **`20260923_172626_0a4eadc`**.
  - **Since then: Plan F** (the imbalance-c3 knowability fix, 2026-09-24, the commit after `754a642`;
    [`PLAN_F_imbalance_c3_knowability.md`](PLAN_F_imbalance_c3_knowability.md)). Canonical rule: IMBALANCE_FILL_SEMANTICS.md
    "Knowability — the c3 rule". In short: an imbalance exists from its first c3 (`ImbalanceInstance.formed_at`);
    `has_unfilled_imbalance` takes the moment of the question as a keyword-only, required `evaluated_at` (FibTracker
    passes `market_structure.event_moment(ev)`; `None` = an explicit no-cut); the POI sweep enters an imbalance at
    `max(inst.formed_at, first_active)`. Plan F's save (made at its `/commit-save`) becomes the `/compare` baseline in
    place of `20260923_172626_0a4eadc`.
    Items below that Plan F touches carry a "Plan F" note.
- **Coordinates:** every line reference is an **`e2e0f89` coordinate unless it is marked HEAD**. Convert with the §0
  offset table.
- **Full per-row inventories:** the sibling folder [`plan_e_inputs/`](plan_e_inputs/README.md) (rendered in full from
  the workflow JSON). This digest supersedes them wherever they disagree.
- **Read together with the durable handoff:**
  - memory `MEMORY.md`, `project_zones_timing_audit_20260922.md`, `feedback_naming_discussions.md`;
  - repo `engine_v2/GLOSSARY.md` "Naming Standard", `engine_v2/ARCHITECTURE.md` "`ev.idx` convention", and
    `engine_v2/plans/PLAN_D_poi_activation_moment.md` §7 / §9.

**Sources.** Three read-only inventory workflows, each consolidated, deduplicated and re-grepped over all of
`engine_v2` except `legacy_2025*`:
- `extreme_final` (`plan_e_inputs/extreme_final.txt`): the final classification of every "extreme" / "ext" name
  (101 rows). Earlier rename proposals are restated in the final vocabulary as `final_name`.
- `anchor_feas` (`plan_e_inputs/anchor_feas.txt`):
  - `cons.anchor_rows`: every "anchor" name (50 rows);
  - `cons.idx_read_rows`: every read of `CTS_ESTABLISHED` / `BOS_CONFIRMED` / `CTS_UPDATED` `.idx`, classified
    LOCATION / TIMING / BOTH (74 rows);
  - `feas`: the flip's blast radius, migration plan, risks, ordering hazards and open questions.
- `naming_inv` (`plan_e_inputs/naming_inv.txt`): `cons.rows` (70), `exported_changes`, `frozen_bridge`, `sizing`,
  `proposed_sequencing`.
- The pre-Plan-D zones-timing audit (`plan_e_inputs/zones_timing_audit.txt`) is the source of several §3 items.

All three inventories were re-verified against **HEAD `e2e0f89`**, and all counts were measured on the baseline
`20260922_195430_aadb887`. Commits since then:
- `37bcf76`: docs only.
- `0a4eadc`: Plan D.
  - **Behaviour:** one lookup line in `zones/poi_zones.py` (old `:470` → new `:482`, `int(cts_event.idx)` →
    `int(cts_event.meta["confirmed_at"])`).
  - **Line shifts:** the same commit rewrote comments in `poi_zones.py` and a docstring in
    `zones/structure_lifecycle.py`, and added tests to `tests/test_pooled_structure_build.py` and
    `tests/test_render_sub_projection.py`. Those four files are shifted (table below).
  - **New file:** `tests/test_poi_activation_moment.py`. Its references in this file are HEAD coordinates.
- `fba64f1`: the save.

The current baseline differs from the inventories' baseline only in Plan D's cells: the confluence POI IC 678 changed
1223→1224 (`confirmed_idx` and `activation_history`), and 3 POI meta cells of `cts_established_idx` were re-valued to
the moment. Every other measured value below stands. **Plan F** then moved further cells (listed in
IMBALANCE_FILL_SEMANTICS.md "c3 knowability (2026-09-24, Plan F)"): 4 POI rows re-timed by one candle; 3 fib
`activated_at` cells; sub 5's cycle-0 fib end; one counter fib_lifecycle row (9→8) and one counter POI row (13→12)
removed; the counter chart 153/125 → 151/124. Values below that these touch carry a Plan F note.

**Line numbers.**
- **Code:** `e2e0f89` coordinates are exact for every `.py` file except the four files below (verified with
  `git diff -U0 e2e0f89 HEAD`).
- **Docs:** doc references (`*.md`) are also `e2e0f89` coordinates, but 17 `.md` files were edited by `37bcf76` /
  `0a4eadc`, so they are stale. **Re-grep them; do not trust the numbers.**

| File (Plan D offsets, verified) | `e2e0f89` line range → add |
|---|---|
| `zones/poi_zones.py` | ≤:367 → 0 · :368–:467 → +1 · :468–:502 → +12 · :503–:791 → +13 · :792–:819 → +17 · :820–:866 → +19 · ≥:867 → +24 (e.g. :470→:482, :586→:599, :766→:779, :820→:839, :832→:851, :912→:936, :984→:1008) |
| `zones/structure_lifecycle.py` | ≤:72 → 0 · :73–:99 → +2 · ≥:100 → +3 (e.g. :100→:103, :148→:151, :174-182→:177-185) |
| `tests/test_pooled_structure_build.py` | ≤:78 → 0 · ≥:79 → +8 (e.g. :128-136 → :136-144) |
| `tests/test_render_sub_projection.py` | ≤:723 → 0 · ≥:724 → +81 (e.g. :745 → :826) |

**Plan F shifted six more `.py` files** (offsets from `754a642`, verified with `git diff -U0`). `754a642` equals
`e2e0f89` for every file below except `poi_zones.py`, which takes the Plan D row first. A line inside a span Plan F
rewrote (mostly the `has_unfilled_imbalance` calls) has no offset: re-grep it.

| File (Plan F offsets, verified) | `754a642` line range → add |
|---|---|
| `zones/fib_tracker.py` | ≤:14 → 0 · :15–:110 → +1 · :111–:134 → +3 · :135–:182 → +11 · :183–:306 → +12 · :307–:488 → +61 · :489–:535 → +79/+80 · :540–:693 → +77 · :694–:747 → +86/+87 · :748–:872 → +90 · :874–:890 → +92 · :891–:1109 → +93 · :1110–:1182 → +105 · :1187–:1315 → +102 · :1316–:1435 → +105 · :1436–:1474 → +94…+102 (rewritten calls) · :1475–:1526 → +95 · :1527–:1574 → +86…+92 (rewritten calls) · :1575–:2003 → +83 · :2004–:2064 → +84 · :2065–:2178 → +95 · ≥:2179 → +96 (e.g. :521→:600, :767→:857, :938→:1031, :1056→:1149, :1131→:1236, :1310→:1412, :2090→:2185, :2342→:2438) |
| `zones/poi_zones.py` | ≤:225 → 0 · :229–:235 → +4 · :236–:516 → +5 · :518–:808 → +7 · :813–:919 → +8 · :921–:1030 → +10 · :1031–:1062 → +20 · ≥:1063 → +25 (e.g. :456→:461, :482→:487, :893→:901, :935→:945, :1074→:1099) |
| `structure/market_structure.py` | ≤:163 → 0 · :164–:2049 → +29 · ≥:2050 → +33 (e.g. :1485→:1514, :1981→:2010, :2050→:2083) |
| `zones/cross_cycle_fib.py` | ≤:69 → 0 · :70–:108 → +2 · :109–:123 → +13 · ≥:124 → +14 (e.g. :117→:130, :149→:163, :156→:170) |
| `common/types.py` | ≤:143 → 0 · :144–:193 → +19 · ≥:195 → +23 |
| `patterns/imbalance.py` | ≤:133 → 0 · `has_unfilled_imbalance` rewritten · ≥:163 → +13 |
| tests | `test_cross_cycle_fib.py` ≥:43 → +7 · `test_main_versioned_cross.py` ≥:40 → +7 · `test_poi_lifecycle.py` ≥:18 → +2 · `test_imbalance.py` ≥:348 → +55 · `test_cross_cycle_fib_routine.py` ≥:228 → +35 · `test_poi_activation_moment.py` comments only (no shift) |

### 0.1 FINAL USER DECISIONS: the authority (verbatim from the session; anything in the sources that contradicts them is SUPERSEDED)

- naming standard = moment: `*_established_idx` / `*_confirmed_idx` / `*_at` / `apply_idx`;
- PATTERN realm: `pattern_anchor_idx` = a pattern's FIRST candle, `pattern_end_idx` = its last,
  `pattern_extreme_idx/_price` = the extreme reached inside a pattern;
- MARKET-STRUCTURE realm: `*_anchor_idx` = an ENDPOINT (start or end) of a structure element (`cts_anchor_idx`,
  `bos_anchor_idx`, KL zone anchor, fib anchors, structure start); "anchor" is NOT retired;
- extreme = a recorded price extreme that is not serving as an endpoint (window searches, running extremes, new-extreme
  checks, price-mapping, candle anatomy). The test: "Does it serve as an endpoint in our structure? Or are we simply
  trying to record an extreme price?"
- The KL-zone anchor family and the fib anchor family (incl. their non-idx companions `anchor_high/low`,
  `anchor_bos/cts_price`, `anchor_type`, `anchor_dir`, `anchor_line`) STAY as named.
- `CTS_CONFIRMED.meta["cts_anchor_idx"]` stays (a CTS endpoint).
- Bare `bos_idx`/`cts_idx` on price-paired records stay bare.
- `REVERSAL_WATCH_START` + `REVERSAL_CANDIDATE` + `CTS_ESTABLISHED` meta `anchor_idx` → `pattern_anchor_idx` (together).
- `_extreme_idx_for_cts_event` → `_cts_anchor_idx_for_event`.
- `st.bos_confirmed` → `st.bos`.
- `_price_confirmation(anchor_idx)` / `anchor_end_idx` → `pattern_end_idx`.
- Event-index flip: BOTH `CTS_ESTABLISHED` and `BOS_CONFIRMED` → `ev.idx` = the moment, extremes/anchors in meta as
  `cts_anchor_idx` / `bos_anchor_idx` (NOT `cts_extreme_idx`).
- Event contract to be amended: a rename/meaning change allowed only as an atomic migration (emitter + all readers +
  docs in one commit, `.get(key, fallback)` reads of that key become direct indexing), NO aliases — exact wording to be
  approved by the user in Plan E.
- Sequence: imbalance-at-c3 fix NEXT (own session) [DONE: Plan F, 2026-09-24] → Plan E written + cold-reviewed (stages: pattern-anchor rename +
  contract amendment → explicit anchor fields (accessors, migrate LOCATION reads, split dual-role calls, pin sort keys)
  → timing fixes one per `/compare` (fib timing 4 rows, pool knowable-at/sibling clip 0, probe Phase-2 `check_lo`,
  prev-BOS) → `CTS_ESTABLISHED` + `BOS_CONFIRMED` flip → remaining renames) → the never-established-cycle fallback POI.

### 0.2 How the sources were translated

- **New event-meta key.** Wherever the feasibility text proposes `cts_extreme_idx` / `bos_extreme_idx` as the new key,
  this file writes **`cts_anchor_idx` / `bos_anchor_idx`**. Both keys are already listed in
  `_EVENT_META_IDX_KEYS` (`multitf/entity_df_mutation.py:108`, `:109`), and `bos_anchor_idx` is dead today. So
  **stage E2 needs no shift-list edit** for either key; only E1's `pattern_anchor_idx` must be added to the list.
- **"Retire anchor" dropped.** The following proposals are dropped: `pattern_start_idx`, `close_break_idx`, `scan_idx`,
  `level_idx`, `source_idx`, `_step_scan`, `detect_best_starting_at`, `_schedule_reversal_from_close_break`,
  KL `anchor_idx` → `extreme_idx`, `fib.extreme_line.*`, `_running_cts_extreme`, `select_fib_span_for_cycle`, and the
  `*_extreme_idx` spellings of wave-candle / probe CTS anchors.
  - Pattern-realm names ("anchor" = first candle, including the MS scan candle and the close-break candle) are
    consistent and stay.
- **"Additive aliases, never renames / old keys never removed" is superseded.** The contract amendment allows atomic
  rename migrations with **no aliases**.
- **Moment spellings.** Proposals spelled `*_moment_idx` / `cts_moment` become the moment spellings in 0.1. Plan D
  landed with the name `cts_established_idx` holding the moment.
- **Endpoint values.** `*_extreme_idx` proposals for endpoint values (naming_inv, feas) become `*_anchor_idx`.
  Price-paired bare record names stay bare.

### 0.3 Stage keys used below

| Key | Stage (per 0.1) |
|---|---|
| **E1** | the pattern-anchor rename + the event-contract amendment (+ pattern-realm internal renames) |
| **E2** | explicit anchor fields:<br>- add meta `cts_anchor_idx` on CTS_ESTABLISHED and `bos_anchor_idx` on BOS_CONFIRMED;<br>- accessors;<br>- migrate every LOCATION read;<br>- split the dual-role calls;<br>- pin the sort keys;<br>- decouple the MS BOS state;<br>- the renames coupled to the flip |
| **E3a–d** | the timing fixes, one per `/compare`:<br>- a: fib timing;<br>- b: pool knowable-at / sibling clip (+ reference-zone recency);<br>- c: probe Phase-2 `check_lo`;<br>- d: prev-BOS filter.<br>- **e (proposed, not in the user's list):** the MS in-flight POI-inner resolver as-of (§2.3), its own `/compare`.<br>Optional or undecided items are marked E3? |
| **E4** | the `CTS_ESTABLISHED` + `BOS_CONFIRMED` idx flip (emit sites only) + the docs that invert |
| **E5** | the remaining renames and (unless moved before E2, §1.8) the deletions:<br>- internal ones are byte-identical;<br>- exported ones go under a `/compare` rename map |
| **post** | the never-established-cycle fallback POI (no live case on the reference window since Plan F, §3 #19); the coordinate-hygiene families (each its own `/compare`) |

### 0.4 Measured facts the stages rely on (baseline `20260922_195430_aadb887`; unchanged by Plan D unless noted)

- **`CTS_ESTABLISHED`**
  - Emit site: one, `market_structure.py:1485` (ctor `:1830`).
  - `idx` ≠ `confirmed_at` on 3 of 34 rows, which are 2 unique M15 cycles, both lag 1 and both `one_maru_opposite`
    CONFIRMED:
    - confluence sub 0 cyc 2: 1223 vs 1224 (pattern anchor 1221);
    - sub 3 cyc 1: 2828 vs 2829 (pattern anchor 2824), mirrored in both lenses.
  - H1 lag is 0 on all 5 cycles.
  - The pattern anchor (`meta["anchor_idx"]`) is below the CTS anchor on all 34 rows.
- **`BOS_CONFIRMED`**
  - Emit sites: two, `:1502` and `:1517`.
  - It lags on 34 of 34 rows:
    - H1 (lag 14–76): 96→115, 591→652, 689→703, 728→748, 826→902;
    - confluence: 21 rows, lag 2–411;
    - counter: 8 rows, lag 4–285.
  - BOS_0 is the minimum event idx on all 12 structures.
- **`CTS_UPDATED`**
  - 53 / 259 / 114 rows (H1 / confluence / counter), of which 7 / 23 / 7 are pattern-path.
  - No `CTS_UPDATED` carries `confirmed_at`.
  - The raw path is `meta["via"] == CTS_UPDATED_RAW_VIA` (`"replay_raw"`; a named constant in `market_structure.py`
    since Plan F): its `idx` is the processing candle, a moment. Every other `via` is a breakout-pattern name, whose
    `idx` is the CTS anchor.
  - The pattern path has 1 lagging instance: confluence sub 2 cyc 0, 2468 vs apply 2470, a same-price duplicate of a
    raw update, masked by sub 2's start 2470.
- **Probes on the window**
  - The 9 probe runs used `ref = cts_confirmed` ×5, `cts_updated` ×2 and `ad_hoc_bos_0` ×2, never `cts_established`.
  - There were 3 cache hits.
  - The only Phase-2 run (sub 0) has a lag-0 CTS_0 at 458 (`p2_iter=1`, no reset; run.log:1794).
- **CTS_CONFIRMED timing**
  - The smallest gap from a CTS_CONFIRMED to its cycle's EST moment is 2.
  - No `CTS_THRESHOLD_UPDATED` falls near 1223 or 2828.
- **Pool caps**
  - The sub caps are 1940 and 3611.

---

## 1. Rename table (FINAL vocabulary)

Columns:
- **Exp?** says whether the name is exported, with its `/compare` effect.
- **Frz?** says whether the name is frozen by today's event contract. Frozen names migrate only under the E1 amendment.

### 1.1 Event meta keys and event idx (frozen today → atomic migrations under the amendment)

| Current → final | Emitted / read (file:line) | Exp? (/compare) | Frz? | Stage |
|---|---|---|---|---|
| `CTS_ESTABLISHED.meta["anchor_idx"]` → **`pattern_anchor_idx`** | **Emitted:** `market_structure.py:1491`, as `int(ev.start_idx) if ev.start_idx is not None else int(apply_idx)`. The fallback (a moment) is never taken (extreme_final #73). Draft suggestion, not from the sources: replace it with an assert in E1. **Readers:** `wave_candles.py:295`, `:492-495`, `:553`. These are `.get('anchor_idx')` reads: if they are not moved in the same commit they silently return None and skip the scan (WVMI then changes). **Mirror:** `entity_df_mutation.py:103-112` (`:106`): replace `"anchor_idx"` with `"pattern_anchor_idx"` in `_EVENT_META_IDX_KEYS` only; KL still needs `"anchor_idx"` in `_ZONE_META_IDX_KEYS` `:116`. **Stale comment:** `:1490` "required for Scenario 2 Exception #2". | Key text only, on events meta rows H1 5 / conf 21 / counter 8 and H1 `structure_levels.csv` meta 5; values identical | yes | E1 |
| `REVERSAL_CANDIDATE.meta["anchor_idx"]` → **`pattern_anchor_idx`** | **Emitted:** `:938` (`:894-938`). **Read:** `export_plotly.py:1550-1551` (`rc_by_anchor`), `:1584`. **Docs:** `GLOSSARY.md:28`. Both chart reads are `.get("anchor_idx", ev.idx)` (`:1551`, `:1558`) → `ev.meta["pattern_anchor_idx"]` (direct index, per the amendment) in E1. | events meta: H1 1 (897, apply 902), conf 4 (1936, 2465, 2824, 4196) | yes | E1 (lockstep: the H1 chart joins the two REVERSAL events on this key, `export_plotly.py:1551/1558`) |
| `REVERSAL_WATCH_START.meta["anchor_idx"]` → **`pattern_anchor_idx`** | **Emitted:** `:799` (`:780-806`); both events are emitted with close-break candle `i` from `:710-711` / `:745-746`. **Read:** `export_plotly.py:1558`. | events meta: H1 1, conf 7 | yes | E1 |
| Test fixtures that build these events with meta `anchor_idx` | `test_structure_lifecycle_moment.py:47` (CTS_ESTABLISHED `extra`) and the event fixtures in `test_wave_candles.py` (e.g. `:390`, `:434`). These sit among KL-zone `anchor_idx` fixtures, which STAY (`test_wave_candles.py:58`, `:163`, `:275`…; `test_sub_wvmi.py:60`; `test_wvmi.py:54`; `test_render_sub_projection.py:705-710`). | — | — | E1 (separate the two realms by hand) |
| (new) `CTS_ESTABLISHED.meta["cts_anchor_idx"]` (== `ev.idx` until E4) | Emitted at `:1485` through a new `cts_anchor_idx` param on `_emit_cts_established` (`:1823-1832`). Already in `_EVENT_META_IDX_KEYS` `:108`. | events meta +key on 34 rows (H1 5 / conf 21 / counter 8) + 5 H1 levels rows | new | E2 |
| (new) `BOS_CONFIRMED.meta["bos_anchor_idx"]` (== `ev.idx` until E4) | Emitted at `:1502` and `:1517` through a new `bos_anchor_idx` param on `_emit_bos_confirmed` (`:1927-1938`). Listed but dead in `_EVENT_META_IDX_KEYS` `:109`. | events meta +key on 34 BOS rows (H1 5 / conf 21 / counter 8) + 5 H1 levels rows | new | E2 |
| `CTS_ESTABLISHED.idx`, `BOS_CONFIRMED.idx` → **the moment** (`apply_idx`); `confirmed_at` kept and asserted `== idx` | `:1485`; `:1502`, `:1517` | events `idx`:<br>- EST: 3 cells;<br>- BOS: 34 cells.<br>See 2.8 | yes | E4 |
| `RANGE_STARTED.meta["cts_idx"]` stays bare (decided: it pairs with `meta["cts_price"]`; extreme_final's optional `cts_anchor_idx` rename is superseded) | `:1208`, `:1332`, `:2124` (emit blocks `:1198-1212`, `:1325-1340`, `:2117-2130`) | none (rows carrying it: H1 10 / conf 39 / counter 15). It is slice-local on M15 (§3 #4) | yes | — (stays; add RANGE_STARTED meta to the GLOSSARY "Bare element idx" row in E1 docs) |
| Kept as they are:<br>- `CTS_CONFIRMED`/`CTS_RECONFIRMED.meta["cts_anchor_idx"]`;<br>- `confirmed_at` (emitted `:1493`, `:1507`, `:1522`, `:1868`, `:1909`);<br>- REVERSAL_CANDIDATE `apply_idx` / `expires_idx` (`:939`);<br>- `effective_idx`;<br>- `pb_start` (`:1508`, `:1523`; reset `:1548`);<br>- BOS `source`;<br>- `rv_anchor_failed`.<br>The events CSV `idx` column is written by `debug/export_events.py:13-21` | §1.10 | — | yes | — |

### 1.2 Exported non-event names (CSV headers / zone meta / final.csv)

| Current → final | Where | Exp? (/compare) | Stage |
|---|---|---|---|
| H1 final.csv col 66 `pending_reversal_anchor_idx` → **`pending_reversal_pattern_anchor_idx`** | `market_structure.py:38`, `250`, `891`, `920`, `925`, `962`, `2424-2426`, `2478-2480`; `unified_probe.py:113`; `entity_df_mutation.py:95` | 1 H1 final.csv header line. The sources rate it low priority | E1 (memory records it in the user's 2026-09-23 pattern-anchor decision; confirm, Q4.9) |
| triggers CSV `validated_parent_idx` → **`ref_anchor_idx`** (options open, Q4.4) | **Field:** `sub_structure_pool.py:139`, `:165`; `lifecycle_sweep.py:89`, `:337-338`. **Set:** `entity_df_mutation.py:675` (FC), `:822` (sibling), `:943`/`:983` (reversal = None). **Read:** `debug/probe_fc_finalize.py:166`. **Header:** auto-derived, `debug/export_sub_tables.py:34` (`_TRIGGER_COLUMNS` from `fields(TriggerRecord)`). **Docs:** `GLOSSARY.md:202`. | 2 header lines (conf 8 rows, counter 3 rows); values identical (96.0/591.0/826.0/3760.0; 2609/3621/4000). 41 test lines in 8 files | E5 |
| `StructureLevel.time` → keep the name; **re-source** it from meta `cts_anchor_idx`/`bos_anchor_idx` | **Def:** `common/types.py:62`. **Set:** `market_structure.py:2659`, `:2669` (`t.iloc[ev.idx]`, `:2643-2677`). **Export:** `debug/export_structure.py:13-24` (`:16-17`). **Passed through:** `orchestrator.py:50`, `:597`, `run_replay.py:251`. The charts never read it. | 0 if re-sourced in E2. If not, the 5 BOS rows move at E4 (`anchor_time` was proposed as a rename; Q4.12) | E2 |
| KL meta `source_event_idx`: keep ("the source event's ev.idx") | `kl_zones_v1.py:888`, `:951` (doc `:710`) | Its value becomes the moment on every BOS-zone row at E4: 34 rows (H1 5 / conf 21 / counter 8). It is slice-local on the 29 M15 rows | E4 (doc) |
| `KLZone.source_time`: undecided (naming_inv proposed `confirmed_time`; it holds the time of the raw, unclamped `confirmed_at` next to `source_price` = the anchor price) | `common/types.py:285`; `kl_zones_v1.py:941-942`; `debug/export_zones.py:54`; 6 test lines in 4 files | header of 3 `*_kl_zones.csv` (10/21/8 rows) | Q4.12 |
| `debug/zone_proximity_diag.py` `cts_idx` / `this_cts_idx` / `window_span_post_cts` (a moment: the proximity-trigger candle) → delete the file (its docstring says so) or `cts_prox_idx` | `:17`, `:220`, `:278`, `:282-283` | /tmp CSVs only, not in `/compare` | E5 |
| DONE: POI meta `cts_established_idx` now holds the moment (Plan D) | `poi_zones.py:482`/`:599` (HEAD). The lookup is built at `e2e0f89` `:366-374`; the local is at `:469-471`, the kwarg at `:520`, the param at `:770`, the use at `:791`/`:820`, and the label `cts_est=` at `:613` (`:612-617`) | landed | — |
| Name kept, value fixed: fib meta `activated_at` (moment) | E3a | 4 rows | E3a |

### 1.3 Pattern realm, internal (byte-identical)

| Current → final | Where | Stage |
|---|---|---|
| `TrueFirstBreakout.anchor_idx` → `pattern_anchor_idx` | `true_first_breakout.py:76`, `:249` (never read; MS uses `tfb.pattern.start_idx`, `market_structure.py:1066`); tests `test_true_first_breakout.py:109-118`, `156`, `168-185`, `198`, `255`, `312-328` | E1 |
| `_price_confirmation(anchor_idx)` / `_price_confirmation_1step(anchor_end_idx)` → `pattern_end_idx` | `structure_patterns.py:118-132` (called `:138-139` with `event.end_idx`), `:151-176` (called `:335-336`, `:637-638`, `:705-706`) | E1 |
| wave_candles locals `first_bo_anchor` / `anchor` / `pattern_anchor` → `pattern_anchor_idx` | `wave_candles.py:288`, `:295-303`, `:490-499`, `:546-559` (`:546`: "window [pattern_anchor, cts_anchor_idx+5]") | E1 (same commit as the key) |
| `TrueFirstBreakout.extreme_idx` / `extreme_price` → `pattern_extreme_idx` / `pattern_extreme_price` | `true_first_breakout.py:78-79`, set `:251-252`; tests `test_true_first_breakout.py:116-161`, `:233`; `test_ms_cts0_scan.py:75-76` | E5 (or E1: same realm, byte-identical) |
| `_full_pattern_extreme` (TFB) + `MarketStructure._cts_from_breakout_event` → one shared `_pattern_extreme(...)` → `(pattern_extreme_idx, pattern_extreme_price)`. MS keeps `_cts_from_breakout_event` as a thin adapter returning `(cts_anchor_idx, cts_price)`: the only place a pattern extreme becomes a CTS anchor | `true_first_breakout.py:90-122` (called `:238`); `market_structure.py:1658-1686` (called `:1462`) | E5 |
| TFB locals `ext` / `ext_idx` / `ext_price` → `pattern_extreme_idx` / `_price` | `true_first_breakout.py:238-242`, `251-252` | E5 |
| `_is_strict_new_extreme(current_start, extreme_idx, extreme_price, direction)`: keep the function name; rename its params → `pattern_extreme_idx` / `_price` | `true_first_breakout.py:125-147` (called `:242`); tests `test_strict_new_extreme_*`, `test_true_first_breakout.py:108-141` | E5 |
| Docstring pseudo-id `extreme_candle` (+ test name `..._before_extreme_candle`) → `pattern_extreme_idx` | `true_first_breakout.py:23`, `:196`; `tests/test_true_first_breakout.py:12`, `:105`, `:130` | E5 |
| `continuous()` local `extreme` + `PatternEvent.debug['extreme']`: **keeps its own name** (it spans c0..c2, excludes the confirm candle and is the confirmation threshold). Optional `c0_c2_extreme_price`; never `pattern_extreme_price` | `structure_patterns.py:221-222`, `310`, `322`, `328` (2-candle twin unnamed at `confirmation_threshold()` `:113-116`) | — (exact name: Q4.13) |
| `close_to_ext` → `c1_close_near_c0_extreme` | `structure_patterns.py:242-248` | E5 |

### 1.4 Names containing "extreme" that hold an endpoint → "anchor" (internal)

| Current → final | Where | Stage |
|---|---|---|
| `_extreme_idx_for_cts_event` → **`_cts_anchor_idx_for_event`**; its ESTABLISHED branch reads `meta["cts_anchor_idx"]`. The CONFIRMED fallback `.get('cts_anchor_idx', ev.idx)` becomes direct indexing | `reference_zone.py:88-97`, called `:358` | **E2** (pool keys; §2.2 L1) |
| Local `extreme_idx` in `build_reference_zone_from_cts_event` → `cts_anchor_idx` | `reference_zone.py:358`, `:374`, `:378`, `:395` | E2 |
| `_derive_cts_zone_ad_hoc(extreme_idx)` → param `anchor_idx` (KL family), or inline the one-line alias | `reference_zone.py:257-262` | E2 |
| `parent_extreme_idx` / `parent_extreme_time` (FC resolver) → `parent_bos_anchor_idx` / `parent_bos_anchor_time` | `entity_df_mutation.py:623` (def), `625-627`, `640-641`, `644`, `647`, `652`, `675` (→ `validated_parent_idx`) | **E2**: coupled with `first_confluence_trigger.py:88` reading `meta["bos_anchor_idx"]` |
| `cts1_ext=` labels → `cts1_anchor=` (value from `meta["cts_anchor_idx"]`) | `unified_probe.py:558` (`:555-560`); regex `tests/test_unified_probe.py:782` (`'cts1_ext=10 cts1_moment=10'`: passes silently on a fixture where the two are equal); `debug/probe_fc_finalize.py:27`, `:150`, `:160` (`:150-161`). Leave the landed Plan B text (`plans/PLAN_B_double_cts_early_stop.md:8`, `216-217`, `309-311`) as history | E2 |
| `bos_extreme_abs` → `bos_anchor_abs`, reading `meta["bos_anchor_idx"]` (`int(bos.idx)+slice_begin` gives 57, not 55, after the flip) | `tests/test_render_sub_projection.py:618-638` | E2 |
| Fixture `_make_second_cts_moment_after_extreme_data` → `..._after_anchor_data` | `tests/test_unified_probe.py:210`. Imported by name in `tests/test_poi_activation_moment.py:16`, `:34` (+10 uses; Plan D) and `PLAN_D §5` | E5 |
| Test names contrasting the moment with "the extreme" → `..._not_the_anchor` / `anchor_equals_moment` / `probe_end_idx_is_cts_anchor_...` / `..._record_anchor` | `test_unified_probe.py:837`, `:1047`; `test_parent_tables.py:111`, `:257`; `test_structure_lifecycle_moment.py:94`, `:115`, `:206`; `test_ms_stop_after_cts.py:382`; `test_first_confluence_trigger.py:60`; `test_first_trigger_migration.py:518` | E5 |
| RESOLVED: `cts_extreme` comment (`poi_zones.py:862-866`) | Plan D rewrote it (HEAD `:881-890`, "Load-bearing: CTS_ESTABLISHED.idx (the CTS anchor) <= cts_established_idx (the moment)…") | Plan D rewrote it; the new text inverts at E4 (→ §1.9) |

### 1.5 Bare or moment-spelled names holding an anchor → `*_anchor_idx`

Each name below reads an event `.idx` that flips. The rename marks that its source has moved to meta, which makes the
E2 diff self-checking.

| Current → final | Where | Stage |
|---|---|---|
| MS establish-block locals `cts_idx` / `bos_idx` (+ helper returns) → `cts_anchor_idx` / `bos_anchor_idx` | `market_structure.py:1462`, `1485-1486`, `1501-1503`, `1516-1518`, `1541`, `1551`, `1554`; returns `:1677-1686`, `:1744-1765`, `:2229`, `:2253-2261` | E2 |
| `_initial_bos_before_first_cts(cts_idx)` → param `cts_anchor_idx`. Flag the dormant fallback at `:2240/:2242` (`_select_bos_on_breakout`), which passes `breakout_apply_idx`, a MOMENT, into it | `:1739-1765`; called `:1501`, `:2242` | E2 |
| `_emit_cts_established(idx)` / `_emit_bos_confirmed(idx)`: add params `cts_anchor_idx` / `bos_anchor_idx` → meta (E2). E4 then passes `apply_idx` as `idx`. (`_emit_cts_confirmed_once(idx)` takes the moment and is fine, `:1851`) | `:1823-1832`, `:1927-1938`; called `:1485`, `:1502`, `:1517` | E2 / E4 |
| **`st.bos_confirmed` → `st.bos`** (a bare Point parallel to `st.cts`), built from the `bos_anchor_idx` param and **not** from the emitted idx | def `:183`; set `:1937`; read `:1995`, `2005-2006`, `2035`, `2040-2041`, `2498-2499` | **E2** (the decoupling must precede E4) |
| `st.cts_confirmed_for_idx` → `confirmed_cts_anchor_idx` (it sits next to `cts_confirmed_idx` `:210`, which IS the moment) | `:175`, `:1528`, `:1859`, `:1882` | E5 |
| MS local `cts_anchor` → `cts_anchor_idx` (optional) | `:1858-1859`, `:1869`, `:1882`, `:1903`, `:1910` | E5 |
| structure_engine locals `cts_est_idx` / `cts_established_idx` → `cts_anchor_idx` (reading meta after the flip), **or delete** the test-only paths (Q4.10) | `structure_engine.py:573-576`, `:777-779`; Scenario-3 `cts_est[0].idx + 1`, `cts_est[1].idx` at `:427-432`, `:449-484`. No production caller (only `tests/test_scenario3.py`, `tests/test_bounded_structure.py`). `MARKET_STRUCTURE_SPEC.md:505/555` says keying on the anchor is deliberate. Comments `:463-465`, `:574` wrongly call the candle "the pullback confirmation" | E2 or delete |
| identify_start `StartDecision.meta["last_cts_confirmed_idx"/"_price"]` + local `cts_idx` → `last_confirmed_cts_anchor_idx` / `last_confirmed_cts_price`; local `cts_anchor_idx` | `identify_start.py:166-176`, `:209-210` (reason `'scenario_2:base_last_cts_confirmed'` `:172`); no reader; test-only scenario 2 | E5 |
| `ReferenceZone.source_event_idx` → `anchor_idx` | **Def:** `reference_zone.py:85`. **Doc:** `:62-72`; it lists never-constructed sources `'parent_cts'`/`'parent_bos'` at `:81-82`. **Set:** `:144`, `:151`, `:250`, `:374`, `:395`. **Read:** `structure_engine.py:278`, `entity_df_mutation.py:448`, `:774`, `:798`, `:913`. **Tests:** 34 lines in 4 files. | E5 (byte-identical) |
| first_confluence_trigger local `end_idx` → `probe_end_idx` | `first_confluence_trigger.py:93-103`, `:123` | E5 |
| FC resolver `parent_end_idx` / `parent_cts_time` → `parent_cts_anchor_idx` / `parent_cts_anchor_time` | `entity_df_mutation.py:624`, `:645-649` (interpolated into `ProbeFailure.detail`; 0 baseline rows) | E5 |
| FC resolver `m15_end_idx` → `m15_probe_end_idx` (the price-mapping output; differs from `LowerTFResult.meta["m15_end_idx"]` `:1236-1237`, which is the lifecycle end) | `entity_df_mutation.py:649-668` | E5 |
| `fb_idx` (sibling fallback) → `fallback_anchor_idx`. It also collides with `WVMIRecord.fb_idx`, which is shifted at `:321`/`:371` | `entity_df_mutation.py:784-795` | E5 |
| first_confluence_pipeline local `bos_idx` → `bos_anchor_idx`. `FirstConfluenceTrigger.input_idx` keeps its role name; its docstring "(== BOS_CONFIRMED.ev.idx)" becomes false | `first_confluence_pipeline.py:34-39`; `types.py:168`; set `first_confluence_trigger.py:88` | E2 |
| orchestrator `bos_by_cycle` / tuple local `bos_idx` → `bos_anchor_by_cycle` / `bos_anchor_idx` | `orchestrator.py:166`, `:209`, `:212-213`, `:224-226` | E2 |
| orchestrator `last_bos_by_sid` → `last_bos_anchor_by_sid`; `prev_bos_lines` keys `start_idx`/`end_idx` → `bos_anchor_idx` / `cts_anchor_idx` (historical values under real-time lifecycle names today). Which value the line END takes is Q4.6 | `orchestrator.py:262-298` (label `[prev_bos_line]`); shifted at `entity_df_mutation.py:338-341`; read `export_plotly.py:2688-2698`, `export_m15_chart.py:760`, `:796`, `:2593` (naming_inv: `:1695`, `:2595`) | E2 |
| uc1 local `cts_idx` → `cts_anchor_idx`; the fallback to `cts_ev.idx` (a moment) becomes direct indexing; delete meta key `cts_idx` (no reader) | `uc1_trigger.py:80-92`, `:117`, `:121` | E5 |
| `FibTracker.on_cts_established(bos_idx)` → `bos_anchor_idx`; local `cts_idx = int(event.idx)` → `cts_anchor_idx` (from meta) **plus a new local `cts_established_idx` = the moment** for the timing uses (§2.4). Plan F: the public `on_cts_established` is now a wrapper that runs the body `_on_cts_established` under `_evaluating(event)`, so the handler already holds the moment as `self._evaluated_at` (`event_moment(event)` = `confirmed_at`); today only the imbalance knowability cut reads it | `fib_tracker.py:480-565` (param `:484`; `:521`; fill as-of `:537`; `_bos_by_cycle` `:542`); Scenario-1 check `:768` | E2 (split) → E3a (timing uses switch) |
| poi `cts_idx_at_t` → `cts_anchor_idx_at_t` (it forces the split: the transition time stays the moment, cond1's location comes from meta) | `poi_zones.py:832`, `:840`, `:871`, `:912`, `:930`, `:946` (HEAD +19/+24) | E2 |
| `compute_bos_inner_from_event(bos_idx)` → `bos_anchor_idx` (the caller passes it positionally) | `kl_zones_v1.py:1103-1122`; caller `market_structure.py:1539-1542` | E2 |
| charts `next_bos_idx` → `next_bos_anchor_idx` (both charts) | `export_plotly.py:1197` (`:1192-1223`); `export_m15_chart.py:588` (`:584-612`), `:1183`, `:2077` (`:2073-2097`), `:2138` | E2 |
| probe_fc_finalize `bos_idx_h1` / header "BOS" → `bos_anchor_h1` | `debug/probe_fc_finalize.py:175`, `:208` | E5 |
| sibling-clip local `abs_idx` → `abs_ev_idx` (feas: `abs_knowable_idx`), decided together with the E3b clip fix | `entity_df_mutation.py:481` | E5 |
| `cts0_ref` → `bos0_ref` (it holds a BOS_0 ReferenceZone from `build_ad_hoc_bos0_reference_zone`) | `structure_engine.py:188` (+ its uses) | E5 |

### 1.6 Bare names holding a MOMENT → moment spelling

| Current → final | Where | Stage |
|---|---|---|
| zone_proximity `bos_conf_idx_by_key` / `next_bos_idx` → `bos_confirmed_idx_by_key` / `next_bos_confirmed_idx`; comment "not BOS extreme" → "not the BOS anchor" | `zone_proximity.py:306-311` (`:310`), `:348-350` | E5 |
| compute_cycle_lifecycle `cts_est_by_key` / loop `cts_idx` → `cts_established_idx_by_key` / `cts_established_idx` (naming_inv's `cts_moment` is superseded) | `structure_lifecycle.py:100-114`, `:118-120` (HEAD +3) | E5 |
| wave_candles `raw_cts` (CTS_CONFIRMED idx, a moment) → `raw_cts_confirmed_idx` | `wave_candles.py:624` | E5 |
| Reversal scope, undecided: `reversal_confirmed_by_sid` / FibTracker `reversal_confirmed_idx` hold `REVERSAL_CANDIDATE.meta["apply_idx"]`, the *scheduled* moment (proposed `reversal_apply_idx_by_sid`). `_synth_reversal_trigger` `reversal_apply_idx` holds the *confirmed* R (proposed `reversal_idx`) | `orchestrator.py:148-155`; `fib_tracker.py:352`, `:486`, `:678`, `:743`, `:804`, `:1109`; `entity_df_mutation.py:1020`, `:1036` | Q4.14. **The orchestrator half RESOLVED 2026-09-27 (MAIN B):** re-sourced to the realised reversal (`compute_reversal_idx_by_sid`), renamed `reversal_idx_by_new_sid`; the FibTracker name is accurate since. `_synth_reversal_trigger` open |

### 1.7 Misnomers ("ext" = extension)

| Current → final | Where | Stage |
|---|---|---|
| `ext_end_idx` (param of `_replacement_break_point`) → `extend_to_idx` | `export_m15_chart.py:431` (def), `460`, `482`, `493`; caller `:628` | E5 |
| `"EXT"` point tag → `"EXTEND"` (optional) | `export_m15_chart.py:646`, `:2121` (appended), `:1170`, `:2128` (filters), `:509` (doc) | E5 |

### 1.8 Deletions (byte-identical; stage open for Plan E: before E2 avoids migrating dead `.idx` reads through the E2 grep gate (naming_inv step 2a put deletions first); otherwise E5 — Q4.17)

| Delete | Where | Size |
|---|---|---|
| The retired `lifecycle_end_idx` chain: `cts_est_idx_by_key` + `next_cycle_start` + `lifecycle_end_idx` (+ the trigger-side `reversal_idx_by_sid`). It is unread since Plan C; its comments calling `CTS_ESTABLISHED.ev.idx` "canonical per PART4 §5" are stale | `first_confluence_trigger.py:41`, `50`, `60-69`, `106-115`, `125`; `subsequent_confluence_trigger.py:17`, `78`, `88-95`, `109-118`, `141`; `subsequent_counter_trigger.py:19`, `78`, `88-95`, `109-118`, `142`; `uc1_trigger.py:48-55`, `95-103`, `115`, `131` (the run.log label `[uc1_trigger] lifecycle_end_idx=`); `types.py:27`, `107`, `145`, `160`, `171`; pass-through in the 3 `*_pipeline.py`. **Test pins:** `test_first_confluence_trigger.py:63`, `:76-118`; `test_subsequent_confluence_trigger.py:130-160`; `test_subsequent_counter_trigger.py:181-213` | 33 prod lines / 9 files; 34 test lines / 12 files |
| `validated_h1_start` (LowerTFResult.meta) / `validated_parent_start` (SidRecord.meta): no reader, not in `_SUBS_COLUMNS` | `entity_df_mutation.py:1245`; `sid_records.py:81`, `:107` | 3 prod lines; 21 test lines / 6 files |
| `MultiTFTrigger.start_time` / `start_price` (unread; the FC `start_price` takes the wrong candle side) | `types.py:25-26`; set `uc1_trigger.py:87-92`, `first_confluence_pipeline.py:35-39`, `subsequent_confluence_pipeline.py:32-37` | ~27 prod / ~25 test lines |
| MS dead state `proximity_confirmed_idx` | `market_structure.py` (naming_inv row 61) | — |
| Dead `_is_new_cts_extreme` + commented `cts_ext` + `_cts_price_at` | `market_structure.py:1693-1699`, `:1688`; commented caller `:1251-1254` | — |
| Unread `probe_end_idx` meta key on first_counter / subsequent_*; the residual `Subsequent*Trigger.end_idx` → `probe_end_idx` (Plan C residual) | `uc1_trigger.py:122`; `subsequent_*_pipeline.py:54-55`; `types.py:105`, `:143` | — |
| uc1 meta key `cts_idx` (no reader; same row as the uc1 rename in §1.5) | `uc1_trigger.py:117`, `:121` | — |
| `debug/zone_proximity_diag.py` (marked "delete after the investigation") | — | — |
| `common/types.Zone` (no importers at HEAD; `KLZone` / `POIZone` / `ReferenceZone` are separate classes) | `common/types.py:70` | — |
| Optional (fib family kept): the write-only `FibRetracement.anchor_high_idx` / `anchor_low_idx` fields | `features/fibonacci.py:39-40`, `:102-103`, `:143-144`, `:190-191` | Q4.15 |

### 1.9 Prose to reword (with E4 unless noted: every "ev.idx IS the extreme" sentence inverts)

- **Code prose where "extreme" means a BOS/CTS endpoint → "CTS anchor" / "BOS anchor":**
  - `market_structure.py:495-496`, `1183`, `1459-1470` (also stale: "new-extreme check below", removed
    `_cts0_new_extreme_passes`), `1660`, `1873`, `2026`;
  - `structure_engine.py:246`;
  - `reference_zone.py:12`, `62-72`, `92-93`, `284-285`;
  - `unified_probe.py:452-454`, `548-553`;
  - `fib_tracker.py:1638-1639`, `1935-1936`, `1960`;
  - `structure_lifecycle.py:15`, `70-73`, `98-99` (Plan D already says "CTS anchor"; `:99` "never a lifecycle
    value" inverts at E4);
  - `poi_zones.py` HEAD `:468-479` and `:881-890` (Plan D text: "CTS_ESTABLISHED.idx (the CTS anchor)" inverts at E4;
    the pre-window reasoning must be restated for idx = moment, see §2.2 L3);
  - `kl_zones_v1.py:705-713`, `887`, `951`;
  - `sub_structure_pool.py:84-101`;
  - `parent_tables.py:15`, `30`, `113-115`;
  - `entity_df_mutation.py:388-389`, `448-450`, `609-613`, `765`;
  - `first_confluence_trigger.py:5-9`, `35-37`, `64-68`, `93-99`;
  - `first_confluence_pipeline.py:5-8`;
  - `types.py:153-154`, `168-169`;
  - `uc1_trigger.py:29`, `48`, `80-84`, `94`;
  - `export_m15_chart.py:306-307`;
  - `export_plotly.py:1105`.
- **Code prose where "extreme" is raw: keep.**
  - `market_structure.py:222`, `637`, `729`, `861`, `1287`, `1301`, `1512`, `1704`, `1741` (stale "Cycle 1 BOS"),
    `1888`, `2067`, `2185-2193`, `2220`;
  - `unified_probe.py:32`, `231`, `323-325`, `358`;
  - `reference_zone.py:172`, `176`, `233`, `325`, `327`;
  - `kl_zones_v1.py:369-382`, `403`, `438-449`, `469`, `835`;
  - `fib_tracker.py:202`, `1383`, `1509`, `1871`, `2072`, `2391`;
  - `subsequent_*_trigger.py:5-9`;
  - `types.py:91`, `104`, `123`, `142`;
  - `subsequent_*_pipeline.py:5`, `35`;
  - `parent_tables.py:25`;
  - `entity_df_mutation.py:634-636`;
  - `export_m15_chart.py:438-446`, `2626`;
  - `export_plotly.py:1208`;
  - `run_replay.py:128`.
- **"Anchor" / "extreme" prose fixes (any stage; the wording is wrong today):**
  - base_idx prose "anchor candle of the zone base pattern" → "first candle of the base pattern": `kl_zones_v1.py:709`,
    `:966`; `KL_ZONES_SPEC.md:31`, `:338`. The GLOSSARY `base_idx` entry was fixed in `37bcf76`.
    `export_m15_chart.py:49` "drawn from the anchor" → "drawn from its base candle" (anchor row #45).
  - `wave_candles.py:180-181` → "the zone's meta anchor_idx (BOS/CTS anchor)".
  - The KL docstring "ev.idx: the level index (where the level is anchored)" at `kl_zones_v1.py:704-713`, `:887` is
    wrong for CTS zones, whose event is CTS_CONFIRMED (idx = moment).
  - "first structural anchor = min event idx" (`structure_lifecycle.py:156`, `poi_zones.py:381-383`) → "earliest event
    idx"; after the flip, "first event moment".
  - `market_structure.py:1547-1548` "clear pullback anchor" is a moment (`last_pullback_pat_apply_idx`).
    `:1312-1314` "range anchored to CTS" is fine in the MS realm (range_start_idx = the CTS anchor).
  - `data_bridge.py:95-107` and `entity_df_mutation.py:629-639`: "parent CTS/BOS extreme" → "parent CTS/BOS anchor".
    Keep "extreme" in the candle-side and "no NEW extreme" phrases.
  - The `cross_cycle_fib.py:40-42` docstring → "CTS-side anchor (CTS_n, or the running extreme past CTS_n when
    pre-established)".
  - `structure_engine.py:449-450`: "to anchor the check window's lower bound at cts_est[0].idx + 1".
  - The false comments at `entity_df_mutation.py:481-482` (E3b) and `reference_zone.py:344-347` (E4).
- **Top-level docs** (`e2e0f89` coordinates; required in the flip commit because the table and headings invert):
  - `GLOSSARY.md:28`, `42`, `54`, `57-58`, `202`, `220`, `224`, `226`;
  - the `ARCHITECTURE.md:97-131` "ev.idx convention" table, plus `:226`, `241-242`, `359`;
  - `PRE_REFACTOR_INVARIANTS.md:77-102`;
  - `WORKFLOWS.md:159`;
  - `GOTCHAS.md`: 51 hits, including the headings `:671` "…Not the CTS Extreme" and `:688` "BOS_CONFIRMED `ev.idx` Is
    the BOS Extreme…" (both still present at HEAD);
  - `LANDMINES.md`: 33 hits (feas: `:493`, `:1929-1953`);
  - `PART4_REFACTOR_SPEC.md`: 78 hits;
  - `MARKET_STRUCTURE_SPEC.md:338`, `423`, `505-508`, `555`;
  - `KL_ZONES_SPEC.md`: 30 hits;
  - `FIB_LIFECYCLE_SPEC.md`: 11 hits;
  - `CROSS_CYCLE_FIB_SPEC.md:194`, `198a`, `439-441` (also `:89-90`, which cites a stale `:1803`);
  - `POI_ZONES_SPEC.md:413`, `453-455`;
  - `WAVE_CANDLES_SPEC.md:55-58`, `237`, `323`;
  - `CHARTING_SPEC.md:368`;
  - `.claude/skills/compare/SKILL.md:244`;
  - MEMORY "Key Architecture Points".
  - Landed plans A/B/C stay as history.
- **E1 docs:** the contract amendment (`LANDMINES.md:32-40` "Event Contract Rules" rule 2, and `ARCHITECTURE.md:65`
  "Event contracts"); the ARCHITECTURE "`ev.idx` convention" and "Anchor has two realms" rows; GLOSSARY `anchor_idx`;
  `WAVE_CANDLES_SPEC.md:103`; `zones/structure_lifecycle.py` HEAD `:73-74` (Plan D docstring) and the ARCHITECTURE
  "Bound and frequency" paragraph (HEAD `:111-112`): `meta["anchor_idx"]` → `meta["pattern_anchor_idx"]`; the GLOSSARY
  "Bare element idx" row gains `RANGE_STARTED.meta["cts_idx"]` (§1.1).
- **Frozen-bridge rows still missing from the ARCHITECTURE table (naming_inv step 0; verified absent at HEAD):**
  - `BOS_CONFIRMED.meta["pb_start"]` = the pullback-path CTS_CONFIRMED moment (None on cycle 0 / proximity-only
    cycles);
  - BOS `source` labels;
  - `RANGE_STARTED` keys: `cts_idx`/`cts_price` = the CTS anchor; `pullback_apply_idx`/`proximity_apply_idx` = the
    confirmation moment; `start_idx`/`confirm_idx` = the offline range candidate / confirm;
  - `structure_levels.csv` `time` (the anchor's time; its meta is a copy of the event meta);
  - the M15 coordinate note (only `confirmed_at`, `apply_idx`, `anchor_idx`, `cts_anchor_idx`, plus the dead
    `confirmed_idx` / `bos_anchor_idx`, are rebased);
  - the event-type-token rule (`_emit_cts_established`, `on_cts_established`, `stop_after_cts_established`, FibTracker
    phases `pre_established`/`established`, legend "CTS (confirmed)" name the event or phase; the standard governs only
    `*_idx`- and `*_time`-valued names).

### 1.10 Stays as named

- **KL-zone anchor family.**
  - KL meta `anchor_idx`: `kl_zones_v1.py:892-899`, `:954`. Readers `orchestrator.py:124-131` (`z_anchor`) and
    `wave_candles.py:403`. Listed in `_ZONE_META_IDX_KEYS` `entity_df_mutation.py:116`. Exported: meta of all 39 KL
    rows.
  - `identify_base_pattern(anchor_idx)` + `anchor_low`/`anchor_high`/`anchor_dir`: `kl_zones_v1.py:28-79`, `159-195`,
    `198-231`, `595-690`, `1103-1126`. Also `identify_inside_bar_pattern` / `identify_1candle_pattern(anchor_idx)`.
  - `identify_wave_candles(anchor_idx, anchor_type)` + the `_bos_*`/`_cts_*` params: `wave_candles.py:4`, `166-243`,
    `277-413`.
  - `WaveCandleResult.meta["anchor_idx"]`: `wave_candles.py:270-273`, `442-445`.
  - `build_ad_hoc_bos0_reference_zone(anchor_idx)` / `_derive_zone_ad_hoc(anchor_idx)`: `reference_zone.py:155-207`,
    `210-251`. Callers `structure_engine.py:174`, `:188`, `unified_probe.py:327`, `entity_df_mutation.py:388`,
    `401-407`, via `_build_first_confluence_ref_zone` `:660`, `:786`.
  - The wave_candles `cts_anchor_idx` locals: `:403-413`, `:449-533`, `:536-571`.
  - `_find_prior_cts_anchor`: `:152-159`, used `:320-321`; imported by `test_wave_candles.py:21`, `:242-247`.
- **Fib anchor family.**
  - `select_fib_anchor_for_cycle` + `anchor_bos_idx/_price`, `anchor_cts_idx/_price`: `fib_tracker.py:100-193`,
    `870-955`; `poi_zones.py:30`, `996-1067`; MS in-flight via `compute_poi_inners_for_cycle`
    (`market_structure.py:72-79`, `219`, `2023-2047`).
  - `FibRetracement.anchor_high/_low(_idx)`: `features/fibonacci.py:27-50`, `59-146`, `177-191`. Set at
    `fib_tracker.py:974-991`, `1020-1039`, `1391-1409`, `2276-2294`, `2346-2363`, `2400-2418` and
    `poi_zones.py:842-856`, `1044-1057`.
  - Cross-cycle `anchor_idx`/`anchor_price`: `cross_cycle_fib.py:61-62`; `fib_tracker.py:2129-2130`, `2266-2267`,
    `2334-2376`, `2382-2398`, `2492-2493`.
  - `_m15_extend_cross_anchor` / `new_anchor_*`: `:2243-2247`, `:2334-2438`.
  - `_running_extreme_anchor`: `:1894-1919`, sole call `:2097`. Both words are literally true.
  - The `fib.anchor_line.active/.historical` style keys: `style_registry.py:156-162`; `export_plotly.py:2516-2617`.
  - The "START ANCHOR" prose: `fib_tracker.py:1876-1878`, `2209-2239`, `2388`. It names the BOS side of the span; an
    optional clarification.
  - The FibState "anchors" prose.
- **CTS anchor at confirmation.**
  - `CTS_CONFIRMED`/`CTS_RECONFIRMED.meta["cts_anchor_idx"]` (`market_structure.py:1869`, `:1910`).
  - Its readers: `kl_zones_v1.py:896`, `reference_zone.py:96`, `unified_probe.py:605`,
    `first_confluence_trigger.py:100`, `export_plotly.py:1088-1089`, `export_m15_chart.py:521`, `:2025`,
    `fib_tracker.py:1638-1642`, `wave_candles.py:152-159`, `uc1_trigger.py:81-82`.
  - Its copies: `unified_probe` `cts0_anchor_idx` (`:604-607`), `probe_fc_finalize` `cts_anchor_h1`/`loh_cts_anchor`
    (`"CTSanch"`/`"LOH(anch)"`, `:178`, `:182`, `:201-219`).
  - Its test fixtures: `test_first_confluence_trigger.py:19-24`, `43`, `68`, `149-151`; `test_cross_cycle_fib.py:41-44`;
    `test_main_versioned_cross.py:38-41`; `test_first_trigger_migration.py:259-272`, `522`, `580`, `609`, `780-812`
    (`:807/:812` pin the shift); `test_wave_candles.py:227`, `244`; `test_unified_probe.py:219`, `976-981`, `1033`,
    `1087-1097`; `test_reversal_resolver.py:286-293`.
- **Bare element idx on price-paired records** (GLOSSARY "Bare element idx" row, landed in `37bcf76`).
  - FibState `bos_idx`/`cts_idx`: `fib_tracker.py:54-57`, written `:1045-1047`, `:1415`, `:2301-2303`, `:2369`,
    `:2428`. Exported as fib_lifecycle.csv cols 13-14 by `export_fib_lifecycle.py:24-25`, `:58-59`. Shifted at
    `entity_df_mutation.py:288-289`. Read at `export_plotly.py:2552-2559`, `2646-2649`.
  - POI meta `bos_idx`/`cts_idx`: `poi_zones.py:584-585` → HEAD `:597-598`.
  - `find_ic_candidates` locals: `poi_zones.py:203-204`, `:211`, `:232-233`.
  - final.csv `cts_idx`/`bos_idx`: cols 53, 56. Written at `market_structure.py:37`, `2405-2407`, `2465`, `2498`,
    with invariants at `2571-2606`. Listed at `unified_probe.py:102-104` and `entity_df_mutation.py:84-86`. Read at
    `identify_start.py:167` and `probe_fc_finalize.py:46`.
  - MS `cycle0_data` keys / FibTracker `_cross_cycle_data['cycle0']` / `compute_poi_inners_for_cycle(bos_idx, cts_idx)`:
    - `market_structure.py:2040-2062` (docs `59-81`, `2025-2026`);
    - `fib_tracker.py:757-765`, `1310-1312`, `1336`, `1356`, `1432-1433`, `1523-1524`, `1794`, `1801`;
    - `poi_zones.py:984-1040`.
  - `CrossEligibility.bos_idx/cts_idx` + `_bos_by_cycle`/`_cts_by_cycle`: `cross_cycle_fib.py:40-52`, `63-64`,
    `126-186`; `fib_tracker.py:272`, `277`, `542`, `1642`, `1177`, `1214`, `1565`, `1943`, `1963`, `1988-2000`,
    `2167-2175`, `2205`, `2499`. Only the docstring is reworded.
  - Fib meta `cycle1_bos_idx`: `fib_tracker.py:938`, `1442`, `1532`. Its producer must stay the anchor after the
    flip.
  - Fib locals `c0_bos_idx`, `c0_cts_idx`, `bos1_idx`, `bos_x_idx`: `:1432-1437`, `1523-1528`, `1568-1570`, `1584`,
    `2272-2285`, `2301`, `2332`.
  - Fib log labels: `:647`, `791`, `914`, `947`, `1101`, `1189-1190`, `1423`, `1518`, `1721`, `1751`, `1757`, `1774`,
    `1795`, `1801`, `1810`, `2332`, `2376`, `2438`.
  - Chart locals `bos_idx`/`cts_idx`, `cts_by_idx`/`bos_by_idx`.
  - `range_start_idx`: document only; it is the CTS anchor on 2 of 3 paths. `market_structure.py:191`, `1193`,
    `1314`, `2108`.
- **Frozen labels.**
  - BOS `source` = `initial_prior_extreme`/`pullback_extreme`. Emitted `market_structure.py:1506`, `:1521`; printed
    `kl_zones_v1.py:770`. Fixtures `test_lifecycle_sweep_predicted_table.py:100-125`, `test_parent_tables.py:58`,
    `test_structure_lifecycle_moment.py:55`. `LANDMINES.md:131`, `:177`. Exported in the events and levels meta.
  - BOS_THRESHOLD_UPDATED reason `rv_anchor_failed` (the close-break candle, pattern realm): `:722`, `:756`; tests
    `test_ms_bounded_equals_truncated.py:186`, `392`, `398`, `test_ms_stop_after_cts.py:362`. Exported on 3 conf rows.
- **Pattern-realm scan-cursor names** (consistent with "anchor" = a pattern's first candle).
  - `_step_anchor(i)`: `market_structure.py:973`, with prose at `253`, `268`, `370`, `463`, `495`, `569`, `641`,
    `1001`, `1016`, `1039`, `1176`, `1191`, `1575`, `1614`, `1624`, `1964`. Monkeypatched by name in
    `test_ms_stop_after_cts.py:71-85`.
  - `_best_bopb_pattern_at_anchor` + `bos_frozen_for_anchor`: `:1045`, `991`, `1113-1123`, `1467`.
  - `_schedule_reversal_from_anchor(anchor_idx)` + the expiry `anchor` + the `anchor=` RV labels: `:894`, `832-862`,
    `880`, `925`, `962`. The function name stays; its idx-valued param `anchor_idx` and the expiry local `anchor` are an
    optional-rename candidate → `pattern_anchor_idx` (Q4.13).
  - `detect_best_for_anchor`: `structure_patterns.py:663`; callers `market_structure.py:905`, `1077`, `1102`, `1123`,
    `pattern_engine.py:56`.
  - `TrueFirstBreakout._anchor_candidates`: `:150`, `:208-209`.
  - `anchor_shift` and the 12 `*_as0..2` final.csv columns: `candles_v2.py:169-250`, `304-316`;
    `candle_classifier.py:71`.
  - `PatternEvent.start_idx`/`end_idx`/`confirmation_idx` + final.csv `pat_*`: `common/types.py:47-52`;
    `pattern_engine.py:45-47`, `72-76`.
  - `rc_by_anchor`: `export_plotly.py:1550`.
  - `anchors_by_sub`: `export_m15_chart.py:431-499`, `1044-1061`. This one is MS realm (a structure start).
- **More "anchor" / "extreme" sites that stay** (listed for the rename sweeps):
  - Plotly `anchor` / `xanchor` / `yanchor`: `export_m15_chart.py:1729`, `1747`, `2618`, `2625`; `export_plotly.py:368`,
    `1358-1453`, `2820`. Exclude them from any bulk rename.
  - English-verb "anchor", to reword or ignore: `poi_zones.py:225-226`; `entity_df_mutation.py:608`;
    `structure_engine.py:449-450`; `tests/test_bounded_structure.py:6`; `tests/test_main_reversal_probe.py:92`.
  - "Structural anchor" = `starting_idx` (MS realm, fine):
    - `unified_probe.py:7-10`, `8-9`, `135-137`, `358`; `structure_engine.py:864`; `multitf/types.py:59`, `71`;
      `sid_records.py:76`; `sub_structure_pool.py:139`;
    - `export_m15_chart.py:18`, `23`, `267-268`, `286`, `802`, `1110` (`:49` is a base-candle mislabel, reworded
      in §1.9);
    - `GLOSSARY.md:186`, `225`;
    - tests `test_unified_probe.py:15`, `20`, `858`; `test_sid_records.py:258`; `test_reversal_resolver.py:255`,
      `267`; `test_render_sub_projection.py:39`, `164`, `470`, `613`, `745`; `test_m15_chart_ownership.py:6`,
      `100-110`, `170`, `248-259` (4 test names), `437-500` (`anchors_by_sub`).
  - Readers of the `anchor_shift` family: `debug/candle_size_distribution.py:59`; tests `test_candles_v2.py:127-200`,
    `300`, `314`; `structure_patterns.py` `_ROW_FIELDS`; `kl_zones_v1.py:185`.
  - Fib-family tests: `test_cross_cycle_fib_routine.py:47-223`, `test_cross_cycle_fib.py:158`,
    `test_main_versioned_cross.py:85`, `244`.
  - FibState "anchors" prose: `fib_tracker.py:53`, `63`, `69-70`, `202`, `396`, `500`, `712`, `973`, `1017`, `1112`,
    `1164`, `1375`; `entity_df_mutation.py:271`.
  - `_running_extreme_anchor` prose: `fib_tracker.py:2072`, `2391`; `CROSS_CYCLE_FIB_SPEC.md:38`, `89-90`, `136-137`,
    `310`; `POI_ZONES_SPEC.md:223`, `230`.
  - FibRetracement construction sites (extreme_final ranges): `fib_tracker.py:879-1039`, `1391-1410`, `2276-2419`;
    `poi_zones.py:1027-1067`. naming_inv gives the callers as `fib_tracker.py:977-2409` and `poi_zones.py:1045-1049`.
  - `compute_poi_inners_for_cycle` params: `poi_zones.py:986-988`.
  - MS cycle0_data: def `market_structure.py:228`, `:2013`, `:2050-2058`.
  - Chart locals: `cts_by_idx` / `bos_by_idx` / `sid_confirmed` / `last_confirmed` (`export_plotly.py:1084-1085`,
    `1157-1219`, `1331-1476`, `2551-2553`; `export_m15_chart.py:578-580`, `1197-1225`). Legend / hover labels "CTS
    (confirmed)" etc. name event types: they stay, and changing them changes the figure.
  - `base_idx` readers in the charts: `export_plotly.py:1656`, `1729`; `export_m15_chart.py:1278`, `1352`, `2220`,
    `2290`.
  - Comments about `enforce_cts0` / the pattern-start candle (pattern realm, fine; `:343` describes the removed
    gate): `market_structure.py:343`, `359-360`, `1469-1470`.
- **Role and other names.**
  - `probe_end_idx`: `unified_probe.py:375`, `403`, `538`, `696`, `768`; `types.py:154-157`, `169`;
    `first_confluence_trigger.py:123`; `first_confluence_pipeline.py:54`; `entity_df_mutation.py:511`, `539-590`,
    `619`, `668`, `810`, `930-978`; `GLOSSARY.md:224`.
  - `FirstConfluenceTrigger.input_idx`.
  - KL `source_event_idx`.
  - `enforce_cts0_new_extreme`: optional rename to `cts0_scan_mode`, about 23 sites (Q4.13).
  - `starting_idx`.
  - Main `SidRecord.creation_event_idx`: re-source at E4 (§2.2 B10).
  - `activated_at`, `revert_idx`, `reactivated_at`, `deactivated_at`: names kept; values become moments (E3a).
    `current_candle` left this list with Plan F: the moment is now its own parameter, `evaluated_at`, and E3a keeps ONE
    moment parameter, so `current_candle` is renamed to its fill-horizon role or folded into `evaluated_at` (§2.4
    item 2).
  - `knowable_at_idx`.
  - The test helper `_cts0_established_idx` (`test_main_reversal_probe.py:109-118`): correct under the final
    vocabulary; its value becomes correct at E4.
  - `probe_fc_finalize` `cts_est`/`cts_est_h1`/`loh_cts_est` (`"CTSest"`/`"LOH(est)"`, `:75-80`, `:177`, `:183`,
    `:201`): the moment after the flip.
  - Plotly `xanchor`/`yanchor`.
  - Event-type-derived identifiers.

### 1.11 Where "extreme" survives (after E5)

1. **Window / retrace searches**
   - `_select_extreme_retrace_candidate` (`unified_probe.py:197-218`; callers `:402`, `:640`; tests
     `test_unified_probe.py:364-392`);
   - `_window_extreme_idx(..., extreme_dir)` (`entity_df_mutation.py:680-699`, sole caller `:784`);
   - `_input_idx_window_extreme` / `_input_idx_window_extreme_toward_cts` (`subsequent_confluence_trigger.py:40-62`,
     call `:129`; `subsequent_counter_trigger.py:37-60`, call `:130`). These are exported through `probe_input_idx`
     in `unresolved_triggers.csv`;
   - the identify_start scenario-1 lookback "extremes" and `meta` `hi_idx`/`lo_idx` (`identify_start.py:48-58`,
     `85-93`, `118-121`);
   - the `_replacement_break_point` counter-move search;
   - `_find_prospective_bos` ("running-extreme pullback candle").
2. **Running extremes**: `_running_extreme_anchor`, and the CROSS_CYCLE_FIB / POI_ZONES_SPEC "running extreme".
3. **"New extreme" checks**:
   - `_is_strict_new_extreme` (`true_first_breakout.py:125`);
   - `enforce_cts0_new_extreme`;
   - the fib "CTS moved to a new extreme" guard;
   - "no NEW extreme can form";
   - the raw-path "new wick extreme".
4. **Price-mapping**:
   - `map_candle_to_lower_tf(parent_extreme_dir)` (`data_bridge.py:92-123`; callers `entity_df_mutation.py:641`,
     `:649`; `LANDMINES.md:788`);
   - `_find_m15_by_extreme` (`export_m15_chart.py:2692-2712`, doc `:2620-2627`, call `:2730`). It duplicates
     `map_candle_to_lower_tf`;
   - "price-extreme mapper" (GLOSSARY LOH, `parent_tables.py:25`, `ARCHITECTURE.md:359`).
5. **Candle anatomy**: `c1_close_near_c0_extreme`; the KL base candle's outer/body extremes; the PB dot at the
   low/high.
6. **Pattern realm**: `pattern_extreme_idx/_price`, and `continuous()`'s own 3-candle `extreme`.
7. **Frozen labels**: `initial_prior_extreme` / `pullback_extreme`.
8. **Test names meaning raw extremes (keep)**:
   - `test_candles_v2.py:258`;
   - `test_first_trigger_migration.py:163`, `171`, `184`;
   - `test_kl_zone_thresholds.py:79`, `125`;
   - `test_m15_chart_ownership.py:468`, `494` (`:468` "replacing_anchor_not_the_literal_extreme" already uses the
     final vocabulary);
   - `test_true_first_breakout.py:217`.

---

## 2. The flip (E2 → E3 → E4)

### 2.1 Mechanics

- **CTS_ESTABLISHED.** There is one emit site (`market_structure.py:1485`).
  - **E2:** pass `cts_anchor_idx=cts_idx` into meta.
  - **E4:** pass `int(apply_idx)` as `idx`, keep `confirmed_at` (assert `== idx`), and keep `pattern_anchor_idx` and
    `cts_anchor_idx`.
- **BOS_CONFIRMED.** There are two emit sites (`:1502`, `:1517`).
  - **E2:** `_emit_bos_confirmed(idx, price, bos_anchor_idx=…)` writes the meta key and builds **`st.bos` =
    Point(bos_anchor_idx, price)** and `bos_threshold` from that param. Today both are built from the same `idx`
    argument (`:1927-1938`, set `:1937`); this is the "MS state decoupling". CTS state is already separate
    (`st.cts = Point(cts_idx)`, `:1554`).
  - **E4:** pass `apply_idx` as `idx`.
- **MS state STAYS on the anchor.**
  - These must not change: `st.cts`; `_initial_bos_before_first_cts(cts_idx)` (`:1502`, window `[start, cts_idx)`);
    `range_start_idx`; the sd-prox gate `i > st.cts.idx` (`:1960-1987`, `:1981`); the POI-inner resolver + cycle0_data
    (`:2003-2014`, `:2040-2062`).
  - Redefining the `cts_idx` local as the moment breaks BOS_0 selection.
- **Unaffected in MS**: the Plan A assert (`:498-501`), the Plan B CTS_EST count (`:543`), `_rewind_to` (it replays
  and reads no idx), and RANGE_STARTED meta / `range_start_idx` / the final.csv `cts_idx` / `bos_idx` columns (these
  read state).
- **Key order.** After E1, CTS_ESTABLISHED carries `pattern_anchor_idx` (the pattern's first candle) and, from E2,
  `cts_anchor_idx` (the CTS endpoint). The feasibility study rejected the spelling `cts_anchor_idx` only because of
  the bare `anchor_idx` key, and **E1 before E2 removes that collision.**
- **Accessors** (feas' `structure/event_fields.py`, translated):
  - `cts_anchor_idx(ev)`: CTS_ESTABLISHED and CTS_CONFIRMED/RECONFIRMED → `meta["cts_anchor_idx"]`; CTS_UPDATED →
    `ev.idx` (unless it joins the flip, Q4.1);
  - `bos_anchor_idx(ev)` → `meta["bos_anchor_idx"]`;
  - `pattern_anchor_idx(ev)`;
  - a moment accessor (name open, Q4.16). Plan F landed one for the three CTS types, `market_structure.event_moment(ev)`,
    defined next to the emitter: CTS_ESTABLISHED → `meta["confirmed_at"]`; CTS_UPDATED → `ev.idx` on the raw path
    (`meta["via"] == CTS_UPDATED_RAW_VIA`), `None` on the pattern path; CTS_THRESHOLD_UPDATED → `ev.idx` (the
    processing candle); any other type raises `ValueError`. Extend it or move it into the accessor module; do not add
    a second.
  - Rules:
    - direct indexing only, never `.get(key, ev.idx)`;
    - one fixture factory `make_cts_established(anchor, moment, …)` (+ a BOS equivalent);
    - a grep gate: no production `.idx` read of CTS_ESTABLISHED / BOS_CONFIRMED outside the accessor module.
  - Pattern-path CTS_UPDATED stays an anchor-indexed event with no recorded moment, so readers of mixed CTS lists need
    the accessor even after E4.

### 2.2 LOCATION reads that must migrate to meta `cts_anchor_idx` / `bos_anchor_idx` FIRST (E2)

**CTS_ESTABLISHED / CTS (≈10 production sites in 9 modules):**

| # | Site | What | Pool key / probe? | On the window |
|---|---|---|---|---|
| L1 | `reference_zone.py:88-97`: the EST/UPD branch of `_extreme_idx_for_cts_event`, feeding the ad-hoc CTS zone (`:378`) and `source_event_idx` (`:374`, `:395`) | The probe input and the ad-hoc base candle. Consumers:<br>- `structure_engine.py:278-296`: the H1 main reversal sets the next H1 sid's start (window: ref=cts_confirmed, input 652 → start 689);<br>- `entity_df_mutation.py:774-798`, `:822`: sibling, first_counter / subsequent_*;<br>- `:903-926`, `:974`: the reversal handoff (degeneracy test `:914`; cache key `input_abs`) | **YES**:<br>- `starting_idx` = the pool key;<br>- `sub_id` numbering;<br>- the probe-cache key (dir, input) and hit/miss (3 hits depend on input equality) | **SILENT** (no probe used an EST reference). It needs a synthetic lagging fixture with an EST winner and lag > 0 |
| L2 | `fib_tracker.py:521` `cts_idx = int(event.idx)` (`:518-529`, incl. the price fallback `df.h/l[cts_idx]`). It fans out to:<br>- FibState / FibRetracement (`995-1103`);<br>- the `_m15_cross_check` anchor (`609-618`);<br>- the `has_unfilled_imbalance` range end (`531-539`);<br>- the c0 snapshot (`757-765`), lock-step with MS cycle0_data (`market_structure.py:72-81`);<br>- the new-extreme guards (`1310`, `1383-1385`, `1509-1511`, `2342-2343`, `2396`);<br>- Scenario 1 (`767-771`) | fib 100% point, IC scan range, imbalance ranges | no | Unmigrated, fib `cts_idx` would go 1223→1224 and 2828→2829 (fib_lifecycle `cts_idx`, the POI IC range `poi_zones.py:203-244`, H1 fib lines) |
| L3 | `poi_zones.py:911-946` `cts_idx_at_t` in cond1 (HEAD `:935-970`) + the pre-window apply `:862-878` + `cts_events_by_key` sort `:403-416` | the location half of cond1 | no | At HEAD a lagging EST is pre-window (anchor < moment <= first_active; strict `ev_idx < first_active`, HEAD `:893-894`); a lag-0 EST with first_active == moment is an in-window transition. After E4 the lagging ESTs with first_active == moment (conf IC 678 @1224; IC 2808 @2829, both lenses) move to an in-window transition at first_active. Transitions at one idx are applied atomically before evaluation (HEAD `:945-962`), so the result is equivalent once cond1 reads meta `cts_anchor_idx`. The E4 `/compare` must confirm `activation_history` is unchanged on these 3 rows |
| L4 | `wave_candles.py:461-514` (feas: `:477-514`) `_cts_bib_last_breakout`:<br>- the sort at `:481`/`:482`;<br>- the test at `ev_idx`;<br>- `range(pattern_anchor, ev_idx)` at `:495`;<br>- `range(ev_idx+1, next)`;<br>- `:117-124` | the CTS BIB scan feeding WVMI | no | Live only for CTS zones with "base inside bar" (H1 has 3; the H1 EST lag is 0). Sub impact unmeasured |
| L5 | `market_structure.py:2643-2677` `structure_levels` `time` | re-source | no | 0 if re-sourced |
| L6 | `export_m15_chart.py:550-580` (`:550`, `:566-580`): the sub trailing-CTS dot / vertex. It feeds the recent-vs-prior spans and, one hop on, `_replacement_break_point` (`:430-495`, `:620-638`; `lo` = last point + 1 at `:481`) | x would move to the moment while y stays `ev.price` | no | 0 visible (a lagging EST is never the trailing CTS) |
| L7 | `export_m15_chart.py:2048-2069` (`:2048`, `:2061-2069`): the H1-overlay trailing CTS | same | no | 0 |
| L8 | `export_plotly.py:1146-1186` (`:1146-1147`, `:1171-1186`): the H1 trailing CTS marker | same | no | 0 |
| L9 | `orchestrator.py:279-288`: the prev-BOS line END value (`:286` already compares an anchor against a moment) | the line end | no | 0 (H1 EST(1,0) 703 == `confirmed_at`) |
| L10 | `unified_probe.py:603-607`: the dead fallback `meta.get('cts_anchor_idx', first_cts.idx)` (→ direct index), plus the `:555-560` `cts1_ext` label; `probe_fc_finalize.py:150-161` | log / fallback | no | — |

Test-only paths: `structure_engine.py:449-484`, `:563-578`, `:767-782` (Q4.10).

**BOS_CONFIRMED (≈15 sites; a BOS flip without this migration is catastrophic):**

| # | Site | What | Pool key? | Unmigrated effect |
|---|---|---|---|---|
| B1 | MS `_emit_bos_confirmed` (`:1927-1938`) → `st.bos_confirmed`. Consumers: the BOS-inner resolver (`:1538-1545`), the POI / fib leg (`:2003-2014`), cycle0_data (`:2040-2062`), the df `bos_idx` (`:2498`) | MS state | no | final.csv `bos_idx`, the POI inners and cycle0_data shift |
| B2 | KL BOS branch `kl_zones_v1.py:886-900`, `:935`, `:951-966`: `source_event_idx = int(ev.idx)` → `anchor_idx` (`:893-896`) → `identify_base_pattern` (`:899`) → meta `anchor_idx` (`:954`). `confirmed_idx` (`:889`) already reads `confirmed_at`. KL iterates in emission order, not sorted | every BOS zone's geometry | no | 34 BOS zones (5 H1 + 21 conf + 8 counter) → wave candles → WVMI |
| B3 | BOS wave candles on KL meta `anchor_idx` (`wave_candles.py:137-148`, `211-370`, fed by `orchestrator.py:120-137`) | one hop | no | follows B2 |
| B4 | orchestrator `bos_by_cycle` (`:208-227`) → the fib BOS point, the own / prior-cycle imbalance ranges (`fib_tracker.py:541-542`, `757-765`, `1177-1182`, `1214-1226`, `1565-1573`, `1963-2000`, `2166-2220`, `2499-2516`) and `cycle1_bos_idx` (`:938`; cond2 / cond3 at `:1442-1454`, `:1532-1543`) | fib | no | **Every H1 fib collapses** (BOS moment == EST idx on all 5 H1 cycles, so `bos_idx == cts_idx` and `range(bos, cts+1)` spans one candle); the POIs go with it |
| B5 | `first_confluence_trigger.py:88` `input_idx = int(ev.idx)` → `:122` meta `probe_input_idx` → `first_confluence_pipeline.py:34-53` → `entity_df_mutation.py:618-675`: price-mapped, it becomes the probe input and the ad-hoc BOS_0 base. Also `orchestrator.py:849`, `:857`; `probe_fc_finalize.py:175`, `:208` | **the FC POOL KEY** | **YES**: subs 0/2/4, keys (1,385), (1,2365) via the cache, (-1,3304); `validated_parent_idx` 96/591/826; unresolved `probe_input_idx` 689/728 | The inputs go 96→115, 591→652, 689→703, 728→748, 826→902. Test pin: `test_first_confluence_trigger.py:51` (`t.input_idx == 20  # BOS extreme`) |
| B6 | the prev-BOS line START `last_bos_by_sid` (`orchestrator.py:266-292`, mirrored at `entity_df_mutation.py:338-339`) | H1 chart | no | H1 sid 1 line 591→652 |
| B7 | Every BOS dot and PB→BOS endpoint:<br>- H1: `export_plotly.py:1069-1096`, `1130-1143`, `1192-1223`;<br>- M15 sub: `export_m15_chart.py:513`, `536-547`, `584-612`, `670-672`, `1183-1187`;<br>- H1 overlay: `2021-2045`, `2104-2134`, `302-319` (`_wave_touches_window` / `_split_polyline_by_wave`), `2073-2097`, `2138-2142` | charts | no | Dots move 2–411 candles; drawn / hidden lifecycle decisions can change; `_prior_line_segments` (recent-vs-prior spans) inherits the dot shift |
| B8 | `structure_levels` 5 BOS rows | = L5 | no | 5 time cells |
| B9 | `compute_struct_start_by_sid` `min(ev.idx)` (`structure_lifecycle.py:174-182`, HEAD `:177-185`; rule prose `:156`). Consumers:<br>- `kl_zones_v1.py:1033/1048`;<br>- `poi_zones.py:392/531/821`;<br>- `fib_tracker.py:422/441-443`;<br>- `parent_tables.py:99/144-146`;<br>- `export_m15_chart.py:2112-2122/2366-2378`;<br>- `export_plotly.py:1931-1950` | the lifecycle-start base (TIMING computed from a LOCATION) | no | The min becomes the BOS moment: H1 sid0 96→115; subs 454→458, 1797→1816, 2365→2368, 2639→2649, 3304→3306, 3760→3786, 4027→4031, 3621→3656. Masked in most consumers. **A decision (Q4.5)**. Pins: `test_structure_lifecycle_moment.py:179-184`, `test_parent_tables.py:116` |
| B10 | main `SidRecord.creation_event_idx` = `min ev.idx` (`sid_records.py:20`, `:33-41`; called `orchestrator.py:507-508`) | no reader | no | 96→115; no CSV / chart effect; tests `test_sid_records.py:31-104` |
| B11 | the `sub_wvmi` sort (`sub_wvmi.py:68`, `82-84`) | ordering only | no | see §2.5 |
| B12 | the KL debug tuples (`kl_zones_v1.py:752-772`) | stdout | no | print idx + meta anchor; drop the `bos_prev` read |

### 2.3 TIMING reads that become CORRECT (E3; each is its own `/compare`)

| Stage | Site | Predicted delta (window) |
|---|---|---|
| done | **POI activation gate (Plan D)** | landed: 1 row + 3 meta cells |
| **E3a** | **FibTracker timing.**<br>- `activated_at` writes: `fib_tracker.py:602`, `664`, `731`, `782`, `846`, `865`, `937`, `956`, `1200`, `1340`, `1360`, `1588` (= `cts_idx`); `:2313`, `:2520` (= `current_candle`). `current_candle` = CTS_ESTABLISHED.idx from `:614`/`:1968`/`:1992` and CTS_UPDATED.idx from `:1223`. It is a real moment on the CTS_THRESHOLD_UPDATED path (`:2090`) and on the RAW CTS_UPDATED path (`ev.idx` = the processing candle), and the CTS anchor on CTS_ESTABLISHED and on pattern-path CTS_UPDATED (corrected by Plan F; `event_moment`).<br>- Consumers: `:1056` → the new_cycle terminal `:1071`/`:1086`; `_mark_first_active` `:1099`, `:2324`/`:2329`.<br>- The fill-check as-of: `:537`, `:1184`, `:1320`, `:1463`. (`:1446` struck: it sits in `_update_fib_cts`'s dead cross branch, which is never reached; its deletion is a Plan F §7 hygiene item.) Plan F already moved the knowability cut of these reads to the moment (`evaluated_at`); their fill horizon `check_to_idx` is unchanged (`cts_idx`: the EST anchor at `:537`, the update's `idx` at the others), so only that half is E3a's. `:1320` is the cycle-0 cache write, which Plan F keeps uncut on purpose (a cached value is judged at its use).<br>- The set-if-absent merge `:427-429`: the moment-based `cycle_life` end is only set-if-absent, so the anchor-based fib terminal wins today.<br>- Cross `current_candle` (§2.4).<br>- `scenario1_revert` / `revert_idx` (`:823-826`, `:1837-1858`).<br>- `reactivated_at`/`deactivated_at` (`:1469`, `:1473`, `:1550`, `:1553`) are CTS_UPDATED values (already the moment on the raw path, the anchor on the pattern path) and stay until Q4.1 | **4 fib_lifecycle rows**, on M15_confluence lines 14, 15 and M15_counter lines 2, 3:<br>- sub 3 cyc 0: `end_idx` 2828.0→2829.0 (new_cycle; KL/POI already end at 2829);<br>- sub 3 cyc 1: meta `activated_at` 239→240 (slice-local);<br>- H1: 0; POIs: 0.<br>Re-measure. (Plan F left these cells and line numbers in place: on the same sub 3 cyc 0 rows it re-valued meta `activated_at` 61→62, and it removed counter line 7.) The spec conflicts: FIB_LIFECYCLE_SPEC §7 cases 2-3 ("zone-consistent"), and the wording at §15.3 / `:888` |
| **E3b** | `knowable_at_idx` (`sub_structure_pool.py:84-100`; `pooled_structure_build.py:34-43`, `71-72`; ← `render_sub_projection` `entity_df_mutation.py:1202`): special-case CTS_ESTABLISHED on `confirmed_at` (BOS already is). The sibling clip + recency (`entity_df_mutation.py:478-489`) and `reference_zone.py:335-338` (window) / `:344-357` (recency) move to the moment. Fix the comment at `:481`. This is the documented known limit at `LANDMINES.md:683-690` / PART4 §17.8 / §17.12. Plan F §7: `knowable_at_idx` and `unified_probe._second_cts_moment` converge onto `market_structure.event_moment` here (one moment concept) | **0** (caps 1940 / 3611). It closes §17.12 for EST. In general it can move pool keys |
| **E3c** | unified_probe Phase-2 `check_lo = int(first_cts.idx)+1` (`:602`; window `:629-642`; docstring `:489-490`) → moment + 1, aligning with Phase 1 (`:403`). The label `cts0_est=` (`:671`) prints `first_cts.idx` (the anchor) and is not touched by the `check_lo` change. Re-source it in this commit to the existing moment local `cts0_est_idx` (`:594`), following naming_inv's option, so from E3c on it prints the moment under its moment name. It prints only on a Phase-2 reset, and the window's one Phase-2 run has no reset (`p2_iter=1`, lag 0), so run.log is unchanged on the window. (The 8 run.log `cts0_est=` lines on the window are all Phase-1 resets from `:424`, which already prints the moment `tfb.est_idx`.) | **0** predicted (verify). In general it can shift `starting_idx` |
| **E3d** | the prev-BOS line FILTER (`orchestrator.py:279-288`) on the moment | **0**. Where the line ENDS is Q4.6 |
| E3? | the event processing order (Q4.3) · Scenario 1 (Q4.2) · the struct_start base (Q4.5) · the `cycle1_bos_idx` cond3 as-of (Q4.8) · the test-only Exception-2 windows (Q4.10) | 0 (H1 lag 0) / unmeasured |
| **E3e (proposed; not in the user's decided list — confirm)** | **The MS in-flight POI-inner resolver evaluates fills as-of the CTS anchor.** Path: `_refresh_poi_inners_for_cycle` → `compute_poi_inners_for_cycle` / `select_fib_anchor_for_cycle` (`market_structure.py` ~`1989-2062`; `fib_tracker.py:100-127`); `_update_cycle0_data` likewise uses the anchor as the imbalance as-of (`market_structure.py:2050`; resolver `:2003-2014`). It feeds the sd-zone-proximity CTS confirmation, so it is engine-side (MS events can move). The audit's same-class sweep classified it "unclear, not measured". Plan F settled the knowability half: both MS reads stay UNCUT by decision (explicit `evaluated_at=None`; the snapshot's only reader is gated `i > st.cts.idx`, so no decision uses a gap before it forms), so E3e concerns only the fill horizon | **not measured** — measure before the plan predicts. Its own `/compare`. MS ↔ FibTracker parity must hold (§2.4 item 9; since Plan F it has one documented exception, the M1 activation divergence) |
| E3? (behaviour mixing, naming_inv step 7) | `find_ic_candidates` `check_to_idx = cts_idx` (`poi_zones.py:232-233`; its knowability is settled by Plan F, §2.4 item 8); the charts use `next_bos_idx` (an anchor) as an exclusive TIME bound on pullback STATE_CHANGED events | unmeasured |
| pre-E2 or E5 (delete; §1.8) | the `lifecycle_end_idx` chain (unread) | 0 |
| automatic at E4 | `probe_fc_finalize` `cts_est` LOH (`:79-80`, `:177`, `:183`) | console |

Already correct (reference implementations):
- `compute_cycle_lifecycle` / `build_parent_tables` (`structure_lifecycle.py:96-114`, `parent_tables.py:107-141`);
- zone_proximity (`:199-206`, `306-311`, `346-353`, `374-407`);
- `first_confluence_trigger.py:89` `trigger_event_idx` (LOH-mapped to `trigger_idx` at `orchestrator.py:854`);
- unified_probe `_second_cts_moment` / `cts0_est_idx` (`:450-456`, `:565`, `:594`).

These carry `.get('confirmed_at', ev.idx)` fallbacks, which become direct indexing.

### 2.4 Dual-role calls to split in E2 (both halves fed today's value; E3 switches the time half)

1. **`FibTracker.on_cts_established` local `cts_idx`** (`:521`).
   - Anchor uses: FibState, the cross anchor, the c0 snapshot, the IC / imbalance range end.
   - Time uses: the fill as-of (`:537`), `activated_at`, `current_candle`, Scenario 1, the revert terminal (`:825`).
2. **`_m15_cross_check` / `resolve_cross_cycle_eligibility(current_candle=cts_idx, anchor_idx=cts_idx)`**
   (`fib_tracker.py:609-618`, `1218-1227`, `1966-2004`, `2122-2258`; `cross_cycle_fib.py:117-159`).
   - One source today (`:614-615`, `:1968`, `:1993`).
   - Live on the window: sub 3 cyc 1 `cross_failed` (`activated_at` 2828); confluence sub 0 cyc 2's EST at 1223
     extends the pre-established cross that started at 1169.
   - The split (revised by Plan F): Plan F added the moment as its own keyword-only, required parameter
     `evaluated_at` (FibTracker passes `self._evaluated_at`, the handled event's `event_moment`; the MS in-flight
     resolver passes `None`), which drives the knowability cut in all three of the routine's `has_unfilled_imbalance`
     calls. `current_candle` still carries today's value in both of its roles (the own-test window end and the fill
     horizon): a moment on CTS_THRESHOLD_UPDATED and raw CTS_UPDATED, the CTS anchor on CTS_ESTABLISHED and
     pattern-path CTS_UPDATED. **E3a keeps ONE moment parameter, `evaluated_at`:** `current_candle` does not become a
     second moment; it is renamed to its fill-horizon role or folded into `evaluated_at`. `anchor_idx` stays (fib
     family, anchor).
3. **`has_unfilled_imbalance(df, min(bos,cts), max(bos,cts), cts_idx)` at EST** (`fib_tracker.py:531-539`): the range
   is anchors; the as-of is a time. **Plan F (landed 2026-09-24; items 3-6 and 8-9 were re-read against it):** the
   as-of is now two values. The knowability cut asks at the moment (`evaluated_at` = `confirmed_at`, through
   `FibTracker._has_unfilled`); the fill horizon `check_to_idx` is still `cts_idx` (the anchor), and only that half is
   left for E3a. (zone_proximity, named in the pre-landing note, calls no imbalance primitive: it reads POI activity
   through `poi_active_as_of` and moves with the POI rows.)
4. **`_update_fib_cts` / create-on-fail** (`:1181-1187`, `1443-1474`, `1535-1574`): the range, the as-of, and
   `reactivated_at` / `deactivated_at`. Plan F: every read here goes through `_has_unfilled` at the event's moment
   (raw CTS_UPDATED → its `idx`; pattern path → no cut, `event_moment` is `None`), so the deactivate reason
   `all_imbalances_filled` now means "no FORMED unfilled imbalance". The fill horizon is the update's `idx`: already
   the moment on the raw path.
5. **The c0 snapshot** `c0['cts_idx']` (`:177-178`, `757-765`, `1308-1321`, `1431-1439`, `1522-1530`): the range, and
   cond2's "@CTS_0" as-of (`select_fib_anchor_for_cycle`'s labels; `_update_cycle1_main` calls this read cond1).
   Plan F: the cycle-0 liveness cache `c0["has_unfilled"]` (cond2) is stored UNCUT (`evaluated_at=None`). It is
   judged at its later use (CTS_1 ESTABLISHED, after CTS_0, when every gap in `[BOS_0, CTS_0]` has formed), which keeps
   it equal to the MS mirror. A decision taken at the write event asks again at that event's moment
   (`FibTracker._c0_has_unfilled_now`). The range and the fill horizon are unchanged.
6. **`cycle1_bos_idx`** (`:938`, `1442-1454`, `1532-1543`): the range start, and cond3's as-of. Plan F: these reads
   pass the moment too. cond3's window ends at CTS_0, before any moment, so the cut cannot change it (only its fill
   horizon is open, Q4.8). The cycle-1 own window `[BOS_1, CTS]` ends at the update and is cut at its moment (no cut on
   the pattern path), in lock-step with the create-on-fail read.
7. **POI transitions** (`poi_zones.py:911-946` → HEAD `:935-970`): the time (transition) and the location (cond1).
   The same for the pre-window split + sort (`:403-416`, `:829`, `:862-878`).
8. **`find_ic_candidates`** `range(fib.bos_idx, fib.cts_idx+1)` + `has_unfilled_imbalance(end=cts_idx,
   check_to_idx=cts_idx)` (`poi_zones.py:203-244`, called `:456`, `:1074`; `select_ic_variants` `247-317`). Plan F:
   both callers pass `evaluated_at=None` explicitly. `derive_poi_zones` identifies ICs retrospectively on the final fib
   (measured 0 cells even if cut), and the POI sweep now decides WHEN a POI can go live (it enters an imbalance at
   `max(inst.formed_at, first_active)`); the MS in-flight caller is item 9. The fill horizon `check_to_idx=cts_idx`
   (the anchor) is unchanged and is still this item's question.
9. **The MS POI resolver / cycle0_data** (`market_structure.py:2003-2014`, `2040-2062`). This is MS state and is
   unaffected by the emit flip, but it must keep parity with FibTracker (`fib_tracker.py:521`, `:1131`) in the same
   change. Plan F: both MS reads pass `evaluated_at=None` explicitly (no knowability cut, by decision: the snapshot's
   only reader is gated `i > st.cts.idx`). Parity with FibTracker now holds for cond2 and cond3 only. cond1 and the
   activation can diverge (the accepted M1 divergence: 0 cases on the window; LANDMINES "Scenario 2 anchor
   agreement"; IMBALANCE_FILL_SEMANTICS "Decided at the event — and the accepted divergence").
10. **The sibling clip** (`entity_df_mutation.py:478-489`): `abs_idx` is the clip time and the winner's anchor is the
    probe input. The highest-risk site: one expression mixes both roles, and it sets the pool key.
11. **`_resolve_reversal_start`** (`entity_df_mutation.py:903-926`, `:974`): the recency pick is a time; the probe
    input, the degeneracy test and the cache key are location.
12. **`build_reference_zone_from_cts_event`**: the `idx_window` filter + recency sort are time; the zone base is
    location.
13. **The prev-BOS end** (`orchestrator.py:279-288`): the filter is a time; the end value is a location.
14. **The charts' `next_bos_idx`**: the PB-search bound is a time; the endpoint is a location (`export_plotly.py:1192-1223`;
    `export_m15_chart.py:584-612`, `2073-2097`). The H1-overlay `_wave_touches_window` intersects anchor-located wave
    spans with a timing window (`export_m15_chart.py:302-319`, `2021-2045`).

### 2.5 Sort keys to pin (E2) + ordering hazards

**Pin to the anchor accessor so that the flip reorders nothing:**
- `orchestrator.py:146` `(e.idx, e.type)` feeds:
  - the fib loop `203-236`;
  - the prev-BOS lines `267-288`;
  - zone proximity;
  - the WVMI loops `357-368`;
  - the trigger detectors;
  - `build_parent_tables`.
- `sub_wvmi.py:68`;
- `poi_zones.py:416` (HEAD `:417`);
- `wave_candles.py:481`/`:482`;
- `reference_zone.py:353`.

`unified_probe.py:274` and `structure_engine.py:427-432` are monotone and safe.

**Hazards** (0 instances on the window, which is why they need unit tests):

| # | Hazard |
|---|---|
| H1 | The fib loop needs BOS(S,C) before EST(S,C) (`orchestrator.py:208-213`). After the joint flip the two tie at the moment. Today that order holds **only by alphabet** ("BOS_CONFIRMED" < "CTS_ESTABLISHED"), and **MS emits EST before BOS** (baseline row order), so a stable moment-only sort skips the fibs. A BOS-only flip skips the fibs on lagging cycles |
| H2 | An sd-prox CTS_CONFIRMED can land ON the EST moment (gate `i > st.cts.idx`, `market_structure.py:1981`), and "CTS_CONFIRMED" < "CTS_ESTABLISHED". So `on_cts_confirmed` would run before `on_cts_established` and the fib would never lock. The smallest gap on the window is 2 |
| H3 | A prior-cycle `CTS_THRESHOLD_UPDATED` with idx in [EST anchor, moment) would be dispatched while `_m15_phase == 'pre_established'` (`fib_tracker.py:2081-2084`), adding extra cross-fib checks. There are none near 1223 / 2828 |
| H4 | Mixed-scale compares, if only one side is migrated: the fib new-extreme guards (`fib_tracker.py:1310`, `1383`, `1509`, `2342`, `2396`) would silently drop an update in (anchor, moment]. POI cond1 has the same shape |
| H5 | The reference-zone tie: EST at its moment can tie an sd-prox CTS_CONFIRMED. The CONFIRMED > ESTABLISHED tie-break then decides (acceptable), but the comment at `reference_zone.py:344-347` must change |

Fix options (Q4.3): pin to the anchor accessor permanently, or move to an explicit type rank (BOS < EST < UPD <
THRESHOLD < CONFIRMED) in moment order.

### 2.6 Unaffected reads

- **KL**: the CTS_ESTABLISHED branch (`kl_zones_v1.py:874-885`) reads no idx; CTS zones use
  `CTS_CONFIRMED.cts_anchor_idx`.
- **compute_cycle_lifecycle / build_parent_tables / zone_proximity / first_confluence `trigger_event_idx`**: already
  moment-based.
- **The MS gates / state**, as listed in §2.1.
- **The entity_df_mutation shift** (`:103-124`, `222-225`, `486-488`) is unaffected in itself; see risk R5.
- **The CTS_ESTABLISHED ordering sorts** at `unified_probe.py:274` / `structure_engine.py:427-432`.
- **`wvmi.py` and `poi_lifecycle.py`**: no idx reads of these types.
- **`probe_fc_finalize.py:153-154`, `162`, `193-194`** (the Plan A leak check `max_ev` / `leak`): still valid after the
  flip; its max may grow by the lag.

### 2.7 Tests

**E4, EST part: 7 pins break outright (+3 contract-illegal Plan D fixtures).** 6 of them (the inventories' set) break at
E4 unless re-pointed earlier: re-point them in **E2** (feas stage 1a), once meta `cts_anchor_idx` exists, so E4 edits
only the expected idx values; the `confirmed_at` pins survive.
- `test_ms_cts0_scan.py:75` (`cts0.idx == tfb.extreme_idx`; after E5 `pattern_extreme_idx`);
- `test_unified_probe.py:1056-1060` (`(idx, confirmed_at) = (9, 10)`);
- `test_ms_stop_after_cts.py:318` (the `(9, 10)` tuple);
- `test_structure_lifecycle_moment.py:184` (`[_cts_est(10,0,0,11)] → {0: 10}`);
- `test_parent_tables.py:250-268` (negative-control fixtures with idx ≠ `confirmed_at` become contract-illegal);
- `test_reversal_resolver.py:284-296` (extreme branch = `winner.idx`).

Plan D tests (HEAD coords, added after the inventories):
- `test_poi_activation_moment.py:87` (the `(9, 10)` tuple): the 7th pin; re-point to meta `cts_anchor_idx`;
- `:138-139`, `:242-243` and `test_render_sub_projection.py` HEAD `:793-801` are fixtures with idx != `confirmed_at`
  (contract-illegal after E4): re-author them via the E2 fixture factory;
- the `test_poi_activation_moment.py:1-16` docstring ("CTS_ESTABLISHED.idx (the CTS anchor)") rewords at E4.

**Fallback pin that conflicts with the no-fallback contract:** `test_unified_probe.py:847-853`
(`test_falls_back_to_idx_without_the_meta_key`, `meta={}` at `:850-851`) pins `_second_cts_moment`'s
`.get("confirmed_at", ev.idx)` fallback. Delete or invert it (expect `KeyError`) when that read becomes direct indexing.

**Becomes correct by construction:** `test_main_reversal_probe.py:109-117`, `212-219`. Today it pins EST `.idx` ==
`tfb.est_idx` (the moment) while `test_ms_cts0_scan.py:75` (and `:81-83`) pins the same field to the anchor. Both pass
only because the fixture has lag 0.

**Silent label pin:** `test_unified_probe.py:782` (the `cts1_ext=` regex).

**Re-author to the flipped contract.** These LOCATION fixtures carry no `confirmed_at`, so they pass unchanged until
re-authored. Exception since Plan F: the `_ev` helpers of the first two files set `confirmed_at = idx` (lag 0) on
CTS_ESTABLISHED and `via = CTS_UPDATED_RAW_VIA` on CTS_UPDATED, because FibTracker now reads both by direct index
(`event_moment`); their lines below are pre-Plan-F (+7, §0):
- `test_cross_cycle_fib.py:64-65`, `140-319`;
- `test_main_versioned_cross.py:86`, `110`, `207-208`, `244`;
- `test_wave_candles.py:198-224`, `389-618`;
- `test_render_sub_projection.py:609-638`, `705-710`.

**TIMING pins whose meaning flips to correct** (they pass explicit values, so they still pass):
- `test_sub_structure_pool.py:562-564` (`knowable_at_idx('CTS_ESTABLISHED', 10, confirmed_at=14) == 10`: E3b changes
  it);
- `test_render_sub_projection.py:312-323`, `575-603`;
- `test_first_trigger_migration.py:258-277`, `600-812` (the sibling input would break for EST candidates unless it
  reads the anchor key);
- `test_sub_wvmi.py:108-241`;
- `test_wvmi.py:446-547`.

**Unaffected by EST:**
- `test_pooled_structure_build.py:128-136`;
- `test_ms_bounded_equals_truncated.py:265-378`;
- `test_bounded_structure.py:182`, `204`;
- `test_sid_records.py:31-104`;
- `test_lifecycle_sweep_predicted_table.py:64-125`;
- `test_zone_proximity.py`.

**Breadth:** 18 test files build CTS_ESTABLISHED events (~139 hits), hence one fixture factory. A strict accessor raises
`KeyError` on old fixtures.

**E4, BOS part (additional):**
- `test_render_sub_projection.py:618-638` (`bos_extreme_abs`: 57 not 55), `:705-710` (KL anchor == BOS_0 idx ==
  `_STARTING_IDX`);
- `test_first_confluence_trigger.py:51`;
- `test_structure_lifecycle_moment.py:179-184` + `test_parent_tables.py:116` (the struct_start base, Q4.5);
- `test_sid_records.py:31-104` (the main `creation_event_idx` fixture values);
- `test_lifecycle_sweep_predicted_table.py:64-125` (the BOS values);
- the BOS pins in `test_wave_candles.py`.

**E1:** the event-fixture `anchor_idx` keys (see §1.1); `test_true_first_breakout.py` (`anchor_idx`).
`test_ms_stop_after_cts.py:71-85` (the `_step_anchor` monkeypatch) is untouched, since the name stays.

**E3b:** `test_sub_structure_pool.py:562-564`.

**Pinned sort tests:** new tests for H1–H3.

### 2.8 Per-stage predicted `/compare` (vs `20260923_172626_0a4eadc`)

Written against `20260923_172626_0a4eadc`. Since Plan F, predict against Plan F's save: Plan F's cells
(IMBALANCE_FILL_SEMANTICS.md "c3 knowability (2026-09-24, Plan F)") are then the baseline, and the counter chart
count is 151/124.

| Stage | Predicted | Verification |
|---|---|---|
| E1 pattern-anchor rename + amendment | Events meta **key text** changes on 47 rows (H1 7 = CTS_EST 5 + RC 1 + RWS 1; conf 32 = 21 + 4 + 7; counter 8) and on 5 H1 `structure_levels` meta rows; values identical. +1 H1 final.csv header line if `pending_reversal_*` is included. Charts identical | Reverse key substitution → byte-identical. WVMI byte-identical proves the `wave_candles` readers moved |
| E2 explicit anchor fields | The events meta **gains** `cts_anchor_idx` on 34 CTS_EST rows (H1 5 / conf 21 / counter 8) and `bos_anchor_idx` on 34 BOS rows (5 / 21 / 8); the levels meta gains 10 (5 + 5). Nothing else changes in the CSVs. run.log: the Phase-2 early-stop label `cts1_ext=` becomes `cts1_anchor=` (values identical) | A meta-key-stripping diff of all 24 CSVs is empty; charts identical (85/245, 151/124 since Plan F, 294/233); suite green; the grep gate; synthetic lagging fixtures (an EST winner for L1; a lagging BOS for B5) |
| E3a fib timing | 4 fib_lifecycle rows (§2.3) | re-measure first. A unit regression fixture already exists in shape: `test_unified_probe._make_second_cts_moment_after_extreme_data` (EST(0,1) idx 9, `confirmed_at` 10) through `_run_downstream_pipeline` (fib_mode `cross_cycle` or `h1`; `project_to_window` in the source) gives the cycle-1 fib `start_idx` = meta `activated_at` = 9 (the anchor) against the moment 10, and the cycle-0 fib `end_idx` 9 (re-run 2026-09-23; unchanged under Plan F, re-run 2026-09-24). E3a should make these 10 |
| E3b pool clip / recency | 0 | a synthetic cap in [anchor, moment) |
| E3c `check_lo` | 0 | verify the Phase-2 run |
| E3d prev-BOS filter | 0 | — |
| E3? type-rank order / struct_start base | 0 / unmeasured (H1 sid0 96→115, masked) | per decision |
| E4 flip | **Events `idx`:**<br>- CTS_EST: 3 cells (conf 1223→1224, 2828→2829; counter 2828→2829); H1 0;<br>- BOS: 34 cells (H1 96→115, 591→652, 689→703, 728→748, 826→902; conf 21, lag 2–411; counter 8, lag 4–285).<br>**KL meta `source_event_idx`** on the 34 BOS-zone rows (verified on `20260923_172626_0a4eadc`: KL `source_kind` BOS = 5 / 21 / 8; all 29 M15 values slice-local).<br>**POI `activation_history` on IC 678 / IC 2808** (the pre-window → in-window path switch, §2.2 L3): expected identical.<br>**Zero change** in `structure_levels` (re-sourced in E2), POI `cts_established_idx` (already the moment) and fib `cycle1_bos_idx`.<br>**Everything else byte-identical**, including triggers / subs / pool keys and the charts.<br>**run.log** `kl_zones` debug prints and probe labels change | cell-for-cell against this table |
| E5 remaining renames | Internal: 24/24 byte-identical; no run.log label renames expected. The naming_inv list is superseded: `cts0_est` is re-sourced in E3c; the `[fib] CTS idx=` labels stay (§1.10); the `[kl_zones][events]` prints change at E4 (B12); `[uc1_trigger] lifecycle_end_idx=` goes with the §1.8 deletion. Re-check with the silent-skip grep. Exported: the triggers header `validated_parent_idx` (2 lines) | reverse substitution. Keep chart hover / legend text out of byte-identical steps |
| post: fallback POI | **0 on this window since Plan F.** The delta predicted here (counter CSV 13→12, shapes 125→124, traces 153→151) already happened in Plan F, which removed the only live case (§3 #19). The general defect stays parked | — |
| post: coordinate hygiene (one family per `/compare`, M15 rows only) | POI `bos_idx`/`cts_idx` 43 rows · fib `activated_at` 30, `locked_at` 25, `reactivated_at` 2 · KL `source_event_idx` 29, `expanded_last_idx` 4 · event meta RANGE_STARTED `cts_idx` (39 + 15), `pb_start` (21 + 8), `pullback_apply_idx` (7 + 2), `start_idx`/`confirm_idx`/`effective_idx`/`expires_idx` | after the renames, so each key enters the shift list once |

**Sizing** (naming_inv, re-scoped):
- **Internal renames:** ~95 identifiers in ~22 production files.
  - ~45 single-module locals and labels touch no tests.
  - ~25 cross-module renames touch tests: `ReferenceZone.source_event_idx` 34 test lines / 4 files;
    `validated_h1/parent_start` 21 / 6 (deleted); `on_cts_established(bos_idx=)` 2 files; `_find_prior_cts_anchor` stays.
- **E2 bulk** (feas):
  - 10 CTS + ~15 BOS location sites;
  - ~5 sort pins;
  - the fib_tracker role split across ~8 helpers;
  - the accessor module;
  - a fixture factory touching up to 18 test files.
- **E4:** about 10 lines of code plus the docs pass.
- **Optional big renames, deferred:** `probe_end_idx` (94 prod / 150 test lines), `creation_event_idx` (17 / 18),
  `enforce_cts0_new_extreme` (14 / 7), test `CONF` (208).
- **Scope-creep guard:** 863 "anchor" hits in 53 .py files and 368 in 18 .md files. Keep the renames out of the
  behaviour stages.

### 2.9 Risks (condensed from feas)

| # | Risk |
|---|---|
| R1 | **Pool keys**: L1 and B5 set probe inputs, and L1 is invisible on this window, so unit fixtures must pin it |
| R2 | **Silent mixed-scale compares** (H4) |
| R3 | **Chart dots**: they move in x while y stays `ev.price`; `_replacement_break_point` inherits the shift |
| R4 | **Ordering** (H1–H3) |
| R5 | **`_EVENT_META_IDX_KEYS` is unmaintained.** Guard-test proposal (E1, before any key moves): every `*_idx` / `*_at` meta key on mirrored sub events (and zone meta) is entity-absolute, so a new idx key (`pattern_anchor_idx` in E1; EST `cts_anchor_idx` / BOS `bos_anchor_idx` are already listed) fails the test unless it joins `_EVENT_META_IDX_KEYS` / `_ZONE_META_IDX_KEYS`. The existing slice-local keys (§3 #4) need an explicit allow-list until their coordinate-hygiene `/compare` |
| R6 | **The window barely exercises the flip**: only 2 lagging EST cycles. Every migrated site needs a synthetic lagging fixture, e.g. `(idx, confirmed_at) = (9, 10)` |
| R7 | **Fixture drift**: use one factory; never `.get(key, ev.idx)` |
| R8 | **MS state must stay on the anchor** |
| R9 | **Contract / doc drift.** Code, docs and memory must move in the same commit. External CSV readers see the idx change |
| R10 | **Residual exceptions after E4**: pattern-path CTS_UPDATED `.idx` is still an anchor with **no recorded moment** (the §17.12 latent leak), so "ev.idx = moment" is still not universal. Benefit: a flipped event is no longer back-dated, which removes a live-mode look-ahead class |

### 2.10 Design-constraint checklist (the stage-E1/E2 gate; details in the sections cited)

| # | Constraint | Where |
|---|---|---|
| i | Flip only at the emit site (`market_structure.py:1485`: pass `apply_idx` + meta `cts_anchor_idx`). `st.cts`, `_initial_bos_before_first_cts(cts_idx)`, `range_start_idx`, the sd-prox gate and the POI-inner resolver stay on the anchor; redefining the `cts_idx` local breaks BOS_0 selection | §2.1, R8 |
| ii | `_emit_bos_confirmed` builds `st.bos_confirmed` and `bos_threshold` from the emitted idx: decouple (`st.bos` from the `bos_anchor_idx` param) BEFORE the BOS flip | §2.1, §1.5 |
| iii | After an EST-only flip, the fib new-extreme guards (`fib_tracker.py` ~1310/1383/1509/2342/2396) compare mixed scales and silently drop CTS_UPDATEDs in (anchor, moment]; POI cond1 has the same shape | H4, R2 |
| iv | MS emits CTS_ESTABLISHED before BOS_CONFIRMED, so a stable moment-only sort breaks the fib loop (BOS-before-EST holds only alphabetically today) | H1, Q4.3 |
| v | The KL `source_event_idx` value changes on BOS rows at the BOS flip, and the KL BOS-zone derivation (`kl_zones_v1.py:893-896`) must read meta `bos_anchor_idx` | B2, §1.2 |
| vi | The `pattern_anchor_idx` rename moves in ONE commit: the wave_candles readers `:295` / `:492` / `:553`, the H1 chart REVERSAL join (`export_plotly.py:1551` / `:1558`, direct index) and `_EVENT_META_IDX_KEYS` | §1.1, §3 #5 |
| vii | `test_unified_probe.py:850-851` builds `meta={}`, pinning `_second_cts_moment`'s `.get("confirmed_at", ev.idx)` fallback, which conflicts with the no-fallback contract | §2.7 |
| viii | 18 test files build CTS_ESTABLISHED: use one fixture factory, and assert `confirmed_at == idx` after the flip | §2.1 Accessors, §2.7 Breadth, R7 |

---

## 3. Latent bugs and export-hygiene items found (none fire on the window unless noted)

| # | Item | Site | Effect / action |
|---|---|---|---|
| 1 | `return self.df, self.events, self.levels`: `self.levels` is never assigned | `market_structure.py:450` (verified at HEAD) | `AttributeError` on the `start_idx >= n` early return. One-line fix, any stage |
| 2 | Conflicting test pins on `CTS_ESTABLISHED.idx`: one pins the moment, the other the anchor | `test_main_reversal_probe.py:109-117`, `212-219` vs `test_ms_cts0_scan.py:75` (+ `:81-83`) | Both pass only on lag-0 fixtures. Resolved by E4 (re-point `:75` to meta) |
| 3 | **Silent cross-kind `.get` fallbacks: 16 sites (+1 at the flip).** The sources and memory said "15", but the enumeration has 16:<br>- moment → anchor: `first_confluence_trigger.py:89`, `zone_proximity.py:311`, `unified_probe.py:456`, `:594`, `kl_zones_v1.py:889`;<br>- anchor → CTS_CONFIRMED moment: `export_m15_chart.py:521`, `:2025`, `export_plotly.py:1089`, `uc1_trigger.py:82`, `reference_zone.py:96`, `fib_tracker.py:1640`, `kl_zones_v1.py:896`;<br>- CTS name → BOS anchor: `fib_tracker.py:1433`, `:1524` (`c0.get('cts_idx', new_state.bos_idx)`);<br>- BOS name → CTS anchor: `fib_tracker.py:1442`, `:1532` (`meta.get('cycle1_bos_idx', cts_idx)`).<br>A 17th becomes cross-kind at E4: `unified_probe.py:605` (`.get('cts_anchor_idx', first_cts.idx)`; the sources said `:606`).<br>Moment name → CTS anchor (from extreme_final #30 / naming_inv #2, not in the 16-site set): `fib_tracker.py:1056` `.get("activated_at", cts_idx)` → direct index at E3a. | as listed | Direct indexing / assert (as `parent_tables` / `structure_lifecycle` / Plan D do). The amendment makes it mandatory for migrated keys. Byte-identical on the baseline; check fixtures that omit the keys first (e.g. `test_zone_proximity` `_bos_confirmed(confirmed_at=None)`) |
| 4 | **Slice-local exported meta keys on M15** (missing from the shift lists):<br>- `pb_start`;<br>- REVERSAL_* `expires_idx` (e.g. idx 1936 next to `expires_idx` 1537);<br>- STATE_CHANGED `effective_idx`;<br>- RANGE_STARTED `cts_idx`/`start_idx`/`confirm_idx`/`pullback_apply_idx`/`proximity_apply_idx`;<br>- KL `source_event_idx` (50 next to `anchor_idx` 454; 29 rows) and `expanded_last_idx` (4);<br>- POI `bos_idx`/`cts_idx` (50/380 vs the fib CSV's 454/784, `slice_begin` 404; 43 rows);<br>- fib `activated_at` (30), `locked_at` (25), `reactivated_at` (2; while `deactivated_at` IS shifted: 83 vs 2667), `cycle1_bos_idx` (0 M15 rows on the window);<br>- WaveCandleResult.meta `anchor_idx` (attrs only; `entity_df_mutation.py:297-312` copies without shifting) | `entity_df_mutation.py:103-124`, `:274` (fib shift tuple), `:288-289` | One coordinate-hygiene `/compare` per family (§2.8 post). Add the entity-absolute test (R5) |
| 5 | **Shift-list coupling**: renaming a key in `_ZONE_META_IDX_KEYS` / `_EVENT_META_IDX_KEYS` without moving the entry reverts the M15 values to slice-local (KL `anchor_idx` 29 rows, POI `cts_established_idx` 43 rows) | `entity_df_mutation.py:103-124` | E1 must swap `anchor_idx` → `pattern_anchor_idx` in the EVENT list only |
| 6 | **Export coupling**: `_TRIGGER_COLUMNS` / `_UNRESOLVED_COLUMNS` are derived from `fields(TriggerRecord)` / `fields(UnresolvedTrigger)`, so renaming a field (`validated_parent_idx`, `probe_input_idx`, `probe_finalize_idx`, `trigger_idx`, `parent_floor_idx`) silently renames a `/compare` header | `debug/export_sub_tables.py:34` | Treat as an exported rename |
| 7 | `data_bridge` swallows OANDA chunk errors (`[data_bridge] ERROR fetching …`): one audit run silently got 2500 of 4228 M15 candles. With no H1 triggers there is no `[data_bridge]` line, so the fetch gate fails rather than returning N/A | `multitf/data_bridge.py:36-47` | Covered by the `/compare` fetch gate (`37bcf76`). **CLOSED 2026-09-27:** the fetch retries a failed chunk request twice, then raises (an all-empty fetch raises too); the gate reads N/A from the log (compare skill §2b) |
| 8 | `validated_parent_idx` mixes timeframes (H1 96 next to M15 454), contradicting the TriggerRecord docstring "all idxs entity-absolute M15"; for sibling rows it duplicates `probe_input_idx` | triggers CSV | Q4.4 |
| 9 | FC `MultiTFTrigger.start_price` uses the wrong candle side (`h` for a bullish parent BOS) | `first_confluence_pipeline.py:35-39` | Unread; delete (§1.8) |
| 10 | `close_to_ext` uses an absolute 0.00015 price tolerance, which does not scale with pip size | `structure_patterns.py:242-248` | Note for strategy tuning |
| 11 | The dormant fallback passes `breakout_apply_idx` (a moment) into `_initial_bos_before_first_cts(cts_idx=)` | `market_structure.py:2240`/`:2242` | Flag in E2 |
| 12 | The KL debug print reads meta `bos_prev`, which MS never sets; BOS_CONFIRMED is printed twice | `kl_zones_v1.py:752-772` | E4: print idx + meta anchor explicitly |
| 13 | Dead code: `_is_new_cts_extreme` (`market_structure.py:1693`), `proximity_confirmed_idx`; `ReferenceZone` doc lists never-constructed sources (`reference_zone.py:81-82`); `_find_m15_by_extreme` duplicates `map_candle_to_lower_tf` | as listed | Delete or delegate (E5) |
| 14 | Stale or false comments:<br>- `entity_df_mutation.py:481-482` ("ev.idx == knowable-at for CTS types");<br>- `structure_engine.py:463-465`, `:574` ("pullback confirmation");<br>- `market_structure.py:1490`, `:1741`, `:1459-1470`;<br>- `CROSS_CYCLE_FIB_SPEC.md:89-90` (a stale `:1803`);<br>- `reference_zone.py:344-347` "the engine never emits two CTS-type events at the same idx for the same cycle" is false TODAY, not only after E4: confluence sub 2 has a raw + pattern-path CTS_UPDATED duplicate at 2468 (§0.4);<br>- the POI lookup defaults a missing `structure_id` / `cycle_id` to 0 (`poi_zones.py` HEAD `:372-373`) while `compute_cycle_lifecycle` skips such events, so its "asserts first" guard does not cover them (Plan D landing nit, only qualified in the HEAD `:468-479` comment) | as listed | §1.9 |
| 15 | Name collisions:<br>- `fb_idx` vs `WVMIRecord.fb_idx`;<br>- `m15_end_idx` local vs the meta key;<br>- `next_bos_idx` is the BOS moment in zone_proximity (`:348`) but the BOS anchor in `export_plotly.py:1197` / `export_m15_chart.py:588`/`:2077`;<br>- `cts_confirmed_for_idx` (anchor) next to `cts_confirmed_idx` (moment) | as listed | §1.5 / §1.6 |
| 16 | `KLZone.source_time` / `source_price` pairs a moment's time with an anchor's price | `kl_zones_v1.py:941-942` | Q4.12 |
| 17 | `knowable_at_idx` keys REVERSAL_CANDIDATE on its idx although its moment is `meta["apply_idx"]` | `sub_structure_pool.py:84-100` | Outside the 3 types; note for E3b |
| 18 | Parked by Plan D §7:<br>- `poi_active_as_of` treats the end as inclusive while the scan is exclusive (`poi_lifecycle.py:69` vs `poi_zones.py:483`; only the H1 proximity caller; 0 delta);<br>- the POI_ZONES_SPEC §3.2 IC idx constraints are not implemented (dead `scenario_context`) | — | not Plan E |
| 19 | The never-established-cycle fallback POI: a POI whose `(sid, cycle)` has no CTS_ESTABLISHED takes `fib_state.cts_idx` (a fib anchor) under the moment name `cts_established_idx`. **No live case since Plan F.** The only one on the window was the twin of IC 3654 attached to a cross fib that FibTracker PRE-CREATED for a cycle 1 that counter sub 5 never establishes (sub 5 only ever establishes cycle 0): value 3806, `end_idx` None, drawn past sub 5's end 4083, its fib uncapped. Plan F no longer creates that fib: at the CTS_THRESHOLD_UPDATED at 3806, the only imbalance in the own window is the single-c2 instance (3806, 3806), not yet formed. The fib row and the POI are gone | `poi_zones.py:482` (HEAD `754a642`; `:487` after Plan F) fallback | the post-E item (§2.8); 0 delta on this window |
| 20 | **Pattern-path CTS_UPDATED sets `st.cts` unconditionally**: no new-extreme check, so it could regress the CTS. 0 observed | `market_structure.py` ~`1549-1554` (the "Update current CTS point (always)" block) | Record; decide with Q4.1. **FIXED 2026-09-27** (user: spec-strict — both paths share `_is_new_cts_extreme`; the window's one same-candle tie, conf sub 2 2470, is gone: 1 exported row) |
| 21 | **The anchor-first offline selection can establish a cycle LATER than a live engine would, never earlier.** Continuous SUCCESS is checked before an earlier 2-candle SUCCESS (`structure_patterns.py:685-698`), and anchors inside a back-fill are never pattern-tested. Code-read only | `structure_patterns.py:685-698` | **Live-mode relevant** (Phase 3 / live driver), not Plan E |
| 22 | The dormant `_select_bos_on_breakout` fallback passes the apply MOMENT into `_initial_bos_before_first_cts(cts_idx=)` | = #11 | E2 |
| 23 | `reversal_confirmed_by_sid` holds `REVERSAL_CANDIDATE.meta["apply_idx"]`, a last-seen *scheduled* prediction, under a "confirmed" name, and feeds the FibTracker revert / Scenario-1 terminals | `orchestrator.py:148-155` | §1.6 / Q4.14. **FIXED 2026-09-27 (MAIN B, user option A):** the realised reversal, `reversal_idx_by_new_sid`; 0 cells on the reference window (every candidate realised at its apply); a discarded candidate had put a phantom 'reversal' terminal on the fibs |

---

## 4. Open questions that remain for the user

**Resolved since the sources were written** (do not re-ask):
- the new key names (`cts_anchor_idx` / `bos_anchor_idx`);
- the BOS flip (YES);
- aliases (none: atomic migration);
- the retirement of "anchor" (no; two realms);
- `pattern_extreme_idx/_price` (confirmed);
- bare records stay bare (the GLOSSARY row landed);
- `REVERSAL_WATCH_START` → `pattern_anchor_idx` (confirmed);
- POI meta `cts_established_idx` (Plan D landed it as the moment);
- timing fixes before the flip, one per `/compare` (yes);
- the sequencing (0.1): E1 pattern-anchor rename + contract amendment → E2 explicit anchor fields → E3 timing fixes →
  E4 EST + BOS flip → E5 remaining renames;
- the contract amendment's shape (atomic migration, no aliases, `.get(key, fallback)` → direct indexing); only its
  exact wording is open (Q11);
- `_extreme_idx_for_cts_event` → `_cts_anchor_idx_for_event`; `st.bos_confirmed` → `st.bos`;
- the KL-zone and fib anchor families stay as named, including their non-idx companions;
- `RANGE_STARTED.meta["cts_idx"]` stays bare (it pairs with `cts_price`; §1.1).

**Still open:**

1. **Pattern-path `CTS_UPDATED`.** Should it join the flip (`idx = apply_idx` + meta `cts_anchor_idx`), only gain meta
   `confirmed_at` (additive; `apply_idx` is in scope at `market_structure.py:1551`), or stay as is? It is 7 / 23 / 7
   rows. It affects the reference-zone recency, the sibling clip, the wave_candles sort, POI `cts_idx_at_t` and fib
   `reactivated_at`. FibTracker `on_cts_updated` (`fib_tracker.py:1105-1131`, dispatched at `orchestrator.py:229-231`)
   would then need the anchor accessor. Should it also carry meta `cts_anchor_idx` so that every CTS event shares one accessor path?
   Plan F adds a reader: FibTracker's imbalance knowability cut is skipped on a pattern-path update
   (`event_moment` returns `None`; 0 cells on the window). A recorded moment would turn the cut on there (Plan F §7).
2. **Fib Scenario 1** `CTS_0 idx >= reversal_confirmed_idx` (`fib_tracker.py:767-771`, `:1344-1348`;
   `POI_ZONES_SPEC.md:160-161`; the same test at `orchestrator.py:284-287`): a LOCATION question (is the CTS_0 anchor
   past the reversal apply?) or a TIMING one (was CTS_0 established after it)?
3. **Event processing order** (`orchestrator.py:146`, `sub_wvmi.py:68`): pin it to the anchor accessor permanently, or
   move to moment order with an explicit type rank (BOS < EST < UPD < THRESHOLD < CONFIRMED)? The joint flip makes
   this mandatory before E4 (H1).
4. **`validated_parent_idx`**:
   - (a) rename it to `ref_anchor_idx` and keep mixed timeframes;
   - (b) replace it with an FC-only `parent_bos_anchor_idx` (H1);
   - (c) store the M15 value, which makes it redundant with `probe_input_idx`, then drop it.
5. **`compute_struct_start_by_sid` base** under the BOS flip: pin it to the BOS_0 anchor (today) or move it to a
   moment? Moving gives H1 sid0 96→115 and subs +2..+35, mostly masked; it needs its own `/compare`. The same question
   covers the main `SidRecord.creation_event_idx` (split it as `first_event_idx`?).
6. **Prev-BOS line END** on the H1 chart: the new structure's CTS anchor (today) or its establishment moment? A
   chart-review question with 0 difference on the window.
7. **E4 as one `/compare` or two** (EST 3 cells, then BOS 34 + KL meta)? One cause per `/compare` argues for two. With
   the E2 sort pins, either order is safe.
8. **`cycle1_bos_idx` cond3 as-of**: measure "BOS_1 did not fill cycle 0" at BOS_1's anchor (today) or at its moment?
   Its producer stays the anchor either way.
9. **Confirm** (memory already counts it in E1): `pending_reversal_anchor_idx` → `pending_reversal_pattern_anchor_idx`
   (an H1 final.csv header) in E1.
10. **Test-only Scenario-3 / Exception-2 paths** (`structure_engine.compute_structure_scenario_3`,
    `compute_structure_from_start`; `tests/test_scenario3.py`, `tests/test_bounded_structure.py`): delete them, or
    migrate them (and to the anchor or the moment)?
11. **The exact contract-amendment wording** (`LANDMINES.md` "Event Contract Rules" rule 2 + `ARCHITECTURE.md` "Event
    contracts"). The user approves it in Plan E.
12. **Undecided exported renames**:
    - `KLZone.source_time` → `confirmed_time`?
    - `StructureLevel.time`: keep it (re-sourced) or rename it to `anchor_time` now that the amendment exists?
    - KL `source_event_idx`: keep it (the recommendation) or split it?
13. **Optional renames** (the recommendation is none):
    - `enforce_cts0_new_extreme` → `cts0_scan_mode`: ~23 sites (`market_structure.py:293`, `357-365`, `1062`, `1464`;
      `structure_engine.py:171`, `206-211`, `840`, `897`; `unified_probe.py:475`, `539`; `entity_df_mutation.py:1097`;
      7 test sites) + 2 spec headings (`MARKET_STRUCTURE_SPEC.md:45`, `PART4_REFACTOR_SPEC.md:727`);
    - `starting_idx` → an `*_anchor_idx` form;
    - the idx-valued pattern-realm param `_schedule_reversal_from_anchor(anchor_idx)` and its expiry local `anchor`
      (`market_structure.py:894`, `:832-862`) → `pattern_anchor_idx`, per the user rule "for candle patterns, always
      label explicitly pattern_anchor_idx". Function names stay;
    - `anchor_shift` and the 12 final.csv `*_as0..2` columns (`candles_v2.py:169-250`, `304-316`; §1.10): keep (the
      default; pattern realm: the big-candle lookback cutoff `idx - anchor_shift`, measured from a pattern's first
      candle) or rename? An exported header change (12 H1 final.csv columns);
    - the exact name of `continuous()`'s 3-candle extreme (keep `extreme` or `c0_c2_extreme_price`).
14. **Reversal-scope names**: `reversal_confirmed_by_sid` / `reversal_confirmed_idx` hold the *scheduled*
    `REVERSAL_CANDIDATE.apply_idx` (→ `reversal_apply_idx_*`?), and `_synth_reversal_trigger`'s `reversal_apply_idx`
    holds the *confirmed* R (→ `reversal_idx`?). In Plan E or separate? (2026-09-27: the orchestrator map now holds
    the realised reversal — `reversal_idx_by_new_sid` — so `reversal_confirmed_idx` is accurate; the
    `_synth_reversal_trigger` half stays open.)
15. **The write-only `FibRetracement.anchor_high_idx/anchor_low_idx`**: delete the unread fields, or keep them (the fib
    family stays as named)?
16. **Accessor-module naming**: the moment accessor's name under the final moment vocabulary (feas used `moment_idx(ev)`,
    which is not a moment spelling in 0.1; Plan F has since landed `market_structure.event_moment(ev)` for the three
    CTS types, §2.1). Should the existing moment names outside 0.1 (`ParentTables.cts_moment` /
    `bos_moment`, `TrueFirstBreakout.est_idx`, `ProbeResult.cts0_est_idx`) be respelled as `*_established_idx`?
17. **Scheduling**:
    - the §1.8 deletions: before E2 (so E2 does not migrate dead `.idx` reads through its grep gate) or in E5?
    - the non-Plan-E queue: the coordinate-hygiene families (one `/compare` each), the fetch-gate N/A edge case, the
      frozen-bridge rows missing from the ARCHITECTURE table (put them in E1's docs?), and `self.levels` (any stage).
18. **E3e, the MS in-flight POI-inner resolver as-of** (§2.3): add it to the timing-fix stage as its own `/compare`
    (proposed; it is engine-side and unmeasured), or park it? (Plan F settled its knowability half, which stays uncut;
    the question is only the fill horizon.)
