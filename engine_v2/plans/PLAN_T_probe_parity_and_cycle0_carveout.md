# PLAN_T — the first_confluence probe reads what its sub reads + the cycle-0 carve-out on the main sid 0 and its first_confluence sub only (the user's carve-out revisit, 2026-10-27)

**Status: COMPLETE 2026-10-27 — T1 LANDED (`453f227`, save S_T1 `20261027_131836_453f227`, = §2.3 exactly); T2 LANDED
(= §3.3 exactly; §5).** Baseline **S_wvmi_marks**
`artifacts/commits/week9-entries-decisions/20261026_230619_2dc9e5d` (source `2dc9e5d`; HEAD at drafting `3f2a29c`, code =
`2dc9e5d`); suite 2039. Session order item 5b (register `IDEA_PARKING_LOT.md` §D "SESSION ORDER"). TWO landings, in the
user's order ("Probe first"), each with its own measured prediction, `/compare` and chart look: §2 T1 the probe → §3 T2
the carve-out.

Inputs (`qa_cycle0_carveout_inputs/`): every Q&A round verbatim + the measurements
`user_spec_20261027_cycle0_carveout.md` (rounds 1–5); the variant patchers on a `git archive HEAD` copy in a short
`%TEMP%` dir — `make_variant_c0.py DIR shadow|drop`, `make_variant_probe_tf.py DIR` (T1's pip half),
`make_variant_rule_c_sub0.py DIR` (T2's rule, hack-identified — §3.3); the tables `c0_table.py DIR` (one row per
structure's cycle 0) and `c0_evdiff.py BASE CUR` (per-sub event diff).

---

## 1. The decisions (verbatim in the design file)

- **The carve-out stays on the main sid 0 and on the sub that the first_confluence of the main sid 0 cycle 0 CREATES
  (+ that trigger's probe); it is dropped on every other cycle 0** (main reversals, every other sub). Round 1: "C: main
  sid 0 only" → the user: "I think what we actually can/should do is keep the carve-out on both the main sid 0 and the
  sub_id 0" … "sub_0 would effectively always be fired via FC". Keyed on what the sub IS, not on `sub_id == 0`
  (author's refinement, round 1 reply).
- **The first_confluence probe's MS uses the caller's timeframe values (M15: proximity 5 / gap 30), not H1's 8 / 50**
  (round 4: "I do think that we need to update to 5/30 and be consistent") **and its sub's cross-fib limit** (round 5:
  "yes let's include … the probe should and need to be the same as final MS"). No move to copying the probe's results
  into the sub ("let's also not force anything right now").
- Order: T1 (the probe) first, then T2 (round 5).

## 2. T1 — the first_confluence probe's MS reads what its sub reads

### 2.1 Today (code at `2dc9e5d`)

- Only `first_confluence` runs the probe's MS (`unified_probe._run_phase2`, `enable_phase2=True`, one production
  caller: `entity_df_mutation.py:748`); the sibling / reversal / main-reversal probes are deterministic only (no MS).
- `_run_phase2` builds its MS with `_make_market_structure(df_probe, …, cross_fib_limit=CrossFibLimit(None, "none"))`
  (`unified_probe.py:541-554`) — NO `timeframe` → the helper's default `"H1"` → `proximity_pips` 8, `min_gap_pips` 50
  on M15 candles (`structure_engine.py:78-79`; the docstring's "Parity carve-out … a deferred cleanup", from `df11996`
  2026-05-05 — an inherited default, never a choice: design file round 2) — and NO `pip_size` → MS's default 0.0001
  (`market_structure.py:280, 396-398`; harmless on NZD_USD, 100× off on a JPY pair).
- The sub's own geometry: `compute_bounded_structure(…, timeframe="M15", cross_fib_limit=P.localized(slice_begin))`
  (`entity_df_mutation.py:1205-1215`) → 5 / 30, `_pip_size_from_pair(df)` (`structure_engine.py:393, 400`), the sub's `P`.
- The probe's reset already reads the caller's timeframe (`DEFAULT_PROBE_RESET_PIPS` / `_WICK`,
  `unified_probe.py:771-776`) and `pip_size` (`:770`) — unchanged.

### 2.2 The change

1. `unified_probe(…, cross_fib_limit: Optional[CrossFibLimit] = None)` — REQUIRED (assert) when `enable_phase2=True`,
   ignored otherwise; `_run_phase2(…, timeframe=…, pip_size=…, cross_fib_limit=…)` (keyword-only) →
   `_make_market_structure(…, timeframe=timeframe, pip_size=pip_size, cross_fib_limit=cross_fib_limit)`.
2. The limit = THE formula `cross_fib_limit_for("first_confluence", (S, C), t.direction, s, …)` (the one the sweep's
   `_resolve_cross_fib_limit` and the post-sweep tripwire use): for a first_confluence `NO_LIMIT` on C = 0,
   `parent_cts_limit(h1_events, S, C)` on C ≥ 1 (a price + its anchor price, no indices → nothing to localize; the
   probe's frame is the entity-absolute `m15_df`). `s` is read only by the counter branch and the x−1 branch
   (`cross_fib_limit.py:240-241, 247`); a first_confluence takes neither — **a NEW assert** in the orchestrator's
   `_resolve_start`: a first_confluence's `t.direction == tables.parent_sd[S]` (true by construction today,
   `first_confluence_pipeline.py:47`, but not asserted anywhere); `s` = the trigger's `trigger_idx`. **Computed LAZILY,
   on a probe-cache MISS only** (a zero-argument callable handed down to `_probe_with_cache`, called just before
   `unified_probe`): today `parent_cts_limit` runs only at a pool miss; eager evaluation would run its two asserts
   (`:161, :167`) on FC (0,1) / (1,2) for the first time (both pass — checked by the review: one CTS_ESTABLISHED each,
   `cts_frozen` 0.58448 / 0.57376, one previous-cycle CTS_CONFIRMED each). Threading: `_resolve_start` (has
   `sorted_events`, `tables`, `orchestrator.py:1144-1151`) → `_resolve_trigger_m15_start(…, cross_fib_limit=…)` →
   `_resolve_first_confluence_via_unified_probe` → `_probe_with_cache` → `unified_probe`. The sibling path takes none.
   `debug/probe_fc_finalize.py:135` updated.
3. `_make_market_structure`'s `timeframe` default is KEPT (the review: `compute_structure` / `compute_bounded_structure`
   keep their own H1 defaults, so making only the helper's required buys nothing); its parity paragraph is replaced by
   one line ("every caller passes its timeframe; the probe since PLAN_T T1").
4. **A probe↔sub consistency tripwire** (log only; the user's "its own 2nd CTS should still be consistent"): when a
   first_confluence probe ran (NOT a cache hit) and its start CREATES the geometry, a `WARNING [sweep] probe/sub
   disagree …` line when — `second_cts_reached`: the finalize ≠ the sub's own 2nd CTS_ESTABLISHED moment (anchor
   order, entity-absolute); `no_retrace`: the sub's own cycle-0 CTS_CONFIRMED lands BEFORE the finalize (the
   finalize = the probe's cycle-0 confirmation, else its probe end — which the check does not know, so a probe
   confirming before its sub is NOT caught). As built: an injected callable of the sweep (§5). Silent on the window under T1 and T2 (baseline / `c0T2`: 1020 = 1020); would
   have fired on `c0drop` (probe 1020, sub [458, 685, 1020]). Cheap; skip grep unchanged (33).
5. **Accepted approximations — where a first_confluence probe still differs from its sub after T1 (+ T2):** (a) a
   probe-CACHE hit runs no probe at all (on the window FC (0,1) and FC (1,2), APPROX hits at keys (1,2365) — the
   reversal probe — and (−1,3304) — the subsequent_counter probe; both Phase 1 only); (b) a probe whose start lands on
   an EXISTING sub: the sub keeps its creator's limit, `bos0_inner` (and, with T2, its flag) —
   `lifecycle_sweep.py:334-336`, `entity_df_mutation.py:1259-1261`; "the sub's limit" is then the FC trigger's; (c) the
   probe's MS is BOUNDED at `probe_end_idx` (truncation semantics, Plan A). The frame difference (the sub's slice
   re-derives `is_range_*` and imbalances, `entity_df_mutation.py:1181-1196`) was checked by the review and cannot cause
   a divergence on the probe's span. (b) and (c) are what item 4's tripwire watches; (a) is the §17.8 cache's
   first-probe-is-truth (unchanged). New eager surface (safe; the landing review): the limit FACTORY (its direction
   assert + an events copy) runs for every non-pending, non-collapsed first_confluence reaching the resolver — cache
   hits and resolver failures included; the formula itself runs on a probe-cache miss only, so a miss that then lands
   on an existing pool sub now evaluates `parent_cts_limit` (before, only a pool miss did).

### 2.3 Prediction — CSVs and charts IDENTICAL (or a raise)

By construction, under today's carve-out (verified by the review, every line cited there):
1. MS reads the two pip values in ONE path — the proximity CTS confirmation: the buffer in
   `_check_proximity_at_candle` (`market_structure.py:1892`) and the gap in `_maybe_confirm_cts_via_proximity`
   (`:1914`); `proximity_pips` also lands in a proximity CTS_CONFIRMED's meta (`:1977`). `self.timeframe` is otherwise
   read only in log prints. `pip_size` scales the same two values (NZD_USD: 0.0001 either way).
2. MS reads the embedded tracker's output in ONE path — the POI inner (`_refresh_poi_inners_for_cycle` →
   `_check_proximity_at_candle`, + `config.fill_threshold`) — handler return values discarded.
3. The probe reads, from its MS run: the CTS_ESTABLISHED events (anchor order), cycle 0's CTS_CONFIRMED, a reversal
   and the reversal mask (`market_state` / `structure_id`) only when fewer than 2 CTS (a run that never left cycle 0),
   `df_probe` prices for the retrace candidate, `ms.early_stop_idx` only for a print. Up to the 2nd CTS_ESTABLISHED the
   run is in cycle −1 / 0 (a CTS_ESTABLISHED is emitted only at a pattern apply, never inside a range look-ahead; a VOID
   one is restored inside its own step), where the carve-out blocks every proximity confirmation → the same events;
   nothing after the 2nd CTS is read.
"Or a raise": the limit is read during cycle 0 by the tracker handlers (`ic_reference`, `find_cross_limit_break`), and
cycle-1 events after the 2nd CTS reach `check_processing_order` / `_check_invariants_df`; a different raise would
propagate (no catch in the sweep) — visible, not silent. On the window the probe's MS runs ONCE — FC (0,0)
(`[probe_cache] miss … key=(1,385)`), whose limit is `NO_LIMIT` = today's — so T1 there = 8 / 50 → 5 / 30 on one run.

### 2.4 Tests and mutants

- New pins: (a) an FC probe's MS gets the caller's timeframe values — parametrized M15 (5 / 30) and M5 (3 / 15), so a
  hard-coded `"M15"` dies; (b) its `pip_size` = `_pip_size_from_pair` — a JPY-tagged frame (0.01); (c) it gets the
  handed limit object; (d) `unified_probe(enable_phase2=True)` without a limit raises; (e) the resolver's limit ==
  `cross_fib_limit_for("first_confluence", …)` on C = 0 (`NO_LIMIT`) and C ≥ 1 (`parent_cts_threshold`, price = the
  parent CTS_ESTABLISHED's `cts_frozen`) and is NOT computed on a cache hit (lazy); (f) the direction assert fires on a
  counter-direction first_confluence; (g) the tripwire: silent when equal, one WARNING when the sub's 2nd CTS differs
  (a fake geometry), silent on a cache hit.
- Under the carve-out no BEHAVIOUR test can tell 8 / 50 from 5 / 30 in the probe — the mutants (drop `timeframe=`, drop
  `pip_size=`, `CrossFibLimit(None, "none")` again, the resolver handing no limit, eager evaluation) die by the
  construction pins; T2 adds the behaviour pin.
- **Existing tests that change** (the review's census): the REQUIRED limit with `enable_phase2=True` — 16 calls in 4
  files (`test_unified_probe.py` 758, 779, 817, 884, 1100, 1114, 1150, 1171, 1189, 1206, 1260, 1273;
  `test_ms_stop_after_cts.py:393`; `test_plan_h_landing_b.py:292, 298` — `:298` expects `ICSelectionError` from
  `_run_phase2`, the new assert must not fire first; `test_plan_i_s5_ms_reads_the_fib.py:287`); PLAN_I F7's pin
  `test_the_probe_passes_no_p_and_does_not_depend_on_the_read` (`test_plan_i_s5_ms_reads_the_fib.py:290`, asserts
  `{CrossFibLimit(None, "none")}`) → rewritten to "the probe passes its sub's limit"; the direct `_run_phase2` calls
  (`test_ms_stop_after_cts.py:463, 468`; `test_unified_probe.py:858, 861, 1001, 1006, 1303`) — keyword-only args with
  defaults where the test does not care; the first_confluence resolver calls (`test_first_trigger_migration.py`, ~22)
  — the resolver's new parameter defaults to `None` and the assert fires only when Phase 2 actually runs (the one
  production caller always passes it — spy-pinned), so only the tests that reach Phase 2 pass a limit.

### 2.5 Docs (same commit)

`_make_market_structure` docstring; `unified_probe._run_phase2` docstring + the call-site comment
(`unified_probe.py:550-552`, PLAN_I §9.2 F7's premise); MARKET_STRUCTURE_SPEC "Trigger threshold per timeframe"
(`:541-544`: "The start-candle probe's Phase-2 run builds through the helper WITHOUT `timeframe` … inert by design" →
it passes the caller's timeframe + pip size) and its "Pip size derived from df.attrs" (`:545`, now true for the probe);
LANDMINES (`:250`, "the probe: an explicit `CrossFibLimit(None, "none")`") → its sub's limit; CROSS_CYCLE_FIB_SPEC
(`:275`) likewise; register §E "The start-candle probe's market structure runs at the H1 proximity values on M15" →
CLOSED by T1; the design file; PLAN_I stays historical (a dated pointer to PLAN_T at F7); the `/compare` skill's
PLAN_T T1 entry (zero change; the tripwire line is new and silent).

### 2.6 Measured (variant `%TEMP%/c0T1` = `make_variant_probe_tf.py` alone on `git archive HEAD` — the pip half; the
limit half is `NO_LIMIT` = today's on the window's one probe MS run, the pip size 0.0001 = MS's default on NZD_USD)

Fetch gate PASS; skip grep 33; 59.0 s. **30 / 30 CSVs byte-equal; the three charts equal by trace name (`fig_named_diff.py`:
83 / 234, 158 / 136, 306 / 307, layouts equal); the probe's log lines (`[probe_cache]`, `[unified_probe phase2] early
stop: … stop_idx=1021 … cts1_moment=1020`, the FC resolver line) identical to the baseline's.** = §2.3. The landing's
own replay is the measurement of the limit half + the lazy call + the tripwire (predicted: the same, the tripwire
silent).

## 3. T2 — the cycle-0 carve-out on the main sid 0 and its first_confluence sub only

### 3.1 The rule — and its general effect

Proximity may confirm a cycle-0 CTS EXCEPT on (a) the main sid 0 (`compute_structure`'s `structure_id == 0` — the
trading-open structure: its start is Scenario 1's pick, no probe, no structural reference) and (b) the sub CREATED by
the first_confluence of the main sid 0 cycle 0 (`H1.main`, parent (0, 0)) — its probe input IS the main sid 0's BOS_0
anchor (H1 96 → M15 385 on the window) — and that trigger's probe. ("Created": the sub is built on that trigger's
pool miss; its first live record may be another — the distinction `check_cross_fib_limits` makes.) Every other cycle
0 (main reversals, every other sub) is guarded by the narrow-gap gate alone, like every cycle k > 0. Why the two:
design file "How each structure's cycle 0 is found" + round 1.

In general (NOT on the window): an earlier, proximity cycle-0 confirmation also (i) shrinks an unblocked FC probe's
retrace window (`[CTS_0 moment + 1, the confirmation's CTS anchor − 1]`, `unified_probe.py:616-621`) and moves its
`no_retrace` finalize (`:638-641`) → can move the sub's START (its pool key) and lifecycle start; (ii) on a main
reversal sid N ≥ 1, moves FC (N, 0)'s `parent_cts_anchor_idx` (the confirmed CTS's anchor, `first_confluence_trigger.py:83`)
→ that probe's end. Neither happens on this window (§3.3); the spec rewrite says so.

### 3.2 The change

1. `MarketStructure(…, cycle0_proximity_blocked: bool = True)` (the default = today's behaviour for direct construction
   — 8 test files); the gate `st.cts_cycle_id > 0` → `(st.cts_cycle_id > 0 or not self.cycle0_proximity_blocked)`.
   **Name proposed — the user's call** (alternatives: `block_cycle0_proximity`, `cycle0_proximity_carve_out`).
2. ONE predicate, `cycle0_proximity_blocked_for(trigger_type, parent_cycle, parent_path) -> bool` =
   `trigger_type == "first_confluence" and parent_cycle == (0, 0) and parent_path == "H1.main"`, **evaluated ONCE per
   trigger in the sweep's `_resolve_and_record`** and handed to both consumers (not re-derived from the resolver's
   default `parent_path`):
   - the probe: `resolve_start(t, idx, cycle0_proximity_blocked=…)` → `_resolve_trigger_m15_start` → the FC resolver
     → `_probe_with_cache` → `unified_probe(…, cycle0_proximity_blocked=…)` → `_run_phase2` → `_make_market_structure`;
   - the geometry, on a pool MISS: `build_geometry(key, bos0_inner, limit, blocked)` → orchestrator `_build_geometry`
     (`orchestrator.py:1152`) → `build_or_get_geometry` → `compute_bounded_structure` → `_make_market_structure`; on a
     pool HIT `blocked=None`, asserted like the limit (`entity_df_mutation.py:1260, 1262`), so a hit cannot silently
     drop a differing flag; stored on `PooledStructure.cycle0_proximity_blocked` for traceability — **NOT exported**
     (`debug/export_sub_tables.py` has an explicit column list; no save-format change).
   - A tripwire: when the predicate is True and (`res.cache_hit` or the geometry was NOT created by this trigger) → a
     `[sweep] WARNING` (the "FC (0,0) is the main's earliest trigger" premise, checked instead of assumed).
3. Defaults: REQUIRED at `_make_market_structure` (3 production callers) and at `build_or_get_geometry` (the single
   production owner of a sub's geometry; 6 test calls); `compute_bounded_structure(…, cycle0_proximity_blocked: bool =
   False)` and `unified_probe(…, cycle0_proximity_blocked: bool = False)` — False = the rule for every sub / probe but
   one; their one production caller each passes it explicitly (spy-pinned). The sweep's `build_geometry` /
   `resolve_start` callables change signature → the ten fake builders in 5 test files are updated.
4. `compute_structure`: `cycle0_proximity_blocked=(structure_id == 0)`. The reversal-born subs (trigger `reversal`,
   parent (0, 0) included) → False.
5. `debug/probe_fc_finalize.py` passes the predicate's value.

### 3.3 Prediction (measured — variant `%TEMP%/c0T2` = `make_variant_c0.py drop` + `make_variant_probe_tf.py` +
`make_variant_rule_c_sub0.py`; behaviour only; the limit half of T1 is moot on the window, §2.3)

The hack identifies sub 0's geometry and its first_confluence probe by their start candle (2025-11-21 15:30, sd +1) and
the main sid 0 by tf H1 + structure_id 0. Equal to the planned rule on the window (the review): exactly one Phase-2 MS
ran, from 454 (`p2_iter=1`; the resets 385 → 441 → 454 are in the deterministic phase, no MS); one geometry at key (M15,
+1, 454); the only H1 MS runs are `compute_structure`'s; the `[c0probe]` counts show both runs caught and blocked.

Fetch gate PASS; skip grep 33 (= the baseline); 65.0 s. **6 / 30 CSVs differ, subs 4, 7, 8 only; H1, the triggers,
the sub list, every sub's start / end, KL / POI / IC, the WVMI records: byte-equal** (re-derived independently by the
review).

| CSV | change |
|---|---|
| M15 counter structure_events | sub 8: CTS_CONFIRMED 4103 (`one_maru_opposite`) → 4098 (`sd_zone_proximity`, POI 0.57484) + CTS_RECONFIRMED 4103 (591 → 592 events) |
| M15 confluence structure_events | sub 4: 3489 → 3336 (POI 0.57658) + CTS_RECONFIRMED 3489; sub 7: 4157 → 4071 (POI 0.57653) + CTS_RECONFIRMED 4157; sub 8 as counter (1388 → 1391) |
| M15 counter fib_lifecycle | sub 8 c0: `meta.locked_at` 4103 → 4098 (1 row) |
| M15 confluence fib_lifecycle | subs 4, 7, 8 c0: `meta.locked_at` 3489 → 3336, 4157 → 4071, 4103 → 4098 (3 rows) |
| M15 counter wvmi_lps | sub 8 c0: 6 → 8 rows — the first temporary LP 4102 (as of 4103) → 4098, 4101, 4102 (74 → 76) |
| M15 confluence wvmi_lps | sub 4 c0: 11 → 25 (+14 temporary LPs 3336 … 3488); sub 7 c0: 11 → 26 (+15, 4070 as of 4071 … 4156); sub 8 6 → 8 (185 → 216) |

Charts (`fig_named_diff.py`): H1 equal (83 / 234); counter 158 / 136, layout equal, ONE trace changed — `M15 WVMI LP
buy` (+2 marks); confluence 306 / 307, layout equal, TWO traces changed — `M15 WVMI LP buy` (+16: sub 4 +14, sub 8
+2) and `M15 WVMI LP sell` (+15: sub 7). The confirmation moment itself is not drawn on the M15 charts. Both
tripwires silent.

### 3.4 Tests and mutants

- Pins (synthetic frames, one per side, modelled on sub 4's 3304–3336 run and sub 7's 4000–4071): a cycle 0 with a
  gap ≥ the M15 floor whose wick comes within 5 pips of the POI / BOS inner → confirms by proximity when not blocked;
  does NOT when blocked (the pattern confirms later, no CTS_RECONFIRMED); a gap of exactly the floor → allowed (`>=` —
  NO existing boundary pin, the review), one tick under → not. `compute_structure` blocks sid 0 and not sid 1 — a
  CONSTRUCTION spy (a Scenario-1 behaviour fixture is impractical). The predicate's truth table (FC (0,0) on `H1.main`
  → True; FC (0,1), FC (1,0), `subsequent_confluence` (0,0), `reversal` (0,0), FC (0,0) on another parent path →
  False). The sweep hands ONE evaluation to both the probe and the geometry (spies); a pool hit passes `None`; the
  FC-(0,0)-not-creator tripwire fires on a fake hit. The behaviour pin T1 left open: an unblocked FC probe whose cycle 0
  confirms by proximity at a gap in [30, 50) reaches its 2nd CTS where its sub does (silent consistency tripwire).
- Mutants (sketched against the fixtures BEFORE the run — each competing pair must differ AND straddle the boundary):
  drop `st.cts_cycle_id > 0 or`; drop `not`; each predicate conjunct; `(0, 0)` → `(0, 1)`; `structure_id == 0` →
  `True` / `False`; the geometry / the probe passing a constant; `>=` → `>`. The `/compare` cannot kill the predicate
  or `structure_id` mutants (H1 sid 0's 16 cycle-0 hits all have gaps < 50; subs 1, 2 have no cycle-0 hit; FC (0,1) /
  (1,2) are cache + pool hits) — only the truth-table + spy pins can, so they are required.

### 3.5 Docs (same commit)

MARKET_STRUCTURE_SPEC "Dual CTS confirmation paths" (the gate list) + "Cycle-0 carve-out (proximity disabled)"
rewritten (the rule + where it applies + why + its general effect §3.1 + the history: 2026-05-11 / 12, the
true-first-breakout redesign after it; the "Under review" note removed; the BOS_0 wording: the extreme between the
structure's start and CTS_0's anchor, not "before the structure began"); LANDMINES "Dual CTS Proximity Confirmation —
Cycle-0 + Rule 1 Are BOTH Required" rewritten (title + scope + the 2026-10-27 decision; the concrete cases re-measured:
sid 0 at 116 — POI inner 6.2 p under CTS 0.56126, gap 31.8 p; sid 1 c0 no hit); register §D 5b → DONE (+ "Recently
closed"); `engine_v2/CLAUDE.md` status; the `/compare` skill's PLAN_T T2 entry (no save-format boundary — same columns,
positional `cmp_save` fine; figures with `fig_named_diff.py`); memory.

### 3.6 Chart look

A zoomed render (`zoom_png_volume.py`) of sub 7 (4000–4160) and sub 4 (3300–3490) on the confluence chart: the earlier
LP marks. Whether the M15 chart should also SHOW a proximity confirmation (a mark at 4071 / 3336 / 4098) — the user's
call at the look (not in this plan).

## 4. Reviews and cost

- **Plan cold review — DONE 2026-10-27:** 2 read-only reviewers, 528k tokens (engineering 280k / 65 tool uses;
  adversarial 248k / 55). No BLOCKER. Folded in: the probe's `pip_size` (both reviewers); the direction assert is NEW,
  not existing, and the `s` wording; the limit computed lazily; the existing-test census; T2's defaults, the missing
  `_build_geometry` link, `blocked=None` on a pool hit; the "Trigger threshold per timeframe" spec passage; "identical,
  or a raise"; the reversal-mask read; FC (1,2)'s hit is on the subsequent_counter probe; 8 test files (not 13);
  `timeframe` stays defaulted; the remaining probe↔sub differences (§2.2.5); the flag not exported; why the hack is
  valid; the probe↔sub consistency tripwire (§2.2.4); one predicate evaluation + the creator tripwire (§3.2.2); T2's
  general effect (§3.1); the M5-parametrized pin, the gap boundary pin, the construction spy, which mutants `/compare`
  cannot kill. Verified correct by the reviews: every cited line, the byte-identical trace, the threading, one geometry
  build site, one H1 MS construction, `"H1.main"`, FC (0,0) the earliest fire, the T2 numbers, the hack = the rule.
- Landing reviews: 1 reviewer per landing, ~0.3M each.

## 5. Build log

### T1 — LANDED 2026-10-27 (the user: "Yes, build T1", then "Commit + save, then T2") — source `453f227`, save **S_T1**
`20261027_131836_453f227` (save commit `4d7318c`, trunk `1d7a6a0`; = `LATEST_week9-entries-decisions`)

- **Code:** `unified_probe._run_phase2` (keyword-only `timeframe` / `pip_size` / `cross_fib_limit`, REQUIRED) →
  `_make_market_structure(…, timeframe, pip_size, cross_fib_limit)`; `unified_probe(…, cross_fib_limit=None)`, asserted
  when Phase 2 runs; `cross_fib_limit.first_confluence_probe_limit(trigger, h1_events=, parent_tables=)` — the direction
  assert + a zero-argument callable over THE formula (a named helper rather than a closure in the orchestrator, so the
  assert and the formula are unit-testable); the orchestrator's `_resolve_start` builds it for a first_confluence →
  `_resolve_trigger_m15_start(…, cross_fib_limit=)` → `_resolve_first_confluence_via_unified_probe` →
  `_probe_with_cache` (calls it on a MISS) → `unified_probe`. **The consistency tripwire is an INJECTED callable**
  (`run_lifecycle_sweep(…, check_probe_sub_consistency=None)`; the orchestrator passes
  `entity_df_mutation.check_probe_sub_consistency`) — §2.2.4 put it inside the sweep, but the sweep's unit tests build
  stub geometries with no events and the module is pure orchestration by contract; the check reads the geometry with
  the probe's own helpers (`_collect_cts_established`, `_second_cts_moment`). `_make_market_structure`'s parity
  paragraph replaced; `debug/probe_fc_finalize.py` hands the limit (THE formula, its parent tables = `parent_sd` only).
- **Tests:** 21 new pins (`tests/test_plan_t_t1_probe_parity.py`: timeframe values × M15 / M5 / H1, pip size × NZD /
  JPY, the handed object, the REQUIRED assert, the cache's lazy call, the formula on C = 0 / C ≥ 1, the direction
  assert, the tripwire × 7 rows, the sweep's once-per-created call, the orchestrator's wiring, the dispatcher + FC
  resolver pass-through). Changed: the 12 + 1 + 2 Phase-2 calls (+ the limit), the direct `_run_phase2` calls (+ the
  three keywords; `TestPlanCRenames` gets its own `_kw2` so its `end_idx` TypeError still comes from `end_idx`), PLAN_I
  F7's pin → `test_the_probe_passes_the_handed_limit_and_does_not_depend_on_the_read` (identity, not equality), the
  orchestrator closure test's fake resolver (+ `cross_fib_limit`, asserted None for the non-FC triggers). Suite 2039 →
  2060, green (77.6 s).
- **Mutants:** 19 / 19 KILLED (`qa_cycle0_carveout_inputs/mutants_plan_t_t1.py`, full suite each; results
  `%TEMP%/mtT1/results.json`).
- **Replay (run mode, repo):** fetch gate PASS (raw H1 byte-equal; M15 4228 candles in 5 chunks); skip grep 33 (= the
  baseline); the tripwire silent (0 lines); **30 / 30 CSVs byte-equal to S_wvmi_marks; the three charts equal by trace
  name** (83 / 234, 158 / 136, 306 / 307, layouts equal); the probe's log lines identical; `[wvmi] total=5, locked=3`,
  the same 3 `[uc1_trigger]` lines, `rendered 57 LP circles` — = §2.3 / §2.6. 168.9 s wall (the mutant run in
  parallel).
- **Docs:** MARKET_STRUCTURE_SPEC "Trigger threshold per timeframe"; LANDMINES (the wiring item — + the stale
  `FIB_FILL_THRESHOLD` → `IMBALANCE_FILL_THRESHOLD`); CROSS_CYCLE_FIB_SPEC "The embedded tracker"; register §E line
  CLOSED; PLAN_I F7 pointer; this plan; the design file.
- **Landing review** (1 read-only reviewer, 208k tokens / 60 tool uses): no BLOCKER. Fixed: the tripwire's
  `no_retrace` branch (§2.2.4 — built at first with `second_cts_reached` only); laziness pinned at the factory
  (`test_the_first_confluence_limit_is_lazy` — an eager factory SURVIVED the suite and the replay); the orchestrator's
  events + tables wiring pinned with a parent-cycle-1 first_confluence (an `h1_events=[]` mutant SURVIVED); LANDMINES
  (the order-tripwire item's "never depends on the read") + CROSS_CYCLE_FIB_SPEC (the sub formula's readers) +
  `cross_fib_limit.py`'s docstring (a third reader); the sweep hands the check its own `log`; NITs (`match="end_idx"`,
  distinct `cts_frozen` vs CTS_CONFIRMED price, the test module's existing `NO_P`, blank lines); §2.2.5's eager
  surface. The `/compare` skill entry: written with the save (`/commit-save`).
- **After the review fixes:** suite 2065, green (141 s); mutants **26 / 26 KILLED** (the file rewritten for the
  two-branch tripwire + the review's misses; `%TEMP%/mtT1b/results.json`); the replay on the final code: fetch gate
  PASS, skip grep 33, the tripwire silent, **30 / 30 CSVs byte-equal, the three charts equal** — 65.3 s.

### T2 — LANDED 2026-10-27 (the user: "Looks good, commit T2"; a confirmation mark on the M15 charts: "Not needed")

- **Code (= §3.2):** `MarketStructure(cycle0_proximity_blocked=True)` + the gate `st.cts_cycle_id > 0 or not
  self.cycle0_proximity_blocked`; REQUIRED at `_make_market_structure`, `build_or_get_geometry` (None on a pool hit,
  asserted) and the orchestrator's `_resolve_start`; `compute_structure` passes `structure_id == 0`; default False on
  `compute_bounded_structure` / `unified_probe` / `_probe_with_cache` / the FC resolver / `_resolve_trigger_m15_start`
  (+ an assert: only a first_confluence may carry True); the predicate `lifecycle_sweep.cycle0_proximity_blocked_for`
  evaluated ONCE per trigger in `_resolve_and_record`, handed to the probe and (on a pool miss) the geometry; the
  creator tripwire (`WARNING [sweep] the cycle-0 carve-out's first_confluence … did not create its sub`);
  `PooledStructure.cycle0_proximity_blocked` (not exported); `debug/probe_fc_finalize.py` passes the predicate.
- **Tests:** `tests/test_plan_t_t2_cycle0_carveout.py` — 31 pins (the gate × 4 rows, the exact narrow-gap boundary,
  MS's default, the L5 cycle 0 blocked / unblocked, `_make_l4_watch_at_bound` on M15 × both sides — blocked: the pattern,
  no CTS_RECONFIRMED; unblocked: proximity a candle earlier + CTS_RECONFIRMED at the pattern —, the bounded run's
  default, `compute_structure` sid 0 vs the reversal (a construction spy), the builder's REQUIRED flag, the predicate × 7,
  the sweep's one evaluation + the pool hit's None, the creator tripwire × 2, the pool builder × both values, the core →
  the bounded run, Phase 2's MS × both values, the probe chain × both values, the sibling assert); the T1 orchestrator
  pin extended (the flag reaches `_resolve_trigger_m15_start` and `build_or_get_geometry` as the sweep hands it). The
  existing tests keep their pre-T2 meaning by passing `cycle0_proximity_blocked=True` where they build an MS directly
  (18 `_make_market_structure` calls), a direct Phase 2, the `test_ms_reversal_terminal` fixtures (built under the
  universal carve-out: unblocked, the L5 cycle 0 confirms by proximity and the range-back-fill scenario disappears),
  the `test_main_reversal_probe` reconstructions of sid 0, and the probe-vs-full test in `test_unified_probe`; pool
  builder calls pass False on a miss / None on a hit; the sweep fakes take the keyword. Suite 2065 → 2096, green.
  **Descoped (§3.4):** an END-TO-END pin of an unblocked first_confluence probe confirming cycle 0 by proximity and
  reaching its 2nd CTS where its sub does — no synthetic fixture makes the probe's own cycle 0 proximity-confirm (the
  candidates either never establish on the downtrend side or confirm by the pattern, searched 2026-10-27); the
  agreement holds by construction (one evaluation → the probe and its geometry; the same MS values since T1), each link
  is pinned, the log tripwire is unit-pinned (T1), and the replay agrees on the window (FC (0,0): 1020 = 1020).
- **Mutants:** 36 / 36 KILLED (`qa_cycle0_carveout_inputs/mutants_plan_t_t2.py`: 29 + the landing review's 7 —
  the orchestrator's two pass-throughs, the hard-coded True on each hop).
- **Replay (run mode, the final code):** fetch gate PASS; skip grep 33; both tripwires silent; **30 / 30 CSVs equal to
  the measured prediction (`c0T2`)**; vs S_T1 the 6 predicted CSVs, the 3 predicted traces — = §3.3 exactly. 61.1 s.
- **Docs:** MARKET_STRUCTURE_SPEC (the gate list; "Cycle-0 carve-out" rewritten — the rule, why the two, the history,
  what the narrowing changes; the BOS_0 wording; the refresh-gating sentence; the CTS_UPDATED-on-cycle-0 scope);
  LANDMINES (the entry retitled "Dual CTS Proximity Confirmation — the Cycle-0 Carve-out (main sid 0 + its FC sub) +
  Rule 1", the re-measured May cases, the REQUIRED / default map); register (5b DONE + "Recently closed", the probe
  line's trigger); `engine_v2/CLAUDE.md`; the `/compare` skill; 13 review scripts under `plans/` pass the old flag
  (`variant_patch_i.py` left: its strings are anchors into an archived tree).
- **Landing review** (1 read-only reviewer, 222k tokens / 64 tool uses): no BLOCKER; folded in — the orchestrator's
  flag REQUIRED + its pass-through pinned, both flag values on the probe chain and the pool builder, the L4 behaviour
  pin (the blocked branch "the pattern confirms later" was untested), the stale spec / register / skill lines, the
  LANDMINES REQUIRED-vs-default map, the review scripts, the sweep's hints + docstring, the reconstructions' flag, the
  parameter documented.
