# Plan K — The IC reference condition (item 1's rule experiment RX1)

**Status: DRAFTED 2026-10-09; COLD-REVIEWED 2026-10-09 (§9: 2 read-only lenses, 526,601 subagent tokens — no
BLOCKER on fidelity, 5 MAJOR on the code paths, every finding folded below); the user's questions §8 (K-Q1 … K-Q8)
— ALL ANSWERED 2026-10-09 (§8); next: landing A's build, on the user's go.** The rule is the user's and is DECIDED: canonical text `PLAN_H_institutional_pattern_poi.md` §8.9
"RX1" (the rule as decided) + §6 RX1-B, RX1-1 … RX1-4, RX1-T, RX1-W1 … W3, RX1-C1, RX1-E; the user's write-up verbatim
`plan_h_inputs/ic_reference_conditions_spec.md`. This file is HOW it is built: the code audit, the design, two
landings with exact predictions, the tests, the cost. Baseline: **S_S6**
`artifacts/commits/week9-entries-decisions/20261009_020650_5f6bae3` (= `LATEST_<branch>`); HEAD at drafting `2a013d4`.

---

## 1. The rule, as an engineering reading (the canonical home stays PLAN_H RX1)

For a fib record (structure `sid`, direction `sd`, target cycle `T`, BOS anchor in cycle `a` — a single: `a = T`; a
cross: `meta["cross_start_cycle"]`), every §3.1 candidate `i` must pass ONE bundled condition (RX1-B) before the touch
and the institutional pattern:

1. **References.**
   - `prev_cycle_cts` — when `a ≥ 1`: cycle `a−1`'s `CTS_CONFIRMED` price (the confirmed anchor — never a
     `cts_threshold`; the tracker already records it, `_cts_by_cycle`). Its bound = cycle `a`'s `CTS_ESTABLISHED`
     `meta["pattern_anchor_idx"]` — the pattern that broke `CTS_{a−1}` and formed `CTS_a`.
   - The structure reference — when the structure has a cross limit `P` and (`M` is None or `T ≤ M`), cycle 0 included
     (the test of `_cross_allowed`, evaluated WITHOUT calling it: a structure with no limit entry, or `P` None, has no
     structure reference — §9 C03):
     - `prev_structure_bos` (source `reversal_bos_frozen`): the reversing structure's most recent `BOS_CONFIRMED` price
       at its reversal; bound = the realised reversal pattern's anchor (RX1-1);
     - `cross_limit_anchor` (every non-reversal source with a `P`, RX1-2): the anchor behind `P`, read at the same
       moment `P` is read (§3.3); bound = the cross-limit breakout watch's break on `P` (§3.4; RX1-2b, W1–W3), None
       until / unless one confirms.
2. **Choice.** Both → the one further opposite the structure (sd +1: the lower price; sd −1: the higher; RX1-3 — also
   when the previous CTS already went past the BOS anchor); EQUAL prices → the STRUCTURE reference (K-Q5, RX1-K5); one → it; none → the condition PASSES
   (RX1-B, RX1-4).
3. **Price side.** sd +1: `high[i] < ref`; sd −1: `low[i] > ref` (strict; `passes_price_constraint`).
4. **Before-the-break side.** `i < bound` (strict) when the bound is known at the evaluation moment; no known bound →
   passes (RX1-2a). An IC can never BE the bound candle: a breakout / reversal pattern's first candle has the
   structure's own candle direction and closes past the threshold (`check_break`), and a candle of direction sd never
   has agreed direction −sd. "Known": a watch break once its confirmation ≤ the moment, detected by MS's convention
   (K-Q6, RX1-K6); a reversal-started structure's reversal pattern for its WHOLE run, like `P` (K-Q7, RX1-K7).
5. **Where.** At IC selection only (RX1-E) — `select_ip_ic`, i.e. both consumers (the drawn POI on the final fib,
   retrospectively, like §3.1; MS's in-flight read on its as-of fib at the last fed event's moment, §3.1 below). Not in
   the activation sweep. Unchanged: `M`, the legs, the touch, the IP, TB1 / TB2 (TB2 runs over the surviving tied ICs).

## 2. The code audit — every site (HEAD `2a013d4`; line numbers are hints, anchor on the names)

| # | Site | Today | Change |
|---|---|---|---|
| A1 | `zones/poi_zones.py` `select_ip_ic` (:400) | §3.1 → touch → `classify_ics` → rank / TB | takes `ic_reference=` (REQUIRED keyword — no silent bypass); evaluates the condition per candidate after `find_ic_candidates`; landing A: reports it only, B: filters (§4) |
| A2 | `poi_zones.IPCandidate` (:381) | `ip`, `touches_zone` | + `ic_ref_reason: Optional[str]` (None = passes) |
| A3 | `poi_zones.select_ip_records` (:511) | `select_ip_ic(df, fib, cfg, classify_misses=True)` | passes `fib_tracker.ic_reference(fib_state, df, evaluated_at=None)` |
| A4 | `poi_zones.poi_inner_for_fib` (:1300) — MS's read | `poi_inner_for_fib(df, fib, fill)`; everything inside one `try` → `IPSelectionError` | gets the tracker and the moment (`poi_inner_for_fib(df, fib, fill, *, tracker, evaluated_at)`) and calls `tracker.ic_reference(...)` INSIDE its `try` — so a `ValueError` / `IndexError` there surfaces as `IPSelectionError`, never reaching `_build_geometry`'s narrow `except (ValueError, IndexError)` (`entity_df_mutation.py:1159`, which would drop the whole sub — §9 C01); a None fib returns None before the call (C13) |
| A5 | `poi_zones.derive_ip_candidates` + `IPCandidateRecord` (:846) + `debug/export_zones.export_ip_candidates` | 35 columns on a lens CSV | + column `ic_ref_reason` (a string — no mirror shift) |
| A6 | `poi_zones.passes_price_constraint` (:142) | no caller, no test | the price side (reused; signature kept) |
| A7 | `poi_zones.find_ic_candidates` `scenario_context` (:186) | never read | retired (+ its docstring) |
| A8 | `zones/cross_cycle_fib.py` `CrossFibLimit` | `price`, `source`, `from_pool_idx`, `from_sub_id`, `started_by` | + `ref_anchor_price` (None ⟺ `price` None), + `break_pattern_anchor_idx` / `break_confirmed_idx` (set ⟺ `reversal_bos_frozen`); `__post_init__` asserts both; + `localized(offset)` (§3.5) |
| A9 | `zones/structure_lifecycle.compute_main_cross_fib_limit` (:91) | `P` = the reversal's `bos_frozen` | + sid N−1's most recent `BOS_CONFIRMED` price ≤ the reversal; + the reversal's `pattern_anchor_idx` (A13, read through `ef.pattern_anchor_idx`, A19) and moment |
| A10 | `multitf/cross_fib_limit.py` `reversal_limit` / `parent_cts_limit` / `read_prev_sub` / `check_cross_fib_limits` | `P` per source | + `ref_anchor_price` per source (§3.3); reversal: + the break, ENTITY-ABSOLUTE (the reversing geometry's idx + ITS `slice_begin`); the R6-2 tripwire compares the new fields too |
| A11 | `zones/fib_tracker.py` `FibTracker` | `_limits`, `_latest_anchor`, `_cts_by_cycle`, `_finalize_lifecycle_fields(events, floor, cap, cap_reason)` (:469) | + per `(sid, cycle)` the `CTS_ESTABLISHED` pattern anchor (recorded in `_on_cts_established`); + `ic_reference(fib_state, df, *, evaluated_at)` → `ICReference` (plain values only — no cached frame / detector: the tracker is deep-copied at every MS snapshot, §9 C02-snapshot); + `stamp_ic_references(df, lifecycle_cap)` — a NEW finalize step the orchestrator calls after `_finalize_lifecycle_fields` (that one takes no frame — §9 C03); the 4 keys stamped BEFORE the `cross_fib_limit_*` keys (`test_plan_i_s2_cross_fib_limit.py:541` pins those as the last 4) |
| A12 | `patterns/cross_limit_watch.py` (NEW) | — | `find_cross_limit_break(df, threshold, sd, start_idx, *, end_idx, known_at)` (§3.4) |
| A13 | `structure/market_structure.py` `_apply_pattern_at_apply_idx`'s reversal branch (:1301) — the ONE live REVERSAL entry, reached by the pending (:844) AND the winner (:895) path (`_maybe_trigger_reversal`, reason `bos_threshold_broken`, is dead: its only call is commented out at :912; ASSUMPTION, not a decision — a revived patternless reversal would have no bound, i.e. pass) | `STATE_CHANGED(to=reversal)` meta: `from`, `to`, `reason`, `pat`, `bos_frozen`, `effective_idx` | + `pattern_anchor_idx` = `int(ev.start_idx)` of the realised reversal `PatternEvent` (an event-contract ADDITION, format-only — K-Q4; reaches only the 3 `*_structure_events.csv` — charts read `to` + idx, no test pins the full key set) |
| A14 | MS's two refresh sites (`market_structure.py` :1424→1482 the pattern path, :1617–1623 the raw path) → `_refresh_poi_inners_for_cycle` (:1915) | `poi_inner_for_fib(self._resolver_df(), representative_fib, fill)` | + `tracker=self._fib_tracker, evaluated_at=<the last fed event's ef.event_moment>` — passed from BOTH call sites (`apply_idx` on the pattern path, the re-stepped `i` on the raw path; §9 C06); the frame = `self._resolver_df()` (the L5 rule, §9 C07); MS still imports nothing from `zones/` |
| A15 | `multitf/entity_df_mutation.py` | `_FIB_META_IDX_KEYS` (:159), `_EVENT_META_IDX_KEYS`, `_build_geometry` (:1104) | + `ic_ref_break_pattern_anchor_idx`, `ic_ref_break_confirmed_idx` in `_FIB_META_IDX_KEYS`; `pattern_anchor_idx` is ALREADY in `_EVENT_META_IDX_KEYS` (:115 — any event's key shifts); `cross_fib_limit.localized(slice_begin)` at the geometry |
| A16 | `entity_df_mutation.render_sub_projection` (:1288 — the `{0: lim}` literal; `project_to_window` only forwards it, `pooled_structure_build.py:103`) | `cross_fib_limits={0: sub.cross_fib_limit}` | `{0: lim.localized(slice_begin)}` (the projection runs on the geometry's slice — `bounded.df`) |
| A17 | `charting/_zone_render.py` `fib_hover_status_text` (:240) | status · life · IP · limit | + one line: `IC ref <price> (<source>[, cycle <a−1>]) · break <idx>` / `… · break: none` (a reference with no known break, subs 5 / 6) / `IC ref: none` (RX1-C1: hover + CSV only) |
| A18 | tests' static guards | `test_event_meta_idx_keys` (:166 the value-bound pin, :300 / :392 / :600 / :619 the inverse + broad scans), the meta-key guards | the new keys listed; the stamp written as a `meta=` literal in `fib_tracker.py` (or `_NOT_MIRRORED` entries) so the inverse guard sees it; the tracker's internal maps keyed without `*_idx` dict keys |
| A19 | `structure/event_fields.py` `_PATTERN_ANCHOR_TYPES` (:38, :70–76) | no `STATE_CHANGED` — `ef.pattern_anchor_idx` raises on it | + `STATE_CHANGED` (the A9 / A10 readers read the role through `event_fields`, the project rule; §9 C09) |
| A20 | the test-fixture migration (§9 C05) | — | (a) ~40 `CrossFibLimit(...)` constructions with a price (incl. the `== CrossFibLimit(...)` pins; `tests/_cross_fib_limits.py:20`'s helper) gain `ref_anchor_price` (+ the break for reversal ones); (b) `tests/_event_factory.make_cts_established` (:30–38) omits `pattern_anchor_idx` unless passed (59 call lines) → the factory stamps a default (the CTS anchor) so the tracker can read the key by direct index; (c) the reversal `STATE_CHANGED` fixtures fed to the limit builders (`test_plan_i_s2…` :110 / :116 / :347 / :621 / :676 / :755, `…s4` :89, the S5 pins :109–120, the predicted-table test :119) gain `pattern_anchor_idx`; (d) the 3-positional spies / direct calls of `select_ip_ic` / `poi_inner_for_fib` (the S5 pins :90 / :177 / :196 / :213, the S5 reads :95–102 / :296, `test_plan_h_landing_b` :153 / :189 / :199–296, `test_imbalance_c3_knowability` :342 / :350) take the new keywords; (e) the `_Tracker` fakes (`test_institutional_pattern:624` and others) gain `ic_reference` |

Docs (landing A): POI_ZONES_SPEC §3.2 rewritten (the old per-scenario tables → a history appendix), §3 intro, §3.5 (the
CSV column); CROSS_CYCLE_FIB_SPEC §3 (the limit's new fields, the watch); MARKET_STRUCTURE_SPEC (the reversal event's
new key; the refresh's moment); ARCHITECTURE (IC condition #4; the event table); GLOSSARY (IC reference, cross-limit
breakout watch); LANDMINES (the reference built inside the resolver's re-raising scope — C01's trap); the `/compare` +
`/commit-save` skill tables (the CSV column, the save-format boundary); register §D.2; memory.

## 3. Design

### 3.1 One value, two moments

`ICReference(price, source, break_pattern_anchor_idx, break_confirmed_idx)` (frozen; `ICReference.NONE` = no
reference) is built by ONE method, `FibTracker.ic_reference(fib_state, df, *, evaluated_at)`, from the tracker's own
state — the limit, the `M` test, `_cts_by_cycle`, the recorded establishment pattern anchors, the watch. The drawn POI
asks it with `evaluated_at=None` (retrospective); MS asks its embedded tracker, inside `poi_inner_for_fib`'s re-raising
`try` (A4), at the last fed event's moment (A14). `select_ip_ic` receives the value and never computes it — D2 holds by
construction (one selector, one reference builder). The tracker is already deep-copied into MS's state snapshot
(`market_structure.py:599`, PLAN_J J1), so new tracker state needs no snapshot work — kept to plain values (a cached
frame or detector would be deep-copied at every snapshot).

### 3.2 The condition (one pure function)

`ic_reference_reason(df, i, ref, sd, evaluated_at) -> Optional[str]`: None = passes; `ic_past_reference` (the price
side fails — checked first, K-Q2); `ic_not_before_break` (the price side passes, `i ≥ bound` with the bound known). In
landing A it fills `IPCandidate.ic_ref_reason` and the CSV column only; in landing B a candidate with a reason is not
classified for the pick (with `classify_misses` it is classified for the CSV, never picked — the touch's own pattern).

### 3.3 The anchor behind `P`, per source (RX1-2; the decided table, PLAN_H §8.9) — read at the moment `P` is read

| Source | `ref_anchor_price` | On the window |
|---|---|---|
| `reversal_bos_frozen` | the reversing structure's most recent `BOS_CONFIRMED` price by its reversal moment | H1 sid 1: 0.5736 (591); sub 1: 0.57882 (sub 0's 1761); sub 2: 0.57884 (sub 1's 2270); sub 3: 0.58198 (sub 2's 2561) |
| `parent_cts_threshold` | the parent cycle c−1's `CTS_CONFIRMED` price | sub 4: 0.57400 (H1 761) |
| `prev_sub_bos_threshold` | x−1's latest `BOS_CONFIRMED` price in the same read as `P` (`read_prev_sub`'s moment `min(s, x−1.end_idx)`) — the BOS whose threshold `P` is, never its `BOS_THRESHOLD_UPDATED` | sub 5: 0.58105 (sub 4's 3304); sub 6: 0.57112 (sub 5's 3621); sub 7: 0.57806 (sub 6's 3760; `P` = 0.57827 after the wick at 3806) |
| `prev_sub_cts_threshold` | x−1's confirmed cycle's `CTS_CONFIRMED` price | — (unit tests) |
| `prev_sub_latest_cts` / `prev_sub_starting_price` | `P` itself (already an anchor price) | — (unit tests) |
| `none` / `prev_sub_no_cts` | None (no structure reference) | H1 sid 0, sub 0 |

### 3.4 The cross-limit breakout watch (`patterns/cross_limit_watch.py`)

`find_cross_limit_break(df, threshold=P, sd, start_idx, *, end_idx, known_at) -> Optional[(anchor_idx,
confirmed_idx)]`: every candle `A` in `[start_idx, end_idx]` whose CLOSE is past `P` in direction `sd` (W2) opens a
watch: `detect_best_for_anchor(A, sd, P)` — the reversal watch's own detector and pattern set (continuous,
one_maru_opposite, one_maru_continuous, double_maru + confirmations), the break check `check_break` against `P` (W1).
The detector's latest confirmation is `A + 5` already — the reversal watch's window, no separate cap (a cap mutant
would be equivalent, §9 C15). `P` never moves; every later close past it re-arms (W3). Two watches confirming → the
EARLIEST confirmation wins, ties → the earlier anchor (K-Q1, RX1-K1). Detection by MS's convention (K-Q6, RX1-K6):
`BreakoutPatterns(df, end_idx=<the frame end below>)` — its look-ahead, `continuous` first — and a break counts at a
moment `t` iff its confirmation ≤ `known_at = t` (None = retrospective); so the in-flight answer is the final answer
cut by time and the bound never moves.
- `start_idx` = the structure's first candle — its BOS_0 anchor (`_bos_by_cycle[(sid, 0)]`), NOT a sub's lifecycle
  `start_idx`: sub 4's lifecycle starts at 3621, after its break at 3588 (the W1 example "sub 4 → 3588" holds only on
  this reading; from 3621 nothing closes past `P` before its end 3819) — §9 F02 / C12, written down, pinned.
- `end_idx`: MS's read → its resolver frame's end (`self._resolver_df()`, the bounded run's truncation — no candle past
  `end_idx` is read); the retrospective stamp → the structure's lifecycle cap (`lifecycle_cap`; the frame end when
  open), so no break after the structure ended is exported (§9 F05 / C12).
- Pure: no MS state, no structure event (the cross limit is not an MS key level — RX1-2b's own words).
- Prototype (`plan_h_inputs/icref_census.py`, `xlimit_watch_proto.py`): sub 7 → anchor 4188 (`one_maru_opposite`,
  confirmed 4189; closes past from 4183, the patterns at 4183–4187 fail); sub 4 → 3588 (`one_maru_continuous`, 3589);
  subs 5 / 6 → None. CAVEAT (§9 F11): the prototype's frame is rebuilt from the lens chart (volume 1, the whole lens
  frame — not the sub's `[start−50, run_cap]` slice); landing A's check against the engine settles it.

### 3.5 Coordinates (the trap this plan adds)

The new break indices are the first index fields a `CrossFibLimit` carries. Main: the H1 frame — no translation. Subs:
the pool computes the limit in ENTITY-ABSOLUTE M15 indices (the reversing sub's geometry index + ITS `slice_begin`);
the sub's geometry MS and its projection both run on the sub's slice `[start−50, run_cap]` (the projection takes
`bounded.df`), so the limit is translated ONCE per entry point — `lim.localized(slice_begin)` at `_build_geometry`
(:1120 / :1149) and at `render_sub_projection` (:1288) — and the fib meta's two break keys (slice-local in the tracker)
return to entity-absolute through `_FIB_META_IDX_KEYS` in the mirror. The probe's Phase 2 passes `CrossFibLimit(None,
"none")` (`unified_probe.py:553`) — no translation there. Pins on BOTH sides (§9 C04): the projection (sub 2: 2465
entity-absolute in the fib CSV and as the bound) AND the geometry (a spy: the embedded tracker's limit holds the
slice-local index on a reversal-started sub with `slice_begin > 0`).

### 3.6 Exports (RX1-C1: hover + CSV only)

- Fib records (every TF): meta `ic_ref_price`, `ic_ref_source` (`prev_cycle_cts` | `prev_structure_bos` |
  `cross_limit_anchor`; absent = no reference), `ic_ref_break_pattern_anchor_idx`, `ic_ref_break_confirmed_idx`
  (absent = no known break) — the FINAL values (`stamp_ic_references`, A11), placed before the `cross_fib_limit_*` keys.
- `*_ip_candidates.csv`: + `ic_ref_reason` (every row; empty = passes / no reference).
- The fib hover's new line (A17).

## 4. Landings — each ONE cause, its own commit + `/compare` (keyed scripts; name the save)

**A — compute + export, nothing switched.** A1–A20 with the filter OFF (the pick ignores `ic_ref_reason`). `/compare`
vs S_S6 named explicitly. Then `/commit-save` → **S_K_A**, B's baseline (K-Q3).

**V — the variant replay between A and B** (scratch tree, the PLAN_I `replay_variant_i.py` pattern): A + the filter ON.
Its outputs vs S_K_A = B's prediction, MS cascades included (the census sees final picks only; MS's as-of reads can
move where the final pick holds).

**B — the switch (one commit).** The filter ON in `select_ip_ic` for both consumers. `/compare` vs S_K_A == V. Chart
review. POI_ZONES_SPEC / ARCHITECTURE "IC condition #4 implemented" wording moves here. NOTE (§9 C04): V and B run the
same code, so B == V cannot catch a geometry-side bug (a dropped `.localized`, a wrong moment at MS's call site) — the
§6 spy tests do.

## 5. Predictions

**Landing A vs S_S6 (exact, from `icref_census.py`; written before the build; both lenses re-derived by the review):**
- `*_structure_events.csv` (H1 + the confluence lens; the counter lens has none): every `STATE_CHANGED(to=reversal)`
  row gains `pattern_anchor_idx` — H1 sid 0 @902 → 897; sub 0 @1940 → 1936, sub 1 @2470 → 2465, sub 2 @2829 → 2824,
  sub 6 @4200 → 4196. Nothing else in those files.
- `*_fib_lifecycle.csv`: the 4 keys per record — none for H1 sid 0 cycle 0 and sub 0 cycle 0 (no reference); the two
  break keys absent where the reference has no known break (subs 5 / 6). The values per record (26 distinct POI
  records; 32 lens rows):

  | Record | Reference | Bound |
  |---|---|---|
  | H1 (0,1) | prev_cycle_cts 0.58246 (CTS_0 430) | 651 |
  | H1 (1,0) / (1,1) | prev_structure_bos 0.5736 | 897 (confirmed 902) |
  | H1 (1,2) | prev_cycle_cts 0.57400 (CTS_1 761; over 0.5736) | 897 |
  | sub 0 cyc 1 / 2 / 3 / 4 | prev_cycle_cts 0.5732 / 0.5752 / 0.57774 / 0.58246 | 1019 / 1221 / 1720 / 1792 |
  | sub 1 cyc 0 | prev_structure_bos 0.57882 | 1936 |
  | sub 1 cyc 1 | prev_cycle_cts 0.58051 | 1893 |
  | sub 1 cyc 2 | prev_structure_bos 0.57882 (over CTS_1 0.57877 — RX1-3) | 1936 |
  | sub 1 cyc 3 / 4 | prev_cycle_cts 0.57656 / 0.57546 (T > M = 2) | 2230 / 2342 |
  | sub 2 cyc 0 | prev_structure_bos 0.57884 | 2465 |
  | sub 2 cyc 1 | prev_cycle_cts 0.58434 (T > M = 0) | 2607 |
  | sub 3 cyc 0 | prev_structure_bos 0.58198 | 2824 |
  | sub 3 cyc 1 | prev_structure_bos 0.58198 (over CTS_0 0.5818 — RX1-3) | 2824 |
  | sub 3 cyc 2 / 3 | prev_cycle_cts 0.57902 / 0.57400 (CTS_2 3047, confirmed 3086) | 2991 / 3588 (cyc 3's CTS_ESTABLISHED 3589) |
  | sub 4 cyc 0 | cross_limit_anchor 0.57400 | 3588 (confirmed 3589) |
  | sub 5 cyc 0, sub 6 cyc 0 | cross_limit_anchor 0.58105 / 0.57112 | none (never broken) |
  | sub 7 cyc 0 / 1 / 2 | cross_limit_anchor 0.57806 | 4188 (confirmed 4189) |

  (Non-record fib versions get their values from the same rule; the build's check lists them.)
- `*_ip_candidates.csv`: + `ic_ref_reason`; the counts per record = the census's `cand → pass` (e.g. sub 0 cyc 3:
  209 → 85; sub 0 cyc 1: 50 → 14; sub 7 cyc 2: 90 → 69).
- Charts: only the fib hover traces' `customdata` / `hovertemplate` (H1 3, confluence 16, counter 6 drawn records);
  every shape and every other trace equal. Every other CSV md5-equal (POIs, KL, WVMI, triggers, subs, imbalances,
  `_final.csv`). run.log: no new WARNING line. Tests: + the A list (§6), + the migration (A20).

**Landing B vs S_K_A:** = V. The census's floor: of the 26 POI records ONE pick changes — sub 0 cycle 3 (buy, ref
0.57774): 1604 1A.i (high 0.57880) cut → 1447 or 1660 (both 1A.iv maru — TB2 decides); the census's two "tie" records
(sub 1 cyc 1, sub 7 cyc 2) keep their tied groups ({1842, 1875}, {4083, 4136}) and picks. The cascade beyond it from V.

## 6. Tests (each written to kill a named mutant; `mutant_runner.py`, full suite per mutant)

- **The condition:** both sd, price strict at equality (`high == ref` fails), bound strict (`i == bound` fails —
  synthetic), no reference → passes, a reference with no known bound → price only, a bound confirmed after the moment →
  passes (as-of), the reason precedence.
- **The reference:** CTS only; structure only; both — further opposite, both sd; EQUAL prices → the structure reference
  and ITS bound (K-Q5 — a mutant choosing the CTS must die); the gap case per
  sd (RX1-3); `T > M` drops the structure reference, `T = M` keeps it, `M` None keeps it; no limit entry / `P` None →
  no structure reference and NO raise (the `bare` tracker, C03); a cross from cycle 0 (structure only) and a cross from
  `a ≥ 1` (CTS_{a−1}, bound = cycle `a`'s establishment — none on the window); a pre-established cross (`a < T`).
- **The anchor behind `P`:** all 7 sources (§3.3), incl. a `prev_sub_bos_threshold` whose threshold moved (sub 7-like:
  the anchor, not the threshold).
- **The watch:** a close past opens one, a wick past does not; the pattern anchored at the close candle; `P` as the
  break threshold (a pattern closing between the anchor and `P` does not count — sub 7's 4178); re-arm after a failed
  watch; the earliest confirmation over two concurrent watches + the anchor tie (K-Q1); never → None; MS's detection
  convention (K-Q6 — the reviewer's constructed case: at A+1 no break where a cut frame would see a 2-candle SUCCESS,
  `continuous` known at A+2; and the anchor-flip case does NOT flip); a reversal-started structure's bound known from
  the run's start (K-Q7 — a read before the reversal moment already cuts at the reversal anchor); `start_idx` = BOS_0
  (a break before a sub's lifecycle start is found — sub 4-like, F02); the retrospective end capped at the lifecycle
  cap; the loop never reads past `end_idx` (a bounded frame).
- **The reversal stamp (A13):** `pattern_anchor_idx` on `STATE_CHANGED(to=reversal)` on the winner path AND the
  pending path (`test_ms_reversal_on_expiry`'s fixtures); the main limit and `reversal_limit` read it through
  `ef.pattern_anchor_idx` (the later-anchor winner fixture, where no `REVERSAL_CANDIDATE` sits at the apply).
- **Plumbing:** `CrossFibLimit` asserts; `localized`; the R6-2 tripwire on the new fields; `FibTracker.ic_reference`
  as-of vs final; **C01** — a `ValueError` raised inside `ic_reference` during a sub GEOMETRY surfaces as a
  `RuntimeError` (`IPSelectionError`), the sub is not silently dropped; **C04 spies** — the embedded tracker's limit
  holds slice-local break indices on a reversal-started sub with `slice_begin > 0`, and `evaluated_at` == the fed
  event's moment at BOTH refresh sites (pattern path, raw path); D2 on the REAL collaborator (the drawn pick == MS's
  read pick on the final fib, with the reference); the projection round trip (§3.5); the fib meta keys (order: before
  the limit keys) + the mirror shift; the CSV column (+ an `IP: none` row); the hover line rendered (H1 + a lens chart,
  incl. `break: none`); the static index-key guards (A18).
- **B:** a fixture where the IP's best candidate is cut → the next best; all touching patterns cut → no POI; TB2 over
  the survivors only; MS's read filtered too.
- **Mutation:** every branch above + the call sites (A3, A4, A14 ×2, A16, `_build_geometry`'s localization, the stamp);
  the `A + 5` cap noted as an equivalent mutant (C15).

## 7. Size and cost

- A: 1 new module + ~14 files + ~10 docs; the fixture migration (A20: ~40 limit constructions, the event factory, ~15
  reversal fixtures, ~20 spy / direct-call sites) — ~1,400 lines, over half tests; the replay; `/compare`; the A-check
  vs the census; `/commit-save`. Solo.
- V: one variant replay (~1 min) + the keyed report. Solo.
- B: ~3 files + docs; the replay; `/compare` == V; chart review. Solo.
- Reviews: this plan — DONE (§9, 526,601 tokens); each landing — 2 lenses + the mutation run (sized then).

## 8. Questions for the user (K-Q) — ALL ANSWERED 2026-10-09 (AskUserQuestion), folded into §1–§6

| K-Q | Answer | Recorded |
|---|---|---|
| K-Q1 | (a) the earliest confirmation (ties → the earlier anchor) — recommended | PLAN_H §6 RX1-K1 |
| K-Q2 | one column `ic_ref_reason`, the price side first — recommended | here |
| K-Q3 | `/commit-save` after A (S_K_A = B's baseline) — recommended | here |
| K-Q4 | add `STATE_CHANGED(to=reversal).meta["pattern_anchor_idx"]` — recommended | here |
| K-Q5 | (b) **the STRUCTURE reference** on equal prices — NOT the recommended (a) | PLAN_H §6 RX1-K5 |
| K-Q6 | (a) MS's detection convention, a break counting once confirmed — recommended | PLAN_H §6 RX1-K6 |
| K-Q7 | (a) known for the whole run, like `P` — recommended | PLAN_H §6 RX1-K7 |
| K-Q8 | the names accepted as listed | here |

The questions as asked:

- **K-Q1 — which break, when two watches confirm.** (a) The EARLIEST CONFIRMATION wins (ties → the earlier anchor) —
  flip-free: the as-of answer only ever adds a break; (b) the first ANCHOR whose pattern confirms (PLAN_H's prototype
  wording) — then the bound can move later: A0 closes past with a pattern confirming at A0+5, A0+1 closes past with one
  confirming at A0+2 → (a) the break is A0+1 from A0+2 on; (b) it is A0+1 at A0+2 … A0+4, then A0 from A0+5. Never
  binds on the window (sub 7: only 4188 confirms; sub 4: 3588 first).
- **K-Q2 — the reason column.** One column `ic_ref_reason`, the price side reported first when both fail — or two
  boolean columns.
- **K-Q3 — save after A.** `/commit-save` after A (S_K_A) as B's baseline (as PLAN_H R4).
- **K-Q4 — the reversal event key.** `STATE_CHANGED(to=reversal)` gains `meta["pattern_anchor_idx"]` — an event-contract
  addition, format-only; the alternative (the `REVERSAL_CANDIDATE` at the apply) is not robust (a later anchor's winner
  can apply with no candidate there — `compute_main_cross_fib_limit`'s docstring).
- **K-Q5 — equal reference prices** (§9 F01 / C11). The two references can carry the SAME price, and their bounds
  differ (the CTS: the pattern that formed the fib's starting cycle; the structure: the reversal / watch break). Equal
  5-decimal prices are common on this window (H1 CTS_1 761 = sub 3's CTS_2 3047 = 0.57400; H1 BOS 591 = sub 2's BOS_0
  = 0.5736), but no record has both references equal. (a) the previous cycle's CTS (its pattern broke that price first;
  the stricter bound); (b) the structure reference.
- **K-Q6 — how the watch sees the candles after `t`** (§9 C02, MAJOR). MS detects a pattern at its anchor with the
  candles it can see (its look-ahead) and acts at the apply; `detect_best_for_anchor` checks `continuous` FIRST, and
  `continuous` needs A+2. (a) FULL-FRAME detection (up to MS's effective end), a break counting once its confirmation ≤
  `t` — MS's own convention; the as-of answer is the final answer filtered by time, so the bound never moves (RX1-E's
  "cannot flip" holds); (b) the frame CUT at `t` (a live view) — at A+1 a 2-candle pattern confirmed at A+1 is seen,
  while the full frame says `continuous` at A+2; the reviewer constructed a case where the break's ANCHOR moves
  earlier at A+2, so a candidate passing at A+1 is cut at A+2. Measured: 14 anchors on the S_S6 M15 frame (no
  threshold) where the two differ; 0 on the plan's real thresholds (subs 4 / 7 equal the final answer from 3589 /
  4189).
- **K-Q7 — when a reversal-started structure's bound is known** (§9 C08). The reversal pattern is applied as "known
  at the reversal moment", yet the structure's MS run refreshes before it (its retroactive cycles: H1 sid 1 cycle 1 at
  748–761, the reversal at 902; sub 1 cycle 1 at 1894, sub 0 reverses at 1940), while `P` and the reference price are
  applied from the run's start. (a) Known for the WHOLE run (the structure only exists from its reversal; like `P`);
  (b) only from the reversal moment (a candidate in [anchor, moment) passes in-flight but is cut in the final). No
  case on the window.
- **K-Q8 — the names, for one yes** (§9 F04). Fib meta `ic_ref_price`, `ic_ref_source` ∈ {`prev_cycle_cts`,
  `prev_structure_bos`, `cross_limit_anchor`}, `ic_ref_break_pattern_anchor_idx`, `ic_ref_break_confirmed_idx`; CSV
  column `ic_ref_reason` ∈ {`ic_past_reference`, `ic_not_before_break`}; `CrossFibLimit.ref_anchor_price`,
  `.break_pattern_anchor_idx`, `.break_confirmed_idx`, `.localized(offset)`; `ICReference`;
  `FibTracker.ic_reference` / `.stamp_ic_references`; `patterns/cross_limit_watch.py` `find_cross_limit_break`; the
  event key `STATE_CHANGED.meta["pattern_anchor_idx"]`; the concept names "IC reference" and "cross-limit breakout
  watch".

## 9. Cold review (2026-10-09) — 2 read-only lenses, fresh agents, main checkout: 526,601 subagent tokens

Fidelity 182,621 (every RX1 row built as decided; the user's three H1 hypotheticals reproduce — cross 0–2: 0.5736,
bound 897; cross 1–2: 0.57902, bound 747; single cycle 2: 0.57400, bound 897; all 26 records + sub 3 cyc 3 match §5);
code paths 343,980 (coordinates, the shift lists, A13, `_cross_allowed`, the predictions check out). Claims re-checked
by me before filing (✓).

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| C01 ✓ | MAJOR | MS calling `tracker.ic_reference` itself sits OUTSIDE `poi_inner_for_fib`'s re-raising `try` — a `ValueError` / `IndexError` there would reach `_build_geometry`'s narrow `except` (:1159) and drop the whole sub (LANDMINES "The POI Resolver Fails Loudly") | A4 / A14: the reference is built inside `poi_inner_for_fib`'s `try`; a test (§6) |
| C02 | MAJOR | The as-of watch with the frame cut at `t` is not MS's detection convention; the break's anchor can move (constructed); 14 differing anchors on the frame, 0 on the real thresholds | K-Q6 (recommend (a)); a pinning test either way |
| C03 ✓ | MAJOR | The finalize stamp: `_finalize_lifecycle_fields` takes no frame; `_cross_allowed` raises for a structure with no limit entry; `test_plan_i_s2…:541` pins the limit keys LAST | A11: a separate `stamp_ic_references(df, lifecycle_cap)`; the `M` test without `_cross_allowed`; the 4 keys before the limit keys |
| C04 | MAJOR | Geometry-side mutants are invisible to every comparison (A has the filter off; B == V by construction; the census sees picks; the window's reversal bounds cut no pick) | §3.5 / §6: spy tests on the geometry's limit and on `evaluated_at` at both refresh sites |
| C05 ✓ | MAJOR | The fixture migration was missing from the audit (~40 `CrossFibLimit` constructions; `make_cts_established` omits `pattern_anchor_idx` by default, 59 call lines; reversal `STATE_CHANGED` fixtures; 3-positional spies; `_Tracker` fakes) | A20; §7 resized |
| C06 | MINOR | "The refresh moment" undefined — `_refresh_poi_inners_for_cycle()` takes no argument; its callers sit at `apply_idx` / the re-stepped `i` | A14: the last fed event's `ef.event_moment`, from both sites |
| C07 | MINOR | The watch must get `self._resolver_df()` (L5), and loop A only over `[start, end_idx]` | A14, §3.4 |
| C08 | MINOR | A reversal bound "known at the reversal moment" vs refreshes of the structure's run before it, while `P` applies from the start | K-Q7 (recommend (a)) |
| C09 | MINOR | `ef.pattern_anchor_idx` raises on `STATE_CHANGED` | A19 |
| C10 | MINOR | The index-key guards: the inverse guard needs the stamp as a `meta=` literal (or `_NOT_MIRRORED`); internal `*_idx` dict keys flagged | A18 |
| C11 / F01 | MINOR | Equal reference prices — undefined, different bounds; the census keeps CTS by list order | K-Q5 |
| C12 / F02 / F05 | MINOR | The watch's `start_idx` (BOS_0 vs a sub's lifecycle start — sub 4: 3621 > its break 3588 ✓) and its retrospective end (the data edge, past the lifecycle cap) | §3.4: BOS_0 (the W1 example), the end capped at `lifecycle_cap`; pinned |
| C13 | MINOR | `ic_reference` with no fib | A4: None returns before the call |
| C14 | NIT | The `CTS_CONFIRMED` price is already recorded (`_cts_by_cycle`, :1070) | reused (§1, A11) |
| C15 | NIT | "Confirm by A+5" is the detector's own maximum — an equivalent mutant | §3.4, §6 |
| C16 | NIT | A16 located in `render_sub_projection` (:1288), `project_to_window` forwards | A16 |
| C17 | NIT | §1.4: `continuous`'s first candle can be normal / pinbar — the argument holds via the candle direction | §1.4 reworded |
| C18 / F07 ✓ | NIT | 26 distinct POI records (32 lens rows), not 27 | §5; PLAN_H RX1 corrected |
| F03 | MINOR | K-Q1 asked only about ties; the ordering itself (earliest confirmation vs first anchor) was hidden | K-Q1 rewritten with both readings + a case |
| F04 | MINOR | New names adopted without the user's yes | K-Q8 |
| F06 | NIT | The break keys are absent for subs 5 / 6 too | §5 |
| F08 | NIT | When the `prev_sub_bos_threshold` anchor is read | §3.3: in the same read as `P` |
| F09 | NIT | A13's "revived patternless reversal passes" is an assumption | A13 marked |
| F10 | NIT | The hover lacks the "reference, no known break" case | A17: `break: none` |
| F11 | NIT | The prototype's frame is rebuilt from the chart, not the engine slice | §3.4 caveat; landing A's check |

## 10. Change log

- 2026-10-09 — drafted from the code audit (HEAD `2a013d4`) + the decided-rule census.
- 2026-10-09 — cold-reviewed (§9, 526,601 tokens): every finding folded (A4 / A11 / A14 / A16 rewritten, A19 / A20
  added; §3.3–§3.6; §5 counts; §6 tests; §7 resized); K-Q1 rewritten, K-Q5 … K-Q8 added.
