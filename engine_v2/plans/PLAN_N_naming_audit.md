# Plan N — The Naming-Standard audit of the never-audited index keys (Week 9 Part 1, before register §D.1 item 3)

**Status (2026-10-12, session 3): BUILT, `/compare`d (= the prediction exactly, §11 / §13), LANDING-REVIEWED (240,968 tok;
0 BLOCKER, 3 MAJOR, 5 MINOR — folded, §12; its hover finding split out as a ready follow-up), mutants 32 / 32 + 3 / 3;
suite 1708; the user's go: "go ahead once review is in" → COMMITTED `1b2f24c` + SAVED S_N `20261012_165925_1b2f24c`
(save `81a0d46`, trunk `7869a24`) = `LATEST_<branch>` until S_N_hover. PLAN_N COMPLETE; the hover follow-up LANDED
2026-10-12 (option B, `9f9406d`, S_N_hover `20261012_194022_9f9406d` — §12 finding 1).** Living document: every
question, answer and decision is recorded here as it happens. Register line: `IDEA_PARKING_LOT.md` §A
"Naming-Standard audit of never-audited index keys" (parked since Plan E; raised at item 3's start, as the register said).

**Canonical rule:** `engine_v2/GLOSSARY.md` "Naming Standard" — every name holding a candle index says WHICH kind of
candle it is (a moment, a pattern anchor, a market-structure anchor, an extreme). **Migration rules:** `LANDMINES.md`
"Event Contract Rules" — rule 2 (never remove an event meta field), rule 3 (a rename / meaning change only as ONE atomic
migration commit: every reader, writer, fixture, registry and current doc in the same commit, `.get` → `meta[key]`,
no alias, the key on every event of the type, the `/compare` prediction in the commit message).
**Baseline:** S_M2 `artifacts/commits/week9-entries-decisions/20261012_023137_10e4c6f` (source `10e4c6f`, save
`3ebe857`, trunk `475bbf9`) = `LATEST_<branch>`; HEAD `dfc2b60` (docs only since). `run.log` in the repo root is the
S_M2 replay's (fetch gate: `Fetched 4228 M15 candles for NZD_USD in 5 chunks`).
**Tools (`plan_n_inputs/`, read-only):** `audit_census.py <save>` — the three keys' census + KL / POI zone sizes (§2);
`pullback_range_candles.py` — the M15 candles of the first pullback-created range (N5's example); `cmp_plan_n.py BASE
CUR_DEBUG` (written at the build — §4: applies the declared transform to the base CSVs and requires text equality).
**Followed by:** register §D.1 item 3, zone size adjustments (the user brings the spec) — this plan lands first so item
3's `/compare` carries no rename diffs and nothing reads a two-kind key. (Item 3 MOVED 2026-10-27 to Week 9
Parts 2 / 3 - the size adjustment goes into the entry price and stop loss, not the zones; register §D.3.)

---

## 1. Decisions (2026-10-12, session 3)

| ID | Decision | The user's answer (verbatim) |
|---|---|---|
| N0 | **When:** the whole audit (all four parts) lands FIRST, as its own small plan + one atomic commit, before item 3 | "A: Full audit first (Recommended)" — over "B: bounds_steps only first", "C: Stay parked, start item 3", "D: After item 3" |
| N1 | **KL `bounds_steps[*].start_idx`:** split the key by step kind — the INIT step carries `base_idx` (the base pattern's first candle, = the zone's `meta["base_idx"]` — a location), an expansion step carries `expanded_idx` (the candle at which the threshold update grew the zone — a moment; the zone-meta past-participle spelling, like `confirmed_idx` / `reconfirmed_idx`). The renderers and the M15 mirror read the step through ONE helper | "1a: base_idx / expanded_idx (Recommended)" — over 1b (`expanded_at`, the event-meta spelling — listed in text), "1c: drop INIT idx", "1d: segment_start_idx role", "1e: keep start_idx" |
| N2 | **`RANGE_STARTED`:** `start_idx` → `range_anchor_idx` (the range's first candle — a market-structure anchor; the code's own words: "Activate range anchored at i"; frees `start_idx`, which everywhere else is a REAL-TIME lifecycle field); `confirm_idx` → `confirmed_at` (the moment, `== ev.idx`, the same key and meaning as on the CTS / BOS events) | "range_anchor_idx + confirmed_at (Recommended)" — over `range_start_anchor_idx`, `range_confirmed_idx`, keep both |
| N3 | **`STATE_CHANGED` `effective_idx`:** `== ev.idx` on EVERY row (always a moment); the 56 `range_confirmed` rows stop carrying the range start there — it stays on the paired `RANGE_STARTED` (same candle, same structure) as `range_anchor_idx`. A rule-3 meaning change on 56 rows | "3a: always ev.idx (Recommended)" — over "3b: 3a + range_anchor_idx" (on all 453 rows, None on 397), "3c: delete effective_idx" (needs a rule-2 exception), "3d: keep" |
| N4 | **One test, two names:** "collapsed" everywhere — `ParentTables.degenerate` / `is_degenerate` → `collapsed` / `is_collapsed`, the unresolved reason `degenerate_parent_cycle` → `collapsed_parent_cycle` (the `*_M15_unresolved_triggers.csv` value, `run.log`), every current doc. "Degenerate" stays where it names a DIFFERENT thing (§3.4) | "4a: collapsed everywhere (Recommended)" — over "4b: degenerate everywhere" (the user's own PLAN_H word: "we do not need to chart any degenerate Fibs"; 74 code lines incl. WVMI's exported `cycle_collapsed`; clashes with "degenerate probe window"), "4c: keep both, glossary" |
| N5 | **`range_anchor_idx` on the pullback- / proximity-created paths = MS's own range start** (`st.range_start_idx` = the CTS anchor, = the event's `cts_anchor_idx`; MARKET_STRUCTURE_SPEC: "range_start_idx = CTS idx"), **AND the H1 chart's range box starts there** (it started at `ev.idx` — the fallback when the key was absent). Example — confluence sub 0 cyc 0, M15 1795 (2025-12-11 14:45, the pullback apply = `ev.idx`, low 0.58162 = the range's lo), CTS anchor 1794 (14:30, high 0.58314 = the range's hi = `cts_price`): `range_anchor_idx` 1794; the box would start at 1794. 0 such ranges on H1 in this window (and the M15 charts draw no range boxes), so no output moves here | "(c) MS's value + fix chart" — over "(b) MS's value, chart kept" (my recommendation: the chart keeps `ev.idx` explicitly, the gap a register item) and "(a) None" (the default first accepted; the cold review's MAJOR 3: None writes the chart's divergence into the event contract, and a later fix would be a second migration of the same cells). The user accepted that the chart rule has no rendered example on this window |

**Defaults stated in text, not separately asked:**
- Out of scope (§7): the MS dataframe columns `range_start_idx` / `range_confirm_idx` and the range label's
  `is_range_confirm_idx` (same spelling smell; not event / zone meta). (Corrected by the landing review: the H1 range
  hover's `range_start_idx=` label is NOT that df column — it shows `_range_left_idx(ev)`, the range's anchor, so N5
  changes it on pullback / proximity ranges, like the box.)

## 2. Measured — the status quo on S_M2 (session 3; `plan_n_inputs/audit_census.py`)

**N1 — KL steps.** 39 KL zones (H1 10, confluence 21, counter 8). Every INIT step's `start_idx == base_idx` (39 / 39).
31 expansion steps, each `start_idx` = its threshold-update event's `ev.idx`: H1 13 on 3 zones (all
`CTS_THRESHOLD_UPDATED`), confluence 18 on 4 zones (all `BOS_THRESHOLD_UPDATED`), counter 0. Steps are appended in
index order (39 / 39 sorted, no ties). **Example** — H1 CTS sell zone, sid 0 cycle 0: INIT `start_idx` 430 (2025-12-10
20:00, the base inside bar = the CTS anchor `anchor_idx` 430), `confirmed_idx` 439 (12-11 05:00), then 9 expansions
447 (12-11 13:00; threshold 0.58246 → 0.58267 — candle 447's high is 0.58267: the moment AND the new extreme), 448,
634 … 639, 650 (12-24 00:00); 11.4 → 31.6 pips. **5 expansion steps precede the zone's (clamped) `confirmed_idx`:** H1
sid 1 cycles 0 / 1 (741, 742, 746, 880 < 902 — collapsed cycles) and confluence sub 6 cyc 0 at anchor 3760 (its
`BOS_THRESHOLD_UPDATED` 3806, `probe_no_break`, < 3819 = sub 6's real-time `start_idx`, the lifecycle floor that
clamped the zone's first-active) — consistent with "the expansion is a moment of the STRUCTURE, the zone's
first-active is clamped later"; nothing to change here, but item 3 must not assume "an expansion happens while the
zone is active".
**Readers** (all draw; none reads the step as a moment): `charting/_zone_render.build_stepped_outline_xy` (sort +
segment x), `charting/export_plotly.py` (~1782–1917: sort, segment start, next step's start, and the confirm-line loop
~1911 / ~1917), `charting/export_m15_chart.py` (~1265–1393 and ~2263–2386: the same five reads for each of the two KL
renderers); the M15 mirror `multitf/entity_df_mutation.py` (~304–316, shifts every step's `start_idx`). **Writers:**
`zones/kl_zones_v1.py` ~851 (expansion), ~969 (INIT); and the three renderer FALLBACK step builders for a zone with no
steps (`export_plotly.py` ~1785–1791, `export_m15_chart.py` ~1266–1268 and ~2266–2270 — `{"start_idx": base_idx, …,
"event": "FALLBACK"}`; unreachable today: every zone has an INIT step).

**N2 — `RANGE_STARTED`.** 65 rows: 56 from the range-label path (H1 10, confluence 33, counter 13), 9 pullback-created
(confluence 7, counter 2), 0 proximity-created. Label path: `confirm_idx == ev.idx` on 56 / 56; `start_idx` = the
range's first candle, `confirm_idx − start_idx` ∈ {2: 39, 3: 9, 4: 2, 5: 6}. **Example** — H1 `RANGE_STARTED` at 117
(2025-11-21 19:00): `start_idx` 115 (17:00 — the breakout candle that established cycle 0, also `cts_anchor_idx`),
`confirm_idx` 117; its back-filled `RANGE_UPDATED` 116 / 117 are emitted AFTER it in the stream. 8 H1 label ranges
have a `RANGE_UPDATED` strictly between start and confirm (why the chart sorts a `RANGE_STARTED` by its start). The 9
pullback rows carry neither key; `pullback_apply_idx == ev.idx` on 9 / 9; MS's state sets `st.range_start_idx =
st.cts.idx` there (`market_structure.py` ~1212; proximity ~1978) — the event's `cts_anchor_idx`. **The reset ordering
(N5):** for each of the 9 pullback ranges, the latest same-structure `RANGE_RESET` before it is at or before its CTS
anchor (2 ties: 2842, 4157 — a tie sorts the reset first); none falls between the anchor and `ev.idx`. **Readers:**
`charting/export_plotly.py` 6 sites (~737 and ~885 the sort keys; ~764, ~905, ~938, ~976 the box / hover-line left
edge; all `ev.meta.get("start_idx", idx)`); the mirror registry `_EVENT_META_IDX_KEYS`;
`tests/test_ms_cts_update_no_regress.py` ~215, `test_ms_bounded_equals_truncated.py` ~366,
`test_plan_i_s5_ms_reads_the_fib.py` ~169. The M15 charts draw no ranges.

**N3 — `STATE_CHANGED`.** 453 rows (H1 63, confluence 296, counter 94); `effective_idx == ev.idx` on 397; on exactly
the 56 `reason="range_confirmed"` rows it is the range start (the example: `ev.idx` 117, `effective_idx` 115). The ONE
override is `market_structure.py` ~1120 (`_set_state(RANGE, confirm_idx, meta={..., "effective_idx": i})`); `_set_state`
defaults it with `m.setdefault("effective_idx", i)`; the other 10 `_set_state` calls and the 2 direct test calls
(`test_ms_reversal_terminal.py` ~311 / ~318) pass none. No production reader (the mirror shifts it).

**N4 — collapsed / degenerate.** The same test on two sides (PLAN_H §8.9 F1c): zones / charts "collapsed" (clamped
`confirmed_idx >= end_idx`; `_zone_render.is_collapsed_cycle_zone`, WVMI `cycle_collapsed`), `ParentTables`
"degenerate" (`floor_m15 >= end_m15`, M15 coordinates). On H1 the same two cycles (1,0) / (1,1). S_M2 output carrying
the word: `*_M15_unresolved_triggers.csv` 4 `reason` cells; `run.log` 15 lines (7 `[parent_tables]` — 5 per-cycle
`degenerate=` + 2 `WARNING … degenerate parent cycle`; 4 `[sweep] UNRESOLVED (skipping) reason=degenerate_parent_cycle`;
4 `[pool] unresolved … reason=degenerate_parent_cycle`).

**Zone sizes (item 3's status quo, recorded here because the census measured them):** KL unchanged since Week 8 —
H1 6.4–31.6 pips (median 13.2; BOS 6.4–13.4, CTS 13.0–31.6), confluence 2.0–14.0 (median 5.0), counter 2.2–15.9 (median
3.9); POI (one per cycle since PLAN_H) H1 5 zones 3.3–11.3 (median 5.0), confluence 19 2.2–5.6 (4.1), counter 7 2.2–5.1
(4.1); H1 candle range median 7.0 (p25 5.2 / p75 9.5 / p90 12.8), ATR14 median 7.8.

## 3. The design

### 3.1 N1 — KL steps (`zones/kl_zones_v1.py` owns the format)
- **Writers.** INIT (~969): `{"base_idx": int(base_idx), "top", "bottom", "event": "INIT"}`. Expansion (~851):
  `{"expanded_idx": ef.event_moment(ev), "top", "bottom", "event", "price"}` — the moment read through the role
  accessor (only `CTS_THRESHOLD_UPDATED` / `BOS_THRESHOLD_UPDATED` reach the branch, ~788 / ~803; `event_moment`
  returns `int(ev.idx)` for both — same value, no raw `ev.idx` read of a role). Key position unchanged (first in the
  step dict), so each CSV cell differs only in the key name.
- **Two helpers, next to the writers** (the producer owns the format, as `event_fields` owns event roles):
  `bounds_step_idx_key(step) -> str` = `"base_idx"` if `step["event"] == "INIT"` else `"expanded_idx"`;
  `bounds_step_draw_idx(step) -> int` = `int(step[bounds_step_idx_key(step)])` — "the candle a step's segment is drawn
  from", a chart ROLE over two honest keys. Strict reads (no `.get`, no `-1` default — rule 3; every step has its key).
- **Readers.** Every renderer read in §2 (sort key, segment start, the next step's start, the confirm-line loop) reads
  `bounds_step_draw_idx(s)`. The mirror shifts `new_step[bounds_step_idx_key(step)]`. `charting/` and `multitf/`
  import the helpers from `zones.kl_zones_v1` (no cycle: its import closure holds no `charting` / `multitf` module —
  the review checked `sys.modules`).
- **The three FALLBACK builders** become INIT-shaped: `{"base_idx": <the same value>, "top", "bottom", "event":
  "INIT"}` — the same drawn x as today (they used `base_idx` already), and the strict helper can read them.
- **Docstring / comment touch-ups** at the two writers, `_zone_render` ("segment begins at base anchor candle"),
  `entity_df_mutation.py` ~111 / ~304.

### 3.2 N2 + N5 — `RANGE_STARTED` (`structure/market_structure.py`) and the H1 range chart
- **Every path emits `"range_anchor_idx": int(st.range_start_idx)`** (MS's own range start, set just above each emit:
  the label path `i` ~1098, the pullback path `st.cts.idx` ~1212, the proximity path `int(st.cts.idx)` ~1978) and
  `"confirmed_at": <the value passed as idx=>` (the label path `confirm_idx`, the pullback path `i`, the proximity
  path `int(candle_idx)`). Label path: in place of `"start_idx"` / `"confirm_idx"` (same positions). Pullback /
  proximity paths: the FIRST two keys.
- **The chart** (`export_plotly.py`, the box builder ~700–853 and the hover-line builder ~856–1000). Two module-level
  helpers used by both builders:
  - `_range_left_idx(ev)` = `int(ev.meta["range_anchor_idx"])` — the box / hover line starts at the range's anchor on
    every path (N5: on the pullback / proximity paths this moves the left edge from `ev.idx` to the CTS anchor).
  - `_range_event_sort_key` — `RANGE_STARTED` sorts at `max(range_anchor_idx, R)`, `R` = the idx of the latest
    same-structure `RANGE_RESET` emitted BEFORE it in the stream with `R <= ev.idx` (none → the anchor); every other
    range event at its `ev.idx`; ties: RESET 0, STARTED 1, UPDATED 2 (unchanged). **Why the `R` guard (new, from the
    N5 analysis):** a pullback range's anchor is the CTS anchor, which can PRECEDE the breakout candle that reset the
    previous range (the anchor-lag cycles — 3 of 34 `CTS_ESTABLISHED` on the reference window); sorted at the bare
    anchor, the old range's reset would be processed after the new start and delete the NEW box. With the guard the
    reset closes the old box at `R` first, then the new box opens and is drawn from its anchor (an overlap with the
    old box's tail is the data's truth: MS says the new range starts at the anchor). 0 instances on S_M2 (§2); the
    label path never has a reset between its anchor and confirm (0 on S_M2), so its order is unchanged.
- **Registry** (`multitf/entity_df_mutation._EVENT_META_IDX_KEYS`): drop `"start_idx"` and `"confirm_idx"`, add
  `"range_anchor_idx"`; `confirmed_at` is already listed (its comment gains RANGE_STARTED).
- No `confirmed_at` reader is type-agnostic (checked twice: `event_fields.event_moment` raises on an unlisted type;
  `conftest.EVENT_IDX_IS` / `_ANCHOR_KEY` validate by type; `parent_tables`, `structure_lifecycle`, `kl_zones_v1`,
  `wave_candles`, `poi_zones`, `unified_probe` filter by type first).
- **Spec (rule 4):** MARKET_STRUCTURE_SPEC gains a `RANGE_STARTED` meta listing (it has none); CHARTING_SPEC the range
  box's left edge + sort rule.

### 3.3 N3 — `STATE_CHANGED` (`structure/market_structure.py`)
- ~1120: `self._set_state(MarketState.RANGE, confirm_idx, meta={"reason": "range_confirmed"})`.
- `_set_state`: `assert "effective_idx" not in m` then `m["effective_idx"] = i` — the key can no longer be overridden,
  so it cannot become two kinds again (a tripwire, like the reversal-terminal assert beside it). Key order on the 56
  rows unchanged (`reason`, then `effective_idx`).

### 3.4 N4 — "collapsed" (`multitf/`)
- `parent_tables.py`: the field `degenerate` → `collapsed`, `is_degenerate` → `is_collapsed`, the per-cycle log
  `degenerate=` → `collapsed=`, the WARNING `degenerate parent cycle` → `collapsed parent cycle`, the module docstring
  formula (~21) and ~108.
- `lifecycle_sweep.py` (~282–288): `is_collapsed`, reason `"collapsed_parent_cycle"`; the docstring table (~20) and the
  documented grep keywords (~46).
- `sub_structure_pool.py` (~86 the reason set, ~248 the docstring).
- `charting/_zone_render.py` ~46 (docstring "degenerate parent cycles").
- **Kept (a different thing):** "degenerate probe window" / "degenerate sibling-CTS probe window" / "degenerate
  reversal probe window" (probe input ≥ probe end — `entity_df_mutation.py`, `structure_engine.py`; LANDMINES ~1079),
  the stepped outline's "degenerate to a plain rectangle", the imbalance "degenerate gap", ARCHITECTURE's "degenerate
  single-entry history", "degenerate runs" in MS, POI's "degenerate range", poi_lifecycle's "degenerate same-idx flip",
  GOTCHAS' zero-width "degenerate sub zone", LANDMINES' "degenerate same-dt".
- **The post-replay grep** (WORKFLOWS.md, `/compare` §2b, `lifecycle_sweep` docstring) gains `collapsed`; `degenerate`
  stays in it (the probe-window warnings still say it). The grep count on the window goes 21 → 22: `collapsed` also
  matches `[chart][fib] rendered 3/5 fib states (... not collapsed / disappeared)`.

## 4. The prediction (`/compare` vs S_M2) — ONE landing

**Figures:** all three charts byte-identical (`plan_m_inputs/fig_hash.py` — every trace and the layout): no hover
prints a meta key name; the H1 range boxes / hover lines read the same anchors on the 10 label ranges and H1 has no
pullback / proximity range (N5's left-edge move has nothing to move here); no reset falls between an anchor and its
`ev.idx`. **CSVs:** 20 of 27 md5-equal; the 7 that change, cell by cell (values unchanged except N3; checked by
`plan_n_inputs/cmp_plan_n.py`, which applies exactly this transform to the S_M2 CSVs and requires text equality):

| CSV | Cells | Change |
|---|---|---|
| `_kl_zones.csv` (H1) | 10 `meta` | every INIT step `'start_idx': N` → `'base_idx': N`; the 13 expansion steps `'start_idx': M` → `'expanded_idx': M` |
| `_M15_confluence_kl_zones.csv` | 21 `meta` | the same (18 expansion steps) |
| `_M15_counter_kl_zones.csv` | 8 `meta` | the same (INIT only) |
| `_structure_events.csv` (H1) | 20 `meta` | 10 `RANGE_STARTED`: `'start_idx': a, 'confirm_idx': b` → `'range_anchor_idx': a, 'confirmed_at': b`; 10 `STATE_CHANGED` (`range_confirmed`): `'effective_idx': a` → `b` (= the row's `idx`) |
| `_M15_confluence_structure_events.csv` | 73 `meta` | 33 label-path `RANGE_STARTED` (as H1); 7 pullback `RANGE_STARTED` gain `'range_anchor_idx': <its cts_anchor_idx>, 'confirmed_at': <its idx>` as the first two keys; 33 `STATE_CHANGED` (as H1) |
| `_M15_counter_structure_events.csv` | 28 `meta` | 13 + 2 + 13, as confluence |
| `_M15_unresolved_triggers.csv` | 4 `reason` | `degenerate_parent_cycle` → `collapsed_parent_cycle` |

Total 164 cells; no row added, removed or reordered; no other column. **`run.log`:** (1) the 15 lines of §2 N4 change
their word (`degenerate=` → `collapsed=`, `degenerate parent cycle` → `collapsed parent cycle`,
`reason=degenerate_parent_cycle` → `reason=collapsed_parent_cycle`); (2) the 5 `market_structure.py:2472: FutureWarning`
lines (run.log 1, 3, 5, 3881, 7765) shift their LINE NUMBER by the net lines the edits add above 2472 — measured at
the build: **2472 → 2480** (+8; the pytest warnings show it) — the text after the number equal; every other line equal up to timings. The post-replay grep: 22 lines (21 + the `[chart][fib] … not
collapsed` line). The fetch gate first: `Fetched 4228 M15 candles for NZD_USD in 5 chunks`. **Save-format boundary:**
saves before this landing carry the old keys and reason value.

**Why one `/compare` is one cause:** the four parts touch disjoint cells, each predicted exactly; all are the same
declared migration (names / one meaning on 56 cells / MS's value on 9 new cells); N5's chart change moves nothing on
this window, so the figures must not move at all.

## 5. Tests, pins, mutants

**Moved with the migration (rule 3: fixtures too):** `tests/test_event_meta_idx_keys.py` (the registry list ~245, the
nested-key map `NESTED_IDX_LISTS` → per-step via `bounds_step_idx_key`, the static map ~614 — `_NOT_MIRRORED
["kl_zones_v1"]` becomes `{"expanded_idx"}` (`base_idx` is already a zone key) — and the fixture ~494 / ~524, whose INIT
step gains `"event": "INIT"`); `tests/test_meta_key_renames.py` (the KL expansion pins ~241–278 read the new keys);
`tests/test_ms_cts_update_no_regress.py` ~215; `tests/test_ms_bounded_equals_truncated.py` ~366;
`tests/test_plan_i_s5_ms_reads_the_fib.py` ~169; the "degenerate" pins in `test_parent_tables.py`,
`test_lifecycle_sweep_predicted_table.py`, `test_lifecycle_sweep_unit.py`, `test_sub_tables_export.py`,
`test_sub_structure_pool.py` (the parent-cycle sense only; test names too). `test_plan_m_landing_2.py` ~339 /
`test_lifecycle_sweep_predicted_table.py` ~121 carry `effective_idx` 902 on a reversal row whose `idx` is 902 —
unchanged. The full list is whatever `grep` finds at build time; this list is the minimum.

**New pins (each must FAIL on its named mutant — "a pin whose two outcomes the domain makes equal cannot fail"):**
- **P1** (N3) a real run with a `range_confirmed` row: `effective_idx == ev.idx` on EVERY `STATE_CHANGED`. Mutant:
  restore the `effective_idx: i` override (start ≠ confirm on every label range: gap ≥ 2).
- **P2** (N3) `_set_state(..., meta={"effective_idx": x})` raises. Mutant: drop the assert.
- **P3** (N2 / N5) every `RANGE_STARTED` emit literal carries `range_anchor_idx` and `confirmed_at` (static, like the
  existing `cts_anchor_idx` pin), and real runs covering all THREE paths (the `test_meta_key_renames` fixture's
  label + pullback ranges; the proximity range of `test_ms_reversal_terminal.py` ~173): label `range_anchor_idx` <
  `confirmed_at == ev.idx`; pullback / proximity `range_anchor_idx == cts_anchor_idx` (≠ `ev.idx` — the forbidden
  input), `confirmed_at == ev.idx`. Mutants: swap the two label values; set the pullback anchor to `i`; the proximity
  anchor to `candle_idx`.
- **P4** (N2 / N5 chart, the box AND the hover-line builders): (a) a label range with a `RANGE_UPDATED` strictly
  between its anchor and confirm (the S_M2 shape: STARTED @117 anchor 115, UPDATED @116): the update splits the box
  at 116 — mutant: the sort key uses `ev.idx`; (b) a pullback range: its box starts at `range_anchor_idx` (≠ `ev.idx`)
  — mutant: `_range_left_idx` returns `ev.idx`; (c) the reset guard: a same-structure `RESET` at `R` with anchor `A < R
  <= ev.idx`: the old box ends at `R` and the new box exists, drawn from `A` — mutant: sort at the bare anchor.
- **P5** (N1) a real run with an expansion: INIT carries exactly `base_idx` (== the zone's `meta["base_idx"]`, ≠ its
  `confirmed_idx` — the forbidden input: a base before the moment), each expansion exactly `expanded_idx` == its update
  event's `ev.idx`; no step carries `start_idx`. Mutants: INIT writes `confirmed_idx`; expansion writes the zone's
  `anchor_idx`.
- **P6** (N1 helpers, one pin per CALL SITE): a golden render of a 2-step zone (INIT `base_idx` ≠ the expansion's
  `expanded_idx`, the confirm time inside the expansion step) through the stepped outline (`_zone_render`), the H1 KL
  renderer (segment starts + the confirm line's y-range) and each of the two M15 KL renderers (the same); the mirror
  shifts both step kinds. Per-site mutant: that site left on `s.get("start_idx", -1)` (the realistic miss); the mirror:
  shift only the INIT step. **Declared unpinnable:** the KL sort-key reads — steps are appended in index order (39 / 39
  sorted, no ties), so a sort mutant changes nothing (the sort stays as defence; noted in the code).
- **P7** (N4 + N1/N2, static) no production string carries `degenerate_parent_cycle`, `is_degenerate`, or a
  `.degenerate` attribute; no `RANGE_STARTED` emit literal / step literal carries `start_idx` / `confirm_idx`.
- **Real-data provenance:** any pin that embeds S_M2 rows also gets a provenance test against the save (mutant trees
  have no `artifacts/`).

**Mutants:** `plan_e_inputs/review_scripts/mutant_runner.py`, full suite, a SHORT `%TEMP%` dir; target = every mutant
above + the landing review's.

## 6. Docs at the landing (current docs; dated records stay as written)

- `KL_ZONES_SPEC.md` "Zone expansion" (the step keys, the helpers, the INIT / expansion kinds).
- `GLOSSARY.md`: the `bounds_steps` row; the `start_idx` / `confirm_idx` rows (the RANGE_STARTED sense moves to
  `range_anchor_idx` / `confirmed_at`); ~287 the ParentTables row (`degenerate` / `is_degenerate`); ~289 "degenerate
  parent cycle" → "collapsed parent cycle" + one "collapsed" entry naming both sides; the Naming Standard's status
  paragraph (PLAN_N closed the audit).
- `ARCHITECTURE.md`: the `STATE_CHANGED` row (~118), the "Not audited" paragraph (~120 → RANGE_STARTED audited), the
  parent-tables lines ~475–483.
- `MARKET_STRUCTURE_SPEC.md`: a NEW `RANGE_STARTED` meta listing (rule 4), incl. N5.
- `LANDMINES.md`: ~159 the sort-order row; ~1686 ("each loop must match exactly one producer-side key" → two keys,
  chosen by `event`) and the rule of thumb ~1738; ~1692 the shift table; ~853, ~2029–2030 ("DEGENERATE parent cycle" +
  its link), the section title ~2446 and its body ~2449–2490 (the title, its link here and GOTCHAS ~1819 renamed
  together), ~2105. **Kept as written:** ~1727–1736 (a dated "Concrete case (2026-05-18, fixed)" narrative).
- `GOTCHAS.md`: ~312–324 the range sort example; ~1818–1819.
- `CHARTING_SPEC.md`: ~115, ~138; the range box rule (N5: left edge = `range_anchor_idx`, the sort with the reset
  guard).
- `PART4_REFACTOR_SPEC.md` (current text only): 518, 742, 1390, 1420, 1533–1535, 2603, 2888, and §17.6 / §17.7 at
  3075, 3119, 3185, 3193–3211.
- Code comments: `export_plotly.py` ~723–736 (the sort comment), `entity_df_mutation.py` ~111 / ~304.
- `WORKFLOWS.md` ~81–97 (the grep, the expected lines, the count); the `/compare` skill (§2b grep + expected unresolved
  lines + the count; the S_N save-format boundary); `IDEA_PARKING_LOT.md` (the §A line closed); memory (`MEMORY.md`
  lines naming the keys, the pool ground-truth note, `project_week9_part1_inputs.md`).

## 7. Out of scope

- MS dataframe columns `range_start_idx` / `range_confirm_idx` (`market_structure.py` ~39) and the label column
  `is_range_confirm_idx` (`range_label.py`): not event / zone meta; left as they are.
- Every "degenerate" that names a different thing (§3.4).
- (The chart gap first planned as a register line is closed by N5 in this landing.)

## 8. Reviews and cost

- **Cold review of this plan:** DONE (§9) — one fresh read-only reviewer, no worktree, 269,131 tok, 87 tool calls.
- **Build:** solo. **`/compare`** against S_M2 with §4 as the written prediction.
- **Landing review:** one fresh read-only reviewer (est. 300–450K tok) + the mutant run.
- **Commit** only on the user's go; then `/commit-save` → S_N.

## 9. The cold review (2026-10-12, session 3) — DONE, every finding folded

Verified with no finding: every §4 count (the census re-run + direct CSV reads; 27 CSVs, the keys in exactly the 7
named); the charts carry none of the keys; no `confirmed_at` collision; N3's assert is safe; N1's `event_moment` valid
for every type reaching the branch; no import cycle.

| # | Sev | Finding | Folded |
|---|---|---|---|
| 1 | MAJOR | `run.log`: the 5 `market_structure.py:2472: FutureWarning` lines shift their line number (edits above 2472) | §4 (2); + the grep 21 → 22 (`collapsed` matches the fib-render line) — §3.4, §4, §6 |
| 2 | MAJOR | 3 renderer FALLBACK step builders write `"start_idx"` (`event: "FALLBACK"`) — the strict helper would KeyError | §2 writers, §3.1 (INIT-shaped) |
| 3 | MAJOR | `range_anchor_idx = None` on the pullback / proximity paths misstates MS's own range start (MARKET_STRUCTURE_SPEC ~270–272; `market_structure.py` ~1213 / ~1977) | asked → the user's N5 "(c) MS's value + fix chart"; the reset guard §3.2 |
| 4 | MINOR | `test_ms_cts_update_no_regress.py` ~215 reads `e.meta["start_idx"]` | §2, §5 |
| 5 | MINOR | doc sites missing (LANDMINES ~1686 / ~1738 / ~2029–2030 / title ~2446 + GOTCHAS ~1819 link; PART4 current-text sites; GLOSSARY ~287; `parent_tables.py` ~108; code comments; MS spec has no RANGE_STARTED listing); LANDMINES ~1727–1736 is dated — keep | §3.4, §6 |
| 6 | MINOR | P4 missed the two sort-key sites (a swap matters only with a `RANGE_UPDATED` between anchor and confirm — 8 H1 ranges, e.g. @117 / 115 / 116) | P4 (a) |
| 7 | MINOR | P6: the confirm-line reads unnamed; the KL sort keys unpinnable (steps sorted, no ties); the realistic per-site mutant is a site left on `s.get("start_idx", -1)` | P6 |
| 8 | MINOR | P3 skipped the proximity path (0 on S_M2; fixtures have one) | P3 |
| 9 | MINOR | no tool proves "values unchanged except N3" (`cmp_meta_rename.py` renames top-level keys only) | `plan_n_inputs/cmp_plan_n.py` (§4) |

## 10. The build — as built (2026-10-12, session 3)

**Code (one atomic migration, uncommitted until the user's go):**
- N1: `zones/kl_zones_v1.py` — the two helpers `bounds_step_idx_key` / `bounds_step_draw_idx` (module top), INIT
  writes `base_idx`, an expansion writes `expanded_idx = ef.event_moment(ev)`; `charting/_zone_render.py`
  (`build_stepped_outline_xy`: sort + step x + next step x), `charting/export_plotly.py` (the H1 KL renderer: sort,
  segment start, next step, the confirm-line loop's two reads; its fallback builder INIT-shaped),
  `charting/export_m15_chart.py` (both KL renderers, the same five reads each; both fallbacks INIT-shaped) read through
  the helper; `multitf/entity_df_mutation.py` shifts `new_step[bounds_step_idx_key(new_step)]` (strict).
- N2 / N5: `structure/market_structure.py` — all three `RANGE_STARTED` emits carry `range_anchor_idx =
  int(st.range_start_idx)` + `confirmed_at`; `charting/export_plotly.py` — `_range_left_idx` + `_range_events_chart_order`
  (module level; the reset guard), used by the box builder and the hover-line builder (its three replay passes);
  `_EVENT_META_IDX_KEYS`: `start_idx` / `confirm_idx` out, `range_anchor_idx` in.
- N3: the `range_confirmed` override removed; `_set_state` asserts no override and sets `effective_idx = i`.
- N4: `multitf/parent_tables.py` (`collapsed`, `is_collapsed`, the log + WARNING text), `lifecycle_sweep.py` (the check,
  the reason, the docstring grep), `sub_structure_pool.py` (the reason set), `_zone_render.py` (docstring).

**Tests:** the migration moved 10 files' fixtures / pins (`test_event_meta_idx_keys` — `NESTED_IDX_LISTS` now maps to a
key FUNCTION per item; `test_meta_key_renames`, `test_ms_cts_update_no_regress`, `test_ms_bounded_equals_truncated`,
`test_plan_i_s5_ms_reads_the_fib`; the parent-cycle "degenerate" in `test_parent_tables`,
`test_lifecycle_sweep_predicted_table`, `test_lifecycle_sweep_unit` (its "degenerate reversal probe window" line kept),
`test_sub_tables_export`, `test_sub_structure_pool`). New: `tests/test_plan_n_naming_audit.py` (17 tests, P1–P7; P4 and
P6 render the real H1 / M15 charts). Suite: **1707 passed + the OANDA smoke test deselected** (1690 + 17) at the
first replay; **1708** after the landing review's fold (+2 pins, −1 moved to the hover follow-up — 18 in the module).

**Declared unpinnable (no mutant listed):** the KL step sort keys (appended in index order); the three renderer
fallback step builders (unreachable — every zone has an INIT step); the first two replay passes of the H1 range
hover-line builder (`export_plotly.py` ~920–985: they build `hover_data` / `open_ranges` that nothing reads — dead
state, pre-existing; only the third pass, `range_by_candle`, reaches the figure). The mirror's both-kinds shift is
pinned by `test_event_meta_idx_keys` (`_synthetic_result` INIT + expansion, `_check_mirrored`).

**Docs moved in the same change:** KL_ZONES_SPEC "Zone expansion"; GLOSSARY (`bounds_steps`, `start_idx`, the new
`range_anchor_idx` row replacing `confirm_idx`, ParentTables, "collapsed parent cycle", "collapsed cycle", the Naming
Standard's PLAN_N status); ARCHITECTURE (`STATE_CHANGED` row, the RANGE_STARTED paragraph, the parent-tables / sweep
lines); MARKET_STRUCTURE_SPEC (a new "The `RANGE_STARTED` event" table); CHARTING_SPEC (the range box rule, the KL step
x, the collapsed wording); LANDMINES (the anchor-vs-moment row, the nested-key rule + table + rule of thumb, the
"Collapsed Parent Cycles" section + its link); GOTCHAS (the range sort order, the link); PART4 (current text, §17.7
renamed with a history note); WORKFLOWS + the `/compare` skill (the grep gains `collapsed`, 21 → 22 lines, the reason
/ WARNING names). **Kept as dated records:** LANDMINES ~853 / ~2109 and PART4 ~3469 / ~3491 (Plan C replay
measurements), LANDMINES ~1727–1736 (the 2026-05-18 narrative), the `/compare` skill ~647 (the Plan C chart review).

**Mutants:** `plan_n_inputs/mutants_n.py` — **32 / 32 KILLED** (`mutants_n_result.txt`): N3 ×2, N2 / N5 ×5, the range
chart ×5 (sort, left edge, reset guard, box site, hover pass 3), KL writers / helper / mirror ×4, every renderer read
×14, the collapsed reason / WARNING ×2.

## 11. `/compare` vs S_M2 (2026-10-12, session 3) — = the prediction exactly

Replay 49.2 s wall (fetch gate PASS: 4228 M15 candles in 5 chunks). `plan_n_inputs/cmp_plan_n.py`: **20 md5-equal, 7
transform-equal** — changed cells H1 KL 10, confluence KL 21, counter KL 8, H1 events 20, confluence events 73, counter
events 28, unresolved 4 = **164** (§4's table, cell for cell). Figures: `plan_m_inputs/fig_hash.py` — all three
fingerprints EQUAL to S_M2 (83 / 239, 282 / 276, 137 / 123). `run.log`: 9105 lines both; 29 changed lines = the 15
`degenerate` → `collapsed` lines + the 5 FutureWarnings `market_structure.py:2472` → `2480` + 9 timing lines; nothing
else (normalised diff: 0). The post-replay grep: 22 lines (21 + the `[chart][fib] … not collapsed` line). UNEXPECTED:
none.

## 12. The landing review (2026-10-12, session 3) — one fresh read-only reviewer, 240,968 tok; every finding folded

Tally 0 BLOCKER, 3 MAJOR, 5 MINOR. Verified OK: the label-path sort order unchanged (ties only within a type, the guard
never fires on the label path — a reset emitted before a label STARTED has `R < i`, `R == i` on the post-apply check,
where RESET sorts first); no inverted box (every close has a key ≥ the STARTED's ≥ its anchor); the `_set_state`
assert safe; the strict mirror read safe (KL zones come only from `derive_kl_zones_v1`); `st.range_start_idx` never None
at the 3 emits; the MS-spec table, the KL "5 of 31" and every current M15 RANGE_STARTED re-measured; no tolerant read
of a new key.

| # | Sev | Finding | Folded |
|---|---|---|---|
| 1 | MAJOR | N5 makes the hover builder's pass 3 wrong ACROSS structures: it snapshotted EVERY structure at an event's `ev.idx` while replaying a STARTED at its anchor — a pullback range anchored before another structure's reset froze that structure as open (its line vanished at the reset, came back at the moment) and started its own line early (the reviewer's probe: sid 0 reset 12, sid 1 pullback 14 anchored 8). Label ranges had the same exposure on their 2–5 candle window; 0 instances on S_M2 | BUILT, then SPLIT OUT: the per-structure snapshot fix changed the S_M2 H1 figure (§13) — the review's "0 instances" was wrong: the defect is PRE-EXISTING on a label range (sid 1 anchored 748 / confirmed 750 vs sid 0's update at 748). A second cause, so it is a ready-to-land follow-up (`plan_n_inputs/followup_hover_per_structure.md`: the code, the pin, the mutant, the measured 4-trace delta), a register §A line, a CHARTING_SPEC known-defect note and an `export_plotly` comment; this landing keeps the original pass 3. **→ LANDED 2026-10-12 (the next session), widened by the user to option B — the hover line = the box, candle for candle (snapshot at each event's replay candle; it also fixes a second, same-structure symptom: a label range's confirm candle showed its anchor-time state, 16 candles on S_N) — with the dead passes 1–2 deleted — committed `9f9406d`, saved S_N_hover `20261012_194022_9f9406d`: `followup_hover_per_structure.md` "The landing decision"** |
| 2 | MAJOR | 3 surviving mutants: LR1 the hover builder back on an idx sort (P4 checked only `customdata[1]`), LR2 the guard not keyed per structure, LR3 the `r <= idx` condition dropped | pins: P4's `range:top:sid0` first x = t[6]; `test_p4_the_reset_guard_is_per_structure_and_only_for_an_earlier_reset` (the helper directly; LR3's input is outside the domain — it pins the coded rule); `plan_n_inputs/mutants_n_review.py` re-runs LR1–LR3 + the restored all-structures snapshot |
| 3 | MAJOR | GLOSSARY ~272 (`probe_input_idx`) / ~295 (UnresolvedTrigger) still named `degenerate_parent_cycle` | fixed (my doc sweep had grepped those rows' neighbours only) |
| 4 | MINOR | the helper's docstring: "the latest reset with `R <= ev.idx`" ≠ the code (the latest reset, IF `R <= ev.idx`); keyed by `structure_id` alone = H1-only | docstring corrected |
| 5 | MINOR | PLAN_N §1 / §7 called the hover's `range_start_idx=` the MS df column — it is the anchor (`_range_left_idx`) | corrected in place |
| 6 | MINOR | CHARTING_SPEC: the same-structure overlap when `A < R`, the `R <= ev.idx` condition, the hover line's start | added |
| 7 | MINOR | the register §A line still "DECIDED"; the S_N boundary missing from the `/compare` skill | register closed (+ a hygiene line: the dead hover passes 1–2); the skill boundary at `/commit-save` |
| 8 | MINOR | `cmp_plan_n.py` did not fail on an extra CSV nor assert the cell total | `--expect-cells 164`; an extra CSV fails |

## 13. The fold re-verified (2026-10-12, session 3) — the final landing = the prediction exactly

The per-structure hover fix (finding 1), first folded, made the S_M2 H1 figure differ in exactly 4 range hover traces
(sid 0 at H1 750: lo 0.57809 → 0.57764, last update 747 → 748; sid 1 loses its point at its anchor 748; CSVs and M15
figures unchanged) — a PRE-EXISTING defect on a label range, so a second cause: split out
(`plan_n_inputs/followup_hover_per_structure.md`, register §A), the original pass 3 restored with a known-defect
comment. Final: suite **1708**; LR1–LR3 re-run on the final code **3 / 3 KILLED** (appended to
`mutants_n_review_result.txt`); replay 82.2 s wall, fetch gate PASS; `cmp_plan_n.py --expect-cells 164` **20
md5-equal, 7 transform-equal, 164 cells, OK**; `fig_hash` — **all 3 figures EQUAL to S_M2** (83 / 239, 137 / 123, 282 /
276); `run.log` 9105 lines, 0 differences beyond the 20 predicted lines and timings; grep 22.
