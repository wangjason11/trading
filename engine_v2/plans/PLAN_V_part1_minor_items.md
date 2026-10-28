# PLAN_V — the last Part 1 items: 8a the §A hygiene trio, 8b the same-candle replacement chain's `ended_by`, 8c the agreed direction for wave candles + WVMI (SESSION ORDER item 8, the user's scope 2026-10-28)

**Status: 2026-10-28 — 8a LANDED (`955489b`, = its prediction, §1.5), 8b LANDED (`01f5282`, = its prediction, §2.4);
one save after both (the user: "Two commits, one save"): S_V8b `20261028_141322_245620c` (source `245620c`, save
`67c4d05`, trunk `29276a0`; = S_U3b byte for byte; the new `/compare` baseline). 8c MEASURED + DECIDED (§3.1–3.2);
8c LANDED (`1318c82`) = the variant byte for byte (§3.3; the user's chart look: "looks good go ahead"), saved:
S_V8c `20261028_153518_1318c82` (save `fb6fb81`, trunk `8e6a53a`; = `LATEST_<branch>`) — item 8 DONE, Week 9 Part 1
complete.** Baseline **S_U3b**
`artifacts/commits/week9-entries-decisions/20261028_130713_9c3949d` (source `9c3949d`; HEAD at drafting `409c8ac`, code
= `9c3949d`); suite 2154. The scope is the user's (register `IDEA_PARKING_LOT.md` §D "SESSION ORDER" item 8, 2026-10-28):
three landings in the order 8a → 8b → 8c, one cause per `/compare` (the two zero-change landings first, so 8c's
`/compare` carries one cause). Inputs: `plans/plan_v_inputs/`.

---

## 1. 8a — the §A hygiene trio (ONE landing, one commit)

**Prediction: ZERO output change** — 30 / 30 CSVs byte-equal, the three figures' full JSON equal
(`plans/plan_e_inputs/review_scripts/cmp_fig_full.py`), the PNGs byte-equal; the two `[floor_seed]` lines unchanged;
the suite = 2154 + the new pins. Every value below is verified at `9c3949d` (line numbers there).

### 1.1 (i) The charting audit's style / code items (memory `project_charting_audit_20260921.md`)

| Item | Today (`charting/…`) | The change | Why it is zero change |
|---|---|---|---|
| #7 the sub-native zone outline | `export_m15_chart.py:1289-1290` (KL), `:1575-1576` (POI): `f"rgba(0, 0, 0, {line_op})"`, width `0.5` | new key `zone.m15.outline` = `{"rgb": "0, 0, 0", "width": 0.5}`; colour = `_rgba_from_rgb(rgb, line_op)` | `_rgba_from_rgb` = `f"rgba({rgb}, {opacity})"` (`export_plotly.py:53`) → the same string; `0.5` the same float |
| #8 the hover-label colours | `export_m15_chart.py:1732` (the M15 prev-BOS line), `:1927` (every M15 dot layer): `royalblue` / `white`; `:2824` (the H1 proximity triggers on the M15 charts): `black` / `white` | new keys `hoverlabel.m15_structure` = `{"bgcolor": "royalblue", "font_color": "white"}`, `hoverlabel.zone_proximity` = `{"bgcolor": "black", "font_color": "white"}`; `hoverlabel=dict(<key>)` | the same two keys in the same order |
| #9 the zone hover hitboxes | `export_m15_chart.py:1380` (M15 KL), `:1664` (M15 POI), `:2379` (H1-overlay KL), `:2633` (H1-overlay POI): `{"width": 6, "color": "rgba(0,0,0,0)"}` | read `zone.kl.hover_line` / `zone.poi.hover_line` as the H1 chart does (`export_plotly.py:1783-1784`, `:2478-2479`) | the keys hold that exact dict |
| #11 the dead reversal-watch line | key `structure.reversal_watch_line` (`style_registry.py:82`) read nowhere; `export_plotly.py:1588-1590` computes `watch_active` / `prev_active` / `watch_y`, never plotted (`prev_active`, `watch_y` = ruff F841) | delete the key + the three lines; the `if` guard stays (it gates the `rc` markers inside it), its banner comment corrected | dead code |
| the 8 unused locals (ruff F841) | `export_m15_chart.py:1079` `time_by_idx_m15`, `:1080` `idx_set_m15`, `:1128` `sub_dir`, `:1145` `sid_wvmis`, `:2037` `rev_confirmed_by_sid`, `:2274` `most_recent_kl_sid`, `:2534` `most_recent_poi_sid`, `:2693` `h1_times` | delete; a name orphaned by a deletion (`wvmis_by_sid`?) re-checked with ruff and deleted if dead | each is a pure read (`sub_dir = int(starting_sd)` could only raise on a None — never on the window: the chart renders) |

**Found while verifying (same class, zero change — folded in unless the user says no):**
- **More hardcoded hitboxes** (#9's twins, hardcoded on BOTH charts): the wave-candle vertical hitbox `dict(width=8, …)`
  (`export_m15_chart.py:1546` sub, `:2526` H1 overlay; `export_plotly.py:2087`) and the fib hitbox `dict(width=6, …)`
  (`export_m15_chart.py:1709`; `export_plotly.py:2572`) → new keys `wave_candle.hover_line` (width 8) and
  `fib.hover_line` (width 6), the same `rgba(0,0,0,0)`. After 8a no hover-hitbox literal is left in the chart code.
  (The outline rects' `fillcolor="rgba(0,0,0,0)"` is "no fill", not a style — kept.)
- **Five more unused registry keys**: `zone.h1_overlay.kl.buy/sell`, `zone.h1_overlay.poi.buy/sell` — the M15 charts'
  H1 overlay reads `zone.kl.*` / `zone.poi.*` (`export_m15_chart.py:2287-2289` "Item 5: legacy `zone.h1_overlay.kl.*`
  bases supersede…", `:2548`); the POI pair still holds the pre-2026-09-20 brown — and `fib.label` (audit #10:
  "defined but unreferenced", still so). Delete; CHARTING_SPEC's style-key table names the `zone.h1_overlay.*` keys as
  the H1 overlay's (a doc error) → corrected.
- **The audit's unverified list, the zero-change part — VERIFIED**: the stale "(dashed)" banners on the H1-overlay
  paths (`export_m15_chart.py:2104` swing lines, `:2426` wave candles, `:2643` prev-BOS — every one draws solid: its
  registry key has no `dash`) and the dashed `.get` fallback at `:2654` (never taken — the key exists) → corrected;
  CHARTING_SPEC's style-key table omits the M15 `*_prior` keys → added.
- **The rest of the unverified list stays in §A** — each would change a rendered figure (not a zero-change landing):
  the H1 chart drawing a collapsed cycle's fib / proximity markers on collapsed-cycle zones / the opt-in M15.counter KL
  overlay with no collapsed skip; the wave-candle `dash` dropped by the H1 chart; the forming-candle wave verticals'
  live style; the H1-chart-vs-overlay filter quirks; the latent "all-forming sub draws no live segment" — and PART4
  §16.7 (zone text labels specified, none rendered — build or drop is the user's call).

### 1.2 (ii) The dead `_run_multi_tf_dual(poi_zones=…)` read — STALE, nothing to change

`poi_zones` IS read: `pipeline/orchestrator.py:1126`, `build_cycle_proximity_scans(h1_df, sorted_events, kl_zones,
poi_zones, …)` — since PLAN_R step 2 (`817d2a9`, the proximity overlay inside the sweep). The register line (§A,
2026-10-0x) predates it. Closed in the register; no code change.

### 1.3 (iii) The H1 frame's "H1" stamp — assert H1 at the replay's entry

Today `run_pipeline` classifies with `apply_candle_classification(df, floor_seed=floor_seeds.get("H1"))` — no
`timeframe`, so the default "H1" is stamped (`features/candle_classifier.py:29`, `:44`) — while `run_replay` keys the
seeds by `CONFIG.timeframe` (`run_replay.py:229`, `:315`). Passing `CONFIG.timeframe` to the classifier alone would make
an M15-main run LOOK supported while the orchestrator stays H1-main everywhere else: the registry `"H1.main"`
(`orchestrator.py:604-607`), `parent_tf="H1"` (`:621`, `:639`, `:659`), the proximity scans' `"H1"` (`:1126`), the
floor-seed key (`:513`, `:518`). So: **assert**. `run_replay.main()`'s first statement:
`assert CONFIG.timeframe == "H1", …` (the message names why: the pipeline is H1-main). `run_replay.main` is the one
production entry; the debug / plan scripts that read `CONFIG.timeframe` are one-offs (left alone).

### 1.4 Tests + mutants (8a)

A revert to a literal survives byte-equality by construction (the registry holds the same value), so each site gets a
behavioural pin: **patch the registry key to a sentinel → render → the figure carries the sentinel at that site**
(`tests/test_plan_v_registry_styles.py`, on the existing harnesses: `test_plan_n_naming_audit._render_m15` /
`_two_step_zone` for the KL zones, `test_fib_chart_render._render_m15` + the sub POI of
`test_plan_h_landing_b.py:560` and the H1 POI of `test_chart_controls.py:129`, `_render_proximity_triggers_overlay` as in
`test_proximity_trigger_hover.py:77`; the wave-candle / fib / dot / prev-BOS fixtures from their own render tests).
(iii): `run_replay.main()` with `CONFIG.timeframe="M15"` (monkeypatched) raises before the history fetch (a recording
stub — the mutant without the assert reaches the fetch with "M15": the harm).

Mutants (`plans/plan_e_inputs/review_scripts/mutant_runner.py`, full suite): each registry read reverted to its literal
(one per site: 2 outline, 3 hover label, 4 zone hitbox, 5 wave / fib hitbox) + the assert removed — each killed by
its pin. Deletions (#11, the keys, the locals) have nothing to pin: the zero-change `/compare` is their check.

Docs (same commit): CHARTING_SPEC (the style-key table: `zone.m15.outline`, `hoverlabel.*`, `wave_candle.hover_line`,
`fib.hover_line`, `*_prior`; the Reversal row without the watch line; the H1-overlay row = `zone.kl.*` / `zone.poi.*`;
the outline table rows name `zone.m15.outline`); the register §A (three lines → closed / the remaining unverified list);
the memory audit file (#7–#11 FIXED).

### 1.5 Result (2026-10-28) — = the prediction

Replay 65.9 s wall; `/compare` vs S_U3b (run mode): 30 / 30 CSVs md5-equal, the three figures' full JSON equal
(`cmp_fig_full.py`; 79 / 302 / 154 traces), the PNGs byte-equal, the `[floor_seed]` lines unchanged, skip grep 33
(`plan_v_inputs/v8a_compare.txt`). Suite 2164 (2154 + the 10 pins of `tests/test_plan_v_registry_styles.py`).
Mutants 15 / 15 KILLED, each by its own pin (`plan_v_inputs/mutants_8a.py` / `.txt`). As built, beyond §1.1: the
deletions orphaned `all_wvmi_records` → `wvmis_by_sid` (built, never read), `all_poi_sids` and the
`_get_reversal_confirmed_by_sid` import — deleted too (ruff F clean on `export_m15_chart.py`); `export_plotly.py`'s
pre-existing F401 (`make_subplots`) / F541 left alone (not F841, not in the scope). The hover-label reads carry no
literal fallback (`dict(_style(key))`); the hitbox / outline reads keep the file's `.get(key, <literal>)` idiom.
The sub wave-candle hitbox site has no render fixture — pinned by a source count (as
`test_wave_candles.test_every_wave_candle_hover_site_uses_the_label_helper` does); the other two are rendered.

---

## 2. 8b — the same-candle replacement chain names the replacer that survives

**Prediction: ZERO output change** — 0 on the window (the baseline log carries no `which never started` WARNING —
measured on 8a's replay, `plan_v_inputs/`); 30 / 30 CSVs + the figures byte-equal.

### 2.1 Today (`multitf/lifecycle_sweep.py`)

Incumbent A (live), co-starters B then C (same lens, parent `(S, C)`, direction; RECORD_START at the same candle `t`,
seq order). B's phase 1 queues RECORD_END(A, `same_dir_replacement`, B's pool). C's phase 1 queues RECORD_END(A, …,
C's pool) and freezes B on the spot (the same-idx collision branch, `:493-509`: "the later seq replaces the earlier",
B's `ended_by` = C). Phase 3 (`_phase3_record_end`, `:531-538`) applies `min(entries, key=priority)` — the FIRST of the
equal-priority entries — so A's `ended_by` = B's pool, a sub that never starts (`ended_by_sub_id` None, the WARNING at
`:606-610`). With three co-starters B, C, D the same holds one level down: B's `ended_by` = C, which D froze — C never
starts either.

### 2.2 The fix (the user's pick, 2026-10-28: "`ended_by` = the replacer that survives")

1. **Phase 3: of the highest-priority entries, the LAST queued.** Equal-priority `same_dir_replacement` entries for one
   record at one candle come only from that candle's RECORD_STARTs of the same `(lens, S, C, direction)`, run in seq
   order; each later starter freezes the earlier co-starters, and a starter that is frozen itself (phase 1's "the sub
   froze while pending") returns before queueing — so the last queued entry names the one that survives the candle.
   (`reversal` / `parent_end` entries carry no replacer — unchanged.)
2. **Phase 1, the same-idx collision branch: re-point the chain.** When `rec` freezes co-starter `inc` at `t`, every
   record of the same `(lens, S, C)` that `inc` froze at `t` (`trigger_end_idx == t`, `same_dir_replacement`,
   `ended_by_pool_idx == inc.pool_idx`) now names `rec`'s pool.

The `_after_sweep` WARNING stays as the guard for any other path.

### 2.3 Tests + mutants (8b)

- The pin `test_lifecycle_sweep_unit.py::test_a_replacer_that_never_starts_leaves_ended_by_empty_and_warns` (it pins
  TODAY's behaviour) BECOMES `test_a_same_candle_replacement_chain_names_the_replacer_that_survives`: the same fixture;
  A's `(end_idx, end_reason, ended_by_pool_idx, ended_by_sub_id)` = `(400, "same_dir_replacement", 2, 1)` (C), B's
  `ended_by` = C, no `which never started` line.
- New: three co-starters (B, C, D) → A and B both name D; no WARNING.
- Mutants: phase 3 back to the first entry (killed by the first pin); the re-point dropped (killed by the second);
  the re-point without its `trigger_end_idx == t` filter (a record that `inc`'s pool ended at an EARLIER candle would be
  re-pointed — a third pin builds that case if the fixture allows it; else the mutant is reported as unpinned).

Docs: the module docstring's phase table (row 3) + PART4 §17's phase table (`PART4_REFACTOR_SPEC.md:540`); the register
D.2 entry → closed.

### 2.4 Result (2026-10-28) — = the prediction

Measured on 8a's replay (the baseline sweep): 0 `which never started`, 0 `started records for` (multi-incumbent), 0
`same-idx start collision` lines. 8b's replay (on top of 8a) 101.3 s wall (the mutants ran alongside): 30 / 30 CSVs
md5-equal, the figures' full JSON + the PNGs equal, the gate + skip grep 33 unchanged, the same three counts 0
(`plan_v_inputs/v8b_compare.txt`). As built: the re-point's filter is `trigger_end_idx == t` AND
`ended_by_pool_idx == inc.pool_idx` (the reason is implied — dropped, it could never be pinned); its pins: the
three-co-starter test also carries an opposite-direction pair frozen at the same candle (kills "any ender"), and
`test_the_re_point_leaves_a_record_ended_at_an_earlier_candle` (a post-end zero-length counter record whose
`ended_by` = the replacer's pool — kills "any candle"). Same-candle co-starters are ordered by
`trigger_event_idx` (`_resolver_by_event_idx`): a sibling trigger type cannot lag its finalize.
Suite 2166 (8a's 2164 + 8b's two new tests); mutants 4 / 4 KILLED (`plan_v_inputs/mutants_8b.py` / `.txt`).
The earlier landings' mutant anchors in every file 8a / 8b touched (`export_m15_chart.py`, `export_plotly.py`,
`style_registry.py`, `run_replay.py`, `lifecycle_sweep.py`): 0 stale of the 147 that matched once at `409c8ac`
(`plan_v_inputs/anchor_scan.py 409c8ac` → `anchor_scan.txt`).

---

## 3. 8c — the agreed direction for wave candles + WVMI (a behaviour change)

Measured first (the window's volume-only dojis; a VARIANT replay of the one-line change in a short `%TEMP%` dir from
`git archive`), shown to the user with a candle table; the user decides; then the plan and the landing = the variant
byte for byte.

### 3.1 The measurement (2026-10-28, `plan_v_inputs/`)

The change: `_is_qualified` = `agreed_direction(direction, vol_dir) == required_dir` (the `direction !=
required_dir` early return dropped) — it admits exactly the volume-only doji (direction 0, vol_dir = the required
direction); a split candle still fails (agreed = `AGREED_SPLIT`). One helper, so both homes move (wave candles' last
pullback / last breakout searches; the WVMI temporary LP's candidates). The first-wave candles read the compound rule
(`_is_compound_first_wave`) — untouched.

- **The window's volume-only dojis** (`v8c_measure.py base` → `v8c_diff.py`, the replay's frames with their floor
  seeds): H1 13 dojis, 8 volume-only (+1: 5, −1: 3); M15 100 dojis, 59 volume-only (+1: 25, −1: 34).
- **Wave candles: 0 of 53 results move** (H1 main + every unique sub).
- **WVMI: 2 of 24 records change — only their temporary-LP history**; every record's own fields (the final /
  locked `lp_idx`, the momenta) unchanged. Sub 3 (sell zones; on both lenses): cycle 1 gains temporary LPs at M15
  2907 and 2913, cycle 2 at 3252 — each a volume-only doji (vol_dir +1) closing further in the pullback direction
  than the LP before it, replaced by the next candle; the rows before them end one candle earlier
  (`lp_status_idx` 2908 → 2907, 2914 → 2913, 3253 → 3252) (`v8c_diff.txt`).
- **The variant replay** (`git archive HEAD` + the one-line change, `%TEMP%\v8c`; 93.3 s wall, the gate + skip grep
  33 unchanged) = the measurement: 28 / 30 CSVs md5-equal; the two lens `*_wvmi_lps.csv` +3 rows each (215 → 218,
  77 → 80 lines) + the three `lp_status_idx` cells; one trace per M15 chart, 'M15 WVMI LP sell': confluence 112 →
  115 dots, counter 47 → 50 (the three new x: 2025-12-30 04:45, 06:15, 2026-01-05 19:00); the H1 chart and its PNG
  equal (`v8c_variant_compare.txt`).

### 3.2 The user's decisions (2026-10-28, AskUserQuestion after the measurement above)

- 8a + 8b: "Two commits, one save" — 8a then 8b (one cause each), one `/commit-save` at 8b's HEAD (the save =
  S_U3b byte for byte).
- The rule: "Apply it" — the landing = the variant byte for byte; GLOSSARY "qualified candle", WAVE_CANDLES_SPEC,
  WVMI_SPEC updated.
- The chart gap: "Draw it in its vol_dir colour" — a wave candle's vertical takes its candle direction, or its
  `vol_dir` when the direction is 0 (only the doji case changes; zero change on this window — no direction-0 wave
  candle exists on it, so no rendered example: CHARTING_SPEC states the rule, a synthetic render pins it).

### 3.3 The landing (2026-10-28) — = the variant byte for byte; the user's chart look: "looks good go ahead"

Code: `zones/wave_candles._is_qualified` (the `direction != required_dir` early return dropped; the docstring); the
three wave-candle renderers (`export_m15_chart.py` the sub lines + the H1 overlay, `export_plotly.py` the H1 chart):
a direction-0 candle takes its `vol_dir` (0 without the column), a true doji is still skipped. 0 of the window's 103
wave-candle indices sits on a direction-0 candle (both rules), so the renderer half is zero change here.
Replay 103.8 s wall. vs the variant (`%TEMP%\v8c`): 30 / 30 CSVs md5-equal, the figures' full JSON + PNGs equal. vs
S_V8b (the `/compare`, run mode): the two lens `*_wvmi_lps.csv` (+3 rows each, three `lp_status_idx` cells), one
trace per M15 chart ('M15 WVMI LP sell': confluence 112 → 115 dots, counter 47 → 50) + their PNGs; the H1 chart,
its PNG and the other 28 CSVs equal; the gate + skip grep 33 unchanged (`plan_v_inputs/v8c_landing_compare.txt`).
Tests: the PLAN_H grid pin `test_qualified_rule_reads_the_agreed_direction_unchanged` (it pinned the OLD rule's
equality) → `test_qualified_rule_is_the_agreed_direction_incl_the_volume_only_doji` (an explicit table); +
`TestIsQualified` × 2, `test_wvmi.py::…::test_a_volume_only_doji_is_a_candidate` (the real collaborator), the three
renderer pins (`test_plan_v_registry_styles.py`: the fixture's wave candles turned into volume-only dojis draw the same
verticals on the H1 chart + the H1 overlay; the M15 sub site by a source count). Suite 2172. Mutants 4 / 4 KILLED
(`plan_v_inputs/mutants_8c.py` / `.txt`). Docs: GLOSSARY (qualified candle, agreed direction, volume-only doji),
WAVE_CANDLES_SPEC (the definition + the edge-case table), WVMI_SPEC (the LP candidates), CHARTING_SPEC §10 (the
colour rule), the register D.2 entry closed.

**Found 2026-10-28 (while building 8a) — a chart question 8c raises:** every wave-candle renderer skips a candle
whose `direction` is 0 (`export_m15_chart.py:1493-1494` the sub, `:2475-2476` the H1 overlay; `export_plotly.py:2034-2035`
the H1 chart — the vertical is coloured by the candle's direction). A volume-only doji has direction 0, so under 8c it
would become a wave candle in the CSVs / WVMI but draw NO vertical on any chart. Show the user with the variant (draw
it in its agreed direction's colour, or keep it off the chart). → DECIDED §3.2 ("Draw it in its vol_dir colour"), LANDED §3.3.
