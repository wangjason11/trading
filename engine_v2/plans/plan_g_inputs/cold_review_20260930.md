# Plan G cold review (2026-09-30) — the implementation checklist

Canonical findings summary + the Q7–Q10 decisions: `PLAN_G_wvmi_unique_sub.md` §8 (+ Q10: a pin for a lock LP past the
cap falling back to the temp LP). This file keeps lens 3's measured test table, new pins,
mutant list and doc-site list (line numbers as of `5fa986b` — `5a4af14` later inserted 4 lines in WVMI_SPEC after
:65 and 6 in PART4 after :3291; grep, don't trust a line number; read-only review at `5fa986b`; the 8 affected test files passed — 160 tests), plus lens
1's exact run.log / cell numbers. Scratch scripts of the lenses: `%TEMP%/pe5/pg1..3/` (not a handoff).

## Test table (lens 3)

| Test | Action |
|---|---|
| `test_sub_wvmi.py` (6) | DELETE; port the create / lock / unlocked / no-CTS cases to the helper with `wvmi="none"` (record meta `{}`) |
| `test_sub_wvmi_per_sub::TestWindow` (5) | REWRITE as the post-pass: the record always exists; the trigger triple is set inside the window, None outside |
| `::TestLensRestriction` (2) | REWRITE per lens: an entry of the other lens gives None on this lens |
| `::test_earliest_by_m15_idx_wins_across_lenses` | REWRITE to the opposite rule: confluence rows (8, ZPT), counter rows (6, SCT) |
| `::test_unsorted…` / `::test_tie…` / `::test_entry_the_mapper_cannot_place…` | KEEP the intent (tie: both entries on one lens) |
| `::test_sweep_is_called_once…` | DELETE |
| `::test_records_persisted_into_every_lens…` | MOVE into the render test (shift, field == meta, deep copies, source untouched) |
| `::test_single_lens…` / `::test_persist_appends…` | REWRITE: render order kept; records joined to their sub by `sub_id`, not position |
| `::test_sub_listed_twice…` | REWRITE as "no double persistence" |
| `::test_sub_with_no_trigger…` | INVERT: records exist, trigger idx / type None, `parent_path_id` "H1.main" (Q9) |
| `::test_counts…` | REWRITE: sub C now counts → acted 3, records 9, by_lens {conf 5, ctr 7} |
| `::test_by_lens_key_order…` | KEEP |
| `::test_sweep_yielding_no_records…` | REWRITE to "0 records → not acted" |
| `test_wvmi_lp_bound.py` | drop the module-level `sub_wvmi` import (it fails all 6 at collection); `test_the_sub_sweep_passes_no_ends` → "the projection passes every cycle end"; keep the other 5; fix the docstring (l.8–9) + l.74 comment |
| `test_event_meta_idx_keys.py` | delete `test_persist_facade_wvmi…`; extend `test_mirror_shifts_every_site_exactly_once` (source untouched, per-lens field, `_Wvmi` stub + `structure_path_id` / `cycle_collapsed`); add WVMI to `test_mirrored_index_values_are_source_plus_slice_begin` and `test_every_int_meta_value_is_listed_or_known_non_index` (fields = source + 5; `triggered_by_event_idx` exempt — in `NEVER_LISTED`, not in `NON_INDEX_INT_META_KEYS`) |
| `test_render_sub_projection.py` | add `"wvmi"` to `_element_metas`; WVMI values in `test_open_sub_paints_to_edge…`; the lens-subset test on an open window too (its capped windows have 0 records) |
| `test_plan_e_role_pins.py` | `_PINNED` (:170) lists `multitf/sub_wvmi.py` → the helper's module |
| `test_meta_key_renames.py` | a WVMI row in `test_exporters_write_every_meta_dict_verbatim` |
| unchanged | `test_wvmi.py`, `test_e4_simulation.py`, `test_zone_proximity_reversal_cap.py`, `test_pooled_structure_build.py`, `test_fib_never_established_cap.py`, `test_poi_activation_moment.py`, `test_sid_records.py`, `test_lifecycle_sweep_unit.py` |

## New pins (lens 3; values measured at `5fa986b`)

- Sub projection on the double-rewind data (`project_to_window(compute_bounded_structure(_prepare_df(
  _make_double_rewind_data()), 0, 1), …)`; h1 and cross_cycle give the same): 3 records — (0,0) FB/LB/FP/LP (1,1,3,4)
  locked by 1; (0,1) (5,7,9,9) locked by 2; (0,2) (10,11,13,14) temp; meta `{}`. With `check_zone_proximity` stubbed
  to `{}` the sub still has 3 records and the main 0.
- LP bound (`_post_reversal_lure`), floor 5: cap 17 / "reversal" → (0,2) LP 14, pullback 1.0 (the no-ends mutant:
  19 / 0.7); cap 15 / "parent_end" → captured ends `{(0,0): 8, (0,1): 12, (0,2): 15}` == `compute_cycle_lifecycle` on
  the clipped events.
- `cycle_collapsed` on subs: double-rewind floor 9 → [T, F, F], floor 13 → [T, T, F], the open (0,2) always F;
  geometry fixture floor 107 → (107,107) T, floor 108 → (108,107) T (kills an `==` mutant), floor 90 → F; every flag
  == `_zone_render.collapsed_cycles(kl_zones)`.
- Main: `_run_downstream_pipeline(…, lifecycle_floor=13)` → [T, T, F]; no floor → [F, F, F]; main values unchanged
  with meta exactly `[(tbei 4|9|14), ZPT, 'H1.main', trigger_inner 0.602|0.599|0.601, proximity_pips 8]` in that key
  order; gate stub: a cycle whose first trigger is opp_sd gets no main record.
- Dual-lens render (`_two_lens_records`, open window): 1 record per lens (56,86,88,106) temp; field and meta path ==
  the lens path; the two lenses' objects and meta dicts distinct; `res.wvmi_records` keeps slice-local (51,81,83,101)
  and `first_path`.
- Post-pass: per-lens triple confluence (8, ZPT, 'H1.main'), counter (6, SCT, 'H1.main'); only-one-lens-has-a-trigger
  → that lens its triple, the other idx / type None and `parent_path_id` "H1.main" (Q9, decided); the first three meta keys are the triple, then the attribution
  keys, no extra keys; `type(idx) is int` and parent coords (6, not the LOH 27); one lens's pass leaves the other
  untouched.
- No double persistence: per lens `len(attrs["wvmi"]) == Σ len(res.wvmi_records)` over its subs; every
  `(sub_id, sid, cycle)` once; `persist_facade_wvmi_to_entity_df` and `multitf.sub_wvmi` gone.
- Exporter: header `cycle_collapsed` right after `lp_locked`; rows (lp_locked True, collapsed False) and (False, True)
  as written; `meta` == `str(r.meta)` incl. the None-triple row; `triggered_by_event_idx` an int text (`Int64`).
- Counts: a dual-lens sub → acted 1, by_lens {conf n, ctr n}.
- Stream → lens mapping helper: confluence = ZPT + SUBSEQUENT_COUNTER, counter = SUBSEQUENT_CONFLUENCE.
- `wvmi` validation: an unknown value raises.

## Mutants (lens 3; id — change → the pin that kills it)

G1a default `wvmi="none"` → stubbed main gate · G1b projection passes `"first_sd_prox"` → stub, sub still 3 records ·
G1c `"off"` kept for subs → render, 1 record per lens · G1d no ends for subs → lure LP 14 · G1e ends passed as end+1 →
captured ends · (+ the frame: no `iloc[:cap+1]` → a synthetic FP-past-cap pin, lens 2) · G2a `>=` → `>` → floor 9 /
floor 107 · G2b `>=` → `==` → floor 108 · G2c flag keyed on (sid, cycle+1) → floor 9 [T, F, F] · G2d flag on subs only
→ main floor 13 · G2e open cycle flagged → the (0,2) F pin · G2f dataclass default True → any F pin · G3a field not
set by the mirror → dual-lens field == lens · G3b field set on the source before the deepcopy → source untouched ·
G3c `copy.copy` instead of deepcopy → per-lens meta path · G3d persister kept → row counts · G4a streams unioned
across lenses → per-lens triple · G4b stream dict swapped at the call site → the mapping pin · G4c `meta.update`
instead of prepending → key order · G4d an extra `*_m15_idx` key → exact key set · G4e the LOH idx stored → parent
coords · G4f `parent_path_id` None on a no-trigger row (Q9) → the None pin · G4g `<=` → `<` → a window-edge case · G4h records joined by position →
the two-subs pin · G4i `acted` counted per lens → counts · X1 the column = `r.lp_locked` → exporter · X2
`m.get("cycle_collapsed")` → exporter · X3 column order moved → header · M1 `trigs[0]` → "any sd" → the opp_sd-first
stub.

## Doc sites that become false (lens 3; `*` = missed by PLAN_G rev 1 §6)

WVMI_SPEC: :14–17 (Gating: subs ungated); :55–147 "Sub entities" rewrite; *:158–165 ("only gated sids" → main only);
*:194–199 + *:268–278 (the sub "frame end, inclusive / no ends passed" → end − 1 via the table); *:281–289 (the
sub-sweep frame guarantee → the capped frame); *:311–316 (the tracker path → the mirror sets the field per lens;
`WVMIRecord._records` → `WVMITracker._records`); Pipeline Integration + Fields (`cycle_collapsed`, the per-lens path).
PART4: *§6.3 :1490–1494, *§6.5 :1536–1538, *§7 table :1558, §8 banner :1582–1595, *§8.1 (the main keeps its gate),
§8.3–8.5 superseded, *§8.6 lock table (not implemented; replace), §8.7 (+ `cycle_collapsed`), *§10.1 :1890–1892 +
§10.2 :1909, §17.10 settled, *§17.12 :3380 done, :1039 (a trigger-metadata window). GOTCHAS :1375–1424 SUPERSEDED
banner ("Why this is the correct model" / "Do NOT restore" now false). LANDMINES: *:394 (gate main-only), *:629
(`sub_wvmi`'s loop → the helper), *:919–980 (Rule 2, the gate table, the Trap, :975), :1060–1062 ("Why": attribution
only; `cycle_collapsed`; `triggered_by_event_idx` may be None), :1847–1850, "WVMI Constraints" #2 / #4 (lens 2).
ARCHITECTURE (whole file missed): *:406–411, *:479–480, *:499–501, :337–344 stale. GLOSSARY :198 (sub-row
`triggered_by_event_idx`: per lens, parent coords, may be None) + `cycle_collapsed` + "collapsed cycle".
*`engine_v2/CLAUDE.md:24` ("Next: the WVMI pass" → done). *`.claude/skills/compare/SKILL.md:480–484` (column
vocabulary / save-format boundary). *review_scripts README (the shadows → "pre-Plan-G code, ≤ `5fa986b`"). Memory
`project_wvmi_lifecycle_deferred.md` (the 2026-09-28 scope bullets "no record before `start_idx`", "export once per
sub" contradicted by Q2 / Q3). Code docstrings: orchestrator :129–132, 332–334, 356–359, 400, 634–638, 658–659,
679–693, 788, 981; pooled_structure_build :79; entity_df_mutation :1228, :1278; types :305–306.
Unchanged: CHARTING_SPEC (146, 285, 360–361), WORKFLOWS. Keep as written (dated): PART4 §13 steps (:2100–2122, 2229)
+ §17.11 (:3346–3358); WVMI_SPEC's 2026-09-20 "Measured consequence" + the REMOVED-2026-05-27 box; the GOTCHAS body +
"How it bit (2026-05-27)"; LANDMINES Rule 1's 3d.iii note; PLAN_C / D / E.

## Lens 1 numbers (for §5)

Summary line: `acted=8 records=18 by_started_by={'first_confluence': 5, 'reversal': 9, 'first_counter': 1,
'subsequent_confluence': 1, 'subsequent_counter': 2} by_lens={'confluence': 17, 'counter': 6}` (counting lens-df
records instead would give 23). Per-projection `[wvmi] total/locked`: sub 0 4/3, sub 1 4/3, sub 2 2/1, sub 3 3/3,
sub 4 1/0, sub 5 1/0, sub 6 1/0, sub 7 2/2. The exporter float rendering was measured by running the real
`export_wvmi` on a None-mixed column (`871.0`, `954.0`, empty).
