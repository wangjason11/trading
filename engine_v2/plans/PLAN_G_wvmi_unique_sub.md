# Plan G — WVMI on the unique sub (lifecycle-governed, exported like zones)

Status: **LANDED 2026-09-30 (§9)** — rev 2 (2026-09-30): measured, 6 decisions, cold-reviewed (3 lenses, SOUND WITH FIXES, folded — §8); Q7–Q10 decided the same day; implemented against §4–§6 and the checklist the same day; `/compare` == §5 in every cell. Heavy tier: options →
user decisions → written plan with predicted per-CSV deltas → plan cold review (3–4 lenses) → implement (likely the
next session). Memory: `project_wvmi_lifecycle_deferred.md` (the 2026-09-28 opening, the user's direction).

## 1. Why

User direction (2026-09-28, verbatim): "once we have the pooled sub MS completed, we would actually move WVMI directly
to unique subs as opposed to existing on lens / sub records level. Conceptually, this is the same as zones. Since
unique subs are where trading decisions will be made and governed by life cycle, it would make sense WVMI lives here as
well." Today (Plan C's minimal §17.10) a sub gets WVMI only when a parent-trigger stream entry of one of its lenses
lands inside its window; the records are then persisted into every lens df stamped with the SWEEPING lens path.
Nothing reads a momentum value yet (CSV export only; the M15 chart collects `sid_wvmis` and never draws them), so
every choice below is CSV-only — it must land before the strategy layer reads WVMI.

## 2. Measured inventory (HEAD `896ffd4`, reference window; tools `review_scripts/wvmi_shadow.py` +
`wvmi_gate_shadow.py`, behaviour-neutral replays)

**Per unique sub** (all 8 rendered; `life` = `compute_cycle_lifecycle` with the sub's floor / cap — the KL / POI /
fib table; `[s, s)` = a COLLAPSED cycle, rendered inert on zones; "rec" = what `WVMITracker` yields with NO gate):

| sub | lenses | start → end (reason) | today swept by | cycles: life / CTS_CONF moment → rec (FP, LP, locked) |
|---|---|---|---|---|
| 0 | conf | 1020 → 1940 (reversal) | — (no stream entry in window) | c0 [1020,1020) collapsed / 802 → FP 787 LP 917 L · c1 [1020,1224) / 1025 → 1021, 1107 L · c2 [1224,1721) / 1257 → none (no CTS wave FP) · c3 [1721,1793) / 1748 → 1747, 1761 L · c4 [1793,1940) / 1795 → 1794, 1898 temp |
| 1 | conf | 1940 → 2470 (reversal) | — | c0 [1940,1940) collapsed / 1818 → 1817, 1837 L · c1 [1940,1940) collapsed / 1902 → 1900, 1935 L · c2 [1940,2231) / 1948 → none (no CTS wave FP) · c3 [2231,2343) / 2236 → 2232, 2272 L · c4 [2343,2470) / 2367 → 2365, 2455 temp |
| 2 | conf | 2470 → 2829 (reversal) | — | c0 [2470,2608) / 2560 → 2559, 2563 L · c1 [2608,2829) / 2612 → 2611, 2734 temp |
| 3 | conf + ctr | 2829 → 3611 (parent_end) | conf H1 710 (M15 2843) | c0 [2829,2829) collapsed / 2752 → 2736, 2757 L (**today's pre-start record**) · c1 [2829,2992) / 2844 → 2843, 2914 L · c2 [2992,3589) / 3086 → 3048, 3299 L · c3 [3589,3611) no CTS_CONF |
| 4 | conf | 3621 → 3819 (same_dir_replacement) | conf H1 926 | c0 [3621,3819) / 3741 → 3647, 3773 temp |
| 5 | ctr | 3707 → 4083 (same_dir_replacement) | ctr H1 954 (M15 3819) | c0 [3707,4083) / 3786 → 3761, 4002 temp (CTS_CONF before the sweeping trigger — the record does not depend on it) |
| 6 | conf | 3819 → 4200 (reversal) | conf H1 1020 | c0 [3819,4200) / 4157 → 4000, 4177 temp |
| 7 | conf + ctr | 4083 → open (edge 4227) | conf H1 1020 (var4) | c0 [4083,4157) / 4103 → 4083, 4129 L · c1 [4157,4179) / 4159 → 4158, 4167 L · c2 [4179, —) no CTS_CONF |

- **Today:** 5 subs swept (3–7), 8 records; exported as 13 rows (confluence 7: sub 3 ×3, 4, 6, 7 ×2; counter 6: sub 3
  ×3, 5, 7 ×2) — the dual-lens subs 3 and 7 on BOTH lens CSVs, and their counter rows stamped
  `H1.main >> M15.confluence` (the sweeping lens).
- **No gate** would add 10 records (subs 0 ×4, 1 ×4, 2 ×2) → 18. Every record's VALUES are gate-independent (the
  trigger is attribution meta only; 8/8 of today's records identical under "no gate").
- **Collapsed-cycle (pre-start) records:** 4 of the 18 — sub 0 c0, sub 1 c0 + c1, sub 3 c0 (today's one). Every one is
  a cycle established and confirmed before the sub's `start_idx`, whose lifecycle clamps to `[start, start)`.
- **How zones treat a collapsed cycle (the "like zones" reference):** exported as a row, inert — KL `status`
  "inactive", no activation, empty `activation_history`; fib `status` "inactive"; charts hide them. Zones of a dual-lens
  sub are mirrored into BOTH lens CSVs, each stamped with ITS OWN lens path (sub 3 / sub 7 KL, POI, fib: confluence rows
  `M15.confluence`, counter rows `M15.counter`) — so "like zones" is NOT "once per sub".
- **The sub LP end:** a sub frame ends AT the sub end (`iloc[0:end + 1]`, inclusive); the sub sweep passes no cycle
  ends (pin `test_wvmi_lp_bound.py::test_the_sub_sweep_passes_no_ends`), so a temp LP may take the end candle; the
  main stops at the cycle's `end − 1` since 2026-09-28. Measured: bounding the sub search at `end − 1` changes **0
  values** on the window (18/18 records identical) — only unlocked last-cycle records can differ, and none picks its
  end candle here.
- **Main** (gate = the cycle's first sd zone-proximity trigger): 3 records of 5 cycles — (0,1) FP 653 LP 896 temp,
  (1,1) 762 / 824 L, (1,2) 905 / 1055 temp. No gate would add (0,0) 436 / 590 L and (1,0) 712 / 728 L. (1,0) and (1,1)
  are COLLAPSED (`[902, 902)`: sid 1's struct start is the reversal 902) — so (1,1), a record today, is a
  collapsed-cycle record too.

## 3. Decisions (asked 2026-09-30)

**USER DECISIONS (2026-09-30), all four as recommended:** Q1 (A) no gate on subs — every rendered sub gets WVMI,
triggers as metadata; the main KEEPS its first-sd gate. Q2 (A) collapsed-cycle records kept, marked inert like zones
(the option text named main (1,1) as affected too — the flag is computed from the same lifecycle table on the main).
Q3 (A) exported like zones — each lens CSV the sub is on, stamped with that lens's own path. Q4 (A) the sub temp-LP
search stops at the cycle's `end − 1` like the main; `test_the_sub_sweep_passes_no_ends` is rewritten on purpose.
Follow-ups (same day): **Q5 trigger metadata PER LENS** — a lens's rows name that lens's first stream trigger inside the
sub's window (blank if none), matching the per-lens path stamp; **Q6 the inert flag = `cycle_collapsed`** (bool field +
CSV column like `lp_locked`; the codebase's term — `_zone_render.is_collapsed_cycle_zone` / `collapsed_cycles`; `status`
stays the computation state). Found while asking: today a dual-lens sub's counter rows CONTRADICT themselves — column
`structure_path_id` = `M15.confluence` (the record field, the sweeping lens) while `meta["structure_path_id"]` =
`M15.counter` (`persist_facade_wvmi_to_entity_df` stamps only the meta).


- **Q1 GATE** — (A, my lean) none on subs: every rendered sub gets WVMI; the sub's parent triggers stay as metadata;
  (B) none on subs AND on the main (the main's first-sd gate goes too: +2 main rows); (C) keep today's per-sub trigger
  gate (subs 0–2 keep none).
- **Q2 COLLAPSED-CYCLE records** — (A) keep, marked inert like zones (a `status`-like flag; exported, never "live");
  (B) drop (no record for a cycle whose lifecycle collapses — sub 3 c0 disappears; on the main (1,1) too, if Q1 = B or
  the rule is applied to the main); (C) keep as today, unmarked.
- **Q3 EXPORT shape** — (A) like zones: each lens CSV the sub is on, each row stamped with its own lens path
  (duplicates stay for dual-lens subs, the stamp is fixed); (B) once per sub: one pool-level `*_M15_wvmi.csv`
  (like `*_M15_unresolved_triggers.csv`), the two lens `_wvmi.csv` files go; (C) once per sub on its first lens only.
- **Q4 LP end** — (A) sub search stops at the cycle's `end − 1` like the main (0 values change here; the pin is
  rewritten consciously); (B) keep the inclusive frame end.

## 4. Design (rev 2 — the cold review folded in, §8; Q7–Q10 decided 2026-09-30)

- **G1 — sub WVMI is computed INSIDE the unique sub's projection, like KL / POI / fib.** `_run_downstream_pipeline`'s
  `skip_wvmi: bool` becomes ONE parameter `wvmi: str` ∈ {`"first_sd_prox"` (default — the main, unchanged), `"none"`
  (a sub: every CTS_CONFIRMED is offered to the tracker), `"off"` (no WVMI — the test callers that pass
  `skip_wvmi=True` today)}; any other value raises (a typo must not silently gate subs). `check_zone_proximity` stays
  on the main path exactly as today — it also feeds the var3 / var4 detectors and the §8.5 streams — and is never run
  for a sub (`zone_proximity_triggers` stays `{}`), independent of `wvmi`. `project_to_window` passes `wvmi="none"`.
  The tracker loop (today duplicated in the main branch and in `sub_wvmi.compute_parent_driven_sub_wvmi`) becomes ONE
  helper in `pipeline/orchestrator.py` that builds its `WVMITracker` through the orchestrator module (the LP-bound pins
  monkeypatch `orch.WVMITracker`), sorts by `ef.processing_order_key` (the `test_plan_e_role_pins` `_PINNED` guard
  moves with it), keeps the main's gate, proximity-meta key order and insertion order, and builds `cycle_end_by_key`
  from the same `compute_cycle_lifecycle` table KL / POI read — **Q4: the sub LP search stops at `end − 1`**.
  **The tracker's FRAME is `df.iloc[:lifecycle_cap + 1]` when a cap is set** (= today's `LowerTFResult.df`, the frame
  §2's shadow measured): the natural-end `bounded.df` bounds only the temp LP — FP (≤ CTS_CONFIRMED moment + 10), LB (≤
  CTS anchor + 5) and the lock LP (≤ BOS_{n+1} anchor + 5) could otherwise read candles past a capped sub's end (a
  look-ahead; 0 on the window — the closest CTS_CONFIRMED sits 43 candles before its cap). The helper asserts every
  record key has a table end when a cap is set (today unreachable: a kept CTS_CONFIRMED implies its CTS_ESTABLISHED is
  kept, and the cap gives every key an end). `render_sub_projection` already carries `down["wvmi_records"]` into
  `LowerTFResult.wvmi_records` (today empty; it stays slice-local with the first lens's path and NO trigger meta — only
  the lens copies get G3 / G4). Locks: the same knowable-at-clipped events as today. **Q10: a lock LP outside the tracker's frame (BOS_{n+1}'s last
  wave candle past a capped sub's end, or past the main's data edge) falls back to the temp LP** (bounded ≤ end − 1 —
  as the lock already does when BOS_{n+1} has no last wave candle); today it keeps the idx with volume / pullback
  None. 0 on the window; pinned on a synthetic lock past the cap.
- **G2 — `cycle_collapsed` (Q2 / Q6).** `WVMIRecord.cycle_collapsed: bool = False`, set by the helper from the same
  table: `start >= end` (end not None) — equal, measured on every KL row, to `_zone_render.is_collapsed_cycle_zone` on
  the cycle's BOS zone (KL `confirmed_idx = max(BOS moment, struct start)` = the table start). Main AND subs.
  Exported as a column right after `lp_locked`. A flag only — no start / end / reason fields (the WVMI-record
  lifecycle removed 2026-05-27 stays removed). It describes the CYCLE: a record created ON its cycle's end candle
  (CTS_CONFIRMED moment == cap; kept by the inclusive knowable-at clip, cycle `[s, cap)` live) is not flagged —
  documented edge, 0 on the window.
- **G3 — export like zones (Q3).** The mirror (`mirror_lower_tf_result_to_entity_df` step 8 — it already DEEP-COPIES
  each record into EVERY lens df with that lens's attribution, so no aliasing across lenses) persists them and now
  also sets the record FIELD `structure_path_id` to the lens path (today only the meta — the self-contradicting
  counter rows); the source record stays untouched. `persist_facade_wvmi_to_entity_df` and `multitf/sub_wvmi.py`
  (`ParentTrigger`, `compute_parent_driven_sub_wvmi`) are deleted (kept, the persister would write every record twice;
  no other path writes `attrs["wvmi"]` — sid records, facades, the H1 overlay checked). The exporter writes
  `triggered_by_event_idx` as pandas nullable `Int64`: once the column mixes ints and None, `to_csv` renders every int
  as a float (`710` → `710.0` on all 7 existing confluence rows — measured by running the exporter); `Int64` keeps
  `710` and writes None as an empty cell.
- **G4 — trigger metadata per lens (Q1 "triggers as metadata", Q5).** The lens → stream mapping becomes one helper
  returning `{lens: stream}` (confluence = `ZONE_PROXIMITY_TRIGGER` + `SUBSEQUENT_COUNTER_TRIGGER`; counter =
  `SUBSEQUENT_CONFLUENCE_TRIGGER` — §8.5's cadence classes; pinned: a swap would rewrite every sub row's trigger
  fields and only `/compare` would see it). `_assign_sub_wvmi_per_sub` becomes a post-pass over each lens df's
  `attrs["wvmi"]`, joining each record to its sub on `meta["sub_id"]` (never position) and mutating only that lens's
  copy: the FIRST entry of the lens's stream (LOH-mapped, unchanged) inside `[start_idx, m15_end_idx]` gives
  `triggered_by_event_idx` (PARENT coords, int) / `triggered_by_event_type`; no entry → both PRESENT with None;
  `parent_path_id` is ALWAYS `"H1.main"` (Q9 — the parent entity, trigger or not). The meta carries no trigger key before G4; they are written FIRST
  (`{**trigger, **{k: v for k, v in meta.items() if k not in trigger}}`) — today's key order exactly. The records exist
  whatever the stream says (the trigger is knowable later than the record — attribution, stamped retroactively, never
  a gate). **Q7: the names stay `triggered_by_*`; on a sub row they mean "the lens's first WVMI-class trigger inside
  the sub's window" (attribution) — a declared MEANING change under LANDMINES "Event Contract Rules" rule 3 (the
  prediction in the commit; every reader — the exporter, the tests; UC1 reads main records only — and every doc in
  it; GLOSSARY names the ROLE). Q8: the stream is §8.5's WVMI-class cadence (confirmed after the review showed it is
  cross-class: confluence = sd-prox incl. var4, counter = var3).** The run.log summary counts per UNIQUE sub from `all_results[i].wvmi_records` with
  `sorted(lenses)` (counting lens-df records would double the dual-lens subs: 23).
- **Unchanged:** the main's gate, records, values and meta; every sub record's FB / LB / FP / LP / momentum / `status`
  except where `end − 1` bites (0 on the window; today's 8 records measured identical); the charts (nothing draws WVMI
  — the M15 chart collects `sid_wvmis` and never renders them); the UC1 main-trigger gate (reads main records only).

## 5. Predicted `/compare` (vs the Post-E·5 save; values from §2's shadow on the capped frame = G1's frame)

Diff the three WVMI CSVs KEYED by `(sub_id, bos_structure_id, bos_cycle_id)` and by column NAME (`cmp_save.py` is
positional: the 10 inserted rows and the mid-row column would misreport every later cell); also check per row that the
`structure_path_id` column == `meta["structure_path_id"]`, and that each key appears once per lens CSV.

| CSV | Rows | Change |
|---|---|---|
| H1 `wvmi.csv` | 3 → 3 | + column `cycle_collapsed` after `lp_locked`; (1,1) `True`, (0,1) / (1,2) `False` ((1,2)'s end is None); nothing else |
| M15 confluence `wvmi.csv` | 7 → 17 | + column; + 10 rows (sub 0 c0 c1 c3 c4, sub 1 c0 c1 c3 c4, sub 2 c0 c1 — §2's values, path `H1.main >> M15.confluence`, trigger idx / type None, `parent_path_id` `"H1.main"`; flagged: sub 0 c0, sub 1 c0 + c1), in render order BEFORE sub 3's rows; the 7 existing rows unchanged except the column (sub 3 c0 `True`) — `triggered_by_event_idx` stays `710` / `926` / `1020` (`Int64`) |
| M15 counter `wvmi.csv` | 6 → 6 | + column; sub 3 ×3: path `H1.main >> M15.confluence` → `H1.main >> M15.counter` (column; the meta already said counter), trigger 710 `ZONE_PROXIMITY_TRIGGER` → 871 `SUBSEQUENT_CONFLUENCE_TRIGGER` (column + meta), c0 `True`; sub 7 ×2: path → `H1.main >> M15.counter`, trigger idx / type 1020 `SUBSEQUENT_COUNTER_TRIGGER` → None (`parent_path_id` stays `"H1.main"`); sub 5 unchanged except the column (`954` stays an int) |
| the other 21 CSVs | — | byte-identical |
| figures | — | identical (85/245, 151/124, 294/233) |
| run.log | — | only `[wvmi]` lines: each sub projection prints `[wvmi] total/locked` — sub 0 4/3, sub 1 4/3, sub 2 2/1, sub 3 3/3, sub 4 1/0, sub 5 1/0, sub 6 1/0, sub 7 2/2 — instead of "skipped (parent-event-driven …)" (8 lines); today's 13 sub CREATED / LOCKED lines keep their exact text (the shorter LP range is a prefix of today's and holds its best candle), now inside each projection's block; 10 new CREATED + 7 new LOCKED lines (the CREATED lines' creation-time temp LP is not in the shadow — its content is measured at the replay); the summary `[multi_tf:dual] sub wvmi acted=8 records=18 by_started_by={'first_confluence': 5, 'reversal': 9, 'first_counter': 1, 'subsequent_confluence': 1, 'subsequent_counter': 2} by_lens={'confluence': 17, 'counter': 6}` |

## 6. Blast radius (for the implementation; lens 3's measured test table is the checklist — §8)

Code: `pipeline/orchestrator.py` (the helper, `wvmi`, the stream-mapping helper, the G4 post-pass, docstrings /
prints at ~129–132, 332–334, 356–359, 400, 634–638, 658–659, 679–693, 788, 981), `multitf/pooled_structure_build.py`
(`project_to_window` + :79), `multitf/entity_df_mutation.py` (mirror step 8 sets the field; delete the persister;
:1228, :1278), `multitf/sub_wvmi.py` (delete), `common/types.py` (`WVMIRecord.cycle_collapsed`; :305–306),
`debug/export_wvmi.py` (column + `Int64`). Tests (lens 3's table, §8): DELETE `test_sub_wvmi.py` (port its create /
lock / unlocked / no-CTS cases to the helper with `wvmi="none"`); REWRITE `test_sub_wvmi_per_sub.py` against G4;
`test_wvmi_lp_bound.py` (its MODULE-LEVEL `sub_wvmi` import would fail all 6 at collection; `test_the_sub_sweep_
passes_no_ends` → "the projection passes every cycle end" — Q4, on purpose); `test_plan_e_role_pins.py` (`_PINNED`
lists `multitf/sub_wvmi.py`); `test_event_meta_idx_keys.py` (the persister test goes; the mirror test gains WVMI:
the source untouched, the per-lens field, `_Wvmi` stub + `structure_path_id` / `cycle_collapsed`, the two index-key
guards that skip WVMI today); `test_render_sub_projection.py` (the fixture's capped windows have 0 records — real
pins use `project_to_window` on `_make_double_rewind_data` / `_post_reversal_lure`, values measured in §8);
`test_meta_key_renames.py` (a WVMI row in the exporter test). Docs (every site that becomes false — lens 3's list, §8):
LANDMINES "Sub WVMI is Parent-Event-Driven; `source_kinds` is a Return-Only Filter" (:919–980 — Rule 2, the gate
table, the "Don't" Trap, :975 — INVERTED by G1; Rule 1 stands), "WVMI Constraints" #2 / #4, :394, :629,
:1060–1062, :1847–1850, "WVMI Records Carry Mixed-Coordinate Meta" (still true; `triggered_by_event_idx` may be None);
WVMI_SPEC (Gating :14–17, "Sub entities" :55–147, :158–165, :194–199, :268–289, :311–316 incl. `WVMITracker._records`,
Pipeline Integration, Fields); PART4 §6.3 / §6.5 / §7 table / §8 banner / §8.1 / §8.3–8.6 / §8.7 / §10.1–10.2 /
§17.10 / §17.12 / :1039; GOTCHAS "Sub WVMI is Trigger-Centric, Not Sid-Centric" (SUPERSEDED banner); ARCHITECTURE
:337–344 (stale `lp_status`), :406–411, :479–480, :499–501; GLOSSARY :198 + `cycle_collapsed` / "collapsed cycle";
`engine_v2/CLAUDE.md:24`; the compare skill's column vocabulary / save-format note; review_scripts README
(`wvmi_shadow.py` / `wvmi_gate_shadow.py` → "pre-Plan-G code, ≤ `5fa986b`"); memory
`project_wvmi_lifecycle_deferred.md`. Dated history stays (PART4 §13 steps + §17.11, WVMI_SPEC's 2026-09-20
"Measured consequence" + the REMOVED-2026-05-27 box, the GOTCHAS body, LANDMINES Rule 1's 3d.iii note, PLAN_C/D/E).
Save-format boundary: the WVMI CSVs' new column, rows and per-lens path / trigger values.

## 7. Plan cold review (2026-09-30) — DONE, 3 lenses, 773,816 tokens (§8)

## 8. Cold review record + open questions (2026-09-30)

All three lenses: **SOUND WITH FIXES, 0 BLOCKER.** Lens 1 mechanism / predictions (203,105 tokens): §5's cells hold
(today's 8 records' values and `status` unchanged, row order = render order, per-lens triggers match the streams,
the `cycle_collapsed` set); MAJOR the `triggered_by_event_idx` float rendering (→ G3 `Int64`); MAJOR the shadow
measured the capped frame, not G1's natural-end frame (→ G1 frame); MINOR the exact summary line + count source,
`{**trigger, **meta}` letting a seeded None win (→ G4); NIT keyed diff, per-sub total/locked, full paths (→ §5).
Lens 2 design risks (244,145 tokens): MAJOR M1 the natural-end frame bounds only the temp LP (→ G1); MAJOR M2
`triggered_by_*` changes MEANING on sub rows (→ Q7); m3 the per-lens stream is §8.5's opposite-side cadence (→ Q8);
m4 LANDMINES / WVMI_SPEC sites G1 inverts (→ §6); m5 `parent_path_id` None (→ Q9); m6 the cap-candle record (→ G2,
documented); m7 `check_zone_proximity` feeds var3 / var4 (→ G1); NITs one parameter, the shadows break, join on
`sub_id` (→ G1 / G4 / §6); checked safe: no other `attrs["wvmi"]` writer, deepcopy per lens, locks, callers, the
main path, exceptions. Lens 3 tests / docs (326,566 tokens): MAJOR LANDMINES :919–980 inverted, `test_plan_e_role_pins`
+ `test_wvmi_lp_bound` break outright, no pin on the stream → lens mapping, the rendered-sub fixture has no capped
WVMI record (→ §6 / G4); MINOR the None form, the source-untouched check, the `_Wvmi` stub, the two index guards,
`/compare` tooling, the shadows, `wvmi` validation (→ G1 / G3 / G4 / §5 / §6); a measured test table, new pins with
values (double-rewind sub projection: (0,0) (1,1,3,4) locked by 1, (0,1) (5,7,9,9) locked by 2, (0,2) (10,11,13,14)
temp; lure data floor 5 cap 17: (0,2) LP 14 / pullback 1.0 — the no-ends mutant gives 19 / 0.7; `cycle_collapsed`
floor 9 → [T, F, F], floor 13 → [T, T, F], geometry floor 108 → T kills an `==` mutant; main `lifecycle_floor=13` →
[T, T, F]) and 28 mutants (G1a–e, G2a–f, G3a–d, G4a–i, X1–3, M1) — the implementation's checklist — the full test
table, pins, mutants and doc sites: `plans/plan_g_inputs/cold_review_20260930.md`.

**Asked + DECIDED 2026-09-30 (all as recommended): Q7 (A) keep the names, declared meaning change; Q8 (A) the
WVMI-class stream; Q9 (A) `parent_path_id` always `"H1.main"`; Q10 (A) a lock LP outside the frame falls back to the
temp LP.** The questions as asked:
- **Q7** sub rows' `triggered_by_*`: no gate means the record exists whatever the trigger — the keys become an
  attribution ("the lens's first class trigger inside the sub's window"), not the initiating event §8.7 / LANDMINES
  :975 define (main rows keep the gate meaning). (A) keep the names, declared as a rule-3 MEANING change (prediction in
  the commit, every reader + doc in it; GLOSSARY names the role, like `probe_input_idx`); (B) rename on sub rows.
  Example: sub 3 counter c0, created at M15 2752, "triggered by" H1 871 (M15 3487) — 735 candles and two cycles later.
- **Q8** which trigger a lens's rows name: (A) the §8.5 WVMI-class stream (as decided in Q5 — confluence = sd-prox
  class incl. var4 `subsequent_counter`, counter = CTS-prox class = var3 `subsequent_confluence`; sub 7 counter →
  None, sub 6 → `SUBSEQUENT_COUNTER` 1020, sub 5 → `SUBSEQUENT_CONFLUENCE` 954); (B) the lens's own first
  TriggerRecord — the trigger that put the sub on that lens (sub 7 counter → `subsequent_counter` 1020, sub 6 →
  `subsequent_confluence` 954, sub 5 → `first_counter` 926; never None).
- **Q9** `parent_path_id` on a no-trigger row: (A) always `"H1.main"` (the parent entity, not the trigger); (B) None
  with the other two.
- **Q10** a lock LP past a capped sub's end (BOS_{n+1}'s last wave candle ≤ its anchor + 5 can pass the cap; 0 on the
  window): (A) fall back to the temp LP (bounded ≤ end − 1 — the lock already does this when BOS_{n+1} has no last wave
  candle); (B) today's behaviour: keep the idx, volume / pullback None.

## 9. Landed (2026-09-30)

One atomic commit (code + tests + every §6 doc site; the §5 prediction in its message). Implemented exactly as §4:
`_run_downstream_pipeline(wvmi=...)` validated against `WVMI_MODES`; ONE helper `_compute_wvmi_records` (main +
every sub projection; frame `df.iloc[:cap + 1]`, ends + `cycle_collapsed` from the lifecycle table, the capped-record
assert); `_first_sd_prox_gate`; Q10 in `WVMITracker.on_bos_confirmed` (a lock LP not in the frame → the temp LP, for
the main's data edge too); the mirror sets the copy's FIELD path; `_wvmi_trigger_streams_by_lens` +
`_stamp_sub_wvmi_trigger_meta` + `_count_sub_wvmi` replace `_assign_sub_wvmi_per_sub`; `multitf/sub_wvmi.py` and
`persist_facade_wvmi_to_entity_df` deleted; the exporter's `cycle_collapsed` column + `Int64`. One interpretation
stated before coding: `check_zone_proximity` runs iff `wvmi == "first_sd_prox"` (the main path, exactly as before —
`skip_wvmi=True` callers skipped it too).

**Verification (reference window, vs the Post-E·5 save `20260930_005619_5af674c`):** replay 45 s wall; the keyed WVMI
diff (`review_scripts/cmp_wvmi_keyed.py`) == §5 in every cell (H1 3 rows + the column, (1,1) `True`; confluence
7 → 17 rows with §2's values in render order, the 7 old rows unchanged but the column, `710` still an int; counter 6
rows: sub 3 path + trigger 871 `SUBSEQUENT_CONFLUENCE_TRIGGER`, sub 7 path + trigger None, sub 5 unchanged); every row
path column == meta path, every key once per CSV; the other 21 CSVs byte-identical; figures identical (85/245,
151/124, 294/233); run.log: warnings identical, outside the `[wvmi]` lines only the summary line (== §5 verbatim),
the 8 per-projection totals == §5, today's 13 sub CREATED / LOCKED lines verbatim inside their blocks, + 10 CREATED /
7 LOCKED. Tests 1087 → 1111 (+ the OANDA smoke). My mutation loop: `review_scripts/mutants_plan_g.py` 47/47 KILLED
(4 min).

**Where the implementation differed from the checklist (re-measured on the landed code):** the lure LP pin (cap 17)
cannot kill the no-ends mutant on the capped frame — `iloc[:18]` excludes the lure candle 19 — so it runs on an OPEN
projection (floor 5: LP 14, the mutant 19); a new pin (an FP ON the cap candle is read) kills an `iloc[:cap]`
off-by-one; the call-site mutants (the var3 / var4 swap, counts fed per-lens results, the parent path from a lens)
survived every unit pin and die only in a stubbed-collaborator driver test
(`test_sub_wvmi_per_sub::test_the_driver_feeds_each_lens_its_stream_and_counts_each_unique_sub_once`); a
`project_to_window` == bounded-build WVMI equality pin was VACUOUS on its fixture (both empty) and was not added.
Found on the way, not in scope (memory `_INBOX.md`): `locked_by_cycle_id` renders `1.0` in every WVMI CSV (a
None-mixed int column, pre-existing — the save has it too); `lp_idx` / `fp_idx` would likewise the first time one is
None.

**Landing review (2026-09-30, 2 read-only lenses in the main checkout, 515,928 tokens):** conformance (284,441
tokens) — CONFORMS WITH FIXES, 0 BLOCKER / MAJOR: MINOR the exporter still read the two trigger keys with `.get`
although this commit declares a rule-3 meaning change on them (an unstamped sub copy would export as "no trigger");
MINOR WAVE_CANDLES_SPEC :218 / :287 called every WVMI record sd-prox-gated; NITs an uncapped record without a
lifecycle row raised a bare KeyError, no guard on a `sub_id` twice on a lens, stale lines in FIB_LIFECYCLE_SPEC / PART4
(the M15 hover claim — no chart renders WVMI), the int-meta guard never sees a stamped record (covered by
`TestMetaShape`'s exact key list — no change). Mutation (231,487 tokens) — 30 new mutants, 21 SURVIVED (6 equivalent):
MAJOR every G4 test stubbed the LOH mapper with a lambda ignoring both frames, so `parent_df` / `m15_df` swapped (in the
post-pass or at the call site) survived; MAJOR the main gate's meta could come from a LATER sd trigger (UC1 copies
`triggered_by_event_idx` into first_counter's `trigger_idx`); MINOR window edges tested only 4 candles apart, the row
order of a sub's records (render hand-off + mirror; the fixtures hold 0 / 1 record), a lock LP ON the cap candle;
NITs the summary's key order, the creation loop's event order, exporter columns (`structure_path_id` from the field,
`parent_path_id` without a default), per-record meta aliasing.

**Fold-in (byte-identical, one commit):** the exporter reads `triggered_by_event_idx` / `_type` strictly; the helper
asserts a lifecycle row for every record (the capped-end assert it replaces was implied by the table's cap term);
the post-pass asserts a `sub_id` once per lens; docs (WAVE_CANDLES_SPEC, FIB_LIFECYCLE_SPEC, PART4 §6.3 / §16.5 WVMI
row / dated §17.10 caveat, WVMI_SPEC, LANDMINES); pins for every real survivor (the REAL mapper on aligned frames — a
unit pin and the driver pin —, one-candle window edges, the first sd's gate meta, record / key order, the cap-candle
lock LP, the processing order, exporter columns, distinct metas, the two asserts, the strict read). The review's
mutants joined `review_scripts/mutants_plan_g.py` (equivalents listed, not run): **73/73 KILLED** after the fold-in
(8 min; my own F2 — the type read back to `.get` — first survived: the unstamped-record pin lacked BOTH keys, so the
idx read raised first; pinned each key on its own). Replay after the fold-in: 24/24 CSVs byte-identical to `83c7a12`'s, figures identical.

**Int64 (the user's decision at the chart pause, 2026-09-30: "sure we can apply the int64 fix"; its own commit and
`/compare`):** `export_wvmi` writes EVERY integer column as nullable `Int64` (`_INT_COLUMNS`: `parent_sid`,
`parent_cycle_id`, `sub_id`, `bos_structure_id`, `bos_cycle_id`, `locked_by_cycle_id`, `fb_idx` / `lb_idx` / `fp_idx` /
`lp_idx`, `triggered_by_event_idx`). Prediction == measured: 18 cells, every locked row's `locked_by_cycle_id`
`N.0` -> `N` (H1 1, confluence 12, counter 5); nothing else (the all-empty and all-int columns print the same); the
other 21 CSVs + figures identical. Pin: every int column prints `7` next to a None row, the int column set WRITTEN OUT
in the test (my first pin read the exporter's own list and shrank with it — 3 of 6 `I*` mutants survived); 6/6
killed after the fix.

