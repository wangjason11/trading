# Plan G — WVMI on the unique sub (lifecycle-governed, exported like zones)

Status: **MEASURED + OPTIONS (2026-09-30, HEAD `896ffd4`); decisions pending; no code yet.** Heavy tier: options →
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

## 4. Design (as decided; for the cold review)

- **G1 — sub WVMI is computed INSIDE the unique sub's projection, like KL / POI / fib.** `_run_downstream_pipeline`
  gains `wvmi_gate: str = "first_sd_prox"` (the main, unchanged) | `"none"` (a sub: no zone-proximity check, every
  CTS_CONFIRMED is offered to the tracker); `skip_wvmi=True` keeps meaning "no WVMI at all" (the tests that pass it).
  `project_to_window` passes `skip_wvmi=False, wvmi_gate="none"`. The tracker loop (today duplicated in the main
  branch and in `sub_wvmi.compute_parent_driven_sub_wvmi`) becomes ONE helper; both paths build `cycle_end_by_key`
  from the same `compute_cycle_lifecycle` table KL / POI read (the sub's floor / cap) — **Q4: the sub LP search stops
  at `end − 1`**. `render_sub_projection` already carries `down["wvmi_records"]` into `LowerTFResult.wvmi_records`
  (today empty). The projection runs `bounded.df` (the natural-end frame): every record's search is bounded by its
  cycle's end, and a capped sub's cycles all end at or before the cap; the events are knowable-at-clipped at the cap as
  today, so locks are unchanged.
- **G2 — `cycle_collapsed` (Q2 / Q6).** `WVMIRecord.cycle_collapsed: bool = False`, set by the helper from the same
  table: `start >= end` (end not None) — the criterion `_zone_render.is_collapsed_cycle_zone` applies to the cycle's KL
  zone. Main AND subs. Exported as a column right after `lp_locked`. No start / end / reason fields (the WVMI-record
  lifecycle removed 2026-05-27 stays removed — a flag only).
- **G3 — export like zones (Q3).** The mirror (`mirror_lower_tf_result_to_entity_df` step 8, which already copies
  `result.wvmi_records` into EVERY lens df with that lens's attribution) persists them; it now also sets the record
  FIELD `structure_path_id` to the lens path (today only the meta — the self-contradicting counter rows).
  `persist_facade_wvmi_to_entity_df` and `multitf/sub_wvmi.py` (`ParentTrigger`, `compute_parent_driven_sub_wvmi`)
  are deleted (dead; kept they would persist every record twice).
- **G4 — trigger metadata per lens (Q1 "triggers as metadata", Q5).** `_assign_sub_wvmi_per_sub` becomes a
  post-pass over each lens df's `attrs["wvmi"]`: for a record of sub `s` on lens `l`, the FIRST entry of lens `l`'s
  stream (`_confluence_trigger_stream` / `_counter_trigger_stream`, LOH-mapped, unchanged) inside `[start_idx,
  m15_end_idx]` gives `triggered_by_event_idx` (parent coords) / `triggered_by_event_type` / `parent_path_id`
  (`"H1.main"`); no entry → all three None. Written FIRST in the meta dict (`{**trigger, **meta}`), today's key
  order. The run.log summary line keeps its shape (`acted` = subs with >= 1 record).
- **Unchanged:** the main's gate, records and values; every record's FB / LB / FP / LP / momentum except where
  `end − 1` bites (0 on the window); the WVMI CSV exporter apart from the new column; charts (nothing draws WVMI —
  the M15 chart collects `sid_wvmis` and never renders them); the UC1 main-trigger gate (reads main records only).

## 5. Predicted `/compare` (vs the Post-E·5 save; every value from §2's shadow)

| CSV | Rows | Change |
|---|---|---|
| H1 `wvmi.csv` | 3 → 3 | + column `cycle_collapsed` (header); (1,1) `True`, (0,1) / (1,2) `False`; nothing else |
| M15 confluence `wvmi.csv` | 7 → 17 | + column; + 10 rows (sub 0 c0 c1 c3 c4, sub 1 c0 c1 c3 c4, sub 2 c0 c1 — §2's values, `M15.confluence`, trigger fields None; flagged: sub 0 c0, sub 1 c0 + c1), in render order BEFORE sub 3's rows; the 7 existing rows unchanged except the column (sub 3 c0 `True`) |
| M15 counter `wvmi.csv` | 6 → 6 | + column; sub 3 ×3: `structure_path_id` column `M15.confluence` → `M15.counter`, trigger 710 `ZONE_PROXIMITY_TRIGGER` → 871 `SUBSEQUENT_CONFLUENCE_TRIGGER` (column + meta), c0 `True`; sub 7 ×2: path → `M15.counter`, trigger 1020 `SUBSEQUENT_COUNTER_TRIGGER` → None (idx / type / `parent_path_id`); sub 5 unchanged except the column |
| the other 21 CSVs | — | byte-identical |
| figures | — | identical (85/245, 151/124, 294/233) |
| run.log | — | only `[wvmi]` lines (the 8 sub projections print `[wvmi] total=… locked=…` instead of "skipped"; sub CREATED lines 8 → 18 and LOCKED 5 → 12, now inside each projection's block) + the `[multi_tf:dual] sub wvmi …` summary (acted 5 → 8, records 8 → 18; its by_lens / by_started_by recounted) |

## 6. Blast radius (for the implementation)

Code: `pipeline/orchestrator.py` (the helper, `wvmi_gate`, the G4 post-pass), `multitf/pooled_structure_build.py`
(`project_to_window`), `multitf/entity_df_mutation.py` (mirror step 8 sets the field; delete the persister),
`multitf/sub_wvmi.py` (delete), `common/types.py` (`WVMIRecord.cycle_collapsed`), `debug/export_wvmi.py` (column).
Tests: `test_sub_wvmi.py` + `test_sub_wvmi_per_sub.py` (rewritten against G1 / G4), `test_wvmi_lp_bound.py`
(`test_the_sub_sweep_passes_no_ends` → "the sub projection passes every cycle end" — Q4, on purpose),
`test_event_meta_idx_keys.py:545` (the persister call → the mirror), `test_render_sub_projection.py`, new pins
(values of an ungated sub, the flag on a collapsed cycle — main and sub —, per-lens trigger meta incl. the None case,
the field == meta path on a dual-lens sub, no double persistence). Docs: WVMI_SPEC ("Sub entities", Pipeline
Integration, Fields), PART4 §17.10 + the §8.3–8.5 notes, GOTCHAS "Sub WVMI is Trigger-Centric, Not Sid-Centric"
(superseded), LANDMINES "WVMI Records Carry Mixed-Coordinate Meta" (still true for `triggered_by_event_idx`),
GLOSSARY (`cycle_collapsed`), memory `project_wvmi_lifecycle_deferred.md`. Save-format boundary: the WVMI CSVs' new
column + rows.

## 7. Plan cold review (before implementation)

3 lenses, read-only, against this file + the code at HEAD: (1) **mechanism / predictions** — re-derive §5 from §2's
shadow and the code, hunt any row or cell the design would move that §5 does not list (mirror order, meta key order,
the natural-end frame in the projection); (2) **design risks** — double persistence, the knowable-at clip vs the
tracker's frame, a sub cycle with no table entry, the lock across the cap, the main path untouched; (3) **tests /
docs** — every pin that encodes the old sweep, what the new pins must kill (value / swap / extra-key / exporter
shapes). Estimate ≈ 150–250k tokens per lens (≈ 0.5–0.75M total) — asked before launch.
