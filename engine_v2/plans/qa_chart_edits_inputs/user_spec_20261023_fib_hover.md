# SESSION ORDER item 5 — chart updates & edits: the Fib zone hover (the user, 2026-10-23, verbatim)

Context: the third chart edit, after the BOS / CTS dot hover (`user_spec_20261022_dot_hover.md`, `ab2fcc1`) and the
KL / POI zone hover (`user_spec_20261023_zone_hover.md`, `56bb20d`, save S_zone_hover `20261023_153743_56bb20d` = the
`/compare` baseline). Before the spec the user asked what the fib hover's `IC ref …` and `limit …` lines log (answered
from POI_ZONES_SPEC §3.2 and CROSS_CYCLE_FIB_SPEC §3 with the window's rendered values; the user: "I think everything
else I have a good understanding for FIb Zone hover data").

## The user's spec (verbatim)

> OK here are the changes I'd like to make. Please see below. This also standardizes Fib zones to be consistent across
> main and sub. There are also some fields I chose to remove since they relate more to MS and IC rather than
> specifically Fib Zones. Those I think we don't need since they are already labeled in other structure elements that
> are more relevant.
>
> Main
> TF=x
> Fib Zone
> sid=x | cycle_id=x ← updated format
> BOS: …
> CTS: …
> ic_ref: price=x (prev_cycle_cts, cycle x) | before=x ← updated formatting & naming
> limit_price=x | <source> [(sub <x>)] | cycle_limit=<cycle | none> ← updated formatting & naming
> status=x
> lifecycle=[x,y] ← updated naming from life to be consistent with other labels
> 61.8%=x
> 80%=x
> struct_direction=x ← new
>
>
> Sub
> TF=x
> Fib Zone
> sub_id=x | cycle_id=x
> first record: lens=x | parent_sid=x | parent_cycle_id=x
> BOS: …
> CTS: …
> ic_ref: price=x (prev_cycle_cts, cycle x) | below=x
> limit_price=x | <source> [(sub <x>)] | cycle_limit=<cycle | none>
> status=x
> lifecycle=[x,y]
> 61.8%=x
> 80%=x
> struct_direction=x
>
> Please let me know if anything is unclear or doesn't make sense. Or if you have any questions or confusions in
> general. We can discuss.

## Today's hover (S_zone_hover, the current charts — 27 fib hovers: H1 3, counter 7, confluence 17)

TF / **Fib Zone** / main `sid=` + `cycle=` (two lines), sub `sub_id=X | first record parent_sid=… parent_cycle_id=…` +
`sub window=[s,e] reason` + `records: …` + `cycle_id=` / `BOS: idx= price=` / `CTS: idx= price=` / `61.8%=` / `80%=` /
`status=<status>[ (locked)]` / `life=[start, end] reason` (`–` = never started, `open`) / the IC outcome line / the IC
reference line / the limit line — one status-cell builder for both charts, `_zone_render.fib_hover_status_text`. No
fib on the M15 charts' H1 overlay.

- `status`: ended (locked) 22, active 2, active (locked) 1, inactive (locked) 2.
- `life`: `[–, 4140] new_cycle` 2 (sub 8 cycle 0, both lenses), open 3, ended `new_cycle` 11 / `reversal` 6 /
  `same_dir_replacement` 5.
- The IC outcome line: a pick 24 (`IP <rank> · IC <idx>` 16, `No pattern · IC <idx> · …` 8; `· zone 80-100
  (fallback)` on 4); NO IC 3 — H1 sid 0 cycle 0 `IC: none (all_ic_candidates_pinbar)`, confluence sub 1 cycle 3
  `IC: none (no_ic_candidate) · zone 80-100 (fallback)`, confluence sub 2 cycle 0 `IC: none (all_ic_candidates_pinbar)
  · zone 80-100 (fallback)`; `IC: not evaluated (no POI record)` 0. A fib with no IC has no POI, so this line is the
  chart's only place for its reason.
- The IC reference line: `prev_cycle_cts` 18, `cross_limit_anchor` 7 (`break: none` 5), `prev_structure_bos` 1,
  `IC ref: none` 1 (H1 sid 0 cycle 0).
- The limit line: `limit: none (start_trading)` 2 (H1 sid 0), `limit: none (first_confluence on parent cycle 0)` 4
  (confluence sub 0); with a price: `reversal_bos_frozen` 1 + `(sub 1)` 2, `parent_bos_threshold` 16,
  `prev_sub_bos_threshold (sub x)` 2; `M`: 0 / 2 / none; no moved limit on this window.

## Questions raised back (2026-10-23)

1. The IC outcome line on a fib with NO IC (3 on the window; no POI exists for them) — drop everywhere as written, keep
   it only there, or keep it on every fib?
2. A structure with NO limit (6 on the window) — `cycle_limit=none` there would read "every cycle may cross" while no
   cycle may (CROSS_CYCLE_FIB_SPEC §2: no `P` → a single); its form?
- Read, not asked: `below=` on the sub's `ic_ref` line read as `before=` (main's; the value is a candle index);
  `ic_ref: NA (no reference)` / `| before=NA (no break)`; `lifecycle=[s, e] | end_reason=…` like the zones
  (`[x, open] | end_reason=open`, `[NA (never active), e]`); `status=` kept with its ` (locked)`; `BOS:` / `CTS:` kept
  as today; a moved limit `limit_price=<P> (moved N× from <start>) | …`.
- Flagged, not in scope: the fib's `end_reason=new_cycle` (the fib stream's terminal) and the zones' `next_cycle` (the
  cycle table, `compute_cycle_lifecycle`) name nearly the same end — an engine value, renaming it is a meta-contract
  migration (LANDMINES "Event Contract Rules" rule 3), not a hover change.

## Decisions (the user, 2026-10-23, AskUserQuestion — each the recommended option)

1. **The IC outcome line stays only on a fib with NO IC** (A): `ic_idx=NA (<reason>)` (`all_ic_candidates_pinbar` /
   `no_ic_candidate`) + ` | ic_zone=80-100 (fallback)` when Step 1A also searched 80–100; `ic_idx=NA (not evaluated:
   no POI record)` for a fib the POI record gate skipped. Placed — the user: "Let's put it under lifecycle". A fib with
   a POI drops the line (its POI hover carries `ic_idx` / `ic_reason`). Not B (dropped everywhere: the reason only in
   `*_ic_candidates.csv`) nor C (on every fib: repeats the POI hover on 24 fibs).
2. **No limit → `limit_price=NA (<why>)` only** (A) — no source, no `cycle_limit` (neither applies; `cycle_limit=none`
   with a limit = every cycle may cross, with none = no cycle may). Not B (`| cycle_limit=NA (no limit: no cross
   fib)`) nor C (the template literally, `limit_price=NA | none | cycle_limit=none` — reads the opposite).
- The "read, not asked" defaults above stand (the user did not object).

## Prediction (written before the replay, vs S_zone_hover `20261023_153743_56bb20d`)

Figure-only: 27 / 27 CSVs md5-equal; traces / shapes / layouts unchanged (81/234, 153/136, 301/307); the hover
changes on the fib hover traces only, every one of them — H1 3 (`fib:sid0:c0`, `fib:sid0:c1`, `fib:sid1:c2`), counter
7 and confluence 17 (`M15 fib`); `fig_scope.py BASE CUR '^(fib:|M15 fib$)'` → SCOPE OK.

## Built 2026-10-23 (figure-only)

Builders (`_zone_render`): `fib_hovertemplate(tf)` + `fib_hover_customdata` (slots: 0 the identity line(s), 1–4 the
anchors, 5 `<br>`ic_ref + `<br>`limit, 6 status, 7 lifecycle [+ `<br>`the no-IC line], 8–9 the levels, 10
`struct_direction`) — ONE template for the H1 chart and the M15 lens charts; `fib_hover_lifecycle_text` (asserts
`end_idx` None ⇔ `end_reason` None; `NA (never active)` = `HOVER_NEVER_ACTIVE`), `fib_hover_no_ic_text`;
`fib_hover_ic_ref_text` / `fib_hover_limit_text` relabelled; `fib_hover_status_text` removed. Sub identity lines =
`export_m15_chart._sub_zone_id_lines` (the zones'); `sub_records_str` removed (its last reader); `sub_window_str` stays
for the wave-candle hover. Canonical description: CHARTING_SPEC §8 "Hover"; POI_ZONES_SPEC "Hover Data";
CROSS_CYCLE_FIB_SPEC §3 "Exported". Tests: `test_fib_chart_render.py` (the label order on a real pipeline, the
lifecycle / no-IC lines, the asserts, three window fibs rendered in full) + the re-pinned strings of PLAN_H / I / K /
M / P / S (PLAN_S's fib-hover narrowing test → the POI hover alone names the preference).
Prediction = measured vs S_zone_hover: 27 / 27 CSVs md5-equal; traces / layouts unchanged (81, 153, 301); the hover
changed on the fib traces only — H1 3 (`fib:sid0:c0` / `c1` / `fib:sid1:c2`), counter 7, confluence 17 (`M15 fib`)
— SCOPE OK; every one of the 27 rendered hovers carries the 11 + identity lines, the `ic_idx` line on 3. Skip grep 23,
fetch 4228. Suite 1963 (1959 + 4).
The user's chart look (2026-10-23, verbatim): "looks good go ahead".
