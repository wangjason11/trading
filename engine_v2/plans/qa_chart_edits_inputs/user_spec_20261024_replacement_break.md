# SESSION ORDER item 5 — chart updates & edits: a replaced sub's line on the lens that did not end it (the user, 2026-10-24, verbatim)

Context: the first edit of item 5's second pass, "the charts' other elements" (the hover-label pass closed 2026-10-24),
on S_prox_hover `20261024_105742_fb0189b` = the `/compare` baseline.

## The user's request (verbatim)

> I previously implemented some additional logic that covers how we would connect the structure straight lines from
> the end of a structure to the beginning of the next structure. Now that we've made some changes to proximity
> triggers which resulted in new subs created, some structure lines are a bit off again, and I'd like to see what
> change we need to handle these exceptions.
>
> Specifically, I'm looking at sub_3 where the dotted line connects idx 3047 to 3490, where the sub ends. What I like
> the line to do instead is go from 3047 to 3304 then down to 3490. Previously I think we did have some logic
> implemented that governs how a line like this would end (I may need a refresher on exactly what that was, but I
> think it was along the lines of connecting the line to the opposite direction sub's CTS or something like that).
> So I'm wondering how that logic was not able to do this here, and or if we need to update or add something to get
> the right line behavior.

## The existing rule (the refresher)

CHARTING_SPEC "Replaced subs run through one more point" (chart review 2026-09-22b, `export_m15_chart.
_replacement_break_point`): a sub whose `end_reason` is `same_dir_replacement` breaks its final segment at the
counter-move extreme (a `−1` sub → the highest high, a `+1` sub → the lowest low) over `(its last point, the REPLACING
sub's anchor]`, drawn with a PB dot. Replacement only — a reversal-ended sub's line already ends at its extreme, a
`parent_end` one at the parent's candle. It does NOT read the opposite-direction sub: here the break candle 3304 is
also sub 4's (`3230/+1`) CTS, a coincidence of the same swing high.

## Why it did not fire (measured on S_prox_hover, both M15 charts)

Sub 3 = `2639/−1`, window `[2829, 3491]`, `end_reason=same_dir_replacement` (pool-wide), replaced by sub 5 =
`3304/−1` (live from 3491). The rule finds the replacer through THIS LENS's TriggerRecords — the record with
`end_reason=same_dir_replacement` AND `end_idx ==` the sub's end:

| lens | sub 3's record | record end_idx | end_reason | ended_by_sub_id | break |
|---|---|---|---|---|---|
| counter | first_counter @2995 | 3491 (= the sub's end) | same_dir_replacement | 5 | 3304 @ 0.58105 — drawn |
| confluence | reversal @2829 | 3589 (≠ 3491) | same_dir_replacement | 5 | none — the record is skipped |

The sub's end (3491) is pool-wide — the counter record set it — but the lookup is lens-local, so on the confluence
chart the line runs 3047 → 3490 straight (dotted), while the counter chart already draws 3047 → 3304 (PB) → 3490.
The same mismatch in the other direction: sub 5 (`3304/−1`, ended 3819 by sub 7 = `3760/−1`, a confluence-only sub)
breaks at 3760 @ 0.57806 on confluence, but on counter (its only counter record ends 3611 `parent_end`) runs
3621 → 3819 straight. Candles (the M15 frame decoded from the chart): 3047 low 0.57400 (sub 3's last CTS); highest high
over [3048, 3304] = 3304 @ 0.58105; over [3622, 3760] = 3760 @ 0.57806; 3490 close 0.57556.

## The decision (the user, 2026-10-24)

Options put: A — one replacer per sub (pool-wide; both charts read it; confluence sub 3 and counter sub 5 both break);
B — keep the per-lens lookup without the equal-end check (fixes confluence sub 3 only; counter sub 5 stays straight);
C — leave as is. The user: **"A: one replacer per sub (Recommended)"**.

## What landed

- The sweep's phase 4 keeps the WINNING record's replacer on the sub (`PooledStructure.ended_by_pool_idx` → the final
  `ended_by_sub_id` at numbering — the TriggerRecord pair, now on the sub too); the driver looks the replacer up in the
  pool and hands its anchor to `render_sub_projection` (`ended_by_starting_idx`; asserted: set iff
  `end_reason = same_dir_replacement`); the projection meta and `SidRecord.meta` carry `ended_by_sub_id` /
  `ended_by_starting_idx` (not exported — the subs CSV writes a fixed column list); `_replacement_break_point` reads
  `sid_rec.meta["ended_by_starting_idx"]` (a KeyError on a replaced sub without it — never a silent straight line) and
  no longer takes the lens records or the lens's anchor map.
- Prediction = result: FIGURE-ONLY, 27 / 27 CSVs md5-equal to S_prox_hover; H1 81 / 234 unchanged; counter 153 → 154
  traces (+ `M15 PB sub5 (prior)` 3760 @ 0.57806; sub 5's dotted run 3621 → 3760 → 3819), confluence 301 → 302
  (+ `M15 PB sub3 (prior)` 3304 @ 0.58105; sub 3's dotted run 3047 → 3304 → 3490); shapes unchanged (136, 307); no
  other trace changed (`fig_named_diff.py`, name-keyed); `chart_census.py` vs S_prox_hover: exactly `dots_PB` +1 on
  (0,1,5) counter and (0,1,3) confluence. Both new PB dots are hollow (prior): every segment around them overlaps a
  more recent sub (sub 4's 3230 → 3304, sub 5's 3304 → 3621; sub 6's 3621 → 3760 → 4000). Skip grep 23.
- Tests: suite 1972 → 1977; mutants 14 / 14 KILLED (`mutants_replacement_break.py`).
- The user's chart look (2026-10-24): **"charts look good go ahead"**.
- Tools (this folder): `m15_frame.py` (the M15 candle frame decoded from a chart — the saves carry no M15 CSV),
  `sub_lines.py` (a sub's drawn points as `idx@price` on both charts), `fig_named_diff.py` (a name-keyed figure diff
  for a change that adds / removes a trace), `zoom_png.py` (a zoomed PNG for a chart look).
