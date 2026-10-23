# SESSION ORDER item 5 — chart updates & edits: the BOS / CTS marker hover (the user, 2026-10-22, verbatim)

Context: after the audit of the M15 sub dot hover vs S_S3 (`20261021_130400_7b62a47`) — `export_m15_chart.
_render_m15_dots_layer`; the H1 chart's dot hover = `export_plotly` ~:1330–1520 (`TF=1H` / idx / kind / price / sid /
cycle_id / struct_direction; the unconfirmed-CTS and PB dots have no `TF` line); the M15 charts' H1-overlay dots =
`export_m15_chart` ~:2215 (the same 7 lines). The quirks found by the audit (PB `cycle_id=0` hardcoded, PB = replacement
break vs pullback, first record vs the per-lens records line, `time_1H` unused, `layer` constant) — the user: "Let's
come back to it later to address them in more detail."

## The user's spec (verbatim)

> 1. Relative_dir. The "-" labeling confused me without your explanation and I thought that it was a bug. Let's go
> with something more intuitive and understandable like "NA (before start_idx)"
>
> 2. For sub window=[start,end] reason. To be more precise and specific, let's change to
> "sub_window=[start, end] | end_reason=reason".
>
> Adding the | to more clearly delineate the fields. Furthermore, also adding the label "end_reason" makes it more
> clear rather than just giving the reason name
>
> 3. For "records:" the labeling is ambiguous. Without your explanation, I would not have known that this means the
> sub's trigger records on this chart's lens only. Let's change the naming to "len's records". For the data fields, I
> think we should also be more clear and precise. So for each record, this should be the format:
> "lens(parent_sid,parent_cycle) | trigger=trigger_type | trigger_idx=trigger_idx | start_idx=start_idx"
>
> I forget if we agreed to use _dt or _idx, but the label name should be whatever name we agreed on previously.
>
> This adds names to trigger, trigger_idx, and start_idx so that we are more clear on exactly what these values are.
>
> 4. I'd like to reformat the "sub_id | first record parent_sid / parent_cycle_id" and "cycle_id" fields we currently
> have to below:
> "sub_sid=x | cycle_id=x"
> "first record: parent_sid=x | parent_cycle_id=x"
>
> This becomes more intuitive when you group the sub's own cycle number with the unique sub_id. Furthermore, we also
> separate parent first record attribution from unique sub identifier. The data should remain the same, we are just
> changing the appearance.
>
> 5. I also want to add the trigger that initiated each structure to the hover data for BOS / CTS. This would
> actually be for both main and sub charts. For main it would be trading start (or whatever name we chose) and
> reversal. For sub it would be the first record 4 sub triggers (first/subsequent + confluence/counter) plus
> reversal. Furthermore, I would also like to add the trigger dt or idx. So here's what it would look like:
> Main:
> trigger=reversal | trigger_idx (or _dt, whichever we previously agreed on)=idx
>
> Sub:
> fr trigger=reversal | fr trigger_idx (or _dt, whichever we previously agreed on)=idx
>
> fr being shorthand for first record
>
> To summarize, here's how I want BOS/CTS hover data to look
>
> Main
> TF=x
> idx=x
> kind=x
> price=x
> sid=x | cycle_id=x
> trigger=reversal | trigger_idx=idx
> struct_window=[start, end] | end_reason=reason
> struct_direction=x
>
> Sub
> TF=x
> idx=x
> idx_1H=x
> kind=x
> price=x
> sub_sid=x | cycle_id=x
> first record: parent_sid=x | parent_cycle_id=x
> fr trigger=reversal | fr trigger_idx=x
> records: lens(parent_sid,parent_cycle) | trigger=trigger_type | trigger_idx=trigger_idx | start_idx=start_idx
> phase=x | layer=x
> sub_window=[start, end] | end_reason=reason
> relative_dir=x
> struct_direction=x

## Open points raised back (2026-10-22) — answers to be recorded below

- The suffix: `_idx` (GLOSSARY "Naming Standard", 2026-09-23; never `_dt`).
- `sub_sid` vs `sub_id`; sid 0's `start_trading` trigger index; the records label + layout; the first record's lens +
  the repeats (the 2026-10-10 rule: a hover line never repeats a value its hover already shows).

## Decisions

- **Q1 (2026-10-22): `sub_id`**, not `sub_sid` — the field the value comes from; `sub_sid` is the retired pre-pool
  per-parent-cycle identity (today only a main row's structure id on `SidRecord`). Line: `sub_id=x | cycle_id=x`.
- **Q2: `trigger=start_trading | trigger_idx=NA`** — `start_trading` has no trigger candle (the backtest's Scenario 1
  picks sid 0's start once, looking back from the window's last candle 1057; its pick, 96, is sid 0's BOS anchor — a
  location). A reversal-started main structure: `trigger_idx` = the previous sid's reversal candle (sid 1: 902).
- **Q3: the label `lens records:`, one record per hover line** (subs 2 and 5 on confluence have 2 records each).
- **Q4: `first record: lens=<lens> | parent_sid=x | parent_cycle_id=x` + `fr trigger=… | fr trigger_idx=…`, ALWAYS
  shown** — the user asked why first ("Can you explain in more detail why you suggest adding lens? And what's the
  2026-10-10 rule …? Why is that an issue here?"); explained: (1) the lens — `reversal` is the one trigger type that
  does not name its lens; counter sub 3's first record is a CONFLUENCE reversal at 2829 (sub 2 ends there by reversal),
  absent from the counter chart's `lens records:` (the only such case of the 12 sub/chart pairs on S_S3; confluence
  subs 5 / 8 start `subsequent_counter`, which names it); every `lens records:` entry carries its lens; (2) the
  repeat — when the first record is on this lens (9 of 12 pairs) the two lines repeat the `lens records:` first entry;
  accepted as an EXCEPTION to the 2026-10-10 no-repeat rule: different meanings (the sub's pool-wide origin vs this
  lens's list) that coincide, and a fixed hover shape.
- Defaults stated and not objected to — accepted with the user's go ("let's commit this first", 2026-10-22): the main
  format on the M15 charts' H1-overlay dots too (one builder); `struct_window` start = the real-time lifecycle start
  (sid 0: 115) like `sub_window`; every dot kind (CTS, BOS, unconfirmed CTS, PB), the H1 unconfirmed-CTS / PB dots
  gain `TF=1H`; `NA (before start_idx)` exactly on the forming dots (fail loudly otherwise); `†` → `| zero_length`; an
  open window `[x, open] | end_reason=open`; a main reversal's `trigger_idx` kept although it equals the window start;
  PB `cycle_id=0` left for the quirks pass.

## Landed 2026-10-22 (figure-only)

Builders: main — `_zone_render.main_dot_structure_lines` / `main_dot_customdata` / `MAIN_DOT_HOVERTEMPLATE` (the H1
chart's 6 dot trace sites + the M15 charts' H1 overlay); sub — `export_m15_chart._sub_dot_hover_lines` +
`_sub_relative_dir_at` (`_render_m15_dots_layer`). Canonical description: CHARTING_SPEC §1 "Dot hover" (main) and the
M15 "Dot hover" paragraph (sub). Tests `engine_v2/tests/test_structure_dot_hover.py` (8). Verified against S_S3: 27 / 27
CSVs md5-equal; traces 81 / 153 / 301 and layouts unchanged; the hover changed on the dot traces only (H1 5, counter
14 = 11 sub + 3 overlay, confluence 21 = 18 + 3) — `fig_scope.py BASE_DIR CUR_DIR` (this folder); every rendered dot
hover (H1 11, counter 27, confluence 52 dots), `NA (before start_idx)` ⇔ `phase=forming` on all —
`dot_hovers_new.py CHART_DIR` (this folder). Suite 1949.

## Quirks found by the audit — DEFERRED (the user 2026-10-21: "Let's come back to it later to address them in more detail")

1. A PB dot's `cycle_id` is a hardcoded placeholder 0 — sub (`export_m15_chart._build_sub_polylines`: counter sub 3's
   PB at 3304 shows 0, its last CTS is cycle 2) and main (`export_plotly`'s pullback dot: sid 0's PB at 683 shows 0,
   its last CTS 652 is cycle 1).
2. A sub PB dot is either a pullback or a replacement break point, both `kind=PB` — all 3 PB dots on S_S3 are
   replacement break points (subs 3 / 5 / 6, `same_dir_replacement`).
3. `idx` = the dot's ANCHOR candle; the confirmation moment (`confirmed_at`), the `confirmation_method`
   (pullback / sd_zone_proximity) and the pattern (`via`) are not shown; `phase` is judged at the anchor (sub 0's BOS
   anchored 917 hovers `forming` although it was confirmed at 1020 = the sub's start).
4. `time_1H` sits in the sub dot customdata (slot 9) but is never displayed; `layer=recent` on all 63 sub dots at
   S_S3 (the line is constant on this window).
5. The M15 charts' H1 overlay stores an unconfirmed CTS as kind `CTS` (`export_m15_chart._render_h1_overlay`) while
   the H1 chart labels the same dot `CTS (established|updated)` — not reachable on S_S3 (sid 1's last H1 point is a
   confirmed CTS).

Resolved by this change (was an audit quirk): `records:` listed THIS lens's records while `first record parent` was
pool-wide, with nothing saying so — now `first record: lens=… | …` and `lens records:` (Q3 / Q4).
