# SESSION ORDER item 5 — chart updates & edits: the wave candle hover (the user, 2026-10-23, verbatim)

Context: the fourth chart edit, after the BOS / CTS dot hover (`user_spec_20261022_dot_hover.md`, `ab2fcc1`), the
KL / POI zone hover (`user_spec_20261023_zone_hover.md`, `56bb20d`) and the Fib zone hover
(`user_spec_20261023_fib_hover.md`, `563f5f8`, save S_fib_hover `20261023_191547_563f5f8` = the `/compare` baseline).

## The user's spec (verbatim)

> Ok now let's move on to wave candle labels. Below are the updated labels I want:
>
> Main
> TF=x
> Wave Candle
> idx=x
> BOS/CTS zone: sid=3 | cycle_id=2
> Role: …
> struct_window=[start, end]
> volume: x
> struct_direction=x
>
> Sub
> Wave Candle
> TF=x
> idx=x | idx_1H=x
> BOS/CTS zone: sub_id=3 | cycle_id=2
> first record: parent_sid=x | parent_cycle_id=x
> Role: …
> sub_window=[start, end] | end_reason=reason
> volume: x
> struct_direction=x
>
> Let me know if you have any questions or need any clarifications. An additional item around wave candles I believe
> we deferred was "adding WVMI back" to the wave candle lines. Do you have more details around this item and exactly
> where this WVMI was previously?

## Today's hover (S_fib_hover; one builder for the zone + role lines, `wave_candles.wave_candle_hover_lines`)

- H1 chart (11 lines) and the M15 charts' H1 overlay (11 each): `TF=1H` / **Wave Candle** / `idx=` / `<BOS|CTS> zone:
  sid=S cycle=N` / `Role: <FB|LB|FP|LP> (<name>)` / `Volume: <v>` — e.g. `idx=651` / `CTS zone: sid=0 cycle=1` / `Role:
  LB (last breakout)` / `Volume: 3032`.
- M15 sub lines (counter 22, confluence 57): `TF=15M` / **Wave Candle** / `idx=<i>  idx_1H=<h>` / the zone line with
  `sub_id=` / the role line / `first record parent_sid=… parent_cycle=…` / `sub window=[s,e] reason` / `Volume:` — e.g.
  counter sub 3: `idx=2841  idx_1H=710` / `CTS zone: sub_id=3 cycle=1` / `Role: LB (last breakout)` / `first record
  parent_sid=0 parent_cycle=1` / `sub window=[2829,3491] same_dir_replacement` / `Volume: 758`.
- The first record's lens differs from the chart's on 3 subs: sub 3 on the counter chart (first record confluence),
  subs 5 and 8 on the confluence chart (first record counter). The main dot hover's window line:
  `struct_window=[115, 902] | end_reason=reversal` (sid 0), `struct_window=[902, open] | end_reason=open` (sid 1).

## "Adding WVMI back" — what it was (answered 2026-10-23, from git `e64556a` and the docs)

- Until 2026-05-28 every wave-candle hover (H1, M15 sub lines, the H1 overlay) carried `Weighted vol: <volume × the
  WVMI weight>` (FB / FP weight 1, LB × `lb_weight`, LP × `lp_weight`), and on an LB or LP line, after `---`:
  `<Buy|Sell> momentum: <x.xxxx>` (LB: `breakout_momentum`; LP: `pullback_momentum`) + `FB wt vol` / `LB wt vol` (LB)
  or `FP wt vol` / `LP wt vol` (LP). Looked up by candle index (`wvmi_by_idx`).
- Removed by `e64556a` (2026-05-28, "Wave-candle chart rendering: per-candle lifecycle gating + WVMI lifecycle
  removal" — "Chart simplifications: drop momentum hover + Weighted vol"); WAVE_CANDLES_SPEC "Rendering
  simplifications": "Full WVMI momentum markers deferred (revisit later)"; PART4 §16.5's WVMI row: "The planned design:
  locked records show final values via hover; in-progress records re-render whenever `update_temporary_lp` shifts the
  temp LP". NOT in the register until 2026-10-23 (a line added to §G).
- Measured on S_fib_hover (`wvmi_cover.py`, scratchpad): WVMI records exist only for proximity-triggered cycles on H1
  (3: sid 0 c1, sid 1 c1 collapsed, sid 1 c2) and per sub cycle on M15 (counter 7, confluence 18). WVMI's `lp_idx` is
  its own TEMPORARY last pullback, not the drawn LP line: on H1 neither non-collapsed record's LP (896, 1055) is a drawn
  line, so the old idx lookup would show pullback momentum on no H1 line; on M15 some LPs coincide with a drawn line,
  some with another sub's or role's line (sub 5 FB 3300 = a drawn FP on confluence) — an idx-only lookup can attach
  the wrong record. A design question for its own change.

## Questions raised back (2026-10-23)

1. The sub's `first record:` line — the spec has no `lens=`; the dot / zone / fib hovers show `first record:
   lens=… | parent_sid=… | parent_cycle_id=…` (one line, `_sub_dot_hover_lines`). Keep `lens=` or drop it here?
2. The main `struct_window=[start, end]` — no `| end_reason=` in the spec, while the sub's window has one and the main
   dot hover shows `struct_window=[115, 902] | end_reason=reversal`. Add it or not?
3. `volume: x` (a colon, as written) or `volume=x` (the labelling rule's `name=`)?
- Read, not asked: the sub's `Wave Candle` before `TF=` read as a slip (every hover starts with `TF=`); `Role:`
  unchanged; the H1 lines on the M15 charts take the main format; `struct_direction` = the structure's (a sub's own).

## Decisions (the user, 2026-10-23, AskUserQuestion)

1. **Keep `lens=`** (recommended): the sub's line = `first record: lens=… | parent_sid=… | parent_cycle_id=…`, the dot /
   zone / fib hovers' line (one builder).
2. **Add `| end_reason=`** (recommended): the main window = `struct_window=[115, 902] | end_reason=reversal`, the main
   dot hover's line (`main_dot_structure_lines`).
3. **`volume=3032`** (the labelling rule's `name=`, not the spec's colon).
- The "read, not asked" defaults above stand.

## Prediction (written before the replay, vs S_fib_hover `20261023_191547_563f5f8`)

Figure-only: 27 / 27 CSVs md5-equal; traces / layouts unchanged (81, 153, 301); the hover changes on the wave-candle
hover traces only (unnamed traces; every one) — H1 11, counter 33 (22 sub + 11 H1 overlay), confluence 68 (57 + 11).
`chart_census.py` (fixed here — it crashed on the KL / POI zone customdata since `56bb20d`, keyed the sub dots by
their hover strings since `ab2fcc1`, and keyed every M15 wave candle `?` since the 2026-09-30 `Role:` line): S_S3
`20261021_130400_7b62a47` vs this run → every key's delta 0 (all four hover edits are figure-only).

## Built 2026-10-23 (figure-only)

`_zone_render.wave_candle_hovertemplate` (the whole hover, ONE builder for the three sites) +
`struct_direction_by_sid` (a structure's direction from its KL zones' meta, asserted one per structure — a
`WaveCandleResult` carries none; read at the chart, so no CSV changes); `wave_candles.wave_candle_hover_lines`'s zone
line `<owner> | cycle_id=N`; the main window = `main_dot_structure_lines`' line, the sub's = `_sub_dot_hover_lines`'
`sub_window=` line, the sub's first record = its `first record:` line; `sub_window_str` removed (its last reader).
`engine_v2/debug/chart_census.py` fixed (every hover format → one `(parent_sid, parent_cycle, sub_id)` string key;
GOTCHAS "A hover-format change breaks the chart-reading debug tools silently"). Canonical description: CHARTING_SPEC
§10; WAVE_CANDLES_SPEC "Rendering simplifications". Tests `test_wave_candles.py` (the zone line re-pinned, the
call-site pin counts both builders, the main + sub hovers in full, `struct_direction_by_sid`).
Prediction = measured vs S_fib_hover: 27 / 27 CSVs md5-equal; traces / layouts unchanged; the hover changed on the
112 wave-candle traces only (H1 11; counter 22 + 11; confluence 57 + 11 — `wc_scope.py`, scratchpad: WAVE SCOPE OK,
no unchanged wave-candle hover); every hover in the spec's order; census S_S3 → this run: 0 deltas on both lenses,
the M15 wave candles keyed per sub. Skip grep 23, fetch 4228. Suite 1966 (1963 + 3).
The user's chart look (2026-10-23, verbatim): "charts look good".
