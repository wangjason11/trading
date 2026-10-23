# SESSION ORDER item 5 — chart updates & edits: the KL / POI zone hover (the user, 2026-10-23, verbatim)

Context: the second chart edit, after the BOS / CTS dot hover (`user_spec_20261022_dot_hover.md`, landed `ab2fcc1`,
save S_dot_hover `20261022_200747_ab2fcc1` = the `/compare` baseline). The user, 2026-10-22: "On the other structure
elements, I will provide additional details and specs on what changes to make there."

## The user's spec (verbatim)

> Ok let's move on now to zones. We won't need to make as many changes here. Below is what I want the updated labels
> to look like:
>
> Main KL
> TF=x
> KL Zone
> side=x
> sid=x | cycle_id=x ← updated to put sid & cycle together
> base_pattern=x
> base_idx=x
> confirmed_idx=x
> lifecycle=[start, end] ← new field that should contain start_idx and end_idx of structure lifecycle
> top=x
> bottom=x
> zone_size=x ← new field taking measuring the distance between top & bottom. Format: xx.x pips
> struct_direction=x ← new
>
> Main POI
> TF=x
> POI Zone
> side=x
> sid=x | cycle_id=x
> ic_idx=x
> confirmed_idx=x
> IC_reason=xx ← more clear labeling
> lifecycle=[start, end] ← effectively can replace status since it provides entire lifecycle information
> top=x
> bottom=x
> zone_size=x
> struct_direction=x ← moved
>
>
> Sub KL
> TF=x
> KL Zone
> side=x
> sub_id=x | cycle_id=x ← updated & copies over labels from what we did for BOS/CTS
> first record: lens=x | parent_sid=x | parent_cycle_id=x ← updated & copies over labels from what we did for BOS/CTS
> base_pattern=x
> base_idx=x
> confirmed_idx=x
> lifecycle=[start, end]
> top=x
> bottom=x
> zone_size=x
> struct_direction=x
>
> Sub POI
> TF=x
> POI Zone
> side=x
> sub_id=x | cycle_id=x
> first record: lens=x | parent_sid=x | parent_cycle_id=x
> ic_idx=x
> confirmed_idx=x
> IC_reason=xx
> lifecycle=[start, end]
> top=x
> bottom=x
> zone_size=x
> struct_direction=x
>
> I think this way we somewhat also standardize the formatting of the labels between POI & KL and also between main &
> sub. Let me know if anything is unclear or doesn't make sense. Or if you have any questions or confusions in general.
> We can discuss.

## Today's hovers (S_dot_hover, measured with `zone_hovers.py SAVE_DIR`, this folder)

- Main KL (`export_plotly` ~:1793; the M15 charts' H1 overlay `export_m15_chart` ~:2396, the same 11 lines): TF /
  KL Zone / side / structure_id / struct_direction / base_pattern / base_idx / confirmed_idx / cycle_id / top /
  bottom. One hover per `bounds_steps` step (an expanding zone has several: H1 sid 0 cycle 0's CTS KL zone, 10 steps,
  11.4 → 31.6 pips).
- Main POI (`export_plotly` ~:2416; overlay ~:2669): TF / POI Zone (bold) / side / structure_id / struct_direction /
  ic_idx / confirmed_idx / cycle_id / the unlabelled IC line(s) (`_zone_render.poi_ic_hover_text`, 1–3 lines) / top /
  bottom / status.
- Sub KL / POI (`export_m15_chart` ~:1401 / ~:1709): as main, with `sub_id=X | first record parent_sid=…
  parent_cycle_id=…`, `sub window=[s,e] reason`, `records: …` after `side`.
- Zone data: a KL zone has ONE active stretch from `confirmed_idx` (its clamped first-active) to its cycle's end; a POI's
  `confirmed_idx` is its MOST RECENT activation (`activation_history`), not its first — 4 of the 18 drawn POIs switched
  off and on (H1 sid 1 cycle 2: on 902, off 952, on 954 → `confirmed_idx=954`; sub 1 cycle 2: on 1940 … last on 2230).
  Every zone inherits its cycle's `end_idx` / `end_reason` (`compute_cycle_lifecycle`). Drawn POI statuses on this
  window: `ended` / `active` only.

## Questions raised back (2026-10-23)

1. `lifecycle=[start, end]` — the zone's own active window (first active → end; GLOSSARY "the three clocks"), its
   cycle's window, or the structure's (`struct_window` / `sub_window`)?
2. `| end_reason=…` on the lifecycle line, like `struct_window` / `sub_window`?
3. `IC_reason=` — today 1–3 unlabelled lines (the pattern or ranking line, Step 1B's prior-imbalance line, the 80–100
   fallback line; tiebreak lines not on this window): label the first only, every line, or one line?
4. The label's case: `IC_reason` (as written) or `ic_reason` (like `ic_idx`)?
- Read, not asked: the sub zones drop `sub window` / `records:` (they stay on the sub's BOS / CTS dots); the H1 zones
  on the M15 charts take the main format; `zone_size` per hover step, 1 decimal, the pip from
  `structure_engine._pip_size_from_pair`; `status` dropped; an open zone `[x, open]`.

## Decisions (the user, 2026-10-23, AskUserQuestion — each the recommended option)

1. **`lifecycle` = the zone's OWN active window** (A): from the first candle the zone is active (`activation_history`'s
   first `active` entry) to its end (the cycle's `end_idx`). Not the cycle's window (B: differs on the H1 CTS KL zones,
   115 vs 439, and sub 3's POIs, 2829 vs 2841) nor the structure's (C: `struct_window` / `sub_window`, on the dots).
   On a KL zone the start always equals `confirmed_idx` (one active stretch) — the repeat is kept for one hover
   shape; on a POI `confirmed_idx` is the MOST RECENT activation, so the start differs when it switched off and on.
2. **`lifecycle=[start, end] | end_reason=<reason>`** like `struct_window` / `sub_window`; open → `[x, open] |
   end_reason=open`.
3. **Every IC line labelled** (b): `ic_reason=` (the pattern or ranking line), `ic_tiebreak=`, `ic_prior_imbalance=`,
   `ic_zone=`, in today's order.
4. **Lowercase**: `ic_reason` (like `ic_idx`), not `IC_reason`.
- The "read, not asked" defaults above stand (the user did not object).

## Built 2026-10-23 (figure-only)

Builders (`_zone_render`): `zone_lifecycle_line`, `zone_size_text`, `main_zone_id_line`, `kl_zone_hovertemplate(tf)` +
`kl_zone_customdata`, `poi_zone_hovertemplate(tf)` + `poi_zone_customdata`, `HOVER_NEVER_ACTIVE`; `poi_ic_hover_text`
labels its lines; the sub identity lines `export_m15_chart._sub_zone_id_lines`. Six hover sites: the H1 chart's KL /
POI, the M15 charts' sub KL / POI and H1-overlay KL / POI. Canonical description: CHARTING_SPEC §5 / §7 "Hover",
POI_ZONES_SPEC "Hover Data". Tests `engine_v2/tests/test_zone_hover.py` (10); the IC-text pins of PLAN_H / L / P / S
re-pinned to the labels (values unchanged; the POI customdata slots ic_idx 3 → 2, IC lines 6 → 4); a PLAN_N fixture
gave a KL zone `end_idx=40` with `end_reason=None` — impossible on a real zone (`compute_cycle_lifecycle` sets both),
caught by the new end ⇔ reason assert; the fixture now says `next_cycle`.
Prediction = measured vs S_dot_hover: 27 / 27 CSVs md5-equal; traces / shapes / layouts unchanged (81/234, 153/136,
301/307); the hover changed on the zone traces only (two per zone / KL step) — H1: KL 30 (15 steps), POI 4;
counter: sub KL 14, sub POI 14, overlay KL 30, overlay POI 4; confluence: sub KL 76, sub POI 30, overlay KL 30,
overlay POI 4 (`fig_scope.py BASE CUR '^(KL zone|POI zone|M15 KL zone|M15 POI zone|H1 KL zone \(overlay\)|H1 POI
zone \(overlay\))'`). Skip grep 23, fetch 4228. Suite 1959 (1949 + 10).
The user's chart look (2026-10-23, verbatim): "charts look good, go ahead and commit".
