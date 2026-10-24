# SESSION ORDER item 5 — chart updates & edits: the zone proximity trigger marker hover (the user, 2026-10-23, verbatim)

Context: the fifth chart edit, after the BOS / CTS dot hover (`ab2fcc1`), the KL / POI zone hover (`56bb20d`), the Fib
zone hover (`563f5f8`) and the wave candle hover (`18711dc`, save S_wave_hover `20261023_194643_18711dc` = the
`/compare` baseline).

## The user's spec (verbatim)

> no changes needed for Prev BOS line hover. Let's table WVMI back on chart for now. Next, let's move onto zone
> proximity trigger Markers. Below shows what I want:
>
> TF=x
> idx=x
> sid=x | cycle_id=x
> type=x
> trigger_price=x ← price that triggered proximity
> zone_inner=x ← reference price that triggered proximity
> distance_pips=x ← this should be the same as the distance between trigger_price and zone_inner
> zone_kind=x ← can remove this one since it seems to have redundant information as type
> zone_confirmed_idx=x
> sub_trigger=x ← this should be the sub trigger name that the zone proximity trigger would trigger
> struct_direction=x
>
> TF=x
> idx=x ← consistent naming with other sub labels
> idx_1H=x
> first record: parent_sid=x | parent_cycle_id=x
> parent_direction=x
> type=x
> trigger_price=x
> zone_inner=x
> distance_pips=x
> zone_confirmed_idx=x
> sub_trigger=x
> struct_direction=x
>
> Let me know if you have any questions or need any clarifications.

- The Prev BOS line hover: no change (the user, verbatim above) — its `first record parent_sid=… parent_cycle=…`
  string stays.
- WVMI back on the chart: TABLED by the user ("Let's table WVMI back on chart for now") — register §G.

## Today's hover (S_wave_hover; CHARTING_SPEC §12)

- H1 chart (`export_plotly`, 8 markers): `TF=H1` (the trigger's own `timeframe`) / `idx` / `sid` / `struct_direction` /
  `cycle_id` / `type` (`<sd|opp_sd>:<BOS|POI|CTS>`; a proximity-overlay sub zone: `opp_sd: sub_3 c2 POI 0.58096`,
  `zone_proximity.proximity_trigger_type_label`) / `distance_pips` / `zone_kind` / `zone_confirmed_idx`.
- The M15 charts (`export_m15_chart._render_proximity_triggers_overlay`, the same 8 H1 triggers on each, on the M15
  candle holding the H1 wick): `TF=1H>>15M` / `idx_15M` / `idx_1H` / `parent_sid` / `parent struct_direction` /
  `parent_cycle_id` / the same five.
- The 8 triggers: (0,1) 748 sd:POI 5.2, 820 opp_sd sub_3 c2 POI 3.2, 872 sd:BOS 1.9; (1,1) — COLLAPSED — 820 sd:BOS
  6.8, 871 opp_sd:CTS 6.6; (1,2) 935 sd:POI 3.6, 954 opp_sd sub_6 c0 POI 4.5, 1035 sd:POI 5.6.
- `distance_pips` IS the wick-to-inner distance already: the wick = the H1 trigger candle's low when the approach is
  from above (sd on a buy structure, opp_sd on a sell), else its high; signed (positive = stopped short of the inner,
  negative = crossed into the zone — all 8 positive). E.g. 748: low 0.57764, inner 0.57712 (the WVMI record's
  `trigger_inner`) → 5.2; 820 (0,1) opp_sd: high 0.58064, inner 0.58096 → 3.2.
- The sub trigger each one starts (the detectors: a cycle's FIRST trigger, sd → the WVMI gate → UC1 →
  `first_counter`; an opp_sd after an sd → `subsequent_confluence`; an sd at list index ≥ 2 → `subsequent_counter`),
  matched to the pool by trigger_idx = the H1 candle's last M15 candle: 748 → first_counter 2995 (sub 3), 820 (0,1) →
  subsequent_confluence 3283 (sub 4), 872 → subsequent_counter 3491 (sub 5), 935 → first_counter 3743 (sub 6), 954 →
  subsequent_confluence 3819 (sub 7), 1035 → subsequent_counter 4143 (sub 8); the collapsed (1,1)'s 820 →
  first_counter 3283 and 871 → subsequent_confluence 3487 are UNRESOLVED (`*_M15_unresolved_triggers.csv`,
  `collapsed_parent_cycle`) — no sub is built.

## Questions raised back (2026-10-23)

1. The M15 marker's identity line — `first record: parent_sid=… | parent_cycle_id=…` as written, or `parent_sid=… |
   parent_cycle_id=…`? The marker IS the H1 trigger (its parent structure's sid / cycle); "first record" is a sub's
   pool-wide first trigger record — a different thing.
2. The M15 marker's `struct_direction` next to `parent_direction` — the parent's again (a repeat), the direction of the
   sub the trigger starts (`first_counter` / `subsequent_counter` → −parent, `subsequent_confluence` → +parent), or
   dropped?
3. `sub_trigger` on the collapsed cycle's two triggers (no sub built) — the type alone, the type flagged `(not built:
   collapsed parent cycle)`, or the type + the resulting `sub_id`?
- Read, not asked: `TF=1H` on the H1 chart (every other hover; today `H1`), the M15 marker keeps `TF=1H>>15M`;
  `idx_15M` → `idx`; `trigger_price` = the H1 wick above (the M15 chart shows the same H1 value); `distance_pips`
  unchanged and kept signed; the sub-zone `type` drops its inner (`opp_sd: sub_3 c2 POI`) — `zone_inner` shows it;
  `zone_kind` removed; `zone_confirmed_idx` unchanged (its two meanings — register §G "Overall chart revamp" (2)).

## Decisions (the user, 2026-10-24, AskUserQuestion)

1. **`parent_sid=… | parent_cycle_id=…`** on the M15 marker (recommended) — no `first record:` prefix (the marker is the
   H1 trigger itself).
2. **Drop `struct_direction` on the M15 marker** (not the recommended "the started sub's direction"; not the parent's
   repeat) — the M15 marker shows `parent_direction` only. The H1 marker keeps `struct_direction`.
3. **The type + a flag on a collapsed parent cycle** (recommended): `sub_trigger=first_counter (not built: collapsed
   parent cycle)` on 820 / 871 of (1,1); the plain type elsewhere. Not the type alone, nor the type + `sub_id`.
- The "read, not asked" defaults above stand.

## Prediction (written before the replay, vs S_wave_hover `20261023_194643_18711dc`)

Figure-only: 27 / 27 CSVs md5-equal; traces / layouts unchanged (81, 153, 301); the hover changes on the one
`zone_proximity:trigger` trace per chart only (8 markers each; `fig_scope.py BASE CUR '^zone_proximity:trigger$'`).
`sub_trigger` per marker = the pool records above: 748 first_counter, 820 (0,1) subsequent_confluence, 872
subsequent_counter, 935 first_counter, 954 subsequent_confluence, 1035 subsequent_counter; (1,1)'s 820 first_counter
and 871 subsequent_confluence `(not built: collapsed parent cycle)`. `trigger_price` / `distance_pips` as measured
above (748: 0.57764, 5.2). Census vs S_S3: 0 deltas. The run log gains no `[uc1_trigger]` line (the chart's UC1 call
is silenced).

## Built 2026-10-24 (figure-only)

`zone_proximity.proximity_approach_from_above` + `proximity_trigger_price_and_distance` (the wick and its signed
distance — ONE helper for both markers; each site had its own copy of the approach / gap rule);
`proximity_trigger_type_label` drops a sub zone's inner; `_zone_render.proximity_sub_trigger_types` (the detectors'
final lists on the H1 frame: `attrs["subsequent_*_triggers"]` + UC1 over the main WVMI records, its log silenced) +
`proximity_sub_trigger_text` (+ the collapsed flag via `collapsed_cycles`; `HOVER_NO_SUB_TRIGGER`). The H1 marker
asserts an H1 trigger (`TF=1H`). Canonical description: CHARTING_SPEC §12. Tests `test_proximity_trigger_hover.py`
(both copies' line order, the M15 copy's H1 wick, the wick + sign on all four approach cases, the sub-trigger text and
types) + `test_proximity_overlay.py` re-pinned (the type label, the customdata slots).
Prediction = measured vs S_wave_hover: 27 / 27 CSVs md5-equal; traces / layouts unchanged; the hover changed on the
`zone_proximity:trigger` trace only (one per chart, 8 markers; `fig_scope.py` SCOPE OK); all 8 `sub_trigger` values =
the pool records (the two on the collapsed (1,1) flagged); every `trigger_price` = its candle's wick; census vs S_S3 0
deltas; the run log's `[uc1_trigger]` lines 3 → 3. Skip grep 23, fetch 4228. Suite 1972 (1966 + 6).
The user's go (2026-10-24, after the chart look): "charts look good, go ahead and commit".
