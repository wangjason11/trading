# Charting Spec (Plotly) — through Week 8

Primary file: `export_plotly.py`
Styles: `style_registry.py`

---

## Philosophy

- The chart is the primary debugger.
- Everything (except OHLC) must be toggleable in config.
- Shapes don't hover, so hover interactions use transparent helper traces.
- **All formatting centralized in `style_registry.py`** for easy adjustments.

## Hover label conventions (the user's hover-label pass, SESSION ORDER item 5, 2026-10-22 → 24)

Set by the user's five hover specs (dots, KL / POI zones, fibs, wave candles, proximity trigger markers — each
element's format is its section below + its spec in `plans/qa_chart_edits_inputs/user_spec_2026102*_*.md`); the
verbatim reasons: memory `feedback_naming_discussions.md`. A new or edited hover follows them:
- Every value is labelled `name=value` (lowercase snake_case, e.g. `ic_reason=` although a spec wrote `IC_reason`);
  fields sharing a line are split by ` | `; a missing value is `name=NA (<why>)` — never a bare `-` / symbol.
- The first line is the timeframe: `TF=1H` (an H1 element, also on the M15 charts' H1 overlay), `TF=15M` (an M15
  element); `TF=1H>>15M` on an H1 value drawn on its M15 candle — the proximity marker's M15 copy, and the H1 WVMI LP
  circle's (§13; the user, 2026-10-26).
- The candle index is `idx=` on the element's own timeframe; an M15 element adds `idx_1H=`. (The WVMI LP marks name
  their candle `lp_idx=` — on the M15 charts an H1 circle `lp_idx_1H=` + its M15 `lp_idx=`, §13.)
- Identity: `sid=S | cycle_id=C` (main); a sub `sub_id=X | cycle_id=Y` + `first record: lens=… | parent_sid=P |
  parent_cycle_id=C`; a marker that IS the parent's (the proximity marker's M15 copy) `parent_sid=… |
  parent_cycle_id=…` with no prefix.
- Windows `[start, end] | end_reason=…` (`struct_window=`, `sub_window=`, `lifecycle=`, `open` while open);
  `struct_direction=` on the main KL / POI / fib / wave-candle / proximity hovers.
- A hover never repeats a value it already shows (accepted exceptions: the sub dot hover's `first record:` line, a KL
  zone's lifecycle start = its `confirmed_idx`), and shows its OWN element's fields — unless the owning element is
  absent (a fib with no IC keeps `ic_idx=NA (<reason>)`).
- Not yet on these conventions (the pass did not touch them — `plans/qa_chart_edits_inputs/hover_inventory.py`):
  the OHLC / imbalance candle, volume / Vol EMA (`Volume:` / `EMA(20):`), the range top / bottom lines, the reversal
  candidate, the state / `bo` labels (`sid` and `struct_direction` on separate lines, no `cycle_id`), the
  structure-pattern markers (no `TF=` on H1); the Prev BOS line kept as is (the user, 2026-10-24: "no changes needed
  for Prev BOS line hover").

---

## Style Registry

All visual formatting (colors, opacities, line widths, marker sizes) lives in `style_registry.py`.

### Style Key Categories

| Category | Keys | Purpose |
|----------|------|---------|
| Candle types | `candle.pinbar`, `candle.maru` | Markers for classified candles |
| Patterns | `pattern.engulfing.*`, `pattern.star.*` | Pattern markers |
| Structure markers | `structure.success.*`, `structure.confirmed.*` | Breakout pattern triangles |
| Range | `range.candle`, `range.rect` | Range candle dots, rectangles |
| Structure lines | `structure.swing_line`, `structure.cts`, `structure.bos` | Confirmed swing line/dots |
| Reversal | `structure.reversal_watch_line`, `structure.reversal_watch_start` | Reversal candidate markers |
| KL Zones | `zone.kl.buy`, `zone.kl.sell`, `zone.kl.hover_line` | Zone fills, confirm lines |
| POI Zones | `zone.poi.buy`, `zone.poi.sell`, `zone.poi.hover_line` | POI zone fills |
| Fibonacci | `fib.anchor_line.active` / `.historical`, `fib.zone_rect.active` / `.historical` | Fib box (H1 chart + the M15 lens charts) |
| Imbalance | `imbalance.bullish`, `imbalance.bearish` | Imbalance candle colors |
| Volume | `volume.bar.up/down/neutral`, `volume.ema_line`, `volume.spike_marker`, `volume.headroom` | Volume bars, EMA, spike markers, the axis headroom |
| WVMI LP marks | `wvmi.lp_row`, `wvmi.lp_dot.sub.buy/sell`, `wvmi.lp_dot.main.buy/sell` | The rows above the volume bars, the sub dots, the H1 circles (§13) |
| Wave candles | `wave_candle.bullish`, `wave_candle.bearish` | Wave candle vertical lines |
| M15 structure | `structure.m15.swing_line`, `structure.m15.cts/bos` | M15 chart blue elements |
| M15 zones | `zone.m15.kl.buy/sell` | M15 zones on H1 chart (dashed) |
| H1 overlay | `zone.h1_overlay.kl.*`, `zone.h1_overlay.poi.*` | H1 zones on M15 chart |
| Hover lines | `hover_line.range` | Invisible hitbox lines |
| Chart layout | `chart.layout`, `chart.axis` | Background, grid, axis styling |
| Chart controls | `chart.controls` (`idx_half_window`, `price_pad`, `min_candles`, `wheel_step`, `idx_mark_color`) | The control bar above each chart (§14) |

### Changing Colors/Styling

To change any visual element:
1. Edit the corresponding key in `style_registry.py`
2. No changes needed in `export_plotly.py`

---

## Overlays (current)

### 1) Structure swing line (confirmed CTS/BOS + extras)
- Dots at confirmed CTS and confirmed BOS points
- Straight lines connecting points per structure_id
- Final segment to last candle close (most recent sid only)
- **Extra: Unconfirmed CTS after final confirmed BOS** — if a sid's last confirmed level is BOS and a CTS_ESTABLISHED or CTS_UPDATED event follows, the latest such CTS is drawn as a dot and connected with a line. If this is the active sid, the line extends to the last candle close.
- **Extra: Pullback dot after final confirmed CTS** — for non-active sids where the last confirmed level is CTS, the most recent pullback event (before the next sid's first BOS) is drawn as a dot at the candle's extreme price (low for uptrend pb, high for downtrend pb). A line connects the last confirmed CTS to this dot.
- **Extra: Cross-structure pb→BOS line** — from the pullback dot to the first BOS_CONFIRMED of the next sid. This line and the pb dot belong to the prior sid (same opacity/styling).
- Opacity: active sid = 1.0, prior sids = 0.5
- **Dot hover** (every CTS / BOS / unconfirmed-CTS / PB dot; ONE template, `_zone_render.MAIN_DOT_HOVERTEMPLATE` +
  `main_dot_customdata`, on the H1 chart AND the M15 charts' H1 overlay — the user's spec 2026-10-22,
  `plans/qa_chart_edits_inputs/user_spec_20261022_dot_hover.md`):
  ```
  TF=1H
  idx=96                                         ← the dot's ANCHOR candle (not the confirmation moment)
  kind=BOS                                       ← CTS | BOS | CTS (established|updated) | PB
  price=0.55808
  sid=0 | cycle_id=0
  trigger=start_trading | trigger_idx=NA
  struct_window=[115, 902] | end_reason=reversal
  struct_direction=1
  ```
  The two structure lines come from `_zone_render.main_dot_structure_lines(events)`: `trigger` =
  `structure_lifecycle.compute_main_started_by` — `reversal` (`trigger_idx` = the previous sid's reversal
  confirmation, `compute_reversal_idx_by_sid[sid − 1]`; it always equals the window's start, by the reversal handoff)
  or `start_trading` (`trigger_idx=NA`: no trigger candle — the backtest's Scenario 1 picks sid 0's start once, from
  the whole window; its pick is sid 0's BOS anchor, a location); `struct_window` = the REAL-TIME lifecycle,
  `compute_struct_start_by_sid` (the first CTS_ESTABLISHED moment, or the reversal handoff — the start the zones and
  wave candles floor against; sid 0: 115, not its 96 anchor) to the sid's reversal (`end_reason=reversal`) or
  `open`. So a dot can sit before its own window (sid 0's BOS at 96; sid 1's collapsed cycles' dots 689–826 before
  `[902, open]`). Known: a PB dot's `cycle_id` is a placeholder 0 (the pullback has no cycle in the dot tuple —
  deferred: `plans/qa_chart_edits_inputs/user_spec_20261022_dot_hover.md` "Quirks")
- Style keys: `structure.swing_line`, `structure.cts`, `structure.bos`

### 2) Range rectangles
- Built from the range events (`RANGE_STARTED` / `RANGE_UPDATED` / `RANGE_RESET`), one open range per structure
- A range's box (and its hover line's `range_start_idx=`) starts at `RANGE_STARTED.meta["range_anchor_idx"]` — MS's
  own range start on every path: the range-label candle, or the CTS anchor on a pullback- / proximity-created range
  (PLAN_N N5, 2026-10-12; those two paths started at the event's idx until then — 0 such ranges on the H1 reference
  window)
- Replay order (`export_plotly._range_events_chart_order` — `_range_events_with_replay_idx` with each event's replay
  candle — shared by the box and hover-line builders): a
  `RANGE_STARTED` sorts at its anchor — so the `RANGE_UPDATED`s back-filled between a label range's anchor and its
  confirm candle split its box — but never before `R`, the latest same-structure `RANGE_RESET` emitted before it,
  if `R <= ev.idx` (a pullback range's CTS anchor can precede the breakout that reset the previous range; the reset
  then closes the OLD box at `R` first, and the new box is still drawn from its anchor — so the two boxes of one
  structure OVERLAP on `[anchor, R]`, which is the data's truth: MS says the new range starts at the anchor); other
  events at their idx; ties RESET < STARTED < UPDATED. Keyed by `structure_id` (H1 only)
- The hover lines carry the boxes' data, candle for candle (the PLAN_N hover follow-up, 2026-10-12): the hover
  builder replays in the same order and snapshots every open range at each event's REPLAY candle (a `RANGE_STARTED` at
  its reset-guarded anchor, the rest at their idx); a candle shows the latest snapshot at or before it. So a range's
  line starts where its box starts — its anchor, on every path (a label range, a pullback- / proximity-created range);
  at `R` when the reset guard moved its replay, the overlap `[anchor, R)` hovering the OLD range (one hover point per
  structure per candle) — and a back-filled update shows from the candle it is stamped at. `range_start_idx=` shows the
  anchor, `last_update_idx=` the latest update replayed (the anchor until the first). On a sliced chart (`idx_range` /
  `max_points`) a range open before the slice shows from the slice's first candle, like its clamped box. (Until then the
  snapshot was taken at the event's idx while a STARTED replayed at its anchor: another structure froze as of that
  anchor, and a label range's confirm candle — and the candles forward-filled from it — showed its anchor-time state,
  e.g. sid 1 on H1 716–723 hovered hi 0.58002 under a box drawn at 0.58108; detail
  `plans/plan_n_inputs/followup_hover_per_structure.md`)
- Segmented whenever bounds change while `range_active==1`
- Ends on range deactivation or reversal state
- Style key: `range.rect`

### 3) Reversal candidate markers
- Purple X markers at reversal watch start events
- Style key: `structure.reversal_watch_start`

### 4) Structure-pattern markers (triangles)
- Filters by enabled pattern names and statuses (SUCCESS / CONFIRMED)
- Style keys: `structure.success.up/down`, `structure.confirmed.up/down`

### 5) KL zones (Week 6, updated Item 5 / 2026-05-20)
- Reads `df.attrs["kl_zones"]`
- Filters zones to the N most recent `structure_id`s (configurable)
- Deterministic draw ordering:
  - inactive first (under), active last (over)
  - older first, newer last
- **3-tier opacity (MAIN H1 chart only)**: active=100%, non-active+recent_sid=50%, non-active+prior_sid=20%. Sub charts use the per-TF tier system (see "Sub-chart per-TF tier" below).
- **Outline (Item 5)**: every zone now has a stepped-polygon outline tracing the outer contour of all `bounds_steps` (single trace per zone, not N rect outlines). Outline color matches the confirm-line color (green/red by side), width matches `confirm_line_width` (=2). Opacity = `confirm_opacity_active × tier`.
- **Fill (Item 5)**: only the active stretch is filled. KL has at most ONE active stretch `[confirmed_idx, end_idx]` (KL has no per-candle deactivation). Pre-activation region renders as outline only. End at `end_idx`.
- Stepwise bounds from `meta["bounds_steps"]` still drive per-step (top, bottom) — the fill within the active stretch is intersected with each step's x-range so each step's y-bounds apply. A step's x is `kl_zones_v1.bounds_step_draw_idx(step)`: the INIT step's `base_idx`, an expansion's `expanded_idx` (PLAN_N N1; `start_idx` on both until then).
- Single vertical confirm line at `confirmed_idx`.
- **Collapsed-cycle zones (2026-09-20):** a KL zone whose cycle COLLAPSED under its structure's lifecycle floor (`status == "inactive"`, known `end_idx`, clamped `confirmed_idx >= end_idx` — `_zone_render.is_collapsed_cycle_zone`) is not drawn, on the H1 chart, the M15 charts and the H1 overlay; it existed geometrically but was never active in real time (on H1: sid 1's retroactive cycles (1,0)/(1,1), the "collapsed parent cycles" of `ParentTables`; on a sub: its forming-phase cycles). The rows stay in the CSVs. A zone inactive for any other reason is still drawn as an outline.
- **Hover** (one per bounds step; ONE template, `_zone_render.kl_zone_hovertemplate(tf)` + `kl_zone_customdata`, on
  the H1 chart, the M15 charts' H1 overlay and the sub zones — the user's spec 2026-10-23,
  `plans/qa_chart_edits_inputs/user_spec_20261023_zone_hover.md`; e.g. H1 sid 0 cycle 0's BOS zone):
  ```
  TF=1H
  KL Zone
  side=buy
  sid=0 | cycle_id=0                             ← sub: sub_id=X | cycle_id=X + the dot hover's first record: line
  base_pattern=base inside bar
  base_idx=96
  confirmed_idx=115
  lifecycle=[115, 652] | end_reason=next_cycle   ← the zone's OWN active window (open: [x, open] | end_reason=open)
  top=0.55910
  bottom=0.55808
  zone_size=10.2 pips                            ← this step's top − bottom
  struct_direction=1
  ```
  `lifecycle` = the zone's own active window (`zone_lifecycle_line`: the first `active` entry of `activation_history`
  → the cycle's `end_idx` / `end_reason` every zone inherits) — not its cycle's window, not its structure's (the
  dots' `struct_window` / `sub_window`). A KL zone has ONE active stretch from `confirmed_idx` (asserted), so the
  start repeats `confirmed_idx` — kept for one hover shape with the POI (the user 2026-10-23). `zone_size` in pips,
  one decimal (`zone_size_text`; the pip = `structure_engine._pip_size_from_pair`). Until 2026-10-23: `structure_id=`,
  `struct_direction` third, `cycle_id` on its own line, no lifecycle / size; the sub's `first record parent_sid=…`,
  `sub window=`, `records:` lines.
- Style keys: `zone.kl.buy`, `zone.kl.sell`, `zone.kl.hover_line`

### 6) Imbalance candle highlighting (Week 7)
- Candles with FVG imbalance get distinct colors: every flagged **c2**
  (`is_imbalance == 1`, the middle candle; a merged run highlights each of its
  c2s)
- A rendering of the pattern's **location**, not a time: each highlighted c2's
  gap exists only once its c3 closes, one candle later (an instance exists from
  its first c3, `ImbalanceInstance.formed_at = start_idx + 1`). A POI confirm
  line driven by a new gap therefore sits at least one candle AFTER the run's
  first highlighted candle, never on it (Plan F, 2026-09-24;
  IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule")
- Bullish imbalance: Lime Green `rgba(50, 205, 50, 0.8)`
- Bearish imbalance: Amber Yellow `rgba(235, 190, 0, 0.8)`
- Entire candle (body + wicks) colored (Plotly limitation)
- Style keys: `imbalance.bullish`, `imbalance.bearish`

### 7) POI zones (Week 7, updated Item 5 / 2026-05-20; side tints + collapsed-cycle skip 2026-09-20/21)
- Reads `df.attrs["poi_zones"]`
- Fib-based zones from Institutional Candle identification
- **Side tints (2026-09-21):** buy = gold-lime `rgb(225, 220, 30)` with a dark-olive
  confirm line (`60, 90, 20`); sell = amber `rgb(255, 180, 30)` with a dark-brick
  confirm line (`120, 45, 15`) — `zone.poi.buy` / `zone.poi.sell`; the two sides now
  read apart like KL zones. Opacities unchanged (0.9 active / 0.12 inactive).
- **Collapsed-cycle skip (2026-09-20):** POIs whose cycle's BOS KL zone is collapsed
  (`_zone_render.collapsed_cycles`) are not drawn on the H1 chart, the M15 charts or
  the H1 overlay — on H1 that is sid 1's retroactive cycles (1,0)/(1,1) (the
  "collapsed parent cycles"); see "Collapsed-cycle zones" under the KL section.
- **3-tier opacity (MAIN H1 chart only)**: tier picked by `meta["status"]` (3-state lifecycle convention). Sub charts use per-TF tier.
- **Outline (Item 5)**: single rect outline (no fill) spanning `[start_time, end_time]`. Color matches the side's confirm line (`confirm_line_rgb`: dark olive `60,90,20` buy / dark brick `120,45,15` sell since the 2026-09-21 side tints; the `101,67,33` brown in the code is only a `.get` fallback), width = `confirm_line_width` (=2). Opacity = `confirm_opacity_active × tier`.
- **Fill (Item 5)**: ONE filled rect per active stretch in `meta["activation_history"]` — each `A` event paired with the next `D` (or `end_idx` for trailing activate). Inactive stretches render as outline only.
- **Confirm lines (Item 5)**: ONE vertical line per `A` event in `activation_history` (replaces the prior single line at `confirmed_idx`). For zones derived before activation_history was added, falls back to a single line at `confirmed_idx`.
- **Hover** (ONE template, `_zone_render.poi_zone_hovertemplate(tf)` + `poi_zone_customdata`, on all three POI hover
  sites — the H1 chart, the M15 lens charts' sub POIs and their H1 overlay; the user's spec 2026-10-23,
  `plans/qa_chart_edits_inputs/user_spec_20261023_zone_hover.md`; e.g. sub 8 cycle 2's POI on the M15 charts):
  ```
  TF=15M
  POI Zone                                        ← bold
  side=buy
  sub_id=8 | cycle_id=2                           ← main: sid=X | cycle_id=X (no first record line)
  first record: lens=counter | parent_sid=1 | parent_cycle_id=2
  ic_idx=4131
  confirmed_idx=4179                              ← the MOST RECENT activation
  ic_reason=IP 3B.iii · legs 1-1-3 (4130 | 4131 | 4132–4134)
  ic_prior_imbalance=4118 unfilled · preferred 1 of 2 IC candidates
  ic_zone=80-100 (fallback)
  lifecycle=[4179, open] | end_reason=open        ← the zone's OWN active window: FIRST activation → end
  top=0.57462
  bottom=0.57422
  zone_size=4.0 pips
  struct_direction=1
  ```
  `lifecycle` as on the KL zones (`zone_lifecycle_line`); on a POI the start = its FIRST activation while
  `confirmed_idx` = its most recent (`poi_zones`), so they differ when the POI switched off and on (4 of the 18 drawn
  POIs on the reference window — H1 sid 1 cycle 2: `confirmed_idx=954`, `lifecycle=[902, open]`); a POI that never
  activated shows `NA (never active)` for both. The `lifecycle` line replaced `status=` (the user: "effectively can
  replace status since it provides entire lifecycle information"; an open POI that is currently switched off is told
  only by the fill — none on the reference window).
  The IC lines, every line labelled (one builder, `_zone_render.poi_ic_hover_text`; labels since 2026-10-23, the
  values unchanged): `ic_reason=IP <rank> · legs a-b-c (initial | opp | continuation)` — each leg `first–last`, one
  candle its index (PLAN_H HV, landing B 2026-10-06; `versions=` until then) — + `ic_tiebreak=over <IC>: IC candle
  type` / `…: most recent before imbalance <c2>` when a tiebreaker decided (POI_ZONES_SPEC §3.4; `TB1 over …` / `TB2
  over …` until 2026-10-09; `…: immediate unfilled imbalance <c2>` from then until PLAN_P, 2026-10-13 — the hover now
  rejects that value) — + `ic_zone=80-100 (fallback)` on an IC from Step 1A (`ic_zone_band`, PLAN_P; nothing on the
  61.8–80 band; an unknown band fails loudly).
  The builder asserts the leg counts match the leg indices (a mirror that shifted one but not the other fails at render).
  A RANKED IC (`ip_outcome` `no_pattern` — POI_ZONES_SPEC §3.6, PLAN_L L2, 2026-10-10): `ic_reason=No pattern ·
  ranking rule <rule>` + ` over <runner-up>` + ` (gap <c2>)` when the deciding rule has them (`exclude_pinbar` neither,
  `ic_candle_type` no gap), or `ic_reason=No pattern · only IC candidate` (one candidate entered Step 3 — nothing
  ranked; `only preferred IC candidate` when Step 1B narrowed the field to it, PLAN_S step 3) — "No pattern" in the
  place of `IP <rank>` (since PLAN_S step 3 Step 2's outcome over the candidates Steps 2–3 ran on — POI_ZONES_SPEC
  §3.7), and NO `IC <idx>` (the hover's own `ic_idx=` line names the IC — PLAN_H §6 ICR-15); the builder asserts a
  ranked POI's pattern meta is all None and its ranking values match its rule. No new visual style (hover + CSV only,
  ICR-7). **Step 1B's line** (PLAN_S step 3, 2026-10-21; the user's rounds 6–7; `_poi_prior_imbalance_hover_text`),
  after the `ic_reason` / `ic_tiebreak` lines and before the band line, ONLY when the prior-imbalance preference
  NARROWED the field (`poi_zones.prior_imbalance_narrowed`: outcome `preferred` and fewer passers than IC
  candidates): `ic_prior_imbalance=<c2> unfilled · preferred <passers> of <candidates> IC candidates` + ` (over
  pattern <rank> at <IC>)` when it outranked a pattern (POI meta `prior_imbalance_*`, `ic_candidate_count`) — e.g.
  confluence sub 0 cycle 2's 1142: `ic_reason=No pattern · only preferred IC candidate` / `ic_prior_imbalance=1140
  unfilled · preferred 1 of 3 IC candidates`; sub 3 cycle 1's 2821 adds ` (over pattern 3B.iii at 2808)`; nothing
  when every candidate passed or 1B fell back. The builder asserts an outranked pattern only with a narrowing
  preference, and a preferred IC's gap. Until 2026-10-23 the IC lines carried no labels (`Tiebreak over …`, `Prior
  imbalance …`, `IC zone 80-100 (fallback)`), the hover had `structure_id=`, `struct_direction` third, `cycle_id` on
  its own line and a closing `status=`, and the sub POI the `first record parent_sid=…` / `sub window=` / `records:`
  lines.
- Style keys: `zone.poi.buy`, `zone.poi.sell`, `zone.poi.hover_line`

### 8) Fibonacci box (Week 7; H1 chart — and the M15 lens charts since 2026-10-06, PLAN_H §8.9 F1)
- Reads the FibTracker states (`attrs["fib_states"]`; on an M15 lens grouped by `meta["sub_id"]`). ONE draw gate for
  both charts (`_zone_render.fib_record_drawn`; the user, 2026-10-06): a record is drawn iff it is not `disappeared`,
  its cycle is not COLLAPSED (`collapsed_cycles(kl_zones)`: the cycle ended before its structure was live, so its KL /
  POI are hidden too — on H1 the collapsed parent cycles (1,0) / (1,1), on a sub its forming-phase cycles) and it is
  locked OR condition-active, ended or not — an ended never-locked record (e.g. a sub's last cycle ended by
  `parent_end`) is drawn; `active` False (a superseded cross version, a fib invalidated by an imbalance fill) vanishes
  (FIB_LIFECYCLE_SPEC §9.1 / §9.3)
- Per fib (`export_plotly._add_fib_box`, the one drawer): dotted horizontal lines at 0% (BOS anchor) and 100% (CTS anchor) plus dotted verticals at both ends (`fib.anchor_line.*`), and a dotted rectangle between 61.8% and 80% (`fib.zone_rect.*`), spanning the BOS → CTS candles; invisible hover points across the zone. The rectangle and the hover's 61.8 / 80 levels stay 61.8–80 on a record whose IC came from Step 1A (the 80–100 fallback, PLAN_P) — that POI sits below the box (buy) / above it (sell); its POI hover names the band (`ic_zone=80-100 (fallback)`), and the fib hover's no-IC line does on a fib with no IC
- "Bright" = the live active version (`.active` keys; `_zone_render.fib_is_bright`); every other drawn record renders faded (`.historical` keys)
- Hover (the user's spec 2026-10-23 — `plans/qa_chart_edits_inputs/user_spec_20261023_fib_hover.md`; ONE template +
  customdata for both charts, `_zone_render.fib_hovertemplate(tf)` + `fib_hover_customdata`), one line each: `TF` /
  **Fib Zone** / `sid=<sid> | cycle_id=<cycle>` (M15: `sub_id=<id> | cycle_id=<the sub's cycle>` + the dot hover's
  `first record: lens=… | parent_sid=… | parent_cycle_id=…` line, `_sub_zone_id_lines`) / `BOS: idx= price=` / `CTS:
  idx= price=` (the anchors) / the IC REFERENCE / the cross-fib LIMIT / `status=<status>[ (locked)]` /
  `lifecycle=[<start>, <end | open>] | end_reason=<reason | open>` (`fib_hover_lifecycle_text`: the fib's own
  `start_idx` / `end_idx` — `NA (never active)` when it never activated; `end_idx` None ⇔ `end_reason` None, asserted)
  / the no-IC line / `61.8%=` / `80%=` / `struct_direction=`. E.g. H1 sid 1 cycle 2: `sid=1 | cycle_id=2` /
  `BOS: idx=826 price=0.58105` / `CTS: idx=905 price=0.57112` / `ic_ref: price=0.57400 (prev_cycle_cts, cycle 1) |
  before=897` / `limit_price=0.57360 | reversal_bos_frozen | cycle_limit=2` / `status=active (locked)` /
  `lifecycle=[902, open] | end_reason=open` / `61.8%=0.57726` / `80%=0.57906` / `struct_direction=-1`.
  - The IC REFERENCE (`fib_hover_ic_ref_text`, PLAN_K A17 — POI_ZONES_SPEC §3.2; the fib meta `ic_ref_*`): `ic_ref:
    price=<price> (<source>[, cycle <T−1>]) | before=<idx>` — an IC candidate must lie strictly before `before` (the
    break's pattern anchor); the cycle only for `prev_cycle_cts` (the target cycle's previous one, a cross too since
    PLAN_O); `before=NA (no break)` with no known break (the price side only — counter subs 6 / 8 on the window);
    `ic_ref: NA (no reference)` (the condition passes). No line on a record without the `cross_fib_limit_*` meta.
  - The cross-fib LIMIT (`fib_hover_limit_text`, PLAN_I R6-7 — CROSS_CYCLE_FIB_SPEC §3): `limit_price=<P> |
    <source>[ (sub <from>)] | cycle_limit=<M | none>` (`cycle_limit` = `M`, the last cycle a cross may target;
    `none` = no cycle's CTS got past `P`, every cycle may cross — e.g. confluence sub 7 `limit_price=0.57112 |
    prev_sub_bos_threshold (sub 6) | cycle_limit=none`); a `P` that MOVED by the structure's end (the record carries
    `cross_fib_limit_move_count`, PLAN_M CL-6) qualifies the price: `limit_price=0.57630 (moved 7× from 0.57328) |
    parent_bos_threshold | cycle_limit=4`; without `P`, `limit_price=NA (<why>)` ALONE — no cycle may cross, so no
    `cycle_limit` (the user's decision 2: `cycle_limit=none` would read the opposite) — `start_trading`,
    `first_confluence on parent cycle 0`, `no previous sub`, `prev_sub_no_cts (sub <from>)`. No line on a record
    without the `cross_fib_limit_*` meta. Known limit: the reason reads the record's `started_by` / `parent_cycle_id`
  (the sub's first LIVE record) while `P` came from its CREATING trigger (R6-2 (a)); they differ only when the
  creating record is zero-length, and wherever their two rules give a different `P` the R6-2 tripwire WARNs (0 on the
  reference window).
  - The no-IC line (`fib_hover_no_ic_text`) — ONLY on a fib with NO IC, which has no POI to carry the reason (the
    user's decision 1; a pick's IC, its pattern or ranking rule, Step 1B's preference and its band are on its POI hover,
    §7): the cycle's row in `attrs["ic_candidates"]` (`ic_rows_by_fib_record`: keyed by (sid, cycle, BOS anchor, CTS
    anchor), prefixed with `sub_id` on a lens — a sub's sid restarts at 0) → `ic_idx=NA (<reason>)` (`no_ic_candidate`
    / `all_ic_candidates_pinbar`, POI_ZONES_SPEC §3.5; PLAN_H C1) + ` | ic_zone=80-100 (fallback)` when Step 1A
    searched 80–100 too (PLAN_P: no IC in either zone; the row's band is asserted); no row → `ic_idx=NA (not
    evaluated: no POI record)` (the POI record gate skipped it: ended before locking, or no CTS_ESTABLISHED). On the
    window: H1 sid 0 cycle 0, confluence sub 1 cycle 3 and sub 2 cycle 0.
  - Until 2026-10-23: `sid=` / `cycle=` on two lines (M15: `sub_id=… | first record parent_sid=… parent_cycle_id=…`,
    `sub window=`, `records:`, `cycle_id=`), the levels before a status cell of `status` / `life=[start, end] reason`
    (`–` = never active) / every cycle's IC outcome (`IP <rank> · IC <idx>`, `No pattern · IC <idx> · …` — PLAN_H §6
    ICR-15 kept the index as "the fib box's only pointer to its IC"; the user dropped it: "already labeled in other
    structure elements that are more relevant" — `IC: none (…)` / `IC: not evaluated (…)`, ` · zone 80-100
    (fallback)`) / `IC ref <price> (<source>) · break <idx>` / `limit <P> · <source> · M=<M>` (`limit: none (<why>)`);
    no `struct_direction`.
- M15 lens charts: every box (its 5 shapes) and its hover share ONE legend group (`m15_fib`); the first box's rectangle
  is the legend entry "M15 fib" — one click toggles every fib on the chart. No owner filter (a persisting element, like
  the zones). Toggle `fib.lines` (default on, both charts)
- Style keys: `fib.anchor_line.active` / `.historical`, `fib.zone_rect.active` / `.historical` (`fib.label` is defined in the registry but unused)

### 9) Volume overlay (Week 7)
- Volume bars at bottom 15% of chart (overlay approach, not subplot)
- Bars colored by `vol_dir`: green (+1), red (-1), gray (0)
- Volume EMA(20) blue line
- Orange diamond markers on candles with volume spikes (`is_vol_spike`)
- The volume axis = `[0, the bars' max × volume.headroom]` (1.4 — the tallest bar reaches 71 % of the panel, leaving
  the WVMI LP rows above it, §13): in the static layout (`export_plotly._volume_axis_range`, so the PNG matches) and
  in the zoom JS (the visible bars' max; ×1.1 and the layout's autorange until 2026-10-26)
- Dynamic y-axis auto-scaling on zoom (JavaScript callback); the price axis re-fits after a time-only change via the
  control bar's script (§14), which runs after this one
- Unified border around price + volume as single chart
- Hover data: idx, time, volume/EMA values
- Style keys: `volume.bar.up`, `volume.bar.down`, `volume.bar.neutral`, `volume.ema_line`, `volume.spike_marker`

### 10) Wave candle vertical lines (Week 8)
- Reads `df.attrs["wave_candles"]` (list of `WaveCandleResult`)
- Full-height vertical lines at each zone's first and last wave candle — four lines per cycle: FB (`BOS.first`), LP (`BOS.last`, attributed to the previous cycle), LB (`CTS.last`), FP (`CTS.first`); the line → role table and the per-candle cycle-lifecycle visibility gate are canonical in `zones/WAVE_CANDLES_SPEC.md` "Chart Rendering & Lifecycle"
- Green for bullish candle direction, red for bearish; neutral (dir=0) skipped
- Uniform opacity (no 3-tier multiplier — WAVE_CANDLES_SPEC "Rendering simplifications")
- Drawn with `yref="paper"` (y0=0, y1=1) so lines span full chart height
- Filtered to `selected_sids` (same filter as KL zones)
- **Hover overlay:** Invisible `go.Scatter` trace (8px wide, `rgba(0,0,0,0)`) with 12 evenly-spaced y-points per line (Plotly only detects hover near data points, not along line segments)
  - The user's spec 2026-10-23 (`plans/qa_chart_edits_inputs/user_spec_20261023_wave_candle_hover.md`), ONE builder for the H1 chart, the M15 sub lines and the M15 charts' H1 overlay (`_zone_render.wave_candle_hovertemplate`; the zone + role lines from `wave_candles.wave_candle_hover_lines`): `TF` / **Wave Candle** / `idx=` (M15: `idx=<i> | idx_1H=<h>`) / `<BOS|CTS> zone: sid=S | cycle_id=N` (M15: `sub_id=X | cycle_id=N`, then the dot hover's `first record: lens=… | parent_sid=… | parent_cycle_id=…` line — kept with `lens=`, the user's decision 1) / `Role: <FB|LB|FP|LP> (<name>)` (the LP names its owning cycle) / the window — main: the dot hover's `struct_window=[s, e] | end_reason=<reversal | open>` (`main_dot_structure_lines`; decision 2 added the reason), M15: `sub_window=[s, e] | end_reason=…` / `volume=<raw volume>` (decision 3: `=`, not the spec's colon) / `struct_direction=` (the structure's, from its KL zones' meta — `struct_direction_by_sid`; a `WaveCandleResult` carries none). E.g. H1 idx 651: `idx=651` / `CTS zone: sid=0 | cycle_id=1` / `Role: LB (last breakout)` / `struct_window=[115, 902] | end_reason=reversal` / `volume=3032` / `struct_direction=1`. Until 2026-10-23: `cycle=` unlabelled after a space, `Volume: <v>`, no window / direction on the main; the sub's `first record parent_sid=… parent_cycle=…` after the role and `sub window=[s,e] reason`. No WVMI momentum and no weighted volume (removed 2026-05-28, WAVE_CANDLES_SPEC "Rendering simplifications"); WVMI is drawn as its own marks above the volume bars since 2026-10-26 (§13)
- Style keys: `wave_candle.bullish`, `wave_candle.bearish`

### 11) Subordinate overlays on parent charts (Week 8 / Part 4)

The H1 chart can overlay zones from subordinate sub-entities (M15.counter,
M15.confluence, future deeper-nested subs). These are visual aides — the
sub structures still compute regardless of whether they render.

**Config-gated, default off.** Per-sub-entity, per-element-kind toggles
live under `zones.subordinate_overlays` in `chart_cfg`:

```python
"zones": {
    "KL": True, ...,
    "subordinate_overlays": {
        # Missing keys default to off — H1 chart stays clean by default.
        "M15.counter":    {"KL": True},   # opt-in example
        # "M15.confluence": {"KL": True},
    },
}
```

The master `zones.KL` switch still gates ALL KL rendering (H1-native + any
subordinate overlay). Subordinate flags layer on top — both must be true
to render. Same pattern will extend to `POI`, `fib`, etc. as new sub
element kinds come online (counter/confluence variants are still being
built in Part 4; new overlay kinds will be additive).

**Currently rendered (when enabled):**
- M15.counter KL zones — read from the registered M15 sub-entity's
  `df.attrs["kl_zones"]` via the registry (§13.5.c.iii; the legacy
  `dfx.attrs["lower_tf_results"]` facade is gone), dashed rectangles with
  lower opacity, mapped to H1 x-axis
- Style keys: `zone.m15.kl.buy`, `zone.m15.kl.sell`

**Not yet rendered on H1 (future work):** M15 POIs, M15 fibs, M15.confluence
zones, deeper-nested sub elements. Adding any of these means: (a) new
config key under `subordinate_overlays`, (b) a new rendering block in
`export_plotly.py` gated on that key, (c) style keys in `style_registry.py`.

**Distinct from parent overlays on sub charts.** The M15 chart overlays
H1 elements onto M15 — that's a separate mechanism with its own rendering
logic in `export_m15_chart.py`. Under Item 5 (2026-05-20), H1 overlays on
the M15 chart read their style bases from `zone.kl.*` / `zone.poi.*`
(unified with native H1) and apply the sub-chart per-TF `main_tf` tier
(0.2) for opacity. The legacy `zone.h1_overlay.*` style entries are no
longer used by the rendering code — they remain in `style_registry.py`
unreferenced.

### 12) Zone-proximity trigger markers (H1 + the M15 charts' H1 overlay)

One × marker per H1 zone-proximity trigger (`attrs["zone_proximity_triggers"]`, `zones/zone_proximity.py`), style
`zone_proximity.trigger`, above a red candle / below a green one. The H1 chart draws it on the trigger candle
(`export_plotly`); the M15 charts on the M15 candle holding the trigger's wick (`export_m15_chart.
_render_proximity_triggers_overlay`, the price-extreme mapper `map_candle_to_lower_tf`). Every trigger is drawn — a
collapsed cycle's included (no lifecycle filter, like the scan itself; the zones it fired against are hidden).
Hover (`customdata`; the user's spec 2026-10-23 / 24 — `plans/qa_chart_edits_inputs/user_spec_20261023_proximity_
marker_hover.md`): the H1 chart `TF=1H` / `idx` / `sid=<S> | cycle_id=<C>` / `type` / `trigger_price` / `zone_inner` /
`distance_pips` / `zone_confirmed_idx` / `sub_trigger` / `struct_direction`; the M15 charts `TF=1H>>15M` / `idx` (the
M15 candle) / `idx_1H` / `parent_sid=<S> | parent_cycle_id=<C>` (no `first record:` prefix — the marker IS the H1
trigger; the user's decision 1) / `parent_direction` / the same `type` … `sub_trigger` — no `struct_direction` (decision
2). E.g. H1 748: `sid=0 | cycle_id=1` / `type=sd:POI` / `trigger_price=0.57764` / `zone_inner=0.57712` /
`distance_pips=5.2` / `zone_confirmed_idx=652` / `sub_trigger=first_counter` / `struct_direction=1`.
- `type` = `<sd|opp_sd>:<BOS|POI|CTS>` on a parent zone — on a proximity-overlay zone (PLAN_R step 2) it names the
  source: `<sd|opp_sd>: sub_<id> c<cycle> <BOS|POI>`, e.g. `opp_sd: sub_3 c2 POI` (its inner is the `zone_inner=` line;
  `… POI 0.58096` until 2026-10-24) — one helper for both charts, `zone_proximity.proximity_trigger_type_label`.
- `trigger_price` = the H1 trigger candle's wick — its low when the price approaches the inner from above (an sd
  trigger of a buy structure, an opp_sd of a sell; `zone_proximity.proximity_approach_from_above`), else its high —
  and `distance_pips` = that wick to `zone_inner` (`trigger_inner`), rounded to 0.1, signed: positive = stopped short,
  negative = crossed into the zone (`proximity_trigger_price_and_distance`, both charts; the M15 copy shows the H1
  values).
- `sub_trigger` = the sub trigger the proximity trigger starts (`_zone_render.proximity_sub_trigger_types`, from the
  detectors' final lists on the H1 frame): a cycle's first trigger, sd → `first_counter` (UC1 over the cycle's first
  sd trigger, `zone_proximity.first_sd_proximity_triggers`; over the main WVMI record until 2026-10-26); an opp_sd after an sd → `subsequent_confluence`; a later sd → `subsequent_counter`; + ` (not built:
  collapsed parent cycle)` on a collapsed parent cycle (the pool builds no sub there — its record is unresolved,
  `collapsed_parent_cycle`; decision 3), `NA (no sub trigger)` when none. Until 2026-10-24: `TF=H1`, `sid` /
  `struct_direction` / `cycle_id` on three lines, `zone_kind`, no trigger price / inner / sub trigger (M15: `idx_15M`,
  `parent struct_direction`).
- `zone_confirmed_idx` branches on the SOURCE first: a sub zone's is the H1 candle it counts from
  (`meta["prox_ref_usable_idx"]`; the parent's (sid, cycle, kind) lookups would mislabel it — the POI one silently,
  with the parent sd POI's floor). On a parent zone it depends on the kind:
  - **BOS / CTS:** the KL zone's `confirmed_idx` — lifecycle-CLAMPED (a collapsed cycle's shows the structure's
    lifecycle start, AFTER its own trigger: 902 on structure 1 cycle 1's 820 / 871 markers on the reference window);
  - **POI:** `poi_unclamped_activation_floor_idx` — the POI's activation floor without the lifecycle-start term
    (PLAN_Q, 2026-10-14; before it the start of the active stretch holding the trigger, −1 outside one).
  The two differ on a collapsed cycle (S_O's structure 1 cycle 1: the BOS zone 902, the POI 748) — register §G
  "Overall chart revamp" item (2).

### 13) WVMI LP marks above the volume bars (H1 + the M15 charts; "WVMI back on the chart", 2026-10-26)

The user's design (rounds 2–3, 9, 11, 12 — `plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`): one
mark per per-LP row (`WVMIRecord.lp_rows`, WVMI_SPEC "Per-LP rows") — every temporary LP a record has had and the
locked one — so the chart shows a record's WVMI as of each LP candle, the way it moves while the pullback nears the
zones, not only its end-of-run value. No combined WVMI value (the user, round 3: WVMI = the pair, read by the four
combinations of `breakout_momentum` / `pullback_momentum` around 1).
- **Which rows:** every row of every NON-collapsed record (`_zone_render.wvmi_lp_marks` = the per-LP CSV's rows,
  `export_wvmi_lps`); a collapsed record (`cycle_collapsed`) is hidden like its wave candles (round 1); an LP before
  its sub's `start_idx` IS drawn (round 11). No other filter: no §16.5 owner filter on the sub dots (one dot per row),
  no H1-overlay window filter and no `num_structures` filter on the H1 circles. A zoomed export (`idx_range`) draws
  the marks of its own candles.
- **Where:** on the volume panel, one fixed row per zone side above the bars — sell (red) just above them, buy (green)
  one step higher (the user, round 2: "The green dot should be slightly above the red one"; round 3 "fixed row") —
  `wvmi.lp_row` 0.80 / 0.91 of the panel, on the marks' own axis `yaxis3` (`overlaying="y2"`, range [0, 1], fixed,
  hidden — `export_plotly._wvmi_lp_axis_layout`), so the rows hold whatever the volume zoom does; §9's headroom keeps
  the bars below them.
- **What:** a sub's LP = a small filled dot (`wvmi.lp_dot.sub.<side>`, size 5) at its LP candle on its lens chart
  (`m15_df.attrs["wvmi"]`, entity-absolute). An H1 LP = an open circle (`wvmi.lp_dot.main.<side>`, size 8 — the user's final
  value after 11 / 9 / 7, rounds 12–14): on the H1 chart at its LP candle (`H1.main`'s `attrs["wvmi"]`); on both
  M15 charts at the LAST M15 candle of its hour (`multitf/entity_df_mutation._map_parent_idx_to_m15_hour_end`, the
  timing mapper — the user's Q6 a), where it fits around a same-side sub dot on that candle (round 2: "the parent
  circle should fit neatly around the sub dot"). Colour = the zone side only; the status is in the hover (round 2).
  Traces, one per side (`export_plotly._add_wvmi_lp_traces`): the H1 chart `WVMI LP <side>`; the M15 charts
  `M15 WVMI LP <side>` (the sub dots), then `H1 WVMI LP <side>` (the circles, drawn over the dots); each mark's hover
  text in `customdata` (`hovertemplate="%{customdata}"`).
- **Hover** (`_zone_render.wvmi_lp_hovertext` — the drafts the user accepted, rounds 3–4; `zone_side=`, round 9;
  `TF=1H>>15M`, round 12):
  - a sub dot: `TF=15M` / **Sub WVMI** / `sub_id=X | cycle_id=Y | relative_dir=<counter | confluence | NA (before
    start_idx)>` (the sub's at the LP candle — `multitf/types.sub_relative_dir_at` over the lens's `SidRecord`, the
    per-LP CSV's reader) / `lp_idx=… | lp_status=…` / `zone_side=… | breakout_momentum=… | pullback_momentum=…`;
  - an H1 circle on the H1 chart: `TF=1H` / **Main WVMI** / `sid=S | cycle_id=C` / `lp_idx=… | lp_status=…` / the
    same last line; on the M15 charts: `TF=1H>>15M`, and the LP line `lp_idx_1H=<H1> | lp_idx=<its M15 candle> |
    lp_status=…` (the status idx stays the H1 one — the accepted draft);
  - `as_of_idx=` (`as_of_idx_1H=` on an H1 circle's M15 copy) joins the LP line, after the LP, only when the LP became
    the LP on a later candle (`wvmi_lp_idx_fields`); `lp_status=<status> (<reason> at <idx>)`, an `open` row
    `open (still moving at the data end)` (`wvmi_lp_status_text`); the momenta to 3 decimals — the pullback momentum
    is the ROW's (this LP's), `NA (not computed)` if missing (never on a drawn record: a record needs FB / FP volume).
  - E.g. confluence 4129: `TF=15M` / `Sub WVMI` / `sub_id=8 | cycle_id=0 | relative_dir=NA (before start_idx)` /
    `lp_idx=4129 | lp_status=locked (cycle 1's BOS confirmed at 4157)` / `zone_side=buy | breakout_momentum=1.288 |
    pullback_momentum=0.468`; its first LP `lp_idx=4102 | as_of_idx=4103 | lp_status=temporary (replaced by a further
    close at 4129)`.
- **A circled dot's two hovers:** Plotly's closest-point hover subtracts each marker's radius from the cursor
  distance (floored at `1 − 3 / radius`), so the dot (the smaller radius) wins at the centre and the circle on its
  ring (size 8: the dot 0 vs the circle 1 − 3 / 4 = 0.25 at the centre) — derived from Plotly's scatter hover rule;
  checked by the user in the browser 2026-10-27: "the browser circle surrounding sub dot looks good".
- **Reference window (S_lp_rows' records):** the H1 chart 57 circles (42 buy / 15 sell); counter 74 sub dots (27 /
  47) + the 57 H1 circles; confluence 185 (88 / 97) + 57. A circle around a sub dot: counter 1 (3755), confluence 7
  (2735, 3499, 3587, 3755, 4179, 4183, 4195). No candle carries both a buy and a sell mark. A zoomed look that keeps
  the volume panel: `plans/qa_chart_edits_inputs/zoom_png_volume.py` (`zoom_png.py` flattens it).

### 14) The chart control bar (H1 + the M15 charts; item 5's final part, 2026-10-27)

The user's ask (2026-10-27): "1) adding a toggle to hide parent (1H) chart overlay elements, and 2) adding an ability
to zoom into certain parts of the chart via a) a +/- switch or tool and b) allowing user to input an idx for the chart
to zoom into". Design rounds 1–2, the prototype and the plan: `plans/qa_chart_edits_inputs/
user_spec_20261027_parent_toggle_and_zoom.md`. A bar of HTML controls above each chart's title — HTML only (the PNG is
unchanged), one script `charting/chart_controls.js`, appended after the volume zoom JS by
`export_plotly._inject_chart_controls_js` with its config `_chart_controls_cfg` (both exporters; with or without the
volume panel):
`[Parent (1H): shown | hidden]` (M15 charts only) `|` `−` `+` `|` `Box zoom` `Pan` `|` `idx [ ]` `[15M | 1H]` (M15
charts only) `Go`.
- **Parent (1H)** — an on-chart button (the user's round 1; PART4 §16.4's legend group was not built) that hides /
  shows every element `_render_h1_overlay` draws: the H1 swing + PB→BOS lines, CTS / BOS / PB dots, "bo" and
  pb / pr / rv labels, KL zones (fills, outlines, confirm lines, hover lines), POI zones, wave-candle lines + their
  hover hitboxes, the Prev BOS line — 60 traces + 40 shapes on each M15 chart of the reference window. They carry
  `legendgroup = "parent_overlay"` (`export_m15_chart.PARENT_OVERLAY_GROUP`, set by `_tag_parent_overlay` on whatever
  that call drew; no legend entry shows it). NOT covered (the user: "All except both"): the H1 zone-proximity trigger
  markers (§12 — they keep their own legend entry) and the H1 WVMI LP circles (§13). Shown on load.
- **− / +** — the TIME axis only: double / halve the visible candles about the centre (at least
  `chart.controls.min_candles` 10). **The wheel** over the plot area does the same about the cursor
  (`wheel_step` 1.2 per notch; outside the plot area the page scrolls). Plotly's own toolbar +/− (both axes about
  the centre, no re-fit) stays as it was.
- **Box zoom | Pan** — plotly's `dragmode`, the active one dark. Pan moves the time axis only (`yaxis.fixedrange` while
  panning — the toolbar's Pan icon too; the switch tracks it) and keeps the window's candle count: plotly pans by
  CLOCK time, which across a rangebreak (weekend / holiday) shrank the window or stuck it, so on release the window is
  re-set to the same candles shifted by the candles dragged across (a small snap is visible at the release after a
  drag over a gap). Box zoom is plotly's (a thin horizontal box zooms the time axis only).
- **The price re-fit** — after every time-only change (− / +, the wheel, an idx jump, a pan's release, an x-only box,
  a drag on the time axis labels) the price axis = the visible candles' low..high + `price_pad` 6 % each side. A box
  zoom that sets a price range keeps it. The volume panel re-fits as before (§9).
- **The idx box** — an idx of the chart's TF, or on an M15 chart the H1 idx (`1H` in the selector — the candle hover's
  `idx_1H`, its hour's M15 candles): Go / Enter centres the chart on it with `idx_half_window` 48 candles either side
  (97 visible, on every chart; H1 was 24 in the first prototype — "too zoomed in", round 2) and highlights the candle(s)
  with a blue band (`idx_mark_color`) that stays until the next jump; Go on an EMPTY box clears it (round 2). An idx
  outside the data shows the nearest candle with the note `no <TF> candle at idx <v> - showing idx <n>`; a non-number:
  `not an idx: <text>`.
- **The candle table** — `_chart_controls_cfg` ships `[idx, time (UTC wall clock), low, high(, idx_1H)]` for every
  DRAWN candle (the `idx_range` slice of a zoomed export) in time order — from the frame, not the plotly traces (the
  imbalance candles are separate candlestick traces). The reference window: H1 1058 candles (0..1057), each M15 chart
  4228 (0..4227, idx_1H 0..1056).
- Checked headless (choreographer, kaleido's driver): `plans/qa_chart_edits_inputs/proto_headless_check.py CHART_HTML
  SHOTS STEPS` drives the bar and prints the visible idx span, the axes, the parent counts, the highlight and the note
  per step; `controls_landing_check.py` = the landing's figure / CSV check.

---

## Zone fill semantics — Item 5 (2026-05-20)

All zone rectangles (KL + POI, both charts) follow these rules:

| Element | Rule |
|---|---|
| Outline | Always drawn for every drawn zone (collapsed-cycle zones are skipped entirely — §5). Spans full `[base_idx, end_idx]` (or `start_time`→`end_time` for POI). For KL with `bounds_steps`, traces a stepped polygon around the outer contour of all steps (single trace, not N rect outlines). Color matches the zone's confirm-line color. Opacity = `confirm_opacity_active × tier`. |
| Fill | ONLY on active stretches. Pre-activation, post-deactivation, and inter-stretch gaps render as outline only. |
| KL active stretches | One: `[confirmed_idx, end_idx]`. No reactivation logic today. |
| POI active stretches | N stretches, one per `A → next D` pair in `meta["activation_history"]`. Trailing `A` with no `D` extends to `end_idx`. |
| KL confirm line | One vertical line at `confirmed_idx`. |
| POI confirm lines | N vertical lines, one per `A` event in `activation_history`. Fallback to single line at `confirmed_idx` for legacy zones without history. |
| Outline color (KL) | Side color: green `rgb(0,180,0)` (buy) / red `rgb(220,0,0)` (sell). Sub-native M15 zones use thin black `rgba(0,0,0, line_op)` width 0.5. |
| Outline color (POI) | Side confirm-line colour: dark olive `rgb(60,90,20)` (buy) / dark brick `rgb(120,45,15)` (sell). Sub-native M15 zones use thin black `rgba(0,0,0, line_op)` width 0.5. |

Implementation: shared helpers in `engine_v2/charting/_zone_render.py`:
- `compute_kl_active_stretches(zone, render_end_idx)`
- `compute_poi_active_stretches(zone, render_end_idx)`
- `build_stepped_outline_xy(bounds_steps, x_resolver, x_start, x_end, ...)`
- `select_subordinate_tf_tier(zone_tf, primary_sub_tf, smallest_sub_tf)`

---

## Opacity composition tables

### Main H1 chart — existing 3-tier cascade (unchanged tier logic)

**Tier selection:**
- KL: `meta["status"]=="active"` → `active`; else if `structure_id == most_recent_sid` → `recent_inactive`; else `prior_inactive`. (Phase 3, 2026-05-26: KL adopted the active/inactive/ended convention — the old `meta["active"] AND end_time is None` test is gone.)
- POI: `meta["status"]=="active"` → `active`; else if `structure_id == most_recent_poi_sid` → `recent_inactive`; else `prior_inactive`.

**Tier multipliers** (`STYLE["opacity_tiers"]`): `active=1.0`, `recent_inactive=0.5`, `prior_inactive=0.2`.

**Base style values:**
- `zone.kl.{buy,sell}`: `fill_opacity_active=0.4`, `confirm_opacity_active=0.9`
- `zone.poi.{buy,sell}`: `fill_opacity_active=0.9`, `confirm_opacity_active=0.9`

**Composed final opacities** (`base × tier`):

| Zone | State (tier) | Active-stretch fill | Outline + confirm line |
|---|---|---|---|
| KL | active (×1.0) | 0.4 × 1.0 = **0.40** | 0.9 × 1.0 = **0.90** |
| KL | recent_inactive (×0.5) | 0.4 × 0.5 = **0.20** | 0.9 × 0.5 = **0.45** |
| KL | prior_inactive (×0.2) | 0.4 × 0.2 = **0.08** | 0.9 × 0.2 = **0.18** |
| POI | active (×1.0) | 0.9 × 1.0 = **0.90** | 0.9 × 1.0 = **0.90** |
| POI | recent_inactive (×0.5) | 0.9 × 0.5 = **0.45** | 0.9 × 0.5 = **0.45** |
| POI | prior_inactive (×0.2) | 0.9 × 0.2 = **0.18** | 0.9 × 0.2 = **0.18** |

### Sub M15 chart — per-TF tier (Item 5, NEW)

**Tier selection** (zones only; sid-tied elements keep their current treatment):
- `meta["timeframe"]` matches the chart's primary sub TF (M15) → `sub_tf` (0.5)
- Anything else (typically H1 overlays) → `main_tf` (0.2)
- Future 3-TF case: `sub_tf_smallest` (1.0) reserved

**Tier multipliers** (`STYLE["opacity_tiers.subordinate_chart"]`): `main_tf=0.2`, `sub_tf=0.5`, `sub_tf_smallest=1.0`.

**Base style values** reused from the main-chart entries (same `zone.kl.*` / `zone.poi.*` keys).

**Composed final opacities:**

| Zone | TF context (tier) | Active-stretch fill | Outline + confirm line |
|---|---|---|---|
| KL | main-TF overlay, H1 (×0.2) | 0.4 × 0.2 = **0.08** | 0.9 × 0.2 = **0.18** |
| KL | sub-TF native, M15 (×0.5) | 0.4 × 0.5 = **0.20** | 0.9 × 0.5 = **0.45** |
| POI | main-TF overlay, H1 (×0.2) | 0.9 × 0.2 = **0.18** | 0.9 × 0.2 = **0.18** |
| POI | sub-TF native, M15 (×0.5) | 0.9 × 0.5 = **0.45** | 0.9 × 0.5 = **0.45** |

**Sub zones capped by their sub's window** (historical: this paragraph described "cascaded sub zones" with `meta["deactivated_by"]="overwritten_by_sid_N"` capped by `_tag_old_sid_on_overwrite`; the cascade was DELETED 2026-05-26 and `deactivated_by` no longer exists on zones). Under the pool (Plan C, 2026-09-20) a sub-native KL/POI zone's `end_idx` / `end_time` comes from `compute_cycle_lifecycle` with `lifecycle_cap` = the unique sub's `end_idx` and `cap_reason` = the sub's `end_reason` (`render_sub_projection` → `project_to_window`), so a zone of a sub that ended by `same_dir_replacement` / `parent_end` / `reversal` has its visible extent truncated at the sub's window end. Such zones stay rendered (outline + any active stretch) and use the same per-TF tier as every other sub-native zone.

### Outline color/width matrix

| Context | Outline color | Outline width |
|---|---|---|
| Main H1 chart, KL | `rgb(0,180,0)` (buy) / `rgb(220,0,0)` (sell) at `confirm_opacity_active × tier` | 2 |
| Main H1 chart, POI | side confirm colour — olive `rgb(60,90,20)` (buy) / brick `rgb(120,45,15)` (sell) — at `confirm_opacity_active × tier` | 2 |
| Sub M15 chart, sub-native KL/POI (M15) | `rgba(0,0,0, line_op)` (thin black) | 0.5 |
| Sub M15 chart, main-TF overlay KL | side color at `confirm_opacity_active × main_tf_tier` | 2 |
| Sub M15 chart, main-TF overlay POI | brown at `confirm_opacity_active × main_tf_tier` | 2 |

These tables are derived from `style_registry.py` + the rendering code in `export_plotly.py` / `export_m15_chart.py` — keep in sync when those values change.

---

## M15 Dedicated Chart (`export_m15_chart.py`)

A separate chart file renders M15 candles with both M15 structure and H1 overlay elements. This is NOT the same as the H1 chart — it has its own rendering logic.

### Architecture
- **Entry:** `export_m15_chart_plotly(registry=..., path_id=..., title=..., ...)` — registry-only (§13.5.e); resolves its M15 entity + parent overlay from the registry. Reads sub data from `m15_df.attrs["events" / "kl_zones" / "poi_zones" / "wave_candles" / "wvmi" / "prev_bos_lines"]` + `"fib_states"` (each sub's fib records, drawn since 2026-10-06 — §8 / PLAN_H §8.9 F1; also `*_M15_{lens}_fib_lifecycle.csv`) and `"ic_candidates"` (the fib hover's IP line), grouped by each snapshot's **`meta["sub_id"]`** (`_sub_identity`) per the `m15_df.attrs["sids"]` `SidRecord` manifest (one row per unique sub rendered on this lens; `_sid_record_identity` = `sub_id` for a sub row, `sub_sid` for a main row), with `m15_df.attrs["triggers"]` (this lens's `TriggerRecord`s) for hover attribution. (Plan C, 2026-09-20 — replaces the pre-pool identity tuple `(parent_sid, parent_cycle_id, sub_sid)`; `sub_sid` no longer exists on any sub artifact.)
- **M15 candles** as the base OHLC
- **M15 structure** (swing lines, CTS/BOS dots, prev BOS lines) in **royalblue**;
  where two structures draw over the same candles, the older one's segments are
  drawn in the navy dotted PRIOR style (see "Recent vs prior" below)
- **H1 overlay** (swing lines, CTS/BOS dots, prev BOS lines) in **black**; since
  the chart review of 2026-09-21 the overlay's swing / PB→BOS lines are
  **lifecycle-filtered per wave** — a wave that was never live on the H1
  structure's lifecycle is not drawn at all (see "H1 overlay structure lines"
  below). The H1 chart itself is unchanged.
- Connector lines are solid for every live wave (M15 and H1); only M15 waves
  that were never live are dotted; prev-BOS lines are always solid

### Color Differentiation Convention
| Element | M15 (sub-native) | H1 (main-TF overlay) |
|---------|------------------|----------------------|
| Swing lines / dots | Royalblue | Black |
| Prev BOS lines | Royalblue | Black |
| Hover background | Royalblue | Default |
| Wave candle (WVMI) lines | Dashed | Solid |
| KL/POI zone outline (Item 5) | Thin black 0.5px (tier-faded) | Matches confirm color, width 2 |
| KL/POI zone fill (Item 5) | Active stretches only, sub_tf tier (×0.5) | Active stretches only, main_tf tier (×0.2) |

### Style Keys (M15-specific)
| Key | Purpose |
|-----|---------|
| `structure.m15.swing_line` | M15 swing connector line (royalblue) |
| `structure.m15.cts` / `.bos` | M15 CTS/BOS dots (royalblue) |
| `prev_bos_line.m15` | M15 prev BOS line (royalblue) |
| `structure.h1_overlay.swing_line` | H1 swing line on M15 chart (black, width 1) |
| `prev_bos_line.h1_overlay` | H1 prev BOS line on M15 chart (black, width 1) |
| `wave_candle.h1_overlay.*` | H1 wave candle lines on M15 chart (solid) |
| `zone.h1_overlay.kl.*` / `.poi.*` | **Deprecated by Item 5** — H1 overlay zones now read base values from `zone.kl.*` / `zone.poi.*` and apply the `main_tf` per-TF tier. Entries remain in `style_registry.py` but are no longer consumed by the rendering code. |

### Opacity
- All swing/connector lines use **flat opacity** from the style (no per-sid multiplier).
- **Zone opacity follows the per-TF tier system** (Item 5) — see "Opacity composition tables" above. The legacy 3-tier (active/recent_inactive/prior_inactive) is the MAIN H1 chart only; sub charts use main_tf (0.2) for H1 overlays and sub_tf (0.5) for M15 native.
- **No dot-trace opacity tier.** Every M15 dot and line renders at its flat registry opacity — consistent with "flat opacity" above. (A per-parent tier, `_m15_opacity_tier_for_events` + `_compute_m15_tier_context_from_sids`, was computed but never applied — audit 2026-09-21 — and was deleted 2026-09-30, user decision. The hover's informational parent fields come from each sub's first record via `_sid_parent`.)

### Sub ownership, identity and hover — PART4 §16.5 rev 2 (Plan C, landed 2026-09-20)

This is the M15-chart half of `PART4_REFACTOR_SPEC.md §16.5` / `§17.9`; the
numbered §16.5 rules live in PART4, this section states them as the chart
implements them (`export_m15_chart.py`).

**Identity = `sub_id`** (the unique sub, `PART4 §17.2`). One `SidRecord` per
unique sub on the lens (`sub_id` set, `sub_sid = None`, `parent_sid` /
`parent_cycle_id` None — parent attribution is on the record table
`attrs["triggers"]`). Every element of a sub is grouped by `meta["sub_id"]`;
the informational `parent_sid` / `parent_cycle_id` / `use_case` on a snapshot
are never used for grouping. Trace names carry the identity: sid-tied dots are
`M15 {CTS|BOS|CTS (unconf)|PB} sub{sub_id}` (`_render_m15_dots`), KL outlines
are `M15 KL outline sub{sub_id} c{cycle_id}`, swing lines are
`M15 swing[ (prior)] sub{sub_id}_m15s{internal_sid}` and the cross-structure
line is `M15 PB→BOS sub{sub_id}`; only the prev-BOS trace name still carries the
first record's parent `h1s{parent_sid}c{parent_cycle_id}` (informational).

**Ownership = the sub's real-time lifecycle window, per direction.**
`_compute_owner_by_idx_dir(sid_records, edge_idx)` builds
`owner_by_idx_dir[(candle, direction)] = sub_id` over each sub's
`[start_idx, end_idx or edge_idx]` (`SidRecord.start_idx` /
`SidRecord.end_event_idx` — the lifecycle, NOT the structural anchor
`creation_event_idx = starting_idx`), keyed by `(candle, starting_sd)` because
a `+1` and a `−1` sub may both be live on one chart (`PART4 §17.5`; e.g.
`4027/+1` on the confluence chart from 4083 while `3760/−1` runs to 4200 —
opposite directions, intended). Rows are walked in `(start_idx, sub_id)` order
so a later start wins a same-direction overlap. Rows with `start_idx is None`
(a sub with no live record) are skipped — logged, not rendered.

**Sid-tied elements draw only where owned** (`_owned_here(idx)`): CTS/BOS
dots, unconfirmed-CTS and PB dots, swing lines (the last-sid extension walks
back to the last owned candle), PB→BOS lines, prev-BOS lines (hidden if their
start candle is not owned), wave-candle verticals (per candle, on top of the
per-candle cycle lifecycle gate of `WAVE_CANDLES_SPEC.md`, whose `cycle_life`
the chart reads off the sub's KL BOS-zone meta `confirmed_idx` / `end_idx` /
`end_reason`). WVMI records are grouped by `sub_id` too; their LP marks
(§13) take NO owner filter — one dot per per-LP row.

**Two ownership layers (chart review 2026-09-20, option 2).** The rule as first
landed hid everything before `start_idx`, which broke every BOS→CTS line
mid-structure. Now: `_is_live(idx) = idx >= sub.start_idx` picks the layer —
a LIVE candle is drawn iff `_compute_owner_by_idx_dir` (the lifecycle window,
later start wins among live subs) names this sub; a FORMING candle (the sub's
`[starting_idx, start_idx)` span, `_compute_forming_by_idx_dir`, later anchor
wins among forming subs) is drawn iff that map names this sub — independently
of any live sub of the same direction, so a structure forming under a live one
stays visible (`3304/−1` forming 3304→3621 under live `2639/−1`). Ownership
decides WHICH points exist; the wave rule below decides their STYLE.

**Recent vs prior (chart review 2026-09-22) — what solid and dotted mean.**
A *segment* is one straight piece of a sid-tied line between two consecutive
drawn points — over the ANCHOR candles the line runs through
(`cts_anchor_idx` / `bos_anchor_idx`), never the confirmation candles — including the
most recent internal sid's extension to the last owned candle and each
cross-structure PB→BOS line. Where the structures of two different subs draw
segments over the same candles, the **most recent** structure is solid and the
**prior** one's segment is dotted:

- **Recency** = `_recency_key` = the hierarchical `(parent_sid,
  parent_cycle_id, sub_id)` tuple (the first two from the sub's first record,
  informational; `sub_id` is the canonical identity, numbered by lifecycle start since PLAN_I S1). The structure
  that occupies a candle span later always has the higher tuple; on every
  window measured the tuple orders exactly like `sub_id`.
- **Overlap** = `lo_a < hi_b and lo_b < hi_a` — MORE THAN ONE shared candle.
  Two segments meeting at a single join candle are not an overlap (reference
  window: sub `1797/−1` 2270→2365 vs sub `2365/+1` 2365→2557, and sub `2639/−1`
  3047→3304 vs sub `3304/−1` 3304→3621 — all four stay solid).
- **Direction-agnostic**: a `+1` and a `−1` sub crowding the same candles are
  still ordered (sub `3760/−1`'s extension 4000→4200 is dotted under sub
  `4027/+1`, though both are live).
- **Whole segments**: a segment that overlaps for even part of its span is
  dotted end to end (sub `3304/−1`'s extension 3621→3818 overlaps sub
  `3760/−1` only over [3760, 3818] and is dotted in full).
- **No overlap ⇒ solid**, whether or not the structure was ever live in real
  time — sub `454/+1`'s 454→784→917 prefix is solid because nothing else draws
  there. Conversely a live structure's segment CAN be dotted once a later one
  supersedes it, so an active zone may hang on a dotted segment; the real-time
  lifecycle is carried by the zones and by the hover `phase`.
- **Per lens**: overlap is computed over the segments drawn on THAT chart, so a
  sub can be prior on one lens and solid on the other (sub `2639/−1`'s
  3304→3611 is dotted on confluence, where sub `3304/−1` supersedes it, and
  solid on counter, where that sub is not drawn). This is a deliberate
  exception to "every lens draws the sub identically", which still holds for
  the window and the geometry.

**Replaced subs run through one more point.** A sub whose `end_reason` is
`same_dir_replacement` is superseded while price keeps moving, so its final
segment would be dragged from its last confirmed point straight to the handover
candle, through a real swing extreme. Such a sub's line breaks at that extreme
(`_replacement_break_point`, drawn with a **PB dot**): the counter-move extreme
(`−1` sub pulls up → highest high; `+1` → lowest low, the same convention as the
other pullback dots) over `(last point, the REPLACING structure's anchor]`.
Bounding at the replacing anchor is what makes it the structural swing rather
than a later marginal overshoot — sub `3304/−1` breaks at 3760 (0.57806, where
the sibling structures put the swing), not at the literal highest high 3806
(0.57827; MS saw it — sub `3621/+1` fires `CTS_THRESHOLD_UPDATED@3806` — and
kept the swing at 3760). The resulting segments mirror the sibling structures
point for point (sub `3304/−1`'s 3621→3760 IS sub `3621/+1`'s segment; sub
`3621/+1`'s 3760→4000 IS sub `3760/−1`'s), and the break also resolves the
partial overlap: 3621→3760 meets sub `3760/−1` only at the join candle so it
stays solid, while 3760→3818 is prior. **The replacing structure is the SUB's,
pool-wide — the same break on every lens** (2026-10-24, the user's decision —
`plans/qa_chart_edits_inputs/user_spec_20261024_replacement_break.md`): the
sweep keeps the replacer of the record that ENDS the sub (§17.5's winner, on
whichever lens; `PooledStructure.ended_by_sub_id`) and the projection hands its
anchor to the chart (`SidRecord.meta["ended_by_starting_idx"]`), so a lens whose
own record ended later, or that does not draw the replacer at all, breaks alike.
Until then the replacer came from THIS lens's record ending at the sub's end:
confluence sub `2639/−1` (ended at 3491 by its counter record; its confluence
record ran to 3589) ran 3047 → 3490 straight, now 3047 → 3304 (PB) → 3490 as on
counter; and counter sub `3304/−1` (replaced by the confluence-only `3760/−1`)
ran 3621 → 3819 straight, now 3621 → 3760 (PB) → 3819 as on confluence.
**Only replacement** gets this: measured
on the reference window, every reversal-ended sub's final segment already ends
at its counter-move extreme (0–2 candles) and `parent_end` ends at the parent's
candle — they are already one clean wave.

Implementation: `_prior_line_segments` (pure, over every drawn segment of the
lens) decides the set; `_group_flag_runs` merges consecutive segments of one
style into a single trace, the boundary point belonging to both. Ownership is
unchanged and still decides which points exist at all (two ownership layers,
above) — where two same-direction structures claim one candle the older one's
points are hidden, not dotted. **Dots follow their segments:** filled (recent
style) iff the dot ends at least one solid segment, else a dimmed open circle;
a lone point with no segment is solid. **Prev-BOS lines carry no lifecycle or
recency formatting at all** — always solid, and they never make another segment
prior. Hover carries both facts: `phase=live|forming` (the real-time
`idx >= start_idx`) and `layer=recent|prior` (why this style).
Styles: `structure.m15.swing_line_prior` (navy, `dash: dot`, opacity 0.75),
`structure.m15.{cts,bos}_prior` (navy open circles, 1.5-px ring, opacity 0.85)
— renamed from `*_forming` with this rule. Trace names: `M15 swing (prior)
sub{sub_id}_m15s{sid}` / `M15 swing sub{sub_id}_m15s{sid}` (one per run),
`M15 PB→BOS sub{sub_id}`, dots `M15 {kind} sub{sub_id}[ (prior)]`; only the
prev-BOS trace still carries the first record's parent `h1s{parent_sid}c{cycle}`
(informational). All M15 structure dots are `size 3.6` (+20%, 2026-09-20).
The superseded 2026-09-21 rule (solid iff the segment's span touched
`[start_idx, end_idx]`, `_wave_touches_window` / `_split_polyline_by_wave`) is
still what the H1 overlay filter below uses.

**H1 overlay structure lines — lifecycle filter (chart review 2026-09-21).**
On the sub charts the H1 overlay (`_render_h1_overlay`) draws only the H1 waves
that were live at some point: each wave (segment between consecutive H1
points, or the most recent sid's extension to the last candle, or a PB→BOS
line of the prior sid) is drawn iff its candle span intersects its sid's
window `[overlay start, reversal idx]` — the start is a LOCATION
(`_h1_overlay_window_start_by_sid`: the sid's first structural anchor, BOS_0's,
or the reversal handoff for a sid whose predecessor reversed; user decision
2026-09-25, Plan E E3f landing review — NOT `compute_struct_start_by_sid`, whose
base is the first CTS_ESTABLISHED moment, a lifecycle time: keyed on the moment,
a lagging CTS_0's defining leg BOS_0 → CTS_0 would be hidden), the end
`compute_reversal_idx_by_sid` — the same `_wave_touches_window` predicate as the
sub rule, but hidden instead of dotted. Waves never live are not drawn: on the
reference window sid 1's retroactive (1,0)/(1,1) waves 689→710→728→761→826,
which precede its 902 start and crowd the sub structures they overlap; sid 1's
826→905 wave (it spans 902) and everything after it are drawn, and all of sid 0
(incl. its PB@683 → BOS(sid 1)@689 line, inside sid 0's `[96, 902]`). An H1 dot
is drawn iff a drawn wave or PB→BOS line touches it (so the BOS@689 dot stays as
the end of sid 0's PB→BOS line). H1 prev-BOS lines, `bo/pb/pr/rv` labels and
zones are unaffected. **The H1 chart itself is unchanged** — it draws every sid
in full (prior sids at 50%) as the high-level market picture.

**Persisting elements:** KL / POI rectangles render the sub's
snapshots (and the sub's fib boxes since 2026-10-06 — §8: the H1 chart's gate, one "M15 fib" legend toggle) — drawn from the anchor (`base_idx` / `ic_idx`), active from
`start_idx` (the KL/POI first-active clamp), ended at the sub's `end_idx` (the
projection's cap). Opacity is the per-TF tier. **Exception (chart review
2026-09-20, option 1): collapsed-cycle zones are not drawn** — a KL zone with
`status="inactive"` and clamped `confirmed_idx >= end_idx`
(`_zone_render.is_collapsed_cycle_zone`: the sub's forming-phase cycles, which
ended at/before `start_idx`) and the POIs of those cycles
(`collapsed_cycles(sid_kls)` / `is_poi_of_collapsed_cycle`). They existed
geometrically but were never tradeable; the forming dots/lines already show
that geometry; their fibs are not drawn either (2026-10-06, §8). Their rows stay in the KL/POI/fib CSVs. A POI that is inactive
because it never met its activation conditions inside a live cycle (cycle not
collapsed) is still drawn as an outline. The same predicate governs the H1
chart and the H1 overlay (below).

**Every lens draws the same window.** A sub on both charts (e.g. `2639/−1`:
a confluence record reversal-born from `2365/+1` and a counter `first_counter`
record, both in H1 (0,1)) is drawn over the SUB's window `[start_idx, end_idx]`
on both charts — the counter chart draws it from the sub's `start_idx` (2829 on
the reference window), not from its own record's `start_idx` (2843). Reverses
Stage 3.2b's per-lens "earliest trigger of this lens" start.

**Dot hover** (`_render_m15_dots_layer`; every sub CTS / BOS / unconfirmed-CTS / PB dot — the user's spec
2026-10-22, `plans/qa_chart_edits_inputs/user_spec_20261022_dot_hover.md`; e.g. confluence sub 5's BOS on the
reference window):
```
TF=15M
idx=3304                                      ← the dot's ANCHOR candle (not the confirmation moment)
idx_1H=826                                    ← the H1 candle whose hour holds it
kind=BOS
price=0.58105
sub_id=5 | cycle_id=0                         ← NOT the bounded run's internal structure_id (restarts at 0 per sub)
first record: lens=counter | parent_sid=0 | parent_cycle_id=1
fr trigger=subsequent_counter | fr trigger_idx=3491
lens records:
confluence(0,1) | trigger=reversal | trigger_idx=3589 | start_idx=3589
confluence(1,2) | trigger=first_confluence | trigger_idx=3611 | start_idx=3611
phase=forming | layer=recent
sub_window=[3491, 3819] | end_reason=same_dir_replacement
relative_dir=NA (before start_idx)
struct_direction=-1
```
`first record` / `fr trigger` = the sub's POOL-WIDE first record (`SidRecord.meta["first_record"]`) — it can sit on the
other lens (here the counter one), hence its `lens` (`reversal` is the one trigger type that does not name its lens);
when it is on this lens the two lines repeat the first `lens records` entry — kept as an accepted exception to the
no-repeat rule (different meanings; a fixed hover shape; the user 2026-10-22). `lens records` = THIS lens's
TriggerRecords (`attrs["triggers"]`), one per line, in `(start_idx, seq)` order; a zero-length record ends
`| zero_length`; none → `lens records: none`. `sub_window` = the real-time lifecycle `[start_idx, end_idx | open]`,
`end_reason ∈ {reversal, same_dir_replacement, parent_end, open}`. `relative_dir` = the §17.3 step function
(`SidRecord.relative_dir_segments`) at the dot's candle — `NA (before start_idx)` before the sub's start, exactly the
`phase=forming` candles (`multitf/types.sub_relative_dir_at` — the one reader, shared with the per-LP WVMI CSV since
2026-10-26; a live candle without a segment fails loudly; `-` until 2026-10-22).
One builder for the sub lines, `_sub_dot_hover_lines`; the sub KL / POI hovers reuse its `first record:` line
(`_sub_zone_id_lines`, since 2026-10-23 — §5 / §7 "Hover"), and so do the sub fib hover (§8) and the sub wave-candle
hover (§10, with the `sub_window=` line too; both 2026-10-23). The M15 Prev BOS line hover still carries the older
`first record parent_sid=… parent_cycle=…` string. Known (deferred — the
spec file's "Quirks" section): a PB dot's `cycle_id` is a placeholder 0; a PB dot is either a pullback or a
replacement break point (same `kind=PB`).

---

## Config toggles

Chart defaults define toggles for:
- `candle_types`: pinbar, maru, normal, big_maru, big_normal
- `patterns`: engulfing, star, continuous, double_maru, one_maru_continuous, one_maru_opposite
- `struct_state`: labels (bo/pb/pr/rv)
- `range_visual`: rectangles
- `structure`: levels, labels
- `zones`: KL, OB, POI, wave_candles
- `fib`: lines (both charts; the M15 renderer since 2026-10-06 — §8)
- `imbalance`: highlight
- `volume`: bars, ema_line, spike_marker, wvmi_lps (the WVMI LP marks, §13 — drawn only with the bars)
- `range_candle_marker`: False (disabled to avoid overlap with volume spike markers)

---

## Style Registry Pattern

**IMPORTANT:** All chart styling parameters must be defined in `style_registry.py`, not hardcoded in `export_plotly.py`.

This includes:
- Colors, line widths, dash patterns
- Marker sizes, symbols, opacity
- Fill colors, border styles
- Opacity tier multipliers (active vs historical visibility)

**Pattern:**
1. Define style in `style_registry.py` under `STYLE` dict
2. Reference via `_style("key.subkey")` in `export_plotly.py`
3. Always provide sensible defaults in `.get()` calls

**Example:**
```python
# style_registry.py
"my_line": {
    "line": {"width": 2, "color": "black"},
}

# export_plotly.py
line=_style("my_line").get("line", {"width": 2, "color": "black"}),
```

---

## Debug ergonomics

- Always prefer adding new information via:
  1) df columns (so it exports into CSV and can be inspected)
  2) event meta
  3) chart hovertemplate using customdata
- Avoid adding permanent "noisy labels"; keep them behind cfg toggles.
