# Charting Spec (Plotly) — through Week 8

Primary file: `export_plotly.py`
Styles: `style_registry.py`

---

## Philosophy

- The chart is the primary debugger.
- Everything (except OHLC) must be toggleable in config.
- Shapes don't hover, so hover interactions use transparent helper traces.
- **All formatting centralized in `style_registry.py`** for easy adjustments.

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
| Fibonacci | `fib.line`, `fib.label` | Fib retracement lines |
| Imbalance | `imbalance.bullish`, `imbalance.bearish` | Imbalance candle colors |
| Volume | `volume.bar.up/down/neutral`, `volume.ema_line`, `volume.spike_marker` | Volume bars, EMA, spike markers |
| Wave candles | `wave_candle.bullish`, `wave_candle.bearish` | Wave candle vertical lines |
| M15 structure | `structure.m15.swing_line`, `structure.m15.cts/bos` | M15 chart blue elements |
| M15 zones | `zone.m15.kl.buy/sell` | M15 zones on H1 chart (dashed) |
| H1 overlay | `zone.h1_overlay.kl.*`, `zone.h1_overlay.poi.*` | H1 zones on M15 chart |
| Hover lines | `hover_line.range` | Invisible hitbox lines |
| Chart layout | `chart.layout`, `chart.axis` | Background, grid, axis styling |

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
- Style keys: `structure.swing_line`, `structure.cts`, `structure.bos`

### 2) Range rectangles
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
- Stepwise bounds from `meta["bounds_steps"]` still drive per-step (top, bottom) — the fill within the active stretch is intersected with each step's x-range so each step's y-bounds apply.
- Single vertical confirm line at `confirmed_idx`.
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
  "degenerate parent cycles"); see "Collapsed-cycle zones" under the KL section.
- **3-tier opacity (MAIN H1 chart only)**: tier picked by `meta["status"]` (3-state lifecycle convention). Sub charts use per-TF tier.
- **Outline (Item 5)**: single rect outline (no fill) spanning `[start_time, end_time]`. Color matches the side's confirm line (`confirm_line_rgb`: dark olive `60,90,20` buy / dark brick `120,45,15` sell since the 2026-09-21 side tints; the `101,67,33` brown in the code is only a `.get` fallback), width = `confirm_line_width` (=2). Opacity = `confirm_opacity_active × tier`.
- **Fill (Item 5)**: ONE filled rect per active stretch in `meta["activation_history"]` — each `A` event paired with the next `D` (or `end_idx` for trailing activate). Inactive stretches render as outline only.
- **Confirm lines (Item 5)**: ONE vertical line per `A` event in `activation_history` (replaces the prior single line at `confirmed_idx`). For zones derived before activation_history was added, falls back to a single line at `confirmed_idx`.
- Style keys: `zone.poi.buy`, `zone.poi.sell`, `zone.poi.hover_line`

### 8) Fibonacci lines (Week 7)
- Horizontal dashed lines at Fib retracement levels
- Style keys: `fib.line`, `fib.label`

### 9) Volume overlay (Week 7)
- Volume bars at bottom 15% of chart (overlay approach, not subplot)
- Bars colored by `vol_dir`: green (+1), red (-1), gray (0)
- Volume EMA(20) blue line
- Orange diamond markers on candles with volume spikes (`is_vol_spike`)
- Dynamic y-axis auto-scaling on zoom (JavaScript callback)
- Unified border around price + volume as single chart
- Hover data: idx, time, volume/EMA values
- Style keys: `volume.bar.up`, `volume.bar.down`, `volume.bar.neutral`, `volume.ema_line`, `volume.spike_marker`

### 10) Wave candle vertical lines (Week 8)
- Reads `df.attrs["wave_candles"]` (list of `WaveCandleResult`)
- Full-height vertical lines at last pullback and first breakout candle positions
- Green for bullish candle direction, red for bearish; neutral (dir=0) skipped
- Uses `color_rgb` in style so opacity can be composed with tier multiplier
- Opacity follows parent KL zone's 3-tier multiplier (active/recent_inactive/prior_inactive)
- Drawn with `yref="paper"` (y0=0, y1=1) so lines span full chart height
- Filtered to `selected_sids` (same filter as KL zones)
- **Hover overlay:** Invisible `go.Scatter` trace (8px wide, `rgba(0,0,0,0)`) with 12 evenly-spaced y-points per line (Plotly only detects hover near data points, not along line segments)
  - All candles: idx, BOS zone attribution (sid + cycle), raw volume, weighted volume
  - LB/LP only (bottom section): momentum value (Buy/Sell by candle direction), paired weighted volumes
  - WVMI lookup built from `df.attrs["wvmi"]` mapping each idx to `(WVMIRecord, role)` where role is FB/LB/FP/LP
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
- M15.counter KL zones — reads `dfx.attrs["lower_tf_results"]`, dashed
  rectangles with lower opacity, mapped to H1 x-axis
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

---

## Zone fill semantics — Item 5 (2026-05-20)

All zone rectangles (KL + POI, both charts) follow these rules:

| Element | Rule |
|---|---|
| Outline | Always drawn. Spans full `[base_idx, end_idx]` (or `start_time`→`end_time` for POI). For KL with `bounds_steps`, traces a stepped polygon around the outer contour of all steps (single trace, not N rect outlines). Color matches the zone's confirm-line color. Opacity = `confirm_opacity_active × tier`. |
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
- **Entry:** `export_m15_chart_plotly(registry=..., path_id=..., title=..., ...)` — registry-only (§13.5.e); resolves its M15 entity + parent overlay from the registry. Reads sub data from `m15_df.attrs["events" / "kl_zones" / "poi_zones" / "fib_states" / "wave_candles" / "wvmi" / "prev_bos_lines"]` grouped by each snapshot's **`meta["sub_id"]`** (`_sub_identity`) per the `m15_df.attrs["sids"]` `SidRecord` manifest (one row per unique sub rendered on this lens; `_sid_record_identity` = `sub_id` for a sub row, `sub_sid` for a main row), with `m15_df.attrs["triggers"]` (this lens's `TriggerRecord`s) for hover attribution. (Plan C, 2026-09-20 — replaces the pre-pool identity tuple `(parent_sid, parent_cycle_id, sub_sid)`; `sub_sid` no longer exists on any sub artifact.)
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
- **The dot-trace opacity tier is INERT** (audit 2026-09-21; predates the pool). `_m15_opacity_tier_for_events` is called once (the prev-BOS block) and its result is discarded; `_render_m15_dots` ignores `is_active_trigger` / `most_recent_psid` / `recent_cycles`, so every M15 dot and line renders at its flat registry opacity — consistent with "flat opacity" above. `_compute_m15_tier_context_from_sids` (from each sub's FIRST record, `SidRecord.meta["first_record"]`) still feeds the hover's informational parent fields. Removing the dead path is a pending cleanup.

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
`end_reason`). WVMI records are grouped by `sub_id` too but the M15 chart
renders no WVMI hover (wave-candle hover only).

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
drawn points — over the EXTREME candles the line runs through
(`cts_anchor_idx` / `bos_anchor_idx`), never the confirmation candles — including the
most recent internal sid's extension to the last owned candle and each
cross-structure PB→BOS line. Where the structures of two different subs draw
segments over the same candles, the **most recent** structure is solid and the
**prior** one's segment is dotted:

- **Recency** = `_recency_key` = the hierarchical `(parent_sid,
  parent_cycle_id, sub_id)` tuple (the first two from the sub's first record,
  informational; `sub_id` is the canonical monotonic identity). The structure
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
stays solid, while 3760→3818 is prior. **Only replacement** gets this: measured
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
real-time lifecycle window `[struct_start_by_sid, reversal idx]`
(`zones.structure_lifecycle.compute_struct_start_by_sid` with the reversal
handoff / `compute_reversal_idx_by_sid` — the helpers the overlay's wave-candle
block already uses) — the same `_wave_touches_window` predicate as the sub
rule, but hidden instead of dotted. Waves never live are not drawn: on the
reference window sid 1's retroactive (1,0)/(1,1) waves 689→710→728→761→826,
which precede its 902 start and crowd the sub structures they overlap; sid 1's
826→905 wave (it spans 902) and everything after it are drawn, and all of sid 0
(incl. its PB@683 → BOS(sid 1)@689 line, inside sid 0's `[96, 902]`). An H1 dot
is drawn iff a drawn wave or PB→BOS line touches it (so the BOS@689 dot stays as
the end of sid 0's PB→BOS line). H1 prev-BOS lines, `bo/pb/pr/rv` labels and
zones are unaffected. **The H1 chart itself is unchanged** — it draws every sid
in full (prior sids at 50%) as the high-level market picture.

**Persisting elements:** KL / POI rectangles and fibs render the sub's
snapshots — drawn from the anchor (`base_idx` / `ic_idx`), active from
`start_idx` (the KL/POI first-active clamp), ended at the sub's `end_idx` (the
projection's cap). Opacity is the per-TF tier. **Exception (chart review
2026-09-20, option 1): collapsed-cycle zones are not drawn** — a KL zone with
`status="inactive"` and clamped `confirmed_idx >= end_idx`
(`_zone_render.is_collapsed_cycle_zone`: the sub's forming-phase cycles, which
ended at/before `start_idx`) and the POIs of those cycles
(`collapsed_cycles(sid_kls)` / `is_poi_of_collapsed_cycle`). They existed
geometrically but were never tradeable; the forming dots/lines already show
that geometry. Their rows stay in the KL/POI CSVs. A POI that is inactive
because it never met its activation conditions inside a live cycle (cycle not
collapsed) is still drawn as an outline. The same predicate governs the H1
chart and the H1 overlay (below).

**Every lens draws the same window.** A sub on both charts (e.g. `2639/−1`:
a confluence record reversal-born from `2365/+1` and a counter `first_counter`
record, both in H1 (0,1)) is drawn over the SUB's window `[start_idx, end_idx]`
on both charts — the counter chart draws it from the sub's `start_idx` (2829 on
the reference window), not from its own record's `start_idx` (2843). Reverses
Stage 3.2b's per-lens "earliest trigger of this lens" start.

**Hover** (`_render_m15_dots` customdata): `sub_id` (NOT the bounded run's
internal `structure_id`, which restarts at 0 per sub), `struct_direction`,
`relative_dir` at that candle (the §17.3 step function read from
`SidRecord.relative_dir_segments`), `sub window=[start_idx,end_idx|open] <end_reason|open>`,
and `records:` the sub's record list `{lens}({parent_sid},{parent_cycle_id})
{trigger_type} {trigger_idx}→{start_idx}` (a `†` suffix marks a zero-length
record), plus the first record's `parent_sid` / `parent_cycle_id`
(informational). `end_reason ∈ {reversal, same_dir_replacement, parent_end}`
or `open`.

---

## Config toggles

Chart defaults define toggles for:
- `candle_types`: pinbar, maru, normal, big_maru, big_normal
- `patterns`: engulfing, star, continuous, double_maru, one_maru_continuous, one_maru_opposite
- `struct_state`: labels (bo/pb/pr/rv)
- `range_visual`: rectangles
- `structure`: levels, labels
- `zones`: KL, OB, POI, wave_candles
- `fib`: lines
- `imbalance`: highlight
- `volume`: bars, ema_line, spike_marker
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
