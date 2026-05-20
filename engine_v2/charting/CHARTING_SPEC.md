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
- Candles with FVG imbalance get distinct colors
- Bullish imbalance: Lime Green `rgba(50, 205, 50, 0.8)`
- Bearish imbalance: Amber Yellow `rgba(235, 190, 0, 0.8)`
- Entire candle (body + wicks) colored (Plotly limitation)
- Style keys: `imbalance.bullish`, `imbalance.bearish`

### 7) POI zones (Week 7, updated Item 5 / 2026-05-20)
- Reads `df.attrs["poi_zones"]`
- Fib-based zones from Institutional Candle identification
- **3-tier opacity (MAIN H1 chart only)**: tier picked by `meta["status"]` (3-state lifecycle convention). Sub charts use per-TF tier.
- **Outline (Item 5)**: single rect outline (no fill) spanning `[start_time, end_time]`. Color matches confirm line (`confirm_line_rgb = 101,67,33` brown), width = `confirm_line_width` (=2). Opacity = `confirm_opacity_active × tier`.
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
| Outline color (POI) | Brown `rgb(101,67,33)` (matches confirm line). Sub-native M15 zones use thin black `rgba(0,0,0, line_op)` width 0.5. |

Implementation: shared helpers in `engine_v2/charting/_zone_render.py`:
- `compute_kl_active_stretches(zone, render_end_idx)`
- `compute_poi_active_stretches(zone, render_end_idx)`
- `build_stepped_outline_xy(bounds_steps, x_resolver, x_start, x_end, ...)`
- `select_subordinate_tf_tier(zone_tf, primary_sub_tf, smallest_sub_tf)`

---

## Opacity composition tables

### Main H1 chart — existing 3-tier cascade (unchanged tier logic)

**Tier selection:**
- KL: `meta["active"]==True AND end_time is None` → `active`; else if `structure_id == most_recent_sid` → `recent_inactive`; else `prior_inactive`.
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

**Cascaded sub zones** (`meta["deactivated_by"]="overwritten_by_sid_N"`) stay rendered — their `end_time` was already capped by `_tag_old_sid_on_overwrite`, so their visible extent is naturally truncated to the cascade boundary. They use the same per-TF tier as non-cascaded sub zones.

### Outline color/width matrix

| Context | Outline color | Outline width |
|---|---|---|
| Main H1 chart, KL | `rgb(0,180,0)` (buy) / `rgb(220,0,0)` (sell) at `confirm_opacity_active × tier` | 2 |
| Main H1 chart, POI | `rgb(101,67,33)` (brown) at `confirm_opacity_active × tier` | 2 |
| Sub M15 chart, sub-native KL/POI (M15) | `rgba(0,0,0, line_op)` (thin black) | 0.5 |
| Sub M15 chart, main-TF overlay KL | side color at `confirm_opacity_active × main_tf_tier` | 2 |
| Sub M15 chart, main-TF overlay POI | brown at `confirm_opacity_active × main_tf_tier` | 2 |

These tables are derived from `style_registry.py` + the rendering code in `export_plotly.py` / `export_m15_chart.py` — keep in sync when those values change.

---

## M15 Dedicated Chart (`export_m15_chart.py`)

A separate chart file renders M15 candles with both M15 structure and H1 overlay elements. This is NOT the same as the H1 chart — it has its own rendering logic.

### Architecture
- **Entry:** `export_m15_chart_plotly(m15_df, h1_df, lower_tf_results, ...)`
- **M15 candles** as the base OHLC
- **M15 structure** (swing lines, CTS/BOS dots, prev BOS lines) in **royalblue**
- **H1 overlay** (swing lines, CTS/BOS dots, prev BOS lines) in **black**
- All connector lines are **solid** (no dash) for both M15 and H1

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
