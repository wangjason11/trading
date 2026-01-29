# Charting Spec (Plotly) — through Week 7

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
| Hover lines | `hover_line.range` | Invisible hitbox lines |
| Chart layout | `chart.layout`, `chart.axis` | Background, grid, axis styling |

### Changing Colors/Styling

To change any visual element:
1. Edit the corresponding key in `style_registry.py`
2. No changes needed in `export_plotly.py`

---

## Overlays (current)

### 1) Structure swing line (confirmed CTS/BOS)
- Dots at confirmed CTS and confirmed BOS points
- Straight lines connecting points
- Final segment to last candle close
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

### 5) KL zones (Week 6)
- Reads `df.attrs["kl_zones"]`
- Filters zones to the N most recent `structure_id`s (configurable)
- Deterministic draw ordering:
  - inactive first (under), active last (over)
  - older first, newer last
- 3-tier opacity: active=100%, non-active+recent_sid=50%, non-active+prior_sid=20%
- Stepwise bounds from `meta["bounds_steps"]`
- Style keys: `zone.kl.buy`, `zone.kl.sell`, `zone.kl.hover_line`

### 6) Imbalance candle highlighting (Week 7)
- Candles with FVG imbalance get distinct colors
- Bullish imbalance: Lime Green `rgba(50, 205, 50, 0.8)`
- Bearish imbalance: Gold `rgba(255, 215, 0, 0.8)`
- Entire candle (body + wicks) colored (Plotly limitation)
- Style keys: `imbalance.bullish`, `imbalance.bearish`

### 7) POI zones (Week 7)
- Reads `df.attrs["poi_zones"]`
- Fib-based zones from Institutional Candle identification
- Style keys: `zone.poi.buy`, `zone.poi.sell`, `zone.poi.hover_line`

### 8) Fibonacci lines (Week 7)
- Horizontal dashed lines at Fib retracement levels
- Style keys: `fib.line`, `fib.label`

---

## Config toggles

Chart defaults define toggles for:
- `candle_types`: pinbar, maru, normal, big_maru, big_normal
- `patterns`: engulfing, star, continuous, double_maru, one_maru_continuous, one_maru_opposite
- `struct_state`: labels (bo/pb/pr/rv)
- `range_visual`: rectangles
- `structure`: levels, labels
- `zones`: KL, OB, POI
- `fib`: lines
- `imbalance`: highlight

---

## Debug ergonomics

- Always prefer adding new information via:
  1) df columns (so it exports into CSV and can be inspected)
  2) event meta
  3) chart hovertemplate using customdata
- Avoid adding permanent "noisy labels"; keep them behind cfg toggles.
