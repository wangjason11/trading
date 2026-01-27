# Charting Spec (Plotly) — through Week 6

Primary file: `export_plotly.py`.【fileciteturn1file8】  
Styles: `style_registry.py`.【fileciteturn1file4】

---

## Philosophy

- The chart is the primary debugger.
- Everything (except OHLC) must be toggleable in config.
- Shapes don’t hover, so hover interactions use transparent helper traces.

---

## Overlays (current)

### 1) Structure swing line (confirmed CTS/BOS)
- Dots at confirmed CTS and confirmed BOS points
- Straight lines connecting points
- Final segment to last candle close
Uses style keys:
- `structure.swing_line`
- `structure.cts`
- `structure.bos`【fileciteturn1file1】

### 2) Range rectangles
- Segmented whenever bounds change while `range_active==1`.
- Ends on range deactivation or reversal state.
Uses style key: `range.rect`【fileciteturn1file12】

### 3) Reversal candidate markers
Current format uses purple X markers (the “old formatting” retained in export_plotly).【fileciteturn2file0】

### 4) Structure-pattern markers (triangles)
- Filters by enabled pattern names and statuses (SUCCESS / CONFIRMED)
- Uses style keys:
  - `structure.success.up/down`
  - `structure.confirmed.up/down`【fileciteturn1file15】

### 5) KL zones (Week 6)
- Reads `df.attrs["kl_zones"]`
- Filters zones to the most recent `structure_id`
- Deterministic draw ordering:
  - inactive first (under), active last (over)
  - older first, newer last
- Adds:
  - filled rectangle (shape)
  - confirm line (shape)
  - transparent hover lines (scatter traces) because shapes don’t hover【fileciteturn2file0】

Zone styling lives in style_registry under:
- `zone.kl.buy`, `zone.kl.sell`
- `zone.kl.hover_line`【fileciteturn1file5】

---

## Config toggles

Chart defaults define toggles for:
- candle types markers
- patterns
- struct_state labels
- range rectangles
- structure swing line
- zones: KL, OB (future)【fileciteturn1file8】

---

## Debug ergonomics

- Always prefer adding new information via:
  1) df columns (so it exports into CSV and can be inspected)
  2) event meta
  3) chart hovertemplate using customdata
- Avoid adding permanent “noisy labels”; keep them behind cfg toggles.

