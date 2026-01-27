# Forex Trader Engine v2 — Project Overview (through Week 6)

This repository is building an **explainable, visualization-first, event-driven** automated trading engine for **Forex** (initially designed around OANDA data + execution, with backtesting + replay as the primary development loop).

This doc distills everything agreed/implemented so far (from the original late-Dec 2025 proposal through **Week 6** progress).

---

## What exists today (high level)

### The pipeline (current ordering)
The engine runs a deterministic pipeline and produces:
- A dataframe enriched with candle features, pattern events, market structure labels, and zone metadata.
- A list of PatternEvents (structure patterns).
- A list of StructureLevels (downstream interface).
- A list of KL zones (attached to `df.attrs["kl_zones"]` for chart overlay).

Current orchestrator ordering (Week 6): candle features → structure patterns → KL base features → market structure → KL zones → charting overlay.【fileciteturn2file1】

### The “development loop”
1. Fetch history (OANDA provider) and run **replay** on a fixed dataset.
2. Export CSV artifacts.
3. Render an interactive Plotly chart with all overlays toggleable.
4. Use the chart as the “truth” to validate logic; iterate.

`run_replay.py` shows the current top-level replay workflow and how outputs are exported.【fileciteturn1file10】

---

## Where we are in the 10-week plan (through Week 6)

**Completed (implemented and chart-validated):**
- Candle feature computation + classification (candles_v2)
- Structure pattern engine (BreakoutPatterns: continuous/double_maru/one_maru_*)
- Market Structure engine (CTS/BOS cycles + range logic + reversal watch)
- Charting (export_plotly + style registry)
- KL Zones v1 (event-driven zones derived from structure confirmation events)

**In progress / deferred (known upcoming):**
- Week 6 Part 2: “Identify Market Structure start candle of analysis” has been discussed; final “starting point” logic is intentionally deferred until Zones influence it (per project decision).
- Week 7+: Indicator overlays + additional POI/OB zones + execution layer.

---

## Core invariants (what we will NOT compromise)

1. **Explainability > performance**
   - Every decision should be traceable to explicit events and visible on the chart.
2. **Interfaces before behavior**
   - Downstream contracts remain stable even if logic evolves.
3. **Visualization-first validation**
   - No “silent” logic changes; anything meaningful must show up in replay charts.
4. **Event-driven / sequential**
   - “No skipping”: the structure engine processes anchors sequentially and uses explicit rewinds only when required.
5. **Scope control**
   - Build minimum viable versions first (v1), validate visually, then iterate.

---

## Repository map (key modules)

### Pipeline
- `engine_v2/pipeline/orchestrator.py`: orchestrates the pipeline, attaches zones, returns PipelineResult.【fileciteturn2file1】

### Market structure
- `engine_v2/structure/market_structure.py`: core state machine; emits StructureEvents and writes debug columns to df.【fileciteturn1file0】
- `engine_v2/structure/structure_engine.py`: wrapper that calls MarketStructure and returns df/events/levels (used by orchestrator).

### Structure patterns
- `engine_v2/patterns/structure_patterns.py`: BreakoutPatterns implementation and pattern selection rules.【fileciteturn2file3】

### Zones (Week 6)
- `engine_v2/zones/kl_zones_v1.py`: KL zones v1, derived from structure events; includes base features + thresholds mapping + zone expansion steps.【fileciteturn2file2】

### Charting
- `engine_v2/charting/export_plotly.py`: interactive chart export; overlays for patterns/structure/range/zones; selects zones by most recent structure_id.【fileciteturn2file0】
- `engine_v2/charting/style_registry.py`: styling tokens for chart components (including zones).【fileciteturn1file4】

### Replay / debugging
- `engine_v2/run_replay.py`: fetch, run pipeline, export artifacts, render chart(s).【fileciteturn1file10】

---

## What a new engineer should do first

1. Run `run_replay.py` for a known pair/timeframe and generate:
   - raw CSV
   - final CSV
   - plotly chart HTML (or interactive output)
2. Turn on overlays incrementally:
   - patterns → structure states/labels → range rectangles → structure swing line → KL zones
3. Pick a known index-based scenario and validate:
   - pattern detection timing
   - CTS/BOS confirmation timing
   - range activation/expansion rules
   - reversal watch semantics
   - zone creation, expansion, and active/inactive behavior

---

## Glossary (engine’s domain language)

- **struct_direction**: The direction of the *current structure regime*. +1 uptrend, -1 downtrend.
- **CTS (Continue the Structure)**: A continuation swing level established/confirmed after pullback logic.
- **BOS (Break of Structure)**: A break level that is confirmed on breakout logic (and later used for reversal checks).
- **range**: A consolidation regime tracked with (range_hi, range_lo) and expansion rules.
- **reversal watch**: A temporary monitoring window that starts after a close-break of BOS threshold, freezing the BOS barrier until resolved.
- **structure_id**: A regime identifier; increments on reversal. Zones are visualized for the most recent structure_id.
- **cts_cycle_id**: Cycle counter within a structure_id; increments when establishing a new CTS cycle.

---

## The docs in this bundle

- `ARCHITECTURE.md` — system design, invariants, interfaces, and event taxonomy
- `PATTERNS_SPEC.md` — structure pattern rules + confirmation timing
- `MARKET_STRUCTURE_SPEC.md` — full CTS/BOS/range/reversal semantics
- `KL_ZONES_SPEC.md` — KL zone construction, thresholds mapping, expansion, and chart semantics
- `CHARTING_SPEC.md` — chart overlay rules, style registry keys, and debug ergonomics
- `WORKFLOWS.md` — replay workflow, debugging checklist, branching + PR habits

