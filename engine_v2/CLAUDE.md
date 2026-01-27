# CLAUDE.md — Project Context for Claude Code

> This file is automatically read by Claude Code on startup.

## Project Overview

This is an **explainable, visualization-first, event-driven** automated trading engine for Forex. The primary development loop is **backtesting + replay**, not live trading.

**Current status:** Week 6 of 10-week syllabus (KL Zones v1 complete)

**Core philosophy:** This is a **research engine first, trading bot second**. Every decision must be inspectable, replayable, and explainable.

---

## Key Documentation (READ THESE)

Before making changes, read the relevant spec:

| File | Purpose |
|------|---------|
| `README_PROJECT_OVERVIEW.md` | High-level project overview |
| `ARCHITECTURE.md` | System design, invariants, event contracts |
| `MARKET_STRUCTURE_SPEC.md` | CTS/BOS/Range/Reversal semantics |
| `KL_ZONES_SPEC.md` | Zone construction, thresholds, expansion |
| `CHARTING_SPEC.md` | Chart overlay rules, style registry |
| `WORKFLOWS.md` | Replay workflow, debugging checklist |

---

## Non-Negotiable Guardrails

### 1. Research Engine First
- Every decision must be traceable to explicit events
- No trade should exist that cannot be reconstructed visually

### 2. Interfaces Frozen Before Behavior Optimized
- Core objects (Zone, PatternEvent, TradeIntent) have stable interfaces
- Internal logic can evolve; contracts cannot break

### 3. Event-Driven by Design
- All logic uses events and state transitions, NOT bulk dataframe transforms
- Key events: Candle closed, Pattern formed, BOS/CTS created/invalidated, Zone created/mitigated/broken

### 4. Visibility Before Performance
- Slow but explainable > fast but opaque
- No micro-optimizations until logic is trusted

### 5. Charting Is the Primary Test Framework
- If it can't be verified visually, it isn't verified
- The chart is the canonical debugger

### 6. No Premature Optimization
- No Optuna/parameter tuning until behavior is visually trusted
- No "parameter fishing" to fix logical flaws

### 7. ML Gating Rule
- ML may score/rank/filter/adjust confidence
- ML may NOT replace market structure logic or override risk controls

---

## Project Structure

```
forex_engine_v2/
├── engine_v2/
│   ├── pipeline/          # Orchestrator, ordering
│   ├── structure/         # MarketStructure state machine
│   ├── patterns/          # BreakoutPatterns (structure patterns)
│   ├── zones/             # KL zones v1
│   ├── charting/          # export_plotly, style_registry
│   ├── candles/           # Candle classification
│   └── data/              # Data fetching (OANDA)
├── run_replay.py          # Main entry point
└── artifacts/             # Generated charts, CSVs
```

---

## Pipeline Ordering (LOCKED for Week 6)

```
candle features → structure patterns → base features → market structure → KL zones → charting
```

**Critical:** Base features MUST be computed BEFORE structure so zone resolution is stable.

---

## Domain Glossary

- **struct_direction**: Current regime direction (+1 uptrend, -1 downtrend)
- **CTS (Continue the Structure)**: Continuation swing level confirmed by pullback
- **BOS (Break of Structure)**: Break level confirmed by breakout
- **range**: Consolidation bounded by (range_hi, range_lo)
- **reversal watch**: Monitoring window after close-break of BOS threshold
- **structure_id**: Regime identifier; increments on reversal
- **cts_cycle_id**: Cycle counter within a structure_id

---

## Development Workflow

### Before making changes:
1. Run `run_replay.py` on a known dataset
2. Generate chart + CSV artifacts
3. Visually verify current behavior

### After making changes:
1. Run `pytest` (if tests exist for that module)
2. Run `run_replay.py` again
3. Compare chart output to previous
4. Commit only when behavior matches expectations

### Branching rules:
- One branch per week from `main` (e.g., `week6-kl-zones`)
- Merge to `main` only when Definition of Done is met

---

## Debug Checklist (When Something Looks Wrong)

1. **Confirm pipeline ordering** — zones depend on base features; base features before structure
2. **Confirm structure_id filtering** — chart shows most recent structure_id only
3. **Confirm timing indices** — check apply_idx, confirmed_at, confirmed_idx
4. **Confirm thresholds** — zone expansion comes from threshold-update events only

---

## Code Style & Conventions

- Prefer explicit names tied to domain language (CTS/BOS/range/reversal)
- Docstrings carry canonical semantics
- Debug columns use `_debug` suffix
- No silent logic changes — everything shows up in replay charts

---

## Current Week Focus (Week 6)

**Goal:** KL Zones v1 — zones as first-class objects with lifecycle + strength

**Completed:**
- Zone creation from structure events
- Base features (compute_base_features)
- Threshold mapping (pinbar, star, long-tail, etc.)
- Zone expansion via bounds_steps
- Active/inactive zone tracking per structure_id
- Chart overlay with hover info

**Definition of Done:**
- Strong zones visually look strong
- Click zone → see strength + flags
- Zone bounds match expected behavior on test dataset

---

## What NOT To Do

- ❌ Don't rewrite modules without reading their spec doc first
- ❌ Don't add features not in current week's scope (park them in `ideas.md`)
- ❌ Don't optimize before logic is visually validated
- ❌ Don't break existing event contracts
- ❌ Don't skip the chart verification step
