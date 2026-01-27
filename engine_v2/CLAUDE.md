# CLAUDE.md — Project Context for Claude Code

> This file is automatically read by Claude Code on startup.

## Project Overview

This is an **explainable, visualization-first, event-driven** automated trading engine for Forex. The primary development loop is **backtesting + replay**.

**Core philosophy:** Research engine first, trading bot second. Every decision must be inspectable, replayable, and explainable.

---

## Current Status

**Week 6 of 10-week syllabus**

| Completed | In Progress |
|-----------|-------------|
| KL Zones v1 (creation, expansion, charting) | Identify Start (start candle selection) |
| Market Structure (CTS/BOS/Range/Reversal) | |
| Structure Patterns (breakout patterns) | |
| Candle Classification | |

**Note:** Original syllabus had Zones in Week 7, but we pulled it forward to Week 6. Indicators (original Week 6) deferred to later.

---

## Quick Commands

```bash
# Run replay pipeline (generates charts + CSVs)
python -m engine_v2.run_replay

# Run tests
pytest

# Output location
artifacts/debug/*.csv    # Raw and final dataframes
artifacts/debug/*.html   # Interactive Plotly charts
```

---

## Key Files (Hot Paths)

```
engine_v2/
├── run_replay.py                    # Entry point - run this first
├── config.py                        # Pair/timeframe/date config
├── pipeline/orchestrator.py         # Pipeline ordering (LOCKED)
├── structure/
│   ├── market_structure.py          # CTS/BOS state machine (core)
│   ├── structure_engine.py          # Wrapper for orchestrator
│   ├── identify_start.py            # Start candle selection (NEW)
│   └── structure_v1.py              # Swing/level computation
├── zones/kl_zones_v1.py             # Zone derivation from events
├── patterns/structure_patterns.py   # Breakout pattern detection
├── features/candles_v2.py           # Candle classification
├── charting/
│   ├── export_plotly.py             # Chart generation
│   └── style_registry.py            # Visual styling
└── debug/                           # CSV export utilities
```

---

## Pipeline Ordering (LOCKED)

```
candle features → structure patterns → base features → market structure → KL zones → charting
```

**Critical:** Base features MUST run BEFORE structure so zone resolution is stable.

---

## Key Documentation

| File | What's Inside |
|------|---------------|
| `MARKET_STRUCTURE_SPEC.md` | CTS/BOS/Range/Reversal semantics |
| `KL_ZONES_SPEC.md` | Zone construction, thresholds, expansion |
| `CHARTING_SPEC.md` | Chart overlay rules, style registry |
| `ARCHITECTURE.md` | System design, event contracts |
| `PROJECT_PRINCIPLES.md` | Non-negotiable guardrails |
| `WORKFLOWS.md` | Debugging checklist |

---

## Domain Glossary

### Structure Terms
- **struct_direction**: Regime direction (+1 uptrend, -1 downtrend)
- **CTS**: Continue the Structure — continuation swing confirmed by pullback
- **BOS**: Break of Structure — break level confirmed by breakout
- **range**: Consolidation bounded by (range_hi, range_lo)
- **reversal watch**: Monitoring window after close-break of BOS threshold
- **structure_id**: Regime identifier; increments on reversal
- **cts_cycle_id**: Cycle counter within a structure_id

### Zone Terms
- **base_pattern**: Candle pattern at zone anchor (pinbar, star, long-tail, etc.)
- **base_idx**: Index of the zone anchor candle
- **bounds_steps**: History of zone boundary changes (INIT → expansions)
- **outer/inner threshold**: Zone boundary prices before top/bottom conversion

### Timing Terms
- **apply_idx**: Candle where pattern effect is applied (end_idx or confirmation_idx)
- **confirmed_idx**: Candle where zone becomes confirmed for charting
- **confirmed_at**: Candle index in event meta where level was confirmed
- **too_early**: Flag when identify_start finds extreme before min_history

---

## Guardrails (Summary)

Full details in `PROJECT_PRINCIPLES.md`. Key points:

1. **Research engine first** — every decision traceable to events
2. **Interfaces frozen** — contracts stable, internals can evolve
3. **Event-driven** — state transitions, not bulk transforms
4. **Visibility > performance** — slow but explainable wins
5. **Chart is the debugger** — if it can't be verified visually, it isn't verified
6. **No premature optimization** — no Optuna until logic is trusted

---

## Debug Checklist

When something looks wrong:

1. **Pipeline ordering** — base features before structure?
2. **structure_id filtering** — chart shows most recent structure_id only
3. **Timing indices** — check apply_idx, confirmed_at, confirmed_idx
4. **Thresholds** — zone expansion only from threshold-update events
5. **Config** — correct pair/timeframe/dates in config.py?

---

## Development Workflow

```
1. Run run_replay.py → generate baseline chart
2. Make changes
3. Run pytest (if tests exist)
4. Run run_replay.py → compare to baseline
5. Commit when behavior matches expectations
```

**Branching:** One branch per week (e.g., `week6-kl-zones`). Merge to `main` when Definition of Done is met.

---

## What NOT To Do

- Don't rewrite modules without reading their spec first
- Don't add features outside current week's scope (park in `IDEA_PARKING_LOT.md`)
- Don't optimize before logic is visually validated
- Don't break existing event contracts
- Don't skip chart verification
