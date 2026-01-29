# CLAUDE.md — Project Context for Claude Code

> This file is automatically read by Claude Code on startup.

## Project Overview

This is an **explainable, visualization-first, event-driven** automated trading engine for Forex. The primary development loop is **backtesting + replay**.

**Core philosophy:** Research engine first, trading bot second. Every decision must be inspectable, replayable, and explainable.

---

## Current Status

**Week 6 complete — Ready for Week 7**

| Completed (Week 6) | In Progress |
|-----------|-------------|
| KL Zones v1 (creation, expansion, charting) | |
| Market Structure (CTS/BOS/Range/Reversal) | |
| Structure Patterns (breakout patterns) | |
| Candle Classification | |
| Identify Start (Exception 1 & 2 probe logic) | |
| Multi-structure support (sid 0 → sid 1 transitions) | |

**Note:** Original syllabus had Zones in Week 7, but we pulled it forward to Week 6. Indicators (original Week 6) deferred to later.

**Note:** On any given week, we may deviate slightly from the original 10-week plan. We may also return to prior week topics for additional debugging and checking how they interact with new elements we are building.

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
│   ├── identify_start.py            # Start candle selection
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

**Critical:** Base features MUST run BEFORE structure. See `LANDMINES.md` for details.

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
| `GOTCHAS.md` | Debugging lessons learned |
| `LANDMINES.md` | Critical constraints, things to avoid |
| `GLOSSARY.md` | Domain terminology reference |

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

See `GOTCHAS.md` for detailed debugging lessons.

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
