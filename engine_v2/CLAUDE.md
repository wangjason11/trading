# CLAUDE.md — Project Context for Claude Code

> This file is automatically read by Claude Code on startup.

## Project Overview

This is an **explainable, visualization-first, event-driven** automated trading engine for Forex. The primary development loop is **backtesting + replay**.

**Core philosophy:** Research engine first, trading bot second. Every decision must be inspectable, replayable, and explainable.

---

## Current Status

**Week 8 In Progress** (branch: `week8-volmom-multitf`)

| Part | Focus | Status |
|------|-------|--------|
| Part 1 | Scenario 3 for start candle identification | Pending |
| Part 2 | Volume momentum indicator | Pending |
| Part 3 | Multi-timeframe analysis (subordinate structures + overlay) | Pending |

**Pre-Week 8 fix:** Exception 2 probe relaxed from CTS_CONFIRMED to CTS_ESTABLISHED (`bbb6d32`).

**Note:** Original syllabus had multi-TF in Week 8. Parts 1 & 2 revisit prior-week topics to strengthen the single-TF foundation before Part 3 layers on multi-TF.

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
artifacts/charts/*.html  # Interactive Plotly charts
```

**Slash Commands:**
- `/commit-save [message]` — Commit, run replay, save outputs to timestamped folder
- `/compare` — Compare current replay against last `/commit-save` to detect regressions

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
│   └── identify_start.py            # Start candle selection
├── zones/kl_zones_v1.py             # KL Zone derivation from events
├── zones/poi_zones.py               # POI Zone derivation (Fib + IC)
├── zones/fib_tracker.py             # Fibonacci lifecycle management
├── patterns/imbalance.py            # Imbalance (FVG) pattern detection
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
candle features → structure patterns → base features → market structure → KL zones → POI zones → charting
```

**Critical:** Base features MUST run BEFORE structure. See `LANDMINES.md` for details.

---

## Key Documentation

| File | What's Inside |
|------|---------------|
| `MARKET_STRUCTURE_SPEC.md` | CTS/BOS/Range/Reversal semantics |
| `KL_ZONES_SPEC.md` | Zone construction, thresholds, expansion |
| `zones/POI_ZONES_SPEC.md` | POI zones (Fib + IC) specification |
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

1. **Trace the full flow first** — Never debug isolated functions. One change cascades through:
   `candle classification → patterns → state machine → CTS/BOS → zones → charting`
2. **Understand before fixing** — Find *why* it's wrong, not just *what* to change
3. **Pipeline ordering** — base features before structure?
4. **structure_id filtering** — chart shows most recent structure_id only
5. **Timing indices** — check apply_idx, confirmed_at, confirmed_idx
6. **Thresholds** — zone expansion only from threshold-update events
7. **Config** — correct pair/timeframe/dates in config.py?

See `GOTCHAS.md` for detailed debugging lessons (including cascading effect examples).

---

## Development Workflow

```
1. Run run_replay.py → generate baseline chart
2. Make changes
3. Run pytest (if tests exist)
4. Run run_replay.py → compare to baseline
5. Run /compare → verify no unintended changes to prior logic
6. Commit when behavior matches expectations
```

**IMPORTANT:** Run `/compare` before every commit and merge to catch regressions and unintended side effects.

**Branching:** One branch per week (e.g., `week6-kl-zones`). Merge to `main` when Definition of Done is met.
