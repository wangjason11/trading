---
name: prepare
description: Review memory, documentation, and codebase to build context before receiving new feature specs.
user-invocable: true
allowed-tools: Read, Glob, Grep, Task
argument-hint:
---

# Prepare for New Feature Development

Build comprehensive understanding of the codebase before receiving new feature specifications. This ensures you have full context of the architecture, data flow, and how each component impacts the next.

## Instructions

### 1. Review Memory and Documentation

Read all key documentation files to understand current state and constraints:

```
CLAUDE.md                                    # Project status, current week's focus
engine_v2/ARCHITECTURE.md                    # System design, event contracts
engine_v2/PROJECT_PRINCIPLES.md              # Non-negotiable guardrails
engine_v2/GLOSSARY.md                        # Domain terminology
engine_v2/GOTCHAS.md                         # Debugging lessons, common mistakes
engine_v2/LANDMINES.md                       # Critical constraints
engine_v2/structure/MARKET_STRUCTURE_SPEC.md # CTS/BOS/Range/Reversal semantics
engine_v2/zones/KL_ZONES_SPEC.md             # Zone construction and behavior
engine_v2/zones/POI_ZONES_SPEC.md            # POI/Fib zone specification
engine_v2/charting/CHARTING_SPEC.md          # Chart overlay rules
engine_v2/WORKFLOWS.md                       # Development workflows
```

### 2. Review Codebase Structure

Get an overview of all modules and their purposes. **Always verify against actual files** — use Glob to confirm what exists:

```
engine_v2/
├── config.py                        # Pair/timeframe/date config
├── run_replay.py                    # Entry point - run this first
├── common/
│   └── types.py                     # Core data types (StructureLevel, KLZone, etc.)
├── data/
│   └── fetcher.py                   # OANDA data fetching
├── pipeline/
│   └── orchestrator.py              # Pipeline ordering (LOCKED)
├── features/
│   └── candles_v2.py                # Candle classification
├── structure/
│   ├── market_structure.py          # CTS/BOS state machine (core)
│   ├── structure_engine.py          # Multi-structure wrapper
│   └── identify_start.py            # Start candle selection
├── zones/
│   ├── kl_zones_v1.py               # KL Zone derivation from events
│   ├── wave_candles.py              # Wave candle identification
│   ├── fib_tracker.py               # Fibonacci lifecycle management
│   ├── poi_zones.py                 # POI Zone derivation (Fib + IC)
│   └── wvmi.py                      # Wave Volume Momentum Indicator
├── multitf/
│   ├── types.py                     # MultiTFTrigger, LowerTFResult
│   ├── data_bridge.py               # Fetch/prepare lower-TF data
│   ├── uc1_trigger.py               # UC1 trigger detection
│   └── lower_tf_pipeline.py         # Lower-TF pipeline runner
├── patterns/
│   ├── structure_patterns.py        # Breakout pattern detection
│   └── imbalance.py                 # Imbalance (FVG) pattern detection
├── charting/
│   ├── export_plotly.py             # H1 chart generation
│   ├── export_m15_chart.py          # M15 dedicated chart (H1 overlay)
│   └── style_registry.py           # Visual styling
├── debug/
│   └── export_structure.py          # CSV export utilities
└── tests/
    ├── test_smoke.py                # Smoke tests
    └── test_wave_candles.py         # Wave candle tests
```

### 3. Trace run_replay.py Pipeline

**Read and understand the pipeline ordering in `orchestrator.py`:**

1. **features/candles_v2.py** - Candle classification (pinbar, maru, star, etc.)
2. **patterns/structure_patterns.py** - Structure pattern detection
3. **patterns/imbalance.py** - Imbalance (FVG) detection
4. **structure/market_structure.py** - CTS/BOS state machine (via structure_engine.py)
5. **zones/kl_zones_v1.py** - KL zone derivation from structure events
6. **zones/wave_candles.py** - Wave candle identification per KL zone
7. **zones/fib_tracker.py** - Fibonacci lifecycle management
8. **zones/poi_zones.py** - POI zone derivation (Fib + IC)
9. **zones/wvmi.py** - WVMI (after POI zones — needs POI inner bounds for activation gate)
10. **multitf/** - Multi-TF analysis (UC1: 15M reverse from H1 CTS + WVMI activation)
11. **charting/export_plotly.py** - Chart generation with all overlays

**For each module, understand:**
- What data/events it receives as input
- What processing/transformations it performs
- What data/events it produces as output
- How its output feeds into downstream modules

### 4. Understand Data Flow

Trace how data transforms through the pipeline:

```
Raw OHLCV → Candle Features → Patterns → Market Structure (CTS/BOS state machine)
                                                ↓
                                         StructureEvent[]
                                                ↓
                                    KLZone[] + WaveCandleResult[]
                                                ↓
                                    FibTracker → POIZone[]
                                                ↓
                                    Interactive HTML chart with overlays
```

### 5. Report Readiness

Once you have reviewed everything, provide a summary:

1. **Current project status** (from CLAUDE.md)
2. **Key architecture points** relevant to upcoming work
3. **Active constraints/landmines** to keep in mind
4. **Any questions or clarifications** before receiving specs

End with: "Ready for specifications."

## Why This Matters

Building features without full context leads to:
- Architectural violations
- Breaking existing functionality
- Missing integration points
- Redundant implementations

This preparation ensures you can design solutions that fit cleanly into the existing system.
