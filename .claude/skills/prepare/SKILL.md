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

Get an overview of all modules and their purposes:

```
engine_v2/
├── config.py              # Configuration constants
├── run_replay.py          # Main entry point - orchestrates the pipeline
├── common/
│   ├── types.py           # Core data types (SwingPoint, StructureLevel, etc.)
│   └── utils.py           # Shared utilities
├── data/
│   └── fetcher.py         # OANDA data fetching
├── pipeline/
│   ├── step_*.py          # Pipeline processing steps
│   └── candle_classifier.py
├── structure/
│   ├── swing_detector.py  # Swing high/low detection
│   ├── structure_builder.py # CTS/BOS detection
│   └── bos_projection.py  # BOS target projections
├── zones/
│   ├── kl_zone_builder.py # Key level zone construction
│   └── poi_zone_builder.py # POI zone construction
├── patterns/
│   └── structure_patterns.py # Candle pattern detection
├── charting/
│   ├── chart_builder.py   # Main chart construction
│   ├── style_registry.py  # Visual styling
│   └── overlays/          # Individual overlay modules
└── tests/
    └── test_smoke.py      # Smoke tests
```

### 3. Trace run_replay.py Pipeline

**Read and understand each step in sequence:**

1. **run_replay.py** - Entry point, orchestrates the full pipeline
2. **data/fetcher.py** - Fetches OHLCV data from OANDA
3. **pipeline/step_raw.py** - Raw candle processing
4. **structure/swing_detector.py** - Detects swing highs/lows
5. **structure/structure_builder.py** - Builds CTS/BOS events
6. **structure/bos_projection.py** - Projects BOS targets
7. **zones/kl_zone_builder.py** - Constructs key level zones
8. **zones/poi_zone_builder.py** - Constructs POI zones with Fib levels
9. **charting/chart_builder.py** - Builds the final visualization

**For each module, understand:**
- What data/events it receives as input
- What processing/transformations it performs
- What data/events it produces as output
- How its output feeds into downstream modules

### 4. Understand Data Flow

Trace how data transforms through the pipeline:

```
Raw OHLCV → Swings → Structure Events (CTS/BOS) → Zones → Chart
                ↓
         SwingPoint[]
                ↓
         StructureLevel[] (with bos_type, cts_confirmed, etc.)
                ↓
         KLZone[] + POIZone[]
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
