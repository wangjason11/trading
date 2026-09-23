---
name: prepare
description: Review memory, documentation, and codebase to build context before receiving new feature specs.
user-invocable: true
allowed-tools: Read, Glob, Grep, Task
argument-hint:
---

# Prepare for New Feature Development

Build comprehensive understanding of the codebase before receiving new feature specifications. This ensures you have full context of the architecture, data flow, and how each component impacts the next.

## When most context is already loaded (targeted refresh)

If much of the codebase has already been read earlier in this session (e.g.,
immediately after an implementation task and `/remember`, or after a prior
`/prepare`), a full re-read from scratch is unnecessary and wasteful of
context budget. In that case:

1. **Confirm which docs/files were read earlier in the conversation** and
   assume they're still fresh unless git indicates changes.
2. **Focus the refresh on the area relevant to the upcoming feature** —
   re-read the specific module(s) and spec(s) the new work will touch, plus
   any neighbors that might be affected.
3. **Skip unchanged broad-scope docs** (e.g., ARCHITECTURE.md, PROJECT_PRINCIPLES.md)
   unless the feature explicitly intersects them.
4. **Still run the readiness report** in Section 5 — but note in the report
   which pieces came from earlier-in-session context vs fresh reads.

The full workflow below is the cold-start protocol. Use it when starting a
fresh session, when it's been many turns since docs were touched, or when
the upcoming feature crosses many modules.

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
engine_v2/IMBALANCE_FILL_SEMANTICS.md        # Canonical imbalance fill predicate (two-stroke state machine)
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
│   └── provider_oanda.py            # OANDA data fetching (`get_history`)
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
│   ├── types.py                     # MultiTFTrigger, LowerTFResult, SidRecord
│   ├── data_bridge.py               # Fetch/prepare lower-TF data
│   ├── uc1_trigger.py               # first_counter trigger detection (+ *_trigger.py / *_pipeline.py per variation)
│   ├── sub_structure_pool.py        # Pool data model: PooledStructure (unique sub), TriggerRecord, UnresolvedTrigger, probe cache (PART4 §17)
│   ├── parent_tables.py             # Static parent-cycle floor/end tables from H1 events (§17.6)
│   ├── lifecycle_sweep.py           # The sub-structure driver — ordered sweep over moments (§17.6)
│   ├── entity_df_mutation.py        # Start resolvers + probe cache, geometry builder, per-sub projection + mirror (§17.8–§17.9)
│   ├── pooled_structure_build.py    # project_to_window (one downstream derivation per unique sub)
│   └── sid_records.py               # SidRecord builders (main + sub)   [lower_tf_pipeline.py was deleted by Plan C, 2026-09-20]
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

### 6. Persist what you found (do not skip)

A `/prepare` that surfaces findings — spec-vs-code divergences, stale docs,
suspected bugs, contradictions between docs — has produced work that lives
only in this session unless you write it down. Gitignored artifacts
(`artifacts/_prep_reports/`, replay logs) and the session scratchpad are
**not** a handoff; a 2026-08 audit session left ten reports there with no
memory entry and they were nearly lost.

- If the session will end before the findings are acted on, run `/remember`
  (at minimum: a memory file + a `MEMORY.md` pointer) before it ends.
- Prefer a **live reproduction** over inference for any load-bearing claim
  (e.g. run the actual probe and capture the real `ProbeResult`, rather than
  inferring `finalize_condition` from CSV values) — one August inference was
  wrong and cost a discussion round to unwind.
- See `memory/feedback_persist_prepare_audits.md`.
- When the upcoming work is a **fix plan for a recorded defect** (a LANDMINES /
  GOTCHAS / memory entry), treat the entry as a pointer, not the spec: re-read
  every code site the mechanism can touch and put an **audit table** in the
  plan (each read/emission site, classified: fix / already bounded /
  equivalent / write-only / out of scope). The 2026-09 MS bounds-leak entry
  named one of four sites, and the fix's real footprint (one probe path, not
  every bounded build) only appeared from the audit. See
  `memory/feedback_cold_review_plans.md`.

## Documentation cadence for the session ahead

`/prepare` opens a session; the continuous-documentation rule governs the rest of
it (`memory/feedback_continuous_documentation.md`, loaded via `MEMORY.md`):
capture learnings as they happen, file + reconcile at every checkpoint. Two things
to do HERE: (1) if `memory/_INBOX.md` is non-empty, drain it before starting new
work — it means a previous session ended before its last checkpoint; (2) treat the
findings of this `/prepare` itself as the session's first capture (Section 6).

## Why This Matters

Building features without full context leads to:
- Architectural violations
- Breaking existing functionality
- Missing integration points
- Redundant implementations

This preparation ensures you can design solutions that fit cleanly into the existing system.
