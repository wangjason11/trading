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
| KL Zones v1 (creation, expansion, charting) | |
| Market Structure (CTS/BOS/Range/Reversal) | |
| Structure Patterns (breakout patterns) | |
| Candle Classification | |
| Identify Start (Exception 1 & 2 probe logic) | |

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

---

## Lessons Learned (Known Gotchas)

### Debugging Philosophy: Understand Before Fixing

**Principle:** It's not about simply making a fix to get the right answers/values. It's more important to understand **why** it was wrong in the first place so we implement the right fix and get the right answers the correct way.

**Why this matters:**
- A "lucky fix" that happens to produce correct output may mask deeper issues
- Without understanding the root cause, similar bugs will reappear elsewhere
- The fix itself might be wrong even if the output looks correct (e.g., fixing sid 1 by accidentally modifying sid 0)
- Future development builds on current understanding — wrong mental models compound

**Approach:**
1. Before changing code, articulate *why* the current behavior is wrong
2. Trace the logic to find the exact point where expected != actual
3. Verify the fix addresses the root cause, not just the symptom
4. Confirm the fix doesn't have unintended side effects on other parts

### Multi-Structure Start Detection (Exception 1 & 2)

**Problem:** After sid N reversal, determining the correct start_idx for sid N+1.

**Flow:**
1. **Exception 1** (in `identify_start.py`): Check if any candle after last confirmed CTS but before reversal has higher high (uptrend) or lower low (downtrend). If so, start from that extreme.
2. **Exception 2** (probe in `structure_engine.py`): If Exception 1 not triggered, run a "probe" (dry run) of MarketStructure from original candidate to reversal_confirmed. If pullback confirmed AND price reached near CTS zone outer bound (within 15 pips of inner), start from that candle instead.

**Key insight:** Probe runs on a **copy** of df with `end_idx` parameter. If Exception 2 triggers, discard probe data entirely. If not, keep probe data and continue.

### `_initial_bos_before_first_cts` Must Respect `start_idx`

**Bug:** When MarketStructure starts from non-zero `start_idx`, the window for finding the initial BOS extreme was `self.df.iloc[0:cts_idx]` — looking back to index 0 instead of `start_idx`.

**Symptom:** First BOS for sid 1 appeared at wrong index (e.g., 652 instead of 689).

**Fix:** Change to `self.df.iloc[self.start_idx:cts_idx]` and offset the result: `bos_idx = self.start_idx + rel`.

### Isolation Principle for Multi-Structure Debugging

**Rule:** When fixing issues in sid N+1, **never modify data for sid N** (events, df columns, zones) prior to reversal.

**Why:** sid 0 events are already committed to `all_events`. Changing BOS/CTS event structure (e.g., switching from anchor idx to apply_idx) affects ALL structures, not just the one being debugged.

### Event Metadata Should Include structure_id and struct_direction

**Why:** Downstream consumers (charting, zone derivation) need to know which structure an event belongs to. Without this metadata, filtering by structure_id requires fragile df lookups.

**Events that need this:** STATE_CHANGED, RANGE_CONFIRMED, RANGE_UPDATED, RANGE_RESET, RANGE_BREAK_CONFIRMED.

### DataFrame Columns Get Overwritten by Subsequent Structures

**Problem:** When processing sid N+1, columns like `market_state`, `structure_id`, etc. are overwritten. After sid 1 runs, querying `df[df["market_state"] == "reversal"]` for sid 0 returns **empty** because those rows now have sid 1's values.

**Symptom:** Range rectangles or zones for sid 0 extend past reversal because the code couldn't find the reversal point.

**Solution:** Use **events** instead of df columns for cross-structure queries. Events are appended (not overwritten) and preserve metadata like `structure_id`. Example:
```python
# BAD: df columns overwritten
rev_mask = df["market_state"] == "reversal"  # Empty for sid 0!

# GOOD: events preserved
rev_events = [ev for ev in events if ev.type == "STATE_CHANGED" and ev.meta.get("to") == "reversal"]
rev_by_sid = {ev.meta["structure_id"]: ev.idx for ev in rev_events}
```

### Range Event Sort Order for Charting

**Problem:** Range events must be sorted carefully for correct rendering:
1. RANGE_STARTED has `confirm_idx` (when event fires) but also `start_idx` (logical start)
2. RANGE_UPDATED events may occur between `start_idx` and `confirm_idx`
3. RANGE_RESET at idx N may coincide with RANGE_STARTED with `start_idx=N`

**Solution:** Custom sort key with correct priorities:
```python
# Priority: RANGE_RESET=0, RANGE_STARTED=1, RANGE_UPDATED=2
# RANGE_RESET must come BEFORE RANGE_STARTED at same idx (close old before open new)
# RANGE_STARTED sorts by start_idx (not confirm_idx) to process before RANGE_UPDATED in its window

def sort_key(e):
    if e.type == "RANGE_STARTED":
        return (e.meta.get("start_idx", e.idx), 1)  # Use start_idx
    if e.type == "RANGE_RESET":
        return (e.idx, 0)  # Highest priority
    return (e.idx, 2)  # RANGE_UPDATED
```

**Why this matters:** Without correct sort order, a new range may be immediately closed by a RANGE_RESET that should have closed the *previous* range.
