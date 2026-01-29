# LANDMINES.md — Critical Constraints and Things to Avoid

> These are the rules that, if violated, will cause hard-to-debug issues. Read before making changes.

---

## What NOT To Do

- Don't rewrite modules without reading their spec first
- Don't add features outside current week's scope (park in `IDEA_PARKING_LOT.md`)
- Don't optimize before logic is visually validated
- Don't break existing event contracts
- Don't skip chart verification

---

## Pipeline Ordering Constraint

```
candle features → structure patterns → base features → market structure → KL zones → charting
```

**MUST:** Base features MUST run BEFORE market structure so zone resolution is stable.

**Why:** Market structure depends on swing detection from base features. If base features run after structure, the structure module will operate on stale or missing swing data, leading to incorrect BOS/CTS detection.

**Enforcement:** Pipeline ordering is defined in `pipeline/orchestrator.py` and marked as LOCKED.

---

## Event Contract Rules

Events are the communication backbone of the system. Breaking contracts causes cascading failures.

**Rules:**
1. **Never change event type names** — downstream consumers filter by exact string match
2. **Never remove fields from event.meta** — existing code may depend on them
3. **Adding fields is OK** — but document them in the relevant spec file
4. **Events are append-only** — never modify an event after it's emitted
5. **Events must include structure_id** — so downstream consumers can filter by structure

**Key events and their consumers:**
| Event Type | Primary Consumer |
|------------|------------------|
| STATE_CHANGED | Charting, Zones |
| CTS_CONFIRMED | Zones, Patterns |
| BOS_CONFIRMED | Zones, Patterns |
| RANGE_* | Charting (rectangles) |

---

## Structure ID Isolation Principle

**Rule:** When fixing issues in structure_id N+1, **never modify data for structure_id N**.

**What this means:**
- Don't change events that were emitted during sid N processing
- Don't overwrite df columns for rows that belong to sid N
- Don't alter zone boundaries established during sid N

**Why:** Each structure_id represents a complete, committed market regime. Retroactively changing sid 0 while debugging sid 1 leads to:
- Inconsistent event history
- Charts that don't match the underlying data
- Bugs that "fix themselves" when you run the full pipeline but reappear in isolation

**Safe approach:** If sid N has a bug, fix it in the code that processes sid N, then re-run the entire pipeline from scratch.

---

## DataFrame Column Overwrite Hazard

**Problem:** Columns like `market_state`, `structure_id`, `swing_dir`, etc. are overwritten when processing each structure. You cannot reliably query df columns for "which rows were in reversal for sid 0" after sid 1 has been processed.

**Landmine:** Code like this will fail silently:
```python
# WRONG: Returns empty or wrong results after sid 1 runs
rev_idx = df[df["market_state"] == "reversal"].index[0]
```

**Safe approach:** Use events for cross-structure queries:
```python
# RIGHT: Events preserve structure_id metadata
rev_event = next(e for e in events
                 if e.type == "STATE_CHANGED"
                 and e.meta.get("to") == "reversal"
                 and e.meta.get("structure_id") == target_sid)
rev_idx = rev_event.idx
```

---

## Zone Threshold Mutations

**Rule:** Zone boundaries should only change via THRESHOLD_UPDATED events, never by direct assignment.

**Why:** The charting system reads `bounds_steps` history to render zone expansions. Direct mutation skips this history, causing:
- Zones that appear at wrong sizes on chart
- Expansion timing that doesn't match actual price action

---

## Index Boundary Errors

Common sources of off-by-one bugs:

| Pattern | Risk |
|---------|------|
| `df.iloc[start:end]` | `end` is exclusive — double-check you're including the right candle |
| `range(start, end)` | Same — `end` is exclusive |
| `df.loc[start:end]` | `end` is **inclusive** — different from iloc! |
| Using confirm_idx vs start_idx | RANGE_STARTED has both — use start_idx for sort order, confirm_idx for timing |
