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
candle features → structure patterns → imbalance → market structure → KL zones → wave candles → Fib tracking → POI zones → WVMI → charting
```

**MUST:** Base features MUST run BEFORE market structure so zone resolution is stable.
**MUST:** WVMI MUST run AFTER POI zones — it depends on POI zone inner bounds for its proximity activation gate.

**Why:** Market structure depends on candle classification and pattern detection from base features. WVMI's activation gate checks whether price retraces within 20 pips of zone inner bounds (both KL and POI), so POI zones must exist first.

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

---

## WVMI Constraints

1. **Zero FB/FP volume blocks WVMI creation** — division by zero guard. Ensure candle features (volume) are computed before WVMI runs.
2. **Temp LP only locks on BOS_n+1** — do not assume `lp_locked=True` until BOS of the next cycle confirms. Until then, LP and pullback_momentum can shift every candle.
3. **buy_momentum/sell_momentum are direction-mapped** — for buy zones: buy=breakout, sell=pullback. For sell zones: reversed. Always check `zone_side` when interpreting.
4. **Proximity activation gate is mandatory** — WVMI records are only created for cycles where price actually retraces within 20 pips of the closest active zone inner bound. Scan window: `[CTS_CONFIRMED confirmed_at + 1, next_BOS_CONFIRMED confirmed_at - 1]` or `[CTS_CONFIRMED confirmed_at + 1, REVERSAL apply_idx - 1]`. Uses `ev.meta["confirmed_at"]` for both CTS and BOS (not `ev.idx` — see GOTCHAS.md "BOS_CONFIRMED ev.idx" entry). Uses only active zones at each candle.

---

## Lower-TF Slice Must Include Lookback Buffer

**Rule:** When creating a DataFrame slice for lower-TF structures (e.g., M15), always include **at least 50 candles** before the structure start point.

**Why:** Multiple components depend on neighbor candles:
- `find_base_threshold` uses ±5 neighbor window → fails when `base_idx=0` (no left neighbors)
- Wave candle lookback needs 15-50 candles before the zone
- Imbalance fill checking needs prior candles

**Wrong:**
```python
trigger_df = m15_df.iloc[start_idx:end_idx + 1].copy()  # start_idx becomes 0
```

**Correct:**
```python
lookback = 50
slice_begin = max(0, start_idx - lookback)
trigger_df = m15_df.iloc[slice_begin:end_idx + 1].copy()
start_in_slice = start_idx - slice_begin  # Offset from slice start
```

**Symptom if wrong:** Zone bounds computed incorrectly (e.g., `find_base_threshold` returns None or wrong value), wave candles missing, or imbalance checks failing silently.

---

## Lower-TF Zones and POIs Must Be Capped at Lifecycle End

**Rule:** After running the downstream pipeline for a lower-TF structure, any zone or POI with `end_time=None` must be capped to the last candle of the M15 slice with `deactivated_by: "lifecycle_end"`.

**Why:** The M15 structure's lifecycle is bounded by the parent H1 cycle (ends at next BOS or reversal). Without capping, zones with `end_time=None` render as extending indefinitely on the H1 chart — past the lifecycle boundary.

**Implementation:** In `lower_tf_pipeline.py`, after the downstream pipeline, use `dataclasses.replace()` to cap open zones:
```python
if zone.end_time is None:
    zone = replace(zone, end_time=last_time,
                   meta={**zone.meta, "active": False, "deactivated_by": "lifecycle_end"})
```

---

## Scenario 3 Pip Tolerance Scales with Timeframe

**Rule:** The BOS_0 probe exception pip tolerance must be appropriate for the timeframe:
- **H1:** 10 pips (default)
- **M15:** 3 pips
- **M5:** 1 pip (future)

**Why:** A 10-pip tolerance on M15 is too aggressive — it triggers false restarts because M15 price movements are smaller. The tolerance controls how close a candle must get to the BOS_0 zone outer bound to trigger a probe restart.

**Implementation:** Pass `pip_tolerance_pips` to `compute_structure_scenario_3()`. Lower-TF pipeline uses 3 for M15.

---

## Scenario 3 Constraints

1. **original_bos0_bounds captured at iteration 0 only** — subsequent probe iterations reuse the first BOS_0 zone for exception evaluation. Do not re-derive bounds mid-loop.
2. **Phase 2 only runs if status == "finalized"** — pending results contain Phase 1 probe data only (no multi-structure continuation).
3. **Exception evaluation checks inner bound, not outer** — proximity is measured as "candle high/low within tolerance of zone inner bound" (the bound closer to current price).
4. **Exception check window starts from CTS_EST + 1** — the CTS_ESTABLISHED candle itself is the pullback confirmation, naturally near the zone. Exclude it from the exception check (see GOTCHAS.md).

---

## Lower-TF Scenario 3 May Fail — Handle Gracefully

**Rule:** `compute_structure_scenario_3()` can raise `ValueError` or `IndexError` if the M15 structure reverses before finding a CTS_CONFIRMED. The lower-TF pipeline must catch these and return `None` (skip that trigger), not crash.

**Why it happens:** Some WVMI-activated cycles produce triggers where the M15 data has too few candles or the structure reverses too quickly (e.g., cycle=0 in a short-lived structure). The `identify_start` function raises `ValueError` when it can't find CTS_CONFIRMED before a reversal.

**Implementation:** In `lower_tf_pipeline.py`, wrap `compute_structure_scenario_3()` in a try/except:
```python
try:
    s3_result = compute_structure_scenario_3(...)
except (ValueError, IndexError) as exc:
    print(f"[lower_tf] WARNING: Scenario 3 failed for ...: {exc}")
    return None
```

---

## Wrapping Logic in Loops: Preserve Post-Loop Behavior

**Rule:** When wrapping existing single-shot logic in an iteration loop, the behavior AFTER the loop must remain identical to the original code paths. The loop only changes what happens WITHIN iterations.

**Example:** Exception 2 probe was single-shot: exception → discard + outer loop; no exception → keep probe. When adding iterative re-probing, the post-loop paths must still map to the same two outcomes. Don't introduce new paths (like "keep the re-probe") that didn't exist before.

**Checklist when adding iteration to existing logic:**
1. Identify ALL exit paths in the original code (e.g., "exception triggered" vs "no exception")
2. Map each iteration outcome to the SAME original exit path
3. The iteration only refines WHICH value is used, not WHAT happens with it
