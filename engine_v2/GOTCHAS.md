# GOTCHAS.md — Debugging Lessons Learned

> Accumulated debugging wisdom from development. These are hard-won insights that help avoid repeat mistakes.

---

## Debugging Philosophy: Understand Before Fixing

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

---

## Multi-Structure Start Detection (Exception 1 & 2)

**Problem:** After sid N reversal, determining the correct start_idx for sid N+1.

**Flow:**
1. **Exception 1** (in `identify_start.py`): Check if any candle after last confirmed CTS but before reversal has higher high (uptrend) or lower low (downtrend). If so, start from that extreme.
2. **Exception 2** (probe in `structure_engine.py`): If Exception 1 not triggered, run a "probe" (dry run) of MarketStructure from original candidate to reversal_confirmed. If pullback confirmed AND price reached near CTS zone outer bound (within 15 pips of inner), start from that candle instead.

**Key insight:** Probe runs on a **copy** of df with `end_idx` parameter. If Exception 2 triggers, discard probe data entirely. If not, keep probe data and continue.

---

## `_initial_bos_before_first_cts` Must Respect `start_idx`

**Bug:** When MarketStructure starts from non-zero `start_idx`, the window for finding the initial BOS extreme was `self.df.iloc[0:cts_idx]` — looking back to index 0 instead of `start_idx`.

**Symptom:** First BOS for sid 1 appeared at wrong index (e.g., 652 instead of 689).

**Fix:** Change to `self.df.iloc[self.start_idx:cts_idx]` and offset the result: `bos_idx = self.start_idx + rel`.

---

## Isolation Principle for Multi-Structure Debugging

**Rule:** When fixing issues in sid N+1, **never modify data for sid N** (events, df columns, zones) prior to reversal.

**Why:** sid 0 events are already committed to `all_events`. Changing BOS/CTS event structure (e.g., switching from anchor idx to apply_idx) affects ALL structures, not just the one being debugged.

---

## Event Metadata Should Include structure_id and struct_direction

**Why:** Downstream consumers (charting, zone derivation) need to know which structure an event belongs to. Without this metadata, filtering by structure_id requires fragile df lookups.

**Events that need this:** STATE_CHANGED, RANGE_CONFIRMED, RANGE_UPDATED, RANGE_RESET, RANGE_BREAK_CONFIRMED.

---

## DataFrame Columns Get Overwritten by Subsequent Structures

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

---

## Range Event Sort Order for Charting

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

---

## Fib State Storage: Key by (structure_id, cycle_id) Not Just structure_id

**Problem:** When tracking Fib states per cycle, keying by `structure_id` alone causes old cycle states to be overwritten when a new cycle starts.

**Symptom:** Fib lines for earlier cycles don't extend to their correct final CTS because the state was replaced by the next cycle's state.

**Solution:** Use a tuple key `(structure_id, cycle_id)` for the `_fibs` dictionary:
```python
# BAD: Old cycle state lost when new cycle starts
self._active_fibs: Dict[int, FibState] = {}  # {structure_id: state}

# GOOD: Each cycle's state preserved
self._fibs: Dict[tuple, FibState] = {}  # {(structure_id, cycle_id): state}
```

**Why this matters:** For charting, we need each cycle's final locked state to draw Fib lines correctly. With single-key storage, only the most recent cycle's state is available.

---

## Scenario 1 Revert: Zone Touch Direction

**Problem:** When checking if BOS_1 "touches" the prev BOS zone, the comparison direction depends on zone type.

**Gotcha:** For a buy zone (bullish prev structure), the outer threshold is the BOTTOM. "Touching" the zone means price is AT OR ABOVE the outer (crossing into the zone from below).

**Correct logic:**
```python
if prev_sd == 1:  # Buy zone - outer is bottom, zone sits ABOVE
    return bos1_price >= prev_bos_outer  # Touch = at or above
else:  # Sell zone - outer is top, zone sits BELOW
    return bos1_price <= prev_bos_outer  # Touch = at or below
```

**Symptom if wrong:** Scenario 1 doesn't revert when it should (or vice versa).
