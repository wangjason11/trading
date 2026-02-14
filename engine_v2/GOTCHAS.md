# GOTCHAS.md — Debugging Lessons Learned

> Accumulated debugging wisdom from development. These are hard-won insights that help avoid repeat mistakes.

---

## Debugging Philosophy: Trace the Full Flow

**Principle:** When debugging why output differs from expectations or previous iterations, always review the full codebase and trace through the run_replay logic to understand how each step impacts the next. Never look at individual fragments or functions in isolation.

**Why this matters:**
- One small change can have **cascading effects** that completely change structures and events downstream
- A candle classification change → different patterns → different state transitions → different CTS/BOS timing → different zone boundaries
- Looking at just a function's behavior without understanding how it's called and what uses its output will be misleading

**Example (Week 7 debugging):**
- `is_special_maru` fix changed idx=707 from `normal` to `maru`
- This broke the `continuous` Pattern3 (normal + normal + maru) at 707-709
- Without Pattern3, reversal watch at idx=707 failed
- Reversal moved from idx=710 to idx=748 (38 candles later!)
- This shifted all of sid=1's timing, moving BOS from idx=728 to idx=826
- Root cause was only found by tracing: candle classification → pattern detection → state machine → reversal trigger → structure creation

**Approach:**
1. Run replay and capture the full event stream
2. Compare events between "before" and "after" states
3. Find the FIRST divergence point (not the symptom, the root cause)
4. Trace backward: what feeds into that divergence?
5. Trace forward: how does that divergence cascade?

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
2. **Exception 2** (iterative probe in `structure_engine.py`): If Exception 1 not triggered, run a bounded "probe" (dry run) of MarketStructure from original candidate to reversal_confirmed. If CTS established AND price reached near CTS zone outer bound (within 15 pips of inner), discard the probe and re-probe from the exception candle. Iterate until no exception triggers or max 10 iterations.

**Key insight:** Probes run on a **copy** of df with `end_idx` parameter.
- **No exception ever triggered:** Keep the first probe's data (events + levels + df), continue from `reversal_confirmed_idx + 1`.
- **Any exception triggered:** Discard ALL probes. Use the settled `exc2_candidate` as `start_idx` in the outer loop, which runs a full unbounded MarketStructure from there.

---

## Bounded Probe + Same structure_id in Outer Loop = Fresh Run

**Problem:** Exception 2 probes run with `end_idx=reversal_confirmed_idx`. After the probe, the outer `compute_structure` loop continues with the same `structure_id`. If you "keep" a bounded probe's data and then `continue` the outer loop, the outer loop runs a **new** `MarketStructure` for the same sid from `reversal_confirmed_idx + 1` — this is a fresh run that doesn't know about the probe's CTS/BOS.

**When this is correct:** When no exception triggered, the probe is the authoritative data for the portion up to `reversal_confirmed`, and the outer loop's fresh run from `reversal_confirmed + 1` independently continues the structure.

**When this is wrong:** When an exception triggered and you re-probed, keeping the re-probe (bounded) and continuing the outer loop means sid N+1 effectively starts fresh from `reversal_confirmed + 1`, losing the re-probe's context. The correct behavior is to **discard** all probes and let the outer loop run a full unbounded structure from the settled start.

**Rule:** If any exception triggered during iterative probing, never keep the probe. Always discard and pass the settled candidate to the outer loop.

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

---

## FibState: Check Both `active` AND `locked` for IC Detection

**Problem:** When finding IC candidates for POI zones, a Fib that has been "locked" (CTS confirmed) is still valid for IC detection, but the code only checked `fib_state.active`.

**Gotcha:** A locked Fib means CTS was confirmed — the bounds are finalized but the Fib is still valid for zone creation. Once locked, `active=False` but the Fib should still produce POI zones.

**Wrong:**
```python
if not fib_state.active:
    return []  # Misses locked Fibs!
```

**Correct:**
```python
if not (fib_state.active or fib_state.locked):
    return []  # Valid if active OR locked
```

**Symptom if wrong:** POI zones missing for cycles where CTS was confirmed before the current candle.

---

## POI Zone End Time and Status Must Be Calculated

**Problem:** POI zone `end_time` and `status` must be derived from events, not left as defaults.

**End Time Priority (must implement all):**
1. **Reversal:** All zones for the structure end at `reversal_confirmed_idx`
2. **New CTS:** Cycle N zones end when CTS_N+1 is established
3. **No event:** Zone extends to chart end (`end_time = None`)

**Status Logic:**
- If `end_idx is not None`, status MUST be `"inactive"` (zone has ended)
- Only zones with `end_idx = None` AND current cycle AND active Fib can be `"active"`

**Wrong:**
```python
# Status ignores end_idx
status = "active" if is_current and fib_state.active else "inactive"
```

**Correct:**
```python
if end_idx is not None:
    status = "inactive"  # Zone has ended
else:
    status = "active" if is_current and fib_state.active else "inactive"
```

**Symptom if wrong:** Zones extend to chart end instead of stopping at reversal/next CTS, or zones show as "active" with full opacity when they should be faded.

---

## Legacy vs Pipeline Data Divergence

**Problem:** `structure_v1.py` (fractal swing detector) was a legacy module that computed structure levels independently from the pipeline's CTS/BOS state machine. The debug CSV exports used legacy data while charting used pipeline data, making CSV-to-chart comparisons unreliable.

**Symptom:** Structure levels in the CSV didn't match what the chart showed — different indices, different prices, different zone boundaries.

**Resolution:** Removed `structure_v1.py` entirely. Debug CSVs now export from `res.structure` (pipeline data), ensuring CSV and chart always agree.

**Lesson:** When two code paths compute the same concept (e.g., "structure levels") from different sources, they WILL diverge over time. Consolidate to a single source of truth early.

---

## Star Pattern Window: Centered on Anchor, Not Forward-Looking

**Problem:** The old `compute_base_features` used a **forward-looking** window for star patterns, which was incorrect.

**Old (wrong) logic:**
```python
# Forward-looking: anchor is FIRST candle of 3-candle window
star0 = row[anchor]        # maru/normal
star1 = row[anchor + 1]    # pinbar (middle)
star2 = row[anchor + 2]    # maru/normal
```

**Correct logic:**
```python
# Centered: anchor is MIDDLE candle (must be pinbar)
idx1 = anchor - 1    # maru/normal
idx2 = anchor        # pinbar (the BOS/CTS candle)
idx3 = anchor + 1    # maru/normal
```

**Why forward-looking was wrong:**
- The star pattern should center on the BOS/CTS anchor candle
- In a star pattern, the anchor (confirmation candle) IS the pinbar
- Forward-looking incorrectly made the anchor the first maru/normal

**Symptom:** Zones incorrectly identified as "no base star 2nd big" when the anchor wasn't actually part of a valid star pattern. Example: anchor=710 (normal) was matched with 711 (pinbar) + 712 (normal), but the correct check should be 709 + 710 + 711 which fails because 710 isn't a pinbar.

**Resolution:** Base patterns are now identified on-demand with structure-aware context, using centered windows for star patterns.

---

## df Event Columns Can Miss Events Within a Single Structure

**Problem:** The `cts_event` and `bos_event` df columns are one-shot markers that get cleared by `_write_df_row` on the next step. If another operation (e.g., range detection) runs on the same candle after the event is emitted, the marker can be overwritten before the final df snapshot.

**Example:** idx=903 had a `CTS_UPDATED` event (visible in structure_events and structure_levels CSV) but the df column showed `nan` because range processing cleared it.

**This is distinct from the cross-structure overwrite hazard** — that's about sid N+1 overwriting sid N's rows. This is about events being lost *within* the same structure.

**Rule:** Always use the events list (or `_structure_events.csv`) for event comparisons, never df columns like `cts_event`/`bos_event`. The df columns are useful as quick visual indicators but are not authoritative.

---

## Plotly Hover on Vertical Lines Requires Multiple Data Points

**Problem:** `go.Scatter` with `mode="lines"` only triggers hover near actual data points, not at intermediate positions along the line segment. A vertical line with only 2 points (top and bottom) only hovers when the cursor is at the very top or bottom.

**Fix:** Spread multiple evenly-spaced y-values (e.g., 12 points) along the vertical line so hover targets exist throughout its height:
```python
_n_pts = 12
_y_pts = [y_min + i * (y_max - y_min) / (_n_pts - 1) for i in range(_n_pts)]
fig.add_trace(go.Scatter(x=[time] * _n_pts, y=_y_pts, mode="lines", ...))
```

**This does NOT affect horizontal lines** — horizontal hover traces work fine with 2 endpoints because the cursor naturally moves along the x-axis.

---

## WVMI Weight Defaults to 0.7, Not 1.0

**Problem:** `_compute_last_wave_weight` returns 1.0 only for specific candle type + size combinations. The default is 0.7 (not 1.0).

**Weight tiers:**
- **1.0** — `is_big_normal_as0 AND (maru OR normal)`, or `is_big_maru_as0 AND pinbar AND pinbar_dir != wave_dir`
- **0.5** — `round(body_pct * 100) <= 10` (doji-like)
- **0.7** — everything else (default)

**Gotcha:** If you see momentum values that seem "too low", check the weight tier. A large-volume candle with weight 0.7 produces lower momentum than expected.

---

## Wave Candle Lookback Window Differs by Cycle

**Problem:** BOS BIB backward fallback uses different lookback windows depending on cycle:
- **cycle 0:** 15 candles (early structure, less price history)
- **cycle 1+:** 50 candles (more history available, pullback can be further back)

**Gotcha:** If a wave candle appears "too far back" from the anchor, it's likely cycle 1+ using the 50-candle window. If it's "missing" in cycle 0, the 15-candle window may be too narrow.

---

## Wave Candle Selection: Closest to Outer, Not Closest in Time

**Problem:** Wave candles are selected by "close distance to outer bound", not by time proximity to the anchor.

**Why this matters:** Two qualified candles near the anchor — one at idx-3 (close far from outer) and one at idx-8 (close near outer) — the algorithm picks idx-8 because its close is closer to the zone's outer threshold.

**Gotcha:** When debugging "wrong" wave candle selection, check the outer bound distance, not the anchor proximity.

---

## `_cts_from_breakout_event`: Include Confirmation Candle in Extreme Search

**Problem:** `_cts_from_breakout_event` determines the CTS price by finding the extreme (max high for bullish, min low for bearish) across the pattern's candle span. Originally it only searched `[start_idx..end_idx]` (the pattern candles), but CONFIRMED patterns have an additional confirmation candle beyond `end_idx`.

**Symptom:** CTS price didn't reflect the full price range of the confirmed pattern. For a 3-candle continuous pattern confirmed at `end_idx + 2`, the confirming candle's extreme was ignored.

**Fix:** Extend the search span to include `confirmation_idx` when present:
```python
if ev.confirmation_idx is not None:
    e = max(e, int(ev.confirmation_idx))
```

**Rule:** For SUCCESS patterns (no confirmation needed), the span is `[start_idx..end_idx]`. For CONFIRMED patterns, the span is `[start_idx..confirmation_idx]`.
