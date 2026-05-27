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
1. **Exception 1** (in `identify_start.py`): Check if any candle after last confirmed CTS but before reversal has higher high (uptrend) or lower low (downtrend). If so, override start_idx to that extreme.
2. **Exception 2** (iterative probe in `structure_engine.py`): Always runs regardless of Exception 1. Starts from Exception 1's override start_idx if it triggered, or from base last CTS idx if it didn't. Runs a bounded "probe" (dry run) of MarketStructure from candidate to reversal_confirmed. If CTS established AND price reached near CTS zone outer bound (within 10 pips of inner), discard the probe and re-probe from the exception candle. Iterate until no exception triggers or max 10 iterations.

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

## Wave Candle Selection: BOS vs CTS Use Different Metrics

**BOS zones** (both BIB and non-BIB) select last pullback by **closest to outer bound** — the candle whose close is nearest the zone's outer threshold wins, regardless of time order.

**CTS zones** select last breakout by **first qualified candle closing within the zone** — temporal order wins. The search window starts at the breakout pattern's `anchor_idx` (or `cts_anchor_idx - 5` if earlier).

**Gotcha:** When debugging wave candle selection, check which zone type it is first. BOS uses distance metric, CTS uses first-match.

---

## CTS BIB Last Breakout: Pattern Scan-Back for CTS_ESTABLISHED

**Problem:** In `_cts_bib_last_breakout`, when the CTS_ESTABLISHED event candle is a direct match (qualified + wick enters zone + closes within zone), the algorithm returned it immediately. But the event candle is the *last* candle of the pattern — earlier candles in the pattern may also close within the zone and better represent the initial breakout moment.

**Example:** CTS_ESTABLISHED at idx=636 with `anchor_idx=634` (pattern=one_maru_continuous spans 634-636). idx=635 is a qualified maru that also closes within the zone, but the old code returned 636 without checking.

**Fix:** When CTS_ESTABLISHED event candle matches AND has `anchor_idx` in meta, scan from `anchor_idx` forward to `ev_idx` (exclusive). Return the first qualified candle closing within the zone. If none found, fall back to the event candle.

```python
anchor = ev.meta.get("anchor_idx")
if anchor is not None and ev.type == "CTS_ESTABLISHED":
    anchor = int(anchor)
    for j in range(anchor, ev_idx):
        if _is_qualified(df, j, bo_dir) and _closes_within_zone(df, j, zone):
            return j
return ev_idx
```

**Why only CTS_ESTABLISHED:** CTS_UPDATED events don't have pattern info — they're threshold shifts, not pattern-based events.

---

## Zone Proximity Trigger: Scan Starts AT CTS_CONFIRMED Candle

**Rule:** `check_zone_proximity` scans starting at the CTS_CONFIRMED candle
itself (`ev.meta["confirmed_at"]`, NOT `+ 1`). The CTS_CONFIRMED candle is
the candle that confirmed the CTS via pullback pattern — past the pullback
extreme — so it's a valid first candle to evaluate proximity. The first
sd-direction proximity match captures the post-pullback retracement (or,
in some cases, may fire on the confirmation candle itself).

**Earlier mistake to avoid:** scanning from `CTS_ESTABLISHED + 1` was
incorrect because CTS_ESTABLISHED happens at the CTS extreme — between
that candle and CTS_CONFIRMED, the entire pullback unfolds. Any proximity
match during that window is the pullback itself, not a post-pullback
retracement (this was the original `check_proximity_activation` design
intent, kept under the new naming).

**Lesson:** When designing proximity/trigger checks relative to structure
events, carefully consider what phase of the cycle the event occurs in.
CTS_ESTABLISHED ≠ CTS_CONFIRMED in terms of where price is relative to
zones.

---

## Zone Proximity Trigger: Scan Window Must Be Bounded by Zone Activity

**Problem:** The proximity scan must stop when the cycle's zones become inactive. Without a scan end boundary, the scan can continue past the cycle boundary and find proximity matches that belong to a different cycle.

**Example:** sid=0 cycle=1 had CTS_CONFIRMED at idx=640 and BOS_CONFIRMED for cycle=2 also at idx=640. The scan window was empty [641, 639] — correctly skipped. Without the boundary, the scan would have continued to idx=710 and found a match that belonged to structure 1.

**Key boundaries:**
- Next BOS_CONFIRMED for `(sid, cycle_id + 1)` → current cycle zones become inactive
- REVERSAL_CANDIDATE `apply_idx` for sid → structure ends

**Also:** Within the scan window, only use active POI zones at each candle (check `confirmed_idx <= candle <= end_idx`). The BOS KL zone is throughout-active. The CTS KL zone (used for opp_sd triggers) is also throughout-active within this window — `CTS_(n+1)_ESTABLISHED`, which deactivates CTS_n zone, fires AT or AFTER `next_BOS.confirmed_at`, so within `[CTS_n_conf, next_BOS.confirmed_at - 1]` the CTS_n zone is still alive.

---

## Auto-Extend Index Shift: `fetch_history_with_auto_extend` vs `get_history`

**Problem:** `run_replay.py` uses `fetch_history_with_auto_extend` which extends the start date backward (e.g., Dec 1 → Nov 15) to ensure enough candle history for features. Using plain `get_history(CONFIG.pair, CONFIG.timeframe, CONFIG.start, CONFIG.end)` directly produces different indices for the same candle.

**Symptom:** When debugging wave candle selection in a standalone script, all indices were shifted ~240 candles compared to the replay output.

**Rule:** Always use `fetch_history_with_auto_extend` (or replay output data) when investigating candle indices. Never load data with `get_history` directly unless you account for the index offset.

---

## CTS_CONFIRMED `ev.idx` Is the Confirmation Candle, Not the CTS Extreme

**Problem:** `CTS_CONFIRMED` events have `ev.idx` set to the **confirmation candle** (where the pullback confirmed the CTS level), not the CTS extreme candle (where the high/low was set). The extreme candle index is in `ev.meta["cts_anchor_idx"]`.

**Example:** H1 CTS extreme at idx=652, pullback confirmed at idx=683. `CTS_CONFIRMED` event has `ev.idx=683` and `ev.meta["cts_anchor_idx"]=652`.

**Symptom:** Two known bites. (1) Mapping CTS to lower TFs (UC1 trigger): using `ev.idx` started the M15 structure 31 candles too late. (2) `first_confluence` (var 1) probe `end_idx`: the spec said "parent CTS_CONFIRMED idx", which was read as `ev.idx` (the confirmation candle) — over-extending the probe window past the CTS and shifting the confluence sub's validated start. Fixed 2026-05-26 to use `cts_anchor_idx` (`first_confluence_trigger.py`); spec §4.3.2 disambiguated to "CTS extreme idx".

**Fix:** Use `ev.meta["cts_anchor_idx"]` when you need the CTS extreme candle. Nuance on the fallback: `ev.meta.get("cts_anchor_idx", ev.idx)` is fine for a *best-effort display* read, but for a **load-bearing** value (e.g. a probe bound) do NOT fall back to `ev.idx` — a silent fallback re-introduces the exact confirmation-candle bug. `cts_anchor_idx` is an invariant whenever `CTS_CONFIRMED` fired, so a strict `[...]` access is correct and fails loud if the invariant ever breaks.

**Related fields:**
- `ev.idx` = confirmation candle (where pullback confirmed)
- `ev.meta["confirmed_at"]` = same as `ev.idx` (confirmation candle)
- `ev.meta["cts_anchor_idx"]` = CTS extreme candle (where high/low was set)

---

## BOS_CONFIRMED `ev.idx` Is the BOS Extreme, Not the Confirmation Candle

**Problem:** Unlike every other event type, `BOS_CONFIRMED` has `ev.idx` set to the **BOS extreme candle** (where the BOS level price was set), not the confirmation candle. The confirmation candle index is in `ev.meta["confirmed_at"]`.

**Convention mismatch:**
| Event | `ev.idx` means |
|-------|---------------|
| CTS_ESTABLISHED | confirmation candle |
| CTS_CONFIRMED | confirmation candle |
| REVERSAL_CANDIDATE | anchor candle (with `apply_idx` in meta) |
| **BOS_CONFIRMED** | **BOS extreme candle** (exception!) |

**Symptom:** Using `ev.idx` for timing boundaries (scan windows, lifecycle ends) cuts the window short — the BOS isn't actually known until `confirmed_at`, which can be many candles later.

**Bugs caused by this:**
- UC1 lifecycle boundary ended at BOS extreme instead of confirmation → M15 structure too short to finalize
- WVMI scan window ended at BOS extreme → missed valid proximity activations

**Fix:** Always use `int(ev.meta.get("confirmed_at", ev.idx))` when you need the candle where BOS was actually confirmed.

**Related fields:**
- `ev.idx` = BOS extreme candle (where the BOS price level was set)
- `ev.price` = BOS price level
- `ev.meta["confirmed_at"]` = confirmation candle (when the breakout was detected)
- `ev.meta["pb_start"]` = pullback start index

---

## Exception Check Must Exclude CTS_ESTABLISHED Candle

**Problem:** The Scenario 3 BOS_0 probe exception and Exception 2 probe both check whether price returns near the zone after a pullback. The check window was `[CTS_EST.idx, CTS_EST_next.idx]`, which **includes** the CTS_ESTABLISHED candle itself.

**Why this is wrong:** CTS_ESTABLISHED is the pullback confirmation candle — for a bearish structure (sd=-1), CTS marks a HIGH, so the CTS_ESTABLISHED candle's high is naturally near the BOS zone. Including it in the exception check causes false restarts.

**Example:** M15 probe iteration 0 had CTS_EST at idx=53 with high=0.58486, only 4.8 pips from BOS_0 outer=0.58534. This trivially triggered the exception because the pullback confirmation candle was inherently close to the zone.

**Fix:** Start the check from `CTS_EST.idx + 1`:
```python
# Before (wrong):
exc_idx = _find_closest_candle_to_outer(df, cts_est[0].idx, cts_est[1].idx, ...)

# After (correct):
exc_idx = _find_closest_candle_to_outer(df, cts_est[0].idx + 1, cts_est[1].idx, ...)
```

**Applies to:** Scenario 3 Phase 1 probe, Exception 2 in `compute_structure`, and Exception 2 in Scenario 3 Phase 2.

**Lesson:** Same principle as "WVMI scan must start at CTS_CONFIRMED" — the pullback candle itself should never be used to evaluate post-pullback conditions.

---

## MarketStructure Re-Detects Patterns Internally

**Problem:** `MarketStructure` creates its own `BreakoutPatterns(self.df)` instance and calls `detect_best_for_anchor()` at each candle during processing. It does NOT use pre-computed `pat_dir`/`pat_status` columns from the pattern detection pipeline step.

**Why this matters:** When debugging pattern detection issues in structure processing, don't look at `df["pat_dir"]` columns — those were computed by the pipeline's pattern detection step and are NOT used by `MarketStructure`. Instead trace through `BreakoutPatterns.detect_best_for_anchor()`.

**Implication:** Pattern detection in the pipeline (step 2) serves zone base-pattern identification and debugging. Structure detection (step 4) has its own independent pattern detection.

---

## Style Registry `.get()` Defaults: Never Default to an Active Value

**Problem:** When retrieving optional style properties (like `"dash"`) from the style registry, using `line_info.get("dash", "dash")` causes the property to always apply — even when the style intentionally omits it to mean "no dash" (solid line).

**Example:** H1 wave candle lines should be solid after removing `"dash": "dash"` from the style. But the rendering code had:
```python
dash = line_info.get("dash", "dash")  # Default "dash" means always dashed!
```

**Fix:** Default to `None` (or a falsy value) and conditionally apply:
```python
dash = line_info.get("dash")  # None when absent
line_props = dict(color=final_color, width=line_width)
if dash:
    line_props["dash"] = dash
```

**Rule:** When a style property's absence means "don't apply this property", always default to `None`/falsy — never to an active value. This applies to `dash`, `symbol`, `fillpattern`, etc.

---

## CTS_THRESHOLD_UPDATED Event Timing Guarantees

`CTS_THRESHOLD_UPDATED` has specific emission rules that aren't obvious from
event name alone. These guarantees matter whenever you want to hook into the
event (e.g., the M15 reverse cross-fib pre-established phase).

**Emission source:** Only emitted by `_sync_thresholds_from_range` in
`market_structure.py`. Nowhere else.

**Preconditions for emission:**
1. `range_active == True`
2. `range_hi` (sd=+1) or `range_lo` (sd=-1) CHANGED from its previous value
   (monotonic in the structure direction)
3. `prev is not None` (initial range creation doesn't emit; first threshold
   update after range_start does)

**Cycle_id on the event:** Event carries `meta["cycle_id"] = cts_cycle_id`
AT EMISSION TIME. `cts_cycle_id` only increments at the NEXT CTS_ESTABLISHED
(via the "establishing_new_cycle" branch). So:
- Between CTS_n CONFIRMED and CTS_n+1 ESTABLISHED: events carry `cycle_id = n`
- After CTS_n+1 ESTABLISHED: events would carry `cycle_id = n+1` — but
  range deactivates at the breakout that establishes CTS_n+1, so in
  practice no more threshold-updated events fire for cycle n at that point

**Lifetime window:** CTS_n CONFIRMED (creates range) → next BOS_n+1
CONFIRMED (deactivates range) OR reversal OR end-of-data. Within that
window, events fire whenever price makes a new running extreme past
range_hi/range_lo.

**Unification of "price touches CTS_n" with "threshold updates":**
After CTS_n CONFIRMED, `range_hi = cts_n_price` (sd=+1) or
`range_lo = cts_n_price` (sd=-1). The first wick past CTS_n expands the
range bound and emits the first `CTS_THRESHOLD_UPDATED` event. This is
why Mode C's "first cross-fib check when price touches CTS_n" reduces to
listening for the first CTS_THRESHOLD_UPDATED for cycle n — no separate
touch detector needed.

---

## Avoid Unicode Glyphs in Print Statements (Windows cp1252)

**Problem:** Windows' default stdout encoding is cp1252. Characters outside
that range — even common ones like `→` (U+2192), `←`, `≥`, `≤` — crash the
program with `UnicodeEncodeError: 'charmap' codec can't encode character
'→'`. The crash is path-dependent: it only happens when execution
reaches the print statement, so a code path that's rarely exercised may
hide the issue for a long time.

**Symptom:** Replay or test crashes mid-run with a charmap error;
traceback points at a `print(f"... {arrow} ...")` line.

**Fix:** Use ASCII alternatives in `print()` statements:
- `→` / `->` (or `=>`)
- `≥` / `>=`
- `≤` / `<=`
- `±` / `+/-`

Comments and docstrings can keep the unicode (they're not printed via
stdout). Same for log/CSV output if encoded as UTF-8.

**Recently caught:**
- `fib_tracker.py` cross-fib activation print used `→`. Crashed when
  proximity-confirmed CTS triggered the cross-fib path that hadn't fired
  in earlier replays.
- Pattern: defensive `print()` formatting should use ASCII unless the
  caller has explicitly configured a UTF-8 stdout.

---

## Merged Imbalance Instances Carry Hindsight Bias in Backtest

**Problem:** `ImbalanceInstance.gap_top` / `gap_bottom` depend on `df[end_idx+1]`
(the last c3 of the run). Detection is one-pass over the full df, so a merged
run like idx 118-125 is known in full from the start of a backtest. A Fib
activation query at `check_to=CTS_idx=120` sees the complete instance with
bounds computed from `df[126].low` (bullish) — data that wouldn't exist yet
in a true live-timing simulation.

**Why we accept it:** The pre-existing `compute_imbalance` already had 1-candle
hindsight (flag at idx 120 requires df[121]). Merging extends the window
from 1 candle to the length of the run. The fill check is still time-bounded
by `check_to_idx`, so the practical impact on Fib/POI activation is small:
the scan `(end_idx, check_to_idx]` is empty when `end_idx >= check_to_idx`,
resulting in "unfilled" — which matches the expected behavior at that point
in time.

**When to revisit:** When moving to a live pipeline, detection must run
incrementally per candle and instances must grow via explicit
`IMBALANCE_EXTENDED` events (or equivalent in-place updates) so queries only
see what was known at query time.

---

## Imbalance Instances Must Be Re-Computed After Slicing

**Problem:** `df.attrs["imbalances"]` stores `ImbalanceInstance` objects with
`start_idx`/`end_idx` referring to the original df's index space. After
`df.iloc[...].copy().reset_index(drop=True)` (used in the M15 lower-TF
pipeline), the df has new 0..N indices, but the attrs list still holds the
old indices — silently wrong.

**Symptom:** Fib activation and POI IC validation would silently use
instances with indices pointing at the wrong candles (or out of bounds).

**Fix:** In `multitf/lower_tf_pipeline.py`, re-run `compute_imbalance` on the
sliced `trigger_df` after `reset_index`. The `is_imbalance` column itself
copies correctly; only the attrs list is stale.

**Rule:** Any time you slice + reset_index a df that had `compute_imbalance`
run on it, call `compute_imbalance` again. The detection is a local pass so
it produces correct instances relative to the new index space.

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

---

## Narrow-Cycle Gap: Event-Based, Start-of-Candle, Cycle-Scoped

Rules 2/3 in `zones/zone_proximity.py::check_zone_proximity` classify
each candidate trigger candle as narrow-mode or wide-mode based on
`|cts_threshold − bos_threshold|`. Two distinct hazards apply here, and
the current implementation addresses both.

**Hazard 1 — Self-rescue:** an opp_sd candle (wicks toward CTS zone in
struct direction) can extend `cts_threshold` via a same-idx
`CTS_THRESHOLD_UPDATED`. If the gap is evaluated using post-extension
state, the candle's own wick could lift the cycle from narrow → wide
and make itself eligible by its own action.

**Hazard 2 — DataFrame column overwrite:** when a cycle's scan window
extends past a reversal into the next structure's rows (e.g.,
sid=0 cycle=2's scan extends up to `reversal_apply_idx - 1`, but sid=1
starts processing at `reversal_apply_idx` itself), `df["cts_threshold"]`
and `df["bos_threshold"]` get nulled out / overwritten by sid=1. df
columns are NOT safe for cross-structure reads (see LANDMINES
"DataFrame Column Overwrite Hazard").

**Fix:** Compute gap from events, not df columns, applying events with
`idx < current_candle` (start-of-candle):

- `_build_cycle_threshold_timeline(sorted_events, sid, cycle_id)`
  returns the `BOS_CONFIRMED + CTS_CONFIRMED + *_THRESHOLD_UPDATED`
  events for this `(sid, cycle_id)`, sorted by `(idx, type)`.
- A pointer walks the timeline as the per-candle scan advances. At each
  candle `i`, events with `idx < i` are applied to running cts/bos
  thresholds; events at `idx == i` are NOT (start-of-candle).
- For `i == scan_start` (the `CTS_CONFIRMED` candle), an initial pass
  applies all events at `idx ≤ scan_start` — including
  `CTS_CONFIRMED` itself and any same-idx `BOS_THRESHOLD_UPDATED`
  events — so the cycle starts with a defined gap.

**The in-MarketStructure Rule 1 check does NOT need either fix:**
sd-proximity-confirmation candles wick AWAY from CTS (sd=+1: wick
DOWN, while `cts_threshold = range_hi` extends only on UP wicks).
And the check reads `st.cts.price` / `st.bos_threshold` from the
in-memory state object, which is per-MarketStructure-run and isn't
overwritten by the next sid.

**Worked example (narrow cycle with mid-cycle crossing, sd=+1, H1):**
- Events: `BOS_CONFIRMED price=1.1970` at idx 80; `CTS_CONFIRMED
  price=1.2000` at idx 100 (= scan_start); `CTS_THRESHOLD_UPDATED
  price=1.2050 prev=1.2000` at idx 115.
- At idx 100: initial pass applies BOS_CONFIRMED + CTS_CONFIRMED. Gap
  = 30p (< 50p H1 → narrow).
- idx 105 opp_sd attempt: gap = 30p (narrow). Trigger fires, consumes
  Rule 3 opp_sd cap.
- idx 110: narrow, opp_sd cap consumed → skip.
- idx 115 sd attempt: gap still 30p (CTS_THRESHOLD_UPDATED at 115 NOT
  yet applied — start-of-candle). But Rule 3 sd cap not yet consumed
  → sd fires.
- idx 116 onward: gap = 50p (CTS_THRESHOLD_UPDATED at 115 now applied)
  → wide → caps lifted. Alternation continues unrestricted.

**Rule:** Whenever a candle's own action can mutate the state used for
gating, evaluate the gate using state from BEFORE the candle's action.
And whenever a cycle-scoped query can cross into a later structure's
rows, query events (which carry `(sid, cycle_id)` in meta), not df
columns.

**Verification workflow lesson:** the column-overwrite bug above survived
a `/compare` pass that compared aggregate event counts and row-level
diffs. The /compare diff DID flag the relevant cycle (30-row
`market_state` shift, `CTS_RECONFIRMED -1`) but the surface read of
"yep, that's the expected Rule 1 effect" missed that the affected
cycle was ALSO producing 13 proximity triggers when Rule 3 should have
capped it at 2. After implementing per-cycle proximity rules, ALWAYS
also drill into per-cycle outcomes — run `engine_v2.debug.zone_proximity_diag`
and check the per-cycle `alt_list` counts against expectations. Aggregate
event counts can hide cycle-specific bypass bugs (Rule 3 capping correctly
in one narrow cycle but bypassed in another wouldn't change global
BOS/CTS counts at all).

---

## Proximity Check Was Pattern-Gated, Now Per-Candle

**Problem:** Stage 1 + Stage 2 spec said "per-candle proximity check after
CTS_ESTABLISHED" — but the implementation lived inside `_step_anchor`'s
no-winner branch (historical lines ~824-836), meaning it only ran on anchor
candles where no breakout/pullback/reversal pattern fired.

**Why this miss matters:** A multi-candle pattern with `anchor=A,
apply_idx=A+2` causes the main loop in `run()` to jump from anchor `A`
directly to `A+3` (via `_step_anchor` returning `next_i = apply_idx + 1`).
Candles `A+1` (back-fill via `_replay_step_no_patterns(freeze_range=True)`)
and `A+2` (apply via `_apply_pattern_at_apply_idx` then
`_replay_step_no_patterns`) are never visited as anchors — so the
per-anchor proximity check was structurally blind to them.

**Concrete miss (2026-05-13, sid=1 cycle=1, NZD_USD H1):** candle 809
surged 173 pips up to wick into the in-flight POI proximity band, but it
was a back-fill candle of a `one_maru_continuous` pullback pattern
(`apply_idx=810`). Under per-anchor-without-winner, the proximity check
never saw it; CTS confirmed at 810 via pullback. Under per-candle (Fix B),
it fires at 809 (modulo the Scenario 2 mismatch — see related notes).

**Fix shape (Fix B, landed 2026-05-13):**
1. Extracted proximity check into `_maybe_confirm_cts_via_proximity(i)` method
   on `MarketStructure`.
2. Called at top of `_replay_step_no_patterns(i)` after
   `_maybe_update_cts_pre_confirm` (so any CTS update at the current candle
   is reflected first; the `i > st.cts.idx` guard naturally excludes the
   candle that just updated CTS).
3. Inline block in `_step_anchor`'s no-winner branch deleted.
4. Now runs on every candle (anchor, back-fill, apply, fallthrough).

**Order matters within `_replay_step_no_patterns`:** place the proximity
hook AFTER `_maybe_update_cts_pre_confirm` but BEFORE `_bos_barrier_step` /
`_write_df_row`. If proximity confirms CTS, the range gets created via
`_fire_cts_confirmation_via_proximity` (Option B: range seeded by
proximity candle's wick), and subsequent steps see the correct state.

**Related issue surfaced by per-candle:** Fix B exposed a pre-existing
divergence between MarketStructure's in-flight POI snapshot and the
downstream FibTracker-derived POI zones (Scenario 2 cross-cycle). See
`memory/project_proximity_scenario2_fix.md` for the resolution plan.

---

## POI Inner Snapshot Was Stale on Raw-Extreme CTS_UPDATED

**Problem:** `_refresh_poi_inners_for_cycle()` was called only from
`_apply_pattern_at_apply_idx` (around line 1354 in `market_structure.py`),
which handles CTS_UPDATED events emitted via the breakout-pattern path.
But CTS can also UPDATE via the raw-extreme path in
`_maybe_update_cts_pre_confirm` (called per candle from
`_replay_step_no_patterns` with `via="replay_raw"`). That path did NOT
refresh the snapshot.

**Symptom (2026-05-13, sid=1 cycle=2, NZD_USD H1):** CTS_ESTABLISHED at
idx 902 (price=0.57228) snapshotted POI inners as `[0.57813]` (POI #2
only). Then raw-extreme CTS_UPDATEDs at idx 903 (price 0.57159) and 905
(price 0.57112) extended CTS but did NOT refresh — snapshot stayed
`[0.57813]`. The proximity check thereafter saw only POI #2; it should
have included POI #1 (inner=0.57720) which
`compute_poi_inners_for_cycle(cts=905/0.57112)` correctly returns. Result:
proximity didn't fire at candle 926 (the visually obvious trigger, wicked
into POI #1's band) — fired later at 935 against POI #2 instead, with a
much shallower retracement.

**Fix shape (Fix A, landed 2026-05-13):** add
`self._refresh_poi_inners_for_cycle()` after each `st.cts = Point(...)`
assignment in both branches of `_maybe_update_cts_pre_confirm`.

**Diagnostic technique used:** monkey-patch `_refresh_poi_inners_for_cycle`
to log invocations and the returned `st.poi_inners_for_cycle`. Also
monkey-patch `_check_proximity_at_candle` to trace per-candle invocations
and what `poi_inners` it received. Both invaluable when the divergence is
between "what got computed by the resolver" vs "what the proximity gate
saw at this candle".

**Lesson:** Whenever multiple code paths can emit the same conceptual event
(here, CTS_UPDATED), audit ALL paths for required side effects (here, POI
snapshot refresh). Memory's claim "POI snapshot refreshed at CTS_ESTABLISHED
+ each CTS_UPDATED" was accurate AS SPEC but the implementation missed the
raw-extreme path — exactly the kind of spec/code drift covered by
`memory/feedback_spec_writing_precision.md`.

**Closure (2026-05-13):** Fix A is no longer the full story. The 935 → 926
shift also required the Scenario-2 anchor agreement fix (in-flight POI in
MarketStructure now picks anchors via the shared
`zones/fib_tracker.py::select_fib_anchor_for_cycle` utility used by
FibTracker downstream). Without that, MS's in-flight resolver could
silently disagree with FibTracker on which POI inners exist for a cycle
— a separate hazard from the stale-snapshot one this entry describes.
See LANDMINES "Scenario 2 anchor agreement" for the closing rule.

---

## Fib Retracement Zone Math Is Direction-Aware

**Problem:** Computing a Fibonacci retracement level requires the
direction-aware formula. The bullish formula and the bearish formula
give DIFFERENT prices for the same `pct`, and using the bullish formula
for both directions silently places the zone on the wrong side of the
swing.

**The math** (matches `features/fibonacci.py::calculate_fib_price`):

```python
range_size = anchor_high - anchor_low
retracement = range_size * (pct / 100.0)
if direction == 1:   # bullish (price moved up, retracement DOWN from high)
    fib_level = anchor_high - retracement
else:                # bearish (price moved down, retracement UP from low)
    fib_level = anchor_low + retracement
```

**Anchor convention** (matches `FibTracker._create_fib_retracement`):
- For `sd == +1` (bullish): `anchor_high = cts_price`, `anchor_low = bos_price`
- For `sd == -1` (bearish): `anchor_high = bos_price`, `anchor_low = cts_price`

**Concrete bug (2026-05-18, `_compute_poi_activation_history`):**
the helper computed the 61.8/80% zone bounds using only the bullish
formula `anchor_high - range * pct` for both directions. For sd=-1, this
placed the zone BELOW the swing low instead of inside the swing — every
bearish-cycle POI failed condition 5 (variant overlap) at every candle,
never activated, came out with `status=ended, current=[]`. Fixed by
threading direction into the level math. 7 POIs went from "never
activated" to fully-populated activation histories.

**Rule:** any caller that computes a Fib level outside `FibRetracement`
(e.g., bare math without constructing the dataclass) MUST handle direction.
Prefer constructing a `FibRetracement` and calling `price_at_pct` if the
overhead is acceptable.

---

## Pitfall: Positional resolver args drift when the protocol grows

**Problem:** The `PoiInnersResolver` protocol in `market_structure.py`
grew from 8 → 10 positional args when c0_data + fill_threshold were
threaded in. The first wiring pass passed 9 positional args, with
`st.cycle0_data` (a dict) landing in the 9th slot — which
`compute_poi_inners_for_cycle` still treats as `fill_threshold`. The
inner `float(fill_threshold)` call raised `TypeError: float() argument
must be a string or a real number, not 'dict'`; the function's
try/except swallowed it and returned `[]`. The proximity check then saw
empty POI inners and fired against BOS inner only — a silent regression
that *looked* like the resolver had simply failed to find ICs.

**Diagnostic:** `_refresh_poi_inners_for_cycle` debug print logged
`inners=[]` while a direct test call to `compute_poi_inners_for_cycle`
with the same arguments returned the expected POI inner list. The split
between "works direct, returns [] in-replay" is the tell that an
exception is being swallowed inside the resolver. Temporarily replacing
`except Exception: return []` with `except Exception as _exc:
traceback.print_exc(); return []` surfaced the underlying TypeError on
the first replay.

**Rule:** When you grow a `Callable[..., ...]` protocol with more
positional args, audit every CALL SITE — type hints don't enforce arity
at runtime, and a positional/keyword mix-up will silently truncate to the
wrong parameter binding inside try/except. If the protocol has a
swallowing try/except for safety, plan to temporarily widen it during
the rollout so latent type errors surface.

---

## Pandas `.loc` Has ~35× Overhead vs Numpy Positional in Tight Inner Loops

**Problem:** Inner loops that read OHLC values per candle (e.g.,
precomputing each imbalance instance's `fill_idx`, scanning for a level
crossing, walking a multi-candle range to find an extreme) commonly reach
for `df.loc[idx, "l"]` or `series.loc[idx]`. Each `.loc` lookup pays ~5–30µs
of Python overhead (label resolution + slow path). Scaled up to ~200K
iterations per call across many instances, this can silently turn a
millisecond-class operation into a second-class one.

**Microbench (3000-candle df, 60 inner scans, 10 reps):**

| Approach | Time/call |
|---|---|
| `df["l"].loc[idx]` per element | 0.70 ms |
| `np.where(arr <= level)` vectorized | 0.16 ms |
| `arr[idx]` positional per element | 0.02 ms |

Numpy positional is ~35× faster than pandas `.loc`. Vectorized `np.where`
is ~4× faster than `.loc` but has setup overhead that loses to positional
loops when scan lengths are short.

**Pattern that triggers it:** `for inst in N_instances: for idx in
range(start, end): if df.loc[idx, "l"] <= level: ...`. Worst when both
loops are wide.

**Fix pattern:**

```python
# Once, outside both loops:
l_arr = df["l"].to_numpy(dtype=float, copy=False)
df_idx_arr = df.index.to_numpy()
if df_idx_arr.dtype.kind in "iu" and len(df) > 0 \
        and df_idx_arr[0] == 0 and df_idx_arr[-1] == len(df) - 1:
    # Contiguous RangeIndex (the post-`reset_index` case) — pos == idx
    idx_to_pos = None
else:
    idx_to_pos = {int(v): i for i, v in enumerate(df_idx_arr)}

def pos(idx):
    if idx_to_pos is None:
        return idx if 0 <= idx < len(df) else -1
    return idx_to_pos.get(idx, -1)

# Hot loop: vectorized + positional
start_pos = pos(scan_start)
if start_pos >= 0:
    slice_l = l_arr[start_pos:]
    hits = np.where(slice_l <= fill_level)[0]
    if hits.size:
        hit_idx = int(df_idx_arr[start_pos + int(hits[0])]) \
                  if idx_to_pos is not None else start_pos + int(hits[0])
```

**When to bother:** only when the inner loop is genuinely hot. `.loc` is
fine for one-shot reads in setup/diagnostic code. Profile before
optimizing — the per-POI activation history sweep in `zones/poi_zones.py`
landed numpy-based even though measured overhead was ~70ms total across
14 calls; the rewrite was defensive against larger datasets, not a fix
for an observed bottleneck.

**Subtleties:**
- `to_numpy(copy=False)` returns a view when possible; cheap.
- For non-RangeIndex (sparse / non-zero-based), build the
  `idx_to_pos` dict once. `np.searchsorted` works too if df.index is sorted.
- Don't forget to translate the positional hit back to df idx for callers
  that store it (`int(df_idx_arr[hit_pos])`). On RangeIndex they're equal,
  but elsewhere they aren't.

**Concrete example:** `engine_v2/zones/poi_zones.py::_compute_fill_idx_cache`
(landed 2026-05-18).

---

## Sub Lifecycle Cap Is Needed Even When the Parent Cycle Has Ended

**Why it exists:** when a subordinate is built, its MarketStructure runs on
a slice bounded by the parent cycle's lifecycle end (or end-of-data if the
parent cycle is still open). The sub's *most-recent* zone is almost always
**open** at the slice end — not because the parent cycle is still active,
but because the slice **truncates the sub's ongoing structure** and nothing
within the slice fired to close that last zone. The lifecycle cap
(`multitf/entity_df_mutation.py`, `deactivated_by="lifecycle_end"`) closes
these dangling open zones/POIs/fibs at the slice's last candle.

**The trap:** assuming the cap only matters when the parent cycle hasn't
ended (so "the sub zone can stay open like a main zone"). Wrong — the cap is
load-bearing precisely in the **common, closed-parent** case. KL zones only
close via sub-internal events (CTS_ESTABLISHED early-end, same-side
replacement, sub reversal) — all of which are the **sub's own** MarketStructure
events. The parent cycle ending is an H1-level event that bounds the slice
but emits no M15 structure event, so it never closes the dangling zone via
those mechanisms. Without the cap, that zone keeps `end_time=None` and the
chart renders it to the far-right edge (`x1 = t_last_m15`), across regions
the sub never analyzed — zones are NOT clipped by the `owner_by_idx`/§16.5
filter (that only governs dots/lines).

**Contrast with main:** a main open zone at end-of-data genuinely *is*
active at the chart's right edge (`end_time=None` is correct there) — main
runs over all data, has no parent boundary. So the cap is **sub-only**.

**Note (2026-05-25 redesign):** under the sub-structure lifecycle redesign,
this cap survives as the "structure ends when parent next cycle/sid starts"
end condition in the unified lifecycle model. See
`memory/project_sub_structure_lifecycle_redesign.md`.

---

## Diagnosing Backward / Degenerate Sub Zones (`end_time <= start_time`)

**Signature:** a sub KL zone (or POI) whose `end_time` is *before* (backward)
or *equal to* (degenerate, zero-width) its `start_time`. On the chart it
renders as a rectangle running the "wrong way" (e.g. left edge after the
right edge) or as nothing, and "never activates" (the fill is empty because
the active stretch is empty).

**Historical root cause (RESOLVED — cascade removed in redesign Phase 4,
2026-05-25):** the old cascade (`_tag_old_sid_on_overwrite`) capped a
superseded sid's `end_time` to the *overwriter sid's structure-start idx*
(`m15_start_idx`), without checking that the zone's own start was already
at/after that boundary. When a later sub's slice began before a prior sub's
late-formed zone, the cap landed before the zone's start. The sub-structure
lifecycle redesign (merge-and-bound, Phase 2) removed this whole class of bug
by bounding each sid's run so the phantom late zone is never produced; Phase 4
then deleted the cascade helper outright. See
`memory/project_sub_structure_lifecycle_redesign.md`.

**If a backward/degenerate sub zone resurfaces:** the cascade is gone, so a
new instance would point to a different mechanism (e.g. a zone-end cap landing
before its start, or a bad slice→entity translation). The per-sub
`*_kl_zones.csv` + `*_sids.csv` debug exports in `run_replay.py` are still
emitted — inspect them directly for `end_time <= start_time` (check both the
`<` backward and `==` degenerate cases). The purpose-built
`analyze_cascade_backward.py` tool was deleted with the cascade.

---

## Sub WVMI is Trigger-Centric, Not Sid-Centric

**Rule:** A sub sid (M15.confluence / M15.counter) earns a WVMI record **iff a
parent trigger of its entity's class lands inside its active window**
`[start_trigger_idx, m15_end_idx]` — NOT based on how the sid was born. There is
no `use_case` branch; `_assign_trigger_centric_sub_wvmi` (orchestrator) iterates
the parent trigger stream (confluence = main-first-sd-prox + each var 4; counter
= each var 3) and sweeps whichever sid is active at each trigger.

**Why this is the correct model:** sub WVMI exists to read the sub's momentum at
the moment the *parent* signals confluence/counter (§8.3–8.5). A reversal-born
sid is therefore covered only when a *later* parent trigger falls in its window —
exactly mirroring how the **main** entity behaves (a main reversal doesn't
self-create a WVMI record; the new sid waits for its next proximity gate). A sub
reversal is neither an sd-prox nor a CTS-prox parent event, so it is **not** a
trigger.

**The latent bug this replaced (pre-2026-05-26):** the old per-sid dispatch
(`_confluence_wvmi_for_facade` / `_counter_wvmi_for_facade`) switched on
`trigger.use_case` and **hard-skipped `use_case="reversal"`** (`return []`). On
data where no parent trigger ever lands inside a reversal sid's window this gives
the right answer *by accident*; the day a var 4 (confluence) or a later var 3
(counter) fires while a reversal sid is active, the per-sid code wrongly produces
nothing. The trigger-centric pass gates by window, so it is correct in that case
with **no change to current output** (NZD_USD 2025-12→2026-01 has no var 4 and no
trigger landing on a reversal sid, so WVMI counts are unchanged).

**Do NOT** "restore" the old pre-Phase-2 behavior where a reversal continuation
inherited the chain-root's trigger. That coverage was an *over-sweep artifact* of
the old model merging pre/post-reversal structure into one swept sid; the
redesign made the reversal a sid boundary, which correctly makes the reversal sid
independently gated.

---

## POI `confirmed_idx` Is a Lossy Scalar — Use Per-Candle `activation_history`

**Problem:** A POI's `meta["confirmed_idx"]` collapses its whole
`activation_history` to a single int — and `poi_zones.py` (the loop at
~line 545) sets it to the idx of the **LAST** `active=True` event,
overwriting on each activation and **never clearing it on deactivation**.
A POI can flap active→inactive→active multiple times within one cycle, so
this scalar does NOT represent the POI's live state. Any consumer that
treats `[confirmed_idx, end_idx]` as a single active interval is blind to
every earlier active stretch.

**Concrete bug (2026-05-26, sid=1 cycle=2, NZD_USD H1):** POI(inner=0.5772)
had `activation_history = [905:A, 952:D, 953:A, 992:D, 997:A]` →
`confirmed_idx` collapsed to **997**. The `sd:POI` proximity gate
(`zone_proximity.py`) checked `idx < confirmed_idx` and so treated the POI
as inactive until 997 — even though it was genuinely active at idx 926
(inside the `[905,951]` stretch). Result: the cycle's first `sd` trigger
slipped 926→1017, and the `954` `opp_sd` was locked out behind it (opp_sd
can only fire after an sd). Two proximity candles vanished. Fixed by making
the gate ask per-candle activation instead of comparing against the scalar.

**Fix shape:** new pure leaf module `zones/poi_lifecycle.py`:
- `active_stretches_from_history(history, open_end_idx)` — pairs
  `active=True` with the next `active=False` into `(start,end)` stretches
  (the shared primitive; `charting/_zone_render.compute_poi_active_stretches`
  delegates to it so chart fills and the gate can't drift apart).
- `poi_active_as_of(zone, idx)` — True iff `idx` falls in an active stretch
  (respects `end_idx` as a hard cap). The proximity gate calls this.
- `poi_confirmed_idx_as_of(zone, idx)` — start of the active stretch
  containing `idx` (the per-candle replacement for the scalar; the chart
  hover `z_conf` now shows this, e.g. 905 at candle 926, not the collapsed 997).

**Rule:** to answer "is this POI active at candle X?" (or "what is its
confirmed idx as of X?"), ALWAYS walk `activation_history` via
`zones/poi_lifecycle.py` — never read the scalar `confirmed_idx`. The scalar
survives only as a legacy chart fallback (zones with no history) and the
debug print. See LANDMINES "POI Activation Is Per-Candle, Not a Scalar Span"
and POI_ZONES_SPEC "Zone Data Fields".

---

## A POI's existence is gated on its fib being open-or-locked — cap a fib *before* POI derivation and the POI vanishes

**Principle:** POI derivation reads each fib through the gate `(fib.active AND
fib.end_idx is None) OR fib.locked` (`poi_zones.py:446`). `locked` == "the fib's
CTS was confirmed." So a **never-confirmed** fib's ONLY ticket into POI is the
left branch — "still open" (`end_idx is None`). The moment something stamps an
`end_idx` on it, that ticket is void and (being unlocked) it has no `locked`
fallback → **no POI is derived from it**.

**Why this matters (cap ordering):** `_finalize_lifecycle_fields` runs *before*
`derive_poi_zones` in `_run_downstream_pipeline`. If the cycle end (cap) is fed
into the fib's terminal *there* (pre-POI), a still-forming sub fib looks "ended"
at POI time and its POI is suppressed. The baseline applied the sub cap *after*
POI (post-hoc loop), so those fibs were still open at POI time → POI built. Moving
the cap pre-POI is therefore a **behavior change**, not a refactor: it suppresses
POIs from fibs that ended without ever CTS-confirming. (This is intentional for
**real** structural ends — reversal / genuine next-cycle — but the **data/window
edge must NOT cap** them; see LANDMINES + PART4 §5 "data boundary is not a
terminator". `locked` fibs are immune either way — they always feed POI.)

**Rule:** when changing *when* a fib's `end_idx` is set relative to
`derive_poi_zones`, expect sub-POI count changes. KL/POI can inherit a cap freely
(nothing is derived *from* them); fib cannot, because POI is derived from it.

---

## `artifacts/debug/` accumulates files from prior runs with DIFFERENT config windows — glob the current window or you'll compare stale data

**Symptom:** a `/compare`-style row-level diff showed sub KL zones "vanishing"
(10 rows → 4) and a non-open zone's end mysteriously shifting — a phantom
regression that didn't exist.

**Root cause:** `artifacts/debug/` is never cleared between runs. A prior session
used a shorter window (`...2026-01-07...`); the current window is
`...2026-01-20...`. A loose glob like `glob('artifacts/debug/*_M15_counter_kl_zones.csv')`
returned the **stale 2026-01-07 file** (fewer rows, shorter window), not the
current one — so the diff compared the 2026-01-20 baseline against a 2026-01-07
output and reported bogus losses.

**Rule:** when loading a debug CSV for comparison, **match the current config
window in the basename** (`NZD_USD_H1_<start>_<end>_...`), don't bare-glob the
suffix. The `/compare` skill is safe because it iterates the *baseline* folder's
filenames (correct window) and looks each up by basename in `artifacts/debug` —
so trust `/compare`'s md5 verdict over an ad-hoc glob. (Periodically delete
stale-window files from `artifacts/debug/`.)
