# GOTCHAS.md — Debugging Lessons Learned

> Accumulated debugging wisdom from development. These are hard-won insights that help avoid repeat mistakes.

---

## Per-cell `.iloc[]` on a df with heavy `.attrs` → pandas `__finalize__` deepcopy explosion (FIXED 2026-07-06)

**Symptom:** the multi-TF sub-build (`multi_tf_dual`) dominated the replay —
477s of a 535s total. A `cProfile` run showed **822 MILLION `copy.deepcopy`
calls** (~1500s cumtime under the profiler), routed through pandas
`generic.py::__finalize__` (265k calls), all under **one function**:
`patterns/range_label.py::apply_is_range_labels` (19 calls, ~83s each).

**Root cause:** `apply_is_range_labels` read OHLC per cell via
`out.iloc[t]["c"]` / `out.iloc[i]["l"]` inside a double `for` loop. **Every
`.iloc[]` row access builds a row-Series, which calls pandas
`NDFrame.__finalize__`, which does `self.attrs = deepcopy(other.attrs)`.** The
working df's `.attrs` carried the `imbalances` instance list (167
`ImbalanceInstance` dataclasses), so each single cell read deep-copied ~3,100
nested objects. `n × k` cell reads × 16 sub slices ≈ 822M deepcopies. NOT an
algorithmic cost — a pandas-metadata-propagation cost masquerading as one.

**Fix:** numpy-ize the loop — extract `c/l/h/is_range/is_range_confirm_idx/
is_range_lag` to numpy arrays once, run the *identical* iteration on arrays,
write the 3 columns back once. No `.iloc` in the loop → no `__finalize__` → no
attrs deepcopy. **Byte-identical** (21/21 CSVs vs `c61e670`; 394 tests; ref-impl
equivalence on 7 cases incl. edge/attrs). `multi_tf_dual` 477s→59s (**8.1×**);
total replay 535s→111s (**4.8×**), wall 8m58s→1m52s.

**General rule (load-bearing for future perf work):** NEVER do per-cell
`.iloc[]` / `.at[]` reads in a loop on a DataFrame that carries objects in
`.attrs`. pandas deep-copies `attrs` on virtually every operation via
`__finalize__`; when `attrs` holds the imbalance/zone lists (as every per-sub
`trigger_df` does after `compute_imbalance`), any such loop becomes
catastrophic. Extract numpy arrays first — same lesson as "Pandas `.loc` Has
~35× Overhead vs Numpy Positional", **amplified** by the attrs deepcopy. When
hunting the next hotspot (e.g. `ImbalanceInstance.is_filled`, the chart export
path), check first whether it loops with `.iloc`/`.at` on a heavy-attrs df.

**Follow-up 2026-07-06 (same session) — the tax was DISTRIBUTED, so a systemic
fix beat whack-a-mole.** After range_label, a re-profile showed `deepcopy` was
still #1 (203s cumtime under cProfile) but now spread across ~67k
`__finalize__` calls from MANY pandas ops (`_box_col_values` / column boxing,
`_construct_result` / Series arithmetic, `astype`, `isna`, datetime accessor,
`_slice`, `take`, `concat`) throughout the per-sub downstream — not one loop.
Root: every pandas op on a working df deep-copies `df.attrs`, and attrs carry
the `imbalances` list. **Systemic fix: `ImbalanceInstance.__deepcopy__` returns
`self`** (it's frozen + read-only-by-convention; engine code only ever
reads/mutates the ORIGINAL instances — `compute_imbalance` creates them and each
working df pins the same list by reference — so pandas' transient `__finalize__`
copies are never read; sharing identity changes only cost, not values). One
method, no reader changes, kills the distributed tax across ALL downstream ops.
Plus numpy-ized `is_filled` (its `df.loc[idx,col]` per-candle loop = the ~35×
`.loc` scalar overhead). Combined: `multi_tf_dual` 58.6s→35.6s; total
110.8s→77.8s (**6.9× vs the original 535s**). Byte-identical: 21/21 CSVs, 394
tests, 2400-case `is_filled` equivalence. **Lesson: when a deepcopy tax is
spread across many pandas ops rather than one loop, make the heavy `attrs`
payload deepcopy-cheap (`__deepcopy__`→self on frozen/read-only objects) instead
of chasing each call site** — but only when you can prove engine code never
mutates a pandas-transient copy (validate with `/compare`).

---

## Base/inside-bar zone inner could land beyond the outer (inverted zone) — FIXED 2026-06-01

**Symptom:** a `base inside bar` KL zone rendered *above* its own anchor candle
— inner (0.58266) sat *above* the outer = base_high (0.58244), so the zone
detached from and sat outside the base. Surfaced on confluence `(0,0,1)` BOS at
anchor 1797.

**Root cause:** `find_base_threshold` built the inner from ±5 neighbour bodies
testing BOTH `o` and `c` with **no bound against the outer**. A neighbour body
*beyond* the outer (here the peak candle 1794, close 0.58266 > base_high
0.58244) could be selected as the inner → inversion. The candles that qualified
the *inside-bar pattern* (inside the anchor) were not the candles that built the
inner (the peak, above the anchor).

**Fix:** test only the neighbour's **inner-edge body point** (`min(o,c)` when
outer=base_high, `max(o,c)` when outer=base_low), **drop points beyond the
outer**, and take the 2nd such point closest to the outer. Inner is now
guaranteed within the outer, containing the base. Full rule in
`KL_ZONES_SPEC.md` "`find_base_threshold` inner — inner-edge rule".

**Scope of the change (verified by /compare):** surgical — only diverges from
the old logic when a neighbour's inner-edge point is beyond the outer, so a full
replay changed exactly ONE zone (the 1797 one); every well-behaved base/inside-
bar zone was byte-identical, and there was no structure/proximity/POI cascade.
**Follow-up (c) DONE 2026-06-20** — pass over the OTHER zone-forming candle
patterns for the same inversion class. Findings:
- **2-candle + star are inversion-safe BY CONSTRUCTION** — their inner is always
  an `o`/`c`/`mid_price` of a candle *within* the pattern window, and
  `compute_base_window_features` sets `base_low/base_high` = min/max over that
  *whole* window, so `inner ∈ [base_low, base_high]` ⊆ outer always. (No change;
  pinned by `tests/test_kl_zone_thresholds.py`.)
- **pinbar (`find_pinbar_threshold`) WAS inversion-capable** (its inner is a
  neighbour `o`/`c`, sourced from outside the base window, with no bound) → got
  an **inner-side bound** (drop candidates beyond the outer before the
  closest-neighbour pick). Deliberately NOT the base/inside-bar inner-edge rule:
  delegating pinbar to `find_base_threshold` was tried and **reverted** because
  the 2nd-closest-widening + ±5 pooling + body-bottom-only selection *over-widened*
  pinbar zones (H1 sid=1 cyc0 BOS @689: 0.58348→0.58298, 7.6→12.6 pips). Pinbar
  zones must stay tight (key off the pinbar's own body/tail). The bound is inert
  in the normal case → byte-identical. See `KL_ZONES_SPEC.md` "Pinbar-specific
  inner threshold".

---

## "base inside bar" means anchor ENGULFS neighbours, not nested-inside-prior (naming + knife-edge)

`identify_inside_bar_pattern` (kl_zones_v1) flags an anchor `"base inside bar"`
when **≥2 of its ±5 neighbours have their entire low–high range CONTAINED within
the anchor's range** — i.e. the **anchor is the larger, engulfing candle**. This is
the OPPOSITE of the conventional "inside bar" (a small candle nested inside the
prior). It's checked FIRST in `identify_base_pattern` (before 2-candle / pinbar /
star / plain `base`); its inner comes from `find_base_threshold` (same as plain
`base`).

**Knife-edge sensitivity:** the test compares full ranges, so a sub-pip wick on a
single neighbour flips it. Concrete (2026-06-07, NZD_USD M15): swing-high candle
1794 missed `base inside bar` only because neighbour 1793's low sat **0.2 pip**
below 1794's low; had it qualified, 1794's zone inner would route through
`find_base_threshold` instead of the 2-candle `"no base"` path and could shift.
This is current/intended behavior — noted because it's surprising and relevant to
the deferred **(c) other-zone-pattern inversion pass** (pinbar / 2-candle / star).

---

## `bounded.reversal_idx` is the single source — don't re-derive reversal idx from a `market_state` mask in the bounded path (2026-06-01)

**Symptom:** the sub-reversal unified probe (then in `build_one_sid`; since Plan C
`entity_df_mutation._resolve_reversal_start`) drifted every
reversal-born start *later*, and in one cycle collapsed the start onto the
window edge → the reversal-born sid (and its child) were dropped. The probe's
`end_idx` was logged as ~600 candles *past* the actual reversal.

**Root cause:** I re-derived the probe's `end_idx` inline as
`bounded.df.loc[(market_state=="reversal")].index.max()`. Two defects:
- **Missing `& (structure_id == 0)` filter.** `compute_bounded_structure`
  passes `end_idx`, which triggers terminal stamping that marks rows *past*
  `end_idx` as `market_state="reversal"` with `structure_id=-1` (see its own
  comment). Without the structure_id filter the mask swept those phantom rows.
- **`.max()` on a single bounded structure.** Nothing overwrites rows after the
  reversal (no next structure), so the "reversal" state persists to the window
  end → `.max()` drifts to the edge.

**The real lesson (single source of truth):** `compute_bounded_structure`
already exposes the correct value as `bounded.reversal_idx` (`.min()` + the
structure_id filter) — and `build_one_sid` already used it 84 lines earlier for
the lifecycle cap. I re-computed (incorrectly) a value I already had in hand.
**Fix:** `reversal_end_local = int(bounded.reversal_idx)`. *(Plan C, 2026-09-20:
`_resolve_reversal_start` goes further and ASSERTS `bounded.reversal_idx == R - slice_begin`
against the sweep's `natural_reversal_idx` before probing, and the probe's bound is now
spelled `probe_end_idx`; `build_or_get_geometry` sets `sub.natural_reversal_idx =
bounded.reversal_idx + slice_begin` — the one canonical source, read by the sweep, the
record's `reversal` end candidate and the spawn rule.)*

**Why main's idiom didn't transfer:** `compute_structure` (main) uses
`rev_mask.index.max()` correctly because it (a) filters by `structure_id`, (b)
passes no `end_idx` (no terminal stamping), and (c) runs a *next* structure that
overwrites post-reversal rows. Copying the surface idiom into the bounded
single-structure context without re-checking those invariants is the trap —
the same class-2 "correct intent, untraced data-dependency across a seam" bug
as [[feedback-implement-against-docs]]. See also [[feedback-single-source-of-truth]].

**Generalize:** before deriving a value from raw df state, check whether the
function/result you already hold exposes it canonically. Re-derivation is where
bounded-vs-unbounded (and slice-vs-entity) invariant mismatches hide.

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

> **⚠ Step 4 (2026-06-20) — this section + the next ("Bounded Probe + Same
> structure_id") now describe the LEGACY path only.** `compute_structure` (H1
> main) no longer uses Scenario 2 / Exception 1 / Exception 2 for reversals — it
> migrated to `unified_probe` + scan-from-start (the same path subs use; see
> MARKET_STRUCTURE_SPEC "Per-reversal continuation differs by function"). The
> Exc1/Exc2 + bounded-probe-keep behavior below survives only in
> `compute_structure_from_start` (no production caller) and
> `compute_structure_scenario_3` Phase 2 (tests). Three Step-4 gotchas for the
> NEW main path:
> - **`reversal_start_idx == reversal_confirmed_idx`** — the per-sid reversal
>   mask `(market_state=="reversal") & (structure_id==sid)` matches **exactly one
>   candle** (the reversal apply candle). MS sets REVERSAL state at one candle then
>   breaks; the terminal forward-stamp floods `market_state` forward but NOT
>   `structure_id`, so later rows are excluded by the sid filter. The old dual
>   names were vestigial — `compute_structure` now uses the single `.min()` apply
>   idx (WARNs, doesn't crash, if the invariant is ever violated).
> - **`unified_probe` must be imported LOCALLY inside `compute_structure`** —
>   `unified_probe.py` imports `_make_market_structure`/`_pip_size_from_pair` from
>   `structure_engine`, so a top-level import is circular. (Same reason
>   `entity_df_mutation` imports it inside functions.)
> - The reversal probe's `end_idx` = that single reversal apply idx (supreme
>   bound); the reversed structure runs UNBOUNDED to its own next reversal.

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

## Sub charts can show CTS wave candles even though sub KL zones are BOS-only

**Symptom:** an M15 sub chart renders a `CTS.first`/`CTS.last` wave candle (e.g.
the LB at idx=4082 from `M15.counter` sid=0 cyc=1 in 2025-12→2026-01) and you
go looking for the parent CTS KL zone to cross-reference — but the sub's KL
zone list (and KL CSV) only contains `BOS` rows. Confusion: "where did the CTS
wave candle come from if there's no CTS zone?"

**Mechanism:** the orchestrator derives the **full** BOS+CTS zone set
internally (`derive_kl_zones_v1(..., source_kinds=None)`), feeds it to
`compute_wave_candles` so wave candles see both kinds, and **then** narrows
the returned/charted zone list with the caller's `source_kinds` filter. For
sub callers (`pooled_structure_build.project_to_window`, `source_kinds=("BOS",)`;
was `entity_df_mutation.build_one_sid` before Plan C) that filter is `["BOS"]`, so
the chart-visible KL zone list drops CTS — but the wave candles already
computed off those CTS zones survive and render. See
`pipeline/orchestrator.py:86-108` for the in-code comment.

**Practical:** if you need the source CTS zone bounds for a sub wave candle,
either (a) re-derive the unfiltered zone set, or (b) read the CTS
extremes from the structure events (`CTS_ESTABLISHED.idx`/`meta`) — they're
the same source the zones came from.

**Related:** the hover label for these wave candles still hardcodes
"BOS zone:" — there's a TODO at each of the 3 hover sites
(`export_plotly.py`, two in `export_m15_chart.py`) to branch on
`wc.source_kind` for a correct label.

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

**Fix:** In `multitf/entity_df_mutation._build_geometry` (was
`multitf/lower_tf_pipeline.py`, then `build_one_sid`), re-run `compute_imbalance` on the
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
filter (that only governs dots/lines; `owner_by_idx_dir[(candle, direction)]` since Plan C).

**Contrast with main:** a main open zone at end-of-data genuinely *is*
active at the chart's right edge (`end_time=None` is correct there) — main
runs over all data, has no parent boundary. So the cap is **sub-only**.

**Note (2026-05-25 redesign):** under the sub-structure lifecycle redesign,
this cap survives as the "structure ends when parent next cycle/sid starts"
end condition in the unified lifecycle model. See
`memory/project_sub_structure_lifecycle_redesign.md`.

**Note (Plan C, 2026-09-20):** the premise above — "the slice truncates the sub's
ongoing structure" — is gone: sub geometry runs to the DATA EDGE
(`build_or_get_geometry`), so the last zone of an ended sub is not truncated, it is
simply OPEN in a geometry that outlives the sub. The cap is therefore MORE
load-bearing, not less: `render_sub_projection` passes the unique sub's `end_idx` as
`lifecycle_cap` (+ `end_reason` as `cap_reason`) into the derivations, and every zone /
POI / fib inherits that end through `compute_cycle_lifecycle`. `deactivated_by` /
`"lifecycle_end"` no longer exist on any artifact — the end reads `end_idx` /
`end_reason ∈ {reversal, same_dir_replacement, parent_end}`. LANDMINES "Lower-TF Zones,
POIs, and Fibs Must Be Capped at Lifecycle End".

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
`*_kl_zones.csv` + `*_subs.csv` / `*_triggers.csv` (Plan C; `*_sids.csv` is
removed) debug exports in `run_replay.py` are still
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

**Plan C (2026-09-20) — same principle, one sweep per UNIQUE SUB (PART4 §17.10,
the user's stated lean, NOT a settled WVMI design).** `orchestrator._assign_sub_wvmi_per_sub`
runs once over the per-sub projections: window = `[sub.start_idx, sub.end_idx or edge]`
(`result.meta["start_idx"]` / `["m15_end_idx"]` — the sub's REAL-TIME lifecycle; the
old `start_trigger_idx` is split into `trigger_idx` / `probe_finalize_idx` / `start_idx`
and the WVMI window reads the lifecycle value); stream = the union of the confluence
stream and the counter stream over ALL parent cycles (a sub spans cycles), each entry
tagged with its lens and LOH-mapped, RESTRICTED to the sub's `lenses`; the FIRST such
trigger (by M15 idx, then parent idx) inside the window sweeps the sub once
(`compute_parent_driven_sub_wvmi`, `sub_path_id` = the sweeping trigger's lens path);
dedup key `sub_id`; the records are persisted with `persist_facade_wvmi_to_entity_df`
into EVERY lens df the sub is on. Measured on the first Plan C replay: sub `2639/−1`'s
WVMI rows now also appear on the counter lens df and sub `4027/+1`'s also on the
confluence lens df (rows the baseline's per-(sub, lens) sweeps did not produce on those
lenses — the sub's one sweep is persisted to every lens it is on); WVMI record meta carries
`sub_id` (`sub_sid` gone). The deferred WVMI pass may return to per-(sub, lens) sweeps —
do not build on the per-sub choice.

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

## Cross-referencing a sub record to its zone: join on `sub_id`, NOT the internal `structure_id`

**Rule (Plan C, 2026-09-20 — was `sub_sid` before the pool):** to match a
subordinate WVMI record to its KL / POI / fib record (e.g. in a `/compare`
validation), join on **`sub_id` + `cycle_id`** — and read `sub_id` from **both**
artifacts. `sub_id` is the unique sub's global creation index (PART4 §17.9) and
is stamped by the mirror (`entity_df_mutation._sub_attribution`) into every
structural artifact on a lens df: event meta, KL / POI / fib / wave-candle /
WVMI record meta, `prev_bos_lines[*]["meta"]`, `SidRecord.sub_id`, and the
`sub_id` column of the WVMI CSV (the events / zones / POI / fib CSVs carry it
inside the serialised `meta` dict). `parent_sid` / `parent_cycle_id` /
`use_case` in that meta are INFORMATIONAL (the sub's first live record) and
must never be part of the join — a sub spans parent cycles, and both of its
records stamp the same first-record values. Do **NOT** join on the record's
internal `structure_id` (`WVMIRecord.bos_structure_id`,
`KLZone.meta["structure_id"]`): inside a single-structure sub run the internal
`structure_id` is **always 0** (the run stops at the sub's first reversal — a
reversal-born successor is a NEW unique sub with its own `sub_id`), so every
sub collapses to `structure_id == 0` and the join silently mismatches.

**Why two different fields exist:** `sub_id` is the pool's structure identity
(one per unique `(parent_path, sub_tf, direction, starting_idx)`); the
internal `structure_id` is the MarketStructure counter *within one bounded
run* (always 0 for a sub). Both are stamped into zone / WVMI meta; they
coincide only for `sub_id == 0`. `sub_sid` is GONE from every structural
artifact — only a MAIN `SidRecord` keeps `sub_sid = structure_id`; the
per-parent-cycle counter survives as `TriggerRecord.trigger_sub_sid`, which
lives on the record table (`attrs["triggers"]`, `_triggers.csv`) and is not
stamped on zones. A record's `sub_id` is the FK to its sub, so joining a
`_triggers.csv` row to its zones also goes through `sub_id`.

**History (pre-pool, 2026-05-27):** the rule then read "join on
`(parent_sid, parent_cycle_id, sub_sid, cycle_id)`", where `sub_sid` was the
entity-local per-parent-cycle chain counter (0, 1, 2, … resetting each parent
cycle — the merge-and-bound redesign's identity) stamped on every artifact.

**How it bit (2026-05-27, WVMI lifecycle validation):** cross-checking each sub
WVMI `(end_idx, end_reason)` against its KL BOS-zone end, joining on
`structure_id` reported spurious mismatches (a `sub_sid=0` WVMI record matched the
wrong sub's zone). Re-joining on `sub_sid` made all 8 sub records match exactly.
The values were correct all along — WVMI inherits the **same**
`compute_cycle_lifecycle` end as KL/POI/fib (identical events + slice-local floor/
cap), so a real mismatch there would mean a wiring bug, not a data bug.

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

## Sub structure events out-of-idx-order? Suspect entity-absolute pollution of `is_range_confirm_idx`

**Symptom on a sub chart:** BOS and CTS dot markers don't alternate the
way they should — by chart-marker idx, you see `BOS → BOS → CTS → CTS`
where you expect `BOS → CTS → BOS → CTS`. Equivalent in the
`_M15_<entity>_structure_events.csv` dump: the `cycle_id` of confirmed
events goes forward then backward (e.g., `0, 1, 0, 1`) when sorted by
the chart's marker idx (`BOS_CONFIRMED.idx` for BOS, `cts_anchor_idx`
for CTS).

**Root cause:** `is_range_confirm_idx` is computed entity-absolute on
the wide M15 df, then sub slices retain those values after
`reset_index(drop=True)`. The MS engine reads them as slice-local,
polluting back-fill bounds and leaking the CTS extreme past the eventual
pullback apply candle. See LANDMINES "Sub Slices Must Re-Derive
`is_range_*` Labels After `reset_index`" for the full mechanism.

**Quick diagnostic:** dump the sub's structure events CSV and filter
`RANGE_STARTED` events for the `sub_id` (was `sub_sid`). If you see any `idx` value
≥ `len(M15) * 2` (e.g., 8026 when the M15 data has ~4228 rows), the
pollution is present:

```python
import pandas as pd, ast
df = pd.read_csv("artifacts/debug/..._M15_counter_structure_events.csv")
df['meta_d'] = df['meta'].apply(lambda s: ast.literal_eval(s) if isinstance(s,str) else {})
rs = df[df['type'] == 'RANGE_STARTED']
# any idx > ~4500 on a ~4228-candle M15 window is post-double-shift garbage
print(rs[['idx', 'meta']].head())
```

**Fix:** `_build_geometry` (was `build_one_sid`) re-derives the range columns after the slice.
If you add a NEW pre-computed positional-index column to the M15 prep
pipeline, you must also re-derive it on the sub slice for the same reason.

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

---

## Filtering MS's CTS_EST list to enforce a "true first breakout" is wrong

When implementing a "the first CTS_EST only counts if its anchor's extreme is a
new running max/min" rule, the obvious first attempt is to run MS, collect the
CTS_EST events, and walk the list to find the first one passing the rule —
discarding earlier ones. **This is incorrect.**

**Why:** MS's state machine establishes cycle N+1 only after cycle N is
CONFIRMED, and the new BOS / CTS detection uses the *prior CTS's price* as
threshold (`establishing_new_cycle` requires `st.cts_phase == "CONFIRMED"`,
then BOS_n+1 is selected against that). If MS's first CTS_EST is rejected
externally, MS's subsequent CTS_EST events were computed against the
*rejected* first CTS's threshold — they're not the "true" 2nd / 3rd CTSes
for the alternate timeline that should have skipped the rejected one.
Filtering the precomputed list yields a chain that doesn't structurally hold
together.

**Correct approach (true-first-breakout cycle-0 redesign, 2026-06-07). The lesson above still holds; the mechanics below superseded the earlier `df.pat`-walk / partial-gate approaches:**
- **ONE shared routine** `engine_v2/structure/true_first_breakout.py::find_true_first_breakout` encodes the 4 conditions: anchor closes past the BOS_0 inner (mechanism B — re-detect via the detectors with `break_threshold=bos0_inner`, NOT the threshold-free `df.pat`), valid pattern (confirmation allowed), **strict** full-pattern new extreme over `[current_start, extreme_candle)`, earliest apply/confirm idx (tie-break `continuous>dm>omc>omo`).
- The unified probe's **deterministic method** calls this routine (no MS) to decide the start; MS's **pre-CTS_0 scan-from-start mode** (`enforce_cts0_new_extreme=True`, now REQUIRING `bos0_inner`) calls the SAME routine to re-find and establish cycle-0 via its normal path, so probe and MS agree by construction. The old partial anchor-extreme gate (`_cts0_new_extreme_passes`) and the old df.pat Phase-1 walk were REMOVED. Seed-and-resume was rejected in favor of scan-from-start.
- Detail: [[project-true-first-breakout-cycle0]] (memory).

**The deferred main sid=0|cycle=0 fix (Commit 2) uses the same scan-from-start mechanism** — pass `enforce_cts0_new_extreme=True` + `bos0_inner` at the main pipeline's structure call. Not yet wired.

---

## Cycle-0 Fib activation was one-shot at CTS_ESTABLISHED in cross_cycle (subs) — FIXED 2026-06-08

**Symptom:** a sub's cycle-0 Fib (and its POIs) never appears even though cycle 0
has an unfilled sd-imbalance — but only when that imbalance becomes unfilled-in-window
*after* the cycle-0 CTS-established candle.

**Root cause:** in `fib_mode="cross_cycle"` (all M15 subs) the cycle-0 single Fib
was evaluated **once, at `CTS_0_ESTABLISHED`** (`_handle_cross_cycle_cts_established`,
`cycle_id==0`): activate iff `has_unfilled` at that instant, else `NO FIB`. The
cycle-0 `CTS_UPDATED` handler then did `if key not in self._fibs: return None` — it
only *extended* an already-active Fib and **never first-activated** one. So if the
establishment candle had no unfilled imbalance, the cycle-0 Fib could never form,
even as the CTS extended and unfilled imbalances appeared.

**The asymmetry that revealed it:** the H1-main `sid≥1` cycle-0 path
(`_handle_cycle0_cts_updated`, Scenario 1) ALREADY re-checks and first-activates on
`CTS_UPDATED` (lines ~1213-1231). Only the cross_cycle (subs) path was one-shot.

**Fix:** `_handle_cross_cycle_cts_updated` `cycle_id==0` now first-activates when the
Fib isn't yet active — recompute `has_unfilled` over `[BOS_0, cts_idx]` as-of
`cts_idx` (BOS from `_bos_by_cycle[(sid,0)]`) and activate if present
(`meta["activated_on"]="update"`), mirroring the Scenario-1 path. Idempotent (once
active it routes to the update path). H1 main unaffected (its path already did this).
/compare vs `f2f5e35`: H1 main + confluence byte-identical; counter gains exactly
1 cycle-0 Fib + 2 POIs (sub (0,2,0)); kl_zones/sids/structure_events/wvmi unchanged
(a fib-tracker activation is downstream of MS).

**Surfaced by** the true-first-breakout cycle-0 work: moving CTS_0 changed which
imbalances fell in cycle-0's window at the establishment instant, exposing that the
one-shot was silently dropping cycle-0 Fibs whose imbalance arrived later.

---

## `bos_threshold` is reset to the ORIGINAL BOS at CTS confirmation, discarding expansion (FIXED 2026-06-01)

**Symptom:** a structure reverses *earlier* than it should. Two MS runs with
byte-identical event streams for hundreds of bars suddenly diverge at the
reversal: the buggy one reverses against a *stale, un-expanded* BOS threshold.

**Root cause:** `bos_threshold` follows a different lifecycle from
`cts_threshold`, but the code treats them the same at CTS confirmation:

- **BOS locks at BOS_CONFIRMED** (`_emit_bos_confirmed`, ~line 1813 sets
  `st.bos_threshold = price`). From then on it legitimately *expands* via the
  barrier wick-cross probes (`_bos_barrier_step`, ~lines 624/640/658/674,
  emitting `BOS_THRESHOLD_UPDATED`). The reversal/breakout check reads this
  stored `st.bos_threshold`.
- **CTS only locks at CTS_CONFIRMED** — so re-initializing `cts_threshold` at
  confirmation is correct.
- **The bug:** both CTS-confirmation paths *also* re-init `bos_threshold` to
  `bos_confirmed.price` (the ORIGINAL extreme) — pullback path ~line 1500,
  proximity path ~line 2013. This **discards any expansion that happened in
  the window `[BOS_CONFIRMED, CTS_CONFIRMED]`**. It's a copy of the (correct)
  `cts_threshold` init wrongly applied to a threshold with a different
  lifecycle.

**Contrast — CTS breakout (establishing a new cycle) does this correctly:** it
uses `_range_breakout_threshold()` = live `range_hi`/`range_lo`, which always
tracks the expanding zone. Only the BOS-reversal path reads a stored value
that CTS-confirmation clobbers. (Verify with main zone base 430: its CTS
breakout fires at the *expanded* threshold 636, not 635.)

**Why it surfaces on pullback far more than proximity:** the bug is symmetric
(both confirmation paths reset), but pullback confirmation usually lands much
LATER than proximity (proximity fires as soon as price nears the zone;
pullback waits for a full pattern), so the `[BOS_CONFIRMED, CTS_CONFIRMED]`
window is longer → more chance a BOS expansion falls inside it → more chance
of clobber. A BOS confirmed directly at its expanded extreme (no pre-confirm
expansion) makes the reset a no-op, which is why it stayed latent so long.

**Diagnosis tip:** the frozen value is in the `REVERSAL_CANDIDATE` event meta
as `bos_frozen`. Compare it against the `BOS_THRESHOLD_UPDATED` price the zone
expanded to — if `bos_frozen` < the expanded value (sd=+1) it's the stale
original. Concrete instance (Session 3, 2026-06-01): confluence (1,2) sub
BOS confirmed 0.57806 → expanded 0.57827 @3806 → CTS_CONFIRMED@4157 reset to
0.57806 → reversed at 4179 (against 0.57806) instead of 4200 (against 0.57827).

**Two sibling inconsistencies found in the same review** (pullback vs proximity
CTS-confirmation paths should behave identically post-confirm):
- `cts_threshold` is synced inline at confirmation on the pullback path
  (`_sync_thresholds_from_range`, ~line 1504) but NOT the proximity path
  (relies on the next candle's `_expand_range` → 1-candle lag).
- Proximity confirmation *creates a range* but never calls `_set_state`, so it
  leaves `range_active=True` with state still BREAKOUT — the only
  range-creating path that doesn't set a coherent state
  (`_finalize_range_candidate_offline` sets RANGE ~line 1116; the pullback
  path sets PULLBACK ~line 1503). NOTE: `st.state` gates pattern dispatch
  (~line 992 — pullback patterns suppressed in NONE/PULLBACK/PULLBACK_RANGE,
  allowed in BREAKOUT/RANGE), so changing this can shift downstream pattern
  eligibility — the highest-risk of the three fixes.

**Fix (DONE 2026-06-01, all 3 in one commit):** deleted the `bos_threshold`
reset at both CTS-confirmation paths (kept `cts_threshold`); added the inline
`_sync_thresholds_from_range(candle_idx)` to the proximity path; added
`_set_state(MarketState.RANGE, candle_idx, ...)` inside the proximity path's
`if not st.range_active:` range-creation block (gated so it only fires when
proximity CREATES the range; prior state is provably BREAKOUT there). Verified
blast radius: the only behavioral state-gate (pattern dispatch ~line 992)
buckets BREAKOUT and RANGE identically, so fix #3 shifts no pattern
eligibility.

**/compare vs `8865c10`:** H1.main 8/8 + M15.counter 5/5 byte-identical;
deltas confined to M15.confluence. The documented (1,2,1) `subsequent_confluence`
reversal corrected **4179→4200** — `bos_frozen` went from the stale `0.57806`
to the fully-expanded `0.57898` (barrier probes had pushed it even past the
`0.57827` first spotted). Same 13 sids, no drops. Fixes #2/#3 produced zero
independent byte-changes on this window (proven by H1.main + counter
byte-identical — proximity never hit the pre-expanded-range or
create-range-from-BREAKOUT case there); their correctness rests on the static
blast-radius analysis. 309 tests green. See
`project_unified_identify_start_probe.md` (MS fix = Step 3 lead item).

---

## A Lifecycle Floor That Lives in a Build Function Is Lost by Any Path That Bypasses It (3.2b, 2026-09-10)

**Symptom:** Stage 3.2b (`9fd3143`) added `render_unique_sub` to render each
pooled sub once per lens. Chart review showed five sub structures that were
inert outlines in the baseline (`confirmed_idx 3611, status inactive`) suddenly
fully live, zones filled hundreds of candles early, and a counter sub cut off
at 2844 "by a structure not on that chart". 418 tests green throughout.

**Cause:** the parent-cycle lifecycle floor (plan B1) was computed as a
**local variable inside `build_one_sid`**
(`_floor_abs = max(start_trigger_idx, parent_floor_m15)`), never stored on
the `TriggerRecord` or the `PooledStructure`. `render_unique_sub` bypassed
`build_one_sid`, read the record's raw `trigger_dt`, and the floor was
silently gone. The cross-chain cut was a *consequence*: same-direction
replacement compared other subs' equally-unfloored starts. Root cause was the
SPEC — §17.6 defined `start = min(trigger_dt)` and stopped; the code
implemented it faithfully ("incorrect docs, faithfully implemented" — the
mirror image of the usual failure).

**Lesson:** a lifecycle value that clamps *when an object is active* is a
property of that object. Put it ON the record/sub (`start_idx` = already
floored) at creation; never re-derive it per consumer. And when a spec rule
is a *refinement* of an existing rule (§17.6 refining §5), the spec must
restate the parent rule, not assume it. Diagnosed by a keyed diff of the KL
zone CSVs (`(anchor_idx, base_idx, cycle_id, struct_direction)` — immune to
the `sub_id` renumber): every parent-sid-1 zone flipped `inactive→ended`.
Canonical fix = the TriggerRecord model, `memory/project_sub_structure_pool_architecture.md`.

**Fixed by Plan C (2026-09-20).** The floor is a FIELD of `TriggerRecord`:
`start_idx = max(probe_finalize_idx, trigger_idx, parent_floor_idx)`, set ONCE
at record creation by `lifecycle_sweep._Sweep._resolve_and_record` (step 5),
with `parent_floor_idx = ParentTables.floor(S, C)` from
`multitf/parent_tables.py` kept alongside as the diagnostic copy. Every
consumer — the sub-end aggregation (phase 4), the sibling read
(`start_idx <= hi`), the projection's `lifecycle_floor`
(`render_sub_projection`: `sub.start_idx - slice_begin`, where `sub.start_idx`
is the first live record's `start_idx`), the chart's `owner_by_idx_dir`
window, the WVMI window, `SidRecord.start_idx`, `_triggers.csv` — reads the
stored value; nothing re-derives it. `build_one_sid`, `_floor_abs`,
`render_unique_sub` and `TriggerRecord.trigger_dt` are gone. Measured on the
first Plan C replay: the four (1,0)/(1,1) triggers — the source of every
"suddenly live" zone in that diff — are unresolved (`degenerate_parent_cycle`,
LANDMINES "Degenerate Parent Cycles"), and every record's `start_idx` equals
the predicted table. The spec side of the lesson was applied too: PART4 §17.4
states the floor on the record and restates the §5 clamp it refines.

---

## `ProbeResult.finalize_idx` Is Native-M15 OR a Mapped H1 Value Depending on `finalize_condition` — Don't Assume Either (2026-09-10)

`finalize_idx` is a single int but its provenance differs by exit:

| `finalize_condition` | value | frame |
|---|---|---|
| `second_cts_reached` | 2nd `CTS_ESTABLISHED` moment (`meta["confirmed_at"]`) from the probe's own MS run — the candle the early stop keys on (Plan B, 2026-09-20; was `.idx`, the extreme — equal on this window: 1020/1020, 2608/2608) | native sub-TF |
| `reversal_in_probe` | reversal apply idx | native sub-TF |
| `no_retrace` (cycle 0 confirmed in-window) | `CTS_0_CONFIRMED.idx` | native sub-TF |
| `no_retrace` (else) / `end_idx_reached` / Phase-1 | `probe_end_idx` (the probe's search bound — was spelled `end_idx` before Plan C renamed it, 2026-09-20; the `end_idx_reached` condition NAME is unchanged) | **mapped** from the parent (price-mapped `cts_anchor_idx` for `first_confluence`; last-of-hour for the sibling-referencing variations; the reversal candle for the reversal handoff — native) |

**Its role under §17 (Plan C):** `ProbeResult.finalize_idx` becomes the
record's **`probe_finalize_idx`** — a HISTORICAL field, ONE input to
`TriggerRecord.start_idx = max(probe_finalize_idx, trigger_idx,
parent_floor_idx)`, never a lifecycle value by itself (it may precede the
trigger: FC(0,1) 2608 < 2611; it may exceed it: FC(0,0) 1020 > 463). On a
probe-cache hit it is INHERITED raw from the first probe of that key — the
record's own probe never ran, so its `probe_finalize_condition` is the cached
one too (LANDMINES "Probe Cache Keys Are Shared by Reversal Handoffs": three
such records on the reference window; e.g. FC(0,1) now carries 2470
`no_retrace` from the sub-1 reversal handoff instead of its own 2608
`second_cts_reached`, and `start_idx` is 2611 either way). Read
`_triggers.csv` with that in mind: a `probe_finalize_idx` far below
`trigger_idx` with a Phase-1-looking condition on an FC record is a hit, not a
mapper bug. The probe's OUTPUT anchor is `ProbeResult.starting_idx` (was
`start_idx`) — the pool key.

Live run on the 2025-11→2026-01 window (`debug/probe_fc_finalize.py`):
FC(0,0) 1020 and FC(0,1) 2608 are native 2nd-CTS values (2608 lands 3 candles
*before* its own trigger at 2611 — the parent floor repairs it); FC(1,1) 3047
and FC(1,2) 3621 are the price-mapped parent anchor; FC(1,0) 2844 was a native
`CTS_0_CONFIRMED` that only existed because of the MS bounds leak (LANDMINES
"Bounded MS Runs Must Not Read Past `end_idx`") — **fixed by Plan A
(2026-09-19): FC(1,0) is now 2843, the else-branch `end_idx`, so all three
`no_retrace` FCs are the price-mapped bound.** So: (a) the `else end_idx`
branch is the COMMON case, not an edge; (b) do not "fix the mapping of
finalize_idx" — only the `end_idx`-derived branch is mapped, and changing that
mapping also moves the probe's search bound and therefore `starting_idx` (the
pool key); (c) historically, when a value looked one candle past a mapped
bound the cause was the bounds leak, not the mapper — since Plan A the MS
post-run assert makes that impossible, so a value past a mapped bound now
points at the mapper (or at L5b, the probe's ad-hoc BOS_0 read).

**Two different "anchor"s** (GLOSSARY): the parent's `cts_anchor_idx` (H1,
`CTS_CONFIRMED.meta`, the probe bound) vs the probe's own M15 `cts0_anchor`
(Phase-2 MS `CTS_0_CONFIRMED`). Conflating them cost a full discussion round.

---

## `REVERSAL_CANDIDATE.meta["pattern"]` Is Always `"?"` (found 2026-09-19, not yet fixed)

`_schedule_reversal_from_anchor` (and `_maybe_apply_pending_reversal`'s debug line) reads the
pattern name as `getattr(ev_r, "pat", None) or getattr(ev_r, "pattern", None) or "?"`, but
`PatternEvent` carries it as `.name` — so every `REVERSAL_CANDIDATE` event (and the `[RV_SCHEDULE]` /
`[RV_APPLY]` debug lines) report `pattern='?'`. The `STATE_CHANGED → reversal` event's `meta["pat"]`
is correct (it uses `ev.name`), so use that when you need the reversing pattern. Found by the Plan A
L4 fixture verifier; a one-line fix (`ev_r.name`) for a later commit — it changes event meta, so it
gets its own `/compare`.

---

## `ref=cts_confirmed` in a Replay Log Does NOT Mean the Derived CTS Zone Was Used (2026-09-19)

`build_reference_zone_from_cts_event` has two branches: the CONFIRMED branch (find the existing CTS
KL zone for `(structure_id, cycle_id)`) and the ad-hoc fallback (`_derive_cts_zone_ad_hoc`). **Both**
label the returned `ReferenceZone.source` by the winning EVENT's type — the derived branch hard-codes
`"cts_confirmed"` (`reference_zone.py:373`) and the ad-hoc branch maps `CTS_CONFIRMED → "cts_confirmed"`,
`CTS_UPDATED → "cts_updated"`, else `"cts_established"` (`:383-388`). So the `ref=…` token in the probe
log lines cannot tell you which branch ran. A review this session inferred "8 probes use the derived
zone" from that token; the truth is the opposite:

**For subs the CONFIRMED branch is dead.** `_find_existing_cts_kl_zone` requires `source_kind == "CTS"`,
but `_run_downstream_pipeline` derives the full KL set internally and returns `source_kinds=["BOS"]` for
subs — and that BOS-only list is what both readers get (the reversal probe reads `downstream["kl_zones"]`,
the sibling-CTS read reads the mirrored `attrs["kl_zones"]`; the sub KL CSVs contain zero CTS rows). Every
sub reference zone is ad-hoc-derived today. Consequence for Plan C: passing `kl_zones=[]` to the
primitive from the pool path is exactly behaviour-preserving. If you want to know which branch ran, log
inside the primitive, not the `source` field. Follow-up (unscheduled): whether subs *should* get the
derived CTS zone.

**Landed that way (Plan C, 2026-09-20).** Both pool-path readers pass `kl_zones=[]` to
`build_reference_zone_from_cts_event` for exactly this reason: the sibling read
`_build_sibling_cts_ref_zone_from_pool` (`kl_zones=[]`, `df=` the shared entity-absolute M15 frame,
`sid=0`, `idx_window=(lo, hi)`) and the reversal handoff `_resolve_reversal_start` (`bounded.events, [],
bounded.df, sid=0, idx_window=None`). Neither reader has a KL zone list to hand over any more — pool
geometry is cap-free MS output with no KL derivation (`build_or_get_geometry`), and the per-trigger
mirror-then-read of `attrs["kl_zones"]` is gone — so `[]` is not only behaviour-preserving, it is the
only honest input. The `ref=cts_confirmed` / `ref=cts_updated` / `ref=cts_established` token in the
`[entity_compute] unified_probe (...)` lines is still the winning EVENT's type. Reopening the derived-zone
question now means deriving CTS KL zones from the pool geometry on demand — its own `/compare`.

---

## A Cycle Cannot Be Established Inside an Open Reversal Watch — MS Invariant 4 (2026-09-20)

Found while crafting Plan B's §4.1 "quiescence" fixture (a reversal watch open at the moment the
2nd `CTS_ESTABLISHED` fires). Establishing a cycle writes the new BOS (`_emit_bos_confirmed`,
`BOS_1` = the pullback extreme, `_select_bos_on_breakout`) and therefore moves `bos_threshold`; the
close-break candle that opened the watch lies inside that pullback window with `l <` the frozen
BOS, so **any breakout that establishes a cycle while a watch is active moves `bos_threshold`
during the watch** and `_check_invariants_df` raises `[INV] bos_threshold changed during reversal
watch` post-run (default `debug_invariants=True`; every real caller). Two search-built fixtures
reached the state and both raised; they only "worked" with `debug_invariants=False`. The reachable
shape is the reverse order **inside one `_step_anchor` call**: the breakout applies (BOS written),
then `_post_apply_range_check(apply_idx)` back-fills the apply candle's range window and a
close-break in that back-fill opens the watch — `tests/test_ms_stop_after_cts.py::_make_watch_over_second_cts_data`
(2nd CTS idx 9 / moment 10, watch at 11 inside the same step, reversal at 14). Whether invariant 4
*should* forbid the first order is an open MS question, not a Plan B one — record, don't change.

Also reproduced there: the Plan B §2 "rebuilt-prefix" exception (`_make_double_rewind_data`) — two
expiry-rewinds, one before the 2nd CTS and one after the early stop; `_rewind_to` replays from 0
ignoring the earlier jump (LANDMINES "MarketStructure Deep-Couples…" 1), so the exit classifier
reads `cts_est=[2, 8, 12]` while the early stop read the post-J1 `[2, 12]`. No instance on the
reference window (zero rewinds in any FC Phase-2 run); pinned by a strict `xfail`.

---

## Phase-1 Incumbents Must Be STARTED Records, Not Interval-Active Ones (Plan C, 2026-09-20)

**Found by:** the same-idx collision case in `tests/test_lifecycle_sweep_unit.py` while
implementing the sweep (`multitf/lifecycle_sweep.py`). Plan C §4.3′ wrote phase 1 as `inc =
pool.active_record(lens, S, C, direction, at_idx=t)` (the interval rule on `trigger_end_idx`:
`not is_zero_length and start_idx <= t and (trigger_end_idx is None or trigger_end_idx > t)`) and
"queue a `same_dir_replacement` end for the incumbent at `t`".

**The trap:** the interval rule is a property of the RECORD, not of the sweep's progress. When two
records of the same `(lens, parent_sid, parent_cycle_id, direction)` for DIFFERENT subs both start at
`t` (the acausal collision §17.6 warns about), the earlier-`seq` record's `RECORD_START` runs first; at
that moment `pool.active_record` also returns the later-`seq` record — its `start_idx <= t` and it has
no end yet — because its own `RECORD_START` is merely still queued. The earlier record would then
freeze the LATER one as its "incumbent", inverting the collision rule ("the later `seq` replaces the
earlier"), and the sub-level `start_idx` / `end_idx` of both subs would come out swapped.

**Fix (as landed, `_phase1_record_start`):** the incumbent candidates are filtered to records whose
start moment has RUN — `r.seq in self._active and r.is_active_at(t)` — so only a started record can be
replaced. And when the collision is real (`inc.start_idx == rec.start_idx`), the incumbent is frozen
IMMEDIATELY in phase 1 (`trigger_end_idx = t`, `end_reason = "same_dir_replacement"`, `ended_by_sub_id`,
`end_idx`, removed from `_active`; `WARNING [sweep] same-idx start collision` logged) rather than queued
to phase 3 — it is zero-length by construction (`end == start`), and a zero-length record must
participate in nothing, so deferring to phase 3 would let phase 2 (`SUB_START`) read it as live for one
moment. The ordinary replacement (`inc.start_idx < t`) is still queued to phase 3 as a candidate.
`pool.active_record` keeps the pure interval rule (it is the right test everywhere else — the sibling
read, the spawn rule's `is_live_at_reversal`, the post-sweep asserts).

**General form:** in a moment-ordered sweep, "active" has two readings — the record's interval says
where it WILL be active; the sweep's `_active` set says which starts have been APPLIED. Any phase that
mutates another record based on "who is active now" must use the applied set, or it will act on a
record that does not exist yet at that phase (LANDMINES "The Sweep Phase Order Is Load-Bearing",
rules 4 and 6). Not observable on the reference window (no same-idx collision there); the unit test is
the only guard.

---

## A Predicted +1 Shift Can Be Masked by an Equal Floor (Plan C, 2026-09-20)

**Prediction:** the moment-not-extreme rule (`compute_cycle_lifecycle` start =
`max(CTS_ESTABLISHED.meta["confirmed_at"], struct_start, floor)` instead of `.idx`) was expected to
move exactly three M15 sub-cycle lifecycle starts by +1 candle — the three saved M15 `CTS_ESTABLISHED`
events whose extreme precedes the apply candle: 1223→1224 and 2828→2829 ×2.

**Measured on the first Plan C replay:** ONE visible shift — sub `454/+1`'s cycle-1 END
1223→1224 (the pass-through end follows the next cycle's clamped start; that next cycle's
`CTS_ESTABLISHED` has extreme 1223 and moment 1224). The two 2828→2829 cycle starts belong to sub
`2639/−1`, whose lifecycle floor is 2829 on BOTH lenses under Plan C — the unique sub's `start_idx`,
which is its reversal-born confluence record's `start_idx = R = 2829` (its counter record starts
later, at 2843, but the floor the projection passes is the SUB's start, and there is one projection
for both lenses). The zone layer clamps every cycle start at that `lifecycle_floor`, so
`max(extreme 2828, floor 2829) = 2829` and `max(moment 2829, floor 2829) = 2829` — identical. A
clamp EQUAL to the new value hides the shift; nothing was wrong on either side.

**Rule for the next prediction of this kind:** count a lifecycle shift as observable only where the
floor that will be applied is STRICTLY below the new value. For a sub cycle that floor is the unique
sub's `start_idx` (`render_sub_projection` → `lifecycle_floor`); for a main cycle it is the reversal
handoff `struct_start`. Write the prediction as "N candidates, M observable" and list the masked ones
with the floor that masks them, so a `/compare` showing fewer shifts than candidates is not mistaken
for a lost change — and one showing MORE is still a regression.
