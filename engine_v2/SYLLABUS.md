# **🧠 Final 10-Week Syllabus (Course-Style)**

---

## **Week 1 — Stabilize & Wrap Existing Code (Foundation Week)**

### **Goal**

Everything imports, runs on **one pair / one TF**, and exports a **chart payload**.

### **Work**

* Create new package (don’t delete old code):  
  Forex\_Algo/engine\_v2/  
    data/  
    features/  
    patterns/  
    structure/  
    zones/  
    indicators/  
    execution/  
    backtest/  
    charting/

* Wrap (do not rewrite yet):

  * `candles.py` → `CandleClassifier`

  * `candle_patterns.py` → `PatternEngine`

  * `market_structure.py` → `StructureEngine`

  * `indicators.py` → `IndicatorEngine`

  * `zones.py` → `ZoneEngine` (KL \+ OB/POI stubs)

* Fix **only hard runtime/syntax bugs** (undefined vars, `==` vs `=`, pandas row mutation issues).

### **Charting (start immediately)**

* Add `export_payload()`:

  * candles

  * candle\_type

  * pattern markers

  * BOS/CTS (even if rough)

* Output `artifacts/EUR_USD_M15.json`

### **Tools**

* VS Code

* Cursor Pro (refactors, find usages)

* pytest (1 smoke test minimum)

### **Deliverable**

* `run_replay.py` runs end-to-end

* JSON payload generated successfully

---

## **Week 2 — Simple Chart UI v0 (Static Viewer)**

### **Goal**

You can **see** candles \+ pattern markers to sanity-check logic.

### **Work**

* Build `chart_ui/`:

  * static HTML \+ JS

  * TradingView Lightweight Charts

* Load JSON payload from disk

* Render:

  * candles

  * pattern markers

* Add checkboxes to toggle marker types

### **Tools**

* Lightweight Charts

* Browser devtools

* `python -m http.server`

### **Deliverable**

* Open `index.html`

* See candles \+ multiple pattern markers plotted correctly

---

## **Week 3 — Lock Candle Classification (Tests \+ Visual Overlay)**

### **Goal**

Candle typing becomes **stable and trusted**.

### **Work**

* Standardize OHLC columns (`o/h/l/c`) everywhere

* Refactor `candles.py` into:

  * `compute_candle_metrics()`

  * `classify_candles()`

* Write **20–30 pytest tests** for:

  * maru thresholds

  * pinbar logic

  * doji / edge cases

### **Charting**

* Add candle-type overlay (small label or color)

* Toggle on/off

### **Tools**

* pytest

* Cursor Pro (test generation, refactors)

### **Deliverable**

* All candle tests pass

* Visual candle types match expectations

---

## **Week 4 — Pattern Engine Cleanup \+ Debuggable Markers**

### **Goal**

Multi-candle patterns are **reliable and explainable**.

### **Work**

* Fix pandas logic errors (`.isin`, shifts, boolean ops)

* Normalize outputs:

  * `PatternEvent(name, direction, time, tf, meta)`

* Separate:

  * breakout/continuation patterns

  * entry patterns

* Ensure no unintended lookahead

### **Charting**

* Click a marker → show debug panel:

  * pattern name

  * thresholds used

  * confirmation levels

### **Tools**

* pytest

* Cursor Pro (cross-file fixes)

### **Deliverable**

* You can click any pattern and see *why* it fired

---

## **Week 5 — Market Structure v1 (BOS / CTS Lifecycle)**

### **Goal**

Structure logic is **deterministic, replayable, and visible**.

### **Work**

* Rewrite `StructureEngine` orchestration:

  * candle-by-candle state machine

  * breakout / pullback labeling

  * BOS/CTS creation \+ invalidation

* Assign stable IDs to BOS/CTS

### **Charting**

* Draw BOS/CTS as horizontal lines

* Show breakout/pullback annotations

### **Tools**

* pytest (replay parity test)

* Cursor Pro (refactor orchestration)

### **Deliverable**

* BOS/CTS lines visually match expected structure

---

## **Week 6 — Indicators Engine \+ Indicator Overlays**

### **Goal**

Indicators are **usable features**, not broken helpers.

### **Work**

* Fix `indicators.py`:

  * pure functions

  * consistent signatures

  * always return df

* Focus v1 indicators:

  * imbalance / liquidity

  * volume spike (optional)

* Add indicator columns for zone & entry logic

### **Charting**

* Toggle liquidity / imbalance markers

* Confirm clustering matches intuition

### **Tools**

* pytest

* Cursor Pro

### **Deliverable**

* Indicators visible and trustworthy

---

## **Week 7 — Zones v1 (KL vs OB/POI) \+ Strength Scoring**

### **Goal**

Zones are **first-class objects** with lifecycle \+ strength.

### **Work**

* Split ZoneEngine:

  * **KL zones** → tied to BOS/CTS

  * **OB/POI zones** → base patterns \+ indicators

* Define:

  * bounds

  * active/mitigated/broken

  * `strength_score`

  * `strength_flags`

### **Charting**

* Draw zones as rectangles

* Click zone → show strength \+ flags

### **Tools**

* pytest

* Cursor Pro

### **Deliverable**

* Strong zones *look* strong visually

---

## **Week 8 — Multi-Timeframe Confluence \+ HTF Overlays**

### **Goal**

You can debug **confluence vs non-confluence** trades visually.

### **Work**

* Run pipeline on **3 TFs**

* Build `ContextSnapshot`:

  * HTF trend

  * active HTF zones

  * LTF candidates

* Persist artifacts per TF

### **Charting**

* Overlay HTF BOS/CTS & zones on LTF chart

* Dropdowns: pair / TF

* Checkbox: show HTF context

### **Tools**

* Cursor Pro

* simple artifact store

### **Deliverable**

* LTF chart with HTF context overlays

> **As built (user decisions 2026-09-30).** The dropdowns, the "show HTF context" checkbox and the `ContextSnapshot`
> were DROPPED as superseded by the Part 4 entity design: one static chart per entity / lens (`H1.main`,
> `H1.main >> M15.confluence`, `H1.main >> M15.counter`) with the HTF overlay always on replaces the dropdowns and the
> checkbox, and the entity registry holds every snapshot field (HTF trend = the H1.main structure; active HTF zones =
> the H1 zone attrs; LTF candidates = the `TriggerRecord`s / subs). "Run pipeline on 3 TFs": H1 + M15 today; the M5
> lens (`H1.main >> M5.counter`) is scheduled after the Week 8 merge, before Entries (`IDEA_PARKING_LOT.md` §D).
>
> **Week 8 DoD signed off by the user 2026-09-30** ("HTF and LTF context aligns logically; confluence vs
> non-confluence is visually obvious"), after Part 4 closed the same day; `week8-volmom-multitf` merged to `main`
> after that day's `/commit-save`.

---

## **Week 9 — Entries & Decisions**

> **Restructured by the user 2026-09-30** (was "Entry Planner v1 + Visual Trade Intents"), in three parts — a rough
> outline; the user details each part as Week 9 reaches it:
> - **Part 1 — Entry prerequisite items.** The list and its order live in ONE place: `IDEA_PARKING_LOT.md` §D. Order
>   set by the user 2026-10-01, the heavy items first: (1) one POI per cycle, (2) single-cycle fib POIs on subs +
>   the overall sub cross-fib logic (2026-10-06: the cross-cycle fib revamp — one logic for main and sub — NEXT,
>   item 1's chart debugging paused until after it), (3) zone size adjustments, (4) pinbar body-pip floor calibration; the rest (entry
>   confirmation candle patterns, the M5 lens, zone strength scoring, ...) is decided after those four.
>   **Updated by the user 2026-10-27** (after the chart updates, SESSION ORDER item 5, were complete): (3) zone size
>   adjustments MOVED to Parts 2 / 3 (below) and the entry confirmation candle patterns MOVED to Part 2; the pinbar
>   floor (4) is the last major Part 1 item, then the remaining minor pre-Part-2 items close Part 1 (register §D
>   SESSION ORDER items 7–8, §D.3). **2026-10-28:** item 7 (the pinbar floor, PLAN_U) DONE; the user scoped item 8
>   (the §A hygiene trio, the replacement-chain `ended_by` fix, the agreed direction for wave candles + WVMI) and
>   moved the candle-direction / rejection confirmation at a zone touch, zone strength scoring and the M5 layer to
>   Parts 2 / 3 (register §D SESSION ORDER item 8, §D.3). Item 8 DONE 2026-10-28 (PLAN_V 8a / 8b / 8c) — **Part 1
>   COMPLETE 2026-10-28**.
> - **Part 2 — Entry decision build & logic** — the original Week 9 plan below, + the entry confirmation candle
>   patterns (moved here from Part 1 by the user 2026-10-27) and the zone size at entry and stop-loss (below); + since
>   2026-10-28 the candle-direction / rejection confirmation at a zone touch (an entry condition, next to the
>   candle patterns), zone strength scoring (with the zone size) and the M5 layer when an entry needs M5 (register
>   §D.3).
> - **Part 3 — Other decisions build & logic** — to be defined by the user.

### **Part 2 / 3 — Zone size at entry and stop-loss (moved from Part 1 by the user 2026-10-27)**

The zones keep their base form; the size adjustment is applied when the entry price and the stop loss are set, at
the moment price comes to the zone (register §D.3). The user, verbatim (2026-10-27):

> "Now originally I wanted to work on zone adjustments as part of Week 9 Part 1, however I've since changed my mind.
> We will incorporate this directly into entry & decisions (Part 2 & 3) rather than as a separate item that adjusts
> existing zones."
>
> "Originally, the idea was to make zone adjustments when the zone sizes are too small or too big. As previously
> mentioned, zones will often be reference points for not only entry decisions, but also decisions around where to set
> our stop losses and entry price. Entry price will likely be near zone inner while stop loss around zone outer. Thus,
> the size of the zones are quite important to optimize our entry & stop loss decisions. If the zone is too big, we
> could have a large stop loss distance and take on higher risk vs. a small zone could lead to a small stop loss
> distance that increases the likelihood of the stop loss getting triggered, resulting in a losing trade."
>
> "Thus the goal of zone adjustments was to help optimize for accuracy and risk when we make decisions around setting
> entry parameters entry price and stop loss based on our zones. I've come to the realization that instead of making
> adjustments to or editing the zones when they are too small or large, we simply implement the adjustment logic
> directly to entry & stop loss when we need to make those decisions when price comes to those zones. This preserves
> the zones in their original form whereas changing the zones themselves would have additional consequences such as
> altering proximity triggers."

Why it matters for the build: a zone's inner / outer edges are read by the proximity triggers (and through them the
subs) — resizing the zones would move those; adjusting only the entry / stop derived from a zone leaves every
structure output as it is. Inputs on file: the measured zone sizes (memory `project_week9_part1_inputs.md`), the
"Stop/TP builder" and "Enforce RR ≥ 3" items below.

### **Part 2 — the original plan (Entry Planner v1 + Visual Trade Intents)**

### **Goal**

See **what the bot would do** before execution.

### **Work**

* EntryPlanner:

  * set-and-forget (strong zones)

  * wait-for-trigger (weak zones)

* Stop/TP builder

* Enforce **RR ≥ 3**

* Position sizing rules

### **Charting**

* Plot:

  * entry

  * stop

  * TP(s)

* Click intent → RR \+ reasons

### **Tools**

* pytest

* Cursor Pro

### **Deliverable**

* Trade intents visually match your discretionary logic

---

## **Week 10 — Backtesting, Optimization, Live-Ready Hardening**

### **Goal**

Close the loop: **simulate → optimize → prepare for live**.

### **Work**

* Event-driven backtester

* SimExecutionAdapter (spread \+ fills)

* Metrics \+ trade logs

* Optuna \+ walk-forward

* Live Oanda adapter (dry-run)

* Risk controls:

  * max spread

  * daily loss cap

  * kill switch

  * restart recovery

### **Charting**

* Plot executed trades

* Filter winners/losers

* Click trade → full reasoning trail

### **Tools**

* Optuna

* Docker (EC2)

* Cursor Pro

### **Deliverable**

* Backtest \+ visual debug

* EC2 dry-run bot running safely

---

# **⏱ Realistic Outcome at End of This Syllabus**

By the end of this plan, you will have:

✅ A clean, modular trading engine  
 ✅ Deterministic multi-TF market structure  
 ✅ Zones \+ strength logic you can **see and debug**  
 ✅ Entry & management rules enforced by code  
 ✅ Event-driven backtesting \+ optimization  
 ✅ A simple but powerful chart UI for sanity checks  
 ✅ A live-ready architecture (with safety rails)

# **🔧 Global Tooling & Workflow (applies every week)**

## **Core tools**

* **VS Code** – editor, Git, terminal, debugging

* **Cursor Pro** – semantic navigation, refactors, test generation

* **ChatGPT** – architecture, logic validation, full-file generation

* **Python 3.10+**

* **pytest** – tests

* **ruff \+ black** – lint/format

* **SQLite** – persistence (zones, trades, events)

* **Optuna** – optimization

* **TradingView Lightweight Charts** – chart UI

* **Docker** (Week 9–10) – deployment on EC2

## **Daily workflow (important)**

1. Create a small branch: `weekX-dayY-topic`

2. Ask ChatGPT for **one focused change** (e.g., “refactor PatternEngine”)

3. Apply in **Cursor Pro** (let it fix imports/usages)

4. Run:

   * `pytest`

   * `python run_replay.py`

5. Open chart UI → sanity check visually

6. Commit \+ merge when green

This keeps progress linear and prevents refactor explosions.