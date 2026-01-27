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

---

## **Week 9 — Entry Planner v1 \+ Visual Trade Intents**

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