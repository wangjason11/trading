# **🛡 Project Guardrails & Principles**

*(Multi-Timeframe Forex Trading System)*

This document defines the **non-negotiable principles** guiding the design, implementation, validation, and evolution of the trading system.  
 It exists to prevent scope creep, overfitting, fragile logic, and loss of interpretability.

This document is **authoritative** alongside:

* the 10-week syllabus

* the recommended chat separation strategy

---

## **1️⃣ Research Engine First, Trading Bot Second**

The system is built as a **market-structure research engine** before it is treated as an automated trader.

**Implications**

* Every decision must be inspectable, replayable, and explainable.

* No trade should exist that cannot be reconstructed visually and logically.

* Live trading is a *by-product* of a correct research engine, not the goal itself.

**Principle**

*The engine must always be able to explain itself before it is allowed to trade.*

---

## **2️⃣ Interfaces Are Frozen Before Behavior Is Optimized**

Core objects (candles, patterns, structure, zones, trades) must have **stable interfaces** even while their internal logic evolves.

**Examples**

* `Zone` objects always expose: bounds, timeframe, strength, flags, status

* `PatternEvent` objects always expose: name, direction, time, metadata

* `TradeIntent` objects always expose: entry, stop, TP, RR, provenance

**Principle**

*We freeze contracts early so logic can evolve safely.*

---

## **3️⃣ Event-Driven by Design**

All logic is structured around **events and state transitions**, not bulk dataframe transformations.

**Key events**

* Candle closed

* Pattern formed

* BOS / CTS created or invalidated

* Zone created / mitigated / broken

* Trade intent created

* Trade state transitioned

**Benefits**

* Deterministic backtesting

* Accurate replay

* Identical live vs historical behavior

* Clear attribution of decisions

**Principle**

*If a decision can’t be expressed as an event, it isn’t ready.*

---

## **4️⃣ Visibility Before Performance**

Correctness and interpretability always take precedence over speed.

**Rules**

* Slow but explainable logic is acceptable

* Fast but opaque logic is not

* Vectorization and micro-optimizations are postponed until logic is trusted

**Principle**

*Readable, inspectable logic beats clever code.*

---

## **5️⃣ Charting Is a First-Class Test Framework**

Charts are not cosmetic — they are **primary validation tools**.

Charts must be able to show:

* candle types

* pattern markers

* BOS / CTS levels

* zone rectangles

* entry / stop / TP levels

* executed trades

**Principle**

*If it can’t be verified visually, it isn’t verified.*

---

## **6️⃣ Weekly Definition of Done (DoD)**

Each week has an explicit **definition of done** based on trust and verification, not code volume.

**Examples**

* “I can visually confirm BOS/CTS correctness on multiple historical segments”

* “Zone strength flags match my discretionary intuition”

* “Entries plotted on chart make sense without explanation”

**Principle**

*Progress is measured by confidence, not commits.*

---

## **7️⃣ Scope Control via Idea Parking Lot**

New ideas are inevitable. Uncontrolled implementation is forbidden.

**Process**

* New ideas go into an `ideas.md` / `future_work.md`

* Each idea includes:

  * problem it solves

  * where it would plug in

  * why it’s not implemented yet

* Ideas are implemented only when the syllabus reaches that layer

**Principle**

*Ideas are captured immediately, implemented deliberately.*

---

## **8️⃣ Optimization Discipline**

Optimization is structured, controlled, and delayed until logic is stable.

**Rules**

* No optimization without walk-forward validation

* No parameter tuning before behavior is trusted visually

* No “parameter fishing” to fix logical flaws

**Principle**

*Optimization refines logic — it never replaces it.*

---

## **9️⃣ Machine Learning Gating Rule**

Machine learning is introduced **only after the system becomes boring**.

ML may:

* score

* rank

* filter

* adjust confidence

ML may **not**:

* replace market structure logic

* generate raw trade decisions

* override risk controls

**Principle**

*ML enhances judgment — it does not invent it.*

---

## **🔟 Strategy Versioning Is Mandatory**

Every backtest, chart payload, and trade log must record the **strategy version**.

**Example**

`strategy_version: v1.2.0`  
`- candle_classification_v2`  
`- zone_strength_v1`  
`- entry_confirmation_required`  
`- management_BE_at_1R`

**Principle**

*If you can’t name the logic, you can’t trust the result.*

---

## **🧭 Final Operating Principle**

This project prioritizes:

1. Correctness

2. Explainability

3. Reproducibility

4. Stability

5. Performance (last)

If any trade-off is required, decisions must follow this order.

