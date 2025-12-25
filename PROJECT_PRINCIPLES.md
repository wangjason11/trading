# Project Principles
Multi-Timeframe Forex Trading System

This document defines the core, non-negotiable principles governing the design,
implementation, validation, and evolution of this trading system.

These principles take precedence over speed, convenience, or short-term results.

---

## 1. Explainability > Performance
All logic must be explainable, inspectable, and reproducible.

- Slow but explainable logic is acceptable.
- Fast but opaque logic is not.
- No trade should exist that cannot be reconstructed visually and logically.

> If the system cannot explain itself, it is not allowed to trade.

---

## 2. Interfaces Before Behavior
Core objects must have stable interfaces even while their internal logic evolves.

Examples:
- Candle → metrics, type, direction
- PatternEvent → name, direction, time, metadata
- Zone → bounds, timeframe, strength, flags, status
- TradeIntent → entry, stop, TP, RR, provenance

Interfaces are frozen early so downstream logic does not break during iteration.

---

## 3. Visualization Is Mandatory
Visual validation is a first-class requirement, not a debugging convenience.

All of the following must be visually inspectable:
- Candle classifications
- Pattern markers
- BOS / CTS levels
- Zones (KL and OB/POI)
- Entries, stops, take profits
- Executed trades

> If it cannot be verified visually, it is not verified.

---

## 4. Research Engine First
The system is built as a research engine before it is treated as a trading bot.

Live trading is a by-product of:
- correct structure detection
- trusted zones
- explainable entries
- controlled risk

---

## 5. Event-Driven Design
All decisions must be expressible as events and state transitions.

This enables:
- deterministic backtesting
- accurate replay
- identical live vs historical behavior

---

## 6. Optimization Discipline
Optimization refines logic — it never replaces it.

- No optimization before logic is trusted visually.
- No parameter tuning to compensate for flawed logic.
- Walk-forward validation is mandatory.

---

## 7. Machine Learning Gating
Machine learning is introduced only after the system becomes stable and boring.

ML may score, rank, or filter decisions.
ML may not replace structure, zones, or risk logic.

---

## 8. Strategy Versioning
Every result must be traceable to a strategy version.

If the logic cannot be named, the result cannot be trusted.

---
