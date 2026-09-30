<<<<<<< HEAD
# Multi-Timeframe Forex Trading System

This repository contains a research-first, event-driven, multi-timeframe forex trading system.
The system is designed to formalize discretionary market-structure concepts into a deterministic,
testable, and explainable trading engine.

The project prioritizes correctness, interpretability, and reproducibility over speed or complexity.

---

## 🎯 Project Goals

- Encode discretionary market structure concepts (BOS, CTS, zones, patterns) into code
- Support multi-timeframe analysis (HTF context → LTF execution)
- Provide strong visual validation via charting
- Enable deterministic backtesting and walk-forward optimization
- Support safe automation via Oanda (dry-run → live)
- Preserve explainability at every decision point

---

## 🧠 Design Philosophy (Read This First)

This system follows several non-negotiable principles:

- **Explainability > Performance**
- **Interfaces before behavior**
- **Visual validation is mandatory**
- **Research engine first, trading bot second**
- **Event-driven by design**
- **Optimization refines logic — it never replaces it**

These principles are documented in detail in `PROJECT_PRINCIPLES.md`.

---

## 🗂 Repository Structure

```text
Forex_Algo/
├── PROJECT_PRINCIPLES.md        # Core project guardrails
├── WEEKLY_DEFINITION_OF_DONE.md # Definition of success per week
├── IDEA_PARKING_LOT.md          # Deferred-items register: EVERY deferred / parked / planned-but-unbuilt item + ideas
├── STRATEGY_VERSIONING.md       # Strategy versioning convention
├── README.md                    # This file
│
├── engine_v2/                   # Core trading engine (active development)
│   ├── data/                    # Data access & candle feeds
│   ├── features/                # Candle classification & basic features
│   ├── patterns/                # Multi-candle pattern detection
│   ├── structure/               # Market structure (BOS / CTS)
│   ├── zones/                   # KL and OB/POI zones
│   ├── indicators/              # Supporting indicators (liquidity, imbalance, etc.)
│   ├── execution/               # Order & trade management
│   ├── backtest/                # Backtesting & simulation
│   ├── charting/                # Chart payload export
│   └── run_replay.py            # Main replay / research entrypoint
│
├── artifacts/                   # Generated outputs (not source code)
│   ├── charts/
│   ├── backtests/
│   └── logs/
│
└── ideas/                       # Optional deeper experiment notes
=======
# trading
>>>>>>> f318e9d013647d88a9028ba4b5c5b297b634f449
