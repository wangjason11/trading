# GLOSSARY.md — Domain Terminology Reference

> Definitions for domain-specific terms used throughout the codebase. Refer here when encountering unfamiliar terminology.

---

## Structure Terms

| Term | Definition |
|------|------------|
| **struct_direction** | Regime direction: +1 for uptrend, -1 for downtrend |
| **CTS** | Continue the Structure — continuation swing confirmed by pullback |
| **BOS** | Break of Structure — break level confirmed by breakout |
| **range** | Consolidation period bounded by (range_hi, range_lo) |
| **reversal watch** | Monitoring window after close-break of BOS threshold |
| **structure_id** | Regime identifier; increments on reversal (sid 0, sid 1, etc.) |
| **cts_cycle_id** | Cycle counter within a structure_id (resets on reversal) |

---

## Zone Terms

| Term | Definition |
|------|------------|
| **base_pattern** | Candle pattern at zone anchor (pinbar, star, long-tail, etc.) |
| **base_idx** | Index of the zone anchor candle |
| **bounds_steps** | History of zone boundary changes (INIT → expansions) |
| **outer threshold** | Zone boundary price (before top/bottom conversion) |
| **inner threshold** | Zone boundary price closer to current price |
| **KL zone** | Key Level zone — derived from CTS/BOS confirmation events |

---

## Timing Terms

| Term | Definition |
|------|------------|
| **apply_idx** | Candle where pattern effect is applied (end_idx or confirmation_idx) |
| **confirmed_idx** | Candle where zone becomes confirmed for charting |
| **confirmed_at** | Candle index in event meta where level was confirmed |
| **start_idx** | Logical start of a range or structure (may differ from confirm_idx) |
| **confirm_idx** | Candle index when event was emitted (may differ from start_idx) |
| **too_early** | Flag when identify_start finds extreme before min_history |

---

## Event Terms

| Term | Definition |
|------|------------|
| **event.idx** | Candle index where event was emitted |
| **event.meta** | Dictionary of event-specific metadata |
| **STATE_CHANGED** | Transition between market states (e.g., cts → bos) |
| **CTS_CONFIRMED** | Continue-the-structure level confirmed |
| **BOS_CONFIRMED** | Break-of-structure level confirmed |
| **RANGE_STARTED** | New consolidation range detected |
| **RANGE_UPDATED** | Range bounds expanded |
| **RANGE_RESET** | Range closed (breakout or reversal) |

---

## Pattern Terms

| Term | Definition |
|------|------------|
| **pinbar** | Single candle with long wick rejecting price level |
| **star** | Small-bodied candle indicating indecision |
| **long-tail** | Candle with extended tail showing price rejection |
| **maru** | Strong-bodied candle (marubozu-like) |
| **breakout pattern** | Multi-candle pattern signaling structure break |

---

## Pipeline Terms

| Term | Definition |
|------|------------|
| **orchestrator** | Central pipeline coordinator (pipeline/orchestrator.py) |
| **feature** | Computed column added to dataframe (e.g., swing detection) |
| **base features** | Foundation features required by market structure |
| **charting** | Final pipeline stage that generates visualizations |
