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
| **base inside bar** | Pattern where anchor candle has ≥2 neighboring candles entirely within its range |
| **base_idx** | Index of the zone anchor candle |
| **anchor_idx** | Index of the BOS/CTS candle used for pattern identification |
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

## Wave Candle Terms

| Term | Definition |
|------|------------|
| **wave candle** | Candle at the boundary between two waves (ending wave → starting wave) at a KL zone |
| **last wave candle** | Final candle of the ending wave (last pullback for BOS, last breakout for CTS) |
| **first wave candle** | Initial candle of the starting wave (first breakout for BOS, first pullback for CTS) |
| **qualified candle** | Candle with correct direction AND vol_dir matching (vol_dir == candle dir or vol_dir == 0) |
| **compound first-wave** | First wave candle condition: (big_normal & (maru OR normal)) OR (big_maru & pinbar & pinbar_dir == wave_dir) |
| **BIB (base inside bar)** | Zone base pattern requiring special forward/backward search logic for wave candles |

---

## WVMI Terms

| Term | Definition |
|------|------------|
| **WVMI** | Wave Volume Momentum Indicator — measures BOS zone strength via volume ratios of wave candle pairs |
| **breakout momentum** | `(LB_vol * LB_weight) / FB_vol` — locked at CTS_n confirmation |
| **pullback momentum** | `(LP_vol * LP_weight) / FP_vol` — shifts until BOS_n+1 locks it |
| **last wave weight** | Weight (0.5/0.7/1.0) applied to LB/LP volume based on candle type and size |
| **temporary LP** | Last Pullback candle that shifts to qualified candle closest to outer bound; finalizes on BOS_n+1 |
| **buy_momentum / sell_momentum** | Direction-labeled wrappers: buy zone → buy=breakout, sell=pullback; sell zone → reversed |
| **vol_dir** | Volume direction column: +1 (buying pressure), -1 (selling pressure), 0 (neutral/no signal) |

---

## Scenario 3 Terms

| Term | Definition |
|------|------------|
| **Scenario 3** | Arbitrary-start structure analysis with iterative BOS_0 probe validation |
| **BOS_0 zone** | First BOS zone from the initial CTS_ESTABLISHED event; used for exception evaluation |
| **Phase 1** | Iterative probing: validate start_idx by checking if price reaches BOS_0 zone inner bound |
| **Phase 2** | Multi-structure continuation from finalized Phase 1 (same logic as `compute_structure`) |
| **finalized** | Scenario 3 status: probe validated, full structure analysis complete |
| **pending** | Scenario 3 status: insufficient data (< 2 CTS_ESTABLISHED), accessible but unfinalized |
| **pip_tolerance** | Distance threshold (default 10 pips) for exception evaluation near zone inner bound |

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
