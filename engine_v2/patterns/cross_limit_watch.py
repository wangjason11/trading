# engine_v2/patterns/cross_limit_watch.py
"""The cross-limit breakout watch (PLAN_K §3.4; PLAN_H §6 RX1-2b, RX1-W1 … W3, RX1-K1, RX1-K6).

A structure started by a NON-reversal trigger that has a cross-fib limit `P` uses the anchor behind `P` as its
"previous structure BOS" equivalent for the IC reference (RX1-2). `P` is not a MarketStructure key level — MS never
checks a breakout against it — so the pattern that BROKE it is found here, the way the reversal watch finds a
reversal against its frozen BOS threshold, but on the BREAKOUT wave (the user, 2026-10-09):

- a candle whose CLOSE is past `P` in the structure's direction opens a watch (RX1-W2; `check_break`'s test);
- the breakout pattern is anchored at that candle: `BreakoutPatterns.detect_best_for_anchor(A, sd, P)` — the
  reversal watch's own detector and pattern set, `P` the break threshold (RX1-W1); its latest confirmation is the
  detector's own `A + 5` (the reversal watch's window — no separate cap);
- `P` never moves: every later close past it re-arms (RX1-W3);
- two watches confirming → the EARLIEST CONFIRMATION wins, ties → the earlier anchor (RX1-K1);
- detection by MS's convention (RX1-K6): the detector sees the frame up to `end_idx` (its look-ahead, `continuous`
  first), and a break COUNTS at a moment `t` iff its confirmation ≤ `known_at = t` — so the in-flight answer is the
  final answer cut by time and the bound never moves.

Pure: no MS state, no structure event. Positional indices (MS / sub frames are RangeIndex, `loc == iloc`).
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from engine_v2.patterns.structure_patterns import BreakoutPatterns, PatternStatus


def find_cross_limit_break(
    df: pd.DataFrame,
    threshold: float,
    sd: int,
    start_idx: int,
    *,
    end_idx: Optional[int] = None,
    known_at: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """`(pattern_anchor_idx, confirmed_idx)` of the break of `threshold` (= the cross limit `P`), or None.

    `start_idx` = the structure's first candle (its BOS_0 anchor — a sub's lifecycle start can lie after its break:
    sub 4, start 3621, break 3588); `end_idx` = the last candle the detector may read (None = the frame's last —
    what every engine caller passes: `FibTracker.ic_reference` hands MS's resolver frame or the projection's slice,
    both already cut at the run's data edge; a cut at a structure's END would be the frame-cut reading K-Q6 rejected,
    PLAN_K §11.2 LA1); `known_at` = the evaluation moment (MS: the fed event's; the final reference: the structure's
    lifecycle cap; None = every confirmation up to `end_idx`).
    """
    sd = int(sd)
    assert sd in (1, -1), f"find_cross_limit_break: sd must be +1 / -1, got {sd!r}"
    last = len(df) - 1 if end_idx is None else min(len(df) - 1, int(end_idx))
    if int(start_idx) > last:
        return None
    bp = BreakoutPatterns(df, end_idx=last)
    closes = df["c"].to_numpy()
    thr = float(threshold)
    best: Optional[Tuple[int, int]] = None
    for a in range(max(0, int(start_idx)), last + 1):
        if best is not None and a >= best[1]:
            break  # every pattern spans >= 2 candles: a later anchor confirms after `a`, so not earlier than best
        c = float(closes[a])
        if not (c > thr if sd == 1 else c < thr):
            continue
        ev = bp.detect_best_for_anchor(a, sd, thr)
        if ev is None:
            continue
        apply_idx = ev.confirmation_idx if ev.status == PatternStatus.CONFIRMED else ev.end_idx
        if apply_idx is None:
            continue
        apply_idx = int(apply_idx)
        if known_at is not None and apply_idx > int(known_at):
            continue
        if best is None or (apply_idx, a) < (best[1], best[0]):
            best = (a, apply_idx)
    return best
