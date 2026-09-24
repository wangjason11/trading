"""Shared cycle-0 true-first-breakout search (DESIGN LOCKED 2026-06-07).

The ONE routine that both the unified probe (deterministic method) and
MarketStructure's pre-CTS_0 scan mode call to locate a structure's cycle-0
CTS establishment. Full spec lives in
`memory/project_true_first_breakout_cycle0.md`; this module implements the
four conditions verbatim so the probe and MS can never diverge:

    A structure's cycle-0 CTS (CTS_0_EST) is the EARLIEST breakout meeting
    ALL of:
      1. Anchor breaks the BOS_0 inner — the anchor candle's CLOSE is past
         `bos0_inner` in `direction` (a hard gate, even on the confirm
         path). Delegated to the detectors' `check_break` via the
         `break_threshold=bos0_inner` argument (mechanism B).
      2. Valid breakout pattern (continuous / double_maru /
         one_maru_continuous / one_maru_opposite), RE-DETECTED with
         `break_threshold = bos0_inner` (mechanism B — never the
         threshold-free `df.pat`). 30%-body failures may be CONFIRMED via
         the detectors' existing pattern-extreme confirmation.
      3. Strict new extreme — the FULL-pattern extreme (max-high for +1 /
         min-low for -1 across ALL pattern candles incl. the confirm
         candle) is a STRICT new extreme over `[current_start,
         extreme_candle)` (`>` highs / `<` lows; ties do NOT count).
      4. Earliest apply/confirm idx wins (NOT anchor idx). Tie-break
         `continuous > double_maru > one_maru_continuous > one_maru_opposite`
         — a CYCLE-0-SPECIFIC order, distinct from the global
         `detect_best_for_anchor` priority.

This routine is the inner search for ONE `current_start` with ONE fixed
`bos0_inner` threshold. The retrace-reset loop and the *moving* of the
BOS_0 threshold across resets (iteration 1 = reference inner; iteration 2+
= ad-hoc BOS_0 at the moved start) live in the CALLER (the probe), which
re-invokes this routine per reset with the new `current_start` /
`bos0_inner`. Cycle 0 only — cycles 1+ break the prior CTS by construction
and are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from engine_v2.common.types import PatternEvent, PatternStatus
from engine_v2.patterns.structure_patterns import BreakoutPatterns


# Cycle-0-specific tie-break order (gotcha #2/#3). This is ONLY used to
# break ties at an equal apply/confirm idx within this routine. It is
# deliberately DIFFERENT from the global `detect_best_for_anchor` priority
# (`continuous > one_maru_opposite > one_maru_continuous > double_maru`),
# which other MS callers depend on — do NOT change that one.
_CYCLE0_PRIORITY = {
    "continuous": 0,
    "double_maru": 1,
    "one_maru_continuous": 2,
    "one_maru_opposite": 3,
}


@dataclass(frozen=True)
class TrueFirstBreakout:
    """The winning cycle-0 breakout located by `find_true_first_breakout`.

    `est_idx` is the apply/confirm idx — the candle that establishes
    cycle-0 CTS (gotcha #15). It equals MS's `_apply_idx`
    (`confirmation_idx` for CONFIRMED patterns, else `end_idx`) so the
    probe and MS agree on the establishment candle.

    `extreme_idx` / `extreme_price` are the FULL-pattern extreme (incl. the
    confirm candle) — the CTS price/idx MS records via
    `_cts_from_breakout_event`. `pattern` is the raw winning PatternEvent
    so MS can drive its existing establishment path with it.
    """
    pattern: PatternEvent
    pattern_anchor_idx: int  # the breakout pattern's first candle (pattern.start_idx)
    est_idx: int         # apply/confirm idx (= cycle-0 establishment candle)
    extreme_idx: int     # full-pattern extreme candle idx (incl. confirm candle)
    extreme_price: float  # the strict new-extreme value


def _apply_idx(pat: PatternEvent) -> Optional[int]:
    """The apply/confirm idx of a pattern — identical to MS's `_apply_idx`
    (confirmation candle for CONFIRMED, else the pattern's `end_idx`)."""
    if pat.status == PatternStatus.CONFIRMED:
        return pat.confirmation_idx
    return pat.end_idx


def _full_pattern_extreme(
    df: pd.DataFrame,
    pat: PatternEvent,
    direction: int,
) -> Optional[Tuple[int, float]]:
    """Full-pattern extreme over `[start_idx .. max(end_idx, confirm_idx)]`.

    Mirrors `MarketStructure._cts_from_breakout_event` (which extends the
    span to include the confirmation candle): max-high for +1, min-low for
    -1. Returns `(extreme_idx, extreme_price)` as POSITIONAL idx, or None
    if the span is out of bounds.
    """
    if pat.start_idx is None or pat.end_idx is None:
        return None
    s = int(pat.start_idx)
    e = int(pat.end_idx)
    if pat.confirmation_idx is not None:
        e = max(e, int(pat.confirmation_idx))
    if e < s:
        s, e = e, s
    n = len(df)
    if s < 0 or e >= n:
        return None
    span = df.iloc[s : e + 1]
    if span.empty:
        return None
    if direction == 1:
        vals = span["h"].astype(float).values
        k = int(vals.argmax())
        return s + k, float(vals[k])
    vals = span["l"].astype(float).values
    k = int(vals.argmin())
    return s + k, float(vals[k])


def _is_strict_new_extreme(
    df: pd.DataFrame,
    current_start: int,
    extreme_idx: int,
    extreme_price: float,
    direction: int,
) -> bool:
    """STRICT new extreme over the half-open window `[current_start,
    extreme_idx)` (gotcha #1 strict, #6 window).

    For +1: `extreme_price` strictly greater than every prior high. For
    -1: strictly less than every prior low. An empty prior window
    (`extreme_idx <= current_start`) trivially passes — there is nothing
    earlier to beat.
    """
    if extreme_idx <= current_start:
        return True
    prior = df.iloc[current_start:extreme_idx]  # iloc end-exclusive → [current_start, extreme_idx)
    if prior.empty:
        return True
    if direction == 1:
        return float(extreme_price) > float(prior["h"].astype(float).max())
    return float(extreme_price) < float(prior["l"].astype(float).min())


def _anchor_candidates(
    bp: BreakoutPatterns,
    idx: int,
    direction: int,
    bos0_inner: Optional[float],
) -> List[Tuple[PatternEvent, int]]:
    """Per-anchor breakout candidates (mechanism B), each paired with its
    cycle-0 priority.

    Re-detects every breakout pattern at `idx` with
    `break_threshold=bos0_inner` and `do_confirm=True` (so 30%-body
    failures resolve their confirmation — gotcha #4: confirmation tests
    the pattern's OWN extreme, NOT `bos0_inner`). Only SUCCESS / CONFIRMED
    patterns are returned; FAIL_NEEDS_CONFIRM (confirmation never landed)
    is dropped.
    """
    raw = [
        bp.continuous(idx, direction, bos0_inner, do_confirm=True),
        bp.double_maru(idx, direction, bos0_inner, do_confirm=True),
        bp.one_maru_continuous(idx, direction, bos0_inner, do_confirm=True),
        bp.one_maru_opposite(idx, direction, bos0_inner, do_confirm=True),
    ]
    out: List[Tuple[PatternEvent, int]] = []
    for pat in raw:
        if pat is None:
            continue
        if pat.status not in (PatternStatus.SUCCESS, PatternStatus.CONFIRMED):
            continue
        prio = _CYCLE0_PRIORITY.get(str(pat.name))
        if prio is None:
            continue
        out.append((pat, prio))
    return out


def find_true_first_breakout(
    bp: BreakoutPatterns,
    current_start: int,
    upper_idx: int,
    direction: int,
    bos0_inner: Optional[float],
) -> Optional[TrueFirstBreakout]:
    """Locate the cycle-0 true-first-breakout in `[current_start, upper_idx]`.

    Scans every anchor for breakout candidates (mechanism B, threshold =
    `bos0_inner`), keeps those whose full-pattern extreme is a strict new
    extreme over `[current_start, extreme_candle)`, and returns the one
    with the earliest apply/confirm idx (tie-break by `_CYCLE0_PRIORITY`).

    Returns None when no candidate in the window qualifies — the caller
    (probe) treats that as "no breakout found" (finalize / continue past
    the window per the no-breakout handoff rule).

    Notes
    -----
    - Indices are POSITIONAL (the detectors use `df.iloc`; callers pass
      positional `start`/`end`). `df` is assumed 0-based RangeIndexed
      (loc == iloc), the standard MS / slice convention.
    - A candidate's apply/confirm idx is always `>= pattern_anchor_idx + 1`, so
      once a best `est_idx = E` is found, no anchor at `idx >= E` can
      tie or beat it (its est would be `>= idx + 1 > E`). The scan stops
      there — bounding work to roughly `[current_start, CTS_0_EST]`.
    - The confirm candle must land within the window: a candidate whose
      `est_idx > upper_idx` is rejected (its confirmation needs data past
      the bound — that is the caller's edge-pending concern, dormant in
      backtest).
    """
    df = bp.df
    n = len(df)
    lo = int(current_start)
    hi = min(int(upper_idx), n - 1)
    if lo > hi:
        return None

    best: Optional[TrueFirstBreakout] = None
    best_key: Optional[Tuple[int, int]] = None

    for idx in range(lo, hi + 1):
        # Early stop: any candidate here has est >= idx + 1, so once a
        # winner with est_idx = best_key[0] exists, idx >= that est can
        # neither tie nor beat it.
        if best_key is not None and idx >= best_key[0]:
            break

        for pat, prio in _anchor_candidates(bp, idx, direction, bos0_inner):
            est = _apply_idx(pat)
            if est is None or int(est) > hi:
                continue
            ext = _full_pattern_extreme(df, pat, direction)
            if ext is None:
                continue
            ext_idx, ext_price = ext
            if not _is_strict_new_extreme(df, lo, ext_idx, ext_price, direction):
                continue
            key = (int(est), int(prio))
            if best_key is None or key < best_key:
                best_key = key
                best = TrueFirstBreakout(
                    pattern=pat,
                    pattern_anchor_idx=int(pat.start_idx),
                    est_idx=int(est),
                    extreme_idx=int(ext_idx),
                    extreme_price=float(ext_price),
                )

    return best
