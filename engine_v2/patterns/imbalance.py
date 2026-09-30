# engine_v2/patterns/imbalance.py
"""
Imbalance (FVG) pattern detection for POI Zones and Fib activation.

Two concepts:
- Imbalance candle: a c2 (middle) candle whose neighbors (c1, c3) form a gap
  and whose own direction matches the gap direction. Flagged per-candle via
  `is_imbalance == 1` on the dataframe (charting consumes this).
- Imbalance instance: one or more consecutive same-direction imbalance candles
  merged into a single entity with merged gap bounds. Stored as
  ImbalanceInstance objects in `df.attrs["imbalances"]` (Fib/POI consume this).

Detection rules:
- Bullish FVG: c1.high < c3.low AND c2 direction == +1
- Bearish FVG: c1.low  > c3.high AND c2 direction == -1
- Consecutive same-direction imbalance candles form one merged instance.

Merged gap bounds:
- Bullish: gap_bottom = first c1.high, gap_top = last c3.low
- Bearish: gap_bottom = last c3.high,  gap_top = first c1.low
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from engine_v2.common.types import ImbalanceInstance


def compute_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute imbalance candles (column flag) and imbalance instances (attrs list).

    Sets:
    - df["is_imbalance"] (0/1): 1 for every c2 candle that forms an FVG with
      matching c2 direction.
    - df.attrs["imbalances"]: List[ImbalanceInstance] with merged bounds.

    Requires df to already have `direction` column from candle classification.
    """
    df = df.copy()
    df["is_imbalance"] = 0

    if len(df) < 3 or "direction" not in df.columns:
        df.attrs["imbalances"] = []
        return df

    # ---- Pass 1: flag individual imbalance candles ----
    h_vals = df["h"].values
    l_vals = df["l"].values
    dir_vals = df["direction"].values
    is_imb_col = df.columns.get_loc("is_imbalance")

    for idx in range(1, len(df) - 1):
        c1_h = h_vals[idx - 1]
        c1_l = l_vals[idx - 1]
        c3_h = h_vals[idx + 1]
        c3_l = l_vals[idx + 1]
        c2_dir = int(dir_vals[idx])

        # Bullish FVG + bullish c2
        if c1_h < c3_l and c2_dir == 1:
            df.iloc[idx, is_imb_col] = 1
        # Bearish FVG + bearish c2
        elif c1_l > c3_h and c2_dir == -1:
            df.iloc[idx, is_imb_col] = 1

    # ---- Pass 2: merge consecutive same-direction flags into instances ----
    instances: List[ImbalanceInstance] = []
    is_imb_vals = df["is_imbalance"].values

    i = 1
    while i < len(df) - 1:
        if is_imb_vals[i] != 1:
            i += 1
            continue

        direction = int(dir_vals[i])
        run_start = i
        run_end = i
        j = i + 1
        while (
            j < len(df) - 1
            and is_imb_vals[j] == 1
            and int(dir_vals[j]) == direction
        ):
            run_end = j
            j += 1

        if direction == 1:
            gap_bottom = float(h_vals[run_start - 1])  # first c1.high
            gap_top = float(l_vals[run_end + 1])       # last c3.low
        else:
            gap_bottom = float(h_vals[run_end + 1])    # last c3.high
            gap_top = float(l_vals[run_start - 1])     # first c1.low

        instances.append(
            ImbalanceInstance(
                start_idx=run_start,
                end_idx=run_end,
                direction=direction,
                gap_top=gap_top,
                gap_bottom=gap_bottom,
                gap_size=gap_top - gap_bottom,
            )
        )
        i = j

    df.attrs["imbalances"] = instances
    return df


# ---------------------------------------------------------------------------
# Instance-based aggregation helpers
# ---------------------------------------------------------------------------

def has_unfilled_imbalance(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    check_to_idx: int,
    fill_threshold: float = 0.70,
    *,
    direction: Optional[int] = None,
    evaluated_at: Optional[int],
) -> bool:
    """True if at least one imbalance instance overlapping `[start_idx, end_idx]`
    is unfilled as of `check_to_idx`, counting only instances that have FORMED by
    the moment the question is asked (`evaluated_at`).

    Two as-ofs (IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule"):
    ``check_to_idx`` is the fill horizon, ``evaluated_at`` the moment of the
    question.

    Parameters
    ----------
    start_idx, end_idx
        Inclusive window in which the instance must overlap.
    check_to_idx
        The FILL HORIZON — `inst.is_filled` scans `(inst.end_idx, check_to_idx]`
        for the two-stroke fill. Callers pick it for the question asked (the
        handled event's moment, a reference event's moment such as BOS_1's, or
        the current candle — moments on every FibTracker site since Plan E
        E3a / E3a′); IC cond3 keeps the fib's CTS anchor by decision (T2).
    evaluated_at
        Keyword-only and REQUIRED. The MOMENT the question is asked. An instance
        counts only once its first c3 has closed (`inst.formed_at <=
        evaluated_at`), and only its formed prefix is tested against the window
        (`inst.overlaps_formed_prefix`). The cut is keyed on the moment, never on
        `check_to_idx` (which can be an anchor that precedes the moment).
        ``None`` = an explicit "no knowability cut" — retrospective questions,
        the unchanged MS in-flight resolver, cached values judged at their later
        use — and is today's answer.
    direction
        Struct-direction filter; every production caller passes ``sd`` (all
        Fib / scenario / POI checks are sd-direction strict since 2026-05-23 —
        IMBALANCE_FILL_SEMANTICS.md). ``None`` accepts any direction.
    fill_threshold
        Retracement fraction (default 0.70) at which stroke 1 fires.
    """
    for inst in df.attrs.get("imbalances", []):
        if direction is not None and inst.direction != direction:
            continue
        if evaluated_at is None:
            if not inst.overlaps(start_idx, end_idx):
                continue
        elif not inst.overlaps_formed_prefix(start_idx, end_idx, evaluated_at):
            continue
        if not inst.is_filled(df, check_to_idx, fill_threshold):
            return True
    return False
