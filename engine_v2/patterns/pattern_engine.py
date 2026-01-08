from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS
from engine_v2.common.types import PatternEvent, PatternStatus
from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.patterns.range_label import apply_is_range_labels, RangeLabelConfig

@dataclass
class PatternEngineResult:
    df: pd.DataFrame
    events: List[PatternEvent]
    notes: str = ""


def detect_patterns(df: pd.DataFrame, *, break_threshold: Optional[float] = None) -> PatternEngineResult:
    """
    Week 4 scope: Structure-only candle patterns used for breakout/pullback primitives.

    Input df is expected to already have candles_v2 feature columns:
      - direction, candle_type, body_len, candle_len
      - is_big_normal_as0, is_big_normal_as2
      - is_big_maru_as0
      - plus o/h/l/c

    Output:
      - df with marker/debug columns added
      - a list of PatternEvents

    Note: This engine does NOT decide BOS/CTS and does not tag breakout vs pullback.
    """
    _validate_input(df)
    _validate_structure_pattern_inputs(df)

    df2 = df.copy()

    # marker/debug columns
    df2["pat"] = ""
    df2["pat_dir"] = 0
    df2["pat_status"] = ""
    df2["pat_start_idx"] = -1
    df2["pat_end_idx"] = -1
    df2["pat_confirm_idx"] = -1

    bp = BreakoutPatterns(df2)
    events: List[PatternEvent] = []

    n = len(df2)
    for idx in range(n):
        for direction in (1, -1):
            ev = bp.detect_first_success(idx, direction, break_threshold)
            # if idx == 100 and direction == 1:
                # print("[DEBUG detect_first_success idx=100]", ev)
            if ev is None:
                continue

            apply_idx = ev.confirmation_idx if ev.status == PatternStatus.CONFIRMED else ev.end_idx
            if apply_idx is None or apply_idx < 0 or apply_idx >= n:
                continue

            # keep first marker if collision
            row_i = df2.index[apply_idx]
            if df2.at[row_i, "pat"] == "":
                df2.at[row_i, "pat"] = ev.name
                df2.at[row_i, "pat_dir"] = ev.direction
                df2.at[row_i, "pat_status"] = ev.status.value
                df2.at[row_i, "pat_start_idx"] = ev.start_idx
                df2.at[row_i, "pat_end_idx"] = ev.end_idx
                df2.at[row_i, "pat_confirm_idx"] = (
                    ev.confirmation_idx if ev.confirmation_idx is not None else -1
                )

                # only record events that actually produced a marker
                events.append(ev)


    # -------------------------------------------------
    # Week 4 Add-on: break_continue (1-candle continuation)
    # -------------------------------------------------
    eps = 0.0001

    for i in range(1, n):
        # Don't overwrite an existing marker
        cur_row = df2.index[i]
        if df2.at[cur_row, "pat"] != "":
            continue

        prev_row = df2.index[i - 1]

        prev_pat = str(df2.at[prev_row, "pat"])
        prev_status = str(df2.at[prev_row, "pat_status"])
        prev_dir = int(df2.at[prev_row, "pat_dir"])

        # Previous candle must have been a structure-pattern marker and match direction.
        # Only disallowed: one_maru_opposite SUCCESS (CONFIRMED is OK per your rule).
        if prev_pat == "":
            continue
        if prev_status not in (PatternStatus.SUCCESS.value, PatternStatus.CONFIRMED.value):
            continue
        if prev_pat == "one_maru_opposite" and prev_status == PatternStatus.SUCCESS.value:
            continue

        direction = prev_dir
        if direction not in (1, -1):
            continue

        # Current candle conditions
        cur_type = str(df2.at[cur_row, "candle_type"])
        cur_dir = int(df2.at[cur_row, "direction"])
        if cur_type not in ("normal", "maru"):
            continue
        if cur_dir != direction:
            continue

        prev_high = float(df2.at[prev_row, "h"])
        prev_low = float(df2.at[prev_row, "l"])
        cur_close = float(df2.at[cur_row, "c"])

        # Continuation close condition (mirror for bearish)
        if direction == 1:
            if cur_close < (prev_high - eps):
                continue
        else:
            if cur_close > (prev_low + eps):
                continue

        # Mark df columns
        df2.at[cur_row, "pat"] = "break_continue"
        df2.at[cur_row, "pat_dir"] = direction
        df2.at[cur_row, "pat_status"] = PatternStatus.SUCCESS.value
        df2.at[cur_row, "pat_start_idx"] = i - 1
        df2.at[cur_row, "pat_end_idx"] = i
        df2.at[cur_row, "pat_confirm_idx"] = -1

        # Add event (only if marker placed)
        events.append(
            PatternEvent(
                name="break_continue",
                direction=direction,
                start_idx=i - 1,
                end_idx=i,
                status=PatternStatus.SUCCESS,
                confirmation_threshold=None,
                confirmation_idx=None,
                break_threshold_used=None,
                meta={"prev_pat": prev_pat, "prev_status": prev_status},
            )
        )


    # -------------------------------------------------
    # Day 3: is_range candle labeling (event-confirmed)
    # -------------------------------------------------
    df2 = apply_is_range_labels(df2, RangeLabelConfig())

    _validate_output(df2)
    return PatternEngineResult(
        df=df2,
        events=events,
        notes=f"structure pattern events={len(events)}",
    )


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------
def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pattern_engine] Missing required candle columns: {missing}")
    if df.empty:
        raise ValueError("[pattern_engine] Input df is empty")


def _validate_structure_pattern_inputs(df: pd.DataFrame) -> None:
    required = [
        "o",
        "h",
        "l",
        "c",
        "direction",
        "candle_type",
        "body_len",
        "candle_len",
        "is_big_normal_as0",
        "is_big_normal_as1",
        "is_big_normal_as2",
        "is_big_maru_as0",
        "mid_price",
        "body_pct",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[pattern_engine] Missing structure-pattern feature columns: {missing}")


def _validate_output(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pattern_engine] Output df missing required candle columns: {missing}")
