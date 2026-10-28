"""The rolling pinbar body floor (PLAN_U §3 / §3.1; the user, 2026-10-27 / 28 — CANDLE_BODY_FLOOR_NOTES "Current logic").

The floor of candle i = `numpy.percentile` (linear) of the bodies of the `params.pinbar_floor_window` most recent
pinbars STRICTLY before i, at `params.pinbar_floor_percentile`; no full window -> NaN (no floor). Which pinbars fill
the window = `params.pinbar_floor_window_source`:
  - "natural": the body-% pinbars (the floor not applied) — a candle's floor depends only on the candles before it;
  - "final": the final pinbars (body-% + the ones the floor demoted), candle by candle in index order, each
    demotion entering the next windows.

Pure: no I/O. A replay's history before its frame (the seed, `run_replay.fetch_floor_seed`) only fills the first
windows; the frame's own features stay frame-only.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from engine_v2.common.types import COL_TIME
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import classify_candles, compute_candle_metrics, floor_demotes

WINDOW_SOURCES = ("natural", "final")


def rolling_pinbar_body_floor(body_len, body_pct_type, params: CandleParams) -> Tuple[np.ndarray, np.ndarray]:
    """`(floor, demoted)` over the rows given, in time order: `floor` one float per row (price units, NaN = no full
    window), `demoted` the rows the floor makes pinbars (`floor_demotes` — the predicate step 4 applies)."""
    source = params.pinbar_floor_window_source
    if source not in WINDOW_SOURCES:
        raise ValueError(f"[pinbar_body_floor] pinbar_floor_window_source must be one of {WINDOW_SOURCES}, "
                         f"got {source!r}")
    size = int(params.pinbar_floor_window)
    pct = float(params.pinbar_floor_percentile)
    body = np.asarray(body_len, dtype=float).reshape(-1)
    btype = np.asarray(body_pct_type).reshape(-1)
    n = len(body)
    floor = np.full(n, np.nan)
    demoted = np.zeros(n, dtype=bool)
    window: deque = deque(maxlen=size)   # the window's pinbar bodies, oldest first
    current = np.nan                     # the floor of the next candle
    for i in range(n):
        floor[i] = current
        demoted[i] = bool(floor_demotes(body[i], btype[i], floor[i]))
        if btype[i] == "pinbar" or (source == "final" and demoted[i]):
            window.append(body[i])
            if len(window) == size:
                current = float(np.percentile(np.asarray(window), pct))
    return floor, demoted


def _body_pct_types(df: pd.DataFrame, params: CandleParams) -> pd.DataFrame:
    """Metrics + the body-% types (`classify_candles` without a floor) of `df`'s rows."""
    return classify_candles(compute_candle_metrics(df), params)


def pinbar_body_floor_for_frame(frame: pd.DataFrame, seed: Optional[pd.DataFrame],
                                params: CandleParams) -> Tuple[np.ndarray, np.ndarray]:
    """`rolling_pinbar_body_floor` over `seed` + `frame` (time order), the frame's part returned; `seed` = None -> the
    frame alone (the candles before its `pinbar_floor_window`-th pinbar get no floor). The seed contributes ONLY to
    the windows; asserted: its last candle is strictly before the frame's first."""
    fr = _body_pct_types(frame, params)
    if seed is None or len(seed) == 0:
        return rolling_pinbar_body_floor(fr["body_len"], fr["candle_type"], params)
    seed_last = pd.to_datetime(seed[COL_TIME].iloc[-1], utc=True)
    frame_first = pd.to_datetime(frame[COL_TIME].iloc[0], utc=True)
    assert seed_last < frame_first, \
        f"[pinbar_body_floor] the seed ends {seed_last}, not strictly before the frame's first candle {frame_first}"
    sd = _body_pct_types(seed, params)
    body = np.concatenate([sd["body_len"].to_numpy(dtype=float), fr["body_len"].to_numpy(dtype=float)])
    btype = np.concatenate([sd["candle_type"].to_numpy(), fr["candle_type"].to_numpy()])
    floor, demoted = rolling_pinbar_body_floor(body, btype, params)
    s = len(sd)
    return floor[s:], demoted[s:]
