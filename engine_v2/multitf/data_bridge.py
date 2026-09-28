"""Data bridge for multi-TF analysis.

Handles fetching lower-TF data, applying base features, and mapping
H1 candle times to M15 indices.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from engine_v2.data.provider_oanda import get_history
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance

# A chunk whose request fails (any non-200 or `requests` error) is retried
# this many times before the fetch raises (the 2026-09-22 partial fetch was
# two HTTP 504s).
_FETCH_RETRIES = 2
_FETCH_RETRY_WAIT_S = 5.0


def fetch_lower_tf_data(
    pair: str,
    lower_tf: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int = 14,
) -> pd.DataFrame:
    """Fetch lower-TF data covering the parent TF date range.

    Called once per replay, not per trigger. Fetches in chunks to stay
    within OANDA's 5000 candle limit per request.

    Fails loudly — a partial frame never reaches the engine (this function
    used to print a failed chunk and carry on: one run silently got 2500 of
    4228 M15 candles and exited 0). A chunk whose request fails is retried
    (`_fetch_chunk`); a chunk that still fails raises `RuntimeError`
    ``[data_bridge] ERROR fetching …``. A window that yields no candles at all
    raises too: the parent frame spans it, so the lower TF has data. One empty
    chunk is not an error (it is left out of the chunk count).
    """
    chunks = []
    cur_start = start.to_pydatetime()
    final_end = end.to_pydatetime()

    while cur_start < final_end:
        cur_end = min(cur_start + timedelta(days=chunk_days), final_end)
        chunk = _fetch_chunk(pair, lower_tf, cur_start, cur_end)
        if not chunk.empty:
            chunks.append(chunk)

        cur_start = cur_end

    if not chunks:
        raise RuntimeError(f"[data_bridge] ERROR: no {lower_tf} data for {pair} {start} - {end}")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df.attrs["pair"] = pair
    print(f"[data_bridge] Fetched {len(df)} {lower_tf} candles for {pair} in {len(chunks)} chunks")
    return df


def _fetch_chunk(pair: str, lower_tf: str, chunk_start: datetime, chunk_end: datetime) -> pd.DataFrame:
    """One chunk request, retried `_FETCH_RETRIES` times when the request fails.

    Retried: any non-200 status (`get_history` raises `RuntimeError`; a 4xx
    too) and any `requests.RequestException` (a network error, a non-JSON
    body); each retry prints one ``[data_bridge] RETRY k/N …`` line. Raised at
    once: `_load_creds` errors (FileNotFoundError / KeyError / ValueError) and
    a JSON payload missing candle fields (KeyError / ValueError). The exception
    text is collapsed onto one line (a gateway error body is multi-line HTML),
    so every retry / error line keeps the ``[data_bridge]`` prefix.
    """
    for attempt in range(_FETCH_RETRIES + 1):
        try:
            return get_history(pair=pair, timeframe=lower_tf, start=chunk_start, end=chunk_end)
        except (RuntimeError, requests.RequestException) as e:
            why = " ".join(str(e).split())
            if attempt == _FETCH_RETRIES:
                raise RuntimeError(
                    f"[data_bridge] ERROR fetching {lower_tf} chunk {chunk_start}-{chunk_end} "
                    f"after {_FETCH_RETRIES} retries: {why}"
                ) from e
            print(
                f"[data_bridge] RETRY {attempt + 1}/{_FETCH_RETRIES} {lower_tf} chunk "
                f"{chunk_start}-{chunk_end} in {_FETCH_RETRY_WAIT_S:g} s: {why}"
            )
            time.sleep(_FETCH_RETRY_WAIT_S)
    raise AssertionError("unreachable")


def prepare_lower_tf_data(
    df_raw: pd.DataFrame,
    *,
    timeframe: str = "M15",
) -> pd.DataFrame:
    """Apply candle classification, patterns, and imbalance to raw lower-TF data.

    Called once per replay. Returns the prepared df ready for structure analysis.

    `timeframe` selects the per-TF body-pip floor for big_maru/big_normal flags
    (see `apply_candle_classification`). Defaults to "M15" since that's the
    only configured lower TF today.
    """
    # 1) Candle features
    c_res = apply_candle_classification(df_raw, timeframe=timeframe)

    # 2) Pattern engine
    p_res = detect_patterns(c_res.df)

    # 3) Imbalance patterns
    df_prepared = compute_imbalance(p_res.df)

    # Preserve attrs
    df_prepared.attrs.update(df_raw.attrs)

    return df_prepared


def map_candle_to_lower_tf(
    h1_time: pd.Timestamp,
    parent_extreme_dir: int,
    m15_df: pd.DataFrame,
) -> Optional[int]:
    """Map an H1 hour to the M15 candle holding that hour's price extreme.

    H1 candle at time T covers M15 candles at T+0, T+15, T+30, T+45.

    - ``parent_extreme_dir == +1`` → the M15 candle with the HIGHEST high in
      the hour (tie: last) — for a parent-TF price location on the +1 side
      (e.g. the CTS anchor of an up-structure).
    - ``parent_extreme_dir == -1`` → the M15 candle with the LOWEST low
      (tie: last).

    The callers choose the side:
    - the ``first_confluence`` probe (`entity_df_mutation`): its input, the
      parent BOS anchor, with ``-lower_sd`` (spec §4.3.1: the side of the
      parent hour that anchors the OUTER of the sub's reference zone), and its
      end, the parent CTS anchor (``parent_cts_anchor_idx``) with ``+lower_sd``
      (the structure ceiling / floor) — the M15 ``probe_end_idx``;
    - the M15 chart's H1 zone-proximity markers (display): the side of the
      trigger wick (``-1`` when price approaches from above).
    A price-location mapper only — every TIMING value uses the LOH mapper
    (`entity_df_mutation._map_parent_idx_to_m15_hour_end`); never unify them.

    Returns the M15 DataFrame index, or None (with a WARNING print) if no
    candles fall in the hour.
    """
    h1_time = pd.to_datetime(h1_time, utc=True)
    h1_end = h1_time + timedelta(hours=1)

    # Filter M15 candles within this H1 hour
    m15_times = pd.to_datetime(m15_df["time"], utc=True)
    mask = (m15_times >= h1_time) & (m15_times < h1_end)
    candidates = m15_df[mask]

    if candidates.empty:
        print(f"[data_bridge] WARNING: No M15 candles in H1 window {h1_time} - {h1_end}")
        return None

    if parent_extreme_dir == 1:
        # Match the H1 high → M15 with the highest high (tie: last)
        max_high = candidates["h"].max()
        matches = candidates[candidates["h"] == max_high]
        best_idx = int(matches.index[-1])
    else:
        # Match the H1 low → M15 with the lowest low (tie: last)
        min_low = candidates["l"].min()
        matches = candidates[candidates["l"] == min_low]
        best_idx = int(matches.index[-1])

    return best_idx
