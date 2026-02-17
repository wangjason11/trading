"""Data bridge for multi-TF analysis.

Handles fetching lower-TF data, applying base features, and mapping
H1 candle times to M15 indices.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd

from engine_v2.data.provider_oanda import get_history
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance


def fetch_lower_tf_data(
    pair: str,
    lower_tf: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int = 14,
) -> Optional[pd.DataFrame]:
    """Fetch lower-TF data covering the parent TF date range.

    Called once per replay, not per trigger. Fetches in chunks to stay
    within OANDA's 5000 candle limit per request.
    """
    chunks = []
    cur_start = start.to_pydatetime()
    final_end = end.to_pydatetime()

    while cur_start < final_end:
        cur_end = min(cur_start + timedelta(days=chunk_days), final_end)
        try:
            chunk = get_history(
                pair=pair,
                timeframe=lower_tf,
                start=cur_start,
                end=cur_end,
            )
            if not chunk.empty:
                chunks.append(chunk)
        except Exception as e:
            print(f"[data_bridge] ERROR fetching {lower_tf} chunk {cur_start}-{cur_end}: {e}")

        cur_start = cur_end

    if not chunks:
        print(f"[data_bridge] WARNING: No {lower_tf} data for {pair} {start} - {end}")
        return None

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df.attrs["pair"] = pair
    print(f"[data_bridge] Fetched {len(df)} {lower_tf} candles for {pair} in {len(chunks)} chunks")
    return df


def prepare_lower_tf_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply candle classification, patterns, and imbalance to raw lower-TF data.

    Called once per replay. Returns the prepared df ready for structure analysis.
    """
    # 1) Candle features
    c_res = apply_candle_classification(df_raw)

    # 2) Pattern engine
    p_res = detect_patterns(c_res.df)

    # 3) Imbalance patterns
    df_prepared = compute_imbalance(p_res.df)

    # Preserve attrs
    df_prepared.attrs.update(df_raw.attrs)

    return df_prepared


def map_candle_to_lower_tf(
    h1_time: pd.Timestamp,
    h1_extreme_price: float,
    h1_sd: int,
    m15_df: pd.DataFrame,
) -> Optional[int]:
    """Map an H1 candle to the M15 candle whose extreme matches.

    H1 candle at time T covers M15 candles at T+0, T+15, T+30, T+45.
    For sd=+1 (CTS marks high): find M15 candle with highest high; tie -> last.
    For sd=-1 (CTS marks low): find M15 candle with lowest low; tie -> last.

    Returns the M15 DataFrame index, or None if no candles found.
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

    if h1_sd == 1:
        # CTS marks HIGH -> find M15 with highest high (tie: last)
        max_high = candidates["h"].max()
        matches = candidates[candidates["h"] == max_high]
        best_idx = int(matches.index[-1])
    else:
        # CTS marks LOW -> find M15 with lowest low (tie: last)
        min_low = candidates["l"].min()
        matches = candidates[candidates["l"] == min_low]
        best_idx = int(matches.index[-1])

    return best_idx
