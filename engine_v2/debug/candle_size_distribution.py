"""Candle size (range + body) distribution tables, by body-% candle type.

Standalone measurement tool for the pinbar body floor (features/pinbar_body_floor.py + candles_v2.py:classify_candles
step 4). It answers: how are candle RANGE (`candle_len`) and BODY (`body_len`) lengths distributed at each TF, split
by the body-% candle type (the floor not applied)?

Run:
    python -m engine_v2.debug.candle_size_distribution

Methodology (important for reproducibility):
  - Uses the SAME data the pipeline uses: H1 via `fetch_history_with_auto_extend`
    (so the H1 candle set matches a real replay exactly), and M15/M5 fetched over
    the resulting H1 time span via `data_bridge.fetch_lower_tf_data`. M5 is NOT run
    through market structure anywhere in the engine — here we only fetch + classify
    it, purely for this measurement.
  - Classification is the NATURAL (body-%) classification: an all-NaN `body_floor`
    (no floor on any candle). This is deliberate — the floor acts ON the natural
    maru/normal population, so the tables show that population before the floor
    touches it.
  - Pip size is resolved per-pair (JPY=0.01 else 0.0001), matching the engine.
  - "non-maru" = normal + pinbar.

The fixed-floor demotion table this tool printed until PLAN_U U2 (2026-10-28) is retired with the fixed per-TF floor
it measured (and it compared without the EPS guard). The rolling floor's measurements live in
`plans/qa_pinbar_floor_inputs/` (`rolling_floor_demotions.py`, `plan_u_prediction.py`, ...).

To regenerate after a data-window change, just rerun — it reads engine_v2.config
for pair/window/timeframe. Edit GROUPS / LOWER_TFS below to adjust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine_v2.config import CONFIG
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import compute_candle_features
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
from engine_v2.run_replay import fetch_history_with_auto_extend, _floor_to_day_start


# --- knobs ---------------------------------------------------------------
LOWER_TFS = ("M15", "M5")          # fetched + classified (no MS); H1 always included
GROUPS = ("ALL", "MARU", "NON-MARU", "NORMAL", "PINBAR")


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in str(pair).upper() else 0.0001


def _classify_natural(df_raw: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Compute candle features with NO pinbar floor (natural classes): an all-NaN `body_floor`."""
    df_raw = df_raw.copy()
    df_raw.attrs["pair"] = pair
    return compute_candle_features(df_raw, CandleParams(), anchor_shifts=(0, 1, 2),
                                   body_floor=np.full(len(df_raw), np.nan))


def _group_mask(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "ALL":
        return pd.Series(True, index=df.index)
    if name == "NON-MARU":
        return df["candle_type"] != "maru"
    return df["candle_type"] == name.lower()


def _stats(series_pips: pd.Series) -> tuple:
    s = series_pips
    return (s.min(), s.quantile(.25), s.median(), s.mean(), s.quantile(.75), s.max())


def report_distribution(df: pd.DataFrame, tf: str, pip: float) -> None:
    df = df.copy()
    df["len_p"] = df["candle_len"] / pip
    df["body_p"] = df["body_len"] / pip
    n = len(df)
    groups = [(g, df[_group_mask(df, g)]) for g in GROUPS]

    print(f"\n############ {tf}  (all={n}) ############")
    print("counts: " + "  ".join(f"{g}={len(d)} ({len(d)/n*100:.1f}%)" for g, d in groups))
    for metric, col in [("candle_len (range)", "len_p"), ("body_len", "body_p")]:
        print(f"\n  --- {metric} [pips] ---")
        print(f"  {'group':9} {'min':>6} {'25%':>7} {'median':>7} {'avg':>7} {'75%':>7} {'max':>7}")
        for g, d in groups:
            mn, p25, md, av, p75, mx = _stats(d[col])
            print(f"  {g:9} {mn:6.2f} {p25:7.2f} {md:7.2f} {av:7.2f} {p75:7.2f} {mx:7.2f}")


def main() -> None:
    pair = CONFIG.pair
    pip = _pip_size(pair)

    # H1: identical fetch to a real replay (auto-extend lookback).
    df_h1, eff_start, _ = fetch_history_with_auto_extend(
        pair=pair,
        timeframe=CONFIG.timeframe,
        start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end,
        lookback_days=183,
        min_history=50,
        extend_days=4,
        max_extend_iters=8,
    )
    tmin = pd.to_datetime(df_h1["time"]).min()
    tmax = pd.to_datetime(df_h1["time"]).max()
    print(f"pair={pair} pip={pip}  H1 span {tmin} .. {tmax} ({len(df_h1)} candles)")

    classified = [(CONFIG.timeframe, _classify_natural(df_h1, pair))]
    for tf in LOWER_TFS:
        raw = fetch_lower_tf_data(pair, tf, tmin, tmax)  # raises on a failed / empty fetch
        classified.append((tf, _classify_natural(raw, pair)))

    print("\n" + "=" * 30 + " DISTRIBUTION STATS (natural classification) " + "=" * 30)
    for tf, df in classified:
        report_distribution(df, tf, pip)


if __name__ == "__main__":
    main()
