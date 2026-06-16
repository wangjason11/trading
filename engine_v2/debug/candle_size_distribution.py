"""Candle size (range + body) distribution + pinbar-floor demotion tables.

Standalone calibration tool for the `pinbar_body_pip_floor` candle-classification
rule (features/candle_params.py + candles_v2.py:classify_candles). It answers two
questions used to pick the per-TF floor values:

  1. How are candle RANGE (`candle_len`) and BODY (`body_len`) lengths distributed
     at each TF, split by candle_type?
  2. For a given body-pip floor, how many natural maru/normal candles would be
     demoted to `pinbar`?

Run:
    python -m engine_v2.debug.candle_size_distribution

Methodology (important for reproducibility):
  - Uses the SAME data the pipeline uses: H1 via `fetch_history_with_auto_extend`
    (so the H1 candle set matches a real replay exactly), and M15/M5 fetched over
    the resulting H1 time span via `data_bridge.fetch_lower_tf_data`. M5 is NOT run
    through market structure anywhere in the engine — here we only fetch + classify
    it, purely for this calibration exercise.
  - Classification is the NATURAL (ratio-based) classification: candle features are
    computed with a fresh `CandleParams()`, whose `pinbar_body_pip_floor` defaults
    to 0.0 (floor DISABLED). This is deliberate — the floor's job is to act ON the
    natural maru/normal population, so the tables must show that population before
    the floor touches it. (With the floor enabled, the demoted candles would already
    be `pinbar` and the "how many would be demoted" question becomes circular.)
  - Pip size is resolved per-pair (JPY=0.01 else 0.0001), matching the engine.
  - "non-maru" = normal + pinbar. The demotion table only counts maru+normal, since
    pinbars are already pinbar (a floor can't demote them further).

To regenerate after a data-window change, just rerun — it reads engine_v2.config
for pair/window/timeframe. Edit FLOORS / GROUPS / LOWER_TFS below to adjust.
"""
from __future__ import annotations

import pandas as pd

from engine_v2.config import CONFIG
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import compute_candle_features
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
from engine_v2.run_replay import fetch_history_with_auto_extend, _floor_to_day_start


# --- knobs ---------------------------------------------------------------
LOWER_TFS = ("M15", "M5")          # fetched + classified (no MS); H1 always included
FLOORS = [round(0.25 * k, 2) for k in range(1, 13)]  # 0.25 .. 3.00 pips
GROUPS = ("ALL", "MARU", "NON-MARU", "NORMAL", "PINBAR")


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in str(pair).upper() else 0.0001


def _classify_natural(df_raw: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Compute candle features with the pinbar floor DISABLED (natural classes)."""
    df_raw = df_raw.copy()
    df_raw.attrs["pair"] = pair
    return compute_candle_features(df_raw, CandleParams(), anchor_shifts=(0, 1, 2))


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


def report_demotion(df: pd.DataFrame, tf: str, pip: float) -> None:
    df = df.copy()
    df["body_p"] = df["body_len"] / pip
    n = len(df)
    nmaru = int((df.candle_type == "maru").sum())
    nnorm = int((df.candle_type == "normal").sum())
    print(f"\n{tf} (all={n}, maru={nmaru}, normal={nnorm}):")
    print(f"  {'floor':>6} {'tot demoted':>13} {'maru hit':>17} {'normal hit':>19}")
    for fl in FLOORS:
        dm = int(((df.candle_type == "maru") & (df.body_p < fl)).sum())
        dn = int(((df.candle_type == "normal") & (df.body_p < fl)).sum())
        tot = dm + dn
        mp = dm / nmaru * 100 if nmaru else 0.0
        npc = dn / nnorm * 100 if nnorm else 0.0
        print(f"  {fl:5.2f}p {tot:5} ({tot/n*100:4.1f}%) {dm:5} ({mp:4.1f}% of maru) {dn:5} ({npc:5.1f}% of normal)")


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
        raw = fetch_lower_tf_data(pair, tf, tmin, tmax)
        if raw is None:
            print(f"[candle_size_distribution] WARNING: no {tf} data; skipping")
            continue
        classified.append((tf, _classify_natural(raw, pair)))

    print("\n" + "=" * 30 + " DISTRIBUTION STATS (natural classification) " + "=" * 30)
    for tf, df in classified:
        report_distribution(df, tf, pip)

    print("\n\n" + "=" * 30 + " FLOOR DEMOTION IMPACT (natural -> pinbar) " + "=" * 30)
    for tf, df in classified:
        report_demotion(df, tf, pip)


if __name__ == "__main__":
    main()
