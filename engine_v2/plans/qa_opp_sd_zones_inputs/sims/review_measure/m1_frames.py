"""Review measurement: fetch the H1 + M15 frames exactly like the engine, then check mappings + fallback extremes.
Writes nothing to the repo; pickles frames into review_measure/."""
import sys, os, pickle
import pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
from engine_v2.multitf.data_bridge import fetch_lower_tf_data, prepare_lower_tf_data
from engine_v2.multitf.entity_df_mutation import _map_parent_idx_to_m15_hour_end, _window_extreme_idx

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
print("H1 rows", len(df), "first", df['time'].iloc[0], "last", df['time'].iloc[-1], "index0", df.index[0])
h1_start = pd.to_datetime(df["time"].iloc[0], utc=True); h1_end = pd.to_datetime(df["time"].iloc[-1], utc=True)
m15 = prepare_lower_tf_data(fetch_lower_tf_data(CONFIG.pair, "M15", h1_start, h1_end))
print("M15 rows", len(m15), "first", m15['time'].iloc[0], "last", m15['time'].iloc[-1], "index0", m15.index[0], m15.index[-1])
pickle.dump((df[['time','o','h','l','c']].copy(), m15[['time','o','h','l','c']].copy()), open(os.path.join(OUT, "frames.pkl"), "wb"))

# compare H1 to S_Q CSV
S = "artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/NZD_USD_H1_2025-11-15_2026-01-20_final.csv"
sq = pd.read_csv(S)
same = (len(sq) == len(df)) and all((sq[c].round(5).values == df[c].astype(float).round(5).values).all() for c in "ohlc")
print("H1 OHLC == S_Q final.csv:", same)

# H1 -> M15 mapping: 4h..4h+3 ?
bad = []
for h in range(len(df)):
    loh = _map_parent_idx_to_m15_hour_end(h, df, m15)
    if loh != 4 * h + 3:
        bad.append((h, loh))
print("LOH(h) != 4h+3 count:", len(bad), bad[:10])
for h in (96, 748, 754, 771, 772, 820, 872, 902, 935, 946, 947, 1044, 1045, 1046, 1047, 1048, 1049, 1057):
    t = pd.to_datetime(df.loc[h, 'time'], utc=True)
    ms = m15.index[(pd.to_datetime(m15['time'], utc=True) >= t) & (pd.to_datetime(m15['time'], utc=True) < t + pd.Timedelta(hours=1))].tolist()
    print(f"H1 {h} {t} M15 {ms} LOH {_map_parent_idx_to_m15_hour_end(h, df, m15)}  H1 o/h/l/c {df.loc[h,'o']} {df.loc[h,'h']} {df.loc[h,'l']} {df.loc[h,'c']}")

# fallback extremes for (1,2) first_counter: hi = LOH(935); lowest low (lower_sd = +1 -> -lower_sd = -1)
hi = _map_parent_idx_to_m15_hour_end(935, df, m15)
for lo in (0, 3611):
    i = _window_extreme_idx(m15, lo, hi, -1)
    print(f"fallback lo={lo} hi={hi}: M15 {i} time {m15.loc[i,'time']} low {m15.loc[i,'l']} H1 of it {i//4}")
i0 = _window_extreme_idx(m15, 0, hi, -1); i1 = _window_extreme_idx(m15, 3611, hi, -1)
print("pips between", round((float(m15.loc[i1,'l']) - float(m15.loc[i0,'l'])) / 0.0001, 1))
