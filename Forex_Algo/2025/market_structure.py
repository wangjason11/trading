import pandas as pd
import pandas_ta as ta
import numpy as np
import math
import matplotlib.pyplot as plt
import optuna
import time
import tpqoa
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from itertools import product
from candles import Candle
from candles import CandlePatterns


import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


class IdentifyStart():
    def compute_layer_swings(df, order, threshold):
        highs = df["h"].values
        lows = df["l"].values
        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        low_idx = argrelextrema(lows, np.less, order=order)[0]
        swings = pd.Series(index=df.index, dtype=object)

        for idx in high_idx:
            if idx == 0 or idx >= len(df): continue
            if (highs[idx] - lows[max(0, idx - order)]) >= threshold:
                swings.iloc[idx] = "high"

        for idx in low_idx:
            if idx == 0 or idx >= len(df): continue
            if (highs[max(0, idx - order)] - lows[idx]) >= threshold:
                swings.iloc[idx] = "low"

        return swings


    def apply_multi_layer_swing_detection(df, config):
        df = df.copy()
        l1_order, l1_thresh = config[0]
        l2_order, l2_thresh = config[1]
        l3_order, l3_thresh = config[2]

        # Layer 1
        l1_swings = compute_layer_swings(df, l1_order, l1_thresh)
        df["layer1_swing"] = l1_swings
        df["layer1_price"] = df["h"].where(l1_swings == "high").fillna(df["l"].where(l1_swings == "low"))

        # Layer 2
        df_l1 = df[df["layer1_swing"].notna()].copy()
        l2_swings = compute_layer_swings(df_l1, l2_order, l2_thresh)
        df["layer2_swing"] = l2_swings.reindex(df.index)
        df["layer2_price"] = df["h"].where(df["layer2_swing"] == "high").fillna(df["l"].where(df["layer2_swing"] == "low"))

        # Layer 3
        df_l2 = df[df["layer2_swing"].notna()].copy()
        l3_swings = compute_layer_swings(df_l2, l3_order, l3_thresh)
        df["layer3_swing"] = l3_swings.reindex(df.index)
        df["layer3_price"] = df["h"].where(df["layer3_swing"] == "high").fillna(df["l"].where(df["layer3_swing"] == "low"))

        return df


    def find_algo_start_final_fixed(df, swing_col="layer3_swing", price_col="layer3_price"):
        swings = df[df[swing_col].notna()].copy()
        swings_rev = swings[::-1]  # Reverse chronological

        seen = []
        highs = 0
        lows = 0

        for idx, row in swings_rev.iterrows():
            swing_type = row[swing_col]
            seen.append((idx, swing_type))

            if swing_type == "high":
                highs += 1
            elif swing_type == "low":
                lows += 1

            if highs >= 2 and lows >= 2 and len(seen) >= 5:
                break

        if len(seen) < 4:
            return None

        candidate_idx, candidate_type = seen[3]
        candidate_price = df.loc[candidate_idx, price_col]

        prev_idx, prev_type = seen[4]
        if prev_type == candidate_type:
            prev_price = df.loc[prev_idx, price_col]
            if candidate_type == "high" and prev_price > candidate_price:
                return prev_idx
            if candidate_type == "low" and prev_price < candidate_price:
                return prev_idx

        return candidate_idx


    BEST_NONZERO_CONFIG = ((3, 0.00003), (3, 0.00003), (2, 0.00003))
    df_processed = apply_multi_layer_swing_detection(df, BEST_NONZERO_CONFIG)
    start_index = find_algo_start_final_fixed(df_processed)


class MarketStructure():
    def breakout_pullback(self, starting_direction):
        price_range = {"breakout": None, "pullback": None}
        historical_BOS = []
        hisotrical_CTS = []
        pre_breakout_cts = []
        skip = []
        trend_direction = starting_direction

        df = self.df
        df.bopb = None
        df.pulse_new_high = None
        df.pullback_new_low = None
        df.wave_direction = starting_direction
        df.market_cycle_direction = starting_direction

        for idx, row in df.iterrows():
            if idx == 0:
                historical_BOS.append((idx, row.time, row.l)) if starting_direction == 1 else historical_BOS.append((idx, row.time, row.h))
            if idx in skip:
                continue
            if trend_direction != starting_direction:
                row.bopb == "reversal"
                break
            if row.bopb == "breakout":
                self.log_historical_boscts(self, idx, starting_direction, pre_breakout_cts[-1]) if pre_breakout_cts != [] and idx - pre_breakout_cts[-1][0] in range(1,5) 
                if df.loc[idx-1,"candle_type"].str.contains("pinbar") and df.loc[idx-1,"is_range"] == 1:
                    price_range = self.set_range(idx-1, starting_direction)
                    breakout_pattern = self.check_pattern(idx, starting_direction, price_range["breakout"])
                    if breakout_pattern == True:
                        continue
                    else:
                        price_range = self.update_range(starting_direction=starting_direction, high=row.h, low=row.l, idx=idx-1, price_range=None)
                        continue
                elif row.direction != starting_direction:
                    price_range = self.set_range(idx, starting_direction)
                    pullback_pattern = self.check_pattern(idx, -starting_direction, price_range["pullback"])
                    if pullback_pattern == True:
                        continue
                    elif row.is_range == 1:
                        price_range = self.set_range(idx, starting_direction)
                        continue
                    continue
                elif row.candle_type.str.contains("pinbar") and row.is_range == 1:
                    price_range = self.set_range(idx, starting_direction)
                    continue
            elif row.bopb == "pullback":
                trend_direction = self.check_reversal(idx, -starting_direction, [historical_BOS[-1][2], price_range["pullback"]])
                if trend_direction != starting_direction:
                    continue
                price_range = self.update_range(starting_direction=starting_direction, high=row.h, low=row.l, idx=None, price_range=price_range)
                continue
            breakout_pattern = self.check_pattern(idx, starting_direction, price_range["breakout"])
            if breakout_pattern == True:
                continue
            pullback_pattern = self.check_pattern(idx, -starting_direction, price_range["pullback"])
            if pullback_pattern == True:
                continue
            elif df.loc[idx-1,"bopb"] == "breakout":
                if price_range == {"breakout": None, "pullback": None}:
                    if ~row.candle_type.str.contains("pinbar") and row.big_normal == 1 and row.c > df.loc[idx-1,"h"] if starting_direction == 1 else row.c < df.loc[idx-1,"l"]:
                        df.loc[idx,"bopb"] == "breakout"
                        if (self.df.iloc[idx].h > self.historical_CTS[-1][2] if starting_direction == 1 else self.df.iloc[idx].l < self.historical_CTS[-1][2]):
                            self.historical_CTS[-1] = (idx, df.time, df.h if direction == 1 else df.l) 
                        continue
                    elif row.is_range != 1:
                        df.loc[idx,"bopb"] = "breakout"
                        if (self.df.iloc[idx].h > self.historical_CTS[-1][2] if starting_direction == 1 else self.df.iloc[idx].l < self.historical_CTS[-1][2]):
                            self.historical_CTS[-1] = (idx, df.time, df.h if direction == 1 else df.l) 
                        continue                        
            elif df.loc[idx-1,"bopb"] == "pullback":
                if row.h > price_range["breakout"] if starting_direction == 1 else row.l < price_range["breakout"]:
                    breakout_pattern = self.check_pattern(idx, starting_direction, price_range["breakout"])
                    if breakout_pattern == True:
                        continue
                    else:
                        row.bopb = None
                        price_range = self.update_range(starting_direction=starting_direction, high=row.h, low=row.l, idx=None, price_range=price_range)
                        self.log_historical_boscts(self, idx, starting_direction)
                        continue
                else: 
                    row.bopb = "pullback"
                    price_range = self.update_range(starting_direction=starting_direction, high=row.h, low=row.l, idx=None, price_range=price_range)
                    trend_direction = self.check_reversal(idx, -starting_direction, [historical_BOS[-1][2], price_range["pullback"]])
                    if trend_direction != starting_direction:
                        continue
                    continue
            self.log_historical_boscts(self, idx, starting_direction)
            if price_range != {"breakout": None, "pullback": None}:
                price_range = self.update_range(starting_direction=starting_direction, high=row.h, low=row.l, idx=None, price_range=price_range)
                trend_direction = self.check_reversal(idx, -starting_direction, [historical_BOS[-1][2], price_range["pullback"]])
                if trend_direction != starting_direction:
                    continue
                else:
                    if self.historical_CTS[-2][0] < self.historical_BOS[-1][0] < self.historical_CTS[-1][0]:
                        continue
                    else:
                        farthest_pullback = self.get_farthest_pullback(self.historical_CTS[-2][0], self.historical_CTS[-1][0])
                        if farthest_pullback:
                            row.bopb = "pullback" if (row.l < farthest_pullback[2] if starting_direction == 1 else row.h > farthest_pullback[2]) else None
            else:
                if df.loc[idx-1,"candle_type"].str.contains("pinbar") and df.loc[idx-1,"is_range"] == 1:
                    price_range = self.set_range(idx-1, starting_direction)
                    price_range = self.update_range(starting_direction=starting_direction, high=row.h, low=row.l, idx=None, price_range=price_range)
                elif row.direction != starting_direction and row.is_range == 1:
                    price_range = self.set_range(idx, starting_direction)
                    pullback_pattern = self.check_pattern(idx, -starting_direction)
                    if pullback_patern == True:
                        continue
                    continue
                elif row.candle_type.str.contains("pinbar") and row.is_range == 1:
                    price_range = self.set_range(idx, starting_direction)
                    continue

    def check_reversal(self, idx, starting_direction, break_thresholds):
        bos = max(break_thresholds) if starting_direction == 1 else min(break_thresholds)
        reversal_pattern = self.check_pattern(idx, -starting_direction, bos)
        if reversal_pattern == True:
            return -starting_direction
        else:
            return starting_direction



    def log_historical_boscts(self, idx, starting_direction, cts=None):
        row = self.df.iloc[idx]
        bopb = row.bopb
        time = row.time
        high = row.h
        low = row.l

        def get_last_cts_idx():
            if not self.historical_CTS:
                return 0
            return self.historical_CTS[-1][0]

        def get_farthest_pullback(start_idx, end_idx):
            pullbacks = [(i, self.df.iloc[i].time, self.df.iloc[i].l if starting_direction == 1 else self.df.iloc[i].h)
                    for i in range(start_idx, end_idx)]
            if "pullback" in [self.df.iloc[i].bopb for i, _, _, in pullbacks]:
                return (min(pullbacks, key=lambda x: x[2]) if starting_direction == 1 else max(pullbacks, key=lambda x: x[2])) if pullbacks else None
            else:
                return None

        start_idx = get_last_cts_idx()
        farthest_pullback = get_farthest_pullback(start_idx, idx)

        if bopb == "breakout" and cts is not None:
            # Log BOS
            if farthest_pullback:
                self.historical_BOS.append(farthest_pullback)

            # Log CTS
            new_potential_cts = ((idx, time, high) if high > cts[2] else cts) if starting_direction == 1 else ((idx, time, low) if low < cts[2] else cts)
            if not self.historical_CTS:
                self.historical_CTS.append(new_potential_cts)
            else:
                if farthest_pullback:
                    _, _, last_cts = self.historical_CTS[-1]
                    if starting_direction == 1:
                        if new_potential_cts[2] > last_cts:
                            self.historical_CTS.append(new_potential_cts)
                    else:
                        if new_potential_cts[2] < last_cts:
                            self.historical_CTS.append(new_potential_cts)
                else:
                    self.historical_CTS[-1] = new_potential_cts
                    if self.historical_CTS[-2][0] < self.historical_BOS[-1][0] < self.historical_CTS[-1][0]:
                        pass
                    else:
                        farthest_pullback = get_farthest_pullback(self.historical_CTS[-2][0], self.historical_CTS[-1][0])
                        if farthest_pullback:
                            self.historical_BOS.append(farthest_pullback)

        else:
            if self.historical_CTS and (self.df.iloc[idx].h > self.historical_CTS[-1][2] if starting_direction == 1 else self.df.iloc[idx].l < self.historical_CTS[-1][2]):
                if farthest_pullback == None:
                    self.historical_CTS[-1] = (idx, row.time, self.df.iloc[idx].h) if starting_direction == 1 else (idx, row.time, self.df.iloc[idx].l)
                else:
                    self.historical_CTS.append((idx, row.time, self.df.iloc[idx].h) if starting_direction == 1 else (idx, row.time, self.df.iloc[idx].l))


    def set_range(self, idx, starting_direction):
       return {
            "breakout": df.loc[idx,"h"], 
            "pullback": df.loc[idx,"l"]
        } if starting_direction == 1 else {
            "breakout": df.loc[idx,"l"], 
            "pullback": df.loc[idx,"h"]
        }

    def update_range(self, starting_direction, high, low, idx=None, price_range=None):
        if price_range is not None:
            return {
                "breakout": max(price_range["breakout"], high), 
                "pullback": min(price_range["pullback"], low)
            } if starting_direction == 1 else {
                "breakout": min(price_range["breakout"], low), 
                "pullback": max(price_range["pullback"], high)
            }
        if idx is not None:
            return {
                "breakout": max(df.loc[idx-1,"h"], high), 
                "pullback": min(df.loc[idx-1,"l"], low)
            } if starting_direction == 1 else {
                "breakout": min(df.loc[idx-1,"l"], low), 
                "pullback": max(df.loc[idx-1,"h"], high)
            }








    def success_pattern_processing(self, future_idx, direction, starting_direction):
        df.loc[future_idx,"bopb"] == "breakout" if direction == starting_direction else df.loc[future_idx,"bopb"] == "pullback"
        skipped_idx = range(idx+1, future_idx)
        if list(skipped_idx) == []:
            high = (idx, df.loc[idx, 'time'], df.loc[idx, 'h'])
            low = (idx, df.loc[idx, 'time'], df.loc[idx, 'l'])
        else:
            highest_idx = max(skipped_idx, key=lambda i: df.loc[i, 'h'])
            lowest_idx = min(skipped_idx, key=lambda i: df.loc[i, 'l'])
            high = (highest_idx, df.loc[highest_idx, 'time'], df.loc[highest_idx, 'h'])
            low = (lowest_idx, df.loc[lowest_idx, 'time'], df.loc[lowest_idx, 'l'])
            # high = max([df.loc[i,"h"] for i in skipped_idx])
            # low = min([df.loc[i,"l"] for i in skipped_idx])
        skip.extend(skipped_idx)
        if df.loc[future_idx,"bopb"] == "breakout":
            price_range = {"breakout": None, "pullback": None}
            pre_breakout_cts.append(high if starting_direction == 1 else low)
        elif price_range == {"breakout": None, "pullback": None}:
            price_range = self.set_range(idx, starting_direction)
        else:
            price_range = self.update_range(starting_direction=starting_direction, high=high, low=low, idx=None, price_range=price_range)


                

    def check_pattern(self, idx, direction, break_threshold=None):
        direction, status = CandlePatterns.continuous(idx, direction, break_threshold)
        if status == "success":
            self.success_pattern_processing(idx+2, direction)
            return True
        status, confirmation_threshold = CandlePatterns.double_maru(idx, direction, break_threshold)
        if status == "success":
            self.success_pattern_processing(idx+1, direction)
            return True
        elif status == "fail":
            status, future_idx = CandlePatterns.price_confirmation(idx+1, direction, confirmation_threshold)
            if status == True:
                self.success_pattern_processing(future_idx, direction)
                return True
        status, confirmation_threshold = CandlePatterns.one_maru_continuous(idx, direction, break_threshold)
        if status == "success":
            self.success_pattern_processing(idx+1, direction)
            return True
        elif status == "fail":
            status, future_idx = CandlePatterns.price_confirmation(idx+1, direction, confirmation_threshold)
            if status == True:
                self.success_pattern_processing(future_idx, direction)
                return True
        status, confirmation_threshold = CandlePatterns.one_maru_opposite(idx, direction, break_threshold)
        if status == "success":
            self.success_pattern_processing(idx+1, direction)
            return True
        elif status == "fail":
            status, future_idx = CandlePatterns.price_confirmation(idx+1, direction, confirmation_threshold)
            if status == True:
                self.success_pattern_processing(future_idx, direction)
                return True
        return False






Class 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score
from itertools import product

# Load Excel data
xls = pd.ExcelFile("/mnt/data/H4_pairs.xlsx")
pairs = xls.sheet_names

# Load and format each sheet
dfs_new_raw = {}
for pair in pairs:
    df = xls.parse(pair)
    df.columns = [col.strip().lower() for col in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    dfs_new_raw[pair] = df

# Layered swing detection using extrema
from scipy.signal import argrelextrema

class MarketStructure():
    def compute_layer_swings(df, order, threshold):
        highs = df["h"].values
        lows = df["l"].values
        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        low_idx = argrelextrema(lows, np.less, order=order)[0]

        swings = pd.Series(index=df.index, dtype=object)
        for idx in high_idx:
            if idx == 0 or idx >= len(df): continue
            if (highs[idx] - lows[max(0, idx - order)]) >= threshold:
                swings.iloc[idx] = "high"
        for idx in low_idx:
            if idx == 0 or idx >= len(df): continue
            if (highs[max(0, idx - order)] - lows[idx]) >= threshold:
                swings.iloc[idx] = "low"

        return swings

    def apply_multi_layer_swing_detection(df, config):
        df = df.copy()
        # Layer 1
        l1_order, l1_thresh = config[0]
        l1_swings = compute_layer_swings(df, l1_order, l1_thresh)
        df["layer1_swing"] = l1_swings
        df["layer1_price"] = df["h"].where(l1_swings == "high").fillna(df["l"].where(l1_swings == "low"))
        # Layer 2
        l2_order, l2_thresh = config[1]
        df_l1 = df[df["layer1_swing"].notna()].copy()
        l2_swings = compute_layer_swings(df_l1, l2_order, l2_thresh)
        df["layer2_swing"] = l2_swings.reindex(df.index)
        df["layer2_price"] = df["h"].where(df["layer2_swing"] == "high").fillna(df["l"].where(df["layer2_swing"] == "low"))
        # Layer 3
        l3_order, l3_thresh = config[2]
        df_l2 = df[df["layer2_swing"].notna()].copy()
        l3_swings = compute_layer_swings(df_l2, l3_order, l3_thresh)
        df["layer3_swing"] = l3_swings.reindex(df.index)
        df["layer3_price"] = df["h"].where(df["layer3_swing"] == "high").fillna(df["l"].where(df["layer3_swing"] == "low"))
        return df

def evaluate_multilayer_auc(config):
    aucs = []
    for pair in pairs:
        df = apply_multi_layer_swing_detection(dfs_new_raw[pair], config)
        true_labels = dfs_new_raw[pair]["swing"].map({"high": 1, "low": 1}).fillna(0).astype(int)
        pred_labels = df["layer3_swing"].notna().astype(int)
        if true_labels.sum() > 0:
            try:
                auc = roc_auc_score(true_labels, pred_labels)
                aucs.append(auc)
            except:
                continue
    return np.mean(aucs)

# Optimized config (non-zero thresholds)
best_nonzero_config = ((3, 0.00003), (3, 0.00003), (2, 0.00003))

# Plotting function using index-based x-axis
def plot_candles_with_swings_custom(df, title: str, layer: str = "layer3"):
    fig, ax = plt.subplots(figsize=(14, 6))
    candle_width = 0.6
    prices = df[["o", "h", "l", "c"]].values
    for i, (o, h, l, c) in enumerate(prices):
        color = "green" if c >= o else "red"
        ax.plot([i, i], [l, h], color="black", linewidth=1)
        rect = Rectangle(
            (i - candle_width / 2, min(o, c)),
            candle_width,
            abs(c - o),
            facecolor=color,
            edgecolor="black"
        )
        ax.add_patch(rect)
    swing_col = f"{layer}_swing"
    for idx, row in df.iterrows():
        i = df.index.get_loc(idx)
        if row[swing_col] == "high":
            ax.plot(i, row["h"], marker="v", color="green", markersize=10)
        elif row[swing_col] == "low":
            ax.plot(i, row["l"], marker="^", color="green", markersize=10)
    ax.set_title(f"{title} - Layer 3 Swings (ROC-AUC Optimal)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Price")
    ax.set_xlim(-1, len(df) + 1)
    plt.tight_layout()
    plt.show()

# Apply and plot for each pair
dfs_best_nonzero = {}
for pair in pairs:
    df = apply_multi_layer_swing_detection(dfs_new_raw[pair], best_nonzero_config)
    dfs_best_nonzero[pair] = df
    plot_candles_with_swings_custom(df, title=pair, layer="layer3")
