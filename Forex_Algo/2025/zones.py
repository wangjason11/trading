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


class KeyLevel():
    self.buy_zone = []
    self.sell_zone = []

    def check_base_pattern(self, idx, starting_direction, bos=True):
        if bos == True:
            if starting_direction == df.loc[idx,"direction"]:
                return (idx-1, df.loc[idx-1,"base_pattern"])
            else:
                return (idx, df.loc[idx,"base_pattern"])
        else:
            if starting_direction == df.loc[idx,"direction"]:
                return (idx, df.loc[idx,"base_pattern"])
            else:
                return (idx-1, df.loc[idx-1,"base_pattern"])


    def zone_thresholds(self, base_idx, starting_direction, zone_pattern, bos=True):
        if bos == True:
            if zone_pattern == "no base 2nd big":
                return (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    df.loc[base_idx+1, "mid_price"]
                    ) if starting_direction == 1 else (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    df.loc[base_idx+1, "mid_price"]
                    )
            elif zone_pattern == "no base 1st big":
                return (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    df.loc[base_idx+1, "c"]
                    ) if starting_direction == 1 else (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    df.loc[base_idx+1, "c"]
                    )
            elif zone_pattern == "no base":
                return (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    df.loc[base_idx, "o"]
                    ) if starting_direction == 1 else (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    df.loc[base_idx, "o"]
                    )
            elif zone_pattern == "up pinbar" and direction == 1:
                return (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    df.loc[base_idx, "base_min_close_open"]
                    )
            elif zone_pattern == "down pinbar" and direction == -1:
                return (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    df.loc[base_idx, "base_max_close_open"]
                    )
            else:
                return (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    self.find_base_threshold(df, base_idx, starting_direction, bos)
                    ) if starting_direction == 1 else (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    self.find_base_threshold(df, base_idx, starting_direction, bos)
                    )
        else:
            if zone_pattern == "no base 2nd big":
                return (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    df.loc[base_idx+1, "mid_price"]
                    ) if starting_direction == 1 else (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    df.loc[base_idx+1, "mid_price"]
                    )
            elif zone_pattern == "no base 1st big":
                return (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    df.loc[base_idx+1, "c"]
                    ) if starting_direction == 1 else (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    df.loc[base_idx+1, "c"]
                    )
            elif zone_pattern == "no base":
                return (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    df.loc[base_idx, "o"]
                    ) if starting_direction == 1 else (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    df.loc[base_idx, "o"]
                    )
            elif zone_pattern == "up pinbar" and direction == -1:
                return (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    df.loc[base_idx, "base_min_close_open"]
                    )
            elif zone_pattern == "down pinbar" and direction == 1:
                return (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    df.loc[base_idx, "base_max_close_open"]
                    )
            else:
                return (
                    base_idx,
                    df.loc[base_idx, "base_high"], 
                    self.find_base_threshold(df, base_idx, starting_direction, bos)
                    ) if starting_direction == 1 else (
                    base_idx,
                    df.loc[base_idx, "base_low"], 
                    self.find_base_threshold(df, base_idx, starting_direction, bos)
                    )


    def find_base_threshold(self, df, idx, starting_direction, bos=True):
        left = max(0, idx - 5)
        right = min(len(df), idx + 6)

        # Exclude idx itself
        neighbor_df = df.iloc[left:idx].copy()
        neighbor_df = pd.concat([neighbor_df, df.iloc[idx+1:right]])

        # Get minimum of open and close as potential levels
        candidates_desc = sorted(set(np.minimum(neighbor_df['o'], neighbor_df['c'])), reverse=True)
        candidates_asc = sorted(set(np.maximum(neighbor_df['o'], neighbor_df['c'])), reverse=False)

        if bos == True:
            for level in candidates_asc if starting_direction == 1 else candidates_desc:
                count = ((neighbor_df['o'] <= level) & (neighbor_df['c'] <= level)).sum() if starting_direction == 1 else ((neighbor_df['o'] >= level) & (neighbor_df['c'] >= level)).sum()
                if count >= 2:
                    result = level
                if count >= 3:
                    result = level
                    return result
                return result if result
        else:
            for level in candidates_desc if starting_direction == 1 else candidates_asc:
                count = ((neighbor_df['o'] >= level) & (neighbor_df['c'] >= level)).sum() if starting_direction == 1 else ((neighbor_df['o'] <= level) & (neighbor_df['c'] <=3 level)).sum()
                if count >= 2:
                    result = level
                if count >= 3:
                    result = level
                    return result
                return result if result


    def build_zones(self, starting_direction, bopb):
        buy_zone = []
        sell_zone = []

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

        bos_idx, bos_time, bos_price = historical_BOS[-1]
        cts_idx, cts_time, cts_price = historical_CTS[-1]
        is_bullish = starting_direction == 1
        bos_zone = self.buy_zone if is_bullish else self.sell_zone
        cts_zone = self.sell_zone if is_bullish else self.buy_zone

        if bopb == "breakout":
            base_idx, zone_pattern = self.check_base_pattern(bos_idx, starting_direction, bos=True)
            zone = self.zone_thresholds(base_idx, starting_direction, zone_pattern, bos=True)

            if not bos_zone:
                bos_zone.append(zone + (idx, idx, True))
            else:
                start_idx = get_last_cts_idx()
                if get_farthest_pullback(start_idx, bos_idx):
                    bos_zone[-1][4] = idx
                    bos_zone[-1][5] = False
                    bos_zone.append(zone + (idx, idx, True))

            if cts_zone:
                cts_zone[-1][4] = idx
                cts_zone[-1][5] = False

        elif bopb == "pullback":
            base_idx, zone_pattern = self.check_base_pattern(cts_idx, starting_direction, bos=False)
            zone = self.zone_thresholds(base_idx, starting_direction, zone_pattern, bos=False)

            if not cts_zone:
                cts_zone[-1][4] = idx
                cts_zone[5] = False
                cts_zone.append(zone + (idx, idx, True))
            elif cts_zone[-1][0] != cts_idx:
                cts_zone[-1][4] = idx
                cts_zone[5] = False
                cts_zone.append(zone + (idx, idx, True))

        else:
            if bos_price * is_bullish < bos_zone[-1][1] * is_bullish:
                bos_zone[-1][1] = bos_price
                bos_zone[-1][4] = idx
            elif cts_price * is_bullish > cts_zone[-1][1] * is_bullish:
                cts_zone[-1][1] = cts_price
                cts_zone[-1][4] = idx

        self.buy_zone = bos_zone if is_bullish else cts_zone
        self.sell_zone = cts_zone if is_bullish else bos_zone