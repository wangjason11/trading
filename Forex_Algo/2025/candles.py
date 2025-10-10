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


class Candle:
    def attributes(self):
        df.body_length = (df.c - df.o).abs()
        df.candle_length = df.h - df.l
        df.direction = 1 if df.c > df.o else (-1 if df.c < df.o else 0)
        df.mid_price = (df.h + df.l) / 2

    def classification(self, df, pinbar, pinbar_distance, maru, special_maru, special_maru_distance):

        df.candle_type = "normal"

        # 1. Pinbar: body fully within top x% or bottom x%
        is_pinbar = df.body_length < pinbar * df.candle_length 

        df.loc[is_pinbar, "candle_type"] = "pinbar"

        is_pinbar_up = df[["open", "close"]].min(axis=1) > df.l + (1 - pinbar_distance) * df.candle_length # body in top x%
        is_pinbar_down = df[["open", "close"]].max(axis=1) < df.l + pinbar_distance * df.candle_length # body in bottom x%

        df.loc[(is_pinbar_up & is_pinbar), "candle_type"] = "up pinbar"
        df.loc[(is_pinbar_down & is_pinbar), "candle_type"] = "down pinbar"

        # 2. Marubuzo: body >= 70% of full candle and not already classified
        is_maru = df.body_length >= maru * df.candle_length
        df.loc[is_maru, "candle_type"] = "maru"

        # 3. Special Marubuzo: body >= 50% and candle near top or bottom 10%
        is_up_maru = (
            (df.direction == 1) &
            (df.body_length >= special_maru * df.candle_length) &
            (df.c >= df.l + (1 - special_maru_distance)* df.candle_length)
        )
        df.loc[is_up_maru, "candle_type"] = "maru"

        is_down_maru = (
            (df.direction == -1) &
            (df.body_length >= special_maru * df.candle_length) &
            (df.c <= df.l + special_maru_distance * df.candle_length)
        )
        df.loc[is_down_maru, "candle_type"] = "maru"

    def secondary_classification(self, df, lookback, maru_threshold, normal_threshold):
        """
        Adds:
          - "big_maru": 1 if maru candle is >= 70% of the largest of last 5 maru candles
          - "big_normal": 1 if normal or maru candle is >= 50% of the largest of last 5 maru candles
        Requires: `classification()` to be run first
        """

        # --- Safety check ---
        required_cols = ["candle_type", "candle_length"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Run classification() first.")

        # --- Init output columns ---
        df.big_maru = 0
        df.big_normal = 0
        df.big_normal_1 = 0
        df.big_normal_2 = 0

        # --- Define type groups ---
        maru_types = ["maru"]
        target_types = ["maru", "normal", "up pinbar", "down pinbar", "pinbar"]

        # --- Get maru candle indices for lookback ---
        maru_idx = df[df["candle_type"].isin(maru_types)].index

        # for i in range(len(maru_idx)):
        #     current_idx = maru_idx[i]
        #     current_len = df.loc[current_idx, "candle_length"]
            
        #     # Look back up to x previous maru candles
        #     prior_idxs = maru_idx[max(i - lookback, 0):i]
        #     if len(prior_idxs) == 0:
        #         continue  # skip if no prior marus

        #     max_prior_len = df.loc[prior_idxs, "candle_length"].max()

        #     if current_len >= maru_threshold * max_prior_len:
        #         df.at[current_idx, "big_maru"] = 1

        for i in range(len(df)):
            if df.at[i, "candle_type"] not in target_types:
                continue

            # 1. big_normal (uses marus before i)
            prior_maru = maru_idx[maru_idx < i][-lookback:]
            if len(prior_maru) > 0:
                max_len = df.loc[prior_maru, "candle_length"].max()
                if df.at[i, "candle_length"] >= maru_threshold * max_len:
                    df.at[i, "big_maru"] = 1
                if df.at[i, "candle_length"] >= normal_threshold * max_len:
                    df.at[i, "big_normal"] = 1

            # 2. big_normal_1 (uses marus before i-1)
            if i - 1 >= 0:
                prior_maru_2 = maru_idx[maru_idx < i - 1][-lookback:]
                if len(prior_maru_2) > 0:
                    max_len_2 = df.loc[prior_maru_2, "candle_length"].max()
                    if df.at[i, "candle_length"] >= normal_threshold * max_len_2:
                        df.at[i, "big_normal_1"] = 1

            # 3. big_normal_2 (uses marus before i-2)
            if i - 2 >= 0:
                prior_maru_3 = maru_idx[maru_idx < i - 2][-lookback:]
                if len(prior_maru_3) > 0:
                    max_len_3 = df.loc[prior_maru_3, "candle_length"].max()
                    if df.at[i, "candle_length"] >= normal_threshold * max_len_3:
                        df.at[i, "big_normal_2"] = 1

    def range(self, start, end):
        df = self.df
        df["is_range"] = 0

        for i in range(len(df) - start):  # make sure we have at least 2 candles ahead
            c0 = df.iloc[i]
            high = c0.h
            low = c0.l

            for j in range(start, end+1):  # future candles 2 through 5
                if i + j >= len(df):
                    break
                fwd = df.iloc[i + j]
                if low <= fwd.c <= high:
                    df.at[i, "is_range"] = 1
                    break  # only need one match


class CandlePatterns:
    def __init__(self, df):
        self.df = df.copy()

    def confirmation_threshold(self, i, direction):
        c0 = self.df.iloc[i]
        c1 = self.df.iloc[i + 1]
        return max(c0.h, c1.h) if direction == 1 else min(c0.l, c1.l)

    def check_break(self, candle, threshold, direction):
        if threshold is None:
            return True
        return candle.c > threshold if direction == 1 else candle.c < threshold

    def body_check(self, candle, threshold, percent=0.5, direction=1):
        if threshold is None:
            return True
        multiplier = 1 + percent if direction == 1 else 1 - percent
        return candle.body_length >= threshold * multiplier

    def continuous(self, idx, direction, break_threshold=None):
        if idx + 2 >= len(self.df):
            return None
        df = self.df
        c0, c1, c2 = df.iloc[idx], df.iloc[idx + 1], df.iloc[idx + 2]
        if all(c.direction == direction for c in [c0, c1, c2]):
            if not self.check_break(c0, break_threshold, direction):
                return None
            cond1 = (
                c0.candle_type in ["normal", "maru"] and
                c1.candle_type == "pinbar" and
                c2.candle_type == "maru" and
                c0.big_normal == 1 and
                c2.big_normal_2 == 1 and
                (c2.c > max(c0.h, c1.h) if direction == 1 else c2.c < min(c0.l, c1.l))
            )
            close_to_high = abs(c1.c - c0.h) <= 0.0010 if direction == 1 else abs(c1.c - c0.l) <= 0.0010
            cond2 = (
                c0.candle_type == "pinbar" and
                close_to_high and
                c2.candle_type == "maru" and
                c2.big_normal_2 == 1 and
                (c2.c > max(c0.h, c1.h) if direction == 1 else c2.c < min(c0.l, c1.l))
            )
            cond3 = (
                c0.candle_type == "normal" and c0.big_normal == 1 and
                c1.candle_type == "normal" and c1.big_normal == 1 and
                c2.candle_type == "maru" and c2.big_normal_2 == 1 and
                (c2.c > max(c0.h, c1.h) if direction == 1 else c2.c < min(c0.l, c1.l))
            )
            if cond1 or cond2 or cond3:
                return "success"
        return None

    def double_maru(self, idx, direction, break_threshold=None):
        if idx + 1 >= len(self.df):
            return None, None
        df = self.df
        c0 = df.iloc[idx]
        c1 = df.iloc[idx + 1]
        if c0.direction == direction and c1.direction == direction:
            c0_valid = c0.candle_type == "maru" and c0.big_normal == 1
            c1_valid = (
                c1.candle_type == "maru" and
                (c1.c > c0.c if direction == 1 else c1.c < c0.c) and
                c1.candle_length >= 0.7 * c0.candle_length
            )
            if not self.check_break(c0, break_threshold, direction):
                return None, None
            if c0_valid and c1_valid:
                return "success", self.confirmation_threshold(idx, direction)
            elif (c0_valid and not c1_valid) or (not c0_valid and c1_valid and c0.candle_type == "normal" and c0.big_maru == 1):
                return "fail", self.confirmation_threshold(idx, direction)
        return None, None

    def one_maru_continuous(self, idx, direction, break_threshold=None, break_percent=0.3, small_body_tail=0.5):
        if idx + 1 >= len(self.df):
            return None, None
        df = self.df
        c0 = df.iloc[idx]
        c1 = df.iloc[idx + 1]
        if c0.direction == direction and c1.direction == direction:
            c0_pass = (
                c0.candle_type == "maru" and
                c0.big_maru == 1 and
                self.body_check(c0, break_threshold, break_percent, direction)
            )
            c1_pass = c1.l > (c0.l + small_body_tail * c0.candle_length) if direction == 1 else c1.h < (c0.l + small_body_tail * c0.candle_length)
            if c0_pass and c1_pass:
                return "success", self.confirmation_threshold(idx, direction)
            elif c0_pass ^ c1_pass:
                if (
                    (not c0_pass and (
                        (c0.candle_type == "normal" and c0.big_maru == 1) or
                        (c0.candle_type == "maru" and c0.big_normal == 1))) or
                    (c0.candle_type == "maru" and c0.big_maru == 1 and not self.body_check(c0, break_threshold, break_percent, direction))
                ):
                    return "fail", self.confirmation_threshold(idx, direction)
        return None, None

    def one_maru_opposite(self, idx, direction, break_threshold=None, break_percent=0.3, small_body_tail=0.5, small_body_size=0.35):
        if idx + 1 >= len(self.df):
            return None, None
        df = self.df
        c0 = df.iloc[idx]
        c1 = df.iloc[idx + 1]
        if c0.direction == direction and c1.direction == -direction:
            c0_pass = (
                c0.candle_type == "maru" and
                c0.big_maru == 1 and
                self.body_check(c0, break_threshold, break_percent, direction)
            )
            c1_len_check = c1.candle_length < small_body_size * c0.candle_length
            c1_pos_check = c1.l > (c0.l + small_body_tail * c0.candle_length) if direction == 1 else c1.h < (c0.l + small_body_tail * c0.candle_length)
            c1_pass = c1_len_check and c1_pos_check
            if c0_pass and c1_pass:
                return "success", self.confirmation_threshold(idx, direction)
            elif c0_pass ^ c1_pass:
                if (
                    (not c0_pass and (
                        (c0.candle_type == "normal" and c0.big_maru == 1) or
                        (c0.candle_type == "maru" and c0.big_normal == 1))) or
                    (c0.candle_type == "maru" and c0.big_maru == 1 and not self.body_check(c0, break_threshold, break_percent, direction))
                ):
                    return "fail", self.confirmation_threshold(idx, direction)
        return None, None

    def price_confirmation(self, idx, direction, confirmation_threshold):
        df = self.df
        for j in range(1, 5):
            future_idx = idx + j
            if future_idx >= len(df):
                break
            fwd = df.iloc[future_idx]
            if fwd.direction != direction:
                continue
            if fwd.candle_type not in ["normal", "maru"]:
                continue
            if direction == 1 and fwd.c >= confirmation_threshold:
                return True, future_idx
            if direction == -1 and fwd.c <= confirmation_threshold:
                return True, future_idx
        return False, None


    def base_patterns(self, df, idx, length_threshold):
        df = self.df
        df.base_pattern = np.where((
                df.big_normal.shift(1) == 1 &
                df.big_normal_1 == 1 &
                df.direction.shift(1) != df.direction),
                np.where((
                    df.candle_type.shift(1) in ["maru", "normal"] &
                    df.big_normal.shift(1) == 1),
                    np.where(df.candle_length.shift(1) < length_threshold * df.candle_length, "no base 2nd big", 
                        np.where(df.candle_length.shift(1) > length_threshold * df.candle_length, "no base 1st big", "no base")
                        ), 
                    np.where((
                        df.candle_type.shift(1) == "up pinbar" & 
                        df.candle_type == "up pinbar"), 
                        "no base long tails up",
                        np.where((
                            df.candle_type.shift(1) == "up pinbar" & 
                            df.candle_type == "up pinbar"),
                            "no base long tails down", "base"
                            )
                        )
                    ), 
                "base"
            )
        df.base_low = np.where(df.base_pattern != "base", np.minimum(df.l, df.l.shift(-1)), None)
        df.base_high = np.where(df.base_pattern != "base", np.maximum(df.h, df.h.shift(-1)), None)
        df.base_min_close_open = np.where(df.base_pattern != "base", np.minimum(df.c, df.c.shift(-1), df.o, df.o.shift(-1)), None)
        df.base_max_close_open = np.where(df.base_pattern != "base", np.maximum(df.c, df.c.shift(-1), df.o, df.o.shift(-1)), None)





patterns = CandlePatterns(df)
patterns.continuous(break_threshold=1.1000)
patterns.double_maru(break_threshold=1.1000)
patterns.one_maru_continuous(break_threshold=0.0010, break_percent=0.3)
patterns.one_maru_opposite(break_threshold=0.0010, break_percent=0.3)

df_with_patterns = patterns.get_df()

        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"position"] = df.loc[row[0],"open"]
                if abs(df.loc[row[0],"open"]) == 1:
                    df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
            elif abs(df.loc[row[0]-1,"position"]) == 1:
                # df.loc[row[0],"recentPHL"] = df.loc[row[0], phl[df.loc[row[0]-1, "position"]]]
                df.loc[row[0],"HL"] = df.loc[row[0], hl[df.loc[row[0]-1, "position"]]]
                df.loc[row[0],"LH"] = df.loc[row[0], lh[df.loc[row[0]-1, "position"]]]
