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