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


class Candle():
    def attributes(self):
        df.body_length = (df.c - df.o).abs()
        df.candle_length = df.h - df.l
        df.direction = 1 if df.c > df.o else (-1 if df.c < df.o else 0)

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
        df.loc[is_up_maru, "candle_type"] = "up maru"

        is_down_maru = (
            (df.direction == -1) &
            (df.body_length >= special_maru * df.candle_length) &
            (df.c <= df.l + special_maru_distance * df.candle_length)
        )
        df.loc[is_down_maru, "candle_type"] = "down maru"

    def secondary_classification(self, df, lookback, maru_threshold, normal_threshold):
        """
        Adds "big_maru" column: for maru-type candles, checks if the candle is big compared to last 5 maru candles.
        """
        required_cols = ["candle_type", "candle_length"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Run classification() first.")
        
        # Initialize column
        df.big_maru = 0

        # Get indices of maru-like candles
        maru_types = ["maru", "up maru", "down maru"]
        maru_idx = df[df["candle_type"].isin(maru_types)].index

        for i in range(len(maru_idx)):
            current_idx = maru_idx[i]
            current_len = df.loc[current_idx, "candle_length"]
            
            # Look back up to x previous maru candles
            prior_idxs = maru_idx[max(i - lookback, 0):i]
            if len(prior_idxs) == 0:
                continue  # skip if no prior marus

            max_prior_len = df.loc[prior_idxs, "candle_length"].max()

            if current_len >= threshold * max_prior_len:
                df.at[current_idx, "big_maru"] = 1

        # Initialize new column
        df.big_normal = 0

        # Valid candle types for assigning big_normal
        normal_types = ["normal", "maru", "up maru", "down maru"]
        maru_types = ["maru", "up maru", "down maru"]
        
        # Get all maru candle indices
        # maru_idx = df[df["candle_type"].isin(maru_types)].index

        for i in range(len(df)):
            if df.at[i, "candle_type"] not in normal_types:
                continue

            # Find maru indices before current candle
            prior_maru = maru_idx[maru_idx < i]
            prior_idxs = prior_maru[-lookback:]  # get last 5

            if len(prior_idxs) == 0:
                continue

            max_prior_len = df.loc[prior_idxs, "candle_length"].max()
            if df.at[i, "candle_length"] >= normal_threshold * max_prior_len:
                df.at[i, "big_normal"] = 1







    def secondary_classification(self, df, pinbar, pinbar_distance, maru, special_maru, special_maru_distance):








class MarketStructure():
    def detect_recent_major_move_start(df, swing_window=5, trend_lookback=20):
        df = df.copy()
        
        # Step 1: Detect swing highs/lows
        df["swing_high"] = df["high"][(df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))]
        df["swing_low"] = df["low"][(df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))]

        # Optional: Make swings more robust by looking at N candles before/after
        def is_local_max(series):
            return series.iloc[len(series)//2] == series.max()

        def is_local_min(series):
            return series.iloc[len(series)//2] == series.min()

        df["swing_high"] = df["high"].rolling(window=swing_window, center=True).apply(
            lambda x: x[len(x)//2] if is_local_max(x) else np.nan, raw=False
        )

        df["swing_low"] = df["low"].rolling(window=swing_window, center=True).apply(
            lambda x: x[len(x)//2] if is_local_min(x) else np.nan, raw=False
        )

        # Step 2: Detect recent trend direction
        recent_close = df["close"].iloc[-trend_lookback:]
        if recent_close.iloc[-1] > recent_close.iloc[0]:
            trend = "up"
        else:
            trend = "down"

        # Step 3: Traverse backward to find most recent major low/high that started this trend
        if trend == "up":
            swing_points = df[df["swing_low"].notna()].iloc[::-1]  # reverse chronological
            for idx, row in swing_points.iterrows():
                # Has price moved significantly upward since this swing?
                if df["close"].iloc[-1] > row["low"] + 0.003:  # adjust threshold as needed
                    return idx, "low", row["low"]
        else:
            swing_points = df[df["swing_high"].notna()].iloc[::-1]
            for idx, row in swing_points.iterrows():
                if df["close"].iloc[-1] < row["high"] - 0.003:
                    return idx, "high", row["high"]
        
        return None, None, None  # fallback if no point found








import pandas as pd
import numpy as np

# 1. Calculate candle_length and body_length
df["candle_length"] = df["high"] - df["low"]
df["body_length"] = (df["close"] - df["open"]).abs()

# 2. Calculate 14-period moving average of body_length
df["avg_body_14"] = df["body_length"].rolling(window=14).mean()

# 3. Define position of the body within the candle (body_top and body_bottom)
df["body_top"] = df[["open", "close"]].max(axis=1)
df["body_bottom"] = df[["open", "close"]].min(axis=1)

# 4. Define thresholds for pinbar positioning
df["upper_40"] = df["low"] + 0.6 * df["candle_length"]
df["lower_40"] = df["low"] + 0.4 * df["candle_length"]

# 5. Initialize a column for candle type
df["candle_type"] = "unclassified"

# 6. Classification logic in order of priority

# Pinbar: body fully within top or bottom 40%
is_top_pinbar = df["body_bottom"] >= df["upper_40"]
is_bottom_pinbar = df["body_top"] <= df["lower_40"]
df.loc[is_top_pinbar | is_bottom_pinbar, "candle_type"] = "pinbar"

# Doji: body < 20% of avg body and not already pinbar
is_doji = (df["body_length"] < 0.2 * df["avg_body_14"]) & (df["candle_type"] == "unclassified")
df.loc[is_doji, "candle_type"] = "doji"

# Indecision: body < 40% of avg body and not already classified
is_indecision = (df["body_length"] < 0.4 * df["avg_body_14"]) & (df["candle_type"] == "unclassified")
df.loc[is_indecision, "candle_type"] = "indecision"

# Marubuzo: body ≥ 60% of full candle and not already classified
is_marubuzo = (df["body_length"] >= 0.6 * df["candle_length"]) & (df["candle_type"] == "unclassified")
df.loc[is_marubuzo, "candle_type"] = "marubuzo"

# Large: body > 60% of avg body and not already classified
is_large = (df["body_length"] > 0.6 * df["avg_body_14"]) & (df["candle_type"] == "unclassified")
df.loc[is_large, "candle_type"] = "large"

# Optional: clean up helper columns if needed
# df.drop(columns=["body_top", "body_bottom", "upper_40", "lower_40"], inplace=True)























        

class TrendAlgoBacktester(tpqoa.tpqoa):
    """ Class for the vectorized backtesting trading strategies.
    """
    def __init__(self, conf_file, instrument, bar_length, date_brackets, leverage):
        """
        Parameters
        ==========
        conf_file: str
            path to and filename of the configuration file,
            e.g. "/home/me/oanda.cfg"
        instrument: str
            ticker symbol (instrument) to be backtested
        bar_length: str
            bar granularity, a string like "S5", "M1" or "D"
        start: str
            start date for data import
        end: str
            end date for data import
        risk: float
            risk ratio parameter to trigger stop losses expressed in decimals
        leverage: float
            leverate ratio available, or 1/margin rate
        ratio: float
            ratio of total available margin willing to risk
        close_out: float
            1 - margin closeout ratio (max loss before stop)
        """
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = bar_length
        # self.fast_bar_length = "M" + str(int(int(bar_length[1:])/3))
        self.fast_bar_count = int(bar_length[1:])
        self.fast_bar_length = "M1"
        self.bar_number = pd.to_timedelta(bar_length[1:] + "min")
        # self.start = start
        # self.end = end
        self.date_brackets = date_brackets
        # self.raw_data = None
        # self.raw_data_fast = None
        self.data = None
        self.data_fast = None
        self.data_filter = None
        self.data_intra = None
        self.data_final = None
        self.leverage = leverage
        # self.ratio = ratio
        self.results = None
        self.perf = None
        self.optimization_results = None
        self.max_return = None
        self.trades = None
        self.total_trades = 0
        self.wins = None
        self.total_wins = 0
        self.accuracy = None
        self.total_accuracy = None
        self.total_return = 1
        self.cv_results = pd.DataFrame()
        # self.get_data()
        # self.get_data_fast()

    def get_data(self, start, end):
        """ Retrieves and prepares the data.
        """
        dfm = self.get_history(instrument = self.instrument, start = start, end = end,
                               granularity = self.bar_length, price = "M", localize = False)[["o", "h", "l", "c"]].dropna()
        dfa = self.get_history(instrument = self.instrument, start = start, end = end,
                               granularity = self.bar_length, price = "A", localize = False).c.dropna().to_frame()
        dfa.rename(columns={"c": "ask"}, inplace=True)
        dfb = self.get_history(instrument = self.instrument, start = start, end = end,
                               granularity = self.bar_length, price = "B", localize = False).c.dropna().to_frame()
        dfb.rename(columns={"c": "bid"}, inplace=True)
        df = pd.concat((dfm, dfa, dfb), axis=1)
        df["returns"] = np.log(df["c"] / df["c"].shift(1))
        df["c_prev"] = df["c"].shift(1)
        df["h_prev"] = df["h"].shift(1)
        df["l_prev"] = df["l"].shift(1)
        df["spread"] = df["ask"] - df["bid"]
        df["ohlc4"] = df[["o", "h", "l", "c"]].mean(axis=1)
        df["hlc3"] = df[["h", "l", "c"]].mean(axis=1)
        df["ohlc4_prev"] = df["ohlc4"].shift(1)
        # df["trading_cost"] = (df["spread"]/2) / df["c"]
        # self.raw_data = df.copy()
        self.data = df.copy()

    def get_data_fast(self, start, end):
        """ Retrieves and prepares the data.
        """
        dfm = self.get_history(instrument = self.instrument, start = start, end = end,
                               granularity = self.fast_bar_length, price = "M", localize = False)[["o", "h", "l", "c"]].dropna()
        dfa = self.get_history(instrument = self.instrument, start = start, end = end,
                               granularity = self.fast_bar_length, price = "A", localize = False).c.dropna().to_frame()
        dfa.rename(columns={"c": "ask"}, inplace=True)
        dfb = self.get_history(instrument = self.instrument, start = start, end = end,
                               granularity = self.fast_bar_length, price = "B", localize = False).c.dropna().to_frame()
        dfb.rename(columns={"c": "bid"}, inplace=True)
        df = pd.concat((dfm, dfa, dfb), axis=1)
        # df["returns"] = np.log(df["c"] / df["c"].shift(1))
        df["spread"] = df["ask"] - df["bid"]
        # df["trading_cost"] = (df["spread"]/2) / df["c"]
        # self.raw_data_fast = df.copy()
        self.data_fast = df.copy()

    def ifnull(self, var, val):
        if var == np.nan:
            return val
        return var

    def kernel_regression(self, exclude, relative_weight, lookback):
        """  Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel
        """
        df = self.data.copy().reset_index()
        df["kern_y"] = np.nan
        df["current_weight_prev"] = np.nan
        df["cumm_weight_prev"] = np.nan

        # iterate through positions and calculate 
        for row in df.iterrows():
            if row[0] > exclude:
                current_weight = 0
                cumm_weight = 0
                for i in range(exclude+2):
                    y = df.loc[row[0]-i, "c"]
                    w = (1 + i**2 / (2*relative_weight*lookback**2))**(-relative_weight)
                    current_weight += y*w
                    cumm_weight += w
                    if i == 1:
                        df.loc[row[0],"current_weight_prev"] = current_weight
                        df.loc[row[0],"cumm_weight_prev"] = cumm_weight
                    if i == 0:
                        df.loc[row[0],"weight"] = w
                df.loc[row[0],"kern_y"] = current_weight / cumm_weight
        
        # df["kern_y_prev"] = df["kern_y"].shift(1)
        # df["y_threshold_c"] = df["current_weight_prev"]**2 / (df["weight"] * (df["cumm_weight_prev"] - df["current_weight_prev"]))
        df["y_threshold_c"] = df["current_weight_prev"] / df["cumm_weight_prev"]
        self.data = self.data.join(df.set_index("time")[["kern_y", "y_threshold_c"]])

    def volatility_filter(self, short_window, long_window):
        """ Returns RSI Signal based on input window.
        Parameters
        ==========
        window: int
            Window parameter for RSI calculation
        """
        df = self.data.copy()
        df["hlc_ATR"] = df.ta.atr(high="h", low="l", close="c", length=short_window)
        df["hlc_ATR_{}".format(long_window)] = df.ta.atr(high="h", low="l", close="c", length=long_window)
        df["vol_filter"] = np.where(df["hlc_ATR"] > df["hlc_ATR_{}".format(long_window)], 1, 0)
        # df["TR_threshold"] = df["hlc_ATR_{}".format(long_window)].shift(1)
        self.data = self.data.join(df[["vol_filter"# , "TR_threshold"
            ]])

    def klm(self, ema_length):
        """  Calculate Kalman-Like Moving Filter slope decline compared to historical average 
        """
        df = self.data.copy().reset_index()
        df["ohlc_diff"] = np.nan
        df["hl_diff"] = np.nan
        df["klmf"] = np.nan
        df["abs_ratio"] = np.nan
        df["alpha"] = np.nan
        df["abs_slope"] = np.nan

        # iterate through positions and calculate 
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"ohlc_diff"] = 0
                df.loc[row[0],"hl_diff"] = 0
                df.loc[row[0],"klmf"] = 0
            else:
                df.loc[row[0],"ohlc_diff"] = 0.2 * (df.loc[row[0],"ohlc4"] - df.loc[row[0]-1,"ohlc4"]) + 0.8 * self.ifnull(df.loc[row[0]-1,"ohlc_diff"], 0)
                df.loc[row[0],"hl_diff"] = 0.1 * (df.loc[row[0],"h"] - df.loc[row[0],"l"]) + 0.8 * self.ifnull(df.loc[row[0]-1,"hl_diff"], 0)
                df.loc[row[0],"abs_ratio"] = abs(df.loc[row[0],"ohlc_diff"] / df.loc[row[0],"hl_diff"])
                df.loc[row[0],"alpha"] = (math.sqrt(df.loc[row[0],"abs_ratio"]**4 + 16 * df.loc[row[0],"abs_ratio"]**2) - df.loc[row[0],"abs_ratio"]**2) / 8
                df.loc[row[0],"klmf"] = df.loc[row[0],"alpha"] * df.loc[row[0],"ohlc4"] + (1 - df.loc[row[0],"alpha"]) * self.ifnull(df.loc[row[0]-1,"klmf"], 0)
        
        df["abs_slope"] = abs(df["klmf"] - df["klmf"].shift(1))
        df["slope_EMA"] = df.ta.ema(close = "abs_slope", length=ema_length)
        # df["slope_EMA"] = 1.0 * (df["abs_slope"]*(2/(ema_length+1)) + df["slope_EMA"].shift(1)*(1-(2/(ema_length+1))))
        df["normalized_slope_decline"] = (df["abs_slope"] - df["slope_EMA"]) / df["slope_EMA"]

        df["ohlc_diff_prev"] = df["ohlc_diff"].shift(1)
        df["hl_diff_prev"] = df["hl_diff"].shift(1)
        df["klmf_prev"] = df["klmf"].shift(1)
        df["slope_EMA_prev"] = df["slope_EMA"].shift(1)
        self.data = self.data.join(df.set_index("time")[["normalized_slope_decline", "ohlc_diff_prev", "hl_diff_prev", "klmf_prev", "slope_EMA_prev"]])

    def klm_filter(self, ema_length):
        """  Calculate Kalman-Like Moving Filter slope decline compared to historical average 
        """
        df = self.data_filter.copy()
        alpha = 2 / (ema_length + 1)
        df["ohlc_diff_intra"] = 0.2 * (df["ohlc4_intra"] - df["ohlc4_prev"]) + 0.8 * df["ohlc_diff_prev"].fillna(0)
        df["hl_diff_intra"] = 0.1 * (df["h_intra"] - df["l_intra"]) + 0.8 * df["hl_diff_prev"].fillna(0)
        df["abs_ratio_intra"] = abs(df["ohlc_diff_intra"] / df["hl_diff_intra"])
        df["alpha_intra"] = (np.sqrt(df["abs_ratio_intra"]**4 + 16 * df["abs_ratio_intra"]**2) - df["abs_ratio_intra"]**2) / 8
        df["klmf_intra"] = df["alpha_intra"] * df["ohlc4_intra"] + (1 - df["alpha_intra"]) * df["klmf_prev"].fillna(0)
        
        df["abs_slope_intra"] = abs(df["klmf_intra"] - df["klmf_prev"]).fillna(0)
        df["slope_EMA_intra"] = df["abs_slope_intra"]*alpha + df["slope_EMA_prev"]*(1-alpha)
        df["normalized_slope_decline_intra"] = (df["abs_slope_intra"] - df["slope_EMA_intra"]) / df["slope_EMA_intra"]
        self.data_filter = self.data_filter.join(df[["normalized_slope_decline_intra"]])

    def data_fast_resample(self, time_filter=0):
        df = self.data_fast.copy()
        df["time_column"] = df.index
        df["resample_time"] = df["time_column"].resample(str(self.fast_bar_count)+"T", label = "left").first().reindex(df.index, method="ffill")
        df = df.copy().reset_index()

        for row in df.iterrows():
            if row[0] > 0 and df.loc[row[0],"resample_time"] == df.loc[row[0]-1,"resample_time"]:
                df.loc[row[0],"o"] = df.loc[row[0]-1,"o"]
                df.loc[row[0],"h"] = np.maximum(df.loc[row[0],"h"], df.loc[row[0]-1,"h"])
                df.loc[row[0],"l"] = np.minimum(df.loc[row[0],"l"], df.loc[row[0]-1,"l"])
        df["ohlc4"] = df[["o", "h", "l", "c"]].mean(axis=1)
        df["hlc3"] = df[["h", "l", "c"]].mean(axis=1)
        self.data_fast = df.copy().set_index("time")
        self.data_fast = self.data_fast.join(self.data, on="resample_time", how="right", lsuffix="_intra")
        self.data_fast["tr_intra"] = np.maximum(np.maximum((self.data_fast["h_intra"] - self.data_fast["l_intra"]), (self.data_fast["h_intra"] - self.data_fast["c_prev"].fillna(0)).abs()), (self.data_fast["l_intra"] - self.data_fast["c_prev"].fillna(0)).abs())
        if time_filter > 0:
            self.data_fast = self.data_fast[(pd.to_datetime(self.data_fast["time_column"]) >= pd.to_datetime(self.data_fast["resample_time"]) + self.bar_number*time_filter)]

    def strategy(self, date_range, vol, klm, kn_exclude, kn_weight, lookback, rsi, wt_lengths, wt_sma, cci, adx, ld_size, ld_bars, ld_nn, ld_nratio, klm_threshold, nth_enter, intra_nth, knn_threshold, stop_params, session, hour_time, double_stop, time_filter):

        start = pd.to_datetime(date_range[1]) - (self.bar_number*ld_size + self.bar_number*250).round(freq="d")
        start = start.strftime("%Y-%m-%d")
        end = pd.to_datetime(date_range[2])
        end = end.strftime("%Y-%m-%d")

        fast_start = pd.to_datetime(date_range[1]) - (self.bar_number*20).round(freq="d")
        fast_start = fast_start.strftime("%Y-%m-%d")

        self.get_data(start, end)
        self.get_data_fast(fast_start, end)
        self.volatility_filter(short_window=vol[0], long_window=vol[1])
        self.klm(ema_length=klm)
        self.kernel_regression(exclude=kn_exclude, relative_weight=kn_weight, lookback=lookback)
        self.data_fast_resample(time_filter=time_filter)

        self.data_fast["time_close"] = self.data_fast.index.tz_convert("America/New_York")
        self.data_filter = self.data_fast.copy().reset_index()
        self.data_filter = self.data_filter[(self.data_filter["tr_intra"] > self.data_filter["TR_threshold"]) & (self.data_filter["tr_intra"] < self.data_filter["TR_threshold"])]
        self.klm_filter(ema_length=klm)

        self.data_filter = self.data_filter[self.data_filter["normalized_slope_decline_intra"] >= klm_threshold]
        self.data_filter = self.data_filter[((pd.to_datetime(self.data_filter["time_close"]).dt.time >= pd.to_datetime("{}:{}".format(session[0], hour_time[0])).time()) & 
                (pd.to_datetime(self.data_filter["time_close"]).dt.time < pd.to_datetime("{}:{}".format(session[1], hour_time[1])).time()))]

        self.data_filter = self.data_filter[["time_column", "resample_time", "c_rsi{}_intra_scaled".format(rsi[0]), "wave_distance_intra_scaled", "c_CCI_intra_scaled", "adx_intra_scaled", 
                                      "c_rsi{}_intra_scaled".format(rsi[1]), "c_intra", "bid_intra", "ask_intra", "tr_intra", "TR_threshold", "normalized_slope_decline_intra"
                                      ]]

        self.data_final = self.data_intra[self.data_intra["open_total"] == intra_nth].join(self.data, how="right", lsuffix="_intra")
        # self.data_final = self.data_final.set_index("time")
        self.data_final["open"] = self.data_final["open"].fillna(0)

        self.data_final["time_close"] = self.data_final.index.tz_convert("America/New_York")
        self.data_final = self.data_final[(self.data_final.index >= date_range[1]) & (self.data_final.index <= date_range[2])].copy()
        self.initiate_position(close_out=True, stop_params=stop_params, double_stop=double_stop)
        
        # calculate cumulative strategy returns
        self.update_strategy()
        self.results = self.data_final.copy().drop(["o", "h", "l", "ask", "bid", "spread"], axis=1)
       
        # self.hold = self.data_final["creturns"].iloc[-1] # absolute performance of buy and hold strategy
        self.perf = self.data_final["cstrategy"].iloc[-1] # absolute performance of the strategy
        self.max_return = self.data_final["cstrategy"].max()
        # self.outperf = self.perf - self.hold # out-/underperformance of strategy
        self.trades = self.data_final["trades"].sum()
        self.wins = self.data_final[self.data_final["win"] > 0]["win"].sum()
        returns_pos = self.data_final[self.data_final["trade_return"] > 0]["trade_return"].mean()
        returns_neg = self.data_final[self.data_final["trade_return"] < 0]["trade_return"].mean()
        self.accuracy = self.wins / self.data_final["win"].abs().sum() if self.data_final["win"].abs().sum() > 0 else 1.0
        return {"params": (vol, klm, kn_exclude, kn_weight, lookback, rsi, wt_lengths, wt_sma, cci, adx, ld_size, ld_bars, ld_nn, 
                           ld_nratio, klm_threshold, nth_enter, intra_nth, knn_threshold, stop_params, session, hour_time, double_stop, time_filter), 
                f"wins{date_range[0]}": self.wins, f"trades{date_range[0]}": self.trades, f"accuracy{date_range[0]}": self.accuracy,
                f"strat_return{date_range[0]}": round(self.perf, 6)
                , f"max_return{date_range[0]}": round(self.max_return, 6), f"returns_pos{date_range[0]}": round(returns_pos, 6), f"returns_neg{date_range[0]}": round(returns_neg, 6)
                , f"score{date_range[0]}": round(self.perf*self.accuracy, 6), "index": date_range[0]
                }

    def initiate_position(self, close_out, stop_params, double_stop):
        """ Determine positions based on open & close signals and stop loss signals
        Parameters
        ==========
        close_out: boolean
            determine whether to update stop loss close signals based on margin closeout loss parameter
        swing_stop: boolean
            determine whether to update stop loss close signals based on recent swing highs and lows
        risk_stop: boolean
            determine whether to update stop loss close signals based on max loss risk parameter
        real_return: boolean
            determine whether to use returns including trading costs
        """
        df = self.data_final.copy().reset_index()
        df["open_price"] = np.nan
        df["open_true"] = np.nan
        df["close_real"] = np.nan
        df["close_price"] = np.nan
        df["close_true"] = np.nan
        df["position"] = np.nan
        # df["recentPHL"] = np.nan
        df["max_active_return"] = np.nan
        df["HL"] = np.nan
        df["LH"] = np.nan
        price_open = {1: "ask_intra", -1: "bid_intra"}
        price_close = {1: "bid", -1: "ask"}
        # phl = {1: "recentPL", -1: "recentPH"}
        hl = {1: "l", -1: "h"}
        lh = {1: "h", -1: "l"}
        
        # iterate through positions and define initial close signals
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"position"] = df.loc[row[0],"open"]
                if abs(df.loc[row[0],"open"]) == 1:
                    df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
            elif abs(df.loc[row[0]-1,"position"]) == 1:
                # df.loc[row[0],"recentPHL"] = df.loc[row[0], phl[df.loc[row[0]-1, "position"]]]
                df.loc[row[0],"HL"] = df.loc[row[0], hl[df.loc[row[0]-1, "position"]]]
                df.loc[row[0],"LH"] = df.loc[row[0], lh[df.loc[row[0]-1, "position"]]]
                df.loc[row[0], "max_active_return"] = ((df.loc[row[0], "LH"] - df.loc[row[0]-1, "open_true"]) * df.loc[row[0]-1, "position"]) / df.loc[row[0]-1, "open_true"] * self.leverage
                df.loc[row[0], "max_active_return"] = df.loc[row[0], "max_active_return"] if df.loc[row[0]-1, "max_active_return"] == np.nan else df.loc[row[0], "max_active_return"] if df.loc[row[0], "max_active_return"] > df.loc[row[0]-1, "max_active_return"] else df.loc[row[0]-1, "max_active_return"]
                df.loc[row[0],"close_real"] = df.loc[row[0], price_close[df.loc[row[0]-1, "position"]]]
                if close_out and ((df.loc[row[0], "HL"] - df.loc[row[0]-1, "open_true"]) * df.loc[row[0]-1, "position"]) / df.loc[row[0]-1, "open_true"] * self.leverage < -stop_params:
                    df.loc[row[0],"close_true"] = -stop_params/self.leverage*df.loc[row[0]-1, "open_true"]/df.loc[row[0]-1, "position"] + df.loc[row[0]-1, "open_true"]
                    df.loc[row[0],"position"] = 0
                    if abs(df.loc[row[0],"open"]) == 1:
                        df.loc[row[0],"position"] = df.loc[row[0],"open"]
                        df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                elif pd.to_datetime(df.loc[row[0],"time"]).tz_convert("America/New_York").time() == pd.to_datetime("16:45").time():
                    # df.loc[row[0],"close_true"] = df.loc[row[0], price_close[df.loc[row[0]-1, "position"]]]
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"position"] = 0
                # elif pd.to_datetime(df.loc[row[0],"time"]).tz_convert("America/New_York").time() == pd.to_datetime("{}:{}".format(session[2], hour_time[2])).time():
                    # df.loc[row[0],"close_true"] = df.loc[row[0], price_close[df.loc[row[0]-1, "position"]]]
                    # df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    # df.loc[row[0],"position"] = 0
                elif abs(df.loc[row[0],"open"]) == 1 and df.loc[row[0]-1,"position"] != df.loc[row[0],"open"]:
                    df.loc[row[0],"position"] = df.loc[row[0],"open"]
                    df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                    # df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"close_true"] = df.loc[row[0], price_open[-df.loc[row[0], "position"]]]
                elif abs(df.loc[row[0],"open"]) == 1 and df.loc[row[0]-1,"position"] == df.loc[row[0],"open"]:
                    df.loc[row[0],"position"] = df.loc[row[0]-1,"position"]
                    df.loc[row[0],"open_true"] = df.loc[row[0]-1,"open_true"]
                # elif risk_stop and ((df.loc[row[0], "close_real"] - df.loc[row[0]-1, "open_true"]) * df.loc[row[0]-1, "position"]) / df.loc[row[0]-1, "open_true"] * self.leverage < -stop_params[0] and df.loc[row[0],"macd_crossover_close_2"] != df.loc[row[0]-1,"position"]:
                    # df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    # df.loc[row[0],"position"] = 0
                # elif df.loc[row[0],"macd_crossover_close_2"] != df.loc[row[0]-1,"position"] and df.loc[row[0], "max_active_return"] > mar:
                    # df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    # df.loc[row[0],"position"] = 0
                # elif df.loc[row[0],"macd_crossover_close"] != df.loc[row[0]-1,"position"] and (df.loc[row[0], "close_real"] - df.loc[row[0]-1, "open_true"])*df.loc[row[0]-1,"position"] > 0:
                    # df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    # df.loc[row[0],"position"] = 0
                # elif swing_stop and (df.loc[row[0],"c"] - df.loc[row[0],"recentPHL"]) * df.loc[row[0]-1, "position"] < 0:
                    # df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    # df.loc[row[0],"position"] = 0
                elif np.sign(df.loc[row[0],"c"] - df.loc[row[0],"y_threshold_c"]) != df.loc[row[0]-1,"position"] and df.loc[row[0],"c"] - df.loc[row[0],"y_threshold_c"] != 0:
                    if double_stop:
                        if np.sign(df.loc[row[0]-1,"y_pred"] - df.loc[row[0]-2,"y_pred"]) != df.loc[row[0]-1,"position"]:
                            df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                            df.loc[row[0],"position"] = 0
                    else:
                        df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                        df.loc[row[0],"position"] = 0
                else:
                    df.loc[row[0],"position"] = df.loc[row[0]-1,"position"]
                    df.loc[row[0],"open_true"] = df.loc[row[0]-1,"open_true"]
            elif abs(df.loc[row[0],"open"]) == 1:
                df.loc[row[0],"position"] = df.loc[row[0],"open"]
                df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
            else:
                df.loc[row[0],"position"] = df.loc[row[0]-1,"position"]

            # if row[0] == len(df) - 1:
                # if abs(df.loc[row[0],"position"]) == 1:
                    # df.loc[row[0],"close_true"] = df.loc[row[0], price_close[df.loc[row[0], "position"]]]
                    # df.loc[row[0],"position"] = 0
        # self.data = self.data.join(df.set_index("time")[["open_true", "close_true", "position", "recentPHL", "HL"]])
        self.data_final = self.data_final.join(df.set_index("time")[["open_true", "close_true", "position", "HL", "LH"]])

    def update_strategy(self):
        """ Updates cumulative strategy returns based on open and close prices.
        Parameters
        ==========
        real_return: boolean
            determine whether to calculate cumulative strategy returns including trading costs
        """
        df = self.data_final.copy().reset_index()
        df["cstrategy"] = np.nan
        # df["trades"] = df["position"].diff().fillna(0).abs()

        # determine when a trade takes place and cumulative calculate strategy returns
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"cstrategy"] = 1
            elif math.isnan(df.loc[row[0], "close_true"]): 
                df.loc[row[0],"cstrategy"] = df.loc[row[0]-1,"cstrategy"]
            else:
                df.loc[row[0],"cstrategy"] = df.loc[row[0]-1,"cstrategy"] * ((df.loc[row[0], "close_true"] - df.loc[row[0]-1, "open_true"])*df.loc[row[0]-1, "position"]/df.loc[row[0]-1, "open_true"] * self.leverage + 1)

        df["trades"] = np.where(df["cstrategy"].diff() == 0, 0, 1)
        df.loc[0, "trades"] = 0

        # df["creturns"] = df["returns"].cumsum().apply(np.exp)
        df["win"] = np.sign(df["cstrategy"].diff().fillna(0))
        df["trade_return"] = df["cstrategy"].pct_change()
        self.data_final = self.data_final.join(df.set_index("time")[["trades", "cstrategy", "win", "trade_return"]])

    def parallel_optimize(self):
        """
        vol = [x for x in range(5, 16, 1)]
        klm = [x for x in range(50, 501, 50)]
        kn_exclude = [x for x in range(2, 26, 1)]

        rsi = product(range(5,16,1), range(5,16,1))
        rsi = [(x[0], x[1]) for x in rsi if x[0] > x[1]]
        # rsi = [(14,9), (13,5)]

        wt_lengths = product(range(5,15,1), range(5,15,1))
        wt_lengths = [(x[0], x[1]) for x in rsi if x[0] < x[1]]
        # wt_lengths = [(10,11), (9,13)]

        wt_sma = [x for x in range(3, 11, 1)]
        cci = [x for x in range(10, 31, 2)]
        adx = [x for x in range(10, 31, 2)]
        ld_size = [x for x in range(1000, 10001, 500)]
        ld_bars = [x for x in range(2, 9, 1)]
        ld_nn = [x for x in range(3, 12, 1)]
        ld_nratio = [x/100 for x in range(50, 91, 5)]
        klm_threshold = [(x-10)/10 for x in range(0, 21, 1)]
        nth_enter = [3]
        intra_nth = [1,2,3]
        knn_threshold = [0,1,2,3,4,5]
        double_stop = [False, True]

        session = product(["22", "23", "00", "01", "02"], ["13", "14", "15", "16"])
        # session = [(x[0], x[1], x[2]) for x in session if x[2] >= x[1]]
        # session = [("23", "13", "14")]

        hour_time = product(["00", "15", "30", "45"], ["00", "15", "30", "45"], ["00", "15", "30", "45"])
        # hour_time = [("30", "15", "45")]

        stop_params = [0.02, 0.03, 0.04, 0.05]
        """
        vol = [(1,10)]
        klm = [200]
        kn_exclude = [24]
        kn_weight = [3]
        lookback = [3]
        klm_threshold = [-0.4]
        nth_enter = [2]
        stop_params = [0.25]
        # session = product(["22", "23", "00", "01", "02"], ["13", "14", "15", "16"])
        # session = product(["02"], ["14"])
        session = [("02", "14")]
        hour_time = [("30", "00")]
        double_stop = [False]
        # time_filter = [0, 1/3]
        time_filter = [0]

        combinations = list(product(self.date_brackets, vol, klm, kn_exclude, kn_weight, lookback, rsi, wt_lengths, wt_sma, cci, adx, ld_size, ld_bars, ld_nn, 
                           ld_nratio, klm_threshold, nth_enter, intra_nth, knn_threshold, stop_params, session, hour_time, double_stop, time_filter)
        )
        print(f"Parameter combinations: {len(combinations)}")

        # params_df = pd.read_excel(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\trade_results_eurusd_cv_ldc.xlsx", usecols = ["params"])
        # params_df = params_df.drop_duplicates()
        # params = params_df["params"].tolist()
        # combinations = [x for x in combinations if str(x[1:]) not in params]

        print(f"Cores available: {cpu_count()}")
        print(f"Reduced combinations: {len(combinations)}")

        with Pool(cpu_count()) as pool:
            self.optimization_results = pool.starmap(self.strategy, combinations)

    def cross_val_combine(self):
        self.parallel_optimize()
        results = [x for x in self.optimization_results]
        # self.cv_results = pd.DataFrame(results)
        # self.cv_results = self.cv_results.set_index("params")

        for x in self.date_brackets:
            data = [d for d in results if d["index"] == x[0]]
            df = pd.DataFrame(data)
            df = df.set_index("params")
            df = df.drop(["index"], axis=1)
            self.cv_results = self.cv_results.join(df, how = "outer")

        self.cv_results["total_wins"] = self.cv_results.filter(regex=("wins")).sum(axis=1)
        self.cv_results["total_trades"] = self.cv_results.filter(regex=("trades")).sum(axis=1)
        self.cv_results["total_accuracy"] = np.where(self.cv_results["total_trades"] > 0, self.cv_results["total_wins"] / self.cv_results["total_trades"], 1.0)
        self.cv_results["total_return"] = self.cv_results.filter(regex=("strat_return")).prod(axis=1)
        self.cv_results["return_std"] = self.cv_results.filter(regex=("strat_return")).std(axis=1)
        self.cv_results["pos_re"] = self.cv_results.filter(regex=("returns_pos")).mean(axis=1)
        self.cv_results["neg_re"] = self.cv_results.filter(regex=("returns_neg")).mean(axis=1)
        self.cv_results["total_score"] = self.cv_results["total_return"] * self.cv_results["total_accuracy"]
        self.cv_results = self.cv_results.sort_values(by=["total_score", "total_accuracy", "pos_re"], ascending=False)


if __name__ == "__main__":

    start_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)))

    # trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", "EUR_USD", "M15", [(1, "2016-11-05", "2017-02-05"), (2, "2020-01-12", "2020-04-12"), (3, "2023-06-01", "2023-09-01")], 50)
    trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", "EUR_USD", "M15", [(1, "2023-06-01", "2023-09-01")], 50)
    # calculate strategy returns
    trader.cross_val_combine()

    trader.cv_results.to_csv(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\trade_results_eurusd_cv_ldc_bidask.csv", mode="w", index=True, header=True)

    end_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end_time)))
    print(1.0*(end_time - start_time)/60)