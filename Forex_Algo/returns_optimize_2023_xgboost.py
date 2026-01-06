import pandas as pd
import pandas_ta as ta
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
import tpqoa
import xgboost as xgb
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from itertools import product
from multiprocessing import Pool, cpu_count
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_sample_weight
from tempfile import mkdtemp


class TrendAlgoBacktester(tpqoa.tpqoa):
    """ Class for the vectorized backtesting trading strategies.
    """
    def __init__(self, conf_file, instrument, bar_length, start, end, leverage, train_range, test_range, window, gain, loss):
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
        self.fast_bar_length = "M" + str(int(int(bar_length[1:])/3))
        self.bar_number = pd.to_timedelta(bar_length[1:] + "min")
        self.start = start
        self.end = end
        self.raw_data = None
        self.data = None
        self.leverage = leverage
        self.train_range = train_range
        self.test_range = test_range
        self.window = window
        self.gain = gain
        self.loss = loss
        # self.ratio = ratio
        self.results = None
        self.perf = None
        self.optimization_results = None
        self.max_return = None
        self.trades = None
        self.wins = None
        self.cv_results = pd.DataFrame()
        self.date_brackets = None
        self.model = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.roc_auc = None
        self.f1 = None
        self.importance_df = None
        self.y_compare = None
        self.test_df = None
        self.pred_accuracy = None
        self.get_data()

    def get_data(self):
        """ Retrieves and prepares the data.
        """
        dfm = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "M", localize = False)[["o", "h", "l", "c"]].dropna()
        dfa = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "A", localize = False).c.dropna().to_frame()
        dfa.rename(columns={"c": "ask"}, inplace=True)
        dfb = self.get_history(instrument = self.instrument, start = self.start, end = self.end,
                               granularity = self.bar_length, price = "B", localize = False).c.dropna().to_frame()
        dfb.rename(columns={"c": "bid"}, inplace=True)
        df = pd.concat((dfm, dfa, dfb), axis=1)
        # df = df.resample(pd.to_timedelta("15min"), label = "right").last().ffill()
        df["returns"] = np.log(df["c"] / df["c"].shift(1))
        df["spread"] = df["ask"] - df["bid"]
        # df["trading_cost"] = (df["spread"]/2) / df["c"]
        self.raw_data = df.copy()
        self.data = df.copy()

    def ema_crossover(self, short, longg):
        """ Returns EMA Crossover Signal based on input parameters.
        Parameters
        ==========
        short: int
            Short EMA window
        longg: int
            Long EMA window
        """
        df = self.data.copy()
        df["ema_{}".format(short)] = df["c"].ewm(span=short, adjust=False, min_periods=short).mean()
        df["ema_{}".format(longg)] = df["c"].ewm(span=longg, adjust=False, min_periods=longg).mean()
        df["ema_crossover_{}_{}".format(short, longg)] = (df["ema_{}".format(short)] - df["ema_{}".format(longg)])
        df["ema_crossover_{}_{}_signal".format(short, longg)] = np.where(df["ema_crossover_{}_{}".format(short, longg)] > 0, 1,
                                                                          np.where(df["ema_crossover_{}_{}".format(short, longg)] < 0, -1, 0)
                                                                          )
        self.data = self.data.join(df[["ema_crossover_{}_{}".format(short, longg), "ema_crossover_{}_{}_signal".format(short, longg)]])

    def macd_crossover(self, short, longg, signal, num# length
        ):
        """ Returns MACD Crossover Signal based on input parameters.
        Parameters
        ==========
        short: int
            Short EMA window
        longg: int
            Long EMA window
        signal: int
            MACD EMA smoothing window
        """
        df = self.data.copy()
        df.ta.macd(close="c", fast=short, slow=longg, signal=signal, append=True)
        df["trend_macd_crossover_signal_{}".format(num)] = np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)] > 0, 1,
                                               np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)] < 0, -1, 0)
                                               )
        df["MACDhO_{}_{}_{}".format(short, longg, signal)] = df["MACDh_{}_{}_{}".format(short, longg, signal)]
        # df["macd_distance"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].abs()
        # df["macd_change"] = np.where(np.sign(df["MACDh_{}_{}_{}".format(short, longg, signal)]).diff().ne(0), 0, df["macd_distance"] - df["macd_distance"].shift(1))
        # df["macd_std"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].rolling(length, min_periods=length).std()
        # df["macd_upper"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].rolling(length, min_periods=length).std() * -1
        self.data = self.data.join(df[["MACDhO_{}_{}_{}".format(short, longg, signal), 
                                       "trend_macd_crossover_signal_{}".format(num) # , "macd_distance", "macd_change", "macd_std"
            ]])

    def macd_crossover_close(self, short, longg, signal, # length
        ):
        """ Returns MACD Crossover Signal based on input parameters.
        Parameters
        ==========
        short: int
            Short EMA window
        longg: int
            Long EMA window
        signal: int
            MACD EMA smoothing window
        """
        df = self.data.copy()
        df.ta.macd(close="c", fast=short, slow=longg, signal=signal, append=True)
        df["macd_crossover_close"] = np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)] > 0, 1,
                                               np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)] < 0, -1, 0)
                                               )
        df["MACDhC_{}_{}_{}".format(short, longg, signal)] = df["MACDh_{}_{}_{}".format(short, longg, signal)]
        # df["macd_distance"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].abs()
        # df["macd_change"] = np.where(np.sign(df["MACDh_{}_{}_{}".format(short, longg, signal)]).diff().ne(0), 0, df["macd_distance"] - df["macd_distance"].shift(1))
        # df["macd_std"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].rolling(length, min_periods=length).std()
        # df["macd_upper"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].rolling(length, min_periods=length).std() * -1
        self.data = self.data.join(df[["MACDhC_{}_{}_{}".format(short, longg, signal), 
                                       "macd_crossover_close" # , "macd_distance", "macd_change", "macd_std"
            ]])

    def rsi(self, window):
        """ Returns RSI Signal based on input window.
        Parameters
        ==========
        window: int
            Window parameter for RSI calculation
        """
        df = self.data.copy()
        df.ta.rsi(close="c", length=window, append=True)
        df["trend_rsi_{}_signal".format(window)] = np.where(df["RSI_{}".format(window)] > 50, 1,
                                                      np.where(df["RSI_{}".format(window)] < 50, -1, 0)
                                                      )
        self.data = self.data.join(df[["RSI_{}".format(window), "trend_rsi_{}_signal".format(window)]])

    def aroon(self, window):
        """ Returns Aroon Oscillator based on input window.
        Parameters
        ==========
        window: lenght
            Window parameter for RSI calculation
        """
        df = self.data.copy()
        df.ta.aroon(high="h", low="l", length=window, append=True)
        self.data = self.data.join(df[["AROONOSC_{}".format(window)]])

    def consensus_trend(self):
        """ Returns whether MACD Crossover and RSI Trends are consistent.
        """
        df = self.data.copy()
        df["consensus_signal"] = 0
        df["consensus_signal"] = np.where(df.filter(regex=("trend")).sum(axis=1)/df.filter(regex=("trend")).count(axis=1) >= 1, 1,
                                          np.where(df.filter(regex=("trend")).sum(axis=1)/df.filter(regex=("trend")).count(axis=1) <= -1, -1, 0)
                                          )
        df["consensus_signal"] = np.where(df["consensus_signal"].shift(1) == df["consensus_signal"], 0, df["consensus_signal"])
        self.data = self.data.join(df[["consensus_signal"]])

    def ma_crossover(self, short, longg):
        """ Returns MACD Crossover Signal based on input parameters.
        Parameters
        ==========
        short: tuple
            short moving averagae parameter tuple (int, string) with window size and moving average type ("EMA", "SMA") 
        longg: tuple
            long moving averagae parameter tuple (int, string) with window size and moving average type ("EMA", "SMA") 
        """
        df = self.data.copy()
        if short[1] == "EMA":
            df["mac_short"] = df["c"].ewm(span=short[0], adjust=False, min_periods=short[0]).mean()
        if short[1] == "SMA":
            df["mac_short"] = df["c"].rolling(short[0], min_periods=short[0]).mean()
        if longg[1] == "EMA":
            df["mac_long"] = df["c"].ewm(span=longg[0], adjust=False, min_periods=longg[0]).mean()
        if longg[1] == "SMA":
            df["mac_long"] = df["c"].rolling(longg[0], min_periods=longg[0]).mean()
        df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] = (df["mac_short"] - df["mac_long"])
        df["MAC_{}{}_{}{}_signal".format(short[1], short[0], longg[1], longg[0])] = np.where(df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] > 0, 1,
                                                                                             np.where(df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] < 0, -1, 0)
                                                                                             )
        self.data = self.data.join(df[["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0]), "MAC_{}{}_{}{}_signal".format(short[1], short[0], longg[1], longg[0])]])

    def ma_crossover_neg(self, short, longg):
        """ Returns MACD Crossover Signal based on input parameters.
        Parameters
        ==========
        short: tuple
            short moving averagae parameter tuple (int, string) with window size and moving average type ("EMA", "SMA") 
        longg: tuple
            long moving averagae parameter tuple (int, string) with window size and moving average type ("EMA", "SMA") 
        """
        df = self.data.copy()
        if short[1] == "EMA":
            df["mac_short"] = df["c"].ewm(span=short[0], adjust=False, min_periods=short[0]).mean()
        if short[1] == "SMA":
            df["mac_short"] = df["c"].rolling(short[0], min_periods=short[0]).mean()
        if longg[1] == "EMA":
            df["mac_long"] = df["c"].ewm(span=longg[0], adjust=False, min_periods=longg[0]).mean()
        if longg[1] == "SMA":
            df["mac_long"] = df["c"].rolling(longg[0], min_periods=longg[0]).mean()
        df["MAC_neg_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] = (df["mac_short"] - df["mac_long"])
        df["MAC_neg_{}{}_{}{}_signal".format(short[1], short[0], longg[1], longg[0])] = np.where(df["MAC_neg_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] > 0, -1,
                                                                                             np.where(df["MAC_neg_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] < 0, 1, 0)
                                                                                             )
        self.data = self.data.join(df[["MAC_neg_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0]), "MAC_neg_{}{}_{}{}_signal".format(short[1], short[0], longg[1], longg[0])]])


    def volatility_osc(self, length, multiplier):
        """ Returns volatilitiy oscillator based on input length window.
        Parameters
        ==========
        length: int
            window parameter for volatility standard deivation calculation
        """        
        df = self.data.copy()
        df["spike_{}".format(length)] = df["c"] - df["o"]
        df["upper_{}".format(length)] = df["spike_{}".format(length)].rolling(length, min_periods=length).std()
        df["lower_{}".format(length)] = df["spike_{}".format(length)].rolling(length, min_periods=length).std() * -1
        df["hl_spike_{}".format(length)] = (df["h"] - df["l"])
        df["hl_upper_{}".format(length)] = df["hl_spike_{}".format(length)].rolling(length, min_periods=length).std()
        df["hl_lower_{}".format(length)] = df["hl_spike_{}".format(length)].rolling(length, min_periods=length).std() * -1
        df["hl_lower_{}".format(length)]
        #df["vol_signal_{}".format(length)] = np.where((df["spike"] > 0) & (df["spike"] > multiplier[0]*df["upper"]) & (df["spike"]/df["upper"] < multiplier[1]), 1,
        #                                              np.where((df["spike"] < 0) & (df["spike"] < multiplier[0]*df["lower"]) & (df["spike"]/df["lower"] < multiplier[1]), -1, 0)
        #                                              )
        df["vol_diff_{}".format(length)] = np.where(df["spike_{}".format(length)] > 0, df["spike_{}".format(length)] - df["upper_{}".format(length)],
                                                     np.where(df["spike_{}".format(length)] < 0, df["spike_{}".format(length)] - df["lower_{}".format(length)], 0))
        # df["vol_ldiff_{}".format(length)] = df["spike_{}".format(length)] - df["lower_{}".format(length)]
        #df["hl_vol_udiff_{}".format(length)] = df["hl_spike_{}".format(length)] - df["hl_upper_{}".format(length)]
        #df["hl_vol_ldiff_{}".format(length)] = df["hl_spike_{}".format(length)] - df["hl_lower_{}".format(length)]
        df["hl_vol_diff_{}".format(length)] = np.where(df["hl_spike_{}".format(length)] > 0, df["hl_spike_{}".format(length)] - df["hl_upper_{}".format(length)],
                                                     np.where(df["hl_spike_{}".format(length)] < 0, df["hl_spike_{}".format(length)] - df["hl_lower_{}".format(length)], 0))
        #self.data = self.data.join(df[["spike_{}".format(length), "upper_{}".format(length), "lower_{}".format(length), "hl_spike_{}".format(length), "hl_upper_{}".format(length), "hl_lower_{}".format(length)#, "vol_signal_{}".format(length)
        #    ]])
        #self.data = self.data.join(df[["spike_{}".format(length), "upper_{}".format(length), "lower_{}".format(length), "hl_spike_{}".format(length), "hl_upper_{}".format(length), "hl_lower_{}".format(length)#, "vol_signal_{}".format(length)
        #    ]])
        self.data = self.data.join(df[["vol_diff_{}".format(length), "hl_vol_diff_{}".format(length), "spike_{}".format(length), "hl_spike_{}".format(length)]])


    def high_low_swing(self, window):
        df = self.data.copy()
        df["rlow_{}".format(window)] = df["l"].rolling(window, min_periods=window).min()
        df["rhigh_{}".format(window)] = df["h"].rolling(window, min_periods=window).max()
        df = df.reset_index(drop=False)
        df["rlow_time"] = df[["l"]].rolling(window, min_periods=window).apply(lambda x: x.idxmin())
        df["rhigh_time"] = df[["h"]].rolling(window, min_periods=window).apply(lambda x: x.idxmax())
        df["rsince_high"] = df.index - df["rhigh_time"]
        df["rsince_low"] = df.index - df["rlow_time"]
        df["r_hl_diff_{}".format(window)] = np.where(df["rlow_time"] > df["rhigh_time"], df["rlow_{}".format(window)] - df["rhigh_{}".format(window)], np.where(df["rlow_time"] < df["rhigh_time"], df["rhigh_{}".format(window)] - df["rlow_{}".format(window)], 0))
        df["r_hl_since_{}".format(window)] = np.where(df["rlow_time"] > df["rhigh_time"], df["rsince_low"], df["rsince_high"])
        df["r_hl_distance_{}".format(window)] = (df["rsince_high"] - df["rsince_low"]).abs()
        df = df.set_index(["time"], drop=True)
        df["rlow_{}".format(window)] = df["rlow_{}".format(window)] - df["c"]
        df["rhigh_{}".format(window)] = df["rhigh_{}".format(window)] - df["c"]
        self.data = self.data.join(df[["rlow_{}".format(window), "rhigh_{}".format(window), "r_hl_diff_{}".format(window), "r_hl_since_{}".format(window), "r_hl_distance_{}".format(window)]])

    def swing(self, left, right, high_low=True):
        """ Returns swign high and lows based on left and right window parameters.
        Parameters
        ==========
        left: int
            window parameter for historical periods 
        right: int
            window parameter for future periods
        high_low: boolean
            determine whether to calculate swing high and lows based on close or period high & low
        """
        price = {False : "c", True : "h"}

        df = self.data.copy()
        df["swing_d_{}".format(right+1)] = df[price[high_low]].shift(0)
        for x in range(1, right+1):
            df["swing_d_{}".format(x)] = df[price[high_low]].shift(-x)
        for x in range(1, left+1):
            df["swing_d_{}".format(right+1+x)] = df[price[high_low]].shift(x)
        df["maxPH"] = df.filter(regex=("swing_d_")).max(axis=1)
        df["PH"] = np.where(df["maxPH"] == df["swing_d_{}".format(right+1)], df["swing_d_{}".format(right+1)], np.nan)
        df["recentPH"] = df["PH"].shift(right).astype(float).fillna(method="ffill")

        price = {False : "c", True: "l"}
        df["swing_d_{}".format(right+1)] = df[price[high_low]].shift(0)
        for x in range(1, right+1):
            df["swing_d_{}".format(x)] = df[price[high_low]].shift(-x)
        for x in range(1, left+1):
            df["swing_d_{}".format(right+1+x)] = df[price[high_low]].shift(x)
        df["minPL"] = df.filter(regex=("swing_d_")).min(axis=1)
        df["PL"] = np.where(df["minPL"] == df["swing_d_{}".format(right+1)], df["swing_d_{}".format(right+1)], np.nan)
        df["recentPL"] = df["PL"].shift(right).astype(float).fillna(method="ffill")
        self.data = self.data.join(df[["recentPL", "recentPH"]])

    def feature_engineering(self, emac, macdc, rsi, mac, vol_length, multiplier, macdc2, mac_neg, macdcc, aroon, hls):
        """ Backtests the trading strategy.
        """

        # *****************add strategy-specific signals here*********************
        self.ema_crossover(short=emac[0], longg=emac[1])
        self.macd_crossover(short=macdc[0], longg=macdc[1], signal=macdc[2], num=1# , length=100
            )
        self.macd_crossover(short=macdc2[0], longg=macdc2[1], signal=macdc2[2], num=2# , length=100
            )
        self.rsi(window=rsi[0])
        self.rsi(window=rsi[1])
        self.consensus_trend()
        self.ma_crossover(short=(mac[0], "EMA"), longg=(mac[1], "EMA"))
        self.ma_crossover(short=(mac[2], "EMA"), longg=(mac[3], "SMA"))
        self.ma_crossover_neg(short=(mac_neg[0], "EMA"), longg=(mac_neg[1], "SMA"))
        self.volatility_osc(length=vol_length[0], multiplier=multiplier)
        self.volatility_osc(length=vol_length[1], multiplier=multiplier)
        self.volatility_osc(length=vol_length[2], multiplier=multiplier)
        # self.swing(left=swing[0], right=swing[1], high_low=True)
        self.macd_crossover_close(short=macdcc[0], longg=macdcc[1], signal=macdcc[2])
        self.aroon(window=aroon[0])
        self.aroon(window=aroon[1])
        self.high_low_swing(window=hls[0])
        self.high_low_swing(window=hls[1])
        self.high_low_swing(window=hls[2])
        # ************************************************************************

        self.data = self.data.copy().dropna()
        self.data["low"] = self.data["l"].rolling(self.window, min_periods=self.window).min().shift(-self.window)
        self.data["high"] = self.data["h"].rolling(self.window, min_periods=self.window).max().shift(-self.window)
        df = self.data.reset_index(drop=False)
        df["low_time"] = df[["l"]].rolling(self.window, min_periods=self.window).apply(lambda x: x.idxmin()).shift(-self.window)
        df["high_time"] = df[["h"]].rolling(self.window, min_periods=self.window).apply(lambda x: x.idxmax()).shift(-self.window)
        self.data = df.set_index(["time"], drop=True)

        self.data["long_max_return"] = ((self.data["high"] - self.data["ask"]) * 1) / self.data["ask"] * 50 
        self.data["short_max_return"] = ((self.data["low"] - self.data["bid"]) * -1) / self.data["bid"] * 50 
        self.data["long_loss"] = ((self.data["low"] - self.data["ask"]) * 1) / self.data["ask"] * 50 
        self.data["short_loss"] = ((self.data["high"] - self.data["bid"]) * -1) / self.data["bid"] * 50

        self.data["open_signal"] = np.where(self.data["long_max_return"] > self.gain,
                                     np.where(self.data["long_loss"] < -self.loss, 
                                              np.where(self.data["low_time"] > self.data["high_time"], 1, 0), 1
                                              ),
                                     0)

        self.data["open_signal"] = np.where(self.data["short_max_return"] > self.gain,
                                     np.where(self.data["short_loss"] < -self.loss, 
                                              np.where(self.data["high_time"] > self.data["low_time"], 2, self.data["open_signal"]), 2
                                              ),
                                     self.data["open_signal"])

        # self.data = self.data.copy().drop(["o", "h", "l", "ask", "bid", "spread", "recentPL", "recentPH"], axis=1)
        self.data["time_est"] = self.data.index.tz_convert("America/New_York")
        #self.data["optimal_time1"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("23:00").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time2"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("23:15").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time3"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("23:30").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time4"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("23:45").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time5"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("00:00").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time6"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("00:15").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time7"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("00:30").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time8"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("00:45").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time9"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("01:00").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time10"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("01:15").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time11"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("01:30").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time12"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("00:45").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time13"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:00").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time14"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:15").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time1"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:00").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time2"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:15").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time3"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:30").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time4"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:45").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:15").time()), 1, 0)
        #self.data["optimal_time5"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:00").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:00").time()), 1, 0)
        #self.data["optimal_time6"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:15").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:00").time()), 1, 0)
        self.data["optimal_time7"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:30").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:00").time()), 1, 0)
        #self.data["optimal_time8"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:45").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("12:00").time()), 1, 0)
        #self.data["optimal_time9"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:00").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("11:45").time()), 1, 0)
        #self.data["optimal_time10"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:15").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("11:45").time()), 1, 0)
        #self.data["optimal_time11"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:30").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("11:45").time()), 1, 0)
        #self.data["optimal_time12"] = np.where((pd.to_datetime(self.data["time_est"]).dt.time >= pd.to_datetime("02:45").time()) & (pd.to_datetime(self.data["time_est"]).dt.time < pd.to_datetime("11:45").time()), 1, 0)
        self.data = self.data.copy().dropna()

    def random_date(self, start, end):
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        return random_date

    def generate_dates(self, start, end):
        train_start = self.random_date(start, end)
        train_end = self.random_date((train_start + relativedelta(years=5)).strftime("%Y-%m-%d"), (datetime.strptime(end, "%Y-%m-%d") + relativedelta(years=5)).strftime("%Y-%m-%d"))
        test_start = train_end + timedelta(days=1)
        test_end = test_start + relativedelta(years=2)
        return (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"), test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))

    def gsearch_class(self, X, y, p_grid):
        cachedir = mkdtemp()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = X_train.reset_index(drop=False)
        y_train = y_train.reset_index(drop=False)

        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.2)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.2)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X_train, y_train, test_size=0.2)

        range1 = self.generate_dates("2002-06-01", "2017-06-01")
        print(range1)
        range2 = self.generate_dates("2002-06-01", "2017-06-01")
        print(range2)
        range3 = self.generate_dates("2002-06-01", "2017-06-01")
        print(range3)

        train_indices = [
                          [x for x in X_train1[(X_train1["time"] >= range1[0]) & (X_train1["time"] <= range1[1])].index],
                          [x for x in X_train2[(X_train2["time"] >= range2[0]) & (X_train2["time"] <= range2[1])].index],
                          [x for x in X_train3[(X_train3["time"] >= range3[0]) & (X_train3["time"] <= range3[1])].index]
                          ]

        test_indices = [
                          [x for x in X_train1[(X_train1["time"] >= range1[2]) & (X_train1["time"] <= range1[3])].index],
                          [x for x in X_train2[(X_train2["time"] >= range2[2]) & (X_train2["time"] <= range2[3])].index],
                          [x for x in X_train3[(X_train3["time"] >= range3[2]) & (X_train3["time"] <= range3[3])].index]
                          ]

        cv = zip(train_indices, test_indices)

        X_train = X_train.drop(["time"], axis=1)
        # X_test = X_test.drop(["time"], axis=1)
        y_train = y_train["open_signal"]
        # y_test = y_test["open_signal"]

        train_sample_weight = compute_sample_weight("balanced", y_train)
        xgb_model = xgb.XGBClassifier(objective="multi:softprob",
                                                          booster="gbtree",
                                                          n_jobs=-1,
                                                          num_class=3
                                                          # sample_weight=train_sample_weight,
                                                          # random_state=88
                                                          )
        pipe = Pipeline([("estimator", xgb_model)], memory=cachedir)
        # scores = ["roc_auc", "accuracy", "precision", "recall", "f1", "f1_micro", "f1_macro", "f1_weighted"]
        f1 = metrics.make_scorer(metrics.f1_score, average = "macro")
        accuracy = metrics.make_scorer(metrics.accuracy_score)
        precision = metrics.make_scorer(metrics.precision_score, average = "macro")
        recall = metrics.make_scorer(metrics.recall_score, average = "macro")
        # roc_auc = metrics.make_scorer(metrics.roc_auc_score, average = "weighted", multi_class="ovr")
        # scores = ["roc_auc", "accuracy", "precision", "recall"]
        scores = {"precision": precision, "recall": recall, "f1": f1}
        print("Gridsearching...")
        clf = GridSearchCV(pipe, param_grid=p_grid, cv=cv, scoring=scores, refit=False, pre_dispatch=3)
        clf.fit(X_train, y_train, **{"estimator__sample_weight": train_sample_weight})

        return clf

    def XGBC_train(self, X, y, features, session):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        train_sample_weight = compute_sample_weight("balanced", y_train)
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            booster="gbtree",
            n_jobs=-1,
            n_estimators=300,
            learning_rate=0.01,
            gamma=0.01,
            max_depth=18,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            colsample_bynode=0.8,
            reg_alpha=0,
            reg_lambda=0.02
            )
        xgb_model.fit(X_train, y_train, sample_weight=train_sample_weight)

        df = self.data[(self.data.index >= self.test_range[0]) & (self.data.index <= self.test_range[1])]
        df = df[(pd.to_datetime(df["time_est"]).dt.time >= pd.to_datetime("{}:30".format(session[0])).time()) & 
                               (pd.to_datetime(df["time_est"]).dt.time < pd.to_datetime("{}:00".format(session[1])).time())
                               ]
        """
        X_test = df[["ema_crossover_60_95", "ema_crossover_60_95_signal", "MACDhO_11_35_3",
               "trend_macd_crossover_signal_1", "MACDhO_10_39_3",
               "trend_macd_crossover_signal_2", "RSI_15", "trend_rsi_15_signal",
               "RSI_5", "trend_rsi_5_signal", "consensus_signal", "MAC_EMA4_EMA10",
               "MAC_EMA4_EMA10_signal", "MAC_EMA19_SMA24", "MAC_EMA19_SMA24_signal",
               "MAC_neg_EMA11_SMA33", "MAC_neg_EMA11_SMA33_signal", "spike", "upper",
               "lower", "c", "h", "l",
               "hl_spike", "hl_upper", "hl_lower", "returns",
               "MACDhC_8_38_3", "macd_crossover_close",
               "AROONOSC_25", "AROONOSC_155",
               "rlow_25", "rhigh_25", "r_hl_diff_25", "r_hl_since_25", "r_hl_distance_25",
               "rlow_155", "rhigh_155", "r_hl_diff_155", "r_hl_since_155", "r_hl_distance_155"
               ]]
        X_test = df[["ema_crossover_60_95", "ema_crossover_60_95_signal", "MACDhO_11_35_3",
               "MACDhO_10_39_3",
               "RSI_15",
               "RSI_5", "consensus_signal", "MAC_EMA4_EMA10",
               "MAC_EMA4_EMA10_signal", "MAC_EMA19_SMA24", "MAC_EMA19_SMA24_signal",
               "MAC_neg_EMA11_SMA33", "MAC_neg_EMA11_SMA33_signal", "spike", "upper",
               "lower", "c", "h", "l",
               "hl_spike", "hl_upper", "hl_lower", "returns",
               "MACDhC_8_38_3",
               "AROONOSC_25", "AROONOSC_155",
               "rlow_25", "rhigh_25", "r_hl_diff_25", "r_hl_since_25", "r_hl_distance_25",
               "rlow_155", "rhigh_155", "r_hl_diff_155", "r_hl_since_155", "r_hl_distance_155"
               ]]
        """
        X_test = df[["ema_crossover_60_95", "ema_crossover_60_95_signal", "MACDhO_11_35_3",
               "MACDhO_10_39_3",
               "RSI_15",
               "RSI_5", "consensus_signal", "MAC_EMA4_EMA10",
               "MAC_EMA4_EMA10_signal", "MAC_EMA19_SMA24", "MAC_EMA19_SMA24_signal",
               "MAC_neg_EMA11_SMA33", "MAC_neg_EMA11_SMA33_signal",
               "c", "h", "l", "returns",
               "vol_diff_155", "hl_vol_diff_155", "spike_155", "hl_spike_155",
               "vol_diff_35", "hl_vol_diff_35", "spike_35", "hl_spike_35",
               "vol_diff_5", "hl_vol_diff_5", "spike_5", "hl_spike_5",
               # "MACDhC_8_38_3",
               "AROONOSC_25", "AROONOSC_155",
               "rlow_25", "rhigh_25", "r_hl_diff_25", "r_hl_since_25", "r_hl_distance_25",
               "rlow_35", "rhigh_35", "r_hl_diff_35", "r_hl_since_35", "r_hl_distance_35",
               "rlow_155", "rhigh_155", "r_hl_diff_155", "r_hl_since_155", "r_hl_distance_155"
               ]]
        y_test = df["open_signal"]
        #X_test = df_time[["ema_crossover_60_95", "MACDhO_11_35_3",
        #       "MACDhO_10_39_3",
        #       "RSI_15",
        #       "RSI_5", "MAC_EMA4_EMA10",
        #       "MAC_EMA19_SMA24",
        #       "MAC_neg_EMA11_SMA33", "spike", "upper",
        #       "lower", # "c", "h", "l",
        #       "hl_spike", "hl_upper", "hl_lower"
        #       ]]
        y_pred = xgb_model.predict(X_test)
        predictions = [value for value in y_pred]
        self.model = xgb_model

        self.accuracy = metrics.accuracy_score(y_test, predictions)
        self.precision = metrics.precision_score(y_test, predictions, average = "macro")
        self.recall = metrics.recall_score(y_test, predictions, average = "macro")
        self.f1 = metrics.f1_score(y_test, predictions, average = "macro")
        # self.roc_auc = metrics.roc_auc_score(y_test, predictions, average = "weighted", multi_class="ovr")
        print("accuracy: %.2f%%" % (self.accuracy * 100.0), "precision: %.2f%%" % (self.precision * 100.0), "recall: %.2f%%" % (self.recall * 100.0), "f1: %.2f%%" % (self.f1 * 100.0))

        importance = xgb_model.feature_importances_
        self.importance_df = pd.DataFrame(importance, columns=["importance"])
        self.importance_df["features"] = features
        self.importance_df = self.importance_df.sort_values(by="importance", ascending=False)
        self.y_compare = pd.DataFrame({"y_test":y_test, "y_pred":predictions})
        print(self.importance_df)

    def XGBC_predict(self, session):
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
        self.test_df = self.data[(self.data.index >= self.test_range[0]) & (self.data.index <= self.test_range[1])]
        df_time = self.test_df[(pd.to_datetime(self.test_df["time_est"]).dt.time >= pd.to_datetime("{}:30".format(session[0])).time()) & 
                               (pd.to_datetime(self.test_df["time_est"]).dt.time < pd.to_datetime("{}:00".format(session[1])).time())
                               ]
        X_test = df_time[["ema_crossover_60_95", "ema_crossover_60_95_signal", "MACDhO_11_35_3",
               "MACDhO_10_39_3",
               "RSI_15",
               "RSI_5", "consensus_signal", "MAC_EMA4_EMA10",
               "MAC_EMA4_EMA10_signal", "MAC_EMA19_SMA24", "MAC_EMA19_SMA24_signal",
               "MAC_neg_EMA11_SMA33", "MAC_neg_EMA11_SMA33_signal",
               "c", "h", "l", "returns",
               "vol_diff_155", "hl_vol_diff_155", "spike_155", "hl_spike_155",
               "vol_diff_35", "hl_vol_diff_35", "spike_35", "hl_spike_35",
               "vol_diff_5", "hl_vol_diff_5", "spike_5", "hl_spike_5",
               # "MACDhC_8_38_3",
               "AROONOSC_25", "AROONOSC_155",
               "rlow_25", "rhigh_25", "r_hl_diff_25", "r_hl_since_25", "r_hl_distance_25",
               "rlow_35", "rhigh_35", "r_hl_diff_35", "r_hl_since_35", "r_hl_distance_35",
               "rlow_155", "rhigh_155", "r_hl_diff_155", "r_hl_since_155", "r_hl_distance_155"
               ]]
        #X_test = df_time[["ema_crossover_60_95", "MACDhO_11_35_3",
        #       "MACDhO_10_39_3",
        #       "RSI_15",
        #       "RSI_5", "MAC_EMA4_EMA10",
        #       "MAC_EMA19_SMA24",
        #       "MAC_neg_EMA11_SMA33", "spike", "upper",
        #       "lower", # "c", "h", "l",
        #       "hl_spike", "hl_upper", "hl_lower"
        #       ]]
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        y_pred_prob = [max(x) for x in y_proba]
        df = df_time.reset_index(drop=False)
        # df["open_pred"] = y_pred
        df["open_proba"] = y_pred_prob
        # df["open"] = np.where(df["open_pred"] == 2, -1, df["open_pred"])
        df["open_pred"] = [1 if x[1] >= 0.55 else -1 if x[2] >= 0.55 else 0 for x in y_proba]
        df["open"] = df["open_pred"]
        df_time = df.set_index(["time"], drop=True)
        self.test_df = self.test_df.join(df_time[["open_pred", "open", "open_proba"]])
        self.test_df["open"] = np.where(~((pd.to_datetime(self.test_df["time_est"]).dt.time >= pd.to_datetime("{}:30".format(session[0])).time()) & 
            (pd.to_datetime(self.test_df["time_est"]).dt.time < pd.to_datetime("{}:00".format(session[1])).time())), 0, self.test_df["open"])
        self.test_df["open_pred"] = self.test_df["open_pred"].fillna(0)

    def initiate_position(self, stop_params):
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
        df = self.test_df.copy().reset_index()
        df["counter"] = 0
        df["open_price"] = np.nan
        df["open_true"] = np.nan
        df["close_real"] = np.nan
        df["close_price"] = np.nan
        df["close_true"] = np.nan
        df["position"] = np.nan
        df["HL"] = np.nan
        df["LH"] = np.nan
        price_open = {1: "ask", -1: "bid"}
        price_close = {1: "bid", -1: "ask"}
        hl = {1: "l", -1: "h"}
        lh = {1: "h", -1: "l"}
        
        # iterate through positions and define initial close signals
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"position"] = df.loc[row[0],"open"]
                if abs(df.loc[row[0],"open"]) == 1:
                    df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                    df.loc[row[0],"counter"] = 1
            elif abs(df.loc[row[0]-1,"position"]) == 1:
                df.loc[row[0],"HL"] = df.loc[row[0], hl[df.loc[row[0]-1, "position"]]]
                df.loc[row[0],"LH"] = df.loc[row[0], lh[df.loc[row[0]-1, "position"]]]
                df.loc[row[0],"close_real"] = df.loc[row[0], price_close[df.loc[row[0]-1, "position"]]]
                if ((df.loc[row[0], "HL"] - df.loc[row[0]-1, "open_true"]) * df.loc[row[0]-1, "position"]) / df.loc[row[0]-1, "open_true"] * self.leverage < -stop_params[0]:
                    df.loc[row[0],"close_true"] = -stop_params[0]/self.leverage*df.loc[row[0]-1, "open_true"]/df.loc[row[0]-1, "position"] + df.loc[row[0]-1, "open_true"]
                    df.loc[row[0],"position"] = 0
                    df.loc[row[0],"counter"] = 0
                    if abs(df.loc[row[0],"open"]) == 1:
                        df.loc[row[0],"position"] = df.loc[row[0],"open"]
                        df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                        df.loc[row[0],"counter"] = 1
                elif ((df.loc[row[0], "LH"] - df.loc[row[0]-1, "open_true"]) * df.loc[row[0]-1, "position"]) / df.loc[row[0]-1, "open_true"] * self.leverage > stop_params[1]:
                    df.loc[row[0],"close_true"] = stop_params[1]/self.leverage*df.loc[row[0]-1, "open_true"]/df.loc[row[0]-1, "position"] + df.loc[row[0]-1, "open_true"]
                    df.loc[row[0],"position"] = 0
                    df.loc[row[0],"counter"] = 0
                    if abs(df.loc[row[0],"open"]) == 1:
                        df.loc[row[0],"position"] = df.loc[row[0],"open"]
                        df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                        df.loc[row[0],"counter"] = 1
                elif pd.to_datetime(df.loc[row[0],"time"]).tz_convert("America/New_York").time() == pd.to_datetime("16:45").time():
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"position"] = 0
                    df.loc[row[0],"counter"] = 0
                elif abs(df.loc[row[0],"open"]) == 1 and df.loc[row[0]-1,"position"] != df.loc[row[0],"open"]:
                    df.loc[row[0],"position"] = df.loc[row[0],"open"]
                    df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                    df.loc[row[0],"counter"] = 1
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                elif abs(df.loc[row[0],"open"]) == 1 and df.loc[row[0]-1,"position"] == df.loc[row[0],"open"]:
                    df.loc[row[0],"position"] = df.loc[row[0]-1,"position"]
                    df.loc[row[0],"open_true"] = df.loc[row[0]-1,"open_true"]
                    df.loc[row[0],"counter"] = 1
                elif df.loc[row[0]-1,"counter"] == 12:
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"position"] = 0
                    df.loc[row[0],"counter"] = 0
                else:
                    df.loc[row[0],"position"] = df.loc[row[0]-1,"position"]
                    df.loc[row[0],"open_true"] = df.loc[row[0]-1,"open_true"]
                    df.loc[row[0],"counter"] += 1
            elif abs(df.loc[row[0],"open"]) == 1:
                df.loc[row[0],"position"] = df.loc[row[0],"open"]
                df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                df.loc[row[0],"counter"] = 1
            else:
                df.loc[row[0],"position"] = df.loc[row[0]-1,"position"]

        self.test_df = self.test_df.join(df.set_index("time")[["open_true", "close_true", "position", "HL", "LH"]])

    def update_strategy(self):
        """ Updates cumulative strategy returns based on open and close prices.
        Parameters
        ==========
        real_return: boolean
            determine whether to calculate cumulative strategy returns including trading costs
        """
        df = self.test_df.copy().reset_index()
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
        self.test_df = self.test_df.join(df.set_index("time")[["trades", "cstrategy", # "creturns", 
                                   "win", "trade_return"]])

    def backtest(self):
        self.feature_engineering((60,95), (11,35,3), (15,5), (4,10,19,24), (155, 35, 5), (1.0, 4.7), (10,39,3), (11,33), (8,38,3), (25, 155), (25, 35, 155))
        df = self.data.copy()
        df = df[df["optimal_time7"]==1]
        df = df[(df.index >= self.train_range[0]) & (df.index <= self.train_range[1])]
        y = df["open_signal"]
        X = df[["ema_crossover_60_95", "ema_crossover_60_95_signal", "MACDhO_11_35_3",
               "trend_macd_crossover_signal_1", "MACDhO_10_39_3",
               "trend_macd_crossover_signal_2", "RSI_15", "trend_rsi_15_signal",
               "RSI_5", "trend_rsi_5_signal", "consensus_signal", "MAC_EMA4_EMA10",
               "MAC_EMA4_EMA10_signal", "MAC_EMA19_SMA24", "MAC_EMA19_SMA24_signal",
               "MAC_neg_EMA11_SMA33", "MAC_neg_EMA11_SMA33_signal", "spike", "upper",
               "lower", "c", "h", "l",
               "hl_spike", "hl_upper", "hl_lower", "returns",
               "MACDhC_8_38_3", "macd_crossover_close"
               ]]
        X = df[["ema_crossover_60_95", "ema_crossover_60_95_signal", "MACDhO_11_35_3",
               "MACDhO_10_39_3",
               "RSI_15",
               "RSI_5", "consensus_signal", "MAC_EMA4_EMA10",
               "MAC_EMA4_EMA10_signal", "MAC_EMA19_SMA24", "MAC_EMA19_SMA24_signal",
               "MAC_neg_EMA11_SMA33", "MAC_neg_EMA11_SMA33_signal",
               "c", "h", "l", "returns",
               "vol_diff_155", "hl_vol_diff_155", "spike_155", "hl_spike_155",
               "vol_diff_35", "hl_vol_diff_35", "spike_35", "hl_spike_35",
               "vol_diff_5", "hl_vol_diff_5", "spike_5", "hl_spike_5",
               # "MACDhC_8_38_3",
               "AROONOSC_25", "AROONOSC_155",
               "rlow_25", "rhigh_25", "r_hl_diff_25", "r_hl_since_25", "r_hl_distance_25",
               "rlow_35", "rhigh_35", "r_hl_diff_35", "r_hl_since_35", "r_hl_distance_35",
               "rlow_155", "rhigh_155", "r_hl_diff_155", "r_hl_since_155", "r_hl_distance_155"
               ]]
        #X = df[["ema_crossover_60_95", "MACDhO_11_35_3",
        #       "MACDhO_10_39_3",
        #       "RSI_15",
        #       "RSI_5", "MAC_EMA4_EMA10",
        #       "MAC_EMA19_SMA24",
        #       "MAC_neg_EMA11_SMA33", "spike", "upper",
        #       "lower", # "c", "h", "l",
        #       "hl_spike", "hl_upper", "hl_lower"
        #       ]]
        features = X.columns.tolist()
        self.XGBC_train(X, y, features, self.test_range, ("02", "12"))
        self.XGBC_predict(self.test_range, ("02", "12"))
        self.initiate_position(stop_params=(0.05, 0.2))
        # calculate cumulative strategy returns
        self.update_strategy()
        # self.results = self.test_df.copy().drop(["o", "h", "l", "ask", "bid", "spread", "recentPL", "recentPH"], axis=1)

        self.test_df["accuracy"] = np.where(self.test_df["open_pred"] == 0, np.nan, np.where(self.test_df["open_pred"] == self.test_df["open_signal"], 1, 0))
        self.pred_accuracy = self.test_df["accuracy"].mean()
        # self.hold = self.test_df["creturns"].iloc[-1] # absolute performance of buy and hold strategy
        self.perf = self.test_df["cstrategy"].iloc[-1] # absolute performance of the strategy
        self.max_return = self.test_df["cstrategy"].max()
        # self.outperf = self.perf - self.hold # out-/underperformance of strategy
        self.trades = self.test_df["trades"].sum()
        self.wins = self.test_df[self.test_df["win"] > 0]["win"].sum()
        returns_pos = self.test_df[self.test_df["trade_return"] > 0]["trade_return"].mean()
        returns_neg = self.test_df[self.test_df["trade_return"] < 0]["trade_return"].mean()
        self.accuracy = self.wins / self.test_df["win"].abs().sum() if self.test_df["win"].abs().sum() > 0 else 1.0
        results =  {
                    "test_range": self.test_range, "pred_accuracy": self.pred_accuracy, "trades": self.trades, "accuracy": self.accuracy, "strat_return": round(self.perf, 6), 
                    "max_return": round(self.max_return, 6), "returns_pos": round(returns_pos, 6), "returns_neg": round(returns_neg, 6), "score": round(self.perf*self.accuracy, 6)
                    }
        print(results)
        self.cv_results = pd.DataFrame([results])
        self.cv_results = self.cv_results.set_index("test_range")

if __name__ == "__main__":

    trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", "EUR_USD", "M15", "2002-06-01", "2023-02-03", 50, ("2002-06-01", "2022-10-22"), ("2022-10-23", "2023-02-03"), 12, 0.20, 0.05)
    trader.backtest()
    trader.test_df.to_csv(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\xgboost_returns.csv", mode="w", index=True, header=True)
    trader.cv_results.to_csv(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\xgboost_agg_results.csv", mode="a", index=True, header=False)

