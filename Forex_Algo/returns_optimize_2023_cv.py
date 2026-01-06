import pandas as pd
import pandas_ta as ta
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import tpqoa
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from itertools import product


class TrendAlgoBacktester(tpqoa.tpqoa):
    """ Class for the vectorized backtesting trading strategies.
    """
    def __init__(self, conf_file, instrument, bar_length, start, end, leverage):
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
        # self.ratio = ratio
        self.results = None
        self.perf = None
        self.optimization_results = None
        self.max_return = None
        self.trades = None
        self.wins = None
        self.accuracy = None
        self.cv_results = pd.DataFrame()
        self.date_brackets = None
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
        df["ema_crossover_{}_{}".format(short, longg)] = df["ema_{}".format(short)] - df["ema_{}".format(longg)]
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

    def macd_crossover_close(self, short, longg, signal, threshold # length
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
        df["macd_crossover_close_{}".format(signal)] = np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)]/df["c"] > threshold, 1,
                                               np.where(df["MACDh_{}_{}_{}".format(short, longg, signal)]/df["c"] < -threshold, -1, 0)
                                               )
        df["MACDhC_{}_{}_{}".format(short, longg, signal)] = df["MACDh_{}_{}_{}".format(short, longg, signal)]
        # df["macd_distance"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].abs()
        # df["macd_change"] = np.where(np.sign(df["MACDh_{}_{}_{}".format(short, longg, signal)]).diff().ne(0), 0, df["macd_distance"] - df["macd_distance"].shift(1))
        # df["macd_std"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].rolling(length, min_periods=length).std()
        # df["macd_upper"] = df["MACDh_{}_{}_{}".format(short, longg, signal)].rolling(length, min_periods=length).std() * -1
        self.data = self.data.join(df[["MACDhC_{}_{}_{}".format(short, longg, signal), 
                                       "macd_crossover_close_{}".format(signal) # , "macd_distance", "macd_change", "macd_std"
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
        df["MAC_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] = df["mac_short"] - df["mac_long"]
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
        df["MAC_neg_{}{}_{}{}".format(short[1], short[0], longg[1], longg[0])] = df["mac_short"] - df["mac_long"]
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
        df["vol_signal_{}".format(length)] = np.where((df["spike_{}".format(length)] > 0) & (df["spike_{}".format(length)] > multiplier[0]*df["upper_{}".format(length)]) & (df["spike_{}".format(length)]/df["upper_{}".format(length)] < multiplier[1]), 1,
                                                      np.where((df["spike_{}".format(length)] < 0) & (df["spike_{}".format(length)] < multiplier[0]*df["lower_{}".format(length)]) & (df["spike_{}".format(length)]/df["lower_{}".format(length)] < multiplier[1]), -1, 0)
                                                      )
        self.data = self.data.join(df[["spike_{}".format(length), "upper_{}".format(length), "lower_{}".format(length), "vol_signal_{}".format(length)]])

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

    def initiate_position(self, close_out, swing_stop, risk_stop, session, hour_time, stop_params, mar):
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
        df = self.data.copy().reset_index()
        df["open_price"] = np.nan
        df["open_true"] = np.nan
        df["close_real"] = np.nan
        df["close_price"] = np.nan
        df["close_true"] = np.nan
        df["position"] = np.nan
        df["recentPHL"] = np.nan
        df["max_active_return"] = np.nan
        df["HL"] = np.nan
        price_open = {1: "ask", -1: "bid"}
        price_close = {1: "bid", -1: "ask"}
        phl = {1: "recentPL", -1: "recentPH"}
        hl = {1: "l", -1: "h"}
        lh = {1: "h", -1: "l"}
        
        # iterate through positions and define initial close signals
        for row in df.iterrows():
            if row[0] == 0:
                df.loc[row[0],"position"] = df.loc[row[0],"open"]
                if abs(df.loc[row[0],"open"]) == 1:
                    df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
            elif abs(df.loc[row[0]-1,"position"]) == 1:
                df.loc[row[0],"recentPHL"] = df.loc[row[0], phl[df.loc[row[0]-1, "position"]]]
                df.loc[row[0],"HL"] = df.loc[row[0], hl[df.loc[row[0]-1, "position"]]]
                df.loc[row[0],"LH"] = df.loc[row[0], lh[df.loc[row[0]-1, "position"]]]
                df.loc[row[0], "max_active_return"] = ((df.loc[row[0], "LH"] - df.loc[row[0]-1, "open_true"]) * df.loc[row[0]-1, "position"]) / df.loc[row[0]-1, "open_true"] * self.leverage
                df.loc[row[0], "max_active_return"] = df.loc[row[0], "max_active_return"] if df.loc[row[0]-1, "max_active_return"] == np.nan else df.loc[row[0], "max_active_return"] if df.loc[row[0], "max_active_return"] > df.loc[row[0]-1, "max_active_return"] else df.loc[row[0]-1, "max_active_return"]
                df.loc[row[0],"close_real"] = df.loc[row[0], price_close[df.loc[row[0]-1, "position"]]]
                if close_out and ((df.loc[row[0], "HL"] - df.loc[row[0]-1, "open_true"]) * df.loc[row[0]-1, "position"]) / df.loc[row[0]-1, "open_true"] * self.leverage < -stop_params[1]:
                    df.loc[row[0],"close_true"] = -stop_params[1]/self.leverage*df.loc[row[0]-1, "open_true"]/df.loc[row[0]-1, "position"] + df.loc[row[0]-1, "open_true"]
                    df.loc[row[0],"position"] = 0
                    if abs(df.loc[row[0],"open"]) == 1:
                        df.loc[row[0],"position"] = df.loc[row[0],"open"]
                        df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                elif pd.to_datetime(df.loc[row[0],"time"]).tz_convert("America/New_York").time() == pd.to_datetime("16:45").time():
                    # df.loc[row[0],"close_true"] = df.loc[row[0], price_close[df.loc[row[0]-1, "position"]]]
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"position"] = 0
                elif pd.to_datetime(df.loc[row[0],"time"]).tz_convert("America/New_York").time() == pd.to_datetime("{}:{}".format(session[2], hour_time[2])).time():
                    # df.loc[row[0],"close_true"] = df.loc[row[0], price_close[df.loc[row[0]-1, "position"]]]
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"position"] = 0
                elif abs(df.loc[row[0],"open"]) == 1 and df.loc[row[0]-1,"position"] != df.loc[row[0],"open"]:
                    df.loc[row[0],"position"] = df.loc[row[0],"open"]
                    df.loc[row[0],"open_true"] = df.loc[row[0], price_open[df.loc[row[0], "position"]]]
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                elif abs(df.loc[row[0],"open"]) == 1 and df.loc[row[0]-1,"position"] == df.loc[row[0],"open"]:
                    df.loc[row[0],"position"] = df.loc[row[0]-1,"position"]
                    df.loc[row[0],"open_true"] = df.loc[row[0]-1,"open_true"]
                elif risk_stop and ((df.loc[row[0], "close_real"] - df.loc[row[0]-1, "open_true"]) * df.loc[row[0]-1, "position"]) / df.loc[row[0]-1, "open_true"] * self.leverage < -stop_params[0] and df.loc[row[0],"macd_crossover_close_2"] != df.loc[row[0]-1,"position"]:
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"position"] = 0
                elif df.loc[row[0],"macd_crossover_close_2"] != df.loc[row[0]-1,"position"] and df.loc[row[0], "max_active_return"] > mar:
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"position"] = 0
                elif df.loc[row[0],"macd_crossover_close_3"] != df.loc[row[0]-1,"position"] and (df.loc[row[0], "close_real"] - df.loc[row[0]-1, "open_true"])*df.loc[row[0]-1,"position"] > 0:
                    df.loc[row[0],"close_true"] = df.loc[row[0],"close_real"]
                    df.loc[row[0],"position"] = 0
                elif swing_stop and (df.loc[row[0],"c"] - df.loc[row[0],"recentPHL"]) * df.loc[row[0]-1, "position"] < 0:
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
        self.data = self.data.join(df.set_index("time")[["open_true", "close_true", "position", "recentPHL", "HL"]])

    def update_strategy(self):
        """ Updates cumulative strategy returns based on open and close prices.
        Parameters
        ==========
        real_return: boolean
            determine whether to calculate cumulative strategy returns including trading costs
        """
        df = self.data.copy().reset_index()
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
        self.data = self.data.join(df.set_index("time")[["trades", "cstrategy", # "creturns", 
                                   "win", "trade_return"]])

    # def optimize_strategy(self, emac, macdc, macdcf, macdcc, rsi, mac, vol_length, swing, session, risk_param, close_param, start = None, end = None):
    def optimize_strategy(self, date_range, emac, macdc, macdcc1, macdcc2, rsi, mac, vol_length, swing, session, hour_time, stop_params, multiplier, macdc2, mar# , mac_neg
                          ):
        """ Backtests the trading strategy.
        """
        time.sleep(1)      

        self.data = self.raw_data[(self.raw_data.index >= date_range[1]) & (self.raw_data.index <= date_range[2])]

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
        # self.ma_crossover_neg(short=(mac_neg[0], "EMA"), longg=(mac_neg[1], "SMA"))
        self.volatility_osc(length=vol_length[0], multiplier=multiplier)
        # self.volatility_osc(length=vol_length[1], multiplier=multiplier)
        self.swing(left=swing[0], right=swing[1], high_low=True)
        self.macd_crossover_close(short=macdcc1[0], longg=macdcc1[1], signal=macdcc1[2], threshold=macdcc1[3])
        self.macd_crossover_close(short=macdcc2[0], longg=macdcc2[1], signal=macdcc2[2], threshold=macdcc2[3])
        # ************************************************************************

        self.data = self.data.copy().dropna()
        self.data["open"] = np.where(self.data.filter(regex=("signal")).mean(axis=1).abs() == 1.0, self.data.filter(regex=("signal")).mean(axis=1), 0)
        self.data["time_close"] = self.data.index.tz_convert("America/New_York")
        if int(session[0]) >= 17:
            self.data["open"] = np.where(~((pd.to_datetime(self.data["time_close"]).dt.time >= pd.to_datetime("{}:{}".format(session[0], hour_time[0])).time()) | 
                  (pd.to_datetime(self.data["time_close"]).dt.time < pd.to_datetime("{}:{}".format(session[1], hour_time[1])).time())), 0, self.data["open"]
            )
        if int(session[0]) < 17:
            self.data["open"] = np.where(~((pd.to_datetime(self.data["time_close"]).dt.time >= pd.to_datetime("{}:{}".format(session[0], hour_time[0])).time()) & 
                (pd.to_datetime(self.data["time_close"]).dt.time < pd.to_datetime("{}:{}".format(session[1], hour_time[1])).time())), 0, self.data["open"]
            )

        self.initiate_position(close_out=True, swing_stop=True, risk_stop=True, session=session, hour_time=hour_time, stop_params=stop_params, mar=mar)
        
        # calculate cumulative strategy returns
        self.update_strategy()
        self.results = self.data.copy().drop(["o", "h", "l", "ask", "bid", "spread", "recentPL", "recentPH"], axis=1)
       
        # self.hold = self.data["creturns"].iloc[-1] # absolute performance of buy and hold strategy
        self.perf = self.data["cstrategy"].iloc[-1] # absolute performance of the strategy
        self.max_return = self.data["cstrategy"].max()
        # self.outperf = self.perf - self.hold # out-/underperformance of strategy
        self.trades = self.data["trades"].sum()
        self.wins = self.data[self.data["win"] > 0]["win"].sum()
        returns_pos = self.data[self.data["trade_return"] > 0]["trade_return"].mean()
        returns_neg = self.data[self.data["trade_return"] < 0]["trade_return"].mean()
        self.accuracy = self.wins / self.data["win"].abs().sum() if self.data["win"].abs().sum() > 0 else 1.0
        return {"params": (emac, macdc, macdcc1, macdcc2, rsi, mac, vol_length, swing, session, hour_time, stop_params, multiplier, macdc2, mar# , mac_neg
                           ), 
                f"wins{date_range[0]}": self.wins, f"trades{date_range[0]}": self.trades, f"accuracy{date_range[0]}": self.accuracy, # round(self.hold, 6), 
                f"strat_return{date_range[0]}": round(self.perf, 6) # , round(self.outperf, 6)
                , f"max_return{date_range[0]}": round(self.max_return, 6), f"returns_pos{date_range[0]}": round(returns_pos, 6), f"returns_neg{date_range[0]}": round(returns_neg, 6)
                , f"score{date_range[0]}": round(self.perf*self.accuracy, 6), "index": date_range[0]
                }

    def parallel_optimize(self):

        # *****************set emac parameters*****************

        # emac = [(50, 100), (50, 200), (70, 95)]
        # emac = [(70, 95)]
        # emac = [(50,200), (150,220), (170,200), (160,210), (180,190)]
        # emac = product(range(50,115,5), range(80,215,5))
        # emac = product(range(50,115,5), range(70,145,5))
        # emac = [(x[0], x[1]) for x in emac if x[1] > x[0]]
        # emac = [(70,95), (75,90), (65,105), (65,100), (50,115), (60,110), (70,80)]
        # emac = [(70,95), (75,90), (65,105), (65,100)]
        # emac = [(70,95), (60, 95), (65,90)]
        emac = [(60,95)]

        # *****************set macdc parameters*****************

        # macdc = product([9], [28], [3])
        # macdc = product(range(9,16,1), [25,26,27,28,29,34,35], [3,4,5])
        # macdc = [(x[0], x[1], x[2]) for x in macdc if x[1] > x[0]]
        # macdc = [(11,35,3), (9,28,3), (15,27,3), (15,25,3), (14,26,3), (12,29,3), (13,28,3), (10,34,3), (10,25,3), (14,25,3), (14,29,3)]
        # macdc = [(11,35,3), (9,28,3), (15,27,3), (12,29,3), (10,25,3)]
        # macdc = [(9,28,3), (11,35,3), (15,27,3), (15,25,3), (14,26,3)]
        # macdc = [(9,28,3), (11,35,3), (15,27,3)]
        # macdc = [(15,27,3)]
        # macdc = [(11,35,3), (15,27,3), (12,29,3), (9,28,3)]
        # macdc = [(11,35,3), (15,27,3), (11,35,2), (15,27,2)]
        # macdc = [(11,35,3), (15,27,3)]
        macdc = [(11,35,3)]

        # *****************set macdcc parameters*****************

        # macdcc = product([9], [28], [3])
        # macdcc = product(range(5,105,10), range(5,105,10), [2,3,5,8,10])
        # macdcc = product(range(5,20,1), range(6,51,1), [3,5,8,9,12])
        # macdcc = product(range(7,15,1), range(10,56,1), range(2,15,1))
        # macdcc = [(x[0], x[1], x[2]) for x in macdcc if x[1] > x[0]]
        # macdcc = [(8,38,3), (9,32,3), (8,39,3), (9,31,3), (7,48,3), (7,49,3), (8,15,9), (9,15,8), (8,40,3), (8,41,3), (13,26,3), (9,14,8), (7,12,12), (8,14,9), (10,29,3), (7,50,3), (16,20,3), (10,27,3)]
        # macdcc = [(8,38,3), (9,32,3), (8,39,3), (9,31,3)]
        # macdcc = [(8,38,3), (9,32,3), (8,39,3)]
        # macdcc = [(8,38,3), (9,32,3)]
        # macdcc = [(8,38,3)]
        # macdcc = [(11,35,3), (9,28,3), (15,27,3), (15,25,3), (14,26,3), (12,29,3), (13,28,3), (10,34,3), (10,25,3), (14,25,3), (14,29,3), (8,38,3), (9,32,3)]
        # macdcc = [(9,28,3), (8,38,3), (10,25,3), (9,28,2), (8,38,2), (10,25,2)]
        # macdcc = [(9,28,3), (8,38,3), (10,25,3)]
        # macdcc = [(8,38,3), (9,28,3), (11,35,3), (15,27,3), (15,25,3), (14,26,3)]
        # macdcc = [(8,38,3), (9,28,3), (11,35,3), (15,27,3)]
        # macdcc = [(8,38,3), (9,28,3)]
        # macdcc = [(8,38,3), (9,28,3), (11,35,3)]
        # macdcc = [(11,35,3), (15,27,3), (15,25,3), (9,28,3)]
        # macdcc = [(11,35,3), (15,27,3)]
        # macdcc = [(11,35,3)]
        # macdcc = macdc
        # macdcc = [(8,38,3), (9,32,3), (8,39,3), (9,31,3), (11,35,3), (15,27,3)]
        macdcc1 = product([8], [38], [3], range(0,300,10))
        macdcc1 = [(x[0], x[1], x[2], x[3]/10000000) for x in macdcc1 if x[1] > x[0]]

        macdcc2 = product([8], [38], [2], range(0,300,10))
        macdcc2 = [(x[0], x[1], x[2], x[3]/10000000) for x in macdcc2 if x[1] > x[0]]

        # *****************set rsi parameters*****************

        # rsi = product([12,13,14,15,16], [5,7,9,11,13])
        # rsi = product([10,13,15,17], [5,7,9,11,13])
        # rsi = product(range(10,21,1), range(3,16,1))
        # rsi = [(x[0], x[1]) for x in rsi if x[0] > x[1]]
        # rsi = [(15,5), (13,7), (13,5), (15,7), (15,8), (15,6), (13,8), (13,6), (14,5), (12,5), (14,7), (12,7), (14,6), (12,6), (14,8), (12,8), (12,4), (13,4), (14,4), (15,4)]
        # rsi = [(15,5), (12,4), (12,5), (13,5), (14,5)]
        # rsi = [(15,5), (15,4), (10,5), (10,4)]
        # rsi = [(15,5), (15,11), (15,13)]
        # rsi = product(range(10,21,1), range(3,16,1))
        # rsi = [(x[0], x[1]) for x in rsi if x[0] > x[1]]
        # rsi = [(15,5), (14,5), (13,5), (12,5), (11,5), (10,5), (9,5), (8,5), (7,5), (6,5)]
        rsi = [(15,5)]

        # *****************set mac parameters*****************

        # mac = product([5,6,7,8,9], [6,7,8,9,10], [15,18,19,20,21,25,30], [20,24,25,26,30,35,40])
        # mac = product([4,5,6,7,8,9], [6,7,8,9,10], [15,18,20,21,25,30], [20,25,26,30,35,40])
        # mac = product([4,5,6], [6,8,10,15], [15,20,25,30], [20,25,30,35])
        # mac = product([4,5,6], [6,7,8,9,10], [20], [25])
        # mac = [(x[0], x[1], x[2], x[3]) for x in mac if x[0] < x[1] < x[2] < x[3]]
        """
        mac = [(3,5,20,25), (4,5,20,25), (5,7,20,25), (4,7,20,25), (3,8,20,25), (4,12,20,25)
               , (3,5,17,23), (4,5,17,23), (5,7,17,23), (4,7,17,23), (3,8,17,23), (4,12,17,23)
               , (3,5,20,23), (4,5,20,23), (5,7,20,23), (4,7,20,23), (3,8,20,23), (4,12,20,23)
               , (3,5,20,27), (4,5,20,27), (5,7,20,27), (4,7,20,27), (3,8,20,27), (4,12,20,27)
               , (3,5,17,27), (4,5,17,27), (5,7,17,27), (4,7,17,27), (3,8,17,27), (4,12,17,27)

               , (5,15,20,25), (5,15,17,23), (5,15,20,23), (5,15,20,27), (5,15,17,27)
               , (6,10,20,25), (6,10,17,23), (6,10,20,23), (6,10,20,27), (6,10,17,27)
               , (4,10,20,25), (4,10,17,23), (4,10,20,23), (4,10,20,27), (4,10,17,27)

               , (4,12,20,23), (5,7,20,23), (3,8,20,23), (4,10,20,23), (3,6,20,23), (3,7,20,23)
               , (4,12,17,20), (5,7,17,20), (3,8,17,20), (4,10,17,20), (3,6,17,20), (3,7,17,20)
               , (4,12,22,25), (5,7,22,25), (3,8,22,25), (4,10,22,25), (3,6,22,25), (3,7,22,25)
               , (4,12,13,15), (5,7,13,15), (3,8,13,15), (4,10,13,15), (3,6,13,15), (3,7,13,15)

               , (4,12,20,23), (5,7,20,23), (3,8,20,23), (4,10,20,23), (3,6,20,23), (3,7,20,23), (3,12,20,23), (5,11,20,23), (5,10,20,23), (4,11,20,23), (5,12,20,23)
               , (4,12,21,24), (5,7,21,24), (3,8,21,24), (4,10,21,24), (3,6,21,24), (3,7,21,24), (3,12,21,24), (5,11,21,24), (5,10,21,24), (4,11,21,24), (5,12,21,24)
               , (4,12,19,24), (5,7,19,24), (3,8,19,24), (4,10,19,24), (3,6,19,24), (3,7,19,24), (3,12,19,24), (5,11,19,24), (5,10,19,24), (4,11,19,24), (5,12,19,24)
               , (4,12,18,23), (5,7,18,23), (3,8,18,23), (4,10,18,23), (3,6,18,23), (3,7,18,23), (3,12,18,23), (5,11,18,23), (5,10,18,23), (4,11,18,23), (5,12,18,23)
               , (4,12,18,24), (5,7,18,24), (3,8,18,24), (4,10,18,24), (3,6,18,24), (3,7,18,24), (3,12,18,24), (5,11,18,24), (5,10,18,24), (4,11,18,24), (5,12,18,24)
               , (4,12,18,22), (5,7,18,22), (3,8,18,22), (4,10,18,22), (3,6,18,22), (3,7,18,22), (3,12,18,22), (5,11,18,22), (5,10,18,22), (4,11,18,22), (5,12,18,22)

               , (4,12,20,24), (5,7,20,24), (3,8,20,24), (4,10,20,24), (3,6,20,24), (3,7,20,24), (3,12,20,24), (5,11,20,24), (5,10,20,24), (4,11,20,24), (5,12,20,24), (4,8,20,24), (3,7,20,24), (4,15,20,24)
               , (4,12,23,27), (5,7,23,27), (3,8,23,27), (4,10,23,27), (3,6,23,27), (3,7,23,27), (3,12,23,27), (5,11,23,27), (5,10,23,27), (4,11,23,27), (5,12,23,27), (4,8,23,27), (3,7,23,27), (4,15,23,27)
               , (4,12,26,30), (5,7,26,30), (3,8,26,30), (4,10,26,30), (3,6,26,30), (3,7,26,30), (3,12,26,30), (5,11,26,30), (5,10,26,30), (4,11,26,30), (5,12,26,30), (4,8,26,30), (3,7,26,30), (4,15,26,30)
               , (4,12,20,23), (5,7,20,23), (3,8,20,23), (4,10,20,23), (3,6,20,23), (3,7,20,23), (3,12,20,23), (5,11,20,23), (5,10,20,23), (4,11,20,23), (5,12,20,23), (4,8,20,23), (3,7,20,23), (4,15,20,23)

               , (4,8,20,23), (3,7,20,23), (4,15,20,23)
               , (4,8,19,24), (3,7,19,24), (4,15,19,24)
               , (4,8,22,25), (3,7,22,25), (4,15,22,25)
               , (4,8,21,24), (3,7,21,24), (4,15,21,24)

               , (3,11,20,23), (3,9,20,23), (5,9,20,23), (4,13,20,23), (5,13,20,23), (3,13,20,23)
               , (3,11,19,24), (3,9,19,24), (5,9,19,24), (4,13,19,24), (5,13,19,24), (3,13,19,24)
               , (3,11,20,24), (3,9,20,24), (5,9,20,24), (4,13,20,24), (5,13,20,24), (3,13,20,24)
               , (3,11,21,24), (3,9,21,24), (5,9,21,24), (4,13,21,24), (5,13,21,24), (3,13,21,24)
               , (3,11,20,25), (3,9,20,25), (5,9,20,25), (4,13,20,25), (5,13,20,25), (3,13,20,25)
               , (3,11,20,27), (3,9,20,27), (5,9,20,27), (4,13,20,27), (5,13,20,27), (3,13,20,27)
               , (3,11,18,24), (3,9,18,24), (5,9,18,24), (4,13,18,24), (5,13,18,24), (3,13,18,24)

               , (4,14,20,23), (4,15,20,23), (4,16,20,23), (4,17,20,23)
               , (4,14,19,24), (4,15,19,24), (4,16,19,24), (4,17,19,24)
               , (4,14,21,24), (4,15,21,24), (4,16,21,24), (4,17,21,24)
               , (4,14,22,25), (4,15,22,25), (4,16,22,25), (4,17,22,25)

               , (5,14,20,23), (5,15,20,23), (5,16,20,23), (5,17,20,23)
               , (5,14,19,24), (5,15,19,24), (5,16,19,24), (5,17,19,24)
               , (5,14,21,24), (5,15,21,24), (5,16,21,24), (5,17,21,24)
               , (5,14,22,25), (5,15,22,25), (5,16,22,25), (5,17,22,25)

               , (3,15,20,23), (3,15,20,23), (3,16,20,23), (3,17,20,23)
               , (3,15,19,24), (3,15,19,24), (3,16,19,24), (3,17,19,24)
               , (3,15,21,24), (3,15,21,24), (3,16,21,24), (3,17,21,24)
               , (3,15,22,25), (3,15,22,25), (3,16,22,25), (3,17,22,25)

               , (2,15,20,23), (2,15,20,23), (2,16,20,23), (2,17,20,23)
               , (2,15,19,24), (2,15,19,24), (2,16,19,24), (2,17,19,24)
               , (2,15,21,24), (2,15,21,24), (2,16,21,24), (2,17,21,24)
               , (2,15,22,25), (2,15,22,25), (2,16,22,25), (2,17,22,25)

               , (2,8,20,23), (2,8,20,23), (2,8,20,23), (2,8,20,23)
               , (2,10,19,24), (2,10,19,24), (2,10,19,24), (2,10,19,24)
               , (2,11,21,24), (2,11,21,24), (2,11,21,24), (2,11,21,24)
               , (2,12,22,25), (2,12,22,25), (2,12,22,25), (2,12,22,25)
               ]
        """
        """
        mac = [
            (4,12,20,23), (4,11,20,23), (4,13,20,23), (4,10,20,23), (5,10,20,23), (3,14,20,23), (3,15,20,23), (3,16,20,23), (3,17,20,23)
            , (4,12,19,24), (4,11,19,24), (4,13,19,24), (4,10,19,24), (5,10,19,24), (3,14,19,24), (3,15,19,24), (3,16,19,24), (3,17,19,24)
            , (4,12,22,25), (4,11,22,25), (4,13,22,25), (4,10,22,25), (5,10,22,25), (3,14,22,25), (3,15,22,25), (3,16,22,25), (3,17,22,25), (3,20,22,25)
            , (4,12,20,25), (4,11,20,25), (4,13,20,25), (4,10,20,25), (5,10,20,25), (3,14,20,25), (3,15,20,25), (3,16,20,25), (3,17,20,25)
            , (4,12,24,28), (4,11,24,28), (4,13,24,28), (4,10,24,28), (5,10,24,28), (3,14,24,28), (3,15,24,28), (3,16,24,28), (3,17,24,28), (3,20,24,28)
            ]
        """
        """
        mac = [(4,10,20,25), (4,10,20,23), (4,10,19,24), (4,10,22,25)
                , (4,12,20,25), (4,12,20,23), (4,12,19,24), (4,12,22,25)
                , (3,15,20,25), (3,15,20,23), (3,15,19,24), (3,15,22,25)
                , (5,6,20,25), (5,6,20,23), (5,6,19,24), (5,6,22,25)
               ]
        """
        # mac = [(5,6,17,23), (5,6,20,25), (5,6,21,24)]
        # mac = [(5,6,20,25), (4,10,20,25), (4,8,20,25), (4,9,20,25), (5,6,22,25), (4,10,22,25)]
        # mac = [(4,10,20,25), (5,6,20,25), (4,7,20,25), (4,8,20,25), (4,9,20,25)]
        # mac = [(4,10,20,25), (5,6,20,25), (4,7,20,25), (4,8,20,25), (4,9,20,25)]
        # mac = [(4,10,20,25), (5,6,20,25)]
        # mac = [(5,11,21,36)]
        # mac = product(range(4,10,1), range(5,15,1), range(19,31,1), range(23,41,2))
        # mac = [(x[0], x[1], x[2], x[3]) for x in mac if x[0] < x[1] < x[2] < x[3]]
        # mac = [(4,7,20,25), (4,8,20,25), (5,6,20,25), (4,9,20,25), (4,10,20,25)]
        # mac = product([2,3,4,5], [6,7,8,9,10,11],[20,22],[25])
        # mac = [(x[0], x[1], x[2], x[3]) for x in mac if x[0] < x[1] < x[2] < x[3]]
        # mac = [(5,6,20,25), (4,10,20,25)]
        # mac = [(5,6,20,25)]
        # mac = [(4,10,20,25)]
        # mac = [(4,7,20,25), (4,8,20,25)]
        # mac = [(4,7,20,25)]
        # mac = [(4,8,20,25), (5,6,20,25), (4,9,20,25), (4,10,20,25)]
        # mac = [(5,6,20,25), (4,10,20,25)]
        # mac = [(4,10,20,25)]
        # mac = [(4,7,20,25), (4,8,20,25), (4,10,20,25)]
        # mac = [(5,6,20,25), (4,8,20,25), (4,10,20,25)]
        # mac = [(4,8,20,25), (4,10,20,25)]
        # mac = [(4,10,19,24), (5,6,19,24), (4,5,19,24)]
        mac = [(4,10,19,24)]

        # mac_neg = product(range(3,21,1), range(20,51,1))
        # mac_neg = [(x[0], x[1]) for x in mac_neg if x[0] < x[1]]

        # *****************set vol_length parameters*****************

        # vol_length = [100,125,130,145,150,155,160,175,180,200,210]
        # vol_length = [100,125,130,145,150,155,160,175,180,200]
        # vol_length = [100,125,150,155,160,180,200]
        # vol_length = [155]
        # vol_length = product([155], range(3,151,1))
        # vol_length = [(x[0], x[1]) for x in vol_length if x[1] < x[0]]
        vol_length = [(155,25)]

        # *****************set swing parameters*****************

        # swing = product([9], [9])
        # swing = [(x[0], x[1]) for x in swing]
        # swing = [(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10)]
        # swing = [(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]
        swing = [(8,8)]

        # session_end = ["20"]
        # hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
        # hours = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
        # session = product(hours, hours)
        # session = product(["00", "01", "02", "03", "04", "05", "06", "07"], ["17", "18", "19", "20", "21", "22", "23"])
        # session = product(["02", "04", "05"], ["17", "18", "19"])
        # session = product(["02", "03", "04", "05", "06", "07", "08"], ["16", "17", "18", "19", "20", "21", "22"])
        # session = product(["00", "01", "02", "03", "04", "05"], ["17", "18", "19", "20", "21"])
        # session = [(x[0], x[1]) for x in session if x[0] < x[1]]
        # session = [("00", "13"), ("23", "13"), ("23", "14"), ("00", "14")]
        # session = [("23", "13")]
        # session = product(["01", "02", "03", "04", "05"], ["17", "18", "19", "20"])
        # session = product(["19", "20", "21", "22", "23", "00", "01"], ["12", "13", "14", "15", "16"])
        # session = product(["19", "20", "21", "22", "23", "00"], ["12", "13", "14", "15", "16"], ["12", "13", "14", "15", "16", "17"])
        # session = product(["01", "02", "03", "04", "05", "06", "07", "08", "09"], ["13"], ["14"])
        # session = product(["22", "23", "00"], ["12", "13", "14", "15", "16"], ["12", "13", "14", "15", "16"])
        # session = product(["22", "23", "00"], ["12", "13", "14", "15", "16"], ["17"])
        # session = product(["23"], ["13"], ["13", "14", "15"])
        # session = [(x[0], x[1], x[2]) for x in session if x[2] >= x[1]]
        session = [("23", "13", "14")]
        # session = product(["19", "20", "21", "22", "23", "00", "01"], ["12", "13", "14"])
        # session = product(["19", "20", "21", "22", "23", "00"], ["12", "13"])
        # session = product(["21", "22", "23", "00"], ["13", "14", "15", "16"])
        # session = product(["21", "22", "23", "00"], ["13", "14"])
        # session = product(["23", "00"], ["13", "14"])
        # session = product(["02", "04", "05"], ["17", "18", "19"])
        # session = product(["02", "04", "05"], ["18"])
        # session = product(["04", "05"], ["18"])
        # session = [("05", "18")]
        # session = [("04", "18"), ("05", "18")]
        # session = [("02", "18"), ("04", "18"), ("05", "18")]

        # hour_time = product(["00", "15", "30", "45"], ["00", "15", "30", "45"], ["00", "15", "30", "45"])
        # hour_time = product(["00", "15", "30", "45"], ["00", "15", "30", "45"], ["00", "45"])
        # hour_time = product(["00", "15", "30", "45"], ["15"], ["00", "15", "30", "45"])
        # hour_time = [(x[0], x[1], x[2]) for x in hour_time]
        hour_time = [("30", "15", "45")]

        # risk_param = [0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03]
        # risk_param = [0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03]
        # risk_param = [0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025]
        # risk_param = [0.024]
        # risk_param = range(24,51,1)
        # risk_param = [x/1000 for x in risk_param]
        # risk_param = [0.024, 0.026, 0.033, 0.037, 0.038, 0.039, 0.04, 0.041, 0.047, 0.05]
        # risk_param = [0.025, 0.035, 0.05, 0.07, 0.075, 0.1]
        # risk_param = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        # risk_param = [0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.5]
        # risk_param = [x / 1000 for x in range(35, 51, 1)]
        # risk_param = [0.025, 0.03, 0.035, 0.04, 0.045]
        # risk_param = [0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045]
        # risk_param = [0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045]
        # risk_param = [0.04, 0.045]
        # risk_param = [x / 1000 for x in range(20, 35, 1)]
        risk_param = [0.045]

        # close_param = [0.2, 0.21, 0.22, 0.23, 0.24, 0.25]
        # close_param = [0.1, 0.13, 0.15, 0.17, 0.2, 0.21, 0.22, 0.25, 0.3]
        # close_param = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25]
        # close_param = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
        # close_param = [0.03, 0.035, 0.04, 0.045, 0.05]
        # close_param = [0.025, 0.026, 0.027, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045]
        # close_param = [0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04]
        # close_param = [0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052
        #                , 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.06]
        # close_param = [0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05]
        # close_param = [x / 1000 for x in range(44, 56, 1)]
        # close_param = [0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05]
        # close_param = [0.048, 0.05]
        close_param = [0.05]
        # close_param = [0.24, 0.25, 0.26, 0.27]
        # close_param = [0.3]
        # close_param = [0.26, 0.3]
        # close_param = [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27]
        # close_param = range(25,270,5)
        # close_param = range(200,270,5)
        # close_param = [x/1000 for x in close_param]
        # close_param = [0.265]

        stop_params = product(risk_param, close_param)
        stop_params = [(x[0], x[1]) for x in stop_params if x[0] < x[1]]
        # stop_params = [(x[0], x[1]) for x in stop_params]
        # stop_params = [(0.03, 0.035), (0.04, 0.045), (0.04, 0.05)]
        # stop_params = [(0.03, 0.035), (0.035, 0.04), (0.04, 0.045)]

        # risk_params, close_params = map(list, zip(*stop_params))

        # start = [self.start if start is None else start]
        # end = [self.end if start is None else end]

        # date_range = self.date_brackets

        # multiplier = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
        # multiplier = product([0.25, 0.5, 0.67, 0.75, 0.8, 0.9, 1.0], [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 5.0, 1000.0])
        # multiplier = [(x[0], x[1]) for x in multiplier if x[0] < x[1]]
        """
        multiplier = [(1.0, 10000.0), (1.0, 2.0), (1.0, 2.5), (1.0, 3.0), (1.0, 3.5), (1.0, 4.0)
                      , (1.0, 4.5), (1.0, 5.0), (1.0, 5.5), (1.0, 6.0), (1.0, 6.5), (1.0, 7.0), (1.0, 7.5)
                      , (1.0, 8.0), (1.0, 8.5), (1.0, 9.0), (1.0, 9.5), (1.0, 10.0), (1.0, 10.5), (1.0, 11.0)
                      ]
        """
        """
        multiplier = [(1.0, 4.0), (1.0, 4.1), (1.0, 4.2), (1.0, 4.3), (1.0, 4.4), (1.0, 4.5), (1.0, 4.6), (1.0, 4.7), (1.0, 4.8), (1.0, 4.9), (1.0, 5.0)
                      , (1.0, 5.1), (1.0, 5.2), (1.0, 5.3), (1.0, 5.4), (1.0, 5.5), (1.0, 5.6), (1.0, 5.7), (1.0, 5.8), (1.0, 5.9), (1.0, 6.0)
                      , (1.0, 6.1), (1.0, 6.2), (1.0, 6.3), (1.0, 6.4), (1.0, 6.5), (1.0, 6.6), (1.0, 6.7), (1.0, 6.8), (1.0, 6.9), (1.0, 7.0)
                      , (1.0, 7.1), (1.0, 7.2), (1.0, 7.3), (1.0, 7.4), (1.0, 7.5), (1.0, 7.6), (1.0, 7.7), (1.0, 7.8), (1.0, 7.9), (1.0, 8.0)
                      , (1.0, 8.1), (1.0, 8.2), (1.0, 8.3), (1.0, 8.4), (1.0, 8.5), (1.0, 8.6), (1.0, 8.7), (1.0, 8.8), (1.0, 8.9), (1.0, 9.0)
                      ]
        """
        # multiplier = [(1.0, 4.5), (1.0, 4.6), (1.0, 4.7), (1.0, 4.8), (1.0, 4.9), (1.0, 5.0)
        #               , (1.0, 5.1), (1.0, 5.2), (1.0, 5.3), (1.0, 5.4), (1.0, 5.5), (1.0, 5.6), (1.0, 5.7), (1.0, 5.8), (1.0, 5.9), (1.0, 6.0)
        #               , (1.0, 6.1), (1.0, 6.2), (1.0, 6.3), (1.0, 6.4), (1.0, 6.5), (1.0, 6.6), (1.0, 6.7), (1.0, 6.8), (1.0, 6.9), (1.0, 7.0)
        #               ]
        # multiplier = [(1.0, 4.5), (1.0, 4.6), (1.0, 4.7), (1.0, 6.5), (1.0, 6.6), (1.0, 6.7)]
        # multiplier = [(1.0, 4.7), (0.9, 4.7), (0.8, 4.7), (0.7, 4.7), (0.6, 4.7), (0.5, 4.7)]
        # multiplier = [(1.0, 4.7), (1.0, 1000)]
        multiplier = [(1.0, 4.7)]

        # macdc2 = product(range(8,14,1), range(39,45,1), [3])
        # macdc2 = [(x[0], x[1], x[2]) for x in macdc2 if x[1] > x[0]]
        # macdc2 = [(11,22,3), (13,39,3), (10,39,3), (9,28,3), (10,15,3)]
        # macdc2 = [(11,22,3), (13,39,3), (10,39,3), (9,28,3), (10,25,3)]
        # macdc2 = [(11,22,3), (10,39,3), (9,28,3)]
        macdc2 = [(10,39,3)]
        mar = [0.1, 0.15, 0.2]

        date_range = [(1, "2021-10-15", "2022-01-15"), (2, "2022-03-01", "2022-09-01"), (3, "2022-10-23", "2023-02-07")]
        self.date_brackets = date_range

        combinations = list(product(date_range, emac, macdc, macdcc1, macdcc2, rsi, mac, vol_length, swing, session, hour_time, stop_params, multiplier, macdc2, mar# , mac_neg
                                    ))
        print(f"Parameter combinations: {len(combinations)}")

        # params_df = pd.read_excel(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\trade_results_eurusd_2close.xlsx", usecols = ["params"])
        # params_df = params_df.drop_duplicates()
        # params = params_df["params"].tolist()
        # combinations = [x for x in combinations if str(x[1:]) not in params]

        print(f"Cores available: {cpu_count()}")
        print(f"Reduced combinations: {len(combinations)}")

        with Pool(cpu_count()) as pool:
            self.optimization_results = pool.starmap(self.optimize_strategy, combinations)

        # self.optimization_results = sorted(results, key = lambda x: x[7], reverse = True)
        # self.optimization_results = results
        # self.max_return = max(results, key = lambda t: t[7])
        # self.optimal_parameters = self.max_return[3]
        # self.optimal_returns = self.max_return[7]
        # self.max_return = (self.start, self.end, max_return[0], max_return[2])

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
        # self.cv_results["total_hold"] = self.cv_results.filter(regex=("hold")).multiply(axis=1)
        # self.cv_results["total_return"] = self.cv_results.filter(regex=("strat_return")).multiply(axis=1)
        self.cv_results["total_return"] = self.cv_results["strat_return1"] * self.cv_results["strat_return2"] * self.cv_results["strat_return3"]
        # self.cv_results["total_outperf"] = self.cv_results["total_return"] - self.cv_results["total_hold"]
        # self.cv_results["return_std"] = self.cv_results.filter(regex=("strat_return")).std(axis=1)
        # self.cv_results["score1"] = self.cv_results["total_return"] - self.cv_results["return_std"]
        # self.cv_results["score2"] = self.cv_results["total_return"] - 1.5 * self.cv_results["return_std"]
        # self.cv_results["score3"] = (self.cv_results["total_return"] - self.cv_results["return_std"]) * self.cv_results["total_accuracy"]
        # self.cv_results["score"] = self.cv_results["total_return"] * self.cv_results["total_accuracy"] - self.cv_results["return_std"]
        # self.cv_results = self.cv_results.sort_values(by=["total_return", "score4", "score2", "score1"], ascending=False)
        self.cv_results["pos_re"] = self.cv_results.filter(regex=("returns_pos")).mean(axis=1)
        self.cv_results["neg_re"] = self.cv_results.filter(regex=("returns_neg")).mean(axis=1)
        self.cv_results["total_score"] = self.cv_results["total_return"] * self.cv_results["total_accuracy"]
        self.cv_results = self.cv_results.sort_values(by=["total_score", "total_accuracy", "pos_re"], ascending=False)


if __name__ == "__main__":

    start_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)))

    # d1 = ["2021-10-15", "2022-01-15"]
    # d2 = ["2022-03-01", "2022-09-01"]
    # d3 = ["2022-10-23", "2023-01-18"]

    trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", "EUR_USD", "M15", "2021-10-15", "2023-02-07", 50)

    # calculate strategy returns
    trader.cross_val_combine()

    trader.cv_results.to_csv(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\trade_results_eurusd_2close.csv", mode="w", index=True, header=True)
    # df = pd.read_csv(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\trade_results_eurusd_2close.csv")
    # df = df.drop_duplicates(subset=["params"])
    # df = df.sort_values(by=["total_score", "total_accuracy", "pos_re"], ascending=False)
    # df.to_csv(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\trade_results_eurusd_2close.csv", index=False, header=True)

    end_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end_time)))
    print(1.0*(end_time - start_time)/60)
