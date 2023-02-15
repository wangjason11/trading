import math
import numpy as np
import pandas as pd
import pandas_ta as ta
import random
import smtplib
import ssl
import time
import tpqoa

from csv import writer
from datetime import datetime, timedelta
from time import sleep


class TrendAlgoTrader(tpqoa.tpqoa):
    """ Class for executing trades based on strategy.
    """
    def __init__(self, conf_file, instrument, bar_length, risk_loss, ratio, stop_loss, session_start, session_end, close_time):
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
        risk_loss: float
            risk loss close out parameter to trigger position close at end of bar (expressed in decimals)
        ratio: float
            ratio of total available margin willing to risk (expressed in decimals)
        stop_loss: float
            parameter to trigger stop losses (expressed in decimals)
        session_start: str
            hour to start trading session, a string like "00", "04", "23"
        session_end: str
            hour to end trading session, a string like "00", "04", "23"
        """
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = bar_length
        self.risk_loss = risk_loss
        self.ratio = ratio
        self.stop_loss = stop_loss
        # self.fast_stop_loss = fast_stop_loss
        self.session_start = session_start
        self.session_end = session_end
        self.bar_number = pd.to_timedelta(bar_length[1:] + "min")
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = 0
        self.position = 0
        self.begin_NAV = float(self.get_account_summary()["NAV"])
        self.daily_results = None
        self.current_price = None
        self.previous_price = None
        self.current_seconds = None
        self.previous_seconds = None
        self.trades = 0
        self.max_active_return = 0
        self.open_units = 0
        self.close_time = close_time
        self.manual = False

    def get_most_recent(self, days = 5):
        """ Retrieves data.
        Parameters
        ==========
        days: int
            historical days to look back to retrieve initial data
        """
        while True:
            time.sleep(2)
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)

            # get historical data to initiate trading session
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = self.bar_length, price = "M", localize = False)[["o", "h", "l", "c"]].dropna()

            df = df.resample(self.bar_number, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            self.data = df.copy()
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_number:
                break

    def start_trading(self, max_attempts, wait, wait_increase, days = 5):
        """ Function to start trading session. 
        Parameters
        ==========
        max_attempts: int
            max stream restart attempts
        wait: int
            seconds to wait after connection failure to restart stream
        wait_increase: int
            wait time increase after connection failure
        days: int
            historical days to look back to retrieve initial data
        """
        print("\n" + 100 * "=")
        print("Starting Session {}".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")), end = "\n")
        attempt = 0 
        success = False
        while True:
            try:
                self.get_most_recent(days)
                self.stream_data(self.instrument)
            except Exception as e:
                print(e, end = " | ")
            else:
                success = True
                break
            finally:
                attempt += 1
                print("Attempt : {}".format(attempt), end = "\n")
                if success == False:
                    if attempt >= max_attempts:
                        print("max attempts reached")
                        try:
                            time.sleep(wait)
                            self.terminate_session(cause = "Unexpected Session Stop (too many errors)")
                            self.send_mail(subject = "Unexpected Session Stop (too many errors)", content = "Check if want to manually restart session.")
                        except Exception as e:
                            print(e, end = " | ")
                            print("Could not terminate session properly")
                            self.send_mail(subject = "Could not terminate session properly", content = "Check positions now!")
                        finally:
                            break
                    else:
                        time.sleep(wait)
                        if attempt % 5 == 0:
                            wait += wait_increase
                        self.tick_data = pd.DataFrame()
                        print("\n" + 100 * "=")
                        print("Restarting Session {}".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")), end = "\n")

    def close_outs(self, trade_return):
        """ Execute close out conditions.
        """

        # slow stop loss close condition
        if trade_return < -self.stop_loss:
            self.close_order("stop loss")
            return

        """
        elif recent_tick.time().second % 2 == 0:
            self.previous_price = self.current_price
            self.current_price = (ask + bid)/2
            self.previous_seconds = self.current_seconds
            self.current_seconds = recent_tick.time().second
            if self.previous_price is not None and self.current_price is not None:
                returns_2sec = (self.current_price - self.previous_price) * self.position / self.previous_price

                # fast stop loss close condition
                if returns_2sec < -self.fast_stop_loss and self.current_seconds - self.previous_seconds < 5:
                    self.close_order("fast stop loss: {}".format(round(returns_2sec, 4)))
                    return
        """

        # profit protect close order conditions
        if self.max_active_return > 0.8 and trade_return/self.max_active_return < 0.8:
            self.close_order("profit protect: {}".format(round(self.max_active_return, 4)))
        elif self.max_active_return > 0.6 and trade_return/self.max_active_return < 0.75:
            self.close_order("profit protect: {}".format(round(self.max_active_return, 4)))
        elif self.max_active_return > 0.5 and trade_return/self.max_active_return < 0.7:
            self.close_order("profit protect: {}".format(round(self.max_active_return, 4)))
        elif self.max_active_return > 0.4 and trade_return/self.max_active_return < 0.65:
            self.close_order("profit protect: {}".format(round(self.max_active_return, 4)))
        elif self.max_active_return > 0.3 and trade_return/self.max_active_return < 0.55:
            self.close_order("profit protect: {}".format(round(self.max_active_return, 4)))
        elif self.max_active_return > 0.25 and trade_return/self.max_active_return < 0.5:
            self.close_order("profit protect: {}".format(round(self.max_active_return, 4)))

    def on_success(self, time, bid, ask):
        """ Method called when new data is retrieved. 
        """

        # record current tick
        print(self.ticks, end = "\r", flush = True)
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])

        order_position = {1: "GOING LONG", -1: "GOING SHORT"}

        # execute stop loss orders
        positions = self.get_positions()
        if positions != []:
            if self.position == 0:
                print("\n" + "mismatch in position ----------")
                print("recorded_units = {} | recorded_position = {}".format(self.units, self.position))
                self.trades += 1
                if float(positions[0]["long"]["units"]) != 0:
                    self.position = 1
                elif abs(float(positions[0]["short"]["units"])) != 0:
                    self.position = -1
                print("new_recorded_units = {} | new_recorded_position = {}".format(self.units, self.position), end = "\n")
            if self.units != max(abs(float(positions[0]["short"]["units"])), float(positions[0]["long"]["units"])):
                print("\n" + "mismatch units ----------")
                print("recorded_units = {} | recorded_position = {}".format(self.units, self.position))
                self.units = max(abs(float(positions[0]["short"]["units"])), float(positions[0]["long"]["units"]))
                print("new_recorded_units = {} | new_recorded_position = {}".format(self.units, self.position), end = "\n")
                # sleep(1)
                # transaction = self.get_transactions()[-1]
                # self.report_trade(transaction, order_position[self.position] + " (manual)")
                if self.units % 1000 == 0:
                    self.manual = True

            trade_return = float(positions[0]["unrealizedPL"]) / float(positions[0]["marginUsed"])
            if trade_return > self.max_active_return:
                self.max_active_return = trade_return
            self.close_outs(trade_return)

            diff = trade_return / self.max_active_return if self.max_active_return > 0 else 0
            if self.max_active_return > 0.25 and diff < 0.45 and (recent_tick.tz_convert("America/New_York").time() >= pd.to_datetime("08:30").time() or recent_tick.tz_convert("America/New_York").time() < pd.to_datetime("02:00").time()):
                if recent_tick.time().second == 0:
                    content = f"mar: {round(self.max_active_return, 4)}, return: {round(trade_return, 4)}, diff: {round(diff,4)}"
                    self.send_mail(subject = "max return alert", content = content)
        else:
            if self.position != 0 or self.units !=0:
                print("\n" + "mismatch position or units ----------")
                print("recorded_units = {} | recorded_position = {}".format(self.units, self.position))
                self.units = 0
                self.position = 0
                print("new_recorded_units = {} | new_recorded_position = {}".format(self.units, self.position), end = "\n")
                # sleep(1)
                # transaction = self.get_transactions()[-1]
                # self.report_trade(transaction, "GOING NEUTRAL (manual: max={})".format(round(self.max_active_return, 4)))
                self.max_active_return = 0
                self.manual = False

        # stop stream and trades condition based on session_end hour
        if recent_tick.tz_convert("America/New_York").time() >= pd.to_datetime("16:57").time() and recent_tick.tz_convert("America/New_York").time() < pd.to_datetime("17:04").time():
            self.terminate_session(cause = "Scheduled Session End.")
            self.daily_results = [datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), self.begin_NAV, float(self.get_account_summary()["NAV"]), float(self.get_account_summary()["NAV"]) - self.begin_NAV, float(self.get_account_summary()["NAV"])/self.begin_NAV-1, self.trades]
            results_print = "daily_returns = {} | end_NAV = {} | trades = {} | cumpl = {} | begin_NAV = {}".format(round(self.daily_results[4], 4), round(self.daily_results[2], 2), self.daily_results[5], round(self.daily_results[3], 2), round(self.daily_results[1], 2))
            self.send_mail(subject = "Scheduled Session End", content = results_print)
            return

        # if current tick completes full bar and is after session_start hour
        if recent_tick - self.last_bar > self.bar_number:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades(recent_tick)
            positions = self.get_positions()
            if positions != []:
                trade_return = float(positions[0]["unrealizedPL"]) / float(positions[0]["marginUsed"])
                print("{} | trade_return = {} | P&L = {} | max_active_return = {}".format(recent_tick, round(trade_return, 4), float(positions[0]["unrealizedPL"]), round(self.max_active_return, 4)))
    
    def resample_and_join(self):
        """ Resample by bar length.
        """
        df = self.tick_data.resample(self.bar_number, label="right").last().ffill().iloc[:-1]
        df["o"] = self.tick_data["EUR_USD"].iloc[0]
        df["h"] = self.tick_data["EUR_USD"].max()
        df["l"] = self.tick_data["EUR_USD"].min()
        df.rename(columns={"EUR_USD": "c"}, inplace=True)

        self.raw_data = pd.concat([self.raw_data, df])

        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
        self.data = self.raw_data.copy()

    def set_parameters(self, stop_loss=None, ratio=None, session_start=None, session_end=None):
        """ Set new parameters for strategy.
        Parameters
        ==========
        risk_loss: float
            risk loss close out parameter to trigger position close at end of bar (expressed in decimals)
        stop_loss: float
            parameter to trigger stop losses (expressed in decimals)
        ratio: float
            ratio of total available margin willing to risk (expressed in decimals)
        session_start: str
            hour to start trading session, a string like "00", "04", "23"
        session_end: str
            hour to end trading session, a string like "00", "04", "23"
        """
        # if risk_loss is not None:
            # self.risk_loss = risk_loss
        if stop_loss is not None:
            self.stop_loss = stop_loss
        # if fast_stop_loss is not None:
            # self.fast_stop_loss = fast_stop_loss
        if ratio is not None:
            self.ratio = ratio
        if session_start is not None:
            self.session_start = session_start
        if session_end is not None:
            self.session_end = session_end
        self.data = self.raw_data.copy()

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
        self.data = self.data.join(df[["ema_crossover_{}_{}_signal".format(short, longg)]])

    def macd_crossover(self, short, longg, signal, num):
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
        self.data = self.data.join(df[["trend_macd_crossover_signal_{}".format(num)]])

    def macd_crossover_close(self, short, longg, signal):
        """ Returns MACD Crossover Close Signal based on input parameters.
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
        self.data = self.data.join(df[["macd_crossover_close"]])

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
        self.data = self.data.join(df[["trend_rsi_{}_signal".format(window)]])

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
        self.data = self.data.join(df[["MAC_{}{}_{}{}_signal".format(short[1], short[0], longg[1], longg[0])]])

    def volatility_osc(self, length, multiplier):
        """ Returns volatilitiy oscillator based on input length window.
        Parameters
        ==========
        length: int
            window parameter for volatility standard deivation calculation
        """        
        df = self.data.copy()
        df["spike"] = df["c"] - df["o"]
        df["upper"] = df["spike"].rolling(length, min_periods=length).std()
        df["lower"] = df["spike"].rolling(length, min_periods=length).std() * -1
        df["vol_signal_{}".format(length)] = np.where((df["spike"] > 0) & (df["spike"] > df["upper"]) & (df["spike"]/df["upper"] < multiplier), 1,
                                                      np.where((df["spike"] < 0) & (df["spike"] < df["lower"]) & (df["spike"]/df["lower"] < multiplier), -1, 0)
                                                      )
        self.data = self.data.join(df[["vol_signal_{}".format(length)]])

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
        price_h = {False : "c", True : "h"}
        price_l = {False : "c", True: "l"}

        df = self.data.copy()
        df["swing_d_{}".format(right+1)] = df[price_h[high_low]].shift(0)
        for x in range(1, right+1):
            df["swing_d_{}".format(x)] = df[price_h[high_low]].shift(-x)
        for x in range(1, left+1):
            df["swing_d_{}".format(right+1+x)] = df[price_h[high_low]].shift(x)
        df["maxPH"] = df.filter(regex=("swing_d_")).max(axis=1)
        df["PH"] = np.where(df["maxPH"] == df["swing_d_{}".format(right+1)], df["swing_d_{}".format(right+1)], np.nan)
        df["recentPH"] = df["PH"].shift(right).astype(float).fillna(method="ffill")

        df["swing_d_{}".format(right+1)] = df[price_l[high_low]].shift(0)
        for x in range(1, right+1):
            df["swing_d_{}".format(x)] = df[price_l[high_low]].shift(-x)
        for x in range(1, left+1):
            df["swing_d_{}".format(right+1+x)] = df[price_l[high_low]].shift(x)
        df["minPL"] = df.filter(regex=("swing_d_")).min(axis=1)
        df["PL"] = np.where(df["minPL"] == df["swing_d_{}".format(right+1)], df["swing_d_{}".format(right+1)], np.nan)
        df["recentPL"] = df["PL"].shift(right).astype(float).fillna(method="ffill")
        self.data = self.data.join(df[["recentPL", "recentPH"]])

    def define_strategy(self): 
        """ Execute strategy-specific signals.
        """

        # *****************add strategy-specific signals here*********************

        # optimized parameters. See returns_optimize3.py
        self.ema_crossover(short=60, longg=95)
        self.macd_crossover(short=11, longg=35, signal=3, num=1)
        self.macd_crossover(short=10, longg=39, signal=3, num=2)
        self.rsi(window=15)
        self.rsi(window=5)
        self.consensus_trend()
        self.ma_crossover(short=(4, "EMA"), longg=(10, "EMA"))
        self.ma_crossover(short=(19, "EMA"), longg=(24, "SMA"))
        self.volatility_osc(length=155, multiplier=4.7)
        self.swing(left=8, right=8, high_low=True)
        self.macd_crossover_close(short=8, longg=38, signal=3)
        # ************************************************************************

        # determine open signal
        self.data["open"] = np.where(self.data.filter(regex=("signal")).mean(axis=1).abs() == 1.0, self.data.filter(regex=("signal")).mean(axis=1), 0)

    def execute_trades(self, recent_tick):
        """ Execute trades.
        """
        phl = {1: "recentPL", -1: "recentPH"}
        open_signal = self.data["open"].iloc[-1]

        # open signal
        if open_signal != 0:
            if recent_tick.tz_convert("America/New_York").time() < pd.to_datetime("{}:15".format(self.session_end)).time() or recent_tick.tz_convert("America/New_York").time() >= pd.to_datetime("{}:30".format(self.session_start)).time():

                # when in netural position, open position
                if self.get_positions() == []:
                    self.open_order(open_signal)

                # when in position, 
                else:
                    if open_signal == self.position: # if buy position same as open position
                        pass
                    else:

                        # close position
                        self.close_order("re-entry")

                        # then open new position
                        self.open_order(open_signal)

        # close signal
        elif self.get_positions() != []:

            # loss greater than risk loss parameter at end of bar
            if float(self.get_positions()[0]["unrealizedPL"]) / float(self.get_positions()[0]["marginUsed"]) < -self.risk_loss:
                self.close_order("risk loss")

            # close signal and positive returns
            elif self.data["macd_crossover_close"].iloc[-1] != self.position:
                if self.manual == False:
                    if float(self.get_positions()[0]["unrealizedPL"]) / float(self.get_positions()[0]["marginUsed"]) > 0:
                        self.close_order("macd close: max={}".format(round(self.max_active_return, 4)))
                # else:
                    # if self.max_active_return > 0.05:
                        # self.close_order("macd close: max={}".format(round(self.max_active_return, 4)))

            # bar closing price beyond recent High/Low threshold
            # elif (self.data["c"].iloc[-1] - self.data[phl[self.position]].iloc[-1]) * self.position < 0:
                # self.close_order("hl loss: {}".format(self.data[phl[self.position]].iloc[-1]))

            # past optimal holding time (close)
            # elif recent_tick.tz_convert("America/New_York").time() >= pd.to_datetime("{}:45".format(self.close_time)).time():
                # self.close_order("auto ending close: max={}".format(round(self.max_active_return, 4)))

    def open_remain(self, open_signal, order_position):
        """ Creates open order.
        Parameters
        ==========
        open_signal: int
            long (1) or short (-1) position 
        """
        attempt = 0
        while float(self.get_account_summary()["marginAvailable"]) > 5000 and attempt < 10:
            try:
                order = self.create_order(self.instrument, units = open_signal * self.open_units, suppress = False, ret = True)
            except Exception as e:
                print(e, " failed open remain", end = "\n")
                attempt += 1
                print("Attempt : {}".format(attempt), end = "\n")
                self.send_mail(subject = "Open Order error", content = "failed open remain")
                time.sleep(6)
                tick_time, ask, bid = self.get_prices(self.instrument)
                bid_ask = {1: ask, -1: bid}
                self.open_units = math.floor(float(self.get_account_summary()["marginAvailable"])*self.ratio/float(self.get_account_summary()["marginRate"])/bid_ask[open_signal])
            else:
                # self.units += self.open_units
                # self.position = open_signal
                self.report_trade(order, order_position[open_signal])

    def open_ten_mm(self, open_signal, order_position):
        """ Creates open order.
        Parameters
        ==========
        open_signal: int
            long (1) or short (-1) position 
        """
        attempt = 0
        while self.open_units > 10000000 and attempt < 10:
            try:
                order = self.create_order(self.instrument, units = open_signal * 10000000, suppress = False, ret = True)
            except Exception as e:
                print(e, " failed open 10mm", end = "\n")
                attempt += 1
                print("Attempt : {}".format(attempt), end = "\n")
                self.send_mail(subject = "Open Order error", content = "failed open 10mm")
            else:
                # self.units += self.open_units
                # self.position = open_signal
                self.report_trade(order, order_position[open_signal])
            finally:
                time.sleep(6)
                tick_time, ask, bid = self.get_prices(self.instrument)
                bid_ask = {1: ask, -1: bid}
                self.open_units = math.floor(float(self.get_account_summary()["marginAvailable"])*self.ratio/float(self.get_account_summary()["marginRate"])/bid_ask[open_signal])

    def open_order(self, open_signal):
        """ Creates open order.
        Parameters
        ==========
        open_signal: int
            long (1) or short (-1) position
        """
        order_position = {1: "GOING LONG", -1: "GOING SHORT"}
        tick_time, ask, bid = self.get_prices(self.instrument)
        bid_ask = {1: ask, -1: bid}
        self.open_units = math.floor(float(self.get_account_summary()["marginAvailable"])*self.ratio/float(self.get_account_summary()["marginRate"])/bid_ask[open_signal])
        if self.open_units <= 10000000:
            self.open_remain(open_signal, order_position)
        else:
            self.open_ten_mm(open_signal, order_position)
            self.open_remain(open_signal, order_position)

        positions = self.get_positions()
        if positions != []:
            self.trades += 1
            if float(positions[0]["long"]["units"]) != 0:
                self.position = 1
            elif abs(float(positions[0]["short"]["units"])) != 0:
                self.position = -1
            if self.units != max(abs(float(positions[0]["short"]["units"])), float(positions[0]["long"]["units"])):
                self.units = max(abs(float(positions[0]["short"]["units"])), float(positions[0]["long"]["units"]))
        print("recorded_units = {} | recorded_position = {}".format(self.units, self.position), end = "\n")

    def close_order(self, reason):
        """ Creates close order.
        Parameters
        ==========
        reason: str
            reason for closing order 
        """
        open_units = max(abs(float(self.get_positions()[0]["short"]["units"])), float(self.get_positions()[0]["long"]["units"]))
        while open_units > 10000000:
            try:
                order = self.create_order(self.instrument, units = -self.position * 10000000, suppress = False, ret = True)
            except Exception as e:
                print(e, " failed close 10mm", end = "\n")
                self.send_mail(subject = "Close Order error ({})".format(reason), content = "failed close 10mm")
            else:
                self.report_trade(order, "GOING NEUTRAL ({})".format(reason))
            finally:
                time.sleep(6)
                if self.get_positions() != []:
                    open_units = max(abs(float(self.get_positions()[0]["short"]["units"])), float(self.get_positions()[0]["long"]["units"]))
                else:
                    open_units = 0

        while open_units > 0:
            try:
                order = self.create_order(self.instrument, units = -self.position * open_units, suppress = False, ret = True)
            except Exception as e:
                print(e, " failed close remain", end = "\n")
                self.send_mail(subject = "Close Order error ({})".format(reason), content = "failed close remain")
                time.sleep(6)
            else:
                self.report_trade(order, "GOING NEUTRAL ({})".format(reason))
            finally:
                if self.get_positions() != []:
                    open_units = max(abs(float(self.get_positions()[0]["short"]["units"])), float(self.get_positions()[0]["long"]["units"]))
                else:
                    open_units = 0

        if self.get_positions() == []:
            self.units = 0
            self.position = 0
            self.max_active_return = 0
            self.manual = False
        print("recorded_units = {} | recorded_position = {}".format(self.units, self.position), end = "\n")

    def report_trade(self, order, going):
        """ Report trade data and profit loss.
        Parameters
        ==========
        order: obj
            executed order
        going: str
            position change text
        """
        cumpl = float(self.get_account_summary()["NAV"]) - self.begin_NAV
        NAV = float(self.get_account_summary()["NAV"])
        mar = self.max_active_return

        attempt = 0
        while attempt < 3:
            try:
                time = order["time"]
                units = order["units"]
                price = order["price"]
                pl = float(order["pl"])
                returns = pl/(NAV - pl)
        
                print("\n" + 100* "-")
                print("{} | {}".format(time, going))
                print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {} | NAV = {} | returns = {} | max active returns = {}".format(time, units, price, round(pl, 2), round(cumpl, 2), round(NAV, 2), round(returns, 4), round(mar, 4)))
                print(100 * "-" + "\n")
                results = [time, going, units, price, pl, cumpl, NAV, returns, mar]
                with open("trade_returns.csv", "a", newline="") as f:
                    csv_writer = writer(f)
                    csv_writer.writerow(results)

                trade_content = "returns = {} | P&L = {} | mar = {} | price = {} | units = {} | NAV = {} | Cum P&L = {}".format(round(returns, 4), round(pl, 2), round(mar, 4), price, units, round(NAV, 2), round(cumpl, 2))
                self.send_mail(subject = f"{going}", content = trade_content)
            except Exception as e:
                print(e, " reporting error", end = "\n")
                attempt += 1
                self.send_mail(subject = "Reporting error", content = e)
            else:
                break

    def send_mail(self, subject, content):
        """ Send Email Alerts.
        Parameters
        ==========
        subject: str
            email subject
        content: str
            email body content
        """
        ctx = ssl.create_default_context()
        pw = "jxqbyqqezxwdcthg"
        sender = "projectretirealgotrader@gmail.com"
        receiver = "projectretirealgotrader@gmail.com"

        attempt = 0
        while attempt < 3:
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=ctx) as server:
                    server.login(sender, pw)
                    server.sendmail(sender, receiver, f"Subject: {subject}\n{content}")
            except Exception as e:
                attempt += 1
                time.sleep(1)
                ctx = ssl.create_default_context()
            else:
                break

    def terminate_session(self, cause):
        """ Function to end trading session. 
        Parameters
        ==========
        cause: str
            session termination cause
        """
        self.stop_stream = True
        if self.get_positions() != []:
            self.close_order("stop stream")
        print(cause)


if __name__ == "__main__":

    trader = TrendAlgoTrader("oanda.cfg", "EUR_USD", "M15", risk_loss=0.045, ratio=0.9995, stop_loss=0.05, session_start="23", session_end="13", close_time="13")
    trader.start_trading(max_attempts=100, wait=5, wait_increase=1, days=5)
    print("\n" + "Ending Session {}".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")), end = "\n")
    print(100 * "=" + "\n")

    with open("daily_returns.csv", "a", newline="") as f:
        csv_writer = writer(f)
        csv_writer.writerow(trader.daily_results)