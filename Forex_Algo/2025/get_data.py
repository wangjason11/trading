import pandas as pd
# import pandas_ta as ta
import numpy as np
import math
# import matplotlib.pyplot as plt
# import optuna
import time
import tpqoa
from datetime import datetime, timedelta
# from multiprocessing import Pool, cpu_count
from itertools import product


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
        df = self.get_history(instrument = self.instrument, start = start, end = end,
                               granularity = self.bar_length, price = "M", localize = False)[["o", "h", "l", "c"]].dropna()
        # dfa = self.get_history(instrument = self.instrument, start = start, end = end,
        #                        granularity = self.bar_length, price = "A", localize = False).c.dropna().to_frame()
        # dfa.rename(columns={"c": "ask"}, inplace=True)
        # dfb = self.get_history(instrument = self.instrument, start = start, end = end,
        #                        granularity = self.bar_length, price = "B", localize = False).c.dropna().to_frame()
        # dfb.rename(columns={"c": "bid"}, inplace=True)
        # df = pd.concat((dfm, dfa, dfb), axis=1)
        # df["returns"] = np.log(df["c"] / df["c"].shift(1))
        # df["c_prev"] = df["c"].shift(1)
        # df["h_prev"] = df["h"].shift(1)
        # df["l_prev"] = df["l"].shift(1)
        # df["spread"] = df["ask"] - df["bid"]
        # df["ohlc4"] = df[["o", "h", "l", "c"]].mean(axis=1)
        # df["hlc3"] = df[["h", "l", "c"]].mean(axis=1)
        # df["ohlc4_prev"] = df["ohlc4"].shift(1)
        # df["trading_cost"] = (df["spread"]/2) / df["c"]
        # self.raw_data = df.copy()
        self.data = df.copy()

    def adjust_gaps(self):
        df = self.data.copy()
        insert = []

        for i in range(len(df) - 1):
            current_close = df.loc[i, "c"]
            next_open = df.loc[i+1, "o"]

            current_index = df.index[i]
            next_index = df.index[i+1]
            time_difference = next_index - current_index

            if (current_close - next_open).abs() > 0.0001
                midpoint_datetime = current_index + (time_difference / 2)
                high = current_close if current_close > next_open else next_open
                low = next_open if current_close > next_open else current_close
                
                new_row_data = {"o": current_close, "h": high, "l": low, "c": next_open} 
                new_row = pd.DataFrame(new_row_data, index=[midpoint_datetime])
                insert.append((i + 1, new_row))

        # Insert the new rows in reverse order to avoid index shifting issues
        for insertion_idx, new_row in reversed(insert):
            df_before = df.iloc[:insertion_idx]
            df_after = df.iloc[insertion_idx:]
            df = pd.concat([df_before, new_row, df_after]).sort_index() # Sort to maintain chronological order

        self.data = df.copy()

    def main(self):
        start = pd.to_datetime(self.date_brackets[0])
        start = start.strftime("%Y-%m-%d")
        end = pd.to_datetime(self.date_brackets[1])
        end = end.strftime("%Y-%m-%d")
        self.get_data(start, end)
        self.adjust_gaps()


if __name__ == "__main__":

    instrument = "NZD_USD"
    # instrument = "XAU_USD"
    # instrument = "USD_JPY"
    # instrument = "EUR_AUD"
    bar_length = "M15"
    date_brackets = ("2025-06-20", "2025-06-29")
    # trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", "EUR_USD", "M15", [(1, "2016-11-05", "2017-02-05"), (2, "2020-01-12", "2020-04-12"), (3, "2023-06-01", "2023-09-01")], 50)
    trader = TrendAlgoBacktester(r"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\oanda.cfg", instrument, bar_length, date_brackets, 50)
    trader.main()

    trader.data.to_csv(fr"C:\Users\wangj\Documents\codingproj\Project Retire\Forex_Algo\2025\{instrment}_{bar_length}.csv", mode="w", index=True, header=True)