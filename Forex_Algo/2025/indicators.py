import pandas as pd
import numpy as np
import math

def imbalance_indicator(self, df, compare=10, ma=9, urgent_rate=1.0):
    df = df.copy()

    # Split for urgency checks based on direction
    df.body_length_up = df.body_length * (df.direction == 1)
    df.body_length_down = df.body_length * (df.direction == -1)

    df.sma_up = df.body_length_up.rolling(ma).mean()
    df.sma_down = df.body_length_down.rolling(ma).mean()

    df.urgent_up = ((df.direction == 1) & (df.candle_lenth > urgent_rate * df.sma_up)).astype(int)
    df.urgent_down = ((df.direction == -1) & (df.candle_lenth > urgent_rate * df.sma_down)).astype(int)


    # --- Bullish Liquidity ---
    maru_up = (df.direction == 1) & (df.candle_type == "maru")
    maru3_maru1_up = maru_up & ((df.candle_lenth.shift(1) / df.candle_lenth.shift(2)) >= compare / 100)
    m3_h_m1_up = maru3_maru1_up & (df.high.shift(2) < df.low)
    conti_up = (df.direction.shift(2) == 1) & (df.direction.shift(1) == 1)

    df.up_liquidity = (m3_h_m1_up & conti_up & (df.urgent_up == 1)).astype(int)

    # --- Bearish Liquidity ---
    maru_down = (df.direction == -1) & (df.maru == 1)
    maru3_maru1_down = maru_down & ((df.candle_lenth / df.candle_lenth.shift(2)) >= compare / 100)
    m3_h_m1_down = maru3_maru1_down & (df.low.shift(2) > df.high)
    conti_down = (df.direction.shift(2) == -1) & (df.direction.shift(1) == -1)

    df.down_liquidity = (m3_h_m1_down & conti_down & (df.urgent_down == 1)).astype(int)

    # Final signal column
    df.signal = np.where(df.up_liquidity == 1, "up_liquidity",
                     np.where(df.down_liquidity == 1, "down_liquidity", None))

    return df

def volume_spike(self, vol_ratio, lookback):
    df = df.copy()

    df.avg_volume = df.volume.rolling(lookback).mean()
    df.volume_spike = np.where(df.volume > vol_ratio * df.avg_volume.shift(1), 1, 0)


def momentum_angle(self, idx1, idx2):
    """
    Returns the angle in degrees between two candles, treating idx1 as the origin (0,0).

    Parameters:
        idx1 (int): Index of first candle (treated as origin)
        idx2 (int): Index of second candle
        price_type (str): One of 'o', 'h', 'l', 'c' for open, high, low, close

    Returns:
        float: Angle in degrees, positive for upward trend, negative for downward
    """
    if idx1 < 0 or idx2 < 0 or idx1 >= len(self.df) or idx2 >= len(self.df):
        raise IndexError("Indices out of bounds")
    if idx1 == idx2:
        return 0.0  # No distance → flat angle

    y1 = getattr(self.df.iloc[idx1], price_type)
    y2 = getattr(self.df.iloc[idx2], price_type)

    delta_y = y2 - y1
    delta_x = idx2 - idx1

    slope = delta_y / delta_x
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def fibonacci_levels(self, idx1, idx2):
    """
    Returns Fibonacci-like levels between two candles.

    Parameters:
        idx1 (int): First candle index
        idx2 (int): Second candle index
        price_type (str): One of 'o', 'h', 'l', 'c'
        levels (list or None): List of percentage levels (default: [80, 70, 61.8, 50, 30])

    Returns:
        dict: {level_percent: price_value}
    """
    if levels is None:
        levels = [80, 70, 61.8, 50, 30]

    if idx1 < 0 or idx2 < 0 or idx1 >= len(self.df) or idx2 >= len(self.df):
        raise IndexError("Indices out of bounds")

    p1 = getattr(self.df.iloc[idx1], price_type)
    p2 = getattr(self.df.iloc[idx2], price_type)

    delta = p2 - p1
    fib_dict = {}

    for level in levels:
        ratio = 1 - (level / 100)
        value = p1 + delta * ratio
        fib_dict[level] = round(value, 5)

    return fib_dict





