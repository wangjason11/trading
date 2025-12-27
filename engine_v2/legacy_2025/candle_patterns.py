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


class BreakoutPatterns:
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


class BasePatterns:
    def __init__(self, df):
        self.df = df.copy()

    def base_patterns(self, df, length_threshold):
        df = self.df
        df.base_pattern = np.where((
                # df.big_normal.shift(1) == 1 &
                df.big_normal == 1 &
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


class EntryPatterns:
    def __init__(self, df: pd.DataFrame):
        # expects columns: o,h,l,c (and optionally v). Index can be time.
        self.df = df.copy()

    def engulfing(self, df, direction):
        df = self.df
        df.engulfing = 0

        # Current candle body (open/close range)
        cur_body_low  = df[["o", "c"]].min(axis=1)
        cur_body_high = df[["o", "c"]].max(axis=1)

        # Previous candle body (open/close range)
        prev_body_low  = df[["o", "c"]].shift(1).min(axis=1)
        prev_body_high = df[["o", "c"]].shift(1).max(axis=1)

        is_true = (
            # 2nd candle's body engulfs 1st candle's body
            (cur_body_high >= prev_body_high) &
            (cur_body_low  <= prev_body_low)  &

            # Opposite direction vs previous candle
            (df["direction"].shift(1) != df["direction"]) &

            # Only for certain candle types
            (df["candle_type"].isin(["maru", "normal"])) &

            # Match the requested direction (e.g. "up" or "down", or 1/-1)
            (df["direction"] == direction)
        )

        df.loc[is_true, "engulfing"] = direction

    # def engulfing_down(self, df):
    #     df = self.df
    #     df.engulfing_down = 0
    #     is_true = (
    #         (df.h > df.h.shift(1)) & 
    #         (df.l < df.l.shift(1)) & 
    #         (df.direction.shift(1) != df.direction) &
    #         (df.candle_type in ["maru", "normal"])
    #     )
    #     df.loc[is_true, "engulfing_down"] = 1

    def star(self, df, length_threshold, direction):
        df = self.df
        df.star = 0
        is_true = (
            # (df.direction == 1) & 
            # (df.direction.shift(2) == -1) & 
            (df.direction.shift(2) != df.direction) & 
            (df.candle_length > length_threshold * df.candle_length.shift(2)) & 
            (df.candle_type.shift(1) in ["up pinbar", "down pinbar", "pinbar"]) & 
            (df.candle_type.shift(2) in ["maru", "normal"]) & 
            (df.candle_type in ["maru", "normal"]) &
            (df.direction == direction)
        )
        df.loc[is_true, "star"] = direction

    # def evening_star(self, df, length_threshold):
    #     df = self.df
    #     df.evening_star = 0
    #     is_true = (
    #         (df.direction == -1) & 
    #         (df.direction.shift(2) == 1) & 
    #         (df.candle_length > length_threshold * df.candle_length.shift(2)) & 
    #         (df.candle_type.shift(1) in ["up pinbar", "down pinbar", "pinbar"]) &
    #         (df.candle_type.shift(2) in ["maru", "normal"]) & 
    #         (df.candle_type in ["maru", "normal"])
    #     )
    #     df.loc[is_true, "evening_star"] = 1

    def tweezer(self, df, direction):
        df = self.df
        df.tweezer = 0
        is_true = (
            (df.direction.shift(1) != df.direction) & 
            ((
                (df.candle_type.shift(1) in ["up pinbar"]) & 
                (df.candle_type in ["up pinbar"]) &
                (df.direction == direction == 1)) | 
            (
                (df.candle_type.shift(1) in ["down pinbar"]) & 
                (df.candle_type in ["down pinbar"]) &
                (df.direction == direction == -1))
            )
        )

        df.loc[is_true, "tweezer"] == direction

    # def tweezer_top(self, df, length_threshold):
    #     df = self.df
    #     df.tweezer_top = 0
    #     is_true = (
    #         (df.direction == -1) & 
    #         (df.direction.shift(1) == 1) & 
    #         (df.candle_type.shift(1) in ["down pinbar"]) & 
    #         (df.candle_type in ["down pinbar"])
    #     )
    #     df.loc[is_true, "tweezer_top"] = 1

    def mopping(self, df, lookback, length_threshold, direction):
        df = self.df
        df.mopping = 0
        idx = df.index

        for i in range(len(df)):
            if i - 1 >= 0:
                prior = idx[idx < i - 1][-lookback:]
                if len(prior) > 0:
                    max_len = df.loc[prior, "body_length"].max()
                    max_len_idx = df.loc[prior, "body_length"].idxmax()
                    if (df.at[i, "body_length"] <= length_threshold * max_length and 
                        df.at[i-1, "body_length"] <= length_threshold * max_length and
                        df.at[i-1, "direction"] == df.at[i, "direction"] == df.at[max_len_idx, "c"] == -direction and
                        df.at[i, "c"] < df.at[i-1, "c"] < df.at[max_len_idx, "c"] if direction == 1 else df.at[i, "c"] > df.at[i-1, "c"] > df.at[max_len_idx, "c"] and
                        df.at[i, "body_length"] <= length_threshold * df.at[i-1, "body_length"] and
                        df.at[i-1, "body_length"] <= length_threshold * df.at[i, "body_length"]
                    ): 
                        df.at[i, "mopping"] = direction


    @staticmethod
    def _atr(df, period=14):
        tr = np.maximum(df['h'] - df['l'],
                        np.maximum((df['h'] - df['c'].shift()).abs(),
                                   (df['l'] - df['c'].shift()).abs()))
        return tr.rolling(period, min_periods=1).mean()

    @staticmethod
    def _local_minima(series: pd.Series, left=3, right=3):
        """
        A point i is a local min if it's the minimum within [i-left, i+right].
        Returns boolean array.
        """
        rolled_min = series.rolling(left + 1, min_periods=1).min().shift(0)
        # to include right-side window, compare vs forward rolling min via reversed trick
        fwd_min = series[::-1].rolling(right + 1, min_periods=1).min()[::-1]
        is_local_min = (series <= rolled_min) & (series <= fwd_min)
        # strictness: avoid flat shelves—require it to be strictly less than at least one neighbor
        strictly = (series < series.shift(1)) | (series < series.shift(-1))
        return (is_local_min & strictly).fillna(False)

    def detect_double_bottoms(
        self,
        left_right_window=3,          # how many bars on each side to define a trough
        min_separation=5,             # min bars between trough1 and trough2
        max_separation=60,            # max bars between troughs
        tol_type="atr",               # 'atr' or 'pct'
        tol_value=1.0,                # if 'atr': multiples of ATR; if 'pct': e.g. 0.006 = 0.6%
        atr_period=14,
        neckline_break="close",       # 'close' (stronger) or 'high' (earlier)
        confirm_within=15,            # must break neckline within N bars after trough2
        breakout_buffer_atr=0.1,      # how far above neckline (in ATRs) to count as break
        vol_confirm=False,            # require volume expansion on breakout
        vol_ma_period=20,             # compares breakout volume to MA(volume)
        require_divergence=False,     # optional RSI divergence filter
        rsi_period=14,
        risk_atr=1.5,                 # stop below min trough by this ATR multiple
        target_method="measured",     # 'measured' (neckline + height) or 'fixed_rr'
        fixed_rr=2.0                  # used only if target_method='fixed_rr'
    ):
        """
        Returns: list of dicts, each with:
          - t1_idx, t2_idx: indices of the two trough bars
          - neckline_idx, neckline_level
          - breakout_idx
          - entry, stop, target
          - height, rr, score
        """
        df = self.df
        out = []

        # Precompute ATR, RSI (if needed), vol MA (if needed)
        atr = self._atr(df, period=atr_period)
        if require_divergence:
            # simple RSI (wilders or SMA both ok for divergence). We'll use SMA RSI.
            delta = df['c'].diff()
            up = delta.clip(lower=0).rolling(rsi_period).mean()
            dn = (-delta.clip(upper=0)).rolling(rsi_period).mean()
            rs = up / (dn.replace(0, np.nan))
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = None

        if vol_confirm and 'v' in df.columns:
            vol_ma = df['v'].rolling(vol_ma_period, min_periods=1).mean()
        else:
            vol_ma = None

        # 1) Find candidate troughs
        trough_mask = self._local_minima(df['l'], left=left_right_window, right=left_right_window)
        trough_idxs = np.flatnonzero(trough_mask.values)

        if len(trough_idxs) < 2:
            return out

        # tolerance function (price similarity for the two lows)
        def troughs_close_enough(i1, i2):
            l1, l2 = df['l'].iloc[i1], df['l'].iloc[i2]
            if tol_type == "atr":
                # use average ATR around troughs
                base_atr = np.nanmean([atr.iloc[i1], atr.iloc[i2]])
                # protect against zero atr
                base_atr = base_atr if base_atr and np.isfinite(base_atr) else max(1e-8, df['c'].median() * 1e-5)
                return abs(l2 - l1) <= tol_value * base_atr
            else:  # pct
                mid = (l1 + l2) / 2.0
                return abs(l2 - l1) <= tol_value * mid

        # 2) Evaluate pairs of consecutive troughs
        n = len(df)
        for k in range(len(trough_idxs) - 1):
            i1 = trough_idxs[k]
            i2 = trough_idxs[k + 1]
            sep = i2 - i1
            if sep < min_separation or sep > max_separation:
                continue

            l1, l2 = df['l'].iloc[i1], df['l'].iloc[i2]

            # optional RSI divergence: price doesn't make a much lower low, but RSI second low is higher
            if require_divergence and rsi is not None:
                rsi1 = rsi.iloc[i1]
                rsi2 = rsi.iloc[i2]
                if not (np.isfinite(rsi1) and np.isfinite(rsi2)):
                    continue
                # bullish divergence condition: l2 >= l1 (or slightly lower) AND RSI2 > RSI1
                # allow small lower low if still within tolerance
                if not (troughs_close_enough(i1, i2) and (rsi2 > rsi1)):
                    continue

            # price proximity condition (two bottoms similar)
            if not troughs_close_enough(i1, i2):
                continue

            # 3) Neckline = highest high between the troughs
            mid_slice = df.iloc[i1:i2+1]
            neckline_level = mid_slice['h'].max()
            neckline_idx = mid_slice['h'].idxmax()

            # 4) Confirmation: breakout above neckline shortly after second trough
            search_end = min(n - 1, i2 + confirm_within)
            post = df.iloc[i2:search_end+1]
            if neckline_break == "close":
                buffer_px = breakout_buffer_atr * (atr.iloc[i2] if np.isfinite(atr.iloc[i2]) else 0.0)
                broke = post[post['c'] >= neckline_level + buffer_px]
            else:  # by high
                buffer_px = breakout_buffer_atr * (atr.iloc[i2] if np.isfinite(atr.iloc[i2]) else 0.0)
                broke = post[post['h'] >= neckline_level + buffer_px]

            if broke.empty:
                continue

            breakout_idx = broke.index[0]
            entry = df.loc[breakout_idx, 'c'] if neckline_break == "close" else max(df.loc[breakout_idx, 'o'], neckline_level)

            # volume expansion filter (optional)
            if vol_confirm and vol_ma is not None and 'v' in df.columns:
                if not (df.loc[breakout_idx, 'v'] > vol_ma.loc[breakout_idx]):
                    continue

            # 5) Risk / target
            base_atr = atr.iloc[[i1, i2]].mean()
            base_atr = base_atr if np.isfinite(base_atr) else max(1e-8, df['c'].median() * 1e-5)
            stop = min(l1, l2) - risk_atr * base_atr

            if target_method == "measured":
                height = neckline_level - min(l1, l2)   # measured move
                target = neckline_level + height
            else:  # fixed_rr
                risk = entry - stop
                height = neckline_level - min(l1, l2)
                target = entry + fixed_rr * risk

            rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else np.nan

            # 6) Simple score: combine symmetry, separation, and ATR-normalized proximity
            sym = 1.0 - (abs(l2 - l1) / max(base_atr, 1e-8)) / (tol_value + 1e-9)  # 1 is perfect
            sym = float(np.clip(sym, 0, 1))
            sep_norm = np.clip((sep - min_separation) / max(1, (max_separation - min_separation)), 0, 1)
            score = 0.55 * sym + 0.25 * sep_norm + 0.20 * (1 if not vol_confirm else (1 if df.loc[breakout_idx, 'v'] > vol_ma.loc[breakout_idx] else 0))

            out.append({
                "t1_idx": df.index[i1],
                "t2_idx": df.index[i2],
                "t1_low": float(l1),
                "t2_low": float(l2),
                "neckline_idx": neckline_idx,
                "neckline_level": float(neckline_level),
                "breakout_idx": breakout_idx,
                "entry": float(entry),
                "stop": float(stop),
                "target": float(target),
                "height": float(height),
                "rr": float(rr) if np.isfinite(rr) else None,
                "score": round(float(score), 3),
                "params": {
                    "tol_type": tol_type, "tol_value": tol_value,
                    "atr_period": atr_period, "risk_atr": risk_atr,
                    "confirm_within": confirm_within, "breakout_buffer_atr": breakout_buffer_atr
                }
            })

        return out



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



