
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# Final optimized swing detection configuration
BEST_NONZERO_CONFIG = ((3, 0.00003), (3, 0.00003), (2, 0.00003))

def compute_layer_swings(df, order, threshold):
    highs = df['h'].values
    lows = df['l'].values
    high_idx = argrelextrema(highs, np.greater, order=order)[0]
    low_idx = argrelextrema(lows, np.less, order=order)[0]
    swings = pd.Series(index=df.index, dtype=object)

    for idx in high_idx:
        if idx == 0 or idx >= len(df): continue
        if (highs[idx] - lows[max(0, idx - order)]) >= threshold:
            swings.iloc[idx] = 'high'

    for idx in low_idx:
        if idx == 0 or idx >= len(df): continue
        if (highs[max(0, idx - order)] - lows[idx]) >= threshold:
            swings.iloc[idx] = 'low'

    return swings


def apply_multi_layer_swing_detection(df, config):
    df = df.copy()
    l1_order, l1_thresh = config[0]
    l2_order, l2_thresh = config[1]
    l3_order, l3_thresh = config[2]

    # Layer 1
    l1_swings = compute_layer_swings(df, l1_order, l1_thresh)
    df['layer1_swing'] = l1_swings
    df['layer1_price'] = df['h'].where(l1_swings == 'high').fillna(df['l'].where(l1_swings == 'low'))

    # Layer 2
    df_l1 = df[df['layer1_swing'].notna()].copy()
    l2_swings = compute_layer_swings(df_l1, l2_order, l2_thresh)
    df['layer2_swing'] = l2_swings.reindex(df.index)
    df['layer2_price'] = df['h'].where(df['layer2_swing'] == 'high').fillna(df['l'].where(df['layer2_swing'] == 'low'))

    # Layer 3
    df_l2 = df[df['layer2_swing'].notna()].copy()
    l3_swings = compute_layer_swings(df_l2, l3_order, l3_thresh)
    df['layer3_swing'] = l3_swings.reindex(df.index)
    df['layer3_price'] = df['h'].where(df['layer3_swing'] == 'high').fillna(df['l'].where(df['layer3_swing'] == 'low'))

    return df


def find_algo_start_final_fixed(df, swing_col='layer3_swing', price_col='layer3_price'):
    swings = df[df[swing_col].notna()].copy()
    swings_rev = swings[::-1]  # Reverse chronological

    seen = []
    highs = 0
    lows = 0

    for idx, row in swings_rev.iterrows():
        swing_type = row[swing_col]
        seen.append((idx, swing_type))

        if swing_type == 'high':
            highs += 1
        elif swing_type == 'low':
            lows += 1

        if highs >= 2 and lows >= 2 and len(seen) >= 5:
            break

    if len(seen) < 4:
        return None

    candidate_idx, candidate_type = seen[3]
    candidate_price = df.loc[candidate_idx, price_col]

    prev_idx, prev_type = seen[4]
    if prev_type == candidate_type:
        prev_price = df.loc[prev_idx, price_col]
        if candidate_type == 'high' and prev_price > candidate_price:
            return prev_idx
        if candidate_type == 'low' and prev_price < candidate_price:
            return prev_idx

    return candidate_idx
