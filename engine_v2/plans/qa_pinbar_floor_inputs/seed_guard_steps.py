"""PLAN_U (2026-10-28): how many 400-candle steps the seed guard needs before the frame to reach 2 windows (180)
of body-% (natural) pinbars, per TF — measured on a 60-day seed cut back step by step. Result: H1 2 steps (559 candles,
232 natural pinbars), M15 2 steps (480 candles, 189). Run from the repo root:
    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/seed_guard_steps.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))  # the repo root
import pandas as pd, numpy as np
import _frames as F
from engine_v2.config import CONFIG
from engine_v2.features.candle_params import CandleParams
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
raws = F.fetch_frames(("H1", "M15"))
for tf, hours in (("H1", 1.0), ("M15", 0.25)):
    t0 = pd.to_datetime(raws[tf]["time"]).min()
    seed = fetch_lower_tf_data(CONFIG.pair, tf, t0 - pd.Timedelta(days=60), t0)
    seed = seed[pd.to_datetime(seed["time"]) < t0].reset_index(drop=True)
    nat = F.classify(seed, CandleParams())
    t = pd.to_datetime(nat["time"])
    step = pd.Timedelta(hours=400 * hours)
    for k in range(1, 9):
        m = (t >= t0 - k * step).to_numpy()
        npin = int((nat["candle_type"].to_numpy()[m] == "pinbar").sum())
        print(tf, "step", k, "back", t0 - k * step, "candles", int(m.sum()), "natural pinbars", npin)
        if npin >= 180:
            break
