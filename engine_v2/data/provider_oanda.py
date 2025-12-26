from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from engine_v2.common.types import (
    COL_C,
    COL_H,
    COL_L,
    COL_O,
    COL_TIME,
    COL_V,
    REQUIRED_CANDLE_COLS,
)

# OANDA official environments (REST base URLs)
# Practice: https://api-fxpractice.oanda.com
# Live:     https://api-fxtrade.oanda.com
# Source: OANDA Development Guide
# (we'll cite this in docs; code keeps it explicit)
OANDA_REST_BASE = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


@dataclass(frozen=True)
class OandaCredentials:
    access_token: str
    account_id: str
    account_type: str  # "practice" or "live"


def _load_creds(
    cfg_path: str = "oanda.cfg",
    section: str = "oanda",
) -> OandaCredentials:
    """
    Credential loading precedence:
      1) Environment variables:
         OANDA_ACCESS_TOKEN, OANDA_ACCOUNT_ID, OANDA_ACCOUNT_TYPE
      2) Local cfg file (gitignored): oanda.cfg
    """
    env_token = os.getenv("OANDA_ACCESS_TOKEN")
    env_acct = os.getenv("OANDA_ACCOUNT_ID")
    env_type = os.getenv("OANDA_ACCOUNT_TYPE")

    if env_token and env_acct:
        account_type = (env_type or "practice").strip().lower()
        return OandaCredentials(access_token=env_token.strip(), account_id=env_acct.strip(), account_type=account_type)

    cp = configparser.ConfigParser()
    read_ok = cp.read(cfg_path)
    if not read_ok:
        raise FileNotFoundError(
            f"Could not find '{cfg_path}'. Create it locally (gitignored) or set "
            f"OANDA_ACCESS_TOKEN and OANDA_ACCOUNT_ID env vars."
        )
    if section not in cp:
        raise KeyError(f"Missing section [{section}] in {cfg_path}")

    account_id = cp.get(section, "account_id").strip()
    access_token = cp.get(section, "access_token").strip()
    account_type = cp.get(section, "account_type", fallback="practice").strip().lower()

    if not account_id or not access_token:
        raise ValueError(f"{cfg_path} must contain account_id and access_token in section [{section}]")

    if account_type not in OANDA_REST_BASE:
        raise ValueError(f"account_type must be one of {list(OANDA_REST_BASE.keys())}; got '{account_type}'")

    return OandaCredentials(access_token=access_token, account_id=account_id, account_type=account_type)


def _to_rfc3339(dt: datetime) -> str:
    # pandas handles timezone-aware timestamps well; just ISO format is OK for OANDA
    # Ensure you pass UTC datetimes in config for determinism.
    return dt.isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_end_to_now(end: datetime, buffer_seconds: int = 5) -> datetime:
    now = _utc_now().replace(microsecond=0)
    safe_now = now - timedelta(seconds=buffer_seconds)
    return safe_now if end > safe_now else end


def get_history(
    pair: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    price: str = "M",
    cfg_path: str = "oanda.cfg",
) -> pd.DataFrame:
    """
    Fetch historical candles from OANDA v20 InstrumentsCandles endpoint.

    Returns a DataFrame with:
    - required: time, o, h, l, c, volume
    """
    creds = _load_creds(cfg_path=cfg_path)

    base = OANDA_REST_BASE[creds.account_type]
    url = f"{base}/v3/instruments/{pair}/candles"

    headers = {
        "Authorization": f"Bearer {creds.access_token}",
        "Accept-Datetime-Format": "RFC3339",
    }

    end = _clamp_end_to_now(end)

    params = {
        "granularity": timeframe,
        "from": _to_rfc3339(start),
        "to": _to_rfc3339(end),
        "price": price,          # M=mid, B=bid, A=ask
        "includeFirst": "true",
        # "count": 5000,           # safe upper limit; Dec M15 is < 5000
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OANDA candles request failed {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    candles = payload.get("candles", [])

    rows = []
    for c in candles:
        # Keep only completed candles to avoid partial bar issues
        if not c.get("complete", True):
            continue

        t = c["time"]
        # Choose the requested price component
        # payload uses keys like "mid", "bid", "ask"
        key = {"M": "mid", "B": "bid", "A": "ask"}.get(price, "mid")
        p = c.get(key)
        if not p:
            continue

        rows.append(
            {
                COL_TIME: t,
                COL_O: float(p["o"]),
                COL_H: float(p["h"]),
                COL_L: float(p["l"]),
                COL_C: float(p["c"]),
                COL_V: int(c.get("volume", 0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df[COL_TIME] = pd.to_datetime(df[COL_TIME], utc=True)
    df = df.sort_values(COL_TIME).reset_index(drop=True)

    # Enforce required columns exist
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required candle columns: {missing}")

    # volume is optional; keep it if present
    return df
