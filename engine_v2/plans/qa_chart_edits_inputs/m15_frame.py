"""Rebuild the M15 candle frame (idx, time, OHLC) from an M15 chart HTML — the saves carry no M15 candle CSV.

Merges EVERY candlestick trace (the imbalance candles are separate traces: the first trace alone held 3386 of the
4228 candles on the reference window) and BINARY-decodes the OHLC arrays (`{"dtype", "bdata"}`), then calibrates the
entity-absolute idx offset from the sub dot hovers (`customdata[0]` = `idx=`; asserted constant — 0 on the reference
window). Written for the replacement-break edit, 2026-10-24.

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/m15_frame.py CHART_HTML [IDX ...]
Importable: frame(chart_html) -> DataFrame indexed by entity-absolute M15 idx.
"""
import base64
import re
import sys

import numpy as np
import pandas as pd

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv


def _arr(v):
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
    return np.asarray(v)


def frame(path):
    traces, _ = load_fig(path)
    parts = []
    for c in (t for t in traces if t.get("type") == "candlestick"):
        parts.append(pd.DataFrame({"time": pd.to_datetime(pd.Series(c["x"])), "open": _arr(c["open"]),
                                   "high": _arr(c["high"]), "low": _arr(c["low"]), "close": _arr(c["close"])}))
    df = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    pos = {t: i for i, t in enumerate(df["time"])}
    offsets = set()
    for t in traces:
        if not re.match(r"^M15 (CTS|BOS) sub\d+", str(t.get("name"))):
            continue
        tmpl = t.get("hovertemplate") or ""
        cd = t.get("customdata")
        for k, x in enumerate(t.get("x") or []):
            p = pos.get(pd.Timestamp(x).tz_localize(None) if pd.Timestamp(x).tzinfo else pd.Timestamp(x))
            if p is not None and cd is not None and k < len(cd) and tmpl.startswith("TF=15M<br>idx=%{customdata[0]}"):
                offsets.add(int(cd[k][0]) - p)
    assert len(offsets) == 1, offsets
    off = offsets.pop()
    df.index = df.index + off
    df.attrs["offset"] = off
    return df


if __name__ == "__main__":
    f = frame(sys.argv[1])
    print("offset", f.attrs["offset"], "rows", f.index[0], "->", f.index[-1])
    for i in sys.argv[2:]:
        print(f.loc[int(i)].to_dict())
