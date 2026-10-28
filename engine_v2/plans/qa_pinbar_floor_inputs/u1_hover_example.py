"""PLAN_U U1 (2026-10-28): the rendered candle-hover example shown to the user before U1 was locked — fills each
candle trace's hovertemplate from its customdata for a few candles (a plain floor line and a floor pinbar, H1 + the
M15 counter chart). Run from the repo root after a replay, or on a save folder:
    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u1_hover_example.py [CHART_DIR]
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root
chart_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/charts"
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402

STEM = chart_dir.rstrip("/\\") + "/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"

def fill(tpl, cd, tr, k):
    def sub(m):
        key, fmt = m.group(1), m.group(2)
        if key.startswith("customdata["):
            v = cd[int(key[11:-1])]
        else:
            arr = tr[key]
            if isinstance(arr, dict):
                import base64, numpy as np
                arr = np.frombuffer(base64.b64decode(arr["bdata"]), dtype=arr["dtype"])
            v = arr[k]
        if fmt:
            try:
                f = fmt[1:]
                v = format(float(v), f.replace("%", "%")) if not f.endswith("%") else format(float(v), f)
            except Exception:
                pass
        return str(v)
    out = re.sub(r"%\{([a-z_]+(?:\[\d+\])?)(:[^}]*)?\}", sub, tpl).replace("<extra></extra>", "")
    return out.replace("<br>", "\n    ")

for chart, wanted in (("", None), ("_M15_counter", None)):
    data, _ = load_fig(STEM + chart + ".html")
    shown = {"floor": 0, "plain": 0}
    for tr in data:
        if tr.get("type") != "candlestick":
            continue
        for k, cd in enumerate(tr["customdata"]):
            line = cd[12] if chart == "" else cd[14]
            kind = "floor" if "candle_type_before_floor" in line else "plain"
            want = {"floor": 2, "plain": 1}[kind]
            if shown[kind] >= want:
                continue
            if kind == "floor" and shown["floor"] == 1 and cd[3] == "pinbar" and "maru" not in line:
                continue
            shown[kind] += 1
            print(f"--- {chart or '_H1'} trace {tr.get('name')} ({kind}):\n    " + fill(tr["hovertemplate"], cd, tr, k))
