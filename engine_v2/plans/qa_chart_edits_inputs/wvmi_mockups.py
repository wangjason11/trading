"""WVMI back on the chart — MOCK-UPS of options 2 and 4 (the user, 2026-10-24: "I want to compare how 2 would look vs.
4"; scope: every chart, collapsed cycles hidden). Drawn onto COPIES of a save's chart HTMLs (the page's JS kept, the
new traces appended last); no engine change.

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/wvmi_mockups.py SAVE_DIR OUT_DIR

Writes OUT_DIR/{h1,counter,confluence}_opt{2,4}.html.
- Option 2: the wave-candle hover gains the cycle's WVMI, matched by owner + cycle + role (never by idx): FB / LB
  lines the breakout momentum, FP / LP lines the pullback momentum (an LP line only when the record is LOCKED and its
  LP is that line); an OPEN record's temporary LP gets a hollow diamond at its candle's close with the pullback hover.
- Option 4: a log-scale WVMI strip between the price and the volume panels — per record a breakout segment FB → LB at
  the breakout momentum and a pullback segment FP → LP at the pullback momentum (buy-coloured / sell-coloured by the
  wave's direction; H1 records solid, M15 sub records dashed — the wave-candle line convention; an open record's LP end
  is a hollow circle), a dotted reference at 1.0.
Mock-up only: the WVMI CSV exports no volumes / weights, so volumes come from the frames (H1 `_final.csv`, the M15
chart's Volume bars) and the weights are back-solved from the momentum (snapped to 1.0 / 0.7 / 0.5); the H1 records on
the M15 charts sit at the M15 x the H1 overlay's wave-candle lines use (offset measured from those lines).
"""
import base64
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m15_frame import frame as m15_frame  # noqa: E402

SAVE, OUT = sys.argv[1], sys.argv[2]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
os.makedirs(OUT, exist_ok=True)
GREEN, RED = "rgba(0,140,60,0.95)", "rgba(200,30,30,0.95)"
_WC = re.compile(r"TF=(\w+)<br><b>Wave Candle</b><br>idx=(\d+).*?(BOS|CTS) zone: (sid|sub_id)=(\d+) \| cycle_id=(\d+)"
                 r".*?Role: (FB|LB|FP|LP) \(([^)]*)\)")


# ---------------------------------------------------------------- page I/O (keep the page, swap data + layout)
def read_page(path):
    html = open(path, encoding="utf-8").read()
    m = re.search(r"Plotly\.newPlot\(", html)
    i = html.index("[", m.end())
    dec = json.JSONDecoder()
    data, i_end = dec.raw_decode(html, i)
    j = html.index("{", i_end)
    layout, j_end = dec.raw_decode(html, j)
    return html, (i, i_end, data), (j, j_end, layout)


def write_page(path, html, dspan, lspan, data, layout):
    i, i_end, _ = dspan
    j, j_end, _ = lspan
    out = html[:i] + json.dumps(data) + html[i_end:j] + json.dumps(layout) + html[j_end:]
    open(path, "w", encoding="utf-8").write(out)


def _arr(v):
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
    return np.asarray(v)


# ---------------------------------------------------------------- frames
h1 = pd.read_csv(os.path.join(SAVE, "NZD_USD_H1_2025-11-15_2026-01-20_final.csv"))
h1["time"] = pd.to_datetime(h1["time"], utc=True).dt.tz_localize(None)


def m15_full(path, data):
    f = m15_frame(path)
    bar = next(t for t in data if t.get("type") == "bar" and t.get("yaxis") == "y2")
    vol = pd.Series(_arr(bar["y"]), index=pd.to_datetime(pd.Series(bar["x"])).dt.tz_localize(None)
                    if pd.to_datetime(pd.Series(bar["x"])).dt.tz is not None else pd.to_datetime(pd.Series(bar["x"])))
    f["volume"] = f["time"].map(vol)
    return f


def snap_weight(w):
    return min((1.0, 0.7, 0.5), key=lambda c: abs(c - w)) if w is not None and not math.isnan(w) else None


def enrich(rec, fr):
    """Volumes + back-solved weights for a WVMI CSV row (a dict) on its own frame `fr` (indexed by idx)."""
    r = dict(rec)
    for k in ("fb", "lb", "fp", "lp"):
        i = r[f"{k}_idx"]
        r[f"{k}_volume"] = float(fr.loc[int(i), "volume"]) if not pd.isna(i) else None
    r["lb_weight"] = snap_weight(r["breakout_momentum"] * r["fb_volume"] / r["lb_volume"]) if r["lb_volume"] else None
    r["lp_weight"] = (snap_weight(r["pullback_momentum"] * r["fp_volume"] / r["lp_volume"])
                      if r["lp_volume"] and not pd.isna(r["pullback_momentum"]) else None)
    return r


def records(csv, fr, owner_kind):
    d = pd.read_csv(csv)
    d = d[~d["cycle_collapsed"].astype(bool)]                          # the user: collapsed cycles stay hidden
    out = {}
    for _, row in d.iterrows():
        r = enrich(row.to_dict(), fr)
        owner = int(r["sub_id"]) if owner_kind == "sub_id" else int(r["bos_structure_id"])
        out[(owner, int(r["bos_cycle_id"]))] = r
    return out


# ---------------------------------------------------------------- hover text (one builder, both options)
def _dir(r, wave):
    buy = (r["zone_side"] == "buy") == (wave == "breakout")
    return "buy" if buy else "sell"


def _f(v, nd=3):
    return "NA" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _v(v):
    return "NA" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0f}"


def lp_state(r):
    if pd.isna(r["lp_idx"]):
        return "lp=NA (no temporary LP: the cycle ended by FP + 1)"
    if bool(r["lp_locked"]):
        return f"lp=locked (by cycle {int(r['locked_by_cycle_id'])}'s BOS)"
    return "lp=temporary (moves until the next BOS locks it) | status=" + str(r["status"])


def breakout_lines(r):
    return [f"breakout_momentum={_f(r['breakout_momentum'])} ({_dir(r, 'breakout')}) | fb_idx={int(r['fb_idx'])} | "
            f"lb_idx={int(r['lb_idx'])}",
            f"fb_volume={_v(r['fb_volume'])} | lb_volume={_v(r['lb_volume'])} | lb_weight={_f(r['lb_weight'], 1)}"]


def pullback_lines(r):
    lp = "NA" if pd.isna(r["lp_idx"]) else int(r["lp_idx"])
    return [f"pullback_momentum={_f(r['pullback_momentum'])} ({_dir(r, 'pullback')}) | fp_idx={int(r['fp_idx'])} | "
            f"lp_idx={lp}",
            f"fp_volume={_v(r['fp_volume'])} | lp_volume={_v(r['lp_volume'])} | lp_weight={_f(r['lp_weight'], 1)}",
            lp_state(r)]


def ident(owner_kind, owner, cyc):
    return f"{owner_kind}={owner} | cycle_id={cyc}"


# ---------------------------------------------------------------- option 2
def option2(data, recs_by_tf, x_of, close_of, owner_kind_by_tf):
    """Append WVMI to matched wave-candle hovers; return the temporary-LP marker traces."""
    added = Counter()
    for t in data:
        tmpl = t.get("hovertemplate") or ""
        m = _WC.search(tmpl)
        if not m:
            continue
        tf, idx, _zk, ok, owner, cyc, role, name = m.groups()
        idx, owner, cyc = int(idx), int(owner), int(cyc)
        recs = recs_by_tf.get(tf, {})
        if role == "LP":
            c = re.search(r"of cycle (\d+)", name)
            r = recs.get((owner, int(c.group(1)))) if c else None
            if r is None or not bool(r["lp_locked"]) or pd.isna(r["lp_idx"]) or int(r["lp_idx"]) != idx:
                continue
            lines = ["<b>WVMI</b> (this LP's cycle)"] + pullback_lines(r)
        else:
            r = recs.get((owner, cyc))
            if r is None or int(r[f"{role.lower()}_idx"]) != idx:
                continue
            lines = ["<b>WVMI</b>"] + (breakout_lines(r) if role in ("FB", "LB") else pullback_lines(r))
        t["hovertemplate"] = tmpl.replace("<extra></extra>", "<br>" + "<br>".join(lines) + "<extra></extra>")
        added[(tf, role)] += 1
    traces = []
    for tf, recs in recs_by_tf.items():
        xs, ys, texts = [], [], []
        for (owner, cyc), r in sorted(recs.items()):
            if bool(r["lp_locked"]) or pd.isna(r["lp_idx"]):
                continue
            i = int(r["lp_idx"])
            xs.append(x_of(tf, i))
            ys.append(close_of(tf, i))
            texts.append("<br>".join([f"TF={tf}", "<b>WVMI temporary LP</b>", f"idx={i}",
                                      ident(owner_kind_by_tf[tf], owner, cyc)] + pullback_lines(r)))
        if xs:
            traces.append(dict(type="scatter", mode="markers", x=xs, y=ys, text=texts,
                               hovertemplate="%{text}<extra></extra>", name=f"WVMI temporary LP ({tf})",
                               marker=dict(symbol="diamond-open", size=12 if tf == "1H" else 9,
                                           color="rgba(110,40,160,1)", line=dict(width=2))))
    print("   option 2: wave-candle hovers extended", dict(added), "| temporary-LP markers",
          {t["name"]: len(t["x"]) for t in traces})
    return traces


# ---------------------------------------------------------------- option 4
def option4(data, layout, recs_by_tf, x_of, owner_kind_by_tf):
    layout["yaxis"]["domain"] = [0.33, 1.0]
    layout["yaxis2"]["domain"] = [0.0, 0.15]
    layout["yaxis3"] = dict(domain=[0.175, 0.31], anchor="x", type="log", range=[math.log10(0.05), math.log10(12)],
                            tickvals=[0.1, 0.25, 0.5, 1, 2, 4, 8], ticktext=["0.1", "0.25", "0.5", "1", "2", "4", "8"],
                            title=dict(text="WVMI", font=dict(size=11)), showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                            zeroline=False, side="left")
    layout.setdefault("shapes", []).append(dict(type="line", xref="paper", x0=0, x1=1, yref="y3", y0=1, y1=1,
                                                line=dict(color="rgba(0,0,0,0.45)", width=1, dash="dot")))
    layout["shapes"].append(dict(type="line", xref="paper", x0=0, x1=1, yref="paper", y0=0.32, y1=0.32,
                                 line=dict(color="rgba(0,0,0,0.25)", width=1)))
    traces, seen = [], set()
    for tf, recs in recs_by_tf.items():
        for (owner, cyc), r in sorted(recs.items()):
            for wave, a, b, mom in (("breakout", "fb_idx", "lb_idx", "breakout_momentum"),
                                    ("pullback", "fp_idx", "lp_idx", "pullback_momentum")):
                if pd.isna(r[b]) or pd.isna(r[mom]):
                    continue
                i0, i1 = int(r[a]), int(r[b])
                pts = sorted(set(np.linspace(i0, i1, 10).round().astype(int).tolist()))
                color = GREEN if _dir(r, wave) == "buy" else RED
                open_end = wave == "pullback" and not bool(r["lp_locked"])
                size = [8] + [0] * (len(pts) - 2) + [8] if len(pts) > 1 else [8]
                sym = ["circle"] * (len(pts) - 1) + ["circle-open" if open_end else "circle"]
                text = "<br>".join([f"TF={tf}", f"<b>WVMI {wave}</b>", ident(owner_kind_by_tf[tf], owner, cyc)]
                                   + (breakout_lines(r) if wave == "breakout" else pullback_lines(r)))
                grp = f"WVMI ({tf})"
                traces.append(dict(type="scatter", mode="lines+markers", xaxis="x", yaxis="y3",
                                   x=[x_of(tf, p) for p in pts], y=[float(r[mom])] * len(pts),
                                   line=dict(color=color, width=3 if tf == "1H" else 2,
                                             dash="solid" if tf == "1H" else "dash"),
                                   marker=dict(size=size, symbol=sym, color=color, line=dict(width=2, color=color)),
                                   text=[text] * len(pts), hovertemplate="%{text}<extra></extra>",
                                   name=grp, legendgroup=grp, showlegend=grp not in seen))
                seen.add(grp)
    print("   option 4: strip segments", Counter(t["name"] for t in traces))
    return traces


# ---------------------------------------------------------------- per chart
h1_csv = os.path.join(SAVE, f"{STEM}_wvmi.csv")
h1_by_idx = h1.set_index(h1.index)


def build(chart, path, lens):
    html, dspan, lspan = read_page(path)
    data, layout = dspan[2], lspan[2]
    recs_by_tf = {"1H": records(h1_csv, h1_by_idx, "sid")}
    owner_kind_by_tf = {"1H": "sid", "15M": "sub_id"}
    if lens is None:
        def x_of(tf, i):
            return h1.loc[i, "time"].isoformat()

        def close_of(tf, i):
            return float(h1.loc[i, "c"])
    else:
        fr = m15_full(path, data)
        recs_by_tf["15M"] = records(os.path.join(SAVE, f"{STEM}_M15_{lens}_wvmi.csv"), fr, "sub_id")
        # where the H1 overlay draws an H1 candle: measured from its wave-candle lines
        deltas = Counter()
        for t in data:
            m = _WC.search(t.get("hovertemplate") or "")
            if m and m.group(1) == "1H":
                deltas[pd.Timestamp(t["x"][0]).tz_localize(None) - h1.loc[int(m.group(2)), "time"]
                       if pd.Timestamp(t["x"][0]).tzinfo is None else
                       pd.Timestamp(t["x"][0]).tz_convert("UTC").tz_localize(None) - h1.loc[int(m.group(2)), "time"]] += 1
        off = deltas.most_common(1)[0][0]
        print(f"   H1 overlay x offset: {off} ({dict(deltas)})")

        def x_of(tf, i):
            return (h1.loc[i, "time"] + off).isoformat() if tf == "1H" else fr.loc[i, "time"].isoformat()

        def close_of(tf, i):
            return float(h1.loc[i, "c"]) if tf == "1H" else float(fr.loc[i, "close"])
    print(f"== {chart}: records (non-collapsed) " + ", ".join(f"{k} {len(v)}" for k, v in recs_by_tf.items()))
    d2 = json.loads(json.dumps(data))
    extra2 = option2(d2, recs_by_tf, x_of, close_of, owner_kind_by_tf)
    write_page(os.path.join(OUT, f"{chart}_opt2.html"), html, dspan, lspan, d2 + extra2, layout)
    d4, l4 = json.loads(json.dumps(data)), json.loads(json.dumps(layout))
    extra4 = option4(d4, l4, recs_by_tf, x_of, owner_kind_by_tf)
    write_page(os.path.join(OUT, f"{chart}_opt4.html"), html, dspan, lspan, d4 + extra4, l4)


build("h1", os.path.join(SAVE, f"{STEM}.html"), None)
build("counter", os.path.join(SAVE, f"{STEM}_M15_counter.html"), "counter")
build("confluence", os.path.join(SAVE, f"{STEM}_M15_confluence.html"), "confluence")
print("wrote", OUT)
