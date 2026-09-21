"""Per-element / per-sub trace census of a saved M15 chart .html.

Usage: PYTHONPATH=. python engine_v2/debug/chart_census.py <chart.html> [<chart_b.html>]
With two files it prints a side-by-side census diff.
"""
import json, re, sys
from collections import Counter


def load_fig(path):
    html = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r'Plotly\.newPlot\(', html)
    i = html.index('[', m.end())
    dec = json.JSONDecoder()
    data, end = dec.raw_decode(html, i)
    j = html.index('{', end)
    layout, _ = dec.raw_decode(html, j)
    return data, layout


def census(path):
    data, layout = load_fig(path)
    c = Counter()
    for tr in data:
        name = tr.get("name") or ""
        ht = tr.get("hovertemplate") or ""
        cd = tr.get("customdata") or []
        if name == "M15 KL zone":
            r = cd[0]
            c[("KL_hover", (r[9], r[10], r[1]))] += 1
        elif name == "M15 POI zone":
            r = cd[0]
            c[("POI_hover", (r[10], r[11], r[1]))] += 1
        elif name.startswith("M15 KL outline"):
            mm = re.search(r'h1s(\S+)c(\S+)_sub(\d+)', name)
            c[("KL_outline", (mm.group(1), mm.group(2), mm.group(3)) if mm else name)] += 1
        elif re.match(r'^M15 (CTS \(unconf\)|CTS|BOS|PB) ', name):
            kind = re.match(r'^M15 (CTS \(unconf\)|CTS|BOS|PB) ', name).group(1)
            r = cd[0]
            c[(f"dots_{kind}", (r[6], r[7], r[3]))] += 1
        elif name.startswith("M15 swing"):
            mm = re.search(r'h1s(\S+)c(\S+)_m15s(\d+)', name)
            c[("swing", (mm.group(1), mm.group(2), mm.group(3)) if mm else name)] += 1
        elif name.startswith("M15 PB"):
            c[("pb_to_bos", name)] += 1
        elif name.startswith("M15 Prev BOS"):
            c[("prev_bos", name)] += 1
        elif not name and ht.startswith("TF=15M<br><b>Wave Candle"):
            mm = re.search(r'sub_id=(\d+) cycle=(-?\d+)<br>first record parent_sid=(\S+) parent_cycle=(\S+)', ht)
            key = (mm.group(3), mm.group(4), mm.group(1)) if mm else "?"
            c[("wave_hover_M15", key)] += 1
        elif not name and ht.startswith("TF=1H<br><b>Wave Candle"):
            c[("wave_hover_H1", "-")] += 1
        elif name.startswith("H1 ") or name in ("zone_proximity:trigger",):
            c[("H1_overlay", name.split(" sid=")[0].split(" (sid")[0])] += 1
        else:
            c[("base", name)] += 1
    shapes = layout.get("shapes", []) or []
    for sh in shapes:
        if sh.get("type") == "line" and sh.get("yref") == "paper":
            c[("shape_wave_vline", "-")] += 1
        elif sh.get("type") == "line":
            c[("shape_confirm_line", (sh.get("line") or {}).get("color"))] += 1
        elif sh.get("type") == "rect":
            fc = sh.get("fillcolor")
            c[("shape_rect_outline" if fc == "rgba(0,0,0,0)" else "shape_rect_fill", fc)] += 1
        else:
            c[("shape_other", sh.get("type"))] += 1
    c[("TOTAL_TRACES", "-")] = len(data)
    c[("TOTAL_SHAPES", "-")] = len(shapes)
    return c


cs = [census(p) for p in sys.argv[1:]]
keys = sorted(set().union(*[set(c) for c in cs]), key=lambda k: (str(k[0]), str(k[1])))
if len(cs) == 1:
    for k in keys:
        print(f"{cs[0][k]:6d}  {k[0]:20s} {k[1]}")
else:
    print(f"{'A':>6} {'B':>6} {'d':>6}  element / (parent_sid,parent_cycle,sub_id)")
    for k in keys:
        a, b = cs[0][k], cs[1][k]
        flag = "" if a == b else "   <<<"
        print(f"{a:6d} {b:6d} {b-a:6d}  {k[0]:20s} {k[1]}{flag}")
