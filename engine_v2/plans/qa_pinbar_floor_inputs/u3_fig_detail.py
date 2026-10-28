"""PLAN_U U3 /compare: what each changed chart element of `u3_fig_named_diff.txt` holds (BASE = S_U2, CUR = the U3
variant / landing charts) — beyond the candle hovers (OHLC + imbalance traces: every candle's floor changes).

For each changed trace (name-keyed like `fig_named_diff.py`): its points before -> after (x added / removed; for the
unnamed traces their hovertemplate's first line); the changed customdata rows (by x); for the layout, every top-level
key that differs and, for `shapes`, each changed shape (positional — the shape counts are equal).

    PYTHONPATH=. PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3_fig_detail.py BASE CUR [OUT]
"""
import base64
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE, CUR = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).with_suffix(".txt"))
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_"
SKIP = ("OHLC", "imbalance:bearish", "imbalance:bullish")
MAX_CHANGED = 4   # changed points listed per trace (a zone's hover grid changes on every point)
lines = []


def canon(v):
    return json.dumps(v, sort_keys=True, default=str)


def keyed(traces):
    seen, out = Counter(), {}
    for t in traces:
        n = str(t.get("name"))
        out[(n, seen[n])] = t
        seen[n] += 1
    return out


def short(v, n=160):
    s = str(v).replace("<br>", " | ")
    return s if len(s) <= n else s[:n] + "…"


def arr(v):
    """A trace array: a list, or Plotly's binary form {"dtype", "bdata"[, "shape"]} decoded to a list."""
    if isinstance(v, dict) and "bdata" in v:
        a = np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"])
        if "shape" in v:
            a = a.reshape([int(s) for s in str(v["shape"]).split(",")])
        return a.tolist()
    return v


def points(t):
    xs = arr(t.get("x")) or []
    ys = arr(t.get("y")) or [None] * len(xs)
    cd = arr(t.get("customdata")) or [None] * len(xs)
    assert len(ys) == len(xs) and len(cd) == len(xs), (t.get("name"), len(xs), len(ys), len(cd))
    return [(str(x), y, canon(c)) for x, y, c in zip(xs, ys, cd)]


for suffix in ("rk2-5.html", "rk2-5_M15_counter.html", "rk2-5_M15_confluence.html"):
    a, la = load_fig(glob.glob(os.path.join(BASE, STEM + suffix))[0])
    b, lb = load_fig(glob.glob(os.path.join(CUR, STEM + suffix))[0])
    lines.append(f"== {suffix}")
    ka, kb = keyed(a), keyed(b)
    for k in sorted(set(ka) & set(kb)):
        ta, tb = ka[k], kb[k]
        if canon(ta) == canon(tb) or k[0] in SKIP:
            continue
        pa, pb = points(ta), points(tb)
        head = short((tb.get("hovertemplate") or "").split("<br>")[0], 80) if k[0] == "None" else ""
        lines.append(f"   {k} {head!s} points {len(pa)} -> {len(pb)}")
        sa, sb = set(pa), set(pb)
        xa = {p[0] for p in pa}
        xb = {p[0] for p in pb}
        for x in sorted(xa - xb):
            lines.append(f"      - x={x}")
        for x in sorted(xb - xa):
            lines.append(f"      + x={x}")
        changed = [p for p in sorted(sa - sb) if p[0] in xb]
        for p in changed[:MAX_CHANGED]:
            q = [r for r in pb if r[0] == p[0]]
            lines.append(f"      ~ x={p[0]} y {p[1]} -> {[r[1] for r in q]}")
            if q and p[2] != q[0][2]:
                lines.append(f"          customdata {short(p[2])}")
                lines.append(f"                  -> {short(q[0][2])}")
        if len(changed) > MAX_CHANGED:
            lines.append(f"      … {len(changed) - MAX_CHANGED} more changed points, x {changed[MAX_CHANGED][0]} .. "
                         f"{changed[-1][0]}")
        if k[0] == "None" and canon(ta.get("hovertemplate")) != canon(tb.get("hovertemplate")):
            lines.append(f"      hovertemplate {short(ta.get('hovertemplate'), 300)}")
            lines.append(f"                 -> {short(tb.get('hovertemplate'), 300)}")
    keys = sorted(k for k in set(la) | set(lb) if canon(la.get(k)) != canon(lb.get(k)))
    lines.append(f"   layout keys that differ: {keys}")
    for k in keys:
        va, vb = la.get(k), lb.get(k)
        if isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb):
            for i, (x, y) in enumerate(zip(va, vb)):
                if canon(x) != canon(y):
                    dk = sorted(z for z in set(x) | set(y) if canon(x.get(z)) != canon(y.get(z)))
                    lines.append(f"      {k}[{i}] " + "; ".join(f"{z}: {short(x.get(z), 70)} -> {short(y.get(z), 70)}"
                                                           for z in dk))
        elif not isinstance(va, list):
            lines.append(f"      {k}: {short(canon(va), 200)} -> {short(canon(vb), 200)}")

text = "\n".join(lines)
print(text)
open(OUT, "w", encoding="utf-8").write(text + "\n")
