"""PROTOTYPE of item 5's final part (2026-10-27): copies a save's three chart HTMLs with the chart control bar
(`proto_chart_controls.js`) appended, and on the two M15 charts the parent (1H) elements tagged
`legendgroup="parent_h1"` (the button's target) — no engine change, for the user's look before the plan.

The parent set = the user's round-1 coverage, "All except both": every H1-overlay trace and shape except the
zone-proximity triggers and the H1 WVMI LP circles (60 traces + 40 shapes on S_T2 — asserted). The prototype finds them
by name / colour (the engine build tags them where they are drawn).

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/proto_toggle_zoom.py SAVE_DIR OUT_DIR
"""
import glob
import json
import os
import re
import sys

STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_"
HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = "parent_overlay"   # the engine's PARENT_OVERLAY_GROUP (was "parent_h1" in the first prototype)
H1_ALPHAS = ("0.08", "0.18")   # the H1 overlay's tier-0.2 opacities (KL fill 0.4 x 0.2; KL / POI lines + POI fill 0.9 x 0.2)


def split_fig(html):
    """(data, layout, the spans to splice back) — the `chart_census.load_fig` parse, keeping the positions."""
    m = re.search(r"Plotly\.newPlot\(", html)
    dec = json.JSONDecoder()
    i = html.index("[", m.end())
    data, i_end = dec.raw_decode(html, i)
    j = html.index("{", i_end)
    layout, j_end = dec.raw_decode(html, j)
    return data, layout, (i, i_end, j, j_end)


def is_parent_trace(t):
    name = t.get("name")
    if name:
        return name.startswith("H1 ") and not name.startswith("H1 WVMI LP")
    return str(t.get("hovertemplate") or "").startswith("TF=1H<br>")   # the H1 wave-candle hover hitboxes


def _alpha(color):
    m = re.match(r"rgba\(([^)]*)\)", str(color or "").replace(" ", ""))
    return m.group(1).split(",")[3] if m else None


def is_parent_shape(s):
    if s.get("xref") != "x":
        return False                                    # the chart border (paper lines)
    line = s.get("line") or {}
    if s.get("yref") == "paper":
        return s.get("type") == "line" and not line.get("dash")   # H1 wave-candle lines are solid, the M15 ones dashed
    for c in (s.get("fillcolor"), line.get("color")):
        a = _alpha(c)
        if a is not None and any(a.startswith(h) for h in H1_ALPHAS):
            return True
    return False


if __name__ == "__main__":                      # the classifiers above are imported by controls_landing_check.py
    SAVE, OUT = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    js = open(os.path.join(HERE, "proto_chart_controls.js"), encoding="utf-8").read()
    for suffix, tf in (("rk2-5.html", "1H"), ("rk2-5_M15_counter.html", "15M"), ("rk2-5_M15_confluence.html", "15M")):
        src = glob.glob(os.path.join(SAVE, STEM + suffix))[0]
        html = open(src, encoding="utf-8").read()
        data, layout, (i, i_end, j, j_end) = split_fig(html)
        n_tr = n_sh = 0
        if tf == "15M":
            for t in data:
                if is_parent_trace(t):
                    t["legendgroup"] = GROUP
                    n_tr += 1
            for s in layout.get("shapes") or []:
                if is_parent_shape(s):
                    s["legendgroup"] = GROUP
                    n_sh += 1
            assert (n_tr, n_sh) == (60, 40), (suffix, n_tr, n_sh)
        cfg = dict(tf=tf, group=GROUP if tf == "15M" else None, half=48, pad=0.06, minN=10,   # H1 24 -> 48 (round 2)
                   wheel=1.2, markColor="rgba(30,120,255,0.18)")
        html = (html[:i] + json.dumps(data) + html[i_end:j] + json.dumps(layout) + html[j_end:]
                + "\n<script>\n" + js.replace("__CFG__", json.dumps(cfg)) + "\n</script>\n")
        dst = os.path.join(OUT, os.path.basename(src))
        open(dst, "w", encoding="utf-8").write(html)
        print(f"{suffix}: tagged {n_tr} traces + {n_sh} shapes -> {dst}")
