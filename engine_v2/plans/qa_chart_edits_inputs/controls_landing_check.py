"""The landing check of item 5's final part (the control bar, 2026-10-27) against the prediction in
`user_spec_20261027_parent_toggle_and_zoom.md` "The plan + prediction": BASE = a save folder (S_T2), CUR = a replay's
`artifacts/debug` + `artifacts/charts` (or a save folder for both).

  * the 30 CSVs byte-equal (md5);
  * the H1 chart: the figure JSON equal (data + layout);
  * each M15 chart: the same traces / shapes in the same order, the ones that differ differ ONLY by
    `legendgroup = "parent_overlay"`, and they are exactly the ones the prototype's name / colour classifier picks
    (`proto_toggle_zoom.is_parent_trace` / `is_parent_shape`) — 60 traces + 40 shapes on S_T2; the layout otherwise
    equal;
  * the HTML: one control script per chart; its candle table (rows, the idx run, idx_1H).

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/controls_landing_check.py BASE CUR_DEBUG CUR_CHARTS
"""
import glob
import hashlib
import json
import os
import sys

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto_toggle_zoom import is_parent_shape, is_parent_trace  # noqa: E402

BASE, CUR_DEBUG, CUR_CHARTS = sys.argv[1], sys.argv[2], sys.argv[3]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_"
GROUP = "parent_overlay"
ok = True


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def canon(v):
    return json.dumps(v, sort_keys=True, default=str)


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(("  ok   " if cond else "  FAIL ") + what)


# --- the CSVs ---
base_csv = sorted(glob.glob(os.path.join(BASE, "*.csv")))
same = [os.path.basename(p) for p in base_csv
        if os.path.exists(os.path.join(CUR_DEBUG, os.path.basename(p)))
        and md5(p) == md5(os.path.join(CUR_DEBUG, os.path.basename(p)))]
print(f"CSVs: {len(same)} / {len(base_csv)} byte-equal")
check(len(base_csv) == 30 and len(same) == 30, "30 / 30 CSVs byte-equal")

# --- the charts ---
for suffix in ("rk2-5.html", "rk2-5_M15_counter.html", "rk2-5_M15_confluence.html"):
    print(f"== {suffix}")
    bpath = glob.glob(os.path.join(BASE, STEM + suffix))[0]
    cpath = glob.glob(os.path.join(CUR_CHARTS, STEM + suffix))[0]
    bd, bl = load_fig(bpath)
    cd, cl = load_fig(cpath)
    check(len(bd) == len(cd) and len(bl.get("shapes") or []) == len(cl.get("shapes") or []),
          f"counts {len(cd)} traces / {len(cl.get('shapes') or [])} shapes (base {len(bd)} / {len(bl.get('shapes') or [])})")
    m15 = "M15" in suffix
    diff_tr, diff_sh = [], []
    for k, (b, c) in enumerate(zip(bd, cd)):
        if canon(b) != canon(c):
            diff_tr.append(k)
            c2 = {key: v for key, v in c.items() if not (key == "legendgroup" and v == GROUP)}
            check(canon(b) == canon(c2), f"trace {k} {c.get('name')!r} differs only by legendgroup") if canon(b) != canon(c2) else None
    for k, (b, c) in enumerate(zip(bl.get("shapes") or [], cl.get("shapes") or [])):
        if canon(b) != canon(c):
            diff_sh.append(k)
            c2 = {key: v for key, v in c.items() if not (key == "legendgroup" and v == GROUP)}
            check(canon(b) == canon(c2), f"shape {k} differs only by legendgroup") if canon(b) != canon(c2) else None
    want_tr = [k for k, t in enumerate(bd) if m15 and is_parent_trace(t)]
    want_sh = [k for k, s in enumerate(bl.get("shapes") or []) if m15 and is_parent_shape(s)]
    check(diff_tr == want_tr, f"{len(diff_tr)} traces tagged = the classifier's {len(want_tr)}")
    check(diff_sh == want_sh, f"{len(diff_sh)} shapes tagged = the classifier's {len(want_sh)}")
    bl2 = {k: v for k, v in bl.items() if k != "shapes"}
    cl2 = {k: v for k, v in cl.items() if k != "shapes"}
    check(canon(bl2) == canon(cl2), "layout (without shapes) equal")
    html = open(cpath, encoding="utf-8").read()
    check(html.count("(function initChartControls()") == 1 and "__CFG__" not in html, "one control script")
    cfg = json.JSONDecoder().raw_decode(html, html.index("var CFG = ") + len("var CFG = "))[0]
    rows = cfg["candles"]
    idx = [r[0] for r in rows]
    line = (f"cfg tf={cfg['tf']} parentTf={cfg['parentTf']} parentGroup={cfg['parentGroup']} half={cfg['half']}; "
            f"{len(rows)} candles, idx {idx[0]}..{idx[-1]} contiguous={idx == list(range(idx[0], idx[-1] + 1))}")
    if m15:
        h = [r[4] for r in rows]
        line += f"; idx_1H {min(x for x in h if x is not None)}..{max(x for x in h if x is not None)}, None {h.count(None)}"
    print("  " + line)
    check((len(rows), idx[0], idx[-1]) == ((4228, 0, 4227) if m15 else (1058, 0, 1057)), "the candle table's size")
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
