"""Wave-candle hover scope: BASE vs CUR — traces equal except hover fields; every changed trace is a wave-candle hover
on both sides; counts per chart and TF. The wave-candle traces are UNNAMED, so `fig_scope.py`'s name regex cannot
select them; this matches the hovertemplate's `<b>Wave Candle</b>` (the 2026-10-23 wave-candle hover edit).

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/wc_scope.py BASE_SAVE_DIR CUR_SAVE_DIR
"""
import json
import os
import sys
from collections import Counter

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE, CUR = sys.argv[1], sys.argv[2]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
HOVER = ("customdata", "hovertemplate")


def canon(v):
    return json.dumps(v, sort_keys=True, default=str)


ok = True
for suffix in (".html", "_M15_counter.html", "_M15_confluence.html"):
    a, la = load_fig(os.path.join(BASE, STEM + suffix))
    b, lb = load_fig(os.path.join(CUR, STEM + suffix))
    same = len(a) == len(b) and canon(la) == canon(lb)
    changed, bad = Counter(), []
    unchanged_wc = 0
    for i, (ta, tb) in enumerate(zip(a, b)):
        if canon({k: v for k, v in ta.items() if k not in HOVER}) != canon({k: v for k, v in tb.items() if k not in HOVER}):
            bad.append((i, "non-hover field", ta.get("name")))
        hov = any(canon(ta.get(k)) != canon(tb.get(k)) for k in HOVER)
        wa = "<b>Wave Candle</b>" in (ta.get("hovertemplate") or "")
        wb = "<b>Wave Candle</b>" in (tb.get("hovertemplate") or "")
        if hov:
            if wa and wb:
                changed[(tb["hovertemplate"].split("<br>")[0])] += 1
            else:
                bad.append((i, "hover changed on a non-wave-candle trace", ta.get("name")))
        elif wa:
            unchanged_wc += 1
    print(f"== {suffix}: traces {len(a)} -> {len(b)}, layout equal {canon(la) == canon(lb)}; wave-candle hovers "
          f"changed {dict(changed)}; unchanged wave-candle hovers {unchanged_wc}; violations {bad}")
    ok &= same and not bad and unchanged_wc == 0
print("WAVE SCOPE OK" if ok else "WAVE SCOPE VIOLATION")
