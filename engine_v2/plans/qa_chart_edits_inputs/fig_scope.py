"""Figure-only scope check: BASE vs CUR chart HTMLs — same traces in order, layout equal, and every trace equal
except `customdata` / `hovertemplate`; prints which traces' hover fields changed (name -> count), and flags any
change outside the expected dot traces.

    PYTHONPATH=. python fig_scope.py BASE_DIR CUR_DIR
"""
import glob
import json
import os
import re
import sys
from collections import Counter

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE, CUR = sys.argv[1], sys.argv[2]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_"
HOVER = ("customdata", "hovertemplate")
DOT = re.compile(r"^(CTS \(confirmed\)|BOS \(confirmed\)|CTS \(unconfirmed\)|PB \(furthest\)|H1 (CTS|BOS|PB)$|"
                 r"M15 (CTS|BOS|CTS \(unconf\)|PB) sub\d+)")


def canon(v):
    return json.dumps(v, sort_keys=True, default=str)


ok = True
for suffix in ("rk2-5.html", "rk2-5_M15_counter.html", "rk2-5_M15_confluence.html"):
    a, la = load_fig(glob.glob(os.path.join(BASE, STEM + suffix))[0])
    b, lb = load_fig(glob.glob(os.path.join(CUR, STEM + suffix))[0])
    print(f"== {suffix}: traces {len(a)} -> {len(b)}; layout equal: {canon(la) == canon(lb)}")
    ok &= len(a) == len(b) and canon(la) == canon(lb)
    changed, bad = Counter(), []
    for i, (ta, tb) in enumerate(zip(a, b)):
        if ta.get("name") != tb.get("name"):
            bad.append((i, "name", ta.get("name"), tb.get("name")))
            continue
        other_a = {k: v for k, v in ta.items() if k not in HOVER}
        other_b = {k: v for k, v in tb.items() if k not in HOVER}
        if canon(other_a) != canon(other_b):
            bad.append((i, "non-hover field", ta.get("name"),
                        sorted(k for k in set(other_a) | set(other_b) if canon(other_a.get(k)) != canon(other_b.get(k)))))
        if any(canon(ta.get(k)) != canon(tb.get(k)) for k in HOVER):
            changed[ta.get("name")] += 1
            if not DOT.match(str(ta.get("name"))):
                bad.append((i, "hover changed on a non-dot trace", ta.get("name")))
    unchanged_dots = [str(t.get("name")) for t, u in zip(a, b) if DOT.match(str(t.get("name")))
                      and all(canon(t.get(k)) == canon(u.get(k)) for k in HOVER)]
    print(f"   hover changed on {sum(changed.values())} traces: {dict(changed)}")
    print(f"   dot traces with UNCHANGED hover: {unchanged_dots}")
    for x in bad:
        print("   !!", x)
    ok &= not bad and not unchanged_dots
print("SCOPE OK" if ok else "SCOPE VIOLATION")
