"""Render every BOS / CTS / PB dot hover (main + sub) from a chart dir, the template filled per point; check the
invariants of the 2026-10-22 spec; print selected examples.

    PYTHONPATH=. python dot_hovers_new.py CHART_DIR
"""
import glob
import os
import re
import sys
from collections import Counter

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

D = sys.argv[1]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_"
DOT = re.compile(r"^(CTS \(confirmed\)|BOS \(confirmed\)|CTS \(unconfirmed\)|PB \(furthest\)|H1 (CTS|BOS|PB)$|"
                 r"M15 (CTS|BOS|CTS \(unconf\)|PB) sub(\d+))")
_FIELD = re.compile(r"%\{customdata\[(\d+)\](?::([^}]*))?\}")
WANT = {  # (chart, trace-name prefix, idx) -> show
    ("confluence", "M15 CTS sub0", 1245), ("confluence", "M15 BOS sub5", 3304), ("counter", "M15 CTS sub3", 3047),
    ("counter", "M15 CTS (unconf) sub8", 4222), ("h1", "BOS (confirmed) (prior)", 96),
    ("h1", "CTS (confirmed)", 905), ("h1", "PB (furthest) (prior)", 683), ("counter", "H1 CTS", 905),
}


def fill(tmpl, cd):
    def rep(m):
        v = cd[int(m.group(1))]
        return format(v, m.group(2)) if m.group(2) else str(v)
    return _FIELD.sub(rep, tmpl).replace("<extra></extra>", "")


for chart, suffix in (("h1", "rk2-5.html"), ("counter", "rk2-5_M15_counter.html"),
                      ("confluence", "rk2-5_M15_confluence.html")):
    data, _ = load_fig(glob.glob(os.path.join(D, STEM + suffix))[0])
    n, lines_seen, mismatch = 0, Counter(), []
    for t in data:
        name = str(t.get("name", ""))
        if not DOT.match(name):
            continue
        for cd in t.get("customdata") or []:
            n += 1
            text = fill(t["hovertemplate"], cd)
            ls = text.split("<br>")
            for ln in ls:
                key = ln.split("=")[0] if "=" in ln else ln
                lines_seen[key.split(" | ")[0]] += 1
            if name.startswith("M15"):
                phase = next(ln for ln in ls if ln.startswith("phase="))
                rd = next(ln for ln in ls if ln.startswith("relative_dir="))
                if ("forming" in phase) != (rd == "relative_dir=NA (before start_idx)"):
                    mismatch.append((name, ls[1], phase, rd))
                assert rd != "relative_dir=-", (name, ls[1])
            if (chart, name, int(cd[0])) in WANT:
                print(f"--- [{chart}] [{name}] ---")
                print("\n".join(ls))
                print()
    print(f"== {chart}: {n} dots; phase/relative_dir mismatches: {mismatch}")
