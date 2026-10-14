"""PLAN_P — a KEYED hover diff of the three replay charts: the multiset of every distinct hover STRING (customdata /
text / hovertext / hovertemplate cells, `<br>`-split into lines) per chart, BASE vs CUR — works when the trace counts
differ (`cmp_hover.py` is positional and refuses then). Prints the lines only in BASE (-) and only in CUR (+), with
their multiplicity change.

    PYTHONPATH=. python engine_v2/plans/qa_ic_crossfib_inputs/cmp_hover_strings.py BASE_DIR CUR_DIR [--match TEXT]
"""
import glob
import os
import re
import sys
from collections import Counter

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]  # chart_census runs a census on import when given args
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE, CUR = sys.argv[1], sys.argv[2]
MATCH = sys.argv[sys.argv.index("--match") + 1] if "--match" in sys.argv else None
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _cells(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _cells(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _cells(x)
    elif isinstance(v, str):
        yield v


def hover_lines(path):
    data, _layout = load_fig(path)
    out = Counter()
    for t in data:
        name = str(t.get("name", ""))
        for key in ("customdata", "text", "hovertext", "hovertemplate"):
            for s in _cells(t.get(key)):
                if _TIME.match(s) and "<br>" not in s:
                    continue  # a bare timestamp cell (x values carried in customdata), not a hover line
                for line in s.split("<br>"):
                    if line and (MATCH is None or MATCH in line):
                        out[(name, line)] += 1
    return out


STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_"   # the reference window's run (artifacts/charts holds others)
for suffix in ("rk2-5.html", "rk2-5_M15_counter.html", "rk2-5_M15_confluence.html"):
    a_path = glob.glob(os.path.join(BASE, STEM + suffix))
    b_path = glob.glob(os.path.join(CUR, STEM + suffix))
    assert len(a_path) == 1 and len(b_path) == 1, (suffix, a_path, b_path)
    a, b = hover_lines(a_path[0]), hover_lines(b_path[0])
    keys = sorted(set(a) | set(b), key=lambda k: (k[0], k[1]))
    diff = [(k, a[k], b[k]) for k in keys if a[k] != b[k]]
    print(f"== {suffix}: {len(diff)} hover lines differ (distinct lines {len(a)} -> {len(b)})")
    for (name, line), na, nb in diff:
        print(f"   {'-' if nb < na else '+'} [{name}] x{na}->{nb}: {line[:220]}")
