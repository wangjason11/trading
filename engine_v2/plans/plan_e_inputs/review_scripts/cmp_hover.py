"""Hover-text diff of the 3 charts against a baseline folder (2026-09-30, the wave-candle
hover-label fix). `cmp_save.py`'s figure diff compares per trace (name, x, y) + the layout
shapes — NOT `hovertemplate` / `customdata` — so a label-only change is invisible to it.

    python engine_v2/plans/plan_e_inputs/review_scripts/cmp_hover.py BASE_DIR CUR_CHARTS \
        [--match TEXT] [--max N]

Per chart (matched by file name): the trace count on each side, how many traces'
`hovertemplate` changed, how many `customdata` changed, and the first N changed hovers
(before -> after, one `<br>` line per row). `--match TEXT` limits the report to traces whose
baseline OR current hovertemplate contains TEXT (e.g. "Wave Candle"), and also prints how
many changed traces fell OUTSIDE the match (the "nothing else moved" check). Traces are
paired by position, so a trace-count change is reported and the pairing is not trusted.
Exit 1 when a chart is missing or the trace counts differ.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
_argv, sys.argv = sys.argv, sys.argv[:1]  # chart_census runs a census on import
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv


def _lines(ht):
    return (ht or "").replace("<extra></extra>", "").split("<br>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cur_charts")
    ap.add_argument("--match", default=None)
    ap.add_argument("--max", type=int, default=5)
    a = ap.parse_args()
    rc = 0
    for name in sorted(f for f in os.listdir(a.base) if f.endswith(".html")):
        cur = os.path.join(a.cur_charts, name)
        if not os.path.exists(cur):
            print(f"MISSING {name}")
            rc = 1
            continue
        tb, _ = load_fig(os.path.join(a.base, name))
        tc, _ = load_fig(cur)
        print(f"== {name[-40:]}: traces {len(tb)}/{len(tc)}")
        if len(tb) != len(tc):
            print("   trace counts differ — positional pairing not trusted")
            rc = 1
            continue
        changed, outside, cd_changed = [], 0, 0
        for x, y in zip(tb, tc):
            hx, hy = x.get("hovertemplate"), y.get("hovertemplate")
            if x.get("customdata") != y.get("customdata"):
                cd_changed += 1
            if hx == hy:
                continue
            if a.match and a.match not in (hx or "") and a.match not in (hy or ""):
                outside += 1
                continue
            changed.append((hx, hy))
        tag = f" matching {a.match!r}" if a.match else ""
        print(f"   hovertemplate changed{tag}: {len(changed)}"
              + (f"; changed OUTSIDE the match: {outside}" if a.match else "")
              + f"; customdata changed: {cd_changed}")
        for hx, hy in changed[: a.max]:
            lx, ly = _lines(hx), _lines(hy)
            print("   ---")
            for i in range(max(len(lx), len(ly))):
                bx = lx[i] if i < len(lx) else ""
                by = ly[i] if i < len(ly) else ""
                print(f"   {'  ' if bx == by else '!='} {bx!s:<45} | {by}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
