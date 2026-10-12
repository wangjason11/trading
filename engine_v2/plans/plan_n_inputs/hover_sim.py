"""The PLAN_N hover follow-up's decision tool (2026-10-12): an offline replay of the H1 range hover pass on a SAVE's
events — ORIG (the pass as on S_N), A (the per-structure fix, snapshot at ev.idx), B (snapshot at the chart-order KEY
— the option the user chose and that landed); `hover_sim_a2.py` adds A2. Imports the CURRENT `export_plotly` order
helpers (`_range_events_chart_order` etc.). From the repo root:

    PYTHONPATH=. python engine_v2/plans/plan_n_inputs/hover_sim.py SAVE_DIR

Prints: ORIG vs the save's figure (sanity), then per variant the per-(sid, candle) differences vs ORIG.
"""
import ast
import glob
import os
import sys
from types import SimpleNamespace

import pandas as pd

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv
from engine_v2.charting.export_plotly import (  # noqa: E402
    _range_events_chart_order, _range_left_idx, _RANGE_TYPE_PRIORITY, _get_reversal_confirmed_by_sid)

D = sys.argv[1]
ev_csv = [p for p in glob.glob(os.path.join(D, "*rk2-5_structure_events.csv"))]
assert len(ev_csv) == 1, ev_csv
raw = pd.read_csv(ev_csv[0])
h1 = [p for p in glob.glob(os.path.join(D, "*_raw.csv")) if "_H1_" in os.path.basename(p)]
n_candles = len(pd.read_csv(h1[0]))


def _meta(s):
    if isinstance(s, str) and s.startswith("{"):
        return ast.literal_eval(s)
    return {}


events = [SimpleNamespace(idx=int(r.idx), type=r.type, meta=_meta(r.meta), category=r.category, price=r.price)
          for r in raw.itertuples()]
idx_set = set(range(n_candles))
order = _range_events_chart_order(events)
rev_end = _get_reversal_confirmed_by_sid(events)


def _apply(open_ranges, ev, sid, sd, idx):
    if ev.type == "RANGE_STARTED":
        s = _range_left_idx(ev)
        open_ranges[sid] = {"start_idx": s, "hi": float(ev.meta.get("hi", 0)), "lo": float(ev.meta.get("lo", 0)),
                            "last_update_idx": s, "struct_direction": sd}
    elif ev.type == "RANGE_UPDATED":
        if sid in open_ranges:
            open_ranges[sid]["hi"] = float(ev.meta.get("hi", open_ranges[sid]["hi"]))
            open_ranges[sid]["lo"] = float(ev.meta.get("lo", open_ranges[sid]["lo"]))
            open_ranges[sid]["last_update_idx"] = idx
    elif ev.type == "RANGE_RESET":
        open_ranges.pop(sid, None)


def orig():
    open_ranges, rbc = {}, {}
    for ev in order:
        sid, sd, idx = int(ev.meta.get("structure_id", 0)), int(ev.meta.get("struct_direction", 1)), int(ev.idx)
        _apply(open_ranges, ev, sid, sd, idx)
        if idx in idx_set:
            rbc[idx] = {s: dict(r) for s, r in open_ranges.items()}
    for idx in sorted(idx_set):
        if idx not in rbc:
            prev = [i for i in rbc if i < idx]
            rbc[idx] = rbc[max(prev)] if prev else {}
    return rbc


def per_sid(key_of):
    open_ranges, snaps = {}, {}
    for ev in order:
        sid, sd, idx = int(ev.meta.get("structure_id", 0)), int(ev.meta.get("struct_direction", 1)), int(ev.idx)
        _apply(open_ranges, ev, sid, sd, idx)
        r = open_ranges.get(sid)
        snaps.setdefault(sid, {})[key_of(ev)] = dict(r) if r is not None else None
    rbc = {}
    for sid, sn in snaps.items():
        cur = None
        for idx in sorted(set(idx_set) | set(sn)):
            if idx in sn:
                cur = sn[idx]
            if cur is not None and idx in idx_set:
                rbc.setdefault(idx, {})[sid] = cur
    return rbc


def chart_key(ev):
    # the sort key _range_events_chart_order gave this event (recomputed: STARTED at max(anchor, R) if R <= idx)
    return ev._key


# annotate each event with its chart-order key (same rule as _range_events_chart_order)
last_reset = {}
for ev in events:
    if ev.type not in _RANGE_TYPE_PRIORITY:
        continue
    sid, idx = int(ev.meta.get("structure_id", 0)), int(ev.idx)
    if ev.type == "RANGE_STARTED":
        k = _range_left_idx(ev)
        r = last_reset.get(sid)
        if r is not None and r <= idx:
            k = max(k, r)
    else:
        k = idx
        if ev.type == "RANGE_RESET":
            last_reset[sid] = idx
    ev._key = k


def points(rbc):
    """{(sid, idx): (start, hi, lo, last_update)} as the trace builder would emit them."""
    sids = {int(e.meta.get("structure_id", 0)) for e in order}
    out = {}
    for sid in sids:
        end = rev_end.get(sid, max(idx_set) + 1)
        for idx in sorted(idx_set):
            if idx > end:
                continue
            r = rbc.get(idx, {}).get(sid)
            if r is not None:
                out[(sid, idx)] = (r["start_idx"], r["hi"], r["lo"], r["last_update_idx"])
    return out


P0 = points(orig())
# sanity: ORIG vs the save's figure
data, _ = load_fig(glob.glob(os.path.join(D, "*rk2-5.html"))[0])
fig_pts = {}
for tr in data:
    n = tr.get("name") or ""
    if n.startswith("range:top:sid"):
        for c in tr["customdata"]:
            fig_pts[(int(c[5]), int(c[0]))] = (int(c[1]), float(c[2]), float(c[3]), int(c[4]))
print("ORIG sim == S_N figure:", P0 == fig_pts, len(P0), len(fig_pts))


def report(name, P):
    keys = sorted(set(P0) | set(P))
    diffs = [(k, P0.get(k), P.get(k)) for k in keys if P0.get(k) != P.get(k)]
    print(f"\n{name}: {len(diffs)} (sid, candle) differences vs ORIG; points per sid:",
          {s: sum(1 for k in P if k[0] == s) for s in sorted({k[0] for k in P})})
    for k, a, b in diffs:
        print("  ", k, a, "->", b)
    return diffs


report("A (built fix: per sid, snapshot at ev.idx)", points(per_sid(lambda e: int(e.idx))))
report("B (per sid, snapshot at the chart-order key)", points(per_sid(chart_key)))

# census: label STARTEDs (anchor < moment): is the confirm candle's state stale under A?
print("\nRANGE_STARTED census (anchor < moment):")
for ev in events:
    if ev.type == "RANGE_STARTED" and _range_left_idx(ev) < int(ev.idx):
        sid = int(ev.meta.get("structure_id", 0))
        bf = [e for e in events if e.type == "RANGE_UPDATED" and int(e.meta.get("structure_id", 0)) == sid
              and ev._key < int(e.idx) <= int(ev.idx)]
        print(f"  sid {sid} anchor {_range_left_idx(ev)} moment {ev.idx} key {ev._key} hi/lo {ev.meta.get('hi')}/"
              f"{ev.meta.get('lo')} reason {ev.meta.get('reason')}; same-sid UPDATED in (key, moment]: "
              f"{[(e.idx, e.meta.get('hi'), e.meta.get('lo')) for e in bf]}")
