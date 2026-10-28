"""PLAN_X X2 — one table over the per-window census outputs (`x2_census_<START>_<END>.txt`, `x2_ms_census.py`).

    python engine_v2/plans/plan_x_inputs/x2_census_summary.py  > engine_v2/plans/plan_x_inputs/x2_census_summary.txt
"""
import ast
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def section(text, title):
    i = text.index(title)
    j = text.find("\n== ", i + len(title))
    return text[i:j if j != -1 else len(text)]


def counts(sec):
    m = re.search(r"^  (\{.*\})$", sec, re.M)
    return ast.literal_eval(m.group(1)) if m else {}


rows = []
for path in sorted(glob.glob(os.path.join(HERE, "x2_census_20*.txt"))):
    t = open(path, encoding="utf-8").read()
    win = re.search(r"window (\S+) -> (\S+)", t).group(0)
    h1 = re.search(r"H1 frame (\d+) candles", t).group(1)
    m15 = re.search(r"Fetched (\d+) M15", t)
    c11 = counts(section(t, "== 1.1"))
    c13 = counts(section(t, "== 1.3"))
    s21 = section(t, "== 2.1")
    c21 = counts(s21)
    c12 = counts(section(t, "== latent: 1.2"))
    lat = section(t, "== latent: 1.6")
    exp = [l for l in s21.splitlines() if " -> " in l and "k=" in l and "downstream" in l]   # one per decision
    verdicts = {}
    for l in exp:
        v = l.split(" -> ", 1)[1].split(" (")[0]
        verdicts[v] = verdicts.get(v, 0) + 1
    st = re.search(r"streams (\d+), reversals (\d+), RANGE_STARTED (\d+)", lat)
    cand = re.search(r"REVERSAL_CANDIDATEs examined \(both lenses\): (\d+)", lat).group(1)
    rows.append(dict(
        window=win.replace("window ", ""), h1=h1, m15=m15.group(1) if m15 else "-",
        p11=f"{c11.get('resets_examined', 0)} resets / {c11.get('ref_adhoc_examined', 0)} ad-hoc zones; "
            f"past bound {c11.get('resets_anchor_plus5_past_bound', 0)}; DIFFER {c11.get('resets_DIFFER', 0) + c11.get('ref_adhoc_DIFFERS', 0)}",
        p13=f"{c13.get('candles_examined', 0)} candles ({c13.get('candles_with_snapshot_poi', 0)} with a POI); inner differs "
            f"{c13.get('inner_differs', 0)}; DECISION differs {c13.get('DECISION_differs', 0)}",
        p21=f"{sum(v for k, v in c21.items() if k.startswith('false_answers'))} False answers; "
            f"{sum(v for k, v in c21.items() if k.startswith('EXPOSED'))} turn True 1 candle later "
            f"(downstream decisions: {verdicts})",
        p12=f"{c12.get('bounded_runs', 0)} bounded runs, {c12.get('instances_formed_by_bound', 0)} gaps; bounds differ "
            f"{c12.get('bounds_differ', 0)}; DEGENERACY differs {c12.get('DEGENERACY_differs', 0)}",
        lat=f"streams {st.group(1)}, reversals {st.group(2)}, RANGE_STARTED {st.group(3)}, candidates {cand}; "
            f"straddle {lat.count('1.6 straddle')}; B5 exact {lat.count('B5 BOS confirmed on the reversal candle')}; "
            f"B5-info {lat.count('(info) BOS confirmed after')}; duplicate range {lat.count('duplicate range start')}; "
            f"range reset before its confirmation {sum(len(ast.literal_eval(x)) for x in re.findall(r'before its confirmation: (\[.*\])', lat))}",
    ))

for r in rows:
    print(f"== {r['window']}  (H1 {r['h1']} candles, M15 {r['m15']})")
    print(f"  1.1 probe BOS_0 vs bound : {r['p11']}")
    print(f"  1.3 POI snapshot vs as-of: {r['p13']}")
    print(f"  2.1 gap one candle late  : {r['p21']}")
    print(f"  1.2 bounded-run gaps     : {r['p12']}")
    print(f"  1.6 / 1.7 / 1.8          : {r['lat']}")
