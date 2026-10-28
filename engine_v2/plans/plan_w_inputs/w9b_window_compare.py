"""PLAN_W 9b — compare two scratch-tree replays (`w9b_window_replay.py`) of the same window: every CSV by md5, and
for the CSVs that differ the rows added / removed (whole-line multiset diff, so a positional shift is not a delta);
each chart's traces by name (count + the names whose full JSON differs) and the shapes. The fetch lines of both logs.

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9b_window_compare.py BASE_TREE VAR_TREE [OUT.txt]
"""
import collections
import glob
import hashlib
import json
import os
import sys

_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from engine_v2.debug.chart_census import load_fig  # noqa: E402  (the census module runs on import without args)
sys.argv = _argv

base, var = sys.argv[1], sys.argv[2]
out = []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


bc = {os.path.basename(p): p for p in glob.glob(os.path.join(base, "artifacts", "debug", "*.csv"))}
vc = {os.path.basename(p): p for p in glob.glob(os.path.join(var, "artifacts", "debug", "*.csv"))}
out.append(f"CSVs: base {len(bc)}, variant {len(vc)}; only base {sorted(set(bc) - set(vc))}; only variant "
           f"{sorted(set(vc) - set(bc))}")
diff = [n for n in sorted(set(bc) & set(vc)) if md5(bc[n]) != md5(vc[n])]
out.append(f"md5-different: {len(diff)} of {len(set(bc) & set(vc))}")
for n in diff:
    a = collections.Counter(open(bc[n], encoding="utf-8").read().splitlines())
    b = collections.Counter(open(vc[n], encoding="utf-8").read().splitlines())
    out.append(f"  {n}: rows {sum(a.values())} -> {sum(b.values())}, removed {sum((a - b).values())}, "
               f"added {sum((b - a).values())}")
    for line in list((a - b).elements())[:6]:
        out.append(f"     - {line[:260]}")
    for line in list((b - a).elements())[:6]:
        out.append(f"     + {line[:260]}")
for p in sorted(glob.glob(os.path.join(base, "artifacts", "charts", "*.html"))):
    q = os.path.join(var, "artifacts", "charts", os.path.basename(p))
    if not os.path.exists(q):
        out.append(f"FIG {os.path.basename(p)}: MISSING in variant")
        continue
    (da, la), (db, lb) = load_fig(p), load_fig(q)
    na = collections.Counter(t.get("name") for t in da)
    nb = collections.Counter(t.get("name") for t in db)
    ja = collections.Counter(json.dumps(t, sort_keys=True, default=str) for t in da)
    jb = collections.Counter(json.dumps(t, sort_keys=True, default=str) for t in db)
    changed = sorted({json.loads(s).get("name") or "?" for s in (ja - jb) + (jb - ja)}, key=str)
    sa, sb = la.get("shapes", []), lb.get("shapes", [])
    out.append(f"FIG {os.path.basename(p)}: traces {len(da)} -> {len(db)}, shapes {len(sa)} -> {len(sb)}, "
               f"traces differing (multiset) {sum(((ja - jb) + (jb - ja)).values())}, names {changed}; "
               f"count deltas {dict((nb - na) + collections.Counter({k: -v for k, v in (na - nb).items()}))}")
for t in (base, var):
    log = open(os.path.join(t, "run.log"), encoding="utf-8", errors="replace").read().splitlines()
    keep = [l for l in log if l.startswith("[data_bridge] Fetched") or l.startswith("[floor_seed]")
            or "DEBUG traces" in l or "DEBUG shapes" in l or l.startswith("[m15_chart] traces")]
    out.append(f"{t}: " + " | ".join(keep))
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 3:
    open(sys.argv[3], "w", encoding="utf-8").write(text)
