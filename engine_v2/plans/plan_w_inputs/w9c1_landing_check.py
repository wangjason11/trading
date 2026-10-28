"""PLAN_W 9c-1 landing check — the replay vs the base save and the measured variant (§3.2, `w9c_variant_replay.py`, the
subs' CTS zones returned and DRAWN shown).

  - CSVs: every one md5-equal to BASE except the two lens `*_kl_zones.csv`, which are md5-equal to the VARIANT's
    (rows added only vs BASE — counted);
  - the H1 chart: the full figure equal to BASE's;
  - each M15 chart: the traces / shapes OUTSIDE the group `sub_cts_zones` = BASE's (multisets of the full JSON); the
    group's elements all hidden (`visible` False), and — with `visible` / `legendgroup` stripped — = what the VARIANT
    added over BASE (multisets); the HTML carries `"subCtsGroup": "sub_cts_zones"`;
  - the PNGs md5-equal to BASE's (the CTS zones are hidden).

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9c1_landing_check.py BASE_SAVE CUR_DEBUG CUR_CHARTS VARIANT_TREE
"""
import collections
import glob
import hashlib
import json
import os
import sys

_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE, CUR_DEBUG, CUR_CHARTS, VAR = sys.argv[1:5]
GROUP = "sub_cts_zones"
ok = True


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def j(x, strip=False):
    x = dict(x)
    if strip:
        x.pop("visible", None)
        x.pop("legendgroup", None)
    return json.dumps(x, sort_keys=True, default=str)


for p in sorted(glob.glob(os.path.join(BASE, "*.csv"))):
    n = os.path.basename(p)
    cur = os.path.join(CUR_DEBUG, n)
    if n.endswith("_M15_confluence_kl_zones.csv") or n.endswith("_M15_counter_kl_zones.csv"):
        var = os.path.join(VAR, "artifacts", "debug", n)
        a = collections.Counter(open(p, encoding="utf-8").read().splitlines())
        b = collections.Counter(open(cur, encoding="utf-8").read().splitlines())
        good = md5(cur) == md5(var) and not (a - b)
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {n}: = the variant's {md5(cur) == md5(var)}, rows {sum(a.values())} -> "
              f"{sum(b.values())} (removed {sum((a - b).values())}, added {sum((b - a).values())})")
    elif md5(p) != md5(cur):
        ok = False
        print(f"BAD {n}: differs from BASE")
print("CSVs: the other 28 md5-equal to BASE" if ok else "CSVs: see BAD lines")

for p in sorted(glob.glob(os.path.join(BASE, "*.html"))):
    n = os.path.basename(p)
    (bd, bl), (cd, cl) = load_fig(p), load_fig(os.path.join(CUR_CHARTS, n))
    bs, cs = bl.get("shapes", []), cl.get("shapes", [])
    if "_M15_" not in n:
        good = [j(t) for t in bd] == [j(t) for t in cd] and j(bl) == j(cl)
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {n}: the H1 figure equal to BASE's: {good}")
        continue
    (vd, vl) = load_fig(os.path.join(VAR, "artifacts", "charts", n))
    vs = vl.get("shapes", [])
    out_t = collections.Counter(j(t) for t in cd if t.get("legendgroup") != GROUP)
    out_s = collections.Counter(j(s) for s in cs if s.get("legendgroup") != GROUP)
    grp_t = [t for t in cd if t.get("legendgroup") == GROUP]
    grp_s = [s for s in cs if s.get("legendgroup") == GROUP]
    added_t = collections.Counter(j(t) for t in vd) - collections.Counter(j(t) for t in bd)
    added_s = collections.Counter(j(s) for s in vs) - collections.Counter(j(s) for s in bs)
    checks = {
        "rest = BASE (traces)": out_t == collections.Counter(j(t) for t in bd),
        "rest = BASE (shapes)": out_s == collections.Counter(j(s) for s in bs),
        "group all hidden": all(t.get("visible") is False for t in grp_t) and all(s.get("visible") is False for s in grp_s),
        "group = the variant's additions (traces)": collections.Counter(j(t, True) for t in grp_t) == added_t,
        "group = the variant's additions (shapes)": collections.Counter(j(s, True) for s in grp_s) == added_s,
        "the HTML names the group": f'"subCtsGroup": "{GROUP}"' in open(os.path.join(CUR_CHARTS, n), encoding="utf-8").read(),
    }
    good = all(checks.values())
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {n}: traces {len(bd)} -> {len(cd)} (group {len(grp_t)}), shapes {len(bs)} -> "
          f"{len(cs)} (group {len(grp_s)}); " + ", ".join(f"{k}: {v}" for k, v in checks.items()))
for p in sorted(glob.glob(os.path.join(BASE, "*.png"))):
    n = os.path.basename(p)
    good = md5(p) == md5(os.path.join(CUR_CHARTS, n))
    ok &= good
    print(f"{'OK ' if good else 'BAD'} PNG {n} equal: {good}")
print("ALL AS PREDICTED" if ok else "NOT AS PREDICTED")
sys.exit(0 if ok else 1)
