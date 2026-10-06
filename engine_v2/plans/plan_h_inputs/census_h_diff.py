"""PLAN_H §8.4 — diff two `census_h_plugin.py` outputs (A vs V3): per test, the outcome and what moved.

    python engine_v2/plans/plan_h_inputs/census_h_diff.py census_A.json census_V3.json [--detail TEST_SUBSTR]

Classes: FAILS (passes on A, fails on V3 — B must re-pin / re-derive / retire it), MOVED-SILENT (passes on both but its
POIs / direct activation histories / resolver answers moved — a pin that does not see the change), UNMOVED-POI (the
test builds POIs or calls the resolver, nothing moved), NO-POI (never touches the POI layer).
"""
import json
import sys

A = json.load(open(sys.argv[1]))
V = json.load(open(sys.argv[2]))
detail = sys.argv[sys.argv.index("--detail") + 1] if "--detail" in sys.argv else None


def ok(b):
    return all(v in ("passed", "skipped") for v in b.get("outcome", {}).values())


rows = {"FAILS": [], "MOVED-SILENT": [], "UNMOVED-POI": [], "NO-POI": []}
for t in sorted(set(A) | set(V)):
    if t == "<collect>":
        continue
    a, v = A.get(t, {}), V.get(t, {})
    moved = [k for k in ("poi", "act", "res") if a.get(k) != v.get(k)]
    touches = any(a.get(k) for k in ("poi", "act", "res")) or any(v.get(k) for k in ("poi", "act", "res"))
    if ok(a) and not ok(v):
        cls = "FAILS"
    elif moved:
        cls = "MOVED-SILENT"
    elif touches:
        cls = "UNMOVED-POI"
    else:
        cls = "NO-POI"
    n = lambda b: (sum(len(x) for x in b.get("poi", [])), len(b.get("act", [])), len(b.get("res", [])))
    rows[cls].append((t, moved, n(a), n(v)))
    if detail and detail in t:
        print(f"--- {t}  A={a.get('outcome')} V3={v.get('outcome')}")
        for k in ("poi", "act", "res", "sel"):
            print(f"  {k} A : {json.dumps(a.get(k))[:1500]}")
            print(f"  {k} V3: {json.dumps(v.get(k))[:1500]}")

for cls, rs in rows.items():
    print(f"\n== {cls}: {len(rs)}")
    if cls in ("FAILS", "MOVED-SILENT"):
        for t, moved, na, nv in rs:
            print(f"  {t}  moved={moved}  (POIs, direct act, resolver) A={na} V3={nv}")
