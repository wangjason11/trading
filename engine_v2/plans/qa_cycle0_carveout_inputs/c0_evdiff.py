"""Per-sub structure-event diff BASE vs VARIANT (argv[1], argv[2] = debug / save dirs), per M15 lens: the events
(idx, type, cycle, price, via / reason) only in one side, keyed by sub_id."""
import ast
import sys
from pathlib import Path

import pandas as pd

B, V = Path(sys.argv[1]), Path(sys.argv[2])


def load(d, lens):
    f = sorted(d.glob(f"*_M15_{lens}_structure_events.csv"))[0]
    ev = pd.read_csv(f)
    rows = set()
    for r in ev.itertuples():
        m = ast.literal_eval(r.meta)
        tag = m.get("via") or m.get("reason") or m.get("source") or ""
        rows.add((m.get("sub_id"), int(r.idx), r.type, m.get("cycle_id"),
                  None if pd.isna(r.price) else round(float(r.price), 5), tag))
    return rows


for lens in ("counter", "confluence"):
    b, v = load(B, lens), load(V, lens)
    print(f"=== {lens}: base {len(b)} events, variant {len(v)}; only-base {len(b - v)}, only-variant {len(v - b)}")
    subs = sorted({x[0] for x in (b ^ v)})
    for s in subs:
        ob = sorted(x for x in b - v if x[0] == s)
        ov = sorted(x for x in v - b if x[0] == s)
        print(f"  sub {s}: -{len(ob)} +{len(ov)}")
        for x in ob[:14]:
            print(f"     - {x[1:]}")
        for x in ov[:14]:
            print(f"     + {x[1:]}")
