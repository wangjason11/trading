import sys, copy, io, contextlib
sys.path.insert(0, r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2")
from engine_v2.tests.test_unified_probe import _make_second_cts_moment_after_extreme_data, _make_multicycle_data, _prepare_df
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline

import os
MODE=os.environ.get("MODE","h1")
for maker in (_make_second_cts_moment_after_extreme_data, _make_multicycle_data):
    df = _prepare_df(maker())
    res = compute_bounded_structure(df, 0, +1)
    print("==", maker.__name__, "n=", len(df))
    for e in res.events:
        if e.type in ("CTS_ESTABLISHED","BOS_CONFIRMED","CTS_CONFIRMED","CTS_UPDATED","REVERSAL_CANDIDATE"):
            print(" ", e.type, e.idx, {k:v for k,v in e.meta.items() if k in ("structure_id","cycle_id","confirmed_at","anchor_idx","cts_anchor_idx","via")})
    ev2 = copy.deepcopy(res.events)
    for e in ev2:
        if e.type in ("CTS_ESTABLISHED","BOS_CONFIRMED"):
            e.idx = int(e.meta["confirmed_at"])
    outs=[]
    for evs in (res.events, ev2):
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                o=_run_downstream_pipeline(res.df, evs, +1, fib_mode=MODE, skip_wvmi=False)
            except Exception as ex:
                o={"ERR":repr(ex)}
        outs.append(o)
    a,b=outs
    print(" keys", sorted(a.keys()))
    for k in a:
        ra, rb = repr(a.get(k)), repr(b.get(k))
        if ra!=rb: print("  DIFF", k, len(ra), len(rb))
