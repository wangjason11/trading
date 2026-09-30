"""Search for the Plan B §2 "rebuilt-prefix" exception (2026-09-29): the full run and the
`stop_after_cts_established=2` run disagree on the first two CTS_ESTABLISHED moments, the stopped run having
early-stopped. Uses `random_tail_search.trials` (same streams / modes).

  python rebuilt_prefix_search.py OUT.jsonl SEED N [LO HI]

At 94fa63c (pre "a new cycle ends an open watch") it finds the `_make_double_rewind_data` shape (11 in 1500 default
trials); after it: 0 in 36k (24k default + 12k wide).
"""
import io, json, os, sys, importlib.util
from contextlib import redirect_stdout
spec = importlib.util.spec_from_file_location("rts", os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_tail_search.py"))
rts = importlib.util.module_from_spec(spec); spec.loader.exec_module(rts)
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests.test_unified_probe import _prepare_df
out, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lo, hi = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else (None, None)
def est(ms):
    return [int(e.meta["confirmed_at"]) for e in ms.events if e.type == "CTS_ESTABLISHED"]
found = 0
with open(out, "w") as f:
    for t, name, sd, rows, end_idx, _stop in rts.trials(seed, lo, hi):
        if t >= n: break
        try:
            df = _prepare_df(rows)
            kw = dict(struct_direction=sd, start_idx=0, structure_id=0, timeframe="H1", pip_size=_pip_size_from_pair(df), end_idx=end_idx)
            with redirect_stdout(io.StringIO()):
                full = _make_market_structure(df, **kw); full.run()
                st = _make_market_structure(df, stop_after_cts_established=2, **kw); st.run()
        except Exception as e:  # noqa
            continue
        if st.early_stop_idx is not None and est(full)[:2] != est(st)[:2]:
            found += 1
            f.write(json.dumps(dict(t=t, base=name, sd=sd, end=end_idx, full=est(full), stopped=est(st), stop_idx=st.early_stop_idx)) + "\n")
            f.flush()
print("done", seed, found)
