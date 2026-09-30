"""Independent check of the unreachability argument behind 221b420's guard (written by the F3b landing review,
2026-09-29c; persisted 2026-09-29c). The guard in `_replay_step_no_patterns` asserts only "no watch open at its
expiry candle"; this checks the STRONGER invariants that make it unreachable.

At every `_replay_step_no_patterns(i)` entry and exit (non-terminal state):
  W1  pending => watch open
  W2  watch open => pending exists, pending_apply <= expires_idx
  W3  entry with a watch open: i <= pending_apply (no candle past p is stepped while the watch is open)
Also counts guard-adjacent quantities: max (i - watch_start) stepped while open, re-steps under an open watch.
Usage: python watch_invariant_fuzz.py SEED N [LO HI]   (RTS_BASES env honoured, like random_tail_search)
Runs through whichever `engine_v2` tree is first on sys.path (from the repo root: PYTHONPATH=.); exit code 1 on any
violation or run error. Landing review of 221b420: 16k trials (seed 77 x6k default, 78 x6k WIDE 10 40,
79 x4k RTS_BASES=iw7,iw8,dr6), ~24.7k watches, 0 violations, 0 errors.
"""
import os, sys, io
from collections import Counter
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_tail_search as rts  # noqa: E402  (applies its own taps; harmless)
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair  # noqa: E402
from engine_v2.tests.test_unified_probe import _prepare_df  # noqa: E402

MS, REV = rts.MS, rts.REV
viol = Counter(); ex = {}
stats = Counter()
cur = {}

def chk(self, i, where):
    st = self.state
    if st.state == REV:
        return
    pend = st.pending_reversal_apply_idx
    if pend is not None and not st.reversal_watch_active:
        viol["W1 pending without watch " + where] += 1; ex.setdefault("W1", dict(cur, i=i))
    if st.reversal_watch_active:
        if pend is None:
            viol["W2 watch without pending " + where] += 1; ex.setdefault("W2", dict(cur, i=i))
        elif int(pend) > int(st.reversal_watch_expires_idx):
            viol["W2 pending past expiry " + where] += 1; ex.setdefault("W2b", dict(cur, i=i))
        if where == "entry" and pend is not None and int(i) > int(pend):
            viol["W3 stepped past p with watch open"] += 1; ex.setdefault("W3", dict(cur, i=i))

_o = MS._replay_step_no_patterns
def rs(self, i, *, freeze_range=False):
    chk(self, i, "entry")
    st = self.state
    if st.reversal_watch_active and st.state != REV:
        stats["steps_under_watch"] += 1
        seen = cur.setdefault("seen", set())
        if (id(self), int(i)) in seen:
            stats["resteps_under_watch"] += 1
        seen.add((id(self), int(i)))
    out = _o(self, i, freeze_range=freeze_range)
    chk(self, i, "exit")
    return out
MS._replay_step_no_patterns = rs

seed, n = int(sys.argv[1]), int(sys.argv[2])
lo, hi = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else (None, None)
errs = Counter(); nrev = 0; ntrial = 0
for t, name, sd, rows, end_idx, stop_n in rts.trials(seed, lo, hi):
    if t >= n:
        break
    ntrial += 1
    cur.clear(); cur.update(t=t, base=name, sd=sd, end=end_idx)
    try:
        df = _prepare_df(rows)
        ms = _make_market_structure(df, struct_direction=sd, start_idx=0, structure_id=0, timeframe="H1",
                                    pip_size=_pip_size_from_pair(df), end_idx=end_idx,
                                    stop_after_cts_established=stop_n)
        with redirect_stdout(io.StringIO()):
            ms.run()
        nrev += any(e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal" for e in ms.events)
        stats["watches"] += sum(1 for e in ms.events if e.type == "REVERSAL_WATCH_START")
        stats["candidates"] += sum(1 for e in ms.events if e.type == "REVERSAL_CANDIDATE")
    except Exception as e:  # noqa: BLE001
        errs[f"{type(e).__name__}: {str(e)[:120]}"] += 1
print(dict(trials=ntrial, reversal_trials=nrev, errors=dict(errs), violations=dict(viol), stats=dict(stats)))
for k, v in ex.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "seen"})
sys.exit(1 if (viol or errs) else 0)
