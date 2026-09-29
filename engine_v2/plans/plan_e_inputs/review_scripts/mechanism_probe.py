"""Per-mechanism MS taps on given rows (2026-09-29; the fixture re-derivation for "a new cycle ends an open watch").

  python mechanism_probe.py IN.json OUT.json   # IN = [{"name", "sd", "end", "rows": [{o,h,l,c}, ...]}, ...]

Runs each item through the `engine_v2` tree first on PYTHONPATH and records, per run: `f3` (a reversal winner chosen
while a watch is open, applying after its `expires_idx`), `overwrite` (two expiries fired inside one `_step_anchor`
call), `post_expiry` (candles a step kept stepping after an expiry inside it; flag = inside a `_rewind_to` rebuild),
`rir` / `rewinds` (run-loop rewinds, and those entered in REVERSAL), `err` (an exception — e.g. the rebuild's
"reached a reversal" assert), plus the reversal candles, `CTS_ESTABLISHED` moments and `ended` (BOS_CONFIRMED
`ended_watch_pattern_anchor_idx`). Method: run the same IN in scratch trees with a fix reverted / partially applied
and in the repo — a pin must show its mechanism where the fix is missing and be clean here.
"""
import io, json, sys
from contextlib import redirect_stdout
import engine_v2.structure.market_structure as m
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests.test_unified_probe import _prepare_df
MS = m.MarketStructure; REV = m.MarketState.REVERSAL
cur = {}; steps = []
o_best, o_step, o_exp, o_rw, o_rs = MS._best_bopb_pattern_at_anchor, MS._step_anchor, MS._maybe_expire_reversal_watch, MS._rewind_to, MS._replay_step_no_patterns
def best(self, *, i, breakout_th, pullback_th, D):
    w = o_best(self, i=i, breakout_th=breakout_th, pullback_th=pullback_th, D=D); st = self.state
    if w is not None and w[2] == "reversal" and st.reversal_watch_active and int(w[1]) > int(st.reversal_watch_expires_idx):
        cur.setdefault("f3", []).append((int(i), int(w[1]), int(st.reversal_watch_expires_idx), bool(getattr(self, "_in_rewind", False))))
    return w
def step(self, i):
    steps.append({"exp": 0, "after": 0, "rw": bool(getattr(self, "_in_rewind", False)), "i": int(i)})
    try: return o_step(self, i)
    finally:
        s = steps.pop()
        if s["exp"] >= 2: cur.setdefault("overwrite", []).append((s["i"], s["rw"]))
        if s["exp"] and s["after"]: cur.setdefault("post_expiry", []).append((s["i"], s["after"], s["rw"]))
def exp(self, i):
    st = self.state
    fires = st.reversal_watch_active and st.reversal_watch_expires_idx is not None and int(i) >= int(st.reversal_watch_expires_idx)
    out = o_exp(self, i)
    if fires and steps: steps[-1]["exp"] += 1
    return out
def rs(self, i, *, freeze_range=False):
    if steps and steps[-1]["exp"]: steps[-1]["after"] += 1
    return o_rs(self, i, freeze_range=freeze_range)
def rw(self, jump_to, *, seed=None):
    if self.state.state == REV: cur.setdefault("rir", []).append(int(jump_to))
    cur.setdefault("rewinds", []).append(int(jump_to))
    return o_rw(self, jump_to, seed=seed)
MS._best_bopb_pattern_at_anchor, MS._step_anchor, MS._maybe_expire_reversal_watch, MS._rewind_to, MS._replay_step_no_patterns = best, step, exp, rw, rs
res = []
for it in json.load(open(sys.argv[1])):
    cur.clear(); rec = dict(name=it["name"])
    try:
        df = _prepare_df(it["rows"])
        ms = _make_market_structure(df, struct_direction=it["sd"], start_idx=0, structure_id=0, timeframe="H1",
                                    pip_size=_pip_size_from_pair(df), end_idx=it.get("end"))
        with redirect_stdout(io.StringIO()):
            ms.run()
        rec["rev"] = [int(e.idx) for e in ms.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
        rec["est"] = [int(e.idx) for e in ms.events if e.type == "CTS_ESTABLISHED"]
        rec["ended"] = [e.meta.get("ended_watch_pattern_anchor_idx") for e in ms.events if e.meta.get("ended_watch_pattern_anchor_idx") is not None]
    except Exception as e:  # noqa
        rec["err"] = f"{type(e).__name__}: {str(e)[:100]}"
    rec.update(dict(cur)); res.append(rec)
json.dump(res, open(sys.argv[2], "w"))
for r in res: print(r)
