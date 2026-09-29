"""Random-tail search over MarketStructure (the F3 measurement, 2026-09-29).

Verified fixture bases + random candle tails, through whichever `engine_v2` tree is first on PYTHONPATH — run the
SAME seed in two trees (the repo and a scratch `git archive` copy with a variant) and diff the outputs:

  python random_tail_search.py run OUT.jsonl SEED N    # N trials -> one JSON line each
  python random_tail_search.py rows SEED T             # regenerate trial T of SEED (rows JSON + bound + sd)
  python random_tail_search.py compare A.jsonl B.jsonl # errors, counters, signature diffs (reversal candle same?)

Per trial: the base, sd (-1 = the base + tail mirrored at 1.2), an optional bound, the output signature (events + the
MS output columns), the reversal candles, the exception if any (AssertionError included), and the counters `rir`
(run-loop rewinds entered in REVERSAL), `wpe` (reversal winners applying after the open watch's `expires_idx`),
`exp` (expiries fired). All RNG draws happen before the run — never inside the try — so every tree sees the same
trials (a draw inside a try is skipped on an exception and desynchronises the stream). Smoke-run a few trials in the
foreground first: an empty background log is not "no finds" (buffered output; an import error dies silently).
"""
import hashlib, json, sys, traceback
from collections import Counter

import numpy as np

import engine_v2.structure.market_structure as msmod
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data, _make_watch_over_second_cts_data
from engine_v2.tests.test_unified_probe import (_prepare_df, _make_multicycle_data,
                                                _make_second_cts_moment_after_anchor_data)

MS = msmod.MarketStructure
REV = msmod.MarketState.REVERSAL

_o_rw = MS._rewind_to
def _rw(self, jump_to, *, seed=None):
    if self.state.state == REV:
        self._rts_rir = getattr(self, "_rts_rir", 0) + 1
    return _o_rw(self, jump_to, seed=seed)
MS._rewind_to = _rw

_o_best = MS._best_bopb_pattern_at_anchor
def _best(self, *, i, breakout_th, pullback_th, D):
    w = _o_best(self, i=i, breakout_th=breakout_th, pullback_th=pullback_th, D=D)
    st = self.state
    if (w is not None and w[2] == "reversal" and st.reversal_watch_active
            and st.reversal_watch_expires_idx is not None and int(w[1]) > int(st.reversal_watch_expires_idx)):
        self._rts_wpe = getattr(self, "_rts_wpe", 0) + 1
    return w
MS._best_bopb_pattern_at_anchor = _best

_o_exp = MS._maybe_expire_reversal_watch
def _exp(self, i):
    st = self.state
    if st.reversal_watch_active and st.reversal_watch_expires_idx is not None and int(i) >= int(st.reversal_watch_expires_idx):
        self._rts_exp = getattr(self, "_rts_exp", 0) + 1
    return _o_exp(self, i)
MS._maybe_expire_reversal_watch = _exp

BASES = {
    "dr9": _make_double_rewind_data()[:9], "dr10": _make_double_rewind_data()[:10],
    "dr14": _make_double_rewind_data()[:14], "dr15": _make_double_rewind_data()[:15],
    "wo11": _make_watch_over_second_cts_data()[:11], "wo12": _make_watch_over_second_cts_data()[:12],
    "sc11": _make_second_cts_moment_after_anchor_data()[:11], "mc9": _make_multicycle_data()[:9],
}
NAMES = sorted(BASES)
PIP = 0.0001


def _cand(rng, c):
    kind = rng.randint(0, 4)           # 0 maru, 1 normal, 2 pinbar, 3 small
    bear = rng.rand() < 0.6
    body = [rng.uniform(40, 110), rng.uniform(15, 45), rng.uniform(3, 12), rng.uniform(2, 10)][kind] * PIP
    wick_u = [rng.uniform(0, 3), rng.uniform(3, 20), rng.uniform(2, 8), rng.uniform(1, 8)][kind] * PIP
    wick_d = [rng.uniform(0, 3), rng.uniform(3, 20), rng.uniform(25, 60), rng.uniform(1, 8)][kind] * PIP
    if kind == 2 and bear:
        wick_u, wick_d = wick_d, wick_u
    o = c + rng.uniform(-3, 3) * PIP
    cl = o - body if bear else o + body
    return {"o": round(o, 5), "h": round(max(o, cl) + wick_u, 5), "l": round(min(o, cl) - wick_d, 5), "c": round(cl, 5)}


def _mirror(rows, pivot=1.2):
    return [{"o": round(pivot - r["o"], 5), "h": round(pivot - r["l"], 5), "l": round(pivot - r["h"], 5),
             "c": round(pivot - r["c"], 5)} for r in rows]


def trials(seed):
    """Yield (t, base, sd, rows, end_idx) — every draw made before the caller runs anything."""
    rng = np.random.RandomState(int(seed))
    t = 0
    while True:
        name = NAMES[rng.randint(0, len(NAMES))]
        sd = 1 if rng.rand() < 0.5 else -1
        n_tail = rng.randint(4, 12)
        base = BASES[name]
        c, tail = base[-1]["c"], []
        for _ in range(n_tail):
            x = _cand(rng, c)
            tail.append(x)
            c = x["c"]
        bound_draw, bound_off = rng.rand(), rng.randint(0, n_tail)
        rows = base + tail
        if sd == -1:
            rows = _mirror(rows)
        yield t, name, sd, rows, ((len(base) + bound_off) if bound_draw < 0.4 else None)
        t += 1


def _sig(ms):
    cols = sorted(ms._out.keys()) if getattr(ms, "_out", None) else []
    evs = [(e.type, int(e.idx), None if e.price is None else round(float(e.price), 6),
            json.dumps(e.meta, sort_keys=True, default=str)) for e in ms.events]
    df_part = ms.df[cols].astype(str).values.tolist() if cols else []
    rev = [int(e.idx) for e in ms.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
    return hashlib.sha1(json.dumps([evs, df_part], default=str).encode()).hexdigest()[:16], rev


def run(out, seed, n):
    with open(out, "w") as f:
        for t, name, sd, rows, end_idx in trials(seed):
            if t >= int(n):
                break
            rec = dict(t=t, base=name, sd=sd, n=len(rows), end=end_idx)
            try:
                df = _prepare_df(rows)
                ms = _make_market_structure(df, struct_direction=sd, start_idx=0, structure_id=0, timeframe="H1",
                                            pip_size=_pip_size_from_pair(df), end_idx=end_idx)
                ms.run()
                rec["sig"], rec["rev"] = _sig(ms)
                rec.update(rir=getattr(ms, "_rts_rir", 0), wpe=getattr(ms, "_rts_wpe", 0), exp=getattr(ms, "_rts_exp", 0))
            except Exception as e:  # noqa: BLE001 — AssertionError included on purpose
                rec["err"] = f"{type(e).__name__}: {str(e)[:160]}"
                rec["tb"] = traceback.format_exc().splitlines()[-4:-1]
            f.write(json.dumps(rec) + "\n")


def rows(seed, t_want):
    for t, name, sd, rws, end_idx in trials(seed):
        if t == int(t_want):
            print(json.dumps(dict(base=name, sd=sd, end_idx=end_idx, rows=rws)))
            return


def compare(a, b):
    A = [json.loads(l) for l in open(a)]
    B = [json.loads(l) for l in open(b)]
    for tag, R in (("A", A), ("B", B)):
        errs = Counter(r["err"][:80] for r in R if "err" in r)
        print(f"{tag}: n={len(R)} errors={dict(errs)} wpe={sum(1 for r in R if r.get('wpe'))} "
              f"rir={sum(1 for r in R if r.get('rir'))} exp={sum(1 for r in R if r.get('exp'))}")
    diff, ex = Counter(), {}
    for x, y in zip(A, B):
        if "err" in x or "err" in y:
            k = f"A {'err' if 'err' in x else 'ok'} / B {'err' if 'err' in y else 'ok'}"
            if "err" in x and "err" in y and x["err"] == y["err"]:
                k = "same err"
        elif x["sig"] != y["sig"]:
            k = "sig diff, reversal " + ("same" if x["rev"] == y["rev"] else "DIFF")
        else:
            continue
        diff[k] += 1
        ex.setdefault(k, x["t"])
    print("A vs B:", dict(diff), "first trial per kind:", ex)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"run": run, "rows": rows, "compare": compare}[cmd](*args)
