"""Random-tail search over MarketStructure (the F3 measurement, 2026-09-29).

Verified fixture bases + random candle tails, through whichever `engine_v2` tree is first on PYTHONPATH — run the
SAME seed in two trees (the repo and a scratch `git archive` copy with a variant) and diff the outputs:

  python random_tail_search.py run OUT.jsonl SEED N [LO HI]   # N trials -> one JSON line each
  python random_tail_search.py rows SEED T             # regenerate trial T of SEED (rows JSON + bound + sd)
  python random_tail_search.py compare A.jsonl B.jsonl # errors, counters, signature diffs (reversal candle same?)

Per trial: the base, sd (-1 = the base + tail mirrored at 1.2), an optional bound, the output signature (events + the
MS output columns), the reversal candles, the exception if any (AssertionError included), and the counters `rir`
(run-loop rewinds entered in REVERSAL), `wpe` (reversal winners applying after the open watch's `expires_idx`),
`exp` (expiries fired). All RNG draws happen before the run — never inside the try — so every tree sees the same
trials (a draw inside a try is skipped on an exception and desynchronises the stream). Smoke-run a few trials in the
foreground first: an empty background log is not "no finds" (buffered output; an import error dies silently).

In-watch extension (2026-09-29b): `iwe` = cycles established (`_emit_bos_confirmed`) while a reversal watch is open,
with `iw` (per establishment: moment, watch anchor / frozen / expires, pending apply, the new BOS, rewind flag) and the
final reversals' `bos_frozen` (`rev_bf`); `iwu` = CTS_UPDATED emitted while a watch is open. `RTS_NOINV=1` runs with
`debug_invariants=False` and evaluates the df invariants offline on the SAME rows (`inv` = the invariant error or
None) — one run, the verdict and the output. `LO HI` (optional) = the tail-length range (default 4..11, the original
stream) and switches on the WIDE mode: one extra draw per trial for an early stop (`stop_after_cts_established=2`,
p=0.2) — a different stream from the default, so compare WIDE runs only with WIDE runs of the same LO HI.
"""
import hashlib, json, os, sys, traceback
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
        key = "_rts_expo" if getattr(self, "_rts_own", {}).get(st.reversal_watch_start_idx) else "_rts_expb"
        setattr(self, key, getattr(self, key, 0) + 1)
    return _o_exp(self, i)
MS._maybe_expire_reversal_watch = _exp

_o_bc = MS._emit_bos_confirmed
def _bc(self, idx, price, *, bos_anchor_idx, meta=None):
    st = self.state
    if st.reversal_watch_active:
        self._rts_iw = getattr(self, "_rts_iw", []) + [dict(
            i=int(idx), anchor=st.reversal_watch_start_idx, frozen=st.reversal_bos_th_frozen,
            expires=st.reversal_watch_expires_idx, pending=st.pending_reversal_apply_idx, bos_new=float(price),
            rewind=bool(getattr(self, "_in_rewind", False)))]
    return _o_bc(self, idx, price, bos_anchor_idx=bos_anchor_idx, meta=meta)
MS._emit_bos_confirmed = _bc

_o_cu = MS._emit_cts_updated
def _cu(self, idx, price, meta=None):
    if self.state.reversal_watch_active:
        self._rts_iwu = getattr(self, "_rts_iwu", 0) + 1
    return _o_cu(self, idx, price, meta)
MS._emit_cts_updated = _cu

# F3b extension (2026-09-29c): a reversal applying exactly at a watch's expiry E, by path — `aEo` the close-break
# candle's own pattern as its step's winner, `aEl` a later anchor's winner, `aEp` the scheduled pending (0 at HEAD:
# the expiry runs first); `expo` / `expb` = expiries whose watch anchor was the running step's own anchor / a candle
# inside another step (`exp` = both); `wat` = watches that survived their anchor (a pending was scheduled).
_o_sched = MS._schedule_reversal_from_anchor
def _sched(self, anchor_idx, *, bos_frozen):
    out = _o_sched(self, anchor_idx, bos_frozen=bos_frozen)
    st = self.state
    if st.reversal_watch_active and st.pending_reversal_apply_idx is not None:
        self._rts_wat = getattr(self, "_rts_wat", 0) + 1
        own = getattr(self, "_rts_step", None) == int(anchor_idx)
        self._rts_own = {**getattr(self, "_rts_own", {}), int(anchor_idx): own}
    return out
MS._schedule_reversal_from_anchor = _sched

_o_step = MS._step_anchor
def _stp(self, i):
    prev, self._rts_step = getattr(self, "_rts_step", None), int(i)
    try:
        return _o_step(self, i)
    finally:
        self._rts_step = prev
MS._step_anchor = _stp

_o_apat = MS._apply_pattern_at_apply_idx
def _apat(self, ev, apply_idx, kind):
    st = self.state
    if kind == "reversal" and st.reversal_watch_expires_idx is not None and self._apply_idx(ev) == st.reversal_watch_expires_idx:
        if sys._getframe(1).f_code.co_name == "_maybe_apply_pending_reversal":
            key = "_rts_aEp"
        else:
            key = "_rts_aEo" if getattr(self, "_rts_step", None) == st.reversal_watch_start_idx else "_rts_aEl"
        setattr(self, key, getattr(self, key, 0) + 1)
    return _o_apat(self, ev, apply_idx, kind)
MS._apply_pattern_at_apply_idx = _apat

NOINV = os.environ.get("RTS_NOINV") == "1"

BASES = {
    "dr9": _make_double_rewind_data()[:9], "dr10": _make_double_rewind_data()[:10],
    "dr14": _make_double_rewind_data()[:14], "dr15": _make_double_rewind_data()[:15],
    "wo11": _make_watch_over_second_cts_data()[:11], "wo12": _make_watch_over_second_cts_data()[:12],
    "sc11": _make_second_cts_moment_after_anchor_data()[:11], "mc9": _make_multicycle_data()[:9],
}
NAMES = sorted(BASES)
# Targeted bases (opt-in `RTS_BASES=iw7,iw8,dr6`; the default stream is unchanged): the in-watch-cycle pin's prefix
# (`test_ms_new_cycle_ends_watch._in_watch_cycle_rows`: watch 4 open, the breakout 6-7 -> cycle 1 at 7 inside
# it) cut before / after the establishment, and the double-rewind base cut with watch 4 open (candles 6+ random).
if os.environ.get("RTS_BASES"):
    from engine_v2.tests.test_ms_new_cycle_ends_watch import _in_watch_cycle_rows
    BASES.update({"iw7": _in_watch_cycle_rows()[:7], "iw8": _in_watch_cycle_rows()[:8],
                  "dr6": _make_double_rewind_data()[:6]})
    NAMES = os.environ["RTS_BASES"].split(",")
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


def trials(seed, lo=None, hi=None):
    """Yield (t, base, sd, rows, end_idx, stop_n) — every draw made before the caller runs anything."""
    rng = np.random.RandomState(int(seed))
    wide = lo is not None
    t = 0
    while True:
        name = NAMES[rng.randint(0, len(NAMES))]
        sd = 1 if rng.rand() < 0.5 else -1
        n_tail = rng.randint(int(lo), int(hi) + 1) if wide else rng.randint(4, 12)
        base = BASES[name]
        c, tail = base[-1]["c"], []
        for _ in range(n_tail):
            x = _cand(rng, c)
            tail.append(x)
            c = x["c"]
        bound_draw, bound_off = rng.rand(), rng.randint(0, n_tail)
        stop_n = (2 if rng.rand() < 0.2 else None) if wide else None
        rows = base + tail
        if sd == -1:
            rows = _mirror(rows)
        yield t, name, sd, rows, ((len(base) + bound_off) if bound_draw < 0.4 else None), stop_n
        t += 1


def _sig(ms):
    cols = sorted(ms._out.keys()) if getattr(ms, "_out", None) else []
    evs = [(e.type, int(e.idx), None if e.price is None else round(float(e.price), 6),
            json.dumps(e.meta, sort_keys=True, default=str)) for e in ms.events]
    df_part = ms.df[cols].astype(str).values.tolist() if cols else []
    rev = [int(e.idx) for e in ms.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
    return hashlib.sha1(json.dumps([evs, df_part], default=str).encode()).hexdigest()[:16], rev


def run(out, seed, n, lo=None, hi=None):
    with open(out, "w") as f:
        for t, name, sd, rows, end_idx, stop_n in trials(seed, lo, hi):
            if t >= int(n):
                break
            rec = dict(t=t, base=name, sd=sd, n=len(rows), end=end_idx, stop=stop_n)
            ms = None
            try:
                df = _prepare_df(rows)
                ms = _make_market_structure(df, struct_direction=sd, start_idx=0, structure_id=0, timeframe="H1",
                                            pip_size=_pip_size_from_pair(df), end_idx=end_idx,
                                            stop_after_cts_established=stop_n, debug_invariants=not NOINV)
                ms.run()
                rec["sig"], rec["rev"] = _sig(ms)
                rec.update(rir=getattr(ms, "_rts_rir", 0), wpe=getattr(ms, "_rts_wpe", 0), exp=getattr(ms, "_rts_exp", 0))
                rec.update({k: getattr(ms, "_rts_" + k, 0) for k in ("aEo", "aEl", "aEp", "expo", "expb", "wat")})
                if NOINV:
                    try:
                        ms._check_invariants_df()
                        rec["inv"] = None
                    except AssertionError as e:
                        rec["inv"] = str(e)[:160]
            except Exception as e:  # noqa: BLE001 — AssertionError included on purpose
                rec["err"] = f"{type(e).__name__}: {str(e)[:160]}"
                rec["tb"] = traceback.format_exc().splitlines()[-4:-1]
            if ms is not None:
                rec["iwe"] = len(getattr(ms, "_rts_iw", []))
                rec["iwu"] = getattr(ms, "_rts_iwu", 0)
                if rec["iwe"]:
                    rec["iw"] = ms._rts_iw
                    rec["rev_bf"] = [e.meta.get("bos_frozen") for e in ms.events
                                     if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
            f.write(json.dumps(rec) + "\n")


def rows(seed, t_want, lo=None, hi=None):
    for t, name, sd, rws, end_idx, stop_n in trials(seed, lo, hi):
        if t == int(t_want):
            print(json.dumps(dict(base=name, sd=sd, end_idx=end_idx, stop=stop_n, rows=rws)))
            return


def compare(a, b):
    A = [json.loads(l) for l in open(a)]
    B = [json.loads(l) for l in open(b)]
    for tag, R in (("A", A), ("B", B)):
        errs = Counter(r["err"][:80] for r in R if "err" in r)
        print(f"{tag}: n={len(R)} errors={dict(errs)} wpe={sum(1 for r in R if r.get('wpe'))} "
              f"rir={sum(1 for r in R if r.get('rir'))} exp={sum(1 for r in R if r.get('exp'))} "
              f"iwe={sum(1 for r in R if r.get('iwe'))} iwu={sum(1 for r in R if r.get('iwu'))} "
              f"inv={sum(1 for r in R if r.get('inv'))}")
        print(f"   F3b: wat={sum(r.get('wat', 0) for r in R)} aEo={sum(r.get('aEo', 0) for r in R)} "
              f"aEl={sum(r.get('aEl', 0) for r in R)} aEp={sum(r.get('aEp', 0) for r in R)} "
              f"expo={sum(r.get('expo', 0) for r in R)} expb={sum(r.get('expb', 0) for r in R)} "
              f"(trials: exp {sum(1 for r in R if r.get('exp'))}, aE* "
              f"{sum(1 for r in R if r.get('aEo') or r.get('aEl') or r.get('aEp'))})")
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
