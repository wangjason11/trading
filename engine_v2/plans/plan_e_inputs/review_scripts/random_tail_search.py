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

F14 extension (PLAN_I §9.2 F14, §9.3 D1 / D5; 2026-10-08). Every trial (both streams) gets `f14`:
  - `o`  = shape (1): the downstream FibTracker's order tripwire replayed on `processing_order_key` order — a
           cycle T >= 1 CTS_ESTABLISHED (its BOS seen) without (sid, T-1)'s CTS_CONFIRMED handled first
           (= `FibEventOrderError`, raised since S2); [T, cts anchor, moment, T-1's CTS_CONFIRMED idx or None].
  - `s`  = shape (2): fib-handled events (CTS_ESTABLISHED / CTS_UPDATED / CTS_CONFIRMED / CTS_THRESHOLD_UPDATED)
           whose emission order != `processing_order_key` order — each inverted pair as [earlier-emitted type,
           its cycle, its stamped idx, later-emitted type, its cycle, its stamped idx] (S5's at-emit tracker sees
           the emission order; the D5 tripwire compares the two).
  - `npc` = proximity CTS_CONFIRMEDs; `pre` = those inside the NEXT cycle's breakout back-fill window
           [pattern_anchor_idx, moment) (shape (1)'s precondition; the inversion also needs the EST's CTS anchor
           <= the confirmation candle); `cmax` = the highest cycle established.
The WAVES stream (`waves` / `wrows`) builds multi-cycle trends — impulse / pullback / chop legs, pullbacks often
ending in a long-wick touch — so cycles >= 2 and proximity confirmations (cycle >= 1 only) are common; timeframe
H1 / M15 drawn per trial. The REAL stream (`real` / `rrows`) runs random 150–900 candle slices of a year of real
candles (`$RTS_REAL_DIR`, default `%TEMP%/pi/f14`: `NZD_USD_<H1|M15>_2025.csv`, columns time,o,h,l,c,volume — fetch
them with `multitf.data_bridge.fetch_lower_tf_data`), starting at the slice's sd-side extreme in its first 1–29
candles, half of them with a <= 1-pip jitter. `f14 OUT.jsonl` summarises any stream's output. Session 4 (PLAN_I
§9.10): 12,000 trials, shape (1) 0, shape (2) 5 — all PLAN_I §9.2 F16.

  python random_tail_search.py waves OUT.jsonl SEED N
  python random_tail_search.py wrows SEED T
  python random_tail_search.py real OUT.jsonl SEED N
  python random_tail_search.py rrows SEED T
  python random_tail_search.py f14 OUT.jsonl [OUT2.jsonl ...]
"""
import hashlib, json, os, sys, traceback
from collections import Counter

import numpy as np

import engine_v2.structure.event_fields as ef
import engine_v2.structure.market_structure as msmod
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.zones.cross_cycle_fib import CrossFibLimit
from engine_v2.zones.fib_tracker import FibEventOrderError

# PLAN_I S5: MarketStructure carries an embedded FibTracker (REQUIRED limit; no reference price here) and raises
# `FibEventOrderError` at the END of a run whose emission order is not the processing order — the F16 / F17 shapes
# this search hunts. `_ms_run` records it on the instance (`_rts_tripwire`) instead of failing the trial; the run
# is complete when it raises (events, df, invariants).
_NO_P = CrossFibLimit(None, "none")


def _ms_run(ms):
    try:
        ms.run()
    except FibEventOrderError as e:
        ms._rts_tripwire = str(e)[:240]
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data, _make_watch_over_second_cts_data
from engine_v2.tests.test_unified_probe import (_prepare_df, _make_multicycle_data,
                                                _make_second_cts_moment_after_anchor_data)

MS = msmod.MarketStructure
REV = msmod.MarketState.REVERSAL

# The expiry and `_rewind_to` were removed 2026-09-29 (F3b): wrap them only in trees that still have them.
if hasattr(MS, "_rewind_to"):
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

if hasattr(MS, "_maybe_expire_reversal_watch"):
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


_FIB_TYPES = ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED")


def _f14(events):
    """The F14 shapes on one MS run's events (see the module docstring). Keyed on list POSITION, never id()."""
    evs = list(events)
    order = sorted(range(len(evs)), key=lambda j: ef.processing_order_key(evs[j]))
    bos_seen, conf, o, skip = set(), {}, [], []
    for j in order:
        e = evs[j]
        key = (int(e.meta.get("structure_id", 0)), int(e.meta.get("cycle_id", 0)))
        if e.type == "BOS_CONFIRMED":
            bos_seen.add(key)
        elif e.type == "CTS_CONFIRMED":
            conf[key] = int(e.idx)
        elif e.type == "CTS_ESTABLISHED" and key not in bos_seen:
            # the orchestrator loop used to SKIP it silently (PLAN_I F17); since S2b it raises FibEventOrderError
            b = [ef.bos_anchor_idx(x) for x in evs if x.type == "BOS_CONFIRMED"
                 and (int(x.meta.get("structure_id", 0)), int(x.meta.get("cycle_id", 0))) == key]
            skip.append([key[1], ef.cts_anchor_idx(e), b[0] if b else None, int(e.idx),
                         (key[0], key[1] - 1) not in conf and key[1] >= 1])
        elif e.type == "CTS_ESTABLISHED" and key[1] >= 1 and key in bos_seen and (key[0], key[1] - 1) not in conf:
            later = [int(x.idx) for x in evs if x.type == "CTS_CONFIRMED"
                     and (int(x.meta.get("structure_id", 0)), int(x.meta.get("cycle_id", 0))) == (key[0], key[1] - 1)]
            o.append([key[1], ef.cts_anchor_idx(e), int(e.idx), later[0] if later else None])
    fib = [j for j in range(len(evs)) if evs[j].type in _FIB_TYPES]
    pos = {j: r for r, j in enumerate(sorted(fib, key=lambda j: ef.processing_order_key(evs[j])))}
    s = []
    for a in range(len(fib)):
        for b in range(a + 1, len(fib)):
            ja, jb = fib[a], fib[b]
            if pos[ja] > pos[jb]:
                ea, eb = evs[ja], evs[jb]
                s.append([ea.type, int(ea.meta.get("cycle_id", 0)), ef.stamped_idx(ea),
                          eb.type, int(eb.meta.get("cycle_id", 0)), ef.stamped_idx(eb)])
    prox = [e for e in evs if e.type == "CTS_CONFIRMED" and e.meta.get("via") == "sd_zone_proximity"]
    est = {int(e.meta.get("cycle_id", 0)): e for e in evs if e.type == "CTS_ESTABLISHED"}
    pre = []
    for e in prox:
        nxt = est.get(int(e.meta.get("cycle_id", 0)) + 1)
        if nxt is not None and int(nxt.meta["pattern_anchor_idx"]) <= int(e.idx) < int(nxt.idx):
            pre.append([int(e.meta.get("cycle_id", 0)), int(e.idx), int(nxt.meta["pattern_anchor_idx"]),
                        ef.cts_anchor_idx(nxt), int(nxt.idx)])
    return dict(o=o, s=s[:20], ns=len(s), npc=len(prox), pre=pre, cmax=max(est) if est else None, skip=skip)


def run(out, seed, n, lo=None, hi=None):
    with open(out, "w") as f:
        for t, name, sd, rows, end_idx, stop_n in trials(seed, lo, hi):
            if t >= int(n):
                break
            rec = dict(t=t, base=name, sd=sd, n=len(rows), end=end_idx, stop=stop_n)
            ms = None
            try:
                df = _prepare_df(rows)
                ms = _make_market_structure(df, cycle0_proximity_blocked=True, cross_fib_limit=_NO_P, struct_direction=sd, start_idx=0, structure_id=0, timeframe="H1",
                                            pip_size=_pip_size_from_pair(df), end_idx=end_idx,
                                            stop_after_cts_established=stop_n, debug_invariants=not NOINV)
                _ms_run(ms)
                rec["trip"] = getattr(ms, "_rts_tripwire", None)
                rec["sig"], rec["rev"] = _sig(ms)
                rec["f14"] = _f14(ms.events)
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


def _wcand(rng, c, up, kind, long_wick_counter):
    """One candle from close `c`, moving up (bull) or down; kind 0 maru, 1 normal, 2 pinbar, 3 small. Draws a fixed
    number of values whatever the branch (the stream stays aligned across trees)."""
    u = rng.uniform(0, 1, size=6)
    body = [15 + 45 * u[0], 8 + 22 * u[0], 2 + 6 * u[0], 1 + 5 * u[0]][kind] * PIP
    wa = [2 * u[1], 2 + 10 * u[1], 1 + 4 * u[1], 1 + 4 * u[1]][kind] * PIP   # the wick on the move's side
    wb = [2 * u[2], 2 + 10 * u[2], 15 + 35 * u[2], 1 + 4 * u[2]][kind] * PIP  # the wick on the counter side
    if kind == 2 and not long_wick_counter:
        wa, wb = wb, wa
    o = c + (u[3] - 0.5) * 4 * PIP
    cl = o + body if up else o - body
    hi_w, lo_w = (wa, wb) if up else (wb, wa)
    return {"o": round(o, 5), "h": round(max(o, cl) + hi_w, 5), "l": round(min(o, cl) - lo_w, 5), "c": round(cl, 5)}


def wave_trials(seed):
    """Yield (t, sd, tf, base, rows, n_waves) — a bullish multi-wave path (mirrored for sd = -1), every draw before
    the run. Leg plan per wave: [chop 0-4] -> pullback 1-5 candles toward a drawn depth of the last impulse
    (U(0.3, 1.25); a shortfall closed by the last candle's low wick with p = 0.6 — the zone touch) -> impulse 2-7."""
    rng = np.random.RandomState(int(seed))
    t = 0
    while True:
        sd = 1 if rng.rand() < 0.5 else -1
        tf = "H1" if rng.rand() < 0.6 else "M15"
        base = "mc9" if rng.rand() < 0.5 else "none"
        n_waves = rng.randint(3, 10)
        rows = list(BASES["mc9"]) if base == "mc9" else []
        c = rows[-1]["c"] if rows else 0.6000
        if not rows:   # an initial impulse from 0.6000
            for _ in range(rng.randint(3, 7)):
                x = _wcand(rng, c, rng.rand() < 0.85, rng.randint(0, 4), rng.rand() < 0.7)
                rows.append(x)
                c = x["c"]
        for _ in range(n_waves):
            hi_now = max(r["h"] for r in rows)
            lo_imp = min(r["l"] for r in rows[-6:])
            n_chop, n_pb, n_imp = rng.randint(0, 5), rng.randint(1, 6), rng.randint(2, 8)
            depth = rng.uniform(0.3, 1.25)
            touch = rng.rand() < 0.6
            for _ in range(n_chop * (rng.rand() < 0.4)):
                x = _wcand(rng, c, rng.rand() < 0.5, rng.randint(1, 4), rng.rand() < 0.5)
                rows.append(x)
                c = x["c"]
            target = hi_now - depth * max(hi_now - lo_imp, 10 * PIP)
            for k in range(n_pb):
                x = _wcand(rng, c, rng.rand() < 0.2, rng.randint(0, 4), rng.rand() < 0.7)
                if k == n_pb - 1 and touch and x["l"] > target:
                    x["l"] = round(target - rng.uniform(0, 6) * PIP, 5)
                else:
                    rng.uniform(0, 6)   # keep the draw count fixed
                rows.append(x)
                c = x["c"]
            for _ in range(n_imp):
                x = _wcand(rng, c, rng.rand() < 0.8, rng.randint(0, 4), rng.rand() < 0.7)
                rows.append(x)
                c = x["c"]
        if sd == -1:
            rows = _mirror(rows)
        yield t, sd, tf, base, rows, n_waves
        t += 1


def waves(out, seed, n):
    with open(out, "w") as f:
        for t, sd, tf, base, rows_, n_waves in wave_trials(seed):
            if t >= int(n):
                break
            rec = dict(t=t, sd=sd, tf=tf, base=base, n=len(rows_), w=n_waves)
            try:
                df = _prepare_df(rows_)
                ms = _make_market_structure(df, cycle0_proximity_blocked=True, cross_fib_limit=_NO_P, struct_direction=sd, start_idx=0, structure_id=0, timeframe=tf,
                                            pip_size=_pip_size_from_pair(df), debug_invariants=not NOINV)
                _ms_run(ms)
                rec["trip"] = getattr(ms, "_rts_tripwire", None)
                rec["sig"], rec["rev"] = _sig(ms)
                rec["f14"] = _f14(ms.events)
            except Exception as e:  # noqa: BLE001 — AssertionError included on purpose
                rec["err"] = f"{type(e).__name__}: {str(e)[:160]}"
                rec["tb"] = traceback.format_exc().splitlines()[-4:-1]
            f.write(json.dumps(rec) + "\n")
            f.flush()


_REAL = {}


def _real(tf):
    """The REAL stream's source candles: `$RTS_REAL_DIR/NZD_USD_<tf>_2025.csv` (time, o, h, l, c, volume)."""
    if tf not in _REAL:
        import pandas as pd
        d = os.environ.get("RTS_REAL_DIR") or os.path.join(os.environ["TEMP"], "pi", "f14")
        _REAL[tf] = pd.read_csv(os.path.join(d, f"NZD_USD_{tf}_2025.csv"))[["o", "h", "l", "c"]].to_dict("records")
    return _REAL[tf]


def real_trials(seed):
    """Yield (t, sd, tf, off, start, length, jit, rows) — a random slice of a year of real candles; the run starts
    at the slice's sd-side extreme within its first `w` candles (a swing start); `jit` = a <= 1-pip OHLC jitter
    (the order of o / c kept, h / l re-bounded). Every draw before the run."""
    rng = np.random.RandomState(int(seed))
    t = 0
    while True:
        sd = 1 if rng.rand() < 0.5 else -1
        tf = "H1" if rng.rand() < 0.4 else "M15"
        src = _real(tf)
        length = rng.randint(150, 900)
        off = rng.randint(0, len(src) - length)
        w = rng.randint(1, 30)
        jit = rng.rand() < 0.5
        noise = rng.uniform(-1, 1, size=(length, 4)) * PIP
        rows = [dict(r) for r in src[off:off + length]]
        if jit:
            for r, nz in zip(rows, noise):
                o, c = round(r["o"] + nz[0], 5), round(r["c"] + nz[1], 5)
                r.update(o=o, c=c, h=round(max(r["h"] + nz[2], o, c), 5), l=round(min(r["l"] + nz[3], o, c), 5))
        ext = [r["l"] for r in rows[:w]] if sd == 1 else [-r["h"] for r in rows[:w]]
        start = int(np.argmin(ext))
        yield t, sd, tf, off, start, length, jit, rows
        t += 1


def real(out, seed, n):
    with open(out, "w") as f:
        for t, sd, tf, off, start, length, jit, rows_ in real_trials(seed):
            if t >= int(n):
                break
            rec = dict(t=t, sd=sd, tf=tf, off=off, start=start, n=length, jit=jit)
            try:
                df = _prepare_df(rows_)
                ms = _make_market_structure(df, cycle0_proximity_blocked=True, cross_fib_limit=_NO_P, struct_direction=sd, start_idx=start, structure_id=0, timeframe=tf,
                                            pip_size=_pip_size_from_pair(df), debug_invariants=not NOINV)
                _ms_run(ms)
                rec["trip"] = getattr(ms, "_rts_tripwire", None)
                rec["sig"], rec["rev"] = _sig(ms)
                rec["f14"] = _f14(ms.events)
            except Exception as e:  # noqa: BLE001 — AssertionError included on purpose
                rec["err"] = f"{type(e).__name__}: {str(e)[:160]}"
                rec["tb"] = traceback.format_exc().splitlines()[-4:-1]
            f.write(json.dumps(rec) + "\n")
            f.flush()


def rrows(seed, t_want):
    for t, sd, tf, off, start, length, jit, rws in real_trials(seed):
        if t == int(t_want):
            print(json.dumps(dict(sd=sd, tf=tf, off=off, start=start, jit=jit, rows=rws)))
            return


def wrows(seed, t_want):
    for t, sd, tf, base, rws, n_waves in wave_trials(seed):
        if t == int(t_want):
            print(json.dumps(dict(sd=sd, tf=tf, base=base, rows=rws)))
            return


def f14(*paths):
    for p in paths:
        R = [json.loads(l) for l in open(p)]
        ok = [r for r in R if "f14" in r]
        errs = Counter(r["err"][:90] for r in R if "err" in r)
        cm = Counter(min(r["f14"]["cmax"], 4) if r["f14"]["cmax"] is not None else -1 for r in ok)
        print(f"{p}: n={len(R)} ok={len(ok)} errors={dict(errs)}")
        print(f"   cycles established (max, capped 4): {dict(sorted(cm.items()))}")
        print(f"   proximity confirmations: {sum(r['f14']['npc'] for r in ok)} in {sum(1 for r in ok if r['f14']['npc'])} trials;"
              f" inside the next cycle's back-fill (shape-1 precondition): {sum(len(r['f14']['pre']) for r in ok)}"
              f" in {sum(1 for r in ok if r['f14']['pre'])} trials")
        o = [r for r in ok if r["f14"]["o"]]
        s = [r for r in ok if r["f14"]["ns"]]
        print(f"   SHAPE 1 (tripwire): {len(o)} trials {[r['t'] for r in o[:10]]}")
        print(f"   SHAPE 2 (emission != processing order): {len(s)} trials {[r['t'] for r in s[:10]]}")
        kinds = Counter((x[0], x[3]) for r in s for x in r["f14"]["s"])
        if kinds:
            print(f"      pair kinds (earlier-emitted, later-emitted): {dict(kinds)}")
        fe = Counter(r["f14"]["o"][0][0] for r in o)
        if fe:
            print(f"      shape-1 cycles: {dict(fe)}")
        sk = [r for r in ok if r["f14"].get("skip")]
        print(f"   EST SKIPPED by the dispatch (its BOS sorts after it): {sum(len(r['f14']['skip']) for r in sk)} ESTs in"
              f" {len(sk)} trials {[r['t'] for r in sk[:10]]}; of them with T-1 unconfirmed at that point (the hidden"
              f" shape 1): {sum(1 for r in sk for x in r['f14']['skip'] if x[4])}")
        bk = Counter("cts<bos" if x[2] is not None and x[1] < x[2] else "tie/other" for r in sk for x in r["f14"]["skip"])
        if bk:
            print(f"      CTS anchor vs BOS anchor: {dict(bk)}")


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
    {"run": run, "rows": rows, "compare": compare, "waves": waves, "wrows": wrows, "real": real, "rrows": rrows,
     "f14": f14}[cmd](*args)
