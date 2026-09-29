"""Shadow: every MarketStructure run -> reversal accounting (the double-reversal audit, 2026-09-28).

Behaviour-neutral wrappers on `MarketStructure` (class level, so every run is seen: H1 main sids, the FC
probes' Phase 2, every sub build, test fixtures). Per run: the caller, the sid, the final events'
STATE_CHANGED(to=reversal) / (from=reversal) counts, and the dynamic observations:

  enter      a `_set_state(REVERSAL)` that changes the state (site = the caller chain)
  leave      a `_set_state(X != REVERSAL)` while the state IS REVERSAL (site = the caller chain)
  bf_apply   the pending reversal applied inside a FROZEN back-fill (`_replay_step_no_patterns(k, freeze_range=True)`)
  rebuild    a `_step_anchor` inside `_rewind_to`'s rebuild that ends in REVERSAL
  seed_over  `_rewind_to` returned with a state != the rebuild's REVERSAL (the seed restore overwrote it)
  post_ev    events emitted (list order) after the first STATE_CHANGED(to=reversal) of the final list
  sid_meta   reversal events whose meta structure_id != the run's constructor sid

F3 extension (2026-09-28c; the reversal-watch expiry vs a later pattern):
  expiry        a watch expiry that fired (`_maybe_expire_reversal_watch` requested the rewind): candle, the watch's
                anchor / expires_idx / pending apply, whether inside a FROZEN back-fill, the step anchor, the chain
  win_past_exp  a step winner (any kind) chosen while a watch is active whose apply > the watch's expires_idx —
                kind == "reversal" is F3 (`_best_bopb_pattern_at_anchor` does not cap at the expiry; the scheduler does)
  post_expiry   a step that kept stepping candles after an expiry fired inside it (the rest of the step is discarded
                by the run loop's rewind): the candles stepped after it, the step's winner, its end state
  rewind_in_rev a run-loop `_rewind_to` entered while the state IS REVERSAL (the seed restore discards the reversal):
                jump_to, the discarded reversal candle(s), the step that produced it

Import before the code under test: replay via the `runpy` recipe (README), suite via
`PYTHONPATH=<this dir> python -m pytest -p reversal_shadow`. Writes REVERSAL_SHADOW_OUT (default: the temp dir).
"""
import atexit, json, os, sys, tempfile
from collections import Counter

import engine_v2.structure.market_structure as msmod

MS = msmod.MarketStructure
REV = msmod.MarketState.REVERSAL
OUT = os.environ.get("REVERSAL_SHADOW_OUT") or os.path.join(tempfile.gettempdir(), "reversal_shadow.json")
_MS_FILE = os.path.normcase(msmod.__file__)

runs = []          # one dict per run (only the interesting ones keep their details)
all_runs = []      # compact: (caller, tf, sid, start, end, stop_n, n_rev) for every run
agg = Counter()


def _chain(depth=6):
    """(function, line) of the MS frames above the caller of the wrapper, innermost first."""
    out, f = [], sys._getframe(2)
    while f is not None and len(out) < depth:
        if os.path.normcase(f.f_code.co_filename) == _MS_FILE:
            out.append(f"{f.f_code.co_name}:{f.f_lineno}")
        f = f.f_back
    return out


def _outer_caller():
    f = sys._getframe(2)
    while f is not None and os.path.normcase(f.f_code.co_filename) == _MS_FILE:
        f = f.f_back
    names = []
    while f is not None and len(names) < 3:
        mod = os.path.splitext(os.path.basename(f.f_code.co_filename))[0]
        names.append(f"{mod}.{f.f_code.co_name}")
        f = f.f_back
    return names


_cur = []  # stack of per-run records (runs do not nest, but keep it safe)


def _rec(self):
    return _cur[-1] if _cur and _cur[-1]["_ms"] is self else None


_orig_run = MS.run
def run(self):
    r = dict(_ms=self, caller=_outer_caller(), tf=self.timeframe, sid=int(self.state.structure_id),
             start=int(self.start_idx), end=int(self._effective_end), stop_n=self.stop_after_cts_established,
             scan=bool(self.enforce_cts0_new_extreme), enter=[], leave=[], bf_apply=[], rebuild=[], seed_over=[],
             expiry=[], win_past_exp=[], post_expiry=[], rewind_in_rev=[], _steps=[], _last_step=None,
             n_rewinds=0)
    _cur.append(r)
    try:
        out = _orig_run(self)
    finally:
        _cur.pop()
    evs = self.events
    rev_pos = [p for p, e in enumerate(evs) if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
    from_rev = [p for p, e in enumerate(evs) if e.type == "STATE_CHANGED" and e.meta.get("from") == "reversal"]
    r["n_rev"] = len(rev_pos)
    r["rev_idx"] = [int(evs[p].idx) for p in rev_pos]
    r["n_from_rev"] = len(from_rev)
    r["from_rev"] = [(int(evs[p].idx), evs[p].meta.get("to"), evs[p].meta.get("reason")) for p in from_rev]
    r["sid_meta"] = [int(evs[p].meta.get("structure_id")) for p in rev_pos if int(evs[p].meta.get("structure_id")) != r["sid"]]
    post = []
    if rev_pos:
        first = rev_pos[0]
        post = [(e.type, int(e.idx), e.meta.get("to") or e.meta.get("via") or e.meta.get("reason")) for e in evs[first + 1:]]
    r["post_ev"] = post
    kind = "H1" if r["tf"].upper().startswith("H") else r["tf"]
    agg[f"runs[{kind}]"] += 1
    agg["runs"] += 1
    agg["reversals"] += r["n_rev"]
    agg["runs_with_reversal"] += int(r["n_rev"] > 0)
    agg["runs_with_2plus_reversals"] += int(r["n_rev"] > 1)
    agg["STATE_CHANGED_from_reversal"] += r["n_from_rev"]
    agg["runs_with_post_reversal_events"] += int(bool(post))
    agg["post_reversal_events"] += len(post)
    for t, _, _ in post:
        agg[f"post_ev[{t}]"] += 1
    agg["reversal_sid_meta_mismatch"] += len(r["sid_meta"])
    for k in ("enter", "leave", "bf_apply", "rebuild", "seed_over", "expiry", "win_past_exp", "post_expiry",
              "rewind_in_rev"):
        agg[k] += len(r[k])
    agg["run_loop_rewinds"] += r["n_rewinds"]
    for w in r["win_past_exp"]:
        agg["win_past_exp[%s]" % w["kind"]] += 1
    agg["expiry_in_frozen_backfill"] += sum(1 for e in r["expiry"] if e["frozen"])
    interesting = r["n_rev"] > 1 or r["n_from_rev"] or post or r["leave"] or r["bf_apply"] or r["rebuild"] \
        or r["seed_over"] or r["sid_meta"] or r["expiry"] or r["win_past_exp"] or r["post_expiry"] \
        or r["rewind_in_rev"]
    all_runs.append((r["caller"][0], r["tf"], r["sid"], r["start"], r["end"], r["stop_n"], r["n_rev"]))
    del r["_ms"], r["_steps"], r["_last_step"]
    if interesting:
        runs.append(r)
    return out
MS.run = run


_orig_set = MS._set_state
def _set_state(self, new_state, i, meta=None):
    r = _rec(self)
    if r is not None:
        cur = self.state.state
        if new_state == REV and cur != REV:
            r["enter"].append(dict(i=int(i), chain=_chain(), rewind=bool(getattr(self, "_in_rewind", False))))
        elif cur == REV and new_state != REV:
            r["leave"].append(dict(i=int(i), to=new_state.value, chain=_chain(),
                                   rewind=bool(getattr(self, "_in_rewind", False))))
    return _orig_set(self, new_state, i, meta)
MS._set_state = _set_state


_orig_apply_pending = MS._maybe_apply_pending_reversal
def _maybe_apply_pending_reversal(self, i):
    applied = _orig_apply_pending(self, i)
    r = _rec(self)
    if applied and r is not None:
        f = sys._getframe(1)  # _replay_step_no_patterns
        frozen = bool(f.f_locals.get("freeze_range", False))
        if frozen:
            r["bf_apply"].append(dict(i=int(i), chain=_chain(), rewind=bool(getattr(self, "_in_rewind", False))))
    return applied
MS._maybe_apply_pending_reversal = _maybe_apply_pending_reversal


_orig_step = MS._step_anchor
def _step_anchor(self, i):
    r = _rec(self)
    if r is not None:
        r["_steps"].append(dict(i=int(i), winner=None, expiry=None, after_expiry=[],
                                rewind=bool(getattr(self, "_in_rewind", False))))
    try:
        nxt = _orig_step(self, i)
    finally:
        s = r["_steps"].pop() if r is not None else None
    if r is not None:
        s["next"] = int(nxt)
        s["end_state"] = self.state.state.value
        r["_last_step"] = s
        if s["expiry"] is not None and s["after_expiry"]:
            r["post_expiry"].append(dict(step_i=s["i"], expiry_i=s["expiry"]["i"], winner=s["winner"],
                                         after=s["after_expiry"], end_state=s["end_state"], rewind=s["rewind"]))
    if r is not None and getattr(self, "_in_rewind", False) and self.state.state == REV:
        r["rebuild"].append(dict(i=int(i), next=int(nxt)))
    return nxt
MS._step_anchor = _step_anchor


_orig_rewind = MS._rewind_to
def _rewind_to(self, jump_to, *, seed=None):
    r = _rec(self)
    if r is not None:
        r["n_rewinds"] += 1
        if self.state.state == REV:
            rev = [int(e.idx) for e in self.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
            ls = r["_last_step"] or {}
            r["rewind_in_rev"].append(dict(jump_to=int(jump_to), discarded_rev=rev,
                                           seed_state=None if seed is None else str(seed.get("market_state")),
                                           step_i=ls.get("i"), winner=ls.get("winner"),
                                           expiry=ls.get("expiry"), after_expiry=ls.get("after_expiry")))
    n_before = len(r["rebuild"]) if r is not None else 0
    out = _orig_rewind(self, jump_to, seed=seed)
    if r is not None and len(r["rebuild"]) > n_before and self.state.state != REV:
        r["seed_over"].append(dict(jump_to=int(jump_to), state_after=self.state.state.value,
                                   seed_state=None if seed is None else str(seed.get("market_state"))))
    return out
MS._rewind_to = _rewind_to


_orig_best = MS._best_bopb_pattern_at_anchor
def _best_bopb_pattern_at_anchor(self, *, i, breakout_th, pullback_th, D):
    w = _orig_best(self, i=i, breakout_th=breakout_th, pullback_th=pullback_th, D=D)
    r = _rec(self)
    if r is not None and r["_steps"]:
        st = self.state
        s = r["_steps"][-1]
        s["winner"] = None if w is None else (w[2], int(w[1]))
        if (w is not None and st.reversal_watch_active and st.reversal_watch_expires_idx is not None
                and int(w[1]) > int(st.reversal_watch_expires_idx)):
            r["win_past_exp"].append(dict(
                i=int(i), kind=w[2], apply=int(w[1]), D=int(D), watch_anchor=st.reversal_watch_start_idx,
                expires=int(st.reversal_watch_expires_idx), pending=st.pending_reversal_apply_idx,
                state=st.state.value, rewind=bool(getattr(self, "_in_rewind", False))))
    return w
MS._best_bopb_pattern_at_anchor = _best_bopb_pattern_at_anchor


_orig_expire = MS._maybe_expire_reversal_watch
def _maybe_expire_reversal_watch(self, i):
    st = self.state
    fires = (st.reversal_watch_active and st.reversal_watch_expires_idx is not None
             and int(i) >= int(st.reversal_watch_expires_idx) and st.reversal_watch_start_idx is not None)
    info = (st.reversal_watch_start_idx, st.reversal_watch_expires_idx, st.pending_reversal_apply_idx) if fires else None
    out = _orig_expire(self, i)
    r = _rec(self)
    if fires and r is not None:
        f = sys._getframe(1)  # _replay_step_no_patterns
        s = r["_steps"][-1] if r["_steps"] else None
        rec = dict(i=int(i), watch_anchor=info[0], expires=info[1], pending=info[2],
                   frozen=bool(f.f_locals.get("freeze_range", False)), step_i=None if s is None else s["i"],
                   chain=_chain(), rewind=bool(getattr(self, "_in_rewind", False)))
        r["expiry"].append(rec)
        if s is not None and s["expiry"] is None:
            s["expiry"] = rec
    return out
MS._maybe_expire_reversal_watch = _maybe_expire_reversal_watch


_orig_replay_step = MS._replay_step_no_patterns
def _replay_step_no_patterns(self, i, *, freeze_range=False):
    r = _rec(self)
    if r is not None and r["_steps"] and r["_steps"][-1]["expiry"] is not None:
        r["_steps"][-1]["after_expiry"].append((int(i), bool(freeze_range)))
    return _orig_replay_step(self, i, freeze_range=freeze_range)
MS._replay_step_no_patterns = _replay_step_no_patterns


def _dump():
    json.dump(dict(agg=dict(sorted(agg.items())), runs=runs, all_runs=all_runs), open(OUT, "w"), indent=1, default=str)
atexit.register(_dump)
