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

In-watch extension (2026-09-29b; GOTCHAS "A Cycle Cannot Be Established Inside an Open Reversal Watch"):
  in_watch_est  a cycle established (`_emit_bos_confirmed`) while a reversal watch is open: the moment, cycle id, the
                watch's anchor / frozen barrier / expires_idx, its pending reversal (anchor, apply), the BOS before and
                the new one, rewind flag, chain; after the run: `kept` (the final events still hold that cycle's
                BOS_CONFIRMED at that moment), the run's reversal candle(s) + their `bos_frozen`, `pending_applied`
                (a final reversal ON the pending apply with the old frozen barrier), `inv4` (the df invariant-4 rule
                fires on the final rows) and the run's exception type if it raised
  in_watch_upd  a CTS_UPDATED (raw or pattern path) emitted while a watch is open: the moment, via, watch anchor, pending

F3b extension (2026-09-29c; a reversal applying exactly AT a watch's expiry E):
  watch         every watch opened (`_schedule_reversal_from_anchor`): anchor A, E, the pending apply (None -> the
                `rv_anchor_failed` clear), whether A was the running step's own anchor (`own_step`) or a candle inside
                another step's back-fill / apply row, that step's winner (kind, apply), rewind flag
  rev_apply     every reversal applied: path (`winner` = `_step_anchor`'s winner / `pending` = the scheduled pending),
                the open watch's anchor / E at that moment, `at_E` (apply == E), `own` (winner step anchor == the
                watch anchor; the close-break candle's own pattern) — aggregated as rev_apply[path|own/later|at_E/pre_E]
  each `expiry` record also gets `own_step` + the step winner of its watch's opening (joined by the watch anchor)

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
    r = dict(_ms=self, caller=_outer_caller(), test=(os.environ.get("PYTEST_CURRENT_TEST") or "").split(" (")[0],
             tf=self.timeframe, sid=int(self.state.structure_id),
             start=int(self.start_idx), end=int(self._effective_end), stop_n=self.stop_after_cts_established,
             scan=bool(self.enforce_cts0_new_extreme), enter=[], leave=[], bf_apply=[], rebuild=[], seed_over=[],
             expiry=[], win_past_exp=[], post_expiry=[], rewind_in_rev=[], in_watch_est=[], in_watch_upd=[],
             watch=[], rev_apply=[], _watch_by_anchor={},
             _steps=[], _last_step=None, n_rewinds=0, exc=None)
    _cur.append(r)
    try:
        out = _orig_run(self)
    except BaseException as e:
        r["exc"] = f"{type(e).__name__}: {str(e)[:120]}"
        _finish(self, r)
        raise
    finally:
        _cur.pop()
    _finish(self, r)
    return out
MS.run = run


def _inv4_rows(df):
    """The df invariant-4 rule (one watch = equal frozen barrier) on the final rows -> the firing row indices."""
    need = ("reversal_watch_active", "reversal_bos_th_frozen", "bos_threshold")
    if not all(c in df.columns for c in need):
        return []
    a = df["reversal_watch_active"].astype(bool)
    fr, bo = df["reversal_bos_th_frozen"].astype(float), df["bos_threshold"].astype(float)
    ch = a & a.shift(1).fillna(False).astype(bool) & (fr == fr.shift(1)) & (bo != bo.shift(1))
    return [int(x) for x in df.index[ch]]


def _finish(self, r):
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
    rev_bf = [evs[p].meta.get("bos_frozen") for p in rev_pos]
    if r["in_watch_est"]:
        try:
            inv4 = _inv4_rows(self.df)
        except Exception:  # noqa: BLE001 — a crashed run may have no flushed rows
            inv4 = None
        bos_conf = {(int(e.idx), int(e.meta.get("cycle_id", -1))) for e in evs if e.type == "BOS_CONFIRMED"}
        for w in r["in_watch_est"]:
            w["kept"] = (w["i"], w["cycle"]) in bos_conf
            w["rev_idx"] = r["rev_idx"]
            w["rev_bos_frozen"] = rev_bf
            w["pending_applied"] = bool(w["pending"] is not None and w["pending"] in r["rev_idx"]
                                        and w["frozen"] in rev_bf)
            w["inv4"] = inv4
            w["exc"] = r["exc"]
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
              "rewind_in_rev", "in_watch_est", "in_watch_upd"):
        agg[k] += len(r[k])
    for w in r["in_watch_est"]:
        agg["in_watch_est[rewind]" if w["rewind"] else "in_watch_est[run]"] += 1
        agg["in_watch_est[kept]"] += int(w["kept"])
        agg["in_watch_est[pending_applied]"] += int(w["pending_applied"])
        agg["in_watch_est[inv4_fires]"] += int(bool(w["inv4"]))
        agg["in_watch_est[run_raised]"] += int(w["exc"] is not None)
        agg["in_watch_est[pending==moment]"] += int(w["pending"] is not None and w["pending"] == w["i"])
    for u in r["in_watch_upd"]:
        agg["in_watch_upd[%s]" % u["via"]] += 1
    agg["run_loop_rewinds"] += r["n_rewinds"]
    for w in r["win_past_exp"]:
        agg["win_past_exp[%s]" % w["kind"]] += 1
    agg["expiry_in_frozen_backfill"] += sum(1 for e in r["expiry"] if e["frozen"])
    for w in r["watch"]:
        rw = "rebuild" if w["rewind"] else "run"
        if w["pending"] is None:
            agg[f"watch[{rw}|no_pending]"] += 1
        else:
            agg[f"watch[{rw}|pending{'==E' if w['pending'] == w['E'] else '<E'}|{'own_step' if w['own_step'] else 'in_other_step'}]"] += 1
    for a in r["rev_apply"]:
        who = "own" if a["own"] else ("later" if a["own"] is False else "-")
        pos = "no_watch" if a["E"] is None else ("at_E" if a["at_E"] else "pre_E")
        agg[f"rev_apply[{a['path']}|{who}|{pos}]"] += 1
    for e in r["expiry"]:
        w = r["_watch_by_anchor"].get(e["watch_anchor"])
        e["own_step"] = None if w is None else w["own_step"]
        e["watch_step_winner"] = None if w is None else w["step_winner"]
        e["at_edge"] = e["expires"] == r["end"]
        agg["expiry[%s|%s|%s]" % ("rebuild" if e["rewind"] else "run",
                                  "own_step" if e["own_step"] else "in_other_step",
                                  "edge" if e["at_edge"] else "inside")] += 1
    interesting = r["n_rev"] > 1 or r["n_from_rev"] or post or r["leave"] or r["bf_apply"] or r["rebuild"] \
        or r["seed_over"] or r["sid_meta"] or r["expiry"] or r["win_past_exp"] or r["post_expiry"] \
        or r["rewind_in_rev"] or r["in_watch_est"] or r["in_watch_upd"] \
        or any(a["at_E"] for a in r["rev_apply"])
    all_runs.append((r["caller"][0], r["tf"], r["sid"], r["start"], r["end"], r["stop_n"], r["n_rev"]))
    del r["_ms"], r["_steps"], r["_last_step"], r["_watch_by_anchor"]
    if interesting:
        runs.append(r)


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
        if w is not None and w[2] == "reversal":
            # F3b: the E of the watch this reversal belongs to (the open one, or the one a close-break at i opens: D)
            E = int(st.reversal_watch_expires_idx) if st.reversal_watch_active and st.reversal_watch_expires_idx is not None else int(D)
            agg["rev_winner_chosen[%s]" % ("at_E" if int(w[1]) == E else "pre_E")] += 1
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


_orig_bos_conf = MS._emit_bos_confirmed
def _emit_bos_confirmed(self, idx, price, *, bos_anchor_idx, meta=None):
    r, st = _rec(self), self.state
    if r is not None and st.reversal_watch_active:
        r["in_watch_est"].append(dict(
            i=int(idx), cycle=int(st.cts_cycle_id), watch_anchor=st.reversal_watch_start_idx,
            frozen=st.reversal_bos_th_frozen, expires=st.reversal_watch_expires_idx,
            pending=st.pending_reversal_apply_idx, pending_anchor=st.pending_reversal_pattern_anchor_idx,
            bos_before=st.bos_threshold, bos_new=float(price), state=st.state.value,
            rewind=bool(getattr(self, "_in_rewind", False)), chain=_chain()))
    return _orig_bos_conf(self, idx, price, bos_anchor_idx=bos_anchor_idx, meta=meta)
MS._emit_bos_confirmed = _emit_bos_confirmed


_orig_cts_upd = MS._emit_cts_updated
def _emit_cts_updated(self, idx, price, meta=None):
    r, st = _rec(self), self.state
    if r is not None and st.reversal_watch_active:
        r["in_watch_upd"].append(dict(i=int(idx), via=str((meta or {}).get("via")), watch_anchor=st.reversal_watch_start_idx,
                                      expires=st.reversal_watch_expires_idx, pending=st.pending_reversal_apply_idx,
                                      rewind=bool(getattr(self, "_in_rewind", False))))
    return _orig_cts_upd(self, idx, price, meta)
MS._emit_cts_updated = _emit_cts_updated


_orig_sched = MS._schedule_reversal_from_anchor
def _schedule_reversal_from_anchor(self, anchor_idx, *, bos_frozen):
    out = _orig_sched(self, anchor_idx, bos_frozen=bos_frozen)
    r, st = _rec(self), self.state
    if r is not None and st.reversal_watch_active and st.reversal_watch_start_idx == int(anchor_idx):
        s = r["_steps"][-1] if r["_steps"] else None
        w = dict(A=int(anchor_idx), E=st.reversal_watch_expires_idx, pending=st.pending_reversal_apply_idx,
                 own_step=bool(s is not None and s["i"] == int(anchor_idx)),
                 step_i=None if s is None else s["i"], step_winner=None if s is None else s["winner"],
                 rewind=bool(getattr(self, "_in_rewind", False)))
        r["watch"].append(w)
        r["_watch_by_anchor"][int(anchor_idx)] = w
    return out
MS._schedule_reversal_from_anchor = _schedule_reversal_from_anchor


_orig_apply_at = MS._apply_pattern_at_apply_idx
def _apply_pattern_at_apply_idx(self, ev, apply_idx, kind):
    r, st = _rec(self), self.state
    if r is not None and kind == "reversal":
        path = "pending" if sys._getframe(1).f_code.co_name == "_maybe_apply_pending_reversal" else "winner"
        s = r["_steps"][-1] if r["_steps"] else None
        A, E = st.reversal_watch_start_idx, st.reversal_watch_expires_idx
        ap = self._apply_idx(ev)
        own = None if (path == "pending" or A is None or s is None) else (s["i"] == int(A))
        r["rev_apply"].append(dict(path=path, apply=None if ap is None else int(ap), A=A, E=E,
                                   at_E=bool(E is not None and ap is not None and int(ap) == int(E)), own=own,
                                   step_i=None if s is None else s["i"], rewind=bool(getattr(self, "_in_rewind", False))))
    return _orig_apply_at(self, ev, apply_idx, kind)
MS._apply_pattern_at_apply_idx = _apply_pattern_at_apply_idx


def _dump():
    json.dump(dict(agg=dict(sorted(agg.items())), runs=runs, all_runs=all_runs), open(OUT, "w"), indent=1, default=str)
atexit.register(_dump)
