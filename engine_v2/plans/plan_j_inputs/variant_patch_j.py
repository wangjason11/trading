"""PLAN_J (the MS fix plan: PLAN_I R9-1 / R9-2) — the minimal BEHAVIOUR patches of the variant measurements.

    python engine_v2/plans/plan_j_inputs/variant_patch_j.py TREE EDIT[,EDIT...]

TREE = a `git archive` of HEAD 515c449 (`TREE/engine_v2/...`). Each edit replaces an exact text that must occur ONCE
(else AssertionError — the tree is not that HEAD). Every edit adds `PJ_LOG` (a module-level list in
`structure/market_structure.py`) that the runners dump.

  census — instrumentation only (behaviour-neutral): at both RANGE back-fill sites (`_step_anchor`'s no-winner
           branch `[i, min(confirm, D))` and `_post_apply_range_check`'s `[apply, min(confirm, apply + K))`), every
           non-terminal back-fill's state diff (the fields of `MarketStructureState` that changed) and the events it
           emitted, by type. The F16 mechanism: those candles are RE-STEPPED afterwards (the no-winner branch returns
           i + 1; the post-apply branch returns apply + 1) with that state.
  J1a    — F16, "restore": after a NON-terminal range back-fill (both sites) the state, the events it emitted, the
           embedded FibTracker and the output rows are restored to their values before it; the range is then
           finalized with the CTS as of its start candle, and the re-steps process those candles in order. A
           terminal back-fill (a reversal inside it) is kept as today.
  J1b    — F16, "CTS only": no raw CTS update inside a range back-fill (both sites); every other per-candle effect of
           the back-fill (proximity confirmation, the BOS barrier, a reversal watch) kept.
  J1c    — J1a, but the range is seeded (and RANGE_STARTED's meta CTS stamped) with the CTS the back-fill reached —
           the decision-time CTS, as today.
  J2     — F17 (i): a breakout establishing a cycle T >= 1 whose detection threshold (`winner_break_th`) differs from
           `cts_threshold` at its apply is re-detected at its anchor against that threshold; unless the same pattern
           (name, apply) qualifies it is VOID — no event, no range reset, no state change; the step continues at
           apply + 1.
  J2b    — J2 + a void winner's back-fill undone (snapshot before a breakout winner's back-fill; on a void: restore,
           step the anchor candle, continue at anchor + 1 — the skipped anchors are scanned). Edits "J2,J2b".
  J3     — F17 (ii): the cycle >= 1 BOS window ends at the new cycle's CTS anchor (was: the breakout's apply); a
           window whose end precedes its start is logged (today's swap kept).
"""
import sys
from pathlib import Path

TREE, EDITS = Path(sys.argv[1]), sys.argv[2].split(",")
MS = TREE / "engine_v2" / "structure" / "market_structure.py"
src = MS.read_text(encoding="utf-8")


def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, f"{n} matches for {old[:80]!r}"
    src = src.replace(old, new)


# ---- the shared helpers (module level, after the StructureEvent dataclass) ----------------------------------------
HELPERS = '''

PJ_LOG: list = []   # PLAN_J variant instrumentation


def _pj_snapshot(ms, lo, hi):
    import copy
    memo = {id(e): e for e in ms.events}
    memo[id(ms.df)] = ms.df
    trk = None if ms._fib_tracker is None else copy.deepcopy(ms._fib_tracker.__dict__, memo)
    return (copy.deepcopy(ms.state.__dict__, memo), len(ms.events), trk, lo, hi,
            {c: a[lo:hi].copy() for c, a in ms._out.items()})


def _pj_restore(ms, snap):
    sd, n, trk, lo, hi, rows = snap
    ms.state.__dict__.clear()
    ms.state.__dict__.update(sd)
    del ms.events[n:]
    if trk is not None:
        ms._fib_tracker.__dict__.clear()
        ms._fib_tracker.__dict__.update(trk)
    for c, a in rows.items():
        ms._out[c][lo:hi] = a


_PJ_FIELDS = ("state", "cts", "cts_phase", "cts_threshold", "bos_threshold", "cts_confirmed_idx",
              "cts_confirmed_method", "reversal_watch_active", "pending_reversal_apply_idx", "range_active",
              "poi_inners_for_cycle", "bos_inner_for_cycle", "last_pullback_pat_apply_idx", "pullback_fired_for_cycle")


def _pj_census(ms, site, i, hi, before, n_before):
    after = ms.state.__dict__
    changed = {f: [repr(before[f]), repr(after[f])] for f in _PJ_FIELDS if before[f] != after[f]}
    from collections import Counter
    PJ_LOG.append({"kind": "backfill", "site": site, "sid": int(ms.state.structure_id), "tf": ms.timeframe,
                   "i": int(i), "hi": int(hi), "changed": changed,
                   "events": dict(Counter(e.type for e in ms.events[n_before:]))})
'''
ANCHOR = "\n\n@dataclass\nclass MarketStructureState:"

NOWIN_OLD = """                min_d = D if confirm_idx is None else min(confirm_idx, D)
                for k in range(i, min_d):
                    self._replay_step_no_patterns(k, freeze_range=True)
                    # Terminal inside the range back-fill: no finalize, no re-step of i
                    # (see the winner back-fill above).
                    if self.state.state == MarketState.REVERSAL:
                        return k + 1

                self._finalize_range_candidate_offline(i)
"""
POST_OLD = """            min_d = D if confirm_idx is None else min(confirm_idx, D)
            for k in range(i, min_d):
                self._replay_step_no_patterns(k, freeze_range=True)
                # Terminal inside the back-fill: no finalize (`_step_anchor` then
                # ends the step — see its winner back-fill).
                if st.state == MarketState.REVERSAL:
                    return

            self._finalize_range_candidate_offline(i)
"""


def backfill_variant(kind):
    """kind: census | restore | restore_seed | no_raw"""
    def body(indent, ret, site):
        p = " " * indent
        pre, post = [], []
        if kind == "census":
            pre = [f"{p}_pj_b = dict(self.state.__dict__); _pj_n = len(self.events)"]
            post = [f"{p}_pj_census(self, {site!r}, i, min_d, _pj_b, _pj_n)"]
        elif kind in ("restore", "restore_seed"):
            pre = [f"{p}_pj_snap = _pj_snapshot(self, i, min_d)"]
            post = [f"{p}_pj_seed = self.state.cts", f"{p}_pj_restore(self, _pj_snap)"]
        elif kind == "no_raw":
            pre = [f"{p}self._pj_no_raw = True"]
        loop = [f"{p}for k in range(i, min_d):",
                f"{p}    self._replay_step_no_patterns(k, freeze_range=True)",
                f"{p}    if self.state.state == MarketState.REVERSAL:"]
        if kind == "no_raw":
            loop += [f"{p}        self._pj_no_raw = False"]
        loop += [f"{p}        {ret}"]
        if kind == "no_raw":
            post = [f"{p}self._pj_no_raw = False"]
        fin = [f"{p}self._finalize_range_candidate_offline(i)"]
        if kind == "restore_seed":
            fin = [f"{p}_pj_cts_now = self.state.cts", f"{p}self.state.cts = _pj_seed",
                   f"{p}self._finalize_range_candidate_offline(i)", f"{p}self.state.cts = _pj_cts_now"]
        return "\n".join([f"{p}min_d = D if confirm_idx is None else min(confirm_idx, D)"] + pre + loop + post + fin) + "\n"
    rep(NOWIN_OLD, body(16, "return k + 1", "nowin"))
    rep(POST_OLD, body(12, "return", "postapply"))
    if kind == "no_raw":
        rep("""        self._maybe_update_cts_pre_confirm(i, via=CTS_UPDATED_RAW_VIA)

        # Per-candle CTS confirmation via sd zone proximity.""",
            """        if not getattr(self, "_pj_no_raw", False):
            self._maybe_update_cts_pre_confirm(i, via=CTS_UPDATED_RAW_VIA)

        # Per-candle CTS confirmation via sd zone proximity.""")


def j2():
    rep("""            # Breakout breaks range (if active)
            if st.range_active:
                self._deactivate_range(apply_idx, meta={"reason": "range_breakout", "pat": ev.name})
""", """            # PLAN_J J2 (F17 (i)): a breakout establishing cycle T >= 1 must break T-1's cts_threshold as it stands
            # at the apply — re-detected at its anchor when its detection threshold differs; else VOID.
            if st.cts is not None and st.cts_phase == "CONFIRMED" and st.winner_break_th != st.cts_threshold:
                _ev2 = self._bp.detect_best_for_anchor(int(ev.start_idx), self.struct_direction, st.cts_threshold)
                _ok = (_ev2 is not None and _ev2.name == ev.name and self._apply_idx(_ev2) == int(apply_idx))
                PJ_LOG.append({"kind": "J2", "void": not _ok, "tf": self.timeframe, "anchor": int(ev.start_idx),
                               "apply": int(apply_idx), "pat": ev.name, "det_th": st.winner_break_th,
                               "cts_th": st.cts_threshold, "cycle_next": int(st.cts_cycle_id) + 1})
                if not _ok:
                    print(f"[PJ J2] VOID breakout {ev.name} anchor {int(ev.start_idx)} apply {int(apply_idx)}: "
                          f"detection threshold {st.winner_break_th} != cts_threshold {st.cts_threshold} "
                          f"({self.timeframe})")
                    st.winner_break_th = None
                    return

            # Breakout breaks range (if active)
            if st.range_active:
                self._deactivate_range(apply_idx, meta={"reason": "range_breakout", "pat": ev.name})
""")


def j2b():
    """J2 + the void winner's back-fill undone: snapshot before a BREAKOUT winner's back-fill; on a void, restore, step
    the anchor candle itself (as the no-winner path does) and continue at anchor + 1 — every anchor the void winner
    skipped is scanned. Needs J2 (applied first)."""
    rep("""            for k in range(i, apply_idx):
                self._replay_step_no_patterns(k, freeze_range=True)""",
        """            _pj_wsnap = _pj_snapshot(self, i, apply_idx + 1) if kind == "breakout" else None
            self._pj_void = False
            for k in range(i, apply_idx):
                self._replay_step_no_patterns(k, freeze_range=True)""")
    rep("""            # "live-like": we act as if apply candle just closed
            self._apply_pattern_at_apply_idx(ev, apply_idx, kind)
""", """            # "live-like": we act as if apply candle just closed
            self._apply_pattern_at_apply_idx(ev, apply_idx, kind)
            if self._pj_void:
                _pj_restore(self, _pj_wsnap)
                PJ_LOG.append({"kind": "J2b_rescan", "anchor": int(i), "apply": int(apply_idx)})
                self._replay_step_no_patterns(i)
                return i + 1
""")
    rep("""                    st.winner_break_th = None
                    return

            # Breakout breaks range (if active)""", """                    st.winner_break_th = None
                    self._pj_void = True
                    return

            # Breakout breaks range (if active)""")


def j3():
    rep("bos_anchor_idx, bos_price = self._select_bos_on_breakout(apply_idx)",
        "bos_anchor_idx, bos_price = self._select_bos_on_breakout(apply_idx, cts_anchor_idx)")
    rep("    def _select_bos_on_breakout(self, breakout_apply_idx: int) -> tuple[int, float]:",
        "    def _select_bos_on_breakout(self, breakout_apply_idx: int, cts_anchor_idx: int) -> tuple[int, float]:")
    rep("""        s = int(window_start)
        e = int(breakout_apply_idx)
        if e < s:
            s, e = e, s
""", """        s = int(window_start)
        e = int(cts_anchor_idx)   # PLAN_J J3 (F17 (ii)): the window ends at the new cycle's CTS anchor
        if e < s:
            PJ_LOG.append({"kind": "J3_swap", "tf": self.timeframe, "s": s, "e": e, "apply": int(breakout_apply_idx)})
            s, e = e, s
""")


rep(ANCHOR, HELPERS + ANCHOR)
for edit in EDITS:
    {"census": lambda: backfill_variant("census"), "J1a": lambda: backfill_variant("restore"),
     "J1b": lambda: backfill_variant("no_raw"), "J1c": lambda: backfill_variant("restore_seed"),
     "J2": j2, "J2b": j2b, "J3": j3}[edit]()
MS.write_text(src, encoding="utf-8")
print(f"patched {MS} with {EDITS}")
