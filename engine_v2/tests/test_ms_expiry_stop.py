"""A reversal watch's expiry ends the step that fires it; a later anchor's reversal cannot outlive the watch
(2026-09-29; zones-audit F3, found by the double-reversal landing review).

MARKET_STRUCTURE_SPEC "Expiry inside a step": a close-break at anchor A opens a watch that expires at
E = min(A + range_max_k, effective_end); the expiry requests a rewind to A + 1 (false break). Two gaps let a step
act past its own expiry, all of it then thrown away by the run loop's rewind + seed restore (a direct
`state.state =`):
  F3  `_best_bopb_pattern_at_anchor` capped a frozen-BOS reversal candidate only at D = anchor + 5, not at the
      watch's E (the scheduler `_schedule_reversal_from_anchor` does) — for an anchor after A the winner could
      apply after E: its back-fill fired the expiry, the reversal applied anyway, the rewind discarded it;
  D   every step kept running after an expiry inside it (winner back-fill -> apply, range back-fill -> finalize,
      `_post_apply_range_check` -> finalize): a second expiry there OVERWROTE the pending jump + seed, and a
      `_rewind_to` rebuild (which ignores nested jumps, LANDMINES "MarketStructure Deep-Couples…" 1(a)) replayed
      the discarded reversal and crashed on its "never reaches a reversal" assert.
Fix (user decision 2026-09-29, option A+D): the candidate is capped at the open watch's E (the scheduler's `>`:
apply == E stays a winner); every back-fill / the post-apply check ends the step at the expiry, returning the jump
target (the run loop honours it; a rebuild resumes there, as it did after the old continuation); the run loop
asserts a rewind is never requested in REVERSAL. Live-like: at E no reversal has completed, so the watch expires;
a pattern completing later re-qualifies against the post-expiry barrier after the rewind.

Reference window (`review_scripts/reversal_shadow.py` over every MS run): 12 runs, 0 expiries -> byte-identical.
Suite: 42 expiries, 0 F3, 26 range back-fills stepping past their expiry (now stopped). The 23 first-pass ones
change nothing (the rewind discarded the rest); the 3 inside `_rewind_to` rebuilds change the rebuilt prefix
(`test_mechanism`'s full run — pinned below). Random tails (12k per tree): pre-fix (edeef26) 192 F3 winners, 200
rewinds entered in REVERSAL, 1 rebuild-assert crash; A+D 0 / 0 / 0 rebuild-assert crashes (a separate, pre-existing
df-invariant false positive on back-to-back watches remained — fixed next, `test_ms_invariant_watch_identity.py`).
The fixtures below come from those searches (rows embedded).

Re-derived 2026-09-29 when a new cycle started to END an open watch (MARKET_STRUCTURE_SPEC "A new cycle ends an open
watch"): every fixture here was `_make_double_rewind_data`-based, and its first-pass cycle 1 @8 — established inside
the watch opened at 4 — now ends that watch, so the step-9 mechanisms no longer arose. Candle 8 is replaced by one
closing back below CTS_0 (`_no_cycle_at_8`): the watch survives to 9 as before. Each end-to-end pin was checked in
two scratch trees with the new rule — the F3 fix reverted (all fail) and the cap alone (the stop pins fail) — and on
this tree (pass). The jump-overwrite case has no known instance under the new rule (see the note at its former fixture).
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.structure.market_structure import MarketState, MarketStructure
from engine_v2.structure.structure_engine import _make_market_structure, compute_bounded_structure
from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _R, _prepare_df


# ---------------------------------------------------------------------------------------------------------------
# Candle 8 of `_make_double_rewind_data` completes a breakout (cycle 1 @8, new high .6066) inside the watch opened at
# 4 — which ENDS that watch since 2026-09-29. This candle closes back at .6040, below CTS_0 .6042: no cycle at 8.
_C8 = _R(0.60540, 0.60560, 0.60380, 0.60400)


def _no_cycle_at_8(rows: list[dict]) -> list[dict]:
    rows = list(rows)
    rows[8] = _C8
    return rows


# F3 — sd=+1, start 0, NZD_USD H1 (persisted 2026-09-28 by the landing review; `_make_double_rewind_data()[:9]` +
# 5 candles, candle 8 = `_C8` since 2026-09-29). 0-2 cycle 0 established at 2 (CTS .6042, BOS_0 = l0 .5998).
#   4   bear maru closing .5982: pullback -> CTS_CONFIRMED@4 + range; close-breaks BOS_0 -> watch A=4, E=9; the
#       anchor-4 reversal (`one_maru_opposite(-1)`) confirms at 9 = E -> pending apply 9.
#   9   bear maru closing .5962: the expiry fires here (it precedes the pending apply 9 in the per-candle step:
#       false break, BOS := l4 .5980, rewind to 5). Anchor 9's own reversal against the frozen .5998 applies at 10.
#       Pre-fix chose it as step 9's winner (cap D = 14): its back-fill fired the expiry at 9, the reversal applied at
#       10, the run loop rewound IN REVERSAL and the seed restore discarded it. Now: capped at E = 9 -> step 9 has no
#       winner while the watch is open, and its step ends at the expiry candle 9 (nothing at 10 runs).
#   After the rewind (both): candle 5 probes the BOS to .5978; 9 close-breaks it -> watch 9 (E 13), pending
#   `one_maru_continuous(-1)` 9-10 -> the reversal @10 (bos_frozen .5978) — the final outcome is unchanged.
def _f3_rows() -> list[dict]:
    return _no_cycle_at_8(_make_double_rewind_data()[:9]) + [
        _R(0.60640, 0.60660, 0.59600, 0.59620),   # 9
        _R(0.59700, 0.59720, 0.58800, 0.58820),   # 10
        _R(0.58820, 0.58840, 0.58300, 0.58320),   # 11
        _R(0.58320, 0.58400, 0.58250, 0.58380),   # 12
        _R(0.58380, 0.58420, 0.58300, 0.58350),   # 13
    ]


# Rebuild crash — `_no_cycle_at_8(_make_double_rewind_data())` at `end_idx=17` (sd=+1; 2026-09-29, replacing seed 12
# trial 3597, whose F3-into-a-rebuild path needed the in-watch cycle @8). Watch A=4, E=9 (pending 9). Step 7's winner
# is a BREAKOUT applying at 11, past E: its back-fill fires the expiry at 9 (J1, rewind to 5). The post-J1 path
# establishes cycle 1 at 11, a watch at 14 expires at the bound 17 (J2, rewind to 15), and the `_rewind_to` rebuild
# replays 0..14 IGNORING the nested J1 jump. Without the stop the rebuild's step 7 went on past the expiry (10, 11:
# the breakout applied, a cycle the post-J1 path never had) and that discarded continuation reached a reversal at step
# 14 -> AssertionError "rewind rebuild reached a reversal" (the F3 fix reverted AND the cap alone both crash). Now the
# rebuild's step 7 ends at the nested expiry and resumes at 5 (Deep-Couples 1(a): on the un-reset state).
def _rebuild_crash_rows() -> list[dict]:
    return _no_cycle_at_8(_make_double_rewind_data())


# Jump overwrite (seed 12 trial 796, persisted until 2026-09-29): with the cap ALONE a step's back-fill ran on past
# the expiry, a second watch opened there expired inside the same step and OVERWROTE the jump target + seed. That
# fixture's step-9 pullback winner existed only in the BREAKOUT state left by the in-watch cycle @8 — which now ENDS
# the watch — so its pin was retired. No instance is known under the new rule (0 in 56k cap-only random tails incl.
# 20k from an open watch, 0 in a 66-variant sweep of its candles 7-8); not proven unreachable. The site it exercised
# (the winner back-fill stops at the expiry and returns the jump target) is pinned by
# `test_the_winner_backfill_stops_at_the_expiry` below.


# Boundary — seed 21 trial 231 (`_make_double_rewind_data()[:6]` + 4, sd=+1): watch A=4, E=9, pending 9; anchor 8's
# reversal against the frozen .5998 applies exactly at 9 = E -> still a winner (the scheduler's `>`), applied by
# the winner path before the apply-row step's expiry check -> the reversal @9, no rewind (unchanged).
def _boundary_rows() -> list[dict]:
    return list(_make_double_rewind_data()[:6]) + [
        _R(0.59898, 0.60328, 0.59771, 0.60185),   # 6
        _R(0.60186, 0.60258, 0.60137, 0.60212),   # 7
        _R(0.60187, 0.60248, 0.59711, 0.59817),   # 8
        _R(0.59844, 0.59852, 0.58946, 0.58962),   # 9
    ]


def _rows(fn, sd: int, native_sd: int = 1) -> list[dict]:
    return fn() if sd == native_sd else _noreg._mirror(fn())


def _price(p: float, sd: int) -> float:
    return p if sd == 1 else round(_noreg._MIRROR - p, 5)


@pytest.fixture
def trace(monkeypatch):
    """Class-level taps on every MS run: rewinds (jump_to, state at entry), winners chosen while a watch is open
    (anchor, kind, apply, expires) and candles stepped inside a step after an expiry fired in it."""
    log = {"rewinds": [], "winners": [], "after_expiry": []}
    steps = []
    o_rw, o_best, o_step = MarketStructure._rewind_to, MarketStructure._best_bopb_pattern_at_anchor, MarketStructure._step_anchor
    o_exp, o_rs = MarketStructure._maybe_expire_reversal_watch, MarketStructure._replay_step_no_patterns

    def rw(self, jump_to, *, seed=None):
        log["rewinds"].append((int(jump_to), self.state.state.value))
        return o_rw(self, jump_to, seed=seed)

    def best(self, *, i, breakout_th, pullback_th, D):
        w = o_best(self, i=i, breakout_th=breakout_th, pullback_th=pullback_th, D=D)
        st = self.state
        if w is not None and st.reversal_watch_active and not getattr(self, "_in_rewind", False):
            log["winners"].append((int(i), w[2], int(w[1]), int(st.reversal_watch_expires_idx)))
        return w

    def step(self, i):
        steps.append([False])
        try:
            return o_step(self, i)
        finally:
            steps.pop()

    def exp(self, i):
        st = self.state
        fires = st.reversal_watch_active and int(i) >= int(st.reversal_watch_expires_idx)
        out = o_exp(self, i)
        if fires and steps:
            steps[-1][0] = True
        return out

    def rs(self, i, *, freeze_range=False):
        if steps and steps[-1][0]:
            log["after_expiry"].append(int(i))
        return o_rs(self, i, freeze_range=freeze_range)

    monkeypatch.setattr(MarketStructure, "_rewind_to", rw)
    monkeypatch.setattr(MarketStructure, "_best_bopb_pattern_at_anchor", best)
    monkeypatch.setattr(MarketStructure, "_step_anchor", step)
    monkeypatch.setattr(MarketStructure, "_maybe_expire_reversal_watch", exp)
    monkeypatch.setattr(MarketStructure, "_replay_step_no_patterns", rs)
    return log


def _run(rows, sd: int):
    with redirect_stdout(io.StringIO()):
        return compute_bounded_structure(_prepare_df(rows), 0, sd)


def _reversals(events):
    return [(int(e.idx), e.meta["from"], e.meta["pat"], e.meta["bos_frozen"])
            for e in events if e.type == "STATE_CHANGED" and e.meta["to"] == "reversal"]


def _cycles(events):
    return [(int(e.idx), e.meta["cycle_id"]) for e in events if e.type == "CTS_ESTABLISHED"]


def _assert_no_action_past_an_expiry(log) -> None:
    assert not [w for w in log["winners"] if w[1] == "reversal" and w[2] > w[3]]   # F3: capped at the watch's E
    assert log["after_expiry"] == []                                              # D: the step ends at the expiry
    assert all(state != "reversal" for _, state in log["rewinds"])                 # nothing to discard


@pytest.mark.parametrize("sd", [1, -1])
def test_f3_the_expiry_resolves_the_watch_not_a_later_anchors_reversal(sd, trace):
    df = _prepare_df(_rows(_f3_rows, sd))
    ms = _make_market_structure(df, struct_direction=sd, start_idx=0)
    bos0 = _price(0.5998, sd)
    assert ms._apply_idx(ms._bp.detect_best_for_anchor(4, -sd, bos0)) == 9   # the watch's pending apply == E
    assert ms._apply_idx(ms._bp.detect_best_for_anchor(9, -sd, bos0)) == 10  # anchor 9's reversal: past E

    res = _run(_rows(_f3_rows, sd), sd)
    _assert_no_action_past_an_expiry(trace)
    assert trace["winners"] == []                     # pre-fix: (9, "reversal", 10, 9) — past E
    assert trace["rewinds"] == [(5, "pullback")]       # pre-fix: (5, "reversal") — the reversal discarded
    assert _reversals(res.events) == [(10, "pullback_range", "one_maru_continuous", _price(0.5978, sd))]
    assert _cycles(res.events) == [(2, 0)]
    assert res.reversal_idx == 10


@pytest.mark.parametrize("sd", [1, -1])
def test_a_rebuild_no_longer_replays_a_continuation_into_a_reversal(sd, trace):
    with redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_rows(_rebuild_crash_rows, sd)), 0, sd, end_idx=17)
    _assert_no_action_past_an_expiry(trace)
    assert trace["winners"] == [(7, "breakout", 11, 9)]   # a breakout past E: the step stops at the expiry anyway
    assert trace["rewinds"] == [(5, "pullback_range"), (15, "pullback")]
    assert _reversals(res.events) == [] and res.reversal_idx is None
    assert _cycles(res.events) == [(2, 0), (11, 1)]


@pytest.mark.parametrize("sd", [1, -1])
def test_a_later_anchors_reversal_applying_at_the_expiry_still_wins(sd, trace):
    res = _run(_rows(_boundary_rows, sd), sd)
    assert trace["winners"] == [(8, "reversal", 9, 9)]
    assert trace["rewinds"] == []
    assert _reversals(res.events) == [(9, "pullback_range", "one_maru_continuous", _price(0.5998, sd))]
    assert res.reversal_idx == 9


def test_a_rebuild_no_longer_replays_a_discarded_continuation():
    """The rebuild-crash fixture's event list (sd +1, `end_idx=17`): the J2 rebuild (0..14) meets J1's expiry at 9
    inside step 7's winner back-fill, ends the step there and re-steps from 5 on the un-reset state (Deep-Couples 1(a):
    the rebuilt list runs backwards in time, 9 -> 5). Without the stop the step went on to the breakout at 11 and the
    run crashed (above)."""
    from engine_v2.tests.test_ms_stop_after_cts import _make
    ms = _make(_rebuild_crash_rows(), end_idx=17)
    with redirect_stdout(io.StringIO()):
        ms.run()
    sig = [(e.type, int(e.idx)) for e in ms.events]
    assert sig[9:15] == [("STATE_CHANGED", 6), ("BOS_THRESHOLD_UPDATED", 9), ("BOS_THRESHOLD_UPDATED", 5),
                         ("REVERSAL_WATCH_START", 9), ("BOS_THRESHOLD_UPDATED", 9), ("RANGE_RESET", 11)]
    assert _cycles(ms.events) == [(2, 0), (11, 1)]
    assert len(sig) == 28


# ---------------------------------------------------------------------------------------------------------------
# Unit pins per site (stubs, like test_ms_reversal_terminal's): the cap, the four stops, the run-loop tripwire.
def _ms(sd: int = 1):
    with redirect_stdout(io.StringIO()):
        ms = _make_market_structure(_prepare_df(_f3_rows()), struct_direction=sd, start_idx=0)
    ms._init_output_arrays(len(ms.df))
    return ms


@pytest.mark.parametrize("apply, kind", [(9, "reversal"), (10, None)])
def test_the_reversal_candidate_is_capped_at_the_open_watchs_expiry(apply, kind, monkeypatch):
    ms = _ms()
    st = ms.state
    st.state, st.reversal_watch_active, st.reversal_bos_th_frozen, st.reversal_watch_expires_idx = (
        MarketState.RANGE, True, 0.5998, 9)
    ev = object()
    monkeypatch.setattr(ms._bp, "detect_best_for_anchor", lambda i, d, thr: ev if thr == 0.5998 else None)
    monkeypatch.setattr(ms, "_apply_idx", lambda e: apply)
    w = ms._best_bopb_pattern_at_anchor(i=8, breakout_th=None, pullback_th=None, D=13)
    assert (None if w is None else w[2]) == kind


def _jump_at(ms, k_fire: int, stepped: list):
    def step(k, *, freeze_range=False):
        stepped.append((k, freeze_range))
        if k == k_fire:
            ms.state.jump_to_idx = 5
    return step


def test_the_winner_backfill_stops_at_the_expiry(monkeypatch):
    ms, stepped, applied = _ms(), [], []
    monkeypatch.setattr(ms, "_best_bopb_pattern_at_anchor", lambda **kw: (object(), 12, "pullback"))
    monkeypatch.setattr(ms, "_replay_step_no_patterns", _jump_at(ms, 10, stepped))
    monkeypatch.setattr(ms, "_apply_pattern_at_apply_idx", lambda *a: applied.append(a))
    assert ms._step_anchor(9) == 5                    # the jump target (a rebuild resumes there)
    assert stepped == [(9, True), (10, True)] and applied == []


def test_the_step_stops_after_an_expiry_in_the_post_apply_backfill(monkeypatch):
    ms, stepped = _ms(), []

    def apply(ev, apply_idx, kind):   # stands in for a breakout whose post-apply back-fill expired the watch
        ms.state.jump_to_idx = 5

    monkeypatch.setattr(ms, "_best_bopb_pattern_at_anchor", lambda **kw: (object(), 11, "breakout"))
    monkeypatch.setattr(ms, "_replay_step_no_patterns", lambda k, *, freeze_range=False: stepped.append((k, freeze_range)))
    monkeypatch.setattr(ms, "_apply_pattern_at_apply_idx", apply)
    assert ms._step_anchor(9) == 5
    assert stepped == [(9, True), (10, True)]         # the back-fill only — no apply-row re-write at 11


def test_the_range_backfill_stops_at_the_expiry(monkeypatch):
    ms, stepped, finalized = _ms(), [], []
    ms.state.state = MarketState.BREAKOUT             # ranges allowed, none active
    monkeypatch.setattr(ms, "_best_bopb_pattern_at_anchor", lambda **kw: None)
    monkeypatch.setattr(ms, "_is_range_candle_given_confirm", lambda i: (True, i + 4))
    monkeypatch.setattr(ms, "_replay_step_no_patterns", _jump_at(ms, 10, stepped))
    monkeypatch.setattr(ms, "_finalize_range_candidate_offline", lambda i: finalized.append(i))
    assert ms._step_anchor(9) == 5
    assert stepped == [(9, True), (10, True)] and finalized == []   # no finalize, no re-step of 9


def test_the_post_apply_range_backfill_stops_at_the_expiry(monkeypatch):
    ms, stepped, finalized = _ms(), [], []
    monkeypatch.setattr(ms, "_is_range_candle_given_confirm", lambda i: (True, i + 4))
    monkeypatch.setattr(ms, "_replay_step_no_patterns", _jump_at(ms, 10, stepped))
    monkeypatch.setattr(ms, "_finalize_range_candidate_offline", lambda i: finalized.append(i))
    ms._post_apply_range_check(9)
    assert stepped == [(9, True), (10, True)] and finalized == []


def test_the_run_loop_never_rewinds_a_reversal(monkeypatch):
    ms = _ms()

    def step(i):
        ms.state.state = MarketState.REVERSAL
        ms.state.jump_to_idx = 1
        return i + 1

    monkeypatch.setattr(ms, "_step_anchor", step)
    with pytest.raises(AssertionError, match="rewind requested in REVERSAL"):
        ms.run()
