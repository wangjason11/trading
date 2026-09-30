"""Plan B — `MarketStructure(stop_after_cts_established=N)`: the opt-in early stop.

Contract under test (`plans/PLAN_B_double_cts_early_stop.md` §2 / §3.1; MARKET_STRUCTURE_SPEC
"Early stop after N CTS_ESTABLISHED"): with the option set, `run()` ends at the FIRST QUIESCENT
point (no reversal watch active, no pending reversal, no pending rewind) after the N-th
`CTS_ESTABLISHED`, records it in `early_stop_idx` (the first anchor NOT processed) and hands
back the event list built so far. Everything else — the `end_idx` bound, the post-run assert,
the output rows — is unchanged. With the option unset (`None`, the default) the run is
byte-identical to today's.

Fixtures:
  - `_make_multicycle_data` (test_unified_probe, Plan A §5.3): 4 CTS cycles established at
    2/10/15/20 (confirmed_at == idx), cycle 0 CONFIRMED at 7, no reversal, no watch, no
    rewind — so the stopped run must be an exact ORDERED PREFIX of the unbounded run.
  - `_make_watch_over_second_cts_data` (below, Plan B §4.1 "quiescence"): a BOS close-break
    with a pending reversal is open at the moment the 2nd CTS is established, so the stop
    must wait for the watch to resolve (crafted + independently verified 2026-09-20).
  - `_make_double_rewind_data` (below, Plan B §1/§2 "rebuilt-prefix" exception): until 2026-09-29
    two expiry-rewinds, one before the 2nd CTS and one after the stop point; `_rewind_to` ignores
    the earlier jump, so the exit classification read a prefix the machine had already
    superseded — the one mechanism under which early stop and classify-at-exit differ. Since a new
    cycle ENDS an open watch (MARKET_STRUCTURE_SPEC "A new cycle ends an open watch") the first
    rewind is gone and the two agree (`TestRebuiltPrefixException`); since F3b (2026-09-29) MS never
    rewinds at all (the expiry rewind was removed), so the exception cannot arise.
"""
from __future__ import annotations

import json

import pytest

from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import MarketStructure, StructureEvent
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.structure.structure_engine import (
    _make_market_structure,
    _pip_size_from_pair,
    compute_bounded_structure,
)
from engine_v2.structure.unified_probe import _run_phase2, unified_probe
from engine_v2.tests.test_unified_probe import (
    _make_multicycle_data,
    _prepare_df,
    _ref_zone_uptrend,
)
from engine_v2.zones.zone_proximity import DEFAULT_PROBE_RESET_PIPS, DEFAULT_PROBE_RESET_WICK

_START = 0
_SD = 1


def _make(raw, *, end_idx=None, **kw):
    """Construct one MS exactly as `compute_bounded_structure` wires it (resolvers + pip
    size), plus the extra kwargs under test. Not yet run."""
    df = _prepare_df(raw)
    return _make_market_structure(
        df,
        struct_direction=_SD,
        start_idx=_START,
        structure_id=0,
        timeframe="H1",
        pip_size=_pip_size_from_pair(df),
        end_idx=end_idx,
        **kw,
    )


def _run(raw, *, end_idx=None, **kw):
    """`_make` + `run()`. Returns the MS instance (events, df, early_stop_idx)."""
    ms = _make(raw, end_idx=end_idx, **kw)
    ms.run()
    return ms


def _run_logged(raw, *, end_idx=None, **kw):
    """Like `_run`, but records the state after EVERY `_step_anchor` call:
    (anchor, next_i, n CTS_ESTABLISHED so far, reversal_watch_active, pending_reversal_apply_idx).
    This is what `_should_stop_after_cts` sees at the stop check."""
    ms = _make(raw, end_idx=end_idx, **kw)
    log = []
    orig = ms._step_anchor

    def step(i):
        nxt = orig(i)
        st = ms.state
        log.append((int(i), int(nxt), sum(1 for ev in ms.events if ev.type == "CTS_ESTABLISHED"),
                    bool(st.reversal_watch_active), st.pending_reversal_apply_idx))
        return nxt

    ms._step_anchor = step
    ms.run()
    return ms, log


def _sig(events):
    """Ordered whole-event signature: type, idx, price, full meta."""
    return [
        (ev.type, int(ev.idx), None if ev.price is None else round(float(ev.price), 6),
         json.dumps(ev.meta, sort_keys=True, default=str))
        for ev in events
    ]


def _cts_est(events):
    return [ev for ev in events if ev.type == "CTS_ESTABLISHED"]


def _out_cols(ms):
    from engine_v2.structure.market_structure import (
        _OUT_FLOAT_NAN, _OUT_FLOAT_NEG1, _OUT_INT_NEG1, _OUT_INT_ZERO, _OUT_OBJ_EMPTY,
    )
    return list(_OUT_INT_NEG1 + _OUT_INT_ZERO + _OUT_FLOAT_NEG1 + _OUT_FLOAT_NAN + _OUT_OBJ_EMPTY)


# ---------------------------------------------------------------------------
# §4.1 — the stop option on the multi-cycle fixture
# ---------------------------------------------------------------------------

class TestStopAfterCtsEstablished:
    def test_fixture_precondition_no_watch_no_rewind(self):
        """The prefix assertions below are exact only when the unbounded run has no rewind
        after the stop point — this fixture has no reversal watch at all (watch-start
        events survive rebuilds, so their absence proves no expiry-rewind happened)."""
        full = _run(_make_multicycle_data())
        assert [ev.type for ev in full.events if ev.type == "REVERSAL_WATCH_START"] == []
        assert [(int(e.idx), int(e.meta["confirmed_at"])) for e in _cts_est(full.events)] == \
            [(2, 2), (10, 10), (15, 15), (20, 20)]

    def test_stop_after_two_is_an_ordered_prefix_of_the_full_run(self):
        raw = _make_multicycle_data()
        full = _run(raw)
        stopped = _run(raw, stop_after_cts_established=2)

        est = _cts_est(stopped.events)
        assert len(est) >= 2
        assert stopped.early_stop_idx is not None
        # The 2nd CTS (cycle 1) is established by the breakout anchored at 9 applying at
        # 10 -> the step returns next_i = 11, quiescent -> the stop lands there.
        assert stopped.early_stop_idx == 11
        assert int(est[1].meta["confirmed_at"]) == 10
        # Nothing at/after the 3rd cycle of the full run was reached.
        third_moment = int(_cts_est(full.events)[2].meta["confirmed_at"])
        assert max(int(ev.idx) for ev in stopped.events) < third_moment
        # No reversal rows.
        assert not (stopped.df["market_state"].astype(str) == "reversal").any()
        # Ordered prefix (not an idx filter: an event stamped at an anchor or a label — a
        # pattern-path CTS_UPDATED, a RANGE_STARTED at its confirm_idx; a BOS_CONFIRMED
        # until Plan E E4b — can be <= stop_idx though emitted after it).
        assert _sig(full.events)[: len(stopped.events)] == _sig(stopped.events)
        assert len(stopped.events) < len(full.events)

    def test_stopped_rows_before_the_stop_match_the_full_run(self):
        """Output rows written before the stop point are the full run's rows (no rewind in
        this fixture). The last step's range back-fill may have written rows up to
        `range_max_k` past `early_stop_idx` (a range candidate at the apply candle is
        stamped at its label `confirm_idx` <= i+5 — the same in both runs; on the reference
        window FC(0,0) stops at 1021 with `RANGE_STARTED@1022`), so the "never written"
        guarantee (structure_id -1, market_state "") starts at `early_stop_idx + range_max_k`."""
        raw = _make_multicycle_data()
        full = _run(raw)
        stopped = _run(raw, stop_after_cts_established=2)
        cols = _out_cols(stopped)
        s = int(stopped.early_stop_idx)
        a = full.df.loc[: s - 1, cols].reset_index(drop=True)
        b = stopped.df.loc[: s - 1, cols].reset_index(drop=True)
        assert a.equals(b)
        untouched_from = s + int(stopped.range_max_k)
        assert (stopped.df.loc[untouched_from:, "structure_id"].astype(int) == -1).all()
        assert (stopped.df.loc[untouched_from:, "market_state"].astype(str) == "").all()
        assert max(int(ev.idx) for ev in stopped.events) < untouched_from

    def test_option_unset_is_identical_to_today(self):
        """The default path is untouched: omitted kwarg == explicit None == the
        `compute_bounded_structure` primitive (no new default there, plan §3.4)."""
        raw = _make_multicycle_data()
        omitted = _run(raw)
        explicit = _run(raw, stop_after_cts_established=None)
        prim = compute_bounded_structure(_prepare_df(raw), _START, _SD)
        assert omitted.early_stop_idx is None
        assert explicit.early_stop_idx is None
        assert _sig(omitted.events) == _sig(explicit.events) == _sig(prim.events)
        cols = _out_cols(omitted)
        assert omitted.df[cols].equals(explicit.df[cols])
        assert omitted.df[cols].equals(prim.df[cols])

    @pytest.mark.parametrize("bad", [0, -1])
    def test_option_must_be_at_least_one(self, bad):
        df = _prepare_df(_make_multicycle_data())
        with pytest.raises(ValueError):
            MarketStructure(df, _SD, stop_after_cts_established=bad)

    def test_stop_after_one_stops_before_the_second_cycle(self):
        stopped = _run(_make_multicycle_data(), stop_after_cts_established=1)
        assert stopped.early_stop_idx is not None
        assert len(_cts_est(stopped.events)) >= 1
        assert max(int(ev.idx) for ev in stopped.events) < 10

    def test_stop_is_in_addition_to_the_bound(self):
        """`end_idx` still applies (plan §3.4). Bound past the stop point: same result as the
        stop alone. Bound before the 2nd CTS: the option is inert and the run is the plain
        bounded run (`early_stop_idx` None, nothing past the bound)."""
        raw = _make_multicycle_data()
        alone = _run(raw, stop_after_cts_established=2)
        both = _run(raw, end_idx=14, stop_after_cts_established=2)
        assert both.early_stop_idx == alone.early_stop_idx == 11
        assert _sig(both.events) == _sig(alone.events)

        inert = _run(raw, end_idx=9, stop_after_cts_established=2)
        plain = _run(raw, end_idx=9)
        assert inert.early_stop_idx is None
        assert len(_cts_est(inert.events)) == 1
        assert _sig(inert.events) == _sig(plain.events)
        assert max(int(ev.idx) for ev in inert.events) <= 9

    def test_early_stop_debug_line(self, capsys):
        raw = _make_multicycle_data()
        df = _prepare_df(raw)
        ms = _make_market_structure(
            df, struct_direction=_SD, start_idx=_START, structure_id=0, timeframe="H1",
            pip_size=_pip_size_from_pair(df), stop_after_cts_established=2,
        )
        ms.debug = True
        ms.run()
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.startswith("[EARLY_STOP]")]
        assert lines == [f"[EARLY_STOP] i={ms.early_stop_idx} cts_established>=2"]


    def test_stop_on_the_last_in_bound_step_is_not_an_early_stop(self):
        """The `i <= effective_end` guard (a refinement of the plan's §3.1.3 snippet): with
        `end_idx=10` the 2nd CTS is established by the last in-bound step (apply 10 ->
        `next_i` 11 > 10) — nothing is pre-empted, so `early_stop_idx` stays None and no
        `[EARLY_STOP]` line is printed although two CTS_ESTABLISHED exist ("stopped early"
        must be read from `early_stop_idx`, never from the count). With `end_idx=11` the
        stop fires at 11 == effective_end (one in-bound anchor pre-empted)."""
        raw = _make_multicycle_data()
        at_edge = _run(raw, end_idx=10, stop_after_cts_established=2)
        assert len(_cts_est(at_edge.events)) == 2
        assert at_edge.early_stop_idx is None
        assert _sig(at_edge.events) == _sig(_run(raw, end_idx=10).events)

        one_past = _run(raw, end_idx=11, stop_after_cts_established=2)
        assert one_past.early_stop_idx == 11 == one_past._effective_end
        assert _sig(one_past.events) == _sig(at_edge.events)

    def test_helper_quiescence_gate(self):
        """`_should_stop_after_cts` in isolation, on a stub event list: True only when the
        option is set, the count is reached, and no watch / pending reversal is open."""
        ms = _make(_make_multicycle_data(), stop_after_cts_established=2)
        est = lambda i: make_cts_established(cts_anchor_idx=i, confirmed_at=i, price=0.6,
                                             cycle_id=None, struct_direction=None)
        ms.events = [est(2), est(10)]
        assert ms._should_stop_after_cts() is True
        ms.state.reversal_watch_active = True
        assert ms._should_stop_after_cts() is False
        ms.state.reversal_watch_active = False
        ms.state.pending_reversal_apply_idx = 12
        assert ms._should_stop_after_cts() is False
        ms.state.pending_reversal_apply_idx = None
        ms.events = [est(2)]
        assert ms._should_stop_after_cts() is False
        ms.events = [est(2), est(10)]
        ms.stop_after_cts_established = None
        assert ms._should_stop_after_cts() is False


# ---------------------------------------------------------------------------
# §4.1 quiescence — a watch open across the 2nd CTS delays the stop
# ---------------------------------------------------------------------------

def _R(o, h, l, c):
    return {"o": round(o, 5), "h": round(h, 5), "l": round(l, 5), "c": round(c, 5)}


def _make_watch_over_second_cts_data() -> list[dict]:
    """Plan B §4.1 "quiescence" fixture (crafted + independently verified 2026-09-20; sd=+1,
    start 0, 18 H1 candles, NZD_USD). Candles 0-8 = `_make_multicycle_data()[:9]` (cycle 0
    established at 2, CTS_CONFIRMED at 7, range hi .6122 / lo .6088, climb 8 inside it).

    9   big bull maru closing .6145 ABOVE range_hi = c0 of `one_maru_opposite(+1)`; its high
        .6147 is the cycle-1 CTS extreme (the CTS anchor 9).
    10  SMALL bearish normal (cand2 valid) -> OMO SUCCESS, apply 10 -> 2nd CTS_ESTABLISHED
        (anchor 9, confirmed_at = idx 10 — a natural anchor != moment case), BOS_1 = l7 .6088. Bearish +
        `is_range` (candle 12 closes inside its range) -> `_post_apply_range_check(10)`
        back-fills 10..11 INSIDE THE SAME `_step_anchor(9)` call, AFTER the BOS_1 write.
    11  big bear maru closing .6060 < BOS_1 .6088 -> during that back-fill: BOS close-break
        -> REVERSAL_WATCH_START (expires 16) + REVERSAL_CANDIDATE anchor 11 apply 14
        (`one_maru_opposite(-1)` FAIL_NEEDS_CONFIRM, threshold min(l11, l12) = .6040).
        The step returns next_i = 11 with the watch OPEN and a pending reversal at 14.
    12  bull pinbar (gapped open .6100, close .6140 inside candle 10's range -> range
        confirmed at 12; h .6146 >= mid(11) so the OMO(-1) cand2 fails; not a maru).
    13  small bull normal (skipped by the -1 confirmation scan).
    14  bear maru closing .6035 <= .6040 -> confirms the reversal -> applied at 14 (terminal),
        watch cleared. Quiescent from here: the stop lands at the next anchor, 15.
    15-17 bull fillers.

    Why the watch must open AFTER the BOS_1 write within the same step: a cycle established
    while a watch is already open always moves `bos_threshold` (BOS_1 = the pullback
    extreme, below the frozen BOS) and trips MS invariant 4 ("bos_threshold changed during
    reversal watch") — see GOTCHAS "A cycle cannot be established inside an open reversal
    watch". `_post_apply_range_check` is the only path that processes later candles inside
    the establishing step.
    """
    rows = list(_make_multicycle_data()[:9])
    rows += [
        _R(0.61150, 0.61470, 0.61130, 0.61450),   # 9
        _R(0.61440, 0.61460, 0.61360, 0.61380),   # 10
        _R(0.61380, 0.61400, 0.60580, 0.60600),   # 11
        _R(0.61000, 0.61460, 0.60400, 0.61400),   # 12
        _R(0.61400, 0.61460, 0.61380, 0.61440),   # 13
        _R(0.61440, 0.61460, 0.60330, 0.60350),   # 14
        _R(0.60350, 0.60430, 0.60330, 0.60410),   # 15
        _R(0.60410, 0.60490, 0.60390, 0.60470),   # 16
        _R(0.60470, 0.60550, 0.60450, 0.60530),   # 17
    ]
    return rows


class TestQuiescence:
    def test_fixture_precondition_watch_spans_the_second_cts(self):
        """Unbounded run: the 2nd CTS (idx 9, moment 10) is emitted by the step anchored at 9,
        which returns with the watch open (started 11, expires 16) and a pending reversal at
        14; the reversal applies at 14; no rewind."""
        full, log = _run_logged(_make_watch_over_second_cts_data())
        assert [(ef.cts_anchor_idx(e), int(e.meta["confirmed_at"])) for e in _cts_est(full.events)] == [(2, 2), (9, 10)]
        ws = [(int(e.idx), int(e.meta["expires_idx"])) for e in full.events if e.type == "REVERSAL_WATCH_START"]
        assert ws == [(11, 16)]
        cands = [(int(e.idx), int(e.meta["apply_idx"])) for e in full.events if e.type == "REVERSAL_CANDIDATE"]
        assert cands == [(11, 14)]
        rev = full.df.index[full.df["market_state"].astype(str) == "reversal"]
        assert int(rev.min()) == 14
        # The step that emitted the 2nd CTS returned NON-quiescent.
        second = next(entry for entry in log if entry[2] == 2)
        assert second == (9, 11, 2, True, 14)
        assert full.early_stop_idx is None

    def test_stop_waits_for_the_watch_to_resolve(self):
        """With the option: no stop at 11 (watch open + pending reversal); the reversal
        applies at 14; the stop lands at 15 >= the resolution candle. The event list equals
        the unbounded run's (which itself ends at the reversal) — the assertions are on
        `early_stop_idx` and the watch-start ordering, not on a shorter list."""
        raw = _make_watch_over_second_cts_data()
        full = _run(raw)
        stopped, log = _run_logged(raw, stop_after_cts_established=2)
        assert [e[:2] for e in log] == [(0, 3), (3, 5), (5, 6), (6, 8), (8, 9), (9, 11), (11, 15)]
        assert stopped.early_stop_idx == 15
        assert stopped.early_stop_idx > 11            # the immediate (non-quiescent) point
        assert stopped.early_stop_idx >= 14           # the watch's resolution candle (apply)
        assert any(e.type == "REVERSAL_WATCH_START" and int(e.idx) < stopped.early_stop_idx
                   for e in stopped.events)
        assert _sig(stopped.events) == _sig(full.events)
        assert int(stopped.df.index[stopped.df["market_state"].astype(str) == "reversal"].min()) == 14
        # Bounded past the resolution: same stop.
        assert _run(raw, end_idx=17, stop_after_cts_established=2).early_stop_idx == 15

    def test_naive_stop_would_have_stopped_inside_the_open_watch(self):
        """The control: ignoring the quiescence gate stops at 11 with the watch still open and
        a reversal pending — the event list a pending rewind/reversal was about to change."""
        ms = _make(_make_watch_over_second_cts_data(), stop_after_cts_established=2)
        ms._should_stop_after_cts = lambda: sum(1 for e in ms.events if e.type == "CTS_ESTABLISHED") >= 2
        ms.run()
        assert ms.early_stop_idx == 11
        assert ms.state.reversal_watch_active is True
        assert ms.state.pending_reversal_apply_idx == 14
        assert not (ms.df["market_state"].astype(str) == "reversal").any()

    def test_bound_inside_the_watch_window(self):
        """`end_idx=13`: the reversal would apply at 14 > the clamped expiry 13, so it is never
        scheduled — the anchor fails at once (`rv_anchor_failed`), the state is quiescent when
        the 2nd-CTS step returns, and the stop lands at 11 (a prefix of the plain bounded run).
        `end_idx=14`: anchor 11 is processed as an ANCHOR (the stop rightly did not fire at
        11), so the reversal is applied as the anchor's winner at the edge (LANDMINES L4: the
        false-break discard applies only on the pending-apply path) — next_i 15 > 14, nothing
        pre-empted, `early_stop_idx` None, identical to the plain bounded run."""
        raw = _make_watch_over_second_cts_data()
        b13 = _run(raw, end_idx=13, stop_after_cts_established=2)
        assert b13.early_stop_idx == 11
        assert [int(e.meta["expires_idx"]) for e in b13.events if e.type == "REVERSAL_WATCH_START"] == [13]
        assert [e for e in b13.events if e.type == "REVERSAL_CANDIDATE"] == []
        assert not (b13.df["market_state"].astype(str) == "reversal").any()
        plain13 = _run(raw, end_idx=13)
        assert _sig(plain13.events)[: len(b13.events)] == _sig(b13.events)

        b14 = _run(raw, end_idx=14, stop_after_cts_established=2)
        assert b14.early_stop_idx is None
        assert int(b14.df.index[b14.df["market_state"].astype(str) == "reversal"].min()) == 14
        assert _sig(b14.events) == _sig(_run(raw, end_idx=14).events)

    def test_phase2_probe_finalizes_at_the_moment_not_the_anchor(self):
        """End-to-end Plan B §4.3 on a natural fixture: the 2nd CTS has idx 9 but moment 10;
        the probe (early stop at 15) finalizes `second_cts_reached` at 10, and equals the
        classify-at-exit result."""
        df = _prepare_df(_make_watch_over_second_cts_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), 17, "H1", enable_phase2=True)
        assert res.finalize_condition == "second_cts_reached"
        assert res.finalize_idx == 10
        assert res.starting_idx == 0
        on, off = _phase2_pair(df, 17)
        assert on == off
        assert on.finalize_idx == 10


# ---------------------------------------------------------------------------
# §1/§2 rebuilt-prefix exception — the one mechanism under which early stop and
# classify-at-exit can differ (reproduced 2026-09-20; no instance on the reference window;
# no known instance since 2026-09-29 — see `TestRebuiltPrefixException`)
# ---------------------------------------------------------------------------

def _make_double_rewind_data() -> list[dict]:
    """Two expiry-rewinds: J1 BEFORE the 2nd CTS and J2 AFTER the stop point (sd=+1, start 0,
    20 H1 candles; crafted during the Plan B cold review, 2026-09-20).

    0-2  BOS_0 .5998; CTS_0 .6042 at 2.
    3-4  pullback -> CTS_CONFIRMED@4; candle 4 also close-breaks BOS_0 -> watch (expires 9),
         `one_maru_opposite(-1)` FAIL_NEEDS_CONFIRM, confirmed exactly at 9 == expiry.
    6-8  breakout -> cycle 1 established at 8 (FIRST PASS) — INSIDE the open watch: since
         2026-09-29 it ENDS the watch (MARKET_STRUCTURE_SPEC "A new cycle ends an open watch"),
         so there is no J1: cycles at 2 / 8 / 12, the reversal at 17; the rest of this
         docstring is the pre-2026-09-29 path (`tests/test_ms_new_cycle_ends_watch.py`).
    9    bearish normal closing <= .5978 -> apply == expiry -> J1 expiry-rewind, jump_to 5,
         seed restore puts the RANGE state on candle 5 -> after J1 cycle 1 is re-established
         only at 12 (anchor 11). Post-J1 truth: cts_est = [(2,2), (12,12)].
    13-14 pullback -> CTS_CONFIRMED; candle 14 close-breaks the new BOS -> J2 watch (expires
         min(19, B)); `one_maru_opposite(-1)` FAIL_NEEDS_CONFIRM, threshold .5895.
    17   down maru closing <= .5895 -> confirms at 17. With `end_idx=17` the apply equals the
         clamped expiry -> J2 expiry-rewind, jump_to 15 -> `_rewind_to` replays from 0
         IGNORING J1 (LANDMINES "MarketStructure Deep-Couples…" point 1) -> the rebuilt prefix
         has cycle 1 at 8 AGAIN: cts_est = [(2,2), (8,8), (12,12)].
    With `end_idx` 13 / 16 / 18 the J2 apply is unscheduled / pending / a reversal, no second
    rewind happens, and early stop == classify-at-exit. Since F3b (2026-09-29) the J2 pending
    applies AT the bound 17 too (a reversal) — no rewind at any bound.
    """
    return [
        _R(0.60300, 0.60320, 0.59980, 0.60000),   # 0
        _R(0.60000, 0.60220, 0.59990, 0.60200),   # 1
        _R(0.60200, 0.60420, 0.60190, 0.60400),   # 2
        _R(0.60400, 0.60410, 0.60130, 0.60150),   # 3
        _R(0.60150, 0.60160, 0.59800, 0.59820),   # 4
        _R(0.59820, 0.59960, 0.59780, 0.59900),   # 5
        _R(0.59900, 0.60220, 0.59880, 0.60200),   # 6
        _R(0.60200, 0.60560, 0.60180, 0.60540),   # 7
        _R(0.60540, 0.60660, 0.60520, 0.60640),   # 8
        _R(0.60640, 0.60660, 0.59180, 0.59700),   # 9
        _R(0.59700, 0.60120, 0.59680, 0.60100),   # 10
        _R(0.60100, 0.60720, 0.60080, 0.60700),   # 11
        _R(0.60700, 0.60920, 0.60680, 0.60900),   # 12
        _R(0.60900, 0.60920, 0.60380, 0.60400),   # 13
        _R(0.60400, 0.60420, 0.58980, 0.59000),   # 14
        _R(0.59000, 0.59150, 0.58950, 0.59100),   # 15
        _R(0.59100, 0.59320, 0.59080, 0.59300),   # 16
        _R(0.59300, 0.59320, 0.58780, 0.58800),   # 17
        _R(0.58800, 0.59020, 0.58780, 0.59000),   # 18
        _R(0.59000, 0.59220, 0.58980, 0.59200),   # 19
    ]


def _phase2_pair(df, end_idx):
    """(`_run_phase2` as shipped, `_run_phase2` with the stop popped) on `df`."""
    import engine_v2.structure.unified_probe as up
    ref = _ref_zone_uptrend()
    pip = 0.0001
    kw = dict(reset_tol=DEFAULT_PROBE_RESET_PIPS["H1"] * pip,
              wick_cap=DEFAULT_PROBE_RESET_WICK["H1"] * pip, max_iterations=10)
    on = _run_phase2(df, 0, ref.inner, ref.outer, 1, ref, end_idx, **kw)
    orig = up._make_market_structure
    try:
        up._make_market_structure = lambda d, **k: orig(
            d, **{a: b for a, b in k.items() if a != "stop_after_cts_established"})
        off = _run_phase2(df, 0, ref.inner, ref.outer, 1, ref, end_idx, **kw)
    finally:
        up._make_market_structure = orig
    return on, off


class TestRebuiltPrefixException:
    def test_mechanism(self):
        """B=17. Until 2026-09-29 the full run rewound twice (J1 to 5, J2 to 15) and its rebuilt
        prefix — `_rewind_to` replaying from 0 IGNORING J1 — held THREE CTS_ESTABLISHED
        [(2,2),(8,8),(12,12)] while the stopped run stopped at 13 on the post-J1 prefix
        [(2,2),(12,12)]: classify-at-exit (finalize 8) != early stop (finalize 12), pinned by a
        strict xfail. J1 existed only because cycle 1, established at 8 inside the watch opened
        at 4, left that watch open; a new cycle now ENDS the watch (MARKET_STRUCTURE_SPEC "A new
        cycle ends an open watch"), so the full run rewinds once (15, after the stop point), the
        stopped run stops at 9 with [(2,2),(8,8)] — an exact prefix — and the two agree at every
        bound (finalize 8). No instance of the exception is known under the new rule (0 in 36k
        random tails; this fixture with candle 8 kept below the CTS, no cycle there, rewinds twice
        but its prefixes agree). Since F3b (2026-09-29) no expiry fires — a pending reversal
        confirming ON its watch's expiry candle applies — and the expiry rewind was removed, so the
        exception is unreachable: the full run's watch 14 (expires at the bound 17 == its pending
        apply) REVERSES at 17 (before: a rewind to 15); the prefixes agree. Kept as the
        prefix-agreement pin."""
        raw = _make_double_rewind_data()
        full = _make(raw, end_idx=17)
        full.run()
        stopped = _make(raw, end_idx=17, stop_after_cts_established=2)
        stopped.run()
        assert [int(e.idx) for e in full.events
                if e.type == "STATE_CHANGED" and e.meta["to"] == "reversal"] == [17]
        assert stopped.early_stop_idx == 9
        assert [(int(e.idx), int(e.meta["confirmed_at"])) for e in _cts_est(full.events)] == [(2, 2), (8, 8), (12, 12)]
        assert [(int(e.idx), int(e.meta["confirmed_at"])) for e in _cts_est(stopped.events)] == [(2, 2), (8, 8)]
        assert _sig(full.events)[: len(stopped.events)] == _sig(stopped.events)

        df = _prepare_df(raw)
        for B in (13, 16, 17, 18):
            on_b, off_b = _phase2_pair(df, B)
            assert on_b == off_b, B
            assert (on_b.finalize_condition, on_b.finalize_idx) == ("second_cts_reached", 8), B


def test_run_with_start_past_the_frame_returns_empty_levels():
    """`run()`'s start-past-the-frame early return used `self.levels`, an
    attribute that never existed (AttributeError; latent, never reached on the
    window). Plan E E1b: it builds the levels like the normal path does — from
    the (empty) events."""
    df = _prepare_df(_make_multicycle_data())
    out_df, events, levels = MarketStructure(df, 1, start_idx=len(df)).run()
    assert len(out_df) == len(df)
    assert events == [] and levels == []
