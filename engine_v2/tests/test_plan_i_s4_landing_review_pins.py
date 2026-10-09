"""PLAN_I S3b + S4 landing review (2026-10-08), lens B's pins for the mutants that survived the full suite: D08 (main's
CTS_THRESHOLD_UPDATED dispatch — I2 through the downstream at the H1 timeframe), H03 (every main sid's limit), A05 / A06
(the previous single's obsolete stamp; a re-activated `cross_failed` single never ends its own cycle), I02 / I05 (I8 at
the reactivation and on a main sid 1), W01 (an update after the confirmation never moves the locked cross).

Fixtures: the S3 pin file's flat frame (its own mirror at the pivot 3.0 — sd = -1 is the exact mirror)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.tests.test_plan_i_s3_cross_fib_limit_applied import (
    _conf, _df, _est, _ev, _gap, _kind, _lim, _m, _q, _three_cycles, _tracker,
)
from engine_v2.tests._cross_fib_limits import NO_P


def _upd(idx, price, cyc, sd):
    return _ev("CTS_UPDATED", idx, price, cyc, sd, via=CTS_UPDATED_RAW_VIA)


# ---------------------------------------------------------------------------------------------------------------------
# A05: the obsoleted previous single carries `obsolete_reason: new_cycle` (a fib-CSV `meta` cell)
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_the_previous_single_is_stamped_obsolete(sd):
    t = _three_cycles(None, sd, upto=1)                                    # no P: cycle 1 a single
    assert _kind(t, 1) == "single cross_fib_limit"
    assert t._fibs[(0, 0)].meta["obsolete_reason"] == "new_cycle"
    assert "obsolete_reason" not in t._fibs[(0, 1)].meta


# ---------------------------------------------------------------------------------------------------------------------
# A06: a `cross_failed` single re-created (inactive → active via `_activate_fib`) never ends its OWN cycle
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_a_reactivated_cross_failed_single_never_terminates_its_own_cycle(sd):
    """The F5 shape (cycle 0 dead → cycle 1's `cross_failed` single; the update to 48 fills its window → deactivated),
    then a new gap at 52: the update to 56 fails the cross again → the inactive single is re-activated through
    `_activate_fib`. Cycle 1 must stay open (no `new_cycle` terminal on itself)."""
    t = _tracker(_lim(9.0, sd))
    df = _df(80, [_gap(15, sd, 1.48, 1.44), _gap(35, sd, 1.30, 1.20), _gap(52, sd, 1.40, 1.30)])
    col, lvl = ("l", lambda v: v) if sd == 1 else ("h", lambda v: _m(v, -1))
    df.loc[22:, col] = lvl(1.43)                                           # fills cycle 0's gap only
    _q(t.on_cts_established, _est(20, 1.15, 0, sd), df, bos_idx=10, bos_price=_m(1.0, sd))
    _q(t.on_cts_confirmed, _conf(25, 1.15, 0, sd, 20))
    _q(t.on_cts_established, _est(40, 1.35, 1, sd), df, bos_idx=30, bos_price=_m(1.2, sd))
    assert _kind(t, 1) == "single cross_failed"
    df.loc[45:49, col] = lvl(1.19)                                         # fills cycle 1's gap 35
    _q(t.on_cts_updated, _upd(48, 1.37, 1, sd), df)
    assert _kind(t, 1) is None
    _q(t.on_cts_updated, _upd(56, 1.45, 1, sd), df)                        # gap 52 (formed 53) unfilled
    assert _kind(t, 1) == "single cross_failed"
    assert (0, 1) not in t._terminal and t._terminal[(0, 0)] == (40, "new_cycle")


# ---------------------------------------------------------------------------------------------------------------------
# I02: I8 at the single's reactivation inside `_update_fib_cts`
# ---------------------------------------------------------------------------------------------------------------------

def test_i8_a_single_is_never_reactivated_over_an_active_cross():
    t = _three_cycles(9.0, 1, upto=1)                                      # cycle 1: cross (0->1) active
    key, cross = t._get_latest_cross(0, 1)
    t._fibs[(0, 1)] = replace(cross, active=False, meta={"activated_at": 40})   # a superseded single, inactive
    df = _df(80, [_gap(35, 1, 1.30, 1.20)])
    t._in_event, t._evaluated_at = True, 42
    with pytest.raises(AssertionError, match="I8: sid=0 cycle=1 single activated while cross"):
        _q(t._update_fib_cts, (0, 1), 42, 1.37, df)


# ---------------------------------------------------------------------------------------------------------------------
# I05: I8 holds on a MAIN structure (sid >= 1) too — no gate left after the mode's removal
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_i8_holds_on_a_main_sid_1(sd):
    from engine_v2.tests.test_plan_i_s4_main_one_path import _est as _est_s4, _ev as _ev_s4, _main_tracker
    t = _main_tracker(9.0, sd)
    df = _df(120, [_gap(65, sd, 1.30, 1.20), _gap(78, sd, 1.40, 1.30)])
    if sd == 1:
        df.at[77, "l"] = 1.45
    else:
        df.at[77, "h"] = _m(1.45, -1)
    _q(t.on_cts_established, _est_s4(70, 1.35, 1, 0, sd), df, 60, _m(1.2, sd))
    _q(t.on_cts_confirmed, _ev_s4("CTS_CONFIRMED", 75, 1.35, 1, 0, sd, cts_anchor_idx=70))
    _q(t.on_cts_threshold_updated, _ev_s4("CTS_THRESHOLD_UPDATED", 80, 1.52, 1, 0, sd), df)
    assert t._get_latest_cross(1, 1)[1].active
    with pytest.raises(AssertionError, match="I8: sid=1 cycle=1 single activated while cross"):
        _q(t._activate_fib, sid=1, cycle_id=1, sd=sd, bos_idx=77, bos_price=_m(1.45, sd), cts_idx=80,
           cts_price=_m(1.52, sd), meta={"activated_at": 80})


# ---------------------------------------------------------------------------------------------------------------------
# W01: an update of a CONFIRMED cross cycle (the F16 order) never moves its locked cross
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_an_update_after_the_confirmation_never_moves_the_locked_cross(sd):
    t = _three_cycles(9.0, sd, upto=1)                                     # cycle 1: cross (0->1) at 40
    _q(t.on_cts_confirmed, _conf(45, 1.35, 1, sd, 40))
    key, locked = t._get_latest_cross(0, 1)
    assert locked.locked and locked.cts_idx == 40
    df = _df(80, [_gap(15, sd, 1.10, 1.00), _gap(35, sd, 1.30, 1.20)])
    assert _q(t.on_cts_updated, _upd(47, 1.37, 1, sd), df) is None
    assert t._fibs[key] == locked


# ---------------------------------------------------------------------------------------------------------------------
# D08: the orchestrator dispatches CTS_THRESHOLD_UPDATED on the H1 (main) run too — main's pre-established cross (I2)
# ---------------------------------------------------------------------------------------------------------------------

def test_the_h1_downstream_drives_the_pre_established_cross():
    """`test_fib_never_established_cap`'s climb (cycle 0 confirmed at 7, a threshold update on 9-14 and 16-22, cycle 1
    never established) through the downstream at its DEFAULT timeframe (H1, the main's call) with a reference price no
    anchor reaches: the cycle-1 cross is pre-created from the threshold updates exactly as on a sub. (A dispatch gated to
    the sub timeframes — the retired "no-op in h1 mode" — leaves main without I2.)"""
    import contextlib
    import io

    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests._cross_fib_limits import unreached
    from engine_v2.tests.test_fib_never_established_cap import _CAP, _rows
    from engine_v2.tests.test_unified_probe import _prepare_df

    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_rows()[:_CAP + 1]), 0, 1, timeframe="M15", cross_fib_limit=NO_P)
        out = _run_downstream_pipeline(res.df, res.events, 1, wvmi="off", cross_fib_limits={0: unreached(1)})
    assert [e for e in res.events if e.type == "CTS_THRESHOLD_UPDATED"]
    fib1 = [f for f in out["fib_states"] if (f.structure_id, f.cycle_id) == (0, 1)]
    assert fib1 and all(f.meta.get("cross_start_cycle") == 0 for f in fib1)


# ---------------------------------------------------------------------------------------------------------------------
# H03: the H1 call hands EVERY main sid its limit (the S2 spy's fixture has sid 0 only)
# ---------------------------------------------------------------------------------------------------------------------

def test_run_pipeline_hands_every_main_sid_its_limit(monkeypatch):
    """`run_pipeline` with the market structure stubbed to `test_plan_i_s2_cross_fib_limit._main_events()` (sids 0 /
    1 / 2; sid 0 reverses into sid 1): the downstream receives the limits of all three sids, sid 1's P = 0.5736."""
    import contextlib
    import io
    from types import SimpleNamespace

    import engine_v2.pipeline.orchestrator as orch
    import engine_v2.tests.test_render_sub_projection as RSP
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import _main_events
    from engine_v2.zones.cross_cycle_fib import CrossFibLimit
    from engine_v2.zones.structure_lifecycle import compute_main_cross_fib_limits

    class _Stop(Exception):
        pass

    seen = {}

    def spy(df, events, sd, **kw):
        seen.update(lim=kw.get("cross_fib_limits"), events=events)
        raise _Stop

    monkeypatch.setattr(orch, "compute_structure",
                        lambda d: SimpleNamespace(df=d, events=_main_events(), struct_direction=-1, notes={}))
    monkeypatch.setattr(orch, "_run_downstream_pipeline", spy)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_Stop):
        orch.run_pipeline(RSP._make_raw_df(RSP._make_flat_prefix() + RSP._make_reversing_data()))
    assert seen["lim"] == compute_main_cross_fib_limits(seen["events"])
    assert set(seen["lim"]) == {0, 1, 2}
    assert seen["lim"][1] == CrossFibLimit(0.5736, "reversal_bos_frozen", started_by="reversal",
                                           ref_anchor_price=0.5731, break_pattern_anchor_idx=897, break_confirmed_idx=902)
