"""PLAN_I S3 (§9.3 D1, §1.4; R6-1 (d), R7-3): the cross-fib LIMIT applied on the sub path (`cross_cycle` mode).

A cross may target cycle T ≥ 1 only if the structure has a reference price `P` and no cycle k < T has a latest CTS
anchor STRICTLY beyond `P` — the first cycle beyond (`M`) is the LAST cycle a cross may target. A cycle past it (and
every cycle ≥ 1 of a structure without `P`) gets the cycle-0 single: activated at its establishment if its own
[BOS, CTS] holds an unfilled sd imbalance, else late on a CTS_UPDATED that finds one; labelled `via:
"cross_fib_limit"`; no pre-established cross. A tracker consulted without a limit entry raises `CrossFibLimitError`.
I8: a single and a cross are never both active on one cycle.

Fixture: the flat frame (o = c = 1.5, h = 1.501, l = 1.499 — its own mirror at the pivot 3.0, so `sd = -1` is the exact
mirror of `sd = +1`). Three cycles, each with an unfilled gap in its own [BOS, CTS]: CTS_0 1.15 (anchor 20), CTS_1 1.35
(40), CTS_2 1.55 (60) for a buy structure — without a limit cycle 1 crosses (0→1) and cycle 2 (0→2).
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace

import pandas as pd
import pytest

from engine_v2.common.types import ImbalanceInstance
from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.zones.cross_cycle_fib import CrossFibLimit
from engine_v2.zones.fib_tracker import CrossFibLimitError, FibTracker, FibTrackerConfig

PIV = 3.0


def _m(p, sd):
    return p if sd == 1 else round(PIV - p, 6)


def _gap(start, sd, top, bottom):
    if sd == 1:
        return ImbalanceInstance(start, start, 1, top, bottom, round(top - bottom, 6))
    return ImbalanceInstance(start, start, -1, _m(bottom, -1), _m(top, -1), round(top - bottom, 6))


def _df(n, gaps):
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "o": [1.5] * n, "h": [1.501] * n, "l": [1.499] * n, "c": [1.5] * n,
        "direction": [0] * n, "is_imbalance": [0] * n,
    })
    df.attrs["imbalances"] = list(gaps)
    return df


def _lim(p, sd):
    return CrossFibLimit(None, "none") if p is None else CrossFibLimit(_m(p, sd), "parent_cts_threshold")


def _tracker(limit):
    limits = None if limit is ... else {0: limit}
    return FibTracker(FibTrackerConfig(fill_threshold=0.70), fib_mode="cross_cycle", cross_fib_limits=limits)


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _est(moment, price, cyc, sd):
    return make_cts_established(cts_anchor_idx=moment, confirmed_at=moment, price=_m(price, sd), structure_id=0,
                                cycle_id=cyc, struct_direction=sd)


def _ev(etype, idx, price, cyc, sd, **meta):
    return StructureEvent(idx=idx, category="STRUCTURE", type=etype, price=_m(price, sd),
                          meta=dict(structure_id=0, cycle_id=cyc, struct_direction=sd, **meta))


def _conf(idx, price, cyc, sd, anchor):
    return _ev("CTS_CONFIRMED", idx, price, cyc, sd, cts_anchor_idx=anchor)


_CYCLES = [  # (BOS idx, BOS price, CTS anchor = moment, CTS price, confirmation idx, own gap)
    (10, 1.00, 20, 1.15, 25, (15, 1.10, 1.00)),
    (30, 1.20, 40, 1.35, 45, (35, 1.30, 1.20)),
    (50, 1.40, 60, 1.55, None, (55, 1.50, 1.40)),
]


def _three_cycles(p, sd, upto=2):
    t = _tracker(_lim(p, sd))
    df = _df(80, [_gap(g[0], sd, g[1], g[2]) for *_, g in _CYCLES])
    for c, (bos, bos_p, cts, cts_p, conf, _g) in enumerate(_CYCLES[:upto + 1]):
        _q(t.on_cts_established, _est(cts, cts_p, c, sd), df, bos_idx=bos, bos_price=_m(bos_p, sd))
        if conf is not None and c < upto:
            _q(t.on_cts_confirmed, _conf(conf, cts_p, c, sd, cts))
    return t


def _kind(t, cyc):
    """'cross k->c' (the active latest version), 'single <via>', or None."""
    latest = t._get_latest_cross(0, cyc)
    single = t._fibs.get((0, cyc))
    assert not (latest is not None and latest[1].active and single is not None and single.active)   # I8
    if latest is not None and latest[1].active:
        return f"cross {latest[1].meta['cross_start_cycle']}->{cyc}"
    if single is not None and single.active:
        return f"single {single.meta.get('via')}"
    return None


# ---------------------------------------------------------------------------------------------------------------------
# The rule: M = the first cycle strictly beyond P = the last cross target
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("p, expected", [
    (None, ["single cross_fib_limit", "single cross_fib_limit"]),        # no P: every cycle >= 1 single
    (1.10, ["single cross_fib_limit", "single cross_fib_limit"]),        # CTS_0 1.15 beyond -> M = 0
    (1.15, ["cross 0->1", "single cross_fib_limit"]),                    # CTS_0 AT P is not beyond; CTS_1 is -> M = 1
    (1.20, ["cross 0->1", "single cross_fib_limit"]),                    # M = 1
    (1.40, ["cross 0->1", "cross 0->2"]),                                # CTS_2 1.55 beyond -> M = 2 (target 2 allowed)
    (9.00, ["cross 0->1", "cross 0->2"]),                                # never reached
])
def test_the_limit_decides_cross_or_single_per_cycle(sd, p, expected):
    t = _three_cycles(p, sd)
    assert [_kind(t, 1), _kind(t, 2)] == expected
    assert _kind(t, 0) == "single None"                                  # cycle 0: the single, never labelled
    for c in (1, 2):
        s = t._fibs.get((0, c))
        if s is not None and s.meta.get("via") == "cross_fib_limit":
            bos, bos_p, cts, cts_p, _conf_idx, _g = _CYCLES[c]
            assert (s.bos_idx, s.bos_price, s.cts_idx, s.cts_price) == (bos, _m(bos_p, sd), cts, _m(cts_p, sd))
            assert s.meta["activated_at"] == cts and "activated_on" not in s.meta


@pytest.mark.parametrize("sd", [1, -1])
def test_cross_allowed_reads_the_latest_anchor_strictly(sd):
    t = _tracker(_lim(1.20, sd))
    t._latest_anchor = {(0, 0): (20, _m(1.20, sd), sd), (0, 1): (40, _m(1.21, sd), sd)}
    assert t._cross_allowed(0, 1) and not t._cross_allowed(0, 2)        # CTS_0 AT P; CTS_1 beyond -> M = 1
    t._latest_anchor[(0, 0)] = (22, _m(1.2001, sd), sd)                  # cycle 0's latest anchor moved beyond
    assert not t._cross_allowed(0, 1)                                    # M = 0


def test_no_price_allows_no_cross_and_a_missing_entry_raises():
    assert not _tracker(_lim(None, 1))._cross_allowed(0, 1)
    with pytest.raises(CrossFibLimitError, match="sid=0 cycle=1: no cross_fib_limit entry"):
        _tracker(...)._cross_allowed(0, 1)
    assert issubclass(CrossFibLimitError, RuntimeError)
    assert not issubclass(CrossFibLimitError, (ValueError, IndexError))   # a geometry's except never swallows it


def test_a_structure_without_an_entry_raises_at_its_first_cycle_1_event_only():
    t = _tracker(...)
    df = _df(80, [_gap(g[0], 1, g[1], g[2]) for *_, g in _CYCLES])
    _q(t.on_cts_established, _est(20, 1.15, 0, 1), df, bos_idx=10, bos_price=1.0)     # cycle 0: no consult
    _q(t.on_cts_confirmed, _conf(25, 1.15, 0, 1, 20))
    with pytest.raises(CrossFibLimitError):
        _q(t.on_cts_established, _est(40, 1.35, 1, 1), df, bos_idx=30, bos_price=1.2)


def test_the_threshold_path_raises_without_an_entry_too():
    """The pre-established check consults the limit as well (landing review M4)."""
    t = _tracker(...)
    df = _df(80, [_gap(g[0], 1, g[1], g[2]) for *_, g in _CYCLES])
    _q(t.on_cts_established, _est(20, 1.15, 0, 1), df, bos_idx=10, bos_price=1.0)
    _q(t.on_cts_confirmed, _conf(25, 1.15, 0, 1, 20))
    with pytest.raises(CrossFibLimitError, match="sid=0 cycle=1"):
        _q(t.on_cts_threshold_updated, _ev("CTS_THRESHOLD_UPDATED", 27, 1.2, 0, 1), df)


# ---------------------------------------------------------------------------------------------------------------------
# The past-limit single: the cycle-0 single's behaviour
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_the_past_limit_single_activates_late_on_an_update(sd):
    """Cycle 1 past the limit (no P) with NO gap in its own [30, 40] at its establishment → no fib; a raw
    CTS_UPDATED to 48 finds the gap at 44 (formed at 45) → the single activates there, labelled."""
    t = _tracker(_lim(None, sd))
    df = _df(80, [_gap(15, sd, 1.10, 1.00), _gap(44, sd, 1.30, 1.20)])
    _q(t.on_cts_established, _est(20, 1.15, 0, sd), df, bos_idx=10, bos_price=_m(1.0, sd))
    _q(t.on_cts_confirmed, _conf(25, 1.15, 0, sd, 20))
    assert _q(t.on_cts_established, _est(40, 1.35, 1, sd), df, bos_idx=30, bos_price=_m(1.2, sd)) is None
    assert (0, 1) not in t._fibs and t._get_latest_cross(0, 1) is None
    out = _q(t.on_cts_updated, _ev("CTS_UPDATED", 48, 1.40, 1, sd, via=CTS_UPDATED_RAW_VIA), df)
    assert out is t._fibs[(0, 1)] and out.active
    assert (out.bos_idx, out.cts_idx, out.cts_price) == (30, 48, _m(1.40, sd))
    assert out.meta == {"activated_at": 48, "fib_mode": "cross_cycle", "activated_on": "update",
                        "via": "cross_fib_limit"}
    assert t._get_latest_cross(0, 1) is None


@pytest.mark.parametrize("sd", [1, -1])
def test_the_past_limit_single_extends_on_an_update(sd):
    t = _three_cycles(1.10, sd, upto=1)                                  # M = 0: cycle 1 single at 40
    _q(t.on_cts_updated, _ev("CTS_UPDATED", 42, 1.37, 1, sd, via=CTS_UPDATED_RAW_VIA),
       _df(80, [_gap(g[0], sd, g[1], g[2]) for *_, g in _CYCLES]))
    s = t._fibs[(0, 1)]
    assert (s.cts_idx, s.cts_price, s.active, s.meta["via"]) == (42, _m(1.37, sd), True, "cross_fib_limit")
    # EXTENDED, not re-activated (landing review M5): the establishment's activation stands, one history entry more.
    assert s.meta["activated_at"] == 40 and "activated_on" not in s.meta
    assert s.cts_history == ((40, _m(1.35, sd)), (42, _m(1.37, sd)))
    assert t._get_latest_cross(0, 1) is None


# ---------------------------------------------------------------------------------------------------------------------
# No pre-established cross past the limit
# ---------------------------------------------------------------------------------------------------------------------

def _pre_established(p, sd):
    """Cycle 0 (gap 15) established at 20 (CTS 1.2), confirmed at 25 → cycle 1 pre_established; a dip at 27 makes 27
    the prospective BOS_1; a CTS_THRESHOLD_UPDATED at 30 finds the own gap 29 (formed at 30) — within the limit it
    creates the pre-established cross (0→1)."""
    t = _tracker(_lim(p, sd))
    df = _df(40, [_gap(15, sd, 1.10, 1.00), _gap(29, sd, 1.40, 1.30)])
    if sd == 1:
        df.at[27, "l"] = 1.45
    else:
        df.at[27, "h"] = _m(1.45, -1)
    _q(t.on_cts_established, _est(20, 1.2, 0, sd), df, bos_idx=10, bos_price=_m(0.9, sd))
    _q(t.on_cts_confirmed, _conf(25, 1.2, 0, sd, 20))
    assert t._m15_phase[(0, 1)] == "pre_established"
    _q(t.on_cts_threshold_updated, _ev("CTS_THRESHOLD_UPDATED", 30, 1.52, 0, sd), df)
    return t


@pytest.mark.parametrize("sd", [1, -1])
def test_the_pre_established_cross_is_refused_past_the_limit(sd):
    within = _pre_established(9.0, sd)
    latest = within._get_latest_cross(0, 1)
    assert latest is not None and latest[1].active and latest[1].meta["cross_start_cycle"] == 0
    assert _pre_established(1.1, sd)._get_latest_cross(0, 1) is None   # CTS_0 1.2 beyond -> M = 0
    assert _pre_established(None, sd)._get_latest_cross(0, 1) is None  # no P


# ---------------------------------------------------------------------------------------------------------------------
# I8: never a single and a cross both active on one cycle (both orders)
# ---------------------------------------------------------------------------------------------------------------------

def test_i8_a_single_never_activates_over_an_active_cross():
    t = _three_cycles(9.0, 1, upto=1)                                    # cycle 1: cross (0->1) active
    assert _kind(t, 1) == "cross 0->1"
    with pytest.raises(AssertionError, match="I8: sid=0 cycle=1 single activated while cross"):
        _q(t._activate_fib, sid=0, cycle_id=1, sd=1, bos_idx=30, bos_price=1.2, cts_idx=40, cts_price=1.35,
           meta={"activated_at": 40})


def test_i8_a_cross_is_never_created_over_an_active_single():
    t = _three_cycles(None, 1, upto=1)                                   # cycle 1: the single active
    assert _kind(t, 1) == "single cross_fib_limit"
    with pytest.raises(AssertionError, match="I8: sid=0 cycle=1 cross created / revived while the single"):
        _q(t._m15_create_cross, 0, 1, 1, (10, 1.0), 40, 1.35, 0, 40)


def test_i8_the_cross_failed_fallback_supersedes_the_cross_first():
    """The allowed path's single (cross fails at an update: cycle 0's gap filled) deactivates the cross BEFORE it
    activates — no I8 trip; the superseded cross stays as an inactive record."""
    t = _tracker(_lim(9.0, 1))
    df = _df(80, [_gap(15, 1, 1.48, 1.44), _gap(35, 1, 1.30, 1.20)])   # cycle 0's gap ABOVE cycle 1's
    _q(t.on_cts_established, _est(20, 1.15, 0, 1), df, bos_idx=10, bos_price=1.0)
    _q(t.on_cts_confirmed, _conf(25, 1.15, 0, 1, 20))
    _q(t.on_cts_established, _est(40, 1.35, 1, 1), df, bos_idx=30, bos_price=1.2)
    assert _kind(t, 1) == "cross 0->1"
    df.loc[41:, "l"] = 1.43                                              # fills cycle 0's gap only
    _q(t.on_cts_updated, _ev("CTS_UPDATED", 42, 1.37, 1, 1, via=CTS_UPDATED_RAW_VIA), df)
    assert _kind(t, 1) == "single cross_failed"
    assert not t._get_latest_cross(0, 1)[1].active


# =====================================================================================================================
# Landing review (2026-10-08): lens B's gap pins (mutants A13, U05, U09, T03, I04 / I05, I10) and lens A's m1 (F16)
# =====================================================================================================================

def _frame(sd):
    return _df(80, [_gap(g[0], sd, g[1], g[2]) for *_, g in _CYCLES])


# ---------------------------------------------------------------------------------------------------------------------
# U09: the update path's limit decision at the boundary target == M
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_the_cycle_at_M_keeps_its_cross_on_an_update(sd):
    """P = 1.40: CTS_2 1.55 is the first anchor beyond -> M = 2 = the last cross target. A CTS_UPDATED of cycle 2
    (62, 1.57) must re-run the cross check (extend 0->2 in place) — NOT take the past-limit single path."""
    t = _three_cycles(1.40, sd)
    assert _kind(t, 2) == "cross 0->2"
    _q(t.on_cts_updated, _ev("CTS_UPDATED", 62, 1.57, 2, sd, via=CTS_UPDATED_RAW_VIA), _frame(sd))
    assert _kind(t, 2) == "cross 0->2"
    assert (0, 2) not in t._fibs                                           # no single for the cycle at M
    assert t._get_latest_cross(0, 2)[1].cts_idx == 62                      # extended in place


# ---------------------------------------------------------------------------------------------------------------------
# I04 / I05: I8 at the in-place revival
# ---------------------------------------------------------------------------------------------------------------------

def test_i8_a_cross_is_never_revived_over_an_active_single():
    t = _three_cycles(None, 1, upto=1)                                     # cycle 1: the single active
    assert _kind(t, 1) == "single cross_fib_limit"
    with pytest.raises(AssertionError, match="I8: sid=0 cycle=1 cross created / revived while the single"):
        _q(t._m15_reactivate_cross_in_place, (0, 1, "cross", 0), t._fibs[(0, 1)], 40, 1.35, 41)


# ---------------------------------------------------------------------------------------------------------------------
# I10: I8 is about ACTIVE records — a superseded (inactive) single never blocks a cross
# ---------------------------------------------------------------------------------------------------------------------

def test_i8_an_inactive_single_does_not_block_a_cross():
    t = _three_cycles(None, 1, upto=1)
    t._fibs[(0, 1)] = replace(t._fibs[(0, 1)], active=False)
    _q(t._m15_create_cross, 0, 1, 1, (10, 1.0), 40, 1.35, 0, 40)
    latest = t._get_latest_cross(0, 1)
    assert latest is not None and latest[1].active


# ---------------------------------------------------------------------------------------------------------------------
# A13: the limit is read PER STRUCTURE (sid), not off sid 0
# ---------------------------------------------------------------------------------------------------------------------

def test_the_limit_is_read_per_structure():
    t = FibTracker(FibTrackerConfig(fill_threshold=0.70), fib_mode="cross_cycle",
                   cross_fib_limits={0: _lim(9.0, 1), 1: _lim(None, 1)})
    assert t._cross_allowed(0, 1) and not t._cross_allowed(1, 1)
    only0 = FibTracker(FibTrackerConfig(fill_threshold=0.70), fib_mode="cross_cycle",
                       cross_fib_limits={0: _lim(9.0, 1)})
    with pytest.raises(CrossFibLimitError, match="sid=1 cycle=1: no cross_fib_limit entry"):
        only0._cross_allowed(1, 1)


# ---------------------------------------------------------------------------------------------------------------------
# U08 / T03: the limit is consulted only where a cross could be made (cycle >= 1, threshold path in phase)
# ---------------------------------------------------------------------------------------------------------------------

def test_cycle_0_updates_and_an_out_of_phase_threshold_never_consult_the_limit():
    """A tracker WITHOUT an entry: cycle 0's establishment, a CTS_UPDATED of cycle 0 and a CTS_THRESHOLD_UPDATED
    whose target (cycle 1) is not pre_established (cycle 0 not yet confirmed) must not raise."""
    t = _tracker(...)
    df = _frame(1)
    _q(t.on_cts_established, _est(20, 1.15, 0, 1), df, bos_idx=10, bos_price=1.0)
    _q(t.on_cts_updated, _ev("CTS_UPDATED", 22, 1.17, 0, 1, via=CTS_UPDATED_RAW_VIA), df)          # U08
    assert t._m15_phase.get((0, 1)) is None
    _q(t.on_cts_threshold_updated, _ev("CTS_THRESHOLD_UPDATED", 23, 1.2, 0, 1), df)                # T03
    assert t._fibs[(0, 0)].cts_idx == 22


# ---------------------------------------------------------------------------------------------------------------------
# U05: the `via: "cross_fib_limit"` label is for cycles >= 1 only — cycle 0's late activation stays unlabelled
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_a_cycle_0_late_activation_is_never_labelled(sd):
    """Cycle 0 established at 20 with no gap in [10, 20]; a raw CTS_UPDATED to 26 finds the gap at 22 (formed at 23)
    -> the cycle-0 single activates on the update with exactly the pre-S3 meta (no `via`)."""
    t = _tracker(_lim(9.0, sd))
    df = _df(40, [_gap(22, sd, 1.10, 1.00)])
    assert _q(t.on_cts_established, _est(20, 1.15, 0, sd), df, bos_idx=10, bos_price=_m(1.0, sd)) is None
    out = _q(t.on_cts_updated, _ev("CTS_UPDATED", 26, 1.20, 0, sd, via=CTS_UPDATED_RAW_VIA), df)
    assert out is t._fibs[(0, 0)] and out.active
    assert out.meta == {"activated_at": 26, "fib_mode": "cross_cycle", "activated_on": "update"}


@pytest.mark.parametrize("sd", [1, -1])
def test_an_update_of_a_confirmed_cycle_never_moves_its_latest_anchor(sd):
    """Lens A m1 — the F16 order (PLAN_I §9.2): cycle 0 confirmed at 25 with anchor 28 (1.25, beyond P 1.20 → M = 0),
    then its earlier, LESS extreme raw update at 26 (1.19) processed after the confirmation. Before the fix it moved
    cycle 0's latest anchor back (M → None), a threshold update at 27 created a pre-established cross for cycle 1, and
    cycle 1's establishment (past the limit) then tripped I8. Now the confirmed anchor wins: M stays 0 throughout."""
    t = _tracker(_lim(1.20, sd))
    df = _df(60, [_gap(15, sd, 1.10, 1.00), _gap(26, sd, 1.47, 1.46), _gap(35, sd, 1.30, 1.20)])
    if sd == 1:
        df.at[26, "l"] = 1.45                                           # the dip -> prospective BOS_1 = 26
    else:
        df.at[26, "h"] = _m(1.45, -1)
    _q(t.on_cts_established, _est(20, 1.15, 0, sd), df, bos_idx=10, bos_price=_m(1.0, sd))
    _q(t.on_cts_confirmed, _conf(25, 1.25, 0, sd, 28))
    assert t.cross_fib_limit_cycle(0) == 0 and not t._cross_allowed(0, 1)
    _q(t.on_cts_updated, _ev("CTS_UPDATED", 26, 1.19, 0, sd, via=CTS_UPDATED_RAW_VIA), df)
    assert t._latest_anchor[(0, 0)] == (28, _m(1.25, sd), sd) and not t._cross_allowed(0, 1)
    _q(t.on_cts_threshold_updated, _ev("CTS_THRESHOLD_UPDATED", 27, 1.25, 0, sd), df)
    assert t._get_latest_cross(0, 1) is None
    _q(t.on_cts_updated, _ev("CTS_UPDATED", 28, 1.25, 0, sd, via=CTS_UPDATED_RAW_VIA), df)
    _q(t.on_cts_established, _est(40, 1.35, 1, sd), df, bos_idx=30, bos_price=_m(1.20, sd))
    assert _kind(t, 1) == "single cross_fib_limit"
