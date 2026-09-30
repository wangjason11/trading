"""A never-established cycle's pre-created fib ends at the subordinate cap (2026-09-28; zones-audit "never-established-
cycle fallback POI").

`cross_cycle` fib mode (every sub variant; H1 `h1` mode has no pre-established phase) pre-creates the NEXT cycle's
cross fib once cycle n's CTS is confirmed (FIB_LIFECYCLE_SPEC §6). If that next cycle never establishes, its fib has
no `compute_cycle_lifecycle` row. A REVERSAL already ended it (`set_reversal_terminals` stamps every fib of the
ended sid) and `poi_zones` then built no POIs on it (an ended, unlocked fib); the sub's lifecycle CAP did not, so the
fib stayed open and its POIs ran to the frame edge past the sub's end, never activating (their fallback floor is the
fib's latest CTS candle). Fix (user decision, option A "cap like the reversal"): `_finalize_lifecycle_fields` feeds
the cap as an end candidate for such fibs (FIB_LIFECYCLE_SPEC §15.4 candidate 3) — restoring what the post-hoc sub
fib cap loop removed in `559db50` (§15.6) had done for them. A still-LIVE pre-created fib (an open-ended sub) keeps
no end; since 2026-09-29 (user decision, option B = "N": no POI until the cycle establishes) it builds no POI either —
before, its POI got a dead fallback floor (the fib's latest CTS candle) and never activated.

Reference window: 0 never-established fib rows on either lens (counter 8/8, confluence 21/21 established) —
byte-identical. The fixture is real engine output (a random-tail search on `_make_multicycle_data()[:9]`, 2026-09-28).
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.multitf.pooled_structure_build import project_to_window
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_unified_probe import _R, _make_multicycle_data, _prepare_df

# sd=+1, start 0, NZD_USD, run at M15 pips in `cross_cycle` mode (a sub's downstream).
#   0-8  = `_make_multicycle_data()[:9]`: cycle 0 established at 2 (BOS .5998; CTS anchor 2 -> 5 .6122 via the
#        CTS_UPDATEDs at 3/4/5), CTS_CONFIRMED at 7 (the 6-7 pullback; IC 7 [.6088-.6107]), range active.
#   9    the first range-sync CTS_THRESHOLD_UPDATED -> FibTracker pre-creates the cycle-1 cross fib (BOS .5998 -> the
#        running high) and ends the cycle-0 fib at 9 (`new_cycle`).
#   10-23 a climb: a threshold update on every candle but 15 (its high .61904 < .61917), no breakout pattern -> cycle 1
#        never establishes.
# Capped at 22 (a sub whose data continues to 23): before the fix the cycle-1 fib stayed open (end None) and its POI
# on IC 7 had no end (drawn to 23), `inactive`, with the fallback floor 22.
_CAP, _IC = 22, 7


def _rows() -> list[dict]:
    rows = list(_make_multicycle_data()[:9])
    rows += [
        _R(0.61150, 0.61369, 0.61088, 0.61281),   # 9
        _R(0.61351, 0.61500, 0.61348, 0.61472),   # 10
        _R(0.61472, 0.61533, 0.61325, 0.61405),   # 11
        _R(0.61536, 0.61658, 0.61531, 0.61629),   # 12
        _R(0.61629, 0.61744, 0.61594, 0.61701),   # 13
        _R(0.61701, 0.61917, 0.61628, 0.61834),   # 14
        _R(0.61834, 0.61904, 0.61595, 0.61862),   # 15
        _R(0.61862, 0.62033, 0.61802, 0.61979),   # 16
        _R(0.62044, 0.62193, 0.62029, 0.62154),   # 17
        _R(0.62154, 0.62211, 0.61917, 0.62175),   # 18
        _R(0.62175, 0.62240, 0.62140, 0.62199),   # 19
        _R(0.62199, 0.62324, 0.62151, 0.62177),   # 20
        _R(0.62177, 0.62393, 0.62116, 0.62323),   # 21
        _R(0.62463, 0.62622, 0.62461, 0.62597),   # 22
        _R(0.62597, 0.62797, 0.62541, 0.62703),   # 23
    ]
    return rows


def _reversing_rows() -> list[dict]:
    """The same climb to 22, then a crash through BOS .5998: close-break at 23, reversal at 24."""
    return _rows()[:23] + [
        _R(0.62597, 0.62610, 0.59880, 0.59900),   # 23
        _R(0.59900, 0.59920, 0.59500, 0.59520),   # 24
        _R(0.59520, 0.59560, 0.59480, 0.59540),   # 25
        _R(0.59540, 0.59580, 0.59500, 0.59560),   # 26
        _R(0.59560, 0.59600, 0.59520, 0.59580),   # 27
    ]


def _downstream(rows, *, end_idx=None, cap=None):
    """A sub-style build: bounded MS at the cap, events clipped to it, the cross_cycle downstream with the cap.
    Asserts the path's precondition: cycle 1 has no CTS_ESTABLISHED and a pre-created cycle-1 cross fib exists."""
    with redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(rows), 0, 1, timeframe="M15", end_idx=end_idx)
        events = [e for e in res.events if cap is None or int(e.idx) <= cap]
        out = _run_downstream_pipeline(res.df, events, 1, fib_mode="cross_cycle", timeframe="M15",
                                       structure_path_id="M15.test", skip_wvmi=True, lifecycle_cap=cap)
    assert not [e for e in events if e.type == "CTS_ESTABLISHED" and e.meta["cycle_id"] == 1]
    fib1 = [f for f in out["fib_states"] if (f.structure_id, f.cycle_id) == (0, 1)]
    assert fib1 and all(f.meta.get("cross_start_cycle") == 0 for f in fib1)
    return out, fib1, [z for z in out["poi_zones"] if (z.meta["structure_id"], z.meta["cycle_id"]) == (0, 1)]


def test_the_cap_ends_a_never_established_cycles_fib_and_builds_no_poi():
    _out, fib1, pois1 = _downstream(_rows(), end_idx=_CAP, cap=_CAP)
    assert {(f.end_idx, f.end_reason, f.status) for f in fib1} == {(_CAP, "lifecycle_end", "ended")}
    assert pois1 == []


def test_the_production_projection_caps_it_with_the_subs_end_reason():
    """The sub path itself (landing review F3): a natural-end run projected onto a lifecycle window by
    `project_to_window` (knowable-at clip + the downstream with `cap_reason` = the sub's end reason). Cap 15 != the
    fib's CTS candle 14, and `parent_end` is a real sub end reason (never the default `lifecycle_end`)."""
    with redirect_stdout(io.StringIO()):
        natural = compute_bounded_structure(_prepare_df(_rows()), 0, 1, timeframe="M15")
        down = project_to_window(natural, floor=2, cap=15, cap_reason="parent_end", direction=1, timeframe="M15")
    assert [(e.idx, e.meta["cycle_id"]) for e in down["events"] if e.type == "CTS_ESTABLISHED"] == [(2, 0)]
    fib1 = [f for f in down["fib_states"] if (f.structure_id, f.cycle_id) == (0, 1)]
    assert {(f.cts_idx, f.end_idx, f.end_reason, f.status) for f in fib1} == {(14, 15, "parent_end", "ended")}
    assert not [z for z in down["poi_zones"] if z.meta["cycle_id"] == 1]


def test_a_reversal_ends_it_the_same_way():
    """The path the cap now mirrors (unchanged by the fix): the realised reversal at 24 ends the fib, no POI."""
    _out, fib1, pois1 = _downstream(_reversing_rows())
    assert {(f.end_idx, f.end_reason) for f in fib1} == {(24, "reversal")}
    assert pois1 == []


def test_an_open_ended_run_keeps_the_live_fib_but_builds_no_poi():
    """No cap, no reversal (an open-ended sub at the data edge): the pre-created fib is still live, so it keeps no end —
    and its cycle never established, so it builds no POI (option N, 2026-09-29). Before: one POI on IC 7 (the fib's
    61.8-80% zone at .62622 = [.60508-.60989], ~57% overlap, V30), end None, `inactive`, a fallback floor 22 = the
    fib's CTS candle, never activated; the only POI on this sub (cycle 0's zone [.60228-.60454] misses IC 7)."""
    out, fib1, pois1 = _downstream(_rows()[:_CAP + 1])
    assert {(f.end_idx, f.status) for f in fib1} == {(None, "active")}
    assert pois1 == []
    assert out["poi_zones"] == []
