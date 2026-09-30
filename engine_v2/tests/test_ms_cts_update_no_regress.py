"""A pre-confirm `CTS_UPDATED` only on a strict new extreme — on BOTH paths (2026-09-27; zones-audit latent bug (a)).

MARKET_STRUCTURE_SPEC "CTS": `CTS_UPDATED` fires when the CTS anchor moves to a NEW extreme. The raw path
(`_maybe_update_cts_pre_confirm`) always checked it; the pattern path (a continuation breakout while the CTS is
unconfirmed) set `st.cts` to the breakout's pattern extreme unconditionally. Because the raw path already moved
`st.cts` through the back-filled candles, a breakout after a shallow dip that tops out BELOW the current CTS
regressed it — and a later candle between the two levels then drew a spurious raw "new extreme", so the pullback
confirmed the wrong candle. Fix (user decision B): both paths share `_is_new_cts_extreme` (strict; a tie keeps
the first occurrence); a non-new-extreme breakout emits nothing, leaves `st.cts` and skips the POI-inner refresh,
but still breaks the range / sets BREAKOUT / records `last_breakout_pat_apply_idx`.

Reference window (2026-09-27 shadow over every MS run): 39 such breakouts, 0 regressions, 1 same-candle tie (conf
sub 2 cyc 0: pattern extreme 2468 == the raw update's candle 2468, apply 2470) — its duplicate CTS_UPDATED at 2470
is the fix's only exported delta.
"""
from __future__ import annotations

import pytest

from engine_v2.structure import event_fields as ef
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_unified_probe import _R, _make_multicycle_data, _prepare_df

# sd=+1, cycle 0 (pullback-only confirmation: sd-zone proximity is gated off on cycle 0).
#   0-4  bull marus -> CTS_ESTABLISHED at 2 (anchor 2), pattern-path CTS_UPDATED at 4 (a real new extreme)
#   5    bull normal, long upper wick: raw CTS_UPDATED to h5 .6135 = the true high
#   6-7  bear normals (no pullback pattern), dip to .6088
#   8-9  two bull marus = one_maru_continuous(+1), apply 9, pattern extreme h9 .6132 < .6135
#   10-11 bull pinbars below .6132; 12 bull pinbar h .6133 (between .6132 and .6135)
#   13-14 two bear marus -> pullback -> CTS_CONFIRMED at 14
#   15-17 fillers
# Before the fix: CTS_UPDATED .6132 @9 (regress), raw CTS_UPDATED .6133 @12, CTS_CONFIRMED anchor 12 @ .6133.
# After: nothing at 9 or 12, CTS_CONFIRMED anchor 5 @ .6135.
_PATTERN_APPLY, _TRUE_HIGH, _POKE, _CONFIRM = 9, 5, 12, 14


def _regress_rows(h9: float = 0.6132, c9: float = 0.6131) -> list[dict]:
    rows = []
    p = 0.6000
    for _ in range(5):
        rows.append(_R(p, p + 0.0022, p - 0.0002, p + 0.0020))
        p += 0.0020
    rows += [
        _R(0.6100, 0.6135, 0.6098, 0.6120),   # 5
        _R(0.6120, 0.6125, 0.6103, 0.6108),   # 6
        _R(0.6108, 0.6113, 0.6088, 0.6095),   # 7
        _R(0.6095, 0.6128, 0.6094, 0.6127),   # 8
        _R(0.6127, h9, 0.6126, c9),           # 9
        _R(0.6126, 0.6129, 0.6118, 0.6127),   # 10
        _R(0.6126, 0.6129, 0.6118, 0.6127),   # 11
        _R(0.6126, 0.6133, 0.6118, 0.6127),   # 12
        _R(0.6127, 0.6128, 0.6105, 0.6107),   # 13
        _R(0.6107, 0.6108, 0.6085, 0.6087),   # 14
    ]
    rows += [_R(0.6087, 0.6095, 0.6083, 0.6093)] * 3   # 15-17
    return rows


_MIRROR = 1.2200   # sd=-1 twin: every price p -> _MIRROR - p (h/l swap), candle shapes and types preserved


def _mirror(rows: list[dict]) -> list[dict]:
    return [_R(_MIRROR - r["o"], _MIRROR - r["l"], _MIRROR - r["h"], _MIRROR - r["c"]) for r in rows]


def _run(rows: list[dict], sd: int):
    return compute_bounded_structure(_prepare_df(rows), 0, sd)


def _breakout_applied_at(res, idx: int) -> bool:
    """Precondition: a breakout pattern was applied at `idx` (the path under test). The state is already
    BREAKOUT there, so no STATE_CHANGED marks it — the df column does."""
    col = res.df["last_breakout_pat_apply_idx"]
    return int(col.iloc[idx]) == idx and int(col.iloc[idx - 1]) != idx


def _px(price: float, sd: int) -> float:
    return round(price if sd == 1 else _MIRROR - price, 5)


def _cts_updates(events):
    return [e for e in events if e.type == "CTS_UPDATED"]


@pytest.mark.parametrize("sd", [1, -1])
def test_a_breakout_below_the_current_cts_does_not_regress_it(sd):
    rows = _regress_rows() if sd == 1 else _mirror(_regress_rows())
    res = _run(rows, sd)
    events = res.events
    assert _breakout_applied_at(res, _PATTERN_APPLY)
    upd = {e.idx: e for e in _cts_updates(events)}
    assert _PATTERN_APPLY not in upd                       # no pattern-path update to .6132
    assert _POKE not in upd                                # no spurious raw "new extreme" at .6133
    assert round(upd[_TRUE_HIGH].price, 5) == _px(0.6135, sd) and upd[_TRUE_HIGH].meta["via"] == "replay_raw"
    confirmed = [e for e in events if e.type == "CTS_CONFIRMED"]
    assert len(confirmed) == 1 and confirmed[0].idx == _CONFIRM
    assert confirmed[0].meta["cts_anchor_idx"] == _TRUE_HIGH
    assert round(confirmed[0].price, 5) == _px(0.6135, sd)


@pytest.mark.parametrize("sd", [1, -1])
def test_a_breakout_beyond_the_current_cts_still_updates_it_on_the_pattern_path(sd):
    """The positive half: 3-4 is a one_maru_continuous whose extreme h4 .6102 clears the raw-updated h3 .6082."""
    rows = _regress_rows() if sd == 1 else _mirror(_regress_rows())
    upd = {e.idx: e for e in _cts_updates(_run(rows, sd).events)}
    assert upd[4].meta["via"] == "one_maru_continuous" and upd[4].meta["cts_anchor_idx"] == 4
    assert upd[4].meta["confirmed_at"] == 4 and round(upd[4].price, 5) == _px(0.6102, sd)


def test_a_breakout_tying_the_current_cts_keeps_the_first_occurrence():
    """h9 == h5 .6135 exactly: a tie is not a new extreme (the raw path's strict `>`), so the anchor stays 5."""
    res = _run(_regress_rows(h9=0.6135, c9=0.6134), 1)
    events = res.events
    assert _breakout_applied_at(res, _PATTERN_APPLY)
    assert _PATTERN_APPLY not in {e.idx for e in _cts_updates(events)}
    confirmed = [e for e in events if e.type == "CTS_CONFIRMED"]
    assert len(confirmed) == 1 and confirmed[0].meta["cts_anchor_idx"] == _TRUE_HIGH


@pytest.mark.parametrize("rows, sd", [
    (_regress_rows(), 1), (_mirror(_regress_rows()), -1), (_regress_rows(h9=0.6135, c9=0.6134), 1),
    (_make_multicycle_data(), 1),
], ids=["regress", "regress_mirror", "tie", "multicycle"])
def test_every_cts_updated_is_a_strict_new_extreme_of_its_cycle(rows, sd):
    """Running invariant over each cycle: every CTS_UPDATED (either path) lies strictly beyond the CTS it replaces."""
    cur, n_upd = {}, 0
    for e in _run(rows, sd).events:
        if e.type == "CTS_ESTABLISHED":
            cur[e.meta["cycle_id"]] = e.price
        elif e.type == "CTS_UPDATED":
            prev = cur[e.meta["cycle_id"]]
            assert (e.price > prev) if sd == 1 else (e.price < prev), (e.idx, e.price, prev)
            cur[e.meta["cycle_id"]] = e.price
            n_upd += 1
    assert n_upd >= 3


def test_no_cts_move_means_no_poi_inner_refresh(monkeypatch):
    """The skipped breakout does not refresh MS's POI-inner snapshot either: it emits no CTS_UPDATED, so
    FibTracker does not advance its fill horizon there, and the two layers stay in lock-step (LANDMINES
    "Scenario 2 anchor agreement"). The real new extremes (4 pattern, 5 raw) still refresh."""
    from engine_v2.structure.market_structure import MarketStructure

    moments = []
    orig = MarketStructure._refresh_poi_inners_for_cycle

    def _spy(self, moment_idx):
        moments.append(int(moment_idx))
        return orig(self, moment_idx)

    monkeypatch.setattr(MarketStructure, "_refresh_poi_inners_for_cycle", _spy)
    res = _run(_regress_rows(), 1)
    assert _breakout_applied_at(res, _PATTERN_APPLY)
    assert {2, 4, _TRUE_HIGH} <= set(moments)
    assert _PATTERN_APPLY not in moments and _POKE not in moments


# --- landing review (2026-09-27): what a NON-moving breakout still does, and the refresh arguments ------------
# A tie with an active range is REAL (the reference window's conf sub 2 2470 was one: a big maru breaks the range
# during a back-fill, `freeze_range` leaves `range_hi`, the raw path takes the maru as the CTS, and the breakout's
# extreme is that maru) — so the skipped path's range break / BREAKOUT / post-apply range check need pins.


def _range_tie_rows(filler_closes_inside_14: bool = False) -> list[dict]:
    """(Landing review 2026-09-27, mutation lens.) `_regress_rows` 0-12 (range 10-12 confirmed at 12, range_hi = CTS .6135, pre-confirm) +
    13 big bull maru h .6160 (breaks the range; the raw path takes it in the back-fill, range frozen) +
    14 small bear normal -> one_maru_opposite(+1) applied at 14, pattern extreme 13 = a TIE with the
    raw-updated CTS -> no CTS_UPDATED, but the breakout must still break the range + set BREAKOUT.
    `filler_closes_inside_14`: fillers close inside candle 14 -> 14 is itself a range candle (confirm 16),
    found only by the breakout's `_post_apply_range_check(14)`."""
    rows = _regress_rows()[:13]
    c = 0.6127
    rows.append(_R(c, c + 0.0033, c - 0.0002, c + 0.0031))            # 13
    c += 0.0031
    rows.append(_R(c - 0.0001, c + 0.0001, c - 0.0009, c - 0.0007))   # 14
    c -= 0.0007
    if filler_closes_inside_14:
        rows += [_R(c, c + 0.0003, c - 0.0002, c + 0.0001)] * 4      # 15-18
    else:
        rows += [_R(c, c + 0.0002, c - 0.0006, c - 0.0004)] * 3      # 15-17
    return rows


@pytest.mark.parametrize("sd", [1, -1])
def test_a_tied_breakout_from_a_range_still_breaks_the_range_and_sets_breakout(sd):
    rows = _range_tie_rows() if sd == 1 else _mirror(_range_tie_rows())
    res = _run(rows, sd)
    ev = res.events
    assert int(res.df["range_active"].iloc[13]) == 1                       # range active going in
    assert int(res.df["last_breakout_pat_apply_idx"].iloc[14]) == 14       # the breakout applied at 14
    upd = [e for e in ev if e.type == "CTS_UPDATED"]
    assert [e.meta["via"] for e in upd if e.idx == 13] == ["replay_raw"] and not [e for e in upd if e.idx == 14]
    assert [(e.idx, e.meta["reason"]) for e in ev if e.type == "RANGE_RESET"] == [(14, "range_breakout")]
    sc = [(e.meta["from"], e.meta["to"], e.meta["reason"]) for e in ev if e.type == "STATE_CHANGED" and e.idx == 14]
    assert sc == [("range", "breakout", "breakout_pattern")]
    assert res.df["market_state"].iloc[14] == "breakout" and int(res.df["range_active"].iloc[14]) == 0


@pytest.mark.parametrize("sd", [1, -1])
def test_a_tied_breakout_still_runs_the_post_apply_range_check(sd):
    rows = _range_tie_rows(True) if sd == 1 else _mirror(_range_tie_rows(True))
    res = _run(rows, sd)
    assert not [e for e in res.events if e.type == "CTS_UPDATED" and e.idx == 14]
    rs = [(e.idx, e.meta["start_idx"]) for e in res.events if e.type == "RANGE_STARTED" and e.idx > 14]
    assert rs == [(16, 14)]                    # the apply candle 14 is the range candle (post-apply check)


@pytest.mark.parametrize("fixture", ["regress", "tied"])
def test_every_poi_refresh_uses_the_moved_cts(monkeypatch, fixture):
    """(Landing review.) Each CTS_UPDATED (either path) refreshes with (its CTS anchor, its moment) -- the refresh runs
    AFTER st.cts moves. The pattern-path half lost its pin when the lagging (24, 25) update went away."""
    import engine_v2.structure.structure_engine as se
    from engine_v2.tests.test_imbalance_c3_knowability import _multicycle_with_tied_pattern_breakout
    calls = []
    real = se.compute_poi_inners_for_cycle

    def spy(df, bos_idx, bos_price, cts_idx, *a, **k):
        calls.append((int(cts_idx), k["fill_horizon_idx"]))
        return real(df, bos_idx, bos_price, cts_idx, *a, **k)

    monkeypatch.setattr(se, "compute_poi_inners_for_cycle", spy)
    rows = _regress_rows() if fixture == "regress" else _multicycle_with_tied_pattern_breakout()
    res = _run(rows, 1)
    pat = [e for e in res.events if e.type == "CTS_UPDATED" and e.meta["via"] != ef.CTS_UPDATED_RAW_VIA]
    assert pat
    for e in [e for e in res.events if e.type == "CTS_UPDATED"]:
        assert (ef.cts_anchor_idx(e), ef.event_moment(e)) in calls, (e.idx, e.meta["via"])


def test_ms_cycle0_snapshot_refresh_passes_the_moment_for_a_lagging_cts0(monkeypatch):
    """(Landing review; a pre-existing gap.) `_refresh_poi_inners_for_cycle` hands `_update_cycle0_data` the MOMENT: a cycle-0
    one_maru_opposite (big bull maru 1 = the CTS_0 anchor, small bear normal 2 = the apply) establishes
    CTS_0 anchor 1 / moment 2. The existing E3a' pins call `_update_cycle0_data` directly, so the call-site
    argument is unpinned."""
    from engine_v2.structure.market_structure import MarketStructure
    rec = []
    orig = MarketStructure._update_cycle0_data

    def spy(self, m):
        rec.append((int(m), int(self.state.cts.idx)))
        return orig(self, m)

    monkeypatch.setattr(MarketStructure, "_update_cycle0_data", spy)
    p = 0.6000
    rows = [_R(p, p + 0.0022, p - 0.0002, p + 0.0020)]; p += 0.0020
    rows.append(_R(p, p + 0.0052, p - 0.0002, p + 0.0050)); p += 0.0050
    rows.append(_R(p - 0.0001, p + 0.0001, p - 0.0009, p - 0.0007)); p -= 0.0007
    rows += [_R(p, p + 0.0002, p - 0.0006, p - 0.0004)] * 4
    res = _run(rows, 1)
    est = [e for e in res.events if e.type == "CTS_ESTABLISHED"]
    assert [(ef.cts_anchor_idx(e), ef.event_moment(e)) for e in est] == [(1, 2)]
    assert (2, 1) in rec and not [m for m, _c in rec if m == 1]
