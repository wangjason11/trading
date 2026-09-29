"""MS df invariant 4 ("bos_threshold changed during reversal watch") compares rows of ONE watch (2026-09-29; landing
review of 52ac1e9, item 1).

MARKET_STRUCTURE_SPEC "Invariants / guard checks": while a reversal watch is open its BOS barrier is frozen, so
`bos_threshold` must not change between two rows of the same watch — the case it caught was a cycle established while
a watch is open (GOTCHAS "A Cycle Cannot Be Established Inside an Open Reversal Watch — MS Invariant 4"); since
2026-09-29 the new cycle ENDS the watch (`test_ms_new_cycle_ends_watch.py`), so the check is a pure tripwire. The check
compared every two consecutive `reversal_watch_active` rows, but an expiry rewinds to anchor + 1 and that candle can
open a NEW watch on the moved barrier: two back-to-back watches, a false positive that ended the run (the check is
on by default and its AssertionError escapes the sub build's `except (ValueError, IndexError)`). Fix (user decision
2026-09-29): compare only rows with the same `reversal_bos_th_frozen` — a watch opened after an expiry freezes the
moved BOS, strictly beyond the old frozen barrier.

Measured before the fix (the old and the new rule evaluated on the same output rows): the review fixture perturbed
600x — 410 fires, all false positives (the frozen barrier differs and a REVERSAL_WATCH_START sits on the row); 24k
wide random tails — 3, all false positives; suite 0 / 1899 runs; reference window 0 (no expiries) -> byte-identical.
The new rule fired 0 times on those samples. The guarded case was reachable on real rows (the landing review built
one) until 2026-09-29, when a new cycle started to end the open watch — that fixture now runs clean
(`test_ms_new_cycle_ends_watch.py`); the guard stays pinned on constructed rows below.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.structure.structure_engine import _make_market_structure, compute_bounded_structure
from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
from engine_v2.tests.test_unified_probe import _R, _make_multicycle_data, _prepare_df


# The review fixture (its random-tail search, perturbed base; sd=-1: BOS above price, a close ABOVE it close-breaks).
#   8   breakout -> cycle 1 established, BOS .60207.
#   9   bull maru closing .60259 > .60207 -> watch 9 (frozen .60207, expires 14); its reversal confirms only at 14 = E
#       -> the expiry wins: false break, BOS := h9 .60278, rewind to 10.
#   10  re-run: closes .60556 > the new barrier .60278 -> watch 10 (frozen .60278, expires 15), pending reversal at 14.
#   14  watch 10's reversal applies -> REVERSAL @14.
# Rows 9 and 10 are both `reversal_watch_active` with bos .60207 -> .60278: two watches, not one. Pre-fix: AssertionError
# "bos_threshold changed during reversal watch at idx=10" (identically on edeef26, before the F3 fix).
def _rows() -> list[dict]:
    return [
        _R(0.59775, 0.60095, 0.59755, 0.60075), _R(0.60066, 0.60076, 0.59846, 0.59866),   # 0, 1
        _R(0.59786, 0.59796, 0.59566, 0.59586), _R(0.59589, 0.59859, 0.59579, 0.59839),   # 2, 3
        _R(0.59774, 0.60124, 0.59764, 0.60104), _R(0.60167, 0.60207, 0.60027, 0.60087),   # 4, 5
        _R(0.60048, 0.60068, 0.59728, 0.59748), _R(0.59739, 0.59759, 0.59379, 0.59399),   # 6, 7
        _R(0.59511, 0.59531, 0.59391, 0.59411), _R(0.59345, 0.60278, 0.59327, 0.60259),   # 8, 9
        _R(0.60254, 0.60618, 0.60190, 0.60556), _R(0.60528, 0.60645, 0.60476, 0.60601),   # 10, 11
        _R(0.60573, 0.60990, 0.60464, 0.60535), _R(0.60545, 0.60687, 0.60072, 0.60614),   # 12, 13
        _R(0.60615, 0.60999, 0.60438, 0.60900), _R(0.60926, 0.61057, 0.60866, 0.61013),   # 14, 15
        _R(0.60994, 0.61163, 0.60477, 0.60591), _R(0.60584, 0.60586, 0.60053, 0.60078),   # 16, 17
        _R(0.60051, 0.60139, 0.59587, 0.59642), _R(0.59619, 0.59695, 0.59510, 0.59575),   # 18, 19
        _R(0.59591, 0.59787, 0.59210, 0.59257), _R(0.59259, 0.59386, 0.58801, 0.59356),   # 20, 21
        _R(0.59330, 0.59339, 0.58666, 0.58677), _R(0.58679, 0.59119, 0.58498, 0.59025),   # 22, 23
        _R(0.59026, 0.59034, 0.58558, 0.58564),                                           # 24
    ]


def _price(p: float, sd: int) -> float:
    return p if sd == -1 else round(_noreg._MIRROR - p, 5)


@pytest.mark.parametrize("sd", [-1, 1])
def test_back_to_back_watches_are_not_one_watch(sd):
    rows = _rows() if sd == -1 else _noreg._mirror(_rows())
    with redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(rows), 0, sd)   # debug_invariants on (the default)
    df = res.df
    # the shape: rows 9 and 10 active, each with its own frozen barrier, the BOS moved between them
    assert [int(df.loc[i, "reversal_watch_active"]) for i in (9, 10)] == [1, 1]
    assert [round(float(df.loc[i, "reversal_bos_th_frozen"]), 5) for i in (9, 10)] == [_price(0.60207, sd),
                                                                                        _price(0.60278, sd)]
    assert [round(float(df.loc[i, "bos_threshold"]), 5) for i in (9, 10)] == [_price(0.60207, sd), _price(0.60278, sd)]
    assert [int(e.idx) for e in res.events if e.type == "REVERSAL_WATCH_START"][-2:] == [9, 10]
    assert res.reversal_idx == 14


# Constructed rows: invariant 4 alone, on a clean run's df (no watch anywhere) with two rows set by hand.
def _clean_ms():
    with redirect_stdout(io.StringIO()):
        ms = _make_market_structure(_prepare_df(_make_multicycle_data()), struct_direction=1, start_idx=0)
        ms.run()
    assert not ms.df["reversal_watch_active"].astype(bool).any()
    return ms


def _set_watch_rows(ms, rows):
    for i, (frozen, bos) in rows.items():
        ms.df.loc[i, "reversal_watch_active"] = 1
        ms.df.loc[i, "reversal_bos_th_frozen"] = frozen
        ms.df.loc[i, "bos_threshold"] = bos


def test_a_bos_move_inside_one_watch_still_raises():
    """The guard kept: the same frozen barrier on both rows, the BOS moved (a cycle established inside the watch)."""
    ms = _clean_ms()
    _set_watch_rows(ms, {12: (0.6088, 0.6088), 13: (0.6088, 0.6070)})
    with pytest.raises(AssertionError, match="bos_threshold changed during reversal watch at idx=13"):
        ms._check_invariants_df()


def test_a_new_watch_on_the_next_row_does_not_raise():
    ms = _clean_ms()
    _set_watch_rows(ms, {12: (0.6088, 0.6088), 13: (0.6070, 0.6070)})
    ms._check_invariants_df()
