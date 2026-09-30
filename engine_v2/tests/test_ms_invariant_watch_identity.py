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

Since F3b (2026-09-29) no watch expiry fires at all (a pending confirming ON the expiry candle applies), so the
back-to-back shape no longer arises either; both rules are pure tripwires now, the frozen-barrier one kept.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.structure.structure_engine import _make_market_structure
from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df


# The review fixture (back-to-back watches: an expiry's rewind to anchor + 1 opening a new watch there, rows 9 and 10
# active with different frozen barriers) was pinned here until 2026-09-29 (F3b): a pending reversal confirming ON its
# watch's expiry candle now APPLIES (MARKET_STRUCTURE_SPEC "A reversal confirming on E applies"), so no expiry fires, no
# rewind happens and the second watch never opens — its watch 9 reverses at 14 on the frozen .60207. The rows moved to
# `test_ms_reversal_on_expiry.py` (a P3 pin); the frozen-barrier rule stays pinned on the constructed rows below.


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
