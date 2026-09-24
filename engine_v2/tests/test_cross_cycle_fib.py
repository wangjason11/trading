"""Unit tests for the `cross_cycle` fib mode in FibTracker.

This mode (formerly `m15_reverse`) supports generalized cross-cycle fibs +
the `pre_established` phase. Used by all subordinate-structure pipelines.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import ImbalanceInstance
from engine_v2.structure.market_structure import CTS_UPDATED_RAW_VIA, StructureEvent
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig


# ---------- Test helpers ----------

def _make_df_with_imbalances(n_candles: int, instances: list[ImbalanceInstance]) -> pd.DataFrame:
    """Build a minimal OHLC df with pre-populated imbalance instances.

    Default candle levels: o=c=1.5, h=1.501, l=1.499 — far enough from the
    fill levels used in these tests (gaps around 1.0–1.5) that no imbalance
    is incidentally filled. Override df.at[idx, "l"]/"h" in specific tests
    to simulate fills.
    """
    data = {
        "time": pd.date_range("2026-01-01", periods=n_candles, freq="15min"),
        "o": [1.5] * n_candles,
        "h": [1.501] * n_candles,
        "l": [1.499] * n_candles,
        "c": [1.5] * n_candles,
        "direction": [0] * n_candles,
        "is_imbalance": [0] * n_candles,
    }
    df = pd.DataFrame(data)
    df.attrs["imbalances"] = list(instances)
    return df


def _ev(ev_type: str, idx: int, price: float, sid: int, cycle_id: int, sd: int,
        cts_anchor_idx: int = None) -> StructureEvent:
    meta = {"structure_id": sid, "cycle_id": cycle_id, "struct_direction": sd}
    # The event contract FibTracker reads by direct index (event_moment): a
    # CTS_ESTABLISHED carries its moment (lag 0 here), a CTS_UPDATED its via
    # (the raw path here).
    if ev_type == "CTS_ESTABLISHED":
        meta["confirmed_at"] = idx
    if ev_type == "CTS_UPDATED":
        meta["via"] = CTS_UPDATED_RAW_VIA
    if cts_anchor_idx is not None:
        meta["cts_anchor_idx"] = cts_anchor_idx
    return StructureEvent(idx=idx, category="STRUCTURE", type=ev_type,
                          price=price, meta=meta)


def _make_tracker() -> FibTracker:
    return FibTracker(FibTrackerConfig(fill_threshold=0.70), fib_mode="cross_cycle")


# ---------- Cycle 0 behaves as simple single (no cross) ----------

def test_cycle_0_single_activates_with_unfilled_imbalance():
    # Imbalance between BOS_0=10 and CTS_0=20, gap not filled up to 20
    inst = ImbalanceInstance(
        start_idx=13, end_idx=13, direction=1,
        gap_top=1.10, gap_bottom=1.00, gap_size=0.10,
    )
    df = _make_df_with_imbalances(30, [inst])

    tracker = _make_tracker()
    ev = _ev("CTS_ESTABLISHED", 20, 1.2, sid=0, cycle_id=0, sd=1)
    result = tracker.on_cts_established(ev, df, bos_idx=10, bos_price=0.9)

    assert result is not None
    assert result.structure_id == 0 and result.cycle_id == 0
    assert result.active is True
    assert "cross_cycle" not in result.meta  # single fib, not cross
    # Phase set to "established"
    assert tracker._m15_phase.get((0, 0)) == "established"


def test_cycle_0_no_fib_without_imbalance():
    df = _make_df_with_imbalances(30, [])
    tracker = _make_tracker()
    ev = _ev("CTS_ESTABLISHED", 20, 1.2, sid=0, cycle_id=0, sd=1)
    result = tracker.on_cts_established(ev, df, bos_idx=10, bos_price=0.9)
    assert result is None


def test_cycle_0_first_activates_on_later_update():
    """Cycle-0 Fib that did NOT activate at CTS_EST must still be able to
    first-activate on a later CTS_UPDATED when an unfilled imbalance appears
    (2026-06-08 fix — removes the cross_cycle cycle-0 one-shot asymmetry)."""
    # Imbalance at idx=23 — OUTSIDE [BOS_0=10, CTS_0_EST=20], so no fib at EST;
    # a later CTS_UPDATED to 25 brings it into [10, 25] unfilled → activate.
    inst = ImbalanceInstance(
        start_idx=23, end_idx=23, direction=1,
        gap_top=1.10, gap_bottom=1.00, gap_size=0.10,
    )
    df = _make_df_with_imbalances(40, [inst])
    tracker = _make_tracker()

    est = tracker.on_cts_established(
        _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0, 1), df, bos_idx=10, bos_price=0.9,
    )
    assert est is None                       # one-shot would stop here
    assert (0, 0) not in tracker._fibs

    upd = tracker.on_cts_updated(_ev("CTS_UPDATED", 25, 1.25, 0, 0, 1), df)
    assert upd is not None
    assert upd.structure_id == 0 and upd.cycle_id == 0
    assert upd.active is True
    assert upd.meta.get("activated_on") == "update"
    assert (0, 0) in tracker._fibs


def test_cycle_0_update_no_activation_when_still_no_imbalance():
    """Negative: with no unfilled imbalance, a CTS_UPDATED must NOT
    spuriously activate a cycle-0 Fib."""
    df = _make_df_with_imbalances(40, [])
    tracker = _make_tracker()
    assert tracker.on_cts_established(
        _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0, 1), df, bos_idx=10, bos_price=0.9,
    ) is None
    assert tracker.on_cts_updated(_ev("CTS_UPDATED", 25, 1.25, 0, 0, 1), df) is None
    assert (0, 0) not in tracker._fibs


# ---------- Cross activation: (0→1) ----------

def test_cross_activation_cycle_1():
    # Cycle 0: imbalance at idx 15 (in [10, 20])
    # Cycle 1: imbalance at idx 35 (in [30, 40])
    inst_0 = ImbalanceInstance(
        start_idx=15, end_idx=15, direction=1,
        gap_top=1.10, gap_bottom=1.00, gap_size=0.10,
    )
    inst_1 = ImbalanceInstance(
        start_idx=35, end_idx=35, direction=1,
        gap_top=1.30, gap_bottom=1.20, gap_size=0.10,
    )
    df = _make_df_with_imbalances(60, [inst_0, inst_1])

    tracker = _make_tracker()

    # Cycle 0 activation and confirmation
    ev0_est = _ev("CTS_ESTABLISHED", 20, 1.15, sid=0, cycle_id=0, sd=1)
    tracker.on_cts_established(ev0_est, df, bos_idx=10, bos_price=1.0)

    ev0_conf = _ev("CTS_CONFIRMED", 25, 1.15, sid=0, cycle_id=0, sd=1, cts_anchor_idx=20)
    tracker.on_cts_confirmed(ev0_conf)

    # Phase setup: (0, 0) confirmed, (0, 1) pre_established
    assert tracker._m15_phase[(0, 0)] == "confirmed"
    assert tracker._m15_phase[(0, 1)] == "pre_established"

    # Cycle 1 CTS_ESTABLISHED (BOS_1 at idx 30, CTS_1 at idx 40)
    ev1_est = _ev("CTS_ESTABLISHED", 40, 1.35, sid=0, cycle_id=1, sd=1)
    result = tracker.on_cts_established(ev1_est, df, bos_idx=30, bos_price=1.2)

    # Should activate cross (0→1)
    assert result is not None
    assert result.meta.get("cross_cycle") is True
    assert result.meta.get("cross_start_cycle") == 0
    # Cross anchors: BOS_0 to CTS_1
    assert result.bos_idx == 10
    assert result.cts_idx == 40
    # Phase flipped
    assert tracker._m15_phase[(0, 1)] == "established"


# ---------- Cross activation spanning multiple cycles: (0→2) ----------

def test_cross_activation_cycle_2_multi_jump():
    # All three cycles have unfilled imbalances
    insts = [
        ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
        ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10),
        ImbalanceInstance(55, 55, 1, 1.50, 1.40, 0.10),
    ]
    df = _make_df_with_imbalances(80, insts)

    tracker = _make_tracker()

    # Set up cycles 0 and 1 (activate + confirm)
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 20, 1.15, 0, 0, 1), df, 10, 1.0)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 25, 1.15, 0, 0, 1, cts_anchor_idx=20))
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 40, 1.35, 0, 1, 1), df, 30, 1.2)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 45, 1.35, 0, 1, 1, cts_anchor_idx=40))

    # Cycle 2 activation
    ev2_est = _ev("CTS_ESTABLISHED", 60, 1.55, 0, 2, 1)
    result = tracker.on_cts_established(ev2_est, df, bos_idx=50, bos_price=1.4)

    # Cross (0→2) expected: all cycles still alive
    assert result is not None
    assert result.meta.get("cross_cycle") is True
    assert result.meta.get("cross_start_cycle") == 0
    assert result.bos_idx == 10  # BOS_0
    assert result.cts_idx == 60  # CTS_2


# ---------- Cross fails when a prior cycle's imbalance is filled ----------

def test_cross_fails_when_prior_cycle_filled():
    # Cycle 0 imbalance in [10, 20], narrow gap that gets filled by idx 22
    # Cycle 1 imbalance in [30, 40], still unfilled
    inst_0 = ImbalanceInstance(
        start_idx=15, end_idx=15, direction=1,
        gap_top=1.10, gap_bottom=1.00, gap_size=0.10,
    )
    inst_1 = ImbalanceInstance(
        start_idx=35, end_idx=35, direction=1,
        gap_top=1.30, gap_bottom=1.20, gap_size=0.10,
    )
    df = _make_df_with_imbalances(60, [inst_0, inst_1])
    # Fill cycle 0's imbalance: set idx 22 low <= 1.10 - 0.10*0.70 = 1.03
    df.at[22, "l"] = 1.02

    tracker = _make_tracker()

    # Cycle 0 (imbalance exists at activation time, check_to=20 → unfilled)
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 20, 1.15, 0, 0, 1), df, 10, 1.0)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 25, 1.15, 0, 0, 1, cts_anchor_idx=20))

    # Cycle 1 CTS_ESTABLISHED at idx 40 — cycle 0 imbalance fill checked up to 40:
    # low at idx 22 = 1.02 ≤ 1.03 → cycle 0 filled → dead
    result = tracker.on_cts_established(_ev("CTS_ESTABLISHED", 40, 1.35, 0, 1, 1), df, 30, 1.2)

    # Cross fails → single fib activated as fallback (established phase)
    assert result is not None
    assert result.meta.get("cross_cycle") is not True
    assert result.meta.get("via") == "cross_failed"
    assert result.bos_idx == 30  # BOS_1
    assert result.cts_idx == 40  # CTS_1

    # Cycle 0 marked dead
    assert 0 in tracker._dead_cycles[0]


# ---------- Cross shrinking (0→2 becomes 1→2) via CTS_UPDATED ----------

def test_cross_shrinks_on_cts_updated_when_earlier_cycle_fills():
    # All cycles have imbalances; cycle 0's gap is set to get filled at idx 65
    insts = [
        ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
        ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10),
        ImbalanceInstance(55, 55, 1, 1.50, 1.40, 0.10),
    ]
    df = _make_df_with_imbalances(80, insts)

    tracker = _make_tracker()
    # Set up cycles 0, 1, 2
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 20, 1.15, 0, 0, 1), df, 10, 1.0)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 25, 1.15, 0, 0, 1, cts_anchor_idx=20))
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 40, 1.35, 0, 1, 1), df, 30, 1.2)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 45, 1.35, 0, 1, 1, cts_anchor_idx=40))

    # Cycle 2 CTS_ESTABLISHED → should give cross (0→2)
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 60, 1.55, 0, 2, 1), df, 50, 1.4)

    latest = tracker._get_latest_cross(0, 2)
    assert latest is not None
    assert latest[1].meta.get("cross_start_cycle") == 0
    assert latest[1].meta.get("version") == 0

    # Fill only cycle 0 (fill candle inside cycle 0's imbalance scan range but
    # before cycle 1's imbalance at 35, so cycle 1's is_filled scan (starting
    # at 36) doesn't see it). Gap fill level for cycle 0 = 1.10 - 0.07 = 1.03.
    df.at[22, "l"] = 1.02

    # CTS_UPDATED extends CTS_2 from 60 → 65
    ev_upd = _ev("CTS_UPDATED", 65, 1.60, 0, 2, 1)
    tracker.on_cts_updated(ev_upd, df)

    # Old cross (0→2) v0 should be deactivated with "cross_shortened"
    old_key = (0, 2, "cross", 0)
    assert old_key in tracker._fibs
    assert not tracker._fibs[old_key].active
    assert tracker._fibs[old_key].meta.get("deactivated_by") == "cross_shortened"

    # New cross (1→2) v1 should be active
    new_latest = tracker._get_latest_cross(0, 2)
    assert new_latest is not None
    assert new_latest[1].active
    assert new_latest[1].meta.get("cross_start_cycle") == 1
    assert new_latest[1].meta.get("version") == 1
    assert new_latest[1].bos_idx == 30  # BOS_1


# ---------- Cross fails entirely on CTS_UPDATED → fall back to single ----------

def test_cross_to_single_fallback_on_cts_updated():
    # Only cycle 1 has an imbalance at start. At check time, cycle 0 is dead.
    insts = [
        ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
        ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10),
    ]
    df = _make_df_with_imbalances(60, insts)

    tracker = _make_tracker()
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 20, 1.15, 0, 0, 1), df, 10, 1.0)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 25, 1.15, 0, 0, 1, cts_anchor_idx=20))

    # Cycle 1 starts with cross (0→1)
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 40, 1.35, 0, 1, 1), df, 30, 1.2)
    latest = tracker._get_latest_cross(0, 1)
    assert latest is not None and latest[1].meta.get("cross_start_cycle") == 0

    # Fill only cycle 0 (fill candle before cycle 1's imbalance at 35 so
    # cycle 1's is_filled scan starting at 36 doesn't see it).
    df.at[22, "l"] = 1.02
    tracker.on_cts_updated(_ev("CTS_UPDATED", 45, 1.40, 0, 1, 1), df)

    # Cross deactivated with cross_failed
    old_key = (0, 1, "cross", 0)
    assert tracker._fibs[old_key].meta.get("deactivated_by") == "cross_failed"
    assert not tracker._fibs[old_key].active

    # Single fib activated with via="cross_failed"
    single = tracker._fibs.get((0, 1))
    assert single is not None
    assert single.active
    assert single.meta.get("via") == "cross_failed"
    assert single.bos_idx == 30  # BOS_1
    assert single.cts_idx == 45  # extended CTS


# ---------- Dead-cycle caching (no re-check once dead) ----------

def test_dead_cycle_cache_skips_repeated_check():
    # Cycle 0 with imbalance that fills early
    inst_0 = ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10)
    inst_1 = ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10)
    df = _make_df_with_imbalances(60, [inst_0, inst_1])
    df.at[22, "l"] = 1.02  # fill cycle 0

    tracker = _make_tracker()
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 20, 1.15, 0, 0, 1), df, 10, 1.0)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 25, 1.15, 0, 0, 1, cts_anchor_idx=20))

    # Cycle 1 ESTABLISHED — cycle 0 found filled, marked dead
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 40, 1.35, 0, 1, 1), df, 30, 1.2)
    assert 0 in tracker._dead_cycles[0]

    # CTS_UPDATED — cycle 0 should stay dead, walk skips it immediately
    tracker.on_cts_updated(_ev("CTS_UPDATED", 45, 1.40, 0, 1, 1), df)
    assert 0 in tracker._dead_cycles[0]


# ---------- Phase gating: threshold-updated ignored when not pre_established ----------

def test_threshold_updated_noop_outside_pre_established():
    tracker = _make_tracker()
    df = _make_df_with_imbalances(30, [])

    # No phase set → on_cts_threshold_updated is a no-op
    ev = StructureEvent(
        idx=25, category="STRUCTURE", type="CTS_THRESHOLD_UPDATED",
        price=1.15,
        meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1},
    )
    # Should not raise, should not create fibs
    tracker.on_cts_threshold_updated(ev, df)
    assert len(tracker._fibs) == 0


# ---------- H1 mode ignores CTS_THRESHOLD_UPDATED ----------

def test_h1_mode_ignores_threshold_updated():
    tracker = FibTracker(FibTrackerConfig(fill_threshold=0.70), fib_mode="h1")
    df = _make_df_with_imbalances(30, [])
    ev = StructureEvent(
        idx=25, category="STRUCTURE", type="CTS_THRESHOLD_UPDATED",
        price=1.15,
        meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1},
    )
    tracker.on_cts_threshold_updated(ev, df)
    assert len(tracker._fibs) == 0


# ---------- Lock on CTS_n+1 CONFIRMED ----------

def test_cycle_1_confirmed_locks_cross_and_transitions_phase():
    insts = [
        ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
        ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10),
    ]
    df = _make_df_with_imbalances(60, insts)

    tracker = _make_tracker()
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 20, 1.15, 0, 0, 1), df, 10, 1.0)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 25, 1.15, 0, 0, 1, cts_anchor_idx=20))
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 40, 1.35, 0, 1, 1), df, 30, 1.2)
    # Cross (0→1) active
    latest = tracker._get_latest_cross(0, 1)
    assert latest is not None and latest[1].active
    assert not latest[1].locked

    # Confirm cycle 1 — should lock the cross
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 45, 1.35, 0, 1, 1, cts_anchor_idx=40))

    latest_after = tracker._get_latest_cross(0, 1)
    assert latest_after is not None
    assert latest_after[1].locked
    # Phase transitions
    assert tracker._m15_phase[(0, 1)] == "confirmed"
    assert tracker._m15_phase[(0, 2)] == "pre_established"
