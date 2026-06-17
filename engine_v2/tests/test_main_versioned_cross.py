"""Unit tests for the H1-main Scenario-2 cross in VERSIONED storage (§11a-ii).

11a-ii relocates the main cycle-1 cross from the `_cross_cycle_data` named slot
(`["cross_cycle"]`) + the `_fibs[(sid,1)]` mirror into the versioned key
`(sid, 1, "cross", 0)` + `_cross_version`, retiring the `normal_cycle1` scratch
slot (create-on-fail instead). The decision logic is unchanged (11a-i routed it
through the shared routine); these tests pin the STORAGE behavior:

  * cross stored at the versioned key, NOT `(sid,1)`; scratch slots not written;
  * meta is preserved exactly (no `version`/`fib_mode` keys → CSV columns stay
    empty → byte-identical);
  * update extends in place; confirm locks; the next cycle's activation still
    obsoletes the now-versioned cross (terminal `new_cycle`);
  * cond2-deactivation (cycle-1 own imbalance fills) deactivates the cross with
    no single created (the FALLBACK-to-normal path is unreachable for h1 since
    cond1/cond3 are fixed and cond2 == the normal's own check).
"""
from __future__ import annotations

import pandas as pd

from engine_v2.common.types import ImbalanceInstance
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig


def _df(n, instances):
    data = {
        "time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "o": [1.5] * n, "h": [1.501] * n, "l": [1.499] * n, "c": [1.5] * n,
        "direction": [0] * n, "is_imbalance": [0] * n,
    }
    df = pd.DataFrame(data)
    df.attrs["imbalances"] = list(instances)
    return df


def _ev(t, idx, price, sid, cyc, sd, cts_anchor_idx=None):
    meta = {"structure_id": sid, "cycle_id": cyc, "struct_direction": sd}
    if cts_anchor_idx is not None:
        meta["cts_anchor_idx"] = cts_anchor_idx
    return StructureEvent(idx=idx, category="STRUCTURE", type=t, price=price, meta=meta)


def _tracker():
    return FibTracker(FibTrackerConfig(fill_threshold=0.70), fib_mode="h1")


# cycle 0: BOS_0=10 CTS_0=20  imbalance@15 (gap 1.00-1.10)
# cycle 1: BOS_1=30 CTS_1=40  imbalance@35 (gap 1.20-1.30)
# cycle 2: BOS_2=50 CTS_2=60  imbalance@55 (gap 1.40-1.50)
_RV = 100  # reversal_confirmed_idx > CTS_0 → Scenario 1 resolves FALSE


def _drive_to_scenario2_cross(tracker, df):
    """sid=1: cycle 0 (S1 FALSE) → cycle 1 EST scenario_2_cross."""
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 20, 1.15, 1, 0, 1), df,
                              bos_idx=10, bos_price=1.0, reversal_confirmed_idx=_RV)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 25, 1.15, 1, 0, 1, cts_anchor_idx=20))
    return tracker.on_cts_established(
        _ev("CTS_ESTABLISHED", 40, 1.35, 1, 1, 1), df,
        bos_idx=30, bos_price=1.2, reversal_confirmed_idx=_RV,
        prev_bos_outer=None, prev_sd=None,
    )


def _insts():
    return [
        ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
        ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10),
        ImbalanceInstance(55, 55, 1, 1.50, 1.40, 0.10),
    ]


def test_scenario2_cross_stored_at_versioned_key():
    df = _df(80, _insts())
    tracker = _tracker()
    cross = _drive_to_scenario2_cross(tracker, df)

    assert cross is not None
    # Stored at the versioned key, NOT the single key.
    assert (1, 1, "cross", 0) in tracker._fibs
    assert (1, 1) not in tracker._fibs
    assert tracker._cross_version[(1, 1)] == 0
    # Cross anchors: BOS_0 -> CTS_1
    assert cross.bos_idx == 10 and cross.cts_idx == 40
    # Meta preserved exactly — no version/fib_mode keys (byte-identical CSV).
    assert cross.meta.get("cross_cycle") is True
    assert cross.meta.get("scenario") == 2
    assert cross.meta.get("cycle1_bos_idx") == 30
    assert "version" not in cross.meta
    assert "fib_mode" not in cross.meta
    # Scratch fib slots retired; cycle0 decision input kept.
    assert "cross_cycle" not in tracker._cross_cycle_data[1]
    assert "normal_cycle1" not in tracker._cross_cycle_data[1]
    assert "cycle0" in tracker._cross_cycle_data[1]


def test_cross_extends_in_place_on_update():
    df = _df(80, _insts())
    tracker = _tracker()
    _drive_to_scenario2_cross(tracker, df)

    tracker.on_cts_updated(_ev("CTS_UPDATED", 45, 1.40, 1, 1, 1), df)

    latest = tracker._get_latest_cross(1, 1)
    assert latest is not None
    key, fib = latest
    assert key == (1, 1, "cross", 0)        # same version, in place
    assert fib.cts_idx == 45 and fib.active
    assert tracker._cross_version[(1, 1)] == 0
    assert (1, 1) not in tracker._fibs       # still no single


def test_confirm_locks_versioned_cross():
    df = _df(80, _insts())
    tracker = _tracker()
    _drive_to_scenario2_cross(tracker, df)
    tracker.on_cts_updated(_ev("CTS_UPDATED", 45, 1.40, 1, 1, 1), df)

    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 48, 1.40, 1, 1, 1, cts_anchor_idx=45))

    fib = tracker._fibs[(1, 1, "cross", 0)]
    assert fib.locked is True
    assert fib.meta.get("locked_at") == 48


def test_next_cycle_obsoletes_versioned_cross():
    """Cycle-2 activation must still set the new_cycle terminal on the cross even
    though it now lives at a versioned key (the additive _activate_fib obsolete)."""
    df = _df(80, _insts())
    tracker = _tracker()
    _drive_to_scenario2_cross(tracker, df)
    tracker.on_cts_confirmed(_ev("CTS_CONFIRMED", 48, 1.35, 1, 1, 1, cts_anchor_idx=40))

    # Cycle 2 (plain single) activates at idx 60 → obsoletes cycle-1 cross.
    tracker.on_cts_established(_ev("CTS_ESTABLISHED", 60, 1.55, 1, 2, 1), df,
                              bos_idx=50, bos_price=1.4, reversal_confirmed_idx=_RV)

    cross = tracker._fibs[(1, 1, "cross", 0)]
    assert cross.meta.get("obsolete_reason") == "new_cycle"
    assert tracker._terminal[(1, 1)] == (60, "new_cycle")
    # Cycle 2 single created at the normal key.
    assert (1, 2) in tracker._fibs


def test_cross_deactivates_when_own_imbalance_fills_no_single():
    """cond2 (cycle-1 own) fills → cross deactivates; no single is created
    (FALLBACK-to-normal is unreachable for h1: cond2 == the normal's own check)."""
    df = _df(80, _insts())
    df.at[42, "l"] = 1.22   # fills cycle-1 imbalance@35 within (35, 45]
    tracker = _tracker()
    _drive_to_scenario2_cross(tracker, df)

    tracker.on_cts_updated(_ev("CTS_UPDATED", 45, 1.40, 1, 1, 1), df)

    cross = tracker._fibs[(1, 1, "cross", 0)]
    assert cross.active is False
    assert cross.meta.get("deactivated_at") == 45
    assert (1, 1) not in tracker._fibs   # no fallback single materialized


def test_scenario3_single_unchanged_at_single_key():
    """Scenario 3 (no cross) still stores a plain single at (sid,1)."""
    # Only cycle-1 has an imbalance; cycle 0 has none → cond2/cond3 false → S3.
    df = _df(80, [ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10)])
    tracker = _tracker()
    res = _drive_to_scenario2_cross(tracker, df)

    assert res is not None
    assert (1, 1) in tracker._fibs                  # plain single key
    assert tracker._get_latest_cross(1, 1) is None  # no versioned cross
    assert res.meta.get("scenario") == 3
    assert res.bos_idx == 30 and res.cts_idx == 40  # intra anchors (BOS_1->CTS_1)
