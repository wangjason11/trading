"""Unit tests for `multitf/entity_df_mutation.py`.

c.i landing scope: validates `_tag_old_sid_on_overwrite` cascade
mechanics on a synthetic entity df with two fabricated sids. The
production pilot path doesn't exercise the cascade (c.i pilot has no
prior sid to overwrite), so this test is the sole validation that the
cascade machinery works before c.ii lands.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import KLZone, WVMIRecord
from engine_v2.zones.poi_zones import POIZone
from engine_v2.zones.fib_tracker import FibState
from engine_v2.multitf.entity_df_mutation import _tag_old_sid_on_overwrite


def _make_entity_df_with_sids() -> pd.DataFrame:
    """Build a minimal entity df with synthetic 2-sid state on attrs."""
    times = pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "o": [1.0] * 20,
        "h": [1.1] * 20,
        "l": [0.9] * 20,
        "c": [1.0] * 20,
        "volume": [100.0] * 20,
    })

    t0, t5, t10, t15 = times[0], times[5], times[10], times[15]

    # sid 0: open KL zone (no end_time), POI zone with end_time after boundary,
    # active-unlocked Fib, open WVMI record
    df.attrs["kl_zones"] = [
        KLZone(
            start_time=t0, end_time=None, side="buy",
            top=1.05, bottom=0.95, source_kind="BOS",
            source_time=t0, source_price=1.0,
            meta={"entity_sid": 0, "structure_id": 0, "cycle_id": 0, "active": True},
        ),
        # sid 1: open KL zone — should NOT be tagged
        KLZone(
            start_time=t10, end_time=None, side="buy",
            top=1.05, bottom=0.95, source_kind="BOS",
            source_time=t10, source_price=1.0,
            meta={"entity_sid": 1, "structure_id": 0, "cycle_id": 0, "active": True},
        ),
    ]
    df.attrs["poi_zones"] = [
        POIZone(
            start_time=t0, end_time=t15, side="buy",
            top=1.05, bottom=0.95,
            ic_idx=2,
            meta={"entity_sid": 0, "structure_id": 0, "active": True},
        ),
    ]
    df.attrs["fib_states"] = [
        FibState(
            structure_id=0, cycle_id=0, struct_direction=1,
            bos_idx=2, bos_price=1.0,
            cts_idx=4, cts_price=0.95,
            active=True, locked=False,
            meta={"entity_sid": 0},
        ),
        # locked fib — active stays True per spec; just deactivated_by added
        FibState(
            structure_id=0, cycle_id=1, struct_direction=1,
            bos_idx=6, bos_price=1.05,
            cts_idx=8, cts_price=1.0,
            active=True, locked=True,
            meta={"entity_sid": 0},
        ),
    ]
    df.attrs["wvmi"] = [
        WVMIRecord(
            bos_structure_id=0, bos_cycle_id=0, zone_side="buy",
            lp_locked=False,
            meta={"entity_sid": 0},
        ),
        WVMIRecord(
            bos_structure_id=0, bos_cycle_id=1, zone_side="buy",
            lp_locked=True,  # already locked — should not be re-locked
            meta={"entity_sid": 0, "lp_locked_by": "BOS_CONFIRMED"},
        ),
    ]
    return df


def test_cascade_tags_old_sid_kl_zones_and_caps_bounds():
    df = _make_entity_df_with_sids()
    boundary_idx = 10
    boundary_time = pd.to_datetime(df.loc[boundary_idx, "time"], utc=True)

    _tag_old_sid_on_overwrite(df, prior_sid_id=0, new_sid_id=1, boundary_idx=boundary_idx)

    kls = df.attrs["kl_zones"]
    # sid 0 zone: tagged, end_time capped at boundary
    assert kls[0].meta["deactivated_by"] == "overwritten_by_sid_1"
    assert kls[0].meta["active"] is False
    assert kls[0].end_time == boundary_time
    # sid 1 zone: untouched
    assert "deactivated_by" not in kls[1].meta
    assert kls[1].end_time is None
    assert kls[1].meta["active"] is True


def test_cascade_caps_poi_zone_end_time_only_if_after_boundary():
    df = _make_entity_df_with_sids()
    boundary_idx = 10  # boundary at t10

    _tag_old_sid_on_overwrite(df, prior_sid_id=0, new_sid_id=1, boundary_idx=boundary_idx)

    poi = df.attrs["poi_zones"][0]
    boundary_time = pd.to_datetime(df.loc[boundary_idx, "time"], utc=True)
    assert poi.meta["deactivated_by"] == "overwritten_by_sid_1"
    assert poi.meta["active"] is False
    # POI's end_time was t15 (after boundary t10) → should be capped at t10
    assert poi.end_time == boundary_time


def test_cascade_caps_poi_zone_only_open_or_late_end_times():
    """POI zones with end_time BEFORE boundary should keep their end_time."""
    df = _make_entity_df_with_sids()
    # Edit the POI to end BEFORE boundary
    early_end = pd.to_datetime(df.loc[5, "time"], utc=True)
    df.attrs["poi_zones"][0] = POIZone(
        start_time=df.attrs["poi_zones"][0].start_time,
        end_time=early_end,
        side="buy", top=1.05, bottom=0.95,
        ic_idx=2,
        meta={"entity_sid": 0, "active": True},
    )
    boundary_idx = 10

    _tag_old_sid_on_overwrite(df, prior_sid_id=0, new_sid_id=1, boundary_idx=boundary_idx)

    poi = df.attrs["poi_zones"][0]
    # deactivated_by tag still applied even though bounds don't change
    assert poi.meta["deactivated_by"] == "overwritten_by_sid_1"
    # end_time stays at the earlier value
    assert poi.end_time == early_end


def test_cascade_flips_active_unlocked_fib_to_inactive():
    df = _make_entity_df_with_sids()
    boundary_idx = 10

    _tag_old_sid_on_overwrite(df, prior_sid_id=0, new_sid_id=1, boundary_idx=boundary_idx)

    fibs = df.attrs["fib_states"]
    # Active-unlocked fib: flips to inactive
    assert fibs[0].active is False
    assert fibs[0].meta["deactivated_by"] == "overwritten_by_sid_1"
    assert fibs[0].meta["deactivated_at"] == boundary_idx
    # Locked fib: stays active (the lock is the truth), still gets tag
    assert fibs[1].active is True
    assert fibs[1].locked is True
    assert fibs[1].meta["deactivated_by"] == "overwritten_by_sid_1"


def test_cascade_locks_open_wvmi_with_overwrite_reason():
    df = _make_entity_df_with_sids()
    boundary_idx = 10

    _tag_old_sid_on_overwrite(df, prior_sid_id=0, new_sid_id=1, boundary_idx=boundary_idx)

    wvmis = df.attrs["wvmi"]
    # Open WVMI record: locked with overwrite reason
    assert wvmis[0].lp_locked is True
    assert wvmis[0].meta["lp_locked_by"] == "overwritten_by_sid_1"
    # Already-locked WVMI record: lock_by reason unchanged (the original lock
    # reason is preserved — overwrite doesn't double-stamp)
    assert wvmis[1].lp_locked is True
    assert wvmis[1].meta["lp_locked_by"] == "BOS_CONFIRMED"


def test_cascade_only_affects_prior_sid():
    """Prior_sid_id=0 cascade leaves sid=1 snapshots completely untouched."""
    df = _make_entity_df_with_sids()
    boundary_idx = 10

    _tag_old_sid_on_overwrite(df, prior_sid_id=0, new_sid_id=1, boundary_idx=boundary_idx)

    # sid 1 KL zone (only one in fixture)
    sid1_zone = df.attrs["kl_zones"][1]
    assert sid1_zone.meta.get("entity_sid") == 1
    assert "deactivated_by" not in sid1_zone.meta
    assert sid1_zone.end_time is None
    assert sid1_zone.meta["active"] is True


def test_cascade_no_op_when_attrs_lists_missing():
    """Cascade tolerates missing snapshot lists on attrs."""
    times = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "o": [1.0] * 5, "h": [1.1] * 5, "l": [0.9] * 5, "c": [1.0] * 5,
        "volume": [100.0] * 5,
    })
    # No attrs lists at all
    _tag_old_sid_on_overwrite(df, prior_sid_id=0, new_sid_id=1, boundary_idx=2)
    # No exception raised; nothing to verify on the df itself
