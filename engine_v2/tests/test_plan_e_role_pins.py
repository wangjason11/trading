"""Plan E E2b role pins (landing-review mutation lens): each site that E2b split
into a LOCATION (the anchor) and a TIME (today the anchor too, marked for an E3
stage) is pinned with an E4-shaped event (`idx` = the moment 12, anchor 9) so a
location read switched to the moment — or a raw `ev.idx` read — fails.

The TIME pins state TODAY's value (the anchor); the E3 stage named in the site's
marker flips them to the moment.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

import engine_v2
import engine_v2.zones.poi_zones as poi_zones
from engine_v2.common.types import ImbalanceInstance
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.tests.test_cross_cycle_fib_routine import _BOS, _CTS, _INST0, _INST1
from engine_v2.tests.test_cross_cycle_fib_routine import _df as _routine_df
from engine_v2.tests.test_imbalance_c3_knowability import _df, _gap, _quiet, _tracker
from engine_v2.zones.cross_cycle_fib import resolve_cross_cycle_eligibility
from engine_v2.zones.fib_tracker import select_fib_anchor_for_cycle


def _e4_est(anchor=20, moment=22, **kw):
    return make_cts_established(cts_anchor_idx=anchor, confirmed_at=moment, idx=moment,
                                price=1.2, **kw)


# --- FibTracker EST: location vs time -------------------------------------------

@pytest.mark.illegal_event_contract
@pytest.mark.parametrize("mode", ["h1", "cross_cycle"])
def test_fib_tracker_est_reads_the_anchor_for_the_fib_and_today_for_activated_at(mode):
    tracker = _tracker(mode)
    df = _df(40, [_gap(15)])
    # sid 0 cycle 1 (h1 simple flow activates at cycle >= 1); cross_cycle cycle 0.
    cyc = 1 if mode == "h1" else 0
    fib = _quiet(tracker.on_cts_established, _e4_est(structure_id=0, cycle_id=cyc), df,
                 bos_idx=10, bos_price=0.9)
    assert fib is not None and fib.active
    assert fib.cts_idx == 20                        # LOCATION: the CTS anchor, never the moment
    assert fib.meta["activated_at"] == 20           # TIME: today the anchor; Plan E E3a → 22


# --- the shared routine / anchor selector: window vs horizon --------------------

def test_routine_own_window_end_and_fill_horizon_are_independent():
    """Window [30, 40] (location), horizon 38 (time): the own gap is filled only
    at 39, so asked at 38 it is still unfilled."""
    df = _routine_df(60, [_INST0, _INST1])
    df.at[39, "l"] = 1.22
    e = resolve_cross_cycle_eligibility(
        df=df, target_cycle=1, sd=1, own_window_end_idx=40, fill_horizon_idx=38, own_imb_start=30,
        anchor_idx=40, anchor_price=1.35, bos_by_cycle={0: _BOS[0]}, cts_by_cycle={0: _CTS[0]},
        dead_cycles=set(), fill_threshold=0.70, fill_as_of="snapshot",
        prior_cached_liveness={0: True}, evaluated_at=None, snapshot_horizon_idx=30)
    assert e.own_has is True


def test_anchor_selector_asks_cond3_at_the_snapshot_horizon():
    """Cycle 0's gap fills at 33: after BOS_1 (30), before CTS_1 (40). cond3 is
    asked at the snapshot horizon (BOS_1) → not yet filled → the cross forms."""
    df = _routine_df(60, [_INST0, _INST1])
    df.at[33, "l"] = 1.02
    c0 = {"bos_idx": 10, "bos_price": 1.0, "cts_idx": 20, "cts_price": 1.15, "has_unfilled": True}
    r = select_fib_anchor_for_cycle(df, 1, 1, 30, 1.2, 40, 1.35, c0, 0.70, struct_direction=1,
                                    evaluated_at=None, fill_horizon_idx=40, snapshot_horizon_idx=30)
    assert r[0] == 10 and r[-1] == "scenario_2_cross"


# --- POI sweep: cond1 is a location -----------------------------------------------

def _poi_history(anchor, moment=7):
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([
        {"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
        for i in range(14)
    ])
    # IC at 6, inside the 61.8-80 band of (BOS 0.6000, CTS 0.6100).
    df.loc[6, ["o", "h", "l", "c"]] = [0.6034, 0.6035, 0.6025, 0.6026]
    imb = ImbalanceInstance(start_idx=8, end_idx=8, direction=1,     # formed at 9, after the IC
                            gap_top=0.6090, gap_bottom=0.6080, gap_size=0.0010)
    cts = make_cts_established(cts_anchor_idx=anchor, confirmed_at=moment, idx=moment,
                               price=0.6100, struct_direction=None)
    return poi_zones._compute_poi_activation_history(
        df, ic_idx=6, cts_established_idx=moment, sd=1, scan_end=12, fill_threshold=0.70,
        bos_price=0.6000, cts_events=[cts], fib_min_pct=61.8, fib_max_pct=80.0,
        variant_thresholds={"V30": 0.3, "V60": 0.6, "V90": 0.9},
        imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=None,
    )


@pytest.mark.illegal_event_contract
def test_poi_sweep_cond1_reads_the_cts_anchor():
    """An IC (6) past the CTS anchor (5) but before the moment (7): cond1 ("the
    IC lies inside the fib", `cts_at_t >= ic`) reads the ANCHOR → never active.
    A moment read would activate it at 9."""
    assert _poi_history(anchor=5) == []


@pytest.mark.illegal_event_contract
def test_poi_sweep_positive_control():
    """The same fixture with the anchor at the IC activates (so the pin above is not vacuous)."""
    hist = _poi_history(anchor=6)
    assert hist and hist[0]["idx"] == 9 and hist[0]["active"] is True


# --- source guard: the pinned sort call sites -----------------------------------

_PINNED = ("pipeline/orchestrator.py", "multitf/sub_wvmi.py", "zones/zone_proximity.py",
           "zones/poi_zones.py", "zones/wave_candles.py", "structure/reference_zone.py",
           "structure/unified_probe.py")


def test_no_pinned_module_sorts_events_on_a_raw_idx():
    """A sort / max key that reads `.idx` of an event would undo the E2 pins (the
    E4 flip would reorder). Keys must go through `ef.` (processing_order_key /
    stamped_idx / an anchor accessor)."""
    root = Path(engine_v2.__file__).parent
    offenders = []
    for rel in _PINNED:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            name = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            if name not in ("sort", "sorted", "max", "min"):
                continue
            for kw in n.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Lambda):
                    if any(isinstance(a, ast.Attribute) and a.attr == "idx" for a in ast.walk(kw.value.body)):
                        offenders.append(f"{rel}:{n.lineno}")
    assert offenders == []
