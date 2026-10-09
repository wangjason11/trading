"""Unit tests for the shared cross-cycle eligibility routine.

`resolve_cross_cycle_eligibility` (zones/cross_cycle_fib.py) is the §11a
"extract": the body of FibTracker._cross_check steps 1-3. These tests pin it
directly (no FibTracker): the own-imbalance test and the dead-cycle backward
walk, each prior cycle checked to the current candle. (The H1-main "snapshot"
fill-as-of policy — Scenario-2 cond1 / cond2 / cond3 at target 1 — and its
truth-table tests were retired with MS's in-flight resolver, its last reader,
in PLAN_I S5; CROSS_CYCLE_FIB_SPEC.md Appendix A.)

Fill fixtures mirror test_cross_cycle_fib.py: bullish gaps with gap_top <= the
default close (1.5), so stroke 2 (close >= gap_top) is always satisfied and a
fill is driven purely by a stroke-1 low override inside the scan window
(inst.end_idx, check_to_idx].
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import ImbalanceInstance
from engine_v2.zones.cross_cycle_fib import (
    CrossEligibility,
    resolve_cross_cycle_eligibility,
)


# ---------- helpers ----------

def _df(n_candles: int, instances: list[ImbalanceInstance]) -> pd.DataFrame:
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


# Standard cycle anchors used across tests:
#   cycle 0: BOS_0=10 CTS_0=20   imbalance at 15  (gap 1.00-1.10, fill low<=1.03)
#   cycle 1: BOS_1=30 CTS_1=40   imbalance at 35  (gap 1.20-1.30, fill low<=1.23)
#   cycle 2: BOS_2=50 CTS_2=60   imbalance at 55  (gap 1.40-1.50, fill low<=1.43)
_INST0 = ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10)
_INST1 = ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10)
_INST2 = ImbalanceInstance(55, 55, 1, 1.50, 1.40, 0.10)

_BOS = {0: (10, 1.0), 1: (30, 1.2), 2: (50, 1.4)}
_CTS = {0: (20, 1.15), 1: (40, 1.35), 2: (60, 1.55)}


def _resolve_current(df, target_cycle, current_candle, own_imb_start,
                     anchor_idx, anchor_price, dead=None):
    """Subordinate-mode resolve over cycles [0, target_cycle)."""
    return resolve_cross_cycle_eligibility(
        df=df, target_cycle=target_cycle, sd=1, own_window_end_idx=current_candle, fill_horizon_idx=current_candle,
        own_imb_start=own_imb_start, anchor_idx=anchor_idx, anchor_price=anchor_price,
        bos_by_cycle={k: _BOS[k] for k in range(target_cycle)},
        cts_by_cycle={k: _CTS[k] for k in range(target_cycle)},
        dead_cycles=set() if dead is None else dead,
        fill_threshold=0.70, evaluated_at=None,
    )


# ---------- a prior cycle filled inside the target's window is dead (the current candle) ----------

def test_current_prior_cycle_filled_inside_the_target_window_is_dead():
    """Cycle-0 imbalance is unfilled as of BOS_1 (=30) but fills in the
    BOS_1->CTS_1 window (idx 33): checked to the current candle (40) it is dead
    → no cross. (Re-derived PLAN_I S5 from `test_snapshot_vs_current_superset_case`,
    whose retired snapshot half kept the cross by checking @30.)"""
    df = _df(60, [_INST0, _INST1])
    df.at[33, "l"] = 1.02  # in (15, 40] but not (15, 30]

    cur = _resolve_current(df, target_cycle=1, current_candle=40, own_imb_start=30,
                           anchor_idx=40, anchor_price=1.35)
    assert cur.own_has is True
    assert cur.crosses is False
    assert cur.earliest_x == 1


# ---------- current-mode dead-cycle backward walk ----------

def test_current_multi_cycle_all_live_crosses_from_zero():
    df = _df(80, [_INST0, _INST1, _INST2])
    e = _resolve_current(df, target_cycle=2, current_candle=60, own_imb_start=50,
                         anchor_idx=60, anchor_price=1.55)
    assert e.crosses is True
    assert e.earliest_x == 0
    assert (e.bos_idx, e.cts_idx) == (10, 60)  # BOS_0 -> CTS_2


def test_current_walk_stops_at_dead_cycle_and_shrinks_start():
    # Fill cycle 0 (as of current=60) → walk stops at k=0 → earliest_x=1
    df = _df(80, [_INST0, _INST1, _INST2])
    df.at[22, "l"] = 1.02
    dead: set = set()
    e = _resolve_current(df, target_cycle=2, current_candle=60, own_imb_start=50,
                         anchor_idx=60, anchor_price=1.55, dead=dead)
    assert e.crosses is True
    assert e.earliest_x == 1
    assert e.bos_idx == 30  # BOS_1
    assert 0 in dead       # walk memoized cycle 0 as dead (mutated in place)


def test_current_no_eligible_prior_no_cross():
    # Fill BOTH prior cycles → earliest_x == target → no cross
    df = _df(80, [_INST0, _INST1, _INST2])
    df.at[22, "l"] = 1.02   # cycle 0 dead
    df.at[38, "l"] = 1.22   # cycle 1 dead (in (35, 60])
    e = _resolve_current(df, target_cycle=2, current_candle=60, own_imb_start=50,
                         anchor_idx=60, anchor_price=1.55)
    assert e.own_has is True
    assert e.crosses is False
    assert e.earliest_x == 2


def test_current_own_imbalance_filled_gates_out():
    # Fill the target cycle's own imbalance → own_has False, no walk needed
    df = _df(80, [_INST0, _INST1, _INST2])
    df.at[57, "l"] = 1.42   # cycle 2 own gap filled (in (55, 60])
    e = _resolve_current(df, target_cycle=2, current_candle=60, own_imb_start=50,
                         anchor_idx=60, anchor_price=1.55)
    assert e.own_has is False
    assert e.crosses is False
    # anchors mirror the CTS-side inputs when own fails
    assert (e.bos_idx, e.cts_idx) == (60, 60)


def test_current_prepopulated_dead_cache_short_circuits_walk():
    # cycle 1 pre-seeded dead → walk breaks immediately at k=1 → no cross
    df = _df(80, [_INST0, _INST1, _INST2])
    e = _resolve_current(df, target_cycle=2, current_candle=60, own_imb_start=50,
                         anchor_idx=60, anchor_price=1.55, dead={1})
    assert e.own_has is True
    assert e.crosses is False
    assert e.earliest_x == 2


def test_current_missing_cycle_data_stops_walk():
    # No bos/cts maps for any prior cycle → walk stops, no cross
    df = _df(80, [_INST1])
    e = resolve_cross_cycle_eligibility(
        df=df, target_cycle=1, sd=1, own_window_end_idx=40, fill_horizon_idx=40, own_imb_start=30,
        anchor_idx=40, anchor_price=1.35,
        bos_by_cycle={}, cts_by_cycle={}, dead_cycles=set(),
        fill_threshold=0.70, evaluated_at=None,
    )
    assert e.own_has is True
    assert e.crosses is False
    assert e.earliest_x == 1


# ---------- direction filter ----------

def test_direction_filter_ignores_counter_direction_imbalance():
    # A bearish imbalance in cycle 1's range must NOT satisfy a sd=+1 own-test.
    bearish = ImbalanceInstance(35, 35, -1, 1.30, 1.20, 0.10)
    df = _df(60, [bearish])
    e = resolve_cross_cycle_eligibility(
        df=df, target_cycle=1, sd=1, own_window_end_idx=40, fill_horizon_idx=40, own_imb_start=30,
        anchor_idx=40, anchor_price=1.35,
        bos_by_cycle={0: _BOS[0]}, cts_by_cycle={0: _CTS[0]},
        dead_cycles=set(), fill_threshold=0.70, evaluated_at=None,
    )
    assert e.own_has is False  # counter-direction imbalance filtered out


# ---------- knowability: the own-imbalance test is asked at the moment (Plan F) ----------

def _resolve_own(df, evaluated_at):
    return resolve_cross_cycle_eligibility(
        df=df, target_cycle=1, sd=1, own_window_end_idx=40, fill_horizon_idx=40, own_imb_start=30,
        anchor_idx=40, anchor_price=1.35,
        bos_by_cycle={0: _BOS[0]}, cts_by_cycle={0: _CTS[0]},
        dead_cycles=set(), fill_threshold=0.70,
        evaluated_at=evaluated_at,
    )


def test_own_gap_not_formed_at_the_moment_is_not_counted():
    # The only own gap has c2 == current_candle (40): its c3 is 41.
    df = _df(60, [_INST0, ImbalanceInstance(40, 40, 1, 1.30, 1.20, 0.10)])
    assert _resolve_own(df, evaluated_at=40).own_has is False
    assert _resolve_own(df, evaluated_at=41).own_has is True
    assert _resolve_own(df, evaluated_at=None).own_has is True   # uncut (no production caller since PLAN_I S5)


def test_evaluated_at_is_required_on_the_routine():
    """(The anchor selector's half retired with it, PLAN_I S5.)"""
    df = _df(60, [_INST0, _INST1])
    with pytest.raises(TypeError, match="evaluated_at"):
        resolve_cross_cycle_eligibility(
            df=df, target_cycle=1, sd=1, own_window_end_idx=40, fill_horizon_idx=40, own_imb_start=30,
            anchor_idx=40, anchor_price=1.35, bos_by_cycle={}, cts_by_cycle={},
            dead_cycles=set(), fill_threshold=0.70,
        )


@pytest.mark.parametrize("retired", [{"fill_as_of": "current"}, {"prior_cached_liveness": {0: True}},
                                     {"snapshot_horizon_idx": None}, {"target_ceiling": None}])
def test_the_snapshot_policy_parameters_are_gone(retired):
    """PLAN_I S5 deleted the routine's "snapshot" fill-as-of policy with MS's resolver (CROSS_CYCLE_FIB_SPEC §7)."""
    df = _df(60, [_INST0, _INST1])
    with pytest.raises(TypeError):
        resolve_cross_cycle_eligibility(
            df=df, target_cycle=1, sd=1, own_window_end_idx=40, fill_horizon_idx=40, own_imb_start=30,
            anchor_idx=40, anchor_price=1.35, bos_by_cycle={}, cts_by_cycle={},
            dead_cycles=set(), fill_threshold=0.70, evaluated_at=None, **retired,
        )
