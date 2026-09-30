"""Unit tests for zones/poi_lifecycle.py — per-candle POI activation walk."""
from __future__ import annotations

from types import SimpleNamespace

from engine_v2.zones.poi_lifecycle import (
    active_stretches_from_history,
    poi_active_as_of,
    poi_confirmed_idx_as_of,
)


def _zone(history, end_idx=None):
    return SimpleNamespace(meta={"activation_history": history, "end_idx": end_idx})


# The real sid1 cyc2 POI(0.5772) history that surfaced the bug — as of save
# 0a4eadc, pre-c3 (Plan F moved the re-activations 953→954 and 997→998). A
# fixture for the history walk; the values need not track the engine.
REAL = [
    {"idx": 905, "active": True},
    {"idx": 952, "active": False},
    {"idx": 953, "active": True},
    {"idx": 992, "active": False},
    {"idx": 997, "active": True},
]


def test_active_stretches_pairs_flaps():
    # open_end past the last event -> trailing active extends to open_end.
    assert active_stretches_from_history(REAL, open_end_idx=1100) == [
        (905, 951), (953, 991), (997, 1100),
    ]


def test_active_stretches_truncates_at_open_end():
    # open_end=926 lands inside the first active stretch -> single stretch
    # closed at 926 (the bug candle).
    assert active_stretches_from_history(REAL, open_end_idx=926) == [(905, 926)]


def test_active_as_of_covers_first_stretch():
    z = _zone(REAL)
    assert poi_active_as_of(z, 926) is True   # inside [905,951]
    assert poi_active_as_of(z, 905) is True   # activate candle itself
    assert poi_active_as_of(z, 951) is True   # last active before deactivate


def test_active_as_of_false_in_gap_and_on_deactivate():
    z = _zone(REAL)
    assert poi_active_as_of(z, 952) is False  # deactivation candle -> inactive
    assert poi_active_as_of(z, 992) is False  # second deactivation candle
    assert poi_active_as_of(z, 996) is False  # mid second gap


def test_active_as_of_reactivation_and_before_first():
    z = _zone(REAL)
    assert poi_active_as_of(z, 997) is True   # reactivation
    assert poi_active_as_of(z, 1050) is True  # trailing active, still open
    assert poi_active_as_of(z, 904) is False  # before first activate


def test_end_idx_hard_caps_activity():
    z = _zone(REAL, end_idx=960)
    assert poi_active_as_of(z, 955) is True
    assert poi_active_as_of(z, 997) is False  # past lifecycle end


def test_confirmed_idx_as_of_is_stretch_start():
    z = _zone(REAL)
    # At 926, the "confirmed as of" idx is 905 (the stretch start), NOT the
    # collapsed-scalar 997.
    assert poi_confirmed_idx_as_of(z, 926) == 905
    assert poi_confirmed_idx_as_of(z, 970) == 953
    assert poi_confirmed_idx_as_of(z, 1000) == 997
    assert poi_confirmed_idx_as_of(z, 952) is None  # inactive -> no confirmed idx
    assert poi_confirmed_idx_as_of(z, 904) is None


def test_empty_history_never_active():
    z = _zone([])
    assert poi_active_as_of(z, 100) is False
    assert poi_confirmed_idx_as_of(z, 100) is None
    assert active_stretches_from_history([], open_end_idx=100) == []
