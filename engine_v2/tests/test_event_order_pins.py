"""Plan E E2b pins: the event processing order and the CTS reference pick are
frozen against the E4 flip (PLAN_E §2 item 4, §6.4; PLAN_E_inputs §2.5 hazards
H1–H5, risk R1).

Each case is built in BOTH shapes — today's (`ev.idx` = the anchor) and E4's
(`ev.idx` = the moment, `confirmed_at`) — and must give the same answer. The E4
shape breaks the current contract, hence `illegal_event_contract`.
"""
from __future__ import annotations

import pytest

import engine_v2.structure.reference_zone as rz
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established

pytestmark = pytest.mark.illegal_event_contract

SHAPES = ["today", "e4"]


def _est(anchor, moment, shape, **kw):
    return make_cts_established(cts_anchor_idx=anchor, confirmed_at=moment,
                                idx=anchor if shape == "today" else moment, **kw)


def _bos(anchor, moment, shape, **kw):
    return make_bos_confirmed(bos_anchor_idx=anchor, confirmed_at=moment,
                              idx=anchor if shape == "today" else moment, **kw)


def _ev(etype, idx, category="STRUCTURE", **meta):
    return StructureEvent(idx=idx, category=category, type=etype, price=1.0, meta=meta)


def _order(evs):
    return [e.type for e in sorted(evs, key=ef.processing_order_key)]


@pytest.mark.parametrize("est_shape,bos_shape", [
    ("today", "today"), ("e4", "e4"),
    ("today", "e4"),   # E4b before E4a (BOS-only flip): a raw (idx, type) sort puts EST (9) first
    ("e4", "today"),
])
def test_h1_bos_before_est_of_the_same_cycle(est_shape, bos_shape):
    """The fib loop needs BOS(S,C) before EST(S,C); MS emits EST first. The
    anchors keep BOS (3) before EST (9) in every flip order (E4a / E4b land in
    either order, PLAN_E §3)."""
    evs = [_est(9, 10, est_shape), _bos(3, 10, bos_shape)]
    assert _order(evs) == ["BOS_CONFIRMED", "CTS_ESTABLISHED"]


@pytest.mark.parametrize("shape", SHAPES)
def test_h2_est_before_a_confirmation_on_its_moment(shape):
    """An sd-prox CTS_CONFIRMED can land ON the EST moment; "CTS_CONFIRMED" <
    "CTS_ESTABLISHED" would run it first and the fib would never lock."""
    evs = [_ev("CTS_CONFIRMED", 10, cts_anchor_idx=9, confirmed_at=10), _est(9, 10, shape)]
    assert _order(evs) == ["CTS_ESTABLISHED", "CTS_CONFIRMED"]


@pytest.mark.parametrize("shape", SHAPES)
def test_h3_prior_threshold_update_between_anchor_and_moment(shape):
    """A CTS_THRESHOLD_UPDATED in [EST anchor, moment) stays AFTER the EST (it
    would otherwise dispatch while the cycle is still `pre_established`)."""
    evs = [_ev("CTS_THRESHOLD_UPDATED", 9), _est(8, 10, shape)]
    assert _order(evs) == ["CTS_ESTABLISHED", "CTS_THRESHOLD_UPDATED"]


@pytest.fixture
def _stub_ad_hoc(monkeypatch):
    """The ad-hoc zone geometry is not under test: record the base candle."""
    seen = []

    def _derive(df, extreme_idx, source_sd):
        seen.append(int(extreme_idx))
        return (1.0, 0.9, "sell")

    monkeypatch.setattr(rz, "_derive_cts_zone_ad_hoc", _derive)
    return seen


@pytest.mark.parametrize("shape", SHAPES)
def test_l1_est_winner_with_lag_feeds_the_anchor_to_the_probe(shape, _stub_ad_hoc):
    """R1: the reference's `source_event_idx` is the probe input and the pool key.
    An EST winner whose anchor (9) precedes its moment (10) must give 9 — today
    and after E4 (invisible on the reference window: no probe used an EST)."""
    zone = rz.build_reference_zone_from_cts_event(
        [_bos(3, 10, shape), _est(9, 10, shape)], [], None, sid=0, probe_direction=-1,
    )
    assert zone.source == "cts_established"
    assert zone.source_event_idx == 9
    assert _stub_ad_hoc == [9]


@pytest.mark.parametrize("shape", SHAPES)
def test_h5_reference_recency_tie_with_a_confirmation(shape, _stub_ad_hoc):
    """H5: an EST (anchor 9, moment 10) and a CTS_CONFIRMED at 10. Today the EST
    stamps at 9, so the CONFIRMED is the most recent; pinned through E4 (Plan E
    E3b moves the recency key to the moment, where the CONFIRMED > ESTABLISHED
    tie-break decides the same way)."""
    conf = _ev("CTS_CONFIRMED", 10, structure_id=0, cycle_id=0, cts_anchor_idx=9, confirmed_at=10)
    zone = rz.build_reference_zone_from_cts_event(
        [_est(9, 10, shape), conf], [], None, sid=0, probe_direction=-1,
    )
    assert zone.source == "cts_confirmed"
    assert zone.source_event_idx == 9


@pytest.mark.parametrize("shape", SHAPES)
def test_reference_window_filters_on_the_stamped_idx_until_e3b(shape, _stub_ad_hoc):
    """The sibling window is a TIME filter, still on the stamped idx (the anchor
    for EST) until Plan E E3b: an EST anchored at 9 (moment 12) is inside
    [5, 10] today and after E4."""
    zone = rz.build_reference_zone_from_cts_event(
        [_est(9, 12, shape)], [], None, sid=0, probe_direction=-1, idx_window=(5, 10),
    )
    assert zone is not None and zone.source_event_idx == 9
