"""The conftest event-contract validator tests itself (Plan E E2a landing review:
it is the main safeguard of the emitters, so its checks are pinned)."""
from __future__ import annotations

import pytest

from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established
from engine_v2.tests.conftest import EventContractViolation, validate_event_contract


def _raw(etype, idx, **meta):
    return StructureEvent(idx=idx, category="STRUCTURE", type=etype, meta=meta)


@pytest.mark.illegal_event_contract
def test_accepts_a_legal_event():
    validate_event_contract(make_cts_established(cts_anchor_idx=9, confirmed_at=10))
    validate_event_contract(make_bos_confirmed(bos_anchor_idx=3, confirmed_at=10))
    validate_event_contract(_raw("CTS_CONFIRMED", 25))  # other types are not checked


@pytest.mark.illegal_event_contract
def test_rejects_idx_that_is_not_the_contract_index():
    """`idx` is the moment on both types (Plan E E4a CTS_ESTABLISHED, E4b
    BOS_CONFIRMED) — the pre-E4 shape (`idx` = the anchor) is rejected."""
    with pytest.raises(EventContractViolation, match=r"idx 9 != meta\['confirmed_at'\]"):
        validate_event_contract(make_cts_established(cts_anchor_idx=9, confirmed_at=10, idx=9))
    with pytest.raises(EventContractViolation, match=r"idx 3 != meta\['confirmed_at'\]"):
        validate_event_contract(make_bos_confirmed(bos_anchor_idx=3, confirmed_at=10, idx=3))


@pytest.mark.illegal_event_contract
def test_rejects_missing_keys():
    with pytest.raises(EventContractViolation, match="cts_anchor_idx"):
        validate_event_contract(_raw("CTS_ESTABLISHED", 9, confirmed_at=10))
    with pytest.raises(EventContractViolation, match="confirmed_at"):
        validate_event_contract(_raw("BOS_CONFIRMED", 3, bos_anchor_idx=3))


@pytest.mark.illegal_event_contract
@pytest.mark.parametrize("bad", [9.0, "9", True])
def test_rejects_non_int_values(bad):
    with pytest.raises(EventContractViolation, match="not an int"):
        validate_event_contract(_raw("CTS_ESTABLISHED", 9, confirmed_at=10, cts_anchor_idx=bad))


@pytest.mark.illegal_event_contract
def test_rejects_a_numpy_int():
    np = pytest.importorskip("numpy")
    with pytest.raises(EventContractViolation, match="not an int"):
        validate_event_contract(_raw("BOS_CONFIRMED", 3, confirmed_at=10, bos_anchor_idx=np.int64(3)))


def test_violation_escapes_except_exception():
    """A BaseException: neither `pytest.raises(Exception)` nor an engine
    `except Exception` catch-and-skip path can swallow it."""
    assert issubclass(EventContractViolation, BaseException)
    assert not issubclass(EventContractViolation, Exception)


def test_the_autouse_hook_is_active():
    """Unmarked test: constructing an illegal event raises at construction."""
    with pytest.raises(EventContractViolation):
        make_cts_established(cts_anchor_idx=9, confirmed_at=10, idx=9)


# --- Plan E E3·0: CTS_UPDATED (landing review) ----------------------------------

@pytest.mark.illegal_event_contract
def test_cts_updated_legal_shapes():
    validate_event_contract(_raw("CTS_UPDATED", 24, via="replay_raw"))
    validate_event_contract(_raw("CTS_UPDATED", 24, via="one_maru_opposite", confirmed_at=25))
    validate_event_contract(_raw("CTS_UPDATED", 24, via="continuous", confirmed_at=24))


@pytest.mark.illegal_event_contract
@pytest.mark.parametrize("meta", [
    {"via": "continuous"},                         # pattern path without its moment
    {"via": "continuous", "confirmed_at": 23},     # moment before the anchor
    {"via": "continuous", "confirmed_at": 25.0},   # not a Python int
    {"via": "replay_raw", "confirmed_at": 24},     # the raw path carries none
])
def test_cts_updated_rejects(meta):
    with pytest.raises(EventContractViolation, match="CTS_UPDATED"):
        validate_event_contract(_raw("CTS_UPDATED", 24, **meta))


@pytest.mark.illegal_event_contract
def test_cts_updated_requires_via():
    """Plan E E3b: every CTS_UPDATED reader asks `event_moment`, which needs `via`."""
    with pytest.raises(EventContractViolation, match="via"):
        validate_event_contract(_raw("CTS_UPDATED", 24))
