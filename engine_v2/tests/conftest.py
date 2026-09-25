"""Shared test hooks.

Event-contract validator (Plan E E2a; PLAN_E §6.1): every CTS_ESTABLISHED /
BOS_CONFIRMED constructed during a test — by a fixture or by the engine — must
carry `confirmed_at` and its anchor key (`cts_anchor_idx` / `bos_anchor_idx`) as
ints, and its `idx` must be the index the contract names for its type
(`EVENT_IDX_IS`). Build test events with `tests/_event_factory.py`.

A test that deliberately builds an illegal event takes
`@pytest.mark.illegal_event_contract`.

Limits (landing review, 2026-09-24): only `__init__` is checked — `deepcopy`
copies (the M15 mirror, the sibling clip, the pooled build) and meta edits made
after construction (e.g. `mutate=` hooks) bypass it, and so do events built in
module-scoped fixtures (set up before this function-scoped hook). The mirror is
pinned by `test_event_meta_idx_keys.test_anchor_keys_are_int_and_idx_is_the_contract_index`;
a test that edits event meta after construction calls `validate_event_contract`
itself (Plan E E4a review: the Plan D `mutate=` hooks do).
"""

from __future__ import annotations

import pytest

from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA

# What `ev.idx` is on CTS_ESTABLISHED / BOS_CONFIRMED, per type: "moment"
# (`idx == meta["confirmed_at"]`) or "anchor" (`idx == meta[<anchor key>]`).
# CTS_ESTABLISHED flipped to the moment in Plan E E4a, BOS_CONFIRMED in E4b —
# each together with the factory default (`tests/_event_factory.py`).
EVENT_IDX_IS = {"CTS_ESTABLISHED": "moment", "BOS_CONFIRMED": "moment"}

_ANCHOR_KEY = {"CTS_ESTABLISHED": "cts_anchor_idx", "BOS_CONFIRMED": "bos_anchor_idx"}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "illegal_event_contract: the test builds CTS_ESTABLISHED / BOS_CONFIRMED "
        "events that violate the event contract on purpose",
    )


class EventContractViolation(BaseException):
    """A BaseException on purpose: `pytest.raises(AssertionError / KeyError /
    Exception)` in a test, or an engine `except Exception` catch-and-skip path
    (e.g. a failed sub build), must not swallow a contract violation."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise EventContractViolation(msg)


def _is_int(v) -> bool:
    # A Python int exactly: `_shift_meta_indices` shifts only `int`, so a numpy
    # integer under an index key would be exported slice-local silently.
    return type(v) is int


def _validate_cts_updated(ev) -> None:
    """Plan E E3·0: a pattern-path CTS_UPDATED records its moment (the apply
    candle) as an int `confirmed_at >= idx` (idx = the CTS anchor until E4c); a
    raw-path one (`via == CTS_UPDATED_RAW_VIA`) carries none — its idx IS the
    moment. `via` itself is required (Plan E E3b: every CTS_UPDATED reader now
    asks `event_fields.event_moment`, which needs it)."""
    meta = ev.meta or {}
    _check("via" in meta, f"CTS_UPDATED at idx {ev.idx} lacks meta['via'] (use tests/_event_factory.py)")
    if meta["via"] == CTS_UPDATED_RAW_VIA:
        _check("confirmed_at" not in meta,
               f"raw-path CTS_UPDATED at idx {ev.idx} must not carry confirmed_at")
        return
    ca = meta.get("confirmed_at")
    _check(_is_int(ca) and ca >= ev.idx, (
        f"pattern-path CTS_UPDATED at idx {ev.idx} (via {meta['via']!r}) needs an int "
        f"meta['confirmed_at'] >= idx, got {ca!r}"
    ))


def validate_event_contract(ev) -> None:
    """Raise EventContractViolation if a CTS_ESTABLISHED / BOS_CONFIRMED /
    CTS_UPDATED breaks the contract."""
    if ev.type == "CTS_UPDATED":
        _validate_cts_updated(ev)
        return
    key = _ANCHOR_KEY.get(ev.type)
    if key is None:
        return
    meta = ev.meta or {}
    for k in ("confirmed_at", key):
        _check(k in meta, f"{ev.type} at idx {ev.idx} lacks meta[{k!r}] (use tests/_event_factory.py)")
        _check(_is_int(meta[k]), f"{ev.type} meta[{k!r}] = {meta[k]!r} is not an int")
    role = EVENT_IDX_IS[ev.type]
    named = key if role == "anchor" else "confirmed_at"
    _check(ev.idx == meta[named], (
        f"{ev.type}: idx {ev.idx} != meta[{named!r}] {meta[named]} (ev.idx is the {role})"
    ))


@pytest.fixture(autouse=True)
def _event_contract_validator(request, monkeypatch):
    if request.node.get_closest_marker("illegal_event_contract"):
        yield
        return
    original = StructureEvent.__init__

    def _init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        validate_event_contract(self)

    monkeypatch.setattr(StructureEvent, "__init__", _init)
    yield
