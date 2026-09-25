"""`structure/event_fields.py` — the named reads of a CTS / BOS event's indices
(Plan E E2a; PLAN_E §6.1)."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import engine_v2
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established, make_event


def _ev(etype, idx, **meta):
    return StructureEvent(idx=idx, category="STRUCTURE", type=etype, price=1.0, meta=meta)


# --- accessors: direct index, the named key, strict types -----------------------

def test_cts_anchor_idx_reads_the_meta_key_not_idx():
    # Since Plan E E4a a CTS_ESTABLISHED's idx is its moment (10): the accessor
    # must read the key.
    est = make_cts_established(cts_anchor_idx=9, confirmed_at=10)
    assert est.idx == 10
    assert ef.cts_anchor_idx(est) == 9
    conf = _ev("CTS_CONFIRMED", 25, cts_anchor_idx=20, confirmed_at=25)
    assert ef.cts_anchor_idx(conf) == 20
    assert ef.cts_anchor_idx(_ev("CTS_RECONFIRMED", 30, cts_anchor_idx=20)) == 20
    # A pattern-path CTS_UPDATED stamps its moment (14) since Plan E E4c; its
    # anchor (12) is the meta key. A raw-path one's idx is both.
    assert ef.cts_anchor_idx(_ev("CTS_UPDATED", 14, via="continuous", confirmed_at=14,
                                 cts_anchor_idx=12)) == 12
    assert ef.cts_anchor_idx(_ev("CTS_UPDATED", 13, via="replay_raw")) == 13


def test_bos_anchor_idx_reads_the_meta_key_not_idx():
    # Since Plan E E4b a BOS_CONFIRMED's idx is its moment (10): the accessor
    # must read the key.
    bos = make_bos_confirmed(bos_anchor_idx=3, confirmed_at=10)
    assert bos.idx == 10
    assert ef.bos_anchor_idx(bos) == 3


def test_pattern_anchor_idx():
    est = make_cts_established(cts_anchor_idx=9, confirmed_at=10, pattern_anchor_idx=8)
    assert ef.pattern_anchor_idx(est) == 8
    assert ef.pattern_anchor_idx(_ev("REVERSAL_CANDIDATE", 5, pattern_anchor_idx=5)) == 5


@pytest.mark.illegal_event_contract
def test_accessors_raise_on_a_missing_key():
    """No `.get(key, ev.idx)` fallback (LANDMINES "Event Contract Rules")."""
    with pytest.raises(KeyError):
        ef.cts_anchor_idx(_ev("CTS_ESTABLISHED", 9, confirmed_at=10))
    with pytest.raises(KeyError):
        ef.bos_anchor_idx(_ev("BOS_CONFIRMED", 3, confirmed_at=10))
    with pytest.raises(KeyError):
        ef.pattern_anchor_idx(_ev("CTS_ESTABLISHED", 9, confirmed_at=10, cts_anchor_idx=9))


def test_accessors_raise_on_the_wrong_type():
    with pytest.raises(ValueError):
        ef.cts_anchor_idx(make_bos_confirmed(bos_anchor_idx=3, confirmed_at=10))
    with pytest.raises(ValueError):
        ef.bos_anchor_idx(make_cts_established(cts_anchor_idx=9, confirmed_at=10))
    with pytest.raises(ValueError):
        ef.pattern_anchor_idx(_ev("CTS_UPDATED", 12, via="continuous", confirmed_at=12, cts_anchor_idx=12))


def test_event_moment_extended_to_bos_and_confirmations():
    assert ef.event_moment(make_bos_confirmed(bos_anchor_idx=3, confirmed_at=10)) == 10
    assert ef.event_moment(make_cts_established(cts_anchor_idx=9, confirmed_at=10)) == 10
    assert ef.event_moment(_ev("CTS_RECONFIRMED", 30, cts_anchor_idx=20)) == 30


def test_processing_order_key_is_the_pre_e4_idx_type_order():
    """`processing_order_key` == the pre-E4 `(ev.idx, ev.type)` for every type:
    a CTS_ESTABLISHED / BOS_CONFIRMED / pattern-path CTS_UPDATED keys on its
    anchor (9 / 3 / 12), not its idx (the moment 10 / 10 / 13 since Plan E E4a /
    E4b / E4c); every other type on its idx."""
    evs = [
        make_cts_established(cts_anchor_idx=9, confirmed_at=10),
        make_bos_confirmed(bos_anchor_idx=3, confirmed_at=10),
        _ev("CTS_UPDATED", 13, via="continuous", confirmed_at=13, cts_anchor_idx=12),
        _ev("CTS_CONFIRMED", 14, cts_anchor_idx=12, confirmed_at=14),
        StructureEvent(idx=11, category="RANGE", type="RANGE_STARTED", meta={}),
    ]
    assert (evs[0].idx, evs[1].idx, evs[2].idx) == (10, 10, 13)
    assert [ef.processing_order_key(e) for e in evs] == [
        (9, "CTS_ESTABLISHED"), (3, "BOS_CONFIRMED"), (12, "CTS_UPDATED"),
        (14, "CTS_CONFIRMED"), (11, "RANGE_STARTED"),
    ]
    assert [ef.stamped_idx(e) for e in evs] == [9, 3, 12, 14, 11]


def test_stamped_idx_reads_the_anchor_keys_not_idx():
    """Frozen against E4: an EST / BOS whose idx is the moment still stamps at
    its anchor; CONFIRMED stays at its idx (the confirmation candle)."""
    assert ef.stamped_idx(make_cts_established(cts_anchor_idx=9, confirmed_at=10)) == 9
    assert ef.stamped_idx(make_bos_confirmed(bos_anchor_idx=3, confirmed_at=10)) == 3
    assert ef.stamped_idx(_ev("CTS_CONFIRMED", 14, cts_anchor_idx=12, confirmed_at=14)) == 14


# --- the factory ------------------------------------------------------------------

def test_factory_idx_defaults_to_the_contract_index_and_carries_both_keys():
    """The moment on both types (Plan E E4a CTS_ESTABLISHED, E4b BOS_CONFIRMED);
    `make_event`'s `idx` argument is the anchor either way."""
    est = make_cts_established(cts_anchor_idx=9, confirmed_at=10, structure_id=2, cycle_id=1)
    assert (est.idx, est.meta["cts_anchor_idx"], est.meta["confirmed_at"]) == (10, 9, 10)
    est = make_event("CTS_ESTABLISHED", 9, confirmed_at=10, structure_id=2, cycle_id=1)
    assert (est.idx, est.meta["cts_anchor_idx"], est.meta["confirmed_at"]) == (10, 9, 10)
    bos = make_event("BOS_CONFIRMED", 3, confirmed_at=10, structure_id=2, cycle_id=1)
    assert (bos.idx, bos.meta["bos_anchor_idx"], bos.meta["confirmed_at"]) == (10, 3, 10)
    assert "struct_direction" not in bos.meta  # make_event adds no key the caller omitted


# --- the qualified-call rule (Q16) ----------------------------------------------

_ACCESSORS = {"cts_anchor_idx", "bos_anchor_idx", "pattern_anchor_idx", "event_moment",
              "processing_order_key", "stamped_idx", "*"}


def test_no_module_imports_an_accessor_by_name():
    """`from ...event_fields import cts_anchor_idx` would collide with the many
    locals of the same name (`cts_anchor_idx = cts_anchor_idx(ev)` raises
    UnboundLocalError). Import the module and call `ef.<accessor>(ev)`."""
    root = Path(engine_v2.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        if "legacy_2025" in path.parts:  # archived, not importable
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if (isinstance(n, ast.ImportFrom) and n.module == "engine_v2.structure.event_fields"
                    and any(a.name in _ACCESSORS for a in n.names)):
                offenders.append(f"{path.relative_to(root)}:{n.lineno}")
    assert offenders == []


# --- MS state stays on the anchor (the st.bos decoupling, PLAN_E §6.1) ----------

def test_emit_bos_confirmed_builds_st_bos_from_the_anchor_param():
    from engine_v2.structure.market_structure import MarketStructure
    from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df
    ms = MarketStructure(_prepare_df(_make_multicycle_data()), 1)
    # idx = X (the moment, emitted since Plan E E4b), bos_anchor_idx = Y (the location).
    ms._emit_bos_confirmed(12, 0.95, bos_anchor_idx=7, meta={"confirmed_at": 12})
    assert ms.state.bos.idx == 7
    assert ms.state.bos.price == 0.95 and ms.state.bos_threshold == 0.95
    ev = ms.events[-1]
    assert (ev.idx, ev.meta["bos_anchor_idx"]) == (12, 7)


# --- no silent cross-kind fallback on a contract key (Plan E E2d) ----------------

_FALLBACK = re.compile(
    r"""\.get\(\s*["'](confirmed_at|cts_anchor_idx|bos_anchor_idx|pattern_anchor_idx|cycle1_bos_idx)["']\s*,"""
    r"""|\bc0\.get\(\s*["'](bos_idx|cts_idx)["']\s*,"""
)


def test_no_production_module_falls_back_on_a_contract_index_key():
    """LANDMINES "Event Contract Rules" rule 3: every emitter sets these keys, so a
    reader indexes them directly — a `.get(key, <default>)` would silently read
    another kind of index (the anchor for a moment, or vice versa)."""
    root = Path(engine_v2.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        if {"legacy_2025", "tests", "plans"} & set(path.parts):
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _FALLBACK.search(line):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert offenders == []
