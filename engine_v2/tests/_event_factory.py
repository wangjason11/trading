"""The one factory for test-built CTS_ESTABLISHED / BOS_CONFIRMED events (Plan E
E2a; PLAN_E §6.1). Keyword-only; parameter names equal the meta keys.

`idx` defaults to what the contract names (ARCHITECTURE "`ev.idx` convention";
`conftest.EVENT_IDX_IS`): the MOMENT `confirmed_at` on both types (Plan E E4a
CTS_ESTABLISHED, E4b BOS_CONFIRMED). The anchors are the keyword arguments
`cts_anchor_idx` / `bos_anchor_idx`.

`tests/conftest.py` validates every CTS_ESTABLISHED / BOS_CONFIRMED built during a
test against the same contract, so a hand-built event that skips this factory
still has to be legal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.structure.market_structure import StructureEvent


def make_cts_established(
    *,
    cts_anchor_idx: int,
    confirmed_at: int,
    price: Optional[float] = None,
    structure_id: Optional[int] = 0,
    cycle_id: Optional[int] = 0,
    struct_direction: Optional[int] = 1,
    pattern_anchor_idx: Optional[int] = None,
    idx: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> StructureEvent:
    """A CTS_ESTABLISHED. `meta` adds extra keys (e.g. `via`); it may not carry
    the keys this factory owns. An attribution argument set to None omits its key."""
    m = _extra(meta, ("cts_anchor_idx", "confirmed_at", "pattern_anchor_idx"))
    if pattern_anchor_idx is not None:
        m["pattern_anchor_idx"] = int(pattern_anchor_idx)
    m["confirmed_at"] = int(confirmed_at)
    _attribution(m, structure_id, cycle_id, struct_direction)
    m["cts_anchor_idx"] = int(cts_anchor_idx)
    return StructureEvent(
        idx=int(confirmed_at if idx is None else idx),
        category="STRUCTURE", type="CTS_ESTABLISHED", price=price, meta=m,
    )


def make_bos_confirmed(
    *,
    bos_anchor_idx: int,
    confirmed_at: int,
    price: Optional[float] = None,
    structure_id: Optional[int] = 0,
    cycle_id: Optional[int] = 0,
    struct_direction: Optional[int] = 1,
    idx: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> StructureEvent:
    """A BOS_CONFIRMED. `meta` adds extra keys (e.g. `source`, `pb_start`)."""
    m = _extra(meta, ("bos_anchor_idx", "confirmed_at"))
    m["confirmed_at"] = int(confirmed_at)
    _attribution(m, structure_id, cycle_id, struct_direction)
    m["bos_anchor_idx"] = int(bos_anchor_idx)
    return StructureEvent(
        idx=int(confirmed_at if idx is None else idx),
        category="STRUCTURE", type="BOS_CONFIRMED", price=price, meta=m,
    )


def make_cts_updated(
    *,
    cts_anchor_idx: int,
    via: str,
    confirmed_at: Optional[int] = None,
    price: Optional[float] = None,
    idx: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> StructureEvent:
    """A CTS_UPDATED. Raw path (`via == CTS_UPDATED_RAW_VIA`): `idx` = the anchor =
    the processing candle, no `confirmed_at` / `cts_anchor_idx` keys. Pattern
    path: `idx` defaults to the moment `confirmed_at` (Plan E E4c) and the
    anchor goes to meta `cts_anchor_idx`. `meta` adds attribution / extra keys."""
    m = _extra(meta, ("cts_anchor_idx", "confirmed_at", "via"))
    m["via"] = via
    if via == CTS_UPDATED_RAW_VIA:
        assert confirmed_at is None, "a raw-path CTS_UPDATED carries no confirmed_at"
        return StructureEvent(idx=int(cts_anchor_idx if idx is None else idx), category="STRUCTURE",
                              type="CTS_UPDATED", price=price, meta=m)
    assert confirmed_at is not None, "a pattern-path CTS_UPDATED needs its moment confirmed_at"
    m["confirmed_at"] = int(confirmed_at)
    m["cts_anchor_idx"] = int(cts_anchor_idx)
    return StructureEvent(idx=int(confirmed_at if idx is None else idx), category="STRUCTURE",
                          type="CTS_UPDATED", price=price, meta=m)


def make_event(
    etype: str,
    idx: int,
    *,
    price: Optional[float] = None,
    category: str = "STRUCTURE",
    **meta: Any,
) -> StructureEvent:
    """Any structure event; for CTS_ESTABLISHED / BOS_CONFIRMED the `idx`
    ARGUMENT is the ANCHOR (`cts_anchor_idx` / `bos_anchor_idx`) and the event
    goes through the typed factory, whose default sets the emitted `ev.idx`
    (the moment on both types since Plan E E4a / E4b). `meta` must then carry
    `confirmed_at`; `structure_id` / `cycle_id` / `struct_direction` /
    `pattern_anchor_idx` are taken from it. A CTS_UPDATED goes through
    `make_cts_updated` with the `idx` ARGUMENT as its anchor (the pattern path's
    emitted `ev.idx` is `confirmed_at` since Plan E E4c)."""
    if etype == "CTS_UPDATED" and "via" in meta:
        m = dict(meta)
        assert category == "STRUCTURE"
        return make_cts_updated(cts_anchor_idx=idx, via=m.pop("via"),
                                confirmed_at=m.pop("confirmed_at", None), price=price, meta=m)
    if etype in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        m = dict(meta)
        kw: Dict[str, Any] = {
            "confirmed_at": m.pop("confirmed_at"),
            "price": price,
            "structure_id": m.pop("structure_id", None),
            "cycle_id": m.pop("cycle_id", None),
            "struct_direction": m.pop("struct_direction", None),
        }
        assert category == "STRUCTURE"
        if etype == "CTS_ESTABLISHED":
            return make_cts_established(
                cts_anchor_idx=idx, pattern_anchor_idx=m.pop("pattern_anchor_idx", None), meta=m, **kw,
            )
        return make_bos_confirmed(bos_anchor_idx=idx, meta=m, **kw)
    return StructureEvent(idx=int(idx), category=category, type=etype, price=price, meta=dict(meta))


def _extra(meta: Optional[Dict[str, Any]], owned: tuple) -> Dict[str, Any]:
    m = dict(meta or {})
    clash = [k for k in owned if k in m]
    assert not clash, f"pass {clash} as keyword arguments, not in meta"
    return m


def _attribution(m: Dict[str, Any], structure_id, cycle_id, struct_direction) -> None:
    """None omits the key (fixtures that test a missing attribution key)."""
    for k, v in (("cycle_id", cycle_id), ("structure_id", structure_id), ("struct_direction", struct_direction)):
        if v is not None:
            m[k] = int(v)
