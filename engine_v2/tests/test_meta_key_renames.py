"""Post-E·4 (2026-09-29d; PLAN_E §9.5): the Naming-Standard meta renames — every emit site, no alias.

`BOS_CONFIRMED` `pb_start` -> `last_pullback_apply_idx` (a moment: the last pullback pattern's apply candle since the
previous cycle); `RANGE_STARTED` `cts_idx` -> `cts_anchor_idx`; POI `bos_idx` / `cts_idx` -> `bos_anchor_idx` /
`cts_anchor_idx` (LANDMINES "Event Contract Rules" rule 3). The mirror guard (`test_event_meta_idx_keys`) finds index
keys by NAME, and only on the paths its fixture runs or its static scan cannot excuse: `pb_start` carried no index
suffix, and `cts_idx` stays a legal key of MS's cycle-0 cache, so an old key restored at ONE emit path (the proximity
range, the cycle >= 1 BOS) would pass it. These pins read every emit literal statically and one real run whose events
cover all three RANGE_STARTED paths and the BOS of cycles 0, 1 and 2.

Post-E·5 (2026-09-30; PLAN_E §9.6): fib `cycle1_bos_idx` -> `cycle1_bos_anchor_idx` (BOS_1's anchor on the cycle-1 cross
fib; `fib_tracker` reads it for cond2's window), KL `pb_reconfirm_idx` -> `reconfirmed_idx` (the CTS_RECONFIRMED
moment), and the KL `expanded_last_idx` / `_price` / `_event` copies of `bounds_steps[-1]` deleted. The name guard sees
only index-suffixed keys, so the non-index `expanded_last_price` / `_event` and every VALUE are pinned here.
"""
from __future__ import annotations

import ast
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import engine_v2
from engine_v2.structure import market_structure
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
from engine_v2.tests.test_event_meta_idx_keys import _render_both_lenses
from engine_v2.tests.test_render_sub_projection import geometry, m15_df  # noqa: F401 (fixtures)
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.zones import poi_zones
from engine_v2.zones.kl_zones_v1 import derive_kl_zones_v1


def _tree(module):
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _str_consts(module) -> set[str]:
    return {n.value for n in ast.walk(_tree(module)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _kw(call: ast.Call, name: str):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _dict_keys(node) -> set[str]:
    """The string keys of a dict literal — or of the dict literal passed to a wrapper call
    (`meta=self._end_watch_superseded_by_new_cycle(apply, {...})`)."""
    if isinstance(node, ast.Call):
        node = next((a for a in node.args if isinstance(a, ast.Dict)), None)
    assert isinstance(node, ast.Dict), ast.dump(node)[:120] if node is not None else None
    return {k.value for k in node.keys if isinstance(k, ast.Constant)}


def _calls(module, pred):
    return [n for n in ast.walk(_tree(module)) if isinstance(n, ast.Call) and pred(n)]


def test_every_range_started_emit_literal_carries_cts_anchor_idx():
    calls = _calls(market_structure, lambda n: isinstance(_kw(n, "type"), ast.Constant)
                   and _kw(n, "type").value == "RANGE_STARTED")
    assert len(calls) == 3          # offline finalize, pullback_created_range, proximity_created_range
    for c in calls:
        keys = _dict_keys(_kw(c, "meta"))
        assert {"cts_anchor_idx", "cts_price"} <= keys and "cts_idx" not in keys, sorted(keys)


def test_every_bos_confirmed_emit_carries_last_pullback_apply_idx():
    calls = _calls(market_structure, lambda n: isinstance(n.func, ast.Attribute)
                   and n.func.attr == "_emit_bos_confirmed")
    assert len(calls) == 2          # cycle 0 (initial_prior_extreme), cycle >= 1 (pullback_extreme)
    for c in calls:
        assert "last_pullback_apply_idx" in _dict_keys(_kw(c, "meta"))
    assert "pb_start" not in _str_consts(market_structure)


def test_poi_meta_says_anchor_and_no_old_key_is_left():
    metas = [_dict_keys(_kw(c, "meta")) for c in _calls(poi_zones, lambda n: isinstance(_kw(n, "meta"), ast.Dict))]
    poi = [k for k in metas if "cts_established_idx" in k]
    assert len(poi) == 1 and {"bos_anchor_idx", "cts_anchor_idx"} <= poi[0]
    # The fib's own `bos_idx` / `cts_idx` are read as attributes (`fib_state.bos_idx`), never as meta keys — the
    # env-gated debug print included (rule 3: `meta[key]`, no `.get` fallback).
    assert not {"bos_idx", "cts_idx"} & _str_consts(poi_zones)


@pytest.mark.parametrize("sd", [1, -1])
def test_a_real_run_emits_only_the_new_keys(sd):
    rows = _make_double_rewind_data()
    with redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(rows if sd == 1 else _noreg._mirror(rows)), 0, sd)
    # VALUES too (the landing review's surviving mutant set the proximity path's key to the candle): the CTS anchor
    # when the range was decided — the offline one (@13, start 9) was finalized in step 9, before cycle 2 established
    # at 12, so it carries the cycle-1 anchor 8 (a pre-existing look-ahead stamp, not this migration's).
    rs = [(int(e.idx), e.meta.get("reason"), e.meta["cts_anchor_idx"]) for e in res.events if e.type == "RANGE_STARTED"]
    assert rs == [(4, "pullback_created_range", 2), (9, "proximity_created_range", 8), (13, None, 8),
                  (14, "pullback_created_range", 12)]
    bos = [(e.meta["cycle_id"], e.meta["last_pullback_apply_idx"]) for e in res.events if e.type == "BOS_CONFIRMED"]
    assert bos == [(0, None), (1, 4), (2, None)]        # cycle 2's retracement fired no pullback pattern
    old = {"pb_start", "cts_idx", "bos_idx"}             # on NO event or level (an extra key on any path)
    assert not [(e.type, int(e.idx)) for e in res.events if old & set(e.meta)]
    assert not [lv for lv in res.levels if old & set(lv.meta)]


def test_poi_anchors_are_the_owning_fibs_anchors(geometry, m15_df):
    """The POI pair's VALUES (the review's surviving mutant swapped them): each POI's `bos_anchor_idx` /
    `cts_anchor_idx` are the `bos_idx` / `cts_idx` of a fib of its (structure, cycle) — slice-local source frame."""
    res, _ = _render_both_lenses(geometry, m15_df)
    fibs = {}
    for f in res.fib_states:
        fibs.setdefault((f.structure_id, f.cycle_id), set()).add((f.bos_idx, f.cts_idx))
    assert res.poi_zones
    for z in res.poi_zones:
        m = z.meta
        assert (m["bos_anchor_idx"], m["cts_anchor_idx"]) in fibs[(m["structure_id"], m["cycle_id"])], m


# --- Post-E·5 (2026-09-30) -------------------------------------------------------------------------------------------

_POST_E5_OLD = ("cycle1_bos_idx", "pb_reconfirm_idx", "expanded_last_")


def test_no_production_string_carries_a_post_e5_old_key():
    """Every production module — an exporter alias included (no exporter test exists; Post-E·4 review MINOR 4): no
    string constant holds an old key. Comments keep the history."""
    root = Path(engine_v2.__file__).parent
    hits = []
    for path in root.rglob("*.py"):
        if {"legacy_2025", "tests", "plans"} & set(path.parts):
            continue
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                hits += [(path.name, n.lineno, old) for old in _POST_E5_OLD if old in n.value]
    assert hits == []


def _kl_run(sd, extra_events=(), *, after_threshold=(), drop_types=()):
    """KL over the fixture's real run. `extra_events` are appended (the post-pass reads them in any order);
    `after_threshold` go right after the run's BOS_THRESHOLD_UPDATED @9 (the main loop tracks the ACTIVE zone
    in list order — appended at the end they would expand cycle 2's zone); `drop_types` are removed."""
    rows = _make_double_rewind_data()
    with redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(rows if sd == 1 else _noreg._mirror(rows)), 0, sd)
        events = []
        for e in res.events:
            if e.type in drop_types:
                continue
            events.append(e)
            if e.type == "BOS_THRESHOLD_UPDATED" and int(e.idx) == 9:
                events.extend(after_threshold)
        zones = derive_kl_zones_v1(res.df, [*events, *extra_events], struct_direction=sd)
    return res, zones


def _bos1(zones):
    (z,) = [z for z in zones if z.source_kind == "BOS" and z.meta["cycle_id"] == 1]
    return z


@pytest.mark.parametrize("sd", [1, -1])
def test_kl_expansion_lives_in_bounds_steps_only(sd):
    """The fixture's one expansion (cycle 1's BOS zone, BOS_THRESHOLD_UPDATED @9): `bounds_steps[-1]` carries its
    moment, price and event, the zone's bounds are that step's, `expanded` marks exactly the multi-step zones, and no
    `expanded_last_*` copy is written."""
    res, zones = _kl_run(sd)
    (thr,) = [e for e in res.events if e.type == "BOS_THRESHOLD_UPDATED"]
    (z,) = [z for z in zones if z.meta.get("expanded")]
    last = z.meta["bounds_steps"][-1]
    assert (z.source_kind, z.meta["cycle_id"], int(thr.idx)) == ("BOS", 1, 9)
    assert (last["start_idx"], last["price"], last["event"]) == (9, float(thr.price), "BOS_THRESHOLD_UPDATED")
    assert (last["top"], last["bottom"]) == (z.top, z.bottom)
    assert all(bool(z.meta.get("expanded")) == (len(z.meta["bounds_steps"]) > 1) for z in zones)
    assert not [k for z in zones for k in z.meta if k.startswith("expanded_last")]
    # Any other copy under a new name (the review's surviving `last_expansion_price`): the expansion adds exactly
    # the `expanded` flag to the zone's keys (the same zone without the threshold event).
    _, plain = _kl_run(sd, drop_types=("BOS_THRESHOLD_UPDATED",))
    assert set(_bos1(zones).meta) == set(_bos1(plain).meta) | {"expanded"}


@pytest.mark.parametrize("sd", [1, -1])
def test_kl_a_second_expansion_is_appended_as_the_last_step(sd):
    """`bounds_steps[-1]` is the LAST expansion only if steps are APPENDED in event order (the review's surviving
    `insert` mutant): a more extreme BOS_THRESHOLD_UPDATED @10 after the run's @9 → steps [INIT 4, 9, 10], the last
    one = the @10 event, and the zone's outer bound = its price."""
    res, _ = _kl_run(sd)
    (thr,) = [e for e in res.events if e.type == "BOS_THRESHOLD_UPDATED"]
    price = float(thr.price) - 0.001 * sd            # a buy zone (sd +1) expands DOWN, a sell zone UP
    second = StructureEvent(idx=10, category="STRUCTURE", type="BOS_THRESHOLD_UPDATED", price=price,
                            meta={"prev": float(thr.price), "reason": "probe_no_break", "cycle_id": 1,
                                  "structure_id": 0, "struct_direction": sd})
    _, zones = _kl_run(sd, after_threshold=[second])
    z = _bos1(zones)
    steps = z.meta["bounds_steps"]
    assert [s["start_idx"] for s in steps] == [4, 9, 10]
    assert (steps[-1]["event"], steps[-1]["price"]) == ("BOS_THRESHOLD_UPDATED", price)
    assert (z.bottom if sd == 1 else z.top) == price


def test_exporters_write_every_meta_dict_verbatim(tmp_path, geometry, m15_df):
    """The export layer (no exporter test existed — the Post-E·4 review's MINOR 4, the Post-E·5 review's X2 / X3: an
    alias built from string pieces, or `bounds_steps` dropped, in an exporter survived): each CSV row's `meta` text is
    exactly `str()` of the object's meta — KL (incl. an expansion and a reconfirm), fib and events from the fixture's
    run, POI from a rendered sub."""
    import csv
    from engine_v2.debug.export_events import export_structure_events
    from engine_v2.debug.export_fib_lifecycle import export_fib_lifecycle
    from engine_v2.debug.export_zones import export_kl_zones, export_poi_zones
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline

    rec = StructureEvent(idx=11, category="STRUCTURE", type="CTS_RECONFIRMED", price=None, meta={
        "via": "synthetic", "cycle_id": 1, "structure_id": 0, "struct_direction": 1, "confirmed_at": 11,
        "cts_anchor_idx": 8, "confirmation_method": "pullback"})
    with redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_make_double_rewind_data()), 0, 1)
        events = [*res.events, rec]
        down = _run_downstream_pipeline(res.df, events, 1, skip_wvmi=True)
    sub, _ = _render_both_lenses(geometry, m15_df)
    assert any("reconfirmed_idx" in z.meta for z in down["kl_zones"])
    assert any(z.meta.get("expanded") for z in down["kl_zones"]) and down["fib_states"] and sub.poi_zones
    for name, export, objs in (("kl", export_kl_zones, down["kl_zones"]),
                               ("fib", export_fib_lifecycle, down["fib_states"]),
                               ("events", export_structure_events, events),
                               ("poi", export_poi_zones, sub.poi_zones)):
        path = tmp_path / f"{name}.csv"
        export(objs, path)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["meta"] for r in rows] == [str(o.meta) for o in objs], name


@pytest.mark.parametrize("sd", [1, -1])
def test_kl_reconfirm_records_the_reconfirmed_moment(sd):
    """Cycle 1's CTS was confirmed by sd-zone proximity @9 (anchor 8); a CTS_RECONFIRMED @11 (built as
    `_emit_cts_reconfirmed` builds it — no fixture fires one) upgrades that zone only: `reconfirmed_idx` = 11 (the
    moment — not the proximity moment 9, not the anchor 8), `confirmed_idx` keeps 9, the method becomes "pullback",
    and nothing else in the zone changes."""
    rec = StructureEvent(idx=11, category="STRUCTURE", type="CTS_RECONFIRMED", price=None, meta={
        "via": "synthetic", "cycle_id": 1, "structure_id": 0, "struct_direction": sd, "confirmed_at": 11,
        "cts_anchor_idx": 8, "confirmation_method": "pullback"})
    _, before = _kl_run(sd)
    _, zones = _kl_run(sd, [rec])
    (cts1,) = [z for z in before if z.source_kind == "CTS" and z.meta["cycle_id"] == 1]
    assert (cts1.meta["confirmed_idx"], cts1.meta["anchor_idx"], cts1.meta["confirmation_method"]) == (
        9, 8, "sd_zone_proximity")
    up = [z for z in zones if "reconfirmed_idx" in z.meta]
    assert [(z.source_kind, z.meta["cycle_id"]) for z in up] == [("CTS", 1)]
    m = up[0].meta
    assert (m["reconfirmed_idx"], m["confirmed_idx"], m["anchor_idx"], m["confirmation_method"]) == (
        11, 9, 8, "pullback")
    assert {k: v for k, v in m.items() if k not in ("reconfirmed_idx", "confirmation_method")} == {
        k: v for k, v in cts1.meta.items() if k != "confirmation_method"}
    assert not [z for z in zones if "pb_reconfirm_idx" in z.meta]
