"""Post-E·4 (2026-09-29d; PLAN_E §9.5): the Naming-Standard meta renames — every emit site, no alias.

`BOS_CONFIRMED` `pb_start` -> `last_pullback_apply_idx` (a moment: the last pullback pattern's apply candle since the
previous cycle); `RANGE_STARTED` `cts_idx` -> `cts_anchor_idx`; POI `bos_idx` / `cts_idx` -> `bos_anchor_idx` /
`cts_anchor_idx` (LANDMINES "Event Contract Rules" rule 3). The mirror guard (`test_event_meta_idx_keys`) finds index
keys by NAME, and only on the paths its fixture runs or its static scan cannot excuse: `pb_start` carried no index
suffix, and `cts_idx` stays a legal key of MS's cycle-0 cache, so an old key restored at ONE emit path (the proximity
range, the cycle >= 1 BOS) would pass it. These pins read every emit literal statically and one real run whose events
cover all three RANGE_STARTED paths and the BOS of cycles 0, 1 and 2.
"""
from __future__ import annotations

import ast
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from engine_v2.structure import market_structure
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.zones import poi_zones


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
    rs = [e for e in res.events if e.type == "RANGE_STARTED"]
    assert {e.meta.get("reason") for e in rs} == {None, "pullback_created_range", "proximity_created_range"}
    assert all(type(e.meta["cts_anchor_idx"]) is int and "cts_idx" not in e.meta for e in rs)
    bos = [(e.meta["cycle_id"], e.meta["last_pullback_apply_idx"]) for e in res.events if e.type == "BOS_CONFIRMED"]
    assert bos == [(0, None), (1, 4), (2, None)]        # cycle 2's retracement fired no pullback pattern
    assert not any("pb_start" in e.meta for e in res.events)
    assert all("pb_start" not in lv.meta for lv in res.levels)
