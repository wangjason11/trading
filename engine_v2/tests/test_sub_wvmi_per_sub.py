"""Plan G G4 — sub WVMI trigger metadata PER LENS (`plans/PLAN_G_wvmi_unique_sub.md`, 2026-09-30), the run.log
counts per unique sub, the lens -> stream mapping and the driver's step-6 wiring.

Rewritten from the Plan C `_assign_sub_wvmi_per_sub` tests (one trigger-gated sweep per unique sub). Plan G computes
a sub's records inside its projection, ungated (`test_plan_g_wvmi.py`); the mirror puts one copy on every lens df the
sub is on. The rule pinned here, as written in PLAN_G §4 G4 (every expected value derived from it):

  * per lens: the lens's WVMI-class parent trigger stream (confluence = the sd-prox class `ZONE_PROXIMITY_TRIGGER` +
    `SUBSEQUENT_COUNTER_TRIGGER`; counter = the CTS-prox class `SUBSEQUENT_CONFLUENCE_TRIGGER` — §8.5, Q8), each entry
    LOH-mapped once (`_map_parent_idx_to_m15_hour_end`), sorted by `(m15_idx, parent_idx)`;
  * per sub on that lens: the FIRST entry inside `[start_idx, m15_end_idx]` (inclusive; the edge for an open sub);
  * every record of the lens df, joined to its sub on `meta["sub_id"]` (never position), gets as its FIRST meta keys
    `triggered_by_event_idx` (the PARENT idx, a Python int) / `triggered_by_event_type` — both None when no entry lands
    in the window — and `parent_path_id` = "H1.main" always (Q9); the records exist whatever the stream says;
  * counts: per UNIQUE sub from its projection's own `wvmi_records` (`acted` = subs with a record).

The LOH mapper is monkeypatched on its defining module (the orchestrator imports it inside the function).
"""
from __future__ import annotations

import contextlib
import io
from copy import deepcopy
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pandas as pd
import pytest

import engine_v2.pipeline.orchestrator as orch
from engine_v2.common.types import WVMIRecord
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.sub_structure_pool import LENS_CONFLUENCE, LENS_COUNTER
from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger
from engine_v2.pipeline.orchestrator import (
    _count_sub_wvmi,
    _stamp_sub_wvmi_trigger_meta,
    _wvmi_trigger_streams_by_lens,
)

LENS_PATHS = {
    LENS_CONFLUENCE: "H1.main >> M15.confluence",
    LENS_COUNTER: "H1.main >> M15.counter",
}
PARENT_PATH = "H1.main"
ZPT, SCT, SCONF = "ZONE_PROXIMITY_TRIGGER", "SUBSEQUENT_COUNTER_TRIGGER", "SUBSEQUENT_CONFLUENCE_TRIGGER"
_TRIPLE = ("triggered_by_event_idx", "triggered_by_event_type", "parent_path_id")
_ATTRIBUTION = ("structure_path_id", "use_case", "parent_sid", "parent_cycle_id", "timeframe", "parent_tf",
                "sub_id", "started_by")


def _loh(p: int) -> int:
    """The stub LOH map: parent hour p -> the LAST of its four M15 candles, 4p + 3."""
    return 4 * int(p) + 3


@pytest.fixture(autouse=True)
def _stub_loh(monkeypatch):
    monkeypatch.setattr("engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
                        lambda p, _parent, _m15: _loh(p))


def _make_sub(sub_id: int, *, start_idx: int, m15_end_idx: int, lenses: Tuple[str, ...], n: int = 1,
              started_by: str = "first_confluence") -> LowerTFResult:
    """A projection shaped like `render_sub_projection`'s: `n` slice-local records (meta `{}` — the projection's),
    the meta keys the mirror / the post-pass / the counts read."""
    recs = [WVMIRecord(bos_structure_id=0, bos_cycle_id=k, zone_side="buy", structure_path_id=LENS_PATHS[lenses[0]],
                       fb_idx=1 + 10 * k, lb_idx=2 + 10 * k, fp_idx=3 + 10 * k, lp_idx=4 + 10 * k)
            for k in range(n)]
    return LowerTFResult(
        trigger=MultiTFTrigger(parent_tf="H1", parent_sid=0, parent_cycle_id=1, parent_sd=1, use_case=started_by,
                               lower_tf="M15", lower_sd=1),
        df=pd.DataFrame(), events=[], kl_zones=[], wave_candles=[], fib_states=[], poi_zones=[],
        wvmi_records=recs, prev_bos_lines=[], status="finalized",
        meta={"sub_id": sub_id, "start_idx": start_idx, "m15_end_idx": m15_end_idx, "lenses": tuple(sorted(lenses)),
              "started_by": started_by, "slice_begin": 100, "timeframe": "M15", "parent_tf": "H1"},
    )


def _lens_dfs() -> Dict[str, pd.DataFrame]:
    return {lens: pd.DataFrame(index=range(4)) for lens in LENS_PATHS}


def _run(subs, streams, lens_dfs=None):
    """Mirror each sub into its lenses (the REAL mirror: one deep copy per lens, the lens's path), then the post-pass
    with `results_by_lens` built as the driver builds it."""
    lens_dfs = lens_dfs or _lens_dfs()
    results_by_lens = {lens: [] for lens in LENS_PATHS}
    for res in subs:
        for lens in res.meta["lenses"]:
            edm.mirror_lower_tf_result_to_entity_df(lens_dfs[lens], res, structure_path_id=LENS_PATHS[lens])
            results_by_lens[lens].append(res)
    _stamp_sub_wvmi_trigger_meta(results_by_lens, streams, parent_df=pd.DataFrame(), m15_df=pd.DataFrame(),
                                 lens_dfs=lens_dfs, parent_path_id=PARENT_PATH)
    return lens_dfs


def _triples(lens_df) -> List[tuple]:
    return [tuple(r.meta[k] for k in _TRIPLE) for r in lens_df.attrs.get("wvmi", [])]


def _streams(conf=(), ctr=()):
    return {LENS_CONFLUENCE: list(conf), LENS_COUNTER: list(ctr)}


# --- the window: [start_idx, m15_end_idx], inclusive both ends ---------------------------------------------------------

class TestWindow:
    START, END = _loh(5), _loh(9)          # 23, 39

    @pytest.mark.parametrize("parent_idx, inside", [(4, False), (5, True), (9, True), (10, False)])
    def test_window_is_inclusive_on_both_ends(self, parent_idx, inside):
        """LOH(5) = 23 == start and LOH(9) = 39 == end are inside; one hour before / after is not. The record exists
        either way (no gate) — only its trigger fields differ."""
        sub = _make_sub(1, start_idx=self.START, m15_end_idx=self.END, lenses=(LENS_CONFLUENCE,))
        d = _run([sub], _streams(conf=[(parent_idx, ZPT)]))
        expected = (parent_idx, ZPT, PARENT_PATH) if inside else (None, None, PARENT_PATH)
        assert _triples(d[LENS_CONFLUENCE]) == [expected]

    def test_open_sub_window_ends_at_the_edge_it_was_given(self):
        edge = _loh(12)
        sub = _make_sub(1, start_idx=self.START, m15_end_idx=edge, lenses=(LENS_COUNTER,))
        d = _run([sub], _streams(ctr=[(13, SCONF), (12, SCONF)]))      # LOH 55 > edge; LOH 51 == edge
        assert _triples(d[LENS_COUNTER]) == [(12, SCONF, PARENT_PATH)]


# --- per lens ----------------------------------------------------------------------------------------------------------

class TestPerLens:
    def test_an_entry_of_the_other_lens_gives_none_on_this_lens(self):
        """A counter-stream entry inside a confluence-only sub's window is NOT this lens's trigger."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE,))
        d = _run([sub], _streams(ctr=[(7, SCONF)]))
        assert _triples(d[LENS_CONFLUENCE]) == [(None, None, PARENT_PATH)]
        assert "wvmi" not in d[LENS_COUNTER].attrs

    def test_a_dual_lens_sub_gets_each_lens_its_own_first_trigger(self):
        """The opposite of the Plan C rule (one earliest trigger across lenses swept the sub, its lens decided the
        path): the confluence rows name the confluence stream's first entry (8, ZPT), the counter rows the counter
        stream's (6, SCONF) — even though 6 is earlier."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE, LENS_COUNTER), n=2)
        d = _run([sub], _streams(conf=[(8, ZPT)], ctr=[(6, SCONF)]))
        assert _triples(d[LENS_CONFLUENCE]) == [(8, ZPT, PARENT_PATH)] * 2
        assert _triples(d[LENS_COUNTER]) == [(6, SCONF, PARENT_PATH)] * 2
        for lens in LENS_PATHS:
            assert {(r.structure_path_id, r.meta["structure_path_id"]) for r in d[lens].attrs["wvmi"]} == {
                (LENS_PATHS[lens], LENS_PATHS[lens])}

    def test_only_one_lens_has_a_trigger(self):
        """Q9: the other lens's rows carry idx / type None and `parent_path_id` "H1.main" (the parent entity)."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE, LENS_COUNTER))
        d = _run([sub], _streams(ctr=[(6, SCONF)]))
        assert _triples(d[LENS_CONFLUENCE]) == [(None, None, PARENT_PATH)]
        assert _triples(d[LENS_COUNTER]) == [(6, SCONF, PARENT_PATH)]

    def test_the_two_lens_copies_never_share_a_meta_dict(self):
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE, LENS_COUNTER))
        d = _run([sub], _streams(conf=[(8, ZPT)], ctr=[(6, SCONF)]))
        (c,), (k,) = d[LENS_CONFLUENCE].attrs["wvmi"], d[LENS_COUNTER].attrs["wvmi"]
        assert c is not k and c.meta is not k.meta
        assert sub.wvmi_records[0].meta == {}                     # the projection's record untouched
        assert sub.wvmi_records[0].structure_path_id == LENS_PATHS[LENS_CONFLUENCE]


# --- the first entry: sorted, ties, unplaceable entries ----------------------------------------------------------------

class TestFirstEntry:
    def test_an_unsorted_stream_is_sorted_before_selection(self):
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE,))
        d = _run([sub], _streams(conf=[(8, SCT), (6, ZPT)]))
        assert _triples(d[LENS_CONFLUENCE]) == [(6, ZPT, PARENT_PATH)]

    def test_a_tie_on_the_m15_idx_is_broken_by_the_parent_idx(self, monkeypatch):
        """Two parent idxs can map to one M15 candle (the real mapper falls back to the last M15 before a gap hour):
        the lower parent idx wins — both entries on ONE lens, the later-listed one lower."""
        monkeypatch.setattr("engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
                            lambda p, _a, _b: {6: 27, 7: 27}.get(int(p), _loh(p)))
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE,))
        d = _run([sub], _streams(conf=[(7, ZPT), (6, SCT)]))
        assert _triples(d[LENS_CONFLUENCE]) == [(6, SCT, PARENT_PATH)]

    def test_an_entry_the_mapper_cannot_place_is_dropped(self, monkeypatch):
        monkeypatch.setattr("engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
                            lambda p, _a, _b: None if int(p) == 6 else _loh(p))
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE,))
        d = _run([sub], _streams(conf=[(6, ZPT), (8, SCT)]))
        assert _triples(d[LENS_CONFLUENCE]) == [(8, SCT, PARENT_PATH)]


# --- the meta shape and the join -----------------------------------------------------------------------------------------

class TestMetaShape:
    def test_the_triple_comes_first_then_the_attribution_and_nothing_else(self):
        """Today's key order exactly (the Plan C sweep wrote the triple, the persister the attribution after it); the
        idx is the PARENT idx 6 as a Python int, not the LOH 27."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_COUNTER,))
        d = _run([sub], _streams(ctr=[(6, SCONF)]))
        (rec,) = d[LENS_COUNTER].attrs["wvmi"]
        assert list(rec.meta) == [*_TRIPLE, *_ATTRIBUTION]
        assert type(rec.meta["triggered_by_event_idx"]) is int and rec.meta["triggered_by_event_idx"] == 6

    def test_a_trigger_key_already_in_the_meta_is_replaced(self):
        """The trigger wins over any seeded key (a `{**trigger, **meta}` merge would let a seeded None survive)."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_COUNTER,))
        d = _lens_dfs()
        edm.mirror_lower_tf_result_to_entity_df(d[LENS_COUNTER], sub, structure_path_id=LENS_PATHS[LENS_COUNTER])
        d[LENS_COUNTER].attrs["wvmi"][0].meta["triggered_by_event_idx"] = None
        _stamp_sub_wvmi_trigger_meta({LENS_CONFLUENCE: [], LENS_COUNTER: [sub]}, _streams(ctr=[(6, SCONF)]),
                                     parent_df=pd.DataFrame(), m15_df=pd.DataFrame(), lens_dfs=d,
                                     parent_path_id=PARENT_PATH)
        (rec,) = d[LENS_COUNTER].attrs["wvmi"]
        assert rec.meta["triggered_by_event_idx"] == 6 and list(rec.meta)[:3] == list(_TRIPLE)

    def test_records_are_joined_to_their_sub_by_sub_id_not_position(self):
        """Two subs on one lens, their records REVERSED on the lens df: each record still gets its own sub's
        trigger."""
        a = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE,))
        b = _make_sub(2, start_idx=_loh(10), m15_end_idx=_loh(14), lenses=(LENS_CONFLUENCE,))
        d = _lens_dfs()
        for res in (a, b):
            edm.mirror_lower_tf_result_to_entity_df(d[LENS_CONFLUENCE], res, structure_path_id="x")
        d[LENS_CONFLUENCE].attrs["wvmi"].reverse()
        _stamp_sub_wvmi_trigger_meta({LENS_CONFLUENCE: [a, b], LENS_COUNTER: []},
                                     _streams(conf=[(6, ZPT), (12, SCT)]), parent_df=pd.DataFrame(),
                                     m15_df=pd.DataFrame(), lens_dfs=d, parent_path_id=PARENT_PATH)
        assert [(r.meta["sub_id"], r.meta["triggered_by_event_idx"]) for r in d[LENS_CONFLUENCE].attrs["wvmi"]] == [
            (2, 12), (1, 6)]

    def test_no_double_persistence(self):
        """One copy per lens per record: `len(attrs["wvmi"])` == the sum of the lens's subs' records; every
        (sub_id, sid, cycle) once per lens."""
        a = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9), lenses=(LENS_CONFLUENCE, LENS_COUNTER), n=2)
        b = _make_sub(2, start_idx=_loh(10), m15_end_idx=_loh(14), lenses=(LENS_CONFLUENCE,), n=3)
        d = _run([a, b], _streams(conf=[(6, ZPT)]))
        for lens, n in ((LENS_CONFLUENCE, 5), (LENS_COUNTER, 2)):
            keys = [(r.meta["sub_id"], r.bos_structure_id, r.bos_cycle_id) for r in d[lens].attrs["wvmi"]]
            assert len(keys) == n == len(set(keys))


# --- the counts (run.log) -------------------------------------------------------------------------------------------------

class TestCounts:
    def test_every_sub_with_records_counts_whatever_its_triggers(self):
        """A: first_confluence, {confluence}, 2 records; B: first_counter, {confluence, counter}, 3; C: reversal,
        {counter}, 4 (no trigger anywhere — under Plan C it was not swept) -> acted 3, records 9, by_lens
        {confluence 2 + 3, counter 3 + 4}."""
        subs = [_make_sub(1, start_idx=0, m15_end_idx=9, lenses=(LENS_CONFLUENCE,), n=2),
                _make_sub(2, start_idx=10, m15_end_idx=19, lenses=(LENS_CONFLUENCE, LENS_COUNTER), n=3,
                          started_by="first_counter"),
                _make_sub(3, start_idx=20, m15_end_idx=29, lenses=(LENS_COUNTER,), n=4, started_by="reversal")]
        assert _count_sub_wvmi(subs) == {"acted": 3, "records": 9,
                                         "by_started_by": {"first_confluence": 2, "first_counter": 3, "reversal": 4},
                                         "by_lens": {LENS_CONFLUENCE: 5, LENS_COUNTER: 7}}

    def test_a_dual_lens_sub_is_counted_once(self):
        sub = _make_sub(1, start_idx=0, m15_end_idx=9, lenses=(LENS_CONFLUENCE, LENS_COUNTER), n=2)
        c = _count_sub_wvmi([sub])
        assert (c["acted"], c["records"], c["by_lens"]) == (1, 2, {LENS_CONFLUENCE: 2, LENS_COUNTER: 2})

    def test_by_lens_key_order_is_sorted_not_set_order(self, monkeypatch):
        """`by_lens` is printed in run.log: its key order must not follow the set iteration order (per process: the
        string hash seed). The orchestrator's `set` iterates REVERSE-sorted here."""
        class _ReverseIterSet(set):
            def __iter__(self):
                return iter(sorted(set.__iter__(self), reverse=True))

        monkeypatch.setattr(orch, "set", _ReverseIterSet, raising=False)
        sub = _make_sub(1, start_idx=0, m15_end_idx=9, lenses=(LENS_CONFLUENCE, LENS_COUNTER), n=2)
        assert list(_count_sub_wvmi([sub])["by_lens"]) == [LENS_CONFLUENCE, LENS_COUNTER]

    def test_a_sub_without_records_is_not_acted(self):
        sub = _make_sub(1, start_idx=0, m15_end_idx=9, lenses=(LENS_CONFLUENCE,), n=0)
        assert _count_sub_wvmi([sub]) == {"acted": 0, "records": 0, "by_started_by": {}, "by_lens": {}}


# --- the lens -> stream mapping -----------------------------------------------------------------------------------------

def test_each_lens_reads_its_own_wvmi_class_stream():
    """§8.5 / Q8: confluence = the main's first sd-prox per cycle (ZPT) + each var 4 (SUBSEQUENT_COUNTER), counter =
    each var 3 (SUBSEQUENT_CONFLUENCE); a cycle whose first proximity trigger is opp_sd adds no ZPT; per cycle the
    ZPT first, then its var 4s by trigger idx; cycles in the given order. A swap would rewrite every sub row's trigger
    fields and only `/compare` would see it."""
    t = lambda i, d: SimpleNamespace(idx=i, direction=d)                                            # noqa: E731
    v = lambda s, c, i: SimpleNamespace(parent_sid=s, parent_cycle_id=c, trigger_event_idx=i)       # noqa: E731
    zpt = {(0, 1): [t(11, "sd"), t(13, "opp_sd")], (0, 2): [t(15, "opp_sd"), t(16, "sd")], (1, 0): [t(40, "sd")]}
    var3 = [v(1, 0, 44), v(0, 1, 20)]
    var4 = [v(0, 2, 31), v(0, 1, 14), v(0, 2, 30)]
    assert _wvmi_trigger_streams_by_lens([(0, 1), (0, 2), (1, 0)], zpt, var3, var4) == {
        LENS_CONFLUENCE: [(11, ZPT), (14, SCT), (30, SCT), (31, SCT), (40, ZPT)],
        LENS_COUNTER: [(20, SCONF), (44, SCONF)],
    }
    assert _wvmi_trigger_streams_by_lens([(0, 1)], None, [], []) == {LENS_CONFLUENCE: [], LENS_COUNTER: []}


# --- the driver's step 6 (`_run_multi_tf_dual`): which triggers feed which lens, and the summary ---------------------

def test_the_driver_feeds_each_lens_its_stream_and_counts_each_unique_sub_once(monkeypatch):
    """The collaborators are stubbed on their defining modules (the driver imports them inside the function) — the
    pattern of `test_lifecycle_sweep_unit`'s wiring pin. One dual-lens sub (window [40, 100]) with two records; the
    H1 triggers: cycle (0,1) first proximity sd @11 + var 3 @20; cycle (0,2) opp_sd first + var 4 @30. The lens
    streams handed to the post-pass, the stamped lens copies and the printed summary are pinned; step 7 is stopped."""
    from engine_v2.multitf.sub_structure_pool import StructureKey
    from engine_v2.tests.test_render_sub_projection import _record, _set_lifecycle

    class _Stop(Exception):
        pass

    mt = lambda _v, _h1: SimpleNamespace(parent_sid=0, parent_cycle_id=1, lower_sd=+1)   # noqa: E731
    for mod in ("first_confluence_pipeline", "subsequent_confluence_pipeline", "subsequent_counter_pipeline"):
        monkeypatch.setattr(f"engine_v2.multitf.{mod}.to_multi_tf_trigger", mt)
    monkeypatch.setattr("engine_v2.multitf.uc1_trigger.detect_uc1_triggers", lambda *a, **k: [])
    monkeypatch.setattr("engine_v2.multitf.data_bridge.fetch_lower_tf_data",
                        lambda *a, **k: pd.DataFrame({"time": range(200)}))
    monkeypatch.setattr("engine_v2.multitf.data_bridge.prepare_lower_tf_data", lambda df: df)
    monkeypatch.setattr("engine_v2.multitf.parent_tables.build_parent_tables",
                        lambda *a, **k: SimpleNamespace(cycles=lambda: [(0, 1), (0, 2)]))

    def _sweep(_triggers, *, pool, **_kw):
        sub, _ = pool.get_or_create(StructureKey("H1.main", "M15", 1, 35))
        _record(pool, sub, None, LENS_CONFLUENCE, start_idx=40, seq=0, trigger_type="first_confluence",
                parent_sid=0, parent_cycle_id=1)
        _record(pool, sub, None, LENS_COUNTER, start_idx=45, seq=1, trigger_type="subsequent_confluence",
                parent_sid=0, parent_cycle_id=1)
        _set_lifecycle(sub, 40, 100, "parent_end")
        return SimpleNamespace(unresolved=[], spawned=[])

    def _render(sub, _m15, *, lens_paths, lens_dfs, timeframe):
        res = _make_sub(sub.sub_id, start_idx=sub.start_idx, m15_end_idx=sub.end_idx,
                        lenses=tuple(sorted(sub.lenses())), n=2)
        for lens in res.meta["lenses"]:
            edm.mirror_lower_tf_result_to_entity_df(lens_dfs[lens], res, structure_path_id=lens_paths[lens])
        return res

    streams_seen = []

    def _stamp(results_by_lens, streams_by_lens, **kw):
        streams_seen.append((deepcopy(streams_by_lens), kw["parent_path_id"],
                             {k: [r.meta["sub_id"] for r in v] for k, v in results_by_lens.items()}))
        lens_dfs_seen.update(kw["lens_dfs"])
        return _real_stamp(results_by_lens, streams_by_lens, **kw)

    def _stop(*_a, **_k):
        raise _Stop

    _real_stamp = orch._stamp_sub_wvmi_trigger_meta
    lens_dfs_seen: Dict[str, pd.DataFrame] = {}
    monkeypatch.setattr("engine_v2.multitf.lifecycle_sweep.run_lifecycle_sweep", _sweep)
    monkeypatch.setattr("engine_v2.multitf.entity_df_mutation.render_sub_projection", _render)
    monkeypatch.setattr(orch, "_stamp_sub_wvmi_trigger_meta", _stamp)
    monkeypatch.setattr(orch, "build_sid_records_for_subordinate", _stop)

    t = lambda i, d: SimpleNamespace(idx=i, direction=d)                                              # noqa: E731
    v3 = SimpleNamespace(parent_sid=0, parent_cycle_id=1, trigger_event_idx=20, input_idx=15)
    v4 = SimpleNamespace(parent_sid=0, parent_cycle_id=2, trigger_event_idx=30, input_idx=25)
    h1 = pd.DataFrame({"time": pd.date_range("2025-11-17", periods=40, freq="h", tz="UTC")})
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(_Stop):
        orch._run_multi_tf_dual(
            h1, [], [], [], [], {}, None, first_confluence_triggers=[],
            subsequent_confluence_triggers=[v3], subsequent_counter_triggers=[v4],
            main_zone_proximity_triggers={(0, 1): [t(11, "sd")], (0, 2): [t(15, "opp_sd"), t(16, "sd")]},
        )
    assert streams_seen == [({LENS_CONFLUENCE: [(11, ZPT), (30, SCT)], LENS_COUNTER: [(20, SCONF)]},
                             PARENT_PATH, {LENS_CONFLUENCE: [0], LENS_COUNTER: [0]})]
    # window [40, 100]: LOH(11) = 47 and LOH(20) = 83 inside, LOH(30) = 123 outside
    assert _triples(lens_dfs_seen[LENS_CONFLUENCE]) == [(11, ZPT, PARENT_PATH)] * 2
    assert _triples(lens_dfs_seen[LENS_COUNTER]) == [(20, SCONF, PARENT_PATH)] * 2
    assert ("[multi_tf:dual] sub wvmi acted=1 records=2 by_started_by={'first_confluence': 2} "
            "by_lens={'confluence': 2, 'counter': 2}") in out.getvalue().splitlines()
