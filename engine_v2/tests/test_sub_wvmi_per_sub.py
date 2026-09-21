"""Unit tests pinning `pipeline/orchestrator._assign_sub_wvmi_per_sub`
(Plan C §6.4 / PART4 §17.10 minimal / WVMI_SPEC "Sub" cadence block).

The rule, as written in the plan/spec (every expected value below is derived
from it, never from what the code prints):

  * one sweep per UNIQUE sub (dedup key `sub_id`);
  * window = the sub's real-time lifecycle `[start_idx, m15_end_idx]`
    (inclusive; `m15_end_idx` is the data edge for an open sub);
  * stream = union of the confluence and counter parent-trigger streams,
    each entry LOH-mapped once (`_map_parent_idx_to_m15_hour_end`) and
    tagged with its lens, sorted by `(m15_idx, parent_idx)`, RESTRICTED to
    the sub's lenses;
  * the FIRST entry inside the window sweeps the sub once via
    `compute_parent_driven_sub_wvmi(result, sub_path_id=lens_paths[lens],
    parent_trigger=ParentTrigger(idx=parent_idx, event_type, "H1.main"))`
    — the sweeping trigger's lens decides the records' `structure_path_id`;
  * no entry inside the window -> no WVMI for that sub;
  * the records are persisted into EVERY lens df the sub is on
    (`persist_facade_wvmi_to_entity_df(lens_dfs[l], result,
    structure_path_id=lens_paths[l])`), each stamped with the §17.9
    attribution (`sub_id` + informational `started_by` / `use_case` /
    `parent_sid` / `parent_cycle_id`); the four wave-candle idxs are shifted
    slice-local -> entity-absolute by `meta["slice_begin"]`;
    `triggered_by_event_idx` stays in parent-df coords (LANDMINE "WVMI
    Records Carry Mixed-Coordinate Meta");
  * counts `{"acted", "records", "by_started_by", "by_lens"}` feed the
    greppable `[multi_tf:dual] sub wvmi acted=... records=...` log line.

The orchestrator imports the LOH mapper and the sweep INSIDE the function,
so both are monkeypatched on their defining modules
(`engine_v2.multitf.entity_df_mutation` / `engine_v2.multitf.sub_wvmi`).
`persist_facade_wvmi_to_entity_df` is the real one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from engine_v2.common.types import WVMIRecord
from engine_v2.multitf.sub_structure_pool import LENS_CONFLUENCE, LENS_COUNTER
from engine_v2.multitf.sub_wvmi import ParentTrigger
from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger
from engine_v2.pipeline.orchestrator import _assign_sub_wvmi_per_sub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LENS_PATHS = {
    LENS_CONFLUENCE: "H1.main >> M15.confluence",
    LENS_COUNTER: "H1.main >> M15.counter",
}
PARENT_PATH = "H1.main"

# Simple LOH stand-in: parent hour p -> the LAST of its four M15 candles
# (M15 idx 0 aligned to parent idx 0), i.e. 4p + 3.
def _loh_4p3(p: int, parent_df: pd.DataFrame, m15_df: pd.DataFrame) -> int:
    return 4 * int(p) + 3


def _loh(p: int) -> int:
    """The expected M15 idx of parent idx `p` under `_loh_4p3` (test side)."""
    return 4 * p + 3


def _make_trigger(
    use_case: str = "first_confluence",
    parent_sid: int = 0,
    parent_cycle_id: int = 1,
) -> MultiTFTrigger:
    return MultiTFTrigger(
        parent_tf="H1",
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle_id,
        parent_sd=1,
        use_case=use_case,
        lower_tf="M15",
        lower_sd=1,
        start_time=pd.Timestamp("2024-01-01", tz="UTC"),
        start_price=1.0,
        lifecycle_end_idx=None,
    )


def _make_sub(
    sub_id: int,
    *,
    start_idx: int,
    m15_end_idx: int,
    lenses: Tuple[str, ...],
    started_by: str = "first_confluence",
    slice_begin: int = 0,
    use_case: str = "first_confluence",
    parent_sid: int = 0,
    parent_cycle_id: int = 1,
) -> LowerTFResult:
    """A sub projection shaped like `render_sub_projection`'s LowerTFResult
    (only the meta keys `_assign_sub_wvmi_per_sub` / the persister read)."""
    return LowerTFResult(
        trigger=_make_trigger(use_case, parent_sid, parent_cycle_id),
        df=pd.DataFrame(),
        events=[],
        kl_zones=[],
        wave_candles=[],
        fib_states=[],
        poi_zones=[],
        wvmi_records=[],
        prev_bos_lines=[],
        status="finalized",
        meta={
            "sub_id": sub_id,
            "start_idx": start_idx,
            "m15_end_idx": m15_end_idx,
            "lenses": tuple(sorted(lenses)),
            "started_by": started_by,
            "slice_begin": slice_begin,
        },
    )


@dataclass
class _Call:
    result: LowerTFResult
    sub_path_id: str
    parent_trigger: ParentTrigger


@dataclass
class _RecordingSweep:
    """Stand-in for `compute_parent_driven_sub_wvmi`: records every call and
    returns `n_by_sub_id[sub_id]` (default 1) slice-LOCAL fake records that
    mimic the real sweep's outputs — `structure_path_id` = the `sub_path_id`
    it was handed, meta stamped with the ParentTrigger's three fields."""
    n_by_sub_id: Dict[int, int] = field(default_factory=dict)
    calls: List[_Call] = field(default_factory=list)

    def __call__(self, result, sub_path_id, parent_trigger) -> List[WVMIRecord]:
        self.calls.append(_Call(result, sub_path_id, parent_trigger))
        n = self.n_by_sub_id.get(result.meta["sub_id"], 1)
        recs = []
        for k in range(n):
            recs.append(WVMIRecord(
                bos_structure_id=0,
                bos_cycle_id=k + 1,
                zone_side="buy",
                structure_path_id=sub_path_id,
                fb_idx=1 + 10 * k,
                lb_idx=2 + 10 * k,
                fp_idx=3 + 10 * k,
                lp_idx=None if k == 0 else 4 + 10 * k,   # first record: LP not yet found
                meta={
                    "triggered_by_event_idx": parent_trigger.idx,
                    "triggered_by_event_type": parent_trigger.event_type,
                    "parent_path_id": parent_trigger.parent_path_id,
                },
            ))
        return recs


@pytest.fixture
def lens_dfs() -> Dict[str, pd.DataFrame]:
    return {
        LENS_CONFLUENCE: pd.DataFrame({"time": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")}),
        LENS_COUNTER: pd.DataFrame({"time": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")}),
    }


@pytest.fixture
def frames() -> Dict[str, pd.DataFrame]:
    # Ignored by the stub mapper; present because the signature requires them.
    return {
        "parent_df": pd.DataFrame({"time": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")}),
        "m15_df": pd.DataFrame({"time": pd.date_range("2024-01-01", periods=16, freq="15min", tz="UTC")}),
    }


@pytest.fixture
def sweep(monkeypatch) -> _RecordingSweep:
    stub = _RecordingSweep()
    monkeypatch.setattr(
        "engine_v2.multitf.sub_wvmi.compute_parent_driven_sub_wvmi", stub,
    )
    monkeypatch.setattr(
        "engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
        _loh_4p3,
    )
    return stub


def _run(sub_results, streams, *, lens_dfs, frames):
    return _assign_sub_wvmi_per_sub(
        sub_results,
        streams,
        parent_df=frames["parent_df"],
        m15_df=frames["m15_df"],
        lens_dfs=lens_dfs,
        lens_paths=LENS_PATHS,
    )


def _wvmi(lens_df: pd.DataFrame) -> list:
    return list(lens_df.attrs.get("wvmi", []))


# ---------------------------------------------------------------------------
# Window = [start_idx, m15_end_idx], inclusive both ends
# ---------------------------------------------------------------------------

class TestWindow:
    # Sub window in M15 coords: start = LOH(5) = 23, end = LOH(9) = 39.
    START, END = _loh(5), _loh(9)

    @pytest.mark.parametrize(
        "parent_idx, expect_swept",
        [
            (4, False),   # LOH(4) = 19 < 23 = start  -> outside (before)
            (5, True),    # LOH(5) = 23 == start      -> inside (closed lower bound)
            (9, True),    # LOH(9) = 39 == end        -> inside (closed upper bound)
            (10, False),  # LOH(10) = 43 > 39 = end   -> outside (after)
        ],
    )
    def test_window_is_inclusive_on_both_ends(
        self, sweep, lens_dfs, frames, parent_idx, expect_swept,
    ):
        """§17.10: window = `[start_idx, m15_end_idx]` — a trigger whose
        LOH-mapped M15 idx equals either bound sweeps; one candle past
        either bound does not."""
        sub = _make_sub(1, start_idx=self.START, m15_end_idx=self.END,
                        lenses=(LENS_CONFLUENCE,))
        streams = {
            LENS_CONFLUENCE: [(parent_idx, "ZONE_PROXIMITY_TRIGGER")],
            LENS_COUNTER: [],
        }
        counts = _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        assert len(sweep.calls) == (1 if expect_swept else 0)
        assert counts["acted"] == (1 if expect_swept else 0)
        assert len(_wvmi(lens_dfs[LENS_CONFLUENCE])) == (1 if expect_swept else 0)
        if expect_swept:
            assert sweep.calls[0].parent_trigger.idx == parent_idx

    def test_open_sub_window_ends_at_the_edge_it_was_given(
        self, sweep, lens_dfs, frames,
    ):
        """An open sub's `m15_end_idx` is the data edge (render_sub_projection
        substitutes it) — the function treats it like any other closed upper
        bound: a trigger at the edge sweeps, past it does not."""
        edge = _loh(12)
        sub = _make_sub(1, start_idx=self.START, m15_end_idx=edge,
                        lenses=(LENS_COUNTER,))
        streams = {
            LENS_CONFLUENCE: [],
            LENS_COUNTER: [(13, "SUBSEQUENT_CONFLUENCE_TRIGGER"),   # LOH(13) = 55 > edge
                           (12, "SUBSEQUENT_CONFLUENCE_TRIGGER")],  # LOH(12) = 51 == edge
        }
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)
        assert [c.parent_trigger.idx for c in sweep.calls] == [12]


# ---------------------------------------------------------------------------
# Stream restricted to the sub's lenses
# ---------------------------------------------------------------------------

class TestLensRestriction:
    def test_trigger_on_a_lens_the_sub_is_not_on_is_ignored(
        self, sweep, lens_dfs, frames,
    ):
        """§17.10: "restricted to the sub's lenses" — a counter-stream entry
        inside the window does NOT sweep a confluence-only sub, even though
        it is the only entry in the window."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_CONFLUENCE,))
        streams = {
            LENS_CONFLUENCE: [],
            LENS_COUNTER: [(7, "SUBSEQUENT_CONFLUENCE_TRIGGER")],  # LOH(7)=31 in [23,39]
        }
        counts = _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        assert sweep.calls == []
        assert counts == {"acted": 0, "records": 0, "by_started_by": {}, "by_lens": {}}
        assert _wvmi(lens_dfs[LENS_CONFLUENCE]) == []
        assert _wvmi(lens_dfs[LENS_COUNTER]) == []

    def test_foreign_lens_entry_is_skipped_over_not_just_ignored_when_alone(
        self, sweep, lens_dfs, frames,
    ):
        """The restriction is a FILTER on the sorted union, not a veto: an
        earlier foreign-lens entry is skipped and the first OWN-lens entry
        inside the window sweeps."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_COUNTER,))
        streams = {
            LENS_CONFLUENCE: [(6, "ZONE_PROXIMITY_TRIGGER")],        # earlier, foreign
            LENS_COUNTER: [(8, "SUBSEQUENT_CONFLUENCE_TRIGGER")],    # later, own lens
        }
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        assert len(sweep.calls) == 1
        call = sweep.calls[0]
        assert call.parent_trigger.idx == 8
        assert call.parent_trigger.event_type == "SUBSEQUENT_CONFLUENCE_TRIGGER"
        assert call.sub_path_id == LENS_PATHS[LENS_COUNTER]


# ---------------------------------------------------------------------------
# First trigger by (LOH m15 idx, parent idx) across lenses
# ---------------------------------------------------------------------------

class TestFirstTriggerSelection:
    def test_earliest_by_m15_idx_wins_across_lenses(
        self, sweep, lens_dfs, frames,
    ):
        """§17.10: "the first trigger (by idx) inside the window sweeps" —
        across BOTH lenses of a two-lens sub. The counter entry (parent 6,
        LOH 27) precedes the confluence entry (parent 8, LOH 35) even though
        the confluence stream is listed first, so the counter lens sweeps and
        decides `sub_path_id`."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_CONFLUENCE, LENS_COUNTER))
        streams = {
            LENS_CONFLUENCE: [(8, "ZONE_PROXIMITY_TRIGGER")],
            LENS_COUNTER: [(6, "SUBSEQUENT_CONFLUENCE_TRIGGER")],
        }
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        assert len(sweep.calls) == 1
        call = sweep.calls[0]
        assert call.sub_path_id == LENS_PATHS[LENS_COUNTER]
        assert call.parent_trigger == ParentTrigger(
            idx=6, event_type="SUBSEQUENT_CONFLUENCE_TRIGGER", parent_path_id=PARENT_PATH,
        )

    def test_unsorted_stream_within_one_lens_is_sorted_before_selection(
        self, sweep, lens_dfs, frames,
    ):
        """The union is sorted by (m15_idx, parent_idx) — a later entry listed
        first in its own stream must not win."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_CONFLUENCE,))
        streams = {
            LENS_CONFLUENCE: [(8, "SUBSEQUENT_COUNTER_TRIGGER"),
                              (6, "ZONE_PROXIMITY_TRIGGER")],
            LENS_COUNTER: [],
        }
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)
        assert [c.parent_trigger.idx for c in sweep.calls] == [6]
        assert sweep.calls[0].parent_trigger.event_type == "ZONE_PROXIMITY_TRIGGER"

    def test_tie_on_m15_idx_is_broken_by_parent_idx(
        self, monkeypatch, sweep, lens_dfs, frames,
    ):
        """Two parent idxs can LOH-map to the same M15 candle (the real
        mapper falls back to the last M15 before a gap hour). The sort key is
        `(m15_idx, parent_idx)`, so the LOWER parent idx wins the tie — here
        parent 6 (counter) over parent 7 (confluence) even though the
        confluence stream is enumerated first."""
        def loh_with_gap(p, parent_df, m15_df):
            return {6: 27, 7: 27}.get(int(p), 4 * int(p) + 3)
        monkeypatch.setattr(
            "engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
            loh_with_gap,
        )
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_CONFLUENCE, LENS_COUNTER))
        streams = {
            LENS_CONFLUENCE: [(7, "ZONE_PROXIMITY_TRIGGER")],
            LENS_COUNTER: [(6, "SUBSEQUENT_CONFLUENCE_TRIGGER")],
        }
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        assert len(sweep.calls) == 1
        assert sweep.calls[0].parent_trigger.idx == 6
        assert sweep.calls[0].sub_path_id == LENS_PATHS[LENS_COUNTER]

    def test_entry_the_mapper_cannot_place_is_dropped(
        self, monkeypatch, sweep, lens_dfs, frames,
    ):
        """An entry whose LOH map is None (parent idx not in the parent frame)
        never sweeps; the next placeable entry does."""
        def loh_partial(p, parent_df, m15_df):
            return None if int(p) == 6 else 4 * int(p) + 3
        monkeypatch.setattr(
            "engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
            loh_partial,
        )
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_CONFLUENCE,))
        streams = {
            LENS_CONFLUENCE: [(6, "ZONE_PROXIMITY_TRIGGER"),
                              (8, "SUBSEQUENT_COUNTER_TRIGGER")],
            LENS_COUNTER: [],
        }
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)
        assert [c.parent_trigger.idx for c in sweep.calls] == [8]


# ---------------------------------------------------------------------------
# One sweep: call shape, persistence into every lens, attribution, shift
# ---------------------------------------------------------------------------

class TestSweepAndPersist:
    def test_sweep_is_called_once_with_sweeping_lens_path_and_parent_trigger(
        self, sweep, lens_dfs, frames,
    ):
        """`compute_parent_driven_sub_wvmi(result, sub_path_id=lens_paths[lens],
        parent_trigger=ParentTrigger(idx=parent_idx, event_type, "H1.main"))`
        — exactly once, with the projection object itself."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_CONFLUENCE,))
        streams = {
            LENS_CONFLUENCE: [(6, "ZONE_PROXIMITY_TRIGGER"),
                              (8, "SUBSEQUENT_COUNTER_TRIGGER")],   # second in window: NOT a 2nd sweep
            LENS_COUNTER: [],
        }
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        assert len(sweep.calls) == 1
        call = sweep.calls[0]
        assert call.result is sub
        assert call.sub_path_id == LENS_PATHS[LENS_CONFLUENCE]
        assert call.parent_trigger == ParentTrigger(
            idx=6, event_type="ZONE_PROXIMITY_TRIGGER", parent_path_id=PARENT_PATH,
        )

    def test_records_persisted_into_every_lens_df_shifted_and_attributed(
        self, sweep, lens_dfs, frames,
    ):
        """§17.10: persisted into EVERY lens df the sub is on. Per lens df:
          - one deep copy per record (distinct objects from the projection's
            and from the other lens's);
          - `fb/lb/fp/lp_idx` shifted by `meta["slice_begin"]` (None stays
            None);
          - meta `structure_path_id` = THAT lens's path (§17.9 attribution),
            while the record's `structure_path_id` attribute = the SWEEPING
            lens's path (the sweep's `sub_path_id`);
          - meta `sub_id` / `started_by` / `use_case` / `parent_sid` /
            `parent_cycle_id` / `timeframe` / `parent_tf` stamped;
          - `triggered_by_event_idx` untouched (parent-df coords)."""
        sweep.n_by_sub_id[1] = 2
        slice_begin = 100
        sub = _make_sub(
            1, start_idx=_loh(5), m15_end_idx=_loh(9),
            lenses=(LENS_CONFLUENCE, LENS_COUNTER),
            started_by="first_counter", slice_begin=slice_begin,
            use_case="first_counter", parent_sid=3, parent_cycle_id=2,
        )
        # Sweeping trigger = counter (parent 6 -> LOH 27, the earliest).
        streams = {
            LENS_CONFLUENCE: [(8, "ZONE_PROXIMITY_TRIGGER")],
            LENS_COUNTER: [(6, "SUBSEQUENT_CONFLUENCE_TRIGGER")],
        }
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        # The projection carries the sweep's (slice-local) records as returned.
        assert len(sub.wvmi_records) == 2
        assert sub.wvmi_records[0].fb_idx == 1          # unshifted on the facade

        for lens in (LENS_CONFLUENCE, LENS_COUNTER):
            recs = _wvmi(lens_dfs[lens])
            assert len(recs) == 2, lens
            for k, rec in enumerate(recs):
                # deep copies: not the facade's objects
                assert all(rec is not w for w in sub.wvmi_records)
                # shift by slice_begin: (1,2,3,None) + 100 for k=0; (11,12,13,14) + 100 for k=1
                assert rec.fb_idx == 1 + 10 * k + slice_begin
                assert rec.lb_idx == 2 + 10 * k + slice_begin
                assert rec.fp_idx == 3 + 10 * k + slice_begin
                assert rec.lp_idx == (None if k == 0 else 4 + 10 * k + slice_begin)
                # sweeping lens decides the record's structure_path_id attr
                assert rec.structure_path_id == LENS_PATHS[LENS_COUNTER]
                # per-lens attribution stamped on meta
                assert rec.meta["structure_path_id"] == LENS_PATHS[lens]
                assert rec.meta["sub_id"] == 1
                assert rec.meta["started_by"] == "first_counter"
                assert rec.meta["use_case"] == "first_counter"
                assert rec.meta["parent_sid"] == 3
                assert rec.meta["parent_cycle_id"] == 2
                assert rec.meta["timeframe"] == "M15"
                assert rec.meta["parent_tf"] == "H1"
                # parent-df coords, never translated
                assert rec.meta["triggered_by_event_idx"] == 6
                assert rec.meta["triggered_by_event_type"] == "SUBSEQUENT_CONFLUENCE_TRIGGER"
                assert rec.meta["parent_path_id"] == PARENT_PATH

        # the two lens dfs hold distinct copies
        conf, ctr = _wvmi(lens_dfs[LENS_CONFLUENCE]), _wvmi(lens_dfs[LENS_COUNTER])
        assert all(a is not b for a in conf for b in ctr)

    def test_single_lens_sub_is_persisted_only_into_its_lens(
        self, sweep, lens_dfs, frames,
    ):
        """"every lens df the sub is on" — and no other: a confluence-only sub
        leaves the counter lens df untouched."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_CONFLUENCE,))
        streams = {LENS_CONFLUENCE: [(6, "ZONE_PROXIMITY_TRIGGER")], LENS_COUNTER: []}
        _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        assert len(_wvmi(lens_dfs[LENS_CONFLUENCE])) == 1
        assert "wvmi" not in lens_dfs[LENS_COUNTER].attrs

    def test_persist_appends_to_existing_lens_wvmi(
        self, sweep, lens_dfs, frames,
    ):
        """Records are APPENDED to `attrs["wvmi"]` — a second sub on the same
        lens must not overwrite the first sub's records (two subs swept ->
        two entries on the shared lens df, in sub-list order)."""
        sub_a = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                          lenses=(LENS_CONFLUENCE,))
        sub_b = _make_sub(2, start_idx=_loh(10), m15_end_idx=_loh(14),
                          lenses=(LENS_CONFLUENCE,))
        streams = {
            LENS_CONFLUENCE: [(6, "ZONE_PROXIMITY_TRIGGER"), (12, "ZONE_PROXIMITY_TRIGGER")],
            LENS_COUNTER: [],
        }
        _run([sub_a, sub_b], streams, lens_dfs=lens_dfs, frames=frames)

        recs = _wvmi(lens_dfs[LENS_CONFLUENCE])
        assert [r.meta["sub_id"] for r in recs] == [1, 2]
        assert [r.meta["triggered_by_event_idx"] for r in recs] == [6, 12]


# ---------------------------------------------------------------------------
# Dedup by sub_id, not-swept subs, counts
# ---------------------------------------------------------------------------

class TestDedupAndCounts:
    def test_sub_listed_twice_is_swept_once(self, sweep, lens_dfs, frames):
        """§17.10: dedup key `sub_id` — the same sub appearing twice in the
        projection list (two distinct objects) is swept once, persisted once,
        counted once."""
        first = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                          lenses=(LENS_CONFLUENCE,))
        again = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                          lenses=(LENS_CONFLUENCE,))
        streams = {LENS_CONFLUENCE: [(6, "ZONE_PROXIMITY_TRIGGER")], LENS_COUNTER: []}
        counts = _run([first, again], streams, lens_dfs=lens_dfs, frames=frames)

        assert len(sweep.calls) == 1
        assert sweep.calls[0].result is first
        assert len(first.wvmi_records) == 1
        assert again.wvmi_records == []
        assert len(_wvmi(lens_dfs[LENS_CONFLUENCE])) == 1
        assert counts["acted"] == 1
        assert counts["records"] == 1

    def test_sub_with_no_trigger_in_window_is_not_swept(
        self, sweep, lens_dfs, frames,
    ):
        """§17.10: "No entry inside the window -> no WVMI for that sub" —
        triggers on the sub's own lens exist before and after the window; no
        sweep, no persist, zero counts."""
        sub = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                        lenses=(LENS_CONFLUENCE, LENS_COUNTER))
        streams = {
            LENS_CONFLUENCE: [(2, "ZONE_PROXIMITY_TRIGGER"),        # LOH 11 < 23
                              (11, "SUBSEQUENT_COUNTER_TRIGGER")],  # LOH 47 > 39
            LENS_COUNTER: [(3, "SUBSEQUENT_CONFLUENCE_TRIGGER")],   # LOH 15 < 23
        }
        counts = _run([sub], streams, lens_dfs=lens_dfs, frames=frames)

        assert sweep.calls == []
        assert sub.wvmi_records == []
        assert _wvmi(lens_dfs[LENS_CONFLUENCE]) == []
        assert _wvmi(lens_dfs[LENS_COUNTER]) == []
        assert counts == {"acted": 0, "records": 0, "by_started_by": {}, "by_lens": {}}

    def test_counts_by_started_by_and_by_lens(self, sweep, lens_dfs, frames):
        """Counts derived from the rule: `acted` = subs whose sweep produced
        records; `records` = total records; `by_started_by[sb]` += the sub's
        record count; `by_lens[l]` += the sub's record count for EVERY lens
        the sub is on (records land on every lens df).

          A: first_confluence, {confluence}, 2 records
          B: first_counter,    {confluence, counter}, 3 records
          C: reversal,         {counter}, no trigger in window -> not counted
        -> acted 2, records 5, by_started_by {fc: 2, fcounter: 3},
           by_lens {confluence: 2 + 3 = 5, counter: 3}."""
        sweep.n_by_sub_id.update({1: 2, 2: 3, 3: 4})
        sub_a = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                          lenses=(LENS_CONFLUENCE,), started_by="first_confluence")
        sub_b = _make_sub(2, start_idx=_loh(10), m15_end_idx=_loh(14),
                          lenses=(LENS_CONFLUENCE, LENS_COUNTER), started_by="first_counter")
        sub_c = _make_sub(3, start_idx=_loh(20), m15_end_idx=_loh(24),
                          lenses=(LENS_COUNTER,), started_by="reversal")
        streams = {
            LENS_CONFLUENCE: [(6, "ZONE_PROXIMITY_TRIGGER"), (12, "ZONE_PROXIMITY_TRIGGER")],
            LENS_COUNTER: [(30, "SUBSEQUENT_CONFLUENCE_TRIGGER")],   # LOH 123 > C's end 99
        }
        counts = _run([sub_a, sub_b, sub_c], streams, lens_dfs=lens_dfs, frames=frames)

        assert counts["acted"] == 2
        assert counts["records"] == 5
        assert counts["by_started_by"] == {"first_confluence": 2, "first_counter": 3}
        assert counts["by_lens"] == {LENS_CONFLUENCE: 5, LENS_COUNTER: 3}
        # and the lens dfs agree with by_lens
        assert len(_wvmi(lens_dfs[LENS_CONFLUENCE])) == 5
        assert len(_wvmi(lens_dfs[LENS_COUNTER])) == 3

    def test_sweep_yielding_no_records_marks_the_sub_swept_but_not_acted(
        self, sweep, lens_dfs, frames,
    ):
        """A sub whose sweep returns no records (no CTS_CONFIRMED passed the
        wave-candle guards) is still SWEPT (dedup: a repeat listing does not
        re-sweep it) but does not count as `acted` and adds no records."""
        sweep.n_by_sub_id[1] = 0
        first = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                          lenses=(LENS_CONFLUENCE,))
        again = _make_sub(1, start_idx=_loh(5), m15_end_idx=_loh(9),
                          lenses=(LENS_CONFLUENCE,))
        streams = {LENS_CONFLUENCE: [(6, "ZONE_PROXIMITY_TRIGGER")], LENS_COUNTER: []}
        counts = _run([first, again], streams, lens_dfs=lens_dfs, frames=frames)

        assert len(sweep.calls) == 1
        assert counts == {"acted": 0, "records": 0, "by_started_by": {}, "by_lens": {}}
        assert _wvmi(lens_dfs[LENS_CONFLUENCE]) == []
