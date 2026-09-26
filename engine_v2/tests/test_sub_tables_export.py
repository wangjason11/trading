"""Sub-table CSV exports (Plan C §6.3 / PART4 §17.9 "Exports").

`debug/export_sub_tables.export_sub_tables(lens_dfs, basename, out_dir)` writes,
per lens df, `{basename}_M15_{lens}_subs.csv` (one row per unique sub on that
lens, from `attrs["sids"]`) and `{basename}_M15_{lens}_triggers.csv` (one row
per record on that lens incl. zero-length ones, from `attrs["triggers"]`), plus
ONE pool-wide `{basename}_M15_unresolved_triggers.csv` (from
`attrs["unresolved_triggers"]`, the same object on every lens df). Every file
gets a header row even when its list is empty; `lenses` /
`relative_dir_segments` / `extra_trigger_idxs` are serialised as strings.

The exports are decoupled from the chart loop: `run_replay.main` calls
`export_sub_tables` BEFORE the M15 chart loop so a chart-export exception
cannot prevent them (checked here as a source-order guard).
"""
from __future__ import annotations

import ast
import csv
import json
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

import engine_v2
from engine_v2.debug.export_sub_tables import export_sub_tables
from engine_v2.multitf.sub_structure_pool import TriggerRecord, UnresolvedTrigger
from engine_v2.multitf.types import SidRecord


# --- fixtures ------------------------------------------------------------------

_SUBS_COLUMNS = [
    "sub_id", "direction", "starting_idx", "start_idx", "end_idx", "end_reason",
    "natural_reversal_idx", "lenses", "relative_dir_segments", "n_records",
    "first_record_lens", "first_record_parent_sid", "first_record_parent_cycle_id",
    "first_record_trigger_type", "first_record_trigger_idx",
]

# `_triggers.csv` = every TriggerRecord field in declaration order except the
# opaque `source_trigger`, plus the derived `is_zero_length`.
_TRIGGER_FIELDS = [f.name for f in _dc_fields(TriggerRecord) if f.name != "source_trigger"]
_UNRESOLVED_FIELDS = [f.name for f in _dc_fields(UnresolvedTrigger)]


def _sub(sub_id, direction, starting_idx, start_idx, end_idx, end_reason,
         natural_reversal_idx, lenses, segments, n_records, first_record) -> SidRecord:
    """A §2.5 sub-level SidRecord (sub_sid None, sub_id set, parent fields
    None; `first_record` = the 6-key dict of the sub's first live record)."""
    return SidRecord(
        sub_sid=None, starting_sd=direction,
        creation_event_idx=starting_idx, end_event_idx=end_idx, end_reason=end_reason,
        parent_sid=None, parent_cycle_id=None,
        sub_id=sub_id, start_idx=start_idx,
        lenses=tuple(lenses), relative_dir_segments=tuple(segments),
        meta={
            "natural_reversal_idx": natural_reversal_idx,
            "n_records": n_records,
            "first_record": dict(first_record),
            "slice_begin": starting_idx - 50,
        },
    )


class _OpaqueTrigger:
    """Stands in for the MultiTFTrigger on `source_trigger` — must NOT be
    serialised (the column is omitted)."""
    def __init__(self):
        self.use_case = "subsequent_confluence"
        self.marker = "SOURCE_TRIGGER_MUST_NOT_BE_EXPORTED"


def _confluence_fixture() -> Tuple[List[SidRecord], List[TriggerRecord]]:
    """Predicted-table rows on the CONFLUENCE lens
    (`reference_pool_redesign_groundtruth.md`; sub_id in creation order:
    3304/-1 -> 4, 3760/-1 -> 6, 4027/+1 -> 7; H1 sid 1 parent_sd = -1).

    sub 6 = 3760/-1: conf (1,2) tss1 subsequent_confluence, trigger 3819 =
      LOH(954) = 4*954+3, finalize == trigger (non-FC), floor 3611 ->
      start = max(3819, 3819, 3611) = 3819; own reversal 4200 -> trigger_end
      4200, end = max(4200, 3819) = 4200, reason reversal. Sub [3819, 4200].
    sub 7 = 4027/+1: conf (1,2) tss2 reversal-born from sub 6 at R=4200 ->
      trigger = finalize = 4200, start = max(4200, 4200, 3611) = 4200, open;
      relative_dir = counter (+1 != parent_sd -1). Sub window [4083, None]
      (its counter-lens subsequent_counter record started at 4083).
    zero-length (HYPOTHETICAL, not on the window): a (1,2) tss3
      subsequent_confluence re-trigger of sub 4 (3304/-1, frozen at 3819 by
      sub 6) at trigger 3903 = LOH(975) = 4*975+3 -> start = 3903 > sub.end
      3819 -> post-end re-trigger (§4.3 step 6): trigger_end = 3819 (frozen),
      end = max(3819, 3903) = 3903, reason/ended_by copied
      (same_dir_replacement / 6); is_zero_length = 3819 <= 3903 = True.
    """
    sids = [
        _sub(6, -1, 3760, 3819, 4200, "reversal", 4200, ("confluence",),
             ((3819, "confluence"),), 1,
             {"lens": "confluence", "parent_sid": 1, "parent_cycle_id": 2,
              "trigger_type": "subsequent_confluence", "trigger_idx": 3819,
              "start_idx": 3819}),
        _sub(7, +1, 4027, 4083, None, None, None, ("counter", "confluence"),
             ((4083, "counter"),), 2,
             {"lens": "counter", "parent_sid": 1, "parent_cycle_id": 2,
              "trigger_type": "subsequent_counter", "trigger_idx": 4083,
              "start_idx": 4083}),
    ]
    # `parent_bos_anchor_idx` values below are PASS-THROUGH stubs (the exporter writes what
    # it is given). The real rule — FC-only, None for sibling / reversal types (PLAN_E Q4) —
    # is pinned in test_first_trigger_migration / test_reversal_resolver.
    triggers = [
        TriggerRecord(
            lens="confluence", parent_sid=1, parent_cycle_id=2, trigger_sub_sid=1,
            sub_id=6, trigger_type="subsequent_confluence", trigger_idx=3819,
            probe_finalize_idx=3819, probe_finalize_condition="no_retrace",
            parent_bos_anchor_idx=954, probe_input_idx=3760, starting_idx=3760, direction=-1, sub_tf="M15",
            relative_dir="confluence", parent_floor_idx=3611, start_idx=3819,
            source_trigger=_OpaqueTrigger(),
            trigger_end_idx=4200, end_idx=4200, end_reason="reversal",
            ended_by_sub_id=None, seq=8,
            # HYPOTHETICAL later (1,2) confluence triggers absorbed into this record.
            extra_trigger_idxs=[3823, 3827],
        ),
        TriggerRecord(
            lens="confluence", parent_sid=1, parent_cycle_id=2, trigger_sub_sid=2,
            sub_id=7, trigger_type="reversal", trigger_idx=4200,
            probe_finalize_idx=4200, probe_finalize_condition="reversal_handoff",
            parent_bos_anchor_idx=None, probe_input_idx=4000, starting_idx=4027, direction=1, sub_tf="M15",
            relative_dir="counter", parent_floor_idx=3611, start_idx=4200,
            source_trigger=_OpaqueTrigger(),
            trigger_end_idx=None, end_idx=None, end_reason=None,
            ended_by_sub_id=None, seq=10, extra_trigger_idxs=[],
        ),
        TriggerRecord(
            lens="confluence", parent_sid=1, parent_cycle_id=2, trigger_sub_sid=3,
            sub_id=4, trigger_type="subsequent_confluence", trigger_idx=3903,
            probe_finalize_idx=3903, probe_finalize_condition="no_retrace",
            parent_bos_anchor_idx=975, probe_input_idx=3304, starting_idx=3304, direction=-1, sub_tf="M15",
            relative_dir="confluence", parent_floor_idx=3611, start_idx=3903,
            source_trigger=_OpaqueTrigger(),
            trigger_end_idx=3819, end_idx=3903, end_reason="same_dir_replacement",
            ended_by_sub_id=6, seq=11, extra_trigger_idxs=[],
        ),
    ]
    assert [t.is_zero_length for t in triggers] == [False, False, True]   # fixture sanity
    return sids, triggers


def _unresolved_fixture() -> List[UnresolvedTrigger]:
    """The predicted table's 4 unresolved rows (degenerate cycles (1,0), (1,1)):
    FC(1,0) trigger 2815 = LOH(703); FC(1,1) 2995 = LOH(748); first_counter(1,1)
    3283 = LOH(820); subsequent_confluence(1,1) 3487 = LOH(871). Directions:
    parent sd of sid 1 is -1 -> confluence types -1, first_counter +1.
    `probe_input_idx` = the FC's H1 BOS idx (689 / 728) where known."""
    return [
        UnresolvedTrigger("confluence", 1, 0, "first_confluence", 2815, -1, 689,
                          "degenerate_parent_cycle", "floor 3611 >= end 3611"),
        UnresolvedTrigger("confluence", 1, 1, "first_confluence", 2995, -1, 728,
                          "degenerate_parent_cycle", "floor 3611 >= end 3611"),
        UnresolvedTrigger("counter", 1, 1, "first_counter", 3283, 1, None,
                          "degenerate_parent_cycle", "floor 3611 >= end 3611"),
        UnresolvedTrigger("confluence", 1, 1, "subsequent_confluence", 3487, -1, None,
                          "degenerate_parent_cycle", "floor 3611 >= end 3611"),
    ]


def _lens_df(sids, triggers, unresolved) -> pd.DataFrame:
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC"),
                       "close": [0.6, 0.61, 0.62]})
    df.attrs["sids"] = list(sids)
    df.attrs["triggers"] = list(triggers)
    if unresolved is not None:
        df.attrs["unresolved_triggers"] = unresolved     # same object on every lens df
    return df


@pytest.fixture
def lens_dfs():
    """Confluence populated; counter EMPTY (no records at all); the pool-wide
    unresolved list is the same object on both dfs."""
    sids, triggers = _confluence_fixture()
    unresolved = _unresolved_fixture()
    return {
        "confluence": _lens_df(sids, triggers, unresolved),
        "counter": _lens_df([], [], unresolved),
    }


# --- CSV helpers -------------------------------------------------------------------

def _read(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r]     # drop blank lines
    assert rows, f"{path.name} has no header row"
    header = rows[0]
    return header, [dict(zip(header, r)) for r in rows[1:]]


def _num(cell: str) -> Optional[float]:
    """A numeric CSV cell; None-ish spellings ('' / 'None' / 'nan') -> None.
    Integer columns that also hold None may be written as floats ('4200.0')."""
    s = cell.strip()
    if s in ("", "None", "nan", "NaN"):
        return None
    return float(s)


def _seq(cell: str) -> Any:
    """Parse a 'JSON-ish' serialised sequence (json.dumps or str(list/tuple))
    and normalise tuples to lists so both spellings compare equal."""
    try:
        val = json.loads(cell)
    except (ValueError, TypeError):
        val = ast.literal_eval(cell)

    def _norm(v):
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        return v
    return _norm(val)


# --- file set ----------------------------------------------------------------------

def test_writes_subs_and_triggers_per_lens_and_one_unresolved_file(lens_dfs, tmp_path):
    out = export_sub_tables(lens_dfs, "base", tmp_path)

    expected = {
        "base_M15_confluence_subs.csv",
        "base_M15_confluence_triggers.csv",
        "base_M15_counter_subs.csv",
        "base_M15_counter_triggers.csv",
        "base_M15_unresolved_triggers.csv",
    }
    written = {p.name for p in tmp_path.iterdir()}
    assert written == expected
    # The returned {label: path} map covers exactly those files.
    assert {Path(p).name for p in out.values()} == expected
    assert all(Path(p).exists() for p in out.values())
    # The old per-lens `_sids.csv` is gone (its role is split across the two).
    assert not any(n.endswith("_sids.csv") for n in written)


def test_unresolved_file_written_once_pool_wide(lens_dfs, tmp_path):
    """Both lens dfs carry the SAME unresolved list; it is written ONCE — one
    file, 4 rows (not 8), every UnresolvedTrigger field in declaration order."""
    export_sub_tables(lens_dfs, "base", tmp_path)
    files = sorted(tmp_path.glob("*unresolved*"))
    assert [p.name for p in files] == ["base_M15_unresolved_triggers.csv"]
    header, rows = _read(files[0])
    assert header == _UNRESOLVED_FIELDS
    assert len(rows) == 4
    assert [r["trigger_type"] for r in rows] == [
        "first_confluence", "first_confluence", "first_counter", "subsequent_confluence",
    ]
    assert [_num(r["trigger_idx"]) for r in rows] == [2815, 2995, 3283, 3487]
    assert [r["lens"] for r in rows] == ["confluence", "confluence", "counter", "confluence"]
    assert [_num(r["direction"]) for r in rows] == [-1, -1, 1, -1]
    assert [_num(r["probe_input_idx"]) for r in rows] == [689, 728, None, None]
    assert all(r["reason"] == "degenerate_parent_cycle" for r in rows)
    assert all(r["detail"] == "floor 3611 >= end 3611" for r in rows)


def test_unresolved_file_has_header_when_no_lens_carries_the_attr(tmp_path):
    sids, triggers = _confluence_fixture()
    dfs = {"confluence": _lens_df(sids, triggers, None), "counter": _lens_df([], [], None)}
    export_sub_tables(dfs, "base", tmp_path)
    header, rows = _read(tmp_path / "base_M15_unresolved_triggers.csv")
    assert header == _UNRESOLVED_FIELDS
    assert rows == []


# --- _subs.csv ---------------------------------------------------------------------

def test_subs_csv_columns_and_one_row_per_sid_record(lens_dfs, tmp_path):
    export_sub_tables(lens_dfs, "base", tmp_path)
    header, rows = _read(tmp_path / "base_M15_confluence_subs.csv")
    assert header == _SUBS_COLUMNS
    assert len(rows) == 2                                   # one per SidRecord
    assert [_num(r["sub_id"]) for r in rows] == [6, 7]
    assert [_num(r["direction"]) for r in rows] == [-1, 1]
    assert [_num(r["starting_idx"]) for r in rows] == [3760, 4027]
    assert [_num(r["start_idx"]) for r in rows] == [3819, 4083]
    assert [_num(r["end_idx"]) for r in rows] == [4200, None]
    assert rows[0]["end_reason"] == "reversal"
    assert rows[1]["end_reason"] in ("", "None")            # open sub
    assert [_num(r["natural_reversal_idx"]) for r in rows] == [4200, None]
    assert [_num(r["n_records"]) for r in rows] == [1, 2]
    assert [r["first_record_lens"] for r in rows] == ["confluence", "counter"]
    assert [_num(r["first_record_parent_sid"]) for r in rows] == [1, 1]
    assert [_num(r["first_record_parent_cycle_id"]) for r in rows] == [2, 2]
    assert [r["first_record_trigger_type"] for r in rows] == [
        "subsequent_confluence", "subsequent_counter",
    ]
    assert [_num(r["first_record_trigger_idx"]) for r in rows] == [3819, 4083]


def test_subs_csv_serialises_lenses_and_segments_as_strings(lens_dfs, tmp_path):
    export_sub_tables(lens_dfs, "base", tmp_path)
    _header, rows = _read(tmp_path / "base_M15_confluence_subs.csv")
    # Cells are strings that parse back to the sequences (json.dumps or str()).
    # PLAN-AMBIGUITY: the plan/contract say "JSON-ish str" without fixing
    # json.dumps vs str(); both are accepted here, content is pinned.
    assert isinstance(rows[0]["lenses"], str) and rows[0]["lenses"]
    assert _seq(rows[0]["lenses"]) == ["confluence"]
    assert sorted(_seq(rows[1]["lenses"])) == ["confluence", "counter"]
    assert _seq(rows[0]["relative_dir_segments"]) == [[3819, "confluence"]]
    assert _seq(rows[1]["relative_dir_segments"]) == [[4083, "counter"]]


def test_empty_lens_writes_header_only_files(lens_dfs, tmp_path):
    """The counter lens has no records: both of its files exist with the full
    header row and zero data rows."""
    export_sub_tables(lens_dfs, "base", tmp_path)
    header, rows = _read(tmp_path / "base_M15_counter_subs.csv")
    assert header == _SUBS_COLUMNS
    assert rows == []
    header, rows = _read(tmp_path / "base_M15_counter_triggers.csv")
    assert [c for c in header if c != "is_zero_length"] == _TRIGGER_FIELDS
    assert "is_zero_length" in header
    assert rows == []


# --- _triggers.csv -----------------------------------------------------------------

def test_triggers_csv_columns_every_field_except_source_trigger_plus_is_zero_length(
    lens_dfs, tmp_path,
):
    export_sub_tables(lens_dfs, "base", tmp_path)
    header, rows = _read(tmp_path / "base_M15_confluence_triggers.csv")
    assert "source_trigger" not in header
    assert "is_zero_length" in header
    # Declaration order for the dataclass fields; `is_zero_length` is the only
    # extra column (appended — its position is not load-bearing).
    assert [c for c in header if c != "is_zero_length"] == _TRIGGER_FIELDS
    assert set(header) == set(_TRIGGER_FIELDS) | {"is_zero_length"}
    # Spot-check the names the plan cares about are really there.
    for c in ("lens", "parent_sid", "parent_cycle_id", "trigger_sub_sid", "sub_id",
              "trigger_type", "trigger_idx", "probe_finalize_idx",
              "probe_finalize_condition", "parent_bos_anchor_idx", "probe_input_idx", "starting_idx",
              "direction", "sub_tf", "relative_dir", "parent_floor_idx", "start_idx",
              "trigger_end_idx", "end_idx", "end_reason", "ended_by_sub_id", "seq",
              "extra_trigger_idxs"):
        assert c in header, c
    assert len(rows) == 3                                   # incl. the zero-length record


def test_triggers_csv_rows_values(lens_dfs, tmp_path):
    export_sub_tables(lens_dfs, "base", tmp_path)
    _header, rows = _read(tmp_path / "base_M15_confluence_triggers.csv")
    r6, r7, rz = rows
    assert (_num(r6["sub_id"]), _num(r6["trigger_sub_sid"])) == (6, 1)
    assert (_num(r6["trigger_idx"]), _num(r6["probe_finalize_idx"])) == (3819, 3819)
    assert (_num(r6["parent_floor_idx"]), _num(r6["start_idx"])) == (3611, 3819)
    assert (_num(r6["trigger_end_idx"]), _num(r6["end_idx"])) == (4200, 4200)
    assert r6["end_reason"] == "reversal"
    assert r6["relative_dir"] == "confluence"
    assert r6["probe_finalize_condition"] == "no_retrace"
    assert _num(r6["parent_bos_anchor_idx"]) == 954
    assert _num(r6["probe_input_idx"]) == 3760
    assert r6["sub_tf"] == "M15"
    assert _num(r6["seq"]) == 8

    assert (_num(r7["sub_id"]), _num(r7["trigger_sub_sid"])) == (7, 2)
    assert r7["trigger_type"] == "reversal"
    assert _num(r7["parent_bos_anchor_idx"]) is None          # reversal-born
    assert _num(r7["probe_input_idx"]) == 4000                # the reversal handoff's M15 input
    assert _num(r7["trigger_end_idx"]) is None and _num(r7["end_idx"]) is None
    assert r7["end_reason"] in ("", "None")
    assert r7["relative_dir"] == "counter"

    assert (_num(rz["sub_id"]), _num(rz["trigger_sub_sid"])) == (4, 3)
    assert (_num(rz["start_idx"]), _num(rz["trigger_end_idx"]), _num(rz["end_idx"])) == (3903, 3819, 3903)
    assert rz["end_reason"] == "same_dir_replacement"
    assert _num(rz["ended_by_sub_id"]) == 6

    # The opaque source trigger never leaks into any cell.
    for row in rows:
        assert all("SOURCE_TRIGGER_MUST_NOT_BE_EXPORTED" not in v for v in row.values())


def test_triggers_csv_is_zero_length_and_extra_trigger_idxs(lens_dfs, tmp_path):
    export_sub_tables(lens_dfs, "base", tmp_path)
    _header, rows = _read(tmp_path / "base_M15_confluence_triggers.csv")
    assert [r["is_zero_length"].strip().lower() for r in rows] == ["false", "false", "true"]
    # extra_trigger_idxs serialised as a string sequence.
    assert isinstance(rows[0]["extra_trigger_idxs"], str)
    assert _seq(rows[0]["extra_trigger_idxs"]) == [3823, 3827]
    assert _seq(rows[1]["extra_trigger_idxs"]) == []
    assert _seq(rows[2]["extra_trigger_idxs"]) == []


def test_lens_filter_is_the_caller_s_responsibility_rows_are_written_as_given(tmp_path):
    """Each lens df's `attrs["triggers"]` already holds ONLY that lens's
    records (§6.3); the exporter writes what it is given, per lens."""
    sids, triggers = _confluence_fixture()
    ctr_rec = TriggerRecord(
        lens="counter", parent_sid=1, parent_cycle_id=2, trigger_sub_sid=1,
        sub_id=7, trigger_type="subsequent_counter", trigger_idx=4083,
        probe_finalize_idx=4083, probe_finalize_condition="no_retrace",
        parent_bos_anchor_idx=1020, probe_input_idx=4000, starting_idx=4027, direction=1, sub_tf="M15",
        relative_dir="counter", parent_floor_idx=3611, start_idx=4083,
        seq=9,
    )
    dfs = {
        "confluence": _lens_df(sids, triggers, []),
        "counter": _lens_df([sids[1]], [ctr_rec], []),
    }
    export_sub_tables(dfs, "base", tmp_path)
    _h, conf_rows = _read(tmp_path / "base_M15_confluence_triggers.csv")
    _h, ctr_rows = _read(tmp_path / "base_M15_counter_triggers.csv")
    assert [r["lens"] for r in conf_rows] == ["confluence"] * 3
    assert [r["lens"] for r in ctr_rows] == ["counter"]
    assert _num(ctr_rows[0]["sub_id"]) == 7
    _h, ctr_subs = _read(tmp_path / "base_M15_counter_subs.csv")
    assert [_num(r["sub_id"]) for r in ctr_subs] == [7]


# --- run_replay ordering (SOURCE-ORDER GUARD) ----------------------------------------

def test_run_replay_calls_export_sub_tables_before_the_m15_chart_loop():
    """SOURCE-ORDER GUARD (cheap, textual — not a behavioural test): §6.3 says
    the sub-table CSVs are decoupled from the chart loop and written even if the
    chart export raises. `run_replay.main` must therefore call
    `export_sub_tables(` BEFORE the first `export_m15_chart_plotly(` call in
    the file. Reads the source text; does not import `run_replay` (OANDA)."""
    src_path = Path(engine_v2.__file__).resolve().parent / "run_replay.py"
    src = src_path.read_text(encoding="utf-8")
    i_tables = src.find("export_sub_tables(")
    i_chart = src.find("export_m15_chart_plotly(")
    assert i_tables != -1, "run_replay.py does not call export_sub_tables("
    assert i_chart != -1, "run_replay.py does not call export_m15_chart_plotly("
    assert i_tables < i_chart, (
        "export_sub_tables( must be called before the M15 chart loop "
        f"(found at {i_tables} vs chart call at {i_chart})"
    )
