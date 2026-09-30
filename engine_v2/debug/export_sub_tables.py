"""Sub-structure pool tables → CSV (PART4 §17.9 exports).

Per lens df: `attrs["sids"]` (one `SidRecord` per unique sub rendered on that
lens), `attrs["triggers"]` (that lens's `TriggerRecord`s, incl. zero-length)
and `attrs["unresolved_triggers"]` (pool-wide, the same list on every lens
df). Written DECOUPLED from the chart loop — `run_replay.main` calls this
BEFORE the M15 chart exports, inside its own try/except, so a chart-export
exception cannot lose the tables.

Files (per replay basename):
  `{basename}_M15_{lens}_subs.csv`        one row per sub on the lens
  `{basename}_M15_{lens}_triggers.csv`    one row per record on the lens
  `{basename}_M15_unresolved_triggers.csv` one row per unresolved trigger (pool-wide, written once)

A header row is always written, even for an empty table. `lenses`,
`relative_dir_segments` and `extra_trigger_idxs` are serialised as their
Python `repr` strings; `source_trigger` is never exported.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from engine_v2.multitf.sub_structure_pool import TriggerRecord, UnresolvedTrigger


_SUBS_COLUMNS = [
    "sub_id", "direction", "starting_idx", "start_idx", "end_idx", "end_reason",
    "natural_reversal_idx", "lenses", "relative_dir_segments", "n_records",
    "first_record_lens", "first_record_parent_sid", "first_record_parent_cycle_id",
    "first_record_trigger_type", "first_record_trigger_idx",
]
_TRIGGER_COLUMNS = [f.name for f in fields(TriggerRecord) if f.name != "source_trigger"] + ["is_zero_length"]
_UNRESOLVED_COLUMNS = [f.name for f in fields(UnresolvedTrigger)]


def _sub_row(rec: Any) -> Dict[str, Any]:
    meta = rec.meta or {}
    fr = meta.get("first_record") or {}
    return {
        "sub_id": rec.sub_id,
        "direction": rec.starting_sd,
        "starting_idx": rec.creation_event_idx,
        "start_idx": rec.start_idx,
        "end_idx": rec.end_event_idx,
        "end_reason": rec.end_reason,
        "natural_reversal_idx": meta.get("natural_reversal_idx"),
        "lenses": repr(tuple(rec.lenses or ())),
        "relative_dir_segments": repr(tuple(rec.relative_dir_segments or ())),
        "n_records": meta.get("n_records"),
        "first_record_lens": fr.get("lens"),
        "first_record_parent_sid": fr.get("parent_sid"),
        "first_record_parent_cycle_id": fr.get("parent_cycle_id"),
        "first_record_trigger_type": fr.get("trigger_type"),
        "first_record_trigger_idx": fr.get("trigger_idx"),
    }


def _trigger_row(r: TriggerRecord) -> Dict[str, Any]:
    row = {}
    for name in _TRIGGER_COLUMNS:
        if name == "is_zero_length":
            row[name] = r.is_zero_length
        elif name == "extra_trigger_idxs":
            row[name] = repr(list(r.extra_trigger_idxs))
        else:
            row[name] = getattr(r, name)
    return row


def _unresolved_row(u: UnresolvedTrigger) -> Dict[str, Any]:
    return {name: getattr(u, name) for name in _UNRESOLVED_COLUMNS}


def export_sub_tables(
    lens_dfs: Dict[str, pd.DataFrame],
    basename: str,
    out_dir: str | Path = "artifacts/debug",
) -> Dict[str, Path]:
    """Write the three pool tables for `basename`. Returns `{label: path}`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    unresolved: Optional[List[UnresolvedTrigger]] = None
    for lens, df in lens_dfs.items():
        attrs = getattr(df, "attrs", {}) or {}
        sids = list(attrs.get("sids", []) or [])
        triggers = list(attrs.get("triggers", []) or [])
        if unresolved is None and "unresolved_triggers" in attrs:
            unresolved = list(attrs.get("unresolved_triggers") or [])

        p_subs = out_dir / f"{basename}_M15_{lens}_subs.csv"
        pd.DataFrame([_sub_row(s) for s in sids], columns=_SUBS_COLUMNS).to_csv(p_subs, index=False)
        written[f"{lens}_subs"] = p_subs

        p_trig = out_dir / f"{basename}_M15_{lens}_triggers.csv"
        pd.DataFrame(
            [_trigger_row(r) for r in sorted(triggers, key=lambda r: r.seq)],
            columns=_TRIGGER_COLUMNS,
        ).to_csv(p_trig, index=False)
        written[f"{lens}_triggers"] = p_trig

    p_unres = out_dir / f"{basename}_M15_unresolved_triggers.csv"
    pd.DataFrame(
        [_unresolved_row(u) for u in (unresolved or [])], columns=_UNRESOLVED_COLUMNS,
    ).to_csv(p_unres, index=False)
    written["unresolved_triggers"] = p_unres
    return written
