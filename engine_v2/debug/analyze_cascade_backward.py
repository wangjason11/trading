"""Find all KL zones where the cascade left end_time < start_time.

For each such zone, report:
- entity (M15.confluence / M15.counter)
- parent_sid, parent_cycle_id
- entity_sid (the zone's owner)
- overwriter_sid (from deactivated_by tag)
- new_sid_creation_idx (M15 idx where the overwriter sid's slice begins;
  = cascade_boundary_idx)
- new_sid_trigger_event_idx (parent-TF idx where the overwriter's trigger
  fired — looked up from the sub's SidRecord meta)
- base_idx, start_idx (= bounds_steps[0].start_idx), confirmed_idx
- end_idx (the cascade boundary — same as new_sid_creation_idx)

Usage: python -m engine_v2.debug.analyze_cascade_backward <basename>
e.g.   python -m engine_v2.debug.analyze_cascade_backward \\
         NZD_USD_H1_2025-11-15_2026-01-07_sd-1_eps0p0001_rk2-5
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

DEBUG_DIR = Path("artifacts/debug")


def _parse_meta(s: str) -> dict:
    """Meta column is repr'd as a Python dict string."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def analyze_entity(basename: str, leaf_label: str) -> list[dict]:
    kl_path = DEBUG_DIR / f"{basename}_{leaf_label}_kl_zones.csv"
    sids_path = DEBUG_DIR / f"{basename}_{leaf_label}_sids.csv"

    if not kl_path.exists():
        print(f"  [skip] no KL zone file: {kl_path.name}")
        return []

    kl_df = pd.read_csv(kl_path)
    kl_df["meta_dict"] = kl_df["meta"].apply(_parse_meta)

    sids_df = pd.DataFrame()
    sid_meta_by_id: dict[int, dict] = {}
    sid_record_by_id: dict[int, dict] = {}
    if sids_path.exists():
        sids_df = pd.read_csv(sids_path)
        sids_df["meta_dict"] = sids_df["meta"].apply(_parse_meta)
        for _, r in sids_df.iterrows():
            sid_meta_by_id[int(r["sid"])] = r["meta_dict"]
            sid_record_by_id[int(r["sid"])] = r.to_dict()

    rows: list[dict] = []
    for _, r in kl_df.iterrows():
        start_t = pd.to_datetime(r["start_time"], utc=True, errors="coerce")
        end_t = pd.to_datetime(r["end_time"], utc=True, errors="coerce")
        if pd.isna(end_t):
            continue
        if pd.isna(start_t):
            continue
        if end_t >= start_t:
            continue  # well-ordered — not a bug instance

        m = r["meta_dict"]
        entity_sid = m.get("entity_sid")
        overwriter_sid = m.get("cascade_overwriter_sid")
        if overwriter_sid is None:
            # fall back: parse from deactivated_by tag
            deactivated_by = m.get("deactivated_by", "") or ""
            if deactivated_by.startswith("overwritten_by_sid_"):
                try:
                    overwriter_sid = int(deactivated_by.rsplit("_", 1)[-1])
                except Exception:
                    overwriter_sid = None

        boundary_idx = m.get("cascade_boundary_idx")
        overwriter_meta = sid_meta_by_id.get(int(overwriter_sid), {}) if overwriter_sid is not None else {}
        overwriter_record = sid_record_by_id.get(int(overwriter_sid), {}) if overwriter_sid is not None else {}

        # fall back: derive boundary from the overwriter sid's creation_event_idx
        if boundary_idx is None and overwriter_record:
            boundary_idx = overwriter_record.get("creation_event_idx")

        new_sid_trigger_idx = overwriter_meta.get("trigger_event_idx")
        new_sid_use_case = overwriter_meta.get("use_case")

        # Owner sid's starting idx (creation_event_idx in the entity's sid
        # record). This is the M15 slice begin of the sub that produced
        # this zone.
        entity_record = sid_record_by_id.get(int(entity_sid), {}) if entity_sid is not None else {}
        entity_sid_creation_idx = entity_record.get("creation_event_idx")

        steps = m.get("bounds_steps", [])
        first_step_start = (
            int(steps[0].get("start_idx")) if steps and "start_idx" in steps[0] else None
        )

        rows.append({
            "entity": leaf_label,
            "parent_sid": m.get("parent_sid"),
            "parent_cycle_id": m.get("parent_cycle_id"),
            "entity_sid": entity_sid,
            "internal_sid": m.get("structure_id"),
            "internal_cycle_id": m.get("cycle_id"),
            "struct_direction": entity_record.get("starting_sd"),
            "entity_sid_creation_idx": entity_sid_creation_idx,
            "use_case": m.get("use_case"),
            "overwriter_sid": overwriter_sid,
            "overwriter_struct_direction": (
                sid_record_by_id.get(int(overwriter_sid), {}).get("starting_sd")
                if overwriter_sid is not None else None
            ),
            "overwriter_use_case": new_sid_use_case,
            "new_sid_creation_idx": boundary_idx,
            "new_sid_trigger_event_idx": new_sid_trigger_idx,
            "base_idx": m.get("base_idx"),
            "first_step_start_idx": first_step_start,
            "confirmed_idx": m.get("confirmed_idx"),
            "end_idx_from_cascade": boundary_idx,
            "side": r["side"],
            "source_kind": r["source_kind"],
            "start_time": start_t,
            "end_time": end_t,
        })

    return rows


def main(basename: str) -> None:
    print(f"Analyzing cascade backward-rectangle artifacts for: {basename}")
    print()

    all_rows: list[dict] = []
    for leaf in ("M15_counter", "M15_confluence"):
        print(f"== {leaf} ==")
        rows = analyze_entity(basename, leaf)
        print(f"  backward-rectangle count: {len(rows)}")
        all_rows.extend(rows)
        print()

    if not all_rows:
        print("No backward-rectangle KL zones found.")
        return

    out_df = pd.DataFrame(all_rows)
    print("=== Backward-rectangle KL zones (end_time < start_time) ===")
    cols = [
        "entity",
        "parent_sid",
        "parent_cycle_id",
        "entity_sid",
        "internal_sid",
        "internal_cycle_id",
        "struct_direction",
        "entity_sid_creation_idx",
        "use_case",
        "overwriter_sid",
        "overwriter_struct_direction",
        "overwriter_use_case",
        "new_sid_creation_idx",
        "new_sid_trigger_event_idx",
        "base_idx",
        "first_step_start_idx",
        "confirmed_idx",
        "end_idx_from_cascade",
        "side",
        "source_kind",
    ]
    print(out_df[cols].to_string(index=False))

    out_path = DEBUG_DIR / f"{basename}_cascade_backward.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nFull dump saved to: {out_path}")

    # Context rows for the user's request: all M15.confluence sids' identity
    # data (no cascade columns). Helps cross-reference the backward-rect
    # rows against the full sid lineage.
    sids_path = DEBUG_DIR / f"{basename}_M15_confluence_sids.csv"
    if sids_path.exists():
        sids_df = pd.read_csv(sids_path)
        sids_df["meta_dict"] = sids_df["meta"].apply(_parse_meta)
        sids_df["use_case"] = sids_df["meta_dict"].apply(lambda d: d.get("use_case"))
        sids_df["trigger_event_idx"] = sids_df["meta_dict"].apply(
            lambda d: d.get("trigger_event_idx"))
        print()
        print("=== M15.confluence sids inventory (context) ===")
        print(sids_df[[
            "sid", "starting_sd", "creation_event_idx", "end_event_idx",
            "parent_sid", "parent_cycle_id", "use_case", "trigger_event_idx",
        ]].rename(columns={
            "sid": "entity_sid",
            "starting_sd": "struct_direction",
            "creation_event_idx": "entity_sid_creation_idx",
        }).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        basename = sys.argv[1]
    else:
        basename = "NZD_USD_H1_2025-11-15_2026-01-07_sd-1_eps0p0001_rk2-5"
    main(basename)
