"""PLAN_H landing B — mutants of the switch (run: `review_scripts/mutant_runner.py`, full suite, from the repo root).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_h_inputs/mutants_plan_h_b.py --out <short dir>

Each mutant reverts / bends ONE landing-B rule or call site (PLAN_H §8.5 / §8.6 "Mutation tests"). `old` is matched
against the file read in TEXT mode (LF line ends).
"""
PZ = "engine_v2/zones/poi_zones.py"
ORCH = "engine_v2/pipeline/orchestrator.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
ZR = "engine_v2/charting/_zone_render.py"
H1C = "engine_v2/charting/export_plotly.py"
M15C = "engine_v2/charting/export_m15_chart.py"
EXP = "engine_v2/debug/export_zones.py"

MUT = [
    # --- the activation sweep -------------------------------------------------------------------------------------
    ("B01 R1 floor dropped", PZ,
     "    first_active = max(cts_established_idx, ic_idx, int(pattern_confirmed_idx))",
     "    first_active = max(cts_established_idx, ic_idx)"),
    ("B02 cond5 always true", PZ, "        cond5 = touching\n", "        cond5 = True\n"),
    ("B03 touch exclusive edges", PZ,
     "        return ic_touches_zone(ic_high, ic_low, zone_lo, zone_hi)\n",
     "        return ic_high > zone_lo and ic_low < zone_hi\n"),
    ("B04 the old reason", PZ, '                reasons.append("ic_not_touching_fib_zone")',
     '                reasons.append("variant_below_v30")'),
    ("B05 no re-check at CTS transitions", PZ,
     "        if cts_changed:\n            touching = _touches()\n",
     "        if cts_changed:\n            pass\n"),
    ("B06 pre-window touch off", PZ,
     "            in_window_cts.append(ev)\n    touching = _touches()\n",
     "            in_window_cts.append(ev)\n    touching = False\n"),
    ("B07 the as-of zone mirrored", PZ,
     "        zone_lo, zone_hi = fib_zone_bounds(anchor_high, anchor_low, sd, config)\n",
     "        zone_lo, zone_hi = fib_zone_bounds(anchor_high, anchor_low, -sd, config)\n"),
    # --- derive_poi_zones / select_ip_records ----------------------------------------------------------------------
    ("B08 call site: R1 floor = the IC", PZ,
     "            pattern_confirmed_idx=int(pick.pattern_confirmed_idx),\n",
     "            pattern_confirmed_idx=ic_idx,\n"),
    ("B09 meta ip_rank lost", PZ, '                "ip_rank": pick.ip_rank,\n', '                "ip_rank": None,\n'),
    ("B10 meta tiebreak never copied", PZ,
     "        tb = rec.selection.tiebreak or (None, None, None, None)\n",
     "        tb = (None, None, None, None)\n"),
    ("B11 meta pattern_confirmed_idx = the end", PZ,
     '                "pattern_confirmed_idx": int(pick.pattern_confirmed_idx),\n',
     '                "pattern_confirmed_idx": int(pick.pattern_end_idx),\n'),
    ("B12 meta key order (a tiebreak key after status)", PZ,
     '                "ip_tb_imbalance_first_c2_idx": tb[3],\n                "status": status,\n',
     '                "status": status,\n                "ip_tb_imbalance_first_c2_idx": tb[3],\n'),
    ("B13 records: classify_misses off", PZ,
     "            selection=select_ip_ic(df, fib_state, config, classify_misses=True),\n",
     "            selection=select_ip_ic(df, fib_state, config),\n"),
    ("B14 records: the locked-without-EST assert gone", PZ,
     '            assert not fib_state.locked, (\n                f"[poi_zones] locked fib {key} has no CTS_ESTABLISHED (event contract)")\n            continue\n        out.append(IPRecordSelection(',
     '            continue\n        out.append(IPRecordSelection('),
    ("B15 records: the cycle term = the CTS anchor (Plan D revert)", PZ,
     '            cts_established_idx=int(cts_event.meta["confirmed_at"]),\n',
     '            cts_established_idx=int(ef.cts_anchor_idx(cts_event)),\n'),
    ("B16 records: no lifecycle gate", PZ,
     "        if not _poi_record_gate(fib_state):\n            continue\n        key = (int(fib_state.structure_id), int(fib_state.cycle_id))\n",
     "        key = (int(fib_state.structure_id), int(fib_state.cycle_id))\n"),
    # --- the resolver ------------------------------------------------------------------------------------------------
    ("B17 resolver: IP errors swallowed (the PR1 revert)", PZ,
     "    except Exception as exc:\n        raise IPSelectionError(",
     "    except Exception as exc:\n        return []\n        raise IPSelectionError("),
    ("B18 resolver: re-raise the original", PZ,
     "    except Exception as exc:\n        raise IPSelectionError(",
     "    except Exception as exc:\n        raise\n        raise IPSelectionError("),
    ("B19 resolver: sell inner = the high", PZ,
     '    return [float(df.loc[ic_idx, "h"]) if sd == 1 else float(df.loc[ic_idx, "l"])]',
     '    return [float(df.loc[ic_idx, "h"])]'),
    ("B20 resolver: the old variants", PZ,
     "    if sel.pick is None:\n        return []\n    ic_idx = int(sel.pick.ic_idx)\n",
     "    cands = find_ic_candidates(df, fib_state, config)\n    picks = select_ic_variants(cands, df, fib_state, config) if cands else {}\n    if not picks:\n        return []\n    ic_idx = max(picks)\n"),
    # --- the orchestrator / mirror / geometry scope ----------------------------------------------------------------
    ("B21 orchestrator: POIs re-select (not the shared records)", ORCH,
     "        cap_reason=cap_reason,\n        ip_records=ip_records,\n    )",
     "        cap_reason=cap_reason,\n    )"),
    ("B22 mirror: pattern_confirmed_idx not shifted", EDM, '    "pattern_confirmed_idx",  # POI',
     '    # "pattern_confirmed_idx",  # POI'),
    ("B23 mirror: runner-up not shifted", EDM, '    "ip_tb_runner_up_ic_idx",       # POI',
     '    # "ip_tb_runner_up_ic_idx",       # POI'),
    ("B24 geometry scope widened", EDM,
     "    except (ValueError, IndexError):\n        # The documented geometry failure",
     "    except (ValueError, IndexError, RuntimeError):\n        # The documented geometry failure"),
    # --- hovers / export -------------------------------------------------------------------------------------------
    ("B25 hover: one-candle legs as ranges", ZR, '        return f"{lo}" if lo == hi else f"{lo}–{hi}"',
     '        return f"{lo}–{hi}"'),
    ("B26 hover: TB2 line dropped", ZR, '    elif tb == "TB2":\n        text += (',
     '    elif tb == "TB2":\n        text += "" and ('),
    ("B27 hover: no consistency assert", ZR,
     "    assert (opp_first - anchor, ic - opp_first + 1, end - ic) == (n_init, n_opp, n_cont), (",
     "    assert True or (opp_first - anchor, ic - opp_first + 1, end - ic) == (n_init, n_opp, n_cont), ("),
    ("B28 H1 chart site: no IP text", H1C, "            ip_str = poi_ip_hover_text(z.meta, z.ic_idx)\n",
     '            ip_str = ""\n'),
    # B29 re-anchored 2026-10-06c (PLAN_H §8.5c): F1 replaced the H1 fib hover's inline `ip_candidates` loop by
    # `_zone_render.fib_hover_status_text` — the same bend (a picked record reads "IP: none") at its new site.
    ("B29 fib hover: IP none on picked records", ZR,
     "    elif ip_row.ic_idx is None:\n",
     "    elif ip_row.ic_idx is not None:\n"),
    ("B30 M15 overlay site: no IP text", M15C,
     "            ip_str = poi_ip_hover_text(poi.meta, poi.ic_idx)   # PLAN_H HV\n            poi_cd = [[",
     '            ip_str = ""\n            poi_cd = [['),
    ("B31 M15 sub-POI site: no IP text", M15C,
     "                ip_str = poi_ip_hover_text(poi.meta, poi.ic_idx)   # PLAN_H HV\n",
     '                ip_str = ""\n'),
    ("B32 export: ip_rank column from the old key", EXP, '            "ip_rank": m["ip_rank"],',
     '            "ip_rank": m.get("versions"),'),
]
